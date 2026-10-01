from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any


DEFAULT_FLOW_ROOT = None
DEFAULT_CELL_LIBRARY = None


class DCPTError(RuntimeError):
    """Commercial-flow configuration or execution failure."""


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise DCPTError(f"Cannot load commercial-flow module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class DCPTFlow:
    root: Path
    core: ModuleType
    spec2rtl: ModuleType

    @classmethod
    def load(cls, root: str | Path | None = DEFAULT_FLOW_ROOT) -> "DCPTFlow":
        if root is None:
            raise DCPTError("Provide an explicit synthesis flow root (--dcpt_flow_root / --dcpt-flow-root).")
        resolved = Path(root).expanduser().resolve()
        core_path = resolved / "resynthesize_dataset.py"
        spec_path = resolved / "synthesize_verilogeval_spec2rtl.py"
        if not core_path.is_file() or not spec_path.is_file():
            raise DCPTError(
                f"Expected resynthesize_dataset.py and synthesize_verilogeval_spec2rtl.py under {resolved}"
            )
        if str(resolved) not in sys.path:
            sys.path.insert(0, str(resolved))
        core = _load_module("revolution_syn_tool_core", core_path)
        # The VerilogEval helper imports resynthesize_dataset by its original name.
        sys.modules.setdefault("resynthesize_dataset", core)
        spec2rtl = _load_module("revolution_syn_tool_spec2rtl", spec_path)
        required = (
            "default_config",
            "job_identity",
            "job_dir",
            "parse_dc_result",
            "parse_pt_power",
            "preflight",
            "required_dc_outputs",
            "required_pt_outputs",
            "run_one_job",
            "site_tool_environment",
        )
        missing = [name for name in required if not hasattr(core, name)]
        if missing or not hasattr(spec2rtl, "render_sdc"):
            raise DCPTError(f"Commercial flow is missing required APIs: {missing}")
        return cls(resolved, core, spec2rtl)

    def config(self, config_path: str | Path | None = None) -> Any:
        if config_path is None:
            return self.core.default_config()
        return self.core.ToolConfig.from_json(Path(config_path).expanduser().resolve())

    def preflight(
        self, work_root: Path, workers: int, config_path: str | Path | None = None
    ) -> dict[str, Any]:
        config = self.config(config_path)
        if workers < 1 or workers > config.max_workers:
            raise DCPTError(
                f"DC/PT workers must be between 1 and {config.max_workers}; got {workers}"
            )
        return self.core.preflight(config, work_root, workers)


def load_benchmark_manifest(benchmark_path: Path) -> dict[str, Any]:
    path = benchmark_path / "benchmark.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DCPTError(f"Missing or invalid DC/PT benchmark manifest: {path}") from exc
    if data.get("schema_version") != 1:
        raise DCPTError(f"Unsupported DC/PT benchmark schema in {path}")
    return data


def load_problem_point(
    benchmark_path: Path, problem: str, clock_ns: float
) -> dict[str, Any]:
    path = benchmark_path / f"{problem}_dcpt.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DCPTError(f"Missing or invalid DC/PT sidecar: {path}") from exc
    for point in data.get("points", []):
        if float(point.get("clock_ns", -1)) == float(clock_ns):
            if not point.get("sdc"):
                raise DCPTError(f"Clock point {clock_ns:g} ns has no SDC in {path}")
            return point
    raise DCPTError(f"No {clock_ns:g} ns point in {path}")


class DCPTJobRunner:
    def __init__(
        self,
        flow: DCPTFlow,
        work_root: Path,
        config_path: str | Path | None = None,
    ) -> None:
        self.flow = flow
        self.work_root = work_root
        self.config = flow.config(config_path)
        self._tool_env: dict[str, str] | None = None

    def _environment(self) -> dict[str, str]:
        if self._tool_env is None:
            self._tool_env = self.flow.core.site_tool_environment()
        return self._tool_env

    def run(
        self,
        rtl: str,
        sdc: str,
        design_name: str,
        top_module: str,
        run_pt: bool = True,
    ) -> tuple[dict[str, Any], Path]:
        core = self.flow.core
        job_id = core.job_identity(rtl, sdc, top_module)
        row = {
            "job_id": job_id,
            "source_kind": "dataset",
            "design_name": design_name,
            "top_module": top_module,
            "rtl": rtl,
            "sdc": sdc,
            "support_paths": "[]",
        }
        directory = core.job_dir(self.work_root, job_id)
        dc_outputs = core.required_dc_outputs(directory, top_module)
        if not all(Path(path).is_file() for path in dc_outputs):
            result = core.run_one_job(
                row, "dc", self.work_root, self.config, self._environment()
            )
            if not result["ok"]:
                raise DCPTError(f"Design Compiler failed: {result['error']}")
        if run_pt:
            pt_outputs = core.required_pt_outputs(directory, top_module)
            if not all(Path(path).is_file() for path in pt_outputs):
                result = core.run_one_job(
                    row, "pt", self.work_root, self.config, self._environment()
                )
                if not result["ok"]:
                    raise DCPTError(f"PrimeTime failed: {result['error']}")
        return row, directory

    def metrics(self, directory: Path, top_module: str) -> dict[str, Any]:
        core = self.flow.core
        dc = core.parse_dc_result(directory, top_module)
        pt = core.parse_pt_power(
            directory / "pt" / "report" / f"{top_module}.report_power.rpt"
        )
        return {
            "tns": dc["tns"],
            "wns": dc["wns"],
            "eff_clk_period": dc["delay_value"],
            "power": pt["total_power_value"],
            "area": dc["area_value"],
        }

    def copy_netlist(self, directory: Path, top_module: str, destination: Path) -> None:
        source = directory / "dc" / "report" / f"{top_module}_syn.v"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
