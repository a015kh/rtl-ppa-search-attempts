from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .dcpt import (
    DEFAULT_CELL_LIBRARY,
    DEFAULT_FLOW_ROOT,
    DCPTError,
    DCPTFlow,
    DCPTJobRunner,
    load_benchmark_manifest,
    load_problem_point,
)
from .evaluation import SynthesisEvaluator, VerilogEvaluator, VerilogEvaluatorICRTL


class DCPTEvaluator(SynthesisEvaluator):
    """Evaluate candidates with the site Design Compiler/PrimeTime flow."""

    power_unit = "mW"

    def __init__(
        self,
        benchmark_path: str | Path,
        flow_root: str | Path = DEFAULT_FLOW_ROOT,
        config_path: str | Path | None = None,
        cell_library: str | Path = DEFAULT_CELL_LIBRARY,
        clocks: list[float] | None = None,
        constraint_tolerance_ns: float = 1e-6,
    ) -> None:
        self.benchmark_path = Path(benchmark_path).expanduser().resolve()
        manifest = load_benchmark_manifest(self.benchmark_path)
        self.clk_period = float(manifest["optimization_clock_ns"])
        requested_clocks = clocks or [self.clk_period]
        self.clocks = list(dict.fromkeys(float(clock) for clock in requested_clocks))
        if self.clk_period not in self.clocks:
            self.clocks.insert(0, self.clk_period)
        if any(clock <= 0 for clock in self.clocks):
            raise DCPTError("DC/PT clock constraints must be positive.")
        self.constraint_tolerance_ns = float(constraint_tolerance_ns)
        self.flow = DCPTFlow.load(flow_root)
        self.config_path = config_path
        if cell_library is None:
            raise DCPTError("Provide an explicit gate-level cell library (--dcpt_cell_library / --cell-library).")
        self.cell_library = Path(cell_library).expanduser().resolve()
        if not self.cell_library.is_file():
            raise DCPTError(
                f"DC gate-level cell library not found: {self.cell_library}"
            )

    def _runner(self, output_directory: str) -> DCPTJobRunner:
        generation_dir = Path(output_directory).resolve()
        return DCPTJobRunner(
            self.flow, generation_dir.parent / ".dcpt_cache", self.config_path
        )

    def _annotate_timing(
        self, metrics: dict[str, Any], clock_ns: float
    ) -> dict[str, Any]:
        timing_slack_ns = clock_ns - float(metrics["eff_clk_period"])
        return {
            **metrics,
            "clock_ns": clock_ns,
            "timing_slack_ns": timing_slack_ns,
            "constraint_met": timing_slack_ns >= -self.constraint_tolerance_ns,
        }

    def _check_dc_functionality(
        self,
        netlist: Path,
        test_sv_file: str,
        ref_sv_file: str | None,
        verilog_evaluator: VerilogEvaluator,
    ) -> tuple[bool, str]:
        if isinstance(verilog_evaluator, VerilogEvaluatorICRTL):
            gate_test = Path(test_sv_file)
        else:
            gate_test = netlist.with_suffix(".gate_test.sv")
            test_source = Path(test_sv_file).read_text(encoding="utf-8")
            test_source = re.sub(
                r"\s*\$dump(?:file|vars)\s*\([^;]*;", "", test_source
            )
            gate_test.write_text(test_source, encoding="utf-8")
        result = verilog_evaluator.evaluate(
            generated_sv_file=[str(netlist), str(self.cell_library)],
            test_sv_file=str(gate_test),
            ref_sv_file=ref_sv_file,
            top_module_name=(
                "test"
                if isinstance(verilog_evaluator, VerilogEvaluatorICRTL)
                else "tb"
            ),
        )
        output = str(result.get("simulation_stdout", ""))
        mismatch = re.search(r"^Mismatches:\s*(\d+)", output, re.M)
        passed = result.get("status") == "success" and (
            (mismatch is not None and int(mismatch.group(1)) == 0)
            or "===========Your Design Passed===========" in output
            or "All tests PASS!" in output
        )
        return passed, (
            f"Compilation Log:\n{result.get('compilation_stderr', '')}\n\n"
            f"Simulation Log:\n{output}\n{result.get('simulation_stderr', '')}"
        )

    def evaluate(
        self,
        verilog_file: str,
        problem_name: str,
        synth_top_module_name: str,
        output_directory: str,
        report_base_path: str,
        verilog_evaluator: VerilogEvaluator,
        test_sv_file: str,
        ref_sv_file: str | None,
    ) -> dict[str, Any]:
        Path(output_directory).mkdir(parents=True, exist_ok=True)
        rtl = Path(verilog_file).read_text(encoding="utf-8")
        runner = self._runner(output_directory)
        clock_results: list[dict[str, Any]] = []
        primary_metrics: dict[str, Any] | None = None
        primary_directory: Path | None = None

        for clock_ns in self.clocks:
            try:
                point = load_problem_point(self.benchmark_path, problem_name, clock_ns)
                _, directory = runner.run(
                    rtl,
                    str(point["sdc"]),
                    problem_name,
                    synth_top_module_name,
                    False,
                )
            except (DCPTError, OSError, ValueError) as exc:
                if clock_ns == self.clk_period:
                    return self._failure(False, False, str(exc))
                clock_results.append(
                    {"clock_ns": clock_ns, "success": False, "error": str(exc)}
                )
                continue

            clock_label = f"{clock_ns:g}".replace(".", "p")
            if clock_ns == self.clk_period:
                netlist = Path(verilog_file).with_suffix(".syn.v")
                primary_directory = directory
            else:
                netlist = Path(report_base_path + f".clock_{clock_label}.syn.v")
            runner.copy_netlist(directory, synth_top_module_name, netlist)
            functional, functional_log = self._check_dc_functionality(
                netlist, test_sv_file, ref_sv_file, verilog_evaluator
            )
            if not functional:
                if clock_ns == self.clk_period:
                    return self._failure(
                        True,
                        False,
                        f"{directory / 'dc' / 'design_dc.log'}\n\n{functional_log}",
                    )
                clock_results.append(
                    {
                        "clock_ns": clock_ns,
                        "success": False,
                        "error": functional_log,
                    }
                )
                continue

            try:
                runner.run(
                    rtl,
                    str(point["sdc"]),
                    problem_name,
                    synth_top_module_name,
                    True,
                )
                metrics = runner.metrics(directory, synth_top_module_name)
            except (DCPTError, OSError, ValueError) as exc:
                if clock_ns == self.clk_period:
                    return self._failure(True, True, str(exc))
                clock_results.append(
                    {"clock_ns": clock_ns, "success": False, "error": str(exc)}
                )
                continue

            point_metrics = self._annotate_timing(metrics, clock_ns)
            constraint_met = point_metrics["constraint_met"]
            clock_results.append(
                {
                    "clock_ns": clock_ns,
                    "success": True,
                    "constraint_met": constraint_met,
                    "metrics": point_metrics,
                    "work_directory": str(directory),
                }
            )
            if clock_ns == self.clk_period:
                primary_metrics = point_metrics

        if primary_metrics is None or primary_directory is None:
            return self._failure(
                False, False, "Primary DC/PT clock point was not evaluated."
            )

        metrics = {**primary_metrics, "clock_results": clock_results}

        ppa_path = Path(report_base_path + "_synthesis_report.ppa")
        keys = ("tns", "wns", "eff_clk_period", "power", "area")
        ppa_path.write_text(
            ",".join(keys) + "\n" + ",".join(str(metrics[key]) for key in keys) + "\n",
            encoding="utf-8",
        )
        metrics["report_path"] = str(ppa_path)
        Path(report_base_path + ".dcpt.json").write_text(
            json.dumps(
                {
                    "clock_ns": self.clk_period,
                    "metrics": primary_metrics,
                    "clock_results": clock_results,
                    "power_unit": self.power_unit,
                    "work_directory": str(primary_directory),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "synthesis_success": True,
            "synthesis_functionality_success": True,
            "ppa_success": True,
            "synthesis_log": str(primary_directory / "pt" / "design_pt.log"),
            "ppa_metrics": metrics,
        }

    @staticmethod
    def _failure(
        synthesis_success: bool, functionality_success: bool, log: str
    ) -> dict[str, Any]:
        return {
            "synthesis_success": synthesis_success,
            "synthesis_functionality_success": functionality_success,
            "ppa_success": False,
            "synthesis_log": log,
            "ppa_metrics": None,
        }
