from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .cvdp import CVDP_CLOCKS_NS, CVDPError, NativeCocotbRunner, project_sources
from .dcpt import DEFAULT_CELL_LIBRARY, DEFAULT_FLOW_ROOT, DCPTError, DCPTFlow, DCPTJobRunner


def _strip_comments(text: str) -> str:
    return re.sub(r"//[^\n]*|/\*.*?\*/", "", text, flags=re.S)


def module_ports(rtl: str, top: str) -> dict[str, list[str]]:
    clean = _strip_comments(rtl)
    match = re.search(
        rf"\bmodule\s+{re.escape(top)}\b(?:\s*#\s*\(.*?\))?\s*\((.*?)\)\s*;",
        clean,
        re.S,
    )
    if match is None:
        raise CVDPError(f"Cannot find ANSI module header for top {top}.")
    ports: dict[str, list[str]] = {"input": [], "output": [], "inout": []}
    direction: str | None = None
    for raw in match.group(1).split(","):
        part = raw.strip()
        found = re.search(r"\b(input|output|inout)\b", part)
        if found:
            direction = found.group(1)
            part = part[found.end() :]
        if direction is None:
            continue
        part = re.sub(r"\[[^]]+\]", " ", part)
        identifiers = re.findall(r"[A-Za-z_][A-Za-z0-9_$]*", part)
        identifiers = [
            item
            for item in identifiers
            if item not in {"wire", "reg", "logic", "signed", "unsigned", "var"}
        ]
        if identifiers:
            ports[direction].append(identifiers[-1])
    return ports


def module_parameter_clause(rtl: str, top: str) -> str | None:
    """Return the top module's raw parameter clause, including `#(...)`."""
    clean = _strip_comments(rtl)
    match = re.search(rf"\bmodule\s+{re.escape(top)}\b\s*(#\s*\()", clean)
    if match is None:
        return None
    start = match.start(1)
    opening = clean.find("(", start)
    depth = 0
    for index in range(opening, len(clean)):
        if clean[index] == "(":
            depth += 1
        elif clean[index] == ")":
            depth -= 1
            if depth == 0:
                return clean[start : index + 1]
    raise CVDPError(f"Unterminated parameter clause for top {top}.")


def add_netlist_parameters(netlist: str, top: str, clause: str | None) -> str:
    if not clause:
        return netlist
    pattern = rf"(\bmodule\s+{re.escape(top)}\b)\s*(\()"
    rewritten, count = re.subn(pattern, rf"\1 {clause} \2", netlist, count=1)
    if count != 1:
        raise CVDPError(f"Cannot add parameters to synthesized top {top}.")
    return rewritten


def infer_clocks(harness_text: str, input_ports: Iterable[str]) -> list[str]:
    inputs = list(input_ports)
    explicit = set(
        re.findall(r"\bClock\s*\(\s*dut\.([A-Za-z_][A-Za-z0-9_$]*)", harness_text)
    )
    common = {
        name
        for name in inputs
        if re.search(r"(?:^|_)(?:clk|clock)(?:$|_)", name, flags=re.I)
    }
    return sorted((explicit | common) & set(inputs))


def render_sdc(
    period_ns: float,
    clocks: list[str],
    input_ports: list[str],
    output_ports: list[str],
) -> str:
    period = f"{period_ns:.6f}"
    delay = min(1.0, 0.2 * period_ns)
    lines = ["# Generated CVDP multi-clock constraints"]
    if clocks:
        for index, clock in enumerate(clocks):
            lines.append(
                f"create_clock -name cvdp_clk_{index} -period {period} [get_ports {{{clock}}}]"
            )
            lines.append(f"set_ideal_network [get_ports {{{clock}}}]")
        constraint_clock = "cvdp_clk_0"
    else:
        lines.append(f"create_clock -name cvdp_vclk -period {period}")
        constraint_clock = "cvdp_vclk"
    data_inputs = [name for name in input_ports if name not in clocks]
    if data_inputs:
        lines.append(
            f"set_input_delay {delay:.6f} -clock {constraint_clock} "
            f"[get_ports {{{' '.join(data_inputs)}}}]"
        )
    if output_ports:
        lines.append(
            f"set_output_delay {delay:.6f} -clock {constraint_clock} [all_outputs]"
        )
    lines.extend(
        [
            "set_dont_touch_network [all_clocks]",
            "set_fix_hold [all_clocks]",
            "set_clock_uncertainty 0.1 [all_clocks]",
            "set_clock_latency 0.5 [all_clocks]",
            "set_max_fanout 20 [current_design]",
            "set_load 1 [all_outputs]",
            "set_drive 0.1 [all_inputs]",
        ]
    )
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class CVDPDCPTResult:
    task_id: str
    source_digest: str
    rtl_passed: bool
    clocks: tuple[dict[str, Any], ...]

    @property
    def eligible(self) -> bool:
        return self.rtl_passed and any(
            item.get("success")
            and item.get("gate_passed")
            and item.get("constraint_met")
            for item in self.clocks
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "source_digest": self.source_digest,
            "rtl_passed": self.rtl_passed,
            "eligible": self.eligible,
            "clock_results": list(self.clocks),
        }


class CVDPProjectDCPTEvaluator:
    def __init__(
        self,
        cache_root: Path,
        flow_root: str | Path = DEFAULT_FLOW_ROOT,
        config_path: str | Path | None = None,
        cell_library: str | Path = DEFAULT_CELL_LIBRARY,
        clocks: Iterable[float] = CVDP_CLOCKS_NS,
        constraint_tolerance_ns: float = 1e-6,
        harness_timeout_seconds: int = 900,
        sandbox_harness: bool = True,
    ) -> None:
        self.flow = DCPTFlow.load(flow_root)
        self.runner = DCPTJobRunner(self.flow, cache_root.resolve(), config_path)
        if cell_library is None:
            raise DCPTError("Provide an explicit gate-level cell library (--dcpt_cell_library / --cell-library).")
        self.cell_library = Path(cell_library).expanduser().resolve()
        if not self.cell_library.is_file():
            raise CVDPError(f"Cell simulation library not found: {self.cell_library}")
        self.clocks = tuple(dict.fromkeys(float(clock) for clock in clocks))
        if not self.clocks or any(clock <= 0 for clock in self.clocks):
            raise CVDPError("CVDP DC/PT clocks must be positive.")
        self.constraint_tolerance_ns = float(constraint_tolerance_ns)
        self.harness = NativeCocotbRunner(harness_timeout_seconds, sandbox_harness)

    @staticmethod
    def _combined_rtl(sources: list[Path]) -> str:
        sections = []
        for source in sources:
            sections.append(f"// CVDP source: {source}\n{source.read_text(encoding='utf-8')}\n")
        return "\n".join(sections)

    def evaluate(
        self,
        task_id: str,
        workspace: Path,
        source_digest: str,
        *,
        run_rtl: bool = True,
    ) -> CVDPDCPTResult:
        from .cvdp import harness_environment

        env = harness_environment(workspace)
        top = env.get("TOPLEVEL", "").strip()
        if not top:
            raise CVDPError(f"Task {task_id} has no TOPLEVEL in its harness environment.")
        sources = project_sources(workspace, env)
        rtl = self._combined_rtl(sources)
        ports = module_ports(rtl, top)
        parameter_clause = module_parameter_clause(rtl, top)
        harness_text = "\n".join(
            path.read_text(encoding="utf-8", errors="replace")
            for path in sorted((workspace / "src").rglob("*.py"))
        )
        clocks = infer_clocks(harness_text, ports["input"])
        if run_rtl:
            rtl_result = self.harness.run(workspace)
            if not rtl_result.passed:
                return CVDPDCPTResult(
                    task_id,
                    source_digest,
                    False,
                    (
                        {
                            "success": False,
                            "stage": "rtl_simulation",
                            "error": (rtl_result.stdout + "\n" + rtl_result.stderr)[-12000:],
                        },
                    ),
                )
        results: list[dict[str, Any]] = []
        netlist_dir = workspace / "rundir" / "dcpt_netlists"
        netlist_dir.mkdir(parents=True, exist_ok=True)
        simulator = env.get("SIM", "icarus").lower()
        if simulator in {"xrun", "xcelium"}:
            simulator = "vcs"
        for period in self.clocks:
            sdc = render_sdc(period, clocks, ports["input"], ports["output"])
            try:
                _, directory = self.runner.run(rtl, sdc, task_id, top, False)
                label = f"{period:g}".replace(".", "p")
                netlist = netlist_dir / f"{top}.{label}ns.v"
                self.runner.copy_netlist(directory, top, netlist)
                if parameter_clause:
                    netlist.write_text(
                        add_netlist_parameters(
                            netlist.read_text(encoding="utf-8"), top, parameter_clause
                        ),
                        encoding="utf-8",
                    )
                gate_result = self.harness.run(
                    workspace,
                    {
                        "SIM": simulator,
                        "TOPLEVEL_LANG": "verilog",
                        "VERILOG_SOURCES": f"{netlist} {self.cell_library}",
                        "CVDP_FIXED_GATE_NETLIST": "1",
                    },
                )
                if not gate_result.passed:
                    results.append(
                        {
                            "clock_ns": period,
                            "success": False,
                            "gate_passed": False,
                            "stage": "gate_simulation",
                            "error": (gate_result.stdout + "\n" + gate_result.stderr)[-12000:],
                            "work_directory": str(directory),
                        }
                    )
                    continue
                self.runner.run(rtl, sdc, task_id, top, True)
                metrics = self.runner.metrics(directory, top)
                wns = float(metrics["wns"])
                constraint_met = wns >= -self.constraint_tolerance_ns
                results.append(
                    {
                        "clock_ns": period,
                        "success": True,
                        "gate_passed": True,
                        "constraint_met": constraint_met,
                        "metrics": {**metrics, "clock_ns": period},
                        "work_directory": str(directory),
                    }
                )
            except (CVDPError, DCPTError, OSError, ValueError) as exc:
                results.append(
                    {
                        "clock_ns": period,
                        "success": False,
                        "gate_passed": False,
                        "stage": "dcpt",
                        "error": str(exc),
                    }
                )
        return CVDPDCPTResult(task_id, source_digest, True, tuple(results))


def pareto_front(results: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    feasible = [
        dict(item)
        for item in results
        if item.get("success") and item.get("gate_passed") and item.get("constraint_met")
    ]

    def objectives(item: Mapping[str, Any]) -> tuple[float, float, float]:
        metrics = item["metrics"]
        return (
            float(item["clock_ns"]),
            float(metrics["area"]),
            float(metrics["power"]),
        )

    front = []
    for candidate in feasible:
        c = objectives(candidate)
        dominated = False
        for other in feasible:
            if other is candidate:
                continue
            o = objectives(other)
            if all(left <= right for left, right in zip(o, c)) and any(
                left < right for left, right in zip(o, c)
            ):
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return sorted(front, key=objectives)


def save_result(path: Path, result: CVDPDCPTResult) -> None:
    data = result.to_dict()
    data["pareto_front"] = pareto_front(data["clock_results"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
