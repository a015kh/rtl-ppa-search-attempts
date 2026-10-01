#!/usr/bin/env python3
"""Resumable Idea-Guided Refinement for the ICRTL benchmark suite."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import os
import re
try:
    import resource
except ImportError:
    resource = None
import shlex
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence


from . import settings
EVOLVE_DIR = Path(__file__).resolve().parent
REPO_ROOT = settings.path('benchmark_root')
SITE_TOOL_PROFILE = settings.path('tool_profile')
DB_DIR = settings.path('db_dir')
LIB_DIR = settings.path('lib_dir')
CELL_VERILOG = settings.path('cell_verilog')
SYN_LIB_DIR = settings.path('synthesis_library_dir')
DC_BIN = settings.tool('dc_shell')
PT_BIN = settings.tool('pt_shell')
PENALTY = -100000.0
SCORE_SCALE = 100000.0
# PrimeTime/SDF values can differ from the DC constraint by a few picoseconds
# because delays are serialized with finite precision. Treat only this numerical
# boundary as clean; substantive synthesis timing violations remain strict.
PT_TIMING_EPSILON_NS = 0.005


@dataclasses.dataclass(frozen=True)
class Task:
    name: str
    spec: Path
    baseline: Path
    testbench: Path
    vcd_strip_path: str
    top: str = "TOP"
    io_delay_fraction: float = 0.5
    native: Any | None = None


TASK_NAMES = (
    "Q1_LBP",
    "Q2_CONV",
    "Q3_HC",
    "Q4_JAM",
    "Q5_DT",
    "Q6_REF",
    "Q7_GEO",
    "Q8_IOT",
    "Q9_SECC",
    "Q10_GEMM",
)

TASKS = {
    name: Task(
        name=name,
        spec=REPO_ROOT / name / "SPEC.md",
        baseline=REPO_ROOT / "VCS" / name / "initial.sv",
        testbench=REPO_ROOT / "VCS" / name / "00_TB",
        vcd_strip_path=(
            "test/TOP_Control/TOP" if name == "Q10_GEMM" else "test/TOP"
        ),
        # Q8 has a legal same-cycle input-to-output datapath. Applying 0.5T
        # at both boundaries leaves zero delay for that path, so use a
        # conventional 10% external budget on each side for this task.
        io_delay_fraction=0.1 if name == "Q8_IOT" else 0.5,
    )
    for name in TASK_NAMES
}


@dataclasses.dataclass
class EvalResult:
    candidate_sha256: str
    period_ns: float
    mode: str
    rtl_pass: bool = False
    gate_pass: bool | None = None
    timing_clean: bool = False
    simulation_time_ps: int | None = None
    latency_ns: float | None = None
    area: float | None = None
    slack_ns: float | None = None
    power_mw: float | None = None
    at: float | None = None
    ppa: float | None = None
    score: float = PENALTY
    failure_stage: str | None = None
    feedback: str = ""
    runtimes_s: dict[str, float] = dataclasses.field(default_factory=dict)

    @property
    def valid_proxy(self) -> bool:
        return (
            self.rtl_pass
            and self.timing_clean
            and self.area is not None
            and self.latency_ns is not None
        )

    @property
    def valid_full(self) -> bool:
        return (
            self.valid_proxy
            and self.gate_pass is True
            and self.power_mw is not None
            and self.ppa is not None
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalResult":
        return cls(**data)


class ExperimentError(RuntimeError):
    pass


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def period_tag(period: float) -> str:
    return f"{period:g}".replace(".", "p")


def extract_last(pattern: str, text: str, flags: int = 0) -> str | None:
    matches = re.findall(pattern, text, flags)
    if not matches:
        return None
    match = matches[-1]
    return match[0] if isinstance(match, tuple) else match


def parse_simulation(text: str) -> tuple[bool, int | None]:
    passed = re.search(r"\bAll tests PASS!?\b", text, re.IGNORECASE) is not None
    raw_time = extract_last(r"Time:\s*(\d+)\s*ps", text)
    return passed, int(raw_time) if raw_time else None


def parse_area(text: str) -> float | None:
    raw = extract_last(r"Total cell area:\s*([0-9.eE+-]+)", text)
    return float(raw) if raw else None


def parse_slack(text: str) -> tuple[float | None, bool]:
    matches = re.findall(
        r"slack\s+\((MET|VIOLATED)\)\s+(-?[0-9.eE+]+)", text, re.IGNORECASE
    )
    if not matches:
        return None, False
    status, value = matches[0]
    slack = float(value)
    if status.upper() == "VIOLATED" and slack > 0:
        slack = -slack
    return slack, status.upper() == "MET" and slack >= 0


def parse_power_mw(text: str) -> float | None:
    raw = extract_last(r"Total Power\s*=\s*([0-9.eE+-]+)", text)
    return float(raw) * 1000.0 if raw else None


def compact_log(text: str, limit: int = 80) -> str:
    interesting = [
        line
        for line in text.splitlines()
        if "matplotlib" not in line
        if re.search(
            r"error|fatal|fail|violat|unresolved|cannot|syntax|warning|slack|area",
            line,
            re.IGNORECASE,
        )
    ]
    source = interesting if interesting else text.splitlines()
    return "\n".join(source[-limit:])


def run_logged(
    command: Sequence[str],
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    timeout: int,
    memory_limit_gb: int | None = None,
) -> tuple[int, str, float]:
    start = time.monotonic()
    preexec_fn = None
    if memory_limit_gb is not None:
        limit_bytes = memory_limit_gb * 1024**3

        def limit_address_space() -> None:
            if resource is None:
                raise ExperimentError("EDA execution requires a POSIX workstation")
            resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))

        preexec_fn = limit_address_space
    try:
        completed = subprocess.run(
            list(command),
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            timeout=timeout,
            check=False,
            preexec_fn=preexec_fn,
        )
        returncode = completed.returncode
        output = completed.stdout
    except subprocess.TimeoutExpired as exc:
        returncode = 124
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr
        output = (stdout or "") + (stderr or "") + f"\nTIMEOUT after {timeout}s\n"
    elapsed = time.monotonic() - start
    atomic_write_text(log_path, output)
    return returncode, output, elapsed


def load_eda_environment() -> dict[str, str]:
    if not SITE_TOOL_PROFILE.is_file():
        raise ExperimentError(f"Missing site tool profile: {SITE_TOOL_PROFILE}")
    shell = (
        f"source {shlex.quote(str(SITE_TOOL_PROFILE))} >/dev/null && "
        f"{shlex.join(settings.TOOL_SELECTION)} >/dev/null && env -0"
    )
    completed = subprocess.run(
        ["/bin/bash", "-lc", shell],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )
    if completed.returncode:
        raise ExperimentError(
            "Unable to load `tool 1 4 6`: "
            + completed.stderr.decode(errors="replace").strip()
        )
    result: dict[str, str] = {}
    for item in completed.stdout.split(b"\0"):
        if item and b"=" in item:
            key, value = item.split(b"=", 1)
            result[key.decode()] = value.decode(errors="surrogateescape")
    if not (result.get("LM_LICENSE_FILE") or result.get("SNPSLMD_LICENSE_FILE")):
        raise ExperimentError("The site tool setup did not export a Synopsys license.")
    return result


def preflight(env: dict[str, str], model: str, effort: str) -> dict[str, Any]:
    required = [
        SITE_TOOL_PROFILE,
        DC_BIN,
        PT_BIN,
        DB_DIR / "typical.db",
        DB_DIR / "slow.db",
        DB_DIR / "fast.db",
        SYN_LIB_DIR / "dw_foundation.sldb",
        CELL_VERILOG,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise ExperimentError("Missing required EDA files:\n" + "\n".join(missing))

    checks: dict[str, Any] = {"model": model, "reasoning_effort": effort}
    commands = {
        "codex": ["codex", "--version"],
        "codex_login": ["codex", "login", "status"],
        "vcs": ["vcs", "-ID"],
        "dc": [str(DC_BIN), "-version"],
        "pt": [str(PT_BIN), "-version"],
    }
    for name, command in commands.items():
        completed = subprocess.run(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            timeout=60,
            check=False,
        )
        output = completed.stdout.strip()
        checks[name] = {"returncode": completed.returncode, "output": output[:2000]}
        if name in {"codex", "codex_login"} and completed.returncode:
            raise ExperimentError(f"{name} preflight failed:\n{output}")
        if name == "codex_login" and "Logged in" not in output:
            raise ExperimentError("Codex CLI is not logged in.")

    catalog = subprocess.run(
        ["codex", "debug", "models", "--bundled"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        timeout=60,
        check=False,
    )
    if catalog.returncode:
        raise ExperimentError("Cannot inspect the installed Codex model catalog.")
    try:
        payload = json.loads(catalog.stdout)
    except json.JSONDecodeError as exc:
        raise ExperimentError(f"Invalid Codex model catalog: {exc}") from exc
    record = next((item for item in payload.get("models", []) if item.get("slug") == model), None)
    if record is None:
        raise ExperimentError(f"Codex model {model!r} is unavailable.")
    supported = {item["effort"] for item in record.get("supported_reasoning_levels", [])}
    if effort not in supported:
        raise ExperimentError(
            f"Reasoning effort {effort!r} is unavailable for {model}; "
            f"supported: {', '.join(sorted(supported))}"
        )
    checks["model_catalog"] = {
        "slug": record["slug"],
        "supported_reasoning_levels": sorted(supported),
    }
    return checks


def render_sdc(task: Task, period: float) -> str:
    return f"""set cycle {period:g}
create_clock -period $cycle -name clk [get_ports clk]
set_dont_touch_network [get_clocks clk]
set_fix_hold [get_clocks clk]
set_ideal_network [get_ports clk]
set_clock_uncertainty -hold 0.005 [get_clocks clk]
set_clock_uncertainty -setup 0.1 [get_clocks clk]
set_clock_latency 0.1 [get_clocks clk]
set_false_path -from [get_ports rst]
set_drive 1 [all_inputs]
set_load 0.05 [all_outputs]
set_input_delay [expr $cycle * {task.io_delay_fraction:g}] -clock clk [remove_from_collection [all_inputs] [get_ports clk]]
set_output_delay [expr $cycle * {task.io_delay_fraction:g}] -clock clk [all_outputs]
"""


def render_dc_tcl() -> str:
    return f"""file mkdir Netlist
file mkdir Report
define_design_lib work -path ./work
set DESIGN TOP
set search_path ". {DB_DIR} {SYN_LIB_DIR} $search_path"
set target_library [list "typical.db" "slow.db" "fast.db"]
set link_library "* $target_library dw_foundation.sldb"
set symbol_library [list "generic.sdb"]
set synthetic_library "dw_foundation.sldb"
set hdlin_translate_off_skip_text TRUE
set edifout_netlist_only TRUE
set verilogout_no_tri true
set compile_fix_multiple_port_nets TRUE
analyze -format sverilog ../01_RTL/initial.sv
elaborate $DESIGN
current_design $DESIGN
link
source -echo -verbose ./syn.sdc
set_fix_hold [get_clocks clk]
check_design > Report/check_design.txt
check_timing > Report/check_timing.txt
uniquify
set_fix_multiple_port_nets -all -buffer_constants [get_designs *]
compile
current_design $DESIGN
report_timing -delay min -max_paths 10 > Report/TOP_syn.timing_min
report_timing -delay max -max_paths 10 > Report/TOP_syn.timing_max
report_area -hierarchy > Report/TOP_syn.area
set bus_inference_style {{%s[%d]}}
set bus_naming_style {{%s[%d]}}
set hdlout_internal_busses true
change_names -hierarchy -rule verilog
write -format ddc -hierarchy -output Netlist/top_syn.ddc
write -format verilog -hierarchy -output Netlist/top_syn.v
write_sdf -version 2.1 -context verilog -load_delay cell Netlist/top_syn.sdf
write_sdc Netlist/top_syn.sdc -version 1.8
exit
"""


def render_pt_tcl(task: Task) -> str:
    return f"""set search_path ". {DB_DIR} {SYN_LIB_DIR} $search_path"
set target_library [list "typical.db" "slow.db" "fast.db"]
set link_library "* $target_library dw_foundation.sldb"
set synthetic_library "dw_foundation.sldb"
set power_enable_analysis TRUE
set power_analysis_mode time_based
read_file -format verilog ../02_SYN/Netlist/top_syn.v
current_design TOP
link
read_sdf -load_delay net ../02_SYN/Netlist/top_syn.sdf
read_sdc ../02_SYN/Netlist/top_syn.sdc
update_timing -full
set worst_path [get_timing_paths -delay_type max -nworst 1]
set wns [get_attribute $worst_path slack]
set wns_file [open top.wns w]
puts $wns_file $wns
close $wns_file
report_timing -delay_type max -nworst 1 > top.timing
read_vcd -strip_path {task.vcd_strip_path} ../03_GATE/top.vcd
update_power
report_power > top.power
exit
"""


def patch_testbench_period(source: Path, destination: Path, period: float) -> None:
    text = source.read_text(encoding="utf-8", errors="replace")
    updated, count = re.subn(
        r"(`define\s+CYCLE\s+)([0-9.]+)",
        lambda match: match.group(1) + f"{period:g}",
        text,
        count=1,
        flags=re.IGNORECASE,
    )
    if count != 1:
        raise ExperimentError(f"Cannot set `CYCLE in {source}")
    atomic_write_text(destination, updated)


class CommercialEvaluator:
    def __init__(self, task: Task, env: dict[str, str], cache_root: Path):
        self.task = task
        self.env = env
        self.cache_root = cache_root

    def _work_dir(self, candidate: Path, period: float, mode: str) -> Path:
        digest = sha256_file(candidate)
        constraint_tag = (
            ""
            if self.task.io_delay_fraction == 0.5
            else f"_io{self.task.io_delay_fraction:g}"
        )
        return (
            self.cache_root
            / digest
            / f"{mode}_{period_tag(period)}ns{constraint_tag}"
        )

    def _prepare(self, candidate: Path, period: float, work: Path) -> None:
        for name in ("01_RTL", "02_SYN", "03_GATE", "04_POWER"):
            (work / name).mkdir(parents=True, exist_ok=True)
        if not (work / "00_TB").exists():
            shutil.copytree(self.task.testbench, work / "00_TB")
        shutil.copy2(candidate, work / "01_RTL/initial.sv")
        patch_testbench_period(
            self.task.testbench / "test.sv", work / "00_TB/test.sv", period
        )
        atomic_write_text(work / "02_SYN/syn.sdc", render_sdc(self.task, period))
        atomic_write_text(work / "02_SYN/syn.tcl", render_dc_tcl())
        atomic_write_text(work / "04_POWER/pt_script.tcl", render_pt_tcl(self.task))

    @staticmethod
    def _prune_proxy_artifacts(work: Path) -> None:
        """Keep reports/logs while removing large rebuildable proxy artifacts."""
        files = [
            work / "01_RTL/simv",
            work / "01_RTL/ucli.key",
            work / "01_RTL/elabMhIdDidWork.txt",
            work / "02_SYN/default.svf",
        ]
        directories = [
            work / "01_RTL/simv.daidir",
            work / "01_RTL/csrc",
            work / "02_SYN/work",
            work / "02_SYN/Netlist",
            work / "02_SYN/alib-52",
        ]
        for path in files:
            if path.is_file() or path.is_symlink():
                path.unlink()
        for path in directories:
            if path.is_dir():
                shutil.rmtree(path)

    def _finish(self, result_path: Path, result: EvalResult, work: Path) -> EvalResult:
        atomic_write_json(result_path, result.to_dict())
        if result.mode == "proxy":
            self._prune_proxy_artifacts(work)
        return result

    def evaluate(self, candidate: Path, period: float, full: bool) -> EvalResult:
        mode = "full" if full else "proxy"
        candidate_hash = sha256_file(candidate)
        work = self._work_dir(candidate, period, mode)
        result_path = work / "result.json"
        if result_path.is_file():
            cached = EvalResult.from_dict(json.loads(result_path.read_text()))
            if (
                cached.candidate_sha256 == candidate_hash
                and cached.period_ns == period
                and cached.mode == mode
            ):
                # PrimeTime 2025.06 may emit an irrelevant optional matplotlib
                # GUI-import traceback in batch mode. Older cache entries could
                # consequently label a real timing violation as a power error.
                if (
                    cached.failure_stage in {"power", "timing"}
                    and cached.power_mw is not None
                    and cached.at is not None
                    and cached.slack_ns is not None
                    and -PT_TIMING_EPSILON_NS <= cached.slack_ns < 0
                ):
                    cached.timing_clean = True
                    cached.failure_stage = None
                    cached.ppa = cached.at * cached.power_mw
                    cached.feedback = (
                        f"Timing clean within {PT_TIMING_EPSILON_NS:.3f}ns "
                        f"SDF precision tolerance; slack={cached.slack_ns:.6f}ns; "
                        f"power={cached.power_mw:.6f}mW; PPA={cached.ppa:.3f}"
                    )
                    atomic_write_json(result_path, cached.to_dict())
                return cached

        self._prepare(candidate, period, work)
        result = EvalResult(candidate_hash, period, mode)

        rtl_compile = [
            "vcs",
            "-timescale=1ns/10ps",
            "-full64",
            "-sverilog",
            "-debug_access+all",
            "+lint=TFIPC-L",
            "-o",
            "simv",
            "initial.sv",
            "../00_TB/test.sv",
        ]
        rc, output, elapsed = run_logged(
            rtl_compile,
            work / "01_RTL",
            self.env,
            work / "01_RTL/compile.log",
            600,
        )
        result.runtimes_s["rtl_compile"] = elapsed
        if rc:
            result.failure_stage = "rtl_compile"
            result.feedback = compact_log(output)
            return self._finish(result_path, result, work)

        rc, output, elapsed = run_logged(
            [str(work / "01_RTL/simv")],
            work / "01_RTL",
            self.env,
            work / "01_RTL/simulation.log",
            60,
        )
        result.runtimes_s["rtl_simulation"] = elapsed
        result.rtl_pass, result.simulation_time_ps = parse_simulation(output)
        if result.simulation_time_ps is not None:
            result.latency_ns = result.simulation_time_ps / 1000.0
        if rc or not result.rtl_pass or result.latency_ns is None:
            result.failure_stage = "rtl_simulation"
            result.feedback = compact_log(output)
            return self._finish(result_path, result, work)

        rc, output, elapsed = run_logged(
            [str(DC_BIN), "-f", "syn.tcl"],
            work / "02_SYN",
            self.env,
            work / "02_SYN/dc.log",
            1800,
            memory_limit_gb=int(self.env.get("EVOLVE_DC_MEMORY_GB", "16")),
        )
        result.runtimes_s["dc"] = elapsed
        area_file = work / "02_SYN/Report/TOP_syn.area"
        timing_file = work / "02_SYN/Report/TOP_syn.timing_max"
        area_text = area_file.read_text(errors="replace") if area_file.is_file() else ""
        timing_text = timing_file.read_text(errors="replace") if timing_file.is_file() else ""
        result.area = parse_area(area_text)
        result.slack_ns, result.timing_clean = parse_slack(timing_text)
        if rc or result.area is None or result.slack_ns is None:
            result.failure_stage = "synthesis"
            result.feedback = compact_log(output + "\n" + timing_text + "\n" + area_text)
            return self._finish(result_path, result, work)
        if not result.timing_clean:
            result.failure_stage = "timing"
            result.feedback = compact_log(timing_text)
            return self._finish(result_path, result, work)

        result.at = result.area * result.latency_ns
        result.score = -result.at / SCORE_SCALE
        result.feedback = (
            f"RTL PASS; period={period:g}ns; latency={result.latency_ns:.3f}ns; "
            f"area={result.area:.3f}; slack={result.slack_ns:.3f}ns; "
            f"AT={result.at:.3f}; score={result.score:.6f}"
        )
        if not full:
            return self._finish(result_path, result, work)

        gate_compile = [
            "vcs",
            "-timescale=1ns/10ps",
            "-full64",
            "-sverilog",
            "-debug_access+all",
            "+maxdelays",
            "-negdelay",
            "+neg_tchk#tbed",
            "+define+SDF",
            str(CELL_VERILOG),
            "../02_SYN/Netlist/top_syn.v",
            "../00_TB/test.sv",
            "-o",
            "simv",
        ]
        rc, output, elapsed = run_logged(
            gate_compile,
            work / "03_GATE",
            self.env,
            work / "03_GATE/compile.log",
            900,
        )
        result.runtimes_s["gate_compile"] = elapsed
        if rc:
            result.gate_pass = False
            result.failure_stage = "gate_compile"
            result.feedback = compact_log(output)
            return self._finish(result_path, result, work)

        rc, output, elapsed = run_logged(
            [str(work / "03_GATE/simv")],
            work / "03_GATE",
            self.env,
            work / "03_GATE/simulation.log",
            1800,
        )
        result.runtimes_s["gate_simulation"] = elapsed
        result.gate_pass, _ = parse_simulation(output)
        if rc or not result.gate_pass or not (work / "03_GATE/top.vcd").is_file():
            result.failure_stage = "gate_simulation"
            result.feedback = compact_log(output)
            return self._finish(result_path, result, work)

        rc, output, elapsed = run_logged(
            [str(PT_BIN), "-f", "pt_script.tcl"],
            work / "04_POWER",
            self.env,
            work / "04_POWER/pt.log",
            900,
        )
        result.runtimes_s["pt"] = elapsed
        power_file = work / "04_POWER/top.power"
        power_text = power_file.read_text(errors="replace") if power_file.is_file() else ""
        result.power_mw = parse_power_mw(power_text)
        pt_wns_file = work / "04_POWER/top.wns"
        if pt_wns_file.is_file():
            try:
                result.slack_ns = float(pt_wns_file.read_text().strip())
                result.timing_clean = result.slack_ns >= -PT_TIMING_EPSILON_NS
            except ValueError:
                result.slack_ns = None
                result.timing_clean = False
        if not result.timing_clean:
            result.failure_stage = "timing"
            result.feedback = (
                f"PrimeTime timing violation: slack={result.slack_ns:.6f}ns; "
                f"power measurement completed: {result.power_mw:.6f}mW"
                if result.slack_ns is not None and result.power_mw is not None
                else "PrimeTime did not report a timing-clean result."
            )
            return self._finish(result_path, result, work)
        if rc or result.power_mw is None:
            result.failure_stage = "power"
            result.feedback = compact_log(output + "\n" + power_text)
            return self._finish(result_path, result, work)
        result.ppa = result.at * result.power_mw
        result.failure_stage = None
        result.feedback += f"; power={result.power_mw:.6f}mW; PPA={result.ppa:.3f}"
        return self._finish(result_path, result, work)


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS nodes (
    chain_id INTEGER NOT NULL,
    node_index INTEGER NOT NULL,
    idea_json TEXT NOT NULL,
    candidate_path TEXT NOT NULL,
    candidate_sha256 TEXT NOT NULL,
    thread_id TEXT,
    status TEXT NOT NULL,
    period_ns REAL NOT NULL,
    score REAL NOT NULL,
    at REAL,
    result_json TEXT NOT NULL,
    PRIMARY KEY (chain_id, node_index)
);
"""


def open_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.executescript(SCHEMA_SQL)
    connection.commit()
    return connection


def metadata_get(connection: sqlite3.Connection, key: str) -> str | None:
    row = connection.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def metadata_set(connection: sqlite3.Connection, key: str, value: Any) -> None:
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
        (key, json.dumps(value)),
    )
    connection.commit()


def git(command: Sequence[str], cwd: Path) -> str:
    completed = subprocess.run(
        ["git", *command],
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        check=False,
    )
    if completed.returncode:
        raise ExperimentError(f"git {' '.join(command)} failed:\n{completed.stdout}")
    return completed.stdout


def initialize_chain_workspace(workspace: Path, task: Task) -> None:
    if (workspace / ".git").is_dir():
        return
    workspace.mkdir(parents=True, exist_ok=True)
    shutil.copy2(task.baseline, workspace / "initial.sv")
    shutil.copy2(task.spec, workspace / "SPEC.md")
    atomic_write_text(
        workspace / "AGENTS.md",
        f"""# Candidate workspace rules

Edit only `initial.sv`. Preserve the `{task.top}` module name, every port, reset polarity,
and all externally visible behavior. Do not create helper files or modify SPEC.md.
The external harness performs simulation and synthesis after each turn.
""",
    )
    git(["init", "-q"], workspace)
    git(["config", "user.name", "ICRTL EvolVE"], workspace)
    git(["config", "user.email", "evolve@localhost"], workspace)
    git(["add", "initial.sv", "SPEC.md", "AGENTS.md"], workspace)
    git(["commit", "-q", "-m", "baseline"], workspace)


def parse_thread_id(jsonl_path: Path) -> str | None:
    if not jsonl_path.is_file():
        return None
    for line in jsonl_path.read_text(errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        for key in ("thread_id", "session_id"):
            if isinstance(event.get(key), str):
                return event[key]
        if event.get("type") in {"thread.started", "session.started"}:
            nested = event.get("thread") or event.get("session") or {}
            if isinstance(nested, dict):
                for key in ("id", "thread_id", "session_id"):
                    if isinstance(nested.get(key), str):
                        return nested[key]
    return None


class CodexRunner:
    def __init__(self, env: dict[str, str], model: str, effort: str):
        self.env = env
        self.model = model
        self.effort = effort

    def _common(self) -> list[str]:
        return [
            "--ignore-user-config",
            "--strict-config",
            "-m",
            self.model,
            "-c",
            f'model_reasoning_effort="{self.effort}"',
            "-c",
            'approval_policy="never"',
            "-c",
            'sandbox_mode="read-only"',
            "-c",
            'web_search="disabled"',
        ]

    def generate_idea(
        self, task: Task, index: int, previous: list[dict[str, str]], output_dir: Path
    ) -> dict[str, str]:
        output_dir.mkdir(parents=True, exist_ok=True)
        spec = task.spec.read_text(encoding="utf-8", errors="replace")
        baseline = task.baseline.read_text(encoding="utf-8", errors="replace")
        prompt = f"""Act as a senior RTL architect optimizing {task.name}.
Propose one high-level architectural optimization for lower area × latency while
preserving exact interface and testbench behavior. It must be materially distinct
from the prior ideas below. All required context is included here; do not call tools
or attempt to read files. Return only the requested JSON object.

Prior ideas:
{json.dumps(previous, indent=2)}

SPECIFICATION:
{spec}

CURRENT RTL:
{baseline}
"""
        jsonl = output_dir / f"idea_{index:03d}.events.jsonl"
        response = output_dir / f"idea_{index:03d}.json"
        command = [
            "codex",
            "exec",
            *self._common(),
            "-s",
            "read-only",
            "--json",
            "--output-schema",
            str(EVOLVE_DIR / "idea.schema.json"),
            "-o",
            str(response),
            prompt,
        ]
        with jsonl.open("w", encoding="utf-8") as stdout, (
            output_dir / f"idea_{index:03d}.stderr.log"
        ).open("w", encoding="utf-8") as stderr:
            completed = subprocess.run(
                command,
                cwd=str(REPO_ROOT),
                env=self.env,
                stdout=stdout,
                stderr=stderr,
                text=True,
                timeout=1800,
                check=False,
            )
        if completed.returncode or not response.is_file():
            raise ExperimentError(f"Codex idea generation failed for idea {index}.")
        try:
            idea = json.loads(response.read_text())
        except json.JSONDecodeError as exc:
            raise ExperimentError(f"Codex returned invalid idea JSON: {exc}") from exc
        required = {"title", "idea", "expected_effect", "risk"}
        if not required.issubset(idea):
            raise ExperimentError(f"Codex idea is missing fields: {required - set(idea)}")
        return {key: str(idea[key]) for key in required}

    def edit_candidate(
        self,
        workspace: Path,
        prompt: str,
        output_dir: Path,
        thread_id: str | None,
        top: str = "TOP",
    ) -> str:
        output_dir.mkdir(parents=True, exist_ok=True)
        jsonl = output_dir / "codex.events.jsonl"
        response = output_dir / "codex.final.txt"
        if thread_id:
            command = [
                "codex",
                "exec",
                "resume",
                *self._common(),
                "--json",
                "--output-schema",
                str(EVOLVE_DIR / "candidate.schema.json"),
                "-o",
                str(response),
                thread_id,
                prompt,
            ]
        else:
            command = [
                "codex",
                "exec",
                *self._common(),
                "-C",
                str(workspace),
                "-s",
                "read-only",
                "--json",
                "--output-schema",
                str(EVOLVE_DIR / "candidate.schema.json"),
                "-o",
                str(response),
                prompt,
            ]
        with jsonl.open("w", encoding="utf-8") as stdout, (
            output_dir / "codex.stderr.log"
        ).open("w", encoding="utf-8") as stderr:
            completed = subprocess.run(
                command,
                cwd=str(workspace),
                env=self.env,
                stdout=stdout,
                stderr=stderr,
                text=True,
                timeout=3600,
                check=False,
            )
        if completed.returncode:
            raise ExperimentError(f"Codex candidate edit failed in {workspace}.")
        new_thread = parse_thread_id(jsonl) or thread_id
        if not new_thread:
            raise ExperimentError("Codex JSONL did not contain a session/thread id.")
        if not response.is_file():
            raise ExperimentError("Codex did not produce a candidate response.")
        try:
            candidate = json.loads(response.read_text())
        except json.JSONDecodeError as exc:
            raise ExperimentError(f"Codex returned invalid candidate JSON: {exc}") from exc
        verilog = str(candidate.get("verilog", "")).strip()
        if not re.search(rf"\bmodule\s+{re.escape(top)}\b", verilog) or not re.search(
            r"\bendmodule\b", verilog
        ):
            raise ExperimentError(
                f"Codex candidate does not contain a complete {top} module."
            )
        atomic_write_text(workspace / "initial.sv", verilog + "\n")
        status = git(["status", "--porcelain"], workspace).splitlines()
        if status not in ([], [" M initial.sv"]):
            raise ExperimentError(f"Candidate workspace changed unexpectedly: {status}")
        return new_thread


def make_root_prompt(
    idea: dict[str, str], spec_text: str, current_rtl: str, top: str = "TOP"
) -> str:
    return f"""Implement this PPA optimization idea by editing only initial.sv:

Title: {idea['title']}
Idea: {idea['idea']}
Expected effect: {idea['expected_effect']}
Risk: {idea['risk']}

All required context is included below. Do not call tools or attempt to read files.
Preserve the exact {top} interface, reset behavior, functional results, and protocol
timing unless a cycle reduction remains fully compatible with the supplied protocol.
Return the complete synthesizable SystemVerilog candidate and concise summary using
the requested JSON schema. Do not wrap the RTL in Markdown fences.

SPECIFICATION:
{spec_text}

CURRENT RTL:
{current_rtl}
"""


def make_refine_prompt(
    idea: dict[str, str],
    result: EvalResult,
    baseline: EvalResult,
    best_at: float | None,
    spec_text: str,
    current_rtl: str,
    top: str = "TOP",
) -> str:
    relation = ""
    if result.at is not None and baseline.at is not None:
        relation = f" Current AT is {result.at / baseline.at:.4f}× baseline."
    best = f" Best archived AT is {best_at:.3f}." if best_at is not None else ""
    return f"""Refine initial.sv again, continuing the chain for idea `{idea['title']}`.

Evaluator feedback:
{result.feedback}
{relation}{best}

If compilation or simulation failed, diagnose and repair correctness first. If it
passed, make a targeted architectural or RTL improvement to reduce area × latency,
using the area and timing feedback. Preserve the exact {top} interface and observable
behavior. All required context is included below; do not call tools or read files.
Return the complete synthesizable SystemVerilog candidate and summary using the
requested JSON schema, without Markdown fences.

SPECIFICATION:
{spec_text}

CURRENT RTL:
{current_rtl}
"""


def record_node(
    connection: sqlite3.Connection,
    chain_id: int,
    node_index: int,
    idea: dict[str, str],
    candidate: Path,
    thread_id: str,
    result: EvalResult,
) -> None:
    connection.execute(
        """INSERT OR REPLACE INTO nodes(
            chain_id, node_index, idea_json, candidate_path, candidate_sha256,
            thread_id, status, period_ns, score, at, result_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            chain_id,
            node_index,
            json.dumps(idea, sort_keys=True),
            str(candidate),
            result.candidate_sha256,
            thread_id,
            "evaluated",
            result.period_ns,
            result.score,
            result.at,
            json.dumps(result.to_dict(), sort_keys=True),
        ),
    )
    connection.commit()


def record_generated_node(
    connection: sqlite3.Connection,
    chain_id: int,
    node_index: int,
    idea: dict[str, str],
    candidate: Path,
    thread_id: str | None,
    period_ns: float,
    codex_generation_s: float | None = None,
) -> None:
    """Checkpoint generated RTL before starting a restart-sensitive EDA run."""
    candidate_hash = sha256_file(candidate)
    pending = EvalResult(
        candidate_sha256=candidate_hash,
        period_ns=period_ns,
        mode="proxy",
        feedback="Generated candidate is awaiting evaluation.",
        runtimes_s=(
            {"codex_generation": codex_generation_s}
            if codex_generation_s is not None
            else {}
        ),
    )
    connection.execute(
        """INSERT OR REPLACE INTO nodes(
            chain_id, node_index, idea_json, candidate_path, candidate_sha256,
            thread_id, status, period_ns, score, at, result_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            chain_id,
            node_index,
            json.dumps(idea, sort_keys=True),
            str(candidate),
            candidate_hash,
            thread_id,
            "generated",
            period_ns,
            PENALTY,
            None,
            json.dumps(pending.to_dict(), sort_keys=True),
        ),
    )
    connection.commit()


def load_baseline_results(path: Path) -> dict[float, EvalResult]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text())
    return {float(key): EvalResult.from_dict(value) for key, value in data.items()}


def evaluate_baseline(
    evaluator: CommercialEvaluator,
    baseline: Path,
    periods: list[float],
    output_path: Path,
) -> dict[float, EvalResult]:
    results = load_baseline_results(output_path)
    for period in periods:
        if period not in results or not results[period].valid_full:
            print(f"[baseline] full commercial evaluation at {period:g} ns", flush=True)
            results[period] = evaluator.evaluate(baseline, period, full=True)
            atomic_write_json(
                output_path, {str(key): value.to_dict() for key, value in results.items()}
            )
    return results


def choose_search_period(results: dict[float, EvalResult]) -> float:
    valid = [result for result in results.values() if result.valid_full]
    if not valid:
        details = "\n".join(
            f"{period:g}ns: {result.failure_stage}: {result.feedback}"
            for period, result in sorted(results.items())
        )
        raise ExperimentError("No timing-clean baseline operating point:\n" + details)
    return min(valid, key=lambda item: item.at if item.at is not None else float("inf")).period_ns


def export_nodes(connection: sqlite3.Connection, path: Path) -> None:
    rows = connection.execute(
        """SELECT chain_id, node_index, candidate_path, candidate_sha256,
                  period_ns, score, at, result_json
           FROM nodes ORDER BY chain_id, node_index"""
    ).fetchall()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "chain_id",
                "node_index",
                "candidate_path",
                "candidate_sha256",
                "period_ns",
                "score",
                "at",
                "rtl_pass",
                "timing_clean",
                "failure_stage",
                "codex_generation_s",
            ]
        )
        for row in rows:
            result = json.loads(row[7])
            writer.writerow(
                [
                    *row[:7],
                    result["rtl_pass"],
                    result["timing_clean"],
                    result["failure_stage"],
                    result.get("runtimes_s", {}).get("codex_generation"),
                ]
            )


def select_finalists(
    connection: sqlite3.Connection, count: int, baseline: Path
) -> list[tuple[str, Path]]:
    rows = connection.execute(
        """SELECT candidate_sha256, candidate_path, MIN(at)
           FROM nodes
           WHERE status = 'evaluated' AND at IS NOT NULL
           GROUP BY candidate_sha256
           ORDER BY MIN(at) ASC
           LIMIT ?""",
        (count,),
    ).fetchall()
    finalists = [(row[0], Path(row[1])) for row in rows]
    if not finalists:
        finalists.append((sha256_file(baseline), baseline))
    return finalists


def validate_finalists(
    evaluator: CommercialEvaluator,
    finalists: list[tuple[str, Path]],
    periods: list[float],
    baseline: dict[float, EvalResult],
    output_dir: Path,
) -> dict[str, Any]:
    points: list[dict[str, Any]] = []
    for rank, (candidate_hash, candidate_path) in enumerate(finalists, start=1):
        for period in periods:
            print(
                f"[finalist {rank}] full commercial evaluation at {period:g} ns",
                flush=True,
            )
            result = evaluator.evaluate(candidate_path, period, full=True)
            baseline_result = baseline[period]
            ratio = None
            if (
                result.valid_full
                and baseline_result.valid_full
                and result.ppa is not None
                and baseline_result.ppa
            ):
                ratio = result.ppa / baseline_result.ppa
            points.append(
                {
                    "rank": rank,
                    "candidate_sha256": candidate_hash,
                    "candidate_path": str(candidate_path),
                    "period_ns": period,
                    "baseline_ppa": baseline_result.ppa,
                    "ppa_ratio": ratio,
                    "result": result.to_dict(),
                }
            )
    valid = [point for point in points if point["result"]["ppa"] is not None]
    best = min(valid, key=lambda point: point["result"]["ppa"]) if valid else None
    improved = [point for point in valid if point["ppa_ratio"] is not None and point["ppa_ratio"] < 1]
    best_improvement = min(improved, key=lambda point: point["ppa_ratio"]) if improved else None
    summary = {"points": points, "best": best, "best_same_period_improvement": best_improvement}
    atomic_write_json(output_dir / "summary.json", summary)
    with (output_dir / "finalists.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                "candidate_sha256",
                "candidate_path",
                "period_ns",
                "rtl_pass",
                "gate_pass",
                "timing_clean",
                "area",
                "latency_ns",
                "slack_ns",
                "power_mw",
                "ppa",
                "baseline_ppa",
                "ppa_ratio",
                "failure_stage",
            ]
        )
        for point in points:
            result = point["result"]
            writer.writerow(
                [
                    point["rank"],
                    point["candidate_sha256"],
                    point["candidate_path"],
                    point["period_ns"],
                    result["rtl_pass"],
                    result["gate_pass"],
                    result["timing_clean"],
                    result["area"],
                    result["latency_ns"],
                    result["slack_ns"],
                    result["power_mw"],
                    result["ppa"],
                    point["baseline_ppa"],
                    point["ppa_ratio"],
                    result["failure_stage"],
                ]
            )
    if best:
        best_dir = output_dir / "best"
        best_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(best["candidate_path"]), best_dir / "initial.sv")
        atomic_write_json(best_dir / "metrics.json", best)
    return summary


def parse_periods(raw: str) -> list[float]:
    periods = [float(item) for item in raw.split(",") if item.strip()]
    if not periods or any(value <= 0 for value in periods):
        raise argparse.ArgumentTypeError("periods must be positive comma-separated numbers")
    return periods


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(TASKS), default="Q3_HC")
    parser.add_argument("--ideas", type=int, default=12)
    parser.add_argument("--nodes-per-chain", type=int, default=5)
    parser.add_argument("--periods", type=parse_periods, default=parse_periods("3,4,5,6,7"))
    parser.add_argument("--finalists", type=int, default=3)
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--baseline-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.ideas < 1 or args.nodes_per_chain < 1 or args.finalists < 1:
        raise ExperimentError("ideas, nodes-per-chain, and finalists must be positive")
    task = TASKS[args.task]
    run_dir = (
        args.run_dir
        if args.run_dir is not None
        else EVOLVE_DIR / "runs" / f"{args.task.lower()}_igr_60"
    ).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    env = load_eda_environment()
    checks = preflight(env, args.model, args.reasoning_effort)
    atomic_write_json(run_dir / "preflight.json", checks)
    print("[preflight] Codex and EDA environment are ready", flush=True)
    if args.preflight_only:
        return 0

    connection = open_database(run_dir / "state.sqlite3")
    config = {
        "task": args.task,
        "ideas": args.ideas,
        "nodes_per_chain": args.nodes_per_chain,
        "periods": args.periods,
        "finalists": args.finalists,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
    }
    old_config = metadata_get(connection, "config")
    if old_config is not None and json.loads(old_config) != config:
        raise ExperimentError(
            "The run directory already contains a different configuration. "
            "Use a new --run-dir or restore the original arguments."
        )
    metadata_set(connection, "config", config)

    evaluator = CommercialEvaluator(task, env, run_dir / "evaluations")
    baseline = evaluate_baseline(
        evaluator, task.baseline, args.periods, run_dir / "baseline.json"
    )
    search_period = choose_search_period(baseline)
    metadata_set(connection, "search_period_ns", search_period)
    print(f"[baseline] search proxy period: {search_period:g} ns", flush=True)
    if args.baseline_only:
        return 0

    codex = CodexRunner(env, args.model, args.reasoning_effort)
    spec_text = task.spec.read_text(encoding="utf-8", errors="replace")
    ideas: list[dict[str, str]] = []
    ideas_dir = run_dir / "ideas"
    for chain_id in range(args.ideas):
        idea_path = ideas_dir / f"idea_{chain_id:03d}.json"
        if idea_path.is_file():
            idea = json.loads(idea_path.read_text())
        else:
            print(f"[idea {chain_id + 1}/{args.ideas}] generating", flush=True)
            idea_start = time.monotonic()
            idea = codex.generate_idea(task, chain_id, ideas, ideas_dir)
            idea_elapsed = time.monotonic() - idea_start
            atomic_write_json(idea_path, idea)
            atomic_write_json(
                ideas_dir / f"idea_{chain_id:03d}.timing.json",
                {"idea_generation_s": idea_elapsed},
            )
        ideas.append(idea)

    baseline_proxy = evaluator.evaluate(task.baseline, search_period, full=False)
    best_at_row = connection.execute("SELECT MIN(at) FROM nodes WHERE at IS NOT NULL").fetchone()
    best_at = best_at_row[0] if best_at_row and best_at_row[0] is not None else baseline_proxy.at

    for chain_id, idea in enumerate(ideas):
        chain_dir = run_dir / "chains" / f"chain_{chain_id:03d}"
        workspace = chain_dir / "workspace"
        initialize_chain_workspace(workspace, task)
        existing = {
            row[0]: row
            for row in connection.execute(
                """SELECT node_index, thread_id, status, candidate_path, result_json
                   FROM nodes WHERE chain_id = ? ORDER BY node_index""",
                (chain_id,),
            ).fetchall()
        }
        thread_id = existing[max(existing)][1] if existing else None
        for node_index in range(args.nodes_per_chain):
            if node_index in existing:
                row = existing[node_index]
                if row[2] == "generated":
                    candidate = Path(row[3])
                    pending = EvalResult.from_dict(json.loads(row[4]))
                    print(
                        f"[chain {chain_id + 1}/{args.ideas}, "
                        f"node {node_index + 1}/{args.nodes_per_chain}] "
                        "resuming saved EDA evaluation",
                        flush=True,
                    )
                    shutil.copy2(candidate, workspace / "initial.sv")
                    result = evaluator.evaluate(
                        candidate, search_period, full=False
                    )
                    result.runtimes_s.update(pending.runtimes_s)
                    atomic_write_json(
                        candidate.parent / "evaluation.json", result.to_dict()
                    )
                    record_node(
                        connection,
                        chain_id,
                        node_index,
                        idea,
                        candidate,
                        thread_id,
                        result,
                    )
                    git(["add", "initial.sv"], workspace)
                    git(
                        [
                            "commit",
                            "-q",
                            "--allow-empty",
                            "-m",
                            f"chain {chain_id} node {node_index}",
                        ],
                        workspace,
                    )
                    export_nodes(connection, run_dir / "nodes.csv")
                else:
                    result = EvalResult.from_dict(json.loads(row[4]))
                if result.at is not None:
                    best_at = min(best_at, result.at) if best_at is not None else result.at
                continue

            node_dir = chain_dir / "nodes" / f"node_{node_index:03d}"
            current_rtl = (workspace / "initial.sv").read_text(
                encoding="utf-8", errors="replace"
            )
            prompt = (
                make_root_prompt(idea, spec_text, current_rtl, task.top)
                if node_index == 0
                else make_refine_prompt(
                    idea, result, baseline_proxy, best_at, spec_text, current_rtl, task.top
                )
            )
            atomic_write_text(node_dir / "prompt.md", prompt)
            print(
                f"[chain {chain_id + 1}/{args.ideas}, "
                f"node {node_index + 1}/{args.nodes_per_chain}] Codex",
                flush=True,
            )
            candidate = node_dir / "initial.sv"
            codex_start = time.monotonic()
            try:
                thread_id = codex.edit_candidate(
                    workspace, prompt, node_dir, thread_id, task.top
                )
            except ExperimentError as exc:
                codex_elapsed = time.monotonic() - codex_start
                # A malformed/truncated model response or transient Codex failure
                # is a failed search node, not a reason to lose the whole run.
                # Keep the last valid parent as the chain state so the next turn
                # can continue repairing/refining it in the same Codex thread.
                message = f"Codex generation failed: {exc}"
                atomic_write_text(node_dir / "generation_error.txt", message + "\n")
                recovered_thread = parse_thread_id(
                    node_dir / "codex.events.jsonl"
                )
                if recovered_thread:
                    thread_id = recovered_thread
                shutil.copy2(workspace / "initial.sv", candidate)
                result = EvalResult(
                    candidate_sha256=sha256_file(candidate),
                    period_ns=search_period,
                    mode="proxy",
                    failure_stage="codex_candidate",
                    feedback=message,
                    runtimes_s={"codex_generation": codex_elapsed},
                )
                print(
                    f"[chain {chain_id + 1}/{args.ideas}, "
                    f"node {node_index + 1}/{args.nodes_per_chain}] "
                    f"generation failed; recording penalty and continuing",
                    flush=True,
                )
            else:
                codex_elapsed = time.monotonic() - codex_start
                shutil.copy2(workspace / "initial.sv", candidate)
                record_generated_node(
                    connection,
                    chain_id,
                    node_index,
                    idea,
                    candidate,
                    thread_id,
                    search_period,
                    codex_elapsed,
                )
                result = evaluator.evaluate(candidate, search_period, full=False)
                result.runtimes_s["codex_generation"] = codex_elapsed
            atomic_write_json(node_dir / "evaluation.json", result.to_dict())
            record_node(
                connection, chain_id, node_index, idea, candidate, thread_id, result
            )
            git(["add", "initial.sv"], workspace)
            git(
                [
                    "commit",
                    "-q",
                    "--allow-empty",
                    "-m",
                    f"chain {chain_id} node {node_index}",
                ],
                workspace,
            )
            if result.at is not None:
                best_at = min(best_at, result.at) if best_at is not None else result.at
            export_nodes(connection, run_dir / "nodes.csv")

    finalists = select_finalists(connection, args.finalists, task.baseline)
    if finalists[0][1] == task.baseline:
        print(
            "[finalists] no valid generated candidate; validating baseline as fallback",
            flush=True,
        )
    summary = validate_finalists(
        evaluator, finalists, args.periods, baseline, run_dir
    )
    export_nodes(connection, run_dir / "nodes.csv")
    print(f"[done] results: {run_dir / 'summary.json'}", flush=True)
    improvement = summary.get("best_same_period_improvement")
    if improvement:
        print(
            f"[done] best same-period PPA ratio: {improvement['ppa_ratio']:.6f}",
            flush=True,
        )
    else:
        print("[done] no finalist improved same-period baseline PPA", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExperimentError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
