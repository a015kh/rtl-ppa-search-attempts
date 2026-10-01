#!/usr/bin/env python3
"""Resumable four-pipeline IGR experiment for the ICRTL HC benchmark."""

from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import sqlite3
import statistics
import subprocess
import time
from typing import Any, Iterator, Sequence

from .evolve import (
    CodexRunner,
    ExperimentError,
    TASKS,
    atomic_write_json,
    atomic_write_text,
    initialize_chain_workspace,
    load_eda_environment,
    parse_thread_id,
    preflight,
    sha256_file,
)

from . import settings

HERE = Path(__file__).resolve().parent
REPO = settings.path('benchmark_root')
DEFAULT_PIPELINES = (
    "yosys_openroad",
    "trained_model",
    "yosys_trained_model",
    "dc_pt",
)
PERIODS = (1.0, 3.0, 5.0, 8.0, 10.0, 15.0)
RAW_CHECKPOINT = settings.path('raw_checkpoint')
YOSYS_CHECKPOINT = settings.path('yosys_checkpoint')
PDK_ROOT = settings.path('nangate_synopsys_root')
REV_ROOT = settings.ROOT
REV_PDK = settings.path('revolution_pdk')
CELL_MODELS = REV_PDK / 'work_around_yosys/cells.v'
OPENROAD = settings.tool('openroad')
YOSYS = settings.tool('yosys')
DC = settings.tool('dc_shell')
PT = settings.tool('pt_shell')
SYN_LIB = settings.path('synthesis_library_dir')
MAP_ADDERS = settings.path('map_adders')
MAP_LATCHES = settings.path('map_latches')
TIMING_EPSILON_NS = 0.005
MODEL_METRICS = ("area", "delay", "total_power")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def period_tag(period: float) -> str:
    return f"{period:g}".replace(".", "p")


def parse_periods(raw: str) -> list[float]:
    result = [float(value) for value in raw.split(",") if value.strip()]
    if not result or any(value <= 0 for value in result):
        raise argparse.ArgumentTypeError("periods must be positive comma-separated values")
    return result


def parse_pipelines(raw: str) -> list[str]:
    result = [value.strip() for value in raw.split(",") if value.strip()]
    unknown = set(result) - set(DEFAULT_PIPELINES)
    if not result or unknown:
        raise argparse.ArgumentTypeError(
            f"pipelines must be selected from {','.join(DEFAULT_PIPELINES)}"
        )
    return result


SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS metadata(
  key TEXT PRIMARY KEY, value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions(
  session_id INTEGER PRIMARY KEY AUTOINCREMENT,
  started_at TEXT NOT NULL, completed_at TEXT, status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS timing_events(
  event_id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id INTEGER NOT NULL,
  pipeline TEXT, chain_id INTEGER, node_index INTEGER,
  candidate_sha256 TEXT, period_ns REAL, stage TEXT NOT NULL,
  attempt INTEGER NOT NULL DEFAULT 1, status TEXT NOT NULL,
  started_at TEXT NOT NULL, completed_at TEXT, wall_seconds REAL,
  returncode INTEGER, cache_hit INTEGER NOT NULL DEFAULT 0,
  details_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS roots(
  chain_id INTEGER PRIMARY KEY, idea_json TEXT NOT NULL,
  candidate_path TEXT, candidate_sha256 TEXT, thread_id TEXT,
  status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS nodes_v2(
  pipeline TEXT NOT NULL, chain_id INTEGER NOT NULL, node_index INTEGER NOT NULL,
  candidate_path TEXT NOT NULL, candidate_sha256 TEXT NOT NULL,
  thread_id TEXT, status TEXT NOT NULL, score REAL,
  selected_period_ns REAL, feedback TEXT NOT NULL,
  PRIMARY KEY(pipeline,chain_id,node_index)
);
CREATE TABLE IF NOT EXISTS point_results(
  pipeline TEXT NOT NULL, candidate_sha256 TEXT NOT NULL, period_ns REAL NOT NULL,
  status TEXT NOT NULL, score REAL, area_um2 REAL, cycles INTEGER,
  latency_ns REAL, delay_ns REAL, slack_ns REAL, power_mw REAL,
  result_json TEXT NOT NULL,
  PRIMARY KEY(pipeline,candidate_sha256,period_ns)
);
"""


class State:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.db.execute(
            """UPDATE timing_events SET status='interrupted',completed_at=?
               WHERE status='running'""",
            (utc_now(),),
        )
        self.db.commit()
        row = self.db.execute(
            "INSERT INTO sessions(started_at,status) VALUES(?,'running') RETURNING session_id",
            (utc_now(),),
        ).fetchone()
        self.session_id = int(row[0])
        self.db.commit()

    def close(self, status: str = "completed") -> None:
        self.db.execute(
            "UPDATE sessions SET completed_at=?,status=? WHERE session_id=?",
            (utc_now(), status, self.session_id),
        )
        self.db.commit()
        self.db.close()

    def metadata(self, key: str, value: Any | None = None) -> Any:
        if value is None:
            row = self.db.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
            return json.loads(row[0]) if row else None
        self.db.execute(
            "INSERT OR REPLACE INTO metadata(key,value) VALUES(?,?)",
            (key, json.dumps(value, sort_keys=True)),
        )
        self.db.commit()
        return value

    @contextlib.contextmanager
    def event(
        self,
        stage: str,
        *,
        pipeline: str | None = None,
        chain: int | None = None,
        node: int | None = None,
        candidate: str | None = None,
        period: float | None = None,
        cache_hit: bool = False,
        details: dict[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        started_at = utc_now()
        started = time.monotonic()
        row = self.db.execute(
            """INSERT INTO timing_events(
                 session_id,pipeline,chain_id,node_index,candidate_sha256,period_ns,
                 stage,status,started_at,cache_hit,details_json)
               VALUES(?,?,?,?,?,?,?,'running',?,?,?) RETURNING event_id""",
            (
                self.session_id, pipeline, chain, node, candidate, period, stage,
                started_at, int(cache_hit), json.dumps(details or {}, sort_keys=True),
            ),
        ).fetchone()
        event_id = int(row[0])
        self.db.commit()
        result: dict[str, Any] = {}
        status = "success"
        try:
            yield result
        except BaseException as exc:
            status = "failed"
            result.setdefault("error", f"{type(exc).__name__}: {exc}")
            raise
        finally:
            if cache_hit:
                status = "cache_hit"
            self.db.execute(
                """UPDATE timing_events SET status=?,completed_at=?,wall_seconds=?,
                   returncode=?,details_json=? WHERE event_id=?""",
                (
                    status, utc_now(), time.monotonic() - started,
                    result.get("returncode"),
                    json.dumps({**(details or {}), **result}, sort_keys=True),
                    event_id,
                ),
            )
            self.db.commit()

    def run(
        self,
        command: Sequence[str],
        cwd: Path,
        log: Path,
        timeout: int,
        env: dict[str, str],
        **event_args: Any,
    ) -> tuple[int, str]:
        log.parent.mkdir(parents=True, exist_ok=True)
        with self.event(details={"command": list(command)}, **event_args) as event:
            started = time.monotonic()
            process = subprocess.Popen(
                list(command), cwd=cwd, env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, errors="replace",
                start_new_session=True,
            )
            try:
                output, _ = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    output, _ = process.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    output, _ = process.communicate()
                output += f"\nTIMEOUT after {timeout}s\n"
                event["timeout"] = timeout
            event["returncode"] = process.returncode
            event["process_wall_seconds"] = time.monotonic() - started
            atomic_write_text(log, output)
            return int(process.returncode or 0), output


@dataclasses.dataclass
class Point:
    pipeline: str
    period_ns: float
    status: str = "failed"
    area_um2: float | None = None
    cycles: int | None = None
    latency_ns: float | None = None
    delay_ns: float | None = None
    slack_ns: float | None = None
    power_mw: float | None = None
    score: float | None = None
    feedback: str = ""
    extra: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return self.status == "success" and self.score is not None


def extract_number(pattern: str, text: str) -> float | None:
    found = re.findall(pattern, text, re.I | re.M)
    return float(found[-1]) if found else None


def power_to_mw(value: float, unit: str) -> float:
    scales = {"W": 1000.0, "mW": 1.0, "uW": 1e-3, "nW": 1e-6, "pW": 1e-9}
    return value * scales.get(unit, 1000.0)


def parse_power_mw(text: str) -> float | None:
    value = extract_number(r"Total Power\s*=\s*([0-9.eE+-]+)", text)
    if value is None:
        # OpenROAD table: Total ... <power>
        match = re.findall(
            r"^\s*Total\s+[0-9.eE+-]+\s+[0-9.eE+-]+\s+[0-9.eE+-]+"
            r"\s+([0-9.eE+-]+)(?:\s|$)",
            text,
            re.M,
        )
        return power_to_mw(float(match[-1]), "W") if match else None
    if value is None:
        return None
    unit_match = re.search(r"Power\s+Units?\s*:\s*(\w+)", text, re.I)
    return power_to_mw(value, unit_match.group(1) if unit_match else "W")


def score_point(point: Point) -> Point:
    values = (point.area_um2, point.latency_ns, point.power_mw)
    if point.status == "success" and all(
        value is not None and math.isfinite(value) and value > 0 for value in values
    ):
        point.score = -float(point.area_um2 * point.latency_ns * point.power_mw)
    else:
        point.score = None
    return point


def patch_testbench(source: Path, destination: Path, period: float) -> None:
    text = source.read_text(encoding="utf-8", errors="replace")
    literal = f"{period:.6f}".rstrip("0")
    if literal.endswith("."):
        literal += "0"
    updated, count = re.subn(
        r"(`define\s+CYCLE\s+)([0-9.]+)", rf"\g<1>{literal}", text, count=1,
        flags=re.I,
    )
    if count != 1:
        raise ExperimentError(f"cannot patch CYCLE in {source}")
    atomic_write_text(destination, updated)


def parse_vcs(output: str, period: float) -> tuple[bool, int | None, float | None]:
    passed = bool(re.search(r"\bAll tests PASS!?\b", output, re.I))
    match = re.findall(r"total time:\s*(\d+)\s*cycles", output, re.I)
    cycles = int(match[-1]) if match else None
    return passed, cycles, cycles * period if cycles is not None else None


def pdk_files() -> dict[str, Path]:
    root = PDK_ROOT / "NanGate45"
    return {
        "lib": root / "lib/NangateOpenCellLibrary_typical.lib",
        "db": root / "db/NangateOpenCellLibrary_typical.db",
        "tech_lef": root / "lef/NangateOpenCellLibrary.tech.lef",
        "cell_lef": root / "lef/NangateOpenCellLibrary.macro.mod.lef",
    }


def render_sdc(period: float) -> str:
    io = 0.6 if period == 1.0 else 1.0
    return f"""create_clock -name clk -period {period:g} [get_ports clk]
set_input_delay {io:g} -clock clk [get_ports {{rst gray_valid gray_data}}]
set_output_delay {io:g} -clock clk [all_outputs]
set_clock_uncertainty 0.1 [all_clocks]
set_clock_latency 0.5 [all_clocks]
set_load 1 [all_outputs]
set_drive 0.1 [get_ports {{rst gray_valid gray_data}}]
"""


class FunctionalVCS:
    def __init__(self, state: State, run_dir: Path, env: dict[str, str], periods: list[float]):
        self.state, self.root, self.env, self.periods = state, run_dir / "functional", env, periods
        self.tb = TASKS["Q3_HC"].testbench

    def evaluate(
        self, candidate: Path, pipeline: str, chain: int, node: int
    ) -> dict[float, tuple[bool, int | None, float | None]]:
        digest = sha256_file(candidate)
        results: dict[float, tuple[bool, int | None, float | None]] = {}
        for period in self.periods:
            work = self.root / digest / f"{period_tag(period)}ns"
            result_path = work / "result.json"
            if result_path.is_file():
                data = json.loads(result_path.read_text())
                with self.state.event(
                    "vcs_functional", pipeline=pipeline, chain=chain, node=node,
                    candidate=digest, period=period, cache_hit=True,
                ):
                    pass
                results[period] = (data["passed"], data["cycles"], data["latency_ns"])
                continue
            rtl_dir = work / "01_RTL"
            tb_dir = work / "00_TB"
            rtl_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, rtl_dir / "initial.sv")
            shutil.copytree(self.tb, tb_dir, dirs_exist_ok=True)
            patch_testbench(self.tb / "test.sv", tb_dir / "test.sv", period)
            compile_cmd = [
                "vcs", "-timescale=1ns/10ps", "-full64", "-sverilog", "-o", "simv",
                "initial.sv", "../00_TB/test.sv",
            ]
            rc, output = self.state.run(
                compile_cmd, rtl_dir, rtl_dir / "compile.log", 600, self.env,
                stage="vcs_compile", pipeline=pipeline, chain=chain, node=node,
                candidate=digest, period=period,
            )
            if rc:
                result = (False, None, None)
            else:
                rc, output = self.state.run(
                    [str(rtl_dir / "simv")], rtl_dir, rtl_dir / "simulation.log", 60, self.env,
                    stage="vcs_simulation", pipeline=pipeline, chain=chain, node=node,
                    candidate=digest, period=period,
                )
                result = parse_vcs(output, period)
                if rc:
                    result = (False, result[1], result[2])
            atomic_write_json(
                result_path,
                {"passed": result[0], "cycles": result[1], "latency_ns": result[2]},
            )
            results[period] = result
        return results


def render_yosys(candidate: Path, output: Path, liberty: Path, period: float) -> str:
    abc = REV_ROOT / "scripts/ref/ref.abc.script"
    adders = MAP_ADDERS
    latches = MAP_LATCHES
    return f"""yosys -import
read_verilog -defer -sv {{{candidate}}}
read_liberty -lib {{{liberty}}}
synth -top TOP -flatten
opt -purge
extract_fa
techmap -map {{{adders}}}
techmap
opt -fast -purge
techmap -map {{{latches}}}
dfflibmap -liberty {{{liberty}}}
opt
abc -D {int(period * 1000)} -script {{{abc}}} -liberty {{{liberty}}}
setundef -zero
splitnets
opt_clean -purge
hilomap -singleton -hicell LOGIC1_X1 Z -locell LOGIC0_X1 Z
insbuf -buf BUF_X1 A Z
write_verilog -noattr -noexpr -nohex -nodec {{{output}}}
"""


class ModelProxy:
    def __init__(self, state: State, checkpoint: Path, device: str, name: str):
        self.state, self.checkpoint, self.device, self.name = state, checkpoint, device, name
        self.torch = self.model = self.tokenizer = None

    def load(self) -> None:
        if self.model is not None:
            return
        with self.state.event("model_checkpoint_load", pipeline=self.name, details={"checkpoint": str(self.checkpoint)}):
            try:
                import torch
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            except ImportError as exc:
                raise ExperimentError(
                    "Trained pipelines require the optional PPA dependencies; install the release with [ppa]."
                ) from exc
            self.torch = torch
            self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, local_files_only=True)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.checkpoint, local_files_only=True, dtype=torch.bfloat16
            ).to(self.device)
            self.model.eval()
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()

    def predict(
        self, rtl: str, period: float, seed: int, event_args: dict[str, Any]
    ) -> tuple[dict[str, float] | None, dict[str, Any]]:
        self.load()
        prompt = (
            f"Predict the {', '.join(MODEL_METRICS)} for clock period {period:.1f}ns. "
            f"The RTL code is as follows:\n{rtl}\n"
        )
        with self.state.event("model_input_prepare", **event_args):
            tokens = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=settings.INFERENCE.get('max_input_len', 2048)
            )
            input_count = len(self.tokenizer(prompt, truncation=False)["input_ids"])
            batch = {key: value.to(self.device) for key, value in tokens.items()}
        self.torch.manual_seed(seed)
        if self.device.startswith("cuda"):
            self.torch.cuda.manual_seed_all(seed)
            self.torch.cuda.synchronize()
        with self.state.event("model_inference", **event_args):
            with self.torch.no_grad():
                ids = self.model.generate(
                    **batch, max_new_tokens=128, num_return_sequences=settings.INFERENCE.get('num_samples', 4),
                    do_sample=True, pad_token_id=self.tokenizer.pad_token_id,
                )
            if self.device.startswith("cuda"):
                self.torch.cuda.synchronize()
        with self.state.event("model_decode", **event_args):
            outputs = self.tokenizer.batch_decode(ids, skip_special_tokens=True)
            draws = []
            for output in outputs:
                try:
                    values = [float(x.strip()) for x in output.splitlines() if x.strip()]
                except ValueError:
                    continue
                if len(values) == 3 and all(math.isfinite(x) and x > 0 for x in values):
                    draws.append(values)
            medians = (
                {key: statistics.median(row[i] for row in draws) for i, key in enumerate(MODEL_METRICS)}
                if draws else None
            )
        return medians, {
            "draws": draws, "raw_outputs": outputs, "input_tokens": input_count,
            "truncated": input_count > settings.INFERENCE.get('max_input_len', 2048), "seed": seed,
        }

    def close(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
            if self.device.startswith("cuda"):
                self.torch.cuda.empty_cache()


class PipelineEvaluator:
    def __init__(
        self, name: str, state: State, run_dir: Path, env: dict[str, str],
        periods: list[float], vcs: FunctionalVCS, device: str,
        raw_checkpoint: Path, yosys_checkpoint: Path,
    ):
        self.name, self.state, self.root, self.env = name, state, run_dir / "evaluations", env
        self.periods, self.vcs = periods, vcs
        checkpoint = raw_checkpoint if name == "trained_model" else yosys_checkpoint
        self.model = ModelProxy(state, checkpoint, device, name) if "trained_model" in name else None

    def close(self) -> None:
        if self.model:
            self.model.close()

    def evaluate(self, candidate: Path, chain: int, node: int) -> tuple[float | None, float | None, str]:
        digest = sha256_file(candidate)
        functional = self.vcs.evaluate(candidate, self.name, chain, node)
        points = []
        for period in self.periods:
            cached = self.state.db.execute(
                """SELECT result_json FROM point_results
                   WHERE pipeline=? AND candidate_sha256=? AND period_ns=?""",
                (self.name, digest, period),
            ).fetchone()
            if cached:
                point = Point(**json.loads(cached[0]))
                with self.state.event(
                    "pipeline_point", pipeline=self.name, chain=chain, node=node,
                    candidate=digest, period=period, cache_hit=True,
                ):
                    pass
                points.append(point)
                continue
            passed, cycles, latency = functional[period]
            if not passed or cycles is None or latency is None:
                point = Point(
                    self.name, period, status="functional_failed", cycles=cycles,
                    latency_ns=latency, feedback="VCS functional simulation failed",
                )
            else:
                point = self._evaluate_point(candidate, digest, period, chain, node)
                with self.state.event(
                    "result_parse_and_score", pipeline=self.name, chain=chain,
                    node=node, candidate=digest, period=period,
                ):
                    point.cycles, point.latency_ns = cycles, latency
                    score_point(point)
            with self.state.event(
                "checkpoint_write", pipeline=self.name, chain=chain, node=node,
                candidate=digest, period=period,
            ):
                self._record_point(digest, point)
            points.append(point)
        valid = [point for point in points if point.valid]
        best = max(valid, key=lambda point: point.score) if valid else None
        feedback = self._feedback(points, best)
        return (
            best.score if best else None,
            best.period_ns if best else None,
            feedback,
        )

    def _work(self, digest: str, period: float) -> Path:
        work = self.root / self.name / digest / f"{period_tag(period)}ns"
        work.mkdir(parents=True, exist_ok=True)
        return work

    def _evaluate_point(
        self, candidate: Path, digest: str, period: float, chain: int, node: int
    ) -> Point:
        work = self._work(digest, period)
        atomic_write_text(work / "design.sdc", render_sdc(period))
        args = dict(
            pipeline=self.name, chain=chain, node=node, candidate=digest, period=period
        )
        if self.name == "trained_model":
            medians, extra = self.model.predict(
                candidate.read_text(errors="replace"), period,
                int.from_bytes(hashlib.sha256(f"{digest}:{period}:raw".encode()).digest()[:4], "big"),
                args,
            )
            return self._model_point(period, medians, extra)
        if self.name == "yosys_trained_model":
            netlist = work / "netlist.v"
            script = work / "yosys.tcl"
            atomic_write_text(script, render_yosys(candidate, netlist, REV_PDK / "Nangate45_typ.lib", period))
            rc, output = self.state.run(
                [str(YOSYS), "-Q", "-T", str(script)], work, work / "yosys.log", 1800,
                self.env, stage="yosys", **args,
            )
            if rc or not netlist.is_file():
                return Point(self.name, period, feedback=output[-4000:])
            medians, extra = self.model.predict(
                netlist.read_text(errors="replace"), period,
                int.from_bytes(hashlib.sha256(f"{digest}:{period}:yosys".encode()).digest()[:4], "big"),
                args,
            )
            return self._model_point(period, medians, extra)
        if self.name == "yosys_openroad":
            return self._yosys_openroad(candidate, digest, period, work, args)
        return self._dc_pt(candidate, digest, period, work, args)

    @staticmethod
    def _model_point(period: float, values: dict[str, float] | None, extra: dict[str, Any]) -> Point:
        if values is None:
            return Point("", period, feedback="all model samples failed to decode", extra=extra)
        delay = float(values["delay"])
        clean = delay <= period + TIMING_EPSILON_NS
        return Point(
            "", period, status="success" if clean else "timing_failed",
            area_um2=float(values["area"]), delay_ns=delay, slack_ns=period - delay,
            power_mw=float(values["total_power"]), extra=extra,
            feedback="model prediction" if clean else "predicted timing violation",
        )

    def _yosys_openroad(
        self, candidate: Path, digest: str, period: float, work: Path, args: dict[str, Any]
    ) -> Point:
        pdk = pdk_files()
        netlist = work / "netlist.v"
        ys = work / "yosys.tcl"
        atomic_write_text(ys, render_yosys(candidate, netlist, pdk["lib"], period))
        rc, output = self.state.run(
            [str(YOSYS), "-Q", "-T", str(ys)], work, work / "yosys.log", 1800,
            self.env, stage="yosys", **args,
        )
        if rc or not netlist.is_file():
            return Point(self.name, period, feedback=output[-4000:])
        ort = work / "openroad.tcl"
        atomic_write_text(
            ort,
            f"""read_lef {{{pdk['tech_lef']}}}
read_lef {{{pdk['cell_lef']}}}
read_liberty {{{pdk['lib']}}}
read_verilog {{{netlist}}}
link_design TOP
read_sdc {{{work / 'design.sdc'}}}
initialize_floorplan -site FreePDK45_38x28_10R_NP_162NW_34O -utilization 50 -aspect_ratio 1.0 -core_space 0.0
source {{{REV_PDK / 'Nangate45.tracks'}}}
set_power_activity -input -activity 0.4 -duty 0.5
report_units
report_design_area
report_worst_slack -max
report_checks -path_delay max -group_count 1 -endpoint_count 1
report_power
exit
""",
        )
        rc, output = self.state.run(
            [str(OPENROAD), "-no_init", "-exit", "-threads", "1", str(ort)],
            work, work / "openroad.log", 1800, self.env, stage="openroad", **args,
        )
        area = extract_number(r"Design area\s+([0-9.eE+-]+)", output)
        slack = extract_number(r"worst slack\s+(-?[0-9.eE+-]+)", output)
        power = parse_power_mw(output)
        clean = slack is not None and slack >= -TIMING_EPSILON_NS
        return Point(
            self.name, period,
            status="success" if rc == 0 and clean and area and power else "timing_failed",
            area_um2=area, delay_ns=(period - slack) if slack is not None else None,
            slack_ns=slack, power_mw=power, feedback=output[-4000:],
        )

    def _dc_pt(
        self, candidate: Path, digest: str, period: float, work: Path, args: dict[str, Any]
    ) -> Point:
        pdk = pdk_files()
        netlist, area_report = work / "netlist.v", work / "area.rpt"
        dc_tcl = work / "dc.tcl"
        atomic_write_text(
            dc_tcl,
            f"""set search_path ". {pdk['db'].parent} {SYN_LIB} $search_path"
set target_library "{pdk['db'].name}"
set link_library "* $target_library dw_foundation.sldb"
set synthetic_library "dw_foundation.sldb"
set_host_options -max_cores 1
set verilogout_no_tri true
analyze -format sverilog {{{candidate}}}
elaborate TOP
current_design TOP
link
source -echo -verbose {{{work / 'design.sdc'}}}
set_ideal_network [get_ports clk]
set_dont_touch_network [all_clocks]
set_fix_hold [all_clocks]
set_max_fanout 20 [get_ports {{rst gray_valid gray_data}}]
uniquify
set_fix_multiple_port_nets -all -buffer_constants [get_designs *]
compile
write -format verilog -hierarchy -output {{{netlist}}}
report_area > {{{area_report}}}
report_timing -delay_type max > {{{work / 'dc_timing.rpt'}}}
exit
""",
        )
        rc, output = self.state.run(
            [str(DC), "-no_home_init", "-f", str(dc_tcl)], work, work / "dc.log",
            3600, self.env, stage="dc", **args,
        )
        if rc or not netlist.is_file():
            return Point(self.name, period, feedback=output[-4000:])
        fixed = work / "power_fixed.rpt"
        pt_tcl = work / "pt_fixed.tcl"
        atomic_write_text(
            pt_tcl,
            f"""set search_path ". {pdk['db'].parent} $search_path"
set target_library "{pdk['db'].name}"
set link_library "* $target_library"
set_host_options -max_cores 1
set power_enable_analysis true
set power_analysis_mode averaged
read_verilog {{{netlist}}}
current_design TOP
link
source -echo -verbose {{{work / 'design.sdc'}}}
set CLK_PORTS [filter_collection [all_inputs] "is_clock_used_as_clock == true"]
set DATA_PORTS [remove_from_collection [all_inputs] $CLK_PORTS]
set_switching_activity -static_probability 0.5 -toggle_rate 0.4 -clock_domains * $DATA_PORTS
update_timing
update_power
report_timing -delay_type max -nworst 1 > {{{work / 'pt_timing.rpt'}}}
report_power > {{{fixed}}}
exit
""",
        )
        rc, output = self.state.run(
            [str(PT), "-no_init", "-f", str(pt_tcl)], work, work / "pt.log", 900,
            self.env, stage="pt_fixed", **args,
        )
        area_text = area_report.read_text(errors="replace") if area_report.is_file() else ""
        timing_text = (work / "pt_timing.rpt").read_text(errors="replace") if (work / "pt_timing.rpt").is_file() else ""
        power_text = fixed.read_text(errors="replace") if fixed.is_file() else ""
        area = extract_number(r"Total cell area:\s*([0-9.eE+-]+)", area_text)
        slack = extract_number(r"slack\s+\((?:MET|VIOLATED)\)\s+(-?[0-9.eE+-]+)", timing_text)
        if slack is not None and "slack (VIOLATED)" in timing_text and slack > 0:
            slack = -slack
        power = parse_power_mw(power_text)
        clean = slack is not None and slack >= -TIMING_EPSILON_NS
        return Point(
            self.name, period,
            status="success" if rc == 0 and clean and area and power else "timing_failed",
            area_um2=area, delay_ns=(period - slack) if slack is not None else None,
            slack_ns=slack, power_mw=power,
            feedback=(timing_text + "\n" + power_text)[-4000:],
            extra={"netlist": str(netlist)},
        )

    def _record_point(self, digest: str, point: Point) -> None:
        if not point.pipeline:
            point.pipeline = self.name
        payload = dataclasses.asdict(point)
        self.state.db.execute(
            """INSERT OR REPLACE INTO point_results(
                 pipeline,candidate_sha256,period_ns,status,score,area_um2,cycles,
                 latency_ns,delay_ns,slack_ns,power_mw,result_json)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                self.name, digest, point.period_ns, point.status, point.score,
                point.area_um2, point.cycles, point.latency_ns, point.delay_ns,
                point.slack_ns, point.power_mw, json.dumps(payload, sort_keys=True),
            ),
        )
        self.state.db.commit()

    @staticmethod
    def _feedback(points: list[Point], best: Point | None) -> str:
        rows = []
        for p in points:
            rows.append(
                f"{p.period_ns:g}ns: {p.status}; area={p.area_um2}; "
                f"cycles={p.cycles}; latency={p.latency_ns}; delay={p.delay_ns}; "
                f"slack={p.slack_ns}; power={p.power_mw}; score={p.score}"
            )
        if best:
            rows.append(
                f"Best proxy point: {best.period_ns:g}ns, "
                f"PPA={-best.score:.6f}, score={best.score:.6f}"
            )
        else:
            rows.append("No functionally passing, timing-clean PPA point.")
        return "\n".join(rows)


def root_prompt(idea: dict[str, str], spec: str, baseline: str) -> str:
    return f"""Implement this HC PPA optimization idea in initial.sv.
Title: {idea['title']}
Idea: {idea['idea']}
Expected effect: {idea['expected_effect']}
Risk: {idea['risk']}
Preserve the exact TOP interface and behavior. Return a complete synthesizable TOP
module using the requested JSON schema. Do not call tools.
SPECIFICATION:
{spec}
CURRENT RTL:
{baseline}
"""


def refine_prompt(
    idea: dict[str, str], spec: str, rtl: str, feedback: str, pipeline: str
) -> str:
    return f"""Refine initial.sv for the `{pipeline}` PPA search and idea `{idea['title']}`.
The score is exactly -(area_um2 * latency_ns * total_power_mW); improve it while
preserving all functional behavior and the exact TOP interface.
Evaluator feedback:
{feedback}
SPECIFICATION:
{spec}
CURRENT RTL:
{rtl}
Return the complete synthesizable TOP module using the requested JSON schema.
"""


def init_workspace(path: Path, task: Any, rtl: Path) -> None:
    initialize_chain_workspace(path, task)
    shutil.copy2(rtl, path / "initial.sv")
    subprocess.run(["git", "add", "initial.sv"], cwd=path, check=True)
    subprocess.run(
        ["git", "commit", "-q", "--allow-empty", "-m", "shared root"],
        cwd=path, check=True,
    )


def record_node(
    state: State, pipeline: str, chain: int, node: int, candidate: Path,
    thread: str | None, score: float | None, period: float | None, feedback: str,
) -> None:
    state.db.execute(
        """INSERT OR REPLACE INTO nodes_v2(
             pipeline,chain_id,node_index,candidate_path,candidate_sha256,thread_id,
             status,score,selected_period_ns,feedback)
           VALUES(?,?,?,?,?,?, 'evaluated',?,?,?)""",
        (
            pipeline, chain, node, str(candidate), sha256_file(candidate), thread,
            score, period, feedback,
        ),
    )
    state.db.commit()


def create_ideas_and_roots(
    state: State, run_dir: Path, task: Any, codex: CodexRunner, ideas_count: int
) -> None:
    ideas: list[dict[str, str]] = []
    spec = task.spec.read_text(errors="replace")
    baseline = task.baseline.read_text(errors="replace")
    for chain in range(ideas_count):
        row = state.db.execute("SELECT * FROM roots WHERE chain_id=?", (chain,)).fetchone()
        if row and row["status"] == "generated":
            ideas.append(json.loads(row["idea_json"]))
            continue
        idea_path = run_dir / "ideas" / f"idea_{chain:03d}.json"
        if idea_path.is_file():
            idea = json.loads(idea_path.read_text())
        else:
            print(f"[idea {chain + 1}/{ideas_count}] generating", flush=True)
            with state.event("idea_generation", chain=chain):
                idea = codex.generate_idea(task, chain, ideas, run_dir / "ideas")
            atomic_write_json(idea_path, idea)
        root_dir = run_dir / "shared_roots" / f"chain_{chain:03d}"
        workspace = root_dir / "workspace"
        initialize_chain_workspace(workspace, task)
        candidate = root_dir / "initial.sv"
        if not candidate.is_file():
            print(f"[root {chain + 1}/{ideas_count}] Codex", flush=True)
            try:
                with state.event("codex_root_generation", chain=chain):
                    thread = codex.edit_candidate(
                        workspace, root_prompt(idea, spec, baseline), root_dir, None
                    )
            except ExperimentError as exc:
                # A malformed response consumes a node but must not destroy a
                # long resumable experiment. Use the baseline as this root.
                atomic_write_text(root_dir / "generation_error.txt", f"{exc}\n")
                thread = parse_thread_id(root_dir / "codex.events.jsonl")
                shutil.copy2(task.baseline, workspace / "initial.sv")
            shutil.copy2(workspace / "initial.sv", candidate)
        else:
            thread = parse_thread_id(root_dir / "codex.events.jsonl")
        state.db.execute(
            """INSERT OR REPLACE INTO roots(
                 chain_id,idea_json,candidate_path,candidate_sha256,thread_id,status)
               VALUES(?,?,?,?,?,'generated')""",
            (chain, json.dumps(idea, sort_keys=True), str(candidate), sha256_file(candidate), thread),
        )
        state.db.commit()
        ideas.append(idea)


def run_pipeline(
    state: State, run_dir: Path, task: Any, codex: CodexRunner,
    evaluator: PipelineEvaluator, ideas_count: int, nodes_per_chain: int,
) -> None:
    pipeline = evaluator.name
    spec = task.spec.read_text(errors="replace")
    with state.event("pipeline_search", pipeline=pipeline):
        for chain in range(ideas_count):
            root = state.db.execute("SELECT * FROM roots WHERE chain_id=?", (chain,)).fetchone()
            idea = json.loads(root["idea_json"])
            root_path = Path(root["candidate_path"])
            workspace = run_dir / "searches" / pipeline / f"chain_{chain:03d}" / "workspace"
            init_workspace(workspace, task, root_path)
            thread: str | None = None
            feedback = ""
            for node in range(nodes_per_chain):
                existing = state.db.execute(
                    "SELECT * FROM nodes_v2 WHERE pipeline=? AND chain_id=? AND node_index=?",
                    (pipeline, chain, node),
                ).fetchone()
                if existing and existing["status"] == "evaluated":
                    feedback, thread = existing["feedback"], existing["thread_id"]
                    shutil.copy2(existing["candidate_path"], workspace / "initial.sv")
                    continue
                if node == 0:
                    candidate = root_path
                else:
                    node_dir = run_dir / "searches" / pipeline / f"chain_{chain:03d}" / f"node_{node:03d}"
                    node_dir.mkdir(parents=True, exist_ok=True)
                    prompt = refine_prompt(
                        idea, spec, (workspace / "initial.sv").read_text(errors="replace"),
                        feedback, pipeline,
                    )
                    atomic_write_text(node_dir / "prompt.md", prompt)
                    print(
                        f"[{pipeline} chain {chain + 1}/{ideas_count}, "
                        f"node {node + 1}/{nodes_per_chain}] Codex",
                        flush=True,
                    )
                    try:
                        with state.event(
                            "codex_refinement", pipeline=pipeline, chain=chain, node=node
                        ):
                            thread = codex.edit_candidate(workspace, prompt, node_dir, thread)
                    except ExperimentError as exc:
                        atomic_write_text(node_dir / "generation_error.txt", f"{exc}\n")
                        thread = parse_thread_id(node_dir / "codex.events.jsonl") or thread
                    candidate = node_dir / "initial.sv"
                    shutil.copy2(workspace / "initial.sv", candidate)
                print(
                    f"[{pipeline} chain {chain + 1}/{ideas_count}, "
                    f"node {node + 1}/{nodes_per_chain}] six-clock evaluation",
                    flush=True,
                )
                score, period, feedback = evaluator.evaluate(candidate, chain, node)
                record_node(
                    state, pipeline, chain, node, candidate, thread, score, period, feedback
                )
                shutil.copy2(candidate, workspace / "initial.sv")
                subprocess.run(["git", "add", "initial.sv"], cwd=workspace, check=True)
                subprocess.run(
                    ["git", "commit", "-q", "--allow-empty", "-m", f"node {node}"],
                    cwd=workspace, check=True,
                )


def select_finalists(state: State, pipelines: list[str], count: int) -> dict[str, list[sqlite3.Row]]:
    result = {}
    for pipeline in pipelines:
        rows = state.db.execute(
            """SELECT * FROM nodes_v2 WHERE pipeline=? AND score IS NOT NULL
               ORDER BY score DESC""",
            (pipeline,),
        ).fetchall()
        unique: list[sqlite3.Row] = []
        seen: set[str] = set()
        for row in rows:
            if row["candidate_sha256"] in seen:
                continue
            seen.add(row["candidate_sha256"])
            unique.append(row)
            if len(unique) == count:
                break
        result[pipeline] = unique
    return result


def enable_vcd_dump(testbench: str) -> str:
    pattern = re.compile(
        r"`ifdef\s+SDF\s*\n"
        r"(initial\s+begin\s*\n\s*\$dumpfile\(\"top\.vcd\"\);\s*\n"
        r"\s*\$dumpvars\(0,\s*test\.TOP\);\s*\n\s*end)\s*\n"
        r"`endif",
        re.I,
    )
    updated, count = pattern.subn(r"\1", testbench, count=1)
    if count != 1:
        raise ExperimentError("cannot enable HC VCD dump in copied testbench")
    return updated


def vcd_golden_power(
    state: State,
    run_dir: Path,
    env: dict[str, str],
    candidate: Path,
    digest: str,
    period: float,
    netlist: Path,
) -> tuple[float | None, str, float | None]:
    work = run_dir / "golden_vcd" / digest / f"{period_tag(period)}ns"
    result_path = work / "result.json"
    if result_path.is_file():
        data = json.loads(result_path.read_text())
        with state.event(
            "golden_vcd", pipeline="dc_pt", candidate=digest, period=period,
            cache_hit=True,
        ):
            pass
        return data.get("power_mw"), data["status"], data.get("annotation_percent")
    gate = work / "03_GATE"
    tb_dir = work / "00_TB"
    pt_dir = work / "04_PT_VCD"
    gate.mkdir(parents=True, exist_ok=True)
    pt_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(TASKS["Q3_HC"].testbench, tb_dir, dirs_exist_ok=True)
    patch_testbench(TASKS["Q3_HC"].testbench / "test.sv", tb_dir / "test.sv", period)
    atomic_write_text(
        tb_dir / "test.sv",
        enable_vcd_dump((tb_dir / "test.sv").read_text(errors="replace")),
    )
    rc, output = state.run(
        [
            "vcs", "-timescale=1ns/10ps", "-full64", "-sverilog",
            str(CELL_MODELS), str(netlist), "../00_TB/test.sv", "-o", "simv",
        ],
        gate, gate / "compile.log", 900, env, stage="gate_vcd_compile",
        pipeline="dc_pt", candidate=digest, period=period,
    )
    status = "gate_compile_failed"
    power = coverage = None
    if rc == 0:
        rc, output = state.run(
            [str(gate / "simv")], gate, gate / "simulation.log", 1800, env,
            stage="gate_vcd_simulation", pipeline="dc_pt", candidate=digest,
            period=period,
        )
        passed, _, _ = parse_vcs(output, period)
        vcd = gate / "top.vcd"
        status = "gate_simulation_failed"
        if rc == 0 and passed and vcd.is_file():
            pdk = pdk_files()
            power_report = pt_dir / "power.rpt"
            activity_report = pt_dir / "activity.rpt"
            script = pt_dir / "pt_vcd.tcl"
            atomic_write_text(
                script,
                f"""set search_path ". {pdk['db'].parent} $search_path"
set target_library "{pdk['db'].name}"
set link_library "* $target_library"
set power_enable_analysis true
set power_analysis_mode time_based
read_verilog {{{netlist}}}
current_design TOP
link
source -echo -verbose {{{netlist.parent / 'design.sdc'}}}
read_vcd -strip_path test/TOP {{{vcd}}}
update_power
report_switching_activity > {{{activity_report}}}
report_power > {{{power_report}}}
exit
""",
            )
            rc, output = state.run(
                [str(PT), "-no_init", "-f", str(script)], pt_dir, pt_dir / "pt.log",
                900, env, stage="pt_vcd", pipeline="dc_pt", candidate=digest,
                period=period,
            )
            text = power_report.read_text(errors="replace") if power_report.is_file() else ""
            power = parse_power_mw(text)
            coverage = extract_number(
                r"Number of annotated nets\s*=\s*\d+\s*\(([0-9.]+)%\)",
                output,
            )
            status = "success" if rc == 0 and power is not None else "pt_vcd_failed"
    atomic_write_json(
        result_path,
        {"power_mw": power, "status": status, "annotation_percent": coverage},
    )
    return power, status, coverage


def validate_golden(
    state: State, run_dir: Path, env: dict[str, str], periods: list[float],
    vcs: FunctionalVCS, finalists: dict[str, list[sqlite3.Row]],
) -> None:
    evaluator = PipelineEvaluator(
        "dc_pt", state, run_dir / "golden", env, periods, vcs, "cuda:0",
        RAW_CHECKPOINT, YOSYS_CHECKPOINT,
    )
    seen: set[str] = set()
    rows = []
    baseline_digest = sha256_file(TASKS["Q3_HC"].baseline)
    baseline_scores = {
        row["period_ns"]: row["score"]
        for row in state.db.execute(
            """SELECT period_ns,score FROM point_results
               WHERE pipeline='dc_pt' AND candidate_sha256=?""",
            (baseline_digest,),
        )
    }
    baseline_rows = []
    for point in state.db.execute(
        """SELECT * FROM point_results WHERE pipeline='dc_pt'
           AND candidate_sha256=? ORDER BY period_ns""",
        (baseline_digest,),
    ):
        payload = json.loads(point["result_json"])
        netlist = Path(payload.get("extra", {}).get("netlist", ""))
        vcd_power = None
        vcd_status = "netlist_unavailable"
        annotation = None
        if netlist.is_file():
            vcd_power, vcd_status, annotation = vcd_golden_power(
                state, run_dir, env, TASKS["Q3_HC"].baseline, baseline_digest,
                float(point["period_ns"]), netlist,
            )
        baseline_rows.append(
            {
                "period_ns": point["period_ns"], "status": point["status"],
                "score": point["score"], "area_um2": point["area_um2"],
                "cycles": point["cycles"], "latency_ns": point["latency_ns"],
                "delay_ns": point["delay_ns"], "slack_ns": point["slack_ns"],
                "fixed_power_mw": point["power_mw"], "vcd_power_mw": vcd_power,
                "vcd_status": vcd_status, "vcd_annotation_percent": annotation,
            }
        )
    atomic_write_json(run_dir / "baseline_golden.json", baseline_rows)
    for source_pipeline, selected in finalists.items():
        for rank, node in enumerate(selected, 1):
            digest = node["candidate_sha256"]
            if digest not in seen:
                evaluator.evaluate(Path(node["candidate_path"]), -1, rank)
                seen.add(digest)
            for point in state.db.execute(
                """SELECT * FROM point_results WHERE pipeline='dc_pt'
                   AND candidate_sha256=? ORDER BY period_ns""",
                (digest,),
            ):
                payload = json.loads(point["result_json"])
                netlist = Path(payload.get("extra", {}).get("netlist", ""))
                vcd_power = None
                vcd_status = "netlist_unavailable"
                annotation = None
                if netlist.is_file():
                    vcd_power, vcd_status, annotation = vcd_golden_power(
                        state, run_dir, env, Path(node["candidate_path"]), digest,
                        float(point["period_ns"]), netlist,
                    )
                baseline_score = baseline_scores.get(point["period_ns"])
                ratio = (
                    baseline_score / point["score"]
                    if baseline_score is not None and point["score"] not in (None, 0)
                    else None
                )
                proxy = state.db.execute(
                    """SELECT * FROM point_results WHERE pipeline=?
                       AND candidate_sha256=? AND period_ns=?""",
                    (source_pipeline, digest, point["period_ns"]),
                ).fetchone()
                def relative_error(proxy_value: Any, golden_value: Any) -> float | None:
                    if proxy_value is None or golden_value in (None, 0):
                        return None
                    return (float(proxy_value) - float(golden_value)) / float(golden_value)
                rows.append(
                    {
                        "source_pipeline": source_pipeline, "proxy_rank": rank,
                        "candidate_sha256": digest, "period_ns": point["period_ns"],
                        "golden_status": point["status"], "golden_score": point["score"],
                        "area_um2": point["area_um2"], "cycles": point["cycles"],
                        "latency_ns": point["latency_ns"], "delay_ns": point["delay_ns"],
                        "slack_ns": point["slack_ns"], "fixed_power_mw": point["power_mw"],
                        "vcd_power_mw": vcd_power, "vcd_status": vcd_status,
                        "vcd_annotation_percent": annotation,
                        "baseline_golden_score": baseline_score,
                        "baseline_over_candidate_ppa": ratio,
                        "proxy_score": proxy["score"] if proxy else None,
                        "proxy_area_um2": proxy["area_um2"] if proxy else None,
                        "proxy_delay_ns": proxy["delay_ns"] if proxy else None,
                        "proxy_power_mw": proxy["power_mw"] if proxy else None,
                        "area_relative_error": relative_error(
                            proxy["area_um2"] if proxy else None, point["area_um2"]
                        ),
                        "delay_relative_error": relative_error(
                            proxy["delay_ns"] if proxy else None, point["delay_ns"]
                        ),
                        "power_relative_error": relative_error(
                            proxy["power_mw"] if proxy else None, point["power_mw"]
                        ),
                    }
                )
    if rows:
        path = run_dir / "finalists.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    atomic_write_json(run_dir / "finalists.json", rows)


def export_reports(state: State, run_dir: Path, pipelines: list[str]) -> None:
    events = [dict(row) for row in state.db.execute("SELECT * FROM timing_events ORDER BY event_id")]
    if events:
        with (run_dir / "timings.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(events[0]))
            writer.writeheader()
            writer.writerows(events)
    summary = {"pipelines": {}, "shared_wall_seconds": 0.0, "active_wall_seconds": 0.0}
    for event in events:
        wall = float(event["wall_seconds"] or 0)
        if event["pipeline"] is None:
            summary["shared_wall_seconds"] += wall
    finalist_rows = (
        json.loads((run_dir / "finalists.json").read_text())
        if (run_dir / "finalists.json").is_file()
        else []
    )
    for pipeline in pipelines:
        nodes = state.db.execute(
            "SELECT COUNT(*),SUM(score IS NOT NULL),MAX(score) FROM nodes_v2 WHERE pipeline=?",
            (pipeline,),
        ).fetchone()
        search_wall = state.db.execute(
            """SELECT COALESCE(SUM(wall_seconds),0) FROM timing_events
               WHERE pipeline=? AND stage='pipeline_search'""",
            (pipeline,),
        ).fetchone()[0]
        startup_wall = state.db.execute(
            """SELECT COALESCE(SUM(wall_seconds),0) FROM timing_events
               WHERE pipeline=? AND stage='model_checkpoint_load'""",
            (pipeline,),
        ).fetchone()[0]
        golden_rows = [
            row for row in finalist_rows
            if row["source_pipeline"] == pipeline and row["golden_score"] is not None
        ]
        best_golden = max(golden_rows, key=lambda row: row["golden_score"]) if golden_rows else None
        ratio = best_golden["baseline_over_candidate_ppa"] if best_golden else None
        summary["pipelines"][pipeline] = {
            "nodes": nodes[0], "valid_nodes": nodes[1], "best_proxy_score": nodes[2],
            "search_wall_seconds": search_wall,
            "model_startup_wall_seconds": startup_wall,
            "cold_search_wall_seconds": search_wall + startup_wall,
            "best_golden_score": best_golden["golden_score"] if best_golden else None,
            "best_golden_period_ns": best_golden["period_ns"] if best_golden else None,
            "baseline_over_candidate_ppa": ratio,
            "golden_ppa_reduction_percent": (
                100.0 * (1.0 - 1.0 / ratio) if ratio not in (None, 0) else None
            ),
        }
    dc_wall = summary["pipelines"].get("dc_pt", {}).get("cold_search_wall_seconds")
    if dc_wall:
        for data in summary["pipelines"].values():
            wall = data["cold_search_wall_seconds"]
            data["speedup_vs_dc_pt"] = dc_wall / wall if wall else None
    sessions = [dict(row) for row in state.db.execute("SELECT * FROM sessions ORDER BY session_id")]
    summary["sessions"] = sessions
    report_time = dt.datetime.now(dt.timezone.utc)
    summary["active_wall_seconds"] = sum(
        (
            (
                dt.datetime.fromisoformat(row["completed_at"])
                if row["completed_at"] is not None
                else report_time
            )
            - dt.datetime.fromisoformat(row["started_at"])
        ).total_seconds()
        for row in sessions
    )
    atomic_write_json(run_dir / "summary.json", summary)
    with (run_dir / "pipeline_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "pipeline", "nodes", "valid_nodes", "best_proxy_score",
            "search_wall_seconds", "model_startup_wall_seconds",
            "cold_search_wall_seconds", "speedup_vs_dc_pt",
            "best_golden_score", "best_golden_period_ns",
            "baseline_over_candidate_ppa", "golden_ppa_reduction_percent",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for pipeline, data in summary["pipelines"].items():
            writer.writerow({"pipeline": pipeline, **data})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["Q3_HC"], default="Q3_HC")
    parser.add_argument("--pipelines", type=parse_pipelines, required=True)
    parser.add_argument("--periods", type=parse_periods, default=list(PERIODS))
    parser.add_argument("--ideas", type=int, default=12)
    parser.add_argument("--nodes-per-chain", type=int, default=5)
    parser.add_argument("--finalists", type=int, default=3)
    parser.add_argument("--score", choices=["ppa"], default="ppa")
    parser.add_argument("--golden-power", choices=["both"], default="both")
    parser.add_argument("--shared-roots", action="store_true")
    parser.add_argument("--vcs-period-policy", choices=["all"], default="all")
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--raw-checkpoint", type=Path, default=RAW_CHECKPOINT)
    parser.add_argument("--yosys-checkpoint", type=Path, default=YOSYS_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def validate_files(args: argparse.Namespace) -> None:
    pdk = pdk_files()
    # Every run performs DC/PT golden comparison and VCD power validation.
    required = [DC, PT, pdk['lib'], pdk['db'], CELL_MODELS]
    if 'trained_model' in args.pipelines:
        required.append(args.raw_checkpoint)
    if 'yosys_trained_model' in args.pipelines:
        required.append(args.yosys_checkpoint)
    if set(args.pipelines) & {'yosys_openroad', 'yosys_trained_model'}:
        required.extend([YOSYS, MAP_ADDERS, MAP_LATCHES,
                         REV_PDK / 'Nangate45_typ.lib',
                         REV_ROOT / 'scripts/ref/ref.abc.script'])
    if 'yosys_openroad' in args.pipelines:
        required.extend([OPENROAD, pdk['tech_lef'], pdk['cell_lef'], REV_PDK / 'Nangate45.tracks'])
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise ExperimentError("missing four-pipeline dependencies:\n" + "\n".join(missing))


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.ideas < 1 or args.nodes_per_chain < 1 or args.finalists < 1:
        raise ExperimentError("ideas, nodes-per-chain, and finalists must be positive")
    validate_files(args)
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    state = State(run_dir / "state_v2.sqlite3")
    status = "failed"
    evaluators: list[PipelineEvaluator] = []
    try:
        config = {
            "schema_version": 2, "task": args.task, "pipelines": args.pipelines,
            "periods": args.periods, "ideas": args.ideas,
            "nodes_per_chain": args.nodes_per_chain, "finalists": args.finalists,
            "score": args.score, "golden_power": args.golden_power,
            "shared_roots": args.shared_roots, "vcs_period_policy": args.vcs_period_policy,
            "model": args.model, "reasoning_effort": args.reasoning_effort,
            "raw_checkpoint": str(args.raw_checkpoint.resolve()),
            "yosys_checkpoint": str(args.yosys_checkpoint.resolve()),
            "device": args.device,
        }
        previous = state.metadata("config")
        if previous is not None and previous != config:
            raise ExperimentError("run directory contains a different configuration")
        state.metadata("config", config)
        with state.event("preflight"):
            env = load_eda_environment()
            checks = preflight(env, args.model, args.reasoning_effort)
            validate_files(args)
            atomic_write_json(run_dir / "preflight.json", checks)
        print("[preflight] Codex, VCS, four pipelines, and PDK are ready", flush=True)
        if args.preflight_only:
            status = "completed"
            return 0
        task = TASKS[args.task]
        codex = CodexRunner(env, args.model, args.reasoning_effort)
        vcs = FunctionalVCS(state, run_dir, env, args.periods)
        # Golden baseline is always available for same-period comparison.
        baseline_eval = PipelineEvaluator(
            "dc_pt", state, run_dir, env, args.periods, vcs, args.device,
            args.raw_checkpoint, args.yosys_checkpoint,
        )
        baseline_eval.evaluate(task.baseline, -1, -1)
        create_ideas_and_roots(state, run_dir, task, codex, args.ideas)
        for pipeline in args.pipelines:
            evaluator = PipelineEvaluator(
                pipeline, state, run_dir, env, args.periods, vcs, args.device,
                args.raw_checkpoint, args.yosys_checkpoint,
            )
            evaluators.append(evaluator)
            evaluator.evaluate(task.baseline, -1, -1)
            run_pipeline(
                state, run_dir, task, codex, evaluator,
                args.ideas, args.nodes_per_chain,
            )
            evaluator.close()
        with state.event("finalist_selection"):
            finalists = select_finalists(state, args.pipelines, args.finalists)
        with state.event("golden_validation"):
            validate_golden(state, run_dir, env, args.periods, vcs, finalists)
        with state.event("report_generation"):
            export_reports(state, run_dir, args.pipelines)
        # Refresh exports once the report-generation event itself is terminal.
        export_reports(state, run_dir, args.pipelines)
        print(f"[done] {run_dir / 'summary.json'}", flush=True)
        status = "completed"
        return 0
    finally:
        for evaluator in evaluators:
            evaluator.close()
        state.close(status)


if __name__ == "__main__":
    raise SystemExit(main())
