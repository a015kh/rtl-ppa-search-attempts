"""Prepare test JSONL from supplied, completed DC/PT measurements, offline.

Named policies preserve recovered local result conversion/filtering behavior.
No synthesis, benchmark download, prediction, or train/test split is performed.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile
from typing import Any, Iterator

from .dataset_row_helpers import ResynthesisError, format_cell_reports, format_timing_report


class TestPreparationError(RuntimeError):
    """Invalid configuration or supplied measurement input."""


class RejectedPoint(Exception):
    """A measured point rejected by a named historical filtering policy."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


POLICIES = {"legacy_result_json", "dcpt_inputs_reports", "resynthesis_success"}
FLOAT_REGEX = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
KEY_MAP_V3 = {
    "area": "area", "Total Area": "area_value", "Delay": "delay_value",
    "delay": "delay", "Total Power": "total_power_value",
    "Static Power": "static_power_value", "static_power": "static_power",
    "switching_power": "switching_power", "TNS": "tns", "WNS": "wns",
    "SDC": "SDC", "Clock": "clock", "Has Clock": "has_clock",
    "synth": "synth", "synthesis_time": "synthesis_time",
}

# Recovered local report/conversion functions; no commercial tool execution.

def extract_area(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            match = re.search(r'Total cell area:\s*([\d.]+)', line)
            if match:
                return float(match.group(1))
    return None


def extract_power(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        if len(lines) >= 2:
            line = lines[-2].strip()
            matches = re.findall(r'([\d.]+(?:e[+-]?\d+)?)', line)
            if matches:
                try:
                    return float(matches[-1])
                except ValueError:
                    pass
    return None


def extract_qor(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            match = re.search(
                r'Design\s+WNS:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+TNS:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)',
                line
            )
            if match:
                wns = float(match.group(1))
                tns = float(match.group(2))
                return wns, tns
    return None, None


def extract_delay(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            match = re.search(r'data arrival time\s+([+-]?\d+(?:\.\d+)?)', line)
            if match:
                return float(match.group(1))
    return None


def extract_clock(sdc_content: str):
    if "# Clock" not in sdc_content:
        m = re.search(
            r'set\s+cycle_vclk\s+([0-9]+(?:\.[0-9]+)?)',
            sdc_content,
            re.IGNORECASE
        )
        if not m:
            return (False, 1.0)
        return (False, float(m.group(1)))

    m = re.search(
        r'set\s+cycle_[A-Za-z0-9_]+\s+([0-9]+(?:\.[0-9]+)?)',
        sdc_content,
        re.IGNORECASE
    )
    if not m:
        return (False, None)

    period = float(m.group(1))
    return (True, period)


def extract_cell_usage(file_path):
    cell_data = defaultdict(lambda: {"count": 0, "area": 0.0})

    in_table = False
    with open(file_path, 'r') as f:
        for line in f:
            line = line.rstrip()
            
            # Detect start of table
            if line.startswith("Cell") and "Reference" in line and "Area" in line:
                in_table = True
                continue
            if not in_table:
                continue

            # End of table
            if line.startswith("Total"):
                break

            # Parse cell rows (U8, U9, ... or count_reg[0], etc.)
            match = re.match(r"\s*\S+\s+(\S+)\s+(?:\S+\s+)?([\d.]+)", line)

            if match:
                cell_name = match.group(1)  # Reference
                
                try:
                    area = float(match.group(2))
                    
                except ValueError:
                    continue
                cell_data[cell_name]["count"] += 1
                cell_data[cell_name]["area"] = area

    return cell_data


def format_cell_summary(cell_data):
    # Collect summary and equation parts
    area_summaries = []
    equation_parts = []
    total_area = 0.0

    for cell_name, info in cell_data.items():
        count = info["count"]
        area = info["area"]
        subtotal = count * area
        total_area += subtotal

        # First part: "Area of X is Y"
        area_summaries.append(f"Area of {cell_name} is {area:.2f}")

        # Second part: "count*area"
        equation_parts.append(f"{count}*{area:.2f}")

    # Join summaries into one sentence
    area_summary_str = ". ".join(area_summaries) + "."

    # Build equation string
    equation_str = " + ".join(equation_parts)

    # Final area string
    area_str = (
        f"{area_summary_str}  \n"
        f"In total, we can compute {equation_str}  = {total_area:.2f}\n"
        f"Thus, the total area is {total_area:.2f}"
    )

    # Synthesis string
    synth_str = "After synthesis, this design has " + ", ".join(
        f"{info['count']} {cell}" for cell, info in cell_data.items()
    ) + "."

    return synth_str, area_str


def extract_cell_delay(report_file):
    num_re = re.compile(r'^[-+]?(?:\d+(?:\.\d+)?|\.\d+)$')

    started = False
    skip_next = False
    tokens = []
    total_delay = None

    with open(report_file, 'r', encoding='utf-8', errors='ignore') as f:
        for raw in f:
            line = raw.rstrip('\n')

            if not started:
                if line.strip().lower().startswith("point"):
                    started = True
                    skip_next = True 
                continue

            if skip_next:
                skip_next = False
                continue  

            if 'data arrival time' in line.lower():
                m = re.search(r'([-+]?\d+(?:\.\d+)?)\s*$', line)
                if m:
                    total_delay = float(m.group(1))
                break

            tokens.extend(line.split())

    critical_path = []
    label_parts = []
    incr = None
    state = 'label'

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        is_num = bool(num_re.match(tok))
        is_rf  = tok.lower() in ('r', 'f')

        if state == 'label':
            if is_num:
                incr = float(tok)   
                state = 'need_path'
            else:
                label_parts.append(tok)
            i += 1
            continue

        if state == 'need_path':
            if is_num:
                
                state = 'maybe_rf'
            i += 1
            continue

        if state == 'maybe_rf':
            if is_rf:
                i += 1 

            label = ' '.join(label_parts).strip()
            if label and incr is not None:
                critical_path.append((label, incr))

            label_parts = []
            incr = None
            state = 'label'
            continue

    if state in ('need_path', 'maybe_rf') and label_parts and incr is not None:
        label = ' '.join(label_parts).strip()
        if label:
            critical_path.append((label, incr))

    gates_desc = []
    delays = []
    delay_lines = []
    for gate_text, incr_val in critical_path:
        gates_desc.append(gate_text)
        delays.append(f"{incr_val:.2f}")
        delay_lines.append(f"Delay of {gate_text} is {incr_val:.2f}")

    gates_str = ', '.join(gates_desc)
    delays_sum_str = ' + '.join(delays)
    total_incr = sum(float(d) for d in delays)

    #total_delay_str = f"{total_delay:.2f}" if total_delay is not None else "N/A"

    output = (
        f'The critical path goes through the following gates: {gates_str}. '
        f'{", ".join(delay_lines)}. In total, we can compute {delays_sum_str} = {total_incr:.2f}\n'
        f'Thus, the total delay is {total_incr:.2f}'
    )
    return output


def extract_detail_power(file_path):
    with open(file_path, "r") as f:
        report_text = f.read()

    # --- Extract total dynamic and leakage power from summary ---
    dynamic_match = re.search(r"Total Dynamic Power\s*=\s*([\d\.Ee+-]+)\s*(\w+)", report_text)
    leakage_match = re.search(r"Cell Leakage Power\s*=\s*([\d\.Ee+-]+)\s*(\w+)", report_text)

    total_dynamic_val, total_dynamic_unit = float(dynamic_match.group(1)), dynamic_match.group(2)
    total_leakage_val, total_leakage_unit = float(leakage_match.group(1)), leakage_match.group(2)

    # --- Extract power group table ---
    group_pattern = re.compile(
        r"(\w+)\s+([\d\.Ee+-]+)\s+([\d\.Ee+-]+)\s+([\d\.Ee+-]+)\s+([\d\.Ee+-]+)",
        re.MULTILINE
    )

    groups = {}
    for match in group_pattern.finditer(report_text):
        group = match.group(1)
        internal = float(match.group(2))
        switching = float(match.group(3))
        leakage = float(match.group(4))
        total = float(match.group(5))
        groups[group] = {
            "internal": internal,
            "switching": switching,
            "leakage": leakage,
            "total": total
        }

    # --- Normalize units ---
    unit_scale = {
        "W": 1.0,
        "mW": 1e-3,
        "uW": 1e-6,
        "nW": 1e-9,
        "pW": 1e-12,
    }
    W_to_mW = 1e3  # final output in mW

    # Convert totals to W
    total_dynamic_W = total_dynamic_val * unit_scale[total_dynamic_unit]
    total_leakage_W = total_leakage_val * unit_scale[total_leakage_unit]

    # Convert group values to W
    for g, vals in groups.items():
        vals["internal_W"] = vals["internal"] * unit_scale.get("mW", 1.0)
        vals["switching_W"] = vals["switching"] * unit_scale.get("mW", 1.0)
        vals["leakage_W"] = vals["leakage"] * unit_scale.get("pW", 1.0)

    # --- Construct explanation strings ---
    # Static power
    static_parts = []
    static_terms = []
    static_sum_W = 0
    for g, vals in groups.items():
        if vals["leakage_W"] > 0:
            mW_val = vals["leakage_W"] * W_to_mW
            static_parts.append(f"Leakage power of {g} is {mW_val:.4e} mW")
            static_terms.append(f"{mW_val:.4e}")
            static_sum_W += vals["leakage_W"]
    static_expr = " + ".join(static_terms)
    static_msg = (
        ", ".join(static_parts)
        + f". In total, we can compute {static_expr} = {static_sum_W*W_to_mW:.4e} mW.\n"
        + f"Thus, the total static power is {static_sum_W*W_to_mW:.4e} mW."
    )

    # Dynamic power
    dyn_parts = []
    dyn_terms = []
    dyn_sum_W = 0
    for g, vals in groups.items():
        if vals["internal_W"] > 0 or vals["switching_W"] > 0:
            internal_mW = vals["internal_W"] * W_to_mW
            switching_mW = vals["switching_W"] * W_to_mW
            dyn_parts.append(
                f"The internal power of {g} is {internal_mW:.4e} mW, "
                f"and the switching power of {g} is {switching_mW:.4e} mW"
            )
            dyn_terms.append(f"{internal_mW:.4e}")
            dyn_terms.append(f"{switching_mW:.4e}")
            dyn_sum_W += vals["internal_W"] + vals["switching_W"]
    dyn_expr = " + ".join(dyn_terms)
    dyn_msg = (
        ". ".join(dyn_parts)
        + f". In total, we can compute {dyn_expr} = {dyn_sum_W*W_to_mW:.4e} mW.\n"
        + f"Thus, the total dynamic power is {dyn_sum_W*W_to_mW:.4e} mW."
    )

    return static_msg, dyn_msg


def extract_primetime_power(file_path):
    """Parse PrimeTime averaged power report and return total power (mW),
    static and dynamic explanation strings.
    """
    with open(file_path, 'r') as f:
        text = f.read()

    # Patterns in PrimeTime averaged power report
    total_match = re.search(r"Total Power\s*=\s*([\d\.Ee+-]+)", text)
    internal_match = re.search(r"Cell Internal Power\s*=\s*([\d\.Ee+-]+)", text)
    switching_match = re.search(r"Net Switching Power\s*=\s*([\d\.Ee+-]+)", text)
    leakage_match = re.search(r"Cell Leakage Power\s*=\s*([\d\.Ee+-]+)", text)

    unit_scale = {
        "W": 1.0,
        "mW": 1e-3,
        "uW": 1e-6,
        "nW": 1e-9,
        "pW": 1e-12,
    }

    unit_match = re.search(r"Power\s+Units?\s*:\s*(\w+)", text, re.IGNORECASE)
    unit = unit_match.group(1) if unit_match else "W"
    unit_factor = unit_scale.get(unit, 1.0)

    def to_mW(val_str):
        try:
            return float(val_str) * unit_factor * 1e3
        except Exception:
            return None

    total_mW = to_mW(total_match.group(1)) if total_match else None
    internal_mW = to_mW(internal_match.group(1)) if internal_match else None
    switching_mW = to_mW(switching_match.group(1)) if switching_match else None
    leakage_mW = to_mW(leakage_match.group(1)) if leakage_match else None

    # Build explanation strings similar to extract_detail_power
    if leakage_mW is None:
        static_msg = "Leakage power is N/A."
    else:
        static_msg = (
            f"Leakage power is {leakage_mW:.4e} mW. "
            f"Thus, the total static power is {leakage_mW:.4e} mW."
        )

    internal_str = "N/A" if internal_mW is None else f"{internal_mW:.4e}"
    switching_str = "N/A" if switching_mW is None else f"{switching_mW:.4e}"
    if internal_mW is None or switching_mW is None:
        dyn_total_str = "N/A"
    else:
        dyn_total_str = f"{(internal_mW + switching_mW):.4e}"

    dyn_msg = (
        f"The internal power is {internal_str} mW, and the switching power is {switching_str} mW. "
        f"In total, the dynamic power is {dyn_total_str} mW."
    )

    return total_mW, static_msg, dyn_msg


def extract_start_time(file_path):
    """Extract start time from command.log"""
    with open(file_path, 'r') as f:
        for line in f:
            match = re.search(r'#@ # Date:\s+(.*)', line)
            if match:
                return datetime.strptime(match.group(1).strip(), "%a %b %d %H:%M:%S %Y")
    return None


def extract_end_time(file_path):
    """Extract end time from aaa_area.log"""
    with open(file_path, 'r') as f:
        for line in f:
            match = re.search(r'Date\s*:\s+(.*)', line)
            if match:
                return datetime.strptime(match.group(1).strip(), "%a %b %d %H:%M:%S %Y")
    return None


def dc_top_module_from_tcl(tcl_path: Path) -> str | None:
    try:
        text = tcl_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None

    for pattern in (
        r"^\s*current_design\s+(\S+)",
        r"^\s*elaborate\s+(\S+)",
    ):
        match = re.search(pattern, text, flags=re.M)
        if match:
            return match.group(1).strip('"{}')
    return None


def parse_number(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text, re.I | re.M)
    return float(match.group(1)) if match else None


def parse_dc_result(directory: Path, top: str) -> dict[str, Any]:
    report = directory / "dc" / "report"
    area_text = (report / f"{top}_area.log").read_text(encoding="utf-8", errors="ignore")
    timing_text = (report / f"{top}_timing.log").read_text(encoding="utf-8", errors="ignore")
    qor_text = (report / f"{top}_qor.log").read_text(encoding="utf-8", errors="ignore")
    cell_path = report / f"{top}_cell.log"
    power_text = (report / f"{top}_power.log").read_text(encoding="utf-8", errors="ignore")
    area = parse_number(r"Total cell area:\s*([\d.eE+-]+)", area_text)
    delay = parse_number(r"data arrival time\s+([\d.eE+-]+)", timing_text)
    if delay is None and re.search(r"\bNo paths\.\s*$", timing_text, re.I | re.M):
        delay = 0.0
    qor = re.search(r"Design\s+WNS:\s*([\d.eE+-]+)\s+TNS:\s*([\d.eE+-]+)", qor_text)
    if area is None or delay is None or qor is None:
        raise ResynthesisError(f"Incomplete DC report for {top}: {report}")
    wns, tns = float(qor.group(1)), float(qor.group(2))
    cells = cell_path.read_text(encoding="utf-8", errors="ignore") if cell_path.is_file() else ""
    synth_summary, area_summary = format_cell_reports(cells)
    delay_summary = format_timing_report(timing_text)
    total_power = parse_number(r"Total Dynamic Power\s*=\s*([\d.eE+-]+)", power_text)
    return {
        "area_value": area,
        "delay_value": delay,
        "wns": wns,
        "tns": tns,
        "synth": synth_summary,
        "area": area_summary,
        "delay": delay_summary,
        "dc_total_power": total_power,
    }


def parse_pt_power(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    total = parse_number(r"Total Power\s*=\s*([\d.eE+-]+)", text)
    internal = parse_number(r"Cell Internal Power\s*=\s*([\d.eE+-]+)", text)
    switching = parse_number(r"Net Switching Power\s*=\s*([\d.eE+-]+)", text)
    leakage = parse_number(r"Cell Leakage Power\s*=\s*([\d.eE+-]+)", text)
    if total is None:
        raise ResynthesisError(f"Cannot parse PT power report: {path}")
    unit_match = re.search(r"Power\s+Units?\s*:\s*(\w+)", text, re.I)
    unit = unit_match.group(1) if unit_match else "W"
    to_mw = {"W": 1000.0, "mW": 1.0, "uW": 1e-3, "nW": 1e-6, "pW": 1e-9}.get(unit, 1000.0)
    total_mw = total * to_mw
    leakage_mw = leakage * to_mw if leakage is not None else None
    switching_mw = ((internal or 0.0) + (switching or 0.0)) * to_mw
    return {
        "total_power_value": total_mw,
        "static_power_value": leakage_mw,
        "switching_power_value": switching_mw,
        "static_power": f"Total static power is {leakage_mw:.4e} mW." if leakage_mw is not None else "Leakage power is N/A.",
        "switching_power": f"Total dynamic power is {switching_mw:.4e} mW.",
    }


def extract_data_v3(verilog_data):
    rtl = verilog_data["RTL"]
    syn = verilog_data["Syn"]
    all_data = []
    for result in syn:
        static_power = float(re.findall(FLOAT_REGEX, result["static_power"])[-1])
        switching_power = float(re.findall(FLOAT_REGEX, result["switching_power"])[-1])
        new_result = {}
        for k, v in result.items():
            if k in KEY_MAP_V3:
                new_result[KEY_MAP_V3[k]] = v
        new_result["design_name"] = verilog_data["Name"]
        new_result["rtl"] = rtl
        new_result["static_power_value"] = static_power
        new_result["switching_power_value"] = switching_power
        all_data.append(new_result)
        # {
        #     "design_name": verilog_data["Name"],
        #     "rtl": rtl,
        #     "area": result["area"],
        #     "area_value": result["Total Area"],
        #     "delay_value": result["Delay"],
        #     "delay": result["delay"],
        #     "switching_power_value": result["Total Power"],
        #     "static_power_value": static_power,
        #     "static_power": result["static_power"],
        #     "switching_power": result["switching_power"],
        #     "tns": result["TNS"],
        #     "wns": result["WNS"],
        #     "SDC": result["SDC"],
        #     "clock": result["Clock"],
        #     "has_clock": result["Has Clock"],
        #     "synth": result["synth"],
        #     "synthesis_time": result["synthesis_time"],
        # }
    return all_data


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TestPreparationError(f"Cannot read JSON input {path}: {exc}") from exc


def _path(base: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise TestPreparationError(f"{label} must be an explicit nonempty path")
    candidate = Path(value).expanduser()
    return (base / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TestPreparationError(f"{label} must be a JSON object")
    return value


def _fields(value: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise TestPreparationError(f"Unknown {label} fields: {sorted(unknown)}")


def load_config(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    raw = _object(_read_json(path), "configuration")
    _fields(raw, {"output_root", "shards"}, "configuration")
    output = _path(path.parent, raw.get("output_root"), "output_root")
    shards = raw.get("shards")
    if not isinstance(shards, list) or not shards:
        raise TestPreparationError("shards must be a nonempty list")
    parsed = []
    names = set()
    for index, entry in enumerate(shards):
        entry = _object(entry, f"shards[{index}]")
        policy = entry.get("policy")
        if policy not in POLICIES:
            raise TestPreparationError(f"Unsupported policy: {policy!r}")
        allowed = {"name", "policy"} | {
            "legacy_result_json": {"inputs"},
            "dcpt_inputs_reports": {"jobs", "power_source"},
            "resynthesis_success": {"input", "manifest", "work_root"},
        }[policy]
        _fields(entry, allowed, f"{policy} shard")
        name = entry.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.jsonl", name):
            raise TestPreparationError("shard name must be a plain .jsonl filename")
        if name in names:
            raise TestPreparationError(f"Duplicate shard name: {name}")
        names.add(name)
        result = {"name": name, "policy": policy}
        if policy == "legacy_result_json":
            inputs = entry.get("inputs")
            if not isinstance(inputs, list) or not inputs:
                raise TestPreparationError("legacy_result_json inputs must be a nonempty list")
            result["inputs"] = [_path(path.parent, p, "inputs entry") for p in inputs]
        elif policy == "dcpt_inputs_reports":
            result["jobs"] = _path(path.parent, entry.get("jobs"), "jobs")
            if entry.get("power_source") not in {"primetime", "mixed"}:
                raise TestPreparationError("power_source must explicitly be primetime or mixed")
            result["power_source"] = entry["power_source"]
        else:
            for key in ("input", "manifest", "work_root"):
                result[key] = _path(path.parent, entry.get(key), key)
        parsed.append(result)
    return {"output_root": output, "shards": parsed, "config_path": path}


def _overlap(a: Path, b: Path) -> bool:
    return a == b or a in b.parents or b in a.parents


def _jobs(spec: dict[str, Any]) -> list[dict[str, Any]]:
    raw = _read_json(spec["jobs"])
    if not isinstance(raw, list):
        raise TestPreparationError("jobs JSON must contain a list")
    parsed = []
    required = {"name", "rtl", "top_module", "dc_tcl", "sdc", "dc_report_dir", "pt_report_dir"}
    for index, item in enumerate(raw):
        item = _object(item, f"jobs[{index}]")
        _fields(item, required | {"command_log"}, "report job")
        missing = required - set(item)
        if missing:
            raise TestPreparationError(f"jobs[{index}] missing fields: {sorted(missing)}")
        job = dict(item)
        for key in ("name", "top_module"):
            if not isinstance(job[key], str) or not job[key].strip():
                raise TestPreparationError(f"jobs[{index}] requires {key}")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*", job["top_module"]):
            raise TestPreparationError("top_module must be a plain Verilog identifier")
        for key in ("rtl", "dc_tcl", "sdc", "dc_report_dir", "pt_report_dir", "command_log"):
            if key in job:
                job[key] = _path(spec["jobs"].parent, job[key], key)
        parsed.append(job)
    return parsed


def preflight(config: dict[str, Any]) -> dict[str, Any]:
    output = config["output_root"]
    if output.exists():
        raise TestPreparationError("output_root already exists; select a new directory")
    # Keep generated payloads outside this package/source checkout.
    source_root = Path(__file__).resolve().parents[2]
    if source_root == output or source_root in output.parents:
        raise TestPreparationError("output_root must be outside the package/source checkout")
    protected = [config["config_path"]]
    report_jobs = 0
    for spec in config["shards"]:
        if spec["policy"] == "legacy_result_json":
            files = spec["inputs"]
        elif spec["policy"] == "dcpt_inputs_reports":
            files = [spec["jobs"]]
            for job in _jobs(spec):
                report_jobs += 1
                if not job["rtl"].is_file():
                    raise TestPreparationError(f"Required RTL input not found: {job['rtl']}")
                protected.extend(job[key] for key in ("rtl", "dc_tcl", "sdc", "dc_report_dir", "pt_report_dir", "command_log") if key in job)
        else:
            files = [spec["input"], spec["manifest"]]
            if not spec["work_root"].is_dir():
                raise TestPreparationError(f"work_root not found: {spec['work_root']}")
            protected.append(spec["work_root"])
            _check_manifest(spec["manifest"])
        for candidate in files:
            if not candidate.is_file():
                raise TestPreparationError(f"Required input not found: {candidate}")
            protected.append(candidate)
    if any(_overlap(output, candidate) for candidate in protected):
        raise TestPreparationError("output_root overlaps an input or report location")
    return {"status": "ready", "shards": len(config["shards"]), "report_jobs": report_jobs, "offline": True}


def _connect_manifest(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise TestPreparationError(f"Manifest not found: {path}")
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _check_manifest(path: Path) -> None:
    with closing(_connect_manifest(path)) as connection:
        required = {
            "source_rows": {"shard", "line_number", "job_id"},
            "jobs": {"job_id", "dc_status", "pt_status", "top_module", "sdc", "rtl", "design_name", "dc_elapsed"},
        }
        for table, names in required.items():
            columns = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
            if not names.issubset(columns):
                raise TestPreparationError(f"Manifest {table} missing required columns: {sorted(names - columns)}")


def _legacy_rows(spec: dict[str, Any], audit: dict[str, Any]) -> Iterator[dict[str, Any]]:
    for path in spec["inputs"]:
        values = _read_json(path)
        if not isinstance(values, list):
            raise TestPreparationError(f"Result JSON must be a list: {path}")
        for entry in values:
            _object(entry, "result design")
            if not isinstance(entry.get("Name"), str) or not isinstance(entry.get("RTL"), str) or not isinstance(entry.get("Syn"), list):
                raise TestPreparationError("Result designs require Name, RTL, and a Syn list")
            audit["input_designs"] += 1
            audit["input_points"] += len(entry["Syn"])
            try:
                rows = extract_data_v3(entry)
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise TestPreparationError("Invalid historical result/explanation fields; conversion was not completed") from exc
            yield from rows


def _collect_report_point(job: dict[str, Any], power_source: str) -> dict[str, Any]:
    report = job["dc_report_dir"]
    candidates = []
    for candidate in (dc_top_module_from_tcl(job["dc_tcl"]), job["top_module"]):
        if candidate and candidate not in candidates and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*", candidate):
            candidates.append(candidate)
    stem = next((candidate for candidate in candidates if (report / (candidate + "_syn.v")).is_file()), None)
    if stem is None:
        for path in sorted(report.glob("*_syn.v")):
            candidate = path.name[:-len("_syn.v")]
            if candidate not in candidates:
                candidates.append(candidate)
        stem = next((candidate for candidate in candidates if all((report / (candidate + suffix)).is_file() for suffix in ("_area.log", "_qor.log", "_timing.log"))), None)
    if stem is None:
        raise RejectedPoint("missing_report_stem")
    area_file = report / (stem + "_area.log")
    qor_file = report / (stem + "_qor.log")
    timing_file = report / (stem + "_timing.log")
    required = [job["sdc"], area_file, qor_file, timing_file, report / (stem + "_syn.v")]
    if not all(path.is_file() for path in required):
        raise RejectedPoint("missing_required_report_or_netlist")
    area = extract_area(str(area_file))
    if area is None:
        raise RejectedPoint("missing_area")
    if area == 0:
        raise RejectedPoint("zero_area")
    pt_file = job["pt_report_dir"] / (stem + ".report_power.rpt")
    if not pt_file.is_file():
        pt_file = job["pt_report_dir"] / (job["top_module"] + ".report_power.rpt")
    dc_power = report / (stem + "_power.log")
    if pt_file.is_file():
        power, static_power, dynamic_power = extract_primetime_power(str(pt_file))
        source = "primetime"
    elif power_source == "mixed" and dc_power.is_file():
        power = extract_power(str(dc_power))
        static_power, dynamic_power = extract_detail_power(str(dc_power))
        source = "design_compiler"
    else:
        raise RejectedPoint("missing_power_report")
    wns, tns = extract_qor(str(qor_file))
    delay = extract_delay(str(timing_file))
    if None in (power, wns, tns) or delay is None:
        raise RejectedPoint("missing_required_numeric_result")
    start = extract_start_time(str(job["command_log"])) if job.get("command_log", Path()).is_file() else None
    end = extract_end_time(str(area_file))
    elapsed = (end - start).total_seconds() if start and end else None
    sdc = job["sdc"].read_text(encoding="utf-8", errors="ignore")
    has_clock, clock = extract_clock(sdc)
    cell_file = report / (stem + "_cell.log")
    cells = extract_cell_usage(str(cell_file)) if cell_file.is_file() else {}
    synth, area_text = format_cell_summary(cells) if cells else ("", "")
    audit_result = {
        "SDC": sdc, "Has Clock": has_clock, "Clock": clock, "Total Area": area,
        "Delay": delay, "WNS": wns, "TNS": tns, "Total Power": power,
        "synth": synth, "area": area_text, "delay": extract_cell_delay(str(timing_file)),
        "static_power": static_power, "switching_power": dynamic_power,
        "synthesis_time": elapsed,
    }
    return {"result": audit_result, "power_source": source}


def _report_rows(spec: dict[str, Any], audit: dict[str, Any]) -> Iterator[dict[str, Any]]:
    for job in _jobs(spec):
        audit["input_designs"] += 1
        audit["input_points"] += 1
        try:
            collected = _collect_report_point(job, spec["power_source"])
            # Exactly the historical explanation-to-label conversion; no netlist output.
            row = extract_data_v3({"Name": job["name"], "RTL": job["rtl"].read_text(encoding="utf-8", errors="ignore"), "Syn": [collected["result"]]})[0]
        except RejectedPoint as exc:
            audit["reasons"][exc.reason] += 1
            continue
        except (OSError, KeyError, IndexError, ValueError, TypeError, AttributeError) as exc:
            # Historical malformed reports could abort; the release exposes these skips.
            audit["reasons"]["report_parse_or_conversion_error"] += 1
            continue
        audit["power_sources"][collected["power_source"]] += 1
        yield row


def _resynthesis_rows(spec: dict[str, Any], audit: dict[str, Any]) -> Iterator[dict[str, Any]]:
    with closing(_connect_manifest(spec["manifest"])) as connection, spec["input"].open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                original = _object(json.loads(line), "source row")
            except (ValueError, TypeError) as exc:
                raise TestPreparationError(f"Invalid source JSONL line {line_number}") from exc
            audit["input_points"] += 1
            mapping = connection.execute(
                "SELECT j.* FROM source_rows r JOIN jobs j ON j.job_id=r.job_id WHERE r.shard=? AND r.line_number=?",
                (spec["input"].name, line_number),
            ).fetchone()
            if mapping is None:
                audit["reasons"]["unmapped_source_row"] += 1
                continue
            if mapping["dc_status"] != "success" or mapping["pt_status"] != "success":
                reason = "dc_not_success" if mapping["dc_status"] != "success" else "pt_not_success"
                audit["reasons"][reason] += 1
                continue
            job_id = str(mapping["job_id"])
            if not re.fullmatch(r"[A-Za-z0-9_-]+", job_id) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*", str(mapping["top_module"])):
                raise TestPreparationError("Unsafe manifest job/module identifier")
            directory = spec["work_root"] / "jobs" / job_id[:2] / job_id
            try:
                result = {**parse_dc_result(directory, mapping["top_module"]), **parse_pt_power(directory / "pt" / "report" / (mapping["top_module"] + ".report_power.rpt")), "synthesis_time": float(mapping["dc_elapsed"])}
                result.pop("dc_total_power", None)
            except (OSError, ResynthesisError, ValueError, TypeError):
                audit["reasons"]["result_collection_error"] += 1
                continue
            # Preserve export_final_jsonls.py's compact no-netlist schema/fallbacks.
            row = {"SDC": mapping["sdc"], "has_clock": original.get("has_clock", True), "clock": original.get("clock", "")}
            for field in ("area_value", "delay_value", "wns", "tns", "total_power_value", "static_power_value", "switching_power_value"):
                row[field] = result.get(field, original.get(field))
            for field in ("synth", "area", "delay", "static_power", "switching_power"):
                row[field] = result.get(field, original.get(field, ""))
            row["synthesis_time"] = result.get("synthesis_time", original.get("synthesis_time"))
            row["design_name"] = mapping["design_name"]
            row["rtl"] = mapping["rtl"]
            yield row


def build_command(config: dict[str, Any]) -> dict[str, Any]:
    preflight(config)
    output = config["output_root"]
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging.", dir=output.parent))
    audits = {}
    try:
        for spec in config["shards"]:
            audit = {"policy": spec["policy"], "input_designs": 0, "input_points": 0, "accepted": 0, "reasons": Counter(), "power_sources": Counter()}
            processor = {"legacy_result_json": _legacy_rows, "dcpt_inputs_reports": _report_rows, "resynthesis_success": _resynthesis_rows}[spec["policy"]]
            destination = staging / spec["name"]
            digest = hashlib.sha256()
            with destination.open("wb") as target:
                for row in processor(spec, audit):
                    data = (json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")
                    target.write(data)
                    digest.update(data)
                    audit["accepted"] += 1
            audit["skipped"] = sum(audit["reasons"].values())
            audit["reasons"] = dict(audit["reasons"])
            audit["power_sources"] = dict(audit["power_sources"])
            if spec["policy"] == "resynthesis_success":
                audit["input_designs"] = None  # Source rows, not independently counted designs.
            audit["output_sha256"] = digest.hexdigest()
            audits[spec["name"]] = audit
        summary = {"format": "test-preparation-audit-v1", "offline": True, "shards": audits}
        (staging / "audit.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        # Reserve exclusively and publish without replacing an existing result.
        output.mkdir()
        if os.name == "nt":
            output.rmdir()
            os.rename(staging, output)
        else:
            os.replace(staging, output)
        return summary
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for operation in ("preflight", "build"):
        command = sub.add_parser(operation)
        command.add_argument("--config", required=True, type=Path, help="Explicit JSON input/output configuration.")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        result = preflight(config) if args.command == "preflight" else build_command(config)
    except (TestPreparationError, OSError, sqlite3.Error, ValueError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
