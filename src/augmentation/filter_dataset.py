from __future__ import annotations

import argparse
import csv
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .dataset_io import DatasetRecord, load_dataset_records, sanitize_id, save_records


DEFAULT_GATE_COUNT_CUTOFF = 50
DEFAULT_YOSYS_LOG_ROOT = Path("outputs/augmentation/step0_yosys")
DEFAULT_STAT_ROOT = Path("outputs/augmention/step0_yosys_stat")
MEMORY_REGEX = re.compile(
    r"(?ix)"
    r"\b(?:rom|ram|sram|dram|bram|cache|memory|mem|fifo|buffer)\b"
    r"|"
    r"\$readmem[hb]\b"
    r"|"
    r"\b(?:reg|logic)\s*(?:signed\s*)?(?:\[[^\]]+\]\s*)?\w+\s*\[[^\]]+\]"
    r"|"
    r"\.(?:mem|hex|bin|coe|mif)\b"
)
GATE_COUNT_PATTERNS = (
    re.compile(r"(?im)^\s*Number\s+of\s+cells:\s*([0-9][0-9_,]*)\s*$"),
    re.compile(r"(?im)^\s*Chip\s+area\s+for\s+.*?:\s*[-+0-9.eE]+\s*$"),
)
_WORKER_STAT_ROOT: Path | None = None
_WORKER_YOSYS_LOG_ROOT: Path | None = None
_WORKER_MEMORY_REGEX: re.Pattern[str] | None = None
_WORKER_GATE_COUNT_CUTOFF = DEFAULT_GATE_COUNT_CUTOFF
_WORKER_DROP_MISSING_GATE_COUNT = False
_WORKER_STAT_INDEX: dict[str, tuple[str, ...]] = {}
_WORKER_YOSYS_LOG_INDEX: dict[str, tuple[str, ...]] = {}
_WORKER_PPA_INDEX: dict[str, tuple[str, ...]] = {}


@dataclass
class FilterDecision:
    keep: bool
    reasons: list[str]
    gate_count: int | None
    num_memories: int | None
    num_memory_bits: int | None
    stat_json_path: str | None
    yosys_log_path: str | None
    ppa_path: str | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Filter augmentation datasets by removing memory-like modules and "
            "small designs below the REvolution gate-count cutoff."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--yosys-log-root",
        type=Path,
        default=DEFAULT_YOSYS_LOG_ROOT,
        help=(
            "Root directory containing step0 Yosys logs. Defaults to "
            "outputs/augmentation/step0_yosys."
        ),
    )
    parser.add_argument(
        "--stat-root",
        type=Path,
        default=DEFAULT_STAT_ROOT,
        help=(
            "Root directory containing step0 Yosys stat.json files. Defaults to "
            "outputs/augmention/step0_yosys_stat."
        ),
    )
    parser.add_argument(
        "--gate-count-cutoff",
        type=int,
        default=DEFAULT_GATE_COUNT_CUTOFF,
        help=(
            "Drop rows whose parsed Yosys gate/cell count is smaller than this value. "
            "Use a negative value to disable the gate-count filter."
        ),
    )
    parser.add_argument(
        "--memory-regex",
        default=MEMORY_REGEX.pattern,
        help="Case-insensitive regex used to identify memory-like modules.",
    )
    parser.add_argument(
        "--drop-missing-gate-count",
        action="store_true",
        help="Drop rows when no Yosys log or gate count can be found.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional JSON report path. Defaults to <output>/filter_report.json.",
    )
    parser.add_argument(
        "--removed-csv",
        type=Path,
        help="Optional CSV path for removed rows. Defaults to <output>/removed_rows.csv.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help=(
            "Number of worker processes used to filter rows. Use 1 to run serially. "
            "Defaults to the number of CPUs."
        ),
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=128,
        help="Number of records sent to each worker task batch.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output)
    report_path = args.report or output_dir / "filter_report.json"
    removed_csv_path = args.removed_csv or output_dir / "removed_rows.csv"
    yosys_log_root = _resolve_yosys_log_root(args.yosys_log_root)
    stat_root = _resolve_stat_root(args.stat_root)
    memory_regex = re.compile(args.memory_regex, re.IGNORECASE | re.MULTILINE)

    records = load_dataset_records(args.input, require_description=False)
    stat_index = build_artifact_index(stat_root, "stat.json")
    yosys_log_index = build_artifact_index(yosys_log_root, "yosys.log")
    ppa_index = build_artifact_index(yosys_log_root, "*.ppa")
    kept_rows: list[dict[str, Any]] = []
    removed_rows: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}

    results = iter_filter_results(
        records,
        workers=args.workers,
        chunksize=args.chunksize,
        stat_root=stat_root,
        yosys_log_root=yosys_log_root,
        memory_regex=memory_regex,
        gate_count_cutoff=args.gate_count_cutoff,
        drop_missing_gate_count=args.drop_missing_gate_count,
        stat_index=stat_index,
        yosys_log_index=yosys_log_index,
        ppa_index=ppa_index,
    )
    for record, decision in tqdm(
        results,
        desc="Filtering dataset",
        total=len(records),
    ):
        row = _row_with_filter_metadata(record, decision)
        if decision.keep:
            kept_rows.append(row)
        else:
            removed_rows.append(row)
            for reason in decision.reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

    output_path = save_records(kept_rows, output_dir)
    _write_removed_csv(removed_csv_path, removed_rows)
    _write_report(
        report_path,
        input_path=str(args.input),
        output_path=str(output_path),
        yosys_log_root=str(yosys_log_root),
        stat_root=str(stat_root),
        total_rows=len(records),
        kept_rows=len(kept_rows),
        removed_rows=len(removed_rows),
        reason_counts=reason_counts,
        gate_count_cutoff=args.gate_count_cutoff,
        drop_missing_gate_count=args.drop_missing_gate_count,
    )
    print(f"Wrote filtered dataset to {output_path}")
    print(f"Wrote filter report to {report_path}")
    print(f"Wrote removed rows to {removed_csv_path}")


def iter_filter_results(
    records: list[DatasetRecord],
    *,
    workers: int,
    chunksize: int,
    stat_root: Path,
    yosys_log_root: Path,
    memory_regex: re.Pattern[str],
    gate_count_cutoff: int,
    drop_missing_gate_count: bool,
    stat_index: dict[str, tuple[str, ...]],
    yosys_log_index: dict[str, tuple[str, ...]],
    ppa_index: dict[str, tuple[str, ...]],
):
    if workers <= 1:
        for record in records:
            yield (
                record,
                filter_record(
                    record,
                    stat_root=stat_root,
                    yosys_log_root=yosys_log_root,
                    memory_regex=memory_regex,
                    gate_count_cutoff=gate_count_cutoff,
                    drop_missing_gate_count=drop_missing_gate_count,
                    stat_index=stat_index,
                    yosys_log_index=yosys_log_index,
                    ppa_index=ppa_index,
                ),
            )
        return

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(
            stat_root,
            yosys_log_root,
            memory_regex.pattern,
            memory_regex.flags,
            gate_count_cutoff,
            drop_missing_gate_count,
            stat_index,
            yosys_log_index,
            ppa_index,
        ),
    ) as executor:
        yield from executor.map(
            _filter_record_worker,
            records,
            chunksize=max(1, chunksize),
        )


def _init_worker(
    stat_root: Path,
    yosys_log_root: Path,
    memory_regex_pattern: str,
    memory_regex_flags: int,
    gate_count_cutoff: int,
    drop_missing_gate_count: bool,
    stat_index: dict[str, tuple[str, ...]],
    yosys_log_index: dict[str, tuple[str, ...]],
    ppa_index: dict[str, tuple[str, ...]],
) -> None:
    global _WORKER_STAT_ROOT
    global _WORKER_YOSYS_LOG_ROOT
    global _WORKER_MEMORY_REGEX
    global _WORKER_GATE_COUNT_CUTOFF
    global _WORKER_DROP_MISSING_GATE_COUNT
    global _WORKER_STAT_INDEX
    global _WORKER_YOSYS_LOG_INDEX
    global _WORKER_PPA_INDEX

    _WORKER_STAT_ROOT = stat_root
    _WORKER_YOSYS_LOG_ROOT = yosys_log_root
    _WORKER_MEMORY_REGEX = re.compile(memory_regex_pattern, memory_regex_flags)
    _WORKER_GATE_COUNT_CUTOFF = gate_count_cutoff
    _WORKER_DROP_MISSING_GATE_COUNT = drop_missing_gate_count
    _WORKER_STAT_INDEX = stat_index
    _WORKER_YOSYS_LOG_INDEX = yosys_log_index
    _WORKER_PPA_INDEX = ppa_index


def _filter_record_worker(record: DatasetRecord) -> tuple[DatasetRecord, FilterDecision]:
    if (
        _WORKER_STAT_ROOT is None
        or _WORKER_YOSYS_LOG_ROOT is None
        or _WORKER_MEMORY_REGEX is None
    ):
        raise RuntimeError("filter worker was not initialized")
    return (
        record,
        filter_record(
            record,
            stat_root=_WORKER_STAT_ROOT,
            yosys_log_root=_WORKER_YOSYS_LOG_ROOT,
            memory_regex=_WORKER_MEMORY_REGEX,
            gate_count_cutoff=_WORKER_GATE_COUNT_CUTOFF,
            drop_missing_gate_count=_WORKER_DROP_MISSING_GATE_COUNT,
            stat_index=_WORKER_STAT_INDEX,
            yosys_log_index=_WORKER_YOSYS_LOG_INDEX,
            ppa_index=_WORKER_PPA_INDEX,
        ),
    )


def filter_record(
    record: DatasetRecord,
    *,
    stat_root: Path,
    yosys_log_root: Path,
    memory_regex: re.Pattern[str],
    gate_count_cutoff: int,
    drop_missing_gate_count: bool,
    stat_index: dict[str, tuple[str, ...]] | None = None,
    yosys_log_index: dict[str, tuple[str, ...]] | None = None,
    ppa_index: dict[str, tuple[str, ...]] | None = None,
) -> FilterDecision:
    reasons: list[str] = []
    if is_memory_module(record, memory_regex):
        reasons.append("memory_regex")

    stat_json_path = find_stat_json(record, stat_root, stat_index=stat_index)
    stat = parse_stat_json(stat_json_path) if stat_json_path else {}
    num_memories = _to_int(stat.get("num_memories"))
    num_memory_bits = _to_int(stat.get("num_memory_bits"))
    if (num_memories is not None and num_memories > 0) or (
        num_memory_bits is not None and num_memory_bits > 0
    ):
        reasons.append("yosys_stat_memory")

    yosys_log_path = find_yosys_log(
        record,
        yosys_log_root,
        yosys_log_index=yosys_log_index,
    )
    ppa_path = find_ppa_report(
        record,
        yosys_log_root,
        ppa_index=ppa_index,
    )
    if ppa_path is None:
        reasons.append("missing_ppa")

    gate_count = _to_int(stat.get("num_cells"))
    if gate_count is None:
        gate_count = parse_gate_count(yosys_log_path) if yosys_log_path else None
    if gate_count_cutoff >= 0:
        if gate_count is None:
            if drop_missing_gate_count:
                reasons.append("missing_gate_count")
        elif gate_count < gate_count_cutoff:
            reasons.append("below_gate_count_cutoff")

    return FilterDecision(
        keep=not reasons,
        reasons=reasons,
        gate_count=gate_count,
        num_memories=num_memories,
        num_memory_bits=num_memory_bits,
        stat_json_path=str(stat_json_path) if stat_json_path else None,
        yosys_log_path=str(yosys_log_path) if yosys_log_path else None,
        ppa_path=str(ppa_path) if ppa_path else None,
    )


def is_memory_module(record: DatasetRecord, memory_regex: re.Pattern[str]) -> bool:
    text = "\n".join(
        [
            record.sample_id,
            record.top_module,
            record.description,
            record.verilog,
        ]
    )
    return bool(memory_regex.search(text))


def find_yosys_log(
    record: DatasetRecord,
    yosys_log_root: Path,
    *,
    yosys_log_index: dict[str, tuple[str, ...]] | None = None,
) -> Path | None:
    raw_path = _nested_get(record.raw, ("synthesis", "yosys_log_path"))
    if raw_path:
        path = Path(str(raw_path))
        if path.exists():
            return path
        candidate = yosys_log_root / path.name
        if candidate.exists():
            return candidate

    candidates = [
        yosys_log_root / sanitize_id(record.sample_id) / "baseline" / "yosys.log",
        yosys_log_root / sanitize_id(record.sample_id) / sanitize_id("baseline") / "yosys.log",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = _indexed_artifact_matches(
        record.sample_id,
        artifact_index=yosys_log_index,
    )
    if matches is None:
        matches = list(yosys_log_root.glob(f"**/{sanitize_id(record.sample_id)}/**/yosys.log"))
    if len(matches) == 1:
        return matches[0]
    if matches:
        return sorted(matches, key=lambda path: len(path.parts))[0]
    return None


def find_stat_json(
    record: DatasetRecord,
    stat_root: Path,
    *,
    stat_index: dict[str, tuple[str, ...]] | None = None,
) -> Path | None:
    raw_path = _nested_get(record.raw, ("synthesis", "stat_json_path"))
    if raw_path:
        path = Path(str(raw_path))
        if path.exists():
            return path

    candidates = [
        stat_root / sanitize_id(record.sample_id) / "baseline" / "stat.json",
        stat_root / sanitize_id(record.sample_id) / sanitize_id("baseline") / "stat.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = _indexed_artifact_matches(
        record.sample_id,
        artifact_index=stat_index,
    )
    if matches is None:
        matches = list(stat_root.glob(f"**/{sanitize_id(record.sample_id)}/**/stat.json"))
    if len(matches) == 1:
        return matches[0]
    if matches:
        return sorted(matches, key=lambda path: len(path.parts))[0]
    return None


def build_artifact_index(root: Path, filename: str) -> dict[str, tuple[str, ...]]:
    if not root.exists():
        return {}

    index: dict[str, list[str]] = {}
    for path in root.rglob(filename):
        for parent in path.parents:
            if parent == root:
                break
            sample_id = parent.name
            if sample_id in {"baseline", sanitize_id("baseline")}:
                continue
            index.setdefault(sample_id, []).append(str(path))
    return {key: tuple(value) for key, value in index.items()}


def find_ppa_report(
    record: DatasetRecord,
    yosys_log_root: Path,
    *,
    ppa_index: dict[str, tuple[str, ...]] | None = None,
) -> Path | None:
    raw_path = _nested_get(record.raw, ("synthesis", "ppa_path"))
    if raw_path:
        path = Path(str(raw_path))
        if path.exists():
            return path
        candidate = yosys_log_root / path.name
        if candidate.exists():
            return candidate

    sample_id = sanitize_id(record.sample_id)
    report_name = f"{sample_id}_baseline_synthesis_report.ppa"
    candidates = [
        yosys_log_root / sample_id / "baseline" / report_name,
        yosys_log_root / sample_id / sanitize_id("baseline") / report_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = _indexed_artifact_matches(
        record.sample_id,
        artifact_index=ppa_index,
    )
    if matches is None:
        matches = list(yosys_log_root.glob(f"**/{sample_id}/**/*.ppa"))
    if len(matches) == 1:
        return matches[0]
    if matches:
        return sorted(matches, key=lambda path: len(path.parts))[0]
    return None


def _indexed_artifact_matches(
    sample_id: str,
    *,
    artifact_index: dict[str, tuple[str, ...]] | None,
) -> list[Path] | None:
    if artifact_index is None:
        return None
    return [Path(path) for path in artifact_index.get(sanitize_id(sample_id), ())]


def parse_stat_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}

    design = data.get("design")
    if isinstance(design, dict):
        return design

    modules = data.get("modules")
    if isinstance(modules, dict) and modules:
        module_stats = [
            value for value in modules.values() if isinstance(value, dict)
        ]
        if len(module_stats) == 1:
            return module_stats[0]
        if module_stats:
            return max(
                module_stats,
                key=lambda item: _to_int(item.get("num_cells")) or 0,
            )
    return {}


def parse_gate_count(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    counts = [
        int(match.group(1).replace(",", "").replace("_", ""))
        for match in GATE_COUNT_PATTERNS[0].finditer(text)
    ]
    if counts:
        return counts[-1]
    cell_table_count = _parse_cell_table_count(text)
    if cell_table_count is not None:
        return cell_table_count
    return None


def _parse_cell_table_count(text: str) -> int | None:
    in_table = False
    total = 0
    saw_cell = False
    for line in text.splitlines():
        if re.search(r"^\s*===\s+\S+\s+===", line):
            in_table = False
            total = 0
            saw_cell = False
            continue
        if re.search(r"^\s*Cell\s+type\s+Count\s*$", line, re.IGNORECASE):
            in_table = True
            total = 0
            saw_cell = False
            continue
        if not in_table:
            continue
        match = re.match(r"^\s*(\S+)\s+([0-9][0-9_,]*)\s*$", line)
        if match:
            saw_cell = True
            total += int(match.group(2).replace(",", "").replace("_", ""))
        elif saw_cell and not line.strip():
            return total
    return total if in_table and saw_cell else None


def _row_with_filter_metadata(
    record: DatasetRecord, decision: FilterDecision
) -> dict[str, Any]:
    row = dict(record.raw)
    row.update(
        {
            "sample_id": record.sample_id,
            "description": record.description,
            "verilog": record.verilog,
            "top_module": record.top_module,
            "filter_reasons": decision.reasons,
            "filter_gate_count": decision.gate_count,
            "filter_num_memories": decision.num_memories,
            "filter_num_memory_bits": decision.num_memory_bits,
            "filter_stat_json_path": decision.stat_json_path,
            "filter_yosys_log_path": decision.yosys_log_path,
            "filter_ppa_path": decision.ppa_path,
        }
    )
    return row


def _nested_get(row: dict[str, Any], keys: tuple[str, ...]) -> Any | None:
    value: Any = row
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _resolve_yosys_log_root(path: Path) -> Path:
    if path.exists():
        return path
    if path == DEFAULT_YOSYS_LOG_ROOT:
        return DEFAULT_YOSYS_LOG_ROOT
    return path


def _resolve_stat_root(path: Path) -> Path:
    if path.exists():
        return path
    src_relative = Path("src") / path
    if src_relative.exists():
        return src_relative
    return path


def _to_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def _write_removed_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "top_module",
        "clock",
        "filter_reasons",
        "filter_gate_count",
        "filter_num_memories",
        "filter_num_memory_bits",
        "filter_stat_json_path",
        "filter_yosys_log_path",
        "filter_ppa_path",
    ]
    csv_rows = []
    for row in rows:
        csv_row = dict(row)
        csv_row["filter_reasons"] = ";".join(row.get("filter_reasons", []))
        csv_rows.append(csv_row)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)


def _write_report(
    path: Path,
    *,
    input_path: str,
    output_path: str,
    yosys_log_root: str,
    stat_root: str,
    total_rows: int,
    kept_rows: int,
    removed_rows: int,
    reason_counts: dict[str, int],
    gate_count_cutoff: int,
    drop_missing_gate_count: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "input": input_path,
        "output": output_path,
        "yosys_log_root": yosys_log_root,
        "stat_root": stat_root,
        "total_rows": total_rows,
        "kept_rows": kept_rows,
        "removed_rows": removed_rows,
        "reason_counts": reason_counts,
        "gate_count_cutoff": gate_count_cutoff,
        "drop_missing_gate_count": drop_missing_gate_count,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
