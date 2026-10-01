from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .models import PpaMetrics

VERILOG_COLUMNS = (
    "verilog",
    "verilog_code",
    "rtl",
    "code",
    "module",
    "generated_verilog",
    "original_verilog",
)
ID_COLUMNS = ("id", "sample_id", "name", "problem_name", "module_id")
TOP_COLUMNS = ("design_name", "top_module", "module_name")
SPLIT_FILE_STEMS = ("train", "validation", "valid", "dev", "test")
METADATA_FILE_NAMES = {
    "dataset_info.json",
    "state.json",
    "manifest.json",
    "README.md",
}


@dataclass
class DatasetRecord:
    sample_id: str
    design_name: str | None
    description: str
    verilog: str
    top_module: str
    ppa: PpaMetrics = field(default_factory=PpaMetrics)
    raw: dict[str, Any] = field(default_factory=dict)


def load_dataset_records(
    path: str | Path, *, require_description: bool = True
) -> list[DatasetRecord]:
    """Load records from a directory or file into the augmentation schema."""

    dataset_path = Path(path)
    rows = _load_rows(dataset_path)
    records: list[DatasetRecord] = []

    for idx, row in enumerate(rows):
        description = row.get("description")
        if description is None and require_description:
            raise ValueError(
                f"Dataset row {idx} has no 'description' column. "
                "The augmentation flow requires text descriptions."
            )

        verilog = _first_present(row, VERILOG_COLUMNS)
        if verilog is None:
            raise ValueError(
                f"Dataset row {idx} has no Verilog column. "
                f"Tried: {', '.join(VERILOG_COLUMNS)}."
            )

        sample_id = _first_present(row, ID_COLUMNS) or f"sample_{idx:06d}"
        top_module = _first_present(row, TOP_COLUMNS) or infer_top_module(verilog)

        records.append(
            DatasetRecord(
                sample_id=sanitize_id(str(sample_id)),
                design_name=(
                    str(row["design_name"])
                    if _has_value(row.get("design_name"))
                    else None
                ),
                description="" if description is None else str(description),
                verilog=str(verilog),
                top_module=str(top_module),
                ppa=_extract_ppa(row),
                raw=row,
            )
        )

    return records


def save_records(
    records: Iterable[dict[str, Any]],
    output_dir: str | Path,
    *,
    filename: str = "train.jsonl",
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_path / filename
    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    _write_dataset_manifest(output_path, jsonl_path)
    return jsonl_path


def append_record(record: dict[str, Any], jsonl_path: str | Path) -> None:
    path = Path(jsonl_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_existing_keys(jsonl_path: str | Path) -> set[tuple[str, str, int, int]]:
    path = Path(jsonl_path)
    if not path.exists():
        return set()

    keys: set[tuple[str, str, int, int]] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            keys.add(
                (
                    str(row.get("sample_id")),
                    str(row.get("target")),
                    int(row.get("generation", -1)),
                    int(row.get("candidate_index", -1)),
                )
            )
    return keys


def infer_top_module(verilog: str) -> str:
    match = re.search(r"\bmodule\s+([A-Za-z_][A-Za-z0-9_$]*)\b", verilog)
    if not match:
        raise ValueError("Could not infer top module name from Verilog text.")
    return match.group(1)


def sanitize_id(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return sanitized.strip("._") or "sample"


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")

    if path.is_dir():
        hf_rows = _try_load_hf_dataset(path)
        if hf_rows is not None:
            return hf_rows

        candidates = [
            path / "train.jsonl",
            path / "valid.jsonl",
        ]
        for candidate in candidates:
            if candidate.exists():
                return _load_rows(candidate)

        supported = sorted(
            p
            for p in path.iterdir()
            if _is_supported_data_file(p)
        )
        if supported:
            rows: list[dict[str, Any]] = []
            for file_path in supported:
                rows.extend(_load_rows(file_path))
            return rows

        raise FileNotFoundError(f"No supported dataset file found in {path}")

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if suffix == ".json":
        data = _load_json_or_jsonl(path)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("data", "records", "examples"):
                if isinstance(data.get(key), list):
                    return data[key]
        raise ValueError(f"Unsupported JSON dataset shape in {path}")
    if suffix == ".csv":
        return pd.read_csv(path).to_dict(orient="records")
    if suffix == ".parquet":
        return pd.read_parquet(path).to_dict(orient="records")
    raise ValueError(f"Unsupported dataset file extension: {path}")


def _is_supported_data_file(path: Path) -> bool:
    if not path.is_file() or path.name in METADATA_FILE_NAMES:
        return False
    if path.suffix.lower() not in {".jsonl", ".json", ".csv", ".parquet"}:
        return False
    if path.stem in SPLIT_FILE_STEMS:
        return True
    return path.name in {
        "train.jsonl",
        "valid.jsonl",
    }


def _load_json_or_jsonl(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except JSONDecodeError:
            f.seek(0)
            return [json.loads(line) for line in f if line.strip()]


def _try_load_hf_dataset(path: Path) -> list[dict[str, Any]] | None:
    if not (path / "dataset_info.json").exists() and not (path / "state.json").exists():
        return None
    try:
        from datasets import Dataset, DatasetDict, load_from_disk
    except ImportError:
        return None

    dataset = load_from_disk(str(path))
    if isinstance(dataset, DatasetDict):
        if "train" in dataset:
            dataset = dataset["train"]
        else:
            dataset = next(iter(dataset.values()))
    if not isinstance(dataset, Dataset):
        return None
    return [dict(row) for row in dataset]


def _first_present(row: dict[str, Any], columns: tuple[str, ...]) -> Any | None:
    for column in columns:
        if column in row and _has_value(row[column]):
            return row[column]
    return None


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value != ""
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return True
    if isinstance(missing, bool):
        return not missing
    return True


def _extract_ppa(row: dict[str, Any]) -> PpaMetrics:
    if isinstance(row.get("ppa"), dict):
        return PpaMetrics.from_mapping(row["ppa"])
    if isinstance(row.get("ppa_metrics"), dict):
        return PpaMetrics.from_mapping(row["ppa_metrics"])
    mapped = dict(row)
    if "area" not in mapped and "area_value" in mapped:
        mapped["area"] = mapped["area_value"]
    if "power" not in mapped and "total_power_value" in mapped:
        mapped["power"] = mapped["total_power_value"]
    if "eff_clk_period" not in mapped and "delay_value" in mapped:
        mapped["eff_clk_period"] = mapped["delay_value"]
    return PpaMetrics.from_mapping(mapped)


def _write_dataset_manifest(output_dir: Path, jsonl_path: Path) -> None:
    manifest = {
        "format": "jsonl",
        "data_file": jsonl_path.name,
        "note": "Use this directory as the augmentation dataset input/output root.",
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
