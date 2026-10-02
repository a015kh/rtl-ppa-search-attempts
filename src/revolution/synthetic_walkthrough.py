"""Run the real offline dataset workflow using newly invented tiny inputs.

The labels, RTL and source identities are fixtures, not measured PPA evidence.
Only Python's standard library is needed. No model or synthesis tool is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from . import dataset_generation as dg
from . import test_dataset_preparation as tp


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [row for _, row in dg.jsonl_rows(path)]


def fixture_row(index: int, *, augmented: bool = False) -> dict[str, Any]:
    """Invent a measured-row-shaped fixture; values have no physical meaning."""
    name = (f"sample_{index:06d}_area_gen1_cand0" if augmented
            else f"synthetic_{index}")
    module = f"synthetic_augmented_{index}" if augmented else name
    expression = "~a" if augmented else "a"
    return {
        "SDC": "create_clock -period 2.0 -name clk [get_ports clk]",
        "has_clock": True, "clock": 2.0,
        "area_value": 10.0 + index, "delay_value": 0.5,
        "wns": 1.5, "tns": 0.0, "total_power_value": 0.25,
        "static_power_value": 0.10, "switching_power_value": 0.15,
        "synth": "fabricated cell summary", "area": "fabricated area report",
        "delay": "fabricated timing report", "static_power": "fabricated static power 0.10",
        "switching_power": "fabricated dynamic power 0.15", "synthesis_time": 0.01,
        "design_name": name,
        "rtl": (f"module {module}(input clk, input a, output y); "
                f"assign y = {expression}; endmodule"),
    }


def _prepare_inputs(directory: Path) -> tuple[Path, Path]:
    measured = directory / "inputs" / "measured"
    base = [fixture_row(index) for index in range(8)]
    _write_rows(measured / "train.jsonl", base[:4])
    # The first measurement wins when an identity is duplicated later.
    _write_rows(measured / "valid.jsonl", base[4:] + [dict(base[0], area_value=999.0)])
    _write_rows(directory / "inputs" / "seed.jsonl", [{"rtl": row["rtl"]} for row in base])
    _write_rows(measured / "metadata" / "source_catalog.jsonl", [
        {
            "rtl_hash": dg.sha256_text(dg.normalized_rtl(row["rtl"])),
            "source_dataset": "synthetic/invented", "source_revision": "fixture-v1",
            "source_split": "train", "source_row_index": index,
            "module_id": f"synthetic-module-{index}",
        }
        for index, row in enumerate(base)
    ])
    _write_rows(measured / "augmentation.jsonl",
                [fixture_row(index, augmented=True) for index in range(8)])
    test_results = []
    for index in (20, 21):
        row = fixture_row(index)
        test_results.append({
            "Name": row["design_name"], "RTL": row["rtl"],
            "Syn": [{
                "SDC": row["SDC"], "Has Clock": row["has_clock"], "Clock": row["clock"],
                "Total Area": row["area_value"], "Delay": row["delay_value"],
                "WNS": row["wns"], "TNS": row["tns"], "Total Power": row["total_power_value"],
                **{field: row[field] for field in (
                    "synth", "area", "delay", "static_power", "switching_power", "synthesis_time")},
            }],
        })
    (directory / "inputs" / "test-results.json").write_text(
        json.dumps(test_results, indent=2) + "\n", encoding="utf-8")
    test_configuration = directory / "test-dataset.json"
    test_configuration.write_text(json.dumps({
        "output_root": "prepared-tests",
        "shards": [{"name": "test_synthetic.jsonl", "policy": "legacy_result_json",
                    "inputs": ["inputs/test-results.json"]}],
    }, indent=2) + "\n", encoding="utf-8")
    configuration = directory / "dataset.toml"
    configuration.write_text('''# All inputs here are newly fabricated fixtures.
[dataset]
output_root = "dataset"
legacy_final = "inputs/measured"
seed_train = "inputs/seed.jsonl"
catalog_cache = "cache/source_catalog.jsonl"
split_seed = 42
valid_fraction = 0.25
split_tolerance = 0.001

[[upstream]]
name = "synthetic/invented"
revision = "fixture-v1"
split = "train"
rtl_field = "rtl"
priority = 0

[[augmentation]]
origin = "synthetic_augmentation"
legacy_shard = "augmentation.jsonl"
model = "synthetic-generator"
forced_source = ""

[tests]
shards = ["test_synthetic.jsonl"]

[tests.sources]
"test_synthetic.jsonl" = "prepared-tests/test_synthetic.jsonl"
''', encoding="utf-8")
    return configuration, test_configuration


def run_walkthrough(output: Path) -> dict[str, Any]:
    output = output.expanduser().resolve()
    source_directory = Path(__file__).resolve().parent
    checkout = source_directory.parents[1]
    if output.is_relative_to(source_directory) or (
        (checkout / "pyproject.toml").is_file() and output.is_relative_to(checkout)
    ):
        raise ValueError("Choose a new output directory outside the source checkout")
    if output.exists():
        raise ValueError("Output already exists; choose a new directory")
    output.mkdir(parents=True, exist_ok=False)
    configuration, test_configuration = _prepare_inputs(output)
    test_config = tp.load_config(test_configuration)
    tp.preflight(test_config)
    prepared = tp.build_command(test_config)
    config = dg.load_config(configuration)
    preflight = dg.preflight(config, allow_network=False)
    built = dg.build_command(config, allow_network=False, publish_result=True)
    verified = dg.verify_dataset(config.output_root, config)
    expected = {
        "train.jsonl": 6, "valid.jsonl": 2,
        "train_aug.jsonl": 6, "valid_aug.jsonl": 2, "test_synthetic.jsonl": 2,
    }
    records = {name: _read_rows(config.output_root / name) for name in expected}
    train = records["train.jsonl"] + records["train_aug.jsonl"]
    valid = records["valid.jsonl"] + records["valid_aug.jsonl"]
    train_lineages = {row["lineage_id"] for row in train}
    valid_lineages = {row["lineage_id"] for row in valid}
    original = next(row for row in train + valid if row["design_name"] == "synthetic_0")
    checks = {
        "test_result_conversion": prepared["shards"]["test_synthetic.jsonl"]["accepted"] == 2,
        "offline_preflight": preflight["status"] == "ok",
        "published_dataset": built["status"] == "published",
        "expected_split_counts": verified["counts"] == expected,
        "duplicate_keeps_first_measurement": original["area_value"] == 10.0,
        "no_training_validation_lineage_overlap": not (train_lineages & valid_lineages),
        "augmentation_keeps_seed_lineage": all(
            row["seed_present_in_base"] for row in
            records["train_aug.jsonl"] + records["valid_aug.jsonl"]),
        "numeric_labels_unchanged": all(
            row["delay_value"] == 0.5 and row["total_power_value"] == 0.25
            for rows in records.values() for row in rows),
        "test_rows_remain_test": all(
            row["record_kind"] == "test" and row["source_split"] == "test"
            for row in records["test_synthetic.jsonl"]),
    }
    summary = {
        "status": "passed" if all(checks.values()) else "failed",
        "fixture_only": True,
        "purpose": "Offline engineering walkthrough; no measured PPA or model-quality evidence",
        "split_counts": verified["counts"],
        "prepared_test_rows": prepared["shards"]["test_synthetic.jsonl"]["accepted"],
        "training_lineages": len(train_lineages), "validation_lineages": len(valid_lineages),
        "lineage_overlap": len(train_lineages & valid_lineages),
        "output_sha256": verified["output_sha256"], "checks": checks,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if summary["status"] != "passed":
        raise RuntimeError("Synthetic walkthrough failed; inspect summary.json")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="New directory outside the checkout for synthetic inputs and outputs")
    args = parser.parse_args(argv)
    try:
        summary = run_walkthrough(args.output)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Walkthrough error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
