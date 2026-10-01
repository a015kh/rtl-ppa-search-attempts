from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from augmention.dataset_io import load_dataset_records


DEFAULT_OUTPUT_FILENAME = "sample_design_names.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recover the original design_name for augmentation samples and write "
            "a {sample_id: design_name} JSON map into the augmentation work folder."
        )
    )
    parser.add_argument(
        "input_data_folder",
        help="Original dataset folder or file that still contains design_name.",
    )
    parser.add_argument(
        "output_work_folder",
        help="Augmentation work folder, for example outputs/augmention/eoh_v6.2.0_random_000_099.",
    )
    parser.add_argument(
        "--output-filename",
        default=DEFAULT_OUTPUT_FILENAME,
        help=f"Name of the JSON file to write inside output_work_folder. Default: {DEFAULT_OUTPUT_FILENAME}.",
    )
    parser.add_argument(
        "--all-input-samples",
        action="store_true",
        help=(
            "Write every sample found in the input dataset. By default, only sample_ids "
            "already present in the work folder are written when they can be detected."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input_data_folder)
    work_dir = Path(args.output_work_folder)
    work_dir.mkdir(parents=True, exist_ok=True)

    records = load_dataset_records(input_path, require_description=False)
    design_names_by_sample = {
        record.sample_id: record.design_name
        for record in records
        if record.design_name is not None
    }

    module_names_by_sample = _discover_work_module_names(work_dir)
    sample_ids = None if args.all_input_samples else _discover_work_sample_ids(work_dir)
    if sample_ids:
        design_names_by_sample = {
            sample_id: design_names_by_sample.get(sample_id)
            or module_names_by_sample[sample_id]
            for sample_id in sorted(sample_ids)
            if sample_id in design_names_by_sample or sample_id in module_names_by_sample
        }

    output_path = work_dir / args.output_filename
    output_path.write_text(
        json.dumps(design_names_by_sample, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        f"Wrote {len(design_names_by_sample)} design-name mappings to {output_path}"
    )


def _discover_work_sample_ids(work_dir: Path) -> set[str]:
    sample_ids: set[str] = set()
    summary_path = work_dir / "evolution_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for selection in summary.get("target_selections", []):
            sample_id = selection.get("sample_id")
            if sample_id:
                sample_ids.add(str(sample_id))
        for result in summary.get("best_by_sample_target", {}).values():
            sample_id = result.get("sample_id")
            if sample_id:
                sample_ids.add(str(sample_id))

    for child in work_dir.iterdir() if work_dir.exists() else ():
        if child.is_dir():
            sample_ids.add(child.name)
    return sample_ids


def _discover_work_module_names(work_dir: Path) -> dict[str, str]:
    module_names: dict[str, str] = {}
    if not work_dir.exists():
        return module_names

    for sample_dir in sorted(child for child in work_dir.iterdir() if child.is_dir()):
        module_name = _find_first_module_name(sample_dir)
        if module_name:
            module_names[sample_dir.name] = module_name
    return module_names


def _find_first_module_name(sample_dir: Path) -> str | None:
    for verilog_path in sorted(sample_dir.rglob("*.sv")):
        if verilog_path.name.endswith(".syn.v"):
            continue
        module_name = _extract_module_name(verilog_path)
        if module_name:
            return module_name
    for verilog_path in sorted(sample_dir.rglob("*.v")):
        if verilog_path.name.endswith(".syn.v"):
            continue
        module_name = _extract_module_name(verilog_path)
        if module_name:
            return module_name
    return None


def _extract_module_name(path: Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    match = re.search(r"\bmodule\s+([A-Za-z_][A-Za-z0-9_$]*)\b", text)
    return match.group(1) if match else None


if __name__ == "__main__":
    main()
