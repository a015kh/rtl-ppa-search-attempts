from __future__ import annotations

import argparse
import hashlib
import html
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


CODE_COLUMNS = ("verilog", "rtl", "verilog_code", "code", "module")
DESCRIPTION_COLUMNS = ("description", "instruction", "prompt")
ID_COLUMNS = ("sample_id", "id", "name", "problem_name", "module_id")
TOP_COLUMNS = ("top_module", "design_name", "module_name")
METRIC_COLUMNS = (
    "area",
    "area_value",
    "delay",
    "delay_value",
    "power",
    "total_power_value",
    "wns",
    "tns",
    "clock",
)
SYNTHESIS_COMPARE_COLUMNS = {
    "area": ("area_value", "area"),
    "delay": ("delay_value", "eff_clk_period"),
    "power": ("total_power_value", "power"),
    "wns": ("wns", "wns"),
    "tns": ("tns", "tns"),
}


def main() -> None:
    args = build_parser().parse_args()
    path = Path(args.path)
    files = list_jsonl_files(path)
    if not files:
        raise SystemExit(f"No .jsonl files found under {path}")

    if args.list:
        for file_path in files:
            print(file_path)
        return

    for file_path in files:
        summarize_file(file_path, limit=args.limit)

    if args.dedup_output:
        write_dedup_jsonl(
            files=files,
            output=Path(args.dedup_output),
            key_mode=args.dedup_key,
            limit=args.limit,
        )

    if args.html:
        write_html_report(
            files=files,
            output=Path(args.html),
            samples_per_file=args.samples,
            char_limit=args.char_limit,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read RTL JSONL datasets and inspect description/code quality."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="data",
        help="Dataset directory or JSONL file. Default: data",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum rows per file for terminal statistics.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Only list discovered JSONL files.",
    )
    parser.add_argument(
        "--html",
        help="Write an HTML review report with description/code samples.",
    )
    parser.add_argument(
        "--dedup-output",
        help=(
            "Write a deduplicated JSONL file. Rows with the same normalized "
            "description and Verilog/RTL are treated as duplicates."
        ),
    )
    parser.add_argument(
        "--dedup-key",
        choices=("code-description", "code"),
        default="code-description",
        help=(
            "Deduplication key. Default keeps one row per unique RTL plus "
            "description pair; code keeps one row per unique RTL."
        ),
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=12,
        help="Samples per file in the HTML report.",
    )
    parser.add_argument(
        "--char-limit",
        type=int,
        default=5000,
        help="Maximum characters shown per description/code block in HTML.",
    )
    return parser


def list_jsonl_files(path: Path) -> list[Path]:
    if path.is_file() and path.suffix == ".jsonl":
        return [path]
    if path.is_dir():
        return sorted(path.rglob("*.jsonl"))
    return []


def summarize_file(path: Path, *, limit: int | None) -> None:
    stats = collect_stats(path, limit=limit)
    print(f"\n== {path} ==")
    print(f"rows: {stats['rows']}")
    print(f"keys: {', '.join(stats['keys'])}")
    print(f"description coverage: {stats['description_rows']}/{stats['rows']}")
    print(f"code coverage: {stats['code_rows']}/{stats['rows']}")
    print(f"top modules: {format_counter(stats['tops'], 8)}")
    print(f"missing description rows: {format_counter(stats['missing_description'], 8)}")
    print(f"missing code rows: {format_counter(stats['missing_code'], 8)}")
    for name, values in stats["metrics"].items():
        print(f"{name}: {format_numbers(values)}")
    if stats["synthesis_compare"]:
        print("Design Compiler vs Yosys:")
        for name, values in stats["synthesis_compare"].items():
            print(f"  {name}: {format_comparison(values)}")
    if stats["errors"]:
        print(f"json errors: {format_counter(stats['errors'], 5)}")


def collect_stats(path: Path, *, limit: int | None) -> dict[str, Any]:
    keys: set[str] = set()
    tops: Counter[str] = Counter()
    missing_description: Counter[str] = Counter()
    missing_code: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    metrics: dict[str, list[float]] = defaultdict(list)
    synthesis_compare: dict[str, list[dict[str, float]]] = defaultdict(list)
    rows = 0
    description_rows = 0
    code_rows = 0

    for line_no, row in iter_jsonl(path):
        if limit is not None and rows >= limit:
            break
        if isinstance(row, str):
            errors[f"line {line_no}: {row}"] += 1
            continue
        rows += 1
        keys.update(row.keys())
        row_id = str(first_present(row, ID_COLUMNS) or f"line_{line_no}")
        top = str(first_present(row, TOP_COLUMNS) or "(unknown)")
        tops[top] += 1

        if first_present(row, DESCRIPTION_COLUMNS):
            description_rows += 1
        else:
            missing_description[row_id] += 1
        if first_present(row, CODE_COLUMNS):
            code_rows += 1
        else:
            missing_code[row_id] += 1

        for name in METRIC_COLUMNS:
            value = row.get(name)
            if isinstance(value, (int, float)):
                metrics[name].append(float(value))
        for name, pair in extract_synthesis_comparison(row).items():
            synthesis_compare[name].append(pair)

    return {
        "rows": rows,
        "keys": sorted(keys),
        "tops": tops,
        "description_rows": description_rows,
        "code_rows": code_rows,
        "missing_description": missing_description,
        "missing_code": missing_code,
        "metrics": metrics,
        "synthesis_compare": synthesis_compare,
        "errors": errors,
    }


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any] | str]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                yield line_no, exc.msg
                continue
            if not isinstance(row, dict):
                yield line_no, f"expected object, got {type(row).__name__}"
                continue
            yield line_no, row


def extract_synthesis_comparison(row: dict[str, Any]) -> dict[str, dict[str, float]]:
    synthesis = row.get("synthesis")
    if not isinstance(synthesis, dict):
        return {}
    yosys_metrics = synthesis.get("metrics")
    if not isinstance(yosys_metrics, dict):
        return {}

    comparisons: dict[str, dict[str, float]] = {}
    for name, (dc_key, yosys_key) in SYNTHESIS_COMPARE_COLUMNS.items():
        dc_value = numeric(row.get(dc_key))
        yosys_value = numeric(yosys_metrics.get(yosys_key))
        if dc_value is None or yosys_value is None:
            continue
        comparisons[name] = {
            "dc": dc_value,
            "yosys": yosys_value,
            "delta": yosys_value - dc_value,
        }
        if dc_value != 0:
            comparisons[name]["ratio"] = yosys_value / dc_value
    return comparisons


def numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def write_dedup_jsonl(
    *, files: list[Path], output: Path, key_mode: str, limit: int | None
) -> None:
    seen: set[str] = set()
    rows_seen = 0
    rows_written = 0
    duplicates = 0

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as out:
        for file_path in files:
            file_rows = 0
            for _, row in iter_jsonl(file_path):
                if isinstance(row, str):
                    continue
                if limit is not None and file_rows >= limit:
                    break
                file_rows += 1
                rows_seen += 1

                key = dedup_key(row, key_mode=key_mode)
                if key in seen:
                    duplicates += 1
                    continue
                seen.add(key)
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows_written += 1

    print(
        f"\nWrote deduplicated JSONL: {output}\n"
        f"dedup key: {key_mode}; input rows: {rows_seen}; "
        f"output rows: {rows_written}; duplicates removed: {duplicates}"
    )


def dedup_key(row: dict[str, Any], *, key_mode: str) -> str:
    code = normalize_text(str(first_present(row, CODE_COLUMNS) or ""))
    description = normalize_text(str(first_present(row, DESCRIPTION_COLUMNS) or ""))
    payload = code if key_mode == "code" else f"{code}\0{description}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_text(value: str) -> str:
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in normalized.splitlines()).strip()


def write_html_report(
    *, files: list[Path], output: Path, samples_per_file: int, char_limit: int
) -> None:
    sections = []
    for file_path in files:
        cards = []
        for line_no, row in iter_jsonl(file_path):
            if len(cards) >= samples_per_file:
                break
            if isinstance(row, str):
                continue
            cards.append(render_sample(file_path, line_no, row, char_limit=char_limit))
        if cards:
            sections.append(f"<section><h2>{escape(file_path)}</h2>{''.join(cards)}</section>")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_page("".join(sections)), encoding="utf-8")
    print(f"\nWrote HTML report: {output}")


def render_sample(
    file_path: Path, line_no: int, row: dict[str, Any], *, char_limit: int
) -> str:
    row_id = first_present(row, ID_COLUMNS) or f"line {line_no}"
    top = first_present(row, TOP_COLUMNS) or "(unknown top)"
    description = truncate(str(first_present(row, DESCRIPTION_COLUMNS) or ""), char_limit)
    code = truncate(str(first_present(row, CODE_COLUMNS) or ""), char_limit)
    metrics = []
    for name in METRIC_COLUMNS:
        if name in row:
            metrics.append(f"<span>{escape(name)}: {escape(row[name])}</span>")
    for name, pair in extract_synthesis_comparison(row).items():
        ratio = pair.get("ratio")
        ratio_text = "" if ratio is None else f", ratio {ratio:.4g}"
        metrics.append(
            "<span>"
            f"{escape(name)} DC/Yosys: {pair['dc']:.4g} / {pair['yosys']:.4g}"
            f"{escape(ratio_text)}"
            "</span>"
        )
    return f"""
<article class="sample">
  <header>
    <div><strong>{escape(row_id)}</strong> <span>{escape(top)}</span></div>
    <small>{escape(file_path)}:{line_no}</small>
  </header>
  <div class="metrics">{''.join(metrics)}</div>
  <div class="columns">
    <div><h3>Description</h3><pre>{escape(description)}</pre></div>
    <div><h3>Verilog / RTL</h3><pre>{escape(code)}</pre></div>
  </div>
</article>
"""


def render_page(body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>RTL JSONL Dataset Review</title>
  <style>
    body {{ margin: 0; font: 14px/1.45 system-ui, sans-serif; color: #1f2937; background: #f6f7f9; }}
    h1, h2, h3 {{ margin: 0; }}
    h1 {{ padding: 24px 28px 8px; font-size: 24px; }}
    h2 {{ padding: 20px 28px 10px; font-size: 18px; }}
    .sample {{ margin: 14px 28px 24px; border: 1px solid #d7dce2; border-radius: 8px; background: white; overflow: hidden; }}
    header {{ display: flex; justify-content: space-between; gap: 16px; padding: 12px 14px; background: #eef2f6; }}
    header span, small {{ color: #5f6b7a; }}
    .metrics {{ display: flex; flex-wrap: wrap; gap: 8px; padding: 10px 14px; border-bottom: 1px solid #e5e7eb; }}
    .metrics span {{ padding: 2px 7px; border: 1px solid #d7dce2; border-radius: 999px; background: #fafafa; }}
    .columns {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1.3fr); gap: 0; }}
    .columns > div {{ min-width: 0; padding: 14px; }}
    .columns > div + div {{ border-left: 1px solid #e5e7eb; }}
    h3 {{ margin-bottom: 8px; font-size: 13px; color: #435064; }}
    pre {{ margin: 0; white-space: pre-wrap; overflow-wrap: anywhere; font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    @media (max-width: 900px) {{
      .columns {{ grid-template-columns: 1fr; }}
      .columns > div + div {{ border-left: 0; border-top: 1px solid #e5e7eb; }}
      header {{ flex-direction: column; }}
    }}
  </style>
</head>
<body>
  <h1>RTL JSONL Dataset Review</h1>
  {body}
</body>
</html>
"""


def first_present(row: dict[str, Any], names: tuple[str, ...]) -> Any | None:
    for name in names:
        value = row.get(name)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def format_counter(counter: Counter[str], n: int) -> str:
    if not counter:
        return "(none)"
    return ", ".join(f"{key}={value}" for key, value in counter.most_common(n))


def format_numbers(values: list[float]) -> str:
    if not values:
        return "(none)"
    return (
        f"count={len(values)}, min={min(values):.4g}, "
        f"median={statistics.median(values):.4g}, max={max(values):.4g}"
    )


def format_comparison(values: list[dict[str, float]]) -> str:
    dc_values = [value["dc"] for value in values]
    yosys_values = [value["yosys"] for value in values]
    deltas = [value["delta"] for value in values]
    ratios = [value["ratio"] for value in values if "ratio" in value]
    text = (
        f"count={len(values)}, "
        f"DC median={statistics.median(dc_values):.4g}, "
        f"Yosys median={statistics.median(yosys_values):.4g}, "
        f"delta median={statistics.median(deltas):.4g}"
    )
    if ratios:
        text += f", Yosys/DC median={statistics.median(ratios):.4g}"
    return text


def truncate(value: str, char_limit: int) -> str:
    if len(value) <= char_limit:
        return value
    return value[:char_limit] + "\n... [truncated]"


def escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


if __name__ == "__main__":
    main()
