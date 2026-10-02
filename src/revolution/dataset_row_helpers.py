"""Exact JSONL and DC report-formatting helpers used by the historical builder."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterator
class ResynthesisError(RuntimeError):
    """An actionable preparation or execution failure."""


def jsonl_rows(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ResynthesisError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ResynthesisError(f"Expected object at {path}:{line_number}")
            yield line_number, row


def format_cell_reports(text: str) -> tuple[str, str]:
    """Convert a DC report_cell listing to the compact dataset text format."""
    cells: dict[str, dict[str, float | int]] = {}
    in_table = False
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if line.lstrip().startswith("Cell") and "Reference" in line and "Area" in line:
            in_table = True
            continue
        if not in_table:
            continue
        if line.lstrip().startswith("Total"):
            break
        match = re.match(r"\s*\S+\s+(\S+)\s+(?:\S+\s+)?([\d.]+)(?:\s|$)", line)
        if not match:
            continue
        reference, area_text = match.groups()
        area = float(area_text)
        if reference not in cells:
            cells[reference] = {"count": 0, "area": area}
        cells[reference]["count"] = int(cells[reference]["count"]) + 1

    if not cells:
        return (
            "After synthesis, this design has no cells.",
            "In total, we can compute 0 = 0.00\nThus, the total area is 0.00",
        )

    area_summaries = []
    equation_parts = []
    synth_parts = []
    total_area = 0.0
    for reference, info in cells.items():
        count = int(info["count"])
        area = float(info["area"])
        area_summaries.append(f"Area of {reference} is {area:.2f}")
        equation_parts.append(f"{count}*{area:.2f}")
        synth_parts.append(f"{count} {reference}")
        total_area += count * area

    synth = "After synthesis, this design has " + ", ".join(synth_parts) + "."
    area = (
        ". ".join(area_summaries)
        + ".  \nIn total, we can compute "
        + " + ".join(equation_parts)
        + f"  = {total_area:.2f}\nThus, the total area is {total_area:.2f}"
    )
    return synth, area


def format_timing_report(text: str) -> str:
    """Convert a DC report_timing path to the compact dataset text format."""
    if re.search(r"\bNo paths\.\s*$", text, re.I | re.M):
        return (
            "The critical path goes through the following gates: none. "
            "In total, we can compute 0 = 0.00\nThus, the total delay is 0.00"
        )
    number = re.compile(r"^[-+]?(?:\d+(?:\.\d+)?|\.\d+)$")
    tokens: list[str] = []
    started = False
    skip_separator = False
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not started:
            if line.strip().lower().startswith("point"):
                started = True
                skip_separator = True
            continue
        if skip_separator:
            skip_separator = False
            continue
        if "data arrival time" in line.lower():
            break
        tokens.extend(line.split())

    path: list[tuple[str, float]] = []
    label_parts: list[str] = []
    increment: float | None = None
    state = "label"
    index = 0
    while index < len(tokens):
        token = tokens[index]
        is_number = bool(number.match(token))
        if state == "label":
            if is_number:
                increment = float(token)
                state = "path"
            else:
                label_parts.append(token)
            index += 1
        elif state == "path":
            if is_number:
                state = "edge"
            index += 1
        else:
            if token.lower() in {"r", "f"}:
                index += 1
            label = " ".join(label_parts).strip()
            if label and increment is not None:
                path.append((label, increment))
            label_parts = []
            increment = None
            state = "label"

    if label_parts and increment is not None:
        path.append((" ".join(label_parts).strip(), increment))
    if not path:
        return ""

    labels = ", ".join(label for label, _ in path)
    delay_lines = ", ".join(f"Delay of {label} is {value:.2f}" for label, value in path)
    equation = " + ".join(f"{value:.2f}" for _, value in path)
    total = sum(value for _, value in path)
    return (
        f"The critical path goes through the following gates: {labels}. "
        f"{delay_lines}. In total, we can compute {equation} = {total:.2f}\n"
        f"Thus, the total delay is {total:.2f}"
    )
