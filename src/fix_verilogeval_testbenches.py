#!/usr/bin/env python3
"""Fix VerilogEval testbenches for older/stricter Icarus Verilog parsers.

The affected testbenches reference tb_mismatch in the dumpvars initial block
before declaring tb_match/tb_mismatch. This script moves:

    wire tb_match;        // Verification
    wire tb_mismatch = ~tb_match;

above the dumpvars initial block.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
import re


TB_MATCH_RE = re.compile(r"^\s*wire\s+tb_match\s*;\s*(?://.*)?$")
TB_MISMATCH_RE = re.compile(r"^\s*wire\s+tb_mismatch\s*=\s*~tb_match\s*;\s*$")
INITIAL_BEGIN_RE = re.compile(r"^\s*initial\s+begin\b")


@dataclass(frozen=True)
class FixResult:
    path: Path
    changed: bool
    reason: str


def find_dump_initial(lines: list[str]) -> int | None:
    """Return the line index of the initial block containing dumpfile/dumpvars."""
    for index, line in enumerate(lines):
        if "$dumpfile" not in line and "$dumpvars" not in line:
            continue
        for candidate in range(index, -1, -1):
            if INITIAL_BEGIN_RE.match(lines[candidate]):
                return candidate
    return None


def find_tb_declaration_block(lines: list[str]) -> tuple[int, int] | None:
    """Return [start, end) for the tb_match/tb_mismatch declaration block."""
    for start, line in enumerate(lines):
        if not TB_MATCH_RE.match(line):
            continue
        for mismatch_index in range(start + 1, min(start + 4, len(lines))):
            if TB_MISMATCH_RE.match(lines[mismatch_index]):
                end = mismatch_index + 1
                while end < len(lines) and lines[end].strip() == "":
                    end += 1
                return start, end
    return None


def fix_text(text: str) -> tuple[str, bool, str]:
    lines = text.splitlines(keepends=True)
    dump_initial = find_dump_initial(lines)
    if dump_initial is None:
        return text, False, "no dumpvars initial block found"

    decl_block = find_tb_declaration_block(lines)
    if decl_block is None:
        return text, False, "no tb_match/tb_mismatch declaration block found"

    decl_start, decl_end = decl_block
    if decl_start < dump_initial:
        return text, False, "already fixed"

    block_lines = lines[decl_start:decl_end]
    remaining = lines[:decl_start] + lines[decl_end:]
    fixed_lines = remaining[:dump_initial] + block_lines + remaining[dump_initial:]
    return "".join(fixed_lines), True, "moved declarations before dumpvars block"


def fix_file(path: Path, *, write: bool, backup_suffix: str | None) -> FixResult:
    original = path.read_text(encoding="utf-8")
    fixed, changed, reason = fix_text(original)
    if changed and write:
        if backup_suffix:
            backup_path = path.with_name(path.name + backup_suffix)
            if not backup_path.exists():
                shutil.copy2(path, backup_path)
        path.write_text(fixed, encoding="utf-8")
    return FixResult(path=path, changed=changed, reason=reason)


def iter_testbenches(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(root.glob("*_test.sv"))


def parse_args(argv: list[str]) -> argparse.Namespace:
    default_root = (
        Path(__file__).resolve().parent
        / ".."
        / "data"
        / "bench"
        / "VerilogEval-Spec-to-RTL"
    )
    parser = argparse.ArgumentParser(
        description="Move tb_match/tb_mismatch declarations before dumpvars blocks."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[default_root],
        help="Benchmark directory or individual *_test.sv files to fix.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write changes in place. Without this flag, only reports planned fixes.",
    )
    parser.add_argument(
        "--backup-suffix",
        default=".bak",
        help="Backup suffix used with --write. Use an empty string to disable backups.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print the final summary.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    backup_suffix = args.backup_suffix or None

    files: list[Path] = []
    for path in args.paths:
        resolved = path.expanduser().resolve()
        if not resolved.exists():
            print(f"ERROR: path does not exist: {resolved}", file=sys.stderr)
            return 2
        files.extend(iter_testbenches(resolved))

    if not files:
        print("ERROR: no *_test.sv files found", file=sys.stderr)
        return 2

    results = [
        fix_file(path, write=args.write, backup_suffix=backup_suffix) for path in files
    ]
    changed = [result for result in results if result.changed]

    if not args.quiet:
        action = "fixed" if args.write else "would fix"
        for result in results:
            if result.changed:
                print(f"{action}: {result.path}")

    verb = "fixed" if args.write else "would fix"
    print(f"{verb} {len(changed)} of {len(results)} testbench file(s)")
    if not args.write and changed:
        print("Run again with --write to apply changes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
