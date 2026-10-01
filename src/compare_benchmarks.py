#!/usr/bin/env python3
"""
compare_benchmarks.py

Compares the RTL-OPT benchmark against RTLLM and VerilogEval-Spec-to-RTL benchmarks.
Classifies each RTL-OPT problem as:
  - "Original"  : same design exists in another benchmark (exact name match)
  - "Variant"   : a related/modified version of a problem in another benchmark (partial match)
  - "Other"     : unique to RTL-OPT, no counterpart found

Also runs iverilog equivalence checking between the suboptimal RTL-OPT design and its
expert-optimized _ref version to verify functional correctness.

Usage:
    python compare_benchmarks.py [options]

Options:
    --rtlopt-dir        Path to RTL-OPT benchmark/ directory
    --rtllm-dir         Path to RTLLM benchmark directory
    --verilogeval-dir   Path to VerilogEval-Spec-to-RTL directory
    --output            Output path for markdown report
"""

import argparse
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MatchResult:
    rtlopt_design: str           # RTL-OPT directory name (suboptimal design)
    rtlopt_file: Optional[str]   # Path to RTL-OPT source file
    ref_file: Optional[str]      # Path to the _ref (expert) source file
    matched_problem: str         # e.g. "Prob008_comparator_4bit" or "—"
    source_benchmark: str        # "RTLLM", "VerilogEval", or "None"
    match_type: str              # "Original", "Variant", or "Other"
    iverilog_result: str         # "PASS", "FAIL", "COMPILE_ERROR", "NO_REF", etc.
    iverilog_details: str = ""   # Extra details from simulation


# ---------------------------------------------------------------------------
# Parameter resolution
# ---------------------------------------------------------------------------

def parse_params(verilog_text: str) -> dict[str, int]:
    """Extract default integer parameter values: {name: value}."""
    params: dict[str, int] = {}
    for m in re.finditer(r"\bparameter\s+(?:\w+\s+)?(\w+)\s*=\s*(\d+)", verilog_text):
        try:
            params[m.group(1)] = int(m.group(2))
        except ValueError:
            pass
    return params


def resolve_width(width_str: str, params: dict[str, int]) -> str:
    """
    Substitute known parameter values in width strings like '[N-1:0]' → '[31:0]'.
    Falls back to original string if arithmetic evaluation fails.
    """
    if not width_str:
        return ""
    result = width_str
    for name, val in params.items():
        result = re.sub(r"\b" + re.escape(name) + r"\b", str(val), result)

    def eval_range(m: re.Match) -> str:
        try:
            hi = eval(m.group(1), {"__builtins__": {}})   # noqa: S307
            lo = eval(m.group(2), {"__builtins__": {}})   # noqa: S307
            return f"[{int(hi)}:{int(lo)}]"
        except Exception:
            return m.group(0)

    result = re.sub(r"\[([0-9\s+\-*/()\[\]]+):([0-9\s+\-*/()\[\]]+)\]", eval_range, result)
    return result


def width_bits(width_str: str) -> int:
    """Return bit width from a concrete string like '[7:0]' → 8, '' → 1."""
    if not width_str:
        return 1
    m = re.search(r"\[(\d+)\s*:\s*(\d+)\]", width_str)
    if m:
        return abs(int(m.group(1)) - int(m.group(2))) + 1
    return 1


# ---------------------------------------------------------------------------
# Verilog port parsing
# ---------------------------------------------------------------------------

def parse_ports(verilog_text: str) -> dict:
    """
    Parse a Verilog module and return:
      'name', 'inputs', 'outputs', 'inouts', 'has_clk', 'has_rst'
    Handles: ANSI port declarations, parameterized widths (substitutes defaults),
    and old-style port lists (parses the module body).
    """
    result = {
        "name": "",
        "inputs": [],
        "outputs": [],
        "inouts": [],
        "has_clk": False,
        "has_rst": False,
    }

    # Strip comments for clean parsing
    clean = re.sub(r"//[^\n]*", "", verilog_text)
    clean = re.sub(r"/\*.*?\*/", "", clean, flags=re.DOTALL)

    m = re.search(r"\bmodule\s+(\w+)", clean)
    if not m:
        return result
    result["name"] = m.group(1)

    params = parse_params(clean)

    def register_port(direction: str, width_raw: str, names_str: str) -> None:
        width = resolve_width(width_raw.strip(), params)
        # Skip multi-dimensional array ports (they have an extra [..] after the name)
        names = [
            n.strip() for n in names_str.split(",")
            if n.strip() and re.match(r"^\w+$", n.strip())
        ]
        for name in names:
            if direction == "input":
                result["inputs"].append((name, width))
                if re.search(r"clk|clock", name, re.I):
                    result["has_clk"] = True
                if re.search(r"rst|reset", name, re.I):
                    result["has_rst"] = True
            elif direction == "output":
                result["outputs"].append((name, width))
            else:
                result["inouts"].append((name, width))

    module_start = clean.find("module ")
    if module_start == -1:
        return result

    # --- Attempt ANSI-style parsing ---
    after_mod = clean[module_start:]
    after_no_params = re.sub(r"#\s*\(.*?\)", "", after_mod, count=1, flags=re.DOTALL)
    pl_m = re.search(r"\(([^)]*)\)", after_no_params, re.DOTALL)

    if pl_m:
        port_str = pl_m.group(1)
        # Split port declarations on direction keywords, then parse each chunk.
        # E.g.: "input [3:0] A, B,\n    output eq, gt, lt" becomes two sections.
        segments = re.split(r"\b(input|output|inout)\b", port_str)
        # segments = [pre, dir1, body1, dir2, body2, ...]
        i = 1
        while i + 1 < len(segments):
            direction = segments[i].strip()
            body = segments[i + 1]
            i += 2
            # Strip modifiers (reg, wire, signed, logic) — may be multiple
            body = re.sub(r"^\s*(reg|wire|signed|logic)(\s+(reg|wire|signed|logic))*\s+", " ", body.strip()).strip()
            # Check if a width bracket is present
            width_m = re.match(r"(\[[\[\]\w\s+\-*:]+\])\s*", body)
            if width_m:
                width_raw = width_m.group(1)
                names_str = body[width_m.end():]
            else:
                width_raw = ""
                names_str = body
            # names_str may end with trailing comma/whitespace — clean up
            names_str = names_str.rstrip(", \t\n")
            register_port(direction, width_raw, names_str)

    # --- Fallback: old-style (non-ANSI) — parse body declarations ---
    if not result["inputs"] and not result["outputs"]:
        first_semi = clean.find(";", module_start)
        if first_semi != -1:
            body = clean[first_semi + 1:]
            for m2 in re.finditer(
                r"\b(input|output|inout)\s+(?:reg\s+)?(?:signed\s+)?"
                r"([\[\]\w\s+\-*:]+?)?\s+(\w+(?:\s*,\s*\w+)*)\s*;",
                body,
            ):
                register_port(m2.group(1), m2.group(2) or "", m2.group(3))

    return result



# ---------------------------------------------------------------------------
# Equivalence testbench generation
# ---------------------------------------------------------------------------

COMB_TB = """\
`timescale 1ns/1ps
module eq_tb;
{input_decls}
{dut_out_decls}
{ref_out_decls}

  {dut_mod} dut (
{dut_conn}
  );
  {ref_mod} ref_inst (
{ref_conn}
  );

  integer i, errors;
  initial begin
    errors = 0;
    for (i = 0; i < 2000; i = i + 1) begin
{apply}
      #1;
{check}
    end
    if (errors == 0) $display("ALL TESTS PASS! (0 mismatches in 2000 random vectors)");
    else             $display("FAIL: %0d mismatches", errors);
    $finish;
  end
endmodule
"""

SEQ_TB = """\
`timescale 1ns/1ps
module eq_tb;
  reg clk;
{rst_decl}
{input_decls}
{dut_out_decls}
{ref_out_decls}

  initial clk = 0;
  always #5 clk = ~clk;

  {dut_mod} dut (
{dut_conn}
  );
  {ref_mod} ref_inst (
{ref_conn}
  );

  integer i, errors;
  initial begin
    errors = 0;
{rst_init}
    @(posedge clk); #1;
{rst_release}
    for (i = 0; i < 500; i = i + 1) begin
{apply}
      @(posedge clk); #1;
{check}
    end
    if (errors == 0) $display("ALL TESTS PASS! (0 mismatches in 500 clock cycles)");
    else             $display("FAIL: %0d mismatches", errors);
    $finish;
  end
endmodule
"""


def generate_equiv_tb(dut_ports: dict, ref_ports: dict, tb_path: Path) -> bool:
    """Generate an equivalence testbench. Returns True on success."""
    dut_mod = dut_ports["name"]
    ref_mod = ref_ports["name"]

    if not dut_mod or not ref_mod:
        return False

    is_seq = dut_ports["has_clk"]

    # Clock/reset port names
    dut_clk = next((n for n, _ in dut_ports["inputs"] if re.search(r"clk|clock", n, re.I)), "clk")
    ref_clk = next((n for n, _ in ref_ports["inputs"] if re.search(r"clk|clock", n, re.I)), "clk")
    dut_rst = next((n for n, _ in dut_ports["inputs"] if re.search(r"rst|reset", n, re.I)), None)
    ref_rst = next((n for n, _ in ref_ports["inputs"] if re.search(r"rst|reset", n, re.I)), None)

    # Non-clock/reset data inputs
    def data_inputs(ports: dict) -> list:
        return [(n, w) for n, w in ports["inputs"]
                if not re.search(r"clk|clock|rst|reset", n, re.I)]

    dut_data_in = data_inputs(dut_ports)
    ref_data_in = data_inputs(ref_ports)
    n_in = min(len(dut_data_in), len(ref_data_in))
    paired_in = list(zip(dut_data_in[:n_in], ref_data_in[:n_in]))

    n_out = min(len(dut_ports["outputs"]), len(ref_ports["outputs"]))
    paired_out = list(zip(dut_ports["outputs"][:n_out], ref_ports["outputs"][:n_out]))

    if n_out == 0:
        return False

    # Build TB strings
    input_decls, apply_lines = [], []
    dut_out_decls, ref_out_decls, check_lines = [], [], []
    dut_conn, ref_conn = [], []
    bit_off = 0

    if is_seq:
        dut_conn.append(f"    .{dut_clk}(clk)")
        ref_conn.append(f"    .{ref_clk}(clk)")

    for (d_name, d_width), (r_name, r_width) in paired_in:
        bits = width_bits(d_width)
        wd = f"[{bits - 1}:0] " if bits > 1 else ""
        input_decls.append(f"  reg {wd}{d_name};")
        sl = f"$random" if bits == 1 else f"$random"
        apply_lines.append(f"      {d_name} = $random;")
        dut_conn.append(f"    .{d_name}({d_name})")
        ref_conn.append(f"    .{r_name}({d_name})")
        bit_off += bits

    for (d_name, d_width), (r_name, r_width) in paired_out:
        bits = width_bits(d_width)
        wd = f"[{bits - 1}:0] " if bits > 1 else ""
        dut_out_decls.append(f"  wire {wd}dut_{d_name};")
        ref_out_decls.append(f"  wire {wd}ref_{r_name};")
        dut_conn.append(f"    .{d_name}(dut_{d_name})")
        ref_conn.append(f"    .{r_name}(ref_{r_name})")
        check_lines.append(
            f"      if (dut_{d_name} !== ref_{r_name}) begin\n"
            f"        $display(\"MISMATCH {d_name} at i=%0d dut=%b ref=%b\","
            f" i, dut_{d_name}, ref_{r_name});\n"
            f"        errors = errors + 1;\n"
            f"      end"
        )

    if is_seq:
        if dut_rst:
            rst_decl = f"  reg {dut_rst};"
            rst_init = f"    {dut_rst} = 0;"
            rst_release = f"    {dut_rst} = 1; @(posedge clk); #1;"
            dut_conn.insert(1 if paired_in else 1, f"    .{dut_rst}({dut_rst})")
        else:
            rst_decl = rst_init = rst_release = ""
        if ref_rst:
            rst_fallback = dut_rst if dut_rst else "1'b1"
            ref_conn.insert(1, f"    .{ref_rst}({rst_fallback})")

        tb = SEQ_TB.format(
            rst_decl=rst_decl,
            input_decls="\n".join(input_decls),
            dut_out_decls="\n".join(dut_out_decls),
            ref_out_decls="\n".join(ref_out_decls),
            dut_mod=dut_mod,
            ref_mod=ref_mod,
            dut_conn=",\n".join(dut_conn),
            ref_conn=",\n".join(ref_conn),
            rst_init=rst_init,
            rst_release=rst_release,
            apply="\n".join(apply_lines),
            check="\n".join(check_lines),
        )
    else:
        tb = COMB_TB.format(
            input_decls="\n".join(input_decls),
            dut_out_decls="\n".join(dut_out_decls),
            ref_out_decls="\n".join(ref_out_decls),
            dut_mod=dut_mod,
            ref_mod=ref_mod,
            dut_conn=",\n".join(dut_conn),
            ref_conn=",\n".join(ref_conn),
            apply="\n".join(apply_lines),
            check="\n".join(check_lines),
        )

    tb_path.write_text(tb)
    return True


# ---------------------------------------------------------------------------
# iverilog verification
# ---------------------------------------------------------------------------

def find_rtl_source(design_dir: Path) -> Optional[Path]:
    """Find the first .sv or .v file in a directory."""
    for ext in (".sv", ".v"):
        files = sorted(design_dir.glob(f"*{ext}"))
        if files:
            return files[0]
    return None


def run_iverilog_verification(
    design_name: str,
    rtlopt_file: Path,
    ref_file: Path,
    tmpdir: Path,
) -> tuple[str, str]:
    """
    Auto-generate equivalence testbench and run iverilog.
    Returns (result_str, details).
    """
    dut_ports = parse_ports(rtlopt_file.read_text(errors="replace"))
    ref_ports = parse_ports(ref_file.read_text(errors="replace"))

    if not dut_ports["name"] or not ref_ports["name"]:
        return "PARSE_ERROR", "Could not parse module name"

    if not dut_ports["outputs"] or not ref_ports["outputs"]:
        return "PARSE_ERROR", (
            f"No parseable ports: dut_in={len(dut_ports['inputs'])} "
            f"dut_out={len(dut_ports['outputs'])} "
            f"ref_in={len(ref_ports['inputs'])} "
            f"ref_out={len(ref_ports['outputs'])}"
        )

    tb_path = tmpdir / f"{design_name}_eq_tb.v"
    if not generate_equiv_tb(dut_ports, ref_ports, tb_path):
        return "PARSE_ERROR", "Could not generate equivalence testbench"

    out_bin = tmpdir / f"{design_name}_sim"
    compile_cmd = ["iverilog", "-g2012", "-o", str(out_bin),
                   str(rtlopt_file), str(ref_file), str(tb_path)]

    try:
        cr = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return "TIMEOUT", "compile timed out"

    if cr.returncode != 0:
        err = (cr.stderr or "").strip()
        return "COMPILE_ERROR", (err[:400] + "..." if len(err) > 400 else err)

    try:
        sr = subprocess.run(["vvp", str(out_bin)], capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return "TIMEOUT", "simulation timed out"

    output = (sr.stdout + sr.stderr).strip()
    lo = output.lower()
    if "all tests pass" in lo:
        return "PASS", output[:200]
    if "fail" in lo or "mismatch" in lo:
        return "FAIL", output[:300]
    if sr.returncode == 0:
        return "PASS", f"(exit 0, no failure text) {output[:100]}"
    return "FAIL", output[:300]


# ---------------------------------------------------------------------------
# Benchmark loading
# ---------------------------------------------------------------------------

def load_rtlopt_designs(rtlopt_dir: Path) -> list[str]:
    return sorted(
        d.name for d in rtlopt_dir.iterdir()
        if d.is_dir() and not d.name.endswith("_ref")
    )


def load_benchmark_problems(problems_file: Path) -> list[str]:
    if not problems_file.exists():
        return []
    return [l.strip() for l in problems_file.read_text().splitlines() if l.strip()]


def problem_stem(name: str) -> str:
    return re.sub(r"^Prob\d+_", "", name)


# ---------------------------------------------------------------------------
# Matching logic
# ---------------------------------------------------------------------------

def find_best_match(
    rtlopt_name: str,
    rtllm_problems: list[str],
    verilogeval_problems: list[str],
) -> tuple[str, str, str]:
    """Returns (matched_problem, source_benchmark, match_type)."""
    all_probs = [(p, "RTLLM") for p in rtllm_problems] + \
                [(p, "VerilogEval") for p in verilogeval_problems]

    # Phase 1: exact name match
    for prob, src in all_probs:
        if problem_stem(prob) == rtlopt_name:
            return prob, src, "Original"

    # Phase 2: token overlap / substring match
    rtl_tokens = set(rtlopt_name.split("_"))
    best: Optional[tuple[str, str, str]] = None
    best_score = 0

    for prob, src in all_probs:
        stem = problem_stem(prob)
        stem_tokens = set(stem.split("_"))
        if rtlopt_name in stem or stem in rtlopt_name:
            score = 3 + len(rtl_tokens & stem_tokens)
        else:
            common = len(rtl_tokens & stem_tokens)
            score = common

        if score > best_score or (score == best_score and score > 0 and
                                   best and src == "RTLLM" and best[1] == "VerilogEval"):
            best_score = score
            best = (prob, src, "Variant")

    if best and best_score > 0:
        return best
    return "—", "None", "Other"


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def compare_benchmarks(
    rtlopt_dir: Path,
    rtllm_dir: Path,
    verilogeval_dir: Path,
    output_file: Path,
) -> list[MatchResult]:

    rtlopt_designs = load_rtlopt_designs(rtlopt_dir)
    rtllm_problems = load_benchmark_problems(rtllm_dir / "problems.txt")
    verilogeval_problems = load_benchmark_problems(verilogeval_dir / "problems.txt")

    print(f"RTL-OPT designs:      {len(rtlopt_designs)}")
    print(f"RTLLM problems:       {len(rtllm_problems)}")
    print(f"VerilogEval problems: {len(verilogeval_problems)}")
    print()

    results: list[MatchResult] = []

    with tempfile.TemporaryDirectory(prefix="rtlopt_compare_") as tmpdir:
        tmpdir_path = Path(tmpdir)

        for design_name in rtlopt_designs:
            design_dir = rtlopt_dir / design_name
            ref_dir = rtlopt_dir / f"{design_name}_ref"

            rtlopt_file = find_rtl_source(design_dir)
            ref_file = find_rtl_source(ref_dir) if ref_dir.exists() else None

            matched_problem, source_benchmark, match_type = find_best_match(
                design_name, rtllm_problems, verilogeval_problems
            )

            if rtlopt_file and ref_file:
                iv_result, iv_details = run_iverilog_verification(
                    design_name, rtlopt_file, ref_file, tmpdir_path
                )
            elif not rtlopt_file:
                iv_result, iv_details = "NO_SOURCE", "No .v/.sv source found"
            else:
                iv_result, iv_details = "NO_REF", "No _ref directory found"

            icon = {"PASS": "✅", "FAIL": "❌", "COMPILE_ERROR": "🔧",
                    "NO_REF": "—", "NO_SOURCE": "⚠️", "PARSE_ERROR": "🔍",
                    "TIMEOUT": "⏱"}.get(iv_result, "?")

            print(
                f"  {design_name:<30} | {match_type:<10} | {source_benchmark:<12} | "
                f"{matched_problem:<42} | {icon} {iv_result}"
            )

            results.append(MatchResult(
                rtlopt_design=design_name,
                rtlopt_file=str(rtlopt_file) if rtlopt_file else None,
                ref_file=str(ref_file) if ref_file else None,
                matched_problem=matched_problem,
                source_benchmark=source_benchmark,
                match_type=match_type,
                iverilog_result=iv_result,
                iverilog_details=iv_details,
            ))

    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(results: list[MatchResult], output_file: Path) -> None:
    by_type: dict[str, int] = {}
    by_bench: dict[str, int] = {}
    by_iv: dict[str, int] = {}
    for r in results:
        by_type[r.match_type] = by_type.get(r.match_type, 0) + 1
        by_bench[r.source_benchmark] = by_bench.get(r.source_benchmark, 0) + 1
        by_iv[r.iverilog_result] = by_iv.get(r.iverilog_result, 0) + 1

    lines = [
        "# RTL-OPT Benchmark Comparison Report\n",
        "Comparison of RTL-OPT designs against RTLLM and VerilogEval-Spec-to-RTL.\n"
        "Functional equivalence between each suboptimal design and its `_ref` version "
        "is verified with iverilog.\n",
        "## Summary\n",
        f"- **Total RTL-OPT designs**: {len(results)}\n",
        "### Classification\n",
        "| Match Type | Count |", "|---|---|",
    ]
    for mt in ("Original", "Variant", "Other"):
        lines.append(f"| {mt} | {by_type.get(mt, 0)} |")
    lines += [
        "",
        "### Source Benchmark\n",
        "| Benchmark | Count |", "|---|---|",
    ]
    for src in sorted(by_bench):
        lines.append(f"| {src} | {by_bench[src]} |")
    lines += [
        "",
        "### iverilog Equivalence Results\n",
        "| Result | Count |", "|---|---|",
    ]
    for res in ("PASS", "FAIL", "COMPILE_ERROR", "PARSE_ERROR", "NO_REF", "NO_SOURCE", "TIMEOUT"):
        if res in by_iv:
            lines.append(f"| {res} | {by_iv[res]} |")
    lines += ["", "## Detailed Results\n",
              "| RTL-OPT Design | Match Type | Source Benchmark | Matched Problem | "
              "iverilog Result | Notes |",
              "|---|---|---|---|---|---|"]

    for r in results:
        details = r.iverilog_details.replace("|", "\\|").replace("\n", " ").strip()[:200]
        lines.append(
            f"| `{r.rtlopt_design}` | **{r.match_type}** | {r.source_benchmark} | "
            f"`{r.matched_problem}` | {r.iverilog_result} | {details} |"
        )

    lines += ["", "## Classification Breakdown\n"]
    for match_type in ("Original", "Variant", "Other"):
        subset = [r for r in results if r.match_type == match_type]
        if not subset:
            continue
        lines.append(f"### {match_type} ({len(subset)} designs)\n")
        for r in subset:
            iv_icon = "✅" if r.iverilog_result == "PASS" else (
                "❌" if r.iverilog_result == "FAIL" else "—")
            lines.append(
                f"- `{r.rtlopt_design}` → `{r.matched_problem}` "
                f"({r.source_benchmark}) {iv_icon} {r.iverilog_result}"
            )
        lines.append("")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(lines))
    print(f"\nReport saved to: {output_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare RTL-OPT benchmark against RTLLM and VerilogEval."
    )
    parser.add_argument(
        "--rtlopt-dir",
        default="p-RTL-codegen/src/text2text_regression/examples/rtl-opt/benchmark",
    )
    parser.add_argument("--rtllm-dir", default="REvolution/data/bench/RTLLM")
    parser.add_argument(
        "--verilogeval-dir", default="REvolution/data/bench/VerilogEval-Spec-to-RTL"
    )
    parser.add_argument(
        "--output", default="REvolution/src/benchmark_comparison_report.md"
    )
    args = parser.parse_args()

    base = Path.cwd()

    def resolve(p: str) -> Path:
        pp = Path(p)
        return pp if pp.is_absolute() else base / pp

    rtlopt_dir = resolve(args.rtlopt_dir)
    rtllm_dir = resolve(args.rtllm_dir)
    verilogeval_dir = resolve(args.verilogeval_dir)
    output_file = resolve(args.output)

    for label, path in [("RTL-OPT dir", rtlopt_dir), ("RTLLM dir", rtllm_dir),
                         ("VerilogEval dir", verilogeval_dir)]:
        if not path.exists():
            print(f"ERROR: {label} not found: {path}", file=sys.stderr)
            sys.exit(1)

    print("=" * 80)
    print("RTL-OPT Benchmark Comparison")
    print("=" * 80)
    print()

    results = compare_benchmarks(rtlopt_dir, rtllm_dir, verilogeval_dir, output_file)
    generate_report(results, output_file)

    print("\n" + "=" * 80)
    print("CLASSIFICATION SUMMARY")
    print("=" * 80)
    by_type: dict[str, list[str]] = {}
    for r in results:
        by_type.setdefault(r.match_type, []).append(r.rtlopt_design)
    for mt in ("Original", "Variant", "Other"):
        designs = by_type.get(mt, [])
        print(f"\n{mt} ({len(designs)}):")
        for d in designs:
            print(f"  {d}")

    print("\n" + "=" * 80)
    print("IVERILOG EQUIVALENCE RESULTS")
    print("=" * 80)
    by_iv: dict[str, list[MatchResult]] = {}
    for r in results:
        by_iv.setdefault(r.iverilog_result, []).append(r)
    for res in ("PASS", "FAIL", "COMPILE_ERROR", "PARSE_ERROR", "NO_REF", "NO_SOURCE", "TIMEOUT"):
        items = by_iv.get(res, [])
        if items:
            print(f"\n{res} ({len(items)}):")
            for r in items:
                preview = r.iverilog_details.replace("\n", " ")[:100]
                print(f"  {r.rtlopt_design}" + (f": {preview}" if preview else ""))


if __name__ == "__main__":
    main()
