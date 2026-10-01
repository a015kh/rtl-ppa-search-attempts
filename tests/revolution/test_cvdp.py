import os
import subprocess
import time
from pathlib import Path

from revolution.cvdp import (
    NativeCocotbRunner,
    harness_environment,
    normalize_native_row,
    parse_dotenv,
    translate_container_paths,
)
from revolution.cvdp_dcpt import (
    add_netlist_parameters,
    infer_clocks,
    module_parameter_clause,
    pareto_front,
    render_sdc,
)


def test_normalize_nonagentic_and_agentic_rows():
    nonagentic = normalize_native_row(
        {
            "id": "n",
            "categories": ["cid003"],
            "input": {"prompt": "optimize", "context": {"rtl/a.sv": "old"}},
            "output": {"context": {"rtl/a.sv": "new"}},
            "harness": {"files": {"src/.env": "TOPLEVEL=a"}},
        }
    )
    assert nonagentic["seed"] == {"rtl/a.sv": "new"}
    assert nonagentic["targets"] == ["rtl/a.sv"]

    agentic = normalize_native_row(
        {
            "id": "a",
            "prompt": "repair",
            "context": {"rtl/a.sv": "old"},
            "patch": {"rtl/a.sv": "new"},
            "harness": {"src/.env": "TOPLEVEL=a", "src/cache.pyc": None},
        }
    )
    assert not agentic["heavy"]
    assert "src/cache.pyc" not in agentic["harness"]


def test_dotenv_and_container_path_translation(tmp_path: Path):
    workspace = tmp_path.resolve()
    (workspace / "src").mkdir()
    (workspace / "runtime").mkdir()
    (workspace / "src" / ".env").write_text(
        "VERILOG_SOURCES=/code/rtl/a.sv\nPYTHONPATH=/src\nSIM=icarus\n"
    )
    assert parse_dotenv("A = one\n# x\nB='two'") == {"A": "one", "B": "two"}
    translated = translate_container_paths("/code/a /src/b /rundir/c", workspace)
    assert str(workspace / "code" / "a") in translated
    env = harness_environment(workspace)
    assert env["VERILOG_SOURCES"] == str(workspace / "code" / "rtl" / "a.sv")
    assert env["PYTHONPATH"].startswith(str(workspace / "runtime"))


def test_multiclock_sdc_and_clock_detection():
    clocks = infer_clocks("await Clock(dut.clk_b, 10, unit='ns').start()", ["clk_a", "clk_b", "data"])
    assert clocks == ["clk_a", "clk_b"]
    sdc = render_sdc(5.0, clocks, ["clk_a", "clk_b", "data"], ["out"])
    assert sdc.count("create_clock -name cvdp_clk_") == 2
    assert "set_input_delay 1.000000" in sdc
    assert "data" in sdc


def test_gate_netlist_recovers_elaborated_top_parameters():
    rtl = "module top #(parameter WIDTH = 4, parameter N = WIDTH + 1) (input x); endmodule"
    clause = module_parameter_clause(rtl, "top")
    netlist = add_netlist_parameters("module top ( x ); input x; endmodule", "top", clause)
    assert "module top #(parameter WIDTH = 4, parameter N = WIDTH + 1) (" in netlist


def test_native_runner_timeout_kills_descendant_process(tmp_path: Path):
    workspace = tmp_path.resolve()
    for directory in ("code", "src", "rundir"):
        (workspace / directory).mkdir()
    child_pid = workspace / "child.pid"
    (workspace / "docker-compose.yml").write_text(
        "services:\n"
        "  test:\n"
        "    command: >-\n"
        f"      python -c 'import os,time; p=os.fork(); "
        f"open(\"{child_pid}\",\"w\").write(str(p)); time.sleep(60)'\n",
        encoding="utf-8",
    )

    result = NativeCocotbRunner(timeout_seconds=1, sandbox=False).run(workspace)

    assert result.returncode == 124
    assert "Harness timed out" in result.stderr
    pid = int(child_pid.read_text(encoding="utf-8"))
    for _ in range(20):
        try:
            state = Path(f"/proc/{pid}/stat").read_text().split()[2]
        except FileNotFoundError:
            break
        if state == "Z":
            break
        time.sleep(0.05)
    else:
        subprocess.run(["kill", "-KILL", str(pid)], check=False)
        raise AssertionError(f"descendant process {pid} survived harness timeout")




def test_pareto_excludes_constraint_violations_and_dominated_points():
    def point(clock, area, power, constraint=True):
        return {
            "clock_ns": clock,
            "success": True,
            "gate_passed": True,
            "constraint_met": constraint,
            "metrics": {"area": area, "power": power},
        }

    front = pareto_front(
        [
            point(5, 10, 10),
            point(5, 12, 12),
            point(3, 20, 8),
            point(1, 1, 1, False),
        ]
    )
    assert [(item["clock_ns"], item["metrics"]["area"]) for item in front] == [
        (3, 20),
        (5, 10),
    ]
