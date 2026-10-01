from types import SimpleNamespace

from revolution.evolve import cvdp
from revolution.evolve import evolve


def test_remove_cvdp_option():
    argv = ["--cvdp-task", "task", "--baseline-only"]
    assert cvdp._remove_option(argv, "--cvdp-task") == "task"
    assert argv == ["--baseline-only"]


def test_combined_seed_keeps_all_target_modules():
    task = SimpleNamespace(
        targets=("rtl/a.sv", "rtl/b.sv"),
        seed={"rtl/a.sv": "module a; endmodule", "rtl/b.sv": "module b; endmodule"},
    )
    combined = cvdp._combined_seed(task)
    assert "CVDP target: rtl/a.sv" in combined
    assert "module a" in combined
    assert "CVDP target: rtl/b.sv" in combined
    assert "module b" in combined


def test_prompts_use_task_top_module():
    idea = {"title": "x", "idea": "x", "expected_effect": "x", "risk": "x"}
    prompt = evolve.make_root_prompt(idea, "spec", "rtl", "gcd_top")
    assert "Preserve the exact gcd_top interface" in prompt
