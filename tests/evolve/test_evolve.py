from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


EVOLVE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVOLVE_DIR))

from revolution.evolve import evolve  # noqa: E402


class ParserTests(unittest.TestCase):
    def test_simulation_pass_and_final_time(self) -> None:
        text = """
        Time: 100 ps
        ----------------All tests PASS!-----------------
        Time: 429000 ps
        """
        self.assertEqual(evolve.parse_simulation(text), (True, 429000))

    def test_simulation_accepts_pass_without_exclamation(self) -> None:
        self.assertEqual(
            evolve.parse_simulation("All tests PASS\nTime: 12300 ps\n"),
            (True, 12300),
        )

    def test_area_power_and_met_slack(self) -> None:
        self.assertEqual(evolve.parse_area("Total cell area: 17810.817878"), 17810.817878)
        self.assertAlmostEqual(
            evolve.parse_power_mw("Total Power = 6.373e-04 (100.00%)"), 0.6373
        )
        self.assertEqual(evolve.parse_slack("slack (MET) 2.25"), (2.25, True))

    def test_violated_slack_is_not_clean(self) -> None:
        self.assertEqual(
            evolve.parse_slack("slack (VIOLATED) -0.42"), (-0.42, False)
        )
        self.assertEqual(
            evolve.parse_slack("slack (VIOLATED) 0.42"), (-0.42, False)
        )

    def test_period_parser(self) -> None:
        self.assertEqual(evolve.parse_periods("3,4.5,7"), [3.0, 4.5, 7.0])

    def test_optional_matplotlib_error_is_not_reported(self) -> None:
        text = (
            "ModuleNotFoundError: No module named 'matplotlib'\n"
            "slack (VIOLATED) -0.003\n"
        )
        self.assertNotIn("matplotlib", evolve.compact_log(text))
        self.assertIn("slack", evolve.compact_log(text))


class ArtifactTests(unittest.TestCase):
    def test_testbench_period_is_changed_only_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.sv"
            target = root / "target.sv"
            source.write_text("`define CYCLE 10\n`define OTHER 10\n")
            evolve.patch_testbench_period(source, target, 3.5)
            self.assertEqual(
                target.read_text(), "`define CYCLE 3.5\n`define OTHER 10\n"
            )

    def test_lowercase_cycle_period_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.sv"
            target = root / "target.sv"
            source.write_text("`define cycle 10.0\n")
            evolve.patch_testbench_period(source, target, 7)
            self.assertEqual(target.read_text(), "`define cycle 7\n")

    def test_all_benchmark_tasks_are_registered(self) -> None:
        self.assertEqual(set(evolve.TASKS), set(evolve.TASK_NAMES))
        for task in evolve.TASKS.values():
            self.assertEqual(task.spec, evolve.REPO_ROOT / task.name / "SPEC.md")
            self.assertEqual(task.baseline, evolve.REPO_ROOT / "VCS" / task.name / "initial.sv")
            self.assertEqual(task.testbench, evolve.REPO_ROOT / "VCS" / task.name / "00_TB")

    @unittest.skipUnless(os.environ.get("RTL_PPA_ICRTL_BENCHMARK"), "External benchmark payload is not included")
    def test_external_benchmark_artifacts_exist(self) -> None:
        root = Path(os.environ["RTL_PPA_ICRTL_BENCHMARK"])
        for name in evolve.TASK_NAMES:
            self.assertTrue((root / name / "SPEC.md").is_file())
            self.assertTrue((root / "VCS" / name / "initial.sv").is_file())
            self.assertTrue((root / "VCS" / name / "00_TB" / "test.sv").is_file())

    def test_q8_reserves_internal_combinational_io_budget(self) -> None:
        q8_sdc = evolve.render_sdc(evolve.TASKS["Q8_IOT"], 7.0)
        self.assertIn("$cycle * 0.1", q8_sdc)
        q3_sdc = evolve.render_sdc(evolve.TASKS["Q3_HC"], 7.0)
        self.assertIn("$cycle * 0.5", q3_sdc)

    def test_eval_result_round_trip(self) -> None:
        result = evolve.EvalResult(
            candidate_sha256="abc",
            period_ns=6.0,
            mode="proxy",
            rtl_pass=True,
            timing_clean=True,
            latency_ns=858.0,
            area=17800.0,
            at=15272400.0,
            score=-152.724,
        )
        self.assertEqual(
            evolve.EvalResult.from_dict(json.loads(json.dumps(result.to_dict()))),
            result,
        )

    def test_cached_prime_time_rounding_violation_is_migrated(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            candidate = root / "initial.sv"
            candidate.write_text("module TOP; endmodule\n")
            task = evolve.Task("test", root / "SPEC.md", candidate, root, "test/TOP")
            evaluator = evolve.CommercialEvaluator(task, {}, root / "cache")
            work = evaluator._work_dir(candidate, 6.0, "full")
            work.mkdir(parents=True)
            cached = evolve.EvalResult(
                evolve.sha256_file(candidate),
                6.0,
                "full",
                rtl_pass=True,
                gate_pass=True,
                timing_clean=False,
                latency_ns=10.0,
                area=20.0,
                slack_ns=-0.003,
                power_mw=2.0,
                at=200.0,
                failure_stage="timing",
            )
            evolve.atomic_write_json(work / "result.json", cached.to_dict())
            migrated = evaluator.evaluate(candidate, 6.0, full=True)
            self.assertTrue(migrated.valid_full)
            self.assertEqual(migrated.ppa, 400.0)
            self.assertIsNone(migrated.failure_stage)

    def test_search_period_uses_lowest_valid_at(self) -> None:
        slow = evolve.EvalResult(
            "a", 5.0, "full", rtl_pass=True, gate_pass=True, timing_clean=True,
            latency_ns=715.0, area=18400.0, power_mw=1.0, at=13156000.0, ppa=13156000.0
        )
        fast = evolve.EvalResult(
            "a", 6.0, "full", rtl_pass=True, gate_pass=True, timing_clean=True,
            latency_ns=858.0, area=14500.0, power_mw=1.0, at=12441000.0, ppa=12441000.0
        )
        self.assertEqual(evolve.choose_search_period({5.0: slow, 6.0: fast}), 6.0)

    def test_database_keeps_one_node_per_chain_index(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            connection = evolve.open_database(Path(temp) / "state.sqlite3")
            candidate = Path(temp) / "initial.sv"
            candidate.write_text("module TOP; endmodule\n")
            result = evolve.EvalResult(
                evolve.sha256_file(candidate), 6.0, "proxy", score=evolve.PENALTY
            )
            idea = {
                "title": "x",
                "idea": "y",
                "expected_effect": "z",
                "risk": "r",
            }
            evolve.record_node(connection, 0, 0, idea, candidate, "thread", result)
            evolve.record_node(connection, 0, 0, idea, candidate, "thread", result)
            count = connection.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
            self.assertEqual(count, 1)

    def test_finalist_selection_falls_back_to_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            connection = evolve.open_database(root / "state.sqlite3")
            baseline = root / "initial.sv"
            baseline.write_text("module TOP; endmodule\n")
            finalists = evolve.select_finalists(connection, 3, baseline)
            self.assertEqual(finalists, [(evolve.sha256_file(baseline), baseline)])


class CodexEventTests(unittest.TestCase):
    def test_thread_id_from_current_jsonl_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "events.jsonl"
            path.write_text(
                '{"type":"thread.started","thread_id":"1234-abcd"}\n'
                '{"type":"turn.completed"}\n'
            )
            self.assertEqual(evolve.parse_thread_id(path), "1234-abcd")


if __name__ == "__main__":
    unittest.main()
