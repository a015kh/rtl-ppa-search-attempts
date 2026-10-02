"""Fabricated completed measurements exercise the recovered test-data stages."""

from __future__ import annotations

import copy
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

from revolution import test_dataset_preparation as prep


def synthetic_result(name="tiny", area=12.5):
    return {
        "Name": name, "RTL": "module tiny(input a, output y); assign y=a; endmodule",
        "Syn": [{
            "SDC": "# Clock: clk\nset cycle_clk 2.0\n", "Has Clock": True,
            "Clock": 2.0, "Total Area": area, "Delay": 0.5, "WNS": -0.12,
            "TNS": -0.24, "Total Power": 0.3, "synth": "1 INV, 1 BUF",
            "area": "Total area is 12.5", "delay": "Total delay is 0.5",
            "static_power": "Total static power is 0.05 mW.",
            "switching_power": "Total dynamic power is 0.25 mW.",
            "synthesis_time": 2.0, "netlist": "excluded synthetic netlist",
        }],
    }


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="test-ppa-preparation-")
        self.root = Path(self.temp.name)
        self.config_path = self.root / "experiment.json"

    def tearDown(self):
        self.temp.cleanup()

    def write_json(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def config(self, shards, output="prepared"):
        self.write_json("experiment.json", {"output_root": output, "shards": shards})
        return prep.load_config(self.config_path)

    def legacy_spec(self, values=None, name="test.jsonl"):
        self.write_json("results.json", [synthetic_result()] if values is None else values)
        return {"name": name, "policy": "legacy_result_json", "inputs": ["results.json"]}

    def rows(self, name="test.jsonl", output="prepared"):
        return [json.loads(line) for line in (self.root / output / name).read_text(encoding="utf-8").splitlines()]

    def reports(self, directory, top="tiny", area=12.5, unit="uW"):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{top}_area.log").write_text(f"Total cell area: {area}\nDate : Wed Jan 01 00:00:03 2025\n", encoding="utf-8")
        (directory / f"{top}_qor.log").write_text("Design WNS: -0.12 TNS: -0.24\n", encoding="utf-8")
        (directory / f"{top}_timing.log").write_text("Point Incr Path\n-----------\nU1/Y (INV) 0.20 0.20 r\nU2/Y (BUF) 0.30 0.50 r\ndata arrival time 0.50\n", encoding="utf-8")
        (directory / f"{top}_cell.log").write_text("Cell Reference Area\nU1 INV 0.30\nU2 BUF 0.40\nTotal\n", encoding="utf-8")
        (directory / f"{top}_power.log").write_text("Total Dynamic Power = 2.0 mW\nCell Leakage Power = 1.0 mW\ncombinational 1.0 1.0 1000000000.0 3.0\nTotal 3.0\nEnd\n", encoding="utf-8")
        (directory / f"{top}_syn.v").write_text("module tiny(input a, output y); assign y=a; endmodule\n", encoding="utf-8")
        return f"Power Units : {unit}\nTotal Power = 300\nCell Internal Power = 100\nNet Switching Power = 150\nCell Leakage Power = 50\n"

    def report_spec(self, area=12.5, power_source="primetime", stem="tiny"):
        dc = self.root / "completed/dc/reports"
        pt = self.root / "completed/pt/reports"
        pt.mkdir(parents=True)
        power = self.reports(dc, top=stem, area=area)
        (pt / f"{stem}.report_power.rpt").write_text(power, encoding="utf-8")
        rtl = self.root / "completed/tiny.sv"
        rtl.write_text(synthetic_result()["RTL"], encoding="utf-8")
        tcl = self.root / "completed/design_dc.tcl"
        tcl.write_text(f"elaborate {stem}\ncurrent_design {stem}\n", encoding="utf-8")
        sdc = self.root / "completed/design.sdc"
        sdc.write_text(synthetic_result()["Syn"][0]["SDC"], encoding="utf-8")
        log = self.root / "completed/command.log"
        log.write_text("#@ # Date: Wed Jan 01 00:00:01 2025\n", encoding="utf-8")
        self.write_json("jobs.json", [{"name": "tiny", "rtl": "completed/tiny.sv", "top_module": "tiny", "dc_tcl": "completed/design_dc.tcl", "sdc": "completed/design.sdc", "dc_report_dir": "completed/dc/reports", "pt_report_dir": "completed/pt/reports", "command_log": "completed/command.log"}])
        return {"name": "test.jsonl", "policy": "dcpt_inputs_reports", "jobs": "jobs.json", "power_source": power_source}

    def resynthesis_spec(self, cases=None):
        cases = cases or [("good", "success", "success", 12.5)]
        work = self.root / "resynthesis"
        work.mkdir()
        manifest = work / "manifest.sqlite3"
        with closing(sqlite3.connect(manifest)) as db:
            db.execute("CREATE TABLE jobs (job_id TEXT, dc_status TEXT, pt_status TEXT, top_module TEXT, sdc TEXT, rtl TEXT, design_name TEXT, dc_elapsed REAL)")
            db.execute("CREATE TABLE source_rows (shard TEXT, line_number INTEGER, job_id TEXT)")
            for index, (job_id, dc_status, pt_status, area) in enumerate(cases, 1):
                if job_id == "unmapped":
                    continue
                db.execute("INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (job_id, dc_status, pt_status, "tiny", "set cycle_vclk 2.0", synthetic_result()["RTL"], "tiny", 2.0))
                db.execute("INSERT INTO source_rows VALUES (?, ?, ?)", ("source.jsonl", index, job_id))
                if area is not None:
                    directory = work / "jobs" / job_id[:2] / job_id
                    power = self.reports(directory / "dc/report", area=area)
                    (directory / "pt/report").mkdir(parents=True)
                    (directory / "pt/report/tiny.report_power.rpt").write_text(power, encoding="utf-8")
            db.commit()
        (self.root / "source.jsonl").write_text("\n".join(json.dumps({"has_clock": False, "clock": 2.0, "design_name": "old"}) for _ in cases) + "\n", encoding="utf-8")
        return {"name": "test.jsonl", "policy": "resynthesis_success", "input": "source.jsonl", "manifest": "resynthesis/manifest.sqlite3", "work_root": "resynthesis"}

    def test_legacy_flattening_exact_fields_and_multiple_points(self):
        first = synthetic_result()
        second = copy.deepcopy(first["Syn"][0])
        second["Clock"] = 3.0
        second["synthesis_time"] = -2.0  # Historical time labels are not repaired.
        first["Syn"].append(second)
        config = self.config([self.legacy_spec([first])])
        audit = prep.build_command(config)["shards"]["test.jsonl"]
        self.assertEqual(self.rows(), prep.extract_data_v3(first))
        self.assertEqual([row["clock"] for row in self.rows()], [2.0, 3.0])
        self.assertEqual(self.rows()[1]["synthesis_time"], -2.0)
        self.assertNotIn("netlist", self.rows()[0])
        self.assertEqual((audit["input_designs"], audit["input_points"], audit["accepted"], audit["skipped"]), (1, 2, 2, 0))

    def test_legacy_uses_last_numeric_explanation_token(self):
        result = synthetic_result()
        result["Syn"][0]["static_power"] = "0.01 + 0.02 = 3.0000e-02 mW."
        config = self.config([self.legacy_spec([result])])
        prep.build_command(config)
        self.assertEqual(self.rows()[0]["static_power_value"], 0.03)

    def test_legacy_zero_labels_are_not_filtered(self):
        config = self.config([self.legacy_spec([synthetic_result(area=0.0)])])
        prep.build_command(config)
        self.assertEqual(self.rows()[0]["area_value"], 0.0)

    def test_legacy_preserves_configured_file_order(self):
        self.write_json("one.json", [synthetic_result("one")])
        self.write_json("two.json", [synthetic_result("two")])
        spec = {"name": "test.jsonl", "policy": "legacy_result_json", "inputs": ["two.json", "one.json"]}
        prep.build_command(self.config([spec]))
        self.assertEqual([row["design_name"] for row in self.rows()], ["two", "one"])

    def test_missing_explanation_number_fails_without_partial_output(self):
        result = synthetic_result()
        result["Syn"][0]["static_power"] = "N/A"
        config = self.config([self.legacy_spec([result])])
        with self.assertRaisesRegex(prep.TestPreparationError, "conversion was not completed"):
            prep.build_command(config)
        self.assertFalse(config["output_root"].exists())
        self.assertFalse(list(self.root.glob(".prepared.staging.*")))

    def test_report_policy_recovers_ppa_clock_explanations_and_elapsed(self):
        config = self.config([self.report_spec()])
        audit = prep.build_command(config)["shards"]["test.jsonl"]
        row = self.rows()[0]
        self.assertEqual((row["area_value"], row["delay_value"], row["total_power_value"]), (12.5, 0.5, 0.3))
        self.assertEqual((row["static_power_value"], row["switching_power_value"]), (0.05, 0.25))
        self.assertEqual((row["has_clock"], row["clock"], row["synthesis_time"]), (True, 2.0, 2.0))
        self.assertEqual(audit["power_sources"], {"primetime": 1})
        self.assertIn("After synthesis", row["synth"])

    def test_report_tcl_top_has_historical_stem_priority(self):
        config = self.config([self.report_spec(stem="actual")])
        prep.build_command(config)
        self.assertEqual(self.rows()[0]["area_value"], 12.5)

    def test_report_zero_area_is_audited_and_excluded(self):
        config = self.config([self.report_spec(area=0.0)])
        audit = prep.build_command(config)["shards"]["test.jsonl"]
        self.assertEqual(self.rows(), [])
        self.assertEqual(audit["reasons"], {"zero_area": 1})
        self.assertEqual(audit["input_points"], audit["accepted"] + audit["skipped"])

    def test_missing_netlist_is_audited(self):
        spec = self.report_spec()
        (self.root / "completed/dc/reports/tiny_syn.v").unlink()
        audit = prep.build_command(self.config([spec]))["shards"]["test.jsonl"]
        self.assertEqual(audit["reasons"], {"missing_required_report_or_netlist": 1})

    def test_missing_numeric_qor_is_audited(self):
        spec = self.report_spec()
        (self.root / "completed/dc/reports/tiny_qor.log").write_text("No result\n", encoding="utf-8")
        audit = prep.build_command(self.config([spec]))["shards"]["test.jsonl"]
        self.assertEqual(audit["reasons"], {"missing_required_numeric_result": 1})

    def test_primetime_policy_never_uses_dc_fallback(self):
        spec = self.report_spec()
        (self.root / "completed/pt/reports/tiny.report_power.rpt").unlink()
        audit = prep.build_command(self.config([spec]))["shards"]["test.jsonl"]
        self.assertEqual(audit["reasons"], {"missing_power_report": 1})

    def test_explicit_mixed_policy_recovers_dc_power_fallback(self):
        spec = self.report_spec(power_source="mixed")
        (self.root / "completed/pt/reports/tiny.report_power.rpt").unlink()
        audit = prep.build_command(self.config([spec]))["shards"]["test.jsonl"]
        row = self.rows()[0]
        self.assertEqual(audit["power_sources"], {"design_compiler": 1})
        self.assertEqual((row["total_power_value"], row["static_power_value"], row["switching_power_value"]), (3.0, 1.0, 2.0))

    def test_optional_command_log_does_not_fabricate_time(self):
        spec = self.report_spec()
        (self.root / "completed/command.log").unlink()
        prep.build_command(self.config([spec]))
        self.assertIsNone(self.rows()[0]["synthesis_time"])

    def test_resynthesis_success_status_and_collection_filter(self):
        cases = [("good", "success", "success", 12.5), ("dc_bad", "failed", "success", 12.5), ("pt_bad", "success", "failed", 12.5), ("unmapped", "success", "success", None), ("missing", "success", "success", None), ("zero", "success", "success", 0.0)]
        spec = self.resynthesis_spec(cases)
        before = hashlib.sha256((self.root / spec["manifest"]).read_bytes()).hexdigest()
        audit = prep.build_command(self.config([spec]))["shards"]["test.jsonl"]
        after = hashlib.sha256((self.root / spec["manifest"]).read_bytes()).hexdigest()
        self.assertEqual(before, after)
        self.assertEqual((audit["input_points"], audit["accepted"], audit["skipped"]), (6, 2, 4))
        self.assertEqual(audit["reasons"], {"dc_not_success": 1, "pt_not_success": 1, "unmapped_source_row": 1, "result_collection_error": 1})
        self.assertEqual([row["area_value"] for row in self.rows()], [12.5, 0.0])
        self.assertEqual(self.rows()[0]["design_name"], "tiny")
        self.assertNotIn("netlist", self.rows()[0])

    def test_resynthesis_no_paths_retains_zero_delay(self):
        spec = self.resynthesis_spec()
        (self.root / "resynthesis/jobs/go/good/dc/report/tiny_timing.log").write_text("No paths.\n", encoding="utf-8")
        prep.build_command(self.config([spec]))
        self.assertEqual(self.rows()[0]["delay_value"], 0.0)
        self.assertIn("following gates: none", self.rows()[0]["delay"])

    def test_resynthesis_unit_conversion_and_compact_schema(self):
        spec = self.resynthesis_spec()
        prep.build_command(self.config([spec]))
        row = self.rows()[0]
        self.assertEqual(set(row), {"SDC", "has_clock", "clock", "area_value", "delay_value", "wns", "tns", "total_power_value", "static_power_value", "switching_power_value", "synth", "area", "delay", "static_power", "switching_power", "synthesis_time", "design_name", "rtl"})
        self.assertEqual((row["total_power_value"], row["static_power_value"], row["switching_power_value"]), (0.3, 0.05, 0.25))

    def test_manifest_schema_is_required_before_output_creation(self):
        spec = self.resynthesis_spec()
        with closing(sqlite3.connect(self.root / spec["manifest"])) as db:
            db.execute("DROP TABLE source_rows")
            db.commit()
        with self.assertRaisesRegex(prep.TestPreparationError, "missing required columns"):
            prep.preflight(self.config([spec]))

    def test_preflight_reports_explicit_offline_dispatch(self):
        config = self.config([self.legacy_spec()])
        self.assertEqual(prep.preflight(config), {"status": "ready", "shards": 1, "report_jobs": 0, "offline": True})
        self.assertFalse(config["output_root"].exists())

    def test_relative_paths_resolve_from_config_directory(self):
        config = self.config([self.legacy_spec()])
        self.assertEqual(config["shards"][0]["inputs"], [self.root / "results.json"])
        self.assertEqual(config["output_root"], self.root / "prepared")

    def test_output_existing_directory_is_never_replaced(self):
        config = self.config([self.legacy_spec()])
        config["output_root"].mkdir()
        (config["output_root"] / "keep.txt").write_text("keep", encoding="utf-8")
        with self.assertRaisesRegex(prep.TestPreparationError, "already exists"):
            prep.build_command(config)
        self.assertEqual((config["output_root"] / "keep.txt").read_text(), "keep")

    def test_output_cannot_overlap_input_or_reports(self):
        spec = self.report_spec()
        with self.assertRaisesRegex(prep.TestPreparationError, "overlaps"):
            prep.preflight(self.config([spec], output="completed/dc/reports/new-data"))

    def test_output_cannot_be_inside_public_source(self):
        config = self.config([self.legacy_spec()])
        config["output_root"] = Path(prep.__file__).resolve().parents[2] / "new-test-payload"
        with self.assertRaisesRegex(prep.TestPreparationError, "outside"):
            prep.preflight(config)

    def test_invalid_policies_filenames_and_duplicate_shards(self):
        for update in ({"policy": "unknown"}, {"name": "../test.jsonl"}, {"name": "test.txt"}):
            with self.subTest(update=update):
                spec = self.legacy_spec()
                spec.update(update)
                with self.assertRaises(prep.TestPreparationError):
                    self.config([spec])
        spec = self.legacy_spec()
        with self.assertRaisesRegex(prep.TestPreparationError, "Duplicate"):
            self.config([spec, spec])

    def test_unknown_config_fields_and_implicit_power_are_rejected(self):
        spec = self.report_spec()
        del spec["power_source"]
        with self.assertRaisesRegex(prep.TestPreparationError, "explicitly"):
            self.config([spec])
        spec = self.legacy_spec()
        spec["include_failed"] = True
        with self.assertRaisesRegex(prep.TestPreparationError, "Unknown"):
            self.config([spec])

    def test_missing_explicit_input_is_preflight_failure(self):
        spec = self.legacy_spec()
        (self.root / "results.json").unlink()
        with self.assertRaisesRegex(prep.TestPreparationError, "Required input"):
            prep.preflight(self.config([spec]))

    def test_audit_checksum_and_counts_match_written_rows(self):
        config = self.config([self.legacy_spec()])
        summary = prep.build_command(config)
        saved = json.loads((config["output_root"] / "audit.json").read_text())
        self.assertEqual(saved, summary)
        audit = saved["shards"]["test.jsonl"]
        self.assertEqual(audit["output_sha256"], hashlib.sha256((config["output_root"] / "test.jsonl").read_bytes()).hexdigest())
        self.assertNotIn(str(self.root), json.dumps(saved))

    def test_launcher_help_and_execution_from_another_directory(self):
        script = Path(prep.__file__).resolve().parents[2] / "scripts/prepare_test_dataset.py"
        config = self.config([self.legacy_spec()])
        for arguments in (["--help"], ["preflight", "--config", str(self.config_path)], ["build", "--config", str(self.config_path)]):
            result = subprocess.run([sys.executable, str(script), *arguments], cwd=self.root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(config["output_root"].is_dir())

    def test_cli_requires_configuration(self):
        with self.assertRaises(SystemExit) as caught:
            prep.main(["build"])
        self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
