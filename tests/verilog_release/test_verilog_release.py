"""A source-only release must never inherit measured/private row payloads."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from revolution.verilog_release import _forbidden_regex, _unsafe_rtl, build_release, file_sha256, load_config, preflight, verify_release


ROOT = Path(__file__).resolve().parents[2]
NOTICE_RTL = "// Copyright Example authors. Licensed under MIT.\r\nmodule demo(input a, output b);\r\n  assign b = a;\r\nendmodule\r\n"


def registry(**changes):
    return {"category": "example", "upstream_url": "https://example.org/rtl",
            "license": "MIT", "license_url": "https://example.org/LICENSE",
            "redistribution_status": "cleared", "allow_base": True, **changes}


def measured(rtl=NOTICE_RTL, kind="original", source="private-source-key", **changes):
    return {"rtl": rtl, "record_kind": kind, "source_dataset": source,
            "area_value": 123.4, "delay_value": 0.77, "total_power_value": 2.1,
            "SDC": "private constraint payload", "clock": 1.0,
            "design_name": "private-design-key", "record_id": "private-measurement-id",
            "source_artifact": "/private-input/archive/train.jsonl", "source_revision": "private-revision",
            "synth": "private synthesis report", **changes}


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.home = Path(self.temporary.name)

    def config(self, splits=None, sources=None, **extras):
        inputs = []
        for label, rows in (splits or {"train": [measured()]}).items():
            path = self.home / f"input-{label}.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            inputs.append({"split": label, "path": path.name})
        config = {"schema_version": 1, "inputs": inputs,
                  "sources": sources if sources is not None else {"private-source-key": registry()}, **extras}
        path = self.home / "private-config.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        return path

    def output(self):
        return self.home / "output"

    def test_exact_rtl_and_notices_preserved_and_every_private_measurement_field_omitted(self):
        config = self.config()
        before = (self.home / "input-train.jsonl").read_bytes()
        manifest = build_release(config, self.output())
        record = json.loads((self.output() / "verilog.jsonl").read_text())
        self.assertEqual(set(record), {"rtl_id", "rtl", "source_categories", "record_kinds", "original_splits"})
        self.assertEqual(record["rtl"], NOTICE_RTL)
        self.assertEqual(record["rtl_id"], hashlib.sha256(NOTICE_RTL.encode()).hexdigest())
        self.assertEqual(record["record_kinds"], ["base"])
        self.assertEqual(manifest["release_status"], "ready-for-redistribution")
        for path in self.output().iterdir():
            value = path.read_text()
            for forbidden in ("private-source-key", "private-design-key", "private-measurement-id",
                              "private synthesis report", "private constraint payload", "private-revision", str(config)):
                self.assertNotIn(forbidden, value)
        self.assertEqual((self.home / "input-train.jsonl").read_bytes(), before)
        self.assertEqual(verify_release(self.output())["unique_rtl_texts"], 1)

    def test_exact_global_dedup_removes_clock_repeats_and_unions_splits_categories_and_kinds(self):
        config = self.config({"train": [measured(clock=1), measured(clock=2), measured(kind="augmentation")],
                              "valid": [measured(source="different-private-source", kind="external")],
                              "test": [measured(kind="test")]},
                             {"private-source-key": registry(allow_generated=True, allow_evaluation=True),
                              "different-private-source": registry(category="other")})
        manifest = build_release(config, self.output())
        records = list(map(json.loads, (self.output() / "verilog.jsonl").read_text().splitlines()))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["source_categories"], ["example", "other"])
        self.assertEqual(records[0]["record_kinds"], ["base", "evaluation", "generated"])
        self.assertEqual(records[0]["original_splits"], ["test", "train", "valid"])
        self.assertEqual(manifest["counts"], {"measured_rows": 5, "included_rows": 5, "withheld_rows": 0, "unique_rtl_texts": 1})
        self.assertEqual(sum(row["unique_rtl_texts"] for row in manifest["split_distribution"]), 3)
        self.assertEqual(sum(row["unique_rtl_texts"] for row in manifest["source_distribution"]), 2)

    def test_dedup_is_exact_and_does_not_claim_semantic_equivalence(self):
        alternative = NOTICE_RTL.replace("  assign", "    assign")
        manifest = build_release(self.config({"train": [measured(), measured(alternative)]}), self.output())
        self.assertEqual(manifest["counts"]["unique_rtl_texts"], 2)

    def test_all_inclusion_flags_default_deny_and_unidentified_sources_are_withheld(self):
        source = registry()
        source.pop("allow_base")
        config = self.config({"train": [measured(), measured(kind="augmentation"), measured(kind="test"),
                                       measured(source="unknown")]}, {"private-source-key": source})
        manifest = build_release(config, self.output())
        self.assertEqual(manifest["counts"]["unique_rtl_texts"], 0)
        self.assertEqual(manifest["exclusion_reasons"], {"unapproved_base": 1, "unapproved_generated": 1,
                                                         "unapproved_evaluation": 1, "unidentified_source": 1})
        self.assertEqual(manifest["release_status"], "staging-only")

    def test_base_approval_does_not_approve_generated_or_evaluation(self):
        manifest = build_release(self.config({"train": [measured(), measured(kind="augmentation"),
                                                        measured(kind="test")]}), self.output())
        self.assertEqual(manifest["counts"]["included_rows"], 1)
        self.assertEqual(manifest["exclusion_reasons"], {"unapproved_generated": 1, "unapproved_evaluation": 1})

    def test_unknown_kind_is_withheld_instead_of_inferred_from_name(self):
        manifest = build_release(self.config({"train": [measured(kind="unknown")]}), self.output())
        self.assertEqual(manifest["exclusion_reasons"], {"unknown_record_kind": 1})

    def test_missing_and_invalid_utf8_rtl_are_counted_without_export(self):
        manifest = build_release(self.config({"train": [measured(rtl="  "), measured(rtl=5),
                                                        measured(rtl="module x; // \ud800\nendmodule")]}), self.output())
        self.assertEqual(manifest["exclusion_reasons"], {"missing_rtl": 2, "invalid_rtl_utf8": 1})

    def test_private_paths_ip_credentials_reports_library_cells_and_local_literals_are_withheld(self):
        samples = [
            ("private_identifier_or_path", "// /home/private-user/tool/run\nmodule demo; endmodule"),
            ("private_identifier_or_path", "// 192.0.2.12\nmodule demo; endmodule"),
            ("credential_marker", "// api_key=do-not-publish\nmodule demo; endmodule"),
            ("synthesis_or_timing_report", "// Generated by Design Compiler\nmodule demo; endmodule"),
            ("cell_library_artifact", "module demo; NAND2_X1 inst(); endmodule"),
            ("cell_library_artifact", "// library: secret-process.lib\nmodule demo; endmodule"),
            ("forbidden_literal", "// PrivateLibraryMarker\nmodule demo; endmodule"),
        ]
        config = self.config({"train": [measured(rtl=rtl) for _, rtl in samples]}, forbidden_literals=["privatelibrarymarker"])
        manifest = build_release(config, self.output())
        self.assertEqual(manifest["counts"]["included_rows"], 0)
        self.assertEqual(manifest["exclusion_reasons"], {"private_identifier_or_path": 2, "credential_marker": 1,
                                                        "synthesis_or_timing_report": 1, "cell_library_artifact": 2,
                                                        "forbidden_literal": 1})
        self.assertNotIn("PrivateLibraryMarker", (self.output() / "manifest.json").read_text())

    def test_compiled_forbidden_literals_preserve_casefolded_substring_semantics(self):
        cases = [
            ([], "module demo; endmodule"),
            (["prefix", "prefix-long"], "// longer-PREFIX-long-suffix\nmodule demo; endmodule"),
            (["prefix-long", "prefix"], "// embeddedPREFIXsuffix\nmodule demo; endmodule"),
            (["array[3]", "wire.*", "a+b", "$q", "back\\slash"], "// wire_anything\nmodule demo; endmodule"),
            (["array[3]", "wire.*", "a+b", "$q", "back\\slash"], "// array[3] back\\slash\nmodule demo; endmodule"),
            (["Straße", "İ"], "// STRASSE\nmodule demo; endmodule"),
            (["Straße", "İ"], "// i\u0307\nmodule demo; endmodule"),
            ([""], "module demo; endmodule"),
            ([f"synthetic-forbidden-{index:04d}" for index in range(500)], "// synthetic-forbidden-0499-suffix\nmodule demo; endmodule"),
            ([f"synthetic-forbidden-{index:04d}" for index in range(500)], "// synthetic-forbidden-0500\nmodule demo; endmodule"),
        ]
        _forbidden_regex.cache_clear()
        for literals, rtl in cases:
            with self.subTest(literals=len(literals), text=rtl):
                expected = any(literal.casefold() in rtl.casefold() for literal in literals)
                self.assertEqual(_unsafe_rtl(rtl, literals) == "forbidden_literal", expected)
                self.assertEqual(_unsafe_rtl(rtl, literals) == "forbidden_literal", expected)
        self.assertGreaterEqual(_forbidden_regex.cache_info().hits, len(cases))
        self.assertEqual(_forbidden_regex.cache_info().maxsize, 16)

    def test_repeated_measurement_rows_reuse_safety_decision_but_keep_row_withholding_counts(self):
        unsafe = "// SourcePrivateMarker\nmodule demo; endmodule"
        rows = [measured(clock=clock) for clock in range(6)]
        rows.extend(measured(rtl=unsafe, clock=clock) for clock in range(6))
        config = self.config({"train": rows}, forbidden_literals=["SourcePrivateMarker"])
        with patch("revolution.verilog_release._unsafe_rtl", wraps=_unsafe_rtl) as check:
            manifest = preflight(config)
        actual_texts = [call.args[0] for call in check.call_args_list]
        self.assertEqual(actual_texts.count(NOTICE_RTL), 1)
        self.assertEqual(actual_texts.count(unsafe), 1)
        self.assertEqual(manifest["counts"], {"measured_rows": 12, "included_rows": 6,
                                               "withheld_rows": 6, "unique_rtl_texts": 1})
        self.assertEqual(manifest["exclusion_reasons"], {"forbidden_literal": 6})

    def test_generic_rtl_synthesis_pragmas_and_author_notice_remain_unchanged(self):
        rtl = "// Copyright public-author <author@example.org>\n// synopsys translate_off\nmodule demo; endmodule\n"
        manifest = build_release(self.config({"train": [measured(rtl=rtl)]}), self.output())
        self.assertEqual(manifest["counts"]["included_rows"], 1)
        self.assertEqual(json.loads((self.output() / "verilog.jsonl").read_text())["rtl"], rtl)

    def test_unverified_or_conditional_rights_keep_explicitly_included_candidate_staging_only(self):
        for status in ("unverified", "conditional"):
            with self.subTest(status=status):
                config = self.config(sources={"private-source-key": registry(redistribution_status=status)})
                self.assertEqual(preflight(config)["release_status"], "staging-only")

    def test_omitted_redistribution_status_defaults_unverified(self):
        source = registry()
        source.pop("redistribution_status")
        self.assertEqual(preflight(self.config(sources={"private-source-key": source}))["release_status"], "staging-only")

    def test_missing_upstream_evidence_can_be_staged_without_invented_references(self):
        manifest = build_release(self.config(sources={"private-source-key": registry(
            upstream_url=None, license_url=None, license="unresolved", redistribution_status="unverified")}), self.output())
        self.assertEqual(manifest["release_status"], "staging-only")
        self.assertIsNone(manifest["source_categories"][0]["references"][0]["upstream_url"])
        self.assertEqual(verify_release(self.output())["status"], "verified")
        with self.assertRaisesRegex(ValueError, "Cleared sources require"):
            load_config(self.config(sources={"private-source-key": registry(upstream_url=None)}))

    def test_preflight_writes_no_package_or_original_changes(self):
        config = self.config()
        before = {path.name: path.read_bytes() for path in self.home.iterdir()}
        manifest = preflight(config)
        self.assertEqual(manifest["outputs"], {})
        self.assertEqual({path.name: path.read_bytes() for path in self.home.iterdir()}, before)

    def test_optional_input_digest_validated_but_never_exported(self):
        config = self.config()
        value = json.loads(config.read_text())
        digest = file_sha256(self.home / "input-train.jsonl")
        value["inputs"][0]["sha256"] = digest
        config.write_text(json.dumps(value))
        build_release(config, self.output())
        for path in self.output().iterdir():
            self.assertNotIn(digest, path.read_text())
        value["inputs"][0]["sha256"] = "0" * 64
        config.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "SHA256 does not match"):
            preflight(config)

    def test_deterministic_bytes_independent_of_input_row_and_split_order(self):
        rows = [measured(), measured(NOTICE_RTL + "\n"), measured(clock=2)]
        config = self.config({"valid": rows, "train": rows[::-1]})
        first = self.home / "first"
        build_release(config, first)
        value = json.loads(config.read_text())
        value["inputs"].reverse()
        config.write_text(json.dumps(value))
        for input_entry in value["inputs"]:
            path = self.home / input_entry["path"]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows[::-1]))
        second = self.home / "second"
        build_release(config, second)
        self.assertEqual({path.name: path.read_bytes() for path in first.iterdir()},
                         {path.name: path.read_bytes() for path in second.iterdir()})

    def test_existing_output_and_source_checkout_and_input_overlap_are_refused(self):
        config = self.config()
        output = self.output()
        output.mkdir()
        (output / "keep").write_text("preserve")
        for unsafe in (output, ROOT / "unwanted-data-package", self.home / "input-train.jsonl", self.home):
            with self.subTest(output=unsafe):
                with self.assertRaises(ValueError):
                    build_release(config, unsafe)
        self.assertEqual((output / "keep").read_text(), "preserve")

    def test_unsafe_labels_duplicate_splits_and_nonboolean_approvals_are_refused(self):
        config = self.config()
        original = json.loads(config.read_text())
        cases = []
        case = json.loads(json.dumps(original)); case["inputs"][0]["split"] = "../private"; cases.append(case)
        case = json.loads(json.dumps(original)); case["inputs"].append(case["inputs"][0]); cases.append(case)
        case = json.loads(json.dumps(original)); case["sources"]["private-source-key"]["allow_base"] = 1; cases.append(case)
        case = json.loads(json.dumps(original)); case["sources"]["private-source-key"]["category"] = "Private User"; cases.append(case)
        for case in cases:
            with self.subTest(config=case):
                config.write_text(json.dumps(case))
                with self.assertRaises(ValueError):
                    load_config(config)

    def test_credentialed_private_or_non_https_registry_links_are_refused(self):
        for url in ("http://example.org/rtl", "https://user:secret@example.org/rtl", "https://192.0.2.1/rtl",
                    "https://[::1]/rtl", "https://[fd00::1]/rtl", "https://[2001:db8::1]/rtl",
                    "https://localhost/rtl", "https://example.org/rtl?token=secret", "https://example.org:private-value/rtl"):
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    load_config(self.config(sources={"private-source-key": registry(upstream_url=url)}))

    def test_malformed_rows_leave_no_partial_package(self):
        config = self.config()
        source = self.home / "input-train.jsonl"
        source.write_text(json.dumps(measured()) + "\ninvalid-json\n")
        with self.assertRaisesRegex(ValueError, "Malformed JSONL"):
            build_release(config, self.output())
        self.assertFalse(self.output().exists())

    def test_symlink_inputs_are_refused_when_supported(self):
        config = self.config()
        link = self.home / "linked-input.jsonl"
        try:
            link.symlink_to(self.home / "input-train.jsonl")
        except OSError:
            self.skipTest("Creating symlinks is unavailable in this environment")
        value = json.loads(config.read_text())
        value["inputs"][0]["path"] = link.name
        config.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "Symlink"):
            load_config(config)

    def test_verification_rejects_extra_payload_file(self):
        build_release(self.config(), self.output())
        (self.output() / "predictions.csv").write_text("private")
        with self.assertRaisesRegex(ValueError, "exactly the four"):
            verify_release(self.output())

    def test_verification_rejects_unknown_record_fields_even_with_recomputed_hash(self):
        build_release(self.config(), self.output())
        rtl_path = self.output() / "verilog.jsonl"
        record = json.loads(rtl_path.read_text())
        record["area"] = 123.4
        rtl_path.write_text(json.dumps(record) + "\n")
        manifest_path = self.output() / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["outputs"]["verilog.jsonl"] = {"sha256": file_sha256(rtl_path), "bytes": rtl_path.stat().st_size}
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "Unexpected Verilog record fields"):
            verify_release(self.output())

    def test_verification_rejects_tampered_checksum_and_unknown_manifest_fields(self):
        build_release(self.config(), self.output())
        rtl_path = self.output() / "verilog.jsonl"
        rtl_path.write_text(rtl_path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            verify_release(self.output())
        path = self.output() / "manifest.json"
        value = json.loads(path.read_text()); value["private_original_sha256"] = "0" * 64
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "Unexpected public manifest fields"):
            verify_release(self.output())

    def test_verification_rejects_fake_ready_status_for_unreviewed_source(self):
        build_release(self.config(sources={"private-source-key": registry(redistribution_status="unverified")}), self.output())
        path = self.output() / "manifest.json"
        value = json.loads(path.read_text()); value["release_status"] = "ready-for-redistribution"
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "Ready status"):
            verify_release(self.output())

    def test_verification_rejects_invented_category_without_distribution_even_when_rehashed(self):
        build_release(self.config(), self.output())
        rtl_path = self.output() / "verilog.jsonl"
        record = json.loads(rtl_path.read_text())
        record["source_categories"].append("invented")
        rtl_path.write_text(json.dumps(record) + "\n")
        manifest_path = self.output() / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["source_categories"].append({"category": "invented",
            "references": manifest["source_categories"][0]["references"]})
        manifest["outputs"]["verilog.jsonl"] = {"sha256": file_sha256(rtl_path), "bytes": rtl_path.stat().st_size}
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "requires a source distribution row"):
            verify_release(self.output())

    def test_malformed_rights_statuses_are_rejected_as_validation_errors(self):
        config = self.config()
        value = json.loads(config.read_text())
        value["sources"]["private-source-key"]["redistribution_status"] = []
        config.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "Redistribution status"):
            load_config(config)
        build_release(self.config(), self.output())
        manifest_path = self.output() / "manifest.json"
        original = json.loads(manifest_path.read_text())
        for location in ("release_status", "reference_status"):
            with self.subTest(location=location):
                value = json.loads(json.dumps(original))
                if location == "release_status":
                    value["release_status"] = []
                else:
                    value["source_categories"][0]["references"][0]["redistribution_status"] = []
                manifest_path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    verify_release(self.output())

    def test_cli_runs_from_other_directory_without_optional_dependencies(self):
        config = self.config()
        script = ROOT / "scripts/prepare_verilog_release.py"
        command = [sys.executable, "-I", "-S", str(script)]
        for args in (("preflight", "--config", str(config)), ("build", "--config", str(config), "--output", str(self.output())),
                     ("verify", "--package", str(self.output()))):
            result = subprocess.run([*command, *args], cwd=self.home, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            parsed = json.loads(result.stdout)
            self.assertNotIn(str(config), result.stdout)
            self.assertNotIn("private-source-key", result.stdout)
            self.assertIsInstance(parsed, dict)


if __name__ == "__main__":
    unittest.main()
