"""A reviewed exact file cannot approve another kind, source or transformation."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest

from revolution.licensed_verilog_release import build_reviewed_release, load_review, verify_reviewed_release

RTL = "// Copyright Example contributors\nmodule buffer(input a, output y); assign y=a; endmodule\n"
NOTICE = "Copyright (c) 2025 Example contributors\nPermission is hereby granted, free of charge, to any person obtaining a copy.\n"
ROOT = Path(__file__).resolve().parents[2]


class ReviewedReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)
        self.upstream = self.home / "original.sv"
        self.upstream.write_bytes(RTL.encode())
        self.notice = self.home / "LICENSE"
        self.notice.write_bytes(NOTICE.encode())
        self.ledger = self.home / "rights.jsonl"
        self.proof = {"rtl_id": hashlib.sha256(RTL.encode()).hexdigest(), "source_dataset": "source-a",
            "record_kind": "evaluation", "upstream_url": "https://github.com/example/rtl",
            "upstream_revision": "a" * 40, "upstream_file": "benchmarks/original.sv",
            "upstream_file_sha256": hashlib.sha256(RTL.encode()).hexdigest(),
            "local_source_file": str(self.upstream), "notice_ids": ["example"]}

    def config(self, rows=None, proofs=None, **changes):
        self.ledger.write_text("".join(json.dumps(p) + "\n" for p in (proofs if proofs is not None else [self.proof])))
        measured = self.home / "measured.jsonl"
        default = {"rtl": RTL, "source_dataset": "source-a", "record_kind": "test", "area_value": 999,
                   "source_artifact": "/home/secret/private.jsonl", "SDC": "private-data"}
        measured.write_text("".join(json.dumps(r) + "\n" for r in (rows if rows is not None else [default, default])))
        def source(category):
            return {"category": category, "upstream_url": "https://github.com/example/rtl",
                    "license": "MIT", "license_url": "https://github.com/example/rtl/blob/main/LICENSE",
                    "redistribution_status": "unverified", "allow_base": True, "allow_generated": True, "allow_evaluation": True}
        value = {"schema_version": 1, "inputs": [{"split": "test", "path": str(measured)}],
                 "sources": {"source-a": source("first"), "source-b": source("second")},
                 "rights_ledger": str(self.ledger), "notices": [{"id": "example", "path": str(self.notice),
                    "sha256": hashlib.sha256(NOTICE.encode()).hexdigest(), "license": "MIT",
                    "upstream_url": "https://github.com/example/rtl", "upstream_revision": "a" * 40, "upstream_file": "LICENSE"}], **changes}
        path = self.home / "config.json"
        path.write_text(json.dumps(value))
        return path

    def row(self, **changes):
        return {"rtl": RTL, "source_dataset": "source-a", "record_kind": "test", "area_value": 9, **changes}

    def test_exact_code_full_notice_and_memberships_without_private_fields(self):
        config = self.config()
        before = {p.name: p.read_bytes() for p in self.home.iterdir() if p.is_file()}
        result = build_reviewed_release(config, self.home / "package")
        self.assertEqual(result["counts"], {"included_rows": 2, "measured_rows": 2, "unique_rtl_texts": 1, "withheld_rows": 0})
        emitted = json.loads((self.home / "package/data/verilog.jsonl").read_text())
        self.assertEqual(emitted["rtl"], RTL)
        self.assertEqual((self.home / "package/licenses/example.txt").read_bytes(), NOTICE.encode())
        for path in (self.home / "package").rglob("*"):
            if path.is_file():
                self.assertNotIn("source-a", path.read_text())
                self.assertNotIn("secret", path.read_text())
                self.assertNotIn(str(self.home), path.read_text())
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.home.iterdir() if p.is_file()})
        self.assertEqual(verify_reviewed_release(self.home / "package")["rights_records"], 1)

    def test_hash_approval_is_scoped_to_source_kind_and_exact_text(self):
        rows = [self.row(), self.row(source_dataset="source-b"), self.row(record_kind="augmentation"),
                self.row(record_kind="original"), self.row(rtl=RTL.replace("\n", "\r\n")), self.row(rtl=RTL + "\n")]
        result = build_reviewed_release(self.config(rows), self.home / "package")
        self.assertEqual(result["counts"]["included_rows"], 1)
        self.assertEqual(result["exclusion_reasons"], {"unapproved_record": 5})

    def test_source_flags_are_an_additional_gate(self):
        cfg = self.config()
        value = json.loads(cfg.read_text()); value["sources"]["source-a"]["allow_evaluation"] = False
        cfg.write_text(json.dumps(value))
        result = build_reviewed_release(cfg, self.home / "package")
        self.assertEqual(result["exclusion_reasons"], {"unapproved_evaluation": 2})

    def test_license_approval_cannot_override_private_literal_filter(self):
        result = build_reviewed_release(self.config(forbidden_literals=["assign y=a"]), self.home / "package")
        self.assertEqual(result["exclusion_reasons"], {"forbidden_literal": 2})
        self.assertEqual(list((self.home / "package").glob("licenses/*")), [])

    def test_incorrect_source_or_notice_bytes_fail_before_output(self):
        for what in ("source", "notice"):
            with self.subTest(what=what):
                cfg = self.config()
                target = self.upstream if what == "source" else self.notice
                old = target.read_bytes(); target.write_bytes(old + b"\n")
                with self.assertRaisesRegex(ValueError, "SHA256|bytes"):
                    build_reviewed_release(cfg, self.home / "package")
                self.assertFalse((self.home / "package").exists())
                target.write_bytes(old)

    def test_unsafe_evidence_revision_path_and_notice_scope_are_rejected(self):
        for changes in ({"upstream_revision": "main"}, {"upstream_file": "../rtl.sv"},
                        {"upstream_file_sha256": "0" * 64}, {"notice_ids": []},
                        {"upstream_url": "https://github.com/another/repo"}):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    load_review(self.config(proofs=[self.proof | changes]))

    def test_duplicate_and_unused_proofs_do_not_approve_extra_data(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            load_review(self.config(proofs=[self.proof, self.proof]))
        unused = self.proof | {"source_dataset": "source-b"}
        result = build_reviewed_release(self.config(proofs=[self.proof, unused]), self.home / "package")
        self.assertEqual(result["counts"]["included_rows"], 2)
        self.assertEqual(verify_reviewed_release(self.home / "package")["rights_records"], 1)

    def test_tampering_unexpected_payload_and_existing_output_are_rejected(self):
        cfg = self.config(); output = self.home / "package"
        build_reviewed_release(cfg, output)
        with self.assertRaisesRegex(ValueError, "already exists"):
            build_reviewed_release(cfg, output)
        (output / "private-report.txt").write_text("unexpected")
        with self.assertRaisesRegex(ValueError, "payload"):
            verify_reviewed_release(output)
        (output / "private-report.txt").unlink()
        with (output / "licenses/example.txt").open("a") as stream:
            stream.write("changed")
        with self.assertRaisesRegex(ValueError, "checksum"):
            verify_reviewed_release(output)

    def test_launcher_help_and_verification_from_other_directory(self):
        cfg = self.config(); output = self.home / "package"
        build_reviewed_release(cfg, output)
        for args in (["--help"], ["verify", "--output", str(output)]):
            command = [sys.executable, "-I", "-S", str(ROOT / "scripts/prepare_licensed_verilog_release.py"), *args]
            result = subprocess.run(command, cwd=self.home, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_io_and_encoding_failures_do_not_print_private_input_locations(self):
        cfg = self.config()
        self.notice.write_bytes(b"private-fragment-\xff")
        value = json.loads(cfg.read_text())
        value["notices"][0]["sha256"] = hashlib.sha256(self.notice.read_bytes()).hexdigest()
        cfg.write_text(json.dumps(value))
        result = subprocess.run([sys.executable, str(ROOT / "scripts/prepare_licensed_verilog_release.py"),
            "build", "--config", str(cfg), "--output", str(self.home / "package")], cwd=self.home, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertNotIn(str(self.home), result.stderr)
        self.assertNotIn("private-fragment", result.stderr)
        self.assertIn("invalid input or inaccessible file", result.stderr)


if __name__ == "__main__":
    unittest.main()
