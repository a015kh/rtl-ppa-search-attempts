"""Upstream recipes are offline metadata, never an implicit data release."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "retrieve_public_verilog_sources.py"
SPEC = importlib.util.spec_from_file_location("source_recipes", SCRIPT)
assert SPEC and SPEC.loader
recipes = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recipes)


class SourceRecipeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def test_metadata_only_selected_sources_and_no_local_locations(self):
        output = self.directory / "recipes"
        with patch("urllib.request.urlopen", side_effect=AssertionError("network access")):
            manifest = recipes.write_recipes(["pyranet", "metrex", "metrex"], output)
        self.assertEqual(sorted(path.name for path in output.iterdir()), ["README.md", "sources.json"])
        self.assertEqual([source["source_key"] for source in manifest["sources"]], ["metrex", "pyranet"])
        self.assertEqual(json.loads((output / "sources.json").read_text(encoding="utf-8")), manifest)
        for flag in ("contains_rtl", "contains_ppa", "redistribution_permission_established", "recreates_historical_dataset"):
            self.assertFalse(manifest[flag])
        for file in output.iterdir():
            text = file.read_text(encoding="utf-8")
            self.assertNotIn(str(self.directory), text)
            self.assertNotIn("module ", text)
        self.assertEqual(manifest["sources"][0]["dataset_revision"], "b9cb51395445da1586e5b0b3ba6f7c91851823c5")

    def test_unknown_and_empty_selection_fail_before_file_creation(self):
        for selection in ([], ["unrecognized"]):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                recipes.write_recipes(selection, self.directory / "output")
            self.assertFalse((self.directory / "output").exists())

    def test_checkout_output_is_rejected(self):
        for output in (ROOT, ROOT / "new-recipes"):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "outside"):
                recipes.write_recipes(["metrex"], output)

    def test_existing_output_is_not_overwritten(self):
        output = self.directory / "existing"
        output.mkdir()
        sentinel = output / "sources.json"
        sentinel.write_text("keep this unchanged", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "never overwritten"):
            recipes.write_recipes(["metrex"], output)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep this unchanged")

    def test_mutable_references_are_not_labeled_historical_pins(self):
        for name in ("mg-verilog", "openrtlset", "rtlcoder"):
            source = recipes.recipe_manifest([name])["sources"][0]
            self.assertIsNone(source["dataset_revision"])
            self.assertIsNone(source["historical_import_revision"])
            self.assertEqual(source["revision_kind"], "mutable-current-reference")

    def test_cli_listing_and_help_work_from_another_directory_without_writes(self):
        listing = subprocess.run([sys.executable, str(SCRIPT), "list"], cwd=self.directory,
                                 check=True, capture_output=True, text=True)
        self.assertEqual(len(json.loads(listing.stdout)["sources"]), 6)
        subprocess.run([sys.executable, str(SCRIPT), "--help"], cwd=self.directory,
                       check=True, capture_output=True, text=True)
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_cli_writes_selected_metadata_from_another_directory(self):
        output = self.directory / "new-recipes"
        result = subprocess.run([sys.executable, str(SCRIPT), "write", "--source", "rtlcoder",
                                 "--output-dir", str(output)], cwd=self.directory,
                                check=True, capture_output=True, text=True)
        manifest = json.loads(result.stdout)
        self.assertEqual([item["source_key"] for item in manifest["sources"]], ["rtlcoder"])
        self.assertFalse(manifest["redistribution_permission_established"])
        self.assertEqual(json.loads((output / "sources.json").read_text(encoding="utf-8")), manifest)
        self.assertEqual(sorted(file.name for file in output.iterdir()), ["README.md", "sources.json"])

    def test_cli_requires_selection_and_external_output(self):
        command = [sys.executable, str(SCRIPT)]
        for arguments in (["write"], ["write", "--source", "metrex"],
                          ["write", "--output-dir", str(self.directory / "new")],
                          ["write", "--source", "unknown", "--output-dir", str(self.directory / "new")]):
            with self.subTest(arguments=arguments):
                result = subprocess.run(command + arguments, cwd=self.directory, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
        self.assertEqual(list(self.directory.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
