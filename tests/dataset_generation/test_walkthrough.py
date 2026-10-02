"""The user walkthrough must run without research artifacts or dependencies."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from revolution.synthetic_walkthrough import run_walkthrough


ROOT = Path(__file__).resolve().parents[2]


class WalkthroughTests(unittest.TestCase):
    def test_real_walkthrough_and_existing_output_protection(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "example"
            summary = run_walkthrough(output)
            self.assertEqual(summary["status"], "passed")
            self.assertEqual(sum(summary["split_counts"].values()), 18)
            self.assertEqual(summary["lineage_overlap"], 0)
            saved = (output / "summary.json").read_bytes()
            self.assertEqual(json.loads(saved), summary)
            with self.assertRaisesRegex(ValueError, "already exists"):
                run_walkthrough(output)
            self.assertEqual((output / "summary.json").read_bytes(), saved)

    def test_launcher_from_other_directory_using_isolated_python(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "example"
            # Isolated Python ignores PYTHONPATH and site packages, proving the
            # source walkthrough does not rely on installed ML/EDA dependencies.
            result = subprocess.run(
                [sys.executable, "-I", "-S", str(ROOT / "scripts/synthetic_walkthrough.py"),
                 "--output", str(output)], cwd=temporary, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["status"], "passed")

    def test_reject_output_within_checkout(self):
        with self.assertRaisesRegex(ValueError, "outside the source checkout"):
            run_walkthrough(ROOT / "would-be-synthetic-output")
        self.assertFalse((ROOT / "would-be-synthetic-output").exists())
