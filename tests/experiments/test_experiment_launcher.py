"""Exercise release dispatch without loading any model or research dataset."""

import importlib.util
import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[2]
LAUNCHER = REPO / "scripts/run_experiment.py"
spec = importlib.util.spec_from_file_location("release_experiment_launcher", LAUNCHER)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class ExperimentLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root / "dataset"
        self.model = self.root / "checkpoint"
        self.output = self.root / "run"
        self.dataset.mkdir()
        self.model.mkdir()

    def invoke(self, args):
        return subprocess.run([sys.executable, str(LAUNCHER), *args], cwd=self.root,
                              text=True, capture_output=True)

    def arguments(self, choice):
        return [choice, "--dataset", str(self.dataset), "--model", str(self.model),
                "--output", str(self.output)]

    def test_every_selection_dispatches_from_unrelated_directory(self):
        for choice in launcher.ENTRYPOINTS:
            with self.subTest(choice=choice):
                result = self.invoke([*self.arguments(choice), "--dry-run", "--", "--do_train"])
                self.assertEqual(result.returncode, 0, result.stderr)
                command = json.loads(result.stdout)["command"]
                self.assertTrue(Path(command[1]).is_file())
                self.assertEqual(command[command.index("--dataset_dir") + 1], str(self.dataset.resolve()))
                model_option = "--model_name" if choice == "pair-classifier" else "--base_model_dir"
                self.assertEqual(command[command.index(model_option) + 1], str(self.model))
                self.assertEqual(command[command.index("--output_dir") + 1], str(self.output.resolve()))
                self.assertIn("--do_train", command)
                self.assertFalse(self.output.exists())

    def test_all_artifact_arguments_are_required(self):
        for option in ["--dataset", "--model", "--output"]:
            with self.subTest(option=option):
                args = self.arguments("number-token-loss")
                offset = args.index(option)
                del args[offset:offset + 2]
                result = self.invoke([*args, "--dry-run"])
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(option, result.stderr)

    def test_relative_checkpoint_is_resolved_before_working_directory_changes(self):
        args = self.arguments("number-token-loss")
        args[args.index("--model") + 1] = "checkpoint"
        result = self.invoke([*args, "--dry-run"])
        self.assertEqual(result.returncode, 0, result.stderr)
        command = json.loads(result.stdout)["command"]
        self.assertEqual(command[command.index("--base_model_dir") + 1], str(self.model.resolve()))

    def test_relative_resume_input_remains_exact(self):
        args = self.arguments("latent-losses")
        result = self.invoke([*args, "--dry-run", "--", "--do_train", "--resume_from_checkpoint", "--checkpoint_path", "checkpoint"])
        self.assertEqual(result.returncode, 0, result.stderr)
        command = json.loads(result.stdout)["command"]
        self.assertEqual(command[command.index("--checkpoint_path") + 1], str(self.model.resolve()))

    def test_forwarded_flags_cannot_override_output(self):
        result = self.invoke([*self.arguments("number-token-loss"), "--dry-run", "--", "--output_dir=elsewhere"])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("artifact locations", result.stderr)

    def test_output_cannot_overlap_inputs(self):
        with self.assertRaisesRegex(ValueError, "separate directories"):
            launcher.validate_locations(self.dataset, str(self.model), self.dataset / "output", dry_run=True)
        with self.assertRaisesRegex(ValueError, "separate directories"):
            launcher.validate_locations(self.dataset, str(self.model), self.model, dry_run=True)

    def test_actual_dispatch_rejects_existing_output_before_model_load(self):
        self.output.mkdir()
        result = self.invoke([*self.arguments("number-token-loss"), "--", "--do_train"])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("new directory", result.stderr)

    def test_resume_requires_exact_checkpoint(self):
        result = self.invoke([*self.arguments("latent-losses"), "--dry-run", "--", "--do_train", "--resume_from_checkpoint"])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--checkpoint_path", result.stderr)

    def test_number_loss_arguments_cover_shared_dataset_interface(self):
        # This cross-module contract previously lacked three configuration
        # attributes, so help/import checks alone could not detect the issue.
        core = ast.parse((REPO / "src/text2text_regression/finetune_ppa.py").read_text(encoding="utf-8"))
        configure = next(node for node in core.body if isinstance(node, ast.FunctionDef) and node.name == "configure_dataset")
        required = {node.attr for node in ast.walk(configure)
                    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "args"}
        experiment = ast.parse((REPO / "experiments/training/finetune_ppa_losses.py").read_text(encoding="utf-8"))
        declared = {node.args[0].value.removeprefix("--") for node in ast.walk(experiment)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_argument" and node.args
                    and isinstance(node.args[0], ast.Constant)}
        self.assertEqual(required - declared, set())


if __name__ == "__main__":
    unittest.main()
