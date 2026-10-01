#!/usr/bin/env python3
"""Dispatch preserved research attempts using explicit user-owned artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ENTRYPOINTS = {
    "number-token-loss": "finetune_ppa_losses.py",
    "latent-losses": "finetune_ppa_latent_losses.py",
    "mixed-rank": "finetune_ppa_mixed_rank.py",
    "pair-classifier": "finetune_ppa_pair_classifier.py",
    "t5gemma-pair-classifier": "finetune_ppa_t5gemma_pair_classifier.py",
    "post-finetune": "post_finetune_ppa.py",
}
MANAGED_ARGUMENTS = {
    "--dataset_dir", "--base_model_dir", "--model_name", "--output_dir", "--logging_dir"
}


def is_inside(path: Path, ancestor: Path) -> bool:
    return path == ancestor or ancestor in path.parents


def resolve_model_reference(model: str) -> str:
    """Resolve user-local paths before dispatch changes its working directory."""
    path = Path(model).expanduser()
    if path.exists() or path.is_absolute() or model.startswith((".", "~")):
        return str(path.resolve())
    return model


def validate_locations(dataset: Path, model: str, output: Path, *, dry_run: bool):
    """Prevent generated outputs from overwriting input artifacts."""
    inputs = [dataset]
    model_path = Path(model).expanduser()
    # Hub model IDs are accepted explicitly. Local paths are also protected.
    if model_path.exists() or model_path.is_absolute() or model.startswith("."):
        inputs.append(model_path.resolve())
    for input_path in inputs:
        if is_inside(output, input_path) or is_inside(input_path, output):
            raise ValueError("Output and input artifacts must be separate directories.")
    if not dry_run:
        if not dataset.is_dir():
            raise ValueError("Dataset must be an existing directory of JSONL split files.")
        if output.exists():
            raise ValueError("Output must be a new directory; choose a fresh run location.")


def build_command(args, forwarded: list[str], *, repo: Path | None = None):
    """Translate the public artifact interface into each preserved CLI."""
    root = repo or Path(__file__).resolve().parents[1]
    forwarded = list(forwarded)
    for token in forwarded:
        if token.split("=", 1)[0] in MANAGED_ARGUMENTS:
            raise ValueError("Use --dataset, --model, and --output for artifact locations.")
    if "--resume_from_checkpoint" in forwarded:
        if args.experiment in {"number-token-loss", "latent-losses", "mixed-rank"}:
            if not any(token.split("=", 1)[0] == "--checkpoint_path" for token in forwarded):
                raise ValueError("Resuming requires an explicit --checkpoint_path.")
    local_options = {"--checkpoint_path"}
    if args.experiment in {"pair-classifier", "t5gemma-pair-classifier"}:
        local_options.add("--resume_from_checkpoint")
    model_options = {"--eval_model_dir", "--encoder_model_name"}
    for index, token in enumerate(forwarded):
        option, separator, inline_value = token.partition("=")
        if option not in local_options | model_options:
            continue
        value_index = index if separator else index + 1
        if value_index >= len(forwarded):
            raise ValueError(f"{option} requires an explicit value.")
        value = inline_value if separator else forwarded[value_index]
        if not value or value.startswith("--"):
            raise ValueError(f"{option} requires an explicit value.")
        normalized = (str(Path(value).expanduser().resolve())
                      if option in local_options else resolve_model_reference(value))
        forwarded[value_index] = f"{option}={normalized}" if separator else normalized
    dataset = Path(args.dataset).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    model = resolve_model_reference(args.model)
    help_only = "--help" in forwarded or "-h" in forwarded
    validate_locations(dataset, model, output, dry_run=args.dry_run or help_only)
    model_option = "--model_name" if args.experiment == "pair-classifier" else "--base_model_dir"
    command = [
        sys.executable, str(root / "experiments/training" / ENTRYPOINTS[args.experiment]),
        "--dataset_dir", str(dataset), model_option, model,
        "--output_dir", str(output),
    ]
    if args.experiment in {"number-token-loss", "latent-losses", "mixed-rank"}:
        command += ["--logging_dir", str(output / "logs")]
    command += forwarded
    return command


def main(argv: list[str] | None = None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--" in arguments:
        split = arguments.index("--")
        launch_args, forwarded = arguments[:split], arguments[split + 1:]
    else:
        launch_args, forwarded = arguments, []
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("experiment", choices=ENTRYPOINTS)
    parser.add_argument("--dataset", required=True, help="Directory of compatible JSONL split files.")
    parser.add_argument("--model", required=True, help="Exact checkpoint directory or explicit Hub model ID.")
    parser.add_argument("--output", required=True, help="New directory for the run artifacts.")
    parser.add_argument("--dry-run", action="store_true", help="Print dispatch without loading or training.")
    args = parser.parse_args(launch_args)
    try:
        command = build_command(args, forwarded)
        if args.dry_run:
            print(json.dumps({"command": command}, indent=2))
            return 0
        if not any(token in {"--do_train", "--do_eval", "--help", "-h"} for token in forwarded):
            raise ValueError("Choose --do_train and/or --do_eval after the -- separator.")
    except ValueError as exc:
        parser.error(str(exc))
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env.pop("PYTHONHOME", None)
    env["PYTHONPATH"] = os.pathsep.join([
        str(root / "experiments/training"),
        str(root / "src/text2text_regression"),
        str(root / "src"),
    ])
    env.setdefault("WANDB_MODE", "offline")
    return subprocess.call(command, cwd=root, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
