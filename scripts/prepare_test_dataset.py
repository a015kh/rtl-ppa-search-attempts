#!/usr/bin/env python3
"""Prepare test JSONL from explicitly supplied completed measurement inputs."""

from pathlib import Path
import sys


SOURCE = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from revolution.test_dataset_preparation import main


if __name__ == "__main__":
    raise SystemExit(main())
