#!/usr/bin/env python3
"""Assemble a PPA dataset from explicitly configured measured inputs."""

from pathlib import Path
import sys


SOURCE = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from revolution.dataset_generation import main


if __name__ == "__main__":
    raise SystemExit(main())
