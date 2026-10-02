#!/usr/bin/env python3
"""Compute aggregate rank metrics without loading a trained model."""
from pathlib import Path
import sys

SOURCE = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from revolution.rank_evaluation import main

if __name__ == "__main__":
    raise SystemExit(main())
