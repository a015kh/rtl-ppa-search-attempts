#!/usr/bin/env python3
"""Run the offline synthetic walkthrough from any working directory."""

from pathlib import Path
import sys

SOURCE = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from revolution.synthetic_walkthrough import main

if __name__ == "__main__":
    raise SystemExit(main())
