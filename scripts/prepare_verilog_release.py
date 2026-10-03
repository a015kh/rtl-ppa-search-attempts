#!/usr/bin/env python3
"""Prepare and verify a Verilog-only candidate dataset from explicit source policies."""
from pathlib import Path
import sys

SOURCE = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from revolution.verilog_release import main

if __name__ == "__main__":
    raise SystemExit(main())
