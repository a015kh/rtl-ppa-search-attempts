"""Compatibility entry point for the canonical evolution runner."""

from pathlib import Path
import runpy
import sys


if __name__ == "__main__":
    source_root = Path(__file__).resolve().parents[1] / "src"
    sys.path.insert(0, str(source_root))
    runpy.run_path(str(source_root / "run_evolution.py"), run_name="__main__")
