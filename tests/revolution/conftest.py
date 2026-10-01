"""Synthetic fixtures for unit tests; no collected RTL or cell library data."""

import os
import shutil

import pytest


@pytest.fixture(autouse=True)
def fake_binaries(monkeypatch):
    """Resolve only the explicit fake commands whose execution tests mock."""
    original_which = shutil.which
    fake_paths = {"/fake/iverilog", "/fake/vvp"}

    def resolve(command, *args, **kwargs):
        if os.fspath(command) in fake_paths:
            return os.fspath(command)
        return original_which(command, *args, **kwargs)

    monkeypatch.setattr(shutil, "which", resolve)
    return {"iverilog": "/fake/iverilog", "vvp": "/fake/vvp"}


@pytest.fixture
def minimal_sv_files(tmp_path):
    files = {
        "dut": ("dut.sv", "module dut(input wire a, output wire y); assign y = a; endmodule\n"),
        "tb": ("tb.sv", "module tb; reg a; wire y; dut sample(a, y); initial begin a = 0; #1 $finish; end endmodule\n"),
        "pdk": ("synthetic_cell.v", "module synthetic_cell(input wire a, output wire y); assign y = a; endmodule\n"),
    }
    result = {"dir": str(tmp_path)}
    for key, (name, content) in files.items():
        path = tmp_path / name
        path.write_text(content, encoding="utf-8")
        result[key] = str(path)
    return result


@pytest.fixture
def no_chmod(monkeypatch):
    """Compilation subprocesses are mocked and create no executable to chmod."""
    monkeypatch.setattr(os, "chmod", lambda *args, **kwargs: None)


@pytest.fixture
def pdk_cells(tmp_path):
    """A newly authored dummy cell in the compatibility layout expected by tests."""
    path = tmp_path / "data/pdk/Nangate45/work_around_yosys/cells.v"
    path.parent.mkdir(parents=True)
    path.write_text(
        "module synthetic_cell(input wire a, output wire y); assign y = a; endmodule\n",
        encoding="utf-8",
    )
    return str(path)
