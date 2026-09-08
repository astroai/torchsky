"""Optional external CLI detection used by map compatibility benchmarks."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("torchsky.cpp")

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

import bench_map_compat as bench_map_compat  # noqa: E402


def test_swarp_binary_probe_returns_none_or_existing_path() -> None:
    found = bench_map_compat._find_swarp_binary()
    assert found is None or found.is_file()


def test_montage_binary_probe_returns_none_or_existing_path() -> None:
    for name in ("mProject", "mProjectPP"):
        found = bench_map_compat._find_montage_binary(name)
        assert found is None or found.is_file()


def test_montage_binary_probe_uses_installed_image_tools(tmp_path, monkeypatch):
    monkeypatch.setattr(bench_map_compat, "ROOT", tmp_path / "checkout")
    executable = tmp_path / "mProject"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert bench_map_compat._find_montage_binary("mProject") == executable
    assert bench_map_compat._find_montage_binary("mProjectPP") is None
