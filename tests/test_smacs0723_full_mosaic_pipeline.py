from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

pytest.importorskip("requests")

pytest.importorskip("requests")

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from smacs0723_full_mosaic_pipeline import (  # noqa: E402
    _band_filename_contains,
    _ensure_local_cpp,
    build_parser,
)


def test_band_filename_contains_matches_long_wave_only() -> None:
    assert _band_filename_contains("F277W", "b") == "nrcblong"
    assert _band_filename_contains("F356W", "a") == "nrcalong"
    assert _band_filename_contains("F200W", "b") is None


def test_parser_defaults_are_simple() -> None:
    args = build_parser().parse_args([])
    assert str(args.root).endswith("/data")
    assert args.input_product == "cal"
    assert args.train_model == "spectral_basis_field"
    assert args.train_steps == 12


def test_ensure_local_cpp_installs_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"installed": False, "calls": 0}

    def _fake_import_module(name: str):
        if name == "torchsky":
            return object()
        if name == "torchsky.cpp":
            if state["installed"]:
                return object()
            raise ModuleNotFoundError(name)
        return importlib.import_module(name)

    def _fake_check_call(cmd, cwd):
        del cmd, cwd
        state["calls"] += 1
        state["installed"] = True

    monkeypatch.setattr(
        "smacs0723_full_mosaic_pipeline.importlib.import_module", _fake_import_module
    )
    monkeypatch.setattr(
        "smacs0723_full_mosaic_pipeline.subprocess.check_call", _fake_check_call
    )

    _ensure_local_cpp()

    assert state["calls"] == 1
