"""Committed radio/mm-style FITS fixture (see ``scripts/gen_radio_mm_fixture_fits.py``)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytest.importorskip("torchsky.cpp")

import torchsky  # noqa: E402

_FIXTURE = Path(__file__).resolve().parent / "data" / "radio_mm_stub.fits"


def test_radio_mm_stub_fixture_loads() -> None:
    assert _FIXTURE.is_file(), (
        f"missing {_FIXTURE}; run scripts/gen_radio_mm_fixture_fits.py"
    )
    img, hdr = torchsky.read(str(_FIXTURE), return_header=True)
    assert img.ndim == 2
    assert int(hdr["NAXIS1"]) == int(img.shape[1])
    assert float(hdr["RESTFRQ"]) == pytest.approx(2.30538e11)
    assert float(hdr["BMAJ"]) == pytest.approx(0.015)
    assert torch.isfinite(img).all()
