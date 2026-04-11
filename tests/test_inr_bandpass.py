"""Tests for :mod:`torchsky.filter_profiles.inr_bandpass` (no network by default)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from torchsky.filter_profiles import (
    bandpass_from_transmission_file,
    bandpass_gaussian,
    field_observation_spectral_kwargs,
    resolve_inr_bandpass,
)


def test_bandpass_gaussian() -> None:
    spec = bandpass_gaussian(
        band_name="F200W",
        lambda_eff=20000.0,
        lambda_min=17000.0,
        lambda_max=23000.0,
        sigma=None,
    )
    fo = field_observation_spectral_kwargs(spec)
    assert fo["band_name"] == "F200W"
    assert fo["spectral_min"] == 17000.0
    assert fo["spectral_max"] == 23000.0
    assert fo["wavelength"] == 20000.0
    assert fo["transmission_sigma"] > 0.0


def test_bandpass_from_transmission_file(tmp_path: Path) -> None:
    p = tmp_path / "curve.txt"
    wl = np.linspace(1.0, 10.0, 20)
    tr = np.exp(-0.5 * ((wl - 5.5) / 1.2) ** 2)
    lines = ["# test", "wavelength transmission"]
    for a, b in zip(wl, tr):
        lines.append(f"{a} {b}")
    p.write_text("\n".join(lines), encoding="utf-8")
    spec = bandpass_from_transmission_file(p, band_name="X1")
    assert spec["band_name"] == "X1"
    assert "transmission_wavelength" in spec
    assert spec["transmission_wavelength"].shape == spec["transmission_throughput"].shape
    fo = field_observation_spectral_kwargs(spec)
    assert fo["spectral_max"] > fo["spectral_min"]
    assert isinstance(fo["transmission_wavelength"], torch.Tensor)
    assert fo["transmission_wavelength"].shape == fo["transmission_throughput"].shape


def test_resolve_builtin_table() -> None:
    tab = {"F277W": (24000.0, 32000.0, 27700.0, 1200.0)}
    spec = resolve_inr_bandpass(
        source="builtin",
        band_name="F277W",
        builtin_table=tab,
        svo_facility="JWST",
        svo_instrument="NIRCAM",
        svo_filter_id=None,
        svo_prefer="classic",
        transmission_path=None,
        transmission_dir=None,
        transmission_unit="angstrom",
        gaussian_lambda_eff=None,
        gaussian_lambda_min=None,
        gaussian_lambda_max=None,
        gaussian_sigma=None,
    )
    assert spec["wavelength"] == 27700.0


def test_resolve_gaussian() -> None:
    spec = resolve_inr_bandpass(
        source="gaussian",
        band_name="Z",
        builtin_table={},
        svo_facility="JWST",
        svo_instrument="NIRCAM",
        svo_filter_id=None,
        svo_prefer="classic",
        transmission_path=None,
        transmission_dir=None,
        transmission_unit="angstrom",
        gaussian_lambda_eff=5000.0,
        gaussian_lambda_min=4000.0,
        gaussian_lambda_max=6000.0,
        gaussian_sigma=100.0,
    )
    assert spec["transmission_sigma"] == 100.0


@pytest.mark.parametrize(
    ("prefer", "expect_substr"),
    [
        ("classic", "JWST/NIRCam.F277W"),
        ("2025", "NIRCam2025"),
    ],
)
def test_bandpass_from_svo_mock(
    monkeypatch: pytest.MonkeyPatch, prefer: str, expect_substr: str
) -> None:
    pytest.importorskip("astroquery")

    class FakeRow:
        def __init__(self) -> None:
            self._d = {
                "filterID": "JWST/NIRCam.F277W",
                "WavelengthEff": 27278.0,
                "WavelengthMin": 23673.0,
                "WavelengthMax": 32203.0,
                "WavelengthPivot": 27617.0,
                "FWHM": 7064.0,
            }

        def __getitem__(self, k: str):
            return self._d[k]

    class FakeRow2025:
        def __getitem__(self, k: str):
            d = {
                "filterID": "JWST/NIRCam2025.F277W",
                "WavelengthEff": 27000.0,
                "WavelengthMin": 23000.0,
                "WavelengthMax": 31000.0,
                "WavelengthPivot": 27500.0,
                "FWHM": 7000.0,
            }
            return d[k]

    fake_table = [FakeRow2025(), FakeRow()]

    def fake_get_filter_list(*args: object, **kwargs: object):
        return fake_table

    monkeypatch.setattr(
        "astroquery.svo_fps.SvoFps.get_filter_list",
        fake_get_filter_list,
    )

    wl_curve = np.linspace(23673.0, 32203.0, 32)
    tr_curve = np.exp(-0.5 * ((wl_curve - 27617.0) / 2000.0) ** 2)

    def fake_get_transmission_data(_fid: str):
        return {"Wavelength": wl_curve, "Transmission": tr_curve}

    monkeypatch.setattr(
        "astroquery.svo_fps.SvoFps.get_transmission_data",
        fake_get_transmission_data,
    )

    from torchsky.filter_profiles.inr_bandpass import bandpass_from_svo

    spec = bandpass_from_svo(
        band_name="F277W",
        facility="JWST",
        instrument="NIRCAM",
        filter_id=None,
        prefer=prefer,  # type: ignore[arg-type]
    )
    assert expect_substr in str(spec["svo_filter_id"])
    assert "transmission_wavelength" in spec
    fo = field_observation_spectral_kwargs(spec)
    assert fo["transmission_wavelength"].numel() >= 2
