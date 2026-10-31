"""Tests for :mod:`torchsky.filter_profiles.field_bandpass` (no network by default)."""

from __future__ import annotations

from pathlib import Path
import sys
import types
from typing import Any

import numpy as np
import pytest
import torch

from torchsky.filter_profiles import (
    bandpass_from_transmission_file,
    bandpass_gaussian,
    field_observation_spectral_kwargs,
    resolve_field_bandpass,
)


def test_removed_bandpass_names_are_absent() -> None:
    import torchsky.filter_profiles as filter_profiles

    assert not hasattr(filter_profiles, "FieldSpectralSource")
    assert not hasattr(filter_profiles, "InrSpectralSource")
    assert not hasattr(filter_profiles, "resolve_inr_bandpass")


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
    assert (
        spec["transmission_wavelength"].shape == spec["transmission_throughput"].shape
    )
    fo = field_observation_spectral_kwargs(spec)
    assert fo["spectral_max"] > fo["spectral_min"]
    assert isinstance(fo["transmission_wavelength"], torch.Tensor)
    assert fo["transmission_wavelength"].shape == fo["transmission_throughput"].shape


def test_resolve_builtin_table() -> None:
    tab = {"F277W": (24000.0, 32000.0, 27700.0, 1200.0)}
    spec = resolve_field_bandpass(
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
    spec = resolve_field_bandpass(
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


def _install_fake_astroquery(monkeypatch: pytest.MonkeyPatch, fake_table: list[Any], wl_curve: np.ndarray, tr_curve: np.ndarray) -> None:
    aq_mod = types.ModuleType("astroquery")
    svo_mod = types.ModuleType("astroquery.svo_fps")

    class FakeSvoFps:
        @staticmethod
        def get_filter_list(*args: object, **kwargs: object) -> list[Any]:
            return fake_table

        @staticmethod
        def get_transmission_data(_fid: str) -> dict[str, np.ndarray]:
            return {"Wavelength": wl_curve, "Transmission": tr_curve}

    svo_mod.SvoFps = FakeSvoFps  # type: ignore[attr-defined]
    aq_mod.svo_fps = svo_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "astroquery", aq_mod)
    monkeypatch.setitem(sys.modules, "astroquery.svo_fps", svo_mod)


@pytest.mark.parametrize(
    ("prefer", "expect_substr"),
    [
        ("classic", "JWST/NIRCam.F277W"),
        ("2025", "NIRCam2025"),
    ],
)
def test_bandpass_from_svo_mock(
    monkeypatch: pytest.MonkeyPatch, prefer: str, expect_substr: str, tmp_path: Path
) -> None:
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

        def __getitem__(self, k: str) -> Any:
            return self._d[k]

    class FakeRow2025:
        def __getitem__(self, k: str) -> Any:
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
    wl_curve = np.linspace(23673.0, 32203.0, 32)
    tr_curve = np.exp(-0.5 * ((wl_curve - 27617.0) / 2000.0) ** 2)

    _install_fake_astroquery(monkeypatch, fake_table, wl_curve, tr_curve)

    from torchsky.filter_profiles.field_bandpass import bandpass_from_svo

    spec = bandpass_from_svo(
        band_name="F277W",
        facility="JWST",
        instrument="NIRCAM",
        filter_id=None,
        prefer=prefer,  # type: ignore[arg-type]
        cache_dir=tmp_path,
    )
    assert expect_substr in str(spec["svo_filter_id"])
    assert "transmission_wavelength" in spec
    fo = field_observation_spectral_kwargs(spec)
    assert fo["transmission_wavelength"].numel() >= 2


def test_svo_cache_hit_offline(tmp_path: Path) -> None:
    """Verify loading from cache requires zero network and zero astroquery import."""
    from torchsky.filter_profiles.field_bandpass import bandpass_from_svo

    cache_dir = tmp_path / "svo_cache"
    cache_dir.mkdir()

    # Pre-populate filter cache for Subaru/HSC.g
    wl = np.linspace(4000.0, 5500.0, 50).tolist()
    tr = np.exp(-0.5 * ((np.array(wl) - 4800.0) / 300.0) ** 2).tolist()
    filter_file = cache_dir / "filter_Subaru_HSC.g.json"
    import json

    with open(filter_file, "w", encoding="utf-8") as f:
        json.dump({"filter_id": "Subaru/HSC.g", "wavelength": wl, "transmission": tr}, f)

    # 1. Direct by filter_id
    spec = bandpass_from_svo(
        band_name="g",
        facility="Subaru",
        instrument="HSC",
        filter_id="Subaru/HSC.g",
        prefer="classic",
        cache_dir=cache_dir,
    )
    assert spec["svo_filter_id"] == "Subaru/HSC.g"
    assert spec["band_name"] == "G"
    assert 4700.0 < spec["wavelength"] < 4900.0
    assert isinstance(spec["transmission_wavelength"], torch.Tensor)

    # Pre-populate index cache for Subaru HSC
    index_file = cache_dir / "index_Subaru_HSC.json"
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump([{"filterID": "Subaru/HSC.g"}], f)

    # 2. Resolved via facility/instrument (filter_id=None)
    spec2 = bandpass_from_svo(
        band_name="g",
        facility="Subaru",
        instrument="HSC",
        filter_id=None,
        prefer="classic",
        cache_dir=cache_dir,
    )
    assert spec2["svo_filter_id"] == "Subaru/HSC.g"
    assert spec2["band_name"] == "G"


def test_svo_truncated_cache_entry_is_refetched(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A damaged cache entry must trigger a refetch instead of a hard failure."""
    cache_dir = tmp_path / "svo_cache"
    cache_dir.mkdir()
    bad = cache_dir / "filter_Subaru_HSC.g.json"
    # Mismatched column lengths: unusable, but valid JSON.
    bad.write_text(
        '{"filter_id": "Subaru/HSC.g", "wavelength": [4000.0, 4100.0], '
        '"transmission": [0.1]}',
        encoding="utf-8",
    )

    fake_table = [{"filterID": "Subaru/HSC.g"}]
    wl_curve = np.linspace(4000.0, 5500.0, 40)
    tr_curve = np.exp(-0.5 * ((wl_curve - 4800.0) / 300.0) ** 2)
    _install_fake_astroquery(monkeypatch, fake_table, wl_curve, tr_curve)

    from torchsky.filter_profiles.field_bandpass import bandpass_from_svo

    spec = bandpass_from_svo(
        band_name="g",
        facility="Subaru",
        instrument="HSC",
        filter_id="Subaru/HSC.g",
        prefer="classic",
        cache_dir=cache_dir,
    )
    assert spec["svo_filter_id"] == "Subaru/HSC.g"
    assert spec["transmission_wavelength"].numel() == wl_curve.size


def test_svo_cache_write_and_reload(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify SvoFps download populates cache and second call reloads without SvoFps."""
    cache_dir = tmp_path / "svo_cache"

    fake_table = [{"filterID": "Subaru/HSC.r"}]
    wl_curve = np.linspace(5500.0, 7000.0, 40)
    tr_curve = np.exp(-0.5 * ((wl_curve - 6200.0) / 400.0) ** 2)

    _install_fake_astroquery(monkeypatch, fake_table, wl_curve, tr_curve)

    from torchsky.filter_profiles.field_bandpass import bandpass_from_svo

    # First call: writes to cache
    spec1 = bandpass_from_svo(
        band_name="r",
        facility="Subaru",
        instrument="HSC",
        filter_id=None,
        prefer="classic",
        cache_dir=cache_dir,
    )
    assert spec1["svo_filter_id"] == "Subaru/HSC.r"
    assert (cache_dir / "index_Subaru_HSC.json").is_file()
    assert (cache_dir / "filter_Subaru_HSC.r.json").is_file()

    # Now remove astroquery completely from sys.modules to guarantee second call hits cache
    monkeypatch.delitem(sys.modules, "astroquery.svo_fps", raising=False)
    monkeypatch.delitem(sys.modules, "astroquery", raising=False)

    spec2 = bandpass_from_svo(
        band_name="r",
        facility="Subaru",
        instrument="HSC",
        filter_id=None,
        prefer="classic",
        cache_dir=cache_dir,
    )
    assert spec2["svo_filter_id"] == "Subaru/HSC.r"
    assert np.isclose(spec1["wavelength"], spec2["wavelength"])


def test_svo_cache_env_var(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify TORCHSKY_SVO_CACHE environment variable is honored."""
    import json
    from torchsky.filter_profiles.field_bandpass import bandpass_from_svo

    env_dir = tmp_path / "env_svo_cache"
    env_dir.mkdir()
    monkeypatch.setenv("TORCHSKY_SVO_CACHE", str(env_dir))

    filter_file = env_dir / "filter_Test_Cam.Z.json"
    wl = [8000.0, 8500.0, 9000.0]
    tr = [0.1, 0.8, 0.1]
    with open(filter_file, "w", encoding="utf-8") as f:
        json.dump({"filter_id": "Test/Cam.Z", "wavelength": wl, "transmission": tr}, f)

    spec = bandpass_from_svo(
        band_name="Z",
        facility="Test",
        instrument="Cam",
        filter_id="Test/Cam.Z",
        prefer="classic",
    )
    assert spec["svo_filter_id"] == "Test/Cam.Z"
    assert spec["band_name"] == "Z"


def test_svo_cache_bypass(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify use_cache=False bypasses cache reading and writing."""
    cache_dir = tmp_path / "bypass_cache"

    fake_table = [{"filterID": "Test/Cam.i"}]
    wl_curve = np.linspace(7000.0, 8500.0, 30)
    tr_curve = np.ones_like(wl_curve)

    _install_fake_astroquery(monkeypatch, fake_table, wl_curve, tr_curve)

    from torchsky.filter_profiles.field_bandpass import bandpass_from_svo

    spec = bandpass_from_svo(
        band_name="i",
        facility="Test",
        instrument="Cam",
        filter_id="Test/Cam.i",
        prefer="classic",
        cache_dir=cache_dir,
        use_cache=False,
    )
    assert spec["svo_filter_id"] == "Test/Cam.i"
    assert not cache_dir.exists()


def test_svo_cache_atomic_cleanup(tmp_path: Path) -> None:
    """Verify temporary files are cleaned up if serialization fails."""
    from torchsky.filter_profiles.field_bandpass import _atomic_write_json

    target = tmp_path / "corrupt.json"

    class NonSerializable:
        pass

    with pytest.raises(TypeError):
        _atomic_write_json(target, NonSerializable())

    assert not target.exists()
    assert len(list(tmp_path.glob("*.tmp.*"))) == 0


def test_resolve_field_bandpass_svo_cache(tmp_path: Path) -> None:
    """Verify resolve_field_bandpass passes svo_cache_dir and hits cached filter."""
    import json
    from torchsky.filter_profiles import resolve_field_bandpass

    cache_dir = tmp_path / "field_svo_cache"
    cache_dir.mkdir()
    filter_file = cache_dir / "filter_Subaru_HSC.i.json"
    wl = [7000.0, 7600.0, 8200.0]
    tr = [0.05, 0.9, 0.05]
    with open(filter_file, "w", encoding="utf-8") as f:
        json.dump({"filter_id": "Subaru/HSC.i", "wavelength": wl, "transmission": tr}, f)

    spec = resolve_field_bandpass(
        source="svo",
        band_name="i",
        builtin_table={},
        svo_facility="Subaru",
        svo_instrument="HSC",
        svo_filter_id="Subaru/HSC.i",
        svo_prefer="classic",
        svo_cache_dir=cache_dir,
        svo_use_cache=True,
        transmission_path=None,
        transmission_dir=None,
        transmission_unit="angstrom",
        gaussian_lambda_eff=None,
        gaussian_lambda_min=None,
        gaussian_lambda_max=None,
        gaussian_sigma=None,
    )
    assert spec["svo_filter_id"] == "Subaru/HSC.i"
    assert spec["band_name"] == "I"
    assert np.isclose(spec["wavelength"], 7600.0)


def test_photon_quadrature_matches_fine_integral() -> None:
    import torch

    from torchsky.filter_profiles import photon_quadrature

    wl = torch.linspace(1.0, 2.0, 2001, dtype=torch.float64)
    tr = torch.exp(-0.5 * ((wl - 1.5) / 0.15) ** 2)
    nodes, weights = photon_quadrature(wl, tr, 8)
    assert nodes.shape == (8,) and torch.allclose(weights.sum(), torch.tensor(1.0, dtype=torch.float64))
    assert bool((nodes[1:] > nodes[:-1]).all())
    # smooth spectrum: equal-mass quadrature reproduces the photon-weighted band average
    g = lambda lam: 1.0 + 0.7 * lam**2
    dens = tr / wl
    exact = float(torch.trapezoid(g(wl) * dens, wl) / torch.trapezoid(dens, wl))
    approx = float((g(nodes) * weights).sum())
    assert abs(approx - exact) / exact < 2e-3


def test_photon_quadrature_conventions_and_validation() -> None:
    import pytest
    import torch

    from torchsky.filter_profiles import photon_quadrature

    wl = torch.linspace(1.0, 3.0, 501, dtype=torch.float64)
    tr = torch.ones_like(wl)
    fnu, _ = photon_quadrature(wl, tr, 1, flux_convention="fnu")
    flam, _ = photon_quadrature(wl, tr, 1, flux_convention="flambda")
    assert float(fnu) < 2.0 < float(flam)  # 1/lambda weighting pulls the median blueward
    with pytest.raises(ValueError):
        photon_quadrature(wl, torch.zeros_like(wl), 4)
    with pytest.raises(ValueError):
        photon_quadrature(wl, tr, 0)
