"""Physical invariants for the real-exposure benchmark adapter."""

from dataclasses import fields
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from bench_map_smacs0723_field_fit import _scaled_field_observation, _training_intensity_scale
from bench_map_smacs0723_real import _field_observation_kwargs_for_band, _sci_header
from torchsky.map.field.types import FieldObservation
from torchsky.map.field.bandpass import bandpass_quadrature


def test_scale_preserves_metadata_and_noise():
    plane = torch.ones((2, 2))
    observation = FieldObservation(
        image=plane * 10,
        header={"BUNIT": "MJy/sr", "PHOTMJSR": 3.4, "PIXAR_SR": 2.1e-14, "CRDS_CTX": "jwst_1234.pmap"},
        err=plane * 2,
        weight=plane / 4,
        var_poisson=plane * 3,
        var_rnoise=plane,
        value_unit="MJy/sr",
        spectral_unit="angstrom",
        exposure_time_s=837,
        collecting_area_m2=25,
        sky_background=plane * 5,
        background_offset=2,
    )
    scaled = _scaled_field_observation(observation, intensity_scale=5)
    assert torch.equal(scaled.image, plane * 2)
    assert torch.allclose(scaled.err.square(), plane * 4 / 25)
    assert torch.allclose(scaled.var_poisson + scaled.var_rnoise, plane * 4 / 25)
    assert torch.allclose(scaled.weight, plane * 25 / 4)
    assert torch.equal(scaled.sky_background, plane)
    assert scaled.background_offset == 0.4
    assert scaled.header == observation.header
    changed = {
        "image",
        "header",
        "mask",
        "weight",
        "err",
        "var_poisson",
        "var_rnoise",
        "var_flat",
        "sky_background",
        "instrument_background",
        "detector_background",
        "background_offset",
    }
    for field in fields(observation):
        if field.init and field.name not in changed:
            assert getattr(scaled, field.name) is getattr(observation, field.name)
    with pytest.raises(ValueError, match="positive"):
        _scaled_field_observation(observation, intensity_scale=0)


def test_measured_photon_i_nu_response(tmp_path):
    wavelength = np.linspace(8000.0, 10000.0, 1001)
    transmission = np.ones_like(wavelength)
    np.savez(
        tmp_path / "JWST_NIRCam2025.F090W.npz",
        wavelength_angstrom=wavelength,
        transmission=transmission,
        filter_id="JWST/NIRCam2025.F090W",
    )
    kwargs = _field_observation_kwargs_for_band("F090W", measured_passband_dir=tmp_path)
    observation = FieldObservation(image=torch.zeros((1, 1)), header={}, **kwargs)
    lam, weights = bandpass_quadrature(
        observation,
        num_spectral_samples=1001,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    assert float(weights.sum()) == pytest.approx(1.0)
    assert float((weights * 7).sum()) == pytest.approx(7.0)
    # Independent analytic integral for I_nu(lambda)=lambda under a flat throughput.
    assert float((weights * lam).sum()) == pytest.approx(
        2000 / np.log(10000 / 8000), rel=1e-6
    )
    with pytest.raises(FileNotFoundError):
        _field_observation_kwargs_for_band("F150W", measured_passband_dir=tmp_path)


def test_normalization_never_uses_held_out_pixels():
    plane = torch.ones((2, 2))
    train = FieldObservation(image=plane * 3, header={}, weight=plane, mask=plane)
    held_out = FieldObservation(image=plane * 100, header={}, weight=plane, mask=plane)
    assert _training_intensity_scale([train, held_out], ["train", "test"]) == (3., 1)
    held_out.image.fill_(1.e9)
    assert _training_intensity_scale([train, held_out], ["train", "test"]) == (3., 1)
    assert torch.equal(_scaled_field_observation(train, intensity_scale=3).image, plane)
    with pytest.raises(ValueError, match="training exposures"):
        _training_intensity_scale([held_out], ["test"])


def test_sci_header_retains_primary_calibration_provenance(monkeypatch):
    headers = {"SCI": {"BUNIT": "MJy/sr", "PIXAR_SR": 2.e-14},
               0: {"BUNIT": "wrong primary unit", "PHOTMJSR": 3., "CRDS_CTX": "jwst_1234.pmap"}}
    monkeypatch.setattr("bench_map_smacs0723_real.torchsky.get_header", lambda path, hdu: headers[hdu])
    header = _sci_header(Path("example.fits"))
    assert header == {"BUNIT": "MJy/sr", "PIXAR_SR": 2.e-14, "PHOTMJSR": 3., "CRDS_CTX": "jwst_1234.pmap"}
