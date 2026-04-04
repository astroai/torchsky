from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

from bench_map_smacs0723_field_fit import (  # noqa: E402
    _real_fit_kwargs,
    _robust_intensity_scale,
    _scaled_field_observation,
    _summarize_row_markdown,
)
from bench_map_smacs0723_real import _field_observation_kwargs_for_band  # noqa: E402
from torchsky.map import FieldObservation  # noqa: E402


def test_summarize_row_markdown_lists_models() -> None:
    text = _summarize_row_markdown(
        [
            {
                "model": "coadd_bilinear",
                "support_rmse": 0.1,
                "uniform_rmse": 0.2,
                "support_flux_ratio": 1.01,
                "support_frac": 0.9,
                "crop_mode": "best_overlap",
                "fit_time_ms": 12.3,
            }
        ]
    )
    assert "coadd_bilinear" in text
    assert "best_overlap" in text


def test_field_observation_kwargs_for_band_define_non_degenerate_bandpass() -> None:
    kwargs = _field_observation_kwargs_for_band("F277W")
    assert kwargs["band_name"] == "F277W"
    assert kwargs["spectral_min"] < kwargs["spectral_max"]
    assert kwargs["wavelength"] == kwargs["transmission_center"]


def test_field_observation_kwargs_for_band_can_disable_bandpass() -> None:
    kwargs = _field_observation_kwargs_for_band("F277W", observation_mode="exact_wavelength")
    assert kwargs["band_name"] == "F277W"
    assert kwargs["wavelength"] == 27700.0
    assert kwargs["spectral_min"] is None
    assert kwargs["spectral_max"] is None
    assert kwargs["transmission_center"] is None


def test_robust_intensity_scale_uses_weighted_rms() -> None:
    scale = _robust_intensity_scale(
        [
            (
                np.array([[1.0, 10.0]], dtype="float32"),
                np.array([[1.0, 0.01]], dtype="float32"),
                np.array([[1.0, 1.0]], dtype="float32"),
                {},
            )
        ]
    )
    assert 1.0 <= scale < 2.0


def test_scaled_field_observation_preserves_bandpass_metadata() -> None:
    observation = FieldObservation(
        image=torch.ones((2, 2), dtype=torch.float32),
        header={"NAXIS1": 2, "NAXIS2": 2},
        weight=torch.ones((2, 2), dtype=torch.float32),
        mask=torch.ones((2, 2), dtype=torch.float32),
        wavelength=27700.0,
        spectral_min=24000.0,
        spectral_max=32000.0,
        transmission_center=27700.0,
        transmission_sigma=500.0,
        band_name="F277W",
    )
    scaled = _scaled_field_observation(observation, intensity_scale=10.0)
    assert scaled.wavelength == observation.wavelength
    assert scaled.spectral_min == observation.spectral_min
    assert scaled.spectral_max == observation.spectral_max
    assert scaled.transmission_center == observation.transmission_center
    assert scaled.transmission_sigma == observation.transmission_sigma
    assert scaled.band_name == observation.band_name


def test_real_fit_kwargs_use_pixel_sampling_for_exact_wavelength_debug_mode() -> None:
    kwargs = _real_fit_kwargs(
        "spectral_basis_field",
        crop_size=32,
        steps=4,
        observation_mode="exact_wavelength",
        num_observations=9,
    )
    assert kwargs["num_spectral_samples"] == 1
    assert kwargs["pixel_samples_per_observation"] == 512
    assert kwargs["training_patch_size"] is None
    assert kwargs["observations_per_step"] == 1
