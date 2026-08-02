from __future__ import annotations
import pytest

pytest.importorskip("PIL")

import sys
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("PIL")

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

from bench_map_smacs0723_field_fit_joint import _format_markdown, _summarize_by_model  # noqa: E402
from bench_map_smacs0723_field_fit_joint import _background_subtracted_source_stats  # noqa: E402
from bench_map_smacs0723_field_fit_joint import _foam_training_tile_side  # noqa: E402
from bench_map_smacs0723_field_fit_joint import _joint_real_fit_kwargs  # noqa: E402
from bench_map_smacs0723_field_fit_joint import _preview_uint8  # noqa: E402
from bench_map_smacs0723_field_fit_joint import _regular_foam_model_kwargs  # noqa: E402
from bench_map_smacs0723_field_fit_joint import _set_foam_geometry_trainable  # noqa: E402
from bench_map_smacs0723_field_fit_joint import _source_metric_mask  # noqa: E402
from bench_map_smacs0723_field_fit_joint import _write_export_products  # noqa: E402
from torchsky.map.field import SpectralPowerFoam  # noqa: E402


def test_joint_benchmark_import_does_not_require_astropy() -> None:
    code = f"""
import builtins
import sys
sys.path.insert(0, {str(BENCHMARKS)!r})
original_import = builtins.__import__
def without_astropy(name, *args, **kwargs):
    if name == 'astropy' or name.startswith('astropy.'):
        raise ModuleNotFoundError(name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = without_astropy
import bench_map_smacs0723_field_fit_joint
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_joint_field_fit_markdown_lists_summary_and_band_rows() -> None:
    text = _format_markdown(
        [
            {
                "band": "F090W",
                "model": "spectral_basis_field",
                "status": "OK",
                "n_exposures": 36,
                "source_contrast_rmse": 0.04,
                "source_rmse": 0.08,
                "support_rmse": 0.1,
                "uniform_rmse": 0.2,
                "aligned_rmse": 0.05,
                "affine_slope": 1.0,
                "source_contrast_slope": 0.9,
                "support_flux_ratio": 1.01,
                "source_flux_ratio": 0.95,
                "support_frac": 0.9,
                "source_frac": 0.2,
                "fit_time_ms": 12.3,
            }
        ]
    )
    assert "Model Summary" in text
    assert "F090W" in text
    assert "spectral_basis_field" in text


def test_joint_field_fit_summary_averages_rows_by_model() -> None:
    summary = _summarize_by_model(
        [
            {
                "band": "F090W",
                "model": "m",
                "status": "OK",
                "source_contrast_rmse": 0.04,
                "source_rmse": 0.08,
                "support_rmse": 0.1,
                "uniform_rmse": 0.3,
                "aligned_rmse": 0.05,
                "affine_slope": 1.0,
                "source_contrast_slope": 0.9,
                "support_flux_ratio": 1.0,
                "source_flux_ratio": 0.95,
                "fit_time_ms": 5.0,
            },
            {
                "band": "F277W",
                "model": "m",
                "status": "OK",
                "source_contrast_rmse": 0.10,
                "source_rmse": 0.18,
                "support_rmse": 0.3,
                "uniform_rmse": 0.5,
                "aligned_rmse": 0.15,
                "affine_slope": 0.8,
                "source_contrast_slope": 0.7,
                "support_flux_ratio": 0.8,
                "source_flux_ratio": 0.75,
                "fit_time_ms": 7.0,
            },
        ]
    )
    assert len(summary) == 1
    assert summary[0]["model"] == "m"
    assert abs(summary[0]["mean_support_rmse"] - 0.2) < 1.0e-6
    assert abs(summary[0]["mean_source_rmse"] - 0.13) < 1.0e-6
    assert abs(summary[0]["mean_source_contrast_rmse"] - 0.07) < 1.0e-6
    assert summary[0]["bands"] == 2
    assert abs(summary[0]["mean_aligned_rmse"] - 0.1) < 1.0e-6
    assert abs(summary[0]["mean_affine_slope"] - 0.9) < 1.0e-6


def test_joint_real_fit_kwargs_use_pixel_sampling() -> None:
    kwargs = _joint_real_fit_kwargs(
        "spectral_basis_field",
        crop_size=32,
        steps=4,
        num_observations=20,
    )
    assert kwargs["observations_per_step"] == 12
    assert kwargs["training_patch_size"] is None
    assert kwargs["pixel_samples_per_observation"] >= 256


def test_joint_real_foam_fit_bounds_psf_training_and_diagnostics() -> None:
    kwargs = _joint_real_fit_kwargs(
        "spectral_power_foam",
        crop_size=512,
        steps=50,
        num_observations=16,
    )

    assert kwargs["training_patch_size"] == 128
    assert kwargs["pixel_samples_per_observation"] is None
    assert kwargs["shared_sky_patch_sampling"] is True
    assert kwargs["streaming_accumulation"] is True
    assert kwargs["diagnostic_spatial_tile_hw"] == (64, 64)
    assert kwargs["lr"] == pytest.approx(2.0e-2)


def test_joint_real_foam_fit_scales_tiles_with_grid_capacity() -> None:
    kwargs = _joint_real_fit_kwargs(
        "spectral_power_foam",
        crop_size=512,
        steps=50,
        num_observations=16,
        foam_grid_side=256,
    )

    assert _foam_training_tile_side(crop_size=512, grid_side=256) == 32
    assert kwargs["training_patch_size"] == 32
    assert kwargs["diagnostic_spatial_tile_hw"] == (32, 32)


def test_regular_foam_initialization_is_deterministic_and_pixel_independent() -> None:
    first = _regular_foam_model_kwargs(grid_side=3, basis_count=4)
    second = _regular_foam_model_kwargs(grid_side=3, basis_count=4)

    assert first["sites"].shape == (9, 2)
    assert first["cell_coefficients"].shape == (9, 4)
    assert bool(torch.all(first["radii"] > 0))
    torch.testing.assert_close(first["sites"], second["sites"])
    torch.testing.assert_close(first["radii"], second["radii"])


def test_regular_foam_footprint_extent_covers_normalized_domain() -> None:
    kwargs = _regular_foam_model_kwargs(grid_side=16, basis_count=4, extent=0.55)
    sites = kwargs["sites"]
    radii = kwargs["radii"]
    corners = torch.tensor(
        [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]
    )
    distance = torch.cdist(corners, sites)

    assert bool(torch.all(distance.min(dim=1).values <= radii.max()))


def test_regular_foam_temperature_is_scale_invariant_in_cell_units() -> None:
    coarse = _regular_foam_model_kwargs(grid_side=16, basis_count=4, extent=0.55)
    fine = _regular_foam_model_kwargs(grid_side=64, basis_count=4, extent=0.55)
    coarse_step = 2.0 * 0.55 / 16.0
    fine_step = 2.0 * 0.55 / 64.0

    assert coarse["temperature_start"] * coarse_step**2 == pytest.approx(1.0)
    assert fine["temperature_start"] * fine_step**2 == pytest.approx(1.0)
    assert coarse["temperature_end"] * coarse_step**2 == pytest.approx(16.0)
    assert fine["temperature_end"] * fine_step**2 == pytest.approx(16.0)


def test_regular_foam_final_temperature_localizes_cell_values() -> None:
    kwargs = _regular_foam_model_kwargs(grid_side=4, basis_count=1, extent=0.55)
    coefficients = torch.arange(16, dtype=torch.float32).unsqueeze(-1)
    model = SpectralPowerFoam(
        kwargs["sites"],
        kwargs["radii"],
        coefficients,
        minimum_radius=kwargs["minimum_radius"],
    )

    prediction = model(
        kwargs["sites"],
        torch.ones((16, 1), dtype=torch.float32),
        power_temperature=kwargs["temperature_end"],
        support_temperature=kwargs["temperature_end"],
    )

    torch.testing.assert_close(prediction, coefficients[:, 0], atol=2.0e-6, rtol=0)


def test_fixed_foam_freezes_only_geometry() -> None:
    kwargs = _regular_foam_model_kwargs(grid_side=4, basis_count=1, extent=0.55)
    model = SpectralPowerFoam(
        kwargs["sites"],
        kwargs["radii"],
        kwargs["cell_coefficients"],
        minimum_radius=kwargs["minimum_radius"],
    )

    _set_foam_geometry_trainable(model, False)

    assert not model.sites.requires_grad
    assert not model.raw_radii.requires_grad
    assert model.cell_coefficients.requires_grad


def test_source_metric_mask_prefers_bright_reference_structure() -> None:
    pred = np.ones((4, 4), dtype=np.float32)
    reference = np.zeros((4, 4), dtype=np.float32)
    reference[1, 1] = 10.0
    reference[2, 2] = 8.0
    support = np.ones((4, 4), dtype=bool)

    mask = _source_metric_mask(pred, reference, support)

    assert mask[1, 1]
    assert mask[2, 2]
    assert mask.sum() <= support.sum()


def test_background_subtracted_source_stats_remove_pedestal() -> None:
    reference = np.zeros((6, 6), dtype=np.float32)
    reference[2:4, 2:4] = np.array([[3.0, 5.0], [4.0, 6.0]], dtype=np.float32)
    pred = reference + 2.0
    support = np.ones((6, 6), dtype=bool)

    stats = _background_subtracted_source_stats(pred, reference, support)

    assert stats["source_rmse"] > 1.0
    assert stats["source_contrast_rmse"] < 1.0e-6


def test_preview_uint8_returns_byte_image() -> None:
    image = np.array([[0.0, 1.0], [2.0, np.nan]], dtype=np.float32)
    preview = _preview_uint8(image)
    assert preview.dtype == np.uint8
    assert preview.shape == image.shape


def test_write_export_products_writes_fits_and_png() -> None:
    reference = np.arange(16, dtype=np.float32).reshape(4, 4)
    prediction = reference + 1.0
    support_mask = np.ones((4, 4), dtype=bool)
    source_mask = np.zeros((4, 4), dtype=bool)
    source_mask[1:3, 1:3] = True
    header = {"NAXIS": 2, "NAXIS1": 4, "NAXIS2": 4}

    with tempfile.TemporaryDirectory() as tmp:
        _write_export_products(
            Path(tmp),
            band="F090W",
            model="spectral_basis_field",
            header=header,
            reference=reference,
            prediction=prediction,
            support_mask=support_mask,
            source_mask=source_mask,
        )
        out = Path(tmp) / "f090w" / "spectral_basis_field"
        assert (out / "reference.fits").exists()
        assert (out / "prediction.fits").exists()
        assert (out / "residual.fits").exists()
        assert (out / "support_mask.fits").exists()
        assert (out / "source_mask.fits").exists()
        assert (out / "reference.png").exists()
        assert (out / "prediction.png").exists()
        assert (out / "residual.png").exists()
