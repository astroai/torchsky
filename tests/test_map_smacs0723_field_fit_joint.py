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

from bench_map_smacs0723_field_fit import _field_target_observation  # noqa: E402
from bench_map_smacs0723_field_fit_joint import (  # noqa: E402
    _calibrated_operator_bundles,
    _freeze_calibrated_operator_model,
    _fitted_gaussian_psf_sigma_pix,
    _format_markdown,
    _summarize_by_model,
)
from bench_map_smacs0723_field_fit_joint import _background_subtracted_source_stats  # noqa: E402
from bench_map_smacs0723_field_fit_joint import _foam_training_tile_side  # noqa: E402
from bench_map_smacs0723_field_fit_joint import _initialize_foam_from_band_coadds  # noqa: E402
from bench_map_smacs0723_field_fit_joint import _joint_real_fit_kwargs  # noqa: E402
from bench_map_smacs0723_field_fit_joint import _preview_uint8  # noqa: E402
from bench_map_smacs0723_field_fit_joint import _regular_foam_model_kwargs  # noqa: E402
from bench_map_smacs0723_field_fit_joint import _set_foam_geometry_trainable  # noqa: E402
from bench_map_smacs0723_field_fit_joint import _source_metric_mask  # noqa: E402
from bench_map_smacs0723_field_fit_joint import _write_export_products  # noqa: E402
from torchsky.map.field import (  # noqa: E402
    FieldDomain,
    FieldObservation,
    SpectralPowerFoam,
    SpectralPowerFoamField,
    make_tangent_plane_header,
)
from torchsky.map import ForwardContext  # noqa: E402
from torchsky.map.operators import LearnedBackgroundOp, LearnedPSFOp  # noqa: E402
from torchsky.map.operators import (  # noqa: E402
    ConditionedBackgroundField,
    SpatioSpectralTemporalEncoder,
)


def test_frozen_calibrated_background_follows_scene_intensity_scale() -> None:
    encoder = SpatioSpectralTemporalEncoder(
        spatial_center=(0.0, 0.0),
        spatial_scale=(1.0, 1.0),
        wavelength_center=15_000.0,
        spectral_scale=1_000.0,
        time_center_mjd=59_737.0,
        temporal_scale_days=1.0,
        spatial_frequencies=0,
        spectral_frequencies=0,
        temporal_frequencies=0,
    )
    background = ConditionedBackgroundField(encoder, hidden_dim=2, hidden_layers=1)
    output = background.network[-1]
    assert isinstance(output, torch.nn.Linear)
    with torch.no_grad():
        output.weight.fill_(4.0)
        output.bias.fill_(2.0)

    _freeze_calibrated_operator_model(
        background, role="background", intensity_scale=10.0
    )

    torch.testing.assert_close(output.weight, torch.full_like(output.weight, 0.4))
    torch.testing.assert_close(output.bias, torch.full_like(output.bias, 0.2))
    assert not background.training
    assert not any(parameter.requires_grad for parameter in background.parameters())


def test_calibrated_bundles_select_frozen_models_by_detector() -> None:
    observation = FieldObservation(
        image=torch.zeros((4, 4)),
        header={"NAXIS1": 4, "NAXIS2": 4},
        wavelength=15_000.0,
    )
    psf = torch.nn.Linear(1, 1)
    background = torch.nn.Linear(1, 1)
    psf.requires_grad_(False)
    background.requires_grad_(False)

    bundles = _calibrated_operator_bundles(
        [observation],
        [ForwardContext(detector="NRCB4", time_mjd=59_737.2)],
        {"NRCB4": {"psf": psf, "background": background}},
    )

    assert len(bundles) == 1
    assert isinstance(bundles[0].psf, LearnedPSFOp)
    assert isinstance(bundles[0].background, LearnedBackgroundOp)
    assert bundles[0].psf.model is psf
    assert bundles[0].background.model is background
    assert bundles[0].background.coordinate_frame == "detector"


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


def test_joint_fitted_psf_uses_frozen_band_contract_and_local_pixel_scale() -> None:
    scale_arcsec = 0.01
    header = {
        "NAXIS": 2,
        "NAXIS1": 16,
        "NAXIS2": 16,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 8.5,
        "CRPIX2": 8.5,
        "CRVAL1": 110.0,
        "CRVAL2": -73.0,
        "CD1_1": -scale_arcsec / 3600.0,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": scale_arcsec / 3600.0,
    }

    sigma_pix = _fitted_gaussian_psf_sigma_pix("F090W", header)
    expected = (0.033 * 1.05) / 2.354820045 / scale_arcsec
    target = _field_target_observation(
        header,
        band_name="F090W",
        psf_sigma_pix=sigma_pix,
    )

    assert sigma_pix == pytest.approx(expected, rel=1.0e-5)
    assert target.psf_spec is not None
    assert target.psf_spec.mode == "gaussian_sigma_pix"
    assert target.psf_spec.sigma_pix == pytest.approx(sigma_pix)


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
    assert kwargs["spatial_sampling"] == "gauss_legendre_2x2"
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
    fixed = _joint_real_fit_kwargs(
        "spectral_power_foam",
        crop_size=512,
        steps=2,
        num_observations=4,
        foam_grid_side=256,
        foam_fixed_geometry=True,
    )
    assert fixed["training_patch_size"] == 128
    assert kwargs["training_patch_size"] == 32
    assert kwargs["diagnostic_spatial_tile_hw"] == (32, 32)


def test_joint_foam_coadd_initializer_covers_cells_and_spectral_bands() -> None:
    header = make_tangent_plane_header(
        5,
        5,
        crval_ra_deg=10.0,
        crval_dec_deg=20.0,
    )
    domain = FieldDomain(
        center_ra_deg=10.0,
        center_dec_deg=20.0,
        spatial_scale_deg=0.1,
        wavelength_center=2.0,
        wavelength_scale=1.0,
    )
    model = SpectralPowerFoamField(
        domain,
        sites=torch.zeros((1, 2), dtype=torch.float32),
        radii=torch.ones(1, dtype=torch.float32),
        cell_coefficients=torch.zeros((1, 2), dtype=torch.float32),
        compact_fixed_geometry=True,
    )
    model.sites.requires_grad_(False)
    model.foam.raw_radii.requires_grad_(False)
    entries = []
    for wavelength, value in ((1.0, 2.0), (3.0, 4.0)):
        observation = FieldObservation(
            image=torch.zeros((5, 5)),
            header=dict(header),
            weight=torch.ones((5, 5)),
            mask=torch.ones((5, 5), dtype=torch.bool),
            wavelength=wavelength,
            band_name=f"B{wavelength:g}",
        )
        entries.append(
            {
                "coadd_pred": np.full((5, 5), value, dtype=np.float32),
                "support_mask": np.ones((5, 5), dtype=bool),
                "target_header": dict(header),
                "target_observation": observation,
            }
        )

    diagnostics = _initialize_foam_from_band_coadds(
        model,
        entries,
        intensity_scale=1.0,
        num_spectral_samples=1,
    )
    basis = torch.stack(
        (
            model.spectral_basis(torch.tensor(1.0)),
            model.spectral_basis(torch.tensor(3.0)),
        )
    )
    reconstructed = basis @ model.cell_coefficients.detach().mT

    torch.testing.assert_close(
        reconstructed[:, 0],
        torch.tensor([2.0, 4.0]),
        atol=2.0e-5,
        rtol=0,
    )
    assert diagnostics["foam_initialized_cell_fraction"] == 1.0
    assert diagnostics["foam_fully_covered_cell_fraction"] == 1.0


def test_joint_foam_coadd_initializer_populates_affine_gradients() -> None:
    header = make_tangent_plane_header(
        7,
        7,
        crval_ra_deg=10.0,
        crval_dec_deg=20.0,
    )
    domain = FieldDomain(
        center_ra_deg=10.0,
        center_dec_deg=20.0,
        spatial_scale_deg=0.1,
        wavelength_center=2.0,
        wavelength_scale=1.0,
    )
    model = SpectralPowerFoamField(
        domain,
        sites=torch.zeros((1, 2), dtype=torch.float32),
        radii=torch.ones(1, dtype=torch.float32),
        cell_coefficients=torch.zeros((1, 1), dtype=torch.float32),
        cell_gradient_coefficients=torch.zeros((1, 1, 2), dtype=torch.float32),
        compact_fixed_geometry=True,
    )
    model.sites.requires_grad_(False)
    model.foam.raw_radii.requires_grad_(False)
    y, x = np.mgrid[:7, :7]
    observation = FieldObservation(
        image=torch.zeros((7, 7)),
        header=dict(header),
        weight=torch.ones((7, 7)),
        mask=torch.ones((7, 7), dtype=torch.bool),
        wavelength=2.0,
        band_name="B2",
    )

    diagnostics = _initialize_foam_from_band_coadds(
        model,
        [
            {
                "coadd_pred": (1.0 + 0.2 * x - 0.1 * y).astype(np.float32),
                "support_mask": np.ones((7, 7), dtype=bool),
                "target_header": dict(header),
                "target_observation": observation,
            }
        ],
        intensity_scale=1.0,
        num_spectral_samples=1,
    )

    assert model.cell_gradient_coefficients is not None
    assert float(model.cell_gradient_coefficients.detach().abs().max()) > 0.0
    assert diagnostics["foam_initial_gradient_coefficient_abs_max"] > 0.0


def test_regular_foam_initialization_is_deterministic_and_pixel_independent() -> None:
    first = _regular_foam_model_kwargs(grid_side=3, basis_count=4)
    second = _regular_foam_model_kwargs(grid_side=3, basis_count=4)

    assert first["sites"].shape == (9, 2)
    assert first["cell_coefficients"].shape == (9, 4)
    assert bool(torch.all(first["radii"] > 0))
    torch.testing.assert_close(first["sites"], second["sites"])
    torch.testing.assert_close(first["radii"], second["radii"])

    affine = _regular_foam_model_kwargs(
        grid_side=3,
        basis_count=4,
        affine_spatial_basis=True,
    )
    assert affine["cell_gradient_coefficients"].shape == (9, 4, 2)
    assert not bool(torch.any(affine["cell_gradient_coefficients"]))


def test_regular_foam_footprint_extent_covers_normalized_domain() -> None:
    kwargs = _regular_foam_model_kwargs(grid_side=16, basis_count=4, extent=0.55)
    sites = kwargs["sites"]
    radii = kwargs["radii"]
    corners = torch.tensor([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
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
