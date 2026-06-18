from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn as nn

from torchsky import WCS
from torchsky.map import (
    BandpassConditionedField,
    FieldDomain,
    FieldFitResult,
    FieldObservation,
    PSFSpec,
    field_image_residual_product,
    field_image_quality_metrics,
    field_quality_acceptance_report,
    field_quality_scorecard,
    fit_field_model,
    make_residual_tile_basis_field,
    render_field_fit_quality,
    render_field_quality,
    render_field_residual,
    render_field_residual_product,
    select_field_residual_tiles,
    pretrain_strainer_trunk_synthetic,
    StrainerSkyEncoderMLPField,
)
from torchsky.map.field.core import _BaseFieldModel
from torchsky.map.field.observation_ops import (
    coordinate_grid,
    observation_weight_sum_scalar,
)
from torchsky.map.operators.bundle import ObservationOperatorBundle


def _tan_header(
    size: int,
    *,
    crval1: float,
    crval2: float,
    scale_deg: float,
) -> dict:
    return {
        "NAXIS": 2,
        "NAXIS1": size,
        "NAXIS2": size,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": size / 2.0,
        "CRPIX2": size / 2.0,
        "CRVAL1": crval1,
        "CRVAL2": crval2,
        "CD1_1": -scale_deg,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": scale_deg,
    }


class MockFieldModel(_BaseFieldModel):
    def __init__(self, domain: FieldDomain) -> None:
        super().__init__(domain)
        # Use coord_dim from _BaseFieldModel
        self.fc = nn.Linear(self.coord_dim, 1)
        self.logvar_param = nn.Parameter(torch.tensor([-1.0]))

    def forward(
        self, ra: torch.Tensor, dec: torch.Tensor, lam: torch.Tensor | float
    ) -> torch.Tensor:
        coords = self._encoded_coords(ra, dec, lam)
        out = self.fc(coords.to(dtype=self.fc.weight.dtype)).squeeze(-1)
        return torch.sigmoid(out) + 1.0e-4

    def forward_distribution(
        self, ra: torch.Tensor, dec: torch.Tensor, lam: torch.Tensor | float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.forward(ra, dec, lam)
        # Construct logvar that varies spatially
        logvar = torch.ones_like(mean) * self.logvar_param
        # Make outer/different parts have higher uncertainty
        logvar = logvar + (ra - float(self.domain.center_ra_deg)).abs() * 2.0
        return mean, logvar


class _ZeroField(_BaseFieldModel):
    def forward(
        self,
        ra: torch.Tensor,
        dec: torch.Tensor,
        lam: torch.Tensor | float,
    ) -> torch.Tensor:
        del dec, lam
        return torch.zeros_like(ra, dtype=torch.float32)


def test_field_diagnostics_computation() -> None:
    domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=900.0,
        coord_mode="sphere_xyz",
    )
    model = MockFieldModel(domain)

    # Construct mock observations
    header = _tan_header(8, crval1=150.0, crval2=2.0, scale_deg=0.01)

    # 1. Observation with mask to test uncertainty_stats
    mask = torch.ones((8, 8), dtype=torch.float32)
    # Mask out the outer parts (RA increases horizontally, CRPIX is in the middle)
    mask[:, :2] = 0.0
    mask[:, -2:] = 0.0

    obs = FieldObservation(
        image=torch.ones((8, 8), dtype=torch.float32),
        header=header,
        weight=torch.ones((8, 8), dtype=torch.float32),
        mask=mask,
        wavelength=5600.0,
        psf_sigma_pix=1.5,
    )

    # Run fit_field_model with a small number of steps
    result = fit_field_model(
        observations=[obs],
        model=model,
        steps=5,
        lr=1.0e-3,
        num_spectral_samples=3,
        loss_kind="gaussian_nll",
        gradient_residual_weight=0.05,
    )

    # Verify diagnostics dict is populated and structure is correct
    diagnostics = result.diagnostics
    assert diagnostics is not None
    assert isinstance(diagnostics, dict)

    # Verify coordinate leakage metrics
    assert "spatial_spectral_leakage" in diagnostics
    assert "gradient_correlation" in diagnostics
    assert isinstance(diagnostics["spatial_spectral_leakage"], float)
    assert isinstance(diagnostics["gradient_correlation"], float)

    assert "quality_metrics" in diagnostics
    quality = diagnostics["quality_metrics"]
    assert quality["weighted_rmse"] >= 0.0
    assert quality["residual_snr"] >= 0.0
    assert quality["prediction_sharpness"] >= 0.0

    # Verify PSF/noise alignment metrics
    assert "reduced_chi2" in diagnostics
    assert "psf_alignment" in diagnostics
    assert isinstance(diagnostics["reduced_chi2"], float)
    assert isinstance(diagnostics["psf_alignment"], float)

    # Verify support-region vs full-grid uncertainty propagation
    assert "uncertainty_stats" in diagnostics
    stats = diagnostics["uncertainty_stats"]
    assert "support_mean_uncertainty" in stats
    assert "outer_mean_uncertainty" in stats
    assert "uncertainty_ratio" in stats
    assert stats["support_mean_uncertainty"] > 0.0
    assert stats["outer_mean_uncertainty"] > 0.0
    # Because RA deviation increases uncertainty, outer_mean should be positive
    assert stats["uncertainty_ratio"] > 0.0

    # Verify performance profile
    assert "performance_profile" in diagnostics
    profile = diagnostics["performance_profile"]
    assert "total_duration_sec" in profile
    assert "step_duration_mean_ms" in profile
    assert "throughput_pixels_per_sec" in profile
    assert "max_memory_allocated_bytes" in profile
    assert profile["total_duration_sec"] >= 0.0
    assert profile["step_duration_mean_ms"] >= 0.0
    assert profile["throughput_pixels_per_sec"] >= 0.0


def test_strainer_pretraining_performance_profiling() -> None:
    domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=900.0,
        coord_mode="sphere_xyz",
    )
    model = StrainerSkyEncoderMLPField(domain)

    # Run synthetic trunk pretraining
    profile = pretrain_strainer_trunk_synthetic(
        model,
        steps=5,
        lr=1.0e-3,
        batch_size=128,
        noise_kind="gaussian",
    )

    # Verify profile dict is returned and has correct structure
    assert profile is not None
    assert isinstance(profile, dict)
    assert "total_duration_sec" in profile
    assert "step_duration_mean_ms" in profile
    assert "throughput_samples_per_sec" in profile
    assert "max_memory_allocated_bytes" in profile
    assert profile["total_duration_sec"] >= 0.0
    assert profile["step_duration_mean_ms"] >= 0.0
    assert profile["throughput_samples_per_sec"] >= 0.0


def test_bandpass_conditioned_field_detaches_representative_wavelength() -> None:
    domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=900.0,
        coord_mode="sphere_xyz",
    )
    model = BandpassConditionedField(domain, hidden_dim=16, bandpass_hidden_dim=8)
    ra = torch.linspace(149.99, 150.01, 4)
    dec = torch.linspace(1.99, 2.01, 4)
    wavelengths = torch.tensor(
        [5200.0, 5600.0, 6100.0],
        dtype=torch.float32,
        requires_grad=True,
    )
    weights = torch.tensor([0.2, 0.6, 0.2], dtype=torch.float64)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rendered = model.render_bandpass(ra, dec, wavelengths, weights)

    assert rendered.shape == ra.shape
    assert not any(
        "requires_grad=True to a scalar" in str(item.message) for item in caught
    )


def test_field_observation_coordinate_cache_tracks_header_mutation() -> None:
    header = _tan_header(5, crval1=150.0, crval2=2.0, scale_deg=0.01)
    observation = FieldObservation(
        image=torch.ones((5, 5), dtype=torch.float32),
        header=header,
        wavelength=5600.0,
    )

    ra_initial, dec_initial = coordinate_grid(
        observation,
        shape=(5, 5),
        device=torch.device("cpu"),
    )
    assert observation._cached_coordinate_signature is not None

    observation.header["CRVAL1"] = 151.0
    ra_shifted, dec_shifted = coordinate_grid(
        observation,
        shape=(5, 5),
        device=torch.device("cpu"),
    )

    assert observation._cached_coordinate_signature is not None
    assert not torch.allclose(ra_initial, ra_shifted)
    torch.testing.assert_close(dec_initial, dec_shifted, atol=2.0e-4, rtol=0.0)


def test_field_quality_metrics_track_snr_and_sharpness() -> None:
    target = torch.zeros((5, 5), dtype=torch.float32)
    target[:, 3:] = 1.0
    blurred = target.clone()
    blurred[:, 2] = 0.5
    weight = torch.ones_like(target)
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[0, :] = False

    metrics = field_image_quality_metrics(
        blurred,
        target,
        weight=weight,
        mask=mask,
    )

    assert metrics["weighted_rmse"] > 0.0
    assert metrics["residual_snr"] > 0.0
    assert metrics["prediction_sharpness"] < metrics["target_sharpness"]
    assert 0.0 < metrics["support_fraction"] < 1.0
    assert abs(metrics["residual_rmse_scale_1"] - metrics["weighted_rmse"]) < 1.0e-7
    assert metrics["residual_rmse_scale_2"] >= 0.0
    assert metrics["residual_rmse_scale_4"] >= 0.0


def test_field_quality_metrics_ignore_nonfinite_pixels_and_weights() -> None:
    prediction = torch.tensor(
        [[1.0, float("nan")], [2.0, 3.0]],
        dtype=torch.float32,
    )
    target = torch.tensor(
        [[1.0, 5.0], [float("inf"), 0.0]],
        dtype=torch.float32,
    )
    weight = torch.tensor(
        [[2.0, 1.0], [1.0, float("nan")]],
        dtype=torch.float32,
    )

    metrics = field_image_quality_metrics(prediction, target, weight=weight)

    assert metrics["weighted_rmse"] == 0.0
    assert metrics["effective_weight_sum"] == 2.0
    assert metrics["support_fraction"] == 0.25


def test_field_residual_product_carries_support_weight_and_metrics() -> None:
    prediction = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    target = torch.tensor([[1.5, 2.0], [0.0, 4.0]], dtype=torch.float32)
    weight = torch.tensor([[2.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    mask = torch.tensor([[1.0, 1.0], [0.0, 1.0]], dtype=torch.float32)

    product = field_image_residual_product(
        prediction,
        target,
        weight=weight,
        mask=mask,
        band_name="r",
    )

    assert product.band_name == "r"
    assert product.prediction.shape == prediction.shape
    assert product.residual[0, 0] == -0.5
    assert product.residual[1, 0] == 0.0
    assert product.support.tolist() == [[True, False], [False, True]]
    assert product.metrics["effective_weight_sum"] == 3.0
    assert product.metrics["support_fraction"] == 0.5


def test_field_residual_product_reports_psf_alignment_for_psf_observations() -> None:
    prediction = torch.zeros((5, 5), dtype=torch.float32)
    prediction[2, 2] = 1.0
    target = torch.zeros_like(prediction)
    weight = torch.ones_like(prediction)
    mask = torch.ones_like(prediction)

    no_psf = field_image_residual_product(
        prediction,
        target,
        weight=weight,
        mask=mask,
    )
    with_psf = field_image_residual_product(
        prediction,
        target,
        weight=weight,
        mask=mask,
        psf_spec=PSFSpec.gaussian(1.0),
    )
    scorecard = field_quality_scorecard((no_psf.metrics, with_psf.metrics))

    alignment = with_psf.metrics["psf_residual_alignment"]
    assert "psf_residual_alignment" not in no_psf.metrics
    assert -1.0 <= alignment <= 1.0
    assert scorecard["aggregate"]["psf_residual_alignment"] == alignment


def test_field_quality_scorecard_groups_bands_and_tracks_extrema() -> None:
    observations = (
        FieldObservation(
            image=torch.ones((2, 2), dtype=torch.float32),
            header=_tan_header(2, crval1=150.0, crval2=2.0, scale_deg=0.01),
            wavelength=5000.0,
            band_name="g",
        ),
        FieldObservation(
            image=torch.ones((2, 2), dtype=torch.float32),
            header=_tan_header(2, crval1=150.0, crval2=2.0, scale_deg=0.01),
            wavelength=6500.0,
            band_name="r",
        ),
        FieldObservation(
            image=torch.ones((2, 2), dtype=torch.float32),
            header=_tan_header(2, crval1=150.0, crval2=2.0, scale_deg=0.01),
            wavelength=6600.0,
            band_name="r",
        ),
    )
    per_observation = (
        {
            "weighted_rmse": 0.2,
            "residual_snr": 5.0,
            "support_fraction": 1.0,
            "effective_weight_sum": 4.0,
        },
        {
            "weighted_rmse": 0.5,
            "residual_snr": 2.0,
            "support_fraction": 0.75,
            "effective_weight_sum": 3.0,
        },
        {
            "weighted_rmse": 0.1,
            "residual_snr": 9.0,
            "support_fraction": 0.5,
            "effective_weight_sum": 1.0,
        },
    )

    scorecard = field_quality_scorecard(
        per_observation,
        observations=observations,
    )

    assert set(scorecard["by_band"]) == {"g", "r"}
    assert scorecard["by_band"]["r"]["observation_count"] == 2
    assert scorecard["extrema"]["highest_weighted_rmse"]["index"] == 1
    assert scorecard["extrema"]["lowest_residual_snr"]["index"] == 1
    assert scorecard["extrema"]["lowest_support_fraction"]["index"] == 2


def test_field_quality_acceptance_report_flags_explicit_threshold_violations() -> None:
    observations = (
        FieldObservation(
            image=torch.ones((2, 2), dtype=torch.float32),
            header=_tan_header(2, crval1=150.0, crval2=2.0, scale_deg=0.01),
            wavelength=5000.0,
            band_name="g",
        ),
        FieldObservation(
            image=torch.ones((2, 2), dtype=torch.float32),
            header=_tan_header(2, crval1=150.0, crval2=2.0, scale_deg=0.01),
            wavelength=6500.0,
            band_name="r",
        ),
    )
    metrics = (
        {
            "weighted_rmse": 0.05,
            "residual_snr": 20.0,
            "support_fraction": 0.95,
            "effective_weight_sum": 10.0,
            "psf_residual_alignment": 0.1,
        },
        {
            "weighted_rmse": 0.2,
            "residual_snr": 3.0,
            "support_fraction": 0.4,
            "effective_weight_sum": 2.0,
        },
    )

    report = field_quality_acceptance_report(
        metrics,
        observations=observations,
        max_weighted_rmse=0.1,
        min_residual_snr=5.0,
        min_support_fraction=0.5,
        min_effective_weight_sum=3.0,
        max_abs_psf_residual_alignment=0.2,
    )

    assert report["passed"] is False
    assert report["scorecard"]["by_band"]["g"]["observation_count"] == 1
    assert {
        (violation["index"], violation["metric"], violation.get("reason"))
        for violation in report["violations"]
    } == {
        (1, "weighted_rmse", None),
        (1, "residual_snr", None),
        (1, "support_fraction", None),
        (1, "effective_weight_sum", None),
        (1, "psf_residual_alignment", "missing"),
    }


def test_select_field_residual_tiles_ranks_weighted_residual_energy() -> None:
    prediction = torch.zeros((4, 4), dtype=torch.float32)
    target = torch.zeros((4, 4), dtype=torch.float32)
    prediction[:2, :2] = 1.0
    prediction[2:, 2:] = 3.0
    weight = torch.ones((4, 4), dtype=torch.float32)
    weight[:2, :2] = 4.0
    mask = torch.ones((4, 4), dtype=torch.float32)
    mask[2:, :2] = 0.0
    product = field_image_residual_product(
        prediction,
        target,
        weight=weight,
        mask=mask,
        band_name="r",
    )

    tiles = select_field_residual_tiles(
        product,
        tile_hw=(2, 2),
        max_tiles=2,
        min_support_fraction=0.25,
    )

    assert len(tiles) == 2
    assert tiles[0].band_name == "r"
    assert (tiles[0].y0, tiles[0].x0) == (2, 2)
    assert tiles[0].residual_energy == 36.0
    assert (tiles[1].y0, tiles[1].x0) == (0, 0)
    assert tiles[1].residual_energy == 16.0
    assert all(tile.support_fraction >= 0.25 for tile in tiles)


def test_residual_tile_basis_field_initializes_local_capacity_from_tiles() -> None:
    domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=900.0,
        coord_mode="local_tangent",
    )
    observation = FieldObservation(
        image=torch.zeros((4, 4), dtype=torch.float32),
        header=_tan_header(4, crval1=150.0, crval2=2.0, scale_deg=0.01),
        wavelength=5600.0,
        band_name="r",
    )
    prediction = torch.zeros((4, 4), dtype=torch.float32)
    prediction[:2, :2] = 1.0
    prediction[2:, 2:] = 3.0
    product = field_image_residual_product(
        prediction,
        observation.image,
        band_name="r",
    )
    model = make_residual_tile_basis_field(
        _ZeroField(domain),
        observation,
        residual_product=product,
        tile_hw=(2, 2),
        max_tiles=1,
        spectral_scale_encoded=10.0,
    )
    wcs = WCS(observation.header)
    center_ra, center_dec = wcs.pixel_to_world(
        torch.tensor(2.5, dtype=torch.float64),
        torch.tensor(2.5, dtype=torch.float64),
    )
    far_ra, far_dec = wcs.pixel_to_world(
        torch.tensor(0.5, dtype=torch.float64),
        torch.tensor(0.5, dtype=torch.float64),
    )

    center_value = model(
        center_ra.to(dtype=torch.float32),
        center_dec.to(dtype=torch.float32),
        5600.0,
    )
    far_value = model(
        far_ra.to(dtype=torch.float32),
        far_dec.to(dtype=torch.float32),
        5600.0,
    )

    assert model.tile_count == 1
    torch.testing.assert_close(model.tile_amplitudes[0], torch.tensor(3.0))
    assert float(center_value.item()) > 2.9
    assert float(far_value.item()) < float(center_value.item())


def test_observation_weight_sum_uses_same_finite_support_mask() -> None:
    observation = FieldObservation(
        image=torch.tensor(
            [[1.0, float("nan")], [2.0, 3.0]],
            dtype=torch.float32,
        ),
        header=_tan_header(2, crval1=150.0, crval2=2.0, scale_deg=0.01),
        weight=torch.tensor(
            [[2.0, 1.0], [float("inf"), -3.0]],
            dtype=torch.float32,
        ),
        mask=torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float32),
        wavelength=5600.0,
    )

    assert observation_weight_sum_scalar(observation) == 2.0


def test_render_field_quality_and_residual_helpers() -> None:
    domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=900.0,
        coord_mode="sphere_xyz",
    )
    model = MockFieldModel(domain)
    obs = FieldObservation(
        image=torch.ones((6, 6), dtype=torch.float32),
        header=_tan_header(6, crval1=150.0, crval2=2.0, scale_deg=0.01),
        weight=torch.ones((6, 6), dtype=torch.float32),
        mask=torch.ones((6, 6), dtype=torch.float32),
        wavelength=5600.0,
    )

    residual = render_field_residual(model, obs, num_spectral_samples=3)
    product = render_field_residual_product(model, obs, num_spectral_samples=3)
    quality = render_field_quality(model, [obs], num_spectral_samples=3)
    quality_with_products = render_field_quality(
        model,
        [obs],
        num_spectral_samples=3,
        include_scorecard=True,
        include_residual_products=True,
    )

    assert residual.shape == obs.image.shape
    assert product.residual.shape == obs.image.shape
    assert set(quality) == {"aggregate", "per_observation"}
    assert len(quality["per_observation"]) == 1
    assert quality["aggregate"]["weighted_rmse"] >= 0.0
    assert quality["aggregate"]["effective_weight_sum"] == 36.0
    assert quality["aggregate"]["residual_rmse_scale_2"] >= 0.0
    assert set(quality_with_products) == {
        "aggregate",
        "per_observation",
        "scorecard",
        "residual_products",
    }
    assert len(quality_with_products["residual_products"]) == 1
    assert quality_with_products["scorecard"]["extrema"]["lowest_support_fraction"][
        "index"
    ] == 0


def test_render_field_quality_helpers_accept_operator_bundles() -> None:
    domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=900.0,
        coord_mode="sphere_xyz",
    )
    model = _ZeroField(domain)
    obs = FieldObservation(
        image=torch.zeros((4, 4), dtype=torch.float32),
        header=_tan_header(4, crval1=150.0, crval2=2.0, scale_deg=0.01),
        weight=torch.ones((4, 4), dtype=torch.float32),
        mask=torch.ones((4, 4), dtype=torch.float32),
        wavelength=5600.0,
        background_offset=3.0,
    )
    empty_bundle = ObservationOperatorBundle()

    implicit_residual = render_field_residual(model, obs, num_spectral_samples=1)
    explicit_residual = render_field_residual(
        model,
        obs,
        num_spectral_samples=1,
        operator_bundle=empty_bundle,
    )
    product = render_field_residual_product(
        model,
        obs,
        num_spectral_samples=1,
        operator_bundle=empty_bundle,
    )
    quality = render_field_quality(
        model,
        [obs],
        num_spectral_samples=1,
        observation_operator_bundles=(empty_bundle,),
    )

    torch.testing.assert_close(implicit_residual, torch.full((4, 4), 3.0))
    torch.testing.assert_close(explicit_residual, torch.zeros((4, 4)))
    torch.testing.assert_close(product.residual, torch.zeros((4, 4)))
    assert quality["aggregate"]["weighted_rmse"] == 0.0

    with pytest.raises(ValueError, match="observation_operator_bundles"):
        render_field_quality(
            model,
            [obs],
            num_spectral_samples=1,
            observation_operator_bundles=(),
        )


def test_render_field_quality_reports_psf_alignment_from_default_bundle() -> None:
    domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=900.0,
        coord_mode="sphere_xyz",
    )
    model = _ZeroField(domain)
    obs = FieldObservation(
        image=torch.zeros((5, 5), dtype=torch.float32),
        header=_tan_header(5, crval1=150.0, crval2=2.0, scale_deg=0.01),
        weight=torch.ones((5, 5), dtype=torch.float32),
        mask=torch.ones((5, 5), dtype=torch.float32),
        wavelength=5600.0,
        psf_sigma_pix=1.0,
    )
    empty_bundle = ObservationOperatorBundle()

    default_quality = render_field_quality(model, [obs], num_spectral_samples=1)
    explicit_quality = render_field_quality(
        model,
        [obs],
        num_spectral_samples=1,
        observation_operator_bundles=(empty_bundle,),
    )

    assert "psf_residual_alignment" in default_quality["per_observation"][0]
    assert "psf_residual_alignment" not in explicit_quality["per_observation"][0]


def test_render_field_fit_quality_uses_fit_operator_bundles() -> None:
    domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=900.0,
        coord_mode="sphere_xyz",
    )
    model = _ZeroField(domain)
    obs = FieldObservation(
        image=torch.zeros((4, 4), dtype=torch.float32),
        header=_tan_header(4, crval1=150.0, crval2=2.0, scale_deg=0.01),
        weight=torch.ones((4, 4), dtype=torch.float32),
        mask=torch.ones((4, 4), dtype=torch.float32),
        wavelength=5600.0,
        background_offset=3.0,
    )
    empty_bundle = ObservationOperatorBundle()
    fit = FieldFitResult(
        model=model,
        loss_history=(1.0,),
        final_loss=1.0,
        observation_operator_bundles=(empty_bundle,),
    )

    quality = render_field_fit_quality(
        fit,
        [obs],
        num_spectral_samples=1,
        include_residual_products=True,
    )
    overridden = render_field_fit_quality(
        fit,
        [obs],
        num_spectral_samples=1,
        observation_operator_bundles=(None,),
    )

    assert quality["aggregate"]["weighted_rmse"] == 0.0
    torch.testing.assert_close(
        quality["residual_products"][0].residual,
        torch.zeros((4, 4)),
    )
    assert overridden["aggregate"]["weighted_rmse"] == 3.0
