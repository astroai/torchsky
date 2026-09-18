from __future__ import annotations
import json
import math
import pytest

pytest.importorskip("PIL")

import sys
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torchsky

pytest.importorskip("PIL")

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

import bench_map_smacs0723_field_fit_joint as joint_benchmark  # noqa: E402
from bench_map_smacs0723_field_fit import _field_target_observation  # noqa: E402
from bench_map_smacs0723_field_fit_joint import (  # noqa: E402
    _calibrated_operator_bundles,
    _cross_fitted_scene_predictions,
    _freeze_calibrated_operator_model,
    _fitted_gaussian_psf_sigma_pix,
    _format_markdown,
    _initialize_joint_scene,
    _training_scene_initialization_product,
    _structured_noise_bundles,
    _structured_noise_diagnostics,
    _structured_scene_fit_kwargs,
    _summarize_by_model,
)
from bench_map_smacs0723_field_fit_joint import _background_subtracted_source_stats  # noqa: E402
from bench_map_smacs0723_field_fit_joint import _foam_training_tile_side  # noqa: E402
from bench_map_smacs0723_field_fit_joint import _foam_target_footprint_bounds  # noqa: E402
from bench_map_smacs0723_field_fit_joint import _initialize_foam_from_band_coadds  # noqa: E402
from bench_map_smacs0723_field_fit_joint import _refine_foam_from_observations_exact  # noqa: E402
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
    make_field_model,
    make_tangent_plane_header,
    render_spectral_power_foam_observation_exact,
)
from torchsky.map import ForwardContext  # noqa: E402
from torchsky.map.operators import (  # noqa: E402
    LearnedBackgroundOp,
    LearnedPSFOp,
    ObservationOperatorBundle,
    StructuredDetectorNoise,
    StructuredNoiseOp,
    field_observation_operator_bundle,
)
from torchsky.map.operators import (  # noqa: E402
    ConditionedBackgroundField,
    SpatioSpectralTemporalEncoder,
)


def _initialization_header(*, crpix1: float = 2.0) -> dict[str, float | int | str]:
    return {
        "NAXIS": 2, "NAXIS1": 3, "NAXIS2": 3,
        "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
        "CRVAL1": 10.0, "CRVAL2": 0.0, "CRPIX1": crpix1, "CRPIX2": 2.0,
        "CDELT1": -0.001, "CDELT2": 0.001,
    }


def _initialization_observation(
    value: float,
    *,
    wavelength: float = 15_000.0,
    band: str = "F150W",
    weight: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    background_offset: float = 0.0,
    header: dict | None = None,
    value_unit: str = "MJy/sr",
    value_semantics: str = "surface_brightness",
) -> FieldObservation:
    shape = (3, 3)
    return FieldObservation(
        image=torch.full(shape, value + background_offset),
        header=dict(header or _initialization_header()),
        weight=torch.ones(shape) if weight is None else weight,
        mask=torch.ones(shape) if mask is None else mask,
        wavelength=wavelength,
        band_name=band,
        background_offset=background_offset,
        value_unit=value_unit,
        value_semantics=value_semantics,
        spectral_unit="angstrom",
    )


def _initialization_product(
    observations: list[FieldObservation], contexts: list[ForwardContext]
):
    return _training_scene_initialization_product(
        observations,
        contexts,
        [field_observation_operator_bundle(observation) for observation in observations],
        target_header=_initialization_header(),
        fit_kwargs={"num_spectral_samples": 1, "spatial_sampling": "center"},
    )


def test_training_scene_initialization_is_order_invariant() -> None:
    observations = [
        _initialization_observation(value)
        for value in (1.0e20, -1.0e20, 3.0)
    ]
    contexts = [
        ForwardContext(exposure_id=exposure_id, extra={"split": "train"})
        for exposure_id in ("a", "b", "c")
    ]

    ordered = _initialization_product(observations, contexts)
    permutation = (0, 2, 1)
    shuffled = _initialization_product(
        [observations[index] for index in permutation],
        [contexts[index] for index in permutation],
    )

    torch.testing.assert_close(ordered.data, shuffled.data, rtol=0.0, atol=0.0)
    torch.testing.assert_close(ordered.weight, shuffled.weight, rtol=0.0, atol=0.0)


def test_training_scene_initialization_subtracts_background_and_orders_bands() -> None:
    high = _initialization_observation(
        7.0, wavelength=20_000.0, band="F200W", background_offset=5.0
    )
    low_mask = torch.ones((3, 3))
    low_mask[0, 0] = 0.0
    low = _initialization_observation(
        2.0, wavelength=9_000.0, band="F090W", mask=low_mask,
        background_offset=3.0,
    )
    product = _initialization_product(
        [high, low],
        [
            ForwardContext(exposure_id="high", extra={"split": "train"}),
            ForwardContext(exposure_id="low", extra={"split": "train"}),
        ],
    )

    assert [band.label for band in product.product.bands] == ["F090W", "F200W"]
    torch.testing.assert_close(product.data[1], torch.full((3, 3), 7.0))
    assert not bool(product.mask[0, 0, 0])
    torch.testing.assert_close(product.data[0, 1:, :], torch.full((2, 3), 2.0))


def test_training_scene_initialization_rejects_held_out_or_mixed_units() -> None:
    observation = _initialization_observation(1.0)
    with pytest.raises(ValueError, match="only training"):
        _initialization_product(
            [observation],
            [ForwardContext(exposure_id="held-out", extra={"split": "validation"})],
        )

    incompatible = _initialization_observation(1.0, value_unit="Jy/sr")
    with pytest.raises(ValueError, match="same calibrated surface-brightness unit"):
        _initialization_product(
            [observation, incompatible],
            [
                ForwardContext(exposure_id="a", extra={"split": "train"}),
                ForwardContext(exposure_id="b", extra={"split": "train"}),
            ],
        )


def test_training_scene_initialization_propagates_zero_weight_support() -> None:
    unsupported = _initialization_observation(
        4.0, wavelength=9_000.0, band="F090W", weight=torch.zeros((3, 3))
    )
    supported = _initialization_observation(6.0, wavelength=15_000.0, band="F150W")
    product = _initialization_product(
        [unsupported, supported],
        [
            ForwardContext(exposure_id="zero", extra={"split": "train"}),
            ForwardContext(exposure_id="supported", extra={"split": "train"}),
        ],
    )

    assert not bool(product.mask[0].any())
    assert bool(product.mask[1].all())
    torch.testing.assert_close(product.data[0], torch.zeros((3, 3)))
    torch.testing.assert_close(product.data[1], torch.full((3, 3), 6.0))


def test_training_scene_initialization_aligns_observations_on_target_wcs() -> None:
    centered = _initialization_observation(0.0)
    centered.image[1, 1] = 4.0
    shifted = _initialization_observation(
        0.0, header=_initialization_header(crpix1=3.0)
    )
    shifted.image[1, 2] = 4.0

    product = _initialization_product(
        [shifted, centered],
        [
            ForwardContext(exposure_id="shifted", extra={"split": "train"}),
            ForwardContext(exposure_id="centered", extra={"split": "train"}),
        ],
    )

    assert float(product.data[0, 1, 1]) == pytest.approx(4.0)
    assert int(torch.argmax(product.data[0])) == 4


def test_training_scene_initialization_preserves_tabulated_response() -> None:
    observation = _initialization_observation(2.0)
    observation.transmission_wavelength = torch.tensor([14_000.0, 15_000.0, 16_000.0])
    observation.transmission_throughput = torch.tensor([0.0, 1.0, 0.0])
    observation.spectral_min = 14_000.0
    observation.spectral_max = 16_000.0

    product = _initialization_product(
        [observation],
        [ForwardContext(exposure_id="tabulated", extra={"split": "train"})],
    )

    band = product.product.bands[0]
    assert band.passband is not None
    assert band.transmission_center is None
    torch.testing.assert_close(
        band.passband.wavelength,
        torch.tensor([14_000.0, 15_000.0, 16_000.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        band.passband.throughput,
        torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
    )


def test_latent_scene_initialization_improves_prediction_through_active_psf() -> None:
    header = {
        **_initialization_header(crpix1=4.0),
        "NAXIS1": 7,
        "NAXIS2": 7,
        "CRPIX2": 4.0,
    }

    class CompactSource(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

        def forward(self, ra, dec, wavelength):
            radius2 = ((ra - 10.0) / 0.0007).square() + (dec / 0.0007).square()
            return 4.0 * torch.exp(-0.5 * radius2) + self.anchor

    template = FieldObservation(
        image=torch.zeros((7, 7)),
        header=header,
        weight=torch.ones((7, 7)),
        mask=torch.ones((7, 7)),
        wavelength=15_000.0,
        band_name="F150W",
        psf_sigma_pix=1.0,
        value_unit="MJy/sr",
        value_semantics="surface_brightness",
        spectral_unit="angstrom",
    )
    bundle = field_observation_operator_bundle(template)
    observed = joint_benchmark.render_field_observation(
        CompactSource(), template, operator_bundle=bundle,
        num_spectral_samples=1, spatial_sampling="center",
    ).detach()
    observation = replace(template, image=observed)
    context = ForwardContext(exposure_id="train", extra={"split": "train"})
    blurred = _training_scene_initialization_product(
        [observation], [context], [bundle], target_header=header,
        fit_kwargs={"num_spectral_samples": 1, "spatial_sampling": "center"},
    )

    latent = joint_benchmark._fit_training_scene_initialization_product(
        [observation], [context], [bundle], target_header=header,
        fit_kwargs={"num_spectral_samples": 1, "spatial_sampling": "center"},
        steps=80, lr=0.1,
    )

    domain = joint_benchmark.estimate_field_domain([observation], coord_mode="local_tangent")
    predictions = []
    with torch.no_grad():
        for product in (blurred, latent):
            model = make_field_model(
                "explicit_spectral_grid_field", domain,
                spatial_resolution=7, spectral_resolution=2,
            )
            model.initialize_from_product(product)
            predictions.append(joint_benchmark.render_field_observation(
                model, observation, operator_bundle=bundle,
                num_spectral_samples=1, spatial_sampling="center",
            ))
    blurred_mse = torch.mean((predictions[0] - observed).square())
    latent_mse = torch.mean((predictions[1] - observed).square())
    assert float(latent_mse) < float(blurred_mse)


def test_latent_scene_initialization_is_order_invariant() -> None:
    observations = [
        _initialization_observation(2.0),
        _initialization_observation(4.0),
    ]
    contexts = [
        ForwardContext(exposure_id="b", extra={"split": "train"}),
        ForwardContext(exposure_id="a", extra={"split": "train"}),
    ]
    bundles = [field_observation_operator_bundle(item) for item in observations]
    kwargs = {
        "target_header": _initialization_header(),
        "fit_kwargs": {"num_spectral_samples": 1, "spatial_sampling": "center"},
        "steps": 2,
        "lr": 0.05,
    }

    ordered = joint_benchmark._fit_training_scene_initialization_product(
        observations, contexts, bundles, **kwargs
    )
    reversed_product = joint_benchmark._fit_training_scene_initialization_product(
        list(reversed(observations)),
        list(reversed(contexts)),
        list(reversed(bundles)),
        **kwargs,
    )

    torch.testing.assert_close(ordered.data, reversed_product.data, rtol=0.0, atol=0.0)
    assert ordered.provenance == reversed_product.provenance


def _initialization_domain() -> FieldDomain:
    return FieldDomain(
        center_ra_deg=10.0,
        center_dec_deg=0.0,
        spatial_scale_deg=0.01,
        wavelength_center=15_000.0,
        wavelength_scale=6_000.0,
        coord_mode="local_tangent",
    )


def _constant_initialization_product(value: float = 3.0):
    observation = _initialization_observation(value)
    return _initialization_product(
        [observation],
        [ForwardContext(exposure_id="train", extra={"split": "train"})],
    )


@pytest.mark.parametrize(
    ("model_kind", "model_kwargs"),
    [
        ("sine_implicit", {"hidden_dim": 4, "hidden_layers": 1}),
        ("wire_implicit", {"hidden_dim": 4, "hidden_layers": 1}),
        (
            "sasnet",
            {
                "hidden_dim": 4,
                "num_frequencies": 4,
                "pixel_footprint_deg": 0.001,
                "hash_levels": 2,
                "hash_features_per_level": 2,
                "hash_log2_size": 4,
                "mask_hidden_dim": 4,
            },
        ),
        ("flair_bla", {"hidden_dim": 4, "hidden_layers": 1}),
    ],
)
def test_joint_scene_initialization_sets_positive_inr_constant(
    model_kind: str, model_kwargs: dict
) -> None:
    model = make_field_model(model_kind, _initialization_domain(), **model_kwargs)
    diagnostics = _initialize_joint_scene(
        model, model_kind, _constant_initialization_product()
    )
    coordinate = torch.tensor([10.0])
    prediction = model(coordinate, torch.tensor([0.0]), torch.tensor([15_000.0]))

    torch.testing.assert_close(prediction, torch.tensor([3.0]), atol=1.0e-5, rtol=0.0)
    assert diagnostics["scene_initialization_training_mean"] == pytest.approx(3.0)


def test_joint_scene_initialization_handles_foam_and_hybrid_without_double_seed() -> None:
    domain = _initialization_domain()
    foam = make_field_model(
        "spectral_power_foam",
        domain,
        sites=torch.tensor([[0.0, 0.0]]),
        radii=torch.tensor([2.0]),
        cell_coefficients=torch.zeros((1, 2)),
        log_intensity_spectra=True,
    )
    foam_diagnostics = _initialize_joint_scene(
        foam, "spectral_power_foam", _constant_initialization_product()
    )
    torch.testing.assert_close(
        foam.cell_coefficients,
        torch.tensor([[math.log(3.0), 0.0]]),
        atol=1.0e-6,
        rtol=0.0,
    )
    assert foam_diagnostics["scene_initialization_policy"] == "common_product"

    hybrid_foam = make_field_model(
        "spectral_power_foam",
        domain,
        sites=torch.tensor([[0.0, 0.0]]),
        radii=torch.tensor([2.0]),
        cell_coefficients=torch.zeros((1, 2)),
        log_intensity_spectra=True,
    )
    hybrid = make_field_model(
        "diffuse_foam_hybrid",
        domain,
        foam=hybrid_foam,
        diffuse_hidden_dim=4,
        diffuse_hidden_layers=1,
    )
    _initialize_joint_scene(
        hybrid, "diffuse_foam_hybrid", _constant_initialization_product()
    )
    torch.testing.assert_close(
        hybrid.localized_foam.cell_coefficients,
        torch.tensor([[math.log(3.0 - 1.0e-4), 0.0]]),
        atol=1.0e-6,
        rtol=0.0,
    )
    assert torch.count_nonzero(hybrid.diffuse_field.output.weight) == 0


def test_sasnet_common_initialization_preserves_upstream_gradient_flow() -> None:
    model = make_field_model(
        "sasnet",
        _initialization_domain(),
        hidden_dim=8,
        num_frequencies=4,
        pixel_footprint_deg=0.001,
        hash_levels=2,
        hash_features_per_level=2,
        hash_log2_size=4,
        mask_hidden_dim=4,
    )
    _initialize_joint_scene(model, "sasnet", _constant_initialization_product())
    prediction = model(
        torch.tensor([10.0]),
        torch.tensor([0.0]),
        torch.tensor([15_000.0]),
    )
    (prediction - 4.0).square().backward()

    readout_gradient = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.readout[0].parameters()
        if parameter.grad is not None
    )
    mask_gradient = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.mask_network.parameters()
        if parameter.grad is not None
    )
    assert readout_gradient > 0.0
    assert mask_gradient > 0.0


def test_positive_inr_initialization_handles_mean_below_floor() -> None:
    model = make_field_model(
        "sine_implicit", _initialization_domain(), hidden_dim=4, hidden_layers=1
    )
    _initialize_joint_scene(model, "sine_implicit", _constant_initialization_product(0.0))
    prediction = model(
        torch.tensor([10.001]),
        torch.tensor([0.001]),
        torch.tensor([16_000.0]),
    )
    (prediction - 1.0).square().backward()

    assert torch.isfinite(prediction).all()
    assert float(model.hidden[0].weight.grad.abs().sum()) > 0.0


@pytest.mark.parametrize(
    ("model_kind", "model_kwargs"),
    [
        (
            "spectral_basis_field",
            {
                "hidden_dim": 4,
                "hidden_layers": 1,
                "num_spatial_frequencies": 1,
                "num_spectral_frequencies": 1,
            },
        ),
        (
            "learned_spectral_basis_field",
            {
                "hidden_dim": 4,
                "hidden_layers": 1,
                "num_spatial_frequencies": 1,
                "basis_dim": 3,
                "spectral_hidden_dim": 4,
            },
        ),
        (
            "sky_encoder_resmlp",
            {
                "hidden_dim": 8,
                "num_blocks": 1,
                "num_fourier_frequencies": 1,
                "spectral_frequencies": 1,
                "multiscale_readout": True,
                "readout_hidden_dim": 4,
            },
        ),
        (
            "sky_lambda_feature_field",
            {
                "base_resolution": 2,
                "num_scales": 1,
                "features_per_plane": 2,
                "scale_hidden_dim": 4,
                "scale_num_blocks": 1,
                "decoder_hidden_dim": 4,
                "decoder_num_blocks": 1,
                "refine_hidden_dim": 4,
                "bandpass_hidden_dim": 4,
            },
        ),
        (
            "operator_constrained_feature_field",
            {
                "base_resolution": 2,
                "num_scales": 1,
                "features_per_plane": 2,
                "scale_hidden_dim": 4,
                "scale_num_blocks": 1,
                "spectral_basis_dim": 3,
                "decoder_hidden_dim": 4,
                "decoder_num_blocks": 1,
                "refine_hidden_dim": 4,
                "bandpass_hidden_dim": 4,
                "spectral_context_dim": 4,
                "residual_hidden_dim": 4,
            },
        ),
    ],
)
def test_joint_scene_initialization_supports_default_model_arms(
    model_kind: str, model_kwargs: dict
) -> None:
    model = make_field_model(model_kind, _initialization_domain(), **model_kwargs)
    _initialize_joint_scene(model, model_kind, _constant_initialization_product())
    coordinate = torch.tensor([10.0])

    prediction = model(coordinate, torch.tensor([0.0]), torch.tensor([15_000.0]))

    torch.testing.assert_close(prediction, torch.tensor([3.0]), atol=1.0e-5, rtol=0.0)


def test_joint_scene_initialization_none_policy_leaves_model_unchanged() -> None:
    model = make_field_model(
        "sine_implicit", _initialization_domain(), hidden_dim=4, hidden_layers=1
    )
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}

    diagnostics = _initialize_joint_scene(
        model, "sine_implicit", _constant_initialization_product(), policy="none"
    )

    assert diagnostics == {"scene_initialization_policy": "none"}
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state[name])


def test_common_scene_initializer_allows_wire() -> None:
    assert "wire_implicit" in joint_benchmark.SCENE_INITIALIZATION_MODEL_KINDS


def test_scene_initialization_product_is_persisted_with_hashes(tmp_path: Path) -> None:
    product = _constant_initialization_product()
    metadata = joint_benchmark._save_scene_initialization_product(
        tmp_path / "initialization",
        product,
        elapsed_ms=12.5,
        pixel_response_convention="optical_only",
        intensity_scale=3.0,
    )

    assert metadata["policy"] == "common_product"
    assert metadata["elapsed_ms"] == 12.5
    assert metadata["bands"] == [
        {"label": "F150W", "transmission_center": 15_000.0, "spectral_unit": "Angstrom"}
    ]
    assert metadata["training_exposure_ids"] == ["train"]
    assert metadata["pixel_response_convention"] == "optical_only"
    assert metadata["stored_value_space"] == "input_surface_brightness"
    assert metadata["intensity_scale"] == 3.0
    for key in ("data", "weight", "support"):
        identity = metadata["files"][key]
        path = Path(identity["path"])
        assert path.is_file()
        assert identity["size"] == path.stat().st_size
        assert identity["sha256"] == joint_benchmark.calibration_fit._sha256(path)
    reopened, header = torchsky.read(metadata["files"]["data"]["path"], return_header=True)
    torch.testing.assert_close(reopened, product.data * 3.0)
    assert header["BUNIT"] == "MJy/sr"
    assert int(header["NAXIS3"]) == 1
    persisted = json.loads((tmp_path / "initialization" / "manifest.json").read_text())
    assert persisted == metadata


def test_foam_initialization_entries_use_common_wcs_and_product_band_order() -> None:
    low = _initialization_observation(2.0, wavelength=9_000.0, band="F090W")
    high = _initialization_observation(7.0, wavelength=20_000.0, band="F200W")
    product = _initialization_product(
        [high, low],
        [
            ForwardContext(exposure_id="high", extra={"split": "train"}),
            ForwardContext(exposure_id="low", extra={"split": "train"}),
        ],
    )
    entries = [
        {
            "band": "F200W",
            "target_header": _initialization_header(crpix1=4.0),
            "target_observation": replace(
                high, psf_sigma_pix=1.0, psf_spec=joint_benchmark.PSFSpec.gaussian(1.0)
            ),
        },
        {
            "band": "F090W",
            "target_header": _initialization_header(crpix1=3.0),
            "target_observation": low,
        },
    ]

    initialized = joint_benchmark._foam_initialization_entries(product, entries)

    assert [entry["band"] for entry in initialized] == ["F090W", "F200W"]
    assert all(entry["target_header"] == product.header for entry in initialized)
    assert all(entry["target_observation"].header == product.header for entry in initialized)
    assert all(entry["target_observation"].psf_spec.mode == "none" for entry in initialized)
    np.testing.assert_array_equal(initialized[0]["coadd_pred"], product.data[0].numpy())
    np.testing.assert_array_equal(initialized[1]["coadd_pred"], product.data[1].numpy())


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


def test_joint_scene_export_integrates_a_quadratic_pixel() -> None:
    class Quadratic(joint_benchmark._ZeroScene):
        def forward(self, ra, dec, wavelength):
            return ((ra - 10.0) / 0.1).square() + self.anchor

    observation = FieldObservation(
        image=torch.zeros((1, 1)), wavelength=15000.,
        header={
            "NAXIS": 2, "NAXIS1": 1, "NAXIS2": 1,
            "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
            "CRVAL1": 10., "CRVAL2": 0., "CRPIX1": 1., "CRPIX2": 1.,
            "CDELT1": -0.1, "CDELT2": 0.1,
        },
    )
    result = joint_benchmark._render_joint_scene_product(
        Quadratic(torch.device("cpu")), observation,
        {"spatial_sampling": "gauss_legendre_2x2", "render_spatial_tile_hw": (1, 1)},
        1, None,
    )
    # Integral of x**2 on [-1/2, 1/2]; centre-only export would give zero.
    assert float(result[0, 0]) == pytest.approx(1 / 12, abs=2e-4)
    assert not result.requires_grad


def test_joint_sasnet_uses_real_training_pixel_scale() -> None:
    header = {
        "NAXIS": 2, "NAXIS1": 4, "NAXIS2": 4,
        "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
        "CRVAL1": 110.0, "CRVAL2": -73.0,
        "CRPIX1": 2.5, "CRPIX2": 2.5,
        "CDELT1": -0.031 / 3600, "CDELT2": 0.031 / 3600,
    }
    observation = FieldObservation(
        image=torch.zeros((4, 4)), header=header, wavelength=15000.0,
    )
    kwargs = joint_benchmark._joint_sasnet_kwargs([observation])
    assert kwargs["pixel_footprint_deg"] == pytest.approx(0.031 / 3600, rel=1e-5)
    assert kwargs["psf_support_deg"] is None
    with pytest.raises(ValueError, match="training observations"):
        joint_benchmark._joint_sasnet_kwargs([])


def test_joint_architecture_comparison_uses_matching_patch_objective() -> None:
    keys = (
        "training_patch_size", "shared_sky_patch_sampling",
        "num_spectral_samples", "training_spectral_sampling", "loss_kind",
        "spatial_sampling", "observations_per_step", "context_loss_weight",
        "spectral_anchor_loss_weight", "render_spatial_tile_hw",
    )
    settings = [
        _joint_real_fit_kwargs(kind, crop_size=32, steps=20, num_observations=30)
        for kind in ("sasnet", "flair_bla", "sine_implicit", "explicit_spectral_grid_field")
    ]
    assert all({key: setting[key] for key in keys} ==
               {key: settings[0][key] for key in keys} for setting in settings)
    assert settings[0]["training_patch_size"] == 32
    assert settings[0]["pixel_samples_per_observation"] is None


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
    psf._torchsky_pixel_response_included = True
    psf._torchsky_pixel_response_provenance = "registered detector SCI stamp"

    bundles = _calibrated_operator_bundles(
        [observation],
        [ForwardContext(detector="NRCB4", time_mjd=59_737.2)],
        {"NRCB4": {"psf": psf, "background": background}},
    )

    assert len(bundles) == 1
    assert isinstance(bundles[0].psf, LearnedPSFOp)
    assert isinstance(bundles[0].background, LearnedBackgroundOp)
    assert bundles[0].psf.model is psf
    assert bundles[0].psf.frozen is True
    assert bundles[0].psf.pixel_response_included is True
    assert bundles[0].psf.pixel_response_provenance == "registered detector SCI stamp"
    assert bundles[0].background.model is background
    assert bundles[0].background.coordinate_frame == "detector"
    assert joint_benchmark._spatial_sampling_for_operator_bundles(
        "gauss_legendre_2x2", bundles
    ) == "center"
    assert joint_benchmark._pixel_response_convention(bundles) == (
        "effective_includes_pixel_response"
    )
    optical = ObservationOperatorBundle(
        psf=LearnedPSFOp(
            psf,
            frozen=True,
            pixel_response_included=False,
            pixel_response_provenance="optical-only calibration",
        )
    )
    with pytest.raises(ValueError, match="different pixel-response conventions"):
        joint_benchmark._spatial_sampling_for_operator_bundles(
            "gauss_legendre_2x2", (*bundles, optical)
        )
    assert joint_benchmark._pixel_response_convention((optical,)) == "optical_only"
    explicit_observation = _initialization_observation(1.0)
    explicit_observation.psf_sigma_pix = 1.0
    explicit_observation.psf_spec = replace(
        joint_benchmark.PSFSpec.gaussian(1.0), pixel_response_included=False
    )
    explicit_optical = field_observation_operator_bundle(explicit_observation)
    assert joint_benchmark._spatial_sampling_for_operator_bundles(
        "center", (explicit_optical,)
    ) == "gauss_legendre_2x2"
    with pytest.raises(ValueError, match="different pixel-response conventions"):
        joint_benchmark._spatial_sampling_for_operator_bundles(
            "gauss_legendre_2x2", (*bundles, explicit_optical)
        )


def test_nominal_learned_psf_arm_declares_optical_only_response() -> None:
    observation = _initialization_observation(1.0, wavelength=9_000.0, band="F090W")
    observation.psf_sigma_pix = 1.25

    joint_benchmark._apply_learned_psf_arm([observation], ["F090W"], grid_side=2)

    assert observation.psf_spec.pixel_response_included is False
    assert observation.psf_spec.mode == "learned_lowrank"


def test_calibration_sample_exclusions_use_full_detector_coordinates() -> None:
    mask = np.ones((12, 12), dtype=bool)
    rows = [{"calibration_bounds_xy": [103., 104., 205., 206.]}]
    full = joint_benchmark._exclude_calibration_samples(
        mask, rows, x0=100., y0=200., psf_radius=1,
    )
    assert not full[4:8, 2:6].any()
    assert full.sum() == mask.size - 16
    crop = joint_benchmark._exclude_calibration_samples(
        mask[3:9, 1:8], rows, x0=101., y0=203., psf_radius=1,
    )
    np.testing.assert_array_equal(crop, full[3:9, 1:8])
    assert mask.all()


def test_exposure_calibration_is_shared_across_overlapping_cutouts() -> None:
    from torchsky.map.field.observation_ops import crop_field_observation

    class Plane(torch.nn.Module):
        def forward(self, x, y, wavelength, time):
            return 2.0 + 0.1 * x + 0.2 * y

    class IdentityPSF(torch.nn.Module):
        kernel_size = 1

        def forward(self, image, *, spatial_x, spatial_y, **kwargs):
            assert float(spatial_x.min()) >= 100
            assert float(spatial_y.min()) >= 200
            return image

    full = FieldObservation(
        image=torch.zeros((8, 8)), wavelength=15000.,
        detector_x0_pix=100., detector_y0_pix=200.,
        header={"NAXIS": 2, "NAXIS1": 8, "NAXIS2": 8,
                "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
                "CRVAL1": 10., "CRVAL2": 0., "CRPIX1": 4.5, "CRPIX2": 4.5,
                "CDELT1": -0.001, "CDELT2": 0.001},
    )
    crops = [crop_field_observation(full, x0=x, y0=y, width=4, height=4)
             for x, y in ((2, 1), (3, 2))]
    context = ForwardContext(detector="NRCB4", survey="F150W",
                             extra={"calibration_exposure_id": "/full/exposure-a.fits"})
    psf, background = IdentityPSF(), Plane()
    psf._torchsky_pixel_response_included = True
    psf._torchsky_pixel_response_provenance = "registered detector SCI stamp"
    models = {("NRCB4", "/full/exposure-a.fits", "F150W"):
              {"psf": psf, "background": background}}
    bundles = _calibrated_operator_bundles(crops, [context, context], models)
    assert all(bundle.psf.model is psf and bundle.background.model is background
               for bundle in bundles)
    images = [joint_benchmark.render_field_observation(
        joint_benchmark._ZeroScene(torch.device("cpu")), crop,
        num_spectral_samples=1, operator_bundle=bundle, forward_context=context,
    ) for crop, bundle in zip(crops, bundles, strict=True)]
    torch.testing.assert_close(images[0][1:, 1:], images[1][:-1, :-1])
    assert float(images[0][0, 0]) == pytest.approx(2 + .1 * 102 + .2 * 201)
    other = ForwardContext(detector="NRCB4", survey="F150W",
                           extra={"calibration_exposure_id": "/full/exposure-b.fits"})
    with pytest.raises(ValueError, match="no calibrated operators"):
        _calibrated_operator_bundles([crops[0]], [other], models)
    other = ForwardContext(detector="NRCB4", survey="F090W",
                           extra={"calibration_exposure_id": "/full/exposure-a.fits"})
    with pytest.raises(ValueError, match="no calibrated operators"):
        _calibrated_operator_bundles([crops[0]], [other], models)


def test_structured_scene_kwargs_disable_invalid_sampling_approximations() -> None:
    kwargs = _structured_scene_fit_kwargs(
        {
            "training_patch_size": 8,
            "pixel_samples_per_observation": 32,
            "shared_sky_patch_sampling": True,
            "diagnostic_spatial_tile_hw": (4, 4),
            "spectral_anchor_observations_per_step": 1,
            "context_observations_per_step": 1,
            "training_spectral_sampling": "mixture_mc",
        },
        num_observations=3,
    )
    assert kwargs["loss_kind"] == "structured_nll"
    assert kwargs["observations_per_step"] == 3
    assert kwargs["training_spectral_sampling"] == "deterministic_quadrature"
    assert kwargs["training_patch_size"] is None
    assert kwargs["pixel_samples_per_observation"] is None
    assert not kwargs["shared_sky_patch_sampling"]
    assert kwargs["diagnostic_spatial_tile_hw"] is None
    assert kwargs["spectral_anchor_observations_per_step"] == 0
    assert kwargs["context_observations_per_step"] == 0


def test_cross_fitted_predictions_exclude_exactly_the_held_out_exposure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observations = tuple(
        FieldObservation(
            image=torch.full((2, 3), float(index)),
            header={"NAXIS1": 3, "NAXIS2": 2},
            wavelength=15_000.0,
        )
        for index in range(3)
    )
    training_calls: list[tuple[int, ...]] = []

    def fake_fit(training, model, *, steps, **kwargs):
        del steps, kwargs
        training_calls.append(tuple(id(observation) for observation in training))
        return SimpleNamespace(model=model, initial_loss=2.0, final_loss=1.0)

    def fake_render(model, observation, **kwargs):
        del model, kwargs
        return torch.zeros_like(observation.image)

    monkeypatch.setattr(joint_benchmark, "fit_field_model", fake_fit)
    monkeypatch.setattr(
        joint_benchmark,
        "render_field_observation",
        fake_render,
    )
    predictions, diagnostics = _cross_fitted_scene_predictions(
        observations,
        model_factory=lambda: torch.nn.Linear(1, 1),
        steps=1,
        fit_kwargs={},
        render_kwargs={},
        seed=7,
    )

    assert len(predictions) == len(observations)
    for held_out_index, training_ids in enumerate(training_calls):
        assert id(observations[held_out_index]) not in training_ids
        assert set(training_ids) == {
            id(observation)
            for index, observation in enumerate(observations)
            if index != held_out_index
        }
        assert diagnostics[held_out_index]["held_out_index"] == held_out_index


def test_structured_noise_bundles_are_frozen_and_exposure_aligned() -> None:
    observations = tuple(
        FieldObservation(
            image=torch.zeros((2, 3)),
            header={"NAXIS1": 3, "NAXIS2": 2},
            err=torch.ones((2, 3)),
            wavelength=15_000.0,
        )
        for _ in range(2)
    )
    models = (StructuredDetectorNoise(), StructuredDetectorNoise())
    bundles = _structured_noise_bundles(observations, models)
    assert len(bundles) == len(observations)
    assert all(isinstance(bundle.noise, StructuredNoiseOp) for bundle in bundles)
    assert all(bundle.noise.frozen for bundle in bundles)
    assert tuple(bundle.noise.model for bundle in bundles) == models


def test_structured_noise_diagnostics_report_calibrated_coverage() -> None:
    observation = FieldObservation(
        image=torch.tensor(
            [[0.0, float("nan"), 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
        ),
        header={"NAXIS1": 4, "NAXIS2": 2},
        var_poisson=torch.tensor(
            [[float("nan"), 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
        ),
        wavelength=15_000.0,
    )
    rows = _structured_noise_diagnostics(
        (observation,),
        (torch.zeros_like(observation.image),),
        (StructuredDetectorNoise(),),
        (0.0,),
    )
    assert rows[0]["valid_pixels"] == 6
    assert rows[0]["active_amplifiers"] == [0, 1, 2, 3]
    assert rows[0]["posterior_weighted_coverage_68"] == pytest.approx(1.0)
    assert rows[0]["posterior_weighted_coverage_95"] == pytest.approx(1.0)


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
    assert "Secondary reference-image comparisons" in text
    assert "F090W" in text
    assert "spectral_basis_field" in text


def test_joint_field_fit_markdown_summarizes_exact_refinement() -> None:
    row = {
        "band": "F090W",
        "model": "spectral_power_foam",
        "status": "OK",
        "n_exposures": 8,
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
        "foam_exact_refinement_initial_rmse": 0.2,
        "foam_exact_refinement_rmse": 0.1,
        "foam_exact_refinement_iterations": 12.0,
        "foam_exact_refinement_relative_residual": 1.0e-6,
        "foam_exact_refinement_preconditioner_setup_seconds": 0.1,
        "foam_exact_refinement_rhs_seconds": 0.2,
        "foam_exact_refinement_pcg_seconds": 0.3,
        "foam_exact_refinement_peak_operator_batch_bytes": float(2**20),
        "foam_exact_refinement_ridge_centered_on_initial": 1.0,
    }

    text = _format_markdown([row])

    assert "Exact Refinement" in text
    assert "| spectral_power_foam | 0.200000 | 0.100000 | 12 |" in text
    assert "| initial |" in text


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


def test_joint_real_fit_kwargs_preserve_psf_with_streamed_tiles() -> None:
    kwargs = _joint_real_fit_kwargs(
        "spectral_basis_field",
        crop_size=32,
        steps=4,
        num_observations=20,
    )
    assert kwargs["observations_per_step"] == 12
    assert kwargs["training_patch_size"] is None
    assert kwargs["pixel_samples_per_observation"] is None
    assert kwargs["streaming_accumulation"] is True
    assert kwargs["render_spatial_tile_hw"] == (256, 256)


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
    assert kwargs["diagnostic_spatial_tile_hw"] == (32, 32)
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
    assert kwargs["diagnostic_spatial_tile_hw"] == (4, 4)


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
        whiten_spectral_basis=True,
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
    assert diagnostics["foam_spectral_basis_rank"] == 2.0
    assert diagnostics["foam_spectral_basis_condition_number"] == pytest.approx(1.0)


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
    assert diagnostics["foam_spectral_basis_condition_number"] == 1.0
    assert diagnostics["foam_initial_cell_variation_abs_p95"] > 0.0
    assert (
        diagnostics["foam_initial_cell_variation_abs_max"]
        >= diagnostics["foam_initial_cell_variation_abs_p95"]
    )


def test_joint_foam_exact_observation_refinement_recovers_native_pixels() -> None:
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
    truth = SpectralPowerFoamField(
        domain,
        sites=torch.tensor([[-0.25, 0.0], [0.25, 0.0]]),
        radii=torch.full((2,), 1.0),
        cell_coefficients=torch.tensor([[1.0, 0.2], [1.5, -0.1]]),
        cell_gradient_coefficients=torch.tensor(
            [[[0.1, -0.05], [0.02, 0.01]], [[-0.08, 0.04], [0.01, -0.02]]]
        ),
        compact_fixed_geometry=True,
    )
    truth.sites.requires_grad_(False)
    truth.foam.raw_radii.requires_grad_(False)
    observations = []
    for wavelength in (1.0, 3.0):
        template = FieldObservation(
            image=torch.zeros((7, 7)),
            header=dict(header),
            weight=torch.ones((7, 7)),
            mask=torch.ones((7, 7), dtype=torch.bool),
            wavelength=wavelength,
            band_name=f"B{wavelength:g}",
        )
        image = render_spectral_power_foam_observation_exact(
            truth,
            template,
            num_spectral_samples=1,
        )
        observations.append(
            FieldObservation(
                image=image.detach(),
                header=dict(header),
                weight=torch.ones((7, 7)),
                mask=torch.ones((7, 7), dtype=torch.bool),
                wavelength=wavelength,
                band_name=f"B{wavelength:g}",
            )
        )
    model = SpectralPowerFoamField(
        domain,
        sites=truth.sites.detach(),
        radii=truth.radii.detach(),
        cell_coefficients=torch.zeros((2, 2)),
        cell_gradient_coefficients=torch.zeros((2, 2, 2)),
        compact_fixed_geometry=True,
    )
    model.sites.requires_grad_(False)
    model.foam.raw_radii.requires_grad_(False)

    diagnostics = _refine_foam_from_observations_exact(
        model,
        observations,
        num_spectral_samples=1,
        max_iterations=100,
        operator_storage_device="cpu",
        observation_batch_size=1,
    )

    assert diagnostics["foam_exact_refinement_rmse"] < 1.0e-4
    assert diagnostics["foam_exact_refinement_observations"] == 2.0
    assert diagnostics["foam_exact_refinement_observation_batch_size"] == 1.0
    assert diagnostics["foam_exact_refinement_observation_batch_auto"] == 0.0
    assert diagnostics["foam_exact_refinement_operator_storage_bytes"] > 0.0
    assert diagnostics["foam_exact_refinement_peak_operator_batch_bytes"] < diagnostics[
        "foam_exact_refinement_operator_storage_bytes"
    ]
    assert diagnostics["foam_exact_refinement_host_offload"] == 1.0
    assert diagnostics["foam_exact_refinement_converged"] == 1.0
    assert diagnostics["foam_exact_refinement_initial_rmse"] >= diagnostics[
        "foam_exact_refinement_rmse"
    ]
    assert diagnostics["foam_exact_refinement_normal_operator_applications"] == (
        diagnostics["foam_exact_refinement_iterations"] + 1.0
    )
    assert diagnostics["foam_exact_refinement_relative_residual"] <= 1.0e-5
    for observation in observations:
        actual = render_spectral_power_foam_observation_exact(
            model,
            observation,
            num_spectral_samples=1,
        )
        torch.testing.assert_close(
            actual,
            observation.image,
            atol=2.0e-4,
            rtol=2.0e-4,
        )


def test_joint_foam_coadd_initializer_handles_empty_support() -> None:
    header = make_tangent_plane_header(
        3,
        3,
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
    observation = FieldObservation(
        image=torch.zeros((3, 3)),
        header=dict(header),
        weight=torch.ones((3, 3)),
        mask=torch.ones((3, 3), dtype=torch.bool),
        wavelength=2.0,
        band_name="B2",
    )

    diagnostics = _initialize_foam_from_band_coadds(
        model,
        [
            {
                "coadd_pred": np.ones((3, 3), dtype=np.float32),
                "support_mask": np.zeros((3, 3), dtype=bool),
                "target_header": dict(header),
                "target_observation": observation,
            }
        ],
        intensity_scale=1.0,
        num_spectral_samples=1,
    )

    assert diagnostics["foam_initialized_cell_fraction"] == 0.0
    assert diagnostics["foam_initial_cell_variation_abs_p95"] == 0.0
    assert diagnostics["foam_initial_cell_variation_abs_max"] == 0.0


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


def test_regular_foam_initialization_can_fill_rectangular_target_bounds() -> None:
    kwargs = _regular_foam_model_kwargs(
        grid_side=2,
        basis_count=1,
        bounds_xy=((-0.25, 0.25), (-0.5, 0.5)),
    )

    torch.testing.assert_close(
        kwargs["sites"],
        torch.tensor(
            [
                [-0.125, -0.25],
                [0.125, -0.25],
                [-0.125, 0.25],
                [0.125, 0.25],
            ]
        ),
    )
    assert kwargs["minimum_radius"] == pytest.approx(0.0625)


def test_foam_target_footprint_bounds_use_only_wcs_geometry() -> None:
    header = make_tangent_plane_header(
        3,
        5,
        crval_ra_deg=10.0,
        crval_dec_deg=20.0,
        cd11=-0.01,
        cd22=0.01,
    )
    domain = FieldDomain(
        center_ra_deg=10.0,
        center_dec_deg=20.0,
        spatial_scale_deg=0.1,
        wavelength_center=2.0,
        wavelength_scale=1.0,
    )

    bounds = _foam_target_footprint_bounds(
        domain,
        [{"target_header": header, "coadd_pred": np.full((3, 5), np.nan)}],
    )

    assert bounds[0][0] < -0.2 < 0.2 < bounds[0][1]
    assert bounds[1][0] < -0.1 < 0.1 < bounds[1][1]


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
        out = Path(tmp) / "fitted_gaussian" / "f090w" / "spectral_basis_field"
        assert (out / "reference.fits").exists()
        assert (out / "prediction.fits").exists()
        assert (out / "residual.fits").exists()
        assert (out / "support_mask.fits").exists()
        assert (out / "source_mask.fits").exists()
        assert (out / "reference.png").exists()
        assert (out / "prediction.png").exists()
        assert (out / "residual.png").exists()
