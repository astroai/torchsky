from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest
from scipy.linalg import solve as scipy_solve
import torch

from torchsky.map.field.gaussian import bind_linear_band_grid_likelihood
from torchsky.map.field.inverse import bind_linear_band_grid_observation_operator
from torchsky.map.field.types import FieldDomain, FieldObservation, PSFSpec
from torchsky.map.likelihood import GaussianLikelihoodSystem
from torchsky.map.operators.background import (
    AdditiveComponent,
    AdditiveFrame,
    AdditiveProcessingState,
    AdditiveStage,
)
from torchsky.map.query import fit_linear_query


class _Coordinates:
    def pixel_to_world(self, x, y):
        return 10.0 + x * 0.01, 20.0 + y * 0.01


def _observations(*, mask=False, psf=False):
    dtype = torch.float64
    domain = FieldDomain(
        center_ra_deg=10.0,
        center_dec_deg=20.0,
        spatial_scale_deg=0.1,
        wavelength_center=5000.0,
        wavelength_scale=500.0,
    )
    bounds = (0.0, 0.02 * math.cos(math.radians(20)), 0.0, 0.02)
    observations = []
    kernel = torch.tensor(
        [[0.0, 0.05, 0.1], [0.15, 0.3, 0.2], [0.0, 0.1, 0.1]], dtype=dtype
    )
    for index in range(2):
        keep = torch.ones(3, 3, dtype=torch.bool)
        if mask:
            keep[index, 1] = False
        weight = torch.full((3, 3), float(index + 1), dtype=dtype)
        weight[~keep] = 0
        observations.append(
            FieldObservation(
                image=torch.arange(9, dtype=dtype).reshape(3, 3) / 10 + index,
                header={"NAXIS1": 3, "NAXIS2": 3},
                weight=weight,
                mask=keep,
                detector_to_sky=_Coordinates(),
                band_name="common-band",
                wavelength=5000.0,
                value_semantics="surface_brightness",
                value_unit="MJy/sr",
                detector_law="gaussian",
                psf_spec=PSFSpec(
                    mode="kernel2d",
                    kernel2d=kernel.clone(),
                    pixel_response_included=True,
                )
                if psf
                else PSFSpec(),
            )
        )
    return tuple(observations), domain, bounds


def _bind(observations, domain, bounds, **kwargs):
    return bind_linear_band_grid_likelihood(
        observations,
        domain,
        spatial_resolution=3,
        spatial_bounds_local_deg=bounds,
        measurement_ids=tuple((f"native-{i}",) for i in range(len(observations))),
        calibration_id="joint-cal-v1",
        provenance_id="native-v1",
        revision=lambda: 0,
        **kwargs,
    )


def _dense_response(observations, domain, bounds):
    operators = [
        bind_linear_band_grid_observation_operator(
            replace(o, additive_components=()),
            domain,
            spatial_resolution=3,
            spatial_bounds_local_deg=bounds,
        )
        for o in observations
    ]
    columns = []
    for basis in torch.eye(9, dtype=torch.float64):
        columns.append(
            torch.cat([op.apply(basis.reshape(3, 3)).flatten() for op in operators])
        )
    return torch.stack(columns, -1)


@pytest.mark.parametrize("mask", [False, True])
def test_calibrated_shared_background_matches_full_joint_density_and_gls(mask):
    observations, domain, bounds = _observations(mask=mask, psf=True)
    bases = tuple(torch.ones(3, 3, 1, dtype=torch.float64) for _ in observations)
    factor = _bind(
        observations,
        domain,
        bounds,
        nuisance_bases=bases,
        nuisance_ids=("shared-background",),
        nuisance_covariance_factor=torch.tensor([[0.4]], dtype=torch.float64),
        nuisance_mean=torch.tensor([0.7], dtype=torch.float64),
    )
    system = GaussianLikelihoodSystem((factor,))
    matrix = _dense_response(observations, domain, bounds)
    keep = torch.cat([o.mask.flatten() for o in observations])
    j = matrix[keep].numpy()
    y = torch.cat([o.image.flatten() for o in observations])[keep].numpy() - 0.7
    weights = torch.cat([o.weight.flatten() for o in observations])[keep].numpy()
    covariance = np.diag(1 / weights) + np.ones((len(y), len(y))) * 0.4**2
    h = j.T @ scipy_solve(covariance, j, assume_a="pos")
    b = j.T @ scipy_solve(covariance, y, assume_a="pos")
    np.testing.assert_allclose(system.rhs().numpy().ravel(), b, atol=1e-11)
    state = torch.arange(9, dtype=torch.float64).reshape(3, 3) / 8
    residual = y - j @ state.numpy().ravel()
    expected_nll = 0.5 * (
        residual @ scipy_solve(covariance, residual, assume_a="pos")
        + len(y) * np.log(2 * np.pi)
        + np.linalg.slogdet(covariance)[1]
    )
    np.testing.assert_allclose(
        system.negative_log_likelihood(state).numpy(), expected_nll, atol=1e-11
    )
    np.testing.assert_allclose(
        system.information(state).numpy().ravel(), h @ state.numpy().ravel(), atol=1e-11
    )


def test_shared_calibration_error_does_not_average_down_independently():
    observations, domain, bounds = _observations()
    bases = tuple(torch.ones(3, 3, 1, dtype=torch.float64) for _ in observations)
    factor = _bind(
        observations,
        domain,
        bounds,
        nuisance_bases=bases,
        nuisance_ids=("common-zero",),
        nuisance_covariance_factor=torch.tensor([[0.4]], dtype=torch.float64),
    )
    result = fit_linear_query(
        GaussianLikelihoodSystem((factor,)), (torch.ones(1, 18, dtype=torch.float64),)
    )
    torch.testing.assert_close(
        result.sampling_covariance,
        torch.tensor([[1 / 27 + 0.4**2]], dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="shared eliminated"):
        second = replace(factor, measurement_ids=("other-native",))
        GaussianLikelihoodSystem((factor, second))


def test_profiled_shared_background_preserves_the_global_zero_level_gauge():
    observations, domain, bounds = _observations()
    factor = _bind(
        observations,
        domain,
        bounds,
        nuisance_bases=tuple(
            torch.ones(3, 3, 1, dtype=torch.float64) for _ in observations
        ),
        nuisance_ids=("unknown-zero",),
    )
    system = GaussianLikelihoodSystem((factor,))
    torch.testing.assert_close(
        system.information(torch.ones(3, 3, dtype=torch.float64)),
        torch.zeros(3, 3, dtype=torch.float64),
        atol=1e-12,
        rtol=0,
    )
    with pytest.raises(ValueError, match="density constants"):
        system.negative_log_likelihood(torch.ones(3, 3, dtype=torch.float64))
    with pytest.raises(
        RuntimeError, match="unidentifiable|converge|residual|non-finite"
    ):
        system.functional_uncertainty(
            torch.ones(3, 3, dtype=torch.float64), iterations=10
        )
    query = torch.zeros(3, 3, dtype=torch.float64)
    query[0, 0], query[0, 1] = 1, -1
    report = system.functional_uncertainty(query, iterations=20, rtol=1e-8)
    assert report.sampling_variance > 0
    torch.testing.assert_close(report.mean_response, query, atol=1e-8, rtol=1e-8)


def test_typed_and_legacy_additive_means_are_removed_at_declared_stages():
    observations, domain, bounds = _observations(psf=True)
    source = torch.arange(9, dtype=torch.float64).reshape(3, 3) / 5
    output = torch.full((3, 3), 0.4, dtype=torch.float64)
    components = (
        AdditiveComponent(
            source,
            AdditiveStage.SAMPLED_RESPONSE_INPUT,
            AdditiveFrame.SAMPLED_SKY,
            "MJy/sr",
            "surface_brightness",
            AdditiveProcessingState.CALIBRATED_IMAGE,
        ),
        AdditiveComponent(
            output,
            AdditiveStage.CALIBRATED_OUTPUT,
            AdditiveFrame.DETECTOR,
            "MJy/sr",
            "surface_brightness",
            AdditiveProcessingState.CALIBRATED_IMAGE,
        ),
    )
    first = observations[0]
    kernel = first.psf_spec.kernel2d
    scattered = torch.zeros_like(source)
    for sy in range(3):
        for sx in range(3):
            for ky in range(3):
                for kx in range(3):
                    dy, dx = sy + ky - 1, sx + kx - 1
                    if 0 <= dy < 3 and 0 <= dx < 3:
                        scattered[dy, dx] += (
                            source[sy, sx] * kernel[ky, kx] / kernel.sum()
                        )
    changed = replace(
        first,
        image=first.image + scattered + output + 0.2,
        detector_background=torch.full_like(source, 0.2),
        additive_components=components,
        processing_state=AdditiveProcessingState.CALIBRATED_IMAGE,
    )
    factor = _bind((changed,), domain, bounds)
    torch.testing.assert_close(factor.data.reshape(3, 3), first.image)
    assert changed.additive_components == components


@pytest.mark.parametrize(
    "mutation",
    [
        "data",
        "weight",
        "mask",
        "header",
        "kernel",
        "background",
        "replacement",
        "calibration",
    ],
)
def test_binding_rejects_mutated_inputs_and_cached_normal_reuse(mutation):
    observations, domain, bounds = _observations(psf=True)
    observations[0].detector_background = torch.zeros(3, 3, dtype=torch.float64)
    calibration = torch.tensor([[0.4]], dtype=torch.float64)
    factor = _bind(
        observations,
        domain,
        bounds,
        nuisance_bases=tuple(
            torch.ones(3, 3, 1, dtype=torch.float64) for _ in observations
        ),
        nuisance_ids=("shared-zero",),
        nuisance_covariance_factor=calibration,
    )
    equations = GaussianLikelihoodSystem((factor,)).normal_equations()
    o = observations[0]
    if mutation == "data":
        o.image.add_(1)
    elif mutation == "weight":
        o.weight.mul_(2)
    elif mutation == "mask":
        o.mask[0, 0] = False
    elif mutation == "header":
        o.header["CRPIX1"] = 2.0
    elif mutation == "kernel":
        o.psf_spec.kernel2d[0, 0] += 0.1
    elif mutation == "background":
        o.detector_background.add_(1)
    elif mutation == "replacement":
        o.image = o.image.clone()
    else:
        calibration.mul_(2)
    with pytest.raises(RuntimeError, match="stale evidence"):
        _ = equations.rhs
    with pytest.raises(RuntimeError, match="stale evidence"):
        factor.operator.apply(torch.ones(3, 3, dtype=torch.float64))


def test_unknown_law_or_different_band_and_duplicate_variants_fail_closed():
    observations, domain, bounds = _observations()
    for bad in [
        replace(observations[1], band_name="other-band"),
        replace(observations[1], wavelength=6000),
        replace(observations[1], value_unit="ADU"),
        replace(observations[1], detector_law="poisson_gaussian"),
    ]:
        with pytest.raises(ValueError, match="band|units|Gaussian"):
            _bind((observations[0], bad), domain, bounds)
    with pytest.raises(ValueError, match="duplicate"):
        bind_linear_band_grid_likelihood(
            observations,
            domain,
            spatial_resolution=3,
            spatial_bounds_local_deg=bounds,
            measurement_ids=(("native",), ("native",)),
            calibration_id="v1",
            provenance_id="p1",
            revision=lambda: 0,
        )
    with pytest.raises(ValueError, match="revision"):
        bind_linear_band_grid_likelihood(
            observations,
            domain,
            spatial_resolution=3,
            spatial_bounds_local_deg=bounds,
            measurement_ids=(("a",), ("b",)),
            calibration_id="v1",
            provenance_id="p1",
        )


def test_conditional_density_counts_only_observed_support():
    observations, domain, bounds = _observations(mask=True)
    factor = _bind(observations, domain, bounds)
    expected = 0.5 * (16 * math.log(2 * math.pi) - 8 * math.log(2))
    torch.testing.assert_close(
        factor.log_normalization, factor.data.new_tensor(expected)
    )
    assert factor.nuisance_treatment == "conditional"


@pytest.mark.parametrize("tile", [None, (2, 1)])
@pytest.mark.parametrize("zero_state", [False, True])
def test_normal_quadratic_preserves_state_and_data_gradients(tile, zero_state):
    observations, domain, bounds = _observations(psf=True)
    observations[0].image.requires_grad_()
    factor = _bind(observations, domain, bounds, spatial_tile_hw=tile)
    system = GaussianLikelihoodSystem((factor,))
    state = torch.full(
        (3, 3), 0.0 if zero_state else 0.3, dtype=torch.float64, requires_grad=True
    )
    direct = system.quadratic(state)
    compressed = system.normal_equations().quadratic(state)
    direct_grad = torch.autograd.grad(
        direct, (state, observations[0].image), retain_graph=True
    )
    compressed_grad = torch.autograd.grad(compressed, (state, observations[0].image))
    torch.testing.assert_close(compressed, direct, atol=1e-11, rtol=1e-11)
    for actual, expected in zip(compressed_grad, direct_grad, strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)


def test_unversioned_inference_data_needs_a_caller_revision():
    observations, domain, bounds = _observations()
    for observation in observations:
        observation.detector_to_sky = None
        observation.header.update(
            {
                "CTYPE1": "RA---TAN",
                "CTYPE2": "DEC--TAN",
                "CRVAL1": 10.0,
                "CRVAL2": 20.0,
                "CRPIX1": 1.0,
                "CRPIX2": 1.0,
                "CDELT1": 0.01,
                "CDELT2": 0.01,
            }
        )
    with torch.inference_mode(), pytest.raises(ValueError, match="caller revision"):
        bind_linear_band_grid_likelihood(
            observations,
            domain,
            spatial_resolution=3,
            spatial_bounds_local_deg=bounds,
            measurement_ids=(("a",), ("b",)),
            calibration_id="v1",
            provenance_id="p1",
        )


@pytest.mark.parametrize(
    "device,dtype",
    [
        ("cpu", torch.float32),
        ("cpu", torch.float64),
        ("cuda", torch.float64),
        ("mps", torch.float32),
    ],
)
def test_joint_query_dtype_device_and_normal_quadratic_contract(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    observations, domain, bounds = _observations(mask=True, psf=True)
    observations = tuple(o.to(device=device, dtype=dtype) for o in observations)
    basis = tuple(torch.ones(3, 3, 1, dtype=dtype, device=device) for _ in observations)
    calibration = torch.tensor([[0.4]], dtype=dtype, device=device)
    factor = _bind(
        observations,
        domain,
        bounds,
        nuisance_bases=basis,
        nuisance_ids=("shared-calibration",),
        nuisance_covariance_factor=calibration,
        spatial_tile_hw=(2, 1),
    )
    system = GaussianLikelihoodSystem((factor,))
    state = torch.full((3, 3), 0.3, dtype=dtype, device=device)
    equations = system.normal_equations()
    torch.testing.assert_close(equations.quadratic(state), system.quadratic(state))
    query = fit_linear_query(system, (torch.ones(1, 18, dtype=dtype, device=device),))
    expected = torch.tensor([[1 / 24 + 0.4**2]], dtype=dtype, device=device)
    torch.testing.assert_close(query.sampling_covariance, expected)
    assert query.coefficients.dtype == dtype
    assert query.coefficients.device.type == device
    calibration.add_(0.01)
    with pytest.raises(RuntimeError, match="stale evidence"):
        _ = query.sampling_covariance


@pytest.mark.parametrize("identifiers", [(" ",), ("same", "same"), ["list-id"]])
def test_invalid_joint_nuisance_identifiers_fail_before_elimination(identifiers):
    observations, domain, bounds = _observations()
    with pytest.raises(ValueError, match="identifiers"):
        _bind(
            observations,
            domain,
            bounds,
            nuisance_bases=tuple(
                torch.ones(3, 3, 1, dtype=torch.float64) for _ in observations
            ),
            nuisance_ids=identifiers,
        )
