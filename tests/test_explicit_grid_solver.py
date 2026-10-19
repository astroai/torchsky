from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from torchfits import write
from torchsky.map.field.inverse import (
    IRLSConfig,
    LinearBandGridField,
    adapt_stationary_calibrated_band_observation,
    bind_linear_band_grid_observation_operator,
    bind_linear_band_grid_observations_to_operator_set,
    linear_band_grid_diagonal_majorizer,
    load_stationary_calibrated_band_observations,
    prepare_linear_band_grid_observation_data,
    solve_joint_multiband_grid_cg,
    solve_linear_band_grid_cg,
    solve_linear_band_grid_lsqr,
)
from torchsky.map.field.rendering import render_field_observation
from torchsky.map.field.types import (
    FieldDomain,
    FieldObservation,
    ForwardContext,
    PSFSpec,
)
from torchsky.map.product import ProductResult
from torchsky.map.operators.background import (
    ConditionedBackgroundField,
    LearnedBackgroundOp,
)
from torchsky.map.operators.bundle import field_observation_operator_bundle
from torchsky.map.operators.learned_kernel import (
    ConditionedKernelPSF,
    SpatioSpectralTemporalEncoder,
)
from torchsky.map.operators.psf import ExplicitPSFOp, LearnedPSFOp
from torchsky.map.operators.radiometry import ExplicitRadiometryOp


class _DetectorToSky:
    def __init__(
        self,
        *,
        center_ra_deg: float,
        center_dec_deg: float,
        bounds: tuple[float, float, float, float],
        shape: tuple[int, int],
    ) -> None:
        self.center_ra_deg = center_ra_deg
        self.center_dec_deg = center_dec_deg
        self.bounds = bounds
        self.shape = shape

    def pixel_to_world(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_min, x_max, y_min, y_max = self.bounds
        height, width = self.shape
        x_local = x_min + x * (x_max - x_min) / max(width - 1, 1)
        y_local = y_min + y * (y_max - y_min) / max(height - 1, 1)
        ra = self.center_ra_deg + x_local / math.cos(
            math.radians(self.center_dec_deg)
        )
        dec = self.center_dec_deg + y_local
        return ra, dec


def _observation(
    *,
    weight: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    psf_spec: PSFSpec | None = None,
) -> tuple[
    FieldObservation,
    FieldDomain,
    tuple[float, float, float, float],
]:
    shape = (4, 4)
    if weight is None:
        weight = torch.ones(shape, dtype=torch.float64)
    bounds = (-0.03, 0.03, -0.02, 0.02)
    domain = FieldDomain(
        center_ra_deg=10.0,
        center_dec_deg=20.0,
        spatial_scale_deg=0.1,
        wavelength_center=20_000.0,
        wavelength_scale=1_000.0,
    )
    observation = FieldObservation(
        image=torch.zeros(shape, dtype=torch.float64),
        header={"NAXIS1": shape[1], "NAXIS2": shape[0]},
        weight=weight,
        mask=mask,
        psf_spec=psf_spec,
        detector_to_sky=_DetectorToSky(
            center_ra_deg=domain.center_ra_deg,
            center_dec_deg=domain.center_dec_deg,
            bounds=bounds,
            shape=shape,
        ),
        band_name="F200W",
        value_semantics="surface_brightness",
    )
    return observation, domain, bounds


def test_linear_band_grid_operator_is_linear() -> None:
    observation, domain, bounds = _observation()
    operator = bind_linear_band_grid_observation_operator(
        observation,
        domain,
        spatial_resolution=4,
        spatial_bounds_local_deg=bounds,
    )
    first = torch.arange(16, dtype=torch.float64).reshape(4, 4)
    second = torch.flip(first, dims=(0, 1))

    combined = operator.apply(1.75 * first - 0.25 * second)
    expected = 1.75 * operator.apply(first) - 0.25 * operator.apply(second)

    torch.testing.assert_close(combined, expected, rtol=1.0e-12, atol=1.0e-12)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_linear_band_grid_operator_satisfies_adjoint_identity(
    dtype: torch.dtype,
) -> None:
    observation, domain, bounds = _observation()
    operator = bind_linear_band_grid_observation_operator(
        observation,
        domain,
        spatial_resolution=4,
        spatial_bounds_local_deg=bounds,
    )
    state = torch.arange(16, dtype=dtype).reshape(4, 4) / 7.0
    dual = torch.linspace(-1.0, 2.0, 16, dtype=dtype).reshape(4, 4)

    lhs = torch.sum(operator.apply(state) * dual)
    rhs = torch.sum(state * operator.adjoint(dual))

    tolerance = 1.0e-5 if dtype == torch.float32 else 1.0e-12
    torch.testing.assert_close(lhs, rhs, rtol=tolerance, atol=tolerance)


def test_linear_band_grid_operator_psf_satisfies_adjoint_identity() -> None:
    kernel = torch.tensor(
        [[0.00, 0.05, 0.10], [0.15, 0.30, 0.20], [0.00, 0.10, 0.10]],
        dtype=torch.float64,
    )
    observation, domain, bounds = _observation(
        psf_spec=PSFSpec(
            mode="kernel2d",
            kernel2d=kernel,
            pixel_response_included=True,
        )
    )
    operator = bind_linear_band_grid_observation_operator(
        observation,
        domain,
        spatial_resolution=5,
        spatial_bounds_local_deg=bounds,
    )
    state = torch.arange(25, dtype=torch.float64).reshape(5, 5) / 11.0
    dual = torch.linspace(-0.7, 1.3, 16, dtype=torch.float64).reshape(4, 4)

    lhs = torch.sum(operator.apply(state) * dual)
    rhs = torch.sum(state * operator.adjoint(dual))

    torch.testing.assert_close(lhs, rhs, rtol=1.0e-11, atol=1.0e-11)


def test_linear_band_grid_operator_excludes_zero_weight_pixels() -> None:
    weight = torch.ones((4, 4), dtype=torch.float64)
    weight[1, 2] = 0.0
    observation, domain, bounds = _observation(weight=weight)
    operator = bind_linear_band_grid_observation_operator(
        observation,
        domain,
        spatial_resolution=4,
        spatial_bounds_local_deg=bounds,
    )
    detector_impulse = torch.zeros((4, 4), dtype=torch.float64)
    detector_impulse[1, 2] = 1.0

    weighted = operator.apply_noise_model(detector_impulse)
    contribution = operator.adjoint(weighted)

    torch.testing.assert_close(weighted, torch.zeros_like(weighted))
    torch.testing.assert_close(contribution, torch.zeros_like(contribution))


def test_linear_band_grid_operator_rejects_empty_weight_support() -> None:
    observation, domain, bounds = _observation(
        weight=torch.zeros((4, 4), dtype=torch.float64)
    )

    with pytest.raises(ValueError, match="positive usable weight"):
        bind_linear_band_grid_observation_operator(
            observation,
            domain,
            spatial_resolution=4,
            spatial_bounds_local_deg=bounds,
        )


def test_linear_band_grid_operator_requires_surface_brightness_semantics() -> None:
    observation, domain, bounds = _observation()
    observation.value_semantics = None

    with pytest.raises(ValueError, match="surface_brightness semantics"):
        bind_linear_band_grid_observation_operator(
            observation,
            domain,
            spatial_resolution=4,
            spatial_bounds_local_deg=bounds,
        )


def test_linear_band_grid_operator_rejects_footprint_outside_bounds() -> None:
    observation, domain, _ = _observation()

    with pytest.raises(ValueError, match="detector footprint lies outside"):
        bind_linear_band_grid_observation_operator(
            observation,
            domain,
            spatial_resolution=4,
            spatial_bounds_local_deg=(-0.02, 0.02, -0.02, 0.02),
        )


def test_linear_band_grid_operator_declares_only_applied_bundle_terms() -> None:
    observation, domain, bounds = _observation()
    observation.wavelength = 20_000.0
    observation.exposure_time_s = 100.0
    observation.background_offset = 2.0

    operator = bind_linear_band_grid_observation_operator(
        observation,
        domain,
        spatial_resolution=4,
        spatial_bounds_local_deg=bounds,
    )

    assert operator.operators.spectral_response is None
    assert operator.operators.radiometry is None
    assert operator.operators.background is None
    assert operator.operators.psf is None
    assert operator.operators.noise is not None


def test_linear_band_grid_operator_set_is_chunk_invariant() -> None:
    first, domain, bounds = _observation(
        weight=torch.linspace(0.5, 1.0, 16, dtype=torch.float64).reshape(4, 4)
    )
    second, _, _ = _observation(
        weight=torch.linspace(1.0, 2.0, 16, dtype=torch.float64).reshape(4, 4)
    )
    state = torch.linspace(-0.2, 1.3, 16, dtype=torch.float64).reshape(4, 4)
    single_chunk = bind_linear_band_grid_observations_to_operator_set(
        (first, second),
        domain,
        spatial_resolution=4,
        spatial_bounds_local_deg=bounds,
        exposure_chunk_size=1,
    )
    joint_chunk = bind_linear_band_grid_observations_to_operator_set(
        (first, second),
        domain,
        spatial_resolution=4,
        spatial_bounds_local_deg=bounds,
        exposure_chunk_size=2,
    )
    reversed_order = bind_linear_band_grid_observations_to_operator_set(
        (second, first),
        domain,
        spatial_resolution=4,
        spatial_bounds_local_deg=bounds,
        exposure_chunk_size=1,
    )

    expected = single_chunk.normal_operator_action(state)
    torch.testing.assert_close(
        expected,
        joint_chunk.normal_operator_action(state),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    torch.testing.assert_close(
        expected,
        reversed_order.normal_operator_action(state),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_linear_band_grid_operator_set_rejects_mixed_bands() -> None:
    first, domain, bounds = _observation()
    second, _, _ = _observation()
    second.band_name = "F300W"

    with pytest.raises(ValueError, match="same band"):
        bind_linear_band_grid_observations_to_operator_set(
            (first, second),
            domain,
            spatial_resolution=4,
            spatial_bounds_local_deg=bounds,
        )


def test_linear_band_grid_operator_applies_fixed_effective_psf() -> None:
    kernel = torch.tensor(
        [[0.0, 0.125, 0.0], [0.125, 0.5, 0.125], [0.0, 0.125, 0.0]],
        dtype=torch.float64,
    )
    psf_spec = PSFSpec(
        mode="kernel2d",
        kernel2d=kernel,
        pixel_response_included=True,
    )
    observation, domain, bounds = _observation(psf_spec=psf_spec)
    operator = bind_linear_band_grid_observation_operator(
        observation,
        domain,
        spatial_resolution=4,
        spatial_bounds_local_deg=bounds,
    )
    state = torch.zeros((4, 4), dtype=torch.float64)
    state[1, 1] = 1.0
    expected = torch.zeros_like(state)
    expected[:3, :3] = kernel

    prediction = operator.apply(state)

    torch.testing.assert_close(prediction, expected, rtol=1.0e-12, atol=1.0e-12)


def test_linear_band_grid_cg_matches_dense_weighted_oracle() -> None:
    weight = torch.linspace(0.5, 2.0, 16, dtype=torch.float64).reshape(4, 4)
    kernel = torch.tensor(
        [[0.0, 0.125, 0.0], [0.125, 0.5, 0.125], [0.0, 0.125, 0.0]],
        dtype=torch.float64,
    )
    observation, domain, bounds = _observation(
        weight=weight,
        psf_spec=PSFSpec(
            mode="kernel2d",
            kernel2d=kernel,
            pixel_response_included=True,
        ),
    )
    operator = bind_linear_band_grid_observation_operator(
        observation,
        domain,
        spatial_resolution=4,
        spatial_bounds_local_deg=bounds,
    )
    truth = torch.linspace(0.2, 1.7, 16, dtype=torch.float64).reshape(4, 4)
    data = operator.apply(truth)
    observation.image = data + 0.75
    observation.background_offset = 0.75
    prepared_data = prepare_linear_band_grid_observation_data(observation)
    damping = 0.05
    basis = torch.eye(16, dtype=torch.float64).reshape(16, 4, 4)
    design = torch.stack(
        [operator.apply(vector).reshape(-1) for vector in basis], dim=1
    )
    precision = torch.diag(weight.reshape(-1))
    dense_normal = design.T @ precision @ design + damping * torch.eye(
        16, dtype=torch.float64
    )
    dense_rhs = design.T @ precision @ data.reshape(-1)
    expected = torch.linalg.solve(dense_normal, dense_rhs).reshape(4, 4)

    result = operator.solve_normal_equation_cg(
        prepared_data,
        iterations=32,
        damping=damping,
    )

    torch.testing.assert_close(result.solution, expected, rtol=1.0e-8, atol=1.0e-8)
    assert result.residual_norm_history[-1] < result.residual_norm_history[0]


def test_linear_band_grid_data_subtracts_fixed_background_and_masks() -> None:
    weight = torch.ones((4, 4), dtype=torch.float64)
    weight[2, 1] = 0.0
    observation, _, _ = _observation(weight=weight)
    observation.image = torch.full((4, 4), 10.0, dtype=torch.float64)
    observation.sky_background = torch.full((4, 4), 1.0, dtype=torch.float64)
    observation.instrument_background = torch.full(
        (4, 4), 2.0, dtype=torch.float64
    )
    observation.detector_background = torch.full(
        (4, 4), 3.0, dtype=torch.float64
    )
    observation.background_offset = 0.5
    calibrated_background = torch.full((4, 4), 0.25, dtype=torch.float64)
    expected = torch.full((4, 4), 3.25, dtype=torch.float64)
    expected[2, 1] = 0.0

    data = prepare_linear_band_grid_observation_data(
        observation,
        calibrated_background=calibrated_background,
    )

    torch.testing.assert_close(data, expected)


def test_linear_band_grid_data_rejects_nonfinite_background() -> None:
    observation, _, _ = _observation()
    observation.sky_background = torch.zeros((4, 4), dtype=torch.float64)
    observation.sky_background[0, 0] = torch.nan

    with pytest.raises(ValueError, match="background components must be finite"):
        prepare_linear_band_grid_observation_data(observation)


@pytest.mark.parametrize("tile_hw", [(2, 2), (3, 2)])
@pytest.mark.parametrize("with_psf", [False, True])
def test_linear_band_grid_operator_full_vs_tiled_parity(
    tile_hw: tuple[int, int],
    with_psf: bool,
) -> None:
    weight = torch.linspace(0.4, 1.9, 16, dtype=torch.float64).reshape(4, 4)
    weight[0, 3] = 0.0
    psf_spec = None
    if with_psf:
        kernel = torch.tensor(
            [[0.02, 0.08, 0.05], [0.12, 0.38, 0.17], [0.03, 0.09, 0.06]],
            dtype=torch.float64,
        )
        psf_spec = PSFSpec(
            mode="kernel2d",
            kernel2d=kernel,
            pixel_response_included=True,
        )
    observation, domain, bounds = _observation(weight=weight, psf_spec=psf_spec)
    full_op = bind_linear_band_grid_observation_operator(
        observation,
        domain,
        spatial_resolution=5,
        spatial_bounds_local_deg=bounds,
    )
    tiled_op = bind_linear_band_grid_observation_operator(
        observation,
        domain,
        spatial_resolution=5,
        spatial_bounds_local_deg=bounds,
        spatial_tile_hw=tile_hw,
    )
    state = torch.linspace(-0.4, 1.8, 25, dtype=torch.float64).reshape(5, 5)
    dual = torch.linspace(-0.9, 1.4, 16, dtype=torch.float64).reshape(4, 4)

    torch.testing.assert_close(
        tiled_op.apply(state),
        full_op.apply(state),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    torch.testing.assert_close(
        tiled_op.adjoint(dual),
        full_op.adjoint(dual),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    torch.testing.assert_close(
        tiled_op.normal_operator_action(state),
        full_op.normal_operator_action(state),
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    data = full_op.apply(state)
    full_cg = full_op.solve_normal_equation_cg(data, iterations=20, damping=0.05)
    tiled_cg = tiled_op.solve_normal_equation_cg(data, iterations=20, damping=0.05)
    torch.testing.assert_close(
        tiled_cg.solution,
        full_cg.solution,
        rtol=1.0e-11,
        atol=1.0e-11,
    )


def test_linear_band_grid_operator_set_full_vs_tiled_parity() -> None:
    kernel = torch.tensor(
        [[0.01, 0.09, 0.05], [0.14, 0.40, 0.16], [0.02, 0.08, 0.05]],
        dtype=torch.float64,
    )
    psf_spec = PSFSpec(
        mode="kernel2d",
        kernel2d=kernel,
        pixel_response_included=True,
    )
    first, domain, bounds = _observation(
        weight=torch.linspace(0.5, 1.5, 16, dtype=torch.float64).reshape(4, 4),
        psf_spec=psf_spec,
    )
    second, _, _ = _observation(
        weight=torch.linspace(0.8, 2.2, 16, dtype=torch.float64).reshape(4, 4),
        psf_spec=psf_spec,
    )
    full_set = bind_linear_band_grid_observations_to_operator_set(
        (first, second),
        domain,
        spatial_resolution=4,
        spatial_bounds_local_deg=bounds,
        exposure_chunk_size=1,
    )
    tiled_set = bind_linear_band_grid_observations_to_operator_set(
        (first, second),
        domain,
        spatial_resolution=4,
        spatial_bounds_local_deg=bounds,
        exposure_chunk_size=1,
        spatial_tile_hw=(2, 3),
    )
    state = torch.linspace(0.1, 1.6, 16, dtype=torch.float64).reshape(4, 4)
    data = (full_set.operators[0].apply(state), full_set.operators[1].apply(state))

    torch.testing.assert_close(
        tiled_set.normal_operator_action(state),
        full_set.normal_operator_action(state),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    full_cg = full_set.solve_normal_equation_cg(data, iterations=24, damping=0.05)
    tiled_cg = tiled_set.solve_normal_equation_cg(data, iterations=24, damping=0.05)
    torch.testing.assert_close(
        tiled_cg.solution,
        full_cg.solution,
        rtol=1.0e-11,
        atol=1.0e-11,
    )


def test_linear_band_grid_operator_adjoint_works_under_no_grad() -> None:
    observation, domain, bounds = _observation()
    operator = bind_linear_band_grid_observation_operator(
        observation,
        domain,
        spatial_resolution=4,
        spatial_bounds_local_deg=bounds,
        spatial_tile_hw=(2, 2),
    )
    state = torch.linspace(0.1, 1.6, 16, dtype=torch.float64).reshape(4, 4)
    dual = torch.linspace(-0.5, 1.1, 16, dtype=torch.float64).reshape(4, 4)

    expected_adj = operator.adjoint(dual)
    with torch.no_grad():
        no_grad_adj = operator.adjoint(dual)
        cg_res = operator.solve_normal_equation_cg(
            operator.apply(state),
            iterations=16,
            damping=0.05,
        )

    torch.testing.assert_close(no_grad_adj, expected_adj, rtol=1.0e-12, atol=1.0e-12)
    assert torch.isfinite(cg_res.solution).all()


def test_linear_band_grid_operator_cpu_cuda_and_multi_dtype_cache_parity() -> None:
    kernel = torch.tensor(
        [[0.0, 0.125, 0.0], [0.125, 0.5, 0.125], [0.0, 0.125, 0.0]],
        dtype=torch.float64,
    )
    observation, domain, bounds = _observation(
        psf_spec=PSFSpec(
            mode="kernel2d",
            kernel2d=kernel,
            pixel_response_included=True,
        )
    )
    operator = bind_linear_band_grid_observation_operator(
        observation,
        domain,
        spatial_resolution=4,
        spatial_bounds_local_deg=bounds,
        spatial_tile_hw=(2, 2),
    )
    state64 = torch.linspace(0.1, 1.6, 16, dtype=torch.float64).reshape(4, 4)
    state32 = state64.to(dtype=torch.float32)

    out64 = operator.normal_operator_action(state64)
    out32 = operator.normal_operator_action(state32)
    torch.testing.assert_close(
        out32.to(dtype=torch.float64),
        out64,
        rtol=1.0e-5,
        atol=1.0e-5,
    )

    if torch.cuda.is_available():
        state_cuda = state64.to(device="cuda")
        out_cuda = operator.normal_operator_action(state_cuda)
        assert out_cuda.device.type == "cuda"
        torch.testing.assert_close(
            out_cuda.cpu(),
            out64,
            rtol=1.0e-11,
            atol=1.0e-11,
        )


def test_solve_linear_band_grid_lsqr_matches_dense_oracle_damped_and_undamped() -> None:
    weight = torch.linspace(0.5, 2.0, 16, dtype=torch.float64).reshape(4, 4)
    kernel = torch.tensor(
        [[0.0, 0.125, 0.0], [0.125, 0.5, 0.125], [0.0, 0.125, 0.0]],
        dtype=torch.float64,
    )
    observation, domain, bounds = _observation(
        weight=weight,
        psf_spec=PSFSpec(
            mode="kernel2d",
            kernel2d=kernel,
            pixel_response_included=True,
        ),
    )
    # Use a wider spatial_bounds_local_deg and resolution=5 so outer grid border cells
    # are strictly outside [-0.5, 0.5] normalized coordinates (making the undamped
    # normal matrix genuinely rank-deficient with exact zero border columns).
    wide_bounds = (-0.08, 0.08, -0.08, 0.08)
    operator = bind_linear_band_grid_observation_operator(
        observation,
        domain,
        spatial_resolution=5,
        spatial_bounds_local_deg=wide_bounds,
    )
    truth = torch.linspace(0.2, 2.6, 25, dtype=torch.float64).reshape(5, 5)
    data = operator.apply(truth)

    basis = torch.eye(25, dtype=torch.float64).reshape(25, 5, 5)
    design = torch.stack(
        [operator.apply(vector).reshape(-1) for vector in basis], dim=1
    )
    sqrt_w = torch.diag(torch.sqrt(weight.reshape(-1)))
    a_weighted = sqrt_w @ design
    b_weighted = sqrt_w @ data.reshape(-1)

    # 1. Damped solve (damping = 0.05)
    damping = 0.05
    dense_normal = a_weighted.T @ a_weighted + damping * torch.eye(
        25, dtype=torch.float64
    )
    expected_damped = torch.linalg.solve(
        dense_normal, a_weighted.T @ b_weighted
    ).reshape(5, 5)
    lsqr_damped = solve_linear_band_grid_lsqr(
        operator,
        data,
        iterations=64,
        damping=damping,
        tolerance=1.0e-12,
    )
    torch.testing.assert_close(
        lsqr_damped.solution,
        expected_damped,
        rtol=1.0e-7,
        atol=1.0e-7,
    )

    # 2. Undamped rank-deficient minimum-norm solve (damping = 0.0)
    rank = int(torch.linalg.matrix_rank(a_weighted).item())
    assert rank == 9
    expected_min_norm = (torch.linalg.pinv(a_weighted) @ b_weighted).reshape(5, 5)
    lsqr_undamped = solve_linear_band_grid_lsqr(
        operator,
        data,
        iterations=64,
        damping=0.0,
        tolerance=1.0e-12,
    )
    torch.testing.assert_close(
        lsqr_undamped.solution,
        expected_min_norm,
        rtol=1.0e-7,
        atol=1.0e-7,
    )


def test_solve_linear_band_grid_lsqr_operator_set_and_validation() -> None:
    first, domain, bounds = _observation(
        weight=torch.linspace(0.5, 1.5, 16, dtype=torch.float64).reshape(4, 4)
    )
    second, _, _ = _observation(
        weight=torch.linspace(1.0, 2.0, 16, dtype=torch.float64).reshape(4, 4)
    )
    op_set = bind_linear_band_grid_observations_to_operator_set(
        (first, second),
        domain,
        spatial_resolution=4,
        spatial_bounds_local_deg=bounds,
    )
    truth = torch.linspace(0.1, 1.6, 16, dtype=torch.float64).reshape(4, 4)
    data = (op_set.operators[0].apply(truth), op_set.operators[1].apply(truth))

    cg_res = op_set.solve_normal_equation_cg(
        data, iterations=48, damping=0.05
    )
    lsqr_res = solve_linear_band_grid_lsqr(
        op_set, data, iterations=48, damping=0.05, tolerance=1.0e-12
    )
    torch.testing.assert_close(
        lsqr_res.solution, cg_res.solution, rtol=1.0e-7, atol=1.0e-7
    )

    with pytest.raises(ValueError, match="iterations must be positive"):
        solve_linear_band_grid_lsqr(op_set, data, iterations=0)
    with pytest.raises(ValueError, match="damping must be finite and non-negative"):
        solve_linear_band_grid_lsqr(op_set, data, damping=-0.1)


def _make_stationary_psf_and_bg_models(
    *,
    bg_bias: float = 0.35,
    spatially_varying_psf: bool = False,
    spatially_varying_bg: bool = False,
) -> tuple[ConditionedKernelPSF, ConditionedBackgroundField]:
    encoder = SpatioSpectralTemporalEncoder(
        spatial_center=(1.5, 1.5),
        spatial_scale=(3.0, 3.0),
        wavelength_center=20_000.0,
        spectral_scale=1.0,
        time_center_mjd=60_000.0,
        temporal_scale_days=1.0,
        spatial_frequencies=0,
        spectral_frequencies=0,
        temporal_frequencies=0,
    )
    psf_model = ConditionedKernelPSF(
        encoder,
        n_basis=2 if spatially_varying_psf else 1,
        kernel_size=3,
        init_fwhm_pix=1.5,
        hidden_dim=8,
    )
    if spatially_varying_psf:
        with torch.no_grad():
            first_mix = psf_model.mixer[0]
            last_mix = psf_model.mixer[-1]
            assert isinstance(first_mix, nn.Linear)
            assert isinstance(last_mix, nn.Linear)
            first_mix.weight.fill_(1.0)
            first_mix.bias.zero_()
            last_mix.weight.copy_(
                torch.linspace(-1.0, 1.0, last_mix.weight.numel()).reshape_as(
                    last_mix.weight
                )
            )
            last_mix.bias.zero_()
            psf_model.kernel_basis.raw_kernels.copy_(
                torch.tensor(
                    [
                        [[0.0, 1.0, 0.0], [1.0, 4.0, 1.0], [0.0, 1.0, 0.0]],
                        [[2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 2.0]],
                    ],
                    dtype=psf_model.kernel_basis.raw_kernels.dtype,
                )
            )
    for p in psf_model.parameters():
        p.requires_grad_(False)

    bg_model = ConditionedBackgroundField(encoder, hidden_dim=8, hidden_layers=1)
    with torch.no_grad():
        for p in bg_model.parameters():
            p.zero_()
        out_linear = bg_model.network[-1]
        assert isinstance(out_linear, nn.Linear)
        out_linear.bias.fill_(bg_bias)
        if spatially_varying_bg:
            first_linear = bg_model.network[0]
            assert isinstance(first_linear, nn.Linear)
            first_linear.weight.fill_(1.0)
            out_linear.weight.fill_(1.0)
    for p in bg_model.parameters():
        p.requires_grad_(False)

    return psf_model, bg_model


def test_adapt_stationary_calibrated_band_observation_happy_path_and_guards() -> None:
    observation, domain, bounds = _observation()
    observation.value_unit = "MJy/sr"
    observation.wavelength = 20_000.0
    observation.image = torch.full((4, 4), 2.35, dtype=torch.float64)

    psf_model, bg_model = _make_stationary_psf_and_bg_models(bg_bias=0.35)
    bundle = field_observation_operator_bundle(observation).with_updates(
        psf=LearnedPSFOp(
            psf_model,
            frozen=True,
            pixel_response_included=True,
            pixel_response_provenance="observed_cal_stars_v2",
        ),
        background=LearnedBackgroundOp(bg_model, coordinate_frame="detector"),
    )

    adapted = adapt_stationary_calibrated_band_observation(
        observation,
        bundle,
        expected_value_unit="MJy/sr",
        background_gauge_policy="training_mean",
    )
    assert adapted.observation.psf_spec is not None
    assert adapted.observation.psf_spec.mode == "kernel2d"
    assert adapted.observation.psf_spec.pixel_response_included is True
    torch.testing.assert_close(
        adapted.prepared_data,
        torch.full((4, 4), 2.0, dtype=torch.float64),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    operator = bind_linear_band_grid_observation_operator(
        adapted.observation,
        domain,
        spatial_resolution=4,
        spatial_bounds_local_deg=bounds,
    )
    res = solve_linear_band_grid_lsqr(
        operator, adapted.prepared_data, iterations=24, damping=0.05
    )
    assert torch.isfinite(res.solution).all()

    # Also verify ExplicitPSFOp stationary adapter branch
    explicit_bundle = field_observation_operator_bundle(
        adapted.observation
    ).with_updates(
        psf=ExplicitPSFOp(
            pixel_response_included=True,
            pixel_response_provenance="explicit_effective_kernel",
        ),
        background=LearnedBackgroundOp(bg_model, coordinate_frame="detector"),
    )
    adapted_explicit = adapt_stationary_calibrated_band_observation(
        adapted.observation,
        explicit_bundle,
        expected_value_unit="MJy/sr",
    )
    torch.testing.assert_close(
        adapted_explicit.effective_psf_kernel,
        adapted.effective_psf_kernel,
        rtol=1.0e-6,
        atol=1.0e-6,
    )

    # Fail-closed guard: raw detector unit
    raw_obs = FieldObservation(**{**observation.__dict__, "value_unit": "DN"})
    with pytest.raises(ValueError, match="raw detector unit"):
        adapt_stationary_calibrated_band_observation(raw_obs, bundle)

    # Fail-closed guard: unapplied radiometry
    rad_obs = FieldObservation(
        **{**observation.__dict__, "exposure_time_s": 100.0}
    )
    with pytest.raises(ValueError, match="unapplied radiometric conversion"):
        adapt_stationary_calibrated_band_observation(rad_obs, bundle)

    bundle_with_rad = bundle.with_updates(radiometry=ExplicitRadiometryOp())
    with pytest.raises(ValueError, match="unapplied radiometric conversion"):
        adapt_stationary_calibrated_band_observation(observation, bundle_with_rad)

    # Fail-closed guard: missing pixel_response_included
    bad_psf_bundle = bundle.with_updates(
        psf=LearnedPSFOp(
            psf_model,
            frozen=True,
            pixel_response_included=False,
            pixel_response_provenance="observed_cal_stars_v2",
        )
    )
    with pytest.raises(ValueError, match="pixel_response_included=True"):
        adapt_stationary_calibrated_band_observation(observation, bad_psf_bundle)

    # Fail-closed guard: spatially varying PSF
    var_psf, _ = _make_stationary_psf_and_bg_models(spatially_varying_psf=True)
    var_psf_bundle = bundle.with_updates(
        psf=LearnedPSFOp(
            var_psf,
            frozen=True,
            pixel_response_included=True,
            pixel_response_provenance="observed_cal_stars_v2",
        )
    )
    with pytest.raises(ValueError, match="calibrated PSF varies across detector"):
        adapt_stationary_calibrated_band_observation(observation, var_psf_bundle)

    # Fail-closed guard: spatially varying background
    _, var_bg = _make_stationary_psf_and_bg_models(spatially_varying_bg=True)
    var_bg_bundle = bundle.with_updates(
        background=LearnedBackgroundOp(var_bg, coordinate_frame="detector")
    )
    with pytest.raises(
        ValueError, match="calibrated background varies across detector"
    ):
        adapt_stationary_calibrated_band_observation(observation, var_bg_bundle)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_load_stationary_calibrated_band_observations_end_to_end(
    tmp_path: Path,
) -> None:
    psf_kernels_path = tmp_path / "psf_kernels.fits"
    psf_weights_path = tmp_path / "psf_weights.fits"
    kernel = torch.tensor(
        [[[0.0, 0.125, 0.0], [0.125, 0.5, 0.125], [0.0, 0.125, 0.0]]],
        dtype=torch.float32,
    )
    weights = torch.ones_like(kernel)
    write(str(psf_kernels_path), kernel, overwrite=True)
    write(str(psf_weights_path), weights, overwrite=True)

    manifest_payload = {
        "extraction": {
            "products": {
                "psf_kernels": {
                    "path": str(psf_kernels_path),
                    "sha256": _sha256_file(psf_kernels_path),
                },
                "psf_pixel_weights": {
                    "path": str(psf_weights_path),
                    "sha256": _sha256_file(psf_weights_path),
                },
            }
        },
        "evidence": [
            {
                "calibration_split": "train",
                "role": "psf",
                "detector": "NRCB1",
                "exposure_id": "exp_train",
                "band": "F200W",
                "detector_x": 1.0,
                "detector_y": 2.0,
                "time_mjd_utc": 60_000.0,
                "sample_weight": 1.0,
                "target_index": 0,
            },
            {
                "calibration_split": "train",
                "role": "background",
                "detector": "NRCB1",
                "exposure_id": "exp_train",
                "band": "F200W",
                "detector_x": 1.0,
                "detector_y": 2.0,
                "time_mjd_utc": 60_000.0,
                "sample_weight": 2.0,
                "target_value": 0.4,
            },
            {
                "calibration_split": "train",
                "role": "psf",
                "detector": "NRCB1",
                "exposure_id": "exp_val",
                "band": "F200W",
                "detector_x": 1.0,
                "detector_y": 2.0,
                "time_mjd_utc": 60_000.0,
                "sample_weight": 1.0,
                "target_index": 0,
            },
            {
                "calibration_split": "train",
                "role": "background",
                "detector": "NRCB1",
                "exposure_id": "exp_val",
                "band": "F200W",
                "detector_x": 1.0,
                "detector_y": 2.0,
                "time_mjd_utc": 60_000.0,
                "sample_weight": 1.0,
                "target_value": 0.7,
            },
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")
    manifest_sha = _sha256_file(manifest_path)

    encoder = SpatioSpectralTemporalEncoder(
        spatial_center=(1.0, 2.0),
        spatial_scale=(1.0, 1.0),
        wavelength_center=19_886.48,
        spectral_scale=1.0,
        time_center_mjd=60_000.0,
        temporal_scale_days=1.0,
        spatial_frequencies=0,
        spectral_frequencies=0,
        temporal_frequencies=0,
    )
    groups = []
    for exp_id, bg_val in (("exp_train", 0.4), ("exp_val", 0.7)):
        psf_m = ConditionedKernelPSF(
            encoder, n_basis=1, kernel_size=3, init_fwhm_pix=1.5, hidden_dim=32
        )
        psf_ckpt = tmp_path / f"psf_{exp_id}.pt"
        torch.save(
            {
                "schema_version": 2,
                "detector": "NRCB1",
                "exposure_id": exp_id,
                "band": "F200W",
                "calibration_scope": "exposure",
                "role": "psf",
                "family": "stationary",
                "pixel_response_included": True,
                "pixel_response_provenance": "observed_cal_stars_v2",
                "state_dict": psf_m.state_dict(),
            },
            psf_ckpt,
        )
        groups.append(
            {
                "detector": "NRCB1",
                "exposure_id": exp_id,
                "band": "F200W",
                "role": "psf",
                "family": "stationary",
                "pixel_response_included": True,
                "pixel_response_provenance": "observed_cal_stars_v2",
                "checkpoint": psf_ckpt.name,
                "checkpoint_sha256": _sha256_file(psf_ckpt),
            }
        )

        bg_m = ConditionedBackgroundField(encoder, hidden_dim=32, hidden_layers=2)
        with torch.no_grad():
            for p in bg_m.parameters():
                p.zero_()
            out_layer = bg_m.network[-1]
            assert isinstance(out_layer, nn.Linear)
            out_layer.bias.fill_(bg_val)
        bg_ckpt = tmp_path / f"bg_{exp_id}.pt"
        torch.save(
            {
                "schema_version": 2,
                "detector": "NRCB1",
                "exposure_id": exp_id,
                "band": "F200W",
                "calibration_scope": "exposure",
                "role": "background",
                "family": "stationary",
                "state_dict": bg_m.state_dict(),
            },
            bg_ckpt,
        )
        groups.append(
            {
                "detector": "NRCB1",
                "exposure_id": exp_id,
                "band": "F200W",
                "role": "background",
                "family": "stationary",
                "checkpoint": bg_ckpt.name,
                "checkpoint_sha256": _sha256_file(bg_ckpt),
            }
        )

    report_path = tmp_path / "operator_fit_report.json"
    report_path.write_text(
        json.dumps(
            {
                "manifest_sha256": manifest_sha,
                "calibration_scope": "exposure",
                "groups": groups,
            }
        ),
        encoding="utf-8",
    )

    passband_file = tmp_path / "f200w_passband.json"
    passband_file.write_text('{"band": "F200W"}', encoding="utf-8")
    pb_sha = _sha256_file(passband_file)

    obs_train, _, _ = _observation()
    obs_train.value_unit = "MJy/sr"
    obs_train.wavelength = 19_886.48
    obs_train.image = torch.full((4, 4), 1.0, dtype=torch.float64)

    obs_val, _, _ = _observation()
    obs_val.value_unit = "MJy/sr"
    obs_val.wavelength = 19_886.48
    obs_val.image = torch.full((4, 4), 1.0, dtype=torch.float64)

    ctx_train = ForwardContext(
        survey="F200W",
        detector="NRCB1",
        extra={
            "calibration_exposure_id": "exp_train",
            "split": "train",
            "measured_passband_path": str(passband_file),
            "measured_passband_sha256": pb_sha,
        },
    )
    ctx_val = ForwardContext(
        survey="F200W",
        detector="NRCB1",
        extra={
            "calibration_exposure_id": "exp_val",
            "split": "val",
            "measured_passband_path": str(passband_file),
            "measured_passband_sha256": pb_sha,
        },
    )

    adapted, audit = load_stationary_calibrated_band_observations(
        (obs_train, obs_val),
        (ctx_train, ctx_val),
        manifest_path=manifest_path,
        fit_report_path=report_path,
        background_gauge_policy="training_mean",
    )
    assert len(adapted) == 2
    assert audit["background_gauge"]["policy"] == "training_mean"
    # Train exposure background (0.4) minus train mean (0.4) -> 0.0 subtracted -> 1.0
    torch.testing.assert_close(
        adapted[0].prepared_data,
        torch.full((4, 4), 1.0, dtype=torch.float64),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    # Val exposure background (0.7) minus train mean (0.4) -> 0.3 subtracted -> 0.7
    torch.testing.assert_close(
        adapted[1].prepared_data,
        torch.full((4, 4), 0.7, dtype=torch.float64),
        rtol=1.0e-6,
        atol=1.0e-6,
    )

    # Hash mismatch on passband fails closed
    bad_ctx_val = ForwardContext(
        survey="F200W",
        detector="NRCB1",
        extra={
            "calibration_exposure_id": "exp_val",
            "split": "val",
            "measured_passband_path": str(passband_file),
            "measured_passband_sha256": "0" * 64,
        },
    )
    with pytest.raises(ValueError, match="measured passband SHA-256"):
        load_stationary_calibrated_band_observations(
            (obs_train, obs_val),
            (ctx_train, bad_ctx_val),
            manifest_path=manifest_path,
            fit_report_path=report_path,
        )


def test_linear_band_grid_field_matches_bound_operator_and_stationary_render() -> None:
    observation, domain, bounds = _observation()
    observation.value_unit = "MJy/sr"
    observation.wavelength = 20_000.0
    observation.image = torch.linspace(-0.5, 2.5, 16, dtype=torch.float64).reshape(4, 4)

    psf_model, bg_model = _make_stationary_psf_and_bg_models(bg_bias=0.35)
    bundle = field_observation_operator_bundle(observation).with_updates(
        psf=LearnedPSFOp(
            psf_model,
            frozen=True,
            pixel_response_included=True,
            pixel_response_provenance="observed_cal_stars_v2",
        ),
        background=LearnedBackgroundOp(bg_model, coordinate_frame="detector"),
    )
    adapted = adapt_stationary_calibrated_band_observation(
        observation,
        bundle,
        expected_value_unit="MJy/sr",
        background_gauge_policy="none",
    )
    bound_op = bind_linear_band_grid_observation_operator(
        adapted.observation,
        domain,
        spatial_resolution=4,
        spatial_bounds_local_deg=bounds,
    )

    model = LinearBandGridField(
        domain,
        spatial_resolution=4,
        spatial_bounds_local_deg=bounds,
        band_name="F200W",
        dtype=torch.float64,
    )
    # Include negative values to confirm signed surface-brightness support
    signed_grid = torch.linspace(-1.2, 2.4, 16, dtype=torch.float64).reshape(4, 4)
    with torch.no_grad():
        model.grid.copy_(signed_grid)

    rendered_obs = render_field_observation(
        model,
        observation,
        operator_bundle=bundle,
    )
    expected_obs = bound_op.apply(signed_grid) + adapted.calibrated_background
    torch.testing.assert_close(
        rendered_obs,
        expected_obs,
        rtol=1.0e-6,
        atol=1.0e-6,
    )

    # Also check render_bandpass invariance across wavelength samples
    ra_deg, dec_deg = model._grid_world_coords()
    direct_eval = model(
        ra_deg,
        dec_deg,
        torch.full_like(ra_deg, 20_000.0),
    )
    bandpass_eval = model.render_bandpass(
        ra_deg,
        dec_deg,
        torch.tensor([19_000.0, 20_000.0, 21_000.0], dtype=torch.float64),
        torch.tensor([0.25, 0.5, 0.25], dtype=torch.float64),
    )
    torch.testing.assert_close(direct_eval, signed_grid, rtol=1.0e-12, atol=1.0e-12)
    torch.testing.assert_close(bandpass_eval, signed_grid, rtol=1.0e-12, atol=1.0e-12)


def test_linear_band_grid_field_initialize_from_product_signed_2d_and_3d() -> None:
    from torchsky.map.product import BandpassSpec, ProductSpec

    _, domain, bounds = _observation()
    header = {
        "NAXIS": 2,
        "NAXIS1": 4,
        "NAXIS2": 4,
        "CRPIX1": 2.5,
        "CRPIX2": 2.5,
        "CRVAL1": domain.center_ra_deg,
        "CRVAL2": domain.center_dec_deg,
        "CDELT1": -(bounds[1] - bounds[0]) / 3.0,
        "CDELT2": (bounds[3] - bounds[2]) / 3.0,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
    }
    plane_f150w = torch.linspace(-0.8, 1.6, 16, dtype=torch.float64).reshape(4, 4)
    plane_f277w = torch.linspace(-2.0, 3.0, 16, dtype=torch.float64).reshape(4, 4)
    cube = torch.stack([plane_f150w, plane_f277w], dim=0)
    spec = ProductSpec(
        target_header=header,
        bands=(
            BandpassSpec(
                label="F150W",
                transmission_center=15_000.0,
                transmission_sigma=1_000.0,
            ),
            BandpassSpec(
                label="F277W",
                transmission_center=27_700.0,
                transmission_sigma=1_000.0,
            ),
        ),
    )
    product = ProductResult(
        product=spec,
        data=cube,
        weight=torch.ones_like(cube),
        header=header,
    )
    model = LinearBandGridField(
        domain,
        spatial_resolution=4,
        spatial_bounds_local_deg=bounds,
        band_name="F277W",
        dtype=torch.float64,
    )
    model.initialize_from_product(product)
    assert float(model.grid.min().item()) < 0.0
    assert torch.isfinite(model.grid).all()


def test_solve_linear_band_grid_cg_and_diagonal_majorizer_match_dense_oracle() -> None:
    weight_1 = torch.linspace(0.5, 2.5, 16, dtype=torch.float64).reshape(4, 4)
    weight_2 = torch.linspace(1.0, 4.0, 16, dtype=torch.float64).reshape(4, 4)
    kernel = torch.tensor(
        [[0.0, 0.125, 0.0], [0.125, 0.5, 0.125], [0.0, 0.125, 0.0]],
        dtype=torch.float64,
    )
    psf_spec = PSFSpec(
        mode="kernel2d",
        kernel2d=kernel,
        pixel_response_included=True,
    )
    obs1, domain, bounds = _observation(weight=weight_1, psf_spec=psf_spec)
    obs2, _, _ = _observation(weight=weight_2, psf_spec=psf_spec)
    op_set = bind_linear_band_grid_observations_to_operator_set(
        (obs1, obs2),
        domain,
        spatial_resolution=4,
        spatial_bounds_local_deg=bounds,
    )
    truth = torch.linspace(-0.4, 1.8, 16, dtype=torch.float64).reshape(4, 4)
    x0 = torch.linspace(-0.2, 1.2, 16, dtype=torch.float64).reshape(4, 4)
    data = (op_set.operators[0].apply(truth), op_set.operators[1].apply(truth))

    damping = 0.05
    smoothness_weight = 0.02

    basis = torch.eye(16, dtype=torch.float64).reshape(16, 4, 4)
    # Build exact dense matrix of (A^T N^-1 A + damping * I + smoothness * L)
    # and check sensitivity majorizer A^T N^-1 1 >= col_sums(A^T N^-1 A) >= diag(A^T N^-1 A)
    dense_ata = torch.stack(
        [op_set.normal_operator_action(vec).reshape(-1) for vec in basis], dim=1
    )
    expected_col_sums = dense_ata.sum(dim=0).reshape(4, 4)
    expected_sensitivity = op_set.weighted_adjoint(
        (torch.ones_like(data[0]), torch.ones_like(data[1]))
    )
    majorizer = linear_band_grid_diagonal_majorizer(
        op_set,
        data,
        damping=damping,
        smoothness_weight=smoothness_weight,
    )
    torch.testing.assert_close(
        majorizer,
        expected_sensitivity + damping + 8.0 * smoothness_weight,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert bool(
        (
            majorizer + 1.0e-12
            >= expected_col_sums + damping + 8.0 * smoothness_weight
        ).all()
    )

    # Build dense discrete Laplacian L
    def _laplacian(x_grid: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x_grid)
        dy = x_grid[1:, :] - x_grid[:-1, :]
        out[:-1, :] = out[:-1, :] - dy
        out[1:, :] = out[1:, :] + dy
        dx = x_grid[:, 1:] - x_grid[:, :-1]
        out[:, :-1] = out[:, :-1] - dx
        out[:, 1:] = out[:, 1:] + dx
        return out

    dense_lap = torch.stack(
        [_laplacian(vec).reshape(-1) for vec in basis], dim=1
    )
    dense_lhs = (
        dense_ata
        + damping * torch.eye(16, dtype=torch.float64)
        + smoothness_weight * dense_lap
    )
    rhs_base = (
        op_set.operators[0].adjoint(weight_1 * data[0])
        + op_set.operators[1].adjoint(weight_2 * data[1])
    ).reshape(-1)
    dense_rhs = rhs_base + damping * x0.reshape(-1)
    expected_sol = torch.linalg.solve(dense_lhs, dense_rhs).reshape(4, 4)

    diag: dict[str, object] = {}
    cg_res = solve_linear_band_grid_cg(
        op_set,
        data,
        initial_solution=x0,
        iterations=32,
        damping=damping,
        smoothness_weight=smoothness_weight,
        prior_reference="initial",
        preconditioner="diagonal_majorizer",
        tolerance=1.0e-12,
        diagnostics=diag,
    )
    torch.testing.assert_close(
        cg_res.solution,
        expected_sol,
        rtol=1.0e-8,
        atol=1.0e-8,
    )
    assert diag["solver"] == "cg"
    assert diag["preconditioner"] == "diagonal_majorizer"
    assert diag["prior_reference"] == "initial"
    assert int(diag["iterations"]) == len(cg_res.residual_norm_history) - 1
    assert int(diag["normal_operator_applications"]) == int(diag["iterations"]) + 1
    assert float(diag["grid_covered_fraction"]) == 1.0


def test_batched_operator_set_mixed_psf_sizes_and_differentiable_cg() -> None:
    k3 = torch.tensor(
        [[0.0, 0.125, 0.0], [0.125, 0.5, 0.125], [0.0, 0.125, 0.0]],
        dtype=torch.float64,
    )
    k5 = torch.zeros((5, 5), dtype=torch.float64)
    k5[1:4, 1:4] = k3
    obs1, domain, bounds = _observation(
        weight=torch.linspace(0.5, 1.5, 16, dtype=torch.float64).reshape(4, 4),
        psf_spec=PSFSpec(mode="kernel2d", kernel2d=k3, pixel_response_included=True),
    )
    obs2, _, _ = _observation(
        weight=torch.linspace(1.0, 2.5, 16, dtype=torch.float64).reshape(4, 4),
        psf_spec=PSFSpec(mode="kernel2d", kernel2d=k5, pixel_response_included=True),
    )
    obs3, _, _ = _observation(
        weight=torch.linspace(0.8, 1.8, 16, dtype=torch.float64).reshape(4, 4),
        psf_spec=None,
    )
    op_set_batched = bind_linear_band_grid_observations_to_operator_set(
        (obs1, obs2, obs3),
        domain,
        spatial_resolution=4,
        spatial_bounds_local_deg=bounds,
        exposure_chunk_size=8,
    )
    op_set_serial = bind_linear_band_grid_observations_to_operator_set(
        (obs1, obs2, obs3),
        domain,
        spatial_resolution=4,
        spatial_bounds_local_deg=bounds,
        exposure_chunk_size=1,
        spatial_tile_hw=(2, 2),
    )
    state = torch.linspace(-0.3, 1.4, 16, dtype=torch.float64).reshape(4, 4)
    for pred_b, pred_s in zip(
        op_set_batched.apply(state), op_set_serial.apply(state), strict=True
    ):
        torch.testing.assert_close(pred_b, pred_s, rtol=1.0e-12, atol=1.0e-12)

    torch.testing.assert_close(
        op_set_batched.normal_operator_action(state),
        op_set_serial.normal_operator_action(state),
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    state_grad = state.clone().requires_grad_(True)
    normal_out = op_set_batched.normal_operator_action(state_grad)
    loss_normal = torch.sum(normal_out.square())
    loss_normal.backward()
    assert state_grad.grad is not None
    assert torch.isfinite(state_grad.grad).all()

    data_0 = op_set_batched.operators[0].apply(state).detach().requires_grad_(True)
    data_1 = op_set_batched.operators[1].apply(state).detach().requires_grad_(True)
    data_2 = op_set_batched.operators[2].apply(state).detach().requires_grad_(True)
    cg_diff = solve_linear_band_grid_cg(
        op_set_batched,
        (data_0, data_1, data_2),
        iterations=16,
        damping=0.05,
        smoothness_weight=0.01,
    )
    torch.sum(cg_diff.solution.square()).backward()
    assert data_0.grad is not None and torch.isfinite(data_0.grad).all()
    assert data_1.grad is not None and torch.isfinite(data_1.grad).all()
    assert data_2.grad is not None and torch.isfinite(data_2.grad).all()


def test_solve_joint_multiband_grid_cg_preserves_point_source_flux_and_suppresses_negative_lobes() -> None:
    from torchsky.map.field.inverse import solve_joint_multiband_grid_cg

    shape = (11, 11)
    bounds = (-0.05, 0.05, -0.05, 0.05)
    domain = FieldDomain(
        center_ra_deg=10.0,
        center_dec_deg=20.0,
        spatial_scale_deg=0.1,
        wavelength_center=20_000.0,
        wavelength_scale=5_000.0,
    )
    k_short = torch.tensor(
        [[0.02, 0.10, 0.02], [0.10, 0.52, 0.10], [0.02, 0.10, 0.02]],
        dtype=torch.float64,
    )
    k_long = torch.tensor(
        [[0.05, 0.12, 0.05], [0.12, 0.32, 0.12], [0.05, 0.12, 0.05]],
        dtype=torch.float64,
    )
    obs_short = FieldObservation(
        image=torch.zeros(shape, dtype=torch.float64),
        header={"NAXIS1": shape[1], "NAXIS2": shape[0]},
        weight=torch.full(shape, 100.0, dtype=torch.float64),
        psf_spec=PSFSpec(
            mode="kernel2d", kernel2d=k_short, pixel_response_included=True
        ),
        detector_to_sky=_DetectorToSky(
            center_ra_deg=domain.center_ra_deg,
            center_dec_deg=domain.center_dec_deg,
            bounds=bounds,
            shape=shape,
        ),
        band_name="F090W",
        value_semantics="surface_brightness",
    )
    obs_long = FieldObservation(
        image=torch.zeros(shape, dtype=torch.float64),
        header={"NAXIS1": shape[1], "NAXIS2": shape[0]},
        weight=torch.full(shape, 100.0, dtype=torch.float64),
        psf_spec=PSFSpec(
            mode="kernel2d", kernel2d=k_long, pixel_response_included=True
        ),
        detector_to_sky=_DetectorToSky(
            center_ra_deg=domain.center_ra_deg,
            center_dec_deg=domain.center_dec_deg,
            bounds=bounds,
            shape=shape,
        ),
        band_name="F277W",
        value_semantics="surface_brightness",
    )
    op_short = bind_linear_band_grid_observations_to_operator_set(
        (obs_short,),
        domain,
        spatial_resolution=11,
        spatial_bounds_local_deg=bounds,
    )
    op_long = bind_linear_band_grid_observations_to_operator_set(
        (obs_long,),
        domain,
        spatial_resolution=11,
        spatial_bounds_local_deg=bounds,
    )
    truth = torch.zeros(shape, dtype=torch.float64)
    truth[5, 5] = 10.0
    data_short = (op_short.operators[0].apply(truth),)
    data_long = (op_long.operators[0].apply(truth),)

    linear_long = solve_linear_band_grid_cg(
        op_long,
        data_long,
        iterations=64,
        damping=5.0e-3,
        smoothness_weight=2.0e-2,
        relative_regularization=True,
        irls_steps=1,
        edge_preserving=False,
        non_negative_weight=0.0,
        tolerance=1.0e-10,
    ).solution

    diags: dict[str, dict[str, object]] = {}
    joint_res = solve_joint_multiband_grid_cg(
        {"F090W": op_short, "F277W": op_long},
        {"F090W": data_short, "F277W": data_long},
        iterations=64,
        irls_steps=3,
        damping=1.0e-3,
        smoothness_weight=2.0e-3,
        relative_regularization=True,
        edge_preserving=True,
        non_negative_weight=0.25,
        tolerance=1.0e-10,
        diagnostics_by_band=diags,
    )
    irls_long = joint_res["F277W"].solution

    def _neg_lobe_frac(img: torch.Tensor) -> float:
        neg = float(torch.clamp(-img, min=0.0).sum().item())
        pos = float(torch.clamp(img, min=0.0).sum().item())
        return neg / max(pos, 1.0e-30)

    assert float(irls_long[5, 5].item()) > float(linear_long[5, 5].item())
    assert _neg_lobe_frac(irls_long) < _neg_lobe_frac(linear_long)
    assert abs(float(irls_long.sum().item()) - 10.0) < abs(
        float(linear_long.sum().item()) - 10.0
    )
    assert diags["F277W"]["solver"] == "joint_multiband_irls_cg"
    assert diags["F277W"]["irls_steps"] == 3


def test_linear_band_grid_heterogeneous_crop_batching_and_poisson_sensitivity() -> None:
    shape_full = (11, 11)
    shape_crop = (9, 10)
    bounds_full = (-0.05, 0.05, -0.05, 0.05)
    bounds_crop = (-0.05, 0.04, -0.05, 0.03)
    domain = FieldDomain(
        center_ra_deg=10.0,
        center_dec_deg=20.0,
        spatial_scale_deg=0.1,
        wavelength_center=20_000.0,
        wavelength_scale=5_000.0,
    )
    kernel = torch.tensor(
        [[0.02, 0.10, 0.02], [0.10, 0.52, 0.10], [0.02, 0.10, 0.02]],
        dtype=torch.float64,
    )
    truth = torch.full(shape_full, 0.2, dtype=torch.float64)
    truth[5, 5] = 200.0

    # Poisson-dominated inverse-variance weight: ~1000x smaller weight at the bright star core
    conv_truth = torch.nn.functional.conv2d(
        truth[None, None], kernel[None, None], padding=1
    )[0, 0]
    weight_full = 1.0 / (0.01 + 0.096 * (conv_truth - 0.2).clamp(min=0.0))

    obs_a = FieldObservation(
        image=torch.zeros(shape_full, dtype=torch.float64),
        header={"NAXIS1": shape_full[1], "NAXIS2": shape_full[0]},
        weight=weight_full,
        psf_spec=PSFSpec(
            mode="kernel2d", kernel2d=kernel, pixel_response_included=True
        ),
        detector_to_sky=_DetectorToSky(
            center_ra_deg=domain.center_ra_deg,
            center_dec_deg=domain.center_dec_deg,
            bounds=bounds_full,
            shape=shape_full,
        ),
        band_name="F150W",
        value_semantics="surface_brightness",
    )
    obs_b = FieldObservation(
        image=torch.zeros(shape_crop, dtype=torch.float64),
        header={"NAXIS1": shape_crop[1], "NAXIS2": shape_crop[0]},
        weight=weight_full[: shape_crop[0], : shape_crop[1]],
        psf_spec=PSFSpec(
            mode="kernel2d", kernel2d=kernel, pixel_response_included=True
        ),
        detector_to_sky=_DetectorToSky(
            center_ra_deg=domain.center_ra_deg,
            center_dec_deg=domain.center_dec_deg,
            bounds=bounds_crop,
            shape=shape_crop,
        ),
        band_name="F150W",
        value_semantics="surface_brightness",
    )
    op_set = bind_linear_band_grid_observations_to_operator_set(
        (obs_a, obs_b),
        domain,
        spatial_resolution=11,
        spatial_bounds_local_deg=bounds_full,
        exposure_chunk_size=4,
    )
    preds = op_set.apply(truth)
    for idx, op in enumerate(op_set.operators):
        torch.testing.assert_close(preds[idx], op.apply(truth), rtol=1e-12, atol=1e-12)

    expected_wadj = torch.zeros_like(truth)
    for op, p in zip(op_set.operators, preds, strict=True):
        expected_wadj = expected_wadj + op.adjoint(op.apply_noise_model(p))
    torch.testing.assert_close(
        op_set.weighted_adjoint(preds), expected_wadj, rtol=1e-12, atol=1e-12
    )
    expected_norm = torch.zeros_like(truth)
    for op in op_set.operators:
        expected_norm = expected_norm + op.normal_operator_action(truth)
    torch.testing.assert_close(
        op_set.normal_operator_action(truth), expected_norm, rtol=1e-12, atol=1e-12
    )

    solved_unweighted = solve_linear_band_grid_cg(
        op_set,
        preds,
        iterations=64,
        damping=5.0e-3 * 200.0,
        smoothness_weight=2.0e-2 * 200.0,
        relative_regularization=False,
        prior_reference="zero",
        tolerance=1.0e-10,
    ).solution
    solved_rel = solve_linear_band_grid_cg(
        op_set,
        preds,
        iterations=64,
        damping=5.0e-3,
        smoothness_weight=2.0e-2,
        relative_regularization=True,
        prior_reference="zero",
        tolerance=1.0e-10,
    ).solution
    assert float(solved_rel[5, 5].item()) > 25.0 * float(solved_unweighted[5, 5].item())
    assert float(solved_rel[5, 5].item()) > 100.0

    solved_irls = solve_linear_band_grid_cg(
        op_set,
        preds,
        iterations=64,
        damping=5.0e-3,
        smoothness_weight=2.0e-2,
        relative_regularization=True,
        prior_reference="zero",
        irls_steps=3,
        edge_preserving=True,
        tolerance=1.0e-10,
    ).solution
    assert float(solved_irls[5, 5].item()) > 180.0
    assert (
        abs(float(solved_irls.sum().item()) - float(truth.sum().item()))
        / float(truth.sum().item())
        < 0.02
    )


def test_solve_linear_band_grid_cg_detector_irls_rejects_cosmic_ray_outlier() -> None:
    shape = (11, 11)
    domain = FieldDomain(
        center_ra_deg=10.0,
        center_dec_deg=20.0,
        spatial_scale_deg=0.1,
        wavelength_center=20_000.0,
        wavelength_scale=5_000.0,
    )
    bounds = (-0.05, 0.05, -0.05, 0.05)
    weight = torch.full(shape, 100.0, dtype=torch.float64)
    truth = torch.full(shape, 1.0, dtype=torch.float64)
    truth[5, 5] = 20.0

    obs_list = tuple(
        FieldObservation(
            image=truth.clone(),
            header={"NAXIS1": shape[1], "NAXIS2": shape[0]},
            weight=weight.clone(),
            detector_to_sky=_DetectorToSky(
                center_ra_deg=domain.center_ra_deg,
                center_dec_deg=domain.center_dec_deg,
                bounds=bounds,
                shape=shape,
            ),
            band_name="F150W",
            value_semantics="surface_brightness",
        )
        for _ in range(4)
    )
    op_set = bind_linear_band_grid_observations_to_operator_set(
        obs_list,
        domain,
        spatial_resolution=11,
        spatial_bounds_local_deg=bounds,
    )
    clean_preds = list(op_set.apply(truth))
    corrupted_preds = [p.clone() for p in clean_preds]
    # Inject a massive cosmic ray (500 MJy/sr = 5000 sigma) at pixel (2, 8) in exposure 0 only
    corrupted_preds[0][2, 8] = 501.0

    l2_sol = solve_linear_band_grid_cg(
        op_set,
        corrupted_preds,
        iterations=64,
        damping=1.0e-3,
        smoothness_weight=2.0e-3,
        relative_regularization=True,
        prior_reference="zero",
        irls_steps=1,
        data_weight_fn="none",
        tolerance=1.0e-10,
    ).solution

    tukey_diag: dict[str, object] = {}
    tukey_sol = solve_linear_band_grid_cg(
        op_set,
        corrupted_preds,
        iterations=96,
        damping=1.0e-3,
        smoothness_weight=2.0e-3,
        relative_regularization=True,
        prior_reference="zero",
        irls_steps=4,
        edge_preserving=True,
        data_weight_fn="tukey",
        irls_config=IRLSConfig(weight_fn="tukey", use_mad_scaling=True),
        tolerance=1.0e-10,
        diagnostics=tukey_diag,
    ).solution

    power_sol = solve_joint_multiband_grid_cg(
        {"F150W": op_set},
        {"F150W": corrupted_preds},
        iterations=96,
        irls_steps=4,
        damping=1.0e-3,
        smoothness_weight=2.0e-3,
        relative_regularization=True,
        prior_reference="zero",
        edge_preserving=True,
        data_weight_fn="power",
        tolerance=1.0e-10,
    )["F150W"].solution

    # L2 is pulled to ~126 MJy/sr at (2, 8), whereas Tukey and Student-t power IRLS recover ~1.0 MJy/sr
    assert float(l2_sol[2, 8].item()) > 100.0
    assert abs(float(tukey_sol[2, 8].item()) - 1.0) < 0.05
    assert abs(float(power_sol[2, 8].item()) - 1.0) < 0.25
    assert abs(float(tukey_sol[5, 5].item()) - 20.0) < 0.25
    assert tukey_diag["data_weight_fn"] == "tukey"
    assert float(tukey_diag["detector_outlier_fraction"]) > 0.0

