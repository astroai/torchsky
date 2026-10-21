from __future__ import annotations

import math

import pytest
import torch

from torchsky.wcs import (
    AstrometricResidualFeatures,
    AstrometricResidualLadder,
    BayesianModeState,
    CorrectionEnvelope,
    ExperimentalConditionalResidual,
    ResidualComponentInput,
    ResidualModeDesign,
    ResidualSupport,
    TangentPlaneGauge,
    VersionedResidualState,
    fit_bayesian_modes,
    fit_sparse_gp_residual,
    interpretable_residual_modes,
    predict_bayesian_modes,
    shadow_update_bayesian_modes,
    update_bayesian_modes,
)


DTYPE = torch.float64
DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")
if torch.backends.mps.is_available():
    DEVICES.append("mps")


def _affine_gauge(
    count: int, *, dtype: torch.dtype = DTYPE, device: str = "cpu"
) -> TangentPlaneGauge:
    x = torch.linspace(-1.0, 1.0, count, dtype=dtype, device=device)
    y = torch.cos(torch.linspace(0.0, 2.0, count, dtype=dtype, device=device))
    design = torch.zeros((count, 2, 6), dtype=dtype, device=device)
    design[:, 0, 0] = 1.0
    design[:, 1, 1] = 1.0
    design[:, 0, 2] = x
    design[:, 0, 3] = y
    design[:, 1, 4] = x
    design[:, 1, 5] = y
    precision = torch.tensor(
        [[2.0, 0.3], [0.3, 1.4]], dtype=dtype, device=device
    ).expand(count, 2, 2)
    return TangentPlaneGauge(
        design, precision, rank_rtol=1.0e-6 if dtype == torch.float32 else 1.0e-10
    )


def _random_design(
    count: int, widths: tuple[int, int, int] = (3, 2, 2)
) -> ResidualModeDesign:
    generator = torch.Generator().manual_seed(31)
    tensors = tuple(
        torch.randn((count, 2, width), generator=generator, dtype=DTYPE)
        for width in widths
    )
    return ResidualModeDesign(*tensors)


def _vector_kernel(
    left: torch.Tensor,
    right: torch.Tensor,
    length: torch.Tensor,
    amplitude: float,
    output: torch.Tensor,
) -> torch.Tensor:
    delta = (left[:, None] - right[None]) / length
    scalar = amplitude**2 * torch.exp(-0.5 * delta.square().sum(-1))
    return (
        (scalar[..., None, None] * output)
        .permute(0, 2, 1, 3)
        .reshape(2 * left.shape[0], 2 * right.shape[0])
    )


def test_exact_weighted_gauge_projection_null_parameterization_and_covariance() -> None:
    count = 12
    gauge = _affine_gauge(count)
    generator = torch.Generator().manual_seed(7)
    value = torch.randn((count, 2), generator=generator, dtype=DTYPE)
    projected = gauge.project(value)
    torch.testing.assert_close(
        gauge.gauge_residual(projected),
        torch.zeros(gauge.dimension, dtype=DTYPE),
        atol=2.0e-13,
        rtol=0,
    )
    torch.testing.assert_close(
        gauge.project(projected), projected, atol=2.0e-13, rtol=0
    )

    basis = gauge.null_basis()
    weighted_design = (gauge.precision @ gauge.design).reshape(
        2 * count, gauge.dimension
    )
    torch.testing.assert_close(
        weighted_design.mT @ basis,
        torch.zeros((gauge.dimension, basis.shape[1]), dtype=DTYPE),
        atol=2.0e-13,
        rtol=0,
    )
    coordinates = gauge.coordinates(projected)
    torch.testing.assert_close(
        gauge.from_coordinates(coordinates), projected, atol=3.0e-13, rtol=0
    )

    covariance = torch.eye(2, dtype=DTYPE).expand(count, 2, 2).clone()
    covariance[:, 0, 1] = 0.2
    covariance[:, 1, 0] = 0.2
    projected_covariance = gauge.project_covariance(covariance)
    precision = torch.zeros((2 * count, 2 * count), dtype=DTYPE)
    rows = torch.arange(count) * 2
    precision[rows, rows] = gauge.precision[:, 0, 0]
    precision[rows, rows + 1] = gauge.precision[:, 0, 1]
    precision[rows + 1, rows] = gauge.precision[:, 1, 0]
    precision[rows + 1, rows + 1] = gauge.precision[:, 1, 1]
    torch.testing.assert_close(
        gauge.design.reshape(2 * count, -1).mT @ precision @ projected_covariance,
        torch.zeros((gauge.dimension, 2 * count), dtype=DTYPE),
        atol=3.0e-13,
        rtol=0,
    )


def test_bayesian_modes_recover_projected_signal_and_keep_terms_separate() -> None:
    count = 80
    gauge = _affine_gauge(count)
    raw_design = _random_design(count)
    design = ResidualModeDesign(
        gauge.project_design(raw_design.stable),
        gauge.project_design(raw_design.drift),
        gauge.project_design(raw_design.local),
    )
    coefficient = torch.tensor(
        [0.08, -0.04, 0.03, 0.02, -0.05, 0.07, -0.01], dtype=DTYPE
    )
    residual = torch.einsum("nak,k->na", design.matrix, coefficient)
    noise = torch.eye(2, dtype=DTYPE).expand(count, 2, 2) * 1.0e-6
    state = fit_bayesian_modes(
        design, residual, noise, prior_scale=(10.0, 10.0, 10.0), gauge=gauge
    )
    prediction = predict_bayesian_modes(state, design)
    torch.testing.assert_close(prediction.mean, residual, atol=2.0e-7, rtol=0)
    torch.testing.assert_close(
        prediction.mean,
        prediction.stable + prediction.drift + prediction.local,
    )
    torch.testing.assert_close(
        gauge.gauge_residual(prediction.mean),
        torch.zeros(gauge.dimension, dtype=DTYPE),
        atol=2.0e-12,
        rtol=0,
    )


def test_interpretable_modes_use_continuous_features_without_identifier_embeddings() -> (
    None
):
    count = 10
    focal = torch.stack(
        (torch.linspace(-1.0, 1.0, count), torch.linspace(1.0, -0.5, count)), -1
    ).to(DTYPE)
    features = AstrometricResidualFeatures(
        focal,
        focal * 0.2,
        torch.eye(2, dtype=DTYPE)[torch.arange(count) % 2],
        torch.linspace(-0.4, 1.1, count, dtype=DTYPE),
        torch.linspace(0.0, 3.0, count, dtype=DTYPE),
        torch.linspace(1.0, 1.8, count, dtype=DTYPE),
        torch.linspace(-math.pi, math.pi, count, dtype=DTYPE),
        torch.stack(
            (
                torch.linspace(-1.0, 1.0, count, dtype=DTYPE),
                torch.ones(count, dtype=DTYPE),
            ),
            -1,
        ),
    )
    modes = interpretable_residual_modes(features)
    assert modes.widths == (6, 18, 4)
    assert all("detector" in name for name in modes.stable_names)
    assert all("atmosphere" in name for name in modes.local_names)
    assert features.tensor().shape == (count, 13)


def test_online_component_update_and_shadow_anchor_no_forgetting() -> None:
    count = 36
    design = _random_design(count)
    noise = torch.eye(2, dtype=DTYPE).expand(count, 2, 2) * 0.01
    initial = BayesianModeState(
        torch.zeros(sum(design.widths), dtype=DTYPE),
        torch.eye(sum(design.widths), dtype=DTYPE),
        design.widths,
    )
    local_slice = initial.component_slice("local")
    truth = torch.tensor([0.2, -0.15], dtype=DTYPE)
    residual = torch.einsum("nak,k->na", design.local, truth)
    updated = update_bayesian_modes(initial, design, residual, noise, component="local")
    torch.testing.assert_close(
        updated.mean[initial.component_slice("stable")],
        initial.mean[initial.component_slice("stable")],
    )
    torch.testing.assert_close(updated.mean[local_slice], truth, atol=5.0e-3, rtol=0)
    assert updated.version == initial.version + 1

    empty_features = torch.empty((0, 3), dtype=DTYPE)
    empty_residuals = torch.empty((0, 2), dtype=DTYPE)
    empty_covariance = torch.empty((0, 2, 2), dtype=DTYPE)
    active = VersionedResidualState(
        4,
        3,
        "features-v2",
        "affine-gauge-v1",
        initial,
        empty_features,
        empty_residuals,
        empty_covariance,
        torch.randn(
            (count, 3), generator=torch.Generator().manual_seed(2), dtype=DTYPE
        ),
        residual,
        noise,
    )
    shadow = shadow_update_bayesian_modes(
        active,
        design,
        residual,
        noise,
        design,
        component="local",
        max_anchor_nll_increase=0.0,
    )
    assert shadow.accepted
    assert shadow.active.version == 4
    assert shadow.candidate.version == 5
    assert shadow.activate().parent_version == 4
    torch.testing.assert_close(shadow.active.coefficients.mean, initial.mean)
    rejected = shadow_update_bayesian_modes(
        active,
        design,
        -residual,
        noise,
        design,
        component="local",
        max_anchor_nll_increase=0.0,
    )
    assert not rejected.accepted
    assert rejected.reason == "anchor_no_forgetting_gate"
    with pytest.raises(RuntimeError, match="cannot be activated"):
        rejected.activate()
    replayed = active.append_replay(
        active.anchor_features,
        active.anchor_residuals,
        active.anchor_covariance,
        capacity=7,
    )
    assert replayed.replay_features.shape == (7, 3)
    assert replayed.version == active.version


def test_sparse_gp_with_training_inducing_points_matches_dense_gp_oracle() -> None:
    generator = torch.Generator().manual_seed(12)
    train = torch.randn((7, 3), generator=generator, dtype=DTYPE)
    query = torch.randn((5, 3), generator=generator, dtype=DTYPE)
    residual = torch.randn((7, 2), generator=generator, dtype=DTYPE)
    noise = torch.tensor([[0.05, 0.01], [0.01, 0.08]], dtype=DTYPE).expand(7, 2, 2)
    length = torch.tensor([0.7, 1.1, 1.5], dtype=DTYPE)
    output = torch.tensor([[1.0, 0.35], [0.35, 0.8]], dtype=DTYPE)
    amplitude = 0.6
    jitter = 1.0e-9
    support = ResidualSupport(
        torch.minimum(train.amin(0), query.amin(0)) - 0.1,
        torch.maximum(train.amax(0), query.amax(0)) + 0.1,
        20.0,
    )
    sparse = fit_sparse_gp_residual(
        train,
        residual,
        noise,
        train,
        length_scale=length,
        amplitude=amplitude,
        output_covariance=output,
        support=support,
        jitter=jitter,
    )
    prediction = sparse.predict(query)

    jitter_blocks = torch.zeros((14, 14), dtype=DTYPE)
    for index in range(train.shape[0]):
        jitter_blocks[2 * index : 2 * index + 2, 2 * index : 2 * index + 2] = (
            jitter * output
        )
    train_prior = (
        _vector_kernel(train, train, length, amplitude, output) + jitter_blocks
    )
    cross = _vector_kernel(query, train, length, amplitude, output)
    dense_noise = torch.zeros((14, 14), dtype=DTYPE)
    for index in range(7):
        dense_noise[2 * index : 2 * index + 2, 2 * index : 2 * index + 2] = noise[index]
    solve = torch.linalg.solve(train_prior + dense_noise, residual.reshape(-1))
    expected_mean = (cross @ solve).reshape(-1, 2)
    expected_full = _vector_kernel(
        query, query, length, amplitude, output
    ) - cross @ torch.linalg.solve(train_prior + dense_noise, cross.mT)
    expected_covariance = torch.stack(
        tuple(
            expected_full[2 * index : 2 * index + 2, 2 * index : 2 * index + 2]
            for index in range(query.shape[0])
        )
    )
    torch.testing.assert_close(prediction.mean, expected_mean, atol=2.0e-8, rtol=2.0e-8)
    torch.testing.assert_close(
        prediction.covariance, expected_covariance, atol=2.0e-8, rtol=2.0e-8
    )


def test_sparse_gp_strict_ood_zero_fallback_and_covariance_coverage() -> None:
    train = torch.linspace(-1.0, 1.0, 10, dtype=DTYPE)[:, None]
    residual = torch.stack((torch.sin(train[:, 0]), torch.cos(train[:, 0])), -1)
    noise = torch.eye(2, dtype=DTYPE).expand(10, 2, 2) * 0.04
    support = ResidualSupport(
        torch.tensor([-1.1], dtype=DTYPE), torch.tensor([1.1], dtype=DTYPE), 1.5
    )
    model = fit_sparse_gp_residual(
        train,
        residual,
        noise,
        train[::2],
        length_scale=torch.tensor([0.35], dtype=DTYPE),
        support=support,
    )
    prediction = model.predict(torch.tensor([[0.1], [3.0]], dtype=DTYPE))
    assert prediction.supported.tolist() == [True, False]
    torch.testing.assert_close(prediction.mean[1], torch.zeros(2, dtype=DTYPE))
    torch.testing.assert_close(
        prediction.covariance[1], torch.zeros((2, 2), dtype=DTYPE)
    )

    calibration_query = torch.linspace(-0.8, 0.8, 8, dtype=DTYPE)[:, None]
    calibration_prediction = model.predict(calibration_query)
    calibration_factor = torch.linalg.cholesky(calibration_prediction.covariance)
    calibration_residual = (
        calibration_prediction.mean + 3.0 * calibration_factor[:, :, 0]
    )
    calibrated = model.calibrate_covariance(calibration_query, calibration_residual)
    assert calibrated.covariance_scale == pytest.approx(4.5, rel=1.0e-10)

    factor = torch.linalg.cholesky(prediction.covariance[0])
    samples = (
        torch.randn((6000, 2), generator=torch.Generator().manual_seed(6), dtype=DTYPE)
        @ factor.mT
    )
    squared = torch.linalg.vector_norm(
        torch.linalg.solve_triangular(factor, samples.mT, upper=False).mT, dim=-1
    ).square()
    coverage = float((squared <= 5.991464547).to(DTYPE).mean())
    assert 0.94 < coverage < 0.96


def test_conditional_residual_is_permutation_invariant_zero_few_shot_and_leak_safe() -> (
    None
):
    torch.manual_seed(9)
    model = ExperimentalConditionalResidual(
        5,
        hidden_dim=12,
        support_radius=0.8,
        min_context=2,
        support_lower=torch.full((5,), -2.0),
        support_upper=torch.full((5,), 2.0),
    ).to(DTYPE)
    context = torch.randn((9, 5), dtype=DTYPE) * 0.2
    residual = torch.randn((9, 2), dtype=DTYPE) * 0.05
    query = torch.randn((6, 5), dtype=DTYPE) * 0.2
    baseline = model(context, residual, query)
    permutation = torch.randperm(
        context.shape[0], generator=torch.Generator().manual_seed(3)
    )
    permuted = model(context[permutation], residual[permutation], query)
    torch.testing.assert_close(permuted.mean, baseline.mean, atol=2.0e-15, rtol=0)
    torch.testing.assert_close(
        permuted.covariance, baseline.covariance, atol=2.0e-15, rtol=0
    )

    zero_shot = model(
        torch.empty((0, 5), dtype=DTYPE),
        torch.empty((0, 2), dtype=DTYPE),
        query,
    )
    assert not bool(zero_shot.supported.any())
    torch.testing.assert_close(zero_shot.mean, torch.zeros_like(zero_shot.mean))
    far_query = query.clone()
    far_query[:, :2] += 10.0
    far = model(context, residual, far_query)
    assert not bool(far.supported.any())
    torch.testing.assert_close(far.mean, torch.zeros_like(far.mean))
    assert baseline.supported.any()
    assert bool(
        torch.linalg.eigvalsh(baseline.covariance[baseline.supported]).gt(0).all()
    )

    with pytest.raises(ValueError, match="own context"):
        model.predict(
            context,
            residual,
            query,
            context_reference_ids=torch.arange(9),
            query_reference_ids=torch.tensor([20, 21, 22, 23, 24, 3]),
        )


def test_ladder_gauge_support_envelopes_and_component_fallback() -> None:
    count = 10
    gauge = _affine_gauge(count)
    envelope = CorrectionEnvelope(0.5, 0.2, 0.3, gauge_tolerance=1.0e-10)
    ladder = AstrometricResidualLadder(gauge, envelope)
    generator = torch.Generator().manual_seed(27)
    valid_mean = torch.randn((count, 2), generator=generator, dtype=DTYPE) * 0.03
    covariance = torch.eye(2, dtype=DTYPE).expand(count, 2, 2) * 0.02
    support = torch.ones(count, dtype=torch.bool)
    support[-1] = False
    valid = ResidualComponentInput(
        "low_rank",
        valid_mean,
        covariance,
        support,
        torch.zeros((count, 2, 2), dtype=DTYPE),
    )
    rejected = ResidualComponentInput(
        "neural",
        torch.full((count, 2), 2.0, dtype=DTYPE),
        covariance,
        torch.ones(count, dtype=torch.bool),
    )
    bad_jacobian = ResidualComponentInput(
        "gp",
        valid_mean,
        covariance,
        torch.ones(count, dtype=torch.bool),
        torch.eye(2, dtype=DTYPE).expand(count, 2, 2),
    )
    prediction = ladder.predict((valid, rejected, bad_jacobian))
    assert prediction.components[0].fallback_reason is None
    assert prediction.components[1].fallback_reason == "insufficient_trusted_support"
    assert prediction.components[2].fallback_reason == "insufficient_trusted_support"
    torch.testing.assert_close(prediction.mean[-1], torch.zeros(2, dtype=DTYPE))
    torch.testing.assert_close(
        gauge.gauge_residual(prediction.mean),
        torch.zeros(gauge.dimension, dtype=DTYPE),
        atol=2.0e-13,
        rtol=0,
    )
    precision = torch.zeros((2 * count, 2 * count), dtype=DTYPE)
    rows = 2 * torch.arange(count)
    precision[rows, rows] = gauge.precision[:, 0, 0]
    precision[rows, rows + 1] = gauge.precision[:, 0, 1]
    precision[rows + 1, rows] = gauge.precision[:, 1, 0]
    precision[rows + 1, rows + 1] = gauge.precision[:, 1, 1]
    torch.testing.assert_close(
        gauge.design.reshape(2 * count, -1).mT @ precision @ prediction.full_covariance,
        torch.zeros((gauge.dimension, 2 * count), dtype=DTYPE),
        atol=3.0e-13,
        rtol=0,
    )


@pytest.mark.parametrize("device", DEVICES)
def test_residual_models_device_gradient_and_compile(device: str) -> None:
    dtype = torch.float32
    context = torch.randn((12, 6), device=device, dtype=dtype)
    residual = torch.randn((12, 2), device=device, dtype=dtype)
    query = (torch.randn((7, 6), device=device, dtype=dtype) * 0.1).requires_grad_()
    context[:, :2] *= 0.1
    model = ExperimentalConditionalResidual(
        6, hidden_dim=8, support_radius=1.0, min_context=1, query_chunk_size=4
    ).to(device=device, dtype=dtype)
    prediction = model(context, residual, query)
    loss = prediction.mean.square().sum() + prediction.covariance.sum()
    loss.backward()
    assert query.grad is not None
    assert bool(torch.isfinite(query.grad).all())
    assert all(parameter.grad is not None for parameter in model.parameters())

    count = 8
    gauge = _affine_gauge(count, dtype=dtype, device=device)
    gauge_value = torch.randn(
        (count, 2), device=device, dtype=dtype, requires_grad=True
    )
    gauge.project(gauge_value).square().sum().backward()
    assert gauge_value.grad is not None
    torch.testing.assert_close(
        gauge.gauge_residual(gauge.project(gauge_value.detach())),
        torch.zeros(gauge.dimension, device=device, dtype=dtype),
        atol=3.0e-5,
        rtol=0,
    )

    train = torch.linspace(-1.0, 1.0, count, device=device, dtype=dtype)[:, None]
    train_residual = torch.stack((train[:, 0].square(), train[:, 0].pow(3)), -1)
    gp = fit_sparse_gp_residual(
        train,
        train_residual,
        torch.eye(2, device=device, dtype=dtype).expand(count, 2, 2) * 0.1,
        train[::2],
        length_scale=torch.ones(1, device=device, dtype=dtype),
    )
    gp_query = torch.tensor(
        [[0.1], [0.3]], device=device, dtype=dtype, requires_grad=True
    )
    gp_prediction = gp.predict(gp_query)
    (gp_prediction.mean.square().sum() + gp_prediction.covariance.sum()).backward()
    assert gp_query.grad is not None
    assert bool(torch.isfinite(gp_query.grad).all())

    if device == "cpu" and hasattr(torch, "compile"):

        def compiled_chunk(
            context_features: torch.Tensor,
            context_encoding: torch.Tensor,
            query_features: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            result = model._chunk(context_features, context_encoding, query_features)
            return result.mean, result.covariance

        encoding = model.context_encoder(torch.cat((context, residual), dim=-1))
        eager = compiled_chunk(context, encoding, query.detach())
        compiled = torch.compile(compiled_chunk, backend="eager", fullgraph=True)(
            context, encoding, query.detach()
        )
        torch.testing.assert_close(compiled[0], eager[0])
        torch.testing.assert_close(compiled[1], eager[1])


def test_chunked_conditional_scaling_and_low_rank_state_size() -> None:
    class TrackingConditionalResidual(ExperimentalConditionalResidual):
        def __init__(self) -> None:
            super().__init__(4, hidden_dim=8, support_radius=1.0, query_chunk_size=17)
            self.largest_chunk = 0

        def _chunk(
            self,
            context_features: torch.Tensor,
            context_encoding: torch.Tensor,
            query_features: torch.Tensor,
        ):
            self.largest_chunk = max(self.largest_chunk, int(query_features.shape[0]))
            return super()._chunk(context_features, context_encoding, query_features)

    model = TrackingConditionalResidual()
    context = torch.randn((41, 4)) * 0.1
    residual = torch.randn((41, 2))
    query = torch.randn((233, 4)) * 0.1
    output = model(context, residual, query)
    assert output.mean.shape == (233, 2)
    assert model.largest_chunk == 17

    count = 4096
    generator = torch.Generator().manual_seed(8)
    design = ResidualModeDesign(
        torch.randn((count, 2, 2), generator=generator),
        torch.randn((count, 2, 2), generator=generator),
        torch.randn((count, 2, 2), generator=generator),
    )
    values = torch.randn((count, 2), generator=generator)
    noise = torch.eye(2).expand(count, 2, 2)
    state = fit_bayesian_modes(design, values, noise)
    assert state.covariance.shape == (6, 6)
    assert state.covariance.numel() < count
