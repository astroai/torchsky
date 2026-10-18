from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from torchsky.map.evidence import EvidenceSnapshot
from torchsky.map.likelihood import GaussianLikelihoodFactor, GaussianLikelihoodSystem
from torchsky.map.operators import ObservationOperatorBundle
from torchsky.map.query import fit_linear_query
from torchsky.map.scenes import DiscreteTargetImage
from torchsky.map.solvers import BoundObservationOperator
from torchsky.map.target import TargetRenderSpec


def _model(*, device="cpu", dtype=torch.float64, revision=None):
    response = torch.tensor([[1.0, 0.4], [0.2, 1.0]], dtype=dtype, device=device)
    precision = torch.tensor([2.0, 0.7], dtype=dtype, device=device)
    calibration = torch.ones((), dtype=dtype, device=device)
    support = torch.ones(2, dtype=dtype, device=device)
    data = torch.tensor([1.2, -0.3], dtype=dtype, device=device)

    def forward(scene, target):
        assert isinstance(scene, DiscreteTargetImage)
        return calibration * (response @ scene.data.flatten())

    def adjoint(value):
        return (calibration * (response.T @ value)).reshape(1, 2)

    operator = BoundObservationOperator(
        ObservationOperatorBundle(),
        TargetRenderSpec(header={}, shape=(1, 2)),
        apply_operator=forward,
        apply_adjoint=adjoint,
        apply_noise=lambda value: precision * support * value,
    )
    tensors = dict(
        response=response, precision=precision, calibration=calibration, support=support
    )
    dependencies = EvidenceSnapshot.capture(tensors, revision=revision)
    factor = GaussianLikelihoodFactor(
        operator, data, ("native-a",), "cal-v1", "raw-v1", dependencies=dependencies
    )
    return GaussianLikelihoodSystem((factor,)), tensors


@pytest.mark.parametrize(
    "changed", ["data", "response", "precision", "calibration", "support"]
)
def test_changed_physics_blocks_factor_system_and_cached_evidence(changed):
    system, tensors = _model()
    state = torch.tensor([[0.4, -0.2]], dtype=torch.float64)
    equations = system.normal_equations()
    held_information = equations.information
    query = fit_linear_query(system, (torch.eye(2, dtype=torch.float64),))
    tensor = system.factors[0].data if changed == "data" else tensors[changed]
    tensor.add_(0.1)
    accesses = [
        lambda: system.factors[0].residual(state),
        lambda: system.factors[0].precision(tensor),
        system.rhs,
        lambda: system.information(state),
        lambda: system.quadratic(state),
        system.normal_equations,
        system.solve,
        lambda: system.functional_uncertainty(state),
        lambda: equations.rhs,
        lambda: equations.data_quadratic,
        lambda: equations.information(state),
        lambda: held_information(state),
        lambda: equations.quadratic(state),
        lambda: query.sampling_covariance,
        lambda: query.coefficients,
        lambda: fit_linear_query(system, (torch.eye(2, dtype=torch.float64),)),
    ]
    for access in accesses:
        with pytest.raises(RuntimeError, match="stale evidence"):
            access()


def test_default_data_tracking_and_operator_rebinding_require_rebuild():
    system, _ = _model()
    # Dependency omission still guards native data, but cannot inspect closures.
    factor = replace(system.factors[0], dependencies=None)
    plain = GaussianLikelihoodSystem((factor,))
    old = plain.normal_equations()
    factor.data.mul_(2)
    with pytest.raises(RuntimeError, match="data"):
        _ = old.rhs
    rebuilt = GaussianLikelihoodSystem((replace(factor),))
    torch.testing.assert_close(rebuilt.normal_equations().rhs, rebuilt.rhs())
    factor = rebuilt.factors[0]
    old = rebuilt.normal_equations()
    factor.operator.apply_noise = lambda value: value
    with pytest.raises(RuntimeError, match="operator binding"):
        _ = old.data_quadratic


def test_default_normalization_tracking_blocks_cached_density_constants():
    system, _ = _model()
    normalization = torch.tensor(1.0, dtype=torch.float64)
    factor = replace(system.factors[0], log_normalization=normalization)
    system = GaussianLikelihoodSystem((factor,))
    equations = system.normal_equations()
    normalization.add_(1)
    with pytest.raises(RuntimeError, match="log_normalization"):
        _ = equations.log_normalization
    with pytest.raises(RuntimeError, match="log_normalization"):
        system.negative_log_likelihood(torch.zeros(1, 2, dtype=torch.float64))


@pytest.mark.parametrize("changed", ["template", "offset", "penalty"])
def test_query_source_mutation_invalidates_cached_covariance(changed):
    system, _ = _model()
    template = torch.eye(2, dtype=torch.float64)
    offset = torch.zeros(2, dtype=torch.float64)
    penalty = torch.eye(2, dtype=torch.float64) * 0.2
    query = fit_linear_query(
        system, (template,), offset_predictions=(offset,), penalty_precision=penalty
    )
    dict(template=template, offset=offset, penalty=penalty)[changed].add_(0.1)
    with pytest.raises(RuntimeError, match="stale evidence"):
        _ = query.sampling_covariance
    fresh = fit_linear_query(
        system, (template,), offset_predictions=(offset,), penalty_precision=penalty
    )
    assert torch.isfinite(fresh.sampling_covariance).all()


def test_revision_invalidates_old_stats_and_rebuilding_preserves_quadratic():
    revision = {"model": ("cal-v1", "gauge-a", 0)}
    system, tensors = _model(revision=lambda: revision["model"])
    old = system.normal_equations()
    revision["model"] = ("cal-v2", "gauge-b", 1)
    with pytest.raises(RuntimeError, match="revision"):
        _ = old.rhs
    # An old dependency declaration cannot be reused to disguise a revision.
    with pytest.raises(RuntimeError, match="revision"):
        replace(system.factors[0])
    dependencies = EvidenceSnapshot.capture(tensors, revision=lambda: revision["model"])
    fresh = GaussianLikelihoodSystem(
        (replace(system.factors[0], dependencies=dependencies),)
    )
    state = torch.tensor([[0.1, 0.2]], dtype=torch.float64)
    torch.testing.assert_close(
        fresh.quadratic(state), fresh.normal_equations().quadratic(state)
    )


@pytest.mark.parametrize("token", [[], {"revision": 0}, torch.tensor(0)])
def test_mutable_revision_tokens_fail_closed(token):
    with pytest.raises(TypeError, match="immutable"):
        EvidenceSnapshot.capture(revision=lambda: token)


def test_none_metadata_in_revision_tokens_remains_immutable_and_guarded():
    revision = [("fixed-band", None, (None, 0))]
    snapshot = EvidenceSnapshot.capture(revision=lambda: revision[0])
    snapshot.validate()
    revision[0] = ("fixed-band", None, (None, 1))
    with pytest.raises(RuntimeError, match="revision"):
        snapshot.validate()
    EvidenceSnapshot.capture(revision=lambda: None).validate()


def test_inference_tensor_requires_explicit_revision_and_bump():
    with torch.inference_mode():
        with pytest.raises(ValueError, match="explicit revision"):
            _model()
        revision = [0]
        system, _ = _model(revision=lambda: revision[0])
        equations = system.normal_equations()
        query_deps = EvidenceSnapshot.capture(revision=lambda: revision[0])
        query = fit_linear_query(
            system, (torch.eye(2, dtype=torch.float64),), dependencies=query_deps
        )
        assert torch.isfinite(equations.rhs).all()
        system.factors[0].data.add_(1)
        revision[0] += 1
        with pytest.raises(RuntimeError, match="revision"):
            _ = equations.data_quadratic
        with pytest.raises(RuntimeError, match="revision"):
            _ = query.sampling_covariance


def test_unchanged_versions_preserve_repeated_autograd_and_independent_dense_math():
    system, tensors = _model()
    data = system.factors[0].data.requires_grad_()
    # Setting requires_grad changes no tensor content/version.
    equations = system.normal_equations()
    state = torch.tensor([[0.3, -0.4]], dtype=torch.float64, requires_grad=True)
    response = tensors["response"]
    weight = torch.diag(tensors["precision"])
    h = response.T @ weight @ response
    b = response.T @ weight @ data
    c = data @ weight @ data
    torch.testing.assert_close(equations.rhs.flatten(), b)
    torch.testing.assert_close(equations.data_quadratic, c)
    for _ in range(2):
        quadratic = equations.quadratic(state)
        torch.testing.assert_close(
            quadratic,
            state.flatten() @ h @ state.flatten() - 2 * state.flatten() @ b + c,
        )
        grad = torch.autograd.grad(quadratic, state, retain_graph=True)[0]
        torch.testing.assert_close(grad.flatten(), 2 * (h @ state.flatten() - b))
    query = fit_linear_query(system, (torch.eye(2, dtype=torch.float64),))
    torch.testing.assert_close(query.coefficients, data)
    torch.testing.assert_close(
        query.sampling_covariance, torch.diag(1 / tensors["precision"])
    )
    torch.testing.assert_close(
        torch.autograd.grad(query.coefficients.sum(), data)[0], torch.ones_like(data)
    )


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_version_tracking_device_contract_without_tensor_transfer(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    dtype = torch.float32 if device == "mps" else torch.float64
    system, tensors = _model(device=device, dtype=dtype)
    query = fit_linear_query(system, (torch.eye(2, dtype=dtype, device=device),))
    assert query.sampling_covariance.device.type == device
    tensors["precision"][0] = 1
    with pytest.raises(RuntimeError, match="precision"):
        _ = query.sampling_covariance


def test_shared_view_mutation_and_mutating_callback_are_detected():
    system, tensors = _model()
    tensors["response"].view(-1)[0].add_(1)
    with pytest.raises(RuntimeError, match="response"):
        system.rhs()
    system, _ = _model()
    factor = system.factors[0]

    def mutating_precision(value):
        factor.data.add_(1)
        return value

    factor.operator.apply_noise = mutating_precision
    factor = replace(factor)
    with pytest.raises(RuntimeError, match="data"):
        factor.precision(torch.ones_like(factor.data))


@pytest.mark.parametrize(
    "changed", ["data", "response", "precision", "calibration", "support", "functional"]
)
def test_functional_uncertainty_cached_outputs_require_unchanged_model_and_direction(
    changed,
):
    system, tensors = _model()
    direction = torch.tensor([[0.2, -0.3]], dtype=torch.float64)
    uncertainty = system.functional_uncertainty(direction, iterations=8, rtol=1e-12)
    tensor = (
        direction
        if changed == "functional"
        else system.factors[0].data
        if changed == "data"
        else tensors[changed]
    )
    tensor.add_(0.1)
    for field in (
        "sampling_variance",
        "mean_response",
        "solve_residual_norm",
        "damping",
    ):
        with pytest.raises(RuntimeError, match="stale evidence"):
            getattr(uncertainty, field)
    if changed == "functional":
        fresh = system.functional_uncertainty(direction, iterations=8, rtol=1e-12)
        response = tensors["response"]
        information = response.T @ torch.diag(tensors["precision"]) @ response
        reference = direction.flatten() @ torch.linalg.solve(
            information, direction.flatten()
        )
        torch.testing.assert_close(fresh.sampling_variance, reference)


def test_functional_uncertainty_explicit_direction_revision_and_inference_requirement():
    system, _ = _model()
    revision = [0]
    with torch.inference_mode():
        direction = torch.tensor([[0.2, -0.3]], dtype=torch.float64)
    with pytest.raises(ValueError, match="explicit revision"):
        system.functional_uncertainty(direction)
    uncertainty = system.functional_uncertainty(
        direction,
        iterations=8,
        rtol=1e-12,
        dependencies=EvidenceSnapshot.capture(revision=lambda: revision[0]),
    )
    assert torch.isfinite(uncertainty.sampling_variance)
    with torch.inference_mode():
        direction.add_(0.1)
    revision[0] += 1
    with pytest.raises(RuntimeError, match="revision"):
        _ = uncertainty.sampling_variance
