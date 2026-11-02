from __future__ import annotations

from dataclasses import replace
import math

import pytest
import torch
from torch import Tensor

from torchsky.map.likelihood import GaussianLikelihoodFactor, GaussianLikelihoodSystem
from torchsky.map.operators import ObservationOperatorBundle
from torchsky.map.operators.nuisance import (
    CalibratedNuisancePrecision,
    ProfiledNuisancePrecision,
)
from torchsky.map.scenes import DiscreteTargetImage, SceneRepresentation
from torchsky.map.solvers import BoundObservationOperator
from torchsky.map.target import TargetRenderSpec


def _operator(matrix: Tensor, precision, *, shape=(2, 2)) -> BoundObservationOperator:
    def forward(scene: SceneRepresentation, target: TargetRenderSpec) -> Tensor:
        del target
        assert isinstance(scene, DiscreteTargetImage)
        return matrix @ scene.data.reshape(-1).to(dtype=matrix.dtype)

    def adjoint(data: Tensor) -> Tensor:
        return (matrix.mH @ data).real.reshape(shape)

    return BoundObservationOperator(
        ObservationOperatorBundle(),
        TargetRenderSpec(header={}, shape=shape),
        apply_operator=forward,
        apply_adjoint=adjoint,
        apply_noise=precision,
    )


def _factor(
    matrix: Tensor, weight: Tensor, data: Tensor, name: str = "exposure-a"
) -> GaussianLikelihoodFactor:
    def precision(value: Tensor) -> Tensor:
        return weight @ value

    return GaussianLikelihoodFactor(
        _operator(matrix, precision), data, (name,), "calibration-v1", "native-v1"
    )


def test_full_normal_quadratic_preserves_native_likelihood_and_gradients() -> None:
    matrix = torch.tensor(
        [
            [1.0, 0.7, 0.0, 0.0],
            [0.0, 1.0, 0.2, 0.0],
            [0.0, 0.0, 1.0, 0.6],
            [0.3, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    weight = torch.diag(torch.tensor([2.0, 1.0, 3.0, 0.5, 1.0], dtype=torch.float64))
    data = torch.tensor([2.0, -1.0, 0.8, 0.2, 0.6], dtype=torch.float64)
    system = GaussianLikelihoodSystem((_factor(matrix, weight, data),))
    state = torch.tensor(
        [[0.4, 0.3], [-0.2, 0.7]], dtype=torch.float64, requires_grad=True
    )
    h, b = matrix.T @ weight @ matrix, matrix.T @ weight @ data
    equations = system.normal_equations()
    torch.testing.assert_close(equations.rhs.flatten(), b)
    torch.testing.assert_close(
        equations.information(state).flatten(), h @ state.flatten()
    )
    direct = (
        (data - matrix @ state.flatten()) @ weight @ (data - matrix @ state.flatten())
    )
    torch.testing.assert_close(system.quadratic(state), direct)
    torch.testing.assert_close(equations.quadratic(state), direct)
    gradient = torch.autograd.grad(equations.quadratic(state), state)[0]
    torch.testing.assert_close(gradient.flatten(), 2 * (h @ state.flatten() - b))
    solution = system.solve(iterations=12).solution.flatten()
    torch.testing.assert_close(
        solution, torch.linalg.solve(h, b), atol=1e-11, rtol=1e-11
    )
    diagonal_estimate = b / h.diagonal()
    assert not torch.allclose(diagonal_estimate, solution)


def test_density_requires_normalization_and_matches_independent_gaussian() -> None:
    matrix = torch.eye(4, dtype=torch.float64)
    covariance = torch.diag(torch.tensor([0.7, 1.2, 2.0, 0.8], dtype=torch.float64))
    factor = _factor(
        matrix,
        torch.linalg.inv(covariance),
        torch.tensor([1.0, 2.0, -1.0, 0.5], dtype=torch.float64),
    )
    state = torch.tensor([[0.3, 0.5], [0.2, -0.3]], dtype=torch.float64)
    missing = GaussianLikelihoodSystem((factor,))
    with pytest.raises(ValueError, match="density constants"):
        missing.negative_log_likelihood(state)
    with pytest.raises(ValueError, match="density constants"):
        missing.normal_equations().negative_log_likelihood(state)
    normalization = 0.5 * (4 * math.log(2 * math.pi) + torch.logdet(covariance))
    system = GaussianLikelihoodSystem(
        (replace(factor, log_normalization=normalization),)
    )
    oracle = -torch.distributions.MultivariateNormal(
        state.flatten(), covariance
    ).log_prob(factor.data)
    torch.testing.assert_close(system.negative_log_likelihood(state), oracle)
    torch.testing.assert_close(
        system.normal_equations().negative_log_likelihood(state), oracle
    )
    profiled = replace(
        factor,
        nuisance_treatment="profiled",
        eliminated_nuisance_ids=("offset",),
        log_normalization=normalization,
    )
    with pytest.raises(ValueError, match="density constants"):
        GaussianLikelihoodSystem((profiled,)).negative_log_likelihood(state)


def test_fitting_noise_scale_retains_normalization_gradient() -> None:
    log_scale = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    scale = log_scale.exp()
    matrix = torch.eye(4, dtype=torch.float64)
    data = torch.tensor([1.0, 2.0, -1.0, 0.5], dtype=torch.float64)
    factor = _factor(matrix, matrix / scale.square(), data)
    factor = replace(
        factor,
        log_normalization=4 * log_scale + data.new_tensor(2 * math.log(2 * math.pi)),
    )
    system = GaussianLikelihoodSystem((factor,))
    state = torch.zeros(2, 2, dtype=torch.float64)
    gradient = torch.autograd.grad(system.negative_log_likelihood(state), log_scale)[0]
    torch.testing.assert_close(gradient, 4 - data.square().sum() / scale.square())


@pytest.mark.parametrize("damping", [0.0, 0.7])
def test_query_uses_sampling_covariance_and_declared_mean_transfer(
    damping: float,
) -> None:
    matrix = torch.tensor(
        [
            [1.0, 0.8, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.4],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    system = GaussianLikelihoodSystem(
        (
            _factor(
                matrix,
                torch.eye(4, dtype=torch.float64),
                torch.zeros(4, dtype=torch.float64),
            ),
        )
    )
    g = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
    result = system.functional_uncertainty(g, iterations=8, damping=damping, rtol=1e-10)
    h = matrix.T @ matrix
    k = h + damping * torch.eye(4, dtype=torch.float64)
    inverse_g = torch.linalg.solve(k, g.flatten())
    torch.testing.assert_close(result.sampling_variance, inverse_g @ h @ inverse_g)
    torch.testing.assert_close(result.mean_response.flatten(), h @ inverse_g)
    assert result.solve_residual_norm < 1e-10
    if damping:
        assert not torch.allclose(result.sampling_variance, g.flatten() @ inverse_g)
    else:
        assert not torch.allclose(result.sampling_variance, 1 / h[0, 0])
        torch.testing.assert_close(result.mean_response, g)


def test_poorer_broad_observation_adds_information_without_blurring_solution() -> None:
    matrix = torch.eye(4, dtype=torch.float64)
    truth = torch.tensor([1.0, -1.0, 3.0, 2.0], dtype=torch.float64)
    broad = torch.full((1, 4), 0.25, dtype=torch.float64)
    sharp_factor = _factor(matrix, torch.eye(4, dtype=torch.float64), truth)
    broad_factor = _factor(
        broad, torch.tensor([[0.1]], dtype=torch.float64), broad @ truth, "old-broad"
    )
    sharp = GaussianLikelihoodSystem((sharp_factor,))
    combined = GaussianLikelihoodSystem(
        (sharp_factor, broad_factor), exposure_chunk_size=2
    )
    torch.testing.assert_close(combined.solve(iterations=8).solution.flatten(), truth)
    g = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
    assert (
        combined.functional_uncertainty(g).sampling_variance
        < sharp.functional_uncertainty(g).sampling_variance
    )
    h_new = 0.1 * broad.T @ broad
    assert torch.linalg.eigvalsh(h_new).min() >= -1e-14


def test_duplicate_derived_products_and_shared_elimination_fail_closed() -> None:
    factor = _factor(torch.eye(4), torch.eye(4), torch.ones(4))
    with pytest.raises(ValueError, match="duplicate native measurement"):
        GaussianLikelihoodSystem(
            (factor, replace(factor, provenance_id="other-reduction"))
        )
    mosaic = replace(factor, measurement_ids=("exposure-a", "exposure-b"))
    with pytest.raises(ValueError, match="duplicate native measurement"):
        GaussianLikelihoodSystem((factor, mosaic))
    first = replace(
        factor,
        nuisance_treatment="calibrated",
        eliminated_nuisance_ids=("shared-flat",),
    )
    second = replace(first, measurement_ids=("exposure-b",))
    with pytest.raises(ValueError, match="joint factor"):
        GaussianLikelihoodSystem((first, second))
    joint = replace(first, measurement_ids=("exposure-a", "exposure-b"))
    assert GaussianLikelihoodSystem((joint,)).measurement_ids == joint.measurement_ids


def test_calibrated_nuisance_mean_and_covariance_are_in_likelihood() -> None:
    matrix = torch.eye(4, dtype=torch.float64)
    basis = torch.ones(4, 1, dtype=torch.float64)
    mean = torch.tensor([0.3], dtype=torch.float64)
    precision = CalibratedNuisancePrecision(
        basis,
        apply_precision=lambda r: r,
        covariance_factor=torch.tensor([[0.5]], dtype=torch.float64),
        mean=mean,
    )
    data = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    factor = GaussianLikelihoodFactor(
        _operator(matrix, precision),
        precision.center(data),
        ("exposure-a",),
        "cal-v1",
        "native-v1",
        "calibrated",
        ("local-background",),
    )
    system = GaussianLikelihoodSystem((factor,))
    effective_covariance = torch.eye(4, dtype=torch.float64) + 0.25 * basis @ basis.T
    torch.testing.assert_close(
        system.rhs().flatten(), torch.linalg.solve(effective_covariance, data - 0.3)
    )
    g = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
    torch.testing.assert_close(
        system.functional_uncertainty(g).sampling_variance, effective_covariance[0, 0]
    )


def test_profiled_gauge_has_no_finite_unregularized_uncertainty() -> None:
    matrix = torch.eye(4, dtype=torch.float64)
    precision = ProfiledNuisancePrecision(
        torch.ones(4, 1, dtype=torch.float64), apply_precision=lambda r: r
    )
    factor = GaussianLikelihoodFactor(
        _operator(matrix, precision),
        torch.zeros(4, dtype=torch.float64),
        ("exposure-a",),
        "cal-v1",
        "native-v1",
        "profiled",
        ("unknown-offset",),
    )
    system = GaussianLikelihoodSystem((factor,))
    with pytest.raises(RuntimeError, match="unidentifiable"):
        system.functional_uncertainty(
            torch.ones(2, 2, dtype=torch.float64), iterations=4
        )
    g = torch.tensor([[1.0, -1.0], [0.0, 0.0]], dtype=torch.float64)
    torch.testing.assert_close(
        system.functional_uncertainty(g).sampling_variance,
        torch.tensor(2.0, dtype=torch.float64),
    )
    with pytest.raises(RuntimeError, match="did not converge"):
        GaussianLikelihoodSystem(
            (
                _factor(
                    torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)),
                    matrix,
                    torch.zeros(4, dtype=torch.float64),
                ),
            )
        ).functional_uncertainty(
            torch.ones(2, 2, dtype=torch.float64), iterations=1, rtol=1e-10
        )


def test_complex_native_vector_has_real_scene_quadratic() -> None:
    matrix = torch.tensor(
        [[1 + 1j, 0, 0.5j, 0], [0, 1 - 0.3j, 0, 1j], [0.2j, 0, 1, 0.2]],
        dtype=torch.complex128,
    )
    weight = torch.diag(torch.tensor([1.0, 2.0, 0.5], dtype=torch.complex128))
    data = torch.tensor([0.2 + 0.3j, -0.4j, 1 + 0.2j], dtype=torch.complex128)
    factor = _factor(matrix, weight, data)
    system = GaussianLikelihoodSystem((factor,))
    state = torch.tensor([[0.2, 0.5], [-0.1, 0.3]], dtype=torch.float64)
    # Real sky coordinates correspond to the augmented real/imaginary design.
    h = (matrix.mH @ weight @ matrix).real
    b = (matrix.mH @ weight @ data).real
    torch.testing.assert_close(system.rhs().flatten(), b)
    torch.testing.assert_close(system.information(state).flatten(), h @ state.flatten())
    torch.testing.assert_close(
        system.quadratic(state), system.normal_equations().quadratic(state)
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("measurement_ids", ()),
        ("measurement_ids", ("a", "a")),
        ("calibration_id", ""),
        ("provenance_id", ""),
        ("nuisance_treatment", "unknown"),
        ("nuisance_treatment", "profiled"),
    ],
)
def test_evidence_metadata_is_validated(field, value) -> None:
    factor = _factor(torch.eye(4), torch.eye(4), torch.ones(4))
    with pytest.raises((ValueError, TypeError)):
        replace(factor, **{field: value})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"iterations": True},
        {"iterations": 0},
        {"damping": float("nan")},
        {"damping": -1.0},
        {"rtol": -1.0},
        {"atol": float("inf")},
        {"rtol": 0.0, "atol": 0.0},
    ],
)
def test_query_rejects_invalid_solver_options(kwargs) -> None:
    system = GaussianLikelihoodSystem(
        (_factor(torch.eye(4), torch.eye(4), torch.ones(4)),)
    )
    with pytest.raises(ValueError):
        system.functional_uncertainty(torch.ones(2, 2), **kwargs)


def test_precision_contract_rejects_shape_dtype_and_nonfinite_outputs() -> None:
    factor = _factor(torch.eye(4), torch.eye(4), torch.ones(4))
    for output in (
        torch.ones(3),
        torch.ones(4, dtype=torch.float64),
        torch.full((4,), float("nan")),
    ):
        invalid = replace(
            factor,
            operator=replace(
                factor.operator, apply_noise=lambda r, output=output: output
            ),
        )
        with pytest.raises((ValueError, TypeError)):
            GaussianLikelihoodSystem((invalid,)).rhs()


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_available_device_likelihood_and_sampling_query(device: str) -> None:
    available = (
        torch.cuda.is_available()
        if device == "cuda"
        else torch.backends.mps.is_available()
    )
    if not available:
        pytest.skip(f"{device} unavailable")
    matrix = torch.eye(4, device=device)
    system = GaussianLikelihoodSystem(
        (_factor(matrix, matrix, torch.ones(4, device=device)),)
    )
    result = system.functional_uncertainty(torch.ones(2, 2, device=device))
    assert result.sampling_variance.device.type == device
    torch.testing.assert_close(
        result.sampling_variance, torch.tensor(4.0, device=device)
    )
