from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import solve as scipy_solve
import torch

from torchsky.map.likelihood import GaussianLikelihoodFactor, GaussianLikelihoodSystem
from torchsky.map.operators import (
    CalibratedNuisancePrecision,
    GaussianPixelResponse,
    ObservationOperatorBundle,
    ProfiledNuisancePrecision,
)
from torchsky.map.query import fit_linear_query
from torchsky.map.solvers import BoundObservationOperator
from torchsky.map.target import TargetRenderSpec


def _system(data, precision, *, name="native-a", **kwargs):
    # Query templates directly use measurement space, without calling the scene.
    def unused(*args):
        raise AssertionError("linear query must not instantiate a field")

    operator = BoundObservationOperator(
        ObservationOperatorBundle(),
        TargetRenderSpec(header={}, shape=(1, 1)),
        apply_operator=unused,
        apply_adjoint=unused,
        apply_noise=precision,
    )
    return GaussianLikelihoodSystem(
        (
            GaussianLikelihoodFactor(
                operator, data, (name,), "cal-v1", "raw-v1", **kwargs
            ),
        )
    )


def _blend(sigma=0.8, *, dtype=torch.float64, device="cpu"):
    y, x = torch.meshgrid(
        torch.arange(-4, 5, dtype=dtype, device=device),
        torch.arange(-4, 5, dtype=dtype, device=device),
        indexing="ij",
    )
    xy = torch.stack((x, y), -1)
    response = GaussianPixelResponse(torch.tensor(sigma, dtype=dtype, device=device))
    return torch.stack(
        [response(xy - xy.new_tensor(center)) for center in [(-0.6, 0.2), (0.7, -0.3)]]
    )


@pytest.mark.parametrize("penalized", [False, True])
def test_blend_matches_independent_dense_gls_and_full_sampling_covariance(penalized):
    templates = _blend()
    matrix = templates.flatten(1).T
    covariance = torch.eye(81, dtype=torch.float64) * 0.3
    covariance += torch.ones_like(covariance) * 0.07
    data = (matrix @ torch.tensor([8.0, -1.3], dtype=torch.float64)).reshape(9, 9)
    offset = torch.linspace(0.0, 0.2, 81, dtype=torch.float64).reshape(9, 9)
    precision = torch.linalg.inv(covariance)
    system = _system(
        data + offset, lambda value: (precision @ value.flatten()).reshape(9, 9)
    )
    penalty = (
        torch.diag(torch.tensor([0.2, 0.6], dtype=torch.float64)) if penalized else None
    )
    result = fit_linear_query(
        system, (templates,), offset_predictions=(offset,), penalty_precision=penalty
    )
    j, c = matrix.numpy(), covariance.numpy()
    fisher = j.T @ scipy_solve(c, j, assume_a="pos")
    g = j.T @ scipy_solve(c, data.numpy().ravel(), assume_a="pos")
    k = fisher + (penalty.numpy() if penalty is not None else np.zeros((2, 2)))
    expected = scipy_solve(k, g, assume_a="pos")
    transfer = scipy_solve(k, fisher, assume_a="pos")
    cov = scipy_solve(k, transfer.T, assume_a="pos")
    np.testing.assert_allclose(result.coefficients.numpy(), expected, atol=1e-12)
    np.testing.assert_allclose(result.information.numpy(), fisher, atol=1e-12)
    np.testing.assert_allclose(result.mean_response.numpy(), transfer, atol=1e-12)
    np.testing.assert_allclose(result.sampling_covariance.numpy(), cov, atol=1e-12)
    assert result.data_rank == 2
    if penalized:
        assert not torch.allclose(
            result.sampling_covariance, torch.linalg.inv(torch.from_numpy(k))
        )
    else:
        assert (
            result.sampling_covariance[0, 1] < 0
        )  # Unpenalized blend anticorrelation.
        torch.testing.assert_close(
            result.coefficients, torch.tensor([8.0, -1.3], dtype=torch.float64)
        )


def test_profiled_background_agrees_with_full_unpenalized_fit():
    templates = _blend()
    j = templates.flatten(1).T
    basis = torch.ones(9, 9, 1, dtype=torch.float64)
    weight = torch.linspace(0.5, 2.0, 81, dtype=torch.float64).reshape(9, 9)
    coefficients = torch.tensor([4.0, 3.0], dtype=torch.float64)
    data = (j @ coefficients).reshape(9, 9) + 0.9
    profile = ProfiledNuisancePrecision(
        basis, apply_precision=lambda value: weight * value
    )
    system = _system(
        data,
        profile.apply,
        nuisance_treatment="profiled",
        eliminated_nuisance_ids=("sky-a",),
    )
    result = fit_linear_query(system, (templates,))
    full_j = torch.cat((j, basis.reshape(-1, 1)), dim=1)
    fisher = full_j.T @ (weight.flatten()[:, None] * full_j)
    full_fit = torch.linalg.solve(
        fisher, full_j.T @ (weight.flatten() * data.flatten())
    )
    torch.testing.assert_close(result.coefficients, full_fit[:2])
    torch.testing.assert_close(
        result.sampling_covariance, torch.linalg.inv(fisher)[:2, :2]
    )
    assert result.nuisance_treatments == ("profiled",)


def test_shared_calibration_in_one_joint_factor_retains_error_floor():
    templates = torch.ones(1, 2, 7, dtype=torch.float64)
    data = torch.full((2, 7), 5.0, dtype=torch.float64)
    basis = torch.ones(2, 7, 1, dtype=torch.float64)
    calibrated = CalibratedNuisancePrecision(
        basis,
        apply_precision=lambda value: value / 0.4,
        covariance_factor=torch.tensor([[0.8]], dtype=torch.float64),
    )
    system = _system(
        data,
        calibrated.apply,
        nuisance_treatment="calibrated",
        eliminated_nuisance_ids=("shared-zero",),
    )
    result = fit_linear_query(system, (templates,))
    torch.testing.assert_close(result.coefficients, data.new_tensor([5.0]))
    torch.testing.assert_close(
        result.sampling_covariance, data.new_tensor([[0.4 / 14 + 0.8**2]])
    )


def test_poorer_native_factor_adds_information_without_blurring_templates():
    sharp, broad = _blend(0.6), _blend(2.0)
    true = torch.tensor([4.0, 2.0], dtype=torch.float64)
    a = _system(torch.einsum("j,jyx->yx", true, sharp), lambda v: v * 3)
    b = _system(
        torch.einsum("j,jyx->yx", true, broad), lambda v: v * 0.2, name="older-b"
    )
    first = fit_linear_query(a, (sharp,))
    both = fit_linear_query(
        GaussianLikelihoodSystem(a.factors + b.factors), (sharp, broad)
    )
    assert torch.linalg.eigvalsh(both.information - first.information).min() >= 0
    torch.testing.assert_close(both.coefficients, true)
    assert (
        torch.linalg.eigvalsh(
            first.sampling_covariance - both.sampling_covariance
        ).min()
        > 0
    )


def test_gauge_is_rejected_or_reported_as_penalty_dependent():
    template = torch.ones(2, 4, dtype=torch.float64)
    system = _system(torch.ones(4, dtype=torch.float64), lambda v: v)
    with pytest.raises(RuntimeError, match="unidentifiable"):
        fit_linear_query(system, (template,))
    result = fit_linear_query(
        system, (template,), penalty_precision=torch.eye(2, dtype=torch.float64)
    )
    assert result.data_rank == 1
    contrast = torch.tensor([1.0, -1.0], dtype=torch.float64)
    torch.testing.assert_close(
        result.mean_response @ contrast, contrast * 0, atol=1e-14, rtol=0
    )
    assert (contrast @ result.sampling_covariance @ contrast).abs() < 1e-14
    # Zero sampling variance in this unsupported contrast is not a confidence bound.


def test_real_coefficients_for_complex_native_measurements():
    template = torch.tensor(
        [[1 + 2j, 2 - 1j, 1j], [3 - 1j, 1j, 2 + 0j]], dtype=torch.complex128
    )
    true = torch.tensor([2.0, -0.7], dtype=torch.float64)
    data = template.T @ true.to(template.dtype)
    weights = torch.tensor([2.0, 0.4, 0.7], dtype=torch.float64)
    result = fit_linear_query(_system(data, lambda v: weights * v), (template,))
    augmented = torch.cat((template.real.T, template.imag.T))
    weight = torch.diag(torch.cat((weights, weights)))
    fisher = augmented.T @ weight @ augmented
    torch.testing.assert_close(result.information, fisher)
    torch.testing.assert_close(result.coefficients, true)
    torch.testing.assert_close(result.sampling_covariance, torch.linalg.inv(fisher))


def test_query_gradients_through_templates_and_calibrated_precision():
    data = torch.tensor([1.0, 2.0, 0.8, -0.2], dtype=torch.float64)
    template = torch.tensor(
        [[1.0, 0.2, 0.7, 0.0], [0.1, 1.0, -0.3, 0.4]],
        dtype=torch.float64,
        requires_grad=True,
    )
    amplitude = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)

    def evaluate(t, a):
        calibrated = CalibratedNuisancePrecision(
            torch.ones(4, 1, dtype=torch.float64),
            apply_precision=lambda v: v,
            covariance_factor=a.reshape(1, 1),
        )
        result = fit_linear_query(
            _system(
                data,
                calibrated.apply,
                nuisance_treatment="calibrated",
                eliminated_nuisance_ids=("zero",),
            ),
            (t,),
        )
        return result.coefficients, result.sampling_covariance

    assert torch.autograd.gradcheck(evaluate, (template, amplitude))


@pytest.mark.parametrize("kind", ["asymmetric", "negative"])
def test_invalid_information_callback_rejects(kind):
    precision = (
        torch.tensor([[1.0, 0.3], [0.0, 1.0]], dtype=torch.float64)
        if kind == "asymmetric"
        else -torch.eye(2, dtype=torch.float64)
    )
    with pytest.raises(ValueError, match="symmetric|positive semidefinite"):
        fit_linear_query(
            _system(torch.ones(2, dtype=torch.float64), lambda v: precision @ v),
            (torch.eye(2, dtype=torch.float64),),
        )


@pytest.mark.parametrize("rank_options", [{"rank_rtol": 0.1}, {"rank_atol": 0.2}])
@pytest.mark.parametrize("bad_kind", ["information", "penalty", "asymmetric"])
def test_relaxed_rank_policy_cannot_authorize_invalid_precision(rank_options, bad_kind):
    identity = torch.eye(2, dtype=torch.float64)
    negative = torch.diag(torch.tensor([1.0, -0.01], dtype=torch.float64))
    bad = torch.tensor([[1.0, 0.01], [0.0, 1.0]], dtype=torch.float64)
    precision = negative if bad_kind == "information" else identity
    penalty = (
        identity
        if bad_kind == "information"
        else negative
        if bad_kind == "penalty"
        else bad
    )
    with pytest.raises(ValueError, match="positive semidefinite|symmetric"):
        fit_linear_query(
            _system(torch.ones(2, dtype=torch.float64), lambda v: precision @ v),
            (identity,),
            penalty_precision=penalty,
            **rank_options,
        )


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf"), True])
def test_invalid_rank_options_reject(bad):
    system = _system(torch.ones(2, dtype=torch.float64), lambda v: v)
    with pytest.raises((TypeError, ValueError), match="rank_rtol"):
        fit_linear_query(system, (torch.eye(2, dtype=torch.float64),), rank_rtol=bad)


def test_invalid_templates_offsets_and_penalty_reject():
    data = torch.ones(3, dtype=torch.float64)
    system = _system(data, lambda v: v)
    template = torch.eye(3, dtype=torch.float64)
    for bad in [
        (),
        (template, template),
        (template.float(),),
        (template[:, :2],),
        (torch.empty(0, 3, dtype=torch.float64),),
        (template * float("nan"),),
    ]:
        with pytest.raises(ValueError, match="template"):
            fit_linear_query(system, bad)
    for penalty in [
        torch.eye(3),
        -template,
        torch.zeros(2, 2, dtype=torch.float64),
        template * float("inf"),
    ]:
        with pytest.raises(ValueError, match="penalty_precision"):
            fit_linear_query(system, (template,), penalty_precision=penalty)
    with pytest.raises(ValueError, match="offset_predictions"):
        fit_linear_query(system, (template,), offset_predictions=())
    with pytest.raises(ValueError, match="measurement input"):
        fit_linear_query(system, (template,), offset_predictions=(data[:1],))
    second = replace(system.factors[0], data=data.float(), measurement_ids=("b",))
    with pytest.raises(ValueError, match="real dtype"):
        fit_linear_query(
            GaussianLikelihoodSystem(system.factors + (second,)),
            (template, template.float()),
        )


@pytest.mark.parametrize(
    "device,dtype",
    [("cpu", torch.float32), ("cuda", torch.float64), ("mps", torch.float32)],
)
def test_available_device_contract(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    templates = _blend(dtype=dtype, device=device)
    true = templates.new_tensor([3.0, 0.7])
    data = torch.einsum("j,jyx->yx", true, templates)
    result = fit_linear_query(_system(data, lambda v: v), (templates,))
    assert result.coefficients.device == data.device
    assert result.sampling_covariance.dtype == data.dtype
    torch.testing.assert_close(result.coefficients, true, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_query_device_first_second_ad_matches_dense_shared_covariance(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    inputs = (
        torch.tensor([[1.0, 0.2, 0.7, 0.0], [0.1, 1.0, -0.3, 0.4]], device=device),
        torch.tensor(0.3, device=device),
        torch.tensor([1.0, 2.0, 0.8, 1.2], device=device),
        torch.tensor([1.0, 2.0, 0.8, -0.2], device=device),
    )
    inputs = tuple(x.requires_grad_() for x in inputs)

    def evaluate(args, *, reference):
        templates, amplitude, weights, data = args
        penalty = torch.diag(data.new_tensor([0.2, 0.4]))
        if reference:
            covariance = torch.diag(
                weights.reciprocal()
            ) + amplitude.square() * torch.ones(
                4, 4, dtype=data.dtype, device=data.device
            )
            information = templates @ torch.linalg.solve(covariance, templates.T)
            rhs = templates @ torch.linalg.solve(covariance, data)
            matrix = information + penalty
            coefficients = torch.linalg.solve(matrix, rhs)
            transfer = torch.linalg.solve(matrix, information)
            sampling_covariance = torch.linalg.solve(matrix, transfer.T)
        else:
            nuisance = CalibratedNuisancePrecision(
                data.new_ones(4, 1),
                apply_precision=lambda v: weights * v,
                covariance_factor=amplitude.reshape(1, 1),
            )
            result = fit_linear_query(
                _system(
                    data,
                    nuisance.apply,
                    nuisance_treatment="calibrated",
                    eliminated_nuisance_ids=("zero",),
                ),
                (templates,),
                penalty_precision=penalty,
            )
            coefficients = result.coefficients
            transfer = result.mean_response
            sampling_covariance = result.sampling_covariance
        loss = (
            coefficients.square().sum()
            + 0.2 * sampling_covariance.sum()
            + 0.1 * transfer.square().sum()
        )
        first = torch.autograd.grad(loss, args, create_graph=True)
        direction = sum(g.sum() for g in first)
        second = torch.autograd.grad(direction, args)
        return (loss.detach().cpu().double(),) + tuple(
            g.detach().cpu().double() for g in (*first, *second)
        )

    expected_inputs = tuple(x.detach().cpu().double().requires_grad_() for x in inputs)
    torch.testing.assert_close(
        evaluate(inputs, reference=False),
        evaluate(expected_inputs, reference=True),
        rtol=4e-4,
        atol=4e-5,
    )
