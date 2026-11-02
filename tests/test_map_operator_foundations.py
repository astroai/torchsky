from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from torchsky import WCS
from torchsky.map.evidence import EvidenceSnapshot
from torchsky.map.operators import (
    AutogradMapping,
    BlockDiagonalCovariance,
    BlockPlan,
    CallableLinearOperator,
    ChainPlan,
    DiagonalCovariance,
    DifferentiableMapping,
    DomainSpec,
    DrizzleResponsePlan,
    LinearOperator,
    LowRankCovariance,
    MaskPlan,
    NuisancePrecisionAdapter,
    ObservationOperatorBundle,
    RestrictionPlan,
    SumPlan,
    bilinear_response,
    direct_copy_response,
    geometry_op_for_kernel,
    plan_fingerprint,
)


def _matrix_operator(matrix: torch.Tensor, key: str) -> CallableLinearOperator:
    return CallableLinearOperator(
        DomainSpec((matrix.shape[1],), matrix.dtype),
        DomainSpec((matrix.shape[0],), matrix.dtype),
        lambda value: matrix @ value,
        lambda value: matrix.mT @ value,
        key,
    )


def _assert_adjoint(
    operator: LinearOperator, source: torch.Tensor, target: torch.Tensor
) -> None:
    torch.testing.assert_close(
        torch.sum(operator.apply(source) * target),
        torch.sum(source * operator.adjoint(target)),
    )


def test_domain_spec_adapts_tensors_and_wcs_without_map_specific_copy() -> None:
    tensor = torch.zeros((2, 3), dtype=torch.float64)
    assert DomainSpec.from_tensor(tensor) == DomainSpec((2, 3), torch.float64)
    wcs = WCS(
        {
            "NAXIS": 2,
            "NAXIS1": 3,
            "NAXIS2": 2,
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
            "CUNIT1": "deg",
            "CUNIT2": "deg",
            "CRPIX1": 2.0,
            "CRPIX2": 1.5,
            "CRVAL1": 0.0,
            "CRVAL2": 0.0,
            "CDELT1": -0.1,
            "CDELT2": 0.1,
        }
    )
    domain = DomainSpec.from_wcs(wcs)
    assert domain.shape == (2, 3)
    assert domain.axes == ("DEC--TAN", "RA---TAN")
    assert domain.units == ("deg", "deg")


def test_linear_compositions_match_dense_oracles_and_exact_adjoints() -> None:
    dtype = torch.float64
    a = torch.tensor([[1.0, 2.0, 0.0], [-1.0, 0.5, 3.0]], dtype=dtype)
    b = torch.tensor([[2.0, -1.0], [0.0, 4.0]], dtype=dtype)
    op_a = _matrix_operator(a, "a")
    op_b = _matrix_operator(b, "b")
    chain = ChainPlan((op_a, op_b))
    source = torch.tensor([0.5, -2.0, 1.0], dtype=dtype)
    target = torch.tensor([3.0, -1.0], dtype=dtype)
    assert isinstance(chain, LinearOperator)
    torch.testing.assert_close(chain.apply(source), b @ a @ source)
    _assert_adjoint(chain, source, target)

    sum_plan = SumPlan((op_a, _matrix_operator(2.0 * a, "two-a")))
    torch.testing.assert_close(sum_plan.apply(source), 3.0 * a @ source)
    _assert_adjoint(sum_plan, source, torch.tensor([1.5, -0.25], dtype=dtype))

    block = BlockPlan((op_a, op_a), ((2,), (2,)))
    block_target = torch.tensor([1.0, 2.0, -3.0, 0.5], dtype=dtype)
    torch.testing.assert_close(block.apply(source), torch.cat((a @ source, a @ source)))
    _assert_adjoint(block, source, block_target)


def test_restriction_mask_noncontiguous_and_empty_domains() -> None:
    base = torch.arange(12, dtype=torch.float64).reshape(3, 4).mT
    assert not base.is_contiguous()
    restriction = RestrictionPlan(
        DomainSpec((4, 3), torch.float64),
        torch.tensor([0, 4, 11], dtype=torch.int64),
    )
    selected = restriction.apply(base)
    torch.testing.assert_close(
        selected, torch.tensor([0.0, 5.0, 11.0], dtype=torch.float64)
    )
    _assert_adjoint(
        restriction, base, torch.tensor([2.0, -1.0, 4.0], dtype=torch.float64)
    )

    mask = MaskPlan(torch.tensor([[True, False], [False, True]]))
    value = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    torch.testing.assert_close(
        mask.apply(value), torch.tensor([[1.0, 0.0], [0.0, 4.0]])
    )
    _assert_adjoint(mask, value, torch.ones_like(value))

    empty = direct_copy_response((0,), dtype=torch.float32)
    assert empty.apply(torch.empty(0)).numel() == 0
    assert empty.adjoint(torch.empty(0)).numel() == 0
    assert empty.apply_squared(torch.empty(0)).numel() == 0
    assert empty.context().shape == (0, 0)


def test_differentiable_mapping_jvp_and_vjp_are_point_dependent_not_adjoint() -> None:
    mapping = AutogradMapping(lambda value: value.square())
    assert isinstance(mapping, DifferentiableMapping)
    assert not isinstance(mapping, LinearOperator)
    point = torch.tensor([2.0, -3.0], dtype=torch.float64)
    direction = torch.tensor([0.5, 4.0], dtype=torch.float64)
    cotangent = torch.tensor([-1.0, 2.0], dtype=torch.float64)
    torch.testing.assert_close(mapping.jvp(point, direction), 2.0 * point * direction)
    torch.testing.assert_close(mapping.vjp(point, cotangent), 2.0 * point * cotangent)


def test_indexed_response_dense_squared_dq_context_and_chunk_parity() -> None:
    x = torch.tensor([[0.25, 1.5], [2.0, -1.0]], dtype=torch.float64)
    y = torch.tensor([[0.5, 1.0], [1.5, 0.0]], dtype=torch.float64)
    plan = bilinear_response((3, 3), x, y, default_chunk_size=2)
    dense = torch.zeros((4, 9), dtype=torch.float64)
    dense[plan.target_indices, plan.source_indices] += plan.coefficients
    source = torch.arange(9, dtype=torch.float64).reshape(3, 3).mT
    assert not source.is_contiguous()

    for chunk_size in (1, 2, 17):
        torch.testing.assert_close(
            plan.apply(source, chunk_size=chunk_size),
            (dense @ source.reshape(-1)).reshape(2, 2),
        )
        target = torch.tensor([[1.0, -2.0], [0.5, 3.0]], dtype=torch.float64)
        torch.testing.assert_close(
            plan.adjoint(target, chunk_size=chunk_size),
            (dense.mT @ target.reshape(-1)).reshape(3, 3),
        )
        variance = torch.linspace(0.5, 2.5, 9, dtype=torch.float64).reshape(3, 3)
        torch.testing.assert_close(
            plan.apply_squared(variance, chunk_size=chunk_size),
            (dense.square() @ variance.reshape(-1)).reshape(2, 2),
        )

    _assert_adjoint(
        plan, source, torch.tensor([[1.0, -2.0], [0.5, 3.0]], dtype=torch.float64)
    )
    flags = torch.tensor([[1, 2, 4], [8, 16, 32], [64, 128, 256]], dtype=torch.int64)
    propagated = plan.apply_flags(flags, chunk_size=1)
    expected = torch.zeros(4, dtype=torch.int64)
    for target_index, source_index in zip(
        plan.target_indices.tolist(), plan.source_indices.tolist(), strict=True
    ):
        expected[target_index] |= flags.reshape(-1)[source_index]
    torch.testing.assert_close(propagated.reshape(-1), expected)
    assert torch.equal(
        plan.contributor_count(chunk_size=1), plan.contributor_count(chunk_size=19)
    )
    assert torch.equal(plan.context(chunk_size=1), plan.context(chunk_size=19))

    exact = bilinear_response(
        (2, 2),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
    )
    assert (
        exact.apply_flags(torch.tensor([[1, 2], [4, 8]], dtype=torch.int64)).item() == 1
    )
    assert exact.contributor_count().item() == 1


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason="MPS unavailable"
            ),
        ),
    ],
)
def test_direct_response_preserves_device_gradient_and_chunk_parity(
    device: str,
) -> None:
    plan = direct_copy_response(
        (3, 4), dtype=torch.float32, device=device, default_chunk_size=3
    )
    value = (
        torch.arange(12, dtype=torch.float32, device=device)
        .reshape(3, 4)
        .requires_grad_()
    )
    output = plan.apply(value, chunk_size=2)
    torch.testing.assert_close(output, value)
    output.square().sum().backward()
    torch.testing.assert_close(value.grad, 2.0 * value)
    torch.testing.assert_close(
        plan.apply_squared(value.detach(), chunk_size=99), value.detach()
    )


def test_existing_drizzle_actions_adapt_without_conflating_squared_response() -> None:
    matrix = torch.tensor([[1.0, 0.5], [0.0, -2.0]], dtype=torch.float64)
    plan = DrizzleResponsePlan(
        DomainSpec((2,), torch.float64),
        DomainSpec((2,), torch.float64),
        lambda value, chunk: matrix @ value,
        lambda value, chunk: matrix.mT @ value,
        lambda value, chunk: matrix.square() @ value,
        "tiny-drizzle",
    )
    value = torch.tensor([2.0, 3.0], dtype=torch.float64)
    torch.testing.assert_close(plan.apply(value), matrix @ value)
    torch.testing.assert_close(plan.apply_squared(value), matrix.square() @ value)
    _assert_adjoint(plan, value, torch.tensor([-1.0, 4.0], dtype=torch.float64))
    with pytest.raises(NotImplementedError, match="flag"):
        plan.apply_flags(torch.ones(2, dtype=torch.int64))


def test_covariance_capabilities_match_dense_and_monte_carlo_query_variance() -> None:
    dtype = torch.float64
    variance = torch.tensor([0.5, 1.5, 2.0], dtype=dtype)
    factor = torch.tensor([[0.2, -0.1], [0.5, 0.3], [-0.4, 0.2]], dtype=dtype)
    covariance = LowRankCovariance(variance, factor)
    dense = torch.diag(variance) + factor @ factor.mT
    value = torch.tensor([1.0, -2.0, 0.5], dtype=dtype)
    torch.testing.assert_close(covariance.apply_covariance(value), dense @ value)
    torch.testing.assert_close(
        covariance.apply_precision(value), torch.linalg.solve(dense, value)
    )
    torch.testing.assert_close(covariance.diagonal(), torch.diagonal(dense))
    torch.testing.assert_close(
        covariance.logdet(), torch.linalg.slogdet(dense).logabsdet
    )
    torch.testing.assert_close(covariance.query_variance(value), value @ dense @ value)

    generator = torch.Generator().manual_seed(1234)
    samples = covariance.sample((80_000,), generator=generator)
    empirical = torch.var(samples @ value, unbiased=True)
    torch.testing.assert_close(
        empirical, covariance.query_variance(value), rtol=0.025, atol=0.0
    )


def test_diagonal_block_and_nuisance_capabilities_are_explicit() -> None:
    first = DiagonalCovariance(torch.tensor([1.0, 4.0], dtype=torch.float64))
    second = DiagonalCovariance(torch.tensor([9.0], dtype=torch.float64))
    block = BlockDiagonalCovariance((first, second))
    value = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)
    torch.testing.assert_close(
        block.apply_covariance(value),
        torch.tensor([2.0, 12.0, 36.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        block.apply_precision(value),
        torch.tensor([2.0, 0.75, 4.0 / 9.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        block.whiten(value), torch.tensor([2.0, 1.5, 4.0 / 3.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        block.diagonal(), torch.tensor([1.0, 4.0, 9.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        block.logdet(), torch.log(torch.tensor(36.0, dtype=torch.float64))
    )

    nuisance = NuisancePrecisionAdapter(lambda tensor: 2.0 * tensor)
    torch.testing.assert_close(nuisance.apply_precision(value), 2.0 * value)
    with pytest.raises(NotImplementedError, match="logdet"):
        nuisance.logdet()


@dataclass
class _MutablePlan:
    fingerprint: object


def test_plan_fingerprints_and_evidence_linkage_fail_closed_on_mutation() -> None:
    first = plan_fingerprint("test", {"revision": 1})
    second = plan_fingerprint("test", {"revision": 2})
    mutable = _MutablePlan(first)
    snapshot = EvidenceSnapshot.capture(plans={"response": mutable})
    snapshot.validate()
    mutable.fingerprint = second
    with pytest.raises(RuntimeError, match="stale evidence plan"):
        snapshot.validate()

    bundle = ObservationOperatorBundle(geometry=geometry_op_for_kernel("bilinear"))
    bundle_snapshot = EvidenceSnapshot.capture(plans={"bundle": bundle})
    assert bundle.geometry is not None
    bundle.geometry.enabled = False
    with pytest.raises(RuntimeError, match="stale evidence plan"):
        bundle_snapshot.validate()

    response = direct_copy_response((2,), dtype=torch.float64)
    with torch.no_grad():
        response.coefficients[0] = 2.0
    with pytest.raises(RuntimeError, match="stale evidence dependency"):
        response.apply(torch.ones(2, dtype=torch.float64))
