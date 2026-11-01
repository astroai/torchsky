from __future__ import annotations

import pytest
import torch

from torchsky.map.solvers import (
    BreakdownReason,
    IdentityPreconditioner,
    ImplicitDifferentiationPolicy,
    SolverDifferentiationMode,
    TensorNullspaceProjection,
    TerminationReason,
    build_state_matrix_operator,
    conjugate_gradient,
    lsqr,
    make_diagonal_preconditioner,
    tree_inner,
)


def test_cg_matches_dense_spd_oracle_and_reports_actual_residual() -> None:
    matrix = torch.tensor(
        [[5.0, 1.0, 0.5], [1.0, 4.0, -0.25], [0.5, -0.25, 2.0]],
        dtype=torch.float64,
    )
    rhs = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    preconditioner = make_diagonal_preconditioner(matrix.diagonal())

    result = conjugate_gradient(
        lambda state: matrix @ state,
        rhs,
        operator_is_hpd=True,
        preconditioner=preconditioner,
        max_iterations=8,
        rtol=1.0e-12,
        atol=1.0e-14,
    )
    actual_residual = torch.linalg.vector_norm(matrix @ result.solution - rhs).item()

    torch.testing.assert_close(result.solution, torch.linalg.solve(matrix, rhs))
    assert result.diagnostics is not None
    assert result.diagnostics.termination is TerminationReason.CONVERGED
    assert result.diagnostics.preconditioner_identity == "diagonal"
    assert result.diagnostics.residual_norm == pytest.approx(actual_residual)
    assert result.residual_norm_history[-1] == pytest.approx(actual_residual)


def test_cg_ill_conditioned_diagonal_uses_declared_preconditioner_identity() -> None:
    diagonal = torch.tensor([1.0e-10, 1.0, 1.0e10], dtype=torch.float64)
    rhs = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float64)

    result = conjugate_gradient(
        lambda state: diagonal * state,
        rhs,
        operator_is_hpd=True,
        preconditioner=make_diagonal_preconditioner(diagonal),
        max_iterations=3,
        rtol=1.0e-12,
    )

    torch.testing.assert_close(result.solution, rhs / diagonal, rtol=1.0e-12, atol=0.0)
    assert result.diagnostics is not None and result.diagnostics.converged
    assert result.diagnostics.preconditioner_applications == 1


def test_cg_requires_hpd_declaration_and_reports_curvature_breakdown() -> None:
    rhs = torch.ones(2, dtype=torch.float64)
    with pytest.raises(ValueError, match="operator_is_hpd=True"):
        conjugate_gradient(lambda state: state, rhs, operator_is_hpd=False)

    result = conjugate_gradient(
        lambda state: torch.tensor([state[0], -state[1]]),
        rhs,
        operator_is_hpd=True,
        max_iterations=2,
        rtol=0.0,
        atol=0.0,
    )

    assert result.diagnostics is not None
    assert result.diagnostics.termination is TerminationReason.BREAKDOWN
    assert result.diagnostics.breakdown is not None
    assert result.diagnostics.breakdown.reason is BreakdownReason.NON_POSITIVE_CURVATURE


def test_projected_cg_solves_singular_gauge_system() -> None:
    matrix = torch.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=torch.float64)
    rhs = torch.tensor([2.0, -2.0], dtype=torch.float64)
    projection = TensorNullspaceProjection(
        torch.ones((1, 2), dtype=torch.float64), state_shape=(2,)
    )

    result = conjugate_gradient(
        lambda state: matrix @ state,
        rhs,
        operator_is_hpd=True,
        projection=projection,
        max_iterations=2,
        rtol=1.0e-12,
    )

    torch.testing.assert_close(result.solution, torch.tensor([1.0, -1.0], dtype=torch.float64))
    torch.testing.assert_close(result.solution.mean(), torch.tensor(0.0, dtype=torch.float64))
    assert result.diagnostics is not None and result.diagnostics.projected


def test_complex_nullspace_projection_uses_hermitian_geometry() -> None:
    basis = torch.tensor([[1.0 + 1.0j, 2.0 - 0.5j]], dtype=torch.complex128)
    projection = TensorNullspaceProjection(basis, state_shape=(2,))
    state = torch.tensor([0.5 - 1.0j, 3.0 + 2.0j], dtype=torch.complex128)

    projected = projection.project(state)

    torch.testing.assert_close(torch.sum(torch.conj(basis[0]) * projected), torch.zeros((), dtype=torch.complex128))


def test_lsqr_matches_dense_rectangular_and_minimum_norm_oracles() -> None:
    over = torch.tensor(
        [[1.0, 2.0], [3.0, -1.0], [0.5, 4.0], [2.0, 1.0]],
        dtype=torch.float64,
    )
    data = torch.tensor([1.0, -2.0, 0.5, 3.0], dtype=torch.float64)
    result = lsqr(
        lambda state: over @ state,
        lambda residual: over.mT @ residual,
        data,
        max_iterations=8,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    expected = torch.linalg.lstsq(over, data).solution

    torch.testing.assert_close(result.solution, expected, rtol=1.0e-10, atol=1.0e-10)
    assert result.diagnostics is not None and result.diagnostics.converged
    assert result.diagnostics.normal_residual_norm is not None

    under = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=torch.float64)
    under_data = torch.tensor([1.0, 2.0], dtype=torch.float64)
    minimum_norm = lsqr(
        lambda state: under @ state,
        lambda residual: under.mT @ residual,
        under_data,
        max_iterations=6,
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    torch.testing.assert_close(
        minimum_norm.solution,
        torch.linalg.pinv(under) @ under_data,
        rtol=1.0e-10,
        atol=1.0e-10,
    )


def test_lsqr_damping_matches_augmented_dense_oracle() -> None:
    matrix = torch.tensor(
        [[1.0, 2.0], [3.0, -1.0], [0.5, 4.0]], dtype=torch.float64
    )
    rhs = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    damping = 0.25

    result = lsqr(
        lambda state: matrix @ state,
        lambda residual: matrix.mT @ residual,
        rhs,
        damping=damping,
        max_iterations=8,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    expected = torch.linalg.solve(
        matrix.mT @ matrix + damping * torch.eye(2, dtype=torch.float64),
        matrix.mT @ rhs,
    )

    torch.testing.assert_close(result.solution, expected, rtol=1.0e-10, atol=1.0e-10)


def test_state_tree_cg_supports_blocks_and_preserves_adjoint_identity() -> None:
    rhs = {
        "sky": torch.tensor([2.0, -4.0], dtype=torch.float64),
        "calibration": (torch.tensor([3.0], dtype=torch.float64),),
    }

    def action(state):
        return {
            "sky": 2.0 * state["sky"],
            "calibration": (3.0 * state["calibration"][0],),
        }

    result = conjugate_gradient(
        action,
        rhs,
        operator_is_hpd=True,
        preconditioner=IdentityPreconditioner(),
        max_iterations=4,
        rtol=1.0e-12,
    )

    torch.testing.assert_close(result.solution["sky"], rhs["sky"] / 2.0)
    torch.testing.assert_close(
        result.solution["calibration"][0], rhs["calibration"][0] / 3.0
    )
    probe = action(rhs)
    torch.testing.assert_close(tree_inner(action(rhs), probe), tree_inner(rhs, action(probe)))


def test_state_tree_lsqr_supports_rectangular_block_operators() -> None:
    rhs = (
        torch.tensor([1.0, 2.0], dtype=torch.float64),
        torch.tensor([-1.0], dtype=torch.float64),
    )

    def operator(state):
        return (state["x"], state["x"].sum().reshape(1) + state["offset"])

    def adjoint(data):
        return {
            "x": data[0] + data[1].expand(2),
            "offset": data[1],
        }

    result = lsqr(
        operator,
        adjoint,
        rhs,
        max_iterations=5,
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    torch.testing.assert_close(result.solution["x"], rhs[0])
    torch.testing.assert_close(result.solution["offset"], torch.tensor([-4.0], dtype=torch.float64))


def test_solver_inputs_handle_empty_and_noncontiguous_tensors() -> None:
    empty = torch.empty(0, dtype=torch.float32)
    empty_result = conjugate_gradient(
        lambda state: state,
        empty,
        operator_is_hpd=True,
        max_iterations=2,
    )
    assert empty_result.solution.shape == (0,)
    assert empty_result.diagnostics is not None and empty_result.diagnostics.converged

    storage = torch.arange(12, dtype=torch.float64).reshape(3, 4)
    rhs = storage.mT
    assert not rhs.is_contiguous()
    result = conjugate_gradient(
        lambda state: 2.0 * state,
        rhs,
        operator_is_hpd=True,
        max_iterations=2,
        rtol=1.0e-12,
    )
    torch.testing.assert_close(result.solution, rhs / 2.0)


def test_nonfinite_inputs_fail_closed_and_operator_nonfinite_is_diagnosed() -> None:
    with pytest.raises(ValueError, match="rhs must be finite"):
        conjugate_gradient(
            lambda state: state,
            torch.tensor([float("nan")]),
            operator_is_hpd=True,
        )

    result = conjugate_gradient(
        lambda state: torch.full_like(state, float("inf")),
        torch.ones(2),
        operator_is_hpd=True,
    )
    assert result.diagnostics is not None
    assert result.diagnostics.termination is TerminationReason.NONFINITE
    assert result.diagnostics.breakdown is not None
    assert result.diagnostics.breakdown.reason is BreakdownReason.NONFINITE_OPERATOR


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
        ),
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason="MPS unavailable"
            ),
        ),
    ],
)
def test_cg_and_lsqr_preserve_device(device: str) -> None:
    matrix = torch.tensor([[3.0, 1.0], [1.0, 2.0]], device=device)
    rhs = torch.tensor([1.0, -2.0], device=device)
    cg_result = conjugate_gradient(
        lambda state: matrix @ state,
        rhs,
        operator_is_hpd=True,
        max_iterations=4,
        rtol=1.0e-5,
    )
    rectangular = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], device=device
    )
    data = rectangular @ rhs
    lsqr_result = lsqr(
        lambda state: rectangular @ state,
        lambda residual: rectangular.mT @ residual,
        data,
        max_iterations=4,
        rtol=1.0e-5,
    )

    assert cg_result.solution.device.type == device
    assert lsqr_result.solution.device.type == device
    torch.testing.assert_close(lsqr_result.solution, rhs, rtol=2.0e-4, atol=2.0e-5)


def test_unrolled_cg_and_lsqr_gradients_match_dense_solutions() -> None:
    scale = torch.tensor(2.5, dtype=torch.float64, requires_grad=True)
    rhs = torch.tensor([1.0, -2.0], dtype=torch.float64)
    cg_result = conjugate_gradient(
        lambda state: scale * state,
        rhs,
        operator_is_hpd=True,
        max_iterations=2,
        rtol=1.0e-12,
    )
    cg_result.solution.sum().backward()
    torch.testing.assert_close(scale.grad, -rhs.sum() / scale.detach().square())

    matrix = torch.tensor(
        [[1.0, 2.0], [3.0, -1.0], [0.5, 4.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    data = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    lsqr_result = lsqr(
        lambda state: matrix @ state,
        lambda residual: matrix.mT @ residual,
        data,
        max_iterations=4,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    loss = lsqr_result.solution.square().sum()
    gradient = torch.autograd.grad(loss, matrix)[0]
    dense_loss = torch.linalg.lstsq(matrix, data).solution.square().sum()
    expected_gradient = torch.autograd.grad(dense_loss, matrix)[0]
    torch.testing.assert_close(gradient, expected_gradient, rtol=1.0e-8, atol=1.0e-8)


def test_bound_operator_cg_and_lsqr_keep_existing_solution_surface() -> None:
    state = torch.zeros(2, dtype=torch.float64)
    matrix = torch.tensor([[1.0, 2.0], [3.0, -1.0]], dtype=torch.float64)
    operator = build_state_matrix_operator(state, matrix)
    truth = torch.tensor([0.25, -0.75], dtype=torch.float64)
    data = matrix @ truth

    cg_result = operator.solve_normal_equation_cg(
        data,
        iterations=4,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    lsqr_result = operator.solve_least_squares_lsqr(
        data,
        iterations=4,
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    torch.testing.assert_close(cg_result.solution, truth)
    torch.testing.assert_close(lsqr_result.solution, truth)
    assert cg_result.diagnostics is not None and cg_result.diagnostics.converged
    assert lsqr_result.diagnostics is not None and lsqr_result.diagnostics.converged


def test_diagnostics_memory_is_bounded_by_iteration_budget() -> None:
    calls = 0

    def action(state):
        nonlocal calls
        calls += 1
        return 2.0 * state

    result = conjugate_gradient(
        action,
        torch.ones(1024),
        operator_is_hpd=True,
        max_iterations=32,
        rtol=0.0,
        atol=0.0,
        record_residual_history=False,
    )

    assert result.residual_norm_history == ()
    assert result.diagnostics is not None
    assert result.diagnostics.operator_applications == calls
    assert calls <= 2 * 32 + 1


def test_implicit_differentiation_policy_is_separate_and_fail_closed() -> None:
    result = conjugate_gradient(
        lambda state: state,
        torch.ones(1),
        operator_is_hpd=True,
        max_iterations=1,
    )
    ImplicitDifferentiationPolicy().validate(result)
    with pytest.raises(NotImplementedError, match="policy-only"):
        ImplicitDifferentiationPolicy(
            mode=SolverDifferentiationMode.IMPLICIT
        ).validate(result)


def test_cg_and_lsqr_apply_operator_once_per_iteration() -> None:
    from torchsky.map.solvers._foundation import conjugate_gradient, lsqr

    torch.manual_seed(0)
    a = torch.randn(40, 25, dtype=torch.float64)
    b = torch.randn(40, dtype=torch.float64)
    m = a.T @ a
    rhs = a.T @ b
    reference = torch.linalg.solve(m, rhs)

    calls = {"op": 0}

    def op(x):
        calls["op"] += 1
        return m @ x

    cg = conjugate_gradient(op, rhs, operator_is_hpd=True, max_iterations=200, rtol=1e-12)
    assert torch.linalg.norm(cg.solution - reference) < 1e-9
    # initial residual + one per iteration + one convergence certification
    assert calls["op"] == cg.diagnostics.iterations + 2
    assert cg.diagnostics.operator_applications == calls["op"]

    ops = {"n": 0}

    def fwd(x):
        ops["n"] += 1
        return a @ x

    for damping in (0.0, 0.3):
        ops["n"] = 0
        res = lsqr(fwd, lambda y: a.T @ y, b, max_iterations=200, rtol=1e-12, damping=damping)
        ref = torch.linalg.solve(m + damping * torch.eye(25, dtype=torch.float64), rhs)
        assert torch.linalg.norm(res.solution - ref) < 1e-9
        assert ops["n"] == res.diagnostics.iterations + 2


def test_cg_casts_initial_solution_to_rhs_dtype() -> None:
    from torchsky.map.solvers._foundation import conjugate_gradient

    m = torch.eye(4, dtype=torch.float64) * 2.0
    rhs = torch.ones(4, dtype=torch.float64)
    result = conjugate_gradient(
        lambda x: m @ x,
        rhs,
        operator_is_hpd=True,
        initial_solution=torch.zeros(4, dtype=torch.float32),
    )
    assert result.solution.dtype == torch.float64
    torch.testing.assert_close(result.solution, rhs / 2.0)
