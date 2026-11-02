"""Software checks only; these fixtures provide no scientific evidence."""

import math

import pytest
import torch


def test_neural_pcg_matches_dense_linear_normal_equations() -> None:
    from torchsky.map.field._gauss_newton import gauss_newton_pcg_step

    matrix = torch.tensor([[1.0, 2.0], [3.0, -1.0], [0.5, 1.0]], dtype=torch.float64)
    target = torch.tensor([1.0, 2.0, -1.0], dtype=torch.float64)
    initial = torch.tensor([0.3, -0.1], dtype=torch.float64)
    updated, report = gauss_newton_pcg_step(
        [lambda state: matrix @ state - target],
        initial,
        cg_iterations=4,
    )
    normal = matrix.T @ matrix + report["effective_damping"] * torch.eye(2)
    expected = initial + torch.linalg.solve(
        normal, matrix.T @ (target - matrix @ initial)
    )
    torch.testing.assert_close(updated, expected)
    assert report["final_loss"] < report["initial_loss"]
    assert report["accepted_step_scale"] == 1.0


def test_neural_pcg_backtracks_a_nonlinear_step() -> None:
    from torchsky.map.field._gauss_newton import gauss_newton_pcg_step

    initial = torch.tensor([0.1], dtype=torch.float64)
    updated, report = gauss_newton_pcg_step(
        [lambda state: state.square() - 1.0],
        initial,
        cg_iterations=2,
    )
    assert 0.0 < report["accepted_step_scale"] < 1.0
    assert report["final_loss"] < report["initial_loss"]
    assert updated.item() > initial.item()


def test_neural_pcg_sums_blocks_and_rejects_nonfinite_input() -> None:
    from torchsky.map.field._gauss_newton import gauss_newton_pcg_step

    matrix = torch.tensor([[1.0, 2.0], [3.0, -1.0], [0.5, 1.0]], dtype=torch.float64)
    target = torch.tensor([1.0, 2.0, -1.0], dtype=torch.float64)
    state = torch.tensor([0.3, -0.1], dtype=torch.float64)
    blocks = [
        lambda flat: matrix[:1] @ flat - target[:1],
        lambda flat: matrix[1:] @ flat - target[1:],
    ]
    updated, report = gauss_newton_pcg_step(blocks, state, cg_iterations=4)
    normal = matrix.T @ matrix + report["effective_damping"] * torch.eye(2)
    expected = state + torch.linalg.solve(normal, matrix.T @ (target - matrix @ state))
    torch.testing.assert_close(updated, expected)
    assert report["cg_residual_norm"] < 1.0e-12
    with pytest.raises(ValueError, match="finite"):
        gauss_newton_pcg_step([lambda flat: flat * float("nan")], state)
    unchanged, stationary = gauss_newton_pcg_step(
        [lambda flat: flat * 0.0 + 1.0], state
    )
    torch.testing.assert_close(unchanged, state)
    assert stationary["accepted_step_scale"] == 0.0


class _SimpleDetectorToSky:
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
        ra = self.center_ra_deg + x_local / math.cos(math.radians(self.center_dec_deg))
        dec = self.center_dec_deg + y_local
        return ra, dec
