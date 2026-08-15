import math

import pytest
import torch

from torchsky.map.field import (
    completeness_limit_magnitude,
    paired_bootstrap_improvement,
)


def test_completeness_limits_interpolate_m80_and_m50_without_extrapolation() -> None:
    magnitude = torch.tensor([27.0, 28.0, 29.0, 30.0])
    completeness = torch.tensor([0.95, 0.85, 0.65, 0.35])

    assert completeness_limit_magnitude(
        magnitude, completeness, target=0.8
    ) == pytest.approx(28.25)
    assert completeness_limit_magnitude(
        magnitude, completeness, target=0.5
    ) == pytest.approx(29.5)
    assert math.isnan(
        completeness_limit_magnitude(magnitude, completeness, target=0.2)
    )


def test_paired_bootstrap_reports_directional_held_out_improvement() -> None:
    baseline_error = torch.linspace(1.0, 2.0, 32)
    candidate_error = baseline_error - 0.2
    interval = paired_bootstrap_improvement(
        candidate_error,
        baseline_error,
        larger_is_better=False,
        samples=500,
        seed=12,
    )

    assert interval.estimate > 0.0
    assert interval.lower > 0.0
    assert interval.excludes_zero
