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
    assert math.isnan(completeness_limit_magnitude(magnitude, completeness, target=0.2))


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


@pytest.mark.parametrize("statistic", ["mean", "median"])
def test_paired_bootstrap_chunked_matches_seeded_materialized_reference(
    statistic: str,
) -> None:
    candidate = torch.linspace(-1.0, 2.0, 11, dtype=torch.float64)
    baseline = torch.linspace(-0.5, 1.5, 11, dtype=torch.float64)
    difference = candidate - baseline
    samples = 128
    seed = 37
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randint(11, (samples, 11), generator=generator)
    draws = difference[indices]
    bootstrap = (
        draws.mean(dim=-1) if statistic == "mean" else draws.median(dim=-1).values
    )
    expected = torch.quantile(
        bootstrap,
        torch.tensor([0.025, 0.975], dtype=bootstrap.dtype),
    )

    interval = paired_bootstrap_improvement(
        candidate,
        baseline,
        statistic=statistic,
        samples=samples,
        seed=seed,
        chunk_size=7,
    )

    assert interval.estimate == pytest.approx(float(getattr(difference, statistic)()))
    assert interval.lower == pytest.approx(float(expected[0]))
    assert interval.upper == pytest.approx(float(expected[1]))


def test_paired_bootstrap_seed_is_deterministic() -> None:
    candidate = torch.arange(17, dtype=torch.float32)
    baseline = torch.sin(candidate)

    first = paired_bootstrap_improvement(candidate, baseline, samples=151, seed=91)
    second = paired_bootstrap_improvement(candidate, baseline, samples=151, seed=91)

    assert first == second


@pytest.mark.parametrize(
    ("device", "dtype"),
    [("cpu", torch.float32), ("cpu", torch.float64)]
    + (
        [("cuda", torch.float32), ("cuda", torch.float64)]
        if torch.cuda.is_available()
        else []
    )
    + ([("mps", torch.float32)] if torch.backends.mps.is_available() else []),
)
def test_paired_bootstrap_accepts_supported_device_and_dtype(
    dtype: torch.dtype,
    device: str,
) -> None:
    candidate = torch.linspace(0.0, 1.0, 19, dtype=dtype, device=device)
    baseline = candidate - 0.25

    interval = paired_bootstrap_improvement(
        candidate,
        baseline,
        statistic="mean",
        samples=100,
        seed=5,
        chunk_size=3,
    )

    assert interval.estimate == pytest.approx(0.25)
    assert interval.lower == pytest.approx(0.25)
    assert interval.upper == pytest.approx(0.25)
