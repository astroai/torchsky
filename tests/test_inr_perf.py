"""INR field model performance and parity tests.

Loads ``benchmarks/bench_inr_perf.py`` via importlib and validates training
throughput, render quality, and model preset configurations against the
project benchmark contracts defined in ``docs/benchmarks.md``.

These tests require ``torchfits`` to be installed and functional.

Usage::

    pixi run -e test -- pytest tests/test_inr_perf.py -v
"""

from __future__ import annotations

import math
import sys
import importlib.util
from pathlib import Path
import pytest
import torch

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
pytestmark = pytest.mark.performance


def _load_benchmark_module(name: str, filename: str):
    """Load a benchmark module by name from the benchmarks directory."""
    path = BENCHMARKS / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load benchmark module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_inr_perf = _load_benchmark_module("bench_inr_perf", "bench_inr_perf.py")


# ── Throughput benchmark tests ────────────────────────────────────────


@pytest.mark.parametrize(
    "case_name",
    [c.name for c in _inr_perf.DEFAULT_PERF_CASES],
)
def test_inr_training_reduces_loss(case_name: str) -> None:
    """INR training must reduce loss on synthetic data for every default case."""
    spec = next(c for c in _inr_perf.DEFAULT_PERF_CASES if c.name == case_name)
    tp = _inr_perf.measure_training_throughput(spec, device="cpu")

    assert tp.initial_loss > 0.0, (
        f"initial_loss must be positive, got {tp.initial_loss}"
    )
    assert tp.final_loss > 0.0, f"final_loss must be positive, got {tp.final_loss}"
    assert tp.final_loss < tp.initial_loss, (
        f"loss must improve: {tp.initial_loss:.4f} → {tp.final_loss:.4f}"
    )
    assert tp.loss_improvement_ratio > 1.01, (
        f"loss_improvement_ratio {tp.loss_improvement_ratio:.2f} must be > 1.01"
    )
    assert tp.steps_per_sec > 0.0, f"steps_per_sec {tp.steps_per_sec} must be positive"
    assert tp.pixels_per_sec > 0.0, (
        f"pixels_per_sec {tp.pixels_per_sec} must be positive"
    )
    assert tp.num_params > 0, f"num_params {tp.num_params} must be positive"


def test_inr_throughput_device_independent() -> None:
    """Training throughput on CPU should be measurable and finite."""
    spec = _inr_perf.DEFAULT_PERF_CASES[0]
    tp = _inr_perf.measure_training_throughput(spec, device="cpu")

    assert tp.device == "cpu"
    assert math.isfinite(tp.steps_per_sec)
    assert math.isfinite(tp.pixels_per_sec)
    assert tp.duration_sec > 0.0
    assert tp.total_pixels_processed > 0


def test_inr_render_produces_finite_image() -> None:
    """INR render must produce a finite 2D image matching the target shape."""
    spec = _inr_perf.DEFAULT_PERF_CASES[0]
    rq = _inr_perf.measure_render_quality(spec, device="cpu")

    assert rq.target_shape == spec.target_pixels
    assert rq.inr_render_duration_sec > 0.0
    assert math.isfinite(rq.inr_psnr_db)
    assert math.isfinite(rq.inr_ssim)
    assert 0.0 <= rq.inr_ssim <= 1.0, f"SSIM {rq.inr_ssim} must be in [0, 1]"


def test_inr_render_quality_vs_truth() -> None:
    """INR render should show reasonable PSNR vs ground truth on low-noise data."""
    spec = _inr_perf.DEFAULT_PERF_CASES[0]
    rq = _inr_perf.measure_render_quality(spec, device="cpu")

    assert math.isfinite(rq.inr_psnr_vs_truth_db), "INR vs truth PSNR must be finite"
    assert math.isfinite(rq.inr_ssim_vs_truth), "INR vs truth SSIM must be finite"
    assert 0.0 <= rq.inr_ssim_vs_truth <= 1.0, (
        f"INR vs truth SSIM {rq.inr_ssim_vs_truth} must be in [0, 1]"
    )
    # With low noise (0.01 sigma), PSNR should be reasonable
    assert rq.inr_psnr_vs_truth_db > 10.0, (
        f"INR vs truth PSNR {rq.inr_psnr_vs_truth_db:.1f} dB too low; "
        "model should recover structure from low-noise observations"
    )
    assert rq.inr_ssim_vs_truth > 0.3, (
        f"INR vs truth SSIM {rq.inr_ssim_vs_truth:.3f} too low"
    )


# ── Model preset validation ───────────────────────────────────────────


def test_default_perf_cases_cover_recommended_models() -> None:
    """Default benchmark cases must cover the three recommended model kinds."""
    from torchsky.map.field.registry import RECOMMENDED_FIELD_MODEL_KINDS

    covered = {c.model_kind for c in _inr_perf.DEFAULT_PERF_CASES}
    for recommended in RECOMMENDED_FIELD_MODEL_KINDS:
        assert recommended in covered, (
            f"recommended model {recommended!r} not covered by DEFAULT_PERF_CASES"
        )


def test_perf_specs_are_well_formed() -> None:
    """Every INRPerfSpec must have valid fields."""
    for spec in _inr_perf.DEFAULT_PERF_CASES:
        assert spec.steps > 0, f"{spec.name}: steps must be positive"
        assert spec.lr > 0.0, f"{spec.name}: lr must be positive"
        assert spec.patch_size > 0, f"{spec.name}: patch_size must be positive"
        assert spec.patches_per_obs > 0, (
            f"{spec.name}: patches_per_obs must be positive"
        )
        assert spec.obs_per_step > 0, f"{spec.name}: obs_per_step must be positive"
        assert spec.target_pixels[0] > 0 and spec.target_pixels[1] > 0
        assert spec.num_observation_images > 0
        assert spec.image_size[0] > 0 and spec.image_size[1] > 0
        assert spec.spectral_min < spec.spectral_max
        assert spec.noise_sigma > 0.0


def test_inr_perf_exports_public_api() -> None:
    """Benchmark module must export the expected __all__ symbols."""
    expected = {
        "DEFAULT_PERF_CASES",
        "INRPerfSpec",
        "RenderQualityComparison",
        "TrainingThroughput",
        "build_truth_image",
        "build_perf_observations",
        "estimate_domain_from_observations",
        "measure_render_quality",
        "measure_training_throughput",
    }
    actual = set(_inr_perf.__all__)
    assert expected == actual, (
        f"__all__ mismatch: missing {expected - actual}, extra {actual - expected}"
    )


# ── Coadd vs INR quality regression ───────────────────────────────────


def test_inr_coadd_comparison() -> None:
    """Coadd baseline must not regress significantly vs INR render quality.

    The coadd is built from noise-free observations while the INR trains on
    noisy data, so the coadd should perform at least comparably.  Verifies
    that the coadd PSNR is within 5 dB of the INR PSNR and SSIM within 0.2
    — catching major regressions while allowing INR to occasionally
    outperform on fine structure.
    """
    spec = _inr_perf.DEFAULT_PERF_CASES[0]
    rq = _inr_perf.measure_render_quality(spec, device="cpu")

    # A valid coadd must have been produced
    assert rq.coadd_library != "none", (
        "coadd library is 'none'; expected a real coadd baseline"
    )

    # Both INR and coadd quality metrics must be finite
    assert math.isfinite(rq.inr_psnr_vs_truth_db), "INR vs truth PSNR must be finite"
    assert math.isfinite(rq.coadd_psnr_vs_truth_db), (
        "coadd vs truth PSNR must be finite"
    )
    assert math.isfinite(rq.inr_ssim_vs_truth), "INR vs truth SSIM must be finite"
    assert math.isfinite(rq.coadd_ssim_vs_truth), "coadd vs truth SSIM must be finite"

    # SSIM bounds
    assert 0.0 <= rq.inr_ssim_vs_truth <= 1.0
    assert 0.0 <= rq.coadd_ssim_vs_truth <= 1.0

    # Coadd PSNR should not be dramatically worse than INR (5 dB tolerance)
    psnr_diff = rq.coadd_psnr_vs_truth_db - rq.inr_psnr_vs_truth_db
    assert psnr_diff > -5.0, (
        f"coadd PSNR ({rq.coadd_psnr_vs_truth_db:.1f} dB) is {abs(psnr_diff):.1f} dB "
        f"below INR PSNR ({rq.inr_psnr_vs_truth_db:.1f} dB); "
        f"diff {psnr_diff:.1f} exceeds -5.0 dB tolerance"
    )

    # Coadd SSIM should not be dramatically worse than INR (0.2 tolerance)
    ssim_diff = rq.coadd_ssim_vs_truth - rq.inr_ssim_vs_truth
    assert ssim_diff > -0.2, (
        f"coadd SSIM ({rq.coadd_ssim_vs_truth:.4f}) is {abs(ssim_diff):.4f} "
        f"below INR SSIM ({rq.inr_ssim_vs_truth:.4f}); "
        f"diff {ssim_diff:.4f} exceeds -0.2 tolerance"
    )


# ── SWarp coadd parity test ───────────────────────────────────────────


@pytest.mark.skipif(
    _inr_perf._map_compat is None or _inr_perf._map_compat._find_swarp_binary() is None,
    reason="SWarp binary not installed or not on PATH",
)
def test_inr_swarp_parity() -> None:
    """INR render vs SWarp coadd: both must produce reasonable quality vs truth.

    Skips when the SWarp binary is not found on PATH or in the external-tools
    prefix directory.
    """

    spec = _inr_perf.DEFAULT_PERF_CASES[0]
    rq = _inr_perf.measure_render_quality(spec, device="cpu")

    # SWarp must have been successfully invoked
    assert rq.coadd_library.startswith("swarp"), (
        f"expected swarp coadd, got {rq.coadd_library!r}"
    )

    # Coadd should produce finite quality metrics vs ground truth
    assert math.isfinite(rq.coadd_psnr_vs_truth_db), (
        "coadd vs truth PSNR must be finite"
    )
    assert math.isfinite(rq.coadd_ssim_vs_truth), "coadd vs truth SSIM must be finite"
    assert 0.0 <= rq.coadd_ssim_vs_truth <= 1.0, (
        f"coadd vs truth SSIM {rq.coadd_ssim_vs_truth} must be in [0, 1]"
    )

    # With noise-free coadd images, SWarp should achieve high quality
    assert rq.coadd_psnr_vs_truth_db > 10.0, (
        f"coadd vs truth PSNR {rq.coadd_psnr_vs_truth_db:.1f} dB too low; "
        "SWarp should recover structure from noise-free observations"
    )
    assert rq.coadd_ssim_vs_truth > 0.5, (
        f"coadd vs truth SSIM {rq.coadd_ssim_vs_truth:.3f} too low"
    )

    # INR should also produce finite quality metrics (already tested in
    # test_inr_render_quality_vs_truth, but verify alongside SWarp here)
    assert math.isfinite(rq.inr_psnr_vs_truth_db), "INR vs truth PSNR must be finite"
    assert 0.0 <= rq.inr_ssim_vs_truth <= 1.0, (
        f"INR vs truth SSIM {rq.inr_ssim_vs_truth} must be in [0, 1]"
    )

    # Co-add duration must be measurable (non-zero wall clock time)
    assert rq.coadd_duration_sec > 0.0, (
        f"coadd duration {rq.coadd_duration_sec} must be positive"
    )


# ── GPU smoke tests (skip when CUDA unavailable) ──────────────────────


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_inr_training_throughput_cuda() -> None:
    """INR training on CUDA should be measurable and produce correct metrics."""
    spec = _inr_perf.DEFAULT_PERF_CASES[0]
    tp = _inr_perf.measure_training_throughput(spec, device="cuda")

    assert tp.device.startswith("cuda")
    assert tp.steps_per_sec > 0.0
    assert tp.pixels_per_sec > 0.0
    assert tp.final_loss < tp.initial_loss


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_inr_render_quality_cuda() -> None:
    """INR render on CUDA should produce valid quality metrics."""
    spec = _inr_perf.DEFAULT_PERF_CASES[0]
    rq = _inr_perf.measure_render_quality(spec, device="cuda")

    assert rq.device.startswith("cuda")
    assert math.isfinite(rq.inr_psnr_db)
    assert 0.0 <= rq.inr_ssim <= 1.0
    assert math.isfinite(rq.inr_psnr_vs_truth_db)


# ── MPS smoke tests (skip when MPS unavailable) ───────────────────────


def _mps_available() -> bool:
    """Return True if the MPS backend is available."""
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


@pytest.mark.skipif(not _mps_available(), reason="MPS not available")
def test_inr_training_throughput_mps() -> None:
    """INR training on MPS should be measurable and produce correct metrics."""
    spec = _inr_perf.DEFAULT_PERF_CASES[0]
    tp = _inr_perf.measure_training_throughput(spec, device="mps")

    assert tp.device == "mps"
    assert tp.steps_per_sec > 0.0
    assert tp.pixels_per_sec > 0.0
    assert tp.final_loss < tp.initial_loss


@pytest.mark.skipif(not _mps_available(), reason="MPS not available")
def test_inr_render_quality_mps() -> None:
    """INR render on MPS should produce valid quality metrics."""
    spec = _inr_perf.DEFAULT_PERF_CASES[0]
    rq = _inr_perf.measure_render_quality(spec, device="mps")

    assert rq.device == "mps"
    assert math.isfinite(rq.inr_psnr_db)
    assert 0.0 <= rq.inr_ssim <= 1.0
    assert math.isfinite(rq.inr_psnr_vs_truth_db)
