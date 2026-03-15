from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_benchmark_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "bench_map_field_fit.py"
    )
    spec = importlib.util.spec_from_file_location("bench_map_field_fit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load benchmark module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_field_fit_bench = _load_benchmark_module()


def test_field_fit_benchmark_improves_loss() -> None:
    row = _field_fit_bench.evaluate_case(
        case="tan_field_fit",
        model_kind="sine_implicit",
        size=24,
        seed=1234,
        steps=24,
        runs=1,
    )
    assert row["final_loss"] < row["initial_loss"]
    assert row["loss_improvement_ratio"] > 1.05
    assert row["reconstruction_rmse"] < 0.6
    assert row["reconstruction_p95_abs_err"] >= row["reconstruction_mae"]
    assert row["band_reconstruction_rmse_max"] >= row["band_reconstruction_rmse_mean"]
    assert row["signal_to_residual_db"] > 0.0


def test_field_fit_benchmark_supports_tan_sip_multicolor_case() -> None:
    fit_row = _field_fit_bench.evaluate_case(
        case="tan_sip_field_fit_multicolor",
        model_kind="wire_implicit",
        size=24,
        seed=1234,
        steps=20,
        runs=1,
    )
    coadd_row = _field_fit_bench.evaluate_case(
        case="tan_sip_field_fit_multicolor",
        model_kind="coadd_bicubic",
        size=24,
        seed=1234,
        steps=20,
        runs=1,
    )
    assert fit_row["projection"] == "TAN-SIP"
    assert fit_row["num_bands"] == 4
    assert fit_row["truth_model"] == "analytic_basis"
    assert fit_row["blue_red_flux_ratio_absolute_error"] >= 0.0
    assert fit_row["reconstruction_rmse"] < coadd_row["reconstruction_rmse"]


def test_field_fit_benchmark_supports_analytic_truth_and_coadd_baseline() -> None:
    fit_row = _field_fit_bench.evaluate_case(
        case="tan_field_fit_analytic",
        model_kind="sine_implicit",
        size=24,
        seed=1234,
        steps=20,
        runs=1,
    )
    coadd_row = _field_fit_bench.evaluate_case(
        case="tan_field_fit_analytic",
        model_kind="coadd_bilinear",
        size=24,
        seed=1234,
        steps=20,
        runs=1,
    )

    assert fit_row["truth_model"] == "analytic_basis"
    assert fit_row["final_loss"] < fit_row["initial_loss"]
    assert coadd_row["type"] == "cube_baseline"
    assert coadd_row["fit_method"] == "weighted_cube_coadd"
    assert coadd_row["reconstruction_rmse"] > 0.0
    assert fit_row["signal_to_residual_db"] > 0.0


def test_field_fit_benchmark_supports_psf_mismatch_case() -> None:
    row = _field_fit_bench.evaluate_case(
        case="tan_field_fit_psf_mismatch",
        model_kind="sine_implicit",
        size=24,
        seed=1234,
        steps=20,
        runs=1,
    )

    assert row["truth_model"] == "sine_implicit"
    assert row["final_loss"] < row["initial_loss"]
    assert row["reconstruction_rmse"] < 0.2


def test_field_fit_benchmark_supports_low_overlap_depth_case() -> None:
    fit_row = _field_fit_bench.evaluate_case(
        case="tan_field_fit_low_overlap_depth",
        model_kind="sine_implicit",
        size=24,
        seed=1234,
        steps=20,
        runs=1,
    )
    coadd_row = _field_fit_bench.evaluate_case(
        case="tan_field_fit_low_overlap_depth",
        model_kind="coadd_bilinear",
        size=24,
        seed=1234,
        steps=20,
        runs=1,
    )

    assert fit_row["max_noise_sigma"] > fit_row["min_noise_sigma"]
    assert fit_row["max_resolution_arcsec_pix"] > fit_row["min_resolution_arcsec_pix"]
    assert fit_row["reconstruction_rmse"] < coadd_row["reconstruction_rmse"]
    assert (
        fit_row["band_flux_relative_error_max"]
        >= fit_row["band_flux_relative_error_mean"]
    )


def test_field_fit_benchmark_emits_recommendations() -> None:
    rows = [
        _field_fit_bench.evaluate_case(
            case="tan_field_fit",
            model_kind=model_kind,
            size=24,
            seed=1234,
            steps=20,
            runs=1,
        )
        for model_kind in (
            "sine_implicit",
            "wire_implicit",
            "gaussian_basis",
            "coadd_bilinear",
            "coadd_bicubic",
        )
    ]
    recs = _field_fit_bench._recommendation_rows(rows)
    assert any(row["profile"] == "default_general" for row in recs)
    assert any(row["profile"] == "high_accuracy" for row in recs)
