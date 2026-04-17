from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

pytest.importorskip("torchsky.cpp")


def _load_benchmark_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "bench_map_multifield_truth.py"
    )
    spec = importlib.util.spec_from_file_location("bench_map_multifield_truth", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load benchmark module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_multifield_bench = _load_benchmark_module()


@pytest.mark.parametrize(
    ("case", "rmse_max", "coverage_min"),
    [
        ("tan_multifield_cube", 0.18, 0.95),
        ("tpv_multifield_cube", 0.30, 0.48),
    ],
)
def test_multifield_truth_cases_stay_bounded(
    tmp_path: Path,
    case: str,
    rmse_max: float,
    coverage_min: float,
) -> None:
    row = _multifield_bench.evaluate_torchsky_case(
        tmp_path,
        case=case,
        size=64,
        seed=1234,
        runs=1,
        resampling_kernel="bilinear",
    )
    assert row["cube_rmse_vs_truth"] <= rmse_max
    assert row["coverage_frac_mean"] >= coverage_min
    assert row["weighted_residual_rms_mean"] >= 0.0
    assert row["background_residual_rms_mean"] >= 0.0
    assert row["background_leakage_fraction_mean"] >= 0.0
    assert row["detrended_residual_rms_mean"] >= 0.0


def test_multifield_truth_emits_recommendations(tmp_path: Path) -> None:
    rows = [
        _multifield_bench.evaluate_torchsky_case(
            tmp_path,
            case="tan_multifield_cube",
            size=48,
            seed=1234,
            runs=1,
            resampling_kernel=kernel,
        )
        for kernel in ("bilinear", "catmull_rom", "mitchell_netravali")
    ]
    recs = _multifield_bench._recommendation_rows(rows)
    assert any(row["profile"] == "default_general" for row in recs)
    assert any(row["profile"] == "high_accuracy_interpolatory" for row in recs)
    assert any(row["profile"] == "smooth_reconstruction" for row in recs)


def test_multifield_truth_conservative_polygon_reports_conservative_geometry(
    tmp_path: Path,
) -> None:
    row = _multifield_bench.evaluate_torchsky_case(
        tmp_path,
        case="tan_multifield_cube",
        size=64,
        seed=1234,
        runs=1,
        resampling_kernel="conservative_polygon",
    )
    assert row["library"] == "torchsky"
    assert row["variant"] == "conservative_polygon"
    assert row["geometry_family"] == "conservative_overlap"
    assert row["coverage_frac_mean"] >= 0.9


def test_multifield_truth_supports_rectangular_variable_input_sizes(
    tmp_path: Path,
) -> None:
    row = _multifield_bench.evaluate_torchsky_case(
        tmp_path,
        case="tan_multifield_rectangular_scale",
        width=80,
        height=56,
        seed=1234,
        runs=1,
        resampling_kernel="bilinear",
    )
    assert row["target_width"] == 80
    assert row["target_height"] == 56
    assert row["min_input_width"] < row["max_input_width"]
    assert row["min_input_height"] < row["max_input_height"]
    assert row["cube_rmse_vs_truth"] <= 0.30
    assert row["background_residual_rms_max"] >= 0.0
    assert row["background_leakage_fraction_max"] >= 0.0


def test_multifield_truth_field_fit_row_is_finite_and_optimized(
    tmp_path: Path,
) -> None:
    row = _multifield_bench.evaluate_torchsky_field_case(
        tmp_path,
        case="tan_multifield_cube",
        size=24,
        seed=1234,
        runs=1,
        field_model_kind="wire_implicit",
        fit_steps=4,
        max_observations=3,
    )

    assert row["library"] == "torchsky_field"
    assert row["method"] == "field_fit"
    assert row["variant"] == "wire_implicit"
    assert row["geometry_family"] == "latent_scene_field"
    assert row["fit_observations"] == 3
    assert row["field_final_loss"] <= row["field_initial_loss"]
    assert row["coverage_frac_mean"] > 0.7
    assert row["cube_rmse_vs_truth"] >= 0.0


@pytest.mark.parametrize(
    ("spherical_mode", "geometry_family"),
    [
        ("healpix_bridge", "spherical_spectral_healpix_bridge"),
        ("harmonic_direct", "spherical_harmonic_direct"),
    ],
)
def test_multifield_truth_spherical_rows_are_finite(
    tmp_path: Path,
    spherical_mode: str,
    geometry_family: str,
) -> None:
    row = _multifield_bench.evaluate_torchsky_spherical_case(
        tmp_path,
        case="tan_multifield_cube",
        size=24,
        seed=1234,
        runs=1,
        spherical_mode=spherical_mode,
    )
    assert row["library"] == "torchsky_spherical"
    assert row["method"] == "spherical_reproject"
    assert row["variant"] == spherical_mode
    assert row["geometry_family"] == geometry_family
    assert row["operator_bundle_geometry"] == "spherical_spectral_geometry"
    assert row["healpix_nside"] >= 1
    assert row["coverage_frac_mean"] > 0.5
    assert row["cube_rmse_vs_truth"] >= 0.0


def test_multifield_truth_joint_spherical_inverse_row_is_finite_and_improves_fit(
    tmp_path: Path,
) -> None:
    row = _multifield_bench.evaluate_torchsky_multiband_spherical_inverse_case(
        tmp_path,
        case="tan_multifield_cube",
        size=24,
        seed=1234,
        runs=1,
    )

    assert row["library"] == "torchsky_spherical"
    assert row["method"] == "spherical_inverse"
    assert row["variant"] == "multiband_kronecker_prior"
    assert row["geometry_family"] == "spherical_harmonic_multiband_joint"
    assert row["operator_bundle_geometry"] == "spherical_spectral_geometry"
    assert row["joint_lmax"] >= 1
    assert row["residual_norm_final"] < row["residual_norm_initial"]
    assert row["spectral_harmonic_prior_norm"] >= 0.0
    assert row["coverage_frac_mean"] > 0.95
    assert row["cube_rmse_vs_truth"] >= 0.0


def test_multifield_random10_case_has_large_mixed_resolutions() -> None:
    spec = _multifield_bench._case_spec(
        "mixed_multifield_random10",
        width=640,
        height=512,
        seed=1234,
    )
    assert len(spec.observations) == 10
    widths = [int(obs.header["NAXIS1"]) for obs in spec.observations]
    heights = [int(obs.header["NAXIS2"]) for obs in spec.observations]
    scales = [abs(float(obs.header["CD1_1"])) for obs in spec.observations]
    assert min(widths) >= 500
    assert max(widths) <= 4000
    assert min(heights) >= 500
    assert max(heights) <= 4000
    assert min(scales) < max(scales)
