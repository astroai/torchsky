from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

from torchsky.map import plan_map_product, stack_map_product


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


def test_field_fit_benchmark_supports_tpv_deep_multiband_exact_case() -> None:
    fit_row = _field_fit_bench.evaluate_case(
        case="tpv_field_fit_deep_multiband_exact",
        model_kind="sky_encoder_mlp",
        size=24,
        seed=1234,
        steps=4,
        runs=1,
    )
    coadd_row = _field_fit_bench.evaluate_case(
        case="tpv_field_fit_deep_multiband_exact",
        model_kind="coadd_bilinear",
        size=24,
        seed=1234,
        steps=4,
        runs=1,
    )

    assert fit_row["projection"] == "TPV"
    assert fit_row["num_bands"] == 5
    assert fit_row["num_observations"] == 10
    assert fit_row["truth_model"] == "analytic_basis"
    assert fit_row["min_psf_sigma_pix"] == fit_row["max_psf_sigma_pix"] == 0.7
    assert fit_row["final_loss"] < fit_row["initial_loss"]
    assert fit_row["reconstruction_rmse"] > 0.0
    assert coadd_row["reconstruction_rmse"] > 0.0
    assert fit_row["support_reference_model_kind"] == "coadd_bilinear"
    assert fit_row["support_reconstruction_rmse"] <= fit_row["reconstruction_rmse"]
    assert coadd_row["support_reconstruction_rmse"] == coadd_row["reconstruction_rmse"]
    assert len(fit_row["support_band_reconstruction_rmse"]) == fit_row["num_bands"]
    assert len(fit_row["support_band_flux_relative_error"]) == fit_row["num_bands"]
    assert len(fit_row["band_centers_angstrom"]) == fit_row["num_bands"]
    assert fit_row["support_reference_coverage_fraction"] < fit_row["coverage_fraction"]
    assert fit_row["model_kwargs"] == {
        "hidden_dim": 192,
        "hidden_layers": 3,
        "num_fourier_frequencies": 10,
        "spectral_frequencies": 6,
    }
    assert coadd_row["model_kwargs"] == {}


def test_tan_exact_transmission_case_populates_all_target_bands_in_coadd_baseline() -> None:
    case_spec = _field_fit_bench._build_case_specs(24)["tan_field_fit_exact_transmission"]
    with _field_fit_bench.tempfile.TemporaryDirectory(prefix="torchsky-field-fit-") as tmpdir_str:
        _, image_observations, _ = _field_fit_bench._simulate_observations(
            case_spec,
            seed=1234,
            tmpdir=Path(tmpdir_str),
        )
        product = plan_map_product(
            image_observations,
            header_overrides=case_spec.target_header,
            spectral_axis=case_spec.spectral_axis,
            spectral_bin_edges=case_spec.spectral_bin_edges,
        )
        result = stack_map_product(image_observations, product, resampling_kernel="bilinear")

    band_weight_sum = result.weight.sum(dim=(1, 2))
    assert len(case_spec.observations) == 6
    assert tuple(band.label for band in product.bands) == ("4800A", "5600A", "6400A")
    assert bool((band_weight_sum > 0.0).all())


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
    assert all("support_reconstruction_rmse" in row for row in recs)
    assert all("support_reconstruction_rmse" in row["criterion"] for row in recs)


def test_field_fit_benchmark_markdown_surfaces_support_metrics() -> None:
    rows = [
        _field_fit_bench.evaluate_case(
            case="tpv_field_fit_deep_multiband_exact",
            model_kind=model_kind,
            size=24,
            seed=1234,
            steps=4,
            runs=1,
        )
        for model_kind in (
            "sky_encoder_mlp",
            "coadd_bilinear",
        )
    ]
    rows.extend(_field_fit_bench._recommendation_rows(rows))
    markdown = _field_fit_bench.write_map_benchmark_artifacts(rows)
    assert "best support accuracy" in markdown
    assert "support_reconstruction_rmse" not in markdown


def test_field_fit_benchmark_uses_tuned_spectral_basis_configs_for_exact_cases() -> None:
    tan_row = _field_fit_bench.evaluate_case(
        case="tan_field_fit_exact_transmission",
        model_kind="spectral_basis_field",
        size=24,
        seed=1234,
        steps=8,
        runs=1,
    )
    tpv_row = _field_fit_bench.evaluate_case(
        case="tpv_field_fit_deep_multiband_exact",
        model_kind="spectral_basis_field",
        size=24,
        seed=1234,
        steps=8,
        runs=1,
    )
    assert tan_row["model_kwargs"] == {
        "hidden_dim": 128,
        "hidden_layers": 2,
        "num_spatial_frequencies": 8,
        "num_spectral_frequencies": 6,
    }
    assert tpv_row["model_kwargs"] == {
        "hidden_dim": 128,
        "hidden_layers": 2,
        "num_spatial_frequencies": 6,
        "num_spectral_frequencies": 4,
    }


def test_field_fit_benchmark_uses_tuned_sky_encoder_configs_for_exact_cases() -> None:
    tan_row = _field_fit_bench.evaluate_case(
        case="tan_field_fit_exact_transmission",
        model_kind="sky_encoder_mlp",
        size=24,
        seed=1234,
        steps=8,
        runs=1,
    )
    tpv_row = _field_fit_bench.evaluate_case(
        case="tpv_field_fit_deep_multiband_exact",
        model_kind="sky_encoder_mlp",
        size=24,
        seed=1234,
        steps=8,
        runs=1,
    )
    assert tan_row["model_kwargs"] == {
        "hidden_dim": 128,
        "hidden_layers": 2,
        "num_fourier_frequencies": 6,
        "spectral_frequencies": 6,
    }
    assert tpv_row["model_kwargs"] == {
        "hidden_dim": 192,
        "hidden_layers": 3,
        "num_fourier_frequencies": 10,
        "spectral_frequencies": 6,
    }


def test_field_fit_benchmark_uses_tuned_sky_encoder_resmlp_config_for_tpv_exact_case() -> None:
    tpv_row = _field_fit_bench.evaluate_case(
        case="tpv_field_fit_deep_multiband_exact",
        model_kind="sky_encoder_resmlp",
        size=24,
        seed=1234,
        steps=8,
        runs=1,
    )
    assert tpv_row["model_kwargs"] == {
        "hidden_dim": 160,
        "num_blocks": 4,
        "num_fourier_frequencies": 10,
        "spectral_frequencies": 4,
    }
