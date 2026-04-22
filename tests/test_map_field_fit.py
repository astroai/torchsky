from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch

from torchsky.map import plan_map_product, stack_map_product


def _load_benchmark_module():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "bench_map_field_fit.py"
    spec = importlib.util.spec_from_file_location("bench_map_field_fit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load benchmark module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_field_fit_bench = _load_benchmark_module()


def test_field_fit_benchmark_step_profile_overrides_sparse_frontier_cases() -> None:
    assert (
        _field_fit_bench._steps_for_case(
            "tpv_field_fit_sparse_large_mosaic_exact_3band_multivisit",
            base_steps=8,
            step_profile="sparse_mosaic_frontier",
        )
        == 32
    )
    assert (
        _field_fit_bench._steps_for_case(
            "tpv_field_fit_sparse_large_mosaic_broad_denseoverlap",
            base_steps=8,
            step_profile="sparse_mosaic_frontier",
        )
        == 16
    )
    assert (
        _field_fit_bench._steps_for_case(
            "tpv_field_fit_sparse_large_mosaic_broad_denseoverlap",
            base_steps=8,
            step_profile="sparse_mosaic_large_frontier",
        )
        == 12
    )
    assert (
        _field_fit_bench._steps_for_case(
            "tan_field_fit",
            base_steps=8,
            step_profile="sparse_mosaic_frontier",
        )
        == 8
    )
    assert (
        _field_fit_bench._steps_for_case(
            "tan_field_fit",
            base_steps=8,
            step_profile=None,
        )
        == 8
    )
    assert (
        _field_fit_bench._warmup_steps_for_case(
            "tpv_field_fit_sparse_large_mosaic_broad_denseoverlap",
            benchmark_steps=16,
        )
        == 2
    )
    assert (
        _field_fit_bench._warmup_steps_for_case(
            "tan_field_fit",
            benchmark_steps=16,
        )
        == 4
    )


def test_field_fit_benchmark_chunked_product_render_matches_full_render() -> None:
    case_spec = _field_fit_bench._build_case_specs(24)[
        "tpv_field_fit_deep_multiband_exact"
    ]
    with _field_fit_bench.tempfile.TemporaryDirectory(
        prefix="torchsky-field-fit-"
    ) as tmpdir_str:
        observations, image_observations, _truth_cube = (
            _field_fit_bench._simulate_observations(
                case_spec,
                seed=1234,
                tmpdir=Path(tmpdir_str),
            )
        )
        product = plan_map_product(
            image_observations,
            header_overrides=case_spec.target_header,
            spectral_axis=case_spec.spectral_axis,
            spectral_bin_edges=case_spec.spectral_bin_edges,
        )
        domain = _field_fit_bench.estimate_field_domain(
            observations,
            coord_mode=case_spec.domain.coord_mode,
        )
        model = _field_fit_bench._make_field_model_for_case(
            "sky_encoder_mlp",
            domain,
            case=case_spec.case,
        )
        full = _field_fit_bench.render_field_product(model, product)
        chunked = _field_fit_bench._render_field_product_chunked(
            model,
            product,
            spatial_chunk_size=8,
        )
    assert torch.allclose(full.data, chunked.data, atol=1.0e-5, rtol=1.0e-5)
    assert torch.allclose(full.weight, chunked.weight, atol=1.0e-5, rtol=1.0e-5)


def test_field_fit_benchmark_improves_loss() -> None:
    row = _field_fit_bench.evaluate_case(
        case="tan_field_fit",
        model_kind="sine_implicit",
        size=24,
        seed=1234,
        steps=32,
        runs=1,
    )
    assert row["final_loss"] < row["initial_loss"]
    assert row["loss_improvement_ratio"] > 1.01
    assert row["reconstruction_rmse"] < 0.6
    assert row["reconstruction_p95_abs_err"] >= row["reconstruction_mae"]
    assert row["band_reconstruction_rmse_max"] >= row["band_reconstruction_rmse_mean"]
    assert row["signal_to_residual_db"] > 0.0


def test_field_fit_benchmark_supports_tan_sip_multicolor_case() -> None:
    fit_row = _field_fit_bench.evaluate_case(
        case="tan_sip_field_fit_multicolor",
        model_kind="sine_implicit",
        size=24,
        seed=1234,
        steps=40,
        runs=1,
    )
    coadd_row = _field_fit_bench.evaluate_case(
        case="tan_sip_field_fit_multicolor",
        model_kind="coadd_bicubic",
        size=24,
        seed=1234,
        steps=40,
        runs=1,
    )
    assert fit_row["projection"] == "TAN-SIP"
    assert fit_row["num_bands"] == 4
    assert fit_row["truth_model"] == "analytic_basis"
    assert fit_row["blue_red_flux_ratio_absolute_error"] >= 0.0
    assert fit_row["reconstruction_rmse"] < coadd_row["reconstruction_rmse"]


def test_field_fit_benchmark_supports_tan_sip_deep_multiband_broad_case() -> None:
    fit_row = _field_fit_bench.evaluate_case(
        case="tan_sip_field_fit_deep_multiband_broad",
        model_kind="sky_encoder_bandpass_field",
        size=24,
        seed=1234,
        steps=20,
        runs=1,
    )
    coadd_row = _field_fit_bench.evaluate_case(
        case="tan_sip_field_fit_deep_multiband_broad",
        model_kind="coadd_bilinear",
        size=24,
        seed=1234,
        steps=20,
        runs=1,
    )
    assert fit_row["projection"] == "TAN-SIP"
    assert fit_row["num_bands"] == 5
    assert fit_row["num_observations"] == 10
    assert fit_row["min_psf_sigma_pix"] == fit_row["max_psf_sigma_pix"] == 0.7
    assert fit_row["support_reference_model_kind"] == "coadd_bilinear"
    assert (
        fit_row["support_reconstruction_rmse"] <= fit_row["reconstruction_rmse"] + 0.005
    )
    assert coadd_row["support_reconstruction_rmse"] == coadd_row["reconstruction_rmse"]
    assert fit_row["uniform_reconstruction_rmse"] >= 0.0
    assert (
        coadd_row["uniform_reconstruction_rmse"]
        >= coadd_row["support_reconstruction_rmse"]
    )
    assert "fringe_extrapolated_reconstruction_rmse" in fit_row
    assert "far_extrapolated_reconstruction_rmse" in fit_row
    assert "support_single_overlap_reconstruction_rmse" in fit_row
    assert "support_multi_overlap_reconstruction_rmse" in fit_row
    assert fit_row["model_kwargs"] == {
        "hidden_dim": 160,
        "hidden_layers": 3,
        "num_fourier_frequencies": 8,
        "spectral_frequencies": 6,
        "num_bandpass_frequencies": 8,
        "bandpass_hidden_dim": 64,
    }
    assert (
        _field_fit_bench._field_fit_kwargs_for_model(
            "sky_encoder_bandpass_field",
            case="tan_sip_field_fit_deep_multiband_broad",
        )["training_spectral_sampling"]
        == "mixture_mc"
    )
    assert (
        _field_fit_bench._field_fit_kwargs_for_model(
            "sky_encoder_bandpass_field",
            case="tan_sip_field_fit_deep_multiband_broad",
        )["spectral_anchor_observations_per_step"]
        == 1
    )
    assert fit_row["uncertainty_kind"] == "native_inverse_variance"
    assert fit_row["uncertainty_extrapolated_mean_sigma"] >= (
        fit_row["uncertainty_support_mean_sigma"] - 1.0e-5
    )


def test_field_fit_benchmark_supports_sparse_large_mosaic_cases() -> None:
    tan_spec = _field_fit_bench._build_case_specs(128)[
        "tan_sip_field_fit_sparse_large_mosaic_broad"
    ]
    tpv_spec = _field_fit_bench._build_case_specs(128)[
        "tpv_field_fit_sparse_large_mosaic_exact"
    ]
    tpv_dense_spec = _field_fit_bench._build_case_specs(128)[
        "tpv_field_fit_sparse_large_mosaic_exact_densebands"
    ]
    tpv_broad_dense_spec = _field_fit_bench._build_case_specs(128)[
        "tpv_field_fit_sparse_large_mosaic_broad_denseoverlap"
    ]
    tan_fit_kwargs = _field_fit_bench._field_fit_kwargs_for_model(
        "sky_lambda_kplanes_field",
        case="tan_sip_field_fit_sparse_large_mosaic_broad",
    )
    tpv_fit_kwargs = _field_fit_bench._field_fit_kwargs_for_model(
        "sky_lambda_kplanes_field",
        case="tpv_field_fit_sparse_large_mosaic_exact",
    )
    tpv_feature_fit_kwargs = _field_fit_bench._field_fit_kwargs_for_model(
        "sky_lambda_feature_field",
        case="tpv_field_fit_sparse_large_mosaic_exact",
    )
    opcon_fit_kwargs = _field_fit_bench._field_fit_kwargs_for_model(
        "operator_constrained_feature_field",
        case="tpv_field_fit_sparse_large_mosaic_exact_3band_multivisit",
    )
    explicit_grid_fit_kwargs = _field_fit_bench._field_fit_kwargs_for_model(
        "explicit_spectral_grid_field",
        case="tpv_field_fit_sparse_large_mosaic_exact_3band_multivisit",
    )

    assert tan_spec.target_header["NAXIS1"] == 128
    assert tan_spec.observations[0].header["NAXIS1"] < tan_spec.target_header["NAXIS1"]
    assert len(tan_spec.observations) == 20
    assert tan_spec.spectral_bin_edges is not None
    assert tan_spec.domain.coord_mode == "sphere_xyz"
    assert tan_fit_kwargs["loss_kind"] == "mse"
    assert _field_fit_bench._field_model_kwargs_for_case(
        "sky_lambda_kplanes_field",
        case="tan_sip_field_fit_sparse_large_mosaic_broad",
    ) == {
        "base_resolution": 24,
        "num_scales": 4,
        "features_per_plane": 16,
        "hidden_dim": 128,
        "hidden_layers": 3,
        "refine_hidden_dim": 64,
    }
    assert _field_fit_bench._field_model_kwargs_for_case(
        "sky_lambda_feature_field",
        case="tpv_field_fit_sparse_large_mosaic_exact",
    ) == {
        "base_resolution": 48,
        "num_scales": 4,
        "features_per_plane": 24,
        "scale_hidden_dim": 192,
        "scale_num_blocks": 4,
        "decoder_hidden_dim": 192,
        "decoder_num_blocks": 4,
        "refine_hidden_dim": 96,
        "bandpass_hidden_dim": 96,
    }
    assert _field_fit_bench._field_model_kwargs_for_case(
        "operator_constrained_feature_field",
        case="tpv_field_fit_sparse_large_mosaic_exact_3band_multivisit",
    ) == {
        "base_resolution": 48,
        "num_scales": 4,
        "features_per_plane": 16,
        "scale_hidden_dim": 160,
        "scale_num_blocks": 3,
        "spectral_basis_dim": 4,
        "decoder_hidden_dim": 160,
        "decoder_num_blocks": 3,
        "refine_hidden_dim": 80,
        "bandpass_hidden_dim": 64,
        "spectral_context_dim": 96,
        "residual_hidden_dim": 64,
    }
    assert _field_fit_bench._field_model_kwargs_for_case(
        "explicit_spectral_grid_field",
        case="tpv_field_fit_sparse_large_mosaic_exact_3band_multivisit",
        size=32,
    ) == {
        "spatial_resolution": 32,
        "spectral_resolution": 8,
    }

    assert tpv_spec.target_header["NAXIS1"] == 128
    assert tpv_spec.observations[0].header["NAXIS1"] < tpv_spec.target_header["NAXIS1"]
    assert len(tpv_spec.observations) == 20
    assert tpv_spec.spectral_bin_edges is None
    assert tpv_spec.domain.coord_mode == "sphere_xyz"
    assert tpv_fit_kwargs["curriculum"] == "none"
    assert tpv_fit_kwargs["loss_reduction"] == "per_band"
    assert tpv_feature_fit_kwargs["loss_kind"] == "mse"
    assert tpv_feature_fit_kwargs["loss_reduction"] == "band_global_mix"
    assert tpv_feature_fit_kwargs["weight_decay"] == 1.0e-5
    assert opcon_fit_kwargs["loss_kind"] == "mse"
    assert opcon_fit_kwargs["loss_reduction"] == "band_global_mix"
    assert opcon_fit_kwargs["spectral_observability_weight"] == 5.0e-4
    assert opcon_fit_kwargs["residual_head_penalty_weight"] == 3.0e-6
    assert opcon_fit_kwargs["context_observations_per_step"] == 0
    assert opcon_fit_kwargs["context_loss_weight"] == 0.0
    assert explicit_grid_fit_kwargs["loss_kind"] == "mse"
    assert explicit_grid_fit_kwargs["loss_reduction"] == "global"
    assert explicit_grid_fit_kwargs["observations_per_step"] == 9
    assert (
        _field_fit_bench._field_fit_kwargs_for_model(
            "sky_lambda_feature_field_hetero",
            case="tpv_field_fit_sparse_large_mosaic_exact",
        )["loss_kind"]
        == "gaussian_nll"
    )
    assert _field_fit_bench._field_model_kwargs_for_case(
        "sky_lambda_feature_field_hetero",
        case="tpv_field_fit_sparse_large_mosaic_exact",
    ) == {
        "base_resolution": 48,
        "num_scales": 4,
        "features_per_plane": 24,
        "scale_hidden_dim": 192,
        "scale_num_blocks": 4,
        "decoder_hidden_dim": 192,
        "decoder_num_blocks": 4,
        "refine_hidden_dim": 96,
        "bandpass_hidden_dim": 96,
    }
    assert _field_fit_bench._field_model_kwargs_for_case(
        "sky_lambda_kplanes_field",
        case="tpv_field_fit_sparse_large_mosaic_exact",
    ) == {
        "base_resolution": 48,
        "num_scales": 4,
        "features_per_plane": 16,
        "hidden_dim": 128,
        "hidden_layers": 3,
        "refine_hidden_dim": 64,
    }
    assert _field_fit_bench._field_model_kwargs_for_case(
        "sky_lambda_atlas_field",
        case="tpv_field_fit_sparse_large_mosaic_exact",
    ) == {
        "atlas_side": 2,
        "atlas_sigma": 0.2,
        "base_resolution": 48,
        "num_scales": 4,
        "features_per_plane": 16,
        "hidden_dim": 128,
        "hidden_layers": 3,
        "refine_hidden_dim": 64,
    }
    assert tpv_dense_spec.target_header["NAXIS1"] == 128
    assert len(tpv_dense_spec.observations) == 18
    assert tpv_dense_spec.spectral_axis.values.numel() == 9
    assert tpv_dense_spec.domain.coord_mode == "sphere_xyz"
    assert _field_fit_bench._field_model_kwargs_for_case(
        "sky_lambda_feature_field",
        case="tpv_field_fit_sparse_large_mosaic_exact_densebands",
    ) == {
        "base_resolution": 48,
        "num_scales": 4,
        "features_per_plane": 24,
        "scale_hidden_dim": 192,
        "scale_num_blocks": 4,
        "decoder_hidden_dim": 192,
        "decoder_num_blocks": 4,
        "refine_hidden_dim": 96,
        "bandpass_hidden_dim": 128,
    }
    assert tpv_broad_dense_spec.target_header["NAXIS1"] == 128
    assert len(tpv_broad_dense_spec.observations) == 18
    assert tpv_broad_dense_spec.spectral_axis.values.numel() == 9
    assert tpv_broad_dense_spec.spectral_bin_edges is not None
    assert tpv_broad_dense_spec.domain.coord_mode == "sphere_xyz"
    assert (
        _field_fit_bench._field_fit_kwargs_for_model(
            "sky_lambda_feature_field",
            case="tpv_field_fit_sparse_large_mosaic_broad_denseoverlap",
        )["training_spectral_sampling"]
        == "mixture_mc"
    )
    tpv_broad_dense_fit_kwargs = _field_fit_bench._field_fit_kwargs_for_model(
        "sky_lambda_feature_field",
        case="tpv_field_fit_sparse_large_mosaic_broad_denseoverlap",
    )
    assert tpv_broad_dense_fit_kwargs["loss_reduction"] == "band_global_mix"
    assert tpv_broad_dense_fit_kwargs["observations_per_step"] == 9
    assert tpv_broad_dense_fit_kwargs["observation_sampling"] == "band_balanced"
    assert tpv_broad_dense_fit_kwargs["training_patch_size"] == 24
    assert tpv_broad_dense_fit_kwargs["training_patches_per_observation"] == 1
    assert _field_fit_bench._field_model_kwargs_for_case(
        "sky_lambda_feature_field",
        case="tpv_field_fit_sparse_large_mosaic_broad_denseoverlap",
    ) == {
        "base_resolution": 48,
        "num_scales": 4,
        "features_per_plane": 24,
        "scale_hidden_dim": 192,
        "scale_num_blocks": 4,
        "decoder_hidden_dim": 192,
        "decoder_num_blocks": 4,
        "refine_hidden_dim": 96,
        "bandpass_hidden_dim": 128,
    }
    assert _field_fit_bench._field_model_kwargs_for_case(
        "operator_constrained_feature_field",
        case="tpv_field_fit_sparse_large_mosaic_broad_denseoverlap",
        size=128,
    ) == {
        "base_resolution": 24,
        "num_scales": 3,
        "features_per_plane": 12,
        "scale_hidden_dim": 96,
        "scale_num_blocks": 2,
        "spectral_basis_dim": 6,
        "decoder_hidden_dim": 96,
        "decoder_num_blocks": 2,
        "refine_hidden_dim": 48,
        "bandpass_hidden_dim": 64,
        "spectral_context_dim": 64,
        "residual_hidden_dim": 48,
    }
    tpv_broad_dense_fit_kwargs_large = _field_fit_bench._field_fit_kwargs_for_model(
        "sky_lambda_feature_field",
        case="tpv_field_fit_sparse_large_mosaic_broad_denseoverlap",
        size=128,
    )
    assert tpv_broad_dense_fit_kwargs_large["loss_kind"] == "mse"
    assert tpv_broad_dense_fit_kwargs_large["loss_reduction"] == "global"
    assert (
        tpv_broad_dense_fit_kwargs_large["training_spectral_sampling"]
        == "deterministic_quadrature"
    )
    assert tpv_broad_dense_fit_kwargs_large["num_spectral_samples"] == 3
    assert tpv_broad_dense_fit_kwargs_large["observations_per_step"] == 1
    assert tpv_broad_dense_fit_kwargs_large["training_patch_size"] == 12
    assert tpv_broad_dense_fit_kwargs_large["lr"] == 1.0e-3
    tpv_broad_dense_fit_kwargs_large_long = (
        _field_fit_bench._field_fit_kwargs_for_model(
            "sky_lambda_feature_field",
            case="tpv_field_fit_sparse_large_mosaic_broad_denseoverlap",
            size=128,
            benchmark_steps=12,
        )
    )
    assert tpv_broad_dense_fit_kwargs_large_long["lr"] == 3.0e-4
    assert _field_fit_bench._field_model_kwargs_for_case(
        "sky_lambda_feature_field",
        case="tpv_field_fit_sparse_large_mosaic_broad_denseoverlap",
        size=128,
    ) == {
        "base_resolution": 24,
        "num_scales": 3,
        "features_per_plane": 12,
        "scale_hidden_dim": 96,
        "scale_num_blocks": 2,
        "decoder_hidden_dim": 96,
        "decoder_num_blocks": 2,
        "refine_hidden_dim": 48,
        "bandpass_hidden_dim": 64,
    }
    tpv_threeband_spec = _field_fit_bench._build_case_specs(128)[
        "tpv_field_fit_sparse_large_mosaic_exact_3band_multivisit"
    ]
    assert tpv_threeband_spec.target_header["NAXIS1"] == 128
    assert len(tpv_threeband_spec.observations) == 45
    assert tpv_threeband_spec.spectral_axis.values.numel() == 3
    assert tpv_threeband_spec.spectral_bin_edges is None
    opcon_large_fit_kwargs = _field_fit_bench._field_fit_kwargs_for_model(
        "operator_constrained_feature_field",
        case="tpv_field_fit_sparse_large_mosaic_broad_denseoverlap",
        size=128,
        benchmark_steps=12,
    )
    assert opcon_large_fit_kwargs["loss_kind"] == "mse"
    assert opcon_large_fit_kwargs["loss_reduction"] == "band_global_mix"
    assert opcon_large_fit_kwargs["observations_per_step"] == 1
    assert opcon_large_fit_kwargs["training_patch_size"] == 12
    assert opcon_large_fit_kwargs["lr"] == 3.0e-4


def test_field_fit_benchmark_supports_operator_constrained_feature_field() -> None:
    tpv_threeband_spec = _field_fit_bench._build_case_specs(32)[
        "tpv_field_fit_sparse_large_mosaic_exact_3band_multivisit"
    ]
    row = _field_fit_bench.evaluate_case(
        case="tpv_field_fit_sparse_large_mosaic_exact_3band_multivisit",
        model_kind="operator_constrained_feature_field",
        size=32,
        seed=1234,
        steps=4,
        runs=1,
    )
    assert row["model_kind"] == "operator_constrained_feature_field"
    assert row["support_reconstruction_rmse"] >= 0.0
    assert row["uniform_reconstruction_rmse"] >= 0.0
    assert row["model_kwargs"]["spectral_basis_dim"] == 4
    assert tpv_threeband_spec.domain.coord_mode == "sphere_xyz"
    assert (
        _field_fit_bench._field_fit_kwargs_for_model(
            "sky_lambda_feature_field",
            case="tpv_field_fit_sparse_large_mosaic_exact_3band_multivisit",
        )["loss_reduction"]
        == "global"
    )
    tpv_threeband_fit_kwargs = _field_fit_bench._field_fit_kwargs_for_model(
        "sky_lambda_feature_field",
        case="tpv_field_fit_sparse_large_mosaic_exact_3band_multivisit",
    )
    assert tpv_threeband_fit_kwargs["observations_per_step"] == 9
    assert tpv_threeband_fit_kwargs["observation_sampling"] == "band_balanced"
    assert tpv_threeband_fit_kwargs["training_patch_size"] == 24
    assert tpv_threeband_fit_kwargs["training_patches_per_observation"] == 1
    assert _field_fit_bench._field_model_kwargs_for_case(
        "sky_lambda_feature_field",
        case="tpv_field_fit_sparse_large_mosaic_exact_3band_multivisit",
    ) == {
        "base_resolution": 48,
        "num_scales": 4,
        "features_per_plane": 24,
        "scale_hidden_dim": 192,
        "scale_num_blocks": 4,
        "decoder_hidden_dim": 192,
        "decoder_num_blocks": 4,
        "refine_hidden_dim": 96,
        "bandpass_hidden_dim": 128,
    }
    assert (
        _field_fit_bench._field_fit_kwargs_for_model(
            "sky_lambda_feature_atlas_field",
            case="tpv_field_fit_sparse_large_mosaic_exact_3band_multivisit",
        )["loss_reduction"]
        == "global"
    )
    assert _field_fit_bench._field_model_kwargs_for_case(
        "sky_lambda_feature_atlas_field",
        case="tpv_field_fit_sparse_large_mosaic_exact_3band_multivisit",
    ) == {
        "atlas_side": 2,
        "atlas_sigma": 0.2,
        "topk_experts": 2,
        "base_resolution": 48,
        "num_scales": 4,
        "features_per_plane": 24,
        "scale_hidden_dim": 192,
        "scale_num_blocks": 4,
        "decoder_hidden_dim": 192,
        "decoder_num_blocks": 4,
        "refine_hidden_dim": 96,
        "bandpass_hidden_dim": 128,
    }


def test_field_fit_benchmark_reports_uncertainty_for_coadd_and_field() -> None:
    coadd_row = _field_fit_bench.evaluate_case(
        case="tpv_field_fit_deep_multiband_exact",
        model_kind="coadd_bilinear",
        size=24,
        seed=1234,
        steps=8,
        runs=1,
    )
    field_row = _field_fit_bench.evaluate_case(
        case="tpv_field_fit_deep_multiband_exact",
        model_kind="sky_encoder_mlp",
        size=24,
        seed=1234,
        steps=16,  # short fits need enough steps for support-density proxy ordering vs chance
        runs=1,
    )

    assert coadd_row["uncertainty_kind"] == "native_inverse_variance"
    assert field_row["uncertainty_kind"] == "support_density_proxy"
    assert coadd_row["uncertainty_top_bottom_abs_err_ratio"] >= 1.0
    assert field_row["uncertainty_top_bottom_abs_err_ratio"] >= 1.0


def test_field_fit_benchmark_uses_tuned_structured_broad_defaults() -> None:
    assert _field_fit_bench._field_model_kwargs_for_case(
        "sky_lambda_kplanes_field",
        case="tan_sip_field_fit_deep_multiband_broad",
    ) == {
        "base_resolution": 24,
        "num_scales": 4,
        "features_per_plane": 16,
        "hidden_dim": 128,
        "hidden_layers": 3,
        "refine_hidden_dim": 64,
    }
    assert _field_fit_bench._field_model_kwargs_for_case(
        "sky_lambda_feature_field",
        case="tpv_field_fit_deep_multiband_broad",
    ) == {
        "base_resolution": 32,
        "num_scales": 4,
        "features_per_plane": 24,
        "scale_hidden_dim": 160,
        "scale_num_blocks": 4,
        "decoder_hidden_dim": 160,
        "decoder_num_blocks": 4,
        "refine_hidden_dim": 64,
        "bandpass_hidden_dim": 96,
    }
    assert _field_fit_bench._field_model_kwargs_for_case(
        "sky_lambda_atlas_field",
        case="tpv_field_fit_deep_multiband_broad",
    ) == {
        "atlas_side": 3,
        "atlas_sigma": 0.16,
        "topk_experts": 2,
        "base_resolution": 32,
        "num_scales": 4,
        "features_per_plane": 16,
        "hidden_dim": 128,
        "hidden_layers": 3,
        "refine_hidden_dim": 64,
    }


def test_field_fit_benchmark_evaluates_sparse_large_mosaic_cases() -> None:
    tan_row = _field_fit_bench.evaluate_case(
        case="tan_sip_field_fit_sparse_large_mosaic_broad",
        model_kind="coadd_bilinear",
        size=64,
        seed=1234,
        steps=2,
        runs=1,
    )
    tpv_row = _field_fit_bench.evaluate_case(
        case="tpv_field_fit_sparse_large_mosaic_exact",
        model_kind="coadd_bilinear",
        size=64,
        seed=1234,
        steps=2,
        runs=1,
    )

    assert tan_row["num_observations"] == 20
    assert tan_row["num_bands"] == 5
    assert tan_row["projection"] == "TAN-SIP"
    assert tan_row["support_reference_model_kind"] == "coadd_bilinear"
    assert tan_row["reconstruction_rmse"] > 0.0

    assert tpv_row["num_observations"] == 20
    assert tpv_row["num_bands"] == 5
    assert tpv_row["projection"] == "TPV"
    assert tpv_row["support_reference_model_kind"] == "coadd_bilinear"
    assert tpv_row["reconstruction_rmse"] > 0.0


def test_field_fit_benchmark_supports_sky_lambda_kplanes_field() -> None:
    fit_row = _field_fit_bench.evaluate_case(
        case="tan_field_fit_exact_transmission",
        model_kind="sky_lambda_kplanes_field",
        size=24,
        seed=1234,
        steps=4,
        runs=1,
    )

    assert fit_row["model_kind"] == "sky_lambda_kplanes_field"
    assert fit_row["final_loss"] < fit_row["initial_loss"]
    assert fit_row["reconstruction_rmse"] > 0.0
    assert fit_row["support_reference_model_kind"] == "coadd_bilinear"
    assert fit_row["support_reconstruction_rmse"] <= fit_row["reconstruction_rmse"]


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


def test_field_fit_benchmark_supports_tpv_deep_multiband_broad_case() -> None:
    fit_row = _field_fit_bench.evaluate_case(
        case="tpv_field_fit_deep_multiband_broad",
        model_kind="sky_encoder_bandpass_field",
        size=24,
        seed=1234,
        steps=8,
        runs=1,
    )
    coadd_row = _field_fit_bench.evaluate_case(
        case="tpv_field_fit_deep_multiband_broad",
        model_kind="coadd_bilinear",
        size=24,
        seed=1234,
        steps=8,
        runs=1,
    )
    assert fit_row["projection"] == "TPV"
    assert fit_row["num_bands"] == 5
    assert fit_row["num_observations"] == 10
    assert fit_row["min_psf_sigma_pix"] == fit_row["max_psf_sigma_pix"] == 0.7
    assert fit_row["support_reference_model_kind"] == "coadd_bilinear"
    assert fit_row["support_reconstruction_rmse"] > 0.0
    assert coadd_row["support_reconstruction_rmse"] == coadd_row["reconstruction_rmse"]
    assert fit_row["model_kwargs"] == {
        "hidden_dim": 192,
        "hidden_layers": 3,
        "num_fourier_frequencies": 10,
        "spectral_frequencies": 6,
        "num_bandpass_frequencies": 8,
        "bandpass_hidden_dim": 64,
    }


def test_field_fit_benchmark_supports_explicit_spectral_grid_field() -> None:
    fit_row = _field_fit_bench.evaluate_case(
        case="tpv_field_fit_sparse_large_mosaic_exact_3band_multivisit",
        model_kind="explicit_spectral_grid_field",
        size=32,
        seed=1234,
        steps=4,
        runs=1,
    )

    assert fit_row["model_kind"] == "explicit_spectral_grid_field"
    assert fit_row["final_loss"] < fit_row["initial_loss"]
    assert fit_row["reconstruction_rmse"] > 0.0
    assert fit_row["support_reference_model_kind"] == "coadd_bilinear"
    assert fit_row["model_kwargs"] == {
        "spatial_resolution": 32,
        "spectral_resolution": 8,
    }


def test_field_fit_benchmark_uses_tuned_sky_encoder_resmlp_configs_for_broad_cases() -> (
    None
):
    assert _field_fit_bench._field_model_kwargs_for_case(
        "sky_encoder_resmlp",
        case="tan_sip_field_fit_deep_multiband_broad",
    ) == {
        "hidden_dim": 224,
        "num_blocks": 5,
        "num_fourier_frequencies": 10,
        "spectral_frequencies": 8,
    }
    assert _field_fit_bench._field_model_kwargs_for_case(
        "sky_encoder_resmlp",
        case="tpv_field_fit_deep_multiband_broad",
    ) == {
        "hidden_dim": 224,
        "num_blocks": 5,
        "num_fourier_frequencies": 10,
        "spectral_frequencies": 8,
        "multiscale_readout": True,
        "readout_hidden_dim": 192,
    }


def test_tan_exact_transmission_case_populates_all_target_bands_in_coadd_baseline() -> (
    None
):
    case_spec = _field_fit_bench._build_case_specs(24)[
        "tan_field_fit_exact_transmission"
    ]
    with _field_fit_bench.tempfile.TemporaryDirectory(
        prefix="torchsky-field-fit-"
    ) as tmpdir_str:
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
        result = stack_map_product(
            image_observations, product, resampling_kernel="bilinear"
        )

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
        steps=48,
        runs=1,
    )
    coadd_row = _field_fit_bench.evaluate_case(
        case="tan_field_fit_low_overlap_depth",
        model_kind="coadd_bilinear",
        size=24,
        seed=1234,
        steps=48,
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


def test_field_fit_benchmark_uses_tuned_spectral_basis_configs_for_exact_cases() -> (
    None
):
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


def test_field_fit_benchmark_uses_tuned_sky_encoder_resmlp_config_for_tpv_exact_case() -> (
    None
):
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
