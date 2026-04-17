from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from astropy.io import fits
import torchsky


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

import bench_map_compat as bench_map_compat  # noqa: E402
import bench_map_ground_truth as bench_map_ground_truth  # noqa: E402
import bench_map_projection_mosaic_suite as projection_mosaic_suite  # noqa: E402
import bench_map_tan_sip_suite as tan_sip_suite  # noqa: E402
import run_ground_truth_competitive_large_union as competitive_runner  # noqa: E402

from torchsky.map import ImageObservation, plan_image_plane_target  # noqa: E402


def test_plan_image_plane_target_matches_montage_style_template(tmp_path) -> None:
    """Montage ``mProject`` templates should round-trip to the same WCS grid we plan."""
    header = {
        "NAXIS": 2,
        "NAXIS1": 32,
        "NAXIS2": 24,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 16.0,
        "CRPIX2": 12.0,
        "CRVAL1": 11.0,
        "CRVAL2": -4.0,
        "CD1_1": -0.01,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": 0.01,
    }
    path_in = tmp_path / "in.fits"
    torchsky.write(
        str(path_in),
        torch.ones((24, 32), dtype=torch.float32),
        header=header,
        overwrite=True,
    )
    tpl_path = tmp_path / "target.hdr"
    bench_map_compat._write_header_template(tpl_path, header)
    tpl_hdr = fits.Header.fromtextfile(str(tpl_path))

    planned = plan_image_plane_target(
        [ImageObservation(str(path_in))],
        header_overrides=dict(tpl_hdr),
    )
    w_plan = torchsky.WCS(planned.header)
    w_tpl = torchsky.WCS(dict(tpl_hdr))
    for px, py in ((1.0, 1.0), (16.0, 12.0), (32.0, 24.0)):
        tpx = torch.tensor(px, dtype=torch.float64)
        tpy = torch.tensor(py, dtype=torch.float64)
        ra_p, dec_p = w_plan.pixel_to_world(tpx, tpy)
        ra_t, dec_t = w_tpl.pixel_to_world(tpx, tpy)
        torch.testing.assert_close(ra_p, ra_t, atol=1e-6, rtol=1e-9)
        torch.testing.assert_close(dec_p, dec_t, atol=1e-6, rtol=1e-9)


def test_plan_image_plane_target_matches_montage_template_cdelta_only(tmp_path) -> None:
    """Montage-style headers often use ``CDELT*`` without an explicit ``CD`` matrix."""
    header = {
        "NAXIS": 2,
        "NAXIS1": 28,
        "NAXIS2": 20,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 14.0,
        "CRPIX2": 10.0,
        "CRVAL1": 45.0,
        "CRVAL2": -22.0,
        "CDELT1": -0.008,
        "CDELT2": 0.008,
    }
    path_in = tmp_path / "in_cdelta.fits"
    torchsky.write(
        str(path_in),
        torch.ones((20, 28), dtype=torch.float32),
        header=header,
        overwrite=True,
    )
    tpl_path = tmp_path / "target_cdelta.hdr"
    bench_map_compat._write_header_template(tpl_path, header)
    tpl_hdr = fits.Header.fromtextfile(str(tpl_path))
    planned = plan_image_plane_target(
        [ImageObservation(str(path_in))],
        header_overrides=dict(tpl_hdr),
    )
    w_plan = torchsky.WCS(planned.header)
    w_tpl = torchsky.WCS(dict(tpl_hdr))
    for px, py in ((1.0, 1.0), (14.0, 10.0), (28.0, 20.0)):
        tpx = torch.tensor(px, dtype=torch.float64)
        tpy = torch.tensor(py, dtype=torch.float64)
        ra_p, dec_p = w_plan.pixel_to_world(tpx, tpy)
        ra_t, dec_t = w_tpl.pixel_to_world(tpx, tpy)
        torch.testing.assert_close(ra_p, ra_t, atol=1e-6, rtol=1e-9)
        torch.testing.assert_close(dec_p, dec_t, atol=1e-6, rtol=1e-9)


def test_montage_header_template_includes_core_fits_cards(tmp_path):
    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = 32
    header["NAXIS2"] = 24
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CRPIX1"] = 16.0
    header["CRPIX2"] = 12.0
    header["CRVAL1"] = 11.0
    header["CRVAL2"] = -4.0
    header["CD1_1"] = -0.01
    header["CD1_2"] = 0.0
    header["CD2_1"] = 0.0
    header["CD2_2"] = 0.01

    path = tmp_path / "target.hdr"
    bench_map_compat._write_header_template(path, dict(header))
    text = path.read_text()

    assert "SIMPLE" in text
    assert "BITPIX" in text
    assert "NAXIS   =                    2" in text


def test_build_compat_markdown_includes_geometry_family_columns() -> None:
    markdown = bench_map_compat._build_compat_markdown(
        [
            {
                "case": "demo",
                "library": "torchsky",
                "variant": "bilinear",
                "geometry_family": "interpolation",
                "status": "OK",
                "time_ms": 1.0,
                "max_abs_data_diff_vs_torchsky": None,
            },
            {
                "case": "demo",
                "library": "reproject_interp",
                "variant": "bilinear",
                "geometry_family": "interpolation",
                "status": "OK",
                "time_ms": 2.0,
                "max_abs_data_diff_vs_torchsky": 0.01,
            },
            {
                "case": "demo",
                "library": "pairwise",
                "pair": "torchsky/reproject_interp",
                "status": "OK",
                "time_ms": 0.0,
                "max_abs_data_diff": 0.02,
            },
            {
                "case": "demo",
                "library": "conservative_parity",
                "status": "OK",
                "geometry_family": "conservative_overlap",
                "max_abs_data_diff": 0.003,
                "max_abs_weight_diff": 0.0,
            },
        ]
    )
    assert "| case | fastest backend | fastest geometry | closest external to torchsky | closest geometry | conservative parity |" in markdown
    assert "| demo | torchsky:bilinear (1.000 ms) | interpolation | reproject_interp:bilinear (0.01) | interpolation | data 0.003, support 0 |" in markdown


def test_compat_default_torchsky_kernels_include_conservative_overlap() -> None:
    assert "conservative_polygon" in bench_map_compat._TORCHSKY_KERNELS
    assert (
        bench_map_compat._geometry_family_for_torchsky_kernel("conservative_polygon")
        == "conservative_overlap"
    )


def test_conservative_overlap_parity_row_compares_support_masks() -> None:
    conservative = {
        "time_ms": 2.0,
    }
    reproject_exact = {
        "time_ms": 4.0,
        "variant": "exact",
    }
    data = torch.tensor([[1.0, 2.0], [0.0, 3.0]], dtype=torch.float32)
    reference = torch.tensor([[1.0, 2.1], [0.0, 3.0]], dtype=torch.float32)
    weight = torch.tensor([[1.0, 0.5], [0.0, 0.2]], dtype=torch.float32)
    reference_weight = torch.tensor([[2.0, 1.0], [0.0, 0.8]], dtype=torch.float32)

    row = bench_map_compat._conservative_overlap_parity_row(
        "demo",
        conservative,
        data,
        weight,
        reproject_exact,
        reference,
        reference_weight,
    )

    assert row["library"] == "conservative_parity"
    assert row["geometry_family"] == "conservative_overlap"
    assert row["comparison_weight_mode"] == "support_mask"
    assert row["max_abs_weight_diff"] == 0.0
    assert row["max_abs_data_diff"] > 0.0


def test_competitive_runner_uses_rectangular_extent_for_case_size() -> None:
    assert competitive_runner._effective_case_size(512, None, None) == 512
    assert competitive_runner._effective_case_size(512, 1024, 768) == 1024
    assert competitive_runner._effective_case_size(512, 2048, 1536) == 2048


def test_tan_sip_suite_parses_extent_specs() -> None:
    assert tan_sip_suite._parse_extent_specs("512x384, 1024x768") == [
        (512, 384),
        (1024, 768),
    ]


def test_tan_sip_suite_parses_diversity_case_specs() -> None:
    assert tan_sip_suite._parse_diversity_case_specs(
        "tan_multires_noise@256,tan_random10_large_union_rect_weighted_masked_tan@512x384"
    ) == [
        tan_sip_suite._DiversityCase(case="tan_multires_noise", size=256, width=None, height=None),
        tan_sip_suite._DiversityCase(
            case="tan_random10_large_union_rect_weighted_masked_tan",
            size=512,
            width=512,
            height=384,
        ),
    ]


def test_tan_sip_suite_parses_method_specs() -> None:
    assert tan_sip_suite._parse_method_specs("torchsky:bilinear,montage,swarp:bilinear") == [
        ("torchsky", "bilinear"),
        ("montage", None),
        ("swarp", "bilinear"),
    ]


def test_tan_sip_suite_quick_profile_defaults_are_routine_sized() -> None:
    profile = tan_sip_suite._resolve_profile("quick")

    assert profile.tan_extents == ((512, 384), (1024, 768))
    assert profile.diversity_cases[0] == "tan_aligned_low_noise@256"
    assert profile.diversity_cases[-1] == "tan_random10_large_union_rect_weighted_masked_tan@512x384"
    assert "torchsky:lanczos3" in profile.diversity_methods
    assert "swarp:lanczos3" in profile.diversity_methods
    assert profile.method_sweep_case == "tan_random10_large_union_rect_weighted_masked_tan"
    assert profile.method_sweep_size == 512
    assert profile.runs == 1
    assert profile.torchsky_runs == 2
    assert "torchsky:bilinear" in profile.tan_methods
    assert "drizzle:point" in profile.sip_methods


def test_tan_sip_suite_full_profile_defaults_include_graduation_extent() -> None:
    profile = tan_sip_suite._resolve_profile("full")

    assert profile.tan_extents[-1] == (2048, 1536)
    assert profile.diversity_cases[-1] == "tan_random10_large_union_rect_weighted_masked_tan@1024x768"
    assert profile.method_sweep_case == "tan_random10_large_union_rect_weighted_masked_tan"
    assert profile.method_sweep_size == 1024
    assert profile.torchsky_runs == 3
    assert "drizzle:square" in profile.sip_methods


def test_projection_mosaic_suite_parses_projection_cases() -> None:
    assert projection_mosaic_suite._parse_projection_case_specs(
        "SIN:tan_aligned_low_noise@768,CEA:tan_random10_large_union_rect_weighted_masked_tan@1024x640"
    ) == [
        projection_mosaic_suite._ProjectionCase(
            projection="SIN",
            source_case="tan_aligned_low_noise",
            size=768,
            width=None,
            height=None,
        ),
        projection_mosaic_suite._ProjectionCase(
            projection="CEA",
            source_case="tan_random10_large_union_rect_weighted_masked_tan",
            size=1024,
            width=1024,
            height=640,
        ),
    ]


def test_projection_mosaic_suite_parses_stack_vs_inr_cases() -> None:
    assert projection_mosaic_suite._parse_stack_vs_inr_case_specs(
        "CEA:tan_field_fit@768x512,AIT:tan_sip_field_fit_multicolor@1024x512"
    ) == [
        projection_mosaic_suite._StackVsINRCase(
            projection="CEA",
            field_case="tan_field_fit",
            size=768,
            width=768,
            height=512,
        ),
        projection_mosaic_suite._StackVsINRCase(
            projection="AIT",
            field_case="tan_sip_field_fit_multicolor",
            size=1024,
            width=1024,
            height=512,
        ),
    ]


def test_projection_mosaic_suite_parses_inr_model_specs() -> None:
    assert projection_mosaic_suite._parse_inr_model_spec("sine_implicit@sphere_xyz") == (
        "sine_implicit",
        "sphere_xyz",
    )
    assert projection_mosaic_suite._parse_inr_model_spec("fourier_mlp@sphere_xyz") == (
        "fourier_mlp",
        "sphere_xyz",
    )
    assert projection_mosaic_suite._parse_inr_model_spec("sky_encoder_mlp@sphere_xyz") == (
        "sky_encoder_mlp",
        "sphere_xyz",
    )
    assert projection_mosaic_suite._parse_inr_model_spec("sky_encoder_resmlp@sphere_xyz") == (
        "sky_encoder_resmlp",
        "sphere_xyz",
    )
    assert projection_mosaic_suite._parse_inr_model_spec("hashgrid_mlp@sphere_xyz") == (
        "hashgrid_mlp",
        "sphere_xyz",
    )
    assert projection_mosaic_suite._parse_inr_model_spec("coadd_bilinear") == (
        "coadd_bilinear",
        None,
    )


def test_projection_mosaic_suite_quick_profile_covers_projection_breadth() -> None:
    profile = projection_mosaic_suite._resolve_profile("quick")

    assert profile.truth_cases[0] == "SIN:tan_aligned_low_noise@768"
    assert "AIT:tan_random10_large_union@1024x512" in profile.truth_cases
    assert "ZEA:tan_low_overlap_exptime_transmission@768" in profile.truth_cases
    assert "STG:tan_random10_large_union@768" in profile.truth_cases
    assert "MER:tan_multires_noise@1024x640" in profile.truth_cases
    assert "MOL:tan_random10_large_union@1024x512" in profile.truth_cases
    assert len(profile.truth_cases) == 8
    assert profile.equal_psf_cases == (
        "SIN:tan_aligned_low_noise@768",
        "ARC:tan_aligned_low_noise@768",
        "AIT:tan_aligned_low_noise@1024x512",
    )
    assert profile.method_sweep_cases == (
        "CEA:tan_random10_large_union_rect_weighted_masked_tan@1024x640",
        "AIT:tan_random10_large_union@1024x512",
    )
    assert profile.method_sweep_kernels == ("bilinear", "lanczos3")
    assert profile.method_sweep_methods == ("weighted_mean", "mean", "median")
    assert "montage" in ",".join(profile.montage_methods)
    assert profile.montage_cases == (
        "CEA:tan_multires_noise@1024x640",
        "AIT:tan_multires_noise@1024x512",
    )
    assert profile.stack_vs_inr_cases == (
        "CEA:tan_field_fit@768x512",
        "AIT:tan_sip_field_fit_multicolor@768x512",
    )
    assert profile.stack_vs_inr_models == (
        "coadd_bilinear",
        "coadd_lanczos3",
        "wire_implicit@sphere_xyz",
        "sky_encoder_mlp@sphere_xyz",
        "hashgrid_mlp@sphere_xyz",
    )
    assert profile.field_steps == 8


def test_projection_mosaic_suite_full_profile_expands_to_all_families() -> None:
    profile = projection_mosaic_suite._resolve_profile("full")

    assert len(profile.truth_cases) == 40
    assert profile.truth_cases[-1] == "MOL:tan_random10_large_union_rect_weighted_masked_tan@1536x768"
    assert profile.equal_psf_cases == (
        "SIN:tan_aligned_low_noise@768",
        "ARC:tan_aligned_low_noise@768",
        "STG:tan_aligned_low_noise@768",
        "CEA:tan_aligned_low_noise@1024x640",
        "MER:tan_aligned_low_noise@1024x640",
        "AIT:tan_aligned_low_noise@1024x512",
        "MOL:tan_aligned_low_noise@1024x512",
    )
    assert profile.method_sweep_cases == (
        "ARC:tan_multires_noise@768",
        "STG:tan_random10_large_union@1536",
        "CEA:tan_random10_large_union_rect_weighted_masked_tan@1536x768",
        "MER:tan_random10_large_union_rect_weighted_masked_tan@1536x768",
        "AIT:tan_random10_large_union@1536x768",
        "MOL:tan_random10_large_union@1536x768",
    )
    assert profile.method_sweep_methods == ("weighted_mean", "mean", "median")
    assert "AIT:tan_sip_field_fit_multicolor@1024x512" in profile.stack_vs_inr_cases
    assert "wire_implicit@sphere_xyz" in profile.stack_vs_inr_models
    assert "fourier_mlp@sphere_xyz" in profile.stack_vs_inr_models
    assert "sky_encoder_mlp@sphere_xyz" in profile.stack_vs_inr_models
    assert "hashgrid_mlp@sphere_xyz" in profile.stack_vs_inr_models
    assert profile.field_steps == 36


def test_projection_mosaic_suite_projection_header_sets_expected_cards() -> None:
    cea = projection_mosaic_suite._projection_target_header(
        "CEA",
        width=1024,
        height=640,
        crval1=12.0,
        crval2=-3.0,
        scale_deg=0.02,
    )
    ait = projection_mosaic_suite._projection_target_header(
        "AIT",
        width=1024,
        height=512,
        crval1=12.0,
        crval2=-3.0,
        scale_deg=0.05,
    )

    assert cea["CTYPE1"] == "RA---CEA"
    assert cea["CTYPE2"] == "DEC--CEA"
    assert cea["PV2_1"] == 1.0
    assert cea["NAXIS1"] == 1024
    assert cea["NAXIS2"] == 640
    assert ait["CTYPE1"] == "RA---AIT"
    assert ait["CTYPE2"] == "DEC--AIT"
    assert "PV2_1" not in ait


def test_projection_mosaic_suite_support_region_masks_split_positive_weights() -> None:
    weight = torch.tensor(
        [[0.0, 1.0, 2.0], [3.0, 4.0, 8.0]],
        dtype=torch.float32,
    )

    masks = projection_mosaic_suite._support_region_masks(weight)

    fringe = masks["support_fringe_q25"]
    core = masks["support_core_q75"]

    assert fringe.dtype == torch.bool
    assert core.dtype == torch.bool
    assert fringe.shape == weight.shape
    assert core.shape == weight.shape
    assert bool(fringe[0, 0]) is False
    assert bool(core[0, 0]) is False
    assert bool(fringe[0, 1]) is True
    assert bool(core[1, 2]) is True


def test_projection_mosaic_suite_parse_name_list_requires_values() -> None:
    assert projection_mosaic_suite._parse_name_list("bilinear, lanczos3", label="kernel") == [
        "bilinear",
        "lanczos3",
    ]
    with pytest.raises(ValueError, match="at least one kernel is required"):
        projection_mosaic_suite._parse_name_list(" , ", label="kernel")


def test_projection_mosaic_suite_annotates_stack_vs_inr_against_best_direct() -> None:
    rows = [
        {
            "suite": "stack_vs_inr",
            "case": "CEA:tan_field_fit@768x512",
            "projection": "CEA",
            "type": "direct_stack",
            "model_kind": "coadd_bilinear",
            "status": "OK",
            "fit_time_ms": 10.0,
            "reconstruction_rmse": 0.20,
        },
        {
            "suite": "stack_vs_inr",
            "case": "CEA:tan_field_fit@768x512",
            "projection": "CEA",
            "type": "direct_stack",
            "model_kind": "coadd_lanczos3",
            "status": "OK",
            "fit_time_ms": 15.0,
            "reconstruction_rmse": 0.15,
        },
        {
            "suite": "stack_vs_inr",
            "case": "CEA:tan_field_fit@768x512",
            "projection": "CEA",
            "type": "field_fit",
            "model_kind": "sine_implicit",
            "status": "OK",
            "fit_time_ms": 100.0,
            "reconstruction_rmse": 0.12,
        },
    ]

    annotated = projection_mosaic_suite._annotate_stack_vs_inr_rows(rows)
    by_model = {row["model_kind"]: row for row in annotated}

    assert by_model["coadd_bilinear"]["best_direct_model_kind"] == "coadd_lanczos3"
    assert by_model["coadd_bilinear"]["best_direct_reconstruction_rmse"] == pytest.approx(0.15)
    assert by_model["coadd_bilinear"]["rmse_delta_vs_best_direct"] == pytest.approx(0.05)
    assert by_model["coadd_bilinear"]["fit_time_ratio_vs_best_direct"] == pytest.approx(10.0 / 15.0)
    assert by_model["sine_implicit"]["best_inr_model_kind"] == "sine_implicit"
    assert by_model["sine_implicit"]["best_inr_reconstruction_rmse"] == pytest.approx(0.12)
    assert by_model["sine_implicit"]["rmse_delta_vs_best_direct"] == pytest.approx(-0.03)
    assert by_model["sine_implicit"]["rmse_delta_vs_best_inr"] == pytest.approx(0.0)


def test_projection_mosaic_suite_markdown_includes_inr_common_psf_columns() -> None:
    markdown = projection_mosaic_suite._format_markdown(
        [
            {
                "suite": "stack_vs_inr",
                "case": "CEA:tan_field_fit@768x512",
                "projection": "CEA",
                "field_case": "tan_field_fit",
                "n_obs": 3,
                "num_bands": 1,
                "model_kind": "sine_implicit",
                "field_coord_mode": "sphere_xyz",
                "type": "field_fit",
                "status": "OK",
                "target_width": 768,
                "target_height": 512,
                "common_psf_sigma_pix": 0.9,
                "observation_psf_mode": "implicit_shared",
                "evaluation_psf_mode": "common_shared",
                "fit_time_ms": 12.0,
                "reconstruction_rmse": 0.1,
                "reconstruction_mae": 0.08,
                "reconstruction_p95_abs_err": 0.2,
                "reconstruction_max_abs_err": 0.4,
                "best_direct_model_kind": "coadd_bilinear",
                "best_direct_reconstruction_rmse": 0.12,
                "rmse_delta_vs_best_direct": -0.02,
                "fit_time_ratio_vs_best_direct": 20.0,
                "best_inr_model_kind": "sine_implicit",
                "best_inr_reconstruction_rmse": 0.1,
                "rmse_delta_vs_best_inr": 0.0,
                "signal_to_residual_db": 20.0,
                "total_flux_relative_error": 0.01,
                "coverage_fraction": 1.0,
            }
        ],
        profile="quick",
    )

    assert "## INR Common-PSF" in markdown
    assert "field_coord_mode" in markdown
    assert "common_psf_sigma_pix" in markdown


def test_diff_summary_reports_common_support_metrics_for_coverage_masks() -> None:
    data = torch.tensor([[1.0, 3.0], [0.0, 8.0]], dtype=torch.float32)
    weight = torch.tensor([[1.0, 0.0], [2.0, 4.0]], dtype=torch.float32)
    ref_data = torch.tensor([[1.5, 0.0], [2.0, 10.0]], dtype=torch.float32)
    ref_weight = torch.tensor([[3.0, 0.0], [0.0, 9.0]], dtype=torch.float32)
    mask = torch.tensor([[True, False], [False, True]])

    summary = bench_map_compat._diff_summary(
        data,
        weight,
        ref_data,
        ref_weight,
        suffix="_vs_ref",
        data_mask=mask,
        compare_weight_as_mask=True,
    )

    assert summary["comparison_support_frac_vs_ref"] == pytest.approx(0.5)
    assert summary["rmse_data_diff_vs_ref"] == pytest.approx((2.125) ** 0.5)
    assert summary["mae_data_diff_vs_ref"] == pytest.approx(1.25)
    assert summary["rmse_weight_diff_vs_ref"] == pytest.approx((0.25) ** 0.5)
    assert summary["mae_weight_diff_vs_ref"] == pytest.approx(0.25)
    assert summary["max_abs_data_diff_vs_ref"] == pytest.approx(2.0)
    assert summary["max_abs_weight_diff_vs_ref"] == pytest.approx(1.0)


def test_align_array_to_target_falls_back_to_torchsky_when_reproject_missing(monkeypatch) -> None:
    header = {
        "NAXIS": 2,
        "NAXIS1": 4,
        "NAXIS2": 4,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 2.5,
        "CRPIX2": 2.5,
        "CRVAL1": 10.0,
        "CRVAL2": -2.0,
        "CD1_1": -1.0e-4,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": 1.0e-4,
    }
    target_header = dict(header)
    target_header["CRPIX1"] = 3.0
    target_header["CRPIX2"] = 3.0

    array = np.arange(16, dtype=np.float32).reshape(4, 4)
    monkeypatch.setattr(bench_map_compat, "reproject_interp", None)

    aligned = bench_map_compat._align_array_to_target(
        array,
        header=header,
        target_header=target_header,
        order="bilinear",
    )

    assert aligned.shape == (4, 4)
    assert np.isfinite(aligned).all()


def test_tan_rect_weighted_masked_tan_case_stays_pure_tan(tmp_path: Path) -> None:
    observations, target_header, _truth = bench_map_ground_truth.make_case(
        tmp_path,
        case="tan_random10_large_union_rect_weighted_masked_tan",
        size=512,
        width=512,
        height=384,
        seed=1234,
    )

    assert target_header["NAXIS1"] == 512
    assert target_header["NAXIS2"] == 384
    assert observations
    assert all("TPV" not in str(item.header.get("CTYPE1", "")) for item in observations)


def test_dfreproject_validation_skips_pv_distorted_source_headers() -> None:
    source = fits.Header()
    source["NAXIS"] = 2
    source["NAXIS1"] = 32
    source["NAXIS2"] = 24
    source["CTYPE1"] = "RA---TPV"
    source["CTYPE2"] = "DEC--TPV"
    source["CRPIX1"] = 16.0
    source["CRPIX2"] = 12.0
    source["CRVAL1"] = 11.0
    source["CRVAL2"] = -4.0
    source["CD1_1"] = -0.01
    source["CD1_2"] = 0.0
    source["CD2_1"] = 0.0
    source["CD2_2"] = 0.01
    source["PV1_1"] = 1.0
    source["PV2_2"] = 1.0

    target = fits.Header()
    target["NAXIS"] = 2
    target["NAXIS1"] = 32
    target["NAXIS2"] = 24
    target["CTYPE1"] = "RA---TAN"
    target["CTYPE2"] = "DEC--TAN"
    target["CRPIX1"] = 16.0
    target["CRPIX2"] = 12.0
    target["CRVAL1"] = 11.0
    target["CRVAL2"] = -4.0
    target["CD1_1"] = -0.01
    target["CD1_2"] = 0.0
    target["CD2_1"] = 0.0
    target["CD2_2"] = 0.01

    with pytest.raises(RuntimeError, match="PV/TPV-distorted source grids"):
        bench_map_compat._validate_dfreproject_headers(dict(source), dict(target))


def test_dfreproject_validation_allows_plain_tan_headers() -> None:
    source = {
        "NAXIS": 2,
        "NAXIS1": 32,
        "NAXIS2": 24,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 16.0,
        "CRPIX2": 12.0,
        "CRVAL1": 11.0,
        "CRVAL2": -4.0,
        "CD1_1": -0.01,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": 0.01,
    }
    target = dict(source)

    bench_map_compat._validate_dfreproject_headers(source, target)


def test_known_large_union_boundary_hotspot_reproduces_against_swarp(tmp_path: Path) -> None:
    if bench_map_compat._find_swarp_binary() is None:
        pytest.skip("SWarp CLI is unavailable")

    observations, target_header, _truth = bench_map_ground_truth.make_case(
        tmp_path,
        case="tan_random10_large_union_rect_weighted_masked",
        size=1024,
        width=1024,
        height=768,
        seed=1234,
    )
    tan_items = [
        item
        for item in observations
        if "TPV" not in str(item.header.get("CTYPE1", ""))
    ]

    hotspot_x = 480
    hotspot_y = 654
    target_wcs = bench_map_ground_truth.WCS(target_header)
    ra, dec = target_wcs.pixel_to_world(
        torch.tensor([hotspot_x], dtype=torch.float64),
        torch.tensor([hotspot_y], dtype=torch.float64),
    )

    hotspot_item = tan_items[3]
    stripped = bench_map_ground_truth.ImageObservation(
        hotspot_item.observation.path,
        hdu=hotspot_item.observation.hdu,
        weight=hotspot_item.observation.weight,
        photometric_scale=hotspot_item.observation.photometric_scale,
        background_offset=hotspot_item.observation.background_offset,
        wavelength=hotspot_item.observation.wavelength,
        spectral_min=hotspot_item.observation.spectral_min,
        spectral_max=hotspot_item.observation.spectral_max,
        transmission_center=hotspot_item.observation.transmission_center,
        transmission_sigma=hotspot_item.observation.transmission_sigma,
        band_name=hotspot_item.observation.band_name,
        weight_path=hotspot_item.observation.weight_path,
        weight_hdu=hotspot_item.observation.weight_hdu,
        mask_path=None,
        mask_hdu=hotspot_item.observation.mask_hdu,
    )
    source_wcs = bench_map_ground_truth.WCS(hotspot_item.header)
    x_src, y_src = source_wcs.world_to_pixel(ra, dec)

    torchsky_result = bench_map_ground_truth.image_plane_mod.coadd_image_plane(
        [stripped],
        target_header=target_header,
        resampling_kernel="bilinear",
        value_semantics="surface_brightness",
    )
    swarp_data_np, swarp_weight_np = bench_map_compat._coadd_with_swarp(
        [stripped],
        target_header=target_header,
        resampling_type="BILINEAR",
    )
    swarp_weight = torch.from_numpy(swarp_weight_np).to(dtype=torch.float32)

    assert -1.0e-3 < float(x_src.item()) < 0.0
    assert float(torchsky_result.weight[hotspot_y, hotspot_x].item()) == 0.0
    assert float(swarp_weight[hotspot_y, hotspot_x].item()) > 0.0


def test_swarp_weighted_coadd_uses_direct_sum_on_aligned_inputs(tmp_path: Path) -> None:
    if bench_map_compat._find_swarp_binary() is None:
        pytest.skip("SWarp CLI is unavailable")

    header = {
        "NAXIS": 2,
        "NAXIS1": 16,
        "NAXIS2": 12,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 8.0,
        "CRPIX2": 6.0,
        "CRVAL1": 11.0,
        "CRVAL2": -4.0,
        "CD1_1": -0.01,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": 0.01,
    }
    path_a = tmp_path / "aligned_a.fits"
    path_b = tmp_path / "aligned_b.fits"
    data_a = torch.arange(16 * 12, dtype=torch.float32).reshape(12, 16)
    data_b = data_a + 5.0
    torchsky.write(str(path_a), data_a, header=header, overwrite=True)
    torchsky.write(str(path_b), data_b, header=header, overwrite=True)

    obs_list = [
        bench_map_ground_truth.ImageObservation(str(path_a)),
        bench_map_ground_truth.ImageObservation(str(path_b)),
    ]

    direct_numer = None
    direct_weight = None
    for obs in obs_list:
        image = bench_map_ground_truth.torchsky.read(obs.path, hdu=obs.hdu).to(dtype=torch.float32)
        weight = bench_map_compat._load_weight_map(obs, image)
        if direct_numer is None:
            direct_numer = image * weight
            direct_weight = weight
        else:
            direct_numer = direct_numer + image * weight
            direct_weight = direct_weight + weight
    assert direct_numer is not None
    assert direct_weight is not None
    direct_data = torch.where(
        direct_weight > 0.0,
        direct_numer / torch.clamp(direct_weight, min=1e-8),
        torch.zeros_like(direct_weight),
    )

    swarp_data_np, swarp_weight_np = bench_map_compat._coadd_with_swarp(
        obs_list,
        target_header=header,
        resampling_type="BILINEAR",
    )
    swarp_data = torch.from_numpy(swarp_data_np)
    swarp_weight = torch.from_numpy(swarp_weight_np)

    assert torch.allclose(swarp_data, direct_data, atol=0.0, rtol=0.0)
    assert torch.allclose(swarp_weight, direct_weight, atol=0.0, rtol=0.0)


def test_write_swarp_config_includes_requested_fscalastro_type(tmp_path: Path) -> None:
    header = {
        "NAXIS": 2,
        "NAXIS1": 16,
        "NAXIS2": 12,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 8.0,
        "CRPIX2": 6.0,
        "CRVAL1": 11.0,
        "CRVAL2": -4.0,
        "CD1_1": -0.01,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": 0.01,
    }
    config_path = tmp_path / "swarp.conf"
    bench_map_compat._write_swarp_config(
        config_path,
        target_header=header,
        imageout_path=tmp_path / "out.fits",
        weightout_path=tmp_path / "out.weight.fits",
        combine_type="SUM",
        resampling_type="BILINEAR",
        fscalastro_type="NONE",
    )
    text = config_path.read_text(encoding="utf-8")
    assert "FSCALASTRO_TYPE        NONE" in text


def test_write_swarp_config_rejects_unknown_fscalastro_type(tmp_path: Path) -> None:
    header = {
        "NAXIS": 2,
        "NAXIS1": 16,
        "NAXIS2": 12,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 8.0,
        "CRPIX2": 6.0,
        "CRVAL1": 11.0,
        "CRVAL2": -4.0,
        "CD1_1": -0.01,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": 0.01,
    }
    with pytest.raises(ValueError, match="fscalastro_type"):
        bench_map_compat._write_swarp_config(
            tmp_path / "bad.conf",
            target_header=header,
            imageout_path=tmp_path / "out.fits",
            weightout_path=tmp_path / "out.weight.fits",
            combine_type="SUM",
            resampling_type="BILINEAR",
            fscalastro_type="AUTO",
        )


def test_check_time_regressions_flags_slow_rows() -> None:
    template = {
        "case": "case_a",
        "library": "torchsky",
        "method": "torchsky",
        "variant": "bilinear",
        "status": "OK",
    }
    err = bench_map_compat._check_time_regressions(
        [{**template, "time_ms": 50.0}],
        [{**template, "max_time_ms": 10.0}],
        tolerance=0.0,
    )
    assert err
    ok = bench_map_compat._check_time_regressions(
        [{**template, "time_ms": 5.0}],
        [{**template, "max_time_ms": 10.0}],
        tolerance=0.0,
    )
    assert not ok
