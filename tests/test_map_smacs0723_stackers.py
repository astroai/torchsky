from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torchsky
from torchsky.map import ImageObservation


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

from bench_map_compat import _as_fits_header
from bench_map_smacs0723_stackers import (
    SMACS_FULL_MOSAIC_PARITY_PAIRS,
    _build_full_mosaic_parity_specs,
    _build_method_specs,
    _build_validation_method_specs,
    _compute_pairwise_parity_metrics,
    _compute_region_metrics,
    _fixed_cutout,
    _filter_exposure_paths_by_patterns,
    _format_markdown,
    _injection_manifest_sign,
    _injection_manifest_tensors,
    _resolve_benchmark_target_header,
    _product_and_metric_arrays,
    _run_coadd_method,
    _run_legacy_method,
    _score_product_parity_pairs,
    _validate_expected_exposure_count,
    _wcs_overlap_probe,
    run_stacker_benchmark,
)


def test_wcs_overlap_probe_compares_torchsky_and_astropy(tmp_path: Path) -> None:
    header = {
        "NAXIS": 2,
        "NAXIS1": 8,
        "NAXIS2": 8,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 4.5,
        "CRPIX2": 4.5,
        "CRVAL1": 12.0,
        "CRVAL2": -3.0,
        "CDELT1": -0.01,
        "CDELT2": 0.01,
    }
    path = tmp_path / "probe.fits"
    torchsky.write(str(path), torch.ones((8, 8)), header=header, overwrite=True)

    row = _wcs_overlap_probe([ImageObservation(str(path))], header)[0]

    assert row["torch_inside_samples"] == 25
    assert row["astropy_inside_samples"] == 25
    assert row["max_target_pixel_disagreement"] < 1.0e-8


def test_build_method_specs_includes_linearized_legacy_targets() -> None:
    specs = _build_method_specs(include_inr=True)
    by_name = {spec["name"]: spec for spec in specs}
    assert by_name["swarp_bilinear"]["target_mode"] == "published_swarp_tan"
    assert by_name["montage"]["target_mode"] == "published_linearized"
    assert by_name["montage_tan"]["target_mode"] == "published_swarp_tan"
    assert by_name["torchsky_drizzle_square"]["target_mode"] == "published_native"
    assert by_name["torchsky_drizzle_point"]["target_mode"] == "published_native"
    assert by_name["torchsky_conservative_polygon"]["target_mode"] == "published_native"
    assert by_name["torchsky_splat_box"]["target_mode"] == "published_native"
    assert by_name["reproject_exact"]["family"] == "reproject"
    assert by_name["reproject_adaptive"]["family"] == "reproject"
    assert by_name["inr_operator_constrained_feature_field"]["family"] == "inr"
    assert by_name["torchsky_splat_box"]["geometry_family"] == "forward_splat"
    assert by_name["torchsky_drizzle_square"]["geometry_family"] == "exact_drizzle"
    assert (
        by_name["torchsky_conservative_polygon"]["geometry_family"]
        == "conservative_overlap"
    )
    assert by_name["reproject_exact"]["geometry_family"] == "conservative_overlap"
    assert (
        by_name["inr_operator_constrained_feature_field"]["geometry_family"]
        == "latent_scene"
    )


def test_compute_region_metrics_reports_affine_and_local_terms() -> None:
    reference = np.ones((8, 8), dtype=np.float32)
    prediction = reference.copy()
    prediction[0, 0] = 2.0
    support = np.ones_like(reference, dtype=bool)

    metrics = _compute_region_metrics(prediction, reference, support)

    assert metrics["support_rmse"] > 0.0
    assert metrics["uniform_rmse"] > 0.0
    assert "center_support_rmse" in metrics
    assert "edge_support_rmse" in metrics
    assert "support_rmse_after_affine_fit" in metrics


def test_validation_specs_freeze_drizzle_and_swarp_grid() -> None:
    specs = _build_validation_method_specs((0.6, 1.0))
    by_name = {spec["name"]: spec for spec in specs}

    assert list(by_name) == [
        "torchsky_bilinear",
        "torchsky_lanczos3",
        "torchsky_drizzle_square_pf0p600",
        "drizzle_square_pf0p600",
        "torchsky_drizzle_square_pf1p000",
        "drizzle_square_pf1p000",
        "swarp_bilinear",
        "swarp_lanczos3",
    ]
    assert by_name["drizzle_square_pf0p600"]["pixfrac"] == 0.6
    assert by_name["torchsky_drizzle_square_pf0p600"]["drizzle_pixfrac"] == 0.6
    assert {spec["target_mode"] for spec in specs} == {"published_swarp_tan"}


def test_full_mosaic_parity_specs_pair_strongest_common_grid_kernels() -> None:
    specs = _build_full_mosaic_parity_specs()

    assert [spec["name"] for spec in specs] == [
        "torchsky_drizzle_square_pf0p800",
        "drizzle_square_pf0p800",
        "torchsky_lanczos3",
        "swarp_lanczos3",
        "torchsky_conservative_polygon",
        "montage_tan",
    ]
    assert {spec["target_mode"] for spec in specs} == {"published_swarp_tan"}
    assert [pair["library"] for pair in SMACS_FULL_MOSAIC_PARITY_PAIRS] == [
        "drizzle",
        "swarp",
        "montage",
    ]
    assert {
        tuple(spec["spatial_tile_hw"])
        for spec in specs
        if spec["family"] == "torchsky"
    } == {(1024, 1024)}


def test_expected_exposure_count_fails_closed() -> None:
    paths = [Path("a.fits"), Path("b.fits")]

    _validate_expected_exposure_count(paths, expected=2, band="F277W", module="b")
    with pytest.raises(RuntimeError, match="F277W module b selected 2 exposures; expected 9"):
        _validate_expected_exposure_count(
            paths,
            expected=9,
            band="F277W",
            module="b",
        )


def test_pairwise_parity_metrics_are_scale_invariant_and_report_support() -> None:
    reference = np.array([[1.0, 2.0], [3.0, 0.0]], dtype=np.float32)
    candidate = 1.01 * reference
    reference_weight = np.array([[1.0, 2.0], [3.0, 0.0]], dtype=np.float32)
    candidate_weight = np.array([[10.0, 20.0], [30.0, 10.0]], dtype=np.float32)

    metrics = _compute_pairwise_parity_metrics(
        reference,
        reference_weight,
        candidate,
        candidate_weight,
    )

    assert metrics["common_support_pixel_count"] == 3
    assert metrics["support_iou_vs_torchsky"] == pytest.approx(0.75)
    assert metrics["normalized_rmse_data_diff_vs_torchsky"] == pytest.approx(0.01)
    assert metrics["affine_slope_vs_torchsky"] == pytest.approx(1.01)
    assert metrics["normalized_rmse_weight_diff_vs_torchsky"] == pytest.approx(0.0)


def test_pattern_filter_uses_complete_filename_token() -> None:
    paths = [
        Path("jw02736001001_02101_00003_nrcblong_cal.fits"),
        Path("jw02736001001_02101_00008_nrcblong_cal.fits"),
        Path("jw02736001001_02101_00009_nrcblong_cal.fits"),
    ]

    assert _filter_exposure_paths_by_patterns(paths, (3, 8)) == paths[:2]


def test_empty_injection_manifest_restores_coordinate_rank() -> None:
    world, flux = _injection_manifest_tensors(
        {
            "world_radec_deg": [],
            "band_flux_jy": {"F277W": []},
        },
        band="F277W",
    )

    assert world.shape == (0, 2)
    assert flux.shape == (0,)


def test_injection_manifest_sign_is_explicit_and_consistent() -> None:
    assert _injection_manifest_sign({}) == 1
    assert _injection_manifest_sign(
        {"injection_control": {"mode": "negative", "sign": -1}}
    ) == -1
    with pytest.raises(ValueError, match="inconsistent"):
        _injection_manifest_sign(
            {"injection_control": {"mode": "positive", "sign": -1}}
        )
    with pytest.raises(ValueError, match="inconsistent"):
        _injection_manifest_sign(
            {"injection_control": {"mode": "negative", "sign": -1.0}}
        )
    with pytest.raises(ValueError, match="unsupported"):
        _injection_manifest_sign(
            {"injection_control": {"mode": "mystery", "sign": 1}}
        )


def test_explicit_target_header_is_copied_and_shape_checked() -> None:
    explicit = {"NAXIS1": 16, "NAXIS2": 16, "CRVAL1": 12.0}
    selected, mode = _resolve_benchmark_target_header(
        "published_native",
        crop_size=16,
        explicit_header=explicit,
        published_target_header={},
        linearized_target_header={},
        swarp_target_header={},
    )
    selected["CRVAL1"] = 13.0
    assert mode == "explicit_fits"
    assert explicit["CRVAL1"] == 12.0
    with pytest.raises(ValueError, match="shape must match"):
        _resolve_benchmark_target_header(
            "published_native",
            crop_size=8,
            explicit_header=explicit,
            published_target_header={},
            linearized_target_header={},
            swarp_target_header={},
        )


def test_fixed_cutout_uses_registered_origin() -> None:
    image = np.arange(36, dtype=np.float32).reshape(6, 6)

    cutout, x0, y0 = _fixed_cutout(image, size=3, x0=2, y0=1)

    assert (x0, y0) == (2, 1)
    np.testing.assert_array_equal(cutout, image[1:4, 2:5])


def test_non_native_products_keep_method_grid_while_metrics_are_aligned(
    monkeypatch,
) -> None:
    def fake_align(array, *, order, **_kwargs):
        return array + (10.0 if order == "bilinear" else 20.0)

    monkeypatch.setattr(
        "bench_map_smacs0723_stackers._align_array_to_target",
        fake_align,
    )
    pred = np.arange(4, dtype=np.float32).reshape(2, 2)
    weight = np.ones((2, 2), dtype=np.float32)

    product_pred, product_weight, metric_pred, metric_weight = (
        _product_and_metric_arrays(
            pred,
            weight,
            target_mode_used="published_swarp_tan",
            target_header={"grid": "method"},
            published_target_header={"grid": "reference"},
        )
    )

    np.testing.assert_array_equal(product_pred, pred)
    np.testing.assert_array_equal(product_weight, weight)
    np.testing.assert_array_equal(metric_pred, pred + 10.0)
    np.testing.assert_array_equal(metric_weight, weight + 20.0)


def test_product_only_persists_supported_product_without_reference_metrics(
    monkeypatch, tmp_path: Path
) -> None:
    header = {
        "NAXIS": 2,
        "NAXIS1": 2,
        "NAXIS2": 2,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 1.5,
        "CRPIX2": 1.5,
        "CRVAL1": 12.0,
        "CRVAL2": -3.0,
        "CDELT1": -0.01,
        "CDELT2": 0.01,
    }
    source_path = tmp_path / "source.fits"
    torchsky.write(
        str(source_path), torch.ones((2, 2)), header=header, overwrite=True
    )
    monkeypatch.setattr(
        "bench_map_smacs0723_stackers._read_ext",
        lambda *_args, **_kwargs: np.ones((2, 2), dtype=np.float32),
    )
    monkeypatch.setattr(
        "bench_map_smacs0723_stackers._sci_header", lambda *_args: dict(header)
    )
    monkeypatch.setattr(
        "bench_map_smacs0723_stackers._prepare_aux_files",
        lambda *_args, **_kwargs: (source_path, source_path, source_path),
    )
    monkeypatch.setattr(
        "bench_map_smacs0723_stackers._wcs_overlap_probe",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        "bench_map_smacs0723_stackers._run_coadd_method",
        lambda *_args, **_kwargs: (
            np.ones((2, 2), dtype=np.float32),
            np.ones((2, 2), dtype=np.float32),
            {"max_overlap": 1},
        ),
    )
    spec = dict(_build_method_specs(include_inr=False)[0])
    out = tmp_path / "products"

    rows = run_stacker_benchmark(
        band="F090W",
        reference_path=tmp_path / "reference.fits",
        exposure_paths=[source_path],
        specs=[spec],
        crop_size=2,
        crop_mode="center",
        overlap_step=1,
        input_product="aligned",
        filename_contains="",
        linearize_sip=False,
        inr_steps=1,
        target_header_override=header,
        product_out_dir=out,
        product_only=True,
    )

    assert rows[0]["status"] == "OK"
    assert rows[0]["product_support_pixel_count"] == 4
    assert rows[0]["reference_metrics_skipped"] is True
    assert (out / "torchsky_bilinear_sci.fits").exists()
    assert (out / "torchsky_bilinear_wht.fits").exists()

    monkeypatch.setattr(
        "bench_map_smacs0723_stackers._align_array_to_target",
        lambda array, **_kwargs: np.zeros_like(array),
    )
    compared = run_stacker_benchmark(
        band="F090W",
        reference_path=tmp_path / "reference.fits",
        exposure_paths=[source_path],
        specs=[spec],
        crop_size=2,
        crop_mode="center",
        overlap_step=1,
        input_product="aligned",
        filename_contains="",
        linearize_sip=False,
        inr_steps=1,
        target_header_override=header,
        product_out_dir=out,
    )[0]
    assert compared["status"] == "FAIL"
    assert compared["product_support_pixel_count"] == 4
    assert compared["reference_support_pixel_count"] == 0
    assert np.isfinite(compared["time_ms"])
    assert "no overlap with the reference metric grid" in compared["message"]


def test_score_product_parity_pairs_compares_persisted_common_grid(
    tmp_path: Path,
) -> None:
    header = {
        "NAXIS": 2,
        "NAXIS1": 2,
        "NAXIS2": 2,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 1.5,
        "CRPIX2": 1.5,
        "CRVAL1": 12.0,
        "CRVAL2": -3.0,
        "CDELT1": -0.01,
        "CDELT2": 0.01,
    }
    products = {}
    for name, scale in (("native", 1.0), ("official", 1.01)):
        science_path = tmp_path / f"{name}_sci.fits"
        weight_path = tmp_path / f"{name}_wht.fits"
        torchsky.write(
            str(science_path),
            scale * torch.arange(1, 5, dtype=torch.float32).reshape(2, 2),
            header=header,
            overwrite=True,
        )
        torchsky.write(
            str(weight_path),
            torch.ones((2, 2)),
            header=header,
            overwrite=True,
        )
        products[name] = {
            "name": name,
            "variant": name,
            "status": "OK",
            "time_ms": 1.0,
            "product_path": str(science_path),
            "product_weight_path": str(weight_path),
        }

    rows = _score_product_parity_pairs(
        list(products.values()),
        (
            {
                "torchsky": "native",
                "official": "official",
                "library": "example",
                "variant": "strong",
            },
        ),
    )

    assert len(rows) == 1
    assert rows[0]["status"] == "OK"
    assert rows[0]["case"] == "smacs_full_mosaic"
    assert rows[0]["library"] == "example"
    assert rows[0]["comparison_reference_variant"] == "native"
    assert rows[0]["normalized_rmse_data_diff_vs_torchsky"] == pytest.approx(0.01)


def test_upstream_drizzle_receives_selected_pixfrac(monkeypatch) -> None:
    received: dict[str, float] = {}

    def fake_drizzle(*args, **kwargs):
        received["pixfrac"] = kwargs["pixfrac"]
        return np.zeros((2, 2), dtype=np.float32), np.ones((2, 2), dtype=np.float32)

    monkeypatch.setattr(
        "bench_map_smacs0723_stackers._coadd_with_real_drizzle",
        fake_drizzle,
    )
    _run_legacy_method(
        [],
        {"runner": "drizzle", "kernel": "square", "pixfrac": 0.6},
        {"NAXIS1": 2, "NAXIS2": 2},
    )

    assert received["pixfrac"] == 0.6


def test_torchsky_stacker_receives_selected_spatial_tile(monkeypatch) -> None:
    received: dict[str, object] = {}

    def fake_coadd(*args, **kwargs):
        received["spatial_tile_hw"] = kwargs["spatial_tile_hw"]
        return SimpleNamespace(
            data=torch.zeros((2, 2)),
            weight=torch.ones((2, 2)),
            overlap_count=torch.ones((2, 2), dtype=torch.int32),
        )

    monkeypatch.setattr(
        "bench_map_smacs0723_stackers.coadd_image_plane",
        fake_coadd,
    )
    _, _, extra = _run_coadd_method(
        [],
        {
            "kernel": "lanczos3",
            "value_semantics": "surface_brightness",
            "stacking_method": "weighted_mean",
            "spatial_tile_hw": (1024, 1024),
        },
        {"NAXIS1": 2, "NAXIS2": 2},
    )

    assert received["spatial_tile_hw"] == (1024, 1024)
    assert extra["compute_device"] == "cpu"


def test_format_markdown_includes_geometry_family_column() -> None:
    markdown = _format_markdown(
        [
            {
                "variant": "torchsky:splat_box:surface_brightness:pixfrac=1.000",
                "family": "torchsky",
                "geometry_family": "forward_splat",
                "status": "OK",
                "target_mode_used": "published_native",
                "support_rmse": 0.1,
                "uniform_rmse": 0.2,
                "support_flux_ratio": 1.0,
                "center_support_rmse": 0.05,
                "edge_support_rmse": 0.15,
                "support_rmse_after_affine_fit": 0.09,
                "time_ms": 12.3,
            }
        ]
    )
    assert "| variant | family | geometry | status | target |" in markdown
    assert (
        "| torchsky:splat_box:surface_brightness:pixfrac=1.000 | torchsky | forward_splat | OK | published_native |"
        in markdown
    )


def test_as_fits_header_sanitizes_multiline_non_ascii_strings() -> None:
    header = _as_fits_header(
        {
            "NAXIS": 2,
            "NAXIS1": 4,
            "NAXIS2": 4,
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
            "CRPIX1": 2.0,
            "CRPIX2": 2.0,
            "BADTXT": "line1\nline2\u2603",
            "COMMENT": "\nignore me",
        }
    )
    assert header["BADTXT"] == "line1 line2"
    assert "COMMENT" not in header
