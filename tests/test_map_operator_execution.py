from __future__ import annotations

import math
import torch
import pytest
from types import SimpleNamespace

from torchsky import WCS
from torchsky.map.observation import ImageObservation
from torchsky.map.operators import (
    ConservativePolygonRemapOp,
    ExactDrizzleGeometryOp,
    ForwardSplatGeometryOp,
    InterpolationGeometryOp,
    MAP_RESAMPLING_KERNELS,
    MAP_STACKING_METHODS,
    MAP_VALUE_SEMANTICS,
    accumulate_projected_observation,
    finalize_projected_stack,
    headers_equivalent_for_direct_copy,
    is_plain_tan_header,
    masked_median_stack,
    normalized_header_for_shape,
    pixel_area_sr_map,
    pixel_area_sr_at,
    prepare_image_observation_data,
    project_image_observation_geometry,
    representative_pixel_area_sr,
    source_bbox_on_target,
    stack_inputs_from_projected_observation,
    target_shape_from_header,
    validate_image_plane_2d,
    wcs_has_distortion,
    wrapped_delta_ra_deg,
    validate_map_resampling_kernel,
    validate_map_stacking_method,
    validate_map_value_semantics,
)
from torchsky.map.operators import execution as execution_mod
from torchsky.map.operators import pixel_area as pixel_area_mod


def test_overlap_background_scratch_bytes_are_linear_in_observations() -> None:
    shape = (5, 7)

    weighted = execution_mod.overlap_background_scratch_bytes(
        3,
        shape,
        stacking_method="weighted_mean",
    )
    median = execution_mod.overlap_background_scratch_bytes(
        3,
        shape,
        stacking_method="median",
    )

    # Per pixel: 3 * (numer, weight, variance, overlap) + 2 graph RHS
    # planes + one disk-backed float32 compact-fit scratch plane.
    assert weighted == 5 * 7 * (3 * 16 + 2 * 4 + 4)
    # Median/mean omit the unused variance plane.
    assert median == 5 * 7 * (3 * 12 + 2 * 4 + 4)


def test_overlap_background_scratch_bytes_validate_inputs() -> None:
    with pytest.raises(ValueError, match="n_observations"):
        execution_mod.overlap_background_scratch_bytes(
            0,
            (2, 2),
            stacking_method="weighted_mean",
        )
    with pytest.raises(ValueError, match="stacking_method"):
        execution_mod.overlap_background_scratch_bytes(
            2,
            (2, 2),
            stacking_method="sigma_clipped",
        )


def test_prepare_image_observation_data_uses_weight_and_calibration_callbacks() -> None:
    calls: dict[str, object] = {}

    def _load_weight_map(obs, image, *, device):
        calls["load"] = (obs, tuple(image.shape), device)
        return torch.full_like(image, 2.0, dtype=torch.float32)

    def _apply_calibration(obs, image, weight):
        calls["calibration"] = (obs, tuple(image.shape), float(weight.mean().item()))
        return image.to(dtype=torch.float32) + 3.0, weight * 5.0

    observation = ImageObservation(
        "mock.fits",
        weight_path="weights.fits",
        photometric_scale=1.5,
    )
    image = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)

    prepared = prepare_image_observation_data(
        observation,
        image,
        device=torch.device("cpu"),
        resampling_kernel="bilinear",
        load_weight_map_fn=_load_weight_map,
        apply_calibration_fn=_apply_calibration,
    )

    assert "load" in calls
    assert "calibration" in calls
    torch.testing.assert_close(prepared.image, image + 3.0)
    torch.testing.assert_close(prepared.source_weight, torch.full_like(image, 10.0))
    assert prepared.operator_bundle.radiometry is not None


def test_prepare_image_observation_data_sanitizes_nan_when_no_calibration_needed() -> (
    None
):
    image = torch.tensor([[1.0, float("nan")]], dtype=torch.float32)
    prepared = prepare_image_observation_data(
        ImageObservation("mock.fits"),
        image,
        device=torch.device("cpu"),
        resampling_kernel="bilinear",
        load_weight_map_fn=lambda obs, arr, *, device: torch.ones_like(arr),
        apply_calibration_fn=lambda obs, arr, w: (arr, w),
    )

    torch.testing.assert_close(prepared.image, torch.tensor([[1.0, 0.0]]))
    torch.testing.assert_close(prepared.source_weight, torch.tensor([[1.0, 0.0]]))


def test_project_image_observation_geometry_dispatches_by_operator_family(
    monkeypatch,
) -> None:
    calls: list[str] = []

    def _fake_square(*args, **kwargs):
        calls.append("drizzle_square")
        return "numer", "weight"

    def _fake_point(*args, **kwargs):
        calls.append("drizzle_point")
        return "numer", "weight"

    def _fake_conservative(*args, **kwargs):
        calls.append("conservative")
        return "numer", "weight"

    monkeypatch.setattr(execution_mod, "exact_square_drizzle_project", _fake_square)
    monkeypatch.setattr(execution_mod, "exact_point_drizzle_project", _fake_point)
    monkeypatch.setattr(
        execution_mod, "conservative_polygon_project", _fake_conservative
    )

    resample_calls: list[str] = []
    splat_calls: list[str] = []

    def resample_fn(*args, **kwargs):
        resample_calls.append("interp")
        return "n", "w"

    def splat_fn(*args, **kwargs):
        splat_calls.append("splat")
        return "n", "w"

    assert project_image_observation_geometry(
        "image",
        "weight",
        "source_wcs",
        "target",
        geometry=ExactDrizzleGeometryOp(),
        resampling_kernel="drizzle_square",
        value_semantics="surface_brightness",
        drizzle_pixfrac=1.0,
        pixel_area_sr_map_fn="area",
        chunk_pixels=256,
        resample_weighted_to_target_fn=resample_fn,
        forward_project_weighted_to_target_fn=splat_fn,
    ) == ("numer", "weight")
    assert project_image_observation_geometry(
        "image",
        "weight",
        "source_wcs",
        "target",
        geometry=ConservativePolygonRemapOp(),
        resampling_kernel="conservative_polygon",
        value_semantics="surface_brightness",
        drizzle_pixfrac=1.0,
        pixel_area_sr_map_fn="area",
        chunk_pixels=256,
        resample_weighted_to_target_fn=resample_fn,
        forward_project_weighted_to_target_fn=splat_fn,
    ) == ("numer", "weight")
    assert project_image_observation_geometry(
        "image",
        "weight",
        "source_wcs",
        "target",
        geometry=ForwardSplatGeometryOp(),
        resampling_kernel="splat_box",
        value_semantics="surface_brightness",
        drizzle_pixfrac=1.0,
        pixel_area_sr_map_fn="area",
        chunk_pixels=256,
        resample_weighted_to_target_fn=resample_fn,
        forward_project_weighted_to_target_fn=splat_fn,
    ) == ("n", "w")
    assert project_image_observation_geometry(
        "image",
        "weight",
        "source_wcs",
        "target",
        geometry=InterpolationGeometryOp(),
        resampling_kernel="bilinear",
        value_semantics="surface_brightness",
        drizzle_pixfrac=1.0,
        pixel_area_sr_map_fn="area",
        chunk_pixels=256,
        resample_weighted_to_target_fn=resample_fn,
        forward_project_weighted_to_target_fn=splat_fn,
    ) == ("n", "w")

    assert calls == ["drizzle_square", "conservative"]
    assert splat_calls == ["splat"]
    assert resample_calls == ["interp"]


def test_source_bbox_on_target_clips_plain_tan_source_boundary() -> None:
    source_header = {
        "NAXIS": 2,
        "NAXIS1": 8,
        "NAXIS2": 8,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 4.0,
        "CRPIX2": 4.0,
        "CRVAL1": 12.0,
        "CRVAL2": -3.0,
        "CDELT1": -0.01,
        "CDELT2": 0.01,
    }
    target_header = dict(source_header)
    target_header["NAXIS1"] = 16
    target_header["NAXIS2"] = 16
    target_header["CRPIX1"] = 8.0
    target_header["CRPIX2"] = 8.0

    bbox = source_bbox_on_target(
        WCS(source_header),
        (8, 8),
        WCS(target_header),
        (16, 16),
        margin=1,
        edge_samples=5,
    )

    assert bbox == (2, 14, 2, 14)


def test_plain_tan_and_distortion_predicates_cover_fast_policy_inputs() -> None:
    header = {
        "NAXIS": 2,
        "NAXIS1": 8,
        "NAXIS2": 8,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 4.0,
        "CRPIX2": 4.0,
        "CRVAL1": 12.0,
        "CRVAL2": -3.0,
        "CDELT1": -0.01,
        "CDELT2": 0.01,
    }
    assert is_plain_tan_header(header)
    assert not is_plain_tan_header({**header, "CTYPE1": "RA---TPV"})
    assert not wcs_has_distortion(WCS(header))

    tpv_header = dict(header)
    tpv_header["CTYPE1"] = "RA---TPV"
    tpv_header["CTYPE2"] = "DEC--TPV"
    assert wcs_has_distortion(WCS(tpv_header))


def test_pixel_area_helpers_are_positive_and_consistent_for_plain_tan() -> None:
    header = {
        "NAXIS": 2,
        "NAXIS1": 8,
        "NAXIS2": 8,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 4.0,
        "CRPIX2": 4.0,
        "CRVAL1": 12.0,
        "CRVAL2": -3.0,
        "CDELT1": -0.01,
        "CDELT2": 0.01,
    }
    wcs = WCS(header)
    area = pixel_area_sr_map(wcs, (8, 8), device=torch.device("cpu"))
    representative = representative_pixel_area_sr(
        wcs,
        (8, 8),
        device=torch.device("cpu"),
    )

    assert area.shape == (8, 8)
    assert bool((area > 0.0).all())
    assert float(torch.median(area).item()) == pytest.approx(representative, rel=0.02)
    sampled = pixel_area_sr_at(
        wcs,
        torch.tensor([0.0, 3.0, 7.0]),
        torch.tensor([1.0, 4.0, 6.0]),
    )
    torch.testing.assert_close(
        sampled,
        area[
            torch.tensor([1, 4, 6]),
            torch.tensor([0, 3, 7]),
        ].to(torch.float64),
        rtol=2e-6,
        atol=0.0,
    )
    torch.testing.assert_close(
        wrapped_delta_ra_deg(torch.tensor([359.0]), torch.tensor([1.0])),
        torch.tensor([-2.0]),
    )


def test_large_tan_pixel_area_matches_exact_jacobian_at_center_and_corner() -> None:
    header = {
        "NAXIS": 2,
        "NAXIS1": 401,
        "NAXIS2": 401,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 201.0,
        "CRPIX2": 201.0,
        "CRVAL1": 137.0,
        "CRVAL2": 42.0,
        "CDELT1": -0.01,
        "CDELT2": 0.01,
    }
    area = pixel_area_sr_map(
        WCS(header),
        (401, 401),
        device=torch.device("cpu"),
        chunk_size=37,
    )
    d2r = math.pi / 180.0
    center_expected = (0.01 * d2r) ** 2
    corner_radius_sq = 2.0 * (2.0 * d2r) ** 2
    corner_expected = center_expected / (1.0 + corner_radius_sq) ** 1.5

    assert float(area[200, 200].item()) == pytest.approx(center_expected, rel=2e-7)
    assert float(area[0, 0].item()) == pytest.approx(corner_expected, rel=2e-7)
    assert float(area[0, 0].item()) < float(area[200, 200].item())


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("sip", object()),
        ("cpdis1", object()),
        ("cpdis2", object()),
        ("det2im1", object()),
        ("det2im2", object()),
        ("_pixel_affine_matrix", torch.eye(2)),
        ("_is_tpv", True),
        ("_is_tnx", True),
        ("_is_zpx", True),
    ],
)
def test_tan_area_fast_path_rejects_every_distortion_kind(
    attribute: str, value: object
) -> None:
    values = {
        "_use_fast_tan": True,
        "sip": None,
        "cpdis1": None,
        "cpdis2": None,
        "det2im1": None,
        "det2im2": None,
        "_pixel_affine_matrix": None,
        "_is_tpv": False,
        "_is_tnx": False,
        "_is_zpx": False,
    }
    values[attribute] = value
    assert not pixel_area_mod._is_undistorted_tan(SimpleNamespace(**values))


def test_tpv_pixel_area_uses_spherical_polygons_and_chunking() -> None:
    header = {
        "NAXIS": 2,
        "NAXIS1": 19,
        "NAXIS2": 17,
        "CTYPE1": "RA---TPV",
        "CTYPE2": "DEC--TPV",
        "CRPIX1": 10.0,
        "CRPIX2": 9.0,
        "CRVAL1": 12.0,
        "CRVAL2": -3.0,
        "CDELT1": -0.05,
        "CDELT2": 0.05,
        "PV1_1": 1.0,
        "PV1_4": 0.01,
        "PV2_1": 1.0,
        "PV2_4": -0.008,
    }
    wcs = WCS(header)
    area_small_chunks = pixel_area_sr_map(
        wcs, (17, 19), device=torch.device("cpu"), chunk_size=3
    )
    area_single_chunk = pixel_area_sr_map(
        wcs, (17, 19), device=torch.device("cpu"), chunk_size=64
    )
    samples = pixel_area_sr_at(
        wcs,
        torch.tensor([0.0, 9.0, 18.0], dtype=torch.float32),
        torch.tensor([0.0, 8.0, 16.0], dtype=torch.float32),
    )

    assert area_small_chunks.dtype == torch.float32
    assert area_small_chunks.device.type == "cpu"
    assert samples.dtype == torch.float64
    assert samples.device.type == "cpu"
    torch.testing.assert_close(area_small_chunks, area_single_chunk)
    torch.testing.assert_close(
        samples,
        area_small_chunks[
            torch.tensor([0, 8, 16]), torch.tensor([0, 9, 18])
        ].to(torch.float64),
        rtol=2e-7,
        atol=0.0,
    )
    assert float((area_small_chunks.max() - area_small_chunks.min()).item()) > 0.0


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_tan_pixel_area_preserves_requested_mps_device_and_dtype() -> None:
    header = {
        "NAXIS": 2,
        "NAXIS1": 17,
        "NAXIS2": 13,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 9.0,
        "CRPIX2": 7.0,
        "CRVAL1": 12.0,
        "CRVAL2": -3.0,
        "CDELT1": -0.05,
        "CDELT2": 0.05,
    }
    cpu_area = pixel_area_sr_map(
        WCS(header), (13, 17), device=torch.device("cpu"), chunk_size=4
    )
    mps_area = pixel_area_sr_map(
        WCS(header).to(torch.device("mps")),
        (13, 17),
        device=torch.device("mps"),
        chunk_size=4,
    )

    assert mps_area.dtype == torch.float32
    assert mps_area.device.type == "mps"
    torch.testing.assert_close(mps_area.cpu(), cpu_area, rtol=2e-6, atol=0.0)


def test_map_config_validators_share_public_option_sets() -> None:
    assert validate_map_resampling_kernel("bilinear") == "bilinear"
    assert validate_map_value_semantics("flux") == "flux"
    assert validate_map_stacking_method("median") == "median"
    assert validate_map_stacking_method("sigma_clipped") == "sigma_clipped"
    assert "conservative_polygon" in MAP_RESAMPLING_KERNELS
    assert MAP_VALUE_SEMANTICS == ("surface_brightness", "flux")
    assert MAP_STACKING_METHODS == (
        "weighted_mean",
        "mean",
        "median",
        "sigma_clipped",
        "chi2",
        "min",
        "max",
        "sum",
    )
    with pytest.raises(ValueError, match="resampling_kernel"):
        validate_map_resampling_kernel("unsupported")
    with pytest.raises(ValueError, match="value_semantics"):
        validate_map_value_semantics("counts")
    with pytest.raises(ValueError, match="stacking_method"):
        validate_map_stacking_method("trimmed_mean")


def test_image_grid_helpers_validate_shape_and_headers() -> None:
    image = torch.zeros((2, 3), dtype=torch.float32)
    assert validate_image_plane_2d(image) is image
    with pytest.raises(ValueError, match="2D images only"):
        validate_image_plane_2d(torch.zeros((1, 2, 3), dtype=torch.float32))

    header = {"NAXIS1": 5, "NAXIS2": 4, "CTYPE1": "RA---TAN"}
    assert target_shape_from_header(header, fallback=(2, 3)) == (4, 5)
    assert target_shape_from_header({"CTYPE1": "RA---TAN"}, fallback=(2, 3)) == (2, 3)
    with pytest.raises(ValueError, match="positive NAXIS1/NAXIS2"):
        target_shape_from_header({"NAXIS1": 0, "NAXIS2": 4}, fallback=(2, 3))

    normalized = normalized_header_for_shape(
        {"NAXIS1": 9, "NAXIS2": 8, "A": 1},
        (6, 7),
        overrides={"B": 2},
    )
    assert normalized == {"NAXIS": 2, "NAXIS1": 7, "NAXIS2": 6, "A": 1, "B": 2}

    source = {
        "NAXIS1": 7,
        "NAXIS2": 6,
        "CTYPE1": "RA---TAN",
        "CHECKSUM": "old",
        "A_COMMENT": "source",
    }
    target = {
        "NAXIS1": 7,
        "NAXIS2": 6,
        "CTYPE1": "RA---TAN",
        "CHECKSUM": "new",
        "A_COMMENT": "target",
    }
    assert headers_equivalent_for_direct_copy(source, target)
    assert not headers_equivalent_for_direct_copy(
        source,
        {**target, "CTYPE1": "RA---TPV"},
    )


def test_stack_inputs_from_projected_observation_applies_flux_area_for_mean_stack() -> (
    None
):
    numer = torch.tensor([[4.0, 0.0], [6.0, 0.0]], dtype=torch.float32)
    weight = torch.tensor([[2.0, 0.0], [3.0, 0.0]], dtype=torch.float32)
    pixel_area_sr = torch.full((2, 2), 10.0, dtype=torch.float32)

    contrib_value, contrib_weight = stack_inputs_from_projected_observation(
        numer,
        weight,
        stacking_method="mean",
        value_semantics="flux",
        pixel_area_sr=pixel_area_sr,
    )

    torch.testing.assert_close(
        contrib_value,
        torch.tensor([[20.0, 0.0], [20.0, 0.0]], dtype=torch.float32),
    )
    torch.testing.assert_close(
        contrib_weight,
        torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32),
    )


def test_masked_median_stack_ignores_unsupported_values() -> None:
    median, support = masked_median_stack(
        [
            torch.tensor([[1.0, 100.0]], dtype=torch.float32),
            torch.tensor([[3.0, -5.0]], dtype=torch.float32),
            torch.tensor([[5.0, 7.0]], dtype=torch.float32),
        ],
        [
            torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            torch.tensor([[1.0, 1.0]], dtype=torch.float32),
            torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        ],
    )

    torch.testing.assert_close(median, torch.tensor([[3.0, -5.0]], dtype=torch.float32))
    torch.testing.assert_close(support, torch.tensor([[3.0, 1.0]], dtype=torch.float32))


def test_accumulate_projected_observation_handles_windowed_cube_coefficients() -> None:
    numer = torch.zeros((2, 4, 5), dtype=torch.float32)
    denom = torch.zeros((2, 4, 5), dtype=torch.float32)
    overlap = torch.zeros((2, 4, 5), dtype=torch.int32)
    coeffs = torch.tensor([[[1.0]], [[0.25]]], dtype=torch.float32)

    accumulate_projected_observation(
        numer,
        denom,
        overlap,
        projected_numer=torch.tensor([[2.0, 4.0], [6.0, 8.0]], dtype=torch.float32),
        projected_weight=torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        projected_overlap=torch.ones((2, 2), dtype=torch.int32),
        stacking_method="weighted_mean",
        value_semantics="surface_brightness",
        target_window=(1, 3, 2, 4),
        coeffs=coeffs,
    )

    expected_numer = torch.zeros((2, 4, 5), dtype=torch.float32)
    expected_numer[0, 1:3, 2:4] = torch.tensor([[2.0, 4.0], [6.0, 8.0]])
    expected_numer[1, 1:3, 2:4] = torch.tensor([[0.5, 1.0], [1.5, 2.0]])
    expected_denom = torch.zeros((2, 4, 5), dtype=torch.float32)
    expected_denom[0, 1:3, 2:4] = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    expected_denom[1, 1:3, 2:4] = torch.tensor([[0.25, 0.5], [0.75, 1.0]])
    expected_overlap = torch.zeros((2, 4, 5), dtype=torch.int32)
    expected_overlap[:, 1:3, 2:4] = 1

    torch.testing.assert_close(numer, expected_numer)
    torch.testing.assert_close(denom, expected_denom)
    torch.testing.assert_close(overlap, expected_overlap)


def test_finalize_projected_stack_handles_flux_weighted_mean_cube() -> None:
    cube, weight = finalize_projected_stack(
        torch.tensor(
            [
                [[2.0, 4.0], [0.0, 8.0]],
                [[1.0, 0.0], [3.0, 6.0]],
            ],
            dtype=torch.float32,
        ),
        torch.tensor(
            [
                [[1.0, 2.0], [0.0, 4.0]],
                [[2.0, 0.0], [3.0, 6.0]],
            ],
            dtype=torch.float32,
        ),
        stacking_method="weighted_mean",
        value_semantics="flux",
        pixel_area_sr=torch.tensor([[[10.0, 20.0], [30.0, 40.0]]], dtype=torch.float32),
    )

    torch.testing.assert_close(
        cube,
        torch.tensor(
            [
                [[20.0, 40.0], [0.0, 80.0]],
                [[5.0, 0.0], [30.0, 40.0]],
            ],
            dtype=torch.float32,
        ),
    )
    torch.testing.assert_close(
        weight,
        torch.tensor(
            [
                [[0.01, 0.005], [0.0, 0.0025]],
                [[0.02, 0.0], [1.0 / 300.0, 0.00375]],
            ],
            dtype=torch.float32,
        ),
        atol=1e-7,
        rtol=0.0,
    )
