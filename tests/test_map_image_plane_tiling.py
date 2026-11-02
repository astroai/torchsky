from __future__ import annotations
import math
from pathlib import Path
import pytest
import torch
from conftest import assert_close_on_device
import torchfits
import torchsky
import torchsky.map.image_plane_coadd as image_plane_coadd_mod
import torchsky.map.image_plane_common as image_plane_common_mod
import torchsky.map.operators.execution as execution_mod
from torchsky.map.operators.geometry_drizzle import (
    _batch_quads_pixel_areas,
)
import torchsky.map.windowing as windowing_mod
from torchsky.map.image_plane_coadd import (
    coadd_image_plane,
)
from torchsky.map.planning import (
    plan_image_plane_target,
)
from torchsky.map.observation import ImageObservation

pytest.importorskip("torchsky._cpp")
from map_image_plane_support import (
    _astropy_sip_fixture,
    _simple_header,
)


def test_batched_square_drizzle_areas_match_independent_quads() -> None:
    quad_x = torch.tensor(
        [[-0.2, 0.8, 0.8, -0.2], [0.4, 1.6, 1.4, 0.2]],
        dtype=torch.float64,
    )
    quad_y = torch.tensor(
        [[-0.1, -0.1, 0.9, 0.9], [0.3, 0.5, 1.7, 1.5]],
        dtype=torch.float64,
    )
    centers = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)

    batched = _batch_quads_pixel_areas(quad_x, quad_y, centers)
    independent = torch.stack(
        [
            _batch_quads_pixel_areas(
                quad_x[index : index + 1],
                quad_y[index : index + 1],
                centers[index : index + 1],
            )[0]
            for index in range(2)
        ]
    )

    assert_close_on_device(batched, independent, rtol=0.0, atol=1.0e-12)


@pytest.mark.needs_float64
def test_exact_square_drizzle_projects_sip_footprints() -> None:
    real_path = _astropy_sip_fixture()
    _, header = torchfits.read(str(real_path), return_header=True)
    target_header = dict(header)
    target_header["CRPIX1"] = float(target_header["CRPIX1"]) + 0.25

    result = coadd_image_plane(
        [ImageObservation(str(real_path))],
        target_header=target_header,
        resampling_kernel="drizzle_square",
        drizzle_pixfrac=0.8,
    )

    assert bool((result.weight > 0.0).any())
    assert bool(torch.isfinite(result.data).all())


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_target_wcs_grids_mps_match_cpu() -> None:
    wcs = torchsky.WCS(_simple_header())
    ra_cpu, dec_cpu = windowing_mod._build_target_world_grid(
        wcs, 8, 8, torch.device("cpu")
    )
    ra_mps, dec_mps = windowing_mod._build_target_world_grid(
        wcs, 8, 8, torch.device("mps")
    )
    quad_cpu = windowing_mod._quadrature_target_world_grid(
        wcs, (8, 8), device=torch.device("cpu"), offset_x=0.25, offset_y=-0.25
    )
    quad_mps = windowing_mod._quadrature_target_world_grid(
        wcs, (8, 8), device=torch.device("mps"), offset_x=0.25, offset_y=-0.25
    )

    assert ra_mps.device.type == "mps"
    assert dec_mps.device.type == "mps"
    assert_close_on_device(ra_mps.cpu(), ra_cpu.to(dtype=torch.float32))
    assert_close_on_device(dec_mps.cpu(), dec_cpu.to(dtype=torch.float32))
    assert_close_on_device(quad_mps[0].cpu(), quad_cpu[0].to(dtype=torch.float32))
    assert_close_on_device(quad_mps[1].cpu(), quad_cpu[1].to(dtype=torch.float32))


def test_small_tan_target_cap_uses_exact_corner_separation() -> None:
    header = _simple_header()
    header.update(
        {
            "NAXIS1": 32,
            "NAXIS2": 32,
            "CRPIX1": 16.5,
            "CRPIX2": 16.5,
            "CDELT1": -0.1,
            "CDELT2": 0.1,
        }
    )
    prepared = windowing_mod._prepare_target(
        header=header,
        target_wcs=torchsky.WCS(header),
        target_shape=(32, 32),
        device=torch.device("cpu"),
        include_world_grid=False,
    )
    plane_radius_rad = math.sqrt(2.0) * 1.55 * math.pi / 180.0
    expected_deg = math.degrees(math.atan(plane_radius_rad))

    assert prepared.cap_radius_deg == pytest.approx(expected_deg, rel=1e-12)
    assert prepared.cap_radius_deg > 2.19


def test_off_center_tan_cap_contains_every_target_pixel() -> None:
    header = _simple_header()
    header.update(
        {
            "NAXIS1": 32,
            "NAXIS2": 32,
            "CRPIX1": 48.0,
            "CRPIX2": -8.0,
            "CDELT1": -0.1,
            "CDELT2": 0.1,
        }
    )
    wcs = torchsky.WCS(header)
    prepared = windowing_mod._prepare_target(
        header=header,
        target_wcs=wcs,
        target_shape=(32, 32),
        device=torch.device("cpu"),
    )
    assert prepared.ra_deg is not None and prepared.dec_deg is not None
    actual_radius = windowing_mod._spherical_cap_radius_from_world_grid(
        prepared.ra_deg,
        prepared.dec_deg,
        center_ra_deg=prepared.center_ra_deg,
        center_dec_deg=prepared.center_dec_deg,
    )

    assert prepared.cap_radius_deg >= actual_radius - 1e-12
    assert prepared.cap_radius_deg > 0.0


def test_distorted_target_disables_spherical_cap_rejection() -> None:
    header = _simple_header()
    header.update(
        {
            "CTYPE1": "RA---TPV",
            "CTYPE2": "DEC--TPV",
            "PV1_1": 1.0,
            "PV1_4": 0.02,
            "PV2_2": 1.0,
            "PV2_6": -0.015,
        }
    )
    prepared = windowing_mod._prepare_target(
        header=header,
        target_wcs=torchsky.WCS(header),
        target_shape=(8, 8),
        device=torch.device("cpu"),
        include_world_grid=False,
    )

    assert prepared.cap_radius_deg == 180.0


def test_small_target_cap_does_not_reject_off_center_tpv_overlap(
    tmp_path: Path,
) -> None:
    target_header = _simple_header()
    target_header.update(
        {
            "NAXIS1": 32,
            "NAXIS2": 32,
            "CRPIX1": 16.5,
            "CRPIX2": 16.5,
            "CDELT1": -0.1,
            "CDELT2": 0.1,
        }
    )
    target_wcs = torchsky.WCS(target_header)
    source_ra, source_dec = target_wcs.pixel_to_world(
        torch.tensor(29.0, dtype=torch.float64),
        torch.tensor(16.0, dtype=torch.float64),
    )
    source_header = {
        "NAXIS": 2,
        "NAXIS1": 8,
        "NAXIS2": 8,
        "CTYPE1": "RA---TPV",
        "CTYPE2": "DEC--TPV",
        "CRPIX1": 4.5,
        "CRPIX2": 4.5,
        "CRVAL1": float(source_ra.item()),
        "CRVAL2": float(source_dec.item()),
        "CDELT1": -0.1,
        "CDELT2": 0.1,
        "PV1_1": 1.0,
        "PV2_1": 1.0,
    }
    source_path = tmp_path / "off_center_tpv.fits"
    torchfits.write(
        str(source_path),
        torch.ones((8, 8), dtype=torch.float32),
        header=source_header,
        overwrite=True,
    )

    result = coadd_image_plane(
        [ImageObservation(str(source_path))],
        target_header=target_header,
        resampling_kernel="bilinear",
    )

    assert bool((result.weight[:, 24:] > 0.0).any())


def _assert_source_pixel_edges_inside_target(
    source_header: dict,
    target_header: dict,
) -> None:
    width = int(source_header["NAXIS1"])
    height = int(source_header["NAXIS2"])
    edges = torch.tensor(
        [
            [-0.5, -0.5],
            [width - 0.5, -0.5],
            [width - 0.5, height - 0.5],
            [-0.5, height - 0.5],
        ],
        dtype=torch.float64,
    )
    world = torchsky.WCS(source_header).pixel_to_world(edges)
    target_pixels = torchsky.WCS(target_header).world_to_pixel(world)
    assert bool(torch.all(target_pixels[:, 0] >= -0.500001))
    assert bool(
        torch.all(target_pixels[:, 0] <= float(target_header["NAXIS1"]) - 0.499999)
    )
    assert bool(torch.all(target_pixels[:, 1] >= -0.500001))
    assert bool(
        torch.all(target_pixels[:, 1] <= float(target_header["NAXIS2"]) - 0.499999)
    )


def test_plan_union_footprint_is_ra_seam_safe(tmp_path: Path) -> None:
    headers: list[dict] = []
    observations: list[ImageObservation] = []
    for index, center_ra in enumerate((359.99, 0.01)):
        header = {
            **_simple_header(),
            "NAXIS1": 16,
            "NAXIS2": 16,
            "CRPIX1": 8.5,
            "CRPIX2": 8.5,
            "CRVAL1": center_ra,
            "CRVAL2": 0.0,
            "CDELT1": -0.001,
            "CDELT2": 0.001,
        }
        path = tmp_path / f"seam_{index}.fits"
        torchfits.write(str(path), torch.ones((16, 16)), header=header, overwrite=True)
        headers.append(header)
        observations.append(ImageObservation(str(path)))

    target = plan_image_plane_target(
        observations,
        union_footprint=True,
        footprint_padding=1.0,
        max_footprint_pixels=10_000,
    )

    assert target.shape[1] < 64
    assert (
        min(
            abs(float(target.header["CRVAL1"])),
            abs(float(target.header["CRVAL1"]) - 360.0),
        )
        < 0.1
    )
    for header in headers:
        _assert_source_pixel_edges_inside_target(header, target.header)


def test_plan_union_footprint_accepts_rotated_cd(tmp_path: Path) -> None:
    header = {
        **_simple_header(),
        "NAXIS1": 16,
        "NAXIS2": 16,
        "CRPIX1": 8.5,
        "CRPIX2": 8.5,
        "CRVAL1": 120.0,
        "CRVAL2": 80.0,
        "CD1_1": 0.0,
        "CD1_2": 0.001,
        "CD2_1": 0.001,
        "CD2_2": 0.0,
    }
    header.pop("CDELT1")
    header.pop("CDELT2")
    path = tmp_path / "rotated.fits"
    torchfits.write(str(path), torch.ones((16, 16)), header=header, overwrite=True)

    target = plan_image_plane_target(
        [ImageObservation(str(path))],
        union_footprint=True,
        footprint_padding=1.0,
    )

    assert math.isclose(abs(float(target.header["CD1_1"])), 0.001)
    assert max(target.shape) <= 20
    _assert_source_pixel_edges_inside_target(header, target.header)


def test_plan_union_footprint_contains_curved_sip_edges(tmp_path: Path) -> None:
    size = 100
    header = {
        **_simple_header(),
        "NAXIS1": size,
        "NAXIS2": size,
        "CRPIX1": 50.5,
        "CRPIX2": 50.5,
        "CRVAL1": 120.0,
        "CRVAL2": 5.0,
        "CDELT1": -0.001,
        "CDELT2": 0.001,
        "CTYPE1": "RA---TAN-SIP",
        "CTYPE2": "DEC--TAN-SIP",
        "A_ORDER": 5,
        "B_ORDER": 5,
        "A_3_0": 5.0e-4,
        "A_5_0": -2.0e-7,
    }
    path = tmp_path / "curved_sip.fits"
    torchfits.write(str(path), torch.ones((size, size)), header=header, overwrite=True)

    target = plan_image_plane_target(
        [ImageObservation(str(path))],
        union_footprint=True,
        footprint_padding=1.0,
    )

    edge = torch.linspace(-0.5, size - 0.5, 1001, dtype=torch.float64)
    low = torch.full_like(edge, -0.5)
    high = torch.full_like(edge, size - 0.5)
    boundary = torch.cat(
        (
            torch.stack((edge, low), dim=1),
            torch.stack((high, edge), dim=1),
            torch.stack((edge.flip(0), high), dim=1),
            torch.stack((low, edge.flip(0)), dim=1),
        ),
        dim=0,
    )
    world = torchsky.WCS(header).pixel_to_world(boundary)
    target_pixels = torchsky.WCS(target.header).world_to_pixel(world)

    assert bool(torch.all(target_pixels[:, 0] >= -0.500001))
    assert bool(torch.all(target_pixels[:, 0] <= float(target.shape[1]) - 0.499999))
    assert bool(torch.all(target_pixels[:, 1] >= -0.500001))
    assert bool(torch.all(target_pixels[:, 1] <= float(target.shape[0]) - 0.499999))


def test_coadd_image_plane_identical_wcs(tmp_path) -> None:
    path_a = tmp_path / "a.fits"
    path_b = tmp_path / "b.fits"
    header = _simple_header()
    data_a = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    data_b = data_a + 2.0
    torchfits.write(str(path_a), data_a, header=header, overwrite=True)
    torchfits.write(str(path_b), data_b, header=header, overwrite=True)

    result = coadd_image_plane(
        [ImageObservation(str(path_a)), ImageObservation(str(path_b))]
    )
    expected = (data_a + data_b) / 2.0
    assert_close_on_device(result.data, expected, atol=1e-4, rtol=1e-4)
    assert result.weight.shape == expected.shape
    assert result.overlap_count is not None
    assert_close_on_device(
        result.overlap_count,
        torch.full((8, 8), 2, dtype=torch.int32),
    )


def test_coadd_image_plane_spatial_tile_single_canvas_matches_non_tiled(
    tmp_path,
) -> None:
    """One tile covering the full output matches non-tiled coadd (same code path)."""
    path_a = tmp_path / "tile_a.fits"
    path_b = tmp_path / "tile_b.fits"
    header = _simple_header()
    data_a = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    data_b = data_a + 2.0
    torchfits.write(str(path_a), data_a, header=header, overwrite=True)
    torchfits.write(str(path_b), data_b, header=header, overwrite=True)
    obs = [ImageObservation(str(path_a)), ImageObservation(str(path_b))]
    full = coadd_image_plane(obs)
    tiled = coadd_image_plane(obs, spatial_tile_hw=(8, 8))
    assert_close_on_device(tiled.data, full.data, atol=1e-4, rtol=1e-4)
    assert_close_on_device(tiled.weight, full.weight, atol=1e-4, rtol=1e-4)
    assert full.overlap_count is not None and tiled.overlap_count is not None
    assert_close_on_device(tiled.overlap_count, full.overlap_count)


def test_spatial_tile_ignores_non_grid_header_metadata(tmp_path, monkeypatch) -> None:
    path = tmp_path / "metadata.fits"
    header = _simple_header()
    image = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    torchfits.write(str(path), image, header=header, overwrite=True)
    original_load = image_plane_coadd_mod.load_observation_image_and_header

    def load_with_metadata(*args, **kwargs):
        loaded, loaded_header = original_load(*args, **kwargs)
        return loaded, {**loaded_header, "CHECKSUM": "source-only"}

    def reject_preparation(*args, **kwargs):
        raise AssertionError("metadata-only differences must not prepare a target")

    monkeypatch.setattr(
        image_plane_coadd_mod,
        "load_observation_image_and_header",
        load_with_metadata,
    )
    monkeypatch.setattr(image_plane_coadd_mod, "_prepare_target", reject_preparation)
    result = coadd_image_plane(
        [ImageObservation(str(path))],
        target_header=header,
        spatial_tile_hw=(8, 8),
    )
    assert_close_on_device(result.data, image)


def test_coadd_image_plane_spatial_tiles_are_exposure_major(
    tmp_path, monkeypatch
) -> None:
    path_a = tmp_path / "stream_a.fits"
    path_b = tmp_path / "stream_b.fits"
    header = _simple_header()
    torchfits.write(
        str(path_a),
        torch.arange(64, dtype=torch.float32).reshape(8, 8),
        header=header,
        overwrite=True,
    )
    torchfits.write(str(path_b), torch.full((8, 8), 5.0), header=header, overwrite=True)
    observations = [ImageObservation(str(path_a)), ImageObservation(str(path_b))]
    expected = coadd_image_plane(observations)

    read_calls: list[tuple[str, int | str | None]] = []
    prepare_calls: list[str] = []
    original_read = torchfits.read
    original_prepare = execution_mod.prepare_image_observation_data

    def counted_read(path, *, hdu=None, **kwargs):
        read_calls.append((str(path), hdu))
        return original_read(path, hdu=hdu, **kwargs)

    def counted_prepare(observation, image, **kwargs):
        prepare_calls.append(str(observation.path))
        return original_prepare(observation, image, **kwargs)

    monkeypatch.setattr(image_plane_common_mod, "read", counted_read)
    monkeypatch.setattr(
        image_plane_coadd_mod, "prepare_image_observation_data", counted_prepare
    )
    tiled = coadd_image_plane(observations, spatial_tile_hw=(3, 5))

    assert read_calls == [(str(path_a), None), (str(path_b), None)]
    assert prepare_calls == [str(path_a), str(path_b)]
    assert_close_on_device(tiled.data, expected.data, atol=1e-4, rtol=1e-4)
    assert_close_on_device(tiled.weight, expected.weight, atol=1e-4, rtol=1e-4)
    assert tiled.overlap_count is not None and expected.overlap_count is not None
    assert_close_on_device(tiled.overlap_count, expected.overlap_count)


def test_tiled_target_cache_hits_misses_admission_and_zero_capacity() -> None:
    header = _simple_header()
    target_header = {**header, "CRPIX1": float(header["CRPIX1"]) + 0.2}
    observations = [
        ImageObservation(
            data=torch.arange(64, dtype=torch.float32).reshape(8, 8) + index,
            header=header,
            weight=1.0 + index,
            weight_is_inverse_variance=True,
        )
        for index in range(4)
    ]

    cached = coadd_image_plane(
        observations,
        target_header=target_header,
        spatial_tile_hw=(4, 4),
        target_cache_capacity=4,
    )
    uncached = coadd_image_plane(
        observations,
        target_header=target_header,
        spatial_tile_hw=(4, 4),
        target_cache_capacity=0,
    )
    evicted = coadd_image_plane(
        observations,
        target_header=target_header,
        spatial_tile_hw=(4, 4),
        target_cache_capacity=2,
    )

    assert cached.provenance is not None
    assert uncached.provenance is not None
    assert evicted.provenance is not None
    assert cached.provenance["target_tile_cache"] == {
        "capacity": 4,
        "size": 4,
        "hits": 12,
        "misses": 4,
        "evictions": 0,
        "admission_skips": 0,
    }
    assert uncached.provenance["target_tile_cache"] == {
        "capacity": 0,
        "size": 0,
        "hits": 0,
        "misses": 16,
        "evictions": 0,
        "admission_skips": 0,
    }
    assert evicted.provenance["target_tile_cache"] == {
        "capacity": 2,
        "size": 2,
        "hits": 6,
        "misses": 10,
        "evictions": 0,
        "admission_skips": 8,
    }
    assert_close_on_device(cached.data, uncached.data)
    assert_close_on_device(cached.weight, uncached.weight)
    assert_close_on_device(cached.overlap_count, uncached.overlap_count)
    assert_close_on_device(cached.context, uncached.context)
    assert cached.inverse_variance is not None and uncached.inverse_variance is not None
    assert cached.correlation_factor is not None
    assert uncached.correlation_factor is not None
    assert_close_on_device(cached.inverse_variance, uncached.inverse_variance)
    assert_close_on_device(cached.correlation_factor, uncached.correlation_factor)
    assert_close_on_device(evicted.data, uncached.data)


def test_tiled_target_cache_preserves_source_gradients() -> None:
    header = _simple_header()
    target_header = {**header, "CRPIX2": float(header["CRPIX2"]) - 0.25}
    base = torch.linspace(0.5, 3.0, 64).reshape(8, 8)

    def evaluate(capacity: int) -> tuple[torch.Tensor, torch.Tensor]:
        source = base.clone().requires_grad_(True)
        result = coadd_image_plane(
            [
                ImageObservation(data=source, header=header),
                ImageObservation(data=source * 1.5, header=header),
            ],
            target_header=target_header,
            spatial_tile_hw=(4, 4),
            target_cache_capacity=capacity,
        )
        coefficients = torch.linspace(-1.0, 1.0, 64).reshape(8, 8)
        torch.sum(result.data * coefficients.to(result.data.device)).backward()
        assert source.grad is not None
        return result.data.detach(), source.grad.detach()

    uncached_data, uncached_gradient = evaluate(0)
    cached_data, cached_gradient = evaluate(4)

    assert_close_on_device(cached_data, uncached_data)
    assert_close_on_device(cached_gradient, uncached_gradient)


def test_target_cache_capacity_must_be_non_negative() -> None:
    header = _simple_header()
    observation = ImageObservation(data=torch.ones((8, 8)), header=header)
    with pytest.raises(ValueError, match="non-negative"):
        coadd_image_plane(
            [observation],
            spatial_tile_hw=(4, 4),
            target_cache_capacity=-1,
        )


def test_coadd_image_plane_does_not_reuse_overwritten_input(tmp_path) -> None:
    path = tmp_path / "mutable.fits"
    header = _simple_header()
    torchfits.write(str(path), torch.ones((8, 8)), header=header, overwrite=True)
    first = coadd_image_plane([ImageObservation(str(path))])
    torchfits.write(str(path), torch.full((8, 8), 7.0), header=header, overwrite=True)
    second = coadd_image_plane([ImageObservation(str(path))])

    assert_close_on_device(first.data, torch.ones((8, 8)))
    assert_close_on_device(second.data, torch.full((8, 8), 7.0))


@pytest.mark.parametrize(
    ("kernel", "value_semantics", "stacking_method"),
    [
        ("bilinear", "surface_brightness", "weighted_mean"),
        ("lanczos3", "surface_brightness", "weighted_mean"),
        ("bilinear", "surface_brightness", "mean"),
        ("bilinear", "flux", "weighted_mean"),
        ("drizzle_square", "surface_brightness", "weighted_mean"),
        ("conservative_polygon", "surface_brightness", "weighted_mean"),
    ],
)
@pytest.mark.needs_float64("conservative_polygon", "drizzle_square")
def test_exposure_major_tiles_match_shifted_wcs_coadd(
    tmp_path, kernel, value_semantics, stacking_method
) -> None:
    path_a = tmp_path / f"tile_{kernel}_a.fits"
    path_b = tmp_path / f"tile_{kernel}_b.fits"
    header_a = _simple_header()
    header_b = dict(header_a)
    header_b["CRPIX1"] = float(header_b["CRPIX1"]) + 0.35
    header_b["CRPIX2"] = float(header_b["CRPIX2"]) - 0.2
    target_header = dict(header_a)
    target_header["CRPIX1"] = float(target_header["CRPIX1"]) + 0.1
    data = torch.arange(64, dtype=torch.float32).reshape(8, 8) + 1.0
    torchfits.write(str(path_a), data, header=header_a, overwrite=True)
    torchfits.write(str(path_b), 0.8 * data + 2.0, header=header_b, overwrite=True)
    observations = [ImageObservation(str(path_a)), ImageObservation(str(path_b))]
    kwargs = {
        "target_header": target_header,
        "resampling_kernel": kernel,
        "value_semantics": value_semantics,
        "stacking_method": stacking_method,
    }

    expected = coadd_image_plane(observations, **kwargs)
    tiled = coadd_image_plane(observations, spatial_tile_hw=(3, 5), **kwargs)

    assert_close_on_device(tiled.data, expected.data, atol=2e-4, rtol=2e-4)
    assert_close_on_device(tiled.weight, expected.weight, atol=2e-4, rtol=2e-4)
    assert tiled.overlap_count is not None and expected.overlap_count is not None
    assert_close_on_device(tiled.overlap_count, expected.overlap_count)


@pytest.mark.parametrize("kernel", ["bilinear", "conservative_polygon", "splat_box"])
@pytest.mark.needs_float64("conservative_polygon", "splat_box")
def test_tiled_squared_coefficient_variance_matches_full_projection(
    tmp_path,
    kernel,
) -> None:
    path = tmp_path / f"tile_variance_{kernel}.fits"
    source_header = _simple_header()
    target_header = {
        **source_header,
        "CRPIX1": float(source_header["CRPIX1"]) + 0.35,
        "CRPIX2": float(source_header["CRPIX2"]) - 0.2,
    }
    data = torch.arange(64, dtype=torch.float32).reshape(8, 8) + 1.0
    torchfits.write(str(path), data, header=source_header, overwrite=True)
    observations = [
        ImageObservation(
            str(path),
            weight=2.0,
            weight_is_inverse_variance=True,
        )
    ]
    kwargs = {
        "target_header": target_header,
        "resampling_kernel": kernel,
    }

    full = coadd_image_plane(observations, **kwargs)
    tiled = coadd_image_plane(observations, spatial_tile_hw=(3, 5), **kwargs)

    assert full.inverse_variance is not None
    assert tiled.inverse_variance is not None
    assert full.correlation_factor is not None
    assert tiled.correlation_factor is not None
    assert_close_on_device(tiled.data, full.data, atol=2e-4, rtol=2e-4)
    assert_close_on_device(tiled.weight, full.weight, atol=2e-4, rtol=2e-4)
    assert_close_on_device(
        tiled.inverse_variance,
        full.inverse_variance,
        atol=2e-4,
        rtol=2e-4,
    )
    assert_close_on_device(
        tiled.correlation_factor,
        full.correlation_factor,
        atol=2e-4,
        rtol=2e-4,
    )


@pytest.mark.needs_float64
def test_square_drizzle_tiles_keep_drops_overlapping_outer_canvas(
    tmp_path,
) -> None:
    path = tmp_path / "tile_square_rotated.fits"
    source_header = _simple_header()
    target_header = dict(source_header)
    angle = math.radians(12.0)
    scale = 0.01
    target_header.pop("CDELT1")
    target_header.pop("CDELT2")
    target_header.update(
        {
            "CRPIX1": float(target_header["CRPIX1"]) + 0.3,
            "CRPIX2": float(target_header["CRPIX2"]) - 0.2,
            "CD1_1": -scale * math.cos(angle),
            "CD1_2": scale * math.sin(angle),
            "CD2_1": scale * math.sin(angle),
            "CD2_2": scale * math.cos(angle),
        }
    )
    data = torch.arange(64, dtype=torch.float32).reshape(8, 8) + 1.0
    torchfits.write(str(path), data, header=source_header, overwrite=True)
    kwargs = {
        "target_header": target_header,
        "resampling_kernel": "drizzle_square",
        "value_semantics": "surface_brightness",
        "drizzle_pixfrac": 1.0,
    }

    full = coadd_image_plane([ImageObservation(str(path))], **kwargs)
    tiled = coadd_image_plane(
        [ImageObservation(str(path))],
        spatial_tile_hw=(3, 3),
        **kwargs,
    )

    assert_close_on_device(tiled.data, full.data, atol=2e-4, rtol=2e-4)
    assert_close_on_device(tiled.weight, full.weight, atol=2e-4, rtol=2e-4)

    mask_path = tmp_path / "outer_center_mask.fits"
    outer_mask = torch.zeros_like(data)
    outer_mask[:, 0] = 1.0
    torchfits.write(str(mask_path), outer_mask, header=source_header, overwrite=True)
    outer_target = dict(source_header)
    outer_target["CRPIX1"] = float(outer_target["CRPIX1"]) - 0.6
    outer = coadd_image_plane(
        [ImageObservation(str(path), mask_path=str(mask_path))],
        target_header=outer_target,
        resampling_kernel="drizzle_square",
        drizzle_pixfrac=1.0,
    )
    outer_tiled = coadd_image_plane(
        [ImageObservation(str(path), mask_path=str(mask_path))],
        target_header=outer_target,
        resampling_kernel="drizzle_square",
        drizzle_pixfrac=1.0,
        spatial_tile_hw=(3, 3),
    )
    assert int(torch.count_nonzero(outer.weight)) > 0
    assert_close_on_device(outer_tiled.weight, outer.weight)
