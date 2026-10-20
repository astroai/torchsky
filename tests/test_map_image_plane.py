from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest
import torch
from torchfits import HDUList, TensorHDU

import torchsky
import torchsky.map.image_plane as image_plane_mod
import torchsky.map.image_plane_coadd as image_plane_coadd_mod
import torchsky.map.image_plane_common as image_plane_common_mod
import torchsky.map.image_plane_projection as image_plane_projection_mod
import torchsky.map.operators.execution as execution_mod
import torchsky.map.operators.geometry_drizzle as drizzle_geometry_mod
from torchsky.map.operators.geometry_drizzle import (
    _batch_quads_pixel_areas,
    _scanner_row_bounds,
)
import torchsky.map.operators.resampling as resampling_mod
import torchsky.map.windowing as windowing_mod
from torchsky.map import stack_product
from torchsky.map.image_plane import (
    coadd_image_plane,
    coadd_image_plane_cube,
    coadd_image_plane_multiband,
    plan_image_plane_cube_target,
    plan_image_plane_cube_target_from_header,
    plan_image_plane_target,
)
from torchsky.map.observation import ImageObservation
from torchsky.map.product import plan_map_product
from torchsky.map.sky import (
    CrossExposureSkyCorrector,
    MeshSkyEstimator,
    PolynomialSkyModel,
)
from torchsky.map.operators import (
    pixel_area_sr_map,
    representative_pixel_area_sr,
    validate_image_plane_2d,
)
from torchsky.spectral import SpectralAxis
import torchfits

pytest.importorskip("torchsky._cpp")


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

    torch.testing.assert_close(batched, independent, rtol=0.0, atol=1.0e-12)


def test_drizzle_scanner_falls_back_when_detector_corners_are_invalid() -> None:
    ys, xs = torch.meshgrid(
        torch.arange(3, dtype=torch.float64),
        torch.arange(3, dtype=torch.float64),
        indexing="ij",
    )
    pixmap = torch.stack((xs, ys), dim=-1)
    pixmap[0, 0] = torch.nan

    row_min, row_max = _scanner_row_bounds(
        pixmap,
        source_shape=(3, 3),
        target_shape=(3, 3),
    )

    torch.testing.assert_close(row_min, torch.zeros(3, dtype=torch.int64))
    torch.testing.assert_close(row_max, torch.full((3,), 2, dtype=torch.int64))


def test_drizzle_scanner_does_not_reject_curved_interior_overlap() -> None:
    pixmap = torch.full((5, 5, 2), 10.0, dtype=torch.float64)
    pixmap[1:4, 1:4, 0] = torch.arange(3, dtype=torch.float64).repeat(3, 1)
    pixmap[1:4, 1:4, 1] = torch.arange(3, dtype=torch.float64).repeat(3, 1).T

    row_min, row_max = _scanner_row_bounds(
        pixmap,
        source_shape=(5, 5),
        target_shape=(3, 3),
    )

    torch.testing.assert_close(row_min, torch.zeros(5, dtype=torch.int64))
    torch.testing.assert_close(row_max, torch.full((5,), 4, dtype=torch.int64))


def test_exact_square_drizzle_bypasses_corner_scanner_for_sip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_path = _astropy_sip_fixture()
    _, header = torchfits.read(str(real_path), return_header=True)
    target_header = dict(header)
    target_header["CRPIX1"] = float(target_header["CRPIX1"]) + 0.25

    def reject_scanner(*args, **kwargs):
        raise AssertionError("the corner scanner is unsafe for SIP")

    monkeypatch.setattr(drizzle_geometry_mod, "_scanner_row_bounds", reject_scanner)
    result = coadd_image_plane(
        [ImageObservation(str(real_path))],
        target_header=target_header,
        resampling_kernel="drizzle_square",
        drizzle_pixfrac=0.8,
    )

    assert bool((result.weight > 0.0).any())


def _simple_header() -> dict:
    return {
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
    torch.testing.assert_close(ra_mps.cpu(), ra_cpu.to(dtype=torch.float32))
    torch.testing.assert_close(dec_mps.cpu(), dec_cpu.to(dtype=torch.float32))
    torch.testing.assert_close(quad_mps[0].cpu(), quad_cpu[0].to(dtype=torch.float32))
    torch.testing.assert_close(quad_mps[1].cpu(), quad_cpu[1].to(dtype=torch.float32))


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


def test_small_target_cap_does_not_reject_off_center_tpv_overlap(tmp_path: Path) -> None:
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


def _astropy_sip_fixture() -> Path:
    import astropy

    return (
        Path(astropy.__file__).resolve().parent
        / "nddata"
        / "tests"
        / "data"
        / "sip-wcs.fits"
    )


def _clean_header(header: dict) -> dict:
    out = {}
    for key, value in header.items():
        # The fixture tensors are decoded floating values, not stored integers.
        if str(key).endswith("_COMMENT") or key in {"BZERO", "BSCALE", "BLANK"}:
            continue
        out[key] = value
    return out


def _write_real_sip_mef(tmp_path: Path, *, shifted: bool) -> Path:
    image, header = torchfits.read(str(_astropy_sip_fixture()), return_header=True)
    data = image.to(dtype=torch.float32).cpu()
    sci1 = _clean_header(dict(header))
    sci1["EXTNAME"] = "SCI"
    sci1["EXTVER"] = 1
    sci2 = _clean_header(dict(header))
    sci2["EXTNAME"] = "SCI"
    sci2["EXTVER"] = 2
    if shifted:
        sci2["CRVAL1"] = float(sci2["CRVAL1"]) + 0.01
        sci2["CRVAL2"] = float(sci2["CRVAL2"]) - 0.01

    err1 = _clean_header(dict(header))
    err1["EXTNAME"] = "ERR"
    err1["EXTVER"] = 1
    err2 = _clean_header(dict(header))
    err2["EXTNAME"] = "ERR"
    err2["EXTVER"] = 2
    dq1 = _clean_header(dict(header))
    dq1["EXTNAME"] = "DQ"
    dq1["EXTVER"] = 1
    dq2 = _clean_header(dict(header))
    dq2["EXTNAME"] = "DQ"
    dq2["EXTVER"] = 2

    weight1 = torch.ones_like(image, dtype=torch.float32).cpu()
    weight2 = torch.ones_like(image, dtype=torch.float32).cpu()
    weight1[:, : image.shape[1] // 3] = 2.0
    weight2[:, (2 * image.shape[1]) // 3 :] = 3.0
    mask1 = torch.ones_like(image, dtype=torch.float32).cpu()
    mask2 = torch.ones_like(image, dtype=torch.float32).cpu()
    mask1[:, (2 * image.shape[1]) // 3 :] = 0.0
    mask2[:, : image.shape[1] // 3] = 0.0

    path = tmp_path / ("real_sip_shifted_mef.fits" if shifted else "real_sip_mef.fits")
    HDUList(
        [
            TensorHDU(torch.empty(0)),
            TensorHDU(data=data, header=sci1),
            TensorHDU(data=weight1, header=err1),
            TensorHDU(data=mask1, header=dq1),
            TensorHDU(data=data, header=sci2),
            TensorHDU(data=weight2, header=err2),
            TensorHDU(data=mask2, header=dq2),
        ]
    ).write(str(path), overwrite=True)
    return path


def _rotated_cd(
    scale_deg: float, rotation_deg: float
) -> tuple[float, float, float, float]:
    theta = torch.deg2rad(torch.tensor(float(rotation_deg), dtype=torch.float64)).item()
    cos_t = float(torch.cos(torch.tensor(theta)).item())
    sin_t = float(torch.sin(torch.tensor(theta)).item())
    return (
        -scale_deg * cos_t,
        scale_deg * sin_t,
        scale_deg * sin_t,
        scale_deg * cos_t,
    )


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


def _tpv_header(
    size: int,
    *,
    crval1: float,
    crval2: float,
    rotation_deg: float = 0.0,
    scale_deg: float = 0.01,
) -> dict:
    header = {}
    header["NAXIS"] = 2
    header["NAXIS1"] = size
    header["NAXIS2"] = size
    header["CTYPE1"] = "RA---TPV"
    header["CTYPE2"] = "DEC--TPV"
    header["CRPIX1"] = size / 2.0
    header["CRPIX2"] = size / 2.0
    header["CRVAL1"] = crval1
    header["CRVAL2"] = crval2
    cd11, cd12, cd21, cd22 = _rotated_cd(scale_deg, rotation_deg)
    header["CD1_1"] = cd11
    header["CD1_2"] = cd12
    header["CD2_1"] = cd21
    header["CD2_2"] = cd22
    header["PV1_0"] = 0.0
    header["PV1_1"] = 1.0
    header["PV1_2"] = -0.02
    header["PV1_4"] = 2.0e-4
    header["PV1_5"] = -3.0e-4
    header["PV1_6"] = 1.5e-4
    header["PV1_7"] = 2.0e-6
    header["PV1_39"] = 2.0e-11
    header["PV2_0"] = 0.0
    header["PV2_1"] = 0.02
    header["PV2_2"] = 1.0
    header["PV2_4"] = -1.0e-4
    header["PV2_5"] = 2.5e-4
    header["PV2_6"] = -2.0e-4
    header["PV2_7"] = -1.5e-6
    header["PV2_39"] = -1.0e-11
    return header


def _write_synthetic_tpv_mef(tmp_path: Path, *, size: int = 64) -> Path:
    base_header = _clean_header(_simple_header())
    base_header["NAXIS1"] = size
    base_header["NAXIS2"] = size
    base_header["CRPIX1"] = size / 2.0
    base_header["CRPIX2"] = size / 2.0
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, size, dtype=torch.float32),
        torch.linspace(-1.0, 1.0, size, dtype=torch.float32),
        indexing="ij",
    )
    image = (
        0.7 * torch.exp(-((xx + 0.2) ** 2 + (yy - 0.15) ** 2) / 0.08)
        + 0.9 * torch.exp(-((xx - 0.35) ** 2 + (yy + 0.25) ** 2) / 0.03)
        + 0.2 * torch.sin(5.0 * xx)
        + 0.1 * torch.cos(4.0 * yy)
    ).to(dtype=torch.float32)
    sci1 = base_header.copy()
    sci1["EXTNAME"] = "SCI"
    sci1["EXTVER"] = 1
    sci2 = _tpv_header(
        size,
        crval1=float(base_header["CRVAL1"]) + 0.42,
        crval2=float(base_header["CRVAL2"]) - 0.18,
        rotation_deg=14.0,
        scale_deg=0.01,
    )
    sci2["EXTNAME"] = "SCI"
    sci2["EXTVER"] = 2
    err1 = sci1.copy()
    err1["EXTNAME"] = "ERR"
    err2 = sci2.copy()
    err2["EXTNAME"] = "ERR"
    dq1 = sci1.copy()
    dq1["EXTNAME"] = "DQ"
    dq2 = sci2.copy()
    dq2["EXTNAME"] = "DQ"
    weight1 = (0.7 + 1.3 * (xx + 1.0) / 2.0).to(dtype=torch.float32)
    weight2 = (0.6 + 1.6 * (yy + 1.0) / 2.0).to(dtype=torch.float32)
    mask1 = torch.ones_like(image, dtype=torch.float32)
    mask2 = torch.ones_like(image, dtype=torch.float32)
    mask1[:, size // 3 : size // 3 + max(2, size // 16)] = 0.0
    mask2[size // 2 - max(2, size // 20) : size // 2 + max(2, size // 20), :] = 0.0
    path = tmp_path / "synthetic_tpv_mef.fits"
    HDUList(
        [
            TensorHDU(torch.empty(0)),
            TensorHDU(data=image.cpu(), header=sci1),
            TensorHDU(data=weight1.cpu(), header=err1),
            TensorHDU(data=mask1.cpu(), header=dq1),
            TensorHDU(data=image.cpu(), header=sci2),
            TensorHDU(data=weight2.cpu(), header=err2),
            TensorHDU(data=mask2.cpu(), header=dq2),
        ]
    ).write(str(path), overwrite=True)
    return path


def _sky_from_header(
    header: dict,
    *,
    amplitude: float = 1.0,
    center_ra_deg: float | None = None,
    center_dec_deg: float | None = None,
) -> torch.Tensor:
    wcs = torchsky.WCS(header)
    height = int(header["NAXIS2"])
    width = int(header["NAXIS1"])
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float64),
        torch.arange(width, dtype=torch.float64),
        indexing="ij",
    )
    ra_deg, dec_deg = wcs.pixel_to_world(x, y)
    if center_ra_deg is None:
        center_ra_deg = float(header["CRVAL1"])
    if center_dec_deg is None:
        center_dec_deg = float(header["CRVAL2"])
    dra = ((ra_deg - center_ra_deg + 180.0) % 360.0) - 180.0
    ddec = dec_deg - center_dec_deg
    sky = (
        3.0
        + amplitude * torch.exp(-((dra / 0.035) ** 2 + (ddec / 0.028) ** 2))
        + 0.15 * dra
        - 0.08 * ddec
        + 0.04 * dra * ddec
    )
    return sky.to(dtype=torch.float32)


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
    torch.testing.assert_close(result.data, expected, atol=1e-4, rtol=1e-4)
    assert result.weight.shape == expected.shape
    assert result.overlap_count is not None
    torch.testing.assert_close(
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
    torch.testing.assert_close(tiled.data, full.data, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(tiled.weight, full.weight, atol=1e-4, rtol=1e-4)
    assert full.overlap_count is not None and tiled.overlap_count is not None
    torch.testing.assert_close(tiled.overlap_count, full.overlap_count)


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
    original_read = image_plane_mod.read
    original_prepare = image_plane_mod.prepare_image_observation_data

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
    torch.testing.assert_close(tiled.data, expected.data, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(tiled.weight, expected.weight, atol=1e-4, rtol=1e-4)
    assert tiled.overlap_count is not None and expected.overlap_count is not None
    torch.testing.assert_close(tiled.overlap_count, expected.overlap_count)


def test_coadd_image_plane_does_not_reuse_overwritten_input(tmp_path) -> None:
    path = tmp_path / "mutable.fits"
    header = _simple_header()
    torchfits.write(str(path), torch.ones((8, 8)), header=header, overwrite=True)
    first = coadd_image_plane([ImageObservation(str(path))])
    torchfits.write(str(path), torch.full((8, 8), 7.0), header=header, overwrite=True)
    second = coadd_image_plane([ImageObservation(str(path))])

    torch.testing.assert_close(first.data, torch.ones((8, 8)))
    torch.testing.assert_close(second.data, torch.full((8, 8), 7.0))


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

    torch.testing.assert_close(tiled.data, expected.data, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(tiled.weight, expected.weight, atol=2e-4, rtol=2e-4)
    assert tiled.overlap_count is not None and expected.overlap_count is not None
    torch.testing.assert_close(tiled.overlap_count, expected.overlap_count)


@pytest.mark.parametrize("kernel", ["bilinear", "conservative_polygon", "splat_box"])
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
    torch.testing.assert_close(tiled.data, full.data, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(tiled.weight, full.weight, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(
        tiled.inverse_variance,
        full.inverse_variance,
        atol=2e-4,
        rtol=2e-4,
    )
    torch.testing.assert_close(
        tiled.correlation_factor,
        full.correlation_factor,
        atol=2e-4,
        rtol=2e-4,
    )


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

    torch.testing.assert_close(tiled.data, full.data, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(tiled.weight, full.weight, atol=2e-4, rtol=2e-4)

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
    torch.testing.assert_close(outer_tiled.weight, outer.weight)


def test_cli_map_coadd_writes_output(tmp_path, monkeypatch) -> None:
    from torchsky.cli import main

    path_a = tmp_path / "cli_a.fits"
    path_b = tmp_path / "cli_b.fits"
    header = _simple_header()
    data_a = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    data_b = data_a + 2.0
    torchfits.write(str(path_a), data_a, header=header, overwrite=True)
    torchfits.write(str(path_b), data_b, header=header, overwrite=True)
    out = tmp_path / "coadd_cli.fits"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "torchsky",
            "map",
            str(out),
            str(path_a),
            str(path_b),
            "--no-auto-tile",
            "--no-union-footprint",
        ],
    )
    assert main() == 0
    got, _ = torchfits.read(str(out), return_header=True)
    torch.testing.assert_close(
        got[1:-1, 1:-1], (data_a + data_b)[1:-1, 1:-1] / 2.0, atol=1e-4, rtol=1e-4
    )


def test_coadd_image_plane_mean_stacking_averages_observations_equally(
    tmp_path,
) -> None:
    path_a = tmp_path / "mean_a.fits"
    path_b = tmp_path / "mean_b.fits"
    header = _simple_header()
    data_a = torch.full((8, 8), 1.0, dtype=torch.float32)
    data_b = torch.full((8, 8), 9.0, dtype=torch.float32)
    torchfits.write(str(path_a), data_a, header=header, overwrite=True)
    torchfits.write(str(path_b), data_b, header=header, overwrite=True)

    weighted = coadd_image_plane(
        [
            ImageObservation(str(path_a), weight=1.0),
            ImageObservation(str(path_b), weight=9.0),
        ],
        stacking_method="weighted_mean",
    )
    mean = coadd_image_plane(
        [
            ImageObservation(str(path_a), weight=1.0),
            ImageObservation(str(path_b), weight=9.0),
        ],
        stacking_method="mean",
    )

    torch.testing.assert_close(
        weighted.data,
        torch.full((8, 8), 8.2, dtype=torch.float32),
        atol=1e-5,
        rtol=1e-5,
    )
    assert weighted.inverse_variance is None
    torch.testing.assert_close(
        mean.data,
        torch.full((8, 8), 5.0, dtype=torch.float32),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        mean.weight,
        torch.full((8, 8), 2.0, dtype=torch.float32),
        atol=1e-5,
        rtol=1e-5,
    )


def test_coadd_image_plane_median_stacking_uses_observation_median(tmp_path) -> None:
    path_a = tmp_path / "median_a.fits"
    path_b = tmp_path / "median_b.fits"
    path_c = tmp_path / "median_c.fits"
    header = _simple_header()
    data_a = torch.full((8, 8), 1.0, dtype=torch.float32)
    data_b = torch.full((8, 8), 9.0, dtype=torch.float32)
    data_c = torch.full((8, 8), 100.0, dtype=torch.float32)
    torchfits.write(str(path_a), data_a, header=header, overwrite=True)
    torchfits.write(str(path_b), data_b, header=header, overwrite=True)
    torchfits.write(str(path_c), data_c, header=header, overwrite=True)

    weighted = coadd_image_plane(
        [
            ImageObservation(str(path_a), weight=1.0),
            ImageObservation(str(path_b), weight=1.0),
            ImageObservation(str(path_c), weight=20.0),
        ],
        stacking_method="weighted_mean",
    )
    median = coadd_image_plane(
        [
            ImageObservation(str(path_a), weight=1.0),
            ImageObservation(str(path_b), weight=1.0),
            ImageObservation(str(path_c), weight=20.0),
        ],
        stacking_method="median",
    )

    assert float(weighted.data[0, 0].item()) > 80.0
    torch.testing.assert_close(
        median.data,
        torch.full((8, 8), 9.0, dtype=torch.float32),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        median.weight,
        torch.full((8, 8), 3.0, dtype=torch.float32),
        atol=1e-5,
        rtol=1e-5,
    )


def test_spilled_exact_median_matches_memory_and_cleans_scratch(tmp_path) -> None:
    values = [
        torch.tensor([[1.0, 20.0, 7.0], [4.0, 8.0, 3.0]]),
        torch.tensor([[5.0, 10.0, 9.0], [2.0, 6.0, 1.0]]),
        torch.tensor([[3.0, 30.0, 8.0], [6.0, 7.0, 5.0]]),
    ]
    support = [
        torch.tensor([[True, True, False], [True, True, True]]),
        torch.tensor([[True, False, True], [True, True, True]]),
        torch.tensor([[True, True, True], [False, True, True]]),
    ]
    expected = execution_mod.masked_median_stack(values, support)
    scratch = tmp_path / "scratch"

    actual = execution_mod.spilled_exact_median_stack(
        zip(values, support, strict=True),
        n_observations=3,
        shape=(2, 3),
        scratch_dir=scratch,
        working_memory_bytes=32,
    )

    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])
    assert scratch.is_dir()
    assert list(scratch.iterdir()) == []


def test_spilled_exact_median_rejects_insufficient_scratch_before_projection(
    tmp_path, monkeypatch
) -> None:
    consumed = False

    def contributions():
        nonlocal consumed
        consumed = True
        yield torch.ones((4, 4)), torch.ones((4, 4), dtype=torch.bool)

    class _Usage:
        free = 1

    monkeypatch.setattr(execution_mod.shutil, "disk_usage", lambda _path: _Usage())

    with pytest.raises(OSError, match="scratch.*requires"):
        execution_mod.spilled_exact_median_stack(
            contributions(),
            n_observations=2,
            shape=(4, 4),
            scratch_dir=tmp_path / "too-small",
        )

    assert consumed is False


def test_spilled_exact_median_cleans_scratch_after_projection_failure(
    tmp_path,
) -> None:
    scratch = tmp_path / "failed-scratch"

    def contributions():
        yield torch.ones((2, 2)), torch.ones((2, 2), dtype=torch.bool)
        raise RuntimeError("projection failed")

    with pytest.raises(RuntimeError, match="projection failed"):
        execution_mod.spilled_exact_median_stack(
            contributions(),
            n_observations=2,
            shape=(2, 2),
            scratch_dir=scratch,
        )

    assert list(scratch.iterdir()) == []


def test_coadd_image_plane_tiled_median_uses_and_cleans_scratch(tmp_path) -> None:
    header = _simple_header()
    observations = []
    for index, level in enumerate((1.0, 5.0, 30.0)):
        path = tmp_path / f"median_tiled_{index}.fits"
        torchfits.write(
            str(path),
            torch.full((8, 8), level, dtype=torch.float32),
            header=header,
            overwrite=True,
        )
        observations.append(ImageObservation(str(path)))
    scratch = tmp_path / "median-scratch"

    full = coadd_image_plane(
        observations,
        stacking_method="median",
        scratch_dir=scratch,
    )
    tiled = coadd_image_plane(
        observations,
        stacking_method="median",
        spatial_tile_hw=(3, 5),
        scratch_dir=scratch,
    )

    torch.testing.assert_close(tiled.data, full.data)
    torch.testing.assert_close(tiled.weight, full.weight)
    assert tiled.context is not None and full.context is not None
    torch.testing.assert_close(tiled.context, full.context)
    assert tiled.overlap_count is not None and full.overlap_count is not None
    torch.testing.assert_close(tiled.overlap_count, full.overlap_count)
    assert list(scratch.iterdir()) == []


def test_coadd_image_plane_sigma_clipped_rejects_outlier_and_tracks_context(
    tmp_path,
) -> None:
    header = _simple_header()
    observations = []
    for index, level in enumerate((1.0, 1.1, 0.9, 1.05, 100.0)):
        path = tmp_path / f"clip_{index}.fits"
        torchfits.write(
            str(path),
            torch.full((8, 8), level, dtype=torch.float32),
            header=header,
            overwrite=True,
        )
        observations.append(
            ImageObservation(
                str(path),
                weight=4.0,
                weight_is_inverse_variance=True,
            )
        )
    scratch = tmp_path / "clip-scratch"

    clipped = coadd_image_plane(
        observations,
        stacking_method="sigma_clipped",
        clip_sigma=4.0,
        clip_iterations=3,
        scratch_dir=scratch,
    )

    torch.testing.assert_close(
        clipped.data,
        torch.full((8, 8), 1.0125),
        rtol=0.0,
        atol=1.0e-6,
    )
    torch.testing.assert_close(clipped.weight, torch.full((8, 8), 16.0))
    assert clipped.inverse_variance is not None
    torch.testing.assert_close(
        clipped.inverse_variance,
        torch.full((8, 8), 16.0),
    )
    assert clipped.overlap_count is not None
    torch.testing.assert_close(
        clipped.overlap_count,
        torch.full((8, 8), 4, dtype=torch.int32),
    )
    assert clipped.context is not None
    torch.testing.assert_close(
        clipped.context[0],
        torch.full((8, 8), 0b01111, dtype=torch.int64),
    )
    assert list(scratch.iterdir()) == []


def test_coadd_image_plane_tiled_sigma_clipped_matches_full(tmp_path) -> None:
    header = _simple_header()
    observations = []
    for index, level in enumerate((2.0, 2.1, 1.9, 2.05, 80.0)):
        path = tmp_path / f"clip_tiled_{index}.fits"
        torchfits.write(
            str(path),
            torch.full((8, 8), level, dtype=torch.float32),
            header=header,
            overwrite=True,
        )
        observations.append(ImageObservation(str(path)))
    scratch = tmp_path / "clip-tiled-scratch"
    kwargs = {
        "stacking_method": "sigma_clipped",
        "clip_sigma": 4.0,
        "clip_iterations": 3,
        "scratch_dir": scratch,
    }

    full = coadd_image_plane(observations, **kwargs)
    tiled = coadd_image_plane(observations, spatial_tile_hw=(3, 5), **kwargs)

    torch.testing.assert_close(tiled.data, full.data)
    torch.testing.assert_close(tiled.weight, full.weight)
    assert full.inverse_variance is None
    assert tiled.inverse_variance is None
    assert tiled.context is not None and full.context is not None
    torch.testing.assert_close(tiled.context, full.context)
    assert list(scratch.iterdir()) == []


def test_spilled_sigma_clipped_rejects_scratch_shortage_before_projection(
    tmp_path, monkeypatch
) -> None:
    consumed = False

    def contributions():
        nonlocal consumed
        consumed = True
        yield (
            torch.ones((2, 2)),
            torch.ones((2, 2)),
            torch.ones((2, 2)),
            None,
        )

    class _Usage:
        free = 1

    monkeypatch.setattr(execution_mod.shutil, "disk_usage", lambda _path: _Usage())

    with pytest.raises(OSError, match="sigma-clipped scratch requires"):
        execution_mod.spilled_sigma_clipped_stack(
            contributions(),
            n_observations=2,
            shape=(2, 2),
            scratch_dir=tmp_path / "too-small-clipped",
        )

    assert consumed is False


def test_spilled_sigma_clipping_uses_explicit_propagated_noise(tmp_path) -> None:
    values = [0.0, 0.0, 10.0]

    without_noise = execution_mod.spilled_sigma_clipped_stack(
        (
            (
                torch.tensor([[value]]),
                torch.ones((1, 1)),
                None,
                None,
            )
            for value in values
        ),
        n_observations=3,
        shape=(1, 1),
        scratch_dir=tmp_path / "relative-weights",
    )
    with_noise = execution_mod.spilled_sigma_clipped_stack(
        (
            (
                torch.tensor([[value]]),
                torch.ones((1, 1)),
                torch.tensor([[variance]]),
                None,
            )
            for value, variance in zip(values, (1.0, 1.0, 10_000.0))
        ),
        n_observations=3,
        shape=(1, 1),
        scratch_dir=tmp_path / "explicit-noise",
    )

    torch.testing.assert_close(without_noise.data, torch.zeros((1, 1)))
    torch.testing.assert_close(
        without_noise.overlap_count,
        torch.full((1, 1), 2, dtype=torch.int32),
    )
    torch.testing.assert_close(
        with_noise.data,
        torch.full((1, 1), 10.0 / 3.0),
    )
    torch.testing.assert_close(
        with_noise.overlap_count,
        torch.full((1, 1), 3, dtype=torch.int32),
    )
    assert with_noise.variance_weight is not None
    torch.testing.assert_close(
        with_noise.variance_weight,
        torch.full((1, 1), 10_002.0),
    )


def test_coadd_image_plane_sigma_clipped_rejects_overlap_background(tmp_path) -> None:
    path = tmp_path / "clip_overlap.fits"
    torchfits.write(
        str(path),
        torch.ones((8, 8)),
        header=_simple_header(),
        overwrite=True,
    )

    with pytest.raises(NotImplementedError, match="per-exposure"):
        coadd_image_plane(
            [ImageObservation(str(path))],
            stacking_method="sigma_clipped",
            cross_exposure_sky_corrector=CrossExposureSkyCorrector(),
        )


def test_tiled_overlap_background_uses_one_read_per_exposure_and_cleans_scratch(
    tmp_path,
    monkeypatch,
) -> None:
    from torchsky.map.sky import PolynomialSkyModel

    header = _simple_header()
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, 8),
        torch.linspace(-1.0, 1.0, 8),
        indexing="ij",
    )
    scene = torch.exp(-3.0 * (xx.square() + yy.square()))
    backgrounds = (
        torch.zeros_like(scene),
        4.0 + 0.5 * xx - 0.25 * yy,
        -3.0 + 0.2 * xx + 0.4 * yy,
    )
    observations = []
    for index, background in enumerate(backgrounds):
        path = tmp_path / f"overlap_stream_{index}.fits"
        torchfits.write(
            str(path),
            scene + background,
            header=header,
            overwrite=True,
        )
        observations.append(ImageObservation(str(path)))
    corrector = CrossExposureSkyCorrector(
        estimator=PolynomialSkyModel(order=1, statistic="huber")
    )
    expected = coadd_image_plane(
        observations,
        cross_exposure_sky_corrector=corrector,
    )

    read_calls: list[str] = []
    original_read = image_plane_mod.read

    def counted_read(path, **kwargs):
        read_calls.append(str(path))
        return original_read(path, **kwargs)

    monkeypatch.setattr(image_plane_common_mod, "read", counted_read)
    scratch = tmp_path / "overlap-scratch"
    tiled = coadd_image_plane(
        observations,
        spatial_tile_hw=(3, 5),
        scratch_dir=scratch,
        cross_exposure_sky_corrector=corrector,
    )

    assert read_calls == [str(observation.path) for observation in observations]
    torch.testing.assert_close(tiled.data, expected.data, atol=2.0e-4, rtol=2.0e-4)
    torch.testing.assert_close(tiled.weight, expected.weight, atol=2.0e-4, rtol=2.0e-4)
    assert tiled.context is not None and expected.context is not None
    torch.testing.assert_close(tiled.context, expected.context)
    assert tiled.overlap_count is not None and expected.overlap_count is not None
    torch.testing.assert_close(tiled.overlap_count, expected.overlap_count)
    assert scratch.is_dir()
    assert list(scratch.iterdir()) == []


@pytest.mark.parametrize(
    "estimator",
    [
        pytest.param(
            PolynomialSkyModel(order=1, statistic="huber"),
            id="polynomial",
        ),
        pytest.param(
            MeshSkyEstimator(
                block_size=4,
                statistic="median",
                mesh_filter="median",
                interpolate="bicubic",
            ),
            id="mesh",
        ),
    ],
)
def test_tiled_overlap_background_renders_compact_models(
    tmp_path,
    monkeypatch,
    estimator,
) -> None:
    header = _simple_header()
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, 8),
        torch.linspace(-1.0, 1.0, 8),
        indexing="ij",
    )
    scene = torch.exp(-3.0 * (xx.square() + yy.square()))
    observations = []
    for index, background in enumerate(
        (torch.zeros_like(scene), 4.0 + 0.5 * xx - 0.25 * yy)
    ):
        path = tmp_path / f"overlap_compact_{index}.fits"
        torchfits.write(str(path), scene + background, header=header, overwrite=True)
        observations.append(ImageObservation(str(path)))
    corrector = CrossExposureSkyCorrector(estimator=estimator)
    expected = coadd_image_plane(
        observations,
        cross_exposure_sky_corrector=corrector,
    )

    def reject_full_plane(*_args, **_kwargs):
        raise AssertionError("full difference-model rendering was used")

    monkeypatch.setattr(corrector, "fit_difference_model", reject_full_plane)
    result = coadd_image_plane(
        observations,
        spatial_tile_hw=(3, 5),
        scratch_dir=tmp_path / "overlap-compact-scratch",
        cross_exposure_sky_corrector=corrector,
    )

    torch.testing.assert_close(result.data, expected.data, atol=2.0e-4, rtol=2.0e-4)


def test_tiled_overlap_background_rejects_noncompact_estimators(tmp_path) -> None:
    header = _simple_header()
    observations = []
    for index in range(2):
        path = tmp_path / f"overlap_noncompact_{index}.fits"
        torchfits.write(
            str(path),
            torch.ones((8, 8), dtype=torch.float32) + index,
            header=header,
            overwrite=True,
        )
        observations.append(ImageObservation(str(path)))

    with pytest.raises(ValueError, match="compact tile rendering"):
        coadd_image_plane(
            observations,
            spatial_tile_hw=(3, 5),
            scratch_dir=tmp_path / "overlap-noncompact-scratch",
            cross_exposure_sky_corrector=CrossExposureSkyCorrector(
                estimator=lambda image, _mask: torch.zeros_like(image)
            ),
        )


def test_tiled_overlap_background_uses_one_global_chain_graph(tmp_path) -> None:
    from torchsky.map.sky import PolynomialSkyModel

    header = _simple_header()
    base = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    observations = []
    for index, offset in enumerate((2.0, 7.0, 17.0)):
        image_path = tmp_path / f"overlap_chain_{index}.fits"
        mask_path = tmp_path / f"overlap_chain_mask_{index}.fits"
        mask = torch.ones((8, 8), dtype=torch.float32)
        if index == 0:
            mask[:, :4] = 0.0
        elif index == 2:
            mask[:, 4:] = 0.0
        torchfits.write(
            str(image_path),
            base + offset,
            header=header,
            overwrite=True,
        )
        torchfits.write(str(mask_path), mask, header=header, overwrite=True)
        observations.append(
            ImageObservation(str(image_path), mask_path=str(mask_path))
        )
    corrector = CrossExposureSkyCorrector(
        estimator=PolynomialSkyModel(order=0, statistic="huber")
    )

    expected = coadd_image_plane(
        observations,
        cross_exposure_sky_corrector=corrector,
    )
    tiled = coadd_image_plane(
        observations,
        spatial_tile_hw=(8, 4),
        scratch_dir=tmp_path / "chain-scratch",
        cross_exposure_sky_corrector=corrector,
    )

    torch.testing.assert_close(tiled.data, expected.data, atol=2.0e-4, rtol=0.0)
    torch.testing.assert_close(tiled.weight, expected.weight)


def test_spilled_overlap_background_preserves_propagated_inverse_variance(
    tmp_path,
) -> None:
    from torchsky.map.sky import PolynomialSkyModel

    header = _simple_header()
    observations = []
    for index, offset in enumerate((0.0, 6.0)):
        path = tmp_path / f"overlap_variance_{index}.fits"
        torchfits.write(
            str(path),
            torch.arange(64, dtype=torch.float32).reshape(8, 8) + offset,
            header=header,
            overwrite=True,
        )
        observations.append(
            ImageObservation(
                str(path),
                weight=4.0,
                weight_is_inverse_variance=True,
            )
        )

    result = coadd_image_plane(
        observations,
        spatial_tile_hw=(3, 5),
        scratch_dir=tmp_path / "overlap-variance-scratch",
        cross_exposure_sky_corrector=CrossExposureSkyCorrector(
            estimator=PolynomialSkyModel(order=0, statistic="huber")
        ),
    )

    assert result.inverse_variance is not None
    torch.testing.assert_close(
        result.inverse_variance,
        torch.full((8, 8), 8.0),
    )


def test_tiled_overlap_background_rejects_scratch_shortage_before_projection(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "overlap-capacity.fits"
    torchfits.write(
        str(path),
        torch.ones((8, 8)),
        header=_simple_header(),
        overwrite=True,
    )
    prepared = False
    original_prepare = image_plane_mod.prepare_image_observation_data

    def counted_prepare(*args, **kwargs):
        nonlocal prepared
        prepared = True
        return original_prepare(*args, **kwargs)

    class _Usage:
        free = 1

    monkeypatch.setattr(
        image_plane_mod,
        "prepare_image_observation_data",
        counted_prepare,
    )
    monkeypatch.setattr(execution_mod.shutil, "disk_usage", lambda _path: _Usage())

    with pytest.raises(OSError, match="overlap background scratch requires"):
        coadd_image_plane(
            [ImageObservation(str(path)), ImageObservation(str(path))],
            spatial_tile_hw=(4, 4),
            scratch_dir=tmp_path / "too-small-overlap",
            cross_exposure_sky_corrector=CrossExposureSkyCorrector(),
        )

    assert prepared is False


def test_stack_map_product_passes_mean_stacking_to_single_band_coadd(tmp_path) -> None:
    path_a = tmp_path / "product_mean_a.fits"
    path_b = tmp_path / "product_mean_b.fits"
    header = _simple_header()
    data_a = torch.full((8, 8), 2.0, dtype=torch.float32)
    data_b = torch.full((8, 8), 10.0, dtype=torch.float32)
    torchfits.write(str(path_a), data_a, header=header, overwrite=True)
    torchfits.write(str(path_b), data_b, header=header, overwrite=True)

    observations = [
        ImageObservation(str(path_a), weight=1.0),
        ImageObservation(str(path_b), weight=5.0),
    ]
    product = plan_map_product(observations)

    result = stack_product(
        observations,
        product,
        stacking_method="mean",
    )

    torch.testing.assert_close(
        result.data.squeeze(0),
        torch.full((8, 8), 6.0, dtype=torch.float32),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        result.weight.squeeze(0),
        torch.full((8, 8), 2.0, dtype=torch.float32),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize("stacking_method", ["median", "sigma_clipped"])
def test_coadd_image_plane_cube_rejects_order_statistic_stacking(
    tmp_path, stacking_method
) -> None:
    path = tmp_path / "cube_median.fits"
    header = _simple_header()
    data = torch.ones((8, 8), dtype=torch.float32)
    torchfits.write(str(path), data, header=header, overwrite=True)

    observations = [ImageObservation(str(path), wavelength=5000.0)]
    spectral_axis = SpectralAxis(
        values=torch.tensor([5000.0], dtype=torch.float64),
        unit="Angstrom",
        type="WAVE",
    )

    with pytest.raises(ValueError, match="single-band"):
        coadd_image_plane_cube(
            observations,
            spectral_axis=spectral_axis,
            stacking_method=stacking_method,
        )


def test_coadd_image_plane_respects_target_header_shape(tmp_path) -> None:
    path = tmp_path / "a.fits"
    header = _simple_header()
    data = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    torchfits.write(str(path), data, header=header, overwrite=True)

    target_header = dict(header)
    target_header["NAXIS1"] = 6
    target_header["NAXIS2"] = 5

    result = coadd_image_plane(
        [ImageObservation(str(path))],
        target_header=target_header,
    )

    assert tuple(result.data.shape) == (5, 6)
    assert tuple(result.weight.shape) == (5, 6)


@pytest.mark.parametrize("kernel", ("bilinear", "drizzle_square"))
def test_coadd_image_plane_ignores_non_finite_pixels(
    tmp_path,
    kernel: str,
) -> None:
    path_a = tmp_path / "a.fits"
    path_b = tmp_path / "b.fits"
    header = _simple_header()
    data_a = torch.ones((8, 8), dtype=torch.float32)
    data_b = 3.0 * torch.ones((8, 8), dtype=torch.float32)
    data_a[2, 3] = torch.nan
    torchfits.write(str(path_a), data_a, header=header, overwrite=True)
    torchfits.write(str(path_b), data_b, header=header, overwrite=True)

    result = coadd_image_plane(
        [ImageObservation(str(path_a)), ImageObservation(str(path_b))],
        resampling_kernel=kernel,
    )

    assert torch.isfinite(result.data).all()
    assert result.data[2, 3].item() == 3.0
    assert result.weight[2, 3].item() == 1.0


def test_coadd_image_plane_applies_known_photometric_scale_and_background(
    tmp_path,
) -> None:
    path_ref = tmp_path / "ref.fits"
    path_raw = tmp_path / "raw.fits"
    header = _simple_header()
    truth = torch.full((8, 8), 2.0, dtype=torch.float32)
    raw = torch.full((8, 8), 6.0, dtype=torch.float32)
    torchfits.write(str(path_ref), truth, header=header, overwrite=True)
    torchfits.write(str(path_raw), raw, header=header, overwrite=True)

    result = coadd_image_plane(
        [
            ImageObservation(str(path_ref)),
            ImageObservation(
                str(path_raw),
                photometric_scale=2.0,
                background_offset=5.0,
            ),
        ]
    )

    torch.testing.assert_close(result.data, truth, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        result.weight,
        torch.full((8, 8), 1.25, dtype=torch.float32),
        atol=1e-5,
        rtol=1e-5,
    )


def test_coadd_image_plane_rescales_weight_map_with_photometric_scale(tmp_path) -> None:
    path = tmp_path / "scaled.fits"
    weight_path = tmp_path / "scaled_wht.fits"
    header = _simple_header()
    raw = torch.full((8, 8), 7.0, dtype=torch.float32)
    weight = torch.full((8, 8), 8.0, dtype=torch.float32)
    torchfits.write(str(path), raw, header=header, overwrite=True)
    torchfits.write(str(weight_path), weight, header=header, overwrite=True)

    result = coadd_image_plane(
        [
            ImageObservation(
                str(path),
                weight_path=str(weight_path),
                photometric_scale=2.0,
                background_offset=5.0,
            )
        ]
    )

    torch.testing.assert_close(
        result.data,
        torch.full((8, 8), 4.0, dtype=torch.float32),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        result.weight,
        torch.full((8, 8), 2.0, dtype=torch.float32),
        atol=1e-5,
        rtol=1e-5,
    )


def test_coadd_image_plane_rejects_non_2d_images(tmp_path) -> None:
    path = tmp_path / "cube.fits"
    header = _simple_header()
    header["NAXIS"] = 3
    header["NAXIS3"] = 2
    data = torch.ones((2, 8, 8), dtype=torch.float32)
    torchfits.write(str(path), data, header=header, overwrite=True)

    with pytest.raises(ValueError, match="2D images only"):
        coadd_image_plane([ImageObservation(str(path))])


def test_coadd_image_plane_rejects_unknown_resampling_kernel(tmp_path) -> None:
    path = tmp_path / "a.fits"
    header = _simple_header()
    data = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    torchfits.write(str(path), data, header=header, overwrite=True)

    with pytest.raises(ValueError, match="resampling_kernel"):
        coadd_image_plane(
            [ImageObservation(str(path))],
            target_header={**header, "CRVAL1": float(header["CRVAL1"]) + 0.01},
            resampling_kernel="spline5",
        )


def test_coadd_image_plane_rejects_unknown_value_semantics(tmp_path) -> None:
    path = tmp_path / "a.fits"
    header = _simple_header()
    data = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    torchfits.write(str(path), data, header=header, overwrite=True)

    with pytest.raises(ValueError, match="value_semantics"):
        coadd_image_plane(
            [ImageObservation(str(path))],
            value_semantics="counts",
        )


def test_coadd_image_plane_rejects_unknown_stacking_method(tmp_path) -> None:
    path = tmp_path / "stacking_unknown.fits"
    header = _simple_header()
    data = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    torchfits.write(str(path), data, header=header, overwrite=True)

    with pytest.raises(ValueError, match="stacking_method"):
        coadd_image_plane(
            [ImageObservation(str(path))],
            stacking_method="trimmed_mean",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("photometric_scale", 0.0),
        ("photometric_scale", -1.0),
        ("background_offset", float("nan")),
    ],
)
def test_coadd_image_plane_rejects_invalid_calibration_metadata(
    tmp_path: Path,
    field: str,
    value: float,
) -> None:
    path = tmp_path / "invalid_calibration.fits"
    header = _simple_header()
    data = torch.ones((8, 8), dtype=torch.float32)
    torchfits.write(str(path), data, header=header, overwrite=True)

    kwargs = {field: value}
    with pytest.raises(ValueError, match=field):
        coadd_image_plane([ImageObservation(str(path), **kwargs)])


@pytest.mark.parametrize("kernel", ["nearest", "bilinear", "bicubic"])
def test_coadd_image_plane_passes_resampling_kernel_to_grid_sample(
    tmp_path, monkeypatch, kernel: str
) -> None:
    path_a = tmp_path / f"a_{kernel}.fits"
    path_b = tmp_path / f"b_{kernel}.fits"
    header_a = _simple_header()
    header_b = dict(header_a)
    header_b["CRVAL1"] = float(header_b["CRVAL1"]) + 0.02
    data = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    torchfits.write(str(path_a), data, header=header_a, overwrite=True)
    torchfits.write(str(path_b), data + 1.0, header=header_b, overwrite=True)

    calls: list[str] = []
    original = resampling_mod.F.grid_sample

    def recorded(*args, **kwargs):
        calls.append(kwargs["mode"])
        return original(*args, **kwargs)

    monkeypatch.setattr(resampling_mod.F, "grid_sample", recorded)

    result = coadd_image_plane(
        [ImageObservation(str(path_a)), ImageObservation(str(path_b))],
        resampling_kernel=kernel,
    )

    assert result.data.shape == (8, 8)
    assert calls
    assert set(calls) == {kernel}


@pytest.mark.parametrize("kernel", ["bilinear", "catmull_rom"])
def test_sample_image_plane_batched_channels_matches_separate_sampling(
    kernel: str,
) -> None:
    image = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    weight = 1.0 + torch.flip(image, dims=(0,)) / 64.0
    support = torch.ones_like(image, dtype=torch.float32)
    support[:, :2] = 0.0
    x_src = torch.linspace(0.25, 6.75, 8, dtype=torch.float32).repeat(8, 1)
    y_src = torch.linspace(0.1, 6.9, 8, dtype=torch.float32).unsqueeze(1).repeat(1, 8)
    grid = torch.stack(
        [
            (2.0 * x_src / 7.0) - 1.0,
            (2.0 * y_src / 7.0) - 1.0,
        ],
        dim=-1,
    )
    stacked = torch.stack([image, weight, support], dim=0)

    combined = resampling_mod.sample_image_plane_batched_channels(
        stacked,
        x_src=x_src,
        y_src=y_src,
        grid=grid,
        kernel=kernel,
    )
    separate = torch.stack(
        [
            resampling_mod.sample_image_plane_batched(
                plane,
                x_src=x_src,
                y_src=y_src,
                grid=grid,
                kernel=kernel,
            )
            for plane in (image, weight, support)
        ],
        dim=0,
    )

    torch.testing.assert_close(combined, separate, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("kernel", ["lanczos3", "lanczos5"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_lanczos_sampling_preserves_constant_on_complete_support(kernel: str, dtype) -> None:
    image = torch.ones((16, 16), dtype=dtype)
    x_src = torch.tensor([[5.25, 8.6], [6.7, 9.1]], dtype=dtype)
    y_src = torch.tensor([[6.4, 7.2], [8.3, 5.8]], dtype=dtype)
    grid = torch.stack(
        [
            (2.0 * x_src / 15.0) - 1.0,
            (2.0 * y_src / 15.0) - 1.0,
        ],
        dim=-1,
    )

    sampled = resampling_mod.sample_image_plane(
        image,
        x_src=x_src,
        y_src=y_src,
        grid=grid,
        kernel=kernel,
    )

    # The dense tensor reduction rounds each signed stencil term. Its forward
    # error bound is gamma_n * sum(abs(weights)), not a platform-specific ULP.
    support = resampling_mod.kernel_support(kernel)
    offsets = torch.arange(-support + 1, support + 1, dtype=dtype)
    norms = []
    for coordinate in (x_src, y_src):
        delta = (coordinate - coordinate.floor()).unsqueeze(-1) - offsets
        coefficients = resampling_mod.kernel_weight(kernel, delta)
        coefficients = coefficients / coefficients.sum(-1, keepdim=True)
        norms.append(coefficients.abs().sum(-1))
    operations = (2 * support) ** 2 + 4 * support
    error = operations * torch.finfo(dtype).eps
    bound = error / (1 - error) * float((norms[0] * norms[1]).max())
    torch.testing.assert_close(sampled, torch.ones_like(sampled), atol=bound, rtol=0.0)


@pytest.mark.parametrize(
    "kernel",
    ["nearest", "bilinear", "bicubic", "lanczos3", "catmull_rom"],
)
def test_sample_image_plane_diagonal_variance_matches_basis_oracle(kernel) -> None:
    height, width = 5, 6
    source_variance = torch.arange(
        1,
        height * width + 1,
        dtype=torch.float32,
    ).reshape(height, width)
    x_src = torch.tensor([[1.25, 3.6], [0.2, 4.75]], dtype=torch.float32)
    y_src = torch.tensor([[2.4, 1.1], [3.7, 0.3]], dtype=torch.float32)
    grid = torch.stack(
        [
            (2.0 * x_src / (width - 1)) - 1.0,
            (2.0 * y_src / (height - 1)) - 1.0,
        ],
        dim=-1,
    )
    basis = torch.eye(height * width, dtype=torch.float32).reshape(
        height * width,
        height,
        width,
    )
    coefficients = resampling_mod.sample_image_plane_batched_channels(
        basis,
        x_src=x_src,
        y_src=y_src,
        grid=grid,
        kernel=kernel,
    )
    expected = (torch.square(coefficients) * source_variance.reshape(-1, 1, 1)).sum(
        dim=0
    )

    actual = resampling_mod.sample_image_plane_diagonal_variance(
        source_variance,
        x_src=x_src,
        y_src=y_src,
        kernel=kernel,
    )

    torch.testing.assert_close(actual, expected, atol=2.0e-6, rtol=2.0e-6)


def test_batched_quadrature_variance_squares_mean_coefficients() -> None:
    height, width = 5, 6
    source_variance = torch.arange(
        1,
        height * width + 1,
        dtype=torch.float32,
    ).reshape(height, width)
    x_src = torch.tensor(
        [
            [[1.2, 3.4], [2.1, 4.2]],
            [[1.7, 3.8], [2.6, 4.6]],
        ],
        dtype=torch.float32,
    )
    y_src = torch.tensor(
        [
            [[2.3, 1.2], [3.1, 0.4]],
            [[2.7, 1.6], [3.5, 0.8]],
        ],
        dtype=torch.float32,
    )
    sample_valid = torch.tensor(
        [
            [[True, True], [True, False]],
            [[True, False], [True, True]],
        ]
    )
    grid = torch.stack(
        [
            (2.0 * x_src / (width - 1)) - 1.0,
            (2.0 * y_src / (height - 1)) - 1.0,
        ],
        dim=-1,
    )
    basis = torch.eye(height * width, dtype=torch.float32).reshape(
        height * width,
        height,
        width,
    )
    sampled_basis = resampling_mod.sample_image_plane_batched_channels(
        basis,
        x_src=x_src,
        y_src=y_src,
        grid=grid,
        kernel="bilinear",
    )
    mean_coefficients = (
        sampled_basis * sample_valid.unsqueeze(1).to(dtype=torch.float32)
    ).mean(dim=0)
    expected = (
        torch.square(mean_coefficients) * source_variance.reshape(-1, 1, 1)
    ).sum(dim=0)

    actual = resampling_mod.sample_image_plane_diagonal_variance(
        source_variance,
        x_src=x_src,
        y_src=y_src,
        kernel="bilinear",
        average_coordinate_batch=True,
        sample_valid=sample_valid,
    )

    torch.testing.assert_close(actual, expected, atol=2.0e-6, rtol=2.0e-6)


def test_bilinear_diagonal_variance_matches_monte_carlo() -> None:
    generator = torch.Generator().manual_seed(20260831)
    draws = 20_000
    height, width = 4, 5
    source_variance = torch.linspace(
        0.5,
        2.0,
        height * width,
        dtype=torch.float32,
    ).reshape(height, width)
    x_src = torch.full((1, 1), 2.3, dtype=torch.float32)
    y_src = torch.full((1, 1), 1.4, dtype=torch.float32)
    expected = resampling_mod.sample_image_plane_diagonal_variance(
        source_variance,
        x_src=x_src,
        y_src=y_src,
        kernel="bilinear",
    ).squeeze()
    noise = torch.randn(
        (draws, height, width),
        generator=generator,
        dtype=torch.float32,
    ) * torch.sqrt(source_variance)
    x_batch = x_src.expand(draws, 1, 1)
    y_batch = y_src.expand(draws, 1, 1)
    grid = torch.stack(
        [
            (2.0 * x_batch / (width - 1)) - 1.0,
            (2.0 * y_batch / (height - 1)) - 1.0,
        ],
        dim=-1,
    )
    samples = resampling_mod.sample_image_plane_batched_channels(
        noise.unsqueeze(1),
        x_src=x_batch,
        y_src=y_batch,
        grid=grid,
        kernel="bilinear",
    ).reshape(draws)

    empirical = torch.var(samples, unbiased=True)
    assert float(empirical) == pytest.approx(float(expected), rel=0.03)


def test_bilinear_local_correlation_factor_predicts_large_aperture_variance() -> (
    None
):
    generator = torch.Generator().manual_seed(20260901)
    draws = 10_000
    width = 64
    aperture_pixels = width - 1
    x_src = torch.arange(0.5, width - 0.5, dtype=torch.float32).unsqueeze(0)
    y_src = torch.zeros_like(x_src)
    source_variance = torch.ones((2, width), dtype=torch.float32)
    diagonal = resampling_mod.sample_image_plane_diagonal_variance(
        source_variance,
        x_src=x_src,
        y_src=y_src,
        kernel="bilinear",
    )
    local_factor = diagonal.reciprocal()
    torch.testing.assert_close(local_factor, torch.full_like(local_factor, 2.0))

    noise = torch.randn(
        (draws, 2, width),
        generator=generator,
        dtype=torch.float32,
    )
    x_batch = x_src.expand(draws, 1, aperture_pixels)
    y_batch = y_src.expand(draws, 1, aperture_pixels)
    grid = torch.stack(
        [
            (2.0 * x_batch / (width - 1)) - 1.0,
            y_batch * 2.0 - 1.0,
        ],
        dim=-1,
    )
    samples = resampling_mod.sample_image_plane_batched_channels(
        noise.unsqueeze(1),
        x_src=x_batch,
        y_src=y_batch,
        grid=grid,
        kernel="bilinear",
    ).squeeze(1)
    aperture_sums = samples.sum(dim=(-2, -1))
    empirical_factor = torch.var(aperture_sums, unbiased=True) / diagonal.sum()
    exact_finite_aperture_factor = 2.0 - 1.0 / float(aperture_pixels)

    assert float(empirical_factor) == pytest.approx(
        exact_finite_aperture_factor,
        rel=0.03,
    )


def test_finite_aperture_uncertainty_keeps_covariance_across_tiles(
    tmp_path: Path,
) -> None:
    source_header = _simple_header()
    target_header = {**source_header, "CRPIX1": source_header["CRPIX1"] + 0.5}
    path = tmp_path / "half-pixel-noise.fits"
    torchfits.write(
        str(path),
        torch.zeros((8, 8), dtype=torch.float32),
        header=source_header,
        overwrite=True,
    )
    observations = [
        ImageObservation(
            str(path),
            weight=1.0,
            weight_is_inverse_variance=True,
        )
    ]
    result = coadd_image_plane(
        observations,
        target_header=target_header,
        resampling_kernel="bilinear",
        spatial_tile_hw=(2, 3),
    )
    aperture = torch.zeros_like(result.data)
    aperture[3, 1:7] = 1.0

    with torch.no_grad():
        uncertainty = image_plane_mod.evaluate_image_plane_aperture_uncertainty(
            observations,
            result,
            aperture,
            resampling_kernel="bilinear",
            spatial_tile_hw=(2, 3),
        )

    aperture_pixels = 6
    expected_factor = 2.0 - 1.0 / aperture_pixels
    assert float(uncertainty.diagonal_variance) == pytest.approx(
        0.5 * aperture_pixels,
        rel=0.0,
        abs=2.0e-5,
    )
    assert float(uncertainty.variance) == pytest.approx(
        aperture_pixels - 0.5,
        rel=0.0,
        abs=3.0e-5,
    )
    assert float(uncertainty.correlation_factor) == pytest.approx(
        expected_factor,
        rel=0.0,
        abs=2.0e-5,
    )

    generator = torch.Generator().manual_seed(20260901)
    draws = 10_000
    target_x = torch.arange(1, 7, dtype=torch.float32)
    target_y = torch.full_like(target_x, 3.0)
    ra, dec = torchsky.WCS(target_header).pixel_to_world(target_x, target_y)
    source_x, source_y = torchsky.WCS(source_header).world_to_pixel(ra, dec)
    x_batch = source_x.expand(draws, 1, -1)
    y_batch = source_y.expand(draws, 1, -1)
    grid = torch.stack(
        [
            (2.0 * x_batch / 7.0) - 1.0,
            (2.0 * y_batch / 7.0) - 1.0,
        ],
        dim=-1,
    )
    detector_noise = torch.randn(
        (draws, 8, 8), generator=generator, dtype=torch.float32
    )
    resampled = resampling_mod.sample_image_plane_batched_channels(
        detector_noise.unsqueeze(1),
        x_src=x_batch,
        y_src=y_batch,
        grid=grid,
        kernel="bilinear",
    )
    empirical_variance = torch.var(
        resampled.sum(dim=(-2, -1)), unbiased=True
    )
    assert float(empirical_variance) == pytest.approx(
        float(uncertainty.variance), rel=0.03
    )


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_finite_aperture_uncertainty_device_parity(
    tmp_path: Path,
    device: str,
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    source_header = _simple_header()
    target_header = {**source_header, "CRPIX1": source_header["CRPIX1"] + 0.5}
    path = tmp_path / f"aperture-{device}.fits"
    torchfits.write(
        str(path), torch.zeros((8, 8)), header=source_header, overwrite=True
    )
    observations = [
        ImageObservation(
            str(path), weight=1.0, weight_is_inverse_variance=True
        )
    ]
    cpu_result = coadd_image_plane(
        observations,
        target_header=target_header,
        resampling_kernel="bilinear",
        device="cpu",
    )
    accelerator_result = coadd_image_plane(
        observations,
        target_header=target_header,
        resampling_kernel="bilinear",
        device=device,
    )
    aperture = torch.zeros((8, 8), dtype=torch.float32)
    aperture[3, 1:7] = 1.0

    cpu = image_plane_mod.evaluate_image_plane_aperture_uncertainty(
        observations,
        cpu_result,
        aperture,
        resampling_kernel="bilinear",
        spatial_tile_hw=(2, 3),
        device="cpu",
    )
    accelerator = image_plane_mod.evaluate_image_plane_aperture_uncertainty(
        observations,
        accelerator_result,
        aperture.to(device),
        resampling_kernel="bilinear",
        spatial_tile_hw=(2, 3),
        device=device,
    )

    torch.testing.assert_close(
        accelerator.variance.cpu(),
        cpu.variance.to(dtype=accelerator.variance.dtype),
        rtol=5.0e-5,
        atol=5.0e-6,
    )


@pytest.mark.parametrize(
    "kernel",
    ["bilinear", "drizzle_square", "conservative_polygon", "splat_box"],
)
def test_finite_aperture_uncertainty_matches_source_basis_oracle(
    tmp_path: Path,
    kernel: str,
) -> None:
    source_header = {**_simple_header(), "NAXIS1": 4, "NAXIS2": 4}
    target_header = {**source_header, "CRPIX1": source_header["CRPIX1"] + 0.35}
    path = tmp_path / f"aperture-{kernel}.fits"
    image = torch.zeros((4, 4), dtype=torch.float32)
    torchfits.write(str(path), image, header=source_header, overwrite=True)
    observations = [
        ImageObservation(
            str(path),
            weight=1.0,
            weight_is_inverse_variance=True,
        )
    ]
    result = coadd_image_plane(
        observations,
        target_header=target_header,
        resampling_kernel=kernel,
        drizzle_pixfrac=0.8,
    )
    aperture = torch.zeros_like(result.data)
    aperture[1:3, 1:3] = torch.tensor(
        [[1.0, 0.75], [0.5, -0.25]], dtype=aperture.dtype
    )

    uncertainty = image_plane_mod.evaluate_image_plane_aperture_uncertainty(
        observations,
        result,
        aperture,
        resampling_kernel=kernel,
        drizzle_pixfrac=0.8,
        spatial_tile_hw=(2, 2),
    )

    responses = []
    for source_index in range(image.numel()):
        basis = torch.zeros_like(image)
        basis.reshape(-1)[source_index] = 1.0
        torchfits.write(str(path), basis, header=source_header, overwrite=True)
        basis_result = coadd_image_plane(
            observations,
            target_header=target_header,
            resampling_kernel=kernel,
            drizzle_pixfrac=0.8,
        )
        responses.append(torch.sum(aperture * basis_result.data))
    expected = torch.sum(torch.square(torch.stack(responses)))

    torch.testing.assert_close(
        uncertainty.variance,
        expected.to(dtype=uncertainty.variance.dtype),
        rtol=2.0e-5,
        atol=2.0e-6,
    )


@pytest.mark.parametrize(
    "kernel",
    sorted(image_plane_mod._FINITE_APERTURE_COVARIANCE_KERNELS),
)
def test_finite_aperture_uncertainty_single_pixel_matches_diagonal_plane(
    tmp_path: Path,
    kernel: str,
) -> None:
    source_header = _simple_header()
    target_header = {**source_header, "CRPIX1": source_header["CRPIX1"] + 0.2}
    path = tmp_path / f"single-pixel-{kernel}.fits"
    torchfits.write(
        str(path), torch.zeros((8, 8)), header=source_header, overwrite=True
    )
    observations = [
        ImageObservation(
            str(path), weight=2.0, weight_is_inverse_variance=True
        )
    ]
    result = coadd_image_plane(
        observations,
        target_header=target_header,
        resampling_kernel=kernel,
        drizzle_pixfrac=0.7,
    )
    aperture = torch.zeros_like(result.data)
    aperture[3, 3] = 1.0

    uncertainty = image_plane_mod.evaluate_image_plane_aperture_uncertainty(
        observations,
        result,
        aperture,
        resampling_kernel=kernel,
        drizzle_pixfrac=0.7,
        spatial_tile_hw=(3, 3),
    )

    assert result.inverse_variance is not None
    expected = result.inverse_variance[3, 3].reciprocal()
    torch.testing.assert_close(
        uncertainty.variance,
        expected.to(dtype=uncertainty.variance.dtype),
        rtol=3.0e-5,
        atol=3.0e-6,
    )


def test_finite_aperture_uncertainty_matches_distorted_wcs_basis_oracle(
    tmp_path: Path,
) -> None:
    source_header = {
        **_simple_header(),
        "NAXIS1": 4,
        "NAXIS2": 4,
        "CTYPE1": "RA---TAN-SIP",
        "CTYPE2": "DEC--TAN-SIP",
        "A_ORDER": 2,
        "B_ORDER": 2,
        "A_2_0": 2.0e-2,
        "A_1_1": -1.0e-2,
        "B_0_2": -1.5e-2,
        "B_1_1": 1.0e-2,
    }
    target_header = {**source_header, "CRPIX1": source_header["CRPIX1"] + 0.3}
    path = tmp_path / "aperture-sip.fits"
    image = torch.zeros((4, 4), dtype=torch.float32)
    torchfits.write(str(path), image, header=source_header, overwrite=True)
    observations = [
        ImageObservation(
            str(path), weight=1.0, weight_is_inverse_variance=True
        )
    ]
    result = coadd_image_plane(
        observations,
        target_header=target_header,
        resampling_kernel="bilinear",
    )
    aperture = torch.zeros_like(result.data)
    aperture[1:3, 1:3] = 1.0
    uncertainty = image_plane_mod.evaluate_image_plane_aperture_uncertainty(
        observations,
        result,
        aperture,
        resampling_kernel="bilinear",
        spatial_tile_hw=(2, 2),
    )

    responses = []
    for source_index in range(image.numel()):
        basis = torch.zeros_like(image)
        basis.reshape(-1)[source_index] = 1.0
        torchfits.write(str(path), basis, header=source_header, overwrite=True)
        basis_result = coadd_image_plane(
            observations,
            target_header=target_header,
            resampling_kernel="bilinear",
        )
        responses.append(torch.sum(aperture * basis_result.data))
    expected = torch.sum(torch.square(torch.stack(responses)))

    torch.testing.assert_close(
        uncertainty.variance,
        expected.to(dtype=uncertainty.variance.dtype),
        rtol=3.0e-5,
        atol=3.0e-6,
    )


def test_distorted_surface_brightness_area_scaling_is_tile_invariant(
    tmp_path: Path,
) -> None:
    source_header = {
        **_simple_header(),
        "CTYPE1": "RA---TAN-SIP",
        "CTYPE2": "DEC--TAN-SIP",
        "A_ORDER": 2,
        "B_ORDER": 2,
        "A_2_0": 2.0e-3,
        "B_0_2": -1.5e-3,
    }
    target_header = {**source_header, "CRPIX1": source_header["CRPIX1"] + 0.3}
    path = tmp_path / "tile-area-sip.fits"
    torchfits.write(
        str(path),
        torch.arange(64, dtype=torch.float32).reshape(8, 8),
        header=source_header,
        overwrite=True,
    )
    observations = [
        ImageObservation(
            str(path), weight=2.0, weight_is_inverse_variance=True
        )
    ]

    full = coadd_image_plane(
        observations,
        target_header=target_header,
        resampling_kernel="bilinear",
    )
    tiled = coadd_image_plane(
        observations,
        target_header=target_header,
        resampling_kernel="bilinear",
        spatial_tile_hw=(3, 3),
    )

    torch.testing.assert_close(tiled.data, full.data, rtol=2.0e-5, atol=2.0e-5)
    torch.testing.assert_close(tiled.weight, full.weight, rtol=2.0e-5, atol=2.0e-5)
    assert tiled.inverse_variance is not None
    assert full.inverse_variance is not None
    torch.testing.assert_close(
        tiled.inverse_variance,
        full.inverse_variance,
        rtol=3.0e-5,
        atol=3.0e-5,
    )


def test_flux_aperture_uncertainty_matches_source_basis_oracle(
    tmp_path: Path,
) -> None:
    source_header = {**_simple_header(), "NAXIS1": 4, "NAXIS2": 4}
    target_header = {**source_header, "CRPIX1": source_header["CRPIX1"] + 0.25}
    path = tmp_path / "aperture-flux.fits"
    image = torch.zeros((4, 4), dtype=torch.float32)
    torchfits.write(str(path), image, header=source_header, overwrite=True)
    observations = [
        ImageObservation(
            str(path), weight=3.0, weight_is_inverse_variance=True
        )
    ]
    result = coadd_image_plane(
        observations,
        target_header=target_header,
        resampling_kernel="bilinear",
        value_semantics="flux",
    )
    aperture = torch.zeros_like(result.data)
    aperture[1:3, 1:3] = 1.0
    uncertainty = image_plane_mod.evaluate_image_plane_aperture_uncertainty(
        observations,
        result,
        aperture,
        resampling_kernel="bilinear",
        value_semantics="flux",
        spatial_tile_hw=(2, 2),
    )

    responses = []
    for source_index in range(image.numel()):
        basis = torch.zeros_like(image)
        basis.reshape(-1)[source_index] = 1.0
        torchfits.write(str(path), basis, header=source_header, overwrite=True)
        basis_result = coadd_image_plane(
            observations,
            target_header=target_header,
            resampling_kernel="bilinear",
            value_semantics="flux",
        )
        responses.append(torch.sum(aperture * basis_result.data))
    expected = torch.sum(torch.square(torch.stack(responses))) / 3.0

    torch.testing.assert_close(
        uncertainty.variance,
        expected.to(dtype=uncertainty.variance.dtype),
        rtol=3.0e-5,
        atol=3.0e-6,
    )


def test_aperture_uncertainty_matches_spatial_noise_map_basis_oracle(
    tmp_path: Path,
) -> None:
    source_header = {**_simple_header(), "NAXIS1": 4, "NAXIS2": 4}
    target_header = {**source_header, "CRPIX2": source_header["CRPIX2"] + 0.4}
    image_path = tmp_path / "aperture-spatial-noise.fits"
    weight_path = tmp_path / "aperture-spatial-ivar.fits"
    image = torch.zeros((4, 4), dtype=torch.float32)
    source_inverse_variance = torch.linspace(0.5, 2.0, 16).reshape(4, 4)
    torchfits.write(str(image_path), image, header=source_header, overwrite=True)
    torchfits.write(
        str(weight_path),
        source_inverse_variance,
        header=source_header,
        overwrite=True,
    )
    observations = [
        ImageObservation(
            str(image_path),
            weight_path=str(weight_path),
            weight_is_inverse_variance=True,
        )
    ]
    result = coadd_image_plane(
        observations,
        target_header=target_header,
        resampling_kernel="bilinear",
    )
    aperture = torch.zeros_like(result.data)
    aperture[1:3, 1:3] = torch.tensor(
        [[1.0, -0.5], [0.25, 0.75]], dtype=aperture.dtype
    )
    uncertainty = image_plane_mod.evaluate_image_plane_aperture_uncertainty(
        observations,
        result,
        aperture,
        resampling_kernel="bilinear",
        spatial_tile_hw=(2, 2),
    )

    contributions = []
    for source_index in range(image.numel()):
        basis = torch.zeros_like(image)
        basis.reshape(-1)[source_index] = 1.0
        torchfits.write(
            str(image_path), basis, header=source_header, overwrite=True
        )
        basis_result = coadd_image_plane(
            observations,
            target_header=target_header,
            resampling_kernel="bilinear",
        )
        response = torch.sum(aperture * basis_result.data)
        contributions.append(
            torch.square(response)
            / source_inverse_variance.reshape(-1)[source_index]
        )
    expected = torch.sum(torch.stack(contributions))

    torch.testing.assert_close(
        uncertainty.variance,
        expected.to(dtype=uncertainty.variance.dtype),
        rtol=3.0e-5,
        atol=3.0e-6,
    )


def test_finite_aperture_uncertainty_rejects_missing_noise_and_global_kernels(
    tmp_path: Path,
) -> None:
    header = _simple_header()
    path = tmp_path / "no-noise.fits"
    torchfits.write(
        str(path), torch.zeros((8, 8)), header=header, overwrite=True
    )
    result = coadd_image_plane([ImageObservation(str(path))])
    aperture = torch.ones_like(result.data)

    with pytest.raises(ValueError, match="explicit detector inverse variance"):
        image_plane_mod.evaluate_image_plane_aperture_uncertainty(
            [ImageObservation(str(path))],
            result,
            aperture,
        )

    noisy = [
        ImageObservation(
            str(path), weight=1.0, weight_is_inverse_variance=True
        )
    ]
    noisy_result = coadd_image_plane(noisy)
    changed_noise = [
        ImageObservation(
            str(path), weight=2.0, weight_is_inverse_variance=True
        )
    ]
    with pytest.raises(ValueError, match="observation-operator provenance"):
        image_plane_mod.evaluate_image_plane_aperture_uncertainty(
            changed_noise,
            noisy_result,
            aperture,
        )
    with pytest.raises(ValueError, match="execution provenance does not match"):
        image_plane_mod.evaluate_image_plane_aperture_uncertainty(
            noisy,
            noisy_result,
            aperture,
            resampling_kernel="nearest",
        )
    with pytest.raises(ValueError, match="finite-support"):
        image_plane_mod.evaluate_image_plane_aperture_uncertainty(
            noisy,
            noisy_result,
            aperture,
            resampling_kernel="fft_shift",
        )

    background_result = coadd_image_plane(
        noisy,
        background_estimator=lambda image, _mask: torch.full_like(
            image, image.mean()
        ),
    )
    with pytest.raises(ValueError, match="data-dependent background"):
        image_plane_mod.evaluate_image_plane_aperture_uncertainty(
            noisy,
            background_result,
            aperture,
        )


@pytest.mark.parametrize(
    "kernel",
    [
        "lanczos3",
        "lanczos5",
        "mitchell_netravali",
        "catmull_rom",
        "drizzle_point",
        "drizzle_square",
        "splat_point",
        "splat_box",
    ],
)
def test_coadd_image_plane_accepts_custom_high_accuracy_kernels(
    tmp_path, kernel: str
) -> None:
    path_a = tmp_path / f"a_{kernel}.fits"
    path_b = tmp_path / f"b_{kernel}.fits"
    header_a = _simple_header()
    header_b = dict(header_a)
    header_b["CRVAL1"] = float(header_b["CRVAL1"]) + 0.02
    data = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    torchfits.write(str(path_a), data, header=header_a, overwrite=True)
    torchfits.write(str(path_b), data + 1.0, header=header_b, overwrite=True)

    result = coadd_image_plane(
        [ImageObservation(str(path_a)), ImageObservation(str(path_b))],
        resampling_kernel=kernel,
    )

    assert result.data.shape == (8, 8)
    assert torch.isfinite(result.data).all()


@pytest.mark.parametrize(
    "kernel",
    [
        "drizzle_point",
        "drizzle_square",
        "conservative_polygon",
        "splat_point",
        "splat_box",
    ],
)
def test_coadd_image_plane_forward_geometry_kernels_bypass_grid_sample(
    tmp_path: Path,
    monkeypatch,
    kernel: str,
) -> None:
    path_a = tmp_path / f"drizzle_a_{kernel}.fits"
    path_b = tmp_path / f"drizzle_b_{kernel}.fits"
    header_a = _simple_header()
    header_b = dict(header_a)
    header_b["CRVAL1"] = float(header_b["CRVAL1"]) + 0.02
    data = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    torchfits.write(str(path_a), data, header=header_a, overwrite=True)
    torchfits.write(str(path_b), data + 1.0, header=header_b, overwrite=True)

    calls: list[str] = []
    original = resampling_mod.F.grid_sample

    def recorded(*args, **kwargs):
        calls.append(kwargs["mode"])
        return original(*args, **kwargs)

    monkeypatch.setattr(resampling_mod.F, "grid_sample", recorded)

    result = coadd_image_plane(
        [ImageObservation(str(path_a)), ImageObservation(str(path_b))],
        resampling_kernel=kernel,
    )

    assert result.data.shape == (8, 8)
    assert not calls


def test_coadd_image_plane_conservative_polygon_preserves_shifted_support_and_finite_output(
    tmp_path: Path,
) -> None:
    path = tmp_path / "conservative_polygon_identity.fits"
    header = _simple_header()
    target_header = dict(header)
    target_header["CRVAL1"] = float(target_header["CRVAL1"]) + 0.01
    data = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    torchfits.write(str(path), data, header=header, overwrite=True)

    result = coadd_image_plane(
        [ImageObservation(str(path))],
        target_header=target_header,
        resampling_kernel="conservative_polygon",
        value_semantics="surface_brightness",
    )

    assert torch.isfinite(result.weight).all()
    assert torch.isfinite(result.data).all()
    support = result.weight > 0.0
    assert bool(support.any())


def test_coadd_image_plane_exact_drizzle_square_preserves_flux_under_shift(
    tmp_path: Path,
) -> None:
    path = tmp_path / "drizzle_identity.fits"
    header = _simple_header()
    target_header = dict(header)
    target_header["CRVAL1"] = float(target_header["CRVAL1"]) + 0.01
    data = torch.ones((8, 8), dtype=torch.float32)
    torchfits.write(str(path), data, header=header, overwrite=True)

    result = coadd_image_plane(
        [ImageObservation(str(path))],
        target_header=target_header,
        resampling_kernel="drizzle_square",
        value_semantics="surface_brightness",
        drizzle_pixfrac=1.0,
    )

    assert torch.isfinite(result.weight).all()
    assert torch.isfinite(result.data).all()
    support = result.weight > 0.0
    assert bool(support.any())
    torch.testing.assert_close(
        result.data[support],
        torch.ones_like(result.data[support]),
        atol=1e-2,
        rtol=1e-2,
    )


def test_coadd_fft_shift_matches_bilinear_for_pure_crpix_offset_interior(
    tmp_path: Path,
) -> None:
    """fft_shift uses periodic FFT; interior should track bilinear for CRPIX-only offset."""
    ny, nx = 48, 48
    base = {
        "NAXIS": 2,
        "NAXIS1": nx,
        "NAXIS2": ny,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": nx / 2.0,
        "CRPIX2": ny / 2.0,
        "CRVAL1": 180.0,
        "CRVAL2": 0.0,
        "CDELT1": -0.02,
        "CDELT2": 0.02,
    }
    ha = dict(base)
    ht = dict(base)
    ht["CRPIX1"] = float(ht["CRPIX1"]) + 0.31
    ht["CRPIX2"] = float(ht["CRPIX2"]) - 0.19
    yy, xx = torch.meshgrid(
        torch.arange(ny, dtype=torch.float32),
        torch.arange(nx, dtype=torch.float32),
        indexing="ij",
    )
    blob = torch.exp(-((xx - nx / 2.0) ** 2 + (yy - ny / 2.0) ** 2) / (2.0 * 8.0**2))
    path = tmp_path / "crpix.fits"
    torchfits.write(str(path), blob, header=ha, overwrite=True)
    obs = [ImageObservation(str(path))]
    r_fft = coadd_image_plane(obs, target_header=ht, resampling_kernel="fft_shift")
    r_bi = coadd_image_plane(obs, target_header=ht, resampling_kernel="bilinear")
    inner_y = slice(10, ny - 10)
    inner_x = slice(10, nx - 10)
    mask = (r_fft.weight[inner_y, inner_x] > 0.0) & (
        r_bi.weight[inner_y, inner_x] > 0.0
    )
    assert int(mask.sum().item()) > 50
    torch.testing.assert_close(
        r_fft.data[inner_y, inner_x][mask],
        r_bi.data[inner_y, inner_x][mask],
        atol=0.06,
        rtol=0.06,
    )


def test_coadd_fft_diagonal_affine_tracks_bilinear_scaled_cdelta_interior(
    tmp_path: Path,
) -> None:
    """Diagonal affine FFT path: uniform CDELT scale should track bilinear in the interior."""
    ny, nx = 56, 56
    base = {
        "NAXIS": 2,
        "NAXIS1": nx,
        "NAXIS2": ny,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": nx / 2.0,
        "CRPIX2": ny / 2.0,
        "CRVAL1": 100.0,
        "CRVAL2": 8.0,
        "CDELT1": -0.018,
        "CDELT2": 0.018,
    }
    ha = dict(base)
    ht = dict(base)
    f = 1.05
    ht["CDELT1"] = float(ht["CDELT1"]) * f
    ht["CDELT2"] = float(ht["CDELT2"]) * f
    yy, xx = torch.meshgrid(
        torch.arange(ny, dtype=torch.float32),
        torch.arange(nx, dtype=torch.float32),
        indexing="ij",
    )
    blob = torch.exp(-((xx - nx / 2.0) ** 2 + (yy - ny / 2.0) ** 2) / (2.0 * 9.0**2))
    path = tmp_path / "diag_cdelta.fits"
    torchfits.write(str(path), blob, header=ha, overwrite=True)
    obs = [ImageObservation(str(path))]
    r_fft = coadd_image_plane(
        obs, target_header=ht, resampling_kernel="fft_diagonal_affine"
    )
    r_bi = coadd_image_plane(obs, target_header=ht, resampling_kernel="bilinear")
    inner_y = slice(12, ny - 12)
    inner_x = slice(12, nx - 12)
    mask = (r_fft.weight[inner_y, inner_x] > 0.0) & (
        r_bi.weight[inner_y, inner_x] > 0.0
    )
    assert int(mask.sum().item()) > 80
    torch.testing.assert_close(
        r_fft.data[inner_y, inner_x][mask],
        r_bi.data[inner_y, inner_x][mask],
        atol=0.08,
        rtol=0.08,
    )


def test_coadd_periodic_linear_affine_matches_bilinear_skewed_cd_interior(
    tmp_path: Path,
) -> None:
    """Full linear pixel map (rotated CD) on a torus bilinear path vs ``grid_sample``."""
    ny, nx = 40, 40
    base = {
        "NAXIS": 2,
        "NAXIS1": nx,
        "NAXIS2": ny,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": nx / 2.0,
        "CRPIX2": ny / 2.0,
        "CRVAL1": 50.0,
        "CRVAL2": 1.0,
        "CDELT1": -0.02,
        "CDELT2": 0.02,
    }
    ha = dict(base)
    th_rad = math.radians(12.0)
    sc = 0.02
    ht = {
        "NAXIS": 2,
        "NAXIS1": nx,
        "NAXIS2": ny,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": nx / 2.0,
        "CRPIX2": ny / 2.0,
        "CRVAL1": base["CRVAL1"],
        "CRVAL2": base["CRVAL2"],
        "CD1_1": -sc * math.cos(th_rad),
        "CD1_2": sc * math.sin(th_rad),
        "CD2_1": sc * math.sin(th_rad),
        "CD2_2": sc * math.cos(th_rad),
    }
    yy, xx = torch.meshgrid(
        torch.arange(ny, dtype=torch.float32),
        torch.arange(nx, dtype=torch.float32),
        indexing="ij",
    )
    blob = torch.exp(-((xx - nx / 2.0) ** 2 + (yy - ny / 2.0) ** 2) / (2.0 * 6.0**2))
    path = tmp_path / "skew_pla.fits"
    torchfits.write(str(path), blob, header=ha, overwrite=True)
    obs = [ImageObservation(str(path))]
    r_p = coadd_image_plane(
        obs,
        target_header=ht,
        resampling_kernel="periodic_linear_affine",
    )
    r_b = coadd_image_plane(obs, target_header=ht, resampling_kernel="bilinear")
    inner_y = slice(8, ny - 8)
    inner_x = slice(8, nx - 8)
    mask = (r_p.weight[inner_y, inner_x] > 0.0) & (r_b.weight[inner_y, inner_x] > 0.0)
    torch.testing.assert_close(
        r_p.data[inner_y, inner_x][mask],
        r_b.data[inner_y, inner_x][mask],
        atol=1e-4,
        rtol=1e-4,
    )


def test_coadd_fft_linear_affine_matches_periodic_linear_affine_scaled_cdelta_interior(
    tmp_path: Path,
) -> None:
    """LDU FFT path reduces to periodic bilinear + translation when CD is diagonal-scaled."""
    ny, nx = 56, 56
    base = {
        "NAXIS": 2,
        "NAXIS1": nx,
        "NAXIS2": ny,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": nx / 2.0,
        "CRPIX2": ny / 2.0,
        "CRVAL1": 100.0,
        "CRVAL2": 8.0,
        "CDELT1": -0.018,
        "CDELT2": 0.018,
    }
    ha = dict(base)
    ht = dict(base)
    f = 1.05
    ht["CDELT1"] = float(ht["CDELT1"]) * f
    ht["CDELT2"] = float(ht["CDELT2"]) * f
    yy, xx = torch.meshgrid(
        torch.arange(ny, dtype=torch.float32),
        torch.arange(nx, dtype=torch.float32),
        indexing="ij",
    )
    blob = torch.exp(-((xx - nx / 2.0) ** 2 + (yy - ny / 2.0) ** 2) / (2.0 * 9.0**2))
    path = tmp_path / "diag_ldu.fits"
    torchfits.write(str(path), blob, header=ha, overwrite=True)
    obs = [ImageObservation(str(path))]
    r_pla = coadd_image_plane(
        obs, target_header=ht, resampling_kernel="periodic_linear_affine"
    )
    r_fl = coadd_image_plane(
        obs, target_header=ht, resampling_kernel="fft_linear_affine"
    )
    inner_y = slice(12, ny - 12)
    inner_x = slice(12, nx - 12)
    mask = (r_pla.weight[inner_y, inner_x] > 0.0) & (
        r_fl.weight[inner_y, inner_x] > 0.0
    )
    assert int(mask.sum().item()) > 80
    torch.testing.assert_close(
        r_fl.data[inner_y, inner_x][mask],
        r_pla.data[inner_y, inner_x][mask],
        atol=0.004,
        rtol=0.02,
    )


def test_coadd_spectral_linear_affine_tracks_bilinear_skewed_cd_interior(
    tmp_path: Path,
) -> None:
    """Global spectral affine remapper should track bilinear on linear rotated grids."""
    ny, nx = 40, 40
    base = {
        "NAXIS": 2,
        "NAXIS1": nx,
        "NAXIS2": ny,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": nx / 2.0,
        "CRPIX2": ny / 2.0,
        "CRVAL1": 50.0,
        "CRVAL2": 1.0,
        "CDELT1": -0.02,
        "CDELT2": 0.02,
    }
    ha = dict(base)
    th_rad = math.radians(12.0)
    sc = 0.02
    ht = {
        "NAXIS": 2,
        "NAXIS1": nx,
        "NAXIS2": ny,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": nx / 2.0,
        "CRPIX2": ny / 2.0,
        "CRVAL1": base["CRVAL1"],
        "CRVAL2": base["CRVAL2"],
        "CD1_1": -sc * math.cos(th_rad),
        "CD1_2": sc * math.sin(th_rad),
        "CD2_1": sc * math.sin(th_rad),
        "CD2_2": sc * math.cos(th_rad),
    }
    yy, xx = torch.meshgrid(
        torch.arange(ny, dtype=torch.float32),
        torch.arange(nx, dtype=torch.float32),
        indexing="ij",
    )
    blob = torch.exp(-((xx - nx / 2.0) ** 2 + (yy - ny / 2.0) ** 2) / (2.0 * 6.0**2))
    path = tmp_path / "skew_spectral.fits"
    torchfits.write(str(path), blob, header=ha, overwrite=True)
    obs = [ImageObservation(str(path))]
    r_spec = coadd_image_plane(
        obs,
        target_header=ht,
        resampling_kernel="spectral_linear_affine",
    )
    r_bi = coadd_image_plane(obs, target_header=ht, resampling_kernel="bilinear")
    inner_y = slice(8, ny - 8)
    inner_x = slice(8, nx - 8)
    mask = (r_spec.weight[inner_y, inner_x] > 0.0) & (
        r_bi.weight[inner_y, inner_x] > 0.0
    )
    assert int(mask.sum().item()) > 100
    torch.testing.assert_close(
        r_spec.data[inner_y, inner_x][mask],
        r_bi.data[inner_y, inner_x][mask],
        atol=0.05,
        rtol=0.05,
    )


def test_coadd_spectral_linear_affine_zero_boundary_does_not_wrap_corner_source(
    tmp_path: Path,
) -> None:
    ny, nx = 32, 32
    base = {
        "NAXIS": 2,
        "NAXIS1": nx,
        "NAXIS2": ny,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": nx / 2.0,
        "CRPIX2": ny / 2.0,
        "CRVAL1": 180.0,
        "CRVAL2": 0.0,
        "CDELT1": -0.02,
        "CDELT2": 0.02,
    }
    ha = dict(base)
    ht = dict(base)
    ht["CRPIX1"] = float(ht["CRPIX1"]) + 2.0
    image = torch.zeros((ny, nx), dtype=torch.float32)
    image[2, 2] = 1.0
    path = tmp_path / "spectral_edge.fits"
    torchfits.write(str(path), image, header=ha, overwrite=True)
    obs = [ImageObservation(str(path))]
    r_spec = coadd_image_plane(
        obs,
        target_header=ht,
        resampling_kernel="spectral_linear_affine",
    )
    r_bi = coadd_image_plane(obs, target_header=ht, resampling_kernel="bilinear")
    assert float(r_spec.data[:, -1].abs().max()) < 1e-5
    torch.testing.assert_close(r_spec.data, r_bi.data, atol=0.03, rtol=0.03)


def test_coadd_spectral_nonuniform_tracks_bilinear_on_subpixel_crval_shift(
    tmp_path: Path,
) -> None:
    size = 36
    ha = {
        "NAXIS": 2,
        "NAXIS1": size,
        "NAXIS2": size,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": size / 2.0,
        "CRPIX2": size / 2.0,
        "CRVAL1": 180.0,
        "CRVAL2": 2.0,
        "CDELT1": -0.015,
        "CDELT2": 0.015,
    }
    ht = dict(ha)
    ht["CRVAL1"] = float(ht["CRVAL1"]) + 0.25 * abs(float(ht["CDELT1"]))
    ht["CRVAL2"] = float(ht["CRVAL2"]) - 0.2 * abs(float(ht["CDELT2"]))
    yy, xx = torch.meshgrid(
        torch.arange(size, dtype=torch.float32),
        torch.arange(size, dtype=torch.float32),
        indexing="ij",
    )
    image = (
        torch.exp(-((xx - 18.0) ** 2 + (yy - 16.0) ** 2) / (2.0 * 4.5**2))
        + 0.15 * torch.sin(0.4 * xx)
        + 0.1 * torch.cos(0.35 * yy)
    ).to(dtype=torch.float32)
    path = tmp_path / "tpv_shift_spectral.fits"
    torchfits.write(str(path), image, header=ha, overwrite=True)
    obs = [ImageObservation(str(path))]
    r_spec = coadd_image_plane(
        obs,
        target_header=ht,
        resampling_kernel="spectral_nonuniform",
    )
    r_bi = coadd_image_plane(obs, target_header=ht, resampling_kernel="bilinear")
    inner = (slice(8, size - 8), slice(8, size - 8))
    mask = (r_spec.weight[inner] > 0.0) & (r_bi.weight[inner] > 0.0)
    assert int(mask.sum().item()) > 100
    torch.testing.assert_close(
        r_spec.data[inner][mask],
        r_bi.data[inner][mask],
        atol=0.05,
        rtol=0.05,
    )


def test_coadd_spectral_nonuniform_fast_tracks_direct_nonuniform_on_subpixel_shift(
    tmp_path: Path,
) -> None:
    size = 36
    ha = {
        "NAXIS": 2,
        "NAXIS1": size,
        "NAXIS2": size,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": size / 2.0,
        "CRPIX2": size / 2.0,
        "CRVAL1": 180.0,
        "CRVAL2": 2.0,
        "CDELT1": -0.015,
        "CDELT2": 0.015,
    }
    ht = dict(ha)
    ht["CRVAL1"] = float(ht["CRVAL1"]) + 0.25 * abs(float(ht["CDELT1"]))
    ht["CRVAL2"] = float(ht["CRVAL2"]) - 0.2 * abs(float(ht["CDELT2"]))
    yy, xx = torch.meshgrid(
        torch.arange(size, dtype=torch.float32),
        torch.arange(size, dtype=torch.float32),
        indexing="ij",
    )
    image = (
        torch.exp(-((xx - 18.0) ** 2 + (yy - 16.0) ** 2) / (2.0 * 4.5**2))
        + 0.15 * torch.sin(0.4 * xx)
        + 0.1 * torch.cos(0.35 * yy)
    ).to(dtype=torch.float32)
    path = tmp_path / "tan_fast_spectral.fits"
    torchfits.write(str(path), image, header=ha, overwrite=True)
    obs = [ImageObservation(str(path))]
    r_fast = coadd_image_plane(
        obs,
        target_header=ht,
        resampling_kernel="spectral_nonuniform_fast",
    )
    r_direct = coadd_image_plane(
        obs,
        target_header=ht,
        resampling_kernel="spectral_nonuniform",
    )
    inner = (slice(8, size - 8), slice(8, size - 8))
    mask = (r_fast.weight[inner] > 0.0) & (r_direct.weight[inner] > 0.0)
    assert int(mask.sum().item()) > 100
    torch.testing.assert_close(
        r_fast.data[inner][mask],
        r_direct.data[inner][mask],
        atol=0.04,
        rtol=0.04,
    )


def test_coadd_fft_diagonal_affine_rejects_skewed_cd_matrix(tmp_path: Path) -> None:
    """Off-diagonal CD terms induce iy/jx cross-terms; diagonal FFT path must reject."""
    ny, nx = 24, 24
    base = {
        "NAXIS": 2,
        "NAXIS1": nx,
        "NAXIS2": ny,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": nx / 2.0,
        "CRPIX2": ny / 2.0,
        "CRVAL1": 50.0,
        "CRVAL2": 1.0,
        "CDELT1": -0.02,
        "CDELT2": 0.02,
    }
    ha = dict(base)
    th = math.radians(17.0)
    sc = 0.02
    ht = {
        "NAXIS": 2,
        "NAXIS1": nx,
        "NAXIS2": ny,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": nx / 2.0,
        "CRPIX2": ny / 2.0,
        "CRVAL1": base["CRVAL1"],
        "CRVAL2": base["CRVAL2"],
        "CD1_1": -sc * math.cos(th),
        "CD1_2": sc * math.sin(th),
        "CD2_1": sc * math.sin(th),
        "CD2_2": sc * math.cos(th),
    }
    path = tmp_path / "skew.fits"
    torchfits.write(str(path), torch.ones((ny, nx), dtype=torch.float32), header=ha)
    with pytest.raises(ValueError, match="fft_diagonal_affine"):
        coadd_image_plane(
            [ImageObservation(str(path))],
            target_header=ht,
            resampling_kernel="fft_diagonal_affine",
        )


def test_coadd_fft_shift_rejects_crval_only_offset(
    tmp_path: Path,
) -> None:
    ny, nx = 16, 16
    ha = _simple_header()
    ha["NAXIS1"] = nx
    ha["NAXIS2"] = ny
    ha["CRPIX1"] = nx / 2.0
    ha["CRPIX2"] = ny / 2.0
    ht = dict(ha)
    ht["CRVAL1"] = float(ht["CRVAL1"]) + 0.05
    path = tmp_path / "a.fits"
    torchfits.write(str(path), torch.ones((ny, nx), dtype=torch.float32), header=ha)
    with pytest.raises(ValueError, match="fft_shift"):
        coadd_image_plane(
            [ImageObservation(str(path))],
            target_header=ht,
            resampling_kernel="fft_shift",
        )


def test_coadd_bilinear_subpixel_crval_shift_preserves_finite_support(
    tmp_path: Path,
) -> None:
    """SWarp/reproject-class subpixel sky shifts should stay finite on a smooth blob."""
    ny, nx = 40, 40
    base = {
        "NAXIS": 2,
        "NAXIS1": nx,
        "NAXIS2": ny,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": nx / 2.0,
        "CRPIX2": ny / 2.0,
        "CRVAL1": 200.0,
        "CRVAL2": 10.0,
        "CDELT1": -0.015,
        "CDELT2": 0.015,
    }
    ha = dict(base)
    hb = dict(base)
    hb["CRVAL1"] = float(hb["CRVAL1"]) + 0.4 * float(hb["CDELT1"])
    hb["CRVAL2"] = float(hb["CRVAL2"]) + 0.3 * float(hb["CDELT2"])
    yy, xx = torch.meshgrid(
        torch.arange(ny, dtype=torch.float32),
        torch.arange(nx, dtype=torch.float32),
        indexing="ij",
    )
    blob = torch.exp(-((xx - nx / 2.0) ** 2 + (yy - ny / 2.0) ** 2) / (2.0 * 6.0**2))
    pa = tmp_path / "sub_a.fits"
    pb = tmp_path / "sub_b.fits"
    torchfits.write(str(pa), blob, header=ha, overwrite=True)
    torchfits.write(str(pb), blob * 0.95 + 0.02, header=hb, overwrite=True)
    result = coadd_image_plane(
        [ImageObservation(str(pa)), ImageObservation(str(pb))],
        resampling_kernel="bilinear",
        stacking_method="weighted_mean",
        value_semantics="surface_brightness",
    )
    assert torch.isfinite(result.data).all()
    assert torch.isfinite(result.weight).all()
    assert float(result.weight.max().item()) > 0.0


def test_coadd_bilinear_partial_overlap_at_frame_edge_is_finite(
    tmp_path: Path,
) -> None:
    """SWarp-class stacks often have partial footprint support at tile edges."""
    ny, nx = 32, 32
    base = {
        "NAXIS": 2,
        "NAXIS1": nx,
        "NAXIS2": ny,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": nx / 2.0,
        "CRPIX2": ny / 2.0,
        "CRVAL1": 180.0,
        "CRVAL2": 0.0,
        "CDELT1": -0.02,
        "CDELT2": 0.02,
    }
    path_a = tmp_path / "edge_a.fits"
    path_b = tmp_path / "edge_b.fits"
    ha = dict(base)
    hb = dict(base)
    hb["CRVAL1"] = float(hb["CRVAL1"]) + 0.45 * float(hb["CDELT1"])
    data = torch.ones((ny, nx), dtype=torch.float32)
    torchfits.write(str(path_a), data, header=ha, overwrite=True)
    torchfits.write(str(path_b), data * 1.2, header=hb, overwrite=True)

    result = coadd_image_plane(
        [ImageObservation(str(path_a)), ImageObservation(str(path_b))],
        resampling_kernel="bilinear",
        stacking_method="weighted_mean",
        value_semantics="surface_brightness",
    )
    assert torch.isfinite(result.data).all()
    assert torch.isfinite(result.weight).all()
    assert result.overlap_count is not None
    assert int((result.overlap_count > 0).sum().item()) > 0
    assert int((result.overlap_count > 1).sum().item()) > 0


def test_coadd_image_plane_rejects_non_positive_drizzle_pixfrac(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pixfrac_invalid.fits"
    torchfits.write(
        str(path),
        torch.ones((4, 4), dtype=torch.float32),
        header=_simple_header(),
        overwrite=True,
    )

    with pytest.raises(ValueError, match="drizzle_pixfrac"):
        coadd_image_plane(
            [ImageObservation(str(path))],
            resampling_kernel="drizzle_square",
            drizzle_pixfrac=0.0,
        )




def test_coadd_image_plane_flux_semantics_improves_flux_conservation(tmp_path) -> None:
    source_path = tmp_path / "flux_source.fits"
    target_header = _simple_header()
    source_header = dict(target_header)
    source_header["CDELT1"] = -0.02
    source_header["CDELT2"] = 0.02

    source_wcs = torchsky.WCS(source_header)
    target_wcs = torchsky.WCS(target_header)
    source_area = pixel_area_sr_map(
        source_wcs,
        (8, 8),
        device=torch.device("cpu"),
    )
    target_area = pixel_area_sr_map(
        target_wcs,
        (8, 8),
        device=torch.device("cpu"),
    )
    area_norm = float(torch.median(target_area).item())
    brightness = torch.full((8, 8), 2.0, dtype=torch.float32)
    flux_source = brightness * (source_area / area_norm)
    expected_target_flux = brightness * (target_area / area_norm)
    torchfits.write(str(source_path), flux_source, header=source_header, overwrite=True)

    surface_result = coadd_image_plane(
        [ImageObservation(str(source_path))],
        target_header=target_header,
        value_semantics="surface_brightness",
    )
    flux_result = coadd_image_plane(
        [ImageObservation(str(source_path))],
        target_header=target_header,
        value_semantics="flux",
    )

    surface_flux_err = (
        torch.abs(surface_result.data - expected_target_flux).mean().item()
    )
    flux_flux_err = torch.abs(flux_result.data - expected_target_flux).mean().item()
    assert flux_flux_err < surface_flux_err


def test_coadd_image_plane_weighted_mean_scales_surface_brightness_weights_only_for_coarser_targets(
    tmp_path,
) -> None:
    image = torch.ones((8, 8), dtype=torch.float32)
    weight = torch.ones((8, 8), dtype=torch.float32)

    fine_target = _simple_header()
    coarse_target = dict(fine_target)
    coarse_target["CDELT1"] = -0.02
    coarse_target["CDELT2"] = 0.02

    fine_source = dict(fine_target)
    fine_source_path = tmp_path / "fine_source.fits"
    fine_weight_path = tmp_path / "fine_source_wht.fits"
    torchfits.write(str(fine_source_path), image, header=fine_source, overwrite=True)
    torchfits.write(str(fine_weight_path), weight, header=fine_source, overwrite=True)

    coarse_source = dict(coarse_target)
    coarse_source_path = tmp_path / "coarse_source.fits"
    coarse_weight_path = tmp_path / "coarse_source_wht.fits"
    torchfits.write(str(coarse_source_path), image, header=coarse_source, overwrite=True)
    torchfits.write(
        str(coarse_weight_path), weight, header=coarse_source, overwrite=True
    )

    finer_target_result = coadd_image_plane(
        [
            ImageObservation(
                str(coarse_source_path), weight_path=str(coarse_weight_path)
            )
        ],
        target_header=fine_target,
        value_semantics="surface_brightness",
        stacking_method="weighted_mean",
    )
    coarser_target_result = coadd_image_plane(
        [ImageObservation(str(fine_source_path), weight_path=str(fine_weight_path))],
        target_header=coarse_target,
        value_semantics="surface_brightness",
        stacking_method="weighted_mean",
    )

    fine_area = representative_pixel_area_sr(
        torchsky.WCS(fine_target),
        (8, 8),
        device=torch.device("cpu"),
    )
    coarse_area = representative_pixel_area_sr(
        torchsky.WCS(coarse_target),
        (8, 8),
        device=torch.device("cpu"),
    )

    assert float(finer_target_result.data[4, 4].item()) == pytest.approx(
        1.0, abs=1.0e-4
    )
    assert float(finer_target_result.weight[4, 4].item()) == pytest.approx(
        1.0, abs=1.0e-4
    )
    assert float(coarser_target_result.data[4, 4].item()) == pytest.approx(
        1.0, abs=1.0e-4
    )
    assert float(coarser_target_result.weight[4, 4].item()) == pytest.approx(
        coarse_area / fine_area,
        rel=5.0e-2,
        abs=1.0e-4,
    )


def test_coadd_image_plane_builds_target_grid_once_per_run(
    tmp_path, monkeypatch
) -> None:
    path_a = tmp_path / "a.fits"
    path_b = tmp_path / "b.fits"
    header_a = _simple_header()
    header_b = dict(header_a)
    header_b["CRVAL1"] = header_b["CRVAL1"] + 0.02
    data = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    torchfits.write(str(path_a), data, header=header_a, overwrite=True)
    torchfits.write(str(path_b), data + 1.0, header=header_b, overwrite=True)

    calls = 0
    original = image_plane_mod._prepare_target

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(image_plane_coadd_mod, "_prepare_target", counted)

    result = coadd_image_plane(
        [ImageObservation(str(path_a)), ImageObservation(str(path_b))]
    )

    assert calls == 1
    assert result.data.shape == (8, 8)


def test_project_observation_large_tan_windowing_matches_full_reprojection(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "source.fits"
    source_header = _simple_header()
    source_header["NAXIS1"] = 64
    source_header["NAXIS2"] = 64
    source_header["CRPIX1"] = 32.0
    source_header["CRPIX2"] = 32.0
    data = torch.arange(64 * 64, dtype=torch.float32).reshape(64, 64)
    torchfits.write(str(path), data, header=source_header, overwrite=True)

    target_header = dict(source_header)
    target_header["NAXIS1"] = 128
    target_header["NAXIS2"] = 96
    target_header["CRPIX1"] = 64.0
    target_header["CRPIX2"] = 48.0
    target_shape = (96, 128)
    device = data.device
    prepared_target = image_plane_mod._prepare_target(
        header=target_header,
        target_wcs=image_plane_mod.WCS(target_header),
        target_shape=target_shape,
        device=device,
        include_pixel_area=False,
        include_representative_pixel_area=True,
        include_quadrature=False,
    )
    obs = ImageObservation(str(path))
    image, header = torchfits.read(str(path), return_header=True)
    image = validate_image_plane_2d(image)
    source_wcs = image_plane_mod.WCS(dict(header))

    original_threshold = windowing_mod._LARGE_TAN_WINDOWING_MIN_TARGET_PIXELS
    original_resample = image_plane_mod._resample_weighted_to_target
    try:
        forced_shapes: list[tuple[int, int]] = []

        def wrapped_resample(*args, **kwargs):
            target = args[3]
            forced_shapes.append(tuple(int(v) for v in target.shape))
            return original_resample(*args, **kwargs)

        monkeypatch.setattr(
            image_plane_projection_mod, "_resample_weighted_to_target", wrapped_resample
        )
        monkeypatch.setattr(windowing_mod, "_LARGE_TAN_WINDOWING_MIN_TARGET_PIXELS", 0)
        forced_numer, forced_weight, _, _ = (
            image_plane_mod._project_observation_to_target(
                obs,
                image=image,
                header_dict_obs=dict(header),
                source_wcs=source_wcs,
                prepared_target=prepared_target,
                target_header=target_header,
                target_shape=target_shape,
                device=device,
                footprint_order=None,
                resampling_kernel="bilinear",
                value_semantics="surface_brightness",
            )
        )
        assert forced_shapes
        assert (
            forced_shapes[0][0] < target_shape[0]
            or forced_shapes[0][1] < target_shape[1]
        )

        monkeypatch.setattr(
            windowing_mod,
            "_LARGE_TAN_WINDOWING_MIN_TARGET_PIXELS",
            10**9,
        )
        full_numer, full_weight, _, _ = image_plane_mod._project_observation_to_target(
            obs,
            image=image,
            header_dict_obs=dict(header),
            source_wcs=source_wcs,
            prepared_target=prepared_target,
            target_header=target_header,
            target_shape=target_shape,
            device=device,
            footprint_order=None,
            resampling_kernel="bilinear",
            value_semantics="surface_brightness",
        )
    finally:
        monkeypatch.setattr(
            windowing_mod,
            "_LARGE_TAN_WINDOWING_MIN_TARGET_PIXELS",
            original_threshold,
        )
        monkeypatch.setattr(
            image_plane_mod, "_resample_weighted_to_target", original_resample
        )

    torch.testing.assert_close(forced_numer, full_numer, atol=0.0, rtol=0.0)
    torch.testing.assert_close(forced_weight, full_weight, atol=0.0, rtol=0.0)
    assert forced_weight.sum().item() > 0.0


def test_project_observation_window_no_overlap_skips_geometry(monkeypatch) -> None:
    source_header = _simple_header()
    image = torch.ones((8, 8), dtype=torch.float32)
    target_header = dict(source_header)
    target_header["CRVAL1"] = target_header["CRVAL1"] + 10.0
    target_shape = (8, 8)
    prepared_target = image_plane_mod._prepare_target(
        header=target_header,
        target_wcs=image_plane_mod.WCS(target_header),
        target_shape=target_shape,
        device=torch.device("cpu"),
        include_pixel_area=False,
        include_representative_pixel_area=True,
        include_quadrature=False,
    )

    def no_overlap_window(**kwargs):
        return None, (0, 0, 0, 0)

    def fail_projection(*args, **kwargs):
        raise AssertionError("no-overlap window must not call geometry projection")

    monkeypatch.setattr(image_plane_projection_mod, "_window_large_tan_target", no_overlap_window)
    monkeypatch.setattr(
        image_plane_projection_mod, "project_image_observation_geometry", fail_projection
    )

    projected = image_plane_mod._project_observation_to_target_chunk(
        ImageObservation("mock.fits"),
        image=image,
        header_dict_obs=source_header,
        source_wcs=image_plane_mod.WCS(source_header),
        prepared_target=prepared_target,
        target_header=target_header,
        target_shape=target_shape,
        device=torch.device("cpu"),
        footprint_order=None,
        resampling_kernel="bilinear",
        value_semantics="surface_brightness",
        drizzle_pixfrac=1.0,
    )

    assert projected.target_window == (0, 0, 0, 0)
    assert projected.numer.shape == (0, 0)
    assert projected.weight.shape == (0, 0)
    assert projected.overlap.shape == (0, 0)


def test_surface_brightness_area_scaling_gate_uses_target_vs_harmonic_source_scale(
    tmp_path: Path,
) -> None:
    data = torch.ones((8, 8), dtype=torch.float32)
    path_a = tmp_path / "a.fits"
    path_b = tmp_path / "b.fits"
    header_a = _simple_header()
    header_b = _simple_header()
    header_a["CDELT1"] = -0.006
    header_a["CDELT2"] = 0.006
    header_b["CDELT1"] = -0.014
    header_b["CDELT2"] = 0.014
    torchfits.write(str(path_a), data, header=header_a, overwrite=True)
    torchfits.write(str(path_b), data, header=header_b, overwrite=True)
    observations = [ImageObservation(str(path_a)), ImageObservation(str(path_b))]

    coarse_target = _simple_header()
    coarse_target["CDELT1"] = -0.010
    coarse_target["CDELT2"] = 0.010
    fine_target = _simple_header()
    fine_target["CDELT1"] = -0.004
    fine_target["CDELT2"] = 0.004

    assert image_plane_mod._should_apply_surface_brightness_area_scaling(
        observations,
        target_header=coarse_target,
        target_shape=(8, 8),
    )
    assert not image_plane_mod._should_apply_surface_brightness_area_scaling(
        observations,
        target_header=fine_target,
        target_shape=(8, 8),
    )


def test_coadd_image_plane_skips_target_grid_for_identical_wcs_without_footprints(
    tmp_path, monkeypatch
) -> None:
    path_a = tmp_path / "a.fits"
    path_b = tmp_path / "b.fits"
    header = _simple_header()
    data = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    torchfits.write(str(path_a), data, header=header, overwrite=True)
    torchfits.write(str(path_b), data + 1.0, header=header, overwrite=True)

    calls = 0
    original = image_plane_mod._build_target_world_grid

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(image_plane_coadd_mod, "_build_target_world_grid", counted)

    result = coadd_image_plane(
        [ImageObservation(str(path_a)), ImageObservation(str(path_b))]
    )

    assert calls == 0
    assert result.data.shape == (8, 8)


def test_coadd_image_plane_supports_weight_and_mask_maps(tmp_path) -> None:
    path_a = tmp_path / "a.fits"
    path_b = tmp_path / "b.fits"
    wht_a = tmp_path / "a_wht.fits"
    msk_b = tmp_path / "b_mask.fits"
    header = _simple_header()

    data_a = torch.full((8, 8), 2.0, dtype=torch.float32)
    data_b = torch.full((8, 8), 10.0, dtype=torch.float32)
    weight_a = torch.ones((8, 8), dtype=torch.float32)
    weight_a[:, :4] = 3.0
    mask_b = torch.ones((8, 8), dtype=torch.float32)
    mask_b[:, 4:] = 0.0

    torchfits.write(str(path_a), data_a, header=header, overwrite=True)
    torchfits.write(str(path_b), data_b, header=header, overwrite=True)
    torchfits.write(str(wht_a), weight_a, header=header, overwrite=True)
    torchfits.write(str(msk_b), mask_b, header=header, overwrite=True)

    result = coadd_image_plane(
        [
            ImageObservation(str(path_a), weight_path=str(wht_a)),
            ImageObservation(str(path_b), mask_path=str(msk_b)),
        ],
        footprint_order=6,
    )

    expected_left = (2.0 * 3.0 + 10.0 * 1.0) / 4.0
    expected_right = 2.0
    torch.testing.assert_close(
        result.data[:, :4],
        torch.full((8, 4), expected_left, dtype=torch.float32),
        atol=1e-5,
        rtol=0.0,
    )
    torch.testing.assert_close(
        result.data[:, 4:],
        torch.full((8, 4), expected_right, dtype=torch.float32),
        atol=1e-5,
        rtol=0.0,
    )
    torch.testing.assert_close(
        result.weight[:, :4],
        torch.full((8, 4), 4.0, dtype=torch.float32),
        atol=1e-5,
        rtol=0.0,
    )
    torch.testing.assert_close(
        result.weight[:, 4:],
        torch.full((8, 4), 1.0, dtype=torch.float32),
        atol=1e-5,
        rtol=0.0,
    )
    assert result.overlap_count is not None
    torch.testing.assert_close(
        result.overlap_count[:, :4],
        torch.full((8, 4), 2, dtype=torch.int32),
    )
    torch.testing.assert_close(
        result.overlap_count[:, 4:],
        torch.full((8, 4), 1, dtype=torch.int32),
    )
    assert result.coverage_moc is not None
    assert result.observation_mocs is not None
    assert len(result.observation_mocs) == 2
    assert result.coverage_moc.area_sq_deg > 0.0


def test_coadd_image_plane_rejects_weight_map_shape_mismatch(tmp_path) -> None:
    path = tmp_path / "image.fits"
    weight_path = tmp_path / "weight.fits"
    header = _simple_header()
    torchfits.write(
        str(path),
        torch.ones((8, 8), dtype=torch.float32),
        header=header,
        overwrite=True,
    )
    torchfits.write(
        str(weight_path),
        torch.ones((7, 8), dtype=torch.float32),
        header={**header, "NAXIS2": 7},
        overwrite=True,
    )

    with pytest.raises(ValueError, match="Weight map shape must match"):
        coadd_image_plane(
            [ImageObservation(str(path), weight_path=str(weight_path))],
        )


def test_coadd_image_plane_real_astropy_sip_fixture_with_complementary_masks(
    tmp_path,
) -> None:
    real_path = _astropy_sip_fixture()
    image, header = torchfits.read(str(real_path), return_header=True)
    image = image.to(dtype=torch.float32)

    mask_a = tmp_path / "mask_a.fits"
    mask_b = tmp_path / "mask_b.fits"

    mask_left = torch.zeros_like(image)
    mask_left[:, : image.shape[1] // 2] = 1.0
    mask_right = torch.zeros_like(image)
    mask_right[:, image.shape[1] // 2 :] = 1.0

    torchfits.write(str(mask_a), mask_left, overwrite=True)
    torchfits.write(str(mask_b), mask_right, overwrite=True)

    result = coadd_image_plane(
        [
            ImageObservation(str(real_path), mask_path=str(mask_a)),
            ImageObservation(str(real_path), mask_path=str(mask_b)),
        ],
        footprint_order=6,
    )

    torch.testing.assert_close(result.data, image, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(
        result.weight,
        torch.ones_like(result.weight),
        atol=1e-6,
        rtol=0.0,
    )
    assert result.overlap_count is not None
    torch.testing.assert_close(
        result.overlap_count,
        torch.ones_like(result.overlap_count),
    )
    assert result.coverage_moc is not None
    assert result.observation_mocs is not None
    assert len(result.observation_mocs) == 2
    assert result.coverage_moc.area_sq_deg > 0.0
    assert len(result.coverage_moc.boundary()) >= 1


def test_distorted_flux_quadrature_propagates_tiled_variance() -> None:
    real_path = _astropy_sip_fixture()
    _image, header = torchfits.read(str(real_path), return_header=True)
    target_header = dict(header)
    target_header["CRPIX1"] = float(target_header["CRPIX1"]) + 0.2
    target_header["CRPIX2"] = float(target_header["CRPIX2"]) - 0.15
    observations = [
        ImageObservation(
            str(real_path),
            weight=1.0,
            weight_is_inverse_variance=True,
        )
    ]
    kwargs = {
        "target_header": target_header,
        "resampling_kernel": "bilinear",
        "value_semantics": "flux",
    }

    full = coadd_image_plane(observations, **kwargs)
    tiled = coadd_image_plane(observations, spatial_tile_hw=(11, 13), **kwargs)

    assert full.inverse_variance is not None
    assert tiled.inverse_variance is not None
    assert bool((full.inverse_variance[full.weight > 0.0] > 0.0).all())
    torch.testing.assert_close(tiled.data, full.data, atol=2.0e-4, rtol=2.0e-4)
    torch.testing.assert_close(tiled.weight, full.weight, atol=2.0e-4, rtol=2.0e-4)
    torch.testing.assert_close(
        tiled.inverse_variance,
        full.inverse_variance,
        atol=2.0e-4,
        rtol=2.0e-4,
    )


def test_realistic_small_pixel_flux_identity_preserves_unit_weight_and_ivar() -> None:
    real_path = _astropy_sip_fixture()
    result = coadd_image_plane(
        [
            ImageObservation(
                str(real_path),
                weight=1.0,
                weight_is_inverse_variance=True,
            )
        ],
        value_semantics="flux",
    )

    assert result.inverse_variance is not None
    torch.testing.assert_close(result.weight, torch.ones_like(result.weight))
    torch.testing.assert_close(
        result.inverse_variance,
        torch.ones_like(result.inverse_variance),
    )


def test_coadd_image_plane_supports_mef_science_extensions(tmp_path) -> None:
    mef_path = _write_real_sip_mef(tmp_path, shifted=False)
    image = torchfits.read(str(_astropy_sip_fixture())).to(dtype=torch.float32)

    result = coadd_image_plane(
        [
            ImageObservation(
                str(mef_path),
                hdu=1,
                weight_path=str(mef_path),
                weight_hdu=2,
                mask_path=str(mef_path),
                mask_hdu=3,
            ),
            ImageObservation(
                str(mef_path),
                hdu=4,
                weight_path=str(mef_path),
                weight_hdu=5,
                mask_path=str(mef_path),
                mask_hdu=6,
            ),
        ],
        footprint_order=6,
    )

    assert tuple(result.data.shape) == tuple(image.shape)
    assert result.overlap_count is not None
    assert float(result.weight.max().item()) >= 3.0
    assert result.coverage_moc is not None
    assert result.observation_mocs is not None
    assert len(result.observation_mocs) == 2
    torch.testing.assert_close(
        result.data[:, image.shape[1] // 3 : (2 * image.shape[1]) // 3],
        image[:, image.shape[1] // 3 : (2 * image.shape[1]) // 3],
        atol=1e-4,
        rtol=1e-4,
    )


def test_coadd_image_plane_supports_shifted_mef_science_extensions(tmp_path) -> None:
    mef_path = _write_real_sip_mef(tmp_path, shifted=True)

    result = coadd_image_plane(
        [
            ImageObservation(
                str(mef_path),
                hdu=1,
                weight_path=str(mef_path),
                weight_hdu=2,
                mask_path=str(mef_path),
                mask_hdu=3,
            ),
            ImageObservation(
                str(mef_path),
                hdu=4,
                weight_path=str(mef_path),
                weight_hdu=5,
                mask_path=str(mef_path),
                mask_hdu=6,
            ),
        ],
        footprint_order=6,
    )

    assert tuple(result.data.shape) == (50, 100)
    assert result.overlap_count is not None
    assert int(result.overlap_count.max().item()) >= 1
    assert result.coverage_moc is not None
    assert result.coverage_moc.area_sq_deg > 0.0


def test_coadd_image_plane_supports_tpv_mef_low_overlap(tmp_path) -> None:
    mef_path = _write_synthetic_tpv_mef(tmp_path, size=64)

    result = coadd_image_plane(
        [
            ImageObservation(
                str(mef_path),
                hdu=1,
                weight_path=str(mef_path),
                weight_hdu=2,
                mask_path=str(mef_path),
                mask_hdu=3,
            ),
            ImageObservation(
                str(mef_path),
                hdu=4,
                weight_path=str(mef_path),
                weight_hdu=5,
                mask_path=str(mef_path),
                mask_hdu=6,
            ),
        ],
        footprint_order=6,
    )

    assert tuple(result.data.shape) == (64, 64)
    assert result.overlap_count is not None
    assert int(result.overlap_count.max().item()) >= 1
    assert float((result.weight > 0.0).to(torch.float32).mean().item()) < 1.0
    assert result.coverage_moc is not None
    assert result.observation_mocs is not None
    assert len(result.observation_mocs) == 2


def test_tpv_source_cap_radius_covers_in_bounds_points(tmp_path) -> None:
    mef_path = _write_synthetic_tpv_mef(tmp_path, size=48)
    image, header = torchfits.read(str(mef_path), hdu=4, return_header=True)
    image = image.to(dtype=torch.float32)
    source_wcs = torchsky.WCS(dict(header))
    target_header = _simple_header()
    target_header["NAXIS1"] = 48
    target_header["NAXIS2"] = 48
    target_header["CRPIX1"] = 24.0
    target_header["CRPIX2"] = 24.0
    target_header["CRVAL1"] = float(header["CRVAL1"])
    target_header["CRVAL2"] = float(header["CRVAL2"])
    prepared = image_plane_mod._prepare_target(
        header=target_header,
        target_wcs=torchsky.WCS(target_header),
        target_shape=(48, 48),
        device=image.device,
        include_pixel_area=False,
        include_quadrature=False,
    )
    source_cap = image_plane_mod._tpv_source_cap_radius_deg(
        source_wcs,
        tuple(int(v) for v in image.shape),
    )
    center_sep = image_plane_mod._angular_separation_deg(
        float(source_wcs.alpha0),
        float(source_wcs.delta0),
        prepared.center_ra_deg,
        prepared.center_dec_deg,
    )
    assert center_sep <= source_cap + prepared.cap_radius_deg


def test_coadd_image_plane_multiband_stacks_sorted_wavelengths(tmp_path) -> None:
    header = _simple_header()
    path_g1 = tmp_path / "g1.fits"
    path_g2 = tmp_path / "g2.fits"
    path_r1 = tmp_path / "r1.fits"
    path_r2 = tmp_path / "r2.fits"

    data_g1 = torch.full((8, 8), 1.0, dtype=torch.float32)
    data_g2 = torch.full((8, 8), 3.0, dtype=torch.float32)
    data_r1 = torch.full((8, 8), 10.0, dtype=torch.float32)
    data_r2 = torch.full((8, 8), 14.0, dtype=torch.float32)

    torchfits.write(str(path_g1), data_g1, header=header, overwrite=True)
    torchfits.write(str(path_g2), data_g2, header=header, overwrite=True)
    torchfits.write(str(path_r1), data_r1, header=header, overwrite=True)
    torchfits.write(str(path_r2), data_r2, header=header, overwrite=True)

    result = coadd_image_plane_multiband(
        [
            ImageObservation(str(path_r1), wavelength=6500.0),
            ImageObservation(str(path_g1), wavelength=4800.0),
            ImageObservation(str(path_r2), wavelength=6500.0),
            ImageObservation(str(path_g2), wavelength=4800.0),
        ],
        footprint_order=6,
    )

    torch.testing.assert_close(
        result.wavelength,
        torch.tensor([4800.0, 6500.0], dtype=torch.float64),
    )
    assert result.spectral_axis.unit == "Angstrom"
    assert result.spectral_axis.type == "WAVE"
    torch.testing.assert_close(
        result.data[0],
        torch.full((8, 8), 2.0, dtype=torch.float32),
        atol=1e-5,
        rtol=0.0,
    )
    torch.testing.assert_close(
        result.data[1],
        torch.full((8, 8), 12.0, dtype=torch.float32),
        atol=1e-5,
        rtol=0.0,
    )
    assert tuple(result.data.shape) == (2, 8, 8)
    assert tuple(result.weight.shape) == (2, 8, 8)
    assert result.overlap_count is not None
    torch.testing.assert_close(
        result.overlap_count,
        torch.full((2, 8, 8), 2, dtype=torch.int32),
    )
    assert result.coverage_mocs is not None
    assert len(result.coverage_mocs) == 2
    assert all(
        moc is not None and moc.area_sq_deg > 0.0 for moc in result.coverage_mocs
    )
    assert result.band_observation_counts is not None
    torch.testing.assert_close(
        result.band_observation_counts,
        torch.tensor([2, 2], dtype=torch.int64),
    )


def test_coadd_image_plane_multiband_requires_wavelength(tmp_path) -> None:
    header = _simple_header()
    path = tmp_path / "img.fits"
    torchfits.write(
        str(path),
        torch.ones((8, 8), dtype=torch.float32),
        header=header,
        overwrite=True,
    )

    with pytest.raises(ValueError, match="must define wavelength"):
        coadd_image_plane_multiband([ImageObservation(str(path))])


def test_coadd_image_plane_multiband_real_astropy_fixture(tmp_path) -> None:
    real_path = _astropy_sip_fixture()
    image = torchfits.read(str(real_path)).to(dtype=torch.float32)

    mask_a = tmp_path / "mask_a.fits"
    mask_b = tmp_path / "mask_b.fits"
    mask_left = torch.zeros_like(image)
    mask_left[:, : image.shape[1] // 2] = 1.0
    mask_right = torch.zeros_like(image)
    mask_right[:, image.shape[1] // 2 :] = 1.0
    torchfits.write(str(mask_a), mask_left, overwrite=True)
    torchfits.write(str(mask_b), mask_right, overwrite=True)

    result = coadd_image_plane_multiband(
        [
            ImageObservation(str(real_path), wavelength=4800.0, mask_path=str(mask_a)),
            ImageObservation(str(real_path), wavelength=6500.0, mask_path=str(mask_b)),
        ],
        footprint_order=6,
    )

    assert tuple(result.data.shape) == (2, 50, 100)
    torch.testing.assert_close(
        result.wavelength,
        torch.tensor([4800.0, 6500.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        result.data[0, :, :50], image[:, :50], atol=1e-4, rtol=1e-4
    )
    torch.testing.assert_close(result.data[0, :, 50:], torch.zeros_like(image[:, 50:]))
    torch.testing.assert_close(result.data[1, :, :50], torch.zeros_like(image[:, :50]))
    torch.testing.assert_close(
        result.data[1, :, 50:], image[:, 50:], atol=1e-4, rtol=1e-4
    )
    assert result.overlap_count is not None
    torch.testing.assert_close(
        result.overlap_count[0, :, :50],
        torch.ones_like(result.overlap_count[0, :, :50]),
    )
    torch.testing.assert_close(
        result.overlap_count[0, :, 50:],
        torch.zeros_like(result.overlap_count[0, :, 50:]),
    )
    torch.testing.assert_close(
        result.overlap_count[1, :, :50],
        torch.zeros_like(result.overlap_count[1, :, :50]),
    )
    torch.testing.assert_close(
        result.overlap_count[1, :, 50:],
        torch.ones_like(result.overlap_count[1, :, 50:]),
    )
    torch.testing.assert_close(result.weight[0, :, :50], torch.ones_like(image[:, :50]))
    torch.testing.assert_close(
        result.weight[0, :, 50:], torch.zeros_like(image[:, 50:])
    )
    torch.testing.assert_close(
        result.weight[1, :, :50], torch.zeros_like(image[:, :50])
    )
    torch.testing.assert_close(result.weight[1, :, 50:], torch.ones_like(image[:, 50:]))
    assert result.coverage_mocs is not None
    assert len(result.coverage_mocs) == 2
    assert result.band_observation_counts is not None
    torch.testing.assert_close(
        result.band_observation_counts,
        torch.tensor([1, 1], dtype=torch.int64),
    )


def test_plan_image_plane_target_uses_reference_observation(tmp_path) -> None:
    path_a = tmp_path / "a.fits"
    path_b = tmp_path / "b.fits"
    header_a = _simple_header()
    header_b = dict(header_a)
    header_b["CRVAL1"] = header_b["CRVAL1"] + 0.2
    header_b["NAXIS1"] = 6
    header_b["NAXIS2"] = 5

    torchfits.write(
        str(path_a),
        torch.ones((8, 8), dtype=torch.float32),
        header=header_a,
        overwrite=True,
    )
    torchfits.write(
        str(path_b),
        torch.ones((5, 6), dtype=torch.float32),
        header=header_b,
        overwrite=True,
    )

    planned = plan_image_plane_target(
        [ImageObservation(str(path_a)), ImageObservation(str(path_b))],
        reference_index=1,
    )

    assert planned.shape == (5, 6)
    assert planned.header["NAXIS1"] == 6
    assert planned.header["NAXIS2"] == 5
    assert planned.header["CRVAL1"] == header_b["CRVAL1"]


def test_coadd_image_plane_multiband_accepts_planned_target(tmp_path) -> None:
    path_g = tmp_path / "g.fits"
    path_r = tmp_path / "r.fits"
    header_a = _simple_header()
    header_b = dict(header_a)
    header_b["CRVAL1"] = header_b["CRVAL1"] + 0.03

    torchfits.write(
        str(path_g),
        torch.full((8, 8), 2.0, dtype=torch.float32),
        header=header_a,
        overwrite=True,
    )
    torchfits.write(
        str(path_r),
        torch.full((8, 8), 4.0, dtype=torch.float32),
        header=header_b,
        overwrite=True,
    )

    observations = [
        ImageObservation(str(path_g), wavelength=4800.0),
        ImageObservation(str(path_r), wavelength=6500.0),
    ]
    planned = plan_image_plane_target(observations, reference_index=0)
    result = coadd_image_plane_multiband(
        observations,
        target_header=planned.header,
        footprint_order=6,
    )

    assert tuple(result.data.shape[-2:]) == planned.shape
    assert result.header["NAXIS1"] == planned.header["NAXIS1"]
    assert result.header["NAXIS2"] == planned.header["NAXIS2"]


def test_plan_image_plane_cube_target_builds_band_metadata(tmp_path) -> None:
    header = _simple_header()
    path_g = tmp_path / "g.fits"
    path_r = tmp_path / "r.fits"
    torchfits.write(
        str(path_g),
        torch.ones((8, 8), dtype=torch.float32),
        header=header,
        overwrite=True,
    )
    torchfits.write(
        str(path_r),
        2.0 * torch.ones((8, 8), dtype=torch.float32),
        header=header,
        overwrite=True,
    )

    planned = plan_image_plane_cube_target(
        [
            ImageObservation(str(path_g), wavelength=4800.0, band_name="g"),
            ImageObservation(str(path_r), wavelength=6500.0, band_name="r"),
        ],
        reference_index=0,
    )

    torch.testing.assert_close(
        planned.spectral_axis.values,
        torch.tensor([4800.0, 6500.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        planned.band_metadata.center,
        torch.tensor([4800.0, 6500.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        planned.band_metadata.observation_count,
        torch.tensor([1, 1], dtype=torch.int64),
    )
    torch.testing.assert_close(
        planned.band_metadata.spectral_weight_sum,
        torch.tensor([1.0, 1.0], dtype=torch.float64),
    )
    assert planned.band_metadata.band_names == ("g", "r")


def test_coadd_image_plane_cube_reprojects_spectral_intervals(tmp_path) -> None:
    header = _simple_header()
    path = tmp_path / "band.fits"
    image = torch.full((8, 8), 4.0, dtype=torch.float32)
    torchfits.write(str(path), image, header=header, overwrite=True)

    spectral_axis = SpectralAxis(
        values=torch.tensor([4800.0, 5000.0, 5200.0], dtype=torch.float64),
        unit="Angstrom",
        type="WAVE",
    )
    spectral_bin_edges = torch.tensor(
        [4700.0, 4900.0, 5100.0, 5300.0],
        dtype=torch.float64,
    )

    result = coadd_image_plane_cube(
        [
            ImageObservation(
                str(path),
                weight_is_inverse_variance=True,
                wavelength=5000.0,
                spectral_min=4800.0,
                spectral_max=5200.0,
                band_name="wide",
            )
        ],
        spectral_axis=spectral_axis,
        spectral_bin_edges=spectral_bin_edges,
        footprint_order=6,
    )

    expected_weights = torch.tensor([0.25, 0.50, 0.25], dtype=torch.float32).view(
        3, 1, 1
    )
    torch.testing.assert_close(
        result.weight,
        expected_weights.expand_as(result.weight),
        atol=1e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        result.data,
        torch.full((3, 8, 8), 4.0, dtype=torch.float32),
        atol=1e-6,
        rtol=0.0,
    )
    assert result.inverse_variance is not None
    assert result.correlation_factor is not None
    torch.testing.assert_close(
        result.inverse_variance,
        torch.ones_like(result.inverse_variance),
    )
    torch.testing.assert_close(
        result.correlation_factor,
        torch.ones_like(result.correlation_factor),
    )
    assert result.coverage_mocs is not None
    assert len(result.coverage_mocs) == 3
    assert result.band_metadata.band_names == ("wide", "wide", "wide")
    torch.testing.assert_close(
        result.band_metadata.observation_count,
        torch.tensor([1, 1, 1], dtype=torch.int64),
    )
    torch.testing.assert_close(
        result.band_metadata.spectral_weight_sum,
        torch.tensor([0.25, 0.50, 0.25], dtype=torch.float64),
    )


def test_coadd_image_plane_cube_reprojects_gaussian_transmission_filters(
    tmp_path,
) -> None:
    header = _simple_header()
    path = tmp_path / "band_filter.fits"
    image = torch.full((8, 8), 5.0, dtype=torch.float32)
    torchfits.write(str(path), image, header=header, overwrite=True)

    spectral_axis = SpectralAxis(
        values=torch.tensor([4800.0, 5000.0, 5200.0], dtype=torch.float64),
        unit="Angstrom",
        type="WAVE",
    )
    spectral_bin_edges = torch.tensor(
        [4700.0, 4900.0, 5100.0, 5300.0],
        dtype=torch.float64,
    )

    result = coadd_image_plane_cube(
        [
            ImageObservation(
                str(path),
                wavelength=5000.0,
                spectral_min=4700.0,
                spectral_max=5300.0,
                transmission_center=5000.0,
                transmission_sigma=120.0,
                band_name="gauss",
            )
        ],
        spectral_axis=spectral_axis,
        spectral_bin_edges=spectral_bin_edges,
    )

    sqrt2 = torch.sqrt(torch.tensor(2.0, dtype=torch.float64))
    left = spectral_bin_edges[:-1]
    right = spectral_bin_edges[1:]
    expected = 0.5 * (
        torch.erf((right - 5000.0) / (120.0 * sqrt2))
        - torch.erf((left - 5000.0) / (120.0 * sqrt2))
    )
    expected = expected / expected.sum()
    torch.testing.assert_close(
        result.band_metadata.spectral_weight_sum,
        expected,
        atol=1e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        result.weight,
        expected.to(dtype=torch.float32).view(3, 1, 1).expand_as(result.weight),
        atol=1e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        result.data,
        torch.full((3, 8, 8), 5.0, dtype=torch.float32),
        atol=1e-6,
        rtol=0.0,
    )


def test_coadd_image_plane_cube_rejects_invalid_transmission_sigma(tmp_path) -> None:
    header = _simple_header()
    path = tmp_path / "band_invalid_filter.fits"
    image = torch.full((8, 8), 1.0, dtype=torch.float32)
    torchfits.write(str(path), image, header=header, overwrite=True)

    with pytest.raises(ValueError, match="transmission_sigma"):
        coadd_image_plane_cube(
            [
                ImageObservation(
                    str(path),
                    wavelength=5000.0,
                    spectral_min=4800.0,
                    spectral_max=5200.0,
                    transmission_center=5000.0,
                    transmission_sigma=0.0,
                )
            ],
            spectral_axis=SpectralAxis(
                values=torch.tensor([5000.0], dtype=torch.float64),
                unit="Angstrom",
                type="WAVE",
            ),
            spectral_bin_edges=torch.tensor([4900.0, 5100.0], dtype=torch.float64),
        )


def test_coadd_image_plane_cube_real_astropy_fixture_to_data_cube(tmp_path) -> None:
    real_path = _astropy_sip_fixture()
    image = torchfits.read(str(real_path)).to(dtype=torch.float32)

    mask_a = tmp_path / "mask_a.fits"
    mask_b = tmp_path / "mask_b.fits"
    mask_left = torch.zeros_like(image)
    mask_left[:, : image.shape[1] // 2] = 1.0
    mask_right = torch.zeros_like(image)
    mask_right[:, image.shape[1] // 2 :] = 1.0
    torchfits.write(str(mask_a), mask_left, overwrite=True)
    torchfits.write(str(mask_b), mask_right, overwrite=True)

    planned = plan_image_plane_cube_target(
        [
            ImageObservation(
                str(real_path),
                wavelength=4800.0,
                spectral_min=4700.0,
                spectral_max=4900.0,
                band_name="g",
                mask_path=str(mask_a),
            ),
            ImageObservation(
                str(real_path),
                wavelength=6500.0,
                spectral_min=6400.0,
                spectral_max=6600.0,
                band_name="r",
                mask_path=str(mask_b),
            ),
        ],
        spectral_axis=SpectralAxis(
            values=torch.tensor([4800.0, 6500.0], dtype=torch.float64),
            unit="Angstrom",
            type="WAVE",
        ),
        spectral_bin_edges=torch.tensor([4700.0, 4900.0, 6600.0], dtype=torch.float64),
    )

    result = coadd_image_plane_cube(
        [
            ImageObservation(
                str(real_path),
                wavelength=4800.0,
                spectral_min=4700.0,
                spectral_max=4900.0,
                band_name="g",
                mask_path=str(mask_a),
            ),
            ImageObservation(
                str(real_path),
                wavelength=6500.0,
                spectral_min=6400.0,
                spectral_max=6600.0,
                band_name="r",
                mask_path=str(mask_b),
            ),
        ],
        target=planned,
        footprint_order=6,
    )

    assert tuple(result.data.shape) == (2, 50, 100)
    torch.testing.assert_close(
        result.data[0, :, :50], image[:, :50], atol=1e-4, rtol=1e-4
    )
    torch.testing.assert_close(result.data[0, :, 50:], torch.zeros_like(image[:, 50:]))
    torch.testing.assert_close(result.data[1, :, :50], torch.zeros_like(image[:, :50]))
    torch.testing.assert_close(
        result.data[1, :, 50:], image[:, 50:], atol=1e-4, rtol=1e-4
    )
    cube = result.to_data_cube()
    assert cube.spatial_wcs is not None
    torch.testing.assert_close(cube.spectral_axis.values, result.spectral_axis.values)
    assert result.band_metadata.band_names == ("g", "r")


def test_coadd_image_plane_real_shifted_astropy_sip_fixture(tmp_path) -> None:
    real_path = _astropy_sip_fixture()
    image, header = torchfits.read(str(real_path), return_header=True)
    image = image.to(dtype=torch.float32)
    shifted_header = dict(header)
    shifted_header["CRVAL1"] = float(shifted_header["CRVAL1"]) + 0.01
    shifted_header["CRVAL2"] = float(shifted_header["CRVAL2"]) - 0.01
    shifted_path = tmp_path / "sip_shifted.fits"
    torchfits.write(str(shifted_path), image, header=shifted_header, overwrite=True)

    result = coadd_image_plane(
        [
            ImageObservation(str(real_path)),
            ImageObservation(str(shifted_path)),
        ]
    )

    assert tuple(result.data.shape) == tuple(int(v) for v in image.shape)
    assert torch.isfinite(result.data).all()
    assert torch.isfinite(result.weight).all()
    assert result.weight.max().item() > 0.0


def test_coadd_image_plane_large_sip_target_keeps_roundtrip_world_grid() -> None:
    real_path = _astropy_sip_fixture()
    _image, header = torchfits.read(str(real_path), return_header=True)
    target_header = dict(header)
    target_header["NAXIS1"] = 512
    target_header["NAXIS2"] = 512

    result = coadd_image_plane(
        [ImageObservation(str(real_path))],
        target_header=target_header,
        value_semantics="surface_brightness",
    )

    assert result.data.shape == (512, 512)
    assert torch.isfinite(result.data).all()
    assert result.weight.max().item() > 0.0


def test_plan_image_plane_cube_target_from_header_uses_spectral_wcs(tmp_path) -> None:
    header = _simple_header()
    path = tmp_path / "img.fits"
    torchfits.write(
        str(path),
        torch.ones((8, 8), dtype=torch.float32),
        header=header,
        overwrite=True,
    )

    spectral_header = {
        "NAXIS": 3,
        "NAXIS1": 8,
        "NAXIS2": 8,
        "NAXIS3": 3,
        "CTYPE3": "WAVE",
        "CUNIT3": "Angstrom",
        "CRVAL3": 4800.0,
        "CDELT3": 200.0,
        "CRPIX3": 1.0,
    }
    planned = plan_image_plane_cube_target_from_header(
        [
            ImageObservation(str(path), wavelength=4800.0, band_name="g"),
            ImageObservation(str(path), wavelength=5000.0, band_name="r"),
            ImageObservation(str(path), wavelength=5200.0, band_name="i"),
        ],
        spectral_header,
    )

    torch.testing.assert_close(
        planned.spectral_axis.values,
        torch.tensor([4800.0, 5000.0, 5200.0], dtype=torch.float64),
    )
    assert planned.spectral_bin_edges is not None
    torch.testing.assert_close(
        planned.spectral_bin_edges,
        torch.tensor([4700.0, 4900.0, 5100.0, 5300.0], dtype=torch.float64),
    )
    assert planned.band_metadata.band_names == ("g", "r", "i")
    band_table = planned.to_band_table()
    assert band_table["band_name"] == ("g", "r", "i")


_KERNELS_DETERMINISTIC = [
    "bilinear",
    "lanczos3",
    "catmull_rom",
    "drizzle_point",
    "drizzle_square",
    "conservative_polygon",
    "splat_point",
    "splat_box",
]

_STACKING_METHODS_DETERMINISTIC = [
    "mean",
    "weighted_mean",
    "median",
    "sigma_clipped",
]


@pytest.mark.parametrize("kernel", _KERNELS_DETERMINISTIC)
@pytest.mark.parametrize("stacking_method", _STACKING_METHODS_DETERMINISTIC)
def test_coadd_image_plane_deterministic_accumulation(
    tmp_path, kernel: str, stacking_method: str
) -> None:
    path_a = tmp_path / f"det_a_{kernel}.fits"
    path_b = tmp_path / f"det_b_{kernel}.fits"
    header_a = _simple_header()
    header_b = dict(header_a)
    header_b["CRVAL1"] = float(header_b["CRVAL1"]) + 0.01
    data_a = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    data_b = data_a + 3.0
    torchfits.write(str(path_a), data_a, header=header_a, overwrite=True)
    torchfits.write(str(path_b), data_b, header=header_b, overwrite=True)

    obs = [ImageObservation(str(path_a)), ImageObservation(str(path_b))]
    kwargs: dict[str, object] = dict(
        resampling_kernel=kernel,
        stacking_method=stacking_method,
    )
    result_a = coadd_image_plane(obs, **kwargs)
    result_b = coadd_image_plane(obs, **kwargs)

    assert torch.equal(result_a.data, result_b.data)
    assert torch.equal(result_a.weight, result_b.weight)
    if result_a.overlap_count is not None and result_b.overlap_count is not None:
        assert torch.equal(result_a.overlap_count, result_b.overlap_count)
