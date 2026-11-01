"""Footprint cropping of per-observation target work (any target size / WCS type)."""

from __future__ import annotations

import pytest
import torch

import torchsky
import torchsky.map.image_plane_projection as projection_mod
import torchsky.map.windowing as windowing_mod
from torchsky.map.observation import ImageObservation

pytest.importorskip("torchsky._cpp")


def _sip_header(size: int, *, crval1: float = 120.0, crval2: float = 5.0) -> dict:
    # Strongly curved SIP edges (cubic + quintic terms) on a 96-pixel detector.
    return {
        "NAXIS": 2,
        "NAXIS1": size,
        "NAXIS2": size,
        "CTYPE1": "RA---TAN-SIP",
        "CTYPE2": "DEC--TAN-SIP",
        "CRPIX1": size / 2 + 0.5,
        "CRPIX2": size / 2 + 0.5,
        "CRVAL1": crval1,
        "CRVAL2": crval2,
        "CD1_1": -0.0003,
        "CD1_2": 0.00004,
        "CD2_1": 0.00004,
        "CD2_2": 0.0003,
        "A_ORDER": 5,
        "B_ORDER": 5,
        "A_2_0": 2.0e-5,
        "A_1_1": -1.0e-5,
        "A_3_0": 3.0e-7,
        "A_5_0": -2.0e-10,
        "B_0_2": -1.5e-5,
        "B_1_1": 1.0e-5,
        "B_0_3": 2.0e-7,
        "B_0_5": 1.0e-10,
    }


def _tan_target(size: int, *, crval1: float, crval2: float, scale: float) -> dict:
    return {
        "NAXIS": 2,
        "NAXIS1": size,
        "NAXIS2": size,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": size / 2 + 0.5,
        "CRPIX2": size / 2 + 0.5,
        "CRVAL1": crval1,
        "CRVAL2": crval2,
        "CDELT1": -scale,
        "CDELT2": scale,
    }


def _prepared(header: dict, *, quadrature: bool = False):
    return windowing_mod._prepare_target(
        header=header,
        target_wcs=torchsky.WCS(header),
        target_shape=(int(header["NAXIS2"]), int(header["NAXIS1"])),
        device=torch.device("cpu"),
        include_pixel_area=quadrature,
        include_representative_pixel_area=True,
        include_quadrature=quadrature,
    )


def _project(source_header, target_header, image, *, kernel, semantics, window, monkeypatch):
    target = _prepared(target_header, quadrature=semantics == "flux")
    monkeypatch.setattr(
        windowing_mod,
        "_LARGE_TAN_WINDOWING_MIN_TARGET_PIXELS",
        0 if window else 10**12,
    )
    shape = (int(target_header["NAXIS2"]), int(target_header["NAXIS1"]))
    chunk = projection_mod._project_observation_to_target_chunk(
        ImageObservation("mock.fits"),
        image=image,
        header_dict_obs=dict(source_header),
        source_wcs=torchsky.WCS(dict(source_header)),
        prepared_target=target,
        target_header=target_header,
        target_shape=shape,
        device=torch.device("cpu"),
        footprint_order=None,
        resampling_kernel=kernel,
        value_semantics=semantics,
        drizzle_pixfrac=0.8,
    )
    full = projection_mod._project_observation_to_target(
        ImageObservation("mock.fits"),
        image=image,
        header_dict_obs=dict(source_header),
        source_wcs=torchsky.WCS(dict(source_header)),
        prepared_target=target,
        target_header=target_header,
        target_shape=shape,
        device=torch.device("cpu"),
        footprint_order=None,
        resampling_kernel=kernel,
        value_semantics=semantics,
        drizzle_pixfrac=0.8,
    )
    return chunk, full


@pytest.mark.parametrize("kernel", ["bilinear", "lanczos3", "drizzle_square"])
@pytest.mark.parametrize("semantics", ["surface_brightness", "flux"])
@pytest.mark.parametrize("target_kind", ["tan", "sip"])
def test_footprint_crop_is_identical_to_full_target_for_partial_sip_coverage(
    kernel, semantics, target_kind, monkeypatch
) -> None:
    size = 96
    source_header = _sip_header(size)
    # ~25% of the 160^2 target is covered by the exposure.
    if target_kind == "tan":
        target_header = _tan_target(160, crval1=120.0 + 0.012, crval2=5.0 + 0.014, scale=0.0003)
    else:
        target_header = {
            **_sip_header(160, crval1=120.0 + 0.012, crval2=5.0 + 0.014),
            "CD1_1": -0.0003,
            "CD1_2": 0.0,
            "CD2_1": 0.0,
            "CD2_2": 0.0003,
        }
    gen = torch.Generator().manual_seed(7)
    image = torch.randn((size, size), generator=gen, dtype=torch.float32) + 5.0

    chunk, cropped = _project(
        source_header, target_header, image, kernel=kernel, semantics=semantics, window=True, monkeypatch=monkeypatch
    )
    _, full = _project(
        source_header, target_header, image, kernel=kernel, semantics=semantics, window=False, monkeypatch=monkeypatch
    )

    assert chunk.target_window is not None
    y0, y1, x0, x1 = chunk.target_window
    assert (y1 - y0) * (x1 - x0) < 0.7 * 160 * 160
    assert float(full[1].sum()) > 0.0
    # Inside the window results are bit-identical; outside it nothing was contributed.
    torch.testing.assert_close(cropped[0], full[0], atol=0.0, rtol=0.0)
    torch.testing.assert_close(cropped[1], full[1], atol=0.0, rtol=0.0)
    outside = torch.ones(full[1].shape, dtype=torch.bool)
    outside[y0:y1, x0:x1] = False
    assert float(full[1][outside].abs().sum()) == 0.0
    assert float(full[0][outside].abs().sum()) == 0.0


def test_footprint_bbox_contains_exact_footprint_of_curved_sip_source() -> None:
    size = 96
    source_header = _sip_header(size)
    source_wcs = torchsky.WCS(source_header)
    target_header = _tan_target(220, crval1=120.0 + 0.004, crval2=5.0 - 0.006, scale=0.0003)
    target_wcs = torchsky.WCS(target_header)
    shape = (220, 220)
    bbox = windowing_mod._footprint_bbox_on_target(
        source_wcs, (size, size), target_wcs, shape, margin=0, kernel_half_width=0.0
    )
    assert bbox is not None
    y0, y1, x0, x1 = bbox
    # Exact footprint: every target pixel whose centre maps inside the source domain.
    yy, xx = torch.meshgrid(
        torch.arange(shape[0], dtype=torch.float64),
        torch.arange(shape[1], dtype=torch.float64),
        indexing="ij",
    )
    ra, dec = target_wcs.pixel_to_world(xx, yy)
    xs, ys = source_wcs.world_to_pixel(ra.reshape(-1), dec.reshape(-1))
    inside = (
        (xs >= -0.5) & (xs <= size - 0.5) & (ys >= -0.5) & (ys <= size - 0.5)
    ).reshape(shape)
    assert bool(inside.any())
    iy, ix = torch.nonzero(inside, as_tuple=True)
    assert int(iy.min()) >= y0 and int(iy.max()) < y1
    assert int(ix.min()) >= x0 and int(ix.max()) < x1


def test_window_no_overlap_and_unusable_footprints_fall_back() -> None:
    size = 32
    source_header = _sip_header(size)
    source_wcs = torchsky.WCS(source_header)
    far = _tan_target(40, crval1=120.0 + 5.0, crval2=5.0, scale=0.0003)
    prepared = _prepared(far)
    view, window = windowing_mod._window_large_tan_target(
        source_wcs=source_wcs,
        source_shape=(size, size),
        target=prepared,
        value_semantics="surface_brightness",
        resampling_kernel="lanczos3",
    )
    assert view is None and window == (0, 0, 0, 0)

    # Footprint on the far side of a zenithal target: bbox of the samples is not
    # meaningful, so the full target is used.
    opposite = _tan_target(40, crval1=120.0 + 180.0, crval2=-5.0, scale=0.0003)
    prepared = _prepared(opposite)
    view, window = windowing_mod._window_large_tan_target(
        source_wcs=source_wcs,
        source_shape=(size, size),
        target=prepared,
        value_semantics="surface_brightness",
        resampling_kernel="lanczos3",
    )
    assert view is prepared and window is None

    # Pole inside the footprint of a cylindrical target: no cropping.
    pole_header = {
        **_tan_target(size, crval1=10.0, crval2=89.999, scale=0.0003),
    }
    car = {
        "NAXIS": 2,
        "NAXIS1": 720,
        "NAXIS2": 360,
        "CTYPE1": "RA---CAR",
        "CTYPE2": "DEC--CAR",
        "CRPIX1": 360.5,
        "CRPIX2": 180.5,
        "CRVAL1": 0.0,
        "CRVAL2": 0.0,
        "CDELT1": -0.5,
        "CDELT2": 0.5,
    }
    prepared = _prepared(car)
    view, window = windowing_mod._window_large_tan_target(
        source_wcs=torchsky.WCS(pole_header),
        source_shape=(size, size),
        target=prepared,
        value_semantics="surface_brightness",
        resampling_kernel="bilinear",
    )
    assert view is prepared and window is None


def test_window_cylindrical_target_matches_full(monkeypatch) -> None:
    size = 48
    source_header = _tan_target(size, crval1=30.0, crval2=20.0, scale=0.05)
    car = {
        "NAXIS": 2,
        "NAXIS1": 180,
        "NAXIS2": 90,
        "CTYPE1": "RA---CAR",
        "CTYPE2": "DEC--CAR",
        "CRPIX1": 90.5,
        "CRPIX2": 45.5,
        "CRVAL1": 30.0,
        "CRVAL2": 0.0,
        "CDELT1": -0.5,
        "CDELT2": 0.5,
    }
    # Make the target finer than the source so the footprint is a small box.
    car["CDELT1"] = -0.05
    car["CDELT2"] = 0.05
    car["NAXIS1"] = 200
    car["NAXIS2"] = 160
    car["CRPIX1"] = 100.5
    car["CRPIX2"] = 80.5
    car["CRVAL2"] = 18.0
    image = torch.rand((size, size), dtype=torch.float32) + 1.0
    chunk, cropped = _project(
        source_header, car, image, kernel="bilinear", semantics="surface_brightness", window=True, monkeypatch=monkeypatch
    )
    _, full = _project(
        source_header, car, image, kernel="bilinear", semantics="surface_brightness", window=False, monkeypatch=monkeypatch
    )
    assert chunk.target_window is not None
    torch.testing.assert_close(cropped[0], full[0], atol=0.0, rtol=0.0)
    torch.testing.assert_close(cropped[1], full[1], atol=0.0, rtol=0.0)
    assert float(full[1].sum()) > 0.0
