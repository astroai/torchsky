"""Tests for :mod:`torchsky.map.field.exposure_io` on small synthetic exposure files."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import torchfits

from torchsky.map.field import observation_ops
from torchsky.map.field.exposure_io import (
    ExposureFileData,
    _window_for_sky_box,
    field_observation_from_exposure_file,
)
from torchsky.map.field.observation_ops import observation_pixel_to_world
from torchsky.map.field.types import FieldObservation
from torchsky.map.quality import DQPolicy
from torchsky.wcs import WCS

H, W = 48, 64
CRVAL = (150.1, 2.2)
SCALE_DEG = 0.05 / 3600.0  # 0.05 arcsec per pixel
ROT = math.radians(20.0)


def _wcs_header(
    *, crpix: tuple[float, float] = (32.5, 24.5), crval: tuple[float, float] = CRVAL
) -> dict:
    c, s = math.cos(ROT), math.sin(ROT)
    return {
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": crpix[0],
        "CRPIX2": crpix[1],
        "CRVAL1": crval[0],
        "CRVAL2": crval[1],
        "CUNIT1": "deg",
        "CUNIT2": "deg",
        "CD1_1": -SCALE_DEG * c,
        "CD1_2": SCALE_DEG * s,
        "CD2_1": SCALE_DEG * s,
        "CD2_2": SCALE_DEG * c,
    }


def _planes(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    image = torch.randn(H, W, generator=g)
    err = 0.5 + torch.rand(H, W, generator=g)
    dq = torch.zeros(H, W, dtype=torch.int64)
    return image, err, dq


def _hdu(data: torch.Tensor, name: str, extver: int | None = None, **cards):
    header = {"EXTNAME": name, **cards}
    if extver is not None:
        header["EXTVER"] = extver
    return torchfits.TensorHDU(data=data, header=header)


def _write(
    path: Path,
    image: torch.Tensor,
    err: torch.Tensor,
    dq: torch.Tensor | None,
    *,
    primary: dict | None = None,
    sci_cards: dict | None = None,
    extra: list | None = None,
) -> str:
    cards = {"BUNIT": "MJy/sr", **_wcs_header(), **(sci_cards or {})}
    hdus = [
        torchfits.TensorHDU(
            data=torch.zeros(1),
            header={
                key: value
                for key, value in {
                    "FILTER": "F150W",
                    "PUPIL": "CLEAR",
                    "EFFEXPTM": 100.0,
                    **(primary or {}),
                }.items()
                if value is not None  # None removes a default card
            },
        ),
        _hdu(image, "SCI", 1, **cards),
        _hdu(err, "ERR", 1),
    ]
    if dq is not None:
        hdus.append(_hdu(dq.to(torch.uint32), "DQ", 1))
    hdus.extend(extra or [])
    torchfits.HDUList(hdus).write(str(path), overwrite=True)
    return str(path)


@pytest.fixture
def basic(tmp_path: Path):
    image, err, dq = _planes()
    path = _write(tmp_path / "basic_cal.fits", image, err, dq)
    return path, image, err, dq


def _unit_vectors(lon_deg: torch.Tensor, lat_deg: torch.Tensor) -> torch.Tensor:
    lon, lat = torch.deg2rad(lon_deg), torch.deg2rad(lat_deg)
    return torch.stack(
        (
            torch.cos(lat) * torch.cos(lon),
            torch.cos(lat) * torch.sin(lon),
            torch.sin(lat),
        ),
        -1,
    )


def _oracle_inside(path: str, box: tuple[float, float, float]) -> torch.Tensor:
    """Brute-force box membership of every pixel centre, with vectors rather than the
    gnomonic formulas of the module: offsets along east / north of the box centre."""
    header = dict(torchfits.read_header(path, "SCI"))
    y, x = torch.meshgrid(
        torch.arange(H, dtype=torch.float64),
        torch.arange(W, dtype=torch.float64),
        indexing="ij",
    )
    lon, lat = WCS(header).pixel_to_world(x, y)
    n = _unit_vectors(lon, lat)
    centre = _unit_vectors(
        torch.tensor(box[0], dtype=torch.float64),
        torch.tensor(box[1], dtype=torch.float64),
    )
    pole = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    east = torch.linalg.cross(pole, centre)
    east = east / east.norm()
    north = torch.linalg.cross(centre, east)
    along = n @ centre
    arcsec = 180.0 * 3600.0 / math.pi
    xi, eta = (n @ east) / along * arcsec, (n @ north) / along * arcsec
    return (xi.abs() <= box[2]) & (eta.abs() <= box[2])


def _box_at_pixel(path: str, px: float, py: float, half_arcsec: float):
    header = dict(torchfits.read_header(path, "SCI"))
    lon, lat = WCS(header).pixel_to_world(
        torch.tensor([px], dtype=torch.float64), torch.tensor([py], dtype=torch.float64)
    )
    return float(lon[0]), float(lat[0]), half_arcsec


# ---------------------------------------------------------------- round trip and metadata


def test_round_trip_planes_metadata_and_dtypes(basic) -> None:
    path, image, err, _ = basic
    obs = field_observation_from_exposure_file(path)
    assert obs is not None
    assert obs.image.dtype == torch.float32 and obs.weight.dtype == torch.float32
    assert tuple(obs.image.shape) == (H, W)
    torch.testing.assert_close(obs.image, image.float())
    torch.testing.assert_close(obs.err, err.float())
    torch.testing.assert_close(
        obs.weight, (1.0 / err.double() ** 2).float(), rtol=1e-6, atol=0
    )
    assert obs.mask.dtype == torch.bool and bool(obs.mask.all())
    assert (obs.detector_x0_pix, obs.detector_y0_pix) == (0.0, 0.0)
    assert obs.header["NAXIS1"] == W and obs.header["NAXIS2"] == H
    assert obs.value_unit == "MJy/sr" and obs.value_semantics == "surface_brightness"
    assert obs.exposure_time_s == 100.0 and obs.band_name == "F150W"
    assert obs.detector_to_sky is None
    assert obs.var_poisson is None and obs.var_rnoise is None and obs.var_flat is None


def test_whole_image_equals_full_read_of_every_plane(basic) -> None:
    path, _, _, _ = basic
    obs = field_observation_from_exposure_file(path)
    torch.testing.assert_close(obs.image, torchfits.read(path, hdu="SCI").float())
    torch.testing.assert_close(obs.err, torchfits.read(path, hdu="ERR").float())


# ---------------------------------------------------------------- validity, one cause at a time


@pytest.mark.parametrize(
    "cause",
    [
        "nan_data",
        "inf_data",
        "zero_err",
        "negative_err",
        "nan_err",
        "inf_err",
        "dq_flag",
    ],
)
def test_each_failure_cause_zeroes_weight_and_mask(tmp_path: Path, cause: str) -> None:
    image, err, dq = _planes(1)
    bad = (slice(5, 7), slice(9, 12))
    if cause == "nan_data":
        image[bad] = float("nan")
    elif cause == "inf_data":
        image[bad] = float("inf")
    elif cause == "zero_err":
        err[bad] = 0.0
    elif cause == "negative_err":
        err[bad] = -1.0
    elif cause == "nan_err":
        err[bad] = float("nan")
    elif cause == "inf_err":
        err[bad] = float("inf")
    else:
        dq[bad] = 4
    obs = field_observation_from_exposure_file(
        _write(tmp_path / "f.fits", image, err, dq)
    )
    expected = torch.ones(H, W, dtype=torch.bool)
    expected[bad] = False
    assert torch.equal(obs.mask, expected)
    assert bool((obs.weight[~expected] == 0).all())
    assert bool((obs.weight[expected] > 0).all())
    assert bool((obs.image[~expected] == 0).all())  # no NaN reaches a loss
    assert bool(torch.isfinite(obs.image).all()) and bool(torch.isfinite(obs.err).all())
    torch.testing.assert_close(obs.image[expected], image[expected].float())
    # The certified-observation contract holds for every case.
    assert bool((obs.weight[obs.mask] > 0).all()) and bool(
        (obs.weight[~obs.mask] == 0).all()
    )


def test_dq_policy_default_versus_reject_bits(tmp_path: Path) -> None:
    image, err, dq = _planes(2)
    dq[1, 1] = 2  # SATURATED only
    dq[2, 2] = 1 | 2  # DO_NOT_USE and SATURATED
    dq[3, 3] = 4  # another bit only
    dq[4, 4] = 1 << 31  # a high bit: stored as uint32, must not wrap negative
    path = _write(tmp_path / "dq.fits", image, err, dq)

    strict = field_observation_from_exposure_file(path)
    for yx in ((1, 1), (2, 2), (3, 3), (4, 4)):
        assert not bool(strict.mask[yx])
    assert int(strict.mask.sum()) == H * W - 4

    keep_sat = field_observation_from_exposure_file(
        path, dq_policy=DQPolicy(reject_bits=0xFFFFFFFF & ~2)
    )
    assert bool(keep_sat.mask[1, 1]) and float(keep_sat.weight[1, 1]) > 0
    assert (
        not bool(keep_sat.mask[2, 2])
        and not bool(keep_sat.mask[3, 3])
        and not bool(keep_sat.mask[4, 4])
    )

    only_dnu = field_observation_from_exposure_file(
        path, dq_policy=DQPolicy(reject_bits=1)
    )
    assert (
        bool(only_dnu.mask[1, 1])
        and bool(only_dnu.mask[3, 3])
        and bool(only_dnu.mask[4, 4])
    )
    assert not bool(only_dnu.mask[2, 2])


def test_no_quality_extension_requested(tmp_path: Path) -> None:
    image, err, _ = _planes(3)
    path = _write(tmp_path / "nodq.fits", image, err, None)
    obs = field_observation_from_exposure_file(path, quality_hdu=None)
    assert bool(obs.mask.all())
    with pytest.raises(ValueError, match="quality_hdu"):
        field_observation_from_exposure_file(
            path, quality_hdu=None, dq_policy=DQPolicy()
        )
    with pytest.raises(ValueError, match="'DQ'.*not found"):
        field_observation_from_exposure_file(path)


def test_weight_overflow_in_output_precision_stays_usable_with_clipped_weight(
    tmp_path: Path,
) -> None:
    image, err, dq = _planes(4)
    err[10, 10] = 1e-30  # 1 / err**2 = 1e60: finite in float64, infinite in float32
    path = _write(tmp_path / "tiny.fits", image, err, dq)
    f32 = field_observation_from_exposure_file(path)
    assert bool(f32.mask[10, 10])
    assert float(f32.weight[10, 10]) == torch.finfo(torch.float32).max
    assert bool(torch.isfinite(f32.weight).all())
    assert float(f32.image[10, 10]) == pytest.approx(float(image[10, 10]))
    f64 = field_observation_from_exposure_file(path, dtype=torch.float64)
    assert bool(f64.mask[10, 10]) and float(f64.weight[10, 10]) == pytest.approx(
        1e60, rel=1e-6
    )


def test_unusable_err_still_drops_pixels_near_the_overflow_case(tmp_path: Path) -> None:
    image, err, dq = _planes(4)
    err[3, 3] = 0.0
    err[4, 4] = -1.0
    err[5, 5] = float("nan")
    path = _write(tmp_path / "bad_err.fits", image, err, dq)
    obs = field_observation_from_exposure_file(path)
    assert not bool(obs.mask[3, 3] | obs.mask[4, 4] | obs.mask[5, 5])
    assert int(obs.mask.sum()) == H * W - 3


# ---------------------------------------------------------------- sky-box cut


@pytest.mark.parametrize(
    "centre_pix, half",
    [
        ((30.0, 20.0), 0.6),
        ((10.0, 40.0), 0.9),
        ((60.0, 5.0), 0.4),
        ((33.3, 17.7), 0.04),
    ],
)
def test_sky_box_cut_matches_independent_computation(basic, centre_pix, half) -> None:
    path, image, err, _ = basic
    box = _box_at_pixel(path, *centre_pix, half)
    inside = _oracle_inside(path, box)
    assert bool(inside.any())
    rows, cols = torch.nonzero(inside, as_tuple=True)
    y0, y1, x0, x1 = (
        int(rows.min()),
        int(rows.max()) + 1,
        int(cols.min()),
        int(cols.max()) + 1,
    )

    obs = field_observation_from_exposure_file(path, sky_box=box)
    assert obs is not None
    assert tuple(obs.image.shape) == (y1 - y0, x1 - x0)
    assert (obs.detector_x0_pix, obs.detector_y0_pix) == (float(x0), float(y0))
    torch.testing.assert_close(obs.image, image[y0:y1, x0:x1].float())
    torch.testing.assert_close(obs.err, err[y0:y1, x0:x1].float())

    full = dict(torchfits.read_header(path, "SCI"))
    assert obs.header["NAXIS1"] == x1 - x0 and obs.header["NAXIS2"] == y1 - y0
    assert obs.header["CRPIX1"] == pytest.approx(full["CRPIX1"] - x0, abs=1e-12)
    assert obs.header["CRPIX2"] == pytest.approx(full["CRPIX2"] - y0, abs=1e-12)

    # Sky position of the same detector pixel before and after the cut.
    ly, lx = torch.meshgrid(
        torch.arange(y1 - y0, dtype=torch.float64),
        torch.arange(x1 - x0, dtype=torch.float64),
        indexing="ij",
    )
    ra_cut, dec_cut = observation_pixel_to_world(obs, lx, ly)
    ra_full, dec_full = WCS(full).pixel_to_world(lx + x0, ly + y0)
    assert float((ra_cut - ra_full).abs().max()) < 1e-9
    assert float((dec_cut - dec_full).abs().max()) < 1e-9
    # Nothing outside the rectangle holds a pixel centre inside the box.
    outside = inside.clone()
    outside[y0:y1, x0:x1] = False
    assert not bool(outside.any())


def test_cut_equals_crop_of_the_whole_read(basic) -> None:
    path, _, _, _ = basic
    box = _box_at_pixel(path, 28.0, 22.0, 0.7)
    cut = field_observation_from_exposure_file(path, sky_box=box)
    whole = field_observation_from_exposure_file(path)
    x0, y0 = int(cut.detector_x0_pix), int(cut.detector_y0_pix)
    reference = observation_ops.crop_field_observation(
        whole, x0=x0, y0=y0, width=cut.image.shape[1], height=cut.image.shape[0]
    )
    torch.testing.assert_close(cut.image, reference.image)
    torch.testing.assert_close(cut.weight, reference.weight)
    assert torch.equal(cut.mask, reference.mask)
    assert cut.header == reference.header
    assert (cut.detector_x0_pix, cut.detector_y0_pix) == (
        reference.detector_x0_pix,
        reference.detector_y0_pix,
    )


def test_box_covering_the_chip_keeps_every_pixel(basic) -> None:
    path, image, _, _ = basic
    obs = field_observation_from_exposure_file(path, sky_box=(CRVAL[0], CRVAL[1], 60.0))
    assert tuple(obs.image.shape) == (H, W) and (
        obs.detector_x0_pix,
        obs.detector_y0_pix,
    ) == (0.0, 0.0)
    torch.testing.assert_close(obs.image, image.float())


def test_mask_outside_box_marks_corners_of_the_rectangle(basic) -> None:
    path, _, _, _ = basic
    box = _box_at_pixel(
        path, 32.0, 24.0, 0.8
    )  # a square rotated 20 degrees on the detector
    plain = field_observation_from_exposure_file(path, sky_box=box)
    masked = field_observation_from_exposure_file(
        path, sky_box=box, mask_outside_box=True
    )
    assert bool(plain.mask.all())
    assert 0 < int(masked.mask.sum()) < masked.mask.numel()
    inside = _oracle_inside(path, box)
    x0, y0 = int(masked.detector_x0_pix), int(masked.detector_y0_pix)
    h, w = masked.image.shape
    assert torch.equal(masked.mask, inside[y0 : y0 + h, x0 : x0 + w])
    assert bool((masked.weight[~masked.mask] == 0).all())


def test_window_search_finds_a_box_between_coarse_samples() -> None:
    """A box smaller than the coarse spacing, centred between samples, must still be found."""
    header = _wcs_header()
    wcs = WCS(header)

    def world(x, y):
        return wcs.pixel_to_world(x, y)

    for stride in (4, 8, 32):
        for px, py, half, size in (
            (6.0, 6.0, 0.012, 1),  # a quarter-pixel box on one pixel centre
            (13.0, 21.0, 0.012, 1),
            (50.0, 10.0, 0.012, 1),
            (13.5, 21.5, 0.04, 2),  # between four pixel centres, 0.8 pixel half-size
        ):
            lon, lat = wcs.pixel_to_world(
                torch.tensor([px], dtype=torch.float64),
                torch.tensor([py], dtype=torch.float64),
            )
            found = _window_for_sky_box(
                world, H, W, (float(lon[0]), float(lat[0]), half), stride=stride
            )
            assert found is not None
            x0, y0, x1, y1, inside = found
            assert (x1 - x0, y1 - y0) == (size, size) and bool(inside.all())
            assert (x0, y0) == (math.floor(px), math.floor(py))
    # Between pixel centres and smaller than one pixel: no centre inside, so no window.
    lon, lat = wcs.pixel_to_world(
        torch.tensor([13.5], dtype=torch.float64),
        torch.tensor([21.5], dtype=torch.float64),
    )
    assert (
        _window_for_sky_box(
            world, H, W, (float(lon[0]), float(lat[0]), 0.012), stride=8
        )
        is None
    )


def test_window_search_with_a_strongly_nonlinear_transform() -> None:
    wcs = WCS(_wcs_header())

    def world(x, y):
        # A quadratic distortion of 0.6 pixel at the corners, smooth over a coarse cell.
        u, v = x - 32.0, y - 24.0
        return wcs.pixel_to_world(x + 3.0e-4 * u * u, y + 4.0e-4 * v * v)

    y, x = torch.meshgrid(
        torch.arange(H, dtype=torch.float64),
        torch.arange(W, dtype=torch.float64),
        indexing="ij",
    )
    lon, lat = world(x, y)
    for py, px in ((3, 60), (40, 5), (24, 32)):
        box = (float(lon[py, px]), float(lat[py, px]), 0.06)
        found = _window_for_sky_box(world, H, W, box, stride=8)
        n = _unit_vectors(lon, lat)
        c = _unit_vectors(
            torch.tensor(box[0], dtype=torch.float64),
            torch.tensor(box[1], dtype=torch.float64),
        )
        east = torch.linalg.cross(torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64), c)
        east = east / east.norm()
        north = torch.linalg.cross(c, east)
        scale = 180.0 * 3600.0 / math.pi / (n @ c)
        brute = ((n @ east) * scale).abs().le(box[2]) & ((n @ north) * scale).abs().le(
            box[2]
        )
        rows, cols = torch.nonzero(brute, as_tuple=True)
        assert found is not None
        assert found[:4] == (
            int(cols.min()),
            int(rows.min()),
            int(cols.max()) + 1,
            int(rows.max()) + 1,
        )


def test_empty_cut_returns_none(basic) -> None:
    path, _, _, _ = basic
    far = (CRVAL[0] + 1.0, CRVAL[1], 1.0)  # 1 degree away
    assert field_observation_from_exposure_file(path, sky_box=far) is None
    opposite = ((CRVAL[0] + 180.0) % 360.0, -CRVAL[1], 5.0)  # far side of the sky
    assert field_observation_from_exposure_file(path, sky_box=opposite) is None
    just_off = _box_at_pixel(
        path, 200.0, 24.0, 0.2
    )  # centre 130 pixels beyond the edge
    assert field_observation_from_exposure_file(path, sky_box=just_off) is None


def test_min_valid_pixels_makes_a_cut_empty(tmp_path: Path) -> None:
    image, err, dq = _planes(5)
    dq[:, :] = 1  # everything flagged
    path = _write(tmp_path / "flagged.fits", image, err, dq)
    assert field_observation_from_exposure_file(path) is None
    assert (
        field_observation_from_exposure_file(path, sky_box=(CRVAL[0], CRVAL[1], 5.0))
        is None
    )
    dq[0:3, 0:4] = 0  # 12 usable pixels
    path = _write(tmp_path / "some.fits", image, err, dq)
    assert field_observation_from_exposure_file(path, min_valid_pixels=12) is not None
    assert field_observation_from_exposure_file(path, min_valid_pixels=13) is None


def test_detector_to_sky_replaces_the_header_wcs(tmp_path: Path) -> None:
    image, err, dq = _planes(6)
    decoy = {"CRVAL1": 10.0, "CRVAL2": -40.0}  # header WCS that points elsewhere
    path = _write(tmp_path / "native.fits", image, err, dq, sci_cards=decoy)
    wcs = WCS(_wcs_header())

    class Native:
        def pixel_to_world(self, x, y):
            return wcs.pixel_to_world(x, y)

    box = _box_at_pixel(_write(tmp_path / "ref.fits", image, err, dq), 20.0, 30.0, 0.5)
    assert (
        field_observation_from_exposure_file(path, sky_box=box) is None
    )  # header WCS: no overlap
    obs = field_observation_from_exposure_file(
        path, sky_box=box, detector_to_sky=Native()
    )
    assert obs is not None and isinstance(obs.detector_to_sky, Native)
    x0, y0 = int(obs.detector_x0_pix), int(obs.detector_y0_pix)
    assert (x0, y0) != (0, 0)
    ra, dec = observation_pixel_to_world(
        obs,
        torch.tensor([2.0], dtype=torch.float64),
        torch.tensor([3.0], dtype=torch.float64),
    )
    ra_ref, dec_ref = wcs.pixel_to_world(
        torch.tensor([2.0 + x0], dtype=torch.float64),
        torch.tensor([3.0 + y0], dtype=torch.float64),
    )
    assert abs(float(ra - ra_ref)) < 1e-9 and abs(float(dec - dec_ref)) < 1e-9


# ---------------------------------------------------------------- chips, rate files, variance


def test_multi_chip_extver(tmp_path: Path) -> None:
    image1, err1, dq1 = _planes(7)
    image2, err2, dq2 = _planes(8)
    dq2[0, 0] = 1
    extra = [
        _hdu(image2, "SCI", 2, BUNIT="electrons", **_wcs_header(crval=(150.1, 2.25))),
        _hdu(err2, "ERR", 2),
        _hdu(dq2.to(torch.uint32), "DQ", 2),
    ]
    path = _write(
        tmp_path / "flt.fits",
        image1,
        err1,
        dq1,
        extra=extra,
        primary={"EXPTIME": 30.0, "EFFEXPTM": None},
    )
    one = field_observation_from_exposure_file(path, extver=1)
    two = field_observation_from_exposure_file(path, extver=2)
    torch.testing.assert_close(one.image, image1.float())
    expected_two = image2.float().clone()
    expected_two[0, 0] = 0.0  # flagged in chip 2 only: zeroed
    torch.testing.assert_close(two.image, expected_two)
    assert one.value_unit == "MJy/sr" and two.value_unit == "electrons"
    assert two.value_semantics is None
    assert bool(one.mask[0, 0]) and not bool(two.mask[0, 0])
    assert two.header["CRVAL2"] == pytest.approx(2.25)
    with pytest.raises(ValueError, match="several EXTVER.*pass extver"):
        field_observation_from_exposure_file(path)
    with pytest.raises(ValueError, match="EXTVER=3"):
        field_observation_from_exposure_file(path, extver=3)


def test_count_rate_file_without_area_plane(tmp_path: Path) -> None:
    image, err, dq = _planes(9)
    path = _write(
        tmp_path / "x_rate.fits",
        image,
        err,
        dq,
        sci_cards={"BUNIT": "DN/s", "PIXAR_SR": 9.4e-14},
        primary={"EFFEXPTM": None, "XPOSURE": 250.0},
    )
    obs = field_observation_from_exposure_file(path)
    assert obs.value_unit == "DN/s" and obs.value_semantics is None
    assert obs.exposure_time_s == 250.0
    assert obs.header["PIXAR_SR"] == 9.4e-14  # kept in the header for the caller
    explicit = field_observation_from_exposure_file(
        path, value_semantics="surface_brightness", value_unit="DN/s/pix"
    )
    assert (
        explicit.value_semantics == "surface_brightness"
        and explicit.value_unit == "DN/s/pix"
    )


def test_calibrated_file_with_area_plane_ignores_it(tmp_path: Path) -> None:
    image, err, dq = _planes(10)
    area = torch.full((H, W), 1.0)
    path = _write(
        tmp_path / "x_cal.fits", image, err, dq, extra=[_hdu(area, "AREA", 1)]
    )
    obs = field_observation_from_exposure_file(path)
    assert not hasattr(obs, "area") and bool(obs.mask.all())


def test_variance_planes_read_when_present(tmp_path: Path) -> None:
    image, err, dq = _planes(11)
    poisson, rnoise = torch.rand(H, W), torch.rand(H, W)
    path = _write(
        tmp_path / "var.fits",
        image,
        err,
        dq,
        extra=[_hdu(poisson, "VAR_POISSON", 1), _hdu(rnoise, "VAR_RNOISE", 1)],
    )
    off = field_observation_from_exposure_file(path)
    assert off.var_poisson is None and off.var_rnoise is None
    on = field_observation_from_exposure_file(path, read_variance_planes=True)
    torch.testing.assert_close(on.var_poisson, poisson)
    torch.testing.assert_close(on.var_rnoise, rnoise)
    assert on.var_flat is None  # absent in the file
    cut = field_observation_from_exposure_file(
        path, read_variance_planes=True, sky_box=_box_at_pixel(path, 30.0, 20.0, 0.5)
    )
    x0, y0 = int(cut.detector_x0_pix), int(cut.detector_y0_pix)
    h, w = cut.image.shape
    torch.testing.assert_close(cut.var_rnoise, rnoise[y0 : y0 + h, x0 : x0 + w])


def test_band_name_and_exposure_time_keywords(tmp_path: Path) -> None:
    image, err, dq = _planes(12)
    path = _write(
        tmp_path / "nb.fits",
        image,
        err,
        dq,
        primary={
            "FILTER": "F150W2",
            "PUPIL": "F162M",
            "EFFEXPTM": 12.5,
            "XPOSURE": 99.0,
        },
    )
    obs = field_observation_from_exposure_file(path)
    assert obs.band_name == "F150W2-F162M" and obs.exposure_time_s == 12.5
    override = field_observation_from_exposure_file(
        path, band_name="custom", exposure_time_s=7.0
    )
    assert override.band_name == "custom" and override.exposure_time_s == 7.0
    bare = _write(
        tmp_path / "bare.fits",
        image,
        err,
        dq,
        primary={"FILTER": None, "EFFEXPTM": None},
    )
    none = field_observation_from_exposure_file(bare)
    assert none.band_name is None and none.exposure_time_s is None


@pytest.mark.parametrize(
    "unit, semantics",
    [
        ("MJy/sr", "surface_brightness"),
        ("MJy sr-1", "surface_brightness"),
        ("mJy / arcsec2", "surface_brightness"),
        ("erg/s/cm2/arcsec^-2", "surface_brightness"),
        ("DN/s", None),
        ("electrons", None),
        ("Jy/beam", None),
        ("MJy", None),
    ],
)
def test_value_semantics_follows_the_unit_only_for_per_solid_angle(
    tmp_path: Path, unit, semantics
) -> None:
    image, err, dq = _planes(13)
    obs = field_observation_from_exposure_file(
        _write(tmp_path / "u.fits", image, err, dq, sci_cards={"BUNIT": unit})
    )
    assert obs.value_unit == unit and obs.value_semantics == semantics


# ---------------------------------------------------------------- device, dtype, errors


def test_device_and_dtype_arguments(basic) -> None:
    path, _, err, _ = basic
    obs = field_observation_from_exposure_file(path, device="cpu", dtype=torch.float64)
    assert obs.image.dtype == obs.weight.dtype == obs.err.dtype == torch.float64
    assert obs.mask.dtype == torch.bool and obs.image.device.type == "cpu"
    torch.testing.assert_close(obs.weight, 1.0 / err.double() ** 2, rtol=1e-12, atol=0)
    with pytest.raises(TypeError, match="floating"):
        field_observation_from_exposure_file(path, dtype=torch.int32)
    moved = obs.to(dtype=torch.float32)
    assert moved.weight.dtype == torch.float32


def test_missing_extension_and_file(basic, tmp_path: Path) -> None:
    path, _, _, _ = basic
    with pytest.raises(ValueError, match="'ERRX'.*not found"):
        field_observation_from_exposure_file(path, error_hdu="ERRX")
    with pytest.raises(ValueError, match="'SCIX'.*not found"):
        field_observation_from_exposure_file(path, science_hdu="SCIX")
    with pytest.raises(FileNotFoundError):
        field_observation_from_exposure_file(tmp_path / "absent.fits")


@pytest.mark.parametrize("plane", ["err", "dq"])
def test_shape_mismatch_between_planes(tmp_path: Path, plane: str) -> None:
    image, err, dq = _planes(14)
    if plane == "err":
        err = err[:, :-1]
    else:
        dq = dq[:-1]
    path = _write(tmp_path / "bad.fits", image, err, dq)
    with pytest.raises(ValueError, match="shape"):
        field_observation_from_exposure_file(path)


def test_variance_plane_shape_mismatch(tmp_path: Path) -> None:
    image, err, dq = _planes(15)
    path = _write(
        tmp_path / "bv.fits",
        image,
        err,
        dq,
        extra=[_hdu(torch.rand(H, W - 2), "VAR_RNOISE", 1)],
    )
    with pytest.raises(ValueError, match="VAR_RNOISE.*shape"):
        field_observation_from_exposure_file(path, read_variance_planes=True)


def test_float_dq_plane_is_rejected(tmp_path: Path) -> None:
    image, err, _ = _planes(16)
    hdus = [
        torchfits.TensorHDU(data=torch.zeros(1), header={}),
        _hdu(image, "SCI", 1, **_wcs_header()),
        _hdu(err, "ERR", 1),
        _hdu(torch.zeros(H, W), "DQ", 1),
    ]
    path = str(tmp_path / "fdq.fits")
    torchfits.HDUList(hdus).write(path)
    with pytest.raises(TypeError, match="integer"):
        field_observation_from_exposure_file(path)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sky_box": (150.0, 2.0)},
        {"sky_box": (150.0, 2.0, 0.0)},
        {"sky_box": (150.0, 91.0, 1.0)},
        {"sky_box": (float("nan"), 2.0, 1.0)},
        {"min_valid_pixels": 0},
        {"extver": 0},
        {"exposure_time_s": -1.0},
    ],
)
def test_invalid_arguments_raise(basic, kwargs) -> None:
    path, _, _, _ = basic
    with pytest.raises(ValueError):
        field_observation_from_exposure_file(path, **kwargs)


def test_exported_through_lazy_facade() -> None:
    from torchsky.map.field.exposure_io import __all__ as exported

    assert sorted(exported) == [
        "ExposureFileData",
        "field_observation_from_exposure_file",
    ]


# ------------------------------------------------- quality plane, metadata, band names


def test_default_call_still_returns_a_bare_observation_without_instrument_cards(
    tmp_path: Path,
) -> None:
    image, err, dq = _planes(20)
    path = _write(
        tmp_path / "x_cal.fits",
        image,
        err,
        dq,
        primary={"INSTRUME": "NIRCAM", "DETECTOR": "NRCB1"},
        extra=[_hdu(torch.ones(H, W), "AREA", 1)],
    )
    obs = field_observation_from_exposure_file(path)
    assert isinstance(obs, FieldObservation)
    assert "DETECTOR" not in obs.header and "INSTRUME" not in obs.header


def test_with_extras_returns_dq_plane_instrument_names_and_area(tmp_path: Path) -> None:
    image, err, dq = _planes(21)
    dq[7, 9] = 2
    dq[2, 3] = 1
    area = 1.0 + 0.01 * torch.arange(H * W, dtype=torch.float32).reshape(H, W)
    path = _write(
        tmp_path / "x_cal.fits",
        image,
        err,
        dq,
        primary={"INSTRUME": "NIRCAM", "DETECTOR": "NRCB1"},
        extra=[_hdu(area, "AREA", 1)],
    )
    result = field_observation_from_exposure_file(path, with_extras=True)
    assert isinstance(result, ExposureFileData)
    assert torch.equal(result.quality, dq)
    assert result.quality.dtype == torch.int64
    assert result.metadata["INSTRUME"] == "NIRCAM"
    assert result.metadata["DETECTOR"] == "NRCB1"
    torch.testing.assert_close(result.metadata["AREA"], area)
    plain = field_observation_from_exposure_file(path)
    assert torch.equal(result.observation.image, plain.image)
    assert torch.equal(result.observation.mask, plain.mask)


def test_with_extras_follows_the_sky_box_cut(tmp_path: Path) -> None:
    image, err, dq = _planes(22)
    area = torch.arange(H * W, dtype=torch.float32).reshape(H, W)
    dq = torch.arange(H * W, dtype=torch.int64).reshape(H, W) % 8
    path = _write(
        tmp_path / "x_cal.fits", image, err, dq, extra=[_hdu(area, "AREA", 1)]
    )
    box = (CRVAL[0], CRVAL[1], 0.6)
    result = field_observation_from_exposure_file(
        path, sky_box=box, with_extras=True, dq_policy=DQPolicy(reject_bits=0)
    )
    obs = result.observation
    x0, y0 = int(obs.detector_x0_pix), int(obs.detector_y0_pix)
    h, w = obs.image.shape
    assert (x0, y0) != (0, 0) and result.quality.shape == (h, w)
    assert torch.equal(result.quality, dq[y0 : y0 + h, x0 : x0 + w])
    assert torch.equal(result.metadata["AREA"], area[y0 : y0 + h, x0 : x0 + w])


def test_with_extras_without_area_or_instrument_cards_omits_those_keys(
    tmp_path: Path,
) -> None:
    image, err, _dq = _planes(23)
    path = _write(tmp_path / "x_flt.fits", image, err, None)
    result = field_observation_from_exposure_file(
        path, quality_hdu=None, with_extras=True
    )
    assert result.quality is None
    assert result.metadata == {}
    assert field_observation_from_exposure_file(
        path, quality_hdu=None, with_extras=True
    )


def test_empty_cut_with_extras_is_still_none(basic) -> None:
    path, _, _, _ = basic
    assert (
        field_observation_from_exposure_file(
            path, sky_box=(151.0, 2.2, 0.1), with_extras=True
        )
        is None
    )


def test_zero_fill_can_be_switched_off(tmp_path: Path) -> None:
    image, err, dq = _planes(24)
    dq[6, 8] = 1
    image[12, 13] = float("nan")
    path = _write(tmp_path / "x_cal.fits", image, err, dq)
    filled = field_observation_from_exposure_file(path)
    kept = field_observation_from_exposure_file(path, zero_fill_unusable=False)
    assert float(filled.image[6, 8]) == 0.0 and float(filled.image[12, 13]) == 0.0
    assert float(kept.image[6, 8]) == pytest.approx(float(image[6, 8]))
    assert bool(torch.isnan(kept.image[12, 13]))
    assert torch.equal(kept.mask, filled.mask) and torch.equal(
        kept.weight, filled.weight
    )
    assert float(kept.weight[6, 8]) == 0.0 and not bool(kept.mask[6, 8])
    usable = filled.mask
    assert torch.equal(kept.image[usable], filled.image[usable])


@pytest.mark.parametrize(
    "cards, expected",
    [
        ({"FILTER1": "F814W", "FILTER2": "CLEAR2L"}, "F814W"),
        ({"FILTER1": "CLEAR1L", "FILTER2": "F606W"}, "F606W"),
        ({"FILTER1": "CLEAR1L", "FILTER2": "CLEAR2L"}, None),
        ({"FILTER1": "F475W", "FILTER2": "FR459M"}, "F475W-FR459M"),
        ({"FILTER1": "clear1l", "FILTER2": " f555w "}, "f555w"),
        ({"FILTER": "F160W"}, "F160W"),
        ({"FILTER": "F160W", "FILTER1": "F814W", "FILTER2": "CLEAR2L"}, "F160W"),
    ],
)
def test_hst_band_names_from_filter_wheels(
    tmp_path: Path, cards: dict, expected: str | None
) -> None:
    image, err, dq = _planes(25)
    primary = {"FILTER": None, "PUPIL": None, **cards}
    path = _write(tmp_path / "x_flc.fits", image, err, dq, primary=primary)
    assert field_observation_from_exposure_file(path).band_name == expected


def test_value_semantics_typo_fails_before_the_file_is_read(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="value_semantics"):
        field_observation_from_exposure_file(
            tmp_path / "missing.fits", value_semantics="surface_brigthness"
        )
    image, err, dq = _planes(26)
    path = _write(tmp_path / "x_flt.fits", image, err, dq, sci_cards={"BUNIT": "DN/s"})
    obs = field_observation_from_exposure_file(
        path, value_semantics="count_rate_per_pixel"
    )
    assert obs.value_semantics == "count_rate_per_pixel"
    assert field_observation_from_exposure_file(path).value_semantics is None
