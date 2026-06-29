from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest
import torch
from astropy.io import fits

import torchsky
import torchsky.map.image_plane as image_plane_mod
import torchsky.map.operators.resampling as resampling_mod
import torchsky.map.windowing as windowing_mod
from torchsky.map import (
    ImageObservation,
    coadd_image_plane,
    coadd_image_plane_cube,
    coadd_image_plane_multiband,
    evaluate_map_product,
    plan_map_product,
    plan_image_plane_cube_target,
    plan_image_plane_cube_target_from_header,
    plan_image_plane_target,
    stack_map_product,
)
from torchsky.map.operators import (
    pixel_area_sr_map,
    representative_pixel_area_sr,
    validate_image_plane_2d,
)
from torchsky.spectral import SpectralAxis

pytest.importorskip("torchsky.cpp")


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


def _astropy_sip_fixture() -> Path:
    import astropy

    return (
        Path(astropy.__file__).resolve().parent
        / "nddata"
        / "tests"
        / "data"
        / "sip-wcs.fits"
    )


def _clean_header(header: dict) -> fits.Header:
    out = fits.Header()
    for key, value in header.items():
        if str(key).endswith("_COMMENT"):
            continue
        out[key] = value
    return out


def _write_real_sip_mef(tmp_path: Path, *, shifted: bool) -> Path:
    image, header = torchsky.read(str(_astropy_sip_fixture()), return_header=True)
    data = image.to(dtype=torch.float32).cpu().numpy()
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

    weight1 = torch.ones_like(image, dtype=torch.float32).cpu().numpy()
    weight2 = torch.ones_like(image, dtype=torch.float32).cpu().numpy()
    weight1[:, : image.shape[1] // 3] = 2.0
    weight2[:, (2 * image.shape[1]) // 3 :] = 3.0
    mask1 = torch.ones_like(image, dtype=torch.float32).cpu().numpy()
    mask2 = torch.ones_like(image, dtype=torch.float32).cpu().numpy()
    mask1[:, (2 * image.shape[1]) // 3 :] = 0.0
    mask2[:, : image.shape[1] // 3] = 0.0

    path = tmp_path / ("real_sip_shifted_mef.fits" if shifted else "real_sip_mef.fits")
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(data=data, header=sci1, name="SCI"),
            fits.ImageHDU(data=weight1, header=err1, name="ERR"),
            fits.ImageHDU(data=mask1, header=dq1, name="DQ"),
            fits.ImageHDU(data=data, header=sci2, name="SCI"),
            fits.ImageHDU(data=weight2, header=err2, name="ERR"),
            fits.ImageHDU(data=mask2, header=dq2, name="DQ"),
        ]
    ).writeto(path, overwrite=True)
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


def _tpv_header(
    size: int,
    *,
    crval1: float,
    crval2: float,
    rotation_deg: float = 0.0,
    scale_deg: float = 0.01,
) -> fits.Header:
    header = fits.Header()
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
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(data=image.cpu().numpy(), header=sci1, name="SCI"),
            fits.ImageHDU(data=weight1.cpu().numpy(), header=err1, name="ERR"),
            fits.ImageHDU(data=mask1.cpu().numpy(), header=dq1, name="DQ"),
            fits.ImageHDU(data=image.cpu().numpy(), header=sci2, name="SCI"),
            fits.ImageHDU(data=weight2.cpu().numpy(), header=err2, name="ERR"),
            fits.ImageHDU(data=mask2.cpu().numpy(), header=dq2, name="DQ"),
        ]
    ).writeto(path, overwrite=True)
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
    torchsky.write(str(path_a), data_a, header=header, overwrite=True)
    torchsky.write(str(path_b), data_b, header=header, overwrite=True)

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
    torchsky.write(str(path_a), data_a, header=header, overwrite=True)
    torchsky.write(str(path_b), data_b, header=header, overwrite=True)
    obs = [ImageObservation(str(path_a)), ImageObservation(str(path_b))]
    full = coadd_image_plane(obs)
    tiled = coadd_image_plane(obs, spatial_tile_hw=(8, 8))
    torch.testing.assert_close(tiled.data, full.data, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(tiled.weight, full.weight, atol=1e-4, rtol=1e-4)
    assert full.overlap_count is not None and tiled.overlap_count is not None
    torch.testing.assert_close(tiled.overlap_count, full.overlap_count)


def test_cli_map_coadd_writes_output(tmp_path, monkeypatch) -> None:
    from torchsky.cli import main

    path_a = tmp_path / "cli_a.fits"
    path_b = tmp_path / "cli_b.fits"
    header = _simple_header()
    data_a = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    data_b = data_a + 2.0
    torchsky.write(str(path_a), data_a, header=header, overwrite=True)
    torchsky.write(str(path_b), data_b, header=header, overwrite=True)
    out = tmp_path / "coadd_cli.fits"
    monkeypatch.setattr(
        sys,
        "argv",
        ["torchsky", "map", str(out), str(path_a), str(path_b), "--no-auto-tile"],
    )
    assert main() == 0
    got, _ = torchsky.read(str(out), return_header=True)
    torch.testing.assert_close(got, (data_a + data_b) / 2.0, atol=1e-4, rtol=1e-4)


def test_coadd_image_plane_mean_stacking_averages_observations_equally(
    tmp_path,
) -> None:
    path_a = tmp_path / "mean_a.fits"
    path_b = tmp_path / "mean_b.fits"
    header = _simple_header()
    data_a = torch.full((8, 8), 1.0, dtype=torch.float32)
    data_b = torch.full((8, 8), 9.0, dtype=torch.float32)
    torchsky.write(str(path_a), data_a, header=header, overwrite=True)
    torchsky.write(str(path_b), data_b, header=header, overwrite=True)

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
    torchsky.write(str(path_a), data_a, header=header, overwrite=True)
    torchsky.write(str(path_b), data_b, header=header, overwrite=True)
    torchsky.write(str(path_c), data_c, header=header, overwrite=True)

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


def test_stack_map_product_passes_mean_stacking_to_single_band_coadd(tmp_path) -> None:
    path_a = tmp_path / "product_mean_a.fits"
    path_b = tmp_path / "product_mean_b.fits"
    header = _simple_header()
    data_a = torch.full((8, 8), 2.0, dtype=torch.float32)
    data_b = torch.full((8, 8), 10.0, dtype=torch.float32)
    torchsky.write(str(path_a), data_a, header=header, overwrite=True)
    torchsky.write(str(path_b), data_b, header=header, overwrite=True)

    observations = [
        ImageObservation(str(path_a), weight=1.0),
        ImageObservation(str(path_b), weight=5.0),
    ]
    product = plan_map_product(observations)

    result = stack_map_product(
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


def test_coadd_image_plane_cube_rejects_median_stacking(tmp_path) -> None:
    path = tmp_path / "cube_median.fits"
    header = _simple_header()
    data = torch.ones((8, 8), dtype=torch.float32)
    torchsky.write(str(path), data, header=header, overwrite=True)

    observations = [ImageObservation(str(path), wavelength=5000.0)]
    spectral_axis = SpectralAxis(
        values=torch.tensor([5000.0], dtype=torch.float64),
        unit="Angstrom",
        type="WAVE",
    )

    with pytest.raises(ValueError, match="median"):
        coadd_image_plane_cube(
            observations,
            spectral_axis=spectral_axis,
            stacking_method="median",
        )


def test_coadd_image_plane_respects_target_header_shape(tmp_path) -> None:
    path = tmp_path / "a.fits"
    header = _simple_header()
    data = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    torchsky.write(str(path), data, header=header, overwrite=True)

    target_header = dict(header)
    target_header["NAXIS1"] = 6
    target_header["NAXIS2"] = 5

    result = coadd_image_plane(
        [ImageObservation(str(path))],
        target_header=target_header,
    )

    assert tuple(result.data.shape) == (5, 6)
    assert tuple(result.weight.shape) == (5, 6)


def test_coadd_image_plane_ignores_non_finite_pixels(tmp_path) -> None:
    path_a = tmp_path / "a.fits"
    path_b = tmp_path / "b.fits"
    header = _simple_header()
    data_a = torch.ones((8, 8), dtype=torch.float32)
    data_b = 3.0 * torch.ones((8, 8), dtype=torch.float32)
    data_a[2, 3] = torch.nan
    torchsky.write(str(path_a), data_a, header=header, overwrite=True)
    torchsky.write(str(path_b), data_b, header=header, overwrite=True)

    result = coadd_image_plane(
        [ImageObservation(str(path_a)), ImageObservation(str(path_b))]
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
    torchsky.write(str(path_ref), truth, header=header, overwrite=True)
    torchsky.write(str(path_raw), raw, header=header, overwrite=True)

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
    torchsky.write(str(path), raw, header=header, overwrite=True)
    torchsky.write(str(weight_path), weight, header=header, overwrite=True)

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
    torchsky.write(str(path), data, header=header, overwrite=True)

    with pytest.raises(ValueError, match="2D images only"):
        coadd_image_plane([ImageObservation(str(path))])


def test_coadd_image_plane_rejects_unknown_resampling_kernel(tmp_path) -> None:
    path = tmp_path / "a.fits"
    header = _simple_header()
    data = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    torchsky.write(str(path), data, header=header, overwrite=True)

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
    torchsky.write(str(path), data, header=header, overwrite=True)

    with pytest.raises(ValueError, match="value_semantics"):
        coadd_image_plane(
            [ImageObservation(str(path))],
            value_semantics="counts",
        )


def test_coadd_image_plane_rejects_unknown_stacking_method(tmp_path) -> None:
    path = tmp_path / "stacking_unknown.fits"
    header = _simple_header()
    data = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    torchsky.write(str(path), data, header=header, overwrite=True)

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
    torchsky.write(str(path), data, header=header, overwrite=True)

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
    torchsky.write(str(path_a), data, header=header_a, overwrite=True)
    torchsky.write(str(path_b), data + 1.0, header=header_b, overwrite=True)

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
    torchsky.write(str(path_a), data, header=header_a, overwrite=True)
    torchsky.write(str(path_b), data + 1.0, header=header_b, overwrite=True)

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
    torchsky.write(str(path_a), data, header=header_a, overwrite=True)
    torchsky.write(str(path_b), data + 1.0, header=header_b, overwrite=True)

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
    torchsky.write(str(path), data, header=header, overwrite=True)

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
    torchsky.write(str(path), data, header=header, overwrite=True)

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
    torchsky.write(str(path), blob, header=ha, overwrite=True)
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
    torchsky.write(str(path), blob, header=ha, overwrite=True)
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
    torchsky.write(str(path), blob, header=ha, overwrite=True)
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
    torchsky.write(str(path), blob, header=ha, overwrite=True)
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
    torchsky.write(str(path), blob, header=ha, overwrite=True)
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
    torchsky.write(str(path), image, header=ha, overwrite=True)
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
    torchsky.write(str(path), image, header=ha, overwrite=True)
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
    torchsky.write(str(path), image, header=ha, overwrite=True)
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
    torchsky.write(str(path), torch.ones((ny, nx), dtype=torch.float32), header=ha)
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
    torchsky.write(str(path), torch.ones((ny, nx), dtype=torch.float32), header=ha)
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
    torchsky.write(str(pa), blob, header=ha, overwrite=True)
    torchsky.write(str(pb), blob * 0.95 + 0.02, header=hb, overwrite=True)
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
    torchsky.write(str(path_a), data, header=ha, overwrite=True)
    torchsky.write(str(path_b), data * 1.2, header=hb, overwrite=True)

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
    torchsky.write(
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


def test_coadd_image_plane_clamps_negative_resampled_weights_for_high_order_kernels(
    tmp_path: Path,
) -> None:
    benchmarks_dir = Path(__file__).resolve().parents[1] / "benchmarks"
    if str(benchmarks_dir) not in sys.path:
        sys.path.insert(0, str(benchmarks_dir))
    import bench_map_ground_truth as truth_bench

    observations, target_header, truth = truth_bench.make_case(
        tmp_path,
        case="tan_random10_large_union_rect_weighted_masked",
        size=1024,
        seed=1234,
        width=1024,
        height=768,
    )

    result = coadd_image_plane(
        [item.observation for item in observations],
        target_header=target_header,
        resampling_kernel="bicubic",
    )
    metrics = evaluate_map_product(
        result.data,
        truth,
        weight=result.weight,
        header=target_header,
        value_semantics="surface_brightness",
    )

    assert torch.isfinite(result.data).all()
    assert torch.isfinite(result.weight).all()
    assert bool((result.weight >= 0.0).all())
    assert math.isfinite(float(metrics["total_flux_relative_error"]))


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
    torchsky.write(str(source_path), flux_source, header=source_header, overwrite=True)

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
    torchsky.write(str(fine_source_path), image, header=fine_source, overwrite=True)
    torchsky.write(str(fine_weight_path), weight, header=fine_source, overwrite=True)

    coarse_source = dict(coarse_target)
    coarse_source_path = tmp_path / "coarse_source.fits"
    coarse_weight_path = tmp_path / "coarse_source_wht.fits"
    torchsky.write(str(coarse_source_path), image, header=coarse_source, overwrite=True)
    torchsky.write(
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
    torchsky.write(str(path_a), data, header=header_a, overwrite=True)
    torchsky.write(str(path_b), data + 1.0, header=header_b, overwrite=True)

    calls = 0
    original = image_plane_mod._prepare_target

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(image_plane_mod, "_prepare_target", counted)

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
    torchsky.write(str(path), data, header=source_header, overwrite=True)

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
    image, header = torchsky.read(str(path), return_header=True)
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
            image_plane_mod, "_resample_weighted_to_target", wrapped_resample
        )
        monkeypatch.setattr(
            windowing_mod, "_LARGE_TAN_WINDOWING_MIN_TARGET_PIXELS", 0
        )
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

    monkeypatch.setattr(image_plane_mod, "_window_large_tan_target", no_overlap_window)
    monkeypatch.setattr(
        image_plane_mod, "project_image_observation_geometry", fail_projection
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
    torchsky.write(str(path_a), data, header=header_a, overwrite=True)
    torchsky.write(str(path_b), data, header=header_b, overwrite=True)
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
    torchsky.write(str(path_a), data, header=header, overwrite=True)
    torchsky.write(str(path_b), data + 1.0, header=header, overwrite=True)

    calls = 0
    original = image_plane_mod._build_target_world_grid

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(image_plane_mod, "_build_target_world_grid", counted)

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

    torchsky.write(str(path_a), data_a, header=header, overwrite=True)
    torchsky.write(str(path_b), data_b, header=header, overwrite=True)
    torchsky.write(str(wht_a), weight_a, header=header, overwrite=True)
    torchsky.write(str(msk_b), mask_b, header=header, overwrite=True)

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
    torchsky.write(
        str(path),
        torch.ones((8, 8), dtype=torch.float32),
        header=header,
        overwrite=True,
    )
    torchsky.write(
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
    image, header = torchsky.read(str(real_path), return_header=True)
    image = image.to(dtype=torch.float32)

    mask_a = tmp_path / "mask_a.fits"
    mask_b = tmp_path / "mask_b.fits"

    mask_left = torch.zeros_like(image)
    mask_left[:, : image.shape[1] // 2] = 1.0
    mask_right = torch.zeros_like(image)
    mask_right[:, image.shape[1] // 2 :] = 1.0

    torchsky.write(str(mask_a), mask_left, overwrite=True)
    torchsky.write(str(mask_b), mask_right, overwrite=True)

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


def test_coadd_image_plane_supports_mef_science_extensions(tmp_path) -> None:
    mef_path = _write_real_sip_mef(tmp_path, shifted=False)
    image = torchsky.read(str(_astropy_sip_fixture())).to(dtype=torch.float32)

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
    image, header = torchsky.read(str(mef_path), hdu=4, return_header=True)
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

    torchsky.write(str(path_g1), data_g1, header=header, overwrite=True)
    torchsky.write(str(path_g2), data_g2, header=header, overwrite=True)
    torchsky.write(str(path_r1), data_r1, header=header, overwrite=True)
    torchsky.write(str(path_r2), data_r2, header=header, overwrite=True)

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
    torchsky.write(
        str(path),
        torch.ones((8, 8), dtype=torch.float32),
        header=header,
        overwrite=True,
    )

    with pytest.raises(ValueError, match="must define wavelength"):
        coadd_image_plane_multiband([ImageObservation(str(path))])


def test_coadd_image_plane_multiband_real_astropy_fixture(tmp_path) -> None:
    real_path = _astropy_sip_fixture()
    image = torchsky.read(str(real_path)).to(dtype=torch.float32)

    mask_a = tmp_path / "mask_a.fits"
    mask_b = tmp_path / "mask_b.fits"
    mask_left = torch.zeros_like(image)
    mask_left[:, : image.shape[1] // 2] = 1.0
    mask_right = torch.zeros_like(image)
    mask_right[:, image.shape[1] // 2 :] = 1.0
    torchsky.write(str(mask_a), mask_left, overwrite=True)
    torchsky.write(str(mask_b), mask_right, overwrite=True)

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

    torchsky.write(
        str(path_a),
        torch.ones((8, 8), dtype=torch.float32),
        header=header_a,
        overwrite=True,
    )
    torchsky.write(
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

    torchsky.write(
        str(path_g),
        torch.full((8, 8), 2.0, dtype=torch.float32),
        header=header_a,
        overwrite=True,
    )
    torchsky.write(
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
    torchsky.write(
        str(path_g),
        torch.ones((8, 8), dtype=torch.float32),
        header=header,
        overwrite=True,
    )
    torchsky.write(
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
    torchsky.write(str(path), image, header=header, overwrite=True)

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
    torchsky.write(str(path), image, header=header, overwrite=True)

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
    torchsky.write(str(path), image, header=header, overwrite=True)

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
    image = torchsky.read(str(real_path)).to(dtype=torch.float32)

    mask_a = tmp_path / "mask_a.fits"
    mask_b = tmp_path / "mask_b.fits"
    mask_left = torch.zeros_like(image)
    mask_left[:, : image.shape[1] // 2] = 1.0
    mask_right = torch.zeros_like(image)
    mask_right[:, image.shape[1] // 2 :] = 1.0
    torchsky.write(str(mask_a), mask_left, overwrite=True)
    torchsky.write(str(mask_b), mask_right, overwrite=True)

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
    image, header = torchsky.read(str(real_path), return_header=True)
    image = image.to(dtype=torch.float32)
    shifted_header = dict(header)
    shifted_header["CRVAL1"] = float(shifted_header["CRVAL1"]) + 0.01
    shifted_header["CRVAL2"] = float(shifted_header["CRVAL2"]) - 0.01
    shifted_path = tmp_path / "sip_shifted.fits"
    torchsky.write(str(shifted_path), image, header=shifted_header, overwrite=True)

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


def test_plan_image_plane_cube_target_from_header_uses_spectral_wcs(tmp_path) -> None:
    header = _simple_header()
    path = tmp_path / "img.fits"
    torchsky.write(
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

_STACKING_METHODS_DETERMINISTIC = ["mean", "weighted_mean", "median"]


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
    torchsky.write(str(path_a), data_a, header=header_a, overwrite=True)
    torchsky.write(str(path_b), data_b, header=header_b, overwrite=True)

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
