from __future__ import annotations

from pathlib import Path

import pytest
import torch
from astropy.io import fits

import torchsky
import torchsky.map.image_plane as image_plane_mod
from torchsky.map import (
    ImageObservation,
    coadd_image_plane,
    coadd_image_plane_cube,
    coadd_image_plane_multiband,
    plan_image_plane_cube_target,
    plan_image_plane_cube_target_from_header,
    plan_image_plane_target,
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
        Path(astropy.__file__).resolve().parent / "nddata" / "tests" / "data" / "sip-wcs.fits"
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


def _rotated_cd(scale_deg: float, rotation_deg: float) -> tuple[float, float, float, float]:
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
    original = image_plane_mod.F.grid_sample

    def recorded(*args, **kwargs):
        calls.append(kwargs["mode"])
        return original(*args, **kwargs)

    monkeypatch.setattr(image_plane_mod.F, "grid_sample", recorded)

    result = coadd_image_plane(
        [ImageObservation(str(path_a)), ImageObservation(str(path_b))],
        resampling_kernel=kernel,
    )

    assert result.data.shape == (8, 8)
    assert calls
    assert set(calls) == {kernel}


@pytest.mark.parametrize(
    "kernel",
    ["lanczos3", "lanczos5", "mitchell_netravali", "catmull_rom"],
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


def test_coadd_image_plane_flux_semantics_improves_flux_conservation(tmp_path) -> None:
    source_path = tmp_path / "flux_source.fits"
    target_header = _simple_header()
    source_header = dict(target_header)
    source_header["CDELT1"] = -0.02
    source_header["CDELT2"] = 0.02

    source_wcs = torchsky.WCS(source_header)
    target_wcs = torchsky.WCS(target_header)
    source_area = image_plane_mod._pixel_area_sr_map(
        source_wcs,
        (8, 8),
        device=torch.device("cpu"),
    )
    target_area = image_plane_mod._pixel_area_sr_map(
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

    surface_flux_err = torch.abs(surface_result.data - expected_target_flux).mean().item()
    flux_flux_err = torch.abs(flux_result.data - expected_target_flux).mean().item()
    assert flux_flux_err < surface_flux_err


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
    original = image_plane_mod._build_target_world_grid

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(image_plane_mod, "_build_target_world_grid", counted)

    result = coadd_image_plane(
        [ImageObservation(str(path_a)), ImageObservation(str(path_b))]
    )

    assert calls == 1
    assert result.data.shape == (8, 8)


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
            ImageObservation(str(mef_path), hdu=1, weight_path=str(mef_path), weight_hdu=2, mask_path=str(mef_path), mask_hdu=3),
            ImageObservation(str(mef_path), hdu=4, weight_path=str(mef_path), weight_hdu=5, mask_path=str(mef_path), mask_hdu=6),
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
            ImageObservation(str(mef_path), hdu=1, weight_path=str(mef_path), weight_hdu=2, mask_path=str(mef_path), mask_hdu=3),
            ImageObservation(str(mef_path), hdu=4, weight_path=str(mef_path), weight_hdu=5, mask_path=str(mef_path), mask_hdu=6),
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
            ImageObservation(str(mef_path), hdu=1, weight_path=str(mef_path), weight_hdu=2, mask_path=str(mef_path), mask_hdu=3),
            ImageObservation(str(mef_path), hdu=4, weight_path=str(mef_path), weight_hdu=5, mask_path=str(mef_path), mask_hdu=6),
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
    assert all(moc is not None and moc.area_sq_deg > 0.0 for moc in result.coverage_mocs)
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
    torch.testing.assert_close(result.data[0, :, :50], image[:, :50], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(result.data[0, :, 50:], torch.zeros_like(image[:, 50:]))
    torch.testing.assert_close(result.data[1, :, :50], torch.zeros_like(image[:, :50]))
    torch.testing.assert_close(result.data[1, :, 50:], image[:, 50:], atol=1e-4, rtol=1e-4)
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
    torch.testing.assert_close(result.weight[0, :, 50:], torch.zeros_like(image[:, 50:]))
    torch.testing.assert_close(result.weight[1, :, :50], torch.zeros_like(image[:, :50]))
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

    expected_weights = torch.tensor([0.25, 0.50, 0.25], dtype=torch.float32).view(3, 1, 1)
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


def test_coadd_image_plane_cube_reprojects_gaussian_transmission_filters(tmp_path) -> None:
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
    torch.testing.assert_close(result.data[0, :, :50], image[:, :50], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(result.data[0, :, 50:], torch.zeros_like(image[:, 50:]))
    torch.testing.assert_close(result.data[1, :, :50], torch.zeros_like(image[:, :50]))
    torch.testing.assert_close(result.data[1, :, 50:], image[:, 50:], atol=1e-4, rtol=1e-4)
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
