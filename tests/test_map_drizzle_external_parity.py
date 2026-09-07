"""Parity tests for optional Python ``drizzle`` regression path.

Compares native torchsky drizzle kernels against the low-level Python
``drizzle`` package when it is installed.

Requires: ``torchsky.cpp``, optional ``drizzle``.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

pytest.importorskip("torchsky.cpp")
pytest.importorskip("drizzle")

from astropy.wcs import WCS as AstropyWCS  # noqa: E402
from drizzle.resample import Drizzle as DrizzleResampler  # noqa: E402
from drizzle.utils import calc_pixmap, estimate_pixel_scale_ratio  # noqa: E402

import torchsky
from torchsky.map import ImageObservation, coadd_image_plane


def _header(
    *,
    nx: int = 16,
    ny: int = 16,
    crval1: float = 180.0,
    crval2: float = 0.0,
    rotation_deg: float = 0.0,
    scale: float = 1.0,
) -> dict[str, float | int | str]:
    theta = math.radians(rotation_deg)
    pix = 0.01 * float(scale)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    return {
        "NAXIS": 2,
        "NAXIS1": nx,
        "NAXIS2": ny,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 0.5 * (nx + 1),
        "CRPIX2": 0.5 * (ny + 1),
        "CRVAL1": crval1,
        "CRVAL2": crval2,
        "CD1_1": -pix * cos_t,
        "CD1_2": pix * sin_t,
        "CD2_1": pix * sin_t,
        "CD2_2": pix * cos_t,
    }


def _blob(ny: int = 16, nx: int = 16) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(ny, dtype=torch.float32),
        torch.arange(nx, dtype=torch.float32),
        indexing="ij",
    )
    return (
        0.7 * torch.exp(-((xx - 5.0) ** 2 + (yy - 6.0) ** 2) / 6.0)
        + 1.1 * torch.exp(-((xx - 10.5) ** 2 + (yy - 9.0) ** 2) / 9.0)
        + 0.02 * xx
    )


def _header_for_drizzle(header: dict) -> dict:
    return {
        key: value
        for key, value in header.items()
        if not str(key).endswith("_COMMENT") and key not in {"", "COMMENT", "HISTORY"}
    }


def _stsci_drizzle_reference(
    image: torch.Tensor,
    source_weight: torch.Tensor,
    *,
    header_dict_obs: dict,
    target_header: dict,
    resampling_kernel: str,
    value_semantics: str,
    drizzle_pixfrac: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    kernel = "point" if resampling_kernel == "drizzle_point" else "square"
    target_shape = (int(target_header["NAXIS2"]), int(target_header["NAXIS1"]))
    target_wcs = AstropyWCS(_header_for_drizzle(target_header))
    target_wcs.array_shape = target_shape
    source_shape = tuple(image.shape)
    source_wcs = AstropyWCS(_header_for_drizzle(header_dict_obs))
    source_wcs.array_shape = source_shape
    pixmap = calc_pixmap(source_wcs, target_wcs, shape=source_shape)
    pscale_ratio = estimate_pixel_scale_ratio(
        source_wcs,
        target_wcs,
        refpix_from=source_wcs.wcs.crpix,
        refpix_to=target_wcs.wcs.crpix,
    )
    iscale = (
        1.0 if value_semantics == "surface_brightness" else float(pscale_ratio) ** 2
    )
    # Drizzle 2.2's tight-canvas scanner omits square drops whose centers lie
    # outside the canvas. Give it a full transformed drop margin, then crop.
    # This compares pixel overlap, not that scanner's boundary truncation.
    padding = 0
    if kernel == "square":
        mapping = torch.as_tensor(pixmap)
        extent = sum(float(torch.diff(mapping, dim=axis).abs().max()) for axis in (0, 1))
        padding = 1 + math.ceil(float(drizzle_pixfrac) * extent)
    drizzle = DrizzleResampler(
        kernel=kernel,
        fillval=0.0,
        out_shape=tuple(size + 2 * padding for size in target_shape),
        disable_ctx=True,
    )
    drizzle.add_image(
        torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        .cpu()
        .numpy()
        .astype("float32"),
        exptime=1.0,
        pixmap=pixmap + padding,
        weight_map=torch.nan_to_num(source_weight, nan=0.0, posinf=0.0, neginf=0.0)
        .cpu()
        .numpy()
        .astype("float32"),
        iscale=iscale,
        pixel_scale_ratio=pscale_ratio,
        pixfrac=float(drizzle_pixfrac),
        in_units="cps",
    )
    out_img = torch.as_tensor(drizzle.out_img, dtype=torch.float32, device=image.device)
    out_wht = torch.as_tensor(drizzle.out_wht, dtype=torch.float32, device=image.device)
    if padding:
        out_img = out_img[padding:-padding, padding:-padding]
        out_wht = out_wht[padding:-padding, padding:-padding]
    if value_semantics == "flux":
        out_wht = out_wht / float(iscale * iscale)
    out_img = torch.where(out_wht > 0.0, out_img, torch.zeros_like(out_img))
    numer = out_img * out_wht
    data = torch.where(out_wht > 0.0, numer / out_wht, torch.zeros_like(numer))
    return numer, out_wht, data


def _assert_native_matches_external(
    native,
    *,
    reference_weight: torch.Tensor,
    reference_data: torch.Tensor,
    atol: float,
    rtol: float,
) -> None:
    assert torch.isfinite(native.data).all()
    assert torch.isfinite(native.weight).all()
    torch.testing.assert_close(native.weight, reference_weight, atol=atol, rtol=rtol)
    torch.testing.assert_close(native.data, reference_data, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    ("target_header", "atol"),
    [
        (_header(crval1=180.03), 3e-2),
        (_header(rotation_deg=11.0), 4e-2),
        (_header(scale=1.08), 4e-2),
    ],
)
@pytest.mark.parametrize("pixfrac", [0.7, 1.0])
@pytest.mark.parametrize("value_semantics", ["surface_brightness", "flux"])
@pytest.mark.parametrize("kernel", ["drizzle_point", "drizzle_square"])
def test_external_drizzle_parity_across_affine_cases(
    tmp_path: Path,
    target_header: dict[str, float | int | str],
    atol: float,
    pixfrac: float,
    value_semantics: str,
    kernel: str,
) -> None:
    source_header = _header()
    image = _blob()
    path = tmp_path / f"{kernel}_{value_semantics}_{pixfrac:.1f}.fits"
    torchsky.write(str(path), image, header=source_header, overwrite=True)

    native = coadd_image_plane(
        [ImageObservation(str(path))],
        target_header=dict(target_header),
        resampling_kernel=kernel,
        value_semantics=value_semantics,
        drizzle_pixfrac=pixfrac,
        stacking_method="weighted_mean",
    )
    _, reference_weight, reference_data = _stsci_drizzle_reference(
        image,
        torch.ones_like(image),
        header_dict_obs=source_header,
        target_header=dict(target_header),
        resampling_kernel=kernel,
        value_semantics=value_semantics,
        drizzle_pixfrac=pixfrac,
    )

    _assert_native_matches_external(
        native,
        reference_weight=reference_weight,
        reference_data=reference_data,
        atol=atol,
        rtol=atol,
    )


@pytest.mark.parametrize("kernel", ["drizzle_point", "drizzle_square"])
def test_external_drizzle_parity_respects_weight_and_mask_maps(
    tmp_path: Path,
    kernel: str,
) -> None:
    source_header = _header()
    target_header = _header(crval1=180.02, rotation_deg=7.0)
    image = _blob()
    weight_map = torch.ones_like(image)
    weight_map[:, :3] = 0.0
    weight_map[12:, :] *= 0.25
    mask = torch.ones_like(image)
    mask[4:8, 8:12] = 0.0

    image_path = tmp_path / f"{kernel}_image.fits"
    weight_path = tmp_path / f"{kernel}_weight.fits"
    mask_path = tmp_path / f"{kernel}_mask.fits"
    torchsky.write(str(image_path), image, header=source_header, overwrite=True)
    torchsky.write(str(weight_path), weight_map, header=source_header, overwrite=True)
    torchsky.write(str(mask_path), mask, header=source_header, overwrite=True)

    obs = ImageObservation(
        str(image_path),
        weight=2.0,
        weight_path=str(weight_path),
        mask_path=str(mask_path),
    )
    native = coadd_image_plane(
        [obs],
        target_header=dict(target_header),
        resampling_kernel=kernel,
        value_semantics="surface_brightness",
        drizzle_pixfrac=0.9,
        stacking_method="weighted_mean",
    )
    source_weight = (
        torch.where(mask > 0.0, weight_map, torch.zeros_like(weight_map)) * 2.0
    )
    _, reference_weight, reference_data = _stsci_drizzle_reference(
        image,
        source_weight,
        header_dict_obs=source_header,
        target_header=dict(target_header),
        resampling_kernel=kernel,
        value_semantics="surface_brightness",
        drizzle_pixfrac=0.9,
    )

    _assert_native_matches_external(
        native,
        reference_weight=reference_weight,
        reference_data=reference_data,
        atol=4e-2,
        rtol=4e-2,
    )


def test_external_drizzle_parity_square_support_matches_weighted_uniform_field(
    tmp_path: Path,
) -> None:
    source_header = _header()
    target_header = _header(crval1=180.015, rotation_deg=13.0, scale=0.95)
    image = torch.ones((16, 16), dtype=torch.float32)
    path = tmp_path / "uniform.fits"
    torchsky.write(str(path), image, header=source_header, overwrite=True)

    native = coadd_image_plane(
        [ImageObservation(str(path))],
        target_header=dict(target_header),
        resampling_kernel="drizzle_square",
        value_semantics="surface_brightness",
        drizzle_pixfrac=1.0,
        stacking_method="weighted_mean",
    )
    _, reference_weight, reference_data = _stsci_drizzle_reference(
        image,
        torch.ones_like(image),
        header_dict_obs=source_header,
        target_header=dict(target_header),
        resampling_kernel="drizzle_square",
        value_semantics="surface_brightness",
        drizzle_pixfrac=1.0,
    )

    _assert_native_matches_external(
        native,
        reference_weight=reference_weight,
        reference_data=reference_data,
        atol=2e-2,
        rtol=2e-2,
    )
