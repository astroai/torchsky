"""Parity tests for native conservative overlap against optional ``reproject_exact``."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch
from astropy.io import fits

import torchsky
from torchsky.map import ImageObservation, coadd_image_plane

pytest.importorskip("torchsky.cpp")


def _reproject_installed() -> bool:
    return importlib.util.find_spec("reproject") is not None


pytestmark = pytest.mark.skipif(
    not _reproject_installed(),
    reason="optional ``reproject`` package not installed",
)


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


def _as_fits_header(header: dict) -> fits.Header:
    out = fits.Header()
    for key, value in header.items():
        if str(key).endswith("_COMMENT"):
            continue
        out[key] = value
    return out


def _external_reference(
    image: torch.Tensor,
    source_weight: torch.Tensor,
    *,
    header_dict_obs: dict,
    target_header: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    from reproject import reproject_exact

    target_shape = (int(target_header["NAXIS2"]), int(target_header["NAXIS1"]))
    numer, _ = reproject_exact(
        (
            (torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0) * source_weight)
            .cpu()
            .numpy()
            .astype("float32"),
            _as_fits_header(header_dict_obs),
        ),
        _as_fits_header(target_header),
        shape_out=target_shape,
        return_footprint=True,
    )
    denom, _ = reproject_exact(
        (
            source_weight.cpu().numpy().astype("float32"),
            _as_fits_header(header_dict_obs),
        ),
        _as_fits_header(target_header),
        shape_out=target_shape,
        return_footprint=True,
    )
    numer_t = torch.from_numpy(numer).to(dtype=torch.float32)
    denom_t = torch.from_numpy(denom).to(dtype=torch.float32)
    numer_t = torch.nan_to_num(numer_t, nan=0.0, posinf=0.0, neginf=0.0)
    denom_t = torch.nan_to_num(denom_t, nan=0.0, posinf=0.0, neginf=0.0)
    data_t = torch.where(denom_t > 0.0, numer_t / denom_t, torch.zeros_like(denom_t))
    return denom_t, data_t


@pytest.mark.parametrize(
    ("target_header", "atol"),
    [
        (_header(crval1=180.03), 1e-5),
        (_header(rotation_deg=11.0), 1e-5),
        (_header(scale=1.08), 8e-2),
    ],
)
def test_conservative_polygon_parity_matches_reproject_exact_data_and_support(
    tmp_path: Path,
    target_header: dict[str, float | int | str],
    atol: float,
) -> None:
    source_header = _header()
    image = _blob()
    path = tmp_path / "conservative_affine.fits"
    torchsky.write(str(path), image, header=source_header, overwrite=True)

    native = coadd_image_plane(
        [ImageObservation(str(path))],
        target_header=dict(target_header),
        resampling_kernel="conservative_polygon",
        value_semantics="surface_brightness",
        stacking_method="weighted_mean",
    )
    reference_weight, reference_data = _external_reference(
        image,
        torch.ones_like(image),
        header_dict_obs=source_header,
        target_header=dict(target_header),
    )

    torch.testing.assert_close(native.weight > 0.0, reference_weight > 0.0)
    torch.testing.assert_close(native.data, reference_data, atol=atol, rtol=atol)


def test_conservative_polygon_parity_respects_weight_and_mask_maps(
    tmp_path: Path,
) -> None:
    source_header = _header()
    target_header = _header(crval1=180.02, rotation_deg=7.0)
    image = _blob()
    weight_map = torch.ones_like(image)
    weight_map[:, :3] = 0.0
    weight_map[12:, :] *= 0.25
    mask = torch.ones_like(image)
    mask[4:8, 8:12] = 0.0

    image_path = tmp_path / "conservative_image.fits"
    weight_path = tmp_path / "conservative_weight.fits"
    mask_path = tmp_path / "conservative_mask.fits"
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
        resampling_kernel="conservative_polygon",
        value_semantics="surface_brightness",
        stacking_method="weighted_mean",
    )
    source_weight = (
        torch.where(mask > 0.0, weight_map, torch.zeros_like(weight_map)) * 2.0
    )
    reference_weight, reference_data = _external_reference(
        image,
        source_weight,
        header_dict_obs=source_header,
        target_header=dict(target_header),
    )

    torch.testing.assert_close(native.weight > 0.0, reference_weight > 0.0)
    torch.testing.assert_close(native.data, reference_data, atol=4e-2, rtol=4e-2)
