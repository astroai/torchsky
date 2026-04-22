"""Parity tests for optional Python ``drizzle`` regression path.

Compares native torchsky drizzle kernels against the low-level Python
``drizzle`` package when it is installed.

Requires: ``torchsky.cpp``, optional ``drizzle``.
"""

from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path

import pytest
import torch

import torchsky
from torchsky.map import ImageObservation, coadd_image_plane
from torchsky.map import image_plane as image_plane_mod

pytest.importorskip("torchsky.cpp")


def _drizzle_installed() -> bool:
    return importlib.util.find_spec("drizzle") is not None


pytestmark = pytest.mark.skipif(
    not _drizzle_installed(),
    reason="optional ``drizzle`` package not installed",
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


def _external_reference(
    image: torch.Tensor,
    source_weight: torch.Tensor,
    *,
    header_dict_obs: dict,
    target_header: dict,
    resampling_kernel: str,
    value_semantics: str,
    drizzle_pixfrac: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    old = os.environ.get("TORCHSKY_USE_EXTERNAL_DRIZZLE")
    os.environ["TORCHSKY_USE_EXTERNAL_DRIZZLE"] = "1"
    try:
        exact = image_plane_mod._maybe_external_drizzle_resample_weighted_to_target(
            image,
            source_weight,
            header_dict_obs=header_dict_obs,
            target_header=target_header,
            target_shape=(int(target_header["NAXIS2"]), int(target_header["NAXIS1"])),
            resampling_kernel=resampling_kernel,
            value_semantics=value_semantics,
            drizzle_pixfrac=drizzle_pixfrac,
        )
    finally:
        if old is None:
            os.environ.pop("TORCHSKY_USE_EXTERNAL_DRIZZLE", None)
        else:
            os.environ["TORCHSKY_USE_EXTERNAL_DRIZZLE"] = old
    assert exact is not None
    numer, weight = exact
    data = torch.where(weight > 0.0, numer / weight, torch.zeros_like(numer))
    return numer, weight, data


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
    _, reference_weight, reference_data = _external_reference(
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
    _, reference_weight, reference_data = _external_reference(
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
    _, reference_weight, reference_data = _external_reference(
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
