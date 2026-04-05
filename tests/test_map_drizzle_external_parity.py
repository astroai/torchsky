"""Smoke tests for optional Python ``drizzle`` regression path.

``TORCHSKY_USE_EXTERNAL_DRIZZLE=1`` routes drizzle kernels through the legacy
``drizzle`` package (Astropy WCS + ``calc_pixmap``). Native torch forward-drizzle
uses per-pixel ``pixel_to_world`` / ``world_to_pixel``; **pixel-level agreement
with the external package is benchmark territory**, not a hard CI contract.

Requires: ``torchsky.cpp``, optional ``drizzle``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

import torchsky
from torchsky.map import ImageObservation, coadd_image_plane

pytest.importorskip("torchsky.cpp")


def _drizzle_installed() -> bool:
    return importlib.util.find_spec("drizzle") is not None


pytestmark = pytest.mark.skipif(
    not _drizzle_installed(),
    reason="optional ``drizzle`` package not installed",
)


def _tiny_header(*, crval1_offset: float = 0.0) -> dict:
    return {
        "NAXIS": 2,
        "NAXIS1": 16,
        "NAXIS2": 16,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 8.0,
        "CRPIX2": 8.0,
        "CRVAL1": 180.0 + crval1_offset,
        "CRVAL2": 0.0,
        "CDELT1": -0.01,
        "CDELT2": 0.01,
    }


@pytest.mark.parametrize("value_semantics", ["surface_brightness", "flux"])
@pytest.mark.parametrize("kernel", ["drizzle_point", "drizzle_square"])
def test_external_drizzle_coadd_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    value_semantics: str,
    kernel: str,
) -> None:
    monkeypatch.setenv("TORCHSKY_USE_EXTERNAL_DRIZZLE", "1")
    path_a = tmp_path / "a.fits"
    path_b = tmp_path / "b.fits"
    ha = _tiny_header()
    hb = _tiny_header(crval1_offset=0.03)
    yy, xx = torch.meshgrid(
        torch.arange(16, dtype=torch.float32),
        torch.arange(16, dtype=torch.float32),
        indexing="ij",
    )
    blob = torch.exp(-((xx - 8.0) ** 2 + (yy - 8.0) ** 2) / 8.0)
    torchsky.write(str(path_a), blob, header=ha, overwrite=True)
    torchsky.write(str(path_b), blob * 1.1 + 0.05, header=hb, overwrite=True)

    result = coadd_image_plane(
        [ImageObservation(str(path_a)), ImageObservation(str(path_b))],
        resampling_kernel=kernel,
        value_semantics=value_semantics,
        drizzle_pixfrac=0.9,
        stacking_method="weighted_mean",
    )
    assert result.data.shape == (16, 16)
    assert torch.isfinite(result.data).all()
    assert torch.isfinite(result.weight).all()
    assert int((result.weight > 0).sum().item()) > 10
