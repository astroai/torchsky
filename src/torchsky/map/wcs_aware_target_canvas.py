"""Target-canvas helpers for WCS-aware INR training and rendering.

This module adds an *optional* template-grid path without changing the existing
auto-union-canvas logic.  A 2D FITS image (e.g. a released UPDC i2d mosaic)
can define the common spatial WCS and shape used by the INR.  The science data
used for the loss still come from the native CRF/CAL exposures; the template
supplies geometry only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from astropy.io import fits
from astropy.wcs import WCS as AstropyWCS
from astropy.wcs.utils import proj_plane_pixel_scales

from ..wcs import WCS
from .wcs_aware_inr import UnionCanvas


def _select_2d_hdu(
    hdul: fits.HDUList,
    hdu_selector: str | int | None,
) -> tuple[int, fits.hdu.base.ExtensionHDU | fits.PrimaryHDU]:
    if hdu_selector is not None:
        key: str | int
        if isinstance(hdu_selector, int):
            key = hdu_selector
        else:
            text = str(hdu_selector).strip()
            key = int(text) if text.lstrip("+-").isdigit() else text
        try:
            hdu = hdul[key]
        except (IndexError, KeyError) as exc:
            raise ValueError(f"template FITS has no HDU {hdu_selector!r}") from exc
        data = hdu.data
        if data is None or np.ndim(data) != 2:
            raise ValueError(f"template HDU {hdu_selector!r} is not a 2D image")
        return hdul.index_of(key), hdu

    for index, hdu in enumerate(hdul):
        data = hdu.data
        if data is not None and np.ndim(data) == 2:
            return index, hdu
    raise ValueError("template FITS contains no 2D image HDU")


def load_target_canvas_from_fits(
    path: Path,
    *,
    hdu_selector: str | int | None = None,
) -> tuple[UnionCanvas, dict[str, Any]]:
    """Build a :class:`UnionCanvas` from an existing 2D FITS image grid.

    The returned header contains the celestial WCS normalized through Astropy
    plus explicit NAXIS/NAXIS1/NAXIS2.  This avoids carrying compression/image
    bookkeeping keywords from ``.fits.fz`` products into newly written files,
    while retaining the same celestial pixel/world mapping.
    """

    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(resolved)

    with fits.open(resolved, memmap=False) as hdul:
        index, hdu = _select_2d_hdu(hdul, hdu_selector)
        data = hdu.data
        assert data is not None
        height, width = (int(data.shape[0]), int(data.shape[1]))
        source_header = hdu.header.copy()
        hdu_name = str(getattr(hdu, "name", "") or "PRIMARY")

    astropy_wcs = AstropyWCS(source_header).celestial
    if astropy_wcs.pixel_n_dim != 2 or astropy_wcs.world_n_dim != 2:
        raise ValueError(f"template {resolved} does not contain a 2D celestial WCS")

    wcs_header = dict(astropy_wcs.to_header(relax=True))
    wcs_header["NAXIS"] = 2
    wcs_header["NAXIS1"] = width
    wcs_header["NAXIS2"] = height

    scales = np.asarray(proj_plane_pixel_scales(astropy_wcs), dtype=np.float64) * 3600.0
    if scales.size < 2 or not np.isfinite(scales[:2]).all() or bool((scales[:2] <= 0).any()):
        raise ValueError(f"template {resolved} has an invalid celestial pixel scale")
    pixel_scale_arcsec = float(np.sqrt(scales[0] * scales[1]))

    # Verify the exact header that TorchSky will consume, not only Astropy's WCS.
    torch_wcs = WCS(dict(wcs_header))
    footprint = torch.as_tensor(torch_wcs.footprint(), dtype=torch.float64)
    if footprint.numel() == 0 or not bool(torch.isfinite(footprint).all()):
        raise ValueError(f"template {resolved} is not compatible with TorchSky WCS")

    canvas = UnionCanvas(
        header=wcs_header,
        shape=(height, width),
        pixel_scale_arcsec=pixel_scale_arcsec,
    )
    metadata = {
        "mode": "fits_template",
        "path": str(resolved),
        "hdu_index": int(index),
        "hdu_name": hdu_name,
        "shape": [height, width],
        "pixel_scale_arcsec": pixel_scale_arcsec,
    }
    return canvas, metadata


def sampled_pool_canvas_coverage(
    pools: Sequence[Any],
    canvas: UnionCanvas,
) -> dict[str, Any]:
    """Measure how many already-sampled native training pixels land on a canvas.

    This is a geometry diagnostic only.  It does not remove or resample pixels;
    ``predict_observation`` will continue to mark out-of-canvas points invalid.
    The report therefore makes any template-footprint loss explicit before a
    long training run starts.
    """

    target = WCS(dict(canvas.header))
    rows: list[dict[str, Any]] = []
    total = 0
    inside_total = 0

    for index, pool in enumerate(pools):
        uv = pool.uv.detach().cpu().double()
        source = WCS(dict(pool.header))
        ra, dec = source.pixel_to_world(uv[:, 0], uv[:, 1])
        x, y = target.world_to_pixel(ra, dec)
        valid = (
            torch.isfinite(x)
            & torch.isfinite(y)
            & (x >= 0)
            & (x <= canvas.shape[1] - 1)
            & (y >= 0)
            & (y <= canvas.shape[0] - 1)
        )
        count = int(uv.shape[0])
        inside = int(valid.sum().item())
        total += count
        inside_total += inside
        rows.append(
            {
                "index": index,
                "path": str(pool.path),
                "band": str(pool.band),
                "detector": str(pool.detector),
                "sampled_pixels": count,
                "inside_canvas": inside,
                "outside_canvas": count - inside,
                "inside_fraction": float(inside / count) if count else float("nan"),
            }
        )

    return {
        "sampled_pixels": total,
        "inside_canvas": inside_total,
        "outside_canvas": total - inside_total,
        "inside_fraction": float(inside_total / total) if total else float("nan"),
        "per_exposure": rows,
    }


__all__ = [
    "load_target_canvas_from_fits",
    "sampled_pool_canvas_coverage",
]
