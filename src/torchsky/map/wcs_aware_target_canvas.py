"""Target-canvas helpers for WCS-aware INR training and rendering.

This module adds an optional template-grid path without changing the existing
auto-union-canvas logic. A 2D FITS image (for example a released UPDC i2d
mosaic) can define the common spatial WCS and shape used by the INR. The science
data used for the loss still come from the native CRF/CAL exposures; the
template supplies geometry only.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
import torch
from astropy.io import fits
from astropy.wcs import WCS as AstropyWCS
from astropy.wcs.utils import proj_plane_pixel_scales

from ..wcs import WCS
from .wcs_aware_inr import UnionCanvas


DEFAULT_CANVAS_WCS_CHUNK_PIXELS = 1_000_000


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
    """Build a :class:`UnionCanvas` from an existing 2D FITS image grid."""

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
    if (
        scales.size < 2
        or not np.isfinite(scales[:2]).all()
        or bool((scales[:2] <= 0).any())
    ):
        raise ValueError(f"template {resolved} has an invalid celestial pixel scale")
    pixel_scale_arcsec = float(np.sqrt(scales[0] * scales[1]))

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


def _pool_inside_canvas_mask(
    pool: Any,
    canvas: UnionCanvas,
    *,
    chunk_pixels: int = DEFAULT_CANVAS_WCS_CHUNK_PIXELS,
    progress_label: str | None = None,
) -> torch.Tensor:
    """Return the exact per-sample target-canvas membership mask.

    The membership condition is identical to the original implementation.
    Chunking only bounds temporary WCS tensors and does not change which pixels
    are retained.
    """

    if chunk_pixels <= 0:
        raise ValueError("chunk_pixels must be positive")

    count = int(pool.uv.shape[0])
    keep = torch.empty(count, dtype=torch.bool)
    source = WCS(dict(pool.header))
    target = WCS(dict(canvas.header))
    label = progress_label or Path(pool.path).name
    started = time.perf_counter()

    print(
        f"[target-canvas] {label}: mapping {count:,} native pixels "
        f"in chunks of {chunk_pixels:,}",
        flush=True,
    )

    with torch.inference_mode():
        for start in range(0, count, chunk_pixels):
            stop = min(start + chunk_pixels, count)
            uv = pool.uv[start:stop].detach().cpu().double()
            ra, dec = source.pixel_to_world(uv[:, 0], uv[:, 1])
            x, y = target.world_to_pixel(ra, dec)
            keep[start:stop] = (
                torch.isfinite(x)
                & torch.isfinite(y)
                & (x >= 0)
                & (x <= canvas.shape[1] - 1)
                & (y >= 0)
                & (y <= canvas.shape[0] - 1)
            ).cpu()

            if stop < count:
                elapsed = time.perf_counter() - started
                print(
                    f"[target-canvas] {label}: {stop:,}/{count:,} "
                    f"pixels mapped in {elapsed:.1f}s",
                    flush=True,
                )

    elapsed = time.perf_counter() - started
    inside = int(keep.sum().item())
    print(
        f"[target-canvas] {label}: complete in {elapsed:.1f}s; "
        f"inside={inside:,}, outside={count - inside:,}",
        flush=True,
    )
    return keep


def _coverage_row(index: int, pool: Any, valid: torch.Tensor) -> dict[str, Any]:
    count = int(pool.uv.shape[0])
    inside = int(valid.sum().item())
    return {
        "index": index,
        "path": str(pool.path),
        "band": str(pool.band),
        "detector": str(pool.detector),
        "sampled_pixels": count,
        "inside_canvas": inside,
        "outside_canvas": count - inside,
        "inside_fraction": float(inside / count) if count else float("nan"),
    }


def _coverage_report(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = sum(int(row["sampled_pixels"]) for row in rows)
    inside_total = sum(int(row["inside_canvas"]) for row in rows)
    return {
        "sampled_pixels": total,
        "inside_canvas": inside_total,
        "outside_canvas": total - inside_total,
        "inside_fraction": float(inside_total / total) if total else float("nan"),
        "per_exposure": list(rows),
        "wcs_membership_evaluation_passes": 1,
        "wcs_chunk_pixels": DEFAULT_CANVAS_WCS_CHUNK_PIXELS,
    }


def sampled_pool_canvas_coverage(
    pools: Sequence[Any],
    canvas: UnionCanvas,
) -> dict[str, Any]:
    """Measure how many sampled native training pixels land on a canvas."""

    rows: list[dict[str, Any]] = []
    total_exposures = len(pools)
    for index, pool in enumerate(pools):
        valid = _pool_inside_canvas_mask(
            pool,
            canvas,
            progress_label=(
                f"exposure {index + 1}/{total_exposures} {Path(pool.path).name}"
            ),
        )
        rows.append(_coverage_row(index, pool, valid))
    return _coverage_report(rows)


def filter_sampled_pools_to_canvas(
    pools: Sequence[Any],
    canvas: UnionCanvas,
) -> tuple[list[Any], dict[str, Any]]:
    """Restrict sampled native pixels to the selected target-canvas footprint.

    Each exact membership mask is computed once and reused for both the coverage
    report and filtering. The old implementation computed the same mask once for
    reporting and a second time for filtering.
    """

    filtered: list[Any] = []
    rows: list[dict[str, Any]] = []
    total_exposures = len(pools)

    for index, pool in enumerate(pools):
        keep = _pool_inside_canvas_mask(
            pool,
            canvas,
            progress_label=(
                f"exposure {index + 1}/{total_exposures} {Path(pool.path).name}"
            ),
        )
        if not bool(keep.any()):
            raise ValueError(
                "target canvas contains no sampled pixels for exposure "
                f"{index}: {pool.path}"
            )
        rows.append(_coverage_row(index, pool, keep))
        filtered.append(
            replace(
                pool,
                uv=pool.uv[keep],
                value=pool.value[keep],
                err=pool.err[keep],
            )
        )

    report = _coverage_report(rows)
    report["filter_applied"] = True
    report["post_filter_sampled_pixels"] = int(sum(pool.size for pool in filtered))
    return filtered, report


__all__ = [
    "filter_sampled_pools_to_canvas",
    "load_target_canvas_from_fits",
    "sampled_pool_canvas_coverage",
]
