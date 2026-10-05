"""Pixel-area helpers shared by map projection operators."""

from __future__ import annotations

import torch
from torch import Tensor


def wrapped_delta_ra_deg(ra_a_deg: Tensor, ra_b_deg: Tensor) -> Tensor:
    """Shortest signed RA difference in degrees."""
    return ((ra_a_deg - ra_b_deg + 180.0) % 360.0) - 180.0


def _is_undistorted_tan(wcs) -> bool:
    tan_check = bool(getattr(wcs, "_is_tan", False))
    is_tpv = bool(getattr(wcs, "_is_tpv", False))
    has_sip = getattr(wcs, "sip", None) is not None
    return tan_check and not is_tpv and not has_sip


def pixel_area_sr_at(wcs, x: Tensor, y: Tensor) -> Tensor:
    """Return local pixel solid angles at arbitrary pixel coordinates.

    ``x`` and ``y`` are broadcast to a common shape.  The finite-difference
    Jacobian follows the same convention as :func:`pixel_area_sr_map`, but
    avoids constructing a full detector-sized area map for sparse footprints.
    """
    wcs_dev = getattr(wcs, "device", None)
    if not isinstance(x, Tensor) and not isinstance(y, Tensor):
        target_dev = wcs_dev if isinstance(wcs_dev, torch.device) else torch.device("cpu")
        out_dev = target_dev
    else:
        reference = x if isinstance(x, Tensor) else y
        assert isinstance(reference, Tensor)
        out_dev = reference.device
        target_dev = wcs_dev if isinstance(wcs_dev, torch.device) else reference.device
    coord_dtype = torch.float32 if target_dev.type == "mps" else torch.float64
    x = torch.as_tensor(x, dtype=coord_dtype, device=target_dev)
    y = torch.as_tensor(y, dtype=coord_dtype, device=target_dev)
    x, y = torch.broadcast_tensors(x, y)
    ra_c, dec_c = wcs.pixel_to_world(x, y)
    ra_dx, dec_dx = wcs.pixel_to_world(x + 1.0, y)
    ra_dy, dec_dy = wcs.pixel_to_world(x, y + 1.0)
    cos_dec = torch.cos(torch.deg2rad(dec_c))
    dra_dx = wrapped_delta_ra_deg(ra_dx, ra_c) * cos_dec
    dra_dy = wrapped_delta_ra_deg(ra_dy, ra_c) * cos_dec
    ddec_dx = dec_dx - dec_c
    ddec_dy = dec_dy - dec_c
    det_deg2 = torch.abs(dra_dx * ddec_dy - dra_dy * ddec_dx)
    d2r = torch.pi / 180.0
    return torch.clamp(det_deg2 * (d2r * d2r), min=1e-16).to(device=out_dev)


def pixel_area_sr_map(
    wcs,
    shape: tuple[int, int],
    *,
    device: torch.device,
    chunk_size: int = 128,
) -> Tensor:
    """Per-pixel solid angle in steradians from local WCS Jacobians.

    Uses a constant-area shortcut for undistorted TAN projections and
    computes the full per-pixel map otherwise (row-chunked for memory).
    """
    height, width = int(shape[0]), int(shape[1])
    if _is_undistorted_tan(wcs):
        rep = representative_pixel_area_sr(wcs, shape, device=device)
        return torch.full((height, width), rep, dtype=torch.float32, device=device)
    wcs_dev = getattr(wcs, "device", device)
    if not isinstance(wcs_dev, torch.device):
        wcs_dev = torch.device(wcs_dev)
    coord_dtype = torch.float32 if wcs_dev.type == "mps" else torch.float64
    area_sr = torch.empty(height, width, dtype=torch.float32, device=device)
    for y_start in range(0, height, chunk_size):
        y_end = min(y_start + chunk_size, height)
        y, x = torch.meshgrid(
            torch.arange(y_start, y_end, dtype=coord_dtype, device=wcs_dev),
            torch.arange(width, dtype=coord_dtype, device=wcs_dev),
            indexing="ij",
        )
        area_sr[y_start:y_end] = pixel_area_sr_at(wcs, x, y).to(
            device=device, dtype=torch.float32
        )
    return area_sr


def representative_pixel_area_sr(
    wcs,
    shape: tuple[int, int],
    *,
    device: torch.device,
) -> float:
    """Estimate a representative central-pixel solid angle in steradians."""
    height, width = int(shape[0]), int(shape[1])
    wcs_dev = getattr(wcs, "device", device)
    if not isinstance(wcs_dev, torch.device):
        wcs_dev = torch.device(wcs_dev)
    coord_dtype = torch.float32 if wcs_dev.type == "mps" else torch.float64
    y = torch.tensor([(max(height - 1, 0)) * 0.5], dtype=coord_dtype, device=wcs_dev)
    x = torch.tensor([(max(width - 1, 0)) * 0.5], dtype=coord_dtype, device=wcs_dev)
    ra_c, dec_c = wcs.pixel_to_world(x, y)
    ra_dx, dec_dx = wcs.pixel_to_world(x + 1.0, y)
    ra_dy, dec_dy = wcs.pixel_to_world(x, y + 1.0)
    cos_dec = torch.cos(torch.deg2rad(dec_c))
    dra_dx = wrapped_delta_ra_deg(ra_dx, ra_c) * cos_dec
    dra_dy = wrapped_delta_ra_deg(ra_dy, ra_c) * cos_dec
    ddec_dx = dec_dx - dec_c
    ddec_dy = dec_dy - dec_c
    det_deg2 = torch.abs(dra_dx * ddec_dy - dra_dy * ddec_dx)
    d2r = torch.pi / 180.0
    area_sr = torch.clamp(det_deg2 * (d2r * d2r), min=1e-16)
    return float(area_sr.mean().item())


__all__ = [
    "pixel_area_sr_at",
    "pixel_area_sr_map",
    "representative_pixel_area_sr",
    "wrapped_delta_ra_deg",
]
