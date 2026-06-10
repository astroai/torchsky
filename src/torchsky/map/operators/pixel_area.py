"""Pixel-area helpers shared by map projection operators."""

from __future__ import annotations

import torch
from torch import Tensor


def wrapped_delta_ra_deg(ra_a_deg: Tensor, ra_b_deg: Tensor) -> Tensor:
    """Shortest signed RA difference in degrees."""
    return ((ra_a_deg - ra_b_deg + 180.0) % 360.0) - 180.0


def pixel_area_sr_map(
    wcs,
    shape: tuple[int, int],
    *,
    device: torch.device,
) -> Tensor:
    """Estimate per-pixel solid angle in steradians from local WCS Jacobians."""
    height, width = int(shape[0]), int(shape[1])
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float64, device=device),
        torch.arange(width, dtype=torch.float64, device=device),
        indexing="ij",
    )
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
    return area_sr.to(dtype=torch.float32)


def representative_pixel_area_sr(
    wcs,
    shape: tuple[int, int],
    *,
    device: torch.device,
) -> float:
    """Estimate a representative central-pixel solid angle in steradians."""
    height, width = int(shape[0]), int(shape[1])
    y = torch.tensor([(max(height - 1, 0)) * 0.5], dtype=torch.float64, device=device)
    x = torch.tensor([(max(width - 1, 0)) * 0.5], dtype=torch.float64, device=device)
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
    "pixel_area_sr_map",
    "representative_pixel_area_sr",
    "wrapped_delta_ra_deg",
]
