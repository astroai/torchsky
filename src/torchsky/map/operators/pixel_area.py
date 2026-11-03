"""Pixel-area helpers shared by map projection operators."""

from __future__ import annotations

import torch
from torch import Tensor
from ...device import _as_tensor, _to


def wrapped_delta_ra_deg(ra_a_deg: Tensor, ra_b_deg: Tensor) -> Tensor:
    """Shortest signed RA difference in degrees."""
    return ((ra_a_deg - ra_b_deg + 180.0) % 360.0) - 180.0


def _is_undistorted_tan(wcs) -> bool:
    return bool(getattr(wcs, "_use_fast_tan", False)) and not any(
        (
            getattr(wcs, "sip", None) is not None,
            getattr(wcs, "cpdis1", None) is not None,
            getattr(wcs, "cpdis2", None) is not None,
            getattr(wcs, "det2im1", None) is not None,
            getattr(wcs, "det2im2", None) is not None,
            getattr(wcs, "_pixel_affine_matrix", None) is not None,
            bool(getattr(wcs, "_is_tpv", False)),
            bool(getattr(wcs, "_is_tnx", False)),
            bool(getattr(wcs, "_is_zpx", False)),
        )
    )


def _tan_pixel_area_sr(wcs, x: Tensor, y: Tensor) -> Tensor:
    rel_x = x + 1.0 - float(wcs._crpix0)
    rel_y = y + 1.0 - float(wcs._crpix1)
    xi_deg = wcs._cd00 * rel_x + wcs._cd01 * rel_y
    eta_deg = wcs._cd10 * rel_x + wcs._cd11 * rel_y
    d2r = torch.pi / 180.0
    radius_sq = torch.square(xi_deg * d2r) + torch.square(eta_deg * d2r)
    det_cd_deg2 = abs(wcs._cd00 * wcs._cd11 - wcs._cd01 * wcs._cd10)
    return det_cd_deg2 * (d2r * d2r) / torch.pow(1.0 + radius_sq, 1.5)


def _unit_vectors(ra_deg: Tensor, dec_deg: Tensor) -> Tensor:
    ra = torch.deg2rad(ra_deg)
    dec = torch.deg2rad(dec_deg)
    cos_dec = torch.cos(dec)
    return torch.stack(
        (cos_dec * torch.cos(ra), cos_dec * torch.sin(ra), torch.sin(dec)),
        dim=-1,
    )


def _spherical_triangle_area(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    numerator = torch.abs(torch.sum(a * torch.linalg.cross(b, c), dim=-1))
    denominator = 1.0 + torch.sum(a * b, dim=-1)
    denominator = denominator + torch.sum(b * c, dim=-1) + torch.sum(c * a, dim=-1)
    return 2.0 * torch.atan2(numerator, denominator)


def _spherical_pixel_area_sr(wcs, x: Tensor, y: Tensor) -> Tensor:
    corners = (
        (x - 0.5, y - 0.5),
        (x + 0.5, y - 0.5),
        (x + 0.5, y + 0.5),
        (x - 0.5, y + 0.5),
    )
    vectors = []
    for corner_x, corner_y in corners:
        ra_deg, dec_deg = wcs.pixel_to_world(corner_x, corner_y)
        vectors.append(_unit_vectors(ra_deg, dec_deg))
    return _spherical_triangle_area(vectors[0], vectors[1], vectors[2]) + (
        _spherical_triangle_area(vectors[0], vectors[2], vectors[3])
    )


def _float64_wcs(wcs):
    """The WCS on a device with float64 support.

    MPS has no float64, and world coordinates in float32 degrees are quantised at
    ~1e-5 deg (tens of per cent of an arcsecond pixel), so spherical areas are
    evaluated on the CPU in float64 instead.
    """
    dev = getattr(wcs, "device", None)
    if isinstance(dev, torch.device) and dev.type == "mps":
        return wcs.to(torch.device("cpu"))
    return wcs


def pixel_area_sr_at(wcs, x: Tensor, y: Tensor) -> Tensor:
    """Return local pixel solid angles at arbitrary pixel coordinates.

    ``x`` and ``y`` are broadcast to a common shape.  The finite-difference
    Jacobian follows the same convention as :func:`pixel_area_sr_map`, but
    avoids constructing a full detector-sized area map for sparse footprints.
    """
    caller_wcs_dev = getattr(wcs, "device", None)
    wcs = _float64_wcs(wcs)
    wcs_dev = getattr(wcs, "device", None)
    if not isinstance(x, Tensor) and not isinstance(y, Tensor):
        target_dev = (
            wcs_dev if isinstance(wcs_dev, torch.device) else torch.device("cpu")
        )
        # Results stay on the caller's WCS device even when evaluated on the CPU.
        out_dev = (
            caller_wcs_dev if isinstance(caller_wcs_dev, torch.device) else target_dev
        )
    else:
        reference = x if isinstance(x, Tensor) else y
        assert isinstance(reference, Tensor)
        out_dev = reference.device
        target_dev = wcs_dev if isinstance(wcs_dev, torch.device) else reference.device
    coord_dtype = torch.float32 if target_dev.type == "mps" else torch.float64
    x = _as_tensor(x, dtype=coord_dtype, device=target_dev)
    y = _as_tensor(y, dtype=coord_dtype, device=target_dev)
    x, y = torch.broadcast_tensors(x, y)
    if _is_undistorted_tan(wcs):
        area_sr = _tan_pixel_area_sr(wcs, x, y)
    else:
        area_sr = _spherical_pixel_area_sr(wcs, x, y)
    return torch.clamp(area_sr, min=1e-16).to(device=out_dev)


def pixel_area_sr_map(
    wcs,
    shape: tuple[int, int],
    *,
    device: torch.device,
    chunk_size: int = 128,
    window: tuple[int, int, int, int] | None = None,
) -> Tensor:
    """Per-pixel solid angle in steradians from local WCS Jacobians.

    Uses the exact local TAN Jacobian for undistorted tangent-plane WCS and
    spherical pixel polygons otherwise. Computation is row-chunked for memory.
    ``window=(y0, y1, x0, x1)`` evaluates only those pixels of the ``shape``
    image and returns a ``(y1 - y0, x1 - x0)`` map whose values equal the
    corresponding slice of the full map.
    """
    height, width = int(shape[0]), int(shape[1])
    y0, y1, x0, x1 = (0, height, 0, width) if window is None else window
    y0, x0 = max(int(y0), 0), max(int(x0), 0)
    y1, x1 = min(int(y1), height), min(int(x1), width)
    out_h, out_w = max(y1 - y0, 0), max(x1 - x0, 0)
    wcs = _float64_wcs(wcs)
    wcs_dev = getattr(wcs, "device", device)
    if not isinstance(wcs_dev, torch.device):
        wcs_dev = torch.device(wcs_dev)
    coord_dtype = torch.float32 if wcs_dev.type == "mps" else torch.float64
    area_sr = torch.empty(out_h, out_w, dtype=torch.float32, device=device)
    if out_w == 0:
        return area_sr
    for y_start in range(y0, y1, chunk_size):
        y_end = min(y_start + chunk_size, y1)
        y, x = torch.meshgrid(
            torch.arange(y_start, y_end, dtype=coord_dtype, device=wcs_dev),
            torch.arange(x0, x1, dtype=coord_dtype, device=wcs_dev),
            indexing="ij",
        )
        area_sr[y_start - y0 : y_end - y0] = _to(
            pixel_area_sr_at(wcs, x, y), device=device, dtype=torch.float32
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
    wcs = _float64_wcs(wcs)
    wcs_dev = getattr(wcs, "device", device)
    if not isinstance(wcs_dev, torch.device):
        wcs_dev = torch.device(wcs_dev)
    coord_dtype = torch.float32 if wcs_dev.type == "mps" else torch.float64
    y = torch.tensor([(max(height - 1, 0)) * 0.5], dtype=coord_dtype, device=wcs_dev)
    x = torch.tensor([(max(width - 1, 0)) * 0.5], dtype=coord_dtype, device=wcs_dev)
    return float(pixel_area_sr_at(wcs, x, y).mean().item())


__all__ = [
    "pixel_area_sr_at",
    "pixel_area_sr_map",
    "representative_pixel_area_sr",
    "wrapped_delta_ra_deg",
]
