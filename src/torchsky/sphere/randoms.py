"""Rejection-free sky sampling over MOC, BMOC, and sparse HEALPix footprints."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from . import healpix as _healpix

if TYPE_CHECKING:
    from .moc import BMOC, MOC
    from .sparse import HealSparseMap


def _sample_in_nested_cells(
    nsides: Tensor | int,
    pix_nested: Tensor,
    u_x: Tensor,
    u_y: Tensor,
) -> tuple[Tensor, Tensor]:
    """Map uniform (u_x, u_y) in [0, 1)^2 inside NESTED HEALPix cells to (lon_deg, lat_deg).

    Because the HEALPix curvilinear projection has constant Jacobian
    |d(z, phi)/d(x, y)| = pi / 3 everywhere on all 12 base faces (both polar
    caps and equatorial belt), uniform sampling in face coordinates (x, y) is
    mathematically exact uniform solid-angle sampling on S^2 with zero
    rejection.
    """
    pix_t = torch.as_tensor(pix_nested, dtype=torch.int64)
    device = pix_t.device
    f_dtype = _healpix._float_dtype_for_device(device)
    ns_t = torch.as_tensor(nsides, dtype=torch.int64, device=device)
    npface = ns_t * ns_t
    face_num = torch.div(pix_t, npface, rounding_mode="floor")
    ipf = pix_t - face_num * npface
    ix = _healpix.compact_bits(ipf).to(f_dtype)
    iy = _healpix.compact_bits(ipf >> 1).to(f_dtype)
    ns_f = ns_t.to(f_dtype)
    # Clamp u_x, u_y slightly below 1.0 to stay strictly inside the cell
    eps = 1e-7 if f_dtype == torch.float32 else 1e-15
    ux = u_x.to(device=device, dtype=f_dtype).clamp(min=0.0, max=1.0 - eps)
    uy = u_y.to(device=device, dtype=f_dtype).clamp(min=0.0, max=1.0 - eps)
    x = (ix + ux) / ns_f
    y = (iy + uy) / ns_f

    jrll, jpll = _healpix._face_consts(device)
    jrll_f = jrll[face_num].to(f_dtype)
    jpll_f = jpll[face_num].to(f_dtype)

    jr = jrll_f - x - y
    nr = torch.empty_like(jr)
    z = torch.empty_like(jr)

    north = jr < 1.0
    south = jr > 3.0
    equat = ~(north | south)

    nr[north] = jr[north]
    z[north] = 1.0 - torch.square(nr[north]) / 3.0
    nr[south] = 4.0 - jr[south]
    z[south] = -1.0 + torch.square(nr[south]) / 3.0
    nr[equat] = 1.0
    z[equat] = (2.0 - jr[equat]) * (2.0 / 3.0)

    nr_safe = torch.where(nr == 0.0, torch.ones_like(nr), nr)
    phi = (jpll_f * nr + x - y) * (math.pi * 0.25) / nr_safe
    phi = torch.where(nr == 0.0, torch.zeros_like(phi), phi)

    lon_deg = torch.remainder(torch.rad2deg(phi), 360.0)
    lat_deg = torch.rad2deg(torch.asin(torch.clamp(z, -1.0, 1.0)))
    return lon_deg, lat_deg


def sample_randoms(
    n: int,
    footprint: MOC | BMOC | HealSparseMap | None = None,
    *,
    weights: HealSparseMap | Tensor | str | None = None,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor]:
    """Generate ``n`` random sky coordinates ``(ra_deg, dec_deg)`` uniformly in solid angle.

    Supports rejection-free sampling inside a :class:`MOC`, :class:`BMOC`, or
    :class:`HealSparseMap`, with optional per-pixel depth/completeness weights.
    """
    # Imported here: moc imports this module, and moc imports sparse.
    from .moc import BMOC, MOC, uniq2nest
    from .sparse import HealSparseMap

    n_i = int(n)
    if n_i < 0:
        raise ValueError("n must be non-negative")
    if footprint is None and isinstance(weights, HealSparseMap):
        footprint = weights
        weights = "values"

    if footprint is None:
        dev = torch.device("cpu") if device is None else torch.device(device)
        f_dtype = _healpix._float_dtype_for_device(dev)
        if n_i == 0:
            empty = torch.empty(0, dtype=f_dtype, device=dev)
            return empty, empty
        u_lon = torch.rand(n_i, dtype=f_dtype, device=dev, generator=generator)
        u_z = torch.rand(n_i, dtype=f_dtype, device=dev, generator=generator)
        lon_deg = u_lon * 360.0
        z = 2.0 * u_z - 1.0
        lat_deg = torch.rad2deg(torch.asin(torch.clamp(z, -1.0, 1.0)))
        return lon_deg, lat_deg

    if isinstance(footprint, BMOC):
        footprint = footprint.to_moc(include_boundary=True)

    if isinstance(footprint, MOC):
        dev = footprint.uniq.device if device is None else torch.device(device)
        f_dtype = _healpix._float_dtype_for_device(dev)
        if n_i == 0:
            empty = torch.empty(0, dtype=f_dtype, device=dev)
            return empty, empty
        if footprint.uniq.numel() == 0:
            raise ValueError("cannot sample randoms from an empty MOC")
        uniq = footprint.uniq.to(device=dev)
        nsides, pix_nested = uniq2nest(uniq)
        ns_f = nsides.to(f_dtype)
        cell_weights = 1.0 / (ns_f * ns_f)
        cdf = torch.cumsum(cell_weights, dim=0)
        u_cell = (
            torch.rand(n_i, dtype=f_dtype, device=dev, generator=generator) * cdf[-1]
        )
        idx = torch.searchsorted(cdf, u_cell, right=False).clamp(max=uniq.numel() - 1)
        ux = torch.rand(n_i, dtype=f_dtype, device=dev, generator=generator)
        uy = torch.rand(n_i, dtype=f_dtype, device=dev, generator=generator)
        return _sample_in_nested_cells(nsides[idx], pix_nested[idx], ux, uy)

    if isinstance(footprint, HealSparseMap):
        dev = footprint.pixels.device if device is None else torch.device(device)
        f_dtype = _healpix._float_dtype_for_device(dev)
        if n_i == 0:
            empty = torch.empty(0, dtype=f_dtype, device=dev)
            return empty, empty
        pix = footprint.pixels.to(device=dev)
        if pix.numel() == 0:
            raise ValueError("cannot sample randoms from an empty HealSparseMap")
        if not footprint.nest:
            pix_nest = _healpix.ring2nest(footprint.nside, pix)
        else:
            pix_nest = pix

        if weights is None:
            idx = torch.randint(
                0, pix_nest.numel(), (n_i,), device=dev, generator=generator
            )
        else:
            if isinstance(weights, str):
                if weights == "values":
                    w_raw = footprint.values
                elif footprint.column_names and weights in footprint.column_names:
                    w_raw = footprint[weights].values
                else:
                    raise ValueError(f"unknown weights specification: {weights!r}")
            elif isinstance(weights, HealSparseMap):
                w_raw = weights.get_values(footprint.pixels)
            else:
                w_raw = torch.as_tensor(weights, device=dev)
            w = w_raw.to(device=dev, dtype=f_dtype).reshape(-1)
            if w.numel() != pix_nest.numel():
                raise ValueError(
                    "weights must have one entry per valid pixel in footprint"
                )
            w = torch.where(torch.isfinite(w) & (w > 0), w, torch.zeros_like(w))
            total_w = float(w.sum().item())
            if total_w <= 0.0:
                raise ValueError("weights must have positive sum")
            cdf = torch.cumsum(w, dim=0)
            u_cell = (
                torch.rand(n_i, dtype=f_dtype, device=dev, generator=generator)
                * cdf[-1]
            )
            idx = torch.searchsorted(cdf, u_cell, right=False).clamp(
                max=pix_nest.numel() - 1
            )
        ux = torch.rand(n_i, dtype=f_dtype, device=dev, generator=generator)
        uy = torch.rand(n_i, dtype=f_dtype, device=dev, generator=generator)
        return _sample_in_nested_cells(footprint.nside, pix_nest[idx], ux, uy)

    raise TypeError("footprint must be a MOC, BMOC, HealSparseMap, or None")
