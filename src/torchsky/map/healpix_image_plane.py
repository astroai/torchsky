"""Sample HEALPix ring/nest maps onto tangent-plane FITS grids (torchsky HEALPix only).

This path uses :func:`torchsky.wcs.healpix.get_interp_val` and does **not** require
``reproject``. It is intended for small cubes and regression checks against
``reproject_from_healpix`` when that package is installed.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .. import WCS
from ..wcs import healpix as hp


def sample_healpix_map_on_tan_header(
    healpix_map: Tensor,
    *,
    nside: int,
    nest: bool,
    target_header: dict,
    device: torch.device | None = None,
) -> Tensor:
    """Bilinearly sample a HEALPix map onto the pixel grid of a TAN (or any) WCS header.

    Args:
        healpix_map: Length ``12 * nside**2`` tensor (one map).
        nside: HEALPix resolution parameter.
        nest: ``True`` for NESTED ordering, ``False`` for RING.
        target_header: FITS-like header dict with ``NAXIS1``, ``NAXIS2``, and WCS keys.
        device: Optional compute device (defaults to ``healpix_map``'s device).

    Returns:
        ``[NAXIS2, NAXIS1]`` tensor matching the target image shape.
    """
    m = torch.as_tensor(healpix_map)
    if m.ndim != 1:
        raise ValueError("healpix_map must be a 1D tensor of length 12*nside**2")
    npix = hp.nside2npix(int(nside))
    if int(m.numel()) != npix:
        raise ValueError(f"healpix_map length {m.numel()} does not match nside={nside} (npix={npix})")
    dev = device if device is not None else m.device
    h = int(target_header["NAXIS2"])
    w = int(target_header["NAXIS1"])
    if h < 1 or w < 1:
        raise ValueError("target header must have positive NAXIS1/NAXIS2")

    wcs = WCS(dict(target_header))
    y, x = torch.meshgrid(
        torch.arange(h, dtype=torch.float64, device=dev),
        torch.arange(w, dtype=torch.float64, device=dev),
        indexing="ij",
    )
    ra, dec = wcs.pixel_to_world(x, y)
    flat = hp.get_interp_val(
        m.to(dtype=torch.float64, device=dev),
        ra.reshape(-1),
        dec.reshape(-1),
        nest=nest,
        lonlat=True,
    )
    return flat.reshape(h, w).to(dtype=m.dtype)


def healpix_nside_from_map_len(npix: int) -> int:
    """Return ``nside`` with ``12 * nside**2 == npix``, or raise."""
    if npix % 12 != 0:
        raise ValueError("npix must be divisible by 12")
    nside_f = math.sqrt(npix // 12)
    nside = int(round(nside_f))
    if nside * nside * 12 != npix:
        raise ValueError("npix is not of the form 12*nside**2")
    return nside
