"""Sample HEALPix ring/nest maps onto 2D FITS/WCS image grids (torchsky HEALPix only).

Uses :func:`torchsky.wcs.healpix.get_interp_val` (no ``reproject``). The target grid
can be any valid 2D WCS (TAN, CAR, SIN, ZEA, …); pixel centers are converted with
:class:`torchsky.WCS` then interpolated on the sphere.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .. import WCS
from ..wcs import healpix as hp


def sample_healpix_map_on_image_header(
    healpix_map: Tensor,
    *,
    nside: int,
    nest: bool,
    target_header: dict,
    device: torch.device | None = None,
) -> Tensor:
    """Bilinearly sample a HEALPix map onto the pixel grid of ``target_header``.

    ``CTYPE1``/``CTYPE2`` may be any projection supported by :class:`torchsky.WCS`
    (e.g. ``RA---TAN``, ``RA---CAR``). Sampling is spherical bilinear interpolation
    via :func:`torchsky.wcs.healpix.get_interp_val`.

    Args:
        healpix_map: Length ``12 * nside**2`` tensor (one map).
        nside: HEALPix resolution parameter.
        nest: ``True`` for NESTED ordering, ``False`` for RING.
        target_header: FITS-like header dict with ``NAXIS1``, ``NAXIS2``, and WCS keys.
        device: Optional compute device (defaults to ``healpix_map``'s device).

    Returns:
        ``[NAXIS2, NAXIS1]`` tensor matching the target image shape.

    Note:
        Map values are detached before :func:`torchsky.wcs.healpix.get_interp_val`
        because the native interpolation kernels do not accept tensors that require
        grad through DLPack. Use :func:`render_field_cube` for direct autograd through
        pixel centers, or a future pure-torch interpolator if spherical resampling must
        remain differentiable end-to-end.
    """
    m = torch.as_tensor(healpix_map).detach()
    if m.ndim != 1:
        raise ValueError("healpix_map must be a 1D tensor of length 12*nside**2")
    npix = hp.nside2npix(int(nside))
    if int(m.numel()) != npix:
        raise ValueError(
            f"healpix_map length {m.numel()} does not match nside={nside} (npix={npix})"
        )
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


sample_healpix_map_on_tan_header = sample_healpix_map_on_image_header


def accumulate_image_on_healpix_map_adjoint(
    image: Tensor,
    *,
    nside: int,
    nest: bool,
    source_header: dict,
    device: torch.device | None = None,
) -> Tensor:
    """Apply the exact transpose of HEALPix bilinear image sampling.

    For image-plane values ``y[q]`` and bilinear HEALPix weights ``w_i(q)``,
    this returns the HEALPix map coefficients

    ``x[p] = sum_q w_p(q) y[q]``.

    This is the native torchsky adjoint for
    :func:`sample_healpix_map_on_image_header`.
    """
    img = torch.as_tensor(image)
    if img.ndim != 2:
        raise ValueError("image must be a 2D tensor with shape [NAXIS2, NAXIS1]")
    h = int(source_header["NAXIS2"])
    w = int(source_header["NAXIS1"])
    if tuple(int(v) for v in img.shape) != (h, w):
        raise ValueError("image shape must match source_header NAXIS2/NAXIS1")
    dev = device if device is not None else img.device
    npix = hp.nside2npix(int(nside))
    wcs = WCS(dict(source_header))
    y, x = torch.meshgrid(
        torch.arange(h, dtype=torch.float64, device=dev),
        torch.arange(w, dtype=torch.float64, device=dev),
        indexing="ij",
    )
    ra, dec = wcs.pixel_to_world(x, y)
    valid = (
        torch.isfinite(ra) & torch.isfinite(dec) & torch.isfinite(img.to(device=dev))
    )
    if not bool(valid.any()):
        return torch.zeros(npix, dtype=img.dtype, device=dev)
    pix, weight = hp.get_interp_weights(
        int(nside),
        ra.reshape(-1),
        dec.reshape(-1),
        nest=nest,
        lonlat=True,
    )
    valid_flat = valid.reshape(-1)
    pix_flat = pix.reshape(4, -1)[:, valid_flat]
    weight_flat = weight.reshape(4, -1)[:, valid_flat].to(
        dtype=torch.float64, device=dev
    )
    values_flat = img.reshape(-1)[valid_flat].to(dtype=torch.float64, device=dev)
    out = torch.zeros(npix, dtype=torch.float64, device=dev)
    out.scatter_add_(
        0,
        pix_flat.reshape(-1),
        (weight_flat * values_flat.unsqueeze(0)).reshape(-1),
    )
    return out.to(dtype=img.dtype)


def healpix_nside_from_map_len(npix: int) -> int:
    """Return ``nside`` with ``12 * nside**2 == npix``, or raise."""
    if npix % 12 != 0:
        raise ValueError("npix must be divisible by 12")
    nside_f = math.sqrt(npix // 12)
    nside = int(round(nside_f))
    if nside * nside * 12 != npix:
        raise ValueError("npix is not of the form 12*nside**2")
    return nside
