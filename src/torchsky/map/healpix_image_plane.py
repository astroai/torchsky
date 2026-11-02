"""Sample HEALPix ring/nest maps onto 2D FITS/WCS image grids (torchsky HEALPix only).

Uses :func:`torchsky.sphere.healpix.get_interp_val` (no ``reproject``). The target grid
can be any valid 2D WCS (TAN, CAR, SIN, ZEA, …); pixel centers are converted with
:class:`torchsky.WCS` then interpolated on the sphere.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ..sphere import healpix as hp
from ..wcs import WCS


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
    via :func:`torchsky.sphere.healpix.get_interp_val`.

    Args:
        healpix_map: Length ``12 * nside**2`` tensor (one map).
        nside: HEALPix resolution parameter.
        nest: ``True`` for NESTED ordering, ``False`` for RING.
        target_header: FITS-like header dict with ``NAXIS1``, ``NAXIS2``, and WCS keys.
        device: Optional compute device (defaults to ``healpix_map``'s device).

    Returns:
        ``[NAXIS2, NAXIS1]`` tensor matching the target image shape.

    Note:
        When ``healpix_map.requires_grad`` is ``True``, gradient flow is
        preserved through the Python fallback interpolator which uses pure
        Torch advanced indexing and weighted summation. The C++ fast path is
        used automatically when no gradient tracking is needed.
    """
    m = torch.as_tensor(healpix_map)
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

    work_dtype = torch.float32 if dev.type == "mps" else torch.float64
    wcs = WCS(dict(target_header)).to(dev)
    result = torch.empty(h, w, dtype=m.dtype, device=dev)
    chunk_size = 128
    m_work = m.to(dtype=work_dtype, device=dev)
    for y_start in range(0, h, chunk_size):
        y_end = min(y_start + chunk_size, h)
        n_rows = y_end - y_start
        y, x = torch.meshgrid(
            torch.arange(y_start, y_end, dtype=work_dtype, device=dev),
            torch.arange(w, dtype=work_dtype, device=dev),
            indexing="ij",
        )
        ra, dec = wcs.pixel_to_world(x, y)
        flat = hp.get_interp_val(
            m_work,
            ra.reshape(-1),
            dec.reshape(-1),
            nest=nest,
            lonlat=True,
        )
        result[y_start:y_end] = flat.reshape(n_rows, w).to(dtype=m.dtype)
    return result


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
    if tuple(img.shape) != (h, w):
        raise ValueError("image shape must match source_header NAXIS2/NAXIS1")
    dev = device if device is not None else img.device
    npix = hp.nside2npix(int(nside))
    work_dtype = torch.float32 if dev.type == "mps" else torch.float64
    wcs = WCS(dict(source_header)).to(dev)
    out = torch.zeros(npix, dtype=work_dtype, device=dev)
    img_work = img.to(dtype=work_dtype, device=dev)
    chunk_size = 128
    any_valid = False
    for y_start in range(0, h, chunk_size):
        y_end = min(y_start + chunk_size, h)
        y, x = torch.meshgrid(
            torch.arange(y_start, y_end, dtype=work_dtype, device=dev),
            torch.arange(w, dtype=work_dtype, device=dev),
            indexing="ij",
        )
        ra, dec = wcs.pixel_to_world(x, y)
        chunk_img = img_work[y_start:y_end]
        valid = torch.isfinite(ra) & torch.isfinite(dec) & torch.isfinite(chunk_img)
        if not bool(valid.any()):
            continue
        any_valid = True
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
            dtype=work_dtype, device=dev
        )
        values_flat = chunk_img.reshape(-1)[valid_flat]
        out.scatter_add_(
            0,
            pix_flat.reshape(-1),
            (weight_flat * values_flat.unsqueeze(0)).reshape(-1),
        )
    if not any_valid:
        return torch.zeros(npix, dtype=img.dtype, device=dev)
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
