"""Construct :class:`FieldObservation` lists from binned spectra, IFU cubes, and X-ray-style channels.

Each spectral bin or IFU channel becomes one observation: the integrator still sees a
2D ``image`` (use ``1×1`` for pure 1D spectra) and ``spectral_min`` / ``spectral_max``
from the **native bin edges** you pass in. Keep units consistent across all observations
in a fit (Å, nm, keV, …); set :attr:`FieldObservation.spectral_quantity` to record
whether the axis is wavelength, energy, etc.

RMF/ARF folding and radio/mm frequency axes are left to ingestion layers; optional
``transmission_sigma`` per bin can encode an effective Gaussian width when you have one.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from .core import _header_shape
from .types import FieldObservation, FieldSpectralQuantity


def make_tangent_plane_header(
    height: int,
    width: int,
    *,
    crval_ra_deg: float = 180.0,
    crval_dec_deg: float = 0.0,
    cd11: float = -1.0e-3,
    cd12: float = 0.0,
    cd21: float = 0.0,
    cd22: float = 1.0e-3,
) -> dict[str, float | str | int]:
    """Minimal ``RA---TAN`` / ``DEC--TAN`` header for ``height`` × ``width`` pixels."""
    h = max(int(height), 1)
    w = max(int(width), 1)
    return {
        "NAXIS": 2,
        "NAXIS1": w,
        "NAXIS2": h,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 0.5 * (w + 1),
        "CRPIX2": 0.5 * (h + 1),
        "CRVAL1": float(crval_ra_deg),
        "CRVAL2": float(crval_dec_deg),
        "CUNIT1": "deg",
        "CUNIT2": "deg",
        "CD1_1": float(cd11),
        "CD1_2": float(cd12),
        "CD2_1": float(cd21),
        "CD2_2": float(cd22),
    }


def ifu_cube_spectral_to_leading(datacube: Tensor, *, spectral_axis: int = 0) -> Tensor:
    """Move the spectral axis to index 0 so the tensor is ``(L, H, W)``."""
    return datacube.movedim(int(spectral_axis), 0)


def _per_bin_sigma(
    transmission_sigma: Tensor | Sequence[float | None] | float | None,
    *,
    nbin: int,
) -> list[float | None]:
    if transmission_sigma is None:
        return [None] * nbin
    if isinstance(transmission_sigma, (int, float)):
        s = float(transmission_sigma)
        if not math.isfinite(s) or s <= 0.0:
            return [None] * nbin
        return [s] * nbin
    ts = torch.as_tensor(transmission_sigma).reshape(-1)
    if ts.numel() == 1:
        s = float(ts.item())
        if not math.isfinite(s) or s <= 0.0:
            return [None] * nbin
        return [s] * nbin
    if ts.numel() != nbin:
        raise ValueError(
            "transmission_sigma must be scalar, length nbin, or None "
            f"(got length {ts.numel()}, nbin={nbin})"
        )
    out: list[float | None] = []
    for i in range(nbin):
        s = float(ts[i].item())
        if not math.isfinite(s) or s <= 0.0:
            out.append(None)
        else:
            out.append(s)
    return out


def _field_observations_from_per_bin_bounds(
    spectral_min: Tensor,
    spectral_max: Tensor,
    values: Tensor,
    *,
    spatial_header: dict[str, Any] | None,
    weight: Tensor | None,
    mask: Tensor | None,
    spectral_quantity: FieldSpectralQuantity,
    band_name_prefix: str,
    transmission_sigma: Tensor | Sequence[float | None] | float | None,
    psf_sigma_pix: float,
) -> list[FieldObservation]:
    lo = torch.as_tensor(spectral_min, dtype=torch.float64).reshape(-1)
    hi = torch.as_tensor(spectral_max, dtype=torch.float64).reshape(-1)
    if lo.ndim != 1 or lo.numel() < 1:
        raise ValueError("spectral_min must be a non-empty 1D tensor")
    nbin = int(lo.numel())
    if hi.numel() != nbin:
        raise ValueError("spectral_min and spectral_max must have the same length")
    if not bool(torch.all(torch.isfinite(lo) & torch.isfinite(hi)).item()):
        raise ValueError("spectral bounds must be finite")
    if not bool(torch.all(hi > lo).item()):
        raise ValueError("spectral_max must be greater than spectral_min for every bin")

    vals = torch.as_tensor(values)
    if vals.ndim == 1:
        if vals.shape[0] != nbin:
            raise ValueError(f"values length {vals.shape[0]} must match n_bin={nbin}")
        h, w = 1, 1
        device = vals.device
        dtype = vals.dtype
    elif vals.ndim == 3:
        if int(vals.shape[0]) != nbin:
            raise ValueError(f"values.shape[0]={vals.shape[0]} must equal n_bin={nbin}")
        h, w = int(vals.shape[1]), int(vals.shape[2])
        device = vals.device
        dtype = vals.dtype
    else:
        raise ValueError(
            "values must have shape (n_bin,) or (n_bin, H, W) "
            f"(got shape {tuple(vals.shape)})"
        )

    if spatial_header is None:
        if h != 1 or w != 1:
            raise ValueError(
                "spatial_header is required when values are not a 1D spectrum (H,W) != (1,1)"
            )
        header = make_tangent_plane_header(h, w)
    else:
        header = spatial_header
        eh, ew = _header_shape(header, fallback=(h, w))
        if eh != h or ew != w:
            raise ValueError(
                f"header NAXIS ({ew}x{eh}) does not match values spatial shape {w}x{h}"
            )

    sigmas = _per_bin_sigma(transmission_sigma, nbin=nbin)

    w_tensor: Tensor | None = None
    m_tensor: Tensor | None = None
    if weight is not None:
        w_tensor = torch.as_tensor(weight, dtype=torch.float32, device=device)
        if w_tensor.ndim == 2:
            if w_tensor.shape != (h, w):
                raise ValueError(
                    f"weight spatial shape {tuple(w_tensor.shape)} != {(h, w)}"
                )
        elif w_tensor.ndim == 3:
            if w_tensor.shape != (nbin, h, w):
                raise ValueError(
                    f"weight shape {tuple(w_tensor.shape)} != {(nbin, h, w)}"
                )
        else:
            raise ValueError("weight must be (H, W) or (n_bin, H, W)")
    if mask is not None:
        m_tensor = torch.as_tensor(mask, dtype=torch.bool, device=device)
        if m_tensor.ndim == 2:
            if m_tensor.shape != (h, w):
                raise ValueError(
                    f"mask spatial shape {tuple(m_tensor.shape)} != {(h, w)}"
                )
        elif m_tensor.ndim == 3:
            if m_tensor.shape != (nbin, h, w):
                raise ValueError(
                    f"mask shape {tuple(m_tensor.shape)} != {(nbin, h, w)}"
                )
        else:
            raise ValueError("mask must be (H, W) or (n_bin, H, W)")

    observations: list[FieldObservation] = []
    for i in range(nbin):
        slo = float(lo[i].item())
        shi = float(hi[i].item())
        if vals.ndim == 1:
            img = vals[i].reshape(1, 1).to(dtype=dtype)
        else:
            img = vals[i]
        wi: Tensor | None = None
        ma: Tensor | None = None
        if w_tensor is not None:
            wi = w_tensor[i] if w_tensor.ndim == 3 else w_tensor
        if m_tensor is not None:
            ma = m_tensor[i] if m_tensor.ndim == 3 else m_tensor
        observations.append(
            FieldObservation(
                image=img,
                header=dict(header),
                weight=wi,
                mask=ma,
                spectral_min=slo,
                spectral_max=shi,
                transmission_center=0.5 * (slo + shi),
                transmission_sigma=sigmas[i],
                psf_sigma_pix=float(psf_sigma_pix),
                band_name=f"{band_name_prefix}_{i:06d}",
                spectral_quantity=spectral_quantity,
            )
        )
    return observations


def field_observations_from_bin_edges(
    bin_edges: Tensor | Sequence[float],
    values: Tensor,
    *,
    spatial_header: dict[str, Any] | None = None,
    weight: Tensor | None = None,
    mask: Tensor | None = None,
    spectral_quantity: FieldSpectralQuantity = "wavelength",
    band_name_prefix: str = "bin",
    transmission_sigma: Tensor | Sequence[float | None] | float | None = None,
    psf_sigma_pix: float = 0.0,
) -> list[FieldObservation]:
    """One :class:`FieldObservation` per bin from native ``bin_edges`` and binned values.

    Parameters
    ----------
    bin_edges
        Length ``n_bin + 1``, strictly increasing (same units as the spectral axis).
    values
        Shape ``(n_bin,)`` for a 1D spectrum (stored as ``1×1`` images) or
        ``(n_bin, H, W)`` for an IFU-style stack of channel maps (one map per bin).
    spatial_header
        WCS header for the 2D slice. Required when ``values`` is 3D (IFU). If omitted
        for 1D data, :func:`make_tangent_plane_header` is used with shape ``1×1``.
    weight, mask
        Optional maps matching each per-bin ``image`` (same shape as ``values`` after
        the leading spectral dimension is dropped per observation), or broadcastable
        spatial maps ``(H, W)`` shared across bins.
    spectral_quantity
        ``\"energy\"`` for X-ray channel edges in keV (or eV if you use that consistently),
        ``\"wavelength\"`` for optical/IR, etc.
    band_name_prefix
        Band names are ``f\"{prefix}_{i:06d}\"``.
    transmission_sigma
        Per-bin effective Gaussian width in the **same units as** ``bin_edges``, or a
        scalar broadcast. Invalid or non-positive values are treated as unknown (``None``).
    """
    edges = torch.as_tensor(bin_edges, dtype=torch.float64).reshape(-1)
    if edges.ndim != 1 or edges.numel() < 2:
        raise ValueError("bin_edges must be 1D with length at least 2")
    if not bool(torch.all(edges[1:] > edges[:-1]).item()):
        raise ValueError("bin_edges must be strictly increasing")
    lo = edges[:-1]
    hi = edges[1:]
    return _field_observations_from_per_bin_bounds(
        lo,
        hi,
        values,
        spatial_header=spatial_header,
        weight=weight,
        mask=mask,
        spectral_quantity=spectral_quantity,
        band_name_prefix=band_name_prefix,
        transmission_sigma=transmission_sigma,
        psf_sigma_pix=psf_sigma_pix,
    )


def field_observations_from_bin_centers(
    bin_centers: Tensor | Sequence[float],
    full_widths: Tensor | float,
    values: Tensor,
    *,
    spatial_header: dict[str, Any] | None = None,
    weight: Tensor | None = None,
    mask: Tensor | None = None,
    spectral_quantity: FieldSpectralQuantity = "wavelength",
    band_name_prefix: str = "bin",
    transmission_sigma: Tensor | Sequence[float | None] | float | None = None,
    psf_sigma_pix: float = 0.0,
) -> list[FieldObservation]:
    """Top-hat bins ``[center - width/2, center + width/2]`` (e.g. PHA summaries).

    Bins may be non-contiguous or overlap in axis space; each row is still one
    independent ``[lo, hi]`` interval (unlike :func:`field_observations_from_bin_edges`,
    which assumes a single contiguous edge chain).

    ``full_widths`` may be a scalar broadcast to every bin or a 1D tensor matching
    ``bin_centers``.
    """
    centers = torch.as_tensor(bin_centers, dtype=torch.float64).reshape(-1)
    if centers.ndim != 1 or centers.numel() < 1:
        raise ValueError("bin_centers must be a non-empty 1D tensor")
    nbin = int(centers.numel())
    fw = torch.as_tensor(full_widths, dtype=torch.float64).reshape(-1)
    if fw.numel() == 1:
        half = 0.5 * fw[0].expand(nbin)
    elif fw.numel() == nbin:
        half = 0.5 * fw
    else:
        raise ValueError(
            f"full_widths must be scalar or length n_bin={nbin} (got {fw.numel()})"
        )
    if not bool(torch.all(half > 0).item()):
        raise ValueError("full_widths must be positive")
    lo = centers - half
    hi = centers + half
    return _field_observations_from_per_bin_bounds(
        lo,
        hi,
        values,
        spatial_header=spatial_header,
        weight=weight,
        mask=mask,
        spectral_quantity=spectral_quantity,
        band_name_prefix=band_name_prefix,
        transmission_sigma=transmission_sigma,
        psf_sigma_pix=psf_sigma_pix,
    )


def field_observations_from_ifu_cube(
    datacube: Tensor,
    bin_edges: Tensor | Sequence[float],
    spatial_header: dict[str, Any],
    *,
    spectral_axis: int = 0,
    weight: Tensor | None = None,
    mask: Tensor | None = None,
    spectral_quantity: FieldSpectralQuantity = "wavelength",
    band_name_prefix: str = "ifu",
    transmission_sigma: Tensor | Sequence[float | None] | float | None = None,
    psf_sigma_pix: float = 0.0,
) -> list[FieldObservation]:
    """IFU / image-spectral cube: ``(..., L, ...)`` with spectral axis at ``spectral_axis``.

    After moving that axis to the front, the array must be exactly ``(L, H, W)``.
    For a batch of cubes, slice to one field before calling.
    """
    cube = ifu_cube_spectral_to_leading(datacube, spectral_axis=spectral_axis)
    if cube.ndim != 3:
        raise ValueError(
            "datacube with spectral_axis moved to 0 must be 3D (L, H, W); "
            f"got {cube.ndim}D with shape {tuple(cube.shape)}"
        )
    return field_observations_from_bin_edges(
        bin_edges,
        cube,
        spatial_header=spatial_header,
        weight=weight,
        mask=mask,
        spectral_quantity=spectral_quantity,
        band_name_prefix=band_name_prefix,
        transmission_sigma=transmission_sigma,
        psf_sigma_pix=psf_sigma_pix,
    )
