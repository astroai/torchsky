"""Spherical spectral HEALPix reprojection helpers.

These operators sit on top of torchsky's native spherical harmonic stack rather
than the local image-plane interpolation family. They provide the first explicit
map-side spherical spectral reprojection surface for HEALPix maps and arbitrary
2D WCS image headers.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .. import WCS
from ..sphere import (
    alm2lonlat,
    alm2lonlat_spin,
    alm2map,
    alm_size,
    map2alm,
    map2alm_spin,
    smoothalm,
)
from ..sphere.spectral import _legendre_l_sequence, _wigner_d_l_sequence, alm_index
from ..sphere import healpix as hp
from .healpix_image_plane import sample_healpix_map_on_image_header


def rebandlimit_healpix_map_via_sht(
    healpix_map: Tensor,
    *,
    nside: int,
    nest: bool = False,
    lmax: int | None = None,
    target_nside: int | None = None,
    fwhm_rad: float | None = None,
    beam_window: Tensor | None = None,
    iter: int = 0,
) -> Tensor:
    """Reconstruct a HEALPix map through spherical harmonics and optional beam."""
    m = torch.as_tensor(healpix_map)
    if m.ndim != 1:
        raise ValueError("healpix_map must be a 1D tensor of length 12*nside**2")
    npix = hp.nside2npix(int(nside))
    if int(m.numel()) != npix:
        raise ValueError(
            f"healpix_map length {m.numel()} does not match nside={nside} (npix={npix})"
        )
    ns_out = int(nside if target_nside is None else target_nside)
    ll = int(min(3 * int(nside) - 1, 3 * ns_out - 1) if lmax is None else lmax)
    if ll < 0:
        raise ValueError("lmax must be >= 0")
    alm = map2alm(
        m,
        nside=int(nside),
        lmax=ll,
        nest=nest,
        iter=int(iter),
        pol=False,
    )
    if fwhm_rad is not None or beam_window is not None:
        alm = smoothalm(
            alm,
            lmax=ll,
            mmax=ll,
            fwhm_rad=fwhm_rad,
            beam=beam_window,
            pol=False,
        )
    out = alm2map(
        alm,
        nside=ns_out,
        lmax=ll,
        mmax=ll,
        nest=nest,
        pol=False,
        pixwin=False,
    )
    return torch.as_tensor(out, dtype=m.dtype, device=m.device)


def sample_healpix_map_on_image_header_spectral(
    healpix_map: Tensor,
    *,
    nside: int,
    nest: bool,
    target_header: dict,
    lmax: int | None = None,
    target_nside: int | None = None,
    fwhm_rad: float | None = None,
    beam_window: Tensor | None = None,
    iter: int = 0,
    device: torch.device | None = None,
) -> Tensor:
    """Project a HEALPix map onto an image grid after spherical spectral filtering."""
    hp_map = rebandlimit_healpix_map_via_sht(
        healpix_map,
        nside=nside,
        nest=nest,
        lmax=lmax,
        target_nside=target_nside,
        fwhm_rad=fwhm_rad,
        beam_window=beam_window,
        iter=iter,
    )
    ns_work = int(nside if target_nside is None else target_nside)
    return sample_healpix_map_on_image_header(
        hp_map,
        nside=ns_work,
        nest=nest,
        target_header=target_header,
        device=device,
    )


sample_healpix_map_on_tan_header_spectral = sample_healpix_map_on_image_header_spectral


def sample_alm_on_image_header_spectral(
    alm_values: Tensor,
    *,
    target_header: dict,
    lmax: int | None = None,
    mmax: int | None = None,
    device: torch.device | None = None,
) -> Tensor:
    """Project scalar alm coefficients directly onto an arbitrary WCS image grid."""
    h = int(target_header["NAXIS2"])
    w = int(target_header["NAXIS1"])
    if h < 1 or w < 1:
        raise ValueError("target header must have positive NAXIS1/NAXIS2")
    dev = device if device is not None else torch.as_tensor(alm_values).device
    wcs = WCS(dict(target_header))
    y, x = torch.meshgrid(
        torch.arange(h, dtype=torch.float64, device=dev),
        torch.arange(w, dtype=torch.float64, device=dev),
        indexing="ij",
    )
    ra, dec = wcs.pixel_to_world(x, y)
    out = alm2lonlat(
        alm_values,
        ra,
        dec,
        lmax=lmax,
        mmax=mmax,
    )
    return torch.as_tensor(out, dtype=torch.float64, device=dev)


def sample_healpix_map_on_image_header_harmonic(
    healpix_map: Tensor,
    *,
    nside: int,
    nest: bool,
    target_header: dict,
    lmax: int | None = None,
    fwhm_rad: float | None = None,
    beam_window: Tensor | None = None,
    iter: int = 0,
    device: torch.device | None = None,
) -> Tensor:
    """Project a HEALPix map by harmonic analysis followed by direct alm evaluation."""
    m = torch.as_tensor(healpix_map)
    ll = int(3 * int(nside) - 1 if lmax is None else lmax)
    alm = map2alm(
        m,
        nside=int(nside),
        lmax=ll,
        nest=nest,
        iter=int(iter),
        pol=False,
    )
    if fwhm_rad is not None or beam_window is not None:
        alm = smoothalm(
            alm,
            lmax=ll,
            mmax=ll,
            fwhm_rad=fwhm_rad,
            beam=beam_window,
            pol=False,
        )
    out = sample_alm_on_image_header_spectral(
        alm,
        target_header=target_header,
        lmax=ll,
        mmax=ll,
        device=device,
    )
    return out.to(dtype=m.dtype, device=(device if device is not None else m.device))


sample_alm_on_tan_header_spectral = sample_alm_on_image_header_spectral
sample_healpix_map_on_tan_header_harmonic = sample_healpix_map_on_image_header_harmonic


def sample_spin_alm_on_image_header_spectral(
    alms: tuple[Tensor, Tensor],
    *,
    spin: int,
    target_header: dict,
    lmax: int | None = None,
    mmax: int | None = None,
    device: torch.device | None = None,
) -> Tensor:
    """Project spin-weighted alm coefficients directly onto an arbitrary WCS grid."""
    h = int(target_header["NAXIS2"])
    w = int(target_header["NAXIS1"])
    if h < 1 or w < 1:
        raise ValueError("target header must have positive NAXIS1/NAXIS2")
    dev = device if device is not None else torch.as_tensor(alms[0]).device
    wcs = WCS(dict(target_header))
    y, x = torch.meshgrid(
        torch.arange(h, dtype=torch.float64, device=dev),
        torch.arange(w, dtype=torch.float64, device=dev),
        indexing="ij",
    )
    ra, dec = wcs.pixel_to_world(x, y)
    out = alm2lonlat_spin(
        alms,
        spin,
        ra,
        dec,
        lmax=lmax,
        mmax=mmax,
    )
    return torch.as_tensor(out, dtype=torch.float64, device=dev)


def sample_healpix_spin_map_on_image_header_harmonic(
    maps: Tensor,
    *,
    nside: int,
    spin: int,
    nest: bool,
    target_header: dict,
    lmax: int | None = None,
    iter: int = 0,
    device: torch.device | None = None,
) -> Tensor:
    """Project a spin-weighted HEALPix map via harmonic analysis and direct evaluation."""
    maps_t = torch.as_tensor(maps, dtype=torch.float64)
    if maps_t.ndim != 2 or int(maps_t.shape[0]) != 2:
        raise ValueError("maps must have shape (2, npix)")
    ll = int(3 * int(nside) - 1 if lmax is None else lmax)
    a_e, a_b = map2alm_spin(
        maps_t,
        spin=spin,
        nside=int(nside),
        lmax=ll,
        nest=nest,
        iter=int(iter),
    )
    out = sample_spin_alm_on_image_header_spectral(
        (a_e, a_b),
        spin=spin,
        target_header=target_header,
        lmax=ll,
        mmax=ll,
        device=device,
    )
    return out.to(
        dtype=maps_t.dtype, device=(device if device is not None else maps_t.device)
    )


sample_spin_alm_on_tan_header_spectral = sample_spin_alm_on_image_header_spectral
sample_healpix_spin_map_on_tan_header_harmonic = (
    sample_healpix_spin_map_on_image_header_harmonic
)


def accumulate_image_on_alm_adjoint(
    image: Tensor,
    *,
    source_header: dict,
    lmax: int,
    mmax: int | None = None,
    device: torch.device | None = None,
) -> Tensor:
    """Apply the exact transpose of direct scalar alm image evaluation.

    This is the adjoint of :func:`sample_alm_on_image_header_spectral` under the
    discrete pixel-space inner product used by the image-plane renderer.
    """
    img = torch.as_tensor(image)
    if img.ndim != 2:
        raise ValueError("image must be a 2D tensor with shape [NAXIS2, NAXIS1]")
    h = int(source_header["NAXIS2"])
    w = int(source_header["NAXIS1"])
    if tuple(img.shape) != (h, w):
        raise ValueError("image shape must match source_header NAXIS2/NAXIS1")
    ll = int(lmax)
    mm = int(ll if mmax is None else mmax)
    if mm < 0 or mm > ll:
        raise ValueError("mmax must satisfy 0 <= mmax <= lmax")
    dev = device if device is not None else img.device
    wcs = WCS(dict(source_header))
    y, x = torch.meshgrid(
        torch.arange(h, dtype=torch.float64, device=dev),
        torch.arange(w, dtype=torch.float64, device=dev),
        indexing="ij",
    )
    lon, lat = wcs.pixel_to_world(x, y)
    img_flat = img.reshape(-1).to(dtype=torch.float64, device=dev)
    lon_flat = lon.reshape(-1).to(dtype=torch.float64)
    lat_flat = lat.reshape(-1).to(dtype=torch.float64)
    valid = (
        torch.isfinite(lon_flat) & torch.isfinite(lat_flat) & torch.isfinite(img_flat)
    )
    nalm = alm_size(ll, mm)
    out = torch.zeros(nalm, dtype=torch.complex128, device=dev)
    if not bool(valid.any()):
        return out.to(
            dtype=torch.complex64 if img.dtype == torch.float32 else torch.complex128
        )

    phi = torch.deg2rad(torch.remainder(lon_flat[valid], 360.0))
    x_sin = torch.sin(torch.deg2rad(lat_flat[valid]))
    values = img_flat[valid].to(dtype=torch.complex128)
    offset = 0
    phase_zero = torch.zeros_like(phi)
    for m in range(mm + 1):
        p_seq = _legendre_l_sequence(m, x_sin.to(torch.float64), ll)
        phase = torch.exp(torch.complex(phase_zero, m * phi)).to(torch.complex128)
        ylm = p_seq.to(torch.complex128) * phase.unsqueeze(0)
        coeffs = torch.conj(ylm) @ values
        if m > 0:
            coeffs = 2.0 * coeffs
        l_count = ll - m + 1
        out[offset : offset + l_count] = coeffs[:l_count]
        offset += l_count
    return out.to(
        dtype=torch.complex64 if img.dtype == torch.float32 else torch.complex128
    )


def accumulate_image_on_spin_alm_adjoint(
    maps: Tensor,
    *,
    spin: int,
    source_header: dict,
    lmax: int,
    mmax: int | None = None,
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor]:
    """Apply the exact transpose of direct spin-weighted alm image evaluation."""
    maps_t = torch.as_tensor(maps)
    if maps_t.ndim != 3 or int(maps_t.shape[0]) != 2:
        raise ValueError("maps must have shape (2, NAXIS2, NAXIS1)")
    h = int(source_header["NAXIS2"])
    w = int(source_header["NAXIS1"])
    if tuple(maps_t.shape[-2:]) != (h, w):
        raise ValueError("maps shape must match source_header NAXIS2/NAXIS1")
    raw_spin = float(torch.as_tensor(spin).item())
    spin_i = int(raw_spin)
    if float(spin_i) != raw_spin or spin_i < 0:
        raise ValueError("spin must be a non-negative integer")
    ll = int(lmax)
    mm = int(ll if mmax is None else mmax)
    if mm < 0 or mm > ll:
        raise ValueError("mmax must satisfy 0 <= mmax <= lmax")
    dev = device if device is not None else maps_t.device
    wcs = WCS(dict(source_header))
    y, x = torch.meshgrid(
        torch.arange(h, dtype=torch.float64, device=dev),
        torch.arange(w, dtype=torch.float64, device=dev),
        indexing="ij",
    )
    lon, lat = wcs.pixel_to_world(x, y)
    q_flat = maps_t[0].reshape(-1).to(dtype=torch.float64, device=dev)
    u_flat = maps_t[1].reshape(-1).to(dtype=torch.float64, device=dev)
    lon_flat = lon.reshape(-1).to(dtype=torch.float64)
    lat_flat = lat.reshape(-1).to(dtype=torch.float64)
    valid = (
        torch.isfinite(lon_flat)
        & torch.isfinite(lat_flat)
        & torch.isfinite(q_flat)
        & torch.isfinite(u_flat)
    )
    nalm = alm_size(ll, mm)
    out_e = torch.zeros(nalm, dtype=torch.complex128, device=dev)
    out_b = torch.zeros(nalm, dtype=torch.complex128, device=dev)
    if not bool(valid.any()):
        out_dtype = (
            torch.complex64 if maps_t.dtype == torch.float32 else torch.complex128
        )
        return out_e.to(dtype=out_dtype), out_b.to(dtype=out_dtype)

    phi = torch.deg2rad(torch.remainder(lon_flat[valid], 360.0)).to(torch.float64)
    theta = torch.deg2rad(90.0 - lat_flat[valid]).to(torch.float64)
    alpha_plus = (-0.5 * q_flat[valid] + 0.5j * u_flat[valid]).to(torch.complex128)
    alpha_minus = (-0.5 * q_flat[valid] - 0.5j * u_flat[valid]).to(torch.complex128)
    phase_zero = torch.zeros_like(phi)
    neg_sign = -1.0 if (spin_i & 1) else 1.0

    for m in range(mm + 1):
        phase = torch.exp(torch.complex(phase_zero, m * phi)).to(torch.complex128)
        ell_start = max(m, abs(spin_i))
        if ell_start > ll:
            continue
        d_plus = _wigner_d_l_sequence(ell_start, ll, m, -spin_i, theta)
        d_minus = _wigner_d_l_sequence(ell_start, ll, m, spin_i, theta)
        ell_vec = torch.arange(
            ell_start,
            ll + 1,
            dtype=torch.float64,
            device=theta.device,
        )
        norms = torch.sqrt((2 * ell_vec + 1) / (4.0 * torch.pi))
        m_phase = -1.0 if (m & 1) else 1.0
        spin_phase = -1.0 if (spin_i & 1) else 1.0
        scale = (spin_phase * m_phase) * norms.unsqueeze(1)
        y_plus = (scale * d_plus).to(torch.complex128) * phase.unsqueeze(0)
        y_minus = (scale * d_minus).to(torch.complex128) * phase.unsqueeze(0)
        g_plus = torch.conj(y_plus) @ torch.conj(alpha_plus)
        g_minus = torch.conj(y_minus) @ torch.conj(alpha_minus)
        l_count = ll - ell_start + 1
        start = alm_index(ell_start, m, ll, mm)
        g_plus = g_plus[:l_count]
        g_minus = g_minus[:l_count]
        out_e[start : start + l_count] = neg_sign * g_plus + g_minus
        out_b[start : start + l_count] = 1j * (g_minus - neg_sign * g_plus)

    out_dtype = torch.complex64 if maps_t.dtype == torch.float32 else torch.complex128
    return out_e.to(dtype=out_dtype), out_b.to(dtype=out_dtype)
