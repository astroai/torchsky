"""Galactic foreground screens: distance-integrated 3D dust and DIB absorption.

These are the physical foreground operators consumed by survey forward models:
the formulation's ``MW`` record (``A_V(RA, Dec, d_MW)`` and ISM DIB
absorption). Dust profiles are precomputed per line of sight from a 3D dust
map and interpolated at the current foreground distance ``d_MW``; the DIB
screen is an analytic Gaussian optical-depth approximation of the dominant
diffuse bands (5780, 5797, 6614 Angstrom).
"""

from __future__ import annotations

import torch
from torch import Tensor

# Rest (vacuum) wavelengths of the first diffuse interstellar bands modeled.
DIB_REST_ANGSTROM: tuple[float, ...] = (5780.0, 5797.0, 6614.0)

# Approximate per-unit-E(B-V) integrated optical depth of each band
# (order-of-magnitude strength ranking; refined by science calibration).
DIB_TAU_PER_EBV: tuple[float, ...] = (0.060, 0.022, 0.041)

# Fractional sigma widths (strong bands are ~30-40 Angstrom FWHM).
DIB_SIGMA_ANGSTROM: tuple[float, ...] = (18.0, 8.0, 16.0)


def build_dust_profile_table(
    distances_pc: Tensor,
    a_v_column_pc: Tensor,
) -> dict[str, Tensor]:
    """Pack a monotone cumulative-extinction profile for one line of sight.

    ``a_v_column_pc`` holds the cumulative ``A_V`` accumulated out to each
    distance in ``distances_pc`` (ascending, starting at 0 with ``A_V = 0``).
    The returned table is what survey data pipelines serialize into
    ``meta.json``; training code interpolates it at the current ``d_MW``.
    """
    distances = torch.as_tensor(distances_pc, dtype=torch.float64).reshape(-1)
    column = torch.as_tensor(a_v_column_pc, dtype=torch.float64).reshape(-1)
    if distances.numel() != column.numel():
        raise ValueError("distances_pc and a_v_column_pc must align one-to-one")
    if distances.numel() < 2:
        raise ValueError("a dust profile needs at least two distances")
    if not bool(torch.isfinite(distances).all() and torch.isfinite(column).all()):
        raise ValueError("dust profile values must be finite")
    if bool((distances.diff() <= 0).any()):
        raise ValueError("distances_pc must be strictly ascending")
    if bool((column < 0).any()):
        raise ValueError("cumulative A_V must be non-negative")
    if float(distances[0]) != 0.0 or float(column[0]) != 0.0:
        raise ValueError("profiles must start at distance 0 with A_V 0")
    if bool((column.diff() < 0).any()):
        raise ValueError("cumulative A_V must be non-decreasing with distance")
    return {"distance_pc": distances, "a_v_cumulative": column}


def a_v_at_distance(
    profile: dict[str, Tensor],
    distance_pc: Tensor | float,
    *,
    distance_bin_index: Tensor | None = None,
) -> Tensor:
    """Interpolate cumulative ``A_V(d)`` at the requested foreground distance.

    ``distance_bin_index`` optionally supplies a precomputed searchsorted bin
    (left edge index) so callers can hoist the lookup out of autograd loops;
    gradients flow through the interpolation weights and ``d_MW``.
    Distances beyond the table clamp to the tabulated total column.
    """
    distances = profile["distance_pc"]
    column = profile["a_v_cumulative"]
    d = torch.as_tensor(distance_pc, device=column.device, dtype=column.dtype)
    if d.ndim == 0:
        d = d.reshape(1)
    d = d.clamp(min=0.0)
    if distance_bin_index is None:
        upper = torch.searchsorted(distances, d.detach(), right=False)
    else:
        upper = torch.as_tensor(
            distance_bin_index, device=column.device, dtype=torch.long
        ).reshape(d.shape)
    upper = upper.clamp(min=1, max=distances.numel() - 1)
    lower = upper - 1
    d_lo = distances[lower]
    d_hi = distances[upper]
    weight = (d - d_lo) / (d_hi - d_lo).clamp_min(torch.finfo(d.dtype).tiny)
    weight = weight.clamp(0.0, 1.0)
    return column[lower] + weight * (column[upper] - column[lower])


def extinction_from_a_v(
    wave_angstrom: Tensor,
    a_v: Tensor | float,
    *,
    r_v: float = 3.1,
) -> Tensor:
    """Multiplicative CCM89 extinction screen ``10^(-0.4 A_lambda)``.

    Thin convenience wrapper keeping foreground physics importable from one
    module; the authoritative law lives in :mod:`torchsky.extinction`.
    """
    from .extinction import ccm89

    return torch.pow(10.0, -0.4 * ccm89(wave_angstrom, a_v, r_v=r_v))


def dib_optical_depth(
    wave_angstrom: Tensor,
    ebv: Tensor | float,
    *,
    strength_scale: Tensor | float = 1.0,
    bands: tuple[float, ...] = DIB_REST_ANGSTROM,
    tau_per_ebv: tuple[float, ...] = DIB_TAU_PER_EBV,
    sigma_angstrom: tuple[float, ...] = DIB_SIGMA_ANGSTROM,
) -> Tensor:
    """DIB absorption optical depth ``tau_DIB(lambda)`` at observation frame.

    ``strength_scale`` carries the line-of-sight DIB abundance (e.g. from a
    learned or catalog-scaled field); the total optical depth scales with
    ``E(B-V) * strength_scale``. Each band is a Gaussian in wavelength;
    identity at ``E(B-V) = 0`` or ``strength_scale = 0``.
    """
    if not (len(bands) == len(tau_per_ebv) == len(sigma_angstrom)):
        raise ValueError("bands, tau_per_ebv, and sigma_angstrom must align")
    from .extinction import _align_scale

    wave = torch.as_tensor(wave_angstrom)
    ebv_t = _align_scale(torch.as_tensor(ebv), wave)
    scale = _align_scale(torch.as_tensor(strength_scale), wave)
    tau = torch.zeros(
        torch.broadcast_shapes(wave.shape, ebv_t.shape, scale.shape),
        device=wave.device,
        dtype=wave.dtype,
    )
    for center, tau_0, sigma in zip(bands, tau_per_ebv, sigma_angstrom):
        tau = tau + tau_0 * ebv_t * scale * torch.exp(
            -0.5 * ((wave - center) / sigma).square()
        )
    return tau


def apply_dib_transmission(
    flux: Tensor,
    wave_angstrom: Tensor,
    ebv: Tensor | float,
    *,
    strength_scale: Tensor | float = 1.0,
) -> Tensor:
    """Multiply flux by the DIB transmission ``exp(-tau_DIB)``."""
    tau = dib_optical_depth(wave_angstrom, ebv, strength_scale=strength_scale)
    while tau.ndim < flux.ndim:
        tau = tau.unsqueeze(0)
    return flux * torch.exp(-tau)
