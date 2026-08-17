"""Functional flat ΛCDM / wCDM distances.

No learnable parameters live here. Cosmodist (or any fitter) wraps
:class:`FlatWCDM` with ``nn.Parameter`` and priors.

Defaults are Planck 2018: ``H0=67.66`` km/s/Mpc, ``Om0=0.3111``, ``w0=-1``.
Distances are Mpc; distance modulus is magnitudes. ``c = 299792.458`` km/s.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

SPEED_OF_LIGHT_KM_S = 299792.458
PLANCK18_H0 = 67.66
PLANCK18_OM0 = 0.3111
PLANCK18_W0 = -1.0


@dataclass(frozen=True)
class FlatWCDM:
    """Flat wCDM. ``w0=-1`` is ΛCDM."""

    H0: float = PLANCK18_H0
    Om0: float = PLANCK18_OM0
    w0: float = PLANCK18_W0

    def e(self, z: Tensor) -> Tensor:
        return e_of_z(z, om0=self.Om0, w0=self.w0)

    def comoving_distance(self, z: Tensor, *, n_steps: int = 2048) -> Tensor:
        return comoving_distance(z, H0=self.H0, om0=self.Om0, w0=self.w0, n_steps=n_steps)

    def luminosity_distance(self, z: Tensor, *, n_steps: int = 2048) -> Tensor:
        return luminosity_distance(z, H0=self.H0, om0=self.Om0, w0=self.w0, n_steps=n_steps)

    def angular_diameter_distance(self, z: Tensor, *, n_steps: int = 2048) -> Tensor:
        return angular_diameter_distance(
            z, H0=self.H0, om0=self.Om0, w0=self.w0, n_steps=n_steps
        )

    def distance_modulus(self, z: Tensor, *, n_steps: int = 2048) -> Tensor:
        return distance_modulus(z, H0=self.H0, om0=self.Om0, w0=self.w0, n_steps=n_steps)


def e_of_z(
    z: Tensor,
    *,
    om0: Tensor | float = PLANCK18_OM0,
    w0: Tensor | float = PLANCK18_W0,
) -> Tensor:
    """Dimensionless Hubble function ``E(z) = H(z)/H0`` for flat wCDM."""
    z_t = torch.as_tensor(z)
    om = torch.as_tensor(om0).to(device=z_t.device, dtype=z_t.dtype)
    w = torch.as_tensor(w0).to(device=z_t.device, dtype=z_t.dtype)
    one_plus_z = 1.0 + z_t
    matter = om * torch.pow(one_plus_z, 3)
    de = (1.0 - om) * torch.pow(one_plus_z, 3.0 * (1.0 + w))
    return torch.sqrt(matter + de)


def _hubble_distance_mpc(H0: Tensor | float, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    h0 = torch.as_tensor(H0).to(device=device, dtype=dtype)
    return SPEED_OF_LIGHT_KM_S / h0


def _nonnegative_redshift(z: Tensor) -> Tensor:
    """Cosmological distances are defined for ``z >= 0``.

    Negative values (network overshoot, floating noise) are treated as
    zero so ``D_C(0) = 0`` and ``D_A`` never divides by ``1+z <= 0``.
    """
    z_t = torch.as_tensor(z)
    if z_t.numel() and not bool(torch.isfinite(z_t).all()):
        raise ValueError("redshift must be finite")
    return z_t.clamp(min=0.0)


def comoving_distance(
    z: Tensor,
    *,
    H0: Tensor | float = PLANCK18_H0,
    om0: Tensor | float = PLANCK18_OM0,
    w0: Tensor | float = PLANCK18_W0,
    n_steps: int = 2048,
) -> Tensor:
    """Line-of-sight comoving distance ``D_C`` in Mpc.

    Integrates ``1/E(z')`` on a shared grid to ``max(z)`` then interpolates.
    ``z`` is clamped at zero; ``D_C(0) = 0``.
    """
    if n_steps < 2:
        raise ValueError("n_steps must be >= 2")
    z_t = torch.as_tensor(z)
    if z_t.numel() == 0:
        return z_t
    z_nonneg = _nonnegative_redshift(z_t)
    z_limit = z_nonneg.amax() * 1.01 + 1e-5
    table_size = max(int(n_steps), 2)
    z_grid = torch.linspace(0.0, 1.0, table_size, device=z_t.device, dtype=z_t.dtype) * z_limit
    integrand = 1.0 / e_of_z(z_grid, om0=om0, w0=w0)
    dz = z_limit / (table_size - 1)
    areas = (integrand[:-1] + integrand[1:]) * 0.5 * dz
    integral = torch.cat(
        (
            torch.zeros(1, device=z_t.device, dtype=z_t.dtype),
            torch.cumsum(areas, dim=0),
        )
    )
    d_c_grid = _hubble_distance_mpc(H0, device=z_t.device, dtype=z_t.dtype) * integral
    step = z_limit / (table_size - 1)
    idx_float = z_nonneg / step
    idx_floor = idx_float.floor().to(dtype=torch.long).clamp(0, table_size - 2)
    weight = idx_float - idx_floor.to(dtype=z_t.dtype)
    return d_c_grid[idx_floor] * (1.0 - weight) + d_c_grid[idx_floor + 1] * weight


def luminosity_distance(
    z: Tensor,
    *,
    H0: Tensor | float = PLANCK18_H0,
    om0: Tensor | float = PLANCK18_OM0,
    w0: Tensor | float = PLANCK18_W0,
    n_steps: int = 2048,
) -> Tensor:
    """``D_L = (1+z) D_C`` in Mpc."""
    z_use = _nonnegative_redshift(torch.as_tensor(z))
    return (1.0 + z_use) * comoving_distance(
        z_use, H0=H0, om0=om0, w0=w0, n_steps=n_steps
    )


def angular_diameter_distance(
    z: Tensor,
    *,
    H0: Tensor | float = PLANCK18_H0,
    om0: Tensor | float = PLANCK18_OM0,
    w0: Tensor | float = PLANCK18_W0,
    n_steps: int = 2048,
) -> Tensor:
    """``D_A = D_C / (1+z)`` in Mpc."""
    z_use = _nonnegative_redshift(torch.as_tensor(z))
    return comoving_distance(z_use, H0=H0, om0=om0, w0=w0, n_steps=n_steps) / (
        1.0 + z_use
    )


def distance_modulus(
    z: Tensor,
    *,
    H0: Tensor | float = PLANCK18_H0,
    om0: Tensor | float = PLANCK18_OM0,
    w0: Tensor | float = PLANCK18_W0,
    n_steps: int = 2048,
) -> Tensor:
    """``μ = 5 log10(D_L / 10 pc)`` with ``D_L`` in Mpc.

    The logarithm argument is floored so ``μ(0)`` stays finite; ``D_L(0)``
    itself is zero.
    """
    d_l = luminosity_distance(z, H0=H0, om0=om0, w0=w0, n_steps=n_steps)
    return 5.0 * torch.log10((d_l * 1.0e6 / 10.0).clamp(min=1e-30))
