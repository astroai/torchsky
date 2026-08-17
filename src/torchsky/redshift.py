"""Sample-wise redshift, velocity kinds, and named spectral dimming.

Edge-based flux-conserving Doppler lives in :mod:`torchsky.spectral`
(``relativistic_doppler_shift``, metres per second). This module is the
sample-wise companion used by object SEDs and imaging renderers.

Dimming conventions are named because they are not interchangeable:

- :func:`dimming_f_lambda` — 1D ``F_λ`` densities: ``F_obs = F_rest / (1+z)``
- :func:`dimming_tolman_sb` — surface brightness: Tolman ``(1+z)**-4``
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

C_KM_S = 299792.458

V_KIND_KMS = 0
V_KIND_REDSHIFT = 1
V_KIND_BETA = 2

_V_KIND_MAP = {
    "km/s": V_KIND_KMS,
    "kms": V_KIND_KMS,
    "velocity": V_KIND_KMS,
    "velocity_kms": V_KIND_KMS,
    "redshift": V_KIND_REDSHIFT,
    "z": V_KIND_REDSHIFT,
    "beta": V_KIND_BETA,
}


def require_one_plus_z(one_plus_z: float | Tensor) -> float | Tensor:
    """Reject non-finite or non-positive ``(1+z)`` factors."""
    if isinstance(one_plus_z, Tensor):
        if not bool(torch.isfinite(one_plus_z).all()) or bool((one_plus_z <= 0).any()):
            raise ValueError("one_plus_z must be finite and positive")
        return one_plus_z
    value = float(one_plus_z)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("one_plus_z must be finite and positive")
    return value


def normalize_v_kind(kind: Any) -> int:
    if kind is None:
        return V_KIND_KMS
    if isinstance(kind, int):
        code = int(kind)
        if code in {V_KIND_KMS, V_KIND_REDSHIFT, V_KIND_BETA}:
            return code
        raise ValueError(f"Unknown velocity kind code: {kind!r}")
    if isinstance(kind, str):
        normalized = _V_KIND_MAP.get(kind.lower().strip())
        if normalized is not None:
            return normalized
        raise ValueError(f"Unknown velocity kind: {kind!r}")
    raise TypeError(
        f"Velocity kind must be None, int, or str; got {type(kind).__name__}"
    )


def one_plus_z_from_beta(beta: Tensor, relativistic: bool = True) -> Tensor:
    if relativistic:
        beta = torch.clamp(beta, min=-0.99999, max=0.99999)
        return torch.sqrt((1.0 + beta) / torch.clamp(1.0 - beta, min=1e-12))
    return 1.0 + beta


def one_plus_z_from_velocity_kms(
    v_kms: Tensor, relativistic: bool = True
) -> Tensor:
    return one_plus_z_from_beta(v_kms / C_KM_S, relativistic=relativistic)


def one_plus_z_from_redshift(z: Tensor) -> Tensor:
    return 1.0 + z


def redshift_from_velocity_kms(
    v_kms: Tensor, relativistic: bool = True
) -> Tensor:
    return one_plus_z_from_velocity_kms(v_kms, relativistic=relativistic) - 1.0


def velocity_kms_from_redshift(
    z: Tensor, relativistic: bool = True
) -> Tensor:
    if relativistic:
        one_plus_z = 1.0 + z
        beta = (one_plus_z.pow(2) - 1.0) / torch.clamp(
            one_plus_z.pow(2) + 1.0, min=1e-12
        )
        return beta * C_KM_S
    return z * C_KM_S


def beta_from_redshift(z: Tensor, relativistic: bool = True) -> Tensor:
    if relativistic:
        one_plus_z = 1.0 + z
        return (one_plus_z.pow(2) - 1.0) / torch.clamp(
            one_plus_z.pow(2) + 1.0, min=1e-12
        )
    return z


def convert_velocity_kind(  # noqa: PLR0911
    value: Tensor, from_kind: Any, to_kind: Any, relativistic: bool = True
) -> Tensor:
    from_code = normalize_v_kind(from_kind)
    to_code = normalize_v_kind(to_kind)
    if from_code == to_code:
        return value
    if from_code == V_KIND_KMS:
        if to_code == V_KIND_REDSHIFT:
            return redshift_from_velocity_kms(value, relativistic=relativistic)
        if to_code == V_KIND_BETA:
            return value / C_KM_S
    if from_code == V_KIND_REDSHIFT:
        if to_code == V_KIND_KMS:
            return velocity_kms_from_redshift(value, relativistic=relativistic)
        if to_code == V_KIND_BETA:
            return beta_from_redshift(value, relativistic=relativistic)
    if from_code == V_KIND_BETA:
        if to_code == V_KIND_KMS:
            return value * C_KM_S
        if to_code == V_KIND_REDSHIFT:
            return one_plus_z_from_beta(value, relativistic=relativistic) - 1.0
    return value


def one_plus_z_from_kind(
    value: Tensor, kind: Any, relativistic: bool = True
) -> Tensor:
    kind_code = normalize_v_kind(kind)
    if kind_code == V_KIND_KMS:
        return one_plus_z_from_velocity_kms(value, relativistic=relativistic)
    if kind_code == V_KIND_REDSHIFT:
        return one_plus_z_from_redshift(value)
    if kind_code == V_KIND_BETA:
        return one_plus_z_from_beta(value, relativistic=relativistic)
    return one_plus_z_from_velocity_kms(value, relativistic=relativistic)


def one_plus_z_from_codes(
    value: Tensor,
    kind_codes: Tensor,
    *,
    default_kind: Any = V_KIND_KMS,
    relativistic: bool = True,
) -> Tensor:
    """Convert per-sample velocity kinds without device-synchronizing branches."""
    default_code = normalize_v_kind(default_kind)
    codes = kind_codes.to(device=value.device, dtype=torch.long)
    while codes.ndim < value.ndim:
        codes = codes.unsqueeze(-1)
    recognized = (
        (codes == V_KIND_KMS) | (codes == V_KIND_REDSHIFT) | (codes == V_KIND_BETA)
    )
    codes = torch.where(recognized, codes, default_code)
    from_kms = one_plus_z_from_velocity_kms(value, relativistic=relativistic)
    from_redshift = one_plus_z_from_redshift(value)
    from_beta = one_plus_z_from_beta(value, relativistic=relativistic)
    return torch.where(
        codes == V_KIND_REDSHIFT,
        from_redshift,
        torch.where(codes == V_KIND_BETA, from_beta, from_kms),
    )


def shift_wavelength(lambda_rest: Tensor, one_plus_z: Tensor | float) -> Tensor:
    """``λ_obs = λ_rest * (1+z)``."""
    factor = require_one_plus_z(
        torch.as_tensor(one_plus_z).to(device=lambda_rest.device)
    )
    return lambda_rest * factor


def dimming_f_lambda(one_plus_z: Tensor | float) -> Tensor:
    """``F_λ`` density dimming ``1 / (1+z)``."""
    factor = require_one_plus_z(torch.as_tensor(one_plus_z))
    return 1.0 / factor


def dimming_tolman_sb(one_plus_z: Tensor | float) -> Tensor:
    """Tolman surface-brightness dimming ``(1+z)**-4``."""
    factor = require_one_plus_z(torch.as_tensor(one_plus_z))
    return torch.pow(factor, -4.0)


def apply_redshift_f_lambda(
    flux_rest: Tensor,
    lambda_rest: Tensor,
    one_plus_z: Tensor | float,
) -> tuple[Tensor, Tensor]:
    """Shift a sampled ``F_λ`` spectrum: ``λ' = λ(1+z)``, ``F' = F / (1+z)``."""
    factor = require_one_plus_z(
        torch.as_tensor(one_plus_z).to(device=flux_rest.device, dtype=flux_rest.dtype)
    )
    if isinstance(factor, Tensor) and factor.ndim == flux_rest.ndim - 1:
        factor = factor.unsqueeze(-1)
    lam = torch.as_tensor(lambda_rest, device=flux_rest.device, dtype=flux_rest.dtype)
    return flux_rest / factor, lam * factor


def apply_redshift(
    flux_rest: Tensor,
    lambda_rest: Tensor,
    velocity_or_z: Tensor,
    is_velocity: bool = True,
) -> tuple[Tensor, Tensor]:
    """USPM-shaped wrapper: velocity in km/s or cosmological redshift."""
    if velocity_or_z.ndim == flux_rest.ndim - 1:
        velocity_or_z = velocity_or_z.unsqueeze(-1)
    velocity_or_z = velocity_or_z.to(device=flux_rest.device, dtype=flux_rest.dtype)
    if is_velocity:
        one_plus_z = one_plus_z_from_velocity_kms(velocity_or_z, relativistic=True)
    else:
        one_plus_z = one_plus_z_from_redshift(velocity_or_z)
    return apply_redshift_f_lambda(flux_rest, lambda_rest, one_plus_z)
