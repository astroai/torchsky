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

from .wcs.iau2006 import C_M_S, earth_barycentric_velocity_icrs

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
    """Normalize a velocity-kind name or code to its integer code."""
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
    """``1 + z = sqrt((1+b)/(1-b))``; ``b`` is clamped short of ``+-1``."""
    if relativistic:
        beta = torch.clamp(beta, min=-0.99999, max=0.99999)
        return torch.sqrt((1.0 + beta) / (1.0 - beta))
    return 1.0 + beta


def one_plus_z_from_velocity_kms(v_kms: Tensor, relativistic: bool = True) -> Tensor:
    """Convert radial velocity in km/s to the wavelength factor ``1 + z``."""
    return one_plus_z_from_beta(v_kms / C_KM_S, relativistic=relativistic)


def one_plus_z_from_redshift(z: Tensor) -> Tensor:
    """Convert redshift to the wavelength factor ``1 + z``."""
    return 1.0 + z


def redshift_from_velocity_kms(v_kms: Tensor, relativistic: bool = True) -> Tensor:
    """Convert radial velocity in km/s to redshift."""
    return one_plus_z_from_velocity_kms(v_kms, relativistic=relativistic) - 1.0


def _beta_from_one_plus_z(one_plus_z: Tensor) -> Tensor:
    """Invert ``sqrt((1+b)/(1-b)) = 1 + z``; ``(1+z)^2 + 1 >= 1`` always."""
    squared = one_plus_z.square()
    return (squared - 1.0) / (squared + 1.0)


def velocity_kms_from_redshift(z: Tensor, relativistic: bool = True) -> Tensor:
    """Convert redshift to radial velocity in km/s."""
    if relativistic:
        return _beta_from_one_plus_z(1.0 + z) * C_KM_S
    return z * C_KM_S


def beta_from_redshift(z: Tensor, relativistic: bool = True) -> Tensor:
    """Convert redshift to the dimensionless velocity ratio ``v / c``."""
    if relativistic:
        return _beta_from_one_plus_z(1.0 + z)
    return z


def convert_velocity_kind(  # noqa: PLR0911
    value: Tensor, from_kind: Any, to_kind: Any, relativistic: bool = True
) -> Tensor:
    """Convert values among km/s, redshift, and dimensionless ``v / c``."""
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


def one_plus_z_from_kind(value: Tensor, kind: Any, relativistic: bool = True) -> Tensor:
    """Convert values of the specified velocity kind to ``1 + z``."""
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


def zeta_from_redshift(z: Tensor | float) -> Tensor:
    """Rapidity ``ζ = ln(1+z)``. ``z=0`` is zero."""
    z_t = torch.as_tensor(z)
    return torch.log(require_one_plus_z(1.0 + z_t))


def zeta_from_beta(beta: Tensor | float) -> Tensor:
    """Peculiar rapidity ``ζ_pec = atanh(β)``."""
    beta_t = torch.as_tensor(beta)
    return torch.atanh(torch.clamp(beta_t, min=-0.99999, max=0.99999))


def zeta_from_velocity_kms(v_kms: Tensor | float) -> Tensor:
    """``ζ_pec = atanh(v/c)`` for a line-of-sight velocity in km/s."""
    return zeta_from_beta(torch.as_tensor(v_kms) / C_KM_S)


def one_plus_z_from_zeta(zeta: Tensor | float) -> Tensor:
    """``1+z = exp(ζ)``."""
    return torch.exp(torch.as_tensor(zeta))


def compose_rapidities(*zetas: Tensor | float) -> Tensor:
    """Add rapidities. Equivalent to multiplying the ``(1+z)`` factors."""
    if not zetas:
        raise TypeError("compose_rapidities requires at least one rapidity")
    total = torch.as_tensor(zetas[0])
    for zeta in zetas[1:]:
        total = total + torch.as_tensor(zeta).to(device=total.device, dtype=total.dtype)
    return total


def shift_ln_wavelength(ln_lambda: Tensor, zeta: Tensor | float) -> Tensor:
    """``ln λ' = ln λ + ζ``. Formulation Step 3 is ``ln λ_topo = ln λ_rest + ζ_sys + ζ_BERV``."""
    shift = torch.as_tensor(zeta).to(device=ln_lambda.device, dtype=ln_lambda.dtype)
    while shift.ndim < ln_lambda.ndim:
        shift = shift.unsqueeze(-1)
    return ln_lambda + shift


def los_barycentric_beta(unit_xyz: Tensor, tt_mjd: Tensor) -> Tensor:
    """Earth barycentric LOS velocity as ``β = v/c``.

    ``unit_xyz`` is the ICRS unit vector toward the source. Positive ``β``
    means the observer is receding from the source (redshift). USPM uses the
    same sign: ``v_BERV > 0`` when the observer recedes. That is
    ``β = -(v_earth · n̂) / c`` with ``v_earth`` from
    :func:`torchsky.wcs.iau2006.earth_barycentric_velocity_icrs`.
    """
    n = torch.as_tensor(unit_xyz)
    norm = torch.linalg.vector_norm(n, dim=-1, keepdim=True).clamp_min(1.0e-15)
    n = n / norm
    velocity = earth_barycentric_velocity_icrs(torch.as_tensor(tt_mjd))
    velocity = velocity.to(device=n.device, dtype=n.dtype)
    while velocity.ndim < n.ndim:
        velocity = velocity.unsqueeze(0)
    while n.ndim < velocity.ndim:
        n = n.unsqueeze(0)
    approaching = (velocity * n).sum(dim=-1)
    return -approaching / C_M_S


def zeta_berv_from_sky(ra_deg: float, dec_deg: float, tt_mjd: float) -> float:
    """Approximate line-of-sight BERV rapidity for an explicitly TT epoch.

    Convenience scalar wrapper: ICRS sky position plus TT MJD in, rapidity
    ``ζ_BERV = atanh(v_BERV / c)`` out with the USPM sign convention
    (``v_BERV > 0``: observer receding). A generic MJD without a time scale
    is not enough to certify a topocentric correction, so callers must pass
    an explicit TT epoch.
    """
    from .wcs.frames import radec_to_unit

    unit = radec_to_unit(torch.tensor(float(ra_deg)), torch.tensor(float(dec_deg)))
    beta = los_barycentric_beta(unit, torch.tensor(float(tt_mjd)))
    return float(zeta_from_beta(beta))


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
    """Shift a sampled rest-frame ``F_lambda`` spectrum by velocity or redshift.

    Args:
        flux_rest: Rest-frame flux density per unit wavelength.
        lambda_rest: Rest-frame wavelength samples.
        velocity_or_z: Radial velocity in km/s or redshift.
        is_velocity: Interpret ``velocity_or_z`` as km/s when true.

    Returns:
        Observed-frame flux density and wavelength tensors.

    Examples:
        >>> import torch
        >>> from torchsky.redshift import apply_redshift
        >>> flux, wave = apply_redshift(torch.ones(1), torch.tensor([5000.0]), torch.tensor(0.0))
        >>> wave.tolist()
        [5000.0]
    """
    if velocity_or_z.ndim == flux_rest.ndim - 1:
        velocity_or_z = velocity_or_z.unsqueeze(-1)
    velocity_or_z = velocity_or_z.to(device=flux_rest.device, dtype=flux_rest.dtype)
    if is_velocity:
        one_plus_z = one_plus_z_from_velocity_kms(velocity_or_z, relativistic=True)
    else:
        one_plus_z = one_plus_z_from_redshift(velocity_or_z)
    return apply_redshift_f_lambda(flux_rest, lambda_rest, one_plus_z)

__all__ = [
    "C_KM_S",
    "C_M_S",
    "V_KIND_BETA",
    "V_KIND_KMS",
    "V_KIND_REDSHIFT",
    "apply_redshift",
    "apply_redshift_f_lambda",
    "beta_from_redshift",
    "compose_rapidities",
    "convert_velocity_kind",
    "dimming_f_lambda",
    "dimming_tolman_sb",
    "earth_barycentric_velocity_icrs",
    "los_barycentric_beta",
    "normalize_v_kind",
    "one_plus_z_from_beta",
    "one_plus_z_from_codes",
    "one_plus_z_from_kind",
    "one_plus_z_from_redshift",
    "one_plus_z_from_velocity_kms",
    "one_plus_z_from_zeta",
    "redshift_from_velocity_kms",
    "require_one_plus_z",
    "shift_ln_wavelength",
    "shift_wavelength",
    "velocity_kms_from_redshift",
    "zeta_berv_from_sky",
    "zeta_from_beta",
    "zeta_from_redshift",
    "zeta_from_velocity_kms",
]
