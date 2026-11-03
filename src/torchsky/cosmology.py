"""Differentiable FLRW distance and protected geometry primitives.

No learnable parameters or cosmological priors live here. Science projects pass
plain tensors (including ``nn.Parameter`` instances) through :class:`FLRW`.

Defaults are Planck 2018: ``H0=67.66`` km/s/Mpc, ``Om0=0.3111``, ``w0=-1``.
Distances are Mpc; distance modulus is magnitudes; times are Gyr (Julian
years). ``c = 299792.458`` km/s.
Radiation is neglected. Curvature follows the standard convention ``Ok0 > 0``
for an open universe. ``wa`` uses the CPL law ``w(a) = w0 + wa (1-a)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from .device import _as_tensor, _to

SPEED_OF_LIGHT_KM_S = 299792.458
PLANCK18_H0 = 67.66
PLANCK18_OM0 = 0.3111
PLANCK18_W0 = -1.0
#: Megaparsec in km (IAU 2015 parsec) and gigayear (Julian) in seconds.
_MPC_KM = 3.0856775814913673e19
_GYR_S = 3.15576e16
#: Hubble time ``1/H0`` in Gyr for ``H0 = 1`` km/s/Mpc.
HUBBLE_TIME_GYR = _MPC_KM / _GYR_S


_GL_NODES = 64


def _gauss_legendre_unit(
    n: int, device: torch.device, dtype: torch.dtype
) -> tuple[Tensor, Tensor]:
    """Gauss-Legendre nodes and weights on [0, 1]."""
    import numpy as np

    x, w = np.polynomial.legendre.leggauss(n)
    return (
        _as_tensor(0.5 * (x + 1.0), device=device, dtype=dtype),
        _as_tensor(0.5 * w, device=device, dtype=dtype),
    )


def _floating_tensor(value: Tensor | float) -> Tensor:
    tensor = torch.as_tensor(value)
    return (
        tensor if tensor.is_floating_point() else tensor.to(torch.get_default_dtype())
    )


def _as_parameter(
    value: Tensor | float, *, device: torch.device, dtype: torch.dtype
) -> Tensor:
    """Cast a cosmology parameter to ``dtype`` without a float32 detour.

    ``torch.as_tensor(0.6766)`` is float32 under the default dtype, which loses
    ~6e-8 before a later cast to float64; Python scalars are built directly in
    the target dtype instead.
    """
    if isinstance(value, Tensor):
        return _to(value, device=device, dtype=dtype)
    return _as_tensor(value, device=device, dtype=dtype)


@dataclass(frozen=True)
class APScaling:
    """Alcock--Paczynski geometry at a redshift.

    ``alpha_parallel`` and ``alpha_perpendicular`` are dimensionless ratios
    relative to the supplied reference cosmology. ``f_ap = D_M H / c`` is the
    absolute, dimensionless AP observable for the evaluated cosmology.
    """

    alpha_parallel: Tensor
    alpha_perpendicular: Tensor
    f_ap: Tensor


@dataclass(frozen=True)
class FLRW:
    """Matter + curvature + CPL dark-energy FLRW parameters.

    Fields may be scalar tensors so callers can differentiate with respect to
    cosmological parameters. This container deliberately owns no priors.

    Args:
        H0: Present-day Hubble parameter in km/s/Mpc.
        Om0: Present-day matter density parameter.
        Ok0: Present-day curvature density parameter.
        w0: Dark-energy equation-of-state value at scale factor one.
        wa: CPL dark-energy evolution coefficient.

    Examples:
        >>> import torch
        >>> from torchsky.cosmology import FLRW
        >>> FLRW().comoving_distance(torch.tensor(0.0)).item()
        0.0
    """

    H0: Tensor | float = PLANCK18_H0
    Om0: Tensor | float = PLANCK18_OM0
    Ok0: Tensor | float = 0.0
    w0: Tensor | float = PLANCK18_W0
    wa: Tensor | float = 0.0

    def e(self, z: Tensor) -> Tensor:
        return e_of_z(z, om0=self.Om0, ok0=self.Ok0, w0=self.w0, wa=self.wa)

    def hubble_parameter(self, z: Tensor) -> Tensor:
        return hubble_parameter(
            z,
            H0=self.H0,
            om0=self.Om0,
            ok0=self.Ok0,
            w0=self.w0,
            wa=self.wa,
        )

    def comoving_distance(self, z: Tensor) -> Tensor:
        return comoving_distance(
            z,
            H0=self.H0,
            om0=self.Om0,
            ok0=self.Ok0,
            w0=self.w0,
            wa=self.wa,
        )

    def transverse_comoving_distance(self, z: Tensor) -> Tensor:
        return transverse_comoving_distance(
            z,
            H0=self.H0,
            om0=self.Om0,
            ok0=self.Ok0,
            w0=self.w0,
            wa=self.wa,
        )

    def luminosity_distance(self, z: Tensor) -> Tensor:
        return luminosity_distance(
            z,
            H0=self.H0,
            om0=self.Om0,
            ok0=self.Ok0,
            w0=self.w0,
            wa=self.wa,
        )

    def angular_diameter_distance(self, z: Tensor) -> Tensor:
        return angular_diameter_distance(
            z,
            H0=self.H0,
            om0=self.Om0,
            ok0=self.Ok0,
            w0=self.w0,
            wa=self.wa,
        )

    def differential_comoving_volume(self, z: Tensor) -> Tensor:
        return differential_comoving_volume(z, self)

    def lookback_time(self, z: Tensor) -> Tensor:
        return lookback_time(
            z, H0=self.H0, om0=self.Om0, ok0=self.Ok0, w0=self.w0, wa=self.wa
        )

    def age(self, z: Tensor) -> Tensor:
        return age(z, H0=self.H0, om0=self.Om0, ok0=self.Ok0, w0=self.w0, wa=self.wa)

    def distance_modulus(self, z: Tensor) -> Tensor:
        return distance_modulus(
            z,
            H0=self.H0,
            om0=self.Om0,
            ok0=self.Ok0,
            w0=self.w0,
            wa=self.wa,
        )


@dataclass(frozen=True)
class FlatWCDM(FLRW):
    """Flat wCDM. ``w0=-1`` is ΛCDM."""


def e_of_z(
    z: Tensor,
    *,
    om0: Tensor | float = PLANCK18_OM0,
    ok0: Tensor | float = 0.0,
    w0: Tensor | float = PLANCK18_W0,
    wa: Tensor | float = 0.0,
) -> Tensor:
    """Dimensionless Hubble function ``E(z) = H(z)/H0`` for CPL FLRW."""
    z_t = _floating_tensor(z)
    om = _as_parameter(om0, device=z_t.device, dtype=z_t.dtype)
    ok = _as_parameter(ok0, device=z_t.device, dtype=z_t.dtype)
    w = _as_parameter(w0, device=z_t.device, dtype=z_t.dtype)
    w_a = _as_parameter(wa, device=z_t.device, dtype=z_t.dtype)
    one_plus_z = 1.0 + z_t
    if z_t.numel() and (
        not bool(torch.isfinite(z_t).all()) or bool((one_plus_z <= 0.0).any())
    ):
        raise ValueError("e_of_z requires finite redshift > -1")
    matter = om * torch.pow(one_plus_z, 3)
    curvature = ok * torch.pow(one_plus_z, 2)
    dark_energy_scale = torch.pow(one_plus_z, 3.0 * (1.0 + w + w_a)) * torch.exp(
        -3.0 * w_a * z_t / one_plus_z
    )
    expansion_squared = matter + curvature + (1.0 - om - ok) * dark_energy_scale
    if expansion_squared.numel() and (
        not bool(torch.isfinite(expansion_squared).all())
        or bool((expansion_squared <= 0.0).any())
    ):
        raise ValueError("FLRW parameters produce non-positive E(z)^2")
    return torch.sqrt(expansion_squared)


def hubble_parameter(
    z: Tensor,
    *,
    H0: Tensor | float = PLANCK18_H0,
    om0: Tensor | float = PLANCK18_OM0,
    ok0: Tensor | float = 0.0,
    w0: Tensor | float = PLANCK18_W0,
    wa: Tensor | float = 0.0,
) -> Tensor:
    """Hubble parameter ``H(z)`` in km/s/Mpc."""
    z_t = _floating_tensor(z)
    h0 = _as_parameter(H0, device=z_t.device, dtype=z_t.dtype)
    if not bool(torch.isfinite(h0).all()) or bool((h0 <= 0.0).any()):
        raise ValueError("H0 must be finite and positive")
    return h0 * e_of_z(z_t, om0=om0, ok0=ok0, w0=w0, wa=wa)


def _hubble_distance_mpc(
    H0: Tensor | float, *, device: torch.device, dtype: torch.dtype
) -> Tensor:
    h0 = _as_parameter(H0, device=device, dtype=dtype)
    if not bool(torch.isfinite(h0).all()) or bool((h0 <= 0.0).any()):
        raise ValueError("H0 must be finite and positive")
    return SPEED_OF_LIGHT_KM_S / h0


def _nonnegative_redshift(z: Tensor) -> Tensor:
    """Cosmological distances are defined for ``z >= 0``.

    Negative values (network overshoot, floating noise) are treated as
    zero so ``D_C(0) = 0`` and ``D_A`` never divides by ``1+z <= 0``.
    """
    z_t = _floating_tensor(z)
    if z_t.numel() and not bool(torch.isfinite(z_t).all()):
        raise ValueError("redshift must be finite")
    # where-form keeps a unit gradient exactly at z = 0 (clamp gives 0 there).
    return torch.where(z_t < 0.0, torch.zeros_like(z_t), z_t)


def comoving_distance(
    z: Tensor,
    *,
    H0: Tensor | float = PLANCK18_H0,
    om0: Tensor | float = PLANCK18_OM0,
    ok0: Tensor | float = 0.0,
    w0: Tensor | float = PLANCK18_W0,
    wa: Tensor | float = 0.0,
) -> Tensor:
    """Line-of-sight comoving distance ``D_C`` in Mpc.

    Integrates ``1/E(z')`` per point with fixed Gauss-Legendre nodes in
    ``ln(1+z)``; results are independent of the other batch entries.
    Negative ``z`` is treated as zero; ``D_C(0) = 0``.

    Args:
        z: Redshift tensor.
        H0: Present-day Hubble parameter in km/s/Mpc.
        om0: Present-day matter density parameter.
        ok0: Present-day curvature density parameter.
        w0: Dark-energy equation-of-state value at scale factor one.
        wa: CPL dark-energy evolution coefficient.

    Returns:
        Line-of-sight comoving distances in Mpc with the shape of ``z``.

    Examples:
        >>> import torch
        >>> from torchsky.cosmology import comoving_distance
        >>> comoving_distance(torch.tensor([0.0])).tolist()
        [0.0]
    """
    z_t = _floating_tensor(z)
    if z_t.numel() == 0:
        return z_t
    z_nonneg = _nonnegative_redshift(z_t)
    # Per-point Gauss-Legendre quadrature in u = ln(1+z):
    #   chi = D_H * int_0^{u_max} (1+z(u)) / E(z(u)) du.
    # The integrand is smooth in u out to z ~ 1e3, and every query point has
    # its own nodes, so a result never depends on the rest of the batch.
    nodes, weights = _gauss_legendre_unit(_GL_NODES, z_t.device, z_t.dtype)
    u_max = torch.log1p(z_nonneg).unsqueeze(-1)
    u = u_max * nodes
    one_plus_z = torch.exp(u)
    integrand = one_plus_z / e_of_z(one_plus_z - 1.0, om0=om0, ok0=ok0, w0=w0, wa=wa)
    integral = u_max.squeeze(-1) * torch.sum(integrand * weights, dim=-1)
    return _hubble_distance_mpc(H0, device=z_t.device, dtype=z_t.dtype) * integral


def transverse_comoving_distance(
    z: Tensor,
    *,
    H0: Tensor | float = PLANCK18_H0,
    om0: Tensor | float = PLANCK18_OM0,
    ok0: Tensor | float = 0.0,
    w0: Tensor | float = PLANCK18_W0,
    wa: Tensor | float = 0.0,
) -> Tensor:
    """Transverse comoving distance ``D_M`` in Mpc.

    A series around ``Ok0 = 0`` keeps the value and curvature gradient finite.
    """
    z_t = _floating_tensor(z)
    d_c = comoving_distance(
        z_t,
        H0=H0,
        om0=om0,
        ok0=ok0,
        w0=w0,
        wa=wa,
    )
    ok = _as_parameter(ok0, device=z_t.device, dtype=z_t.dtype)
    d_h = _hubble_distance_mpc(H0, device=z_t.device, dtype=z_t.dtype)
    chi = d_c / d_h
    abs_ok = torch.abs(ok)
    epsilon = torch.as_tensor(torch.finfo(z_t.dtype).eps, device=z_t.device)
    safe_abs_ok = abs_ok.clamp_min(epsilon)
    sqrt_abs_ok = torch.sqrt(safe_abs_ok)
    argument = sqrt_abs_ok * chi
    open_distance = d_h * torch.sinh(argument) / sqrt_abs_ok
    closed_distance = d_h * torch.sin(argument) / sqrt_abs_ok
    series = d_c * (
        1.0
        + ok * torch.square(chi) / 6.0
        + torch.square(ok * torch.square(chi)) / 120.0
    )
    curved = torch.where(ok > 0.0, open_distance, closed_distance)
    # The first omitted term is O((Ok0 * chi^2)^3). Select the series where
    # its second-order retained correction is already at machine precision.
    use_series = abs_ok * torch.square(chi) <= torch.sqrt(epsilon)
    return torch.where(use_series, series, curved)


def luminosity_distance(
    z: Tensor,
    *,
    H0: Tensor | float = PLANCK18_H0,
    om0: Tensor | float = PLANCK18_OM0,
    ok0: Tensor | float = 0.0,
    w0: Tensor | float = PLANCK18_W0,
    wa: Tensor | float = 0.0,
) -> Tensor:
    """``D_L = (1+z) D_M`` in Mpc."""
    z_use = _nonnegative_redshift(torch.as_tensor(z))
    return (1.0 + z_use) * transverse_comoving_distance(
        z_use, H0=H0, om0=om0, ok0=ok0, w0=w0, wa=wa
    )


def angular_diameter_distance(
    z: Tensor,
    *,
    H0: Tensor | float = PLANCK18_H0,
    om0: Tensor | float = PLANCK18_OM0,
    ok0: Tensor | float = 0.0,
    w0: Tensor | float = PLANCK18_W0,
    wa: Tensor | float = 0.0,
) -> Tensor:
    """``D_A = D_M / (1+z)`` in Mpc."""
    z_use = _nonnegative_redshift(torch.as_tensor(z))
    return transverse_comoving_distance(
        z_use,
        H0=H0,
        om0=om0,
        ok0=ok0,
        w0=w0,
        wa=wa,
    ) / (1.0 + z_use)


def differential_comoving_volume(
    z: Tensor,
    cosmology: FLRW = FLRW(),
) -> Tensor:
    """Observed-coordinate volume Jacobian ``dV_c/(dOmega dz)``.

    The result is ``c D_M(z)^2 / H(z)`` in ``Mpc^3 sr^-1`` per unit redshift.
    It is the measure factor needed by observed-coordinate point-process
    likelihoods; selection functions are not included.
    """
    z_use = _nonnegative_redshift(torch.as_tensor(z))
    d_m = cosmology.transverse_comoving_distance(z_use)
    hubble = cosmology.hubble_parameter(z_use)
    return SPEED_OF_LIGHT_KM_S * d_m.square() / hubble


def _hubble_time_gyr(
    H0: Tensor | float, *, device: torch.device, dtype: torch.dtype
) -> Tensor:
    h0 = _as_parameter(H0, device=device, dtype=dtype)
    if not bool(torch.isfinite(h0).all()) or bool((h0 <= 0.0).any()):
        raise ValueError("H0 must be finite and positive")
    return HUBBLE_TIME_GYR / h0


def lookback_time(
    z: Tensor,
    *,
    H0: Tensor | float = PLANCK18_H0,
    om0: Tensor | float = PLANCK18_OM0,
    ok0: Tensor | float = 0.0,
    w0: Tensor | float = PLANCK18_W0,
    wa: Tensor | float = 0.0,
) -> Tensor:
    """Lookback time ``t_L(z) = t_H int_0^z dz' / ((1+z') E(z'))`` in Gyr.

    ``t_H = 1/H0``. The integrand is ``1/E`` in ``u = ln(1+z)`` and uses the
    same per-point 64-node Gauss-Legendre rule as :func:`comoving_distance`.
    Negative ``z`` is treated as zero. Radiation is neglected.

    Args:
        z: Redshift tensor.
        H0: Present-day Hubble parameter in km/s/Mpc.
        om0: Present-day matter density parameter.
        ok0: Present-day curvature density parameter.
        w0: Dark-energy equation-of-state value at scale factor one.
        wa: CPL dark-energy evolution coefficient.

    Returns:
        Lookback times in Gyr with the shape of ``z``.

    Examples:
        >>> import torch
        >>> from torchsky.cosmology import lookback_time
        >>> lookback_time(torch.tensor([0.0])).tolist()
        [0.0]
    """
    z_t = _floating_tensor(z)
    if z_t.numel() == 0:
        return z_t
    z_nonneg = _nonnegative_redshift(z_t)
    nodes, weights = _gauss_legendre_unit(_GL_NODES, z_t.device, z_t.dtype)
    u_max = torch.log1p(z_nonneg).unsqueeze(-1)
    u = u_max * nodes
    integrand = 1.0 / e_of_z(torch.expm1(u), om0=om0, ok0=ok0, w0=w0, wa=wa)
    integral = u_max.squeeze(-1) * torch.sum(integrand * weights, dim=-1)
    return _hubble_time_gyr(H0, device=z_t.device, dtype=z_t.dtype) * integral


def age(
    z: Tensor,
    *,
    H0: Tensor | float = PLANCK18_H0,
    om0: Tensor | float = PLANCK18_OM0,
    ok0: Tensor | float = 0.0,
    w0: Tensor | float = PLANCK18_W0,
    wa: Tensor | float = 0.0,
) -> Tensor:
    """Age of the universe at redshift ``z``, ``t_H int_z^inf dz'/((1+z')E)``, in Gyr.

    In scale factor the integral is ``int_0^{a(z)} da / (a E)``. Substituting
    ``a = a(z) s^2`` makes the integrand ``2 a s / sqrt(a^2 E^2)`` smooth at
    the big bang whenever matter dominates there, so the 64-node
    Gauss-Legendre rule converges to machine precision. ``a^2 E^2`` is
    evaluated directly so no power of ``1+z`` overflows. Requires
    ``Om0 > 0``; negative ``z`` is treated as zero. Radiation is neglected:
    with Planck parameters it shortens the present age by about 0.02 per
    cent, and much more at high redshift.

    Args:
        z: Redshift tensor (``age(0)`` is the present age).
        H0: Present-day Hubble parameter in km/s/Mpc.
        om0: Present-day matter density parameter.
        ok0: Present-day curvature density parameter.
        w0: Dark-energy equation-of-state value at scale factor one.
        wa: CPL dark-energy evolution coefficient.

    Returns:
        Ages in Gyr with the shape of ``z``.

    Examples:
        >>> import torch
        >>> from torchsky.cosmology import age
        >>> round(float(age(torch.tensor(0.0, dtype=torch.float64), H0=70.0, om0=1.0)), 6)
        9.312307
    """
    z_t = _floating_tensor(z)
    if z_t.numel() == 0:
        return z_t
    z_nonneg = _nonnegative_redshift(z_t)
    om = _as_parameter(om0, device=z_t.device, dtype=z_t.dtype)
    if not bool(torch.isfinite(om).all()) or bool((om <= 0.0).any()):
        raise ValueError("age requires Om0 > 0 (matter-dominated big bang)")
    ok = _as_parameter(ok0, device=z_t.device, dtype=z_t.dtype)
    w = _as_parameter(w0, device=z_t.device, dtype=z_t.dtype)
    w_a = _as_parameter(wa, device=z_t.device, dtype=z_t.dtype)
    nodes, weights = _gauss_legendre_unit(_GL_NODES, z_t.device, z_t.dtype)
    a_end = (1.0 / (1.0 + z_nonneg)).unsqueeze(-1)
    a = a_end * nodes.square()
    # a^2 E^2 = Om/a + Ok + Ode a^(-1-3(w0+wa)) exp(-3 wa (1-a)).
    dark_energy = (
        (1.0 - om - ok)
        * torch.pow(a, -1.0 - 3.0 * (w + w_a))
        * torch.exp(-3.0 * w_a * (1.0 - a))
    )
    scaled_squared = om / a + ok + dark_energy
    if not bool(torch.isfinite(scaled_squared).all()) or bool(
        (scaled_squared <= 0.0).any()
    ):
        raise ValueError("FLRW parameters produce non-positive E(z)^2")
    integrand = 2.0 * a_end * nodes / torch.sqrt(scaled_squared)
    integral = torch.sum(integrand * weights, dim=-1)
    return _hubble_time_gyr(H0, device=z_t.device, dtype=z_t.dtype) * integral


def _validated_unit_xyz(value: Tensor, name: str) -> Tensor:
    unit = torch.as_tensor(value)
    if not unit.is_floating_point():
        unit = unit.to(dtype=torch.get_default_dtype())
    if unit.ndim < 1 or unit.shape[-1] != 3:
        raise ValueError(f"{name} must have trailing shape [..., 3]")
    if not bool(torch.isfinite(unit).all()):
        raise ValueError(f"{name} must be finite")
    norm = torch.linalg.vector_norm(unit, dim=-1)
    tolerance = 5.0e-6 if unit.dtype == torch.float32 else 1.0e-10
    if bool((torch.abs(norm - 1.0) > tolerance).any()):
        raise ValueError(f"{name} must contain unit vectors")
    return unit


def _sqrt_with_zero_subgradient(value: Tensor) -> Tensor:
    """Square root of a non-negative value with a defined gradient at zero."""
    nonnegative = value.clamp_min(0.0)
    positive = nonnegative > 0.0
    safe_input = torch.where(positive, nonnegative, torch.ones_like(nonnegative))
    return torch.where(positive, torch.sqrt(safe_input), torch.zeros_like(nonnegative))


def comoving_position(
    unit_xyz: Tensor,
    z: Tensor,
    cosmology: FLRW = FLRW(),
) -> Tensor:
    """Flat-FLRW Cartesian comoving position ``chi(z) * unit_xyz`` in Mpc.

    A curved spatial slice has no global Euclidean Cartesian representation;
    use :func:`comoving_separation` there.
    """
    unit = _validated_unit_xyz(unit_xyz, "unit_xyz")
    ok = torch.as_tensor(cosmology.Ok0)
    if not bool(torch.isfinite(ok).all()) or bool((torch.abs(ok) > 1.0e-12).any()):
        raise ValueError("comoving_position is defined only for Ok0 = 0")
    z_t = _as_tensor(z, device=unit.device, dtype=unit.dtype)
    shape = torch.broadcast_shapes(unit.shape[:-1], z_t.shape)
    unit_b = torch.broadcast_to(unit, shape + (3,))
    z_b = torch.broadcast_to(z_t, shape)
    distance = cosmology.comoving_distance(z_b)
    return distance.unsqueeze(-1) * unit_b


def comoving_separation(
    unit_xyz_a: Tensor,
    z_a: Tensor,
    unit_xyz_b: Tensor,
    z_b: Tensor,
    cosmology: FLRW = FLRW(),
) -> Tensor:
    """Constant-curvature FLRW spatial geodesic separation in Mpc."""
    unit_a = _validated_unit_xyz(unit_xyz_a, "unit_xyz_a")
    unit_b = _to(
        _validated_unit_xyz(unit_xyz_b, "unit_xyz_b"),
        device=unit_a.device,
        dtype=unit_a.dtype,
    )
    z_a_t = _as_tensor(z_a, device=unit_a.device, dtype=unit_a.dtype)
    z_b_t = _as_tensor(z_b, device=unit_a.device, dtype=unit_a.dtype)
    shape = torch.broadcast_shapes(
        unit_a.shape[:-1], unit_b.shape[:-1], z_a_t.shape, z_b_t.shape
    )
    unit_a = torch.broadcast_to(unit_a, shape + (3,))
    unit_b = torch.broadcast_to(unit_b, shape + (3,))
    z_a_t = torch.broadcast_to(z_a_t, shape)
    z_b_t = torch.broadcast_to(z_b_t, shape)
    # ||a-b||^2 / 4 = (1-cos(theta)) / 2 for unit vectors. This avoids the
    # catastrophic cancellation in ``1 - dot(a, b)`` for close float32 pairs.
    half_chord_squared = (
        0.25 * torch.sum(torch.square(unit_a - unit_b), dim=-1)
    ).clamp(0.0, 1.0)
    d_a = cosmology.comoving_distance(z_a_t)
    d_b = cosmology.comoving_distance(z_b_t)
    flat_squared = torch.square(d_a - d_b) + 4.0 * d_a * d_b * half_chord_squared

    ok = _as_parameter(cosmology.Ok0, device=unit_a.device, dtype=unit_a.dtype)
    d_h = _hubble_distance_mpc(cosmology.H0, device=unit_a.device, dtype=unit_a.dtype)
    abs_ok = torch.abs(ok)
    epsilon = torch.as_tensor(torch.finfo(unit_a.dtype).eps, device=unit_a.device)
    safe_abs_ok = abs_ok.clamp_min(epsilon)
    sqrt_abs_ok = torch.sqrt(safe_abs_ok)
    radius = d_h / sqrt_abs_ok
    radial_a = d_a / radius
    radial_b = d_b / radius

    # Cancellation-free half-angle forms of the constant-curvature law of
    # cosines. They retain close angular separations in float32.
    open_half_squared = torch.square(torch.sinh(0.5 * (radial_a - radial_b))) + (
        torch.sinh(radial_a) * torch.sinh(radial_b) * half_chord_squared
    )
    open_distance = (
        2.0 * radius * torch.asinh(_sqrt_with_zero_subgradient(open_half_squared))
    )

    closed_half_squared = (
        torch.square(torch.sin(0.5 * (radial_a - radial_b)))
        + torch.sin(radial_a) * torch.sin(radial_b) * half_chord_squared
    ).clamp(0.0, 1.0)
    closed_half = _sqrt_with_zero_subgradient(closed_half_squared)
    closed_complement = _sqrt_with_zero_subgradient(1.0 - closed_half_squared)
    closed_distance = 2.0 * radius * torch.atan2(closed_half, closed_complement)

    # First curvature correction to the geodesic law of cosines. Besides
    # avoiding a 0/0 branch, this preserves d/dOk0 at the flat limit.
    series_squared = flat_squared + (
        ok
        * torch.square(d_a * d_b)
        * (4.0 * half_chord_squared * (1.0 - half_chord_squared))
        / (3.0 * torch.square(d_h))
    )
    series_distance = _sqrt_with_zero_subgradient(series_squared)
    curved_distance = torch.where(ok > 0.0, open_distance, closed_distance)
    radial_scale_squared = abs_ok * torch.maximum(
        torch.square(d_a / d_h), torch.square(d_b / d_h)
    )
    use_series = radial_scale_squared <= torch.sqrt(epsilon)
    return torch.where(use_series, series_distance, curved_distance)


def alcock_paczynski_scaling(
    z: Tensor,
    cosmology: FLRW,
    reference: FLRW,
) -> APScaling:
    """AP geometry at strictly positive redshift.

    The alpha values are relative to ``reference``; ``f_ap`` is the absolute
    ``D_M(z) H(z) / c`` observable for ``cosmology``.
    """
    z_t = _floating_tensor(z)
    if z_t.numel() and (
        not bool(torch.isfinite(z_t).all()) or bool((z_t <= 0.0).any())
    ):
        raise ValueError("Alcock-Paczynski scaling requires finite z > 0")
    d_m = cosmology.transverse_comoving_distance(z_t)
    d_m_reference = reference.transverse_comoving_distance(z_t)
    h = cosmology.hubble_parameter(z_t)
    h_reference = reference.hubble_parameter(z_t)
    return APScaling(
        alpha_parallel=h_reference / h,
        alpha_perpendicular=d_m / d_m_reference,
        f_ap=d_m * h / SPEED_OF_LIGHT_KM_S,
    )


def distance_modulus(
    z: Tensor,
    *,
    H0: Tensor | float = PLANCK18_H0,
    om0: Tensor | float = PLANCK18_OM0,
    ok0: Tensor | float = 0.0,
    w0: Tensor | float = PLANCK18_W0,
    wa: Tensor | float = 0.0,
) -> Tensor:
    """``μ = 5 log10(D_L / 10 pc)`` with ``D_L`` in Mpc.

    The logarithm argument is floored so ``μ(0)`` stays finite; ``D_L(0)``
    itself is zero.
    """
    d_l = luminosity_distance(z, H0=H0, om0=om0, ok0=ok0, w0=w0, wa=wa)
    return 5.0 * torch.log10((d_l * 1.0e6 / 10.0).clamp(min=1e-30))


__all__ = [
    "APScaling",
    "FLRW",
    "FlatWCDM",
    "HUBBLE_TIME_GYR",
    "PLANCK18_H0",
    "PLANCK18_OM0",
    "PLANCK18_W0",
    "SPEED_OF_LIGHT_KM_S",
    "age",
    "alcock_paczynski_scaling",
    "angular_diameter_distance",
    "comoving_distance",
    "comoving_position",
    "comoving_separation",
    "differential_comoving_volume",
    "distance_modulus",
    "e_of_z",
    "hubble_parameter",
    "lookback_time",
    "luminosity_distance",
    "transverse_comoving_distance",
]
