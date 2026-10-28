"""Milky Way dust laws (CCM89 / O'Donnell94) on torch tensors.

Wavelengths are Angstrom-native internally. Pass ``wavelength_unit='nm'`` at
the boundary. ``A(λ)`` is in magnitudes; transmission is ``10**(-0.4 A)``.

The infrared power law is extrapolated below the CCM89 ``x=0.3`` cutoff so
broadband work stays defined into the far-IR. Optical and UV pieces follow
Cardelli, Clayton & Mathis (1989); ``odonnell94`` replaces the optical
polynomials only.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

import torch
from torch import Tensor

WavelengthUnit = Literal["angstrom", "nm"]
ExtinctionLaw = Literal[
    "ccm89",
    "odonnell94",
    "odonnell",
    "f99",
    "fitzpatrick99",
    "calzetti",
]

# Fitzpatrick (1999) FM90 UV + optical/IR spline, matching kbarbary/extinction
# and IDL FM_UNRED. Knots are inverse microns. Spline values are E(λ-V)/E(B-V).
_F99_X0 = 4.596
_F99_GAMMA = 0.99
_F99_C3 = 3.23
_F99_C4 = 0.41
_F99_C5 = 5.9
_F99_UV_JOIN = 1.0e4 / 2700.0
_F99_XKNOTS = (
    0.0,
    1.0e4 / 26500.0,
    1.0e4 / 12200.0,
    1.0e4 / 6000.0,
    1.0e4 / 5470.0,
    1.0e4 / 4670.0,
    1.0e4 / 4110.0,
    1.0e4 / 2700.0,
    1.0e4 / 2600.0,
)


def ccm89_ab(x_um_inv: Tensor) -> tuple[Tensor, Tensor]:
    """CCM89 ``a(x), b(x)`` for inverse-micron ``x = 1 / λ_μm``.

    Regions follow Cardelli, Clayton & Mathis (1989): IR ``x<1.1`` (power
    law extrapolated below ``x=0.3``), optical ``1.1<=x<3.3``, UV
    ``3.3<=x<8``, far-UV ``8<=x<=10``. Beyond ``x=10`` the far-UV polynomials
    are evaluated at the ``x=10`` boundary; the law is not defined there.
    """
    x = torch.as_tensor(x_um_inv).clamp_min(1e-3)

    a_ir = 0.574 * x**1.61
    b_ir = -0.527 * x**1.61

    opt = (x >= 1.1) & (x < 3.3)
    y = x - 1.82
    # CCM89 eq. (3a): optical ``y^2`` coefficient is 0.50447.
    a_opt = (
        1.0
        + 0.17699 * y
        - 0.50447 * y**2
        - 0.02427 * y**3
        + 0.72085 * y**4
        + 0.01979 * y**5
        - 0.77530 * y**6
        + 0.32999 * y**7
    )
    b_opt = (
        1.41338 * y
        + 2.28305 * y**2
        + 1.07233 * y**3
        - 5.38434 * y**4
        - 0.62251 * y**5
        + 5.30260 * y**6
        - 2.09002 * y**7
    )

    uv = (x >= 3.3) & (x < 8.0)
    fa = torch.where(
        x >= 5.9,
        -0.04473 * (x - 5.9) ** 2 - 0.009779 * (x - 5.9) ** 3,
        torch.zeros_like(x),
    )
    fb = torch.where(
        x >= 5.9,
        0.2130 * (x - 5.9) ** 2 + 0.1207 * (x - 5.9) ** 3,
        torch.zeros_like(x),
    )
    a_uv = 1.752 - 0.316 * x - 0.104 / ((x - 4.67) ** 2 + 0.341) + fa
    b_uv = -3.090 + 1.825 * x + 1.206 / ((x - 4.62) ** 2 + 0.263) + fb

    fuv = x >= 8.0
    y_fuv = x.clamp(max=10.0) - 8.0
    a_fuv = -1.073 - 0.628 * y_fuv + 0.137 * y_fuv**2 - 0.070 * y_fuv**3
    b_fuv = 13.670 + 4.257 * y_fuv - 0.420 * y_fuv**2 + 0.374 * y_fuv**3

    # The four regions partition x > 0, so the IR branch is the remainder.
    a = torch.where(
        fuv, a_fuv, torch.where(uv, a_uv, torch.where(opt, a_opt, a_ir))
    )
    b = torch.where(
        fuv, b_fuv, torch.where(uv, b_uv, torch.where(opt, b_opt, b_ir))
    )
    return a, b


def odonnell94_ab(x_um_inv: Tensor) -> tuple[Tensor, Tensor]:
    """O'Donnell (1994) optical update; CCM89 in UV and IR."""
    a, b = ccm89_ab(x_um_inv)
    x = torch.as_tensor(x_um_inv).clamp_min(1e-3)
    opt = (x >= 1.1) & (x < 3.3)
    y = x - 1.82
    a_opt = (
        1.0
        + 0.104 * y
        - 0.609 * y**2
        + 0.701 * y**3
        + 1.137 * y**4
        - 1.718 * y**5
        - 0.827 * y**6
        + 1.647 * y**7
        - 0.505 * y**8
    )
    b_opt = (
        1.952 * y
        + 2.908 * y**2
        - 3.989 * y**3
        - 7.985 * y**4
        + 11.102 * y**5
        + 5.491 * y**6
        - 10.805 * y**7
        + 3.347 * y**8
    )
    return torch.where(opt, a_opt, a), torch.where(opt, b_opt, b)


def _normalize_law(law: str) -> str:
    name = str(law).lower()
    if name in {"odonnell", "odonnell94"}:
        return "odonnell94"
    if name == "ccm89":
        return "ccm89"
    if name in {"f99", "fitzpatrick99", "fitzpatrick"}:
        return "f99"
    if name in {"calzetti", "calzetti00"}:
        return "calzetti"
    raise ValueError(f"Unknown extinction law: {law!r}")


def _f99_kknots(r_v: Tensor) -> Tensor:
    """F99 spline knot values ``E(λ-V)/E(B-V)`` for ``r_v``; last dim is 9."""
    rv = r_v.unsqueeze(-1)
    rv2 = rv * rv
    k0 = -rv
    k1 = 0.26469 * rv / 3.1 - rv
    k2 = 0.82925 * rv / 3.1 - rv
    k3 = -0.422809 + 1.00270 * rv + 2.13572e-04 * rv2 - rv
    k4 = -5.13540e-02 + 1.00216 * rv - 7.35778e-05 * rv2 - rv
    k5 = 0.700127 + 1.00184 * rv - 3.32598e-05 * rv2 - rv
    k6 = (
        1.19456
        + 1.01707 * rv
        - 5.46959e-03 * rv2
        + 7.97809e-04 * rv2 * rv
        - 4.45636e-05 * rv2 * rv2
        - rv
    )
    x_uv = torch.as_tensor(
        _F99_XKNOTS[7:9], device=r_v.device, dtype=r_v.dtype
    ).expand(*rv.shape[:-1], 2)
    return torch.cat([k0, k1, k2, k3, k4, k5, k6, _f99_uv_k(x_uv, rv)], dim=-1)


@lru_cache(maxsize=8)
def _f99_spline_matrix(device: torch.device, dtype: torch.dtype) -> Tensor:
    """7×7 natural-cubic system for the F99 knots (``m0 = m8 = 0``).

    Cached because the knot abscissae are constants; callers solve against it
    instead of forming an explicit inverse.
    """
    xk = torch.tensor(_F99_XKNOTS, device=device, dtype=dtype)
    h = xk[1:] - xk[:-1]
    matrix = torch.zeros((7, 7), device=device, dtype=dtype)
    for row, i in enumerate(range(1, 8)):
        matrix[row, row] = (h[i - 1] + h[i]) / 3.0
        if row > 0:
            matrix[row, row - 1] = h[i - 1] / 6.0
        if row < 6:
            matrix[row, row + 1] = h[i] / 6.0
    return matrix


def _natural_cubic_eval(x: Tensor, y_knots: Tensor) -> Tensor:
    """Evaluate the F99 natural cubic spline at inverse-micron ``x``."""
    xk = torch.tensor(_F99_XKNOTS, device=x.device, dtype=x.dtype)
    h = xk[1:] - xk[:-1]
    matrix = _f99_spline_matrix(x.device, x.dtype)
    rhs = (y_knots[..., 2:] - y_knots[..., 1:-1]) / h[1:] - (
        y_knots[..., 1:-1] - y_knots[..., :-2]
    ) / h[:-1]
    m_int = torch.linalg.solve(matrix, rhs.transpose(-1, -2)).transpose(-1, -2)
    zeros = torch.zeros_like(y_knots[..., :1])
    m = torch.cat([zeros, m_int, zeros], dim=-1)
    idx = torch.searchsorted(xk, x, right=True).clamp(1, 8)
    i = idx - 1
    batch_idx = torch.arange(x.shape[0], device=x.device)
    hi = h[i]
    t = x - xk[i]
    yi = y_knots[batch_idx, i]
    yip = y_knots[batch_idx, i + 1]
    mi = m[batch_idx, i]
    mip = m[batch_idx, i + 1]
    return (
        mi * (hi - t).pow(3) / (6.0 * hi)
        + mip * t.pow(3) / (6.0 * hi)
        + (yi / hi - mi * hi / 6.0) * (hi - t)
        + (yip / hi - mip * hi / 6.0) * t
    )


def _f99_uv_k(x: Tensor, r_v: Tensor) -> Tensor:
    c2 = -0.824 + 4.717 / r_v
    c1 = 2.030 - 3.007 * c2
    x2 = x * x
    y = x2 - (_F99_X0 * _F99_X0)
    d = x2 / (y * y + x2 * (_F99_GAMMA * _F99_GAMMA))
    k = c1 + c2 * x + _F99_C3 * d
    y_fuv = x - _F99_C5
    k = k + torch.where(
        x >= _F99_C5,
        _F99_C4 * (0.5392 * y_fuv.pow(2) + 0.05644 * y_fuv.pow(3)),
        torch.zeros_like(x),
    )
    return k


def f99_a_lambda(wave_angstrom: Tensor, a_v: Tensor, r_v: Tensor) -> Tensor:
    """Fitzpatrick (1999) ``A(λ)``; wavelengths in Angstrom."""
    x = 10000.0 / wave_angstrom.clamp_min(1e-6)
    x_b, a_b, r_b = torch.broadcast_tensors(x, a_v, r_v)
    x_flat = x_b.reshape(-1)
    a_flat = a_b.reshape(-1)
    r_flat = r_b.reshape(-1)
    k_knots = _f99_kknots(r_flat)
    k_opt = _natural_cubic_eval(x_flat, k_knots)
    a_opt = (a_flat / r_flat) * (k_opt + r_flat)
    a_uv = a_flat * (1.0 + _f99_uv_k(x_flat, r_flat) / r_flat)
    out = torch.where(x_flat < _F99_UV_JOIN, a_opt, a_uv)
    return out.reshape(x_b.shape)


def calzetti_k_ebv(wave_angstrom: Tensor, r_v: Tensor) -> Tensor:
    """Calzetti (2000) ``k(λ) = A(λ)/E(B-V)``. ``λ`` clamped to 0.12–2.20 μm."""
    lam_um = (wave_angstrom / 1.0e4).clamp(0.12, 2.20)
    k_opt = (
        2.659
        * (-2.156 + 1.509 / lam_um - 0.198 / lam_um.pow(2) + 0.011 / lam_um.pow(3))
        + r_v
    )
    k_ir = 2.659 * (-1.857 + 1.040 / lam_um) + r_v
    return torch.where(lam_um <= 0.63, k_opt, k_ir)


def calzetti_a_lambda(wave_angstrom: Tensor, a_v: Tensor, r_v: Tensor) -> Tensor:
    """Calzetti (2000) ``A(λ)``. Default ``R_V`` for the law is 4.05."""
    k = calzetti_k_ebv(wave_angstrom, r_v)
    return (k / r_v) * a_v


def _to_angstrom(wave: Tensor, wavelength_unit: WavelengthUnit) -> Tensor:
    if wavelength_unit not in ("angstrom", "nm"):
        raise ValueError("wavelength_unit must be 'angstrom' or 'nm'")
    values = torch.as_tensor(wave)
    if wavelength_unit == "nm":
        return values * 10.0
    return values


def _compute_ab(wave_angstrom: Tensor, law: str) -> tuple[Tensor, Tensor]:
    x = 10000.0 / torch.clamp(wave_angstrom, min=1e-6)
    if law == "ccm89":
        return ccm89_ab(x)
    if law == "odonnell94":
        return odonnell94_ab(x)
    raise ValueError(f"{law!r} is not a polynomial extinction law")


def _align_scale(scale: Tensor, wave: Tensor) -> Tensor:
    """Broadcast a per-object scale against a spectral (or spatial) grid.

    Batch axes are leading. A 1-D ``[B]`` scale against a 1-D ``[L]`` grid
    becomes ``[B, 1]``, not a last-axis alignment of ``B`` with ``L``.
    """
    aligned = scale.to(device=wave.device, dtype=wave.dtype)
    if aligned.ndim == 0:
        return aligned
    if aligned.ndim == wave.ndim and aligned.shape[-1] != 1:
        aligned = aligned.unsqueeze(-1)
    while aligned.ndim < wave.ndim:
        aligned = aligned.unsqueeze(-1)
    return aligned


def a_lambda(
    wave: Tensor,
    *,
    ebv: Tensor | float | None = None,
    a_v: Tensor | float | None = None,
    r_v: Tensor | float = 3.1,
    law: ExtinctionLaw = "ccm89",
    wavelength_unit: WavelengthUnit = "angstrom",
) -> Tensor:
    """Extinction in magnitudes at ``wave``.

    Supply exactly one of ``ebv`` or ``a_v``. ``A_V = R_V E(B-V)``.
    """
    if (ebv is None) == (a_v is None):
        raise ValueError("provide exactly one of ebv or a_v")
    law_name = _normalize_law(law)
    wave_a = _to_angstrom(wave, wavelength_unit)
    scale = a_v if a_v is not None else ebv
    scale_t = _align_scale(torch.as_tensor(scale), wave_a)
    r_v_t = _align_scale(torch.as_tensor(r_v), wave_a)
    a_v_t = scale_t if a_v is not None else r_v_t * scale_t
    if law_name == "f99":
        return f99_a_lambda(wave_a, a_v_t, r_v_t)
    if law_name == "calzetti":
        return calzetti_a_lambda(wave_a, a_v_t, r_v_t)
    a, b = _compute_ab(wave_a, law_name)
    return (a + b / r_v_t) * a_v_t


def ccm89(
    wave_angstrom: Tensor,
    a_v: Tensor,
    r_v: Tensor | float | None = None,
    *,
    wavelength_unit: WavelengthUnit = "angstrom",
) -> Tensor:
    """CCM89 ``A(λ)`` for ``A_V``; wavelengths default to Angstrom."""
    return a_lambda(
        wave_angstrom,
        a_v=a_v,
        r_v=3.1 if r_v is None else r_v,
        law="ccm89",
        wavelength_unit=wavelength_unit,
    )


def odonnell94(
    wave_angstrom: Tensor,
    a_v: Tensor,
    r_v: Tensor | float | None = None,
    *,
    wavelength_unit: WavelengthUnit = "angstrom",
) -> Tensor:
    """O'Donnell (1994) ``A(λ)`` for ``A_V``; wavelengths default to Angstrom."""
    return a_lambda(
        wave_angstrom,
        a_v=a_v,
        r_v=3.1 if r_v is None else r_v,
        law="odonnell94",
        wavelength_unit=wavelength_unit,
    )


def fitzpatrick99(
    wave_angstrom: Tensor,
    a_v: Tensor,
    r_v: Tensor | float | None = None,
    *,
    wavelength_unit: WavelengthUnit = "angstrom",
) -> Tensor:
    """Fitzpatrick (1999) ``A(λ)`` for ``A_V``; wavelengths default to Angstrom."""
    return a_lambda(
        wave_angstrom,
        a_v=a_v,
        r_v=3.1 if r_v is None else r_v,
        law="f99",
        wavelength_unit=wavelength_unit,
    )


def calzetti(
    wave_angstrom: Tensor,
    a_v: Tensor,
    r_v: Tensor | float | None = None,
    *,
    wavelength_unit: WavelengthUnit = "angstrom",
) -> Tensor:
    """Calzetti (2000) starburst ``A(λ)``. Default ``R_V = 4.05``."""
    return a_lambda(
        wave_angstrom,
        a_v=a_v,
        r_v=4.05 if r_v is None else r_v,
        law="calzetti",
        wavelength_unit=wavelength_unit,
    )


def k_lambda(
    wave: Tensor,
    *,
    law: ExtinctionLaw = "ccm89",
    r_v: Tensor | float = 3.1,
    wavelength_unit: WavelengthUnit = "angstrom",
) -> Tensor:
    """Named ``k(λ) = A(λ)/A_V`` so ``k(λ_V) ≈ 1``. Host screens use this at ``λ_rest``."""
    return a_lambda(
        wave,
        a_v=1.0,
        r_v=r_v,
        law=law,
        wavelength_unit=wavelength_unit,
    )


def transmission(
    wave: Tensor,
    ebv: Tensor | float,
    *,
    r_v: Tensor | float = 3.1,
    law: ExtinctionLaw = "ccm89",
    wavelength_unit: WavelengthUnit = "angstrom",
) -> Tensor:
    """Multiplicative flux attenuation ``10**(-0.4 A)``.

    ``A`` is clamped at zero: a dust screen cannot emit.
    """
    mag = a_lambda(
        wave, ebv=ebv, r_v=r_v, law=law, wavelength_unit=wavelength_unit
    ).clamp_min(0.0)
    return torch.pow(10.0, (-0.4 * mag).clamp(min=-30.0))


def apply_extinction(
    flux: Tensor,
    wavelength: Tensor | None = None,
    ext_params: Tensor | None = None,
    extinction_law_name: str = "ccm89",
    r_v: float = 3.1,
    *,
    wavelength_unit: WavelengthUnit = "angstrom",
) -> Tensor:
    """Apply a dust screen: ``ext_params`` is ``E(B-V)`` broadcast against flux.

    Args:
        flux: Flux-density tensor with wavelength on the trailing axis.
        wavelength: Wavelength samples in ``wavelength_unit``.
        ext_params: Color excess ``E(B-V)``, broadcastable against ``flux``.
        extinction_law_name: Supported extinction-law name.
        r_v: Total-to-selective extinction ratio ``A(V) / E(B-V)``.
        wavelength_unit: ``"angstrom"`` or ``"nm"``.

    Returns:
        Attenuated flux with the same shape, dtype, and device as ``flux``.

    Examples:
        >>> import torch
        >>> from torchsky.extinction import apply_extinction
        >>> apply_extinction(torch.ones(1), torch.tensor([5500.0]), torch.tensor(0.0)).tolist()
        [1.0]
    """
    if wavelength is None or ext_params is None:
        raise TypeError("apply_extinction requires flux, wavelength, and ext_params")
    flux_t = torch.as_tensor(flux)
    wave = torch.as_tensor(wavelength).to(device=flux_t.device, dtype=flux_t.dtype)
    ebv = torch.as_tensor(ext_params).to(device=flux_t.device, dtype=flux_t.dtype)
    if ebv.ndim == flux_t.ndim - 1:
        ebv = ebv.unsqueeze(-1)
    factor = transmission(
        wave,
        ebv,
        r_v=r_v,
        law=extinction_law_name,  # type: ignore[arg-type]
        wavelength_unit=wavelength_unit,
    )
    return flux_t * factor

__all__ = [
    "ExtinctionLaw",
    "WavelengthUnit",
    "a_lambda",
    "apply_extinction",
    "calzetti",
    "calzetti_a_lambda",
    "calzetti_k_ebv",
    "ccm89",
    "ccm89_ab",
    "f99_a_lambda",
    "fitzpatrick99",
    "k_lambda",
    "odonnell94",
    "odonnell94_ab",
    "transmission",
]
