"""Milky Way and starburst dust laws on torch tensors.

Wavelengths are Angstrom-native internally. Pass ``wavelength_unit='nm'`` at
the boundary. ``A(λ)`` is in magnitudes; transmission is ``10**(-0.4 A)``.

The infrared power law is extrapolated below the CCM89 ``x=0.3`` cutoff so
broadband work stays defined into the far-IR. Optical and UV pieces follow
Cardelli, Clayton & Mathis (1989); ``odonnell94`` replaces the optical
polynomials only.

``g23`` (Gordon et al. 2023), ``f19`` (Fitzpatrick et al. 2019), and
``gcc09`` (Gordon, Cartledge & Clayton 2009) use the published coefficients as
implemented in ``dust_extinction`` (K. Gordon, BSD 3-Clause), and fail closed
outside their calibrated wavelength and ``R_V`` ranges instead of
extrapolating.
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
    "g23",
    "f19",
    "gcc09",
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
    a = torch.where(fuv, a_fuv, torch.where(uv, a_uv, torch.where(opt, a_opt, a_ir)))
    b = torch.where(fuv, b_fuv, torch.where(uv, b_uv, torch.where(opt, b_opt, b_ir)))
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
    if name in {"g23", "gordon23"}:
        return "g23"
    if name in {"f19", "fitzpatrick19"}:
        return "f19"
    if name in {"gcc09", "gordon09"}:
        return "gcc09"
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
    x_uv = torch.as_tensor(_F99_XKNOTS[7:9], device=r_v.device, dtype=r_v.dtype).expand(
        *rv.shape[:-1], 2
    )
    return torch.cat([k0, k1, k2, k3, k4, k5, k6, _f99_uv_k(x_uv, rv)], dim=-1)


@lru_cache(maxsize=8)
def _f99_spline_matrix(device: torch.device, dtype: torch.dtype) -> Tensor:
    """7×7 natural-cubic system for the F99 knots (``m0 = m8 = 0``).

    Cached because the knot abscissae are constants; callers solve against it
    instead of forming an explicit inverse.
    """
    with torch.inference_mode(False):
        return _f99_spline_matrix_build(device, dtype)


def _f99_spline_matrix_build(device: torch.device, dtype: torch.dtype) -> Tensor:
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


def _check_range(
    x: Tensor,
    r_v: Tensor,
    x_range: tuple[float, float],
    rv_range: tuple[float, float],
    law: str,
) -> None:
    """Fail closed outside a law's calibrated domain (no extrapolation)."""
    if x.numel() and bool(((x < x_range[0]) | (x > x_range[1])).any()):
        raise ValueError(
            f"{law} is defined for {1e4 / x_range[1]:.0f}-{1e4 / x_range[0]:.0f} Angstrom"
        )
    if r_v.numel() and bool(((r_v < rv_range[0]) | (r_v > rv_range[1])).any()):
        raise ValueError(f"{law} is defined for {rv_range[0]} <= R_V <= {rv_range[1]}")


def _smoothstep(value: Tensor, low: float, high: float) -> Tensor:
    """First-order smoothstep ``3s^2 - 2s^3`` on ``[low, high]``."""
    s = ((value - low) / (high - low)).clamp(0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)


def _drude(x: Tensor, amplitude: float, center: float, fwhm: float) -> Tensor:
    ratio = fwhm / center
    return amplitude * ratio**2 / ((x / center - center / x) ** 2 + ratio**2)


def _modified_drude(
    wave_um: Tensor, amplitude: float, center: float, fwhm: float, asym: float
) -> Tensor:
    gamma = 2.0 * fwhm / (1.0 + torch.exp(asym * (wave_um - center)))
    return (
        amplitude
        * (gamma / center) ** 2
        / ((wave_um / center - center / wave_um) ** 2 + (gamma / center) ** 2)
    )


def _fm90(
    x: Tensor, c1: float, c2: float, c3: float, c4: float, x0: float, gamma: float
) -> Tensor:
    x2 = x * x
    value = c1 + c2 * x + c3 * x2 / ((x2 - x0 * x0) ** 2 + x2 * gamma * gamma)
    y = x - 5.9
    return value + torch.where(
        x >= 5.9, c4 * (0.5392 * y * y + 0.05644 * y**3), torch.zeros_like(x)
    )


# Gordon et al. (2023, ApJ 950, 86), Table 3, as in dust_extinction.G23.
_G23_IR_A = (0.38526, 1.68467, 0.78791, 4.30578, 4.78338)
_G23_IR_SIL = ((0.06652, 9.8434, 2.21205, -0.24703), (0.0267, 19.58294, 17.0, -0.27))
_G23_OPT_A = (-0.35848, 0.7122, 0.08746, -0.05403, 0.00674)
_G23_OPT_B = (0.12354, -2.68335, 2.01901, -0.39299, 0.03355)
_G23_ISS = ((2.288, 0.243), (2.054, 0.179), (1.587, 0.243))
_G23_ISS_A = (0.03893, 0.02965, 0.01747)
_G23_ISS_B = (0.18453, 0.19728, 0.1713)
_G23_UV_A = (0.81297, 0.2775, 1.06295, 0.11303, 4.60, 0.99)
_G23_UV_B = (-2.97868, 1.89808, 3.10334, 0.65484, 4.60, 0.99)
#: Divides the curve to put the F19 ground-based photometry on the HST/STIS
#: scale (Gordon et al. 2023, sec. 5; dust_extinction ``renorm=True``).
_G23_NORM = 0.9854


def _g23_ir_a(x: Tensor) -> Tensor:
    scale, alpha1, alpha2, swave, swidth = _G23_IR_A
    wave = 1.0 / x
    first = scale * wave ** (-alpha1)
    second = scale * swave ** (alpha2 - alpha1) * wave ** (-alpha2)
    weight = _smoothstep(wave, swave - 0.5 * swidth, swave + 0.5 * swidth)
    value = first * (1.0 - weight) + second * weight
    for amplitude, center, fwhm, asym in _G23_IR_SIL:
        value = value + _modified_drude(wave, amplitude, center, fwhm, asym)
    return value


def _g23_opt(x: Tensor, poly: tuple[float, ...], iss: tuple[float, ...]) -> Tensor:
    value = torch.zeros_like(x)
    for power, coefficient in enumerate(poly):
        value = value + coefficient * x**power
    for amplitude, (center, fwhm) in zip(iss, _G23_ISS, strict=True):
        value = value + _drude(x, amplitude, center, fwhm)
    return value


def _g23_ab(x_um_inv: Tensor) -> tuple[Tensor, Tensor]:
    """G23 ``a(x), b(x)`` with ``A(x)/A(V) = [a + b (1/R_V - 1/3.1)] / 0.9854``.

    Gordon, Clayton, Decleir & Fitzpatrick (2023, ApJ 950, 86): IR broken
    power law with silicate features, optical polynomial with three
    intermediate-scale features, and FM90 ultraviolet, joined by smoothstep
    weights over 0.9-1.1 um and 0.30-0.33 um.
    """
    x = torch.as_tensor(x_um_inv)
    wave = 1.0 / x
    a_ir = _g23_ir_a(x)
    b_ir = -1.01251 * x**1.06099
    a_opt = _g23_opt(x, _G23_OPT_A, _G23_ISS_A)
    b_opt = _g23_opt(x, _G23_OPT_B, _G23_ISS_B)
    a_uv = _fm90(x, *_G23_UV_A)
    b_uv = _fm90(x, *_G23_UV_B)
    w_ir = _smoothstep(wave, 0.9, 1.1)
    w_opt = _smoothstep(wave, 0.3, 0.33)
    optir = (x >= 1.0 / 1.1) & (x <= 1.0 / 0.9)
    uvopt = (x >= 1.0 / 0.33) & (x <= 1.0 / 0.3)
    a = torch.where(x < 1.0, a_ir, torch.where(x < 1.0 / 0.3, a_opt, a_uv))
    b = torch.where(x < 1.0, b_ir, torch.where(x < 1.0 / 0.3, b_opt, b_uv))
    a = torch.where(optir, (1.0 - w_ir) * a_opt + w_ir * a_ir, a)
    b = torch.where(optir, (1.0 - w_ir) * b_opt + w_ir * b_ir, b)
    a = torch.where(uvopt, (1.0 - w_opt) * a_uv + w_opt * a_opt, a)
    b = torch.where(uvopt, (1.0 - w_opt) * b_uv + w_opt * b_opt, b)
    return a, b


def g23_a_lambda(wave_angstrom: Tensor, a_v: Tensor, r_v: Tensor) -> Tensor:
    """Gordon et al. (2023) ``A(λ)``; 912 Å to 32 μm, ``2.3 <= R_V <= 5.6``."""
    x = 10000.0 / wave_angstrom
    _check_range(x, torch.as_tensor(r_v), (1.0 / 32.0, 1.0 / 0.0912), (2.3, 5.6), "g23")
    a, b = _g23_ab(x)
    return (a + b * (1.0 / r_v - 1.0 / 3.1)) / _G23_NORM * a_v


def gcc09_a_lambda(wave_angstrom: Tensor, a_v: Tensor, r_v: Tensor) -> Tensor:
    """Gordon, Cartledge & Clayton (2009, with 2014 erratum) ultraviolet ``A(λ)``.

    Defined only for 909-3030 Å (``3.3 <= x <= 11``) and ``2 <= R_V <= 6``;
    the law was not derived for the optical or infrared.
    """
    x = 10000.0 / wave_angstrom
    _check_range(x, torch.as_tensor(r_v), (3.3, 11.0), (2.0, 6.0), "gcc09")
    a = 1.894 - 0.373 * x - 0.0101 / ((x - 4.57) ** 2 + 0.0384)
    b = -3.490 + 2.057 * x + 0.706 / ((x - 4.59) ** 2 + 0.169)
    y = x - 5.9
    far = x >= 5.9
    a = a + torch.where(far, -0.110 * y * y - 0.0100 * y**3, torch.zeros_like(x))
    b = b + torch.where(far, 0.531 * y * y + 0.0544 * y**3, torch.zeros_like(x))
    return (a + b / r_v) * a_v


# Fitzpatrick et al. (2019, ApJ 886, 108), Table 3: x [1/um], k(λ-55) at
# R(55) = 3.02, and its R(55) derivative; as distributed with dust_extinction.
_F19_X = (
    0.000,
    0.455,
    0.606,
    0.800,
    1.000,
    1.100,
    1.200,
    1.250,
    1.300,
    1.350,
    1.400,
    1.450,
    1.500,
    1.550,
    1.600,
    1.650,
    1.700,
    1.750,
    1.800,
    1.818,
    1.850,
    1.900,
    1.950,
    2.000,
    2.050,
    2.100,
    2.150,
    2.200,
    2.250,
    2.273,
    2.300,
    2.350,
    2.400,
    2.450,
    2.500,
    2.550,
    2.600,
    2.650,
    2.700,
    2.750,
    2.800,
    2.850,
    2.900,
    2.950,
    3.000,
    3.100,
    3.200,
    3.300,
    3.400,
    3.500,
    3.600,
    3.700,
    3.800,
    3.900,
    4.000,
    4.100,
    4.200,
    4.300,
    4.400,
    4.500,
    4.600,
    4.700,
    4.800,
    4.900,
    5.000,
    5.100,
    5.200,
    5.300,
    5.400,
    5.500,
    5.600,
    5.700,
    5.800,
    5.900,
    6.000,
    6.100,
    6.200,
    6.300,
    6.400,
    6.500,
    6.600,
    6.700,
    6.800,
    6.900,
    7.000,
    7.100,
    7.200,
    7.300,
    7.400,
    7.500,
    7.600,
    7.700,
    7.800,
    7.900,
    8.000,
    8.100,
    8.200,
    8.300,
    8.400,
    8.500,
    8.600,
    8.700,
)
_F19_K302 = (
    -3.020,
    -2.747,
    -2.528,
    -2.222,
    -1.757,
    -1.567,
    -1.300,
    -1.216,
    -1.070,
    -0.973,
    -0.868,
    -0.750,
    -0.629,
    -0.509,
    -0.407,
    -0.320,
    -0.221,
    -0.133,
    -0.048,
    0.000,
    0.071,
    0.188,
    0.319,
    0.438,
    0.575,
    0.665,
    0.744,
    0.838,
    0.951,
    1.000,
    1.044,
    1.113,
    1.181,
    1.269,
    1.346,
    1.405,
    1.476,
    1.558,
    1.632,
    1.723,
    1.791,
    1.869,
    1.948,
    2.009,
    2.090,
    2.253,
    2.408,
    2.565,
    2.746,
    2.933,
    3.124,
    3.328,
    3.550,
    3.815,
    4.139,
    4.534,
    5.012,
    5.560,
    6.118,
    6.565,
    6.767,
    6.681,
    6.394,
    6.038,
    5.704,
    5.432,
    5.226,
    5.078,
    4.978,
    4.913,
    4.877,
    4.862,
    4.864,
    4.879,
    4.904,
    4.938,
    4.982,
    5.038,
    5.105,
    5.181,
    5.266,
    5.359,
    5.460,
    5.569,
    5.684,
    5.805,
    5.933,
    6.067,
    6.207,
    6.352,
    6.502,
    6.657,
    6.817,
    6.981,
    7.150,
    7.323,
    7.500,
    7.681,
    7.866,
    8.054,
    8.246,
    8.441,
)
_F19_DELTAK = (
    -1.000,
    -0.842,
    -0.728,
    -0.531,
    -0.360,
    -0.284,
    -0.223,
    -0.198,
    -0.173,
    -0.150,
    -0.130,
    -0.110,
    -0.096,
    -0.081,
    -0.063,
    -0.048,
    -0.032,
    -0.017,
    -0.005,
    0.000,
    0.007,
    0.013,
    0.012,
    0.010,
    0.004,
    0.003,
    0.000,
    0.002,
    0.001,
    0.000,
    -0.000,
    0.001,
    0.001,
    -0.002,
    0.000,
    -0.002,
    -0.002,
    -0.006,
    -0.009,
    -0.011,
    -0.017,
    -0.025,
    -0.029,
    -0.037,
    -0.043,
    -0.064,
    -0.092,
    -0.122,
    -0.161,
    -0.201,
    -0.249,
    -0.303,
    -0.366,
    -0.437,
    -0.517,
    -0.603,
    -0.692,
    -0.774,
    -0.843,
    -0.888,
    -0.908,
    -0.903,
    -0.880,
    -0.849,
    -0.816,
    -0.785,
    -0.760,
    -0.741,
    -0.729,
    -0.722,
    -0.722,
    -0.726,
    -0.734,
    -0.745,
    -0.760,
    -0.778,
    -0.798,
    -0.820,
    -0.845,
    -0.870,
    -0.898,
    -0.926,
    -0.956,
    -0.988,
    -1.020,
    -1.053,
    -1.087,
    -1.122,
    -1.158,
    -1.195,
    -1.232,
    -1.270,
    -1.309,
    -1.349,
    -1.389,
    -1.429,
    -1.471,
    -1.513,
    -1.555,
    -1.598,
    -1.641,
    -1.685,
)


@lru_cache(maxsize=1)
def _f19_spline_second_derivatives() -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Not-a-knot cubic spline second derivatives of both F19 columns.

    ``scipy.interpolate.splrep(x, y, s=0)`` (used by ``dust_extinction``) is
    this interpolant. The spline is linear in the tabulated values, so the
    ``R(55)``-dependent curve is the same combination of the two splines.
    """
    import numpy as np

    x = np.asarray(_F19_X, dtype=np.float64)
    h = np.diff(x)
    n = x.size
    matrix = np.zeros((n, n))
    matrix[0, 0], matrix[0, 1], matrix[0, 2] = h[1], -(h[0] + h[1]), h[0]
    matrix[-1, -3], matrix[-1, -2], matrix[-1, -1] = h[-1], -(h[-2] + h[-1]), h[-2]
    for i in range(1, n - 1):
        matrix[i, i - 1] = h[i - 1]
        matrix[i, i] = 2.0 * (h[i - 1] + h[i])
        matrix[i, i + 1] = h[i]
    out = []
    for column in (_F19_K302, _F19_DELTAK):
        y = np.asarray(column, dtype=np.float64)
        slope = np.diff(y) / h
        rhs = np.zeros(n)
        rhs[1:-1] = 6.0 * np.diff(slope)
        out.append(tuple(np.linalg.solve(matrix, rhs).tolist()))
    return out[0], out[1]


def _cubic_spline(
    x: Tensor, y_knots: tuple[float, ...], m_knots: tuple[float, ...]
) -> Tensor:
    xk = torch.tensor(_F19_X, device=x.device, dtype=x.dtype)
    yk = torch.tensor(y_knots, device=x.device, dtype=x.dtype)
    mk = torch.tensor(m_knots, device=x.device, dtype=x.dtype)
    i = (torch.searchsorted(xk, x.detach(), right=True) - 1).clamp(0, xk.numel() - 2)
    h = xk[i + 1] - xk[i]
    t0 = xk[i + 1] - x
    t1 = x - xk[i]
    return (
        mk[i] * t0**3 / (6.0 * h)
        + mk[i + 1] * t1**3 / (6.0 * h)
        + (yk[i] / h - mk[i] * h / 6.0) * t0
        + (yk[i + 1] / h - mk[i + 1] * h / 6.0) * t1
    )


def f19_a_lambda(wave_angstrom: Tensor, a_v: Tensor, r_v: Tensor) -> Tensor:
    """Fitzpatrick et al. (2019) ``A(λ)``; 1150 Å to 3.3 μm, ``2 <= R_V <= 6``.

    F19 is normalized at 5500 Å: ``a_v`` and ``r_v`` are ``A(55)`` and
    ``R(55)``, with ``k(λ-55) = k_3.02 + 0.990 Δk (R(55) - 3.10)``.
    """
    x = 10000.0 / wave_angstrom
    _check_range(x, torch.as_tensor(r_v), (0.3, 8.7), (2.0, 6.0), "f19")
    m_base, m_delta = _f19_spline_second_derivatives()
    base = _cubic_spline(x, _F19_K302, m_base)
    delta = _cubic_spline(x, _F19_DELTAK, m_delta)
    k = base + delta * (r_v - 3.10) * 0.990
    return (k / r_v + 1.0) * a_v


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


def _as_tensor_like(value: Tensor | float, like: Tensor) -> Tensor:
    if isinstance(value, Tensor):
        return value
    return torch.as_tensor(value, device=like.device, dtype=like.dtype)


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
    assert scale is not None
    wave_a = torch.as_tensor(wave_a)
    if not wave_a.is_floating_point():
        wave_a = wave_a.to(torch.get_default_dtype())
    # Python scalars are built in the grid dtype: a float32 detour would move
    # R_V = 2.3 below a law's calibrated range.
    scale_t = _align_scale(_as_tensor_like(scale, wave_a), wave_a)
    r_v_t = _align_scale(_as_tensor_like(r_v, wave_a), wave_a)
    a_v_t = scale_t if a_v is not None else r_v_t * scale_t
    if law_name == "f99":
        return f99_a_lambda(wave_a, a_v_t, r_v_t)
    if law_name == "calzetti":
        return calzetti_a_lambda(wave_a, a_v_t, r_v_t)
    if law_name == "g23":
        return g23_a_lambda(wave_a, a_v_t, r_v_t)
    if law_name == "f19":
        return f19_a_lambda(wave_a, a_v_t, r_v_t)
    if law_name == "gcc09":
        return gcc09_a_lambda(wave_a, a_v_t, r_v_t)
    a, b = _compute_ab(wave_a, law_name)
    return (a + b / r_v_t) * a_v_t


def ccm89(
    wave_angstrom: Tensor,
    a_v: Tensor | float,
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
    mag = a_lambda(wave, ebv=ebv, r_v=r_v, law=law, wavelength_unit=wavelength_unit)
    # where-form keeps a unit gradient exactly at A = 0 (clamp_min gives 0).
    mag = torch.where(mag < 0.0, torch.zeros_like(mag), mag)
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
    "f19_a_lambda",
    "f99_a_lambda",
    "fitzpatrick99",
    "g23_a_lambda",
    "gcc09_a_lambda",
    "k_lambda",
    "odonnell94",
    "odonnell94_ab",
    "transmission",
]
