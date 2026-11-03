"""IAU 2000A lunisolar + planetary nutation (IERS / SOFA nut00a tables).

Coefficients are the IERS-published MHB2000 series as tabulated in ERFA
``nut00a`` (SOFA-derived, BSD-style license; not GPL). Units are 0.1 µas.
"""

from __future__ import annotations

import base64
import gzip
import math

import numpy as np
import torch
from torch import Tensor

from .frames import ARCSEC_TO_RAD
from ..device import _as_tensor, _to

TURNAS = 1_296_000.0
DAS2R = ARCSEC_TO_RAD
U2R = DAS2R / 1.0e7
TWO_PI = 2.0 * math.pi
MJD_J2000 = 51544.5
DJC = 36525.0

_XLS = torch.as_tensor(
    np.frombuffer(
        gzip.decompress(
            base64.b64decode(
                b"""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
AA=="""
            )
        ),
        dtype=np.float64,
    )
    .reshape(678, 11)
    .copy()
)

_XPL = torch.as_tensor(
    np.frombuffer(
        gzip.decompress(
            base64.b64decode(
                b"""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
JZxZfLYAAA=="""
            )
        ),
        dtype=np.int32,
    )
    .reshape(687, 17)
    .copy()
)


def julian_centuries_tt(tt_mjd: Tensor) -> Tensor:
    return (tt_mjd - MJD_J2000) / DJC


def _arcsec_poly_rad(t: Tensor, coeffs: tuple[float, ...]) -> Tensor:
    acc = torch.zeros_like(t)
    for c in reversed(coeffs):
        acc = acc * t + c
    return torch.remainder(acc, TURNAS) * DAS2R


def _rad_lin(t: Tensor, a0: float, a1: float) -> Tensor:
    return torch.remainder(
        _as_tensor(a0, dtype=t.dtype, device=t.device) + a1 * t, TWO_PI
    )


def fal03(t: Tensor) -> Tensor:
    return _arcsec_poly_rad(
        t, (485868.249036, 1717915923.2178, 31.8792, 0.051635, -0.00024470)
    )


def falp03(t: Tensor) -> Tensor:
    return _arcsec_poly_rad(
        t, (1287104.793048, 129596581.0481, -0.5532, 0.000136, -0.00001149)
    )


def faf03(t: Tensor) -> Tensor:
    return _arcsec_poly_rad(
        t, (335779.526232, 1739527262.8478, -12.7512, -0.001037, 0.00000417)
    )


def fad03(t: Tensor) -> Tensor:
    return _arcsec_poly_rad(
        t, (1072260.703692, 1602961601.2090, -6.3706, 0.006593, -0.00003169)
    )


def faom03(t: Tensor) -> Tensor:
    return _arcsec_poly_rad(
        t, (450160.398036, -6962890.5431, 7.4722, 0.007702, -0.00005939)
    )


def fame03(t: Tensor) -> Tensor:
    return _rad_lin(t, 4.402608842, 2608.7903141574)


def fave03(t: Tensor) -> Tensor:
    return _rad_lin(t, 3.176146697, 1021.3285546211)


def fae03(t: Tensor) -> Tensor:
    return _rad_lin(t, 1.753470314, 628.3075849991)


def fama03(t: Tensor) -> Tensor:
    return _rad_lin(t, 6.203480913, 334.0612426700)


def faju03(t: Tensor) -> Tensor:
    return _rad_lin(t, 0.599546497, 52.9690962641)


def fasa03(t: Tensor) -> Tensor:
    return _rad_lin(t, 0.874016757, 21.3299104960)


def faur03(t: Tensor) -> Tensor:
    return _rad_lin(t, 5.481293872, 7.4781598567)


def fapa03(t: Tensor) -> Tensor:
    return (0.024381750 + 0.00000538691 * t) * t


def _tables(device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    return _to(_XLS, device=device, dtype=dtype), _to(_XPL, device=device, dtype=dtype)


def iau2000a_nutation(tt_mjd: Tensor) -> tuple[Tensor, Tensor]:
    """IAU 2000A nutation (dpsi, deps) in radians. Vectorized over ``tt_mjd``."""
    t = julian_centuries_tt(tt_mjd)
    device, dtype = t.device, t.dtype
    xls, xpl = _tables(device, dtype)

    el = fal03(t)
    elp = _arcsec_poly_rad(
        t, (1287104.79305, 129596581.0481, -0.5532, 0.000136, -0.00001149)
    )
    f = faf03(t)
    d = _arcsec_poly_rad(
        t, (1072260.70369, 1602961601.2090, -6.3706, 0.006593, -0.00003169)
    )
    om = faom03(t)

    delaunay = torch.stack((el, elp, f, d, om), dim=-1)
    n_ls = xls[:, :5]
    arg_ls = (delaunay.unsqueeze(-2) * n_ls).sum(dim=-1)
    sarg = torch.sin(arg_ls)
    carg = torch.cos(arg_ls)
    t_b = t.unsqueeze(-1)
    dp_ls = ((xls[:, 5] + xls[:, 6] * t_b) * sarg + xls[:, 7] * carg).sum(dim=-1)
    de_ls = ((xls[:, 8] + xls[:, 9] * t_b) * carg + xls[:, 10] * sarg).sum(dim=-1)

    al = _rad_lin(t, 2.35555598, 8328.6914269554)
    af = _rad_lin(t, 1.627905234, 8433.466158131)
    ad = _rad_lin(t, 5.198466741, 7771.3771468121)
    aom = _rad_lin(t, 2.18243920, -33.757045)
    apa = fapa03(t)
    planets = torch.stack(
        (
            al,
            af,
            ad,
            aom,
            fame03(t),
            fave03(t),
            fae03(t),
            fama03(t),
            faju03(t),
            fasa03(t),
            faur03(t),
            _rad_lin(t, 5.321159000, 3.8127774000),
            apa,
        ),
        dim=-1,
    )
    n_pl = xpl[:, :13]
    arg_pl = (planets.unsqueeze(-2) * n_pl).sum(dim=-1)
    sarg_p = torch.sin(arg_pl)
    carg_p = torch.cos(arg_pl)
    dp_pl = (xpl[:, 13] * sarg_p + xpl[:, 14] * carg_p).sum(dim=-1)
    de_pl = (xpl[:, 15] * sarg_p + xpl[:, 16] * carg_p).sum(dim=-1)

    dpsi = (dp_ls + dp_pl) * U2R
    deps = (de_ls + de_pl) * U2R
    return dpsi, deps


def iau2006_nutation(tt_mjd: Tensor) -> tuple[Tensor, Tensor]:
    """IAU 2000A nutation with IAU 2006 / P03 adjustments (SOFA nut06a)."""
    dpsi, deps = iau2000a_nutation(tt_mjd)
    t = julian_centuries_tt(tt_mjd)
    fj2 = -2.7774e-6 * t
    return dpsi * (1.0 + 0.4697e-6 + fj2), deps * (1.0 + fj2)
