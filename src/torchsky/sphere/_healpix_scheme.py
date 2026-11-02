"""HEALPix pixelisation constants, validators, and ring/nested index arithmetic."""

import math
from typing import Sequence, Tuple

import torch

from .._dtype import work_dtype as _float_dtype_for_device
from torch import Tensor


_JRLL = torch.tensor([2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4], dtype=torch.int64)


_JPLL = torch.tensor([1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7], dtype=torch.int64)


_FACE_CONST_CACHE: dict[tuple[str, int], tuple[Tensor, Tensor]] = {}


_CAP_START_CACHE: dict[tuple[str, int, int], Tensor] = {}


_MAX_PIXEL_RADIUS_CACHE: dict[int, float] = {}


_NB_XOFFSET = torch.tensor([-1, -1, 0, 1, 1, 1, 0, -1], dtype=torch.int64)


_NB_YOFFSET = torch.tensor([0, 1, 1, 1, 0, -1, -1, -1], dtype=torch.int64)


_NB_FACEARRAY = torch.tensor(
    [
        [8, 9, 10, 11, -1, -1, -1, -1, 10, 11, 8, 9],
        [5, 6, 7, 4, 8, 9, 10, 11, 9, 10, 11, 8],
        [-1, -1, -1, -1, 5, 6, 7, 4, -1, -1, -1, -1],
        [4, 5, 6, 7, 11, 8, 9, 10, 11, 8, 9, 10],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [1, 2, 3, 0, 0, 1, 2, 3, 5, 6, 7, 4],
        [-1, -1, -1, -1, 7, 4, 5, 6, -1, -1, -1, -1],
        [3, 0, 1, 2, 3, 0, 1, 2, 4, 5, 6, 7],
        [2, 3, 0, 1, -1, -1, -1, -1, 0, 1, 2, 3],
    ],
    dtype=torch.int64,
)


_NB_SWAPARRAY = torch.tensor(
    [
        [0, 0, 3],
        [0, 0, 6],
        [0, 0, 0],
        [0, 0, 5],
        [0, 0, 0],
        [5, 0, 0],
        [0, 0, 0],
        [6, 0, 0],
        [3, 0, 0],
    ],
    dtype=torch.int64,
)


def _validate_nside(nside: int) -> None:
    if nside <= 0 or (nside & (nside - 1)) != 0:
        raise ValueError("nside must be a positive power of two")
    if nside > (1 << 29):
        raise ValueError("nside must be <= 2**29 (HEALPix order 29)")


def _validate_lonlat(lon: Tensor, lat: Tensor) -> None:
    if not bool(torch.isfinite(lon).all()) or not bool(torch.isfinite(lat).all()):
        raise ValueError("longitude and latitude must be finite")
    if bool(torch.any((lat < -90.0) | (lat > 90.0))):
        raise ValueError("latitude must be in [-90, 90]")


def _as_float64(x: Tensor | float | list[float]) -> Tensor:
    t = torch.as_tensor(x)
    return t.to(dtype=_float_dtype_for_device(t.device))


def _as_int64(x: Tensor | float | Sequence[int]) -> Tensor:
    return torch.as_tensor(x, dtype=torch.int64)


def _face_consts(device: torch.device) -> Tuple[Tensor, Tensor]:
    idx = -1 if device.index is None else int(device.index)
    key = (device.type, idx)
    cached = _FACE_CONST_CACHE.get(key)
    if cached is not None:
        return cached
    jrll = _JRLL.to(device=device)
    jpll = _JPLL.to(device=device)
    _FACE_CONST_CACHE[key] = (jrll, jpll)
    return jrll, jpll


def _cap_ring_starts(nside: int, device: torch.device) -> Tensor:
    idx = -1 if device.index is None else int(device.index)
    key = (device.type, idx, nside)
    cached = _CAP_START_CACHE.get(key)
    if cached is not None:
        return cached
    ir = torch.arange(1, nside + 1, dtype=torch.int64, device=device)
    starts = 2 * ir * (ir - 1)
    _CAP_START_CACHE[key] = starts
    return starts


def _isqrt(v: Tensor) -> Tensor:
    v = v.to(torch.int64)
    # Use native integer sqrt when available, fallback to corrected float path.
    native_isqrt = getattr(torch, "isqrt", None)  # absent from older torch builds
    if native_isqrt is not None:
        try:
            return native_isqrt(v)
        except RuntimeError:
            pass
    f_dtype = _float_dtype_for_device(v.device)
    r = torch.floor(torch.sqrt(v.to(f_dtype))).to(torch.int64)
    r = torch.where((r + 1) * (r + 1) <= v, r + 1, r)
    r = torch.where(r * r > v, r - 1, r)
    return r


def _xyf2nest(nside: int, ix: Tensor, iy: Tensor, face_num: Tensor) -> Tensor:
    npface = nside * nside
    return (
        face_num * npface + _core_mod.spread_bits(ix) + (_core_mod.spread_bits(iy) << 1)
    )


def _nest2xyf(nside: int, pix: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    npface = nside * nside
    face_num = pix // npface
    ipf = pix % npface
    ix = _core_mod.compact_bits(ipf)
    iy = _core_mod.compact_bits(ipf >> 1)
    return ix, iy, face_num


def _xyf2ring(nside: int, ix: Tensor, iy: Tensor, face_num: Tensor) -> Tensor:
    nl4 = 4 * nside
    ncap = 2 * nside * (nside - 1)
    npix = 12 * nside * nside
    jrll, jpll = _face_consts(face_num.device)
    jr = jrll[face_num] * nside - ix - iy - 1

    north = jr < nside
    south = jr > (3 * nside)
    equat = ~(north | south)

    nr_south = nl4 - jr
    nr = torch.where(
        north, jr, torch.where(south, nr_south, torch.full_like(jr, nside))
    )
    n_before_north = 2 * nr * (nr - 1)
    n_before_south = npix - 2 * (nr + 1) * nr
    n_before_equat = ncap + (jr - nside) * nl4
    n_before = torch.where(
        north, n_before_north, torch.where(south, n_before_south, n_before_equat)
    )
    kshift = torch.where(equat, (jr - nside) & 1, torch.zeros_like(jr))

    jp = (jpll[face_num] * nr + ix - iy + 1 + kshift) // 2
    jp = torch.where(jp > nl4, jp - nl4, jp)
    jp = torch.where(jp < 1, jp + nl4, jp)

    return n_before + jp - 1


def _ring2xyf(nside: int, pix: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    ncap = 2 * nside * (nside - 1)
    npix = 12 * nside * nside
    nl2 = 2 * nside
    pix = pix.to(torch.int64)
    jrll, jpll = _face_consts(pix.device)
    starts = _cap_ring_starts(nside, pix.device)

    iring = torch.empty_like(pix)
    iphi = torch.empty_like(pix)
    kshift = torch.empty_like(pix)
    nr = torch.empty_like(pix)
    face_num = torch.empty_like(pix)

    north = pix < ncap
    equat = (pix >= ncap) & (pix < (npix - ncap))
    south = ~(north | equat)

    if north.any():
        p = pix[north]
        ir = torch.bucketize(p, starts, right=True)
        iph = (p + 1) - 2 * ir * (ir - 1)
        iring[north] = ir
        iphi[north] = iph
        kshift[north] = 0
        nr[north] = ir
        face_num[north] = (iph - 1) // ir

    if equat.any():
        p = pix[equat]
        ip = p - ncap
        ir = (ip // (4 * nside)) + nside
        iph = (ip % (4 * nside)) + 1
        ks = (ir + nside) & 1
        ire = ir - nside + 1
        irm = nl2 + 2 - ire
        ifm = (iph - (ire // 2) + nside - 1) // nside
        ifp = (iph - (irm // 2) + nside - 1) // nside
        f = torch.where(ifp == ifm, ifp | 4, torch.where(ifp < ifm, ifp, ifm + 8))

        iring[equat] = ir
        iphi[equat] = iph
        kshift[equat] = ks
        nr[equat] = nside
        face_num[equat] = f

    if south.any():
        p = pix[south]
        ip = npix - p
        irs = torch.bucketize(ip - 1, starts, right=True)
        iph = 4 * irs + 1 - (ip - 2 * irs * (irs - 1))
        ir = 2 * nl2 - irs
        f = 8 + (iph - 1) // irs

        iring[south] = ir
        iphi[south] = iph
        kshift[south] = 0
        nr[south] = irs
        face_num[south] = f

    irt = iring - jrll[face_num] * nside + 1
    ipt = 2 * iphi - jpll[face_num] * nr - kshift - 1
    ipt = torch.where(ipt >= nl2, ipt - 8 * nside, ipt)

    ix = torch.div(ipt - irt, 2, rounding_mode="floor")
    iy = torch.div(-(ipt + irt), 2, rounding_mode="floor")
    return ix, iy, face_num


def _pix2thetaphi_ring(nside: int, pix: Tensor) -> tuple[Tensor, Tensor]:
    """Convert HEALPix RING indices to (theta, phi) in radians."""
    _validate_nside(nside)
    pix_t = _as_int64(pix)
    f_dtype = _float_dtype_for_device(pix_t.device)

    ncap = 2 * nside * (nside - 1)
    npix = 12 * nside * nside
    if torch.any((pix_t < 0) | (pix_t >= npix)):
        raise ValueError("pixel index out of range for nside")

    fact2 = 4.0 / npix

    z = torch.empty_like(pix_t, dtype=f_dtype)
    theta = torch.empty_like(pix_t, dtype=f_dtype)
    phi = torch.empty_like(pix_t, dtype=f_dtype)

    north = pix_t < ncap
    equat = (pix_t >= ncap) & (pix_t < (npix - ncap))
    south = ~(north | equat)

    if north.any():
        p = pix_t[north]
        iring = (1 + _isqrt(1 + 2 * p)) >> 1
        iphi = (p + 1) - 2 * iring * (iring - 1)
        tmp = torch.square(iring.to(f_dtype)) * fact2
        z[north] = 1.0 - tmp
        sint = torch.sqrt(torch.clamp(tmp * (2.0 - tmp), min=0.0))
        theta[north] = torch.atan2(sint, z[north])
        phi[north] = (iphi.to(f_dtype) - 0.5) * ((math.pi / 2.0) / iring.to(f_dtype))

    if equat.any():
        p = pix_t[equat]
        fact1 = (2 * nside) * fact2
        ip = p - ncap
        iring = (ip // (4 * nside)) + nside
        iphi = (ip % (4 * nside)) + 1
        fodd = torch.where(((iring + nside) & 1).bool(), 1.0, 0.5)
        z[equat] = (2 * nside - iring).to(f_dtype) * fact1
        theta[equat] = torch.acos(torch.clamp(z[equat], -1.0, 1.0))
        phi[equat] = (iphi.to(f_dtype) - fodd) * (math.pi / (2 * nside))

    if south.any():
        p = pix_t[south]
        ip = npix - p
        iring = (1 + _isqrt(2 * ip - 1)) >> 1
        iphi = 4 * iring + 1 - (ip - 2 * iring * (iring - 1))
        tmp = torch.square(iring.to(f_dtype)) * fact2
        z[south] = -1.0 + tmp
        sint = torch.sqrt(torch.clamp(tmp * (2.0 - tmp), min=0.0))
        theta[south] = math.pi - torch.atan2(sint, 1.0 - tmp)
        phi[south] = (iphi.to(f_dtype) - 0.5) * ((math.pi / 2.0) / iring.to(f_dtype))

    phi = torch.remainder(phi, 2.0 * math.pi)
    return theta, phi


# Late-bound so monkeypatching ``healpix`` attributes still reaches this code.
from . import healpix as _core_mod  # noqa: E402
