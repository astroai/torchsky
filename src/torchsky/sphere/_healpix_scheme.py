"""HEALPix pixelisation constants, validators, and ring/nested index arithmetic."""

import math
from typing import Sequence, Tuple

import torch

from .._dtype import work_dtype as _float_dtype_for_device
from .._dtype import writable
from torch import Tensor


_JRLL = torch.tensor([2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4], dtype=torch.int64)


_JPLL = torch.tensor([1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7], dtype=torch.int64)


_FACE_CONST_CACHE: dict[tuple[str, int], tuple[Tensor, Tensor]] = {}


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


def _host_checks_enabled() -> bool:
    """Whether data-dependent validation may synchronise with the host.

    Under ``torch.compile`` a ``bool(tensor)`` check splits the graph, so the
    eager-only checks are skipped while tracing (the native kernels still
    validate their inputs).
    """
    return not torch.compiler.is_compiling()


def _validate_lonlat(lon: Tensor, lat: Tensor) -> None:
    if not _host_checks_enabled():
        return
    if not bool(torch.isfinite(lon).all()) or not bool(torch.isfinite(lat).all()):
        raise ValueError("longitude and latitude must be finite")
    if bool(torch.any((lat < -90.0) | (lat > 90.0))):
        raise ValueError("latitude must be in [-90, 90]")


def _validate_pixels(pix: Tensor, npix: int, message: str) -> None:
    if _host_checks_enabled() and bool(torch.any((pix < 0) | (pix >= npix))):
        raise ValueError(message)


def _as_float64(x: Tensor | float | list[float]) -> Tensor:
    """Tensor in the device working dtype; Python/NumPy scalars stay float64.

    ``torch.as_tensor(0.1)`` is float32 (the default dtype), which would round
    Python floats before the float64 cast.
    """
    if isinstance(x, Tensor):
        return x.to(dtype=_float_dtype_for_device(x.device))
    return torch.as_tensor(writable(x), dtype=torch.float64)


def _as_int64(x: Tensor | float | Sequence[int]) -> Tensor:
    return torch.as_tensor(x, dtype=torch.int64)


def _face_consts(device: torch.device) -> Tuple[Tensor, Tensor]:
    if torch.compiler.is_compiling():
        return _JRLL.to(device=device), _JPLL.to(device=device)
    idx = -1 if device.index is None else int(device.index)
    key = (device.type, idx)
    cached = _FACE_CONST_CACHE.get(key)
    if cached is not None:
        return cached
    jrll = _JRLL.to(device=device)
    jpll = _JPLL.to(device=device)
    _FACE_CONST_CACHE[key] = (jrll, jpll)
    return jrll, jpll


def _isqrt(v: Tensor) -> Tensor:
    """Exact ``floor(sqrt(v))`` of non-negative int64 values on any device.

    The float estimate is within one unit in float64 but not in float32 (MPS),
    so two integer Newton steps (``r <- (r + v // r) // 2``, which never
    undershoot the root once ``r >= sqrt(v)``) and a final +/-1 correction make
    it exact up to ``v < 2**62``.
    """
    v = v.to(torch.int64).clamp_min(0)
    f_dtype = _float_dtype_for_device(v.device)
    r = torch.floor(torch.sqrt(v.to(f_dtype))).to(torch.int64).clamp_min(1)
    for _ in range(2):
        r = torch.div(
            r + torch.div(v, r, rounding_mode="floor"), 2, rounding_mode="floor"
        )
        r = r.clamp_min(1)
    r = torch.where(r * r > v, r - 1, r)
    r = torch.where((r + 1) * (r + 1) <= v, r + 1, r)
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


def _ring_cap_index(nside: int, pix: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Branch-free ring bookkeeping of RING pixels.

    Returns ``(region, iring_cap, iphi, iring)`` where ``region`` is -1/0/+1 for
    the south cap, equatorial belt and north cap, ``iring_cap`` is the ring
    number counted from the nearest pole (``nside`` in the belt), ``iphi`` the
    1-based pixel index within the ring and ``iring`` the global 1-based ring.
    Every regime is evaluated for every element and selected with ``where``,
    so shapes never depend on the data (``torch.compile`` traces one graph);
    the denominators of the unselected regimes are kept non-zero.
    """
    ncap = 2 * nside * (nside - 1)
    npix = 12 * nside * nside
    nl4 = 4 * nside
    pix = pix.to(torch.int64)
    north = pix < ncap
    south = pix >= (npix - ncap)

    # Polar caps: ring r starts at 2 r (r - 1), so r = (1 + isqrt(1 + 2 p)) // 2.
    ir_n = (1 + _isqrt(1 + 2 * pix)) >> 1
    iph_n = (pix + 1) - 2 * ir_n * (ir_n - 1)
    ip_s = npix - pix
    ir_s = (1 + _isqrt(2 * ip_s - 1)) >> 1
    iph_s = 4 * ir_s + 1 - (ip_s - 2 * ir_s * (ir_s - 1))
    # Equatorial belt.
    ip_e = pix - ncap
    ir_e = torch.div(ip_e, nl4, rounding_mode="floor") + nside
    iph_e = torch.remainder(ip_e, nl4) + 1

    region = torch.where(
        north,
        torch.ones_like(pix),
        torch.where(south, -torch.ones_like(pix), torch.zeros_like(pix)),
    )
    iring_cap = torch.where(
        north, ir_n, torch.where(south, ir_s, torch.full_like(pix, nside))
    )
    iphi = torch.where(north, iph_n, torch.where(south, iph_s, iph_e))
    iring = torch.where(north, ir_n, torch.where(south, nl4 - ir_s, ir_e))
    return region, iring_cap.clamp_min(1), iphi, iring


def _ring2xyf(nside: int, pix: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    nl2 = 2 * nside
    pix = pix.to(torch.int64)
    jrll, jpll = _face_consts(pix.device)
    region, nr, iphi, iring = _ring_cap_index(nside, pix)
    north = region > 0
    south = region < 0

    face_n = torch.div(iphi - 1, nr, rounding_mode="floor")
    face_s = 8 + face_n
    # Equatorial belt: faces from the ascending/descending edge-line indices.
    ire = iring - nside + 1
    irm = nl2 + 2 - ire
    ifm = torch.div(
        iphi - torch.div(ire, 2, rounding_mode="floor") + nside - 1,
        nside,
        rounding_mode="floor",
    )
    ifp = torch.div(
        iphi - torch.div(irm, 2, rounding_mode="floor") + nside - 1,
        nside,
        rounding_mode="floor",
    )
    face_e = torch.where(ifp == ifm, ifp | 4, torch.where(ifp < ifm, ifp, ifm + 8))
    face_num = torch.where(north, face_n, torch.where(south, face_s, face_e))
    # Out-of-range pixels (unchecked while compiling) must not index out of bounds.
    face_num = face_num.clamp(0, 11)
    kshift = torch.where(north | south, torch.zeros_like(pix), (iring + nside) & 1)

    irt = iring - jrll[face_num] * nside + 1
    ipt = 2 * iphi - jpll[face_num] * nr - kshift - 1
    ipt = torch.where(ipt >= nl2, ipt - 8 * nside, ipt)

    ix = torch.div(ipt - irt, 2, rounding_mode="floor")
    iy = torch.div(-(ipt + irt), 2, rounding_mode="floor")
    return ix, iy, face_num


def _ring_theta_lat(
    tmp: Tensor, region: Tensor, z_belt: Tensor
) -> tuple[Tensor, Tensor]:
    """Colatitude (rad) and latitude (deg) of a ring.

    ``tmp = 1 - |z|`` in the caps (``region`` +1/-1), ``z_belt = z`` in the belt.
    In the caps the pole distance ``atan2(sqrt(tmp (2 - tmp)), 1 - tmp)`` keeps
    full relative precision, where ``acos``/``asin`` of ``1 - tmp`` lose about
    half of the digits (the first rings of order 29 collapsed onto the pole).
    In the belt ``asin(z)`` is well conditioned; ``90 - acos(z)`` would cancel
    (3x worse in float32). The native kernels use the same expressions.
    """
    sint = torch.sqrt(torch.clamp(tmp * (2.0 - tmp), min=0.0))
    pole_dist = torch.atan2(sint, 1.0 - tmp)
    pole_dist_deg = torch.rad2deg(pole_dist)
    z_belt = torch.clamp(z_belt, -1.0, 1.0)
    north = region > 0
    south = region < 0
    theta = torch.where(
        north, pole_dist, torch.where(south, math.pi - pole_dist, torch.acos(z_belt))
    )
    lat = torch.where(
        north,
        90.0 - pole_dist_deg,
        torch.where(south, pole_dist_deg - 90.0, torch.rad2deg(torch.asin(z_belt))),
    )
    return theta, lat


def _pix2ring_coords(nside: int, pix: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """``(theta, phi, lat_deg)`` of RING pixel centres, branch-free."""
    _validate_nside(nside)
    pix_t = _as_int64(pix)
    f_dtype = _float_dtype_for_device(pix_t.device)
    npix = 12 * nside * nside
    _validate_pixels(pix_t, npix, "pixel index out of range for nside")

    fact2 = 4.0 / npix
    fact1 = (2 * nside) * fact2
    region, ir_cap, iphi, iring = _ring_cap_index(nside, pix_t)
    ir_cap_f = ir_cap.to(f_dtype)
    tmp = torch.square(ir_cap_f) * fact2
    z_belt = (2 * nside - iring).to(f_dtype) * fact1
    theta, lat = _ring_theta_lat(tmp, region, z_belt)

    phi_cap = (iphi.to(f_dtype) - 0.5) * ((math.pi / 2.0) / ir_cap_f)
    fodd = torch.where(((iring + nside) & 1).bool(), 1.0, 0.5).to(f_dtype)
    phi_belt = (iphi.to(f_dtype) - fodd) * (math.pi / (2 * nside))
    phi = torch.where(region == 0, phi_belt, phi_cap)
    phi = torch.remainder(phi, 2.0 * math.pi)
    return theta, phi, lat


def _pix2thetaphi_ring(nside: int, pix: Tensor) -> tuple[Tensor, Tensor]:
    """Convert HEALPix RING indices to (theta, phi) in radians."""
    theta, phi, _ = _pix2ring_coords(nside, pix)
    return theta, phi


# Late-bound so monkeypatching ``healpix`` attributes still reaches this code.
from . import healpix as _core_mod  # noqa: E402
