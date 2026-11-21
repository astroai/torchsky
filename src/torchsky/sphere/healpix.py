import math
from typing import Sequence, Tuple

import torch

from .. import _ops
from .._dtype import work_dtype as _float_dtype_for_device
from torch import Tensor

from ._healpix_scheme import (  # noqa: F401
    _FACE_CONST_CACHE,
    _JPLL,
    _JRLL,
    _NB_FACEARRAY,
    _NB_SWAPARRAY,
    _NB_XOFFSET,
    _NB_YOFFSET,
    _as_float64,
    _as_int64,
    _pix2ring_coords,
    _ring_theta_lat,
    _face_consts,
    _host_checks_enabled,
    _isqrt,
    _nest2xyf,
    _pix2thetaphi_ring,
    _ring2xyf,
    _validate_lonlat,
    _validate_nside,
    _validate_pixels,
    _xyf2nest,
    _xyf2ring,
)
from ._healpix_query_support import (  # noqa: F401
    _CELL_RADIUS_CACHE,
    _CELL_RADIUS_EXACT_LEVELS,
    _cell_radius_bound,
    _disc_nest_ranges_torch,
    _get_ring_info2,
    _nest_ranges_by_descent,
    _nest_ranges_to_ring_pixels,
    _normalize_order,
    _normalize_xyz,
    _pixel_range_chunks,
    _query_disc_nest_ranges,
    _ranges_to_sorted_pixels,
    _resolve_dtype,
    _ring_above,
    _ring_pixels_colatitude_band,
    _ring_range_pixels,
    _ring_ranges_nest_pixels,
    _strip_nest_pixels,
    _strip_rings,
)
from ..device import _as_tensor

try:
    from torch.func import vmap
except ImportError:
    vmap = None

UNSEEN = -1.6375e30


def spread_bits(x: Tensor) -> Tensor:
    """Interleave bits for Morton code (x -> x0x1x2...)."""
    x = x.to(torch.int64)
    x = (x | (x << 16)) & 0x0000FFFF0000FFFF
    x = (x | (x << 8)) & 0x00FF00FF00FF00FF
    x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0F
    x = (x | (x << 2)) & 0x3333333333333333
    x = (x | (x << 1)) & 0x5555555555555555
    return x


def compact_bits(x: Tensor) -> Tensor:
    """Extract interleaved bits from Morton code."""
    x = x.to(torch.int64) & 0x5555555555555555
    x = (x | (x >> 1)) & 0x3333333333333333
    x = (x | (x >> 2)) & 0x0F0F0F0F0F0F0F0F
    x = (x | (x >> 4)) & 0x00FF00FF00FF00FF
    x = (x | (x >> 8)) & 0x0000FFFF0000FFFF
    x = (x | (x >> 16)) & 0x00000000FFFFFFFF
    return x


def _lonlat_zt(ra: Tensor, dec: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """``(z, tt, cap)`` of HEALPix ``ang2pix`` for lon/lat in degrees.

    ``z = sin(lat)``, ``tt = 2 phi / pi`` in ``[0, 4)`` and
    ``cap = nside_free * sqrt(3 (1 - |z|))``. ``1 - |z|`` is evaluated as
    ``2 sin^2(colat / 2)`` with ``colat = 90 - |lat|``: the difference
    ``1 - |sin(lat)|`` cancels next to the poles and moved order-29 points
    several rings (healpy switches to ``sin(theta)`` there for the same
    reason).
    """
    z = torch.sin(torch.deg2rad(dec))
    tt = torch.remainder(torch.deg2rad(ra), 2.0 * math.pi) * (2.0 / math.pi)
    half = torch.sin(torch.deg2rad(90.0 - torch.abs(dec)) * 0.5)
    cap = torch.sqrt(3.0 * (2.0 * half * half))
    return z, tt, cap


def _ang2pix_ring_torch(nside: int, ra: Tensor, dec: Tensor) -> Tensor:
    """Branch-free RING ``ang2pix``: both regimes evaluated, selected by ``where``."""
    f_dtype = ra.dtype
    nl4 = 4 * nside
    z, tt, cap = _lonlat_zt(ra, dec)

    # Equatorial belt (|z| <= 2/3).
    temp1 = nside * (0.5 + tt)
    temp2 = nside * z * 0.75
    jp = (temp1 - temp2).to(torch.int64)
    jm = (temp1 + temp2).to(torch.int64)
    ir = nside + 1 + jp - jm
    kshift = 1 - (ir & 1)
    ip = torch.div(jp + jm - nside + kshift + 1, 2, rounding_mode="floor") + 1
    ip = torch.where(ip > nl4, ip - nl4, ip)
    ip = torch.where(ip < 1, ip + nl4, ip)
    pix_belt = 2 * nside * (nside - 1) + (ir - 1) * nl4 + (ip - 1)

    # Polar caps.
    tp = tt - tt.to(torch.int64).to(f_dtype)
    tmp = nside * cap
    jp_c = (tp * tmp).to(torch.int64)
    jm_c = ((1.0 - tp) * tmp).to(torch.int64)
    ir_c = jp_c + jm_c + 1
    ip_c = torch.remainder((tt * ir_c.to(f_dtype)).to(torch.int64), 4 * ir_c)
    pix_cap = torch.where(
        z > 0,
        2 * ir_c * (ir_c - 1) + ip_c,
        12 * nside * nside - 2 * ir_c * (ir_c + 1) + ip_c,
    )
    return torch.where(torch.abs(z) <= (2.0 / 3.0), pix_belt, pix_cap)


def _ang2pix_nested_torch(nside: int, ra: Tensor, dec: Tensor) -> Tensor:
    """Branch-free NESTED ``ang2pix``: both regimes evaluated, selected by ``where``."""
    f_dtype = ra.dtype
    z, tt, cap = _lonlat_zt(ra, dec)

    # Equatorial belt.
    temp1 = nside * (0.5 + tt)
    temp2 = nside * (z * 0.75)
    jp = (temp1 - temp2).to(torch.int64)
    jm = (temp1 + temp2).to(torch.int64)
    ifp = torch.div(jp, nside, rounding_mode="floor")
    ifm = torch.div(jm, nside, rounding_mode="floor")
    face_belt = torch.where(ifp == ifm, ifp | 4, torch.where(ifp < ifm, ifp, ifm + 8))
    ix_belt = jm & (nside - 1)
    iy_belt = nside - (jp & (nside - 1)) - 1

    # Polar caps.
    ntt = tt.to(torch.int64).clamp_max(3)
    tp = tt - ntt.to(f_dtype)
    tmp = nside * cap
    jp_c = (tp * tmp).to(torch.int64).clamp_max(nside - 1)
    jm_c = ((1.0 - tp) * tmp).to(torch.int64).clamp_max(nside - 1)
    north = z >= 0
    face_cap = torch.where(north, ntt, ntt + 8)
    ix_cap = torch.where(north, nside - jm_c - 1, jp_c)
    iy_cap = torch.where(north, nside - jp_c - 1, jm_c)

    belt = torch.abs(z) <= (2.0 / 3.0)
    face_num = torch.where(belt, face_belt, face_cap)
    ix = torch.where(belt, ix_belt, ix_cap)
    iy = torch.where(belt, iy_belt, iy_cap)
    return _xyf2nest(nside, ix, iy, face_num)


def _native_cpu(*tensors: Tensor) -> bool:
    return _ops.has_fixed_dispatcher() and all(t.device.type == "cpu" for t in tensors)


def ang2pix_ring(nside: int, ra: Tensor, dec: Tensor) -> Tensor:
    """Convert RA/Dec (degrees) to HEALPix RING indices."""
    _validate_nside(nside)
    ra_t = _as_float64(ra)
    dec_t = _as_float64(dec)
    ra_t, dec_t = torch.broadcast_tensors(ra_t, dec_t)
    _validate_lonlat(ra_t, dec_t)
    if _native_cpu(ra_t, dec_t):
        return _ops.healpix_ang2pix_ring(nside, ra_t.contiguous(), dec_t.contiguous())
    return _ang2pix_ring_torch(nside, ra_t, dec_t)


def pix2ang_ring(nside: int, pix: Tensor) -> Tuple[Tensor, Tensor]:
    """Convert HEALPix RING indices to RA/Dec (degrees)."""
    _validate_nside(nside)
    pix_t = _as_int64(pix)
    if _ops.has_fixed_dispatcher() and pix_t.device.type == "cpu":
        return _ops.healpix_pix2ang_ring(nside, pix_t.contiguous())

    _theta, phi, dec = _pix2ring_coords(nside, pix_t)
    ra = torch.remainder(torch.rad2deg(phi), 360.0)
    return ra, dec


def ang2pix_nested(nside: int, ra: Tensor, dec: Tensor) -> Tensor:
    """Convert RA/Dec (degrees) to HEALPix NESTED indices."""
    _validate_nside(nside)
    ra_t = _as_float64(ra)
    dec_t = _as_float64(dec)
    ra_t, dec_t = torch.broadcast_tensors(ra_t, dec_t)
    _validate_lonlat(ra_t, dec_t)
    if _native_cpu(ra_t, dec_t):
        return _ops.healpix_ang2pix_nested(nside, ra_t.contiguous(), dec_t.contiguous())
    return _ang2pix_nested_torch(nside, ra_t, dec_t)


def pix2ang_nested(nside: int, pix: Tensor) -> Tuple[Tensor, Tensor]:
    """Convert HEALPix NESTED indices to RA/Dec (degrees)."""
    _validate_nside(nside)
    pix_t = _as_int64(pix)
    if _native_cpu(pix_t):
        return _ops.healpix_pix2ang_nested(nside, pix_t.contiguous())
    f_dtype = _float_dtype_for_device(pix_t.device)

    npix = 12 * nside * nside
    _validate_pixels(pix_t, npix, "pixel index out of range for nside")

    ix, iy, face_num = _nest2xyf(nside, pix_t)
    face_num = face_num.clamp(0, 11)  # unchecked while compiling
    jrll, jpll = _face_consts(face_num.device)

    nl4 = 4 * nside
    fact2 = 4.0 / npix
    fact1 = (2 * nside) * fact2

    jr = jrll[face_num] * nside - ix - iy - 1
    north = jr < nside
    south = jr > (3 * nside)
    belt = ~(north | south)
    region = torch.where(
        north,
        torch.ones_like(jr),
        torch.where(south, -torch.ones_like(jr), torch.zeros_like(jr)),
    )
    nr = torch.where(
        north, jr, torch.where(south, nl4 - jr, torch.full_like(jr, nside))
    )
    kshift = torch.where(belt, (jr - nside) & 1, torch.zeros_like(jr))
    nr_f = nr.clamp_min(1).to(f_dtype)

    tmp = torch.square(nr_f) * fact2
    z_belt = (2 * nside - jr).to(f_dtype) * fact1
    _theta, dec = _ring_theta_lat(tmp, region, z_belt)

    jp = torch.div(jpll[face_num] * nr + ix - iy + 1 + kshift, 2, rounding_mode="floor")
    jp = torch.where(jp > nl4, jp - nl4, jp)
    jp = torch.where(jp < 1, jp + nl4, jp)

    phi = (jp.to(f_dtype) - 0.5 * (kshift.to(f_dtype) + 1.0)) * ((math.pi / 2.0) / nr_f)
    ra = torch.remainder(torch.rad2deg(phi), 360.0)
    return ra, dec


def ring2nest(nside: int, pix_ring: Tensor) -> Tensor:
    """Convert RING pixel indices to NESTED."""
    _validate_nside(nside)
    pix_ring_t = _as_int64(pix_ring)
    if _ops.has_fixed_dispatcher() and pix_ring_t.device.type == "cpu":
        return _ops.healpix_ring2nest(nside, pix_ring_t.contiguous())
    ix, iy, face_num = _ring2xyf(nside, pix_ring_t)
    return _xyf2nest(nside, ix, iy, face_num)


def nest2ring(nside: int, pix_nest: Tensor) -> Tensor:
    """Convert NESTED pixel indices to RING."""
    _validate_nside(nside)
    pix_nest_t = _as_int64(pix_nest)
    if _ops.has_fixed_dispatcher() and pix_nest_t.device.type == "cpu":
        return _ops.healpix_nest2ring(nside, pix_nest_t.contiguous())
    ix, iy, face_num = _nest2xyf(nside, pix_nest_t)
    return _xyf2ring(nside, ix, iy, face_num)


def nside2npix(nside: int) -> int:
    """Return number of pixels for a given NSIDE."""
    _validate_nside(nside)
    return 12 * nside * nside


def order2nside(order: int) -> int:
    """Return NSIDE for a given HEALPix order (nside = 2**order)."""
    if order < 0:
        raise ValueError("order must be non-negative")
    nside = 1 << order
    _validate_nside(nside)
    return nside


def nside2order(nside: int) -> int:
    """Return HEALPix order for a given NSIDE (order = log2(nside))."""
    _validate_nside(nside)
    return int(nside.bit_length() - 1)


def npix2nside(npix: int) -> int:
    """Return NSIDE for a given number of pixels."""
    if npix <= 0 or (npix % 12) != 0:
        raise ValueError("npix must be positive and divisible by 12")
    nside_sq = npix // 12
    nside = int(math.isqrt(nside_sq))
    if nside * nside != nside_sq:
        raise ValueError("npix does not correspond to a valid HEALPix nside")
    _validate_nside(nside)
    return nside


def isnsideok(nside: Tensor | int | list[int], nest: bool = False) -> bool | Tensor:
    """
    Check whether NSIDE values are valid.

    Returns bool for scalar input and bool tensor for array-like input.
    """
    t = torch.as_tensor(nside)
    int_like = torch.ones_like(t, dtype=torch.bool)
    if t.is_floating_point():
        int_like = torch.isfinite(t) & (t == torch.floor(t))
    v = t.to(torch.int64)
    ok = int_like & (v > 0) & (v <= (1 << 29)) & ((v & (v - 1)) == 0)
    if not nest:
        # Keep ring mode semantics simple and consistent with power-of-two NSIDE.
        ok = ok & torch.isfinite(v.to(torch.float64))
    if t.ndim == 0:
        return bool(ok.item())
    return ok


def isnpixok(npix: Tensor | int | list[int]) -> bool | Tensor:
    """
    Check whether NPIX values are valid HEALPix sizes.

    Returns bool for scalar input and bool tensor for array-like input.
    """
    t = torch.as_tensor(npix)
    int_like = torch.ones_like(t, dtype=torch.bool)
    if t.is_floating_point():
        int_like = torch.isfinite(t) & (t == torch.floor(t))
    v = t.to(torch.int64)
    base = int_like & (v > 0) & ((v % 12) == 0)
    nside_sq = torch.where(base, v // 12, torch.ones_like(v))
    nside = _isqrt(nside_sq)
    ok = (
        base
        & (nside <= (1 << 29))
        & (nside * nside == nside_sq)
        & ((nside & (nside - 1)) == 0)
    )
    if t.ndim == 0:
        return bool(ok.item())
    return ok


def nside2pixarea(nside: int, degrees: bool = False) -> float:
    """Return pixel area for a given NSIDE."""
    _validate_nside(nside)
    area_sr = 4.0 * math.pi / float(nside2npix(nside))
    if not degrees:
        return area_sr
    return area_sr * ((180.0 / math.pi) * (180.0 / math.pi))


def nside2resol(nside: int, arcmin: bool = False) -> float:
    """Return approximate resolution (sqrt pixel area)."""
    resol_rad = math.sqrt(nside2pixarea(nside, degrees=False))
    if not arcmin:
        return resol_rad
    return resol_rad * (180.0 * 60.0 / math.pi)


def ang2pix(
    nside: int,
    theta: Tensor | float,
    phi: Tensor | float,
    nest: bool = False,
    lonlat: bool = False,
) -> Tensor:
    """
    Convert angles to pixel indices.

    If lonlat is False, inputs are (theta, phi) in radians with theta=colatitude.
    If lonlat is True, inputs are (lon, lat) in degrees.

    Args:
        nside: HEALPix resolution parameter.
        theta: Colatitude in radians, or longitude in degrees with ``lonlat=True``.
        phi: Longitude in radians, or latitude in degrees with ``lonlat=True``.
        nest: Use nested rather than ring ordering.
        lonlat: Interpret inputs as longitude and latitude in degrees.

    Returns:
        Integer HEALPix indices with the broadcast input shape.

    Examples:
        >>> import torch
        >>> from torchsky.sphere import ang2pix
        >>> ang2pix(1, torch.tensor([0.0]), torch.tensor([0.0]), lonlat=True).shape
        torch.Size([1])
    """
    if lonlat:
        lon = _as_float64(theta)
        lat = _as_float64(phi)
    else:
        theta_t = _as_float64(theta)
        phi_t = _as_float64(phi)
        lon = torch.rad2deg(phi_t)
        lat = 90.0 - torch.rad2deg(theta_t)
    if nest:
        return ang2pix_nested(nside, lon, lat)
    return ang2pix_ring(nside, lon, lat)


def ang2vec(theta: Tensor | float, phi: Tensor | float, lonlat: bool = False) -> Tensor:
    """
    Convert angles to unit vectors.

    If lonlat=False, input is (theta, phi) in radians with theta=colatitude.
    If lonlat=True, input is (lon, lat) in degrees.
    Returns vectors with last dimension 3.
    """
    if lonlat:
        lon = _as_float64(theta)
        lat = _as_float64(phi)
    else:
        theta_t = _as_float64(theta)
        phi_t = _as_float64(phi)
        lon = torch.rad2deg(phi_t)
        lat = 90.0 - torch.rad2deg(theta_t)
    x, y, z = lonlat_to_xyz(lon, lat)
    return torch.stack([x, y, z], dim=-1)


def pix2ang(
    nside: int, ipix: Tensor, nest: bool = False, lonlat: bool = False
) -> Tuple[Tensor, Tensor]:
    """
    Convert pixel indices to angles.

    If lonlat is False, returns (theta, phi) in radians with theta=colatitude.
    If lonlat is True, returns (lon, lat) in degrees.

    Args:
        nside: HEALPix resolution parameter.
        ipix: Integer pixel indices.
        nest: Use nested rather than ring ordering.
        lonlat: Return longitude and latitude in degrees.

    Returns:
        Two tensors containing angular coordinates.

    Examples:
        >>> import torch
        >>> from torchsky.sphere import pix2ang
        >>> lon, lat = pix2ang(1, torch.tensor([0]), lonlat=True)
        >>> lon.shape == lat.shape
        True
    """
    if nest:
        lon, lat = pix2ang_nested(nside, ipix)
    else:
        lon, lat = pix2ang_ring(nside, ipix)
    if lonlat:
        return lon, lat
    theta = torch.deg2rad(90.0 - lat)
    phi = torch.deg2rad(lon)
    return theta, phi


def vec2ang(
    vectors: Tensor | Sequence[Sequence[float]], lonlat: bool = False
) -> Tuple[Tensor, Tensor]:
    """
    Convert vectors to angles.

    Accepts shape (..., 3) or (3, ...). If lonlat=False returns (theta, phi)
    in radians, otherwise returns (lon, lat) in degrees.
    """
    v = torch.as_tensor(vectors)
    if v.ndim == 0:
        raise ValueError("vectors must be at least 1D")
    if v.shape[-1] == 3:
        x, y, z = v[..., 0], v[..., 1], v[..., 2]
    elif v.shape[0] == 3:
        x, y, z = v[0], v[1], v[2]
    else:
        raise ValueError("vectors must have shape (...,3) or (3,...)")
    lon, lat = xyz_to_lonlat(x, y, z)
    if lonlat:
        return lon, lat
    theta = torch.deg2rad(90.0 - lat)
    phi = torch.deg2rad(lon)
    return theta, phi


def vec2pix(nside: int, x: Tensor, y: Tensor, z: Tensor, nest: bool = False) -> Tensor:
    """Convert Cartesian vectors to pixel indices."""
    lon, lat = xyz_to_lonlat(x, y, z)
    if nest:
        return ang2pix_nested(nside, lon, lat)
    return ang2pix_ring(nside, lon, lat)


def pix2vec(
    nside: int, ipix: Tensor, nest: bool = False
) -> Tuple[Tensor, Tensor, Tensor]:
    """Convert pixel indices to Cartesian vectors."""
    lon, lat = pix2ang(nside, ipix, nest=nest, lonlat=True)
    return lonlat_to_xyz(lon, lat)


def boundaries(nside: int, pix: Tensor, step: int = 1, nest: bool = False) -> Tensor:
    """
    Return pixel boundary points as Cartesian vectors.

    Output shape:
    - scalar `pix`: (3, 4*step)
    - 1D `pix`: (Npix, 3, 4*step)
    """
    _validate_nside(nside)
    if step <= 0:
        raise ValueError("step must be positive")

    pix_t = _as_int64(pix)
    if pix_t.ndim > 1:
        raise ValueError("Array has to be one dimensional")
    scalar_input = pix_t.ndim == 0
    if scalar_input:
        pix_t = pix_t.reshape(1)

    if not nest:
        pix_t = ring2nest(nside, pix_t)

    npix = nside2npix(nside)
    if torch.any((pix_t < 0) | (pix_t >= npix)):
        raise ValueError("pixel index out of range for nside")

    ix, iy, face_num = _nest2xyf(nside, pix_t)
    f_dtype = _float_dtype_for_device(pix_t.device)
    ix_f = ix.to(f_dtype)
    iy_f = iy.to(f_dtype)

    u = (torch.arange(step, device=pix_t.device, dtype=f_dtype) / step).unsqueeze(0)
    x = torch.cat(
        [
            (ix_f + 1.0).unsqueeze(1) - u,
            ix_f.unsqueeze(1).expand(-1, step),
            ix_f.unsqueeze(1) + u,
            (ix_f + 1.0).unsqueeze(1).expand(-1, step),
        ],
        dim=1,
    )
    y = torch.cat(
        [
            (iy_f + 1.0).unsqueeze(1).expand(-1, step),
            (iy_f + 1.0).unsqueeze(1) - u,
            iy_f.unsqueeze(1).expand(-1, step),
            iy_f.unsqueeze(1) + u,
        ],
        dim=1,
    )

    face_e = face_num.unsqueeze(1).expand_as(x)
    jrll = _JRLL.to(device=pix_t.device)[face_e].to(f_dtype)
    jpll = _JPLL.to(device=pix_t.device)[face_e].to(f_dtype)

    jr = jrll * nside - x - y
    fact2 = 4.0 / npix
    fact1 = (2 * nside) * fact2
    nr = torch.empty_like(jr)
    z = torch.empty_like(jr)

    north = jr < nside
    south = jr > (3 * nside)
    equat = ~(north | south)

    nr[north] = jr[north]
    z[north] = 1.0 - torch.square(nr[north]) * fact2
    nr[south] = 4 * nside - jr[south]
    z[south] = -1.0 + torch.square(nr[south]) * fact2
    nr[equat] = nside
    z[equat] = (2 * nside - jr[equat]) * fact1

    nr_safe = torch.where(nr == 0.0, torch.ones_like(nr), nr)
    phi = (jpll * nr + x - y) * (math.pi * 0.25) / nr_safe
    phi = torch.where(nr == 0.0, torch.zeros_like(phi), phi)

    # sin(theta) from 1 - |z| in the caps: sqrt(1 - z^2) of z = 1 - tmp loses
    # half of the digits next to the poles.
    tmp = torch.square(nr) * fact2
    cap = north | south
    sin_lat = torch.where(cap, torch.sign(z) * (1.0 - tmp), z)
    cos_lat = torch.where(
        cap,
        torch.sqrt(torch.clamp(tmp * (2.0 - tmp), min=0.0)),
        torch.sqrt(torch.clamp((1.0 - z) * (1.0 + z), min=0.0)),
    )
    sin_phi = torch.sin(phi)
    cos_phi = torch.cos(phi)
    xyz = torch.stack([cos_lat * cos_phi, cos_lat * sin_phi, sin_lat], dim=1)

    if scalar_input:
        return xyz[0]
    return xyz


def neighbors(nside: int, ipix: Tensor | int, nest: bool = False) -> Tensor:
    """
    Return 8 neighbors (SW, W, NW, N, NE, E, SE, S) for input pixels.

    Output shape is `ipix.shape + (8,)` for array inputs and `(8,)` for scalars.
    Missing neighbors are set to -1.
    """
    _validate_nside(nside)
    pix_t = _as_int64(ipix)
    scalar_input = pix_t.ndim == 0
    if scalar_input:
        pix_flat = pix_t.reshape(1)
    else:
        pix_flat = pix_t.reshape(-1)

    npix = nside2npix(nside)
    if torch.any((pix_flat < 0) | (pix_flat >= npix)):
        raise ValueError("ipix out of range for nside")

    if pix_flat.device.type == "cpu" and _ops.has_fixed_dispatcher():
        if nest:
            return _ops.healpix_neighbors_nested(nside, pix_t.contiguous())
        return _ops.healpix_neighbors_ring(nside, pix_t.contiguous())

    if nest:
        ix, iy, face = _nest2xyf(nside, pix_flat)
    else:
        ix, iy, face = _ring2xyf(nside, pix_flat)

    nsm1 = nside - 1
    out = torch.full(
        (pix_flat.numel(), 8), -1, dtype=torch.int64, device=pix_flat.device
    )
    interior = (ix > 0) & (ix < nsm1) & (iy > 0) & (iy < nsm1)

    xoff = _NB_XOFFSET.to(device=pix_flat.device)
    yoff = _NB_YOFFSET.to(device=pix_flat.device)

    if torch.any(interior):
        ix_i = ix[interior]
        iy_i = iy[interior]
        face_i = face[interior]
        x_int = ix_i.unsqueeze(1) + xoff.unsqueeze(0)
        y_int = iy_i.unsqueeze(1) + yoff.unsqueeze(0)
        f_int = face_i.unsqueeze(1).expand_as(x_int)
        if nest:
            vals = _xyf2nest(
                nside, x_int.reshape(-1), y_int.reshape(-1), f_int.reshape(-1)
            )
        else:
            vals = _xyf2ring(
                nside, x_int.reshape(-1), y_int.reshape(-1), f_int.reshape(-1)
            )
        out[interior] = vals.reshape(-1, 8)

    boundary = ~interior
    if torch.any(boundary):
        ix_b = ix[boundary]
        iy_b = iy[boundary]
        face_b = face[boundary]

        facearr = _NB_FACEARRAY.to(device=pix_flat.device)
        swaparr = _NB_SWAPARRAY.to(device=pix_flat.device)
        band = face_b >> 2

        # xoff, yoff are [8]
        # ix_b, iy_b are [Nb]
        # x, y: [Nb, 8]
        x = ix_b.unsqueeze(1) + xoff.unsqueeze(0)
        y = iy_b.unsqueeze(1) + yoff.unsqueeze(0)

        nbnum = torch.full_like(x, 4)

        lx = x < 0
        gx = x >= nside
        x = torch.where(lx, x + nside, x)
        x = torch.where(gx, x - nside, x)
        nbnum = torch.where(lx, nbnum - 1, nbnum)
        nbnum = torch.where(gx, nbnum + 1, nbnum)

        ly = y < 0
        gy = y >= nside
        y = torch.where(ly, y + nside, y)
        y = torch.where(gy, y - nside, y)
        nbnum = torch.where(ly, nbnum - 3, nbnum)
        nbnum = torch.where(gy, nbnum + 3, nbnum)

        # nbnum is [Nb, 8], face_b is [Nb]
        # we want facearr[nbnum, face_b]
        # advanced indexing: facearr is [9, 12]
        # result f is [Nb, 8]
        f = facearr[nbnum, face_b.unsqueeze(1).expand_as(nbnum)]
        valid = f >= 0

        # result_b will store neighbor pixels for boundary points
        result_b = torch.full_like(x, -1, dtype=torch.int64)

        if torch.any(valid):
            # band is [Nb], nbnum is [Nb, 8]
            bits = swaparr[nbnum, band.unsqueeze(1).expand_as(nbnum)]

            xv = x[valid]
            yv = y[valid]
            bv = bits[valid]
            fv = f[valid]

            flip_x = (bv & 1) != 0
            flip_y = (bv & 2) != 0
            swap_xy = (bv & 4) != 0

            xv = torch.where(flip_x, nside - xv - 1, xv)
            yv = torch.where(flip_y, nside - yv - 1, yv)
            x_new = torch.where(swap_xy, yv, xv)
            y_new = torch.where(swap_xy, xv, yv)

            if nest:
                vals = _xyf2nest(nside, x_new, y_new, fv)
            else:
                vals = _xyf2ring(nside, x_new, y_new, fv)

            result_b[valid] = vals

        out[boundary] = result_b

    if scalar_input:
        return out[0]
    return out.reshape(*pix_t.shape, 8)


def get_all_neighbours(
    nside: int,
    theta: Tensor | float | int,
    phi: Tensor | float | None = None,
    nest: bool = False,
    lonlat: bool = False,
) -> Tensor:
    """
    Healpy-compatible neighbors API.

    If `phi` is None, `theta` is interpreted as pixel index. Otherwise,
    `(theta, phi)` are interpreted as angular coordinates.
    """
    _validate_nside(nside)
    if phi is None:
        pix = _as_int64(theta)
    else:
        pix = ang2pix(nside, theta, phi, nest=nest, lonlat=lonlat)
    neigh = neighbors(nside, pix, nest=nest)
    if neigh.ndim == 1:
        return neigh
    return torch.movedim(neigh, -1, 0)


def get_interp_weights(
    nside: int,
    theta: Tensor | float | int,
    phi: Tensor | float | None = None,
    nest: bool = False,
    lonlat: bool = False,
) -> tuple[Tensor, Tensor]:
    """
    Healpy-compatible bilinear interpolation neighbors and weights.

    Returns `(pixels, weights)` with shape `(4, ...)`.
    """
    _validate_nside(nside)
    if phi is None:
        pix_in = _as_int64(theta)
        if pix_in.device.type == "cpu" and _ops.has_fixed_dispatcher():
            # Fast path for pixel-center queries: convert to lon/lat and reuse
            # C++ interpolation-weight kernels. This matches healpy semantics.
            lon_t, lat_t = pix2ang(nside, pix_in, nest=nest, lonlat=True)
            if nest:
                return _ops.healpix_get_interp_weights_nested(
                    nside, lon_t.contiguous(), lat_t.contiguous()
                )
            return _ops.healpix_get_interp_weights_ring(
                nside, lon_t.contiguous(), lat_t.contiguous()
            )

        pix_ring = nest2ring(nside, pix_in) if nest else pix_in
        theta_t, phi_t = _pix2thetaphi_ring(nside, pix_ring)
    else:
        lon_or_theta = _as_float64(theta)
        lat_or_phi = _as_float64(phi)
        lon_or_theta, lat_or_phi = torch.broadcast_tensors(lon_or_theta, lat_or_phi)
        if lonlat:
            lon_t = lon_or_theta
            lat_t = lat_or_phi
        else:
            lon_t = torch.rad2deg(lat_or_phi)
            lat_t = 90.0 - torch.rad2deg(lon_or_theta)
        _validate_lonlat(lon_t, lat_t)

        if (
            lon_t.device.type == "cpu"
            and _ops.has_fixed_dispatcher()
            and not lon_t.requires_grad
            and not lat_t.requires_grad
        ):
            if nest:
                return _ops.healpix_get_interp_weights_nested(
                    nside, lon_t.contiguous(), lat_t.contiguous()
                )
            return _ops.healpix_get_interp_weights_ring(
                nside, lon_t.contiguous(), lat_t.contiguous()
            )

        theta_t = torch.deg2rad(90.0 - lat_t)
        phi_t = torch.deg2rad(lon_t)

    shape = theta_t.shape
    pix, wgt = _interp_weights_torch(nside, theta_t.reshape(-1), phi_t.reshape(-1))
    if nest:
        pix = ring2nest(nside, pix)
    return pix.reshape(4, *shape), wgt.reshape(4, *shape)


def _ring_pair(
    nside: int, ring: Tensor, phi: Tensor, fd: torch.dtype
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Bracketing pixels of ``phi`` on ``ring`` (1..4 nside-1) and the weight of the second."""
    sp, nr, th, shift = _get_ring_info2(nside, ring)
    dphi = (2.0 * math.pi) / nr.to(fd)
    sh = shift.to(fd)
    i1 = torch.floor(phi / dphi - 0.5 * sh).to(torch.int64)
    wt = (phi - (i1.to(fd) + 0.5 * sh) * dphi) / dphi
    i2 = i1 + 1
    i1 = torch.where(i1 < 0, i1 + nr, i1)
    i2 = torch.where(i2 >= nr, i2 - nr, i2)
    return sp + i1, sp + i2, wt, th


def _interp_weights_torch(
    nside: int, theta: Tensor, phi: Tensor
) -> tuple[Tensor, Tensor]:
    """Branch-free healpy ``get_interpol`` on flat RING inputs (radians).

    The polar (``top``/``bottom``) and regular regimes are all evaluated and
    selected with ``where``; rings outside ``1..4 nside-1`` are clamped for the
    unselected regime and their denominators kept non-zero, so neither the
    values nor the gradients of the selected regime see NaNs.
    """
    fd = _float_dtype_for_device(theta.device)
    theta_f = _as_float64(theta)
    phi_f = torch.remainder(_as_float64(phi), 2.0 * math.pi)
    nl4 = 4 * nside
    npix = nside2npix(nside)

    ir1 = _ring_above(nside, torch.cos(theta_f))
    ir2 = ir1 + 1
    top = ir1 == 0
    bottom = ir2 == nl4

    p0, p1, wt1, theta1 = _ring_pair(nside, ir1.clamp(1, nl4 - 1), phi_f, fd)
    p2, p3, wt2, theta2 = _ring_pair(nside, ir2.clamp(1, nl4 - 1), phi_f, fd)
    w0 = 1.0 - wt1
    w1 = wt1
    w2 = 1.0 - wt2
    w3 = wt2

    one = torch.ones_like(theta_f)

    def _span(selected: Tensor, width: Tensor) -> Tensor:
        # At very high nside two rings (or a ring and the pole) can share one
        # float32 colatitude (MPS); the zero width then yields weight 0 for the
        # far ring instead of 0/0.
        return torch.where(selected & (width != 0.0), width, one)

    # Regular: linear in theta between the two rings.
    wtheta_mid = (theta_f - theta1) / _span(~(top | bottom), theta2 - theta1)
    # North pole: the four "virtual" pixels of ring 0 are ring 1 rotated by pi.
    wtheta_top = theta_f / _span(top, theta2)
    fac_top = (1.0 - wtheta_top) * 0.25
    # South pole: likewise with the last ring.
    wtheta_bot = (theta_f - theta1) / _span(bottom, math.pi - theta1)
    fac_bot = wtheta_bot * 0.25

    w0 = torch.where(
        top,
        fac_top,
        torch.where(bottom, w0 * (1.0 - wtheta_bot) + fac_bot, w0 * (1.0 - wtheta_mid)),
    )
    w1 = torch.where(
        top,
        fac_top,
        torch.where(bottom, w1 * (1.0 - wtheta_bot) + fac_bot, w1 * (1.0 - wtheta_mid)),
    )
    w2 = torch.where(
        top, w2 * wtheta_top + fac_top, torch.where(bottom, fac_bot, w2 * wtheta_mid)
    )
    w3 = torch.where(
        top, w3 * wtheta_top + fac_top, torch.where(bottom, fac_bot, w3 * wtheta_mid)
    )
    p0_top = (p2 + 2) & 3
    p1_top = (p3 + 2) & 3
    p2_bot = ((p0 + 2) & 3) + npix - 4
    p3_bot = ((p1 + 2) & 3) + npix - 4
    p0 = torch.where(top, p0_top, p0)
    p1 = torch.where(top, p1_top, p1)
    p2 = torch.where(bottom, p2_bot, p2)
    p3 = torch.where(bottom, p3_bot, p3)
    return torch.stack([p0, p1, p2, p3], dim=0), torch.stack([w0, w1, w2, w3], dim=0)


def get_interp_val(
    m: Tensor | list[float],
    theta: Tensor | float,
    phi: Tensor | float,
    nest: bool = False,
    lonlat: bool = False,
) -> Tensor:
    """
    Healpy-compatible bilinear interpolation of map values.

    `m` can be shape `(npix,)` or `(nmaps, npix)`.
    """
    m_t = torch.as_tensor(m)
    one_map = m_t.ndim == 1
    if one_map:
        maps = m_t.unsqueeze(0)
    elif m_t.ndim == 2:
        maps = m_t
    else:
        raise ValueError("m must be shape (npix,) or (nmaps, npix)")

    npix = int(maps.shape[1])
    nside = npix2nside(npix)

    if (
        _ops.has_fixed_dispatcher()
        and maps.device.type == "cpu"
        and maps.is_floating_point()
        and not maps.requires_grad
    ):
        lon_or_theta = _as_float64(theta)
        lat_or_phi = _as_float64(phi)
        lon_or_theta, lat_or_phi = torch.broadcast_tensors(lon_or_theta, lat_or_phi)
        if lonlat:
            lon_t = lon_or_theta
            lat_t = lat_or_phi
        else:
            lon_t = torch.rad2deg(lat_or_phi)
            lat_t = 90.0 - torch.rad2deg(lon_or_theta)
        _validate_lonlat(lon_t, lat_t)
        if (
            lon_t.device.type == "cpu"
            and not lon_t.requires_grad
            and not lat_t.requires_grad
        ):
            maps_in = m_t if one_map else maps
            operation = (
                _ops.healpix_get_interp_val_nested
                if nest
                else _ops.healpix_get_interp_val_ring
            )
            return operation(
                nside,
                maps_in.contiguous(),
                lon_t.contiguous(),
                lat_t.contiguous(),
            )

    maps_work = (
        maps
        if maps.is_floating_point()
        else maps.to(dtype=_float_dtype_for_device(maps.device))
    )
    pix, wgt = get_interp_weights(nside, theta, phi, nest=nest, lonlat=lonlat)
    qshape = pix.shape[1:]
    pix_f = pix.reshape(4, -1)
    w_f = wgt.reshape(4, -1).to(dtype=maps_work.dtype)
    # Gather all 4 neighbors in one advanced-indexing pass to reduce
    # index_select dispatch overhead on large query batches.
    neigh_vals = maps_work[:, pix_f]  # [nmaps,4,nq]
    out = (neigh_vals * w_f.unsqueeze(0)).sum(dim=1)
    out = out.reshape(maps.shape[0], *qshape)
    if one_map:
        return out[0]
    return out


def max_pixel_radius(nside: int, degrees: bool = False) -> float:
    """Maximum angular distance from any pixel center to any point of its pixel.

    Closed form of ``Healpix_Base::max_pixrad`` (Górski et al. 2005): the
    largest center-to-vertex angle over the sphere is attained by the pixels
    adjacent to the polar-cap/equatorial transition (``z = 2/3``), between the
    center ``(z = 2/3, phi = pi / (4 nside))`` and the vertex
    ``(z = 1 - (1 - 1/nside)^2 / 3, phi = 0)``. Coverings use this as the
    pixel radius in the ``r + 2 rho`` soundness lemma.
    """
    _validate_nside(nside)

    def _unit(z: float, phi: float) -> tuple[float, float, float]:
        s = math.sqrt((1.0 - z) * (1.0 + z))
        return (s * math.cos(phi), s * math.sin(phi), z)

    ax, ay, az = _unit(2.0 / 3.0, math.pi / (4.0 * nside))
    t = (1.0 - 1.0 / nside) ** 2
    bx, by, bz = _unit(1.0 - t / 3.0, 0.0)
    cross = math.hypot(ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx)
    rad = math.atan2(cross, ax * bx + ay * by + az * bz)
    return math.degrees(rad) if degrees else rad


def pixel_resolution_to_nside(
    resolution: float,
    *,
    degrees: bool = False,
    arcmin: bool = False,
    round_to_power_of_two: bool = True,
) -> int:
    """Approximate NSIDE from angular resolution."""
    if resolution <= 0.0:
        raise ValueError("resolution must be positive")
    if degrees and arcmin:
        raise ValueError("set only one of degrees/arcmin")

    if arcmin:
        res_rad = math.radians(resolution / 60.0)
    elif degrees:
        res_rad = math.radians(resolution)
    else:
        res_rad = resolution

    n_est = math.sqrt(math.pi / 3.0) / res_rad
    if n_est < 1.0:
        return 1
    if not round_to_power_of_two:
        return max(1, int(round(n_est)))

    log2n = math.log2(n_est)
    nside = 1 << max(0, int(round(log2n)))
    _validate_nside(nside)
    return nside


def query_disc(
    nside: int,
    vec: Tensor | Sequence[float],
    radius: float,
    *,
    inclusive: bool = False,
    fact: int = 4,
    nest: bool = False,
    chunk_size: int = 1_000_000,
) -> Tensor:
    """
    Return pixels whose centers fall in a spherical disc.

    `vec` is a 3-vector (x, y, z); `radius` is in radians.
    If `inclusive=True` and ``fact==4`` (healpy default), pad the radius by the
    maximum pixel radius. If ``inclusive=True`` and ``fact!=4``, supersample at
    ``nside * fact`` (pad at the fine nside), then degrade to parent pixels —
    healpy-shaped inclusive overlap. ``fact`` is ignored when ``inclusive=False``.

    Performance: when the C++ extension is available on CPU (or CUDA via a host
    copy of ``vec``), queries use **ring pruning** (``O(nside + output)``).
    Without the extension, and on MPS, a torch quadtree descent selects the same
    pixel centres in ``O(boundary cells x depth + output)``, never ``O(npix)``.
    CUDA copies a length-3 ``vec`` to host; that copy is O(1) and far below 10%
    of nside≥512 wall, so the query stays on the CPU kernel.

    Args:
        nside: HEALPix resolution parameter.
        vec: Cartesian unit vector at the disc center.
        radius: Angular radius in radians.
        inclusive: Include pixels overlapping the disc boundary.
        fact: Oversampling factor for inclusive queries.
        nest: Return nested rather than ring indices.
        chunk_size: Accepted for API compatibility; unused.

    Returns:
        One-dimensional integer tensor of matching pixels.

    Examples:
        >>> from torchsky.sphere import query_disc
        >>> query_disc(1, [1.0, 0.0, 0.0], 0.1).ndim
        1
    """
    _validate_nside(nside)
    radius = float(radius)
    if not math.isfinite(radius) or radius < 0.0 or radius > math.pi:
        raise ValueError("radius must be finite and in [0, pi]")
    fact_i = int(fact)
    if fact_i < 1:
        raise ValueError("fact must be a positive integer")
    if nest and (fact_i & (fact_i - 1)) != 0:
        raise ValueError("for nest ordering, fact must be a power of two")
    v_any = torch.as_tensor(vec).reshape(-1)
    if v_any.numel() != 3:
        raise ValueError("vec must be length-3 Cartesian vector")
    out_device = v_any.device
    if not bool(torch.isfinite(v_any).all()):
        raise ValueError("vec must contain exactly 3 finite values")
    if not bool(torch.any(v_any != 0)):
        raise ValueError("vec norm must be positive")

    # Inclusive fact!=4: supersample at nside*fact, then take parent pixels.
    # fact==4 keeps the historical pad-by-max_pixrad path (tests depend on it).
    if inclusive and fact_i != 4:
        nside_hi = int(nside) * fact_i
        _validate_nside(nside_hi)
        pix_hi = query_disc(
            nside_hi,
            v_any,
            radius,
            inclusive=True,
            fact=4,
            nest=True,
            chunk_size=chunk_size,
        )
        if pix_hi.numel() == 0:
            return (
                pix_hi.to(device=out_device) if pix_hi.device != out_device else pix_hi
            )
        parent = torch.unique(torch.div(pix_hi, fact_i * fact_i, rounding_mode="floor"))
        parent = torch.sort(parent).values
        if nest:
            out = parent
        else:
            out = torch.sort(nest2ring(nside, parent)).values
        return out.to(device=out_device) if out.device != out_device else out

    f_dtype = _float_dtype_for_device(out_device)
    # Keep MPS on-device (float32). CPU/CUDA promote via host for C++ parity.
    if out_device.type == "mps":
        v_dev = v_any.detach().to(dtype=f_dtype).reshape(-1)
        vx, vy, vz = _normalize_xyz(v_dev[0], v_dev[1], v_dev[2])
    else:
        v64 = v_any.detach().to(device="cpu").to(dtype=torch.float64).reshape(-1)
        vx, vy, vz = _normalize_xyz(v64[0], v64[1], v64[2])
    radius_eff = min(
        math.pi,
        radius + (max_pixel_radius(nside, degrees=False) if inclusive else 0.0),
    )
    cos_lim = math.cos(radius_eff)

    use_cpp = out_device.type != "mps" and _ops.has_fixed_dispatcher()
    if use_cpp:
        vec_cpu = torch.stack(
            [
                vx.detach().cpu().to(torch.float64),
                vy.detach().cpu().to(torch.float64),
                vz.detach().cpu().to(torch.float64),
            ]
        ).contiguous()
        out = _ops.healpix_query_disc(nside, vec_cpu, float(cos_lim), nest)
        if out_device.type != "cpu":
            return out.to(device=out_device)
        return out

    # Torch fallback (no extension, or MPS): the same quadtree descent as the
    # native kernel, O(boundary cells x depth + output) instead of O(npix).
    ranges = _disc_nest_ranges_torch(
        int(nside), torch.stack((vx, vy, vz)), float(cos_lim)
    )
    if nest:
        out = _ranges_to_sorted_pixels(ranges)
    else:
        out = _nest_ranges_to_ring_pixels(int(nside), ranges)
    return out.to(device=out_device) if out.device != out_device else out


def query_circle(
    nside: int,
    lon: Tensor | float,
    lat: Tensor | float,
    radius: float,
    *,
    degrees: bool = True,
    inclusive: bool = False,
    fact: int = 4,
    nest: bool = False,
    chunk_size: int = 1_000_000,
) -> Tensor:
    """Lon/lat convenience wrapper around `query_disc`."""
    lon_t = _as_float64(lon)
    lat_t = _as_float64(lat)
    if lon_t.ndim != 0 or lat_t.ndim != 0:
        raise ValueError("lon and lat must be scalar center coordinates")
    x, y, z = lonlat_to_xyz(lon_t, lat_t)
    vec = torch.stack((x, y, z))
    if degrees:
        radius = math.radians(radius)
    return query_disc(
        nside,
        vec,
        radius,
        inclusive=inclusive,
        fact=fact,
        nest=nest,
        chunk_size=chunk_size,
    )


def query_circle_vec(
    nside: int,
    vec: Tensor | Sequence[float],
    radius: float,
    *,
    inclusive: bool = False,
    fact: int = 4,
    nest: bool = False,
    chunk_size: int = 1_000_000,
) -> Tensor:
    """Cartesian-vector convenience wrapper around `query_disc`."""
    return query_disc(
        nside,
        vec,
        radius,
        inclusive=inclusive,
        fact=fact,
        nest=nest,
        chunk_size=chunk_size,
    )


def query_strip(
    nside: int,
    theta1: float,
    theta2: float,
    *,
    nest: bool = False,
    inclusive: bool = False,
) -> Tensor:
    """
    Return pixels in a colatitude strip, with healpy ``query_strip`` semantics.

    For ``theta1 < theta2`` (radians) the pixels whose centre colatitude lies in
    ``(theta1, theta2]``; otherwise the two polar strips with centres north of
    ``theta2`` and south of ``theta1``. Ring membership is decided exactly as in
    healpy (``ring_above``), and ``inclusive=True`` adds the adjacent ring on
    each side, which holds every pixel overlapping the strip.

    Uses **ring pruning** (``O(nside + output)``), not a full ``O(npix)`` map scan.
    """
    _validate_nside(nside)
    ring_ranges = _strip_rings(nside, float(theta1), float(theta2), bool(inclusive))
    if nest:
        return _ring_ranges_nest_pixels(nside, ring_ranges)
    return _ring_range_pixels(nside, ring_ranges)


def query_polygon(
    nside: int,
    vertices: Tensor | Sequence[Sequence[float]] | Sequence[float],
    lat_deg: Tensor | Sequence[float] | None = None,
    *,
    inclusive: bool = False,
    nest: bool = False,
    lonlat: bool = False,
    chunk_size: int = 1_000_000,
) -> Tensor:
    """
    Return pixels with centers inside a spherical polygon.

    Convex polygons use the direct half-space query. Concave simple polygons are
    triangulated by :func:`torchsky.sphere.geom.query_polygon_general`.

    Accepts either:
    - ``query_polygon(nside, lon_deg, lat_deg, ...)`` (1D degree sequences/tensors)
    - ``query_polygon(nside, vertices, lonlat=True, ...)`` with shape ``(N, 2)`` or ``(2, N)``
    - ``query_polygon(nside, vertices, lonlat=False, ...)`` with Cartesian shape ``(N, 3)`` or ``(3, N)``

    Args:
        nside: HEALPix resolution parameter.
        vertices: Longitude samples or two-/three-dimensional polygon vertices.
        lat_deg: Latitude samples for the separate longitude/latitude form.
        inclusive: Include pixels overlapping the polygon boundary.
        nest: Return nested rather than ring indices.
        lonlat: Interpret paired vertices as degrees rather than Cartesian vectors.
        chunk_size: Pixel chunk size for the tensor fallback.

    Returns:
        One-dimensional integer tensor of matching pixels.

    Examples:
        >>> from torchsky.sphere import query_polygon
        >>> callable(query_polygon)
        True
    """
    _validate_nside(nside)
    if lat_deg is not None:
        lon_t = torch.as_tensor(vertices)
        dtype = torch.float32 if lon_t.device.type == "mps" else torch.float64
        lon_t = lon_t.to(dtype=dtype).reshape(-1)
        lat_t = _as_tensor(lat_deg, dtype=dtype, device=lon_t.device).reshape(-1)
        v = torch.stack([lon_t, lat_t], dim=-1)
        lonlat = True
    else:
        t_input = torch.as_tensor(vertices)
        dtype = torch.float32 if t_input.device.type == "mps" else torch.float64
        v = t_input.to(dtype=dtype)
        if v.ndim != 2:
            raise ValueError("vertices must be 2D")

    if lonlat:
        if v.shape[-1] == 2:
            lon = v[:, 0]
            lat = v[:, 1]
        elif v.shape[0] == 2:
            lon = v[0]
            lat = v[1]
        else:
            raise ValueError("lonlat vertices must be shape (N,2) or (2,N)")
        vx, vy, vz = lonlat_to_xyz(lon, lat)
        vv = torch.stack([vx, vy, vz], dim=-1)
    else:
        if v.shape[-1] == 3:
            vv = v
        elif v.shape[0] == 3:
            vv = v.transpose(0, 1)
        else:
            raise ValueError("Cartesian vertices must be shape (N,3) or (3,N)")

    if vv.shape[0] < 3:
        raise ValueError("at least 3 vertices required")
    vv = vv / torch.sqrt((vv * vv).sum(dim=-1, keepdim=True)).clamp_min(1.0e-15)

    # Spherical half-space test: for each oriented edge (v_i -> v_{i+1}),
    # inside points lie on the same side of the edge great-circle plane.
    vnext = torch.roll(vv, shifts=-1, dims=0)
    edge_normals = torch.cross(vv, vnext, dim=1)
    edge_norm = torch.sqrt((edge_normals * edge_normals).sum(dim=1))
    if torch.any(edge_norm <= 1.0e-15):
        raise ValueError("degenerate polygon: repeated or colinear adjacent vertices")
    nvert = int(vv.shape[0])
    side = edge_normals @ vv.transpose(0, 1)
    idx = torch.arange(nvert, device=vv.device)
    incident = torch.zeros((nvert, nvert), dtype=torch.bool, device=vv.device)
    incident[idx, idx] = True
    incident[idx, (idx + 1) % nvert] = True
    off_edge = side.masked_fill(incident, 0.0)
    if bool((off_edge > 1.0e-12).any()) and bool((off_edge < -1.0e-12).any()):
        # Circular dependency: geom uses this convex path for triangulated pieces.
        from .geom import query_polygon_general

        lon, lat = xyz_to_lonlat(vv[:, 0], vv[:, 1], vv[:, 2])
        return query_polygon_general(nside, lon, lat, nest=nest, inclusive=inclusive)
    edge_normals = edge_normals / edge_norm.unsqueeze(1)

    centroid = vv.mean(dim=0)
    centroid_norm = torch.sqrt((centroid * centroid).sum())
    if centroid_norm <= 1.0e-15:
        raise ValueError("degenerate polygon: centroid norm too small")
    centroid = centroid / centroid_norm

    # Vectorized orientation check for all edges
    # For each edge i, we need a probe point (v_{i+2} or centroid) to determine side sign.
    # edge_normals: [M, 3], vv: [M, 3]

    # Try using next-next vertex as probe for each edge
    probes = torch.roll(vv, shifts=-2, dims=0)
    dot_probes = (probes * edge_normals).sum(dim=1)

    # If any probe is on the edge, fallback to centroid for that edge
    near_edge = dot_probes.abs() <= 1.0e-12
    if near_edge.any():
        dot_centroid = (centroid.unsqueeze(0) * edge_normals).sum(dim=1)
        dot_probes = torch.where(near_edge, dot_centroid, dot_probes)

    if (dot_probes.abs() <= 1.0e-12).any():
        raise ValueError("degenerate polygon orientation")

    edge_normals = edge_normals * torch.sign(dot_probes).unsqueeze(1)

    tol = math.sin(max_pixel_radius(nside, degrees=False)) if inclusive else 0.0

    d_cent = (centroid.unsqueeze(0) * edge_normals).sum(dim=1)
    centroid_inside = bool(torch.all(d_cent >= -1.0e-11).item())
    cap_radius: float | None = None
    if centroid_inside:
        cos_dots = torch.clamp(vv @ centroid, -1.0, 1.0)
        cap_radius = float(torch.acos(cos_dots).max().item())
        if inclusive:
            cap_radius += max_pixel_radius(nside, degrees=False)
        cap_radius = min(math.pi, cap_radius)

    use_native_cap = (
        not nest
        and centroid_inside
        and vv.device.type != "mps"
        and _ops.has_fixed_dispatcher()
    )
    if not use_native_cap:
        # Hierarchical: interior cells become ranges; only boundary cells are
        # refined, and pixel centres are tested exactly as in the flat path
        # (half-spaces, and the circumscribing cap when the centroid is inside).
        # Also the RING path without the native ring-pruned cap, so no query
        # costs O(npix).
        normals_cpu = edge_normals.detach().to(device="cpu").to(torch.float64)
        cap_c = centroid.detach().to(device="cpu").to(torch.float64)
        cap_cos = math.cos(cap_radius) if cap_radius is not None else None

        def _classify(centres: Tensor, radius: float) -> tuple[Tensor, Tensor]:
            side = centres @ normals_cpu.transpose(0, 1)
            chord = 2.0 * math.sin(0.5 * radius)
            inside = (side.min(dim=1).values - chord) >= (-tol + 1.0e-9)
            outside = ((side + chord) < (-tol - 1.0e-9)).any(dim=1)
            if cap_radius is not None:
                dist = torch.atan2(
                    torch.linalg.cross(centres, cap_c.expand_as(centres), dim=1).norm(
                        dim=1
                    ),
                    centres @ cap_c,
                )
                inside = inside & ((dist + radius) <= (cap_radius - 1.0e-9))
                outside = outside | ((dist - radius) >= (cap_radius + 1.0e-9))
            return inside, outside

        def _leaf(pix: Tensor) -> Tensor:
            x, y, z = pix2vec(nside, pix, nest=True)
            p = torch.stack((x, y, z), dim=-1)
            keep = torch.all(p @ normals_cpu.transpose(0, 1) >= -tol, dim=1)
            if cap_cos is not None:
                keep = keep & ((p @ cap_c) >= cap_cos)
            return keep

        ranges = _nest_ranges_by_descent(nside, _classify, _leaf)
        if nest:
            return _ranges_to_sorted_pixels(ranges).to(device=vv.device)
        return _nest_ranges_to_ring_pixels(nside, ranges).to(device=vv.device)

    def _filter_pixel_chunk(pix: Tensor) -> Tensor:
        if pix.numel() == 0:
            return pix
        x, y, z = pix2vec(nside, pix, nest=nest)
        p = torch.stack([x, y, z], dim=-1)
        side = p @ edge_normals.transpose(0, 1)
        keep = torch.all(side >= -tol, dim=1)
        return pix[keep]

    # Native ring-pruned cap (O(nside + cap pixels)), then exact half-spaces.
    assert cap_radius is not None
    cand = query_disc(
        nside,
        centroid,
        cap_radius,
        inclusive=False,
        nest=False,
        chunk_size=chunk_size,
    )
    parts: list[Tensor] = []
    n_c = int(cand.numel())
    for start in range(0, n_c, chunk_size):
        sl = cand[start : start + chunk_size]
        kept = _filter_pixel_chunk(sl)
        if kept.numel() > 0:
            parts.append(kept)
    if not parts:
        return torch.empty((0,), dtype=torch.int64, device=vv.device)
    return torch.cat(parts, dim=0)


def query_polygon_vec(
    nside: int,
    vertices: Tensor | Sequence[Sequence[float]],
    *,
    inclusive: bool = False,
    nest: bool = False,
    chunk_size: int = 1_000_000,
) -> Tensor:
    """Cartesian-vector convenience wrapper around `query_polygon`."""
    return query_polygon(
        nside,
        vertices,
        inclusive=inclusive,
        nest=nest,
        lonlat=False,
        chunk_size=chunk_size,
    )


def pixel_ranges_to_pixels(
    pixel_ranges: Tensor | Sequence[Sequence[int]], *, inclusive: bool = False
) -> Tensor:
    """
    Expand pixel ranges into explicit pixels.

    Range semantics match `hpgeom.pixel_ranges_to_pixels`:
    - `inclusive=False`: each row is [start, stop) (half-open)
    - `inclusive=True`: each row is [start, stop] (closed)
    """
    ranges = torch.as_tensor(pixel_ranges, dtype=torch.int64)
    if ranges.ndim != 2 or ranges.shape[1] != 2:
        raise ValueError("pixel_ranges must have shape (N, 2)")
    if ranges.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=ranges.device)

    # Vectorized expansion of ranges
    starts = ranges[:, 0]
    stops = ranges[:, 1] + (1 if inclusive else 0)
    lengths = torch.clamp(stops - starts, min=0)

    total_len = int(lengths.sum().item())
    if total_len == 0:
        return torch.empty((0,), dtype=torch.int64, device=ranges.device)

    # Create an array of indices into the original ranges for each output pixel
    # e.g. ranges=[[10,12], [20,21]] -> range_idx=[0, 0, 1]
    range_idx = torch.repeat_interleave(
        torch.arange(ranges.shape[0], device=ranges.device), lengths
    )

    # Offsets within each range
    # range_idx=[0, 0, 1] -> range_offsets=[0, 1, 0]
    # We can compute this with a cumulative sum and resets at range boundaries
    pixel_offsets = torch.arange(total_len, device=ranges.device)
    range_start_indices = torch.cat(
        [
            torch.zeros(1, dtype=torch.int64, device=ranges.device),
            torch.cumsum(lengths[:-1], dim=0),
        ]
    )

    # Subtract start index of each range to get 0-based offset within range
    pixel_offsets = pixel_offsets - range_start_indices[range_idx]

    # Output is starts[range_idx] + pixel_offsets
    return starts[range_idx] + pixel_offsets


def pixels_to_pixel_ranges(pixels: Tensor | Sequence[int]) -> Tensor:
    """
    Compress explicit pixels into sorted disjoint half-open ranges [start, stop).

    Unlike `hpgeom`, this helper is torchsky-native and canonicalizes by
    sorting + deduplicating input pixels before range construction.
    """
    pix = torch.as_tensor(pixels, dtype=torch.int64).reshape(-1)
    if pix.numel() == 0:
        return torch.empty((0, 2), dtype=torch.int64, device=pix.device)

    pix = torch.unique(pix, sorted=True)
    if pix.numel() == 1:
        return torch.stack([pix, pix + 1], dim=1)

    gap = pix[1:] != (pix[:-1] + 1)
    start_idx = torch.cat(
        [
            torch.zeros((1,), dtype=torch.int64, device=pix.device),
            torch.nonzero(gap, as_tuple=False).flatten() + 1,
        ]
    )
    end_idx = torch.cat(
        [
            torch.nonzero(gap, as_tuple=False).flatten() + 1,
            torch.tensor([pix.numel()], dtype=torch.int64, device=pix.device),
        ]
    )
    starts = pix.index_select(0, start_idx)
    stops = pix.index_select(0, end_idx - 1) + 1
    return torch.stack([starts, stops], dim=1)


def upgrade_pixel_ranges(
    nside: int,
    pixel_ranges: Tensor | Sequence[Sequence[int]],
    nside_upgrade: int,
) -> Tensor:
    """
    Upgrade NESTED pixel ranges from `nside` to `nside_upgrade`.

    Semantics match `hpgeom.upgrade_pixel_ranges`, where half-open ranges
    [start, stop) scale by `4**levels`.
    """
    _validate_nside(nside)
    _validate_nside(nside_upgrade)
    if nside_upgrade < nside:
        raise ValueError("The value for nside_upgrade must be >= nside.")
    ranges = torch.as_tensor(pixel_ranges, dtype=torch.int64)
    if ranges.ndim != 2 or ranges.shape[1] != 2:
        raise ValueError("pixel_ranges must have shape (N, 2)")
    if ranges.numel() == 0:
        return torch.empty((0, 2), dtype=torch.int64, device=ranges.device)
    if nside_upgrade == nside:
        return ranges.clone()

    levels = nside2order(nside_upgrade) - nside2order(nside)
    factor = 1 << (2 * levels)
    return ranges * factor


def degrade_pixel_ranges(
    nside: int,
    pixel_ranges: Tensor | Sequence[Sequence[int]],
    nside_degrade: int,
) -> Tensor:
    """
    Degrade NESTED pixel ranges from `nside` to `nside_degrade`.

    Half-open child ranges [start, stop) map to every parent pixel touched by at
    least one child. Output ranges are sorted, disjoint, and half-open.
    """
    _validate_nside(nside)
    _validate_nside(nside_degrade)
    if nside_degrade > nside:
        raise ValueError("The value for nside_degrade must be <= nside.")
    ranges = torch.as_tensor(pixel_ranges, dtype=torch.int64)
    if ranges.ndim != 2 or ranges.shape[1] != 2:
        raise ValueError("pixel_ranges must have shape (N, 2)")
    if ranges.numel() == 0:
        return torch.empty((0, 2), dtype=torch.int64, device=ranges.device)
    if nside_degrade == nside:
        return pixels_to_pixel_ranges(pixel_ranges_to_pixels(ranges, inclusive=False))

    levels = nside2order(nside) - nside2order(nside_degrade)
    factor = 1 << (2 * levels)
    starts = torch.div(ranges[:, 0], factor, rounding_mode="floor")
    stops = torch.div(ranges[:, 1] + factor - 1, factor, rounding_mode="floor")
    parent_ranges = torch.stack((starts, stops), dim=1)
    valid = parent_ranges[:, 1] > parent_ranges[:, 0]
    if not bool(valid.any()):
        return torch.empty((0, 2), dtype=torch.int64, device=ranges.device)
    parent_pixels = pixel_ranges_to_pixels(parent_ranges[valid], inclusive=False)
    return pixels_to_pixel_ranges(parent_pixels)


def upgrade_pixels(
    nside: int,
    pixels: Tensor | Sequence[int],
    nside_upgrade: int,
    *,
    nest: bool = False,
) -> Tensor:
    """
    Upgrade explicit HEALPix pixel ids to a finer resolution.

    Each input pixel expands to all child pixels at `nside_upgrade`. Duplicates are
    removed and output is sorted in the requested output ordering.
    """
    _validate_nside(nside)
    _validate_nside(nside_upgrade)
    if nside_upgrade < nside:
        raise ValueError("The value for nside_upgrade must be >= nside.")

    pix = _as_int64(pixels).reshape(-1)
    if pix.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=pix.device)
    npix = nside2npix(nside)
    if bool(((pix < 0) | (pix >= npix)).any()):
        raise ValueError("pixels contain values outside the input nside range")
    if nside_upgrade == nside:
        return torch.unique(pix, sorted=True)

    levels = nside2order(nside_upgrade) - nside2order(nside)
    pix_nest = pix if nest else ring2nest(nside, pix)
    child_nest = nest_children(pix_nest, levels=levels).reshape(-1)
    child = child_nest if nest else nest2ring(nside_upgrade, child_nest)
    return torch.unique(child, sorted=True)


def degrade_pixels(
    nside: int,
    pixels: Tensor | Sequence[int],
    nside_degrade: int,
    *,
    nest: bool = False,
) -> Tensor:
    """
    Degrade explicit HEALPix pixel ids to a coarser resolution.

    Every parent pixel touched by at least one input child is returned. Duplicates are
    removed and output is sorted in the requested output ordering.
    """
    _validate_nside(nside)
    _validate_nside(nside_degrade)
    if nside_degrade > nside:
        raise ValueError("The value for nside_degrade must be <= nside.")

    pix = _as_int64(pixels).reshape(-1)
    if pix.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=pix.device)
    npix = nside2npix(nside)
    if bool(((pix < 0) | (pix >= npix)).any()):
        raise ValueError("pixels contain values outside the input nside range")
    if nside_degrade == nside:
        return torch.unique(pix, sorted=True)

    levels = nside2order(nside) - nside2order(nside_degrade)
    pix_nest = pix if nest else ring2nest(nside, pix)
    parent_nest = nest_parent(pix_nest, levels=levels)
    parent = parent_nest if nest else nest2ring(nside_degrade, parent_nest)
    return torch.unique(parent, sorted=True)


def query_box(
    nside: int,
    lon_min: float,
    lat_min: float,
    lon_max: float,
    lat_max: float,
    *,
    nest: bool = False,
    inclusive: bool = False,
    chunk_size: int = 1_000_000,
) -> Tensor:
    """
    Return pixels whose centers lie in lon/lat box (degrees).
    Handles longitude wrap-around.

    Uses a **latitude (colatitude) ring preselect** via `query_strip`, then applies
    the longitude filter only on that band — typically far cheaper than scanning all
    ``npix`` pixels when the declination range is narrow.
    """
    _validate_nside(nside)
    lon_min = float(lon_min)
    lon_max = float(lon_max)
    lat_lo = min(float(lat_min), float(lat_max))
    lat_hi = max(float(lat_min), float(lat_max))

    lon_span = lon_max - lon_min if lon_min <= lon_max else (lon_max - lon_min) % 360.0

    if inclusive:
        pad_deg = max_pixel_radius(nside, degrees=True)
        lat_lo -= pad_deg
        lat_hi += pad_deg
        max_abs_lat = max(abs(lat_lo), abs(lat_hi))
        if max_abs_lat >= 90.0:
            lon_span = 360.0
        else:
            lon_pad_deg = pad_deg / max(math.cos(math.radians(max_abs_lat)), 1e-6)
            lon_min -= lon_pad_deg
            lon_max += lon_pad_deg
            lon_span += 2.0 * lon_pad_deg
    lat_lo = max(-90.0, lat_lo)
    lat_hi = min(90.0, lat_hi)

    full_lon = lon_span >= 360.0 - 1e-12
    lon_min = lon_min % 360.0
    lon_max = lon_max % 360.0
    wraps = lon_min > lon_max

    theta1 = math.radians(90.0 - lat_hi)
    theta2 = math.radians(90.0 - lat_lo)
    cand = query_strip(
        nside,
        theta1,
        theta2,
        nest=nest,
        inclusive=False,
    )
    if cand.numel() == 0:
        return cand

    parts: list[Tensor] = []
    n_c = int(cand.numel())
    for start in range(0, n_c, chunk_size):
        pix = cand[start : start + chunk_size]
        lon, lat = pix2ang(nside, pix, nest=nest, lonlat=True)
        lat_keep = (lat >= lat_lo) & (lat <= lat_hi)
        if full_lon:
            keep = lat_keep
        elif wraps:
            keep = lat_keep & ((lon >= lon_min) | (lon <= lon_max))
        else:
            keep = lat_keep & (lon >= lon_min) & (lon <= lon_max)
        if torch.any(keep):
            parts.append(pix[keep])
    if not parts:
        return torch.empty((0,), dtype=torch.int64)
    return torch.cat(parts, dim=0)


def reorder(m: Tensor, *, r2n: bool = False, n2r: bool = False) -> Tensor:
    """Reorder HEALPix maps between RING and NESTED layouts along the last axis."""
    if r2n == n2r:
        raise ValueError("exactly one of r2n/n2r must be True")
    t = torch.as_tensor(m)
    if t.ndim == 0:
        raise ValueError("map must have at least 1 dimension")
    npix = int(t.shape[-1])
    nside = npix2nside(npix)
    idx = torch.arange(npix, dtype=torch.int64, device=t.device)
    perm = ring2nest(nside, idx) if r2n else nest2ring(nside, idx)
    out = torch.empty_like(t)
    out.index_copy_(-1, perm, t)
    return out


def get_nside(m: Tensor | Sequence[float] | Sequence[Sequence[float]]) -> int:
    """Return NSIDE from a map (shape `(npix,)` or `(nmaps, npix)`)."""
    t = torch.as_tensor(m)
    if t.ndim == 1:
        npix = int(t.shape[0])
    elif t.ndim == 2:
        npix = int(t.shape[1])
    else:
        raise ValueError("map must be 1D or 2D")
    return npix2nside(npix)


def get_map_size(m: Tensor | Sequence[float] | Sequence[Sequence[float]]) -> int:
    """Return map size (`npix`) for a map input."""
    return nside2npix(get_nside(m))


def mask_bad(
    m: Tensor | float,
    badval: float = UNSEEN,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-8,
) -> Tensor:
    """Boolean mask for bad (UNSEEN/non-finite) map values."""
    t = torch.as_tensor(m)
    if not t.is_floating_point():
        try:
            bad = torch.tensor(badval, dtype=t.dtype, device=t.device)
            return t == bad
        except (ValueError, TypeError, RuntimeError):
            return torch.zeros_like(t, dtype=torch.bool)
    bad = torch.tensor(float(badval), dtype=t.dtype, device=t.device)
    return (~torch.isfinite(t)) | torch.isclose(t, bad, rtol=rtol, atol=atol)


def _ud_grade_core_nested(
    m_nest: Tensor,
    nside_out: int,
    *,
    pess: bool = False,
    power: float | None = None,
    badval: float = UNSEEN,
) -> Tensor:
    nside_in = npix2nside(int(m_nest.numel()))
    npix_in = int(m_nest.numel())
    npix_out = nside2npix(nside_out)

    if power is not None:
        ratio = (float(nside_out) / float(nside_in)) ** float(power)
    else:
        ratio = 1.0

    if nside_out > nside_in:
        if (nside_out % nside_in) != 0:
            raise ValueError(
                "nside_out must be an integer multiple of nside_in for upgrade"
            )
        rat2 = npix_out // npix_in
        out = m_nest.reshape(npix_in, 1).expand(npix_in, rat2).reshape(npix_out)
        return out * ratio

    if nside_out < nside_in:
        if (nside_in % nside_out) != 0:
            raise ValueError(
                "nside_in must be an integer multiple of nside_out for degrade"
            )
        rat2 = npix_in // npix_out
        mr = m_nest.reshape(npix_out, rat2)
        goods = (~mask_bad(mr, badval=badval)) & torch.isfinite(mr)
        # Match healpy semantics: use multiplication by 0/1 mask instead of
        # `where`, so masked NaN/Inf may still propagate as NaN via `x * 0`.
        summed = torch.sum(mr * goods.to(dtype=mr.dtype), dim=1)
        nhit = goods.sum(dim=1)
        badout = (nhit != rat2) if pess else (nhit == 0)

        nhit_f = nhit.to(dtype=summed.dtype)
        if power is not None:
            nhit_f = nhit_f / ratio
        out = torch.zeros_like(summed)
        nz = nhit_f != 0
        out[nz] = summed[nz] / nhit_f[nz]
        if out.is_floating_point():
            out = out.clone()
            out[badout] = float(badval)
        return out

    return m_nest


def ud_grade(
    map_in: Tensor | Sequence[float] | Sequence[Sequence[float]],
    nside_out: int,
    pess: bool = False,
    badval: float = UNSEEN,
    order_in: str = "RING",
    order_out: str | None = None,
    power: float | None = None,
    dtype: object | None = None,
) -> Tensor:
    """
    Upgrade or degrade resolution of HEALPix map(s), healpy-compatible shape semantics.

    Supports map input shapes `(npix,)` and `(nmaps, npix)`.
    """
    _validate_nside(nside_out)
    in_order = _normalize_order(order_in)
    out_order = in_order if order_out is None else _normalize_order(order_out)

    t = torch.as_tensor(map_in)
    single = t.ndim == 1
    if single:
        maps = t.unsqueeze(0)
    elif t.ndim == 2:
        maps = t
    else:
        raise ValueError("map_in must be 1D or 2D")

    out_dtype = _resolve_dtype(dtype, maps.dtype)
    out_maps: list[Tensor] = []
    for i in range(maps.shape[0]):
        m = maps[i]
        npix = int(m.numel())
        _ = npix2nside(npix)  # validate map length
        m_work = reorder(m, r2n=True) if in_order == "RING" else m
        mout = _ud_grade_core_nested(
            m_work, nside_out, pess=pess, power=power, badval=badval
        )
        if out_order == "RING":
            mout = reorder(mout, n2r=True)
        out_maps.append(mout.to(dtype=out_dtype))

    stacked = torch.stack(out_maps, dim=0)
    return stacked[0] if single else stacked


def lonlat_to_xyz(ra: Tensor, dec: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Convert spherical lon/lat in degrees to unit Cartesian coordinates."""
    ra_t = _as_float64(ra)
    dec_t = _as_float64(dec)
    ra_t, dec_t = torch.broadcast_tensors(ra_t, dec_t)
    _validate_lonlat(ra_t, dec_t)

    lon = torch.deg2rad(ra_t)
    lat = torch.deg2rad(dec_t)
    sin_lat = torch.sin(lat)
    # |lat| <= 90 deg, so cos(lat) >= 0; float32 deg2rad(90) exceeds pi/2 and
    # would flip a polar vector to the opposite longitude.
    cos_lat = torch.cos(lat).clamp_min(0.0)
    sin_lon = torch.sin(lon)
    cos_lon = torch.cos(lon)

    x = cos_lat * cos_lon
    y = cos_lat * sin_lon
    z = sin_lat
    return x, y, z


def xyz_to_lonlat(
    x: Tensor, y: Tensor, z: Tensor, eps: float = 1.0e-15
) -> Tuple[Tensor, Tensor]:
    """Convert Cartesian coordinates to lon/lat in degrees."""
    x_t = _as_float64(x)
    y_t = _as_float64(y)
    z_t = _as_float64(z)
    x_t, y_t, z_t = torch.broadcast_tensors(x_t, y_t, z_t)
    if _host_checks_enabled() and bool(
        torch.any((x_t == 0.0) & (y_t == 0.0) & (z_t == 0.0))
    ):
        raise ValueError("Cartesian vectors must have positive norm")

    n = torch.sqrt(x_t * x_t + y_t * y_t + z_t * z_t).clamp_min(eps)
    x_n = x_t / n
    y_n = y_t / n
    z_n = z_t / n

    # atan2(z, rho) rather than asin(z): asin loses half of the digits next to
    # the poles (|z| -> 1) and has an infinite derivative there. On the polar
    # axis (rho = 0) the sqrt argument is replaced so the gradient is 0, not NaN.
    rho2 = x_n * x_n + y_n * y_n
    on_axis = rho2 == 0.0
    rho = torch.where(on_axis, 0.0, torch.sqrt(torch.where(on_axis, 1.0, rho2)))
    x_safe = torch.where(on_axis, 1.0, x_n)
    lon = torch.where(on_axis, 0.0, torch.atan2(y_n, x_safe))
    ra = torch.remainder(torch.rad2deg(lon), 360.0)
    dec = torch.rad2deg(torch.atan2(z_n, rho))
    return ra, dec


def angular_distance_deg(
    ra1: Tensor, dec1: Tensor, ra2: Tensor, dec2: Tensor
) -> Tensor:
    """Great-circle angular distance between two lon/lat points in degrees."""
    ra1_t = _as_float64(ra1)
    dec1_t = _as_float64(dec1)
    ra2_t = _as_float64(ra2)
    dec2_t = _as_float64(dec2)
    ra1_t, dec1_t, ra2_t, dec2_t = torch.broadcast_tensors(ra1_t, dec1_t, ra2_t, dec2_t)
    _validate_lonlat(ra1_t, dec1_t)
    _validate_lonlat(ra2_t, dec2_t)

    r1 = torch.deg2rad(ra1_t)
    d1 = torch.deg2rad(dec1_t)
    r2 = torch.deg2rad(ra2_t)
    d2 = torch.deg2rad(dec2_t)

    dr = r1 - r2
    dd = d1 - d2
    a = torch.square(torch.sin(dd * 0.5)) + torch.cos(d1) * torch.cos(d2) * (
        torch.square(torch.sin(dr * 0.5))
    )
    return torch.rad2deg(2.0 * torch.asin(torch.sqrt(torch.clamp(a, 0.0, 1.0))))


def spherical_fourier_features(
    ra: Tensor,
    dec: Tensor,
    num_frequencies: int = 6,
    min_frequency: float = 1.0,
    base: float = 2.0,
    include_xyz: bool = True,
    include_lonlat: bool = True,
) -> Tensor:
    """
    Build multi-scale periodic features for spherical positional encoders.

    Output shape is `ra.shape + (D,)`, where:
    - xyz block contributes 3 channels if `include_xyz=True`
    - each frequency contributes 12 channels for xyz sin/cos (if enabled)
    - each frequency contributes 4 channels for lon/lat sin/cos (if enabled)
    """
    if num_frequencies <= 0:
        raise ValueError("num_frequencies must be positive")
    if min_frequency <= 0.0:
        raise ValueError("min_frequency must be positive")
    if base <= 0.0:
        raise ValueError("base must be positive")

    ra_t = _as_float64(ra)
    dec_t = _as_float64(dec)
    ra_t, dec_t = torch.broadcast_tensors(ra_t, dec_t)

    lon = torch.deg2rad(ra_t)
    lat = torch.deg2rad(dec_t)
    x, y, z = lonlat_to_xyz(ra_t, dec_t)

    if not (include_xyz or include_lonlat):
        raise ValueError("at least one of include_xyz/include_lonlat must be True")
    exponents = torch.arange(num_frequencies, dtype=ra_t.dtype, device=ra_t.device)
    freqs = min_frequency * (base**exponents)

    # One broadcast sin/cos over (frequency, coordinate) instead of ten small
    # kernels per frequency. Channel order per frequency: sin, cos of x, y, z
    # (include_xyz), then of lon, lat (include_lonlat).
    coords = ([x, y, z] if include_xyz else []) + ([lon, lat] if include_lonlat else [])
    scaled = torch.stack(coords, dim=-1).unsqueeze(-2) * freqs.unsqueeze(-1)
    periodic = torch.stack([torch.sin(scaled), torch.cos(scaled)], dim=-1).flatten(
        start_dim=-3
    )
    if include_xyz:
        return torch.cat([torch.stack([x, y, z], dim=-1), periodic], dim=-1)
    return periodic


def nest_parent(pix_nest: Tensor, levels: int = 1) -> Tensor:
    """Return NESTED parent index after reducing resolution by `levels`."""
    if levels <= 0:
        raise ValueError("levels must be positive")
    pix_nest_t = _as_int64(pix_nest)
    return pix_nest_t >> (2 * levels)


def nest_children(pix_nest: Tensor, levels: int = 1) -> Tensor:
    """Return all NESTED children after increasing resolution by `levels`."""
    if levels <= 0:
        raise ValueError("levels must be positive")
    pix_nest_t = _as_int64(pix_nest)
    n_children = 1 << (2 * levels)
    offsets = torch.arange(n_children, dtype=torch.int64, device=pix_nest_t.device)
    return (pix_nest_t << (2 * levels)).unsqueeze(-1) + offsets


__all__ = [
    "UNSEEN",
    "ang2pix",
    "ang2pix_nested",
    "ang2pix_ring",
    "ang2vec",
    "angular_distance_deg",
    "boundaries",
    "compact_bits",
    "degrade_pixels",
    "degrade_pixel_ranges",
    "get_all_neighbours",
    "get_interp_val",
    "get_interp_weights",
    "get_map_size",
    "get_nside",
    "isnpixok",
    "isnsideok",
    "lonlat_to_xyz",
    "mask_bad",
    "max_pixel_radius",
    "neighbors",
    "nest2ring",
    "nest_children",
    "nest_parent",
    "npix2nside",
    "nside2npix",
    "nside2order",
    "nside2pixarea",
    "nside2resol",
    "order2nside",
    "pixel_ranges_to_pixels",
    "pixel_resolution_to_nside",
    "pixels_to_pixel_ranges",
    "pix2ang",
    "pix2ang_nested",
    "pix2ang_ring",
    "pix2vec",
    "query_box",
    "query_circle",
    "query_circle_vec",
    "query_disc",
    "query_polygon",
    "query_polygon_vec",
    "query_strip",
    "reorder",
    "ring2nest",
    "spread_bits",
    "ud_grade",
    "upgrade_pixels",
    "upgrade_pixel_ranges",
    "vec2ang",
    "vec2pix",
    "xyz_to_lonlat",
]
