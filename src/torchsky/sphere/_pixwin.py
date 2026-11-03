"""HEALPix pixel window functions computed from the pixel shapes.

The temperature window of a pixelization is the rms over pixels of each
pixel's harmonic response (the definition behind the HEALPix tables):

    w_l^2 = (1/N) sum_p (1/Omega^2) iint_{p x p} P_l(n . n') dOmega dOmega'.

With ``t = |n - n'|^2 / 4 = (1 - n . n') / 2``, the Legendre polynomial is the
terminating hypergeometric series ``P_l(1 - 2t) = 2F1(-l, l+1; 1; t)``, so

    w_l^2 = sum_k a_k(l) mu_k,   mu_k = < t^k >_{pixels, pairs},

and the multipole dependence separates from the geometry. The polarization
window uses the spin-2 analogue ``d^l_{22}(beta) = (1-t)^2 2F1(2-l, l+3; 1; t)``
with the polarization frame parallel-transported across the pixel.

The moments are pair integrals over each pixel, evaluated with Gauss-Legendre
nodes in the pixel's face coordinates (HEALPix is equal-area, so the face
measure is the solid-angle measure). Pixels that touch the pole or straddle
the polar-cap boundary ``|z| = 2/3`` are split into two triangles with
collapsed (Duffy) coordinates, which removes the pole singularity and the
cap/equator kink. Symmetries reduce the work to one face octant of the polar
cap (about ``nside^2/4`` pixel shapes) plus one pixel per equatorial ring.

The series is truncated with a rigorous bound: ``|a_k(l)| <= (l(l+1))^k/(k!)^2``.
"""

from __future__ import annotations

import math
import os
import threading
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

#: Gauss-Legendre order per pixel axis. The pair integrand oscillates by at most
#: a few radians across a pixel for ``l <= 4 nside``; an 8-point rule agrees
#: with a 12-point one to 5e-13 there (6 points: 3e-9, 7 points: 3e-11).
_GL_ORDER = 8
#: Series length; covers ``l <= 4 nside`` with the tail bound below 1e-18.
_SERIES_TERMS = 28
#: Largest multipole relative to ``nside`` (the extent of the HEALPix tables).
LMAX_PER_NSIDE = 4
#: Largest supported nside (the extent of the HEALPix tables); the moments cost
#: ``O(nside^2)`` and take minutes beyond this.
MAX_NSIDE = 8192
_CACHE_VERSION = 1
_CHUNK_PAIRS = 1 << 18

_JRLL = (2, 3)
_JPLL = (1, 0)

_memory_cache: dict[int, tuple[np.ndarray, float]] = {}
_lock = threading.Lock()


def _gauss_legendre(order: int) -> tuple[Tensor, Tensor]:
    x, w = np.polynomial.legendre.leggauss(order)
    return (
        torch.as_tensor(0.5 * (x + 1.0), dtype=torch.float64),
        torch.as_tensor(0.5 * w, dtype=torch.float64),
    )


def _square_rule(order: int) -> tuple[Tensor, Tensor, Tensor]:
    u, w = _gauss_legendre(order)
    px = u[:, None].expand(order, order).reshape(-1)
    py = u[None, :].expand(order, order).reshape(-1)
    return px, py, (w[:, None] * w[None, :]).reshape(-1)


def _two_triangle_rule(order: int) -> tuple[Tensor, Tensor, Tensor]:
    """Unit square split on its anti-diagonal, each half in collapsed coordinates.

    The upper triangle collapses at (1, 1), the lower at (0, 0): the corners
    where a pixel can touch a pole. The anti-diagonal is the cap boundary of a
    straddling pixel.
    """

    u, w = _gauss_legendre(order)
    uu = u[:, None].expand(order, order).reshape(-1)
    vv = u[None, :].expand(order, order).reshape(-1)
    ww = (w[:, None] * w[None, :]).reshape(-1) * uu
    px = torch.cat((1.0 - uu * vv, uu * (1.0 - vv)))
    py = torch.cat((1.0 - uu + uu * vv, uu * vv))
    return px, py, torch.cat((ww, ww))


def _face_vectors(face_row: int, x: Tensor, y: Tensor) -> Tensor:
    """Unit vectors for continuous face coordinates.

    ``face_row`` 0 is a north face (face 0), 1 an equatorial face (face 4).
    Follows the continuous HEALPix projection (``xyf2loc`` in HEALPix C++).
    """

    jr = _JRLL[face_row] - x - y
    cap = jr < 1.0
    nr = torch.where(cap, jr, torch.ones_like(jr))
    z = torch.where(cap, 1.0 - nr * nr / 3.0, (2.0 - jr) * (2.0 / 3.0))
    sin2 = torch.where(
        cap, (nr * nr / 3.0) * (2.0 - nr * nr / 3.0), (1.0 - z) * (1.0 + z)
    )
    sin_theta = torch.sqrt(sin2)
    phi = (math.pi / 4.0) * (_JPLL[face_row] * nr + x - y) / nr
    return torch.stack(
        (sin_theta * torch.cos(phi), sin_theta * torch.sin(phi), z), dim=-1
    )


def _pixel_classes(nside: int) -> list[tuple[int, Tensor, Tensor, Tensor, bool]]:
    """Representative pixels with multiplicities covering all ``12 nside^2``.

    Polar-cap pixels of face 0 with ``ix >= iy`` (mirror symmetry), times four
    faces and two hemispheres; one pixel per equatorial ring, whose pixels are
    congruent under rotation about the polar axis.
    """

    ix, iy = torch.meshgrid(torch.arange(nside), torch.arange(nside), indexing="ij")
    ix, iy = ix.reshape(-1), iy.reshape(-1)
    keep = (ix + iy >= nside - 1) & (ix >= iy)
    ix, iy = ix[keep], iy[keep]
    weight = 8.0 * torch.where(ix == iy, 1.0, 2.0).to(torch.float64)
    special = (ix + iy == nside - 1) | ((ix == nside - 1) & (iy == nside - 1))
    rings = torch.arange(nside + 1, 2 * nside + 1)
    diagonal = 3 * nside - rings - 1
    ring_ix = diagonal.clamp(max=nside - 1)
    ring_iy = diagonal - ring_ix
    ring_weight = (
        4.0 * nside * torch.where(rings < 2 * nside, 2.0, 1.0).to(torch.float64)
    )
    return [
        (0, ix[~special], iy[~special], weight[~special], False),
        (0, ix[special], iy[special], weight[special], True),
        (1, ring_ix, ring_iy, ring_weight, False),
    ]


def _class_moments(
    nside: int,
    face_row: int,
    ix: Tensor,
    iy: Tensor,
    rule: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
    terms: int,
) -> tuple[Tensor, float]:
    """Pixel-scaled pair moments ``<(nside^2 t)^k>`` for ``k = 0..terms``.

    Only the off-diagonal pairs ``i < j`` are formed (``t`` is symmetric and
    vanishes on the diagonal, which therefore adds to ``k = 0`` only).
    """

    px, py, w, first, second = rule
    x = (ix[:, None].to(torch.float64) + px[None]) / nside
    y = (iy[:, None].to(torch.float64) + py[None]) / nside
    vectors = _face_vectors(face_row, x, y)
    diff = vectors[:, first, :] - vectors[:, second, :]
    # Scaling by nside^2 keeps every power in floating-point range for any nside.
    t = (0.25 * float(nside) ** 2) * diff.square().sum(-1)
    power = (2.0 * w[first] * w[second]).expand_as(t)
    moments = []
    for _ in range(terms + 1):
        moments.append(power.sum(-1))
        power = power * t
    moments[0] = moments[0] + w.square().sum()
    return torch.stack(moments, -1), float(t.amax())


def _with_pairs(
    rule: tuple[Tensor, Tensor, Tensor],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    first, second = torch.triu_indices(rule[0].numel(), rule[0].numel(), offset=1)
    return (*rule, first, second)


def _compute_moments(nside: int) -> tuple[np.ndarray, float]:
    terms = _SERIES_TERMS + 2
    square = _with_pairs(_square_rule(_GL_ORDER))
    triangles = _with_pairs(_two_triangle_rule(_GL_ORDER))
    total = torch.zeros(terms + 1, dtype=torch.float64)
    total_weight = 0.0
    t_max = 0.0
    for face_row, ix, iy, weight, special in _pixel_classes(nside):
        rule = triangles if special else square
        chunk = max(1, _CHUNK_PAIRS // rule[3].numel())
        for start in range(0, ix.numel(), chunk):
            stop = start + chunk
            moments, local_max = _class_moments(
                nside, face_row, ix[start:stop], iy[start:stop], rule, terms
            )
            total += (weight[start:stop, None] * moments).sum(0)
            total_weight += float(weight[start:stop].sum())
            t_max = max(t_max, local_max)
    npix = 12 * nside * nside
    if abs(total_weight - npix) > 1e-6 * npix:
        raise RuntimeError("pixel classes do not tile the sphere")
    return (total / npix).numpy(), t_max


def _cache_dir() -> Path:
    override = os.environ.get("TORCHSKY_CACHE_DIR")
    if override:
        base = Path(override).expanduser()
    else:
        xdg = os.environ.get("XDG_CACHE_HOME")
        base = (Path(xdg).expanduser() if xdg else Path.home() / ".cache") / "torchsky"
    return base / "pixwin"


#: Below this nside the moments take well under a second; only larger ones are
#: written to disk.
_DISK_CACHE_MIN_NSIDE = 256


def _load_cached(path: Path) -> tuple[np.ndarray, float] | None:
    try:
        with np.load(path, allow_pickle=False) as data:
            moments = np.asarray(data["moments"], dtype=np.float64)
            t_max = float(data["t_max"])
    except (OSError, ValueError, KeyError):
        return None
    if moments.shape != (_SERIES_TERMS + 3,):
        return None
    if not (np.isfinite(moments).all() and math.isfinite(t_max)):
        return None
    return moments, t_max


def _store_cached(path: Path, values: tuple[np.ndarray, float]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp.npz")
        np.savez(tmp, moments=values[0], t_max=np.float64(values[1]))
        tmp.replace(path)
    except OSError:
        # A read-only cache directory only costs a recomputation next time.
        return


def pixel_moments(nside: int) -> tuple[np.ndarray, float]:
    """Pixel-averaged scaled pair moments for ``nside`` (memory and disk cached)."""

    with _lock:
        cached = _memory_cache.get(nside)
    if cached is not None:
        return cached
    path = _cache_dir() / f"pixwin_n{nside}_v{_CACHE_VERSION}_g{_GL_ORDER}.npz"
    values = None
    if nside >= _DISK_CACHE_MIN_NSIDE:
        values = _load_cached(path)
    if values is None:
        values = _compute_moments(nside)
        if nside >= _DISK_CACHE_MIN_NSIDE:
            _store_cached(path, values)
    with _lock:
        _memory_cache[nside] = values
    return values


def _series(ell: Tensor, moments: np.ndarray, nside: int, shift: int) -> Tensor:
    """``sum_k c_k(l) <t^k>`` for ``2F1(shift - l, l + 1 + shift; 1; t)``."""

    scale = 1.0 / float(nside) ** 2
    coefficient = torch.ones_like(ell)
    total = coefficient * float(moments[0])
    for k in range(1, moments.shape[0]):
        j = float(k - 1)
        coefficient = (
            coefficient
            * (j + shift - ell)
            * (j + ell + 1.0 + shift)
            * (scale / (k * k))
        )
        total = total + coefficient * float(moments[k])
    return total


def computed_pixwin(nside: int, lmax: int) -> tuple[Tensor, Tensor]:
    """Temperature and polarization windows for ``0 <= l <= lmax``."""

    if lmax > LMAX_PER_NSIDE * nside:
        raise ValueError(
            f"lmax={lmax} exceeds {LMAX_PER_NSIDE}*nside={LMAX_PER_NSIDE * nside}; "
            "the pixel window is not defined meaningfully beyond the pixel scale"
        )
    mu, t_max = pixel_moments(nside)
    # (1 - t)^2 t^k in pixel-scaled units: a fixed combination of the moments.
    inv = 1.0 / float(nside) ** 2
    nu = mu[:-2] - 2.0 * inv * mu[1:-1] + inv * inv * mu[2:]
    mu = mu[:-2]
    # Rigorous tail bound of the truncated series at the largest multipole.
    x = float(lmax) * float(lmax + 1) * t_max / float(nside) ** 2
    terms = mu.shape[0] - 1
    tail = x ** (terms + 1) / math.factorial(terms + 1) ** 2
    if tail > 1e-15:
        raise RuntimeError("pixel-window series bound exceeded; increase the terms")
    ell = torch.arange(lmax + 1, dtype=torch.float64)
    temperature = torch.sqrt(_series(ell, mu, nside, 0).clamp_min(0.0))
    spin2 = _series(ell, nu, nside, 2)
    polarization = torch.where(
        ell >= 2, torch.sqrt(spin2.clamp_min(0.0)), torch.zeros_like(ell)
    )
    return temperature, polarization


__all__ = ["LMAX_PER_NSIDE", "MAX_NSIDE", "computed_pixwin", "pixel_moments"]
