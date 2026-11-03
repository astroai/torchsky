"""Alm indexing helpers, device checks, row reshaping, and ring/nest permutations for HEALPix spectral transforms."""

import math

import torch
from torch import Tensor

from .._dtype import work_complex_dtype as _preferred_complex_dtype
from .._dtype import work_dtype as _preferred_real_dtype

from . import healpix as _healpix

from ._spectral_config import (
    _ALM_ELL_CACHE,
    _ALM_M_CACHE,
    _ANGLE_CACHE,
    _NEST2RING_PERM_CACHE,
    _RING2NEST_PERM_CACHE,
)
from ..device import _to


def _infer_lmax_mmax_from_nalm(
    nalm: int, lmax: int | None, mmax: int | None
) -> tuple[int, int]:
    if nalm <= 0:
        raise ValueError("nalm must be positive")
    if lmax is None and mmax is None:
        disc = 1 + 8 * nalm
        root = int(math.isqrt(disc))
        if root * root != disc:
            raise ValueError("cannot infer lmax from nalm")
        ell = (root - 3) // 2
        if (ell + 1) * (ell + 2) // 2 != nalm:
            raise ValueError("cannot infer lmax from nalm")
        return int(ell), int(ell)
    if lmax is None:
        raise ValueError("lmax must be provided when mmax is provided")
    mm = int(lmax if mmax is None else mmax)
    if _core_mod.alm_size(int(lmax), mm) != nalm:
        raise ValueError("nalm does not match provided lmax/mmax")
    return int(lmax), mm


def _infer_lmax_for_fixed_mmax(nalm: int, mmax: int) -> int:
    if nalm <= 0:
        raise ValueError("nalm must be positive")
    mm = int(mmax)
    if mm < 0:
        raise ValueError("mmax must be non-negative")
    num = nalm + (mm * (mm - 1)) // 2
    den = mm + 1
    if num % den != 0:
        raise ValueError("cannot infer lmax from nalm/mmax")
    ll = (num // den) - 1
    if ll < mm or _core_mod.alm_size(ll, mm) != nalm:
        raise ValueError("cannot infer lmax from nalm/mmax")
    return int(ll)


def _alm_ell_array(lmax: int, mmax: int) -> Tensor:
    key = (int(lmax), int(mmax))

    def build() -> Tensor:
        vals = []
        for m in range(mmax + 1):
            vals.extend(range(m, lmax + 1))
        return torch.tensor(vals, dtype=torch.int64)

    return _ALM_ELL_CACHE.get_or_build(key, build)


def _alm_m_array(lmax: int, mmax: int) -> Tensor:
    key = (int(lmax), int(mmax))

    def build() -> Tensor:
        vals = []
        for m in range(mmax + 1):
            vals.extend([m] * (lmax - m + 1))
        return torch.tensor(vals, dtype=torch.int64)

    return _ALM_M_CACHE.get_or_build(key, build)


def _require_sht_device(*values: object, what: str = "HEALPix SHT") -> None:
    """Fused Ishioka C++ is CPU. CUDA/MPS use the ATen ring path; no host copy."""
    allowed = {"cpu", "cuda", "mps"}
    for value in values:
        if value is None:
            continue
        tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
        if tensor.device.type not in allowed:
            raise ValueError(
                f"{what} supports cpu, cuda, and mps; got {tensor.device}. "
                "Fused Ishioka/Wigner C++ stays on CPU. CUDA/MPS keep tensors "
                "on device (ATen ring FFT). Use alm2lonlat for arbitrary coords."
            )


def _map_to_rows(
    map_values: Tensor | list[float] | list[list[float]],
) -> tuple[Tensor, bool]:
    t = torch.as_tensor(map_values)
    _require_sht_device(t, what="HEALPix SHT")
    single = t.ndim == 1
    if single:
        rows = t.unsqueeze(0)
    elif t.ndim == 2:
        rows = t
    else:
        raise ValueError("map_values must be shape (npix,) or (nmaps, npix)")
    return rows.to(dtype=_preferred_real_dtype(rows.device)), single


def _alm_to_rows(
    alm_values: Tensor | list[complex] | list[list[complex]],
) -> tuple[Tensor, bool]:
    t = torch.as_tensor(alm_values)
    _require_sht_device(t, what="HEALPix SHT")
    single = t.ndim == 1
    if single:
        rows = t.unsqueeze(0)
    elif t.ndim == 2:
        rows = t
    else:
        raise ValueError("alm_values must be shape (nalm,) or (nmaps, nalm)")
    return rows.to(dtype=_preferred_complex_dtype(rows.device)), single


def _angles_for_nside(nside: int, nest: bool) -> tuple[Tensor, Tensor, Tensor, float]:
    key = (int(nside), bool(nest))

    def build() -> tuple[Tensor, Tensor, Tensor, float]:
        npix = _healpix.nside2npix(nside)
        pix = torch.arange(npix, dtype=torch.int64)
        theta, phi = _healpix.pix2ang(nside, pix, nest=nest, lonlat=False)
        theta = _to(theta, dtype=torch.float64, device="cpu")
        phi = _to(phi, dtype=torch.float64, device="cpu")
        return theta, phi, torch.cos(theta), 4.0 * math.pi / float(npix)

    return _ANGLE_CACHE.get_or_build(key, build)


def _ring2nest_perm_for_nside(nside: int) -> Tensor:
    ns = int(nside)

    def build() -> Tensor:
        npix = _healpix.nside2npix(ns)
        idx = torch.arange(npix, dtype=torch.int64)
        return _to(
            _healpix.ring2nest(ns, idx), dtype=torch.int64, device="cpu"
        ).contiguous()

    return _RING2NEST_PERM_CACHE.get_or_build(ns, build)


def _nest2ring_perm_for_nside(nside: int) -> Tensor:
    ns = int(nside)

    def build() -> Tensor:
        npix = _healpix.nside2npix(ns)
        idx = torch.arange(npix, dtype=torch.int64)
        return _to(
            _healpix.nest2ring(ns, idx), dtype=torch.int64, device="cpu"
        ).contiguous()

    return _NEST2RING_PERM_CACHE.get_or_build(ns, build)


def _reorder_nest_to_ring_rows(rows: Tensor, nside: int) -> Tensor:
    perm = _ring2nest_perm_for_nside(int(nside))
    if rows.device.type != "cpu":
        perm = perm.to(device=rows.device)
    return torch.index_select(rows, rows.ndim - 1, perm)


def _reorder_ring_to_nest_rows(rows: Tensor, nside: int) -> Tensor:
    perm = _nest2ring_perm_for_nside(int(nside))
    if rows.device.type != "cpu":
        perm = perm.to(device=rows.device)
    return torch.index_select(rows, rows.ndim - 1, perm)


# Late-bound so monkeypatching ``spectral`` attributes still reaches this code.
from . import spectral as _core_mod  # noqa: E402
