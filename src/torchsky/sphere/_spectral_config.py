"""Environment-derived limits, feature flags, and cache namespaces for HEALPix spectral transforms."""

import os

import torch
from torch import Tensor

from .. import _ops as _native_ops

from ._spectral_cache import SpectralCacheManager, SpectralCacheNamespace


_HAS_HEALPIX_SPIN_MAP2ALM_RING_REC = _native_ops.has_spectral_dispatcher()


_HAS_HEALPIX_SPIN_ALM2MAP_RING_REC = _native_ops.has_spectral_dispatcher()


_HAS_HEALPIX_SPIN_INTERPOLATE_REC = _native_ops.has_spectral_dispatcher()


_SPIN_MAX_CACHE_BYTES = int(
    os.environ.get("TORCHSKY_SPIN_MAX_CACHE_BYTES", str(512 * 1024 * 1024))
)


_SCALAR_MAX_CACHE_BYTES = int(
    os.environ.get("TORCHSKY_SCALAR_MAX_CACHE_BYTES", str(512 * 1024 * 1024))
)


_SPECTRAL_MAX_CACHE_BYTES = int(
    os.environ.get("TORCHSKY_SPECTRAL_MAX_CACHE_BYTES", str(512 * 1024 * 1024))
)


_SPECTRAL_MAX_CACHE_ENTRY_BYTES = int(
    os.environ.get("TORCHSKY_SPECTRAL_MAX_CACHE_ENTRY_BYTES", str(128 * 1024 * 1024))
)


_SPECTRAL_CACHE = SpectralCacheManager(
    max_bytes=_SPECTRAL_MAX_CACHE_BYTES,
    max_entry_bytes=_SPECTRAL_MAX_CACHE_ENTRY_BYTES,
)


_SPIN_CACHE_ENTRY_LIMIT = min(
    _SPIN_MAX_CACHE_BYTES,
    _SPECTRAL_MAX_CACHE_ENTRY_BYTES,
    _SPECTRAL_MAX_CACHE_BYTES,
)


_SCALAR_CACHE_ENTRY_LIMIT = min(
    _SCALAR_MAX_CACHE_BYTES,
    _SPECTRAL_MAX_CACHE_ENTRY_BYTES,
    _SPECTRAL_MAX_CACHE_BYTES,
)


_ANGLE_CACHE: SpectralCacheNamespace[tuple[Tensor, Tensor, Tensor, float]] = (
    _SPECTRAL_CACHE.namespace(
        "angles", device="cpu", dtype=(torch.int64, torch.float64)
    )
)


_YLM_CACHE: SpectralCacheNamespace[tuple[tuple[Tensor, ...], tuple[Tensor, ...]]] = (
    _SPECTRAL_CACHE.namespace(
        "ylm",
        device="cpu",
        dtype=torch.complex128,
        max_entry_bytes=_SCALAR_CACHE_ENTRY_LIMIT,
    )
)


_ALM_ELL_CACHE: SpectralCacheNamespace[Tensor] = _SPECTRAL_CACHE.namespace(
    "alm_ell", device="cpu", dtype=torch.int64
)


_ALM_M_CACHE: SpectralCacheNamespace[Tensor] = _SPECTRAL_CACHE.namespace(
    "alm_m", device="cpu", dtype=torch.int64
)


_SPIN_YLM_CACHE: SpectralCacheNamespace[
    tuple[tuple[Tensor, ...], tuple[Tensor, ...]]
] = _SPECTRAL_CACHE.namespace(
    "spin_ylm",
    device="cpu",
    dtype=torch.complex128,
    max_entry_bytes=_SPIN_CACHE_ENTRY_LIMIT,
)


_SPIN_MAP2ALM_MAT_CACHE: SpectralCacheNamespace[tuple[Tensor, Tensor]] = (
    _SPECTRAL_CACHE.namespace(
        "spin_map2alm_mat",
        device="cpu",
        dtype=torch.complex128,
        max_entry_bytes=_SPIN_CACHE_ENTRY_LIMIT,
    )
)


_SPIN_ALM2MAP_MAT_CACHE: SpectralCacheNamespace[
    tuple[Tensor, Tensor, Tensor, Tensor]
] = _SPECTRAL_CACHE.namespace(
    "spin_alm2map_mat",
    device="cpu",
    dtype=torch.complex128,
    max_entry_bytes=_SPIN_CACHE_ENTRY_LIMIT,
)


_SCALAR_MAP2ALM_MAT_CACHE: SpectralCacheNamespace[Tensor] = _SPECTRAL_CACHE.namespace(
    "scalar_map2alm_mat",
    device="cpu",
    dtype=torch.complex128,
    max_entry_bytes=_SCALAR_CACHE_ENTRY_LIMIT,
)


_SCALAR_ALM2MAP_MAT_CACHE: SpectralCacheNamespace[Tensor] = _SPECTRAL_CACHE.namespace(
    "scalar_alm2map_mat",
    device="cpu",
    dtype=torch.complex128,
    max_entry_bytes=_SCALAR_CACHE_ENTRY_LIMIT,
)


_RING_LAYOUT_CACHE: SpectralCacheNamespace[tuple[Tensor, Tensor, Tensor, Tensor]] = (
    _SPECTRAL_CACHE.namespace(
        "ring_layout", device="cpu", dtype=(torch.int64, torch.float64)
    )
)


_RING_SCALAR_BASIS_CACHE: SpectralCacheNamespace[tuple[Tensor, ...]] = (
    _SPECTRAL_CACHE.namespace(
        "ring_scalar_basis",
        device="cpu",
        dtype=torch.complex128,
        max_entry_bytes=_SCALAR_CACHE_ENTRY_LIMIT,
    )
)


_RING_PHI0_PHASE_CACHE: SpectralCacheNamespace[Tensor] = _SPECTRAL_CACHE.namespace(
    "ring_phi0_phase", device="cpu", dtype=torch.complex128
)


_RING_AZIMUTH_PHASE_CACHE: SpectralCacheNamespace[Tensor] = _SPECTRAL_CACHE.namespace(
    "ring_azimuth_phase", device="cpu", dtype=torch.complex128
)


_RING_ALIAS_INDEX_CACHE: SpectralCacheNamespace[Tensor] = _SPECTRAL_CACHE.namespace(
    "ring_alias_index", device="cpu", dtype=torch.int64
)


_RING_SPIN_CONJ_INDEX_CACHE: SpectralCacheNamespace[tuple[Tensor, Tensor, Tensor]] = (
    _SPECTRAL_CACHE.namespace("ring_spin_conj_index", device="cpu", dtype=torch.int64)
)


_RING_PIX2RING_CACHE: SpectralCacheNamespace[Tensor] = _SPECTRAL_CACHE.namespace(
    "ring_pix2ring", device="cpu", dtype=torch.int64
)


_RING2NEST_PERM_CACHE: SpectralCacheNamespace[Tensor] = _SPECTRAL_CACHE.namespace(
    "ring2nest_perm", device="cpu", dtype=torch.int64
)


_NEST2RING_PERM_CACHE: SpectralCacheNamespace[Tensor] = _SPECTRAL_CACHE.namespace(
    "nest2ring_perm", device="cpu", dtype=torch.int64
)


_RING_SPIN_BASIS_CACHE: SpectralCacheNamespace[tuple[Tensor, ...]] = (
    _SPECTRAL_CACHE.namespace(
        "ring_spin_basis",
        device="cpu",
        dtype=torch.complex128,
        max_entry_bytes=_SPIN_CACHE_ENTRY_LIMIT,
    )
)


_RING_SCALAR_BASIS_CONCAT_CACHE: SpectralCacheNamespace[Tensor] = (
    _SPECTRAL_CACHE.namespace(
        "ring_scalar_basis_concat",
        device="cpu",
        dtype=torch.complex128,
        max_entry_bytes=_SCALAR_CACHE_ENTRY_LIMIT,
    )
)


_RING_GROUP_CACHE: SpectralCacheNamespace[
    tuple[tuple[int, int, int, int, int] | None, tuple[tuple[int, Tensor, Tensor], ...]]
] = _SPECTRAL_CACHE.namespace("ring_group", device="cpu", dtype=torch.int64)


_SCALAR_RING_AUTO_MIN_BYTES = int(
    os.environ.get("TORCHSKY_SCALAR_RING_AUTO_MIN_BYTES", "0")
)


_SPIN_RING_AUTO_MIN_BYTES = int(
    os.environ.get("TORCHSKY_SPIN_RING_AUTO_MIN_BYTES", "0")
)


_SPIN_MAP2ALM_RING_AUTO_MIN_BYTES = int(
    os.environ.get(
        "TORCHSKY_SPIN_MAP2ALM_RING_AUTO_MIN_BYTES", str(_SPIN_RING_AUTO_MIN_BYTES)
    )
)


_SPIN_ALM2MAP_RING_AUTO_MIN_BYTES = int(
    os.environ.get(
        "TORCHSKY_SPIN_ALM2MAP_RING_AUTO_MIN_BYTES", str(_SPIN_RING_AUTO_MIN_BYTES)
    )
)


_SPIN_RING_CONCAT_FAST_MAX_BYTES = int(
    os.environ.get("TORCHSKY_SPIN_RING_CONCAT_FAST_MAX_BYTES", str(256 * 1024 * 1024))
)


_SPIN_RING_INTERP_CONCAT_FAST_MAX_BYTES = int(
    os.environ.get(
        "TORCHSKY_SPIN_RING_INTERP_CONCAT_FAST_MAX_BYTES",
        str(_SPIN_RING_CONCAT_FAST_MAX_BYTES),
    )
)


_SPIN_RING_INTEGRATE_CONCAT_FAST_MAX_BYTES = int(
    os.environ.get(
        "TORCHSKY_SPIN_RING_INTEGRATE_CONCAT_FAST_MAX_BYTES",
        str(_SPIN_RING_CONCAT_FAST_MAX_BYTES),
    )
)


_SPIN_RING_INTEGRATE_CONCAT_MAX_NALM = int(
    os.environ.get("TORCHSKY_SPIN_RING_INTEGRATE_CONCAT_MAX_NALM", "4096")
)


_SPIN_RING_INTERP_CONCAT_CPP_ENABLE = (
    os.environ.get("TORCHSKY_SPIN_RING_INTERP_CONCAT_CPP", "1") != "0"
)


_SPIN_RING_INTEGRATE_CONCAT_CPP_ENABLE = (
    os.environ.get("TORCHSKY_SPIN_RING_INTEGRATE_CONCAT_CPP", "1") != "0"
)


_SPIN_RING_FINALIZE_CPP_ENABLE = (
    os.environ.get("TORCHSKY_SPIN_RING_FINALIZE_CPP", "1") != "0"
)


_SPIN_MAP2ALM_RING_CONCAT_CPP_ENABLE = (
    os.environ.get("TORCHSKY_SPIN_MAP2ALM_RING_CONCAT_CPP", "1") != "0"
)


_SPIN_RING_NEST_ENABLE = os.environ.get("TORCHSKY_SPIN_RING_NEST_ENABLE", "1") != "0"


_RING_SMALL_DFT_MAX_NPH = int(os.environ.get("TORCHSKY_RING_SMALL_DFT_MAX_NPH", "0"))


_RING_FOURIER_MODES_CPP_MAX_M = int(
    os.environ.get("TORCHSKY_RING_FOURIER_MODES_CPP_MAX_M", "1024")
)


_RING_FOURIER_SYNTH_CPP_MAX_M = int(
    os.environ.get("TORCHSKY_RING_FOURIER_SYNTH_CPP_MAX_M", "1024")
)


_LEGENDRE_REC_CACHE: SpectralCacheNamespace[
    tuple[tuple[float, ...], tuple[float, ...]]
] = _SPECTRAL_CACHE.namespace("legendre_recurrence", device="cpu", dtype=torch.float64)
