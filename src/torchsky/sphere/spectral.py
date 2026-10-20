"""CPU-first spherical harmonic primitives for HEALPix maps (torch + torchsky C++)."""

import os
import math
import functools
import warnings
from pathlib import Path

import torch
from torch import Tensor

from .. import _ops as _native_ops
from .._dtype import work_complex_dtype as _preferred_complex_dtype
from .._dtype import work_dtype as _preferred_real_dtype
from torchfits import read_tensor as read_image
from torchfits.table import read_torch as read_table

from . import healpix as _healpix
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
_YLM_CACHE: SpectralCacheNamespace[
    tuple[tuple[Tensor, ...], tuple[Tensor, ...]]
] = _SPECTRAL_CACHE.namespace(
    "ylm", device="cpu", dtype=torch.complex128, max_entry_bytes=_SCALAR_CACHE_ENTRY_LIMIT
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
    "spin_ylm", device="cpu", dtype=torch.complex128, max_entry_bytes=_SPIN_CACHE_ENTRY_LIMIT
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
_SCALAR_MAP2ALM_MAT_CACHE: SpectralCacheNamespace[Tensor] = (
    _SPECTRAL_CACHE.namespace(
        "scalar_map2alm_mat",
        device="cpu",
        dtype=torch.complex128,
        max_entry_bytes=_SCALAR_CACHE_ENTRY_LIMIT,
    )
)
_SCALAR_ALM2MAP_MAT_CACHE: SpectralCacheNamespace[Tensor] = (
    _SPECTRAL_CACHE.namespace(
        "scalar_alm2map_mat",
        device="cpu",
        dtype=torch.complex128,
        max_entry_bytes=_SCALAR_CACHE_ENTRY_LIMIT,
    )
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
_RING_AZIMUTH_PHASE_CACHE: SpectralCacheNamespace[Tensor] = (
    _SPECTRAL_CACHE.namespace(
        "ring_azimuth_phase", device="cpu", dtype=torch.complex128
    )
)
_RING_ALIAS_INDEX_CACHE: SpectralCacheNamespace[Tensor] = _SPECTRAL_CACHE.namespace(
    "ring_alias_index", device="cpu", dtype=torch.int64
)
_RING_SPIN_CONJ_INDEX_CACHE: SpectralCacheNamespace[tuple[Tensor, Tensor, Tensor]] = (
    _SPECTRAL_CACHE.namespace(
        "ring_spin_conj_index", device="cpu", dtype=torch.int64
    )
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
    tuple[
        tuple[int, int, int, int, int] | None, tuple[tuple[int, Tensor, Tensor], ...]
    ]
] = _SPECTRAL_CACHE.namespace(
    "ring_group", device="cpu", dtype=torch.int64
)
_SCALAR_RING_AUTO_MIN_BYTES = int(
    os.environ.get("TORCHSKY_SCALAR_RING_AUTO_MIN_BYTES", "0")
)
_SCALAR_RECURRENCE_CPP_ENABLE = (
    os.environ.get("TORCHSKY_SCALAR_MAP2ALM_RECURRENCE_CPP", "1") != "0"
)
_SCALAR_INTERP_CPP_ENABLE = (
    os.environ.get("TORCHSKY_SCALAR_ALM2MAP_RECURRENCE_CPP", "1") != "0"
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
_RING_FOURIER_CPP_ENABLE = os.environ.get("TORCHSKY_RING_FOURIER_CPP", "1") != "0"
_RING_FOURIER_MODES_CPP_ENABLE = (
    os.environ.get("TORCHSKY_RING_FOURIER_MODES_CPP", "1") != "0"
)
_RING_FOURIER_SYNTH_CPP_ENABLE = (
    os.environ.get("TORCHSKY_RING_FOURIER_SYNTH_CPP", "1") != "0"
)
_RING_FOURIER_MODES_CPP_MAX_M = int(
    os.environ.get("TORCHSKY_RING_FOURIER_MODES_CPP_MAX_M", "1024")
)
_RING_FOURIER_SYNTH_CPP_MAX_M = int(
    os.environ.get("TORCHSKY_RING_FOURIER_SYNTH_CPP_MAX_M", "1024")
)

_LEGENDRE_REC_CACHE: SpectralCacheNamespace[
    tuple[tuple[float, ...], tuple[float, ...]]
] = _SPECTRAL_CACHE.namespace("legendre_recurrence", device="cpu", dtype=torch.float64)


def _smoke_test_torch_compile() -> bool:
    """Smoke-test torch.compile at import time so that missing CUDA headers
    (cuda.h) don't cause runtime failures later in Wigner d-matrix helpers."""
    try:

        @torch.compile
        def _test_fn(x: Tensor) -> Tensor:
            return x.sin() + x.cos()

        _test_fn(torch.randn(10, dtype=torch.float64))
        return True
    except Exception:
        warnings.warn(
            "torch.compile unavailable (missing CUDA headers or other issue); "
            "Wigner d-matrix helpers will run uncompiled.",
            stacklevel=2,
        )
        return False


_COMPILE_OK: bool | None = None


def _compile_available() -> bool:
    """Probe torch.compile lazily on first use, never at import time."""
    global _COMPILE_OK
    if _COMPILE_OK is None:
        if os.environ.get("TORCHSKY_SHT_COMPILE", "1") == "0":
            _COMPILE_OK = False
        else:
            _COMPILE_OK = _smoke_test_torch_compile()
    return bool(_COMPILE_OK)


def _maybe_compile(fn):
    """Compile on first call when available, falling back on specialization errors."""
    compiled = None
    compile_disabled = False

    @functools.wraps(fn)
    def _compiled_or_eager(*args, **kwargs):
        nonlocal compiled, compile_disabled
        if compile_disabled or not _compile_available():
            return fn(*args, **kwargs)
        if compiled is None:
            compiled = torch.compile(fn)
        try:
            return compiled(*args, **kwargs)
        except Exception as compile_error:
            try:
                result = fn(*args, **kwargs)
            except Exception:
                raise compile_error
            compile_disabled = True
            warnings.warn(
                f"torch.compile failed for {fn.__name__}; using eager execution",
                RuntimeWarning,
                stacklevel=2,
            )
            return result

    return _compiled_or_eager


def _accumulate_add(
    dst: Tensor, dim: int, index: Tensor, src: Tensor, *, scatter: bool
) -> Tensor:
    """In-place add. MPS has no complex scatter/index_add; split real/imag."""
    if src.is_complex() and src.device.type == "mps":
        real = torch.zeros(dst.shape, dtype=src.real.dtype, device=src.device)
        imag = torch.zeros_like(real)
        if scatter:
            real.scatter_add_(dim, index, src.real)
            imag.scatter_add_(dim, index, src.imag)
        else:
            real.index_add_(dim, index, src.real)
            imag.index_add_(dim, index, src.imag)
        return torch.complex(real, imag)
    if scatter:
        dst.scatter_add_(dim, index, src)
    else:
        dst.index_add_(dim, index, src)
    return dst


_RING_WEIGHT_CACHE: SpectralCacheNamespace[Tensor] = _SPECTRAL_CACHE.namespace(
    "ring_weight", device="cpu", dtype=torch.float64
)
_PIXEL_WEIGHT_CACHE: SpectralCacheNamespace[Tensor] = _SPECTRAL_CACHE.namespace(
    "pixel_weight", device="cpu", dtype=torch.float64
)
_HEALPIX_FULL_WEIGHT_REL = "full_weights/healpix_full_weights_nside_%04d.fits"


def _even_legendre_p(z: Tensor, n_even: int) -> Tensor:
    """``P_0, P_2, ..., P_{2*(n_even-1)}`` via Bonnet recurrence, shape ``(n_even, N)``."""
    if n_even < 1:
        raise ValueError("n_even must be positive")
    one = torch.ones_like(z)
    rows = [one]
    if n_even == 1:
        return torch.stack(rows, dim=0)
    p0 = one
    p1 = z
    for n in range(1, 2 * (n_even - 1) + 1):
        p2 = ((2 * n + 1) * z * p1 - n * p0) / (n + 1)
        p0 = p1
        p1 = p2
        if n % 2 == 1:
            rows.append(p2)
    return torch.stack(rows, dim=0)


def _healpix_ring_quadrature_weights(nside: int) -> Tensor:
    """Per-pixel RING quadrature weights with mean 1.

    Tikhonov-regularized even-Legendre moment matching through
    ``l = 2*min(nside, 128)-2`` (Gorski et al. 2005 equal-area rings).
    ``use_weights`` applies this quadrature. Pixel-weight FITS files are a
    separate product loaded only when ``use_pixel_weights=True``.
    """
    nside_i = int(nside)
    n_even = min(nside_i, 128)
    key = (nside_i, n_even)

    def build() -> Tensor:
        npix = _healpix.nside2npix(nside_i)
        pix = torch.arange(npix, dtype=torch.int64)
        ring = _ring_pix2ring_index(nside_i)
        nring = int(ring.max().item()) + 1
        counts = torch.bincount(ring, minlength=nring).to(torch.float64)
        theta, _ = _healpix.pix2ang(nside_i, pix, nest=False)
        z = torch.cos(theta.to(dtype=torch.float64))
        z_sum = torch.zeros(nring, dtype=torch.float64)
        z_sum.scatter_add_(0, ring, z)
        z_ring = z_sum / counts.clamp_min(1.0)
        design = _even_legendre_p(z_ring, n_even) * counts.unsqueeze(0)
        target = torch.zeros(n_even, dtype=torch.float64)
        target[0] = float(npix)
        ones = torch.ones(nring, dtype=torch.float64)
        ridge = 1.0e-6 * float(npix)
        gram = design @ design.T + ridge * torch.eye(n_even, dtype=torch.float64)
        coeff = torch.linalg.solve(gram, target - design @ ones)
        ring_w = ones + design.T @ coeff
        pix_w = ring_w[ring]
        return pix_w / pix_w.mean().clamp_min(1.0e-30)

    return _RING_WEIGHT_CACHE.get_or_build(key, build)


def _resolve_healpix_datapath(
    datapath: str | os.PathLike[str] | None,
) -> Path | None:
    if datapath is not None:
        return Path(datapath)
    env = os.environ.get("TORCHSKY_HEALPIX_DATAPATH")
    if env:
        return Path(env)
    return None


def _healpix_full_weight_path(nside: int, datapath: Path) -> Path:
    if datapath.is_file():
        return datapath
    return datapath / (_HEALPIX_FULL_WEIGHT_REL % int(nside))


def _read_healpix_full_weight_vector(path: str, nside: int) -> Tensor:
    """Load HEALPix full-pixel weight *corrections* (applied as ``1 + w``)."""
    npix = _healpix.nside2npix(nside)
    candidates: list[Tensor] = []
    for hdu in (1, 0):
        try:
            data = torch.as_tensor(read_image(path, hdu=hdu, mmap=False)).reshape(-1)
        except Exception:
            continue
        if data.numel() > 0:
            candidates.append(data)
    if not candidates:
        try:
            table = read_table(path)
        except Exception as exc:
            raise ValueError(
                f"could not read HEALPix pixel weights from {path!r}"
            ) from exc
        if not hasattr(table, "keys"):
            raise ValueError(f"pixel weight table in {path!r} has no columns")
        first = next(iter(table.keys()))
        candidates.append(torch.as_tensor(table[first]).reshape(-1))
    data = candidates[0].to(dtype=torch.float64)
    if data.numel() == npix:
        return data
    if data.numel() == 3 * npix:
        return data.reshape(3, npix)[0]
    raise ValueError(
        f"pixel weights in {path!r} have {data.numel()} values; expected {npix}"
    )


def _healpix_full_pixel_weight_corrections(
    nside: int, datapath: str | os.PathLike[str] | None
) -> Tensor:
    root = _resolve_healpix_datapath(datapath)
    if root is None:
        raise FileNotFoundError(
            "use_pixel_weights=True requires datapath or TORCHSKY_HEALPIX_DATAPATH "
            "pointing at a healpy-data tree "
            "(full_weights/healpix_full_weights_nside_NNNN.fits). "
            "torchsky does not download or vendor those files"
        )
    path = _healpix_full_weight_path(nside, root)
    key = (int(nside), str(path.resolve()) if path.exists() else str(path))

    def build() -> Tensor:
        if not path.is_file():
            raise FileNotFoundError(
                f"pixel weights file missing at {path}. "
                "Set datapath to a local healpy-data checkout or pass use_pixel_weights=False"
            )
        weights = _read_healpix_full_weight_vector(str(path), int(nside))
        if not bool(torch.isfinite(weights).all().item()):
            raise ValueError(f"pixel weights in {path} are not finite")
        return weights

    return _PIXEL_WEIGHT_CACHE.get_or_build(key, build)


def _gal_cut_keep_mask(
    nside: int, gal_cut_deg: float, *, nest: bool, device: torch.device
) -> Tensor:
    cut = float(gal_cut_deg)
    if cut < 0.0:
        raise ValueError("gal_cut must be non-negative")
    npix = _healpix.nside2npix(nside)
    pix = torch.arange(npix, device=device, dtype=torch.int64)
    _lon, lat = _healpix.pix2ang(nside, pix, nest=nest, lonlat=True)
    return torch.abs(lat) >= cut


def _apply_map2alm_weights(
    rows: Tensor,
    *,
    nside: int,
    gal_cut: float,
    use_weights: bool,
    use_pixel_weights: bool,
    datapath: str | os.PathLike[str] | None = None,
) -> Tensor:
    """Apply galactic cut, RING quadrature, or HEALPix full-pixel weights."""
    if use_weights and use_pixel_weights:
        raise ValueError("use_weights and use_pixel_weights are mutually exclusive")
    out = rows
    if float(gal_cut) > 0.0:
        keep = _gal_cut_keep_mask(nside, float(gal_cut), nest=False, device=rows.device)
        out = out * keep.to(dtype=out.dtype)
    if use_pixel_weights:
        corr = _healpix_full_pixel_weight_corrections(nside, datapath).to(
            device=rows.device, dtype=out.dtype
        )
        out = out * (1.0 + corr).unsqueeze(0)
    elif use_weights:
        w = _healpix_ring_quadrature_weights(nside).to(
            device=rows.device, dtype=out.dtype
        )
        out = out * w.unsqueeze(0)
    return out


def alm_size(lmax: int, mmax: int | None = None) -> int:
    """Return number of alm coefficients in healpy ordering."""
    if lmax < 0:
        raise ValueError("lmax must be non-negative")
    mm = lmax if mmax is None else int(mmax)
    if mm < 0 or mm > lmax:
        raise ValueError("mmax must satisfy 0 <= mmax <= lmax")
    return (mm + 1) * (2 * lmax - mm + 2) // 2


def alm_index(ell: int, m: int, lmax: int, mmax: int | None = None) -> int:
    """Return flat alm index for mode (ell, m) in healpy ordering."""
    if ell < 0 or m < 0:
        raise ValueError("l and m must be non-negative")
    if m > ell:
        raise ValueError("m must be <= l")
    mm = lmax if mmax is None else int(mmax)
    if ell > lmax or m > mm:
        raise ValueError("(l,m) outside lmax/mmax bounds")
    start = m * (lmax + 1) - (m * (m - 1)) // 2
    return int(start + (ell - m))


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
    if alm_size(int(lmax), mm) != nalm:
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
    if ll < mm or alm_size(ll, mm) != nalm:
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
        theta = theta.to(dtype=torch.float64, device="cpu")
        phi = phi.to(dtype=torch.float64, device="cpu")
        return theta, phi, torch.cos(theta), 4.0 * math.pi / float(npix)

    return _ANGLE_CACHE.get_or_build(key, build)


def _ring2nest_perm_for_nside(nside: int) -> Tensor:
    ns = int(nside)

    def build() -> Tensor:
        npix = _healpix.nside2npix(ns)
        idx = torch.arange(npix, dtype=torch.int64)
        return (
            _healpix.ring2nest(ns, idx)
            .to(dtype=torch.int64, device="cpu")
            .contiguous()
        )

    return _RING2NEST_PERM_CACHE.get_or_build(ns, build)


def _nest2ring_perm_for_nside(nside: int) -> Tensor:
    ns = int(nside)

    def build() -> Tensor:
        npix = _healpix.nside2npix(ns)
        idx = torch.arange(npix, dtype=torch.int64)
        return (
            _healpix.nest2ring(ns, idx)
            .to(dtype=torch.int64, device="cpu")
            .contiguous()
        )

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



def _legendre_rec_coeffs(
    m: int, lmax: int
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return recurrence coefficients a_lm, b_lm for l=m+2..lmax."""
    key = (m, lmax)

    def build() -> tuple[tuple[float, ...], tuple[float, ...]]:
        a_list = []
        b_list = []
        for ell in range(m + 2, lmax + 1):
            a_lm = math.sqrt((4.0 * ell * ell - 1.0) / (ell * ell - m * m))
            b_lm = math.sqrt(
                (2.0 * ell + 1.0)
                / (2.0 * ell - 3.0)
                * (((ell - 1.0) * (ell - 1.0)) - m * m)
                / (ell * ell - m * m)
            )
            a_list.append(a_lm)
            b_list.append(b_lm)
        return tuple(a_list), tuple(b_list)

    return _LEGENDRE_REC_CACHE.get_or_build(key, build)


def _legendre_l_sequence_loop(
    x: Tensor,
    prev2: Tensor,
    prev1: Tensor,
    a_list: tuple[float, ...],
    b_list: tuple[float, ...],
) -> list[Tensor]:
    """Inner recurrence loop, potentially compilable."""
    res = []
    p2 = prev2
    p1 = prev1
    for i in range(len(a_list)):
        cur = a_list[i] * x * p1 - b_list[i] * p2
        res.append(cur)
        p2 = p1
        p1 = cur
    return res


def _legendre_l_sequence(m: int, x: Tensor, lmax: int) -> Tensor:
    """Fully normalized Associated Legendre P_l^m(x) for l=m..lmax. shape [lmax-m+1, npix]."""
    # Standard normalization for Ylm includes 1/sqrt(4pi)
    # Pmm = (-1)^m * sqrt( (2m+1)!! / (2m)!! ) * (1-x^2)^(m/2) * sqrt(1/4pi)

    x.shape[0]

    sin2theta = torch.clamp(1.0 - x * x, min=1e-30)

    ln_4pi = math.log(4.0 * math.pi)
    ln2 = math.log(2.0)

    # log_norm_mm = 0.5 * (log(2m+1) - log(4pi) + lgamma(2m+1) - 2*(m*log(2) + lgamma(m+1)))
    log_pmm = 0.5 * (
        math.log(2 * m + 1)
        - ln_4pi
        + math.lgamma(2 * m + 1)
        - 2 * (m * ln2 + math.lgamma(m + 1))
    )
    log_sin_m = 0.5 * m * torch.log(sin2theta)

    pmm = torch.exp(log_pmm + log_sin_m)
    if (m % 2) != 0:
        pmm = -pmm

    if lmax == m:
        return pmm.unsqueeze(0)

    pm1 = x * math.sqrt(2 * m + 3) * pmm

    if lmax == m + 1:
        return torch.stack([pmm, pm1], dim=0)

    a_list, b_list = _legendre_rec_coeffs(m, lmax)

    # Note: torch.compile overhead is often not worth it for small L,
    # but for large L (>512) it can be beneficial.
    # We use a simple loop here which is already faster than the old version.
    seq = [pmm, pm1]
    seq.extend(_legendre_l_sequence_loop(x, pmm, pm1, a_list, b_list))

    return torch.stack(seq, dim=0)



def _ylm_basis(
    nside: int, nest: bool, lmax: int, mmax: int
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
    """Cached Y_lm basis blocks by m: each block has shape [l_count, npix]."""
    key = (int(nside), bool(nest), int(lmax), int(mmax))

    def build() -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
        _, phi, x, _ = _angles_for_nside(nside, nest)
        y_blocks: list[Tensor] = []
        y_conj_blocks: list[Tensor] = []
        m_range = torch.arange(mmax + 1, device=phi.device, dtype=phi.dtype)
        all_phases = torch.exp(
            torch.complex(
                torch.zeros(1, device=phi.device, dtype=phi.dtype),
                m_range.unsqueeze(1) * phi.unsqueeze(0),
            )
        )
        for m in range(mmax + 1):
            p_seq = _legendre_l_sequence(m, x, lmax)
            phase = all_phases[m]
            y = p_seq.to(dtype=torch.complex128) * phase.unsqueeze(0)
            y_blocks.append(y)
            y_conj_blocks.append(torch.conj(y))
        return tuple(y_blocks), tuple(y_conj_blocks)

    estimated_bytes = (
        2
        * alm_size(int(lmax), int(mmax))
        * _healpix.nside2npix(int(nside))
        * 16
    )
    return _YLM_CACHE.get_or_build(key, build, estimated_bytes=estimated_bytes)


def _wigner_d_element(ell: int, m: int, mp: int, theta: Tensor) -> Tensor:
    """
    Compute Wigner small-d element d^ell_{m,mp}(theta).
    Direct implementation using Wigner's formula.
    WARNING: Numerically unstable for large ell. Use for small ell only (e.g. bootstrapping).
    """
    if abs(m) > ell or abs(mp) > ell:
        return torch.zeros_like(theta)

    ct = torch.cos(theta * 0.5)
    st = torch.sin(theta * 0.5)
    out = torch.zeros_like(theta)

    log_pref = 0.5 * (
        math.lgamma(float(ell + m + 1))
        + math.lgamma(float(ell - m + 1))
        + math.lgamma(float(ell + mp + 1))
        + math.lgamma(float(ell - mp + 1))
    )
    kmin = max(0, m - mp)
    kmax = min(ell + m, ell - mp)
    if kmin > kmax:
        return out

    for k in range(kmin, kmax + 1):
        a = ell + m - k
        b = k
        c = mp - m + k
        d = ell - mp - k
        if min(a, b, c, d) < 0:
            continue
        sign = -1.0 if ((k + mp - m) & 1) else 1.0
        log_den = (
            math.lgamma(float(a + 1))
            + math.lgamma(float(b + 1))
            + math.lgamma(float(c + 1))
            + math.lgamma(float(d + 1))
        )
        coeff = sign * math.exp(log_pref - log_den)
        p_ct = 2 * ell + m - mp - 2 * k
        p_st = mp - m + 2 * k
        out = out + coeff * (ct**p_ct) * (st**p_st)
    return out


@_maybe_compile
def _wigner_d_l_sequence(
    ell_min: int, ell_max: int, m: int, mp: int, theta: Tensor
) -> Tensor:
    """
    Compute Wigner small-d elements d^ell_{m,mp}(theta) for ell in [ell_min, ell_max].
    Returns tensor of shape [ell_max - ell_min + 1, npix].
    Uses a stable three-term recurrence relation.
    """
    if ell_max < ell_min:
        return torch.empty(
            (0, theta.shape[0]), dtype=theta.dtype, device=theta.device
        )

    # Bootstrap with direct formula for the first two terms
    # The direct formula is stable and exact for terms near the minimal l (max(|m|,|mp|))
    # where the summation has few terms.
    d0 = _wigner_d_element(ell_min, m, mp, theta)
    if ell_max == ell_min:
        return d0.unsqueeze(0)

    d1 = _wigner_d_element(ell_min + 1, m, mp, theta)
    seq = [d0, d1]

    prev2 = d0
    prev1 = d1

    x = torch.cos(theta)

    # Recurrence relation:
    # l*sqrt((l^2-m^2)(l^2-mp^2)) * d^l_{m,mp} =
    #    (2l-1)(l(l-1)x - m*mp) * d^{l-1}_{m,mp}
    #    - (l-1)*sqrt(((l-1)^2-m^2)((l-1)^2-mp^2)) * d^{l-2}_{m,mp}

    for ell in range(ell_min + 2, ell_max + 1):
        # Recurrence:
        # (K_l / l) * d_l = (2l-1)(x - m*mp/(l(l-1))) * d_{l-1} - (K_{l-1} / (l-1)) * d_{l-2}
        # where K_l = sqrt((l^2 - m^2)(l^2 - mp^2))

        # Current K_l
        sq_l = math.sqrt((ell * ell - m * m) * (ell * ell - mp * mp))
        # Previous K_{l-1} needs to be recomputed or cached.
        # (ell-1)^2 - m^2 ...
        # Actually it's cleaner to compute just what's needed.

        sq_lm1 = math.sqrt(
            (((ell - 1) * (ell - 1)) - m * m) * (((ell - 1) * (ell - 1)) - mp * mp)
        )

        # Coefficients
        # LHS factor: pre_factor = l / K_l
        if sq_l == 0:
            # Should not happen for ell > ell_min >= |m|, |mp|
            # But strictly, if l=|m| or l=|mp|, sq_l=0.
            # Loop starts ell_min+2, so l > |m| and l > |mp| strictly.
            pre_factor = 0.0  # Should be safe
        else:
            pre_factor = float(ell) / sq_l

        term1_scale = 2 * ell - 1
        term1_shift = float(m * mp) / float(ell * (ell - 1))

        term2_val = sq_lm1 / float(ell - 1)

        cur = pre_factor * (
            (term1_scale * (x - term1_shift)) * prev1 - term2_val * prev2
        )

        seq.append(cur)
        prev2 = prev1
        prev1 = cur

    return torch.stack(seq, dim=0)



def _build_spin_ylm_basis(
    nside: int,
    nest: bool,
    lmax: int,
    mmax: int,
    spin: int,
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
    theta, phi, _, _ = _angles_for_nside(nside, nest)
    y_blocks: list[Tensor] = []
    y_conj_blocks: list[Tensor] = []
    # ⚡ Bolt: Vectorized phase multipliers for all m
    m_range = torch.arange(mmax + 1, device=phi.device, dtype=phi.dtype)
    # [mmax+1, npix]
    all_phases = torch.exp(
        torch.complex(
            torch.zeros(1, device=phi.device, dtype=phi.dtype),
            m_range.unsqueeze(1) * phi.unsqueeze(0),
        )
    )
    spin_phase = -1.0 if (spin & 1) else 1.0  # (-1)^spin

    # ⚡ Bolt: Pre-calculate all possible ell norms for the given lmax and spin
    abs_spin = abs(spin)
    ell_all = torch.arange(abs_spin, lmax + 1, dtype=torch.float64, device=theta.device)
    all_ell_norms = torch.sqrt((2 * ell_all + 1) / (4.0 * math.pi))

    for m in range(mmax + 1):
        phase = all_phases[m]
        m_phase = -1.0 if (m & 1) else 1.0  # Condon-Shortley convention

        # ⚡ Bolt: Use pre-calculated norms instead of re-calculating for every m
        ell_start = max(m, abs(spin))
        if ell_start > lmax:
            y_blocks.append(
                torch.zeros(
                    (0, theta.shape[0]), dtype=torch.complex128, device=theta.device
                )
            )
            y_conj_blocks.append(
                torch.zeros(
                    (0, theta.shape[0]), dtype=torch.complex128, device=theta.device
                )
            )
            continue

        # Get d-matrix sequence for all required l
        d_seq = _wigner_d_l_sequence(ell_start, lmax, m, -spin, theta)

        # ⚡ Bolt: Fast slice from pre-calculated norms
        # ell_start - abs_spin is the offset into our pre-calculated norms tensor
        norms = all_ell_norms[ell_start - abs_spin :]

        # Broadcast and multiply: (spin_phase * m_phase) * norm * d * phase
        combined_scale = (spin_phase * m_phase) * norms.unsqueeze(1)
        y = (combined_scale * d_seq).to(torch.complex128) * phase.unsqueeze(0)

        if ell_start > m:
            n_pre = ell_start - m
            # We need to prepend n_pre rows of zeros
            # y currently has shape [ell_max - ell_start + 1, npix]
            # We want [ell_max - m + 1, npix]
            pre_zeros = torch.zeros(
                (n_pre, theta.shape[0]), dtype=torch.complex128, device=theta.device
            )
            y = torch.cat([pre_zeros, y], dim=0)

        y_blocks.append(y)
        y_conj_blocks.append(torch.conj(y))

    return tuple(y_blocks), tuple(y_conj_blocks)


def _spin_ylm_basis(
    nside: int,
    nest: bool,
    lmax: int,
    mmax: int,
    spin: int,
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
    key = (int(nside), bool(nest), int(lmax), int(mmax), int(spin))
    estimated_bytes = (
        2
        * alm_size(int(lmax), int(mmax))
        * _healpix.nside2npix(int(nside))
        * 16
    )
    return _SPIN_YLM_CACHE.get_or_build(
        key,
        lambda: _build_spin_ylm_basis(nside, nest, lmax, mmax, spin),
        estimated_bytes=estimated_bytes,
    )


_SPIN_YLM_CONCAT_CACHE: SpectralCacheNamespace[tuple[Tensor, Tensor]] = (
    _SPECTRAL_CACHE.namespace(
        "spin_ylm_concat",
        device="cpu",
        dtype=torch.complex128,
        max_entry_bytes=_SPIN_CACHE_ENTRY_LIMIT,
    )
)


def _build_spin_ylm_basis_concat(
    nside: int,
    nest: bool,
    lmax: int,
    mmax: int,
    spin: int,
) -> tuple[Tensor, Tensor]:
    """
    Returns concatenated (nalm, npix) matrices for the spin-weighted basis.
    (Y, Y_conj)
    """
    npix = _healpix.nside2npix(int(nside))
    y_blocks, y_conj_blocks = _spin_ylm_basis(nside, nest, lmax, mmax, spin)

    if not y_blocks:
        device = torch.device("cpu")
        dtype = torch.complex128
        return (
            torch.empty((0, npix), dtype=dtype, device=device),
            torch.empty((0, npix), dtype=dtype, device=device),
        )
    return torch.cat(y_blocks, dim=0), torch.cat(y_conj_blocks, dim=0)


def _spin_ylm_basis_concat(
    nside: int,
    nest: bool,
    lmax: int,
    mmax: int,
    spin: int,
) -> tuple[Tensor, Tensor]:
    key = (int(nside), bool(nest), int(lmax), int(mmax), int(spin))
    bytes_needed = (
        2
        * alm_size(int(lmax), int(mmax))
        * _healpix.nside2npix(int(nside))
        * 16
    )
    return _SPIN_YLM_CONCAT_CACHE.get_or_build(
        key,
        lambda: _build_spin_ylm_basis_concat(nside, nest, lmax, mmax, spin),
        estimated_bytes=bytes_needed,
    )


def _spin_map2alm_mats(
    nside: int,
    nest: bool,
    lmax: int,
    mmax: int,
    spin: int,
) -> tuple[Tensor, Tensor] | None:
    """
    Optionally return cached concatenated map2alm matrices for spin transforms.

    Returns `(Yc_plus_t, Yc_minus_t)` with shape `[npix, nalm]` each, where
    `Yc` is the conjugated spin-weighted basis.
    """
    npix = _healpix.nside2npix(int(nside))
    nalm = alm_size(int(lmax), int(mmax))
    bytes_needed = int(npix * nalm * 16 * 2)
    if bytes_needed > _SPIN_CACHE_ENTRY_LIMIT:
        return None

    key = (int(nside), bool(nest), int(lmax), int(mmax), int(spin))

    def build() -> tuple[Tensor, Tensor]:
        _, y_conj_plus_blocks = _spin_ylm_basis(nside, nest, lmax, mmax, spin)
        _, y_conj_minus_blocks = _spin_ylm_basis(nside, nest, lmax, mmax, -spin)
        yc_plus_t = torch.cat(
            [blk.transpose(0, 1) for blk in y_conj_plus_blocks], dim=1
        ).contiguous()
        yc_minus_t = torch.cat(
            [blk.transpose(0, 1) for blk in y_conj_minus_blocks], dim=1
        ).contiguous()
        return yc_plus_t, yc_minus_t

    return _SPIN_MAP2ALM_MAT_CACHE.get_or_build(
        key, build, estimated_bytes=bytes_needed
    )


def _spin_alm2map_mats(
    nside: int,
    nest: bool,
    lmax: int,
    mmax: int,
    spin: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor] | None:
    """
    Optionally return cached concatenated alm2map matrices for spin transforms.

    Returns `(Y_plus, Y_minus, Y_plus_m0, Y_minus_m0)` where:
    - `Y_plus`, `Y_minus` have shape `[nalm, npix]`.
    - `Y_plus_m0`, `Y_minus_m0` are the `m=0` blocks with shape `[lmax+1, npix]`.
    """
    npix = _healpix.nside2npix(int(nside))
    nalm = alm_size(int(lmax), int(mmax))
    bytes_needed = int(npix * nalm * 16 * 2)
    if bytes_needed > _SPIN_CACHE_ENTRY_LIMIT:
        return None

    key = (int(nside), bool(nest), int(lmax), int(mmax), int(spin))

    def build() -> tuple[Tensor, Tensor, Tensor, Tensor]:
        y_plus_blocks, _ = _spin_ylm_basis(nside, nest, lmax, mmax, spin)
        y_minus_blocks, _ = _spin_ylm_basis(nside, nest, lmax, mmax, -spin)
        y_plus = torch.cat(y_plus_blocks, dim=0).contiguous()
        y_minus = torch.cat(y_minus_blocks, dim=0).contiguous()
        return y_plus, y_minus, y_plus_blocks[0], y_minus_blocks[0]

    return _SPIN_ALM2MAP_MAT_CACHE.get_or_build(
        key, build, estimated_bytes=bytes_needed
    )


def _scalar_map2alm_mat(
    nside: int,
    nest: bool,
    lmax: int,
    mmax: int,
) -> Tensor | None:
    npix = _healpix.nside2npix(int(nside))
    nalm = alm_size(int(lmax), int(mmax))
    bytes_needed = int(npix * nalm * 16)
    if bytes_needed > _SCALAR_CACHE_ENTRY_LIMIT:
        return None
    key = (int(nside), bool(nest), int(lmax), int(mmax))

    def build() -> Tensor:
        _, y_conj_blocks = _ylm_basis(nside, nest, lmax, mmax)
        return torch.cat(
            [blk.transpose(0, 1) for blk in y_conj_blocks], dim=1
        ).contiguous()

    return _SCALAR_MAP2ALM_MAT_CACHE.get_or_build(
        key, build, estimated_bytes=bytes_needed
    )


def _scalar_alm2map_mat(
    nside: int,
    nest: bool,
    lmax: int,
    mmax: int,
) -> Tensor | None:
    npix = _healpix.nside2npix(int(nside))
    nalm = alm_size(int(lmax), int(mmax))
    bytes_needed = int(npix * nalm * 16)
    if bytes_needed > _SCALAR_CACHE_ENTRY_LIMIT:
        return None
    key = (int(nside), bool(nest), int(lmax), int(mmax))


    def build() -> Tensor:
        y_blocks, _ = _ylm_basis(nside, nest, lmax, mmax)
        return torch.cat(y_blocks, dim=0).contiguous()

    return _SCALAR_ALM2MAP_MAT_CACHE.get_or_build(
        key, build, estimated_bytes=bytes_needed
    )


def _build_ring_layout_for_nside(
    nside: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    ns = int(nside)
    npix = _healpix.nside2npix(ns)
    nrings = (4 * ns) - 1
    theta, phi, _, _ = _angles_for_nside(ns, False)

    starts: list[int] = []
    lengths: list[int] = []
    theta_ring: list[float] = []
    phi0_ring: list[float] = []
    cursor = 0
    for ring in range(1, nrings + 1):
        if ring < ns:
            nph = 4 * ring
        elif ring <= (3 * ns):
            nph = 4 * ns
        else:
            nph = 4 * ((4 * ns) - ring)
        starts.append(cursor)
        lengths.append(nph)
        theta_ring.append(float(theta[cursor]))
        phi0_ring.append(float(phi[cursor]))
        cursor += nph
    if cursor != npix:
        raise RuntimeError(f"ring layout mismatch for nside={ns}: {cursor} != {npix}")

    return (
        torch.tensor(starts, dtype=torch.int64),
        torch.tensor(lengths, dtype=torch.int64),
        torch.tensor(theta_ring, dtype=torch.float64),
        torch.tensor(phi0_ring, dtype=torch.float64),
    )


def _ring_layout_for_nside(nside: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    ns = int(nside)
    return _RING_LAYOUT_CACHE.get_or_build(
        ns, lambda: _build_ring_layout_for_nside(ns)
    )


def _build_ring_grouping(
    starts_cpu: Tensor,
    lengths_cpu: Tensor,
) -> tuple[
    tuple[int, int, int, int, int] | None, tuple[tuple[int, Tensor, Tensor], ...]
]:
    lengths_list = [int(v) for v in lengths_cpu.tolist()]
    if not lengths_list:
        return None, ()

    counts: dict[int, int] = {}
    for nph in lengths_list:
        counts[nph] = counts.get(nph, 0) + 1
    common_len = max(counts, key=counts.get)
    common_rings = [i for i, nph in enumerate(lengths_list) if nph == common_len]

    common_block: tuple[int, int, int, int, int] | None = None
    processed = [False] * len(lengths_list)
    if common_rings and all(
        (b - a) == 1 for a, b in zip(common_rings, common_rings[1:])
    ):
        start_ring = int(common_rings[0])
        end_ring = int(common_rings[-1])
        pix_start = int(starts_cpu[start_ring].item())
        pix_end = int(starts_cpu[end_ring].item()) + int(common_len)
        common_block = (int(common_len), start_ring, end_ring, pix_start, pix_end)
        for rid in range(start_ring, end_ring + 1):
            processed[rid] = True

    by_len: dict[int, list[int]] = {}
    for rid, nph in enumerate(lengths_list):
        if processed[rid]:
            continue
        by_len.setdefault(int(nph), []).append(int(rid))

    groups: list[tuple[int, Tensor, Tensor]] = []
    for nph, ring_ids in sorted(by_len.items()):
        ring_idx = torch.tensor(ring_ids, dtype=torch.int64)
        starts_sel = torch.index_select(starts_cpu, 0, ring_idx).unsqueeze(1)
        pix_offsets = torch.arange(int(nph), dtype=torch.int64).unsqueeze(0)
        pix_idx = starts_sel + pix_offsets
        groups.append((int(nph), ring_idx, pix_idx))
    return common_block, tuple(groups)


def _ring_grouping_for_nside(
    nside: int,
) -> tuple[
    tuple[int, int, int, int, int] | None, tuple[tuple[int, Tensor, Tensor], ...]
]:
    key = int(nside)

    def build() -> tuple[
        tuple[int, int, int, int, int] | None,
        tuple[tuple[int, Tensor, Tensor], ...],
    ]:
        starts_cpu, lengths_cpu, _, _ = _ring_layout_for_nside(key)
        return _build_ring_grouping(starts_cpu, lengths_cpu)

    return _RING_GROUP_CACHE.get_or_build(key, build)


def _build_ring_scalar_bases_for_nside(
    nside: int, lmax: int, mmax: int
) -> tuple[Tensor, ...]:
    _, _, theta_ring_cpu, _ = _ring_layout_for_nside(int(nside))
    x_ring = torch.cos(theta_ring_cpu)
    blocks: list[Tensor] = []
    for m in range(int(mmax) + 1):
        p_seq = _legendre_l_sequence(m, x_ring, int(lmax))
        blocks.append(p_seq.to(torch.complex128))
    return tuple(blocks)


def _ring_scalar_bases_for_nside(
    nside: int, lmax: int, mmax: int
) -> tuple[Tensor, ...]:
    key = (int(nside), int(lmax), int(mmax))
    estimated_bytes = (
        alm_size(int(lmax), int(mmax)) * ((4 * int(nside)) - 1) * 16
    )
    return _RING_SCALAR_BASIS_CACHE.get_or_build(
        key,
        lambda: _build_ring_scalar_bases_for_nside(nside, lmax, mmax),
        estimated_bytes=estimated_bytes,
    )


def _ring_scalar_basis_concat(nside: int, lmax: int, mmax: int) -> Tensor:
    """
    Returns concatenated (N_alm, N_rings) basis matrix for scalar Y_lm.
    Uses LRU cache with size limit.
    """
    key = (int(nside), int(lmax), int(mmax))
    bytes_needed = alm_size(int(lmax), int(mmax)) * ((4 * int(nside)) - 1) * 16

    def build() -> Tensor:
        blocks = _ring_scalar_bases_for_nside(nside, lmax, mmax)
        if not blocks:
            return torch.empty(
                (0, 0), dtype=torch.complex128, device=torch.device("cpu")
            )
        return torch.cat(blocks, dim=0)

    return _RING_SCALAR_BASIS_CONCAT_CACHE.get_or_build(
        key, build, estimated_bytes=bytes_needed
    )


class ScalarSHTIntegrate(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, s_modes: Tensor, nside: int, lmax: int, mmax: int, pix_w: float
    ) -> Tensor:
        ctx.nside = nside
        ctx.lmax = lmax
        ctx.mmax = mmax
        ctx.pix_w = pix_w

        _, _, theta_ring_cpu, _ = _ring_layout_for_nside(nside)
        nrings = int(theta_ring_cpu.numel())
        weight = torch.full((nrings,), pix_w, dtype=torch.float64, device="cpu")

        s_modes_cpu = s_modes.detach().to(dtype=torch.complex128, device="cpu")

        if _native_ops.has_spectral_dispatcher():
            out = _native_ops.spectral_scalar_integrate_recurrence(
                s_modes_cpu.contiguous(),
                theta_ring_cpu.contiguous(),
                weight.contiguous(),
                int(lmax),
                int(mmax),
            )
        else:
            s_minus = s_modes_cpu.clone()
            alms_stacked = _native_ops.spectral_spin_integrate_recurrence(
                s_modes_cpu.contiguous(),
                s_minus.contiguous(),
                theta_ring_cpu.contiguous(),
                weight.contiguous(),
                int(lmax),
                int(mmax),
                0,
            )
            out = (alms_stacked[0] + alms_stacked[1]) * 0.5

        return out.to(device=s_modes.device)

    @staticmethod
    def backward(ctx, grad_alm: Tensor):
        grad_alm_cpu = grad_alm.detach().to(device="cpu", dtype=torch.complex128)
        _, _, theta_ring_cpu, _ = _ring_layout_for_nside(ctx.nside)

        if _native_ops.has_spectral_dispatcher():
            grad_s = _native_ops.spectral_scalar_interpolate_recurrence(
                grad_alm_cpu.contiguous(),
                theta_ring_cpu.contiguous(),
                int(ctx.lmax),
                int(ctx.mmax),
            )
        else:
            alm_stack = torch.stack([grad_alm_cpu, grad_alm_cpu], dim=0)
            s_modes_stacked = _native_ops.spectral_spin_interpolate_recurrence(
                alm_stack.contiguous(),
                theta_ring_cpu.contiguous(),
                int(ctx.lmax),
                int(ctx.mmax),
                0,
            )
            grad_s = s_modes_stacked[0]

        grad_s = grad_s * ctx.pix_w
        return grad_s.to(device=grad_alm.device), None, None, None, None


class ScalarSHTInterpolate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alm: Tensor, nside: int, lmax: int, mmax: int) -> Tensor:
        ctx.nside = nside
        ctx.lmax = lmax
        ctx.mmax = mmax

        _, _, theta_ring_cpu, _ = _ring_layout_for_nside(nside)
        alm_cpu = alm.detach().to(dtype=torch.complex128, device="cpu")

        if _native_ops.has_spectral_dispatcher():
            out = _native_ops.spectral_scalar_interpolate_recurrence(
                alm_cpu.contiguous(),
                theta_ring_cpu.contiguous(),
                int(lmax),
                int(mmax),
            )
        else:
            alm_stack = torch.stack([alm_cpu, alm_cpu], dim=0)
            s_modes_stacked = _native_ops.spectral_spin_interpolate_recurrence(
                alm_stack.contiguous(),
                theta_ring_cpu.contiguous(),
                int(lmax),
                int(mmax),
                0,
            )
            out = s_modes_stacked[0]

        return out.to(device=alm.device)

    @staticmethod
    def backward(ctx, grad_s_modes: Tensor):
        grad_s_cpu = grad_s_modes.detach().to(device="cpu", dtype=torch.complex128)

        _, _, theta_ring_cpu, _ = _ring_layout_for_nside(ctx.nside)
        nrings = int(theta_ring_cpu.numel())
        weight = torch.ones((nrings,), dtype=torch.float64, device="cpu")

        if _native_ops.has_spectral_dispatcher():
            grad_alm = _native_ops.spectral_scalar_integrate_recurrence(
                grad_s_cpu.contiguous(),
                theta_ring_cpu.contiguous(),
                weight.contiguous(),
                int(ctx.lmax),
                int(ctx.mmax),
            )
        else:
            s_minus = grad_s_cpu.clone()
            alms_stacked = _native_ops.spectral_spin_integrate_recurrence(
                grad_s_cpu.contiguous(),
                s_minus.contiguous(),
                theta_ring_cpu.contiguous(),
                weight.contiguous(),
                int(ctx.lmax),
                int(ctx.mmax),
                0,
            )
            grad_alm = (alms_stacked[0] + alms_stacked[1]) * 0.5

        return grad_alm.to(device=grad_s_modes.device), None, None, None


class SpinSHTIntegrate(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        s_plus: Tensor,
        s_minus: Tensor,
        nside: int,
        lmax: int,
        mmax: int,
        spin: int,
        pix_w: float,
    ) -> tuple[Tensor, Tensor]:
        ctx.nside = nside
        ctx.lmax = lmax
        ctx.mmax = mmax
        ctx.spin = spin
        ctx.pix_w = pix_w

        _, _, theta_ring_cpu, _ = _ring_layout_for_nside(nside)
        nrings = int(theta_ring_cpu.numel())
        w_vec = torch.full((nrings,), pix_w, dtype=torch.float64, device="cpu")

        sp = s_plus.detach().to(dtype=torch.complex128, device="cpu")
        sm = s_minus.detach().to(dtype=torch.complex128, device="cpu")

        alms_stacked = _native_ops.spectral_spin_integrate_recurrence(
            sp.contiguous(),
            sm.contiguous(),
            theta_ring_cpu.contiguous(),
            w_vec.contiguous(),
            int(lmax),
            int(mmax),
            int(spin),
        )
        return alms_stacked[0].to(device=s_plus.device), alms_stacked[1].to(
            device=s_minus.device
        )

    @staticmethod
    def backward(ctx, grad_a_plus: Tensor, grad_a_minus: Tensor):
        gap = grad_a_plus.detach().to(dtype=torch.complex128, device="cpu")
        gam = grad_a_minus.detach().to(dtype=torch.complex128, device="cpu")

        _, _, theta_ring_cpu, _ = _ring_layout_for_nside(int(ctx.nside))
        alm_stack = torch.stack([gap, gam], dim=0)

        s_modes_stacked = _native_ops.spectral_spin_interpolate_recurrence(
            alm_stack.contiguous(),
            theta_ring_cpu.contiguous(),
            int(ctx.lmax),
            int(ctx.mmax),
            int(ctx.spin),
        )

        gsp = s_modes_stacked[0] * ctx.pix_w
        gsm = s_modes_stacked[1] * ctx.pix_w

        return (
            gsp.to(device=grad_a_plus.device),
            gsm.to(device=grad_a_plus.device),
            None,
            None,
            None,
            None,
            None,
        )


class SpinSHTInterpolate(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        a_plus: Tensor,
        a_minus: Tensor,
        nside: int,
        lmax: int,
        mmax: int,
        spin: int,
    ) -> tuple[Tensor, Tensor]:
        ctx.nside = nside
        ctx.lmax = lmax
        ctx.mmax = mmax
        ctx.spin = spin

        _, _, theta_ring_cpu, _ = _ring_layout_for_nside(nside)

        ap = a_plus.detach().to(dtype=torch.complex128, device="cpu")
        am = a_minus.detach().to(dtype=torch.complex128, device="cpu")
        alm_stack = torch.stack([ap, am], dim=0)

        s_modes_stacked = _native_ops.spectral_spin_interpolate_recurrence(
            alm_stack.contiguous(),
            theta_ring_cpu.contiguous(),
            int(lmax),
            int(mmax),
            int(spin),
        )
        return s_modes_stacked[0].to(device=a_plus.device), s_modes_stacked[1].to(
            device=a_minus.device
        )

    @staticmethod
    def backward(ctx, grad_s_plus: Tensor, grad_s_minus: Tensor):
        gsp = grad_s_plus.detach().to(dtype=torch.complex128, device="cpu")
        gsm = grad_s_minus.detach().to(dtype=torch.complex128, device="cpu")

        _, _, theta_ring_cpu, _ = _ring_layout_for_nside(ctx.nside)
        nrings = int(theta_ring_cpu.numel())
        w_vec = torch.ones((nrings,), dtype=torch.float64, device="cpu")

        alms_stacked = _native_ops.spectral_spin_integrate_recurrence(
            gsp.contiguous(),
            gsm.contiguous(),
            theta_ring_cpu.contiguous(),
            w_vec.contiguous(),
            int(ctx.lmax),
            int(ctx.mmax),
            int(ctx.spin),
        )

        return (
            alms_stacked[0].to(device=grad_s_plus.device),
            alms_stacked[1].to(device=grad_s_minus.device),
            None,
            None,
            None,
            None,
        )


def _scalar_ring_integrate_recurrence_cpu(
    s_modes: Tensor,
    *,
    nside: int,
    lmax: int,
    mmax: int,
    pix_w: float,
) -> Tensor:
    """
    Integrate scalar Fourier modes to alms using C++ recurrence (spin=0).

    s_modes: (mmax+1, nrings) complex tensor of Fourier modes per ring
    Returns: (nalm,) complex tensor of alm coefficients
    """
    return ScalarSHTIntegrate.apply(
        s_modes, int(nside), int(lmax), int(mmax), float(pix_w)
    )


def _scalar_ring_interpolate_recurrence_cpu(
    alm: Tensor,
    *,
    nside: int,
    lmax: int,
    mmax: int,
) -> Tensor:
    """
    Interpolate scalar alms to Fourier modes using C++ recurrence (spin=0).

    alm: (nalm,) complex tensor of alm coefficients
    Returns: (mmax+1, nrings) complex tensor of Fourier modes per ring
    """
    return ScalarSHTInterpolate.apply(alm, int(nside), int(lmax), int(mmax))


def _ring_phi0_phase_for_nside(nside: int, mmax: int, *, sign: int) -> Tensor:
    s = 1 if int(sign) >= 0 else -1
    key = (int(nside), int(mmax), int(s))

    def build() -> Tensor:
        _, _, _, phi0_ring_cpu = _ring_layout_for_nside(int(nside))
        m_vals = torch.arange(int(mmax) + 1, dtype=torch.float64)
        ang = (float(s) * m_vals.unsqueeze(1)) * phi0_ring_cpu.unsqueeze(0)
        return torch.exp(torch.complex(torch.zeros_like(ang), ang)).to(
            torch.complex128
        )

    return _RING_PHI0_PHASE_CACHE.get_or_build(key, build)


def _ring_azimuth_phase(nph: int, mmax: int) -> Tensor:
    key = (int(nph), int(mmax))

    def build() -> Tensor:
        n = int(nph)
        m_vals = torch.arange(int(mmax) + 1, dtype=torch.float64).unsqueeze(1)
        j = torch.arange(n, dtype=torch.float64).unsqueeze(0)
        ang = m_vals * (2.0 * math.pi / float(n)) * j
        return torch.exp(torch.complex(torch.zeros_like(ang), ang)).to(
            torch.complex128
        )

    return _RING_AZIMUTH_PHASE_CACHE.get_or_build(key, build)


def _ring_alias_indices(nph: int, mmax: int) -> Tensor:
    key = (int(nph), int(mmax))
    return _RING_ALIAS_INDEX_CACHE.get_or_build(
        key,
        lambda: torch.remainder(
            torch.arange(int(mmax) + 1, dtype=torch.int64), int(nph)
        ),
    )


def _ring_spin_conj_indices(nph: int, mmax: int) -> tuple[Tensor, Tensor, Tensor]:
    """
    Cached index vectors for spin conjugate mode extraction:
      +m indices, -m indices, and concatenated (+m|-m) indices.
    """
    key = (int(nph), int(mmax))

    def build() -> tuple[Tensor, Tensor, Tensor]:
        n = int(nph)
        mm = int(mmax)
        if n > mm:
            idx_plus = torch.arange(mm + 1, dtype=torch.int64)
        else:
            idx_plus = _ring_alias_indices(n, mm)
        idx_minus = torch.remainder(-idx_plus, n)
        return idx_plus, idx_minus, torch.cat([idx_plus, idx_minus], dim=0)

    return _RING_SPIN_CONJ_INDEX_CACHE.get_or_build(key, build)


def _build_ring_spin_bases_for_nside(
    nside: int, lmax: int, mmax: int, spin: int
) -> tuple[Tensor, ...]:
    _, _, theta_ring_cpu, _ = _ring_layout_for_nside(int(nside))
    sp = int(spin)
    spin_phase = -1.0 if (sp & 1) else 1.0
    blocks: list[Tensor] = []
    for m in range(int(mmax) + 1):
        m_phase = -1.0 if (m & 1) else 1.0

        ell_start = max(m, abs(sp))
        if ell_start > int(lmax):
            blocks.append(
                torch.zeros((0, theta_ring_cpu.shape[0]), dtype=torch.complex128)
            )
            continue

        d_seq = _wigner_d_l_sequence(ell_start, int(lmax), m, -sp, theta_ring_cpu)

        ell_vec = torch.arange(
            ell_start, int(lmax) + 1, dtype=torch.float64, device=theta_ring_cpu.device
        )
        norms = torch.sqrt((2 * ell_vec + 1) / (4.0 * math.pi))

        # norms: [L], d_seq: [L, N]
        block = (spin_phase * m_phase * norms.unsqueeze(1) * d_seq).to(torch.complex128)

        if ell_start > m:
            n_pre = ell_start - m
            pre_zeros = torch.zeros(
                (n_pre, theta_ring_cpu.shape[0]), dtype=torch.complex128
            )
            block = torch.cat([pre_zeros, block], dim=0)

        blocks.append(block)
    return tuple(blocks)


def _ring_spin_bases_for_nside(
    nside: int, lmax: int, mmax: int, spin: int
) -> tuple[Tensor, ...]:
    key = (int(nside), int(lmax), int(mmax), int(spin))
    estimated_bytes = (
        alm_size(int(lmax), int(mmax)) * ((4 * int(nside)) - 1) * 16
    )
    return _RING_SPIN_BASIS_CACHE.get_or_build(
        key,
        lambda: _build_ring_spin_bases_for_nside(nside, lmax, mmax, spin),
        estimated_bytes=estimated_bytes,
    )


_RING_SPIN_CONCAT_CACHE: SpectralCacheNamespace[tuple[Tensor, Tensor]] = (
    _SPECTRAL_CACHE.namespace(
        "ring_spin_concat",
        device="cpu",
        dtype=torch.complex128,
        max_entry_bytes=_SPIN_CACHE_ENTRY_LIMIT,
    )
)


def _ring_spin_basis_concat(
    nside: int, lmax: int, mmax: int, spin: int
) -> tuple[Tensor, Tensor]:
    """
    Returns concatenated (N_alm, N_rings) basis matrices for +spin and -spin.
    """
    key = (int(nside), int(lmax), int(mmax), int(spin))
    bytes_needed = (
        2 * alm_size(int(lmax), int(mmax)) * ((4 * int(nside)) - 1) * 16
    )

    def build() -> tuple[Tensor, Tensor]:
        y_plus_blocks = _ring_spin_bases_for_nside(nside, lmax, mmax, spin)
        y_minus_blocks = _ring_spin_bases_for_nside(nside, lmax, mmax, -spin)
        if not y_plus_blocks:
            device = torch.device("cpu")
            dtype = torch.complex128
            return (
                torch.empty((0, 0), dtype=dtype, device=device),
                torch.empty((0, 0), dtype=dtype, device=device),
            )
        return torch.cat(y_plus_blocks, dim=0), torch.cat(y_minus_blocks, dim=0)

    return _RING_SPIN_CONCAT_CACHE.get_or_build(
        key, build, estimated_bytes=bytes_needed
    )


def _ring_fourier_modes(
    rows: Tensor,
    *,
    starts_cpu: Tensor,
    lengths_cpu: Tensor,
    mmax: int,
    nside: int | None = None,
) -> Tensor:
    # rows: [nrows, npix] complex tensor.
    device = rows.device
    dtype = rows.dtype
    nrings = int(lengths_cpu.numel())

    use_cpp = (
        (_RING_FOURIER_MODES_CPP_ENABLE or _RING_FOURIER_CPP_ENABLE)
        and _native_ops.has_spectral_dispatcher()
        and device.type == "cpu"
        and int(mmax) <= _RING_FOURIER_MODES_CPP_MAX_M
        and (not rows.requires_grad)
    )
    if use_cpp:
        return _native_ops.spectral_ring_fourier_modes(
            rows.contiguous(),
            starts_cpu.contiguous(),
            lengths_cpu.contiguous(),
            int(mmax),
        )

    if nside is not None:
        common_block, groups = _ring_grouping_for_nside(int(nside))
    else:
        common_block, groups = _build_ring_grouping(starts_cpu, lengths_cpu)

    out = torch.zeros(
        (int(rows.shape[0]), int(mmax) + 1, nrings), dtype=dtype, device=device
    )
    m_int = torch.arange(int(mmax) + 1, dtype=torch.int64, device=device)

    if common_block is not None:
        common_len, start_ring, end_ring, pix_start, pix_end = common_block
        n_common = end_ring - start_ring + 1
        belt_pixels = rows[:, pix_start:pix_end]
        belt_stack = belt_pixels.view(rows.shape[0], n_common, common_len)
        if int(common_len) <= _RING_SMALL_DFT_MAX_NPH:
            if int(common_len) > int(mmax):
                phase = _ring_azimuth_phase(int(common_len), int(mmax)).to(
                    device=device, dtype=dtype
                )
                dft_sel = torch.matmul(belt_stack, torch.conj(phase).transpose(0, 1))
            else:
                phase_base = _ring_azimuth_phase(
                    int(common_len), int(common_len) - 1
                ).to(device=device, dtype=dtype)
                dft_base = torch.matmul(
                    belt_stack, torch.conj(phase_base).transpose(0, 1)
                )
                col_idx = torch.remainder(m_int, int(common_len))
                dft_sel = torch.index_select(dft_base, 2, col_idx)
            out[:, :, start_ring : end_ring + 1] = dft_sel.permute(0, 2, 1)
        else:
            fft_block = torch.fft.fft(belt_stack, dim=-1)
            if int(common_len) > int(mmax):
                # No aliasing when nph > mmax: use direct contiguous frequency slice.
                fft_sel = fft_block[:, :, : int(mmax) + 1]
            else:
                col_idx = torch.remainder(m_int, int(common_len))
                fft_sel = torch.index_select(fft_block, 2, col_idx)
            out[:, :, start_ring : end_ring + 1] = fft_sel.permute(0, 2, 1)

    for nph, ring_idx_cpu, pix_idx_cpu in groups:
        ring_idx = (
            ring_idx_cpu if device.type == "cpu" else ring_idx_cpu.to(device=device)
        )
        pix_idx = pix_idx_cpu if device.type == "cpu" else pix_idx_cpu.to(device=device)
        ring_stack = rows[:, pix_idx]
        if int(nph) <= _RING_SMALL_DFT_MAX_NPH:
            if int(nph) > int(mmax):
                phase = _ring_azimuth_phase(int(nph), int(mmax)).to(
                    device=device, dtype=dtype
                )
                dft_sel = torch.matmul(ring_stack, torch.conj(phase).transpose(0, 1))
            else:
                phase_base = _ring_azimuth_phase(int(nph), int(nph) - 1).to(
                    device=device, dtype=dtype
                )
                dft_base = torch.matmul(
                    ring_stack, torch.conj(phase_base).transpose(0, 1)
                )
                dft_sel = torch.index_select(
                    dft_base, 2, torch.remainder(m_int, int(nph))
                )
            out.index_copy_(2, ring_idx, dft_sel.permute(0, 2, 1))
        else:
            fft_block = torch.fft.fft(ring_stack, dim=-1)
            if int(nph) > int(mmax):
                fft_sel = fft_block[:, :, : int(mmax) + 1]
            else:
                fft_sel = torch.index_select(
                    fft_block, 2, torch.remainder(m_int, int(nph))
                )
            out.index_copy_(2, ring_idx, fft_sel.permute(0, 2, 1))

    return out


def _ring_fourier_modes_spin_conj(
    p_plus: Tensor,
    *,
    starts_cpu: Tensor,
    lengths_cpu: Tensor,
    mmax: int,
    nside: int | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Specialized spin helper for p_minus = conj(p_plus).

    Returns (s_plus, s_minus) with shape (mmax+1, nrings), where
    s_plus[m, r]  = FFT(p_plus_ring_r)[m mod nph_r]
    s_minus[m, r] = FFT(conj(p_plus_ring_r))[m mod nph_r]
                  = conj(FFT(p_plus_ring_r)[(-m) mod nph_r]).
    """
    if p_plus.ndim != 1:
        raise ValueError("p_plus must be a 1D complex tensor")
    device = p_plus.device
    dtype = p_plus.dtype
    nrings = int(lengths_cpu.numel())

    use_cpp = (
        (_RING_FOURIER_MODES_CPP_ENABLE or _RING_FOURIER_CPP_ENABLE)
        and _native_ops.has_spectral_dispatcher()
        and device.type == "cpu"
        and int(mmax) <= _RING_FOURIER_MODES_CPP_MAX_M
        and (not p_plus.requires_grad)
    )
    if use_cpp:
        out_cpp = _native_ops.spectral_ring_fourier_modes_spin_conj(
            p_plus.contiguous(),
            starts_cpu.contiguous(),
            lengths_cpu.contiguous(),
            int(mmax),
        )
        return out_cpp

    if nside is not None:
        common_block, groups = _ring_grouping_for_nside(int(nside))
    else:
        common_block, groups = _build_ring_grouping(starts_cpu, lengths_cpu)

    s_plus = torch.zeros((int(mmax) + 1, nrings), dtype=dtype, device=device)
    s_minus = torch.zeros_like(s_plus)
    k = int(mmax) + 1

    if common_block is not None:
        common_len, start_ring, end_ring, pix_start, pix_end = common_block
        n_common = end_ring - start_ring + 1
        belt_pixels = p_plus[pix_start:pix_end]
        belt_stack = belt_pixels.view(n_common, common_len)
        fft_block = torch.fft.fft(belt_stack, dim=-1)
        idx_plus_cpu, idx_minus_cpu, idx_both_cpu = _ring_spin_conj_indices(
            int(common_len), int(mmax)
        )
        if device.type == "cpu":
            idx_minus = idx_minus_cpu
            idx_both = idx_both_cpu
        else:
            idx_minus = idx_minus_cpu.to(device=device)
            idx_both = idx_both_cpu.to(device=device)
        if int(common_len) > int(mmax):
            plus_sel = fft_block[:, :k]
            minus_sel = torch.conj(torch.index_select(fft_block, 1, idx_minus))
        else:
            both_sel = torch.index_select(fft_block, 1, idx_both)
            plus_sel = both_sel[:, :k]
            minus_sel = torch.conj(both_sel[:, k:])
        s_plus[:, start_ring : end_ring + 1] = plus_sel.transpose(0, 1)
        s_minus[:, start_ring : end_ring + 1] = minus_sel.transpose(0, 1)

    n_group_rings = 0
    for _, ring_idx_cpu, _ in groups:
        n_group_rings += int(ring_idx_cpu.numel())
    if n_group_rings > 0:
        plus_all = torch.empty((k, n_group_rings), dtype=dtype, device=device)
        minus_all = torch.empty_like(plus_all)
        if device.type == "cpu":
            ring_idx_all = torch.empty((n_group_rings,), dtype=torch.int64)
        else:
            ring_idx_all = torch.empty(
                (n_group_rings,), dtype=torch.int64, device=device
            )
    else:
        plus_all = None
        minus_all = None
        ring_idx_all = None
    write_off = 0

    for nph, ring_idx_cpu, pix_idx_cpu in groups:
        ring_idx = (
            ring_idx_cpu if device.type == "cpu" else ring_idx_cpu.to(device=device)
        )
        pix_idx = pix_idx_cpu if device.type == "cpu" else pix_idx_cpu.to(device=device)
        ring_stack = p_plus[pix_idx]
        fft_block = torch.fft.fft(ring_stack, dim=-1)
        idx_plus_cpu, idx_minus_cpu, idx_both_cpu = _ring_spin_conj_indices(
            int(nph), int(mmax)
        )
        if device.type == "cpu":
            idx_minus = idx_minus_cpu
            idx_both = idx_both_cpu
        else:
            idx_minus = idx_minus_cpu.to(device=device)
            idx_both = idx_both_cpu.to(device=device)
        if int(nph) > int(mmax):
            plus_sel = fft_block[:, :k]
            minus_sel = torch.conj(torch.index_select(fft_block, 1, idx_minus))
        else:
            both_sel = torch.index_select(fft_block, 1, idx_both)
            plus_sel = both_sel[:, :k]
            minus_sel = torch.conj(both_sel[:, k:])
        ng = int(ring_idx.numel())
        if (
            ng > 0
            and plus_all is not None
            and minus_all is not None
            and ring_idx_all is not None
        ):
            plus_all[:, write_off : write_off + ng] = plus_sel.transpose(0, 1)
            minus_all[:, write_off : write_off + ng] = minus_sel.transpose(0, 1)
            ring_idx_all[write_off : write_off + ng] = ring_idx
            write_off += ng

    if (
        n_group_rings > 0
        and plus_all is not None
        and minus_all is not None
        and ring_idx_all is not None
    ):
        s_plus.index_copy_(1, ring_idx_all, plus_all)
        s_minus.index_copy_(1, ring_idx_all, minus_all)

    return s_plus, s_minus


_M_PER_ALM_CACHE: SpectralCacheNamespace[Tensor] = _SPECTRAL_CACHE.namespace(
    "m_per_alm", device="cpu", dtype=torch.int64
)


def _m_per_alm_cached(lmax: int, mmax: int) -> Tensor:
    """Return (nalm,) int64 tensor where entry j holds the m-index for alm j."""
    key = (int(lmax), int(mmax))

    def build() -> Tensor:
        parts = [
            torch.full((lmax - m + 1,), m, dtype=torch.int64)
            for m in range(mmax + 1)
        ]
        return torch.cat(parts)

    return _M_PER_ALM_CACHE.get_or_build(key, build)


def _scalar_alm2map_ring_torch(
    rows: Tensor,
    *,
    nside: int,
    lmax: int,
    mmax: int,
) -> Tensor:
    starts_cpu, lengths_cpu, _, _ = _ring_layout_for_nside(int(nside))
    device = rows.device
    dtype_real = _preferred_real_dtype(device)
    dtype_cplx = _preferred_complex_dtype(device)

    nmaps = int(rows.shape[0])
    nrings = int(lengths_cpu.numel())

    use_cpp_recurrence = (
        _SCALAR_INTERP_CPP_ENABLE
        and _native_ops.has_spectral_dispatcher()
        and device.type == "cpu"
    )

    if use_cpp_recurrence:
        out_s_m = []
        for i in range(nmaps):
            s_modes = _scalar_ring_interpolate_recurrence_cpu(
                rows[i],
                nside=int(nside),
                lmax=int(lmax),
                mmax=int(mmax),
            )
            out_s_m.append(s_modes)
        s_modes_batch = torch.stack(out_s_m, dim=0)

        phase0_pos = _ring_phi0_phase_for_nside(int(nside), int(mmax), sign=1).to(
            dtype=torch.complex128, device=s_modes_batch.device
        )
        c_all = s_modes_batch * phase0_pos.unsqueeze(0)

        if mmax > 0:
            c_all = c_all.clone()
            c_all[:, 1:, :] = 2.0 * c_all[:, 1:, :]
    else:
        # Batched Legendre interpolation (no Python m-loop)
        Y = _ring_scalar_basis_concat(int(nside), int(lmax), int(mmax)).to(
            device=device, dtype=dtype_cplx
        )
        m_idx = _m_per_alm_cached(int(lmax), int(mmax)).to(device=device)

        phi0_phase = _ring_phi0_phase_for_nside(int(nside), int(mmax), sign=1).to(
            device=device, dtype=dtype_cplx
        )

        # alm_weighted: (nmaps, nalm, nrings) = alm[:, j] * Y[j, r]
        alm_cplx = rows.to(dtype=dtype_cplx)
        alm_weighted = alm_cplx.unsqueeze(-1) * Y.unsqueeze(0)  # broadcast

        # Scatter-sum into per-m Fourier modes using m_idx
        f_m_ring = torch.zeros(
            (nmaps, mmax + 1, nrings), dtype=dtype_cplx, device=device
        )
        m_expand = m_idx.unsqueeze(0).unsqueeze(-1).expand_as(alm_weighted)
        f_m_ring = _accumulate_add(f_m_ring, 1, m_expand, alm_weighted, scatter=True)

        # Apply phi0 phase per m
        f_m_ring = f_m_ring * phi0_phase.unsqueeze(0)

        c_all = f_m_ring
        if mmax > 0:
            c_all = c_all.clone()
            c_all[:, 1:, :] = 2.0 * c_all[:, 1:, :]

    out = _ring_fourier_synthesis(
        c_all,
        starts_cpu=starts_cpu,
        lengths_cpu=lengths_cpu,
        nside=int(nside),
    )
    if out.is_complex():
        out = out.real
    return out.to(dtype=dtype_real)


def _scalar_map2alm_ring_torch(
    rows: Tensor,
    *,
    nside: int,
    lmax: int,
    mmax: int,
) -> Tensor:
    starts_cpu, lengths_cpu, _, _ = _ring_layout_for_nside(int(nside))
    device = rows.device
    dtype_cplx = _preferred_complex_dtype(device)

    nmaps = int(rows.shape[0])
    npix = int(rows.shape[1])

    pix_w = (4.0 * math.pi) / float(npix)
    s_all = _ring_fourier_modes(
        rows.to(dtype=dtype_cplx),
        starts_cpu=starts_cpu,
        lengths_cpu=lengths_cpu,
        mmax=mmax,
        nside=int(nside),
    )

    phase0_neg = _ring_phi0_phase_for_nside(int(nside), int(mmax), sign=-1).to(
        device=device, dtype=dtype_cplx
    )
    s_all = s_all * phase0_neg.unsqueeze(0)

    use_cpp_recurrence = (
        _SCALAR_RECURRENCE_CPP_ENABLE
        and _native_ops.has_spectral_dispatcher()
        and device.type == "cpu"
    )

    if use_cpp_recurrence:
        out_alms = []
        for i in range(nmaps):
            alm = _scalar_ring_integrate_recurrence_cpu(
                s_all[i],
                nside=int(nside),
                lmax=int(lmax),
                mmax=int(mmax),
                pix_w=pix_w,
            )
            out_alms.append(alm)
        return torch.stack(out_alms, dim=0)

    # Batched Legendre integration (no Python m-loop)
    Y = _ring_scalar_basis_concat(int(nside), int(lmax), int(mmax)).to(
        device=device, dtype=dtype_cplx
    )
    m_idx = _m_per_alm_cached(int(lmax), int(mmax)).to(device=device)

    # Expand s_all from (nmaps, mmax+1, nrings) to (nmaps, nalm, nrings)
    # using m_idx to select the correct m for each alm
    s_expanded = torch.index_select(s_all, 1, m_idx)  # (nmaps, nalm, nrings)

    # Single batched einsum: alm[b, j] = sum_r(s_expanded[b, j, r] * conj(Y[j, r])) * pix_w
    out = torch.einsum("bnr,nr->bn", s_expanded, Y) * pix_w

    return out


def _ring_fourier_synthesis(
    s_m: Tensor,
    *,
    starts_cpu: Tensor,
    lengths_cpu: Tensor,
    nside: int | None = None,
) -> Tensor:
    """
    Evaluate ring Fourier series from m>=0 coefficients.
    s_m: (mmax+1, nrings), complex128
    Returns: (npix,), complex128
    """
    if s_m.ndim == 2:
        s_m_batched = s_m.unsqueeze(0)
        squeeze_out = True
    elif s_m.ndim == 3:
        s_m_batched = s_m
        squeeze_out = False
    else:
        raise ValueError(
            "s_m must have shape (mmax+1, nrings) or (nrows, mmax+1, nrings)"
        )

    mmax = int(s_m_batched.shape[1]) - 1
    npix = int(starts_cpu[-1].item()) + int(lengths_cpu[-1].item())
    device = s_m_batched.device
    dtype = s_m_batched.dtype
    nrows = int(s_m_batched.shape[0])

    use_cpp = (
        (_RING_FOURIER_SYNTH_CPP_ENABLE or _RING_FOURIER_CPP_ENABLE)
        and _native_ops.has_spectral_dispatcher()
        and device.type == "cpu"
        and int(mmax) <= _RING_FOURIER_SYNTH_CPP_MAX_M
        and (not s_m_batched.requires_grad)
    )
    if use_cpp:
        out_cpp = _native_ops.spectral_ring_fourier_synthesis(
            s_m_batched.contiguous(),
            starts_cpu.contiguous(),
            lengths_cpu.contiguous(),
            npix,
        )
        if squeeze_out:
            return out_cpp[0]
        return out_cpp

    if nside is not None:
        common_block, groups = _ring_grouping_for_nside(int(nside))
    else:
        common_block, groups = _build_ring_grouping(starts_cpu, lengths_cpu)

    out_map = torch.zeros((nrows, npix), dtype=dtype, device=device)

    if common_block is not None:
        common_len, start_ring, end_ring, pix_start, pix_end = common_block
        s_group = s_m_batched[:, :, start_ring : end_ring + 1]
        s_group_perm = s_group.permute(0, 2, 1)
        if int(common_len) <= _RING_SMALL_DFT_MAX_NPH:
            phase = _ring_azimuth_phase(int(common_len), int(mmax)).to(
                device=device, dtype=dtype
            )
            map_vals = torch.matmul(s_group_perm, phase)
        else:
            if int(common_len) > int(mmax):
                map_vals = torch.fft.ifft(
                    s_group_perm * float(common_len), n=int(common_len), dim=-1
                )
            else:
                s_fold = torch.zeros(
                    (s_group_perm.shape[0], s_group_perm.shape[1], int(common_len)),
                    dtype=dtype,
                    device=device,
                )
                alias_idx = _ring_alias_indices(int(common_len), int(mmax))
                alias_idx = (
                    alias_idx if device.type == "cpu" else alias_idx.to(device=device)
                )
                s_fold = _accumulate_add(
                    s_fold, 2, alias_idx, s_group_perm, scatter=False
                )
                map_vals = torch.fft.ifft(s_fold * float(common_len), dim=-1)
        out_map[:, pix_start:pix_end] = map_vals.reshape(nrows, -1)

    for nph, ring_idx_cpu, pix_idx_cpu in groups:
        ring_idx = (
            ring_idx_cpu if device.type == "cpu" else ring_idx_cpu.to(device=device)
        )
        pix_idx = pix_idx_cpu if device.type == "cpu" else pix_idx_cpu.to(device=device)
        s_group = torch.index_select(s_m_batched, 2, ring_idx)
        s_group_perm = s_group.permute(0, 2, 1)
        if int(nph) <= _RING_SMALL_DFT_MAX_NPH:
            phase = _ring_azimuth_phase(int(nph), int(mmax)).to(
                device=device, dtype=dtype
            )
            map_vals = torch.matmul(s_group_perm, phase)
        else:
            if int(nph) > int(mmax):
                map_vals = torch.fft.ifft(s_group_perm * float(nph), n=int(nph), dim=-1)
            else:
                s_fold = torch.zeros(
                    (s_group_perm.shape[0], s_group_perm.shape[1], int(nph)),
                    dtype=dtype,
                    device=device,
                )
                alias_idx = _ring_alias_indices(int(nph), int(mmax))
                alias_idx = (
                    alias_idx if device.type == "cpu" else alias_idx.to(device=device)
                )
                s_fold = _accumulate_add(
                    s_fold, 2, alias_idx, s_group_perm, scatter=False
                )
                map_vals = torch.fft.ifft(s_fold * float(nph), dim=-1)
        out_map[:, pix_idx] = map_vals

    if squeeze_out:
        return out_map[0]
    return out_map


def _ring_pix2ring_index(nside: int) -> Tensor:
    key = int(nside)

    def build() -> Tensor:
        _, lengths_cpu, _, _ = _ring_layout_for_nside(key)
        nrings = int(lengths_cpu.numel())
        ring_ids = torch.arange(nrings, dtype=torch.int64)
        return torch.repeat_interleave(ring_ids, lengths_cpu)

    return _RING_PIX2RING_CACHE.get_or_build(key, build)


def _ring_expand_m0(s0: Tensor, *, nside: int) -> Tensor:
    """
    Expand per-ring m=0 values to per-pixel map.

    s0: (nrings,) complex
    returns: (npix,) complex
    """
    ring_idx_cpu = _ring_pix2ring_index(int(nside))
    ring_idx = (
        ring_idx_cpu if s0.device.type == "cpu" else ring_idx_cpu.to(device=s0.device)
    )
    if s0.ndim == 1:
        return torch.index_select(s0, 0, ring_idx)
    if s0.ndim == 2:
        return torch.index_select(s0, 1, ring_idx)
    raise ValueError("s0 must have shape (nrings,) or (nrows, nrings)")


def _ring_spin_interpolate_blocks(
    coeff_plus: Tensor,
    coeff_minus: Tensor,
    *,
    nside: int,
    lmax: int,
    mmax: int,
    spin: int,
) -> tuple[Tensor, Tensor]:
    nrings = (4 * int(nside)) - 1
    nalm = alm_size(int(lmax), int(mmax))
    bytes_concat = int(nalm * nrings * 32)  # two complex128 concat bases
    use_concat_fast = bytes_concat <= _SPIN_RING_INTERP_CONCAT_FAST_MAX_BYTES
    if use_concat_fast:
        y_plus, y_minus = _ring_spin_basis_concat(
            int(nside), int(lmax), int(mmax), int(spin)
        )
        device = coeff_plus.device
        dtype = coeff_plus.dtype
        if y_plus.numel() == 0:
            empty = torch.zeros((int(mmax) + 1, 0), dtype=dtype, device=device)
            return empty, empty
        y_plus = y_plus.to(device=device, dtype=dtype)
        y_minus = y_minus.to(device=device, dtype=dtype)
        use_cpp = (
            os.environ.get("TORCHSKY_SPIN_RING_INTERP_CONCAT_CPP", "1") != "0"
            and _native_ops.has_spectral_dispatcher()
            and device.type == "cpu"
            and dtype == torch.complex128
            and (not coeff_plus.requires_grad)
            and (not coeff_minus.requires_grad)
        )
        if use_cpp:
            s_plus_cpp, s_minus_cpp = _native_ops.spectral_spin_interpolate_concat(
                coeff_plus.contiguous(),
                coeff_minus.contiguous(),
                y_plus.contiguous(),
                y_minus.contiguous(),
                int(lmax),
                int(mmax),
            )
            return s_plus_cpp, s_minus_cpp
        m_idx = _alm_m_array(int(lmax), int(mmax))
        m_idx = m_idx if device.type == "cpu" else m_idx.to(device=device)
        term_plus = coeff_plus.unsqueeze(1) * y_plus
        term_minus = coeff_minus.unsqueeze(1) * y_minus
        s_plus = torch.zeros(
            (int(mmax) + 1, y_plus.shape[1]), dtype=dtype, device=device
        )
        s_minus = torch.zeros_like(s_plus)
        s_plus = _accumulate_add(s_plus, 0, m_idx, term_plus, scatter=False)
        s_minus = _accumulate_add(s_minus, 0, m_idx, term_minus, scatter=False)
        return s_plus, s_minus

    y_plus_blocks = _ring_spin_bases_for_nside(
        int(nside), int(lmax), int(mmax), int(spin)
    )
    y_minus_blocks = _ring_spin_bases_for_nside(
        int(nside), int(lmax), int(mmax), -int(spin)
    )
    nrings = int(y_plus_blocks[0].shape[1]) if y_plus_blocks else 0
    device = coeff_plus.device
    dtype = coeff_plus.dtype
    s_plus = torch.zeros((int(mmax) + 1, nrings), dtype=dtype, device=device)
    s_minus = torch.zeros_like(s_plus)
    idx = 0
    for m in range(int(mmax) + 1):
        l_count = int(lmax) - m + 1
        y_plus = y_plus_blocks[m].to(device=device, dtype=dtype)
        y_minus = y_minus_blocks[m].to(device=device, dtype=dtype)
        c_plus_m = coeff_plus[idx : idx + l_count]
        c_minus_m = coeff_minus[idx : idx + l_count]
        s_plus[m] = c_plus_m @ y_plus
        s_minus[m] = c_minus_m @ y_minus
        idx += l_count
    return s_plus, s_minus


def _ring_spin_integrate_blocks(
    s_plus: Tensor,
    s_minus: Tensor,
    *,
    nside: int,
    lmax: int,
    mmax: int,
    spin: int,
    pix_w: float,
) -> tuple[Tensor, Tensor]:
    nrings = (4 * int(nside)) - 1
    nalm = alm_size(int(lmax), int(mmax))
    bytes_concat = int(nalm * nrings * 32)  # two complex128 concat bases
    use_concat_fast = (
        bytes_concat <= _SPIN_RING_INTEGRATE_CONCAT_FAST_MAX_BYTES
        and nalm <= _SPIN_RING_INTEGRATE_CONCAT_MAX_NALM
    )
    if use_concat_fast:
        y_plus, y_minus = _ring_spin_basis_concat(
            int(nside), int(lmax), int(mmax), int(spin)
        )
        device = s_plus.device
        dtype = s_plus.dtype
        if y_plus.numel() == 0:
            empty = torch.zeros((nalm,), dtype=dtype, device=device)
            return empty, empty
        y_plus = y_plus.to(device=device, dtype=dtype)
        y_minus = y_minus.to(device=device, dtype=dtype)
        use_cpp = (
            os.environ.get("TORCHSKY_SPIN_RING_INTEGRATE_CONCAT_CPP", "1") != "0"
            and _native_ops.has_spectral_dispatcher()
            and device.type == "cpu"
            and dtype == torch.complex128
            and (not s_plus.requires_grad)
            and (not s_minus.requires_grad)
        )
        if use_cpp:
            c_plus_cpp, c_minus_cpp = _native_ops.spectral_spin_integrate_concat(
                s_plus.contiguous(),
                s_minus.contiguous(),
                y_plus.contiguous(),
                y_minus.contiguous(),
                int(lmax),
                int(mmax),
                float(pix_w),
            )
            return c_plus_cpp, c_minus_cpp
        m_idx = _alm_m_array(int(lmax), int(mmax))
        m_idx = m_idx if device.type == "cpu" else m_idx.to(device=device)
        s_plus_exp = torch.index_select(s_plus, 0, m_idx)
        s_minus_exp = torch.index_select(s_minus, 0, m_idx)
        c_plus = torch.sum(s_plus_exp * y_plus, dim=1) * pix_w
        c_minus = torch.sum(s_minus_exp * y_minus, dim=1) * pix_w
        return c_plus, c_minus

    y_plus_blocks = _ring_spin_bases_for_nside(
        int(nside), int(lmax), int(mmax), int(spin)
    )
    y_minus_blocks = _ring_spin_bases_for_nside(
        int(nside), int(lmax), int(mmax), -int(spin)
    )
    device = s_plus.device
    dtype = s_plus.dtype
    nalm = alm_size(int(lmax), int(mmax))
    c_plus = torch.empty((nalm,), dtype=dtype, device=device)
    c_minus = torch.empty_like(c_plus)
    idx = 0
    for m in range(int(mmax) + 1):
        l_count = int(lmax) - m + 1
        y_plus = y_plus_blocks[m].to(device=device, dtype=dtype)
        y_minus = y_minus_blocks[m].to(device=device, dtype=dtype)
        c_plus[idx : idx + l_count] = (y_plus @ s_plus[m]) * pix_w
        c_minus[idx : idx + l_count] = (y_minus @ s_minus[m]) * pix_w
        idx += l_count
    return c_plus, c_minus


def _spin_alm2map_ring_torch(
    coeff_plus: Tensor,
    coeff_minus: Tensor,
    *,
    nside: int,
    lmax: int,
    mmax: int,
    spin: int,
) -> Tensor:
    """
    Helper for ring-based alm2map_spin.
    """
    use_cpp = (
        _HAS_HEALPIX_SPIN_INTERPOLATE_REC
        and os.environ.get("TORCHSKY_SPIN_ALM2MAP_RECURRENCE_CPP", "1") != "0"
        and coeff_plus.device.type == "cpu"
        and coeff_minus.device.type == "cpu"
    )

    if use_cpp:
        starts_cpu, lengths_cpu, theta_ring_cpu, _ = _ring_layout_for_nside(nside)
        fused_rec = (
            _HAS_HEALPIX_SPIN_ALM2MAP_RING_REC
            and coeff_plus.device.type == "cpu"
            and (not coeff_plus.requires_grad)
            and (not coeff_minus.requires_grad)
        )
        if fused_rec:
            cp_cpu = (
                coeff_plus
                if coeff_plus.dtype == torch.complex128
                else coeff_plus.to(dtype=torch.complex128)
            )
            cm_cpu = (
                coeff_minus
                if coeff_minus.dtype == torch.complex128
                else coeff_minus.to(dtype=torch.complex128)
            )
            phase0_pos = _ring_phi0_phase_for_nside(int(nside), int(mmax), sign=1)
            return _native_ops.spectral_spin_alm2map_ring_recurrence(
                cp_cpu,
                starts_cpu,
                lengths_cpu,
                theta_ring_cpu,
                phase0_pos,
                int(lmax),
                int(mmax),
                int(spin),
                cm_cpu,
                _healpix.nside2npix(int(nside)),
            )
        if coeff_plus.requires_grad or coeff_minus.requires_grad:
            s_plus, s_minus = SpinSHTInterpolate.apply(
                coeff_plus, coeff_minus, int(nside), int(lmax), int(mmax), int(spin)
            )
        else:
            # Fast path: call C++ directly and avoid stack/detach overhead
            _, _, theta_ring_cpu, _ = _ring_layout_for_nside(nside)
            # We still need complex128 CPU tensors for the current C++ impl
            cp_cpu = coeff_plus.to(dtype=torch.complex128, device="cpu").contiguous()
            cm_cpu = coeff_minus.to(dtype=torch.complex128, device="cpu").contiguous()
            alm_stack = torch.stack([cp_cpu, cm_cpu], dim=0)

            s_modes_stacked = _native_ops.spectral_spin_interpolate_recurrence(
                alm_stack,
                theta_ring_cpu.contiguous(),
                int(lmax),
                int(mmax),
                int(spin),
            )
            s_plus = s_modes_stacked[0].to(device=coeff_plus.device)
            s_minus = s_modes_stacked[1].to(device=coeff_minus.device)
    else:
        s_plus, s_minus = _ring_spin_interpolate_blocks(
            coeff_plus,
            coeff_minus,
            nside=int(nside),
            lmax=int(lmax),
            mmax=int(mmax),
            spin=int(spin),
        )

    # 2. Apply phase factors and inverse FFT to get ring values
    phase0_pos = _ring_phi0_phase_for_nside(int(nside), int(mmax), sign=1).to(
        device=s_plus.device, dtype=s_plus.dtype
    )
    s_plus = s_plus * phase0_pos
    s_minus = s_minus * phase0_pos

    # 3. Inverse FFT (synthesis) + final spin combination
    start_pix, lengths, _, _ = _ring_layout_for_nside(int(nside))
    use_cpp_finalize = (
        os.environ.get("TORCHSKY_SPIN_RING_FINALIZE_CPP", "1") != "0"
        and _native_ops.has_spectral_dispatcher()
        and s_plus.device.type == "cpu"
        and (not s_plus.requires_grad)
        and (not s_minus.requires_grad)
    )
    if use_cpp_finalize:
        return _native_ops.spectral_spin_ring_finalize(
            s_plus.contiguous(),
            s_minus.contiguous(),
            start_pix.contiguous(),
            lengths.contiguous(),
            _healpix.nside2npix(int(nside)),
        )

    p_pm_pos = _ring_fourier_synthesis(
        torch.stack([s_plus, s_minus], dim=0),
        starts_cpu=start_pix,
        lengths_cpu=lengths,
        nside=int(nside),
    )
    p_plus_pos = p_pm_pos[0]
    p_minus_pos = p_pm_pos[1]
    p_pm_m0 = _ring_expand_m0(
        torch.stack([s_plus[0], s_minus[0]], dim=0), nside=int(nside)
    )
    p_plus_m0 = p_pm_m0[0]
    p_minus_m0 = p_pm_m0[1]

    if mmax == 0:
        p_plus = p_plus_pos
        p_minus = p_minus_pos
    else:
        p_plus = p_plus_pos + torch.conj(p_minus_pos - p_minus_m0)
        p_minus = p_minus_pos + torch.conj(p_plus_pos - p_plus_m0)

    q = 0.5 * (p_plus + p_minus)
    u = -0.5j * (p_plus - p_minus)
    real_dtype = _preferred_real_dtype(q.device)
    return torch.stack([(-q).real.to(real_dtype), (-u).real.to(real_dtype)], dim=0)


def _cross_cl_from_alm_pair(a1: Tensor, a2: Tensor, lmax: int, mmax: int) -> Tensor:
    ell = _alm_ell_array(lmax, mmax).to(a1.device)
    m = _alm_m_array(lmax, mmax).to(a1.device)
    weights = torch.where(
        m == 0,
        torch.ones_like(m, dtype=torch.float64),
        torch.full_like(m, 2.0, dtype=torch.float64),
    )
    prod = torch.real(a1 * torch.conj(a2)).to(torch.float64) * weights
    cl = torch.zeros(lmax + 1, dtype=torch.float64, device=a1.device)
    cl.scatter_add_(0, ell, prod)
    norm = (2.0 * torch.arange(lmax + 1, dtype=torch.float64, device=a1.device)) + 1.0
    return cl / norm


def map2alm(
    map_values: Tensor | list[float] | list[list[float]],
    *,
    nside: int | None = None,
    lmax: int | None = None,
    mmax: int | None = None,
    nest: bool = False,
    iter: int = 0,
    pol: bool = False,
    use_weights: bool = False,
    use_pixel_weights: bool = False,
    gal_cut: float = 0.0,
    datapath: str | os.PathLike[str] | None = None,
) -> Tensor:
    """
    Compute alm coefficients in healpy ordering (m>=0 only).

    With `pol=True`, input map must be shape `(3, npix)` and return is shape
    `(3, nalm)` in `(T,E,B)` ordering.

    ``use_weights`` applies computed RING quadrature (mean 1).
    ``use_pixel_weights`` loads HEALPix full-pixel weight FITS via torchfits
    from ``datapath`` or ``TORCHSKY_HEALPIX_DATAPATH``
    (``full_weights/healpix_full_weights_nside_NNNN.fits``) and multiplies by
    ``1 + w``. Missing files raise; nothing is downloaded.
    ``gal_cut`` zeros pixels whose RING-scheme latitude is below ``gal_cut``
    degrees (healpy convention).
    CPU uses fused Ishioka/Wigner C++. CUDA/MPS keep maps on device (ATen
    ring FFT); they are not host-copied.
    """
    rows, single = _map_to_rows(map_values)
    npix = int(rows.shape[1])
    ns = _healpix.npix2nside(npix) if nside is None else int(nside)
    if _healpix.nside2npix(ns) != npix:
        raise ValueError("map length does not match nside")

    ll = int(3 * ns - 1 if lmax is None else lmax)
    mm = ll if mmax is None else int(mmax)
    if ll < 0 or mm < 0 or mm > ll:
        raise ValueError("invalid lmax/mmax")

    if nest:
        # Reorder to RING for memory stability on torch backend
        rows = _reorder_nest_to_ring_rows(rows, ns)
        nest = False

    analysis_rows = _apply_map2alm_weights(
        rows,
        nside=ns,
        gal_cut=gal_cut,
        use_weights=use_weights,
        use_pixel_weights=use_pixel_weights,
        datapath=datapath,
    )

    # Iterative solver (Matches healpy behavior)
    alm = _map2alm_single_pass(
        analysis_rows, nside=ns, lmax=ll, mmax=mm, pol=pol, use_pixel_weights=False
    )
    for _ in range(iter):
        map_rec = alm2map(alm, nside=ns, lmax=ll, mmax=mm, nest=False, pol=pol)
        res = rows - _map_to_rows(map_rec)[0]
        analysis_res = _apply_map2alm_weights(
            res,
            nside=ns,
            gal_cut=gal_cut,
            use_weights=use_weights,
            use_pixel_weights=use_pixel_weights,
            datapath=datapath,
        )
        alm = alm + _map2alm_single_pass(
            analysis_res,
            nside=ns,
            lmax=ll,
            mmax=mm,
            pol=pol,
            use_pixel_weights=False,
        )

    if single and not pol:
        return alm[0]
    return alm


def map2alm_batched(
    maps: Tensor,
    *,
    nside: int | None = None,
    lmax: int | None = None,
    mmax: int | None = None,
    nest: bool = False,
    iter: int = 0,
    pol: bool = False,
    use_weights: bool = False,
    use_pixel_weights: bool = False,
    gal_cut: float = 0.0,
    datapath: str | os.PathLike[str] | None = None,
) -> Tensor:
    """
    Batched harmonic analysis with explicit leading batch dimension.

    Scalar maps must be shaped ``(B, npix)`` and return ``(B, nalm)``.
    With ``pol=True``, maps must be ``(B, 3, npix)`` in ``(T,E,B)`` order per batch
    row and the result is ``(B, 3, nalm)``.
    """
    maps_t = torch.as_tensor(maps)
    if pol:
        if maps_t.ndim != 3 or int(maps_t.shape[1]) != 3:
            raise ValueError(
                "map2alm_batched with pol=True expects shape (B, 3, npix) in (T,E,B) order"
            )
        outs: list[Tensor] = []
        for b in range(int(maps_t.shape[0])):
            outs.append(
                map2alm(
                    maps_t[b],
                    nside=nside,
                    lmax=lmax,
                    mmax=mmax,
                    nest=nest,
                    iter=iter,
                    pol=True,
                    use_weights=use_weights,
                    use_pixel_weights=use_pixel_weights,
                    gal_cut=gal_cut,
                    datapath=datapath,
                )
            )
        return torch.stack(outs, dim=0)
    if maps_t.ndim != 2:
        raise ValueError("map2alm_batched with pol=False expects shape (B, npix)")
    return map2alm(
        maps_t,
        nside=nside,
        lmax=lmax,
        mmax=mmax,
        nest=nest,
        iter=iter,
        pol=False,
        use_weights=use_weights,
        use_pixel_weights=use_pixel_weights,
        gal_cut=gal_cut,
        datapath=datapath,
    )


def _map2alm_single_pass(
    rows: Tensor,
    *,
    nside: int,
    lmax: int,
    mmax: int,
    pol: bool,
    use_pixel_weights: bool,
) -> Tensor:
    if pol:
        a_t = _map2alm_single_pass(
            rows[0:1],
            nside=nside,
            lmax=lmax,
            mmax=mmax,
            pol=False,
            use_pixel_weights=use_pixel_weights,
        )
        a_e, a_b = map2alm_spin(
            rows[1:3],
            spin=2,
            nside=nside,
            lmax=lmax,
            mmax=mmax,
            nest=False,
            iter=0,
        )
        return torch.stack([a_t[0], a_e, a_b], dim=0)

    npix = _healpix.nside2npix(nside)
    nalm = alm_size(lmax, mmax)
    bytes_needed = int(npix * nalm * 16)
    ring_mode = os.environ.get("TORCHSKY_SCALAR_MAP2ALM_RING_TORCH", "auto")
    auto_ring = ring_mode == "auto" and bytes_needed >= _SCALAR_RING_AUTO_MIN_BYTES
    use_ring_torch = ring_mode == "force" or ring_mode == "1" or auto_ring

    if use_ring_torch:
        alm_ring = _scalar_map2alm_ring_torch(rows, nside=nside, lmax=lmax, mmax=mmax)
        if torch.isfinite(alm_ring).all():
            return alm_ring
        if ring_mode == "force":
            raise RuntimeError(
                "TORCHSKY_SCALAR_MAP2ALM_RING_TORCH=force produced non-finite values"
            )

    _, _, _, pix_w = _angles_for_nside(nside, nest=False)
    complex_dtype = _preferred_complex_dtype(rows.device)
    map_c = rows.to(dtype=complex_dtype)
    mat = _scalar_map2alm_mat(nside, False, lmax, mmax)
    if mat is not None:
        if mat.device != map_c.device or mat.dtype != complex_dtype:
            mat = mat.to(device=map_c.device, dtype=complex_dtype)
        alm = (map_c @ mat) * pix_w
    else:
        if bytes_needed > 1024 * 1024 * 1024:
            raise RuntimeError(
                f"map2alm failed: basis-based fallback requires {bytes_needed / 1e9:.1f} GB (max 1.0 GB). "
                f"NSIDE={nside}, LMAX={lmax}."
            )
        _, y_conj_blocks = _ylm_basis(nside, False, lmax, mmax)
        alm = torch.zeros(
            (rows.shape[0], nalm), dtype=complex_dtype, device=map_c.device
        )
        for m in range(mmax + 1):
            y_conj = y_conj_blocks[m].to(device=map_c.device, dtype=complex_dtype)
            l_count = lmax - m + 1
            coeffs = (map_c @ y_conj.transpose(0, 1)) * pix_w
            start = alm_index(m, m, lmax, mmax)
            alm[:, start : start + l_count] = coeffs

    return alm


def map2alm_lsq(
    map_values: Tensor | list[float] | list[list[float]],
    lmax: int,
    mmax: int | None = None,
    *,
    nside: int | None = None,
    pol: bool = True,
    tol: float = 1e-10,
    maxiter: int = 20,
    nest: bool = False,
    use_pixel_weights: bool = False,
    gal_cut: float = 0.0,
    datapath: str | os.PathLike[str] | None = None,
) -> tuple[Tensor, float, int]:
    """
    Iterative least-squares map analysis up to (lmax, mmax).

    Returns `(alm, rel_res, n_iter)` where `rel_res` is the residual L2 norm
    divided by the input map L2 norm.
    """
    rows, single = _map_to_rows(map_values)
    npix = int(rows.shape[1])
    ns = _healpix.npix2nside(npix) if nside is None else int(nside)
    if _healpix.nside2npix(ns) != npix:
        raise ValueError("map length does not match nside")

    ll = int(lmax)
    mm = ll if mmax is None else int(mmax)
    if ll < 0 or mm < 0 or mm > ll:
        raise ValueError("invalid lmax/mmax")

    pol_req = bool(pol)
    if pol_req and rows.shape[0] not in (1, 3):
        raise ValueError(
            "Wrong input map (must be a valid healpix map or a sequence of 1 or 3 maps)"
        )
    pol_eff = pol_req and rows.shape[0] == 3

    alm0 = map2alm(
        rows,
        nside=ns,
        lmax=ll,
        mmax=mm,
        nest=nest,
        iter=0,
        pol=pol_eff,
        use_pixel_weights=use_pixel_weights,
        gal_cut=gal_cut,
        datapath=datapath,
    )
    alm_rows, _ = _alm_to_rows(alm0)
    map_norm = float(torch.linalg.norm(rows).item())
    n_iter = 0
    rel_res = float("inf")

    while True:
        rec = alm2map(
            alm_rows,
            nside=ns,
            lmax=ll,
            mmax=mm,
            nest=nest,
            pol=pol_eff,
            pixwin=False,
        )
        rec_rows, _ = _map_to_rows(rec)
        residual = rows - rec_rows
        res_norm = float(torch.linalg.norm(residual).item())
        if map_norm > 0.0:
            rel_res = res_norm / map_norm
        else:
            rel_res = 0.0

        if n_iter >= int(maxiter) or rel_res <= float(tol):
            break

        delta = map2alm(
            residual,
            nside=ns,
            lmax=ll,
            mmax=mm,
            nest=nest,
            iter=0,
            pol=pol_eff,
            use_pixel_weights=use_pixel_weights,
            gal_cut=gal_cut,
            datapath=datapath,
        )
        delta_rows, _ = _alm_to_rows(delta)
        alm_rows = alm_rows + delta_rows
        n_iter += 1

    if single and not pol_eff:
        return alm_rows[0], rel_res, n_iter
    return alm_rows, rel_res, n_iter


def almxfl(
    alm_values: Tensor | list[complex] | list[list[complex]],
    fl: Tensor | list[float],
    *,
    mmax: int | None = None,
    inplace: bool = False,
) -> Tensor:
    """
    Multiply alm coefficient(s) by an ell-dependent transfer function.

    Behaves like healpy's `almxfl`: if `fl` is shorter than `lmax+1`, missing
    multipliers are treated as zero; extra entries are ignored.
    """
    alms_in = torch.as_tensor(alm_values)
    single = alms_in.ndim == 1
    if alms_in.ndim not in (1, 2):
        raise ValueError("alm_values must be shape (nalm,) or (nmaps, nalm)")
    can_inplace = (
        inplace
        and isinstance(alm_values, Tensor)
        and torch.is_complex(alm_values)
        and alm_values.dtype == torch.complex128
        and alms_in.dtype == torch.complex128
    )
    if can_inplace:
        rows = alms_in.unsqueeze(0) if single else alms_in
    else:
        rows = (alms_in.unsqueeze(0) if single else alms_in).to(dtype=torch.complex128)
    nalm = int(rows.shape[-1])
    if mmax is None:
        ll, mm = _infer_lmax_mmax_from_nalm(nalm, None, None)
    else:
        mm = int(mmax)
        ll = _infer_lmax_for_fixed_mmax(nalm, mm)

    fl_t = torch.as_tensor(fl, dtype=torch.float64)
    if fl_t.ndim != 1:
        raise ValueError("fl must be 1D")
    ell = _alm_ell_array(ll, mm).to(rows.device)
    scale = torch.zeros((ll + 1,), dtype=torch.float64, device=rows.device)
    keep = min(int(fl_t.numel()), ll + 1)
    if keep > 0:
        scale[:keep] = fl_t[:keep].to(device=rows.device, dtype=torch.float64)
    gain = scale.index_select(0, ell).to(torch.complex128)

    out = rows if can_inplace else rows.clone()
    out.mul_(gain.unsqueeze(0))
    return out[0] if single else out


def rotate_alm(
    alm_values: Tensor | list[complex] | list[list[complex]],
    psi: float,
    theta: float,
    phi: float,
    *,
    lmax: int | None = None,
    mmax: int | None = None,
) -> Tensor:
    """Rotate scalar alm by zyz Euler angles in radians (healpy convention).

    Returns a new tensor. Modes with ``m<0`` use ``a_{l,-m}=(-1)^m conj(a_{l,m})``.
    ponytail: O(lmax^3) Wigner-d loop; fine for lmax used in tests. Batch the
    recurrence if survey-scale ``lmax`` shows up.
    """
    rows, single = _alm_to_rows(alm_values)
    ll, mm = _infer_lmax_mmax_from_nalm(int(rows.shape[1]), lmax, mmax)
    if mm != ll:
        raise ValueError("rotate_alm requires mmax == lmax")
    device = rows.device
    dtype = rows.dtype
    real_dtype = _preferred_real_dtype(device)
    # healpy rotate_alm zyz: Wigner-d argument is -theta relative to our d^l_{m m'}(beta).
    beta = torch.as_tensor(-float(theta), dtype=real_dtype, device=device)
    psi_t = torch.as_tensor(float(psi), dtype=real_dtype, device=device)
    phi_t = torch.as_tensor(float(phi), dtype=real_dtype, device=device)
    zero = torch.zeros((), dtype=real_dtype, device=device)
    out = torch.zeros_like(rows)
    for ell in range(ll + 1):
        n_m = 2 * ell + 1
        full = torch.zeros((rows.shape[0], n_m), dtype=dtype, device=device)
        for m in range(0, min(ell, mm) + 1):
            idx = alm_index(ell, m, ll, mm)
            coeff = rows[:, idx]
            full[:, ell + m] = coeff
            if m > 0:
                sign = -1.0 if (m & 1) else 1.0
                full[:, ell - m] = sign * torch.conj(coeff)
        rotated = torch.zeros_like(full)
        for m in range(-ell, ell + 1):
            phase_m = torch.exp(torch.complex(zero, -m * phi_t)).to(dtype=dtype)
            acc = torch.zeros((rows.shape[0],), dtype=dtype, device=device)
            for mp in range(-ell, ell + 1):
                dval = _wigner_d_element(ell, m, mp, beta)
                phase_mp = torch.exp(torch.complex(zero, -mp * psi_t)).to(dtype=dtype)
                acc = acc + (dval.to(dtype=dtype) * phase_mp) * full[:, ell + mp]
            rotated[:, ell + m] = phase_m * acc
        for m in range(0, min(ell, mm) + 1):
            out[:, alm_index(ell, m, ll, mm)] = rotated[:, ell + m]
    return out[0] if single else out


def alm2cl(
    alms1: Tensor | list[complex] | list[list[complex]],
    alms2: Tensor | list[complex] | list[list[complex]] | None = None,
    *,
    lmax: int | None = None,
    mmax: int | None = None,
    lmax_out: int | None = None,
    nspec: int | None = None,
) -> Tensor:
    """
    Compute auto/cross power spectra from alm coefficients.

    For a single alm input vector, returns shape `(lmax_out+1,)`.
    For multiple inputs, returns stacked spectra in healpy-compatible ordering.
    """
    a1_rows, a1_single = _alm_to_rows(alms1)
    ll, mm = _infer_lmax_mmax_from_nalm(int(a1_rows.shape[1]), lmax, mmax)
    if alms2 is None:
        a2_rows = a1_rows
    else:
        a2_rows, _ = _alm_to_rows(alms2)
        ll2, mm2 = _infer_lmax_mmax_from_nalm(int(a2_rows.shape[1]), lmax, mmax)
        if ll2 != ll or mm2 != mm:
            raise ValueError("alms2 shape incompatible with lmax/mmax")
    if a1_rows.shape[0] != a2_rows.shape[0]:
        raise ValueError("alms1 and alms2 must have same number of components")

    ncomp = int(a1_rows.shape[0])
    spectra: list[Tensor] = []

    for i in range(ncomp):
        spectra.append(_cross_cl_from_alm_pair(a1_rows[i], a2_rows[i], ll, mm))

    for delta in range(1, ncomp):
        for i in range(0, ncomp - delta):
            j = i + delta
            spectra.append(_cross_cl_from_alm_pair(a1_rows[i], a2_rows[j], ll, mm))

    cls = torch.stack(spectra, dim=0)
    if nspec is not None:
        cls = cls[: int(nspec)]
    ll_out = ll if lmax_out is None else int(lmax_out)
    if ll_out < 0:
        raise ValueError("lmax_out must be non-negative")
    ll_out = min(ll_out, ll)
    cls = cls[..., : ll_out + 1]
    if a1_single and (alms2 is None or torch.as_tensor(alms2).ndim == 1):
        return cls[0]
    return cls


def alm2map(
    alm_values: Tensor | list[complex] | list[list[complex]],
    nside: int,
    *,
    lmax: int | None = None,
    mmax: int | None = None,
    nest: bool = False,
    pol: bool = False,
    pixwin: bool = False,
) -> Tensor:
    """
    Synthesize HEALPix map(s) from alm coefficients.

    For scalar maps, m>0 terms use the real-map relation `2*Re(a_lm Y_lm)`.
    With `pol=True`, `alm_values` must be shape `(3, nalm)` in `(T,E,B)` order.
    Alms on CPU use fused C++. CUDA/MPS keep alms on device (ATen ring FFT).
    """
    rows, single = _alm_to_rows(alm_values)
    ll, mm = _infer_lmax_mmax_from_nalm(int(rows.shape[1]), lmax, mmax)

    if pol:
        alms_pol = torch.as_tensor(alm_values)
        alms_pol = alms_pol.to(dtype=_preferred_complex_dtype(alms_pol.device))
        if alms_pol.ndim != 2 or alms_pol.shape[0] != 3:
            raise ValueError("pol=True requires alm_values shape (3, nalm)")
        ll, mm = _infer_lmax_mmax_from_nalm(int(alms_pol.shape[1]), lmax, mmax)

        a_t = alms_pol[0]
        a_e = alms_pol[1]
        a_b = alms_pol[2]
        if pixwin:
            pixwin_loader = globals()["pixwin"]
            pw_t, pw_p = pixwin_loader(int(nside), lmax=ll, pol=True)
            ell = _alm_ell_array(ll, mm)
            cdt = _preferred_complex_dtype(alms_pol.device)
            st = pw_t.index_select(0, ell).to(device=alms_pol.device, dtype=cdt)
            sp = pw_p.index_select(0, ell).to(device=alms_pol.device, dtype=cdt)
            a_t = a_t * st
            a_e = a_e * sp
            a_b = a_b * sp
        t = alm2map(
            a_t,
            nside=nside,
            lmax=ll,
            mmax=mm,
            nest=nest,
            pol=False,
            pixwin=False,
        )
        qu = alm2map_spin(
            (a_e, a_b),
            nside=nside,
            spin=2,
            lmax=ll,
            mmax=mm,
            nest=nest,
        )
        return torch.stack([t, qu[0], qu[1]], dim=0)

    rows_eff = rows
    if pixwin:
        pixwin_loader = globals()["pixwin"]
        pw = pixwin_loader(int(nside), lmax=ll, pol=False)
        rows_eff = smoothalm(rows, lmax=ll, mmax=mm, beam=pw, pol=False)
        rows_eff, _ = _alm_to_rows(rows_eff)

    ring_mode = os.environ.get("TORCHSKY_SCALAR_ALM2MAP_RING_TORCH", "auto")
    npix = _healpix.nside2npix(nside)
    nalm = alm_size(ll, mm)
    bytes_needed = int(npix * nalm * 16)
    auto_ring = ring_mode == "auto" and bytes_needed >= _SCALAR_RING_AUTO_MIN_BYTES
    use_ring_torch = (not nest) and (
        ring_mode == "force" or ring_mode == "1" or auto_ring
    )
    if use_ring_torch:
        out_ring = _scalar_alm2map_ring_torch(
            rows_eff, nside=int(nside), lmax=ll, mmax=mm
        )
        if torch.isfinite(out_ring).all():
            return out_ring[0] if single else out_ring
        if ring_mode == "force":
            raise RuntimeError(
                "TORCHSKY_SCALAR_ALM2MAP_RING_TORCH=force produced non-finite values; "
                "disable force or lower lmax/nside."
            )

    use_cpp_direct = (
        _native_ops.has_spectral_dispatcher()
        and rows_eff.device.type == "cpu"
        and (not rows_eff.requires_grad)
        and os.environ.get("TORCHSKY_SCALAR_ALM2MAP_CPP", "0") != "0"
    )
    if use_cpp_direct:
        _, phi, x, _ = _angles_for_nside(int(nside), bool(nest))
        out_cpp = _native_ops.spectral_scalar_alm2map_direct(
            rows_eff.contiguous(),
            x.contiguous(),
            phi.contiguous(),
            int(ll),
            int(mm),
        ).to(dtype=torch.float64)
        return out_cpp[0] if single else out_cpp

    mat = _scalar_alm2map_mat(nside, nest, ll, mm)
    real_dtype = _preferred_real_dtype(rows_eff.device)
    complex_dtype = _preferred_complex_dtype(rows_eff.device)
    if mat is not None:
        if mat.device != rows_eff.device or mat.dtype != complex_dtype:
            mat = mat.to(device=rows_eff.device, dtype=complex_dtype)
        m_idx = _alm_m_array(ll, mm)
        gain = torch.where(
            m_idx == 0,
            torch.ones_like(m_idx, dtype=real_dtype),
            torch.full_like(m_idx, 2.0, dtype=real_dtype),
        ).to(dtype=complex_dtype, device=rows_eff.device)
        out = torch.real((rows_eff * gain.unsqueeze(0)) @ mat).to(real_dtype)
    else:
        y_blocks, _ = _ylm_basis(nside, nest, ll, mm)
        out = torch.zeros(
            (rows.shape[0], npix), dtype=real_dtype, device=rows_eff.device
        )
        for m in range(mm + 1):
            y = y_blocks[m].to(device=rows_eff.device, dtype=complex_dtype)
            l_count = ll - m + 1
            start = alm_index(m, m, ll, mm)
            coeffs = rows_eff[:, start : start + l_count]
            contrib = coeffs @ y
            if m == 0:
                out = out + contrib.real
            else:
                out = out + 2.0 * contrib.real
    return out[0] if single else out


def alm2lonlat(
    alm_values: Tensor | list[complex] | list[list[complex]],
    lon_deg: Tensor | list[float] | float,
    lat_deg: Tensor | list[float] | float,
    *,
    lmax: int | None = None,
    mmax: int | None = None,
) -> Tensor:
    """Evaluate scalar alm coefficients at arbitrary sky coordinates."""
    rows, single = _alm_to_rows(alm_values)
    ll, mm = _infer_lmax_mmax_from_nalm(int(rows.shape[1]), lmax, mmax)
    device = rows.device
    real_dtype = _preferred_real_dtype(device)
    complex_dtype = _preferred_complex_dtype(device)

    lon_t = torch.as_tensor(lon_deg, dtype=real_dtype, device=device)
    lat_t = torch.as_tensor(lat_deg, dtype=real_dtype, device=device)
    lon_t, lat_t = torch.broadcast_tensors(lon_t, lat_t)
    coord_shape = lon_t.shape
    phi = torch.deg2rad(torch.remainder(lon_t.reshape(-1), 360.0))
    x = torch.sin(torch.deg2rad(lat_t.reshape(-1)))

    out = torch.zeros((rows.shape[0], phi.shape[0]), dtype=real_dtype, device=device)
    for m in range(mm + 1):
        p_seq = _legendre_l_sequence(m, x, ll)
        phase = torch.exp(torch.complex(torch.zeros_like(phi), m * phi)).to(
            dtype=complex_dtype
        )
        y = p_seq.to(dtype=complex_dtype) * phase.unsqueeze(0)
        l_count = ll - m + 1
        start = alm_index(m, m, ll, mm)
        coeffs = rows[:, start : start + l_count].to(dtype=complex_dtype)
        contrib = coeffs @ y.to(dtype=complex_dtype)
        if m == 0:
            out = out + contrib.real.to(dtype=real_dtype)
        else:
            out = out + 2.0 * contrib.real.to(dtype=real_dtype)
    out = out.reshape(rows.shape[0], *coord_shape)
    return out[0] if single else out


def alm2lonlat_spin(
    alms: tuple[Tensor, Tensor],
    spin: int,
    lon_deg: Tensor | list[float] | float,
    lat_deg: Tensor | list[float] | float,
    *,
    lmax: int | None = None,
    mmax: int | None = None,
) -> Tensor:
    """Evaluate spin-weighted alm coefficients at arbitrary sky coordinates."""
    alm1 = torch.as_tensor(alms[0])
    alm2 = torch.as_tensor(alms[1])
    _require_sht_device(alm1, alm2, what="alm2lonlat_spin")
    raw_spin = float(torch.as_tensor(spin).item())
    spin_i = int(raw_spin)
    if float(spin_i) != raw_spin:
        raise ValueError("spin must be an integer")
    if spin_i < 0:
        raise ValueError("spin must be non-negative")
    ll, mm = _infer_lmax_mmax_from_nalm(int(alm1.numel()), lmax, mmax)
    if int(alm2.numel()) != alm_size(ll, mm):
        raise ValueError("alm arrays must have matching size")

    device = alm1.device
    real_dtype = _preferred_real_dtype(device)
    complex_dtype = _preferred_complex_dtype(device)
    alm1 = alm1.to(dtype=complex_dtype)
    alm2 = alm2.to(dtype=complex_dtype)
    lon_t = torch.as_tensor(lon_deg, dtype=real_dtype, device=device)
    lat_t = torch.as_tensor(lat_deg, dtype=real_dtype, device=device)
    lon_t, lat_t = torch.broadcast_tensors(lon_t, lat_t)
    coord_shape = lon_t.shape
    phi = torch.deg2rad(torch.remainder(lon_t.reshape(-1), 360.0))
    theta = torch.deg2rad(90.0 - lat_t.reshape(-1))

    neg_sign = -1.0 if (spin_i & 1) else 1.0
    coeff_plus = (alm1 + 1j * alm2) * neg_sign
    coeff_minus = alm1 - 1j * alm2
    p_plus = torch.zeros(phi.shape[0], dtype=complex_dtype, device=device)
    p_minus = torch.zeros(phi.shape[0], dtype=complex_dtype, device=device)
    phase_zero = torch.zeros_like(phi)

    for m in range(mm + 1):
        phase = torch.exp(torch.complex(phase_zero, m * phi)).to(complex_dtype)
        ell_start = max(m, abs(spin_i))
        if ell_start > ll:
            continue
        d_plus = _wigner_d_l_sequence(ell_start, ll, m, -spin_i, theta)
        d_minus = _wigner_d_l_sequence(ell_start, ll, m, spin_i, theta)
        ell_vec = torch.arange(
            ell_start, ll + 1, dtype=real_dtype, device=theta.device
        )
        norms = torch.sqrt((2 * ell_vec + 1) / (4.0 * math.pi))
        m_phase = -1.0 if (m & 1) else 1.0
        spin_phase = -1.0 if (spin_i & 1) else 1.0
        scale = (spin_phase * m_phase) * norms.unsqueeze(1)
        y_plus = (scale * d_plus).to(complex_dtype) * phase.unsqueeze(0)
        y_minus = (scale * d_minus).to(complex_dtype) * phase.unsqueeze(0)
        start = alm_index(ell_start, m, ll, mm)
        l_count = ll - ell_start + 1
        c_plus = coeff_plus[start : start + l_count].to(dtype=complex_dtype)
        c_minus = coeff_minus[start : start + l_count].to(dtype=complex_dtype)
        p_plus = p_plus + c_plus @ y_plus.to(dtype=complex_dtype)
        p_minus = p_minus + c_minus @ y_minus.to(dtype=complex_dtype)

    q = 0.5 * (p_plus + p_minus)
    u = -0.5j * (p_plus - p_minus)
    out = torch.stack(
        [(-q).real.to(real_dtype), (-u).real.to(real_dtype)],
        dim=0,
    )
    return out.reshape(2, *coord_shape)


def alm2map_batched(
    alms: Tensor,
    nside: int,
    *,
    lmax: int | None = None,
    mmax: int | None = None,
    nest: bool = False,
    pol: bool = False,
    pixwin: bool = False,
) -> Tensor:
    """
    Batched map synthesis with explicit leading batch dimension.

    Scalar alms must be shaped ``(B, nalm)`` and return ``(B, npix)``.
    With ``pol=True``, ``alms`` must be ``(B, 3, nalm)`` in ``(T,E,B)`` order and the
    result is ``(B, 3, npix)``.
    """
    alms_t = torch.as_tensor(alms)
    if pol:
        if alms_t.ndim != 3 or int(alms_t.shape[1]) != 3:
            raise ValueError(
                "alm2map_batched with pol=True expects shape (B, 3, nalm) in (T,E,B) order"
            )
        outs: list[Tensor] = []
        for b in range(int(alms_t.shape[0])):
            outs.append(
                alm2map(
                    alms_t[b],
                    nside,
                    lmax=lmax,
                    mmax=mmax,
                    nest=nest,
                    pol=True,
                    pixwin=pixwin,
                )
            )
        return torch.stack(outs, dim=0)
    if alms_t.ndim != 2:
        raise ValueError("alm2map_batched with pol=False expects shape (B, nalm)")
    return alm2map(
        alms_t,
        nside,
        lmax=lmax,
        mmax=mmax,
        nest=nest,
        pol=False,
        pixwin=pixwin,
    )




def map2alm_spin(
    maps: Tensor,
    spin: int,
    *,
    nside: int | None = None,
    lmax: int | None = None,
    mmax: int | None = None,
    nest: bool = False,
    iter: int = 0,
    use_weights: bool = False,
    use_pixel_weights: bool = False,
    gal_cut: float = 0.0,
    datapath: str | os.PathLike[str] | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Spin-weighted map->alm transform (Q/U-like pair).

    Input `maps` shape must be `(2, npix)`.
    """
    maps_t = torch.as_tensor(maps)
    maps_t = maps_t.to(dtype=_preferred_real_dtype(maps_t.device))
    _require_sht_device(maps_t, what="map2alm_spin")
    if maps_t.ndim != 2 or maps_t.shape[0] != 2:
        raise ValueError("maps must have shape (2, npix)")
    raw_spin = float(torch.as_tensor(spin).item())
    npix = int(maps_t.shape[1])
    ns = _healpix.npix2nside(npix) if nside is None else int(nside)
    if _healpix.nside2npix(ns) != npix:
        raise ValueError("map length does not match nside")
    ll = int(3 * ns - 1 if lmax is None else lmax)
    mm = ll if mmax is None else int(mmax)
    if ll < 0 or mm < 0 or mm > ll:
        raise ValueError("invalid lmax/mmax")
    try:
        spin_i = int(spin)
    except Exception as exc:
        raise ValueError("spin must be an integer") from exc
    if float(spin_i) != raw_spin:
        raise ValueError("spin must be an integer")
    if spin_i < 0:
        raise ValueError("spin must be non-negative")

    if nest:
        # Reorder to RING for memory stability
        m0 = _reorder_nest_to_ring_rows(maps_t[0:1], ns)[0]
        m1 = _reorder_nest_to_ring_rows(maps_t[1:2], ns)[0]
        maps_t = torch.stack([m0, m1], dim=0)
        nest = False

    analysis_maps = _apply_map2alm_weights(
        maps_t,
        nside=ns,
        gal_cut=gal_cut,
        use_weights=use_weights,
        use_pixel_weights=use_pixel_weights,
        datapath=datapath,
    )

    # Iterative solver
    alm1, alm2 = _map2alm_spin_single_pass(
        analysis_maps, spin_i, nside=ns, lmax=ll, mmax=mm
    )
    for _ in range(iter):
        m_rec_0, m_rec_1 = alm2map_spin((alm1, alm2), ns, spin_i, lmax=ll, mmax=mm)
        res0 = maps_t[0] - m_rec_0
        res1 = maps_t[1] - m_rec_1
        analysis_res = _apply_map2alm_weights(
            torch.stack([res0, res1], dim=0),
            nside=ns,
            gal_cut=gal_cut,
            use_weights=use_weights,
            use_pixel_weights=use_pixel_weights,
            datapath=datapath,
        )
        res_alm1, res_alm2 = _map2alm_spin_single_pass(
            analysis_res, spin_i, nside=ns, lmax=ll, mmax=mm
        )
        alm1 = alm1 + res_alm1
        alm2 = alm2 + res_alm2

    return alm1, alm2


def _map2alm_spin_single_pass(
    maps_t: Tensor,
    spin_i: int,
    *,
    nside: int,
    lmax: int,
    mmax: int,
) -> tuple[Tensor, Tensor]:
    _require_sht_device(maps_t, what="map2alm_spin")
    rows = maps_t

    def _cpu_c128(t: Tensor) -> Tensor:
        if t.device.type != "cpu":
            raise RuntimeError("C++ spin recurrence is CPU-only")
        if t.dtype != torch.complex128:
            t = t.to(dtype=torch.complex128)
        return t.contiguous()

    npix = int(maps_t.shape[1])
    pix_w = (4.0 * math.pi) / float(npix)
    nalm = alm_size(lmax, mmax)
    ring_mode = os.environ.get("TORCHSKY_SPIN_MAP2ALM_RING_TORCH", "auto")
    bytes_needed = int(
        npix * nalm * 32
    )  # two complex matrices at complex128-equivalent footprint
    auto_ring = (
        ring_mode == "auto" and bytes_needed >= _SPIN_MAP2ALM_RING_AUTO_MIN_BYTES
    )
    use_ring_torch = ring_mode == "force" or ring_mode == "1" or auto_ring
    if use_ring_torch:
        starts_cpu, lengths_cpu, theta_ring_cpu, _ = _ring_layout_for_nside(nside)
        rows_ring = rows
        fused_rec = (
            os.environ.get("TORCHSKY_SPIN_MAP2ALM_RECURRENCE_CPP", "1") != "0"
            and _HAS_HEALPIX_SPIN_MAP2ALM_RING_REC
            and rows_ring.device.type == "cpu"
            and (not maps_t.requires_grad)
            and (not rows_ring.requires_grad)
        )
        if fused_rec:
            maps_f64 = (
                rows_ring
                if rows_ring.dtype == torch.float64
                else rows_ring.to(dtype=torch.float64)
            )
            phase0_neg = _ring_phi0_phase_for_nside(int(nside), int(mmax), sign=-1)
            c_stacked = _native_ops.spectral_spin_map2alm_ring_recurrence(
                maps_f64,
                starts_cpu,
                lengths_cpu,
                theta_ring_cpu,
                phase0_neg,
                int(lmax),
                int(mmax),
                int(spin_i),
                float(pix_w),
            )
            c_plus = c_stacked[0]
            c_minus = c_stacked[1]
        else:
            p_plus = torch.complex(rows_ring[0], rows_ring[1])
            nrings = (4 * int(nside)) - 1
            bytes_concat = int(nalm * nrings * 32)  # two complex128 concat bases
            use_concat_fast = (
                bytes_concat <= _SPIN_RING_INTEGRATE_CONCAT_FAST_MAX_BYTES
                and nalm <= _SPIN_RING_INTEGRATE_CONCAT_MAX_NALM
            )
            if use_concat_fast:
                phase0_neg = _ring_phi0_phase_for_nside(int(nside), int(mmax), sign=-1)
                if phase0_neg.dtype != torch.complex128:
                    phase0_neg = phase0_neg.to(dtype=torch.complex128)
                use_fused_cpp = (
                    os.environ.get("TORCHSKY_SPIN_MAP2ALM_RING_CONCAT_CPP", "1") != "0"
                    and _native_ops.has_spectral_dispatcher()
                    and p_plus.device.type == "cpu"
                    and p_plus.dtype == torch.complex128
                    and (not p_plus.requires_grad)
                )
                if use_fused_cpp:
                    y_plus, y_minus = _ring_spin_basis_concat(
                        int(nside), int(lmax), int(mmax), int(spin_i)
                    )
                    if y_plus.device.type != "cpu":
                        use_fused_cpp = False
                    else:
                        c_plus, c_minus = _native_ops.spectral_spin_map2alm_ring_concat(
                            p_plus.contiguous(),
                            starts_cpu.contiguous(),
                            lengths_cpu.contiguous(),
                            phase0_neg.contiguous(),
                            y_plus.contiguous(),
                            y_minus.contiguous(),
                            int(lmax),
                            int(mmax),
                            float(pix_w),
                        )
                if not use_fused_cpp:
                    s_plus, s_minus = _ring_fourier_modes_spin_conj(
                        p_plus,
                        starts_cpu=starts_cpu,
                        lengths_cpu=lengths_cpu,
                        mmax=mmax,
                        nside=int(nside),
                    )
                    phase = (
                        phase0_neg
                        if s_plus.device.type == "cpu"
                        else phase0_neg.to(device=s_plus.device)
                    )
                    s_plus = s_plus * phase
                    s_minus = s_minus * phase
                    c_plus, c_minus = _ring_spin_integrate_blocks(
                        s_plus,
                        s_minus,
                        nside=int(nside),
                        lmax=int(lmax),
                        mmax=int(mmax),
                        spin=int(spin_i),
                        pix_w=float(pix_w),
                    )
            else:
                s_plus, s_minus = _ring_fourier_modes_spin_conj(
                    p_plus,
                    starts_cpu=starts_cpu,
                    lengths_cpu=lengths_cpu,
                    mmax=mmax,
                    nside=int(nside),
                )

                phase0_neg = _ring_phi0_phase_for_nside(
                    int(nside), int(mmax), sign=-1
                ).to(dtype=torch.complex128)
                s_plus = s_plus * phase0_neg
                s_minus = s_minus * phase0_neg

                if (
                    _native_ops.has_spectral_dispatcher()
                    and os.environ.get("TORCHSKY_SPIN_MAP2ALM_RECURRENCE_CPP", "1")
                    != "0"
                    and s_plus.device.type == "cpu"
                ):
                    if s_plus.requires_grad or s_minus.requires_grad:
                        c_plus, c_minus = SpinSHTIntegrate.apply(
                            s_plus,
                            s_minus,
                            int(nside),
                            int(lmax),
                            int(mmax),
                            int(spin_i),
                            float(pix_w),
                        )
                    else:
                        _, _, theta_ring_cpu, _ = _ring_layout_for_nside(int(nside))
                        nrings = int(theta_ring_cpu.numel())
                        w_vec = torch.full(
                            (nrings,), float(pix_w), dtype=torch.float64, device="cpu"
                        )
                        sp_cpu = _cpu_c128(s_plus)
                        sm_cpu = _cpu_c128(s_minus)
                        c_stacked = _native_ops.spectral_spin_integrate_recurrence(
                            sp_cpu,
                            sm_cpu,
                            theta_ring_cpu.contiguous(),
                            w_vec,
                            int(lmax),
                            int(mmax),
                            int(spin_i),
                        )
                        c_plus = c_stacked[0].to(device=s_plus.device)
                        c_minus = c_stacked[1].to(device=s_minus.device)
                else:
                    c_plus, c_minus = _ring_spin_integrate_blocks(
                        s_plus,
                        s_minus,
                        nside=int(nside),
                        lmax=int(lmax),
                        mmax=int(mmax),
                        spin=int(spin_i),
                        pix_w=float(pix_w),
                    )
    else:
        mats = _spin_map2alm_mats(nside, False, lmax, mmax, spin_i)
        if mats is None:
            # Dense cached matrices unavailable (memory cap); use ring path fallback.
            starts_cpu, lengths_cpu, _, _ = _ring_layout_for_nside(nside)
            p_plus = torch.complex(rows[0], rows[1])
            s_plus, s_minus = _ring_fourier_modes_spin_conj(
                p_plus,
                starts_cpu=starts_cpu,
                lengths_cpu=lengths_cpu,
                mmax=mmax,
                nside=int(nside),
            )
            phase0_neg = _ring_phi0_phase_for_nside(int(nside), int(mmax), sign=-1).to(
                dtype=torch.complex128
            )
            s_plus = s_plus * phase0_neg
            s_minus = s_minus * phase0_neg
            if (
                _native_ops.has_spectral_dispatcher()
                and os.environ.get("TORCHSKY_SPIN_MAP2ALM_RECURRENCE_CPP", "1") != "0"
                and s_plus.device.type == "cpu"
            ):
                if s_plus.requires_grad or s_minus.requires_grad:
                    c_plus, c_minus = SpinSHTIntegrate.apply(
                        s_plus,
                        s_minus,
                        int(nside),
                        int(lmax),
                        int(mmax),
                        int(spin_i),
                        float(pix_w),
                    )
                else:
                    _, _, theta_ring_cpu, _ = _ring_layout_for_nside(int(nside))
                    nrings = int(theta_ring_cpu.numel())
                    w_vec = torch.full(
                        (nrings,), float(pix_w), dtype=torch.float64, device="cpu"
                    )
                    sp_cpu = _cpu_c128(s_plus)
                    sm_cpu = _cpu_c128(s_minus)
                    c_stacked = _native_ops.spectral_spin_integrate_recurrence(
                        sp_cpu,
                        sm_cpu,
                        theta_ring_cpu.contiguous(),
                        w_vec,
                        int(lmax),
                        int(mmax),
                        int(spin_i),
                    )
                    c_plus = c_stacked[0].to(device=s_plus.device)
                    c_minus = c_stacked[1].to(device=s_minus.device)
            else:
                c_plus, c_minus = _ring_spin_integrate_blocks(
                    s_plus,
                    s_minus,
                    nside=int(nside),
                    lmax=int(lmax),
                    mmax=int(mmax),
                    spin=int(spin_i),
                    pix_w=float(pix_w),
                )
        else:
            ycp_t, ycm_t = mats
            use_cpp = (
                _native_ops.has_spectral_dispatcher()
                and rows.device.type == "cpu"
                # Torch matmul path is currently faster on Apple CPU; keep C++ path opt-in.
                and os.environ.get("TORCHSKY_SPIN_MAP2ALM_CPP", "0") != "0"
            )
            if use_cpp:
                c_plus, c_minus = _native_ops._spectral_spin_map2alm_from_basis(
                    rows[0].contiguous(),
                    rows[1].contiguous(),
                    ycp_t,
                    ycm_t,
                    float(pix_w),
                )
            else:
                p_plus = torch.complex(rows[0], rows[1])
                p_minus = torch.conj(p_plus)
                c_plus = (p_plus @ ycp_t) * pix_w
                c_minus = (p_minus @ ycm_t) * pix_w

    neg_sign = -1.0 if (spin_i & 1) else 1.0  # (-1)^spin
    a_e = -0.5 * ((neg_sign * c_plus) + c_minus)
    a_b = 0.5j * ((neg_sign * c_plus) - c_minus)
    ell = _alm_ell_array(int(lmax), int(mmax))
    low_ell = ell < abs(spin_i)
    if torch.any(low_ell):
        a_e = a_e.clone()
        a_b = a_b.clone()
        a_e[low_ell] = 0.0
        a_b[low_ell] = 0.0
    return a_e, a_b


def alm2map_spin(
    alms: tuple[Tensor, Tensor],
    nside: int,
    spin: int,
    *,
    lmax: int | None = None,
    mmax: int | None = None,
    nest: bool = False,
) -> Tensor:
    """
    Spin-weighted alm->map transform (Q/U-like pair).

    Returns tensor of shape `(2, npix)`.
    """
    alm1 = torch.as_tensor(alms[0])
    alm2 = torch.as_tensor(alms[1])
    cdt = _preferred_complex_dtype(alm1.device)
    alm1 = alm1.to(dtype=cdt)
    alm2 = alm2.to(dtype=cdt)
    _require_sht_device(alm1, alm2, what="alm2map_spin")
    raw_spin = float(torch.as_tensor(spin).item())
    spin_i = int(raw_spin)
    if float(spin_i) != raw_spin:
        raise ValueError("spin must be an integer")
    if spin_i < 0:
        raise ValueError("spin must be non-negative")
    ll, mm = _infer_lmax_mmax_from_nalm(int(alm1.numel()), lmax, mmax)
    if int(alm2.numel()) != alm_size(ll, mm):
        raise ValueError("alm arrays must have matching size")

    npix = _healpix.nside2npix(int(nside))
    neg_sign = -1.0 if (spin_i & 1) else 1.0  # (-1)^spin
    coeff_plus = (alm1 + 1j * alm2) * neg_sign
    coeff_minus = alm1 - 1j * alm2

    nalm = alm_size(ll, mm)
    ring_mode = os.environ.get("TORCHSKY_SPIN_ALM2MAP_RING_TORCH", "auto")
    bytes_needed = int(npix * nalm * 32)  # two complex basis-like work arrays
    auto_ring = (
        ring_mode == "auto" and bytes_needed >= _SPIN_ALM2MAP_RING_AUTO_MIN_BYTES
    )
    use_ring_torch = (ring_mode == "force" or ring_mode == "1" or auto_ring) and (
        _SPIN_RING_NEST_ENABLE or (not nest)
    )
    if use_ring_torch:
        out_ring = _spin_alm2map_ring_torch(
            coeff_plus,
            coeff_minus,
            nside=int(nside),
            lmax=ll,
            mmax=mm,
            spin=spin_i,
        )
        if torch.isfinite(out_ring).all():
            if nest:
                return _reorder_ring_to_nest_rows(out_ring, int(nside))
            return out_ring
        if ring_mode == "force":
            raise RuntimeError(
                "TORCHSKY_SPIN_ALM2MAP_RING_TORCH=force produced non-finite values; "
                "disable force or lower lmax/nside."
            )

    mats = _spin_alm2map_mats(int(nside), nest, ll, mm, spin_i)
    if mats is not None:
        y_plus, y_minus, y_plus_m0, y_minus_m0 = mats
        use_cpp = (
            _native_ops.has_spectral_dispatcher()
            and alm1.device.type == "cpu"
            # Torch matmul path is currently faster on Apple CPU; keep C++ path opt-in.
            and os.environ.get("TORCHSKY_SPIN_ALM2MAP_CPP", "0") != "0"
        )
        if use_cpp:
            return _native_ops._spectral_spin_alm2map_from_basis(
                coeff_plus.contiguous(),
                coeff_minus.contiguous(),
                y_plus,
                y_minus,
                y_plus_m0,
                y_minus_m0,
            )
        y_plus = y_plus.to(device=alm1.device, dtype=cdt)
        y_minus = y_minus.to(device=alm1.device, dtype=cdt)
        y_plus_m0 = y_plus_m0.to(device=alm1.device, dtype=cdt)
        y_minus_m0 = y_minus_m0.to(device=alm1.device, dtype=cdt)
        p_plus_pos = coeff_plus @ y_plus
        p_minus_pos = coeff_minus @ y_minus
        if mm == 0:
            p_plus = p_plus_pos
            p_minus = p_minus_pos
        else:
            p_plus_m0 = coeff_plus[: ll + 1] @ y_plus_m0
            p_minus_m0 = coeff_minus[: ll + 1] @ y_minus_m0
            p_plus = p_plus_pos + torch.conj(p_minus_pos - p_minus_m0)
            p_minus = p_minus_pos + torch.conj(p_plus_pos - p_plus_m0)
    else:
        Y_plus, _ = _spin_ylm_basis_concat(int(nside), nest, ll, mm, spin_i)
        Y_minus, _ = _spin_ylm_basis_concat(int(nside), nest, ll, mm, -spin_i)
        Y_plus = Y_plus.to(device=alm1.device, dtype=cdt)
        Y_minus = Y_minus.to(device=alm1.device, dtype=cdt)

        # P = C @ Y
        # C: (Nalm,), Y: (Nalm, Npix)
        # P: (Npix,)
        # P = C[None, :] @ Y = (1, Nalm) @ (Nalm, Npix) -> (1, Npix) -> squeeze
        # OR torch.mv(Y.T, C)? Y is (Nalm, Npix). Y.T is (Npix, Nalm).
        # P = Y.T @ C

        # This implies cross-contribution for m > 0.
        # This relationship P_- = conj(P_+) holds for real tensor fields.
        # But here we are computing P from alms.
        # The loop handles general case.
        # But wait, p_plus and p_minus summation is:
        # P_+ = sum_{lm} (E_lm + i B_lm) Y_{s,lm}
        # P_- = sum_{lm} (E_lm - i B_lm) Y_{-s,lm}
        # The input `alms` are (E_lm, B_lm).
        # `coeff_plus` = E + iB = a_{s,lm}
        # `coeff_minus` = E - iB = a_{-s,lm}
        # So P_+ = coeff_plus @ Y_plus
        # P_- = coeff_minus @ Y_minus

        # But what about the `if m > 0` part?
        # That logic in original code:
        # `p_plus = p_plus + torch.conj(p_minus_m)`
        # `p_minus_m` comes from `cm @ ym`.
        # `cm` is `coeff_minus` segment. `ym` is `Y_minus` segment.
        # This suggests that we are enforcing real-field constraints?
        # If the map is real, P_- = conj(P_+).
        # But here `alm2map_spin` returns a map.
        # If the stored ALMs are for real map (only m >= 0 stored), then we must add negative m terms.
        # The stored ALMs typically only have m >= 0.
        # The loop logic accounts for negative m by using symmetry:
        # Y_{s, l, -m} = (-1)^{s+m} Y^*_{-s, l, m}
        # a_{s, l, -m} = (-1)^{s+m} a^*_{-s, l, m} (for real fields)
        # So contribution from -m is related to +m.
        # The code: `p_plus += conj(p_minus_m)`
        # `p_minus_m` = a_{-s, l, m} * Y_{-s, l, m}
        # conj(p_minus_m) = a^*_{-s, l, m} * Y^*_{-s, l, m}
        # This matches the symmetry for negative m terms contributing to P_+.

        # So we can compute positive m contribution (m >= 0) using one matmul.
        # Then we need to add negative m contribution (m < 0).
        # The negative m contribution comes from `m > 0` terms in the loop.

        # Let P_plus_pos = coeff_plus @ Y_plus (sum over m>=0)
        # Let P_minus_pos = coeff_minus @ Y_minus (sum over m>=0)

        P_plus_pos = torch.mv(Y_plus.T, coeff_plus)
        P_minus_pos = torch.mv(Y_minus.T, coeff_minus)

        # m<0 contributions via symmetry:
        # a_{s,l,-m} Y_{s,l,-m} = conj(a_{-s,l,m} Y_{-s,l,m})
        # We sum m>=0 contributions into P_minus_pos, then subtract the m=0
        # term before conjugating to get the m>0 contributions.
        l_count_m0 = ll + 1

        P_plus_m0 = torch.mv(Y_plus[:l_count_m0].T, coeff_plus[:l_count_m0])
        P_minus_m0 = torch.mv(Y_minus[:l_count_m0].T, coeff_minus[:l_count_m0])

        p_plus = P_plus_pos + torch.conj(P_minus_pos - P_minus_m0)
        p_minus = P_minus_pos + torch.conj(P_plus_pos - P_plus_m0)

    q = 0.5 * (p_plus + p_minus)
    u = -0.5j * (p_plus - p_minus)
    # Align sign convention with healpy Q/U synthesis.
    real_dtype = _preferred_real_dtype(q.device)
    return torch.stack([(-q).real.to(real_dtype), (-u).real.to(real_dtype)], dim=0)


def alm2map_der1(
    alm_values: Tensor | list[complex],
    nside: int,
    *,
    lmax: int | None = None,
    mmax: int | None = None,
    nest: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """Synthesize a map and its first angular derivatives from scalar alm.

    Returns ``(map, d_theta, d_phi)`` with healpy semantics: ``d_phi`` is already
    divided by ``sin(theta)``. Derivatives use the spin-1 relation
    ``a_{E,lm} = sqrt(l(l+1)) a_{lm}`` (B=0) via :func:`alm2map_spin`, not
    finite differences on the map.
    """
    alm = torch.as_tensor(alm_values)
    if alm.ndim != 1:
        raise ValueError("alm2map_der1 expects a 1D scalar alm vector")
    cdt = _preferred_complex_dtype(alm.device)
    alm = alm.to(dtype=cdt)
    ll, mm = _infer_lmax_mmax_from_nalm(int(alm.numel()), lmax, mmax)
    m = alm2map(alm, nside, lmax=ll, mmax=mm, nest=nest)
    ell = _alm_ell_array(ll, mm).to(device=alm.device)
    rdtype = _preferred_real_dtype(alm.device)
    fac = torch.sqrt(ell.to(dtype=rdtype) * (ell.to(dtype=rdtype) + 1.0))
    alm_e = alm * fac.to(dtype=alm.dtype)
    alm_b = torch.zeros_like(alm_e)
    qu = alm2map_spin((alm_e, alm_b), nside, 1, lmax=ll, mmax=mm, nest=nest)
    return m, qu[0], qu[1]


def clear_spectral_caches() -> None:
    """Clear internal spectral/SHT lookup caches."""
    _SPECTRAL_CACHE.clear()


__all__ = [
    "alm2lonlat",
    "alm2lonlat_spin",
    "alm2map",
    "alm2map_batched",
    "alm2cl",
    "alm2map_spin",
    "alm2map_der1",
    "alm_index",
    "alm_size",
    "almxfl",
    "anafast",
    "beam2bl",
    "bl2beam",
    "clear_spectral_caches",
    "gaussian_beam",
    "map2alm",
    "map2alm_batched",
    "map2alm_lsq",
    "map2alm_spin",
    "pixwin",
    "rotate_alm",
    "smoothalm",
    "smoothmap",
    "synalm",
    "synfast",
]

# End of file


# Spectra/beam/smoothing operations live in torchsky.sphere.spectral_ops;
# re-export the historical spectral names.
from .spectral_ops import (  # noqa: E402, F401
    anafast,
    beam2bl,
    bl2beam,
    gaussian_beam,
    pixwin,
    smoothalm,
    smoothmap,
    synalm,
    synfast,
)
