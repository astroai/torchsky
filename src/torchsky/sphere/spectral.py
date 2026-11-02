"""CPU-first spherical harmonic primitives for HEALPix maps (torch + torchsky C++)."""

import os
import math
import functools
import warnings

import torch
from torch import Tensor

from .. import _ops as _native_ops
from .._dtype import work_complex_dtype as _preferred_complex_dtype
from .._dtype import work_dtype as _preferred_real_dtype

from . import healpix as _healpix

from ._spectral_config import (  # noqa: F401
    _ALM_ELL_CACHE,
    _ALM_M_CACHE,
    _ANGLE_CACHE,
    _HAS_HEALPIX_SPIN_ALM2MAP_RING_REC,
    _HAS_HEALPIX_SPIN_INTERPOLATE_REC,
    _HAS_HEALPIX_SPIN_MAP2ALM_RING_REC,
    _LEGENDRE_REC_CACHE,
    _NEST2RING_PERM_CACHE,
    _RING2NEST_PERM_CACHE,
    _RING_ALIAS_INDEX_CACHE,
    _RING_AZIMUTH_PHASE_CACHE,
    _RING_FOURIER_MODES_CPP_MAX_M,
    _RING_FOURIER_SYNTH_CPP_MAX_M,
    _RING_GROUP_CACHE,
    _RING_LAYOUT_CACHE,
    _RING_PHI0_PHASE_CACHE,
    _RING_PIX2RING_CACHE,
    _RING_SCALAR_BASIS_CACHE,
    _RING_SCALAR_BASIS_CONCAT_CACHE,
    _RING_SMALL_DFT_MAX_NPH,
    _RING_SPIN_BASIS_CACHE,
    _RING_SPIN_CONJ_INDEX_CACHE,
    _SCALAR_ALM2MAP_MAT_CACHE,
    _SCALAR_CACHE_ENTRY_LIMIT,
    _SCALAR_MAP2ALM_MAT_CACHE,
    _SCALAR_MAX_CACHE_BYTES,
    _SCALAR_RING_AUTO_MIN_BYTES,
    _SPECTRAL_CACHE,
    _SPECTRAL_MAX_CACHE_BYTES,
    _SPECTRAL_MAX_CACHE_ENTRY_BYTES,
    _SPIN_ALM2MAP_MAT_CACHE,
    _SPIN_ALM2MAP_RING_AUTO_MIN_BYTES,
    _SPIN_CACHE_ENTRY_LIMIT,
    _SPIN_MAP2ALM_MAT_CACHE,
    _SPIN_MAP2ALM_RING_AUTO_MIN_BYTES,
    _SPIN_MAP2ALM_RING_CONCAT_CPP_ENABLE,
    _SPIN_MAX_CACHE_BYTES,
    _SPIN_RING_AUTO_MIN_BYTES,
    _SPIN_RING_CONCAT_FAST_MAX_BYTES,
    _SPIN_RING_FINALIZE_CPP_ENABLE,
    _SPIN_RING_INTEGRATE_CONCAT_CPP_ENABLE,
    _SPIN_RING_INTEGRATE_CONCAT_FAST_MAX_BYTES,
    _SPIN_RING_INTEGRATE_CONCAT_MAX_NALM,
    _SPIN_RING_INTERP_CONCAT_CPP_ENABLE,
    _SPIN_RING_INTERP_CONCAT_FAST_MAX_BYTES,
    _SPIN_RING_NEST_ENABLE,
    _SPIN_YLM_CACHE,
    _YLM_CACHE,
)
from ._spectral_weights import (  # noqa: F401
    _HEALPIX_FULL_WEIGHT_REL,
    _PIXEL_WEIGHT_CACHE,
    _RING_WEIGHT_CACHE,
    _apply_map2alm_weights,
    _even_legendre_p,
    _gal_cut_keep_mask,
    _healpix_full_pixel_weight_corrections,
    _healpix_full_weight_path,
    _healpix_ring_quadrature_weights,
    _read_healpix_full_weight_vector,
    _resolve_healpix_datapath,
)
from ._spectral_layout import (  # noqa: F401
    _alm_ell_array,
    _alm_m_array,
    _alm_to_rows,
    _angles_for_nside,
    _infer_lmax_for_fixed_mmax,
    _infer_lmax_mmax_from_nalm,
    _map_to_rows,
    _nest2ring_perm_for_nside,
    _reorder_nest_to_ring_rows,
    _reorder_ring_to_nest_rows,
    _require_sht_device,
    _ring2nest_perm_for_nside,
)
from ._spectral_basis import (  # noqa: F401
    _RING_SPIN_CONCAT_CACHE,
    _SPIN_YLM_CONCAT_CACHE,
    _build_ring_grouping,
    _build_ring_layout_for_nside,
    _build_ring_scalar_bases_for_nside,
    _build_ring_spin_bases_for_nside,
    _build_spin_ylm_basis,
    _build_spin_ylm_basis_concat,
    _legendre_l_sequence,
    _legendre_l_sequence_loop,
    _legendre_rec_coeffs,
    _ring_alias_indices,
    _ring_azimuth_phase,
    _ring_grouping_for_nside,
    _ring_layout_for_nside,
    _ring_phi0_phase_for_nside,
    _ring_scalar_bases_for_nside,
    _ring_scalar_basis_concat,
    _ring_spin_bases_for_nside,
    _ring_spin_basis_concat,
    _ring_spin_conj_indices,
    _scalar_map2alm_mat,
    _spin_alm2map_mats,
    _spin_map2alm_mats,
    _spin_ylm_basis,
    _wigner_d_element,
)
from ._spectral_autograd import (  # noqa: F401
    ScalarSHTIntegrate,
    ScalarSHTInterpolate,
    SpinSHTIntegrate,
    SpinSHTInterpolate,
    _accumulate_add,
    _scalar_ring_integrate_recurrence_cpu,
    _scalar_ring_interpolate_recurrence_cpu,
)
from ._spectral_ring_fourier import (  # noqa: F401
    _ring_fourier_modes,
    _ring_fourier_modes_spin_conj,
    _ring_fourier_synthesis,
)
from ._spectral_kernels import (  # noqa: F401
    _LEGENDRE_SCAN_MAX_ELEMS,
    _M_PER_ALM_CACHE,
    _m_per_alm_cached,
    _ring_expand_m0,
    _ring_pix2ring_index,
    _ring_spin_integrate_blocks,
    _ring_spin_interpolate_blocks,
    _scalar_alm2map_ring_torch,
    _scalar_legendre_chunks,
    _scalar_legendre_scan,
    _scalar_map2alm_ring_torch,
    _scalar_ring_legendre_integrate,
    _scalar_ring_legendre_synthesize,
    _scan_device,
    _spin_alm2map_ring_torch,
)


_SCALAR_RECURRENCE_CPP_ENABLE = (
    os.environ.get("TORCHSKY_SCALAR_MAP2ALM_RECURRENCE_CPP", "1") != "0"
)
_SCALAR_INTERP_CPP_ENABLE = (
    os.environ.get("TORCHSKY_SCALAR_ALM2MAP_RECURRENCE_CPP", "1") != "0"
)
_RING_FOURIER_CPP_ENABLE = os.environ.get("TORCHSKY_RING_FOURIER_CPP", "1") != "0"
_RING_FOURIER_MODES_CPP_ENABLE = (
    os.environ.get("TORCHSKY_RING_FOURIER_MODES_CPP", "1") != "0"
)
_RING_FOURIER_SYNTH_CPP_ENABLE = (
    os.environ.get("TORCHSKY_RING_FOURIER_SYNTH_CPP", "1") != "0"
)


def _smoke_test_torch_compile() -> bool:
    """Smoke-test torch.compile at import time so that missing CUDA headers
    (cuda.h) don't cause runtime failures later in Wigner d-matrix helpers."""
    try:

        @torch.compile
        def _test_fn(x: Tensor) -> Tensor:
            return x.sin() + x.cos()

        _test_fn(torch.randn(10, dtype=torch.float64))
        return True
    except Exception:  # capability probe: dynamo/inductor raise many unrelated types
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
        except (
            Exception
        ) as compile_error:  # compile errors are untyped; eager retry below
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
        2 * alm_size(int(lmax), int(mmax)) * _healpix.nside2npix(int(nside)) * 16
    )
    return _YLM_CACHE.get_or_build(key, build, estimated_bytes=estimated_bytes)


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
        return torch.empty((0, theta.shape[0]), dtype=theta.dtype, device=theta.device)

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


def _spin_ylm_basis_concat(
    nside: int,
    nest: bool,
    lmax: int,
    mmax: int,
    spin: int,
) -> tuple[Tensor, Tensor]:
    key = (int(nside), bool(nest), int(lmax), int(mmax), int(spin))
    bytes_needed = (
        2 * alm_size(int(lmax), int(mmax)) * _healpix.nside2npix(int(nside)) * 16
    )
    return _SPIN_YLM_CONCAT_CACHE.get_or_build(
        key,
        lambda: _build_spin_ylm_basis_concat(nside, nest, lmax, mmax, spin),
        estimated_bytes=bytes_needed,
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
        ell_vec = torch.arange(ell_start, ll + 1, dtype=real_dtype, device=theta.device)
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
    if maps_t.device.type == "mps":
        # MPS has no float64/complex128: run the spin recurrence on the host in
        # float64 (as the scalar scan does) and return float32/complex64 on MPS.
        host = map2alm_spin(
            maps_t.to(device="cpu").to(dtype=torch.float64),
            spin,
            nside=nside,
            lmax=lmax,
            mmax=mmax,
            nest=nest,
            iter=iter,
            use_weights=use_weights,
            use_pixel_weights=use_pixel_weights,
            gal_cut=gal_cut,
            datapath=datapath,
        )
        cdt = _preferred_complex_dtype(maps_t.device)
        return (
            host[0].to(dtype=cdt).to(device=maps_t.device),
            host[1].to(dtype=cdt).to(device=maps_t.device),
        )
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
    except (TypeError, ValueError, OverflowError) as exc:
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
                fused_pair: tuple[Tensor, Tensor] | None = None
                if use_fused_cpp:
                    y_plus, y_minus = _ring_spin_basis_concat(
                        int(nside), int(lmax), int(mmax), int(spin_i)
                    )
                    if y_plus.device.type == "cpu":
                        fused_pair = _native_ops.spectral_spin_map2alm_ring_concat(
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
                if fused_pair is None:
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
                    c_plus, c_minus = fused_pair
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
    if alm1.device.type == "mps" or alm2.device.type == "mps":
        mps_dev = alm1.device if alm1.device.type == "mps" else alm2.device
        host = alm2map_spin(
            (
                alm1.to(device="cpu").to(dtype=torch.complex128),
                alm2.to(device="cpu").to(dtype=torch.complex128),
            ),
            nside,
            spin,
            lmax=lmax,
            mmax=mmax,
            nest=nest,
        )
        return host.to(dtype=_preferred_real_dtype(mps_dev)).to(device=mps_dev)
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
