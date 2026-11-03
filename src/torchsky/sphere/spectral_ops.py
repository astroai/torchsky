"""Pixel-window, beam, power-spectrum, and synthesis operations.

Split out of :mod:`torchsky.sphere.spectral`; these build on the SHT
primitives there and import them lazily to keep module import acyclic.
"""

from __future__ import annotations

import math
import os

import torch
from torch import Tensor

from ..device import require_float64, resolve_device, _to
from . import healpix as _healpix
from ._pixwin import LMAX_PER_NSIDE as _PIXWIN_LMAX_PER_NSIDE
from ._pixwin import MAX_NSIDE as _PIXWIN_MAX_NSIDE
from ._pixwin import computed_pixwin as _computed_pixwin


def _spectral():
    from . import spectral

    return spectral


def anafast(
    map1: Tensor | list[float],
    map2: Tensor | list[float] | None = None,
    *,
    nside: int | None = None,
    lmax: int | None = None,
    mmax: int | None = None,
    nest: bool = False,
    nspec: int | None = None,
    iter: int = 0,
    alm: bool = False,
    pol: bool = True,
    use_weights: bool = False,
    use_pixel_weights: bool = False,
    gal_cut: float = 0.0,
    datapath: str | os.PathLike[str] | None = None,
) -> Tensor | tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
    """Compute auto- or cross-angular power spectra from full-sky maps.

    Args:
        map1: Scalar map or polarized map stack.
        map2: Optional second map for cross spectra.
        nside: Explicit HEALPix resolution, otherwise inferred from map length.
        lmax: Maximum output multipole.
        mmax: Maximum harmonic order.
        nest: Inputs use nested ordering.
        alm: Also return analyzed harmonic coefficients.
        pol: Interpret leading triplets as polarized I/Q/U maps.

    Returns:
        Power spectra, optionally together with harmonic coefficients.

    Examples:
        >>> from torchsky.sphere import anafast
        >>> callable(anafast)
        True
    """
    m1 = torch.as_tensor(map1)
    if m1.ndim not in (1, 2):
        raise ValueError("map1 must be 1D or 2D")
    if map2 is None:
        m2 = m1
    else:
        m2 = torch.as_tensor(map2)
        if m2.ndim != m1.ndim or m2.shape != m1.shape:
            raise ValueError("map2 must match map1 shape")

    npix = int(m1.shape[-1])
    ns = _healpix.npix2nside(npix) if nside is None else int(nside)
    ll = int(3 * ns - 1 if lmax is None else lmax)
    mm = ll if mmax is None else int(mmax)
    if mm < 0 or mm > ll:
        raise ValueError("invalid mmax")

    if m1.ndim == 2 and bool(pol):
        if m1.shape[0] != 3:
            raise ValueError("pol=True expects map shape (3, npix)")
        a1 = _spectral().map2alm(
            m1,
            nside=ns,
            lmax=ll,
            mmax=mm,
            nest=nest,
            iter=iter,
            pol=True,
            use_weights=use_weights,
            use_pixel_weights=use_pixel_weights,
            gal_cut=gal_cut,
            datapath=datapath,
        )
        a2 = (
            a1
            if map2 is None
            else _spectral().map2alm(
                m2,
                nside=ns,
                lmax=ll,
                mmax=mm,
                nest=nest,
                iter=iter,
                pol=True,
                use_weights=use_weights,
                use_pixel_weights=use_pixel_weights,
                gal_cut=gal_cut,
                datapath=datapath,
            )
        )
        cls = _spectral().alm2cl(a1, a2, lmax=ll, mmax=mm, nspec=nspec)
        if alm:
            return (cls, a1) if map2 is None else (cls, a1, a2)
        return cls

    if m1.ndim != 1:
        raise ValueError("pol=False torch backend currently supports only 1D maps")

    a1 = _spectral().map2alm(
        m1,
        nside=ns,
        lmax=ll,
        mmax=mm,
        nest=nest,
        iter=iter,
        pol=False,
        use_weights=use_weights,
        use_pixel_weights=use_pixel_weights,
        gal_cut=gal_cut,
        datapath=datapath,
    )
    a2 = _spectral().map2alm(
        m2,
        nside=ns,
        lmax=ll,
        mmax=mm,
        nest=nest,
        iter=iter,
        pol=False,
        use_weights=use_weights,
        use_pixel_weights=use_pixel_weights,
        gal_cut=gal_cut,
        datapath=datapath,
    )
    cl = _spectral().alm2cl(a1, a2, lmax=ll, mmax=mm, nspec=nspec)
    if alm:
        return (cl, a1) if map2 is None else (cl, a1, a2)
    return cl


def gaussian_beam(fwhm_rad: float, lmax: int, *, pol: bool = False) -> Tensor:
    """Return Gaussian beam transfer function(s) B_l."""
    if fwhm_rad < 0.0:
        raise ValueError("fwhm_rad must be >= 0")
    if lmax < 0:
        raise ValueError("lmax must be non-negative")
    sigma = float(fwhm_rad) / math.sqrt(8.0 * math.log(2.0))
    ell = torch.arange(lmax + 1, dtype=torch.float64)
    base = torch.exp(-0.5 * ell * (ell + 1.0) * (sigma * sigma))
    if not pol:
        return base
    fac_p = math.exp(2.0 * sigma * sigma)
    fac_tp = math.exp(sigma * sigma)
    return torch.stack([base, base * fac_p, base * fac_p, base * fac_tp], dim=1)


def pixwin(
    nside: int,
    *,
    pol: bool = False,
    lmax: int | None = None,
) -> Tensor | tuple[Tensor, Tensor]:
    """HEALPix pixel window function(s) ``w_l``, computed from the pixel shapes.

    ``w_l^2`` is the average over pixels of each pixel's harmonic power,
    ``(1/N) sum_p (1/Omega^2) iint_p P_l(n . n')``, the definition behind the
    HEALPix tables. It is evaluated exactly up to quadrature (better than
    ``1e-12``): a terminating hypergeometric series in the pair separation
    separates ``l`` from pixel moments that are integrated with Gauss-Legendre
    rules on each pixel shape. The polarization window uses the spin-2 kernel
    ``d^l_22`` with the frame parallel-transported across the pixel.

    The moments cost ``O(nside^2)`` once per ``nside`` (about a minute at
    ``nside=2048`` on a laptop CPU) and are cached in memory and, from
    ``nside=256``, under ``$TORCHSKY_CACHE_DIR/pixwin`` (default
    ``~/.cache/torchsky/pixwin``).

    Args:
        nside: HEALPix resolution, a power of two up to 8192.
        pol: Also return the polarization window (zero for ``l < 2``).
        lmax: Largest multipole; default ``3 nside - 1``, at most ``4 nside``.

    Returns:
        Float64 CPU tensor ``w_l`` of length ``lmax + 1``, or ``(w_T, w_P)``.

    Examples:
        >>> from torchsky.sphere import pixwin
        >>> w = pixwin(16)
        >>> w.shape, round(float(w[0]), 12)
        (torch.Size([48]), 1.0)
    """
    ns = int(nside)
    if not bool(_healpix.isnsideok(ns)):
        raise ValueError("Wrong nside value (must be a power of two).")
    if ns > _PIXWIN_MAX_NSIDE:
        raise ValueError(
            f"pixel windows are computed for nside <= {_PIXWIN_MAX_NSIDE}; "
            f"nside={ns} would cost O(nside^2) pixel-shape integrals"
        )
    ll = int((3 * ns - 1) if lmax is None else lmax)
    if ll < 0:
        raise ValueError("lmax must be non-negative")
    if ll > _PIXWIN_LMAX_PER_NSIDE * ns:
        raise ValueError(
            f"lmax={ll} exceeds {_PIXWIN_LMAX_PER_NSIDE}*nside for nside={ns}"
        )
    temperature, polarization = _computed_pixwin(ns, ll)
    if bool(pol):
        return temperature, polarization
    return temperature


def bl2beam(bl: Tensor | list[float], theta: Tensor | list[float]) -> Tensor:
    """Compute circular beam profile b(theta) from harmonic window b_l."""
    bl_t = torch.as_tensor(bl, dtype=torch.float64)
    theta_t = torch.as_tensor(theta, dtype=torch.float64)
    if bl_t.ndim != 1:
        raise ValueError("bl must be 1D")
    if bl_t.numel() == 0:
        raise ValueError("bl must be non-empty")
    x = torch.cos(theta_t.reshape(-1))
    lmax = int(bl_t.numel() - 1)
    # p contains Y_l0 = sqrt((2l+1)/4pi) * P_l
    p = _spectral()._legendre_l_sequence(0, x, lmax)
    # b(theta) = sum_l [ (2l+1)/4pi * bl * P_l ]
    #          = sum_l [ sqrt((2l+1)/4pi) * bl * Y_l0 ]
    ell = torch.arange(lmax + 1, dtype=torch.float64)
    coeff = bl_t * torch.sqrt((2.0 * ell + 1.0) / (4.0 * math.pi))
    out = coeff @ p
    return out.reshape(theta_t.shape)


def beam2bl(
    beam: Tensor | list[float], theta: Tensor | list[float], lmax: int
) -> Tensor:
    """Compute harmonic window b_l from circular beam profile b(theta)."""
    b_t = torch.as_tensor(beam, dtype=torch.float64).reshape(-1)
    th_t = torch.as_tensor(theta, dtype=torch.float64).reshape(-1)
    if b_t.numel() != th_t.numel():
        raise ValueError("beam and theta must have matching size")
    if b_t.numel() < 2:
        raise ValueError("beam/theta must contain at least 2 samples")
    ll = int(lmax)
    if ll < 0:
        raise ValueError("lmax must be non-negative")

    order = torch.argsort(th_t)
    th_s = th_t.index_select(0, order)
    b_s = b_t.index_select(0, order)
    x = torch.cos(th_s)
    # p contains Y_l0 = sqrt((2l+1)/4pi) * P_l
    p = _spectral()._legendre_l_sequence(0, x, ll)
    # bl = 2pi * integral [ b(theta) * P_l(theta) * sin(theta) dtheta ]
    #    = 2pi * sqrt(4pi/(2l+1)) * integral [ b(theta) * Y_l0(theta) * sin(theta) dtheta ]
    integrand = p * (b_s * torch.sin(th_s)).unsqueeze(0)
    dth = th_s[1:] - th_s[:-1]
    trap = 0.5 * (integrand[:, 1:] + integrand[:, :-1]) * dth.unsqueeze(0)
    ell = torch.arange(ll + 1, dtype=torch.float64)
    norm = 2.0 * math.pi * torch.sqrt((4.0 * math.pi) / (2.0 * ell + 1.0))
    return norm * torch.sum(trap, dim=1)


def _cls_order_pairs(n_fields: int, *, new: bool) -> list[tuple[int, int]]:
    if int(n_fields) <= 0:
        raise ValueError("n_fields must be positive")
    n = int(n_fields)
    pairs: list[tuple[int, int]] = []
    if bool(new):
        for d in range(n):
            for i in range(0, n - d):
                pairs.append((i, i + d))
    else:
        for i in range(n):
            for j in range(i, n):
                pairs.append((i, j))
    return pairs


def _infer_field_count_from_ncls(n_cls: int) -> int:
    if n_cls <= 0:
        raise ValueError("number of spectra must be positive")
    disc = 1 + (8 * n_cls)
    root = int(math.isqrt(disc))
    if root * root != disc:
        raise ValueError("cannot infer number of fields from number of spectra")
    n = (root - 1) // 2
    if n * (n + 1) // 2 != n_cls:
        raise ValueError("cannot infer number of fields from number of spectra")
    return int(n)


def _tensor_inputs(cls: object) -> list[Tensor]:
    if isinstance(cls, Tensor):
        return [cls]
    if isinstance(cls, (list, tuple)):
        return [c for c in cls if isinstance(c, Tensor)]
    return []


def _normalize_cls_input(
    cls: Tensor
    | list[Tensor | list[float] | None]
    | tuple[Tensor | list[float] | None, ...],
    *,
    new: bool,
) -> tuple[list[Tensor | None], list[tuple[int, int]], int]:
    if isinstance(cls, Tensor):
        if cls.ndim == 1:
            return [_to(cls, device="cpu", dtype=torch.float64)], [(0, 0)], 1
        if cls.ndim == 2:
            seq: list[Tensor | None] = [
                _to(cls[i], device="cpu", dtype=torch.float64)
                for i in range(int(cls.shape[0]))
            ]
        else:
            raise ValueError("cls tensor must be 1D or 2D")
    elif isinstance(cls, (list, tuple)):
        seq = [
            None
            if c is None
            else _to(torch.as_tensor(c), device="cpu", dtype=torch.float64)
            for c in cls
        ]
    else:
        arr = torch.as_tensor(cls, dtype=torch.float64)
        if arr.ndim != 1:
            raise ValueError("cls must be a 1D spectrum or a sequence of spectra")
        return [arr], [(0, 0)], 1

    if len(seq) == 1:
        if seq[0] is None:
            raise ValueError("single cls spectrum cannot be None")
        return [seq[0]], [(0, 0)], 1

    if len(seq) == 4:
        pairs = (
            [(0, 0), (1, 1), (2, 2), (0, 1)]
            if bool(new)
            else [(0, 0), (0, 1), (1, 1), (2, 2)]
        )
        return seq, pairs, 3

    n_fields = _infer_field_count_from_ncls(len(seq))
    return seq, _cls_order_pairs(n_fields, new=bool(new)), n_fields


def synalm(
    cls: Tensor
    | list[Tensor | list[float] | None]
    | tuple[Tensor | list[float] | None, ...],
    lmax: int | None = None,
    mmax: int | None = None,
    *,
    new: bool = False,
) -> Tensor:
    """Generate alm coefficient vector(s) from power spectra.

    The covariance factorisation and the Gaussian draw run in float64 on the
    CPU (so a seed gives the same coefficients on every device); the result is
    returned on the device of the input spectra (CPU for lists/arrays).
    """
    tensors = _tensor_inputs(cls)
    out_device = resolve_device(None, *tensors) if tensors else torch.device("cpu")
    require_float64(out_device, what="synalm/synfast")
    seq, pairs, n_fields = _normalize_cls_input(cls, new=bool(new))
    ll_data = -1
    for c in seq:
        if c is not None:
            if c.ndim != 1:
                raise ValueError("each cls spectrum must be 1D")
            ll_data = max(ll_data, int(c.numel()) - 1)
    ll = int(ll_data if (lmax is None or int(lmax) < 0) else lmax)
    if ll < 0:
        raise ValueError("cannot infer lmax from empty cls input")
    mm_in = ll if mmax is None else int(mmax)
    mm = ll if mm_in < 0 else min(mm_in, ll)

    # Build batched covariance matrices [ll+1, n_fields, n_fields]
    cov = torch.zeros((ll + 1, n_fields, n_fields), dtype=torch.float64)
    for spec, (i, j) in zip(seq, pairs, strict=False):
        if spec is None:
            continue
        take = min(ll + 1, int(spec.numel()))
        if take <= 0:
            continue
        cov[:take, i, j] = spec[:take]
        cov[:take, j, i] = spec[:take]

    # Ensure positive semi-definite and compute batched Cholesky
    # Symmetric matrices are 0.5 * (A + A^T) to ensure symmetry
    cov = 0.5 * (cov + cov.transpose(-1, -2))

    # Batch Cholesky decomposition
    # Default torch.linalg.cholesky might fail on slightly non-PSD matrices
    # We use eigh to be robust for theoretical spectra with zeros
    eigvals, eigvecs = torch.linalg.eigh(cov)
    scale = eigvals.abs().amax(dim=-1)
    tolerance = torch.finfo(eigvals.dtype).eps * max(n_fields, 1) * scale
    invalid = eigvals[..., 0] < -tolerance
    if bool(torch.any(invalid).item()):
        first_ell = int(torch.nonzero(invalid, as_tuple=False)[0, 0].item())
        raise ValueError(
            f"cls covariance must be positive semidefinite at ell={first_ell}"
        )
    eigvals = torch.clamp(eigvals, min=0.0)
    fac = eigvecs @ (
        torch.sqrt(eigvals).unsqueeze(-1)
        * torch.eye(n_fields, device=cov.device, dtype=cov.dtype)
    )
    # fac shape: [ll+1, n_fields, n_fields]

    nalm = _spectral().alm_size(ll, mm)
    # Generate all (l, m) indices in HEALPix order
    # m=0: (0,0), (1,0), ..., (ll,0)
    # m=1: (1,1), (2,1), ..., (ll,1)
    # ...
    ell_indices = []
    m_indices = []
    for m in range(mm + 1):
        ell_indices.append(torch.arange(m, ll + 1, dtype=torch.int64))
        m_indices.append(torch.full((ll - m + 1,), m, dtype=torch.int64))

    ell_idx = torch.cat(ell_indices)
    m_idx = torch.cat(m_indices)
    # ell_idx.shape == (nalm,)

    # fac shape: [ll+1, n_fields, n_fields]
    # We need fac[ell_idx] -> [nalm, n_fields, n_fields]
    fac_all = fac.index_select(0, ell_idx.to(fac.device))

    # Random vectors:
    # m=0 (m_idx == 0): Purely real
    # m > 0 (m_idx > 0): Complex
    z_real = torch.randn((nalm, n_fields), dtype=fac.dtype, device=fac.device)
    z_imag = torch.randn((nalm, n_fields), dtype=fac.dtype, device=fac.device)

    # For m=0, we only use z_real and no sqrt(2) factor
    # For m>0, we use (z_real + i*z_imag)/sqrt(2)
    # We can handle this with a scale factor per alm
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    scale = torch.where(
        m_idx > 0,
        torch.tensor(inv_sqrt2, device=fac.device),
        torch.tensor(1.0, device=fac.device),
    )

    # V = fac @ Z
    # v_real = fac_all @ (z_real * scale)
    # v_imag = fac_all @ (z_imag * scale)
    # Note: zipper matmul is [nalm, n_fields, n_fields] @ [nalm, n_fields, 1]
    v_real = torch.matmul(
        fac_all, (z_real * scale.unsqueeze(-1)).unsqueeze(-1)
    ).squeeze(-1)
    v_imag = torch.matmul(
        fac_all, (z_imag * scale.unsqueeze(-1)).unsqueeze(-1)
    ).squeeze(-1)

    # Set imaginary part to 0 for m=0
    v_imag = torch.where((m_idx > 0).unsqueeze(-1), v_imag, torch.zeros_like(v_imag))

    # Construct complex output [n_fields, nalm]
    out = torch.complex(v_real.transpose(0, 1), v_imag.transpose(0, 1))

    out = out.to(out_device)
    if n_fields == 1:
        return out[0]
    return out


def synfast(
    cls: Tensor
    | list[Tensor | list[float] | None]
    | tuple[Tensor | list[float] | None, ...],
    nside: int,
    lmax: int | None = None,
    mmax: int | None = None,
    *,
    alm: bool = False,
    pol: bool = True,
    pixwin: bool = False,
    fwhm: float = 0.0,
    sigma: float | None = None,
    new: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Draw and synthesize scalar or polarized maps from power spectra.

    Args:
        cls: Scalar spectrum or polarized spectrum sequence.
        nside: Output HEALPix resolution parameter.
        lmax: Maximum synthesized multipole.
        mmax: Maximum harmonic order.
        alm: Also return the sampled harmonic coefficients.
        pol: Interpret spectra as polarized when possible.
        pixwin: Apply the HEALPix pixel window.
        fwhm: Gaussian smoothing FWHM in radians.
        sigma: Gaussian smoothing sigma in radians, overriding ``fwhm``.
        new: Use the modern polarized-spectrum ordering.

    Returns:
        Synthesized map tensor, optionally with harmonic coefficients.

    Examples:
        >>> from torchsky.sphere import synfast
        >>> callable(synfast)
        True
    """
    alms = synalm(cls, lmax=lmax, mmax=mmax, new=new)
    rows, single = _spectral()._alm_to_rows(alms)
    ll, mm = _spectral()._infer_lmax_mmax_from_nalm(int(rows.shape[1]), lmax, mmax)

    if sigma is not None and float(fwhm) != 0.0:
        raise ValueError("provide only one of fwhm or sigma")
    fwhm_use = float(fwhm)
    if sigma is not None:
        fwhm_use = float(sigma) * math.sqrt(8.0 * math.log(2.0))

    pol_eff = bool(pol) and rows.shape[0] == 3
    rows_eff: Tensor = rows
    if pixwin:
        pixwin_loader = globals()["pixwin"]
        if pol_eff:
            pw_t, pw_p = pixwin_loader(int(nside), pol=True, lmax=ll)
            pw_beam = torch.stack([pw_t, pw_p, pw_p], dim=1)
            rows_eff = smoothalm(rows_eff, lmax=ll, mmax=mm, beam=pw_beam, pol=True)
        else:
            pw = pixwin_loader(int(nside), pol=False, lmax=ll)
            rows_eff = smoothalm(
                rows_eff,
                lmax=ll,
                mmax=mm,
                beam=torch.as_tensor(pw, dtype=torch.float64),
                pol=False,
            )
        rows_eff, _ = _spectral()._alm_to_rows(rows_eff)

    if fwhm_use != 0.0:
        rows_eff = smoothalm(rows_eff, lmax=ll, mmax=mm, fwhm_rad=fwhm_use, pol=pol_eff)
        rows_eff, _ = _spectral()._alm_to_rows(rows_eff)

    map_out = _spectral().alm2map(
        rows_eff[0] if single else rows_eff,
        nside=int(nside),
        lmax=ll,
        mmax=mm,
        nest=False,
        pol=pol_eff,
        pixwin=False,
    )
    if alm:
        alm_out = rows_eff[0] if single else rows_eff
        return map_out, alm_out
    return map_out


def smoothalm(
    alm_values: Tensor | list[complex] | list[list[complex]],
    *,
    lmax: int | None = None,
    mmax: int | None = None,
    fwhm_rad: float | None = None,
    beam: Tensor | None = None,
    pol: bool = False,
) -> Tensor:
    """Apply scalar/polarized beam smoothing to alm coefficients."""
    rows, single = _spectral()._alm_to_rows(alm_values)
    ll, mm = _spectral()._infer_lmax_mmax_from_nalm(int(rows.shape[1]), lmax, mmax)

    if pol:
        if rows.shape[0] != 3:
            raise ValueError("pol=True requires alm_values shape (3, nalm)")
        if beam is None:
            if fwhm_rad is None:
                raise ValueError("provide either beam or fwhm_rad")
            beam_t = gaussian_beam(float(fwhm_rad), ll, pol=True)
        else:
            beam_t = torch.as_tensor(beam, dtype=torch.float64)
        if beam_t.ndim == 1:
            bt = be = bb = beam_t
        elif beam_t.ndim == 2:
            if beam_t.shape[0] >= ll + 1:
                bt = beam_t[:, 0]
                be = beam_t[:, 1] if beam_t.shape[1] > 1 else beam_t[:, 0]
                bb = beam_t[:, 2] if beam_t.shape[1] > 2 else be
            elif beam_t.shape[1] >= ll + 1:
                bt = beam_t[0]
                be = beam_t[1] if beam_t.shape[0] > 1 else beam_t[0]
                bb = beam_t[2] if beam_t.shape[0] > 2 else be
            else:
                raise ValueError("beam must contain length >= lmax+1")
        else:
            raise ValueError("beam must be 1D or 2D")
        if bt.shape[0] < ll + 1 or be.shape[0] < ll + 1 or bb.shape[0] < ll + 1:
            raise ValueError("beam must contain length >= lmax+1")
        ell = _spectral()._alm_ell_array(ll, mm)
        st = bt.index_select(0, ell).to(torch.complex128)
        se = be.index_select(0, ell).to(torch.complex128)
        sb = bb.index_select(0, ell).to(torch.complex128)
        return torch.stack([rows[0] * st, rows[1] * se, rows[2] * sb], dim=0)

    if beam is None:
        if fwhm_rad is None:
            raise ValueError("provide either beam or fwhm_rad")
        beam_t = gaussian_beam(float(fwhm_rad), ll)
    else:
        beam_t = torch.as_tensor(beam, dtype=torch.float64)
        if beam_t.ndim != 1 or beam_t.shape[0] < ll + 1:
            raise ValueError("beam must be 1D with length >= lmax+1")
    ell = _spectral()._alm_ell_array(ll, mm)
    scale = beam_t.index_select(0, ell).to(dtype=torch.complex128)
    out = rows * scale.unsqueeze(0)
    return out[0] if single else out


def smoothmap(
    map_values: Tensor | list[float] | list[list[float]],
    *,
    nside: int | None = None,
    lmax: int | None = None,
    mmax: int | None = None,
    nest: bool = False,
    fwhm_rad: float | None = None,
    sigma: float | None = None,
    beam: Tensor | None = None,
    beam_window: Tensor | None = None,
    pol: bool = False,
    iter: int = 0,
    use_weights: bool = False,
    use_pixel_weights: bool = False,
    gal_cut: float = 0.0,
    datapath: str | os.PathLike[str] | None = None,
) -> Tensor:
    """Smooth HEALPix map(s) with scalar or polarized beam."""
    rows, single = _spectral()._map_to_rows(map_values)
    npix = int(rows.shape[1])
    ns = _healpix.npix2nside(npix) if nside is None else int(nside)
    ll = int(3 * ns - 1 if lmax is None else lmax)
    mm = ll if mmax is None else int(mmax)
    if mm < 0 or mm > ll:
        raise ValueError("invalid mmax")
    if sigma is not None and fwhm_rad is not None:
        raise ValueError("provide only one of fwhm_rad/sigma")
    if beam is not None and beam_window is not None:
        raise ValueError("provide only one of beam/beam_window")
    beam_use = beam if beam_window is None else beam_window
    fwhm_use = fwhm_rad
    if sigma is not None:
        fwhm_use = float(sigma) * math.sqrt(8.0 * math.log(2.0))

    if pol:
        if rows.shape[0] != 3:
            raise ValueError("pol=True requires map_values shape (3, npix)")
        alm = _spectral().map2alm(
            rows,
            nside=ns,
            lmax=ll,
            mmax=mm,
            nest=nest,
            iter=iter,
            pol=True,
            use_weights=use_weights,
            use_pixel_weights=use_pixel_weights,
            gal_cut=gal_cut,
            datapath=datapath,
        )
        alm_sm = smoothalm(
            alm, lmax=ll, mmax=mm, fwhm_rad=fwhm_use, beam=beam_use, pol=True
        )
        return _spectral().alm2map(
            alm_sm,
            nside=ns,
            lmax=ll,
            mmax=mm,
            nest=nest,
            pol=True,
            pixwin=False,
        )

    alm = _spectral().map2alm(
        rows,
        nside=ns,
        lmax=ll,
        mmax=mm,
        nest=nest,
        iter=iter,
        pol=False,
        use_weights=use_weights,
        use_pixel_weights=use_pixel_weights,
        gal_cut=gal_cut,
        datapath=datapath,
    )
    alm_sm = smoothalm(
        alm, lmax=ll, mmax=mm, fwhm_rad=fwhm_use, beam=beam_use, pol=False
    )
    sm = _spectral().alm2map(
        alm_sm,
        nside=ns,
        lmax=ll,
        mmax=mm,
        nest=nest,
        pol=False,
        pixwin=False,
    )
    return sm[0] if single else sm
