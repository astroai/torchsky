"""Internal solver hooks for field fitting and discrete inverse mapmaking."""

from __future__ import annotations

from ._core import BoundObservationOperator
from ._state import (
    _scene_state_tensor,
    build_state_diagonal_operator,
    build_state_kronecker_operator,
    build_state_matrix_operator,
)
from ._types import ApplyNoiseFn


import torch
from torch import Tensor

from ...sphere.spectral import _alm_ell_array
from ..operators.bundle import ObservationOperatorBundle
from ..scenes import (
    SceneRepresentation,
)
from ..target import TargetRenderSpec
from ...device import _as_tensor, _to


def harmonic_degree_diagonal(
    *,
    lmax: int,
    mmax: int | None = None,
    power: float = 1.0,
    ell_offset: float = 0.0,
    amplitude: float = 1.0,
    include_monopole: bool = True,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> Tensor:
    ll = int(lmax)
    mm = int(ll if mmax is None else mmax)
    if ll < 0:
        raise ValueError("lmax must be non-negative")
    if mm < 0 or mm > ll:
        raise ValueError("mmax must satisfy 0 <= mmax <= lmax")
    ell = _alm_ell_array(ll, mm).to(device=device)
    diagonal = float(amplitude) * torch.pow(
        ell.to(dtype=torch.float64) + float(ell_offset),
        float(power),
    )
    if include_monopole:
        diagonal = torch.where(ell == 0, torch.ones_like(diagonal), diagonal)
    else:
        diagonal = torch.where(ell == 0, torch.zeros_like(diagonal), diagonal)
    return diagonal.to(dtype=dtype)


def build_harmonic_degree_prior_operator(
    reference_state: Tensor,
    *,
    lmax: int,
    mmax: int | None = None,
    power: float = 1.0,
    ell_offset: float = 0.0,
    amplitude: float = 1.0,
    include_monopole: bool = True,
    apply_noise: ApplyNoiseFn | None = None,
) -> BoundObservationOperator:
    """Bind a power-law harmonic prior ``amplitude * (l + ell_offset) ** -power``.

    Whitens spherical-harmonic coefficients by the inverse square root of the
    degree-dependent variance. ``reference_state`` fixes the coefficient layout
    and dtype; ``include_monopole`` controls the ``l = 0`` mode.
    """
    state = torch.as_tensor(reference_state)
    if state.ndim == 1:
        diagonal = harmonic_degree_diagonal(
            lmax=lmax,
            mmax=mmax,
            power=power,
            ell_offset=ell_offset,
            amplitude=amplitude,
            include_monopole=include_monopole,
            dtype=torch.float32,
            device=state.device,
        ).to(dtype=state.real.dtype if state.is_complex() else state.dtype)
    elif state.ndim == 2 and int(state.shape[0]) == 2:
        base = harmonic_degree_diagonal(
            lmax=lmax,
            mmax=mmax,
            power=power,
            ell_offset=ell_offset,
            amplitude=amplitude,
            include_monopole=include_monopole,
            dtype=torch.float32,
            device=state.device,
        ).to(dtype=state.real.dtype if state.is_complex() else state.dtype)
        diagonal = torch.stack([base, base], dim=0)
    else:
        raise ValueError(
            "reference_state for harmonic prior must be shape (nalm,) or (2, nalm)"
        )
    return build_state_diagonal_operator(
        state,
        diagonal,
        apply_noise=apply_noise,
    )


def harmonic_dense_covariance(
    *,
    lmax: int,
    mmax: int | None = None,
    ell_correlation_length: float = 1.0,
    diagonal_jitter: float = 1.0e-3,
    amplitude: float = 1.0,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
) -> Tensor:
    """Dense squared-exponential covariance over harmonic coefficient index.

    Entries are ``amplitude * exp(-0.5 * (dl / ell_correlation_length) ** 2)``
    plus ``diagonal_jitter`` on the diagonal, for coefficients up to ``lmax``
    and ``mmax``. Returns a ``(nalm, nalm)`` tensor.
    """
    ll = int(lmax)
    mm = int(ll if mmax is None else mmax)
    if ll < 0:
        raise ValueError("lmax must be non-negative")
    if mm < 0 or mm > ll:
        raise ValueError("mmax must satisfy 0 <= mmax <= lmax")
    if ell_correlation_length <= 0.0:
        raise ValueError("ell_correlation_length must be positive")
    ell = _to(_alm_ell_array(ll, mm), dtype=torch.float64, device=device)
    delta = ell[:, None] - ell[None, :]
    cov = float(amplitude) * torch.exp(
        -0.5 * torch.square(delta / float(ell_correlation_length))
    )
    cov = cov + float(diagonal_jitter) * torch.eye(
        int(cov.shape[0]),
        dtype=cov.dtype,
        device=cov.device,
    )
    return cov.to(dtype=dtype)


def harmonic_cl_diagonal(
    *,
    lmax: int,
    cls: Tensor,
    mmax: int | None = None,
    beam_transfer: Tensor | None = None,
    min_variance: float = 1.0e-8,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> Tensor:
    ll = int(lmax)
    mm = int(ll if mmax is None else mmax)
    if ll < 0:
        raise ValueError("lmax must be non-negative")
    if mm < 0 or mm > ll:
        raise ValueError("mmax must satisfy 0 <= mmax <= lmax")
    cl_t = _as_tensor(cls, dtype=torch.float64, device=device)
    if cl_t.ndim != 1 or int(cl_t.numel()) < ll + 1:
        raise ValueError("cls must be a 1D spectrum with length >= lmax+1")
    variance_l = cl_t[: ll + 1]
    if beam_transfer is not None:
        beam_t = _as_tensor(beam_transfer, dtype=torch.float64, device=device)
        if beam_t.ndim != 1 or int(beam_t.numel()) < ll + 1:
            raise ValueError(
                "beam_transfer must be a 1D spectrum with length >= lmax+1"
            )
        variance_l = variance_l * torch.square(beam_t[: ll + 1])
    variance_l = torch.clamp(variance_l, min=float(min_variance))
    ell = _alm_ell_array(ll, mm).to(device=variance_l.device)
    diagonal = torch.rsqrt(variance_l.index_select(0, ell))
    return diagonal.to(dtype=dtype)


def build_harmonic_dense_prior_operator(
    reference_state: Tensor,
    covariance: Tensor,
    *,
    apply_noise: ApplyNoiseFn | None = None,
    jitter: float = 1.0e-8,
) -> BoundObservationOperator:
    """Bind a Gaussian harmonic prior from a dense coefficient covariance.

    The precision ``(covariance + jitter * I) ** -1`` is factorised once by
    Cholesky and applied as a matrix operator on the flattened coefficients.
    """
    state = torch.as_tensor(reference_state)
    covariance_t = torch.as_tensor(covariance)
    if state.ndim == 1:
        nstate = int(state.numel())
        if covariance_t.shape != (nstate, nstate):
            raise ValueError("covariance shape must match flattened harmonic state")
        eye = torch.eye(nstate, dtype=covariance_t.dtype, device=covariance_t.device)
        precision = torch.linalg.inv(covariance_t + float(jitter) * eye)
        factor = torch.linalg.cholesky(precision)
        return build_state_matrix_operator(
            state,
            factor,
            apply_noise=apply_noise,
        )
    if state.ndim == 2 and int(state.shape[0]) == 2:
        nalm = int(state.shape[1])
        if covariance_t.shape == (nalm, nalm):
            covariance_t = torch.block_diag(covariance_t, covariance_t)
        elif covariance_t.shape != (2 * nalm, 2 * nalm):
            raise ValueError(
                "spin-harmonic covariance must have shape (nalm, nalm) or (2*nalm, 2*nalm)"
            )
        eye = torch.eye(
            int(covariance_t.shape[0]),
            dtype=covariance_t.dtype,
            device=covariance_t.device,
        )
        precision = torch.linalg.inv(covariance_t + float(jitter) * eye)
        factor = torch.linalg.cholesky(precision)
        return build_state_matrix_operator(
            state,
            factor,
            apply_noise=apply_noise,
        )
    raise ValueError(
        "reference_state for dense harmonic prior must be shape (nalm,) or (2, nalm)"
    )


def build_harmonic_cl_prior_operator(
    reference_state: Tensor,
    *,
    lmax: int,
    cls: Tensor | tuple[Tensor, Tensor] | list[Tensor],
    mmax: int | None = None,
    beam_transfer: Tensor | tuple[Tensor, Tensor] | list[Tensor] | None = None,
    min_variance: float = 1.0e-8,
    apply_noise: ApplyNoiseFn | None = None,
) -> BoundObservationOperator:
    """Bind a Gaussian harmonic prior with a per-multipole angular power spectrum.

    The returned operator whitens spherical-harmonic coefficients by
    ``1/sqrt(C_l)`` (optionally times a beam transfer function), for use as a
    regulariser in a joint solve. ``reference_state`` fixes the coefficient
    layout and dtype; ``min_variance`` floors the per-mode variance.
    """
    state = torch.as_tensor(reference_state)
    if state.ndim == 1:
        diagonal = harmonic_cl_diagonal(
            lmax=lmax,
            mmax=mmax,
            cls=torch.as_tensor(cls),
            beam_transfer=None
            if beam_transfer is None
            else torch.as_tensor(beam_transfer),
            min_variance=min_variance,
            dtype=state.real.dtype if state.is_complex() else state.dtype,
            device=state.device,
        )
        return build_state_diagonal_operator(
            state,
            diagonal,
            apply_noise=apply_noise,
        )
    if state.ndim == 2 and int(state.shape[0]) == 2:
        if isinstance(cls, (tuple, list)):
            if len(cls) != 2:
                raise ValueError(
                    "spin harmonic cls must be a single spectrum or a length-2 sequence"
                )
            cls_e, cls_b = cls
        else:
            cls_e = cls_b = cls
        if beam_transfer is None:
            beam_e = beam_b = None
        elif isinstance(beam_transfer, (tuple, list)):
            if len(beam_transfer) != 2:
                raise ValueError(
                    "spin harmonic beam_transfer must be a single spectrum or a length-2 sequence"
                )
            beam_e, beam_b = beam_transfer
        else:
            beam_e = beam_b = beam_transfer
        diag_e = harmonic_cl_diagonal(
            lmax=lmax,
            mmax=mmax,
            cls=torch.as_tensor(cls_e),
            beam_transfer=None if beam_e is None else torch.as_tensor(beam_e),
            min_variance=min_variance,
            dtype=state.real.dtype if state.is_complex() else state.dtype,
            device=state.device,
        )
        diag_b = harmonic_cl_diagonal(
            lmax=lmax,
            mmax=mmax,
            cls=torch.as_tensor(cls_b),
            beam_transfer=None if beam_b is None else torch.as_tensor(beam_b),
            min_variance=min_variance,
            dtype=state.real.dtype if state.is_complex() else state.dtype,
            device=state.device,
        )
        diagonal = torch.stack([diag_e, diag_b], dim=0)
        return build_state_diagonal_operator(
            state,
            diagonal,
            apply_noise=apply_noise,
        )
    raise ValueError(
        "reference_state for harmonic Cl prior must be shape (nalm,) or (2, nalm)"
    )


def build_spin_harmonic_kronecker_prior_operator(
    reference_state: Tensor,
    *,
    harmonic_factor: Tensor,
    eb_coupling: Tensor | None = None,
    apply_noise: ApplyNoiseFn | None = None,
) -> BoundObservationOperator:
    state = torch.as_tensor(reference_state)
    if state.ndim != 2 or int(state.shape[0]) != 2:
        raise ValueError(
            "spin-harmonic Kronecker prior requires reference_state shape (2, nalm)"
        )
    harmonic_t = torch.as_tensor(harmonic_factor)
    if harmonic_t.ndim == 1:
        harmonic_t = torch.diag(harmonic_t)
    if harmonic_t.ndim != 2 or int(harmonic_t.shape[1]) != int(state.shape[1]):
        raise ValueError("harmonic_factor must have shape (out_nalm, nalm) or (nalm,)")
    if eb_coupling is None:
        eb_t = torch.eye(2, dtype=harmonic_t.dtype, device=harmonic_t.device)
    else:
        eb_t = _as_tensor(eb_coupling, dtype=harmonic_t.dtype, device=harmonic_t.device)
        if eb_t.ndim != 2 or tuple(eb_t.shape) != (2, 2):
            raise ValueError("eb_coupling must have shape (2, 2)")
    return build_state_kronecker_operator(
        state,
        eb_t,
        harmonic_t,
        apply_noise=apply_noise,
    )


def teb_cls_precision_factors(
    *,
    lmax: int,
    cls_tt: Tensor,
    cls_ee: Tensor,
    cls_bb: Tensor,
    cls_te: Tensor | None = None,
    cls_tb: Tensor | None = None,
    cls_eb: Tensor | None = None,
    mmax: int | None = None,
    beam_transfer_t: Tensor | None = None,
    beam_transfer_pol: Tensor | None = None,
    min_variance: float = 1.0e-8,
    jitter: float = 1.0e-8,
    dtype: torch.dtype = torch.complex64,
    device: torch.device | None = None,
) -> Tensor:
    ll = int(lmax)
    mm = int(ll if mmax is None else mmax)
    if ll < 0:
        raise ValueError("lmax must be non-negative")
    if mm < 0 or mm > ll:
        raise ValueError("mmax must satisfy 0 <= mmax <= lmax")
    cl_tt = _as_tensor(cls_tt, dtype=torch.float64, device=device)
    cl_ee = _as_tensor(cls_ee, dtype=torch.float64, device=device)
    cl_bb = _as_tensor(cls_bb, dtype=torch.float64, device=device)
    for name, arr in (("cls_tt", cl_tt), ("cls_ee", cl_ee), ("cls_bb", cl_bb)):
        if arr.ndim != 1 or int(arr.numel()) < ll + 1:
            raise ValueError(f"{name} must be a 1D spectrum with length >= lmax+1")
    cl_te = (
        torch.zeros(ll + 1, dtype=torch.float64, device=device)
        if cls_te is None
        else _as_tensor(cls_te, dtype=torch.float64, device=device)
    )
    cl_tb = (
        torch.zeros(ll + 1, dtype=torch.float64, device=device)
        if cls_tb is None
        else _as_tensor(cls_tb, dtype=torch.float64, device=device)
    )
    cl_eb = (
        torch.zeros(ll + 1, dtype=torch.float64, device=device)
        if cls_eb is None
        else _as_tensor(cls_eb, dtype=torch.float64, device=device)
    )
    for name, arr in (("cls_te", cl_te), ("cls_tb", cl_tb), ("cls_eb", cl_eb)):
        if arr.ndim != 1 or int(arr.numel()) < ll + 1:
            raise ValueError(f"{name} must be a 1D spectrum with length >= lmax+1")
    bt = (
        torch.ones(ll + 1, dtype=torch.float64, device=device)
        if beam_transfer_t is None
        else _as_tensor(beam_transfer_t, dtype=torch.float64, device=device)
    )
    bp = (
        torch.ones(ll + 1, dtype=torch.float64, device=device)
        if beam_transfer_pol is None
        else _as_tensor(beam_transfer_pol, dtype=torch.float64, device=device)
    )
    if bt.ndim != 1 or int(bt.numel()) < ll + 1:
        raise ValueError("beam_transfer_t must be a 1D spectrum with length >= lmax+1")
    if bp.ndim != 1 or int(bp.numel()) < ll + 1:
        raise ValueError(
            "beam_transfer_pol must be a 1D spectrum with length >= lmax+1"
        )

    factors_l: list[Tensor] = []
    eye3 = torch.eye(3, dtype=torch.float64, device=device)
    for ell in range(ll + 1):
        cov = torch.tensor(
            [
                [
                    cl_tt[ell] * bt[ell] * bt[ell],
                    cl_te[ell] * bt[ell] * bp[ell],
                    cl_tb[ell] * bt[ell] * bp[ell],
                ],
                [
                    cl_te[ell] * bt[ell] * bp[ell],
                    cl_ee[ell] * bp[ell] * bp[ell],
                    cl_eb[ell] * bp[ell] * bp[ell],
                ],
                [
                    cl_tb[ell] * bt[ell] * bp[ell],
                    cl_eb[ell] * bp[ell] * bp[ell],
                    cl_bb[ell] * bp[ell] * bp[ell],
                ],
            ],
            dtype=torch.float64,
            device=device,
        )
        cov = cov + float(min_variance) * eye3
        precision = torch.linalg.inv(cov + float(jitter) * eye3)
        factors_l.append(torch.linalg.cholesky(precision))
    factors_l_t = torch.stack(factors_l, dim=0)  # (lmax+1, 3, 3)
    ell = _alm_ell_array(ll, mm).to(device=device)
    return factors_l_t.index_select(0, ell).to(dtype=dtype)


def build_teb_cls_prior_operator(
    reference_state: Tensor,
    *,
    lmax: int,
    cls_tt: Tensor,
    cls_ee: Tensor,
    cls_bb: Tensor,
    cls_te: Tensor | None = None,
    cls_tb: Tensor | None = None,
    cls_eb: Tensor | None = None,
    mmax: int | None = None,
    beam_transfer_t: Tensor | None = None,
    beam_transfer_pol: Tensor | None = None,
    min_variance: float = 1.0e-8,
    jitter: float = 1.0e-8,
    apply_noise: ApplyNoiseFn | None = None,
) -> BoundObservationOperator:
    """Bind a joint T/E/B Gaussian prior from auto and cross spectra.

    ``reference_state`` has shape ``(3, nalm)``. The per-multipole 3x3 spectra
    ``C_l`` (``cls_tt``, ``cls_ee``, ``cls_bb`` and optional cross terms) are
    converted to precision factors; ``jitter`` regularises singular blocks.
    """
    state = torch.as_tensor(reference_state)
    if state.ndim != 2 or int(state.shape[0]) != 3:
        raise ValueError("TEB Cl prior requires reference_state shape (3, nalm)")
    factors = teb_cls_precision_factors(
        lmax=lmax,
        mmax=mmax,
        cls_tt=cls_tt,
        cls_ee=cls_ee,
        cls_bb=cls_bb,
        cls_te=cls_te,
        cls_tb=cls_tb,
        cls_eb=cls_eb,
        beam_transfer_t=beam_transfer_t,
        beam_transfer_pol=beam_transfer_pol,
        min_variance=min_variance,
        jitter=jitter,
        dtype=state.dtype if state.is_complex() else torch.complex64,
        device=state.device,
    )
    target = TargetRenderSpec(
        header={
            "NAXIS": 2,
            "NAXIS1": int(state.shape[1]),
            "NAXIS2": 3,
        },
        shape=(3, int(state.shape[1])),
    )

    def _apply_operator(
        scene: SceneRepresentation, target_spec: TargetRenderSpec
    ) -> Tensor:
        del target_spec
        tensor = _scene_state_tensor(scene)
        if tuple(tensor.shape) != tuple(state.shape):
            raise ValueError("TEB Cl prior received a scene with incompatible shape")
        tensor_t = _to(tensor, dtype=factors.dtype, device=factors.device)
        mapped = torch.einsum("nij,jn->in", factors, tensor_t)
        return _to(mapped, dtype=tensor.dtype, device=tensor.device)

    def _apply_adjoint(tensor: Tensor) -> Tensor:
        mapped = torch.as_tensor(tensor)
        if tuple(mapped.shape) != tuple(state.shape):
            raise ValueError(
                "TEB Cl prior adjoint received data with incompatible shape"
            )
        mapped_t = _to(mapped, dtype=factors.dtype, device=factors.device)
        adj = torch.einsum("nji,in->jn", torch.conj(factors), mapped_t)
        return _to(adj, dtype=state.dtype, device=state.device)

    return BoundObservationOperator(
        operators=ObservationOperatorBundle(),
        target=target,
        apply_operator=_apply_operator,
        apply_adjoint=_apply_adjoint,
        apply_noise=apply_noise,
    )
