"""Small, fixed linear queries on native Gaussian likelihood factors.

Templates live in measurement space. A forced flux/blend query does not need
an instantiated sky grid, a global inverse or a catalog-shaped scene. The
complementary scene, response, noise and nuisance treatment are conditional
inputs; changing them requires rebuilding the query.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import math

import torch
from torch import Tensor

from .evidence import EvidenceSnapshot, _GuardedEvidence
from .likelihood import GaussianLikelihoodSystem, NuisanceTreatment


@dataclass(frozen=True)
class LinearQueryResult(_GuardedEvidence):
    """Signed coefficients and fixed-model sampling covariance for real queries.

    ``mean_response`` is the coefficient transfer matrix ``(F + P)^-1 F``.
    A finite covariance after regularization does not identify an unsupported
    mode. ``data_rank``, ``rank_cutoff`` and the transfer retain that distinction.
    ``information_eigenvalues`` records the spectrum before any roundoff repair.
    No uncertainty in the fixed templates/complement or selection is inferred.
    """

    coefficients: Tensor
    information: Tensor
    data_rhs: Tensor
    sampling_covariance: Tensor
    mean_response: Tensor
    penalty_precision: Tensor
    data_rank: int
    information_eigenvalues: Tensor
    rank_cutoff: Tensor
    nuisance_treatments: tuple[NuisanceTreatment, ...]
    _evidence_guard: Callable[[], None] | None = field(
        default=None, repr=False, compare=False
    )
    _guarded_fields = frozenset(
        {
            "coefficients",
            "information",
            "data_rhs",
            "sampling_covariance",
            "mean_response",
            "penalty_precision",
            "data_rank",
            "information_eigenvalues",
            "rank_cutoff",
        }
    )


def _real_scalar(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _symmetric_psd(
    matrix: Tensor, *, name: str, rtol: float, atol: float
) -> tuple[Tensor, Tensor, Tensor]:
    # MPS lacks SVD/Hermitian spectra: analyze only the compact query matrix
    # on CPU, while measurement contractions and Cholesky solves stay native.
    analysis = matrix.to(device="cpu") if matrix.device.type == "mps" else matrix
    scale = torch.linalg.matrix_norm(analysis, ord=2)
    cutoff = scale * rtol + atol
    roundoff = scale * matrix.shape[0] * torch.finfo(matrix.dtype).eps
    if bool(torch.linalg.matrix_norm(analysis - analysis.T, ord=2) > roundoff):
        raise ValueError(f"{name} is not symmetric within floating-point roundoff")
    symmetric = (matrix + matrix.T) * 0.5
    spectral_symmetric = (analysis + analysis.T) * 0.5
    eigenvalues = torch.linalg.eigvalsh(spectral_symmetric)
    if bool(eigenvalues[0] < -roundoff):
        raise ValueError(f"{name} is not positive semidefinite")
    if bool(eigenvalues[0] < 0):
        # Rank thresholds cannot authorize indefinite information. Only repair
        # a negative eigenvalue within the independent floating-point allowance.
        values, vectors = torch.linalg.eigh(spectral_symmetric)
        symmetric = ((vectors * values.clamp_min(0)) @ vectors.T).to(
            device=matrix.device
        )
    return (
        symmetric,
        eigenvalues.to(device=matrix.device),
        cutoff.to(device=matrix.device),
    )


def fit_linear_query(
    system: GaussianLikelihoodSystem,
    templates: tuple[Tensor, ...],
    *,
    offset_predictions: tuple[Tensor, ...] | None = None,
    penalty_precision: Tensor | None = None,
    rank_rtol: float | None = None,
    rank_atol: float = 0.0,
    dependencies: EvidenceSnapshot | None = None,
) -> LinearQueryResult:
    """Fit a small real coefficient vector against native measurements.

    Each template tensor has shape ``(n_parameters, *factor.data.shape)`` and
    the factor's dtype/device. Complex templates are allowed for real-valued
    coefficients, using the factor's real-coordinate precision convention.
    Offsets describe the fixed complementary scene, after calibrated mean
    centering. The exact linear model is ``offset + sum_j theta_j template_j``.
    Nonlinear Jacobians give a conditional local approximation, not an exact
    nonlinear confidence region.

    Accumulate full ``F = sum J^* W J`` and ``g = sum J^* W(y-offset)``. With
    explicitly supplied PSD penalty ``P``, solve ``(F+P) theta=g`` and report
    sampling covariance ``(F+P)^-1 F (F+P)^-1`` and mean transfer. No positivity
    constraint, automatic ridge or diagonal substitution is introduced.

    A numerically deficient solve raises; a supplied penalty may resolve the
    solve while leaving ``data_rank < n_parameters``. The numerical rank cutoff
    is ``rank_atol + rank_rtol ||matrix||_2``; default relative tolerance is
    ``n_parameters * eps``. This is not a scientific identifiability threshold.
    Cholesky is used, without jitter or a matrix inverse. Symmetry/PSD validation
    uses an independent floating-point allowance, not the chosen rank cutoff;
    negative eigenvalues within that allowance alone are repaired to zero.
    Per factor, use p+1 precision actions and O(N p^2) contractions, then an
    O(p^3) small factorization. Callback costs may be superlinear in N. Supply
    bounded stamp/blend templates rather than dense whole-survey ones.

    MPS Float32 uses CPU only for compact query-matrix spectral norms,
    symmetry/PSD/rank checks and roundoff repair, with differentiable Torch
    transfers. Reports return to the input device; measurement arrays/actions,
    contractions and Cholesky solves stay there. CPU/CUDA placement is unchanged.
    No global MPS fallback or altered tolerance/regularization is enabled.

    Source tensor versions and factor dependencies guard the returned cached
    statistics. Declare other mutable query state through ``dependencies``;
    inference tensors require its explicit revision. Refit after any change.
    """
    if not isinstance(system, GaussianLikelihoodSystem):
        raise TypeError("system must be a GaussianLikelihoodSystem")
    system.validate_evidence()
    if not isinstance(templates, tuple) or len(templates) != len(system.factors):
        raise ValueError("templates must have one tensor per likelihood factor")
    if offset_predictions is not None and (
        not isinstance(offset_predictions, tuple)
        or len(offset_predictions) != len(system.factors)
    ):
        raise ValueError("offset_predictions must have one tensor per factor")
    if dependencies is not None and not isinstance(dependencies, EvidenceSnapshot):
        raise TypeError("dependencies must be an EvidenceSnapshot")
    query_tensors = {
        f"template[{index}]": value
        for index, value in enumerate(templates)
        if isinstance(value, Tensor)
    }
    if offset_predictions is not None:
        query_tensors.update(
            {
                f"offset[{index}]": value
                for index, value in enumerate(offset_predictions)
                if isinstance(value, Tensor)
            }
        )
    if isinstance(penalty_precision, Tensor):
        query_tensors["penalty_precision"] = penalty_precision
    snapshot = (
        EvidenceSnapshot.capture(query_tensors)
        if dependencies is None
        else dependencies.with_tensors(query_tensors)
    )

    def validate_query() -> None:
        system.validate_evidence()
        snapshot.validate()

    atol = _real_scalar(rank_atol, name="rank_atol")
    parameter_count: int | None = None
    information: Tensor | None = None
    data_rhs: Tensor | None = None
    for index, (factor, template) in enumerate(
        zip(system.factors, templates, strict=True)
    ):
        if not isinstance(template, Tensor) or (
            template.ndim != factor.data.ndim + 1
            or template.shape[1:] != factor.data.shape
            or template.shape[0] == 0
            or template.dtype != factor.data.dtype
            or template.device != factor.data.device
        ):
            raise ValueError("templates must match measurement shape, dtype and device")
        if not bool(torch.isfinite(template).all()):
            raise ValueError("templates must be finite on the declared support")
        if parameter_count is None:
            parameter_count = template.shape[0]
        if template.shape[0] != parameter_count:
            raise ValueError("all templates must use the same parameter count")
        if information is not None and (
            information.dtype != template.real.dtype
            or information.device != template.device
        ):
            raise ValueError(
                "factors must share real dtype and device for a joint query"
            )
        residual = factor.data
        if offset_predictions is not None:
            offset = offset_predictions[index]
            if not isinstance(offset, Tensor) or (
                offset.shape != factor.data.shape
                or offset.dtype != factor.data.dtype
                or offset.device != factor.data.device
                or not bool(torch.isfinite(offset).all())
            ):
                raise ValueError(
                    "offset measurement input must match shape, dtype and device"
                )
            residual = residual - offset
        weighted_residual = factor.precision(residual).reshape(-1)
        weighted = torch.stack(
            [factor.precision(column).reshape(-1) for column in template]
        )
        flattened = template.reshape(parameter_count, -1)
        contribution = (flattened.conj() @ weighted.T).real
        rhs = (flattened.conj() @ weighted_residual).real
        information = (
            contribution if information is None else information + contribution
        )
        data_rhs = rhs if data_rhs is None else data_rhs + rhs
    assert information is not None and data_rhs is not None
    assert parameter_count is not None
    rtol = (
        parameter_count * torch.finfo(information.dtype).eps
        if rank_rtol is None
        else _real_scalar(rank_rtol, name="rank_rtol")
    )
    if not bool(torch.isfinite(information).all() and torch.isfinite(data_rhs).all()):
        raise RuntimeError("query accumulation produced non-finite information")
    information, eigenvalues, cutoff = _symmetric_psd(
        information, name="query information", rtol=rtol, atol=atol
    )
    data_rank = int((eigenvalues > cutoff).sum().item())
    penalty = torch.zeros_like(information)
    if penalty_precision is not None:
        if not isinstance(penalty_precision, Tensor) or (
            penalty_precision.shape != information.shape
            or penalty_precision.dtype != information.dtype
            or penalty_precision.device != information.device
            or not bool(torch.isfinite(penalty_precision).all())
        ):
            raise ValueError("penalty_precision must match the real information matrix")
        penalty, _, _ = _symmetric_psd(
            penalty_precision, name="penalty_precision", rtol=rtol, atol=atol
        )
    system_matrix, system_eigenvalues, system_cutoff = _symmetric_psd(
        information + penalty, name="query system", rtol=rtol, atol=atol
    )
    if int((system_eigenvalues > system_cutoff).sum().item()) != parameter_count:
        raise RuntimeError("query is numerically unidentifiable; no automatic ridge")
    cholesky, info = torch.linalg.cholesky_ex(system_matrix)
    if int(info.item()) != 0:
        raise RuntimeError("query system Cholesky failed; no automatic jitter")
    coefficients = torch.cholesky_solve(data_rhs[:, None], cholesky)[:, 0]
    transfer = torch.cholesky_solve(information, cholesky)
    covariance = torch.cholesky_solve(transfer.T, cholesky)
    covariance = (covariance + covariance.T) * 0.5
    if not bool(
        torch.isfinite(coefficients).all()
        and torch.isfinite(covariance).all()
        and torch.isfinite(transfer).all()
    ):
        raise RuntimeError("query solve produced non-finite output")
    validate_query()
    return LinearQueryResult(
        coefficients=coefficients,
        information=information,
        data_rhs=data_rhs,
        sampling_covariance=covariance,
        mean_response=transfer,
        penalty_precision=penalty,
        data_rank=data_rank,
        information_eigenvalues=eigenvalues,
        rank_cutoff=cutoff,
        nuisance_treatments=tuple(f.nuisance_treatment for f in system.factors),
        _evidence_guard=validate_query,
    )


__all__ = ["LinearQueryResult", "fit_linear_query"]
