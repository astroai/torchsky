"""Matrix-free precision actions with declared linear Gaussian nuisances.

Profiling an unknown coefficient removes information; propagating externally
calibrated random-effect uncertainty is a different statistical contract. Both
use small nuisance-space factors and a caller-supplied base precision action.
No pixel covariance or implicit information-restoring regularizer is formed.
"""

from __future__ import annotations

from collections.abc import Callable
import math

import torch
from torch import Tensor


PrecisionAction = Callable[[Tensor], Tensor]


class _NuisancePrecision:
    def __init__(
        self,
        basis: Tensor,
        *,
        apply_precision: PrecisionAction,
        valid: Tensor | None,
    ) -> None:
        if not isinstance(basis, Tensor) or basis.ndim < 2:
            raise ValueError("basis must have shape (*data_shape, n_modes)")
        if basis.dtype not in (torch.float32, torch.float64):
            raise TypeError("basis must have real float32 or float64 dtype")
        if basis.device.type not in ("cpu", "cuda", "mps"):
            raise ValueError("precision supports CPU, CUDA and MPS tensors")
        if basis.device.type == "mps" and basis.dtype != torch.float32:
            raise TypeError("MPS precision requires float32 tensors")
        if not callable(apply_precision):
            raise TypeError("apply_precision must be callable")
        self.data_shape = tuple(basis.shape[:-1])
        self.dtype = basis.dtype
        self.device = basis.device
        # MPS lacks the Hermitian eigensolver. Only nuisance-space matrices
        # cross devices; science arrays and Cholesky/apply paths stay native.
        self.spectral_device = (
            torch.device("cpu") if self.device.type == "mps" else self.device
        )
        self._apply_precision = apply_precision
        if valid is None:
            self._valid = torch.ones(
                self.data_shape, dtype=torch.bool, device=self.device
            )
        else:
            if not isinstance(valid, Tensor) or valid.dtype != torch.bool:
                raise TypeError("valid must be a boolean tensor")
            if valid.device != self.device or tuple(valid.shape) != self.data_shape:
                raise ValueError("valid must match basis data shape and device")
            self._valid = valid.clone()
        self._basis = torch.where(self._valid[..., None], basis, 0.0)
        self._finite(self._basis, "basis on valid support")
        self._n_modes = int(basis.shape[-1])
        self._flat_basis = self._basis.reshape(
            math.prod(self.data_shape), self._n_modes
        )

    @staticmethod
    def _finite(tensor: Tensor, name: str) -> None:
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} must contain only finite values")

    def _compatible(self, tensor: Tensor, name: str) -> None:
        if not isinstance(tensor, Tensor):
            raise TypeError(f"{name} must be a tensor")
        if tensor.dtype != self.dtype or tensor.device != self.device:
            raise ValueError(f"{name} must match basis dtype and device")

    def _data(self, tensor: Tensor, name: str) -> Tensor:
        self._compatible(tensor, name)
        if tuple(tensor.shape) != self.data_shape:
            raise ValueError(f"{name} must match basis data shape")
        masked = torch.where(self._valid, tensor, 0.0)
        self._finite(masked, f"{name} on valid support")
        return masked

    def _base_action(self, tensor: Tensor) -> Tensor:
        result = self._apply_precision(self._data(tensor, "precision input"))
        return self._data(result, "base precision output")

    def _weighted_columns(self, columns: Tensor) -> Tensor:
        count = int(columns.shape[-1])
        if count == 0:
            return columns.clone()
        return torch.stack(
            [
                self._base_action(columns[:, index].reshape(self.data_shape)).reshape(
                    -1
                )
                for index in range(count)
            ],
            dim=-1,
        )

    @staticmethod
    def _symmetric_gram(columns: Tensor, weighted: Tensor) -> Tensor:
        gram = columns.T @ weighted
        _NuisancePrecision._finite(gram, "nuisance Gram matrix")
        if gram.numel() == 0:
            return gram
        # Matrix multiplication roundoff scales with the number of summed terms.
        scale = (columns.abs().T @ weighted.abs()).amax()
        tolerance = torch.finfo(gram.dtype).eps * max(int(columns.shape[0]), 1) * scale
        if bool(((gram - gram.T).abs() > tolerance).any()):
            raise ValueError("base precision has an asymmetric nuisance Gram matrix")
        return (gram + gram.T) * 0.5

    def __call__(self, residual: Tensor) -> Tensor:
        return self.apply(residual)

    def apply(self, residual: Tensor) -> Tensor:
        raise NotImplementedError

    def quadratic(self, residual: Tensor) -> Tensor:
        """Return ``residual.T @ W @ residual``, without likelihood normalization."""
        safe = self._data(residual, "residual")
        return torch.sum(safe * self.apply(safe))


class ProfiledNuisancePrecision(_NuisancePrecision):
    r"""Precision after profiling fixed unknown linear coefficients.

    For ``y = A x + B beta + epsilon``, this applies
    ``W0 - W0 B (B.T W0 B)^dagger B.T W0``. ``basis`` is ``B`` with
    shape ``(*data_shape, n_modes)``. The base callback must be a fixed linear,
    symmetric positive-semidefinite precision with matching shape/dtype/device.
    Small nuisance-space checks cannot establish that contract outside its span.

    ``valid`` zeros input and output support. For correlated noise the callback
    must already represent the precision of the observed measurements: masking
    a full-data precision does not marginalize missing observations.

    ``rank_rtol`` defaults to ``n_modes * finfo(dtype).eps``; eigenvalues above
    ``max(rank_atol, rank_rtol * max_abs_eigenvalue)`` are retained. This declared
    numerical-rank policy never regularizes or lifts zero eigenvalues. Rank
    depends on basis scaling: discarding a physically relevant nuisance mode
    can leave apparent precision, so choose tolerances that resolve the declared
    nuisance span. ``nuisance_rank`` reports the retained rank; discarded modes
    are not certified identifiable. An unmeasured sky/background
    gauge stays unmeasured. Negative Gram eigenvalues exceeding matrix-product
    roundoff are rejected, independently of the chosen rank tolerance.

    Full-rank blocks use Cholesky solves. Singular blocks use the small Hermitian
    Moore-Penrose inverse. Autograd is supported on a fixed-rank stratum; rank
    changes are not differentiable. Factors are built once, so rebuild after
    changing basis, mask, or callback parameters, and for a fresh autograd graph.
    Storage is O(N k + k^2), construction uses k base actions and O(N k^2)
    contractions, and each application uses one base action and O(N k + k^2).
    CPU/CUDA support float32/float64; MPS requires float32. The declared
    ``spectral_device`` is CPU for MPS and the data device otherwise. Only the
    k-by-k Gram matrix and scalar roundoff bound go to that device for
    eigenvalue/rank validation and singular Hermitian inversion. Its
    pseudoinverse returns to the data device
    through differentiable Torch transfers. Pixel arrays, base actions,
    contractions, Cholesky factors/solves and output stay on the data device.
    This explicit small-matrix policy does not enable global MPS CPU fallback
    or replace PSD/rank checks with a shifted positive-definiteness test.
    """

    treatment = "profiled_unknown"

    def __init__(
        self,
        basis: Tensor,
        *,
        apply_precision: PrecisionAction,
        valid: Tensor | None = None,
        rank_rtol: float | None = None,
        rank_atol: float = 0.0,
    ) -> None:
        super().__init__(basis, apply_precision=apply_precision, valid=valid)
        self.rank_rtol = (
            self._n_modes * torch.finfo(self.dtype).eps
            if rank_rtol is None
            else float(rank_rtol)
        )
        self.rank_atol = float(rank_atol)
        if not math.isfinite(self.rank_rtol) or self.rank_rtol < 0.0:
            raise ValueError("rank_rtol must be finite and non-negative")
        if not math.isfinite(self.rank_atol) or self.rank_atol < 0.0:
            raise ValueError("rank_atol must be finite and non-negative")
        self._weighted_basis = self._weighted_columns(self._flat_basis)
        gram = self._symmetric_gram(self._flat_basis, self._weighted_basis)
        self._cholesky: Tensor | None = None
        self._pseudoinverse: Tensor | None = None
        self.nuisance_rank = 0
        if self._n_modes:
            spectral_gram = gram.to(device=self.spectral_device)
            eigenvalues = torch.linalg.eigvalsh(spectral_gram)
            roundoff = (
                torch.finfo(self.dtype).eps
                * max(math.prod(self.data_shape), self._n_modes)
                * (self._flat_basis.abs().T @ self._weighted_basis.abs()).amax()
            ).to(device=self.spectral_device)
            if bool((eigenvalues < -roundoff).any()):
                raise ValueError(
                    "base precision has a negative nuisance Gram eigenvalue"
                )
            threshold = torch.maximum(
                spectral_gram.new_tensor(self.rank_atol),
                self.rank_rtol * eigenvalues.abs().amax(),
            )
            if bool((eigenvalues < -torch.minimum(threshold, roundoff)).any()):
                raise ValueError(
                    "rank tolerance retains a negative roundoff nuisance mode"
                )
            self.nuisance_rank = int((eigenvalues > threshold).sum().item())
            if self.nuisance_rank == self._n_modes:
                self._cholesky = torch.linalg.cholesky(gram)
            elif self.nuisance_rank:
                self._pseudoinverse = torch.linalg.pinv(
                    spectral_gram,
                    atol=self.rank_atol,
                    rtol=self.rank_rtol,
                    hermitian=True,
                ).to(device=self.device)

    def apply(self, residual: Tensor) -> Tensor:
        weighted = self._base_action(residual).reshape(-1)
        if self.nuisance_rank == 0:
            return weighted.reshape(self.data_shape)
        rhs = self._flat_basis.T @ weighted
        if self._cholesky is not None:
            coefficients = torch.cholesky_solve(rhs[:, None], self._cholesky)[:, 0]
        else:
            assert self._pseudoinverse is not None
            coefficients = self._pseudoinverse @ rhs
        return (weighted - self._weighted_basis @ coefficients).reshape(self.data_shape)


class CalibratedNuisancePrecision(_NuisancePrecision):
    r"""Woodbury precision for externally calibrated Gaussian random effects.

    A calibration independent of the read/noise term supplies coefficient mean
    ``mean`` and covariance ``Sigma = L L.T`` via ``covariance_factor=L`` of
    shape ``(n_modes, calibration_rank)``. Rank-deficient or zero covariance is
    supported directly, without inverting Sigma. With ``F=B L``, apply
    ``W0 - W0 F solve(I + F.T W0 F, F.T W0)``. Calibration here declares a
    random-effect uncertainty model, not an arbitrary penalty or sky prior.

    Call ``center(data)`` before passing data to normal-equation assembly;
    ``apply`` and ``quadratic`` always consume an already-centered residual.
    ``logdet_covariance_correction`` is ``logdet(I + F.T W0 F)``; add it to
    the base observed covariance log determinant when constructing a Gaussian
    likelihood. The base normalization remains the caller's responsibility.
    For parameter-dependent covariance the quadratic alone is insufficient.

    The callback, mask, dtype/device, factor lifetime and observed-support
    contracts are the same as for ``ProfiledNuisancePrecision``. In its factor
    construction/application costs, replace k by the number of columns of L;
    forming B L additionally costs O(N k r) for r calibration-factor columns. No
    jitter is added. Shared calibration across exposures requires one joint
    basis/precision action, rather than independent per-exposure elimination.
    """

    treatment = "calibrated_random_effect"

    def __init__(
        self,
        basis: Tensor,
        *,
        apply_precision: PrecisionAction,
        covariance_factor: Tensor,
        mean: Tensor | None = None,
        valid: Tensor | None = None,
    ) -> None:
        super().__init__(basis, apply_precision=apply_precision, valid=valid)
        self._compatible(covariance_factor, "covariance_factor")
        if (
            covariance_factor.ndim != 2
            or int(covariance_factor.shape[0]) != self._n_modes
        ):
            raise ValueError(
                "covariance_factor must have shape (n_modes, calibration_rank)"
            )
        self._finite(covariance_factor, "covariance_factor")
        if mean is None:
            mean = basis.new_zeros(self._n_modes)
        self._compatible(mean, "mean")
        if tuple(mean.shape) != (self._n_modes,):
            raise ValueError("mean must have shape (n_modes,)")
        self._finite(mean, "mean")
        self._mean_image = (self._flat_basis @ mean).reshape(self.data_shape)
        self._factor = self._flat_basis @ covariance_factor
        self._weighted_factor = self._weighted_columns(self._factor)
        gram = self._symmetric_gram(self._factor, self._weighted_factor)
        self._cholesky: Tensor | None = None
        if gram.numel():
            eigenvalues = torch.linalg.eigvalsh(gram.to(device=self.spectral_device))
            roundoff = (
                torch.finfo(self.dtype).eps
                * max(math.prod(self.data_shape), int(gram.shape[0]))
                * (self._factor.abs().T @ self._weighted_factor.abs()).amax()
            ).to(device=self.spectral_device)
            if bool((eigenvalues < -roundoff).any()):
                raise ValueError(
                    "base precision has a negative nuisance Gram eigenvalue"
                )
            system = gram + torch.eye(
                gram.shape[0], dtype=self.dtype, device=self.device
            )
            cholesky = torch.linalg.cholesky(system)
            self._cholesky = cholesky
            self.logdet_covariance_correction = 2.0 * cholesky.diagonal().log().sum()
        else:
            self.logdet_covariance_correction = basis.new_zeros(())

    def center(self, data: Tensor) -> Tensor:
        """Subtract the calibrated additive mean, preserving valid support."""
        return self._data(data, "data") - self._mean_image

    def apply(self, residual: Tensor) -> Tensor:
        weighted = self._base_action(residual).reshape(-1)
        if self._cholesky is None:
            return weighted.reshape(self.data_shape)
        rhs = self._factor.T @ weighted
        coefficients = torch.cholesky_solve(rhs[:, None], self._cholesky)[:, 0]
        return (weighted - self._weighted_factor @ coefficients).reshape(
            self.data_shape
        )


__all__ = ["ProfiledNuisancePrecision", "CalibratedNuisancePrecision"]
