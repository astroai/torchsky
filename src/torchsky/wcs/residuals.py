"""Gauge-safe residual models for supplied-reference astrometric calibration."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

ResidualComponent = Literal["stable", "drift", "local"]


def _finite_real(value: Tensor, name: str, *, ndim: int | None = None) -> Tensor:
    tensor = torch.as_tensor(value)
    if not tensor.is_floating_point() or tensor.is_complex():
        raise ValueError(f"{name} must be a real floating-point tensor")
    if ndim is not None and tensor.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not bool(torch.isfinite(tensor).all().detach().cpu().item()):
        raise ValueError(f"{name} must contain only finite values")
    return tensor


def _vector_covariance(value: Tensor, count: int, like: Tensor, name: str) -> Tensor:
    covariance = torch.as_tensor(value, dtype=like.dtype, device=like.device)
    if covariance.shape == (2, 2):
        covariance = covariance.expand(count, 2, 2).clone()
    if covariance.shape != (count, 2, 2):
        raise ValueError(f"{name} must have shape (2, 2) or (N, 2, 2)")
    covariance = 0.5 * (covariance + covariance.mT)
    factor, info = torch.linalg.cholesky_ex(covariance)
    if bool((info != 0).any().detach().cpu().item()):
        raise ValueError(f"{name} must be symmetric positive definite")
    if not bool(torch.isfinite(factor).all().detach().cpu().item()):
        raise ValueError(f"{name} must be finite")
    return covariance


def _block_diagonal(blocks: Tensor) -> Tensor:
    count = int(blocks.shape[0])
    result = blocks.new_zeros((2 * count, 2 * count))
    index = torch.arange(count, device=blocks.device)
    rows = 2 * index
    result[rows, rows] = blocks[:, 0, 0]
    result[rows, rows + 1] = blocks[:, 0, 1]
    result[rows + 1, rows] = blocks[:, 1, 0]
    result[rows + 1, rows + 1] = blocks[:, 1, 1]
    return result


def _diagonal_blocks(matrix: Tensor, count: int) -> Tensor:
    shaped = matrix.reshape(count, 2, count, 2)
    index = torch.arange(count, device=matrix.device)
    return shaped[index, :, index, :]


def _gauge_precision_statistics(
    matrix: Tensor,
    values: Tensor,
    covariance: Tensor,
    gauge: TangentPlaneGauge,
) -> tuple[Tensor, Tensor]:
    projected_matrix = gauge.project_design(matrix)
    projected_values = gauge.project(values)
    precision_matrix = torch.linalg.solve(covariance, projected_matrix)
    precision_values = torch.linalg.solve(
        covariance, projected_values.unsqueeze(-1)
    ).squeeze(-1)
    precision_gauge = torch.linalg.solve(covariance, gauge.design)
    gauge_gram = torch.einsum("nai,naj->ij", gauge.design, precision_gauge)
    gauge_matrix = torch.einsum("nai,nak->ik", gauge.design, precision_matrix)
    gauge_values = torch.einsum("nai,na->i", gauge.design, precision_values)
    normal = torch.einsum("nak,nal->kl", projected_matrix, precision_matrix)
    information = torch.einsum("nak,na->k", projected_matrix, precision_values)
    normal = normal - gauge_matrix.mT @ torch.linalg.solve(gauge_gram, gauge_matrix)
    information = information - gauge_matrix.mT @ torch.linalg.solve(
        gauge_gram, gauge_values
    )
    return normal, information


@dataclass(frozen=True)
class TangentPlaneGauge:
    """Exact weighted projection and null-space coordinates for vector residuals."""

    design: Tensor
    precision: Tensor
    rank_rtol: float = 1.0e-10

    def __post_init__(self) -> None:
        design = _finite_real(self.design, "design", ndim=3)
        if design.shape[1] != 2 or design.shape[2] == 0:
            raise ValueError("design must have shape (N, 2, G) with G > 0")
        precision = _vector_covariance(
            self.precision, int(design.shape[0]), design, "precision"
        )
        if not math.isfinite(self.rank_rtol) or self.rank_rtol <= 0.0:
            raise ValueError("rank_rtol must be finite and positive")
        gram = torch.einsum("nai,nab,nbj->ij", design, precision, design)
        factor, info = torch.linalg.cholesky_ex(gram)
        diagonal = torch.abs(torch.diagonal(factor))
        if bool((info != 0).any().detach().cpu().item()) or bool(
            (diagonal.square().amin() <= self.rank_rtol * diagonal.square().amax())
            .detach()
            .cpu()
            .item()
        ):
            raise ValueError("gauge design is rank deficient in the supplied metric")
        object.__setattr__(self, "design", design)
        object.__setattr__(self, "precision", precision)

    @property
    def count(self) -> int:
        return int(self.design.shape[0])

    @property
    def dimension(self) -> int:
        return int(self.design.shape[2])

    def _active(self, active: Tensor | None) -> Tensor:
        if active is None:
            return torch.ones(self.count, dtype=torch.bool, device=self.design.device)
        mask = torch.as_tensor(active, dtype=torch.bool, device=self.design.device)
        if mask.shape != (self.count,):
            raise ValueError("active must have shape (N,)")
        if int(mask.sum().detach().cpu().item()) * 2 <= self.dimension:
            raise ValueError("active support is too small to define the gauge")
        return mask

    def _gram(self, active: Tensor) -> Tensor:
        design = self.design[active]
        precision = self.precision[active]
        gram = torch.einsum("nai,nab,nbj->ij", design, precision, design)
        factor, info = torch.linalg.cholesky_ex(gram)
        if bool((info != 0).any().detach().cpu().item()):
            raise ValueError("active support does not identify every gauge mode")
        return factor

    def project(self, value: Tensor, active: Tensor | None = None) -> Tensor:
        values = torch.as_tensor(
            value, dtype=self.design.dtype, device=self.design.device
        )
        if values.shape != (self.count, 2):
            raise ValueError("value must have shape (N, 2)")
        mask = self._active(active)
        factor = self._gram(mask)
        rhs = torch.einsum(
            "nai,nab,nb->i",
            self.design[mask],
            self.precision[mask],
            values[mask],
        )
        coefficient = torch.cholesky_solve(rhs[:, None], factor).squeeze(-1)
        projected = values - torch.einsum("nai,i->na", self.design, coefficient)
        return torch.where(mask[:, None], projected, torch.zeros_like(projected))

    def project_design(self, value: Tensor, active: Tensor | None = None) -> Tensor:
        values = torch.as_tensor(
            value, dtype=self.design.dtype, device=self.design.device
        )
        if values.ndim != 3 or values.shape[:2] != (self.count, 2):
            raise ValueError("value must have shape (N, 2, K)")
        mask = self._active(active)
        factor = self._gram(mask)
        rhs = torch.einsum(
            "nai,nab,nbk->ik",
            self.design[mask],
            self.precision[mask],
            values[mask],
        )
        coefficient = torch.cholesky_solve(rhs, factor)
        projected = values - torch.einsum("nai,ik->nak", self.design, coefficient)
        return torch.where(mask[:, None, None], projected, torch.zeros_like(projected))

    def projection_matrix(self, active: Tensor | None = None) -> Tensor:
        mask = self._active(active)
        selected = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        design = self.design[selected].reshape(-1, self.dimension)
        precision = _block_diagonal(self.precision[selected])
        gram = design.mT @ precision @ design
        local = torch.eye(
            design.shape[0], dtype=design.dtype, device=design.device
        ) - design @ torch.linalg.solve(gram, design.mT @ precision)
        full = design.new_zeros((2 * self.count, 2 * self.count))
        coordinates = torch.stack((2 * selected, 2 * selected + 1), dim=-1).reshape(-1)
        full[coordinates[:, None], coordinates[None, :]] = local
        return full

    def project_covariance(
        self, covariance: Tensor, active: Tensor | None = None
    ) -> Tensor:
        value = torch.as_tensor(
            covariance, dtype=self.design.dtype, device=self.design.device
        )
        if value.shape == (self.count, 2, 2):
            value = _block_diagonal(value)
        if value.shape != (2 * self.count, 2 * self.count):
            raise ValueError("covariance must have shape (N, 2, 2) or (2N, 2N)")
        projection = self.projection_matrix(active)
        result = projection @ value @ projection.mT
        return 0.5 * (result + result.mT)

    def null_basis(self, active: Tensor | None = None) -> Tensor:
        mask = self._active(active)
        selected = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        design = self.design[selected].reshape(-1, self.dimension)
        precision = _block_diagonal(self.precision[selected])
        weighted_design = precision @ design
        basis, triangular = torch.linalg.qr(weighted_design, mode="complete")
        diagonal = torch.abs(torch.diagonal(triangular[: self.dimension]))
        if bool(
            (diagonal <= self.rank_rtol * diagonal.max()).any().detach().cpu().item()
        ):
            raise ValueError("active support does not identify every gauge mode")
        return basis[:, self.dimension :]

    def coordinates(self, value: Tensor, active: Tensor | None = None) -> Tensor:
        mask = self._active(active)
        projected = self.project(value, mask)[mask].reshape(-1)
        return self.null_basis(mask).mT @ projected

    def from_coordinates(
        self, coordinates: Tensor, active: Tensor | None = None
    ) -> Tensor:
        mask = self._active(active)
        basis = self.null_basis(mask)
        values = torch.as_tensor(
            coordinates, dtype=self.design.dtype, device=self.design.device
        )
        if values.shape != (basis.shape[1],):
            raise ValueError("coordinates have the wrong null-space dimension")
        result = self.design.new_zeros((self.count, 2))
        result[mask] = (basis @ values).reshape(-1, 2)
        return result

    def gauge_residual(self, value: Tensor) -> Tensor:
        values = torch.as_tensor(
            value, dtype=self.design.dtype, device=self.design.device
        )
        if values.shape != (self.count, 2):
            raise ValueError("value must have shape (N, 2)")
        return torch.einsum("nai,nab,nb->i", self.design, self.precision, values)


@dataclass(frozen=True)
class AstrometricResidualFeatures:
    """Continuous and explicitly encoded features; identifiers are never embedded."""

    focal_xy: Tensor
    ccd_xy: Tensor
    band_encoding: Tensor
    color: Tensor
    time: Tensor
    airmass: Tensor
    parallactic_angle: Tensor
    telemetry: Tensor

    def __post_init__(self) -> None:
        focal = _finite_real(self.focal_xy, "focal_xy", ndim=2)
        if focal.shape[1] != 2:
            raise ValueError("focal_xy must have shape (N, 2)")
        count = int(focal.shape[0])
        device = focal.device
        dtype = focal.dtype
        for name, width in (
            ("ccd_xy", 2),
            ("band_encoding", None),
            ("telemetry", None),
        ):
            value = _finite_real(getattr(self, name), name, ndim=2).to(
                device=device, dtype=dtype
            )
            if value.shape[0] != count or (
                width is not None and value.shape[1] != width
            ):
                raise ValueError(f"{name} has an incompatible shape")
            object.__setattr__(self, name, value)
        for name in ("color", "time", "airmass", "parallactic_angle"):
            value = _finite_real(getattr(self, name), name, ndim=1).to(
                device=device, dtype=dtype
            )
            if value.shape != (count,):
                raise ValueError(f"{name} must have shape (N,)")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "focal_xy", focal)

    @property
    def count(self) -> int:
        return int(self.focal_xy.shape[0])

    def tensor(self) -> Tensor:
        angle = self.parallactic_angle
        return torch.cat(
            (
                self.focal_xy,
                self.ccd_xy,
                self.band_encoding,
                self.color[:, None],
                self.time[:, None],
                self.airmass[:, None],
                torch.sin(angle)[:, None],
                torch.cos(angle)[:, None],
                self.telemetry,
            ),
            dim=-1,
        )


@dataclass(frozen=True)
class ResidualModeDesign:
    stable: Tensor
    drift: Tensor
    local: Tensor
    stable_names: tuple[str, ...] = ()
    drift_names: tuple[str, ...] = ()
    local_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        stable = _finite_real(self.stable, "stable", ndim=3)
        if stable.shape[1] != 2:
            raise ValueError("mode tensors must have shape (N, 2, K)")
        for name in ("drift", "local"):
            value = _finite_real(getattr(self, name), name, ndim=3).to(
                dtype=stable.dtype, device=stable.device
            )
            if value.shape[:2] != stable.shape[:2]:
                raise ValueError("all mode tensors must share shape (N, 2, K)")
            object.__setattr__(self, name, value)
        for tensor_name, names_name in (
            ("stable", "stable_names"),
            ("drift", "drift_names"),
            ("local", "local_names"),
        ):
            width = int(getattr(self, tensor_name).shape[2])
            names = getattr(self, names_name)
            if names and len(names) != width:
                raise ValueError(f"{names_name} must name every mode")
        object.__setattr__(self, "stable", stable)

    @property
    def count(self) -> int:
        return int(self.stable.shape[0])

    @property
    def widths(self) -> tuple[int, int, int]:
        return (
            int(self.stable.shape[2]),
            int(self.drift.shape[2]),
            int(self.local.shape[2]),
        )

    @property
    def matrix(self) -> Tensor:
        return torch.cat((self.stable, self.drift, self.local), dim=-1)

    def component_slice(self, component: ResidualComponent) -> slice:
        stable, drift, local = self.widths
        if component == "stable":
            return slice(0, stable)
        if component == "drift":
            return slice(stable, stable + drift)
        return slice(stable + drift, stable + drift + local)


def interpretable_residual_modes(
    features: AstrometricResidualFeatures,
) -> ResidualModeDesign:
    """Build named affine, color, and detector residual-mode columns."""
    x, y = features.focal_xy.unbind(-1)
    quadratic = torch.stack(
        (x.square() - y.square(), 2.0 * x * y, x.square() + y.square()), dim=-1
    )
    zero = torch.zeros_like(quadratic)
    stable_x = torch.stack((quadratic, zero), dim=1)
    stable_y = torch.stack((zero, quadratic), dim=1)
    stable = torch.cat((stable_x, stable_y), dim=-1)
    time = features.time - features.time.mean()
    time_scale = torch.sqrt(torch.mean(time.square())).clamp_min(
        torch.finfo(time.dtype).eps
    )
    drivers = torch.cat((time[:, None] / time_scale, features.telemetry), dim=-1)
    drift = (stable[:, :, :, None] * drivers[:, None, None, :]).reshape(
        features.count, 2, -1
    )
    angle = features.parallactic_angle
    parallel = torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)
    perpendicular = torch.stack((parallel[:, 1], -parallel[:, 0]), dim=-1)
    atmosphere_scalar = (features.airmass - 1.0)[:, None] * quadratic[:, :2]
    local = (
        torch.stack(
            (
                atmosphere_scalar[:, :, None] * parallel[:, None, :],
                atmosphere_scalar[:, :, None] * perpendicular[:, None, :],
            ),
            dim=-1,
        )
        .permute(0, 2, 1, 3)
        .reshape(features.count, 2, -1)
    )
    stable_names = (
        "detector_x_quadrupole_plus",
        "detector_x_quadrupole_cross",
        "detector_x_radial_quadratic",
        "detector_y_quadrupole_plus",
        "detector_y_quadrupole_cross",
        "detector_y_radial_quadratic",
    )
    driver_names = ("time",) + tuple(
        f"telemetry_{index}" for index in range(features.telemetry.shape[1])
    )
    drift_names = tuple(
        f"{mode}:{driver}" for mode in stable_names for driver in driver_names
    )
    local_names = (
        "atmosphere_parallel_quadrupole_plus",
        "atmosphere_perpendicular_quadrupole_plus",
        "atmosphere_parallel_quadrupole_cross",
        "atmosphere_perpendicular_quadrupole_cross",
    )
    return ResidualModeDesign(
        stable, drift, local, stable_names, drift_names, local_names
    )


@dataclass(frozen=True)
class BayesianModeState:
    mean: Tensor
    covariance: Tensor
    widths: tuple[int, int, int]
    version: int = 1

    def __post_init__(self) -> None:
        mean = _finite_real(self.mean, "mean", ndim=1)
        covariance = _finite_real(self.covariance, "covariance", ndim=2).to(
            dtype=mean.dtype, device=mean.device
        )
        if covariance.shape != (mean.numel(), mean.numel()):
            raise ValueError("covariance has an incompatible shape")
        if sum(self.widths) != mean.numel() or min(self.widths) < 0:
            raise ValueError("widths do not partition the coefficient vector")
        _, info = torch.linalg.cholesky_ex(0.5 * (covariance + covariance.mT))
        if bool((info != 0).any().detach().cpu().item()):
            raise ValueError("coefficient covariance must be positive definite")
        if self.version < 1:
            raise ValueError("version must be positive")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", 0.5 * (covariance + covariance.mT))

    def component_slice(self, component: ResidualComponent) -> slice:
        stable, drift, local = self.widths
        if component == "stable":
            return slice(0, stable)
        if component == "drift":
            return slice(stable, stable + drift)
        return slice(stable + drift, stable + drift + local)


@dataclass(frozen=True)
class BayesianModePrediction:
    stable: Tensor
    drift: Tensor
    local: Tensor
    mean: Tensor
    covariance: Tensor


def fit_bayesian_modes(
    design: ResidualModeDesign,
    residual: Tensor,
    noise_covariance: Tensor,
    *,
    prior_scale: tuple[float, float, float] = (0.1, 0.05, 0.1),
    gauge: TangentPlaneGauge | None = None,
) -> BayesianModeState:
    """Fit residual modes with Gaussian priors and correlated noise."""
    values = _finite_real(residual, "residual", ndim=2).to(
        dtype=design.stable.dtype, device=design.stable.device
    )
    if values.shape != (design.count, 2):
        raise ValueError("residual must have shape (N, 2)")
    covariance = _vector_covariance(
        noise_covariance, design.count, values, "noise_covariance"
    )
    matrix = design.matrix
    if gauge is not None:
        if gauge.count != design.count:
            raise ValueError("gauge and mode design row counts differ")
        data_normal, data_information = _gauge_precision_statistics(
            matrix, values, covariance, gauge
        )
    else:
        factor = torch.linalg.cholesky(covariance)
        whitened_values = torch.linalg.solve_triangular(
            factor, values.unsqueeze(-1), upper=False
        ).squeeze(-1)
        whitened_matrix = torch.linalg.solve_triangular(factor, matrix, upper=False)
        flat_matrix = whitened_matrix.reshape(-1, matrix.shape[-1])
        flat_values = whitened_values.reshape(-1)
        data_normal = flat_matrix.mT @ flat_matrix
        data_information = flat_matrix.mT @ flat_values
    if len(prior_scale) != 3 or any(
        not math.isfinite(value) or value <= 0.0 for value in prior_scale
    ):
        raise ValueError("prior_scale values must be finite and positive")
    prior = torch.cat(
        tuple(
            matrix.new_full((width,), 1.0 / scale**2)
            for width, scale in zip(design.widths, prior_scale, strict=True)
        )
    )
    precision = data_normal + torch.diag(prior)
    information = data_information
    coefficient_factor = torch.linalg.cholesky(precision)
    mean = torch.cholesky_solve(information[:, None], coefficient_factor).squeeze(-1)
    posterior = torch.cholesky_inverse(coefficient_factor)
    return BayesianModeState(mean, posterior, design.widths)


def predict_bayesian_modes(
    state: BayesianModeState, design: ResidualModeDesign
) -> BayesianModePrediction:
    """Predict residual-mode means and covariance from a fitted state."""
    if state.widths != design.widths:
        raise ValueError("state and design mode widths differ")
    matrix = design.matrix
    stable_slice = state.component_slice("stable")
    drift_slice = state.component_slice("drift")
    local_slice = state.component_slice("local")
    stable = torch.einsum(
        "nak,k->na", matrix[..., stable_slice], state.mean[stable_slice]
    )
    drift = torch.einsum("nak,k->na", matrix[..., drift_slice], state.mean[drift_slice])
    local = torch.einsum("nak,k->na", matrix[..., local_slice], state.mean[local_slice])
    flat = matrix.reshape(-1, matrix.shape[-1])
    covariance = flat @ state.covariance @ flat.mT
    return BayesianModePrediction(
        stable, drift, local, stable + drift + local, covariance
    )


def update_bayesian_modes(
    state: BayesianModeState,
    design: ResidualModeDesign,
    residual: Tensor,
    noise_covariance: Tensor,
    *,
    component: ResidualComponent = "local",
) -> BayesianModeState:
    """Update one residual component while preserving the other components."""
    if state.widths != design.widths:
        raise ValueError("state and design mode widths differ")
    values = _finite_real(residual, "residual", ndim=2).to(
        dtype=state.mean.dtype, device=state.mean.device
    )
    if values.shape != (design.count, 2):
        raise ValueError("residual must have shape (N, 2)")
    covariance = _vector_covariance(
        noise_covariance, design.count, values, "noise_covariance"
    )
    selected = state.component_slice(component)
    fixed_prediction = predict_bayesian_modes(state, design).mean
    selected_matrix = design.matrix[..., selected]
    selected_mean = state.mean[selected]
    target = (
        values
        - fixed_prediction
        + torch.einsum("nak,k->na", selected_matrix, selected_mean)
    )
    factor = torch.linalg.cholesky(covariance)
    whitened_target = torch.linalg.solve_triangular(
        factor, target.unsqueeze(-1), upper=False
    ).squeeze(-1)
    whitened_matrix = torch.linalg.solve_triangular(
        factor, selected_matrix, upper=False
    ).reshape(-1, selected_matrix.shape[-1])
    prior_covariance = state.covariance[selected, selected]
    prior_precision = torch.linalg.inv(prior_covariance)
    precision = prior_precision + whitened_matrix.mT @ whitened_matrix
    information = (
        prior_precision @ selected_mean
        + whitened_matrix.mT @ whitened_target.reshape(-1)
    )
    posterior = torch.linalg.inv(precision)
    posterior_mean = posterior @ information
    mean = state.mean.clone()
    mean[selected] = posterior_mean
    result_covariance = state.covariance.clone()
    result_covariance[selected, :] = 0.0
    result_covariance[:, selected] = 0.0
    result_covariance[selected, selected] = posterior
    return BayesianModeState(mean, result_covariance, state.widths, state.version + 1)


@dataclass(frozen=True)
class ResidualSupport:
    lower: Tensor
    upper: Tensor
    max_inducing_distance: float

    def __post_init__(self) -> None:
        lower = _finite_real(self.lower, "lower", ndim=1)
        upper = _finite_real(self.upper, "upper", ndim=1).to(
            dtype=lower.dtype, device=lower.device
        )
        if upper.shape != lower.shape or not bool((upper > lower).all()):
            raise ValueError("support bounds must have matching increasing coordinates")
        if (
            not math.isfinite(self.max_inducing_distance)
            or self.max_inducing_distance <= 0
        ):
            raise ValueError("max_inducing_distance must be finite and positive")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def contains(
        self, features: Tensor, inducing: Tensor, length_scale: Tensor
    ) -> Tensor:
        values = torch.as_tensor(
            features, dtype=self.lower.dtype, device=self.lower.device
        )
        if values.ndim != 2 or values.shape[1] != self.lower.numel():
            raise ValueError("features have an incompatible shape")
        finite = torch.isfinite(values).all(dim=-1)
        bounded = ((values >= self.lower) & (values <= self.upper)).all(dim=-1)
        safe_values = torch.where(finite[:, None], values, self.lower)
        distance = torch.cdist(
            safe_values / length_scale,
            inducing.to(values) / length_scale,
        ).amin(dim=-1)
        return finite & bounded & (distance <= self.max_inducing_distance)


def _rbf_kernel(
    left: Tensor, right: Tensor, length_scale: Tensor, amplitude: Tensor
) -> Tensor:
    difference = (left[:, None, :] - right[None, :, :]) / length_scale
    return amplitude.square() * torch.exp(-0.5 * difference.square().sum(dim=-1))


def _vector_kernel(scalar: Tensor, output_covariance: Tensor) -> Tensor:
    return (
        (scalar[..., None, None] * output_covariance)
        .permute(0, 2, 1, 3)
        .reshape(scalar.shape[0] * 2, scalar.shape[1] * 2)
    )


@dataclass(frozen=True)
class SparseGPResidualPrediction:
    mean: Tensor
    covariance: Tensor
    supported: Tensor


@dataclass(frozen=True)
class SparseGPResidual:
    inducing_features: Tensor
    length_scale: Tensor
    amplitude: Tensor
    output_covariance: Tensor
    posterior_mean: Tensor
    posterior_covariance: Tensor
    inducing_covariance: Tensor
    support: ResidualSupport
    covariance_scale: float = 1.0
    jitter: float = 1.0e-8

    def __post_init__(self) -> None:
        inducing = _finite_real(self.inducing_features, "inducing_features", ndim=2)
        length = _finite_real(self.length_scale, "length_scale", ndim=1).to(inducing)
        if length.shape != (inducing.shape[1],) or not bool((length > 0).all()):
            raise ValueError("length_scale must be positive with shape (F,)")
        amplitude = torch.as_tensor(
            self.amplitude, dtype=inducing.dtype, device=inducing.device
        )
        if amplitude.numel() != 1 or not bool(
            torch.isfinite(amplitude) & (amplitude > 0)
        ):
            raise ValueError("amplitude must be finite and positive")
        output = _vector_covariance(
            self.output_covariance, 1, inducing, "output_covariance"
        )[0]
        size = 2 * inducing.shape[0]
        mean = _finite_real(self.posterior_mean, "posterior_mean", ndim=1).to(inducing)
        covariance = _finite_real(
            self.posterior_covariance, "posterior_covariance", ndim=2
        ).to(inducing)
        inducing_covariance = _finite_real(
            self.inducing_covariance, "inducing_covariance", ndim=2
        ).to(inducing)
        if mean.shape != (size,) or covariance.shape != (size, size):
            raise ValueError("posterior tensors have incompatible shapes")
        if inducing_covariance.shape != (size, size):
            raise ValueError("inducing_covariance has an incompatible shape")
        if self.support.lower.numel() != inducing.shape[1]:
            raise ValueError("support and inducing feature dimensions differ")
        if not math.isfinite(self.covariance_scale) or self.covariance_scale <= 0:
            raise ValueError("covariance_scale must be finite and positive")
        if not math.isfinite(self.jitter) or self.jitter <= 0:
            raise ValueError("jitter must be finite and positive")
        object.__setattr__(self, "inducing_features", inducing)
        object.__setattr__(self, "length_scale", length)
        object.__setattr__(self, "amplitude", amplitude.reshape(()))
        object.__setattr__(self, "output_covariance", output)
        object.__setattr__(self, "posterior_mean", mean)
        object.__setattr__(self, "posterior_covariance", covariance)
        object.__setattr__(self, "inducing_covariance", inducing_covariance)

    def predict(self, features: Tensor) -> SparseGPResidualPrediction:
        query = torch.as_tensor(
            features,
            dtype=self.inducing_features.dtype,
            device=self.inducing_features.device,
        )
        if query.ndim != 2 or query.shape[1] != self.inducing_features.shape[1]:
            raise ValueError("features have an incompatible shape")
        supported = self.support.contains(
            query, self.inducing_features, self.length_scale
        )
        safe = torch.where(supported[:, None], query, self.support.lower)
        scalar = _rbf_kernel(
            safe, self.inducing_features, self.length_scale, self.amplitude
        )
        cross = _vector_kernel(scalar, self.output_covariance)
        action = torch.linalg.solve(self.inducing_covariance, cross.mT).mT
        mean = (action @ self.posterior_mean).reshape(-1, 2)
        row_action = action.reshape(-1, 2, self.inducing_covariance.shape[0])
        row_cross = cross.reshape(-1, 2, self.inducing_covariance.shape[0])
        conditional = (
            self.amplitude.square() * self.output_covariance[None]
            - row_action @ row_cross.mT
        )
        posterior = row_action @ self.posterior_covariance @ row_action.mT
        covariance = (
            self.covariance_scale
            * 0.5
            * (conditional + posterior + (conditional + posterior).mT)
        )
        mean = torch.where(supported[:, None], mean, torch.zeros_like(mean))
        covariance = torch.where(
            supported[:, None, None], covariance, torch.zeros_like(covariance)
        )
        return SparseGPResidualPrediction(mean, covariance, supported)

    def calibrate_covariance(
        self, features: Tensor, residual: Tensor, *, min_scale: float = 1.0
    ) -> SparseGPResidual:
        prediction = self.predict(features)
        values = torch.as_tensor(
            residual, dtype=prediction.mean.dtype, device=prediction.mean.device
        )
        if values.shape != prediction.mean.shape:
            raise ValueError("residual and prediction shapes differ")
        selected = prediction.supported
        if not bool(selected.any().detach().cpu().item()):
            raise ValueError("calibration requires supported validation rows")
        difference = values[selected] - prediction.mean[selected]
        covariance = prediction.covariance[selected] / self.covariance_scale
        squared = torch.einsum(
            "ni,nij,nj->n", difference, torch.linalg.inv(covariance), difference
        )
        scale = max(float(squared.mean().detach().cpu().item()) / 2.0, min_scale)
        return replace(self, covariance_scale=scale)


def fit_sparse_gp_residual(
    features: Tensor,
    residual: Tensor,
    noise_covariance: Tensor,
    inducing_features: Tensor,
    *,
    length_scale: Tensor,
    amplitude: float = 1.0,
    output_covariance: Tensor | None = None,
    support: ResidualSupport | None = None,
    gauge: TangentPlaneGauge | None = None,
    jitter: float = 1.0e-8,
) -> SparseGPResidual:
    """Fit a sparse Gaussian-process model to tangent-plane residuals."""
    train = _finite_real(features, "features", ndim=2)
    values = _finite_real(residual, "residual", ndim=2).to(train)
    inducing = _finite_real(inducing_features, "inducing_features", ndim=2).to(train)
    if values.shape != (train.shape[0], 2) or inducing.shape[1] != train.shape[1]:
        raise ValueError("training tensors have incompatible shapes")
    noise = _vector_covariance(
        noise_covariance, int(train.shape[0]), train, "noise_covariance"
    )
    if gauge is not None:
        if gauge.count != train.shape[0]:
            raise ValueError("gauge and training row counts differ")
    length = torch.as_tensor(length_scale, dtype=train.dtype, device=train.device)
    if length.shape != (train.shape[1],) or not bool((length > 0).all()):
        raise ValueError("length_scale must be positive with shape (F,)")
    amplitude_tensor = train.new_tensor(amplitude)
    output = (
        torch.eye(2, dtype=train.dtype, device=train.device)
        if output_covariance is None
        else torch.as_tensor(output_covariance, dtype=train.dtype, device=train.device)
    )
    output = _vector_covariance(output, 1, train, "output_covariance")[0]
    scalar_uu = _rbf_kernel(inducing, inducing, length, amplitude_tensor)
    scalar_uu = scalar_uu + jitter * torch.eye(
        inducing.shape[0], dtype=train.dtype, device=train.device
    )
    inducing_covariance = _vector_kernel(scalar_uu, output)
    scalar_fu = _rbf_kernel(train, inducing, length, amplitude_tensor)
    cross = _vector_kernel(scalar_fu, output)
    action = torch.linalg.solve(inducing_covariance, cross.mT).mT
    prior_precision = torch.linalg.inv(inducing_covariance)
    action_blocks = action.reshape(train.shape[0], 2, -1)
    if gauge is None:
        precision_action = torch.linalg.solve(noise, action_blocks).reshape(
            action.shape
        )
        precision_values = torch.linalg.solve(noise, values.unsqueeze(-1)).squeeze(-1)
        data_normal = action.mT @ precision_action
        data_information = action.mT @ precision_values.reshape(-1)
    else:
        data_normal, data_information = _gauge_precision_statistics(
            action_blocks, values, noise, gauge
        )
    posterior_precision = prior_precision + data_normal
    posterior_covariance = torch.linalg.inv(posterior_precision)
    posterior_mean = posterior_covariance @ data_information
    if support is None:
        minimum = train.amin(dim=0)
        maximum = train.amax(dim=0)
        padding = torch.maximum(
            0.05 * (maximum - minimum),
            train.new_full((train.shape[1],), math.sqrt(torch.finfo(train.dtype).eps)),
        )
        support = ResidualSupport(minimum - padding, maximum + padding, 3.0)
    return SparseGPResidual(
        inducing,
        length,
        amplitude_tensor,
        output,
        posterior_mean,
        posterior_covariance,
        inducing_covariance,
        support,
        jitter=jitter,
    )


@dataclass(frozen=True)
class ConditionalResidualPrediction:
    mean: Tensor
    covariance: Tensor
    supported: Tensor
    context_count: Tensor


class ExperimentalConditionalResidual(nn.Module):
    """Experimental compact-support conditional neural process for reference context."""

    experimental = True
    support_lower: Tensor
    support_upper: Tensor

    def __init__(
        self,
        feature_dim: int,
        *,
        hidden_dim: int = 32,
        support_radius: float = 0.25,
        min_context: int = 2,
        minimum_std: float = 1.0e-4,
        query_chunk_size: int = 1024,
        support_lower: Tensor | None = None,
        support_upper: Tensor | None = None,
    ) -> None:
        super().__init__()
        if feature_dim < 2 or hidden_dim < 1 or min_context < 1 or query_chunk_size < 1:
            raise ValueError("invalid conditional residual dimensions")
        if not math.isfinite(support_radius) or support_radius <= 0.0:
            raise ValueError("support_radius must be finite and positive")
        if not math.isfinite(minimum_std) or minimum_std <= 0.0:
            raise ValueError("minimum_std must be finite and positive")
        self.feature_dim = int(feature_dim)
        self.support_radius = float(support_radius)
        self.min_context = int(min_context)
        self.minimum_std = float(minimum_std)
        self.query_chunk_size = int(query_chunk_size)
        self.context_encoder = nn.Sequential(
            nn.Linear(feature_dim + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 5),
        )
        lower = (
            torch.full((feature_dim,), -float("inf"))
            if support_lower is None
            else torch.as_tensor(support_lower)
        )
        upper = (
            torch.full((feature_dim,), float("inf"))
            if support_upper is None
            else torch.as_tensor(support_upper)
        )
        if lower.shape != (feature_dim,) or upper.shape != (feature_dim,):
            raise ValueError("support bounds must have shape (feature_dim,)")
        if not bool((upper > lower).all()):
            raise ValueError("support bounds must be increasing")
        self.register_buffer("support_lower", lower)
        self.register_buffer("support_upper", upper)

    @staticmethod
    def _wendland(distance: Tensor) -> Tensor:
        remaining = torch.clamp(1.0 - distance, min=0.0)
        return remaining.pow(4) * (4.0 * distance + 1.0)

    def _chunk(
        self,
        context_features: Tensor,
        context_encoding: Tensor,
        query_features: Tensor,
    ) -> ConditionalResidualPrediction:
        distance = (
            torch.cdist(query_features[:, :2], context_features[:, :2])
            / self.support_radius
        )
        weight = self._wendland(distance)
        count = (weight > 0.0).sum(dim=-1)
        normalization = weight.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(weight.dtype).tiny
        )
        pooled = weight @ context_encoding / normalization
        output = self.decoder(torch.cat((query_features, pooled), dim=-1))
        mean = output[:, :2]
        diagonal = F.softplus(output[:, (2, 4)]) + self.minimum_std
        lower = torch.zeros(
            (query_features.shape[0], 2, 2),
            dtype=query_features.dtype,
            device=query_features.device,
        )
        lower[:, 0, 0] = diagonal[:, 0]
        lower[:, 1, 0] = output[:, 3]
        lower[:, 1, 1] = diagonal[:, 1]
        covariance = lower @ lower.mT
        bounded = (
            (query_features >= self.support_lower)
            & (query_features <= self.support_upper)
        ).all(dim=-1)
        supported = (count >= self.min_context) & bounded
        mean = torch.where(supported[:, None], mean, torch.zeros_like(mean))
        covariance = torch.where(
            supported[:, None, None], covariance, torch.zeros_like(covariance)
        )
        return ConditionalResidualPrediction(mean, covariance, supported, count)

    def forward(
        self,
        context_features: Tensor,
        context_residual: Tensor,
        query_features: Tensor,
    ) -> ConditionalResidualPrediction:
        context = torch.as_tensor(context_features)
        residual = torch.as_tensor(
            context_residual, dtype=context.dtype, device=context.device
        )
        query = torch.as_tensor(
            query_features, dtype=context.dtype, device=context.device
        )
        if context.ndim != 2 or context.shape[1] != self.feature_dim:
            raise ValueError("context_features have an incompatible shape")
        if residual.shape != (context.shape[0], 2):
            raise ValueError("context_residual must have shape (C, 2)")
        if query.ndim != 2 or query.shape[1] != self.feature_dim:
            raise ValueError("query_features have an incompatible shape")
        if context.shape[0] == 0:
            return ConditionalResidualPrediction(
                query.new_zeros((query.shape[0], 2)),
                query.new_zeros((query.shape[0], 2, 2)),
                torch.zeros(query.shape[0], dtype=torch.bool, device=query.device),
                torch.zeros(query.shape[0], dtype=torch.int64, device=query.device),
            )
        encoding = self.context_encoder(torch.cat((context, residual), dim=-1))
        chunks = [
            self._chunk(context, encoding, query[start : start + self.query_chunk_size])
            for start in range(0, query.shape[0], self.query_chunk_size)
        ]
        return ConditionalResidualPrediction(
            torch.cat(tuple(chunk.mean for chunk in chunks)),
            torch.cat(tuple(chunk.covariance for chunk in chunks)),
            torch.cat(tuple(chunk.supported for chunk in chunks)),
            torch.cat(tuple(chunk.context_count for chunk in chunks)),
        )

    def predict(
        self,
        context_features: Tensor,
        context_residual: Tensor,
        query_features: Tensor,
        *,
        context_reference_ids: Tensor | None = None,
        query_reference_ids: Tensor | None = None,
    ) -> ConditionalResidualPrediction:
        if (context_reference_ids is None) != (query_reference_ids is None):
            raise ValueError(
                "both reference ID tensors are required for leakage checks"
            )
        if context_reference_ids is not None and query_reference_ids is not None:
            context_ids = torch.as_tensor(context_reference_ids).reshape(-1)
            query_ids = torch.as_tensor(query_reference_ids).reshape(-1)
            if (
                context_ids.numel() != context_features.shape[0]
                or query_ids.numel() != query_features.shape[0]
            ):
                raise ValueError(
                    "reference ID tensors must match context and query rows"
                )
            leaked = torch.isin(query_ids.to(context_ids.device), context_ids)
            if bool(leaked.any().detach().cpu().item()):
                raise ValueError(
                    "target reference IDs must not appear in their own context"
                )
        return self(context_features, context_residual, query_features)


@dataclass(frozen=True)
class CorrectionEnvelope:
    max_correction: float
    max_jacobian: float
    max_covariance_eigenvalue: float
    gauge_tolerance: float = 1.0e-8

    def __post_init__(self) -> None:
        for name in (
            "max_correction",
            "max_jacobian",
            "max_covariance_eigenvalue",
            "gauge_tolerance",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class ResidualComponentInput:
    name: str
    mean: Tensor
    covariance: Tensor
    supported: Tensor
    jacobian: Tensor | None = None


@dataclass(frozen=True)
class ResidualComponentResult:
    name: str
    mean: Tensor
    covariance: Tensor
    full_covariance: Tensor
    trusted: Tensor
    fallback_reason: str | None


@dataclass(frozen=True)
class ResidualLadderPrediction:
    mean: Tensor
    covariance: Tensor
    full_covariance: Tensor
    components: tuple[ResidualComponentResult, ...]


class AstrometricResidualLadder:
    """Compose global, local, and sparse astrometric residual corrections."""

    def __init__(self, gauge: TangentPlaneGauge, envelope: CorrectionEnvelope) -> None:
        self.gauge = gauge
        self.envelope = envelope

    def _gate(self, component: ResidualComponentInput) -> ResidualComponentResult:
        mean = torch.as_tensor(
            component.mean,
            dtype=self.gauge.design.dtype,
            device=self.gauge.design.device,
        )
        covariance = torch.as_tensor(
            component.covariance, dtype=mean.dtype, device=mean.device
        )
        supported = torch.as_tensor(
            component.supported, dtype=torch.bool, device=mean.device
        )
        if mean.shape != (self.gauge.count, 2):
            raise ValueError("component mean must have shape (N, 2)")
        if covariance.shape != (self.gauge.count, 2, 2):
            raise ValueError("component covariance must have shape (N, 2, 2)")
        if supported.shape != (self.gauge.count,):
            raise ValueError("component supported mask must have shape (N,)")
        finite = torch.isfinite(mean).all(dim=-1) & torch.isfinite(covariance).all(
            dim=(-2, -1)
        )
        bounded = torch.linalg.vector_norm(mean, dim=-1) <= self.envelope.max_correction
        eigenvalue = torch.linalg.eigvalsh(0.5 * (covariance + covariance.mT))[:, -1]
        trusted = (
            supported
            & finite
            & bounded
            & (eigenvalue <= self.envelope.max_covariance_eigenvalue)
        )
        if component.jacobian is not None:
            jacobian = torch.as_tensor(
                component.jacobian, dtype=mean.dtype, device=mean.device
            )
            if jacobian.shape != (self.gauge.count, 2, 2):
                raise ValueError("component jacobian must have shape (N, 2, 2)")
            trusted &= (
                torch.linalg.matrix_norm(jacobian, ord=2) <= self.envelope.max_jacobian
            )
        if int(trusted.sum().detach().cpu().item()) * 2 <= self.gauge.dimension:
            return self._fallback(component.name, mean, "insufficient_trusted_support")
        try:
            projected_mean = self.gauge.project(mean, trusted)
            projected_full = self.gauge.project_covariance(covariance, trusted)
        except ValueError:
            return self._fallback(component.name, mean, "gauge_rank_failure")
        if bool(
            (
                torch.linalg.vector_norm(projected_mean, dim=-1)
                > self.envelope.max_correction
            )
            .any()
            .detach()
            .cpu()
            .item()
        ):
            return self._fallback(component.name, mean, "projected_correction_envelope")
        gauge_error = torch.linalg.vector_norm(
            self.gauge.gauge_residual(projected_mean)
        )
        scale = torch.linalg.vector_norm(projected_mean).clamp_min(1.0)
        if bool(
            (gauge_error > self.envelope.gauge_tolerance * scale).detach().cpu().item()
        ):
            return self._fallback(component.name, mean, "gauge_envelope")
        projected_covariance = _diagonal_blocks(projected_full, self.gauge.count)
        return ResidualComponentResult(
            component.name,
            projected_mean,
            projected_covariance,
            projected_full,
            trusted,
            None,
        )

    def _fallback(
        self, name: str, like: Tensor, reason: str
    ) -> ResidualComponentResult:
        return ResidualComponentResult(
            name,
            torch.zeros_like(like),
            like.new_zeros((self.gauge.count, 2, 2)),
            like.new_zeros((2 * self.gauge.count, 2 * self.gauge.count)),
            torch.zeros(self.gauge.count, dtype=torch.bool, device=like.device),
            reason,
        )

    def predict(
        self, components: Sequence[ResidualComponentInput]
    ) -> ResidualLadderPrediction:
        results = tuple(self._gate(component) for component in components)
        if not results:
            raise ValueError("at least one residual component is required")
        mean = torch.stack(tuple(result.mean for result in results)).sum(dim=0)
        full = torch.stack(tuple(result.full_covariance for result in results)).sum(
            dim=0
        )
        return ResidualLadderPrediction(
            mean, _diagonal_blocks(full, self.gauge.count), full, results
        )


@dataclass(frozen=True)
class VersionedResidualState:
    version: int
    parent_version: int | None
    feature_version: str
    gauge_version: str
    coefficients: BayesianModeState
    replay_features: Tensor
    replay_residuals: Tensor
    replay_covariance: Tensor
    anchor_features: Tensor
    anchor_residuals: Tensor
    anchor_covariance: Tensor

    def __post_init__(self) -> None:
        if self.version < 1 or (
            self.parent_version is not None and self.parent_version >= self.version
        ):
            raise ValueError("state versions must increase monotonically")
        replay = _finite_real(self.replay_features, "replay_features", ndim=2)
        replay_residuals = _finite_real(
            self.replay_residuals, "replay_residuals", ndim=2
        ).to(replay)
        replay_covariance = _vector_covariance(
            self.replay_covariance, int(replay.shape[0]), replay, "replay_covariance"
        )
        anchors = _finite_real(self.anchor_features, "anchor_features", ndim=2).to(
            replay
        )
        anchor_residuals = _finite_real(
            self.anchor_residuals, "anchor_residuals", ndim=2
        ).to(replay)
        anchor_covariance = _vector_covariance(
            self.anchor_covariance, int(anchors.shape[0]), anchors, "anchor_covariance"
        )
        if replay_residuals.shape != (replay.shape[0], 2):
            raise ValueError("replay residuals have an incompatible shape")
        if anchors.shape[1] != replay.shape[1] or anchor_residuals.shape != (
            anchors.shape[0],
            2,
        ):
            raise ValueError("anchor tensors have incompatible shapes")
        object.__setattr__(self, "replay_features", replay)
        object.__setattr__(self, "replay_residuals", replay_residuals)
        object.__setattr__(self, "replay_covariance", replay_covariance)
        object.__setattr__(self, "anchor_features", anchors)
        object.__setattr__(self, "anchor_residuals", anchor_residuals)
        object.__setattr__(self, "anchor_covariance", anchor_covariance)

    def append_replay(
        self,
        features: Tensor,
        residuals: Tensor,
        covariance: Tensor,
        *,
        capacity: int,
    ) -> VersionedResidualState:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        new_features = torch.cat(
            (self.replay_features, features.to(self.replay_features))
        )[-capacity:]
        new_residuals = torch.cat(
            (self.replay_residuals, residuals.to(self.replay_residuals))
        )[-capacity:]
        new_covariance = torch.cat(
            (self.replay_covariance, covariance.to(self.replay_covariance))
        )[-capacity:]
        return replace(
            self,
            replay_features=new_features,
            replay_residuals=new_residuals,
            replay_covariance=new_covariance,
        )


@dataclass(frozen=True)
class ShadowUpdateResult:
    active: VersionedResidualState
    candidate: VersionedResidualState
    accepted: bool
    anchor_nll_change: float
    reason: str | None

    def activate(self) -> VersionedResidualState:
        if not self.accepted:
            raise RuntimeError("rejected shadow updates cannot be activated")
        return self.candidate


def _mode_nll(
    state: BayesianModeState,
    design: ResidualModeDesign,
    residual: Tensor,
    covariance: Tensor,
) -> Tensor:
    prediction = predict_bayesian_modes(state, design)
    difference = residual - prediction.mean
    inverse = torch.linalg.inv(covariance)
    quadratic = torch.einsum("ni,nij,nj->n", difference, inverse, difference)
    logdet = torch.logdet(covariance)
    return 0.5 * (quadratic + logdet + 2.0 * math.log(2.0 * math.pi)).mean()


def shadow_update_bayesian_modes(
    active: VersionedResidualState,
    update_design: ResidualModeDesign,
    update_residual: Tensor,
    update_covariance: Tensor,
    anchor_design: ResidualModeDesign,
    *,
    component: ResidualComponent = "local",
    max_anchor_nll_increase: float = 0.0,
) -> ShadowUpdateResult:
    """Evaluate and conditionally accept a residual-mode state update."""
    if not math.isfinite(max_anchor_nll_increase) or max_anchor_nll_increase < 0.0:
        raise ValueError("max_anchor_nll_increase must be finite and non-negative")
    candidate_coefficients = update_bayesian_modes(
        active.coefficients,
        update_design,
        update_residual,
        update_covariance,
        component=component,
    )
    before = _mode_nll(
        active.coefficients,
        anchor_design,
        active.anchor_residuals,
        active.anchor_covariance,
    )
    after = _mode_nll(
        candidate_coefficients,
        anchor_design,
        active.anchor_residuals,
        active.anchor_covariance,
    )
    change = float((after - before).detach().cpu().item())
    accepted = change <= max_anchor_nll_increase
    candidate = replace(
        active,
        version=active.version + 1,
        parent_version=active.version,
        coefficients=candidate_coefficients,
    )
    return ShadowUpdateResult(
        active,
        candidate,
        accepted,
        change,
        None if accepted else "anchor_no_forgetting_gate",
    )


__all__ = [
    "AstrometricResidualFeatures",
    "AstrometricResidualLadder",
    "BayesianModePrediction",
    "BayesianModeState",
    "ConditionalResidualPrediction",
    "CorrectionEnvelope",
    "ExperimentalConditionalResidual",
    "ResidualComponentInput",
    "ResidualComponentResult",
    "ResidualLadderPrediction",
    "ResidualModeDesign",
    "ResidualSupport",
    "ShadowUpdateResult",
    "SparseGPResidual",
    "SparseGPResidualPrediction",
    "TangentPlaneGauge",
    "VersionedResidualState",
    "fit_bayesian_modes",
    "fit_sparse_gp_residual",
    "interpretable_residual_modes",
    "predict_bayesian_modes",
    "shadow_update_bayesian_modes",
    "update_bayesian_modes",
]
