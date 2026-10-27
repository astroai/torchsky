"""Coupled supplied-catalog astrometric and photometric calibration."""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import torch
from torch import Tensor

from .astrometry import (
    AstrometricMatchGraph,
    AstrometryConfig,
    AstrometrySolution,
)
from ._graph import _UnionFind

MissingColorPolicy = Literal["error", "zero", "drop"]


def _finite(value: Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all().detach().cpu().item()):
        raise ValueError(f"{name} must contain only finite values")


def _measurement_covariance(value: Any, flux: Tensor) -> Tensor:
    covariance = torch.as_tensor(value, dtype=flux.dtype, device=flux.device)
    n = int(flux.numel())
    if covariance.ndim == 0:
        covariance = torch.eye(n, dtype=flux.dtype, device=flux.device) * covariance
    elif covariance.shape == (n,):
        covariance = torch.diag(covariance)
    elif covariance.shape != (n, n):
        raise ValueError("flux_covariance must be scalar, (N,), or (N, N)")
    covariance = 0.5 * (covariance + covariance.mT)
    _finite(covariance, "flux_covariance")
    _, info = torch.linalg.cholesky_ex(covariance)
    if bool((info != 0).any().detach().cpu().item()):
        raise ValueError("flux_covariance must be symmetric positive definite")
    return covariance


@dataclass(frozen=True)
class PhotometricMeasurementBlock:
    exposure_id: Hashable
    source_ids: Tensor
    band: Hashable
    flux: Tensor
    flux_covariance: Tensor
    detector_id: Hashable = "detector"
    extinction_basis: Tensor | float | None = None
    cloud_basis: Tensor | float | None = None
    star_flat_basis: Tensor | None = None
    weights: Tensor | None = None
    fit_mask: Tensor | None = None
    epoch: Tensor | float | None = None

    def __post_init__(self) -> None:
        flux = torch.as_tensor(self.flux)
        if not flux.is_floating_point() or flux.ndim != 1 or flux.numel() == 0:
            raise ValueError("flux must be a non-empty real floating-point vector")
        _finite(flux, "flux")
        if not bool((flux > 0).all().detach().cpu().item()):
            raise ValueError("calibration flux measurements must be positive")
        source_ids = torch.as_tensor(
            self.source_ids, dtype=torch.int64, device=flux.device
        )
        if source_ids.shape != flux.shape:
            raise ValueError("source_ids and flux must have matching shape")
        covariance = _measurement_covariance(self.flux_covariance, flux)
        n = int(flux.numel())
        for name in ("extinction_basis", "cloud_basis"):
            value = getattr(self, name)
            if value is None:
                tensor = torch.zeros(n, dtype=flux.dtype, device=flux.device)
            else:
                tensor = torch.as_tensor(value, dtype=flux.dtype, device=flux.device)
                tensor = torch.broadcast_to(tensor, (n,)).clone()
                _finite(tensor, name)
            object.__setattr__(self, name, tensor)
        if self.star_flat_basis is None:
            star_flat = torch.empty((n, 0), dtype=flux.dtype, device=flux.device)
        else:
            star_flat = torch.as_tensor(
                self.star_flat_basis, dtype=flux.dtype, device=flux.device
            )
            if star_flat.ndim != 2 or star_flat.shape[0] != n:
                raise ValueError("star_flat_basis must have shape (N, K)")
            _finite(star_flat, "star_flat_basis")
        if self.weights is None:
            weights = torch.ones(n, dtype=flux.dtype, device=flux.device)
        else:
            weights = torch.as_tensor(
                self.weights, dtype=flux.dtype, device=flux.device
            )
            if weights.shape != (n,):
                raise ValueError("weights must have shape (N,)")
            _finite(weights, "weights")
            if not bool((weights >= 0).all().detach().cpu().item()):
                raise ValueError("weights must be non-negative")
        if self.fit_mask is None:
            fit_mask = torch.ones(n, dtype=torch.bool, device=flux.device)
        else:
            fit_mask = torch.as_tensor(
                self.fit_mask, dtype=torch.bool, device=flux.device
            )
            if fit_mask.shape != (n,):
                raise ValueError("fit_mask must have shape (N,)")
        if self.epoch is not None:
            epoch = torch.as_tensor(self.epoch, dtype=flux.dtype, device=flux.device)
            epoch = torch.broadcast_to(epoch, (n,)).clone()
            _finite(epoch, "epoch")
            object.__setattr__(self, "epoch", epoch)
        object.__setattr__(self, "source_ids", source_ids)
        object.__setattr__(self, "flux", flux)
        object.__setattr__(self, "flux_covariance", covariance)
        object.__setattr__(self, "star_flat_basis", star_flat)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "fit_mask", fit_mask)

    @property
    def count(self) -> int:
        return int(self.flux.numel())


@dataclass(frozen=True)
class PhotometricReference:
    source_ids: Tensor
    bands: tuple[Hashable, ...]
    flux: Tensor
    flux_variance: Tensor

    def __post_init__(self) -> None:
        flux = torch.as_tensor(self.flux)
        if not flux.is_floating_point() or flux.ndim != 2:
            raise ValueError("reference flux must have shape (N, B)")
        if flux.shape[1] != len(self.bands):
            raise ValueError("reference bands and flux columns must align")
        source_ids = torch.as_tensor(
            self.source_ids, dtype=torch.int64, device=flux.device
        )
        variance = torch.as_tensor(
            self.flux_variance, dtype=flux.dtype, device=flux.device
        )
        if source_ids.shape != (flux.shape[0],) or variance.shape != flux.shape:
            raise ValueError("reference arrays have inconsistent shapes")
        present = torch.isfinite(flux)
        if not bool(
            ((~present) | ((flux > 0) & torch.isfinite(variance) & (variance > 0)))
            .all()
            .detach()
            .cpu()
            .item()
        ):
            raise ValueError("present reference fluxes need positive finite variance")
        object.__setattr__(self, "source_ids", source_ids)
        object.__setattr__(self, "flux", flux)
        object.__setattr__(self, "flux_variance", variance)


@dataclass(frozen=True)
class CoupledCalibrationConfig:
    astrometry: AstrometryConfig = field(default_factory=AstrometryConfig)
    max_iterations: int = 6
    photometry_max_iterations: int = 20
    convergence_tolerance: float = 1.0e-7
    photometry_convergence_tolerance: float = 1.0e-8
    huber_delta: float = 3.0
    fit_extinction: bool = True
    fit_cloud: bool = True
    fit_star_flat: bool = True
    fit_color_terms: bool = True
    color_bands: tuple[Hashable, Hashable] | None = None
    color_pivot: float = 0.0
    color_scale: float = -2.5 / math.log(10.0)
    missing_color: MissingColorPolicy = "drop"
    fit_variability: bool = False
    variable_source_ids: tuple[int, ...] = ()
    holdout_source_ids: tuple[int, ...] = ()
    holdout_exposure_ids: tuple[Hashable, ...] = ()
    damping: float = 1.0e-10
    rank_rtol: float = 1.0e-10
    cg_max_iterations: int = 500
    cg_rtol: float = 1.0e-10
    covariance_max_parameters: int = 256

    def __post_init__(self) -> None:
        if self.max_iterations < 1 or self.photometry_max_iterations < 1:
            raise ValueError("iteration limits must be positive")
        if self.cg_max_iterations < 1 or self.covariance_max_parameters < 1:
            raise ValueError("solver limits must be positive")
        for name in (
            "convergence_tolerance",
            "photometry_convergence_tolerance",
            "huber_delta",
            "rank_rtol",
            "cg_rtol",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.damping) or self.damping < 0:
            raise ValueError("damping must be finite and non-negative")
        if self.missing_color not in {"error", "zero", "drop"}:
            raise ValueError("unsupported missing_color policy")


@dataclass(frozen=True)
class PhotometricCalibrationDiagnostics:
    converged: bool
    iterations: int
    objective_history: tuple[float, ...]
    step_history: tuple[float, ...]
    cg_iterations: tuple[int, ...]
    connected_components: tuple[tuple[Hashable, ...], ...]
    gauge_exposures: tuple[Hashable, ...]
    reference_anchored_components: tuple[bool, ...]
    dropped_extinction_exposures: tuple[Hashable, ...]
    dropped_cloud_exposures: tuple[Hashable, ...]
    dropped_star_flat_parameters: tuple[tuple[Hashable, int], ...]
    dropped_color_bands: tuple[Hashable, ...]
    information_rank: int | None
    information_dimension: int
    condition_number: float | None
    matrix_free: bool = True
    assembled_measurement_matrix: bool = False


@dataclass(frozen=True)
class PhotometricCalibrationSolution:
    source_ids: Tensor
    bands: tuple[Hashable, ...]
    source_band_flux: Tensor
    source_band_log_flux: Tensor
    source_coefficient_variance: Tensor
    colors: Tensor
    exposure_ids: tuple[Hashable, ...]
    exposure_log_zeropoint: Tensor
    exposure_extinction: Tensor
    exposure_cloud: Tensor
    detector_ids: tuple[Hashable, ...]
    star_flat_coefficients: Tensor
    band_color_terms: Tensor
    predicted_log_flux: Tensor
    residual_log_flux: Tensor
    robust_weights: Tensor
    variability_offsets: Tensor
    fit_mask: Tensor
    global_covariance: Tensor | None
    diagnostics: PhotometricCalibrationDiagnostics
    provenance: dict[str, object]

    @property
    def converged(self) -> bool:
        return self.diagnostics.converged

    def flux_for(self, source_id: int, band: Hashable) -> Tensor:
        source = torch.searchsorted(
            self.source_ids,
            torch.as_tensor(source_id, device=self.source_ids.device),
        )
        return self.source_band_flux[source, self.bands.index(band)]


@dataclass(frozen=True)
class CoupledCalibrationDiagnostics:
    converged: bool
    iterations: int
    stage_history: tuple[str, ...]
    astrometric_step_history: tuple[float, ...]
    photometric_step_history: tuple[float, ...]
    color_step_history: tuple[float, ...]
    source_holdout_count: int
    exposure_holdout_count: int


@dataclass(frozen=True)
class CoupledCalibrationSolution:
    astrometry: AstrometrySolution
    photometry: PhotometricCalibrationSolution
    diagnostics: CoupledCalibrationDiagnostics
    covariance: dict[str, object]
    provenance: dict[str, object]

    @property
    def converged(self) -> bool:
        return self.diagnostics.converged


def photometric_log_residuals(
    measured_flux: Any,
    predicted_flux: Any,
    flux_covariance: Any,
    *,
    weights: Any | None = None,
    whiten: bool = True,
) -> Tensor:
    """Compute optionally whitened log-flux residuals for calibration."""
    measured = torch.as_tensor(measured_flux)
    predicted = torch.as_tensor(
        predicted_flux, dtype=measured.dtype, device=measured.device
    )
    if measured.shape != predicted.shape or measured.ndim != 1:
        raise ValueError("measured_flux and predicted_flux need matching (N,) shape")
    if not bool(((measured > 0) & (predicted > 0)).all().detach().cpu().item()):
        raise ValueError("log-flux residuals require positive fluxes")
    covariance = _measurement_covariance(flux_covariance, measured)
    jacobian = measured.reciprocal()
    log_covariance = jacobian[:, None] * covariance * jacobian[None, :]
    residual = torch.log(measured) - torch.log(predicted)
    if whiten:
        residual = torch.linalg.solve_triangular(
            torch.linalg.cholesky(log_covariance),
            residual.unsqueeze(-1),
            upper=False,
        ).squeeze(-1)
    if weights is not None:
        weight = torch.as_tensor(weights, dtype=measured.dtype, device=measured.device)
        if weight.shape != measured.shape:
            raise ValueError("weights must have shape (N,)")
        residual = residual * torch.sqrt(weight)
    return residual


class CoupledCalibrationGraph:
    """Fixed-association exposure-source graph with astrometric and flux blocks."""

    def __init__(
        self,
        astrometry_graph: AstrometricMatchGraph,
        photometric_blocks: Sequence[PhotometricMeasurementBlock],
        *,
        reference: PhotometricReference | None = None,
        sed_basis: Tensor | None = None,
    ) -> None:
        blocks = tuple(photometric_blocks)
        if not blocks:
            raise ValueError("at least one photometric measurement block is required")
        dtype = astrometry_graph.observed_plane.dtype
        device = astrometry_graph.observed_plane.device
        if any(block.flux.dtype != dtype or block.flux.device != device for block in blocks):
            raise ValueError("astrometry and photometry must share dtype and device")
        exposure_lookup = {
            identifier: index
            for index, identifier in enumerate(astrometry_graph.exposure_ids)
        }
        detector_lookup = {
            identifier: index
            for index, identifier in enumerate(astrometry_graph.detector_ids)
        }
        associations = set(
            zip(
                astrometry_graph.row_exposure.detach().cpu().tolist(),
                astrometry_graph.source_ids[
                    astrometry_graph.row_source
                ].detach().cpu().tolist(),
                strict=True,
            )
        )
        bands = tuple(dict.fromkeys(block.band for block in blocks))
        band_lookup = {value: index for index, value in enumerate(bands)}
        row_exposure = []
        row_source = []
        row_band = []
        row_detector = []
        block_slices = []
        start = 0
        max_star_flat = 0
        for block in blocks:
            if block.exposure_id not in exposure_lookup:
                raise ValueError("photometry cannot introduce an exposure")
            if block.detector_id not in detector_lookup:
                raise ValueError("photometry cannot introduce a detector")
            exposure = exposure_lookup[block.exposure_id]
            locations = torch.searchsorted(astrometry_graph.source_ids, block.source_ids)
            safe = locations.clamp(max=max(astrometry_graph.source_ids.numel() - 1, 0))
            valid = (locations < astrometry_graph.source_ids.numel()) & (
                astrometry_graph.source_ids[safe] == block.source_ids
            )
            if not bool(valid.all().detach().cpu().item()):
                raise ValueError("photometry cannot create sources")
            for source_id in block.source_ids.detach().cpu().tolist():
                if (exposure, int(source_id)) not in associations:
                    raise ValueError(
                        "photometry association must exist in the astrometric graph"
                    )
            row_exposure.append(
                torch.full(
                    (block.count,), exposure, dtype=torch.int64, device=device
                )
            )
            row_source.append(locations)
            row_band.append(
                torch.full(
                    (block.count,),
                    band_lookup[block.band],
                    dtype=torch.int64,
                    device=device,
                )
            )
            row_detector.append(
                torch.full(
                    (block.count,),
                    detector_lookup[block.detector_id],
                    dtype=torch.int64,
                    device=device,
                )
            )
            stop = start + block.count
            block_slices.append(slice(start, stop))
            start = stop
            max_star_flat = max(max_star_flat, int(block.star_flat_basis.shape[1]))
        if sed_basis is None:
            basis = torch.eye(len(bands), dtype=dtype, device=device)
        else:
            basis = torch.as_tensor(sed_basis, dtype=dtype, device=device)
            if basis.ndim != 2 or basis.shape[0] != len(bands):
                raise ValueError("sed_basis must have shape (n_band, n_coefficient)")
            _finite(basis, "sed_basis")
            analysis = basis.cpu() if device.type == "mps" else basis
            if int(torch.linalg.matrix_rank(analysis).item()) < basis.shape[1]:
                raise ValueError("sed_basis columns must be independent")
        if reference is not None:
            if reference.flux.dtype != dtype or reference.flux.device != device:
                raise ValueError("reference photometry must share dtype and device")
            if set(reference.bands) - set(bands):
                raise ValueError("reference contains bands absent from measurements")
        self.astrometry_graph = astrometry_graph
        self.blocks = blocks
        self.reference = reference
        self.sed_basis = basis
        self.bands = bands
        self.exposure_ids = astrometry_graph.exposure_ids
        self.detector_ids = astrometry_graph.detector_ids
        self.row_exposure = torch.cat(row_exposure)
        self.row_source = torch.cat(row_source)
        self.row_band = torch.cat(row_band)
        self.row_detector = torch.cat(row_detector)
        self.block_slices = tuple(block_slices)
        self.max_star_flat = max_star_flat
        self.flux = torch.cat([block.flux for block in blocks])
        self.log_flux = torch.log(self.flux)
        self.weights = torch.cat([block.weights for block in blocks])
        self.base_fit_mask = torch.cat([block.fit_mask for block in blocks])
        self.extinction_basis = torch.cat([block.extinction_basis for block in blocks])
        self.cloud_basis = torch.cat([block.cloud_basis for block in blocks])
        star_flat_rows = []
        for block in blocks:
            padding = max_star_flat - int(block.star_flat_basis.shape[1])
            star_flat_rows.append(
                torch.nn.functional.pad(block.star_flat_basis, (0, padding))
            )
        self.star_flat_basis = torch.cat(star_flat_rows)
        self.log_covariance = tuple(
            block.flux_covariance
            / (block.flux[:, None] * block.flux[None, :])
            for block in blocks
        )

    def solve(
        self, config: CoupledCalibrationConfig | None = None
    ) -> CoupledCalibrationSolution:
        return _CoupledSolver(
            self, CoupledCalibrationConfig() if config is None else config
        ).solve()


class _PhotometrySolver:
    def __init__(self, graph: CoupledCalibrationGraph, config: CoupledCalibrationConfig):
        self.graph = graph
        self.config = config
        self.dtype = graph.log_flux.dtype
        self.device = graph.log_flux.device
        self.n_row = int(graph.log_flux.numel())
        self.n_source = int(graph.astrometry_graph.source_ids.numel())
        self.n_exposure = len(graph.exposure_ids)
        self.n_detector = len(graph.detector_ids)
        self.n_band = len(graph.bands)
        self.source_width = int(graph.sed_basis.shape[1])
        self.source_dimension = self.n_source * self.source_width
        self.exposure_offset = 0
        self.star_flat_offset = 3 * self.n_exposure
        self.color_term_offset = (
            self.star_flat_offset + self.n_detector * graph.max_star_flat
        )
        self.global_dimension = self.color_term_offset + self.n_band
        self.variable_mask = (
            self._id_mask(config.variable_source_ids)
            if config.fit_variability
            else torch.zeros(
                self.n_source, dtype=torch.bool, device=self.device
            )
        )
        self.source_holdout_mask = self._id_mask(config.holdout_source_ids)
        exposure_holdout = torch.tensor(
            [item in set(config.holdout_exposure_ids) for item in graph.exposure_ids],
            dtype=torch.bool,
            device=self.device,
        )
        self.fit_mask = (
            graph.base_fit_mask
            & ~self.source_holdout_mask[graph.row_source]
            & ~exposure_holdout[graph.row_exposure]
            & (graph.weights > 0)
        )
        self.global_fit_mask = self.fit_mask & ~self.variable_mask[graph.row_source]
        self.extinction_basis = graph.extinction_basis.clone()
        self.cloud_basis = graph.cloud_basis.clone()
        self.star_flat_basis = graph.star_flat_basis.clone()
        self._center_nuisance_bases()
        self.reference_rows = self._reference_rows()
        (
            self.components,
            self.component_exposures,
            self.component_anchored,
            self.gauge_exposure_indices,
        ) = self._components()
        self.global_active, self.dropped = self._global_activity(exposure_holdout)
        self.source_active = self._source_activity()

    def _id_mask(self, identifiers: Sequence[int]) -> Tensor:
        selected = set(int(value) for value in identifiers)
        return torch.tensor(
            [int(value) in selected for value in self.graph.astrometry_graph.source_ids.cpu()],
            dtype=torch.bool,
            device=self.device,
        )

    def _center_nuisance_bases(self) -> None:
        for exposure in range(self.n_exposure):
            rows = self.global_fit_mask & (self.graph.row_exposure == exposure)
            if bool(rows.any()):
                self.extinction_basis[rows] -= self.extinction_basis[rows].mean()
                self.cloud_basis[rows] -= self.cloud_basis[rows].mean()
        for detector in range(self.n_detector):
            rows = self.global_fit_mask & (self.graph.row_detector == detector)
            if bool(rows.any()) and self.graph.max_star_flat:
                self.star_flat_basis[rows] -= self.star_flat_basis[rows].mean(dim=0)

    def _reference_rows(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        reference = self.graph.reference
        if reference is None:
            empty_i = torch.empty(0, dtype=torch.int64, device=self.device)
            empty = torch.empty(0, dtype=self.dtype, device=self.device)
            return empty_i, empty_i, empty, empty
        source_rows = []
        band_rows = []
        values = []
        variances = []
        source_ids = self.graph.astrometry_graph.source_ids
        reference_band = {value: index for index, value in enumerate(reference.bands)}
        for row, source_id in enumerate(reference.source_ids.detach().cpu().tolist()):
            location = int(
                torch.searchsorted(
                    source_ids,
                    torch.tensor(source_id, dtype=torch.int64, device=self.device),
                ).item()
            )
            if location >= self.n_source or int(source_ids[location].item()) != source_id:
                continue
            if bool(self.source_holdout_mask[location]):
                continue
            for band, graph_band in zip(self.graph.bands, range(self.n_band), strict=True):
                if band not in reference_band:
                    continue
                column = reference_band[band]
                flux = reference.flux[row, column]
                if not bool(torch.isfinite(flux)):
                    continue
                source_rows.append(location)
                band_rows.append(graph_band)
                values.append(torch.log(flux))
                variances.append(reference.flux_variance[row, column] / flux.square())
        if not source_rows:
            empty_i = torch.empty(0, dtype=torch.int64, device=self.device)
            empty = torch.empty(0, dtype=self.dtype, device=self.device)
            return empty_i, empty_i, empty, empty
        return (
            torch.tensor(source_rows, dtype=torch.int64, device=self.device),
            torch.tensor(band_rows, dtype=torch.int64, device=self.device),
            torch.stack(values),
            torch.stack(variances),
        )

    def _components(
        self,
    ) -> tuple[
        tuple[tuple[int, ...], ...],
        tuple[tuple[int, ...], ...],
        tuple[bool, ...],
        tuple[int, ...],
    ]:
        source_node_count = self.n_source * self.n_band
        union = _UnionFind(self.n_exposure + source_node_count)
        active_rows = torch.nonzero(self.global_fit_mask, as_tuple=False).flatten()
        active_exposure = self.graph.row_exposure[active_rows].detach().cpu().tolist()
        active_source = self.graph.row_source[active_rows].detach().cpu().tolist()
        active_band = self.graph.row_band[active_rows].detach().cpu().tolist()
        for exposure, source, band in zip(
            active_exposure, active_source, active_band, strict=True
        ):
            union.union(
                int(exposure),
                self.n_exposure + int(source) * self.n_band + int(band),
            )
        roots = sorted(
            {union.find(int(exposure)) for exposure in active_exposure}
        )
        active_nodes = sorted(
            {int(exposure) for exposure in active_exposure}
            | {
                self.n_exposure + int(source) * self.n_band + int(band)
                for source, band in zip(active_source, active_band, strict=True)
            }
        )
        ref_source, ref_band, _, _ = self.reference_rows
        reference_nodes = {
            int(source) * self.n_band + int(band)
            for source, band in zip(
                ref_source.detach().cpu().tolist(),
                ref_band.detach().cpu().tolist(),
                strict=True,
            )
        }
        components = []
        exposures = []
        anchored = []
        gauges = []
        groups: dict[int, list[int]] = {}
        for node in active_nodes:
            groups.setdefault(union.find(node), []).append(node)
        for root in roots:
            grouped = groups[root]
            exp = tuple(node for node in grouped if node < self.n_exposure)
            nodes = tuple(
                node - self.n_exposure
                for node in grouped
                if node >= self.n_exposure
            )
            is_anchored = any(node in reference_nodes for node in nodes)
            components.append(nodes)
            exposures.append(exp)
            anchored.append(is_anchored)
            if not is_anchored and exp:
                gauges.append(exp[0])
        return tuple(components), tuple(exposures), tuple(anchored), tuple(gauges)

    def _global_activity(
        self, exposure_holdout: Tensor
    ) -> tuple[Tensor, dict[str, tuple[Any, ...]]]:
        active = torch.ones(
            self.global_dimension, dtype=self.dtype, device=self.device
        )
        dropped_extinction = []
        dropped_cloud = []
        dropped_flat = []
        dropped_color = []
        for exposure in range(self.n_exposure):
            base = 3 * exposure
            rows = self.global_fit_mask & (self.graph.row_exposure == exposure)
            if bool(exposure_holdout[exposure]) or not bool(rows.any()):
                active[base : base + 3] = 0
                dropped_extinction.append(self.graph.exposure_ids[exposure])
                dropped_cloud.append(self.graph.exposure_ids[exposure])
                continue
            columns = [torch.ones(int(rows.sum()), dtype=self.dtype, device=self.device)]
            if not self.config.fit_extinction:
                active[base + 1] = 0
                dropped_extinction.append(self.graph.exposure_ids[exposure])
            else:
                candidate = self.extinction_basis[rows]
                matrix = torch.stack(columns + [candidate], dim=-1)
                if int(torch.linalg.matrix_rank(matrix, rtol=self.config.rank_rtol)) < 2:
                    active[base + 1] = 0
                    dropped_extinction.append(self.graph.exposure_ids[exposure])
                else:
                    columns.append(candidate)
            if not self.config.fit_cloud:
                active[base + 2] = 0
                dropped_cloud.append(self.graph.exposure_ids[exposure])
            else:
                candidate = self.cloud_basis[rows]
                previous = len(columns)
                matrix = torch.stack(columns + [candidate], dim=-1)
                if int(torch.linalg.matrix_rank(matrix, rtol=self.config.rank_rtol)) < previous + 1:
                    active[base + 2] = 0
                    dropped_cloud.append(self.graph.exposure_ids[exposure])
        for exposure in self.gauge_exposure_indices:
            active[3 * exposure] = 0
        for detector in range(self.n_detector):
            rows = self.global_fit_mask & (self.graph.row_detector == detector)
            for column in range(self.graph.max_star_flat):
                index = self.star_flat_offset + detector * self.graph.max_star_flat + column
                if (
                    not self.config.fit_star_flat
                    or not bool(rows.any())
                    or float(torch.linalg.vector_norm(self.star_flat_basis[rows, column]))
                    <= self.config.rank_rtol
                ):
                    active[index] = 0
                    dropped_flat.append((self.graph.detector_ids[detector], column))
        ref_source, ref_band, _, _ = self.reference_rows
        reference_color = self._colors_from_reference()
        for band in range(self.n_band):
            index = self.color_term_offset + band
            standards = ref_source[ref_band == band]
            identifiable = False
            if standards.numel() >= 2:
                values = reference_color[standards]
                finite = torch.isfinite(values)
                identifiable = bool(finite.sum() >= 2) and float(
                    torch.std(values[finite], correction=0)
                ) > self.config.rank_rtol
            if not self.config.fit_color_terms or not identifiable:
                active[index] = 0
                dropped_color.append(self.graph.bands[band])
        return active, {
            "extinction": tuple(dropped_extinction),
            "cloud": tuple(dropped_cloud),
            "flat": tuple(dropped_flat),
            "color": tuple(dropped_color),
        }

    def _source_activity(self) -> Tensor:
        active = torch.zeros(
            (self.n_source, self.source_width), dtype=self.dtype, device=self.device
        )
        for source in range(self.n_source):
            if bool(self.source_holdout_mask[source]):
                continue
            rows = self.fit_mask & (self.graph.row_source == source)
            bands = self.graph.row_band[rows]
            design = self.graph.sed_basis[bands]
            ref_source, ref_band, _, _ = self.reference_rows
            selected = ref_band[ref_source == source]
            if selected.numel():
                design = torch.cat((design, self.graph.sed_basis[selected]), dim=0)
            if design.numel() == 0:
                continue
            selected_columns: list[int] = []
            rank = 0
            for column in range(self.source_width):
                candidate = selected_columns + [column]
                candidate_design = design[:, candidate]
                if self.device.type == "mps":
                    candidate_design = candidate_design.cpu()
                candidate_rank = int(
                    torch.linalg.matrix_rank(
                        candidate_design, rtol=self.config.rank_rtol
                    )
                )
                if candidate_rank > rank:
                    selected_columns.append(column)
                    rank = candidate_rank
            if selected_columns:
                active[source, selected_columns] = 1
        return active.reshape(-1)

    def _colors_from_reference(self) -> Tensor:
        colors = torch.full(
            (self.n_source,), float("nan"), dtype=self.dtype, device=self.device
        )
        if self.config.color_bands is None or self.graph.reference is None:
            return colors
        left, right = self.config.color_bands
        reference = self.graph.reference
        if left not in reference.bands or right not in reference.bands:
            return colors
        left_index = reference.bands.index(left)
        right_index = reference.bands.index(right)
        source_ids = self.graph.astrometry_graph.source_ids
        for row, source_id in enumerate(reference.source_ids):
            location = torch.searchsorted(source_ids, source_id)
            if int(location) >= self.n_source or source_ids[location] != source_id:
                continue
            flux_left = reference.flux[row, left_index]
            flux_right = reference.flux[row, right_index]
            if bool(torch.isfinite(flux_left) & torch.isfinite(flux_right)):
                colors[location] = self.config.color_scale * (
                    torch.log(flux_left) - torch.log(flux_right)
                )
        return colors

    def _source_forward(self, parameters: Tensor) -> Tensor:
        coefficients = parameters.reshape(self.n_source, self.source_width)
        row_coefficients = coefficients[self.graph.row_source]
        row_basis = self.graph.sed_basis[self.graph.row_band]
        return torch.sum(row_coefficients * row_basis, dim=-1)

    def _source_vjp(self, rows: Tensor) -> Tensor:
        result = torch.zeros(
            (self.n_source, self.source_width), dtype=self.dtype, device=self.device
        )
        result.index_add_(
            0,
            self.graph.row_source,
            rows.unsqueeze(-1) * self.graph.sed_basis[self.graph.row_band],
        )
        return result.reshape(-1)

    def _reference_forward(self, parameters: Tensor) -> Tensor:
        source, band, _, _ = self.reference_rows
        coefficients = parameters.reshape(self.n_source, self.source_width)
        return torch.sum(coefficients[source] * self.graph.sed_basis[band], dim=-1)

    def _reference_vjp(self, rows: Tensor) -> Tensor:
        source, band, _, _ = self.reference_rows
        result = torch.zeros(
            (self.n_source, self.source_width), dtype=self.dtype, device=self.device
        )
        if rows.numel():
            result.index_add_(
                0, source, rows.unsqueeze(-1) * self.graph.sed_basis[band]
            )
        return result.reshape(-1)

    def _colors(self, source_parameters: Tensor) -> Tensor:
        log_flux = source_parameters.reshape(self.n_source, self.source_width) @ self.graph.sed_basis.mT
        if self.config.color_bands is None:
            return torch.zeros(self.n_source, dtype=self.dtype, device=self.device)
        left, right = self.config.color_bands
        if left not in self.graph.bands or right not in self.graph.bands:
            if self.config.missing_color == "error":
                raise ValueError("color bands must occur in the photometric graph")
            return torch.full(
                (self.n_source,),
                float("nan") if self.config.missing_color == "drop" else 0.0,
                dtype=self.dtype,
                device=self.device,
            )
        left_index = self.graph.bands.index(left)
        right_index = self.graph.bands.index(right)
        observed_left = torch.zeros(self.n_source, dtype=torch.bool, device=self.device)
        observed_right = torch.zeros_like(observed_left)
        observed_left[self.graph.row_source[self.fit_mask & (self.graph.row_band == left_index)]] = True
        observed_right[self.graph.row_source[self.fit_mask & (self.graph.row_band == right_index)]] = True
        ref_source, ref_band, _, _ = self.reference_rows
        observed_left[ref_source[ref_band == left_index]] = True
        observed_right[ref_source[ref_band == right_index]] = True
        present = observed_left & observed_right
        color = self.config.color_scale * (
            log_flux[:, left_index] - log_flux[:, right_index]
        )
        if self.config.missing_color == "error" and not bool(present.all()):
            raise ValueError("source colors are missing required bands")
        fill = 0.0 if self.config.missing_color == "zero" else float("nan")
        return torch.where(present, color, torch.full_like(color, fill))

    def _global_forward(self, parameters: Tensor, colors: Tensor) -> Tensor:
        exposure = parameters[: self.star_flat_offset].reshape(self.n_exposure, 3)
        values = exposure[self.graph.row_exposure, 0]
        values = values + exposure[self.graph.row_exposure, 1] * self.extinction_basis
        values = values + exposure[self.graph.row_exposure, 2] * self.cloud_basis
        if self.graph.max_star_flat:
            flat = parameters[
                self.star_flat_offset : self.color_term_offset
            ].reshape(self.n_detector, self.graph.max_star_flat)
            values = values + torch.sum(
                flat[self.graph.row_detector] * self.star_flat_basis, dim=-1
            )
        terms = parameters[self.color_term_offset :]
        row_color = torch.nan_to_num(
            colors[self.graph.row_source] - self.config.color_pivot, nan=0.0
        )
        return values + terms[self.graph.row_band] * row_color

    def _global_vjp(self, rows: Tensor, colors: Tensor) -> Tensor:
        result = torch.zeros(
            self.global_dimension, dtype=self.dtype, device=self.device
        )
        exposure = result[: self.star_flat_offset].reshape(self.n_exposure, 3)
        exposure[:, 0].index_add_(0, self.graph.row_exposure, rows)
        exposure[:, 1].index_add_(
            0, self.graph.row_exposure, rows * self.extinction_basis
        )
        exposure[:, 2].index_add_(
            0, self.graph.row_exposure, rows * self.cloud_basis
        )
        if self.graph.max_star_flat:
            flat = result[
                self.star_flat_offset : self.color_term_offset
            ].reshape(self.n_detector, self.graph.max_star_flat)
            flat.index_add_(
                0, self.graph.row_detector, rows.unsqueeze(-1) * self.star_flat_basis
            )
        row_color = torch.nan_to_num(
            colors[self.graph.row_source] - self.config.color_pivot, nan=0.0
        )
        result[self.color_term_offset :].index_add_(
            0, self.graph.row_band, rows * row_color
        )
        return result

    def _precision_apply(
        self, rows: Tensor, mask: Tensor, robust: Tensor
    ) -> Tensor:
        result = torch.zeros_like(rows)
        for block_slice, covariance in zip(
            self.graph.block_slices, self.graph.log_covariance, strict=True
        ):
            selected = mask[block_slice]
            indices = torch.nonzero(selected, as_tuple=False).flatten()
            if indices.numel() == 0:
                continue
            global_indices = indices + int(block_slice.start or 0)
            factor = torch.sqrt(
                self.graph.weights[global_indices] * robust[global_indices]
            )
            sub_covariance = covariance[indices[:, None], indices[None, :]]
            weighted = factor * rows[global_indices]
            solved = torch.cholesky_solve(
                weighted.unsqueeze(-1), torch.linalg.cholesky(sub_covariance)
            ).squeeze(-1)
            result[global_indices] = factor * solved
        return result

    def _solve_cg(
        self,
        action: Any,
        rhs: Tensor,
        active: Tensor,
    ) -> tuple[Tensor, int]:
        from ..map.solvers import conjugate_gradient

        projected_rhs = rhs * active
        if float(torch.linalg.vector_norm(projected_rhs)) == 0.0:
            return torch.zeros_like(rhs), 0
        result = conjugate_gradient(
            action,
            projected_rhs,
            operator_is_hpd=True,
            max_iterations=self.config.cg_max_iterations,
            rtol=self.config.cg_rtol,
            atol=0.0,
            projection=lambda value: value * active,
        )
        if result.diagnostics is None or result.diagnostics.breakdown is not None:
            message = (
                "missing diagnostics"
                if result.diagnostics is None
                else result.diagnostics.breakdown.message
            )
            raise RuntimeError(f"photometric calibration solve failed: {message}")
        return result.solution * active, result.diagnostics.iterations

    def _solve_source(
        self,
        global_parameters: Tensor,
        colors: Tensor,
        variability: Tensor,
        robust: Tensor,
    ) -> tuple[Tensor, int, Any]:
        target = self.graph.log_flux - self._global_forward(global_parameters, colors) - variability
        weighted_target = self._precision_apply(target, self.fit_mask, robust)
        rhs = self._source_vjp(weighted_target)
        ref_source, ref_band, ref_value, ref_variance = self.reference_rows
        _ = ref_source, ref_band
        if ref_value.numel():
            rhs = rhs + self._reference_vjp(ref_value / ref_variance)

        def action(vector: Tensor) -> Tensor:
            vector = vector * self.source_active
            rows = self._source_forward(vector)
            output = self._source_vjp(
                self._precision_apply(rows, self.fit_mask, robust)
            )
            if ref_value.numel():
                output = output + self._reference_vjp(
                    self._reference_forward(vector) / ref_variance
                )
            return (
                output + self.config.damping * vector
            ) * self.source_active

        solution, iterations = self._solve_cg(
            action, rhs, self.source_active
        )
        return solution, iterations, action

    def _solve_global(
        self, source_parameters: Tensor, colors: Tensor, robust: Tensor
    ) -> tuple[Tensor, int, Any]:
        target = self.graph.log_flux - self._source_forward(source_parameters)
        weighted_target = self._precision_apply(target, self.global_fit_mask, robust)
        rhs = self._global_vjp(weighted_target, colors)

        def action(vector: Tensor) -> Tensor:
            vector = vector * self.global_active
            rows = self._global_forward(vector, colors)
            output = self._global_vjp(
                self._precision_apply(rows, self.global_fit_mask, robust), colors
            )
            return (
                output + self.config.damping * vector
            ) * self.global_active

        solution, iterations = self._solve_cg(
            action, rhs, self.global_active
        )
        return solution, iterations, action

    def _solve_joint(
        self, colors: Tensor, robust: Tensor
    ) -> tuple[Tensor, Tensor, int, Any]:
        source_dimension = self.source_dimension
        active = torch.cat((self.source_active, self.global_active))
        weighted_target = self._precision_apply(
            self.graph.log_flux, self.global_fit_mask, robust
        )
        rhs = torch.cat(
            (
                self._source_vjp(weighted_target),
                self._global_vjp(weighted_target, colors),
            )
        )
        _, _, ref_value, ref_variance = self.reference_rows
        if ref_value.numel():
            rhs[:source_dimension] += self._reference_vjp(
                ref_value / ref_variance
            )

        def action(vector: Tensor) -> Tensor:
            vector = vector * active
            source = vector[:source_dimension]
            global_parameters = vector[source_dimension:]
            rows = self._source_forward(source) + self._global_forward(
                global_parameters, colors
            )
            weighted = self._precision_apply(
                rows, self.global_fit_mask, robust
            )
            source_output = self._source_vjp(weighted)
            if ref_value.numel():
                source_output = source_output + self._reference_vjp(
                    self._reference_forward(source) / ref_variance
                )
            output = torch.cat(
                (source_output, self._global_vjp(weighted, colors))
            )
            return (output + self.config.damping * vector) * active

        solution, iterations = self._solve_cg(action, rhs, active)
        return (
            solution[:source_dimension],
            solution[source_dimension:],
            iterations,
            action,
        )

    def _variability(
        self,
        source_parameters: Tensor,
        global_parameters: Tensor,
        colors: Tensor,
    ) -> Tensor:
        offsets = torch.zeros(self.n_row, dtype=self.dtype, device=self.device)
        if not self.config.fit_variability:
            return offsets
        residual = (
            self.graph.log_flux
            - self._source_forward(source_parameters)
            - self._global_forward(global_parameters, colors)
        )
        selected = self.fit_mask & self.variable_mask[self.graph.row_source]
        offsets[selected] = residual[selected]
        return offsets

    def _robust(self, residual: Tensor) -> Tensor:
        diagonal = torch.cat(
            [torch.diagonal(value) for value in self.graph.log_covariance]
        )
        norm = residual.abs() / torch.sqrt(diagonal)
        delta = torch.as_tensor(
            self.config.huber_delta, dtype=self.dtype, device=self.device
        )
        weights = torch.where(
            norm <= delta,
            torch.ones_like(norm),
            delta / norm.clamp_min(torch.finfo(self.dtype).tiny),
        )
        return weights * self.fit_mask.to(dtype=self.dtype)

    def _objective(self, residual: Tensor, robust: Tensor) -> Tensor:
        weighted = self._precision_apply(residual, self.fit_mask, robust)
        return 0.5 * torch.dot(residual, weighted)

    def _dense_covariance(
        self, action: Any, dimension: int, active: Tensor
    ) -> tuple[Tensor | None, int | None, float | None]:
        indices = torch.nonzero(active > 0, as_tuple=False).flatten()
        if indices.numel() == 0:
            return torch.zeros((dimension, dimension), dtype=self.dtype, device=self.device), 0, None
        if int(indices.numel()) > self.config.covariance_max_parameters:
            return None, None, None
        columns = []
        for index in indices:
            basis = torch.zeros(dimension, dtype=self.dtype, device=self.device)
            basis[index] = 1
            columns.append(action(basis)[indices])
        information = torch.stack(columns, dim=-1)
        information = 0.5 * (information + information.mT)
        analysis = information.cpu() if self.device.type == "mps" else information
        eigenvalues = torch.linalg.eigvalsh(analysis).to(self.device)
        tolerance = self.config.rank_rtol * eigenvalues.abs().amax()
        positive = eigenvalues > tolerance
        rank = int(positive.sum())
        condition = None
        if bool(positive.any()):
            condition = float(
                (eigenvalues[positive].amax() / eigenvalues[positive].amin())
                .detach()
                .cpu()
                .item()
            )
        inverse = torch.linalg.pinv(
            analysis, rtol=self.config.rank_rtol
        ).to(self.device)
        full = torch.full(
            (dimension, dimension), float("nan"), dtype=self.dtype, device=self.device
        )
        full[indices[:, None], indices[None, :]] = inverse
        return full, rank, condition

    def solve(self) -> PhotometricCalibrationSolution:
        source = torch.zeros(
            self.source_dimension, dtype=self.dtype, device=self.device
        )
        global_parameters = torch.zeros(
            self.global_dimension, dtype=self.dtype, device=self.device
        )
        variability = torch.zeros(self.n_row, dtype=self.dtype, device=self.device)
        robust = self.fit_mask.to(dtype=self.dtype)
        objectives = []
        steps = []
        cg_iterations = []
        converged = False
        last_joint_action: Any = None
        for _ in range(self.config.photometry_max_iterations):
            previous_source = source
            previous_global = global_parameters
            previous_colors = self._colors(source)
            source, global_parameters, joint_cg, last_joint_action = self._solve_joint(
                previous_colors, robust
            )
            source, source_cg, _ = self._solve_source(
                global_parameters, previous_colors, variability, robust
            )
            colors = self._colors(source)
            variability = self._variability(source, global_parameters, colors)
            prediction = (
                self._source_forward(source)
                + self._global_forward(global_parameters, colors)
                + variability
            )
            residual = self.graph.log_flux - prediction
            robust = self._robust(residual)
            objective = self._objective(residual, robust)
            objectives.append(float(objective.detach().cpu().item()))
            step = torch.sqrt(
                torch.sum((source - previous_source).square())
                + torch.sum((global_parameters - previous_global).square())
                + torch.sum(
                    (
                        torch.nan_to_num(colors)
                        - torch.nan_to_num(previous_colors)
                    ).square()
                )
            )
            norm = torch.sqrt(
                torch.sum(source.square()) + torch.sum(global_parameters.square())
            ).clamp_min(1.0)
            relative_step = float((step / norm).detach().cpu().item())
            steps.append(relative_step)
            cg_iterations.extend((joint_cg, source_cg))
            if relative_step <= self.config.photometry_convergence_tolerance:
                converged = True
                break
        colors = self._colors(source)
        variability = self._variability(source, global_parameters, colors)
        prediction = (
            self._source_forward(source)
            + self._global_forward(global_parameters, colors)
            + variability
        )
        residual = self.graph.log_flux - prediction
        robust = self._robust(residual)
        log_flux = source.reshape(self.n_source, self.source_width) @ self.graph.sed_basis.mT
        log_flux = torch.where(
            self.source_holdout_mask.unsqueeze(-1),
            torch.full_like(log_flux, float("nan")),
            log_flux,
        )
        joint_active = torch.cat((self.source_active, self.global_active))
        joint_covariance, rank, condition = self._dense_covariance(
            last_joint_action,
            self.source_dimension + self.global_dimension,
            joint_active,
        )
        global_covariance = (
            None
            if joint_covariance is None
            else joint_covariance[self.source_dimension :, self.source_dimension :]
        )
        source_covariance = (
            None
            if joint_covariance is None
            else joint_covariance[: self.source_dimension, : self.source_dimension]
        )
        source_variance = (
            torch.diagonal(source_covariance).reshape(self.n_source, self.source_width)
            if source_covariance is not None
            else torch.full(
                (self.n_source, self.source_width),
                float("nan"),
                dtype=self.dtype,
                device=self.device,
            )
        )
        exposure = global_parameters[: self.star_flat_offset].reshape(
            self.n_exposure, 3
        )
        flat = global_parameters[
            self.star_flat_offset : self.color_term_offset
        ].reshape(self.n_detector, self.graph.max_star_flat)
        diagnostics = PhotometricCalibrationDiagnostics(
            converged=converged,
            iterations=len(steps),
            objective_history=tuple(objectives),
            step_history=tuple(steps),
            cg_iterations=tuple(cg_iterations),
            connected_components=tuple(
                tuple(self.graph.exposure_ids[index] for index in component)
                for component in self.component_exposures
            ),
            gauge_exposures=tuple(
                self.graph.exposure_ids[index]
                for index in self.gauge_exposure_indices
            ),
            reference_anchored_components=self.component_anchored,
            dropped_extinction_exposures=self.dropped["extinction"],
            dropped_cloud_exposures=self.dropped["cloud"],
            dropped_star_flat_parameters=self.dropped["flat"],
            dropped_color_bands=self.dropped["color"],
            information_rank=rank,
            information_dimension=int(joint_active.sum().item()),
            condition_number=condition,
        )
        provenance: dict[str, object] = {
            "method": "supplied-catalog-coupled-log-flux-calibration",
            "selection": "fixed-astrometric-associations",
            "source_creation": False,
            "detection": False,
            "solver": "alternating-matrix-free-cg-irls",
            "flux_gauge": "standards-or-first-exposure-per-component",
            "covariance": "full-within-measurement-block",
            "robust_loss": "Huber",
            "variable_sources_in_global_fit": False,
            "sed_basis_width": self.source_width,
        }
        return PhotometricCalibrationSolution(
            source_ids=self.graph.astrometry_graph.source_ids,
            bands=self.graph.bands,
            source_band_flux=torch.exp(log_flux),
            source_band_log_flux=log_flux,
            source_coefficient_variance=source_variance,
            colors=colors,
            exposure_ids=self.graph.exposure_ids,
            exposure_log_zeropoint=exposure[:, 0],
            exposure_extinction=exposure[:, 1],
            exposure_cloud=exposure[:, 2],
            detector_ids=self.graph.detector_ids,
            star_flat_coefficients=flat,
            band_color_terms=global_parameters[self.color_term_offset :],
            predicted_log_flux=prediction,
            residual_log_flux=residual,
            robust_weights=robust,
            variability_offsets=variability,
            fit_mask=self.fit_mask,
            global_covariance=global_covariance,
            diagnostics=diagnostics,
            provenance=provenance,
        )


class _CoupledSolver:
    def __init__(self, graph: CoupledCalibrationGraph, config: CoupledCalibrationConfig):
        self.graph = graph
        self.config = config

    def _astrometry_fit_mask(self) -> Tensor:
        source_holdout = set(self.config.holdout_source_ids)
        exposure_holdout = set(self.config.holdout_exposure_ids)
        graph = self.graph.astrometry_graph
        source_ids = graph.source_ids[graph.row_source]
        source_mask = torch.tensor(
            [int(value) not in source_holdout for value in source_ids.cpu()],
            dtype=torch.bool,
            device=source_ids.device,
        )
        exposure_mask = torch.tensor(
            [
                graph.exposure_ids[int(value)] not in exposure_holdout
                for value in graph.row_exposure.cpu()
            ],
            dtype=torch.bool,
            device=source_ids.device,
        )
        return source_mask & exposure_mask

    def solve(self) -> CoupledCalibrationSolution:
        fit_mask = self._astrometry_fit_mask()
        initial_config = replace(
            self.config.astrometry, fit_dcr=False, fit_lateral_color=False
        )
        coupled_astrometry_config = replace(
            self.config.astrometry,
            color_pivot=self.config.color_pivot,
            missing_color=self.config.missing_color,
        )
        astrometry = self.graph.astrometry_graph.solve(
            initial_config, fit_mask=fit_mask
        )
        stage_history = ["astrometry"]
        astrometric_steps = []
        photometric_steps = []
        color_steps = []
        photometry: PhotometricCalibrationSolution | None = None
        converged = False
        previous_pose = astrometry.pose_parameters
        previous_global: Tensor | None = None
        previous_colors: Tensor | None = None
        for _ in range(self.config.max_iterations):
            photometry = _PhotometrySolver(self.graph, self.config).solve()
            stage_history.append("photometry/color")
            colors = photometry.colors
            astrometry = self.graph.astrometry_graph.solve(
                coupled_astrometry_config,
                source_colors=colors,
                fit_mask=fit_mask,
            )
            stage_history.extend(("DCR/lateral-color", "astrometry"))
            pose_step = float(
                (
                    torch.linalg.vector_norm(astrometry.pose_parameters - previous_pose)
                    / torch.linalg.vector_norm(astrometry.pose_parameters).clamp_min(1.0)
                )
                .detach()
                .cpu()
                .item()
            )
            current_global = torch.cat(
                (
                    photometry.exposure_log_zeropoint,
                    photometry.exposure_extinction,
                    photometry.exposure_cloud,
                    photometry.star_flat_coefficients.reshape(-1),
                    photometry.band_color_terms,
                )
            )
            if previous_global is None:
                photometric_step = math.inf
                color_step = math.inf
            else:
                photometric_step = float(
                    (
                        torch.linalg.vector_norm(current_global - previous_global)
                        / torch.linalg.vector_norm(current_global).clamp_min(1.0)
                    )
                    .detach()
                    .cpu()
                    .item()
                )
                color_step = float(
                    (
                        torch.linalg.vector_norm(
                            torch.nan_to_num(colors) - torch.nan_to_num(previous_colors)
                        )
                        / torch.linalg.vector_norm(torch.nan_to_num(colors)).clamp_min(1.0)
                    )
                    .detach()
                    .cpu()
                    .item()
                )
            astrometric_steps.append(pose_step)
            photometric_steps.append(photometric_step)
            color_steps.append(color_step)
            previous_pose = astrometry.pose_parameters
            previous_global = current_global
            previous_colors = colors
            if (
                pose_step <= self.config.convergence_tolerance
                and photometric_step <= self.config.convergence_tolerance
                and color_step <= self.config.convergence_tolerance
            ):
                converged = astrometry.converged and photometry.converged
                break
        if photometry is None:
            raise RuntimeError("coupled calibration did not execute photometry")
        diagnostics = CoupledCalibrationDiagnostics(
            converged=converged,
            iterations=len(astrometric_steps),
            stage_history=tuple(stage_history),
            astrometric_step_history=tuple(astrometric_steps),
            photometric_step_history=tuple(photometric_steps),
            color_step_history=tuple(color_steps),
            source_holdout_count=len(self.config.holdout_source_ids),
            exposure_holdout_count=len(self.config.holdout_exposure_ids),
        )
        covariance: dict[str, object] = {
            "astrometry_exposure": astrometry.exposure_covariance,
            "photometry_global": photometry.global_covariance,
            "photometry_source_coefficient_variance": photometry.source_coefficient_variance,
        }
        provenance: dict[str, object] = {
            "method": "alternating-coupled-global-astrometric-photometric-calibration",
            "stages": (
                "astrometry",
                "photometry/color",
                "DCR/lateral-color",
                "astrometry",
            ),
            "geometry_gauge": "reference-catalog-or-component-pose",
            "flux_gauge": "standards-or-relative-component-zeropoint",
            "associations": "caller-supplied-immutable",
            "source_creation": False,
            "detection": False,
            "photometry_rescues_associations": False,
            "variability_enters_astrometry": False,
        }
        return CoupledCalibrationSolution(
            astrometry=astrometry,
            photometry=photometry,
            diagnostics=diagnostics,
            covariance=covariance,
            provenance=provenance,
        )


def solve_coupled_calibration(
    graph: CoupledCalibrationGraph,
    config: CoupledCalibrationConfig | None = None,
) -> CoupledCalibrationSolution:
    """Solve the coupled astrometric and photometric calibration graph."""
    return graph.solve(config)


__all__ = [
    "CoupledCalibrationConfig",
    "CoupledCalibrationDiagnostics",
    "CoupledCalibrationGraph",
    "CoupledCalibrationSolution",
    "PhotometricCalibrationDiagnostics",
    "PhotometricCalibrationSolution",
    "PhotometricMeasurementBlock",
    "PhotometricReference",
    "photometric_log_residuals",
    "solve_coupled_calibration",
]
