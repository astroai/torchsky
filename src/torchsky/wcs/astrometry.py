"""Supplied-catalog global astrometry with block-Schur bundle adjustment."""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from torch import Tensor

from ..catalogs import HealpixIndex
from ..sphere import great_circle_distance
from ._graph import _UnionFind
from .alignment import TangentPlaneFrame
from .semantics import Epoch, TimeSpec

PoseModel = Literal["shift", "rshift", "rscale", "affine", "general"]
MissingPolicy = Literal["error", "zero", "drop"]
RankPolicy = Literal["error", "drop"]
CovarianceSpace = Literal["detector", "tangent"]


def _as_real_tensor(value: Any, name: str) -> Tensor:
    tensor = torch.as_tensor(value)
    if not tensor.is_floating_point() or tensor.is_complex():
        raise ValueError(f"{name} must be a real floating-point tensor")
    if not bool(torch.isfinite(tensor).all().detach().cpu().item()):
        raise ValueError(f"{name} must contain only finite values")
    return tensor


def _covariance(value: Any, n: int, name: str, like: Tensor) -> Tensor:
    covariance = torch.as_tensor(value, dtype=like.dtype, device=like.device)
    if covariance.shape == (2, 2):
        covariance = covariance.expand(n, 2, 2).clone()
    if covariance.shape != (n, 2, 2):
        raise ValueError(f"{name} must have shape (2, 2) or (N, 2, 2)")
    if not bool(torch.isfinite(covariance).all().detach().cpu().item()):
        raise ValueError(f"{name} must contain only finite values")
    symmetric = 0.5 * (covariance + covariance.mT)
    _, info = torch.linalg.cholesky_ex(symmetric)
    if bool((info != 0).any().detach().cpu().item()):
        raise ValueError(f"{name} must be symmetric positive definite")
    return symmetric


def _call_wcs(wcs: Any, detector_xy: Tensor) -> Tensor:
    if hasattr(wcs, "pixel_to_world"):
        try:
            result = wcs.pixel_to_world(detector_xy[..., 0], detector_xy[..., 1])
        except (TypeError, ValueError):
            result = wcs.pixel_to_world(detector_xy)
    elif callable(wcs):
        result = wcs(detector_xy)
    else:
        raise TypeError("approximate_wcs must be callable or provide pixel_to_world")
    if isinstance(result, tuple):
        if len(result) != 2:
            raise ValueError("approximate_wcs must return two sky coordinates")
        world = torch.stack(
            tuple(
                torch.as_tensor(
                    item, dtype=detector_xy.dtype, device=detector_xy.device
                )
                for item in result
            ),
            dim=-1,
        )
    else:
        world = torch.as_tensor(
            result, dtype=detector_xy.dtype, device=detector_xy.device
        )
    if world.shape != detector_xy.shape:
        raise ValueError("approximate_wcs output must have shape (N, 2)")
    if not bool(torch.isfinite(world).all().detach().cpu().item()):
        raise ValueError("approximate_wcs produced non-finite sky coordinates")
    return world


@dataclass(frozen=True)
class AstrometricObservation:
    """One exposure's supplied centroid catalog and match identifiers.

    ``sky_position`` is the approximate sky position in degrees. When omitted,
    it is evaluated from ``detector_xy`` through ``approximate_wcs``. A tangent
    covariance is in square arcseconds; a detector covariance is in square
    pixels and is propagated with a local approximate-WCS Jacobian.
    """

    exposure_id: Hashable
    detector_xy: Tensor
    source_ids: Tensor | None
    centroid_covariance: Tensor
    approximate_wcs: Any | None = field(default=None, repr=False, compare=False)
    sky_position: Tensor | None = None
    epoch: Epoch | Tensor | float | None = None
    detector_id: Hashable = "detector"
    covariance_space: CovarianceSpace = "detector"
    world_jacobian: Tensor | None = None
    color: Tensor | None = None
    dcr_basis: Tensor | None = None
    lateral_color_basis: Tensor | None = None
    zenith_angle_deg: Tensor | float | None = None
    parallactic_angle_deg: Tensor | float | None = None
    parallax_factor: Tensor | None = None

    def __post_init__(self) -> None:
        points = _as_real_tensor(self.detector_xy, "detector_xy")
        if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0:
            raise ValueError("detector_xy must have non-empty shape (N, 2)")
        n = int(points.shape[0])
        covariance = _covariance(
            self.centroid_covariance, n, "centroid_covariance", points
        )
        if self.source_ids is not None:
            identifiers = torch.as_tensor(
                self.source_ids, dtype=torch.int64, device=points.device
            )
            if identifiers.shape != (n,):
                raise ValueError("source_ids must have shape (N,)")
            object.__setattr__(self, "source_ids", identifiers)
        if self.sky_position is None and self.approximate_wcs is None:
            raise ValueError("sky_position or approximate_wcs is required")
        sky = None
        if self.sky_position is not None:
            sky = _as_real_tensor(self.sky_position, "sky_position").to(
                device=points.device, dtype=points.dtype
            )
            if sky.shape != points.shape:
                raise ValueError("sky_position must have shape (N, 2)")
        if self.covariance_space not in {"detector", "tangent"}:
            raise ValueError("covariance_space must be 'detector' or 'tangent'")
        jacobian = None
        if self.world_jacobian is not None:
            jacobian = _as_real_tensor(self.world_jacobian, "world_jacobian").to(
                device=points.device, dtype=points.dtype
            )
            if jacobian.shape != (n, 2, 2):
                raise ValueError("world_jacobian must have shape (N, 2, 2)")
        for name in ("dcr_basis", "lateral_color_basis", "parallax_factor"):
            value = getattr(self, name)
            if value is not None:
                tensor = torch.as_tensor(
                    value, dtype=points.dtype, device=points.device
                )
                if tensor.shape == (2,):
                    tensor = tensor.expand(n, 2).clone()
                if tensor.shape != (n, 2) or not bool(torch.isfinite(tensor).all()):
                    raise ValueError(f"{name} must be finite with shape (2,) or (N, 2)")
                object.__setattr__(self, name, tensor)
        if self.dcr_basis is None and (
            self.zenith_angle_deg is not None or self.parallactic_angle_deg is not None
        ):
            if self.zenith_angle_deg is None or self.parallactic_angle_deg is None:
                raise ValueError(
                    "zenith_angle_deg and parallactic_angle_deg are required together"
                )
            zenith = torch.as_tensor(
                self.zenith_angle_deg, dtype=points.dtype, device=points.device
            )
            angle = torch.as_tensor(
                self.parallactic_angle_deg, dtype=points.dtype, device=points.device
            )
            zenith = torch.broadcast_to(zenith, (n,))
            angle = torch.broadcast_to(angle, (n,))
            if not bool((torch.isfinite(zenith) & torch.isfinite(angle)).all()):
                raise ValueError("DCR angles must be finite")
            if not bool(((zenith >= 0.0) & (zenith < 90.0)).all()):
                raise ValueError("zenith_angle_deg must lie in [0, 90)")
            radians = angle * (torch.pi / 180.0)
            amplitude = torch.tan(zenith * (torch.pi / 180.0))
            object.__setattr__(
                self,
                "dcr_basis",
                amplitude.unsqueeze(-1)
                * torch.stack((torch.sin(radians), torch.cos(radians)), dim=-1),
            )
        if self.color is not None:
            color = torch.as_tensor(
                self.color, dtype=points.dtype, device=points.device
            )
            if color.shape != (n,):
                raise ValueError("color must have shape (N,)")
            object.__setattr__(self, "color", color)
        object.__setattr__(self, "detector_xy", points)
        object.__setattr__(self, "centroid_covariance", covariance)
        object.__setattr__(self, "sky_position", sky)
        object.__setattr__(self, "world_jacobian", jacobian)

    @property
    def n_matches(self) -> int:
        return int(self.detector_xy.shape[0])

    def approximate_sky(self) -> Tensor:
        if self.sky_position is not None:
            return self.sky_position
        return _call_wcs(self.approximate_wcs, self.detector_xy)


@dataclass(frozen=True)
class AstrometryConfig:
    pose_model: PoseModel = "rscale"
    max_iterations: int = 15
    convergence_tolerance: float = 1.0e-8
    huber_delta: float = 3.0
    cg_max_iterations: int = 200
    cg_rtol: float = 1.0e-10
    cg_atol: float = 0.0
    damping: float = 1.0e-12
    fit_detector_grid: bool = False
    detector_grid_shape: tuple[int, int] = (4, 4)
    detector_grid_regularization: float = 1.0e-8
    fit_dcr: bool = False
    fit_lateral_color: bool = False
    color_pivot: float = 0.0
    missing_color: MissingPolicy = "error"
    fit_proper_motion: bool = False
    fit_parallax: bool = False
    reference_epoch: float = 2000.0
    epoch_format: Literal["jd", "mjd", "jyear", "byear", "unix"] = "jyear"
    epoch_scale: Literal["utc", "tai", "tt", "gps", "ut1", "tcb", "tdb"] = "tcb"
    missing_epoch: Literal["error", "reference"] = "error"
    rank_policy: RankPolicy = "error"
    rank_rtol: float = 1.0e-10
    covariance_max_parameters: int = 256
    covariance_jacobian_step: float = 1.0e-3

    def __post_init__(self) -> None:
        if self.pose_model not in {"shift", "rshift", "rscale", "affine", "general"}:
            raise ValueError("unsupported pose_model")
        if self.max_iterations < 1 or self.cg_max_iterations < 1:
            raise ValueError("iteration limits must be positive")
        for name in (
            "convergence_tolerance",
            "huber_delta",
            "cg_rtol",
            "rank_rtol",
            "covariance_jacobian_step",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.cg_atol < 0.0 or not math.isfinite(self.cg_atol):
            raise ValueError("cg_atol must be finite and non-negative")
        for name in ("damping", "detector_grid_regularization"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if min(self.detector_grid_shape) < 2:
            raise ValueError("detector_grid_shape axes must be at least 2")
        if self.missing_color not in {"error", "zero", "drop"}:
            raise ValueError("unsupported missing_color policy")
        if self.rank_policy not in {"error", "drop"}:
            raise ValueError("unsupported rank_policy")
        if self.covariance_max_parameters < 1:
            raise ValueError("covariance_max_parameters must be positive")
        if not math.isfinite(self.reference_epoch):
            raise ValueError("reference_epoch must be finite")


@dataclass(frozen=True)
class AstrometryDiagnostics:
    converged: bool
    iterations: int
    objective_history: tuple[float, ...]
    step_history: tuple[float, ...]
    cg_iterations: tuple[int, ...]
    reduced_rank: int | None
    reduced_dimension: int
    condition_number: float | None
    connected_components: tuple[tuple[Hashable, ...], ...]
    gauge_exposures: tuple[Hashable, ...]
    kinematic_gauge_sources: tuple[int, ...]
    reference_anchored_components: tuple[bool, ...]
    dropped_source_blocks: tuple[int, ...]
    dropped_dcr_exposures: tuple[Hashable, ...]
    dropped_lateral_color_detectors: tuple[Hashable, ...]
    matrix_free_reduced: bool = True
    assembled_observation_matrix: bool = False


@dataclass(frozen=True)
class AstrometrySolution:
    exposure_ids: tuple[Hashable, ...]
    source_ids: Tensor
    pose_model: str
    pose_parameters: Tensor
    source_positions: Tensor
    source_tangent_positions: Tensor
    proper_motion_mas_per_year: Tensor | None
    parallax_mas: Tensor | None
    dcr_coefficients: Tensor | None
    lateral_color_coefficients: Tensor | None
    detector_ids: tuple[Hashable, ...]
    detector_residual_control: Tensor | None
    residuals_arcsec: Tensor
    robust_weights: Tensor
    exposure_covariance: dict[Hashable, Tensor]
    diagnostics: AstrometryDiagnostics
    fit_mask: Tensor

    @property
    def converged(self) -> bool:
        return self.diagnostics.converged

    def pose_for(self, exposure_id: Hashable) -> Tensor:
        return self.pose_parameters[self.exposure_ids.index(exposure_id)]


@dataclass(frozen=True)
class ApproximateMatchResult:
    reference_indices: Tensor
    separation_arcsec: Tensor

    @property
    def matched(self) -> Tensor:
        return self.reference_indices >= 0


def tangent_plane_block_residuals(
    observed: Any,
    predicted: Any,
    centroid_covariance: Any,
    reference_covariance: Any | None = None,
    *,
    robust_weights: Any | None = None,
    whiten: bool = True,
) -> Tensor:
    """Return differentiable two-coordinate residual blocks with full covariance."""
    obs = _as_real_tensor(observed, "observed")
    pred = torch.as_tensor(predicted, dtype=obs.dtype, device=obs.device)
    if obs.shape != pred.shape or obs.shape[-1:] != (2,):
        raise ValueError("observed and predicted must have matching (..., 2) shape")
    covariance = torch.as_tensor(
        centroid_covariance, dtype=obs.dtype, device=obs.device
    )
    if covariance.shape != obs.shape[:-1] + (2, 2):
        covariance = torch.broadcast_to(covariance, obs.shape[:-1] + (2, 2))
    if reference_covariance is not None:
        reference = torch.as_tensor(
            reference_covariance, dtype=obs.dtype, device=obs.device
        )
        covariance = covariance + torch.broadcast_to(reference, covariance.shape)
    residual = obs - pred
    if whiten:
        factor, info = torch.linalg.cholesky_ex(covariance)
        if bool((info != 0).any().detach().cpu().item()):
            raise ValueError("combined covariance must be positive definite")
        residual = torch.linalg.solve_triangular(
            factor, residual.unsqueeze(-1), upper=False
        ).squeeze(-1)
    if robust_weights is not None:
        weights = torch.as_tensor(robust_weights, dtype=obs.dtype, device=obs.device)
        residual = residual * torch.sqrt(weights).unsqueeze(-1)
    return residual


def match_approximate_wcs(
    observation: AstrometricObservation,
    reference_sky: Any,
    *,
    radius_arcsec: float,
    index: HealpixIndex | None = None,
) -> ApproximateMatchResult:
    """Nearest supplied-reference matching, using only ``HealpixIndex`` candidates."""
    reference = _as_real_tensor(reference_sky, "reference_sky").to(
        device=observation.detector_xy.device, dtype=observation.detector_xy.dtype
    )
    if reference.ndim != 2 or reference.shape[1] != 2:
        raise ValueError("reference_sky must have shape (N, 2)")
    if not math.isfinite(radius_arcsec) or radius_arcsec <= 0.0:
        raise ValueError("radius_arcsec must be finite and positive")
    sky = observation.approximate_sky()
    hp_index = index
    if hp_index is None:
        hp_index = HealpixIndex.build(reference[:, 0], reference[:, 1])
    elif hp_index.lon.numel() != reference.shape[0]:
        raise ValueError("index and reference_sky row counts differ")
    matched = torch.full((sky.shape[0],), -1, dtype=torch.int64, device=sky.device)
    separation = torch.full(
        (sky.shape[0],), float("inf"), dtype=sky.dtype, device=sky.device
    )
    radius_deg = float(radius_arcsec) / 3600.0
    for row in range(int(sky.shape[0])):
        candidates = hp_index.query_disc(sky[row, 0], sky[row, 1], radius_deg)
        if candidates.numel() == 0:
            continue
        distances = great_circle_distance(
            sky[row, 0],
            sky[row, 1],
            reference[candidates, 0],
            reference[candidates, 1],
            degrees=True,
        )
        closest = torch.argmin(distances)
        matched[row] = candidates[closest]
        separation[row] = distances[closest] * 3600.0
    return ApproximateMatchResult(matched, separation)


class AstrometricMatchGraph:
    """Bipartite exposure/source graph for supplied matches and reference priors."""

    def __init__(
        self,
        observations: Sequence[AstrometricObservation],
        *,
        reference_source_ids: Any | None = None,
        reference_positions: Any | None = None,
        reference_covariance: Any | None = None,
        reference_proper_motion_mas_per_year: Any | None = None,
        reference_proper_motion_covariance: Any | None = None,
        reference_parallax_mas: Any | None = None,
        reference_parallax_variance: Any | None = None,
        tangent_frame: TangentPlaneFrame | None = None,
    ) -> None:
        items = tuple(observations)
        if not items:
            raise ValueError("at least one observation is required")
        if any(item.source_ids is None for item in items):
            raise ValueError(
                "supplied source_ids are required; approximate matching is opt-in"
            )
        exposure_ids = tuple(item.exposure_id for item in items)
        if len(set(exposure_ids)) != len(exposure_ids):
            raise ValueError("exposure_id values must be unique")
        device = items[0].detector_xy.device
        dtype = items[0].detector_xy.dtype
        for item in items:
            if item.detector_xy.device != device or item.detector_xy.dtype != dtype:
                raise ValueError("all observations must share dtype and device")
        all_ids = torch.cat(
            [item.source_ids for item in items if item.source_ids is not None]
        )
        source_ids = torch.unique(all_ids, sorted=True)
        row_source = torch.cat(
            [
                torch.searchsorted(source_ids, ids)
                for item in items
                if (ids := item.source_ids) is not None  # all present (checked above)
            ]
        )
        row_exposure = torch.cat(
            [
                torch.full((item.n_matches,), index, dtype=torch.int64, device=device)
                for index, item in enumerate(items)
            ]
        )
        sky = torch.cat([item.approximate_sky() for item in items])
        if tangent_frame is None:
            frame_points = sky
            if reference_positions is not None:
                reference_for_frame = _as_real_tensor(
                    reference_positions, "reference_positions"
                ).to(device=device, dtype=dtype)
                frame_points = torch.cat((frame_points, reference_for_frame))
            tangent_frame = TangentPlaneFrame.from_points(
                frame_points[:, 0], frame_points[:, 1]
            )
        plane_x, plane_y = tangent_frame.world_to_plane(sky[:, 0], sky[:, 1])
        observed_plane = torch.stack((plane_x, plane_y), dim=-1) * 3600.0
        detector_xy = torch.cat([item.detector_xy for item in items])
        centroid_covariance = self._plane_covariance(items, tangent_frame)
        detector_ids = tuple(dict.fromkeys(item.detector_id for item in items))
        detector_lookup = {value: index for index, value in enumerate(detector_ids)}
        row_detector = torch.cat(
            [
                torch.full(
                    (item.n_matches,),
                    detector_lookup[item.detector_id],
                    dtype=torch.int64,
                    device=device,
                )
                for item in items
            ]
        )

        n_source = int(source_ids.numel())
        reference_plane = torch.zeros((n_source, 2), dtype=dtype, device=device)
        reference_precision = torch.zeros((n_source, 2, 2), dtype=dtype, device=device)
        reference_mask = torch.zeros(n_source, dtype=torch.bool, device=device)
        reference_motion = torch.zeros((n_source, 2), dtype=dtype, device=device)
        reference_motion_precision = torch.zeros(
            (n_source, 2, 2), dtype=dtype, device=device
        )
        reference_motion_mask = torch.zeros(n_source, dtype=torch.bool, device=device)
        reference_parallax = torch.zeros(n_source, dtype=dtype, device=device)
        reference_parallax_precision = torch.zeros(n_source, dtype=dtype, device=device)
        reference_parallax_mask = torch.zeros(n_source, dtype=torch.bool, device=device)
        if any(
            value is not None
            for value in (
                reference_source_ids,
                reference_positions,
                reference_covariance,
            )
        ):
            if (
                reference_source_ids is None
                or reference_positions is None
                or reference_covariance is None
            ):
                raise ValueError(
                    "reference_source_ids, reference_positions, and "
                    "reference_covariance are required together"
                )
            ref_ids = torch.as_tensor(
                reference_source_ids, dtype=torch.int64, device=device
            )
            positions = _as_real_tensor(reference_positions, "reference_positions").to(
                device=device, dtype=dtype
            )
            if ref_ids.ndim != 1 or positions.shape != (ref_ids.numel(), 2):
                raise ValueError("reference catalog arrays have inconsistent shapes")
            covariance = _covariance(
                reference_covariance,
                int(ref_ids.numel()),
                "reference_covariance",
                positions,
            )
            locations = torch.searchsorted(source_ids, ref_ids)
            valid = locations < n_source
            safe = locations.clamp(max=max(n_source - 1, 0))
            valid &= source_ids[safe] == ref_ids
            if bool(valid.any()):
                x_ref, y_ref = tangent_frame.world_to_plane(
                    positions[valid, 0], positions[valid, 1]
                )
                selected = safe[valid]
                reference_plane[selected] = torch.stack((x_ref, y_ref), dim=-1) * 3600.0
                reference_precision[selected] = torch.linalg.inv(covariance[valid])
                reference_mask[selected] = True
                if reference_proper_motion_mas_per_year is not None:
                    if reference_proper_motion_covariance is None:
                        raise ValueError(
                            "reference_proper_motion_covariance is required with "
                            "reference_proper_motion_mas_per_year"
                        )
                    motion = _as_real_tensor(
                        reference_proper_motion_mas_per_year,
                        "reference_proper_motion_mas_per_year",
                    ).to(device=device, dtype=dtype)
                    if motion.shape != positions.shape:
                        raise ValueError(
                            "reference_proper_motion_mas_per_year must have shape (N, 2)"
                        )
                    motion_covariance = _covariance(
                        reference_proper_motion_covariance,
                        int(ref_ids.numel()),
                        "reference_proper_motion_covariance",
                        positions,
                    )
                    reference_motion[selected] = motion[valid] / 1000.0
                    reference_motion_precision[selected] = (
                        torch.linalg.inv(motion_covariance[valid]) * 1.0e6
                    )
                    reference_motion_mask[selected] = True
                if reference_parallax_mas is not None:
                    if reference_parallax_variance is None:
                        raise ValueError(
                            "reference_parallax_variance is required with "
                            "reference_parallax_mas"
                        )
                    parallax = torch.as_tensor(
                        reference_parallax_mas, dtype=dtype, device=device
                    )
                    variance = torch.as_tensor(
                        reference_parallax_variance, dtype=dtype, device=device
                    )
                    parallax = torch.broadcast_to(parallax, (ref_ids.numel(),))
                    variance = torch.broadcast_to(variance, (ref_ids.numel(),))
                    if not bool(
                        (
                            torch.isfinite(parallax)
                            & torch.isfinite(variance)
                            & (variance > 0)
                        ).all()
                    ):
                        raise ValueError(
                            "reference parallax values must be finite with positive variance"
                        )
                    reference_parallax[selected] = parallax[valid] / 1000.0
                    reference_parallax_precision[selected] = 1.0e6 / variance[valid]
                    reference_parallax_mask[selected] = True
        elif any(
            value is not None
            for value in (
                reference_proper_motion_mas_per_year,
                reference_proper_motion_covariance,
                reference_parallax_mas,
                reference_parallax_variance,
            )
        ):
            raise ValueError("kinematic reference priors require reference positions")

        components, component_exposures, anchored = self._components(
            len(items), n_source, row_exposure, row_source, reference_mask
        )
        self.observations = items
        self.exposure_ids = exposure_ids
        self.detector_ids = detector_ids
        self.source_ids = source_ids
        self.tangent_frame = tangent_frame
        self.row_exposure = row_exposure
        self.row_source = row_source
        self.row_detector = row_detector
        self.detector_xy = detector_xy
        self.observed_sky = sky
        self.observed_plane = observed_plane
        self.centroid_covariance = centroid_covariance
        self.reference_plane = reference_plane
        self.reference_precision = reference_precision
        self.reference_mask = reference_mask
        self.reference_motion = reference_motion
        self.reference_motion_precision = reference_motion_precision
        self.reference_motion_mask = reference_motion_mask
        self.reference_parallax = reference_parallax
        self.reference_parallax_precision = reference_parallax_precision
        self.reference_parallax_mask = reference_parallax_mask
        self.components = components
        self.component_exposures = component_exposures
        self.component_reference_anchored = anchored

    @staticmethod
    def _components(
        n_exposure: int,
        n_source: int,
        row_exposure: Tensor,
        row_source: Tensor,
        reference_mask: Tensor,
    ) -> tuple[
        tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...], tuple[bool, ...]
    ]:
        union = _UnionFind(n_exposure + n_source)
        for exposure, source in zip(
            row_exposure.detach().cpu().tolist(),
            row_source.detach().cpu().tolist(),
            strict=True,
        ):
            union.union(int(exposure), n_exposure + int(source))
        groups: dict[int, list[int]] = {}
        for node in range(n_exposure + n_source):
            groups.setdefault(union.find(node), []).append(node)
        used_roots = sorted({union.find(int(value)) for value in row_exposure.cpu()})
        components = []
        exposures = []
        anchored = []
        ref_cpu = reference_mask.detach().cpu()
        for root in used_roots:
            nodes = groups[root]
            exp = tuple(node for node in nodes if node < n_exposure)
            src = tuple(node - n_exposure for node in nodes if node >= n_exposure)
            components.append(tuple(nodes))
            exposures.append(exp)
            anchored.append(any(bool(ref_cpu[source]) for source in src))
        return tuple(components), tuple(exposures), tuple(anchored)

    @staticmethod
    def _plane_covariance(
        observations: tuple[AstrometricObservation, ...],
        frame: TangentPlaneFrame,
    ) -> Tensor:
        blocks = []
        for observation in observations:
            covariance = observation.centroid_covariance
            if observation.covariance_space == "tangent":
                blocks.append(covariance)
                continue
            if observation.world_jacobian is not None:
                world = observation.approximate_sky()
                angular_step = 1.0e-6

                def project_world(value: Tensor) -> Tensor:
                    x, y = frame.world_to_plane(value[:, 0], value[:, 1])
                    return torch.stack((x, y), dim=-1) * 3600.0

                plus_ra = world.clone()
                minus_ra = world.clone()
                plus_dec = world.clone()
                minus_dec = world.clone()
                plus_ra[:, 0] += angular_step
                minus_ra[:, 0] -= angular_step
                plus_dec[:, 1] += angular_step
                minus_dec[:, 1] -= angular_step
                projection_jacobian = torch.stack(
                    (
                        (project_world(plus_ra) - project_world(minus_ra))
                        / (2.0 * angular_step),
                        (project_world(plus_dec) - project_world(minus_dec))
                        / (2.0 * angular_step),
                    ),
                    dim=-1,
                )
                jacobian = projection_jacobian @ observation.world_jacobian
            else:
                if observation.approximate_wcs is None:
                    raise ValueError(
                        "detector covariance needs approximate_wcs or world_jacobian"
                    )
                step = 1.0e-3
                plus_x = observation.detector_xy.clone()
                minus_x = observation.detector_xy.clone()
                plus_y = observation.detector_xy.clone()
                minus_y = observation.detector_xy.clone()
                plus_x[:, 0] += step
                minus_x[:, 0] -= step
                plus_y[:, 1] += step
                minus_y[:, 1] -= step

                def project(points: Tensor) -> Tensor:
                    world = _call_wcs(observation.approximate_wcs, points)
                    x, y = frame.world_to_plane(world[:, 0], world[:, 1])
                    return torch.stack((x, y), dim=-1) * 3600.0

                dx = (project(plus_x) - project(minus_x)) / (2.0 * step)
                dy = (project(plus_y) - project(minus_y)) / (2.0 * step)
                jacobian = torch.stack((dx, dy), dim=-1)
            propagated = jacobian @ covariance @ jacobian.mT
            blocks.append(0.5 * (propagated + propagated.mT))
        result = torch.cat(blocks)
        _, info = torch.linalg.cholesky_ex(result)
        if bool((info != 0).any().detach().cpu().item()):
            raise ValueError("propagated centroid covariance is not positive definite")
        return result

    def solve(
        self,
        config: AstrometryConfig | None = None,
        *,
        source_colors: Tensor | None = None,
        fit_mask: Tensor | None = None,
    ) -> AstrometrySolution:
        return _AstrometrySolver(
            self,
            AstrometryConfig() if config is None else config,
            source_colors=source_colors,
            fit_mask=fit_mask,
        ).solve()


def _pose_dimension(model: str) -> int:
    return {"shift": 2, "rshift": 3, "rscale": 4, "affine": 6, "general": 6}[model]


def _pose_value_and_jacobian(
    points: Tensor, parameters: Tensor, model: str
) -> tuple[Tensor, Tensor]:
    n = int(points.shape[0])
    dtype = points.dtype
    device = points.device
    jacobian = torch.zeros((n, 2, _pose_dimension(model)), dtype=dtype, device=device)
    jacobian[:, 0, 0] = 1.0
    jacobian[:, 1, 1] = 1.0
    if model == "shift":
        return points + parameters[:2], jacobian
    if model in {"rshift", "rscale"}:
        theta = parameters[2]
        scale = torch.exp(parameters[3]) if model == "rscale" else points.new_ones(())
        cosine = torch.cos(theta)
        sine = torch.sin(theta)
        matrix = scale * torch.stack(
            (torch.stack((cosine, -sine)), torch.stack((sine, cosine)))
        )
        transformed = points @ matrix.mT + parameters[:2]
        derivative = scale * torch.stack(
            (torch.stack((-sine, -cosine)), torch.stack((cosine, -sine)))
        )
        jacobian[:, :, 2] = points @ derivative.mT
        if model == "rscale":
            jacobian[:, :, 3] = points @ matrix.mT
        return transformed, jacobian
    correction = parameters[2:].reshape(2, 2)
    transformed = points + points @ correction.mT + parameters[:2]
    jacobian[:, 0, 2] = points[:, 0]
    jacobian[:, 0, 3] = points[:, 1]
    jacobian[:, 1, 4] = points[:, 0]
    jacobian[:, 1, 5] = points[:, 1]
    return transformed, jacobian


def _epoch_jyear(
    value: Epoch | Tensor | float | None,
    config: AstrometryConfig,
    like: Tensor,
    count: int,
) -> Tensor:
    if value is None:
        if config.missing_epoch == "error" and (
            config.fit_proper_motion or config.fit_parallax
        ):
            raise ValueError("epoch is required when fitting source motion")
        return torch.full(
            (count,), config.reference_epoch, dtype=like.dtype, device=like.device
        )
    spec = TimeSpec(format="jyear", scale=config.epoch_scale)
    if isinstance(value, Epoch):
        epoch = value.to(spec).value.to(dtype=like.dtype, device=like.device)
    else:
        epoch = (
            Epoch(
                torch.as_tensor(value, dtype=like.dtype, device=like.device),
                TimeSpec(format=config.epoch_format, scale=config.epoch_scale),
            )
            .to(spec)
            .value
        )
    return torch.broadcast_to(epoch, (count,)).clone()


class _AstrometrySolver:
    def __init__(
        self,
        graph: AstrometricMatchGraph,
        config: AstrometryConfig,
        *,
        source_colors: Tensor | None = None,
        fit_mask: Tensor | None = None,
    ) -> None:
        self.graph = graph
        self.config = config
        self.dtype = graph.observed_plane.dtype
        self.device = graph.observed_plane.device
        self.n_row = int(graph.row_source.numel())
        self.n_exposure = len(graph.exposure_ids)
        self.n_source = int(graph.source_ids.numel())
        if fit_mask is None:
            self.fit_mask = torch.ones(
                self.n_row, dtype=torch.bool, device=self.device
            )
        else:
            self.fit_mask = torch.as_tensor(
                fit_mask, dtype=torch.bool, device=self.device
            )
            if self.fit_mask.shape != (self.n_row,):
                raise ValueError("fit_mask must have shape (N,) for graph rows")
        self.source_colors = None
        if source_colors is not None:
            self.source_colors = torch.as_tensor(
                source_colors, dtype=self.dtype, device=self.device
            )
            if self.source_colors.shape != (self.n_source,):
                raise ValueError("source_colors must have shape (n_source,)")
        (
            self.fit_components,
            self.fit_component_exposures,
            self.fit_component_reference_anchored,
        ) = self._fit_graph_components()
        self.pose_dim = _pose_dimension(config.pose_model)
        self.source_dim = (
            2
            + (2 if config.fit_proper_motion else 0)
            + (1 if config.fit_parallax else 0)
        )
        self.epochs = torch.cat(
            [
                _epoch_jyear(
                    observation.epoch,
                    config,
                    observation.detector_xy,
                    observation.n_matches,
                )
                for observation in graph.observations
            ]
        )
        self.delta_year = self.epochs - config.reference_epoch
        self.parallax_factor = self._optional_rows("parallax_factor")
        self.color, self.dcr_basis, self.lateral_color_basis = self._color_rows()
        self.grid_basis, self.grid_indices, self.grid_weights = self._grid_design()
        self.grid_dimension = (
            0 if self.grid_basis is None else int(self.grid_basis.shape[1])
        )
        self.pose_offset = 0
        self.dcr_offset = self.n_exposure * self.pose_dim
        self.lateral_offset = self.dcr_offset + (
            self.n_exposure if config.fit_dcr else 0
        )
        self.grid_offset = self.lateral_offset + (
            len(graph.detector_ids) if config.fit_lateral_color else 0
        )
        self.global_dimension = self.grid_offset + (
            len(graph.detector_ids) * self.grid_dimension * 2
        )
        self.gauge_exposure_indices = tuple(
            exposures[0]
            for exposures, anchored in zip(
                self.fit_component_exposures,
                self.fit_component_reference_anchored,
                strict=True,
            )
            if not anchored and exposures
        )
        self.free_global = torch.ones(
            self.global_dimension, dtype=self.dtype, device=self.device
        )
        for exposure in self.gauge_exposure_indices:
            start = exposure * self.pose_dim
            self.free_global[start : start + self.pose_dim] = 0.0
        for exposure in range(self.n_exposure):
            if not bool((self.fit_mask & (graph.row_exposure == exposure)).any()):
                start = exposure * self.pose_dim
                self.free_global[start : start + self.pose_dim] = 0.0
                if config.fit_dcr:
                    self.free_global[self.dcr_offset + exposure] = 0.0
        self.kinematic_gauge_indices: tuple[int, ...] = ()

    def _fit_graph_components(
        self,
    ) -> tuple[
        tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...], tuple[bool, ...]
    ]:
        union = _UnionFind(self.n_exposure + self.n_source)
        active_rows = torch.nonzero(self.fit_mask, as_tuple=False).flatten()
        active_exposure = self.graph.row_exposure[active_rows].detach().cpu().tolist()
        active_source = self.graph.row_source[active_rows].detach().cpu().tolist()
        for exposure, source in zip(active_exposure, active_source, strict=True):
            union.union(int(exposure), self.n_exposure + int(source))
        roots = sorted(
            {union.find(int(exposure)) for exposure in active_exposure}
        )
        active_nodes = sorted(
            {int(exposure) for exposure in active_exposure}
            | {self.n_exposure + int(source) for source in active_source}
        )
        groups: dict[int, list[int]] = {}
        for node in active_nodes:
            groups.setdefault(union.find(node), []).append(node)
        components = []
        exposures = []
        anchored = []
        reference_mask = self.graph.reference_mask.detach().cpu().tolist()
        for root in roots:
            nodes = groups[root]
            exp = tuple(node for node in nodes if node < self.n_exposure)
            src = tuple(node - self.n_exposure for node in nodes if node >= self.n_exposure)
            components.append(tuple(nodes))
            exposures.append(exp)
            anchored.append(any(reference_mask[index] for index in src))
        return tuple(components), tuple(exposures), tuple(anchored)

    def _optional_rows(self, name: str) -> Tensor:
        blocks = []
        for observation in self.graph.observations:
            value = getattr(observation, name)
            if value is None:
                value = torch.zeros(
                    (observation.n_matches, 2),
                    dtype=self.dtype,
                    device=self.device,
                )
            blocks.append(value)
        return torch.cat(blocks)

    def _color_rows(self) -> tuple[Tensor, Tensor, Tensor]:
        colors = []
        bases = []
        lateral_bases = []
        for observation in self.graph.observations:
            color = observation.color
            if self.source_colors is not None:
                if observation.source_ids is None:
                    raise ValueError("source_ids are required for source color coupling")
                locations = torch.searchsorted(self.graph.source_ids, observation.source_ids)
                color = self.source_colors[locations]
            basis = observation.dcr_basis
            lateral_basis = observation.lateral_color_basis
            if color is None:
                color = torch.full(
                    (observation.n_matches,),
                    float("nan"),
                    dtype=self.dtype,
                    device=self.device,
                )
            if basis is None:
                basis = torch.zeros(
                    (observation.n_matches, 2),
                    dtype=self.dtype,
                    device=self.device,
                )
            if lateral_basis is None:
                lateral_basis = torch.zeros(
                    (observation.n_matches, 2),
                    dtype=self.dtype,
                    device=self.device,
                )
            colors.append(color)
            bases.append(basis)
            lateral_bases.append(lateral_basis)
        color_rows = torch.cat(colors)
        if (self.config.fit_dcr or self.config.fit_lateral_color) and bool(
            torch.isnan(color_rows).any()
        ):
            if self.config.missing_color == "error":
                raise ValueError("color is required by missing_color='error'")
            color_rows = torch.nan_to_num(
                color_rows, nan=float(self.config.color_pivot)
            )
        return (
            color_rows - self.config.color_pivot,
            torch.cat(bases),
            torch.cat(lateral_bases),
        )

    def _grid_design(self) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        if not self.config.fit_detector_grid:
            return None, None, None
        height, width = self.config.detector_grid_shape
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, dtype=self.dtype, device=self.device),
            torch.linspace(-1.0, 1.0, width, dtype=self.dtype, device=self.device),
            indexing="ij",
        )
        affine = torch.stack((torch.ones_like(x), x, y), dim=-1).reshape(-1, 3)
        full_q, _ = torch.linalg.qr(affine, mode="complete")
        basis = full_q[:, 3:]
        indices = torch.empty((self.n_row, 4), dtype=torch.int64, device=self.device)
        weights = torch.empty((self.n_row, 4), dtype=self.dtype, device=self.device)
        for detector in range(len(self.graph.detector_ids)):
            mask = self.graph.row_detector == detector
            points = self.graph.detector_xy[mask]
            minimum = points.amin(dim=0)
            extent = points.amax(dim=0) - minimum
            extent = torch.where(extent > 0, extent, torch.ones_like(extent))
            uv = (points - minimum) / extent
            gx = uv[:, 0] * (width - 1)
            gy = uv[:, 1] * (height - 1)
            ix = torch.clamp(torch.floor(gx).to(torch.int64), max=width - 2)
            iy = torch.clamp(torch.floor(gy).to(torch.int64), max=height - 2)
            tx = gx - ix
            ty = gy - iy
            indices[mask] = torch.stack(
                (
                    iy * width + ix,
                    iy * width + ix + 1,
                    (iy + 1) * width + ix,
                    (iy + 1) * width + ix + 1,
                ),
                dim=-1,
            )
            weights[mask] = torch.stack(
                ((1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty),
                dim=-1,
            )
        return basis, indices, weights

    def _initial_source(self) -> Tensor:
        source = torch.zeros(
            (self.n_source, self.source_dim), dtype=self.dtype, device=self.device
        )
        counts = torch.zeros(self.n_source, dtype=self.dtype, device=self.device)
        active_rows = self.fit_mask.to(dtype=self.dtype)
        source[:, :2].index_add_(
            0,
            self.graph.row_source,
            self.graph.observed_plane * active_rows.unsqueeze(-1),
        )
        counts.index_add_(
            0,
            self.graph.row_source,
            active_rows,
        )
        source[:, :2] /= counts.clamp_min(1).unsqueeze(-1)
        source[self.graph.reference_mask, :2] = self.graph.reference_plane[
            self.graph.reference_mask
        ]
        column = 2
        if self.config.fit_proper_motion:
            source[self.graph.reference_motion_mask, column : column + 2] = (
                self.graph.reference_motion[self.graph.reference_motion_mask]
            )
            column += 2
        if self.config.fit_parallax:
            source[self.graph.reference_parallax_mask, column] = (
                self.graph.reference_parallax[self.graph.reference_parallax_mask]
            )
        return source

    def _source_design(self, active: Tensor) -> Tensor:
        design = torch.zeros(
            (self.n_row, 2, self.source_dim), dtype=self.dtype, device=self.device
        )
        design[:, 0, 0] = -1.0
        design[:, 1, 1] = -1.0
        column = 2
        if self.config.fit_proper_motion:
            design[:, 0, column] = -self.delta_year
            design[:, 1, column + 1] = -self.delta_year
            column += 2
        if self.config.fit_parallax:
            design[:, :, column] = -self.parallax_factor
        return design * active[self.graph.row_source].unsqueeze(1)

    def _source_activity(self) -> tuple[Tensor, tuple[int, ...]]:
        active = torch.ones(
            (self.n_source, self.source_dim), dtype=self.dtype, device=self.device
        )
        dropped = []
        if self.source_dim == 2:
            for source in range(self.n_source):
                rows = (self.graph.row_source == source) & self.fit_mask
                if not bool(rows.any()):
                    active[source] = 0.0
                    dropped.append(source)
            return active, tuple(dropped)
        for source in range(self.n_source):
            rows = (self.graph.row_source == source) & self.fit_mask
            count = int(rows.sum())
            if count == 0:
                active[source] = 0.0
                dropped.append(source)
                continue
            block = torch.zeros(
                (count, 2, self.source_dim), dtype=self.dtype, device=self.device
            )
            block[:, 0, 0] = 1.0
            block[:, 1, 1] = 1.0
            column = 2
            if self.config.fit_proper_motion:
                block[:, 0, column] = self.delta_year[rows]
                block[:, 1, column + 1] = self.delta_year[rows]
                column += 2
            if self.config.fit_parallax:
                block[:, :, column] = self.parallax_factor[rows]
            singular = torch.linalg.svdvals(block.reshape(2 * count, self.source_dim))
            tolerance = self.config.rank_rtol * singular[0] if singular.numel() else 0.0
            rank = int((singular > tolerance).sum().item())
            if rank >= self.source_dim:
                continue
            if self.config.rank_policy == "error":
                identifier = int(self.graph.source_ids[source].detach().cpu().item())
                raise ValueError(
                    f"source {identifier} motion block is not identifiable"
                )
            active[source, 2:] = 0.0
            dropped.append(source)
        gauge_sources = []
        for nodes in self.fit_components:
            sources = tuple(
                node - self.n_exposure for node in nodes if node >= self.n_exposure
            )
            if not sources:
                continue
            gauge_source = sources[0]
            column = 2
            needs_gauge = False
            if self.config.fit_proper_motion:
                if not any(
                    bool(self.graph.reference_motion_mask[source]) for source in sources
                ):
                    active[gauge_source, column : column + 2] = 0.0
                    needs_gauge = True
                column += 2
            if self.config.fit_parallax and not any(
                bool(self.graph.reference_parallax_mask[source]) for source in sources
            ):
                active[gauge_source, column] = 0.0
                needs_gauge = True
            if needs_gauge:
                gauge_sources.append(gauge_source)
        self.kinematic_gauge_indices = tuple(gauge_sources)
        return active, tuple(dropped)

    def _dcr_activity(self) -> tuple[Tensor, tuple[int, ...]]:
        active = torch.ones(self.n_exposure, dtype=self.dtype, device=self.device)
        dropped = []
        if not self.config.fit_dcr:
            return active, ()
        vectors = self.color.unsqueeze(-1) * self.dcr_basis
        for exposure in range(self.n_exposure):
            rows = (self.graph.row_exposure == exposure) & self.fit_mask
            norm = torch.linalg.vector_norm(vectors[rows])
            if float(norm.detach().cpu().item()) > self.config.rank_rtol:
                continue
            if self.config.rank_policy == "error":
                raise ValueError(
                    f"DCR block for exposure {self.graph.exposure_ids[exposure]!r} "
                    "is not identifiable"
                )
            active[exposure] = 0.0
            dropped.append(exposure)
        return active, tuple(dropped)

    def _lateral_color_activity(self) -> tuple[Tensor, tuple[int, ...]]:
        active = torch.ones(
            len(self.graph.detector_ids), dtype=self.dtype, device=self.device
        )
        dropped = []
        if not self.config.fit_lateral_color:
            return active, ()
        vectors = self.color.unsqueeze(-1) * self.lateral_color_basis
        for detector in range(len(self.graph.detector_ids)):
            rows = (self.graph.row_detector == detector) & self.fit_mask
            norm = torch.linalg.vector_norm(vectors[rows])
            if float(norm.detach().cpu().item()) > self.config.rank_rtol:
                continue
            if self.config.rank_policy == "error":
                raise ValueError(
                    f"lateral-color block for detector "
                    f"{self.graph.detector_ids[detector]!r} is not identifiable"
                )
            active[detector] = 0.0
            dropped.append(detector)
        return active, tuple(dropped)

    def _grid_control(self, global_parameters: Tensor) -> Tensor | None:
        if self.grid_basis is None:
            return None
        coefficients = global_parameters[self.grid_offset :].reshape(
            len(self.graph.detector_ids), self.grid_dimension, 2
        )
        return torch.einsum("kq,dqc->dkc", self.grid_basis, coefficients)

    def _grid_forward(self, global_parameters: Tensor) -> Tensor:
        control = self._grid_control(global_parameters)
        if control is None or self.grid_indices is None or self.grid_weights is None:
            return torch.zeros((self.n_row, 2), dtype=self.dtype, device=self.device)
        selected = control[self.graph.row_detector[:, None], self.grid_indices]
        return torch.sum(self.grid_weights.unsqueeze(-1) * selected, dim=1)

    def _grid_adjoint(self, rows: Tensor) -> Tensor:
        if (
            self.grid_basis is None
            or self.grid_indices is None
            or self.grid_weights is None
        ):
            return torch.empty(0, dtype=self.dtype, device=self.device)
        raw = torch.zeros(
            (len(self.graph.detector_ids), self.grid_basis.shape[0], 2),
            dtype=self.dtype,
            device=self.device,
        )
        for corner in range(4):
            flat = (
                self.graph.row_detector * self.grid_basis.shape[0]
                + self.grid_indices[:, corner]
            )
            raw.reshape(-1, 2).index_add_(
                0, flat, self.grid_weights[:, corner, None] * rows
            )
        coefficients = torch.einsum("kq,dkc->dqc", self.grid_basis, raw)
        return coefficients.reshape(-1)

    def _model(
        self,
        global_parameters: Tensor,
        source_parameters: Tensor,
        dcr_active: Tensor,
        lateral_active: Tensor,
    ) -> tuple[Tensor, Tensor]:
        observed = torch.empty_like(self.graph.observed_plane)
        pose_jacobian = torch.empty(
            (self.n_row, 2, self.pose_dim), dtype=self.dtype, device=self.device
        )
        for exposure in range(self.n_exposure):
            rows = self.graph.row_exposure == exposure
            start = exposure * self.pose_dim
            value, jacobian = _pose_value_and_jacobian(
                self.graph.observed_plane[rows],
                global_parameters[start : start + self.pose_dim],
                self.config.pose_model,
            )
            observed[rows] = value
            pose_jacobian[rows] = jacobian
        observed = observed + self._grid_forward(global_parameters)
        if self.config.fit_dcr:
            coefficient = global_parameters[
                self.dcr_offset : self.dcr_offset + self.n_exposure
            ]
            row_coefficient = (
                coefficient[self.graph.row_exposure]
                * dcr_active[self.graph.row_exposure]
            )
            observed = (
                observed + (row_coefficient * self.color).unsqueeze(-1) * self.dcr_basis
            )
        if self.config.fit_lateral_color:
            coefficient = global_parameters[
                self.lateral_offset : self.grid_offset
            ]
            row_coefficient = (
                coefficient[self.graph.row_detector]
                * lateral_active[self.graph.row_detector]
            )
            observed = (
                observed
                + (row_coefficient * self.color).unsqueeze(-1)
                * self.lateral_color_basis
            )
        source = source_parameters[self.graph.row_source, :2]
        column = 2
        if self.config.fit_proper_motion:
            source = (
                source
                + self.delta_year.unsqueeze(-1)
                * source_parameters[self.graph.row_source, column : column + 2]
            )
            column += 2
        if self.config.fit_parallax:
            source = (
                source
                + source_parameters[self.graph.row_source, column].unsqueeze(-1)
                * self.parallax_factor
            )
        return observed - source, pose_jacobian

    def _global_jvp(
        self,
        vector: Tensor,
        pose_jacobian: Tensor,
        dcr_active: Tensor,
        lateral_active: Tensor,
    ) -> Tensor:
        pose = vector[: self.dcr_offset].reshape(self.n_exposure, self.pose_dim)
        result = torch.einsum(
            "mcp,mp->mc",
            pose_jacobian,
            pose[self.graph.row_exposure],
        )
        if self.config.fit_dcr:
            dcr = vector[self.dcr_offset : self.dcr_offset + self.n_exposure]
            result = (
                result
                + (
                    dcr[self.graph.row_exposure]
                    * dcr_active[self.graph.row_exposure]
                    * self.color
                ).unsqueeze(-1)
                * self.dcr_basis
            )
        if self.config.fit_lateral_color:
            lateral = vector[self.lateral_offset : self.grid_offset]
            result = (
                result
                + (
                    lateral[self.graph.row_detector]
                    * lateral_active[self.graph.row_detector]
                    * self.color
                ).unsqueeze(-1)
                * self.lateral_color_basis
            )
        if self.grid_basis is not None:
            result = result + self._grid_forward(vector)
        return result

    def _global_vjp(
        self,
        rows: Tensor,
        pose_jacobian: Tensor,
        dcr_active: Tensor,
        lateral_active: Tensor,
    ) -> Tensor:
        result = torch.zeros(
            self.global_dimension, dtype=self.dtype, device=self.device
        )
        pose_contribution = torch.einsum("mcp,mc->mp", pose_jacobian, rows)
        result[: self.dcr_offset].reshape(self.n_exposure, self.pose_dim).index_add_(
            0, self.graph.row_exposure, pose_contribution
        )
        if self.config.fit_dcr:
            dcr_contribution = torch.sum(self.dcr_basis * rows, dim=-1) * self.color
            dcr_contribution = dcr_contribution * dcr_active[self.graph.row_exposure]
            result[self.dcr_offset : self.dcr_offset + self.n_exposure].index_add_(
                0, self.graph.row_exposure, dcr_contribution
            )
        if self.config.fit_lateral_color:
            contribution = (
                torch.sum(self.lateral_color_basis * rows, dim=-1) * self.color
            )
            contribution = contribution * lateral_active[self.graph.row_detector]
            result[self.lateral_offset : self.grid_offset].index_add_(
                0, self.graph.row_detector, contribution
            )
        if self.grid_basis is not None:
            result[self.grid_offset :] = self._grid_adjoint(rows)
        return result

    def _source_jvp(self, source_vector: Tensor, design: Tensor) -> Tensor:
        return torch.einsum("mcp,mp->mc", design, source_vector[self.graph.row_source])

    def _source_vjp(self, rows: Tensor, design: Tensor) -> Tensor:
        result = torch.zeros(
            (self.n_source, self.source_dim), dtype=self.dtype, device=self.device
        )
        result.index_add_(
            0, self.graph.row_source, torch.einsum("mcp,mc->mp", design, rows)
        )
        return result

    @staticmethod
    def _apply_precision(precision: Tensor, rows: Tensor, weights: Tensor) -> Tensor:
        return weights.unsqueeze(-1) * torch.einsum("mcd,md->mc", precision, rows)

    def _normal_blocks(
        self,
        residual: Tensor,
        pose_jacobian: Tensor,
        design: Tensor,
        source_parameters: Tensor,
        robust: Tensor,
        dcr_active: Tensor,
        lateral_active: Tensor,
        source_active: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        precision = torch.linalg.inv(self.graph.centroid_covariance)
        weighted_residual = self._apply_precision(precision, residual, robust)
        global_rhs = -self._global_vjp(
            weighted_residual, pose_jacobian, dcr_active, lateral_active
        )
        source_rhs = -self._source_vjp(weighted_residual, design)
        weighted_design = robust[:, None, None] * torch.einsum(
            "mcd,mdp->mcp", precision, design
        )
        source_normal_rows = torch.einsum("mcp,mcq->mpq", design, weighted_design)
        source_normal = torch.zeros(
            (self.n_source, self.source_dim, self.source_dim),
            dtype=self.dtype,
            device=self.device,
        )
        source_normal.index_add_(0, self.graph.row_source, source_normal_rows)
        for source in range(self.n_source):
            if bool(self.graph.reference_mask[source]):
                prior_precision = self.graph.reference_precision[source]
                prior_residual = (
                    source_parameters[source, :2] - self.graph.reference_plane[source]
                )
                source_normal[source, :2, :2] += prior_precision
                source_rhs[source, :2] -= prior_precision @ prior_residual
            column = 2
            if self.config.fit_proper_motion:
                if bool(self.graph.reference_motion_mask[source]):
                    prior_precision = self.graph.reference_motion_precision[source]
                    prior_residual = (
                        source_parameters[source, column : column + 2]
                        - self.graph.reference_motion[source]
                    )
                    source_normal[source, column : column + 2, column : column + 2] += (
                        prior_precision
                    )
                    source_rhs[source, column : column + 2] -= (
                        prior_precision @ prior_residual
                    )
                column += 2
            if self.config.fit_parallax and bool(
                self.graph.reference_parallax_mask[source]
            ):
                prior_precision = self.graph.reference_parallax_precision[source]
                prior_residual = (
                    source_parameters[source, column]
                    - self.graph.reference_parallax[source]
                )
                source_normal[source, column, column] += prior_precision
                source_rhs[source, column] -= prior_precision * prior_residual
            inactive = source_active[source] == 0
            if bool(inactive.any()):
                source_normal[source, inactive, :] = 0.0
                source_normal[source, :, inactive] = 0.0
                source_normal[source, inactive, inactive] = 1.0
                source_rhs[source, inactive] = 0.0
        inverse = torch.linalg.inv(source_normal)
        return global_rhs, source_rhs, source_normal, inverse

    def _huber_weights(self, residual: Tensor) -> Tensor:
        precision = torch.linalg.inv(self.graph.centroid_covariance)
        squared = torch.einsum("mc,mcd,md->m", residual, precision, residual)
        norm = torch.sqrt(torch.clamp_min(squared, 0.0))
        delta = torch.as_tensor(
            self.config.huber_delta, dtype=self.dtype, device=self.device
        )
        return torch.where(
            norm <= delta,
            torch.ones_like(norm),
            delta / norm.clamp_min(torch.finfo(self.dtype).tiny),
        )

    def _objective(
        self,
        residual: Tensor,
        source_parameters: Tensor | None = None,
        global_parameters: Tensor | None = None,
    ) -> Tensor:
        precision = torch.linalg.inv(self.graph.centroid_covariance)
        norm = torch.sqrt(
            torch.clamp_min(
                torch.einsum("mc,mcd,md->m", residual, precision, residual),
                0.0,
            )
        )
        delta = self.config.huber_delta
        objective = torch.sum(
            self.fit_mask.to(dtype=self.dtype)
            * torch.where(
                norm <= delta, 0.5 * norm.square(), delta * (norm - 0.5 * delta)
            )
        )
        if source_parameters is not None:
            position_residual = source_parameters[:, :2] - self.graph.reference_plane
            objective = objective + 0.5 * torch.sum(
                torch.einsum(
                    "sc,scd,sd->s",
                    position_residual,
                    self.graph.reference_precision,
                    position_residual,
                )
            )
            column = 2
            if self.config.fit_proper_motion:
                motion_residual = (
                    source_parameters[:, column : column + 2]
                    - self.graph.reference_motion
                )
                objective = objective + 0.5 * torch.sum(
                    torch.einsum(
                        "sc,scd,sd->s",
                        motion_residual,
                        self.graph.reference_motion_precision,
                        motion_residual,
                    )
                )
                column += 2
            if self.config.fit_parallax:
                parallax_residual = (
                    source_parameters[:, column] - self.graph.reference_parallax
                )
                objective = objective + 0.5 * torch.sum(
                    self.graph.reference_parallax_precision * parallax_residual.square()
                )
        if global_parameters is not None and self.grid_basis is not None:
            objective = (
                objective
                + 0.5
                * self.config.detector_grid_regularization
                * torch.sum(global_parameters[self.grid_offset :].square())
            )
        return objective

    def solve(self) -> AstrometrySolution:
        from ..map.solvers import conjugate_gradient

        source_active, dropped_source = self._source_activity()
        dcr_active, dropped_dcr = self._dcr_activity()
        lateral_active, dropped_lateral = self._lateral_color_activity()
        global_parameters = torch.zeros(
            self.global_dimension, dtype=self.dtype, device=self.device
        )
        source_parameters = self._initial_source()
        design = self._source_design(source_active)
        objectives = []
        steps = []
        cg_iterations = []
        converged = False
        last_reduced_action: Any = None
        last_pose_jacobian: Tensor | None = None
        last_source_inverse: Tensor | None = None
        robust = torch.ones(self.n_row, dtype=self.dtype, device=self.device)
        for _iteration in range(self.config.max_iterations):
            residual, pose_jacobian = self._model(
                global_parameters, source_parameters, dcr_active, lateral_active
            )
            robust = self._huber_weights(residual) * self.fit_mask.to(dtype=self.dtype)
            objective = self._objective(residual, source_parameters, global_parameters)
            objectives.append(float(objective.detach().cpu().item()))
            global_rhs, source_rhs, _source_normal, source_inverse = (
                self._normal_blocks(
                    residual,
                    pose_jacobian,
                    design,
                    source_parameters,
                    robust,
                    dcr_active,
                    lateral_active,
                    source_active,
                )
            )
            precision = torch.linalg.inv(self.graph.centroid_covariance)

            def cross_to_source(vector: Tensor) -> Tensor:
                rows = self._global_jvp(
                    vector, pose_jacobian, dcr_active, lateral_active
                )
                return self._source_vjp(
                    self._apply_precision(precision, rows, robust), design
                )

            def cross_to_global(source_vector: Tensor) -> Tensor:
                rows = self._source_jvp(source_vector, design)
                return self._global_vjp(
                    self._apply_precision(precision, rows, robust),
                    pose_jacobian,
                    dcr_active,
                    lateral_active,
                )

            def reduced_action(vector: Tensor) -> Tensor:
                vector = vector * self.free_global
                rows = self._global_jvp(
                    vector, pose_jacobian, dcr_active, lateral_active
                )
                direct = self._global_vjp(
                    self._apply_precision(precision, rows, robust),
                    pose_jacobian,
                    dcr_active,
                    lateral_active,
                )
                source_cross = cross_to_source(vector)
                eliminated = torch.einsum("spq,sq->sp", source_inverse, source_cross)
                result = direct - cross_to_global(eliminated)
                if (
                    self.grid_basis is not None
                    and self.config.detector_grid_regularization
                ):
                    result[self.grid_offset :] += (
                        self.config.detector_grid_regularization
                        * vector[self.grid_offset :]
                    )
                result = result + self.config.damping * vector
                return result * self.free_global

            reduced_rhs = global_rhs - cross_to_global(
                torch.einsum("spq,sq->sp", source_inverse, source_rhs)
            )
            reduced_rhs = reduced_rhs * self.free_global
            result = conjugate_gradient(
                reduced_action,
                reduced_rhs,
                operator_is_hpd=True,
                max_iterations=self.config.cg_max_iterations,
                rtol=self.config.cg_rtol,
                atol=self.config.cg_atol,
                projection=lambda value: value * self.free_global,
            )
            if result.diagnostics is None:
                raise RuntimeError("matrix-free solver returned no diagnostics")
            cg_iterations.append(result.diagnostics.iterations)
            if result.diagnostics.breakdown is not None:
                raise RuntimeError(
                    f"reduced astrometry solve failed: {result.diagnostics.breakdown.message}"
                )
            global_step = result.solution
            source_step = torch.einsum(
                "spq,sq->sp",
                source_inverse,
                source_rhs - cross_to_source(global_step),
            )
            scale = 1.0
            current_objective = objective
            while scale >= 1.0 / 128.0:
                trial_global = global_parameters + scale * global_step
                trial_source = source_parameters + scale * source_step
                trial_residual, _ = self._model(
                    trial_global, trial_source, dcr_active, lateral_active
                )
                if (
                    self._objective(trial_residual, trial_source, trial_global)
                    <= current_objective
                ):
                    break
                scale *= 0.5
            global_parameters = global_parameters + scale * global_step
            source_parameters = source_parameters + scale * source_step
            step_norm = torch.sqrt(
                torch.sum((scale * global_step).square())
                + torch.sum((scale * source_step).square())
            )
            parameter_norm = torch.sqrt(
                torch.sum(global_parameters.square())
                + torch.sum(source_parameters.square())
            )
            relative_step = float(
                (step_norm / torch.clamp_min(parameter_norm, 1.0)).detach().cpu().item()
            )
            steps.append(relative_step)
            last_reduced_action = reduced_action
            last_pose_jacobian = pose_jacobian
            last_source_inverse = source_inverse
            if relative_step <= self.config.convergence_tolerance:
                converged = True
                break
        residual, _ = self._model(
            global_parameters, source_parameters, dcr_active, lateral_active
        )
        robust = self._huber_weights(residual) * self.fit_mask.to(dtype=self.dtype)
        source_world = self._source_world(source_parameters[:, :2])
        rank, condition, covariance = self._diagnostics_covariance(
            last_reduced_action, global_parameters
        )
        pose = global_parameters[: self.dcr_offset].reshape(
            self.n_exposure, self.pose_dim
        )
        dcr = None
        if self.config.fit_dcr:
            dcr = global_parameters[self.dcr_offset : self.dcr_offset + self.n_exposure]
        lateral = None
        if self.config.fit_lateral_color:
            lateral = global_parameters[self.lateral_offset : self.grid_offset]
        grid = self._grid_control(global_parameters)
        if grid is not None:
            grid = grid.reshape(
                len(self.graph.detector_ids),
                self.config.detector_grid_shape[0],
                self.config.detector_grid_shape[1],
                2,
            )
        proper_motion = None
        parallax = None
        column = 2
        if self.config.fit_proper_motion:
            proper_motion = source_parameters[:, column : column + 2] * 1000.0
            column += 2
        if self.config.fit_parallax:
            parallax = source_parameters[:, column] * 1000.0
        diagnostics = AstrometryDiagnostics(
            converged=converged,
            iterations=len(steps),
            objective_history=tuple(objectives),
            step_history=tuple(steps),
            cg_iterations=tuple(cg_iterations),
            reduced_rank=rank,
            reduced_dimension=int(self.free_global.sum().item()),
            condition_number=condition,
            connected_components=tuple(
                tuple(self.graph.exposure_ids[index] for index in exposures)
                for exposures in self.fit_component_exposures
            ),
            gauge_exposures=tuple(
                self.graph.exposure_ids[index] for index in self.gauge_exposure_indices
            ),
            kinematic_gauge_sources=tuple(
                int(self.graph.source_ids[index].detach().cpu().item())
                for index in self.kinematic_gauge_indices
            ),
            reference_anchored_components=self.fit_component_reference_anchored,
            dropped_source_blocks=tuple(
                int(self.graph.source_ids[index].detach().cpu().item())
                for index in dropped_source
            ),
            dropped_dcr_exposures=tuple(
                self.graph.exposure_ids[index] for index in dropped_dcr
            ),
            dropped_lateral_color_detectors=tuple(
                self.graph.detector_ids[index] for index in dropped_lateral
            ),
        )
        _ = last_pose_jacobian, last_source_inverse
        return AstrometrySolution(
            exposure_ids=self.graph.exposure_ids,
            source_ids=self.graph.source_ids,
            pose_model=self.config.pose_model,
            pose_parameters=pose,
            source_positions=source_world,
            source_tangent_positions=source_parameters[:, :2],
            proper_motion_mas_per_year=proper_motion,
            parallax_mas=parallax,
            dcr_coefficients=dcr,
            lateral_color_coefficients=lateral,
            detector_ids=self.graph.detector_ids,
            detector_residual_control=grid,
            residuals_arcsec=residual,
            robust_weights=robust,
            exposure_covariance=covariance,
            diagnostics=diagnostics,
            fit_mask=self.fit_mask,
        )

    def _source_world(self, tangent_arcsec: Tensor) -> Tensor:
        ra, dec = self.graph.tangent_frame.plane_to_world(
            tangent_arcsec[:, 0] / 3600.0,
            tangent_arcsec[:, 1] / 3600.0,
        )
        return torch.stack((ra, dec), dim=-1)

    def _diagnostics_covariance(
        self, action: Any, parameters: Tensor
    ) -> tuple[int | None, float | None, dict[Hashable, Tensor]]:
        covariance: dict[Hashable, Tensor] = {}
        free_indices = torch.nonzero(self.free_global > 0, as_tuple=False).squeeze(1)
        n_free = int(free_indices.numel())
        if action is None or n_free == 0:
            zero = torch.zeros(
                (self.pose_dim, self.pose_dim), dtype=self.dtype, device=self.device
            )
            return 0, None, {key: zero.clone() for key in self.graph.exposure_ids}
        if n_free > self.config.covariance_max_parameters:
            return None, None, covariance
        columns = []
        for index in free_indices:
            basis = torch.zeros_like(parameters)
            basis[index] = 1.0
            columns.append(action(basis)[free_indices])
        matrix = torch.stack(columns, dim=-1)
        matrix = 0.5 * (matrix + matrix.mT)
        rank_matrix = matrix - self.config.damping * torch.eye(
            n_free, dtype=self.dtype, device=self.device
        )
        analysis = rank_matrix.cpu() if self.device.type == "mps" else rank_matrix
        singular = torch.linalg.eigvalsh(analysis).to(self.device)
        tolerance = self.config.rank_rtol * singular.abs().amax()
        positive = singular > tolerance
        rank = int(positive.sum().item())
        condition = None
        if bool(positive.any()):
            condition = float(
                (singular[positive].amax() / singular[positive].amin())
                .detach()
                .cpu()
                .item()
            )
        inverse_matrix = matrix.cpu() if self.device.type == "mps" else matrix
        inverse = torch.linalg.pinv(
            inverse_matrix, rtol=self.config.rank_rtol
        ).to(self.device)
        full = torch.zeros(
            (self.global_dimension, self.global_dimension),
            dtype=self.dtype,
            device=self.device,
        )
        full[free_indices[:, None], free_indices[None, :]] = inverse
        for exposure, identifier in enumerate(self.graph.exposure_ids):
            start = exposure * self.pose_dim
            covariance[identifier] = full[
                start : start + self.pose_dim, start : start + self.pose_dim
            ]
        return rank, condition, covariance


def solve_astrometry(
    graph: AstrometricMatchGraph, config: AstrometryConfig | None = None
) -> AstrometrySolution:
    """Solve an astrometric match graph for exposure and source parameters."""
    return graph.solve(config)


__all__ = [
    "ApproximateMatchResult",
    "AstrometricMatchGraph",
    "AstrometricObservation",
    "AstrometryConfig",
    "AstrometryDiagnostics",
    "AstrometrySolution",
    "match_approximate_wcs",
    "solve_astrometry",
    "tangent_plane_block_residuals",
]
