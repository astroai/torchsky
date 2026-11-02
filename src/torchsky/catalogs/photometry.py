"""Block-sparse fixed-template forced photometry for known source catalogs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Literal

import torch
from torch import Tensor

from ..map.field.observation_ops import observation_support_mask
from ..map.field.types import FieldObservation
from ..map.operators.psf import psf_support_radius
from ..scene.sources import SourceCatalogScene, SourceStamp
from ..sphere.moc import MOC
from ..wcs import WCS
from .catalog import Catalog
from .index import HealpixIndex


PhotometrySolver = Literal["signed", "nnls"]
BackgroundModel = Literal["none", "constant", "planar"]


@dataclass(frozen=True)
class SourceGrouping:
    source_exposures: tuple[Tensor, ...]
    exposure_sources: tuple[Tensor, ...]
    blend_groups: tuple[Tensor, ...]
    group_exposures: tuple[Tensor, ...]
    observation_footprints: tuple[MOC, ...]


@dataclass(frozen=True)
class BlendFitDiagnostics:
    group_index: int
    source_indices: Tensor
    exposure_indices: Tensor
    parameter_indices: Tensor
    active_parameter_indices: Tensor
    data_rank: int
    active_rank: int
    parameter_count: int
    active_parameter_count: int
    condition_number: float
    rank_cutoff: Tensor
    information_eigenvalues: Tensor
    gauge_parameter_indices: Tensor
    solver: PhotometrySolver
    converged: bool
    iterations: int

    @property
    def rank_deficient(self) -> bool:
        return self.data_rank < self.parameter_count

    @property
    def gauge_detected(self) -> bool:
        return bool(self.gauge_parameter_indices.numel())


@dataclass(frozen=True)
class BlockSparseCovariance:
    parameter_count: int
    parameter_indices: tuple[Tensor, ...]
    blocks: tuple[Tensor, ...]
    constrained: bool

    def __post_init__(self) -> None:
        if self.parameter_count < 0:
            raise ValueError("parameter_count must be non-negative")
        if len(self.parameter_indices) != len(self.blocks):
            raise ValueError("covariance indices and blocks must align")
        for indices, block in zip(self.parameter_indices, self.blocks, strict=True):
            if indices.dtype != torch.int64 or indices.ndim != 1:
                raise TypeError(
                    "covariance parameter indices must be one-dimensional int64 tensors"
                )
            if block.shape != (indices.numel(), indices.numel()):
                raise ValueError(
                    "each covariance block must match its parameter indices"
                )

    @property
    def device(self) -> torch.device:
        if self.blocks:
            return self.blocks[0].device
        if self.parameter_indices:
            return self.parameter_indices[0].device
        return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        return self.blocks[0].dtype if self.blocks else torch.get_default_dtype()

    def diagonal(self) -> Tensor:
        diagonal = torch.full(
            (self.parameter_count,),
            float("nan"),
            dtype=self.dtype,
            device=self.device,
        )
        for indices, block in zip(self.parameter_indices, self.blocks, strict=True):
            diagonal[indices] = torch.diagonal(block)
        return diagonal

    def apply_covariance(self, value: Tensor) -> Tensor:
        if (
            value.shape != (self.parameter_count,)
            or value.dtype != self.dtype
            or value.device != self.device
        ):
            raise ValueError(
                "covariance input must match parameter shape, dtype, and device"
            )
        output = torch.zeros_like(value)
        covered = torch.zeros(
            self.parameter_count, dtype=torch.bool, device=self.device
        )
        for indices, block in zip(self.parameter_indices, self.blocks, strict=True):
            output[indices] = block @ value[indices]
            covered[indices] = True
        if bool((value[~covered] != 0).any()):
            raise ValueError(
                "covariance is undefined for inactive or unidentifiable parameters"
            )
        return output

    def query_variance(self, query: Tensor) -> Tensor:
        return torch.sum(query * self.apply_covariance(query))

    def to_dense(self) -> Tensor:
        dense = torch.zeros(
            (self.parameter_count, self.parameter_count),
            dtype=self.dtype,
            device=self.device,
        )
        covered = torch.zeros(
            self.parameter_count, dtype=torch.bool, device=self.device
        )
        for indices, block in zip(self.parameter_indices, self.blocks, strict=True):
            dense[indices[:, None], indices[None, :]] = block
            covered[indices] = True
        missing = torch.nonzero(~covered, as_tuple=False).flatten()
        dense[missing, missing] = float("nan")
        return dense


@dataclass(frozen=True)
class ForwardPhotometryResult:
    catalog: Catalog
    scene: SourceCatalogScene
    coefficients: Tensor
    covariance: BlockSparseCovariance
    predictions: tuple[Tensor, ...]
    residuals: tuple[Tensor, ...]
    grouping: SourceGrouping
    diagnostics: tuple[BlendFitDiagnostics, ...]
    background_coefficients: tuple[Tensor, ...]
    provenance: dict[str, object]

    @property
    def fluxes(self) -> Tensor:
        return self.catalog.columns["flux"]

    @property
    def flux(self) -> Tensor:
        return self.fluxes

    @property
    def flux_variance(self) -> Tensor:
        return self.catalog.columns["flux_variance"]


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, left: int, right: int) -> None:
        lhs = self.find(left)
        rhs = self.find(right)
        if lhs == rhs:
            return
        if self.rank[lhs] < self.rank[rhs]:
            lhs, rhs = rhs, lhs
        self.parent[rhs] = lhs
        if self.rank[lhs] == self.rank[rhs]:
            self.rank[lhs] += 1


def _observation_footprint(
    observation: FieldObservation,
    *,
    device: torch.device,
    order: int,
) -> MOC:
    wcs = WCS(observation.header).to(device)
    mask = observation_support_mask(observation).to(device=device)
    return MOC.from_wcs(
        wcs,
        shape=(int(observation.image.shape[0]), int(observation.image.shape[1])),
        max_order=order,
        mask=mask,
    )


def _observation_psf_radius_arcsec(
    observation: FieldObservation,
    *,
    device: torch.device,
) -> float:
    halo_y, halo_x = psf_support_radius(observation.psf_spec)
    halo = max(halo_y, halo_x, 2)
    height, width = (int(value) for value in observation.image.shape)
    wcs = WCS(observation.header).to(device)
    dtype = torch.float32 if device.type == "mps" else torch.float64
    x = torch.tensor(0.5 * (width - 1), dtype=dtype, device=device)
    y = torch.tensor(0.5 * (height - 1), dtype=dtype, device=device)
    lon0, lat0 = wcs.pixel_to_world(x, y)
    lon1, lat1 = wcs.pixel_to_world(x + 1.0, y)
    cos_lat = torch.cos(torch.deg2rad(lat0)).abs().clamp_min(1.0e-8)
    scale = torch.sqrt(((lon1 - lon0) * cos_lat).square() + (lat1 - lat0).square())
    return float(scale.detach().item()) * 3600.0 * halo


def _expanded_source_footprints(
    scene: SourceCatalogScene,
    observations: Sequence[FieldObservation],
    *,
    order: int,
) -> tuple[MOC, ...]:
    psf_radius = max(
        (
            _observation_psf_radius_arcsec(item, device=scene.device)
            for item in observations
        ),
        default=0.0,
    )
    footprints = []
    for index in range(scene.component_count):
        radius = float(scene.support_radius_arcsec[index].detach().item()) + psf_radius
        if radius <= 0.0:
            footprints.append(scene.support_footprint(index))
        else:
            footprints.append(
                MOC.from_circle(
                    float(scene.lon[index].detach().item()),
                    float(scene.lat[index].detach().item()),
                    radius / 3600.0,
                    max_order=order,
                )
            )
    return tuple(footprints)


def build_source_groups(
    scene: SourceCatalogScene,
    observations: Sequence[FieldObservation],
    *,
    index_nside: int | None = None,
    footprint_order: int | None = None,
) -> SourceGrouping:
    """Build exact source-exposure and source-overlap groups with HEALPix/MOCs."""
    if not isinstance(scene, SourceCatalogScene):
        raise TypeError("scene must be a SourceCatalogScene")
    if not observations:
        raise ValueError("at least one observation is required")
    if any(not isinstance(item, FieldObservation) for item in observations):
        raise TypeError("observations must contain FieldObservation objects")
    if any(item.image.device != scene.device for item in observations):
        raise ValueError("scene and observations must share a device")
    order = scene.footprint_order if footprint_order is None else int(footprint_order)
    if scene.device.type == "mps":
        order = min(order, 13)
    index = HealpixIndex.build(scene.lon, scene.lat, nside=index_nside, nest=True)
    observation_footprints = tuple(
        _observation_footprint(item, device=scene.device, order=order)
        for item in observations
    )
    source_footprints = _expanded_source_footprints(scene, observations, order=order)
    psf_radius = max(
        (
            _observation_psf_radius_arcsec(item, device=scene.device)
            for item in observations
        ),
        default=0.0,
    )
    effective_radii = scene.support_radius_arcsec + psf_radius
    maximum_radius = (
        float(effective_radii.max().detach().item()) if effective_radii.numel() else 0.0
    )
    exposure_sources: list[Tensor] = []
    source_exposure_lists: list[list[int]] = [[] for _ in range(scene.component_count)]
    for exposure_index, footprint in enumerate(observation_footprints):
        centered = index.lookup_moc(footprint)
        selected = set(int(value) for value in centered.tolist())
        for source_index, source_footprint in enumerate(source_footprints):
            if source_index not in selected and source_footprint.overlaps(footprint):
                selected.add(source_index)
        indices = torch.tensor(sorted(selected), dtype=torch.int64, device=scene.device)
        exposure_sources.append(indices)
        for source_index in selected:
            source_exposure_lists[source_index].append(exposure_index)
    source_exposures = tuple(
        torch.tensor(values, dtype=torch.int64, device=scene.device)
        for values in source_exposure_lists
    )

    disjoint = _DisjointSet(scene.component_count)
    exposure_sets = [
        set(int(value) for value in values.tolist()) for values in exposure_sources
    ]
    for source_index, source_footprint in enumerate(source_footprints):
        candidates = set(
            int(value) for value in index.lookup_moc(source_footprint).tolist()
        )
        query_radius = (
            float(effective_radii[source_index].detach().item()) + maximum_radius
        ) / 3600.0
        if query_radius > 0.0:
            candidates.update(
                int(value)
                for value in index.query_disc(
                    scene.lon[source_index],
                    scene.lat[source_index],
                    query_radius,
                ).tolist()
            )
        for other in candidates:
            other_index = int(other)
            if other_index <= source_index:
                continue
            if not source_footprint.overlaps(source_footprints[other_index]):
                continue
            if any(
                source_index in exposure_set and other_index in exposure_set
                for exposure_set in exposure_sets
            ):
                disjoint.union(source_index, other_index)
    for left in range(scene.component_count):
        left_start, left_stop = (
            int(value) for value in scene.parameter_slices[left].tolist()
        )
        for right in range(left + 1, scene.component_count):
            right_start, right_stop = (
                int(value) for value in scene.parameter_slices[right].tolist()
            )
            if max(left_start, right_start) < min(left_stop, right_stop):
                disjoint.union(left, right)
    members: dict[int, list[int]] = {}
    for source_index in range(scene.component_count):
        members.setdefault(disjoint.find(source_index), []).append(source_index)
    blend_groups = tuple(
        torch.tensor(values, dtype=torch.int64, device=scene.device)
        for _, values in sorted(members.items(), key=lambda pair: min(pair[1]))
    )
    group_exposures = []
    for group in blend_groups:
        values = sorted(
            {
                exposure
                for source_index in group.tolist()
                for exposure in source_exposure_lists[int(source_index)]
            }
        )
        group_exposures.append(
            torch.tensor(values, dtype=torch.int64, device=scene.device)
        )
    return SourceGrouping(
        source_exposures=source_exposures,
        exposure_sources=tuple(exposure_sources),
        blend_groups=blend_groups,
        group_exposures=tuple(group_exposures),
        observation_footprints=observation_footprints,
    )


def _union_bounds(stamps: Sequence[SourceStamp]) -> tuple[int, int, int, int]:
    return (
        min(int(item.y_slice.start) for item in stamps),
        max(int(item.y_slice.stop) for item in stamps),
        min(int(item.x_slice.start) for item in stamps),
        max(int(item.x_slice.stop) for item in stamps),
    )


def _background_columns(
    model: BackgroundModel,
    shape: tuple[int, int],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    height, width = shape
    if model == "none":
        return torch.empty((0, height, width), dtype=dtype, device=device)
    constant = torch.ones((height, width), dtype=dtype, device=device)
    if model == "constant":
        return constant[None]
    if model != "planar":
        raise ValueError("background must be 'none', 'constant', or 'planar'")
    y = torch.linspace(-1.0, 1.0, height, dtype=dtype, device=device)
    x = torch.linspace(-1.0, 1.0, width, dtype=dtype, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((constant, xx, yy))


def _rank_analysis(
    information: Tensor,
    *,
    rank_rtol: float | None,
    rank_atol: float,
) -> tuple[Tensor, Tensor, int]:
    symmetric = 0.5 * (information + information.T)
    analysis = symmetric.to("cpu") if symmetric.device.type == "mps" else symmetric
    eigenvalues = torch.linalg.eigvalsh(analysis).to(device=information.device)
    if eigenvalues.numel() == 0:
        return eigenvalues, information.new_zeros(()), 0
    relative = (
        information.shape[0] * torch.finfo(information.dtype).eps
        if rank_rtol is None
        else float(rank_rtol)
    )
    cutoff = eigenvalues.abs().max() * relative + float(rank_atol)
    rank = int((eigenvalues > cutoff).sum().item())
    return eigenvalues, cutoff, rank


def _stable_solve(matrix: Tensor, rhs: Tensor, cutoff: Tensor) -> tuple[Tensor, int]:
    if matrix.numel() == 0:
        return rhs.new_empty((0,)), 0
    symmetric = 0.5 * (matrix + matrix.T)
    analysis = symmetric.to("cpu") if symmetric.device.type == "mps" else symmetric
    values, vectors = torch.linalg.eigh(analysis)
    values = values.to(device=matrix.device)
    vectors = vectors.to(device=matrix.device)
    selected = values > cutoff.to(device=matrix.device)
    rank = int(selected.sum().item())
    if rank == 0:
        return torch.zeros_like(rhs), 0
    projected = vectors[:, selected].T @ rhs
    solution = vectors[:, selected] @ (projected / values[selected])
    return solution, rank


def _signed_solve(
    information: Tensor,
    rhs: Tensor,
    cutoff: Tensor,
) -> tuple[Tensor, Tensor, int, int]:
    informative = torch.diagonal(information).abs() > cutoff
    active = torch.nonzero(informative, as_tuple=False).flatten()
    solution = torch.full_like(rhs, float("nan"))
    if active.numel() == 0:
        return solution, active, 0, 0
    block = information[active[:, None], active[None, :]]
    values, local_cutoff, rank = _rank_analysis(block, rank_rtol=None, rank_atol=0.0)
    del values
    solved, solved_rank = _stable_solve(block, rhs[active], local_cutoff)
    solution[active] = solved
    return solution, active, rank, solved_rank


def _nnls_solve(
    information: Tensor,
    rhs: Tensor,
    flux_count: int,
    cutoff: Tensor,
    *,
    max_iterations: int,
) -> tuple[Tensor, Tensor, int, bool, int]:
    parameter_count = rhs.numel()
    solution = torch.zeros_like(rhs)
    informative = torch.diagonal(information).abs() > cutoff
    passive = torch.zeros(parameter_count, dtype=torch.bool, device=rhs.device)
    passive[flux_count:] = informative[flux_count:]
    iterations = 0
    converged = True
    if bool(passive.any()):
        indices = torch.nonzero(passive, as_tuple=False).flatten()
        block = information[indices[:, None], indices[None, :]]
        _, local_cutoff, _ = _rank_analysis(block, rank_rtol=None, rank_atol=0.0)
        solution[indices], _ = _stable_solve(block, rhs[indices], local_cutoff)
    while iterations < max_iterations:
        gradient = rhs - information @ solution
        eligible = (~passive) & informative
        eligible[flux_count:] = False
        if not bool(eligible.any()):
            break
        masked = torch.where(eligible, gradient, torch.full_like(gradient, -torch.inf))
        candidate = int(torch.argmax(masked).item())
        tolerance = float(cutoff.detach().item())
        if float(masked[candidate].detach().item()) <= tolerance:
            break
        passive[candidate] = True
        while True:
            indices = torch.nonzero(passive, as_tuple=False).flatten()
            block = information[indices[:, None], indices[None, :]]
            _, local_cutoff, _ = _rank_analysis(block, rank_rtol=None, rank_atol=0.0)
            trial_values, _ = _stable_solve(block, rhs[indices], local_cutoff)
            trial = torch.zeros_like(solution)
            trial[indices] = trial_values
            negative = passive.clone()
            negative[flux_count:] = False
            negative &= trial <= 0
            if not bool(negative.any()):
                solution = trial
                break
            negative_indices = torch.nonzero(negative, as_tuple=False).flatten()
            denominator = solution[negative_indices] - trial[negative_indices]
            valid = denominator > 0
            if not bool(valid.any()):
                passive[negative_indices] = False
                solution[negative_indices] = 0
                continue
            alpha = torch.min(solution[negative_indices][valid] / denominator[valid])
            solution = solution + alpha * (trial - solution)
            dropped = passive & (solution <= torch.finfo(solution.dtype).eps)
            dropped[flux_count:] = False
            passive[dropped] = False
            solution[dropped] = 0
        iterations += 1
    else:
        converged = False
    active = torch.nonzero(passive, as_tuple=False).flatten()
    rank = 0
    if active.numel():
        block = information[active[:, None], active[None, :]]
        _, _, rank = _rank_analysis(block, rank_rtol=None, rank_atol=0.0)
    return solution, active, rank, converged, iterations


def _covariance_block(
    information: Tensor,
    active: Tensor,
    flux_count: int,
) -> tuple[Tensor, Tensor]:
    active_flux = active[active < flux_count]
    if active.numel() == 0 or active_flux.numel() == 0:
        return active_flux, information.new_empty((0, 0))
    block = information[active[:, None], active[None, :]]
    eigenvalues, cutoff, rank = _rank_analysis(block, rank_rtol=None, rank_atol=0.0)
    del eigenvalues
    if rank != active.numel():
        return active_flux, torch.full(
            (active_flux.numel(), active_flux.numel()),
            float("nan"),
            dtype=information.dtype,
            device=information.device,
        )
    identity = torch.eye(
        active.numel(), dtype=information.dtype, device=information.device
    )
    inverse_columns = []
    for column in identity.T:
        solved, _ = _stable_solve(block, column, cutoff)
        inverse_columns.append(solved)
    inverse = torch.stack(inverse_columns, dim=1)
    inverse = 0.5 * (inverse + inverse.T)
    positions = torch.searchsorted(active, active_flux)
    return active_flux, inverse[positions[:, None], positions[None, :]]


def fit_forced_photometry(
    observations: Sequence[FieldObservation],
    scene: SourceCatalogScene,
    *,
    solver: PhotometrySolver = "signed",
    background: BackgroundModel = "constant",
    index_nside: int | None = None,
    footprint_order: int | None = None,
    chunk_size: int | None = None,
    rank_rtol: float | None = None,
    rank_atol: float = 0.0,
    nnls_max_iterations: int = 256,
) -> ForwardPhotometryResult:
    """Fit known fixed source templates on native exposures without detection."""
    if solver not in {"signed", "nnls"}:
        raise ValueError("solver must be 'signed' or 'nnls'")
    if background not in {"none", "constant", "planar"}:
        raise ValueError("background must be 'none', 'constant', or 'planar'")
    if not math.isfinite(float(rank_atol)) or rank_atol < 0:
        raise ValueError("rank_atol must be finite and non-negative")
    if rank_rtol is not None and (not math.isfinite(float(rank_rtol)) or rank_rtol < 0):
        raise ValueError("rank_rtol must be finite and non-negative")
    if nnls_max_iterations < 1:
        raise ValueError("nnls_max_iterations must be positive")
    grouping = build_source_groups(
        scene,
        observations,
        index_nside=index_nside,
        footprint_order=footprint_order,
    )
    dtype = scene.dtype
    device = scene.device
    predictions = [
        torch.zeros_like(item.image, dtype=dtype, device=device)
        for item in observations
    ]
    coefficients = torch.full(
        (scene.parameter_count,), float("nan"), dtype=dtype, device=device
    )
    covariance_indices: list[Tensor] = []
    covariance_blocks: list[Tensor] = []
    diagnostics: list[BlendFitDiagnostics] = []
    background_results: list[Tensor] = []
    stamp_cache: dict[tuple[int, int], SourceStamp | None] = {}

    for group_index, (sources, exposures) in enumerate(
        zip(grouping.blend_groups, grouping.group_exposures, strict=True)
    ):
        global_parameters = (
            torch.unique(
                torch.cat(
                    [
                        torch.arange(
                            int(scene.parameter_slices[index, 0].item()),
                            int(scene.parameter_slices[index, 1].item()),
                            dtype=torch.int64,
                            device=device,
                        )
                        for index in sources.tolist()
                    ]
                ),
                sorted=True,
            )
            if sources.numel()
            else torch.empty(0, dtype=torch.int64, device=device)
        )
        flux_count = int(global_parameters.numel())
        global_to_local = {
            int(value): index for index, value in enumerate(global_parameters.tolist())
        }
        exposure_payloads: list[
            tuple[int, tuple[int, int, int, int], Tensor, Tensor, Tensor, int]
        ] = []
        background_total = 0
        for exposure_index_value in exposures.tolist():
            exposure_index = int(exposure_index_value)
            observation = observations[exposure_index]
            visible = set(
                int(value)
                for value in grouping.exposure_sources[exposure_index].tolist()
            )
            stamps = []
            for source_index_value in sources.tolist():
                source_index = int(source_index_value)
                if source_index not in visible:
                    continue
                key = (exposure_index, source_index)
                if key not in stamp_cache:
                    stamp_cache[key] = scene.render_component(
                        source_index, observation, chunk_size=chunk_size
                    )
                stamp = stamp_cache[key]
                if stamp is not None:
                    stamps.append(stamp)
            if not stamps:
                continue
            y0, y1, x0, x1 = _union_bounds(stamps)
            height, width = y1 - y0, x1 - x0
            source_columns = torch.zeros(
                (flux_count, height, width), dtype=dtype, device=device
            )
            for stamp in stamps:
                source_index = stamp.source_index
                weights = scene.spectral_weights(observation.band_name).to(dtype=dtype)
                start = int(scene.parameter_slices[source_index, 0].item())
                sy0 = int(stamp.y_slice.start) - y0
                sy1 = int(stamp.y_slice.stop) - y0
                sx0 = int(stamp.x_slice.start) - x0
                sx1 = int(stamp.x_slice.stop) - x0
                rendered = stamp.data.to(dtype=dtype)
                for offset, weight in enumerate(weights):
                    local = global_to_local[start + offset]
                    source_columns[local, sy0:sy1, sx0:sx1] = (
                        source_columns[local, sy0:sy1, sx0:sx1] + weight * rendered
                    )
            background_columns = _background_columns(
                background,
                (height, width),
                dtype=dtype,
                device=device,
            )
            local_background_count = int(background_columns.shape[0])
            background_start = flux_count + background_total
            background_total += local_background_count
            data = observation.image[y0:y1, x0:x1].to(dtype=dtype)
            valid = observation_support_mask(observation)[y0:y1, x0:x1].to(
                device=device
            )
            if observation.weight is None:
                weight = torch.ones_like(data)
            else:
                weight = observation.weight[y0:y1, x0:x1].to(dtype=dtype).clamp_min(0)
            exposure_payloads.append(
                (
                    exposure_index,
                    (y0, y1, x0, x1),
                    source_columns,
                    background_columns,
                    torch.stack((data, weight, valid.to(dtype=dtype))),
                    background_start,
                )
            )
        total_count = flux_count + background_total
        information = torch.zeros(
            (total_count, total_count), dtype=dtype, device=device
        )
        rhs = torch.zeros(total_count, dtype=dtype, device=device)
        for (
            _,
            _,
            source_columns,
            background_columns,
            packed,
            background_start,
        ) in exposure_payloads:
            data, weight, valid_float = packed
            valid = valid_float > 0
            columns = torch.zeros(
                (total_count, *data.shape), dtype=dtype, device=device
            )
            columns[:flux_count] = source_columns
            count = int(background_columns.shape[0])
            columns[background_start : background_start + count] = background_columns
            selected_columns = columns[:, valid]
            selected_weight = weight[valid]
            selected_data = data[valid]
            weighted_columns = selected_columns * selected_weight[None]
            information = information + selected_columns @ weighted_columns.T
            rhs = rhs + weighted_columns @ selected_data
        eigenvalues, cutoff, data_rank = _rank_analysis(
            information,
            rank_rtol=rank_rtol,
            rank_atol=rank_atol,
        )
        if solver == "signed":
            local_solution, active, active_rank, solved_rank = _signed_solve(
                information, rhs, cutoff
            )
            converged = active_rank == solved_rank
            iterations = 1 if active.numel() else 0
        else:
            local_solution, active, active_rank, converged, iterations = _nnls_solve(
                information,
                rhs,
                flux_count,
                cutoff,
                max_iterations=nnls_max_iterations,
            )
            informative_flux = torch.diagonal(information)[:flux_count].abs() > cutoff
            flux_solution = torch.where(
                informative_flux,
                local_solution[:flux_count],
                torch.full_like(local_solution[:flux_count], float("nan")),
            )
            local_solution = torch.cat((flux_solution, local_solution[flux_count:]))
        local_flux = local_solution[:flux_count]
        coefficients[global_parameters] = local_flux
        active_flux_local, covariance = _covariance_block(
            information, active, flux_count
        )
        active_global = global_parameters[active_flux_local]
        if active_global.numel():
            covariance_indices.append(active_global)
            covariance_blocks.append(covariance)
        local_background = local_solution[flux_count:]
        background_results.append(local_background)
        for (
            exposure_index,
            bounds,
            source_columns,
            background_columns,
            _,
            background_start,
        ) in exposure_payloads:
            y0, y1, x0, x1 = bounds
            source_prediction = torch.einsum(
                "p,phw->hw",
                torch.nan_to_num(local_flux, nan=0.0),
                source_columns,
            )
            count = int(background_columns.shape[0])
            background_values = local_solution[
                background_start : background_start + count
            ]
            background_prediction = (
                torch.einsum("p,phw->hw", background_values, background_columns)
                if count
                else torch.zeros_like(source_prediction)
            )
            predictions[exposure_index][y0:y1, x0:x1] = (
                predictions[exposure_index][y0:y1, x0:x1]
                + source_prediction
                + background_prediction
            )
        informative = (
            torch.diagonal(information).abs() > cutoff
            if total_count
            else torch.empty(0, dtype=torch.bool, device=device)
        )
        gauge_mask = ~informative
        if data_rank < total_count and total_count:
            symmetric = 0.5 * (information + information.T)
            analysis = symmetric.to("cpu") if device.type == "mps" else symmetric
            null_values, null_vectors = torch.linalg.eigh(analysis)
            null_vectors = null_vectors[
                :, null_values <= cutoff.to(device=analysis.device)
            ]
            if null_vectors.numel():
                tolerance = math.sqrt(torch.finfo(dtype).eps)
                involved = null_vectors.abs().amax(dim=1) > tolerance
                gauge_mask = gauge_mask | involved.to(device=device)
        gauge_local = torch.nonzero(gauge_mask, as_tuple=False).flatten()
        gauge_global = global_parameters[gauge_local[gauge_local < flux_count]]
        positive_eigenvalues = eigenvalues[eigenvalues > cutoff]
        condition = (
            math.inf
            if positive_eigenvalues.numel() == 0 or data_rank < total_count
            else float(
                (positive_eigenvalues[-1] / positive_eigenvalues[0]).detach().item()
            )
        )
        diagnostics.append(
            BlendFitDiagnostics(
                group_index=group_index,
                source_indices=sources,
                exposure_indices=exposures,
                parameter_indices=global_parameters,
                active_parameter_indices=active_global,
                data_rank=data_rank,
                active_rank=active_rank,
                parameter_count=total_count,
                active_parameter_count=int(active.numel()),
                condition_number=condition,
                rank_cutoff=cutoff,
                information_eigenvalues=eigenvalues,
                gauge_parameter_indices=gauge_global,
                solver=solver,
                converged=converged,
                iterations=iterations,
            )
        )

    covariance_result = BlockSparseCovariance(
        scene.parameter_count,
        tuple(covariance_indices),
        tuple(covariance_blocks),
        constrained=solver == "nnls",
    )
    coefficient_variance = covariance_result.diagonal()
    width = (
        1
        if scene.component_count == 0
        else int((scene.parameter_slices[0, 1] - scene.parameter_slices[0, 0]).item())
    )
    flux_rows = []
    variance_rows = []
    for source_index in range(scene.component_count):
        start, stop = (
            int(value) for value in scene.parameter_slices[source_index].tolist()
        )
        flux_rows.append(coefficients[start:stop])
        variance_rows.append(coefficient_variance[start:stop])
    flux = (
        torch.stack(flux_rows)
        if flux_rows
        else torch.empty((0, width), dtype=dtype, device=device)
    )
    flux_variance = (
        torch.stack(variance_rows)
        if variance_rows
        else torch.empty((0, width), dtype=dtype, device=device)
    )
    catalog = Catalog(
        lon=scene.lon,
        lat=scene.lat,
        columns={
            "component_id": scene.component_ids,
            "morphology_code": scene.morphology_codes,
            "flux": flux,
            "flux_variance": flux_variance,
            "parameter_slices": scene.parameter_slices,
        },
    )
    residuals = tuple(
        observation.image - prediction.to(dtype=observation.image.dtype)
        for observation, prediction in zip(observations, predictions, strict=True)
    )
    provenance: dict[str, object] = {
        "method": "known-source-fixed-template-forward-photometry",
        "solver": solver,
        "background": background,
        "source_count": scene.component_count,
        "parameter_count": scene.parameter_count,
        "exposure_count": len(observations),
        "blend_group_count": len(grouping.blend_groups),
        "footprint_order": min(
            scene.footprint_order if footprint_order is None else int(footprint_order),
            13 if device.type == "mps" else 29,
        ),
        "index": "HealpixIndex",
        "footprints": "MOC",
        "selection": "caller-supplied-known-sources",
        "covariance": "conditional-fixed-template-active-system",
    }
    return ForwardPhotometryResult(
        catalog=catalog,
        scene=scene,
        coefficients=coefficients,
        covariance=covariance_result,
        predictions=tuple(predictions),
        residuals=residuals,
        grouping=grouping,
        diagnostics=tuple(diagnostics),
        background_coefficients=tuple(background_results),
        provenance=provenance,
    )


__all__ = [
    "BackgroundModel",
    "BlendFitDiagnostics",
    "BlockSparseCovariance",
    "ForwardPhotometryResult",
    "PhotometrySolver",
    "SourceGrouping",
    "build_source_groups",
    "fit_forced_photometry",
]
