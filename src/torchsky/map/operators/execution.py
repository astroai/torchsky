"""Executable operator helpers shared by image-plane mapmaking paths."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence

import torch
from torch import Tensor

from .background import SpatialSkyBackgroundOp
from .bundle import ObservationOperatorBundle, image_observation_operator_bundle
from .geometry import (
    ConservativePolygonRemapOp,
    ExactDrizzleGeometryOp,
    ForwardSplatGeometryOp,
)
from .geometry_conservative import conservative_polygon_project
from .geometry_drizzle import exact_point_drizzle_project, exact_square_drizzle_project
from ..sky.corrector import CrossExposureSkyCorrector
from ...device import _as_tensor, _to


class LoadWeightMapFn(Protocol):
    def __call__(self, obs: Any, image: Tensor, *, device: torch.device) -> Tensor: ...


class ApplyCalibrationFn(Protocol):
    def __call__(
        self, obs: Any, image: Tensor, source_weight: Tensor, /
    ) -> tuple[Tensor, Tensor]: ...


_WeightedProjection = tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor | None]
ResampleToTargetFn = Callable[..., _WeightedProjection]
ForwardProjectFn = Callable[..., _WeightedProjection]

_MEDIAN_SCRATCH_BYTES_PER_SAMPLE = 5
_MEDIAN_WORKING_MEMORY_BYTES = 64 * 1024 * 1024
_SIGMA_CLIPPED_SCRATCH_BYTES_PER_SAMPLE = 12
_SIGMA_CLIPPED_WORKING_MEMORY_BYTES = 64 * 1024 * 1024
_OVERLAP_PROJECTED_BYTES_PER_SAMPLE = 12
_OVERLAP_WEIGHTED_VARIANCE_BYTES_PER_SAMPLE = 4
_OVERLAP_GRAPH_RHS_BYTES_PER_SAMPLE = 4
_OVERLAP_COMPACT_FIT_SCRATCH_BYTES_PER_PIXEL = 4


# ---------------------------------------------------------------------------
# Memory-aware spatial tiling
# ---------------------------------------------------------------------------
#
# ``coadd_image_plane`` projects one observation at a time onto the target
# window.  The projection intermediates (target world grid, source pixel
# coordinates, kernel taps, the second 36-tap pass of the exact diagonal
# variance, flux quadrature grids) scale with the number of output pixels that
# are processed at once, so a large target must be processed in tiles.  The
# coefficients below are the measured peak-RSS slope (bytes per output pixel of
# the processed window) of a single-exposure coadd on CPU, taken from the
# difference between two target sizes so the fixed cost of the interpreter, the
# source image and the drizzle chunk buffers drops out; each value is rounded up
# by about 20% as headroom.  They are upper bounds that hold for the
# weighted-mean stack with or without variance propagation (the variance pass
# runs after the sampling pass and reuses its buffers, so it does not move the
# peak).  Re-measure with the memory probe in the PR description whenever a
# sampler changes: an over-estimate only makes tiles smaller.

# Fraction of the free memory (device memory on CUDA, available host memory
# otherwise) that a coadd plans to use.  The remainder is left to the allocator
# cache, other processes sharing the node, and estimation error.
TILE_MEMORY_FRACTION = 0.30
# Hard override of the planning budget in bytes (None: derive from free memory).
# The ``TORCHSKY_TILE_MEMORY_BUDGET_BYTES`` environment variable and the
# ``tile_memory_budget_bytes`` argument of ``coadd_image_plane`` take
# precedence over this constant, in that order.
TILE_MEMORY_BUDGET_BYTES: int | None = None
# Budget used when the free memory cannot be determined.
_TILE_FALLBACK_BUDGET_BYTES = 2 * 1024**3
# Smallest tile edge the planner will produce, even for a tiny budget.
_TILE_MIN_EDGE = 64

_TILE_BYTES_PER_PIXEL_SURFACE_BRIGHTNESS: dict[str, int] = {
    "nearest": 500,
    "bilinear": 500,
    "bicubic": 1100,
    "catmull_rom": 1100,
    "mitchell_netravali": 1100,
    "lanczos2": 1100,
    "lanczos3": 1900,
    "lanczos4": 3300,
    "lanczos5": 4000,
    # Forward drizzle / splat / polygon clipping run in fixed-size source
    # chunks; only the accumulators and the output scale with the window.
    "drizzle_point": 120,
    "drizzle_square": 120,
    "drizzle_turbo": 120,
    "drizzle_gaussian": 120,
    "drizzle_lanczos2": 120,
    "drizzle_lanczos3": 120,
    "conservative_polygon": 120,
    "splat_point": 120,
    "splat_box": 120,
}
_TILE_BYTES_PER_PIXEL_UNKNOWN_KERNEL = 2000
# Extra bytes per pixel for flux semantics (four-point target-pixel quadrature on
# top of the sampling pass): measured +1.3 kB for bilinear and +6.5 kB for
# lanczos3; the larger value is used for every interpolating kernel.
_TILE_FLUX_QUADRATURE_BYTES_PER_PIXEL = 6500
_TILE_FORWARD_KERNEL_PREFIXES = ("drizzle", "splat", "conservative")
# Working memory independent of the window.  Source images are held several
# times (image, weight, their product, calibrated copies); forward kernels also
# keep fixed-size source-chunk buffers (262144 source pixels per chunk,
# ~1.5 kB per chunk pixel measured).
_TILE_SOURCE_BYTES_PER_PIXEL = 40
_TILE_FORWARD_CHUNK_PIXELS = 262_144
_TILE_FORWARD_CHUNK_BYTES_PER_PIXEL = 1500


def projection_bytes_per_pixel(
    resampling_kernel: str,
    value_semantics: str = "surface_brightness",
) -> int:
    """Return the measured peak working set per processed output pixel."""
    base = _TILE_BYTES_PER_PIXEL_SURFACE_BRIGHTNESS.get(
        resampling_kernel, _TILE_BYTES_PER_PIXEL_UNKNOWN_KERNEL
    )
    if value_semantics == "flux" and not resampling_kernel.startswith(
        _TILE_FORWARD_KERNEL_PREFIXES
    ):
        base += _TILE_FLUX_QUADRATURE_BYTES_PER_PIXEL
    return base


def accumulator_bytes_per_pixel(
    n_observations: int,
    *,
    value_semantics: str = "surface_brightness",
    propagate_variance: bool = True,
    include_context: bool = True,
) -> int:
    """Bytes per output pixel of the full-size accumulators and finalized outputs.

    Numerator, weight, overlap count, (variance weight), the finalized data,
    (inverse variance and correlation factor), the flux pixel-area map and the
    int64 context bit words (one per 32 inputs).
    """
    total = 4 * 3 + 4  # numerator, weight, overlap, finalized data
    if propagate_variance:
        total += 4 * 3  # variance weight, inverse variance, correlation factor
    if value_semantics == "flux":
        total += 4
    if include_context:
        total += 8 * math.ceil(max(int(n_observations), 1) / 32)
    return total


def _read_int_file(path: str) -> int | None:
    try:
        with open(path, encoding="ascii") as handle:
            text = handle.read().strip()
    except (OSError, UnicodeDecodeError):
        return None
    if not text or not text.isdigit():
        return None
    return int(text)


def available_host_memory_bytes() -> int | None:
    """Memory a process can still allocate on this host, or None if unknown.

    The minimum of the kernel's ``MemAvailable`` and the remaining allowance of
    the enclosing cgroup (a container or batch-system memory limit, which
    ``/proc/meminfo`` does not reflect).  ``psutil`` is used only when
    importable; no hard dependency.
    """
    candidates: list[int] = []
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    candidates.append(int(line.split()[1]) * 1024)
                    break
    except (OSError, ValueError, IndexError):
        pass
    for limit_file, usage_file in (
        ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
        (
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
            "/sys/fs/cgroup/memory/memory.usage_in_bytes",
        ),
    ):
        limit = _read_int_file(limit_file)
        usage = _read_int_file(usage_file)
        # cgroup v1 reports "unlimited" as a value near 2**63.
        if limit is not None and usage is not None and limit < 2**60:
            candidates.append(max(limit - usage, 0))
    if not candidates:
        try:
            import psutil  # type: ignore[import-not-found]

            candidates.append(int(psutil.virtual_memory().available))
        except Exception:  # noqa: BLE001 - optional dependency, any failure means unknown
            try:
                candidates.append(
                    int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
                )
            except (AttributeError, ValueError, OSError):
                pass
    return min(candidates) if candidates else None


def available_device_memory_bytes(device: torch.device | str) -> int | None:
    """Free memory on *device*, or None if unknown.

    CUDA: driver-reported free memory plus the part of this process's caching
    allocator that is reserved but unused.  Other devices (CPU, MPS with unified
    memory): available host memory.
    """
    device = torch.device(device)
    if device.type == "cuda":
        try:
            free, _total = torch.cuda.mem_get_info(device)
            reusable = max(
                torch.cuda.memory_reserved(device)
                - torch.cuda.memory_allocated(device),
                0,
            )
            return int(free) + int(reusable)
        except (RuntimeError, AttributeError):
            return None
    return available_host_memory_bytes()


def resolve_tile_memory_budget_bytes(
    device: torch.device | str,
    *,
    budget_bytes: int | None = None,
) -> int:
    """Memory budget in bytes for planning spatial tiles.

    Precedence: *budget_bytes*, ``TORCHSKY_TILE_MEMORY_BUDGET_BYTES``,
    :data:`TILE_MEMORY_BUDGET_BYTES`, then ``TILE_MEMORY_FRACTION`` times the
    free memory of *device*.
    """
    if budget_bytes is not None:
        resolved = int(budget_bytes)
        if resolved <= 0:
            raise ValueError("tile_memory_budget_bytes must be positive")
        return resolved
    env = os.environ.get("TORCHSKY_TILE_MEMORY_BUDGET_BYTES")
    if env:
        try:
            resolved = int(float(env))
        except ValueError as exc:
            raise ValueError(
                "TORCHSKY_TILE_MEMORY_BUDGET_BYTES must be a number of bytes"
            ) from exc
        if resolved <= 0:
            raise ValueError("TORCHSKY_TILE_MEMORY_BUDGET_BYTES must be positive")
        return resolved
    if TILE_MEMORY_BUDGET_BYTES is not None:
        return int(TILE_MEMORY_BUDGET_BYTES)
    free = available_device_memory_bytes(device)
    if free is None:
        return _TILE_FALLBACK_BUDGET_BYTES
    return max(int(free * TILE_MEMORY_FRACTION), 1)


def balanced_tile_hw(
    shape: tuple[int, int],
    max_pixels: int,
    *,
    min_edge: int = _TILE_MIN_EDGE,
) -> tuple[int, int]:
    """Largest near-square tile of at most *max_pixels* whose tiling has no slivers.

    The tile edges are the canvas edges divided into equal parts, so the last
    tile is never a thin remainder and every tile respects the same bound.
    """
    height, width = int(shape[0]), int(shape[1])
    if height * width <= max_pixels:
        return height, width
    side = max(int(math.isqrt(max(int(max_pixels), 1))), 1)
    th = min(height, max(side, min_edge))
    tw = min(width, max(int(max_pixels) // th, min_edge))
    th = math.ceil(height / math.ceil(height / th))
    tw = math.ceil(width / math.ceil(width / tw))
    return th, tw


@dataclass(frozen=True)
class SpatialTilePlan:
    """Result of :func:`plan_spatial_tiles`."""

    tile_hw: tuple[int, int] | None
    host_accumulators: bool
    budget_bytes: int
    untiled_bytes: int
    tile_bytes: int
    bytes_per_pixel: int


def plan_spatial_tiles(
    target_shape: tuple[int, int],
    *,
    resampling_kernel: str,
    value_semantics: str,
    n_observations: int,
    source_pixels: int,
    device: torch.device | str,
    propagate_variance: bool = True,
    budget_bytes: int | None = None,
) -> SpatialTilePlan:
    """Choose a spatial tiling that keeps the coadd inside a memory budget.

    The untiled estimate is ``(projection + accumulator bytes per pixel) *
    target pixels`` plus the fixed and source-image terms.  When it fits the
    budget no tiling is used (``tile_hw=None``).  Otherwise the full-size
    accumulators are charged to the budget (to the host when the compute device
    is not the CPU, since they then live on the host and only tile slices move
    to the device) and the remaining budget, divided by the per-pixel working
    set, bounds the tile area.
    """
    device = torch.device(device)
    height, width = int(target_shape[0]), int(target_shape[1])
    pixels = height * width
    budget = resolve_tile_memory_budget_bytes(device, budget_bytes=budget_bytes)
    projection_bpp = projection_bytes_per_pixel(resampling_kernel, value_semantics)
    accumulator_bpp = accumulator_bytes_per_pixel(
        n_observations,
        value_semantics=value_semantics,
        propagate_variance=propagate_variance,
        include_context=False,
    )
    context_bpp = 8 * math.ceil(max(int(n_observations), 1) / 32)
    source_pixels = max(int(source_pixels), 0)
    fixed = _TILE_SOURCE_BYTES_PER_PIXEL * source_pixels
    if resampling_kernel.startswith(_TILE_FORWARD_KERNEL_PREFIXES):
        fixed += (
            min(source_pixels, _TILE_FORWARD_CHUNK_PIXELS)
            * _TILE_FORWARD_CHUNK_BYTES_PER_PIXEL
        )
    untiled = fixed + pixels * (projection_bpp + accumulator_bpp + context_bpp)
    if untiled <= budget:
        return SpatialTilePlan(None, False, budget, untiled, untiled, projection_bpp)
    # Tiled: the context bit words are always host-resident while accumulating.
    # Accumulators move to the host on an accelerator, where only the tile working
    # set counts; the finished maps are returned to the compute device.
    host_accumulators = device.type != "cpu"
    resident = 0 if host_accumulators else pixels * accumulator_bpp
    staged = 4 * (4 if propagate_variance else 3)  # tile slices moved to the device
    available = budget - fixed - resident
    max_pixels = max(int(available // (projection_bpp + staged)), 1)
    tile_hw = balanced_tile_hw((height, width), max_pixels)
    if tile_hw[0] >= height and tile_hw[1] >= width:
        # The fixed terms alone exceed the budget; tiling cannot help.
        return SpatialTilePlan(None, False, budget, untiled, untiled, projection_bpp)
    tile_pixels = tile_hw[0] * tile_hw[1]
    tile_bytes = fixed + resident + tile_pixels * (projection_bpp + staged)
    return SpatialTilePlan(
        tile_hw, host_accumulators, budget, untiled, tile_bytes, projection_bpp
    )


class TileAccumulators:
    """Full-canvas stacking accumulators with windowed staging to the compute device.

    ``numer``, ``denom``, ``overlap`` (and ``variance`` when requested) live on
    ``storage_device``.  When it differs from the compute device (or *staged* is
    forced), :meth:`window` copies only the requested window to the compute device
    and writes it back on exit, so a large canvas never has to fit on the
    accelerator.  Otherwise it yields views and moves nothing.
    """

    def __init__(
        self,
        shape: tuple[int, int],
        *,
        compute_device: torch.device,
        storage_device: torch.device | None = None,
        with_variance: bool = True,
        staged: bool | None = None,
    ) -> None:
        self.shape = (int(shape[0]), int(shape[1]))
        self.compute_device = torch.device(compute_device)
        self.storage_device = (
            self.compute_device
            if storage_device is None
            else torch.device(storage_device)
        )
        self.staged = (
            self.storage_device != self.compute_device
            if staged is None
            else bool(staged)
        )
        self.numer = torch.zeros(
            self.shape, dtype=torch.float32, device=self.storage_device
        )
        self.denom = torch.zeros_like(self.numer)
        self.overlap = torch.zeros(
            self.shape, dtype=torch.int32, device=self.storage_device
        )
        self.variance: Tensor | None = (
            torch.zeros_like(self.numer) if with_variance else None
        )

    @contextmanager
    def window(
        self, y0: int, y1: int, x0: int, x1: int, *, writeback: bool = True
    ) -> Iterator[tuple[Tensor, Tensor, Tensor, Tensor | None]]:
        sl = (slice(y0, y1), slice(x0, x1))
        fields = [self.numer, self.denom, self.overlap]
        if self.variance is not None:
            fields.append(self.variance)
        if not self.staged:
            yield (
                self.numer[sl],
                self.denom[sl],
                self.overlap[sl],
                None if self.variance is None else self.variance[sl],
            )
            return
        views = [field[sl] for field in fields]
        tiles = [view.to(device=self.compute_device, copy=True) for view in views]
        yield (
            tiles[0],
            tiles[1],
            tiles[2],
            tiles[3] if self.variance is not None else None,
        )
        if writeback:
            for view, tile in zip(views, tiles):
                view.copy_(tile)


@dataclass(frozen=True)
class PreparedImageObservationData:
    image: Tensor
    source_weight: Tensor
    operator_bundle: ObservationOperatorBundle


def stack_inputs_from_projected_observation(
    projected_numer: Tensor,
    projected_weight: Tensor,
    *,
    stacking_method: str,
    value_semantics: str,
    pixel_area_sr: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Convert projected numerator/weight tensors into stack inputs."""
    if stacking_method == "weighted_mean":
        return projected_numer, projected_weight
    if stacking_method == "chi2":
        chi2_numer = torch.where(
            projected_weight > 0.0,
            torch.square(projected_numer)
            / torch.clamp(
                projected_weight,
                min=torch.finfo(projected_weight.dtype).tiny,
            ),
            torch.zeros_like(projected_numer),
        )
        support = (projected_weight > 0.0).to(dtype=torch.float32)
        return chi2_numer, support
    projected_image = torch.where(
        projected_weight > 0.0,
        projected_numer / projected_weight,
        torch.zeros_like(projected_numer),
    )
    if value_semantics == "flux":
        if pixel_area_sr is None:
            raise RuntimeError("target pixel area is required for flux semantics")
        projected_image = pixel_area_sr * projected_image
    support = (projected_weight > 0.0).to(dtype=torch.float32)
    return projected_image, support


def masked_median_stack(
    projected_images: Sequence[Tensor],
    support_masks: Sequence[Tensor],
) -> tuple[Tensor, Tensor]:
    """Take a per-pixel median over only supported projected observations."""
    if not projected_images:
        raise ValueError("median stacking requires at least one projected image")
    values = torch.stack(tuple(projected_images), dim=0)
    support = torch.stack(
        tuple(mask.to(dtype=torch.bool) for mask in support_masks), dim=0
    )

    return _masked_median_dim0(values, support)


def _masked_median_dim0(values: Tensor, support: Tensor) -> tuple[Tensor, Tensor]:
    if values.shape != support.shape or values.ndim < 1:
        raise ValueError("median values and support must have matching dimensions")
    fill = torch.full_like(values, float("inf"))
    ordered = torch.sort(torch.where(support, values, fill), dim=0).values
    count = support.sum(dim=0, dtype=torch.int64)
    safe_count = torch.clamp(count, min=1)
    lower_idx = ((safe_count - 1) // 2).unsqueeze(0)
    upper_idx = (safe_count // 2).unsqueeze(0)
    lower = torch.gather(ordered, 0, lower_idx).squeeze(0)
    upper = torch.gather(ordered, 0, upper_idx).squeeze(0)
    median = 0.5 * (lower + upper)
    median = torch.where(count > 0, median, torch.zeros_like(median))
    return median, count.to(dtype=torch.float32)


def exact_median_scratch_bytes(
    n_observations: int,
    shape: tuple[int, int],
) -> int:
    """Return exact storage bytes for float32 values plus uint8 support."""
    n_obs = int(n_observations)
    height, width = int(shape[0]), int(shape[1])
    if n_obs <= 0:
        raise ValueError("n_observations must be positive")
    if height <= 0 or width <= 0:
        raise ValueError("median scratch shape must be positive")
    return n_obs * height * width * _MEDIAN_SCRATCH_BYTES_PER_SAMPLE


def require_exact_median_scratch_capacity(
    n_observations: int,
    shape: tuple[int, int],
    *,
    scratch_dir: str | Path | None = None,
) -> Path:
    """Create the scratch root and fail closed when exact storage cannot fit."""
    required = exact_median_scratch_bytes(n_observations, shape)
    return _require_scratch_capacity(
        required,
        scratch_dir=scratch_dir,
        label="exact median",
    )


def _require_scratch_capacity(
    required_bytes: int,
    *,
    scratch_dir: str | Path | None,
    label: str,
) -> Path:
    root = Path(tempfile.gettempdir()) if scratch_dir is None else Path(scratch_dir)
    root.mkdir(parents=True, exist_ok=True)
    available = int(shutil.disk_usage(root).free)
    if available < required_bytes:
        raise OSError(
            f"{label} scratch requires {required_bytes} bytes but only "
            f"{available} bytes are available at {root}"
        )
    return root


def spilled_exact_median_stack(
    contributions: Iterable[tuple[Tensor, Tensor]],
    *,
    n_observations: int,
    shape: tuple[int, int],
    scratch_dir: str | Path | None = None,
    output_device: torch.device | str | None = None,
    working_memory_bytes: int = _MEDIAN_WORKING_MEMORY_BYTES,
) -> tuple[Tensor, Tensor]:
    """Compute the exact supported median from disk-backed projected planes.

    Contributions are consumed only after the full scratch requirement has been
    checked. The spill path is intentionally execution-only: detector
    projections that require gradients are rejected instead of being detached
    silently.
    """
    n_obs = int(n_observations)
    height, width = int(shape[0]), int(shape[1])
    root = require_exact_median_scratch_capacity(
        n_obs,
        (height, width),
        scratch_dir=scratch_dir,
    )
    work_bytes = int(working_memory_bytes)
    if work_bytes <= 0:
        raise ValueError("working_memory_bytes must be positive")
    n_pixels = height * width
    n_elements = n_obs * n_pixels
    result_device = None if output_device is None else torch.device(output_device)

    with tempfile.TemporaryDirectory(prefix=".torchsky-median-", dir=root) as work:
        work_path = Path(work)
        mapped_values = torch.from_file(
            str(work_path / "values.bin"),
            shared=True,
            size=n_elements,
            dtype=torch.float32,
        ).reshape(n_obs, height, width)
        mapped_support = torch.from_file(
            str(work_path / "support.bin"),
            shared=True,
            size=n_elements,
            dtype=torch.uint8,
        ).reshape(n_obs, height, width)
        try:
            seen = 0
            for values, support in contributions:
                if seen >= n_obs:
                    raise ValueError("median contribution count exceeds n_observations")
                if tuple(values.shape) != (height, width):
                    raise ValueError("median contribution shape does not match target")
                if tuple(support.shape) != (height, width):
                    raise ValueError("median support shape does not match target")
                if values.requires_grad:
                    raise ValueError(
                        "disk-backed median does not support gradient-tracked inputs"
                    )
                if result_device is None:
                    result_device = values.device
                mapped_values[seen].copy_(
                    _to(values.detach(), device="cpu", dtype=torch.float32)
                )
                mapped_support[seen].copy_(
                    _to(support.detach(), device="cpu", dtype=torch.uint8)
                )
                seen += 1
            if seen != n_obs:
                raise ValueError(
                    f"median contribution count {seen} does not match "
                    f"n_observations={n_obs}"
                )

            assert result_device is not None
            median = torch.empty(n_pixels, dtype=torch.float32, device=result_device)
            counts = torch.empty(n_pixels, dtype=torch.float32, device=result_device)
            flat_values = mapped_values.reshape(n_obs, n_pixels)
            flat_support = mapped_support.reshape(n_obs, n_pixels)
            # ponytail: cap reducer temporaries at 64 MiB by default; expose a
            # policy object only if profiling shows workloads need tuning.
            bytes_per_pixel = max(16 * n_obs, 1)
            chunk_pixels = max(1, min(n_pixels, work_bytes // bytes_per_pixel))
            for start in range(0, n_pixels, chunk_pixels):
                stop = min(start + chunk_pixels, n_pixels)
                supported = flat_support[:, start:stop].to(dtype=torch.bool)
                reduced, count = _masked_median_dim0(
                    flat_values[:, start:stop], supported
                )
                median[start:stop].copy_(reduced.to(device=result_device))
                counts[start:stop].copy_(
                    _to(count, device=result_device, dtype=torch.float32)
                )
            del flat_values, flat_support
        finally:
            del mapped_values, mapped_support
    return median.reshape(height, width), counts.reshape(height, width)


def sigma_clipped_scratch_bytes(
    n_observations: int,
    shape: tuple[int, int],
) -> int:
    """Return storage bytes for values, weights, and variance weights."""
    n_obs = int(n_observations)
    height, width = int(shape[0]), int(shape[1])
    if n_obs <= 0:
        raise ValueError("n_observations must be positive")
    if height <= 0 or width <= 0:
        raise ValueError("sigma-clipped scratch shape must be positive")
    return n_obs * height * width * _SIGMA_CLIPPED_SCRATCH_BYTES_PER_SAMPLE


def require_sigma_clipped_scratch_capacity(
    n_observations: int,
    shape: tuple[int, int],
    *,
    scratch_dir: str | Path | None = None,
) -> Path:
    required = sigma_clipped_scratch_bytes(n_observations, shape)
    return _require_scratch_capacity(
        required,
        scratch_dir=scratch_dir,
        label="sigma-clipped",
    )


@dataclass(frozen=True)
class SpilledSigmaClippedStackResult:
    data: Tensor
    weight: Tensor
    variance_weight: Tensor | None
    overlap_count: Tensor
    context: Tensor


@dataclass(frozen=True)
class SpilledOverlapBackgroundStackResult:
    data: Tensor
    weight: Tensor
    normalization: Tensor
    variance_weight: Tensor | None
    overlap_count: Tensor
    context: Tensor


def overlap_background_scratch_bytes(
    n_observations: int,
    shape: tuple[int, int],
    *,
    stacking_method: str,
) -> int:
    """Return exact spill bytes for global overlap-graph rectification."""
    n_obs = int(n_observations)
    height, width = int(shape[0]), int(shape[1])
    if n_obs <= 0:
        raise ValueError("n_observations must be positive")
    if height <= 0 or width <= 0:
        raise ValueError("overlap background scratch shape must be positive")
    if stacking_method not in ("weighted_mean", "mean", "median"):
        raise ValueError(
            "overlap background stacking_method must be weighted_mean, mean, or median"
        )
    projected_bytes = _OVERLAP_PROJECTED_BYTES_PER_SAMPLE
    if stacking_method == "weighted_mean":
        projected_bytes += _OVERLAP_WEIGHTED_VARIANCE_BYTES_PER_SAMPLE
    bytes_per_pixel = (
        n_obs * projected_bytes
        + max(n_obs - 1, 0) * _OVERLAP_GRAPH_RHS_BYTES_PER_SAMPLE
        + (_OVERLAP_COMPACT_FIT_SCRATCH_BYTES_PER_PIXEL if n_obs > 1 else 0)
    )
    return height * width * bytes_per_pixel


def require_overlap_background_scratch_capacity(
    n_observations: int,
    shape: tuple[int, int],
    *,
    stacking_method: str,
    scratch_dir: str | Path | None = None,
) -> Path:
    required = overlap_background_scratch_bytes(
        n_observations,
        shape,
        stacking_method=stacking_method,
    )
    return _require_scratch_capacity(
        required,
        scratch_dir=scratch_dir,
        label="overlap background",
    )


def _overlap_graph_edges(
    mapped_weight: Tensor,
    *,
    tile_hw: tuple[int, int],
) -> list[tuple[int, int, int]]:
    n_obs, height, width = mapped_weight.shape
    tile_height, tile_width = tile_hw
    bboxes: list[tuple[int, int, int, int] | None] = []
    for obs_idx in range(n_obs):
        by0 = height
        by1 = 0
        bx0 = width
        bx1 = 0
        for y0 in range(0, height, tile_height):
            y1 = min(y0 + tile_height, height)
            for x0 in range(0, width, tile_width):
                x1 = min(x0 + tile_width, width)
                tile_valid = mapped_weight[obs_idx, y0:y1, x0:x1] > 0.0
                if not bool(tile_valid.any()):
                    continue
                ys, xs = torch.nonzero(tile_valid, as_tuple=True)
                by0 = min(by0, y0 + int(ys.min().item()))
                by1 = max(by1, y0 + int(ys.max().item()) + 1)
                bx0 = min(bx0, x0 + int(xs.min().item()))
                bx1 = max(bx1, x0 + int(xs.max().item()) + 1)
        bboxes.append((by0, by1, bx0, bx1) if (by1 > by0 and bx1 > bx0) else None)

    edges: list[tuple[int, int, int]] = []
    for left in range(n_obs):
        box_l = bboxes[left]
        if box_l is None:
            continue
        ly0, ly1, lx0, lx1 = box_l
        for right in range(left + 1, n_obs):
            box_r = bboxes[right]
            if box_r is None:
                continue
            ry0, ry1, rx0, rx1 = box_r
            iy0 = max(ly0, ry0)
            iy1 = min(ly1, ry1)
            ix0 = max(lx0, rx0)
            ix1 = min(lx1, rx1)
            if iy1 <= iy0 or ix1 <= ix0:
                continue
            count = 0
            for y0 in range(iy0, iy1, tile_height):
                for x0 in range(ix0, ix1, tile_width):
                    y1 = min(y0 + tile_height, iy1)
                    x1 = min(x0 + tile_width, ix1)
                    overlap = (mapped_weight[left, y0:y1, x0:x1] > 0.0) & (
                        mapped_weight[right, y0:y1, x0:x1] > 0.0
                    )
                    count += int(overlap.sum().item())
            if count > 0:
                edges.append((left, right, count))
    reached = {0}
    changed = True
    while changed:
        changed = False
        for left, right, _ in edges:
            if left in reached and right not in reached:
                reached.add(right)
                changed = True
            elif right in reached and left not in reached:
                reached.add(left)
                changed = True
    if len(reached) != n_obs:
        raise ValueError("overlap graph is disconnected from anchor observation 0")
    return edges


def spilled_overlap_background_stack(
    contributions: Iterable[
        tuple[
            int,
            Tensor,
            Tensor,
            Tensor | None,
            Tensor,
            tuple[int, int, int, int] | None,
        ]
    ],
    *,
    n_observations: int,
    shape: tuple[int, int],
    stacking_method: str,
    value_semantics: str,
    corrector: CrossExposureSkyCorrector,
    scratch_dir: str | Path | None = None,
    output_device: torch.device | str | None = None,
    tile_hw: tuple[int, int] = (1024, 1024),
    pixel_area_sr: Tensor | None = None,
) -> SpilledOverlapBackgroundStackResult:
    """Rectify one global overlap graph from disk-backed projected planes.

    Projection chunks are consumed once after capacity is checked. Pairwise
    models are fitted one at a time, accumulated into an anchored graph RHS,
    and solved in output tiles; no edge-sized model stack is retained.
    """
    n_obs = int(n_observations)
    height, width = int(shape[0]), int(shape[1])
    if n_obs < 2:
        raise ValueError("overlap background spill requires at least two observations")
    if stacking_method not in ("weighted_mean", "mean", "median"):
        raise ValueError(
            "overlap background stacking_method must be weighted_mean, mean, or median"
        )
    tile_height, tile_width = int(tile_hw[0]), int(tile_hw[1])
    if tile_height <= 0 or tile_width <= 0:
        raise ValueError("overlap background tile dimensions must be positive")
    root = require_overlap_background_scratch_capacity(
        n_obs,
        (height, width),
        stacking_method=stacking_method,
        scratch_dir=scratch_dir,
    )
    result_device = torch.device("cpu" if output_device is None else output_device)
    if value_semantics == "flux":
        if pixel_area_sr is None or tuple(pixel_area_sr.shape) != (height, width):
            raise ValueError("flux overlap correction requires target pixel areas")
    n_pixels = height * width
    n_elements = n_obs * n_pixels

    with tempfile.TemporaryDirectory(prefix=".torchsky-overlap-", dir=root) as work:
        work_path = Path(work)
        mapped_numer = torch.from_file(
            str(work_path / "numer.bin"),
            shared=True,
            size=n_elements,
            dtype=torch.float32,
        ).reshape(n_obs, height, width)
        mapped_weight = torch.from_file(
            str(work_path / "weight.bin"),
            shared=True,
            size=n_elements,
            dtype=torch.float32,
        ).reshape(n_obs, height, width)
        mapped_overlap = torch.from_file(
            str(work_path / "overlap.bin"),
            shared=True,
            size=n_elements,
            dtype=torch.int32,
        ).reshape(n_obs, height, width)
        mapped_variance = None
        if stacking_method == "weighted_mean":
            mapped_variance = torch.from_file(
                str(work_path / "variance.bin"),
                shared=True,
                size=n_elements,
                dtype=torch.float32,
            ).reshape(n_obs, height, width)
        mapped_rhs = torch.from_file(
            str(work_path / "rhs.bin"),
            shared=True,
            size=(n_obs - 1) * n_pixels,
            dtype=torch.float32,
        ).reshape(n_obs - 1, height, width)
        mapped_difference = torch.from_file(
            str(work_path / "difference.bin"),
            shared=True,
            size=n_pixels,
            dtype=torch.float32,
        ).reshape(height, width)
        try:
            mapped_numer.zero_()
            mapped_weight.zero_()
            mapped_overlap.zero_()
            mapped_rhs.zero_()
            if mapped_variance is not None:
                mapped_variance.zero_()
            seen: set[int] = set()
            variance_available = stacking_method == "weighted_mean"
            for (
                observation_index,
                numer,
                weight,
                variance_weight,
                overlap,
                target_window,
            ) in contributions:
                obs_idx = int(observation_index)
                if not 0 <= obs_idx < n_obs:
                    raise ValueError(
                        "overlap contribution observation index is invalid"
                    )
                if numer.requires_grad or weight.requires_grad:
                    raise ValueError(
                        "disk-backed overlap correction does not support gradient-tracked inputs"
                    )
                if variance_weight is not None and variance_weight.requires_grad:
                    raise ValueError(
                        "disk-backed overlap correction does not support gradient-tracked variance"
                    )
                if target_window is None:
                    y0, y1, x0, x1 = 0, height, 0, width
                else:
                    y0, y1, x0, x1 = (int(value) for value in target_window)
                    if not (0 <= y0 <= y1 <= height and 0 <= x0 <= x1 <= width):
                        raise ValueError("overlap contribution window is out of bounds")
                window_shape = (y1 - y0, x1 - x0)
                if (
                    tuple(numer.shape) != window_shape
                    or tuple(weight.shape) != window_shape
                    or tuple(overlap.shape) != window_shape
                ):
                    raise ValueError(
                        "overlap contribution shape does not match target window"
                    )
                mapped_numer[obs_idx, y0:y1, x0:x1].copy_(
                    _to(numer.detach(), device="cpu", dtype=torch.float32)
                )
                mapped_weight[obs_idx, y0:y1, x0:x1].copy_(
                    _to(weight.detach(), device="cpu", dtype=torch.float32)
                )
                mapped_overlap[obs_idx, y0:y1, x0:x1].copy_(
                    _to(overlap.detach(), device="cpu", dtype=torch.int32)
                )
                if mapped_variance is not None:
                    if variance_weight is None:
                        variance_available = False
                    else:
                        if tuple(variance_weight.shape) != window_shape:
                            raise ValueError(
                                "overlap variance shape does not match target window"
                            )
                        mapped_variance[obs_idx, y0:y1, x0:x1].copy_(
                            _to(
                                variance_weight.detach(),
                                device="cpu",
                                dtype=torch.float32,
                            )
                        )
                seen.add(obs_idx)
            if seen != set(range(n_obs)):
                raise ValueError("overlap contributions must include every observation")

            edges = _overlap_graph_edges(
                mapped_weight,
                tile_hw=(tile_height, tile_width),
            )
            max_count = max(count for _, _, count in edges)
            laplacian = torch.zeros((n_obs, n_obs), dtype=torch.float32)
            for left, right, count in edges:
                graph_weight = float(count) / float(max_count)
                laplacian[left, left] += graph_weight
                laplacian[right, right] += graph_weight
                laplacian[left, right] -= graph_weight
                laplacian[right, left] -= graph_weight
                edge_weight_planes = mapped_weight
                edge_numer_planes = mapped_numer

                def difference_tiles():
                    for tile_y0 in range(0, height, tile_height):
                        for tile_x0 in range(0, width, tile_width):
                            tile_y1 = min(tile_y0 + tile_height, height)
                            tile_x1 = min(tile_x0 + tile_width, width)
                            left_weight = edge_weight_planes[
                                left, tile_y0:tile_y1, tile_x0:tile_x1
                            ]
                            right_weight = edge_weight_planes[
                                right, tile_y0:tile_y1, tile_x0:tile_x1
                            ]
                            support = (left_weight > 0.0) & (right_weight > 0.0)
                            left_image = torch.where(
                                left_weight > 0.0,
                                edge_numer_planes[
                                    left, tile_y0:tile_y1, tile_x0:tile_x1
                                ]
                                / left_weight,
                                torch.zeros_like(left_weight),
                            )
                            right_image = torch.where(
                                right_weight > 0.0,
                                edge_numer_planes[
                                    right, tile_y0:tile_y1, tile_x0:tile_x1
                                ]
                                / right_weight,
                                torch.zeros_like(right_weight),
                            )
                            yield (
                                tile_y0,
                                tile_y1,
                                tile_x0,
                                tile_x1,
                                torch.where(
                                    support,
                                    right_image - left_image,
                                    torch.zeros_like(left_image),
                                ),
                                support,
                            )

                compact_model = corrector.fit_difference_model_compact(
                    difference_tiles,
                    image_shape=(height, width),
                    scratch=mapped_difference,
                )
                if compact_model is None:
                    raise ValueError(
                        "tiled overlap correction requires an estimator with "
                        "compact tile rendering"
                    )
                for y0 in range(0, height, tile_height):
                    for x0 in range(0, width, tile_width):
                        y1 = min(y0 + tile_height, height)
                        x1 = min(x0 + tile_width, width)
                        model_tile = compact_model.render_window(y0, y1, x0, x1)
                        if left > 0:
                            mapped_rhs[left - 1, y0:y1, x0:x1].add_(
                                model_tile,
                                alpha=-graph_weight,
                            )
                        if right > 0:
                            mapped_rhs[right - 1, y0:y1, x0:x1].add_(
                                model_tile,
                                alpha=graph_weight,
                            )
                del compact_model

            free_laplacian = laplacian[1:, 1:]
            data = torch.zeros(
                (height, width), dtype=torch.float32, device=result_device
            )
            output_weight = torch.zeros_like(data)
            normalization = torch.zeros_like(data)
            variance_out = torch.zeros_like(data) if variance_available else None
            overlap_out = torch.zeros(
                (height, width), dtype=torch.int32, device=result_device
            )
            context = torch.zeros(
                (math.ceil(n_obs / 32), height, width),
                dtype=torch.int64,
                device=result_device,
            )
            for y0 in range(0, height, tile_height):
                for x0 in range(0, width, tile_width):
                    y1 = min(y0 + tile_height, height)
                    x1 = min(x0 + tile_width, width)
                    tile_shape = (y1 - y0, x1 - x0)
                    free_rhs = mapped_rhs[:, y0:y1, x0:x1].reshape(n_obs - 1, -1)
                    free_solution = torch.linalg.solve(
                        free_laplacian,
                        free_rhs,
                    ).reshape(n_obs - 1, *tile_shape)
                    corrections = torch.cat(
                        [torch.zeros((1, *tile_shape)), free_solution],
                        dim=0,
                    ).to(device=result_device)
                    tile_numer = torch.zeros(
                        tile_shape, dtype=torch.float32, device=result_device
                    )
                    tile_denom = torch.zeros_like(tile_numer)
                    tile_overlap = torch.zeros(
                        tile_shape, dtype=torch.int32, device=result_device
                    )
                    tile_variance = (
                        torch.zeros_like(tile_numer)
                        if variance_out is not None
                        else None
                    )
                    median_images: list[Tensor] = []
                    median_support: list[Tensor] = []
                    area_tile = (
                        None
                        if pixel_area_sr is None
                        else pixel_area_sr[y0:y1, x0:x1].to(device=result_device)
                    )
                    for obs_idx in range(n_obs):
                        weight = mapped_weight[obs_idx, y0:y1, x0:x1].to(
                            device=result_device
                        )
                        numer = mapped_numer[obs_idx, y0:y1, x0:x1].to(
                            device=result_device
                        )
                        support = weight > 0.0
                        image = torch.where(
                            support,
                            numer / weight,
                            torch.zeros_like(numer),
                        )
                        corrected = image - corrections[obs_idx]
                        tile_overlap.add_(
                            mapped_overlap[obs_idx, y0:y1, x0:x1].to(
                                device=result_device
                            )
                        )
                        encoded = support.to(dtype=torch.int64) * (1 << (obs_idx % 32))
                        context[obs_idx // 32, y0:y1, x0:x1].bitwise_or_(encoded)
                        if stacking_method == "median":
                            median_images.append(corrected)
                            median_support.append(support)
                        else:
                            contrib_numer, contrib_weight = (
                                stack_inputs_from_projected_observation(
                                    corrected * weight,
                                    weight,
                                    stacking_method=stacking_method,
                                    value_semantics=value_semantics,
                                    pixel_area_sr=area_tile,
                                )
                            )
                            tile_numer.add_(contrib_numer)
                            tile_denom.add_(contrib_weight)
                        if tile_variance is not None:
                            assert mapped_variance is not None
                            tile_variance.add_(
                                mapped_variance[obs_idx, y0:y1, x0:x1].to(
                                    device=result_device
                                )
                            )
                    if stacking_method == "median":
                        tile_data, tile_weight = masked_median_stack(
                            median_images,
                            median_support,
                        )
                        tile_normalization = tile_weight
                    else:
                        tile_normalization = tile_denom
                        tile_data, tile_weight = finalize_projected_stack(
                            tile_numer,
                            tile_denom,
                            stacking_method=stacking_method,
                            value_semantics=value_semantics,
                            pixel_area_sr=area_tile,
                        )
                    data[y0:y1, x0:x1].copy_(tile_data)
                    output_weight[y0:y1, x0:x1].copy_(tile_weight)
                    normalization[y0:y1, x0:x1].copy_(tile_normalization)
                    overlap_out[y0:y1, x0:x1].copy_(tile_overlap)
                    if variance_out is not None and tile_variance is not None:
                        variance_out[y0:y1, x0:x1].copy_(tile_variance)
        finally:
            del mapped_numer, mapped_weight, mapped_overlap
            del mapped_rhs, mapped_difference
            if mapped_variance is not None:
                del mapped_variance
    return SpilledOverlapBackgroundStackResult(
        data=data,
        weight=output_weight,
        normalization=normalization,
        variance_weight=variance_out,
        overlap_count=overlap_out,
        context=context,
    )


def spilled_sigma_clipped_stack(
    contributions: Iterable[
        tuple[
            Tensor,
            Tensor,
            Tensor | None,
            tuple[int, int, int, int] | None,
        ]
    ],
    *,
    n_observations: int,
    shape: tuple[int, int],
    clip_sigma: float = 4.0,
    clip_iterations: int = 3,
    scratch_dir: str | Path | None = None,
    output_device: torch.device | str | None = None,
    working_memory_bytes: int = _SIGMA_CLIPPED_WORKING_MEMORY_BYTES,
) -> SpilledSigmaClippedStackResult:
    """Iteratively clip projected planes and return a weighted mean.

    The robust scale is the Gaussian-consistent median absolute deviation. When
    every contribution supplies explicit propagated variance, that scale is
    combined in quadrature with its per-observation noise. Relative coadd
    weights otherwise affect only the final mean, not the clipping threshold.
    All observation planes live in scratch; only bounded pixel slabs are sorted.
    """
    n_obs = int(n_observations)
    height, width = int(shape[0]), int(shape[1])
    sigma = float(clip_sigma)
    iterations = int(clip_iterations)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("clip_sigma must be finite and positive")
    if iterations <= 0:
        raise ValueError("clip_iterations must be positive")
    work_bytes = int(working_memory_bytes)
    if work_bytes <= 0:
        raise ValueError("working_memory_bytes must be positive")
    root = require_sigma_clipped_scratch_capacity(
        n_obs,
        (height, width),
        scratch_dir=scratch_dir,
    )
    n_pixels = height * width
    n_elements = n_obs * n_pixels
    result_device = None if output_device is None else torch.device(output_device)

    with tempfile.TemporaryDirectory(prefix=".torchsky-clipped-", dir=root) as work:
        work_path = Path(work)
        mapped_values = torch.from_file(
            str(work_path / "values.bin"),
            shared=True,
            size=n_elements,
            dtype=torch.float32,
        ).reshape(n_obs, height, width)
        mapped_weights = torch.from_file(
            str(work_path / "weights.bin"),
            shared=True,
            size=n_elements,
            dtype=torch.float32,
        ).reshape(n_obs, height, width)
        mapped_variance = torch.from_file(
            str(work_path / "variance.bin"),
            shared=True,
            size=n_elements,
            dtype=torch.float32,
        ).reshape(n_obs, height, width)
        try:
            seen = 0
            variance_available = True
            for values, weights, variance_weight, target_window in contributions:
                if seen >= n_obs:
                    raise ValueError(
                        "sigma-clipped contribution count exceeds n_observations"
                    )
                if values.requires_grad or weights.requires_grad:
                    raise ValueError(
                        "disk-backed sigma clipping does not support gradient-tracked inputs"
                    )
                if variance_weight is not None and variance_weight.requires_grad:
                    raise ValueError(
                        "disk-backed sigma clipping does not support gradient-tracked variance"
                    )
                mapped_values[seen].zero_()
                mapped_weights[seen].zero_()
                mapped_variance[seen].zero_()
                if target_window is None:
                    y0, y1, x0, x1 = 0, height, 0, width
                else:
                    y0, y1, x0, x1 = (int(value) for value in target_window)
                    if not (0 <= y0 <= y1 <= height and 0 <= x0 <= x1 <= width):
                        raise ValueError("sigma-clipped target window is out of bounds")
                window_shape = (y1 - y0, x1 - x0)
                if (
                    tuple(values.shape) != window_shape
                    or tuple(weights.shape) != window_shape
                ):
                    raise ValueError(
                        "sigma-clipped contribution shape does not match target window"
                    )
                if result_device is None:
                    result_device = values.device
                values_cpu = _to(values.detach(), device="cpu", dtype=torch.float32)
                weights_cpu = _to(weights.detach(), device="cpu", dtype=torch.float32)
                valid = (
                    torch.isfinite(values_cpu)
                    & torch.isfinite(weights_cpu)
                    & (weights_cpu > 0.0)
                )
                mapped_values[seen, y0:y1, x0:x1].copy_(
                    torch.where(valid, values_cpu, torch.zeros_like(values_cpu))
                )
                mapped_weights[seen, y0:y1, x0:x1].copy_(
                    torch.where(valid, weights_cpu, torch.zeros_like(weights_cpu))
                )
                if variance_weight is None:
                    variance_available = False
                else:
                    if tuple(variance_weight.shape) != window_shape:
                        raise ValueError(
                            "sigma-clipped variance shape does not match target window"
                        )
                    variance_cpu = _to(
                        variance_weight.detach(), device="cpu", dtype=torch.float32
                    )
                    if bool(
                        (~torch.isfinite(variance_cpu[valid])).any()
                        or (variance_cpu[valid] < 0.0).any()
                    ):
                        raise ValueError(
                            "sigma-clipped variance weights must be finite and non-negative"
                        )
                    mapped_variance[seen, y0:y1, x0:x1].copy_(
                        torch.where(valid, variance_cpu, torch.zeros_like(variance_cpu))
                    )
                seen += 1
            if seen != n_obs:
                raise ValueError(
                    f"sigma-clipped contribution count {seen} does not match "
                    f"n_observations={n_obs}"
                )

            assert result_device is not None
            data = torch.empty(n_pixels, dtype=torch.float32, device=result_device)
            weight = torch.empty(n_pixels, dtype=torch.float32, device=result_device)
            overlap_count = torch.empty(
                n_pixels, dtype=torch.int32, device=result_device
            )
            context = torch.zeros(
                (math.ceil(n_obs / 32), n_pixels),
                dtype=torch.int64,
                device=result_device,
            )
            variance_out = (
                torch.empty(n_pixels, dtype=torch.float32, device=result_device)
                if variance_available
                else None
            )
            flat_values = mapped_values.reshape(n_obs, n_pixels)
            flat_weights = mapped_weights.reshape(n_obs, n_pixels)
            flat_variance = mapped_variance.reshape(n_obs, n_pixels)
            # ponytail: the fixed working-set cap bounds repeated robust sorts;
            # make it configurable only after a measured crossover warrants it.
            bytes_per_pixel = max(48 * n_obs, 1)
            chunk_pixels = max(1, min(n_pixels, work_bytes // bytes_per_pixel))
            for start in range(0, n_pixels, chunk_pixels):
                stop = min(start + chunk_pixels, n_pixels)
                values = flat_values[:, start:stop]
                weights = flat_weights[:, start:stop]
                active = weights > 0.0
                for _ in range(iterations):
                    center, count = _masked_median_dim0(values, active)
                    deviation = torch.abs(values - center.unsqueeze(0))
                    mad, _ = _masked_median_dim0(deviation, active)
                    robust_sigma = 1.482602218505602 * mad
                    if variance_available:
                        numerator_variance = flat_variance[:, start:stop]
                        noise_sigma = torch.where(
                            weights > 0.0,
                            torch.sqrt(torch.clamp(numerator_variance, min=0.0))
                            / torch.clamp(weights, min=torch.finfo(weights.dtype).tiny),
                            torch.full_like(weights, float("inf")),
                        )
                        clip_scale = torch.sqrt(
                            torch.square(robust_sigma).unsqueeze(0)
                            + torch.square(noise_sigma)
                        )
                    else:
                        numerical_floor = torch.finfo(values.dtype).eps * torch.clamp(
                            torch.abs(center), min=1.0
                        )
                        clip_scale = torch.maximum(
                            robust_sigma, numerical_floor
                        ).unsqueeze(0)
                    threshold = sigma * clip_scale
                    updated = active & (
                        (deviation <= threshold) | (count.unsqueeze(0) < 3.0)
                    )
                    if torch.equal(updated, active):
                        break
                    active = updated
                active_weight = torch.where(active, weights, torch.zeros_like(weights))
                denominator = active_weight.sum(dim=0)
                numerator = (active_weight * values).sum(dim=0)
                reduced = torch.where(
                    denominator > 0.0,
                    numerator / denominator,
                    torch.zeros_like(numerator),
                )
                data[start:stop].copy_(reduced.to(device=result_device))
                weight[start:stop].copy_(denominator.to(device=result_device))
                count = active.sum(dim=0, dtype=torch.int32)
                overlap_count[start:stop].copy_(count.to(device=result_device))
                if variance_out is not None:
                    variance_chunk = torch.where(
                        active,
                        flat_variance[:, start:stop],
                        torch.zeros_like(flat_variance[:, start:stop]),
                    ).sum(dim=0)
                    variance_out[start:stop].copy_(
                        variance_chunk.to(device=result_device)
                    )
                for obs_idx in range(n_obs):
                    encoded = active[obs_idx].to(dtype=torch.int64) * (
                        1 << (obs_idx % 32)
                    )
                    context[obs_idx // 32, start:stop].bitwise_or_(
                        encoded.to(device=result_device)
                    )
            del flat_values, flat_weights, flat_variance
        finally:
            del mapped_values, mapped_weights, mapped_variance
    return SpilledSigmaClippedStackResult(
        data=data.reshape(height, width),
        weight=weight.reshape(height, width),
        variance_weight=(
            None if variance_out is None else variance_out.reshape(height, width)
        ),
        overlap_count=overlap_count.reshape(height, width),
        context=context.reshape(math.ceil(n_obs / 32), height, width),
    )


def _accumulate_slice(
    numer_slice: Tensor,
    denom_slice: Tensor,
    overlap_slice: Tensor,
    contrib_numer: Tensor,
    contrib_weight: Tensor,
    overlap_contrib: Tensor,
    *,
    stacking_method: str,
) -> None:
    if stacking_method in ("min", "max"):
        supported = contrib_weight > 0.0
        if bool(supported.any()):
            had_support = denom_slice > 0.0
            combined = (
                torch.minimum(numer_slice, contrib_numer)
                if stacking_method == "min"
                else torch.maximum(numer_slice, contrib_numer)
            )
            updated = torch.where(had_support, combined, contrib_numer)
            numer_slice.copy_(torch.where(supported, updated, numer_slice))
            denom_slice.add_(contrib_weight)
            overlap_slice.add_(overlap_contrib)
        return
    numer_slice.add_(contrib_numer)
    denom_slice.add_(contrib_weight)
    overlap_slice.add_(overlap_contrib)


def accumulate_projected_observation(
    numer: Tensor,
    denom: Tensor,
    overlap_count: Tensor,
    *,
    projected_numer: Tensor,
    projected_weight: Tensor,
    projected_overlap: Tensor,
    stacking_method: str,
    value_semantics: str,
    pixel_area_sr: Tensor | None = None,
    target_window: tuple[int, int, int, int] | None = None,
    coeffs: Tensor | None = None,
) -> None:
    """Accumulate a projected observation into image-plane or cube stack tensors."""
    contrib_numer, contrib_weight = stack_inputs_from_projected_observation(
        projected_numer,
        projected_weight,
        stacking_method=stacking_method,
        value_semantics=value_semantics,
        pixel_area_sr=pixel_area_sr,
    )
    if coeffs is None:
        overlap_contrib = projected_overlap
        if target_window is None:
            _accumulate_slice(
                numer,
                denom,
                overlap_count,
                contrib_numer,
                contrib_weight,
                overlap_contrib,
                stacking_method=stacking_method,
            )
            return
        y0, y1, x0, x1 = target_window
        if y1 <= y0 or x1 <= x0:
            return
        _accumulate_slice(
            numer[y0:y1, x0:x1],
            denom[y0:y1, x0:x1],
            overlap_count[y0:y1, x0:x1],
            contrib_numer,
            contrib_weight,
            overlap_contrib,
            stacking_method=stacking_method,
        )
        return
    coeffs_t = _as_tensor(
        coeffs,
        dtype=contrib_numer.dtype,
        device=contrib_numer.device,
    )
    overlap_coeffs = (coeffs_t > 0.0).to(dtype=projected_overlap.dtype)
    denom_contrib = coeffs_t * contrib_weight.unsqueeze(0)
    numer_contrib = (
        contrib_numer.unsqueeze(0).expand_as(denom_contrib)
        if stacking_method in ("min", "max")
        else coeffs_t * contrib_numer.unsqueeze(0)
    )
    overlap_contrib = overlap_coeffs * projected_overlap.unsqueeze(0)
    if target_window is None:
        _accumulate_slice(
            numer,
            denom,
            overlap_count,
            numer_contrib,
            denom_contrib,
            overlap_contrib,
            stacking_method=stacking_method,
        )
        return
    y0, y1, x0, x1 = target_window
    if y1 <= y0 or x1 <= x0:
        return
    _accumulate_slice(
        numer[:, y0:y1, x0:x1],
        denom[:, y0:y1, x0:x1],
        overlap_count[:, y0:y1, x0:x1],
        numer_contrib,
        denom_contrib,
        overlap_contrib,
        stacking_method=stacking_method,
    )


def finalize_projected_stack(
    numer: Tensor,
    denom: Tensor,
    *,
    stacking_method: str,
    value_semantics: str,
    pixel_area_sr: Tensor | None = None,
    median_images: Sequence[Tensor] | None = None,
    median_support: Sequence[Tensor] | None = None,
) -> tuple[Tensor, Tensor]:
    """Finalize accumulated stack tensors into map values and output weights."""
    if stacking_method == "sigma_clipped":
        raise RuntimeError(
            "sigma_clipped stacking requires the disk-backed clipped reducer"
        )
    if stacking_method == "median":
        if median_images is None or median_support is None:
            raise RuntimeError(
                "median stacking requires projected image/support buffers"
            )
        return masked_median_stack(median_images, median_support)
    if stacking_method in ("sum", "min", "max"):
        coadd = torch.where(
            denom > 0.0,
            numer,
            torch.zeros_like(numer),
        )
        return coadd, denom
    if stacking_method == "chi2":
        coadd = torch.where(
            denom > 0.0,
            torch.sqrt(torch.clamp(numer / denom, min=0.0)),
            torch.zeros_like(numer),
        )
        return coadd, denom
    if stacking_method == "weighted_mean" and value_semantics == "flux":
        if pixel_area_sr is None:
            raise RuntimeError("target pixel area is required for flux semantics")
        coadd = torch.where(
            denom > 0.0,
            pixel_area_sr * numer / denom,
            torch.zeros_like(numer),
        )
        out_weight = torch.where(
            denom > 0.0,
            denom
            / torch.clamp(
                torch.square(pixel_area_sr),
                min=torch.finfo(pixel_area_sr.dtype).tiny,
            ),
            torch.zeros_like(denom),
        )
        return coadd, out_weight
    coadd = torch.where(
        denom > 0.0,
        numer / denom,
        torch.zeros_like(numer),
    )
    return coadd, denom


def prepare_image_observation_data(
    observation: object,
    image: Tensor,
    *,
    device: torch.device,
    resampling_kernel: str,
    load_weight_map_fn: LoadWeightMapFn,
    apply_calibration_fn: ApplyCalibrationFn,
    background_estimator: Callable[[Tensor, Tensor | None], Tensor] | None = None,
) -> PreparedImageObservationData:
    """Normalize image-plane observation inputs into operator-ready tensors.

    When *background_estimator* is provided, a
    :class:`SpatialSkyBackgroundOp` is injected into the operator bundle.
    The spatial sky model is estimated and subtracted **after** calibration
    (which handles the constant ``background_offset``), so the spatial model
    captures residual spatially-varying background on calibrated data.
    """
    operator_bundle = image_observation_operator_bundle(
        observation,
        resampling_kernel=resampling_kernel,
    )
    if background_estimator is not None:
        operator_bundle = operator_bundle.with_updates(
            background=SpatialSkyBackgroundOp(background_estimator),
        )
    if operator_bundle.mask_weight is not None or operator_bundle.noise is not None:
        source_weight = load_weight_map_fn(observation, image, device=device)
    else:
        source_weight = _to(torch.isfinite(image), dtype=torch.float32, device=device)
    if operator_bundle.radiometry is not None or operator_bundle.background is not None:
        prepared_image, prepared_weight = apply_calibration_fn(
            observation,
            image,
            source_weight,
        )
    else:
        prepared_image = torch.nan_to_num(
            image.to(dtype=torch.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        prepared_weight = source_weight.to(dtype=torch.float32)
    spatial_bg = operator_bundle.background
    if isinstance(spatial_bg, SpatialSkyBackgroundOp) and spatial_bg.enabled:
        estimator = spatial_bg.background_estimator
        assert estimator is not None
        sky_model = estimator(prepared_image, None)
        prepared_image = prepared_image - sky_model.to(dtype=prepared_image.dtype)
    return PreparedImageObservationData(
        image=prepared_image,
        source_weight=prepared_weight,
        operator_bundle=operator_bundle,
    )


def project_image_observation_geometry(
    image: Tensor,
    source_weight: Tensor,
    source_wcs,
    target,
    *,
    geometry=None,
    operator_bundle: ObservationOperatorBundle | None = None,
    resampling_kernel: str,
    value_semantics: str,
    drizzle_pixfrac: float,
    pixel_area_sr_map_fn,
    chunk_pixels: int,
    resample_weighted_to_target_fn: ResampleToTargetFn,
    forward_project_weighted_to_target_fn: ForwardProjectFn,
    return_variance_weight: bool = False,
) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor | None]:
    """Dispatch geometry execution through operator-family types instead of local branches."""
    if geometry is None:
        if operator_bundle is None:
            raise ValueError("Either geometry or operator_bundle must be provided")
        geometry = operator_bundle.geometry
    if isinstance(geometry, ExactDrizzleGeometryOp):
        if resampling_kernel == "drizzle_square":
            return exact_square_drizzle_project(
                image,
                source_weight,
                source_wcs,
                target,
                value_semantics=value_semantics,
                drizzle_pixfrac=drizzle_pixfrac,
                pixel_area_sr_map_fn=pixel_area_sr_map_fn,
                chunk_pixels=chunk_pixels,
                return_variance_weight=return_variance_weight,
            )
        numer, weight = exact_point_drizzle_project(
            image,
            source_weight,
            source_wcs,
            target,
            value_semantics=value_semantics,
            pixel_area_sr_map_fn=pixel_area_sr_map_fn,
            chunk_pixels=chunk_pixels,
        )
        return (numer, weight, weight) if return_variance_weight else (numer, weight)
    if isinstance(geometry, ConservativePolygonRemapOp):
        conservative_result = conservative_polygon_project(
            image,
            source_weight,
            source_wcs,
            target,
            value_semantics=value_semantics,
            drizzle_pixfrac=drizzle_pixfrac,
            pixel_area_sr_map_fn=pixel_area_sr_map_fn,
            chunk_pixels=chunk_pixels,
            return_variance_weight=return_variance_weight,
        )
        if return_variance_weight:
            if len(conservative_result) == 3:
                return conservative_result
            return conservative_result[0], conservative_result[1], None
        return conservative_result[0], conservative_result[1]
    if isinstance(geometry, ForwardSplatGeometryOp):
        forward_result = forward_project_weighted_to_target_fn(
            image,
            source_weight,
            source_wcs,
            target,
            resampling_kernel=resampling_kernel,
            value_semantics=value_semantics,
            drizzle_pixfrac=drizzle_pixfrac,
            **({"return_variance_weight": True} if return_variance_weight else {}),
        )
        if return_variance_weight:
            if len(forward_result) == 3:
                return forward_result
            return forward_result[0], forward_result[1], None
        return forward_result[0], forward_result[1]
    interpolation_result = resample_weighted_to_target_fn(
        image,
        source_weight,
        source_wcs,
        target,
        resampling_kernel=resampling_kernel,
        value_semantics=value_semantics,
        **({"return_variance_weight": True} if return_variance_weight else {}),
    )
    if return_variance_weight:
        if len(interpolation_result) == 3:
            return interpolation_result
        return interpolation_result[0], interpolation_result[1], None
    return interpolation_result[0], interpolation_result[1]


def source_bbox_on_target(
    source_wcs,
    source_shape: tuple[int, int],
    target_wcs,
    target_shape: tuple[int, int],
    *,
    margin: int,
    edge_samples: int,
) -> tuple[int, int, int, int] | None:
    """Estimate the clipped target-pixel box covered by a source image boundary."""
    h_src, w_src = int(source_shape[0]), int(source_shape[1])
    edge_samples = max(int(edge_samples), 2)
    xs = torch.linspace(
        0.0,
        float(w_src - 1),
        steps=edge_samples,
        dtype=torch.float64,
        device=source_wcs.device,
    )
    ys = torch.linspace(
        0.0,
        float(h_src - 1),
        steps=edge_samples,
        dtype=torch.float64,
        device=source_wcs.device,
    )
    top_x = xs
    top_y = torch.zeros_like(xs)
    bottom_x = xs
    bottom_y = torch.full_like(xs, float(h_src - 1))
    left_x = torch.zeros_like(ys)
    left_y = ys
    right_x = torch.full_like(ys, float(w_src - 1))
    right_y = ys
    x_src = torch.cat([top_x, bottom_x, left_x, right_x], dim=0)
    y_src = torch.cat([top_y, bottom_y, left_y, right_y], dim=0)
    ra_deg, dec_deg = source_wcs.pixel_to_world(x_src, y_src)
    x_tgt, y_tgt = target_wcs.world_to_pixel(ra_deg, dec_deg)
    finite = torch.isfinite(x_tgt) & torch.isfinite(y_tgt)
    if not bool(finite.any()):
        return None
    x_good = x_tgt[finite]
    y_good = y_tgt[finite]
    height, width = int(target_shape[0]), int(target_shape[1])
    x0 = max(int(torch.floor(x_good.min()).item()) - margin, 0)
    x1 = min(int(torch.ceil(x_good.max()).item()) + margin + 1, width)
    y0 = max(int(torch.floor(y_good.min()).item()) - margin, 0)
    y1 = min(int(torch.ceil(y_good.max()).item()) + margin + 1, height)
    if x0 >= x1 or y0 >= y1:
        return None
    return y0, y1, x0, x1


def is_plain_tan_header(header: Mapping[str, object]) -> bool:
    """Return True for simple 2D TAN celestial headers."""
    return str(header.get("CTYPE1", "")).endswith("TAN") and str(
        header.get("CTYPE2", "")
    ).endswith("TAN")


def wcs_has_distortion(wcs) -> bool:
    """Return True when a WCS carries distortion terms unsupported by fast policies."""
    return (
        getattr(wcs, "sip", None) is not None
        or getattr(wcs, "cpdis1", None) is not None
        or getattr(wcs, "cpdis2", None) is not None
        or getattr(wcs, "det2im1", None) is not None
        or getattr(wcs, "det2im2", None) is not None
        or bool(getattr(wcs, "_is_tpv", False))
        or bool(getattr(wcs, "_is_tnx", False))
        or bool(getattr(wcs, "_is_zpx", False))
    )
