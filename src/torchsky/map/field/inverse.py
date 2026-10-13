"""Linear inverse operators for field reconstruction."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ...calibration_evidence import load_operator_calibration_split
from ...wcs import WCS

if TYPE_CHECKING:
    from ..product import MapProductResult

from ..operators.background import ConditionedBackgroundField, LearnedBackgroundOp
from ..operators.bundle import (
    ObservationOperatorBundle,
    field_observation_operator_bundle,
)
from ..operators.learned_kernel import (
    ConditionedKernelPSF,
    SpatioSpectralTemporalEncoder,
)
from ..operators.psf import ExplicitPSFOp, LearnedPSFOp, apply_psf_spec
from ..scenes import DiscreteTargetImage, SceneRepresentation
from ..solvers import (
    BoundObservationOperator,
    BoundObservationOperatorSet,
    LinearSolveResult,
)
from ..target import TargetRenderSpec
from .core import _BaseFieldModel
from .observation_ops import (
    coordinate_grid,
    observation_support_mask,
    wrapped_delta_ra_deg,
)
from .types import FieldDomain, FieldObservation, ForwardContext, PSFSpec

_RAW_COUNT_UNITS = frozenset(
    {
        "adu",
        "count",
        "counts",
        "dn",
        "dn/s",
        "e",
        "e-",
        "e-/s",
        "e/s",
        "electron",
        "electrons",
        "electrons/s",
    }
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_spatial_tile_hw(
    spatial_tile_hw: tuple[int, int] | None,
) -> tuple[int, int] | None:
    if spatial_tile_hw is None:
        return None
    if (
        not isinstance(spatial_tile_hw, tuple)
        or len(spatial_tile_hw) != 2
        or any(
            not isinstance(v, int) or isinstance(v, bool) or v < 1
            for v in spatial_tile_hw
        )
    ):
        raise ValueError("spatial_tile_hw must be a pair of positive integers")
    return int(spatial_tile_hw[0]), int(spatial_tile_hw[1])


def _iter_detector_tiles(
    height: int,
    width: int,
    tile_h: int,
    tile_w: int,
) -> Iterator[tuple[int, int, int, int]]:
    for y0 in range(0, height, tile_h):
        y1 = min(y0 + tile_h, height)
        for x0 in range(0, width, tile_w):
            x1 = min(x0 + tile_w, width)
            yield y0, y1, x0, x1


def prepare_linear_band_grid_observation_data(
    observation: FieldObservation,
    *,
    calibrated_background: Tensor | float | None = None,
) -> Tensor:
    """Subtract fixed additive terms and zero unsupported detector pixels."""
    image = observation.image
    background = torch.full_like(image, float(observation.background_offset))
    for component in (
        observation.sky_background,
        observation.instrument_background,
        observation.detector_background,
    ):
        if component is not None:
            component_t = component.to(
                device=image.device,
                dtype=image.dtype,
            )
            if not bool(torch.isfinite(component_t).all()):
                raise ValueError("background components must be finite")
            background = background + component_t
    if calibrated_background is not None:
        calibrated = torch.as_tensor(
            calibrated_background,
            device=image.device,
            dtype=image.dtype,
        )
        if calibrated.ndim not in (0, 2) or (
            calibrated.ndim == 2 and calibrated.shape != image.shape
        ):
            raise ValueError(
                "calibrated_background must be scalar or match the observation shape"
            )
        if not bool(torch.isfinite(calibrated).all()):
            raise ValueError("calibrated_background must be finite")
        background = background + calibrated
    support = observation_support_mask(observation).to(device=image.device)
    return torch.where(support, image - background, torch.zeros_like(image))


def bind_linear_band_grid_observation_operator(
    observation: FieldObservation,
    domain: FieldDomain,
    *,
    spatial_resolution: int,
    spatial_bounds_local_deg: Sequence[float],
    spatial_tile_hw: tuple[int, int] | None = None,
) -> BoundObservationOperator:
    """Bind a signed, band-integrated surface-brightness grid to one exposure."""
    if domain.coord_mode != "local_tangent":
        raise ValueError("linear band grid requires local_tangent coordinates")
    resolution = int(spatial_resolution)
    if resolution < 2:
        raise ValueError("spatial_resolution must be at least 2")
    if len(spatial_bounds_local_deg) != 4:
        raise ValueError("spatial_bounds_local_deg must contain four values")
    bounds = tuple(float(value) for value in spatial_bounds_local_deg)
    if not all(math.isfinite(value) for value in bounds):
        raise ValueError("spatial_bounds_local_deg must be finite")
    x_min, x_max, y_min, y_max = bounds
    if x_min >= x_max or y_min >= y_max:
        raise ValueError("spatial_bounds_local_deg bounds must be increasing")
    validated_tile_hw = _validate_spatial_tile_hw(spatial_tile_hw)
    if observation.value_semantics != "surface_brightness":
        raise ValueError("linear band grid requires surface_brightness semantics")
    if observation.weight is None:
        raise ValueError("linear band grid requires an explicit inverse-variance weight")
    initial_support = observation_support_mask(observation)
    initial_weight = torch.nan_to_num(
        observation.weight.detach(), nan=0.0, posinf=0.0, neginf=0.0
    )
    usable_weight = torch.where(
        initial_support,
        torch.clamp(initial_weight, min=0.0),
        torch.zeros_like(initial_weight),
    )
    if not bool(usable_weight.sum() > 0.0):
        raise ValueError("linear band grid requires positive usable weight")
    obs_shape = (int(observation.image.shape[0]), int(observation.image.shape[1]))
    footprint_ra, footprint_dec = coordinate_grid(
        observation,
        shape=obs_shape,
        device=observation.image.device,
    )
    cos_dec0 = math.cos(math.radians(domain.center_dec_deg))
    footprint_x = wrapped_delta_ra_deg(footprint_ra, domain.center_ra_deg) * cos_dec0
    footprint_y = footprint_dec - float(domain.center_dec_deg)
    footprint_finite = torch.isfinite(footprint_x) & torch.isfinite(footprint_y)
    if not bool(footprint_finite[initial_support].all()):
        raise ValueError("detector footprint coordinates must be finite on support")
    supported_x = footprint_x[initial_support]
    supported_y = footprint_y[initial_support]
    tolerance = 1.0e-10
    if (
        bool((supported_x < x_min - tolerance).any())
        or bool((supported_x > x_max + tolerance).any())
        or bool((supported_y < y_min - tolerance).any())
        or bool((supported_y > y_max + tolerance).any())
    ):
        raise ValueError("detector footprint lies outside the explicit grid bounds")
    psf_spec = observation.psf_spec
    halo_y = 0
    halo_x = 0
    if psf_spec is not None and psf_spec.mode != "none":
        if psf_spec.mode != "kernel2d":
            raise NotImplementedError(
                "linear band grid currently supports only fixed kernel2d PSFs"
            )
        if not psf_spec.frozen:
            raise ValueError("linear band grid PSF must be frozen")
        if psf_spec.pixel_response_included is not True:
            raise ValueError(
                "linear band grid requires a PSF with detector pixel response included"
            )
        if psf_spec.kernel2d is None or psf_spec.kernel2d.ndim != 2:
            raise ValueError("kernel2d PSF mode requires a 2D kernel")
        halo_y = int(psf_spec.kernel2d.shape[0]) // 2
        halo_x = int(psf_spec.kernel2d.shape[1]) // 2

    target = TargetRenderSpec(
        header={"NAXIS1": resolution, "NAXIS2": resolution},
        shape=(resolution, resolution),
        value_semantics="surface_brightness",
        band_label=observation.band_name,
    )

    sample_grid_cache: dict[tuple[torch.device, torch.dtype], Tensor] = {}
    kernel_cache: dict[tuple[torch.device, torch.dtype], Tensor] = {}

    def _normalized_sample_grid(device: torch.device, dtype: torch.dtype) -> Tensor:
        key = (device, dtype)
        cached = sample_grid_cache.get(key)
        if cached is not None:
            return cached
        ra_deg, dec_deg = coordinate_grid(
            observation,
            shape=obs_shape,
            device=device,
        )
        x_local = wrapped_delta_ra_deg(ra_deg, domain.center_ra_deg) * cos_dec0
        y_local = dec_deg - float(domain.center_dec_deg)
        x_norm = 2.0 * (x_local - x_min) / (x_max - x_min) - 1.0
        y_norm = 2.0 * (y_local - y_min) / (y_max - y_min) - 1.0
        cached = torch.stack((x_norm, y_norm), dim=-1).to(dtype=dtype)
        sample_grid_cache[key] = cached
        return cached

    def _normalized_kernel(device: torch.device, dtype: torch.dtype) -> Tensor:
        assert psf_spec is not None and psf_spec.kernel2d is not None
        key = (device, dtype)
        cached = kernel_cache.get(key)
        if cached is not None:
            return cached
        raw = psf_spec.kernel2d.to(device=device, dtype=dtype)
        normed = raw / torch.clamp(raw.sum(), min=1.0e-8)
        ky, kx = int(normed.shape[0]), int(normed.shape[1])
        cached = normed.view(1, 1, ky, kx)
        kernel_cache[key] = cached
        return cached

    def _forward_tile(
        state_t: Tensor,
        sample_grid: Tensor,
        y0: int,
        y1: int,
        x0: int,
        x1: int,
    ) -> Tensor:
        height, width = obs_shape
        hy0 = max(y0 - halo_y, 0)
        hy1 = min(y1 + halo_y, height)
        hx0 = max(x0 - halo_x, 0)
        hx1 = min(x1 + halo_x, width)
        tile_grid = sample_grid[hy0:hy1, hx0:hx1]
        sampled = (
            F.grid_sample(
                state_t.unsqueeze(0).unsqueeze(0),
                tile_grid.unsqueeze(0),
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            .squeeze(0)
            .squeeze(0)
        )
        if psf_spec is None or psf_spec.mode == "none":
            return sampled
        pad_top = halo_y - (y0 - hy0)
        pad_bottom = halo_y - (hy1 - y1)
        pad_left = halo_x - (x0 - hx0)
        pad_right = halo_x - (hx1 - x1)
        padded = (
            F.pad(
                sampled.unsqueeze(0).unsqueeze(0),
                (pad_left, pad_right, pad_top, pad_bottom),
                mode="constant",
                value=0.0,
            )
            if (pad_top or pad_bottom or pad_left or pad_right)
            else sampled.unsqueeze(0).unsqueeze(0)
        )
        kernel = _normalized_kernel(state_t.device, state_t.dtype)
        return F.conv2d(padded, kernel, padding=0).squeeze(0).squeeze(0)

    def _forward_state(state: Tensor) -> Tensor:
        state_t = torch.as_tensor(state)
        if state_t.shape != (resolution, resolution):
            raise ValueError(
                "linear band grid state shape must match spatial_resolution"
            )
        if not state_t.is_floating_point():
            raise TypeError("linear band grid state must be floating point")
        sample_grid = _normalized_sample_grid(state_t.device, state_t.dtype)
        if validated_tile_hw is None:
            sampled = (
                F.grid_sample(
                    state_t.unsqueeze(0).unsqueeze(0),
                    sample_grid.unsqueeze(0),
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True,
                )
                .squeeze(0)
                .squeeze(0)
            )
            if psf_spec is None or psf_spec.mode == "none":
                return sampled
            return apply_psf_spec(
                sampled,
                psf_spec.to(device=sampled.device, dtype=sampled.dtype),
            )
        tile_h, tile_w = validated_tile_hw
        height, width = obs_shape
        if state_t.requires_grad:
            row_chunks: list[Tensor] = []
            for y0 in range(0, height, tile_h):
                y1 = min(y0 + tile_h, height)
                col_chunks = [
                    _forward_tile(
                        state_t,
                        sample_grid,
                        y0,
                        y1,
                        x0,
                        min(x0 + tile_w, width),
                    )
                    for x0 in range(0, width, tile_w)
                ]
                row_chunks.append(torch.cat(col_chunks, dim=1))
            return torch.cat(row_chunks, dim=0)
        output = torch.empty(obs_shape, device=state_t.device, dtype=state_t.dtype)
        for y0, y1, x0, x1 in _iter_detector_tiles(height, width, tile_h, tile_w):
            output[y0:y1, x0:x1] = _forward_tile(
                state_t, sample_grid, y0, y1, x0, x1
            )
        return output

    def _apply_operator(
        scene: SceneRepresentation,
        target_spec: TargetRenderSpec,
    ) -> Tensor:
        del target_spec
        if not isinstance(scene, DiscreteTargetImage):
            raise TypeError("linear band grid requires a tensor-backed 2D state")
        return _forward_state(scene.data)

    def _apply_adjoint(tensor: Tensor) -> Tensor:
        dual = torch.as_tensor(tensor)
        if dual.shape != observation.image.shape:
            raise ValueError(
                "adjoint input shape must match the bound field observation"
            )
        dtype = dual.dtype if dual.is_floating_point() else torch.float32
        if validated_tile_hw is None:
            with torch.enable_grad():
                probe = torch.zeros(
                    (resolution, resolution),
                    dtype=dtype,
                    device=dual.device,
                    requires_grad=True,
                )
                prediction = _forward_state(probe)
                pairing = torch.sum(prediction * dual.to(dtype=prediction.dtype))
                (gradient,) = torch.autograd.grad(pairing, probe)
            return gradient
        tile_h, tile_w = validated_tile_hw
        height, width = obs_shape
        sample_grid = _normalized_sample_grid(dual.device, dtype)
        total_grad = torch.zeros(
            (resolution, resolution),
            dtype=dtype,
            device=dual.device,
        )
        for y0, y1, x0, x1 in _iter_detector_tiles(height, width, tile_h, tile_w):
            dual_tile = dual[y0:y1, x0:x1]
            if not bool(torch.any(dual_tile != 0)):
                continue
            with torch.enable_grad():
                probe = torch.zeros(
                    (resolution, resolution),
                    dtype=dtype,
                    device=dual.device,
                    requires_grad=True,
                )
                tile_pred = _forward_tile(probe, sample_grid, y0, y1, x0, x1)
                pairing = torch.sum(
                    tile_pred * dual_tile.to(dtype=tile_pred.dtype)
                )
                (tile_grad,) = torch.autograd.grad(pairing, probe)
            total_grad = total_grad + tile_grad
        return total_grad

    weight_cache: dict[tuple[torch.device, torch.dtype], Tensor] = {}

    def _weight_mask(device: torch.device, dtype: torch.dtype) -> Tensor:
        cache_key = (device, dtype)
        weight = weight_cache.get(cache_key)
        if weight is None:
            support = observation_support_mask(observation).to(device=device)
            weight = (
                observation.weight.to(device=device, dtype=dtype)
                if observation.weight is not None
                else torch.ones(obs_shape, device=device, dtype=dtype)
            )
            weight = torch.nan_to_num(weight, nan=0.0, posinf=0.0, neginf=0.0)
            weight = torch.where(
                support,
                torch.clamp(weight, min=0.0),
                torch.zeros_like(weight),
            )
            weight_cache[cache_key] = weight
        return weight

    def _apply_noise(tensor: Tensor) -> Tensor:
        tensor_t = torch.as_tensor(tensor)
        if tensor_t.shape != observation.image.shape:
            raise ValueError(
                "inverse-noise input shape must match the bound field observation"
            )
        return tensor_t * _weight_mask(tensor_t.device, tensor_t.dtype)

    has_psf = bool(psf_spec is not None and psf_spec.mode != "none")

    def _optional_kernel(device: torch.device, dtype: torch.dtype) -> Tensor | None:
        if not has_psf:
            return None
        return _normalized_kernel(device, dtype)

    base_bundle = field_observation_operator_bundle(observation)
    operator_bundle = ObservationOperatorBundle(
        geometry=base_bundle.geometry,
        psf=base_bundle.psf,
        mask_weight=base_bundle.mask_weight,
        noise=base_bundle.noise,
    )
    bound = BoundObservationOperator(
        operators=operator_bundle,
        target=target,
        apply_operator=_apply_operator,
        apply_adjoint=_apply_adjoint,
        apply_noise=_apply_noise,
    )
    object.__setattr__(
        bound,
        "_linear_grid_spec",
        _LinearBandGridExposureSpec(
            resolution=resolution,
            obs_shape=obs_shape,
            spatial_tile_hw=validated_tile_hw,
            has_psf=has_psf,
            sample_grid_fn=_normalized_sample_grid,
            kernel_fn=_optional_kernel,
            weight_fn=_weight_mask,
        ),
    )
    return bound


@dataclass(frozen=True)
class _LinearBandGridExposureSpec:
    resolution: int
    obs_shape: tuple[int, int]
    spatial_tile_hw: tuple[int, int] | None
    has_psf: bool
    sample_grid_fn: Callable[[torch.device, torch.dtype], Tensor]
    kernel_fn: Callable[[torch.device, torch.dtype], Tensor | None]
    weight_fn: Callable[[torch.device, torch.dtype], Tensor]


def _extract_linear_grid_state(
    scene: SceneRepresentation | nn.Module | Tensor,
    resolution: int,
) -> Tensor:
    if isinstance(scene, Tensor):
        state_t = scene
    elif isinstance(scene, DiscreteTargetImage):
        state_t = torch.as_tensor(scene.data)
    else:
        from ..scenes import as_scene_representation

        scene_rep = as_scene_representation(scene)
        if not isinstance(scene_rep, DiscreteTargetImage):
            raise TypeError("linear band grid requires a tensor-backed 2D state")
        state_t = torch.as_tensor(scene_rep.data)
    if state_t.shape != (resolution, resolution):
        raise ValueError("linear band grid state shape must match spatial_resolution")
    if not state_t.is_floating_point():
        raise TypeError("linear band grid state must be floating point")
    return state_t


class _LinearBandGridOperatorSet(BoundObservationOperatorSet):
    """Multi-exposure operator set with batched grid_sample and grouped PSF conv2d."""

    def __init__(
        self,
        operators: tuple[BoundObservationOperator, ...],
        *,
        specs: tuple[_LinearBandGridExposureSpec, ...],
        exposure_chunk_size: int = 16,
    ) -> None:
        super().__init__(operators, exposure_chunk_size=exposure_chunk_size)
        self._specs = specs
        self._resolution = int(specs[0].resolution)
        self._can_batch = all(spec.spatial_tile_hw is None for spec in specs)
        self._group_cache: dict[
            tuple[tuple[int, ...], torch.device, torch.dtype],
            tuple[
                tuple[int, int],
                Tensor,
                Tensor | None,
                Tensor | None,
                Tensor,
                Tensor | None,
            ],
        ] = {}

    def _iter_chunk_indices(self) -> Iterator[tuple[int, ...]]:
        n = len(self.operators)
        for start in range(0, n, self.exposure_chunk_size):
            stop = min(start + self.exposure_chunk_size, n)
            yield tuple(range(start, stop))

    def _batched_group_tensors(
        self,
        indices: tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[
        tuple[int, int],
        Tensor,
        Tensor | None,
        Tensor | None,
        Tensor,
        Tensor | None,
    ]:
        key = (indices, device, dtype)
        cached = self._group_cache.get(key)
        if cached is not None:
            return cached
        specs = [self._specs[idx] for idx in indices]
        shapes = [spec.obs_shape for spec in specs]
        max_h = max(h for h, _ in shapes)
        max_w = max(w for _, w in shapes)
        uniform_shape = all(s == (max_h, max_w) for s in shapes)

        padded_grids: list[Tensor] = []
        padded_weights: list[Tensor] = []
        valid_box: Tensor | None = (
            None
            if uniform_shape
            else torch.zeros(
                (1, len(indices), max_h, max_w), device=device, dtype=dtype
            )
        )
        for b_idx, spec in enumerate(specs):
            h, w = spec.obs_shape
            g = spec.sample_grid_fn(device, dtype)
            wt = spec.weight_fn(device, dtype)
            if (h, w) != (max_h, max_w):
                g_pad = F.pad(
                    g.permute(2, 0, 1),
                    (0, max_w - w, 0, max_h - h),
                    mode="constant",
                    value=0.0,
                ).permute(1, 2, 0)
                wt_pad = F.pad(
                    wt, (0, max_w - w, 0, max_h - h), mode="constant", value=0.0
                )
                assert valid_box is not None
                valid_box[0, b_idx, :h, :w] = 1.0
            else:
                g_pad = g
                wt_pad = wt
                if valid_box is not None:
                    valid_box[0, b_idx, :, :] = 1.0
            padded_grids.append(g_pad)
            padded_weights.append(wt_pad)

        stacked_grids = torch.cat(padded_grids, dim=0).unsqueeze(0)
        stacked_weights = torch.stack(padded_weights, dim=0).unsqueeze(0)
        if not any(spec.has_psf for spec in specs):
            stacked_kernels: Tensor | None = None
            stacked_flipped_kernels: Tensor | None = None
        else:
            raw_kernels = [spec.kernel_fn(device, dtype) for spec in specs]
            max_ky = max(
                int(k.shape[2]) if k is not None else 1 for k in raw_kernels
            )
            max_kx = max(
                int(k.shape[3]) if k is not None else 1 for k in raw_kernels
            )
            padded_kernels: list[Tensor] = []
            for k in raw_kernels:
                if k is None:
                    k_use = torch.ones((1, 1, 1, 1), device=device, dtype=dtype)
                else:
                    k_use = k
                ky, kx = int(k_use.shape[2]), int(k_use.shape[3])
                pad_y = (max_ky - ky) // 2
                pad_x = (max_kx - kx) // 2
                if pad_y > 0 or pad_x > 0:
                    k_use = F.pad(
                        k_use,
                        (pad_x, max_kx - kx - pad_x, pad_y, max_ky - ky - pad_y),
                        mode="constant",
                        value=0.0,
                    )
                padded_kernels.append(k_use)
            stacked_kernels = torch.cat(padded_kernels, dim=0)
            stacked_flipped_kernels = torch.flip(
                stacked_kernels, dims=(-2, -1)
            ).contiguous()
        result = (
            (max_h, max_w),
            stacked_grids,
            stacked_kernels,
            stacked_flipped_kernels,
            stacked_weights,
            valid_box,
        )
        self._group_cache[key] = result
        return result

    def _sample_group(
        self,
        state_t: Tensor,
        max_shape: tuple[int, int],
        stacked_grids: Tensor,
        group_size: int,
    ) -> Tensor:
        res = self._resolution
        max_h, max_w = max_shape
        return F.grid_sample(
            state_t.view(1, 1, res, res),
            stacked_grids,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).view(1, group_size, max_h, max_w)

    def _convolve_sampled(
        self,
        sampled_raw: Tensor,
        stacked_kernels: Tensor | None,
        valid_box: Tensor | None,
        group_size: int,
    ) -> Tensor:
        sampled = sampled_raw if valid_box is None else sampled_raw * valid_box
        if stacked_kernels is None:
            return sampled
        ky, kx = int(stacked_kernels.shape[2]), int(stacked_kernels.shape[3])
        return F.conv2d(
            sampled,
            stacked_kernels,
            padding=(ky // 2, kx // 2),
            groups=group_size,
        )

    def _adjoint_convolve(
        self,
        duals_padded: Tensor,
        stacked_flipped_kernels: Tensor | None,
        valid_box: Tensor | None,
        group_size: int,
    ) -> Tensor:
        if stacked_flipped_kernels is None:
            adj = duals_padded
        else:
            ky, kx = (
                int(stacked_flipped_kernels.shape[2]),
                int(stacked_flipped_kernels.shape[3]),
            )
            adj = F.conv2d(
                duals_padded,
                stacked_flipped_kernels,
                padding=(ky // 2, kx // 2),
                groups=group_size,
            )
        return adj if valid_box is None else adj * valid_box

    def iter_apply_chunks(
        self,
        scene: SceneRepresentation | nn.Module | Tensor,
    ) -> Iterator[tuple[Tensor, ...]]:
        if not self._can_batch:
            yield from super().iter_apply_chunks(scene)
            return
        state_t = _extract_linear_grid_state(scene, self._resolution)
        for chunk_indices in self._iter_chunk_indices():
            (
                max_shape,
                stacked_grids,
                stacked_kernels,
                _,
                _,
                valid_box,
            ) = self._batched_group_tensors(
                chunk_indices, state_t.device, state_t.dtype
            )
            sampled_raw = self._sample_group(
                state_t, max_shape, stacked_grids, len(chunk_indices)
            )
            preds = self._convolve_sampled(
                sampled_raw, stacked_kernels, valid_box, len(chunk_indices)
            ).squeeze(0)
            if valid_box is None:
                yield tuple(preds.unbind(dim=0))
            else:
                out_list: list[Tensor] = []
                for b_idx, idx in enumerate(chunk_indices):
                    h, w = self._specs[idx].obs_shape
                    out_list.append(preds[b_idx, :h, :w])
                yield tuple(out_list)

    def _batched_adjoint_impl(
        self,
        tensors: Sequence[Tensor],
        *,
        weighted: bool,
    ) -> Tensor:
        if len(tensors) != len(self.operators):
            raise ValueError("tensor count must match operator count")
        res = self._resolution
        total: Tensor | None = None
        for chunk_indices in self._iter_chunk_indices():
            first_t = torch.as_tensor(tensors[chunk_indices[0]])
            dtype = first_t.dtype if first_t.is_floating_point() else torch.float32
            device = first_t.device
            (
                max_shape,
                stacked_grids,
                _,
                stacked_flipped_kernels,
                stacked_weights,
                valid_box,
            ) = self._batched_group_tensors(chunk_indices, device, dtype)
            max_h, max_w = max_shape
            padded_duals: list[Tensor] = []
            for idx in chunk_indices:
                t = torch.as_tensor(tensors[idx])
                obs_shape = self._specs[idx].obs_shape
                if t.shape != obs_shape:
                    msg = (
                        "inverse-noise input shape must match the bound field observation"
                        if weighted
                        else "adjoint input shape must match the bound field observation"
                    )
                    raise ValueError(msg)
                t_f = t.to(device=device, dtype=dtype)
                if obs_shape != max_shape:
                    h, w = obs_shape
                    t_f = F.pad(
                        t_f, (0, max_w - w, 0, max_h - h), mode="constant", value=0.0
                    )
                padded_duals.append(t_f)
            duals_padded = torch.stack(padded_duals, dim=0).unsqueeze(0)
            if weighted:
                duals_padded = duals_padded * stacked_weights
            adj_sampled = self._adjoint_convolve(
                duals_padded,
                stacked_flipped_kernels,
                valid_box,
                len(chunk_indices),
            )
            need_graph = bool(adj_sampled.requires_grad)
            with torch.enable_grad():
                probe = torch.zeros(
                    (res, res),
                    dtype=dtype,
                    device=device,
                    requires_grad=True,
                )
                sampled_raw = self._sample_group(
                    probe, max_shape, stacked_grids, len(chunk_indices)
                )
                (grad,) = torch.autograd.grad(
                    sampled_raw,
                    probe,
                    grad_outputs=adj_sampled,
                    create_graph=need_graph,
                )
            total = grad if total is None else total + grad
        assert total is not None
        return total

    def adjoint(self, tensors: Sequence[Tensor]) -> Tensor:
        if not self._can_batch:
            return super().adjoint(tensors)
        return self._batched_adjoint_impl(tensors, weighted=False)

    def weighted_adjoint(self, tensors: Sequence[Tensor]) -> Tensor:
        if not self._can_batch:
            return super().weighted_adjoint(tensors)
        return self._batched_adjoint_impl(tensors, weighted=True)

    def normal_operator_action(
        self,
        scene: SceneRepresentation | nn.Module | Tensor,
    ) -> Tensor:
        if not self._can_batch:
            return super().normal_operator_action(scene)
        state_t = _extract_linear_grid_state(scene, self._resolution)
        res = self._resolution
        need_graph = bool(state_t.requires_grad)
        total: Tensor | None = None
        for chunk_indices in self._iter_chunk_indices():
            (
                max_shape,
                stacked_grids,
                stacked_kernels,
                stacked_flipped_kernels,
                stacked_weights,
                valid_box,
            ) = self._batched_group_tensors(
                chunk_indices, state_t.device, state_t.dtype
            )
            group_size = len(chunk_indices)
            if not need_graph:
                with torch.enable_grad():
                    probe = state_t.detach().requires_grad_(True)
                    sampled_raw = self._sample_group(
                        probe, max_shape, stacked_grids, group_size
                    )
                    with torch.no_grad():
                        preds = self._convolve_sampled(
                            sampled_raw.detach(),
                            stacked_kernels,
                            valid_box,
                            group_size,
                        )
                        weighted_preds = preds * stacked_weights
                        adj_sampled = self._adjoint_convolve(
                            weighted_preds,
                            stacked_flipped_kernels,
                            valid_box,
                            group_size,
                        )
                    (grad,) = torch.autograd.grad(
                        sampled_raw,
                        probe,
                        grad_outputs=adj_sampled,
                    )
            else:
                sampled_live = self._sample_group(
                    state_t, max_shape, stacked_grids, group_size
                )
                preds_live = self._convolve_sampled(
                    sampled_live, stacked_kernels, valid_box, group_size
                )
                weighted_preds = preds_live * stacked_weights
                adj_sampled = self._adjoint_convolve(
                    weighted_preds,
                    stacked_flipped_kernels,
                    valid_box,
                    group_size,
                )
                with torch.enable_grad():
                    probe = torch.zeros(
                        (res, res),
                        dtype=state_t.dtype,
                        device=state_t.device,
                        requires_grad=True,
                    )
                    sampled_probe = self._sample_group(
                        probe, max_shape, stacked_grids, group_size
                    )
                    (grad,) = torch.autograd.grad(
                        sampled_probe,
                        probe,
                        grad_outputs=adj_sampled,
                        create_graph=True,
                    )
            total = grad if total is None else total + grad
        assert total is not None
        return total


def bind_linear_band_grid_observations_to_operator_set(
    observations: Sequence[FieldObservation],
    domain: FieldDomain,
    *,
    spatial_resolution: int,
    spatial_bounds_local_deg: Sequence[float],
    exposure_chunk_size: int = 4,
    spatial_tile_hw: tuple[int, int] | None = None,
) -> BoundObservationOperatorSet:
    """Bind one band's exposures to a shared explicit surface-brightness grid."""
    if not observations:
        raise ValueError("observations must not be empty")
    band_names = {
        str(obs.band_name).strip().upper()
        for obs in observations
        if obs.band_name is not None and str(obs.band_name).strip()
    }
    if len(band_names) > 1:
        raise ValueError(
            "linear band grid operator set requires observations from the same band"
        )
    operators = tuple(
        bind_linear_band_grid_observation_operator(
            observation,
            domain,
            spatial_resolution=spatial_resolution,
            spatial_bounds_local_deg=spatial_bounds_local_deg,
            spatial_tile_hw=spatial_tile_hw,
        )
        for observation in observations
    )
    specs = tuple(getattr(op, "_linear_grid_spec") for op in operators)
    return _LinearBandGridOperatorSet(
        operators,
        specs=specs,
        exposure_chunk_size=exposure_chunk_size,
    )


class LinearBandGridField(_BaseFieldModel):
    """Signed single-band surface-brightness grid over local-tangent bounds."""

    def __init__(
        self,
        domain: FieldDomain,
        *,
        spatial_resolution: int = 256,
        spatial_bounds_local_deg: Sequence[float],
        band_name: str | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__(domain)
        if self.coord_mode != "local_tangent":
            raise ValueError(
                "LinearBandGridField requires local_tangent coordinates"
            )
        self.spatial_resolution = int(spatial_resolution)
        if self.spatial_resolution < 2:
            raise ValueError("spatial_resolution must be at least 2")
        if len(spatial_bounds_local_deg) != 4:
            raise ValueError("spatial_bounds_local_deg must contain four values")
        bounds = tuple(float(value) for value in spatial_bounds_local_deg)
        if not all(math.isfinite(value) for value in bounds):
            raise ValueError("spatial_bounds_local_deg must be finite")
        if bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
            raise ValueError("spatial_bounds_local_deg bounds must be increasing")
        self.spatial_bounds_local_deg: tuple[float, float, float, float] = bounds
        self.band_name: str | None = (
            None
            if band_name is None or not str(band_name).strip()
            else str(band_name).strip().upper()
        )
        self.grid = nn.Parameter(
            torch.zeros(
                (self.spatial_resolution, self.spatial_resolution),
                dtype=dtype,
                device=device,
            )
        )

    def _grid_world_coords(self) -> tuple[Tensor, Tensor]:
        """Return world coordinates ``(ra_deg, dec_deg)`` for the 2D grid nodes."""
        device = self.grid.device
        dtype = torch.float32 if device.type == "mps" else torch.float64
        x_min, x_max, y_min, y_max = self.spatial_bounds_local_deg
        xs = torch.linspace(
            x_min, x_max, steps=self.spatial_resolution, device=device, dtype=dtype
        )
        ys = torch.linspace(
            y_min, y_max, steps=self.spatial_resolution, device=device, dtype=dtype
        )
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        cos_dec = math.cos(math.radians(float(self.domain.center_dec_deg)))
        dra = xx / max(cos_dec, 1.0e-6)
        dec = yy + float(self.domain.center_dec_deg)
        ra = dra + float(self.domain.center_ra_deg)
        return ra, dec

    def initialize_from_product(
        self,
        product: MapProductResult,
        *,
        band_name: str | None = None,
    ) -> None:
        """Warm-start signed grid values from a target-WCS product plane."""
        if self.coord_mode != "local_tangent":
            raise ValueError(
                "LinearBandGridField initialization requires local_tangent"
            )
        data = product.data.to(device=self.grid.device, dtype=self.grid.dtype)
        if data.ndim == 2:
            plane = data
        elif data.ndim == 3:
            target_band = (
                str(band_name).strip().upper()
                if band_name is not None and str(band_name).strip()
                else self.band_name
            )
            if target_band is not None:
                matched_idx: int | None = None
                for idx, band_spec in enumerate(product.product.bands):
                    if str(band_spec.label or "").strip().upper() == target_band:
                        matched_idx = idx
                        break
                if matched_idx is None:
                    raise ValueError(
                        f"MapProductResult has no band matching {target_band!r}"
                    )
                plane = data[matched_idx]
            elif len(product.product.bands) == 1 and int(data.shape[0]) == 1:
                plane = data[0]
            else:
                raise ValueError(
                    "band_name is required to initialize LinearBandGridField from a multi-band product"
                )
        else:
            raise ValueError("product.data must be 2D or 3D")

        target_wcs = WCS(product.header)
        ra_grid, dec_grid = self._grid_world_coords()
        px, py = target_wcs.world_to_pixel(ra_grid, dec_grid)
        target_width = max(int(product.header["NAXIS1"]) - 1, 1)
        target_height = max(int(product.header["NAXIS2"]) - 1, 1)
        x_norm = 2.0 * (px / target_width) - 1.0
        y_norm = 2.0 * (py / target_height) - 1.0
        sample_grid = (
            torch.stack([x_norm, y_norm], dim=-1)
            .unsqueeze(0)
            .to(device=self.grid.device, dtype=self.grid.dtype)
        )
        sampled = (
            F.grid_sample(
                plane.unsqueeze(0).unsqueeze(0),
                sample_grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            .squeeze(0)
            .squeeze(0)
        )
        with torch.no_grad():
            self.grid.copy_(sampled)

    def _coords_unit(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float = 0.0,
    ) -> Tensor:
        encoded = self._encoded_coords(ra_deg, dec_deg, wavelength_angstrom)
        scale = float(self.domain.spatial_scale_deg)
        x_min, x_max, y_min, y_max = self.spatial_bounds_local_deg
        x_unit = (encoded[..., 0] * scale - x_min) / (x_max - x_min)
        y_unit = (encoded[..., 1] * scale - y_min) / (y_max - y_min)
        wavelength_unit = 0.5 * (torch.tanh(encoded[..., 2]) + 1.0)
        return torch.stack((x_unit, y_unit, wavelength_unit), dim=-1).to(
            dtype=self.grid.dtype
        )

    def forward(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float | None = None,
    ) -> Tensor:
        del wavelength_angstrom
        cos_dec0 = math.cos(math.radians(float(self.domain.center_dec_deg)))
        x_local = wrapped_delta_ra_deg(ra_deg, self.domain.center_ra_deg) * cos_dec0
        y_local = dec_deg - float(self.domain.center_dec_deg)
        x_min, x_max, y_min, y_max = self.spatial_bounds_local_deg
        x_norm = 2.0 * (x_local - x_min) / (x_max - x_min) - 1.0
        y_norm = 2.0 * (y_local - y_min) / (y_max - y_min) - 1.0
        sample_grid = (
            torch.stack((x_norm, y_norm), dim=-1)
            .reshape(1, -1, 1, 2)
            .to(device=self.grid.device, dtype=self.grid.dtype)
        )
        sampled = F.grid_sample(
            self.grid.unsqueeze(0).unsqueeze(0),
            sample_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled.reshape(ra_deg.shape)

    def render_bandpass(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelengths: Tensor,
        weights: Tensor,
    ) -> Tensor:
        del wavelengths, weights
        return self.forward(ra_deg, dec_deg)


def _grid_laplacian_action(state: Tensor) -> Tensor:
    if state.ndim != 2:
        raise ValueError("grid smoothness operator requires a 2D grid state")
    out = torch.zeros_like(state)
    if state.shape[1] > 1:
        dx = state[:, 1:] - state[:, :-1]
        out[:, :-1] = out[:, :-1] - dx
        out[:, 1:] = out[:, 1:] + dx
    if state.shape[0] > 1:
        dy = state[1:, :] - state[:-1, :]
        out[:-1, :] = out[:-1, :] - dy
        out[1:, :] = out[1:, :] + dy
    return out


def _grid_weighted_laplacian_action(
    state: Tensor,
    wx: Tensor | None,
    wy: Tensor | None,
) -> Tensor:
    if wx is None and wy is None:
        return _grid_laplacian_action(state)
    if state.ndim != 2:
        raise ValueError("grid smoothness operator requires a 2D grid state")
    out = torch.zeros_like(state)
    if state.shape[1] > 1:
        dx = state[:, 1:] - state[:, :-1]
        if wx is not None:
            dx = wx * dx
        out[:, :-1] = out[:, :-1] - dx
        out[:, 1:] = out[:, 1:] + dx
    if state.shape[0] > 1:
        dy = state[1:, :] - state[:-1, :]
        if wy is not None:
            dy = wy * dy
        out[:-1, :] = out[:-1, :] - dy
        out[1:, :] = out[1:, :] + dy
    return out


def _grid_weighted_laplacian_diagonal_majorizer(
    shape: tuple[int, int],
    wx: Tensor | None,
    wy: Tensor | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if wx is None and wy is None:
        return torch.full(shape, 8.0, device=device, dtype=dtype)
    diag = torch.zeros(shape, device=device, dtype=dtype)
    if shape[1] > 1:
        if wx is not None:
            diag[:, :-1] = diag[:, :-1] + 2.0 * wx
            diag[:, 1:] = diag[:, 1:] + 2.0 * wx
        else:
            diag[:, :-1] = diag[:, :-1] + 2.0
            diag[:, 1:] = diag[:, 1:] + 2.0
    if shape[0] > 1:
        if wy is not None:
            diag[:-1, :] = diag[:-1, :] + 2.0 * wy
            diag[1:, :] = diag[1:, :] + 2.0 * wy
        else:
            diag[:-1, :] = diag[:-1, :] + 2.0
            diag[1:, :] = diag[1:, :] + 2.0
    return diag


try:
    from torchregress.algorithms.irls import (  # type: ignore[import-not-found]
        IRLSConfig as _TorchRegressIRLSConfig,
        calculate_mad as _tr_calculate_mad,
        huber_weights as _tr_huber_weights,
        power_weights as _tr_power_weights,
        tukey_weights as _tr_tukey_weights,
    )

    _HAS_TORCHREGRESS_IRLS = True
except ImportError:
    _HAS_TORCHREGRESS_IRLS = False

    @dataclass
    class _TorchRegressIRLSConfig:  # type: ignore[no-redef]
        weight_fn: str | Callable[..., Tensor] = "tukey"
        weight_params: dict[str, Any] | None = None
        max_iter: int = 50
        tol: float = 1.0e-4
        scale_estimator: str = "mad"
        use_mad_scaling: bool = False

    def _tr_huber_weights(scaled_residuals: Tensor, delta: float = 1.345) -> Tensor:
        abs_r = torch.abs(scaled_residuals)
        return torch.where(
            abs_r <= delta, torch.ones_like(abs_r), delta / (abs_r + 1.0e-8)
        )

    def _tr_tukey_weights(scaled_residuals: Tensor, c: float = 4.685) -> Tensor:
        abs_r = torch.abs(scaled_residuals)
        return torch.where(
            abs_r <= c,
            (1.0 - (scaled_residuals / c) ** 2) ** 2,
            torch.zeros_like(abs_r),
        )

    def _tr_power_weights(
        scaled_residuals: Tensor, a: float = 1.5, b: float = 2.0
    ) -> Tensor:
        abs_r = torch.abs(scaled_residuals)
        return (1.0 + (abs_r / (a + 1.0e-8)) ** b) ** -1

    def _tr_calculate_mad(residuals: Tensor, dim: int = -2) -> Tensor:
        med = torch.median(residuals, dim=dim, keepdim=True).values
        abs_dev = torch.abs(residuals - med)
        mad = torch.median(abs_dev, dim=dim, keepdim=True).values
        return torch.clamp(mad, min=1.0e-8)


IRLSConfig = _TorchRegressIRLSConfig


def _compute_detector_irls_weight(
    residual: Tensor,
    initial_precision: Tensor,
    *,
    data_weight_fn: Literal["none", "huber", "tukey", "power"] | Callable[..., Tensor],
    irls_config: IRLSConfig | None,
    convex_warmup: bool = False,
) -> Tensor:
    fn_spec = (
        irls_config.weight_fn
        if (irls_config is not None and data_weight_fn == "none")
        else data_weight_fn
    )
    if fn_spec == "none":
        return torch.ones_like(residual)
    params: dict[str, Any] = (
        dict(irls_config.weight_params)
        if (irls_config is not None and irls_config.weight_params is not None)
        else {}
    )
    active = initial_precision > 0.0
    sigma_inv = torch.sqrt(torch.clamp(initial_precision, min=0.0))
    scaled_res = residual * sigma_inv
    if irls_config is not None and irls_config.use_mad_scaling and bool(active.any()):
        active_vals = scaled_res[active].unsqueeze(-1)
        mad_scale = (_tr_calculate_mad(active_vals, dim=0) / 0.6745).squeeze()
        scaled_res = scaled_res / torch.clamp(mad_scale, min=1.0)

    if callable(fn_spec):
        weights = fn_spec(scaled_res, **params)
    elif fn_spec == "huber" or (fn_spec == "tukey" and convex_warmup):
        huber_params = {"delta": float(params["delta"])} if "delta" in params else {}
        weights = _tr_huber_weights(scaled_res, **huber_params)
    elif fn_spec == "tukey":
        weights = _tr_tukey_weights(scaled_res, **params)
    elif fn_spec == "power":
        weights = _tr_power_weights(scaled_res, **params)
    else:
        raise ValueError(f"Unknown IRLS data_weight_fn: {fn_spec!r}")
    return torch.where(
        active,
        torch.clamp(weights, min=1.0e-8, max=1.0),
        torch.zeros_like(weights),
    )


def linear_band_grid_diagonal_majorizer(
    operator_or_set: BoundObservationOperator | BoundObservationOperatorSet,
    data: Tensor | Sequence[Tensor],
    *,
    damping: float = 0.0,
    smoothness_weight: float = 0.0,
) -> Tensor:
    """Return the exact non-negative column-sum sensitivity majorizer ``A^T N^-1 1``."""
    if not math.isfinite(float(damping)) or damping < 0.0:
        raise ValueError("damping must be finite and non-negative")
    if not math.isfinite(float(smoothness_weight)) or smoothness_weight < 0.0:
        raise ValueError("smoothness_weight must be finite and non-negative")
    if isinstance(operator_or_set, BoundObservationOperator):
        if isinstance(data, Sequence) and not isinstance(data, Tensor):
            raise TypeError(
                "single BoundObservationOperator requires a 2D data tensor"
            )
        datum = torch.as_tensor(data)
        ones = torch.ones_like(datum)
        sensitivity = operator_or_set.adjoint(
            operator_or_set.apply_noise_model(ones)
        )
    elif isinstance(operator_or_set, BoundObservationOperatorSet):
        if not isinstance(data, Sequence) or isinstance(data, Tensor):
            raise TypeError(
                "BoundObservationOperatorSet requires a sequence of 2D data tensors"
            )
        if len(data) != len(operator_or_set.operators):
            raise ValueError("data count must match operator count")
        ones_tuple = tuple(torch.ones_like(torch.as_tensor(item)) for item in data)
        sensitivity = operator_or_set.weighted_adjoint(ones_tuple)
    else:
        raise TypeError(
            "operator_or_set must be a BoundObservationOperator or BoundObservationOperatorSet"
        )
    diagonal = torch.clamp(sensitivity, min=0.0)
    if damping > 0.0:
        diagonal = diagonal + float(damping)
    if smoothness_weight > 0.0:
        diagonal = diagonal + 8.0 * float(smoothness_weight)
    return diagonal


def solve_linear_band_grid_cg(
    operator_or_set: BoundObservationOperator | BoundObservationOperatorSet,
    data: Tensor | Sequence[Tensor],
    *,
    initial_solution: Tensor | None = None,
    iterations: int = 64,
    damping: float = 0.0,
    smoothness_weight: float = 0.0,
    relative_regularization: bool = False,
    prior_reference: Literal["zero", "initial"] = "zero",
    preconditioner: (
        Literal["diagonal_majorizer", "none"] | Callable[[Tensor], Tensor] | Tensor | None
    ) = "diagonal_majorizer",
    tolerance: float = 1.0e-6,
    irls_steps: int = 1,
    edge_preserving: bool = False,
    non_negative_weight: float = 0.0,
    data_weight_fn: (
        Literal["none", "huber", "tukey", "power"] | Callable[..., Tensor]
    ) = "none",
    irls_config: IRLSConfig | None = None,
    shared_edge_sq: tuple[Tensor, Tensor] | None = None,
    shared_source_sq: Tensor | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> LinearSolveResult:
    """Solve normal equations via Preconditioned Conjugate Gradient or IRLS-PCG."""
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if irls_steps < 1:
        raise ValueError("irls_steps must be positive")
    if not math.isfinite(float(damping)) or damping < 0.0:
        raise ValueError("damping must be finite and non-negative")
    if not math.isfinite(float(smoothness_weight)) or smoothness_weight < 0.0:
        raise ValueError("smoothness_weight must be finite and non-negative")
    if not math.isfinite(float(non_negative_weight)) or non_negative_weight < 0.0:
        raise ValueError("non_negative_weight must be finite and non-negative")
    if prior_reference not in ("zero", "initial"):
        raise ValueError("prior_reference must be 'zero' or 'initial'")
    if not math.isfinite(float(tolerance)) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    if (
        not callable(data_weight_fn)
        and data_weight_fn not in ("none", "huber", "tukey", "power")
    ):
        raise ValueError(
            "data_weight_fn must be 'none', 'huber', 'tukey', 'power', or a callable"
        )
    use_data_irls = bool(data_weight_fn != "none" or irls_config is not None)

    t_setup0 = time.perf_counter()
    is_single_op = isinstance(operator_or_set, BoundObservationOperator)
    data_t: Tensor | None = None
    data_tuple: tuple[Tensor, ...] | None = None
    init_prec_single: Tensor | None = None
    init_prec_tuple: tuple[Tensor, ...] | None = None

    if is_single_op:
        if isinstance(data, Sequence) and not isinstance(data, Tensor):
            raise TypeError(
                "single BoundObservationOperator requires a 2D data tensor"
            )
        data_t = torch.as_tensor(data)
        rhs_base = operator_or_set.adjoint(operator_or_set.apply_noise_model(data_t))
        if use_data_irls:
            init_prec_single = torch.clamp(
                operator_or_set.apply_noise_model(torch.ones_like(data_t)).detach(),
                min=0.0,
            )
    elif isinstance(operator_or_set, BoundObservationOperatorSet):
        if not isinstance(data, Sequence) or isinstance(data, Tensor):
            raise TypeError(
                "BoundObservationOperatorSet requires a sequence of 2D data tensors"
            )
        if len(data) != len(operator_or_set.operators):
            raise ValueError("data count must match operator count")
        data_tuple = tuple(torch.as_tensor(item) for item in data)
        rhs_base = operator_or_set.weighted_adjoint(data_tuple)
        if use_data_irls:
            init_prec_tuple = tuple(
                torch.clamp(
                    op.apply_noise_model(torch.ones_like(d)).detach(), min=0.0
                )
                for op, d in zip(operator_or_set.operators, data_tuple, strict=True)
            )
    else:
        raise TypeError(
            "operator_or_set must be a BoundObservationOperator or BoundObservationOperatorSet"
        )
    rhs_applications = 1

    if initial_solution is None:
        solution = torch.zeros_like(rhs_base)
        has_nonzero_init = False
        init_ref = solution
    else:
        solution = torch.as_tensor(
            initial_solution, dtype=rhs_base.dtype, device=rhs_base.device
        ).clone()
        if solution.shape != rhs_base.shape:
            raise ValueError("initial_solution shape must match operator target shape")
        has_nonzero_init = bool(torch.any(solution != 0.0))
        init_ref = solution.clone()

    need_nonlinear_weights = bool(
        edge_preserving
        or non_negative_weight > 0.0
        or use_data_irls
        or shared_edge_sq is not None
        or shared_source_sq is not None
        or irls_steps > 1
    )

    sensitivity_diag: Tensor | None = None
    if (
        preconditioner == "diagonal_majorizer"
        or relative_regularization
        or need_nonlinear_weights
    ):
        sensitivity_diag = linear_band_grid_diagonal_majorizer(
            operator_or_set,
            data,
            damping=0.0,
            smoothness_weight=0.0,
        )
        rhs_applications += 1

    sens_scale = 1.0
    if sensitivity_diag is not None:
        pos_sens_init = sensitivity_diag[sensitivity_diag > 0.0]
        if pos_sens_init.numel() > 0:
            median_sens = float(torch.quantile(pos_sens_init.double(), 0.50).item())
            if relative_regularization:
                sens_scale = median_sens
        else:
            median_sens = 1.0
    else:
        median_sens = 1.0

    sky_sigma = 1.0 / math.sqrt(max(median_sens, 1.0e-30))
    effective_damping = float(damping) * sens_scale
    effective_smoothness_weight = float(smoothness_weight) * sens_scale
    effective_non_negative_weight = float(non_negative_weight) * sens_scale

    rhs_seconds = time.perf_counter() - t_setup0
    normal_ops = 0
    total_steps_done = 0
    history: list[float] = []
    initial_r0_norm: float | None = None
    converged = False
    preconditioner_name = "none"
    need_grad = bool(rhs_base.requires_grad or solution.requires_grad)

    sky_bg = 0.0
    if (
        relative_regularization
        and prior_reference == "zero"
        and not need_grad
        and sensitivity_diag is not None
    ):
        well_covered = sensitivity_diag >= (0.5 * median_sens)
        if bool(well_covered.any()):
            local_mean = (
                rhs_base.detach()[well_covered].double()
                / sensitivity_diag.detach()[well_covered].double()
            )
            sky_bg = float(torch.median(local_mean).item())

    base_damp_weight: Tensor | None = None
    base_wx: Tensor | None = None
    base_wy: Tensor | None = None
    if (
        relative_regularization
        and sensitivity_diag is not None
        and median_sens > 0.0
    ):
        base_damp_weight = torch.clamp(
            sensitivity_diag.detach() / median_sens,
            min=1.0e-3,
        )
        if base_damp_weight.shape[1] > 1:
            base_wx = torch.minimum(
                base_damp_weight[:, 1:], base_damp_weight[:, :-1]
            )
        if base_damp_weight.shape[0] > 1:
            base_wy = torch.minimum(
                base_damp_weight[1:, :], base_damp_weight[:-1, :]
            )

    t_solve0 = time.perf_counter()
    num_stages = int(irls_steps)
    detector_outlier_fraction = 0.0
    for stage_idx in range(num_stages):
        wx: Tensor | None = base_wx
        wy: Tensor | None = base_wy
        damp_weight: Tensor | None = base_damp_weight
        neg_weight: Tensor | None = None
        stage_rhs_base = rhs_base
        stage_sens_diag = sensitivity_diag
        det_weights: tuple[Tensor, ...] | None = None

        use_stage_weights = (
            stage_idx > 0
            or shared_edge_sq is not None
            or shared_source_sq is not None
            or (
                has_nonzero_init
                and (
                    edge_preserving
                    or effective_non_negative_weight > 0.0
                    or use_data_irls
                )
            )
        )
        if use_stage_weights:
            sol_det = solution.detach()
            is_warmup_stage = bool(stage_idx == 1 and num_stages >= 3)
            if use_data_irls and (stage_idx > 0 or has_nonzero_init):
                if is_single_op:
                    assert isinstance(operator_or_set, BoundObservationOperator)
                    assert data_t is not None and init_prec_single is not None
                    with torch.no_grad():
                        pred_single = operator_or_set.apply(sol_det)
                        res_single = data_t.detach() - pred_single
                        w_single = _compute_detector_irls_weight(
                            res_single,
                            init_prec_single,
                            data_weight_fn=data_weight_fn,
                            irls_config=irls_config,
                            convex_warmup=is_warmup_stage,
                        )
                        det_weights = (w_single,)
                        act_m = init_prec_single > 0.0
                        if bool(act_m.any()):
                            detector_outlier_fraction = float(
                                (w_single[act_m] < 0.5)
                                .to(dtype=torch.float64)
                                .mean()
                                .item()
                            )
                    stage_rhs_base = operator_or_set.adjoint(
                        operator_or_set.apply_noise_model(data_t * det_weights[0])
                    )
                    rhs_applications += 1
                    if sensitivity_diag is not None:
                        stage_sens_diag = torch.clamp(
                            operator_or_set.adjoint(
                                operator_or_set.apply_noise_model(det_weights[0])
                            ),
                            min=0.0,
                        )
                        rhs_applications += 1
                else:
                    assert isinstance(operator_or_set, BoundObservationOperatorSet)
                    assert data_tuple is not None and init_prec_tuple is not None
                    with torch.no_grad():
                        preds_tuple = operator_or_set.apply(sol_det)
                        w_list: list[Tensor] = []
                        out_num = 0
                        act_num = 0
                        for d_e, p_e, w0_e in zip(
                            data_tuple, preds_tuple, init_prec_tuple, strict=True
                        ):
                            w_e = _compute_detector_irls_weight(
                                d_e.detach() - p_e,
                                w0_e,
                                data_weight_fn=data_weight_fn,
                                irls_config=irls_config,
                                convex_warmup=is_warmup_stage,
                            )
                            w_list.append(w_e)
                            act_e = w0_e > 0.0
                            n_act = int(act_e.sum().item())
                            if n_act > 0:
                                act_num += n_act
                                out_num += int((w_e[act_e] < 0.5).sum().item())
                        det_weights = tuple(w_list)
                        if act_num > 0:
                            detector_outlier_fraction = out_num / act_num
                    stage_rhs_base = operator_or_set.weighted_adjoint(
                        tuple(
                            d_e * w_e
                            for d_e, w_e in zip(data_tuple, det_weights, strict=True)
                        )
                    )
                    rhs_applications += 1
                    if sensitivity_diag is not None:
                        stage_sens_diag = torch.clamp(
                            operator_or_set.weighted_adjoint(det_weights),
                            min=0.0,
                        )
                        rhs_applications += 1

            prior_sol = (
                torch.where(
                    stage_sens_diag >= (0.5 * median_sens),
                    stage_rhs_base.detach()
                    / torch.clamp(stage_sens_diag.detach(), min=1.0e-12),
                    sol_det,
                )
                if (use_data_irls and stage_sens_diag is not None)
                else sol_det
            )
            u = (prior_sol - sky_bg) / sky_sigma
            u_dead = torch.sign(u) * torch.clamp(torch.abs(u) - 2.0, min=0.0)
            if edge_preserving or shared_source_sq is not None:
                src_sq = (
                    shared_source_sq.to(device=u.device, dtype=u.dtype)
                    if shared_source_sq is not None
                    else torch.square(u_dead)
                )
                stage_damp = torch.rsqrt(1.0 + src_sq)
                damp_weight = (
                    base_damp_weight * stage_damp
                    if base_damp_weight is not None
                    else stage_damp
                )
            if edge_preserving or shared_edge_sq is not None:
                if shared_edge_sq is not None:
                    gx_sq = shared_edge_sq[0].to(device=u.device, dtype=u.dtype)
                    gy_sq = shared_edge_sq[1].to(device=u.device, dtype=u.dtype)
                else:
                    gx_sq = (
                        torch.square(u_dead[:, 1:] - u_dead[:, :-1])
                        if u_dead.shape[1] > 1
                        else torch.zeros_like(u_dead)
                    )
                    gy_sq = (
                        torch.square(u_dead[1:, :] - u_dead[:-1, :])
                        if u_dead.shape[0] > 1
                        else torch.zeros_like(u_dead)
                    )
                stage_wx = torch.rsqrt(1.0 + gx_sq)
                stage_wy = torch.rsqrt(1.0 + gy_sq)
                wx = base_wx * stage_wx if base_wx is not None else stage_wx
                wy = base_wy * stage_wy if base_wy is not None else stage_wy
            if effective_non_negative_weight > 0.0:
                u_neg_sq = torch.square(torch.clamp(-u, min=0.0))
                neg_weight = effective_non_negative_weight * (u_neg_sq / (1.0 + u_neg_sq))
                if base_damp_weight is not None:
                    neg_weight = neg_weight * base_damp_weight

        rhs = stage_rhs_base
        if prior_reference == "initial" and effective_damping > 0.0 and has_nonzero_init:
            if damp_weight is not None:
                rhs = rhs + (effective_damping * damp_weight) * init_ref
            else:
                rhs = rhs + effective_damping * init_ref
        elif sky_bg != 0.0 and effective_damping > 0.0:
            if damp_weight is not None:
                rhs = rhs + (effective_damping * damp_weight) * sky_bg
            else:
                rhs = rhs + effective_damping * sky_bg

        if preconditioner == "diagonal_majorizer":
            assert stage_sens_diag is not None
            total_diag = stage_sens_diag.clone()
            if effective_damping > 0.0:
                total_diag = total_diag + (
                    effective_damping * damp_weight
                    if damp_weight is not None
                    else effective_damping
                )
            if neg_weight is not None:
                total_diag = total_diag + neg_weight
            if effective_smoothness_weight > 0.0:
                if wx is not None or wy is not None:
                    total_diag = total_diag + effective_smoothness_weight * _grid_weighted_laplacian_diagonal_majorizer(
                        rhs.shape,
                        wx,
                        wy,
                        device=rhs.device,
                        dtype=rhs.dtype,
                    )
                else:
                    total_diag = total_diag + 8.0 * effective_smoothness_weight
            positive = total_diag[total_diag > 0.0]
            floor = (
                max(1.0e-12, 1.0e-3 * float(torch.median(positive).item()))
                if positive.numel() > 0
                else 1.0
            )
            inv_diag = torch.where(
                total_diag > 0.0,
                1.0 / torch.clamp(total_diag, min=floor),
                torch.zeros_like(total_diag),
            )
            preconditioner_name = "diagonal_majorizer"

            def _apply_prec(r: Tensor, _inv: Tensor = inv_diag) -> Tensor:
                return r * _inv

        elif preconditioner is None or preconditioner == "none":
            preconditioner_name = "none"

            def _apply_prec(r: Tensor) -> Tensor:
                return r

        elif isinstance(preconditioner, Tensor):
            preconditioner_name = "tensor_diagonal"
            diag_t = preconditioner.to(device=rhs.device, dtype=rhs.dtype)
            if diag_t.shape != rhs.shape:
                raise ValueError("diagonal preconditioner shape must match target shape")
            inv_diag = torch.where(
                diag_t > 0.0,
                1.0 / torch.clamp(diag_t, min=1.0e-12),
                torch.zeros_like(diag_t),
            )

            def _apply_prec(r: Tensor, _inv: Tensor = inv_diag) -> Tensor:
                return r * _inv

        elif callable(preconditioner):
            preconditioner_name = "callable"
            _apply_prec = preconditioner
        else:
            raise ValueError(f"unsupported preconditioner {preconditioner!r}")

        def _normal_action(
            state: Tensor,
            _wx: Tensor | None = wx,
            _wy: Tensor | None = wy,
            _damp_w: Tensor | None = damp_weight,
            _neg_w: Tensor | None = neg_weight,
            _det_w: tuple[Tensor, ...] | None = det_weights,
        ) -> Tensor:
            nonlocal normal_ops
            normal_ops += 1
            if _det_w is None:
                image = operator_or_set.normal_operator_action(state)
            elif is_single_op:
                assert isinstance(operator_or_set, BoundObservationOperator)
                pred = operator_or_set.apply(state)
                image = operator_or_set.adjoint(
                    operator_or_set.apply_noise_model(pred * _det_w[0])
                )
            else:
                assert isinstance(operator_or_set, BoundObservationOperatorSet)
                preds = operator_or_set.apply(state)
                image = operator_or_set.weighted_adjoint(
                    tuple(
                        p_e * w_e for p_e, w_e in zip(preds, _det_w, strict=True)
                    )
                )
            if effective_damping > 0.0:
                if _damp_w is not None:
                    image = image + (effective_damping * _damp_w) * state
                else:
                    image = image + effective_damping * state
            if _neg_w is not None:
                image = image + _neg_w * state
            if effective_smoothness_weight > 0.0:
                image = image + effective_smoothness_weight * _grid_weighted_laplacian_action(
                    state, _wx, _wy
                )
            return image

        stage_has_nonzero = has_nonzero_init or stage_idx > 0
        residual = rhs - _normal_action(solution) if stage_has_nonzero else rhs.clone()
        r0_norm = float(
            torch.linalg.vector_norm(residual.detach().to(dtype=torch.float64)).item()
        )
        rhs_norm = float(
            torch.linalg.vector_norm(rhs.detach().to(dtype=torch.float64)).item()
        )
        if initial_r0_norm is None:
            initial_r0_norm = r0_norm
            history.append(r0_norm)
        elif num_stages > 1:
            history.append(r0_norm)

        z = _apply_prec(residual).to(dtype=residual.dtype)
        direction = z.clone()
        rz_old_t = torch.sum(
            residual.to(dtype=torch.float64) * z.to(dtype=torch.float64)
        )
        rz_old = float(rz_old_t.detach().item())
        ref_norm = max(rhs_norm, r0_norm, 1.0e-30)
        stage_tol = (
            float(tolerance)
            if stage_idx == num_stages - 1
            else float(tolerance) * 10.0
        )
        converged = r0_norm <= stage_tol * (ref_norm if stage_idx > 0 else 1.0)
        if num_stages == 1:
            stage_max_iters = int(iterations)
        else:
            base_iters = max(8, int(iterations) // num_stages)
            stage_max_iters = (
                int(iterations) - base_iters * (num_stages - 1)
                if stage_idx == num_stages - 1
                else base_iters
            )

        for _ in range(stage_max_iters):
            if converged or rz_old <= 0.0:
                break
            ap = _normal_action(direction)
            denom_t = torch.sum(
                direction.to(dtype=torch.float64) * ap.to(dtype=torch.float64)
            )
            denom = float(denom_t.detach().item())
            if denom <= 0.0:
                break
            if need_grad:
                alpha_t = (rz_old_t / denom_t).to(dtype=solution.dtype)
                solution = solution + alpha_t * direction
                residual = residual - alpha_t * ap
            else:
                alpha = rz_old / denom
                solution = solution + float(alpha) * direction
                residual = residual - float(alpha) * ap
            total_steps_done += 1
            r_norm = float(
                torch.linalg.vector_norm(
                    residual.detach().to(dtype=torch.float64)
                ).item()
            )
            history.append(r_norm)
            if r_norm <= stage_tol * max(r0_norm, ref_norm, 1.0e-30):
                converged = True
                break
            z = _apply_prec(residual).to(dtype=residual.dtype)
            rz_new_t = torch.sum(
                residual.to(dtype=torch.float64) * z.to(dtype=torch.float64)
            )
            rz_new = float(rz_new_t.detach().item())
            if rz_new <= 0.0:
                break
            if need_grad:
                beta_t = (rz_new_t / rz_old_t).to(dtype=direction.dtype)
                direction = z + beta_t * direction
            else:
                beta = rz_new / rz_old
                direction = z + float(beta) * direction
            rz_old_t = rz_new_t
            rz_old = rz_new

    solve_seconds = time.perf_counter() - t_solve0
    assert initial_r0_norm is not None
    final_r_norm = history[-1]
    rel_residual = (
        final_r_norm / max(initial_r0_norm, 1.0e-30)
        if initial_r0_norm > 0.0
        else 0.0
    )
    if diagnostics is not None:
        effective_fn_name = (
            getattr(data_weight_fn, "__name__", "callable")
            if callable(data_weight_fn)
            else (
                str(irls_config.weight_fn)
                if (irls_config is not None and data_weight_fn == "none")
                else str(data_weight_fn)
            )
        )
        diagnostics.update(
            {
                "solver": "irls_cg" if num_stages > 1 else "cg",
                "irls_steps": int(num_stages),
                "edge_preserving": bool(edge_preserving or shared_edge_sq is not None),
                "non_negative_weight": float(non_negative_weight),
                "data_weight_fn": effective_fn_name,
                "torchregress_irls_backend": bool(_HAS_TORCHREGRESS_IRLS),
                "detector_outlier_fraction": float(detector_outlier_fraction),
                "preconditioner": preconditioner_name,
                "damping": float(damping),
                "smoothness_weight": float(smoothness_weight),
                "relative_regularization": bool(relative_regularization),
                "effective_damping": float(effective_damping),
                "effective_smoothness_weight": float(effective_smoothness_weight),
                "effective_non_negative_weight": float(effective_non_negative_weight),
                "sky_sigma": float(sky_sigma),
                "sky_background_level": float(sky_bg),
                "prior_reference": prior_reference,
                "requested_iterations": int(iterations),
                "iterations": int(total_steps_done),
                "converged": bool(converged),
                "tolerance": float(tolerance),
                "normal_operator_applications": int(normal_ops),
                "rhs_adjoint_applications": int(rhs_applications),
                "initial_normal_residual_norm": float(initial_r0_norm),
                "final_normal_residual_norm": float(final_r_norm),
                "relative_normal_residual": float(rel_residual),
                "rhs_and_preconditioner_seconds": float(rhs_seconds),
                "solve_seconds": float(solve_seconds),
            }
        )
        if sensitivity_diag is not None:
            pos_sens = sensitivity_diag[sensitivity_diag > 0.0]
            diagnostics["grid_covered_fraction"] = float(
                (sensitivity_diag > 0.0).to(dtype=torch.float64).mean().item()
            )
            if pos_sens.numel() > 0:
                p10 = float(torch.quantile(pos_sens.double(), 0.10).item())
                p50 = float(torch.quantile(pos_sens.double(), 0.50).item())
                p90 = float(torch.quantile(pos_sens.double(), 0.90).item())
                diagnostics["grid_sensitivity_median"] = p50
                diagnostics["grid_sensitivity_p10_p90_ratio"] = p10 / max(
                    p90, 1.0e-30
                )
            else:
                diagnostics["grid_sensitivity_median"] = 0.0
                diagnostics["grid_sensitivity_p10_p90_ratio"] = 0.0

    return LinearSolveResult(
        solution=solution,
        residual_norm_history=tuple(history),
    )


def solve_joint_multiband_grid_cg(
    operators_by_band: Mapping[str, BoundObservationOperator | BoundObservationOperatorSet],
    data_by_band: Mapping[str, Tensor | Sequence[Tensor]],
    *,
    initial_solutions: Mapping[str, Tensor] | None = None,
    iterations: int = 64,
    irls_steps: int = 3,
    damping: float = 1.0e-3,
    smoothness_weight: float = 2.0e-3,
    relative_regularization: bool = True,
    prior_reference: Literal["zero", "initial"] = "zero",
    preconditioner: Literal["diagonal_majorizer", "none"] = "diagonal_majorizer",
    tolerance: float = 1.0e-7,
    edge_preserving: bool = True,
    non_negative_weight: float = 0.25,
    data_weight_fn: (
        Literal["none", "huber", "tukey", "power"] | Callable[..., Tensor]
    ) = "none",
    irls_config: IRLSConfig | None = None,
    diagnostics_by_band: dict[str, dict[str, Any]] | None = None,
) -> dict[str, LinearSolveResult]:
    """Jointly solve multi-band surface-brightness grids via coupled Vector-Charbonnier IRLS-PCG.

    In each IRLS outer stage, all bands share a common dimensionless multi-band
    structure tensor built from SNR-normalized band solutions ``u_b = x_b / sigma_sky,b``:
    ``shared_source_sq = mean_b(u_b^2)`` and ``shared_edge_sq = (mean_b((dx u_b)^2), mean_b((dy u_b)^2))``.
    High-SNR morphology in sharp short-wavelength bands preserves edges and point-source
    fluxes across all bands simultaneously while keeping each inner PCG solve decoupled per band.
    """
    if not operators_by_band:
        raise ValueError("operators_by_band must not be empty")
    if irls_steps < 1:
        raise ValueError("irls_steps must be positive")
    band_keys = tuple(operators_by_band.keys())
    if set(band_keys) != set(data_by_band.keys()):
        raise ValueError("operators_by_band and data_by_band must have matching band keys")

    num_stages = int(irls_steps)
    if num_stages == 1 or not edge_preserving or len(band_keys) == 1:
        results: dict[str, LinearSolveResult] = {}
        for band in band_keys:
            band_diag: dict[str, Any] = {}
            init_b = initial_solutions.get(band) if initial_solutions is not None else None
            results[band] = solve_linear_band_grid_cg(
                operators_by_band[band],
                data_by_band[band],
                initial_solution=init_b,
                iterations=iterations,
                damping=damping,
                smoothness_weight=smoothness_weight,
                relative_regularization=relative_regularization,
                prior_reference=prior_reference,
                preconditioner=preconditioner,
                tolerance=tolerance,
                irls_steps=num_stages,
                edge_preserving=edge_preserving,
                non_negative_weight=non_negative_weight,
                data_weight_fn=data_weight_fn,
                irls_config=irls_config,
                diagnostics=band_diag,
            )
            if diagnostics_by_band is not None:
                diagnostics_by_band[band] = band_diag
        return results

    current_solutions: dict[str, Tensor] = {}
    if initial_solutions is not None:
        for band in band_keys:
            if band in initial_solutions:
                current_solutions[band] = initial_solutions[band]

    sky_sigmas: dict[str, float] = {}
    sky_bgs: dict[str, float] = {}
    history_by_band: dict[str, list[float]] = {band: [] for band in band_keys}
    per_band_diags: dict[str, dict[str, Any]] = {band: {} for band in band_keys}
    total_iters_by_band: dict[str, int] = {band: 0 for band in band_keys}
    total_normal_ops_by_band: dict[str, int] = {band: 0 for band in band_keys}
    total_rhs_ops_by_band: dict[str, int] = {band: 0 for band in band_keys}
    total_rhs_sec_by_band: dict[str, float] = {band: 0.0 for band in band_keys}
    total_solve_sec_by_band: dict[str, float] = {band: 0.0 for band in band_keys}
    init_r0_by_band: dict[str, float] = {}

    base_iters = max(12, int(iterations) // num_stages)
    for stage_idx in range(num_stages):
        stage_iters = (
            max(16, int(iterations) - base_iters * (num_stages - 1))
            if stage_idx == num_stages - 1
            else base_iters
        )
        stage_tol = (
            float(tolerance)
            if stage_idx == num_stages - 1
            else max(float(tolerance) * 10.0, 1.0e-5)
        )
        shared_edge_sq: tuple[Tensor, Tensor] | None = None
        shared_source_sq: Tensor | None = None
        if stage_idx > 0 and current_solutions:
            u_list: list[Tensor] = []
            for band in band_keys:
                sol_b = current_solutions[band].detach()
                sigma_b = max(sky_sigmas.get(band, 1.0), 1.0e-30)
                bg_b = sky_bgs.get(band, 0.0)
                u_b = (sol_b - bg_b) / sigma_b
                u_b_dead = torch.sign(u_b) * torch.clamp(torch.abs(u_b) - 2.0, min=0.0)
                u_list.append(u_b_dead)
            u_stack = torch.stack(u_list, dim=0)
            shared_source_sq = torch.mean(torch.square(u_stack), dim=0)
            gx_sq = (
                torch.mean(torch.square(u_stack[:, :, 1:] - u_stack[:, :, :-1]), dim=0)
                if u_stack.shape[2] > 1
                else torch.zeros_like(shared_source_sq)
            )
            gy_sq = (
                torch.mean(torch.square(u_stack[:, 1:, :] - u_stack[:, :-1, :]), dim=0)
                if u_stack.shape[1] > 1
                else torch.zeros_like(shared_source_sq)
            )
            shared_edge_sq = (gx_sq, gy_sq)

        is_warmup = bool(stage_idx == 1 and num_stages >= 3)
        req_fn = (
            irls_config.weight_fn
            if (irls_config is not None and data_weight_fn == "none")
            else data_weight_fn
        )
        stage_weight_fn: Literal["none", "huber", "tukey", "power"] | Callable[..., Tensor] = (
            "none"
            if stage_idx == 0
            else ("huber" if (is_warmup and req_fn == "tukey") else data_weight_fn)
        )
        stage_irls_cfg = (
            None
            if stage_idx == 0
            else (
                replace(irls_config, weight_fn="huber")
                if (
                    is_warmup
                    and irls_config is not None
                    and irls_config.weight_fn == "tukey"
                )
                else irls_config
            )
        )
        for band in band_keys:
            stage_diag: dict[str, Any] = {}
            init_b = current_solutions.get(band)
            stage_prior_ref: Literal["zero", "initial"] = (
                prior_reference if stage_idx == 0 else "zero"
            )
            res_b = solve_linear_band_grid_cg(
                operators_by_band[band],
                data_by_band[band],
                initial_solution=init_b,
                iterations=stage_iters,
                damping=damping,
                smoothness_weight=smoothness_weight,
                relative_regularization=relative_regularization,
                prior_reference=stage_prior_ref,
                preconditioner=preconditioner,
                tolerance=stage_tol,
                irls_steps=1,
                edge_preserving=edge_preserving and stage_idx > 0,
                non_negative_weight=non_negative_weight if stage_idx > 0 else 0.0,
                data_weight_fn=stage_weight_fn,
                irls_config=stage_irls_cfg,
                shared_edge_sq=shared_edge_sq,
                shared_source_sq=shared_source_sq,
                diagnostics=stage_diag,
            )
            current_solutions[band] = res_b.solution
            sky_sigmas[band] = float(stage_diag.get("sky_sigma", 1.0))
            sky_bgs[band] = float(stage_diag.get("sky_background_level", 0.0))
            history_by_band[band].extend(res_b.residual_norm_history)
            if band not in init_r0_by_band:
                init_r0_by_band[band] = float(
                    stage_diag.get("initial_normal_residual_norm", 0.0)
                )
            total_iters_by_band[band] += int(stage_diag.get("iterations", 0))
            total_normal_ops_by_band[band] += int(
                stage_diag.get("normal_operator_applications", 0)
            )
            total_rhs_ops_by_band[band] += int(
                stage_diag.get("rhs_adjoint_applications", 0)
            )
            total_rhs_sec_by_band[band] += float(
                stage_diag.get("rhs_and_preconditioner_seconds", 0.0)
            )
            total_solve_sec_by_band[band] += float(
                stage_diag.get("solve_seconds", 0.0)
            )
            per_band_diags[band] = stage_diag

    final_results: dict[str, LinearSolveResult] = {}
    for band in band_keys:
        hist = tuple(history_by_band[band])
        r0_b = init_r0_by_band.get(band, hist[0] if hist else 0.0)
        final_r_b = hist[-1] if hist else 0.0
        rel_r_b = final_r_b / max(r0_b, 1.0e-30) if r0_b > 0.0 else 0.0
        diag_b = per_band_diags[band]
        diag_b.update(
            {
                "solver": "joint_multiband_irls_cg",
                "irls_steps": int(num_stages),
                "edge_preserving": bool(edge_preserving),
                "non_negative_weight": float(non_negative_weight),
                "prior_reference": prior_reference,
                "requested_iterations": int(iterations),
                "iterations": int(total_iters_by_band[band]),
                "normal_operator_applications": int(total_normal_ops_by_band[band]),
                "rhs_adjoint_applications": int(total_rhs_ops_by_band[band]),
                "initial_normal_residual_norm": float(r0_b),
                "final_normal_residual_norm": float(final_r_b),
                "relative_normal_residual": float(rel_r_b),
                "rhs_and_preconditioner_seconds": float(total_rhs_sec_by_band[band]),
                "solve_seconds": float(total_solve_sec_by_band[band]),
            }
        )
        if diagnostics_by_band is not None:
            diagnostics_by_band[band] = diag_b
        final_results[band] = LinearSolveResult(
            solution=current_solutions[band],
            residual_norm_history=hist,
        )
    return final_results


def solve_linear_band_grid_lsqr(
    operator_or_set: BoundObservationOperator | BoundObservationOperatorSet,
    data: Tensor | Sequence[Tensor],
    *,
    initial_solution: Tensor | None = None,
    iterations: int = 32,
    damping: float = 0.0,
    prior_reference: Literal["zero", "initial"] = "zero",
    atol: float = 1.0e-12,
    btol: float = 1.0e-12,
    tolerance: float | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> LinearSolveResult:
    """Solve weighted linear least squares via Golub-Kahan LSQR without squaring cond(A).

    Minimizes ``||N^{-1/2} (A x - y)||_2^2 + damping * ||x - x_ref||_2^2``. Unlike
    normal-equation CG, LSQR remains well-posed for rank-deficient systems even when
    ``damping == 0.0``, converging to the minimum-norm least-squares solution.
    """
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if not math.isfinite(damping) or damping < 0.0:
        raise ValueError("damping must be finite and non-negative")
    if prior_reference not in ("zero", "initial"):
        raise ValueError("prior_reference must be 'zero' or 'initial'")
    if tolerance is not None:
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("tolerance must be finite and non-negative")
        atol = float(tolerance)
        btol = float(tolerance)
    t_setup0 = time.perf_counter()
    is_set = isinstance(operator_or_set, BoundObservationOperatorSet)
    operators: tuple[BoundObservationOperator, ...] = (
        operator_or_set.operators if is_set else (operator_or_set,)
    )
    data_tuple: tuple[Tensor, ...] = (
        tuple(torch.as_tensor(d) for d in data)
        if isinstance(data, Sequence) and not isinstance(data, Tensor)
        else (torch.as_tensor(data),)
    )
    if len(data_tuple) != len(operators):
        raise ValueError("data count must match operator count")

    # Probe N^{-1} diagonal weights via unit tensor application.
    sqrt_weights: list[Tensor] = []
    for op, datum in zip(operators, data_tuple, strict=True):
        ones = torch.ones_like(datum)
        w = torch.clamp(op.apply_noise_model(ones), min=0.0)
        sqrt_weights.append(torch.sqrt(w))

    state_shape = tuple(int(v) for v in operators[0].target.shape)
    dtype = data_tuple[0].dtype
    device = data_tuple[0].device
    x0 = (
        torch.zeros(state_shape, dtype=dtype, device=device)
        if initial_solution is None
        else torch.as_tensor(initial_solution, dtype=dtype, device=device).clone()
    )
    if x0.shape != state_shape:
        raise ValueError("initial_solution shape must match operator target shape")

    sqrt_damp = math.sqrt(float(damping))
    rhs_seconds = time.perf_counter() - t_setup0
    forward_adjoint_pairs = 0

    def _forward_augmented(v: Tensor) -> tuple[tuple[Tensor, ...], Tensor]:
        if is_set:
            preds = operator_or_set.apply(v)
        else:
            preds = (operators[0].apply(v),)
        weighted = tuple(
            sw * p.to(dtype=sw.dtype)
            for sw, p in zip(sqrt_weights, preds, strict=True)
        )
        reg = v * sqrt_damp if sqrt_damp > 0.0 else torch.zeros_like(v)
        return weighted, reg

    def _adjoint_augmented(
        u_obs: tuple[Tensor, ...],
        u_reg: Tensor,
    ) -> Tensor:
        scaled = tuple(
            sw * u.to(dtype=sw.dtype)
            for sw, u in zip(sqrt_weights, u_obs, strict=True)
        )
        adj = (
            operator_or_set.adjoint(scaled)
            if is_set
            else operators[0].adjoint(scaled[0])
        )
        if sqrt_damp > 0.0:
            adj = adj + sqrt_damp * u_reg
        return adj

    def _obs_norm(u_obs: tuple[Tensor, ...], u_reg: Tensor) -> float:
        sq = sum(torch.sum(u.to(dtype=torch.float64).square()) for u in u_obs)
        if sqrt_damp > 0.0:
            sq = sq + torch.sum(u_reg.to(dtype=torch.float64).square())
        return float(torch.sqrt(sq).item())

    def _scale_obs(
        u_obs: tuple[Tensor, ...],
        u_reg: Tensor,
        scale: float,
    ) -> tuple[tuple[Tensor, ...], Tensor]:
        if scale == 0.0:
            return tuple(torch.zeros_like(u) for u in u_obs), torch.zeros_like(u_reg)
        inv = 1.0 / scale
        return tuple(u * inv for u in u_obs), u_reg * inv

    t_solve0 = time.perf_counter()
    pred0 = (
        operator_or_set.apply(x0) if is_set else (operators[0].apply(x0),)
    )
    u_obs = tuple(
        sw * (d.to(dtype=p.dtype) - p)
        for sw, d, p in zip(sqrt_weights, data_tuple, pred0, strict=True)
    )
    u_reg = (
        -sqrt_damp * x0
        if (sqrt_damp > 0.0 and prior_reference == "zero")
        else torch.zeros_like(x0)
    )
    beta = _obs_norm(u_obs, u_reg)
    b_norm = beta
    u_obs, u_reg = _scale_obs(u_obs, u_reg, beta)

    v = _adjoint_augmented(u_obs, u_reg)
    forward_adjoint_pairs += 1
    alpha = float(torch.linalg.vector_norm(v.to(dtype=torch.float64)).item())
    if alpha > 0.0:
        v = v / alpha
    w_vec = v.clone()
    phi_bar = beta
    rho_bar = alpha
    x = x0.clone()
    history: list[float] = []
    steps_done = 0
    converged = beta == 0.0 or alpha == 0.0

    for _ in range(int(iterations)):
        history.append(float(phi_bar))
        if beta == 0.0 or alpha == 0.0:
            converged = True
            break
        av_obs, av_reg = _forward_augmented(v)
        u_obs = tuple(
            av - alpha * u for av, u in zip(av_obs, u_obs, strict=True)
        )
        u_reg = av_reg - alpha * u_reg
        beta = _obs_norm(u_obs, u_reg)
        u_obs, u_reg = _scale_obs(u_obs, u_reg, beta)

        at_u = _adjoint_augmented(u_obs, u_reg)
        forward_adjoint_pairs += 1
        v = at_u - beta * v
        alpha = float(torch.linalg.vector_norm(v.to(dtype=torch.float64)).item())
        if alpha > 0.0:
            v = v / alpha

        rho = math.hypot(rho_bar, beta)
        if rho == 0.0:
            break
        cs = rho_bar / rho
        sn = beta / rho
        theta = sn * alpha
        rho_bar = -cs * alpha
        phi = cs * phi_bar
        phi_bar = abs(sn * phi_bar)

        x = x + (phi / rho) * w_vec
        w_vec = v - (theta / rho) * w_vec
        steps_done += 1

        if phi_bar <= btol * b_norm + atol * alpha:
            history.append(float(phi_bar))
            converged = True
            break

    solve_seconds = time.perf_counter() - t_solve0
    if diagnostics is not None:
        r0 = float(history[0]) if history else float(b_norm)
        rf = float(history[-1]) if history else float(phi_bar)
        diagnostics.update(
            {
                "solver": "lsqr",
                "preconditioner": "none",
                "damping": float(damping),
                "smoothness_weight": 0.0,
                "prior_reference": prior_reference,
                "requested_iterations": int(iterations),
                "iterations": int(steps_done),
                "converged": bool(converged),
                "tolerance": float(btol),
                "normal_operator_applications": int(forward_adjoint_pairs),
                "rhs_adjoint_applications": 0,
                "initial_normal_residual_norm": r0,
                "final_normal_residual_norm": rf,
                "relative_normal_residual": rf / max(r0, 1.0e-30) if r0 > 0.0 else 0.0,
                "rhs_and_preconditioner_seconds": float(rhs_seconds),
                "solve_seconds": float(solve_seconds),
            }
        )

    return LinearSolveResult(
        solution=x,
        residual_norm_history=tuple(history),
    )


@dataclass(frozen=True)
class StationaryCalibratedBandObservation:
    """Verified stationary calibration contract ready for linear grid binding."""

    observation: FieldObservation
    prepared_data: Tensor
    calibrated_background: Tensor
    effective_psf_kernel: Tensor
    provenance: dict[str, Any]


def _extract_stationary_effective_psf(
    observation: FieldObservation,
    operator_bundle: ObservationOperatorBundle,
    *,
    forward_context: ForwardContext | None,
) -> tuple[Tensor, str]:
    psf_op = operator_bundle.psf
    if psf_op is None:
        raise ValueError("stationary calibration adapter requires a calibrated PSF operator")
    if not psf_op.frozen:
        raise ValueError("stationary calibration adapter requires a frozen PSF operator")
    if psf_op.pixel_response_included is not True:
        raise ValueError(
            "stationary calibration adapter requires PSF pixel_response_included=True"
        )
    provenance = psf_op.pixel_response_provenance
    if not isinstance(provenance, str) or not provenance.strip():
        raise ValueError(
            "stationary calibration adapter requires non-empty PSF pixel_response_provenance"
        )
    if isinstance(psf_op, ExplicitPSFOp):
        spec = observation.psf_spec
        if spec is None or spec.mode != "kernel2d" or spec.kernel2d is None:
            raise ValueError(
                "ExplicitPSFOp stationary adapter requires observation.psf_spec kernel2d"
            )
        if not spec.frozen or spec.pixel_response_included is not True:
            raise ValueError(
                "ExplicitPSFOp observation.psf_spec must be frozen with pixel_response_included=True"
            )
        kernel = spec.kernel2d.detach().to(
            device=observation.image.device, dtype=observation.image.dtype
        )
    elif isinstance(psf_op, LearnedPSFOp):
        model = psf_op.model
        if not isinstance(model, ConditionedKernelPSF):
            raise TypeError(
                "stationary LearnedPSFOp must wrap a ConditionedKernelPSF model"
            )
        if any(p.requires_grad for p in model.parameters()):
            raise ValueError("stationary ConditionedKernelPSF parameters must be frozen")
        height, width = map(int, observation.image.shape)
        param = next(model.parameters())
        probe_x = torch.tensor(
            [0.0, float(max(width - 1, 0)), 0.5 * float(max(width - 1, 0))],
            device=param.device,
            dtype=param.dtype,
        ) + float(observation.detector_x0_pix)
        probe_y = torch.tensor(
            [0.0, float(max(height - 1, 0)), 0.5 * float(max(height - 1, 0))],
            device=param.device,
            dtype=param.dtype,
        ) + float(observation.detector_y0_pix)
        wl = float(
            observation.wavelength
            if observation.wavelength is not None
            else (
                0.5 * (float(observation.spectral_min) + float(observation.spectral_max))
                if observation.spectral_min is not None
                and observation.spectral_max is not None
                else 0.0
            )
        )
        time_mjd = (
            None
            if forward_context is None or forward_context.time_mjd is None
            else float(forward_context.time_mjd)
        )
        with torch.no_grad():
            kernels = model.effective_kernels(probe_x, probe_y, wl, time_mjd)
        ref_kernel = kernels[0]
        if not torch.allclose(kernels, ref_kernel.expand_as(kernels), rtol=1.0e-6, atol=1.0e-7):
            raise ValueError(
                "calibrated PSF varies across detector coordinates; stationary adapter requires a coordinate-invariant PSF"
            )
        kernel = ref_kernel.detach().to(
            device=observation.image.device, dtype=observation.image.dtype
        )
    else:
        raise TypeError(f"unsupported PSF operator type {type(psf_op)!r}")

    if (
        kernel.ndim != 2
        or kernel.shape[0] != kernel.shape[1]
        or int(kernel.shape[0]) % 2 != 1
    ):
        raise ValueError("stationary effective PSF kernel must be odd square 2D")
    if not bool(torch.isfinite(kernel).all()):
        raise ValueError("stationary effective PSF kernel must be finite")
    if bool((kernel < -1.0e-7).any()):
        raise ValueError("stationary effective PSF kernel must be non-negative")
    kernel = torch.clamp(kernel, min=0.0)
    kernel_sum = float(kernel.sum().item())
    if not math.isfinite(kernel_sum) or kernel_sum <= 0.0:
        raise ValueError("stationary effective PSF kernel must have positive sum")
    kernel = kernel / kernel.sum()
    return kernel, provenance


def _extract_stationary_calibrated_background(
    observation: FieldObservation,
    operator_bundle: ObservationOperatorBundle,
    *,
    forward_context: ForwardContext | None,
    background_gauge_policy: str,
) -> tuple[Tensor, float]:
    if not isinstance(background_gauge_policy, str) or not background_gauge_policy.strip():
        raise ValueError("background_gauge_policy must be an explicit non-empty string")
    bg_op = operator_bundle.background
    if bg_op is None:
        return torch.zeros_like(observation.image), 0.0
    if not isinstance(bg_op, LearnedBackgroundOp):
        raise TypeError(
            "stationary calibration adapter requires LearnedBackgroundOp or None in bundle.background"
        )
    if bg_op.coordinate_frame != "detector":
        raise ValueError(
            "stationary calibrated background must use detector coordinate_frame"
        )
    model = bg_op.model
    if not isinstance(model, ConditionedBackgroundField):
        raise TypeError(
            "stationary LearnedBackgroundOp must wrap a ConditionedBackgroundField"
        )
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError("stationary ConditionedBackgroundField parameters must be frozen")
    height, width = map(int, observation.image.shape)
    param = next(model.parameters())
    probe_x = torch.tensor(
        [0.0, float(max(width - 1, 0)), 0.5 * float(max(width - 1, 0))],
        device=param.device,
        dtype=param.dtype,
    ) + float(observation.detector_x0_pix)
    probe_y = torch.tensor(
        [0.0, float(max(height - 1, 0)), 0.5 * float(max(height - 1, 0))],
        device=param.device,
        dtype=param.dtype,
    ) + float(observation.detector_y0_pix)
    wl = float(
        observation.wavelength
        if observation.wavelength is not None
        else (
            0.5 * (float(observation.spectral_min) + float(observation.spectral_max))
            if observation.spectral_min is not None
            and observation.spectral_max is not None
            else 0.0
        )
    )
    time_mjd = (
        None
        if forward_context is None or forward_context.time_mjd is None
        else float(forward_context.time_mjd)
    )
    with torch.no_grad():
        values = model(probe_x, probe_y, wl, time_mjd)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("stationary calibrated background evaluation must be finite")
    scalar_val = float(values[0].item())
    if not torch.allclose(values, values[:1].expand_as(values), rtol=1.0e-6, atol=1.0e-7):
        raise ValueError(
            "calibrated background varies across detector coordinates; stationary adapter requires a constant pedestal"
        )
    bg_map = torch.full_like(observation.image, scalar_val)
    return bg_map, scalar_val


def adapt_stationary_calibrated_band_observation(
    observation: FieldObservation,
    operator_bundle: ObservationOperatorBundle,
    *,
    forward_context: ForwardContext | None = None,
    expected_value_unit: str = "MJy/sr",
    background_gauge_policy: str = "training_mean",
    provenance: Mapping[str, Any] | None = None,
) -> StationaryCalibratedBandObservation:
    """Convert a stationary calibrated observation into the fixed linear-grid contract."""
    if observation.value_semantics != "surface_brightness":
        raise ValueError(
            "stationary calibration adapter requires surface_brightness semantics"
        )
    unit = observation.value_unit
    if not isinstance(unit, str) or not unit.strip():
        raise ValueError(
            "stationary calibration adapter requires an explicit observation.value_unit"
        )
    if unit.strip().lower() in _RAW_COUNT_UNITS:
        raise ValueError(
            f"raw detector unit {unit!r} cannot be bound as calibrated surface brightness"
        )
    if expected_value_unit and unit.strip() != expected_value_unit:
        raise ValueError(
            f"observation.value_unit {unit!r} does not match expected {expected_value_unit!r}"
        )
    if not isinstance(observation.band_name, str) or not observation.band_name.strip():
        raise ValueError("stationary calibration adapter requires an explicit band_name")
    if (
        float(getattr(observation, "photometric_scale", 1.0)) != 1.0
        or observation.gain_electrons_per_adu is not None
        or observation.exposure_time_s is not None
        or observation.collecting_area_m2 is not None
        or operator_bundle.radiometry is not None
    ):
        raise ValueError(
            "unapplied radiometric conversion terms must be resolved before binding"
        )
    has_bandpass = (
        observation.passband is not None
        or (
            observation.transmission_wavelength is not None
            and observation.transmission_throughput is not None
        )
        or (
            observation.spectral_min is not None
            and observation.spectral_max is not None
        )
        or observation.wavelength is not None
    )
    if not has_bandpass:
        raise ValueError(
            "stationary calibration adapter requires declared band spectral metadata"
        )

    kernel, psf_provenance = _extract_stationary_effective_psf(
        observation,
        operator_bundle,
        forward_context=forward_context,
    )
    bg_map, bg_scalar = _extract_stationary_calibrated_background(
        observation,
        operator_bundle,
        forward_context=forward_context,
        background_gauge_policy=background_gauge_policy,
    )
    psf_spec = PSFSpec(
        mode="kernel2d",
        kernel2d=kernel,
        frozen=True,
        pixel_response_included=True,
    )
    adapted_observation = replace(
        observation,
        psf_spec=psf_spec,
        psf_sigma_pix=0.0,
    )
    prepared_data = prepare_linear_band_grid_observation_data(
        adapted_observation,
        calibrated_background=bg_map,
    )
    prov: dict[str, Any] = {
        "band_name": adapted_observation.band_name,
        "value_semantics": adapted_observation.value_semantics,
        "value_unit": adapted_observation.value_unit,
        "pixel_response_included": True,
        "pixel_response_provenance": psf_provenance,
        "kernel_shape": tuple(int(v) for v in kernel.shape),
        "background_gauge_policy": background_gauge_policy,
        "calibrated_background_scalar": bg_scalar,
    }
    if provenance is not None:
        for k, v in provenance.items():
            prov[str(k)] = v
    return StationaryCalibratedBandObservation(
        observation=adapted_observation,
        prepared_data=prepared_data,
        calibrated_background=bg_map,
        effective_psf_kernel=kernel,
        provenance=prov,
    )


def _build_stationary_model_from_split(
    arrays: dict[str, Tensor],
    *,
    role: str,
) -> nn.Module:
    def _center_scale(values: Tensor) -> tuple[float, float]:
        vals = values.to(dtype=torch.float64)
        return float(vals.mean()), max(float(vals.max() - vals.min()), 1.0)

    x_center, x_scale = _center_scale(arrays["detector_x"])
    y_center, y_scale = _center_scale(arrays["detector_y"])
    wl_center, wl_scale = _center_scale(arrays["wavelength"])
    t_center, t_scale = _center_scale(arrays["time_mjd"])
    encoder = SpatioSpectralTemporalEncoder(
        spatial_center=(x_center, y_center),
        spatial_scale=(x_scale, y_scale),
        wavelength_center=wl_center,
        spectral_scale=wl_scale,
        time_center_mjd=t_center,
        temporal_scale_days=t_scale,
        spatial_frequencies=0,
        spectral_frequencies=0,
        temporal_frequencies=0,
    )
    if role == "psf":
        kernel_size = int(arrays["target_kernels"].shape[-1])
        model: nn.Module = ConditionedKernelPSF(
            encoder,
            n_basis=1,
            kernel_size=kernel_size,
            init_fwhm_pix=2.0,
            hidden_dim=32,
        )
        for p in model.parameters():
            p.requires_grad_(False)
        return model
    if role == "background":
        model = ConditionedBackgroundField(encoder, hidden_dim=32, hidden_layers=2)
        for p in model.parameters():
            p.requires_grad_(False)
        return model
    raise ValueError(f"unsupported calibration role {role!r}")


def load_stationary_calibrated_band_observations(
    observations: Sequence[FieldObservation],
    contexts: Sequence[ForwardContext],
    *,
    manifest_path: Path,
    fit_report_path: Path,
    intensity_scale: float = 1.0,
    background_gauge_policy: str = "training_mean",
    expected_value_unit: str = "MJy/sr",
) -> tuple[tuple[StationaryCalibratedBandObservation, ...], dict[str, Any]]:
    """Load, hash-verify, and adapt frozen stationary SMACS calibration checkpoints."""
    obs_tuple = tuple(observations)
    ctx_tuple = tuple(contexts)
    if not obs_tuple or len(obs_tuple) != len(ctx_tuple):
        raise ValueError(
            "observations and contexts must be non-empty and have matching lengths"
        )
    if not math.isfinite(intensity_scale) or intensity_scale <= 0.0:
        raise ValueError("intensity_scale must be finite and positive")
    if background_gauge_policy not in {"training_mean", "unshifted_stationary"}:
        raise ValueError(
            "background_gauge_policy must be 'training_mean' or 'unshifted_stationary'"
        )
    manifest_resolved = Path(manifest_path).resolve()
    report_resolved = Path(fit_report_path).resolve()
    manifest_sha = _sha256_file(manifest_resolved)
    report = json.loads(report_resolved.read_text(encoding="utf-8"))
    if report.get("manifest_sha256") != manifest_sha:
        raise ValueError("operator fit report does not match the calibration manifest")
    scope = str(report.get("calibration_scope", "detector"))
    if scope not in {"detector", "exposure"}:
        raise ValueError(f"unsupported calibration scope {scope!r}")

    indexed: dict[tuple[str, str | None, str | None, str, str], dict[str, Any]] = {}
    for group in report.get("groups", ()):
        key = (
            str(group["detector"]).upper(),
            group.get("exposure_id") if scope == "exposure" else None,
            str(group.get("band", "")).upper() if scope == "exposure" else None,
            str(group["role"]),
            str(group["family"]),
        )
        if key in indexed:
            raise ValueError(f"duplicate calibrated operator group {key}")
        indexed[key] = group

    def _context_key(ctx: ForwardContext) -> tuple[str, str | None, str | None]:
        detector = str(ctx.detector or "").strip().upper()
        if not detector:
            raise ValueError("ForwardContext requires a non-empty detector name")
        if scope == "detector":
            return detector, None, None
        exposure_id = ctx.extra.get("calibration_exposure_id")
        band = str(ctx.survey or "").strip().upper()
        if not isinstance(exposure_id, str) or not exposure_id or not band:
            raise ValueError(
                "exposure calibration requires ForwardContext.survey and calibration_exposure_id"
            )
        return detector, exposure_id, band

    unique_keys = sorted({_context_key(ctx) for ctx in ctx_tuple})
    loaded_models: dict[tuple[str, str | None, str | None], dict[str, nn.Module]] = {}
    checkpoint_audits: dict[tuple[str, str | None, str | None], dict[str, Any]] = {}

    for key in unique_keys:
        detector, exposure_id, band = key
        loaded_models[key] = {}
        checkpoint_audits[key] = {}
        for role in ("background", "psf"):
            group = indexed.get((detector, exposure_id, band, role, "stationary"))
            if group is None:
                raise ValueError(
                    f"fit report has no stationary {key}/{role} calibration group"
                )
            ckpt_path = Path(group["checkpoint"])
            if not ckpt_path.is_absolute():
                ckpt_path = (report_resolved.parent / ckpt_path).resolve()
            ckpt_sha = _sha256_file(ckpt_path)
            if ckpt_sha != group.get("checkpoint_sha256"):
                raise ValueError(f"{detector}/{role} checkpoint hash mismatch")
            arrays = load_operator_calibration_split(
                manifest_resolved,
                split="train",
                role=role,
                detector=detector,
                exposure_id=exposure_id,
                band=band,
                verify_hashes=True,
            )
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            if (
                checkpoint.get("schema_version") != 2
                or str(checkpoint.get("detector", "")).upper() != detector
                or checkpoint.get("role") != role
                or checkpoint.get("family") != "stationary"
                or (
                    scope == "exposure"
                    and (
                        checkpoint.get("exposure_id") != exposure_id
                        or str(checkpoint.get("band", "")).upper() != band
                        or checkpoint.get("calibration_scope") != "exposure"
                    )
                )
            ):
                raise ValueError(f"invalid {detector}/{role} checkpoint metadata")
            model = _build_stationary_model_from_split(arrays, role=role)
            model.load_state_dict(checkpoint["state_dict"])
            model.requires_grad_(False)
            model.eval()
            if role == "psf":
                pri = checkpoint.get("pixel_response_included")
                prp = checkpoint.get("pixel_response_provenance")
                if (
                    pri is not True
                    or not isinstance(prp, str)
                    or not prp.strip()
                    or group.get("pixel_response_included") != pri
                    or group.get("pixel_response_provenance") != prp
                ):
                    raise ValueError(f"invalid {detector}/psf pixel-response metadata")
                model._torchsky_pixel_response_included = True
                model._torchsky_pixel_response_provenance = prp
            else:
                assert isinstance(model, ConditionedBackgroundField)
                output_layer = model.network[-1]
                assert isinstance(output_layer, nn.Linear)
                with torch.no_grad():
                    output_layer.weight.div_(float(intensity_scale))
                    output_layer.bias.div_(float(intensity_scale))
            loaded_models[key][role] = model
            checkpoint_audits[key][f"{role}_checkpoint"] = str(ckpt_path)
            checkpoint_audits[key][f"{role}_checkpoint_sha256"] = ckpt_sha

    gauge_audit: dict[str, Any] = {"policy": background_gauge_policy}
    if background_gauge_policy == "training_mean":
        if scope != "exposure":
            raise ValueError(
                "training_mean background gauge requires exposure-scoped calibration"
            )
        train_keys = sorted(
            {
                _context_key(ctx)
                for ctx in ctx_tuple
                if ctx.extra.get("split") == "train"
            }
        )
        if not train_keys:
            raise ValueError(
                "training_mean background gauge requires at least one train exposure"
            )
        band_accum: dict[str, dict[str, float]] = {}
        with torch.no_grad():
            for key in train_keys:
                detector, exposure_id, band = key
                assert band is not None
                bg_model = loaded_models[key]["background"]
                param = next(bg_model.parameters())
                arrays = load_operator_calibration_split(
                    manifest_resolved,
                    split="train",
                    role="background",
                    detector=detector,
                    exposure_id=exposure_id,
                    band=band,
                    verify_hashes=True,
                )
                coords = [
                    arrays[name].to(device=param.device, dtype=param.dtype)
                    for name in ("detector_x", "detector_y", "wavelength", "time_mjd")
                ]
                vals = bg_model(*coords).detach().cpu().double()
                weights = arrays["sample_weight"].detach().cpu().double()
                if (
                    vals.shape != weights.shape
                    or not bool(torch.isfinite(vals).all())
                    or not bool(torch.isfinite(weights).all())
                    or bool((weights < 0.0).any())
                    or float(weights.sum()) <= 0.0
                ):
                    raise ValueError(
                        "training_mean background gauge requires finite positive-weight evidence"
                    )
                entry = band_accum.setdefault(
                    band, {"weighted_sum": 0.0, "weight_sum": 0.0}
                )
                entry["weighted_sum"] += float((vals * weights).sum().item())
                entry["weight_sum"] += float(weights.sum().item())
            band_offsets: dict[str, float] = {}
            for band, entry in band_accum.items():
                band_offsets[band] = entry["weighted_sum"] / entry["weight_sum"]
            for key, ops in loaded_models.items():
                band = key[2]
                if band is None or band not in band_offsets:
                    raise ValueError(
                        f"training_mean background gauge missing training evidence for band {band!r}"
                    )
                bg_model = ops["background"]
                assert isinstance(bg_model, ConditionedBackgroundField)
                out_layer = bg_model.network[-1]
                assert isinstance(out_layer, nn.Linear)
                out_layer.bias.sub_(band_offsets[band])
            gauge_audit["band_offsets_normalized"] = band_offsets

    adapted_list: list[StationaryCalibratedBandObservation] = []
    for obs, ctx in zip(obs_tuple, ctx_tuple, strict=True):
        passband_path = ctx.extra.get("measured_passband_path")
        passband_sha = ctx.extra.get("measured_passband_sha256")
        if passband_path is not None or passband_sha is not None:
            if not passband_path or not passband_sha:
                raise ValueError(
                    "measured passband path and sha256 must both be provided when declared"
                )
            actual_pb_sha = _sha256_file(Path(str(passband_path)))
            if actual_pb_sha != str(passband_sha):
                raise ValueError("measured passband SHA-256 hash mismatch")
        key = _context_key(ctx)
        psf_model = loaded_models[key]["psf"]
        bg_model = loaded_models[key]["background"]
        bundle = field_observation_operator_bundle(obs).with_updates(
            psf=LearnedPSFOp(
                psf_model,
                frozen=True,
                pixel_response_included= bool(
                    getattr(psf_model, "_torchsky_pixel_response_included", False)
                ),
                pixel_response_provenance=str(
                    getattr(psf_model, "_torchsky_pixel_response_provenance", "")
                ),
            ),
            background=LearnedBackgroundOp(bg_model, coordinate_frame="detector"),
        )
        adapted = adapt_stationary_calibrated_band_observation(
            obs,
            bundle,
            forward_context=ctx,
            expected_value_unit=expected_value_unit,
            background_gauge_policy=background_gauge_policy,
            provenance={
                "manifest_sha256": manifest_sha,
                **checkpoint_audits[key],
            },
        )
        adapted_list.append(adapted)

    return tuple(adapted_list), {
        "calibration_scope": scope,
        "manifest": str(manifest_resolved),
        "manifest_sha256": manifest_sha,
        "fit_report": str(report_resolved),
        "intensity_scale": float(intensity_scale),
        "background_gauge": gauge_audit,
    }


__all__ = [
    "IRLSConfig",
    "LinearBandGridField",
    "StationaryCalibratedBandObservation",
    "adapt_stationary_calibrated_band_observation",
    "bind_linear_band_grid_observation_operator",
    "bind_linear_band_grid_observations_to_operator_set",
    "linear_band_grid_diagonal_majorizer",
    "load_stationary_calibrated_band_observations",
    "prepare_linear_band_grid_observation_data",
    "solve_joint_multiband_grid_cg",
    "solve_linear_band_grid_cg",
    "solve_linear_band_grid_lsqr",
]

