"""Linear band-grid operator binding, grid regularizers, and IRLS weights for explicit-grid inverse solves."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


if TYPE_CHECKING:
    pass

from ..scenes import DiscreteTargetImage, SceneRepresentation
from ..solvers import (
    BoundObservationOperator,
    BoundObservationOperatorSet,
)


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
            max_ky = max(int(k.shape[2]) if k is not None else 1 for k in raw_kernels)
            max_kx = max(int(k.shape[3]) if k is not None else 1 for k in raw_kernels)
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


def _tr_huber_weights(scaled_residuals: Tensor, delta: float = 1.345) -> Tensor:
    abs_r = torch.abs(scaled_residuals)
    return torch.where(abs_r <= delta, torch.ones_like(abs_r), delta / (abs_r + 1.0e-8))


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


def _compute_detector_irls_weight(
    residual: Tensor,
    initial_precision: Tensor,
    *,
    data_weight_fn: Literal["none", "huber", "tukey", "power"] | Callable[..., Tensor],
    irls_config: _core_mod.IRLSConfig | None,
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
        weights = cast(Callable[..., Tensor], fn_spec)(scaled_res, **params)
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


# Late-bound so monkeypatching ``inverse`` attributes still reaches this code.
from . import inverse as _core_mod  # noqa: E402
