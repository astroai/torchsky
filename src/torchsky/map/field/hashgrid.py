"""Hash-grid and feature-plane encoding helpers for field models."""

from __future__ import annotations

import math
from typing import Literal, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def hash_grid_smoothstep(t: Tensor) -> Tensor:
    t = torch.clamp(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def hash_grid_ravel_voxels(corner: Tensor, resolution: int, coord_dim: int) -> Tensor:
    idx = corner[..., -1]
    mult = resolution
    for axis in range(coord_dim - 2, -1, -1):
        idx = idx + corner[..., axis] * mult
        mult *= resolution
    return idx


class HashGridEncoder(nn.Module):
    """Multiresolution spatial encoding: dense voxels when small, else hashed."""

    _PRIMES = (
        1,
        2654435761,
        805459861,
        3674653429,
    )

    def __init__(
        self,
        coord_dim: int,
        *,
        num_levels: int = 6,
        features_per_level: int = 4,
        log2_hashmap_size: int | Sequence[int] = 12,
        base_resolution: int = 8,
        finest_resolution: int = 128,
        interpolation: Literal["linear", "smoothstep"] = "linear",
        dense_coarse: bool = True,
        max_dense_cells: int = 4096,
        learnable_level_gain: bool = False,
    ) -> None:
        super().__init__()
        self.coord_dim = int(coord_dim)
        self.num_levels = int(num_levels)
        self.features_per_level = int(features_per_level)
        self.base_resolution = int(base_resolution)
        self.finest_resolution = int(finest_resolution)
        self.interpolation = interpolation
        self.dense_coarse = bool(dense_coarse)
        self.max_dense_cells = int(max_dense_cells)
        if interpolation not in {"linear", "smoothstep"}:
            raise ValueError(
                f"interpolation must be 'linear' or 'smoothstep', got {interpolation!r}"
            )
        if self.num_levels <= 0:
            raise ValueError("num_levels must be positive")
        if self.features_per_level <= 0:
            raise ValueError("features_per_level must be positive")
        if self.coord_dim <= 0 or self.coord_dim > len(self._PRIMES):
            raise ValueError("unsupported hash-grid coordinate dimension")
        if isinstance(log2_hashmap_size, int):
            log2_seq = (int(log2_hashmap_size),) * self.num_levels
        else:
            log2_seq = tuple(log2_hashmap_size)
            if len(log2_seq) != self.num_levels:
                raise ValueError(
                    "len(log2_hashmap_size) must equal num_levels when a sequence is passed"
                )
        self._log2_hashmap_per_level = log2_seq
        if self.num_levels == 1:
            resolutions = [self.base_resolution]
        else:
            growth = math.exp(
                math.log(self.finest_resolution / self.base_resolution)
                / max(self.num_levels - 1, 1)
            )
            resolutions = [
                max(2, int(round(self.base_resolution * (growth**level))))
                for level in range(self.num_levels)
            ]
        self.resolutions = tuple(resolutions)
        self._is_dense: tuple[bool, ...] = tuple(
            self.dense_coarse and (int(res) ** self.coord_dim) <= self.max_dense_cells
            for res in self.resolutions
        )
        self._table_sizes: tuple[int, ...] = tuple(
            int(res) ** self.coord_dim if dense else (1 << log2_seq[level])
            for level, (res, dense) in enumerate(
                zip(self.resolutions, self._is_dense, strict=True)
            )
        )
        self.tables = nn.ParameterList(
            [
                nn.Parameter(
                    1.0e-4
                    * torch.randn(
                        int(self._table_sizes[level]), self.features_per_level
                    )
                )
                for level in range(self.num_levels)
            ]
        )
        self.level_gain: nn.Parameter | None
        if learnable_level_gain:
            self.level_gain = nn.Parameter(torch.ones(self.num_levels))
        else:
            self.register_parameter("level_gain", None)

    def _hash(self, corner: Tensor, table_size: int) -> Tensor:
        hashed = corner[..., 0].to(dtype=torch.int64)
        for dim in range(1, self.coord_dim):
            hashed = torch.bitwise_xor(
                hashed,
                corner[..., dim].to(dtype=torch.int64) * int(self._PRIMES[dim]),
            )
        return torch.remainder(hashed, int(table_size)).to(dtype=torch.long)

    def forward(self, coords_unit: Tensor) -> Tensor:
        flat = coords_unit.reshape(-1, self.coord_dim)
        outputs: list[Tensor] = []
        for level_idx, resolution in enumerate(self.resolutions):
            table = self.tables[level_idx]
            table_size = int(self._table_sizes[level_idx])
            is_dense = self._is_dense[level_idx]
            pos = flat * float(resolution - 1)
            base = torch.floor(pos).to(dtype=torch.int64)
            frac = pos - base.to(dtype=pos.dtype)
            if self.interpolation == "smoothstep":
                frac_w = hash_grid_smoothstep(frac)
            else:
                frac_w = frac
            level_feat = torch.zeros(
                flat.shape[0],
                self.features_per_level,
                device=flat.device,
                dtype=table.dtype,
            )
            for mask in range(1 << self.coord_dim):
                weight = torch.ones(
                    flat.shape[0],
                    device=flat.device,
                    dtype=table.dtype,
                )
                corner = base.clone()
                for dim in range(self.coord_dim):
                    use_upper = bool((mask >> dim) & 1)
                    frac_dim = frac_w[:, dim].to(dtype=table.dtype)
                    if use_upper:
                        corner[:, dim] = corner[:, dim] + 1
                        weight = weight * frac_dim
                    else:
                        weight = weight * (1.0 - frac_dim)
                corner = torch.clamp(corner, min=0, max=resolution - 1)
                if is_dense:
                    idx = hash_grid_ravel_voxels(
                        corner, int(resolution), self.coord_dim
                    ).to(dtype=torch.long)
                    sample = table[idx]
                else:
                    sample = table[self._hash(corner, table_size)]
                level_feat = level_feat + weight.unsqueeze(-1) * sample
            if self.level_gain is not None:
                g = self.level_gain[level_idx].to(
                    device=level_feat.device, dtype=level_feat.dtype
                )
                level_feat = level_feat * g
            outputs.append(level_feat)
        encoded = torch.cat(outputs, dim=-1)
        return encoded.reshape(*coords_unit.shape[:-1], -1)


def sample_feature_plane(plane: Tensor, coord_u: Tensor, coord_v: Tensor) -> Tensor:
    flat_u = coord_u.reshape(-1)
    flat_v = coord_v.reshape(-1)
    grid = torch.stack([2.0 * flat_u - 1.0, 2.0 * flat_v - 1.0], dim=-1)
    grid = grid.view(1, -1, 1, 2).to(dtype=plane.dtype, device=plane.device)
    sampled = F.grid_sample(
        plane.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    sampled = sampled.squeeze(0).squeeze(-1).transpose(0, 1)
    return sampled.reshape(*coord_u.shape, plane.shape[0])


__all__ = [
    "HashGridEncoder",
    "hash_grid_ravel_voxels",
    "hash_grid_smoothstep",
    "sample_feature_plane",
]
