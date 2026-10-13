"""Hybrid factorized neural-field scene and preconditioned recurrent optimizer (Learn2FOAM / RIM).

Combines three complementary components on top of ``torchsky``'s observation operators:

1. **Factorized Scene Representation** (:class:`FactorizedObjectSceneState`, :class:`LocalObjectINRDecoder`):
   Decomposes the continuous sky into a smooth/grid diffuse background plus compact
   sources parameterized by explicit physical coordinates ``(positions, log_fluxes,
   log_radii, sed_coeffs)`` and a compact per-object morphological/chromatic latent
   ``latents`` decoded by a shared local coordinate MLP on canonical support disks.
2. **Cech-Graph Amortized Proposal** (:class:`CechSetProposalNetwork`,
   :func:`propose_factorized_objects_from_coadd`):
   Extracts multi-band SNR peaks from the fast stage-0 preconditioned adjoint coadd
   ``D^{-1} A^T W y`` and refines initial source states via sparse message passing on
   the exact disk-overlap Cech graph (:func:`~torchsky.map.field.foam.sparse_cech_graph`).
3. **Preconditioned Physics-Aware Recurrent Optimizer**
   (:class:`PreconditionedLearn2FoamOptimizer`, :func:`adapt_factorized_object_scene`):
   Iteratively refines the scene state using ``torchregress`` IRLS detector weights,
   diagonal-majorizer preconditioned natural gradients ``D_S^{-1} grad_S L``, local
   back-projected residual stamps, sparse Cech neighbor messages, and discrete
   split/merge/prune primitive adaptation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
import math
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..solvers import BoundObservationOperator, BoundObservationOperatorSet
from .foam import SparseCechGraph, sparse_cech_graph
from .inverse import (
    IRLSConfig,
    _compute_detector_irls_weight,
    linear_band_grid_diagonal_majorizer,
)


@dataclass(frozen=True)
class FactorizedObjectSceneState:
    """Explicit physical + latent state ``S = {p_i, log F_i, log r_i, c_i, z_i}`` for N sources.

    Attributes
    ----------
    positions:
        Source centers ``(N, 2)`` in local tangent-plane degrees ``(x_deg, y_deg)``.
    log_fluxes:
        Log amplitude ``(N,)`` such that ``flux_i = exp(log_fluxes[i]) > 0``.
    log_radii:
        Log support radius ``(N,)`` in local degrees such that ``r_i = exp(log_radii[i]) > 0``.
    sed_coeffs:
        Per-source band/spectral coefficients ``(N, B)`` in log-ratio space around unity
        (``band_scale_{i,b} = softplus(sed_coeffs[i, b]) / log(2)``).
    latents:
        Compact per-source morphological/spectral latent vectors ``(N, D)``.
    diffuse_grids:
        Optional per-band diffuse background grids ``(B, H, W)`` on the target sky grid.
    """

    positions: Tensor
    log_fluxes: Tensor
    log_radii: Tensor
    sed_coeffs: Tensor
    latents: Tensor
    diffuse_grids: Tensor | None = None

    def __post_init__(self) -> None:
        if self.positions.ndim != 2 or self.positions.shape[1] != 2:
            raise ValueError("positions must have shape (N, 2)")
        n = int(self.positions.shape[0])
        if self.log_fluxes.shape != (n,):
            raise ValueError("log_fluxes must have shape (N,)")
        if self.log_radii.shape != (n,):
            raise ValueError("log_radii must have shape (N,)")
        if self.sed_coeffs.ndim != 2 or int(self.sed_coeffs.shape[0]) != n:
            raise ValueError("sed_coeffs must have shape (N, B)")
        if self.latents.ndim != 2 or int(self.latents.shape[0]) != n:
            raise ValueError("latents must have shape (N, D)")
        if self.diffuse_grids is not None and (
            self.diffuse_grids.ndim != 3
            or int(self.diffuse_grids.shape[0]) != int(self.sed_coeffs.shape[1])
        ):
            raise ValueError("diffuse_grids must have shape (B, H, W) matching sed_coeffs bands")

    @property
    def num_objects(self) -> int:
        return int(self.positions.shape[0])

    @property
    def num_bands(self) -> int:
        return int(self.sed_coeffs.shape[1])

    @property
    def latent_dim(self) -> int:
        return int(self.latents.shape[1])

    @property
    def radii(self) -> Tensor:
        return torch.exp(torch.clamp(self.log_radii, min=-12.0, max=4.0))

    @property
    def fluxes(self) -> Tensor:
        return torch.exp(torch.clamp(self.log_fluxes, min=-20.0, max=20.0))

    def band_scales(self) -> Tensor:
        return F.softplus(self.sed_coeffs) / math.log(2.0)

    def detach(self) -> FactorizedObjectSceneState:
        return FactorizedObjectSceneState(
            positions=self.positions.detach(),
            log_fluxes=self.log_fluxes.detach(),
            log_radii=self.log_radii.detach(),
            sed_coeffs=self.sed_coeffs.detach(),
            latents=self.latents.detach(),
            diffuse_grids=(
                self.diffuse_grids.detach() if self.diffuse_grids is not None else None
            ),
        )


class LocalObjectINRDecoder(nn.Module):
    """Shared coordinate-conditioned neural decoder ``f_theta(xi, band; z_i)`` on canonical disks.

    Evaluates a compact C^2 radial profile times a zero-initialized latent modulation
    on canonical coordinates ``xi = (x - p_i) / r_i`` with ``||xi|| <= 1``.
    At ``z_i = 0`` (or before training), each source is an exact smooth radial Wendland-like
    bump ``(1 - ||xi||^2)_+^2`` so physical gradients on ``(p_i, log F_i, log r_i, c_i)``
    are well-conditioned from step 0.
    """

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_dim: int = 32,
        num_frequencies: int = 2,
    ) -> None:
        super().__init__()
        if latent_dim < 1:
            raise ValueError("latent_dim must be positive")
        if hidden_dim < 4:
            raise ValueError("hidden_dim must be >= 4")
        if num_frequencies < 1:
            raise ValueError("num_frequencies must be positive")
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_frequencies = int(num_frequencies)
        coord_in_dim = 2 + 4 * self.num_frequencies + 1 + self.latent_dim
        self.net = nn.Sequential(
            nn.Linear(coord_in_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.out_head = nn.Linear(self.hidden_dim, 1)
        nn.init.zeros_(self.out_head.weight)
        nn.init.zeros_(self.out_head.bias)

    def _encode_coords(self, xi: Tensor, band_coord: Tensor) -> Tensor:
        feats: list[Tensor] = [xi]
        for k in range(self.num_frequencies):
            freq = float(2**k) * math.pi
            feats.append(torch.sin(freq * xi))
            feats.append(torch.cos(freq * xi))
        feats.append(band_coord)
        return torch.cat(feats, dim=-1)

    def forward(
        self,
        xi: Tensor,
        latents: Tensor,
        band_coord: Tensor | float = 0.0,
    ) -> Tensor:
        """Evaluate dimensionless profile on local canonical coordinates ``xi (..., 2)``."""
        r_sq = torch.sum(xi * xi, dim=-1)
        inside = r_sq < 1.0
        window = torch.square(torch.clamp(1.0 - r_sq, min=0.0))
        if not bool(inside.any()):
            return window
        if isinstance(band_coord, (int, float)):
            b_t = torch.full(
                (*xi.shape[:-1], 1),
                float(band_coord),
                device=xi.device,
                dtype=xi.dtype,
            )
        else:
            b_t = band_coord.to(device=xi.device, dtype=xi.dtype)
            if b_t.shape != (*xi.shape[:-1], 1):
                b_t = b_t.expand(*xi.shape[:-1], 1)
        z_expanded = latents.to(device=xi.device, dtype=xi.dtype)
        if z_expanded.shape[:-1] != xi.shape[:-1]:
            z_expanded = z_expanded.expand(*xi.shape[:-1], self.latent_dim)
        enc = self._encode_coords(xi, b_t)
        inp = torch.cat((enc, z_expanded), dim=-1)
        mod = torch.tanh(self.out_head(self.net(inp)).squeeze(-1))
        return window * (1.0 + 0.5 * mod)


def render_factorized_scene_grid(
    state: FactorizedObjectSceneState,
    decoder: LocalObjectINRDecoder,
    *,
    spatial_resolution: int,
    spatial_bounds_local_deg: tuple[float, float, float, float],
    band_coordinates: Sequence[float] | None = None,
) -> Tensor:
    """Render the factorized scene to a multi-band sky grid ``(B, H, W)``."""
    res = int(spatial_resolution)
    if res < 2:
        raise ValueError("spatial_resolution must be >= 2")
    x_min, x_max, y_min, y_max = (float(v) for v in spatial_bounds_local_deg)
    device = state.positions.device
    dtype = state.positions.dtype
    num_bands = state.num_bands
    if band_coordinates is None:
        band_coords = (
            [0.0]
            if num_bands == 1
            else [
                -1.0 + 2.0 * idx / max(num_bands - 1, 1)
                for idx in range(num_bands)
            ]
        )
    else:
        if len(band_coordinates) != num_bands:
            raise ValueError("band_coordinates length must match num_bands")
        band_coords = [float(v) for v in band_coordinates]

    if state.diffuse_grids is not None:
        if state.diffuse_grids.shape != (num_bands, res, res):
            raise ValueError("diffuse_grids shape must match (num_bands, res, res)")
        canvas = state.diffuse_grids.to(device=device, dtype=dtype).clone()
    else:
        canvas = torch.zeros((num_bands, res, res), device=device, dtype=dtype)

    if state.num_objects == 0:
        return canvas

    xs = torch.linspace(x_min, x_max, res, device=device, dtype=dtype)
    ys = torch.linspace(y_min, y_max, res, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid_xy = torch.stack((grid_x, grid_y), dim=-1)  # (H, W, 2)

    radii = state.radii  # (N,)
    fluxes = state.fluxes  # (N,)
    band_scales = state.band_scales()  # (N, B)

    # Evaluate each object only inside its bounding box on the sky grid (O(N * k^2), not O(N * H * W))
    dx_cell = (x_max - x_min) / max(res - 1, 1)
    dy_cell = (y_max - y_min) / max(res - 1, 1)
    pos_det = state.positions.detach()
    rad_det = radii.detach()

    band_coord_t = torch.tensor(band_coords, device=device, dtype=dtype).view(
        num_bands, 1, 1, 1
    )
    for i in range(state.num_objects):
        px = float(pos_det[i, 0].item())
        py = float(pos_det[i, 1].item())
        ri = float(rad_det[i].item())
        c0 = max(0, int(math.floor((px - ri - x_min) / max(dx_cell, 1e-12))))
        c1 = min(res - 1, int(math.ceil((px + ri - x_min) / max(dx_cell, 1e-12))))
        r0 = max(0, int(math.floor((py - ri - y_min) / max(dy_cell, 1e-12))))
        r1 = min(res - 1, int(math.ceil((py + ri - y_min) / max(dy_cell, 1e-12))))
        if r0 > r1 or c0 > c1:
            continue
        sub_xy = grid_xy[r0 : r1 + 1, c0 : c1 + 1]  # (h_i, w_i, 2)
        h_i, w_i = int(sub_xy.shape[0]), int(sub_xy.shape[1])
        xi = ((sub_xy - state.positions[i].view(1, 1, 2)) / radii[i]).unsqueeze(0).expand(
            num_bands, h_i, w_i, 2
        )
        z_i = state.latents[i].view(1, 1, 1, state.latent_dim).expand(
            num_bands, h_i, w_i, state.latent_dim
        )
        b_i = band_coord_t.expand(num_bands, h_i, w_i, 1)
        prof = decoder(xi, z_i, band_coord=b_i)  # (B, h_i, w_i)
        amp = (fluxes[i] * band_scales[i]).view(num_bands, 1, 1)
        canvas[:, r0 : r1 + 1, c0 : c1 + 1] = (
            canvas[:, r0 : r1 + 1, c0 : c1 + 1] + amp * prof
        )
    return canvas


def _sparse_cech_mean_neighbor_messages(
    node_features: Tensor,
    positions: Tensor,
    radii: Tensor,
    edge_mlp: nn.Module,
) -> tuple[Tensor, SparseCechGraph]:
    """Aggregate pairwise relative-geometry messages over the exact Cech overlap graph."""
    graph = sparse_cech_graph(positions.detach(), radii.detach())
    n = int(node_features.shape[0])
    out_dim = int(node_features.shape[1])
    if graph.edges.numel() == 0 or n <= 1:
        return torch.zeros((n, out_dim), device=node_features.device, dtype=node_features.dtype), graph
    src = torch.cat((graph.edges[0], graph.edges[1]), dim=0)
    dst = torch.cat((graph.edges[1], graph.edges[0]), dim=0)
    scale = torch.clamp(radii[dst].unsqueeze(-1), min=1.0e-6)
    rel_pos = (positions[src] - positions[dst]) / scale
    rel_rad = (radii[src] / torch.clamp(radii[dst], min=1.0e-6)).unsqueeze(-1)
    edge_in = torch.cat((node_features[dst], node_features[src], rel_pos, rel_rad), dim=-1)
    msgs = edge_mlp(edge_in)
    agg = torch.zeros((n, msgs.shape[-1]), device=node_features.device, dtype=node_features.dtype)
    agg.index_add_(0, dst, msgs)
    deg = torch.bincount(dst, minlength=n).to(dtype=node_features.dtype).clamp(min=1.0).unsqueeze(-1)
    return agg / deg, graph


def _extract_local_sky_stamps(
    maps: Tensor,
    positions: Tensor,
    *,
    spatial_bounds_local_deg: tuple[float, float, float, float],
    stamp_size: int = 5,
) -> Tensor:
    """Sample local ``(N, B, stamp_size, stamp_size)`` stamps around each source center."""
    if positions.shape[0] == 0:
        return maps.new_empty((0, maps.shape[0], stamp_size, stamp_size))
    x_min, x_max, y_min, y_max = (float(v) for v in spatial_bounds_local_deg)
    res_y, res_x = int(maps.shape[1]), int(maps.shape[2])
    dx = (x_max - x_min) / max(res_x - 1, 1)
    dy = (y_max - y_min) / max(res_y - 1, 1)
    half = stamp_size // 2
    offs = torch.arange(-half, half + 1, device=positions.device, dtype=positions.dtype)
    oy, ox = torch.meshgrid(offs * dy, offs * dx, indexing="ij")
    sample_xy = positions[:, None, None, :] + torch.stack((ox, oy), dim=-1).unsqueeze(0)
    norm_x = 2.0 * (sample_xy[..., 0] - x_min) / max(x_max - x_min, 1.0e-12) - 1.0
    norm_y = 2.0 * (sample_xy[..., 1] - y_min) / max(y_max - y_min, 1.0e-12) - 1.0
    grid = torch.stack((norm_x, norm_y), dim=-1)  # (N, S, S, 2)
    b_count = int(maps.shape[0])
    sampled = F.grid_sample(
        maps.unsqueeze(0),
        grid.view(1, -1, stamp_size, 2),
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.view(b_count, positions.shape[0], stamp_size, stamp_size).permute(1, 0, 2, 3)


class CechSetProposalNetwork(nn.Module):
    """Amortized proposal network over multi-band preconditioned coadd stamps and Cech graph."""

    def __init__(
        self,
        num_bands: int = 1,
        latent_dim: int = 16,
        hidden_dim: int = 32,
        stamp_size: int = 5,
    ) -> None:
        super().__init__()
        self.num_bands = int(num_bands)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.stamp_size = int(stamp_size)
        in_dim = 2 * self.num_bands * self.stamp_size * self.stamp_size + 4
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * self.hidden_dim + 3, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        out_dim = 2 + 1 + 1 + self.num_bands + self.latent_dim
        self.head = nn.Linear(self.hidden_dim, out_dim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        seed_state: FactorizedObjectSceneState,
        coadd_grids: Tensor,
        sensitivity_grids: Tensor,
        *,
        spatial_bounds_local_deg: tuple[float, float, float, float],
    ) -> FactorizedObjectSceneState:
        if seed_state.num_objects == 0:
            return seed_state
        snr_grids = coadd_grids * torch.sqrt(torch.clamp(sensitivity_grids, min=0.0))
        coadd_stamps = _extract_local_sky_stamps(
            coadd_grids,
            seed_state.positions,
            spatial_bounds_local_deg=spatial_bounds_local_deg,
            stamp_size=self.stamp_size,
        )
        snr_stamps = _extract_local_sky_stamps(
            snr_grids,
            seed_state.positions,
            spatial_bounds_local_deg=spatial_bounds_local_deg,
            stamp_size=self.stamp_size,
        )
        n = seed_state.num_objects
        flat_stamps = torch.cat(
            (coadd_stamps.reshape(n, -1), snr_stamps.reshape(n, -1)), dim=-1
        )
        meta = torch.cat(
            (
                seed_state.positions,
                seed_state.log_fluxes.unsqueeze(-1),
                seed_state.log_radii.unsqueeze(-1),
            ),
            dim=-1,
        )
        h = self.encoder(torch.cat((flat_stamps, meta), dim=-1))
        msg, _ = _sparse_cech_mean_neighbor_messages(
            h, seed_state.positions, seed_state.radii, self.edge_mlp
        )
        delta = self.head(h + msg)
        d_pos = 0.25 * torch.tanh(delta[:, :2]) * seed_state.radii.unsqueeze(-1)
        d_flux = 0.5 * torch.tanh(delta[:, 2])
        d_rad = 0.25 * torch.tanh(delta[:, 3])
        d_sed = delta[:, 4 : 4 + self.num_bands]
        d_lat = delta[:, 4 + self.num_bands :]
        return FactorizedObjectSceneState(
            positions=seed_state.positions + d_pos,
            log_fluxes=seed_state.log_fluxes + d_flux,
            log_radii=seed_state.log_radii + d_rad,
            sed_coeffs=seed_state.sed_coeffs + d_sed,
            latents=seed_state.latents + d_lat,
            diffuse_grids=seed_state.diffuse_grids,
        )


def propose_factorized_objects_from_coadd(
    operators_by_band: Sequence[BoundObservationOperator | BoundObservationOperatorSet],
    data_by_band: Sequence[Tensor | Sequence[Tensor]],
    *,
    spatial_resolution: int,
    spatial_bounds_local_deg: tuple[float, float, float, float],
    max_objects: int = 32,
    snr_threshold: float = 3.0,
    initial_radius_cells: float = 2.5,
    latent_dim: int = 16,
    proposal_net: CechSetProposalNetwork | None = None,
) -> tuple[FactorizedObjectSceneState, Tensor, Tensor]:
    """Propose initial :class:`FactorizedObjectSceneState` from multi-band ``(D^{-1} A^T W y, D)``."""
    if not operators_by_band or len(operators_by_band) != len(data_by_band):
        raise ValueError("operators_by_band and data_by_band must be non-empty and aligned")
    res = int(spatial_resolution)
    x_min, x_max, y_min, y_max = (float(v) for v in spatial_bounds_local_deg)

    coadd_list: list[Tensor] = []
    sens_list: list[Tensor] = []
    bg_list: list[Tensor] = []
    for op_or_set, data in zip(operators_by_band, data_by_band, strict=True):
        if isinstance(op_or_set, BoundObservationOperator):
            assert isinstance(data, Tensor)
            rhs = op_or_set.adjoint(op_or_set.apply_noise_model(data))
        else:
            assert isinstance(data, Sequence) and not isinstance(data, Tensor)
            rhs = op_or_set.weighted_adjoint(tuple(torch.as_tensor(d) for d in data))
        sens = linear_band_grid_diagonal_majorizer(op_or_set, data)
        pos_sens = sens[sens > 0.0]
        med_sens = float(torch.median(pos_sens).item()) if pos_sens.numel() > 0 else 1.0
        floor = max(1.0e-12, 1.0e-2 * med_sens)
        coadd = torch.where(sens > 0.0, rhs / torch.clamp(sens, min=floor), torch.zeros_like(rhs))
        well_cov = sens >= (0.5 * med_sens)
        bg_val = (
            float(torch.median(coadd[well_cov]).item())
            if bool(well_cov.any())
            else 0.0
        )
        coadd_list.append(coadd - bg_val)
        sens_list.append(sens)
        bg_list.append(torch.full_like(coadd, bg_val))

    coadd_grids = torch.stack(coadd_list, dim=0)  # (B, H, W)
    sens_grids = torch.stack(sens_list, dim=0)  # (B, H, W)
    diffuse_grids = torch.stack(bg_list, dim=0)  # (B, H, W)

    snr_per_band = coadd_grids * torch.sqrt(torch.clamp(sens_grids, min=0.0))
    combined_snr = torch.sqrt(torch.mean(torch.square(torch.clamp(snr_per_band, min=0.0)), dim=0))
    pooled = F.max_pool2d(combined_snr.view(1, 1, res, res), kernel_size=3, stride=1, padding=1)[0, 0]
    is_peak = (combined_snr >= pooled - 1.0e-12) & (combined_snr >= float(snr_threshold))
    peak_rc = torch.nonzero(is_peak, as_tuple=False)
    if peak_rc.numel() == 0:
        # Fallback to global maximum so proposal is never empty when max_objects >= 1
        flat_idx = int(torch.argmax(combined_snr).item())
        peak_rc = torch.tensor([[flat_idx // res, flat_idx % res]], device=combined_snr.device)

    peak_scores = combined_snr[peak_rc[:, 0], peak_rc[:, 1]]
    order = torch.argsort(peak_scores, descending=True)[: int(max_objects)]
    peak_rc = peak_rc[order]

    device = coadd_grids.device
    dtype = coadd_grids.dtype
    dx = (x_max - x_min) / max(res - 1, 1)
    dy = (y_max - y_min) / max(res - 1, 1)
    xs = x_min + peak_rc[:, 1].to(dtype=dtype) * dx
    ys = y_min + peak_rc[:, 0].to(dtype=dtype) * dy
    positions = torch.stack((xs, ys), dim=-1)

    band_vals = torch.clamp(coadd_grids[:, peak_rc[:, 0], peak_rc[:, 1]].T, min=1.0e-4)  # (N, B)
    mean_flux = torch.mean(band_vals, dim=-1)
    log_fluxes = torch.log(mean_flux)
    ratios = torch.clamp(band_vals / mean_flux.unsqueeze(-1), min=0.1, max=10.0)
    # Inverse softplus around log(2)*ratio: softplus(sed_coeffs)/log(2) == ratios
    target_sp = ratios * math.log(2.0)
    sed_coeffs = torch.log(torch.clamp(torch.expm1(target_sp), min=1.0e-6))

    r0 = max(dx, dy) * float(initial_radius_cells)
    log_radii = torch.full((positions.shape[0],), math.log(max(r0, 1.0e-5)), device=device, dtype=dtype)
    latents = torch.zeros((positions.shape[0], int(latent_dim)), device=device, dtype=dtype)

    state = FactorizedObjectSceneState(
        positions=positions,
        log_fluxes=log_fluxes,
        log_radii=log_radii,
        sed_coeffs=sed_coeffs,
        latents=latents,
        diffuse_grids=diffuse_grids,
    )
    if proposal_net is not None:
        state = proposal_net(
            state,
            coadd_grids,
            sens_grids,
            spatial_bounds_local_deg=spatial_bounds_local_deg,
        )
    return state, coadd_grids, sens_grids


@dataclass(frozen=True)
class SceneOptimizationResult:
    """Output of :class:`PreconditionedLearn2FoamOptimizer`."""

    state: FactorizedObjectSceneState
    sky_grids: Tensor
    loss_history: tuple[float, ...]
    diagnostics: dict[str, Any]


def adapt_factorized_object_scene(
    state: FactorizedObjectSceneState,
    pos_grad_norm: Tensor,
    local_residual_snr: Tensor,
    *,
    spatial_bounds_local_deg: tuple[float, float, float, float],
    split_grad_threshold: float = 0.25,
    split_snr_threshold: float = 4.0,
    merge_distance_fraction: float = 0.35,
    merge_sed_cosine_threshold: float = 0.95,
    prune_flux_threshold: float = 1.0e-5,
    max_objects: int = 128,
) -> tuple[FactorizedObjectSceneState, dict[str, int]]:
    """Split unresolved blends, merge redundant Cech neighbors, and prune dead primitives."""
    n = state.num_objects
    if n == 0:
        return state, {"split": 0, "merged": 0, "pruned": 0}

    fluxes = state.fluxes.detach()
    radii = state.radii.detach()
    keep = fluxes > float(prune_flux_threshold)
    pruned_count = int((~keep).sum().item())

    # Merge nearly coincident Cech neighbors with collinear SEDs
    merged_count = 0
    graph = sparse_cech_graph(state.positions.detach(), radii)
    absorbed = torch.zeros(n, dtype=torch.bool, device=state.positions.device)
    pos_new = state.positions.clone()
    flux_new = fluxes.clone()
    sed_scales = state.band_scales().detach()
    if graph.edges.numel() > 0:
        for e_idx in range(int(graph.edges.shape[1])):
            i = int(graph.edges[0, e_idx].item())
            j = int(graph.edges[1, e_idx].item())
            if not (bool(keep[i]) and bool(keep[j])) or bool(absorbed[i]) or bool(absorbed[j]):
                continue
            dist = float(torch.linalg.vector_norm(pos_new[i] - pos_new[j]).item())
            min_r = float(torch.minimum(radii[i], radii[j]).item())
            if dist <= float(merge_distance_fraction) * min_r:
                cos_sim = float(
                    F.cosine_similarity(
                        sed_scales[i].unsqueeze(0), sed_scales[j].unsqueeze(0)
                    ).item()
                )
                if cos_sim >= float(merge_sed_cosine_threshold):
                    fi = flux_new[i]
                    fj = flux_new[j]
                    f_tot = fi + fj
                    pos_new[i] = (fi * pos_new[i] + fj * pos_new[j]) / torch.clamp(f_tot, min=1e-12)
                    flux_new[i] = f_tot
                    absorbed[j] = True
                    keep[j] = False
                    merged_count += 1

    # Split high-residual, high-positional-gradient sources
    can_split = (
        keep
        & (pos_grad_norm.detach() >= float(split_grad_threshold))
        & (local_residual_snr.detach() >= float(split_snr_threshold))
    )
    kept_indices = torch.nonzero(keep, as_tuple=False).squeeze(-1)
    available_slots = max(0, int(max_objects) - int(kept_indices.numel()))
    split_candidates = torch.nonzero(can_split, as_tuple=False).squeeze(-1)
    if split_candidates.numel() > available_slots:
        scores = local_residual_snr[split_candidates] * pos_grad_norm[split_candidates]
        top = torch.argsort(scores, descending=True)[:available_slots]
        split_candidates = split_candidates[top]
    split_count = int(split_candidates.numel())

    pos_list = [pos_new[kept_indices]]
    log_flux_list = [torch.log(torch.clamp(flux_new[kept_indices], min=1e-12))]
    log_rad_list = [state.log_radii[kept_indices]]
    sed_list = [state.sed_coeffs[kept_indices]]
    lat_list = [state.latents[kept_indices]]

    if split_count > 0:
        x_min, x_max, y_min, y_max = (float(v) for v in spatial_bounds_local_deg)
        lower = torch.tensor([x_min, y_min], device=pos_new.device, dtype=pos_new.dtype)
        upper = torch.tensor([x_max, y_max], device=pos_new.device, dtype=pos_new.dtype)
        # Halve parent flux and shrink parent radius by 1/sqrt(2); spawn antipodal daughter
        is_split_kept = torch.isin(kept_indices, split_candidates)
        log_flux_list[0] = torch.where(
            is_split_kept,
            log_flux_list[0] - math.log(2.0),
            log_flux_list[0],
        )
        log_rad_list[0] = torch.where(
            is_split_kept,
            log_rad_list[0] - 0.5 * math.log(2.0),
            log_rad_list[0],
        )
        offset = 0.25 * radii[split_candidates].unsqueeze(-1)
        pos_list[0] = torch.where(
            is_split_kept.unsqueeze(-1),
            torch.clamp(pos_list[0] - 0.25 * radii[kept_indices].unsqueeze(-1), min=lower, max=upper),
            pos_list[0],
        )
        daughter_pos = torch.clamp(pos_new[split_candidates] + offset, min=lower, max=upper)
        pos_list.append(daughter_pos)
        log_flux_list.append(torch.log(torch.clamp(0.5 * flux_new[split_candidates], min=1e-12)))
        log_rad_list.append(state.log_radii[split_candidates] - 0.5 * math.log(2.0))
        sed_list.append(state.sed_coeffs[split_candidates].clone())
        lat_list.append(state.latents[split_candidates].clone())

    adapted = FactorizedObjectSceneState(
        positions=torch.cat(pos_list, dim=0),
        log_fluxes=torch.cat(log_flux_list, dim=0),
        log_radii=torch.cat(log_rad_list, dim=0),
        sed_coeffs=torch.cat(sed_list, dim=0),
        latents=torch.cat(lat_list, dim=0),
        diffuse_grids=state.diffuse_grids,
    )
    return adapted, {"split": split_count, "merged": merged_count, "pruned": pruned_count}


class PreconditionedLearn2FoamOptimizer(nn.Module):
    """Recurrent physics-aware scene optimizer (Learn2FOAM / Preconditioned RIM).

    At each outer step ``t``:
    1. Evaluates the current continuous scene on the target sky grid ``G_t = f_theta(S_t)``.
    2. Computes detector residuals ``r_{b,e} = y_{b,e} - A_{b,e} G_{b,t}`` and optional
       ``torchregress`` IRLS precision weights ``W_{b,e,t} = W_{b,e,0} * w(r_{b,e} / sigma_{b,e})``.
    3. Back-projects the sufficient statistics to the sky grid in a single batched pass:
       ``g_sky = A^T W_t (A G_t - y)`` and diagonal majorizer ``D_sky = A^T W_t 1``.
    4. Pulls ``(g_sky, D_sky)`` back to the factorized object parameters ``S_t`` via vector-Jacobian
       products to obtain exact gradients ``g_S`` and block-diagonal Gauss-Newton preconditioners
       ``D_S``, forming dimensionless natural gradients ``tilde_g_S = D_S^{-1} g_S``.
    5. Feeds ``(S_t, tilde_g_S, local_residual_stamps, Cech_neighbor_messages, h_t)`` into a
       recurrent GRU + Cech-graph cell ``G_phi`` that predicts a residual correction around the
       damped natural-gradient step, with monotone backtracking line search safety.
    """

    def __init__(
        self,
        decoder: LocalObjectINRDecoder | None = None,
        *,
        num_bands: int = 1,
        latent_dim: int = 16,
        hidden_dim: int = 32,
        stamp_size: int = 5,
        step_scale: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_bands = int(num_bands)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.stamp_size = int(stamp_size)
        self.step_scale = float(step_scale)
        self.decoder = (
            decoder
            if decoder is not None
            else LocalObjectINRDecoder(latent_dim=self.latent_dim, hidden_dim=self.hidden_dim)
        )
        param_dim = 2 + 1 + 1 + self.num_bands + self.latent_dim
        stamp_dim = self.num_bands * self.stamp_size * self.stamp_size
        in_dim = 2 * param_dim + stamp_dim
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * self.hidden_dim + 3, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.gru = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.delta_head = nn.Linear(self.hidden_dim, param_dim)
        self.gate_head = nn.Linear(self.hidden_dim, param_dim)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.gate_head.bias)

    def _pack_object_tensor(self, state: FactorizedObjectSceneState) -> Tensor:
        return torch.cat(
            (
                state.positions,
                state.log_fluxes.unsqueeze(-1),
                state.log_radii.unsqueeze(-1),
                state.sed_coeffs,
                state.latents,
            ),
            dim=-1,
        )

    def _unpack_object_tensor(
        self,
        packed: Tensor,
        diffuse_grids: Tensor | None,
    ) -> FactorizedObjectSceneState:
        b = self.num_bands
        return FactorizedObjectSceneState(
            positions=packed[:, :2],
            log_fluxes=packed[:, 2],
            log_radii=packed[:, 3],
            sed_coeffs=packed[:, 4 : 4 + b],
            latents=packed[:, 4 + b :],
            diffuse_grids=diffuse_grids,
        )

    def forward(
        self,
        initial_state: FactorizedObjectSceneState,
        operators_by_band: Sequence[BoundObservationOperator | BoundObservationOperatorSet],
        data_by_band: Sequence[Tensor | Sequence[Tensor]],
        *,
        spatial_resolution: int,
        spatial_bounds_local_deg: tuple[float, float, float, float],
        steps: int = 4,
        damping: float = 1.0e-2,
        update_diffuse: bool = True,
        diffuse_step_size: float = 0.5,
        data_weight_fn: (
            Literal["none", "huber", "tukey", "power"] | Callable[..., Tensor]
        ) = "none",
        irls_config: IRLSConfig | None = None,
        adapt_every: int = 0,
        backtracking: int = 6,
    ) -> SceneOptimizationResult:
        """Run ``steps`` unrolled physics-aware recurrent refinement iterations."""
        if steps < 1:
            raise ValueError("steps must be positive")
        if len(operators_by_band) != self.num_bands or len(data_by_band) != self.num_bands:
            raise ValueError("operators_by_band and data_by_band must match num_bands")

        state = initial_state
        n_obj = state.num_objects
        device = state.positions.device
        dtype = state.positions.dtype
        hidden = torch.zeros((n_obj, self.hidden_dim), device=device, dtype=dtype)
        use_irls = bool(data_weight_fn != "none" or irls_config is not None)

        # Cache initial detector precisions W_{b,e,0}
        init_precs: list[Tensor | tuple[Tensor, ...]] = []
        for op_or_set, data in zip(operators_by_band, data_by_band, strict=True):
            if isinstance(op_or_set, BoundObservationOperator):
                d_t = torch.as_tensor(data)
                init_precs.append(
                    torch.clamp(
                        op_or_set.apply_noise_model(torch.ones_like(d_t)).detach(),
                        min=0.0,
                    )
                )
            else:
                assert isinstance(data, Sequence) and not isinstance(data, Tensor)
                init_precs.append(
                    tuple(
                        torch.clamp(
                            op.apply_noise_model(torch.ones_like(torch.as_tensor(d))).detach(),
                            min=0.0,
                        )
                        for op, d in zip(op_or_set.operators, data, strict=True)
                    )
                )

        # Cache base sensitivity grids when det_w is None
        base_sens_by_band: list[Tensor] = []
        with torch.no_grad():
            for op_or_set, data in zip(operators_by_band, data_by_band, strict=True):
                if isinstance(op_or_set, BoundObservationOperator):
                    d_t = torch.as_tensor(data, device=device, dtype=dtype)
                    base_sens_by_band.append(
                        torch.clamp(
                            op_or_set.adjoint(op_or_set.apply_noise_model(torch.ones_like(d_t))),
                            min=0.0,
                        )
                    )
                else:
                    assert isinstance(data, Sequence) and not isinstance(data, Tensor)
                    d_tup = tuple(torch.as_tensor(d, device=device, dtype=dtype) for d in data)
                    base_sens_by_band.append(linear_band_grid_diagonal_majorizer(op_or_set, d_tup))

        def _evaluate_sky_and_loss(
            cand_state: FactorizedObjectSceneState,
            det_weights_by_band: list[Tensor | tuple[Tensor, ...] | None] | None = None,
            *,
            compute_stats: bool = True,
        ) -> tuple[Tensor, float, list[Tensor], list[Tensor]]:
            with torch.no_grad():
                sky_grids = render_factorized_scene_grid(
                    cand_state.detach(),
                    self.decoder,
                    spatial_resolution=spatial_resolution,
                    spatial_bounds_local_deg=spatial_bounds_local_deg,
                )
                total_loss = 0.0
                grad_sky_list: list[Tensor] = []
                sens_sky_list: list[Tensor] = []
                for b_idx, (op_or_set, data) in enumerate(
                    zip(operators_by_band, data_by_band, strict=True)
                ):
                    grid_b = sky_grids[b_idx]
                    det_w = (
                        det_weights_by_band[b_idx]
                        if det_weights_by_band is not None
                        else None
                    )
                    if isinstance(op_or_set, BoundObservationOperator):
                        d_t = torch.as_tensor(data, device=device, dtype=dtype)
                        pred = op_or_set.apply(grid_b)
                        diff = pred - d_t
                        w_diff = op_or_set.apply_noise_model(
                            diff if det_w is None else diff * det_w  # type: ignore[arg-type]
                        )
                        total_loss += 0.5 * float(torch.sum(diff.double() * w_diff.double()).item())
                        if compute_stats:
                            grad_sky_list.append(op_or_set.adjoint(w_diff))
                            if det_w is None:
                                sens_sky_list.append(base_sens_by_band[b_idx])
                            else:
                                sens_sky_list.append(
                                    torch.clamp(
                                        op_or_set.adjoint(op_or_set.apply_noise_model(det_w)),  # type: ignore[arg-type]
                                        min=0.0,
                                    )
                                )
                    else:
                        assert isinstance(data, Sequence) and not isinstance(data, Tensor)
                        d_tup = tuple(torch.as_tensor(d, device=device, dtype=dtype) for d in data)
                        preds = op_or_set.apply(grid_b)
                        diffs = tuple(p - d for p, d in zip(preds, d_tup, strict=True))
                        if det_w is not None:
                            assert isinstance(det_w, tuple)
                            scaled_diffs = tuple(
                                df * w for df, w in zip(diffs, det_w, strict=True)
                            )
                        else:
                            scaled_diffs = diffs
                        for op, df, s_df in zip(op_or_set.operators, diffs, scaled_diffs, strict=True):
                            w_df = op.apply_noise_model(s_df)
                            total_loss += 0.5 * float(torch.sum(df.double() * w_df.double()).item())
                        if compute_stats:
                            grad_sky_list.append(op_or_set.weighted_adjoint(scaled_diffs))
                            if det_w is not None:
                                assert isinstance(det_w, tuple)
                                sens_sky_list.append(
                                    torch.clamp(op_or_set.weighted_adjoint(det_w), min=0.0)
                                )
                            else:
                                sens_sky_list.append(base_sens_by_band[b_idx])
                return sky_grids, total_loss, grad_sky_list, sens_sky_list

        loss_history: list[float] = []
        adapt_events: list[dict[str, int]] = []
        x_min, x_max, y_min, y_max = (float(v) for v in spatial_bounds_local_deg)
        lower_xy = torch.tensor([x_min, y_min], device=device, dtype=dtype)
        upper_xy = torch.tensor([x_max, y_max], device=device, dtype=dtype)
        cell_scale = max(
            (x_max - x_min) / max(spatial_resolution - 1, 1),
            (y_max - y_min) / max(spatial_resolution - 1, 1),
        )

        for step_idx in range(steps):
            det_weights_by_band: list[Tensor | tuple[Tensor, ...] | None] = [
                None for _ in range(self.num_bands)
            ]
            if use_irls:
                with torch.no_grad():
                    cur_sky = render_factorized_scene_grid(
                        state,
                        self.decoder,
                        spatial_resolution=spatial_resolution,
                        spatial_bounds_local_deg=spatial_bounds_local_deg,
                    )
                    for b_idx, (op_or_set, data, w0_b) in enumerate(
                        zip(operators_by_band, data_by_band, init_precs, strict=True)
                    ):
                        if isinstance(op_or_set, BoundObservationOperator):
                            assert isinstance(w0_b, Tensor)
                            d_t = torch.as_tensor(data, device=device, dtype=dtype)
                            res = d_t - op_or_set.apply(cur_sky[b_idx])
                            det_weights_by_band[b_idx] = _compute_detector_irls_weight(
                                res,
                                w0_b,
                                data_weight_fn=data_weight_fn,
                                irls_config=irls_config,
                                convex_warmup=(step_idx == 0 and steps >= 2),
                            )
                        else:
                            assert isinstance(data, Sequence) and not isinstance(data, Tensor)
                            assert isinstance(w0_b, tuple)
                            preds = op_or_set.apply(cur_sky[b_idx])
                            det_weights_by_band[b_idx] = tuple(
                                _compute_detector_irls_weight(
                                    torch.as_tensor(d, device=device, dtype=dtype) - p,
                                    w0_e,
                                    data_weight_fn=data_weight_fn,
                                    irls_config=irls_config,
                                    convex_warmup=(step_idx == 0 and steps >= 2),
                                )
                                for d, p, w0_e in zip(data, preds, w0_b, strict=True)
                            )

            _, cur_loss, grad_sky_list, sens_sky_list = _evaluate_sky_and_loss(
                state, det_weights_by_band
            )
            if step_idx == 0:
                loss_history.append(cur_loss)

            grad_sky = torch.stack(grad_sky_list, dim=0)  # (B, H, W)
            sens_sky = torch.stack(sens_sky_list, dim=0)  # (B, H, W)

            # Optional preconditioned diffuse background update on smooth sky residuals
            next_diffuse = state.diffuse_grids
            if update_diffuse and next_diffuse is not None:
                pos_s = sens_sky[sens_sky > 0.0]
                med_s = float(torch.median(pos_s).item()) if pos_s.numel() > 0 else 1.0
                inv_s = torch.where(
                    sens_sky > 0.0,
                    1.0 / torch.clamp(sens_sky + float(damping) * med_s, min=1.0e-12),
                    torch.zeros_like(sens_sky),
                )
                raw_diff_step = -float(diffuse_step_size) * (grad_sky * inv_s)
                smooth_step = F.avg_pool2d(
                    raw_diff_step.unsqueeze(1), kernel_size=3, stride=1, padding=1
                ).squeeze(1)
                next_diffuse = next_diffuse + smooth_step

            if state.num_objects == 0:
                state = replace(state, diffuse_grids=next_diffuse)
                _, after_loss, _, _ = _evaluate_sky_and_loss(
                    state, det_weights_by_band, compute_stats=False
                )
                loss_history.append(after_loss)
                continue

            # Compute exact per-object gradient g_S and Gauss-Newton diagonal majorizer D_S
            packed_init = self._pack_object_tensor(state)
            packed_var = packed_init.detach().clone().requires_grad_(True)
            state_var = self._unpack_object_tensor(packed_var, next_diffuse)
            sky_var = render_factorized_scene_grid(
                state_var,
                self.decoder,
                spatial_resolution=spatial_resolution,
                spatial_bounds_local_deg=spatial_bounds_local_deg,
            )
            (grad_packed,) = torch.autograd.grad(
                sky_var,
                packed_var,
                grad_outputs=grad_sky.detach(),
                retain_graph=self.training,
                create_graph=self.training,
            )

            # Diagonal majorizer approximation in physical units for each parameter block
            with torch.no_grad():
                local_sens = _extract_local_sky_stamps(
                    sens_sky.detach(),
                    state.positions.detach(),
                    spatial_bounds_local_deg=spatial_bounds_local_deg,
                    stamp_size=self.stamp_size,
                )
                mean_obj_sens = torch.clamp(
                    torch.mean(local_sens, dim=(1, 2, 3)), min=1.0e-6
                )  # (N,)
                flux_sq = torch.square(torch.clamp(state.fluxes.detach(), min=1.0e-4))
                rad_sq = torch.square(torch.clamp(state.radii.detach(), min=cell_scale))
                diag_pos = mean_obj_sens * flux_sq / rad_sq
                diag_log_flux = mean_obj_sens * flux_sq
                diag_log_rad = mean_obj_sens * flux_sq
                diag_sed = mean_obj_sens.unsqueeze(-1) * flux_sq.unsqueeze(-1)
                diag_lat = mean_obj_sens.unsqueeze(-1) * flux_sq.unsqueeze(-1)
                diag_packed = torch.cat(
                    (
                        diag_pos.unsqueeze(-1).expand(-1, 2),
                        diag_log_flux.unsqueeze(-1),
                        diag_log_rad.unsqueeze(-1),
                        diag_sed.expand(-1, self.num_bands),
                        diag_lat.expand(-1, self.latent_dim),
                    ),
                    dim=-1,
                )
                med_diag = torch.clamp(
                    torch.median(diag_packed, dim=0, keepdim=True).values, min=1.0e-8
                )
                inv_diag = 1.0 / (diag_packed + float(damping) * med_diag)

            nat_grad = grad_packed * inv_diag
            # Clip natural gradient step to physical trust region
            nat_grad_clipped = torch.cat(
                (
                    torch.clamp(nat_grad[:, :2], min=-cell_scale, max=cell_scale),
                    torch.clamp(nat_grad[:, 2:], min=-0.5, max=0.5),
                ),
                dim=-1,
            )

            whitened_res_sky = -grad_sky.detach() / torch.sqrt(
                torch.clamp(sens_sky.detach(), min=1.0e-12)
            )
            res_stamps = _extract_local_sky_stamps(
                whitened_res_sky,
                state.positions.detach(),
                spatial_bounds_local_deg=spatial_bounds_local_deg,
                stamp_size=self.stamp_size,
            ).reshape(state.num_objects, -1)

            norm_features = torch.cat(
                (
                    packed_init,
                    nat_grad_clipped,
                    res_stamps,
                ),
                dim=-1,
            )
            node_h = self.input_proj(norm_features)
            msg, _ = _sparse_cech_mean_neighbor_messages(
                node_h, state.positions, state.radii, self.edge_mlp
            )
            if hidden.shape[0] != state.num_objects:
                hidden = torch.zeros(
                    (state.num_objects, self.hidden_dim), device=device, dtype=dtype
                )
            hidden = self.gru(node_h + msg, hidden)
            gate = torch.sigmoid(self.gate_head(hidden))
            learned_delta = 0.25 * torch.tanh(self.delta_head(hidden))
            # Scale position component of learned_delta by source radius
            pos_scale = torch.cat(
                (
                    state.radii.unsqueeze(-1).expand(-1, 2),
                    torch.ones(
                        (state.num_objects, packed_init.shape[1] - 2),
                        device=device,
                        dtype=dtype,
                    ),
                ),
                dim=-1,
            )
            full_step = -self.step_scale * (2.0 * gate) * nat_grad_clipped + (
                learned_delta * pos_scale
            )

            # Backtracking line search to guarantee monotone physical loss decrease
            accepted_state = replace(state, diffuse_grids=next_diffuse)
            _, best_loss, _, _ = _evaluate_sky_and_loss(
                accepted_state, det_weights_by_band, compute_stats=False
            )
            min_r_log = math.log(max(0.5 * cell_scale, 1.0e-5))
            max_r_log = math.log(max(0.5 * max(x_max - x_min, y_max - y_min), 1.0e-3))

            for bt in range(max(1, int(backtracking))):
                scale = 0.5**bt
                cand_packed = packed_init + scale * full_step
                cand_state = self._unpack_object_tensor(cand_packed, next_diffuse)
                cand_state = FactorizedObjectSceneState(
                    positions=torch.clamp(cand_state.positions, min=lower_xy, max=upper_xy),
                    log_fluxes=torch.clamp(cand_state.log_fluxes, min=-16.0, max=16.0),
                    log_radii=torch.clamp(cand_state.log_radii, min=min_r_log, max=max_r_log),
                    sed_coeffs=cand_state.sed_coeffs,
                    latents=cand_state.latents,
                    diffuse_grids=next_diffuse,
                )
                _, cand_loss, _, _ = _evaluate_sky_and_loss(
                    cand_state, det_weights_by_band, compute_stats=False
                )
                if math.isfinite(cand_loss) and (cand_loss <= best_loss or self.training):
                    accepted_state = cand_state
                    best_loss = cand_loss
                    break

            state = accepted_state
            loss_history.append(best_loss)

            if adapt_every > 0 and (step_idx + 1) % int(adapt_every) == 0 and step_idx < steps - 1:
                pos_grad_norm = torch.linalg.vector_norm(
                    nat_grad_clipped[:, :2] / torch.clamp(state.radii.unsqueeze(-1), min=1e-6),
                    dim=-1,
                )
                local_snr = torch.amax(
                    torch.abs(res_stamps.view(state.num_objects, -1)), dim=-1
                )
                state, event = adapt_factorized_object_scene(
                    state,
                    pos_grad_norm,
                    local_snr,
                    spatial_bounds_local_deg=spatial_bounds_local_deg,
                )
                adapt_events.append(event)

        final_sky = render_factorized_scene_grid(
            state,
            self.decoder,
            spatial_resolution=spatial_resolution,
            spatial_bounds_local_deg=spatial_bounds_local_deg,
        )
        return SceneOptimizationResult(
            state=state,
            sky_grids=final_sky,
            loss_history=tuple(loss_history),
            diagnostics={
                "initial_loss": float(loss_history[0]),
                "final_loss": float(loss_history[-1]),
                "num_objects_initial": int(initial_state.num_objects),
                "num_objects_final": int(state.num_objects),
                "steps": int(steps),
                "adaptation_events": adapt_events,
            },
        )


__all__ = [
    "CechSetProposalNetwork",
    "FactorizedObjectSceneState",
    "LocalObjectINRDecoder",
    "PreconditionedLearn2FoamOptimizer",
    "SceneOptimizationResult",
    "adapt_factorized_object_scene",
    "propose_factorized_objects_from_coadd",
    "render_factorized_scene_grid",
]
