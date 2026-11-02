"""Neural field-model classes for :mod:`torchsky.map.field`."""

from __future__ import annotations

from typing import Sequence, cast
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ._core_helpers import (
    _BaseFieldModel,
    _atlas_expert_weights,
    _atlas_local_coords,
    _point_bandpass_features,
    _reset_xavier_mlp,
    _smoothness_probe_grid,
)
from .bandpass import (
    bandpass_bounds as _bandpass_bounds,
    bandpass_response as _bandpass_response,
)
from .hashgrid import sample_feature_plane
from .losses import soft_bound_logvar
from .networks import (
    ResidualStack,
)
from .types import (
    FieldDomain,
    FieldObservation,
)


def _triplane_names(spatial_dim: int) -> tuple[str, ...]:
    if spatial_dim == 3:
        return ("xy", "xl", "yl", "xz", "yz", "zl")
    return ("xy", "xl", "yl")


def _sample_triplane(
    planes: nn.ParameterDict,
    *,
    prefix: str,
    x: Tensor,
    y: Tensor,
    lam_unit: Tensor,
    z: Tensor | None,
) -> list[Tensor]:
    sampled = [
        sample_feature_plane(planes[f"{prefix}xy"], x, y),
        sample_feature_plane(planes[f"{prefix}xl"], x, lam_unit),
        sample_feature_plane(planes[f"{prefix}yl"], y, lam_unit),
    ]
    if z is not None:
        sampled.extend(
            [
                sample_feature_plane(planes[f"{prefix}xz"], x, z),
                sample_feature_plane(planes[f"{prefix}yz"], y, z),
                sample_feature_plane(planes[f"{prefix}zl"], z, lam_unit),
            ]
        )
    return sampled


class SkyLambdaPlanesField(_BaseFieldModel):
    def __init__(
        self,
        domain: FieldDomain,
        *,
        plane_resolution: int = 32,
        features_per_plane: int = 8,
        hidden_dim: int = 64,
        hidden_layers: int = 2,
    ) -> None:
        super().__init__(domain)
        self.spatial_dim = 2 if self.coord_mode == "local_tangent" else 3
        self.plane_resolution = int(plane_resolution)
        self.features_per_plane = int(features_per_plane)
        plane_names = _triplane_names(self.spatial_dim)
        self.planes = nn.ParameterDict(
            {
                name: nn.Parameter(
                    1.0e-3
                    * torch.randn(
                        self.features_per_plane,
                        self.plane_resolution,
                        self.plane_resolution,
                    )
                )
                for name in plane_names
            }
        )
        in_dim = len(plane_names) * self.features_per_plane + self.spatial_dim + 1
        layers: list[nn.Module] = []
        current_dim = in_dim
        for _ in range(max(int(hidden_layers), 1)):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.SiLU())
            current_dim = hidden_dim
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(current_dim, 1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        _reset_xavier_mlp(self.hidden, self.output)

    def forward(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        spatial_raw = self._spatial_coords(
            ra_deg,
            dec_deg,
            wavelength_angstrom,
        ).to(dtype=self.output.weight.dtype)
        spatial_unit = self._spatial_unit_box_coords(
            ra_deg,
            dec_deg,
            wavelength_angstrom,
        ).to(dtype=self.output.weight.dtype)
        lam = self._normalized_wavelength(
            wavelength_angstrom,
            device=ra_deg.device,
            dtype=self.output.weight.dtype,
            shape=ra_deg.shape,
        )
        lam_unit = 0.5 * (torch.tanh(lam) + 1.0)
        x = spatial_unit[..., 0]
        y = spatial_unit[..., 1]
        z = spatial_unit[..., 2] if self.spatial_dim == 3 else None
        sampled = _sample_triplane(
            self.planes, prefix="", x=x, y=y, lam_unit=lam_unit, z=z
        )
        plane_feats = torch.cat(sampled, dim=-1)
        feats = torch.cat([spatial_raw, lam.unsqueeze(-1), plane_feats], dim=-1)
        out = self.output(self.hidden(feats)).squeeze(-1)
        return F.softplus(out) + 1.0e-4


class SkyLambdaKPlanesField(_BaseFieldModel):
    def __init__(
        self,
        domain: FieldDomain,
        *,
        base_resolution: int = 16,
        num_scales: int = 3,
        features_per_plane: int = 8,
        hidden_dim: int = 64,
        hidden_layers: int = 2,
        refine_hidden_dim: int = 32,
    ) -> None:
        super().__init__(domain)
        self.spatial_dim = 2 if self.coord_mode == "local_tangent" else 3
        self.base_resolution = int(base_resolution)
        self.num_scales = int(num_scales)
        self.features_per_plane = int(features_per_plane)
        plane_names = _triplane_names(self.spatial_dim)
        self._plane_names = tuple(plane_names)
        self.planes = nn.ParameterDict()
        for scale_idx in range(self.num_scales):
            resolution = self.base_resolution * (2**scale_idx)
            for name in self._plane_names:
                self.planes[f"s{scale_idx}_{name}"] = nn.Parameter(
                    1.0e-3
                    * torch.randn(
                        self.features_per_plane,
                        resolution,
                        resolution,
                    )
                )
        in_dim = (
            len(self._plane_names) * self.num_scales * self.features_per_plane
            + self.spatial_dim
            + 1
        )
        layers: list[nn.Module] = []
        current_dim = in_dim
        for _ in range(max(int(hidden_layers), 1)):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.SiLU())
            current_dim = hidden_dim
        self.hidden = nn.Sequential(*layers)
        self.base_output = nn.Linear(current_dim, 1)
        self.refine = nn.Sequential(
            nn.Linear(current_dim, refine_hidden_dim),
            nn.SiLU(),
            nn.Linear(refine_hidden_dim, 1),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.hidden:
            if type(module) is nn.Linear:
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.uniform_(self.base_output.weight, -1e-2, 1e-2)
        nn.init.constant_(self.base_output.bias, -3.0)
        for module in self.refine:
            if type(module) is nn.Linear:
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def _hidden_features(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> tuple[Tensor, Tensor]:
        spatial_raw = self._spatial_coords(
            ra_deg,
            dec_deg,
            wavelength_angstrom,
        ).to(dtype=self.base_output.weight.dtype)
        spatial_unit = self._spatial_unit_box_coords(
            ra_deg,
            dec_deg,
            wavelength_angstrom,
        ).to(dtype=self.base_output.weight.dtype)
        lam = self._normalized_wavelength(
            wavelength_angstrom,
            device=ra_deg.device,
            dtype=self.base_output.weight.dtype,
            shape=ra_deg.shape,
        )
        lam_unit = 0.5 * (torch.tanh(lam) + 1.0)
        x = spatial_unit[..., 0]
        y = spatial_unit[..., 1]
        z = spatial_unit[..., 2] if self.spatial_dim == 3 else None
        sampled: list[Tensor] = []
        for scale_idx in range(self.num_scales):
            sampled.extend(
                _sample_triplane(
                    self.planes,
                    prefix=f"s{scale_idx}_",
                    x=x,
                    y=y,
                    lam_unit=lam_unit,
                    z=z,
                )
            )
        plane_feats = torch.cat(sampled, dim=-1)
        feats = torch.cat([spatial_raw, lam.unsqueeze(-1), plane_feats], dim=-1)
        hidden = self.hidden(feats)
        return hidden, lam

    def _decode_hidden(
        self,
        hidden: Tensor,
    ) -> Tensor:
        out = self.base_output(hidden).squeeze(-1) + self.refine(hidden).squeeze(-1)
        return F.softplus(out) + 1.0e-4

    def render_bandpass(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelengths: Tensor,
        weights: Tensor,
    ) -> Tensor:
        hidden_planes = []
        for wavelength in wavelengths:
            hidden, _ = self._hidden_features(ra_deg, dec_deg, wavelength)
            hidden_planes.append(hidden)
        stacked_hidden = torch.stack(hidden_planes, dim=0)
        norm_weights = weights.to(dtype=stacked_hidden.dtype)
        norm_weights = norm_weights / torch.clamp(norm_weights.sum(), min=1.0e-8)
        mixed_hidden = torch.sum(
            stacked_hidden * norm_weights.view(-1, *([1] * (stacked_hidden.ndim - 1))),
            dim=0,
        )
        return self._decode_hidden(mixed_hidden)

    def forward(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        hidden, _ = self._hidden_features(
            ra_deg,
            dec_deg,
            wavelength_angstrom,
        )
        return self._decode_hidden(hidden)


class SkyLambdaFeatureField(_BaseFieldModel):
    def __init__(
        self,
        domain: FieldDomain,
        *,
        base_resolution: int = 16,
        num_scales: int = 3,
        features_per_plane: int = 8,
        scale_hidden_dim: int = 64,
        scale_num_blocks: int = 2,
        decoder_hidden_dim: int = 64,
        decoder_num_blocks: int = 2,
        refine_hidden_dim: int = 32,
        bandpass_hidden_dim: int = 32,
        global_hidden_dim: int = 0,
        global_num_blocks: int = 1,
        aggregate_hidden_moments: bool = False,
        coarse_fine_readout: bool = False,
        predict_variance: bool = False,
    ) -> None:
        super().__init__(domain)
        self.spatial_dim = 2 if self.coord_mode == "local_tangent" else 3
        self.base_resolution = int(base_resolution)
        self.num_scales = int(num_scales)
        self.features_per_plane = int(features_per_plane)
        self.scale_hidden_dim = int(scale_hidden_dim)
        self.bandpass_hidden_dim = int(bandpass_hidden_dim)
        self.global_hidden_dim = max(int(global_hidden_dim), 0)
        self.aggregate_hidden_moments = bool(aggregate_hidden_moments)
        self.coarse_fine_readout = bool(coarse_fine_readout)
        self.predict_variance = bool(predict_variance)
        self.predictive_distribution_available = self.predict_variance
        self._curriculum_progress = 1.0
        plane_names = _triplane_names(self.spatial_dim)
        self._plane_names = tuple(plane_names)
        self.planes = nn.ParameterDict()
        for scale_idx in range(self.num_scales):
            resolution = self.base_resolution * (2**scale_idx)
            for name in self._plane_names:
                self.planes[f"s{scale_idx}_{name}"] = nn.Parameter(
                    1.0e-3
                    * torch.randn(
                        self.features_per_plane,
                        resolution,
                        resolution,
                    )
                )
        per_scale_dim = (
            len(self._plane_names) * self.features_per_plane + self.spatial_dim + 1
        )
        self.scale_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(per_scale_dim, self.scale_hidden_dim),
                    nn.SiLU(),
                    ResidualStack(self.scale_hidden_dim, int(scale_num_blocks)),
                )
                for _ in range(self.num_scales)
            ]
        )
        self.scale_gate = nn.Sequential(
            nn.Linear(self.spatial_dim + 1, self.scale_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.scale_hidden_dim, self.num_scales),
        )
        self.bandpass_context = nn.Sequential(
            nn.Linear(4 + 8, self.bandpass_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.bandpass_hidden_dim, self.bandpass_hidden_dim),
            nn.SiLU(),
        )
        if self.global_hidden_dim > 0:
            self.global_encoder = nn.Sequential(
                nn.Linear(self.spatial_dim + 1, self.global_hidden_dim),
                nn.SiLU(),
                ResidualStack(self.global_hidden_dim, int(global_num_blocks)),
            )
        else:
            self.global_encoder = None
        hidden_multiplier = 2 if self.aggregate_hidden_moments else 1
        coarse_multiplier = hidden_multiplier if self.coarse_fine_readout else 0
        self.decoder = nn.Sequential(
            nn.Linear(
                hidden_multiplier * self.scale_hidden_dim
                + coarse_multiplier * self.scale_hidden_dim
                + self.bandpass_hidden_dim
                + hidden_multiplier * self.global_hidden_dim,
                int(decoder_hidden_dim),
            ),
            nn.SiLU(),
            ResidualStack(int(decoder_hidden_dim), int(decoder_num_blocks)),
        )
        self.base_output = nn.Linear(int(decoder_hidden_dim), 1)
        self.refine = nn.Sequential(
            nn.Linear(int(decoder_hidden_dim), int(refine_hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(refine_hidden_dim), 1),
        )
        if self.predict_variance:
            self.logvar_output = nn.Linear(int(decoder_hidden_dim), 1)
            self.logvar_refine = nn.Sequential(
                nn.Linear(int(decoder_hidden_dim), int(refine_hidden_dim)),
                nn.SiLU(),
                nn.Linear(int(refine_hidden_dim), 1),
            )
        else:
            self.logvar_output = None
            self.logvar_refine = None
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for mlp in self.scale_mlps:
            for module in cast(nn.Sequential, mlp):  # each scale MLP is an nn.Sequential
                if type(module) is nn.Linear:
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
        for module in self.scale_gate:
            if type(module) is nn.Linear:
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for module in self.bandpass_context:
            if type(module) is nn.Linear:
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        if self.global_encoder is not None:
            for module in self.global_encoder:
                if type(module) is nn.Linear:
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
        for module in self.decoder:
            if type(module) is nn.Linear:
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.uniform_(self.base_output.weight, -1e-2, 1e-2)
        nn.init.constant_(self.base_output.bias, -3.0)
        for module in self.refine:
            if type(module) is nn.Linear:
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        if self.logvar_output is not None:
            nn.init.uniform_(self.logvar_output.weight, -1e-2, 1e-2)
            nn.init.constant_(self.logvar_output.bias, -2.0)
        if self.logvar_refine is not None:
            for module in self.logvar_refine:
                if type(module) is nn.Linear:
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)

    def set_curriculum_progress(self, progress: float) -> None:
        self._curriculum_progress = min(max(float(progress), 0.0), 1.0)

    def _curriculum_scale_weights(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        eased_progress = math.sqrt(self._curriculum_progress)
        active_extent = 1.0 + (self.num_scales - 1) * eased_progress
        scale_index = torch.arange(
            self.num_scales,
            device=device,
            dtype=dtype,
        )
        return torch.clamp(active_extent - scale_index, min=0.0, max=1.0)

    def _per_scale_hidden_from_components(
        self,
        spatial_raw: Tensor,
        spatial_unit: Tensor,
        lam: Tensor,
    ) -> tuple[Tensor, Tensor]:
        lam_unit = 0.5 * (torch.tanh(lam) + 1.0)
        x = spatial_unit[..., 0]
        y = spatial_unit[..., 1]
        z = spatial_unit[..., 2] if self.spatial_dim == 3 else None
        context = torch.cat([spatial_raw, lam.unsqueeze(-1)], dim=-1)
        per_scale_hidden: list[Tensor] = []
        for scale_idx, mlp in enumerate(self.scale_mlps):
            sampled = _sample_triplane(
                self.planes,
                prefix=f"s{scale_idx}_",
                x=x,
                y=y,
                lam_unit=lam_unit,
                z=z,
            )
            scale_feats = torch.cat([context, torch.cat(sampled, dim=-1)], dim=-1)
            per_scale_hidden.append(mlp(scale_feats))
        scale_hidden = torch.stack(per_scale_hidden, dim=-2)
        return scale_hidden, context

    def _aggregate_scale_stack(
        self,
        scale_hidden: Tensor,
        context: Tensor,
    ) -> Tensor:
        gate_logits = self.scale_gate(context)
        scale_weights = self._curriculum_scale_weights(
            device=gate_logits.device,
            dtype=gate_logits.dtype,
        )
        gate_logits = gate_logits + torch.log(torch.clamp(scale_weights, min=1.0e-6))
        gate = torch.softmax(gate_logits, dim=-1).unsqueeze(-1)
        return torch.sum(scale_hidden * gate, dim=-2)

    def _scale_hidden_from_components(
        self,
        spatial_raw: Tensor,
        spatial_unit: Tensor,
        lam: Tensor,
    ) -> Tensor:
        scale_hidden, context = self._per_scale_hidden_from_components(
            spatial_raw,
            spatial_unit,
            lam,
        )
        return self._aggregate_scale_stack(scale_hidden, context)

    def _global_hidden_from_components(
        self,
        spatial_raw: Tensor,
        lam: Tensor,
    ) -> Tensor | None:
        if self.global_encoder is None:
            return None
        context = torch.cat([spatial_raw, lam.unsqueeze(-1)], dim=-1)
        return self.global_encoder(context)

    def _global_hidden(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor | None:
        spatial_raw = self._spatial_coords(
            ra_deg,
            dec_deg,
            wavelength_angstrom,
        ).to(dtype=self.base_output.weight.dtype)
        lam = self._normalized_wavelength(
            wavelength_angstrom,
            device=ra_deg.device,
            dtype=self.base_output.weight.dtype,
            shape=ra_deg.shape,
        )
        return self._global_hidden_from_components(spatial_raw, lam)

    def _bandpass_features(self, wavelengths: Tensor, weights: Tensor) -> Tensor:
        lam = (
            wavelengths.to(dtype=self.base_output.weight.dtype)
            - float(self.domain.wavelength_center)
        ) / float(self.domain.wavelength_scale)
        norm_weights = weights.to(dtype=self.base_output.weight.dtype)
        norm_weights = norm_weights / torch.clamp(norm_weights.sum(), min=1.0e-8)
        mean = torch.sum(norm_weights * lam)
        centered = lam - mean
        # Element-wise multiplication is faster than torch.square or **2
        centered_sq = centered * centered
        second = torch.sum(norm_weights * centered_sq)
        third = torch.sum(norm_weights * centered_sq * centered)
        fourth = torch.sum(norm_weights * centered_sq * centered_sq)
        feats = [mean, second, third, fourth]
        scales = torch.arange(
            1,
            5,
            device=lam.device,
            dtype=lam.dtype,
        )
        phase = math.pi * scales.view(-1, *([1] * lam.ndim)) * lam
        sin_sum = torch.sum(
            norm_weights * torch.sin(phase), dim=tuple(range(1, 1 + lam.ndim))
        )
        cos_sum = torch.sum(
            norm_weights * torch.cos(phase), dim=tuple(range(1, 1 + lam.ndim))
        )
        freq_feats = torch.stack([sin_sum, cos_sum], dim=1).flatten()
        return torch.cat([torch.stack(feats), freq_feats])

    def _decode_hidden(
        self,
        hidden: Tensor,
        bandpass_latent: Tensor,
        global_hidden: Tensor | None = None,
        coarse_hidden: Tensor | None = None,
    ) -> Tensor:
        if bandpass_latent.ndim < hidden.ndim:
            bandpass_latent = bandpass_latent.view(
                *([1] * (hidden.ndim - 1)),
                bandpass_latent.shape[-1],
            )
            bandpass_latent = bandpass_latent.expand(*hidden.shape[:-1], -1)
        parts = [hidden]
        if coarse_hidden is not None:
            parts.append(coarse_hidden)
        parts.append(bandpass_latent)
        if global_hidden is not None:
            parts.append(global_hidden)
        decoded = self.decoder(torch.cat(parts, dim=-1))
        out = self.base_output(decoded).squeeze(-1) + self.refine(decoded).squeeze(-1)
        return F.softplus(out) + 1.0e-4

    def _decode_distribution(
        self,
        hidden: Tensor,
        bandpass_latent: Tensor,
        global_hidden: Tensor | None = None,
        coarse_hidden: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if bandpass_latent.ndim < hidden.ndim:
            bandpass_latent = bandpass_latent.view(
                *([1] * (hidden.ndim - 1)),
                bandpass_latent.shape[-1],
            )
            bandpass_latent = bandpass_latent.expand(*hidden.shape[:-1], -1)
        parts = [hidden]
        if coarse_hidden is not None:
            parts.append(coarse_hidden)
        parts.append(bandpass_latent)
        if global_hidden is not None:
            parts.append(global_hidden)
        decoded = self.decoder(torch.cat(parts, dim=-1))
        mean = self.base_output(decoded).squeeze(-1) + self.refine(decoded).squeeze(-1)
        mean = F.softplus(mean) + 1.0e-4
        if self.logvar_output is None or self.logvar_refine is None:
            raise ValueError("predictive variance is disabled for this field model")
        logvar = self.logvar_output(decoded).squeeze(-1) + self.logvar_refine(
            decoded
        ).squeeze(-1)
        logvar = soft_bound_logvar(logvar)
        return mean, logvar

    def _aggregate_hidden_stack(
        self,
        stacked_hidden: Tensor,
        norm_weights: Tensor,
    ) -> Tensor:
        mean_hidden = torch.sum(
            stacked_hidden * norm_weights.view(-1, *([1] * (stacked_hidden.ndim - 1))),
            dim=0,
        )
        if not self.aggregate_hidden_moments:
            return mean_hidden
        centered = stacked_hidden - mean_hidden.unsqueeze(0)
        # ⚡ Bolt: Element-wise multiplication is faster than torch.square or **2
        variance = torch.sum(
            (centered * centered)
            * norm_weights.view(-1, *([1] * (stacked_hidden.ndim - 1))),
            dim=0,
        )
        std_hidden = torch.sqrt(torch.clamp(variance, min=1.0e-8))
        return torch.cat([mean_hidden, std_hidden], dim=-1)

    def render_bandpass(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelengths: Tensor,
        weights: Tensor,
    ) -> Tensor:
        hidden_planes = []
        coarse_hidden_planes = []
        global_hiddens = []
        for wavelength in wavelengths:
            spatial_raw = self._spatial_coords(
                ra_deg,
                dec_deg,
                wavelength,
            ).to(dtype=self.base_output.weight.dtype)
            spatial_unit = self._spatial_unit_box_coords(
                ra_deg,
                dec_deg,
                wavelength,
            ).to(dtype=self.base_output.weight.dtype)
            lam = self._normalized_wavelength(
                wavelength,
                device=ra_deg.device,
                dtype=self.base_output.weight.dtype,
                shape=ra_deg.shape,
            )
            scale_hidden, context = self._per_scale_hidden_from_components(
                spatial_raw,
                spatial_unit,
                lam,
            )
            hidden_planes.append(self._aggregate_scale_stack(scale_hidden, context))
            if self.coarse_fine_readout:
                coarse_hidden_planes.append(scale_hidden[..., 0, :])
            global_hidden = self._global_hidden(ra_deg, dec_deg, wavelength)
            if global_hidden is not None:
                global_hiddens.append(global_hidden)
        stacked_hidden = torch.stack(hidden_planes, dim=0)
        norm_weights = weights.to(dtype=stacked_hidden.dtype)
        norm_weights = norm_weights / torch.clamp(norm_weights.sum(), min=1.0e-8)
        mixed_hidden = self._aggregate_hidden_stack(stacked_hidden, norm_weights)
        bandpass_latent = self.bandpass_context(
            self._bandpass_features(wavelengths, weights)
        )
        mixed_coarse_hidden = None
        if coarse_hidden_planes:
            stacked_coarse_hidden = torch.stack(coarse_hidden_planes, dim=0)
            mixed_coarse_hidden = self._aggregate_hidden_stack(
                stacked_coarse_hidden,
                norm_weights,
            )
        mixed_global_hidden = None
        if global_hiddens:
            stacked_global_hidden = torch.stack(global_hiddens, dim=0)
            mixed_global_hidden = self._aggregate_hidden_stack(
                stacked_global_hidden,
                norm_weights,
            )
        return self._decode_hidden(
            mixed_hidden,
            bandpass_latent,
            mixed_global_hidden,
            mixed_coarse_hidden,
        )

    def render_bandpass_distribution(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelengths: Tensor,
        weights: Tensor,
    ) -> tuple[Tensor, Tensor]:
        hidden_planes = []
        coarse_hidden_planes = []
        global_hiddens = []
        for wavelength in wavelengths:
            spatial_raw = self._spatial_coords(
                ra_deg,
                dec_deg,
                wavelength,
            ).to(dtype=self.base_output.weight.dtype)
            spatial_unit = self._spatial_unit_box_coords(
                ra_deg,
                dec_deg,
                wavelength,
            ).to(dtype=self.base_output.weight.dtype)
            lam = self._normalized_wavelength(
                wavelength,
                device=ra_deg.device,
                dtype=self.base_output.weight.dtype,
                shape=ra_deg.shape,
            )
            scale_hidden, context = self._per_scale_hidden_from_components(
                spatial_raw,
                spatial_unit,
                lam,
            )
            hidden_planes.append(self._aggregate_scale_stack(scale_hidden, context))
            if self.coarse_fine_readout:
                coarse_hidden_planes.append(scale_hidden[..., 0, :])
            global_hidden = self._global_hidden(ra_deg, dec_deg, wavelength)
            if global_hidden is not None:
                global_hiddens.append(global_hidden)
        stacked_hidden = torch.stack(hidden_planes, dim=0)
        norm_weights = weights.to(dtype=stacked_hidden.dtype)
        norm_weights = norm_weights / torch.clamp(norm_weights.sum(), min=1.0e-8)
        mixed_hidden = self._aggregate_hidden_stack(stacked_hidden, norm_weights)
        bandpass_latent = self.bandpass_context(
            self._bandpass_features(wavelengths, weights)
        )
        mixed_coarse_hidden = None
        if coarse_hidden_planes:
            stacked_coarse_hidden = torch.stack(coarse_hidden_planes, dim=0)
            mixed_coarse_hidden = self._aggregate_hidden_stack(
                stacked_coarse_hidden,
                norm_weights,
            )
        mixed_global_hidden = None
        if global_hiddens:
            stacked_global_hidden = torch.stack(global_hiddens, dim=0)
            mixed_global_hidden = self._aggregate_hidden_stack(
                stacked_global_hidden,
                norm_weights,
            )
        return self._decode_distribution(
            mixed_hidden,
            bandpass_latent,
            mixed_global_hidden,
            mixed_coarse_hidden,
        )

    def forward(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        spatial_raw = self._spatial_coords(
            ra_deg,
            dec_deg,
            wavelength_angstrom,
        ).to(dtype=self.base_output.weight.dtype)
        spatial_unit = self._spatial_unit_box_coords(
            ra_deg,
            dec_deg,
            wavelength_angstrom,
        ).to(dtype=self.base_output.weight.dtype)
        lam = self._normalized_wavelength(
            wavelength_angstrom,
            device=ra_deg.device,
            dtype=self.base_output.weight.dtype,
            shape=ra_deg.shape,
        )
        scale_hidden, context = self._per_scale_hidden_from_components(
            spatial_raw,
            spatial_unit,
            lam,
        )
        hidden = self._aggregate_scale_stack(scale_hidden, context)
        one_weight = torch.ones(1, device=hidden.device, dtype=hidden.dtype)
        hidden = self._aggregate_hidden_stack(hidden.unsqueeze(0), one_weight)
        coarse_hidden = (
            self._aggregate_hidden_stack(
                scale_hidden[..., 0, :].unsqueeze(0), one_weight
            )
            if self.coarse_fine_readout
            else None
        )
        global_hidden = self._global_hidden(ra_deg, dec_deg, wavelength_angstrom)
        if global_hidden is not None:
            global_hidden = self._aggregate_hidden_stack(
                global_hidden.unsqueeze(0), one_weight
            )
        bandpass_latent = self.bandpass_context(_point_bandpass_features(lam, 4))
        return self._decode_hidden(
            hidden, bandpass_latent, global_hidden, coarse_hidden
        )

    def forward_distribution(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> tuple[Tensor, Tensor]:
        spatial_raw = self._spatial_coords(
            ra_deg,
            dec_deg,
            wavelength_angstrom,
        ).to(dtype=self.base_output.weight.dtype)
        spatial_unit = self._spatial_unit_box_coords(
            ra_deg,
            dec_deg,
            wavelength_angstrom,
        ).to(dtype=self.base_output.weight.dtype)
        lam = self._normalized_wavelength(
            wavelength_angstrom,
            device=ra_deg.device,
            dtype=self.base_output.weight.dtype,
            shape=ra_deg.shape,
        )
        scale_hidden, context = self._per_scale_hidden_from_components(
            spatial_raw,
            spatial_unit,
            lam,
        )
        hidden = self._aggregate_scale_stack(scale_hidden, context)
        one_weight = torch.ones(1, device=hidden.device, dtype=hidden.dtype)
        hidden = self._aggregate_hidden_stack(hidden.unsqueeze(0), one_weight)
        coarse_hidden = (
            self._aggregate_hidden_stack(
                scale_hidden[..., 0, :].unsqueeze(0), one_weight
            )
            if self.coarse_fine_readout
            else None
        )
        global_hidden = self._global_hidden(ra_deg, dec_deg, wavelength_angstrom)
        if global_hidden is not None:
            global_hidden = self._aggregate_hidden_stack(
                global_hidden.unsqueeze(0), one_weight
            )
        bandpass_latent = self.bandpass_context(_point_bandpass_features(lam, 4))
        return self._decode_distribution(
            hidden,
            bandpass_latent,
            global_hidden,
            coarse_hidden,
        )


class OperatorConstrainedFeatureField(_BaseFieldModel):
    spectral_basis_centers: Tensor
    spectral_basis_widths: Tensor
    spectral_log_center: Tensor
    spectral_log_scale: Tensor

    def __init__(
        self,
        domain: FieldDomain,
        *,
        base_resolution: int = 16,
        num_scales: int = 4,
        features_per_plane: int = 12,
        scale_hidden_dim: int = 96,
        scale_num_blocks: int = 2,
        spectral_basis_dim: int = 8,
        decoder_hidden_dim: int = 96,
        decoder_num_blocks: int = 2,
        refine_hidden_dim: int = 48,
        bandpass_hidden_dim: int = 48,
        spectral_context_dim: int | None = None,
        residual_hidden_dim: int = 48,
    ) -> None:
        super().__init__(domain)
        self.spatial_dim = 2 if self.coord_mode == "local_tangent" else 3
        self.base_resolution = int(base_resolution)
        self.num_scales = int(num_scales)
        self.features_per_plane = int(features_per_plane)
        self.scale_hidden_dim = int(scale_hidden_dim)
        self.spectral_basis_dim = int(spectral_basis_dim)
        self.bandpass_hidden_dim = int(bandpass_hidden_dim)
        self.spectral_context_dim = (
            int(spectral_context_dim)
            if spectral_context_dim is not None
            else self.scale_hidden_dim
        )
        self._spectral_observability_cache: dict[
            tuple[tuple[object, ...], ...], Tensor
        ] = {}
        spatial_plane_names = ["xy"] if self.spatial_dim == 2 else ["xy", "xz", "yz"]
        self._spatial_plane_names = tuple(spatial_plane_names)
        self.planes = nn.ParameterDict()
        for scale_idx in range(self.num_scales):
            resolution = self.base_resolution * (2**scale_idx)
            for name in self._spatial_plane_names:
                self.planes[f"s{scale_idx}_{name}"] = nn.Parameter(
                    1.0e-3
                    * torch.randn(
                        self.features_per_plane,
                        resolution,
                        resolution,
                    )
                )
        per_scale_dim = (
            len(self._spatial_plane_names) * self.features_per_plane + self.spatial_dim
        )
        self.scale_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(per_scale_dim, self.scale_hidden_dim),
                    nn.SiLU(),
                    ResidualStack(self.scale_hidden_dim, int(scale_num_blocks)),
                )
                for _ in range(self.num_scales)
            ]
        )
        self.scale_gate = nn.Sequential(
            nn.Linear(self.spatial_dim, self.scale_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.scale_hidden_dim, self.num_scales),
        )
        self.coefficient_head = nn.Sequential(
            nn.Linear(self.scale_hidden_dim, self.scale_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.scale_hidden_dim, self.spectral_basis_dim),
        )
        self.spectral_context = nn.Sequential(
            nn.Linear(2 * self.spectral_basis_dim, self.spectral_context_dim),
            nn.SiLU(),
            nn.Linear(self.spectral_context_dim, self.spectral_context_dim),
            nn.SiLU(),
        )
        self.spectral_affine = nn.Sequential(
            nn.Linear(self.spectral_context_dim, self.scale_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.scale_hidden_dim, 2 * self.scale_hidden_dim),
        )
        self.bandpass_context = nn.Sequential(
            nn.Linear(4 + 8, self.bandpass_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.bandpass_hidden_dim, self.bandpass_hidden_dim),
            nn.SiLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(
                2 * self.scale_hidden_dim
                + self.spectral_context_dim
                + self.bandpass_hidden_dim,
                int(decoder_hidden_dim),
            ),
            nn.SiLU(),
            ResidualStack(int(decoder_hidden_dim), int(decoder_num_blocks)),
        )
        self.base_output = nn.Linear(int(decoder_hidden_dim), 1)
        self.refine = nn.Sequential(
            nn.Linear(int(decoder_hidden_dim), int(refine_hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(refine_hidden_dim), 1),
        )
        self.residual_affine = nn.Sequential(
            nn.Linear(
                self.bandpass_hidden_dim
                + self.spectral_context_dim
                + self.scale_hidden_dim,
                int(residual_hidden_dim),
            ),
            nn.SiLU(),
            nn.Linear(int(residual_hidden_dim), 2 * self.scale_hidden_dim),
        )
        self.residual_decoder = nn.Sequential(
            nn.Linear(
                2 * self.scale_hidden_dim
                + self.spectral_context_dim
                + self.bandpass_hidden_dim,
                int(residual_hidden_dim),
            ),
            nn.SiLU(),
            nn.Linear(int(residual_hidden_dim), 1),
        )
        self._register_spectral_basis()
        self._reset_parameters()

    def _register_spectral_basis(self) -> None:
        lower = max(
            float(self.domain.wavelength_center) - float(self.domain.wavelength_scale),
            1.0,
        )
        upper = max(
            float(self.domain.wavelength_center) + float(self.domain.wavelength_scale),
            lower + 1.0,
        )
        log_center = math.log(float(self.domain.wavelength_center))
        log_scale = max(0.5 * (math.log(upper) - math.log(lower)), 1.0e-3)
        coarse_count = max(1, self.spectral_basis_dim // 3)
        medium_count = max(1, self.spectral_basis_dim // 3)
        fine_count = max(self.spectral_basis_dim - coarse_count - medium_count, 1)
        coarse_centers = torch.linspace(
            -1.0, 1.0, steps=coarse_count, dtype=torch.float32
        )
        medium_centers = torch.linspace(
            -1.0, 1.0, steps=medium_count, dtype=torch.float32
        )
        fine_centers = torch.linspace(-1.0, 1.0, steps=fine_count, dtype=torch.float32)
        centers = torch.cat([coarse_centers, medium_centers, fine_centers], dim=0)[
            : self.spectral_basis_dim
        ]
        widths = torch.cat(
            [
                torch.full((coarse_count,), 0.95, dtype=torch.float32),
                torch.full((medium_count,), 0.45, dtype=torch.float32),
                torch.full((fine_count,), 0.18, dtype=torch.float32),
            ],
            dim=0,
        )[: self.spectral_basis_dim]
        self.register_buffer("spectral_basis_centers", centers, persistent=False)
        self.register_buffer("spectral_basis_widths", widths, persistent=False)
        self.register_buffer(
            "spectral_log_center",
            torch.tensor(log_center, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "spectral_log_scale",
            torch.tensor(log_scale, dtype=torch.float32),
            persistent=False,
        )

    def _reset_parameters(self) -> None:
        for mlp in self.scale_mlps:
            for module in cast(nn.Sequential, mlp):  # each scale MLP is an nn.Sequential
                if type(module) is nn.Linear:
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
        for module in self.scale_gate:
            if type(module) is nn.Linear:
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for module in self.coefficient_head:
            if type(module) is nn.Linear:
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for module in self.spectral_context:
            if type(module) is nn.Linear:
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for module in self.spectral_affine:
            if type(module) is nn.Linear:
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for module in self.bandpass_context:
            if type(module) is nn.Linear:
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for module in self.decoder:
            if type(module) is nn.Linear:
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for module in self.refine:
            if type(module) is nn.Linear:
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for module in self.residual_affine:
            if type(module) is nn.Linear:
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for module in self.residual_decoder[:-1]:
            if type(module) is nn.Linear:
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.uniform_(self.base_output.weight, -1.0e-2, 1.0e-2)
        nn.init.constant_(self.base_output.bias, -3.0)
        final_refine = self.refine[-1]
        assert isinstance(final_refine, nn.Linear)
        nn.init.zeros_(final_refine.weight)
        nn.init.zeros_(final_refine.bias)
        residual_out = self.residual_decoder[-1]
        assert isinstance(residual_out, nn.Linear)
        nn.init.zeros_(residual_out.weight)
        nn.init.zeros_(residual_out.bias)

    def _spatial_feature_stack(
        self,
        spatial_raw: Tensor,
        spatial_unit: Tensor,
    ) -> Tensor:
        x = spatial_unit[..., 0]
        y = spatial_unit[..., 1]
        per_scale_hidden: list[Tensor] = []
        for scale_idx, mlp in enumerate(self.scale_mlps):
            sampled = [sample_feature_plane(self.planes[f"s{scale_idx}_xy"], x, y)]
            if self.spatial_dim == 3:
                z = spatial_unit[..., 2]
                sampled.extend(
                    [
                        sample_feature_plane(self.planes[f"s{scale_idx}_xz"], x, z),
                        sample_feature_plane(self.planes[f"s{scale_idx}_yz"], y, z),
                    ]
                )
            feats = torch.cat([spatial_raw, torch.cat(sampled, dim=-1)], dim=-1)
            per_scale_hidden.append(mlp(feats))
        return torch.stack(per_scale_hidden, dim=-2)

    def _shared_hidden_components_from_spatial(
        self,
        spatial_raw: Tensor,
        spatial_unit: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        scale_hidden = self._spatial_feature_stack(spatial_raw, spatial_unit)
        gate_logits = self.scale_gate(spatial_raw)
        gate = torch.softmax(gate_logits, dim=-1).unsqueeze(-1)
        shared_hidden = torch.sum(scale_hidden * gate, dim=-2)
        if self.num_scales == 1:
            return shared_hidden, shared_hidden, torch.zeros_like(shared_hidden)
        scale_positions = torch.linspace(
            0.0,
            1.0,
            steps=self.num_scales,
            device=scale_hidden.device,
            dtype=scale_hidden.dtype,
        ).view(*([1] * (gate.ndim - 2)), self.num_scales, 1)
        coarse_gate = gate * (1.0 - scale_positions)
        fine_gate = gate * scale_positions
        coarse_gate = coarse_gate / torch.clamp(
            coarse_gate.sum(dim=-2, keepdim=True), min=1.0e-8
        )
        fine_gate = fine_gate / torch.clamp(
            fine_gate.sum(dim=-2, keepdim=True), min=1.0e-8
        )
        coarse_hidden = torch.sum(scale_hidden * coarse_gate, dim=-2)
        fine_hidden = torch.sum(scale_hidden * fine_gate, dim=-2)
        return shared_hidden, coarse_hidden, fine_hidden

    def _normalized_log_wavelength(
        self,
        wavelength_angstrom: Tensor | float,
        *,
        device: torch.device,
        dtype: torch.dtype,
        shape: torch.Size,
    ) -> Tensor:
        wavelength = torch.as_tensor(
            wavelength_angstrom,
            device=device,
            dtype=dtype,
        )
        if wavelength.shape != shape:
            wavelength = torch.broadcast_to(wavelength, shape)
        wavelength = torch.clamp(wavelength, min=1.0)
        return (
            torch.log(wavelength)
            - self.spectral_log_center.to(device=device, dtype=dtype)
        ) / (self.spectral_log_scale.to(device=device, dtype=dtype))

    def _spectral_basis(
        self,
        wavelength_angstrom: Tensor | float,
        *,
        device: torch.device,
        dtype: torch.dtype,
        shape: torch.Size,
    ) -> Tensor:
        lam = self._normalized_log_wavelength(
            wavelength_angstrom,
            device=device,
            dtype=dtype,
            shape=shape,
        )
        centers = self.spectral_basis_centers.to(device=device, dtype=dtype)
        widths = self.spectral_basis_widths.to(device=device, dtype=dtype)
        diff = lam.unsqueeze(-1) - centers
        # ⚡ Bolt: Element-wise multiplication is faster than torch.square or **2
        scaled_diff = diff / torch.clamp(widths, min=1.0e-3)
        basis = torch.exp(-0.5 * (scaled_diff * scaled_diff))
        return basis / torch.clamp(basis.sum(dim=-1, keepdim=True), min=1.0e-8)

    def _bandpass_features(self, wavelengths: Tensor, weights: Tensor) -> Tensor:
        lam = self._normalized_log_wavelength(
            wavelengths.to(dtype=self.base_output.weight.dtype),
            device=wavelengths.device,
            dtype=self.base_output.weight.dtype,
            shape=wavelengths.shape,
        )
        norm_weights = weights.to(dtype=self.base_output.weight.dtype)
        norm_weights = norm_weights / torch.clamp(norm_weights.sum(), min=1.0e-8)
        mean = torch.sum(norm_weights * lam)
        centered = lam - mean
        # Element-wise multiplication is faster than torch.square or **2
        centered_sq = centered * centered
        second = torch.sum(norm_weights * centered_sq)
        third = torch.sum(norm_weights * centered_sq * centered)
        fourth = torch.sum(norm_weights * centered_sq * centered_sq)
        feats = [mean, second, third, fourth]
        scales = torch.arange(1, 5, device=lam.device, dtype=lam.dtype)
        phase = math.pi * scales.view(-1, *([1] * lam.ndim)) * lam
        sin_sum = torch.sum(
            norm_weights * torch.sin(phase), dim=tuple(range(1, 1 + lam.ndim))
        )
        cos_sum = torch.sum(
            norm_weights * torch.cos(phase), dim=tuple(range(1, 1 + lam.ndim))
        )
        freq_feats = torch.stack([sin_sum, cos_sum], dim=1).flatten()
        return torch.cat([torch.stack(feats), freq_feats])

    def _spectral_context_from_basis(
        self,
        coeffs: Tensor,
        basis: Tensor,
    ) -> Tensor:
        return self.spectral_context(torch.cat([coeffs * basis, basis], dim=-1))

    def _bandpass_support_scale(self, wavelengths: Tensor, weights: Tensor) -> Tensor:
        lam = self._normalized_log_wavelength(
            wavelengths.to(dtype=self.base_output.weight.dtype),
            device=wavelengths.device,
            dtype=self.base_output.weight.dtype,
            shape=wavelengths.shape,
        )
        norm_weights = weights.to(dtype=lam.dtype)
        norm_weights = norm_weights / torch.clamp(norm_weights.sum(), min=1.0e-8)
        mean = torch.sum(norm_weights * lam)
        # ⚡ Bolt: Element-wise multiplication is faster than torch.square or **2
        lam_diff = lam - mean
        variance = torch.sum(norm_weights * (lam_diff * lam_diff))
        # Keep the continuous field exact for delta-like bands while allowing
        # broader passbands to use a small support-refinement correction.
        return 1.0 - torch.exp(-12.0 * variance)

    def _decode_with_bandpass(
        self,
        shared_hidden: Tensor,
        fine_hidden: Tensor,
        spectral_context: Tensor,
        bandpass_latent: Tensor,
        *,
        support_scale: Tensor | float = 0.0,
    ) -> Tensor:
        if spectral_context.ndim < shared_hidden.ndim:
            spectral_context = spectral_context.view(
                *([1] * (shared_hidden.ndim - 1)),
                spectral_context.shape[-1],
            )
            spectral_context = spectral_context.expand(*shared_hidden.shape[:-1], -1)
        if bandpass_latent.ndim < shared_hidden.ndim:
            bandpass_latent = bandpass_latent.view(
                *([1] * (shared_hidden.ndim - 1)),
                bandpass_latent.shape[-1],
            )
            bandpass_latent = bandpass_latent.expand(*shared_hidden.shape[:-1], -1)
        spectral_gamma_beta = self.spectral_affine(spectral_context)
        spectral_gamma, spectral_beta = torch.chunk(spectral_gamma_beta, 2, dim=-1)
        spectral_hidden = shared_hidden * (1.0 + spectral_gamma) + spectral_beta
        decoded = self.decoder(
            torch.cat(
                [shared_hidden, spectral_hidden, spectral_context, bandpass_latent],
                dim=-1,
            )
        )
        base = self.base_output(decoded).squeeze(-1) + self.refine(decoded).squeeze(-1)
        detail_hidden = fine_hidden - shared_hidden
        residual_context = torch.cat(
            [detail_hidden, spectral_context, bandpass_latent], dim=-1
        )
        gamma_beta = self.residual_affine(residual_context)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        mod_hidden = (spectral_hidden + detail_hidden) * (1.0 + gamma) + beta
        residual = self.residual_decoder(
            torch.cat(
                [mod_hidden, detail_hidden, spectral_context, bandpass_latent], dim=-1
            )
        ).squeeze(-1)
        support_scale_tensor = torch.as_tensor(
            support_scale,
            device=base.device,
            dtype=base.dtype,
        )
        if support_scale_tensor.ndim < base.ndim:
            support_scale_tensor = support_scale_tensor.view(
                *([1] * base.ndim),
            )
        residual = residual * support_scale_tensor
        return F.softplus(base + residual) + 1.0e-4

    def _shared_hidden_and_coeffs(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        spatial_raw = self._spatial_coords(
            ra_deg,
            dec_deg,
            float(self.domain.wavelength_center),
        ).to(dtype=self.base_output.weight.dtype)
        spatial_unit = self._spatial_unit_box_coords(
            ra_deg,
            dec_deg,
            float(self.domain.wavelength_center),
        ).to(dtype=self.base_output.weight.dtype)
        shared_hidden, _coarse_hidden, fine_hidden = (
            self._shared_hidden_components_from_spatial(
                spatial_raw,
                spatial_unit,
            )
        )
        coeffs = self.coefficient_head(shared_hidden)
        return shared_hidden, coeffs, fine_hidden

    def forward(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        shared_hidden, coeffs, fine_hidden = self._shared_hidden_and_coeffs(
            ra_deg, dec_deg
        )
        lam = self._normalized_log_wavelength(
            wavelength_angstrom,
            device=ra_deg.device,
            dtype=shared_hidden.dtype,
            shape=ra_deg.shape,
        )
        basis = self._spectral_basis(
            wavelength_angstrom,
            device=ra_deg.device,
            dtype=shared_hidden.dtype,
            shape=ra_deg.shape,
        )
        spectral_context = self._spectral_context_from_basis(coeffs, basis)
        bandpass_latent = self.bandpass_context(_point_bandpass_features(lam, 4))
        return self._decode_with_bandpass(
            shared_hidden,
            fine_hidden,
            spectral_context,
            bandpass_latent,
            support_scale=0.0,
        )

    def render_bandpass(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelengths: Tensor,
        weights: Tensor,
    ) -> Tensor:
        shared_hidden, coeffs, fine_hidden = self._shared_hidden_and_coeffs(
            ra_deg, dec_deg
        )
        norm_weights = weights.to(dtype=shared_hidden.dtype)
        norm_weights = norm_weights / torch.clamp(norm_weights.sum(), min=1.0e-8)
        basis_stack = torch.stack(
            [
                self._spectral_basis(
                    wavelength,
                    device=ra_deg.device,
                    dtype=shared_hidden.dtype,
                    shape=ra_deg.shape,
                )
                for wavelength in wavelengths
            ],
            dim=0,
        )
        mixed_basis = torch.sum(
            basis_stack * norm_weights.view(-1, *([1] * (basis_stack.ndim - 1))),
            dim=0,
        )
        spectral_context = self._spectral_context_from_basis(coeffs, mixed_basis)
        bandpass_latent = self.bandpass_context(
            self._bandpass_features(wavelengths, weights)
        )
        return self._decode_with_bandpass(
            shared_hidden,
            fine_hidden,
            spectral_context,
            bandpass_latent,
            support_scale=self._bandpass_support_scale(wavelengths, weights),
        )

    def _observability_mode_weights(
        self,
        observations: Sequence[FieldObservation],
        *,
        device: torch.device,
        dtype: torch.dtype,
        grid_size: int = 65,
    ) -> Tensor:
        cache_key = tuple(
            sorted(
                {
                    (
                        observation.band_name,
                        observation.spectral_min,
                        observation.spectral_max,
                        observation.transmission_center,
                        observation.transmission_sigma,
                    )
                    for observation in observations
                }
            )
        )
        cached = self._spectral_observability_cache.get(cache_key)
        if cached is not None:
            return cached.to(device=device, dtype=dtype)
        lower = max(
            min(_bandpass_bounds(observation)[0] for observation in observations),
            1.0,
        )
        upper = max(_bandpass_bounds(observation)[1] for observation in observations)
        wavelengths = torch.linspace(
            lower,
            upper,
            steps=max(int(grid_size), 9),
            device=device,
            dtype=dtype,
        )
        basis = self._spectral_basis(
            wavelengths,
            device=device,
            dtype=dtype,
            shape=wavelengths.shape,
        )
        responses = []
        for observation in observations:
            response = _bandpass_response(observation, wavelengths)
            response = response / torch.clamp(response.sum(), min=1.0e-8)
            responses.append(torch.sum(response.unsqueeze(-1) * basis, dim=0))
        response_matrix = torch.stack(responses, dim=0)
        # ⚡ Bolt: Element-wise multiplication is faster than torch.square or **2
        visibility = torch.sqrt(torch.sum(response_matrix * response_matrix, dim=0))
        max_visibility = torch.clamp(visibility.max(), min=1.0e-8)
        weights = torch.clamp(
            max_visibility / torch.clamp(visibility, min=1.0e-4) - 1.0, min=0.0
        )
        weights = weights / torch.clamp(weights.mean(), min=1.0e-8)
        self._spectral_observability_cache[cache_key] = weights.detach().cpu()
        return weights

    def spectral_observability_penalty(
        self,
        observations: Sequence[FieldObservation],
        *,
        probe_grid_size: int = 8,
    ) -> Tensor:
        device = self.base_output.weight.device
        dtype = self.base_output.weight.dtype
        probe_ra, probe_dec = _smoothness_probe_grid(
            self.domain,
            grid_size=max(int(probe_grid_size), 2),
            device=device,
        )
        _shared_hidden, coeffs, _fine_hidden = self._shared_hidden_and_coeffs(
            probe_ra,
            probe_dec,
        )
        mode_weights = self._observability_mode_weights(
            observations,
            device=device,
            dtype=dtype,
        )
        return torch.mean(
            torch.square(coeffs) * mode_weights.view(*([1] * (coeffs.ndim - 1)), -1)
        )

    def residual_head_penalty(self) -> Tensor:
        penalty = torch.tensor(
            0.0,
            device=self.base_output.weight.device,
            dtype=self.base_output.weight.dtype,
        )
        for param in self.residual_affine.parameters():
            penalty = penalty + torch.mean(param * param)
        residual_out = self.residual_decoder[-1]
        assert isinstance(residual_out, nn.Linear)
        penalty = penalty + torch.mean(residual_out.weight * residual_out.weight)
        penalty = penalty + torch.mean(residual_out.bias * residual_out.bias)
        return penalty


class SkyLambdaAtlasField(_BaseFieldModel):
    atlas_centers: Tensor

    def __init__(
        self,
        domain: FieldDomain,
        *,
        atlas_side: int = 2,
        atlas_sigma: float = 0.3,
        topk_experts: int | None = None,
        base_resolution: int = 16,
        num_scales: int = 3,
        features_per_plane: int = 8,
        hidden_dim: int = 64,
        hidden_layers: int = 2,
        refine_hidden_dim: int = 32,
    ) -> None:
        super().__init__(domain)
        self.atlas_side = int(atlas_side)
        self.atlas_sigma = float(atlas_sigma)
        self.topk_experts = None if topk_experts is None else max(int(topk_experts), 1)
        centers_1d = torch.linspace(
            0.5 / self.atlas_side,
            1.0 - 0.5 / self.atlas_side,
            self.atlas_side,
            dtype=torch.float32,
        )
        yy, xx = torch.meshgrid(centers_1d, centers_1d, indexing="ij")
        centers = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
        self.register_buffer("atlas_centers", centers, persistent=False)
        self.experts = nn.ModuleList(
            [
                SkyLambdaKPlanesField(
                    domain,
                    base_resolution=base_resolution,
                    num_scales=num_scales,
                    features_per_plane=features_per_plane,
                    hidden_dim=hidden_dim,
                    hidden_layers=hidden_layers,
                    refine_hidden_dim=refine_hidden_dim,
                )
                for _ in range(self.atlas_side * self.atlas_side)
            ]
        )

    def _typed_experts(self) -> list[SkyLambdaKPlanesField]:
        experts: list[SkyLambdaKPlanesField] = []
        for expert in self.experts:
            if not isinstance(expert, SkyLambdaKPlanesField):
                raise TypeError(f"unexpected atlas expert type {type(expert).__name__}")
            experts.append(expert)
        return experts

    def _atlas_local_xy(self, xy: Tensor) -> Tensor:
        return _atlas_local_coords(xy, self.atlas_centers, self.atlas_side)

    def _atlas_expert_forward(
        self,
        expert: SkyLambdaKPlanesField,
        spatial_raw: Tensor,
        spatial_unit: Tensor,
        lam: Tensor,
        lam_unit: Tensor,
        local_xy: Tensor,
    ) -> Tensor:
        x = local_xy[..., 0]
        y = local_xy[..., 1]
        z = spatial_unit[..., 2] if expert.spatial_dim == 3 else None
        sampled: list[Tensor] = []
        for scale_idx in range(expert.num_scales):
            sampled.extend(
                _sample_triplane(
                    expert.planes,
                    prefix=f"s{scale_idx}_",
                    x=x,
                    y=y,
                    lam_unit=lam_unit,
                    z=z,
                )
            )
        plane_feats = torch.cat(sampled, dim=-1)
        local_xy_raw = 2.0 * local_xy - 1.0
        if expert.spatial_dim == 3:
            spatial_feats = torch.cat([local_xy_raw, spatial_raw[..., 2:3]], dim=-1)
        else:
            spatial_feats = local_xy_raw
        feats = torch.cat([spatial_feats, lam.unsqueeze(-1), plane_feats], dim=-1)
        hidden = expert.hidden(feats)
        return hidden

    def _atlas_weights(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        spatial_unit = self._spatial_unit_box_coords(
            ra_deg, dec_deg, wavelength_angstrom
        )
        return _atlas_expert_weights(
            spatial_unit[..., :2],
            self.atlas_centers,
            self.atlas_sigma,
            self.topk_experts,
        )

    def forward(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        spatial_raw = self._spatial_coords(
            ra_deg,
            dec_deg,
            wavelength_angstrom,
        ).to(dtype=self._typed_experts()[0].base_output.weight.dtype)
        spatial_unit = self._spatial_unit_box_coords(
            ra_deg,
            dec_deg,
            wavelength_angstrom,
        ).to(dtype=self._typed_experts()[0].base_output.weight.dtype)
        lam = self._normalized_wavelength(
            wavelength_angstrom,
            device=ra_deg.device,
            dtype=self._typed_experts()[0].base_output.weight.dtype,
            shape=ra_deg.shape,
        )
        lam_unit = 0.5 * (torch.tanh(lam) + 1.0)
        weights = self._atlas_weights(ra_deg, dec_deg, wavelength_angstrom)
        local_xys = self._atlas_local_xy(spatial_unit[..., :2])
        preds = torch.stack(
            [
                expert._decode_hidden(
                    self._atlas_expert_forward(
                        expert,
                        spatial_raw,
                        spatial_unit,
                        lam,
                        lam_unit,
                        local_xys[..., expert_idx, :],
                    )
                )
                for expert_idx, expert in enumerate(self._typed_experts())
            ],
            dim=-1,
        )
        return torch.sum(weights * preds, dim=-1)

    def render_bandpass(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelengths: Tensor,
        weights: Tensor,
    ) -> Tensor:
        norm_weights = weights.to(dtype=self._typed_experts()[0].base_output.weight.dtype)
        norm_weights = norm_weights / torch.clamp(norm_weights.sum(), min=1.0e-8)
        spatial_raw = self._spatial_coords(
            ra_deg,
            dec_deg,
            float(wavelengths[0].item()),
        ).to(dtype=self._typed_experts()[0].base_output.weight.dtype)
        spatial_unit = self._spatial_unit_box_coords(
            ra_deg,
            dec_deg,
            float(wavelengths[0].item()),
        ).to(dtype=self._typed_experts()[0].base_output.weight.dtype)
        local_xys = self._atlas_local_xy(spatial_unit[..., :2])
        atlas_weights = self._atlas_weights(
            ra_deg,
            dec_deg,
            float(wavelengths[0].item()),
        ).to(dtype=self._typed_experts()[0].base_output.weight.dtype)
        mixed_expert_hiddens: list[Tensor] = []
        for expert_idx, expert in enumerate(self._typed_experts()):
            per_lambda_hidden = []
            for wavelength in wavelengths:
                lam = expert._normalized_wavelength(
                    wavelength,
                    device=ra_deg.device,
                    dtype=expert.base_output.weight.dtype,
                    shape=ra_deg.shape,
                )
                lam_unit = 0.5 * (torch.tanh(lam) + 1.0)
                per_lambda_hidden.append(
                    self._atlas_expert_forward(
                        expert,
                        spatial_raw,
                        spatial_unit,
                        lam,
                        lam_unit,
                        local_xys[..., expert_idx, :],
                    )
                )
            stacked_hidden = torch.stack(per_lambda_hidden, dim=0)
            mixed_expert_hiddens.append(
                torch.sum(
                    stacked_hidden
                    * norm_weights.view(-1, *([1] * (stacked_hidden.ndim - 1))),
                    dim=0,
                )
            )
        preds = torch.stack(
            [
                expert._decode_hidden(hidden)
                for expert, hidden in zip(
                    self._typed_experts(), mixed_expert_hiddens, strict=True
                )
            ],
            dim=-1,
        )
        return torch.sum(atlas_weights * preds, dim=-1)


class SkyLambdaFeatureAtlasField(_BaseFieldModel):
    atlas_centers: Tensor

    def __init__(
        self,
        domain: FieldDomain,
        *,
        atlas_side: int = 2,
        atlas_sigma: float = 0.2,
        topk_experts: int | None = None,
        base_resolution: int = 16,
        num_scales: int = 3,
        features_per_plane: int = 8,
        scale_hidden_dim: int = 64,
        scale_num_blocks: int = 2,
        decoder_hidden_dim: int = 64,
        decoder_num_blocks: int = 2,
        refine_hidden_dim: int = 32,
        bandpass_hidden_dim: int = 32,
        global_hidden_dim: int = 0,
        global_num_blocks: int = 1,
        aggregate_hidden_moments: bool = False,
    ) -> None:
        super().__init__(domain)
        self.atlas_side = int(atlas_side)
        self.atlas_sigma = float(atlas_sigma)
        self.topk_experts = None if topk_experts is None else max(int(topk_experts), 1)
        centers_1d = torch.linspace(
            0.5 / self.atlas_side,
            1.0 - 0.5 / self.atlas_side,
            self.atlas_side,
            dtype=torch.float32,
        )
        yy, xx = torch.meshgrid(centers_1d, centers_1d, indexing="ij")
        centers = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
        self.register_buffer("atlas_centers", centers, persistent=False)
        self.experts = nn.ModuleList(
            [
                SkyLambdaFeatureField(
                    domain,
                    base_resolution=base_resolution,
                    num_scales=num_scales,
                    features_per_plane=features_per_plane,
                    scale_hidden_dim=scale_hidden_dim,
                    scale_num_blocks=scale_num_blocks,
                    decoder_hidden_dim=decoder_hidden_dim,
                    decoder_num_blocks=decoder_num_blocks,
                    refine_hidden_dim=refine_hidden_dim,
                    bandpass_hidden_dim=bandpass_hidden_dim,
                    global_hidden_dim=global_hidden_dim,
                    global_num_blocks=global_num_blocks,
                    aggregate_hidden_moments=aggregate_hidden_moments,
                )
                for _ in range(self.atlas_side * self.atlas_side)
            ]
        )

    def _typed_experts(self) -> list[SkyLambdaFeatureField]:
        experts: list[SkyLambdaFeatureField] = []
        for expert in self.experts:
            if not isinstance(expert, SkyLambdaFeatureField):
                raise TypeError(f"unexpected atlas expert type {type(expert).__name__}")
            experts.append(expert)
        return experts

    def _atlas_local_xy(self, xy: Tensor) -> Tensor:
        return _atlas_local_coords(xy, self.atlas_centers, self.atlas_side)

    def _atlas_weights(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        spatial_unit = self._spatial_unit_box_coords(
            ra_deg, dec_deg, wavelength_angstrom
        )
        return _atlas_expert_weights(
            spatial_unit[..., :2],
            self.atlas_centers,
            self.atlas_sigma,
            self.topk_experts,
        )

    def _expert_local_components(
        self,
        expert: SkyLambdaFeatureField,
        spatial_raw: Tensor,
        spatial_unit: Tensor,
        lam: Tensor,
        local_xy: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        local_xy_raw = 2.0 * local_xy - 1.0
        if expert.spatial_dim == 3:
            local_spatial_raw = torch.cat([local_xy_raw, spatial_raw[..., 2:3]], dim=-1)
            local_spatial_unit = torch.cat([local_xy, spatial_unit[..., 2:3]], dim=-1)
        else:
            local_spatial_raw = local_xy_raw
            local_spatial_unit = local_xy
        return local_spatial_raw, local_spatial_unit, lam

    def _expert_prediction(
        self,
        expert: SkyLambdaFeatureField,
        spatial_raw: Tensor,
        spatial_unit: Tensor,
        local_xy: Tensor,
        wavelengths: Tensor,
        weights: Tensor,
    ) -> Tensor:
        hidden_planes = []
        global_hiddens = []
        dtype = expert.base_output.weight.dtype
        for wavelength in wavelengths:
            lam = expert._normalized_wavelength(
                wavelength,
                device=spatial_raw.device,
                dtype=dtype,
                shape=spatial_raw.shape[:-1],
            )
            local_spatial_raw, local_spatial_unit, lam = self._expert_local_components(
                expert,
                spatial_raw,
                spatial_unit,
                lam,
                local_xy,
            )
            hidden_planes.append(
                expert._scale_hidden_from_components(
                    local_spatial_raw,
                    local_spatial_unit,
                    lam,
                )
            )
            global_hidden = expert._global_hidden_from_components(
                local_spatial_raw, lam
            )
            if global_hidden is not None:
                global_hiddens.append(global_hidden)
        stacked_hidden = torch.stack(hidden_planes, dim=0)
        norm_weights = weights.to(dtype=stacked_hidden.dtype)
        norm_weights = norm_weights / torch.clamp(norm_weights.sum(), min=1.0e-8)
        mixed_hidden = expert._aggregate_hidden_stack(stacked_hidden, norm_weights)
        bandpass_latent = expert.bandpass_context(
            expert._bandpass_features(wavelengths, weights)
        )
        mixed_global_hidden = None
        if global_hiddens:
            stacked_global_hidden = torch.stack(global_hiddens, dim=0)
            mixed_global_hidden = expert._aggregate_hidden_stack(
                stacked_global_hidden,
                norm_weights,
            )
        return expert._decode_hidden(mixed_hidden, bandpass_latent, mixed_global_hidden)

    def render_bandpass(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelengths: Tensor,
        weights: Tensor,
    ) -> Tensor:
        reference_wavelength = float(wavelengths[0].item())
        spatial_raw = self._spatial_coords(
            ra_deg,
            dec_deg,
            reference_wavelength,
        ).to(dtype=self._typed_experts()[0].base_output.weight.dtype)
        spatial_unit = self._spatial_unit_box_coords(
            ra_deg,
            dec_deg,
            reference_wavelength,
        ).to(dtype=self._typed_experts()[0].base_output.weight.dtype)
        local_xys = self._atlas_local_xy(spatial_unit[..., :2])
        atlas_weights = self._atlas_weights(
            ra_deg,
            dec_deg,
            reference_wavelength,
        ).to(dtype=self._typed_experts()[0].base_output.weight.dtype)
        preds = torch.stack(
            [
                self._expert_prediction(
                    expert,
                    spatial_raw,
                    spatial_unit,
                    local_xys[..., expert_idx, :],
                    wavelengths,
                    weights,
                )
                for expert_idx, expert in enumerate(self._typed_experts())
            ],
            dim=-1,
        )
        return torch.sum(atlas_weights * preds, dim=-1)

    def forward(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        dtype = self._typed_experts()[0].base_output.weight.dtype
        spatial_raw = self._spatial_coords(
            ra_deg,
            dec_deg,
            wavelength_angstrom,
        ).to(dtype=dtype)
        spatial_unit = self._spatial_unit_box_coords(
            ra_deg,
            dec_deg,
            wavelength_angstrom,
        ).to(dtype=dtype)
        local_xys = self._atlas_local_xy(spatial_unit[..., :2])
        atlas_weights = self._atlas_weights(
            ra_deg,
            dec_deg,
            wavelength_angstrom,
        ).to(dtype=dtype)
        one_weight = torch.ones(1, device=spatial_raw.device, dtype=dtype)
        preds = []
        for expert_idx, expert in enumerate(self._typed_experts()):
            lam = expert._normalized_wavelength(
                wavelength_angstrom,
                device=spatial_raw.device,
                dtype=dtype,
                shape=spatial_raw.shape[:-1],
            )
            local_spatial_raw, local_spatial_unit, lam = self._expert_local_components(
                expert,
                spatial_raw,
                spatial_unit,
                lam,
                local_xys[..., expert_idx, :],
            )
            hidden = expert._scale_hidden_from_components(
                local_spatial_raw,
                local_spatial_unit,
                lam,
            )
            mixed_hidden = expert._aggregate_hidden_stack(
                hidden.unsqueeze(0), one_weight
            )
            bandpass_latent = expert.bandpass_context(
                _point_bandpass_features(lam, 4)
            )
            global_hidden = expert._global_hidden_from_components(
                local_spatial_raw, lam
            )
            mixed_global_hidden = (
                expert._aggregate_hidden_stack(global_hidden.unsqueeze(0), one_weight)
                if global_hidden is not None
                else None
            )
            preds.append(
                expert._decode_hidden(
                    mixed_hidden, bandpass_latent, mixed_global_hidden
                )
            )
        return torch.sum(atlas_weights * torch.stack(preds, dim=-1), dim=-1)
