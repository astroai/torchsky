"""Neural field-model classes for :mod:`torchsky.map.field`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from typing import Literal, Sequence
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ._core_helpers import (
    _BaseFieldModel,
    _fourier_features_vectorized,
    _point_bandpass_features,
    _reset_xavier_mlp,
)
from .hashgrid import HashGridEncoder
from .networks import (
    ResidualMLPBlock,
    reset_logflux_readout_linear,
    reset_silu_mlp_linears,
    silu_mlp_sequential,
)
from .sky_encoding import make_sky_coordinate_encoder, sky_encoder_output_dim
from .types import (
    FieldDomain,
)


if TYPE_CHECKING:
    pass


class SkyEncoderBandpassField(_BaseFieldModel):
    def __init__(
        self,
        domain: FieldDomain,
        *,
        hidden_dim: int = 128,
        hidden_layers: int = 3,
        num_fourier_frequencies: int = 8,
        spectral_frequencies: int = 6,
        num_bandpass_frequencies: int = 6,
        bandpass_hidden_dim: int = 48,
        scale_fourier_to_domain: bool = True,
    ) -> None:
        super().__init__(domain)
        self.num_bandpass_frequencies = int(num_bandpass_frequencies)
        self.encoder = make_sky_coordinate_encoder(
            domain,
            num_fourier_frequencies=num_fourier_frequencies,
            spectral_frequencies=spectral_frequencies,
            scale_fourier_to_domain=scale_fourier_to_domain,
        )
        encoded_dim = sky_encoder_output_dim(self.encoder)
        bandpass_in_dim = 4 + 2 * max(self.num_bandpass_frequencies, 0)
        self.bandpass_hidden = nn.Sequential(
            nn.Linear(bandpass_in_dim, bandpass_hidden_dim),
            nn.SiLU(),
            nn.Linear(bandpass_hidden_dim, bandpass_hidden_dim),
            nn.SiLU(),
        )
        layers: list[nn.Module] = []
        current_dim = encoded_dim + bandpass_hidden_dim
        for _ in range(max(int(hidden_layers), 1)):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.SiLU())
            current_dim = hidden_dim
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(current_dim, 1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.bandpass_hidden:
            if type(module) is nn.Linear:
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)
        reset_silu_mlp_linears(self.hidden)
        reset_logflux_readout_linear(self.output)

    def _bandpass_features(self, wavelengths: Tensor, weights: Tensor) -> Tensor:
        lam = (
            wavelengths.to(dtype=self.hidden[0].weight.dtype)
            - float(self.domain.wavelength_center)
        ) / float(self.domain.wavelength_scale)
        weights = weights.to(dtype=self.hidden[0].weight.dtype)
        weights = weights / torch.clamp(weights.sum(), min=1.0e-8)
        mean = torch.sum(weights * lam)
        centered = lam - mean
        # Element-wise multiplication is faster than torch.square or **2
        centered_sq = centered * centered
        second = torch.sum(weights * centered_sq)
        third = torch.sum(weights * centered_sq * centered)
        fourth = torch.sum(weights * centered_sq * centered_sq)
        feats = [mean, second, third, fourth]
        if self.num_bandpass_frequencies > 0:
            scales = torch.arange(
                1,
                self.num_bandpass_frequencies + 1,
                device=lam.device,
                dtype=lam.dtype,
            )
            phase = math.pi * scales.view(-1, *([1] * lam.ndim)) * lam
            sin_sum = torch.sum(
                weights * torch.sin(phase), dim=tuple(range(1, 1 + lam.ndim))
            )
            cos_sum = torch.sum(
                weights * torch.cos(phase), dim=tuple(range(1, 1 + lam.ndim))
            )
            freq_feats = torch.stack([sin_sum, cos_sum], dim=1).flatten()
            return torch.cat([torch.stack(feats), freq_feats])
        return torch.stack(feats)

    def _encoded_planes(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelengths: Tensor,
    ) -> Tensor:
        encoded = []
        ra64 = ra_deg.to(dtype=torch.float64)
        dec64 = dec_deg.to(dtype=torch.float64)
        for wavelength in wavelengths:
            wavelength_grid = torch.full_like(
                ra64, float(wavelength.item()), dtype=torch.float64
            )
            encoded.append(
                self.encoder(
                    ra64,
                    dec64,
                    wavelength=wavelength_grid,
                ).to(dtype=self.output.weight.dtype)
            )
        return torch.stack(encoded, dim=0)

    def render_bandpass(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelengths: Tensor,
        weights: Tensor,
    ) -> Tensor:
        encoded = self._encoded_planes(ra_deg, dec_deg, wavelengths)
        norm_weights = weights.to(dtype=encoded.dtype)
        norm_weights = norm_weights / torch.clamp(norm_weights.sum(), min=1.0e-8)
        bandpass_latent = self.bandpass_hidden(
            self._bandpass_features(wavelengths, weights)
        )
        bandpass_latent = bandpass_latent.view(1, *([1] * (encoded.ndim - 2)), -1)
        bandpass_latent = bandpass_latent.expand(
            encoded.shape[0], *encoded.shape[1:-1], -1
        )
        hidden_planes = self.hidden(torch.cat([encoded, bandpass_latent], dim=-1))
        mixed_hidden = torch.sum(
            hidden_planes * norm_weights.view(-1, *([1] * (hidden_planes.ndim - 1))),
            dim=0,
        )
        out = self.output(mixed_hidden).squeeze(-1)
        return F.softplus(out) + 1.0e-4

    def forward(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        dtype = self.output.weight.dtype
        ra64 = ra_deg.to(dtype=torch.float64)
        dec64 = dec_deg.to(dtype=torch.float64)
        wavelength_grid = torch.as_tensor(
            wavelength_angstrom,
            device=ra_deg.device,
            dtype=torch.float64,
        )
        if wavelength_grid.shape != ra64.shape:
            wavelength_grid = torch.broadcast_to(wavelength_grid, ra64.shape)
        encoded = self.encoder(
            ra64,
            dec64,
            wavelength=wavelength_grid,
        ).to(dtype=dtype)
        lam = self._normalized_wavelength(
            wavelength_angstrom,
            device=ra_deg.device,
            dtype=dtype,
            shape=ra_deg.shape,
        )
        bandpass_latent = self.bandpass_hidden(
            _point_bandpass_features(lam, self.num_bandpass_frequencies)
        )
        hidden = self.hidden(torch.cat([encoded, bandpass_latent], dim=-1))
        out = self.output(hidden).squeeze(-1)
        return F.softplus(out) + 1.0e-4


class ConditionedSpectralField(_BaseFieldModel):
    def __init__(
        self,
        domain: FieldDomain,
        *,
        hidden_dim: int = 96,
        hidden_layers: int = 3,
        num_spatial_frequencies: int = 6,
        num_spectral_frequencies: int = 6,
        spectral_hidden_dim: int = 48,
    ) -> None:
        super().__init__(domain)
        self.hidden_dim = int(hidden_dim)
        self.hidden_layers = max(int(hidden_layers), 1)
        self.num_spatial_frequencies = int(num_spatial_frequencies)
        self.num_spectral_frequencies = int(num_spectral_frequencies)
        self.spatial_dim = 2 if self.coord_mode == "local_tangent" else 3
        in_dim = self.spatial_dim * (1 + 2 * max(self.num_spatial_frequencies, 0))
        self.input = nn.Linear(in_dim, self.hidden_dim)
        self.hidden = nn.ModuleList(
            nn.Linear(self.hidden_dim, self.hidden_dim)
            for _ in range(self.hidden_layers - 1)
        )
        spectral_in_dim = 3 + 2 * max(self.num_spectral_frequencies, 0)
        self.spectral_mlp = nn.Sequential(
            nn.Linear(spectral_in_dim, spectral_hidden_dim),
            nn.SiLU(),
            nn.Linear(
                spectral_hidden_dim,
                2 * self.hidden_dim * self.hidden_layers,
            ),
        )
        self.output = nn.Linear(self.hidden_dim, 1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input.weight)
        nn.init.zeros_(self.input.bias)
        for layer in self.hidden:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        for module in self.spectral_mlp:
            if type(module) is nn.Linear:
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.uniform_(self.output.weight, -1e-2, 1e-2)
        nn.init.constant_(self.output.bias, -3.0)

    def _spatial_fourier_features(self, spatial: Tensor) -> Tensor:
        return _fourier_features_vectorized(spatial, self.num_spatial_frequencies)

    def _spectral_features(self, lam: Tensor) -> Tensor:
        if self.num_spectral_frequencies > 0:
            scales = torch.arange(
                1,
                self.num_spectral_frequencies + 1,
                device=lam.device,
                dtype=lam.dtype,
            )
            scaled = lam.unsqueeze(-1) * scales
            phase = math.pi * scaled

            sin_feats = torch.sin(phase)
            cos_feats = torch.cos(phase)

            stacked_fourier = torch.stack([sin_feats, cos_feats], dim=-1).view(
                *lam.shape, -1
            )
            base_feats = torch.stack([lam, lam * lam, torch.tanh(lam)], dim=-1)
            return torch.cat([base_feats, stacked_fourier], dim=-1)
        else:
            return torch.stack([lam, lam * lam, torch.tanh(lam)], dim=-1)

    def _film_params(self, lam: Tensor) -> tuple[list[Tensor], list[Tensor]]:
        spec = self._spectral_features(lam)
        params = self.spectral_mlp(spec)
        params = params.view(*lam.shape, self.hidden_layers, 2, self.hidden_dim)
        gammas = [
            params[..., layer_idx, 0, :] for layer_idx in range(self.hidden_layers)
        ]
        betas = [
            params[..., layer_idx, 1, :] for layer_idx in range(self.hidden_layers)
        ]
        return gammas, betas

    def forward(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        lam = self._normalized_wavelength(
            wavelength_angstrom,
            device=ra_deg.device,
            dtype=self.output.weight.dtype,
            shape=ra_deg.shape,
        )
        spatial = self._spatial_coords(ra_deg, dec_deg, wavelength_angstrom).to(
            dtype=self.output.weight.dtype
        )
        x = self._spatial_fourier_features(spatial)
        gammas, betas = self._film_params(lam)
        x = self.input(x)
        x = F.silu(x * (1.0 + gammas[0]) + betas[0])
        for layer_idx, layer in enumerate(self.hidden, start=1):
            x = layer(x)
            x = F.silu(x * (1.0 + gammas[layer_idx]) + betas[layer_idx])
        out = self.output(x).squeeze(-1)
        return F.softplus(out) + 1.0e-4


class HashGridConditionedField(_BaseFieldModel):
    def __init__(
        self,
        domain: FieldDomain,
        *,
        hidden_dim: int = 96,
        hidden_layers: int = 3,
        num_levels: int = 6,
        features_per_level: int = 4,
        base_resolution: int = 8,
        finest_resolution: int = 128,
        num_spectral_frequencies: int = 8,
        log2_hashmap_size: int | Sequence[int] = 12,
        hash_interpolation: Literal["linear", "smoothstep"] = "linear",
        hash_dense_coarse: bool = True,
        hash_max_dense_cells: int = 4096,
        hash_level_gain: bool = False,
    ) -> None:
        super().__init__(domain)
        self.spatial_dim = 2 if self.coord_mode == "local_tangent" else 3
        self.num_spectral_frequencies = int(num_spectral_frequencies)
        self.encoder = HashGridEncoder(
            self.spatial_dim,
            num_levels=num_levels,
            features_per_level=features_per_level,
            base_resolution=base_resolution,
            finest_resolution=finest_resolution,
            log2_hashmap_size=log2_hashmap_size,
            interpolation=hash_interpolation,
            dense_coarse=hash_dense_coarse,
            max_dense_cells=hash_max_dense_cells,
            learnable_level_gain=hash_level_gain,
        )
        spectral_dim = 3 + 2 * max(self.num_spectral_frequencies, 0)
        in_dim = self.spatial_dim + num_levels * features_per_level + spectral_dim
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

    def _spectral_features(self, lam: Tensor) -> Tensor:
        if self.num_spectral_frequencies > 0:
            scales = torch.arange(
                1,
                self.num_spectral_frequencies + 1,
                device=lam.device,
                dtype=lam.dtype,
            )
            scaled = lam.unsqueeze(-1) * scales
            phase = math.pi * scaled

            sin_feats = torch.sin(phase)
            cos_feats = torch.cos(phase)

            stacked_fourier = torch.stack([sin_feats, cos_feats], dim=-1).view(
                *lam.shape, -1
            )
            base_feats = torch.stack([lam, lam * lam, torch.tanh(lam)], dim=-1)
            return torch.cat([base_feats, stacked_fourier], dim=-1)
        else:
            return torch.stack([lam, lam * lam, torch.tanh(lam)], dim=-1)

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
        spectral = self._spectral_features(lam)
        grid_feats = self.encoder(spatial_unit)
        feats = torch.cat([spatial_raw, grid_feats, spectral], dim=-1)
        out = self.output(self.hidden(feats)).squeeze(-1)
        return F.softplus(out) + 1.0e-4


class SkyEncoderMLPField(_BaseFieldModel):
    def __init__(
        self,
        domain: FieldDomain,
        *,
        hidden_dim: int = 96,
        hidden_layers: int = 3,
        num_fourier_frequencies: int = 8,
        spectral_frequencies: int = 5,
        scale_fourier_to_domain: bool = True,
    ) -> None:
        super().__init__(domain)
        self.encoder = make_sky_coordinate_encoder(
            domain,
            num_fourier_frequencies=num_fourier_frequencies,
            spectral_frequencies=spectral_frequencies,
            scale_fourier_to_domain=scale_fourier_to_domain,
        )
        in_dim = sky_encoder_output_dim(self.encoder)
        self.hidden = silu_mlp_sequential(in_dim, hidden_dim, hidden_layers)
        self.output = nn.Linear(hidden_dim, 1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        reset_silu_mlp_linears(self.hidden)
        reset_logflux_readout_linear(self.output)

    def forward(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        wavelength = torch.as_tensor(
            wavelength_angstrom,
            device=ra_deg.device,
            dtype=torch.float64,
        )
        if wavelength.shape != ra_deg.shape:
            wavelength = torch.broadcast_to(wavelength, ra_deg.shape)
        feats = self.encoder(
            ra_deg.to(dtype=torch.float64),
            dec_deg.to(dtype=torch.float64),
            wavelength=wavelength,
        )
        output_dtype = self.output.weight.dtype
        feats = feats.to(dtype=output_dtype)
        out = self.output(self.hidden(feats)).squeeze(-1)
        return F.softplus(out) + 1.0e-4


class StrainerSkyEncoderMLPField(_BaseFieldModel):
    """Sky encoder with explicit trunk (`encoder` + `body`) and `decoder_head` for transfer learning.

    Matches ``SkyEncoderMLPField`` capacity and adds ``reset_decoder_head`` plus
    trunk state-dict helpers for transfer learning.
    """

    def __init__(
        self,
        domain: FieldDomain,
        *,
        hidden_dim: int = 96,
        hidden_layers: int = 3,
        num_fourier_frequencies: int = 8,
        spectral_frequencies: int = 5,
        scale_fourier_to_domain: bool = True,
    ) -> None:
        super().__init__(domain)
        self.encoder = make_sky_coordinate_encoder(
            domain,
            num_fourier_frequencies=num_fourier_frequencies,
            spectral_frequencies=spectral_frequencies,
            scale_fourier_to_domain=scale_fourier_to_domain,
        )
        in_dim = sky_encoder_output_dim(self.encoder)
        self.body = silu_mlp_sequential(in_dim, hidden_dim, hidden_layers)
        self.decoder_head = nn.Linear(hidden_dim, 1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        reset_silu_mlp_linears(self.body)
        reset_logflux_readout_linear(self.decoder_head)

    def reset_decoder_head(self, *, bias_init: float = -3.0) -> None:
        nn.init.uniform_(self.decoder_head.weight, -1e-2, 1e-2)
        nn.init.constant_(self.decoder_head.bias, float(bias_init))

    def trunk_state_dict(self) -> dict[str, Tensor]:
        state = self.state_dict()
        return {
            key: value
            for key, value in state.items()
            if key.startswith("encoder.") or key.startswith("body.")
        }

    def load_trunk_state_dict(
        self,
        state_dict: dict[str, Tensor],
        *,
        strict: bool = True,
    ):
        incompatible = self.load_state_dict(state_dict, strict=False)
        if strict:
            for key in state_dict:
                if key in incompatible.missing_keys:
                    raise RuntimeError(f"failed to load trunk tensor for key {key!r}")
        return incompatible

    def forward(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        wavelength = torch.as_tensor(
            wavelength_angstrom,
            device=ra_deg.device,
            dtype=torch.float64,
        )
        if wavelength.shape != ra_deg.shape:
            wavelength = torch.broadcast_to(wavelength, ra_deg.shape)
        feats = self.encoder(
            ra_deg.to(dtype=torch.float64),
            dec_deg.to(dtype=torch.float64),
            wavelength=wavelength,
        )
        output_dtype = self.decoder_head.weight.dtype
        feats = feats.to(dtype=output_dtype)
        out = self.decoder_head(self.body(feats)).squeeze(-1)
        return F.softplus(out) + 1.0e-4


class SkyEncoderResMLPField(_BaseFieldModel):
    def __init__(
        self,
        domain: FieldDomain,
        *,
        hidden_dim: int = 128,
        num_blocks: int = 4,
        num_fourier_frequencies: int = 8,
        spectral_frequencies: int = 6,
        multiscale_readout: bool = False,
        readout_hidden_dim: int | None = None,
        scale_fourier_to_domain: bool = True,
    ) -> None:
        super().__init__(domain)
        self.multiscale_readout = bool(multiscale_readout)
        self.encoder = make_sky_coordinate_encoder(
            domain,
            num_fourier_frequencies=num_fourier_frequencies,
            spectral_frequencies=spectral_frequencies,
            scale_fourier_to_domain=scale_fourier_to_domain,
        )
        in_dim = sky_encoder_output_dim(self.encoder)
        self.input = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(hidden_dim) for _ in range(max(int(num_blocks), 1))]
        )
        if self.multiscale_readout:
            stacked_dim = hidden_dim * (1 + len(self.blocks))
            readout_width = (
                int(readout_hidden_dim)
                if readout_hidden_dim is not None
                else max(hidden_dim // 2, 64)
            )
            self.output = nn.Sequential(
                nn.Linear(stacked_dim, readout_width),
                nn.SiLU(),
                nn.Linear(readout_width, 1),
            )
        else:
            self.output = nn.Linear(hidden_dim, 1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.input.weight, nonlinearity="relu")
        nn.init.zeros_(self.input.bias)
        if isinstance(self.output, nn.Sequential):
            for module in self.output:
                if type(module) is nn.Linear:
                    nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                    nn.init.zeros_(module.bias)
            final = self.output[-1]
            assert isinstance(final, nn.Linear)
            nn.init.uniform_(final.weight, -1e-2, 1e-2)
            nn.init.constant_(final.bias, 0.0)
        else:
            nn.init.uniform_(self.output.weight, -1e-2, 1e-2)
            nn.init.constant_(self.output.bias, -3.0)

    def forward(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        wavelength = torch.as_tensor(
            wavelength_angstrom,
            device=ra_deg.device,
            dtype=torch.float64,
        )
        if wavelength.shape != ra_deg.shape:
            wavelength = torch.broadcast_to(wavelength, ra_deg.shape)
        feats = self.encoder(
            ra_deg.to(dtype=torch.float64),
            dec_deg.to(dtype=torch.float64),
            wavelength=wavelength,
        )
        output_dtype = (
            self.output[0].weight.dtype
            if isinstance(self.output, nn.Sequential)
            else self.output.weight.dtype
        )
        feats = feats.to(dtype=output_dtype)
        x = F.silu(self.input(feats))
        readout_states = [x] if self.multiscale_readout else None
        for block in self.blocks:
            x = block(x)
            if readout_states is not None:
                readout_states.append(x)
        if readout_states is not None:
            x = torch.cat(readout_states, dim=-1)
        out = self.output(x).squeeze(-1)
        return F.softplus(out) + 1.0e-4


class HashGridMLPField(_BaseFieldModel):
    def __init__(
        self,
        domain: FieldDomain,
        *,
        hidden_dim: int = 64,
        hidden_layers: int = 2,
        num_levels: int = 6,
        features_per_level: int = 4,
        base_resolution: int = 8,
        finest_resolution: int = 128,
        log2_hashmap_size: int | Sequence[int] = 12,
        hash_interpolation: Literal["linear", "smoothstep"] = "linear",
        hash_dense_coarse: bool = True,
        hash_max_dense_cells: int = 4096,
        hash_level_gain: bool = False,
    ) -> None:
        super().__init__(domain)
        self.encoder = HashGridEncoder(
            self.coord_dim,
            num_levels=num_levels,
            features_per_level=features_per_level,
            base_resolution=base_resolution,
            finest_resolution=finest_resolution,
            log2_hashmap_size=log2_hashmap_size,
            interpolation=hash_interpolation,
            dense_coarse=hash_dense_coarse,
            max_dense_cells=hash_max_dense_cells,
            learnable_level_gain=hash_level_gain,
        )
        in_dim = self.coord_dim + num_levels * features_per_level
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
        coords_unit = self._unit_box_coords(ra_deg, dec_deg, wavelength_angstrom)
        grid_feats = self.encoder(coords_unit.to(dtype=self.output.weight.dtype))
        raw = coords_unit.to(dtype=self.output.weight.dtype)
        feats = torch.cat([raw, grid_feats], dim=-1)
        out = self.output(self.hidden(feats)).squeeze(-1)
        return F.softplus(out) + 1.0e-4


class HashGridResMLPField(_BaseFieldModel):
    """Hash-grid spatial encoder + residual MLP decoder (explicit features + refinement)."""

    def __init__(
        self,
        domain: FieldDomain,
        *,
        hidden_dim: int = 96,
        num_res_blocks: int = 3,
        num_levels: int = 6,
        features_per_level: int = 4,
        base_resolution: int = 8,
        finest_resolution: int = 128,
        log2_hashmap_size: int | Sequence[int] = 12,
        hash_interpolation: Literal["linear", "smoothstep"] = "linear",
        hash_dense_coarse: bool = True,
        hash_max_dense_cells: int = 4096,
        hash_level_gain: bool = False,
    ) -> None:
        super().__init__(domain)
        self.encoder = HashGridEncoder(
            self.coord_dim,
            num_levels=num_levels,
            features_per_level=features_per_level,
            base_resolution=base_resolution,
            finest_resolution=finest_resolution,
            log2_hashmap_size=log2_hashmap_size,
            interpolation=hash_interpolation,
            dense_coarse=hash_dense_coarse,
            max_dense_cells=hash_max_dense_cells,
            learnable_level_gain=hash_level_gain,
        )
        in_dim = self.coord_dim + num_levels * features_per_level
        self.input = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(hidden_dim) for _ in range(max(int(num_res_blocks), 1))]
        )
        self.output = nn.Linear(hidden_dim, 1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input.weight)
        nn.init.zeros_(self.input.bias)
        nn.init.uniform_(self.output.weight, -1e-2, 1e-2)
        nn.init.constant_(self.output.bias, -3.0)

    def forward(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        coords_unit = self._unit_box_coords(ra_deg, dec_deg, wavelength_angstrom)
        grid_feats = self.encoder(coords_unit.to(dtype=self.output.weight.dtype))
        raw = coords_unit.to(dtype=self.output.weight.dtype)
        feats = torch.cat([raw, grid_feats], dim=-1)
        x = F.silu(self.input(feats))
        for block in self.blocks:
            x = block(x)
        out = self.output(x).squeeze(-1)
        return F.softplus(out) + 1.0e-4


class HashGridSpectralField(_BaseFieldModel):
    def __init__(
        self,
        domain: FieldDomain,
        *,
        hidden_dim: int = 64,
        hidden_layers: int = 2,
        num_levels: int = 6,
        features_per_level: int = 4,
        base_resolution: int = 8,
        finest_resolution: int = 128,
        num_spectral_frequencies: int = 4,
        log2_hashmap_size: int | Sequence[int] = 12,
        hash_interpolation: Literal["linear", "smoothstep"] = "linear",
        hash_dense_coarse: bool = True,
        hash_max_dense_cells: int = 4096,
        hash_level_gain: bool = False,
    ) -> None:
        super().__init__(domain)
        self.spatial_dim = 2 if self.coord_mode == "local_tangent" else 3
        self.num_spectral_frequencies = int(num_spectral_frequencies)
        self.basis_dim = 3 + 2 * max(self.num_spectral_frequencies, 0)
        self.encoder = HashGridEncoder(
            self.spatial_dim,
            num_levels=num_levels,
            features_per_level=features_per_level,
            base_resolution=base_resolution,
            finest_resolution=finest_resolution,
            log2_hashmap_size=log2_hashmap_size,
            interpolation=hash_interpolation,
            dense_coarse=hash_dense_coarse,
            max_dense_cells=hash_max_dense_cells,
            learnable_level_gain=hash_level_gain,
        )
        in_dim = self.spatial_dim + num_levels * features_per_level
        layers: list[nn.Module] = []
        current_dim = in_dim
        for _ in range(max(int(hidden_layers), 1)):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.SiLU())
            current_dim = hidden_dim
        self.hidden = nn.Sequential(*layers)
        self.coeff_head = nn.Linear(current_dim, self.basis_dim)
        self.bias_head = nn.Linear(current_dim, 1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.hidden:
            if type(module) is nn.Linear:
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.coeff_head.weight)
        nn.init.zeros_(self.coeff_head.bias)
        nn.init.uniform_(self.bias_head.weight, -1e-2, 1e-2)
        nn.init.constant_(self.bias_head.bias, 0.0)

    def _spectral_basis(self, lam: Tensor) -> Tensor:
        if self.num_spectral_frequencies > 0:
            scales = torch.arange(
                1,
                self.num_spectral_frequencies + 1,
                device=lam.device,
                dtype=lam.dtype,
            )
            scaled = lam.unsqueeze(-1) * scales
            phase = math.pi * scaled

            sin_feats = torch.sin(phase)
            cos_feats = torch.cos(phase)

            stacked_fourier = torch.stack([sin_feats, cos_feats], dim=-1).view(
                *lam.shape, -1
            )
            base_feats = torch.stack([torch.ones_like(lam), lam, lam * lam], dim=-1)
            return torch.cat([base_feats, stacked_fourier], dim=-1)
        else:
            return torch.stack([torch.ones_like(lam), lam, lam * lam], dim=-1)

    def forward(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        spatial_unit = self._spatial_unit_box_coords(
            ra_deg,
            dec_deg,
            wavelength_angstrom,
        ).to(dtype=self.coeff_head.weight.dtype)
        spatial_raw = self._spatial_coords(
            ra_deg,
            dec_deg,
            wavelength_angstrom,
        ).to(dtype=self.coeff_head.weight.dtype)
        grid_feats = self.encoder(spatial_unit)
        hidden = self.hidden(torch.cat([spatial_raw, grid_feats], dim=-1))
        coeff = self.coeff_head(hidden)
        bias = self.bias_head(hidden).squeeze(-1)
        lam = self._normalized_wavelength(
            wavelength_angstrom,
            device=ra_deg.device,
            dtype=self.coeff_head.weight.dtype,
            shape=ra_deg.shape,
        )
        spectral_basis = self._spectral_basis(lam)
        out = bias + torch.sum(coeff * spectral_basis, dim=-1)
        return F.softplus(out) + 1.0e-4
