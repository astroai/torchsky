"""Literature architecture baselines for the WCS-aware JWST inverse problem.

A1 is a NeRF-style positional-encoding MLP: joint normalized (x, y, log-lambda)
coordinates are Fourier encoded, processed by an 8-layer ReLU MLP, and the
encoded input is concatenated back into the hidden state after layer 4.

A2 is a SIREN: joint normalized (x, y, log-lambda) coordinates are processed
by sine-activated layers with the initialization proposed by Sitzmann et al.

Both classes intentionally preserve the existing WCS-aware observation
operator. Only the scene representation S_theta(x, y, lambda) changes.
"""

from __future__ import annotations

import math
from typing import Literal, Sequence

import torch
from torch import Tensor, nn

from .wcs_aware_inr import (
    IdentityFeatureAlignment,
    UnionCanvas,
    WCSAwareMultiWavelengthINR,
)

ArchitectureKind = Literal["A1_nerf", "A2_siren"]


class NormalizedPlaneEncoder(nn.Module):
    """Map union-canvas pixel coordinates to [-1, 1]^2 without extra features."""

    def __init__(self, canvas: UnionCanvas) -> None:
        super().__init__()
        self.width = max(int(canvas.shape[1]) - 1, 1)
        self.height = max(int(canvas.shape[0]) - 1, 1)
        self.output_dim = 2

    def forward(self, xy: Tensor) -> Tensor:
        x = 2.0 * xy[:, 0] / float(self.width) - 1.0
        y = 2.0 * xy[:, 1] / float(self.height) - 1.0
        return torch.stack((x, y), dim=-1).float()


class NeRFPositionalEncoding(nn.Module):
    """Canonical NeRF-style sinusoidal encoding for a low-dimensional coordinate."""

    def __init__(self, in_dim: int, num_frequencies: int = 10) -> None:
        super().__init__()
        if in_dim <= 0:
            raise ValueError("in_dim must be positive")
        if num_frequencies <= 0:
            raise ValueError("num_frequencies must be positive")
        self.in_dim = int(in_dim)
        self.num_frequencies = int(num_frequencies)
        frequencies = math.pi * (2.0 ** torch.arange(self.num_frequencies))
        self.register_buffer("frequencies", frequencies.float())
        self.output_dim = self.in_dim * (1 + 2 * self.num_frequencies)

    def forward(self, coordinate: Tensor) -> Tensor:
        if coordinate.ndim != 2 or coordinate.shape[-1] != self.in_dim:
            raise ValueError(
                f"coordinate must have shape [N,{self.in_dim}], got {tuple(coordinate.shape)}"
            )
        phase = coordinate[..., None] * self.frequencies
        sin = torch.sin(phase).reshape(coordinate.shape[0], -1)
        cos = torch.cos(phase).reshape(coordinate.shape[0], -1)
        return torch.cat((coordinate, sin, cos), dim=-1)


class NeRFStyleMLP(nn.Module):
    """NeRF-style PE + 8x256 ReLU MLP with the canonical mid-network skip."""

    def __init__(
        self,
        in_dim: int = 3,
        *,
        num_frequencies: int = 10,
        hidden_dim: int = 256,
        depth: int = 8,
        skip_after_layer: int = 4,
    ) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be at least 2")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if not 0 <= skip_after_layer < depth - 1:
            raise ValueError("skip_after_layer must be between 0 and depth-2")

        self.encoding = NeRFPositionalEncoding(in_dim, num_frequencies)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.skip_after_layer = int(skip_after_layer)

        encoded_dim = self.encoding.output_dim
        layers: list[nn.Linear] = []
        for index in range(self.depth):
            if index == 0:
                layer_in = encoded_dim
            elif index == self.skip_after_layer + 1:
                layer_in = self.hidden_dim + encoded_dim
            else:
                layer_in = self.hidden_dim
            layers.append(nn.Linear(layer_in, self.hidden_dim))
        self.layers = nn.ModuleList(layers)
        self.output = nn.Linear(self.hidden_dim, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # TensorFlow/Keras Dense in the original NeRF code uses Glorot/Xavier
        # initialization by default.
        for layer in [*self.layers, self.output]:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, coordinate: Tensor) -> Tensor:
        encoded = self.encoding(coordinate)
        hidden = encoded
        for index, layer in enumerate(self.layers):
            hidden = torch.relu(layer(hidden))
            if index == self.skip_after_layer:
                hidden = torch.cat((hidden, encoded), dim=-1)
        return self.output(hidden)


class SineLayer(nn.Module):
    """One SIREN layer with the paper initialization."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        omega_0: float,
        is_first: bool,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("SineLayer dimensions must be positive")
        if not math.isfinite(float(omega_0)) or omega_0 <= 0:
            raise ValueError("omega_0 must be finite and positive")
        self.in_features = int(in_features)
        self.omega_0 = float(omega_0)
        self.is_first = bool(is_first)
        self.linear = nn.Linear(in_features, out_features)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / float(self.in_features)
            else:
                bound = math.sqrt(6.0 / float(self.in_features)) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None:
                self.linear.bias.uniform_(-bound, bound)

    def forward(self, coordinate: Tensor) -> Tensor:
        return torch.sin(self.omega_0 * self.linear(coordinate))


class SirenMLP(nn.Module):
    """SIREN with the canonical 256-wide, 3-hidden-layer image-fit defaults."""

    def __init__(
        self,
        in_dim: int = 3,
        *,
        hidden_dim: int = 256,
        hidden_layers: int = 3,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
    ) -> None:
        super().__init__()
        if hidden_layers < 0:
            raise ValueError("hidden_layers must be non-negative")
        self.first = SineLayer(
            in_dim,
            hidden_dim,
            omega_0=first_omega_0,
            is_first=True,
        )
        self.hidden = nn.ModuleList(
            [
                SineLayer(
                    hidden_dim,
                    hidden_dim,
                    omega_0=hidden_omega_0,
                    is_first=False,
                )
                for _ in range(hidden_layers)
            ]
        )
        self.output = nn.Linear(hidden_dim, 1)
        self._reset_output(hidden_omega_0)

    def _reset_output(self, hidden_omega_0: float) -> None:
        bound = math.sqrt(6.0 / float(self.output.in_features)) / float(hidden_omega_0)
        with torch.no_grad():
            self.output.weight.uniform_(-bound, bound)
            if self.output.bias is not None:
                self.output.bias.uniform_(-bound, bound)

    def forward(self, coordinate: Tensor) -> Tensor:
        hidden = self.first(coordinate)
        for layer in self.hidden:
            hidden = layer(hidden)
        return self.output(hidden)


class WCSAwareLiteratureINR(WCSAwareMultiWavelengthINR):
    """Drop-in WCS-aware INR whose scene backbone is A1 NeRF-like or A2 SIREN.

    Geometry, bandpass integration, optional PSF handling, exposure background,
    masking, and likelihood are inherited unchanged from
    :class:`WCSAwareMultiWavelengthINR`.
    """

    def __init__(
        self,
        headers: Sequence[dict],
        canvas: UnionCanvas,
        *,
        architecture: ArchitectureKind,
        wavelength_min: float,
        wavelength_max: float,
        anchor_exposure: int = 0,
        alignment_mode: str = "tweakreg_frozen",
        learn_background: bool = True,
        output_activation: str = "softplus",
        nerf_num_frequencies: int = 10,
        nerf_hidden_dim: int = 256,
        nerf_depth: int = 8,
        nerf_skip_after_layer: int = 4,
        siren_hidden_dim: int = 256,
        siren_hidden_layers: int = 3,
        siren_first_omega_0: float = 30.0,
        siren_hidden_omega_0: float = 30.0,
    ) -> None:
        if architecture not in {"A1_nerf", "A2_siren"}:
            raise ValueError("architecture must be A1_nerf or A2_siren")
        if alignment_mode not in {"original_wcs", "tweakreg_frozen"}:
            raise ValueError(
                "literature architecture ablations intentionally support only "
                "original_wcs or tweakreg_frozen so the scene-backbone comparison "
                "does not introduce an architecture-specific trainable alignment"
            )
        if not math.isfinite(float(wavelength_min)) or wavelength_min <= 0:
            raise ValueError("wavelength_min must be finite and positive")
        if not math.isfinite(float(wavelength_max)) or wavelength_max <= 0:
            raise ValueError("wavelength_max must be finite and positive")
        if wavelength_max < wavelength_min:
            raise ValueError("wavelength_max must be >= wavelength_min")

        # Initialize the common WCS-aware observation machinery. The temporary
        # trunk/decoder are replaced immediately below; keeping the parent class
        # preserves predict_observation(), background handling, PSF integration,
        # and compatibility with the existing training helper.
        super().__init__(
            headers,
            canvas,
            encoding="fourier",
            spatial_frequencies=1,
            spectral_frequencies=0,
            hidden_dim=4,
            latent_dim=2,
            trunk_depth=0,
            decoder_depth=0,
            anchor_exposure=anchor_exposure,
            alignment_mode=alignment_mode,
            learn_background=learn_background,
            output_activation=output_activation,
        )

        self.architecture = architecture
        self.position_encoder = NormalizedPlaneEncoder(canvas)
        self.feature_alignment = IdentityFeatureAlignment(alignment_mode)
        self.trunk = nn.Identity()

        self._single_wavelength = math.isclose(
            float(wavelength_min), float(wavelength_max), rel_tol=0.0, abs_tol=1.0e-12
        )
        self.register_buffer(
            "_log_wavelength_min",
            torch.tensor(math.log(float(wavelength_min)), dtype=torch.float32),
        )
        self.register_buffer(
            "_log_wavelength_max",
            torch.tensor(math.log(float(wavelength_max)), dtype=torch.float32),
        )

        if architecture == "A1_nerf":
            self.decoder = NeRFStyleMLP(
                3,
                num_frequencies=nerf_num_frequencies,
                hidden_dim=nerf_hidden_dim,
                depth=nerf_depth,
                skip_after_layer=nerf_skip_after_layer,
            )
            self.architecture_metadata = {
                "name": "A1_nerf",
                "reference": "Mildenhall et al. 2020, arXiv:2003.08934",
                "joint_coordinate": "(x_norm, y_norm, normalized_log_wavelength)",
                "num_frequencies": int(nerf_num_frequencies),
                "hidden_dim": int(nerf_hidden_dim),
                "depth": int(nerf_depth),
                "skip_after_layer": int(nerf_skip_after_layer),
                "activation": "relu",
            }
        else:
            self.decoder = SirenMLP(
                3,
                hidden_dim=siren_hidden_dim,
                hidden_layers=siren_hidden_layers,
                first_omega_0=siren_first_omega_0,
                hidden_omega_0=siren_hidden_omega_0,
            )
            self.architecture_metadata = {
                "name": "A2_siren",
                "reference": "Sitzmann et al. 2020, arXiv:2006.09661",
                "joint_coordinate": "(x_norm, y_norm, normalized_log_wavelength)",
                "hidden_dim": int(siren_hidden_dim),
                "hidden_layers_after_first": int(siren_hidden_layers),
                "first_omega_0": float(siren_first_omega_0),
                "hidden_omega_0": float(siren_hidden_omega_0),
                "activation": "sine",
            }

    def _spectral_features(self, wavelength: Tensor) -> Tensor:
        """Normalize log wavelength to [-1, 1] for the joint 3D coordinate."""
        logw = torch.log(wavelength.clamp_min(1.0e-12)).float()
        lo = self._log_wavelength_min.to(device=logw.device, dtype=logw.dtype)
        hi = self._log_wavelength_max.to(device=logw.device, dtype=logw.dtype)
        span = hi - lo
        if self._single_wavelength:
            normalized = torch.zeros_like(logw)
        else:
            normalized = 2.0 * (logw - lo) / span - 1.0
        return normalized[:, None]


__all__ = [
    "ArchitectureKind",
    "NeRFPositionalEncoding",
    "NeRFStyleMLP",
    "NormalizedPlaneEncoder",
    "SineLayer",
    "SirenMLP",
    "WCSAwareLiteratureINR",
]
