"""Experimental continuous field models for multiwavelength map fitting."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .. import WCS
from ..encoders import SkyCoordinateEncoder
from ..spectral import SpectralAxis

FieldModelKind = Literal[
    "sine_implicit",
    "wire_implicit",
    "gaussian_basis",
    "fourier_mlp",
    "spectral_basis_field",
    "learned_spectral_basis_field",
    "sky_lambda_planes_field",
    "bandpass_conditioned_field",
    "conditioned_spectral_field",
    "hashgrid_conditioned_field",
    "sky_encoder_mlp",
    "sky_encoder_resmlp",
    "hashgrid_mlp",
    "hashgrid_spectral_field",
]
FieldCoordinateMode = Literal["local_tangent", "sphere_xyz"]
FieldLossKind = Literal["mse", "charbonnier", "huber"]
FieldLRSchedule = Literal["constant", "cosine"]
FieldLossReduction = Literal["per_observation", "per_band", "global"]
FieldCurriculum = Literal["none", "spectral_frequency_ramp"]


@dataclass(frozen=True)
class FieldDomain:
    center_ra_deg: float
    center_dec_deg: float
    spatial_scale_deg: float
    wavelength_center: float
    wavelength_scale: float
    coord_mode: FieldCoordinateMode = "local_tangent"


@dataclass
class FieldObservation:
    image: Tensor
    header: dict
    weight: Tensor | None = None
    mask: Tensor | None = None
    wavelength: float | None = None
    spectral_min: float | None = None
    spectral_max: float | None = None
    transmission_center: float | None = None
    transmission_sigma: float | None = None
    psf_sigma_pix: float = 0.0
    band_name: str | None = None


@dataclass
class FieldFitResult:
    model: nn.Module
    loss_history: tuple[float, ...]
    final_loss: float

    @property
    def initial_loss(self) -> float:
        return self.loss_history[0]


def _wrapped_delta_ra_deg(ra_a_deg: Tensor, ra_b_deg: float) -> Tensor:
    return ((ra_a_deg - float(ra_b_deg) + 180.0) % 360.0) - 180.0


def _header_shape(header: dict, *, fallback: tuple[int, int]) -> tuple[int, int]:
    naxis1 = header.get("NAXIS1")
    naxis2 = header.get("NAXIS2")
    if naxis1 is None or naxis2 is None:
        return fallback
    return int(naxis2), int(naxis1)


def _coordinate_grid(
    header: dict,
    *,
    shape: tuple[int, int],
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    height, width = _header_shape(header, fallback=shape)
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float64, device=device),
        torch.arange(width, dtype=torch.float64, device=device),
        indexing="ij",
    )
    ra_deg, dec_deg = WCS(header).pixel_to_world(x, y)
    return ra_deg, dec_deg


def _module_device(model: nn.Module) -> torch.device:
    parameter = next(model.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(model.buffers(), None)
    if buffer is not None:
        return buffer.device
    return torch.device("cpu")


def _gaussian_kernel1d(sigma_pix: float, *, device: torch.device) -> Tensor:
    radius = max(1, int(math.ceil(3.0 * float(sigma_pix))))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    kernel = torch.exp(-0.5 * torch.square(x / float(sigma_pix)))
    return kernel / torch.clamp(kernel.sum(), min=1e-8)


def _gaussian_blur(image: Tensor, sigma_pix: float) -> Tensor:
    if sigma_pix <= 0.0:
        return image
    kernel = _gaussian_kernel1d(sigma_pix, device=image.device)
    kernel_x = kernel.view(1, 1, 1, -1)
    kernel_y = kernel.view(1, 1, -1, 1)
    out = image.unsqueeze(0).unsqueeze(0)
    out = F.conv2d(out, kernel_x, padding=(0, kernel.numel() // 2))
    out = F.conv2d(out, kernel_y, padding=(kernel.numel() // 2, 0))
    return out.squeeze(0).squeeze(0)


def _bandpass_bounds(observation: FieldObservation) -> tuple[float, float]:
    if observation.spectral_min is not None and observation.spectral_max is not None:
        lower = float(observation.spectral_min)
        upper = float(observation.spectral_max)
    elif observation.wavelength is not None:
        lower = float(observation.wavelength)
        upper = float(observation.wavelength)
    else:
        raise ValueError(
            "FieldObservation must define wavelength or spectral_min/spectral_max"
        )
    if not math.isfinite(lower) or not math.isfinite(upper) or upper < lower:
        raise ValueError("FieldObservation spectral bounds must be finite and ordered")
    return lower, upper


def _bandpass_quadrature(
    observation: FieldObservation,
    *,
    num_spectral_samples: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    if num_spectral_samples < 1:
        raise ValueError("num_spectral_samples must be positive")
    lower, upper = _bandpass_bounds(observation)
    if lower == upper or num_spectral_samples == 1:
        samples = torch.tensor(
            [0.5 * (lower + upper)], dtype=torch.float32, device=device
        )
        return samples, torch.ones_like(samples)
    samples = torch.linspace(
        lower,
        upper,
        steps=int(num_spectral_samples),
        dtype=torch.float32,
        device=device,
    )
    center = (
        float(observation.transmission_center)
        if observation.transmission_center is not None
        else 0.5 * (lower + upper)
    )
    sigma = observation.transmission_sigma
    if sigma is None:
        weights = torch.ones_like(samples)
    else:
        sigma = float(sigma)
        if not math.isfinite(sigma) or sigma <= 0.0:
            raise ValueError(
                "FieldObservation transmission_sigma must be finite and positive"
            )
        weights = torch.exp(-0.5 * torch.square((samples - center) / sigma))
    return samples, weights / torch.clamp(weights.sum(), min=1e-8)


def estimate_field_domain(
    observations: Sequence[FieldObservation],
    *,
    coord_mode: FieldCoordinateMode = "local_tangent",
) -> FieldDomain:
    if not observations:
        raise ValueError("estimate_field_domain requires at least one observation")
    center_ra = float(observations[0].header.get("CRVAL1", 0.0))
    center_dec = float(observations[0].header.get("CRVAL2", 0.0))
    wavelength_lowers: list[float] = []
    wavelength_uppers: list[float] = []
    spatial_scales: list[float] = []
    for observation in observations:
        lower, upper = _bandpass_bounds(observation)
        wavelength_lowers.append(lower)
        wavelength_uppers.append(upper)
        scale_terms = []
        for key in ("CD1_1", "CD1_2", "CD2_1", "CD2_2"):
            value = observation.header.get(key)
            if value is not None:
                scale_terms.append(abs(float(value)))
        spatial_scale = max(scale_terms) if scale_terms else 1.0 / 3600.0
        nx = int(observation.header.get("NAXIS1", observation.image.shape[-1]))
        ny = int(observation.header.get("NAXIS2", observation.image.shape[-2]))
        spatial_scales.append(spatial_scale * max(nx, ny))
    lower = min(wavelength_lowers)
    upper = max(wavelength_uppers)
    wavelength_center = 0.5 * (lower + upper)
    wavelength_scale = max(0.5 * (upper - lower), 1.0)
    return FieldDomain(
        center_ra_deg=center_ra,
        center_dec_deg=center_dec,
        spatial_scale_deg=max(max(spatial_scales), 1e-4),
        wavelength_center=wavelength_center,
        wavelength_scale=wavelength_scale,
        coord_mode=coord_mode,
    )


class _BaseFieldModel(nn.Module):
    def __init__(self, domain: FieldDomain) -> None:
        super().__init__()
        self.domain = domain
        self.coord_mode = domain.coord_mode

    @property
    def coord_dim(self) -> int:
        if self.coord_mode == "local_tangent":
            return 3
        if self.coord_mode == "sphere_xyz":
            return 4
        raise ValueError(f"unsupported field coordinate mode {self.coord_mode!r}")

    def _encoded_coords(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        wavelength = torch.as_tensor(
            wavelength_angstrom,
            device=ra_deg.device,
            dtype=ra_deg.dtype,
        )
        if wavelength.shape != ra_deg.shape:
            wavelength = torch.broadcast_to(wavelength, ra_deg.shape)
        dlam = wavelength - float(self.domain.wavelength_center)
        if self.coord_mode == "local_tangent":
            dra = _wrapped_delta_ra_deg(ra_deg, self.domain.center_ra_deg)
            dra = dra * math.cos(math.radians(self.domain.center_dec_deg))
            ddec = dec_deg - float(self.domain.center_dec_deg)
            return torch.stack(
                [
                    dra / float(self.domain.spatial_scale_deg),
                    ddec / float(self.domain.spatial_scale_deg),
                    dlam / float(self.domain.wavelength_scale),
                ],
                dim=-1,
            )
        if self.coord_mode == "sphere_xyz":
            ra_rad = torch.deg2rad(ra_deg)
            dec_rad = torch.deg2rad(dec_deg)
            cos_dec = torch.cos(dec_rad)
            return torch.stack(
                [
                    cos_dec * torch.cos(ra_rad),
                    cos_dec * torch.sin(ra_rad),
                    torch.sin(dec_rad),
                    dlam / float(self.domain.wavelength_scale),
                ],
                dim=-1,
            )
        raise ValueError(f"unsupported field coordinate mode {self.coord_mode!r}")

    def _unit_box_coords(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        coords = self._encoded_coords(ra_deg, dec_deg, wavelength_angstrom)
        return 0.5 * (torch.tanh(coords) + 1.0)

    def _spatial_coords(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        coords = self._encoded_coords(ra_deg, dec_deg, wavelength_angstrom)
        return coords[..., :-1]

    def _spatial_unit_box_coords(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        spatial = self._spatial_coords(ra_deg, dec_deg, wavelength_angstrom)
        return 0.5 * (torch.tanh(spatial) + 1.0)

    def _normalized_wavelength(
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
        return (
            wavelength - float(self.domain.wavelength_center)
        ) / float(self.domain.wavelength_scale)


class SineImplicitField(_BaseFieldModel):
    def __init__(
        self,
        domain: FieldDomain,
        *,
        hidden_dim: int = 48,
        hidden_layers: int = 2,
        omega0: float = 6.0,
    ) -> None:
        super().__init__(domain)
        self.omega0 = float(omega0)
        layers: list[nn.Linear] = [nn.Linear(self.coord_dim, hidden_dim)]
        for _ in range(max(int(hidden_layers) - 1, 0)):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.hidden = nn.ModuleList(layers)
        self.output = nn.Linear(hidden_dim, 1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        first = True
        for layer in self.hidden:
            fan_in = layer.weight.shape[1]
            if first:
                bound = 1.0 / max(fan_in, 1)
                first = False
            else:
                bound = math.sqrt(6.0 / fan_in) / max(self.omega0, 1.0)
            nn.init.uniform_(layer.weight, -bound, bound)
            nn.init.uniform_(layer.bias, -bound, bound)
        nn.init.uniform_(self.output.weight, -1e-2, 1e-2)
        nn.init.constant_(self.output.bias, 0.0)

    def forward(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        x = self._encoded_coords(ra_deg, dec_deg, wavelength_angstrom).to(
            dtype=self.hidden[0].weight.dtype
        )
        for layer in self.hidden:
            x = torch.sin(self.omega0 * layer(x))
        out = self.output(x).squeeze(-1)
        return F.softplus(out) + 1.0e-4


class _GaborLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, *, omega0: float, sigma0: float) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.omega0 = float(omega0)
        self.sigma0 = float(sigma0)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        fan_in = self.linear.weight.shape[1]
        bound = math.sqrt(6.0 / max(fan_in, 1)) / max(self.omega0, 1.0)
        nn.init.uniform_(self.linear.weight, -bound, bound)
        nn.init.uniform_(self.linear.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        z = self.linear(x)
        return torch.cos(self.omega0 * z) * torch.exp(
            -torch.square(self.sigma0 * z)
        )


class WireImplicitField(_BaseFieldModel):
    def __init__(
        self,
        domain: FieldDomain,
        *,
        hidden_dim: int = 48,
        hidden_layers: int = 2,
        omega0: float = 8.0,
        sigma0: float = 2.5,
    ) -> None:
        super().__init__(domain)
        layers: list[_GaborLayer] = [
            _GaborLayer(self.coord_dim, hidden_dim, omega0=omega0, sigma0=sigma0)
        ]
        for _ in range(max(int(hidden_layers) - 1, 0)):
            layers.append(
                _GaborLayer(hidden_dim, hidden_dim, omega0=omega0, sigma0=sigma0)
            )
        self.hidden = nn.ModuleList(layers)
        self.output = nn.Linear(hidden_dim, 1)
        nn.init.uniform_(self.output.weight, -1e-2, 1e-2)
        nn.init.constant_(self.output.bias, 0.0)

    def forward(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        x = self._encoded_coords(ra_deg, dec_deg, wavelength_angstrom).to(
            dtype=self.output.weight.dtype
        )
        for layer in self.hidden:
            x = layer(x)
        out = self.output(x).squeeze(-1)
        return F.softplus(out) + 1.0e-4


class GaussianBasisField(_BaseFieldModel):
    def __init__(
        self,
        domain: FieldDomain,
        *,
        num_basis: int = 24,
        init_scale: float = 0.35,
    ) -> None:
        super().__init__(domain)
        self.centers = nn.Parameter(torch.empty(int(num_basis), 3))
        self.log_scales = nn.Parameter(torch.full((int(num_basis), 3), math.log(init_scale)))
        self.amplitudes = nn.Parameter(torch.empty(int(num_basis)))
        self.bias = nn.Parameter(torch.zeros(()))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.uniform_(self.centers, -1.0, 1.0)
        nn.init.uniform_(self.amplitudes, -0.05, 0.05)

    def forward(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        coords = self._encoded_coords(ra_deg, dec_deg, wavelength_angstrom).to(
            dtype=self.centers.dtype
        )
        diff = coords.unsqueeze(-2) - self.centers
        scales = F.softplus(self.log_scales) + 1.0e-3
        radial2 = torch.sum(torch.square(diff / scales), dim=-1)
        basis = torch.exp(-0.5 * radial2)
        out = torch.sum(basis * self.amplitudes, dim=-1) + self.bias
        return F.softplus(out) + 1.0e-4


class FourierMLPField(_BaseFieldModel):
    def __init__(
        self,
        domain: FieldDomain,
        *,
        hidden_dim: int = 64,
        hidden_layers: int = 3,
        num_frequencies: int = 6,
        max_frequency: float = 16.0,
    ) -> None:
        super().__init__(domain)
        self.num_frequencies = int(num_frequencies)
        self.max_frequency = float(max_frequency)
        in_dim = self.coord_dim * (1 + 2 * self.num_frequencies)
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
        for module in self.hidden:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.uniform_(self.output.weight, -1e-2, 1e-2)
        nn.init.constant_(self.output.bias, 0.0)

    def _fourier_features(self, coords: Tensor) -> Tensor:
        if self.num_frequencies <= 0:
            return coords
        scales = torch.logspace(
            0.0,
            math.log2(max(self.max_frequency, 1.0)),
            steps=self.num_frequencies,
            base=2.0,
            device=coords.device,
            dtype=coords.dtype,
        )
        feats = [coords]
        for scale in scales:
            phase = math.pi * scale * coords
            feats.append(torch.sin(phase))
            feats.append(torch.cos(phase))
        return torch.cat(feats, dim=-1)

    def forward(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        coords = self._encoded_coords(ra_deg, dec_deg, wavelength_angstrom).to(
            dtype=self.output.weight.dtype
        )
        feats = self._fourier_features(coords)
        out = self.output(self.hidden(feats)).squeeze(-1)
        return F.softplus(out) + 1.0e-4


class SpectralBasisField(_BaseFieldModel):
    def __init__(
        self,
        domain: FieldDomain,
        *,
        hidden_dim: int = 96,
        hidden_layers: int = 2,
        num_spatial_frequencies: int = 6,
        num_spectral_frequencies: int = 4,
    ) -> None:
        super().__init__(domain)
        self.num_spatial_frequencies = int(num_spatial_frequencies)
        self.num_spectral_frequencies = int(num_spectral_frequencies)
        self.spatial_dim = 2 if self.coord_mode == "local_tangent" else 3
        self.basis_dim = 3 + 2 * max(self.num_spectral_frequencies, 0)
        self._curriculum_progress = 1.0
        in_dim = self.spatial_dim * (1 + 2 * max(self.num_spatial_frequencies, 0))
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
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.coeff_head.weight)
        nn.init.zeros_(self.coeff_head.bias)
        nn.init.uniform_(self.bias_head.weight, -1e-2, 1e-2)
        nn.init.constant_(self.bias_head.bias, 0.0)

    def set_curriculum_progress(self, progress: float) -> None:
        self._curriculum_progress = min(max(float(progress), 0.0), 1.0)

    def _active_frequency_count(self, total: int) -> int:
        if total <= 0:
            return 0
        return max(1, int(math.ceil(self._curriculum_progress * total)))

    def _spatial_fourier_features(self, spatial: Tensor) -> Tensor:
        if self.num_spatial_frequencies <= 0:
            return spatial
        scales = torch.arange(
            1,
            self.num_spatial_frequencies + 1,
            device=spatial.device,
            dtype=spatial.dtype,
        )
        active = self._active_frequency_count(self.num_spatial_frequencies)
        feats = [spatial]
        for idx, scale in enumerate(scales):
            phase = math.pi * scale * spatial
            sin_phase = torch.sin(phase)
            cos_phase = torch.cos(phase)
            if idx >= active:
                sin_phase = torch.zeros_like(sin_phase)
                cos_phase = torch.zeros_like(cos_phase)
            feats.append(sin_phase)
            feats.append(cos_phase)
        return torch.cat(feats, dim=-1)

    def _spectral_basis(self, lam: Tensor) -> Tensor:
        feats = [torch.ones_like(lam), lam, torch.square(lam)]
        if self.num_spectral_frequencies > 0:
            scales = torch.arange(
                1,
                self.num_spectral_frequencies + 1,
                device=lam.device,
                dtype=lam.dtype,
            )
            active = self._active_frequency_count(self.num_spectral_frequencies)
            for idx, scale in enumerate(scales):
                phase = math.pi * scale * lam
                sin_phase = torch.sin(phase)
                cos_phase = torch.cos(phase)
                if idx >= active:
                    sin_phase = torch.zeros_like(sin_phase)
                    cos_phase = torch.zeros_like(cos_phase)
                feats.append(sin_phase)
                feats.append(cos_phase)
        return torch.stack(feats, dim=-1)

    def forward(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        spatial = self._spatial_coords(
            ra_deg.to(dtype=torch.float64),
            dec_deg.to(dtype=torch.float64),
            wavelength_angstrom,
        )
        spatial_feats = self._spatial_fourier_features(spatial).to(
            dtype=self.coeff_head.weight.dtype
        )
        hidden = self.hidden(spatial_feats)
        coeff = self.coeff_head(hidden)
        bias = self.bias_head(hidden).squeeze(-1)
        lam = self._normalized_wavelength(
            wavelength_angstrom,
            device=ra_deg.device,
            dtype=torch.float64,
            shape=ra_deg.shape,
        )
        spectral_basis = self._spectral_basis(lam).to(dtype=self.coeff_head.weight.dtype)
        out = bias + torch.sum(coeff * spectral_basis, dim=-1)
        return F.softplus(out) + 1.0e-4


class BandpassConditionedField(_BaseFieldModel):
    def __init__(
        self,
        domain: FieldDomain,
        *,
        hidden_dim: int = 96,
        hidden_layers: int = 2,
        num_spatial_frequencies: int = 6,
        num_bandpass_frequencies: int = 6,
        bandpass_hidden_dim: int = 48,
    ) -> None:
        super().__init__(domain)
        self.num_spatial_frequencies = int(num_spatial_frequencies)
        self.num_bandpass_frequencies = int(num_bandpass_frequencies)
        self.spatial_dim = 2 if self.coord_mode == "local_tangent" else 3
        spatial_in_dim = self.spatial_dim * (1 + 2 * max(self.num_spatial_frequencies, 0))
        spatial_layers: list[nn.Module] = []
        current_dim = spatial_in_dim
        for _ in range(max(int(hidden_layers), 1)):
            spatial_layers.append(nn.Linear(current_dim, hidden_dim))
            spatial_layers.append(nn.SiLU())
            current_dim = hidden_dim
        self.spatial_hidden = nn.Sequential(*spatial_layers)
        bandpass_in_dim = 4 + 2 * max(self.num_bandpass_frequencies, 0)
        self.bandpass_hidden = nn.Sequential(
            nn.Linear(bandpass_in_dim, bandpass_hidden_dim),
            nn.SiLU(),
            nn.Linear(bandpass_hidden_dim, bandpass_hidden_dim),
            nn.SiLU(),
        )
        self.output = nn.Linear(hidden_dim + bandpass_hidden_dim, 1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.spatial_hidden:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for module in self.bandpass_hidden:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.uniform_(self.output.weight, -1e-2, 1e-2)
        nn.init.constant_(self.output.bias, 0.0)

    def _spatial_fourier_features(self, spatial: Tensor) -> Tensor:
        if self.num_spatial_frequencies <= 0:
            return spatial
        scales = torch.arange(
            1,
            self.num_spatial_frequencies + 1,
            device=spatial.device,
            dtype=spatial.dtype,
        )
        feats = [spatial]
        for scale in scales:
            phase = math.pi * scale * spatial
            feats.append(torch.sin(phase))
            feats.append(torch.cos(phase))
        return torch.cat(feats, dim=-1)

    def _bandpass_features(self, wavelengths: Tensor, weights: Tensor) -> Tensor:
        lam = (
            wavelengths.to(dtype=self.output.weight.dtype) - float(self.domain.wavelength_center)
        ) / float(self.domain.wavelength_scale)
        weights = weights.to(dtype=self.output.weight.dtype)
        weights = weights / torch.clamp(weights.sum(), min=1.0e-8)
        mean = torch.sum(weights * lam)
        centered = lam - mean
        second = torch.sum(weights * torch.square(centered))
        third = torch.sum(weights * centered * centered * centered)
        fourth = torch.sum(weights * torch.square(torch.square(centered)))
        feats = [mean, second, third, fourth]
        if self.num_bandpass_frequencies > 0:
            scales = torch.arange(
                1,
                self.num_bandpass_frequencies + 1,
                device=lam.device,
                dtype=lam.dtype,
            )
            for scale in scales:
                phase = math.pi * scale * lam
                feats.append(torch.sum(weights * torch.sin(phase)))
                feats.append(torch.sum(weights * torch.cos(phase)))
        return torch.stack(feats)

    def _spatial_latent(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        spatial = self._spatial_coords(ra_deg, dec_deg, wavelength_angstrom).to(
            dtype=self.output.weight.dtype
        )
        return self.spatial_hidden(self._spatial_fourier_features(spatial))

    def render_bandpass(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelengths: Tensor,
        weights: Tensor,
    ) -> Tensor:
        spatial_latent = self._spatial_latent(ra_deg, dec_deg, float(wavelengths.mean()))
        bandpass_latent = self.bandpass_hidden(self._bandpass_features(wavelengths, weights))
        bandpass_latent = bandpass_latent.view(*([1] * (spatial_latent.ndim - 1)), -1)
        bandpass_latent = bandpass_latent.expand(*spatial_latent.shape[:-1], -1)
        out = self.output(torch.cat([spatial_latent, bandpass_latent], dim=-1)).squeeze(-1)
        return F.softplus(out) + 1.0e-4

    def forward(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        wavelength = torch.as_tensor(
            wavelength_angstrom,
            device=ra_deg.device,
            dtype=self.output.weight.dtype,
        )
        if wavelength.ndim == 0:
            wavelengths = wavelength.view(1)
        else:
            wavelengths = wavelength.reshape(-1)
        weights = torch.ones_like(wavelengths)
        return self.render_bandpass(ra_deg, dec_deg, wavelengths, weights)


class LearnedSpectralBasisField(_BaseFieldModel):
    def __init__(
        self,
        domain: FieldDomain,
        *,
        hidden_dim: int = 96,
        hidden_layers: int = 2,
        num_spatial_frequencies: int = 6,
        basis_dim: int = 16,
        spectral_hidden_dim: int = 32,
    ) -> None:
        super().__init__(domain)
        self.num_spatial_frequencies = int(num_spatial_frequencies)
        self.spatial_dim = 2 if self.coord_mode == "local_tangent" else 3
        self.basis_dim = int(basis_dim)
        self._curriculum_progress = 1.0
        in_dim = self.spatial_dim * (1 + 2 * max(self.num_spatial_frequencies, 0))
        layers: list[nn.Module] = []
        current_dim = in_dim
        for _ in range(max(int(hidden_layers), 1)):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.SiLU())
            current_dim = hidden_dim
        self.hidden = nn.Sequential(*layers)
        self.coeff_head = nn.Linear(current_dim, self.basis_dim)
        self.bias_head = nn.Linear(current_dim, 1)
        self.spectral_basis = nn.Sequential(
            nn.Linear(3, spectral_hidden_dim),
            nn.SiLU(),
            nn.Linear(spectral_hidden_dim, self.basis_dim),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.hidden:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.coeff_head.weight)
        nn.init.zeros_(self.coeff_head.bias)
        nn.init.uniform_(self.bias_head.weight, -1e-2, 1e-2)
        nn.init.constant_(self.bias_head.bias, 0.0)
        for module in self.spectral_basis:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def set_curriculum_progress(self, progress: float) -> None:
        self._curriculum_progress = min(max(float(progress), 0.0), 1.0)

    def _active_frequency_count(self, total: int) -> int:
        if total <= 0:
            return 0
        return max(1, int(math.ceil(self._curriculum_progress * total)))

    def _spatial_fourier_features(self, spatial: Tensor) -> Tensor:
        if self.num_spatial_frequencies <= 0:
            return spatial
        scales = torch.arange(
            1,
            self.num_spatial_frequencies + 1,
            device=spatial.device,
            dtype=spatial.dtype,
        )
        active = self._active_frequency_count(self.num_spatial_frequencies)
        feats = [spatial]
        for idx, scale in enumerate(scales):
            phase = math.pi * scale * spatial
            sin_phase = torch.sin(phase)
            cos_phase = torch.cos(phase)
            if idx >= active:
                sin_phase = torch.zeros_like(sin_phase)
                cos_phase = torch.zeros_like(cos_phase)
            feats.append(sin_phase)
            feats.append(cos_phase)
        return torch.cat(feats, dim=-1)

    def _spectral_features(self, lam: Tensor) -> Tensor:
        return torch.stack([lam, torch.square(lam), torch.tanh(lam)], dim=-1)

    def forward(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        spatial = self._spatial_coords(
            ra_deg.to(dtype=torch.float64),
            dec_deg.to(dtype=torch.float64),
            wavelength_angstrom,
        )
        spatial_feats = self._spatial_fourier_features(spatial).to(
            dtype=self.coeff_head.weight.dtype
        )
        hidden = self.hidden(spatial_feats)
        coeff = self.coeff_head(hidden)
        bias = self.bias_head(hidden).squeeze(-1)
        lam = self._normalized_wavelength(
            wavelength_angstrom,
            device=ra_deg.device,
            dtype=torch.float64,
            shape=ra_deg.shape,
        )
        basis = self.spectral_basis(self._spectral_features(lam).to(dtype=self.coeff_head.weight.dtype))
        out = bias + torch.sum(coeff * basis, dim=-1)
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
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.uniform_(self.output.weight, -1e-2, 1e-2)
        nn.init.constant_(self.output.bias, 0.0)

    def _spatial_fourier_features(self, spatial: Tensor) -> Tensor:
        if self.num_spatial_frequencies <= 0:
            return spatial
        scales = torch.arange(
            1,
            self.num_spatial_frequencies + 1,
            device=spatial.device,
            dtype=spatial.dtype,
        )
        feats = [spatial]
        for scale in scales:
            phase = math.pi * scale * spatial
            feats.append(torch.sin(phase))
            feats.append(torch.cos(phase))
        return torch.cat(feats, dim=-1)

    def _spectral_features(self, lam: Tensor) -> Tensor:
        feats = [lam, torch.square(lam), torch.tanh(lam)]
        if self.num_spectral_frequencies > 0:
            scales = torch.arange(
                1,
                self.num_spectral_frequencies + 1,
                device=lam.device,
                dtype=lam.dtype,
            )
            for scale in scales:
                phase = math.pi * scale * lam
                feats.append(torch.sin(phase))
                feats.append(torch.cos(phase))
        return torch.stack(feats, dim=-1)

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
    ) -> None:
        super().__init__(domain)
        self.spatial_dim = 2 if self.coord_mode == "local_tangent" else 3
        self.num_spectral_frequencies = int(num_spectral_frequencies)
        self.encoder = _HashGridEncoder(
            self.spatial_dim,
            num_levels=num_levels,
            features_per_level=features_per_level,
            base_resolution=base_resolution,
            finest_resolution=finest_resolution,
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
        for module in self.hidden:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.uniform_(self.output.weight, -1e-2, 1e-2)
        nn.init.constant_(self.output.bias, 0.0)

    def _spectral_features(self, lam: Tensor) -> Tensor:
        feats = [lam, torch.square(lam), torch.tanh(lam)]
        if self.num_spectral_frequencies > 0:
            scales = torch.arange(
                1,
                self.num_spectral_frequencies + 1,
                device=lam.device,
                dtype=lam.dtype,
            )
            for scale in scales:
                phase = math.pi * scale * lam
                feats.append(torch.sin(phase))
                feats.append(torch.cos(phase))
        return torch.stack(feats, dim=-1)

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


class _ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.SiLU()
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return self.act(self.norm(residual + x))


class SkyEncoderMLPField(_BaseFieldModel):
    def __init__(
        self,
        domain: FieldDomain,
        *,
        hidden_dim: int = 96,
        hidden_layers: int = 3,
        num_fourier_frequencies: int = 6,
        spectral_frequencies: int = 4,
    ) -> None:
        super().__init__(domain)
        self.encoder = SkyCoordinateEncoder(
            num_fourier_frequencies=num_fourier_frequencies,
            spectral_frequencies=spectral_frequencies,
            include_degree2=True,
        )
        with torch.no_grad():
            dummy = self.encoder(
                torch.zeros(1, dtype=torch.float64),
                torch.zeros(1, dtype=torch.float64),
                wavelength=torch.ones(1, dtype=torch.float64),
            )
        in_dim = int(dummy.shape[-1])
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
        for module in self.hidden:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.uniform_(self.output.weight, -1e-2, 1e-2)
        nn.init.constant_(self.output.bias, 0.0)

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
        ).to(dtype=self.output.weight.dtype)
        out = self.output(self.hidden(feats)).squeeze(-1)
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
    ) -> None:
        super().__init__(domain)
        self.encoder = SkyCoordinateEncoder(
            num_fourier_frequencies=num_fourier_frequencies,
            spectral_frequencies=spectral_frequencies,
            include_degree2=True,
        )
        with torch.no_grad():
            dummy = self.encoder(
                torch.zeros(1, dtype=torch.float64),
                torch.zeros(1, dtype=torch.float64),
                wavelength=torch.ones(1, dtype=torch.float64),
            )
        in_dim = int(dummy.shape[-1])
        self.input = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [_ResidualMLPBlock(hidden_dim) for _ in range(max(int(num_blocks), 1))]
        )
        self.output = nn.Linear(hidden_dim, 1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input.weight)
        nn.init.zeros_(self.input.bias)
        nn.init.uniform_(self.output.weight, -1e-2, 1e-2)
        nn.init.constant_(self.output.bias, 0.0)

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
        ).to(dtype=self.output.weight.dtype)
        x = F.silu(self.input(feats))
        for block in self.blocks:
            x = block(x)
        out = self.output(x).squeeze(-1)
        return F.softplus(out) + 1.0e-4


class _HashGridEncoder(nn.Module):
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
        log2_hashmap_size: int = 12,
        base_resolution: int = 8,
        finest_resolution: int = 128,
    ) -> None:
        super().__init__()
        self.coord_dim = int(coord_dim)
        self.num_levels = int(num_levels)
        self.features_per_level = int(features_per_level)
        self.hashmap_size = 1 << int(log2_hashmap_size)
        self.base_resolution = int(base_resolution)
        self.finest_resolution = int(finest_resolution)
        if self.num_levels <= 0:
            raise ValueError("num_levels must be positive")
        if self.features_per_level <= 0:
            raise ValueError("features_per_level must be positive")
        if self.coord_dim <= 0 or self.coord_dim > len(self._PRIMES):
            raise ValueError("unsupported hash-grid coordinate dimension")
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
        self.tables = nn.ParameterList(
            [
                nn.Parameter(
                    1.0e-4
                    * torch.randn(self.hashmap_size, self.features_per_level)
                )
                for _ in self.resolutions
            ]
        )

    def _hash(self, corner: Tensor) -> Tensor:
        hashed = corner[..., 0].to(dtype=torch.int64)
        for dim in range(1, self.coord_dim):
            hashed = torch.bitwise_xor(
                hashed,
                corner[..., dim].to(dtype=torch.int64) * int(self._PRIMES[dim]),
            )
        return torch.remainder(hashed, self.hashmap_size).to(dtype=torch.long)

    def forward(self, coords_unit: Tensor) -> Tensor:
        flat = coords_unit.reshape(-1, self.coord_dim)
        outputs: list[Tensor] = []
        for resolution, table in zip(self.resolutions, self.tables, strict=True):
            pos = flat * float(resolution - 1)
            base = torch.floor(pos).to(dtype=torch.int64)
            frac = pos - base.to(dtype=pos.dtype)
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
                    frac_dim = frac[:, dim].to(dtype=table.dtype)
                    if use_upper:
                        corner[:, dim] = corner[:, dim] + 1
                        weight = weight * frac_dim
                    else:
                        weight = weight * (1.0 - frac_dim)
                corner = torch.clamp(corner, min=0, max=resolution - 1)
                level_feat = level_feat + weight.unsqueeze(-1) * table[self._hash(corner)]
            outputs.append(level_feat)
        encoded = torch.cat(outputs, dim=-1)
        return encoded.reshape(*coords_unit.shape[:-1], -1)


def _sample_feature_plane(plane: Tensor, coord_u: Tensor, coord_v: Tensor) -> Tensor:
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
    ) -> None:
        super().__init__(domain)
        self.encoder = _HashGridEncoder(
            self.coord_dim,
            num_levels=num_levels,
            features_per_level=features_per_level,
            base_resolution=base_resolution,
            finest_resolution=finest_resolution,
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
        for module in self.hidden:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.uniform_(self.output.weight, -1e-2, 1e-2)
        nn.init.constant_(self.output.bias, 0.0)

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
    ) -> None:
        super().__init__(domain)
        self.spatial_dim = 2 if self.coord_mode == "local_tangent" else 3
        self.num_spectral_frequencies = int(num_spectral_frequencies)
        self.basis_dim = 3 + 2 * max(self.num_spectral_frequencies, 0)
        self.encoder = _HashGridEncoder(
            self.spatial_dim,
            num_levels=num_levels,
            features_per_level=features_per_level,
            base_resolution=base_resolution,
            finest_resolution=finest_resolution,
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
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.coeff_head.weight)
        nn.init.zeros_(self.coeff_head.bias)
        nn.init.uniform_(self.bias_head.weight, -1e-2, 1e-2)
        nn.init.constant_(self.bias_head.bias, 0.0)

    def _spectral_basis(self, lam: Tensor) -> Tensor:
        feats = [torch.ones_like(lam), lam, torch.square(lam)]
        if self.num_spectral_frequencies > 0:
            scales = torch.arange(
                1,
                self.num_spectral_frequencies + 1,
                device=lam.device,
                dtype=lam.dtype,
            )
            for scale in scales:
                phase = math.pi * scale * lam
                feats.append(torch.sin(phase))
                feats.append(torch.cos(phase))
        return torch.stack(feats, dim=-1)

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
        plane_names = ["xy", "xl", "yl"]
        if self.spatial_dim == 3:
            plane_names.extend(["xz", "yz", "zl"])
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
        for module in self.hidden:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.uniform_(self.output.weight, -1e-2, 1e-2)
        nn.init.constant_(self.output.bias, 0.0)

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
        sampled = [
            _sample_feature_plane(self.planes["xy"], x, y),
            _sample_feature_plane(self.planes["xl"], x, lam_unit),
            _sample_feature_plane(self.planes["yl"], y, lam_unit),
        ]
        if self.spatial_dim == 3:
            z = spatial_unit[..., 2]
            sampled.extend(
                [
                    _sample_feature_plane(self.planes["xz"], x, z),
                    _sample_feature_plane(self.planes["yz"], y, z),
                    _sample_feature_plane(self.planes["zl"], z, lam_unit),
                ]
            )
        plane_feats = torch.cat(sampled, dim=-1)
        feats = torch.cat([spatial_raw, lam.unsqueeze(-1), plane_feats], dim=-1)
        out = self.output(self.hidden(feats)).squeeze(-1)
        return F.softplus(out) + 1.0e-4


def make_field_model(
    kind: FieldModelKind,
    domain: FieldDomain,
) -> nn.Module:
    if kind == "sine_implicit":
        return SineImplicitField(domain)
    if kind == "wire_implicit":
        return WireImplicitField(domain)
    if kind == "gaussian_basis":
        return GaussianBasisField(domain)
    if kind == "fourier_mlp":
        return FourierMLPField(domain)
    if kind == "spectral_basis_field":
        return SpectralBasisField(domain)
    if kind == "bandpass_conditioned_field":
        return BandpassConditionedField(domain)
    if kind == "learned_spectral_basis_field":
        return LearnedSpectralBasisField(domain)
    if kind == "sky_lambda_planes_field":
        return SkyLambdaPlanesField(domain)
    if kind == "conditioned_spectral_field":
        return ConditionedSpectralField(domain)
    if kind == "hashgrid_conditioned_field":
        return HashGridConditionedField(domain)
    if kind == "sky_encoder_mlp":
        return SkyEncoderMLPField(domain)
    if kind == "sky_encoder_resmlp":
        return SkyEncoderResMLPField(domain)
    if kind == "hashgrid_mlp":
        return HashGridMLPField(domain)
    if kind == "hashgrid_spectral_field":
        return HashGridSpectralField(domain)
    raise ValueError(
        "kind must be one of {'sine_implicit', 'wire_implicit', 'gaussian_basis', 'fourier_mlp', 'spectral_basis_field', 'bandpass_conditioned_field', 'learned_spectral_basis_field', 'sky_lambda_planes_field', 'conditioned_spectral_field', 'hashgrid_conditioned_field', 'sky_encoder_mlp', 'sky_encoder_resmlp', 'hashgrid_mlp', 'hashgrid_spectral_field'}"
    )


def _field_residual_loss(
    residual: Tensor,
    *,
    loss_kind: FieldLossKind,
    charbonnier_eps: float,
    huber_delta: float,
) -> Tensor:
    if loss_kind == "mse":
        return torch.square(residual)
    if loss_kind == "charbonnier":
        eps = torch.as_tensor(
            float(charbonnier_eps),
            device=residual.device,
            dtype=residual.dtype,
        )
        return torch.sqrt(torch.square(residual) + torch.square(eps))
    if loss_kind == "huber":
        return F.huber_loss(
            residual,
            torch.zeros_like(residual),
            reduction="none",
            delta=float(huber_delta),
        )
    raise ValueError(f"unsupported field loss kind {loss_kind!r}")


def render_field_observation(
    model: nn.Module,
    observation: FieldObservation,
    *,
    num_spectral_samples: int = 9,
) -> Tensor:
    image = observation.image
    if image.ndim != 2:
        raise ValueError("FieldObservation.image must be 2D")
    device = image.device
    ra_deg, dec_deg = _coordinate_grid(
        observation.header,
        shape=tuple(image.shape),
        device=device,
    )
    wavelengths, weights = _bandpass_quadrature(
        observation,
        num_spectral_samples=num_spectral_samples,
        device=device,
    )
    render_bandpass = getattr(model, "render_bandpass", None)
    if callable(render_bandpass):
        rendered = render_bandpass(
            ra_deg.to(dtype=torch.float32),
            dec_deg.to(dtype=torch.float32),
            wavelengths,
            weights,
        )
        if observation.psf_sigma_pix > 0.0:
            rendered = _gaussian_blur(rendered, float(observation.psf_sigma_pix))
        return rendered.to(dtype=image.dtype)
    spectral_planes = []
    for wavelength in wavelengths:
        spectral_planes.append(
            model(
                ra_deg.to(dtype=torch.float32),
                dec_deg.to(dtype=torch.float32),
                wavelength,
            )
        )
    stacked = torch.stack(spectral_planes, dim=0)
    rendered = torch.sum(
        stacked * weights.view(-1, 1, 1).to(dtype=stacked.dtype),
        dim=0,
    )
    if observation.psf_sigma_pix > 0.0:
        rendered = _gaussian_blur(rendered, float(observation.psf_sigma_pix))
    return rendered.to(dtype=image.dtype)


def render_field_cube(
    model: nn.Module,
    *,
    target_header: dict,
    spectral_axis: SpectralAxis,
    shape: tuple[int, int] | None = None,
) -> Tensor:
    fallback = (
        (int(target_header["NAXIS2"]), int(target_header["NAXIS1"]))
        if shape is None
        else shape
    )
    device = _module_device(model)
    ra_deg, dec_deg = _coordinate_grid(target_header, shape=fallback, device=device)
    wavelengths = spectral_axis.to_wavelength().to(device=device, dtype=torch.float32)
    planes = [
        model(ra_deg.to(dtype=torch.float32), dec_deg.to(dtype=torch.float32), lam)
        for lam in wavelengths
    ]
    return torch.stack(planes, dim=0)


def fit_field_model(
    observations: Sequence[FieldObservation],
    model: nn.Module,
    *,
    steps: int = 200,
    lr: float = 5.0e-3,
    num_spectral_samples: int = 9,
    loss_kind: FieldLossKind = "mse",
    charbonnier_eps: float = 1.0e-3,
    huber_delta: float = 5.0e-2,
    loss_reduction: FieldLossReduction = "per_observation",
    lr_schedule: FieldLRSchedule = "constant",
    grad_clip_norm: float | None = None,
    curriculum: FieldCurriculum = "none",
) -> FieldFitResult:
    if not observations:
        raise ValueError("fit_field_model requires at least one observation")
    if steps <= 0:
        raise ValueError("steps must be positive")
    base_lr = float(lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    model.train()
    loss_history: list[float] = []
    for step_idx in range(int(steps)):
        progress = float(step_idx + 1) / max(int(steps), 1)
        set_progress = getattr(model, "set_curriculum_progress", None)
        if callable(set_progress):
            if curriculum == "none":
                set_progress(1.0)
            elif curriculum == "spectral_frequency_ramp":
                set_progress(progress)
            else:
                raise ValueError(f"unsupported curriculum {curriculum!r}")
        if lr_schedule == "constant":
            step_lr = base_lr
        elif lr_schedule == "cosine":
            frac = float(step_idx) / max(int(steps) - 1, 1)
            step_lr = base_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * frac)))
        else:
            raise ValueError(f"unsupported lr_schedule {lr_schedule!r}")
        for group in optimizer.param_groups:
            group["lr"] = step_lr
        optimizer.zero_grad(set_to_none=True)
        total_loss = None
        global_weighted_loss = None
        global_weight_sum = None
        band_weighted_loss = {}
        band_weight_sum = {}
        for observation in observations:
            pred = render_field_observation(
                model,
                observation,
                num_spectral_samples=num_spectral_samples,
            )
            image = observation.image.to(device=pred.device, dtype=pred.dtype)
            residual = pred - image
            weight = (
                observation.weight.to(device=pred.device, dtype=pred.dtype)
                if observation.weight is not None
                else torch.ones_like(pred)
            )
            if observation.mask is not None:
                valid = observation.mask.to(device=pred.device) > 0
                weight = torch.where(valid, weight, torch.zeros_like(weight))
            weighted_loss = torch.sum(
                weight
                * _field_residual_loss(
                    residual,
                    loss_kind=loss_kind,
                    charbonnier_eps=charbonnier_eps,
                    huber_delta=huber_delta,
                )
            )
            weight_sum = weight.sum()
            if loss_reduction == "per_observation":
                denom = torch.clamp(weight_sum, min=1.0e-8)
                obs_loss = weighted_loss / denom
                total_loss = obs_loss if total_loss is None else total_loss + obs_loss
            elif loss_reduction == "per_band":
                band_key = (
                    observation.spectral_min,
                    observation.spectral_max,
                    observation.wavelength,
                    observation.band_name,
                )
                band_weighted_loss[band_key] = (
                    weighted_loss
                    if band_key not in band_weighted_loss
                    else band_weighted_loss[band_key] + weighted_loss
                )
                band_weight_sum[band_key] = (
                    weight_sum
                    if band_key not in band_weight_sum
                    else band_weight_sum[band_key] + weight_sum
                )
            elif loss_reduction == "global":
                global_weighted_loss = (
                    weighted_loss
                    if global_weighted_loss is None
                    else global_weighted_loss + weighted_loss
                )
                global_weight_sum = (
                    weight_sum
                    if global_weight_sum is None
                    else global_weight_sum + weight_sum
                )
            else:
                raise ValueError(f"unsupported loss_reduction {loss_reduction!r}")
        if loss_reduction == "global":
            assert global_weighted_loss is not None and global_weight_sum is not None
            total_loss = global_weighted_loss / torch.clamp(global_weight_sum, min=1.0e-8)
        elif loss_reduction == "per_band":
            total_loss = torch.stack(
                [
                    weighted / torch.clamp(band_weight_sum[band_key], min=1.0e-8)
                    for band_key, weighted in band_weighted_loss.items()
                ]
            ).sum()
        assert total_loss is not None
        loss_history.append(float(total_loss.detach().cpu()))
        total_loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
        optimizer.step()
    model.eval()
    with torch.no_grad():
        final_losses = []
        global_weighted_loss = None
        global_weight_sum = None
        band_weighted_loss = {}
        band_weight_sum = {}
        for observation in observations:
            pred = render_field_observation(
                model,
                observation,
                num_spectral_samples=num_spectral_samples,
            )
            image = observation.image.to(device=pred.device, dtype=pred.dtype)
            residual = pred - image
            weight = (
                observation.weight.to(device=pred.device, dtype=pred.dtype)
                if observation.weight is not None
                else torch.ones_like(pred)
            )
            if observation.mask is not None:
                valid = observation.mask.to(device=pred.device) > 0
                weight = torch.where(valid, weight, torch.zeros_like(weight))
            weighted_loss = torch.sum(
                weight
                * _field_residual_loss(
                    residual,
                    loss_kind=loss_kind,
                    charbonnier_eps=charbonnier_eps,
                    huber_delta=huber_delta,
                )
            )
            weight_sum = weight.sum()
            if loss_reduction == "per_observation":
                denom = torch.clamp(weight_sum, min=1.0e-8)
                final_losses.append(float(weighted_loss / denom))
            elif loss_reduction == "per_band":
                band_key = (
                    observation.spectral_min,
                    observation.spectral_max,
                    observation.wavelength,
                    observation.band_name,
                )
                band_weighted_loss[band_key] = (
                    weighted_loss
                    if band_key not in band_weighted_loss
                    else band_weighted_loss[band_key] + weighted_loss
                )
                band_weight_sum[band_key] = (
                    weight_sum
                    if band_key not in band_weight_sum
                    else band_weight_sum[band_key] + weight_sum
                )
            elif loss_reduction == "global":
                global_weighted_loss = (
                    weighted_loss
                    if global_weighted_loss is None
                    else global_weighted_loss + weighted_loss
                )
                global_weight_sum = (
                    weight_sum
                    if global_weight_sum is None
                    else global_weight_sum + weight_sum
                )
            else:
                raise ValueError(f"unsupported loss_reduction {loss_reduction!r}")
        if loss_reduction == "global":
            assert global_weighted_loss is not None and global_weight_sum is not None
            final_losses = [
                float(global_weighted_loss / torch.clamp(global_weight_sum, min=1.0e-8))
            ]
        elif loss_reduction == "per_band":
            final_losses = [
                float(weighted / torch.clamp(band_weight_sum[band_key], min=1.0e-8))
                for band_key, weighted in band_weighted_loss.items()
            ]
    return FieldFitResult(
        model=model,
        loss_history=tuple(loss_history),
        final_loss=float(sum(final_losses)),
    )
