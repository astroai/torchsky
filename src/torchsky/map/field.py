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
from ..spectral import SpectralAxis

FieldModelKind = Literal["sine_implicit", "wire_implicit", "gaussian_basis"]


@dataclass(frozen=True)
class FieldDomain:
    center_ra_deg: float
    center_dec_deg: float
    spatial_scale_deg: float
    wavelength_center: float
    wavelength_scale: float


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


def estimate_field_domain(observations: Sequence[FieldObservation]) -> FieldDomain:
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
    )


class _BaseFieldModel(nn.Module):
    def __init__(self, domain: FieldDomain) -> None:
        super().__init__()
        self.domain = domain

    def _normalized_coords(
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
        dra = _wrapped_delta_ra_deg(ra_deg, self.domain.center_ra_deg)
        dra = dra * math.cos(math.radians(self.domain.center_dec_deg))
        ddec = dec_deg - float(self.domain.center_dec_deg)
        dlam = wavelength - float(self.domain.wavelength_center)
        return torch.stack(
            [
                dra / float(self.domain.spatial_scale_deg),
                ddec / float(self.domain.spatial_scale_deg),
                dlam / float(self.domain.wavelength_scale),
            ],
            dim=-1,
        )


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
        layers: list[nn.Linear] = [nn.Linear(3, hidden_dim)]
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
        x = self._normalized_coords(ra_deg, dec_deg, wavelength_angstrom).to(
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
        layers: list[_GaborLayer] = [_GaborLayer(3, hidden_dim, omega0=omega0, sigma0=sigma0)]
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
        x = self._normalized_coords(ra_deg, dec_deg, wavelength_angstrom).to(
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
        coords = self._normalized_coords(ra_deg, dec_deg, wavelength_angstrom).to(
            dtype=self.centers.dtype
        )
        diff = coords.unsqueeze(-2) - self.centers
        scales = F.softplus(self.log_scales) + 1.0e-3
        radial2 = torch.sum(torch.square(diff / scales), dim=-1)
        basis = torch.exp(-0.5 * radial2)
        out = torch.sum(basis * self.amplitudes, dim=-1) + self.bias
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
    raise ValueError("kind must be one of {'sine_implicit', 'wire_implicit', 'gaussian_basis'}")


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
) -> FieldFitResult:
    if not observations:
        raise ValueError("fit_field_model requires at least one observation")
    if steps <= 0:
        raise ValueError("steps must be positive")
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    model.train()
    loss_history: list[float] = []
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        total_loss = None
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
            denom = torch.clamp(weight.sum(), min=1.0e-8)
            obs_loss = torch.sum(weight * torch.square(residual)) / denom
            total_loss = obs_loss if total_loss is None else total_loss + obs_loss
        assert total_loss is not None
        loss_history.append(float(total_loss.detach().cpu()))
        total_loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        final_losses = []
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
            denom = torch.clamp(weight.sum(), min=1.0e-8)
            final_losses.append(float(torch.sum(weight * torch.square(residual)) / denom))
    return FieldFitResult(
        model=model,
        loss_history=tuple(loss_history),
        final_loss=float(sum(final_losses)),
    )
