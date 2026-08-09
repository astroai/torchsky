"""Learned multiscale features for explicit physical axes."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

__all__ = [
    "AdaptiveAxisFeatures",
    "AdaptiveSpatioSpectralTemporalEncoder",
]


class AdaptiveAxisFeatures(nn.Module):
    """Learn scale selection on one scalar axis inside explicit physical bounds.

    ``reference_scale`` only normalizes the raw coordinate feature. The
    sinusoidal bank uses physical ``length_bounds`` in the same unit as the
    input axis. Learned gates select useful bands, while one bounded continuous
    dilation shifts the ordered log-spaced bank without crossing either bound.
    """

    def __init__(
        self,
        *,
        center: float,
        reference_scale: float,
        length_bounds: tuple[float, float],
        num_bands: int = 4,
        initial_gate: float = 0.9,
    ) -> None:
        super().__init__()
        center = float(center)
        reference_scale = float(reference_scale)
        lower, upper = map(float, length_bounds)
        bands = int(num_bands)
        if not math.isfinite(center):
            raise ValueError("axis center must be finite")
        if not math.isfinite(reference_scale) or reference_scale <= 0.0:
            raise ValueError("axis reference_scale must be finite and positive")
        if (
            not math.isfinite(lower)
            or not math.isfinite(upper)
            or lower <= 0.0
            or upper <= lower
        ):
            raise ValueError("axis length_bounds must be finite, positive, and ordered")
        if bands < 0:
            raise ValueError("num_bands must be non-negative")
        if not math.isfinite(initial_gate) or not 0.0 < initial_gate < 1.0:
            raise ValueError("initial_gate must lie strictly between zero and one")

        self.center = center
        self.reference_scale = reference_scale
        self.length_bounds = (lower, upper)
        self.num_bands = bands
        self.output_dim = 1 + 2 * bands
        if bands == 0:
            base_log_lengths = torch.empty(0, dtype=torch.float64)
            gate_logits = torch.empty(0, dtype=torch.float32)
            scale_shift = torch.empty(0, dtype=torch.float32)
            self._maximum_log_shift = 0.0
        else:
            log_lower = math.log(lower)
            log_upper = math.log(upper)
            log_step = (log_upper - log_lower) / bands
            base_log_lengths = torch.linspace(
                log_lower + 0.5 * log_step,
                log_upper - 0.5 * log_step,
                bands,
                dtype=torch.float64,
            )
            initial_logit = math.log(initial_gate / (1.0 - initial_gate))
            gate_logits = torch.full((bands,), initial_logit, dtype=torch.float32)
            scale_shift = torch.zeros((), dtype=torch.float32)
            self._maximum_log_shift = 0.5 * log_step
        self.register_buffer("base_log_lengths", base_log_lengths)
        self.gate_logits = nn.Parameter(gate_logits)
        self.scale_shift_raw = nn.Parameter(scale_shift)

    def get_extra_state(self) -> dict[str, float | int | tuple[float, float]]:
        return {
            "center": self.center,
            "reference_scale": self.reference_scale,
            "length_bounds": self.length_bounds,
            "num_bands": self.num_bands,
            "maximum_log_shift": self._maximum_log_shift,
        }

    def set_extra_state(
        self,
        state: dict[str, float | int | tuple[float, float]],
    ) -> None:
        center = float(state["center"])
        reference_scale = float(state["reference_scale"])
        length_bounds = tuple(state["length_bounds"])
        num_bands = int(state["num_bands"])
        maximum_log_shift = float(state["maximum_log_shift"])
        if (
            not math.isfinite(center)
            or not math.isfinite(reference_scale)
            or reference_scale <= 0.0
            or len(length_bounds) != 2
            or not all(math.isfinite(float(value)) for value in length_bounds)
            or float(length_bounds[0]) <= 0.0
            or float(length_bounds[1]) <= float(length_bounds[0])
            or num_bands != self.num_bands
            or not math.isfinite(maximum_log_shift)
            or maximum_log_shift < 0.0
        ):
            raise ValueError("invalid adaptive-axis checkpoint metadata")
        self.center = center
        self.reference_scale = reference_scale
        self.length_bounds = (float(length_bounds[0]), float(length_bounds[1]))
        self._maximum_log_shift = maximum_log_shift

    @property
    def gates(self) -> Tensor:
        return torch.sigmoid(self.gate_logits)

    @property
    def length_scales(self) -> Tensor:
        if self.num_bands == 0:
            return self.base_log_lengths
        shift = self._maximum_log_shift * torch.tanh(self.scale_shift_raw)
        return torch.exp(self.base_log_lengths.to(dtype=shift.dtype) + shift)

    @property
    def gate_weighted_length_scale(self) -> Tensor:
        if self.num_bands == 0:
            return self.gate_logits.new_tensor(self.reference_scale)
        gates = self.gates
        log_lengths = torch.log(self.length_scales.to(dtype=gates.dtype))
        return torch.exp(
            torch.sum(gates * log_lengths) / torch.clamp(gates.sum(), min=1.0e-8)
        )

    def regularization(
        self,
        *,
        gate_weight: float = 1.0,
        scale_shift_weight: float = 1.0,
    ) -> Tensor:
        gate_weight = float(gate_weight)
        scale_shift_weight = float(scale_shift_weight)
        if gate_weight < 0.0 or scale_shift_weight < 0.0:
            raise ValueError("adaptive-axis regularization weights must be non-negative")
        if self.num_bands == 0:
            return self.gate_logits.sum() * 0.0
        return gate_weight * self.gates.mean() + scale_shift_weight * torch.square(
            torch.tanh(self.scale_shift_raw)
        )

    def forward(self, value: Tensor) -> Tensor:
        normalized = (value - self.center) / self.reference_scale
        features = [normalized[..., None]]
        if self.num_bands:
            lengths = self.length_scales.to(device=value.device, dtype=value.dtype)
            angular_frequencies = (2.0 * math.pi) / lengths
            phase = (value[..., None] - self.center) * angular_frequencies
            gates = self.gates.to(device=value.device, dtype=value.dtype)
            features.extend((torch.sin(phase) * gates, torch.cos(phase) * gates))
        return torch.cat(features, dim=-1)


class AdaptiveSpatioSpectralTemporalEncoder(nn.Module):
    """Compose independently adaptive detector, spectral, and temporal axes."""

    def __init__(
        self,
        *,
        spatial_center: tuple[float, float],
        spatial_scale: float | tuple[float, float],
        spatial_length_bounds: tuple[float, float],
        wavelength_center: float,
        spectral_scale: float,
        spectral_length_bounds: tuple[float, float],
        time_center_mjd: float,
        temporal_scale_days: float,
        temporal_length_bounds_days: tuple[float, float],
        spatial_bands: int = 4,
        spectral_bands: int = 3,
        temporal_bands: int = 2,
        initial_gate: float = 0.9,
    ) -> None:
        super().__init__()
        if len(spatial_center) != 2:
            raise ValueError("spatial_center must contain x and y")
        if isinstance(spatial_scale, (int, float)):
            spatial_scales = (float(spatial_scale), float(spatial_scale))
        else:
            if len(spatial_scale) != 2:
                raise ValueError("spatial_scale must be scalar or contain x and y")
            spatial_scales = (float(spatial_scale[0]), float(spatial_scale[1]))
        self.x_axis = AdaptiveAxisFeatures(
            center=float(spatial_center[0]),
            reference_scale=spatial_scales[0],
            length_bounds=spatial_length_bounds,
            num_bands=spatial_bands,
            initial_gate=initial_gate,
        )
        self.y_axis = AdaptiveAxisFeatures(
            center=float(spatial_center[1]),
            reference_scale=spatial_scales[1],
            length_bounds=spatial_length_bounds,
            num_bands=spatial_bands,
            initial_gate=initial_gate,
        )
        self.spectral_axis = AdaptiveAxisFeatures(
            center=float(wavelength_center),
            reference_scale=float(spectral_scale),
            length_bounds=spectral_length_bounds,
            num_bands=spectral_bands,
            initial_gate=initial_gate,
        )
        self.temporal_axis = AdaptiveAxisFeatures(
            center=float(time_center_mjd),
            reference_scale=float(temporal_scale_days),
            length_bounds=temporal_length_bounds_days,
            num_bands=temporal_bands,
            initial_gate=initial_gate,
        )
        self.output_dim = (
            self.x_axis.output_dim
            + self.y_axis.output_dim
            + self.spectral_axis.output_dim
            + self.temporal_axis.output_dim
        )

    def regularization(
        self,
        *,
        gate_weight: float = 1.0,
        scale_shift_weight: float = 1.0,
    ) -> Tensor:
        penalties = [
            axis.regularization(
                gate_weight=gate_weight,
                scale_shift_weight=scale_shift_weight,
            )
            for axis in (
                self.x_axis,
                self.y_axis,
                self.spectral_axis,
                self.temporal_axis,
            )
        ]
        return torch.stack(penalties).mean()

    def scale_diagnostics(self) -> dict[str, dict[str, float | list[float]]]:
        """Return serializable learned-scale state for fit reports."""
        diagnostics: dict[str, dict[str, float | list[float]]] = {}
        for name, axis in (
            ("spatial_x", self.x_axis),
            ("spatial_y", self.y_axis),
            ("spectral", self.spectral_axis),
            ("time_mjd", self.temporal_axis),
        ):
            diagnostics[name] = {
                "length_bounds": list(axis.length_bounds),
                "length_scales": axis.length_scales.detach().cpu().tolist(),
                "gates": axis.gates.detach().cpu().tolist(),
                "gate_weighted_length_scale": float(
                    axis.gate_weighted_length_scale.detach().cpu()
                ),
            }
        return diagnostics

    def forward(
        self,
        spatial_x: Tensor,
        spatial_y: Tensor,
        wavelength: Tensor | float,
        time_mjd: Tensor | float | None = None,
    ) -> Tensor:
        wavelength_tensor = torch.as_tensor(
            wavelength,
            device=spatial_x.device,
            dtype=spatial_x.dtype,
        )
        time_tensor = torch.as_tensor(
            self.temporal_axis.center if time_mjd is None else time_mjd,
            device=spatial_x.device,
            dtype=spatial_x.dtype,
        )
        x, y, spectral, temporal = torch.broadcast_tensors(
            spatial_x,
            spatial_y.to(device=spatial_x.device, dtype=spatial_x.dtype),
            wavelength_tensor,
            time_tensor,
        )
        return torch.cat(
            (
                self.x_axis(x),
                self.y_axis(y),
                self.spectral_axis(spectral),
                self.temporal_axis(temporal),
            ),
            dim=-1,
        )
