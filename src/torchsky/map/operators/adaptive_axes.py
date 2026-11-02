"""Learned multiscale features for explicit physical axes."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, cast

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

    base_log_lengths: Tensor

    def __init__(
        self,
        *,
        center: float,
        reference_scale: float,
        length_bounds: tuple[float, float],
        num_bands: int = 4,
        initial_gate: float = 0.9,
        constant: bool = False,
    ) -> None:
        super().__init__()
        center = float(center)
        reference_scale = float(reference_scale)
        lower, upper = map(float, length_bounds)
        bands = 0 if constant else int(num_bands)
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
        self.constant = bool(constant)
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

    @classmethod
    def from_samples(
        cls,
        coordinates: Tensor,
        *,
        sample_spacing: float | None = None,
        maximum_feature_length: float | None = None,
        num_bands: int = 4,
        initial_gate: float = 0.9,
        constant: bool = False,
    ) -> AdaptiveAxisFeatures:
        """Derive identifiable feature bounds from actual axis sampling.

        The shortest admitted period is twice the largest observed coordinate
        gap (or the declared instrument sampling interval, when larger). This
        is a conservative coverage guard patterned after the regular-sampling
        Nyquist bound, not a guarantee against irregular-sampling aliases. The
        longest defaults to twice the sampled span; slower variation is
        represented by the raw coordinate feature instead of a poorly
        identified fraction of a sinusoid.

        ``coordinates`` must contain calibration/evidence locations, not only
        the two domain endpoints. For a spatially sparse two-dimensional
        calibration set, construct x and y axes from its marginal coordinates
        and validate holes in the joint coverage separately.

        ``constant=True`` emits an exact zero feature for every query, including
        outside the sampled coordinates.
        """
        # Setup runs once; float64 on CPU preserves sub-minute differences in MJD.
        values = (
            torch.as_tensor(coordinates)
            .detach()
            .to(device="cpu", dtype=torch.float64)
            .reshape(-1)
        )
        values = values[torch.isfinite(values)]
        unique = torch.unique(values).sort().values
        if constant:
            if unique.numel() == 0:
                raise ValueError(
                    "a constant axis requires a finite calibration coordinate"
                )
            # These scale/bounds are inert metadata: the feature is exactly zero.
            return cls(
                center=float(unique[0]),
                reference_scale=1.0,
                length_bounds=(1.0, 2.0),
                constant=True,
                initial_gate=initial_gate,
            )
        if unique.numel() < 3:
            raise ValueError(
                "at least three distinct finite calibration coordinates are required"
            )
        lower_value = float(unique[0].cpu())
        upper_value = float(unique[-1].cpu())
        span = upper_value - lower_value
        maximum_gap = float(torch.diff(unique).max().cpu())
        spacing = maximum_gap
        if sample_spacing is not None:
            declared_spacing = float(sample_spacing)
            if not math.isfinite(declared_spacing) or declared_spacing <= 0.0:
                raise ValueError("sample_spacing must be finite and positive")
            spacing = max(spacing, declared_spacing)
        minimum_length = 2.0 * spacing
        # ponytail: slower-than-two-span structure stays in the raw trend; add
        # an explicit non-periodic basis if held-out data identify its curvature.
        maximum_length = (
            2.0 * span
            if maximum_feature_length is None
            else float(maximum_feature_length)
        )
        if not math.isfinite(maximum_length) or maximum_length <= minimum_length:
            raise ValueError(
                "sample coverage does not identify an ordered feature-length interval"
            )
        return cls(
            center=0.5 * (lower_value + upper_value),
            reference_scale=0.5 * span,
            length_bounds=(minimum_length, maximum_length),
            num_bands=num_bands,
            initial_gate=initial_gate,
        )

    def initialize_from_evidence(
        self,
        coordinates: Tensor,
        evidence: Tensor,
        *,
        weights: Tensor | None = None,
        shift_steps: int = 33,
        ridge: float = 1.0e-6,
    ) -> dict[str, float | list[float]]:
        """Initialize dilation and gates by profiled weighted harmonic fitting.

        Each candidate bank dilation is fitted jointly with an intercept and
        linear trend. The best weighted residual selects the dilation. A BIC
        comparison between the full fit and a fit with each sine/cosine pair
        removed initializes that band's gate. This is an initializer only:
        downstream training still learns all scales and gates.

        Evidence may be ``(samples,)`` or ``(samples, channels...)``; whiten
        differently scaled/correlated channels before using the latter.
        Suitable evidence includes PSF-basis coefficients measured from
        calibration stars or masked blank-sky summaries. Do not initialize
        nuisance models from the same unpartitioned science pixels used to
        evaluate them.
        """
        if self.num_bands == 0:
            return {
                "selected_log_scale_shift": 0.0,
                "relative_residual": 1.0,
                "band_delta_bic": [],
                "initial_gates": [],
            }
        steps = int(shift_steps)
        ridge_value = float(ridge)
        if steps < 3 or steps % 2 == 0:
            raise ValueError("shift_steps must be an odd integer of at least three")
        if not math.isfinite(ridge_value) or ridge_value <= 0.0:
            raise ValueError("ridge must be finite and positive")

        coordinate_tensor = torch.as_tensor(coordinates).detach().reshape(-1)
        coordinate_tensor = coordinate_tensor - self.center
        evidence_tensor = torch.as_tensor(evidence).detach()
        if (
            evidence_tensor.ndim == 0
            or evidence_tensor.shape[0] != coordinate_tensor.numel()
        ):
            raise ValueError("evidence must have coordinates on its first axis")
        evidence_tensor = evidence_tensor.reshape(coordinate_tensor.numel(), -1)
        device = coordinate_tensor.device
        work_dtype = torch.float64 if device.type == "cpu" else torch.float32
        coordinate_tensor = coordinate_tensor.to(dtype=work_dtype)
        evidence_tensor = evidence_tensor.to(device=device, dtype=work_dtype)
        if weights is None:
            weight_tensor = torch.ones_like(coordinate_tensor)
        else:
            weight_tensor = torch.as_tensor(weights, device=device, dtype=work_dtype)
            if weight_tensor.shape != coordinate_tensor.shape:
                raise ValueError("weights must have the same shape as coordinates")
        valid = (
            torch.isfinite(coordinate_tensor)
            & torch.isfinite(evidence_tensor).all(dim=1)
            & torch.isfinite(weight_tensor)
            & (weight_tensor > 0)
        )
        coordinate_tensor = coordinate_tensor[valid]
        evidence_tensor = evidence_tensor[valid]
        weight_tensor = weight_tensor[valid]
        design_columns = 2 + 2 * self.num_bands
        if coordinate_tensor.numel() < design_columns + 1:
            raise ValueError(
                f"evidence initializer needs at least {design_columns + 1} valid samples"
            )
        weight_tensor = weight_tensor / weight_tensor.mean()

        def fit(log_shift: Tensor, keep: Tensor | None = None) -> tuple[Tensor, Tensor]:
            log_lengths = self.base_log_lengths.to(device=device, dtype=work_dtype)
            lengths = torch.exp(log_lengths + log_shift)
            phase = coordinate_tensor[:, None] * ((2.0 * math.pi) / lengths)
            design = torch.cat(
                (
                    torch.ones_like(coordinate_tensor[:, None]),
                    (coordinate_tensor / self.reference_scale)[:, None],
                    torch.sin(phase),
                    torch.cos(phase),
                ),
                dim=1,
            )
            if keep is not None:
                design = design[:, keep]
            column_scale = torch.sqrt(
                torch.sum(weight_tensor[:, None] * design.square(), dim=0)
                / weight_tensor.sum()
            ).clamp_min(torch.finfo(work_dtype).eps)
            standardized = design / column_scale
            weighted_design = standardized * torch.sqrt(weight_tensor[:, None])
            weighted_evidence = evidence_tensor * torch.sqrt(weight_tensor[:, None])
            gram = weighted_design.T @ weighted_design
            penalty = (
                torch.eye(gram.shape[0], device=device, dtype=work_dtype) * ridge_value
            )
            penalty[0, 0] = 0.0
            coefficients = torch.linalg.solve(
                gram + penalty,
                weighted_design.T @ weighted_evidence,
            )
            residual = evidence_tensor - standardized @ coefficients
            rss = torch.sum(weight_tensor[:, None] * residual.square())
            return rss, coefficients

        maximum_shift = float(self._maximum_log_shift)
        candidate_shifts = torch.linspace(
            -0.999 * maximum_shift,
            0.999 * maximum_shift,
            steps,
            device=device,
            dtype=work_dtype,
        )
        candidate_rss = torch.stack([fit(shift)[0] for shift in candidate_shifts])
        selected_index = int(torch.argmin(candidate_rss))
        selected_shift = candidate_shifts[selected_index]
        full_rss = candidate_rss[selected_index]
        null_keep = torch.zeros(design_columns, device=device, dtype=torch.bool)
        null_keep[:2] = True
        null_rss, _ = fit(selected_shift, null_keep)

        delta_bic: list[Tensor] = []
        channel_count = evidence_tensor.shape[1]
        sample_count = coordinate_tensor.new_tensor(
            float(coordinate_tensor.numel() * channel_count)
        )
        parameter_difference = 2.0 * channel_count
        tiny = torch.finfo(work_dtype).tiny
        for band in range(self.num_bands):
            keep = torch.ones(design_columns, device=device, dtype=torch.bool)
            keep[2 + band] = False
            keep[2 + self.num_bands + band] = False
            ablated_rss = torch.stack(
                [fit(shift, keep)[0] for shift in candidate_shifts]
            ).min()
            delta_bic.append(
                sample_count
                * torch.log(ablated_rss.clamp_min(tiny) / full_rss.clamp_min(tiny))
                - parameter_difference * torch.log(sample_count)
            )
        delta_bic_tensor = torch.stack(delta_bic)
        initial_gates = torch.sigmoid(0.5 * delta_bic_tensor)
        gate_epsilon = math.sqrt(torch.finfo(self.gate_logits.dtype).eps)
        initial_gates = initial_gates.clamp(gate_epsilon, 1.0 - gate_epsilon)

        shift_fraction = selected_shift / maximum_shift
        shift_fraction = shift_fraction.clamp(-0.999999, 0.999999)
        with torch.no_grad():
            self.scale_shift_raw.copy_(
                torch.atanh(shift_fraction).to(
                    device=self.scale_shift_raw.device,
                    dtype=self.scale_shift_raw.dtype,
                )
            )
            self.gate_logits.copy_(
                torch.logit(initial_gates).to(
                    device=self.gate_logits.device,
                    dtype=self.gate_logits.dtype,
                )
            )
        return {
            "selected_log_scale_shift": float(selected_shift.cpu()),
            "relative_residual": float((full_rss / null_rss.clamp_min(tiny)).cpu()),
            "band_delta_bic": delta_bic_tensor.cpu().tolist(),
            "initial_gates": initial_gates.cpu().tolist(),
        }

    def get_extra_state(self) -> dict[str, float | int | tuple[float, float]]:
        return {
            "center": self.center,
            "reference_scale": self.reference_scale,
            "length_bounds": self.length_bounds,
            "num_bands": self.num_bands,
            "constant": self.constant,
            "maximum_log_shift": self._maximum_log_shift,
        }

    def set_extra_state(
        self,
        state: Mapping[str, Any],
    ) -> None:
        center = float(state["center"])
        reference_scale = float(state["reference_scale"])
        length_bounds = tuple(state["length_bounds"])
        num_bands = int(state["num_bands"])
        maximum_log_shift = float(state["maximum_log_shift"])
        constant = state.get("constant", False)
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
            or not isinstance(constant, bool)
            or (constant and num_bands != 0)
        ):
            raise ValueError("invalid adaptive-axis checkpoint metadata")
        self.center = center
        self.reference_scale = reference_scale
        self.length_bounds = (float(length_bounds[0]), float(length_bounds[1]))
        self._maximum_log_shift = maximum_log_shift
        self.constant = constant

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
            raise ValueError(
                "adaptive-axis regularization weights must be non-negative"
            )
        if self.num_bands == 0:
            return self.gate_logits.sum() * 0.0
        return gate_weight * self.gates.mean() + scale_shift_weight * torch.square(
            torch.tanh(self.scale_shift_raw)
        )

    def _forward_centered(self, centered: Tensor) -> Tensor:
        if self.constant:
            return centered[..., None] * 0.0
        normalized = centered / self.reference_scale
        features = [normalized[..., None]]
        if self.num_bands:
            lengths = self.length_scales.to(
                device=centered.device, dtype=centered.dtype
            )
            angular_frequencies = (2.0 * math.pi) / lengths
            phase = centered[..., None] * angular_frequencies
            gates = self.gates.to(device=centered.device, dtype=centered.dtype)
            features.extend((torch.sin(phase) * gates, torch.cos(phase) * gates))
        return torch.cat(features, dim=-1)

    def forward(self, value: Tensor) -> Tensor:
        return self._forward_centered(value - self.center)


class AdaptiveSpatioSpectralTemporalEncoder(nn.Module):
    """Compose independently adaptive detector, spectral, and temporal axes."""

    def __init__(
        self,
        *,
        spatial_center: tuple[float, float],
        spatial_scale: float | tuple[float, float],
        spatial_length_bounds: (
            tuple[float, float] | tuple[tuple[float, float], tuple[float, float]]
        ),
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
        spectral_constant: bool = False,
        temporal_constant: bool = False,
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
        if isinstance(spatial_length_bounds[0], (int, float)):
            # A single (lower, upper) pair shared by both axes.
            x_length_bounds = y_length_bounds = cast(
                "tuple[float, float]", spatial_length_bounds
            )
        else:
            if len(spatial_length_bounds) != 2:
                raise ValueError(
                    "spatial_length_bounds must be one pair or separate x/y pairs"
                )
            x_length_bounds, y_length_bounds = cast(
                "tuple[tuple[float, float], tuple[float, float]]", spatial_length_bounds
            )
        self.x_axis = AdaptiveAxisFeatures(
            center=float(spatial_center[0]),
            reference_scale=spatial_scales[0],
            length_bounds=x_length_bounds,
            num_bands=spatial_bands,
            initial_gate=initial_gate,
        )
        self.y_axis = AdaptiveAxisFeatures(
            center=float(spatial_center[1]),
            reference_scale=spatial_scales[1],
            length_bounds=y_length_bounds,
            num_bands=spatial_bands,
            initial_gate=initial_gate,
        )
        self.spectral_axis = AdaptiveAxisFeatures(
            center=float(wavelength_center),
            reference_scale=float(spectral_scale),
            length_bounds=spectral_length_bounds,
            num_bands=spectral_bands,
            initial_gate=initial_gate,
            constant=spectral_constant,
        )
        self.temporal_axis = AdaptiveAxisFeatures(
            center=float(time_center_mjd),
            reference_scale=float(temporal_scale_days),
            length_bounds=temporal_length_bounds_days,
            num_bands=temporal_bands,
            initial_gate=initial_gate,
            constant=temporal_constant,
        )
        self.output_dim = (
            self.x_axis.output_dim
            + self.y_axis.output_dim
            + self.spectral_axis.output_dim
            + self.temporal_axis.output_dim
        )

    @classmethod
    def from_samples(
        cls,
        *,
        spatial_x: Tensor,
        spatial_y: Tensor,
        wavelength: Tensor,
        time_mjd: Tensor,
        spatial_sample_spacing: float | tuple[float, float] | None = None,
        spectral_sample_spacing: float | None = None,
        temporal_sample_spacing_days: float | None = None,
        spatial_maximum_feature_length: float | tuple[float, float] | None = None,
        spectral_maximum_feature_length: float | None = None,
        temporal_maximum_feature_length_days: float | None = None,
        spatial_bands: int = 4,
        spectral_bands: int = 3,
        temporal_bands: int = 2,
        initial_gate: float = 0.9,
        spectral_constant: bool = False,
        temporal_constant: bool = False,
    ) -> AdaptiveSpatioSpectralTemporalEncoder:
        """Build all physical axis contracts from calibration sampling."""
        if isinstance(spatial_sample_spacing, (int, float)):
            spatial_spacings = (
                float(spatial_sample_spacing),
                float(spatial_sample_spacing),
            )
        elif spatial_sample_spacing is None:
            spatial_spacings = (None, None)
        else:
            if len(spatial_sample_spacing) != 2:
                raise ValueError(
                    "spatial_sample_spacing must be scalar or contain x and y"
                )
            spatial_spacings = spatial_sample_spacing
        if isinstance(spatial_maximum_feature_length, (int, float)):
            spatial_maxima = (
                float(spatial_maximum_feature_length),
                float(spatial_maximum_feature_length),
            )
        elif spatial_maximum_feature_length is None:
            spatial_maxima = (None, None)
        else:
            if len(spatial_maximum_feature_length) != 2:
                raise ValueError(
                    "spatial_maximum_feature_length must be scalar or contain x and y"
                )
            spatial_maxima = spatial_maximum_feature_length

        x_axis = AdaptiveAxisFeatures.from_samples(
            spatial_x,
            sample_spacing=spatial_spacings[0],
            maximum_feature_length=spatial_maxima[0],
            num_bands=spatial_bands,
            initial_gate=initial_gate,
        )
        y_axis = AdaptiveAxisFeatures.from_samples(
            spatial_y,
            sample_spacing=spatial_spacings[1],
            maximum_feature_length=spatial_maxima[1],
            num_bands=spatial_bands,
            initial_gate=initial_gate,
        )
        spectral_axis = AdaptiveAxisFeatures.from_samples(
            wavelength,
            sample_spacing=spectral_sample_spacing,
            maximum_feature_length=spectral_maximum_feature_length,
            num_bands=spectral_bands,
            initial_gate=initial_gate,
            constant=spectral_constant,
        )
        temporal_axis = AdaptiveAxisFeatures.from_samples(
            time_mjd,
            sample_spacing=temporal_sample_spacing_days,
            maximum_feature_length=temporal_maximum_feature_length_days,
            num_bands=temporal_bands,
            initial_gate=initial_gate,
            constant=temporal_constant,
        )
        return cls(
            spatial_center=(x_axis.center, y_axis.center),
            spatial_scale=(x_axis.reference_scale, y_axis.reference_scale),
            spatial_length_bounds=(x_axis.length_bounds, y_axis.length_bounds),
            wavelength_center=spectral_axis.center,
            spectral_scale=spectral_axis.reference_scale,
            spectral_length_bounds=spectral_axis.length_bounds,
            time_center_mjd=temporal_axis.center,
            temporal_scale_days=temporal_axis.reference_scale,
            temporal_length_bounds_days=temporal_axis.length_bounds,
            spatial_bands=spatial_bands,
            spectral_bands=spectral_bands,
            temporal_bands=temporal_bands,
            initial_gate=initial_gate,
            spectral_constant=spectral_constant,
            temporal_constant=temporal_constant,
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
                "constant": axis.constant,
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
        wavelength_offset = (
            wavelength - self.spectral_axis.center
            if isinstance(wavelength, Tensor)
            else float(wavelength) - self.spectral_axis.center
        )
        wavelength_offset_tensor = torch.as_tensor(
            wavelength_offset, device=spatial_x.device, dtype=spatial_x.dtype
        )
        if time_mjd is None:
            time_offset: Tensor | float = 0.0
        elif isinstance(time_mjd, Tensor):
            time_offset = time_mjd - self.temporal_axis.center
        else:
            time_offset = float(time_mjd) - self.temporal_axis.center
        time_offset_tensor = torch.as_tensor(
            time_offset, device=spatial_x.device, dtype=spatial_x.dtype
        )
        x, y, spectral_offset, temporal_offset = torch.broadcast_tensors(
            spatial_x,
            spatial_y.to(device=spatial_x.device, dtype=spatial_x.dtype),
            wavelength_offset_tensor,
            time_offset_tensor,
        )
        return torch.cat(
            (
                self.x_axis(x),
                self.y_axis(y),
                self.spectral_axis._forward_centered(spectral_offset),
                self.temporal_axis._forward_centered(temporal_offset),
            ),
            dim=-1,
        )
