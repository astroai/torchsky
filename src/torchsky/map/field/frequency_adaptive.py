"""Experimental frequency-controlled continuous-field models.

These adaptations share torchsky's ``(ra, dec, wavelength) -> brightness``
contract and observation operators. They remain outside the production model
registry until the equal-budget scientific bakeoff is complete.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .core import _BaseFieldModel
from .hashgrid import HashGridEncoder
from .types import FieldDomain


def _validate_bandwidths(values: Sequence[float]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result:
        raise ValueError("bandwidths must be non-empty")
    if any(not math.isfinite(value) or value <= 0.0 for value in result):
        raise ValueError("bandwidths must be finite and positive")
    if any(right <= left for left, right in zip(result, result[1:])):
        raise ValueError("bandwidths must be strictly increasing")
    return result


def _positive_finite(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


class BaconField(_BaseFieldModel):
    """Band-limited multiplicative-filter field with non-negative outputs.

    ``bandwidths`` declare coordinate-wise *output* angular bandwidths. Each
    latent readout is constrained to half that bandwidth because the
    non-negative sum-of-squares readout doubles Fourier support. Unlike a
    Softplus readout, this polynomial readout preserves an exact finite bound.

    This follows the spectral construction of Lindell et al., BACON, CVPR 2022,
    while adapting the output to non-negative astronomical surface brightness.
    """

    def __init__(
        self,
        domain: FieldDomain,
        *,
        hidden_dim: int = 128,
        bandwidths: Sequence[float] = (2.0, 4.0, 8.0, 16.0),
        readout_terms: int = 2,
        intensity_floor: float = 0.0,
    ) -> None:
        super().__init__(domain)
        if int(hidden_dim) < 1:
            raise ValueError("hidden_dim must be positive")
        if int(readout_terms) < 1:
            raise ValueError("readout_terms must be positive")
        floor = float(intensity_floor)
        if not math.isfinite(floor) or floor < 0.0:
            raise ValueError("intensity_floor must be finite and non-negative")

        self.hidden_dim = int(hidden_dim)
        self.readout_terms = int(readout_terms)
        self.intensity_floor = floor
        self.output_bandwidths = _validate_bandwidths(bandwidths)
        self.bandwidths = self.output_bandwidths
        self.latent_bandwidths = tuple(
            0.5 * bandwidth for bandwidth in self.output_bandwidths
        )
        increments = (
            self.latent_bandwidths[0],
            *(
                right - left
                for left, right in zip(
                    self.latent_bandwidths, self.latent_bandwidths[1:]
                )
            ),
        )

        self.phases = nn.ParameterList()
        self.linears = nn.ModuleList()
        self.readouts = nn.ModuleList()
        for index, increment in enumerate(increments):
            omega = torch.empty(self.hidden_dim, self.coord_dim)
            nn.init.uniform_(omega, -float(increment), float(increment))
            self.register_buffer(f"omega_{index}", omega)
            phase = nn.Parameter(torch.empty(self.hidden_dim))
            nn.init.uniform_(phase, -math.pi, math.pi)
            self.phases.append(phase)
            if index > 0:
                linear = nn.Linear(self.hidden_dim, self.hidden_dim)
                bound = math.sqrt(6.0 / self.hidden_dim)
                nn.init.uniform_(linear.weight, -bound, bound)
                nn.init.zeros_(linear.bias)
                self.linears.append(linear)
            readout = nn.Linear(self.hidden_dim, self.readout_terms)
            nn.init.xavier_uniform_(readout.weight)
            nn.init.zeros_(readout.bias)
            self.readouts.append(readout)

    @property
    def num_levels(self) -> int:
        return len(self.output_bandwidths)

    def _filter(self, coords: Tensor, level: int) -> Tensor:
        omega: Tensor = getattr(self, f"omega_{level}")
        return torch.sin(coords @ omega.transpose(0, 1) + self.phases[level])

    def _readout(self, hidden: Tensor, level: int) -> Tensor:
        terms = self.readouts[level](hidden)
        return terms.square().mean(dim=-1) + self.intensity_floor

    def forward_level(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
        *,
        level: int,
    ) -> Tensor:
        """Evaluate one cumulative scale, from coarse level zero to finest."""
        selected = int(level)
        if selected < 0 or selected >= self.num_levels:
            raise ValueError(f"level must be in [0, {self.num_levels - 1}]")
        parameter = self.readouts[0].weight
        coords = self._encoded_coords(ra_deg, dec_deg, wavelength_angstrom).to(
            device=parameter.device,
            dtype=parameter.dtype,
        )
        hidden = self._filter(coords, 0)
        for index in range(1, selected + 1):
            hidden = self._filter(coords, index) * self.linears[index - 1](hidden)
        return self._readout(hidden, selected)

    def forward(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        return self.forward_level(
            ra_deg,
            dec_deg,
            wavelength_angstrom,
            level=self.num_levels - 1,
        )


def _encoded_spatial_support(domain: FieldDomain, support_deg: float) -> float:
    if domain.coord_mode == "local_tangent":
        return support_deg / float(domain.spatial_scale_deg)
    if domain.coord_mode == "sphere_xyz":
        return math.radians(support_deg)
    raise ValueError(f"unsupported field coordinate mode {domain.coord_mode!r}")


def _supported_hash_resolution(domain: FieldDomain, support_deg: float) -> int:
    encoded = _encoded_spatial_support(domain, support_deg)
    if domain.coord_mode == "local_tangent":
        unit_support = math.tanh(0.5 * encoded)
    else:
        unit_support = math.sin(min(0.5 * encoded, 0.5 * math.pi))
    # One grid interval must be at least as wide as the resolved footprint at
    # the point of greatest coordinate-map sensitivity.
    intervals = max(1, int(math.floor(1.0 / max(unit_support, 1.0e-15))))
    return intervals + 1


class SASNetField(_BaseFieldModel):
    """Spatially localized, frequency-adaptive experimental field.

    The frozen sinusoidal bank follows the frequency-embedding principle of
    Feng et al., SASNet, CVPR 2026. A sky-only hash grid predicts local masks.
    The mask field and nonlinear readout broaden Fourier support, so this model
    is intentionally **not** described as strictly band-limited.

    ``pixel_footprint_deg`` is required. ``psf_support_deg`` may enlarge the
    effective resolvable scale. Both the sinusoidal frequency bank and the
    finest hash-grid resolution are capped from that effective support.
    """

    is_strictly_band_limited = False

    def __init__(
        self,
        domain: FieldDomain,
        *,
        pixel_footprint_deg: float,
        psf_support_deg: float | None = None,
        num_frequencies: int = 128,
        max_frequency: float = 24.0,
        hidden_dim: int = 128,
        mask_hidden_dim: int = 64,
        hash_levels: int = 6,
        hash_features_per_level: int = 4,
        hash_log2_size: int = 12,
        hash_base_resolution: int = 8,
        hash_finest_resolution: int = 128,
        intensity_floor: float = 0.0,
    ) -> None:
        super().__init__(domain)
        footprint = _positive_finite(
            pixel_footprint_deg,
            name="pixel_footprint_deg",
        )
        if psf_support_deg is None:
            psf_support = None
        else:
            psf_support = _positive_finite(
                psf_support_deg,
                name="psf_support_deg",
            )
        if int(num_frequencies) < 2:
            raise ValueError("num_frequencies must be at least 2")
        requested_frequency = _positive_finite(
            max_frequency,
            name="max_frequency",
        )
        if int(hidden_dim) < 1 or int(mask_hidden_dim) < 1:
            raise ValueError("hidden dimensions must be positive")
        if int(hash_base_resolution) < 2:
            raise ValueError("hash_base_resolution must be at least 2")
        if int(hash_finest_resolution) < int(hash_base_resolution):
            raise ValueError(
                "hash_finest_resolution must be at least hash_base_resolution"
            )
        floor = float(intensity_floor)
        if not math.isfinite(floor) or floor < 0.0:
            raise ValueError("intensity_floor must be finite and non-negative")

        self.pixel_footprint_deg = footprint
        self.psf_support_deg = psf_support
        self.effective_spatial_support_deg = max(
            footprint,
            psf_support if psf_support is not None else footprint,
        )
        encoded_support = _encoded_spatial_support(
            domain,
            self.effective_spatial_support_deg,
        )
        self.max_supported_angular_frequency = math.pi / encoded_support
        self.requested_max_frequency = requested_frequency
        self.max_frequency = min(
            requested_frequency,
            self.max_supported_angular_frequency,
        )
        self.requested_hash_finest_resolution = int(hash_finest_resolution)
        supported_resolution = _supported_hash_resolution(
            domain,
            self.effective_spatial_support_deg,
        )
        actual_base_resolution = min(int(hash_base_resolution), supported_resolution)
        actual_finest_resolution = min(
            int(hash_finest_resolution),
            supported_resolution,
        )
        self.num_frequencies = int(num_frequencies)
        self.intensity_floor = floor
        spatial_dim = self.coord_dim - 1

        directions = torch.randn(self.num_frequencies, self.coord_dim)
        directions = directions / torch.clamp(
            torch.linalg.vector_norm(directions, dim=-1, keepdim=True),
            min=1.0e-8,
        )
        minimum_frequency = min(0.5, self.max_frequency)
        magnitudes = torch.logspace(
            math.log10(minimum_frequency),
            math.log10(self.max_frequency),
            steps=self.num_frequencies,
        )
        self.register_buffer("frequencies", directions * magnitudes.unsqueeze(-1))
        phase = torch.empty(self.num_frequencies)
        nn.init.uniform_(phase, -math.pi, math.pi)
        self.register_buffer("phase", phase)

        self.mask_encoder = HashGridEncoder(
            spatial_dim,
            num_levels=int(hash_levels),
            features_per_level=int(hash_features_per_level),
            log2_hashmap_size=int(hash_log2_size),
            base_resolution=actual_base_resolution,
            finest_resolution=actual_finest_resolution,
            interpolation="smoothstep",
            dense_coarse=True,
        )
        mask_dim = int(hash_levels) * int(hash_features_per_level)
        self.mask_network = nn.Sequential(
            nn.Linear(mask_dim, int(mask_hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(mask_hidden_dim), self.num_frequencies),
        )
        self.readout = nn.Sequential(
            nn.Linear(self.num_frequencies, int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), 1),
        )
        for module in (*self.mask_network, *self.readout):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def spatial_masks(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        """Return learned per-frequency masks for diagnostics."""
        parameter = self.readout[0].weight
        coords_unit = self._unit_box_coords(
            ra_deg,
            dec_deg,
            wavelength_angstrom,
        ).to(device=parameter.device, dtype=parameter.dtype)
        mask_features = self.mask_encoder(coords_unit[..., :-1])
        return torch.sigmoid(self.mask_network(mask_features))

    def forward(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        parameter = self.readout[0].weight
        coords = self._encoded_coords(ra_deg, dec_deg, wavelength_angstrom).to(
            device=parameter.device,
            dtype=parameter.dtype,
        )
        masks = self.spatial_masks(ra_deg, dec_deg, wavelength_angstrom)
        features = torch.sin(
            coords @ self.frequencies.transpose(0, 1) + self.phase
        ) * masks
        value = self.readout(features).squeeze(-1)
        return F.softplus(value) + self.intensity_floor


__all__ = ["BaconField", "SASNetField"]
