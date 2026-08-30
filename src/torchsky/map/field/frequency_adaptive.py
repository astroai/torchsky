"""Frequency-controlled experimental continuous-field models.

The implementations in this module adapt two neural-field ideas to torchsky's
native ``(ra, dec, wavelength) -> surface-brightness`` contract:

* :class:`BaconField` follows the multiplicative-filter topology of BACON with
  frozen, explicitly band-limited sinusoidal input filters and intermediate
  multiscale outputs.
* :class:`SASNetField` follows SASNet's frozen frequency support plus learned
  spatial masks, using torchsky's tensor-native hash-grid encoder to localize
  high-frequency capacity.

These are experimental torchsky adaptations rather than vendor/reference-code
ports.  They intentionally have no dependency beyond PyTorch and torchsky.
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
    if any(right <= left for left, right in zip(result, result[1:], strict=True)):
        raise ValueError("bandwidths must be strictly increasing")
    return result


class BaconField(_BaseFieldModel):
    """Band-limited multiplicative-filter field inspired by BACON.

    ``bandwidths`` are cumulative normalized-coordinate bandwidths.  Frozen
    sinusoidal filters at successive layers contribute only the incremental
    bandwidth, so intermediate readouts expose progressively finer scales.
    """

    def __init__(
        self,
        domain: FieldDomain,
        *,
        hidden_dim: int = 128,
        bandwidths: Sequence[float] = (2.0, 4.0, 8.0, 16.0),
        positive: bool = True,
    ) -> None:
        super().__init__(domain)
        if int(hidden_dim) < 1:
            raise ValueError("hidden_dim must be positive")
        self.hidden_dim = int(hidden_dim)
        self.bandwidths = _validate_bandwidths(bandwidths)
        self.positive = bool(positive)
        increments = (
            self.bandwidths[0],
            *(right - left for left, right in zip(self.bandwidths, self.bandwidths[1:], strict=True)),
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
                nn.init.xavier_uniform_(linear.weight)
                nn.init.zeros_(linear.bias)
                self.linears.append(linear)
            readout = nn.Linear(self.hidden_dim, 1)
            nn.init.xavier_uniform_(readout.weight)
            nn.init.zeros_(readout.bias)
            self.readouts.append(readout)

    @property
    def num_levels(self) -> int:
        return len(self.bandwidths)

    def _filter(self, coords: Tensor, level: int) -> Tensor:
        omega = getattr(self, f"omega_{level}")
        return torch.sin(coords @ omega.transpose(0, 1) + self.phases[level])

    def _readout(self, hidden: Tensor, level: int) -> Tensor:
        value = self.readouts[level](hidden).squeeze(-1)
        return F.softplus(value) if self.positive else value

    def forward_level(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
        *,
        level: int,
    ) -> Tensor:
        """Evaluate one cumulative BACON scale, from coarse ``0`` to finest."""
        if int(level) < 0 or int(level) >= self.num_levels:
            raise ValueError(f"level must be in [0, {self.num_levels - 1}]")
        parameter = self.readouts[0].weight
        coords = self._encoded_coords(ra_deg, dec_deg, wavelength_angstrom).to(
            device=parameter.device, dtype=parameter.dtype
        )
        hidden = self._filter(coords, 0)
        if int(level) == 0:
            return self._readout(hidden, 0)
        for index in range(1, int(level) + 1):
            hidden = self._filter(coords, index) * self.linears[index - 1](hidden)
        return self._readout(hidden, int(level))

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


class SASNetField(_BaseFieldModel):
    """Spatially adaptive sinusoidal field inspired by SASNet.

    A frozen bank of full-coordinate sinusoidal features defines the available
    frequency support.  A multiresolution hash grid over sky position predicts
    per-feature masks, so high-frequency capacity can be localized instead of
    leaking across smooth background regions.
    """

    def __init__(
        self,
        domain: FieldDomain,
        *,
        num_frequencies: int = 128,
        max_frequency: float = 24.0,
        hidden_dim: int = 128,
        mask_hidden_dim: int = 64,
        hash_levels: int = 6,
        hash_features_per_level: int = 4,
        hash_log2_size: int = 12,
        hash_base_resolution: int = 8,
        hash_finest_resolution: int = 128,
        positive: bool = True,
    ) -> None:
        super().__init__(domain)
        if int(num_frequencies) < 2:
            raise ValueError("num_frequencies must be at least 2")
        if not math.isfinite(float(max_frequency)) or float(max_frequency) <= 0.0:
            raise ValueError("max_frequency must be finite and positive")
        if int(hidden_dim) < 1 or int(mask_hidden_dim) < 1:
            raise ValueError("hidden dimensions must be positive")
        self.num_frequencies = int(num_frequencies)
        self.positive = bool(positive)
        spatial_dim = self.coord_dim - 1

        directions = torch.randn(self.num_frequencies, self.coord_dim)
        directions = directions / torch.clamp(
            torch.linalg.vector_norm(directions, dim=-1, keepdim=True), min=1.0e-8
        )
        magnitudes = torch.logspace(
            math.log10(0.5),
            math.log10(float(max_frequency)),
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
            base_resolution=int(hash_base_resolution),
            finest_resolution=int(hash_finest_resolution),
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
        """Return learned per-frequency masks for diagnostic inspection."""
        parameter = self.readout[0].weight
        coords_unit = self._unit_box_coords(
            ra_deg, dec_deg, wavelength_angstrom
        ).to(device=parameter.device, dtype=parameter.dtype)
        spatial_unit = coords_unit[..., :-1]
        mask_features = self.mask_encoder(spatial_unit)
        return torch.sigmoid(self.mask_network(mask_features))

    def forward(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        parameter = self.readout[0].weight
        coords = self._encoded_coords(ra_deg, dec_deg, wavelength_angstrom).to(
            device=parameter.device, dtype=parameter.dtype
        )
        masks = self.spatial_masks(ra_deg, dec_deg, wavelength_angstrom)
        phase = self.phase.to(device=coords.device, dtype=coords.dtype)
        frequencies = self.frequencies.to(device=coords.device, dtype=coords.dtype)
        features = torch.sin(coords @ frequencies.transpose(0, 1) + phase) * masks
        value = self.readout(features).squeeze(-1)
        return F.softplus(value) if self.positive else value


__all__ = ["BaconField", "SASNetField"]
