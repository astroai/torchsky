"""Adaptive local-capacity field components from residual evidence."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from ...wcs import WCS
from .core import _BaseFieldModel
from .quality import FieldResidualProduct, FieldResidualTile, select_field_residual_tiles
from .types import FieldObservation


@dataclass(frozen=True)
class ResidualTileBasisSpec:
    """Encoded centers, scales, and amplitudes for residual-tile basis atoms."""

    centers: Tensor
    scales: Tensor
    amplitudes: Tensor
    tiles: tuple[FieldResidualTile, ...]


def _observation_wavelength(observation: FieldObservation) -> float:
    if observation.wavelength is not None:
        return float(observation.wavelength)
    if observation.transmission_center is not None:
        return float(observation.transmission_center)
    if observation.spectral_min is not None and observation.spectral_max is not None:
        return 0.5 * (float(observation.spectral_min) + float(observation.spectral_max))
    return float(observation.header.get("CRVAL3", 0.0))


def _tile_weighted_mean_residual(
    product: FieldResidualProduct | None,
    tile: FieldResidualTile,
) -> float:
    if product is None:
        return 0.0
    residual = torch.as_tensor(product.residual)[tile.y0 : tile.y1, tile.x0 : tile.x1]
    weight = torch.as_tensor(product.weight, device=residual.device, dtype=residual.dtype)[
        tile.y0 : tile.y1, tile.x0 : tile.x1
    ]
    support = torch.as_tensor(product.support, device=residual.device).to(dtype=torch.bool)[
        tile.y0 : tile.y1, tile.x0 : tile.x1
    ]
    valid_weight = torch.where(
        support,
        torch.clamp(weight, min=0.0),
        torch.zeros_like(weight),
    )
    denom = valid_weight.sum()
    if float(denom.detach().cpu()) <= 0.0:
        return 0.0
    return float((valid_weight * residual).sum().detach().cpu() / denom.detach().cpu())


def _tile_encoded_center_and_scale(
    model: _BaseFieldModel,
    observation: FieldObservation,
    tile: FieldResidualTile,
    *,
    wavelength: float,
    spectral_scale_encoded: float,
) -> tuple[Tensor, Tensor]:
    wcs = WCS(observation.header)
    x_center = 0.5 * (float(tile.x0) + float(tile.x1 - 1))
    y_center = 0.5 * (float(tile.y0) + float(tile.y1 - 1))
    x_samples = torch.tensor(
        [x_center, float(tile.x0), float(tile.x1 - 1), x_center, x_center],
        dtype=torch.float64,
    )
    y_samples = torch.tensor(
        [y_center, y_center, y_center, float(tile.y0), float(tile.y1 - 1)],
        dtype=torch.float64,
    )
    ra, dec = wcs.pixel_to_world(x_samples, y_samples)
    lam = torch.full_like(ra.to(dtype=torch.float32), float(wavelength))
    encoded = model._encoded_coords(
        ra.to(dtype=torch.float32),
        dec.to(dtype=torch.float32),
        lam,
    ).detach()
    center = encoded[0]
    spatial_delta = torch.abs(encoded[1:, :-1] - center[:-1]).amax(dim=0)
    scale = torch.empty_like(center)
    scale[:-1] = torch.clamp(spatial_delta, min=1.0e-6)
    scale[-1] = max(float(spectral_scale_encoded), 1.0e-6)
    return center, scale


def residual_tile_basis_spec_from_observation(
    base_model: _BaseFieldModel,
    observation: FieldObservation,
    tiles: Sequence[FieldResidualTile],
    *,
    residual_product: FieldResidualProduct | None = None,
    spectral_scale_encoded: float = 1.0,
) -> ResidualTileBasisSpec:
    """Convert residual-image tiles into encoded basis-atom parameters."""
    if not tiles:
        raise ValueError("at least one residual tile is required")
    wavelength = _observation_wavelength(observation)
    centers: list[Tensor] = []
    scales: list[Tensor] = []
    amplitudes: list[float] = []
    for tile in tiles:
        center, scale = _tile_encoded_center_and_scale(
            base_model,
            observation,
            tile,
            wavelength=wavelength,
            spectral_scale_encoded=spectral_scale_encoded,
        )
        centers.append(center)
        scales.append(scale)
        amplitudes.append(_tile_weighted_mean_residual(residual_product, tile))
    return ResidualTileBasisSpec(
        centers=torch.stack(centers, dim=0).to(dtype=torch.float32),
        scales=torch.stack(scales, dim=0).to(dtype=torch.float32),
        amplitudes=torch.tensor(amplitudes, dtype=torch.float32),
        tiles=tuple(tiles),
    )


class ResidualTileBasisField(_BaseFieldModel):
    """Add sparse local residual-tile basis atoms on top of a diffuse field."""

    def __init__(
        self,
        base_model: _BaseFieldModel,
        spec: ResidualTileBasisSpec,
        *,
        learn_scales: bool = False,
    ) -> None:
        super().__init__(base_model.domain)
        if int(spec.centers.shape[0]) == 0:
            raise ValueError("ResidualTileBasisField requires at least one tile")
        if spec.centers.shape != spec.scales.shape:
            raise ValueError("centers and scales must have matching shapes")
        if spec.centers.shape[-1] != self.coord_dim:
            raise ValueError("residual-tile basis dimension must match field domain")
        if int(spec.amplitudes.numel()) != int(spec.centers.shape[0]):
            raise ValueError("amplitudes length must match tile count")
        self.base_model = base_model
        self.tiles = tuple(spec.tiles)
        self.register_buffer("tile_centers", spec.centers.detach().clone())
        if learn_scales:
            self.log_tile_scales = nn.Parameter(torch.log(torch.clamp(spec.scales, min=1.0e-6)))
        else:
            self.register_buffer("tile_scales", spec.scales.detach().clone())
            self.log_tile_scales = None
        self.tile_amplitudes = nn.Parameter(spec.amplitudes.detach().clone())

    @property
    def tile_count(self) -> int:
        return int(self.tile_centers.shape[0])

    def _resolved_scales(self, *, dtype: torch.dtype, device: torch.device) -> Tensor:
        if self.log_tile_scales is not None:
            return torch.exp(self.log_tile_scales).to(device=device, dtype=dtype)
        return self.tile_scales.to(device=device, dtype=dtype)

    def residual_correction(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        coords = self._encoded_coords(ra_deg, dec_deg, wavelength_angstrom)
        centers = self.tile_centers.to(device=coords.device, dtype=coords.dtype)
        scales = self._resolved_scales(dtype=coords.dtype, device=coords.device)
        diff = (coords.unsqueeze(-2) - centers) / torch.clamp(scales, min=1.0e-6)
        basis = torch.exp(-0.5 * torch.sum(diff * diff, dim=-1))
        amplitudes = self.tile_amplitudes.to(device=coords.device, dtype=coords.dtype)
        return torch.sum(basis * amplitudes, dim=-1)

    def forward(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        base = self.base_model(ra_deg, dec_deg, wavelength_angstrom)
        return base + self.residual_correction(ra_deg, dec_deg, wavelength_angstrom).to(
            dtype=base.dtype,
            device=base.device,
        )


def make_residual_tile_basis_field(
    base_model: _BaseFieldModel,
    observation: FieldObservation,
    *,
    residual_product: FieldResidualProduct,
    tile_hw: tuple[int, int],
    max_tiles: int,
    min_support_fraction: float = 0.0,
    spectral_scale_encoded: float = 1.0,
    learn_scales: bool = False,
) -> ResidualTileBasisField:
    """Build an adaptive residual-tile field from rendered residual evidence."""
    tiles = select_field_residual_tiles(
        residual_product,
        tile_hw=tile_hw,
        max_tiles=max_tiles,
        min_support_fraction=min_support_fraction,
    )
    if not tiles:
        raise ValueError("no residual tiles satisfy the requested support criteria")
    spec = residual_tile_basis_spec_from_observation(
        base_model,
        observation,
        tiles,
        residual_product=residual_product,
        spectral_scale_encoded=spectral_scale_encoded,
    )
    return ResidualTileBasisField(base_model, spec, learn_scales=learn_scales)


__all__ = [
    "ResidualTileBasisField",
    "ResidualTileBasisSpec",
    "make_residual_tile_basis_field",
    "residual_tile_basis_spec_from_observation",
]
