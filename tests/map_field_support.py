"""Shared helpers for the map_field_support test modules."""

from __future__ import annotations
import math
import torch
from torchsky.map.field import (
    FieldDomain,
    ForwardContext,
)
from torchsky.map.field.core import _BaseFieldModel


def _tan_header(
    size: int,
    *,
    crval1: float,
    crval2: float,
    scale_deg: float,
) -> dict:
    return {
        "NAXIS": 2,
        "NAXIS1": size,
        "NAXIS2": size,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": size / 2.0,
        "CRPIX2": size / 2.0,
        "CRVAL1": crval1,
        "CRVAL2": crval2,
        "CD1_1": -scale_deg,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": scale_deg,
    }


class _ConstantField(_BaseFieldModel):
    def __init__(self, domain: FieldDomain, value: float) -> None:
        super().__init__(domain)
        self.value = torch.nn.Parameter(torch.tensor(float(value), dtype=torch.float32))

    def forward(
        self,
        ra_deg: torch.Tensor,
        dec_deg: torch.Tensor,
        wavelength_angstrom: torch.Tensor | float,
        *,
        forward_context: ForwardContext | None = None,
    ) -> torch.Tensor:
        del dec_deg, wavelength_angstrom, forward_context
        return torch.zeros_like(ra_deg) + self.value


class _GaussianSpotField(_BaseFieldModel):
    def __init__(
        self,
        domain: FieldDomain,
        *,
        amplitude: float,
        center_ra_deg: float,
        center_dec_deg: float,
        sigma_deg: float,
    ) -> None:
        super().__init__(domain)
        sign = 1.0 if amplitude >= 0.0 else -1.0
        magnitude = max(abs(float(amplitude)), 1.0e-6)
        self.log_amplitude = torch.nn.Parameter(
            torch.tensor(math.log(magnitude), dtype=torch.float32)
        )
        self.amplitude_sign = float(sign)
        self.center_ra_deg = torch.nn.Parameter(
            torch.tensor(float(center_ra_deg), dtype=torch.float32)
        )
        self.center_dec_deg = torch.nn.Parameter(
            torch.tensor(float(center_dec_deg), dtype=torch.float32)
        )
        self.sigma_deg = float(sigma_deg)

    def forward(
        self,
        ra_deg: torch.Tensor,
        dec_deg: torch.Tensor,
        wavelength_angstrom: torch.Tensor | float,
        *,
        forward_context: ForwardContext | None = None,
    ) -> torch.Tensor:
        del wavelength_angstrom, forward_context
        sigma = torch.tensor(self.sigma_deg, device=ra_deg.device, dtype=ra_deg.dtype)
        dx = (
            ra_deg - self.center_ra_deg.to(device=ra_deg.device, dtype=ra_deg.dtype)
        ) / sigma
        dy = (
            dec_deg - self.center_dec_deg.to(device=dec_deg.device, dtype=dec_deg.dtype)
        ) / sigma
        amplitude = self.log_amplitude.to(
            device=ra_deg.device, dtype=ra_deg.dtype
        ).exp()
        return (
            self.amplitude_sign
            * amplitude
            * torch.exp(-0.5 * (dx.square() + dy.square()))
        )
