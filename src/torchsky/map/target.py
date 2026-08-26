"""Internal target-render contract shared across mapmaking and field rendering."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from ..spectrophotometry import Passband


@dataclass(frozen=True)
class TargetRenderSpec:
    header: dict[str, Any]
    shape: tuple[int, int]
    value_semantics: str = "surface_brightness"
    band_label: str | None = None
    spectral_min: float | None = None
    spectral_max: float | None = None
    transmission_center: float | None = None
    transmission_sigma: float | None = None
    psf_spec: object | None = None
    spectral_unit: str = "Angstrom"
    passband: Passband | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.header, dict):
            raise TypeError("TargetRenderSpec header must be a dict")
        if len(self.shape) != 2 or any(int(size) <= 0 for size in self.shape):
            raise ValueError("TargetRenderSpec shape must contain two positive sizes")
        if not isinstance(self.spectral_unit, str) or not self.spectral_unit.strip():
            raise ValueError("TargetRenderSpec spectral_unit must be explicit")
        if (self.spectral_min is None) ^ (self.spectral_max is None):
            raise ValueError(
                "TargetRenderSpec spectral bounds must be supplied together"
            )
        if self.spectral_min is not None and self.spectral_max is not None:
            lower = float(self.spectral_min)
            upper = float(self.spectral_max)
            if not math.isfinite(lower) or not math.isfinite(upper) or upper < lower:
                raise ValueError(
                    "TargetRenderSpec spectral bounds must be finite and ordered"
                )
        if self.passband is not None:
            if self.spectral_unit.strip().lower() not in {
                "angstrom",
                "angstroms",
                "aa",
            }:
                raise ValueError(
                    "Passband-backed TargetRenderSpec uses Angstrom coordinates"
                )
            if (
                self.transmission_center is not None
                or self.transmission_sigma is not None
            ):
                raise ValueError(
                    "TargetRenderSpec passband and Gaussian transmission are mutually exclusive"
                )

    @classmethod
    def from_header(
        cls,
        header: dict[str, Any],
        *,
        value_semantics: str = "surface_brightness",
        **kwargs,
    ) -> "TargetRenderSpec":
        shape = (int(header["NAXIS2"]), int(header["NAXIS1"]))
        return cls(
            header=dict(header),
            shape=shape,
            value_semantics=value_semantics,
            **kwargs,
        )

    @classmethod
    def from_map_product(
        cls,
        product,
        *,
        band_index: int = 0,
        psf_spec: object | None = None,
    ) -> "TargetRenderSpec":
        band = product.bands[band_index]
        return cls.from_header(
            product.target_header,
            value_semantics=product.value_semantics,
            band_label=band.label,
            spectral_min=band.spectral_min,
            spectral_max=band.spectral_max,
            transmission_center=band.transmission_center,
            transmission_sigma=band.transmission_sigma,
            psf_spec=psf_spec,
            spectral_unit=band.spectral_unit,
            passband=band.passband,
        )
