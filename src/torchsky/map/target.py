"""Internal target-render contract shared across mapmaking and field rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
        )

