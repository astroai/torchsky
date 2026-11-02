"""Shared map operator configuration literals and validators."""

from __future__ import annotations

from typing import Literal, cast


MapResamplingKernel = Literal[
    "nearest",
    "bilinear",
    "bicubic",
    "lanczos2",
    "lanczos3",
    "lanczos4",
    "lanczos5",
    "mitchell_netravali",
    "catmull_rom",
    "fft_shift",
    "fft_diagonal_affine",
    "periodic_linear_affine",
    "fft_linear_affine",
    "spectral_linear_affine",
    "spectral_nonuniform",
    "spectral_nonuniform_fast",
    "drizzle_point",
    "drizzle_square",
    "drizzle_turbo",
    "drizzle_gaussian",
    "drizzle_lanczos2",
    "drizzle_lanczos3",
    "conservative_polygon",
    "splat_point",
    "splat_box",
]
MapValueSemantics = Literal["surface_brightness", "flux"]
MapStackingMethod = Literal[
    "weighted_mean",
    "mean",
    "median",
    "sigma_clipped",
    "chi2",
    "min",
    "max",
    "sum",
]

MAP_RESAMPLING_KERNELS: tuple[str, ...] = (
    "nearest",
    "bilinear",
    "bicubic",
    "lanczos2",
    "lanczos3",
    "lanczos4",
    "lanczos5",
    "mitchell_netravali",
    "catmull_rom",
    "fft_shift",
    "fft_diagonal_affine",
    "periodic_linear_affine",
    "fft_linear_affine",
    "spectral_linear_affine",
    "spectral_nonuniform",
    "spectral_nonuniform_fast",
    "drizzle_point",
    "drizzle_square",
    "drizzle_turbo",
    "drizzle_gaussian",
    "drizzle_lanczos2",
    "drizzle_lanczos3",
    "conservative_polygon",
    "splat_point",
    "splat_box",
)
MAP_VALUE_SEMANTICS: tuple[str, ...] = ("surface_brightness", "flux")
MAP_STACKING_METHODS: tuple[str, ...] = (
    "weighted_mean",
    "mean",
    "median",
    "sigma_clipped",
    "chi2",
    "min",
    "max",
    "sum",
)


def validate_map_resampling_kernel(resampling_kernel: str) -> MapResamplingKernel:
    """Validate and return a map resampling kernel name."""
    if resampling_kernel not in MAP_RESAMPLING_KERNELS:
        raise ValueError(f"resampling_kernel must be one of {MAP_RESAMPLING_KERNELS}")
    return cast(MapResamplingKernel, resampling_kernel)  # membership checked above


def validate_map_value_semantics(value_semantics: str) -> MapValueSemantics:
    """Validate and return a map value-semantics mode."""
    if value_semantics not in MAP_VALUE_SEMANTICS:
        raise ValueError(f"value_semantics must be one of {MAP_VALUE_SEMANTICS}")
    return cast(MapValueSemantics, value_semantics)  # membership checked above


def validate_map_stacking_method(stacking_method: str) -> MapStackingMethod:
    """Validate and return a map stacking method."""
    if stacking_method not in MAP_STACKING_METHODS:
        raise ValueError(f"stacking_method must be one of {MAP_STACKING_METHODS}")
    return cast(MapStackingMethod, stacking_method)  # membership checked above


__all__ = [
    "MAP_RESAMPLING_KERNELS",
    "MAP_STACKING_METHODS",
    "MAP_VALUE_SEMANTICS",
    "MapResamplingKernel",
    "MapStackingMethod",
    "MapValueSemantics",
    "validate_map_resampling_kernel",
    "validate_map_stacking_method",
    "validate_map_value_semantics",
]
