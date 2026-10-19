"""Public observation, target, configuration, and result types for mapmaking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor

from ..spectral import DataCube, SpectralAxis
from ..sphere.moc import MOC
from ..wcs import WCS
from .operators.config import (
    MapResamplingKernel,
    MapStackingMethod,
    MapValueSemantics,
)


@dataclass
class ImageObservation:
    path: str | None = None
    hdu: int | str | None = None
    weight: float = 1.0
    photometric_scale: float = 1.0
    background_offset: float = 0.0
    wavelength: float | None = None
    wavelength_map: Tensor | None = None
    ifu_wcs: Any | None = None
    spectral_min: float | None = None
    spectral_max: float | None = None
    transmission_center: float | None = None
    transmission_sigma: float | None = None
    band_name: str | None = None
    weight_path: str | None = None
    weight_hdu: int | str | None = 0
    mask_path: str | None = None
    mask_hdu: int | str | None = 0
    primary_beam_path: str | None = None
    primary_beam_hdu: int | str | None = None
    weight_is_inverse_variance: bool = False
    data: Tensor | None = None
    header: dict[str, Any] | None = None
    wcs: WCS | dict[str, Any] | Any | None = None
    weight_map: Tensor | None = None
    mask: Tensor | None = None
    primary_beam: Tensor | None = None
    weight_type: str = "weight"
    weight_thresh: float | None = None
    good_bits: int | None = None
    var_poisson: Tensor | None = None
    var_rnoise: Tensor | None = None
    var_flat: Tensor | None = None
    flux_scale_keyword: str | None = None
    exposure_time_keyword: str | None = None
    gain_keyword: str | None = None
    exposure_time: float | None = None
    head_path: str | None = None
    use_sidecar_head: bool = False
    dcr_zenith_deg: float | None = None
    dcr_parallactic_deg: float = 0.0
    dcr_ref_wavelength: float = 5000.0
    dcr_pressure_hpa: float = 600.0
    dcr_temperature_c: float = 7.0
    tracking_rate: tuple[float, float] | None = None
    mjd_obs: float | None = None
    reference_mjd: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.weight, Tensor) and self.weight.ndim >= 2:
            if self.weight_map is None:
                self.weight_map = self.weight
            self.weight = 1.0


Observation = ImageObservation


@dataclass(frozen=True)
class ResampleConfig:
    """Configuration for single-exposure or mosaic resampling."""

    kernel: MapResamplingKernel = "bilinear"
    pixfrac: float = 1.0
    semantics: MapValueSemantics = "surface_brightness"
    area_scaling: bool | None = None
    spectral_kernel: Literal["linear", "drizzle", "emsm", "nearest"] = "linear"
    spectral_pixfrac: float = 1.0
    emsm_scale: float = 1.0


@dataclass(frozen=True)
class SkyConfig:
    """Configuration for per-exposure, overlap, planar, or mesh sky matching."""

    mode: str = "none"
    stat: str = "huber"
    mesh_size: int = 64
    smooth_scale: int = 3
    regularization: float = 1e-4
    clip_sigma: float = 3.0
    clip_iterations: int = 3
    subtract: bool = True


@dataclass(frozen=True)
class OutlierConfig:
    """Configuration for AstroDrizzle / stcal-compatible cosmic-ray outlier detection."""

    enabled: bool = True
    pixfrac: float = 1.0
    combine: str = "median"
    blot_kernel: Literal[
        "bilinear", "bicubic", "lanczos2", "lanczos3", "lanczos4", "nearest"
    ] = "bilinear"
    snr: tuple[float, float] = (5.0, 4.0)
    scale: tuple[float, float] = (1.2, 0.7)
    min_floor: float = 0.0
    grow: int = 1
    mode: Literal["mask", "weight_zero", "flag_only"] = "mask"


@dataclass(frozen=True)
class StackConfig:
    """Configuration for multi-exposure stacking and uncertainty propagation."""

    method: MapStackingMethod = "weighted_mean"
    clip_sigma: float = 4.0
    clip_iterations: int = 3
    tile_hw: tuple[int, int] | None = None
    tile_size: int | tuple[int, int] | None = None
    footprint_order: int | None = None
    scratch_dir: str | Path | None = None


@dataclass
class MapResult:
    data: Tensor
    weight: Tensor
    header: dict
    inverse_variance: Tensor | None = None
    correlation_factor: Tensor | None = None
    context: Tensor | None = None
    provenance: dict[str, Any] | None = None
    overlap_count: Tensor | None = None
    coverage_moc: MOC | None = None
    observation_mocs: tuple[MOC, ...] | None = None
    var_poisson: Tensor | None = None
    var_rnoise: Tensor | None = None
    var_flat: Tensor | None = None

    @property
    def image(self) -> Tensor:
        return self.data

    @property
    def shape(self) -> torch.Size:
        return self.data.shape

    @property
    def wcs(self) -> WCS:
        return WCS(self.header)


@dataclass
class HealpixResult:
    """Result of direct-to-HEALPix coaddition."""

    data: Any
    weight: Any
    nside: int
    nest: bool = False
    sparse: bool = False
    overlap_count: Any | None = None
    coverage_moc: MOC | None = None

    @property
    def image(self) -> Any:
        return self.data

    @property
    def sparse_map(self) -> Any | None:
        from ..sphere.sparse import HealSparseMap

        return self.data if isinstance(self.data, HealSparseMap) else None


@dataclass(frozen=True)
class ApertureUncertainty:
    """Exact detector variance for one finite weighted map aperture."""

    variance: Tensor
    diagonal_variance: Tensor

    def __post_init__(self) -> None:
        variance = torch.as_tensor(self.variance)
        diagonal = torch.as_tensor(self.diagonal_variance)
        if variance.ndim != 0 or diagonal.ndim != 0:
            raise ValueError("aperture uncertainty values must be scalar tensors")
        if not variance.is_floating_point() or not diagonal.is_floating_point():
            raise ValueError("aperture uncertainty values must be floating point")
        if (
            not bool(torch.isfinite(variance))
            or not bool(torch.isfinite(diagonal))
            or float(variance) < 0.0
            or float(diagonal) <= 0.0
        ):
            raise ValueError(
                "aperture variance must be finite/non-negative and diagonal variance positive"
            )

    @property
    def correlation_factor(self) -> Tensor:
        return torch.as_tensor(self.variance) / torch.as_tensor(
            self.diagonal_variance
        )


@dataclass
class MultibandResult:
    spectral_axis: SpectralAxis
    data: Tensor
    weight: Tensor
    header: dict
    inverse_variance: Tensor | None = None
    correlation_factor: Tensor | None = None
    overlap_count: Tensor | None = None
    coverage_mocs: tuple[MOC | None, ...] | None = None
    band_observation_counts: Tensor | None = None

    @property
    def cube(self) -> Tensor:
        return self.data

    @property
    def wavelength(self) -> Tensor:
        return self.spectral_axis.to_wavelength()


@dataclass(init=False)
class PlannedImagePlaneTarget:
    header: dict
    shape: tuple[int, int]

    def __init__(
        self,
        header: dict[str, Any] | WCS | None = None,
        shape: tuple[int, int] | None = None,
        *,
        wcs: WCS | dict[str, Any] | None = None,
    ) -> None:
        src = wcs if wcs is not None else header
        if src is None:
            raise ValueError("Either header or wcs must be provided to Target")
        if isinstance(src, WCS):
            hdr = src.to_header(relax=True, include_naxis=True)
            inferred_shape = src.array_shape
        elif hasattr(src, "to_header"):
            hdr = dict(src.to_header())
            inferred_shape = getattr(src, "array_shape", None)
        else:
            hdr = dict(src)
            if "NAXIS1" in hdr and "NAXIS2" in hdr:
                inferred_shape = (int(hdr["NAXIS2"]), int(hdr["NAXIS1"]))
            else:
                inferred_shape = None
        if shape is None:
            if inferred_shape is None:
                raise ValueError("shape=(H, W) must be provided when header/wcs lacks NAXIS1/2")
            shape_hw = (int(inferred_shape[0]), int(inferred_shape[1]))
        else:
            shape_hw = (int(shape[0]), int(shape[1]))
        hdr.setdefault("NAXIS", 2)
        hdr["NAXIS1"] = int(shape_hw[1])
        hdr["NAXIS2"] = int(shape_hw[0])
        self.header = hdr
        self.shape = shape_hw

    @property
    def wcs(self) -> WCS:
        return WCS(self.header)


Target = PlannedImagePlaneTarget


@dataclass
class BandMetadataTable:
    band_index: Tensor
    center: Tensor
    lower: Tensor
    upper: Tensor
    observation_count: Tensor
    spectral_weight_sum: Tensor
    band_names: tuple[str | None, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "band_index": self.band_index,
            "center": self.center,
            "lower": self.lower,
            "upper": self.upper,
            "observation_count": self.observation_count,
            "spectral_weight_sum": self.spectral_weight_sum,
            "band_name": self.band_names,
        }


@dataclass
class PlannedImagePlaneCubeTarget:
    spatial_target: PlannedImagePlaneTarget
    spectral_axis: SpectralAxis
    band_metadata: BandMetadataTable
    spectral_bin_edges: Tensor | None = None

    @property
    def header(self) -> dict:
        return self.spatial_target.header

    @property
    def shape(self) -> tuple[int, int]:
        return self.spatial_target.shape

    def to_band_table(self) -> dict[str, object]:
        return self.band_metadata.to_dict()


CubeTarget = PlannedImagePlaneCubeTarget


@dataclass
class CubeResult:
    spectral_axis: SpectralAxis
    band_metadata: BandMetadataTable
    data: Tensor
    weight: Tensor
    header: dict
    spatial_wcs: WCS
    inverse_variance: Tensor | None = None
    correlation_factor: Tensor | None = None
    overlap_count: Tensor | None = None
    coverage_mocs: tuple[MOC | None, ...] | None = None

    @property
    def cube(self) -> Tensor:
        return self.data

    @property
    def wavelength(self) -> Tensor:
        return self.spectral_axis.to_wavelength()

    def to_data_cube(self) -> DataCube:
        return DataCube(
            data=self.data,
            spectral_axis=self.spectral_axis,
            spatial_wcs=self.spatial_wcs,
            header=self.header,
            spectral_axis_index=0,
        )
