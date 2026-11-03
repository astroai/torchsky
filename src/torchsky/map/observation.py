"""Public observation, target, configuration, and result types for mapmaking."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

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
from .quality import DQPolicy, QualityProduct
from ..device import _as_tensor


@dataclass
class ImageObservation:
    """Describe one image exposure and its calibration metadata.

    Args:
        path: Optional input path consumed through the torchfits boundary.
        data: Optional in-memory image tensor, used instead of ``path``.
        header: FITS-style header mapping for in-memory data.
        wcs: Explicit WCS object or header mapping.
        weight: Scalar exposure weight.
        photometric_scale: Multiplicative flux calibration.
        background_offset: Additive background removed before resampling.
        weight_map: Optional per-pixel relative weight tensor (an inverse
            variance only when ``weight_is_inverse_variance=True``).
        mask: Optional Boolean valid-pixel mask.

    Examples:
        >>> import torch
        >>> from torchsky.map import Observation
        >>> observation = Observation(data=torch.ones(2, 2), header={})
        >>> tuple(observation.data.shape)
        (2, 2)
    """

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
    dq_path: str | None = None
    dq_hdu: int | str | None = "DQ"
    dq: Tensor | None = None
    dq_policy: DQPolicy | None = None
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
class IFUObservation:
    """In-memory detector product for runtime-independent IFU cube building.

    Each detector sample (one pixel of ``sci``) is mapped to ``(ra, dec,
    wavelength)`` by ``detector_to_world``. The ``drizzle`` kernel needs the sky
    footprint of every sample, given in one of three ways:

    - by default, the detector pixel square mapped through ``detector_to_world``,
      which is only meaningful when neighbouring detector pixels are neighbours on
      the sky. When slice ids are known (``slice_id`` or the transform's
      ``slice_id_map``), a sample without a same-slice neighbour along a detector
      axis has no grid footprint and planning raises;
    - ``detector_pixel_footprint``: explicit per-sample polygons, shape
      ``(H, W, V, 2)`` in detector pixels or ``(H, W, V, 3)`` in
      ``(ra, dec, wavelength)``;
    - ``aperture_diameter_arcsec``: a circular aperture (an optical fibre)
      centred on each sample's sky position, scalar or one value per sample.
      Its exact area overlap with every output spaxel is used, so fibre-fed
      (row-stacked) spectra need no pixel-grid adjacency.
    """

    sci: Tensor
    detector_to_world: Any
    dq: Tensor | None = None
    variance_components: Mapping[str, Tensor] = field(default_factory=dict)
    slice_id: Tensor | None = None
    detector_pixel_footprint: Tensor | None = None
    spectral_unit: str = "Angstrom"
    spectral_type: str = "WAVE"
    value_semantics: MapValueSemantics = "surface_brightness"
    weight: Tensor | float | None = None
    dq_policy: DQPolicy | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    aperture_diameter_arcsec: Tensor | float | None = None

    def __post_init__(self) -> None:
        science = torch.as_tensor(self.sci)
        if science.ndim != 2 or not science.is_floating_point():
            raise ValueError("IFU science must be a floating detector image")
        if self.aperture_diameter_arcsec is not None:
            if self.detector_pixel_footprint is not None:
                raise ValueError(
                    "give either detector_pixel_footprint or aperture_diameter_arcsec"
                )
            aperture = _as_tensor(
                self.aperture_diameter_arcsec,
                device=science.device,
                dtype=science.dtype,
            )
            if aperture.numel() != 1 and aperture.shape != science.shape:
                raise ValueError(
                    "aperture_diameter_arcsec must be scalar or match the science image"
                )
            if not bool(torch.isfinite(aperture).all()) or bool((aperture <= 0).any()):
                raise ValueError("aperture_diameter_arcsec must be finite and positive")
            object.__setattr__(self, "aperture_diameter_arcsec", aperture)
        if not callable(getattr(self.detector_to_world, "pixel_to_world", None)):
            raise TypeError("detector_to_world must expose pixel_to_world(x, y)")
        if self.dq is not None:
            dq = DQPolicy._coerce(self.dq)
            if dq.shape != science.shape:
                raise ValueError("IFU DQ must match the science image")
            object.__setattr__(self, "dq", dq.to(device=science.device))
        components: dict[str, Tensor] = {}
        for name, value in self.variance_components.items():
            if not isinstance(name, str) or not name:
                raise ValueError("variance component names must be non-empty")
            component = _as_tensor(value, device=science.device, dtype=science.dtype)
            if component.shape != science.shape:
                raise ValueError("IFU variance components must match the science image")
            components[name] = component
        if self.slice_id is not None:
            slice_id = _as_tensor(
                self.slice_id, device=science.device, dtype=torch.int64
            )
            if slice_id.shape != science.shape:
                raise ValueError("IFU slice_id must match the science image")
            object.__setattr__(self, "slice_id", slice_id)
        if self.detector_pixel_footprint is not None:
            footprint = _as_tensor(
                self.detector_pixel_footprint,
                device=science.device,
                dtype=science.dtype,
            )
            if (
                footprint.ndim != 4
                or footprint.shape[:2] != science.shape
                or footprint.shape[-1] not in {2, 3}
                or footprint.shape[-2] < 3
            ):
                raise ValueError(
                    "detector_pixel_footprint must have shape (H, W, vertices>=3, 2|3)"
                )
            object.__setattr__(self, "detector_pixel_footprint", footprint)
        if not isinstance(self.spectral_unit, str) or not self.spectral_unit.strip():
            raise ValueError("IFU spectral_unit must be explicit")
        if self.value_semantics not in {"surface_brightness", "flux"}:
            raise ValueError("IFU value_semantics must be surface_brightness or flux")
        if self.weight is not None:
            weight = _as_tensor(self.weight, device=science.device, dtype=science.dtype)
            if weight.numel() != 1 and weight.shape != science.shape:
                raise ValueError("IFU weight must be scalar or match the science image")
            object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "sci", science)
        object.__setattr__(self, "variance_components", components)
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.sci.shape[0]), int(self.sci.shape[1])


@dataclass(frozen=True)
class ResampleConfig:
    """Configuration for single-exposure or mosaic resampling."""

    kernel: MapResamplingKernel = "bilinear"
    pixfrac: float = 1.0
    semantics: MapValueSemantics = "surface_brightness"
    area_scaling: bool | None = None
    spectral_kernel: Literal["linear", "drizzle", "emsm", "msm", "nearest"] = "linear"
    spectral_pixfrac: float = 1.0
    emsm_scale: float = 1.0
    spatial_roi: Any | None = None
    spectral_roi: Any | None = None
    resolving_power: Any | None = None
    response_chunk_size: int = 65_536
    max_response_entries: int | None = None
    footprint_tolerance: float = 0.02
    footprint_max_depth: int = 10


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
    parity_mode: bool = False
    dq_policy: DQPolicy | None = None


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
    dq: Tensor | None = None
    usability_mask: Tensor | None = None
    rejection_log: tuple[dict[str, Any], ...] = ()
    dq_policy: DQPolicy | None = None

    @property
    def image(self) -> Tensor:
        return self.data

    @property
    def shape(self) -> torch.Size:
        return self.data.shape

    @property
    def wcs(self) -> WCS:
        return WCS(self.header)

    @property
    def quality(self) -> QualityProduct | None:
        if self.dq is None or self.usability_mask is None:
            return None
        return QualityProduct(
            dq=self.dq,
            usability_mask=self.usability_mask,
            rejection_log=self.rejection_log,
            policy=self.dq_policy,
        )


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
        from ..sphere.sparse import SparseHealpixMap

        return self.data if isinstance(self.data, SparseHealpixMap) else None


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
        return torch.as_tensor(self.variance) / torch.as_tensor(self.diagonal_variance)


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
            hdr = dict(getattr(src, "to_header")())
            inferred_shape = getattr(src, "array_shape", None)
        else:
            hdr = dict(src)
            if "NAXIS1" in hdr and "NAXIS2" in hdr:
                inferred_shape = (int(hdr["NAXIS2"]), int(hdr["NAXIS1"]))
            else:
                inferred_shape = None
        if shape is None:
            if inferred_shape is None:
                raise ValueError(
                    "shape=(H, W) must be provided when header/wcs lacks NAXIS1/2"
                )
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


@dataclass(frozen=True, init=False)
class CubeTarget:
    """Physical IFU cube grid: a spatial WCS and explicit spectral bin edges."""

    spatial_wcs: WCS
    shape: tuple[int, int]
    spectral_bin_edges: Tensor
    spectral_unit: str
    spectral_type: str
    value_semantics: MapValueSemantics

    def __init__(
        self,
        spatial_wcs: WCS | dict[str, Any] | PlannedImagePlaneTarget,
        spectral_bin_edges: Tensor | list[float] | tuple[float, ...],
        *,
        shape: tuple[int, int] | None = None,
        spectral_unit: str = "Angstrom",
        spectral_type: str = "WAVE",
        value_semantics: MapValueSemantics = "surface_brightness",
    ) -> None:
        if isinstance(spatial_wcs, PlannedImagePlaneTarget):
            wcs = spatial_wcs.wcs
            target_shape = spatial_wcs.shape
        else:
            wcs = spatial_wcs if isinstance(spatial_wcs, WCS) else WCS(spatial_wcs)
            inferred = wcs.array_shape
            if shape is None and inferred is None:
                raise ValueError(
                    "CubeTarget shape is required when WCS has no array shape"
                )
            if shape is not None:
                target_shape = (int(shape[0]), int(shape[1]))
            else:
                assert inferred is not None  # rejected above
                target_shape = (int(inferred[0]), int(inferred[1]))
        edges = torch.as_tensor(spectral_bin_edges)
        if edges.ndim != 1 or edges.numel() < 2 or not edges.is_floating_point():
            raise ValueError(
                "CubeTarget spectral_bin_edges must be a floating 1D vector"
            )
        if not bool(torch.isfinite(edges).all()) or not bool(
            torch.all(edges[1:] > edges[:-1])
        ):
            raise ValueError(
                "CubeTarget spectral_bin_edges must be finite and increasing"
            )
        if not isinstance(spectral_unit, str) or not spectral_unit.strip():
            raise ValueError("CubeTarget spectral_unit must be explicit")
        if value_semantics not in {"surface_brightness", "flux"}:
            raise ValueError(
                "CubeTarget value_semantics must be surface_brightness or flux"
            )
        object.__setattr__(self, "spatial_wcs", wcs)
        object.__setattr__(self, "shape", target_shape)
        object.__setattr__(self, "spectral_bin_edges", edges.detach().clone())
        object.__setattr__(self, "spectral_unit", spectral_unit)
        object.__setattr__(self, "spectral_type", spectral_type)
        object.__setattr__(self, "value_semantics", value_semantics)

    @property
    def spatial_target(self) -> PlannedImagePlaneTarget:
        return PlannedImagePlaneTarget(wcs=self.spatial_wcs, shape=self.shape)

    @property
    def header(self) -> dict[str, Any]:
        header = self.spatial_wcs.to_header(relax=True, include_naxis=True)
        header["NAXIS1"] = self.shape[1]
        header["NAXIS2"] = self.shape[0]
        return header

    @property
    def spectral_axis(self) -> SpectralAxis:
        edges = self.spectral_bin_edges
        return SpectralAxis(
            values=0.5 * (edges[:-1] + edges[1:]),
            unit=self.spectral_unit,
            type=self.spectral_type,
        )

    @property
    def band_metadata(self) -> BandMetadataTable:
        centers = self.spectral_axis.values.to(dtype=torch.float64).cpu()
        edges = self.spectral_bin_edges.to(dtype=torch.float64).cpu()
        count = centers.numel()
        return BandMetadataTable(
            band_index=torch.arange(count, dtype=torch.int64),
            center=centers,
            lower=edges[:-1],
            upper=edges[1:],
            observation_count=torch.zeros(count, dtype=torch.int64),
            spectral_weight_sum=torch.zeros(count, dtype=torch.float64),
            band_names=tuple(None for _ in range(count)),
        )

    def to_band_table(self) -> dict[str, object]:
        return self.band_metadata.to_dict()


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
    variance_components: Mapping[str, Tensor] = field(default_factory=dict)
    dq: Tensor | None = None
    usability_mask: Tensor | None = None
    context: Tensor | None = None
    covariance: Any | None = None
    response_plan: Any | None = None
    provenance: Mapping[str, Any] | None = None
    value_semantics: MapValueSemantics = "surface_brightness"

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

    @property
    def quality(self) -> QualityProduct | None:
        if self.dq is None or self.usability_mask is None:
            return None
        return QualityProduct(dq=self.dq, usability_mask=self.usability_mask)

    def query_uncertainty(self, query: Tensor) -> ApertureUncertainty:
        value = _as_tensor(query, device=self.data.device, dtype=self.data.dtype)
        if value.shape != self.data.shape:
            raise ValueError("cube uncertainty query must match the cube shape")
        if self.covariance is None:
            raise ValueError("matrix-free cube covariance is unavailable")
        variance = self.covariance.query_variance(value)
        diagonal = self.covariance.diagonal()
        diagonal_variance = torch.sum(value.square() * diagonal)
        return ApertureUncertainty(
            variance=variance, diagonal_variance=diagonal_variance
        )

    def aperture_uncertainty(
        self,
        spatial_aperture: Tensor,
        spectral_weights: Tensor | None = None,
    ) -> ApertureUncertainty:
        spatial = _as_tensor(
            spatial_aperture,
            device=self.data.device,
            dtype=self.data.dtype,
        )
        if spatial.shape != self.data.shape[-2:]:
            raise ValueError("spatial aperture must match the cube spatial shape")
        spectral = (
            torch.ones(
                self.data.shape[0], device=self.data.device, dtype=self.data.dtype
            )
            if spectral_weights is None
            else _as_tensor(
                spectral_weights,
                device=self.data.device,
                dtype=self.data.dtype,
            )
        )
        if spectral.shape != self.data.shape[:1]:
            raise ValueError("spectral weights must match the cube spectral length")
        return self.query_uncertainty(spectral[:, None, None] * spatial[None])
