"""Product-oriented mapmaking and evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
import math
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ..device import default_compute_device
from ..sphere.moc import MOC
from ..spectral import SpectralAxis
from ..spectrophotometry import Passband
from .observation import (
    BandMetadataTable,
    CubeResult,
    ImageObservation,
    MapResult,
    PlannedImagePlaneCubeTarget,
    PlannedImagePlaneTarget,
)
from .quality import QualityProduct
from .image_plane_coadd import coadd_image_plane, coadd_image_plane_cube
from .planning import plan_image_plane_cube_target, plan_image_plane_target
from .field.types import FieldSpatialSampling
from .operators.config import (
    MapResamplingKernel,
    MapStackingMethod,
    MapValueSemantics,
    validate_map_resampling_kernel,
    validate_map_stacking_method,
    validate_map_value_semantics,
)
from .scenes import ContinuousFieldModel, SceneRepresentation, as_scene_representation
from .sky import CrossExposureSkyCorrector
from .solvers import SolverHooks
from .target import TargetRenderSpec


@dataclass(frozen=True)
class BandpassSpec:
    label: str | None = None
    spectral_min: float | None = None
    spectral_max: float | None = None
    transmission_center: float | None = None
    transmission_sigma: float | None = None
    spectral_unit: str = "Angstrom"
    passband: Passband | None = None

    def __post_init__(self) -> None:
        _coerce_band(self)


@dataclass(frozen=True)
class ProductSpec:
    target_header: dict[str, Any]
    bands: tuple[BandpassSpec, ...]
    value_semantics: MapValueSemantics = "surface_brightness"
    footprint_order: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.target_header, dict):
            raise TypeError("ProductSpec target_header must be a dict")
        if not self.bands:
            raise ValueError("ProductSpec requires at least one band")
        for band in self.bands:
            _coerce_band(band)

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.target_header["NAXIS2"]), int(self.target_header["NAXIS1"])

    @property
    def n_bands(self) -> int:
        return len(self.bands)

    def target_render_spec(
        self,
        *,
        band_index: int = 0,
        psf_spec: object | None = None,
    ) -> TargetRenderSpec:
        return TargetRenderSpec.from_map_product(
            self,
            band_index=band_index,
            psf_spec=psf_spec,
        )

    def target_render_specs(
        self,
        *,
        psf_specs: Sequence[object | None] | None = None,
    ) -> tuple[TargetRenderSpec, ...]:
        if psf_specs is None:
            per_band_psf: Sequence[object | None] = (None,) * len(self.bands)
        else:
            per_band_psf = tuple(psf_specs)
            if len(per_band_psf) != len(self.bands):
                raise ValueError("psf_specs must match ProductSpec bands")
        return tuple(
            self.target_render_spec(
                band_index=band_index, psf_spec=per_band_psf[band_index]
            )
            for band_index in range(len(self.bands))
        )


MapUncertaintyKind = Literal[
    "propagated_linear",
    "empirical_robust",
    "model_predictive",
    "ensemble_predictive",
]


@dataclass(frozen=True)
class UncertaintyProduct:
    """Labeled diagonal uncertainty without pretending it is full covariance."""

    variance: Tensor
    kind: MapUncertaintyKind
    components: Mapping[str, Tensor] = dataclass_field(default_factory=dict)
    correlation_factor: Tensor | None = None

    def __post_init__(self) -> None:
        if self.kind not in {
            "propagated_linear",
            "empirical_robust",
            "model_predictive",
            "ensemble_predictive",
        }:
            raise ValueError(f"unsupported map uncertainty kind {self.kind!r}")
        variance = torch.as_tensor(self.variance)
        if not variance.is_floating_point() or variance.ndim not in {2, 3}:
            raise ValueError("map variance must be a floating 2D or 3D tensor")
        if not bool(torch.isfinite(variance).all()) or bool((variance < 0.0).any()):
            raise ValueError("map variance must be finite and non-negative")
        for name, component in self.components.items():
            if not isinstance(name, str) or not name:
                raise ValueError("map uncertainty component names must be non-empty")
            component_t = torch.as_tensor(component)
            if component_t.shape != variance.shape:
                raise ValueError("map uncertainty component shapes must match variance")
            if not bool(torch.isfinite(component_t).all()) or bool(
                (component_t < 0.0).any()
            ):
                raise ValueError(
                    "map uncertainty components must be finite and non-negative"
                )
        if self.correlation_factor is not None:
            factor = torch.as_tensor(self.correlation_factor)
            if factor.numel() != 1 and factor.shape != variance.shape:
                raise ValueError(
                    "map correlation factor must be scalar or match variance shape"
                )
            if not bool(torch.isfinite(factor).all()) or bool((factor < 0.0).any()):
                raise ValueError(
                    "map correlation factor must be finite and non-negative"
                )
            if factor.numel() == 1:
                if float(factor.item()) <= 0.0:
                    raise ValueError("scalar map correlation factor must be positive")
            elif bool(((variance > 0.0) & (factor <= 0.0)).any()):
                raise ValueError(
                    "map correlation factor must be positive wherever variance is positive"
                )

    @property
    def inverse_variance(self) -> Tensor:
        variance = torch.as_tensor(self.variance)
        return torch.where(
            variance > 0.0,
            variance.reciprocal(),
            torch.zeros_like(variance),
        )


@dataclass
class ProductResult:
    product: ProductSpec
    data: Tensor
    weight: Tensor | None
    header: dict[str, Any]
    overlap_count: Tensor | None = None
    coverage_mocs: tuple[MOC | None, ...] | None = None
    uncertainty: UncertaintyProduct | None = None
    mask: Tensor | None = None
    context: Tensor | None = None
    provenance: Mapping[str, Any] | None = None
    quality: QualityProduct | None = None

    @property
    def dq(self) -> Tensor | None:
        return None if self.quality is None else self.quality.dq

    @property
    def usability_mask(self) -> Tensor | None:
        return None if self.quality is None else self.quality.usability_mask

    @property
    def rejection_log(self) -> tuple[Mapping[str, Any], ...]:
        return () if self.quality is None else self.quality.rejection_log

    @property
    def n_bands(self) -> int:
        return int(self.data.shape[0])

    def target_render_spec(
        self,
        *,
        band_index: int = 0,
        psf_spec: object | None = None,
    ) -> TargetRenderSpec:
        return self.product.target_render_spec(
            band_index=band_index,
            psf_spec=psf_spec,
        )

    def target_render_specs(
        self,
        *,
        psf_specs: Sequence[object | None] | None = None,
    ) -> tuple[TargetRenderSpec, ...]:
        return self.product.target_render_specs(psf_specs=psf_specs)


MapUncertaintyMode = Literal["native_inverse_variance", "support_density_proxy"]


def _coerce_band(value: BandpassSpec) -> BandpassSpec:
    if not isinstance(value.spectral_unit, str) or not value.spectral_unit.strip():
        raise ValueError("BandpassSpec spectral_unit must be explicit")
    if value.passband is not None:
        if value.spectral_unit.strip().lower() not in {"angstrom", "angstroms", "aa"}:
            raise ValueError("Passband-backed BandpassSpec uses Angstrom coordinates")
        if (
            value.transmission_center is not None
            or value.transmission_sigma is not None
        ):
            raise ValueError(
                "BandpassSpec passband and Gaussian transmission are mutually exclusive"
            )
    if value.spectral_min is not None and value.spectral_max is None:
        raise ValueError("BandpassSpec requires spectral_max when spectral_min is set")
    if value.spectral_max is not None and value.spectral_min is None:
        raise ValueError("BandpassSpec requires spectral_min when spectral_max is set")
    if value.spectral_min is not None and value.spectral_max is not None:
        lower = float(value.spectral_min)
        upper = float(value.spectral_max)
        if not math.isfinite(lower) or not math.isfinite(upper) or upper < lower:
            raise ValueError("BandpassSpec spectral bounds must be finite and ordered")
    if value.transmission_center is not None and not math.isfinite(
        float(value.transmission_center)
    ):
        raise ValueError("BandpassSpec transmission_center must be finite")
    if value.transmission_sigma is not None:
        sigma = float(value.transmission_sigma)
        if not math.isfinite(sigma) or sigma <= 0.0:
            raise ValueError(
                "BandpassSpec transmission_sigma must be finite and positive"
            )
    return value


def _band_center(band: BandpassSpec) -> float:
    if band.passband is not None:
        return float(band.passband.effective_wavelength.item())
    if band.transmission_center is not None:
        return float(band.transmission_center)
    if band.spectral_min is not None and band.spectral_max is not None:
        return 0.5 * (float(band.spectral_min) + float(band.spectral_max))
    raise ValueError(
        "BandpassSpec requires transmission_center or spectral bounds for product rendering"
    )


def _bands_from_cube_target(
    target: PlannedImagePlaneCubeTarget,
) -> tuple[BandpassSpec, ...]:
    bands: list[BandpassSpec] = []
    metadata = target.band_metadata
    for idx in range(int(metadata.center.numel())):
        lower = float(metadata.lower[idx].item())
        upper = float(metadata.upper[idx].item())
        center = float(metadata.center[idx].item())
        is_interval = not math.isclose(lower, upper, rel_tol=0.0, abs_tol=1.0e-12)
        bands.append(
            BandpassSpec(
                label=metadata.band_names[idx],
                spectral_min=lower if is_interval else None,
                spectral_max=upper if is_interval else None,
                transmission_center=center,
                transmission_sigma=None,
            )
        )
    return tuple(bands)


def _observations_have_spectral_support(
    observations: Sequence[ImageObservation],
) -> bool:
    for observation in observations:
        if (
            observation.wavelength is not None
            or observation.spectral_min is not None
            or observation.spectral_max is not None
            or observation.transmission_center is not None
            or observation.transmission_sigma is not None
        ):
            return True
    return False


def plan_map_product(
    observations: Sequence[ImageObservation],
    *,
    reference_index: int = 0,
    header_overrides: dict[str, Any] | None = None,
    bands: Sequence[BandpassSpec] | None = None,
    spectral_axis: SpectralAxis | None = None,
    spectral_bin_edges: Tensor | None = None,
    spectral_unit: str = "Angstrom",
    spectral_type: str = "WAVE",
    value_semantics: MapValueSemantics = "surface_brightness",
    footprint_order: int | None = None,
    union_footprint: bool = False,
    footprint_padding: float = 1.0,
    footprint_tolerance_px: float = 0.25,
    footprint_max_depth: int = 20,
    max_footprint_pixels: int | None = None,
    device: torch.device | str | None = None,
) -> ProductSpec:
    """Plan a map product from a sequence of image-plane observations.

    When ``union_footprint=True``, the target WCS is computed automatically from
    the union bounding box of all input footprints (like SWarp's automatic
    output grid).  Set ``--reference`` to a FITS file path to override the
    spatial target completely.
    """
    if footprint_order is not None and int(footprint_order) < 0:
        raise ValueError("footprint_order must be non-negative")
    value_semantics = validate_map_value_semantics(value_semantics)
    if bands is not None and (
        spectral_axis is not None or spectral_bin_edges is not None
    ):
        raise ValueError(
            "bands is mutually exclusive with spectral_axis and spectral_bin_edges"
        )
    if bands is not None:
        spatial = plan_image_plane_target(
            observations,
            reference_index=reference_index,
            header_overrides=header_overrides,
            device=device,
            union_footprint=union_footprint,
            footprint_padding=footprint_padding,
            footprint_tolerance_px=footprint_tolerance_px,
            footprint_max_depth=footprint_max_depth,
            max_footprint_pixels=max_footprint_pixels,
        )
        validated_bands = tuple(_coerce_band(band) for band in bands)
        if not validated_bands:
            raise ValueError("ProductSpec requires at least one band")
        return ProductSpec(
            target_header=spatial.header,
            bands=validated_bands,
            value_semantics=value_semantics,
            footprint_order=footprint_order,
        )
    if (
        spectral_axis is not None
        or spectral_bin_edges is not None
        or _observations_have_spectral_support(observations)
    ):
        target = plan_image_plane_cube_target(
            observations,
            reference_index=reference_index,
            header_overrides=header_overrides,
            spectral_axis=spectral_axis,
            spectral_bin_edges=spectral_bin_edges,
            spectral_unit=spectral_unit,
            spectral_type=spectral_type,
            device=device,
            union_footprint=union_footprint,
            footprint_padding=footprint_padding,
            footprint_tolerance_px=footprint_tolerance_px,
            footprint_max_depth=footprint_max_depth,
            max_footprint_pixels=max_footprint_pixels,
        )
        return ProductSpec(
            target_header=target.header,
            bands=_bands_from_cube_target(target),
            value_semantics=value_semantics,
            footprint_order=footprint_order,
        )
    spatial = plan_image_plane_target(
        observations,
        reference_index=reference_index,
        header_overrides=header_overrides,
        device=device,
        union_footprint=union_footprint,
        footprint_padding=footprint_padding,
        footprint_tolerance_px=footprint_tolerance_px,
        footprint_max_depth=footprint_max_depth,
        max_footprint_pixels=max_footprint_pixels,
    )
    reference = observations[reference_index]
    return ProductSpec(
        target_header=spatial.header,
        bands=(
            BandpassSpec(
                label=reference.band_name,
                spectral_min=reference.spectral_min,
                spectral_max=reference.spectral_max,
                transmission_center=reference.transmission_center
                if reference.transmission_center is not None
                else reference.wavelength,
                transmission_sigma=reference.transmission_sigma,
            ),
        ),
        value_semantics=value_semantics,
        footprint_order=footprint_order,
    )


def _band_metadata_from_product(
    product: ProductSpec,
    spectral_axis: SpectralAxis,
    spectral_bin_edges: Tensor | None,
) -> BandMetadataTable:
    n_band = int(spectral_axis.values.numel())
    centers = spectral_axis.values.to(dtype=torch.float64)
    if spectral_bin_edges is None:
        lower = centers.clone()
        upper = centers.clone()
    else:
        lower = spectral_bin_edges[:-1].clone()
        upper = spectral_bin_edges[1:].clone()
    return BandMetadataTable(
        band_index=torch.arange(n_band, dtype=torch.int64),
        center=centers,
        lower=lower,
        upper=upper,
        observation_count=torch.zeros(n_band, dtype=torch.int64),
        spectral_weight_sum=torch.zeros(n_band, dtype=torch.float64),
        band_names=tuple(band.label for band in product.bands),
    )


def _product_to_cube_target(product: ProductSpec) -> PlannedImagePlaneCubeTarget:
    centers = torch.tensor(
        [_band_center(band) for band in product.bands],
        dtype=torch.float64,
    )
    if centers.numel() > 1 and not bool(torch.all(centers[1:] > centers[:-1])):
        raise ValueError(
            "ProductSpec bands must be strictly increasing in spectral center"
        )
    interval_bands = [
        band
        for band in product.bands
        if band.spectral_min is not None and band.spectral_max is not None
    ]
    spectral_bin_edges = None
    if interval_bands:
        if len(interval_bands) != len(product.bands):
            raise ValueError(
                "ProductSpec direct stacking requires all bands to define intervals or none"
            )
        lowers = [
            float(band.spectral_min)
            for band in product.bands
            if band.spectral_min is not None
        ]
        uppers = [
            float(band.spectral_max)
            for band in product.bands
            if band.spectral_max is not None
        ]
        contiguous = True
        for idx in range(len(lowers) - 1):
            contiguous = contiguous and math.isclose(
                uppers[idx],
                lowers[idx + 1],
                rel_tol=0.0,
                abs_tol=1.0e-8,
            )
        if len(product.bands) == 1 or contiguous:
            spectral_bin_edges = torch.tensor(
                [lowers[0], *uppers],
                dtype=torch.float64,
            )
        else:
            raise ValueError(
                "ProductSpec direct stacking currently supports only point bands or "
                "ordered contiguous interval bands"
            )
    spectral_axis = SpectralAxis(values=centers, unit="Angstrom", type="WAVE")
    spatial = PlannedImagePlaneTarget(
        header=dict(product.target_header),
        shape=(
            int(product.target_header["NAXIS2"]),
            int(product.target_header["NAXIS1"]),
        ),
    )
    return PlannedImagePlaneCubeTarget(
        spatial_target=spatial,
        spectral_axis=spectral_axis,
        spectral_bin_edges=spectral_bin_edges,
        band_metadata=_band_metadata_from_product(
            product,
            spectral_axis=spectral_axis,
            spectral_bin_edges=spectral_bin_edges,
        ),
    )


def _wrap_map_result(result: MapResult, product: ProductSpec) -> ProductResult:
    coverage_mocs = None
    if result.coverage_moc is not None:
        coverage_mocs = (result.coverage_moc,)
    uncertainty = None
    if result.inverse_variance is not None:
        inverse_variance = result.inverse_variance.unsqueeze(0)
        variance = torch.where(
            inverse_variance > 0.0,
            inverse_variance.reciprocal(),
            torch.zeros_like(inverse_variance),
        )
        uncertainty = UncertaintyProduct(
            variance=variance,
            kind="propagated_linear",
            components={"detector": variance},
            correlation_factor=(
                result.correlation_factor.unsqueeze(0)
                if result.correlation_factor is not None
                else None
            ),
        )
    return ProductResult(
        product=product,
        data=result.data.unsqueeze(0),
        weight=result.weight.unsqueeze(0),
        header=result.header,
        overlap_count=(
            result.overlap_count.unsqueeze(0)
            if result.overlap_count is not None
            else None
        ),
        coverage_mocs=coverage_mocs,
        uncertainty=uncertainty,
        mask=(
            result.usability_mask.unsqueeze(0)
            if result.usability_mask is not None
            else (result.weight > 0.0).unsqueeze(0)
        ),
        context=(result.context.unsqueeze(0) if result.context is not None else None),
        provenance=result.provenance,
        quality=(
            QualityProduct(
                dq=result.dq.unsqueeze(0),
                usability_mask=result.usability_mask.unsqueeze(0),
                rejection_log=result.rejection_log,
                policy=result.dq_policy,
            )
            if result.dq is not None and result.usability_mask is not None
            else None
        ),
    )


def _wrap_cube_result(
    result: CubeResult,
    product: ProductSpec,
) -> ProductResult:
    uncertainty = None
    if result.inverse_variance is not None:
        variance = torch.where(
            result.inverse_variance > 0.0,
            result.inverse_variance.reciprocal(),
            torch.zeros_like(result.inverse_variance),
        )
        uncertainty = UncertaintyProduct(
            variance=variance,
            kind="propagated_linear",
            components={"detector": variance},
            correlation_factor=result.correlation_factor,
        )
    return ProductResult(
        product=product,
        data=result.data,
        weight=result.weight,
        header=result.header,
        overlap_count=result.overlap_count,
        coverage_mocs=result.coverage_mocs,
        uncertainty=uncertainty,
        mask=result.weight > 0.0,
    )


def stack_product(
    observations: Sequence[ImageObservation],
    product: ProductSpec,
    *,
    resampling_kernel: MapResamplingKernel = "bilinear",
    method: str = "coadd",
    stacking_method: MapStackingMethod = "weighted_mean",
    drizzle_pixfrac: float = 1.0,
    clip_sigma: float = 4.0,
    clip_iterations: int = 3,
    spatial_tile_hw: tuple[int, int] | None = None,
    device: torch.device | str | None = None,
    scratch_dir: str | Path | None = None,
    background_estimator: Callable[[Tensor, Tensor | None], Tensor] | None = None,
    cross_exposure_sky_corrector: CrossExposureSkyCorrector | None = None,
) -> ProductResult:
    """Resample observations and stack them into a requested map product."""
    if method != "coadd":
        raise ValueError("stack_product currently supports only method='coadd'")
    resampling_kernel = validate_map_resampling_kernel(resampling_kernel)
    stacking_method = validate_map_stacking_method(stacking_method)
    if len(product.bands) == 1:
        band = product.bands[0]
        if band.spectral_min is None and band.spectral_max is None:
            result = coadd_image_plane(
                observations,
                target_header=product.target_header,
                footprint_order=product.footprint_order,
                resampling_kernel=resampling_kernel,
                value_semantics=product.value_semantics,
                stacking_method=stacking_method,
                drizzle_pixfrac=drizzle_pixfrac,
                clip_sigma=clip_sigma,
                clip_iterations=clip_iterations,
                spatial_tile_hw=spatial_tile_hw,
                device=device,
                scratch_dir=scratch_dir,
                background_estimator=background_estimator,
                cross_exposure_sky_corrector=cross_exposure_sky_corrector,
            )
            return _wrap_map_result(result, product)
    cube_target = _product_to_cube_target(product)
    result = coadd_image_plane_cube(
        observations,
        target=cube_target,
        footprint_order=product.footprint_order,
        resampling_kernel=resampling_kernel,
        value_semantics=product.value_semantics,
        stacking_method=stacking_method,
        drizzle_pixfrac=drizzle_pixfrac,
        device=device,
        background_estimator=background_estimator,
        cross_exposure_sky_corrector=cross_exposure_sky_corrector,
    )
    return _wrap_cube_result(result, product)


def _coerce_product_plane(
    rendered: Tensor,
    *,
    band_index: int,
    n_bands: int,
    shape: tuple[int, int],
) -> Tensor:
    if rendered.ndim == 2:
        if tuple(rendered.shape) != shape:
            raise ValueError(
                "scene render plane shape must match ProductSpec target shape"
            )
        return rendered
    if rendered.ndim == 3:
        if int(rendered.shape[0]) != n_bands:
            raise ValueError("scene render cube must match ProductSpec band count")
        plane = rendered[band_index]
        if tuple(plane.shape) != shape:
            raise ValueError(
                "scene render cube plane shape must match ProductSpec target shape"
            )
        return plane
    raise ValueError("scene render result must be 2D or 3D")


def render_scene_product(
    scene: SceneRepresentation | nn.Module | Tensor,
    product: ProductSpec,
    *,
    solver_hooks: SolverHooks | None = None,
    num_spectral_samples: int = 9,
) -> ProductResult:
    scene_representation = as_scene_representation(scene)
    shape = (
        int(product.target_header["NAXIS2"]),
        int(product.target_header["NAXIS1"]),
    )
    planes: list[Tensor] = []
    variance_planes: list[Tensor] = []
    uncertainty_kind: MapUncertaintyKind | None = None
    target_specs = product.target_render_specs()
    for band_index, target_spec in enumerate(target_specs):
        if solver_hooks is not None:
            rendered = solver_hooks.render_scene(scene_representation, target_spec)
            plane = _coerce_product_plane(
                rendered,
                band_index=band_index,
                n_bands=product.n_bands,
                shape=shape,
            ).to(dtype=torch.float32)
            if solver_hooks.apply_noise is None:
                variance = None
            else:
                weighted = solver_hooks.apply_noise_model(plane)
                inverse_variance = torch.where(
                    weighted > 0.0,
                    weighted.to(dtype=torch.float32),
                    torch.zeros_like(plane, dtype=torch.float32),
                )
                variance = torch.where(
                    inverse_variance > 0.0,
                    inverse_variance.reciprocal(),
                    torch.zeros_like(inverse_variance),
                )
                uncertainty_kind = "model_predictive"
        elif isinstance(scene_representation, ContinuousFieldModel):
            from .field.rendering import field_model_has_predictive_distribution

            has_distribution = field_model_has_predictive_distribution(
                scene_representation.model
            )
            if has_distribution:
                rendered, logvar = scene_representation.render_distribution_to_target(
                    target_spec,
                    num_spectral_samples=num_spectral_samples,
                )
                variance = torch.exp(
                    _coerce_product_plane(
                        logvar,
                        band_index=band_index,
                        n_bands=product.n_bands,
                        shape=shape,
                    )
                ).to(dtype=torch.float32)
                uncertainty_kind = "model_predictive"
            else:
                rendered = scene_representation.render_to_target(
                    target_spec,
                    num_spectral_samples=num_spectral_samples,
                )
                variance = None
            plane = _coerce_product_plane(
                rendered,
                band_index=band_index,
                n_bands=product.n_bands,
                shape=shape,
            ).to(dtype=torch.float32)
        else:
            rendered = scene_representation.render_to_target(target_spec)
            plane = _coerce_product_plane(
                rendered,
                band_index=band_index,
                n_bands=product.n_bands,
                shape=shape,
            ).to(dtype=torch.float32)
            variance = None
        planes.append(plane)
        if variance is not None:
            variance_planes.append(variance)
    data = torch.stack(planes, dim=0)
    uncertainty = None
    if len(variance_planes) == len(planes):
        total_variance = torch.stack(variance_planes, dim=0)
        uncertainty = UncertaintyProduct(
            variance=total_variance,
            kind=uncertainty_kind or "model_predictive",
            components={"detector": total_variance},
        )
    return ProductResult(
        product=product,
        data=data,
        weight=None,
        header=dict(product.target_header),
        overlap_count=torch.ones_like(data, dtype=torch.int32),
        coverage_mocs=None,
        uncertainty=uncertainty,
        mask=torch.ones_like(data, dtype=torch.bool),
    )


def render_field_product(
    model: nn.Module,
    product: ProductSpec,
    *,
    num_spectral_samples: int = 9,
    spatial_sampling: FieldSpatialSampling = "center",
    spatial_tile_hw: tuple[int, int] | None = None,
) -> ProductResult:
    if spatial_tile_hw is None and spatial_sampling == "center":
        return render_scene_product(
            model,
            product,
            num_spectral_samples=num_spectral_samples,
        )
    if spatial_tile_hw is None:
        spatial_tile_hw = product.shape
    # Tiled rendering: render each spatial tile separately to cap memory.
    from .field.rendering import (
        field_model_has_predictive_distribution,
        render_field_observation,
        render_field_observation_distribution,
    )
    from .field.types import FieldObservation

    h, w = product.shape
    th, tw = int(spatial_tile_hw[0]), int(spatial_tile_hw[1])
    n_bands = product.n_bands
    model_device = next(
        (parameter.device for parameter in model.parameters()),
        default_compute_device(),
    )
    data = torch.zeros((n_bands, h, w), dtype=torch.float32, device=model_device)
    has_distribution = field_model_has_predictive_distribution(model)
    detector_variance = torch.zeros_like(data) if has_distribution else None
    for band_index in range(n_bands):
        band = product.bands[band_index]
        for y0 in range(0, h, th):
            for x0 in range(0, w, tw):
                y1, x1 = min(y0 + th, h), min(x0 + tw, w)
                tile_hdr = dict(product.target_header)
                tile_hdr["NAXIS1"] = x1 - x0
                tile_hdr["NAXIS2"] = y1 - y0
                tile_hdr["CRPIX1"] = float(
                    product.target_header.get("CRPIX1", 1.0)
                ) - float(x0)
                tile_hdr["CRPIX2"] = float(
                    product.target_header.get("CRPIX2", 1.0)
                ) - float(y0)
                tile_obs = FieldObservation(
                    image=torch.zeros(
                        (y1 - y0, x1 - x0),
                        dtype=torch.float32,
                        device=model_device,
                    ),
                    header=tile_hdr,
                    weight=torch.ones(
                        (y1 - y0, x1 - x0),
                        dtype=torch.float32,
                        device=model_device,
                    ),
                    wavelength=band.transmission_center
                    if band.transmission_center is not None
                    else (
                        0.5 * (float(band.spectral_min) + float(band.spectral_max))
                        if band.spectral_min is not None
                        and band.spectral_max is not None
                        else None
                    ),
                    spectral_min=band.spectral_min,
                    spectral_max=band.spectral_max,
                    transmission_center=band.transmission_center,
                    transmission_sigma=band.transmission_sigma,
                    band_name=band.label,
                    passband=band.passband,
                    value_semantics=product.value_semantics,
                    spectral_unit=band.spectral_unit,
                )
                model.eval()
                with torch.no_grad():
                    if detector_variance is None:
                        tile_pred = render_field_observation(
                            model,
                            tile_obs,
                            num_spectral_samples=num_spectral_samples,
                            spatial_sampling=spatial_sampling,
                        )
                    else:
                        tile_pred, tile_logvar = render_field_observation_distribution(
                            model,
                            tile_obs,
                            num_spectral_samples=num_spectral_samples,
                            spatial_sampling=spatial_sampling,
                        )
                data[band_index, y0:y1, x0:x1] = tile_pred
                if detector_variance is not None:
                    detector_variance[band_index, y0:y1, x0:x1] = torch.exp(tile_logvar)
    uncertainty = (
        None
        if detector_variance is None
        else UncertaintyProduct(
            variance=detector_variance,
            kind="model_predictive",
            components={"detector": detector_variance},
        )
    )
    return ProductResult(
        product=product,
        data=data,
        weight=None,
        header=dict(product.target_header),
        overlap_count=torch.ones_like(data, dtype=torch.int32),
        coverage_mocs=None,
        uncertainty=uncertainty,
        mask=torch.ones_like(data, dtype=torch.bool),
    )


def _validated_seed_tuple(
    seeds: Sequence[int],
    *,
    expected: int,
    label: str,
) -> tuple[int, ...]:
    values = tuple(int(seed) for seed in seeds)
    if len(values) != expected:
        raise ValueError(f"{label} must match the number of draws")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must be unique")
    return values


def _streaming_sample_variance(
    samples: Sequence[Tensor],
    *,
    reference: Tensor,
    label: str,
) -> Tensor:
    if len(samples) < 2:
        raise ValueError(f"{label} requires at least two draws")
    mean = torch.zeros_like(reference)
    m2 = torch.zeros_like(reference)
    for count, raw_sample in enumerate(samples, start=1):
        sample = torch.as_tensor(
            raw_sample, device=reference.device, dtype=reference.dtype
        )
        if sample.shape != reference.shape:
            raise ValueError(f"{label} draw shape must match the rendered product")
        if not bool(torch.isfinite(sample).all()):
            raise ValueError(f"{label} draws must be finite")
        delta = sample - mean
        mean = mean + delta / count
        m2 = m2 + delta * (sample - mean)
    return m2 / (len(samples) - 1)


@torch.no_grad()
def render_field_ensemble_product(
    models: Sequence[nn.Module],
    product: ProductSpec,
    *,
    member_seeds: Sequence[int],
    calibration_draws: Sequence[Tensor] | None = None,
    calibration_seeds: Sequence[int] | None = None,
    num_spectral_samples: int = 9,
    spatial_sampling: FieldSpatialSampling = "center",
    spatial_tile_hw: tuple[int, int] | None = None,
) -> ProductResult:
    """Render a seeded ensemble with explicit uncertainty components.

    Between-member sample variance is epistemic. When every member supplies an
    observation-likelihood variance, the mean of those variances is the detector
    component. Optional independently generated calibration draws contribute a
    separate sample-variance component. Rendering is sequential so memory does
    not grow with ensemble size.
    """
    members = tuple(models)
    if len(members) < 2:
        raise ValueError("field ensemble requires at least two models")
    resolved_member_seeds = _validated_seed_tuple(
        member_seeds,
        expected=len(members),
        label="member_seeds",
    )
    mean: Tensor | None = None
    m2: Tensor | None = None
    detector_sum: Tensor | None = None
    has_detector: bool | None = None
    reference_result: ProductResult | None = None
    for count, model in enumerate(members, start=1):
        result = render_field_product(
            model,
            product,
            num_spectral_samples=num_spectral_samples,
            spatial_sampling=spatial_sampling,
            spatial_tile_hw=spatial_tile_hw,
        )
        sample = result.data
        if not bool(torch.isfinite(sample).all()):
            raise ValueError("ensemble member products must be finite")
        member_has_detector = result.uncertainty is not None
        if has_detector is None:
            has_detector = member_has_detector
            reference_result = result
            mean = torch.zeros_like(sample)
            m2 = torch.zeros_like(sample)
            if member_has_detector:
                detector_sum = torch.zeros_like(sample)
        elif member_has_detector != has_detector:
            raise ValueError(
                "ensemble members must all provide detector uncertainty or all omit it"
            )
        assert mean is not None and m2 is not None
        if sample.shape != mean.shape or sample.device != mean.device:
            raise ValueError("ensemble member products must share shape and device")
        delta = sample - mean
        mean = mean + delta / count
        m2 = m2 + delta * (sample - mean)
        if member_has_detector:
            assert result.uncertainty is not None and detector_sum is not None
            detector_sum = detector_sum + result.uncertainty.variance.to(
                device=sample.device,
                dtype=sample.dtype,
            )
    assert mean is not None and m2 is not None and reference_result is not None
    epistemic = m2 / (len(members) - 1)
    components: dict[str, Tensor] = {"epistemic": epistemic}
    total_variance = epistemic
    if detector_sum is not None:
        detector = detector_sum / len(members)
        components["detector"] = detector
        total_variance = total_variance + detector
    resolved_calibration_seeds: tuple[int, ...] | None = None
    if calibration_draws is not None:
        draws = tuple(calibration_draws)
        if calibration_seeds is None:
            raise ValueError("calibration_seeds are required with calibration_draws")
        resolved_calibration_seeds = _validated_seed_tuple(
            calibration_seeds,
            expected=len(draws),
            label="calibration_seeds",
        )
        calibration = _streaming_sample_variance(
            draws,
            reference=mean,
            label="calibration uncertainty",
        )
        components["calibration"] = calibration
        total_variance = total_variance + calibration
    elif calibration_seeds is not None:
        raise ValueError("calibration_draws are required with calibration_seeds")
    provenance = dict(reference_result.provenance or {})
    provenance["uncertainty"] = {
        "ensemble_members": len(members),
        "member_seeds": list(resolved_member_seeds),
        "calibration_draws": 0
        if resolved_calibration_seeds is None
        else len(resolved_calibration_seeds),
        "calibration_seeds": None
        if resolved_calibration_seeds is None
        else list(resolved_calibration_seeds),
        "variance_estimator": "unbiased_sample",
    }
    return ProductResult(
        product=product,
        data=mean,
        weight=None,
        header=dict(product.target_header),
        overlap_count=reference_result.overlap_count,
        coverage_mocs=reference_result.coverage_mocs,
        uncertainty=UncertaintyProduct(
            variance=total_variance,
            kind="ensemble_predictive",
            components=components,
        ),
        mask=reference_result.mask,
        context=None,
        provenance=provenance,
    )


def _as_cube(tensor: Tensor) -> Tensor:
    if tensor.ndim == 2:
        return tensor.unsqueeze(0)
    if tensor.ndim == 3:
        return tensor
    raise ValueError("Map product evaluation expects 2D or 3D tensors")


def _mean_pool_cube(cube: Tensor, *, kernel_size: int) -> Tensor:
    if kernel_size <= 1:
        return cube
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")
    pooled = F.avg_pool2d(
        cube.unsqueeze(0),
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )
    return pooled.squeeze(0)


def estimate_map_inverse_variance(
    result: ProductResult | Tensor,
    *,
    mode: MapUncertaintyMode = "native_inverse_variance",
    support_weight: Tensor | None = None,
    kernel_size: int = 9,
    min_inverse_variance: float = 1.0e-6,
) -> Tensor:
    if min_inverse_variance <= 0.0:
        raise ValueError("min_inverse_variance must be positive")
    if isinstance(result, ProductResult):
        if result.weight is not None:
            weight = _as_cube(result.weight).to(dtype=torch.float32)
        elif mode == "native_inverse_variance" and result.uncertainty is not None:
            weight = _as_cube(result.uncertainty.inverse_variance).to(
                dtype=torch.float32
            )
        elif mode == "support_density_proxy" and support_weight is not None:
            weight = _as_cube(support_weight).to(dtype=torch.float32)
        else:
            raise ValueError(
                "map inverse variance is unavailable; supply an uncertainty "
                "product or opt into support_density_proxy with support_weight"
            )
    else:
        weight = _as_cube(result).to(dtype=torch.float32)
    if mode == "native_inverse_variance":
        return torch.clamp(weight, min=float(min_inverse_variance))
    if mode != "support_density_proxy":
        raise ValueError(f"unsupported uncertainty mode {mode!r}")
    reference = _as_cube(support_weight if support_weight is not None else weight).to(
        dtype=torch.float32,
        device=weight.device,
    )
    if reference.shape != weight.shape:
        raise ValueError("support_weight must match the product weight shape")
    support_mask = reference > 0.0
    local_support = _mean_pool_cube(
        support_mask.to(dtype=torch.float32),
        kernel_size=kernel_size,
    )
    positive = support_mask.to(dtype=torch.bool)
    if bool(positive.any()):
        ref_max = torch.max(reference[positive])
    else:
        ref_max = torch.tensor(1.0, dtype=reference.dtype, device=reference.device)
    normalized_reference = reference / torch.clamp(ref_max, min=1.0e-8)
    local_reference = _mean_pool_cube(
        normalized_reference,
        kernel_size=kernel_size,
    )
    inverse_variance = local_support * (0.25 + 0.75 * local_reference)
    return torch.clamp(inverse_variance, min=float(min_inverse_variance))
