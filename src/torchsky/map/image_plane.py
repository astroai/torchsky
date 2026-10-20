"""Image-plane mapmaking: projection kernels, coadds, and public entry points.

The implementation is split into ``image_plane_common`` (shared helpers),
``image_plane_projection`` (resampling/projection kernels), and
``image_plane_coadd`` (coadds, resampling wrappers, aperture uncertainty).
Historical imports from :mod:`torchsky.map.image_plane` are unchanged.
"""

from __future__ import annotations

from .image_plane_common import (
    load_observation_image_and_header,
    _resolve_map_device,
    _ProjectedImagePlaneObservation,
    _is_splat_kernel,
    _convert_world_between_wcs,
    _splat_for_interpolation,
    _validated_map_resampling_kernel,
    _validated_map_value_semantics,
    _validated_drizzle_pixfrac,
    _validated_map_stacking_method,
    _true_image_shape,
    _observation_header_for_footprint,
    _coerce_target_header,
)

from .image_plane_projection import (
    _requires_roundtrip_validity_check,
    _coarse_grid_shape,
    _interpolate_forward_mapping_from_coarse,
    _requires_target_pixel_quadrature,
    _tpv_affine_envelope,
    _tpv_source_cap_radius_deg,
    _angular_separation_deg,
    _source_pixel_diameter_arcsec,
    _roundtrip_validity_mask,
    _resample_weighted_to_target,
    _drizzle_separable_1d_weight,
    _forward_project_weighted_to_target,
    _project_observation_to_target_chunk,
    _project_observation_to_target,
)

from .image_plane_coadd import (
    _coadd_image_plane_spatial_tiles,
    coadd_image_plane,
    evaluate_image_plane_aperture_uncertainty,
    ImagePlaneMapMaker,
    coadd_image_plane_cube,
    coadd_image_plane_multiband,
    _propagate_observation_variance_components,
    resample,
    blot,
    _blot_spatial_derivative,
    _apply_observation_astrometric_shifts,
    coadd,
    coadd_multiband,
    aperture_uncertainty,
)

# Sky matching and target planning re-export the historical image_plane names.
from .planning import (
    _obs_target_pixel_bbox,
    _plan_union_footprint_target,
    plan,
    plan_cube,
    plan_image_plane_cube_target,
    plan_image_plane_cube_target_from_header,
    plan_image_plane_target,
)
from .sky_matching import _resolve_tile_hw, detect_outliers, match_sky


# Historical re-exports: these names were module-level globals in the old
# monolithic module and are still imported from torchsky.map.image_plane.
from torchfits import read
from ..wcs import WCS
from .calibration import (
    _apply_observation_calibration,
    _effective_observation_photometric_scale,
    _load_observation_weight_map,
    _should_apply_surface_brightness_area_scaling,
)
from .image_plane_coadd import _FINITE_APERTURE_COVARIANCE_KERNELS
from .observation import ImageObservation
from .operators.execution import prepare_image_observation_data
from .windowing import _build_target_world_grid, _prepare_target

__all__ = [
    "load_observation_image_and_header",
    "_resolve_map_device",
    "_ProjectedImagePlaneObservation",
    "_is_splat_kernel",
    "_convert_world_between_wcs",
    "_splat_for_interpolation",
    "_validated_map_resampling_kernel",
    "_validated_map_value_semantics",
    "_validated_drizzle_pixfrac",
    "_validated_map_stacking_method",
    "_true_image_shape",
    "_observation_header_for_footprint",
    "_coerce_target_header",
    "_requires_roundtrip_validity_check",
    "_coarse_grid_shape",
    "_interpolate_forward_mapping_from_coarse",
    "_requires_target_pixel_quadrature",
    "_tpv_affine_envelope",
    "_tpv_source_cap_radius_deg",
    "_angular_separation_deg",
    "_source_pixel_diameter_arcsec",
    "_roundtrip_validity_mask",
    "_resample_weighted_to_target",
    "_drizzle_separable_1d_weight",
    "_forward_project_weighted_to_target",
    "_project_observation_to_target_chunk",
    "_project_observation_to_target",
    "_coadd_image_plane_spatial_tiles",
    "coadd_image_plane",
    "evaluate_image_plane_aperture_uncertainty",
    "ImagePlaneMapMaker",
    "coadd_image_plane_cube",
    "coadd_image_plane_multiband",
    "_propagate_observation_variance_components",
    "resample",
    "blot",
    "_blot_spatial_derivative",
    "_apply_observation_astrometric_shifts",
    "coadd",
    "coadd_multiband",
    "aperture_uncertainty",
    "_obs_target_pixel_bbox",
    "_plan_union_footprint_target",
    "plan",
    "plan_cube",
    "plan_image_plane_cube_target",
    "plan_image_plane_cube_target_from_header",
    "plan_image_plane_target",
    "_resolve_tile_hw",
    "detect_outliers",
    "match_sky",
    "read",
    "WCS",
    "_apply_observation_calibration",
    "_effective_observation_photometric_scale",
    "_load_observation_weight_map",
    "_should_apply_surface_brightness_area_scaling",
    "_FINITE_APERTURE_COVARIANCE_KERNELS",
    "ImageObservation",
    "prepare_image_observation_data",
    "_build_target_world_grid",
    "_prepare_target",
]
