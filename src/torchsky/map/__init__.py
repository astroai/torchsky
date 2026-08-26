"""Lean mapmaking facade.

Implementation modules remain available under ``torchsky.map.*``. The flat
``torchsky.map`` namespace is intentionally limited to promoted coadd/product
entry points and result/specification objects.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

_EXPORTS: dict[str, tuple[str, str]] = {
    # Public API Facade (11 entries)
    "coadd_image_plane": ("torchsky.map.image_plane", "coadd_image_plane"),
    "coadd_image_plane_cube": ("torchsky.map.image_plane", "coadd_image_plane_cube"),
    "coadd_image_plane_multiband": (
        "torchsky.map.image_plane",
        "coadd_image_plane_multiband",
    ),
    "stack_map_product": ("torchsky.map.product", "stack_map_product"),
    "MapProductSpec": ("torchsky.map.product", "MapProductSpec"),
    "TargetRenderSpec": ("torchsky.map.target", "TargetRenderSpec"),
    "MapBandpassSpec": ("torchsky.map.product", "MapBandpassSpec"),
    "MapProductResult": ("torchsky.map.product", "MapProductResult"),
    "MapResult": ("torchsky.map.image_plane", "MapResult"),
    "MultiBandMapResult": ("torchsky.map.image_plane", "MultiBandMapResult"),
    "SpectralCubeMapResult": ("torchsky.map.image_plane", "SpectralCubeMapResult"),
    "splat_bilinear": ("torchsky.map.splat", "splat_bilinear"),
    "splat_to_native": ("torchsky.map.splat", "splat_to_native"),
    "SplatDiagnostics": ("torchsky.map.splat", "SplatDiagnostics"),
    # Compatibility / Internal / Phase 1 entries resolved lazily
    "BandMetadataTable": ("torchsky.map.image_plane", "BandMetadataTable"),
    "ImageObservation": ("torchsky.map.image_plane", "ImageObservation"),
    "ImagePlaneMapMaker": ("torchsky.map.image_plane", "ImagePlaneMapMaker"),
    "MapResamplingKernel": ("torchsky.map.image_plane", "MapResamplingKernel"),
    "MapStackingMethod": ("torchsky.map.image_plane", "MapStackingMethod"),
    "MapValueSemantics": ("torchsky.map.image_plane", "MapValueSemantics"),
    "PlannedImagePlaneTarget": ("torchsky.map.image_plane", "PlannedImagePlaneTarget"),
    "PlannedImagePlaneCubeTarget": (
        "torchsky.map.image_plane",
        "PlannedImagePlaneCubeTarget",
    ),
    "plan_image_plane_cube_target": (
        "torchsky.map.image_plane",
        "plan_image_plane_cube_target",
    ),
    "plan_image_plane_cube_target_from_header": (
        "torchsky.map.image_plane",
        "plan_image_plane_cube_target_from_header",
    ),
    "plan_image_plane_target": ("torchsky.map.image_plane", "plan_image_plane_target"),
    "MapUncertaintyMode": ("torchsky.map.product", "MapUncertaintyMode"),
    "build_map_benchmark_markdown": (
        "torchsky.map.product",
        "build_map_benchmark_markdown",
    ),
    "check_map_benchmark_regressions": (
        "torchsky.map.product",
        "check_map_benchmark_regressions",
    ),
    "estimate_map_inverse_variance": (
        "torchsky.map.product",
        "estimate_map_inverse_variance",
    ),
    "evaluate_map_frontier_regions": (
        "torchsky.map.product",
        "evaluate_map_frontier_regions",
    ),
    "evaluate_map_uncertainty": ("torchsky.map.product", "evaluate_map_uncertainty"),
    "evaluate_map_product": ("torchsky.map.product", "evaluate_map_product"),
    "map_benchmark_row_key": ("torchsky.map.product", "map_benchmark_row_key"),
    "plan_map_product": ("torchsky.map.product", "plan_map_product"),
    "render_field_product": ("torchsky.map.product", "render_field_product"),
    "write_map_benchmark_artifacts": (
        "torchsky.map.product",
        "write_map_benchmark_artifacts",
    ),
    "ExposurePose": ("torchsky.map.exposure_pose", "ExposurePose"),
    "create_exposure_poses": ("torchsky.map.exposure_pose", "create_exposure_poses"),
    "freeze_pose": ("torchsky.map.exposure_pose", "freeze_pose"),
    "ObservationOperatorBundle": (
        "torchsky.map.operators",
        "ObservationOperatorBundle",
    ),
    "ComponentDecompositionDiagnostics": (
        "torchsky.map.solvers",
        "ComponentDecompositionDiagnostics",
    ),
    "build_harmonic_cl_prior_operator": (
        "torchsky.map.solvers",
        "build_harmonic_cl_prior_operator",
    ),
    "build_harmonic_dense_prior_operator": (
        "torchsky.map.solvers",
        "build_harmonic_dense_prior_operator",
    ),
    "build_harmonic_degree_prior_operator": (
        "torchsky.map.solvers",
        "build_harmonic_degree_prior_operator",
    ),
    "build_spin_harmonic_kronecker_prior_operator": (
        "torchsky.map.solvers",
        "build_spin_harmonic_kronecker_prior_operator",
    ),
    "build_conditioned_component_mixture_operator": (
        "torchsky.map.solvers",
        "build_conditioned_component_mixture_operator",
    ),
    "build_state_component_mixture_operator": (
        "torchsky.map.solvers",
        "build_state_component_mixture_operator",
    ),
    "build_state_subspace_operator": (
        "torchsky.map.solvers",
        "build_state_subspace_operator",
    ),
    "build_state_matrix_operator": (
        "torchsky.map.solvers",
        "build_state_matrix_operator",
    ),
    "build_state_low_rank_operator": (
        "torchsky.map.solvers",
        "build_state_low_rank_operator",
    ),
    "build_teb_cls_prior_operator": (
        "torchsky.map.solvers",
        "build_teb_cls_prior_operator",
    ),
    "build_spherical_harmonic_solver_hooks": (
        "torchsky.map.solvers",
        "build_spherical_harmonic_solver_hooks",
    ),
    "build_spherical_healpix_solver_hooks": (
        "torchsky.map.solvers",
        "build_spherical_healpix_solver_hooks",
    ),
    "build_spherical_spin_harmonic_solver_hooks": (
        "torchsky.map.solvers",
        "build_spherical_spin_harmonic_solver_hooks",
    ),
    "build_state_diagonal_operator": (
        "torchsky.map.solvers",
        "build_state_diagonal_operator",
    ),
    "build_state_identity_operator": (
        "torchsky.map.solvers",
        "build_state_identity_operator",
    ),
    "build_state_kronecker_operator": (
        "torchsky.map.solvers",
        "build_state_kronecker_operator",
    ),
    "compose_bound_operators": ("torchsky.map.solvers", "compose_bound_operators"),
    "harmonic_cl_diagonal": ("torchsky.map.solvers", "harmonic_cl_diagonal"),
    "harmonic_dense_covariance": ("torchsky.map.solvers", "harmonic_dense_covariance"),
    "harmonic_degree_diagonal": ("torchsky.map.solvers", "harmonic_degree_diagonal"),
    "make_diagonal_preconditioner": (
        "torchsky.map.solvers",
        "make_diagonal_preconditioner",
    ),
    "HarmonicComponentDecomposition": (
        "torchsky.map.solvers",
        "HarmonicComponentDecomposition",
    ),
    "SurveyInstrumentComponentMixer": (
        "torchsky.map.solvers",
        "SurveyInstrumentComponentMixer",
    ),
    "teb_cls_precision_factors": ("torchsky.map.solvers", "teb_cls_precision_factors"),
    "BandpassConditionedField": ("torchsky.map.field", "BandpassConditionedField"),
    "FIELD_MODEL_PRESET_NAMES": ("torchsky.map.field", "FIELD_MODEL_PRESET_NAMES"),
    "FIELD_MODEL_PRESETS": ("torchsky.map.field", "FIELD_MODEL_PRESETS"),
    "FieldComponentAnchorTerm": ("torchsky.map.field", "FieldComponentAnchorTerm"),
    "FieldCentroidAnchorTerm": ("torchsky.map.field", "FieldCentroidAnchorTerm"),
    "FieldComponentDecomposition": (
        "torchsky.map.field",
        "FieldComponentDecomposition",
    ),
    "FieldCoordinateMode": ("torchsky.map.field", "FieldCoordinateMode"),
    "FieldDomain": ("torchsky.map.field", "FieldDomain"),
    "DetectorToSkyTransform": ("torchsky.map.field", "DetectorToSkyTransform"),
    "FieldFitResult": ("torchsky.map.field", "FieldFitResult"),
    "ForwardContext": ("torchsky.map.field", "ForwardContext"),
    "DetectorLaw": ("torchsky.map.field", "DetectorLaw"),
    "FourierMLPField": ("torchsky.map.field", "FourierMLPField"),
    "FoamInitializedINRField": (
        "torchsky.map.field",
        "FoamInitializedINRField",
    ),
    "HashGridConditionedField": ("torchsky.map.field", "HashGridConditionedField"),
    "HashGridMLPField": ("torchsky.map.field", "HashGridMLPField"),
    "HashGridResMLPField": ("torchsky.map.field", "HashGridResMLPField"),
    "HashGridSpectralField": ("torchsky.map.field", "HashGridSpectralField"),
    "FieldModelPreset": ("torchsky.map.field", "FieldModelPreset"),
    "FieldObservation": ("torchsky.map.field", "FieldObservation"),
    "PSFSpec": ("torchsky.map.field", "PSFSpec"),
    "TimeScale": ("torchsky.map.field", "TimeScale"),
    "FieldSpectralQuantity": ("torchsky.map.field", "FieldSpectralQuantity"),
    "KinematicState": ("torchsky.map.field", "KinematicState"),
    "GaussianBasisField": ("torchsky.map.field", "GaussianBasisField"),
    "LearnedSpectralBasisField": ("torchsky.map.field", "LearnedSpectralBasisField"),
    "OperatorConstrainedFeatureField": (
        "torchsky.map.field",
        "OperatorConstrainedFeatureField",
    ),
    "SkyEncoderBandpassField": ("torchsky.map.field", "SkyEncoderBandpassField"),
    "SkyLambdaAtlasField": ("torchsky.map.field", "SkyLambdaAtlasField"),
    "SkyLambdaFeatureField": ("torchsky.map.field", "SkyLambdaFeatureField"),
    "SkyLambdaFeatureAtlasField": ("torchsky.map.field", "SkyLambdaFeatureAtlasField"),
    "SkyLambdaKPlanesField": ("torchsky.map.field", "SkyLambdaKPlanesField"),
    "SkyLambdaPlanesField": ("torchsky.map.field", "SkyLambdaPlanesField"),
    "SpectralBasisField": ("torchsky.map.field", "SpectralBasisField"),
    "SpectralCapacityProfile": ("torchsky.map.field", "SpectralCapacityProfile"),
    "SpectralDataBudget": ("torchsky.map.field", "SpectralDataBudget"),
    "SpectralModelCapacityHints": ("torchsky.map.field", "SpectralModelCapacityHints"),
    "estimate_field_domain": ("torchsky.map.field", "estimate_field_domain"),
    "field_component_anchor_loss": (
        "torchsky.map.field",
        "field_component_anchor_loss",
    ),
    "field_component_centroid_anchor_loss": (
        "torchsky.map.field",
        "field_component_centroid_anchor_loss",
    ),
    "fit_field_component_decomposition": (
        "torchsky.map.field",
        "fit_field_component_decomposition",
    ),
    "fit_field_model": ("torchsky.map.field", "fit_field_model"),
    "fit_structured_noise_crossfit": (
        "torchsky.map.field",
        "fit_structured_noise_crossfit",
    ),
    "STRAINER_TRUNK_CHECKPOINT_FORMAT": (
        "torchsky.map.field",
        "STRAINER_TRUNK_CHECKPOINT_FORMAT",
    ),
    "load_strainer_trunk_checkpoint": (
        "torchsky.map.field",
        "load_strainer_trunk_checkpoint",
    ),
    "make_field_component_decomposition": (
        "torchsky.map.field",
        "make_field_component_decomposition",
    ),
    "make_field_model": ("torchsky.map.field", "make_field_model"),
    "make_field_model_preset": ("torchsky.map.field", "make_field_model_preset"),
    "pretrain_strainer_trunk_synthetic": (
        "torchsky.map.field",
        "pretrain_strainer_trunk_synthetic",
    ),
    "save_strainer_trunk_checkpoint": (
        "torchsky.map.field",
        "save_strainer_trunk_checkpoint",
    ),
    "evaluate_field_coordinates_vmapped": (
        "torchsky.map.field",
        "evaluate_field_coordinates_vmapped",
    ),
    "field_observations_from_bin_centers": (
        "torchsky.map.field",
        "field_observations_from_bin_centers",
    ),
    "field_observations_from_bin_edges": (
        "torchsky.map.field",
        "field_observations_from_bin_edges",
    ),
    "field_observations_from_ifu_cube": (
        "torchsky.map.field",
        "field_observations_from_ifu_cube",
    ),
    "ifu_cube_spectral_to_leading": (
        "torchsky.map.field",
        "ifu_cube_spectral_to_leading",
    ),
    "get_field_model_preset": ("torchsky.map.field", "get_field_model_preset"),
    "is_recommended_field_model": ("torchsky.map.field", "is_recommended_field_model"),
    "make_tangent_plane_header": ("torchsky.map.field", "make_tangent_plane_header"),
    "render_field_cube": ("torchsky.map.field", "render_field_cube"),
    "render_field_cube_via_healpix": (
        "torchsky.map.field",
        "render_field_cube_via_healpix",
    ),
    "render_field_healpix": ("torchsky.map.field", "render_field_healpix"),
    "render_field_observation": ("torchsky.map.field", "render_field_observation"),
    "render_field_observation_distribution": (
        "torchsky.map.field",
        "render_field_observation_distribution",
    ),
    "render_spectral_power_foam_observation_exact": (
        "torchsky.map.field",
        "render_spectral_power_foam_observation_exact",
    ),
    "spectral_data_budget": ("torchsky.map.field", "spectral_data_budget"),
    "suggest_num_spectral_samples": (
        "torchsky.map.field",
        "suggest_num_spectral_samples",
    ),
    "suggest_spectral_model_capacity": (
        "torchsky.map.field",
        "suggest_spectral_model_capacity",
    ),
    "SurveyDataTerm": ("torchsky.map.field", "SurveyDataTerm"),
    "multi_survey_charbonnier_loss": (
        "torchsky.map.field",
        "multi_survey_charbonnier_loss",
    ),
    "observation_charbonnier_loss": (
        "torchsky.map.field",
        "observation_charbonnier_loss",
    ),
    "predict_observation_mean": ("torchsky.map.field", "predict_observation_mean"),
    "scale_observation_for_rest_frame_scene": (
        "torchsky.map.field",
        "scale_observation_for_rest_frame_scene",
    ),
    "ObservationSpectralSummary": ("torchsky.map.field", "ObservationSpectralSummary"),
    "WireImplicitField": ("torchsky.map.field", "WireImplicitField"),
    "RECOMMENDED_FIELD_MODEL_KINDS": (
        "torchsky.map.field",
        "RECOMMENDED_FIELD_MODEL_KINDS",
    ),
    "SineImplicitField": ("torchsky.map.field", "SineImplicitField"),
    "SkyEncoderResMLPField": ("torchsky.map.field", "SkyEncoderResMLPField"),
    "SkyEncoderMLPField": ("torchsky.map.field", "SkyEncoderMLPField"),
    "StrainerSkyEncoderMLPField": ("torchsky.map.field", "StrainerSkyEncoderMLPField"),
    "field_image_quality_metrics": (
        "torchsky.map.field",
        "field_image_quality_metrics",
    ),
    "ResidualTileBasisField": ("torchsky.map.field", "ResidualTileBasisField"),
    "ResidualTileBasisSpec": ("torchsky.map.field", "ResidualTileBasisSpec"),
    "make_residual_tile_basis_field": (
        "torchsky.map.field",
        "make_residual_tile_basis_field",
    ),
    "residual_tile_basis_spec_from_observation": (
        "torchsky.map.field",
        "residual_tile_basis_spec_from_observation",
    ),
    "field_image_residual_product": (
        "torchsky.map.field",
        "field_image_residual_product",
    ),
    "field_quality_scorecard": (
        "torchsky.map.field",
        "field_quality_scorecard",
    ),
    "field_quality_acceptance_report": (
        "torchsky.map.field",
        "field_quality_acceptance_report",
    ),
    "FieldResidualProduct": ("torchsky.map.field", "FieldResidualProduct"),
    "FieldResidualTile": ("torchsky.map.field", "FieldResidualTile"),
    "render_field_fit_quality": ("torchsky.map.field", "render_field_fit_quality"),
    "render_field_quality": ("torchsky.map.field", "render_field_quality"),
    "render_field_residual": ("torchsky.map.field", "render_field_residual"),
    "render_field_residual_product": (
        "torchsky.map.field",
        "render_field_residual_product",
    ),
    "select_field_residual_tiles": (
        "torchsky.map.field",
        "select_field_residual_tiles",
    ),
    "healpix_nside_from_map_len": (
        "torchsky.map.healpix_image_plane",
        "healpix_nside_from_map_len",
    ),
    "sample_healpix_map_on_image_header": (
        "torchsky.map.healpix_image_plane",
        "sample_healpix_map_on_image_header",
    ),
    "sample_healpix_map_on_tan_header": (
        "torchsky.map.healpix_image_plane",
        "sample_healpix_map_on_tan_header",
    ),
    "accumulate_image_on_healpix_map_adjoint": (
        "torchsky.map.healpix_image_plane",
        "accumulate_image_on_healpix_map_adjoint",
    ),
    "rebandlimit_healpix_map_via_sht": (
        "torchsky.map.healpix_spectral",
        "rebandlimit_healpix_map_via_sht",
    ),
    "sample_alm_on_image_header_spectral": (
        "torchsky.map.healpix_spectral",
        "sample_alm_on_image_header_spectral",
    ),
    "sample_alm_on_tan_header_spectral": (
        "torchsky.map.healpix_spectral",
        "sample_alm_on_tan_header_spectral",
    ),
    "sample_spin_alm_on_image_header_spectral": (
        "torchsky.map.healpix_spectral",
        "sample_spin_alm_on_image_header_spectral",
    ),
    "sample_spin_alm_on_tan_header_spectral": (
        "torchsky.map.healpix_spectral",
        "sample_spin_alm_on_tan_header_spectral",
    ),
    "sample_healpix_map_on_image_header_harmonic": (
        "torchsky.map.healpix_spectral",
        "sample_healpix_map_on_image_header_harmonic",
    ),
    "sample_healpix_map_on_image_header_spectral": (
        "torchsky.map.healpix_spectral",
        "sample_healpix_map_on_image_header_spectral",
    ),
    "sample_healpix_spin_map_on_image_header_harmonic": (
        "torchsky.map.healpix_spectral",
        "sample_healpix_spin_map_on_image_header_harmonic",
    ),
    "sample_healpix_map_on_tan_header_harmonic": (
        "torchsky.map.healpix_spectral",
        "sample_healpix_map_on_tan_header_harmonic",
    ),
    "sample_healpix_map_on_tan_header_spectral": (
        "torchsky.map.healpix_spectral",
        "sample_healpix_map_on_tan_header_spectral",
    ),
    "sample_healpix_spin_map_on_tan_header_harmonic": (
        "torchsky.map.healpix_spectral",
        "sample_healpix_spin_map_on_tan_header_harmonic",
    ),
    "accumulate_image_on_alm_adjoint": (
        "torchsky.map.healpix_spectral",
        "accumulate_image_on_alm_adjoint",
    ),
    "accumulate_image_on_spin_alm_adjoint": (
        "torchsky.map.healpix_spectral",
        "accumulate_image_on_spin_alm_adjoint",
    ),
}


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        module = import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value

    # Keep compatibility with internal module access (e.g. torchsky.map.field)
    if name in {
        "field",
        "image_plane",
        "product",
        "target",
        "solvers",
        "operators",
        "exposure_pose",
    }:
        module = import_module(f"torchsky.map.{name}")
        globals()[name] = module
        return module

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | set(_EXPORTS))


if TYPE_CHECKING:
    from .image_plane import (
        MapResult as MapResult,
        MultiBandMapResult as MultiBandMapResult,
        SpectralCubeMapResult as SpectralCubeMapResult,
        coadd_image_plane as coadd_image_plane,
        coadd_image_plane_cube as coadd_image_plane_cube,
        coadd_image_plane_multiband as coadd_image_plane_multiband,
    )
    from .product import (
        MapBandpassSpec as MapBandpassSpec,
        MapProductResult as MapProductResult,
        MapProductSpec as MapProductSpec,
        stack_map_product as stack_map_product,
    )
    from .target import TargetRenderSpec as TargetRenderSpec


__all__ = [
    "coadd_image_plane",
    "coadd_image_plane_cube",
    "coadd_image_plane_multiband",
    "stack_map_product",
    "MapProductSpec",
    "TargetRenderSpec",
    "MapBandpassSpec",
    "MapProductResult",
    "MapResult",
    "MultiBandMapResult",
    "SpectralCubeMapResult",
]
