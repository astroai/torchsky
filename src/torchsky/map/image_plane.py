"""Image-plane-first mapmaking primitives."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor
from torchfits import read, write

from ..device import default_compute_device
from ..spectral import (
    SpectralAxis,
    spectral_axis_from_header,
    spectral_bin_edges_from_header,
)
from ..sphere.moc import MOC
from ..wcs import WCS
from .fft_affine_ldu import resample_fft_ldu_affine_2d
from .fft_resample import resample_diagonal_affine_fft2d, resample_translation_fft2d
from .io_utils import (
    _observation_header_for_area_scaling,
)
from .io_utils import (
    load_observation_image_and_header as _io_load_observation_image_and_header,
)

# Header probe for area scaling: recoverable failures when reading metadata only.
from .observation import (
    ApertureUncertainty,
    CubeResult,
    CubeTarget,
    ImageObservation,
    MapResult,
    MultibandResult,
    Observation,
    OutlierConfig,
    PlannedImagePlaneCubeTarget,
    PlannedImagePlaneTarget,
    ResampleConfig,
    SkyConfig,
    StackConfig,
    Target,
)
from .operators.background import SkyBackgroundOp
from .operators.bundle import ObservationOperatorBundle
from .operators.config import (
    MapResamplingKernel,
    MapStackingMethod,
    MapValueSemantics,
    validate_map_resampling_kernel,
    validate_map_stacking_method,
    validate_map_value_semantics,
)
from .operators.execution import (
    PreparedImageObservationData,
    accumulate_projected_observation,
    finalize_projected_stack,
    is_plain_tan_header,
    masked_median_stack,
    prepare_image_observation_data,
    project_image_observation_geometry,
    require_exact_median_scratch_capacity,
    require_sigma_clipped_scratch_capacity,
    spilled_exact_median_stack,
    spilled_overlap_background_stack,
    spilled_sigma_clipped_stack,
    stack_inputs_from_projected_observation,
    wcs_has_distortion,
)
from .operators.geometry import ForwardSplatGeometryOp, InterpolationGeometryOp
from .operators.image_grid import (
    headers_equivalent_for_direct_copy,
    normalized_header_for_shape,
    target_shape_from_header,
    validate_image_plane_2d,
)
from .operators.mask_weight import load_image_observation_weight_map
from .operators.noise import ExplicitNoiseOp
from .operators.pixel_area import (
    pixel_area_sr_map,
    representative_pixel_area_sr,
)
from .operators.radiometry import apply_image_observation_calibration
from .operators.resampling import (
    evaluate_tan_projective_homography,
    finalize_periodic_resampled_weighted,
    infer_diagonal_scale_shift_pixel_map,
    infer_fft_translation_shift_yx,
    infer_linear_affine_pixel_map,
    infer_tan_projective_homography,
    sample_image_plane_batched_channels,
    sample_image_plane_diagonal_variance,
)
from .operators.spectral import ExplicitSpectralResponseOp
from .periodic_affine import resample_linear_affine_periodic_bilinear_2d
from .sky import CrossExposureSkyCorrector, MeshSkyEstimator
from .spectral_affine import resample_linear_affine_spectral_2d
from .spectral_nonuniform import resample_nonuniform_spectral_2d
from .spectral_nufft import resample_nonuniform_spectral_fast_2d
from .spectral_projection import (
    _build_band_metadata,
    _build_spectral_projection,
    _observation_spectral_columns,
    _validated_spectral_axis,
    _validated_spectral_bin_edges,
)
from .windowing import (
    _LARGE_TAN_WINDOWING_EDGE_SAMPLES,  # noqa: F401
    _LARGE_TAN_WINDOWING_MARGIN_PIXELS,  # noqa: F401
    _LARGE_TAN_WINDOWING_MIN_TARGET_PIXELS,  # noqa: F401
    _build_target_world_grid,
    _prepare_target,
    _PreparedImagePlaneTarget,
    _quadrature_target_world_grids,
    _subtarget_header_for_window,
    _weight_map_to_moc,
    _window_large_tan_target,
)



def load_observation_image_and_header(
    obs: object,
    device: str | torch.device,
) -> tuple[Tensor, dict]:
    return _io_load_observation_image_and_header(obs, device, read_fn=read)


def _resolve_map_device(device: torch.device | str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return default_compute_device()


_HEADER_PROBE_EXC: tuple[type[BaseException], ...] = (
    OSError,
    RuntimeError,
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
    ImportError,
    ModuleNotFoundError,
)


@dataclass
class _ProjectedImagePlaneObservation:
    numer: Tensor
    weight: Tensor
    variance_weight: Tensor | None
    overlap: Tensor
    moc: MOC | None
    target_window: tuple[int, int, int, int] | None = None


def _accumulate_variance_weight(
    target: Tensor,
    contribution: Tensor,
    target_window: tuple[int, int, int, int] | None,
) -> None:
    if target_window is None:
        target.add_(contribution)
        return
    y0, y1, x0, x1 = target_window
    if y1 > y0 and x1 > x0:
        target[..., y0:y1, x0:x1].add_(contribution)


def _set_context_bit(
    context: Tensor,
    observation_index: int,
    support: Tensor,
    target_window: tuple[int, int, int, int] | None,
) -> None:
    word = int(observation_index) // 32
    bit = int(observation_index) % 32
    encoded = support.to(dtype=torch.int64) * (1 << bit)
    plane = context[word]
    if target_window is None:
        plane.bitwise_or_(encoded)
        return
    y0, y1, x0, x1 = target_window
    if y1 > y0 and x1 > x0:
        plane[y0:y1, x0:x1].bitwise_or_(encoded)


def _observation_operator_provenance(
    observation: ImageObservation,
) -> dict[str, Any]:
    return {
        "path": observation.path,
        "hdu": observation.hdu,
        "weight": float(observation.weight),
        "weight_is_inverse_variance": bool(observation.weight_is_inverse_variance),
        "weight_path": observation.weight_path,
        "weight_hdu": observation.weight_hdu,
        "mask_path": observation.mask_path,
        "mask_hdu": observation.mask_hdu,
        "primary_beam_path": observation.primary_beam_path,
        "primary_beam_hdu": observation.primary_beam_hdu,
        "photometric_scale": float(observation.photometric_scale),
        "background_offset": float(observation.background_offset),
    }


def _context_provenance(
    observations: Sequence[ImageObservation],
    *,
    resampling_kernel: MapResamplingKernel,
    value_semantics: MapValueSemantics,
    stacking_method: MapStackingMethod,
    drizzle_pixfrac: float,
    surface_brightness_area_scaling: bool,
    data_dependent_background: bool,
    has_correlation_factor: bool = False,
) -> dict[str, Any]:
    provenance: dict[str, Any] = {
        "context_word_bits": 32,
        "context_observations": [
            {"index": index, "path": observation.path, "hdu": observation.hdu}
            for index, observation in enumerate(observations)
        ],
        "observation_operators": [
            _observation_operator_provenance(observation)
            for observation in observations
        ],
        "map_execution": {
            "resampling_kernel": resampling_kernel,
            "value_semantics": value_semantics,
            "stacking_method": stacking_method,
            "drizzle_pixfrac": float(drizzle_pixfrac),
            "surface_brightness_area_scaling": bool(
                surface_brightness_area_scaling
            ),
            "data_dependent_background": bool(data_dependent_background),
        },
    }
    if has_correlation_factor:
        provenance["noise_correlation"] = {
            "kind": "local_inverse_enbw",
            "definition": "first_coefficient_weight/squared_coefficient_weight",
            "scope": "exact for locally shift-invariant large apertures; diagnostic otherwise",
        }
    return provenance


def _linear_inverse_variance(
    normalization: Tensor,
    variance_weight: Tensor,
    *,
    value_semantics: MapValueSemantics,
    pixel_area_sr: Tensor | None,
) -> Tensor:
    scaled_normalization = normalization
    if value_semantics == "flux":
        if pixel_area_sr is None:
            raise RuntimeError("target pixel area is required for flux uncertainty")
        scaled_normalization = normalization / torch.clamp(
            pixel_area_sr,
            min=torch.finfo(pixel_area_sr.dtype).tiny,
        )
    inverse_variance = torch.where(
        variance_weight > 0.0,
        torch.square(scaled_normalization) / variance_weight,
        torch.zeros_like(normalization),
    )
    return inverse_variance


def _local_noise_correlation_factor(
    correlation_weight: Tensor,
    variance_weight: Tensor,
) -> Tensor:
    """Return the local inverse-ENBW factor from first/squared coefficients.

    For locally shift-invariant resampling of independent detector noise this is
    the exact large-aperture variance inflation relative to the diagonal-only
    variance. On distorted or edge-truncated grids it remains a local kernel
    diagnostic, not a covariance matrix.
    """
    supported = (correlation_weight > 0.0) & (variance_weight > 0.0)
    return torch.where(
        supported,
        correlation_weight / variance_weight,
        torch.zeros_like(correlation_weight),
    )


_DRIZZLE_FORWARD_CHUNK_PIXELS = 262_144

_DEFAULT_MAP_OPERATOR_BUNDLE = ObservationOperatorBundle(
    geometry=InterpolationGeometryOp(),
    spectral_response=ExplicitSpectralResponseOp(),
    radiometry=None,
    background=SkyBackgroundOp(),
    mask_weight=None,
    noise=ExplicitNoiseOp(),
)


def _should_apply_surface_brightness_area_scaling(
    observations: Sequence[ImageObservation],
    *,
    target_header: dict,
    target_shape: tuple[int, int],
    first_header: dict | None = None,
) -> bool:
    if not observations or not is_plain_tan_header(target_header):
        return True
    target_area_sr = representative_pixel_area_sr(
        WCS(target_header),
        target_shape,
        device=torch.device("cpu"),
    )
    inv_area_sum = 0.0
    for obs_idx, obs in enumerate(observations):
        header_dict_obs = _observation_header_for_area_scaling(
            obs,
            cached_header=first_header if obs_idx == 0 else None,
        )
        if not is_plain_tan_header(header_dict_obs):
            return True
        source_wcs = WCS(header_dict_obs)
        if wcs_has_distortion(source_wcs):
            return True
        source_shape = (
            int(header_dict_obs.get("NAXIS2", 0)),
            int(header_dict_obs.get("NAXIS1", 0)),
        )
        if source_shape[0] <= 0 or source_shape[1] <= 0:
            return True
        source_area_sr = representative_pixel_area_sr(
            source_wcs,
            source_shape,
            device=torch.device("cpu"),
        )
        inv_area_sum += 1.0 / max(float(source_area_sr), 1e-20)
    harmonic_mean_source_area_sr = float(len(observations)) / max(inv_area_sum, 1e-20)
    return float(target_area_sr) >= harmonic_mean_source_area_sr




def _is_splat_kernel(kernel: MapResamplingKernel) -> bool:
    return kernel in {
        "splat_point",
        "splat_box",
        "drizzle_turbo",
        "drizzle_gaussian",
        "drizzle_lanczos2",
        "drizzle_lanczos3",
    }


def _convert_world_between_wcs(
    lon: Tensor,
    lat: Tensor,
    from_wcs: WCS,
    to_wcs: WCS,
) -> tuple[Tensor, Tensor]:
    from_frame = getattr(from_wcs, "celestial_frame", "ICRS")
    to_frame = getattr(to_wcs, "celestial_frame", "ICRS")
    from_eq = getattr(from_wcs, "celestial_equinox", None)
    to_eq = getattr(to_wcs, "celestial_equinox", None)
    if from_frame == to_frame and (
        from_eq == to_eq or from_frame in {"ICRS", "GALACTIC", "ECLIPTIC"}
    ):
        return lon, lat
    orig_shape = lon.shape
    lon_c, lat_c = from_wcs.convert_celestial(
        lon.reshape(-1),
        lat.reshape(-1),
        to_frame,
        to_equinox=to_eq,
    )
    return lon_c.reshape(orig_shape), lat_c.reshape(orig_shape)


_FORWARD_SOURCE_RATIO = 0.2  # use forward projection when source < 20% of target


def _splat_for_interpolation(kernel: MapResamplingKernel) -> MapResamplingKernel | None:
    """Return equivalent splat kernel for interpolation kernels, or None."""
    if kernel == "nearest":
        return "splat_point"
    if kernel == "bilinear":
        return "splat_box"
    return None


def _requires_roundtrip_validity_check(source_wcs: WCS) -> bool:
    # SIP inverse paths still benefit from an explicit round-trip consistency gate.
    # For TPV, the gate was useful while the distorted WCS dispatch could route
    # through a wrong TAN-only shortcut, but after fixing that dispatch the same
    # gate became over-rejective on realistic warped-stack benchmarks.
    return bool(getattr(source_wcs, "sip", None) is not None)


def _coarse_grid_shape(
    height: int, width: int, *, coarse_stride: int = 128
) -> tuple[int, int, int]:
    stride = max(1, min(height, width) // coarse_stride, 1)
    stride = min(stride, 128)
    hc = max(1, height // stride)
    wc = max(1, width // stride)
    if hc < 4 or wc < 4:
        stride = 1
        hc, wc = height, width
    return stride, hc, wc


def _interpolate_forward_mapping_from_coarse(
    source_wcs: WCS,
    target_wcs: WCS,
    height: int,
    width: int,
    *,
    device: torch.device,
    coarse_stride: int = 128,
) -> tuple[Tensor, Tensor]:
    stride, hc, wc = _coarse_grid_shape(height, width, coarse_stride=coarse_stride)

    y_coarse = torch.arange(0, height, stride, dtype=torch.float64, device=device)
    x_coarse = torch.arange(0, width, stride, dtype=torch.float64, device=device)

    yy, xx = torch.meshgrid(y_coarse, x_coarse, indexing="ij")
    ra_c, dec_c = target_wcs.pixel_to_world(xx, yy)
    ra_c, dec_c = _convert_world_between_wcs(ra_c, dec_c, target_wcs, source_wcs)
    x_src_c, y_src_c = source_wcs.world_to_pixel(ra_c, dec_c)

    if stride == 1:
        x_out = x_src_c.reshape(height, width).to(dtype=torch.float32)
        y_out = y_src_c.reshape(height, width).to(dtype=torch.float32)
        return x_out, y_out

    hc, wc = int(y_coarse.numel()), int(x_coarse.numel())
    coarse = (
        torch.stack([x_src_c, y_src_c], dim=0)
        .to(dtype=torch.float32)
        .reshape(1, 2, hc, wc)
    )

    y_g = torch.arange(height, dtype=torch.float32, device=device) / stride
    x_g = torch.arange(width, dtype=torch.float32, device=device) / stride

    # Bicubic grid_sample expects one row/col of margin for boundary handling.
    # The last coarse pixel has index stride*(Nc-1). Pixels beyond that have
    # normalized coordinate > 1 and are clamped to border.
    y_n = y_g / max(hc - 1, 1) * 2.0 - 1.0
    x_n = x_g / max(wc - 1, 1) * 2.0 - 1.0
    gy, gx = torch.meshgrid(y_n.clamp(-1.0, 1.0), x_n.clamp(-1.0, 1.0), indexing="ij")
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)

    full = F.grid_sample(
        coarse, grid, mode="bicubic", padding_mode="border", align_corners=True
    )
    return full[0, 0], full[0, 1]


def _requires_target_pixel_quadrature(
    source_wcs: WCS,
    *,
    value_semantics: MapValueSemantics,
) -> bool:
    if value_semantics != "flux":
        return False
    return bool(
        getattr(source_wcs, "sip", None) is not None
        or getattr(source_wcs, "_is_tpv", False)
    )


def _tpv_affine_envelope(
    source_wcs: WCS,
    image_shape: tuple[int, int],
) -> tuple[float, float, float, float]:
    tpv = source_wcs.tpv
    if tpv is None:
        raise RuntimeError("TPV envelope requires a TPV WCS")
    h_src, w_src = int(image_shape[0]), int(image_shape[1])
    x_corners = torch.tensor(
        [0.0, float(w_src - 1), 0.0, float(w_src - 1)], dtype=torch.float64
    )
    y_corners = torch.tensor(
        [0.0, 0.0, float(h_src - 1), float(h_src - 1)], dtype=torch.float64
    )
    rel_x = x_corners + 1.0 - source_wcs._crpix0
    rel_y = y_corners + 1.0 - source_wcs._crpix1
    u_corners = source_wcs._cd00 * rel_x + source_wcs._cd01 * rel_y
    v_corners = source_wcs._cd10 * rel_x + source_wcs._cd11 * rel_y
    u_abs = float(torch.max(torch.abs(u_corners)).item())
    v_abs = float(torch.max(torch.abs(v_corners)).item())
    r_abs = math.sqrt(u_abs * u_abs + v_abs * v_abs)

    b1 = b2 = a11 = a12 = a21 = a22 = 0.0
    bound_xi = 0.0
    for px, py, pr, coeff in tpv.terms1:
        if coeff == 0.0:
            continue
        if pr == 0 and px == 0 and py == 0:
            b1 += coeff
        elif pr == 0 and px == 1 and py == 0:
            a11 += coeff
        elif pr == 0 and px == 0 and py == 1:
            a12 += coeff
        else:
            bound_xi += abs(coeff) * (u_abs**px) * (v_abs**py) * (r_abs**pr)

    bound_eta = 0.0
    for px, py, pr, coeff in tpv.terms2:
        if coeff == 0.0:
            continue
        if pr == 0 and px == 0 and py == 0:
            b2 += coeff
        elif pr == 0 and px == 0 and py == 1:
            a21 += coeff
        elif pr == 0 and px == 1 and py == 0:
            a22 += coeff
        else:
            bound_eta += abs(coeff) * (v_abs**px) * (u_abs**py) * (r_abs**pr)

    xi_aff = b1 + a11 * u_corners + a12 * v_corners
    eta_aff = b2 + a21 * u_corners + a22 * v_corners
    return (
        float(torch.min(xi_aff).item()) - bound_xi,
        float(torch.max(xi_aff).item()) + bound_xi,
        float(torch.min(eta_aff).item()) - bound_eta,
        float(torch.max(eta_aff).item()) + bound_eta,
    )


def _tpv_source_cap_radius_deg(source_wcs: WCS, image_shape: tuple[int, int]) -> float:
    xi_min, xi_max, eta_min, eta_max = _tpv_affine_envelope(source_wcs, image_shape)
    r_max = math.sqrt(
        max(abs(xi_min), abs(xi_max)) ** 2 + max(abs(eta_min), abs(eta_max)) ** 2
    )
    return math.degrees(math.atan(r_max * (math.pi / 180.0)))


def _angular_separation_deg(
    ra0_deg: float,
    dec0_deg: float,
    ra1_deg: float,
    dec1_deg: float,
) -> float:
    ra0 = math.radians(ra0_deg)
    dec0 = math.radians(dec0_deg)
    ra1 = math.radians(ra1_deg)
    dec1 = math.radians(dec1_deg)
    cos_sep = math.sin(dec0) * math.sin(dec1) + math.cos(dec0) * math.cos(
        dec1
    ) * math.cos(ra0 - ra1)
    cos_sep = min(1.0, max(-1.0, cos_sep))
    return math.degrees(math.acos(cos_sep))


def _source_pixel_diameter_arcsec(
    source_wcs: WCS, shape: tuple[int, int], *, device: torch.device
) -> float:
    area_sr = pixel_area_sr_map(source_wcs, shape, device=device)
    median_area_sr = float(torch.median(area_sr).item())
    if not math.isfinite(median_area_sr) or median_area_sr <= 0.0:
        return 0.0
    radius_rad = math.sqrt(median_area_sr / math.pi)
    return 2.0 * radius_rad * (180.0 / math.pi) * 3600.0


def _roundtrip_validity_mask(
    source_wcs: WCS,
    *,
    x_src: Tensor,
    y_src: Tensor,
    flat_ra: Tensor,
    flat_dec: Tensor,
    image_shape: tuple[int, int],
) -> Tensor:
    if not _requires_roundtrip_validity_check(source_wcs):
        return torch.ones_like(x_src, dtype=torch.bool)
    flat_x = x_src.reshape(-1)
    flat_y = y_src.reshape(-1)
    valid = torch.zeros_like(flat_x, dtype=torch.bool)
    in_bounds = (
        (flat_x >= 0.0)
        & (flat_x <= float(image_shape[1] - 1))
        & (flat_y >= 0.0)
        & (flat_y <= float(image_shape[0] - 1))
    )
    if not bool(in_bounds.any()):
        return valid.reshape_as(x_src)
    ra_back, dec_back = source_wcs.pixel_to_world(flat_x[in_bounds], flat_y[in_bounds])
    dra = (ra_back - flat_ra[in_bounds]) * torch.cos(torch.deg2rad(flat_dec[in_bounds]))
    ddec = dec_back - flat_dec[in_bounds]
    err_arcsec = torch.sqrt(dra * dra + ddec * ddec) * 3600.0
    max_err_arcsec = max(
        _source_pixel_diameter_arcsec(source_wcs, image_shape, device=x_src.device),
        1.0e-3,
    )
    valid[in_bounds] = err_arcsec <= max_err_arcsec
    return valid.reshape_as(x_src)


def _load_observation_weight_map(
    obs: ImageObservation,
    image: Tensor,
    *,
    device: torch.device,
) -> Tensor:
    return load_image_observation_weight_map(obs, image, device=device)


def _effective_observation_photometric_scale(
    obs: ImageObservation,
    hdr: dict | None = None,
) -> float:
    scale = float(obs.photometric_scale)
    flux_kw = getattr(obs, "flux_scale_keyword", None)
    exptime_kw = getattr(obs, "exposure_time_keyword", None)
    if flux_kw is not None or exptime_kw is not None:
        if not isinstance(hdr, dict):
            hdr = getattr(obs, "header", None)
        if not isinstance(hdr, dict):
            hdr = _observation_header_for_area_scaling(obs)
        if flux_kw is not None and flux_kw in hdr:
            scale = scale * float(hdr[flux_kw])
        if exptime_kw is not None and exptime_kw in hdr:
            exptime = float(hdr[exptime_kw])
            if not math.isfinite(exptime) or exptime <= 0.0:
                raise ValueError(f"Header {exptime_kw} must be finite and positive")
            scale = scale / exptime
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("Observation photometric_scale must be finite and positive")
    return scale


def _apply_observation_calibration(
    obs: ImageObservation,
    image: Tensor,
    source_weight: Tensor,
) -> tuple[Tensor, Tensor]:
    scale = _effective_observation_photometric_scale(obs)
    return apply_image_observation_calibration(
        photometric_scale=scale,
        background_offset=obs.background_offset,
        image=image,
        source_weight=source_weight,
    )


def _resample_weighted_to_target(
    image: Tensor,
    source_weight: Tensor,
    source_wcs: WCS,
    target: _PreparedImagePlaneTarget,
    *,
    resampling_kernel: MapResamplingKernel,
    value_semantics: MapValueSemantics,
    return_variance_weight: bool = False,
) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor | None]:
    image = validate_image_plane_2d(image).to(dtype=torch.float32)
    source_weight = validate_image_plane_2d(source_weight).to(dtype=torch.float32)

    h_src, w_src = image.shape
    target_representative_area_sr = target.representative_pixel_area_sr

    # Reproject weighted signal and weight separately, then form the weighted mean
    # only after all observations have been accumulated on the target grid.
    if value_semantics == "flux":
        source_area_sr = pixel_area_sr_map(
            source_wcs,
            tuple(image.shape),
            device=image.device,
        )
        weighted_image = image * source_weight * source_area_sr
        weighted_weight = source_weight * (source_area_sr * source_area_sr)
        source_numerator_variance = weighted_weight
    elif target_representative_area_sr is not None:
        source_area_sr = representative_pixel_area_sr(
            source_wcs,
            tuple(image.shape),
            device=image.device,
        )
        info_density = source_weight / max(source_area_sr, 1e-20)
        weighted_image = image * info_density
        weighted_weight = info_density
        source_numerator_variance = source_weight / max(
            source_area_sr * source_area_sr, 1e-40
        )
    else:
        weighted_image = image * source_weight
        weighted_weight = source_weight
        source_numerator_variance = source_weight
    support_image = (
        (source_weight > 0.0).to(dtype=torch.float32)
        if value_semantics == "flux"
        else None
    )
    coefficient_variance_supported = resampling_kernel in {
        "nearest",
        "bilinear",
        "bicubic",
        "lanczos2",
        "lanczos3",
        "lanczos4",
        "lanczos5",
        "mitchell_netravali",
        "catmull_rom",
    }

    def _result(
        numer: Tensor,
        weight: Tensor,
        variance_weight: Tensor | None = None,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor | None]:
        if return_variance_weight:
            return numer, weight, variance_weight
        return numer, weight

    if resampling_kernel == "fft_shift":
        if _requires_target_pixel_quadrature(
            source_wcs,
            value_semantics=value_semantics,
        ):
            raise ValueError(
                "fft_shift is incompatible with flux quadrature on distorted WCS; "
                "use bilinear or another kernel"
            )
        if (h_src, w_src) != (target.shape[0], target.shape[1]):
            raise ValueError(
                "fft_shift requires matching source and target array shapes "
                f"(got source {(h_src, w_src)} vs target {tuple(target.shape)})"
            )
        try:
            sy, sx = infer_fft_translation_shift_yx(
                source_wcs,
                target.wcs,
                (h_src, w_src),
            )
        except ValueError as exc:
            raise ValueError(
                "fft_shift could not infer a pure pixel translation between "
                "source and target WCS"
            ) from exc
        shift_kw = {"shift_y_pix": float(sy), "shift_x_pix": float(sx)}
        sampled_numer = resample_translation_fft2d(weighted_image, **shift_kw)
        sampled_weight = resample_translation_fft2d(weighted_weight, **shift_kw)
        sampled_support = (
            resample_translation_fft2d(support_image, **shift_kw)
            if support_image is not None
            else None
        )
        sampled_numer, sampled_weight = finalize_periodic_resampled_weighted(
            sampled_numer,
            sampled_weight,
            sampled_support,
            value_semantics=value_semantics,
            target_representative_area_sr=target_representative_area_sr,
        )
        return _result(sampled_numer, sampled_weight)

    if resampling_kernel == "fft_diagonal_affine":
        if _requires_target_pixel_quadrature(
            source_wcs,
            value_semantics=value_semantics,
        ):
            raise ValueError(
                "fft_diagonal_affine is incompatible with flux quadrature on distorted WCS; "
                "use bilinear or another kernel"
            )
        if (h_src, w_src) != (target.shape[0], target.shape[1]):
            raise ValueError(
                "fft_diagonal_affine requires matching source and target array shapes "
                f"(got source {(h_src, w_src)} vs target {tuple(target.shape)})"
            )
        try:
            d_sx, d_bx, d_sy, d_by = infer_diagonal_scale_shift_pixel_map(
                source_wcs,
                target.wcs,
                (h_src, w_src),
            )
        except ValueError as exc:
            raise ValueError(
                "fft_diagonal_affine could not infer a separable diagonal affine pixel map "
                "between source and target WCS"
            ) from exc
        aff_kw = {
            "scale_x_pix": float(d_sx),
            "shift_x_pix": float(d_bx),
            "scale_y_pix": float(d_sy),
            "shift_y_pix": float(d_by),
        }
        sampled_numer = resample_diagonal_affine_fft2d(weighted_image, **aff_kw)
        sampled_weight = resample_diagonal_affine_fft2d(weighted_weight, **aff_kw)
        sampled_support = (
            resample_diagonal_affine_fft2d(support_image, **aff_kw)
            if support_image is not None
            else None
        )
        sampled_numer, sampled_weight = finalize_periodic_resampled_weighted(
            sampled_numer,
            sampled_weight,
            sampled_support,
            value_semantics=value_semantics,
            target_representative_area_sr=target_representative_area_sr,
        )
        return _result(sampled_numer, sampled_weight)

    if resampling_kernel == "periodic_linear_affine":
        if _requires_target_pixel_quadrature(
            source_wcs,
            value_semantics=value_semantics,
        ):
            raise ValueError(
                "periodic_linear_affine is incompatible with flux quadrature on distorted WCS; "
                "use bilinear or another kernel"
            )
        if (h_src, w_src) != (target.shape[0], target.shape[1]):
            raise ValueError(
                "periodic_linear_affine requires matching source and target array shapes "
                f"(got source {(h_src, w_src)} vs target {tuple(target.shape)})"
            )
        try:
            la = infer_linear_affine_pixel_map(
                source_wcs,
                target.wcs,
                (h_src, w_src),
            )
        except ValueError as exc:
            raise ValueError(
                "periodic_linear_affine could not infer a linear affine pixel map "
                "between source and target WCS"
            ) from exc
        a00, a01, a10, a11, pbx, pby = la
        aff_kw = {
            "a00": a00,
            "a01": a01,
            "a10": a10,
            "a11": a11,
            "bx": pbx,
            "by": pby,
        }
        sampled_numer = resample_linear_affine_periodic_bilinear_2d(
            weighted_image, **aff_kw
        )
        sampled_weight = resample_linear_affine_periodic_bilinear_2d(
            weighted_weight, **aff_kw
        )
        sampled_support = (
            resample_linear_affine_periodic_bilinear_2d(support_image, **aff_kw)
            if support_image is not None
            else None
        )
        sampled_numer, sampled_weight = finalize_periodic_resampled_weighted(
            sampled_numer,
            sampled_weight,
            sampled_support,
            value_semantics=value_semantics,
            target_representative_area_sr=target_representative_area_sr,
        )
        return _result(sampled_numer, sampled_weight)

    if resampling_kernel == "fft_linear_affine":
        if _requires_target_pixel_quadrature(
            source_wcs,
            value_semantics=value_semantics,
        ):
            raise ValueError(
                "fft_linear_affine is incompatible with flux quadrature on distorted WCS; "
                "use bilinear or another kernel"
            )
        if (h_src, w_src) != (target.shape[0], target.shape[1]):
            raise ValueError(
                "fft_linear_affine requires matching source and target array shapes "
                f"(got source {(h_src, w_src)} vs target {tuple(target.shape)})"
            )
        try:
            la = infer_linear_affine_pixel_map(
                source_wcs,
                target.wcs,
                (h_src, w_src),
            )
        except ValueError as exc:
            raise ValueError(
                "fft_linear_affine could not infer a linear affine pixel map "
                "between source and target WCS"
            ) from exc
        a00, a01, a10, a11, pbx, pby = la
        aff_kw = {
            "a00": a00,
            "a01": a01,
            "a10": a10,
            "a11": a11,
            "bx": pbx,
            "by": pby,
        }
        try:
            sampled_numer = resample_fft_ldu_affine_2d(weighted_image, **aff_kw)
            sampled_weight = resample_fft_ldu_affine_2d(weighted_weight, **aff_kw)
            sampled_support = (
                resample_fft_ldu_affine_2d(support_image, **aff_kw)
                if support_image is not None
                else None
            )
        except ValueError as exc:
            raise ValueError(
                "fft_linear_affine failed (try periodic_linear_affine for arbitrary affines)"
            ) from exc
        sampled_numer, sampled_weight = finalize_periodic_resampled_weighted(
            sampled_numer,
            sampled_weight,
            sampled_support,
            value_semantics=value_semantics,
            target_representative_area_sr=target_representative_area_sr,
        )
        return _result(sampled_numer, sampled_weight)

    if resampling_kernel == "spectral_linear_affine":
        if _requires_target_pixel_quadrature(
            source_wcs,
            value_semantics=value_semantics,
        ):
            raise ValueError(
                "spectral_linear_affine is incompatible with flux quadrature on distorted WCS; "
                "use bilinear or another kernel"
            )
        if (h_src, w_src) != (target.shape[0], target.shape[1]):
            raise ValueError(
                "spectral_linear_affine requires matching source and target array shapes "
                f"(got source {(h_src, w_src)} vs target {tuple(target.shape)})"
            )
        try:
            la = infer_linear_affine_pixel_map(
                source_wcs,
                target.wcs,
                (h_src, w_src),
            )
        except ValueError as exc:
            raise ValueError(
                "spectral_linear_affine could not infer a linear affine pixel map "
                "between source and target WCS"
            ) from exc
        a00, a01, a10, a11, pbx, pby = la
        aff_kw = {
            "a00": a00,
            "a01": a01,
            "a10": a10,
            "a11": a11,
            "bx": pbx,
            "by": pby,
            "boundary_mode": "zero",
        }
        sampled_numer = resample_linear_affine_spectral_2d(weighted_image, **aff_kw)
        sampled_weight = resample_linear_affine_spectral_2d(weighted_weight, **aff_kw)
        sampled_support = (
            resample_linear_affine_spectral_2d(support_image, **aff_kw)
            if support_image is not None
            else None
        )
        sampled_numer, sampled_weight = finalize_periodic_resampled_weighted(
            sampled_numer,
            sampled_weight,
            sampled_support,
            value_semantics=value_semantics,
            target_representative_area_sr=target_representative_area_sr,
        )
        return _result(sampled_numer, sampled_weight)

    def _sample_on_source_grid(
        x_src: Tensor,
        y_src: Tensor,
        *,
        flat_ra: Tensor | None = None,
        flat_dec: Tensor | None = None,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor | None]:
        # WCS round trips can land an exact edge just outside the image.  Keep
        # the repair much smaller than a pixel even for float32/large mosaics.
        roundoff_tol = min(
            1.0e-7,
            4096.0 * torch.finfo(x_src.dtype).eps * float(max(h_src, w_src, 1)),
        )
        if getattr(source_wcs, "sip", None) is not None:
            # SIP.invert_distortion guarantees residuals to its 1e-6 pixel
            # stopping tolerance. Treat only that solver-sized boundary shell
            # as the exact detector edge so tile batch size cannot change support.
            roundoff_tol = max(roundoff_tol, 1.0e-6)
        x_src = torch.where(
            (x_src < 0.0) & (x_src >= -roundoff_tol),
            torch.zeros_like(x_src),
            x_src,
        )
        x_src = torch.where(
            (x_src > float(w_src - 1)) & (x_src <= float(w_src - 1) + roundoff_tol),
            torch.full_like(x_src, float(w_src - 1)),
            x_src,
        )
        y_src = torch.where(
            (y_src < 0.0) & (y_src >= -roundoff_tol),
            torch.zeros_like(y_src),
            y_src,
        )
        y_src = torch.where(
            (y_src > float(h_src - 1)) & (y_src <= float(h_src - 1) + roundoff_tol),
            torch.full_like(y_src, float(h_src - 1)),
            y_src,
        )
        sampled_variance_weight = None
        if resampling_kernel == "spectral_nonuniform":
            sampled_numer = resample_nonuniform_spectral_2d(
                weighted_image,
                x_src,
                y_src,
                boundary_mode="zero",
            )
            sampled_weight = resample_nonuniform_spectral_2d(
                weighted_weight,
                x_src,
                y_src,
                boundary_mode="zero",
            )
            sampled_support = (
                resample_nonuniform_spectral_2d(
                    support_image,
                    x_src,
                    y_src,
                    boundary_mode="zero",
                )
                if support_image is not None
                else None
            )
        elif resampling_kernel == "spectral_nonuniform_fast":
            sampled_numer = resample_nonuniform_spectral_fast_2d(
                weighted_image,
                x_src,
                y_src,
                boundary_mode="zero",
            )
            sampled_weight = resample_nonuniform_spectral_fast_2d(
                weighted_weight,
                x_src,
                y_src,
                boundary_mode="zero",
            )
            sampled_support = (
                resample_nonuniform_spectral_fast_2d(
                    support_image,
                    x_src,
                    y_src,
                    boundary_mode="zero",
                )
                if support_image is not None
                else None
            )
        else:
            x_norm = (2.0 * x_src / max(w_src - 1, 1)) - 1.0
            y_norm = (2.0 * y_src / max(h_src - 1, 1)) - 1.0
            grid = torch.stack([x_norm, y_norm], dim=-1).to(dtype=torch.float32)

            stacked_planes = (
                torch.stack([weighted_image, weighted_weight, support_image], dim=0)
                if support_image is not None
                else torch.stack([weighted_image, weighted_weight], dim=0)
            )
            sampled_planes = sample_image_plane_batched_channels(
                stacked_planes,
                x_src=x_src,
                y_src=y_src,
                grid=grid,
                kernel=resampling_kernel,
            )
            if x_src.ndim == 2:
                sampled_numer = sampled_planes[0]
                sampled_weight = sampled_planes[1]
                sampled_support = (
                    sampled_planes[2] if support_image is not None else None
                )
            else:
                sampled_numer = sampled_planes[:, 0]
                sampled_weight = sampled_planes[:, 1]
                sampled_support = (
                    sampled_planes[:, 2] if support_image is not None else None
                )
        in_bounds = (
            (x_src >= 0.0)
            & (x_src <= (w_src - 1))
            & (y_src >= 0.0)
            & (y_src <= (h_src - 1))
        )
        if _requires_roundtrip_validity_check(source_wcs):
            in_bounds = in_bounds & _roundtrip_validity_mask(
                source_wcs,
                x_src=x_src,
                y_src=y_src,
                flat_ra=flat_ra,
                flat_dec=flat_dec,
                image_shape=(h_src, w_src),
            )
        if sampled_support is not None:
            in_bounds = in_bounds & (sampled_support > 0.25)
        sampled_numer = torch.where(
            in_bounds,
            sampled_numer,
            torch.zeros_like(sampled_numer),
        )
        sampled_weight = torch.where(
            in_bounds,
            sampled_weight,
            torch.zeros_like(sampled_weight),
        )
        sampled_weight = torch.where(
            sampled_weight > 0.0,
            sampled_weight,
            torch.zeros_like(sampled_weight),
        )
        sampled_numer = torch.where(
            sampled_weight > 0.0,
            sampled_numer,
            torch.zeros_like(sampled_numer),
        )
        if return_variance_weight and coefficient_variance_supported:
            sample_valid = in_bounds & (sampled_weight > 0.0)
            sampled_variance_weight = sample_image_plane_diagonal_variance(
                source_numerator_variance,
                x_src=x_src,
                y_src=y_src,
                kernel=resampling_kernel,
                average_coordinate_batch=x_src.ndim == 3,
                sample_valid=sample_valid,
            )
        if (
            value_semantics == "surface_brightness"
            and target_representative_area_sr is not None
        ):
            sampled_numer = sampled_numer * float(target_representative_area_sr)
            sampled_weight = sampled_weight * float(target_representative_area_sr)
            if sampled_variance_weight is not None:
                sampled_variance_weight = (
                    sampled_variance_weight * float(target_representative_area_sr) ** 2
                )
        return _result(sampled_numer, sampled_weight, sampled_variance_weight)

    def _sample_on_world_grid(
        ra_deg: Tensor, dec_deg: Tensor
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor | None]:
        ra_conv, dec_conv = _convert_world_between_wcs(
            ra_deg, dec_deg, target.wcs, source_wcs
        )
        flat_ra = ra_conv.reshape(-1)
        flat_dec = dec_conv.reshape(-1)
        if (
            bool(getattr(source_wcs, "_is_tpv", False))
            and getattr(source_wcs, "sip", None) is None
            and getattr(target.wcs, "celestial_frame", "ICRS")
            == getattr(source_wcs, "celestial_frame", "ICRS")
        ):
            source_cap_radius_deg = _tpv_source_cap_radius_deg(
                source_wcs,
                (h_src, w_src),
            )
            center_sep_deg = _angular_separation_deg(
                float(source_wcs.alpha0),
                float(source_wcs.delta0),
                target.center_ra_deg,
                target.center_dec_deg,
            )
            if center_sep_deg > source_cap_radius_deg + target.cap_radius_deg:
                zeros = torch.zeros_like(ra_deg, dtype=torch.float32)
                return _result(
                    zeros,
                    zeros,
                    zeros if coefficient_variance_supported else None,
                )
        x_src, y_src = source_wcs.world_to_pixel(flat_ra, flat_dec)
        x_src = x_src.reshape_as(ra_deg)
        y_src = y_src.reshape_as(dec_deg)
        return _sample_on_source_grid(
            x_src,
            y_src,
            flat_ra=flat_ra,
            flat_dec=flat_dec,
        )

    if not _requires_target_pixel_quadrature(
        source_wcs,
        value_semantics=value_semantics,
    ):
        if target.ra_deg is not None:
            return _sample_on_world_grid(target.ra_deg, target.dec_deg)
        h_tgt, w_tgt = int(target.shape[0]), int(target.shape[1])
        if not _requires_roundtrip_validity_check(source_wcs):
            homography = infer_tan_projective_homography(
                source_wcs, target.wcs, (h_tgt, w_tgt)
            )
            if homography is not None:
                x_src, y_src = evaluate_tan_projective_homography(
                    homography,
                    h_tgt,
                    w_tgt,
                    device=image.device,
                )
                return _sample_on_source_grid(
                    x_src, y_src, flat_ra=None, flat_dec=None
                )
        if min(h_tgt, w_tgt) < 500 or _requires_roundtrip_validity_check(source_wcs):
            ra_g, dec_g = _build_target_world_grid(
                target.wcs,
                h_tgt,
                w_tgt,
                device=image.device,
            )
            return _sample_on_world_grid(ra_g, dec_g)
        x_src, y_src = _interpolate_forward_mapping_from_coarse(
            source_wcs,
            target.wcs,
            h_tgt,
            w_tgt,
            device=image.device,
        )
        return _sample_on_source_grid(x_src, y_src, flat_ra=None, flat_dec=None)

    device = target.wcs.device
    accum_numer = torch.zeros(target.shape, dtype=torch.float32, device=device)
    accum_weight = torch.zeros(target.shape, dtype=torch.float32, device=device)
    quadrature_world_grids = target.quadrature_world_grids
    if quadrature_world_grids is None:
        quadrature_world_grids = _quadrature_target_world_grids(
            target.wcs,
            target.shape,
            device=device,
        )
    flat_ra = torch.cat(
        [sub_ra.reshape(-1) for sub_ra, _ in quadrature_world_grids], dim=0
    )
    flat_dec = torch.cat(
        [sub_dec.reshape(-1) for _, sub_dec in quadrature_world_grids], dim=0
    )
    flat_ra, flat_dec = _convert_world_between_wcs(
        flat_ra, flat_dec, target.wcs, source_wcs
    )
    x_src_all, y_src_all = source_wcs.world_to_pixel(flat_ra, flat_dec)
    x_src_b = x_src_all.reshape(len(quadrature_world_grids), *target.shape)
    y_src_b = y_src_all.reshape(len(quadrature_world_grids), *target.shape)
    sub_result = _sample_on_source_grid(
        x_src_b,
        y_src_b,
        flat_ra=flat_ra,
        flat_dec=flat_dec,
    )
    sub_numer, sub_weight = sub_result[:2]
    sub_variance_weight = sub_result[2] if len(sub_result) == 3 else None
    accum_numer = sub_numer.mean(dim=0)
    accum_weight = sub_weight.mean(dim=0)
    accum_numer = torch.where(
        accum_weight > 0.0,
        accum_numer,
        torch.zeros_like(accum_numer),
    )
    return _result(accum_numer, accum_weight, sub_variance_weight)


def _validated_map_resampling_kernel(
    resampling_kernel: str,
) -> MapResamplingKernel:
    return validate_map_resampling_kernel(resampling_kernel)


def _validated_map_value_semantics(
    value_semantics: str,
) -> MapValueSemantics:
    return validate_map_value_semantics(value_semantics)


def _validated_drizzle_pixfrac(drizzle_pixfrac: float) -> float:
    pixfrac = float(drizzle_pixfrac)
    if not math.isfinite(pixfrac) or pixfrac <= 0.0:
        raise ValueError("drizzle_pixfrac must be a finite positive float")
    return pixfrac


def _validated_map_stacking_method(
    stacking_method: str,
) -> MapStackingMethod:
    return validate_map_stacking_method(stacking_method)


def _drizzle_separable_1d_weight(
    delta: Tensor,
    *,
    kernel: MapResamplingKernel,
    pixfrac: float,
) -> Tensor:
    scale = max(float(pixfrac), 1.0e-6)
    u = delta / scale
    if kernel == "drizzle_gaussian":
        # FWHM = pixfrac => sigma = pixfrac / (2 * sqrt(2 * ln(2)))
        sigma_u = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        return torch.where(
            torch.abs(u) <= 2.5,
            torch.exp(-0.5 * torch.square(u / sigma_u)),
            torch.zeros_like(u),
        )
    a = 2.0 if kernel == "drizzle_lanczos2" else 3.0
    return torch.where(
        torch.abs(u) < a,
        torch.sinc(u) * torch.sinc(u / a),
        torch.zeros_like(u),
    )


def _forward_project_weighted_to_target(
    image: Tensor,
    source_weight: Tensor,
    source_wcs: WCS,
    target: _PreparedImagePlaneTarget,
    *,
    resampling_kernel: MapResamplingKernel,
    value_semantics: MapValueSemantics,
    drizzle_pixfrac: float,
    return_variance_weight: bool = False,
) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
    if not _is_splat_kernel(resampling_kernel):
        raise ValueError(
            f"forward projection only supports splat kernels, got {resampling_kernel!r}"
        )

    image = validate_image_plane_2d(image).to(dtype=torch.float32)
    source_weight = validate_image_plane_2d(source_weight).to(dtype=torch.float32)

    if value_semantics == "flux":
        source_area_sr = pixel_area_sr_map(
            source_wcs,
            tuple(image.shape),
            device=image.device,
        )
        weighted_image = image * source_weight * source_area_sr
        weighted_weight = source_weight * (source_area_sr * source_area_sr)
        source_numerator_variance = weighted_weight
    elif target.representative_pixel_area_sr is not None:
        source_area_sr = representative_pixel_area_sr(
            source_wcs,
            tuple(image.shape),
            device=image.device,
        )
        info_density = source_weight / max(source_area_sr, 1e-20)
        weighted_image = image * info_density
        weighted_weight = info_density
        source_numerator_variance = source_weight / max(
            source_area_sr * source_area_sr, 1e-40
        )
    else:
        weighted_image = image * source_weight
        weighted_weight = source_weight
        source_numerator_variance = source_weight

    valid = torch.isfinite(weighted_weight) & (weighted_weight > 0.0)
    if not bool(valid.any()):
        zeros = torch.zeros(target.shape, dtype=torch.float32, device=image.device)
        if return_variance_weight:
            return zeros, zeros, zeros
        return zeros, zeros

    ys, xs = torch.nonzero(valid, as_tuple=True)
    flat_values_numer = weighted_image[ys, xs].to(dtype=torch.float32)
    flat_values_weight = weighted_weight[ys, xs].to(dtype=torch.float32)
    flat_values_variance = (
        source_numerator_variance[ys, xs].to(dtype=torch.float32)
        if return_variance_weight
        else None
    )
    flat_numer = torch.zeros(
        target.shape[0] * target.shape[1], dtype=torch.float32, device=image.device
    )
    flat_weight = torch.zeros_like(flat_numer)
    flat_variance_weight = (
        torch.zeros_like(flat_numer) if return_variance_weight else None
    )
    width = int(target.shape[1])
    height = int(target.shape[0])

    for start in range(0, int(xs.numel()), _DRIZZLE_FORWARD_CHUNK_PIXELS):
        end = min(start + _DRIZZLE_FORWARD_CHUNK_PIXELS, int(xs.numel()))
        x_chunk = xs[start:end].to(dtype=torch.float64)
        y_chunk = ys[start:end].to(dtype=torch.float64)
        numer_chunk = flat_values_numer[start:end]
        weight_chunk = flat_values_weight[start:end]
        variance_chunk = (
            flat_values_variance[start:end]
            if flat_values_variance is not None
            else None
        )

        ra_deg, dec_deg = source_wcs.pixel_to_world(x_chunk, y_chunk)
        ra_deg, dec_deg = _convert_world_between_wcs(
            ra_deg, dec_deg, source_wcs, target.wcs
        )
        x_tgt, y_tgt = target.wcs.world_to_pixel(ra_deg, dec_deg)
        x_tgt = x_tgt.to(dtype=torch.float32)
        y_tgt = y_tgt.to(dtype=torch.float32)
        finite = torch.isfinite(x_tgt) & torch.isfinite(y_tgt)
        if not bool(finite.any()):
            continue

        x_tgt = x_tgt[finite]
        y_tgt = y_tgt[finite]
        numer_chunk = numer_chunk[finite]
        weight_chunk = weight_chunk[finite]
        if variance_chunk is not None:
            variance_chunk = variance_chunk[finite]

        if resampling_kernel == "splat_point":
            ix = torch.round(x_tgt).to(dtype=torch.int64)
            iy = torch.round(y_tgt).to(dtype=torch.int64)
            inside = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
            if not bool(inside.any()):
                continue
            idx = iy[inside] * width + ix[inside]
            flat_numer.scatter_add_(0, idx, numer_chunk[inside])
            flat_weight.scatter_add_(0, idx, weight_chunk[inside])
            if flat_variance_weight is not None and variance_chunk is not None:
                flat_variance_weight.scatter_add_(0, idx, variance_chunk[inside])
            continue

        if resampling_kernel in {
            "drizzle_gaussian",
            "drizzle_lanczos2",
            "drizzle_lanczos3",
        }:
            footprint = max(float(drizzle_pixfrac), 1e-6)
            base_r = (
                2.5
                if resampling_kernel == "drizzle_gaussian"
                else (2.0 if resampling_kernel == "drizzle_lanczos2" else 3.0)
            )
            radius = max(1, int(math.ceil(base_r * footprint)))
            xc = torch.round(x_tgt).to(dtype=torch.int64)
            yc = torch.round(y_tgt).to(dtype=torch.int64)
            offsets = list(range(-radius, radius + 1))
            wx_list = [
                _drizzle_separable_1d_weight(
                    (xc + off).to(dtype=torch.float32) - x_tgt,
                    kernel=resampling_kernel,
                    pixfrac=footprint,
                )
                for off in offsets
            ]
            wy_list = [
                _drizzle_separable_1d_weight(
                    (yc + off).to(dtype=torch.float32) - y_tgt,
                    kernel=resampling_kernel,
                    pixfrac=footprint,
                )
                for off in offsets
            ]
            sum_wx = torch.clamp(torch.stack(wx_list, dim=0).sum(dim=0), min=1.0e-12)
            sum_wy = torch.clamp(torch.stack(wy_list, dim=0).sum(dim=0), min=1.0e-12)
            wx_list = [w / sum_wx for w in wx_list]
            wy_list = [w / sum_wy for w in wy_list]
            for off_y, wy in zip(offsets, wy_list, strict=True):
                iy = yc + off_y
                if not bool(((iy >= 0) & (iy < height)).any()):
                    continue
                for off_x, wx in zip(offsets, wx_list, strict=True):
                    ix = xc + off_x
                    frac = wx * wy
                    inside = (
                        (ix >= 0)
                        & (ix < width)
                        & (iy >= 0)
                        & (iy < height)
                        & (frac > 0.0)
                    )
                    if not bool(inside.any()):
                        continue
                    idx = iy[inside] * width + ix[inside]
                    flat_numer.scatter_add_(0, idx, numer_chunk[inside] * frac[inside])
                    flat_weight.scatter_add_(
                        0, idx, weight_chunk[inside] * frac[inside]
                    )
                    if flat_variance_weight is not None and variance_chunk is not None:
                        flat_variance_weight.scatter_add_(
                            0,
                            idx,
                            variance_chunk[inside] * torch.square(frac[inside]),
                        )
            continue

        footprint = max(float(drizzle_pixfrac), 1e-6)
        x_tgt = x_tgt - 0.5 * (footprint - 1.0)
        y_tgt = y_tgt - 0.5 * (footprint - 1.0)
        x0 = torch.floor(x_tgt).to(dtype=torch.int64)
        y0 = torch.floor(y_tgt).to(dtype=torch.int64)
        dx = torch.clamp((x_tgt - x0.to(dtype=torch.float32)) / footprint, 0.0, 1.0)
        dy = torch.clamp((y_tgt - y0.to(dtype=torch.float32)) / footprint, 0.0, 1.0)
        for off_y, wy in ((0, 1.0 - dy), (1, dy)):
            iy = y0 + off_y
            if not bool(((iy >= 0) & (iy < height)).any()):
                continue
            for off_x, wx in ((0, 1.0 - dx), (1, dx)):
                ix = x0 + off_x
                frac = wx * wy
                inside = (
                    (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height) & (frac > 0.0)
                )
                if not bool(inside.any()):
                    continue
                idx = iy[inside] * width + ix[inside]
                flat_numer.scatter_add_(0, idx, numer_chunk[inside] * frac[inside])
                flat_weight.scatter_add_(0, idx, weight_chunk[inside] * frac[inside])
                if flat_variance_weight is not None and variance_chunk is not None:
                    flat_variance_weight.scatter_add_(
                        0,
                        idx,
                        variance_chunk[inside] * torch.square(frac[inside]),
                    )

    numer = flat_numer.reshape(target.shape)
    weight = flat_weight.reshape(target.shape)
    target_representative_area_sr = target.representative_pixel_area_sr
    if (
        value_semantics == "surface_brightness"
        and target_representative_area_sr is not None
    ):
        numer = numer * float(target_representative_area_sr)
        weight = weight * float(target_representative_area_sr)
        if flat_variance_weight is not None:
            flat_variance_weight = (
                flat_variance_weight * float(target_representative_area_sr) ** 2
            )
    numer = torch.where(weight > 0.0, numer, torch.zeros_like(numer))
    weight = torch.where(weight > 0.0, weight, torch.zeros_like(weight))
    if return_variance_weight:
        assert flat_variance_weight is not None
        variance_weight = flat_variance_weight.reshape(target.shape)
        variance_weight = torch.where(
            weight > 0.0,
            variance_weight,
            torch.zeros_like(variance_weight),
        )
        return numer, weight, variance_weight
    return numer, weight


def _project_observation_to_target_chunk(
    obs: ImageObservation,
    *,
    image: Tensor | None = None,
    header_dict_obs: dict | None = None,
    source_wcs: WCS | None = None,
    prepared_observation: PreparedImageObservationData | None = None,
    prepared_target: _PreparedImagePlaneTarget | None,
    target_header: dict,
    target_shape: tuple[int, int],
    device: torch.device,
    footprint_order: int | None,
    resampling_kernel: MapResamplingKernel,
    value_semantics: MapValueSemantics,
    drizzle_pixfrac: float,
    background_estimator: Callable[[Tensor, Tensor | None], Tensor] | None = None,
) -> _ProjectedImagePlaneObservation:
    if not torch.isfinite(torch.tensor(obs.weight, dtype=torch.float64)):
        raise ValueError("Observation weights must be finite")
    if not (obs.weight > 0.0):
        raise ValueError("Observation weights must be positive")

    if image is None or header_dict_obs is None:
        image, header_dict_obs = load_observation_image_and_header(
            obs, device=str(device)
        )
    image = image.to(device=device)
    if prepared_observation is None:
        prepared_observation = prepare_image_observation_data(
            obs,
            image,
            device=device,
            resampling_kernel=resampling_kernel,
            load_weight_map_fn=_load_observation_weight_map,
            apply_calibration_fn=_apply_observation_calibration,
            background_estimator=background_estimator,
        )
    image = prepared_observation.image
    source_weight = prepared_observation.source_weight
    operator_bundle = prepared_observation.operator_bundle
    has_explicit_noise = operator_bundle.noise is not None
    prepared_target_view = prepared_target
    target_window = None
    if (
        headers_equivalent_for_direct_copy(header_dict_obs, target_header)
        and tuple(image.shape) == target_shape
    ):
        if value_semantics == "flux":
            if prepared_target is None or prepared_target.pixel_area_sr is None:
                raise RuntimeError(
                    "prepared_target with pixel area is required for flux semantics"
                )
            sampled_weight = source_weight.to(dtype=torch.float32) * (
                prepared_target.pixel_area_sr * prepared_target.pixel_area_sr
            )
            sampled_numer = (
                image.to(dtype=torch.float32)
                * source_weight.to(dtype=torch.float32)
                * prepared_target.pixel_area_sr
            )
        else:
            sampled_weight = source_weight.to(dtype=torch.float32)
            sampled_numer = image.to(dtype=torch.float32) * sampled_weight
        sampled_variance_weight = sampled_weight if has_explicit_noise else None
    else:
        if prepared_target is None:
            raise RuntimeError("prepared_target is required for reprojection")
        if source_wcs is None:
            source_wcs = WCS(header_dict_obs)
        prepared_target_view, target_window = _window_large_tan_target(
            source_wcs=source_wcs,
            source_header=header_dict_obs,
            source_shape=tuple(image.shape),
            target=prepared_target,
            target_header=target_header,
            value_semantics=value_semantics,
        )
        if prepared_target_view is None:
            sampled_numer = torch.zeros((0, 0), dtype=torch.float32, device=device)
            sampled_weight = torch.zeros((0, 0), dtype=torch.float32, device=device)
            sampled_variance_weight = (
                torch.zeros_like(sampled_weight) if has_explicit_noise else None
            )
        else:
            _kernel = resampling_kernel
            if isinstance(operator_bundle.geometry, ForwardSplatGeometryOp):
                _forward_remap = _splat_for_interpolation(resampling_kernel)
                if _forward_remap is not None:
                    src_pix = int(image.shape[0]) * int(image.shape[1])
                    tgt_pix = int(prepared_target_view.shape[0]) * int(
                        prepared_target_view.shape[1]
                    )
                    if src_pix < tgt_pix * _FORWARD_SOURCE_RATIO:
                        _kernel = _forward_remap
            geometry_result = project_image_observation_geometry(
                image,
                source_weight,
                source_wcs,
                prepared_target_view,
                operator_bundle=operator_bundle,
                resampling_kernel=_kernel,
                value_semantics=value_semantics,
                drizzle_pixfrac=drizzle_pixfrac,
                pixel_area_sr_map_fn=pixel_area_sr_map,
                representative_pixel_area_fn=representative_pixel_area_sr,
                chunk_pixels=_DRIZZLE_FORWARD_CHUNK_PIXELS,
                resample_weighted_to_target_fn=_resample_weighted_to_target,
                forward_project_weighted_to_target_fn=(
                    _forward_project_weighted_to_target
                ),
                return_variance_weight=has_explicit_noise,
            )
            sampled_numer, sampled_weight = geometry_result[:2]
            sampled_variance_weight = (
                geometry_result[2] if len(geometry_result) == 3 else None
            )
    overlap = (sampled_weight > 0.0).to(torch.int32)
    moc = None
    if footprint_order is not None:
        if prepared_target_view is None:
            moc = MOC(torch.empty((0,), dtype=torch.int64))
        else:
            moc = _weight_map_to_moc(
                sampled_weight,
                prepared_target_view,
                max_order=int(footprint_order),
            )
    return _ProjectedImagePlaneObservation(
        numer=sampled_numer,
        weight=sampled_weight,
        variance_weight=sampled_variance_weight,
        overlap=overlap,
        moc=moc,
        target_window=target_window,
    )


def _project_observation_to_target(
    obs: ImageObservation,
    *,
    image: Tensor | None = None,
    header_dict_obs: dict | None = None,
    source_wcs: WCS | None = None,
    prepared_target: _PreparedImagePlaneTarget | None,
    target_header: dict,
    target_shape: tuple[int, int],
    device: torch.device,
    footprint_order: int | None,
    resampling_kernel: MapResamplingKernel,
    value_semantics: MapValueSemantics,
    drizzle_pixfrac: float = 1.0,
    background_estimator: Callable[[Tensor, Tensor | None], Tensor] | None = None,
) -> tuple[Tensor, Tensor, Tensor, MOC | None]:
    projected = _project_observation_to_target_chunk(
        obs,
        image=image,
        header_dict_obs=header_dict_obs,
        source_wcs=source_wcs,
        prepared_target=prepared_target,
        target_header=target_header,
        target_shape=target_shape,
        device=device,
        footprint_order=footprint_order,
        resampling_kernel=resampling_kernel,
        value_semantics=value_semantics,
        drizzle_pixfrac=drizzle_pixfrac,
        background_estimator=background_estimator,
    )
    if projected.target_window is None:
        return projected.numer, projected.weight, projected.overlap, projected.moc
    y0, y1, x0, x1 = projected.target_window
    sampled_numer = torch.zeros(
        target_shape, dtype=projected.numer.dtype, device=device
    )
    sampled_weight = torch.zeros(
        target_shape, dtype=projected.weight.dtype, device=device
    )
    overlap = torch.zeros(target_shape, dtype=projected.overlap.dtype, device=device)
    if y1 > y0 and x1 > x0:
        sampled_numer[y0:y1, x0:x1] = projected.numer
        sampled_weight[y0:y1, x0:x1] = projected.weight
        overlap[y0:y1, x0:x1] = projected.overlap
    return sampled_numer, sampled_weight, overlap, projected.moc


def _true_image_shape(header: dict) -> tuple[int, int]:
    """Return (height, width) of the image described by *header*.

    Compressed FITS stores the true dimensions in ``ZNAXIS1``/``ZNAXIS2``;
    uncompressed images use the standard ``NAXIS1``/``NAXIS2`` cards.
    """
    if header.get("ZCMPTYPE") is not None:
        nx = int(header.get("ZNAXIS1", header.get("NAXIS1", 0)))
        ny = int(header.get("ZNAXIS2", header.get("NAXIS2", 0)))
    else:
        nx = int(header.get("NAXIS1", 0))
        ny = int(header.get("NAXIS2", 0))
    return (ny, nx)


def _observation_header_for_footprint(
    obs: ImageObservation,
    device: torch.device | str | None = None,
) -> dict:
    """Read the FITS header for *obs*, correcting compressed NAXIS cards."""
    from .io_utils import _observation_header_for_area_scaling

    hdr = _observation_header_for_area_scaling(obs)
    ny, nx = _true_image_shape(hdr)
    if ny > 0 and nx > 0:
        hdr["NAXIS1"] = nx
        hdr["NAXIS2"] = ny
    return hdr


def _spherical_union_center(footprints: Tensor) -> tuple[float, float]:
    """Return a seam-safe mean direction for celestial footprint points."""
    ra = torch.deg2rad(footprints[:, 0].to(dtype=torch.float64))
    dec = torch.deg2rad(footprints[:, 1].to(dtype=torch.float64))
    cos_dec = torch.cos(dec)
    mean = torch.stack(
        (cos_dec * torch.cos(ra), cos_dec * torch.sin(ra), torch.sin(dec)),
        dim=-1,
    ).mean(dim=0)
    norm = torch.linalg.vector_norm(mean)
    if not bool(torch.isfinite(norm)) or float(norm) <= 1.0e-12:
        raise RuntimeError(
            "Union footprint has no unique spherical centre; provide a reference "
            "target or partition the mosaic"
        )
    mean = mean / norm
    center_ra = torch.rad2deg(torch.atan2(mean[1], mean[0])) % 360.0
    center_dec = torch.rad2deg(torch.asin(torch.clamp(mean[2], -1.0, 1.0)))
    return float(center_ra), float(center_dec)


def _world_to_tangent_plane(
    footprint: Tensor,
    *,
    center_ra_deg: float,
    center_dec_deg: float,
) -> tuple[Tensor, Tensor]:
    """Project celestial coordinates onto the exact gnomonic tangent plane."""
    ra = torch.deg2rad(footprint[:, 0].to(dtype=torch.float64))
    dec = torch.deg2rad(footprint[:, 1].to(dtype=torch.float64))
    ra0 = math.radians(center_ra_deg)
    dec0 = math.radians(center_dec_deg)
    dra = torch.remainder(ra - ra0 + math.pi, 2.0 * math.pi) - math.pi
    sin_dec, cos_dec = torch.sin(dec), torch.cos(dec)
    sin_dec0, cos_dec0 = math.sin(dec0), math.cos(dec0)
    denom = sin_dec0 * sin_dec + cos_dec0 * cos_dec * torch.cos(dra)
    if bool(torch.any(~torch.isfinite(denom))) or bool(torch.any(denom <= 0.0)):
        raise RuntimeError(
            "Union footprint crosses the TAN projection horizon; provide a "
            "reference target or partition the mosaic"
        )
    radians_to_degrees = 180.0 / math.pi
    xi = cos_dec * torch.sin(dra) / denom * radians_to_degrees
    eta = (
        (cos_dec0 * sin_dec - sin_dec0 * cos_dec * torch.cos(dra))
        / denom
        * radians_to_degrees
    )
    if bool(torch.any(~torch.isfinite(xi))) or bool(torch.any(~torch.isfinite(eta))):
        raise RuntimeError("Union footprint produced non-finite TAN coordinates")
    return xi, eta


def _adaptive_tangent_boundary(
    wcs: WCS,
    *,
    height: int,
    width: int,
    center_ra_deg: float,
    center_dec_deg: float,
    pixel_scale_deg: float,
    tolerance_px: float,
    max_depth: int,
) -> Tensor:
    def project(pixel: tuple[float, float]) -> Tensor:
        pixel_tensor = torch.tensor([pixel], dtype=torch.float64)
        world = wcs.pixel_to_world(pixel_tensor)
        if isinstance(world, tuple):
            world = torch.stack(world[:2], dim=-1)
        world = world.detach().to(device="cpu", dtype=torch.float64)
        xi, eta = _world_to_tangent_plane(
            world,
            center_ra_deg=center_ra_deg,
            center_dec_deg=center_dec_deg,
        )
        point = torch.stack((xi[0], eta[0]))
        if not bool(torch.isfinite(point).all()):
            raise RuntimeError("Observation boundary produced non-finite coordinates")
        return point

    output: list[Tensor] = []

    def sample_segment(
        p0: tuple[float, float],
        p1: tuple[float, float],
        q0: Tensor,
        q1: Tensor,
        depth: int,
    ) -> None:
        fractions = (0.25, 0.5, 0.75)
        pixels = [
            (
                p0[0] + fraction * (p1[0] - p0[0]),
                p0[1] + fraction * (p1[1] - p0[1]),
            )
            for fraction in fractions
        ]
        projected = [project(pixel) for pixel in pixels]
        errors = [
            torch.linalg.vector_norm(value - ((1.0 - fraction) * q0 + fraction * q1))
            / float(pixel_scale_deg)
            for fraction, value in zip(fractions, projected, strict=True)
        ]
        if max(float(error) for error in errors) <= float(tolerance_px):
            output.append(q1)
            return
        if depth >= int(max_depth):
            raise RuntimeError(
                "Could not bound curved observation footprint within the requested tolerance"
            )
        midpoint = pixels[1]
        q_midpoint = projected[1]
        sample_segment(p0, midpoint, q0, q_midpoint, depth + 1)
        sample_segment(midpoint, p1, q_midpoint, q1, depth + 1)

    corners = (
        (-0.5, -0.5),
        (float(width) - 0.5, -0.5),
        (float(width) - 0.5, float(height) - 0.5),
        (-0.5, float(height) - 0.5),
    )
    for edge_index in range(4):
        p0 = corners[edge_index]
        p1 = corners[(edge_index + 1) % 4]
        q0 = project(p0)
        q1 = project(p1)
        if not output:
            output.append(q0)
        sample_segment(p0, p1, q0, q1, 0)
    return torch.stack(output, dim=0)


def _resolve_plan_frame_prefixes(
    frame: str,
) -> tuple[str, str, str, str | None, float | None]:
    norm = str(frame).strip().lower()
    if norm in {"icrs", "equatorial", "j2000"}:
        return "ICRS", "RA--", "DEC-", "ICRS", None
    if norm == "fk5":
        return "FK5", "RA--", "DEC-", "FK5", 2000.0
    if norm in {"fk4", "b1950"}:
        return "FK4", "RA--", "DEC-", "FK4", 1950.0
    if norm in {"galactic", "gal"}:
        return "GALACTIC", "GLON", "GLAT", None, None
    if norm in {"ecliptic", "ecl"}:
        return "ECLIPTIC", "ELON", "ELAT", None, None
    raise ValueError(
        f"Unsupported coordinate frame {frame!r}; expected 'icrs', 'fk5', 'fk4', 'galactic', or 'ecliptic'"
    )


def _plan_union_footprint_target(
    observations: Sequence[ImageObservation],
    *,
    device: torch.device | str | None = None,
    padding: float = 1.05,
    tolerance_px: float = 0.25,
    max_depth: int = 20,
    projection: str = "TAN",
    frame: str = "icrs",
    center: str | tuple[float, float] = "union",
    pixel_scale: str | float = "median",
    pixel_scale_arcsec: float | None = None,
    shape: tuple[int, int] | None = None,
    rotation_deg: float = 0.0,
) -> tuple[dict, tuple[int, int]]:
    """Plan one rotation- and RA-seam-safe grid for all observations."""
    if not math.isfinite(float(padding)) or float(padding) < 1.0:
        raise ValueError("footprint_padding must be finite and at least 1.0")
    if not math.isfinite(float(tolerance_px)) or float(tolerance_px) <= 0.0:
        raise ValueError("footprint_tolerance_px must be finite and positive")
    if int(max_depth) < 1:
        raise ValueError("footprint_max_depth must be positive")
    proj_code = str(projection).strip().upper()
    target_frame, ctype1_prefix, ctype2_prefix, radesys, equinox = (
        _resolve_plan_frame_prefixes(frame)
    )
    is_default_tan = (
        proj_code == "TAN"
        and target_frame == "ICRS"
        and center == "union"
        and pixel_scale == "median"
        and pixel_scale_arcsec is None
        and shape is None
        and float(rotation_deg) == 0.0
    )

    footprints: list[Tensor] = []
    obs_centers: list[tuple[float, float]] = []
    pixel_scales: list[float] = []
    footprint_records: list[tuple[WCS, int, int]] = []
    for observation in observations:
        header = _observation_header_for_footprint(observation, device=device)
        wcs = WCS(header)
        height, width = _true_image_shape(header)
        pixel_edges = torch.tensor(
            [
                [-0.5, -0.5],
                [float(width) - 0.5, -0.5],
                [float(width) - 0.5, float(height) - 0.5],
                [-0.5, float(height) - 0.5],
            ],
            dtype=torch.float64,
        )
        footprint = wcs.pixel_to_world(pixel_edges)
        if isinstance(footprint, tuple):
            footprint = torch.stack(footprint[:2], dim=-1)
        footprint = footprint.detach().to(device="cpu", dtype=torch.float64)
        if footprint.shape != (4, 2) or not bool(torch.all(torch.isfinite(footprint))):
            raise RuntimeError(
                f"Could not determine finite footprint for {observation.path}"
            )
        if getattr(wcs, "celestial_frame", "ICRS") != target_frame or (
            equinox is not None and getattr(wcs, "celestial_equinox", None) != equinox
        ):
            lon_c, lat_c = wcs.convert_celestial(
                footprint[:, 0],
                footprint[:, 1],
                target_frame,
                to_equinox=equinox,
            )
            footprint = torch.stack((lon_c, lat_c), dim=-1)
        determinant = torch.linalg.det(
            wcs.cd.detach().to(device="cpu", dtype=torch.float64)
        )
        obs_scale = math.sqrt(abs(float(determinant)))
        if not math.isfinite(obs_scale) or obs_scale <= 0.0:
            raise RuntimeError(
                f"Could not determine a non-singular spatial pixel scale for {observation.path}"
            )
        footprints.append(footprint)
        obs_centers.append(_spherical_union_center(footprint))
        pixel_scales.append(obs_scale)
        footprint_records.append((wcs, height, width))
    if not footprints:
        raise RuntimeError("Could not determine any observation footprints")

    footprint = torch.cat(footprints, dim=0)
    if isinstance(center, tuple):
        center_ra = float(center[0]) % 360.0
        center_dec = float(center[1])
    elif str(center).strip().lower() == "most":
        best_idx = 0
        best_score = -1.0
        for idx_i, (ra_i, dec_i) in enumerate(obs_centers):
            score = 0.0
            for idx_j, (ra_j, dec_j) in enumerate(obs_centers):
                sep = _angular_separation_deg(ra_i, dec_i, ra_j, dec_j)
                score += 1.0 / (1.0 + (sep / max(pixel_scales[idx_j] * 512.0, 1e-3)) ** 2)
            if score > best_score:
                best_score = score
                best_idx = idx_i
        center_ra, center_dec = obs_centers[best_idx]
    elif str(center).strip().lower() == "union":
        center_ra, center_dec = _spherical_union_center(footprint)
    else:
        raise ValueError(
            f"Unsupported center={center!r}; expected 'union', 'most', or (lon_deg, lat_deg)"
        )

    if pixel_scale_arcsec is not None:
        resolved_scale = float(pixel_scale_arcsec) / 3600.0
    elif isinstance(pixel_scale, (int, float)):
        resolved_scale = float(pixel_scale)
    else:
        scale_mode = str(pixel_scale).strip().lower()
        if scale_mode == "median":
            resolved_scale = float(
                torch.tensor(pixel_scales, dtype=torch.float64).median()
            )
        elif scale_mode == "min":
            resolved_scale = float(min(pixel_scales))
        elif scale_mode == "max":
            resolved_scale = float(max(pixel_scales))
        else:
            raise ValueError(
                f"Unsupported pixel_scale={pixel_scale!r}; expected 'median', 'min', 'max', or a positive float"
            )
    if not math.isfinite(resolved_scale) or resolved_scale <= 0.0:
        raise ValueError("pixel_scale must be finite and positive")

    if is_default_tan:
        tangent_boundaries = [
            _adaptive_tangent_boundary(
                wcs,
                height=height,
                width=width,
                center_ra_deg=center_ra,
                center_dec_deg=center_dec,
                pixel_scale_deg=resolved_scale,
                tolerance_px=float(tolerance_px),
                max_depth=int(max_depth),
            )
            for wcs, height, width in footprint_records
        ]
        tangent_boundary = torch.cat(tangent_boundaries, dim=0)
        xi = tangent_boundary[:, 0]
        eta = tangent_boundary[:, 1]
        xi_mid = 0.5 * float(xi.min() + xi.max())
        eta_mid = 0.5 * float(eta.min() + eta.max())
        chord_margin_deg = float(tolerance_px) * resolved_scale
        xi_half = (0.5 * float(xi.max() - xi.min()) + chord_margin_deg) * float(padding)
        eta_half = (0.5 * float(eta.max() - eta.min()) + chord_margin_deg) * float(
            padding
        )
        xi_min, xi_max = xi_mid - xi_half, xi_mid + xi_half
        eta_min, eta_max = eta_mid - eta_half, eta_mid + eta_half
        naxis1 = max(1, int(math.ceil((xi_max - xi_min) / resolved_scale)))
        naxis2 = max(1, int(math.ceil((eta_max - eta_min) / resolved_scale)))
        header: dict = {
            "NAXIS": 2,
            "NAXIS1": naxis1,
            "NAXIS2": naxis2,
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
            "CUNIT1": "deg",
            "CUNIT2": "deg",
            "CRVAL1": center_ra,
            "CRVAL2": center_dec,
            "CRPIX1": 0.5 + xi_max / resolved_scale,
            "CRPIX2": 0.5 - eta_min / resolved_scale,
            "CD1_1": -resolved_scale,
            "CD1_2": 0.0,
            "CD2_1": 0.0,
            "CD2_2": resolved_scale,
        }
        return header, (naxis2, naxis1)

    rot_rad = math.radians(float(rotation_deg))
    cos_r = math.cos(rot_rad)
    sin_r = math.sin(rot_rad)
    cd1_1 = -resolved_scale * cos_r
    cd1_2 = -resolved_scale * sin_r
    cd2_1 = -resolved_scale * sin_r
    cd2_2 = resolved_scale * cos_r

    probe_header: dict[str, Any] = {
        "NAXIS": 2,
        "NAXIS1": 100,
        "NAXIS2": 100,
        "CTYPE1": f"{ctype1_prefix}-{proj_code}",
        "CTYPE2": f"{ctype2_prefix}-{proj_code}",
        "CUNIT1": "deg",
        "CUNIT2": "deg",
        "CRVAL1": center_ra,
        "CRVAL2": center_dec,
        "CRPIX1": 1.0,
        "CRPIX2": 1.0,
        "CD1_1": cd1_1,
        "CD1_2": cd1_2,
        "CD2_1": cd2_1,
        "CD2_2": cd2_2,
    }
    if radesys is not None:
        probe_header["RADESYS"] = radesys
    if equinox is not None:
        probe_header["Equinox"] = equinox
        probe_header["EQUINOX"] = equinox
        del probe_header["Equinox"]
    if proj_code in {"COE", "COO", "COD", "COP", "BON"}:
        probe_header["PV2_1"] = center_dec if abs(center_dec) >= 1.0 else 45.0
    elif proj_code == "HPX":
        probe_header["PV2_1"] = 4.0
        probe_header["PV2_2"] = 3.0
    probe_wcs = WCS(probe_header)

    if shape is not None:
        naxis2, naxis1 = int(shape[0]), int(shape[1])
        if naxis1 <= 0 or naxis2 <= 0:
            raise ValueError("shape dimensions must be positive")
        probe_header["NAXIS1"] = naxis1
        probe_header["NAXIS2"] = naxis2
        probe_header["CRPIX1"] = 0.5 * (naxis1 + 1)
        probe_header["CRPIX2"] = 0.5 * (naxis2 + 1)
        return probe_header, (naxis2, naxis1)

    boundary_world: list[Tensor] = []
    t_steps = torch.linspace(0.0, 1.0, 33, dtype=torch.float64)
    for wcs, height, width in footprint_records:
        w_lim = float(width) - 0.5
        h_lim = float(height) - 0.5
        e0 = torch.stack((-0.5 + (w_lim + 0.5) * t_steps, torch.full_like(t_steps, -0.5)), dim=-1)
        e1 = torch.stack((torch.full_like(t_steps, w_lim), -0.5 + (h_lim + 0.5) * t_steps), dim=-1)
        e2 = torch.stack((w_lim - (w_lim + 0.5) * t_steps, torch.full_like(t_steps, h_lim)), dim=-1)
        e3 = torch.stack((torch.full_like(t_steps, -0.5), h_lim - (h_lim + 0.5) * t_steps), dim=-1)
        edges_pix = torch.cat((e0, e1, e2, e3), dim=0)
        lon_e, lat_e = wcs.pixel_to_world(edges_pix[:, 0], edges_pix[:, 1])
        lon_e = lon_e.detach().to(device="cpu", dtype=torch.float64)
        lat_e = lat_e.detach().to(device="cpu", dtype=torch.float64)
        if getattr(wcs, "celestial_frame", "ICRS") != target_frame or (
            equinox is not None and getattr(wcs, "celestial_equinox", None) != equinox
        ):
            lon_e, lat_e = wcs.convert_celestial(
                lon_e, lat_e, target_frame, to_equinox=equinox
            )
        boundary_world.append(torch.stack((lon_e, lat_e), dim=-1))

    all_world = torch.cat(boundary_world, dim=0)
    px, py = probe_wcs.world_to_pixel(all_world[:, 0], all_world[:, 1])
    finite = torch.isfinite(px) & torch.isfinite(py)
    if not bool(finite.any()):
        raise RuntimeError(
            f"Observation footprints could not be projected into {proj_code}"
        )
    px = px[finite]
    py = py[finite]
    px_mid = 0.5 * float(px.min() + px.max())
    py_mid = 0.5 * float(py.min() + py.max())
    px_half = (0.5 * float(px.max() - px.min()) + float(tolerance_px)) * float(padding)
    py_half = (0.5 * float(py.max() - py.min()) + float(tolerance_px)) * float(padding)
    px_min, px_max = px_mid - px_half, px_mid + px_half
    py_min, py_max = py_mid - py_half, py_mid + py_half
    naxis1 = max(1, int(math.ceil(px_max - px_min)))
    naxis2 = max(1, int(math.ceil(py_max - py_min)))
    probe_header["NAXIS1"] = naxis1
    probe_header["NAXIS2"] = naxis2
    probe_header["CRPIX1"] = 1.0 - px_min + 0.5
    probe_header["CRPIX2"] = 1.0 - py_min + 0.5
    return probe_header, (naxis2, naxis1)


def plan_image_plane_target(
    observations: Sequence[ImageObservation],
    *,
    reference_index: int | None = 0,
    header_overrides: dict | None = None,
    device: torch.device | str | None = None,
    union_footprint: bool = False,
    footprint_padding: float = 1.05,
    footprint_tolerance_px: float = 0.25,
    footprint_max_depth: int = 20,
    max_footprint_pixels: int | None = None,
    projection: str = "TAN",
    frame: str = "icrs",
    center: str | tuple[float, float] = "union",
    pixel_scale: str | float = "median",
    pixel_scale_arcsec: float | None = None,
    shape: tuple[int, int] | None = None,
    rotation_deg: float = 0.0,
) -> PlannedImagePlaneTarget:
    if not observations:
        raise ValueError("At least one observation is required")
    if max_footprint_pixels is not None and int(max_footprint_pixels) <= 0:
        raise ValueError("max_footprint_pixels must be positive")

    if union_footprint or reference_index is None:
        header, planned_shape = _plan_union_footprint_target(
            observations,
            device=device,
            padding=footprint_padding,
            tolerance_px=footprint_tolerance_px,
            max_depth=footprint_max_depth,
            projection=projection,
            frame=frame,
            center=center,
            pixel_scale=pixel_scale,
            pixel_scale_arcsec=pixel_scale_arcsec,
            shape=shape,
            rotation_deg=rotation_deg,
        )
        if (
            max_footprint_pixels is not None
            and planned_shape[0] * planned_shape[1] > max_footprint_pixels
        ):
            raise RuntimeError(
                f"Union footprint output {planned_shape[0]}×{planned_shape[1]} "
                f"({planned_shape[0] * planned_shape[1] / 1e6:.1f} MPix) exceeds --max-footprint-pixels="
                f"{max_footprint_pixels}"
            )
        final_shape = target_shape_from_header(
            header_overrides or {},
            fallback=planned_shape,
        )
        return PlannedImagePlaneTarget(
            header=normalized_header_for_shape(
                header,
                final_shape,
                overrides=header_overrides,
            ),
            shape=final_shape,
        )

    if reference_index < 0 or reference_index >= len(observations):
        raise IndexError("reference_index is out of range")

    ref = observations[reference_index]
    read_dev = str(_resolve_map_device(device))
    image, header = load_observation_image_and_header(ref, device=read_dev)
    shape_out = tuple(image.shape) if shape is None else (int(shape[0]), int(shape[1]))
    shape_out = target_shape_from_header(header_overrides or {}, fallback=shape_out)
    normalized_header = normalized_header_for_shape(
        dict(header),
        shape_out,
        overrides=header_overrides,
    )
    return PlannedImagePlaneTarget(header=normalized_header, shape=shape_out)


def plan_image_plane_cube_target(
    observations: Sequence[ImageObservation],
    *,
    reference_index: int | None = 0,
    header_overrides: dict | None = None,
    spectral_axis: SpectralAxis | None = None,
    spectral_bin_edges: Tensor | None = None,
    spectral_unit: str = "Angstrom",
    spectral_type: str = "WAVE",
    device: torch.device | str | None = None,
    union_footprint: bool = False,
    footprint_padding: float = 1.05,
    footprint_tolerance_px: float = 0.25,
    footprint_max_depth: int = 20,
    max_footprint_pixels: int | None = None,
) -> PlannedImagePlaneCubeTarget:
    spatial_target = plan_image_plane_target(
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
    (
        observation_center,
        observation_lower,
        observation_upper,
        observation_filter_center,
        observation_filter_sigma,
        observation_names,
    ) = _observation_spectral_columns(observations)
    if spectral_axis is None:
        if spectral_bin_edges is not None:
            raise ValueError(
                "spectral_axis must be provided when spectral_bin_edges are specified"
            )
        spectral_axis = SpectralAxis(
            values=torch.unique(observation_center, sorted=True),
            unit=spectral_unit,
            type=spectral_type,
        )
    spectral_axis = _validated_spectral_axis(spectral_axis)
    validated_edges = None
    if spectral_bin_edges is not None:
        validated_edges = _validated_spectral_bin_edges(
            spectral_bin_edges,
            n_band=int(spectral_axis.values.numel()),
        )
    projection = _build_spectral_projection(
        observation_center,
        observation_lower,
        observation_upper,
        observation_filter_center,
        observation_filter_sigma,
        target_axis=spectral_axis,
        target_edges=validated_edges,
    )
    return PlannedImagePlaneCubeTarget(
        spatial_target=spatial_target,
        spectral_axis=spectral_axis,
        spectral_bin_edges=validated_edges,
        band_metadata=_build_band_metadata(
            spectral_axis,
            validated_edges,
            projection,
            observation_names,
        ),
    )


def plan_image_plane_cube_target_from_header(
    observations: Sequence[ImageObservation],
    spectral_header: dict,
    *,
    reference_index: int | None = 0,
    header_overrides: dict | None = None,
    use_header_bin_edges: bool = True,
    device: torch.device | str | None = None,
) -> PlannedImagePlaneCubeTarget:
    spectral_axis, _ = spectral_axis_from_header(spectral_header)
    spectral_bin_edges = None
    if use_header_bin_edges:
        spectral_bin_edges, _ = spectral_bin_edges_from_header(spectral_header)
    return plan_image_plane_cube_target(
        observations,
        reference_index=reference_index,
        header_overrides=header_overrides,
        spectral_axis=spectral_axis,
        spectral_bin_edges=spectral_bin_edges,
        device=device,
    )


def plan(
    observations: Sequence[ImageObservation],
    *,
    projection: str = "TAN",
    frame: str = "icrs",
    center: str | tuple[float, float] = "union",
    pixel_scale: str | float = "median",
    pixel_scale_arcsec: float | None = None,
    shape: tuple[int, int] | None = None,
    rotation_deg: float = 0.0,
    padding: float = 1.05,
    reference_index: int | None = None,
    header_overrides: dict | None = None,
    max_pixels: int | None = None,
    device: torch.device | str | None = None,
) -> PlannedImagePlaneTarget:
    """Plan a 2D celestial target grid from a set of observations."""
    return plan_image_plane_target(
        observations,
        reference_index=reference_index,
        header_overrides=header_overrides,
        device=device,
        union_footprint=(reference_index is None),
        footprint_padding=padding,
        max_footprint_pixels=max_pixels,
        projection=projection,
        frame=frame,
        center=center,
        pixel_scale=pixel_scale,
        pixel_scale_arcsec=pixel_scale_arcsec,
        shape=shape,
        rotation_deg=rotation_deg,
    )


def plan_cube(
    observations: Sequence[ImageObservation],
    *,
    spectral_axis: SpectralAxis | None = None,
    spectral_bin_edges: Tensor | None = None,
    spectral_header: dict | None = None,
    spectral_unit: str = "Angstrom",
    spectral_type: str = "WAVE",
    reference_index: int | None = None,
    header_overrides: dict | None = None,
    padding: float = 1.05,
    max_pixels: int | None = None,
    device: torch.device | str | None = None,
) -> PlannedImagePlaneCubeTarget:
    """Plan a 3D spatial-spectral target cube from a set of observations."""
    if spectral_header is not None:
        return plan_image_plane_cube_target_from_header(
            observations,
            spectral_header,
            reference_index=reference_index,
            header_overrides=header_overrides,
            device=device,
        )
    return plan_image_plane_cube_target(
        observations,
        reference_index=reference_index,
        header_overrides=header_overrides,
        spectral_axis=spectral_axis,
        spectral_bin_edges=spectral_bin_edges,
        spectral_unit=spectral_unit,
        spectral_type=spectral_type,
        device=device,
        union_footprint=(reference_index is None),
        footprint_padding=padding,
        max_footprint_pixels=max_pixels,
    )


def _obs_target_pixel_bbox(
    source_wcs: WCS,
    source_shape: tuple[int, int],
    full_target_wcs: WCS,
    *,
    margin: float = 8.0,
) -> tuple[float, float, float, float] | None:
    """Return (y_min, y_max, x_min, x_max) of source_wcs on full_target_wcs, or None if unbounded."""
    hs, ws = int(source_shape[0]), int(source_shape[1])
    if hs <= 0 or ws <= 0:
        return (0.0, -1.0, 0.0, -1.0)
    try:
        t = torch.linspace(0.0, 1.0, 9, dtype=source_wcs.crval.dtype, device=source_wcs.device)
        x0, x1 = -0.5, float(ws) - 0.5
        y0, y1 = -0.5, float(hs) - 0.5
        xs = torch.cat([x0 + t * (x1 - x0), torch.full_like(t, x1), x1 - t * (x1 - x0), torch.full_like(t, x0)])
        ys = torch.cat([torch.full_like(t, y0), y0 + t * (y1 - y0), torch.full_like(t, y1), y1 - t * (y1 - y0)])
        ra, dec = source_wcs.pixel_to_world(xs, ys)
        ra, dec = _convert_world_between_wcs(ra, dec, source_wcs, full_target_wcs)
        tx, ty = full_target_wcs.world_to_pixel(ra, dec)
        if not bool((torch.isfinite(tx) & torch.isfinite(ty)).all()):
            return None
        return (
            float(ty.min().item()) - margin,
            float(ty.max().item()) + margin,
            float(tx.min().item()) - margin,
            float(tx.max().item()) + margin,
        )
    except Exception:
        return None


def _coadd_image_plane_spatial_tiles(
    observations: Sequence[ImageObservation],
    *,
    first_image: Tensor,
    first_header: dict,
    header_dict: dict,
    target_shape: tuple[int, int],
    tile_height: int,
    tile_width: int,
    output_path: str | None,
    resampling_kernel: MapResamplingKernel,
    value_semantics: MapValueSemantics,
    stacking_method: MapStackingMethod,
    drizzle_pixfrac: float,
    clip_sigma: float,
    clip_iterations: int,
    map_device: torch.device,
    scratch_dir: str | Path | None = None,
    background_estimator: Callable[[Tensor, Tensor | None], Tensor] | None = None,
    cross_exposure_sky_corrector: CrossExposureSkyCorrector | None = None,
) -> MapResult:
    """Accumulate a large map with tile-bounded projection intermediates."""
    if not observations:
        raise ValueError("At least one observation is required")
    H, W = int(target_shape[0]), int(target_shape[1])
    th = max(int(tile_height), 1)
    tw = max(int(tile_width), 1)
    device = map_device
    full_target_wcs = WCS(header_dict)
    full_context = torch.zeros(
        (math.ceil(len(observations) / 32), H, W),
        dtype=torch.int64,
        device=device,
    )
    mosaic_surface_brightness_area_scaling: bool | None = None
    if value_semantics == "surface_brightness" and stacking_method == "weighted_mean":
        mosaic_surface_brightness_area_scaling = (
            _should_apply_surface_brightness_area_scaling(
                observations,
                target_header=header_dict,
                target_shape=target_shape,
                first_header=dict(first_header),
            )
        )
    mosaic_representative_area_sr = (
        representative_pixel_area_sr(
            full_target_wcs, target_shape, device=device
        )
        if mosaic_surface_brightness_area_scaling
        else None
    )

    if cross_exposure_sky_corrector is not None and len(observations) == 1:
        cross_exposure_sky_corrector = None

    if cross_exposure_sky_corrector is not None:
        read_dev = str(device)
        full_pixel_area_sr = None
        if value_semantics == "flux":
            full_pixel_area_sr = pixel_area_sr_map(
                full_target_wcs, target_shape, device=device
            )

        def overlap_contributions():
            for obs_idx, obs in enumerate(observations):
                if obs_idx == 0:
                    image = validate_image_plane_2d(first_image).to(device=device)
                    header_dict_obs = dict(first_header)
                else:
                    image, header_dict_obs = load_observation_image_and_header(
                        obs, device=read_dev
                    )
                    image = image.to(device=device)
                prepared_observation = prepare_image_observation_data(
                    obs,
                    image,
                    device=device,
                    resampling_kernel=resampling_kernel,
                    load_weight_map_fn=_load_observation_weight_map,
                    apply_calibration_fn=_apply_observation_calibration,
                    background_estimator=background_estimator,
                )
                source_wcs = WCS(header_dict_obs)
                obs_bbox = _obs_target_pixel_bbox(
                    source_wcs, (int(image.shape[0]), int(image.shape[1])), full_target_wcs
                )
                for y0 in range(0, H, th):
                    for x0 in range(0, W, tw):
                        y1, x1 = min(y0 + th, H), min(x0 + tw, W)
                        if obs_bbox is not None and (
                            y1 <= obs_bbox[0]
                            or y0 >= obs_bbox[1]
                            or x1 <= obs_bbox[2]
                            or x0 >= obs_bbox[3]
                        ):
                            continue
                        tile_shape = (y1 - y0, x1 - x0)
                        sub_header, _ = _subtarget_header_for_window(
                            header_dict,
                            full_shape=(H, W),
                            y0=y0,
                            y1=y1,
                            x0=x0,
                            x1=x1,
                        )
                        need_prepared_target = (
                            value_semantics == "flux"
                            or header_dict_obs != sub_header
                            or tuple(image.shape) != tile_shape
                        )
                        prepared_target = None
                        if need_prepared_target:
                            prepared_target = _prepare_target(
                                header=sub_header,
                                target_wcs=WCS(sub_header),
                                target_shape=tile_shape,
                                device=device,
                                canvas_shape=(H, W),
                                canvas_origin_yx=(y0, x0),
                                include_pixel_area=(value_semantics == "flux"),
                                include_representative_pixel_area=bool(
                                    mosaic_surface_brightness_area_scaling
                                ),
                                include_quadrature=(value_semantics == "flux"),
                                include_world_grid=False,
                            )
                            prepared_target.representative_pixel_area_sr = (
                                mosaic_representative_area_sr
                            )
                        projected = _project_observation_to_target_chunk(
                            obs,
                            image=image,
                            header_dict_obs=header_dict_obs,
                            source_wcs=source_wcs,
                            prepared_observation=prepared_observation,
                            prepared_target=prepared_target,
                            target_header=sub_header,
                            target_shape=tile_shape,
                            device=device,
                            footprint_order=None,
                            resampling_kernel=resampling_kernel,
                            value_semantics=value_semantics,
                            drizzle_pixfrac=drizzle_pixfrac,
                            background_estimator=None,
                        )
                        if projected.target_window is None:
                            global_window = (y0, y1, x0, x1)
                            pixel_area = (
                                None
                                if prepared_target is None
                                else prepared_target.pixel_area_sr
                            )
                        else:
                            py0, py1, px0, px1 = projected.target_window
                            global_window = (
                                y0 + py0,
                                y0 + py1,
                                x0 + px0,
                                x0 + px1,
                            )
                            pixel_area = (
                                None
                                if prepared_target is None
                                or prepared_target.pixel_area_sr is None
                                else prepared_target.pixel_area_sr[py0:py1, px0:px1]
                            )
                        projected_numer = projected.numer
                        projected_weight = projected.weight
                        projected_variance = projected.variance_weight
                        if stacking_method == "median":
                            projected_numer, projected_weight = (
                                stack_inputs_from_projected_observation(
                                    projected.numer,
                                    projected.weight,
                                    stacking_method="median",
                                    value_semantics=value_semantics,
                                    pixel_area_sr=pixel_area,
                                )
                            )
                            projected_variance = None
                        yield (
                            obs_idx,
                            projected_numer,
                            projected_weight,
                            projected_variance,
                            projected.overlap,
                            global_window,
                        )
                del prepared_observation, image

        spilled = spilled_overlap_background_stack(
            overlap_contributions(),
            n_observations=len(observations),
            shape=target_shape,
            stacking_method=stacking_method,
            value_semantics=value_semantics,
            corrector=cross_exposure_sky_corrector,
            scratch_dir=scratch_dir,
            output_device=device,
            tile_hw=(th, tw),
            pixel_area_sr=full_pixel_area_sr,
        )
        full_data = spilled.data
        full_weight = spilled.weight
        full_overlap = spilled.overlap_count
        full_context = spilled.context
        full_inverse_variance = (
            _linear_inverse_variance(
                spilled.normalization,
                spilled.variance_weight,
                value_semantics=value_semantics,
                pixel_area_sr=full_pixel_area_sr,
            )
            if spilled.variance_weight is not None
            else None
        )
        full_correlation_factor = (
            _local_noise_correlation_factor(
                spilled.normalization,
                spilled.variance_weight,
            )
            if spilled.variance_weight is not None
            else None
        )
    elif stacking_method in ("median", "sigma_clipped"):
        if stacking_method == "median":
            require_exact_median_scratch_capacity(
                len(observations),
                (min(th, H), min(tw, W)),
                scratch_dir=scratch_dir,
            )
        else:
            require_sigma_clipped_scratch_capacity(
                len(observations),
                (min(th, H), min(tw, W)),
                scratch_dir=scratch_dir,
            )
        full_data = torch.zeros((H, W), dtype=torch.float32, device=device)
        full_weight = torch.zeros((H, W), dtype=torch.float32, device=device)
        full_inverse_variance: Tensor | None = None
        full_correlation_factor: Tensor | None = None
        full_overlap = torch.zeros((H, W), dtype=torch.int32, device=device)
        for y0 in range(0, H, th):
            for x0 in range(0, W, tw):
                y1, x1 = min(y0 + th, H), min(x0 + tw, W)
                sub_header, _ = _subtarget_header_for_window(
                    header_dict,
                    full_shape=(H, W),
                    y0=y0,
                    y1=y1,
                    x0=x0,
                    x1=x1,
                )
                sub = coadd_image_plane(
                    observations,
                    target_header=sub_header,
                    output_path=None,
                    footprint_order=None,
                    resampling_kernel=resampling_kernel,
                    value_semantics=value_semantics,
                    stacking_method=stacking_method,
                    drizzle_pixfrac=drizzle_pixfrac,
                    clip_sigma=clip_sigma,
                    clip_iterations=clip_iterations,
                    spatial_tile_hw=None,
                    surface_brightness_area_scaling=(
                        mosaic_surface_brightness_area_scaling
                    ),
                    device=map_device,
                    scratch_dir=scratch_dir,
                    background_estimator=background_estimator,
                    cross_exposure_sky_corrector=None,
                )
                full_data[y0:y1, x0:x1] = sub.data.to(device=device)
                full_weight[y0:y1, x0:x1] = sub.weight.to(device=device)
                if sub.inverse_variance is not None:
                    if full_inverse_variance is None:
                        full_inverse_variance = torch.zeros_like(full_weight)
                    full_inverse_variance[y0:y1, x0:x1] = sub.inverse_variance.to(
                        device=device
                    )
                if sub.correlation_factor is not None:
                    if full_correlation_factor is None:
                        full_correlation_factor = torch.zeros_like(full_weight)
                    full_correlation_factor[y0:y1, x0:x1] = (
                        sub.correlation_factor.to(device=device)
                    )
                if sub.overlap_count is not None:
                    full_overlap[y0:y1, x0:x1] = sub.overlap_count.to(device=device)
                if sub.context is not None:
                    full_context[:, y0:y1, x0:x1] = sub.context.to(device=device)
    else:
        numer = torch.zeros((H, W), dtype=torch.float32, device=device)
        denom = torch.zeros((H, W), dtype=torch.float32, device=device)
        variance_weight = torch.zeros_like(denom)
        variance_available = stacking_method == "weighted_mean"
        full_overlap = torch.zeros((H, W), dtype=torch.int32, device=device)
        read_dev = str(device)
        for obs_idx, obs in enumerate(observations):
            if obs_idx == 0:
                image = validate_image_plane_2d(first_image).to(device=device)
                header_dict_obs = dict(first_header)
            else:
                image, header_dict_obs = load_observation_image_and_header(
                    obs, device=read_dev
                )
                image = image.to(device=device)
            prepared_observation = prepare_image_observation_data(
                obs,
                image,
                device=device,
                resampling_kernel=resampling_kernel,
                load_weight_map_fn=_load_observation_weight_map,
                apply_calibration_fn=_apply_observation_calibration,
                background_estimator=background_estimator,
            )
            source_wcs = WCS(header_dict_obs)
            obs_bbox = _obs_target_pixel_bbox(
                source_wcs, (int(image.shape[0]), int(image.shape[1])), full_target_wcs
            )
            for y0 in range(0, H, th):
                for x0 in range(0, W, tw):
                    y1, x1 = min(y0 + th, H), min(x0 + tw, W)
                    if obs_bbox is not None and (
                        y1 <= obs_bbox[0]
                        or y0 >= obs_bbox[1]
                        or x1 <= obs_bbox[2]
                        or x0 >= obs_bbox[3]
                    ):
                        continue
                    tile_shape = (y1 - y0, x1 - x0)
                    sub_header, _ = _subtarget_header_for_window(
                        header_dict,
                        full_shape=(H, W),
                        y0=y0,
                        y1=y1,
                        x0=x0,
                        x1=x1,
                    )
                    need_prepared_target = (
                        value_semantics == "flux"
                        or header_dict_obs != sub_header
                        or tuple(image.shape) != tile_shape
                    )
                    prepared_target = None
                    if need_prepared_target:
                        prepared_target = _prepare_target(
                            header=sub_header,
                            target_wcs=WCS(sub_header),
                            target_shape=tile_shape,
                            device=device,
                            canvas_shape=(H, W),
                            canvas_origin_yx=(y0, x0),
                            include_pixel_area=(value_semantics == "flux"),
                            include_representative_pixel_area=bool(
                                mosaic_surface_brightness_area_scaling
                            ),
                            include_quadrature=(value_semantics == "flux"),
                            include_world_grid=False,
                        )
                        prepared_target.representative_pixel_area_sr = (
                            mosaic_representative_area_sr
                        )
                    projected = _project_observation_to_target_chunk(
                        obs,
                        image=image,
                        header_dict_obs=header_dict_obs,
                        source_wcs=source_wcs,
                        prepared_observation=prepared_observation,
                        prepared_target=prepared_target,
                        target_header=sub_header,
                        target_shape=tile_shape,
                        device=device,
                        footprint_order=None,
                        resampling_kernel=resampling_kernel,
                        value_semantics=value_semantics,
                        drizzle_pixfrac=drizzle_pixfrac,
                        background_estimator=None,
                    )
                    pixel_area_sr = None
                    if value_semantics == "flux":
                        assert prepared_target is not None
                        pixel_area_sr = prepared_target.pixel_area_sr
                        if pixel_area_sr is None:
                            raise RuntimeError(
                                "target pixel area is required for flux semantics"
                            )
                        if projected.target_window is not None:
                            py0, py1, px0, px1 = projected.target_window
                            pixel_area_sr = pixel_area_sr[py0:py1, px0:px1]
                    accumulate_projected_observation(
                        numer[y0:y1, x0:x1],
                        denom[y0:y1, x0:x1],
                        full_overlap[y0:y1, x0:x1],
                        projected_numer=projected.numer,
                        projected_weight=projected.weight,
                        projected_overlap=projected.overlap,
                        stacking_method=stacking_method,
                        value_semantics=value_semantics,
                        pixel_area_sr=pixel_area_sr,
                        target_window=projected.target_window,
                    )
                    if projected.variance_weight is None:
                        variance_available = False
                    elif variance_available:
                        _accumulate_variance_weight(
                            variance_weight[y0:y1, x0:x1],
                            projected.variance_weight,
                            projected.target_window,
                        )
                    _set_context_bit(
                        full_context[:, y0:y1, x0:x1],
                        obs_idx,
                        projected.overlap > 0,
                        projected.target_window,
                    )
            del prepared_observation, image

        full_pixel_area_sr = None
        if value_semantics == "flux":
            full_pixel_area_sr = pixel_area_sr_map(
                WCS(header_dict), target_shape, device=device
            )
        normalization = denom
        full_data, full_weight = finalize_projected_stack(
            numer,
            denom,
            stacking_method=stacking_method,
            value_semantics=value_semantics,
            pixel_area_sr=full_pixel_area_sr,
        )
        full_inverse_variance = (
            _linear_inverse_variance(
                normalization,
                variance_weight,
                value_semantics=value_semantics,
                pixel_area_sr=full_pixel_area_sr,
            )
            if variance_available
            else None
        )
        full_correlation_factor = (
            _local_noise_correlation_factor(normalization, variance_weight)
            if variance_available
            else None
        )
    result = MapResult(
        data=full_data,
        weight=full_weight,
        header=header_dict,
        inverse_variance=full_inverse_variance,
        correlation_factor=full_correlation_factor,
        context=full_context,
        provenance=_context_provenance(
            observations,
            resampling_kernel=resampling_kernel,
            value_semantics=value_semantics,
            stacking_method=stacking_method,
            drizzle_pixfrac=drizzle_pixfrac,
            surface_brightness_area_scaling=bool(
                mosaic_surface_brightness_area_scaling
            ),
            data_dependent_background=(
                background_estimator is not None
                or cross_exposure_sky_corrector is not None
            ),
            has_correlation_factor=full_correlation_factor is not None,
        ),
        overlap_count=full_overlap,
        coverage_moc=None,
        observation_mocs=None,
    )
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        write(output_path, full_data, header=header_dict, overwrite=True)
    return result


def coadd_image_plane(
    observations: Sequence[ImageObservation],
    *,
    target_header: dict | None = None,
    output_path: str | None = None,
    footprint_order: int | None = None,
    resampling_kernel: MapResamplingKernel = "bilinear",
    value_semantics: MapValueSemantics = "surface_brightness",
    stacking_method: MapStackingMethod = "weighted_mean",
    drizzle_pixfrac: float = 1.0,
    clip_sigma: float = 4.0,
    clip_iterations: int = 3,
    spatial_tile_hw: tuple[int, int] | None = None,
    surface_brightness_area_scaling: bool | None = None,
    device: torch.device | str | None = None,
    scratch_dir: str | Path | None = None,
    background_estimator: Callable[[Tensor, Tensor | None], Tensor] | None = None,
    cross_exposure_sky_corrector: CrossExposureSkyCorrector | None = None,
) -> MapResult:
    if not observations:
        raise ValueError("At least one observation is required")
    if footprint_order is not None and footprint_order < 0:
        raise ValueError("footprint_order must be non-negative")
    resampling_kernel = _validated_map_resampling_kernel(resampling_kernel)
    value_semantics = _validated_map_value_semantics(value_semantics)
    stacking_method = _validated_map_stacking_method(stacking_method)
    drizzle_pixfrac = _validated_drizzle_pixfrac(drizzle_pixfrac)
    clip_sigma = float(clip_sigma)
    clip_iterations = int(clip_iterations)
    if not math.isfinite(clip_sigma) or clip_sigma <= 0.0:
        raise ValueError("clip_sigma must be finite and positive")
    if clip_iterations <= 0:
        raise ValueError("clip_iterations must be positive")
    if stacking_method == "sigma_clipped" and cross_exposure_sky_corrector is not None:
        raise NotImplementedError(
            "sigma_clipped stacking with overlap background rectification is not "
            "yet supported; use background_mode='per-exposure'"
        )

    map_device = _resolve_map_device(device)
    read_dev = str(map_device)
    first = observations[0]
    first_image, first_header = load_observation_image_and_header(
        first, device=read_dev
    )
    if target_header is not None and "CRVAL1" in target_header:
        target_shape = target_shape_from_header(
            target_header,
            fallback=tuple(first_image.shape[-2:]),
        )
        header_dict = normalized_header_for_shape(
            dict(target_header),
            target_shape,
            overrides=None,
        )
    else:
        header_dict = normalized_header_for_shape(
            dict(first_header),
            tuple(first_image.shape[-2:]),
            overrides=target_header,
        )
        target_shape = target_shape_from_header(
            header_dict,
            fallback=tuple(first_image.shape[-2:]),
        )
    device = map_device
    if (
        spatial_tile_hw is None
        and footprint_order is None
        and cross_exposure_sky_corrector is not None
        and len(observations) > 1
    ):
        # ponytail: one internal tile policy keeps overlap execution bounded;
        # expose tuning only if scorecards show the 1024-pixel default is poor.
        spatial_tile_hw = (
            min(1024, int(target_shape[0])),
            min(1024, int(target_shape[1])),
        )
    if spatial_tile_hw is not None:
        if footprint_order is not None:
            raise ValueError(
                "spatial_tile_hw is incompatible with footprint_order=... "
                "(MOC accumulation is not defined per tile); omit one of them"
            )
        return _coadd_image_plane_spatial_tiles(
            observations,
            first_image=first_image,
            first_header=first_header,
            header_dict=header_dict,
            target_shape=target_shape,
            tile_height=int(spatial_tile_hw[0]),
            tile_width=int(spatial_tile_hw[1]),
            output_path=output_path,
            resampling_kernel=resampling_kernel,
            value_semantics=value_semantics,
            stacking_method=stacking_method,
            drizzle_pixfrac=drizzle_pixfrac,
            clip_sigma=clip_sigma,
            clip_iterations=clip_iterations,
            map_device=map_device,
            scratch_dir=scratch_dir,
            background_estimator=background_estimator,
            cross_exposure_sky_corrector=cross_exposure_sky_corrector,
        )
    if not (
        value_semantics == "surface_brightness" and stacking_method == "weighted_mean"
    ):
        apply_surface_brightness_area_scaling = False
    elif surface_brightness_area_scaling is not None:
        apply_surface_brightness_area_scaling = surface_brightness_area_scaling
    else:
        apply_surface_brightness_area_scaling = (
            _should_apply_surface_brightness_area_scaling(
                observations,
                target_header=header_dict,
                target_shape=target_shape,
                first_header=dict(first_header),
            )
        )
    prepared_target: _PreparedImagePlaneTarget | None = None

    def ensure_prepared_target() -> _PreparedImagePlaneTarget:
        nonlocal prepared_target
        if prepared_target is None:
            prepared_target = _prepare_target(
                header=header_dict,
                target_wcs=WCS(header_dict),
                target_shape=target_shape,
                device=device,
                include_pixel_area=(value_semantics == "flux"),
                include_representative_pixel_area=apply_surface_brightness_area_scaling,
                include_quadrature=(value_semantics == "flux"),
                include_world_grid=False,
            )
        return prepared_target

    numer = torch.zeros(target_shape, dtype=torch.float32, device=device)
    denom = torch.zeros(target_shape, dtype=torch.float32, device=device)
    variance_weight = torch.zeros_like(denom)
    variance_available = stacking_method == "weighted_mean"
    overlap_count = torch.zeros(target_shape, dtype=torch.int32, device=device)
    context = torch.zeros(
        (math.ceil(len(observations) / 32), *target_shape),
        dtype=torch.int64,
        device=device,
    )
    observation_mocs: list[MOC] | None = [] if footprint_order is not None else None
    use_spilled_median = (
        stacking_method == "median" and cross_exposure_sky_corrector is None
    )
    use_spilled_sigma = stacking_method == "sigma_clipped"
    median_images: list[Tensor] | None = (
        [] if stacking_method == "median" and not use_spilled_median else None
    )
    median_support: list[Tensor] | None = (
        [] if stacking_method == "median" and not use_spilled_median else None
    )
    projected_store: list[dict] | None = (
        []
        if cross_exposure_sky_corrector is not None and stacking_method != "median"
        else None
    )

    def project_median_observation(
        obs_idx: int,
        obs: ImageObservation,
    ) -> tuple[Tensor, Tensor, Tensor, MOC | None]:
        if obs_idx == 0:
            image = first_image
            header_dict_obs = dict(first_header)
        else:
            image, header_dict_obs = load_observation_image_and_header(
                obs, device=read_dev
            )
        need_prepared_target = (
            value_semantics == "flux"
            or footprint_order is not None
            or header_dict_obs != header_dict
            or tuple(image.shape) != target_shape
        )
        source_wcs = None
        if need_prepared_target and not (
            headers_equivalent_for_direct_copy(header_dict_obs, header_dict)
            and tuple(image.shape) == target_shape
        ):
            source_wcs = WCS(header_dict_obs)
        sampled_numer, sampled_weight, overlap, moc = _project_observation_to_target(
            obs,
            image=image,
            header_dict_obs=header_dict_obs,
            source_wcs=source_wcs,
            prepared_target=ensure_prepared_target() if need_prepared_target else None,
            target_header=header_dict,
            target_shape=target_shape,
            device=device,
            footprint_order=footprint_order,
            resampling_kernel=resampling_kernel,
            value_semantics=value_semantics,
            drizzle_pixfrac=drizzle_pixfrac,
            background_estimator=background_estimator,
        )
        contrib_numer, contrib_weight = stack_inputs_from_projected_observation(
            sampled_numer,
            sampled_weight,
            stacking_method="mean",
            value_semantics=value_semantics,
            pixel_area_sr=(
                ensure_prepared_target().pixel_area_sr
                if value_semantics == "flux"
                else None
            ),
        )
        return contrib_numer, contrib_weight > 0.0, overlap, moc

    def spilled_contributions():
        for obs_idx, obs in enumerate(observations):
            contrib, support, overlap, moc = project_median_observation(obs_idx, obs)
            overlap_count.add_(overlap)
            _set_context_bit(context, obs_idx, overlap > 0, None)
            if observation_mocs is not None and moc is not None:
                observation_mocs.append(moc)
            yield contrib, support

    def project_sigma_observation(
        obs_idx: int,
        obs: ImageObservation,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor | None,
        Tensor,
        MOC | None,
        tuple[int, int, int, int] | None,
    ]:
        if obs_idx == 0:
            image = first_image
            header_dict_obs = dict(first_header)
        else:
            image, header_dict_obs = load_observation_image_and_header(
                obs, device=read_dev
            )
        need_prepared_target = (
            value_semantics == "flux"
            or footprint_order is not None
            or header_dict_obs != header_dict
            or tuple(image.shape) != target_shape
        )
        source_wcs = None
        if need_prepared_target and not (
            headers_equivalent_for_direct_copy(header_dict_obs, header_dict)
            and tuple(image.shape) == target_shape
        ):
            source_wcs = WCS(header_dict_obs)
        projected = _project_observation_to_target_chunk(
            obs,
            image=image,
            header_dict_obs=header_dict_obs,
            source_wcs=source_wcs,
            prepared_target=ensure_prepared_target() if need_prepared_target else None,
            target_header=header_dict,
            target_shape=target_shape,
            device=device,
            footprint_order=footprint_order,
            resampling_kernel=resampling_kernel,
            value_semantics=value_semantics,
            drizzle_pixfrac=drizzle_pixfrac,
            background_estimator=background_estimator,
        )
        values = torch.where(
            projected.weight > 0.0,
            projected.numer / projected.weight,
            torch.zeros_like(projected.numer),
        )
        clip_weight = projected.weight
        clip_variance = projected.variance_weight
        if value_semantics == "flux":
            prepared = ensure_prepared_target()
            if prepared.pixel_area_sr is None:
                raise RuntimeError("target pixel area is required for flux semantics")
            pixel_area = prepared.pixel_area_sr
            if projected.target_window is not None:
                y0, y1, x0, x1 = projected.target_window
                pixel_area = pixel_area[y0:y1, x0:x1]
            pixel_area_squared = torch.square(pixel_area)
            safe_pixel_area_squared = torch.clamp(
                pixel_area_squared,
                min=torch.finfo(pixel_area_squared.dtype).tiny,
            )
            values = pixel_area * values
            clip_weight = clip_weight / safe_pixel_area_squared
            if clip_variance is not None:
                clip_variance = clip_variance / safe_pixel_area_squared
        return (
            values,
            clip_weight,
            clip_variance,
            projected.overlap,
            projected.moc,
            projected.target_window,
        )

    def sigma_clipped_contributions():
        for obs_idx, obs in enumerate(observations):
            values, weights, variance, _overlap, moc, target_window = (
                project_sigma_observation(obs_idx, obs)
            )
            if observation_mocs is not None and moc is not None:
                observation_mocs.append(moc)
            yield values, weights, variance, target_window

    spilled_median_result = None
    if use_spilled_median:
        spilled_median_result = spilled_exact_median_stack(
            spilled_contributions(),
            n_observations=len(observations),
            shape=target_shape,
            scratch_dir=scratch_dir,
            output_device=device,
        )
    spilled_sigma_result = None
    if use_spilled_sigma:
        spilled_sigma_result = spilled_sigma_clipped_stack(
            sigma_clipped_contributions(),
            n_observations=len(observations),
            shape=target_shape,
            clip_sigma=clip_sigma,
            clip_iterations=clip_iterations,
            scratch_dir=scratch_dir,
            output_device=device,
        )

    observation_loop = (
        () if use_spilled_median or use_spilled_sigma else enumerate(observations)
    )
    for obs_idx, obs in observation_loop:
        if stacking_method == "median":
            contrib_numer, support, overlap, moc = project_median_observation(
                obs_idx, obs
            )
            assert median_images is not None and median_support is not None
            median_images.append(contrib_numer)
            median_support.append(support)
            overlap_count = overlap_count + overlap
            _set_context_bit(context, obs_idx, overlap > 0, None)
            if observation_mocs is not None and moc is not None:
                observation_mocs.append(moc)
        else:
            if obs_idx == 0:
                image = first_image
                header_dict_obs = dict(first_header)
            else:
                image, header_dict_obs = load_observation_image_and_header(
                    obs, device=read_dev
                )
            need_prepared_target = (
                value_semantics == "flux"
                or footprint_order is not None
                or header_dict_obs != header_dict
                or tuple(image.shape) != target_shape
            )
            source_wcs = None
            if need_prepared_target and not (
                headers_equivalent_for_direct_copy(header_dict_obs, header_dict)
                and tuple(image.shape) == target_shape
            ):
                source_wcs = WCS(header_dict_obs)
            projected = _project_observation_to_target_chunk(
                obs,
                image=image,
                header_dict_obs=header_dict_obs,
                source_wcs=source_wcs,
                prepared_target=ensure_prepared_target()
                if need_prepared_target
                else None,
                target_header=header_dict,
                target_shape=target_shape,
                device=device,
                footprint_order=footprint_order,
                resampling_kernel=resampling_kernel,
                value_semantics=value_semantics,
                drizzle_pixfrac=drizzle_pixfrac,
                background_estimator=background_estimator,
            )
            pixel_area_sr = None
            if value_semantics == "flux":
                prepared = ensure_prepared_target()
                if prepared.pixel_area_sr is None:
                    raise RuntimeError(
                        "target pixel area is required for flux semantics"
                    )
                pixel_area_sr = prepared.pixel_area_sr
                if projected.target_window is not None:
                    y0, y1, x0, x1 = projected.target_window
                    pixel_area_sr = pixel_area_sr[y0:y1, x0:x1]
            accumulate_projected_observation(
                numer,
                denom,
                overlap_count,
                projected_numer=projected.numer,
                projected_weight=projected.weight,
                projected_overlap=projected.overlap,
                stacking_method=stacking_method,
                value_semantics=value_semantics,
                pixel_area_sr=pixel_area_sr,
                target_window=projected.target_window,
            )
            if projected.variance_weight is None:
                variance_available = False
            elif variance_available:
                _accumulate_variance_weight(
                    variance_weight,
                    projected.variance_weight,
                    projected.target_window,
                )
            _set_context_bit(
                context,
                obs_idx,
                projected.overlap > 0,
                projected.target_window,
            )
            if projected_store is not None:
                projected_store.append(
                    {
                        "numer": projected.numer,
                        "weight": projected.weight,
                        "overlap": projected.overlap,
                        "target_window": projected.target_window,
                    }
                )
            if observation_mocs is not None and projected.moc is not None:
                observation_mocs.append(projected.moc)

    if cross_exposure_sky_corrector is not None:
        if stacking_method == "median":
            assert median_images is not None and median_support is not None
            corrector_masks = [~(s > 0.0) for s in median_support]
            corrected = cross_exposure_sky_corrector.correct(
                median_images, corrector_masks
            )
            median_images.clear()
            median_images.extend(corrected)
        else:
            assert projected_store is not None
            images = []
            masks = []
            for ps in projected_store:
                n = ps["numer"]
                w = ps["weight"]
                tw = ps["target_window"]
                img = torch.where(w > 0.0, n / w, torch.zeros_like(n))
                if tw is None:
                    images.append(img)
                    masks.append(~(w > 0.0))
                else:
                    y0, y1, x0, x1 = tw
                    full_img = torch.zeros(
                        target_shape, dtype=img.dtype, device=img.device
                    )
                    full_valid = torch.zeros(
                        target_shape, dtype=torch.bool, device=img.device
                    )
                    if y1 > y0 and x1 > x0:
                        full_img[y0:y1, x0:x1] = img
                        full_valid[y0:y1, x0:x1] = w > 0.0
                    images.append(full_img)
                    masks.append(~full_valid)
            corrected = cross_exposure_sky_corrector.correct(images, masks)
            numer.zero_()
            denom.zero_()
            overlap_count.zero_()
            corr_pixel_area_sr = None
            if value_semantics == "flux":
                prepared = ensure_prepared_target()
                if prepared.pixel_area_sr is None:
                    raise RuntimeError("target pixel area is required for flux semantics")
                corr_pixel_area_sr = prepared.pixel_area_sr
            for i, ps in enumerate(projected_store):
                w = ps["weight"]
                overlap_contrib = ps["overlap"]
                tw = ps["target_window"]
                pa_slice = corr_pixel_area_sr
                if tw is None:
                    corr_numer = corrected[i] * w
                else:
                    y0, y1, x0, x1 = tw
                    corr_numer = corrected[i][y0:y1, x0:x1] * w
                    if pa_slice is not None:
                        pa_slice = pa_slice[y0:y1, x0:x1]
                accumulate_projected_observation(
                    numer,
                    denom,
                    overlap_count,
                    projected_numer=corr_numer,
                    projected_weight=w,
                    projected_overlap=overlap_contrib,
                    stacking_method=stacking_method,
                    value_semantics=value_semantics,
                    pixel_area_sr=pa_slice,
                    target_window=tw,
                )

    if stacking_method == "median":
        if spilled_median_result is not None:
            coadd, denom = spilled_median_result
        else:
            assert median_images is not None and median_support is not None
            coadd, denom = masked_median_stack(median_images, median_support)
    elif spilled_sigma_result is not None:
        coadd = spilled_sigma_result.data
        denom = spilled_sigma_result.weight
        normalization = denom
        variance_available = spilled_sigma_result.variance_weight is not None
        variance_weight = (
            torch.zeros_like(denom)
            if spilled_sigma_result.variance_weight is None
            else spilled_sigma_result.variance_weight
        )
        overlap_count = spilled_sigma_result.overlap_count
        context = spilled_sigma_result.context
        pixel_area_sr = None
    else:
        pixel_area_sr = None
        if value_semantics == "flux":
            prepared = ensure_prepared_target()
            if prepared.pixel_area_sr is None:
                raise RuntimeError("target pixel area is required for flux semantics")
            pixel_area_sr = prepared.pixel_area_sr
        normalization = denom
        coadd, denom = finalize_projected_stack(
            numer,
            denom,
            stacking_method=stacking_method,
            value_semantics=value_semantics,
            pixel_area_sr=pixel_area_sr,
        )
    inverse_variance = None
    correlation_factor = None
    if stacking_method != "median" and variance_available:
        inverse_variance = _linear_inverse_variance(
            normalization,
            variance_weight,
            value_semantics=(
                "surface_brightness"
                if stacking_method == "sigma_clipped"
                else value_semantics
            ),
            pixel_area_sr=pixel_area_sr,
        )
        correlation_factor = _local_noise_correlation_factor(
            normalization,
            variance_weight,
        )
    coverage_moc = None
    if observation_mocs is not None:
        coverage_moc = MOC(torch.empty((0,), dtype=torch.int64))
        for moc in observation_mocs:
            coverage_moc = coverage_moc.union(moc)

    result = MapResult(
        data=coadd,
        weight=denom,
        header=header_dict,
        inverse_variance=inverse_variance,
        correlation_factor=correlation_factor,
        context=context,
        provenance=_context_provenance(
            observations,
            resampling_kernel=resampling_kernel,
            value_semantics=value_semantics,
            stacking_method=stacking_method,
            drizzle_pixfrac=drizzle_pixfrac,
            surface_brightness_area_scaling=apply_surface_brightness_area_scaling,
            data_dependent_background=(
                background_estimator is not None
                or cross_exposure_sky_corrector is not None
            ),
            has_correlation_factor=correlation_factor is not None,
        ),
        overlap_count=overlap_count,
        coverage_moc=coverage_moc,
        observation_mocs=tuple(observation_mocs)
        if observation_mocs is not None
        else None,
    )
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        write(output_path, coadd, header=header_dict, overwrite=True)
    return result


_FINITE_APERTURE_COVARIANCE_KERNELS = frozenset(
    {
        "nearest",
        "bilinear",
        "bicubic",
        "lanczos2",
        "lanczos3",
        "lanczos4",
        "lanczos5",
        "mitchell_netravali",
        "catmull_rom",
        "drizzle_point",
        "drizzle_square",
        "drizzle_turbo",
        "drizzle_gaussian",
        "drizzle_lanczos2",
        "drizzle_lanczos3",
        "conservative_polygon",
        "splat_point",
        "splat_box",
    }
)


def evaluate_image_plane_aperture_uncertainty(
    observations: Sequence[ImageObservation],
    result: MapResult,
    aperture_weights: Tensor,
    *,
    resampling_kernel: MapResamplingKernel = "bilinear",
    value_semantics: MapValueSemantics = "surface_brightness",
    drizzle_pixfrac: float = 1.0,
    spatial_tile_hw: tuple[int, int] | None = None,
    surface_brightness_area_scaling: bool | None = None,
    device: torch.device | str | None = None,
) -> ApertureUncertainty:
    """Propagate independent detector noise through one finite map aperture.

    The aperture is evaluated with the adjoint of the same finite-support
    observation geometry used by :func:`coadd_image_plane`. Source-pixel
    coefficients are accumulated across output tiles before squaring, so shared
    detector pixels contribute their covariance exactly without materializing a
    map covariance matrix. The supplied ``result`` must be the matching weighted
    mean product and provides its already-computed normalization.
    """
    if not observations:
        raise ValueError("At least one observation is required")
    resampling_kernel = _validated_map_resampling_kernel(resampling_kernel)
    if resampling_kernel not in _FINITE_APERTURE_COVARIANCE_KERNELS:
        raise ValueError(
            "finite-aperture covariance requires a finite-support resampling kernel"
        )
    value_semantics = _validated_map_value_semantics(value_semantics)
    drizzle_pixfrac = _validated_drizzle_pixfrac(drizzle_pixfrac)
    if result.data.ndim != 2 or result.weight.ndim != 2:
        raise ValueError("finite-aperture covariance requires a 2D MapResult")
    provenance = result.provenance
    execution = None if provenance is None else provenance.get("map_execution")
    if not isinstance(execution, dict):
        raise ValueError(
            "finite-aperture covariance requires MapResult execution provenance"
        )
    try:
        recorded_pixfrac = float(execution["drizzle_pixfrac"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("MapResult execution provenance is incomplete") from exc
    if (
        execution.get("resampling_kernel") != resampling_kernel
        or execution.get("value_semantics") != value_semantics
        or execution.get("stacking_method") != "weighted_mean"
        or recorded_pixfrac != drizzle_pixfrac
    ):
        raise ValueError(
            "MapResult execution provenance does not match the aperture operator"
        )
    if execution.get("data_dependent_background") is not False:
        raise ValueError(
            "finite-aperture covariance is unavailable after data-dependent "
            "background estimation"
        )
    recorded_observations = provenance.get("context_observations")
    expected_observations = [
        {"index": index, "path": observation.path, "hdu": observation.hdu}
        for index, observation in enumerate(observations)
    ]
    if recorded_observations != expected_observations:
        raise ValueError(
            "MapResult observation provenance does not match the aperture operator"
        )
    recorded_operators = provenance.get("observation_operators")
    expected_operators = [
        _observation_operator_provenance(observation)
        for observation in observations
    ]
    if recorded_operators != expected_operators:
        raise ValueError(
            "MapResult observation-operator provenance does not match the aperture operator"
        )
    target_shape = (int(result.data.shape[0]), int(result.data.shape[1]))
    if tuple(result.weight.shape) != target_shape:
        raise ValueError("MapResult data and weight shapes must match")
    aperture = torch.as_tensor(aperture_weights)
    if not aperture.is_floating_point() or tuple(aperture.shape) != target_shape:
        raise ValueError("aperture_weights must be a floating map-shaped tensor")
    if not bool(torch.isfinite(aperture).all()) or not bool((aperture != 0.0).any()):
        raise ValueError("aperture_weights must be finite and nonzero")
    if result.inverse_variance is None:
        raise ValueError(
            "finite-aperture covariance requires explicit detector inverse variance"
        )
    inverse_variance = torch.as_tensor(result.inverse_variance)
    if tuple(inverse_variance.shape) != target_shape:
        raise ValueError("MapResult inverse variance must match its data shape")

    map_device = _resolve_map_device(device)
    aperture = aperture.to(device=map_device, dtype=torch.float32)
    map_weight = result.weight.to(device=map_device, dtype=torch.float32)
    inverse_variance = inverse_variance.to(device=map_device, dtype=torch.float32)
    accumulator_dtype = (
        torch.float32 if map_device.type == "mps" else torch.float64
    )
    requested = aperture != 0.0
    if bool((requested & ((map_weight <= 0.0) | (inverse_variance <= 0.0))).any()):
        raise ValueError(
            "aperture includes pixels without detector-noise-supported map coverage"
        )
    diagonal_variance = torch.sum(
        torch.square(aperture)
        * torch.where(
            inverse_variance > 0.0,
            inverse_variance.reciprocal(),
            torch.zeros_like(inverse_variance),
        ),
        dtype=accumulator_dtype,
    )

    header_dict = normalized_header_for_shape(
        dict(result.header), target_shape, overrides=None
    )
    if value_semantics == "flux":
        target_area_sr = pixel_area_sr_map(
            WCS(header_dict), target_shape, device=map_device
        ).to(dtype=torch.float32)
        normalization = map_weight * torch.square(target_area_sr)
        linear_form_weight = torch.where(
            normalization > 0.0,
            aperture * target_area_sr / normalization,
            torch.zeros_like(aperture),
        )
    else:
        target_area_sr = None
        linear_form_weight = torch.where(
            map_weight > 0.0,
            aperture / map_weight,
            torch.zeros_like(aperture),
        )

    read_device = str(map_device)
    first_image, first_header = load_observation_image_and_header(
        observations[0], device=read_device
    )
    first_image = first_image.to(device=map_device)
    first_header = dict(first_header)
    if value_semantics != "surface_brightness":
        apply_area_scaling = False
    elif surface_brightness_area_scaling is not None:
        apply_area_scaling = bool(surface_brightness_area_scaling)
    else:
        apply_area_scaling = _should_apply_surface_brightness_area_scaling(
            observations,
            target_header=header_dict,
            target_shape=target_shape,
            first_header=first_header,
        )
    if execution.get("surface_brightness_area_scaling") is not apply_area_scaling:
        raise ValueError(
            "MapResult area-scaling provenance does not match the aperture operator"
        )
    full_representative_area_sr = (
        representative_pixel_area_sr(
            WCS(header_dict), target_shape, device=map_device
        )
        if apply_area_scaling
        else None
    )

    if spatial_tile_hw is None:
        tile_height = min(1024, target_shape[0])
        tile_width = min(1024, target_shape[1])
    else:
        tile_height = int(spatial_tile_hw[0])
        tile_width = int(spatial_tile_hw[1])
        if tile_height <= 0 or tile_width <= 0:
            raise ValueError("spatial_tile_hw dimensions must be positive")

    exact_variance = torch.zeros(
        (), dtype=accumulator_dtype, device=map_device
    )
    for observation_index, observation in enumerate(observations):
        if observation_index == 0:
            image = first_image
            observation_header = first_header
        else:
            image, observation_header = load_observation_image_and_header(
                observation, device=read_device
            )
            image = image.to(device=map_device)
        prepared = prepare_image_observation_data(
            observation,
            image,
            device=map_device,
            resampling_kernel=resampling_kernel,
            load_weight_map_fn=_load_observation_weight_map,
            apply_calibration_fn=_apply_observation_calibration,
            background_estimator=None,
        )
        if prepared.operator_bundle.noise is None:
            raise ValueError(
                "finite-aperture covariance requires explicit detector inverse variance "
                "for every observation"
            )
        source_weight = prepared.source_weight.detach().to(dtype=torch.float32)
        source_image = (
            prepared.image.detach().to(dtype=torch.float32).clone().requires_grad_(True)
        )
        differentiable = PreparedImageObservationData(
            image=source_image,
            source_weight=source_weight,
            operator_bundle=prepared.operator_bundle,
        )
        source_wcs = WCS(observation_header)
        source_gradient = torch.zeros_like(source_image)
        for y0 in range(0, target_shape[0], tile_height):
            for x0 in range(0, target_shape[1], tile_width):
                y1 = min(y0 + tile_height, target_shape[0])
                x1 = min(x0 + tile_width, target_shape[1])
                aperture_tile = linear_form_weight[y0:y1, x0:x1]
                if not bool((aperture_tile != 0.0).any()):
                    continue
                tile_shape = (y1 - y0, x1 - x0)
                sub_header, _ = _subtarget_header_for_window(
                    header_dict,
                    full_shape=target_shape,
                    y0=y0,
                    y1=y1,
                    x0=x0,
                    x1=x1,
                )
                needs_target = (
                    value_semantics == "flux"
                    or observation_header != sub_header
                    or tuple(source_image.shape) != tile_shape
                )
                prepared_target = None
                if needs_target:
                    prepared_target = _prepare_target(
                        header=sub_header,
                        target_wcs=WCS(sub_header),
                        target_shape=tile_shape,
                        device=map_device,
                        canvas_shape=target_shape,
                        canvas_origin_yx=(y0, x0),
                        include_pixel_area=(value_semantics == "flux"),
                        include_representative_pixel_area=apply_area_scaling,
                        include_quadrature=(value_semantics == "flux"),
                        include_world_grid=False,
                    )
                    prepared_target.representative_pixel_area_sr = (
                        full_representative_area_sr
                    )
                with torch.enable_grad():
                    projected = _project_observation_to_target_chunk(
                        observation,
                        image=source_image,
                        header_dict_obs=observation_header,
                        source_wcs=source_wcs,
                        prepared_observation=differentiable,
                        prepared_target=prepared_target,
                        target_header=sub_header,
                        target_shape=tile_shape,
                        device=map_device,
                        footprint_order=None,
                        resampling_kernel=resampling_kernel,
                        value_semantics=value_semantics,
                        drizzle_pixfrac=drizzle_pixfrac,
                        background_estimator=None,
                    )
                    if projected.numer.numel() == 0:
                        continue
                    if projected.target_window is None:
                        projected_aperture = aperture_tile
                    else:
                        py0, py1, px0, px1 = projected.target_window
                        projected_aperture = aperture_tile[py0:py1, px0:px1]
                    linear_form = torch.sum(projected.numer * projected_aperture)
                    if not linear_form.requires_grad:
                        continue
                    gradient = torch.autograd.grad(
                        linear_form,
                        source_image,
                        retain_graph=False,
                        create_graph=False,
                        allow_unused=True,
                    )[0]
                if gradient is not None:
                    source_gradient.add_(gradient.detach())
        source_variance = torch.where(
            source_weight > 0.0,
            source_weight.reciprocal(),
            torch.zeros_like(source_weight),
        )
        exact_variance.add_(
            torch.sum(
                torch.square(source_gradient) * source_variance,
                dtype=accumulator_dtype,
            )
        )
        if observation_index == 0:
            del first_image
        del differentiable, image, prepared, source_gradient, source_image
        del source_variance, source_weight

    return ApertureUncertainty(
        variance=exact_variance,
        diagonal_variance=diagonal_variance,
    )


class ImagePlaneMapMaker:
    """Differentiable image-plane coaddition baseline."""

    def __init__(
        self,
        *,
        target_header: dict | None = None,
        resampling_kernel: MapResamplingKernel = "bilinear",
        value_semantics: MapValueSemantics = "surface_brightness",
        stacking_method: MapStackingMethod = "weighted_mean",
        drizzle_pixfrac: float = 1.0,
        clip_sigma: float = 4.0,
        clip_iterations: int = 3,
        spatial_tile_hw: tuple[int, int] | None = None,
        device: torch.device | str | None = None,
        scratch_dir: str | Path | None = None,
        background_estimator: Callable[[Tensor, Tensor | None], Tensor] | None = None,
        cross_exposure_sky_corrector: CrossExposureSkyCorrector | None = None,
    ) -> None:
        self.target_header = target_header
        self.resampling_kernel = _validated_map_resampling_kernel(resampling_kernel)
        self.value_semantics = _validated_map_value_semantics(value_semantics)
        self.stacking_method = _validated_map_stacking_method(stacking_method)
        self.drizzle_pixfrac = _validated_drizzle_pixfrac(drizzle_pixfrac)
        self.clip_sigma = float(clip_sigma)
        self.clip_iterations = int(clip_iterations)
        self.spatial_tile_hw = (
            None
            if spatial_tile_hw is None
            else (int(spatial_tile_hw[0]), int(spatial_tile_hw[1]))
        )
        self.device = device
        self.scratch_dir = scratch_dir
        self.background_estimator = background_estimator
        self.cross_exposure_sky_corrector = cross_exposure_sky_corrector

    def run(
        self,
        observations: Sequence[ImageObservation],
        *,
        output_path: str | None = None,
        device: torch.device | str | None = None,
    ) -> MapResult:
        eff_device = device if device is not None else self.device
        return coadd_image_plane(
            observations,
            target_header=self.target_header,
            output_path=output_path,
            resampling_kernel=self.resampling_kernel,
            value_semantics=self.value_semantics,
            stacking_method=self.stacking_method,
            drizzle_pixfrac=self.drizzle_pixfrac,
            clip_sigma=self.clip_sigma,
            clip_iterations=self.clip_iterations,
            spatial_tile_hw=self.spatial_tile_hw,
            device=eff_device,
            scratch_dir=self.scratch_dir,
            background_estimator=self.background_estimator,
            cross_exposure_sky_corrector=self.cross_exposure_sky_corrector,
        )


def coadd_image_plane_cube(
    observations: Sequence[ImageObservation],
    *,
    target: PlannedImagePlaneCubeTarget | None = None,
    target_header: dict | None = None,
    footprint_order: int | None = None,
    resampling_kernel: MapResamplingKernel = "bilinear",
    value_semantics: MapValueSemantics = "surface_brightness",
    stacking_method: MapStackingMethod = "weighted_mean",
    drizzle_pixfrac: float = 1.0,
    spectral_axis: SpectralAxis | None = None,
    spectral_bin_edges: Tensor | None = None,
    spectral_unit: str = "Angstrom",
    spectral_type: str = "WAVE",
    reference_index: int = 0,
    device: torch.device | str | None = None,
    background_estimator: Callable[[Tensor, Tensor | None], Tensor] | None = None,
    cross_exposure_sky_corrector: CrossExposureSkyCorrector | None = None,
) -> CubeResult:
    if not observations:
        raise ValueError("At least one observation is required")
    if cross_exposure_sky_corrector is not None:
        raise NotImplementedError(
            "cross_exposure_sky_corrector is not yet implemented for cube coaddition; "
            "use coadd_image_plane with per-band observations"
        )
    if target is not None and (
        target_header is not None
        or spectral_axis is not None
        or spectral_bin_edges is not None
        or reference_index != 0
    ):
        raise ValueError(
            "target is mutually exclusive with target_header, spectral_axis, "
            "spectral_bin_edges, and reference_index"
        )
    map_device = _resolve_map_device(device)
    read_dev = str(map_device)
    resampling_kernel = _validated_map_resampling_kernel(resampling_kernel)
    value_semantics = _validated_map_value_semantics(value_semantics)
    stacking_method = _validated_map_stacking_method(stacking_method)
    drizzle_pixfrac = _validated_drizzle_pixfrac(drizzle_pixfrac)
    if stacking_method in ("median", "sigma_clipped"):
        raise ValueError(
            "coadd_image_plane_cube supports median and sigma_clipped only for "
            "single-band image-plane coadds"
        )

    if target is None:
        target = plan_image_plane_cube_target(
            observations,
            reference_index=reference_index,
            header_overrides=target_header,
            spectral_axis=spectral_axis,
            spectral_bin_edges=spectral_bin_edges,
            spectral_unit=spectral_unit,
            spectral_type=spectral_type,
            device=map_device,
        )

    (
        observation_center,
        observation_lower,
        observation_upper,
        observation_filter_center,
        observation_filter_sigma,
        observation_names,
    ) = _observation_spectral_columns(observations)
    projection = _build_spectral_projection(
        observation_center,
        observation_lower,
        observation_upper,
        observation_filter_center,
        observation_filter_sigma,
        target_axis=target.spectral_axis,
        target_edges=target.spectral_bin_edges,
    )
    band_metadata = _build_band_metadata(
        target.spectral_axis,
        target.spectral_bin_edges,
        projection,
        observation_names,
    )

    first_image, first_header = load_observation_image_and_header(
        observations[0],
        device=read_dev,
    )
    device = map_device
    header_dict = dict(target.header)
    target_shape = target.shape
    target_wcs = WCS(header_dict)
    apply_surface_brightness_area_scaling = (
        value_semantics == "surface_brightness"
        and stacking_method == "weighted_mean"
        and _should_apply_surface_brightness_area_scaling(
            observations,
            target_header=header_dict,
            target_shape=target_shape,
            first_header=dict(first_header),
        )
    )
    prepared_target: _PreparedImagePlaneTarget | None = None

    def ensure_prepared_target() -> _PreparedImagePlaneTarget:
        nonlocal prepared_target
        if prepared_target is None:
            prepared_target = _prepare_target(
                header=header_dict,
                target_wcs=target_wcs,
                target_shape=target_shape,
                device=device,
                include_pixel_area=(value_semantics == "flux"),
                include_representative_pixel_area=apply_surface_brightness_area_scaling,
                include_quadrature=(value_semantics == "flux"),
                include_world_grid=False,
            )
        return prepared_target

    n_band = int(target.spectral_axis.values.numel())
    numer = torch.zeros((n_band, *target_shape), dtype=torch.float32, device=device)
    denom = torch.zeros((n_band, *target_shape), dtype=torch.float32, device=device)
    variance_weight = torch.zeros_like(denom)
    correlation_weight = torch.zeros_like(denom)
    variance_available = stacking_method == "weighted_mean"
    overlap_count = torch.zeros(
        (n_band, *target_shape), dtype=torch.int32, device=device
    )
    coverage_mocs: list[MOC | None] | None = None
    if footprint_order is not None:
        coverage_mocs = [
            MOC(torch.empty((0,), dtype=torch.int64)) for _ in range(n_band)
        ]

    for obs_idx, obs in enumerate(observations):
        if obs_idx == 0:
            image = first_image
            header_dict_obs = dict(first_header)
        else:
            image, header_dict_obs = load_observation_image_and_header(
                obs, device=read_dev
            )
        need_prepared_target = (
            value_semantics == "flux"
            or footprint_order is not None
            or header_dict_obs != header_dict
            or tuple(image.shape) != target_shape
        )
        source_wcs = None
        if need_prepared_target and not (
            headers_equivalent_for_direct_copy(header_dict_obs, header_dict)
            and tuple(image.shape) == target_shape
        ):
            source_wcs = WCS(header_dict_obs)
        projected = _project_observation_to_target_chunk(
            obs,
            image=image,
            header_dict_obs=header_dict_obs,
            source_wcs=source_wcs,
            prepared_target=ensure_prepared_target() if need_prepared_target else None,
            target_header=header_dict,
            target_shape=target_shape,
            device=device,
            footprint_order=footprint_order,
            resampling_kernel=resampling_kernel,
            value_semantics=value_semantics,
            drizzle_pixfrac=drizzle_pixfrac,
            background_estimator=background_estimator,
        )
        coeffs = (
            projection[obs_idx]
            .to(device=device, dtype=torch.float32)
            .view(n_band, 1, 1)
        )
        pixel_area_sr = None
        if value_semantics == "flux":
            prepared = ensure_prepared_target()
            if prepared.pixel_area_sr is None:
                raise RuntimeError("target pixel area is required for flux semantics")
            pixel_area_sr = prepared.pixel_area_sr
            if projected.target_window is not None:
                y0, y1, x0, x1 = projected.target_window
                pixel_area_sr = pixel_area_sr[y0:y1, x0:x1]
        accumulate_projected_observation(
            numer,
            denom,
            overlap_count,
            projected_numer=projected.numer,
            projected_weight=projected.weight,
            projected_overlap=projected.overlap,
            stacking_method=stacking_method,
            value_semantics=value_semantics,
            pixel_area_sr=pixel_area_sr,
            target_window=projected.target_window,
            coeffs=coeffs,
        )
        if projected.variance_weight is None:
            variance_available = False
        elif variance_available:
            _accumulate_variance_weight(
                variance_weight,
                torch.square(coeffs) * projected.variance_weight.unsqueeze(0),
                projected.target_window,
            )
            _accumulate_variance_weight(
                correlation_weight,
                torch.square(coeffs) * projected.weight.unsqueeze(0),
                projected.target_window,
            )
        if coverage_mocs is not None and projected.moc is not None:
            band_indices = torch.nonzero(
                projection[obs_idx] > 0.0, as_tuple=False
            ).flatten()
            for band_idx in band_indices.tolist():
                current = coverage_mocs[band_idx]
                coverage_mocs[band_idx] = (
                    projected.moc if current is None else current.union(projected.moc)
                )

    pixel_area_sr = None
    if value_semantics == "flux":
        prepared = ensure_prepared_target()
        if prepared.pixel_area_sr is None:
            raise RuntimeError("target pixel area is required for flux semantics")
        pixel_area_sr = prepared.pixel_area_sr.unsqueeze(0)
    normalization = denom
    cube, denom = finalize_projected_stack(
        numer,
        denom,
        stacking_method=stacking_method,
        value_semantics=value_semantics,
        pixel_area_sr=pixel_area_sr,
    )
    inverse_variance = (
        _linear_inverse_variance(
            normalization,
            variance_weight,
            value_semantics=value_semantics,
            pixel_area_sr=pixel_area_sr,
        )
        if variance_available
        else None
    )
    correlation_factor = (
        _local_noise_correlation_factor(correlation_weight, variance_weight)
        if variance_available
        else None
    )
    return CubeResult(
        spectral_axis=target.spectral_axis,
        band_metadata=band_metadata,
        data=cube,
        weight=denom,
        header=header_dict,
        spatial_wcs=target_wcs,
        inverse_variance=inverse_variance,
        correlation_factor=correlation_factor,
        overlap_count=overlap_count,
        coverage_mocs=tuple(coverage_mocs) if coverage_mocs is not None else None,
    )


def coadd_image_plane_multiband(
    observations: Sequence[ImageObservation],
    *,
    target_header: dict | None = None,
    footprint_order: int | None = None,
    resampling_kernel: MapResamplingKernel = "bilinear",
    value_semantics: MapValueSemantics = "surface_brightness",
    stacking_method: MapStackingMethod = "weighted_mean",
    spectral_unit: str = "Angstrom",
    spectral_type: str = "WAVE",
    device: torch.device | str | None = None,
) -> MultibandResult:
    cube = coadd_image_plane_cube(
        observations,
        target_header=target_header,
        footprint_order=footprint_order,
        resampling_kernel=resampling_kernel,
        value_semantics=value_semantics,
        stacking_method=stacking_method,
        spectral_axis=None,
        spectral_bin_edges=None,
        spectral_unit=spectral_unit,
        spectral_type=spectral_type,
        device=device,
    )
    return MultibandResult(
        spectral_axis=cube.spectral_axis,
        data=cube.data,
        weight=cube.weight,
        header=cube.header,
        inverse_variance=cube.inverse_variance,
        correlation_factor=cube.correlation_factor,
        overlap_count=cube.overlap_count,
        coverage_mocs=cube.coverage_mocs,
        band_observation_counts=cube.band_metadata.observation_count,
    )


# ---------------------------------------------------------------------------
# Clean top-level mapmaking API
# ---------------------------------------------------------------------------


def _coerce_target_header(
    target: Target | CubeTarget | Mapping[str, Any] | WCS | None,
) -> dict[str, Any] | None:
    if target is None:
        return None
    if isinstance(target, (PlannedImagePlaneTarget, PlannedImagePlaneCubeTarget)):
        return dict(target.header)
    if isinstance(target, WCS):
        return target.to_header()
    if isinstance(target, Mapping):
        return dict(target)
    raise TypeError(
        f"target must be a Target, CubeTarget, header dict, WCS, or None; got {type(target).__name__}"
    )


def _propagate_observation_variance_components(
    observations: Sequence[Observation],
    *,
    target_header: dict[str, Any],
    target_shape: tuple[int, int],
    resampling_kernel: MapResamplingKernel,
    drizzle_pixfrac: float,
    value_semantics: MapValueSemantics,
    device: torch.device,
) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
    has_poisson = any(obs.var_poisson is not None for obs in observations)
    has_rnoise = any(obs.var_rnoise is not None for obs in observations)
    has_flat = any(obs.var_flat is not None for obs in observations)
    if not (has_poisson or has_rnoise or has_flat):
        return None, None, None

    target_wcs = WCS(target_header)
    prepared_target = _prepare_target(
        header=target_header,
        target_wcs=target_wcs,
        target_shape=target_shape,
        device=device,
        include_pixel_area=(value_semantics == "flux"),
        include_representative_pixel_area=False,
        include_quadrature=(value_semantics == "flux"),
        include_world_grid=False,
    )

    def _accumulate_component(attr_name: str) -> Tensor:
        var_num = torch.zeros(target_shape, dtype=torch.float32, device=device)
        var_den = torch.zeros(target_shape, dtype=torch.float32, device=device)
        read_dev = str(device)
        for obs in observations:
            comp = getattr(obs, attr_name)
            image, header_dict_obs = load_observation_image_and_header(
                obs, device=read_dev
            )
            image_dev = image.to(device=device, dtype=torch.float32)
            if comp is None:
                comp_t = torch.zeros_like(image_dev)
            elif isinstance(comp, Tensor):
                comp_t = validate_image_plane_2d(comp).to(
                    device=device, dtype=torch.float32
                )
            else:
                comp_raw, _ = read(
                    str(comp), hdu=obs.hdu, return_header=True, device=read_dev
                )
                comp_t = validate_image_plane_2d(comp_raw).to(
                    device=device, dtype=torch.float32
                )
            source_wcs = WCS(header_dict_obs)
            s_eff = _effective_observation_photometric_scale(obs, header_dict_obs)
            # _apply_observation_calibration multiplies data by s_eff once;
            # pre-multiplying by s_eff yields s_eff^2 * var_raw in calibrated units.
            scaled_var = comp_t * float(s_eff)
            if value_semantics == "flux":
                src_area = pixel_area_sr_map(
                    source_wcs,
                    (int(comp_t.shape[0]), int(comp_t.shape[1])),
                    device=device,
                )
                scaled_var = scaled_var / torch.clamp(
                    torch.square(src_area),
                    min=torch.finfo(src_area.dtype).tiny,
                )
            scaled_var = torch.where(
                torch.isfinite(image_dev),
                scaled_var,
                torch.full_like(scaled_var, float("nan")),
            )
            var_obs = dataclasses.replace(
                obs,
                data=scaled_var,
                wcs=header_dict_obs,
                background_offset=0.0,
            )
            proj = _project_observation_to_target_chunk(
                var_obs,
                image=scaled_var,
                header_dict_obs=header_dict_obs,
                source_wcs=source_wcs,
                prepared_target=prepared_target,
                target_header=target_header,
                target_shape=target_shape,
                device=device,
                footprint_order=None,
                resampling_kernel=resampling_kernel,
                value_semantics="surface_brightness",
                drizzle_pixfrac=drizzle_pixfrac,
                background_estimator=None,
            )
            w = proj.weight
            v_est = torch.where(w > 0.0, proj.numer / w, torch.zeros_like(proj.numer))
            if value_semantics == "flux" and prepared_target.pixel_area_sr is not None:
                pa = prepared_target.pixel_area_sr
                if proj.target_window is not None:
                    y0, y1, x0, x1 = proj.target_window
                    pa = pa[y0:y1, x0:x1]
                v_est = v_est * torch.square(pa)
            contrib_num = torch.square(w) * torch.clamp(v_est, min=0.0)
            if proj.target_window is None:
                var_num.add_(contrib_num)
                var_den.add_(w)
            else:
                y0, y1, x0, x1 = proj.target_window
                if y1 > y0 and x1 > x0:
                    var_num[y0:y1, x0:x1].add_(contrib_num)
                    var_den[y0:y1, x0:x1].add_(w)
        safe_den_sq = torch.clamp(
            torch.square(var_den), min=torch.finfo(var_den.dtype).tiny
        )
        return torch.where(
            var_den > 0.0, var_num / safe_den_sq, torch.zeros_like(var_num)
        )

    vp = _accumulate_component("var_poisson") if has_poisson else None
    vr = _accumulate_component("var_rnoise") if has_rnoise else None
    vf = _accumulate_component("var_flat") if has_flat else None
    return vp, vr, vf


def resample(
    observation: Observation | Sequence[Observation],
    target: Target | Mapping[str, Any] | WCS | None = None,
    *,
    config: ResampleConfig | None = None,
    kernel: MapResamplingKernel | None = None,
    pixfrac: float | None = None,
    semantics: MapValueSemantics | None = None,
    footprint_order: int | None = None,
    device: torch.device | str | None = None,
) -> MapResult | list[MapResult]:
    """Resample one or more observations onto a common target WCS grid without stacking."""
    cfg = config or ResampleConfig()
    resolved_kernel = kernel if kernel is not None else cfg.kernel
    resolved_pixfrac = pixfrac if pixfrac is not None else cfg.pixfrac
    resolved_semantics = semantics if semantics is not None else cfg.semantics
    single = isinstance(observation, ImageObservation)
    obs_list = [observation] if single else list(observation)
    if not obs_list:
        raise ValueError("At least one Observation is required")
    target_header = _coerce_target_header(target)
    if target_header is None and len(obs_list) > 1:
        target_header = dict(plan(obs_list, device=device).header)

    results: list[MapResult] = []
    for obs in obs_list:
        res = coadd_image_plane(
            [obs],
            target_header=target_header,
            footprint_order=footprint_order,
            resampling_kernel=resolved_kernel,
            value_semantics=resolved_semantics,
            stacking_method="weighted_mean",
            drizzle_pixfrac=resolved_pixfrac,
            device=device,
        )
        vp, vr, vf = _propagate_observation_variance_components(
            [obs],
            target_header=res.header,
            target_shape=tuple(res.data.shape),
            resampling_kernel=resolved_kernel,
            drizzle_pixfrac=resolved_pixfrac,
            value_semantics=resolved_semantics,
            device=res.data.device,
        )
        if vp is not None or vr is not None or vf is not None:
            res = dataclasses.replace(res, var_poisson=vp, var_rnoise=vr, var_flat=vf)
        results.append(res)
    return results[0] if single else results


def blot(
    coadd_or_data: MapResult | Tensor,
    source_or_target: Observation
    | Sequence[Observation]
    | Target
    | Mapping[str, Any]
    | WCS,
    *,
    target: Observation
    | Sequence[Observation]
    | Target
    | Mapping[str, Any]
    | WCS
    | None = None,
    kernel: Literal[
        "bilinear", "bicubic", "lanczos2", "lanczos3", "lanczos4", "nearest"
    ] = "bilinear",
    semantics: MapValueSemantics = "surface_brightness",
    device: torch.device | str | None = None,
) -> Tensor | list[Tensor]:
    """Inverse-project (blot) a clean coadd back into one or more distorted exposure frames.

    Supports both:
    - ``blot(coadd_result, observations, kernel=..., semantics=...)``
    - ``blot(coadd_tensor, coadd_wcs_or_header, target=observations, ...)``
    """
    semantics = _validated_map_value_semantics(semantics)
    if kernel not in (
        "bilinear",
        "bicubic",
        "lanczos2",
        "lanczos3",
        "lanczos4",
        "nearest",
    ):
        raise ValueError(
            "blot kernel must be one of 'bilinear', 'bicubic', 'lanczos2', 'lanczos3', 'lanczos4', or 'nearest'"
        )
    if isinstance(coadd_or_data, MapResult):
        coadd_tensor = coadd_or_data.data
        coadd_header = dict(coadd_or_data.header)
        blot_target = source_or_target if target is None else target
    else:
        coadd_tensor = validate_image_plane_2d(coadd_or_data)
        if target is None:
            raise ValueError(
                "When passing a raw Tensor to blot(), specify coadd WCS/header as second argument and target=... as keyword"
            )
        if isinstance(source_or_target, (ImageObservation, Sequence)) and not isinstance(
            source_or_target, Mapping
        ):
            raise TypeError(
                "Second argument must be coadd Target/header/WCS when first argument is a raw Tensor"
            )
        coerced = _coerce_target_header(source_or_target)
        assert coerced is not None
        coadd_header = coerced
        blot_target = target

    map_device = (
        coadd_tensor.device if device is None else _resolve_map_device(device)
    )
    coadd_img = coadd_tensor.to(device=map_device, dtype=torch.float32)
    coadd_wcs = WCS(coadd_header)
    coadd_shape = (int(coadd_img.shape[0]), int(coadd_img.shape[1]))
    h_c, w_c = coadd_shape

    def _blot_single(
        dest: Observation | Target | Mapping[str, Any] | WCS,
    ) -> Tensor:
        if isinstance(dest, ImageObservation):
            obs_img, obs_hdr = load_observation_image_and_header(
                dest, device=str(map_device)
            )
            dest_shape = (int(obs_img.shape[0]), int(obs_img.shape[1]))
            dest_hdr = dict(obs_hdr)
        else:
            coerced_dest = _coerce_target_header(dest)
            if coerced_dest is None:
                raise ValueError("blot destination cannot be None")
            dest_hdr = coerced_dest
            dest_shape = target_shape_from_header(dest_hdr, fallback=coadd_shape)
        dest_wcs = WCS(dest_hdr)
        dest_prep = _prepare_target(
            header=dest_hdr,
            target_wcs=dest_wcs,
            target_shape=dest_shape,
            device=map_device,
            include_pixel_area=(semantics == "flux"),
            include_representative_pixel_area=False,
            include_quadrature=False,
            include_world_grid=False,
        )
        ra_grid, dec_grid = _build_target_world_grid(
            dest_wcs, dest_shape[0], dest_shape[1], device=map_device
        )
        flat_ra = ra_grid.reshape(-1)
        flat_dec = dec_grid.reshape(-1)
        flat_ra, flat_dec = _convert_world_between_wcs(
            flat_ra, flat_dec, dest_wcs, coadd_wcs
        )
        src_x_flat, src_y_flat = coadd_wcs.world_to_pixel(flat_ra, flat_dec)
        src_x = src_x_flat.reshape(dest_shape)
        src_y = src_y_flat.reshape(dest_shape)
        in_bounds = (
            (src_x >= 0.0)
            & (src_x <= float(w_c - 1))
            & (src_y >= 0.0)
            & (src_y <= float(h_c - 1))
        )
        clean_coadd = (
            torch.nan_to_num(coadd_img, nan=0.0, posinf=0.0, neginf=0.0)
            if not bool(torch.isfinite(coadd_img).all())
            else coadd_img
        )
        x_norm = (2.0 * src_x / max(w_c - 1, 1)) - 1.0
        y_norm = (2.0 * src_y / max(h_c - 1, 1)) - 1.0
        grid = torch.stack([x_norm, y_norm], dim=-1).to(dtype=torch.float32)
        sampled = sample_image_plane_batched_channels(
            clean_coadd.unsqueeze(0),
            x_src=src_x,
            y_src=src_y,
            grid=grid,
            kernel=kernel,
        )[0]
        sampled = torch.where(in_bounds, sampled, torch.zeros_like(sampled))
        if semantics == "flux":
            assert dest_prep.pixel_area_sr is not None
            coadd_area = pixel_area_sr_map(
                coadd_wcs, coadd_shape, device=map_device
            )
            coadd_area_at_dest = (
                F.grid_sample(
                    coadd_area.unsqueeze(0).unsqueeze(0),
                    grid.unsqueeze(0),
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True,
                )
                .squeeze(0)
                .squeeze(0)
            )
            area_ratio = dest_prep.pixel_area_sr / torch.clamp(
                coadd_area_at_dest, min=torch.finfo(torch.float32).tiny
            )
            sampled = sampled * area_ratio
        return sampled

    if isinstance(blot_target, Sequence) and not isinstance(
        blot_target, (Mapping, str, bytes)
    ):
        return [_blot_single(item) for item in blot_target]
    return _blot_single(blot_target)  # type: ignore[arg-type]


def _blot_spatial_derivative(blotted: Tensor) -> Tensor:
    """Compute AstroDrizzle / JWST stcal local spatial gradient max(|dI/dx|, |dI/dy|)."""
    H, W = blotted.shape
    padded = F.pad(
        blotted.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="replicate"
    ).squeeze(0).squeeze(0)
    dx_left = torch.abs(blotted - padded[1 : H + 1, 0:W])
    dx_right = torch.abs(blotted - padded[1 : H + 1, 2 : W + 2])
    dy_up = torch.abs(blotted - padded[0:H, 1 : W + 1])
    dy_down = torch.abs(blotted - padded[2 : H + 2, 1 : W + 1])
    return torch.maximum(
        torch.maximum(dx_left, dx_right), torch.maximum(dy_up, dy_down)
    )


def _load_existing_observation_valid_mask(
    obs: Observation,
    image_shape: tuple[int, int],
    device: torch.device,
) -> Tensor | None:
    from .operators.mask_weight import _load_or_validate_aux_plane

    mask = _load_or_validate_aux_plane(
        obs.mask,
        obs.mask_path,
        obs.mask_hdu,
        image_shape=image_shape,
        device=device,
        label="Mask",
        preserve_integer=True,
    )
    if mask is None:
        return None
    if obs.good_bits is not None:
        dq = mask.to(device=device, dtype=torch.int64)
        return (dq & (~int(obs.good_bits))) == 0
    if mask.dtype == torch.bool:
        return mask.to(device=device, dtype=torch.bool)
    return mask.to(device=device, dtype=torch.float32) > 0.0


def detect_outliers(
    observations: Sequence[Observation],
    coadd_result: MapResult | None = None,
    *,
    config: OutlierConfig | None = None,
    snr_primary: float | None = None,
    snr_neighbor: float | None = None,
    scale_primary: float | None = None,
    scale_neighbor: float | None = None,
    min_floor: float | None = None,
    grow: int | None = None,
    kernel: Literal[
        "bilinear", "bicubic", "lanczos2", "lanczos3", "lanczos4", "nearest"
    ]
    | None = None,
    update_masks: bool = True,
    device: torch.device | str | None = None,
) -> tuple[list[Tensor], list[Observation]]:
    """Detect cosmic rays and transient outliers via AstroDrizzle / JWST blot + spatial derivative test.

    Returns ``(outlier_masks, updated_observations)`` where ``outlier_masks[i]`` is a boolean
    tensor with ``True`` at detected cosmic-ray/outlier pixels in exposure ``i``.
    """
    if not observations:
        raise ValueError("At least one Observation is required")
    cfg = config or OutlierConfig()
    s_prim = float(snr_primary if snr_primary is not None else cfg.snr[0])
    s_neigh = float(snr_neighbor if snr_neighbor is not None else cfg.snr[1])
    c_prim = float(scale_primary if scale_primary is not None else cfg.scale[0])
    c_neigh = float(
        scale_neighbor if scale_neighbor is not None else cfg.scale[1]
    )
    floor_val = float(min_floor if min_floor is not None else cfg.min_floor)
    grow_px = int(grow if grow is not None else cfg.grow)
    blot_kernel = kernel if kernel is not None else cfg.blot_kernel
    should_update = bool(update_masks and cfg.mode != "flag_only")

    map_device = _resolve_map_device(device)
    if coadd_result is None:
        coadd_result = coadd_image_plane(
            observations,
            stacking_method="median" if len(observations) >= 3 else "weighted_mean",
            resampling_kernel="bilinear",
            value_semantics="surface_brightness",
            device=map_device,
        )

    blotted_list = blot(
        coadd_result,
        list(observations),
        kernel=blot_kernel,
        semantics="surface_brightness",
        device=map_device,
    )
    assert isinstance(blotted_list, list)

    outlier_masks: list[Tensor] = []
    updated_observations: list[Observation] = []
    read_dev = str(map_device)
    for obs, blotted in zip(observations, blotted_list):
        raw_img, hdr = load_observation_image_and_header(obs, device=read_dev)
        raw_img_dev = raw_img.to(device=map_device, dtype=torch.float32)
        src_w = _load_observation_weight_map(obs, raw_img_dev, device=map_device)
        cal_img, cal_w = _apply_observation_calibration(obs, raw_img_dev, src_w)
        s_eff = _effective_observation_photometric_scale(obs, hdr)
        grad = _blot_spatial_derivative(blotted)
        diff = torch.abs(cal_img - blotted)

        # Estimate per-pixel 1-sigma noise in calibrated units from variance planes, weight map, or robust MAD
        noise_sigma: Tensor | None = None
        if obs.var_poisson is not None or obs.var_rnoise is not None:
            var_tot = torch.zeros_like(cal_img)
            if isinstance(obs.var_poisson, Tensor):
                var_tot = var_tot + obs.var_poisson.to(
                    device=map_device, dtype=torch.float32
                )
            if isinstance(obs.var_rnoise, Tensor):
                var_tot = var_tot + obs.var_rnoise.to(
                    device=map_device, dtype=torch.float32
                )
            noise_sigma = float(s_eff) * torch.sqrt(torch.clamp(var_tot, min=1e-12))
        elif (
            (obs.weight_map is not None or obs.weight_path is not None)
            and obs.weight_type in ("ivm", "rms", "variance")
        ) or obs.weight_is_inverse_variance:
            noise_sigma = torch.where(
                cal_w > 0.0,
                torch.rsqrt(torch.clamp(cal_w, min=1e-12)),
                torch.full_like(cal_img, float("inf")),
            )
        if noise_sigma is None:
            finite_diff = diff[torch.isfinite(diff)]
            if finite_diff.numel() > 0:
                mad_sigma = torch.clamp(
                    1.4826 * torch.median(finite_diff), min=1e-6
                )
            else:
                mad_sigma = torch.tensor(1.0, device=map_device, dtype=torch.float32)
            noise_sigma = torch.clamp(
                torch.sqrt(torch.clamp(torch.abs(blotted), min=0.0) + mad_sigma**2),
                min=float(mad_sigma.item()),
            )

        thresh_primary = c_prim * grad + s_prim * noise_sigma + floor_val
        thresh_neighbor = c_neigh * grad + s_neigh * noise_sigma + floor_val
        primary_cr = (diff > thresh_primary) & torch.isfinite(cal_img)
        if grow_px > 0 and bool(primary_cr.any()):
            k_size = 2 * grow_px + 1
            dilated = (
                F.max_pool2d(
                    primary_cr.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0),
                    kernel_size=k_size,
                    stride=1,
                    padding=grow_px,
                )
                .squeeze(0)
                .squeeze(0)
                > 0.0
            )
            cr_mask = primary_cr | (
                dilated & (diff > thresh_neighbor) & torch.isfinite(cal_img)
            )
        else:
            cr_mask = primary_cr
        outlier_masks.append(cr_mask)
        if should_update:
            obs_shape = (int(raw_img_dev.shape[0]), int(raw_img_dev.shape[1]))
            prev_valid = _load_existing_observation_valid_mask(
                obs, obs_shape, map_device
            )
            valid_mask = (~cr_mask) if prev_valid is None else (prev_valid & (~cr_mask))
            new_wmap = obs.weight_map
            if cfg.mode == "weight_zero" and isinstance(obs.weight_map, Tensor):
                new_wmap = torch.where(
                    ~cr_mask,
                    obs.weight_map.to(device=map_device),
                    torch.zeros_like(obs.weight_map.to(device=map_device)),
                )
            updated_observations.append(
                dataclasses.replace(
                    obs,
                    mask=valid_mask,
                    good_bits=None,
                    weight_map=new_wmap,
                )
            )
        else:
            updated_observations.append(obs)
    return outlier_masks, updated_observations


def _robust_scalar_sky(image: Tensor, bad_mask: Tensor | None, *, clip_sigma: float = 3.0, clip_iterations: int = 3) -> float:
    valid = torch.isfinite(image)
    if bad_mask is not None:
        valid = valid & (~bad_mask.to(device=image.device, dtype=torch.bool))
    vals = image[valid].to(dtype=torch.float32)
    if vals.numel() == 0:
        return 0.0
    for _ in range(max(int(clip_iterations), 1)):
        if vals.numel() < 3:
            break
        med = torch.median(vals)
        mad = torch.median(torch.abs(vals - med))
        sigma = 1.482602218505602 * mad
        if float(sigma.item()) <= 1e-12:
            break
        keep = torch.abs(vals - med) <= float(clip_sigma) * sigma
        if not bool(keep.any()) or int(keep.sum().item()) == int(vals.numel()):
            break
        vals = vals[keep]
    return float(torch.median(vals).item())


def match_sky(
    observations: Sequence[Observation],
    *,
    target: Target | Mapping[str, Any] | WCS | None = None,
    config: SkyConfig | None = None,
    mode: Literal[
        "constant",
        "planar",
        "mesh",
        "per_exposure",
        "local",
        "global",
        "match",
        "global+match",
        "overlap_constant",
        "overlap_planar",
        "overlap_mesh",
        "none",
    ]
    | None = None,
    mesh_size: int | None = None,
    regularization: float | None = None,
    apply: bool | None = None,
    device: torch.device | str | None = None,
) -> tuple[list[Observation], dict[str, Any]]:
    """Solve and optionally subtract sky background across exposures.

    Supports single-exposure mesh background estimation (``"per_exposure"`` / ``"local"``),
    JWST/Roman ``stcal.skymatch`` global modes (``"global"``, ``"match"``, ``"global+match"``),
    and global overlap-graph rectification (``"constant"``, ``"planar"``, ``"mesh"``).
    """
    if not observations:
        raise ValueError("At least one Observation is required")
    cfg = config or SkyConfig(mode="overlap_constant")
    raw_mode = mode if mode is not None else cfg.mode
    if raw_mode == "none":
        return list(observations), {"mode": "none"}
    should_apply = bool(cfg.subtract if apply is None else apply)
    resolved_mesh = int(mesh_size if mesh_size is not None else cfg.mesh_size)
    resolved_reg = float(
        regularization if regularization is not None else cfg.regularization
    )
    map_device = _resolve_map_device(device)
    read_dev = str(map_device)

    if raw_mode in ("per_exposure", "local"):
        estimator = MeshSkyEstimator(
            block_size=resolved_mesh,
            statistic=cfg.stat,
            mesh_filter_size=cfg.smooth_scale,
            self_mask_sigma=cfg.clip_sigma,
        )
        updated: list[Observation] = []
        backgrounds: list[Tensor] = []
        for obs in observations:
            img, hdr = load_observation_image_and_header(obs, device=read_dev)
            img_dev = img.to(device=map_device, dtype=torch.float32)
            obs_shape = (int(img_dev.shape[0]), int(img_dev.shape[1]))
            bad_mask = ~torch.isfinite(img_dev)
            prev_valid = _load_existing_observation_valid_mask(
                obs, obs_shape, map_device
            )
            if prev_valid is not None:
                bad_mask = bad_mask | (~prev_valid)
            bg = estimator(img_dev, bad_mask if bool(bad_mask.any()) else None)
            backgrounds.append(bg)
            if should_apply:
                updated.append(
                    dataclasses.replace(obs, data=img_dev - bg, header=hdr)
                )
            else:
                updated.append(obs)
        return updated, {"mode": raw_mode, "backgrounds": backgrounds}

    if raw_mode == "global":
        sky_levels: list[float] = []
        scales: list[float] = []
        for obs in observations:
            img, hdr = load_observation_image_and_header(obs, device=read_dev)
            img_dev = img.to(device=map_device, dtype=torch.float32)
            src_w = _load_observation_weight_map(obs, img_dev, device=map_device)
            cal_img, cal_w = _apply_observation_calibration(obs, img_dev, src_w)
            s_eff = _effective_observation_photometric_scale(obs, hdr)
            scales.append(s_eff)
            sky_levels.append(
                _robust_scalar_sky(
                    cal_img,
                    ~(cal_w > 0.0),
                    clip_sigma=cfg.clip_sigma,
                    clip_iterations=cfg.clip_iterations,
                )
            )
        global_sky = min(sky_levels) if sky_levels else 0.0
        offsets_t = torch.full(
            (len(observations),),
            float(global_sky),
            dtype=torch.float32,
            device=map_device,
        )
        updated_global: list[Observation] = []
        for obs, s_eff in zip(observations, scales):
            if should_apply:
                updated_global.append(
                    dataclasses.replace(
                        obs,
                        background_offset=float(obs.background_offset)
                        + float(global_sky) / float(s_eff),
                    )
                )
            else:
                updated_global.append(obs)
        return updated_global, {
            "mode": "global",
            "global_sky": float(global_sky),
            "per_exposure_sky": sky_levels,
            "offsets": offsets_t,
        }

    # Global overlap-graph sky matching ("constant"/"match"/"global+match", "planar", or "mesh")
    corrector_mode: Literal["constant", "planar", "mesh"]
    if raw_mode in ("constant", "overlap_constant", "match", "global+match"):
        corrector_mode = "constant"
    elif raw_mode in ("planar", "overlap_planar"):
        corrector_mode = "planar"
    elif raw_mode in ("mesh", "overlap_mesh"):
        corrector_mode = "mesh"
    else:
        raise ValueError(f"Unsupported sky matching mode: {raw_mode!r}")

    target_hdr = _coerce_target_header(target)
    if target_hdr is None:
        target_hdr = dict(plan(observations, device=map_device).header)
    resampled_list = resample(
        observations,
        target_hdr,
        kernel="bilinear",
        semantics="surface_brightness",
        device=map_device,
    )
    assert isinstance(resampled_list, list)
    images = [r.data for r in resampled_list]
    masks = [~(r.weight > 0.0) for r in resampled_list]
    corrector = CrossExposureSkyCorrector(
        mesh_size=resolved_mesh,
        regularization=resolved_reg,
        mode=corrector_mode,
    )

    if corrector_mode == "constant":
        offsets = corrector.fit_offsets(images, masks)
        if raw_mode == "global+match":
            equalized_skies = [
                _robust_scalar_sky(
                    img - off,
                    msk,
                    clip_sigma=cfg.clip_sigma,
                    clip_iterations=cfg.clip_iterations,
                )
                for img, msk, off in zip(images, masks, offsets)
            ]
            global_base = min(equalized_skies) if equalized_skies else 0.0
            offsets = offsets + float(global_base)
        updated_obs: list[Observation] = []
        for obs, off in zip(observations, offsets.tolist()):
            if should_apply:
                s_eff = _effective_observation_photometric_scale(obs)
                updated_obs.append(
                    dataclasses.replace(
                        obs,
                        background_offset=float(obs.background_offset)
                        + float(off) / float(s_eff),
                    )
                )
            else:
                updated_obs.append(obs)
        return updated_obs, {
            "mode": "global+match" if raw_mode == "global+match" else "constant",
            "offsets": offsets,
        }

    if corrector_mode == "planar":
        planes = corrector.fit_planes(images, masks)
        coadd_wcs = WCS(target_hdr)
        coadd_shape = (int(images[0].shape[0]), int(images[0].shape[1]))
        H_c, W_c = coadd_shape
        updated_obs = []
        for idx, obs in enumerate(observations):
            if should_apply:
                img, hdr = load_observation_image_and_header(obs, device=read_dev)
                img_dev = img.to(device=map_device, dtype=torch.float32)
                s_eff = _effective_observation_photometric_scale(obs, hdr)
                obs_wcs = WCS(hdr)
                obs_shape = (int(img_dev.shape[0]), int(img_dev.shape[1]))
                ra_grid, dec_grid = _build_target_world_grid(
                    obs_wcs, obs_shape[0], obs_shape[1], device=map_device
                )
                flat_ra = ra_grid.reshape(-1)
                flat_dec = dec_grid.reshape(-1)
                flat_ra, flat_dec = _convert_world_between_wcs(
                    flat_ra, flat_dec, obs_wcs, coadd_wcs
                )
                cx_flat, cy_flat = coadd_wcs.world_to_pixel(flat_ra, flat_dec)
                cx = cx_flat.reshape(obs_shape)
                cy = cy_flat.reshape(obs_shape)
                u = (cx / max(W_c - 1, 1)) * 2.0 - 1.0
                v = (cy / max(H_c - 1, 1)) * 2.0 - 1.0
                a_i = planes[idx, 0]
                bx_i = planes[idx, 1]
                by_i = planes[idx, 2]
                plane_obs = (a_i + bx_i * u + by_i * v) / float(s_eff)
                updated_obs.append(
                    dataclasses.replace(obs, data=img_dev - plane_obs, header=hdr)
                )
            else:
                updated_obs.append(obs)
        return updated_obs, {"mode": "planar", "planes": planes}

    corrections = corrector.fit_corrections(images, masks)
    updated_obs = []
    for obs, corr in zip(observations, corrections):
        if should_apply:
            img, hdr = load_observation_image_and_header(obs, device=read_dev)
            img_dev = img.to(device=map_device, dtype=torch.float32)
            s_eff = _effective_observation_photometric_scale(obs, hdr)
            corr_obs = blot(
                corr,
                target_hdr,
                target=obs,
                kernel="bilinear",
                semantics="surface_brightness",
                device=map_device,
            )
            assert isinstance(corr_obs, Tensor)
            updated_obs.append(
                dataclasses.replace(
                    obs, data=img_dev - corr_obs / float(s_eff), header=hdr
                )
            )
        else:
            updated_obs.append(obs)
    return updated_obs, {"mode": "mesh", "corrections": corrections}


def _resolve_tile_hw(
    tile_size: int | tuple[int, int] | None,
    spatial_tile_hw: tuple[int, int] | None,
    s_cfg: StackConfig,
) -> tuple[int, int] | None:
    raw = (
        tile_size
        if tile_size is not None
        else (spatial_tile_hw if spatial_tile_hw is not None else (s_cfg.tile_size if s_cfg.tile_size is not None else s_cfg.tile_hw))
    )
    if raw is None:
        return None
    if isinstance(raw, int):
        return (int(raw), int(raw))
    return (int(raw[0]), int(raw[1]))


def _apply_observation_astrometric_shifts(
    observations: Sequence[Observation],
    device: torch.device | str | None,
) -> list[Observation]:
    """Apply DCR and non-sidereal moving-target ephemeris offsets to observation headers."""
    from ..wcs.iau2006 import apply_dcr, apply_ephemeris_offset

    needs_shift = any(
        obs.dcr_zenith_deg is not None
        or (obs.tracking_rate is not None and obs.mjd_obs is not None and obs.reference_mjd is not None)
        for obs in observations
    )
    if not needs_shift:
        return list(observations)

    map_device = _resolve_map_device(device)
    read_dev = str(map_device)
    shifted: list[Observation] = []
    for obs in observations:
        if obs.dcr_zenith_deg is None and not (
            obs.tracking_rate is not None
            and obs.mjd_obs is not None
            and obs.reference_mjd is not None
        ):
            shifted.append(obs)
            continue
        img, hdr = load_observation_image_and_header(obs, device=read_dev)
        hdr_copy = dict(hdr)
        ra0 = torch.tensor(float(hdr_copy.get("CRVAL1", 0.0)), dtype=torch.float64)
        dec0 = torch.tensor(float(hdr_copy.get("CRVAL2", 0.0)), dtype=torch.float64)
        if obs.dcr_zenith_deg is not None and obs.wavelength is not None:
            ra0, dec0 = apply_dcr(
                ra0,
                dec0,
                float(obs.wavelength),
                zenith_angle_deg=float(obs.dcr_zenith_deg),
                parallactic_angle_deg=float(obs.dcr_parallactic_deg),
                reference_wavelength_angstrom=float(obs.dcr_ref_wavelength),
                pressure_hpa=float(obs.dcr_pressure_hpa),
                temperature_c=float(obs.dcr_temperature_c),
            )
        if (
            obs.tracking_rate is not None
            and obs.mjd_obs is not None
            and obs.reference_mjd is not None
        ):
            ra0, dec0 = apply_ephemeris_offset(
                ra0,
                dec0,
                d_ra_cosdec_arcsec_per_hr=float(obs.tracking_rate[0]),
                d_dec_arcsec_per_hr=float(obs.tracking_rate[1]),
                mjd_obs=float(obs.mjd_obs),
                mjd_ref=float(obs.reference_mjd),
            )
        hdr_copy["CRVAL1"] = float(ra0.item())
        hdr_copy["CRVAL2"] = float(dec0.item())
        shifted.append(
            dataclasses.replace(
                obs,
                data=img,
                header=hdr_copy,
                wcs=WCS(hdr_copy),
            )
        )
    return shifted


def coadd(
    observations: Sequence[Observation],
    target: Target | Mapping[str, Any] | WCS | None = None,
    *,
    resample: ResampleConfig | None = None,
    sky: SkyConfig | None = None,
    outliers: OutlierConfig | None = None,
    stack: StackConfig | None = None,
    kernel: MapResamplingKernel | None = None,
    pixfrac: float | None = None,
    semantics: MapValueSemantics | None = None,
    method: MapStackingMethod | None = None,
    clip_sigma: float | None = None,
    clip_iterations: int | None = None,
    footprint_order: int | None = None,
    tile_size: int | tuple[int, int] | None = None,
    spatial_tile_hw: tuple[int, int] | None = None,
    surface_brightness_area_scaling: bool | None = None,
    output_path: str | None = None,
    scratch_dir: str | Path | None = None,
    background_estimator: Callable[[Tensor, Tensor | None], Tensor] | None = None,
    cross_exposure_sky_corrector: CrossExposureSkyCorrector | None = None,
    device: torch.device | str | None = None,
) -> MapResult:
    """Unified 2D image-plane mosaic / coadd entry point.

    Accepts either structured configuration objects (``resample``, ``sky``, ``outliers``, ``stack``)
    or direct keyword arguments (``kernel``, ``pixfrac``, ``semantics``, ``method``, ``tile_size``).
    """
    r_cfg = resample or ResampleConfig()
    s_cfg = stack or StackConfig()
    resolved_kernel = kernel if kernel is not None else r_cfg.kernel
    resolved_pixfrac = pixfrac if pixfrac is not None else r_cfg.pixfrac
    resolved_semantics = semantics if semantics is not None else r_cfg.semantics
    resolved_method = method if method is not None else s_cfg.method
    resolved_clip_sigma = clip_sigma if clip_sigma is not None else s_cfg.clip_sigma
    resolved_clip_iter = (
        clip_iterations if clip_iterations is not None else s_cfg.clip_iterations
    )
    resolved_area_scaling = (
        surface_brightness_area_scaling
        if surface_brightness_area_scaling is not None
        else r_cfg.area_scaling
    )
    resolved_tile_hw = _resolve_tile_hw(tile_size, spatial_tile_hw, s_cfg)
    resolved_footprint_order = (
        footprint_order if footprint_order is not None else s_cfg.footprint_order
    )
    resolved_scratch = scratch_dir if scratch_dir is not None else s_cfg.scratch_dir

    bg_est = background_estimator
    sky_corr = cross_exposure_sky_corrector
    obs_to_stack = _apply_observation_astrometric_shifts(observations, device)
    target_header = _coerce_target_header(target)

    if sky is not None and sky.mode != "none" and sky.subtract:
        if sky.mode in ("per_exposure", "local"):
            bg_est = MeshSkyEstimator(
                block_size=sky.mesh_size,
                statistic=sky.stat,
                mesh_filter_size=sky.smooth_scale,
                self_mask_sigma=sky.clip_sigma,
            )
        elif (
            sky.mode in ("global", "global+match")
            or resolved_method not in ("weighted_mean", "mean", "median")
        ):
            obs_to_stack, _ = match_sky(
                obs_to_stack,
                target=target_header,
                config=sky,
                apply=True,
                device=device,
            )
        elif sky.mode in ("constant", "overlap_constant", "match"):
            sky_corr = CrossExposureSkyCorrector(
                mesh_size=sky.mesh_size,
                regularization=sky.regularization,
                mode="constant",
            )
        elif sky.mode in ("planar", "overlap_planar"):
            sky_corr = CrossExposureSkyCorrector(
                mesh_size=sky.mesh_size,
                regularization=sky.regularization,
                mode="planar",
            )
        elif sky.mode in ("mesh", "overlap_mesh"):
            sky_corr = CrossExposureSkyCorrector(
                mesh_size=sky.mesh_size,
                regularization=sky.regularization,
                mode="mesh",
            )
        else:
            raise ValueError(f"Unsupported SkyConfig.mode: {sky.mode!r}")

    if outliers is not None and outliers.enabled:
        init_coadd = coadd_image_plane(
            obs_to_stack,
            target_header=target_header,
            resampling_kernel="bilinear",
            value_semantics="surface_brightness",
            stacking_method="median" if len(obs_to_stack) >= 3 else "weighted_mean",
            device=device,
        )
        _, obs_to_stack = detect_outliers(
            obs_to_stack,
            init_coadd,
            config=outliers,
            update_masks=True,
            device=device,
        )

    result = coadd_image_plane(
        obs_to_stack,
        target_header=target_header,
        output_path=output_path,
        footprint_order=resolved_footprint_order,
        resampling_kernel=resolved_kernel,
        value_semantics=resolved_semantics,
        stacking_method=resolved_method,
        drizzle_pixfrac=resolved_pixfrac,
        clip_sigma=resolved_clip_sigma,
        clip_iterations=resolved_clip_iter,
        spatial_tile_hw=resolved_tile_hw,
        surface_brightness_area_scaling=resolved_area_scaling,
        device=device,
        scratch_dir=resolved_scratch,
        background_estimator=bg_est,
        cross_exposure_sky_corrector=sky_corr,
    )
    if resolved_method == "weighted_mean":
        vp, vr, vf = _propagate_observation_variance_components(
            obs_to_stack,
            target_header=result.header,
            target_shape=tuple(result.data.shape),
            resampling_kernel=resolved_kernel,
            drizzle_pixfrac=resolved_pixfrac,
            value_semantics=resolved_semantics,
            device=result.data.device,
        )
        if vp is not None or vr is not None or vf is not None:
            result = dataclasses.replace(
                result, var_poisson=vp, var_rnoise=vr, var_flat=vf
            )
    return result


def coadd_multiband(
    observations: Sequence[Observation],
    target: Target | Mapping[str, Any] | WCS | None = None,
    *,
    resample: ResampleConfig | None = None,
    stack: StackConfig | None = None,
    kernel: MapResamplingKernel | None = None,
    semantics: MapValueSemantics | None = None,
    method: MapStackingMethod | None = None,
    footprint_order: int | None = None,
    spectral_unit: str = "Angstrom",
    spectral_type: str = "WAVE",
    device: torch.device | str | None = None,
) -> MultibandResult:
    """Clean multi-band image-plane coadd entry point."""
    r_cfg = resample or ResampleConfig()
    s_cfg = stack or StackConfig()
    obs_shifted = _apply_observation_astrometric_shifts(observations, device)
    return coadd_image_plane_multiband(
        obs_shifted,
        target_header=_coerce_target_header(target),
        footprint_order=footprint_order
        if footprint_order is not None
        else s_cfg.footprint_order,
        resampling_kernel=kernel if kernel is not None else r_cfg.kernel,
        value_semantics=semantics if semantics is not None else r_cfg.semantics,
        stacking_method=method if method is not None else s_cfg.method,
        spectral_unit=spectral_unit,
        spectral_type=spectral_type,
        device=device,
    )


def aperture_uncertainty(
    observations: Sequence[Observation],
    result: MapResult,
    aperture_weights: Tensor,
    *,
    resample: ResampleConfig | None = None,
    kernel: MapResamplingKernel | None = None,
    pixfrac: float | None = None,
    semantics: MapValueSemantics | None = None,
    spatial_tile_hw: tuple[int, int] | None = None,
    surface_brightness_area_scaling: bool | None = None,
    device: torch.device | str | None = None,
) -> ApertureUncertainty:
    """Clean alias for ``evaluate_image_plane_aperture_uncertainty``."""
    r_cfg = resample or ResampleConfig()
    return evaluate_image_plane_aperture_uncertainty(
        observations,
        result,
        aperture_weights,
        resampling_kernel=kernel if kernel is not None else r_cfg.kernel,
        value_semantics=semantics if semantics is not None else r_cfg.semantics,
        drizzle_pixfrac=pixfrac if pixfrac is not None else r_cfg.pixfrac,
        spatial_tile_hw=spatial_tile_hw,
        surface_brightness_area_scaling=(
            surface_brightness_area_scaling
            if surface_brightness_area_scaling is not None
            else r_cfg.area_scaling
        ),
        device=device,
    )


