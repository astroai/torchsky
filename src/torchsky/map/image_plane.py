"""Image-plane-first mapmaking primitives."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Callable, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from ..device import default_compute_device
from ..io import read, write
from ..sphere.moc import MOC
from ..spectral import (
    DataCube,
    SpectralAxis,
    spectral_axis_from_header,
    spectral_bin_edges_from_header,
)
from ..wcs import WCS

from .fft_affine_ldu import resample_fft_ldu_affine_2d
from .fft_resample import resample_diagonal_affine_fft2d, resample_translation_fft2d
from .operators.background import SkyBackgroundOp
from .sky import CrossExposureSkyCorrector
from .operators.bundle import ObservationOperatorBundle
from .operators.geometry import ForwardSplatGeometryOp
from .operators.config import (
    MapResamplingKernel,
    MapStackingMethod,
    MapValueSemantics,
    validate_map_resampling_kernel,
    validate_map_stacking_method,
    validate_map_value_semantics,
)
from .operators.execution import (
    accumulate_projected_observation,
    finalize_projected_stack,
    is_plain_tan_header,
    masked_median_stack,
    prepare_image_observation_data,
    project_image_observation_geometry,
    stack_inputs_from_projected_observation,
    wcs_has_distortion,
)
from .operators.geometry import InterpolationGeometryOp
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
from .windowing import (
    _prepare_target,
    _window_large_tan_target,
    _weight_map_to_moc,
    _subtarget_header_for_window,
    _PreparedImagePlaneTarget,
    _LARGE_TAN_WINDOWING_MIN_TARGET_PIXELS,  # noqa: F401
    _LARGE_TAN_WINDOWING_EDGE_SAMPLES,  # noqa: F401
    _LARGE_TAN_WINDOWING_MARGIN_PIXELS,  # noqa: F401
    _build_target_world_grid,  # noqa: F401
    _quadrature_target_world_grids,
)
from .operators.radiometry import apply_image_observation_calibration
from .operators.resampling import (
    finalize_periodic_resampled_weighted,
    infer_diagonal_scale_shift_pixel_map,
    infer_fft_translation_shift_yx,
    infer_linear_affine_pixel_map,
    sample_image_plane_batched_channels,
)
from .operators.spectral import ExplicitSpectralResponseOp
from .periodic_affine import resample_linear_affine_periodic_bilinear_2d
from .spectral_affine import resample_linear_affine_spectral_2d
from .spectral_nufft import resample_nonuniform_spectral_fast_2d
from .spectral_nonuniform import resample_nonuniform_spectral_2d

# Header probe for area scaling: recoverable failures when reading metadata only.
from .io_utils import (
    read_image_plane_observation,
    _observation_header_for_area_scaling,
)

# Module-level cache for observation reads (persists across coadd_image_plane calls,
# notably for the spatial-tiled path which calls coadd_image_plane per tile).
_obs_image_cache: dict[tuple[str, int | str | None], tuple[Tensor, dict]] = {}


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
class ImageObservation:
    path: str
    hdu: int | str | None = None
    weight: float = 1.0
    photometric_scale: float = 1.0
    background_offset: float = 0.0
    wavelength: float | None = None
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


@dataclass
class MapResult:
    data: Tensor
    weight: Tensor
    header: dict
    overlap_count: Tensor | None = None
    coverage_moc: MOC | None = None
    observation_mocs: tuple[MOC, ...] | None = None


@dataclass
class MultiBandMapResult:
    spectral_axis: SpectralAxis
    data: Tensor
    weight: Tensor
    header: dict
    overlap_count: Tensor | None = None
    coverage_mocs: tuple[MOC | None, ...] | None = None
    band_observation_counts: Tensor | None = None

    @property
    def wavelength(self) -> Tensor:
        return self.spectral_axis.to_wavelength()


@dataclass
class PlannedImagePlaneTarget:
    header: dict
    shape: tuple[int, int]


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


@dataclass
class SpectralCubeMapResult:
    spectral_axis: SpectralAxis
    band_metadata: BandMetadataTable
    data: Tensor
    weight: Tensor
    header: dict
    spatial_wcs: WCS
    overlap_count: Tensor | None = None
    coverage_mocs: tuple[MOC | None, ...] | None = None

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


@dataclass
class _ProjectedImagePlaneObservation:
    numer: Tensor
    weight: Tensor
    overlap: Tensor
    moc: MOC | None
    target_window: tuple[int, int, int, int] | None = None


_DRIZZLE_FORWARD_CHUNK_PIXELS = 262_144

_DEFAULT_MAP_OPERATOR_BUNDLE = ObservationOperatorBundle(
    geometry=InterpolationGeometryOp(),
    spectral_response=ExplicitSpectralResponseOp(),
    radiometry=None,
    background=SkyBackgroundOp(),
    mask_weight=None,
    noise=ExplicitNoiseOp(),
)


def _try_external_drizzle_helpers():
    """Legacy comparator path only; set ``TORCHSKY_USE_EXTERNAL_DRIZZLE=1`` to enable."""
    if os.environ.get("TORCHSKY_USE_EXTERNAL_DRIZZLE", "").lower() not in (
        "1",
        "true",
        "yes",
    ):
        return None
    try:
        from astropy.io import fits as astropy_fits  # type: ignore
        from astropy.wcs import WCS as AstropyWCS  # type: ignore
        from drizzle.resample import Drizzle as DrizzleResampler  # type: ignore
        from drizzle.utils import calc_pixmap, estimate_pixel_scale_ratio  # type: ignore
    except (ImportError, ModuleNotFoundError, AttributeError, OSError):
        return None
    return (
        astropy_fits,
        AstropyWCS,
        DrizzleResampler,
        calc_pixmap,
        estimate_pixel_scale_ratio,
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


def _is_drizzle_kernel(kernel: MapResamplingKernel) -> bool:
    return kernel in {"drizzle_point", "drizzle_square"}


def _is_conservative_kernel(kernel: MapResamplingKernel) -> bool:
    return kernel == "conservative_polygon"


def _is_splat_kernel(kernel: MapResamplingKernel) -> bool:
    return kernel in {"splat_point", "splat_box"}

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


def _coarse_grid_shape(height: int, width: int, *, coarse_stride: int = 128) -> tuple[int, int, int]:
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

    yy, xx = torch.meshgrid(y_coarse, x_coarse, indexing='ij')
    ra_c, dec_c = target_wcs.pixel_to_world(xx, yy)
    x_src_c, y_src_c = source_wcs.world_to_pixel(ra_c, dec_c)

    if stride == 1:
        x_out = x_src_c.reshape(height, width).to(dtype=torch.float32)
        y_out = y_src_c.reshape(height, width).to(dtype=torch.float32)
        return x_out, y_out

    hc, wc = int(y_coarse.numel()), int(x_coarse.numel())
    coarse = torch.stack([x_src_c, y_src_c], dim=0).to(dtype=torch.float32).reshape(1, 2, hc, wc)

    y_g = torch.arange(height, dtype=torch.float32, device=device) / stride
    x_g = torch.arange(width, dtype=torch.float32, device=device) / stride

    # Bicubic grid_sample expects one row/col of margin for boundary handling.
    # The last coarse pixel has index stride*(Nc-1). Pixels beyond that have
    # normalized coordinate > 1 and are clamped to border.
    y_n = y_g / max(hc - 1, 1) * 2.0 - 1.0
    x_n = x_g / max(wc - 1, 1) * 2.0 - 1.0
    gy, gx = torch.meshgrid(y_n.clamp(-1.0, 1.0), x_n.clamp(-1.0, 1.0), indexing='ij')
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)

    full = F.grid_sample(coarse, grid, mode='bicubic', padding_mode='border', align_corners=True)
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


def _apply_observation_calibration(
    obs: ImageObservation,
    image: Tensor,
    source_weight: Tensor,
) -> tuple[Tensor, Tensor]:
    return apply_image_observation_calibration(
        photometric_scale=obs.photometric_scale,
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
) -> tuple[Tensor, Tensor]:
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
    elif target_representative_area_sr is not None:
        source_area_sr = representative_pixel_area_sr(
            source_wcs,
            tuple(image.shape),
            device=image.device,
        )
        info_density = source_weight / max(source_area_sr, 1e-20)
        weighted_image = image * info_density
        weighted_weight = info_density
    else:
        weighted_image = image * source_weight
        weighted_weight = source_weight
    support_image = (
        (source_weight > 0.0).to(dtype=torch.float32)
        if value_semantics == "flux"
        else None
    )

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
        return finalize_periodic_resampled_weighted(
            sampled_numer,
            sampled_weight,
            sampled_support,
            value_semantics=value_semantics,
            target_representative_area_sr=target_representative_area_sr,
        )

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
        return finalize_periodic_resampled_weighted(
            sampled_numer,
            sampled_weight,
            sampled_support,
            value_semantics=value_semantics,
            target_representative_area_sr=target_representative_area_sr,
        )

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
        return finalize_periodic_resampled_weighted(
            sampled_numer,
            sampled_weight,
            sampled_support,
            value_semantics=value_semantics,
            target_representative_area_sr=target_representative_area_sr,
        )

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
        return finalize_periodic_resampled_weighted(
            sampled_numer,
            sampled_weight,
            sampled_support,
            value_semantics=value_semantics,
            target_representative_area_sr=target_representative_area_sr,
        )

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
        return finalize_periodic_resampled_weighted(
            sampled_numer,
            sampled_weight,
            sampled_support,
            value_semantics=value_semantics,
            target_representative_area_sr=target_representative_area_sr,
        )

    def _sample_on_source_grid(
        x_src: Tensor,
        y_src: Tensor,
        *,
        flat_ra: Tensor | None = None,
        flat_dec: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
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
        if (
            value_semantics == "surface_brightness"
            and target_representative_area_sr is not None
        ):
            sampled_numer = sampled_numer * float(target_representative_area_sr)
            sampled_weight = sampled_weight * float(target_representative_area_sr)
        return sampled_numer, sampled_weight

    def _sample_on_world_grid(ra_deg: Tensor, dec_deg: Tensor) -> tuple[Tensor, Tensor]:
        flat_ra = ra_deg.reshape(-1)
        flat_dec = dec_deg.reshape(-1)
        if (
            bool(getattr(source_wcs, "_is_tpv", False))
            and getattr(source_wcs, "sip", None) is None
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
                return zeros, zeros
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
        if min(h_tgt, w_tgt) < 500:
            ra_g, dec_g = _build_target_world_grid(
                target.wcs, h_tgt, w_tgt, device=image.device,
            )
            return _sample_on_world_grid(ra_g, dec_g)
        x_src, y_src = _interpolate_forward_mapping_from_coarse(
            source_wcs, target.wcs, h_tgt, w_tgt, device=image.device,
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
    x_src_all, y_src_all = source_wcs.world_to_pixel(flat_ra, flat_dec)
    x_src_b = x_src_all.reshape(len(quadrature_world_grids), *target.shape)
    y_src_b = y_src_all.reshape(len(quadrature_world_grids), *target.shape)
    sub_numer, sub_weight = _sample_on_source_grid(
        x_src_b,
        y_src_b,
        flat_ra=flat_ra,
        flat_dec=flat_dec,
    )
    accum_numer = sub_numer.mean(dim=0)
    accum_weight = sub_weight.mean(dim=0)
    accum_numer = torch.where(
        accum_weight > 0.0,
        accum_numer,
        torch.zeros_like(accum_numer),
    )
    return accum_numer, accum_weight


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


def _forward_project_weighted_to_target(
    image: Tensor,
    source_weight: Tensor,
    source_wcs: WCS,
    target: _PreparedImagePlaneTarget,
    *,
    resampling_kernel: MapResamplingKernel,
    value_semantics: MapValueSemantics,
    drizzle_pixfrac: float,
) -> tuple[Tensor, Tensor]:
    if not _is_splat_kernel(resampling_kernel):
        raise ValueError(
            f"forward projection only supports splat kernels, got {resampling_kernel!r}"
        )

    image = validate_image_plane_2d(image).to(dtype=torch.float32)
    source_weight = validate_image_plane_2d(source_weight).to(dtype=torch.float32)
    h_src, w_src = image.shape

    if value_semantics == "flux":
        source_area_sr = pixel_area_sr_map(
            source_wcs,
            tuple(image.shape),
            device=image.device,
        )
        weighted_image = image * source_weight * source_area_sr
        weighted_weight = source_weight * (source_area_sr * source_area_sr)
    elif target.representative_pixel_area_sr is not None:
        source_area_sr = representative_pixel_area_sr(
            source_wcs,
            tuple(image.shape),
            device=image.device,
        )
        info_density = source_weight / max(source_area_sr, 1e-20)
        weighted_image = image * info_density
        weighted_weight = info_density
    else:
        weighted_image = image * source_weight
        weighted_weight = source_weight

    valid = torch.isfinite(weighted_weight) & (weighted_weight > 0.0)
    if not bool(valid.any()):
        zeros = torch.zeros(target.shape, dtype=torch.float32, device=image.device)
        return zeros, zeros

    ys, xs = torch.nonzero(valid, as_tuple=True)
    flat_values_numer = weighted_image[ys, xs].to(dtype=torch.float32)
    flat_values_weight = weighted_weight[ys, xs].to(dtype=torch.float32)
    flat_numer = torch.zeros(
        target.shape[0] * target.shape[1], dtype=torch.float32, device=image.device
    )
    flat_weight = torch.zeros_like(flat_numer)
    width = int(target.shape[1])
    height = int(target.shape[0])

    for start in range(0, int(xs.numel()), _DRIZZLE_FORWARD_CHUNK_PIXELS):
        end = min(start + _DRIZZLE_FORWARD_CHUNK_PIXELS, int(xs.numel()))
        x_chunk = xs[start:end].to(dtype=torch.float64)
        y_chunk = ys[start:end].to(dtype=torch.float64)
        numer_chunk = flat_values_numer[start:end]
        weight_chunk = flat_values_weight[start:end]

        ra_deg, dec_deg = source_wcs.pixel_to_world(x_chunk, y_chunk)
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

        if resampling_kernel == "splat_point":
            ix = torch.round(x_tgt).to(dtype=torch.int64)
            iy = torch.round(y_tgt).to(dtype=torch.int64)
            inside = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
            if not bool(inside.any()):
                continue
            idx = iy[inside] * width + ix[inside]
            flat_numer.scatter_add_(0, idx, numer_chunk[inside])
            flat_weight.scatter_add_(0, idx, weight_chunk[inside])
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

    numer = flat_numer.reshape(target.shape)
    weight = flat_weight.reshape(target.shape)
    numer = torch.where(weight > 0.0, numer, torch.zeros_like(numer))
    weight = torch.where(weight > 0.0, weight, torch.zeros_like(weight))
    return numer, weight


def _as_astropy_fits_header_for_drizzle(header: dict):
    helpers = _try_external_drizzle_helpers()
    if helpers is None:  # pragma: no cover - exercised only when drizzle is installed
        raise RuntimeError("external drizzle helpers are unavailable")
    astropy_fits = helpers[0]
    out = astropy_fits.Header()
    for key, value in header.items():
        if str(key).endswith("_COMMENT"):
            continue
        if key in {"", "COMMENT", "HISTORY"}:
            continue
        out[key] = value
    return out


def _maybe_external_drizzle_resample_weighted_to_target(
    image: Tensor,
    source_weight: Tensor,
    *,
    header_dict_obs: dict,
    target_header: dict,
    target_shape: tuple[int, int],
    resampling_kernel: MapResamplingKernel,
    value_semantics: MapValueSemantics,
    drizzle_pixfrac: float,
) -> tuple[Tensor, Tensor] | None:
    if not _is_drizzle_kernel(resampling_kernel):
        return None
    helpers = _try_external_drizzle_helpers()
    if helpers is None:
        return None
    (
        astropy_fits,
        AstropyWCS,
        DrizzleResampler,
        calc_pixmap,
        estimate_pixel_scale_ratio,
    ) = helpers
    kernel = "point" if resampling_kernel == "drizzle_point" else "square"
    target_header_astropy = _as_astropy_fits_header_for_drizzle(target_header)
    source_header_astropy = _as_astropy_fits_header_for_drizzle(header_dict_obs)
    target_wcs = AstropyWCS(target_header_astropy)
    target_wcs.array_shape = target_shape
    source_shape = tuple(image.shape)
    source_wcs = AstropyWCS(source_header_astropy)
    source_wcs.array_shape = source_shape
    pixmap = calc_pixmap(source_wcs, target_wcs, shape=source_shape)
    pscale_ratio = estimate_pixel_scale_ratio(
        source_wcs,
        target_wcs,
        refpix_from=source_wcs.wcs.crpix,
        refpix_to=target_wcs.wcs.crpix,
    )
    # Match ``benchmarks/bench_map_smacs0723_reproduce.py`` / compat drizzle rows:
    # surface-brightness products use unit iscale; flux-style stacking uses pscale^2.
    iscale = (
        1.0 if value_semantics == "surface_brightness" else float(pscale_ratio) ** 2
    )
    drizzle = DrizzleResampler(
        kernel=kernel,
        fillval=0.0,
        out_shape=target_shape,
        disable_ctx=True,
    )
    drizzle.add_image(
        torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        .cpu()
        .numpy()
        .astype("float32"),
        exptime=1.0,
        pixmap=pixmap,
        weight_map=torch.nan_to_num(source_weight, nan=0.0, posinf=0.0, neginf=0.0)
        .cpu()
        .numpy()
        .astype("float32"),
        iscale=iscale,
        pixel_scale_ratio=pscale_ratio,
        pixfrac=float(drizzle_pixfrac),
        in_units="cps",
    )
    out_img = torch.as_tensor(drizzle.out_img, dtype=torch.float32, device=image.device)
    out_wht = torch.as_tensor(drizzle.out_wht, dtype=torch.float32, device=image.device)
    if value_semantics == "flux":
        # drizzle leaves out_wht in input-weight units even when iscale changes
        # the science values; convert it to inverse variance in output units.
        out_wht = out_wht / float(iscale * iscale)
    out_img = torch.where(out_wht > 0.0, out_img, torch.zeros_like(out_img))
    return out_img * out_wht, out_wht


def _validated_spectral_axis(axis: SpectralAxis) -> SpectralAxis:
    values = axis.values.to(dtype=torch.float64)
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("Spectral axis must be one-dimensional and non-empty")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Spectral axis values must be finite")
    if values.numel() > 1 and not bool(torch.all(values[1:] > values[:-1])):
        raise ValueError("Spectral axis values must be strictly increasing")
    return SpectralAxis(
        values=values,
        unit=axis.unit,
        type=axis.type,
        rest_frequency=axis.rest_frequency,
        redshift=axis.redshift,
    )


def _observation_spectral_columns(
    observations: Sequence[ImageObservation],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, tuple[str | None, ...]]:
    centers: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    filter_center: list[float] = []
    filter_sigma: list[float] = []
    names: list[str | None] = []
    for obs in observations:
        if obs.wavelength is None:
            raise ValueError(
                "All observations must define wavelength for spectral cube coaddition"
            )
        center = float(obs.wavelength)
        if not torch.isfinite(torch.tensor(center, dtype=torch.float64)):
            raise ValueError("Observation wavelengths must be finite")
        if (obs.spectral_min is None) != (obs.spectral_max is None):
            raise ValueError(
                "Observations must define both spectral_min and spectral_max together"
            )
        band_lower = center if obs.spectral_min is None else float(obs.spectral_min)
        band_upper = center if obs.spectral_max is None else float(obs.spectral_max)
        if not torch.isfinite(
            torch.tensor(band_lower, dtype=torch.float64)
        ) or not torch.isfinite(torch.tensor(band_upper, dtype=torch.float64)):
            raise ValueError("Observation spectral bounds must be finite")
        if band_upper < band_lower:
            raise ValueError("Observation spectral_max must be >= spectral_min")
        filt_center = (
            center
            if obs.transmission_center is None
            else float(obs.transmission_center)
        )
        if not torch.isfinite(torch.tensor(filt_center, dtype=torch.float64)):
            raise ValueError("Observation transmission_center must be finite")
        filt_sigma = (
            float("nan")
            if obs.transmission_sigma is None
            else float(obs.transmission_sigma)
        )
        if obs.transmission_sigma is not None:
            if (
                not torch.isfinite(torch.tensor(filt_sigma, dtype=torch.float64))
                or filt_sigma <= 0.0
            ):
                raise ValueError(
                    "Observation transmission_sigma must be finite and > 0"
                )
        centers.append(center)
        lower.append(band_lower)
        upper.append(band_upper)
        filter_center.append(filt_center)
        filter_sigma.append(filt_sigma)
        names.append(obs.band_name)
    return (
        torch.tensor(centers, dtype=torch.float64),
        torch.tensor(lower, dtype=torch.float64),
        torch.tensor(upper, dtype=torch.float64),
        torch.tensor(filter_center, dtype=torch.float64),
        torch.tensor(filter_sigma, dtype=torch.float64),
        tuple(names),
    )


def _validated_spectral_bin_edges(edges: Tensor, *, n_band: int) -> Tensor:
    values = edges.to(dtype=torch.float64)
    if values.ndim != 1 or values.numel() != n_band + 1:
        raise ValueError("spectral_bin_edges must have length n_band + 1")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("spectral_bin_edges must be finite")
    if not bool(torch.all(values[1:] > values[:-1])):
        raise ValueError("spectral_bin_edges must be strictly increasing")
    return values


def _normal_cdf(x: Tensor) -> Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _build_gaussian_filter_projection(
    *,
    lower: Tensor,
    upper: Tensor,
    center: float,
    sigma: float,
) -> Tensor:
    widths = max(float(sigma), torch.finfo(torch.float64).tiny)
    z_upper = (upper - float(center)) / widths
    z_lower = (lower - float(center)) / widths
    weights = _normal_cdf(z_upper) - _normal_cdf(z_lower)
    weights = torch.clamp(weights, min=0.0)
    row_sum = weights.sum()
    return weights / torch.clamp(row_sum, min=torch.finfo(weights.dtype).tiny)


def _build_spectral_projection(
    observation_center: Tensor,
    observation_lower: Tensor,
    observation_upper: Tensor,
    observation_filter_center: Tensor,
    observation_filter_sigma: Tensor,
    *,
    target_axis: SpectralAxis,
    target_edges: Tensor | None,
) -> Tensor:
    n_obs = int(observation_center.numel())
    n_band = int(target_axis.values.numel())
    projection = torch.zeros((n_obs, n_band), dtype=torch.float64)
    target_values = target_axis.values.to(dtype=torch.float64)

    if target_edges is None:
        for idx in range(n_obs):
            matches = target_values == observation_center[idx]
            if int(matches.sum().item()) != 1:
                raise ValueError(
                    "Observation wavelengths must match the planned spectral axis exactly; "
                    "provide spectral_bin_edges for interval reprojection"
                )
            projection[idx, int(torch.nonzero(matches, as_tuple=False).item())] = 1.0
        return projection

    left = target_edges[:-1]
    right = target_edges[1:]
    widths = observation_upper - observation_lower

    for idx in range(n_obs):
        if torch.isfinite(observation_filter_sigma[idx]):
            effective_left = torch.maximum(left, observation_lower[idx])
            effective_right = torch.minimum(right, observation_upper[idx])
            positive = effective_right > effective_left
            if bool(positive.any()):
                projection[idx] = _build_gaussian_filter_projection(
                    lower=effective_left,
                    upper=effective_right,
                    center=float(observation_filter_center[idx].item()),
                    sigma=float(observation_filter_sigma[idx].item()),
                )
        elif widths[idx] > 0.0:
            overlap = torch.clamp(
                torch.minimum(right, observation_upper[idx])
                - torch.maximum(left, observation_lower[idx]),
                min=0.0,
            )
            projection[idx] = overlap / widths[idx]
        else:
            center = observation_center[idx]
            in_bin = (center >= left) & (center < right)
            if center == right[-1]:
                in_bin[-1] = True
            projection[idx, in_bin] = 1.0
        if not bool((projection[idx] > 0.0).any()):
            raise ValueError(
                "Observation spectral support does not overlap the planned spectral grid"
            )

    return projection


def _band_names_from_projection(
    projection: Tensor,
    observation_names: tuple[str | None, ...],
) -> tuple[str | None, ...]:
    names: list[str | None] = []
    for band_idx in range(int(projection.shape[1])):
        band_names = {
            observation_names[obs_idx]
            for obs_idx in range(int(projection.shape[0]))
            if projection[obs_idx, band_idx] > 0.0
            and observation_names[obs_idx] is not None
        }
        names.append(next(iter(band_names)) if len(band_names) == 1 else None)
    return tuple(names)


def _build_band_metadata(
    spectral_axis: SpectralAxis,
    spectral_bin_edges: Tensor | None,
    projection: Tensor,
    observation_names: tuple[str | None, ...],
) -> BandMetadataTable:
    centers = spectral_axis.values.to(dtype=torch.float64)
    if spectral_bin_edges is None:
        lower = centers.clone()
        upper = centers.clone()
    else:
        lower = spectral_bin_edges[:-1].clone()
        upper = spectral_bin_edges[1:].clone()
    return BandMetadataTable(
        band_index=torch.arange(int(centers.numel()), dtype=torch.int64),
        center=centers,
        lower=lower,
        upper=upper,
        observation_count=(projection > 0.0).sum(dim=0).to(torch.int64),
        spectral_weight_sum=projection.sum(dim=0),
        band_names=_band_names_from_projection(projection, observation_names),
    )


def _project_observation_to_target_chunk(
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
    drizzle_pixfrac: float,
    background_estimator: Callable[[Tensor, Tensor | None], Tensor] | None = None,
) -> _ProjectedImagePlaneObservation:
    if not torch.isfinite(torch.tensor(obs.weight, dtype=torch.float64)):
        raise ValueError("Observation weights must be finite")
    if not (obs.weight > 0.0):
        raise ValueError("Observation weights must be positive")

    if image is None or header_dict_obs is None:
        image, header_dict_obs = read_image_plane_observation(
            obs.path, hdu=obs.hdu, device=str(device)
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
    image = prepared_observation.image
    source_weight = prepared_observation.source_weight
    operator_bundle = prepared_observation.operator_bundle
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
        else:
            _kernel = resampling_kernel
            if isinstance(operator_bundle.geometry, ForwardSplatGeometryOp):
                _forward_remap = _splat_for_interpolation(resampling_kernel)
                if _forward_remap is not None:
                    src_pix = int(image.shape[0]) * int(image.shape[1])
                    tgt_pix = (
                        int(prepared_target_view.shape[0])
                        * int(prepared_target_view.shape[1])
                    )
                    if src_pix < tgt_pix * _FORWARD_SOURCE_RATIO:
                        _kernel = _forward_remap
            sampled_numer, sampled_weight = project_image_observation_geometry(
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
        cos_dec0 * sin_dec - sin_dec0 * cos_dec * torch.cos(dra)
    ) / denom * radians_to_degrees
    if bool(torch.any(~torch.isfinite(xi))) or bool(torch.any(~torch.isfinite(eta))):
        raise RuntimeError("Union footprint produced non-finite TAN coordinates")
    return xi, eta


def _plan_union_footprint_target(
    observations: Sequence[ImageObservation],
    *,
    device: torch.device | str | None = None,
    padding: float = 1.05,
) -> tuple[dict, tuple[int, int]]:
    """Plan one rotation- and RA-seam-safe TAN grid for all observations."""
    if not math.isfinite(float(padding)) or float(padding) < 1.0:
        raise ValueError("footprint_padding must be finite and at least 1.0")
    footprints: list[Tensor] = []
    pixel_scales: list[float] = []
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
        if footprint.shape != (4, 2) or not bool(
            torch.all(torch.isfinite(footprint))
        ):
            raise RuntimeError(
                f"Could not determine finite footprint for {observation.path}"
            )
        determinant = torch.linalg.det(
            wcs.cd.detach().to(device="cpu", dtype=torch.float64)
        )
        pixel_scale = math.sqrt(abs(float(determinant)))
        if not math.isfinite(pixel_scale) or pixel_scale <= 0.0:
            raise RuntimeError(
                f"Could not determine a non-singular spatial pixel scale for {observation.path}"
            )
        footprints.append(footprint)
        pixel_scales.append(pixel_scale)
    if not footprints:
        raise RuntimeError("Could not determine any observation footprints")

    footprint = torch.cat(footprints, dim=0)
    center_ra, center_dec = _spherical_union_center(footprint)
    xi, eta = _world_to_tangent_plane(
        footprint,
        center_ra_deg=center_ra,
        center_dec_deg=center_dec,
    )
    pixel_scale = float(torch.tensor(pixel_scales, dtype=torch.float64).median())
    xi_mid = 0.5 * float(xi.min() + xi.max())
    eta_mid = 0.5 * float(eta.min() + eta.max())
    xi_half = 0.5 * float(xi.max() - xi.min()) * float(padding)
    eta_half = 0.5 * float(eta.max() - eta.min()) * float(padding)
    xi_min, xi_max = xi_mid - xi_half, xi_mid + xi_half
    eta_min, eta_max = eta_mid - eta_half, eta_mid + eta_half
    naxis1 = max(1, int(math.ceil((xi_max - xi_min) / pixel_scale)))
    naxis2 = max(1, int(math.ceil((eta_max - eta_min) / pixel_scale)))
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
        "CRPIX1": 0.5 + xi_max / pixel_scale,
        "CRPIX2": 0.5 - eta_min / pixel_scale,
        "CD1_1": -pixel_scale,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": pixel_scale,
    }
    return header, (naxis2, naxis1)


def plan_image_plane_target(
    observations: Sequence[ImageObservation],
    *,
    reference_index: int | None = 0,
    header_overrides: Optional[dict] = None,
    device: torch.device | str | None = None,
    union_footprint: bool = False,
    footprint_padding: float = 1.05,
    max_footprint_pixels: int | None = None,
) -> PlannedImagePlaneTarget:
    if not observations:
        raise ValueError("At least one observation is required")
    if max_footprint_pixels is not None and int(max_footprint_pixels) <= 0:
        raise ValueError("max_footprint_pixels must be positive")

    if union_footprint or reference_index is None:
        header, shape = _plan_union_footprint_target(
            observations,
            device=device,
            padding=footprint_padding,
        )
        if max_footprint_pixels is not None and shape[0] * shape[1] > max_footprint_pixels:
            raise RuntimeError(
                f"Union footprint output {shape[0]}×{shape[1]} "
                f"({shape[0] * shape[1] / 1e6:.1f} MPix) exceeds --max-footprint-pixels="
                f"{max_footprint_pixels}"
            )
        final_shape = target_shape_from_header(
            header_overrides or {},
            fallback=shape,
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
    image, header = read(ref.path, hdu=ref.hdu, return_header=True, device=read_dev)
    image = validate_image_plane_2d(image)
    shape = tuple(image.shape)
    shape = target_shape_from_header(header_overrides or {}, fallback=shape)
    normalized_header = normalized_header_for_shape(
        dict(header),
        shape,
        overrides=header_overrides,
    )
    return PlannedImagePlaneTarget(header=normalized_header, shape=shape)


def plan_image_plane_cube_target(
    observations: Sequence[ImageObservation],
    *,
    reference_index: int = 0,
    header_overrides: Optional[dict] = None,
    spectral_axis: SpectralAxis | None = None,
    spectral_bin_edges: Tensor | None = None,
    spectral_unit: str = "Angstrom",
    spectral_type: str = "WAVE",
    device: torch.device | str | None = None,
    union_footprint: bool = False,
    footprint_padding: float = 1.05,
    max_footprint_pixels: int | None = None,
) -> PlannedImagePlaneCubeTarget:
    spatial_target = plan_image_plane_target(
        observations,
        reference_index=reference_index,
        header_overrides=header_overrides,
        device=device,
        union_footprint=union_footprint,
        footprint_padding=footprint_padding,
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
    reference_index: int = 0,
    header_overrides: Optional[dict] = None,
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


def _coadd_image_plane_spatial_tiles(
    observations: Sequence[ImageObservation],
    *,
    header_dict: dict,
    target_shape: tuple[int, int],
    tile_height: int,
    tile_width: int,
    output_path: Optional[str],
    resampling_kernel: MapResamplingKernel,
    value_semantics: MapValueSemantics,
    stacking_method: MapStackingMethod,
    drizzle_pixfrac: float,
    map_device: torch.device,
    background_estimator: Callable[[Tensor, Tensor | None], Tensor] | None = None,
    cross_exposure_sky_corrector: CrossExposureSkyCorrector | None = None,
) -> MapResult:
    """Accumulate a large map in non-overlapping spatial tiles to cap per-tile memory."""
    if not observations:
        raise ValueError("At least one observation is required")
    H, W = int(target_shape[0]), int(target_shape[1])
    th = max(int(tile_height), 1)
    tw = max(int(tile_width), 1)
    first = observations[0]
    read_dev = str(map_device)
    _first_key = (first.path, first.hdu)
    if _first_key in _obs_image_cache:
        first_image, first_header = _obs_image_cache[_first_key]
        first_image = validate_image_plane_2d(first_image)
    else:
        first_image, first_header = read(
            first.path, hdu=first.hdu, return_header=True, device=read_dev
        )
        first_image = validate_image_plane_2d(first_image)
        first_header = dict(first_header)
        _obs_image_cache[_first_key] = (first_image, first_header)
    device = map_device
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
    full_data = torch.zeros((H, W), dtype=torch.float32, device=device)
    full_weight = torch.zeros((H, W), dtype=torch.float32, device=device)
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
                spatial_tile_hw=None,
                surface_brightness_area_scaling=mosaic_surface_brightness_area_scaling,
                device=map_device,
                background_estimator=background_estimator,
                cross_exposure_sky_corrector=cross_exposure_sky_corrector,
            )
            full_data[y0:y1, x0:x1] = sub.data.to(device=device)
            full_weight[y0:y1, x0:x1] = sub.weight.to(device=device)
            if sub.overlap_count is not None:
                full_overlap[y0:y1, x0:x1] = sub.overlap_count.to(device=device)
    result = MapResult(
        data=full_data,
        weight=full_weight,
        header=header_dict,
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
    target_header: Optional[dict] = None,
    output_path: Optional[str] = None,
    footprint_order: int | None = None,
    resampling_kernel: MapResamplingKernel = "bilinear",
    value_semantics: MapValueSemantics = "surface_brightness",
    stacking_method: MapStackingMethod = "weighted_mean",
    drizzle_pixfrac: float = 1.0,
    spatial_tile_hw: tuple[int, int] | None = None,
    surface_brightness_area_scaling: bool | None = None,
    device: torch.device | str | None = None,
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

    map_device = _resolve_map_device(device)
    read_dev = str(map_device)
    first = observations[0]
    _first_key = (first.path, first.hdu)
    if _first_key in _obs_image_cache:
        first_image, first_header = _obs_image_cache[_first_key]
        first_image = validate_image_plane_2d(first_image)
    else:
        first_image, first_header = read(
            first.path, hdu=first.hdu, return_header=True, device=read_dev
        )
        first_image = validate_image_plane_2d(first_image)
        first_header = dict(first_header)
        _obs_image_cache[_first_key] = (first_image, first_header)
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
    if spatial_tile_hw is not None:
        if footprint_order is not None:
            raise ValueError(
                "spatial_tile_hw is incompatible with footprint_order=... "
                "(MOC accumulation is not defined per tile); omit one of them"
            )
        return _coadd_image_plane_spatial_tiles(
            observations,
            header_dict=header_dict,
            target_shape=target_shape,
            tile_height=int(spatial_tile_hw[0]),
            tile_width=int(spatial_tile_hw[1]),
            output_path=output_path,
            resampling_kernel=resampling_kernel,
            value_semantics=value_semantics,
            stacking_method=stacking_method,
            drizzle_pixfrac=drizzle_pixfrac,
            map_device=map_device,
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
    overlap_count = torch.zeros(target_shape, dtype=torch.int32, device=device)
    observation_mocs: list[MOC] | None = [] if footprint_order is not None else None
    median_images: list[Tensor] | None = [] if stacking_method == "median" else None
    median_support: list[Tensor] | None = [] if stacking_method == "median" else None
    projected_store: list[dict] | None = (
        [] if cross_exposure_sky_corrector is not None and stacking_method != "median"
        else None
    )

    # Batch MEF reading: group observations by path, read files with ≥2 HDUs once
    # into the module-level _obs_image_cache (persists across tiled calls).
    from collections import defaultdict as _defaultdict
    _path_hdus: dict[str, list[int | str | None]] = _defaultdict(list)
    for _obs in observations:
        _path_hdus[_obs.path].append(_obs.hdu)
    for _path, _hdus in _path_hdus.items():
        if len(_hdus) > 1:
            _hdu_set = sorted({h for h in _hdus if isinstance(h, int)})
            _hdu_uncached = [h for h in _hdu_set if (_path, h) not in _obs_image_cache]
            if _hdu_uncached:
                try:
                    from torchfits import read_hdus as _read_hdus
                    _imgs, _hdrs = _read_hdus(_path, _hdu_uncached, device=read_dev, return_header=True)
                    for _h, _img, _hdr in zip(_hdu_uncached, _imgs, _hdrs):
                        _obs_image_cache[(_path, _h)] = (validate_image_plane_2d(_img), dict(_hdr))
                except Exception:
                    pass  # fall through to on-demand reads

    # Ensure the first observation is in the cache
    _obs_image_cache.setdefault((first.path, first.hdu), (first_image, dict(first_header)))

    for obs_idx, obs in enumerate(observations):
        key = (obs.path, obs.hdu)
        if key in _obs_image_cache:
            image, header_dict_obs = _obs_image_cache[key]
        else:
            image, header = read(
                obs.path, hdu=obs.hdu, return_header=True, device=read_dev
            )
            image = validate_image_plane_2d(image)
            header_dict_obs = dict(header)
            _obs_image_cache[key] = (image, header_dict_obs)
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
        if stacking_method == "median":
            sampled_numer, sampled_weight, overlap, moc = (
                _project_observation_to_target(
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
            assert median_images is not None and median_support is not None
            median_images.append(contrib_numer)
            median_support.append(contrib_weight > 0.0)
            overlap_count = overlap_count + overlap
            if observation_mocs is not None and moc is not None:
                observation_mocs.append(moc)
        else:
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
            corrected = cross_exposure_sky_corrector.correct(median_images, corrector_masks)
            median_images.clear()
            median_images.extend(corrected)
        else:
            assert projected_store is not None
            images = []
            masks = []
            for ps in projected_store:
                n = ps["numer"]
                w = ps["weight"]
                img = torch.where(w > 0.0, n / w, torch.zeros_like(n))
                images.append(img)
                masks.append(~(w > 0.0))
            corrected = cross_exposure_sky_corrector.correct(images, masks)
            numer.zero_()
            denom.zero_()
            overlap_count.zero_()
            for i, ps in enumerate(projected_store):
                w = ps["weight"]
                overlap_contrib = ps["overlap"]
                corrected_numer = corrected[i] * w
                tw = ps["target_window"]
                if tw is None:
                    numer.add_(corrected_numer)
                    denom.add_(w)
                    overlap_count.add_(overlap_contrib)
                else:
                    y0, y1, x0, x1 = tw
                    if y1 > y0 and x1 > x0:
                        numer[y0:y1, x0:x1].add_(corrected_numer)
                        denom[y0:y1, x0:x1].add_(w)
                        overlap_count[y0:y1, x0:x1].add_(overlap_contrib)

    if stacking_method == "median":
        assert median_images is not None and median_support is not None
        coadd, denom = masked_median_stack(median_images, median_support)
    else:
        pixel_area_sr = None
        if value_semantics == "flux":
            prepared = ensure_prepared_target()
            if prepared.pixel_area_sr is None:
                raise RuntimeError("target pixel area is required for flux semantics")
            pixel_area_sr = prepared.pixel_area_sr
        coadd, denom = finalize_projected_stack(
            numer,
            denom,
            stacking_method=stacking_method,
            value_semantics=value_semantics,
            pixel_area_sr=pixel_area_sr,
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


class ImagePlaneMapMaker:
    """Differentiable image-plane coaddition baseline."""

    def __init__(
        self,
        *,
        target_header: Optional[dict] = None,
        resampling_kernel: MapResamplingKernel = "bilinear",
        value_semantics: MapValueSemantics = "surface_brightness",
        stacking_method: MapStackingMethod = "weighted_mean",
        drizzle_pixfrac: float = 1.0,
        spatial_tile_hw: tuple[int, int] | None = None,
        device: torch.device | str | None = None,
        background_estimator: Callable[[Tensor, Tensor | None], Tensor] | None = None,
        cross_exposure_sky_corrector: CrossExposureSkyCorrector | None = None,
    ) -> None:
        self.target_header = target_header
        self.resampling_kernel = _validated_map_resampling_kernel(resampling_kernel)
        self.value_semantics = _validated_map_value_semantics(value_semantics)
        self.stacking_method = _validated_map_stacking_method(stacking_method)
        self.drizzle_pixfrac = _validated_drizzle_pixfrac(drizzle_pixfrac)
        self.spatial_tile_hw = (
            None
            if spatial_tile_hw is None
            else (int(spatial_tile_hw[0]), int(spatial_tile_hw[1]))
        )
        self.device = device
        self.background_estimator = background_estimator
        self.cross_exposure_sky_corrector = cross_exposure_sky_corrector

    def run(
        self,
        observations: Sequence[ImageObservation],
        *,
        output_path: Optional[str] = None,
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
            spatial_tile_hw=self.spatial_tile_hw,
            device=eff_device,
            background_estimator=self.background_estimator,
            cross_exposure_sky_corrector=self.cross_exposure_sky_corrector,
        )


def coadd_image_plane_cube(
    observations: Sequence[ImageObservation],
    *,
    target: PlannedImagePlaneCubeTarget | None = None,
    target_header: Optional[dict] = None,
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
) -> SpectralCubeMapResult:
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
    if stacking_method == "median":
        raise ValueError(
            "coadd_image_plane_cube currently supports median only for single-band image-plane coadds"
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

    first_image, first_header = read(
        observations[0].path,
        hdu=observations[0].hdu,
        return_header=True,
        device=read_dev,
    )
    first_image = validate_image_plane_2d(first_image)
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
            image, header = read(
                obs.path, hdu=obs.hdu, return_header=True, device=read_dev
            )
            image = validate_image_plane_2d(image)
            header_dict_obs = dict(header)
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
    cube, denom = finalize_projected_stack(
        numer,
        denom,
        stacking_method=stacking_method,
        value_semantics=value_semantics,
        pixel_area_sr=pixel_area_sr,
    )
    return SpectralCubeMapResult(
        spectral_axis=target.spectral_axis,
        band_metadata=band_metadata,
        data=cube,
        weight=denom,
        header=header_dict,
        spatial_wcs=target_wcs,
        overlap_count=overlap_count,
        coverage_mocs=tuple(coverage_mocs) if coverage_mocs is not None else None,
    )


def coadd_image_plane_multiband(
    observations: Sequence[ImageObservation],
    *,
    target_header: Optional[dict] = None,
    footprint_order: int | None = None,
    resampling_kernel: MapResamplingKernel = "bilinear",
    value_semantics: MapValueSemantics = "surface_brightness",
    stacking_method: MapStackingMethod = "weighted_mean",
    spectral_unit: str = "Angstrom",
    spectral_type: str = "WAVE",
    device: torch.device | str | None = None,
) -> MultiBandMapResult:
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
    return MultiBandMapResult(
        spectral_axis=cube.spectral_axis,
        data=cube.data,
        weight=cube.weight,
        header=cube.header,
        overlap_count=cube.overlap_count,
        coverage_mocs=cube.coverage_mocs,
        band_observation_counts=cube.band_metadata.observation_count,
    )
