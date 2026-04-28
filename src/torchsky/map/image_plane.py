"""Image-plane-first mapmaking primitives."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Literal, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from .. import WCS, read, write
from ..device import default_compute_device
from ..sphere.moc import MOC, nest2uniq
from ..spectral import (
    DataCube,
    SpectralAxis,
    spectral_axis_from_header,
    spectral_bin_edges_from_header,
)
from ..wcs import healpix as _healpix

from .fft_affine_ldu import resample_fft_ldu_affine_2d
from .fft_resample import resample_diagonal_affine_fft2d, resample_translation_fft2d
from .operators.background import SkyBackgroundOp
from .operators.bundle import ObservationOperatorBundle
from .operators.execution import (
    accumulate_projected_observation,
    finalize_projected_stack,
    masked_median_stack,
    prepare_image_observation_data,
    project_image_observation_geometry,
    stack_inputs_from_projected_observation,
)
from .operators.geometry import InterpolationGeometryOp
from .operators.mask_weight import load_image_observation_weight_map
from .operators.noise import ExplicitNoiseOp
from .operators.radiometry import apply_image_observation_calibration
from .operators.spectral import ExplicitSpectralResponseOp
from .periodic_affine import resample_linear_affine_periodic_bilinear_2d
from .spectral_affine import resample_linear_affine_spectral_2d
from .spectral_nufft import resample_nonuniform_spectral_fast_2d
from .spectral_nonuniform import resample_nonuniform_spectral_2d

MapResamplingKernel = Literal[
    "nearest",
    "bilinear",
    "bicubic",
    "lanczos3",
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
    "conservative_polygon",
    "splat_point",
    "splat_box",
]
MapValueSemantics = Literal["surface_brightness", "flux"]
MapStackingMethod = Literal["weighted_mean", "mean", "median"]


# Header probe for area scaling: recoverable failures when reading metadata only.
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
class _PreparedImagePlaneTarget:
    header: dict
    wcs: WCS
    shape: tuple[int, int]
    ra_deg: Tensor
    dec_deg: Tensor
    center_ra_deg: float
    center_dec_deg: float
    cap_radius_deg: float
    pixel_area_sr: Tensor | None = None
    representative_pixel_area_sr: float | None = None
    quadrature_world_grids: tuple[tuple[Tensor, Tensor], ...] | None = None


@dataclass
class _ProjectedImagePlaneObservation:
    numer: Tensor
    weight: Tensor
    overlap: Tensor
    moc: MOC | None
    target_window: tuple[int, int, int, int] | None = None


_LARGE_TAN_WINDOWING_MIN_TARGET_PIXELS = 1_500_000
_LARGE_TAN_WINDOWING_EDGE_SAMPLES = 9
_LARGE_TAN_WINDOWING_MARGIN_PIXELS = 2
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


def _validate_image_plane(image: Tensor) -> Tensor:
    if image.ndim != 2:
        raise ValueError("Image-plane mapmaking currently supports 2D images only")
    return image


def _target_shape_from_header(
    header: dict,
    *,
    fallback: tuple[int, int],
) -> tuple[int, int]:
    naxis1 = header.get("NAXIS1")
    naxis2 = header.get("NAXIS2")
    if naxis1 is None or naxis2 is None:
        return fallback
    width = int(naxis1)
    height = int(naxis2)
    if width <= 0 or height <= 0:
        raise ValueError("Target header must define positive NAXIS1/NAXIS2")
    return height, width


def _normalized_header_for_shape(
    header: dict,
    shape: tuple[int, int],
    *,
    overrides: Optional[dict] = None,
) -> dict:
    out = dict(header)
    out["NAXIS"] = 2
    out["NAXIS1"] = int(shape[1])
    out["NAXIS2"] = int(shape[0])
    if overrides is not None:
        out.update(dict(overrides))
    return out


_GRID_EQUIVALENT_IGNORE_KEYS = {
    "COMMENT",
    "HISTORY",
    "SIMPLE",
    "BITPIX",
    "EXTEND",
    "XTENSION",
    "EXTNAME",
    "EXTVER",
    "PCOUNT",
    "GCOUNT",
    "CHECKSUM",
    "DATASUM",
}


def _headers_equivalent_for_direct_copy(source: dict, target: dict) -> bool:
    def normalize(header: dict) -> dict:
        out: dict = {}
        for key, value in header.items():
            if str(key).endswith("_COMMENT"):
                continue
            if key in _GRID_EQUIVALENT_IGNORE_KEYS:
                continue
            out[key] = value
        return out

    return normalize(source) == normalize(target)


def _build_target_world_grid(
    target_wcs: WCS, height: int, width: int, device: torch.device
) -> tuple[Tensor, Tensor]:
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float64, device=device),
        torch.arange(width, dtype=torch.float64, device=device),
        indexing="ij",
    )
    ra, dec = target_wcs.pixel_to_world(x, y)
    return ra, dec


def _wrapped_delta_ra_deg(ra_a_deg: Tensor, ra_b_deg: Tensor) -> Tensor:
    return ((ra_a_deg - ra_b_deg + 180.0) % 360.0) - 180.0


def _pixel_area_sr_map(
    wcs: WCS,
    shape: tuple[int, int],
    *,
    device: torch.device,
) -> Tensor:
    height, width = int(shape[0]), int(shape[1])
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float64, device=device),
        torch.arange(width, dtype=torch.float64, device=device),
        indexing="ij",
    )
    ra_c, dec_c = wcs.pixel_to_world(x, y)
    ra_dx, dec_dx = wcs.pixel_to_world(x + 1.0, y)
    ra_dy, dec_dy = wcs.pixel_to_world(x, y + 1.0)
    cos_dec = torch.cos(torch.deg2rad(dec_c))
    dra_dx = _wrapped_delta_ra_deg(ra_dx, ra_c) * cos_dec
    dra_dy = _wrapped_delta_ra_deg(ra_dy, ra_c) * cos_dec
    ddec_dx = dec_dx - dec_c
    ddec_dy = dec_dy - dec_c
    det_deg2 = torch.abs(dra_dx * ddec_dy - dra_dy * ddec_dx)
    area_sr = torch.clamp(det_deg2 * ((torch.pi / 180.0) ** 2), min=1e-16)
    return area_sr.to(dtype=torch.float32)


def _representative_pixel_area_sr(
    wcs: WCS,
    shape: tuple[int, int],
    *,
    device: torch.device,
) -> float:
    height, width = int(shape[0]), int(shape[1])
    y = torch.tensor([(max(height - 1, 0)) * 0.5], dtype=torch.float64, device=device)
    x = torch.tensor([(max(width - 1, 0)) * 0.5], dtype=torch.float64, device=device)
    ra_c, dec_c = wcs.pixel_to_world(x, y)
    ra_dx, dec_dx = wcs.pixel_to_world(x + 1.0, y)
    ra_dy, dec_dy = wcs.pixel_to_world(x, y + 1.0)
    cos_dec = torch.cos(torch.deg2rad(dec_c))
    dra_dx = _wrapped_delta_ra_deg(ra_dx, ra_c) * cos_dec
    dra_dy = _wrapped_delta_ra_deg(ra_dy, ra_c) * cos_dec
    ddec_dx = dec_dx - dec_c
    ddec_dy = dec_dy - dec_c
    det_deg2 = torch.abs(dra_dx * ddec_dy - dra_dy * ddec_dx)
    area_sr = torch.clamp(det_deg2 * ((torch.pi / 180.0) ** 2), min=1e-16)
    return float(area_sr.mean().item())


def _observation_header_for_area_scaling(
    obs: ImageObservation,
    *,
    cached_header: dict | None = None,
) -> dict:
    if cached_header is not None:
        return dict(cached_header)
    try:
        from .. import Header, cpp as _cpp

        hdu = obs.hdu
        if isinstance(hdu, int) or hdu is None:
            return dict(
                Header(_cpp.read_header_dict(obs.path, 0 if hdu is None else hdu))
            )
    except _HEADER_PROBE_EXC:
        pass
    _, header = read(obs.path, hdu=obs.hdu, return_header=True)
    return dict(header)


def _should_apply_surface_brightness_area_scaling(
    observations: Sequence[ImageObservation],
    *,
    target_header: dict,
    target_shape: tuple[int, int],
    first_header: dict | None = None,
) -> bool:
    if not observations or not _is_plain_tan_header(target_header):
        return True
    target_area_sr = _representative_pixel_area_sr(
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
        if not _is_plain_tan_header(header_dict_obs):
            return True
        source_wcs = WCS(header_dict_obs)
        if getattr(source_wcs, "sip", None) is not None or bool(
            getattr(source_wcs, "_is_tpv", False)
        ):
            return True
        source_shape = (
            int(header_dict_obs.get("NAXIS2", 0)),
            int(header_dict_obs.get("NAXIS1", 0)),
        )
        if source_shape[0] <= 0 or source_shape[1] <= 0:
            return True
        source_area_sr = _representative_pixel_area_sr(
            source_wcs,
            source_shape,
            device=torch.device("cpu"),
        )
        inv_area_sum += 1.0 / max(float(source_area_sr), 1e-20)
    harmonic_mean_source_area_sr = float(len(observations)) / max(inv_area_sum, 1e-20)
    return float(target_area_sr) >= harmonic_mean_source_area_sr


def _sinc(x: Tensor) -> Tensor:
    pix = torch.pi * x
    return torch.where(
        torch.abs(x) < 1e-6,
        torch.ones_like(x),
        torch.sin(pix) / pix,
    )


def _lanczos_weight(x: Tensor, *, a: int) -> Tensor:
    x = torch.abs(x)
    weight = _sinc(x) * _sinc(x / float(a))
    return torch.where(x < float(a), weight, torch.zeros_like(weight))


def _keys_cubic_weight(x: Tensor, *, b: float, c: float) -> Tensor:
    ax = torch.abs(x)
    ax2 = ax * ax
    ax3 = ax2 * ax
    w0 = (
        ((12.0 - 9.0 * b - 6.0 * c) * ax3)
        + ((-18.0 + 12.0 * b + 6.0 * c) * ax2)
        + (6.0 - 2.0 * b)
    ) / 6.0
    w1 = (
        ((-b - 6.0 * c) * ax3)
        + ((6.0 * b + 30.0 * c) * ax2)
        + ((-12.0 * b - 48.0 * c) * ax)
        + (8.0 * b + 24.0 * c)
    ) / 6.0
    return torch.where(
        ax < 1.0,
        w0,
        torch.where(ax < 2.0, w1, torch.zeros_like(ax)),
    )


def _kernel_support(kernel: MapResamplingKernel) -> int:
    if kernel == "lanczos3":
        return 3
    if kernel == "lanczos5":
        return 5
    if kernel in {"mitchell_netravali", "catmull_rom"}:
        return 2
    raise ValueError(f"custom support is undefined for kernel {kernel!r}")


def _kernel_weight(kernel: MapResamplingKernel, x: Tensor) -> Tensor:
    if kernel == "lanczos3":
        return _lanczos_weight(x, a=3)
    if kernel == "lanczos5":
        return _lanczos_weight(x, a=5)
    if kernel == "mitchell_netravali":
        return _keys_cubic_weight(x, b=1.0 / 3.0, c=1.0 / 3.0)
    if kernel == "catmull_rom":
        return _keys_cubic_weight(x, b=0.0, c=0.5)
    raise ValueError(f"custom weight is undefined for kernel {kernel!r}")


def _custom_separable_sample(
    image: Tensor,
    *,
    x_src: Tensor,
    y_src: Tensor,
    kernel: MapResamplingKernel,
) -> Tensor:
    if image.ndim not in {2, 3}:
        raise ValueError(
            "custom separable sampling expects a 2D image or [C, H, W] tensor"
        )
    work = image if image.ndim == 3 else image.unsqueeze(0)
    support = _kernel_support(kernel)
    x_src = x_src.to(dtype=work.dtype)
    y_src = y_src.to(dtype=work.dtype)
    x0 = torch.floor(x_src).to(dtype=torch.int64)
    y0 = torch.floor(y_src).to(dtype=torch.int64)
    offsets = torch.arange(
        -support + 1,
        support + 1,
        device=image.device,
        dtype=torch.int64,
    )
    xs = x0.unsqueeze(-1) + offsets.view(1, 1, -1)
    ys = y0.unsqueeze(-1) + offsets.view(1, 1, -1)
    dx = x_src.unsqueeze(-1) - xs.to(dtype=work.dtype)
    dy = y_src.unsqueeze(-1) - ys.to(dtype=work.dtype)
    wx = _kernel_weight(kernel, dx)
    wy = _kernel_weight(kernel, dy)

    _, h_src, w_src = work.shape
    x_valid = (xs >= 0) & (xs < w_src)
    y_valid = (ys >= 0) & (ys < h_src)
    xs = xs.clamp(0, w_src - 1)
    ys = ys.clamp(0, h_src - 1)
    sample_idx = ys.unsqueeze(-1) * w_src + xs.unsqueeze(-2)
    sampled = (
        work.reshape(work.shape[0], -1)
        .index_select(1, sample_idx.reshape(-1))
        .reshape(
            work.shape[0],
            *x_src.shape,
            sample_idx.shape[-2],
            sample_idx.shape[-1],
        )
    )
    weights = (
        wy.unsqueeze(-1)
        * wx.unsqueeze(-2)
        * y_valid.unsqueeze(-1).to(dtype=work.dtype)
        * x_valid.unsqueeze(-2).to(dtype=work.dtype)
    )
    out = torch.sum(sampled * weights.unsqueeze(0), dim=(-1, -2))
    return out if image.ndim == 3 else out.squeeze(0)


def _sample_image_plane(
    image: Tensor,
    *,
    x_src: Tensor,
    y_src: Tensor,
    grid: Tensor,
    kernel: MapResamplingKernel,
) -> Tensor:
    if kernel in {"lanczos3", "lanczos5", "mitchell_netravali", "catmull_rom"}:
        return _custom_separable_sample(
            image,
            x_src=x_src,
            y_src=y_src,
            kernel=kernel,
        )
    return (
        F.grid_sample(
            image.unsqueeze(0).unsqueeze(0),
            grid.unsqueeze(0),
            mode=kernel,
            padding_mode="zeros",
            align_corners=True,
        )
        .squeeze(0)
        .squeeze(0)
    )


def _sample_image_plane_channels(
    image: Tensor,
    *,
    x_src: Tensor,
    y_src: Tensor,
    grid: Tensor,
    kernel: MapResamplingKernel,
) -> Tensor:
    if image.ndim != 3:
        raise ValueError("channel sampling expects [C, H, W] input")
    if kernel in {"lanczos3", "lanczos5", "mitchell_netravali", "catmull_rom"}:
        return _custom_separable_sample(
            image,
            x_src=x_src,
            y_src=y_src,
            kernel=kernel,
        )
    return F.grid_sample(
        image.unsqueeze(0),
        grid.unsqueeze(0),
        mode=kernel,
        padding_mode="zeros",
        align_corners=True,
    ).squeeze(0)


def _sample_image_plane_batched(
    image: Tensor,
    *,
    x_src: Tensor,
    y_src: Tensor,
    grid: Tensor,
    kernel: MapResamplingKernel,
) -> Tensor:
    if x_src.ndim == 2:
        return _sample_image_plane(
            image,
            x_src=x_src,
            y_src=y_src,
            grid=grid,
            kernel=kernel,
        )
    if kernel in {"lanczos3", "lanczos5", "mitchell_netravali", "catmull_rom"}:
        samples = [
            _custom_separable_sample(
                image,
                x_src=x_src[idx],
                y_src=y_src[idx],
                kernel=kernel,
            )
            for idx in range(int(x_src.shape[0]))
        ]
        return torch.stack(samples, dim=0)
    batch = int(grid.shape[0])
    image_b = (
        image.unsqueeze(0).unsqueeze(0).expand(batch, 1, image.shape[0], image.shape[1])
    )
    return F.grid_sample(
        image_b,
        grid.to(dtype=torch.float32),
        mode=kernel,
        padding_mode="zeros",
        align_corners=True,
    ).squeeze(1)


def _sample_image_plane_batched_channels(
    image: Tensor,
    *,
    x_src: Tensor,
    y_src: Tensor,
    grid: Tensor,
    kernel: MapResamplingKernel,
) -> Tensor:
    if image.ndim != 3:
        raise ValueError("batched channel sampling expects [C, H, W] input")
    if x_src.ndim == 2:
        return _sample_image_plane_channels(
            image,
            x_src=x_src,
            y_src=y_src,
            grid=grid,
            kernel=kernel,
        )
    if kernel in {"lanczos3", "lanczos5", "mitchell_netravali", "catmull_rom"}:
        samples = [
            _custom_separable_sample(
                image,
                x_src=x_src[idx],
                y_src=y_src[idx],
                kernel=kernel,
            )
            for idx in range(int(x_src.shape[0]))
        ]
        return torch.stack(samples, dim=0)
    batch = int(grid.shape[0])
    image_b = image.unsqueeze(0).expand(batch, *image.shape)
    return F.grid_sample(
        image_b,
        grid.to(dtype=torch.float32),
        mode=kernel,
        padding_mode="zeros",
        align_corners=True,
    )


def _is_drizzle_kernel(kernel: MapResamplingKernel) -> bool:
    return kernel in {"drizzle_point", "drizzle_square"}


def _is_conservative_kernel(kernel: MapResamplingKernel) -> bool:
    return kernel == "conservative_polygon"


def _is_splat_kernel(kernel: MapResamplingKernel) -> bool:
    return kernel in {"splat_point", "splat_box"}


def _requires_roundtrip_validity_check(source_wcs: WCS) -> bool:
    # SIP inverse paths still benefit from an explicit round-trip consistency gate.
    # For TPV, the gate was useful while the distorted WCS dispatch could route
    # through a wrong TAN-only shortcut, but after fixing that dispatch the same
    # gate became over-rejective on realistic warped-stack benchmarks.
    return bool(getattr(source_wcs, "sip", None) is not None)


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


def _tpv_candidate_mask(
    source_wcs: WCS,
    xi: Tensor,
    eta: Tensor,
    image_shape: tuple[int, int],
) -> Tensor:
    xi_min, xi_max, eta_min, eta_max = _tpv_affine_envelope(source_wcs, image_shape)


def _spherical_cap_radius_from_world_grid(
    ra_deg: Tensor,
    dec_deg: Tensor,
    *,
    center_ra_deg: float,
    center_dec_deg: float,
) -> float:
    dra = _wrapped_delta_ra_deg(
        ra_deg, torch.tensor(center_ra_deg, dtype=ra_deg.dtype, device=ra_deg.device)
    )
    dec0 = torch.tensor(center_dec_deg, dtype=dec_deg.dtype, device=dec_deg.device)
    cos_dec = torch.cos(dec_deg * (math.pi / 180.0))
    chord = torch.sqrt((dra * cos_dec) ** 2 + (dec_deg - dec0) ** 2)
    return float(torch.max(chord).item())


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
    area_sr = _pixel_area_sr_map(source_wcs, shape, device=device)
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


def _quadrature_target_world_grid(
    target_wcs: WCS,
    shape: tuple[int, int],
    *,
    device: torch.device,
    offset_x: float,
    offset_y: float,
) -> tuple[Tensor, Tensor]:
    height, width = int(shape[0]), int(shape[1])
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float64, device=device) + float(offset_y),
        torch.arange(width, dtype=torch.float64, device=device) + float(offset_x),
        indexing="ij",
    )
    return target_wcs.pixel_to_world(x, y)


def _quadrature_target_world_grids(
    target_wcs: WCS,
    shape: tuple[int, int],
    *,
    device: torch.device,
) -> tuple[tuple[Tensor, Tensor], ...]:
    offset = 0.5 / math.sqrt(3.0)
    offsets = (-offset, offset)
    grids: list[tuple[Tensor, Tensor]] = []
    for offset_y in offsets:
        for offset_x in offsets:
            grids.append(
                _quadrature_target_world_grid(
                    target_wcs,
                    shape,
                    device=device,
                    offset_x=offset_x,
                    offset_y=offset_y,
                )
            )
    return tuple(grids)


def _load_observation_weight_map(
    obs: ImageObservation,
    image: Tensor,
    *,
    device: torch.device,
) -> Tensor:
    return load_image_observation_weight_map(obs, image, device=device)


def _infer_fft_translation_shift_yx(
    source_wcs: WCS,
    target_wcs: WCS,
    shape_hw: tuple[int, int],
) -> tuple[float, float]:
    """Infer ``(shift_y, shift_x)`` for :func:`resample_translation_fft2d` from WCS pair.

    Requires target pixel centers to map to source pixels under a **pure translation**
    (checked at five reference positions). Raises ``ValueError`` if not.

    Convention matches :func:`torchsky.map.fft_resample.resample_translation_fft2d`
    (see ``tests/test_map_fft_resample.test_fft_shift_matches_integer_roll``).
    """
    if getattr(source_wcs, "sip", None) is not None or bool(
        getattr(source_wcs, "_is_tpv", False)
    ):
        raise ValueError(
            "fft_shift requires a linear astrometric frame (no SIP or TPV on the source WCS)"
        )
    h, w = int(shape_hw[0]), int(shape_hw[1])
    dev = torch.device("cpu")
    jx = torch.tensor(
        [0.0, float(w - 1), float(w - 1) * 0.5, 0.0, float(w - 1)],
        device=dev,
        dtype=torch.float64,
    )
    iy = torch.tensor(
        [0.0, 0.0, float(h - 1) * 0.5, float(h - 1), float(h - 1)],
        device=dev,
        dtype=torch.float64,
    )
    ra, dec = target_wcs.pixel_to_world(jx, iy)
    xs, ys = source_wcs.world_to_pixel(ra, dec)
    dx = xs - jx
    dy = ys - iy
    mx = float(dx.mean().item())
    my = float(dy.mean().item())
    if not torch.allclose(dx, torch.full_like(dx, mx), rtol=1e-4, atol=1e-6):
        raise ValueError(
            "fft_shift requires a constant pixel translation between target and source WCS "
            "(column offset not uniform on probe points)"
        )
    if not torch.allclose(dy, torch.full_like(dy, my), rtol=1e-4, atol=1e-6):
        raise ValueError(
            "fft_shift requires a constant pixel translation between target and source WCS "
            "(row offset not uniform on probe points)"
        )
    # Align with ``resample_translation_fft2d`` / integer roll convention (see fft tests).
    return (-my, -mx)


def _infer_diagonal_scale_shift_pixel_map(
    source_wcs: WCS,
    target_wcs: WCS,
    shape_hw: tuple[int, int],
) -> tuple[float, float, float, float]:
    """Infer ``(scale_x, shift_x, scale_y, shift_y)`` for diagonal affine source pixels.

    Target pixel ``(iy, jx)`` (row, column) maps to source column ``scale_x * jx + shift_x``
    and source row ``scale_y * iy + shift_y`` when the mapping is separable (no
    significant cross-terms). Fits a 5×5 probe grid; raises ``ValueError`` if not.
    """
    if getattr(source_wcs, "sip", None) is not None or bool(
        getattr(source_wcs, "_is_tpv", False)
    ):
        raise ValueError(
            "fft_diagonal_affine requires a linear astrometric frame (no SIP or TPV on source)"
        )
    h, w = int(shape_hw[0]), int(shape_hw[1])
    dev = torch.device("cpu")
    j_line = torch.linspace(0.0, float(w - 1), 5, device=dev, dtype=torch.float64)
    i_line = torch.linspace(0.0, float(h - 1), 5, device=dev, dtype=torch.float64)
    ii, jj = torch.meshgrid(i_line, j_line, indexing="ij")
    iy = ii.reshape(-1)
    jx = jj.reshape(-1)
    ra, dec = target_wcs.pixel_to_world(jx, iy)
    xs, ys = source_wcs.world_to_pixel(ra, dec)
    ones = torch.ones_like(jx)
    ax = torch.stack([jx, iy, ones], dim=1)
    sol_x = torch.linalg.lstsq(ax, xs.unsqueeze(-1), rcond=None).solution.squeeze(-1)
    sol_y = torch.linalg.lstsq(
        torch.stack([iy, jx, ones], dim=1), ys.unsqueeze(-1), rcond=None
    ).solution.squeeze(-1)
    sx = float(sol_x[0].item())
    cx = float(sol_x[1].item())
    bx = float(sol_x[2].item())
    sy = float(sol_y[0].item())
    dy = float(sol_y[1].item())
    by = float(sol_y[2].item())
    tol_c = max(1e-3, 1e-4 * max(abs(sx), abs(sy), 1.0))
    if abs(cx) > tol_c or abs(dy) > tol_c:
        raise ValueError(
            "fft_diagonal_affine requires a separable diagonal pixel map (cross-terms too large)"
        )
    xs_fit = sx * jx + cx * iy + bx
    ys_fit = sy * iy + dy * jx + by
    max_x = float(torch.max(torch.abs(xs_fit - xs)).item())
    max_y = float(torch.max(torch.abs(ys_fit - ys)).item())
    if max_x > 5e-3 or max_y > 5e-3:
        raise ValueError(
            "fft_diagonal_affine WCS probes do not match a diagonal affine map"
        )
    if not all(math.isfinite(v) for v in (sx, sy, bx, by)):
        raise ValueError("fft_diagonal_affine inferred non-finite parameters")
    if abs(sx) < 1e-8 or abs(sy) < 1e-8:
        raise ValueError("fft_diagonal_affine inferred degenerate scales")
    return sx, bx, sy, by


def _infer_linear_affine_pixel_map(
    source_wcs: WCS,
    target_wcs: WCS,
    shape_hw: tuple[int, int],
) -> tuple[float, float, float, float, float, float]:
    """Infer full 2×2 affine map from target pixel ``(iy, jx)`` to source ``(ys, xs)``.

    Uses the same 5×5 probe grid as :func:`_infer_diagonal_scale_shift_pixel_map`.
    Returns ``(a00, a01, a10, a11, bx, by)`` with::

        sx = a00 * jx + a01 * iy + bx
        sy = a10 * jx + a11 * iy + by
    """
    if getattr(source_wcs, "sip", None) is not None or bool(
        getattr(source_wcs, "_is_tpv", False)
    ):
        raise ValueError(
            "periodic_linear_affine requires a linear astrometric frame (no SIP or TPV on source)"
        )
    h, w = int(shape_hw[0]), int(shape_hw[1])
    dev = torch.device("cpu")
    j_line = torch.linspace(0.0, float(w - 1), 5, device=dev, dtype=torch.float64)
    i_line = torch.linspace(0.0, float(h - 1), 5, device=dev, dtype=torch.float64)
    ii, jj = torch.meshgrid(i_line, j_line, indexing="ij")
    iy = ii.reshape(-1)
    jx = jj.reshape(-1)
    ra, dec = target_wcs.pixel_to_world(jx, iy)
    xs, ys = source_wcs.world_to_pixel(ra, dec)
    ones = torch.ones_like(jx)
    a_mat = torch.stack([jx, iy, ones], dim=1)
    sol_x = torch.linalg.lstsq(a_mat, xs.unsqueeze(-1), rcond=None).solution.squeeze(-1)
    sol_y = torch.linalg.lstsq(a_mat, ys.unsqueeze(-1), rcond=None).solution.squeeze(-1)
    a00 = float(sol_x[0].item())
    a01 = float(sol_x[1].item())
    bx = float(sol_x[2].item())
    a10 = float(sol_y[0].item())
    a11 = float(sol_y[1].item())
    by = float(sol_y[2].item())
    det = a00 * a11 - a01 * a10
    if not math.isfinite(det) or abs(det) < 1e-12:
        raise ValueError("periodic_linear_affine inferred a degenerate linear map")
    xs_fit = a00 * jx + a01 * iy + bx
    ys_fit = a10 * jx + a11 * iy + by
    max_x = float(torch.max(torch.abs(xs_fit - xs)).item())
    max_y = float(torch.max(torch.abs(ys_fit - ys)).item())
    if max_x > 5e-3 or max_y > 5e-3:
        raise ValueError(
            "periodic_linear_affine WCS probes do not match a linear affine pixel map"
        )
    return a00, a01, a10, a11, bx, by


def _finalize_periodic_resampled_weighted(
    sampled_numer: Tensor,
    sampled_weight: Tensor,
    sampled_support: Tensor | None,
    *,
    value_semantics: MapValueSemantics,
    target_representative_area_sr: float | None,
) -> tuple[Tensor, Tensor]:
    if sampled_support is not None:
        in_bounds = sampled_support > 0.25
    else:
        in_bounds = torch.ones_like(sampled_numer, dtype=torch.bool)
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
    image = _validate_image_plane(image).to(dtype=torch.float32)
    source_weight = _validate_image_plane(source_weight).to(dtype=torch.float32)

    h_src, w_src = image.shape
    target_representative_area_sr = target.representative_pixel_area_sr

    # Reproject weighted signal and weight separately, then form the weighted mean
    # only after all observations have been accumulated on the target grid.
    if value_semantics == "flux":
        source_area_sr = _pixel_area_sr_map(
            source_wcs,
            tuple(image.shape),
            device=image.device,
        )
        weighted_image = image * source_weight * source_area_sr
        weighted_weight = source_weight * torch.square(source_area_sr)
    elif target_representative_area_sr is not None:
        source_area_sr = _representative_pixel_area_sr(
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
            sy, sx = _infer_fft_translation_shift_yx(
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
        return _finalize_periodic_resampled_weighted(
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
            d_sx, d_bx, d_sy, d_by = _infer_diagonal_scale_shift_pixel_map(
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
        return _finalize_periodic_resampled_weighted(
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
            la = _infer_linear_affine_pixel_map(
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
        return _finalize_periodic_resampled_weighted(
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
            la = _infer_linear_affine_pixel_map(
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
        return _finalize_periodic_resampled_weighted(
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
            la = _infer_linear_affine_pixel_map(
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
        return _finalize_periodic_resampled_weighted(
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
        flat_ra: Tensor,
        flat_dec: Tensor,
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
            sampled_planes = _sample_image_plane_batched_channels(
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
        return _sample_on_world_grid(target.ra_deg, target.dec_deg)

    accum_numer = torch.zeros_like(target.ra_deg, dtype=torch.float32)
    accum_weight = torch.zeros_like(target.ra_deg, dtype=torch.float32)
    quadrature_world_grids = target.quadrature_world_grids
    if quadrature_world_grids is None:
        quadrature_world_grids = _quadrature_target_world_grids(
            target.wcs,
            target.shape,
            device=target.ra_deg.device,
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
    allowed = (
        "nearest",
        "bilinear",
        "bicubic",
        "lanczos3",
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
        "conservative_polygon",
        "splat_point",
        "splat_box",
    )
    if resampling_kernel not in allowed:
        raise ValueError(f"resampling_kernel must be one of {allowed}")
    return resampling_kernel


def _validated_map_value_semantics(
    value_semantics: str,
) -> MapValueSemantics:
    allowed = ("surface_brightness", "flux")
    if value_semantics not in allowed:
        raise ValueError(f"value_semantics must be one of {allowed}")
    return value_semantics


def _validated_drizzle_pixfrac(drizzle_pixfrac: float) -> float:
    pixfrac = float(drizzle_pixfrac)
    if not math.isfinite(pixfrac) or pixfrac <= 0.0:
        raise ValueError("drizzle_pixfrac must be a finite positive float")
    return pixfrac


def _validated_map_stacking_method(
    stacking_method: str,
) -> MapStackingMethod:
    allowed = ("weighted_mean", "mean", "median")
    if stacking_method not in allowed:
        raise ValueError(f"stacking_method must be one of {allowed}")
    return stacking_method


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

    image = _validate_image_plane(image).to(dtype=torch.float32)
    source_weight = _validate_image_plane(source_weight).to(dtype=torch.float32)
    h_src, w_src = image.shape

    if value_semantics == "flux":
        source_area_sr = _pixel_area_sr_map(
            source_wcs,
            tuple(image.shape),
            device=image.device,
        )
        weighted_image = image * source_weight * source_area_sr
        weighted_weight = source_weight * torch.square(source_area_sr)
    elif target.representative_pixel_area_sr is not None:
        source_area_sr = _representative_pixel_area_sr(
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
    out_img = torch.where(out_wht > 0.0, out_img, torch.zeros_like(out_img))
    return out_img * out_wht, out_wht


def _prepare_target(
    *,
    header: dict,
    target_wcs: WCS,
    target_shape: tuple[int, int],
    device: torch.device,
    include_pixel_area: bool = False,
    include_representative_pixel_area: bool = False,
    include_quadrature: bool = False,
) -> _PreparedImagePlaneTarget:
    ra_deg, dec_deg = _build_target_world_grid(
        target_wcs,
        target_shape[0],
        target_shape[1],
        device=device,
    )
    center_ra_deg = float(target_wcs.alpha0)
    center_dec_deg = float(target_wcs.delta0)
    return _PreparedImagePlaneTarget(
        header=header,
        wcs=target_wcs,
        shape=target_shape,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        center_ra_deg=center_ra_deg,
        center_dec_deg=center_dec_deg,
        cap_radius_deg=_spherical_cap_radius_from_world_grid(
            ra_deg,
            dec_deg,
            center_ra_deg=center_ra_deg,
            center_dec_deg=center_dec_deg,
        ),
        pixel_area_sr=(
            _pixel_area_sr_map(target_wcs, target_shape, device=device)
            if include_pixel_area
            else None
        ),
        representative_pixel_area_sr=(
            _representative_pixel_area_sr(target_wcs, target_shape, device=device)
            if include_representative_pixel_area
            else None
        ),
        quadrature_world_grids=(
            _quadrature_target_world_grids(
                target_wcs,
                target_shape,
                device=device,
            )
            if include_quadrature
            else None
        ),
    )


def _is_plain_tan_header(header: dict) -> bool:
    return str(header.get("CTYPE1", "")).endswith("TAN") and str(
        header.get("CTYPE2", "")
    ).endswith("TAN")


def _source_bbox_on_target(
    source_wcs: WCS,
    source_shape: tuple[int, int],
    target_wcs: WCS,
    target_shape: tuple[int, int],
    *,
    margin: int,
    edge_samples: int,
) -> tuple[int, int, int, int] | None:
    h_src, w_src = int(source_shape[0]), int(source_shape[1])
    edge_samples = max(int(edge_samples), 2)
    xs = torch.linspace(
        0.0,
        float(w_src - 1),
        steps=edge_samples,
        dtype=torch.float64,
        device=source_wcs.device,
    )
    ys = torch.linspace(
        0.0,
        float(h_src - 1),
        steps=edge_samples,
        dtype=torch.float64,
        device=source_wcs.device,
    )
    top_x = xs
    top_y = torch.zeros_like(xs)
    bottom_x = xs
    bottom_y = torch.full_like(xs, float(h_src - 1))
    left_x = torch.zeros_like(ys)
    left_y = ys
    right_x = torch.full_like(ys, float(w_src - 1))
    right_y = ys
    x_src = torch.cat([top_x, bottom_x, left_x, right_x], dim=0)
    y_src = torch.cat([top_y, bottom_y, left_y, right_y], dim=0)
    ra_deg, dec_deg = source_wcs.pixel_to_world(x_src, y_src)
    x_tgt, y_tgt = target_wcs.world_to_pixel(ra_deg, dec_deg)
    finite = torch.isfinite(x_tgt) & torch.isfinite(y_tgt)
    if not bool(finite.any()):
        return None
    x_good = x_tgt[finite]
    y_good = y_tgt[finite]
    height, width = int(target_shape[0]), int(target_shape[1])
    x0 = max(int(torch.floor(x_good.min()).item()) - margin, 0)
    x1 = min(int(torch.ceil(x_good.max()).item()) + margin + 1, width)
    y0 = max(int(torch.floor(y_good.min()).item()) - margin, 0)
    y1 = min(int(torch.ceil(y_good.max()).item()) + margin + 1, height)
    if x0 >= x1 or y0 >= y1:
        return None
    return y0, y1, x0, x1


def _slice_prepared_target(
    target: _PreparedImagePlaneTarget,
    *,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
) -> _PreparedImagePlaneTarget:
    quadrature = None
    if target.quadrature_world_grids is not None:
        quadrature = tuple(
            (ra[y0:y1, x0:x1], dec[y0:y1, x0:x1])
            for ra, dec in target.quadrature_world_grids
        )
    pixel_area = (
        None if target.pixel_area_sr is None else target.pixel_area_sr[y0:y1, x0:x1]
    )
    return _PreparedImagePlaneTarget(
        header=target.header,
        wcs=target.wcs,
        shape=(y1 - y0, x1 - x0),
        ra_deg=target.ra_deg[y0:y1, x0:x1],
        dec_deg=target.dec_deg[y0:y1, x0:x1],
        center_ra_deg=target.center_ra_deg,
        center_dec_deg=target.center_dec_deg,
        cap_radius_deg=target.cap_radius_deg,
        pixel_area_sr=pixel_area,
        representative_pixel_area_sr=target.representative_pixel_area_sr,
        quadrature_world_grids=quadrature,
    )


def _window_large_tan_target(
    *,
    source_wcs: WCS,
    source_header: dict,
    source_shape: tuple[int, int],
    target: _PreparedImagePlaneTarget,
    target_header: dict,
    value_semantics: MapValueSemantics,
) -> tuple[_PreparedImagePlaneTarget | None, tuple[int, int, int, int] | None]:
    if value_semantics != "surface_brightness":
        return target, None
    if (
        int(target.shape[0]) * int(target.shape[1])
        < _LARGE_TAN_WINDOWING_MIN_TARGET_PIXELS
    ):
        return target, None
    if not _is_plain_tan_header(source_header) or not _is_plain_tan_header(
        target_header
    ):
        return target, None
    if getattr(source_wcs, "sip", None) is not None or bool(
        getattr(source_wcs, "_is_tpv", False)
    ):
        return target, None
    bbox = _source_bbox_on_target(
        source_wcs,
        source_shape,
        target.wcs,
        target.shape,
        margin=_LARGE_TAN_WINDOWING_MARGIN_PIXELS,
        edge_samples=_LARGE_TAN_WINDOWING_EDGE_SAMPLES,
    )
    if bbox is None:
        return None, (0, 0, 0, 0)
    y0, y1, x0, x1 = bbox
    if (
        y0 == 0
        and y1 == int(target.shape[0])
        and x0 == 0
        and x1 == int(target.shape[1])
    ):
        return target, None
    return _slice_prepared_target(target, y0=y0, y1=y1, x0=x0, x1=x1), bbox


def _weight_map_to_moc(
    weight_map: Tensor,
    target: _PreparedImagePlaneTarget,
    *,
    max_order: int,
) -> MOC:
    valid = weight_map > 0.0
    if not bool(valid.any()):
        return MOC(torch.empty((0,), dtype=torch.int64))
    nside = 1 << int(max_order)
    pix = _healpix.ang2pix(
        nside,
        target.ra_deg[valid],
        target.dec_deg[valid],
        nest=True,
        lonlat=True,
    )
    return MOC(nest2uniq(nside, pix.to(torch.int64)))


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
) -> _ProjectedImagePlaneObservation:
    if not torch.isfinite(torch.tensor(obs.weight, dtype=torch.float64)):
        raise ValueError("Observation weights must be finite")
    if obs.weight <= 0.0:
        raise ValueError("Observation weights must be positive")

    if image is None or header_dict_obs is None:
        image, header = read(
            obs.path, hdu=obs.hdu, return_header=True, device=str(device)
        )
        image = _validate_image_plane(image)
        header_dict_obs = dict(header)
    image = image.to(device=device)
    prepared_observation = prepare_image_observation_data(
        obs,
        image,
        device=device,
        resampling_kernel=resampling_kernel,
        load_weight_map_fn=_load_observation_weight_map,
        apply_calibration_fn=_apply_observation_calibration,
    )
    image = prepared_observation.image
    source_weight = prepared_observation.source_weight
    operator_bundle = prepared_observation.operator_bundle
    prepared_target_view = prepared_target
    target_window = None
    if (
        _headers_equivalent_for_direct_copy(header_dict_obs, target_header)
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
        sampled_numer, sampled_weight = project_image_observation_geometry(
            image,
            source_weight,
            source_wcs,
            prepared_target_view,
            geometry=operator_bundle.geometry,
            resampling_kernel=resampling_kernel,
            value_semantics=value_semantics,
            drizzle_pixfrac=drizzle_pixfrac,
            pixel_area_sr_map_fn=_pixel_area_sr_map,
            representative_pixel_area_fn=_representative_pixel_area_sr,
            chunk_pixels=_DRIZZLE_FORWARD_CHUNK_PIXELS,
            resample_weighted_to_target_fn=_resample_weighted_to_target,
            forward_project_weighted_to_target_fn=_forward_project_weighted_to_target,
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


def _subtarget_header_for_window(
    full_header: dict,
    *,
    full_shape: tuple[int, int],
    y0: int,
    y1: int,
    x0: int,
    x1: int,
) -> tuple[dict, tuple[int, int]]:
    """Build WCS header for a pixel window ``[y0,y1) × [x0,x1)`` in the full grid."""
    h = int(y1) - int(y0)
    w = int(x1) - int(x0)
    if h <= 0 or w <= 0:
        raise ValueError("subtarget window must have positive width and height")
    fh, fw = int(full_shape[0]), int(full_shape[1])
    if y0 < 0 or x0 < 0 or y1 > fh or x1 > fw:
        raise ValueError("subtarget window exceeds full image bounds")
    out = dict(full_header)
    out["NAXIS1"] = w
    out["NAXIS2"] = h
    if "CRPIX1" in out:
        out["CRPIX1"] = float(out["CRPIX1"]) - float(x0)
    if "CRPIX2" in out:
        out["CRPIX2"] = float(out["CRPIX2"]) - float(y0)
    return out, (h, w)


def plan_image_plane_target(
    observations: Sequence[ImageObservation],
    *,
    reference_index: int = 0,
    header_overrides: Optional[dict] = None,
    device: torch.device | str | None = None,
) -> PlannedImagePlaneTarget:
    if not observations:
        raise ValueError("At least one observation is required")
    if reference_index < 0 or reference_index >= len(observations):
        raise IndexError("reference_index is out of range")

    ref = observations[reference_index]
    read_dev = str(_resolve_map_device(device))
    image, header = read(ref.path, hdu=ref.hdu, return_header=True, device=read_dev)
    image = _validate_image_plane(image)
    shape = tuple(image.shape)
    normalized_header = _normalized_header_for_shape(
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
) -> PlannedImagePlaneCubeTarget:
    spatial_target = plan_image_plane_target(
        observations,
        reference_index=reference_index,
        header_overrides=header_overrides,
        device=device,
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
) -> MapResult:
    """Accumulate a large map in non-overlapping spatial tiles to cap per-tile memory."""
    if not observations:
        raise ValueError("At least one observation is required")
    H, W = int(target_shape[0]), int(target_shape[1])
    th = max(int(tile_height), 1)
    tw = max(int(tile_width), 1)
    first = observations[0]
    read_dev = str(map_device)
    first_image, first_header = read(
        first.path, hdu=first.hdu, return_header=True, device=read_dev
    )
    first_image = _validate_image_plane(first_image)
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
    first_image, first_header = read(
        first.path, hdu=first.hdu, return_header=True, device=read_dev
    )
    first_image = _validate_image_plane(first_image)
    if target_header is not None and "CRVAL1" in target_header:
        target_shape = _target_shape_from_header(
            target_header,
            fallback=tuple(first_image.shape[-2:]),
        )
        header_dict = _normalized_header_for_shape(
            dict(target_header),
            target_shape,
            overrides=None,
        )
    else:
        header_dict = _normalized_header_for_shape(
            dict(first_header),
            tuple(first_image.shape[-2:]),
            overrides=target_header,
        )
        target_shape = _target_shape_from_header(
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
            )
        return prepared_target

    numer = torch.zeros(target_shape, dtype=torch.float32, device=device)
    denom = torch.zeros(target_shape, dtype=torch.float32, device=device)
    overlap_count = torch.zeros(target_shape, dtype=torch.int32, device=device)
    observation_mocs: list[MOC] | None = [] if footprint_order is not None else None
    median_images: list[Tensor] | None = [] if stacking_method == "median" else None
    median_support: list[Tensor] | None = [] if stacking_method == "median" else None

    for obs_idx, obs in enumerate(observations):
        if obs_idx == 0:
            image = first_image
            header_dict_obs = dict(first_header)
        else:
            image, header = read(
                obs.path, hdu=obs.hdu, return_header=True, device=read_dev
            )
            image = _validate_image_plane(image)
            header_dict_obs = dict(header)
        need_prepared_target = (
            value_semantics == "flux"
            or footprint_order is not None
            or header_dict_obs != header_dict
            or tuple(image.shape) != target_shape
        )
        source_wcs = None
        if need_prepared_target and not (
            _headers_equivalent_for_direct_copy(header_dict_obs, header_dict)
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
            if observation_mocs is not None and projected.moc is not None:
                observation_mocs.append(projected.moc)

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
) -> SpectralCubeMapResult:
    if not observations:
        raise ValueError("At least one observation is required")
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
    first_image = _validate_image_plane(first_image)
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
            image = _validate_image_plane(image)
            header_dict_obs = dict(header)
        need_prepared_target = (
            value_semantics == "flux"
            or footprint_order is not None
            or header_dict_obs != header_dict
            or tuple(image.shape) != target_shape
        )
        source_wcs = None
        if need_prepared_target and not (
            _headers_equivalent_for_direct_copy(header_dict_obs, header_dict)
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
