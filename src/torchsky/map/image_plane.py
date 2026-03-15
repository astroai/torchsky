"""Image-plane-first mapmaking primitives."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Literal, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from .. import WCS, read, write
from ..sphere.moc import MOC, nest2uniq
from ..spectral import (
    DataCube,
    SpectralAxis,
    spectral_axis_from_header,
    spectral_bin_edges_from_header,
)
from ..wcs import healpix as _healpix

MapResamplingKernel = Literal[
    "nearest",
    "bilinear",
    "bicubic",
    "lanczos3",
    "lanczos5",
    "mitchell_netravali",
    "catmull_rom",
]
MapValueSemantics = Literal["surface_brightness", "flux"]


@dataclass
class ImageObservation:
    path: str
    hdu: int | str | None = 0
    weight: float = 1.0
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
    pixel_area_sr: Tensor | None = None
    quadrature_world_grids: tuple[tuple[Tensor, Tensor], ...] | None = None


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
    support = _kernel_support(kernel)
    x_src = x_src.to(dtype=image.dtype)
    y_src = y_src.to(dtype=image.dtype)
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
    dx = x_src.unsqueeze(-1) - xs.to(dtype=image.dtype)
    dy = y_src.unsqueeze(-1) - ys.to(dtype=image.dtype)
    wx = _kernel_weight(kernel, dx)
    wy = _kernel_weight(kernel, dy)

    h_src, w_src = image.shape
    x_valid = (xs >= 0) & (xs < w_src)
    y_valid = (ys >= 0) & (ys < h_src)
    xs = xs.clamp(0, w_src - 1)
    ys = ys.clamp(0, h_src - 1)
    sampled = image[ys.unsqueeze(-1), xs.unsqueeze(-2)]
    weights = (
        wy.unsqueeze(-1)
        * wx.unsqueeze(-2)
        * y_valid.unsqueeze(-1).to(dtype=image.dtype)
        * x_valid.unsqueeze(-2).to(dtype=image.dtype)
    )
    return torch.sum(sampled * weights, dim=(-1, -2))


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


def _requires_roundtrip_validity_check(source_wcs: WCS) -> bool:
    # SIP inverse paths still benefit from an explicit round-trip consistency gate.
    # For TPV, the gate was useful while the distorted WCS dispatch could route
    # through a wrong TAN-only shortcut, but after fixing that dispatch the same
    # gate became over-rejective on realistic warped-stack benchmarks.
    return bool(getattr(source_wcs, "sip", None) is not None)


def _requires_target_pixel_quadrature(source_wcs: WCS) -> bool:
    return bool(
        getattr(source_wcs, "sip", None) is not None
        or getattr(source_wcs, "_is_tpv", False)
    )


def _source_pixel_diameter_arcsec(source_wcs: WCS, shape: tuple[int, int], *, device: torch.device) -> float:
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
    weight_map = torch.isfinite(image).to(dtype=torch.float32, device=device)

    if obs.weight_path is not None:
        source_weight = read(obs.weight_path, hdu=obs.weight_hdu)
        source_weight = _validate_image_plane(source_weight)
        if tuple(int(v) for v in source_weight.shape) != tuple(
            int(v) for v in image.shape
        ):
            raise ValueError("Weight map shape must match the observation image shape")
        source_weight = source_weight.to(device=device, dtype=torch.float32)
        source_weight = torch.where(
            torch.isfinite(source_weight) & (source_weight > 0.0),
            source_weight,
            torch.zeros_like(source_weight),
        )
        weight_map = weight_map * source_weight

    if obs.mask_path is not None:
        mask = read(obs.mask_path, hdu=obs.mask_hdu)
        mask = _validate_image_plane(mask)
        if tuple(int(v) for v in mask.shape) != tuple(int(v) for v in image.shape):
            raise ValueError("Mask shape must match the observation image shape")
        mask = mask.to(device=device, dtype=torch.float32)
        mask = torch.where(
            torch.isfinite(mask) & (mask > 0.0),
            torch.ones_like(mask),
            torch.zeros_like(mask),
        )
        weight_map = weight_map * mask

    return weight_map * float(obs.weight)


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

    # Reproject weighted signal and weight separately, then form the weighted mean
    # only after all observations have been accumulated on the target grid.
    if value_semantics == "flux":
        source_area_sr = _pixel_area_sr_map(
            source_wcs,
            tuple(int(v) for v in image.shape),
            device=image.device,
        )
        weighted_image = (
            torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
            * source_weight
            * source_area_sr
        )
        weighted_weight = source_weight * torch.square(source_area_sr)
    else:
        weighted_image = (
            torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0) * source_weight
        )
        weighted_weight = source_weight

    def _sample_on_source_grid(
        x_src: Tensor,
        y_src: Tensor,
        *,
        flat_ra: Tensor,
        flat_dec: Tensor,
    ) -> tuple[Tensor, Tensor]:
        x_norm = (2.0 * x_src / max(w_src - 1, 1)) - 1.0
        y_norm = (2.0 * y_src / max(h_src - 1, 1)) - 1.0
        grid = torch.stack([x_norm, y_norm], dim=-1).to(dtype=torch.float32)

        sampled_numer = _sample_image_plane(
            weighted_image,
            x_src=x_src,
            y_src=y_src,
            grid=grid,
            kernel=resampling_kernel,
        )
        sampled_weight = _sample_image_plane(
            weighted_weight,
            x_src=x_src,
            y_src=y_src,
            grid=grid,
            kernel=resampling_kernel,
        )
        sampled_support: Tensor | None = None
        if value_semantics == "flux":
            sampled_support = (
                F.grid_sample(
                    (source_weight > 0.0)
                    .to(dtype=torch.float32)
                    .unsqueeze(0)
                    .unsqueeze(0),
                    grid.unsqueeze(0),
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=True,
                )
                .squeeze(0)
                .squeeze(0)
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
        sampled_numer = torch.where(
            sampled_weight > 0.0,
            sampled_numer,
            torch.zeros_like(sampled_numer),
        )
        return sampled_numer, sampled_weight

    def _sample_on_world_grid(ra_deg: Tensor, dec_deg: Tensor) -> tuple[Tensor, Tensor]:
        flat_ra = ra_deg.reshape(-1)
        flat_dec = dec_deg.reshape(-1)
        x_src, y_src = source_wcs.world_to_pixel(flat_ra, flat_dec)
        x_src = x_src.reshape_as(ra_deg)
        y_src = y_src.reshape_as(dec_deg)
        return _sample_on_source_grid(
            x_src,
            y_src,
            flat_ra=flat_ra,
            flat_dec=flat_dec,
        )

    if not _requires_target_pixel_quadrature(source_wcs):
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
    flat_ra = torch.cat([sub_ra.reshape(-1) for sub_ra, _ in quadrature_world_grids], dim=0)
    flat_dec = torch.cat([sub_dec.reshape(-1) for _, sub_dec in quadrature_world_grids], dim=0)
    x_src_all, y_src_all = source_wcs.world_to_pixel(flat_ra, flat_dec)
    n_pix = target.ra_deg.numel()
    for idx in range(len(quadrature_world_grids)):
        start = idx * n_pix
        end = start + n_pix
        sub_ra, sub_dec = quadrature_world_grids[idx]
        sub_x = x_src_all[start:end].reshape_as(sub_ra)
        sub_y = y_src_all[start:end].reshape_as(sub_dec)
        sub_numer, sub_weight = _sample_on_source_grid(
            sub_x,
            sub_y,
            flat_ra=flat_ra[start:end],
            flat_dec=flat_dec[start:end],
        )
        accum_numer = accum_numer + 0.25 * sub_numer
        accum_weight = accum_weight + 0.25 * sub_weight
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


def _prepare_target(
    *,
    header: dict,
    target_wcs: WCS,
    target_shape: tuple[int, int],
    device: torch.device,
    include_pixel_area: bool = False,
) -> _PreparedImagePlaneTarget:
    ra_deg, dec_deg = _build_target_world_grid(
        target_wcs,
        target_shape[0],
        target_shape[1],
        device=device,
    )
    return _PreparedImagePlaneTarget(
        header=header,
        wcs=target_wcs,
        shape=target_shape,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        pixel_area_sr=(
            _pixel_area_sr_map(target_wcs, target_shape, device=device)
            if include_pixel_area
            else None
        ),
        quadrature_world_grids=_quadrature_target_world_grids(
            target_wcs,
            target_shape,
            device=device,
        ),
    )


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
        if not torch.isfinite(torch.tensor(band_lower, dtype=torch.float64)) or not torch.isfinite(
            torch.tensor(band_upper, dtype=torch.float64)
        ):
            raise ValueError("Observation spectral bounds must be finite")
        if band_upper < band_lower:
            raise ValueError("Observation spectral_max must be >= spectral_min")
        filt_center = (
            center if obs.transmission_center is None else float(obs.transmission_center)
        )
        if not torch.isfinite(torch.tensor(filt_center, dtype=torch.float64)):
            raise ValueError("Observation transmission_center must be finite")
        filt_sigma = (
            float("nan")
            if obs.transmission_sigma is None
            else float(obs.transmission_sigma)
        )
        if obs.transmission_sigma is not None:
            if not torch.isfinite(torch.tensor(filt_sigma, dtype=torch.float64)) or filt_sigma <= 0.0:
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
            if projection[obs_idx, band_idx] > 0.0 and observation_names[obs_idx] is not None
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


def _project_observation_to_target(
    obs: ImageObservation,
    *,
    image: Tensor | None = None,
    header_dict_obs: dict | None = None,
    prepared_target: _PreparedImagePlaneTarget | None,
    target_header: dict,
    target_shape: tuple[int, int],
    device: torch.device,
    footprint_order: int | None,
    resampling_kernel: MapResamplingKernel,
    value_semantics: MapValueSemantics,
) -> tuple[Tensor, Tensor, Tensor, MOC | None]:
    if not torch.isfinite(torch.tensor(obs.weight, dtype=torch.float64)):
        raise ValueError("Observation weights must be finite")
    if obs.weight <= 0.0:
        raise ValueError("Observation weights must be positive")

    if image is None or header_dict_obs is None:
        image, header = read(obs.path, hdu=obs.hdu, return_header=True)
        image = _validate_image_plane(image)
        header_dict_obs = dict(header)
    image = image.to(device=device)
    source_weight = _load_observation_weight_map(obs, image, device=device)
    if _headers_equivalent_for_direct_copy(header_dict_obs, target_header) and tuple(int(v) for v in image.shape) == target_shape:
        if value_semantics == "flux":
            if prepared_target is None or prepared_target.pixel_area_sr is None:
                raise RuntimeError("prepared_target with pixel area is required for flux semantics")
            sampled_weight = source_weight.to(dtype=torch.float32) * torch.square(
                prepared_target.pixel_area_sr
            )
            sampled_numer = (
                torch.nan_to_num(
                    image.to(dtype=torch.float32),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                * source_weight.to(dtype=torch.float32)
                * prepared_target.pixel_area_sr
            )
        else:
            sampled_weight = source_weight.to(dtype=torch.float32)
            sampled_numer = torch.nan_to_num(
                image.to(dtype=torch.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ) * sampled_weight
    else:
        if prepared_target is None:
            raise RuntimeError("prepared_target is required for reprojection")
        sampled_numer, sampled_weight = _resample_weighted_to_target(
            image,
            source_weight,
            WCS(header_dict_obs),
            prepared_target,
            resampling_kernel=resampling_kernel,
            value_semantics=value_semantics,
        )
    overlap = (sampled_weight > 0.0).to(torch.int32)
    moc = None
    if footprint_order is not None:
        if prepared_target is None:
            raise RuntimeError("prepared_target is required for footprint MOCs")
        moc = _weight_map_to_moc(
            sampled_weight,
            prepared_target,
            max_order=int(footprint_order),
        )
    return sampled_numer, sampled_weight, overlap, moc


def plan_image_plane_target(
    observations: Sequence[ImageObservation],
    *,
    reference_index: int = 0,
    header_overrides: Optional[dict] = None,
) -> PlannedImagePlaneTarget:
    if not observations:
        raise ValueError("At least one observation is required")
    if reference_index < 0 or reference_index >= len(observations):
        raise IndexError("reference_index is out of range")

    ref = observations[reference_index]
    image, header = read(ref.path, hdu=ref.hdu, return_header=True)
    image = _validate_image_plane(image)
    shape = tuple(int(v) for v in image.shape)
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
) -> PlannedImagePlaneCubeTarget:
    spatial_target = plan_image_plane_target(
        observations,
        reference_index=reference_index,
        header_overrides=header_overrides,
    )
    (
        observation_center,
        observation_lower,
        observation_upper,
        observation_filter_center,
        observation_filter_sigma,
        observation_names,
    ) = (
        _observation_spectral_columns(observations)
    )
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
    )


def coadd_image_plane(
    observations: Sequence[ImageObservation],
    *,
    target_header: Optional[dict] = None,
    output_path: Optional[str] = None,
    footprint_order: int | None = None,
    resampling_kernel: MapResamplingKernel = "bilinear",
    value_semantics: MapValueSemantics = "surface_brightness",
) -> MapResult:
    if not observations:
        raise ValueError("At least one observation is required")
    if footprint_order is not None and footprint_order < 0:
        raise ValueError("footprint_order must be non-negative")
    resampling_kernel = _validated_map_resampling_kernel(resampling_kernel)
    value_semantics = _validated_map_value_semantics(value_semantics)

    first = observations[0]
    first_image, first_header = read(first.path, hdu=first.hdu, return_header=True)
    first_image = _validate_image_plane(first_image)
    header_dict = _normalized_header_for_shape(
        dict(first_header),
        tuple(int(v) for v in first_image.shape[-2:]),
        overrides=target_header,
    )
    target_shape = _target_shape_from_header(
        header_dict,
        fallback=tuple(int(v) for v in first_image.shape[-2:]),
    )
    device = first_image.device
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
            )
        return prepared_target

    numer = torch.zeros(target_shape, dtype=torch.float32, device=device)
    denom = torch.zeros(target_shape, dtype=torch.float32, device=device)
    overlap_count = torch.zeros(target_shape, dtype=torch.int32, device=device)
    observation_mocs: list[MOC] | None = [] if footprint_order is not None else None

    for obs_idx, obs in enumerate(observations):
        if obs_idx == 0:
            image = first_image
            header_dict_obs = dict(first_header)
        else:
            image, header = read(obs.path, hdu=obs.hdu, return_header=True)
            image = _validate_image_plane(image)
            header_dict_obs = dict(header)
        need_prepared_target = (
            value_semantics == "flux"
            or
            footprint_order is not None
            or header_dict_obs != header_dict
            or tuple(int(v) for v in image.shape) != target_shape
        )
        sampled_numer, sampled_weight, overlap, moc = _project_observation_to_target(
            obs,
            image=image,
            header_dict_obs=header_dict_obs,
            prepared_target=ensure_prepared_target() if need_prepared_target else None,
            target_header=header_dict,
            target_shape=target_shape,
            device=device,
            footprint_order=footprint_order,
            resampling_kernel=resampling_kernel,
            value_semantics=value_semantics,
        )
        numer = numer + sampled_numer
        denom = denom + sampled_weight
        overlap_count = overlap_count + overlap
        if observation_mocs is not None and moc is not None:
            observation_mocs.append(moc)

    # Mapmaking is currently an explicit weighted-mean coadd with no hidden
    # photometric renormalization or background matching.
    denom_floor = torch.finfo(denom.dtype).tiny
    if value_semantics == "flux":
        prepared = ensure_prepared_target()
        if prepared.pixel_area_sr is None:
            raise RuntimeError("target pixel area is required for flux semantics")
        coadd = prepared.pixel_area_sr * numer / torch.clamp(denom, min=denom_floor)
        denom = denom / torch.clamp(torch.square(prepared.pixel_area_sr), min=1e-16)
    else:
        coadd = numer / torch.clamp(denom, min=denom_floor)
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
        observation_mocs=tuple(observation_mocs) if observation_mocs is not None else None,
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
    ) -> None:
        self.target_header = target_header
        self.resampling_kernel = _validated_map_resampling_kernel(resampling_kernel)
        self.value_semantics = _validated_map_value_semantics(value_semantics)

    def run(
        self,
        observations: Sequence[ImageObservation],
        *,
        output_path: Optional[str] = None,
    ) -> MapResult:
        return coadd_image_plane(
            observations,
            target_header=self.target_header,
            output_path=output_path,
            resampling_kernel=self.resampling_kernel,
            value_semantics=self.value_semantics,
        )


def coadd_image_plane_cube(
    observations: Sequence[ImageObservation],
    *,
    target: PlannedImagePlaneCubeTarget | None = None,
    target_header: Optional[dict] = None,
    footprint_order: int | None = None,
    resampling_kernel: MapResamplingKernel = "bilinear",
    value_semantics: MapValueSemantics = "surface_brightness",
    spectral_axis: SpectralAxis | None = None,
    spectral_bin_edges: Tensor | None = None,
    spectral_unit: str = "Angstrom",
    spectral_type: str = "WAVE",
    reference_index: int = 0,
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
    resampling_kernel = _validated_map_resampling_kernel(resampling_kernel)
    value_semantics = _validated_map_value_semantics(value_semantics)

    if target is None:
        target = plan_image_plane_cube_target(
            observations,
            reference_index=reference_index,
            header_overrides=target_header,
            spectral_axis=spectral_axis,
            spectral_bin_edges=spectral_bin_edges,
            spectral_unit=spectral_unit,
            spectral_type=spectral_type,
        )

    (
        observation_center,
        observation_lower,
        observation_upper,
        observation_filter_center,
        observation_filter_sigma,
        observation_names,
    ) = (
        _observation_spectral_columns(observations)
    )
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
    )
    first_image = _validate_image_plane(first_image)
    device = first_image.device
    header_dict = dict(target.header)
    target_shape = target.shape
    target_wcs = WCS(header_dict)
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
            )
        return prepared_target

    n_band = int(target.spectral_axis.values.numel())
    numer = torch.zeros((n_band, *target_shape), dtype=torch.float32, device=device)
    denom = torch.zeros((n_band, *target_shape), dtype=torch.float32, device=device)
    overlap_count = torch.zeros((n_band, *target_shape), dtype=torch.int32, device=device)
    coverage_mocs: list[MOC | None] | None = None
    if footprint_order is not None:
        coverage_mocs = [MOC(torch.empty((0,), dtype=torch.int64)) for _ in range(n_band)]

    for obs_idx, obs in enumerate(observations):
        if obs_idx == 0:
            image = first_image
            header_dict_obs = dict(first_header)
        else:
            image, header = read(obs.path, hdu=obs.hdu, return_header=True)
            image = _validate_image_plane(image)
            header_dict_obs = dict(header)
        need_prepared_target = (
            value_semantics == "flux"
            or
            footprint_order is not None
            or header_dict_obs != header_dict
            or tuple(int(v) for v in image.shape) != target_shape
        )
        sampled_numer, sampled_weight, overlap, moc = _project_observation_to_target(
            obs,
            image=image,
            header_dict_obs=header_dict_obs,
            prepared_target=ensure_prepared_target() if need_prepared_target else None,
            target_header=header_dict,
            target_shape=target_shape,
            device=device,
            footprint_order=footprint_order,
            resampling_kernel=resampling_kernel,
            value_semantics=value_semantics,
        )
        coeffs = projection[obs_idx].to(device=device, dtype=torch.float32).view(n_band, 1, 1)
        numer = numer + coeffs * sampled_numer.unsqueeze(0)
        denom = denom + coeffs * sampled_weight.unsqueeze(0)
        overlap_count = overlap_count + (coeffs > 0.0).to(torch.int32) * overlap.unsqueeze(0)
        if coverage_mocs is not None and moc is not None:
            band_indices = torch.nonzero(projection[obs_idx] > 0.0, as_tuple=False).flatten()
            for band_idx in band_indices.tolist():
                current = coverage_mocs[band_idx]
                coverage_mocs[band_idx] = moc if current is None else current.union(moc)

    denom_floor = torch.finfo(denom.dtype).tiny
    if value_semantics == "flux":
        prepared = ensure_prepared_target()
        if prepared.pixel_area_sr is None:
            raise RuntimeError("target pixel area is required for flux semantics")
        pixel_area_sr = prepared.pixel_area_sr.unsqueeze(0)
        cube = pixel_area_sr * numer / torch.clamp(denom, min=denom_floor)
        denom = denom / torch.clamp(torch.square(pixel_area_sr), min=1e-16)
    else:
        cube = numer / torch.clamp(denom, min=denom_floor)
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
    spectral_unit: str = "Angstrom",
    spectral_type: str = "WAVE",
) -> MultiBandMapResult:
    cube = coadd_image_plane_cube(
        observations,
        target_header=target_header,
        footprint_order=footprint_order,
        resampling_kernel=resampling_kernel,
        value_semantics=value_semantics,
        spectral_axis=None,
        spectral_bin_edges=None,
        spectral_unit=spectral_unit,
        spectral_type=spectral_type,
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
