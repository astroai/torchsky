"""Internal target-windowing and patch coordinates calculations."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from ..sphere.moc import MOC, nest2uniq
from ..sphere import healpix as _healpix
from ..wcs import WCS
from .operators.pixel_area import (
    pixel_area_sr_map,
    representative_pixel_area_sr,
    wrapped_delta_ra_deg,
)
from .operators.execution import source_bbox_on_target, wcs_has_distortion
from .operators.config import MapValueSemantics

_LARGE_TAN_WINDOWING_MIN_TARGET_PIXELS = 1_500_000
_LARGE_TAN_WINDOWING_EDGE_SAMPLES = 9
_LARGE_TAN_WINDOWING_MARGIN_PIXELS = 2


@dataclass
class _PreparedImagePlaneTarget:
    header: dict
    wcs: WCS
    shape: tuple[int, int]
    canvas_shape: tuple[int, int]
    canvas_origin_yx: tuple[int, int]
    ra_deg: Tensor | None = None
    dec_deg: Tensor | None = None
    center_ra_deg: float = 0.0
    center_dec_deg: float = 0.0
    cap_radius_deg: float = 0.0
    pixel_area_sr: Tensor | None = None
    representative_pixel_area_sr: float | None = None
    quadrature_world_grids: tuple[tuple[Tensor, Tensor], ...] | None = None


def _build_target_world_grid(
    target_wcs: WCS,
    height: int,
    width: int,
    device: torch.device,
    *,
    subsample: int = 1,
) -> tuple[Tensor, Tensor]:
    grid_device = torch.device("cpu") if device.type == "mps" else device
    y, x = torch.meshgrid(
        torch.arange(0, height, subsample, dtype=torch.float64, device=grid_device),
        torch.arange(0, width, subsample, dtype=torch.float64, device=grid_device),
        indexing="ij",
    )
    ra, dec = target_wcs.pixel_to_world(x, y)
    output_dtype = torch.float32 if device.type == "mps" else ra.dtype
    return ra.to(device=device, dtype=output_dtype), dec.to(
        device=device, dtype=output_dtype
    )


def _spherical_cap_radius_from_world_grid(
    ra_deg: Tensor,
    dec_deg: Tensor,
    *,
    center_ra_deg: float,
    center_dec_deg: float,
) -> float:
    dra = wrapped_delta_ra_deg(
        ra_deg, torch.tensor(center_ra_deg, dtype=ra_deg.dtype, device=ra_deg.device)
    )
    cos_dec = torch.cos(dec_deg * (math.pi / 180.0))
    dra_cos = dra * cos_dec
    dec0 = torch.tensor(center_dec_deg, dtype=dec_deg.dtype, device=dec_deg.device)
    ddec = dec_deg - dec0
    chord = torch.sqrt(dra_cos * dra_cos + ddec * ddec)
    return float(torch.max(chord).item())


def _quadrature_target_world_grid(
    target_wcs: WCS,
    shape: tuple[int, int],
    *,
    device: torch.device,
    offset_x: float,
    offset_y: float,
) -> tuple[Tensor, Tensor]:
    height, width = int(shape[0]), int(shape[1])
    grid_device = torch.device("cpu") if device.type == "mps" else device
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float64, device=grid_device) + float(offset_y),
        torch.arange(width, dtype=torch.float64, device=grid_device) + float(offset_x),
        indexing="ij",
    )
    ra, dec = target_wcs.pixel_to_world(x, y)
    output_dtype = torch.float32 if device.type == "mps" else ra.dtype
    return ra.to(device=device, dtype=output_dtype), dec.to(
        device=device, dtype=output_dtype
    )


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


def _prepare_target(
    *,
    header: dict,
    target_wcs: WCS,
    target_shape: tuple[int, int],
    device: torch.device,
    canvas_shape: tuple[int, int] | None = None,
    canvas_origin_yx: tuple[int, int] = (0, 0),
    include_pixel_area: bool = False,
    include_representative_pixel_area: bool = False,
    include_quadrature: bool = False,
    include_world_grid: bool = True,
) -> _PreparedImagePlaneTarget:
    center_ra_deg = float(target_wcs.alpha0)
    center_dec_deg = float(target_wcs.delta0)
    h, w = int(target_shape[0]), int(target_shape[1])
    canvas_h, canvas_w = (
        (h, w)
        if canvas_shape is None
        else (int(canvas_shape[0]), int(canvas_shape[1]))
    )
    origin_y, origin_x = int(canvas_origin_yx[0]), int(canvas_origin_yx[1])
    if (
        canvas_h <= 0
        or canvas_w <= 0
        or origin_y < 0
        or origin_x < 0
        or origin_y + h > canvas_h
        or origin_x + w > canvas_w
    ):
        raise ValueError("target window must lie inside its science canvas")

    cap_subsample = max(1, min(h, w) // 100)
    if cap_subsample > 1:
        ra_sample, dec_sample = _build_target_world_grid(
            target_wcs, h, w, device=device, subsample=cap_subsample,
        )
        cap_radius_deg = _spherical_cap_radius_from_world_grid(
            ra_sample, dec_sample,
            center_ra_deg=center_ra_deg,
            center_dec_deg=center_dec_deg,
        )
    else:
        cap_radius_deg = 0.0

    if include_world_grid:
        ra_deg, dec_deg = _build_target_world_grid(
            target_wcs, h, w, device=device,
        )
    else:
        ra_deg, dec_deg = None, None

    return _PreparedImagePlaneTarget(
        header=header,
        wcs=target_wcs,
        shape=target_shape,
        canvas_shape=(canvas_h, canvas_w),
        canvas_origin_yx=(origin_y, origin_x),
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        center_ra_deg=center_ra_deg,
        center_dec_deg=center_dec_deg,
        cap_radius_deg=cap_radius_deg,
        pixel_area_sr=(
            pixel_area_sr_map(target_wcs, target_shape, device=device)
            if include_pixel_area
            else None
        ),
        representative_pixel_area_sr=(
            representative_pixel_area_sr(target_wcs, target_shape, device=device)
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


def _slice_prepared_target(
    target: _PreparedImagePlaneTarget,
    *,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
) -> _PreparedImagePlaneTarget:
    sub_header, sub_shape = _subtarget_header_for_window(
        target.header,
        full_shape=target.shape,
        y0=y0,
        y1=y1,
        x0=x0,
        x1=x1,
    )
    quadrature = None
    if target.quadrature_world_grids is not None:
        quadrature = tuple(
            (ra[y0:y1, x0:x1], dec[y0:y1, x0:x1])
            for ra, dec in target.quadrature_world_grids
        )
    pixel_area = (
        None if target.pixel_area_sr is None else target.pixel_area_sr[y0:y1, x0:x1]
    )
    ra_slice = dec_slice = None
    if target.ra_deg is not None:
        ra_slice = target.ra_deg[y0:y1, x0:x1]
        dec_slice = target.dec_deg[y0:y1, x0:x1]
    return _PreparedImagePlaneTarget(
        header=sub_header,
        wcs=WCS(sub_header),
        shape=sub_shape,
        canvas_shape=target.canvas_shape,
        canvas_origin_yx=(
            target.canvas_origin_yx[0] + y0,
            target.canvas_origin_yx[1] + x0,
        ),
        ra_deg=ra_slice,
        dec_deg=dec_slice,
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
    if wcs_has_distortion(target.wcs):
        return target, None
    min_pixels = _LARGE_TAN_WINDOWING_MIN_TARGET_PIXELS
    edge_samples = _LARGE_TAN_WINDOWING_EDGE_SAMPLES
    margin_pixels = (
        max(_LARGE_TAN_WINDOWING_MARGIN_PIXELS, 6)
        if value_semantics == "flux"
        else _LARGE_TAN_WINDOWING_MARGIN_PIXELS
    )

    if int(target.shape[0]) * int(target.shape[1]) < min_pixels:
        return target, None
    bbox = source_bbox_on_target(
        source_wcs,
        source_shape,
        target.wcs,
        target.shape,
        margin=margin_pixels,
        edge_samples=edge_samples,
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
    if target.ra_deg is not None:
        ra_deg, dec_deg = target.ra_deg, target.dec_deg
    else:
        ra_deg, dec_deg = _build_target_world_grid(
            target.wcs, *target.shape, device=weight_map.device,
        )
    pix = _healpix.ang2pix(
        nside,
        ra_deg[valid],
        dec_deg[valid],
        nest=True,
        lonlat=True,
    )
    return MOC(nest2uniq(nside, pix.to(torch.int64)))


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


def canvas_footprint_mask(
    world_ra_deg: Tensor,
    world_dec_deg: Tensor,
    *,
    template_header,
    chunk: int = 1 << 20,
) -> Tensor:
    """Boolean mask marking world positions inside a FITS-template canvas.

    Parses ``template_header`` once with Torchsky's native WCS and evaluates
    pixel coordinates per chunk; True means finite RA/Dec projecting to a pixel
    center within ``[-0.5, N-0.5]`` on both axes (the template footprint
    including its border). The calculation stays on the input device and the
    returned bool tensor preserves the inputs' shape and device.
    """
    ra = torch.as_tensor(world_ra_deg)
    dec = torch.as_tensor(world_dec_deg)
    if ra.shape != dec.shape:
        raise ValueError("world_ra_deg and world_dec_deg must have matching shapes")
    if ra.device != dec.device:
        raise ValueError("world_ra_deg and world_dec_deg must share a device")
    dtype = torch.promote_types(ra.dtype, dec.dtype)
    if not dtype.is_floating_point:
        dtype = torch.float64
    ra = ra.to(dtype=dtype)
    dec = dec.to(dtype=dtype)
    chunk = max(int(chunk), 1)

    naxis1 = int(template_header.get("NAXIS1", 0))
    naxis2 = int(template_header.get("NAXIS2", 0))
    if naxis1 <= 0 or naxis2 <= 0:
        raise ValueError("template_header must define positive NAXIS1 and NAXIS2")
    wcs = WCS(template_header)
    ra_flat = ra.reshape(-1)
    dec_flat = dec.reshape(-1)
    flat = torch.zeros(ra_flat.shape[0], dtype=torch.bool, device=ra.device)
    for start in range(0, ra_flat.shape[0], chunk):
        stop = min(start + chunk, ra_flat.shape[0])
        pixels = wcs.world_to_pixel_differentiable(
            torch.stack((ra_flat[start:stop], dec_flat[start:stop]), dim=-1),
            origin=0,
        )
        x, y = pixels.unbind(dim=-1)
        inside = (
            torch.isfinite(x)
            & torch.isfinite(y)
            & (x >= -0.5)
            & (x <= naxis1 - 0.5)
            & (y >= -0.5)
            & (y <= naxis2 - 0.5)
        )
        flat[start:stop] = inside
    return flat.reshape(ra.shape)
