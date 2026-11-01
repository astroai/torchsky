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
)
from .operators.execution import wcs_has_distortion
from .operators.config import MapValueSemantics

# Footprint cropping applies to every target size: the saving (the per-observation
# gather / splat work is proportional to the window, not the canvas) is unconditional
# and the bbox costs a few hundred WCS evaluations.  The name is historical (it was a
# large-TAN-only threshold) and kept because callers/tests patch it; set it to a large
# value to disable cropping.
_LARGE_TAN_WINDOWING_MIN_TARGET_PIXELS = 0
# Samples per source edge (odd so that the coarse half-sampling used for the
# curvature estimate is a subset of the fine one) and interior lattice size.
_LARGE_TAN_WINDOWING_EDGE_SAMPLES = 33
_LARGE_TAN_WINDOWING_INTERIOR_SAMPLES = 9
_LARGE_TAN_WINDOWING_MARGIN_PIXELS = 2
# Footprints whose samples lie further than this from a zenithal target's reference
# point are not safely described by their bounding box (far hemisphere folds back).
_FOOTPRINT_MAX_ZENITHAL_SEPARATION_DEG = 60.0

# Source-pixel half-width of the support of each resampling kernel; used only to pad
# the footprint window (gather kernels are masked to the source domain, so the padding
# is a safety margin, not a correctness requirement).
_KERNEL_HALF_WIDTH_SOURCE_PIXELS = {
    "nearest": 0.5,
    "bilinear": 1.0,
    "bicubic": 2.0,
    "catmull_rom": 2.0,
    "mitchell_netravali": 2.0,
    "lanczos2": 2.0,
    "lanczos3": 3.0,
    "lanczos4": 4.0,
    "lanczos5": 5.0,
}
_DEFAULT_KERNEL_HALF_WIDTH_SOURCE_PIXELS = 3.0


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
    d2r = math.pi / 180.0
    ra = ra_deg * d2r
    dec = dec_deg * d2r
    ra0 = torch.as_tensor(center_ra_deg * d2r, dtype=ra.dtype, device=ra.device)
    dec0 = torch.as_tensor(center_dec_deg * d2r, dtype=dec.dtype, device=dec.device)
    delta_ra = ra - ra0
    dot = torch.sin(dec) * torch.sin(dec0) + torch.cos(dec) * torch.cos(
        dec0
    ) * torch.cos(delta_ra)
    cross_norm = torch.sqrt(torch.clamp(1.0 - torch.square(dot), min=0.0))
    separation = torch.atan2(cross_norm, dot) / d2r
    return float(torch.max(separation).item())


def _target_cap_radius_deg(
    target_wcs: WCS,
    shape: tuple[int, int],
    *,
    device: torch.device,
    center_ra_deg: float,
    center_dec_deg: float,
) -> float:
    if wcs_has_distortion(target_wcs) or not bool(
        getattr(target_wcs, "_is_tan", False)
    ):
        return 180.0
    height, width = int(shape[0]), int(shape[1])
    grid_device = torch.device("cpu") if device.type == "mps" else device
    coord_dtype = torch.float64
    x_end = float(max(width - 1, 0))
    y_end = float(max(height - 1, 0))
    x = torch.tensor([0.0, x_end, x_end, 0.0], dtype=coord_dtype, device=grid_device)
    y = torch.tensor([0.0, 0.0, y_end, y_end], dtype=coord_dtype, device=grid_device)
    ra_deg, dec_deg = target_wcs.pixel_to_world(x, y)
    return _spherical_cap_radius_from_world_grid(
        ra_deg,
        dec_deg,
        center_ra_deg=center_ra_deg,
        center_dec_deg=center_dec_deg,
    )


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

    cap_radius_deg = _target_cap_radius_deg(
        target_wcs,
        target_shape,
        device=device,
        center_ra_deg=center_ra_deg,
        center_dec_deg=center_dec_deg,
    )

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


class _FootprintUnavailable(Exception):
    """The source footprint cannot be bounded safely; use the full target."""


def _footprint_source_samples(
    h_src: int, w_src: int, n_edge: int, n_inner: int, *, device: torch.device
) -> tuple[Tensor, Tensor, int]:
    """Source pixel coordinates tracing the pixel-border outline plus an interior lattice.

    The outline runs along the outer *pixel borders* (``-0.5 .. N-0.5``): every kernel
    only contributes for samples inside the pixel-centre domain, and the drizzle/splat
    footprints extend half a pixel beyond it.  Returns ``(x, y, n_outline)`` with the
    outline samples first; the outline is ordered so that ``[::2]`` of each edge is the
    half-resolution outline.
    """
    xs = torch.linspace(-0.5, w_src - 0.5, n_edge, dtype=torch.float64, device=device)
    ys = torch.linspace(-0.5, h_src - 0.5, n_edge, dtype=torch.float64, device=device)
    lo_x = torch.full_like(ys, -0.5)
    hi_x = torch.full_like(ys, w_src - 0.5)
    lo_y = torch.full_like(xs, -0.5)
    hi_y = torch.full_like(xs, h_src - 0.5)
    x_edge = torch.cat([xs, xs, lo_x, hi_x])
    y_edge = torch.cat([lo_y, hi_y, ys, ys])
    ix = torch.linspace(0.0, w_src - 1.0, n_inner, dtype=torch.float64, device=device)
    iy = torch.linspace(0.0, h_src - 1.0, n_inner, dtype=torch.float64, device=device)
    gy, gx = torch.meshgrid(iy, ix, indexing="ij")
    return (
        torch.cat([x_edge, gx.reshape(-1)]),
        torch.cat([y_edge, gy.reshape(-1)]),
        int(x_edge.numel()),
    )


def _footprint_bbox_on_target(
    source_wcs: WCS,
    source_shape: tuple[int, int],
    target_wcs: WCS,
    target_shape: tuple[int, int],
    *,
    margin: int,
    kernel_half_width: float,
    edge_samples: int = _LARGE_TAN_WINDOWING_EDGE_SAMPLES,
    interior_samples: int = _LARGE_TAN_WINDOWING_INTERIOR_SAMPLES,
) -> tuple[int, int, int, int] | None:
    """Bounding box ``(y0, y1, x0, x1)`` of a source image footprint on a target grid.

    The outline of the source image (pixel borders, not just corners, because
    distortion bends the edges) and an interior lattice are mapped to the target.  The
    box is padded by ``margin`` target pixels, by the kernel half-width converted to
    target pixels, and by an estimate of the bending between outline samples (the
    growth of the box when the outline sampling is doubled, i.e. 3x the Richardson
    estimate of the remaining ``O(h^2)`` sagitta error) and clipped to the target.

    Returns ``None`` when the footprint lies entirely outside the target and raises
    :class:`_FootprintUnavailable` when the mapping cannot be trusted (non-finite
    samples, differing celestial frames, a footprint that may enclose a pole of a
    non-zenithal target, or a zenithal target seen from far away).
    """
    h_src, w_src = int(source_shape[0]), int(source_shape[1])
    height, width = int(target_shape[0]), int(target_shape[1])
    if h_src < 1 or w_src < 1 or height < 1 or width < 1:
        raise _FootprintUnavailable("empty image")
    n_edge = max(int(edge_samples), 3)
    if n_edge % 2 == 0:
        n_edge += 1
    x_src, y_src, n_outline = _footprint_source_samples(
        h_src,
        w_src,
        n_edge,
        max(int(interior_samples), 2),
        device=source_wcs.device,
    )
    try:
        ra_deg, dec_deg = source_wcs.pixel_to_world(x_src, y_src)
        # Same convention as ``_convert_world_between_wcs`` (which also warns, once,
        # when the real resampling runs): convert only if the frames really differ.
        from_frame = getattr(source_wcs, "celestial_frame", "ICRS")
        to_frame = getattr(target_wcs, "celestial_frame", "ICRS")
        from_eq = getattr(source_wcs, "celestial_equinox", None)
        to_eq = getattr(target_wcs, "celestial_equinox", None)
        if not (
            from_frame == to_frame
            and (from_eq == to_eq or from_frame in {"ICRS", "GALACTIC", "ECLIPTIC"})
        ):
            ra_deg, dec_deg = source_wcs.convert_celestial(
                ra_deg, dec_deg, to_frame, to_equinox=to_eq
            )
        x_tgt, y_tgt = target_wcs.world_to_pixel(ra_deg, dec_deg)
    except Exception as exc:  # any mapping failure means "use the full target"
        raise _FootprintUnavailable(f"mapping failed: {exc}") from exc
    if not (
        bool(torch.isfinite(x_tgt).all())
        and bool(torch.isfinite(y_tgt).all())
        and bool(torch.isfinite(ra_deg).all())
        and bool(torch.isfinite(dec_deg).all())
    ):
        raise _FootprintUnavailable("non-finite footprint samples")
    ra_deg = ra_deg.to("cpu", torch.float64)
    dec_deg = dec_deg.to("cpu", torch.float64)
    x_tgt = x_tgt.to("cpu", torch.float64)
    y_tgt = y_tgt.to("cpu", torch.float64)
    if bool(getattr(target_wcs, "_is_tan", False)) or bool(
        getattr(target_wcs, "_is_tpv", False)
    ):
        # Zenithal projection: a footprint reaching far from the reference point may
        # fold over the horizon, where the box of the samples is meaningless.
        d2r = math.pi / 180.0
        a0 = float(target_wcs.alpha0) * d2r
        d0 = float(target_wcs.delta0) * d2r
        cos_sep = torch.sin(dec_deg * d2r) * math.sin(d0) + torch.cos(
            dec_deg * d2r
        ) * math.cos(d0) * torch.cos(ra_deg * d2r - a0)
        if float(cos_sep.min()) < math.cos(
            _FOOTPRINT_MAX_ZENITHAL_SEPARATION_DEG * d2r
        ):
            raise _FootprintUnavailable("footprint far from zenithal reference point")
    else:
        # A pole inside the footprint makes its image in cylindrical/pseudo-cylindrical
        # targets span the full longitude range, which edge samples do not reveal.
        d2r = math.pi / 180.0
        vec = torch.stack(
            [
                torch.cos(dec_deg * d2r) * torch.cos(ra_deg * d2r),
                torch.cos(dec_deg * d2r) * torch.sin(ra_deg * d2r),
                torch.sin(dec_deg * d2r),
            ],
            dim=-1,
        )
        centre = vec.mean(dim=0)
        norm = float(torch.linalg.norm(centre))
        if norm < 1.0e-3:
            raise _FootprintUnavailable("footprint spans the sphere")
        centre = centre / norm
        radius_deg = math.degrees(
            math.acos(max(-1.0, min(1.0, float((vec @ centre).min()))))
        )
        if abs(math.degrees(math.asin(max(-1.0, min(1.0, float(centre[2])))))) + 2.0 * radius_deg >= 90.0:
            raise _FootprintUnavailable("footprint may enclose a pole")

    def _extent(sel_x: Tensor, sel_y: Tensor) -> tuple[float, float, float, float]:
        return (
            float(sel_x.min()),
            float(sel_x.max()),
            float(sel_y.min()),
            float(sel_y.max()),
        )

    xmin, xmax, ymin, ymax = _extent(x_tgt, y_tgt)
    # Half-resolution outline (every other outline sample) + interior lattice.
    coarse = torch.cat(
        [
            torch.cat(
                [
                    torch.arange(k * n_edge, (k + 1) * n_edge, 2)
                    for k in range(4)
                ]
            ),
            torch.arange(n_outline, int(x_tgt.numel())),
        ]
    )
    cxmin, cxmax, cymin, cymax = _extent(x_tgt[coarse], y_tgt[coarse])
    growth = max(cxmin - xmin, xmax - cxmax, cymin - ymin, ymax - cymax, 0.0)
    ratio = max(xmax - xmin, ymax - ymin) / float(max(h_src, w_src))
    pad = int(margin) + math.ceil(kernel_half_width * max(ratio, 1.0)) + int(
        math.ceil(growth)
    )
    x0 = max(math.floor(xmin) - pad, 0)
    x1 = min(math.ceil(xmax) + pad + 1, width)
    y0 = max(math.floor(ymin) - pad, 0)
    y1 = min(math.ceil(ymax) + pad + 1, height)
    if x0 >= x1 or y0 >= y1:
        return None
    return y0, y1, x0, x1


def _target_supports_window(target_wcs: WCS) -> bool:
    """True when the target WCS survives a CRPIX shift (header-only distortions)."""
    return not (
        getattr(target_wcs, "cpdis1", None) is not None
        or getattr(target_wcs, "cpdis2", None) is not None
        or getattr(target_wcs, "det2im1", None) is not None
        or getattr(target_wcs, "det2im2", None) is not None
        or bool(getattr(target_wcs, "_is_tnx", False))
        or bool(getattr(target_wcs, "_is_zpx", False))
    )


def _window_large_tan_target(
    *,
    source_wcs: WCS,
    source_shape: tuple[int, int],
    target: _PreparedImagePlaneTarget,
    value_semantics: MapValueSemantics,
    resampling_kernel: str | None = None,
) -> tuple[_PreparedImagePlaneTarget | None, tuple[int, int, int, int] | None]:
    """Restrict a prepared target to the footprint of one source image.

    Works for every target size and WCS type (TAN, SIP, TPV, ...): the footprint is
    bounded from the mapped source outline (see :func:`_footprint_bbox_on_target`) and
    the target is sliced to that box.  Pixels outside the box cannot receive any
    contribution from the observation, so the windowed result equals the full-target
    result inside the box and is zero outside.  Falls back to the full target when the
    footprint cannot be bounded safely.  Returns ``(None, (0, 0, 0, 0))`` when the
    footprint misses the target entirely.
    """
    if int(target.shape[0]) * int(target.shape[1]) < _LARGE_TAN_WINDOWING_MIN_TARGET_PIXELS:
        return target, None
    if not _target_supports_window(target.wcs):
        return target, None
    margin_pixels = (
        max(_LARGE_TAN_WINDOWING_MARGIN_PIXELS, 6)
        if value_semantics == "flux"
        else _LARGE_TAN_WINDOWING_MARGIN_PIXELS
    )
    half_width = _KERNEL_HALF_WIDTH_SOURCE_PIXELS.get(
        str(resampling_kernel), _DEFAULT_KERNEL_HALF_WIDTH_SOURCE_PIXELS
    )
    try:
        bbox = _footprint_bbox_on_target(
            source_wcs,
            source_shape,
            target.wcs,
            target.shape,
            margin=margin_pixels,
            kernel_half_width=half_width,
        )
    except _FootprintUnavailable:
        return target, None
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
