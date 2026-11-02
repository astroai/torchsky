"""Image-plane-first mapmaking primitives."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor

from ..device import require_float64
from ..sphere.moc import MOC
from ..wcs import WCS
from .box_overlap import box_overlap_taps
from .fft_affine_ldu import resample_fft_ldu_affine_2d
from .fft_resample import resample_diagonal_affine_fft2d, resample_translation_fft2d
from . import interpolated_mapping as _interpolated_mapping_module
from .interpolated_mapping import (
    interpolated_inverse_mapping,
    resolve_mapping_mode,
)

# Header probe for area scaling: recoverable failures when reading metadata only.
from .observation import (
    ImageObservation,
)
from .operators.config import (
    MapResamplingKernel,
    MapValueSemantics,
)
from .operators.execution import (
    PreparedImageObservationData,
    prepare_image_observation_data,
    project_image_observation_geometry,
)
from .operators.geometry import ForwardSplatGeometryOp
from .operators.image_grid import (
    headers_equivalent_for_direct_copy,
    validate_image_plane_2d,
)
from .operators.pixel_area import (
    pixel_area_sr_at,
    pixel_area_sr_map,
    representative_pixel_area_sr,
)
from .operators.resampling import (
    evaluate_tan_projective_homography,
    finalize_periodic_resampled_weighted,
    infer_diagonal_scale_shift_pixel_map,
    infer_fft_translation_shift_yx,
    infer_linear_affine_pixel_map,
    infer_tan_projective_homography,
    sample_image_plane_batched_channels,
    sample_image_plane_diagonal_variance,
    sample_separable_with_variance,
    supports_fused_variance,
)
from .periodic_affine import resample_linear_affine_periodic_bilinear_2d
from .spectral_affine import resample_linear_affine_spectral_2d
from .spectral_nonuniform import resample_nonuniform_spectral_2d
from .spectral_nufft import resample_nonuniform_spectral_fast_2d
from .windowing import (
    _LARGE_TAN_WINDOWING_EDGE_SAMPLES,  # noqa: F401
    _LARGE_TAN_WINDOWING_MARGIN_PIXELS,  # noqa: F401
    _LARGE_TAN_WINDOWING_MIN_TARGET_PIXELS,  # noqa: F401
    _build_target_world_grid,
    _PreparedImagePlaneTarget,
    _quadrature_target_world_grids,
    _weight_map_to_moc,
    _window_large_tan_target,
)



from .calibration import (  # noqa: F401 - re-exported helpers
    _accumulate_variance_weight,
    _apply_observation_calibration,
    _context_provenance,
    _effective_observation_photometric_scale,
    _linear_inverse_variance,
    _load_observation_weight_map,
    _local_noise_correlation_factor,
    _observation_operator_provenance,
    _set_context_bit,
    _should_apply_surface_brightness_area_scaling,
)

from .image_plane_common import (
    _ProjectedImagePlaneObservation,
    _convert_world_between_wcs,
    _is_splat_kernel,
    _splat_for_interpolation,
    load_observation_image_and_header,
)

_DRIZZLE_FORWARD_CHUNK_PIXELS = 262_144
_FORWARD_SOURCE_RATIO = 0.2  # forward projection when source < 20% of target
# Source pixels a gather kernel can read on either side of floor(x): its
# support radius plus one pixel of margin.  Kernels absent here (spectral,
# FFT, periodic) read the whole image.
_SOURCE_TAP_RADIUS: dict[str, int] = {
    "nearest": 2,
    "bilinear": 2,
    "bicubic": 3,
    "catmull_rom": 3,
    "mitchell_netravali": 3,
    "lanczos2": 3,
    "lanczos3": 4,
    "lanczos4": 5,
    "lanczos5": 6,
}


def _source_tap_window(
    x_src: Tensor,
    y_src: Tensor,
    shape: tuple[int, int],
    *,
    radius: int,
) -> tuple[int, int, int, int]:
    """Smallest source window ``(y0, y1, x0, x1)`` holding every kernel tap.

    Coordinates are clamped to the image first: samples outside it are masked
    after sampling, so only the taps of in-image samples need valid values.
    """
    h_src, w_src = int(shape[0]), int(shape[1])
    finite = torch.isfinite(x_src) & torch.isfinite(y_src)
    if not bool(finite.any()):
        return 0, 0, 0, 0
    xs = x_src[finite].clamp(0.0, float(w_src - 1))
    ys = y_src[finite].clamp(0.0, float(h_src - 1))
    bounds = torch.stack((xs.min(), xs.max(), ys.min(), ys.max())).tolist()
    x0 = max(math.floor(bounds[0]) - radius, 0)
    x1 = min(math.floor(bounds[1]) + radius + 1, w_src)
    y0 = max(math.floor(bounds[2]) - radius, 0)
    y1 = min(math.floor(bounds[3]) + radius + 1, h_src)
    return y0, y1, x0, x1


def _flux_weighted_planes_in_tap_window(
    image: Tensor,
    source_weight: Tensor,
    source_wcs: WCS,
    x_src: Tensor,
    y_src: Tensor,
    *,
    radius: int,
) -> tuple[Tensor, Tensor]:
    """Flux-weighted planes ``image * w * area`` and ``w * area**2``.

    Exact inside the tap window of the sampled positions, zero outside it (no
    tap of an in-image sample reads there).
    """
    shape = (int(image.shape[0]), int(image.shape[1]))
    window = _source_tap_window(x_src, y_src, shape, radius=radius)
    y0, y1, x0, x1 = window
    weighted_image = torch.zeros_like(image)
    weighted_weight = torch.zeros_like(source_weight)
    if y1 > y0 and x1 > x0:
        area = pixel_area_sr_map(
            source_wcs,
            shape,
            device=image.device,
            window=window,
        )
        weight = source_weight[y0:y1, x0:x1]
        weighted_image[y0:y1, x0:x1] = image[y0:y1, x0:x1] * weight * area
        weighted_weight[y0:y1, x0:x1] = weight * (area * area)
    return weighted_image, weighted_weight


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
    # Only a scale is needed (the round-trip tolerance): the median over a 33 x 33 lattice of pixel centres
    # matches the full-map median to far better than the tolerance, without a detector-sized area map.
    h, w = int(shape[0]), int(shape[1])
    ys = torch.linspace(0.0, float(max(h - 1, 0)), min(h, 33), dtype=torch.float64)
    xs = torch.linspace(0.0, float(max(w - 1, 0)), min(w, 33), dtype=torch.float64)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    area_sr = pixel_area_sr_at(source_wcs, xx.reshape(-1), yy.reshape(-1))
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


def _use_interpolated_mapping(
    source_wcs: WCS,
    mapping: str | None,
    *,
    device: torch.device | None = None,
    n_pixels: int | None = None,
) -> bool:
    """Interpolate the target -> source pixel mapping for distorted / non-TAN sources.

    Plain TAN sources keep the analytic path (homography or exact world -> pixel,
    both cheap); SIP and every non-TAN projection (TPV, ...) have an iterative or
    per-pixel inverse that the verified coarse lattice replaces.

    On an accelerator the exact inverse is one wide data-parallel solve, while the
    lattice is a fixed chain of small launch-bound solves; the default mapping then
    switches to the lattice only for windows of at least
    :data:`~torchsky.map.interpolated_mapping.MAPPING_MIN_PIXELS_ACCELERATOR` pixels.  An explicit ``mapping=`` is honoured.
    """
    if resolve_mapping_mode(mapping) != "interpolated":
        return False
    if (
        mapping is None
        and device is not None
        and device.type != "cpu"
        and n_pixels is not None
        and n_pixels < _interpolated_mapping_module.MAPPING_MIN_PIXELS_ACCELERATOR
    ):
        return False
    return bool(
        getattr(source_wcs, "sip", None) is not None
        or not getattr(source_wcs, "_is_tan", False)
    )


def _make_target_to_source_evaluator(
    source_wcs: WCS,
    target_wcs: WCS,
    image_shape: tuple[int, int],
    *,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
):
    """Exact target pixel -> source pixel mapping as ``(x_src, y_src, ok)``.

    ``ok`` is finite-ness and, for SIP sources, the same round-trip consistency gate
    as :func:`_roundtrip_validity_mask` (evaluated at every point; the in-bounds test
    is applied separately by the sampler).
    """
    roundtrip = _requires_roundtrip_validity_check(source_wcs)
    max_err_arcsec = 0.0
    if roundtrip:
        max_err_arcsec = max(
            _source_pixel_diameter_arcsec(
                source_wcs, image_shape, device=torch.device("cpu")
            ),
            1.0e-3,
        )

    def evaluate(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        shape = x.shape
        ra, dec = target_wcs.pixel_to_world(
            x.reshape(-1) + float(offset_x), y.reshape(-1) + float(offset_y)
        )
        ra, dec = _convert_world_between_wcs(ra, dec, target_wcs, source_wcs)
        xs, ys = source_wcs.world_to_pixel(ra, dec)
        ok = torch.isfinite(xs) & torch.isfinite(ys)
        if roundtrip:
            ra_back, dec_back = source_wcs.pixel_to_world(xs, ys)
            dra = (ra_back - ra) * torch.cos(torch.deg2rad(dec))
            ddec = dec_back - dec
            err_arcsec = torch.sqrt(dra * dra + ddec * ddec) * 3600.0
            ok = ok & (err_arcsec <= max_err_arcsec)
        return xs.reshape(shape), ys.reshape(shape), ok.reshape(shape)

    return evaluate


def _interpolated_source_pixels(
    source_wcs: WCS,
    target_wcs: WCS,
    target_shape: tuple[int, int],
    image_shape: tuple[int, int],
    *,
    device: torch.device,
    offsets: tuple[tuple[float, float], ...] = ((0.0, 0.0),),
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Verified-interpolated source pixel coordinates for each target sub-pixel offset.

    Returns ``(x_src, y_src, valid)`` with shape ``(H, W)`` (one offset) or
    ``(len(offsets), H, W)``; ``valid`` is the SIP round-trip validity (``None`` when
    the source has no round-trip gate).
    """
    map_device = torch.device("cpu") if device.type == "mps" else device
    h_src, w_src = image_shape
    xs, ys, oks = [], [], []
    for off_x, off_y in offsets:
        exact = _make_target_to_source_evaluator(
            source_wcs,
            target_wcs,
            image_shape,
            offset_x=off_x,
            offset_y=off_y,
        )

        def evaluate(x: Tensor, y: Tensor, _exact=exact) -> tuple[Tensor, Tensor, Tensor]:
            # The WCS follows its own device; the lattice lives on ``map_device``.
            x_s, y_s, ok = _exact(x, y)
            if x_s.device != map_device:
                x_s, y_s, ok = (
                    x_s.to(device=map_device),
                    y_s.to(device=map_device),
                    ok.to(device=map_device),
                )
            return x_s, y_s, ok

        x_src, y_src, ok = interpolated_inverse_mapping(
            evaluate,
            int(target_shape[0]),
            int(target_shape[1]),
            device=map_device,
            bounds_xy=(float(w_src - 1), float(h_src - 1)),
        )
        xs.append(x_src)
        ys.append(y_src)
        oks.append(ok)
    x_all = xs[0] if len(xs) == 1 else torch.stack(xs, dim=0)
    y_all = ys[0] if len(ys) == 1 else torch.stack(ys, dim=0)
    valid = None
    if _requires_roundtrip_validity_check(source_wcs):
        valid = oks[0] if len(oks) == 1 else torch.stack(oks, dim=0)
    if device.type == "mps":
        x_all = x_all.to(device=device, dtype=torch.float32)
        y_all = y_all.to(device=device, dtype=torch.float32)
        valid = None if valid is None else valid.to(device)
    return x_all, y_all, valid


def _resample_weighted_to_target(
    image: Tensor,
    source_weight: Tensor,
    source_wcs: WCS,
    target: _PreparedImagePlaneTarget,
    *,
    resampling_kernel: MapResamplingKernel,
    value_semantics: MapValueSemantics,
    return_variance_weight: bool = False,
    mapping: str | None = None,
) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor | None]:
    """Gather-resample one observation onto ``target``.

    ``mapping`` selects how the target -> source pixel mapping of distorted sources is
    obtained: ``"interpolated"`` (verified coarse lattice, see
    :mod:`torchsky.map.interpolated_mapping`) or ``"exact"`` (per-pixel inverse);
    ``None`` uses the module default.
    """
    image = validate_image_plane_2d(image).to(dtype=torch.float32)
    source_weight = validate_image_plane_2d(source_weight).to(dtype=torch.float32)

    h_src, w_src = image.shape
    target_representative_area_sr = target.representative_pixel_area_sr

    # Reproject weighted signal and weight separately, then form the weighted mean
    # only after all observations have been accumulated on the target grid.
    # Flux semantics weight every source pixel by its exact solid angle.  For
    # gather kernels with a bounded tap radius only the source pixels inside the
    # taps of the sampled positions matter, so the area map (a spherical pixel
    # polygon per pixel on distorted WCS) is evaluated on that window only.
    windowed_flux_area = (
        value_semantics == "flux" and resampling_kernel in _SOURCE_TAP_RADIUS
    )
    if windowed_flux_area:
        # Placeholders: built from the mapped positions in _sample_on_source_grid,
        # the only consumer for these kernels.
        placeholder = image.new_empty(0)
        weighted_image = weighted_weight = source_numerator_variance = placeholder
    elif value_semantics == "flux":
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
    # Periodic/FFT resamplers spread round-off over the whole frame, so pixels
    # outside the source footprint need an explicit coverage mask even when no
    # flux support is tracked (cuFFT round-off is not exactly zero there).
    periodic_support = (source_weight > 0.0).to(dtype=torch.float32)
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
        sampled_support = resample_translation_fft2d(periodic_support, **shift_kw)
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
        sampled_support = resample_diagonal_affine_fft2d(periodic_support, **aff_kw)
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
        sampled_support = resample_linear_affine_periodic_bilinear_2d(periodic_support, **aff_kw)
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
            sampled_support = resample_fft_ldu_affine_2d(periodic_support, **aff_kw)
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
        sampled_support = resample_linear_affine_spectral_2d(periodic_support, **aff_kw)
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
        valid_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor | None]:
        nonlocal weighted_image, weighted_weight, source_numerator_variance
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
        if windowed_flux_area:
            weighted_image, weighted_weight = _flux_weighted_planes_in_tap_window(
                image,
                source_weight,
                source_wcs,
                x_src,
                y_src,
                radius=_SOURCE_TAP_RADIUS[resampling_kernel],
            )
            source_numerator_variance = weighted_weight
        sampled_variance_weight = None
        fused_variance = None
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
            stacked_planes = (
                torch.stack([weighted_image, weighted_weight, support_image], dim=0)
                if support_image is not None
                else torch.stack([weighted_image, weighted_weight], dim=0)
            )
            if (
                return_variance_weight
                and coefficient_variance_supported
                and x_src.ndim == 2
                and supports_fused_variance(resampling_kernel)
            ):
                # One pass shares tap indices/coefficients between the sampled
                # channels and the diagonal variance (masked below).
                sampled_planes, fused_variance = sample_separable_with_variance(
                    stacked_planes,
                    x_src=x_src,
                    y_src=y_src,
                    kernel=resampling_kernel,
                    source_variance=source_numerator_variance,
                )
            else:
                x_norm = (2.0 * x_src / max(w_src - 1, 1)) - 1.0
                y_norm = (2.0 * y_src / max(h_src - 1, 1)) - 1.0
                grid = torch.stack([x_norm, y_norm], dim=-1).to(dtype=torch.float32)
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
        if _requires_roundtrip_validity_check(source_wcs) and valid_mask is not None:
            in_bounds = in_bounds & valid_mask
        elif _requires_roundtrip_validity_check(source_wcs):
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
            if fused_variance is not None:
                sampled_variance_weight = torch.where(
                    sample_valid, fused_variance, torch.zeros_like(fused_variance)
                )
            else:
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

    def _tpv_disjoint_from_target() -> bool:
        if not (
            bool(getattr(source_wcs, "_is_tpv", False))
            and getattr(source_wcs, "sip", None) is None
            and getattr(target.wcs, "celestial_frame", "ICRS")
            == getattr(source_wcs, "celestial_frame", "ICRS")
        ):
            return False
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
        return center_sep_deg > source_cap_radius_deg + target.cap_radius_deg

    def _zero_result(like_shape) -> tuple[Tensor, Tensor, Tensor | None] | tuple[Tensor, Tensor]:
        zeros = torch.zeros(like_shape, dtype=torch.float32, device=image.device)
        return _result(
            zeros,
            zeros,
            zeros if coefficient_variance_supported else None,
        )

    def _sample_on_world_grid(
        ra_deg: Tensor, dec_deg: Tensor
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor | None]:
        ra_conv, dec_conv = _convert_world_between_wcs(
            ra_deg, dec_deg, target.wcs, source_wcs
        )
        flat_ra = ra_conv.reshape(-1)
        flat_dec = dec_conv.reshape(-1)
        if _tpv_disjoint_from_target():
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

    use_interpolated_mapping = _use_interpolated_mapping(
        source_wcs,
        mapping,
        device=image.device,
        n_pixels=int(target.shape[0]) * int(target.shape[1]),
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
        if use_interpolated_mapping:
            if _tpv_disjoint_from_target():
                return _zero_result((h_tgt, w_tgt))
            x_src, y_src, valid = _interpolated_source_pixels(
                source_wcs,
                target.wcs,
                (h_tgt, w_tgt),
                (h_src, w_src),
                device=image.device,
            )
            return _sample_on_source_grid(x_src, y_src, valid_mask=valid)
        if (
            min(h_tgt, w_tgt) < 500
            or _requires_roundtrip_validity_check(source_wcs)
            or resolve_mapping_mode(mapping) != "interpolated"
        ):
            ra_g, dec_g = _build_target_world_grid(
                target.wcs,
                h_tgt,
                w_tgt,
                device=image.device,
            )
            return _sample_on_world_grid(ra_g, dec_g)
        # Verified lattice (every pixel within the stated tolerance of the exact
        # mapping, last row/column included) instead of an unverified coarse grid.
        x_src, y_src, _ = _interpolated_source_pixels(
            source_wcs,
            target.wcs,
            (h_tgt, w_tgt),
            (h_src, w_src),
            device=image.device,
        )
        return _sample_on_source_grid(x_src, y_src, flat_ra=None, flat_dec=None)

    device = target.wcs.device
    accum_numer = torch.zeros(target.shape, dtype=torch.float32, device=device)
    accum_weight = torch.zeros(target.shape, dtype=torch.float32, device=device)
    if use_interpolated_mapping:
        if _tpv_disjoint_from_target():
            return _zero_result(tuple(target.shape))
        half = 0.5 / math.sqrt(3.0)
        x_src_b, y_src_b, valid = _interpolated_source_pixels(
            source_wcs,
            target.wcs,
            tuple(target.shape),
            (h_src, w_src),
            device=image.device,
            offsets=tuple((ox, oy) for oy in (-half, half) for ox in (-half, half)),
        )
        sub_result = _sample_on_source_grid(x_src_b, y_src_b, valid_mask=valid)
        sub_numer, sub_weight = sub_result[:2]
        sub_variance_weight = sub_result[2] if len(sub_result) == 3 else None
        accum_numer = sub_numer.mean(dim=0)
        accum_weight = sub_weight.mean(dim=0)
        accum_numer = torch.where(
            accum_weight > 0.0, accum_numer, torch.zeros_like(accum_numer)
        )
        return _result(accum_numer, accum_weight, sub_variance_weight)
    quadrature_world_grids = target.quadrature_world_grids
    if quadrature_world_grids is None:
        # Same device as grids prepared with the target (the image's device).
        quadrature_world_grids = _quadrature_target_world_grids(
            target.wcs,
            target.shape,
            device=image.device,
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


def _forward_box_target_widths(
    source_wcs: WCS,
    target_wcs: WCS,
    x_src: Tensor,
    y_src: Tensor,
    x_tgt: Tensor,
    y_tgt: Tensor,
    footprint: float,
) -> tuple[Tensor, Tensor]:
    """Target-pixel extent of a ``footprint``-wide source-pixel box along x and y.

    The local Jacobian ``d(x_t, y_t) / d(x_s, y_s)`` comes from one-pixel forward
    differences (exact for affine maps); the box spans ``footprint`` times the length
    of each target axis' source-pixel row of the Jacobian.
    """
    n = int(x_src.numel())
    xs = torch.cat([x_src + 1.0, x_src])
    ys = torch.cat([y_src, y_src + 1.0])
    ra, dec = source_wcs.pixel_to_world(xs, ys)
    ra, dec = _convert_world_between_wcs(ra, dec, source_wcs, target_wcs)
    xt, yt = target_wcs.world_to_pixel(ra, dec)
    xt = xt.to(dtype=torch.float32)
    yt = yt.to(dtype=torch.float32)
    dxdx, dxdy = xt[:n] - x_tgt, xt[n:] - x_tgt
    dydx, dydy = yt[:n] - y_tgt, yt[n:] - y_tgt
    width_x = footprint * torch.hypot(dxdx, dxdy)
    width_y = footprint * torch.hypot(dydx, dydy)
    return width_x, width_y


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

    require_float64(image.device, what="forward splat projection (drizzle)")
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
        x_chunk = x_chunk[finite]
        y_chunk = y_chunk[finite]
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

        # Box footprint: pixfrac source pixels wide, scaled to target pixels by the
        # local Jacobian, then overlapped exactly with the target pixel grid.
        footprint = max(float(drizzle_pixfrac), 1e-6)
        width_x, width_y = _forward_box_target_widths(
            source_wcs, target.wcs, x_chunk, y_chunk, x_tgt, y_tgt, footprint
        )
        ok = torch.isfinite(width_x) & torch.isfinite(width_y)
        if not bool(ok.all()):
            # Footprints with a non-finite Jacobian are skipped, like non-finite centres.
            x_tgt, y_tgt = x_tgt[ok], y_tgt[ok]
            width_x, width_y = width_x[ok], width_y[ok]
            numer_chunk, weight_chunk = numer_chunk[ok], weight_chunk[ok]
            if variance_chunk is not None:
                variance_chunk = variance_chunk[ok]
            if x_tgt.numel() == 0:
                continue
        first_x, taps_x = box_overlap_taps(x_tgt, width_x)
        first_y, taps_y = box_overlap_taps(y_tgt, width_y)
        for ky in range(taps_y.shape[0]):
            iy = first_y + ky
            if not bool(((iy >= 0) & (iy < height)).any()):
                continue
            for kx in range(taps_x.shape[0]):
                ix = first_x + kx
                frac = taps_x[kx] * taps_y[ky]
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
    prepared_target: _PreparedImagePlaneTarget | None = None,
    target_header: dict,
    target_shape: tuple[int, int],
    device: torch.device,
    footprint_order: int | None,
    resampling_kernel: MapResamplingKernel,
    value_semantics: MapValueSemantics,
    drizzle_pixfrac: float,
    background_estimator: Callable[[Tensor, Tensor | None], Tensor] | None = None,
    prepare_target: Callable[[], _PreparedImagePlaneTarget] | None = None,
    propagate_variance: bool = True,
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
    # Exact diagonal variance is a second sampling pass; skip it on request.
    has_explicit_noise = operator_bundle.noise is not None and propagate_variance
    direct_copy = (
        tuple(image.shape) == target_shape
        and headers_equivalent_for_direct_copy(header_dict_obs, target_header)
    )
    if prepared_target is None and (
        not direct_copy or value_semantics == "flux" or footprint_order is not None
    ):
        if prepare_target is None:
            raise RuntimeError(
                "prepared_target is required for reprojection, flux area, or a footprint"
            )
        prepared_target = prepare_target()
    if not direct_copy and source_wcs is None:
        source_wcs = WCS(header_dict_obs)
    prepared_target_view = prepared_target
    target_window = None
    if direct_copy:
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
        prepared_target_view, target_window = _window_large_tan_target(
            source_wcs=source_wcs,
            source_shape=tuple(image.shape),
            target=prepared_target,
            value_semantics=value_semantics,
            resampling_kernel=resampling_kernel,
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
    prepared_target: _PreparedImagePlaneTarget | None = None,
    target_header: dict,
    target_shape: tuple[int, int],
    device: torch.device,
    footprint_order: int | None,
    resampling_kernel: MapResamplingKernel,
    value_semantics: MapValueSemantics,
    drizzle_pixfrac: float = 1.0,
    background_estimator: Callable[[Tensor, Tensor | None], Tensor] | None = None,
    prepare_target: Callable[[], _PreparedImagePlaneTarget] | None = None,
    propagate_variance: bool = True,
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
        prepare_target=prepare_target,
        propagate_variance=propagate_variance,
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
