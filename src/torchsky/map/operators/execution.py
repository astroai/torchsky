"""Executable operator helpers shared by image-plane mapmaking paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import torch
from torch import Tensor

from .bundle import ObservationOperatorBundle, image_observation_operator_bundle
from .geometry import (
    ConservativePolygonRemapOp,
    ExactDrizzleGeometryOp,
    ForwardSplatGeometryOp,
)
from .geometry_conservative import conservative_polygon_project
from .geometry_drizzle import exact_point_drizzle_project, exact_square_drizzle_project


LoadWeightMapFn = Callable[[object, Tensor, torch.device], Tensor]
ApplyCalibrationFn = Callable[[object, Tensor, Tensor], tuple[Tensor, Tensor]]
ResampleToTargetFn = Callable[..., tuple[Tensor, Tensor]]
ForwardProjectFn = Callable[..., tuple[Tensor, Tensor]]


@dataclass(frozen=True)
class PreparedImageObservationData:
    image: Tensor
    source_weight: Tensor
    operator_bundle: ObservationOperatorBundle


def stack_inputs_from_projected_observation(
    projected_numer: Tensor,
    projected_weight: Tensor,
    *,
    stacking_method: str,
    value_semantics: str,
    pixel_area_sr: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Convert projected numerator/weight tensors into stack inputs."""
    if stacking_method == "weighted_mean":
        return projected_numer, projected_weight
    projected_image = torch.where(
        projected_weight > 0.0,
        projected_numer / projected_weight,
        torch.zeros_like(projected_numer),
    )
    if value_semantics == "flux":
        if pixel_area_sr is None:
            raise RuntimeError("target pixel area is required for flux semantics")
        projected_image = pixel_area_sr * projected_image
    support = (projected_weight > 0.0).to(dtype=torch.float32)
    return projected_image, support


def masked_median_stack(
    projected_images: Sequence[Tensor],
    support_masks: Sequence[Tensor],
) -> tuple[Tensor, Tensor]:
    """Take a per-pixel median over only supported projected observations."""
    if not projected_images:
        raise ValueError("median stacking requires at least one projected image")
    values = torch.stack(tuple(projected_images), dim=0)
    support = torch.stack(
        tuple(mask.to(dtype=torch.bool) for mask in support_masks), dim=0
    )
    fill = torch.full_like(values, float("inf"))
    ordered = torch.sort(torch.where(support, values, fill), dim=0).values
    count = support.sum(dim=0, dtype=torch.int64)
    safe_count = torch.clamp(count, min=1)
    lower_idx = ((safe_count - 1) // 2).unsqueeze(0)
    upper_idx = (safe_count // 2).unsqueeze(0)
    lower = torch.gather(ordered, 0, lower_idx).squeeze(0)
    upper = torch.gather(ordered, 0, upper_idx).squeeze(0)
    median = 0.5 * (lower + upper)
    median = torch.where(count > 0, median, torch.zeros_like(median))
    return median, count.to(dtype=torch.float32)


def accumulate_projected_observation(
    numer: Tensor,
    denom: Tensor,
    overlap_count: Tensor,
    *,
    projected_numer: Tensor,
    projected_weight: Tensor,
    projected_overlap: Tensor,
    stacking_method: str,
    value_semantics: str,
    pixel_area_sr: Tensor | None = None,
    target_window: tuple[int, int, int, int] | None = None,
    coeffs: Tensor | None = None,
) -> None:
    """Accumulate a projected observation into image-plane or cube stack tensors."""
    contrib_numer, contrib_weight = stack_inputs_from_projected_observation(
        projected_numer,
        projected_weight,
        stacking_method=stacking_method,
        value_semantics=value_semantics,
        pixel_area_sr=pixel_area_sr,
    )
    if coeffs is None:
        overlap_contrib = projected_overlap
        if target_window is None:
            numer.add_(contrib_numer)
            denom.add_(contrib_weight)
            overlap_count.add_(overlap_contrib)
            return
        y0, y1, x0, x1 = target_window
        if y1 <= y0 or x1 <= x0:
            return
        numer[y0:y1, x0:x1].add_(contrib_numer)
        denom[y0:y1, x0:x1].add_(contrib_weight)
        overlap_count[y0:y1, x0:x1].add_(overlap_contrib)
        return
    coeffs_t = torch.as_tensor(
        coeffs,
        dtype=contrib_numer.dtype,
        device=contrib_numer.device,
    )
    overlap_coeffs = (coeffs_t > 0.0).to(dtype=projected_overlap.dtype)
    numer_contrib = coeffs_t * contrib_numer.unsqueeze(0)
    denom_contrib = coeffs_t * contrib_weight.unsqueeze(0)
    overlap_contrib = overlap_coeffs * projected_overlap.unsqueeze(0)
    if target_window is None:
        numer.add_(numer_contrib)
        denom.add_(denom_contrib)
        overlap_count.add_(overlap_contrib)
        return
    y0, y1, x0, x1 = target_window
    if y1 <= y0 or x1 <= x0:
        return
    numer[:, y0:y1, x0:x1].add_(numer_contrib)
    denom[:, y0:y1, x0:x1].add_(denom_contrib)
    overlap_count[:, y0:y1, x0:x1].add_(overlap_contrib)


def finalize_projected_stack(
    numer: Tensor,
    denom: Tensor,
    *,
    stacking_method: str,
    value_semantics: str,
    pixel_area_sr: Tensor | None = None,
    median_images: Sequence[Tensor] | None = None,
    median_support: Sequence[Tensor] | None = None,
) -> tuple[Tensor, Tensor]:
    """Finalize accumulated stack tensors into map values and output weights."""
    if stacking_method == "median":
        if median_images is None or median_support is None:
            raise RuntimeError(
                "median stacking requires projected image/support buffers"
            )
        return masked_median_stack(median_images, median_support)
    if stacking_method == "weighted_mean" and value_semantics == "flux":
        if pixel_area_sr is None:
            raise RuntimeError("target pixel area is required for flux semantics")
        coadd = torch.where(
            denom > 0.0,
            pixel_area_sr * numer / denom,
            torch.zeros_like(numer),
        )
        out_weight = torch.where(
            denom > 0.0,
            denom / torch.clamp(torch.square(pixel_area_sr), min=1e-16),
            torch.zeros_like(denom),
        )
        return coadd, out_weight
    coadd = torch.where(
        denom > 0.0,
        numer / denom,
        torch.zeros_like(numer),
    )
    return coadd, denom


def prepare_image_observation_data(
    observation: object,
    image: Tensor,
    *,
    device: torch.device,
    resampling_kernel: str,
    load_weight_map_fn: LoadWeightMapFn,
    apply_calibration_fn: ApplyCalibrationFn,
) -> PreparedImageObservationData:
    """Normalize image-plane observation inputs into operator-ready tensors."""
    operator_bundle = image_observation_operator_bundle(
        observation,
        resampling_kernel=resampling_kernel,
    )
    if operator_bundle.mask_weight is not None or operator_bundle.noise is not None:
        source_weight = load_weight_map_fn(observation, image, device=device)
    else:
        source_weight = torch.isfinite(image).to(dtype=torch.float32, device=device)
    if operator_bundle.radiometry is not None or operator_bundle.background is not None:
        prepared_image, prepared_weight = apply_calibration_fn(
            observation,
            image,
            source_weight,
        )
    else:
        prepared_image = torch.nan_to_num(
            image.to(dtype=torch.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        prepared_weight = source_weight.to(dtype=torch.float32)
    return PreparedImageObservationData(
        image=prepared_image,
        source_weight=prepared_weight,
        operator_bundle=operator_bundle,
    )


def project_image_observation_geometry(
    image: Tensor,
    source_weight: Tensor,
    source_wcs,
    target,
    *,
    geometry,
    resampling_kernel: str,
    value_semantics: str,
    drizzle_pixfrac: float,
    pixel_area_sr_map_fn,
    representative_pixel_area_fn,
    chunk_pixels: int,
    resample_weighted_to_target_fn: ResampleToTargetFn,
    forward_project_weighted_to_target_fn: ForwardProjectFn,
) -> tuple[Tensor, Tensor]:
    """Dispatch geometry execution through operator-family types instead of local branches."""
    if isinstance(geometry, ExactDrizzleGeometryOp):
        if resampling_kernel == "drizzle_square":
            return exact_square_drizzle_project(
                image,
                source_weight,
                source_wcs,
                target,
                value_semantics=value_semantics,
                drizzle_pixfrac=drizzle_pixfrac,
                pixel_area_sr_map_fn=pixel_area_sr_map_fn,
                representative_pixel_area_fn=representative_pixel_area_fn,
                chunk_pixels=chunk_pixels,
            )
        return exact_point_drizzle_project(
            image,
            source_weight,
            source_wcs,
            target,
            value_semantics=value_semantics,
            pixel_area_sr_map_fn=pixel_area_sr_map_fn,
            representative_pixel_area_fn=representative_pixel_area_fn,
            chunk_pixels=chunk_pixels,
        )
    if isinstance(geometry, ConservativePolygonRemapOp):
        return conservative_polygon_project(
            image,
            source_weight,
            source_wcs,
            target,
            value_semantics=value_semantics,
            pixel_area_sr_map_fn=pixel_area_sr_map_fn,
            representative_pixel_area_fn=representative_pixel_area_fn,
            chunk_pixels=chunk_pixels,
        )
    if isinstance(geometry, ForwardSplatGeometryOp):
        return forward_project_weighted_to_target_fn(
            image,
            source_weight,
            source_wcs,
            target,
            resampling_kernel=resampling_kernel,
            value_semantics=value_semantics,
            drizzle_pixfrac=drizzle_pixfrac,
        )
    return resample_weighted_to_target_fn(
        image,
        source_weight,
        source_wcs,
        target,
        resampling_kernel=resampling_kernel,
        value_semantics=value_semantics,
    )


def source_bbox_on_target(
    source_wcs,
    source_shape: tuple[int, int],
    target_wcs,
    target_shape: tuple[int, int],
    *,
    margin: int,
    edge_samples: int,
) -> tuple[int, int, int, int] | None:
    """Estimate the clipped target-pixel box covered by a source image boundary."""
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


def is_plain_tan_header(header: Mapping[str, object]) -> bool:
    """Return True for simple 2D TAN celestial headers."""
    return str(header.get("CTYPE1", "")).endswith("TAN") and str(
        header.get("CTYPE2", "")
    ).endswith("TAN")


def wcs_has_distortion(wcs) -> bool:
    """Return True when a WCS carries distortion terms unsupported by fast policies."""
    return getattr(wcs, "sip", None) is not None or bool(getattr(wcs, "_is_tpv", False))
