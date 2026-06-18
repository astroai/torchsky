"""Quality metrics for continuous-field reconstructions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..exposure_pose import ExposurePose
from ..operators.bundle import (
    ObservationOperatorBundle,
    field_observation_operator_bundle,
)
from ..operators.psf import apply_psf_spec
from .rendering import render_field_observation
from .types import FieldFitResult, FieldObservation, ForwardContext


@dataclass(frozen=True)
class FieldResidualProduct:
    """Rendered prediction, residual map, support mask, and scalar metrics."""

    prediction: Tensor
    target: Tensor
    residual: Tensor
    weight: Tensor
    support: Tensor
    metrics: dict[str, float]
    band_name: str | None = None


@dataclass(frozen=True)
class FieldResidualTile:
    """Support-aware residual energy summary for one image tile."""

    y0: int
    y1: int
    x0: int
    x1: int
    residual_energy: float
    residual_rmse: float
    support_fraction: float
    effective_weight_sum: float
    band_name: str | None = None


def _as_prediction_tensors(
    prediction: Tensor,
    target: Tensor,
    weight: Tensor | None,
    mask: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    pred = torch.as_tensor(prediction)
    if pred.ndim != 2:
        raise ValueError("prediction and target must be 2D images")
    tgt = torch.as_tensor(target, device=pred.device, dtype=pred.dtype)
    if pred.shape != tgt.shape:
        raise ValueError("prediction and target must have the same shape")
    if weight is None:
        w = torch.ones_like(pred)
    else:
        w = torch.as_tensor(weight, device=pred.device, dtype=pred.dtype)
        if w.shape != pred.shape:
            raise ValueError("weight must match prediction shape")
    if mask is None:
        valid = torch.ones_like(pred, dtype=torch.bool)
    else:
        valid = torch.as_tensor(mask, device=pred.device) > 0
        if valid.shape != pred.shape:
            raise ValueError("mask must match prediction shape")
    valid = valid & torch.isfinite(pred) & torch.isfinite(tgt) & torch.isfinite(w)
    finite_weight = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = torch.where(valid, torch.clamp(finite_weight, min=0.0), torch.zeros_like(w))
    return pred, tgt, w, valid


def _weighted_mean(value: Tensor, weight: Tensor) -> Tensor:
    return torch.sum(weight * value) / torch.clamp(weight.sum(), min=1.0e-8)


def _weighted_std(value: Tensor, weight: Tensor) -> Tensor:
    mean = _weighted_mean(value, weight)
    centered = value - mean
    return torch.sqrt(_weighted_mean(centered * centered, weight))


def _sharpness_energy(image: Tensor, weight: Tensor, valid: Tensor) -> Tensor:
    total = torch.zeros((), device=image.device, dtype=image.dtype)
    denom = torch.zeros((), device=image.device, dtype=image.dtype)
    if image.shape[-1] > 1:
        grad_x = image[:, 1:] - image[:, :-1]
        valid_x = valid[:, 1:] & valid[:, :-1]
        weight_x = 0.5 * (weight[:, 1:] + weight[:, :-1])
        weight_x = torch.where(valid_x, weight_x, torch.zeros_like(weight_x))
        total = total + torch.sum(weight_x * grad_x * grad_x)
        denom = denom + weight_x.sum()
    if image.shape[-2] > 1:
        grad_y = image[1:, :] - image[:-1, :]
        valid_y = valid[1:, :] & valid[:-1, :]
        weight_y = 0.5 * (weight[1:, :] + weight[:-1, :])
        weight_y = torch.where(valid_y, weight_y, torch.zeros_like(weight_y))
        total = total + torch.sum(weight_y * grad_y * grad_y)
        denom = denom + weight_y.sum()
    return total / torch.clamp(denom, min=1.0e-8)


def _block_sum_2d(value: Tensor, *, scale: int) -> Tensor:
    if scale <= 1:
        return value
    height, width = int(value.shape[-2]), int(value.shape[-1])
    pad_h = (-height) % int(scale)
    pad_w = (-width) % int(scale)
    if pad_h or pad_w:
        value = F.pad(value, (0, pad_w, 0, pad_h))
    padded_h, padded_w = int(value.shape[-2]), int(value.shape[-1])
    return value.reshape(
        padded_h // int(scale),
        int(scale),
        padded_w // int(scale),
        int(scale),
    ).sum(dim=(1, 3))


def _normalized_residual_scales(scales: Sequence[int]) -> tuple[int, ...]:
    normalized: list[int] = []
    for scale in scales:
        scale_i = int(scale)
        if scale_i < 1:
            raise ValueError("residual_scales entries must be positive integers")
        if scale_i not in normalized:
            normalized.append(scale_i)
    return tuple(normalized)


def _multiscale_residual_rmse(
    residual: Tensor,
    weight: Tensor,
    *,
    scales: Sequence[int],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for scale in _normalized_residual_scales(scales):
        block_weight = _block_sum_2d(weight, scale=scale)
        block_weighted_residual = _block_sum_2d(weight * residual, scale=scale)
        block_residual = block_weighted_residual / torch.clamp(block_weight, min=1.0e-8)
        rmse = torch.sqrt(
            torch.sum(block_weight * block_residual * block_residual)
            / torch.clamp(block_weight.sum(), min=1.0e-8)
        )
        metrics[f"residual_rmse_scale_{scale}"] = float(rmse.detach().cpu())
    return metrics


def _aggregate_quality_metrics(
    per_observation: Sequence[dict[str, float]],
) -> dict[str, float]:
    if not per_observation:
        return {}
    keys = sorted({key for metrics in per_observation for key in metrics})
    weights = [
        max(float(metrics.get("effective_weight_sum", 0.0)), 0.0)
        for metrics in per_observation
    ]
    total_weight = sum(weights)
    aggregate: dict[str, float] = {}
    for key in keys:
        if key == "effective_weight_sum":
            aggregate[key] = total_weight
            continue
        available = [
            (metrics, weight)
            for metrics, weight in zip(per_observation, weights, strict=True)
            if key in metrics
        ]
        if not available:
            continue
        key_weight = sum(weight for _, weight in available)
        if key_weight > 0.0:
            aggregate[key] = (
                sum(
                    float(metrics[key]) * weight
                    for metrics, weight in available
                )
                / key_weight
            )
        else:
            aggregate[key] = sum(metrics[key] for metrics, _ in available) / float(
                len(available)
            )
    return aggregate


def _observation_quality_group_key(
    observation: FieldObservation | None,
    index: int,
) -> str:
    if observation is None:
        return "all"
    if observation.band_name is not None:
        return str(observation.band_name)
    if observation.wavelength is not None:
        return f"lambda={float(observation.wavelength):.8g}"
    if observation.spectral_min is not None or observation.spectral_max is not None:
        return (
            f"range={observation.spectral_min!s}:"
            f"{observation.spectral_max!s}"
        )
    return f"observation_{int(index)}"


def field_quality_scorecard(
    per_observation: Sequence[dict[str, float]],
    *,
    observations: Sequence[FieldObservation] | None = None,
) -> dict[str, Any]:
    """Summarize observation metrics by band plus exact worst-case extrema."""
    if observations is not None and len(observations) != len(per_observation):
        raise ValueError("observations must have the same length as per_observation")
    aggregate = _aggregate_quality_metrics(per_observation)
    if not per_observation:
        return {"aggregate": aggregate, "by_band": {}, "extrema": {}}

    grouped: dict[str, list[dict[str, float]]] = {}
    for idx, metrics in enumerate(per_observation):
        observation = observations[idx] if observations is not None else None
        grouped.setdefault(_observation_quality_group_key(observation, idx), []).append(
            metrics
        )
    by_band = {
        band: {
            "observation_count": len(items),
            **_aggregate_quality_metrics(items),
        }
        for band, items in grouped.items()
    }

    def _extreme(metric: str, *, largest: bool) -> dict[str, float | int]:
        available = [
            (idx, float(metrics[metric]))
            for idx, metrics in enumerate(per_observation)
            if metric in metrics
        ]
        if not available:
            return {"index": -1, "value": float("nan")}
        index, value = (
            max(available, key=lambda item: item[1])
            if largest
            else min(available, key=lambda item: item[1])
        )
        return {"index": int(index), "value": float(value)}

    return {
        "aggregate": aggregate,
        "by_band": by_band,
        "extrema": {
            "highest_weighted_rmse": _extreme("weighted_rmse", largest=True),
            "lowest_residual_snr": _extreme("residual_snr", largest=False),
            "lowest_support_fraction": _extreme("support_fraction", largest=False),
            "lowest_effective_weight_sum": _extreme(
                "effective_weight_sum",
                largest=False,
            ),
        },
    }


def field_quality_acceptance_report(
    per_observation: Sequence[dict[str, float]],
    *,
    observations: Sequence[FieldObservation] | None = None,
    max_weighted_rmse: float | None = None,
    min_residual_snr: float | None = None,
    min_support_fraction: float | None = None,
    min_effective_weight_sum: float | None = None,
    max_abs_psf_residual_alignment: float | None = None,
) -> dict[str, Any]:
    """Evaluate explicit per-observation quality gates with no default thresholds."""
    if observations is not None and len(observations) != len(per_observation):
        raise ValueError("observations must have the same length as per_observation")
    criteria: dict[str, tuple[str, float]] = {}
    if max_weighted_rmse is not None:
        criteria["weighted_rmse"] = ("max", float(max_weighted_rmse))
    if min_residual_snr is not None:
        criteria["residual_snr"] = ("min", float(min_residual_snr))
    if min_support_fraction is not None:
        criteria["support_fraction"] = ("min", float(min_support_fraction))
    if min_effective_weight_sum is not None:
        criteria["effective_weight_sum"] = ("min", float(min_effective_weight_sum))
    if max_abs_psf_residual_alignment is not None:
        criteria["psf_residual_alignment"] = (
            "max_abs",
            float(max_abs_psf_residual_alignment),
        )
    violations: list[dict[str, Any]] = []
    for index, metrics in enumerate(per_observation):
        band = (
            _observation_quality_group_key(observations[index], index)
            if observations is not None
            else None
        )
        for metric, (direction, threshold) in criteria.items():
            if metric not in metrics:
                violations.append(
                    {
                        "index": int(index),
                        "band": band,
                        "metric": metric,
                        "reason": "missing",
                        "threshold": threshold,
                    }
                )
                continue
            value = float(metrics[metric])
            compare_value = abs(value) if direction == "max_abs" else value
            failed = (
                compare_value > threshold
                if direction in {"max", "max_abs"}
                else compare_value < threshold
            )
            if failed:
                violations.append(
                    {
                        "index": int(index),
                        "band": band,
                        "metric": metric,
                        "value": value,
                        "threshold": threshold,
                        "direction": direction,
                    }
                )
    return {
        "passed": not violations,
        "criteria": {
            metric: {"direction": direction, "threshold": threshold}
            for metric, (direction, threshold) in criteria.items()
        },
        "violations": tuple(violations),
        "scorecard": field_quality_scorecard(
            per_observation,
            observations=observations,
        ),
    }


def field_image_quality_metrics(
    prediction: Tensor,
    target: Tensor,
    *,
    weight: Tensor | None = None,
    mask: Tensor | None = None,
    residual_scales: Sequence[int] = (1, 2, 4),
) -> dict[str, float]:
    """Compute weighted residual, SNR, sharpness, and multiscale residual metrics."""
    pred, tgt, w, valid = _as_prediction_tensors(prediction, target, weight, mask)
    residual = torch.where(valid, pred - tgt, torch.zeros_like(pred))
    target_valid = torch.where(valid, tgt, torch.zeros_like(tgt))
    pred_valid = torch.where(valid, pred, torch.zeros_like(pred))

    weighted_rmse = torch.sqrt(_weighted_mean(residual * residual, w))
    weighted_mae = _weighted_mean(torch.abs(residual), w)
    residual_bias = _weighted_mean(residual, w)
    residual_std = _weighted_std(residual, w)
    target_rms = torch.sqrt(_weighted_mean(target_valid * target_valid, w))
    prediction_rms = torch.sqrt(_weighted_mean(pred_valid * pred_valid, w))
    snr = target_rms / torch.clamp(weighted_rmse, min=1.0e-12)
    snr_db = 20.0 * torch.log10(torch.clamp(snr, min=1.0e-12))

    pred_sharpness = _sharpness_energy(pred_valid, w, valid)
    target_sharpness = _sharpness_energy(target_valid, w, valid)
    sharpness_ratio = pred_sharpness / torch.clamp(target_sharpness, min=1.0e-12)
    support_fraction = (valid & (w > 0.0)).to(dtype=pred.dtype).mean()
    effective_weight_sum = w.sum()

    metrics = {
        "weighted_rmse": float(weighted_rmse.detach().cpu()),
        "weighted_mae": float(weighted_mae.detach().cpu()),
        "residual_bias": float(residual_bias.detach().cpu()),
        "residual_std": float(residual_std.detach().cpu()),
        "target_rms": float(target_rms.detach().cpu()),
        "prediction_rms": float(prediction_rms.detach().cpu()),
        "residual_snr": float(snr.detach().cpu()),
        "residual_snr_db": float(snr_db.detach().cpu()),
        "prediction_sharpness": float(pred_sharpness.detach().cpu()),
        "target_sharpness": float(target_sharpness.detach().cpu()),
        "sharpness_ratio": float(sharpness_ratio.detach().cpu()),
        "support_fraction": float(support_fraction.detach().cpu()),
        "effective_weight_sum": float(effective_weight_sum.detach().cpu()),
    }
    metrics.update(_multiscale_residual_rmse(residual, w, scales=residual_scales))
    return metrics


def _psf_residual_alignment(
    residual: Tensor,
    weight: Tensor,
    support: Tensor,
    psf_spec: object | None,
) -> float:
    psf_residual = apply_psf_spec(residual, psf_spec)
    if psf_residual.shape != residual.shape:
        raise ValueError("PSF residual alignment requires shape-preserving PSF output")
    valid_weight = torch.where(
        support,
        torch.clamp(weight, min=0.0),
        torch.zeros_like(weight),
    )
    weight_sum = valid_weight.sum()
    if float(weight_sum.detach().cpu()) <= 0.0:
        return 0.0
    residual_mean = torch.sum(valid_weight * residual) / torch.clamp(
        weight_sum,
        min=1.0e-12,
    )
    psf_mean = torch.sum(valid_weight * psf_residual) / torch.clamp(
        weight_sum,
        min=1.0e-12,
    )
    residual_centered = residual - residual_mean
    psf_centered = psf_residual - psf_mean
    numerator = torch.sum(valid_weight * residual_centered * psf_centered)
    denominator = torch.sqrt(
        torch.sum(valid_weight * residual_centered * residual_centered)
        * torch.sum(valid_weight * psf_centered * psf_centered)
        + 1.0e-12
    )
    return float((numerator / denominator).detach().cpu())


def field_image_residual_product(
    prediction: Tensor,
    target: Tensor,
    *,
    weight: Tensor | None = None,
    mask: Tensor | None = None,
    psf_spec: object | None = None,
    residual_scales: Sequence[int] = (1, 2, 4),
    band_name: str | None = None,
) -> FieldResidualProduct:
    """Build a residual product for one predicted image and target image."""
    pred, tgt, w, valid = _as_prediction_tensors(prediction, target, weight, mask)
    residual = torch.where(valid, pred - tgt, torch.zeros_like(pred))
    metrics = field_image_quality_metrics(
        pred,
        tgt,
        weight=w,
        mask=valid,
        residual_scales=residual_scales,
    )
    support = valid & (w > 0.0)
    if psf_spec is not None and getattr(psf_spec, "mode", "none") != "none":
        metrics["psf_residual_alignment"] = _psf_residual_alignment(
            residual,
            w,
            support,
            psf_spec,
        )
    return FieldResidualProduct(
        prediction=torch.where(valid, pred, torch.zeros_like(pred)),
        target=torch.where(valid, tgt, torch.zeros_like(tgt)),
        residual=residual,
        weight=w,
        support=support,
        metrics=metrics,
        band_name=band_name,
    )


def select_field_residual_tiles(
    product: FieldResidualProduct,
    *,
    tile_hw: tuple[int, int],
    max_tiles: int | None = None,
    min_support_fraction: float = 0.0,
) -> tuple[FieldResidualTile, ...]:
    """Rank fixed image tiles by exact weighted residual energy."""
    tile_h, tile_w = int(tile_hw[0]), int(tile_hw[1])
    if tile_h <= 0 or tile_w <= 0:
        raise ValueError("tile_hw entries must be positive")
    if max_tiles is not None and int(max_tiles) < 1:
        raise ValueError("max_tiles must be positive when provided")
    min_support = float(min_support_fraction)
    if min_support < 0.0 or min_support > 1.0:
        raise ValueError("min_support_fraction must satisfy 0 <= value <= 1")

    residual = torch.as_tensor(product.residual)
    weight = torch.as_tensor(product.weight, device=residual.device, dtype=residual.dtype)
    support = torch.as_tensor(product.support, device=residual.device).to(dtype=torch.bool)
    if residual.ndim != 2:
        raise ValueError("residual product must contain a 2D residual map")
    if weight.shape != residual.shape or support.shape != residual.shape:
        raise ValueError("residual, weight, and support shapes must match")

    height, width = int(residual.shape[-2]), int(residual.shape[-1])
    tiles: list[FieldResidualTile] = []
    for y0 in range(0, height, tile_h):
        y1 = min(y0 + tile_h, height)
        for x0 in range(0, width, tile_w):
            x1 = min(x0 + tile_w, width)
            residual_tile = residual[y0:y1, x0:x1]
            weight_tile = weight[y0:y1, x0:x1]
            support_tile = support[y0:y1, x0:x1]
            support_fraction = float(
                support_tile.to(dtype=torch.float32).mean().detach().cpu()
            )
            if support_fraction < min_support:
                continue
            valid_weight = torch.where(
                support_tile,
                torch.clamp(weight_tile, min=0.0),
                torch.zeros_like(weight_tile),
            )
            energy = torch.sum(valid_weight * residual_tile * residual_tile)
            weight_sum = valid_weight.sum()
            rmse = torch.sqrt(energy / torch.clamp(weight_sum, min=1.0e-8))
            tiles.append(
                FieldResidualTile(
                    y0=int(y0),
                    y1=int(y1),
                    x0=int(x0),
                    x1=int(x1),
                    residual_energy=float(energy.detach().cpu()),
                    residual_rmse=float(rmse.detach().cpu()),
                    support_fraction=support_fraction,
                    effective_weight_sum=float(weight_sum.detach().cpu()),
                    band_name=product.band_name,
                )
            )
    tiles.sort(key=lambda tile: tile.residual_energy, reverse=True)
    if max_tiles is not None:
        tiles = tiles[: int(max_tiles)]
    return tuple(tiles)


def render_field_residual(
    model: nn.Module,
    observation: FieldObservation,
    *,
    num_spectral_samples: int = 9,
    forward_context: ForwardContext | None = None,
    exposure_pose: ExposurePose | None = None,
    operator_bundle: ObservationOperatorBundle | None = None,
    masked: bool = True,
    **render_kwargs: Any,
) -> Tensor:
    """Render ``model`` into ``observation`` and return prediction minus image."""
    prediction = render_field_observation(
        model,
        observation,
        num_spectral_samples=num_spectral_samples,
        forward_context=forward_context,
        exposure_pose=exposure_pose,
        operator_bundle=operator_bundle,
        **render_kwargs,
    )
    image = observation.image.to(device=prediction.device, dtype=prediction.dtype)
    residual = prediction - image
    if masked and observation.mask is not None:
        valid = observation.mask.to(device=prediction.device) > 0
        residual = torch.where(valid, residual, torch.zeros_like(residual))
    return residual


def render_field_residual_product(
    model: nn.Module,
    observation: FieldObservation,
    *,
    num_spectral_samples: int = 9,
    forward_context: ForwardContext | None = None,
    exposure_pose: ExposurePose | None = None,
    operator_bundle: ObservationOperatorBundle | None = None,
    residual_scales: Sequence[int] = (1, 2, 4),
    **render_kwargs: Any,
) -> FieldResidualProduct:
    """Render a field observation and return a structured residual product."""
    resolved_operator_bundle = (
        operator_bundle
        if operator_bundle is not None
        else field_observation_operator_bundle(observation)
    )
    prediction = render_field_observation(
        model,
        observation,
        num_spectral_samples=num_spectral_samples,
        forward_context=forward_context,
        exposure_pose=exposure_pose,
        operator_bundle=resolved_operator_bundle,
        **render_kwargs,
    )
    return field_image_residual_product(
        prediction,
        observation.image,
        weight=observation.weight,
        mask=observation.mask,
        psf_spec=observation.psf_spec
        if resolved_operator_bundle.psf is not None
        else None,
        residual_scales=residual_scales,
        band_name=observation.band_name,
    )


def render_field_quality(
    model: nn.Module,
    observations: Sequence[FieldObservation],
    *,
    num_spectral_samples: int = 9,
    observation_forward_contexts: Sequence[ForwardContext | None] | None = None,
    observation_exposure_poses: Sequence[ExposurePose | None] | None = None,
    observation_operator_bundles: Sequence[ObservationOperatorBundle | None]
    | None = None,
    residual_scales: Sequence[int] = (1, 2, 4),
    include_scorecard: bool = False,
    include_residual_products: bool = False,
    **render_kwargs: Any,
) -> dict[str, Any]:
    """Render observations and summarize residual/SNR/sharpness metrics."""
    contexts = (
        tuple(observation_forward_contexts)
        if observation_forward_contexts is not None
        else tuple(None for _ in observations)
    )
    poses = (
        tuple(observation_exposure_poses)
        if observation_exposure_poses is not None
        else tuple(None for _ in observations)
    )
    operator_bundles = (
        tuple(observation_operator_bundles)
        if observation_operator_bundles is not None
        else tuple(None for _ in observations)
    )
    if len(contexts) != len(observations):
        raise ValueError(
            "observation_forward_contexts must have the same length as observations"
        )
    if len(poses) != len(observations):
        raise ValueError(
            "observation_exposure_poses must have the same length as observations"
        )
    if len(operator_bundles) != len(observations):
        raise ValueError(
            "observation_operator_bundles must have the same length as observations"
        )

    per_observation: list[dict[str, float]] = []
    residual_products: list[FieldResidualProduct] = []
    for observation, context, pose, operator_bundle in zip(
        observations,
        contexts,
        poses,
        operator_bundles,
        strict=True,
    ):
        resolved_operator_bundle = (
            operator_bundle
            if operator_bundle is not None
            else field_observation_operator_bundle(observation)
        )
        prediction = render_field_observation(
            model,
            observation,
            num_spectral_samples=num_spectral_samples,
            forward_context=context,
            exposure_pose=pose,
            operator_bundle=resolved_operator_bundle,
            **render_kwargs,
        )
        product = field_image_residual_product(
            prediction,
            observation.image,
            weight=observation.weight,
            mask=observation.mask,
            psf_spec=observation.psf_spec
            if resolved_operator_bundle.psf is not None
            else None,
            residual_scales=residual_scales,
            band_name=observation.band_name,
        )
        per_observation.append(product.metrics)
        if include_residual_products:
            residual_products.append(product)

    if not per_observation:
        result: dict[str, Any] = {"aggregate": {}, "per_observation": []}
        if include_scorecard:
            result["scorecard"] = field_quality_scorecard(())
        if include_residual_products:
            result["residual_products"] = []
        return result
    aggregate = _aggregate_quality_metrics(per_observation)
    result = {"aggregate": aggregate, "per_observation": per_observation}
    if include_scorecard:
        result["scorecard"] = field_quality_scorecard(
            per_observation,
            observations=observations,
        )
    if include_residual_products:
        result["residual_products"] = residual_products
    return result


def render_field_fit_quality(
    fit_result: FieldFitResult,
    observations: Sequence[FieldObservation],
    *,
    num_spectral_samples: int = 9,
    observation_forward_contexts: Sequence[ForwardContext | None] | None = None,
    observation_exposure_poses: Sequence[ExposurePose | None] | None = None,
    observation_operator_bundles: Sequence[ObservationOperatorBundle | None]
    | None = None,
    residual_scales: Sequence[int] = (1, 2, 4),
    include_scorecard: bool = False,
    include_residual_products: bool = False,
    **render_kwargs: Any,
) -> dict[str, Any]:
    """Render quality summaries using model and observation-law metadata from a fit."""
    poses = (
        fit_result.exposure_poses
        if observation_exposure_poses is None
        else observation_exposure_poses
    )
    bundles = (
        fit_result.observation_operator_bundles
        if observation_operator_bundles is None
        else observation_operator_bundles
    )
    return render_field_quality(
        fit_result.model,
        observations,
        num_spectral_samples=num_spectral_samples,
        observation_forward_contexts=observation_forward_contexts,
        observation_exposure_poses=poses,
        observation_operator_bundles=bundles,
        residual_scales=residual_scales,
        include_scorecard=include_scorecard,
        include_residual_products=include_residual_products,
        **render_kwargs,
    )


__all__ = [
    "FieldResidualProduct",
    "FieldResidualTile",
    "field_image_residual_product",
    "field_image_quality_metrics",
    "field_quality_acceptance_report",
    "field_quality_scorecard",
    "render_field_fit_quality",
    "render_field_quality",
    "render_field_residual",
    "render_field_residual_product",
    "select_field_residual_tiles",
]
