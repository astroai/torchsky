"""Product-oriented mapmaking and evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from .. import WCS
from ..sphere.moc import MOC
from ..spectral import SpectralAxis
from .field import FieldObservation, render_field_observation
from .image_plane import (
    BandMetadataTable,
    ImageObservation,
    MapResamplingKernel,
    MapResult,
    MapValueSemantics,
    PlannedImagePlaneCubeTarget,
    PlannedImagePlaneTarget,
    SpectralCubeMapResult,
    _pixel_area_sr_map,
    _validated_map_resampling_kernel,
    _validated_map_value_semantics,
    coadd_image_plane,
    coadd_image_plane_cube,
    plan_image_plane_cube_target,
    plan_image_plane_target,
)


@dataclass(frozen=True)
class MapBandpassSpec:
    label: str | None = None
    spectral_min: float | None = None
    spectral_max: float | None = None
    transmission_center: float | None = None
    transmission_sigma: float | None = None


@dataclass(frozen=True)
class MapProductSpec:
    target_header: dict[str, Any]
    bands: tuple[MapBandpassSpec, ...]
    value_semantics: MapValueSemantics = "surface_brightness"
    footprint_order: int | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.target_header["NAXIS2"]), int(self.target_header["NAXIS1"])

    @property
    def n_bands(self) -> int:
        return len(self.bands)


@dataclass
class MapProductResult:
    product: MapProductSpec
    data: Tensor
    weight: Tensor
    header: dict[str, Any]
    overlap_count: Tensor | None = None
    coverage_mocs: tuple[MOC | None, ...] | None = None

    @property
    def n_bands(self) -> int:
        return int(self.data.shape[0])


def _coerce_band(value: MapBandpassSpec) -> MapBandpassSpec:
    if value.spectral_min is not None and value.spectral_max is None:
        raise ValueError("MapBandpassSpec requires spectral_max when spectral_min is set")
    if value.spectral_max is not None and value.spectral_min is None:
        raise ValueError("MapBandpassSpec requires spectral_min when spectral_max is set")
    if value.spectral_min is not None and value.spectral_max is not None:
        lower = float(value.spectral_min)
        upper = float(value.spectral_max)
        if not math.isfinite(lower) or not math.isfinite(upper) or upper < lower:
            raise ValueError("MapBandpassSpec spectral bounds must be finite and ordered")
    if value.transmission_center is not None and not math.isfinite(
        float(value.transmission_center)
    ):
        raise ValueError("MapBandpassSpec transmission_center must be finite")
    if value.transmission_sigma is not None:
        sigma = float(value.transmission_sigma)
        if not math.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("MapBandpassSpec transmission_sigma must be finite and positive")
    return value


def _band_center(band: MapBandpassSpec) -> float:
    if band.transmission_center is not None:
        return float(band.transmission_center)
    if band.spectral_min is not None and band.spectral_max is not None:
        return 0.5 * (float(band.spectral_min) + float(band.spectral_max))
    raise ValueError(
        "MapBandpassSpec requires transmission_center or spectral bounds for product rendering"
    )


def _bands_from_cube_target(target: PlannedImagePlaneCubeTarget) -> tuple[MapBandpassSpec, ...]:
    bands: list[MapBandpassSpec] = []
    metadata = target.band_metadata
    for idx in range(int(metadata.center.numel())):
        lower = float(metadata.lower[idx].item())
        upper = float(metadata.upper[idx].item())
        center = float(metadata.center[idx].item())
        is_interval = not math.isclose(lower, upper, rel_tol=0.0, abs_tol=1.0e-12)
        bands.append(
            MapBandpassSpec(
                label=metadata.band_names[idx],
                spectral_min=lower if is_interval else None,
                spectral_max=upper if is_interval else None,
                transmission_center=center,
                transmission_sigma=None,
            )
        )
    return tuple(bands)


def _observations_have_spectral_support(observations: Sequence[ImageObservation]) -> bool:
    for observation in observations:
        if (
            observation.wavelength is not None
            or observation.spectral_min is not None
            or observation.spectral_max is not None
            or observation.transmission_center is not None
            or observation.transmission_sigma is not None
        ):
            return True
    return False


def plan_map_product(
    observations: Sequence[ImageObservation],
    *,
    reference_index: int = 0,
    header_overrides: dict[str, Any] | None = None,
    bands: Sequence[MapBandpassSpec] | None = None,
    spectral_axis: SpectralAxis | None = None,
    spectral_bin_edges: Tensor | None = None,
    spectral_unit: str = "Angstrom",
    spectral_type: str = "WAVE",
    value_semantics: MapValueSemantics = "surface_brightness",
    footprint_order: int | None = None,
) -> MapProductSpec:
    if footprint_order is not None and int(footprint_order) < 0:
        raise ValueError("footprint_order must be non-negative")
    value_semantics = _validated_map_value_semantics(value_semantics)
    if bands is not None and (spectral_axis is not None or spectral_bin_edges is not None):
        raise ValueError("bands is mutually exclusive with spectral_axis and spectral_bin_edges")
    if bands is not None:
        spatial = plan_image_plane_target(
            observations,
            reference_index=reference_index,
            header_overrides=header_overrides,
        )
        validated_bands = tuple(_coerce_band(band) for band in bands)
        if not validated_bands:
            raise ValueError("MapProductSpec requires at least one band")
        return MapProductSpec(
            target_header=spatial.header,
            bands=validated_bands,
            value_semantics=value_semantics,
            footprint_order=footprint_order,
        )
    if spectral_axis is not None or spectral_bin_edges is not None or _observations_have_spectral_support(observations):
        target = plan_image_plane_cube_target(
            observations,
            reference_index=reference_index,
            header_overrides=header_overrides,
            spectral_axis=spectral_axis,
            spectral_bin_edges=spectral_bin_edges,
            spectral_unit=spectral_unit,
            spectral_type=spectral_type,
        )
        return MapProductSpec(
            target_header=target.header,
            bands=_bands_from_cube_target(target),
            value_semantics=value_semantics,
            footprint_order=footprint_order,
        )
    spatial = plan_image_plane_target(
        observations,
        reference_index=reference_index,
        header_overrides=header_overrides,
    )
    reference = observations[reference_index]
    return MapProductSpec(
        target_header=spatial.header,
        bands=(
            MapBandpassSpec(
                label=reference.band_name,
                spectral_min=reference.spectral_min,
                spectral_max=reference.spectral_max,
                transmission_center=reference.transmission_center
                if reference.transmission_center is not None
                else reference.wavelength,
                transmission_sigma=reference.transmission_sigma,
            ),
        ),
        value_semantics=value_semantics,
        footprint_order=footprint_order,
    )


def _band_metadata_from_product(
    product: MapProductSpec,
    spectral_axis: SpectralAxis,
    spectral_bin_edges: Tensor | None,
) -> BandMetadataTable:
    n_band = int(spectral_axis.values.numel())
    centers = spectral_axis.values.to(dtype=torch.float64)
    if spectral_bin_edges is None:
        lower = centers.clone()
        upper = centers.clone()
    else:
        lower = spectral_bin_edges[:-1].clone()
        upper = spectral_bin_edges[1:].clone()
    return BandMetadataTable(
        band_index=torch.arange(n_band, dtype=torch.int64),
        center=centers,
        lower=lower,
        upper=upper,
        observation_count=torch.zeros(n_band, dtype=torch.int64),
        spectral_weight_sum=torch.zeros(n_band, dtype=torch.float64),
        band_names=tuple(band.label for band in product.bands),
    )


def _product_to_cube_target(product: MapProductSpec) -> PlannedImagePlaneCubeTarget:
    centers = torch.tensor(
        [_band_center(band) for band in product.bands],
        dtype=torch.float64,
    )
    if centers.numel() > 1 and not bool(torch.all(centers[1:] > centers[:-1])):
        raise ValueError("MapProductSpec bands must be strictly increasing in spectral center")
    interval_bands = [
        band
        for band in product.bands
        if band.spectral_min is not None and band.spectral_max is not None
    ]
    spectral_bin_edges = None
    if interval_bands:
        if len(interval_bands) != len(product.bands):
            raise ValueError(
                "MapProductSpec direct stacking requires all bands to define intervals or none"
            )
        lowers = [float(band.spectral_min) for band in product.bands if band.spectral_min is not None]
        uppers = [float(band.spectral_max) for band in product.bands if band.spectral_max is not None]
        contiguous = True
        for idx in range(len(lowers) - 1):
            contiguous = contiguous and math.isclose(
                uppers[idx],
                lowers[idx + 1],
                rel_tol=0.0,
                abs_tol=1.0e-8,
            )
        if len(product.bands) == 1 or contiguous:
            spectral_bin_edges = torch.tensor(
                [lowers[0], *uppers],
                dtype=torch.float64,
            )
        else:
            raise ValueError(
                "MapProductSpec direct stacking currently supports only point bands or "
                "ordered contiguous interval bands"
            )
    spectral_axis = SpectralAxis(values=centers, unit="Angstrom", type="WAVE")
    spatial = PlannedImagePlaneTarget(
        header=dict(product.target_header),
        shape=(int(product.target_header["NAXIS2"]), int(product.target_header["NAXIS1"])),
    )
    return PlannedImagePlaneCubeTarget(
        spatial_target=spatial,
        spectral_axis=spectral_axis,
        spectral_bin_edges=spectral_bin_edges,
        band_metadata=_band_metadata_from_product(
            product,
            spectral_axis=spectral_axis,
            spectral_bin_edges=spectral_bin_edges,
        ),
    )


def _wrap_map_result(result: MapResult, product: MapProductSpec) -> MapProductResult:
    coverage_mocs = None
    if result.coverage_moc is not None:
        coverage_mocs = (result.coverage_moc,)
    return MapProductResult(
        product=product,
        data=result.data.unsqueeze(0),
        weight=result.weight.unsqueeze(0),
        header=result.header,
        overlap_count=(
            result.overlap_count.unsqueeze(0) if result.overlap_count is not None else None
        ),
        coverage_mocs=coverage_mocs,
    )


def _wrap_cube_result(
    result: SpectralCubeMapResult,
    product: MapProductSpec,
) -> MapProductResult:
    return MapProductResult(
        product=product,
        data=result.data,
        weight=result.weight,
        header=result.header,
        overlap_count=result.overlap_count,
        coverage_mocs=result.coverage_mocs,
    )


def stack_map_product(
    observations: Sequence[ImageObservation],
    product: MapProductSpec,
    *,
    resampling_kernel: MapResamplingKernel = "bilinear",
    method: str = "coadd",
) -> MapProductResult:
    if method != "coadd":
        raise ValueError("stack_map_product currently supports only method='coadd'")
    resampling_kernel = _validated_map_resampling_kernel(resampling_kernel)
    if len(product.bands) == 1:
        band = product.bands[0]
        if band.spectral_min is None and band.spectral_max is None:
            result = coadd_image_plane(
                observations,
                target_header=product.target_header,
                footprint_order=product.footprint_order,
                resampling_kernel=resampling_kernel,
                value_semantics=product.value_semantics,
            )
            return _wrap_map_result(result, product)
    cube_target = _product_to_cube_target(product)
    result = coadd_image_plane_cube(
        observations,
        target=cube_target,
        footprint_order=product.footprint_order,
        resampling_kernel=resampling_kernel,
        value_semantics=product.value_semantics,
    )
    return _wrap_cube_result(result, product)


def _module_device(model: nn.Module) -> torch.device:
    parameter = next(model.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(model.buffers(), None)
    if buffer is not None:
        return buffer.device
    return torch.device("cpu")


def render_field_product(
    model: nn.Module,
    product: MapProductSpec,
    *,
    num_spectral_samples: int = 9,
) -> MapProductResult:
    device = _module_device(model)
    shape = (
        int(product.target_header["NAXIS2"]),
        int(product.target_header["NAXIS1"]),
    )
    planes: list[Tensor] = []
    for band in product.bands:
        center = (
            float(band.transmission_center)
            if band.transmission_center is not None
            else _band_center(band)
        )
        lower = (
            float(band.spectral_min)
            if band.spectral_min is not None
            else center
        )
        upper = (
            float(band.spectral_max)
            if band.spectral_max is not None
            else center
        )
        rendered = render_field_observation(
            model,
            FieldObservation(
                image=torch.zeros(shape, dtype=torch.float32, device=device),
                header=dict(product.target_header),
                wavelength=center if lower == upper else center,
                spectral_min=lower if lower != upper else None,
                spectral_max=upper if lower != upper else None,
                transmission_center=band.transmission_center,
                transmission_sigma=band.transmission_sigma,
                band_name=band.label,
            ),
            num_spectral_samples=num_spectral_samples,
        )
        planes.append(rendered.to(dtype=torch.float32))
    data = torch.stack(planes, dim=0)
    return MapProductResult(
        product=product,
        data=data,
        weight=torch.ones_like(data),
        header=dict(product.target_header),
        overlap_count=torch.ones_like(data, dtype=torch.int32),
        coverage_mocs=None,
    )


def _as_cube(tensor: Tensor) -> Tensor:
    if tensor.ndim == 2:
        return tensor.unsqueeze(0)
    if tensor.ndim == 3:
        return tensor
    raise ValueError("Map product evaluation expects 2D or 3D tensors")


def evaluate_map_product(
    prediction: MapProductResult | Tensor,
    truth: MapProductResult | Tensor,
    *,
    weight: Tensor | None = None,
    header: dict[str, Any] | None = None,
    value_semantics: MapValueSemantics | None = None,
) -> dict[str, float | None]:
    if isinstance(prediction, MapProductResult):
        pred_data = prediction.data
        pred_weight = prediction.weight
        header = dict(prediction.header)
        value_semantics = prediction.product.value_semantics
    else:
        pred_data = _as_cube(prediction)
        pred_weight = _as_cube(weight) if weight is not None else torch.ones_like(pred_data)
    if isinstance(truth, MapProductResult):
        truth_data = truth.data
        if header is None:
            header = dict(truth.header)
        if value_semantics is None:
            value_semantics = truth.product.value_semantics
    else:
        truth_data = _as_cube(truth)
    if header is None:
        raise ValueError("evaluate_map_product requires a header")
    if value_semantics is None:
        raise ValueError("evaluate_map_product requires value_semantics")
    pred_data = _as_cube(pred_data).to(dtype=torch.float32)
    truth_data = _as_cube(truth_data).to(dtype=torch.float32, device=pred_data.device)
    pred_weight = _as_cube(pred_weight).to(dtype=torch.float32, device=pred_data.device)
    if pred_data.shape != truth_data.shape or pred_data.shape != pred_weight.shape:
        raise ValueError("Prediction, truth, and weight tensors must share the same shape")
    valid = pred_weight > 0.0
    if not bool(valid.any()):
        zeros: dict[str, float | None] = {
            "coverage_fraction": 0.0,
            "zero_support_fraction": 1.0,
            "mean_weight": float(pred_weight.mean().item()),
            "reconstruction_rmse": 0.0,
            "reconstruction_mae": 0.0,
            "reconstruction_p95_abs_err": 0.0,
            "reconstruction_max_abs_err": 0.0,
            "signal_to_residual_ratio": 0.0,
            "signal_to_residual_db": float("-inf"),
            "band_flux_relative_error_mean": 0.0,
            "band_flux_relative_error_max": 0.0,
            "total_flux_truth": 0.0,
            "total_flux_output": 0.0,
            "total_flux_absolute_error": 0.0,
            "total_flux_relative_error": 0.0,
            "blue_red_flux_ratio_truth": None,
            "blue_red_flux_ratio_output": None,
            "blue_red_flux_ratio_absolute_error": None,
            "mid_blue_flux_ratio_truth": None,
            "mid_blue_flux_ratio_output": None,
            "mid_blue_flux_ratio_absolute_error": None,
        }
        return zeros
    residual = pred_data - truth_data
    abs_err = torch.abs(residual)
    band_rmse: list[Tensor] = []
    for band_idx in range(int(pred_data.shape[0])):
        band_valid = valid[band_idx]
        if bool(band_valid.any()):
            band_residual = residual[band_idx][band_valid]
            band_rmse.append(torch.sqrt(torch.mean(torch.square(band_residual))))
        else:
            band_rmse.append(torch.zeros((), dtype=torch.float32, device=pred_data.device))
    band_rmse_tensor = torch.stack(band_rmse)
    valid_flat = valid.reshape(-1)
    signal_flat = truth_data.reshape(-1)[valid_flat]
    residual_flat = residual.reshape(-1)[valid_flat]
    abs_flat = abs_err.reshape(-1)[valid_flat]
    signal_rms = float(torch.sqrt(torch.mean(torch.square(signal_flat))).item())
    residual_rms = float(torch.sqrt(torch.mean(torch.square(residual_flat))).item())
    snr_linear = signal_rms / max(residual_rms, 1.0e-12)
    if value_semantics == "surface_brightness":
        area_sr = _pixel_area_sr_map(
            WCS(header),
            (int(pred_data.shape[-2]), int(pred_data.shape[-1])),
            device=pred_data.device,
        ).unsqueeze(0)
        truth_flux = truth_data * area_sr
        pred_flux = pred_data * area_sr
    else:
        truth_flux = truth_data
        pred_flux = pred_data
    valid_flux = valid.to(dtype=torch.float32)
    band_flux_truth = torch.sum(truth_flux * valid_flux, dim=(1, 2))
    band_flux_output = torch.sum(pred_flux * valid_flux, dim=(1, 2))
    band_flux_rel_err = torch.abs(band_flux_output - band_flux_truth) / torch.clamp(
        torch.abs(band_flux_truth),
        min=1.0e-8,
    )
    metrics: dict[str, float | None] = {
        "coverage_fraction": float(valid.to(dtype=torch.float32).mean().item()),
        "zero_support_fraction": float((~valid).to(dtype=torch.float32).mean().item()),
        "mean_weight": float(pred_weight.mean().item()),
        "reconstruction_rmse": float(
            torch.sqrt(torch.mean(torch.square(residual_flat))).item()
        ),
        "reconstruction_mae": float(torch.mean(abs_flat).item()),
        "reconstruction_p95_abs_err": float(torch.quantile(abs_flat, 0.95).item()),
        "reconstruction_max_abs_err": float(torch.max(abs_flat).item()),
        "signal_to_residual_ratio": snr_linear,
        "signal_to_residual_db": 20.0 * math.log10(max(snr_linear, 1.0e-12)),
        "band_flux_relative_error_mean": float(torch.mean(band_flux_rel_err).item()),
        "band_flux_relative_error_max": float(torch.max(band_flux_rel_err).item()),
        "band_reconstruction_rmse_mean": float(torch.mean(band_rmse_tensor).item()),
        "band_reconstruction_rmse_max": float(torch.max(band_rmse_tensor).item()),
        "total_flux_truth": float(band_flux_truth.sum().item()),
        "total_flux_output": float(band_flux_output.sum().item()),
        "total_flux_absolute_error": float(
            torch.abs(band_flux_output.sum() - band_flux_truth.sum()).item()
        ),
        "total_flux_relative_error": float(
            (
                torch.abs(band_flux_output.sum() - band_flux_truth.sum())
                / torch.clamp(torch.abs(band_flux_truth.sum()), min=1.0e-8)
            ).item()
        ),
        "blue_red_flux_ratio_truth": None,
        "blue_red_flux_ratio_output": None,
        "blue_red_flux_ratio_absolute_error": None,
        "mid_blue_flux_ratio_truth": None,
        "mid_blue_flux_ratio_output": None,
        "mid_blue_flux_ratio_absolute_error": None,
    }
    if int(pred_data.shape[0]) >= 2:
        blue_red_truth = float(
            (band_flux_truth[0] / torch.clamp(band_flux_truth[-1], min=1.0e-8)).item()
        )
        blue_red_output = float(
            (band_flux_output[0] / torch.clamp(band_flux_output[-1], min=1.0e-8)).item()
        )
        metrics["blue_red_flux_ratio_truth"] = blue_red_truth
        metrics["blue_red_flux_ratio_output"] = blue_red_output
        metrics["blue_red_flux_ratio_absolute_error"] = abs(
            blue_red_output - blue_red_truth
        )
    if int(pred_data.shape[0]) >= 3:
        mid = int(pred_data.shape[0] // 2)
        mid_blue_truth = float(
            (band_flux_truth[mid] / torch.clamp(band_flux_truth[0], min=1.0e-8)).item()
        )
        mid_blue_output = float(
            (band_flux_output[mid] / torch.clamp(band_flux_output[0], min=1.0e-8)).item()
        )
        metrics["mid_blue_flux_ratio_truth"] = mid_blue_truth
        metrics["mid_blue_flux_ratio_output"] = mid_blue_output
        metrics["mid_blue_flux_ratio_absolute_error"] = abs(
            mid_blue_output - mid_blue_truth
        )
    return metrics


def _row_duration_ms(row: Mapping[str, Any]) -> float | None:
    value = row.get("fit_time_ms")
    if value is None:
        value = row.get("time_ms")
    if value is None:
        return None
    return float(value)


def _row_label(row: Mapping[str, Any]) -> str:
    if row.get("type") == "field_recommendation":
        return str(row.get("recommended_model"))
    if row.get("library") == "torchsky_recommendation":
        return str(row.get("recommended_variant"))
    model = row.get("model_kind")
    if model is not None:
        return str(model)
    method = row.get("method", row.get("library", "unknown"))
    variant = row.get("variant")
    if variant is None or variant == "" or variant == "default":
        return str(method)
    return f"{method}:{variant}"


def build_map_benchmark_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    cases = sorted({str(row["case"]) for row in rows if "case" in row})
    lines = [
        "| case | best accuracy | best SNR | best flux conservation | fastest | recommended stack | recommended field |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for case in cases:
        case_rows = [
            row
            for row in rows
            if str(row.get("case")) == case and row.get("status", "OK") in {"OK", "INFO"}
        ]
        metric_rows = [
            row
            for row in case_rows
            if row.get("reconstruction_rmse") is not None and row.get("type") != "field_recommendation"
        ]
        best_accuracy = min(
            metric_rows,
            key=lambda row: float(row["reconstruction_rmse"]),
        ) if metric_rows else None
        best_snr = max(
            metric_rows,
            key=lambda row: float(row["signal_to_residual_db"]),
        ) if metric_rows else None
        best_flux = min(
            metric_rows,
            key=lambda row: float(row["total_flux_relative_error"]),
        ) if metric_rows else None
        timed_rows = [row for row in metric_rows if _row_duration_ms(row) is not None]
        fastest = min(timed_rows, key=lambda row: float(_row_duration_ms(row))) if timed_rows else None
        stack_rec = next(
            (
                row
                for row in case_rows
                if row.get("library") == "torchsky_recommendation"
                and row.get("profile") == "default_general"
            ),
            None,
        )
        field_rec = next(
            (
                row
                for row in case_rows
                if row.get("type") == "field_recommendation"
                and row.get("profile") == "default_general"
            ),
            None,
        )

        def fmt_metric(row: Mapping[str, Any] | None, key: str) -> str:
            if row is None:
                return "n/a"
            return f"{_row_label(row)} ({float(row[key]):.4g})"

        def fmt_time(row: Mapping[str, Any] | None) -> str:
            if row is None:
                return "n/a"
            duration = _row_duration_ms(row)
            if duration is None:
                return "n/a"
            return f"{_row_label(row)} ({duration:.3f} ms)"

        lines.append(
            "| "
            + " | ".join(
                [
                    case,
                    fmt_metric(best_accuracy, "reconstruction_rmse"),
                    fmt_metric(best_snr, "signal_to_residual_db"),
                    fmt_metric(best_flux, "total_flux_relative_error"),
                    fmt_time(fastest),
                    _row_label(stack_rec) if stack_rec is not None else "n/a",
                    _row_label(field_rec) if field_rec is not None else "n/a",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def map_benchmark_row_key(row: Mapping[str, Any]) -> str:
    parts = [
        str(row.get("case", "")),
        str(row.get("library", "")),
        str(row.get("type", "")),
        str(row.get("method", "")),
        str(row.get("variant", "")),
        str(row.get("model_kind", "")),
        str(row.get("profile", "")),
    ]
    return "|".join(parts)


def check_map_benchmark_regressions(
    rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    *,
    reconstruction_rmse_tolerance: float = 0.10,
    signal_to_residual_db_tolerance: float = 0.10,
    total_flux_relative_error_tolerance: float = 0.10,
    time_tolerance: float = 0.15,
) -> list[str]:
    baseline_by_key = {
        map_benchmark_row_key(row): row
        for row in baseline_rows
    }
    errors: list[str] = []
    for row in rows:
        key = map_benchmark_row_key(row)
        baseline = baseline_by_key.get(key)
        if baseline is None:
            continue
        current_status = str(row.get("status", "OK"))
        baseline_status = str(baseline.get("status", "OK"))
        if baseline_status == "OK" and current_status != "OK":
            errors.append(f"{key}: status regressed from OK to {current_status}")
            continue
        if baseline_status != "OK":
            continue

        def compare_upper(metric: str, tolerance: float) -> None:
            base_value = baseline.get(metric)
            current_value = row.get(metric)
            if base_value is None or current_value is None:
                return
            allowed = float(base_value) * (1.0 + tolerance)
            if float(current_value) > allowed:
                errors.append(
                    f"{key}: {metric} regressed {float(current_value):.6g} > {allowed:.6g}"
                )

        def compare_lower(metric: str, tolerance: float) -> None:
            base_value = baseline.get(metric)
            current_value = row.get(metric)
            if base_value is None or current_value is None:
                return
            floor = float(base_value) - abs(float(base_value)) * tolerance
            if float(current_value) < floor:
                errors.append(
                    f"{key}: {metric} regressed {float(current_value):.6g} < {floor:.6g}"
                )

        compare_upper("reconstruction_rmse", reconstruction_rmse_tolerance)
        compare_lower("signal_to_residual_db", signal_to_residual_db_tolerance)
        compare_upper(
            "total_flux_relative_error",
            total_flux_relative_error_tolerance,
        )
        if not bool(baseline.get("skip_time_regression", False)):
            if baseline.get("fit_time_ms") is not None or row.get("fit_time_ms") is not None:
                compare_upper("fit_time_ms", time_tolerance)
            else:
                compare_upper("time_ms", time_tolerance)
    return errors


def write_map_benchmark_artifacts(
    rows: Sequence[Mapping[str, Any]],
    *,
    json_out: Path | None = None,
    markdown_out: Path | None = None,
) -> str:
    markdown = build_map_benchmark_markdown(rows)
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(list(rows), indent=2), encoding="utf-8")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(markdown, encoding="utf-8")
    return markdown
