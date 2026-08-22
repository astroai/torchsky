"""Fit and score stationary versus adaptive SMACS operator calibrations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from torchsky.calibration_evidence import load_operator_calibration_split
from torchsky.map.operators import (
    AdaptiveSpatioSpectralTemporalEncoder,
    BackgroundCalibrationTerm,
    ConditionedBackgroundField,
    ConditionedKernelPSF,
    ExtendedResidualKernelPSF,
    LowRankChromaticPSF,
    PSFCalibrationTerm,
    SpatioSpectralTemporalEncoder,
    fit_operator_calibration_terms,
    operator_calibration_identifiability_report,
    operator_calibration_validation_pruning_report,
    operator_calibration_score_report,
)

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return device


def _weighted_mean(value: Tensor, weight: Tensor) -> Tensor:
    weights = weight.to(dtype=torch.float64)
    return (value.to(dtype=torch.float64) * weights).sum() / weights.sum()


def _initial_fwhm_pixels(kernels: Tensor, sample_weight: Tensor) -> float:
    shape = kernels.to(dtype=torch.float64).clamp_min(0.0)
    shape = shape / shape.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0e-12)
    radius = shape.shape[-1] // 2
    offsets = torch.arange(-radius, radius + 1, dtype=torch.float64)
    yy, xx = torch.meshgrid(offsets, offsets, indexing="ij")
    sigma_squared = (shape * (xx.square() + yy.square())).sum(dim=(-2, -1)) / 2.0
    order = torch.argsort(sigma_squared)
    ordered_weight = sample_weight.to(dtype=torch.float64)[order]
    threshold = 0.5 * ordered_weight.sum()
    index = int(
        torch.searchsorted(ordered_weight.cumsum(0), threshold).clamp_max(
            order.numel() - 1
        )
    )
    sigma = math.sqrt(max(float(sigma_squared[order[index]]), 0.25))
    return min(2.354820045 * sigma, float(radius))


def _whiten_evidence(evidence: Tensor, weight: Tensor) -> Tensor:
    values = evidence.to(dtype=torch.float64).reshape(evidence.shape[0], -1)
    weights = weight.to(dtype=torch.float64).reshape(-1, 1)
    mean = (values * weights).sum(dim=0) / weights.sum()
    variance = (weights * (values - mean).square()).sum(dim=0) / weights.sum()
    keep = variance > torch.finfo(torch.float64).eps
    if not bool(keep.any()):
        return torch.zeros((values.shape[0], 1), dtype=torch.float64)
    return (values[:, keep] - mean[keep]) / torch.sqrt(variance[keep])


def _residualize_other_axes(
    evidence: Tensor,
    coordinates: tuple[Tensor, Tensor, Tensor, Tensor],
    *,
    target_axis: int,
    weight: Tensor,
) -> Tensor:
    columns = [torch.ones(evidence.shape[0], dtype=torch.float64)]
    for index, coordinate in enumerate(coordinates):
        if index == target_axis:
            continue
        value = coordinate.to(dtype=torch.float64)
        scale = float(value.max() - value.min())
        if scale > 0.0:
            columns.append((value - value.mean()) / scale)
    design = torch.stack(columns, dim=1)
    weights = weight.to(dtype=torch.float64)
    weighted_design = design * torch.sqrt(weights[:, None])
    weighted_evidence = evidence * torch.sqrt(weights[:, None])
    gram = weighted_design.T @ weighted_design
    ridge = torch.eye(gram.shape[0], dtype=gram.dtype) * 1.0e-6
    ridge[0, 0] = 0.0
    coefficients = torch.linalg.solve(
        gram + ridge,
        weighted_design.T @ weighted_evidence,
    )
    return evidence - design @ coefficients


def _adaptive_encoder(
    arrays: dict[str, Tensor],
    *,
    role: str,
) -> tuple[AdaptiveSpatioSpectralTemporalEncoder, dict[str, Any]]:
    coordinates = (
        arrays["detector_x"],
        arrays["detector_y"],
        arrays["wavelength"],
        arrays["time_mjd"],
    )
    encoder = AdaptiveSpatioSpectralTemporalEncoder.from_samples(
        spatial_x=coordinates[0],
        spatial_y=coordinates[1],
        wavelength=coordinates[2],
        time_mjd=coordinates[3],
        spatial_bands=4,
        spectral_bands=3,
        temporal_bands=2,
    )
    evidence = (
        arrays["target_kernels"]
        if role == "psf"
        else arrays["target_values"][:, None]
    )
    weight = arrays["sample_weight"].to(dtype=torch.float64)
    whitened = _whiten_evidence(evidence, weight)
    diagnostics: dict[str, Any] = {}
    axes = (
        ("spatial_x", encoder.x_axis),
        ("spatial_y", encoder.y_axis),
        ("spectral", encoder.spectral_axis),
        ("time_mjd", encoder.temporal_axis),
    )
    for index, (name, axis) in enumerate(axes):
        residual = _residualize_other_axes(
            whitened,
            coordinates,
            target_axis=index,
            weight=weight,
        )
        diagnostics[name] = axis.initialize_from_evidence(
            coordinates[index], residual, weights=weight
        )
    return encoder, diagnostics


def _stationary_encoder(arrays: dict[str, Tensor]) -> SpatioSpectralTemporalEncoder:
    def center_scale(values: Tensor) -> tuple[float, float]:
        values = values.to(dtype=torch.float64)
        return float(values.mean()), max(float(values.max() - values.min()), 1.0)

    x_center, x_scale = center_scale(arrays["detector_x"])
    y_center, y_scale = center_scale(arrays["detector_y"])
    wavelength_center, spectral_scale = center_scale(arrays["wavelength"])
    time_center, temporal_scale = center_scale(arrays["time_mjd"])
    return SpatioSpectralTemporalEncoder(
        spatial_center=(x_center, y_center),
        spatial_scale=(x_scale, y_scale),
        wavelength_center=wavelength_center,
        spectral_scale=spectral_scale,
        time_center_mjd=time_center,
        temporal_scale_days=temporal_scale,
        spatial_frequencies=0,
        spectral_frequencies=0,
        temporal_frequencies=0,
    )


def _make_model(
    arrays: dict[str, Tensor],
    *,
    role: str,
    family: str,
    device: torch.device,
    psf_model: str = "fitted_gaussian_kernel",
) -> tuple[nn.Module, dict[str, Any]]:
    if family not in {"stationary", "adaptive"}:
        raise ValueError("family must be stationary or adaptive")
    adaptive = family == "adaptive"
    encoder: AdaptiveSpatioSpectralTemporalEncoder | SpatioSpectralTemporalEncoder
    initializer: dict[str, Any]
    if adaptive:
        encoder, initializer = _adaptive_encoder(arrays, role=role)
    else:
        encoder = _stationary_encoder(arrays)
        initializer = {}
    if role == "psf":
        initial_fwhm = _initial_fwhm_pixels(
            arrays["target_kernels"], arrays["sample_weight"]
        )
        if adaptive and psf_model == "learned_lowrank":
            # Learned low-rank chromatic arm: control-grid x Legendre model
            # anchored on calibration-star kernels (never scene residuals).
            from torchsky.map.operators import LowRankChromaticPSF

            model = LowRankChromaticPSF(
                grid_side=4,
                spectral_degree=2,
                base_sigma_pix=initial_fwhm / 2.354820045,
                kernel_size=int(arrays["target_kernels"].shape[-1]),
            )
            initializer["initial_fwhm_pixels"] = initial_fwhm
            initializer["psf_model"] = "learned_lowrank"
            return model.to(device), initializer
        model = ConditionedKernelPSF(
            encoder,
            n_basis=4 if adaptive else 1,
            kernel_size=int(arrays["target_kernels"].shape[-1]),
            init_fwhm_pix=initial_fwhm,
            hidden_dim=32,
        )
        if not adaptive:
            for parameter in model.mixer.parameters():
                parameter.requires_grad_(False)
        initializer["initial_fwhm_pixels"] = initial_fwhm
    else:
        model = ConditionedBackgroundField(encoder, hidden_dim=32, hidden_layers=2)
        output_layer = model.network[-1]
        if not isinstance(output_layer, nn.Linear):
            raise TypeError("background network must end with a linear layer")
        with torch.no_grad():
            output_layer.bias.fill_(
                _weighted_mean(arrays["target_values"], arrays["sample_weight"])
            )
        if not adaptive:
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            model.network[-1].bias.requires_grad_(True)
    return model.to(device), initializer


def _term(
    model: nn.Module,
    arrays: dict[str, Tensor],
    *,
    role: str,
    name: str,
    background_loss_weight: float = 1.0,
) -> PSFCalibrationTerm | BackgroundCalibrationTerm:
    common = {
        "detector_x": arrays["detector_x"],
        "detector_y": arrays["detector_y"],
        "wavelength": arrays["wavelength"],
        "time_mjd": arrays["time_mjd"],
        "sample_weight": arrays["sample_weight"],
        "name": name,
    }
    if role == "psf":
        if not isinstance(
            model, (ConditionedKernelPSF, ExtendedResidualKernelPSF, LowRankChromaticPSF)
        ):
            raise TypeError("PSF calibration requires a conditioned kernel PSF")
        return PSFCalibrationTerm(
            model=model,
            target_kernels=arrays["target_kernels"],
            pixel_weight=arrays["pixel_weight"],
            **common,
        )
    if not isinstance(model, ConditionedBackgroundField):
        raise TypeError("background calibration requires ConditionedBackgroundField")
    return BackgroundCalibrationTerm(
        model=model,
        target_values=arrays["target_values"],
        loss_weight=background_loss_weight,
        **common,
    )


def _background_loss_weight(arrays: dict[str, Tensor]) -> float:
    target = arrays["target_values"]
    weight = arrays["sample_weight"]
    variance = (
        (target - _weighted_mean(target, weight)).square() * weight
    ).sum() / weight.sum()
    return 1.0 / max(float(variance), 1.0e-12)


def _weighted_median(value: Tensor, weight: Tensor) -> float:
    order = torch.argsort(value)
    cumulative = weight[order].to(dtype=torch.float64).cumsum(0)
    index = int(torch.searchsorted(cumulative, 0.5 * cumulative[-1]))
    return float(value[order[index]])


def _subset_arrays(arrays: dict[str, Tensor], mask: Tensor) -> dict[str, Tensor]:
    return {name: value[mask] for name, value in arrays.items()}


def _slice_definitions(
    arrays: dict[str, dict[str, Tensor]],
    rows: dict[str, list[dict[str, Any]]],
    *,
    minimum_samples: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    definitions: list[tuple[str, str, dict[str, Any], dict[str, Tensor]]] = []
    bands = sorted({str(row["band"]).upper() for row in rows["train"]})
    for band in bands:
        definitions.append(
            (
                "band",
                band,
                {"band": band},
                {
                    split: torch.tensor(
                        [str(row["band"]).upper() == band for row in rows[split]],
                        dtype=torch.bool,
                    )
                    for split in ("validation", "test")
                },
            )
        )
    x_median = _weighted_median(
        arrays["train"]["detector_x"], arrays["train"]["sample_weight"]
    )
    y_median = _weighted_median(
        arrays["train"]["detector_y"], arrays["train"]["sample_weight"]
    )
    for x_name, x_high in (("x_low", False), ("x_high", True)):
        for y_name, y_high in (("y_low", False), ("y_high", True)):
            definitions.append(
                (
                    "detector_quadrant",
                    f"{x_name}:{y_name}",
                    {"x_median": x_median, "y_median": y_median},
                    {
                        split: (
                            (arrays[split]["detector_x"] > x_median)
                            if x_high
                            else (arrays[split]["detector_x"] <= x_median)
                        )
                        & (
                            (arrays[split]["detector_y"] > y_median)
                            if y_high
                            else (arrays[split]["detector_y"] <= y_median)
                        )
                        for split in ("validation", "test")
                    },
                )
            )
    time_median = _weighted_median(
        arrays["train"]["time_mjd"], arrays["train"]["sample_weight"]
    )
    for label, high in (("early", False), ("late", True)):
        definitions.append(
            (
                "time_epoch",
                label,
                {"training_time_median_mjd": time_median},
                {
                    split: (
                        (arrays[split]["time_mjd"] > time_median)
                        if high
                        else (arrays[split]["time_mjd"] <= time_median)
                    )
                    for split in ("validation", "test")
                },
            )
        )
    included = []
    skipped = []
    for kind, label, metadata, masks in definitions:
        counts = {split: int(mask.sum()) for split, mask in masks.items()}
        descriptor = {
            "kind": kind,
            "label": label,
            "metadata": metadata,
            "sample_counts": counts,
        }
        if min(counts.values()) < minimum_samples:
            skipped.append(descriptor)
        else:
            descriptor["masks"] = masks
            included.append(descriptor)
    return included, skipped


def _score_slices(
    model: nn.Module,
    arrays: dict[str, dict[str, Tensor]],
    rows: dict[str, list[dict[str, Any]]],
    *,
    role: str,
    detector: str,
    background_loss_weight: float,
    minimum_samples: int,
) -> dict[str, Any]:
    included, skipped = _slice_definitions(
        arrays, rows, minimum_samples=minimum_samples
    )
    reports = []
    for descriptor in included:
        masks = descriptor.pop("masks")
        descriptor["reports"] = {
            split: operator_calibration_score_report(
                [
                    _term(
                        model,
                        _subset_arrays(arrays[split], masks[split]),
                        role=role,
                        name=(
                            f"{detector}:{role}:{split}:"
                            f"{descriptor['kind']}:{descriptor['label']}"
                        ),
                        background_loss_weight=background_loss_weight,
                    )
                ],
                split=split,
            )
            for split in ("validation", "test")
        }
        reports.append(descriptor)
    return {
        "minimum_samples_per_held_out_split": minimum_samples,
        "included": reports,
        "skipped": skipped,
    }


def _fit_group(
    manifest: Path,
    *,
    detector: str,
    role: str,
    family: str,
    steps: int,
    lr: float,
    device: torch.device,
    output_dir: Path,
    seed: int,
    manifest_payload: dict[str, Any],
    minimum_slice_samples: int,
    psf_model: str = "fitted_gaussian_kernel",
) -> dict[str, Any]:
    arrays = {
        split: load_operator_calibration_split(
            manifest,
            split=split,
            role=role,
            detector=detector,
            verify_hashes=True,
        )
        for split in ("train", "validation", "test")
    }
    rows = {
        split: [
            row
            for row in manifest_payload.get("evidence", [])
            if row.get("split") == split
            and row.get("role") == role
            and str(row.get("detector", "")).upper() == detector
        ]
        for split in ("train", "validation", "test")
    }
    if any(len(rows[split]) != arrays[split]["detector_x"].numel() for split in rows):
        raise RuntimeError("manifest row order/count does not match loaded evidence")
    torch.manual_seed(seed)
    model, initializer = _make_model(
        arrays["train"],
        role=role,
        family=family,
        device=device,
        psf_model=psf_model,
    )
    background_loss_weight = (
        _background_loss_weight(arrays["train"]) if role == "background" else 1.0
    )
    train_term = _term(
        model,
        arrays["train"],
        role=role,
        name=f"{detector}:{role}:train",
        background_loss_weight=background_loss_weight,
    )
    fit = fit_operator_calibration_terms(
        [train_term],
        steps=steps,
        lr=lr,
        gate_weight=1.0e-4 if family == "adaptive" else 0.0,
        scale_shift_weight=1.0e-4 if family == "adaptive" else 0.0,
    )
    reports = {
        split: operator_calibration_score_report(
            [
                _term(
                    model,
                    arrays[split],
                    role=role,
                    name=f"{detector}:{role}:{split}",
                    background_loss_weight=background_loss_weight,
                )
            ],
            split=split,
        )
        for split in ("validation", "test")
    }
    checkpoint = output_dir / f"{detector.lower()}_{role}_{family}.pt"
    torch.save(
        {
            "schema_version": 1,
            "detector": detector,
            "role": role,
            "family": family,
            "psf_model": psf_model if role == "psf" else None,
            "state_dict": model.state_dict(),
        },
        checkpoint,
    )
    result: dict[str, Any] = {
        "detector": detector,
        "role": role,
        "family": family,
        "initializer": initializer,
        "fit": fit,
        "validation_report": reports["validation"],
        "test_report": reports["test"],
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "held_out_slices": _score_slices(
            model,
            arrays,
            rows,
            role=role,
            detector=detector,
            background_loss_weight=background_loss_weight,
            minimum_samples=minimum_slice_samples,
        ),
    }
    if family == "adaptive" and isinstance(
        getattr(model, "encoder", None), AdaptiveSpatioSpectralTemporalEncoder
    ):
        result["learned_scales"] = model.encoder.scale_diagnostics()
        result["identifiability"] = {
            split: operator_calibration_identifiability_report(
                _term(
                    model,
                    arrays[split],
                    role=role,
                    name=f"{detector}:{role}:{split}",
                    background_loss_weight=background_loss_weight,
                ),
                split=split,
                seed=seed,
            )
            for split in ("validation", "test")
        }
        if isinstance(model, LowRankChromaticPSF):
            # No adaptive gates to prune on a control-grid chromatic PSF.
            result["validation_pruning"] = {
                "schema_version": 1,
                "selection_split": "validation",
                "pruned_bands": [],
                "note": "learned low-rank chromatic PSF has no adaptive gates",
            }
        else:
            result["validation_pruning"] = (
                operator_calibration_validation_pruning_report(
                    _term(
                        model,
                        arrays["validation"],
                        role=role,
                        name=f"{detector}:{role}:validation",
                        background_loss_weight=background_loss_weight,
                    ),
                    _term(
                        model,
                        arrays["test"],
                        role=role,
                        name=f"{detector}:{role}:test",
                        background_loss_weight=background_loss_weight,
                    ),
                )
            )
    return result


def _held_out_comparisons(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (group["detector"], group["role"], group["family"]): group
        for group in groups
    }
    comparisons = []
    for detector, role, family in sorted(indexed):
        if family != "adaptive":
            continue
        stationary = indexed[(detector, role, "stationary")]
        adaptive = indexed[(detector, role, "adaptive")]
        comparison: dict[str, Any] = {"detector": detector, "role": role}
        for split in ("validation", "test"):
            stationary_loss = float(
                stationary[f"{split}_report"]["terms"][0]["raw_loss"]
            )
            adaptive_loss = float(
                adaptive[f"{split}_report"]["terms"][0]["raw_loss"]
            )
            ratio = adaptive_loss / max(
                stationary_loss, torch.finfo(torch.float64).tiny
            )
            comparison[split] = {
                "stationary_raw_loss": stationary_loss,
                "adaptive_raw_loss": adaptive_loss,
                "adaptive_to_stationary_ratio": ratio,
                "adaptive_improvement_fraction": 1.0 - ratio,
            }
        comparisons.append(comparison)
    return comparisons


def _held_out_slice_comparisons(
    groups: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    indexed = {
        (group["detector"], group["role"], group["family"]): group
        for group in groups
    }
    comparisons = []
    for detector, role, family in sorted(indexed):
        if family != "adaptive":
            continue
        stationary = indexed[(detector, role, "stationary")]
        adaptive = indexed[(detector, role, "adaptive")]
        stationary_slices = {
            (item["kind"], item["label"]): item
            for item in stationary["held_out_slices"]["included"]
        }
        for item in adaptive["held_out_slices"]["included"]:
            key = (item["kind"], item["label"])
            control = stationary_slices[key]
            comparison: dict[str, Any] = {
                "detector": detector,
                "role": role,
                "kind": key[0],
                "label": key[1],
                "metadata": item["metadata"],
                "sample_counts": item["sample_counts"],
            }
            for split in ("validation", "test"):
                stationary_loss = float(
                    control["reports"][split]["terms"][0]["raw_loss"]
                )
                adaptive_loss = float(
                    item["reports"][split]["terms"][0]["raw_loss"]
                )
                ratio = adaptive_loss / max(
                    stationary_loss, torch.finfo(torch.float64).tiny
                )
                comparison[split] = {
                    "stationary_raw_loss": stationary_loss,
                    "adaptive_raw_loss": adaptive_loss,
                    "adaptive_to_stationary_ratio": ratio,
                    "adaptive_improvement_fraction": 1.0 - ratio,
                }
            comparisons.append(comparison)
    return comparisons


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--minimum-slice-samples", type=int, default=20)
    parser.add_argument(
        "--psf-model",
        choices=("fitted_gaussian_kernel", "learned_lowrank"),
        default="fitted_gaussian_kernel",
        help="PSF arm for the adaptive family: conditioned kernel bank "
        "(default) or learned low-rank chromatic control-grid model.",
    )
    args = parser.parse_args()

    manifest = args.manifest.resolve()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if args.minimum_slice_samples < 1:
        raise ValueError("minimum_slice_samples must be positive")
    detectors = sorted(
        {
            str(row["detector"]).upper()
            for row in payload.get("evidence", [])
            if row.get("split") == "train"
        }
    )
    if not detectors:
        raise ValueError("manifest contains no train-split detector evidence")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _device(args.device)
    groups = []
    for detector_index, detector in enumerate(detectors):
        for role_index, role in enumerate(("background", "psf")):
            for family_index, family in enumerate(("stationary", "adaptive")):
                groups.append(
                    _fit_group(
                        manifest,
                        detector=detector,
                        role=role,
                        family=family,
                        steps=args.steps,
                        lr=args.lr,
                        device=device,
                        output_dir=output_dir,
                        seed=args.seed
                        + 100 * detector_index
                        + 10 * role_index
                        + family_index,
                        manifest_payload=payload,
                        minimum_slice_samples=args.minimum_slice_samples,
                        psf_model=args.psf_model,
                    )
                )
    report = {
        "schema_version": 1,
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "device": str(device),
        "steps": args.steps,
        "learning_rate": args.lr,
        "seed": args.seed,
        "detectors": detectors,
        "groups": groups,
        "held_out_comparisons": _held_out_comparisons(groups),
        "held_out_slice_comparisons": _held_out_slice_comparisons(groups),
        "minimum_slice_samples": args.minimum_slice_samples,
        "psf_model": args.psf_model,
    }
    output = output_dir / "operator_calibration_fit.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    console_report = {
        "report": str(output),
        "manifest_sha256": report["manifest_sha256"],
        "detectors": detectors,
        "held_out_comparisons": report["held_out_comparisons"],
        "held_out_slice_comparisons": report["held_out_slice_comparisons"],
        "validation_pruning": [
            {
                "detector": group["detector"],
                "role": group["role"],
                "report": group["validation_pruning"],
            }
            for group in groups
            if group["family"] == "adaptive" and "validation_pruning" in group
        ],
    }
    print(json.dumps(console_report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
