#!/usr/bin/env python3
"""Run A1 NeRF-style or A2 SIREN experiments from a completed baseline.

By default this preserves the original architecture-ablation behavior and reuses
all geometry and sampling settings from the baseline manifest. Optionally,
``--target-canvas-fits`` replaces the reconstruction grid, while
``--max-pixels-per-exposure`` and ``--sampling-mode`` explicitly control the
loaded native-pixel pool and mini-batch traversal. Native CRF/CAL pixels remain
the training targets; a template FITS supplies geometry only.

Artifacts use architecture/geometry/sampling-specific prefixes and are never
silently overwritten.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torchsky  # noqa: E402
from torchsky.map.wcs_aware_epoch_sampling import install_epoch_sampler  # noqa: E402
from torchsky.map.wcs_aware_inr import UnionCanvas  # noqa: E402
from torchsky.map.wcs_aware_literature_inr import (  # noqa: E402
    WCSAwareLiteratureINR,
)
from torchsky.map.wcs_aware_target_canvas import (  # noqa: E402
    filter_sampled_pools_to_canvas,
    load_target_canvas_from_fits,
    sampled_pool_canvas_coverage,
)
from torchsky.map.wcs_aware_training import (  # noqa: E402
    build_union_coverage_mask,
    effective_batch_size,
    load_exposure_pool,
    normalize_pools,
    render_union_band,
    seed_everything,
    train_model,
)


ARCH_CHOICES = ("A1_nerf", "A2_siren")
SAMPLING_CHOICES = ("with_replacement", "epoch_without_replacement")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--architecture", choices=ARCH_CHOICES, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--render-tile-pixels", type=int, default=None)
    parser.add_argument("--coverage-tile-rows", type=int, default=None)
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override baseline training steps; default reuses the baseline value.",
    )
    parser.add_argument(
        "--max-pixels-per-exposure",
        type=int,
        default=None,
        help=(
            "Override the baseline native-pixel pool cap. Use 0 to load every valid "
            "native SCI/ERR pixel; omitted means reuse the baseline cap."
        ),
    )
    parser.add_argument(
        "--sampling-mode",
        choices=SAMPLING_CHOICES,
        default="with_replacement",
        help=(
            "with_replacement preserves historical mini-batch sampling. "
            "epoch_without_replacement walks through a per-exposure shuffled "
            "permutation before that exposure reshuffles."
        ),
    )

    parser.add_argument(
        "--target-canvas-fits",
        type=Path,
        default=None,
        help=(
            "Optional 2D FITS template whose celestial WCS and exact image shape "
            "replace the baseline auto-union canvas. The FITS pixel values are "
            "never used as training targets."
        ),
    )
    parser.add_argument(
        "--target-canvas-hdu",
        default=None,
        help=(
            "Optional HDU name/index for --target-canvas-fits. By default the "
            "first 2D image HDU is selected; this works for compressed .fits.fz mosaics."
        ),
    )
    parser.add_argument(
        "--keep-outside-target-samples",
        action="store_true",
        help=(
            "Do not filter loaded native pixels outside the target canvas. The "
            "default filters them so every sampled batch entry belongs to the "
            "requested reconstruction footprint."
        ),
    )

    parser.add_argument("--nerf-num-frequencies", type=int, default=10)
    parser.add_argument("--nerf-hidden-dim", type=int, default=256)
    parser.add_argument("--nerf-depth", type=int, default=8)
    parser.add_argument("--nerf-skip-after-layer", type=int, default=4)

    parser.add_argument("--siren-hidden-dim", type=int, default=256)
    parser.add_argument("--siren-hidden-layers", type=int, default=3)
    parser.add_argument("--siren-first-omega0", type=float, default=30.0)
    parser.add_argument("--siren-hidden-omega0", type=float, default=30.0)
    return parser


def _load_manifest(path: Path) -> tuple[Path, dict[str, Any]]:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    for key in ("exposures", "intensity_scale", "union_canvas", "bandpass", "args"):
        if key not in payload:
            raise ValueError(f"baseline manifest is missing {key!r}")
    return resolved, payload


def _parse_batch_spec(raw: Any) -> int | list[int]:
    if isinstance(raw, int):
        if raw <= 0:
            raise ValueError("baseline batch size must be positive")
        return int(raw)
    if isinstance(raw, list):
        values = [int(value) for value in raw]
        if len(values) not in {1, 2} or min(values) <= 0:
            raise ValueError(f"invalid baseline batch-size: {raw!r}")
        return values[0] if len(values) == 1 else values
    raise ValueError(f"unsupported baseline batch-size representation: {raw!r}")


def _selected_from_manifest(
    manifest: dict[str, Any],
) -> list[tuple[Path, str, str]]:
    rows = sorted(manifest["exposures"], key=lambda item: int(item["index"]))
    selected: list[tuple[Path, str, str]] = []
    for expected, row in enumerate(rows):
        index = int(row["index"])
        if index != expected:
            raise ValueError(
                f"baseline exposure indices must be contiguous from zero; got {index} at {expected}"
            )
        path = Path(row["path"]).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"baseline exposure {index} does not exist: {path}. "
                "For a tweakreg_frozen baseline this should be the saved corrected FITS."
            )
        selected.append((path, str(row["band"]), str(row["detector"])))
    if not selected:
        raise ValueError("baseline manifest contains no exposures")
    return selected


def _bandpass_from_manifest(
    manifest: dict[str, Any],
    *,
    num_exposures: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    metadata = manifest["bandpass"]
    if not isinstance(metadata, list) or len(metadata) != num_exposures:
        raise ValueError(
            "baseline bandpass metadata must contain one row per exposure"
        )
    wavelengths: list[list[float]] = []
    weights: list[list[float]] = []
    for index, info in enumerate(metadata):
        try:
            wave = [float(value) for value in info["quadrature_wavelength"]]
            weight = [float(value) for value in info["quadrature_weight"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"baseline bandpass row {index} lacks valid quadrature samples"
            ) from exc
        if len(wave) == 0 or len(wave) != len(weight):
            raise ValueError(f"invalid quadrature lengths in baseline row {index}")
        wavelengths.append(wave)
        weights.append(weight)
    wavelength = torch.tensor(wavelengths, dtype=torch.float32)
    weight = torch.tensor(weights, dtype=torch.float32)
    if wavelength.ndim != 2 or weight.shape != wavelength.shape:
        raise ValueError("baseline quadrature must form rectangular [E,L] tensors")
    if not bool(torch.isfinite(wavelength).all()) or bool((wavelength <= 0).any()):
        raise ValueError("baseline quadrature wavelengths must be finite and positive")
    if not bool(torch.isfinite(weight).all()) or bool((weight < 0).any()):
        raise ValueError("baseline quadrature weights must be finite and non-negative")
    weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(1.0e-20)
    return wavelength, weight


def _canvas_from_manifest(manifest: dict[str, Any]) -> UnionCanvas:
    info = manifest["union_canvas"]
    return UnionCanvas(
        header=dict(info["header"]),
        shape=tuple(int(value) for value in info["shape"]),
        pixel_scale_arcsec=float(info["pixel_scale_arcsec"]),
    )


def _validate_baseline_for_architecture_ablation(manifest: dict[str, Any]) -> None:
    args = manifest.get("args", {})
    alignment_mode = str(args.get("alignment_mode", ""))
    if alignment_mode not in {"original_wcs", "tweakreg_frozen"}:
        raise ValueError(
            "A1/A2 architecture experiments require a frozen geometry baseline; "
            f"got alignment_mode={alignment_mode!r}"
        )
    if str(args.get("psf_mode", "none")) != "none":
        raise ValueError(
            "This runner currently targets the completed no-PSF run2. "
            "Use a no-PSF baseline manifest so the observation operator remains fixed."
        )
    if manifest.get("psf") not in ([], None):
        raise ValueError("baseline manifest unexpectedly contains PSF calibration metadata")
    if str(args.get("robust_loss", args.get("loss", ""))) != "whitened_l2":
        raise ValueError(
            "This runner expects the run2 whitened_l2 baseline so the likelihood stays fixed"
        )


def _artifact_prefix(
    architecture: str,
    *,
    target_canvas: bool,
    sampling_mode: str,
    max_pixels_override: int | None,
    resolved_max_pixels: int,
) -> str:
    parts = [architecture]
    parts.append("target_canvas" if target_canvas else "literature_ablation")
    if max_pixels_override is not None:
        parts.append("all_valid" if resolved_max_pixels == 0 else f"maxpix{resolved_max_pixels}")
    if sampling_mode == "epoch_without_replacement":
        parts.append("epoch")
    return "_".join(parts)


def _assert_no_existing_artifacts(output_dir: Path, prefix: str) -> None:
    collisions = sorted(output_dir.glob(f"{prefix}*")) if output_dir.exists() else []
    collisions.extend(sorted(output_dir.glob(f"*_{prefix}_bandpass_scene.fits")))
    if collisions:
        shown = ", ".join(path.name for path in collisions[:8])
        raise FileExistsError(
            f"refusing to overwrite existing {prefix} artifacts: {shown}"
        )


def main() -> int:
    args = build_parser().parse_args()
    baseline_path, baseline = _load_manifest(args.baseline_manifest)
    _validate_baseline_for_architecture_ablation(baseline)

    baseline_args = baseline["args"]
    seed = int(baseline_args.get("seed", 17))
    seed_everything(seed)

    if args.max_pixels_per_exposure is None:
        max_pixels = int(baseline_args.get("max_pixels_per_exposure", 200000))
    else:
        max_pixels = int(args.max_pixels_per_exposure)
    if max_pixels < 0:
        raise ValueError("max-pixels-per-exposure must be non-negative; 0 means all valid")

    if args.sampling_mode == "epoch_without_replacement":
        install_epoch_sampler()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = _artifact_prefix(
        args.architecture,
        target_canvas=args.target_canvas_fits is not None,
        sampling_mode=args.sampling_mode,
        max_pixels_override=args.max_pixels_per_exposure,
        resolved_max_pixels=max_pixels,
    )
    _assert_no_existing_artifacts(output_dir, prefix)

    selected = _selected_from_manifest(baseline)
    bands: list[str] = []
    for _, band, _ in selected:
        if band not in bands:
            bands.append(band)

    rng = np.random.default_rng(seed)
    dq_mode = str(baseline_args.get("dq_mode", "strict"))
    pools = [
        load_exposure_pool(
            path,
            band,
            detector,
            max_pixels=max_pixels,
            dq_mode=dq_mode,
            rng=rng,
        )
        for path, band, detector in selected
    ]

    scale = float(baseline["intensity_scale"])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("baseline intensity_scale must be finite and positive")
    pools = normalize_pools(pools, scale)

    if args.target_canvas_fits is None:
        canvas = _canvas_from_manifest(baseline)
        canvas_metadata: dict[str, Any] = {
            "mode": "baseline_manifest",
            "source_manifest": str(baseline_path),
            "shape": list(canvas.shape),
            "pixel_scale_arcsec": canvas.pixel_scale_arcsec,
        }
        canvas_coverage = sampled_pool_canvas_coverage(pools, canvas)
        canvas_coverage["filter_applied"] = False
    else:
        canvas, canvas_metadata = load_target_canvas_from_fits(
            args.target_canvas_fits,
            hdu_selector=args.target_canvas_hdu,
        )
        if args.keep_outside_target_samples:
            canvas_coverage = sampled_pool_canvas_coverage(pools, canvas)
            canvas_coverage["filter_applied"] = False
        else:
            pools, canvas_coverage = filter_sampled_pools_to_canvas(pools, canvas)
        print(
            json.dumps(
                {
                    "target_canvas": canvas_metadata,
                    "sampled_pixel_coverage": canvas_coverage,
                },
                indent=2,
            ),
            flush=True,
        )

    wavelength, weight = _bandpass_from_manifest(
        baseline,
        num_exposures=len(pools),
    )
    wavelength_min = float(wavelength.min())
    wavelength_max = float(wavelength.max())

    batch_spec = _parse_batch_spec(baseline_args.get("batch_size"))
    actual_batch_size = effective_batch_size(pools, batch_spec)
    if args.target_canvas_fits is None:
        baseline_effective = baseline.get("effective_total_batch_size")
        if baseline_effective is not None and int(baseline_effective) != actual_batch_size:
            raise ValueError(
                "effective batch size differs from the baseline manifest: "
                f"{actual_batch_size} != {baseline_effective}"
            )

    anchor_exposure = int(baseline_args.get("anchor_exposure", 0))
    if not 0 <= anchor_exposure < len(pools):
        raise ValueError("baseline anchor exposure is outside the exposure range")

    alignment_mode = str(baseline_args["alignment_mode"])
    learn_background = not bool(baseline_args.get("no_background", False))
    output_activation = str(baseline_args.get("output_activation", "softplus"))

    model = WCSAwareLiteratureINR(
        [pool.header for pool in pools],
        canvas,
        architecture=args.architecture,
        wavelength_min=wavelength_min,
        wavelength_max=wavelength_max,
        anchor_exposure=anchor_exposure,
        alignment_mode=alignment_mode,
        learn_background=learn_background,
        output_activation=output_activation,
        nerf_num_frequencies=args.nerf_num_frequencies,
        nerf_hidden_dim=args.nerf_hidden_dim,
        nerf_depth=args.nerf_depth,
        nerf_skip_after_layer=args.nerf_skip_after_layer,
        siren_hidden_dim=args.siren_hidden_dim,
        siren_hidden_layers=args.siren_hidden_layers,
        siren_first_omega_0=args.siren_first_omega0,
        siren_hidden_omega_0=args.siren_hidden_omega0,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    coverage_tile_rows = int(
        args.coverage_tile_rows
        if args.coverage_tile_rows is not None
        else baseline_args.get("coverage_tile_rows", 128)
    )
    render_tile_pixels = int(
        args.render_tile_pixels
        if args.render_tile_pixels is not None
        else baseline_args.get("render_tile_pixels", 131072)
    )
    steps = int(args.steps if args.steps is not None else baseline_args.get("steps", 10000))
    if steps <= 0:
        raise ValueError("training steps must be positive")

    sampling_metadata = {
        "mode": args.sampling_mode,
        "max_pixels_per_exposure": max_pixels,
        "all_valid_pixels": max_pixels == 0,
        "epoch_definition": (
            "independent per exposure; reshuffle after that exposure's loaded pool is exhausted"
            if args.sampling_mode == "epoch_without_replacement"
            else None
        ),
        "batch_spec": batch_spec,
        "effective_total_batch_size": actual_batch_size,
    }
    principle_parts = [
        "reuse baseline corrected native exposures, intensity normalization, bandpass, likelihood and optimizer",
        "replace S_theta architecture",
    ]
    if args.target_canvas_fits is not None:
        principle_parts.append("replace the common spatial canvas with the FITS template")
    if args.max_pixels_per_exposure is not None:
        principle_parts.append("override the loaded native-pixel pool cap")
    if args.sampling_mode == "epoch_without_replacement":
        principle_parts.append("traverse each exposure pool without replacement per exposure epoch")
    principle = "; ".join(principle_parts)

    experiment_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    artifact_manifest = {
        "experiment": {
            "architecture": args.architecture,
            "artifact_prefix": prefix,
            "baseline_manifest": str(baseline_path),
            "principle": principle,
            "architecture_metadata": model.architecture_metadata,
            "parameter_count": parameter_count,
            "trainable_parameter_count": trainable_parameter_count,
            "training_steps": steps,
        },
        "sampling": sampling_metadata,
        "target_canvas": canvas_metadata,
        "target_canvas_sample_coverage": canvas_coverage,
        "exposures": [
            {
                "index": index,
                "path": str(pool.path),
                "band": pool.band,
                "detector": pool.detector,
                "shape": list(pool.shape),
                "loaded_pixels_after_canvas_filter": pool.size,
            }
            for index, pool in enumerate(pools)
        ],
        "intensity_scale": scale,
        "union_canvas": {
            "shape": list(canvas.shape),
            "pixel_scale_arcsec": canvas.pixel_scale_arcsec,
            "header": canvas.header,
        },
        "bandpass_wavelength": wavelength.tolist(),
        "bandpass_weight": weight.tolist(),
        "alignment": baseline.get("alignment", {}),
        "effective_total_batch_size": actual_batch_size,
        "baseline_args": baseline_args,
        "experiment_args": experiment_args,
    }
    manifest_path = output_dir / f"{prefix}_manifest.json"
    manifest_path.write_text(
        json.dumps(artifact_manifest, indent=2),
        encoding="utf-8",
    )

    coverage: torch.Tensor | None = None
    coverage_path: Path | None = None
    if not args.skip_render:
        coverage = build_union_coverage_mask(
            [pool.header for pool in pools],
            canvas,
            device=device,
            tile_rows=coverage_tile_rows,
        )
        coverage_path = output_dir / f"{prefix}_union_coverage_mask.fits"
        torchsky.write(
            str(coverage_path),
            coverage.to(torch.uint8).cpu(),
            header=canvas.header,
            overwrite=False,
        )

    checkpoint_path = output_dir / f"{prefix}_checkpoint.pt"
    loss_log_path = output_dir / f"{prefix}_training_loss.jsonl"
    checkpoint_args = {
        "baseline_manifest": str(baseline_path),
        "baseline_args": baseline_args,
        "architecture": args.architecture,
        "architecture_metadata": model.architecture_metadata,
        "sampling": sampling_metadata,
        "target_canvas": canvas_metadata,
        "target_canvas_sample_coverage": canvas_coverage,
        "experiment_args": experiment_args,
    }

    history = train_model(
        model,
        pools,
        wavelength,
        weight,
        psf=None,
        device=device,
        steps=steps,
        batch_size=batch_spec,
        seed=seed,
        learning_rate=float(baseline_args.get("learning_rate", 2.0e-4)),
        alignment_learning_rate=float(
            baseline_args.get("alignment_learning_rate", 2.0e-5)
        ),
        background_learning_rate=float(
            baseline_args.get("background_learning_rate", 5.0e-5)
        ),
        weight_decay=float(baseline_args.get("weight_decay", 1.0e-6)),
        gradient_clip=float(baseline_args.get("gradient_clip", 1.0)),
        robust_loss=str(baseline_args.get("robust_loss", "whitened_l2")),
        robust_delta=float(baseline_args.get("robust_delta", 0.1)),
        max_relative_weight=float(baseline_args.get("max_relative_weight", 100.0)),
        alignment_regularization=float(
            baseline_args.get("alignment_regularization", 1.0e-4)
        ),
        background_regularization=float(
            baseline_args.get("background_regularization", 1.0e-5)
        ),
        psf_offset_chunk=int(baseline_args.get("psf_offset_chunk", 128)),
        checkpoint_psf_chunks=not bool(
            baseline_args.get("no_checkpoint_psf", False)
        ),
        amp=not bool(baseline_args.get("no_amp", False)),
        log_every=int(baseline_args.get("log_every", 100)),
        checkpoint_every=int(baseline_args.get("checkpoint_every", 1000)),
        checkpoint_path=checkpoint_path,
        loss_log_path=loss_log_path,
        loss_log_flush_every=100,
        intensity_scale=scale,
        canvas=canvas,
        checkpoint_args=checkpoint_args,
    )

    outputs: dict[str, str] = {}
    if coverage is not None:
        model.to(device)
        for band in bands:
            representative = next(
                index for index, pool in enumerate(pools) if pool.band == band
            )
            image = render_union_band(
                model,
                canvas,
                wavelength,
                weight,
                representative,
                anchor_exposure=anchor_exposure,
                device=device,
                tile_pixels=render_tile_pixels,
            )
            image = torch.where(coverage, image * scale, torch.zeros_like(image))
            path = output_dir / f"{band}_{prefix}_bandpass_scene.fits"
            torchsky.write(
                str(path),
                image.cpu(),
                header=canvas.header,
                overwrite=False,
            )
            outputs[band] = str(path)

    summary = {
        "architecture": args.architecture,
        "architecture_metadata": model.architecture_metadata,
        "baseline_manifest": str(baseline_path),
        "sampling": sampling_metadata,
        "target_canvas": canvas_metadata,
        "target_canvas_sample_coverage": canvas_coverage,
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "training_steps": steps,
        "checkpoint": str(checkpoint_path),
        "loss_log": str(loss_log_path),
        "coverage_mask": None if coverage_path is None else str(coverage_path),
        "outputs": outputs,
        "alignment": model.alignment_summary(),
        "background_physical": (model.background.detach().cpu() * scale).tolist(),
        "training": [asdict(record) for record in history],
    }
    summary_path = output_dir / f"{prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "architecture": args.architecture,
                "baseline_manifest": str(baseline_path),
                "sampling": sampling_metadata,
                "target_canvas": canvas_metadata,
                "manifest": str(manifest_path),
                "checkpoint": str(checkpoint_path),
                "loss_log": str(loss_log_path),
                "summary": str(summary_path),
                "outputs": outputs,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
