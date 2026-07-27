#!/usr/bin/env python3
"""Train the WCS-aware multi-wavelength INR on native SMACS J0723 data.

The official bandpass path queries JWST/NIRCam throughput curves from SVO FPS
through astroquery and stores local snapshots for reproducibility. Simpler
bandpasses and optional explicit PSF models remain available for ablations.

The historical auto-union canvas remains the default.  ``--target-canvas-fits``
optionally replaces only the common reconstruction grid with the celestial WCS
and exact shape of an existing 2D FITS mosaic (e.g. a released UPDC i2d image).
The template image values are never used in the training loss.
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
from torchsky.map.wcs_aware_calibration import (  # noqa: E402
    build_bandpass_calibration,
    build_psf_calibration,
)
from torchsky.map.wcs_aware_inr import (  # noqa: E402
    WCSAwareMultiWavelengthINR,
    build_union_canvas,
)
from torchsky.map.wcs_aware_target_canvas import (  # noqa: E402
    filter_sampled_pools_to_canvas,
    load_target_canvas_from_fits,
    sampled_pool_canvas_coverage,
)
from torchsky.map.wcs_aware_training import (  # noqa: E402
    build_union_coverage_mask,
    discover_exposures,
    effective_batch_size,
    load_exposure_pool,
    normalize_pools,
    render_union_band,
    robust_intensity_scale,
    run_tweakreg_alignment,
    seed_everything,
    train_model,
)

BAND_SPECS_ANGSTROM: dict[str, tuple[float, float, float]] = {
    "F090W": (9000.0, 8000.0, 10000.0),
    "F150W": (15000.0, 13000.0, 17000.0),
    "F200W": (20000.0, 17000.0, 23000.0),
    "F277W": (27700.0, 24000.0, 32000.0),
    "F356W": (35600.0, 31000.0, 40000.0),
    "F444W": (44400.0, 38000.0, 50000.0),
}


def _parse_csv(text: str) -> list[str]:
    return [item.strip().upper() for item in str(text).split(",") if item.strip()]


def _parse_band_float_map(text: str) -> dict[str, float]:
    output: dict[str, float] = {}
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"expected BAND=value, got {token!r}")
        key, value = token.split("=", 1)
        output[key.strip().upper()] = float(value)
    return output


def _jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exposure-dir",
        type=Path,
        default=Path("/arc/projects/ots/jwst_smacs0723/input"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/arc/projects/ots/jwst_smacs0723/results/wcs_aware_multiband_inr"
        ),
    )
    parser.add_argument(
        "--input-product", choices=("auto", "crf", "cal", "updc"), default="crf"
    )
    parser.add_argument("--bands", default=",".join(BAND_SPECS_ANGSTROM))
    parser.add_argument("--module", choices=("a", "b", "any"), default="b")
    parser.add_argument("--dq-mode", choices=("strict", "do_not_use"), default="strict")
    parser.add_argument("--max-exposures-per-band", type=int, default=None)
    parser.add_argument("--max-pixels-per-exposure", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")

    parser.add_argument(
        "--position-encoding", choices=("fourier", "spherical"), default="fourier"
    )
    parser.add_argument("--run-both-encodings", action="store_true")
    parser.add_argument("--spatial-frequencies", type=int, default=10)
    parser.add_argument("--spherical-min-frequency", type=float, default=1.0)
    parser.add_argument("--spherical-alignment-rank", type=int, default=8)
    parser.add_argument("--spectral-frequencies", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--trunk-depth", type=int, default=4)
    parser.add_argument("--decoder-depth", type=int, default=3)
    parser.add_argument(
        "--output-activation", choices=("linear", "softplus"), default="softplus"
    )
    parser.add_argument("--anchor-exposure", type=int, default=0)
    parser.add_argument(
        "--alignment-mode",
        choices=("original_wcs", "tweakreg_frozen", "trainable"),
        default="trainable",
        help=(
            "original_wcs freezes input WCS; tweakreg_frozen first runs JWST "
            "TweakReg and freezes the corrected WCS; trainable retains the "
            "legacy residual feature alignment."
        ),
    )
    parser.add_argument(
        "--tweakreg-fitgeometry",
        choices=("shift", "rshift", "rscale", "general"),
        default="rshift",
    )
    parser.add_argument("--tweakreg-snr-threshold", type=float, default=10.0)
    parser.add_argument("--tweakreg-minobj", type=int, default=15)
    parser.add_argument("--tweakreg-searchrad", type=float, default=2.0)
    parser.add_argument("--tweakreg-separation", type=float, default=1.0)
    parser.add_argument("--tweakreg-tolerance", type=float, default=0.7)
    parser.add_argument("--tweakreg-nclip", type=int, default=3)
    parser.add_argument("--tweakreg-sigma", type=float, default=3.0)

    # Historical auto-union path. These remain active when no template is supplied.
    parser.add_argument("--union-pixel-scale-arcsec", type=float, default=None)
    parser.add_argument("--union-padding-pixels", type=int, default=32)
    parser.add_argument(
        "--target-canvas-fits",
        type=Path,
        default=None,
        help=(
            "Optional 2D FITS template defining the common spatial WCS and exact "
            "output shape. Its image values are not used for training."
        ),
    )
    parser.add_argument(
        "--target-canvas-hdu",
        default=None,
        help="Optional HDU name/index; default is the first 2D image HDU.",
    )
    parser.add_argument(
        "--keep-outside-target-samples",
        action="store_true",
        help=(
            "Keep sampled native pixels outside a template canvas and let geometry "
            "mask them during the loss. Default behavior filters them before training."
        ),
    )

    parser.add_argument(
        "--bandpass-mode",
        choices=("official", "center", "top_hat", "gaussian"),
        default="official",
    )
    parser.add_argument("--bandpass-samples", type=int, default=9)
    parser.add_argument("--svo-collection", default="JWST/NIRCam2025")
    parser.add_argument(
        "--svo-cache-dir",
        type=Path,
        default=None,
        help="Local NPZ snapshots of SVO curves; defaults under OUTPUT_DIR.",
    )
    parser.add_argument("--svo-timeout", type=float, default=60.0)
    parser.add_argument("--svo-force-refresh", action="store_true")

    parser.add_argument(
        "--psf-mode",
        choices=("none", "gaussian", "npy", "stpsf"),
        default="stpsf",
    )
    parser.add_argument("--psf-size", type=int, default=51)
    parser.add_argument("--psf-fwhm-pix", default="")
    parser.add_argument("--default-psf-fwhm-pix", type=float, default=None)
    parser.add_argument("--psf-kernel-dir", type=Path, default=None)
    parser.add_argument("--psf-cache-dir", type=Path, default=None)
    parser.add_argument("--stpsf-extension", default="DET_DIST")
    parser.add_argument("--allow-static-stpsf-fallback", action="store_true")
    parser.add_argument("--psf-offset-chunk", type=int, default=128)
    parser.add_argument("--no-checkpoint-psf", action="store_true")

    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument(
        "--batch-size",
        type=int,
        nargs="+",
        default=[256],
        metavar="N",
        help=(
            "One integer keeps legacy total-batch behavior. Two integers are "
            "the exact short-wave and long-wave samples per exposure, e.g. "
            "--batch-size 100 400."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--alignment-learning-rate", type=float, default=2.0e-5)
    parser.add_argument("--background-learning-rate", type=float, default=5.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-6)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument(
        "--loss",
        "--robust-loss",
        dest="robust_loss",
        choices=("huber", "charbonnier", "whitened_l2"),
        default="huber",
        help="Training objective; --robust-loss remains as a compatibility alias.",
    )
    parser.add_argument("--robust-delta", type=float, default=0.1)
    parser.add_argument("--max-relative-weight", type=float, default=100.0)
    parser.add_argument("--alignment-regularization", type=float, default=1.0e-4)
    parser.add_argument("--background-regularization", type=float, default=1.0e-5)
    parser.add_argument("--no-background", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--render-tile-pixels", type=int, default=131072)
    parser.add_argument("--coverage-tile-rows", type=int, default=128)
    parser.add_argument("--skip-render", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    seed_everything(args.seed)
    bands = _parse_csv(args.bands)
    unsupported = sorted(set(bands).difference(BAND_SPECS_ANGSTROM))
    if unsupported:
        raise SystemExit("unsupported bands: " + ", ".join(unsupported))

    input_dir = args.exposure_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    svo_cache_dir = (
        args.svo_cache_dir.expanduser().resolve()
        if args.svo_cache_dir is not None
        else output_dir / "calibration_cache" / "svo"
    )
    psf_cache_dir = (
        args.psf_cache_dir.expanduser().resolve()
        if args.psf_cache_dir is not None
        else output_dir / "calibration_cache" / "psf"
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    selected = discover_exposures(
        input_dir,
        input_product=args.input_product,
        bands=bands,
        module=args.module,
        max_exposures_per_band=args.max_exposures_per_band,
    )
    alignment_report: dict[str, Any] = {
        "mode": args.alignment_mode,
        "fitgeometry": None,
        "products": [],
    }
    if args.alignment_mode == "tweakreg_frozen":
        selected, tweakreg_report = run_tweakreg_alignment(
            selected,
            output_dir=(
                output_dir
                / "alignment"
                / f"tweakreg_{args.tweakreg_fitgeometry}"
            ),
            fitgeometry=args.tweakreg_fitgeometry,
            snr_threshold=args.tweakreg_snr_threshold,
            minobj=args.tweakreg_minobj,
            searchrad=args.tweakreg_searchrad,
            separation=args.tweakreg_separation,
            tolerance=args.tweakreg_tolerance,
            nclip=args.tweakreg_nclip,
            sigma=args.tweakreg_sigma,
        )
        alignment_report.update(tweakreg_report)

    rng = np.random.default_rng(args.seed)
    pools = [
        load_exposure_pool(
            path,
            band,
            detector,
            max_pixels=args.max_pixels_per_exposure,
            dq_mode=args.dq_mode,
            rng=rng,
        )
        for path, band, detector in selected
    ]
    batch_values = (
        [args.batch_size] if isinstance(args.batch_size, int) else args.batch_size
    )
    batch_spec: int | list[int] = (
        batch_values[0] if len(batch_values) == 1 else batch_values
    )
    if len(batch_values) not in {1, 2}:
        raise SystemExit("batch-size accepts one total or two SW/LW integers")
    if min(batch_values) <= 0:
        raise SystemExit("batch-size values must be positive")
    if not 0 <= args.anchor_exposure < len(pools):
        raise SystemExit("anchor exposure is outside the selected exposure range")

    scale = robust_intensity_scale(pools)
    pools = normalize_pools(pools, scale)

    if args.target_canvas_fits is None:
        # Preserve the original automatic union-canvas path.
        padding = args.union_padding_pixels
        if args.psf_mode != "none":
            padding = max(padding, args.psf_size // 2 + 2)
        canvas = build_union_canvas(
            [pool.header for pool in pools],
            pixel_scale_arcsec=args.union_pixel_scale_arcsec,
            padding_pixels=padding,
        )
        target_canvas_report: dict[str, Any] = {
            "mode": "auto_union",
            "shape": list(canvas.shape),
            "pixel_scale_arcsec": canvas.pixel_scale_arcsec,
        }
        target_sample_coverage = sampled_pool_canvas_coverage(pools, canvas)
        target_sample_coverage["filter_applied"] = False
    else:
        canvas, target_canvas_report = load_target_canvas_from_fits(
            args.target_canvas_fits,
            hdu_selector=args.target_canvas_hdu,
        )
        if args.keep_outside_target_samples:
            target_sample_coverage = sampled_pool_canvas_coverage(pools, canvas)
            target_sample_coverage["filter_applied"] = False
        else:
            pools, target_sample_coverage = filter_sampled_pools_to_canvas(pools, canvas)
        print(
            json.dumps(
                {
                    "target_canvas": target_canvas_report,
                    "sampled_pixel_coverage": target_sample_coverage,
                },
                indent=2,
            ),
            flush=True,
        )

    actual_batch_size = effective_batch_size(pools, batch_spec)
    if len(batch_values) == 1 and actual_batch_size < len(pools):
        raise SystemExit(
            f"batch-size {actual_batch_size} is smaller than {len(pools)} exposures"
        )

    calibration_inputs = [pool.calibration_input() for pool in pools]
    bandpass = build_bandpass_calibration(
        calibration_inputs,
        mode=args.bandpass_mode,
        num_samples=args.bandpass_samples,
        band_specs=BAND_SPECS_ANGSTROM,
        svo_collection=args.svo_collection,
        svo_cache_dir=svo_cache_dir,
        svo_timeout=args.svo_timeout,
        svo_force_refresh=args.svo_force_refresh,
    )
    psf_calibration = build_psf_calibration(
        calibration_inputs,
        bandpass.wavelength,
        mode=args.psf_mode,
        kernel_size=args.psf_size,
        gaussian_fwhm_by_band=_parse_band_float_map(args.psf_fwhm_pix),
        default_gaussian_fwhm=args.default_psf_fwhm_pix,
        kernel_dir=(
            args.psf_kernel_dir.expanduser().resolve()
            if args.psf_kernel_dir is not None
            else None
        ),
        cache_dir=psf_cache_dir,
        stpsf_extension=args.stpsf_extension,
        allow_static_stpsf_fallback=args.allow_static_stpsf_fallback,
    )

    manifest = {
        "exposures": [
            {
                "index": index,
                "path": str(pool.path),
                "band": pool.band,
                "detector": pool.detector,
                "shape": list(pool.shape),
                "sampled_pixels": pool.size,
            }
            for index, pool in enumerate(pools)
        ],
        "intensity_scale": scale,
        "union_canvas": {
            "shape": list(canvas.shape),
            "pixel_scale_arcsec": canvas.pixel_scale_arcsec,
            "header": canvas.header,
        },
        "target_canvas": target_canvas_report,
        "target_canvas_sample_coverage": target_sample_coverage,
        "bandpass": bandpass.metadata,
        "psf": psf_calibration.metadata,
        "alignment": alignment_report,
        "effective_total_batch_size": actual_batch_size,
        "args": _jsonable_args(args),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    coverage: torch.Tensor | None = None
    if not args.skip_render:
        coverage = build_union_coverage_mask(
            [pool.header for pool in pools],
            canvas,
            device=device,
            tile_rows=args.coverage_tile_rows,
        )
        torchsky.write(
            str(output_dir / "union_coverage_mask.fits"),
            coverage.to(torch.uint8).cpu(),
            header=canvas.header,
            overwrite=True,
        )

    encodings = (
        ["fourier", "spherical"]
        if args.run_both_encodings
        else [args.position_encoding]
    )
    summaries: dict[str, Any] = {}
    checkpoint_args = _jsonable_args(args)
    checkpoint_args["target_canvas"] = target_canvas_report
    checkpoint_args["target_canvas_sample_coverage"] = target_sample_coverage

    for encoding in encodings:
        run_dir = output_dir / encoding
        run_dir.mkdir(parents=True, exist_ok=True)
        model = WCSAwareMultiWavelengthINR(
            [pool.header for pool in pools],
            canvas,
            encoding=encoding,
            spatial_frequencies=args.spatial_frequencies,
            spherical_min_frequency=args.spherical_min_frequency,
            spherical_alignment_rank=args.spherical_alignment_rank,
            spectral_frequencies=args.spectral_frequencies,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            trunk_depth=args.trunk_depth,
            decoder_depth=args.decoder_depth,
            anchor_exposure=args.anchor_exposure,
            alignment_mode=args.alignment_mode,
            learn_background=not args.no_background,
            output_activation=args.output_activation,
        )
        history = train_model(
            model,
            pools,
            bandpass.wavelength,
            bandpass.weight,
            psf=psf_calibration.bank,
            device=device,
            steps=args.steps,
            batch_size=batch_spec,
            seed=args.seed,
            learning_rate=args.learning_rate,
            alignment_learning_rate=args.alignment_learning_rate,
            background_learning_rate=args.background_learning_rate,
            weight_decay=args.weight_decay,
            gradient_clip=args.gradient_clip,
            robust_loss=args.robust_loss,
            robust_delta=args.robust_delta,
            max_relative_weight=args.max_relative_weight,
            alignment_regularization=args.alignment_regularization,
            background_regularization=args.background_regularization,
            psf_offset_chunk=args.psf_offset_chunk,
            checkpoint_psf_chunks=not args.no_checkpoint_psf,
            amp=not args.no_amp,
            log_every=args.log_every,
            checkpoint_every=args.checkpoint_every,
            checkpoint_path=run_dir / "checkpoint.pt",
            loss_log_path=run_dir / "training_loss.jsonl",
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
                    bandpass.wavelength,
                    bandpass.weight,
                    representative,
                    anchor_exposure=args.anchor_exposure,
                    device=device,
                    tile_pixels=args.render_tile_pixels,
                )
                image = torch.where(coverage, image * scale, torch.zeros_like(image))
                path = run_dir / f"{band}_{encoding}_union_scene.fits"
                torchsky.write(
                    str(path), image.cpu(), header=canvas.header, overwrite=True
                )
                outputs[band] = str(path)
        summary = {
            "encoding": encoding,
            "checkpoint": str(run_dir / "checkpoint.pt"),
            "outputs": outputs,
            "target_canvas": target_canvas_report,
            "target_canvas_sample_coverage": target_sample_coverage,
            "alignment": model.alignment_summary(),
            "background_physical": (model.background.detach().cpu() * scale).tolist(),
            "training": [asdict(record) for record in history],
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        summaries[encoding] = summary
        if device.type == "cuda":
            torch.cuda.empty_cache()
    (output_dir / "all_runs_summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
