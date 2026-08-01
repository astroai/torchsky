#!/usr/bin/env python3
"""Render faint-source pipeline products from a completed checkpoint only.

This script never trains.  It reconstructs the exact model and exposure geometry
stored in a stage-1 or stage-2 faint-source checkpoint, renders each band's
bandpass-integrated intrinsic field on the UPDC grid, and optionally applies the
same effective post-bandpass PSF convolution used by the training pipeline.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Sequence

import numpy as np
import torch
from astropy.io import fits
from astropy.wcs import WCS as AstropyWCS
from astropy.wcs.utils import proj_plane_pixel_scales

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for search_path in (ROOT, SRC):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import scripts._smacs0723_wcs_aware_faint_source_pipeline_core as core  # noqa: E402
import torchsky.map.wcs_aware_training as training_mod  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Post-processing output directory. Default: "
            "<checkpoint run>/postprocess_<stage>_step_<global_step>."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--render-tile-pixels", type=int, default=None)

    parser.add_argument("--updc-dir", type=Path, default=None)
    parser.add_argument("--updc-pattern", default=None)
    parser.add_argument(
        "--updc-reference",
        action="append",
        default=[],
        metavar="BAND=FITS",
        help="Override one UPDC reference; may be repeated.",
    )
    parser.add_argument(
        "--updc-zero-is-invalid",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    parser.add_argument(
        "--final-psf-convolution",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Defaults to the value saved in the checkpoint configuration.",
    )
    parser.add_argument("--final-psf-minimum-support", type=float, default=None)
    parser.add_argument("--psf-cache-dir", type=Path, default=None)
    parser.add_argument("--stpsf-data-dir", type=Path, default=None)
    return parser


def load_checkpoint(path: Path) -> tuple[Path, dict[str, Any]]:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    try:
        payload = torch.load(resolved, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(resolved, map_location="cpu")
    if payload.get("format") != "torchsky-smacs0723-faint-source-pipeline-v1":
        raise ValueError(f"unsupported checkpoint format in {resolved}")
    required = {
        "model",
        "intensity_scale",
        "union_header",
        "union_shape",
        "exposure_paths",
        "exposure_bands",
        "bandpass_wavelength",
        "bandpass_weight",
        "configuration",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError("checkpoint is missing: " + ", ".join(missing))
    return resolved, payload


def clean_header_for_wcs(header: fits.Header | dict[str, Any]) -> fits.Header:
    """Make dict-serialized JWST headers safe for AstropyWCS construction."""

    clean = fits.Header()
    for raw_key, raw_value in header.items():
        key = str(raw_key).strip().upper()
        if not key or key in {"COMMENT", "HISTORY"}:
            continue
        value = raw_value
        if isinstance(value, str):
            value = "".join(
                character if 32 <= ord(character) <= 126 else " "
                for character in value
            )
        try:
            clean[key] = value
        except (TypeError, ValueError):
            continue
    return clean


def safe_pixel_scale_arcsec(header: fits.Header | dict[str, Any]) -> float:
    scales = proj_plane_pixel_scales(AstropyWCS(clean_header_for_wcs(header)).celestial)[:2]
    value = float(np.sqrt(np.prod(scales)) * 3600.0)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("could not infer a positive pixel scale")
    return value


def lightweight_pools(
    checkpoint: dict[str, Any],
) -> list[training_mod.ExposurePool]:
    configuration = checkpoint["configuration"]
    exposure_rows = configuration.get("exposures")
    if not isinstance(exposure_rows, list):
        raise ValueError("checkpoint configuration has no exposure metadata")
    rows = sorted(exposure_rows, key=lambda row: int(row["index"]))
    paths = [Path(value).expanduser().resolve() for value in checkpoint["exposure_paths"]]
    bands = [str(value) for value in checkpoint["exposure_bands"]]
    if len(rows) != len(paths) or len(paths) != len(bands):
        raise ValueError("checkpoint exposure metadata lengths differ")

    pools: list[training_mod.ExposurePool] = []
    empty_uv = torch.empty((0, 2), dtype=torch.float32)
    empty_value = torch.empty(0, dtype=torch.float32)
    for expected, (row, path, band) in enumerate(zip(rows, paths, bands, strict=True)):
        if int(row["index"]) != expected:
            raise ValueError("checkpoint exposure indices are not contiguous")
        if str(row["path"]) != str(path):
            raise ValueError(f"exposure path mismatch at index {expected}")
        if str(row["band"]) != band:
            raise ValueError(f"exposure band mismatch at index {expected}")
        if not path.exists():
            raise FileNotFoundError(path)
        with fits.open(path, memmap=True) as hdul:
            if "SCI" not in hdul or hdul["SCI"].data is None:
                raise ValueError(f"{path} has no SCI image")
            shape = tuple(int(value) for value in hdul["SCI"].data.shape)
            if len(shape) != 2:
                raise ValueError(f"{path} SCI is not two-dimensional")
            header = dict(hdul["SCI"].header)
            header["NAXIS1"] = shape[1]
            header["NAXIS2"] = shape[0]
        pools.append(
            training_mod.ExposurePool(
                path=path,
                band=band,
                detector=str(row["detector"]),
                header=header,
                shape=(shape[0], shape[1]),
                uv=empty_uv,
                value=empty_value,
                err=empty_value,
            )
        )
    return pools


def resolve_references(
    cli: argparse.Namespace,
    saved_args: dict[str, Any],
) -> dict[str, Path]:
    saved_overrides = core.parse_band_path_map(saved_args.get("updc_reference", []))
    cli_overrides = core.parse_band_path_map(cli.updc_reference)
    overrides = {**saved_overrides, **cli_overrides}
    root = (
        cli.updc_dir.expanduser().resolve()
        if cli.updc_dir is not None
        else Path(saved_args.get("updc_dir", core.DEFAULT_UPDC_DIR)).expanduser().resolve()
    )
    pattern = str(
        cli.updc_pattern
        if cli.updc_pattern is not None
        else saved_args.get("updc_pattern", core.DEFAULT_UPDC_PATTERN)
    )
    references = {
        band: overrides.get(band, root / pattern.format(band=band)).resolve()
        for band in core.BANDS
    }
    missing = [str(path) for path in references.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing UPDC references: " + ", ".join(missing))
    return references


def prepare_output_directory(path: Path, *, overwrite: bool) -> dict[str, Path]:
    root = path.expanduser().resolve()
    products = root / "products"
    manifest = root / "postprocess_manifest.json"
    summary = root / "postprocess_summary.json"
    failure = root / "postprocess_failure.json"
    collisions = [item for item in (products, manifest, summary, failure) if item.exists()]
    if collisions and not overwrite:
        raise FileExistsError(
            "refusing to overwrite post-processing products: "
            + ", ".join(item.name for item in collisions)
        )
    root.mkdir(parents=True, exist_ok=True)
    if overwrite:
        if products.exists():
            shutil.rmtree(products)
        for item in (manifest, summary, failure):
            item.unlink(missing_ok=True)
    products.mkdir(parents=True, exist_ok=True)
    return {
        "root": root,
        "products": products,
        "manifest": manifest,
        "summary": summary,
        "failure": failure,
    }


def saved_or_default(saved: dict[str, Any], key: str, default: Any) -> Any:
    value = saved.get(key, default)
    return default if value is None else value


def main() -> int:
    args = build_parser().parse_args()
    checkpoint_path, checkpoint = load_checkpoint(args.checkpoint)
    configuration = checkpoint["configuration"]
    saved_args = dict(configuration.get("arguments", {}))
    stage = str(checkpoint.get("stage", "checkpoint"))
    global_step = int(checkpoint.get("global_step", 0))
    default_output = (
        checkpoint_path.parent.parent
        / f"postprocess_{stage}_step_{global_step:06d}"
    )
    paths = prepare_output_directory(
        args.output_dir if args.output_dir is not None else default_output,
        overwrite=args.overwrite,
    )

    try:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")

        baseline_path = Path(configuration["baseline_manifest"]).expanduser().resolve()
        baseline_path, baseline = core.load_manifest(baseline_path)
        core.validate_baseline(baseline)
        baseline_args = baseline["args"]

        pools = lightweight_pools(checkpoint)
        intensity_scale = float(checkpoint["intensity_scale"])
        if not np.isfinite(intensity_scale) or intensity_scale <= 0:
            raise ValueError("checkpoint intensity scale must be finite and positive")

        wavelength = torch.as_tensor(
            checkpoint["bandpass_wavelength"], dtype=torch.float32
        ).cpu()
        weight = torch.as_tensor(
            checkpoint["bandpass_weight"], dtype=torch.float32
        ).cpu()
        if wavelength.ndim != 2 or weight.shape != wavelength.shape:
            raise ValueError("invalid checkpoint bandpass tensor shapes")
        active_wavelength = wavelength[weight > 0]
        wavelength_min = float(active_wavelength.min())
        wavelength_max = float(active_wavelength.max())

        union_shape = tuple(int(value) for value in checkpoint["union_shape"])
        union_info = configuration.get("union_canvas", {})
        canvas = core.UnionCanvas(
            header=dict(checkpoint["union_header"]),
            shape=(union_shape[0], union_shape[1]),
            pixel_scale_arcsec=float(
                union_info.get(
                    "pixel_scale_arcsec",
                    safe_pixel_scale_arcsec(checkpoint["union_header"]),
                )
            ),
        )

        architecture = str(configuration["architecture"])
        model = core.WCSAwareLiteratureINR(
            [pool.header for pool in pools],
            canvas,
            architecture=architecture,
            wavelength_min=wavelength_min,
            wavelength_max=wavelength_max,
            anchor_exposure=int(baseline_args.get("anchor_exposure", 0)),
            alignment_mode=str(baseline_args["alignment_mode"]),
            learn_background=not bool(baseline_args.get("no_background", False)),
            output_activation=str(baseline_args.get("output_activation", "softplus")),
            nerf_num_frequencies=int(saved_or_default(saved_args, "nerf_num_frequencies", 10)),
            nerf_hidden_dim=int(saved_or_default(saved_args, "nerf_hidden_dim", 256)),
            nerf_depth=int(saved_or_default(saved_args, "nerf_depth", 8)),
            nerf_skip_after_layer=int(
                saved_or_default(saved_args, "nerf_skip_after_layer", 4)
            ),
            siren_hidden_dim=int(saved_or_default(saved_args, "siren_hidden_dim", 256)),
            siren_hidden_layers=int(
                saved_or_default(saved_args, "siren_hidden_layers", 3)
            ),
            siren_first_omega_0=float(
                saved_or_default(saved_args, "siren_first_omega0", 30.0)
            ),
            siren_hidden_omega_0=float(
                saved_or_default(saved_args, "siren_hidden_omega0", 30.0)
            ),
        )
        model.load_state_dict(checkpoint["model"], strict=True)
        model.to(device)
        model.eval()

        references = resolve_references(args, saved_args)
        render_tile_pixels = int(
            args.render_tile_pixels
            if args.render_tile_pixels is not None
            else saved_or_default(saved_args, "render_tile_pixels", 65536)
        )
        if render_tile_pixels <= 0:
            raise ValueError("render-tile-pixels must be positive")

        final_psf = bool(
            args.final_psf_convolution
            if args.final_psf_convolution is not None
            else saved_or_default(saved_args, "final_psf_convolution", True)
        )
        minimum_support = float(
            args.final_psf_minimum_support
            if args.final_psf_minimum_support is not None
            else saved_or_default(saved_args, "final_psf_minimum_support", 0.999)
        )
        zero_is_invalid = bool(
            args.updc_zero_is_invalid
            if args.updc_zero_is_invalid is not None
            else saved_or_default(saved_args, "updc_zero_is_invalid", True)
        )

        # Serialized JWST COMMENT/HISTORY cards may contain embedded newlines.
        # Only pixel-scale parsing needs this compatibility wrapper.
        core.pixel_scale_arcsec = safe_pixel_scale_arcsec

        psf_calibration = None
        psf_settings: dict[str, Any] | None = None
        if final_psf:
            psf_mode = str(saved_or_default(saved_args, "psf_mode", "stpsf"))
            source_output = Path(
                saved_args.get("output_dir", checkpoint_path.parent.parent)
            ).expanduser().resolve()
            cache_dir = (
                args.psf_cache_dir.expanduser().resolve()
                if args.psf_cache_dir is not None
                else (
                    Path(saved_args["psf_cache_dir"]).expanduser().resolve()
                    if saved_args.get("psf_cache_dir")
                    else source_output / "calibration_cache" / "psf"
                )
            )
            stpsf_dir = (
                args.stpsf_data_dir.expanduser().resolve()
                if args.stpsf_data_dir is not None
                else Path(
                    saved_or_default(
                        saved_args,
                        "stpsf_data_dir",
                        core.DEFAULT_STPSF_DATA_DIR,
                    )
                ).expanduser().resolve()
            )
            if psf_mode == "stpsf":
                if not stpsf_dir.exists():
                    raise FileNotFoundError(stpsf_dir)
                os.environ["STPSF_PATH"] = str(stpsf_dir)

            kernel_dir_value = saved_args.get("psf_kernel_dir")
            kernel_dir = (
                None
                if not kernel_dir_value
                else Path(kernel_dir_value).expanduser().resolve()
            )
            psf_calibration = core.build_psf_calibration(
                [pool.calibration_input() for pool in pools],
                wavelength,
                mode=psf_mode,
                kernel_size=int(saved_or_default(saved_args, "psf_size", 15)),
                gaussian_fwhm_by_band=core.parse_band_float_map(
                    str(saved_or_default(saved_args, "psf_fwhm_pix", ""))
                ),
                default_gaussian_fwhm=saved_args.get("default_psf_fwhm_pix"),
                kernel_dir=kernel_dir,
                cache_dir=cache_dir,
                stpsf_extension=str(
                    saved_or_default(saved_args, "stpsf_extension", "DET_DIST")
                ),
                allow_static_stpsf_fallback=bool(
                    saved_or_default(
                        saved_args,
                        "allow_static_stpsf_fallback",
                        False,
                    )
                ),
            )
            if psf_calibration.bank is None:
                raise RuntimeError("PSF calibration produced no kernel bank")
            psf_settings = {
                "mode": psf_mode,
                "cache_dir": str(cache_dir),
                "kernel_dir": None if kernel_dir is None else str(kernel_dir),
                "stpsf_data_dir": str(stpsf_dir),
                "metadata": psf_calibration.metadata,
            }

        manifest = {
            "created_utc": core.now_utc(),
            "mode": "postprocess_checkpoint_only",
            "training_performed": False,
            "checkpoint": str(checkpoint_path),
            "checkpoint_stage": stage,
            "checkpoint_stage_step": int(checkpoint.get("stage_step", 0)),
            "checkpoint_global_step": global_step,
            "checkpoint_saved_utc": checkpoint.get("saved_utc"),
            "baseline_manifest": str(baseline_path),
            "architecture": architecture,
            "device": str(device),
            "output_dir": str(paths["root"]),
            "render_tile_pixels": render_tile_pixels,
            "final_psf_convolution": final_psf,
            "final_psf_minimum_support": minimum_support,
            "psf": psf_settings,
            "references": {band: str(path) for band, path in references.items()},
            "intensity_scale": intensity_scale,
            "bands": list(core.BANDS),
        }
        core.write_json(paths["manifest"], manifest)

        products: dict[str, Any] = {}
        with torch.inference_mode():
            for band in core.BANDS:
                reference, header, reference_valid, reference_info = core.read_reference(
                    references[band], zero_is_invalid=zero_is_invalid
                )
                representative, wave, band_weight, bandpass_info = core.bandpass_for_band(
                    pools, wavelength, weight, band
                )
                band_dir = paths["products"] / band
                band_dir.mkdir(parents=True, exist_ok=True)

                intrinsic, intrinsic_valid = core.render_intrinsic_bandpass_field(
                    model=model,
                    canvas=canvas,
                    template_header=header,
                    representative=representative,
                    wavelength=wave,
                    weight=band_weight,
                    intensity_scale=intensity_scale,
                    device=device,
                    tile_pixels=render_tile_pixels,
                )
                intrinsic_path = band_dir / f"{band}_intrinsic_bandpass_field.fits"
                core.write_image(
                    intrinsic_path,
                    intrinsic,
                    header,
                    band=band,
                    stage="INTRINSIC_BANDPASS_FIELD",
                    overwrite=args.overwrite,
                )

                effective_psf_path = None
                if final_psf:
                    assert psf_calibration is not None
                    effective_psf, effective_psf_info = core.effective_post_bandpass_psf(
                        pools=pools,
                        psf_bank=psf_calibration.bank,
                        wavelength_weight=weight,
                        band=band,
                        target_header=header,
                    )
                    final_field, final_valid = core.convolve_field_with_psf(
                        intrinsic,
                        effective_psf,
                        minimum_support=minimum_support,
                    )
                    final_stage = "POST_BANDPASS_EFFECTIVE_PSF_FIELD"
                    effective_psf_path = (
                        band_dir / f"{band}_effective_post_bandpass_psf.fits"
                    )
                    fits.PrimaryHDU(effective_psf.astype(np.float32)).writeto(
                        effective_psf_path,
                        overwrite=args.overwrite,
                        checksum=True,
                    )
                else:
                    final_field = intrinsic.copy()
                    final_valid = intrinsic_valid.copy()
                    final_stage = "INTRINSIC_BANDPASS_FIELD"
                    effective_psf_info = None

                final_path = band_dir / f"{band}_final_detector_input.fits"
                core.write_image(
                    final_path,
                    final_field,
                    header,
                    band=band,
                    stage=final_stage,
                    overwrite=args.overwrite,
                )
                common_valid = (
                    final_valid
                    & reference_valid
                    & np.isfinite(final_field)
                    & np.isfinite(reference)
                )
                common_mask_path = (
                    band_dir / f"{band}_model_UPDC_common_valid_mask.fits"
                )
                core.write_image(
                    common_mask_path,
                    common_valid.astype(np.uint8),
                    header,
                    band=band,
                    stage="MODEL_UPDC_COMMON_VALID_MASK",
                    overwrite=args.overwrite,
                )

                products[band] = {
                    "reference": reference_info,
                    "bandpass": bandpass_info,
                    "intrinsic_bandpass_field": str(intrinsic_path),
                    "final_detector_input": str(final_path),
                    "final_psf_convolution_enabled": final_psf,
                    "effective_post_bandpass_psf": (
                        None if effective_psf_path is None else str(effective_psf_path)
                    ),
                    "effective_post_bandpass_psf_metadata": effective_psf_info,
                    "common_valid_mask": str(common_mask_path),
                    "common_valid_pixels": int(common_valid.sum()),
                }
                core.write_json(
                    band_dir / f"{band}_product_manifest.json",
                    products[band],
                )
                print(
                    json.dumps(
                        {
                            "band": band,
                            "intrinsic": str(intrinsic_path),
                            "final": str(final_path),
                            "common_valid_pixels": int(common_valid.sum()),
                        }
                    ),
                    flush=True,
                )

                del reference, reference_valid, intrinsic, intrinsic_valid
                del final_field, final_valid, common_valid
                gc.collect()

        summary = {
            "status": "complete",
            "completed_utc": core.now_utc(),
            "training_performed": False,
            "checkpoint": str(checkpoint_path),
            "checkpoint_stage": stage,
            "checkpoint_global_step": global_step,
            "manifest": str(paths["manifest"]),
            "final_psf_convolution": {
                "enabled": final_psf,
                "operator_order": (
                    "effective_PSF * final_intrinsic_bandpass_field"
                    if final_psf
                    else "no post-training PSF convolution"
                ),
            },
            "products": products,
            "alignment": model.alignment_summary(),
            "background_physical": (
                model.background.detach().cpu() * intensity_scale
            ).tolist(),
        }
        core.write_json(paths["summary"], summary)
        print(
            json.dumps(
                {
                    "status": "complete",
                    "checkpoint": str(checkpoint_path),
                    "manifest": str(paths["manifest"]),
                    "summary": str(paths["summary"]),
                    "products": str(paths["products"]),
                },
                indent=2,
            ),
            flush=True,
        )
        return 0
    except BaseException as exc:
        core.write_json(
            paths["failure"],
            {
                "status": "failed",
                "failed_utc": core.now_utc(),
                "checkpoint": str(checkpoint_path),
                "error": repr(exc),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
