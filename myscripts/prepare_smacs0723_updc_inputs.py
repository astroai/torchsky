#!/usr/bin/env python3
"""Prepare corrected SMACS J0723 exposures for UPDC.

Stages:

1. Apply Chris Willott's public ``image1overf.sub1fimaging`` correction.
2. Align exposures on a temporary native-scale sky grid.
3. Detect high outliers in two 5-sigma passes.
4. Map the fitted sky statistics back to each detector and replace bad pixels
   with deterministic Gaussian draws.

The paper's original code uses exact footprint voting for step 3 and was not
released. This implementation uses TorchSky's WCS projection and adjoint-ready
target-to-detector operator. It preserves the published two-pass thresholds and
replacement distribution, while recording every replaced detector pixel in a
new ``UPDCMASK`` FITS extension.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from astropy.io import fits


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
BENCH = ROOT / "benchmarks"
for path in (SRC, BENCH):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bench_map_smacs0723_real import _discover_exposure_paths  # noqa: E402
from torchsky.map import ImageObservation, TargetRenderSpec, coadd_image_plane  # noqa: E402
from torchsky.map.operators import bind_image_plane_target_operator  # noqa: E402


@dataclass
class ExposureState:
    path: Path
    data: np.ndarray
    header: fits.Header
    dq: np.ndarray
    valid: np.ndarray
    rejected: np.ndarray


def _load_image1overf(path: Path):
    spec = importlib.util.spec_from_file_location("updc_image1overf", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "sub1fimaging"):
        raise AttributeError(f"{path} does not define sub1fimaging")
    return module.sub1fimaging


def _apply_one_over_f(
    input_path: Path,
    output_path: Path,
    *,
    correction_fn,
    sigma_bgmask: float,
    sigma_1fmask: float,
    splitamps: bool,
    usesegmask: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with fits.open(input_path, memmap=False) as hdul:
        corrected = correction_fn(
            hdul,
            float(sigma_bgmask),
            float(sigma_1fmask),
            bool(splitamps),
            bool(usesegmask),
        )
        subarray = str(hdul[0].header.get("SUBARRAY", "")).upper()
        if subarray == "FULL":
            hdul["SCI"].data[4:2044, 4:2044] = corrected
        elif subarray == "SUB256":
            hdul["SCI"].data[:252, :252] = corrected
        else:
            raise ValueError(
                f"{input_path}: unsupported SUBARRAY={subarray!r} for image1overf"
            )
        hdul[0].header.add_history(
            "UPDC reproduction: image1overf correction applied"
        )
        hdul.writeto(output_path, overwrite=True)


def _downsample_header(
    header: fits.Header,
    *,
    factor: int,
) -> dict[str, Any]:
    if factor < 1:
        raise ValueError("factor must be positive")
    out = header.copy()
    out["NAXIS1"] = int(np.ceil(int(header["NAXIS1"]) / factor))
    out["NAXIS2"] = int(np.ceil(int(header["NAXIS2"]) / factor))
    out["CRPIX1"] = (float(header["CRPIX1"]) - 0.5) / factor + 0.5
    out["CRPIX2"] = (float(header["CRPIX2"]) - 0.5) / factor + 0.5
    for key in ("CD1_1", "CD1_2", "CD2_1", "CD2_2", "CDELT1", "CDELT2"):
        if key in out:
            out[key] = float(out[key]) * factor
    return dict(out)


def _read_state(path: Path) -> ExposureState:
    # JWST DQ arrays use FITS unsigned-integer scaling (BZERO/BSCALE), which
    # Astropy cannot materialize from a memory-mapped image.
    with fits.open(path, memmap=False) as hdul:
        data = np.asarray(hdul["SCI"].data, dtype=np.float32).copy()
        header = hdul["SCI"].header.copy()
        dq = np.asarray(hdul["DQ"].data).copy()
    valid = np.isfinite(data) & (dq == 0)
    return ExposureState(
        path=path,
        data=data,
        header=header,
        dq=dq,
        valid=valid,
        rejected=np.zeros_like(valid),
    )


def _write_mask(path: Path, mask: np.ndarray, header: fits.Header) -> None:
    fits.PrimaryHDU(
        data=mask.astype(np.float32),
        header=header,
    ).writeto(path, overwrite=True)


def _aligned_stack(
    states: list[ExposureState],
    *,
    target_header: dict[str, Any],
    tmpdir: Path,
    device: torch.device,
    pass_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    aligned: list[np.ndarray] = []
    supports: list[np.ndarray] = []
    for index, state in enumerate(states):
        usable = state.valid & ~state.rejected
        mask_path = tmpdir / f"pass{pass_index}_mask_{index:03d}.fits"
        _write_mask(mask_path, usable, state.header)
        result = coadd_image_plane(
            [
                ImageObservation(
                    str(state.path),
                    hdu="SCI",
                    mask_path=str(mask_path),
                    mask_hdu=0,
                )
            ],
            target_header=target_header,
            resampling_kernel="bilinear",
            value_semantics="surface_brightness",
            stacking_method="weighted_mean",
            surface_brightness_area_scaling=False,
            device=device,
        )
        array = result.data.detach().cpu().numpy().astype(np.float32)
        support = result.weight.detach().cpu().numpy() > 0
        aligned.append(np.where(support, array, np.nan))
        supports.append(support)
    return np.stack(aligned, axis=0), np.stack(supports, axis=0)


def _fit_gaussian_maps(
    aligned: np.ndarray,
    *,
    minimum_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite = np.isfinite(aligned)
    count = finite.sum(axis=0)
    # For a Gaussian model, the MLE location and scale are sample mean/std.
    location = np.nanmean(aligned, axis=0).astype(np.float32)
    scale = np.nanstd(aligned, axis=0).astype(np.float32)
    usable = (
        (count >= int(minimum_samples))
        & np.isfinite(location)
        & np.isfinite(scale)
        & (scale > 0)
    )
    location = np.where(usable, location, 0.0).astype(np.float32)
    scale = np.where(usable, scale, 0.0).astype(np.float32)
    return location, scale, usable


def _project_statistic_to_detector(
    statistic: np.ndarray,
    *,
    target: TargetRenderSpec,
    state: ExposureState,
    device: torch.device,
) -> np.ndarray:
    geometry = bind_image_plane_target_operator(
        ImageObservation(str(state.path), hdu="SCI"),
        target,
        observation_header=dict(state.header),
        resampling_kernel="bilinear",
        value_semantics="surface_brightness",
    )
    tensor = torch.as_tensor(statistic, dtype=torch.float32, device=device)
    return geometry.apply(tensor).detach().cpu().numpy().astype(np.float32)


def _detect_pass(
    states: list[ExposureState],
    *,
    target: TargetRenderSpec,
    location: np.ndarray,
    scale: np.ndarray,
    usable_sky: np.ndarray,
    sigma: float,
    device: torch.device,
) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    for state in states:
        detector_location = _project_statistic_to_detector(
            location,
            target=target,
            state=state,
            device=device,
        )
        detector_scale = np.abs(
            _project_statistic_to_detector(
                scale,
                target=target,
                state=state,
                device=device,
            )
        )
        detector_support = (
            _project_statistic_to_detector(
                usable_sky.astype(np.float32),
                target=target,
                state=state,
                device=device,
            )
            > 0.999
        )
        high = (
            detector_support
            & np.isfinite(state.data)
            & np.isfinite(detector_location)
            & np.isfinite(detector_scale)
            & (detector_scale > 0)
            & (state.data > detector_location + float(sigma) * detector_scale)
        )
        newly_rejected = high & ~state.rejected
        state.rejected |= high
        rows.append(
            {
                "new_high_outliers": int(newly_rejected.sum()),
                "total_rejected": int(state.rejected.sum()),
            }
        )
    return rows


def _replace_and_write(
    states: list[ExposureState],
    *,
    target: TargetRenderSpec,
    location: np.ndarray,
    scale: np.ndarray,
    usable_sky: np.ndarray,
    output_dir: Path,
    seed: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index, state in enumerate(states):
        detector_location = _project_statistic_to_detector(
            location,
            target=target,
            state=state,
            device=device,
        )
        detector_scale = np.abs(
            _project_statistic_to_detector(
                scale,
                target=target,
                state=state,
                device=device,
            )
        )
        detector_support = (
            _project_statistic_to_detector(
                usable_sky.astype(np.float32),
                target=target,
                state=state,
                device=device,
            )
            > 0.999
        )
        replace = detector_support & (
            state.rejected | ~np.isfinite(state.data) | (state.dq != 0)
        )
        rng = np.random.default_rng(int(seed) + index)
        draws = rng.normal(detector_location, detector_scale).astype(np.float32)
        corrected = state.data.copy()
        corrected[replace] = draws[replace]

        name = state.path.name
        if name.endswith("_cal_1overf.fits"):
            name = name.replace(
                "_cal_1overf.fits",
                "_cal_1overf_completed.fits",
            )
        else:
            name = f"{state.path.stem}_completed.fits"
        output_path = output_dir / name
        with fits.open(state.path, memmap=False) as hdul:
            hdul["SCI"].data[...] = corrected
            if "UPDCMASK" in hdul:
                del hdul["UPDCMASK"]
            hdul.append(
                fits.ImageHDU(
                    data=replace.astype(np.uint8),
                    name="UPDCMASK",
                )
            )
            hdul[0].header.add_history(
                "UPDC reproduction: two-pass abnormal-pixel completion"
            )
            hdul[0].header["UPDCSEED"] = (
                int(seed) + index,
                "Random seed used for Gaussian pixel completion",
            )
            hdul.writeto(output_path, overwrite=True)
        rows.append(
            {
                "input": str(state.path),
                "output": str(output_path),
                "replaced_pixels": int(replace.sum()),
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-reference", type=Path, required=True)
    parser.add_argument("--image1overf", type=Path, required=True)
    parser.add_argument("--band", default="F277W")
    parser.add_argument("--module", default="b")
    parser.add_argument("--expected-exposures", type=int, default=9)
    parser.add_argument("--sigma-bgmask", type=float, default=3.0)
    parser.add_argument("--sigma-1fmask", type=float, default=2.0)
    parser.add_argument("--splitamps", action="store_true")
    parser.add_argument(
        "--no-segmask",
        action="store_true",
        help="Disable the source segmentation mask in image1overf.",
    )
    parser.add_argument("--outlier-sigma", type=float, default=5.0)
    parser.add_argument("--minimum-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2736)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-one-over-f", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    exposure_paths = _discover_exposure_paths(
        args.input_dir,
        input_product="cal",
        band=str(args.band).upper(),
        module=str(args.module).lower(),
    )
    if args.expected_exposures and len(exposure_paths) != int(
        args.expected_exposures
    ):
        raise RuntimeError(
            f"expected {args.expected_exposures} exposures, "
            f"found {len(exposure_paths)}"
        )

    one_over_f_dir = args.output_dir / "one_over_f"
    completed_dir = args.output_dir / "completed"
    one_over_f_dir.mkdir(parents=True, exist_ok=True)
    correction_fn = None
    if not args.skip_one_over_f:
        correction_fn = _load_image1overf(args.image1overf)

    corrected_paths: list[Path] = []
    for input_path in exposure_paths:
        output_path = one_over_f_dir / input_path.name.replace(
            "_cal.fits",
            "_cal_1overf.fits",
        )
        if args.skip_one_over_f:
            output_path = input_path
        elif output_path.exists():
            print(f"reusing 1/f corrected {output_path}")
        else:
            _apply_one_over_f(
                input_path,
                output_path,
                correction_fn=correction_fn,
                sigma_bgmask=float(args.sigma_bgmask),
                sigma_1fmask=float(args.sigma_1fmask),
                splitamps=bool(args.splitamps),
                usesegmask=not bool(args.no_segmask),
            )
            print(f"1/f corrected {output_path}")
        corrected_paths.append(output_path)

    states = [_read_state(path) for path in corrected_paths]
    with fits.open(args.target_reference, memmap=True) as hdul:
        reference_header = hdul["SCI"].header.copy()
    temporary_header = _downsample_header(reference_header, factor=2)
    target = TargetRenderSpec.from_header(
        temporary_header,
        value_semantics="surface_brightness",
    )

    pass_rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="torchsky_updc_completion_") as tmp:
        tmpdir = Path(tmp)
        for pass_index in (1, 2):
            aligned, _ = _aligned_stack(
                states,
                target_header=temporary_header,
                tmpdir=tmpdir,
                device=device,
                pass_index=pass_index,
            )
            location, scale, usable_sky = _fit_gaussian_maps(
                aligned,
                minimum_samples=int(args.minimum_samples),
            )
            rows = _detect_pass(
                states,
                target=target,
                location=location,
                scale=scale,
                usable_sky=usable_sky,
                sigma=float(args.outlier_sigma),
                device=device,
            )
            pass_rows.append({"pass": pass_index, "exposures": rows})

        # Refit after both rejection passes before drawing replacement values.
        aligned, _ = _aligned_stack(
            states,
            target_header=temporary_header,
            tmpdir=tmpdir,
            device=device,
            pass_index=3,
        )
        location, scale, usable_sky = _fit_gaussian_maps(
            aligned,
            minimum_samples=int(args.minimum_samples),
        )
        output_rows = _replace_and_write(
            states,
            target=target,
            location=location,
            scale=scale,
            usable_sky=usable_sky,
            output_dir=completed_dir,
            seed=int(args.seed),
            device=device,
        )

    summary = {
        "band": str(args.band).upper(),
        "module": str(args.module).lower(),
        "n_exposures": len(states),
        "image1overf": {
            "script": str(args.image1overf),
            "sigma_bgmask": float(args.sigma_bgmask),
            "sigma_1fmask": float(args.sigma_1fmask),
            "splitamps": bool(args.splitamps),
            "usesegmask": not bool(args.no_segmask),
        },
        "completion": {
            "temporary_grid_factor": 2,
            "passes": 2,
            "outlier_sigma": float(args.outlier_sigma),
            "minimum_samples": int(args.minimum_samples),
            "seed": int(args.seed),
            "implementation_note": (
                "TorchSky WCS interpolation replaces the unpublished exact "
                "footprint-voting implementation."
            ),
        },
        "pass_diagnostics": pass_rows,
        "outputs": output_rows,
    }
    summary_path = args.output_dir / "preparation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
