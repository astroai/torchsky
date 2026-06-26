#!/usr/bin/env python3
"""Build exposure-specific HybPSF PC models for SMACS J0723.

Run this script inside the dedicated ``hybpsf`` Pixi environment. It uses the
public HybPSF v1.0.0 API without modifying that repository. Compatibility
patches are applied only in this Python process:

- expose ``source-extractor`` under the legacy command name ``sex``;
- add the missing ``NRCBLONG -> NRCB5`` detector mapping;
- merge each exposure's UPDCMASK into the selected-star masks.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from astropy.io import fits


matplotlib.use("Agg")


@contextmanager
def _redirect_process_output(log_path: Path):
    """Redirect Python, subprocess, and native-library output to one log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    try:
        with log_path.open("w", encoding="utf-8") as log:
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
            yield
            sys.stdout.flush()
            sys.stderr.flush()
    finally:
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)


def _exposure_key(path: Path) -> str:
    for suffix in (
        "_cal_1overf_completed_dqfix.fits",
        "_cal_1overf_completed.fits",
        "_cal_1overf.fits",
        "_cal.fits",
        ".fits",
    ):
        if path.name.endswith(suffix):
            return path.name[: -len(suffix)]
    return path.stem


def _install_sex_alias(root: Path) -> None:
    executable = shutil.which("source-extractor") or shutil.which("sex")
    if executable is None:
        raise RuntimeError("source-extractor/sex is not available")
    alias_dir = root / "bin"
    alias_dir.mkdir(parents=True, exist_ok=True)
    alias = alias_dir / "sex"
    if not alias.exists():
        alias.symlink_to(Path(executable).resolve())
    # HybPSF v1.0.0 otherwise invokes the same executable up to three times:
    # first as ``sex``, then as ``source-extractor``, then as ``sex`` again.
    # Keep only the legacy alias visible to that subprocess.
    os.environ["PATH"] = os.pathsep.join(
        (str(alias_dir), "/usr/local/bin", "/usr/bin", "/bin")
    )


def _import_hybpsf(root: Path):
    sys.path.insert(0, str(root.resolve()))
    import HybPSF  # type: ignore
    from HybPSF import HybPSF as implementation  # type: ignore

    # HybPSF v1.0.0 contains NRCABLONG where NRCBLONG is intended.
    implementation.detector["NRCBLONG"] = "NRCB5"
    implementation.gains["NRCBLONG"] = "1.80"
    return HybPSF, implementation


def _extract_centered_stamp(
    image: np.ndarray,
    *,
    x_fits: float,
    y_fits: float,
    size: int,
) -> np.ndarray:
    """Extract a FITS-coordinate-centered stamp, padding outside as bad."""
    if size <= 0:
        raise ValueError("stamp size must be positive")
    cx = int(np.floor(float(x_fits) - 1.0 + 0.5))
    cy = int(np.floor(float(y_fits) - 1.0 + 0.5))
    y0 = cy - size // 2
    x0 = cx - size // 2
    out = np.ones((size, size), dtype=bool)

    src_y0 = max(y0, 0)
    src_x0 = max(x0, 0)
    src_y1 = min(y0 + size, image.shape[0])
    src_x1 = min(x0 + size, image.shape[1])
    if src_y1 <= src_y0 or src_x1 <= src_x0:
        return out

    dst_y0 = src_y0 - y0
    dst_x0 = src_x0 - x0
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    out[dst_y0:dst_y1, dst_x0:dst_x1] = image[
        src_y0:src_y1,
        src_x0:src_x1,
    ]
    return out


def _merge_updcmask(exposure: Path, stamp_path: Path) -> dict[str, int]:
    with fits.open(exposure, memmap=False) as hdul:
        if "UPDCMASK" not in hdul:
            raise RuntimeError(f"{exposure}: missing UPDCMASK extension")
        abnormal = np.asarray(hdul["UPDCMASK"].data, dtype=bool)

    with fits.open(stamp_path, mode="update", memmap=False) as hdul:
        stars = np.asarray(hdul[0].data)
        catalog = np.asarray(hdul[1].data)
        masks = np.asarray(hdul[2].data)
        if stars.ndim != 3 or masks.shape != stars.shape:
            raise ValueError(
                f"{stamp_path}: unexpected star/mask shapes "
                f"{stars.shape} and {masks.shape}"
            )
        if catalog.ndim != 2 or catalog.shape[0] != stars.shape[0]:
            raise ValueError(f"{stamp_path}: invalid star catalogue shape")

        newly_masked = 0
        for index in range(stars.shape[0]):
            bad_stamp = _extract_centered_stamp(
                abnormal,
                x_fits=float(catalog[index, 1]),
                y_fits=float(catalog[index, 2]),
                size=int(stars.shape[1]),
            )
            before = masks[index] != 0
            masks[index][bad_stamp] = 0
            newly_masked += int((before & bad_stamp).sum())
        hdul[2].data[...] = masks
        hdul[0].header.add_history(
            "UPDCMASK abnormal pixels merged into HybPSF star masks"
        )
        hdul.flush()

    return {
        "selected_stars_before_pc": int(stars.shape[0]),
        "newly_masked_star_pixels": newly_masked,
    }


def _write_science_only_exposure(exposure: Path, output: Path) -> Path:
    """Write PRIMARY+SCI only so SExtractor processes exactly one image."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with fits.open(exposure, memmap=False) as hdul:
        primary = fits.PrimaryHDU(header=hdul[0].header.copy())
        sci = fits.ImageHDU(
            data=np.asarray(hdul["SCI"].data, dtype=np.float32).copy(),
            header=hdul["SCI"].header.copy(),
            name="SCI",
        )
    primary.header.add_history(
        "HybPSF staging file: non-SCI extensions omitted for SExtractor"
    )
    fits.HDUList([primary, sci]).writeto(output, overwrite=True)
    return output


def _build_one(
    exposure: Path,
    *,
    work_dir: Path,
    implementation: Any,
    npcs: int,
    snr_threshold: float,
    oversampling: int,
    stamp_size: int,
) -> dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    staged_exposure = _write_science_only_exposure(
        exposure,
        work_dir / "science_only" / exposure.name,
    )
    stamp_path = implementation._pick_psfstars(
        str(staged_exposure.parent),
        staged_exposure.name,
        str(work_dir),
        int(stamp_size),
    )
    if not stamp_path:
        raise RuntimeError(f"{exposure.name}: HybPSF did not return star stamps")
    stamp_path = Path(stamp_path).resolve()
    if not stamp_path.is_file():
        raise FileNotFoundError(stamp_path)

    mask_diagnostics = _merge_updcmask(exposure, stamp_path)
    pc_path = implementation._get_PC(
        str(work_dir),
        str(stamp_path),
        NPCs=int(npcs),
        method=1,
        osam=int(oversampling),
        SNR_thresh=float(snr_threshold),
        big=0,
    )
    if not pc_path:
        raise RuntimeError(
            f"{exposure.name}: HybPSF failed to construct a PC model"
        )
    pc_path = Path(pc_path).resolve()
    if not pc_path.is_file():
        raise FileNotFoundError(pc_path)

    with fits.open(pc_path, memmap=False) as hdul:
        pcs = np.asarray(hdul[0].data)
        positions = np.asarray(hdul[1].data)
        coefficients = np.asarray(hdul[2].data)
        web_coefficients = np.asarray(hdul[3].data)
        header = hdul[0].header

    if positions.shape[0] < 6:
        raise RuntimeError(
            f"{exposure.name}: only {positions.shape[0]} stars survived; "
            "the public renderer would not produce a HybPSF"
        )
    return {
        "exposure": str(exposure),
        "science_only_exposure": str(staged_exposure),
        "stamp_file": str(stamp_path),
        "pc_file": str(pc_path),
        "pc_shape": list(pcs.shape),
        "stars_used_by_pc": int(positions.shape[0]),
        "coefficient_shape": list(coefficients.shape),
        "web_coefficient_shape": list(web_coefficients.shape),
        "filter": str(header.get("FILTER", "")),
        "detector": str(header.get("DETECTOR", "")),
        "date_beg": str(header.get("DATE-BEG", "")),
        "oversampling": int(header.get("OSAM", oversampling)),
        **mask_diagnostics,
    }


def _validate_pc_file(path: Path, *, expected_npcs: int) -> dict[str, Any]:
    with fits.open(path, memmap=False) as hdul:
        if len(hdul) < 4:
            raise ValueError(f"{path}: expected at least four HDUs")
        pcs = np.asarray(hdul[0].data)
        positions = np.asarray(hdul[1].data)
        coefficients = np.asarray(hdul[2].data)
        web_coefficients = np.asarray(hdul[3].data)
        header = hdul[0].header
    if pcs.ndim != 3 or pcs.shape[0] != int(expected_npcs):
        raise ValueError(f"{path}: invalid PC shape {pcs.shape}")
    if positions.ndim != 2 or positions.shape[0] < 6:
        raise ValueError(f"{path}: only {positions.shape[0]} usable stars")
    if coefficients.shape[:2] != (positions.shape[0], int(expected_npcs)):
        raise ValueError(f"{path}: invalid coefficient shape {coefficients.shape}")
    if web_coefficients.shape[0] != positions.shape[0]:
        raise ValueError(
            f"{path}: invalid WebbPSF coefficient shape "
            f"{web_coefficients.shape}"
        )
    return {
        "pc_shape": list(pcs.shape),
        "stars_used_by_pc": int(positions.shape[0]),
        "coefficient_shape": list(coefficients.shape),
        "web_coefficient_shape": list(web_coefficients.shape),
        "filter": str(header.get("FILTER", "")),
        "detector": str(header.get("DETECTOR", "")),
        "date_beg": str(header.get("DATE-BEG", "")),
        "oversampling": int(header.get("OSAM", 0)),
    }


def _find_reusable_pc(
    output_dir: Path,
    *,
    key: str,
    expected_npcs: int,
) -> tuple[Path, dict[str, Any]] | None:
    candidates = [
        output_dir / "models" / f"{key}_hybpsf_pc.fits",
        *sorted((output_dir / "work" / key).glob("*_PSF.fits")),
        *sorted((output_dir / key).glob("*_PSF.fits")),
    ]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            return candidate, _validate_pc_file(
                candidate,
                expected_npcs=expected_npcs,
            )
        except (OSError, ValueError):
            continue
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hybpsf-root",
        type=Path,
        default=Path("external/HybPSF"),
    )
    parser.add_argument("--exposure-dir", type=Path, required=True)
    parser.add_argument(
        "--exposure-glob",
        default="*_cal_1overf_completed_dqfix.fits",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--npcs", type=int, default=4)
    parser.add_argument("--snr-threshold", type=float, default=20.0)
    parser.add_argument("--oversampling", type=int, default=2, choices=(1, 2))
    parser.add_argument("--stamp-size", type=int, default=80)
    parser.add_argument(
        "--max-exposures",
        type=int,
        default=1,
        help="Use 1 for the smoke test; use 0 for every exposure.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse structurally valid PC models from an interrupted run.",
    )
    args = parser.parse_args()

    root = args.hybpsf_root.expanduser().resolve()
    exposures = sorted(
        args.exposure_dir.expanduser().resolve().glob(args.exposure_glob)
    )
    if args.max_exposures > 0:
        exposures = exposures[: int(args.max_exposures)]
    if not exposures:
        raise FileNotFoundError(
            f"no exposures matching {args.exposure_glob!r}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    work_root = args.output_dir / "work"
    model_root = args.output_dir / "models"
    log_root = args.output_dir / "logs"
    model_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="hybpsf_commands_") as temp:
        _install_sex_alias(Path(temp))
        _, implementation = _import_hybpsf(root)
        rows = []
        total = len(exposures)
        for index, exposure in enumerate(exposures, start=1):
            key = _exposure_key(exposure)
            log_path = log_root / f"{key}.log"
            reusable = None
            if args.resume:
                reusable = _find_reusable_pc(
                    args.output_dir,
                    key=key,
                    expected_npcs=int(args.npcs),
                )
            if reusable is not None:
                internal_pc, diagnostics = reusable
                model_path = model_root / f"{key}_hybpsf_pc.fits"
                if internal_pc.resolve() != model_path.resolve():
                    shutil.copy2(internal_pc, model_path)
                row = {
                    "exposure": str(exposure),
                    "pc_file": str(model_path.resolve()),
                    "internal_pc_file": str(internal_pc.resolve()),
                    "log_file": (
                        str(log_path.resolve()) if log_path.exists() else None
                    ),
                    "reused": True,
                    **diagnostics,
                }
                rows.append(row)
                print(
                    f"[{index}/{total}] {key}: REUSED | "
                    f"stars {row['stars_used_by_pc']} | "
                    f"PC {tuple(row['pc_shape'])}",
                    flush=True,
                )
                continue

            print(f"[{index}/{total}] {key}: building...", flush=True)
            try:
                with _redirect_process_output(log_path):
                    row = _build_one(
                        exposure,
                        work_dir=work_root / key,
                        implementation=implementation,
                        npcs=int(args.npcs),
                        snr_threshold=float(args.snr_threshold),
                        oversampling=int(args.oversampling),
                        stamp_size=int(args.stamp_size),
                    )
            except Exception:
                print(
                    f"[{index}/{total}] {key}: FAILED; see {log_path}",
                    flush=True,
                )
                raise

            internal_pc = Path(str(row["pc_file"]))
            model_path = model_root / f"{key}_hybpsf_pc.fits"
            shutil.copy2(internal_pc, model_path)
            row["internal_pc_file"] = str(internal_pc)
            row["pc_file"] = str(model_path.resolve())
            row["log_file"] = str(log_path.resolve())
            row["reused"] = False
            rows.append(row)
            print(
                f"[{index}/{total}] {key}: OK | "
                f"stars {row['selected_stars_before_pc']} -> "
                f"{row['stars_used_by_pc']} | "
                f"PC {tuple(row['pc_shape'])}",
                flush=True,
            )

    summary = {
        "hybpsf_root": str(root),
        "npcs": int(args.npcs),
        "snr_threshold": float(args.snr_threshold),
        "oversampling": int(args.oversampling),
        "stamp_size": int(args.stamp_size),
        "outputs": rows,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print()
    print(f"Completed {len(rows)} exposure(s).")
    print(f"Models:  {model_root}")
    print(f"Summary: {summary_path}")
    print(f"Logs:    {log_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
