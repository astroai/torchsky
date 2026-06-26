#!/usr/bin/env python3
"""Run SMACS J0723 UPDC preparation with a conservative JWST DQ policy.

This companion script deliberately does not modify
``prepare_smacs0723_updc_inputs.py``. It imports that implementation and
replaces only two internal functions:

1. Pixels are excluded from statistical fitting only when they are outside
   the NIRCam science region, non-finite, or carry the JWST DO_NOT_USE bit.
2. Pixel completion no longer treats every non-zero DQ value as replaceable.

Results are written to ``completed_dqfix`` beside the original ``completed``
directory. The original one-over-f products are reused.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import prepare_smacs0723_updc_inputs as base  # noqa: E402


JWST_DO_NOT_USE = np.uint64(1)
NIRCAM_FULL_FRAME_SIZE = 2048
NIRCAM_REFERENCE_BORDER = 4


def _science_region_mask(shape: tuple[int, ...]) -> np.ndarray:
    """Return the detector region containing science pixels."""
    if len(shape) != 2:
        raise ValueError(f"expected a 2-D detector image, got shape={shape}")

    mask = np.ones(shape, dtype=bool)
    if shape == (NIRCAM_FULL_FRAME_SIZE, NIRCAM_FULL_FRAME_SIZE):
        mask[:] = False
        border = NIRCAM_REFERENCE_BORDER
        mask[border:-border, border:-border] = True
    return mask


def _do_not_use_mask(dq: np.ndarray) -> np.ndarray:
    """Select only pixels carrying the semantic JWST DO_NOT_USE bit."""
    return (dq.astype(np.uint64, copy=False) & JWST_DO_NOT_USE) != 0


def _read_state_dqfix(path: Path) -> base.ExposureState:
    with fits.open(path, memmap=False) as hdul:
        data = np.asarray(hdul["SCI"].data, dtype=np.float32).copy()
        header = hdul["SCI"].header.copy()
        dq = np.asarray(hdul["DQ"].data).copy()

    science_region = _science_region_mask(data.shape)
    finite = np.isfinite(data)
    do_not_use = _do_not_use_mask(dq)
    valid = science_region & finite & ~do_not_use

    return base.ExposureState(
        path=path,
        data=data,
        header=header,
        dq=dq,
        valid=valid,
        rejected=np.zeros_like(valid),
    )


def _replace_and_write_dqfix(
    states: list[base.ExposureState],
    *,
    target: base.TargetRenderSpec,
    location: np.ndarray,
    scale: np.ndarray,
    usable_sky: np.ndarray,
    output_dir: Path,
    seed: int,
    device: base.torch.device,
) -> list[dict[str, Any]]:
    # Preserve the outputs produced by the earlier all-DQ implementation.
    output_dir = output_dir.parent / "completed_dqfix"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for index, state in enumerate(states):
        detector_location = base._project_statistic_to_detector(
            location,
            target=target,
            state=state,
            device=device,
        )
        detector_scale = np.abs(
            base._project_statistic_to_detector(
                scale,
                target=target,
                state=state,
                device=device,
            )
        )
        detector_support = (
            base._project_statistic_to_detector(
                usable_sky.astype(np.float32),
                target=target,
                state=state,
                device=device,
            )
            > 0.999
        )

        science_region = _science_region_mask(state.data.shape)
        nonfinite = ~np.isfinite(state.data)
        do_not_use = _do_not_use_mask(state.dq)

        replace_candidates = state.rejected | nonfinite | do_not_use
        replace = detector_support & science_region & replace_candidates

        rng = np.random.default_rng(int(seed) + index)
        draws = rng.normal(detector_location, detector_scale).astype(np.float32)
        corrected = state.data.copy()
        corrected[replace] = draws[replace]

        name = state.path.name
        if name.endswith("_cal_1overf.fits"):
            name = name.replace(
                "_cal_1overf.fits",
                "_cal_1overf_completed_dqfix.fits",
            )
        else:
            name = f"{state.path.stem}_completed_dqfix.fits"
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
            hdul[0].header.add_history(
                "DQ policy: replace DO_NOT_USE bit only; preserve other DQ flags"
            )
            hdul[0].header["UPDCSEED"] = (
                int(seed) + index,
                "Random seed used for Gaussian pixel completion",
            )
            hdul[0].header["UPDCDQ"] = (
                "DO_NOT_USE",
                "JWST DQ policy used for pixel completion",
            )
            hdul.writeto(output_path, overwrite=True)

        statistical = state.rejected & science_region
        do_not_use_science = do_not_use & science_region
        nonfinite_science = nonfinite & science_region
        rows.append(
            {
                "input": str(state.path),
                "output": str(output_path),
                "science_pixels": int(science_region.sum()),
                "statistical_outliers": int(statistical.sum()),
                "do_not_use_pixels": int(do_not_use_science.sum()),
                "nonfinite_pixels": int(nonfinite_science.sum()),
                "replace_candidates_union": int(
                    (replace_candidates & science_region).sum()
                ),
                "replaced_pixels": int(replace.sum()),
                "replaced_fraction_science": float(
                    replace.sum() / science_region.sum()
                ),
                "unreplaced_candidates_no_sky_support": int(
                    (
                        replace_candidates
                        & science_region
                        & ~detector_support
                    ).sum()
                ),
            }
        )

    return rows


def main() -> int:
    base._read_state = _read_state_dqfix
    base._replace_and_write = _replace_and_write_dqfix
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
