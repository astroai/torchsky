#!/usr/bin/env python3
"""Large-batch ERR-whitened L2 experiment for SMACS J0723.

This entry point preserves the existing six-band model, band-balanced sampler,
optimizer, calibration, and no-PSF ablation while increasing the sampling
budget to 100 pixels per exposure per step for the 36-exposure short-wave
bands and 400 pixels per exposure per step for the 9-exposure long-wave bands.

For the current exposure counts, a total batch size of 21,600 gives exactly
3,600 samples per band:

* F090W/F150W/F200W: 3,600 / 36 = 100 pixels per exposure.
* F277W/F356W/F444W: 3,600 / 9 = 400 pixels per exposure.

The default per-exposure pool cap is 3,450,000 rather than 3,500,000 so that
even the exposure with the fewest valid pixels retains at least roughly 50,000
pixels outside its training pool for held-out native-pixel evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import scripts.smacs0723_wcs_aware_multiband_inr_whitened_l2 as l2  # noqa: E402

base = l2.base
_original_build_parser = base.build_parser


def _large_batch_build_parser() -> argparse.ArgumentParser:
    parser = _original_build_parser()
    for action in parser._actions:
        if action.dest == "batch_size":
            action.default = 21600
            action.help = (
                "Total band-balanced batch size. With the current 36/36/36/"
                "9/9/9 exposure counts, 21600 gives 100 pixels per short-wave "
                "exposure and 400 pixels per long-wave exposure per step."
            )
        elif action.dest == "max_pixels_per_exposure":
            action.default = 3450000
            action.help = (
                "Maximum random valid-pixel training pool per exposure. The "
                "default 3,450,000 preserves at least about 50,000 non-pool "
                "valid pixels in the smallest current exposure."
            )
    return parser


base.build_parser = _large_batch_build_parser


if __name__ == "__main__":
    raise SystemExit(base.main())
