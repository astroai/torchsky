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
from collections import Counter
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
_original_load_exposure_pool = base.load_exposure_pool
_original_train_model = base.train_model
_load_counter = 0


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
        elif action.dest == "log_every":
            action.default = 100
            action.help = "Print training diagnostics every 100 optimizer steps."
    return parser


def _logged_load_exposure_pool(path, band, detector, **kwargs):
    """Expose the otherwise silent, memory-heavy pool construction phase."""
    global _load_counter
    _load_counter += 1
    cap = kwargs.get("max_pixels", "unknown")
    print(
        f"[pool {_load_counter:03d}] loading {band} {Path(path).name} "
        f"(cap={int(cap):,} pixels)",
        flush=True,
    )
    pool = _original_load_exposure_pool(path, band, detector, **kwargs)
    print(
        f"[pool {_load_counter:03d}] ready   {band} {Path(path).name}: "
        f"{pool.size:,} pixels",
        flush=True,
    )
    return pool


def _logged_train_model(*args, **kwargs):
    """Print the effective sampler configuration before the first update."""
    pools = args[1] if len(args) > 1 else kwargs["pools"]
    counts = Counter(pool.band for pool in pools)
    batch_size = int(kwargs["batch_size"])
    steps = int(kwargs["steps"])
    log_every = int(kwargs["log_every"])
    band_count = len(counts)
    per_band = batch_size // band_count

    print(
        f"[train] starting optimizer loop: steps={steps:,}, "
        f"batch_size={batch_size:,}, log_every={log_every}",
        flush=True,
    )
    for band in sorted(counts):
        n_exposure = counts[band]
        print(
            f"[train] {band}: {n_exposure} exposures, "
            f"{per_band:,} samples/band/step, "
            f"{per_band // n_exposure:,} samples/exposure/step",
            flush=True,
        )
    return _original_train_model(*args, **kwargs)


base.build_parser = _large_batch_build_parser
base.load_exposure_pool = _logged_load_exposure_pool
base.train_model = _logged_train_model


if __name__ == "__main__":
    raise SystemExit(base.main())
