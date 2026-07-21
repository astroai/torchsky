#!/usr/bin/env python3
"""ERR-whitened L2 entry point for the SMACS J0723 WCS-aware INR.

This wrapper preserves the existing science/HUBER entry point and reuses the
same band-balanced sampler, model, calibration, optimizer, and rendering code.
The only intended objective change is to optimize the un-clipped whitened
residual loss ``0.5 * ((prediction - target) / ERR) ** 2``.
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

import scripts.smacs0723_wcs_aware_multiband_inr as base  # noqa: E402
import torchsky.map.wcs_aware_training as training  # noqa: E402
from torchsky.map.wcs_aware_objectives import (  # noqa: E402
    err_whitened_robust_loss,
)
from torchsky.map.wcs_aware_sampling import (  # noqa: E402
    sample_band_balanced_batch,
)


training.weighted_robust_loss = err_whitened_robust_loss
training.sample_balanced_batch = sample_band_balanced_batch
base.train_model = training.train_model

_original_build_parser = base.build_parser


def _whitened_l2_build_parser() -> argparse.ArgumentParser:
    parser = _original_build_parser()
    for action in parser._actions:
        if action.dest == "batch_size":
            action.default = 1080
            action.help = (
                "Total band-balanced batch size; 1080 gives 180 samples per "
                "band for the six-band SMACS experiment."
            )
        elif action.dest == "robust_loss":
            action.choices = ("huber", "charbonnier", "whitened_l2")
            action.default = "whitened_l2"
            action.help = (
                "Objective on ERR-whitened residuals. This entry point defaults "
                "to un-clipped whitened L2; huber and charbonnier remain "
                "available for traceable comparisons."
            )
        elif action.dest == "robust_delta":
            action.default = 3.0
            action.help = (
                "Robust transition in ERR-whitened sigma units; ignored by "
                "whitened_l2."
            )
        elif action.dest == "max_relative_weight":
            action.help = (
                "Compatibility option retained by the base CLI; ignored because "
                "ERR whitening already supplies the noise normalization."
            )
    return parser


base.build_parser = _whitened_l2_build_parser


if __name__ == "__main__":
    raise SystemExit(base.main())
