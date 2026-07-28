#!/usr/bin/env python3
"""Run the SMACS J0723 WCS-aware INR with all-valid-pixel epoch sampling.

This launcher preserves the original training script and all of its CLI options.
It changes only two data-loading/sampling defaults:

1. unless explicitly supplied, ``--max-pixels-per-exposure`` is set to ``0``,
   which loads every valid native SCI/ERR pixel into each exposure pool;
2. mini-batches are drawn without replacement from an independently shuffled
   pass over each exposure.  An exposure is reshuffled only after all of its
   loaded pixels have been used once.

Passing an explicit positive ``--max-pixels-per-exposure`` remains supported and
then performs epoch sampling over that capped pool, which is useful for quick
smoke tests.
"""

from __future__ import annotations

from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from torchsky.map.wcs_aware_epoch_sampling import install_epoch_sampler  # noqa: E402


def _has_cli_option(name: str) -> bool:
    return any(argument == name or argument.startswith(name + "=") for argument in sys.argv[1:])


def main() -> None:
    install_epoch_sampler()

    if not _has_cli_option("--max-pixels-per-exposure"):
        sys.argv.extend(["--max-pixels-per-exposure", "0"])

    target = ROOT / "scripts" / "smacs0723_wcs_aware_multiband_inr.py"
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
