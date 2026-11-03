#!/usr/bin/env python
"""Pre-fetch the real-data example samples into the sample cache.

Cache: ``~/.cache/torchsky/samples`` (override ``TORCHSKY_SAMPLE_CACHE``).
Large samples are skipped unless ``--with-large`` is given; ``--only NAME ...``
selects entries. JWST cutouts (``_mast_cutout.CUTOUTS``) are built by HTTP
range reads from MAST. Prints a table (name, size, sha256, status) and exits non-zero
if any requested sample failed.

    pixi run fetch-example-samples [-- --with-large] [-- --only horsehead]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

import _mast_cutout as mc  # noqa: E402
import _sample_data as sd  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--with-large", action="store_true", help="include large samples")
    ap.add_argument("--only", nargs="*", default=None, metavar="NAME")
    args = ap.parse_args(argv)
    names = args.only or [
        n for n, s in sd.SAMPLES.items() if args.with_large or not s["large"]
    ] + list(mc.CUTOUTS)
    print(f"cache: {sd.cache_dir()}")
    print(f"{'name':28s} {'bytes':>12s}  {'sha256':64s}  status")
    failed = 0
    for name in names:
        try:
            if name in mc.CUTOUTS:
                path = mc.fetch_cutout(name, allow_download=True)
            else:
                path = sd.fetch_sample(name, allow_download=True)
            print(f"{name:28s} {path.stat().st_size:12d}  {sd.sha256_file(path)}  ok")
        except (sd.SampleUnavailable, KeyError) as exc:
            failed += 1
            print(f"{name:28s} {'-':>12s}  {'-':64s}  FAIL: {exc}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
