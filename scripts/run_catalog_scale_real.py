#!/usr/bin/env python
"""Real-catalog gate: Gaia-subset × 2MASS/unWISE Parquet/HATS via xmatch.

Set paths with env vars; do not download inside torchsky.

  CATALOG_SCALE_LEFT=/path/gaia.parquet \\
  CATALOG_SCALE_RIGHT=/path/2mass.parquet \\
  pixi run -e test -- python scripts/run_catalog_scale_real.py

Requires the xmatch package. Linux numbers only for claims.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> None:
    left = os.environ.get("CATALOG_SCALE_LEFT")
    right = os.environ.get("CATALOG_SCALE_RIGHT")
    if not left or not right:
        raise SystemExit(
            "Set CATALOG_SCALE_LEFT and CATALOG_SCALE_RIGHT to Parquet or HATS roots"
        )
    try:
        import polars as pl
        from xmatch.matchers import MatchSpec, sky_match
        from xmatch.sources import CatalogueSource
    except ImportError as exc:
        raise SystemExit(f"xmatch is required for the real-catalog gate: {exc}") from exc

    left_path = Path(left)
    right_path = Path(right)
    if not left_path.exists() or not right_path.exists():
        raise SystemExit(f"missing catalog path: {left_path} or {right_path}")

    def _source(name: str, path: Path) -> CatalogueSource:
        if path.is_dir():
            return CatalogueSource(
                name=name,
                is_local=False,
                access_method="hats",
                access_identifier=str(path),
                ra_column="ra",
                dec_column="dec",
            )
        lf = pl.scan_parquet(path)
        src = CatalogueSource(
            name=name,
            is_local=True,
            ra_column="ra",
            dec_column="dec",
            path=path,
        )
        return src.with_frame(lf)

    src1 = _source("left", left_path)
    src2 = _source("right", right_path)
    spec = MatchSpec(radius_arcsec=1.0, fallback_policy="error", find="best")
    if src1.access_method == "hats" or src2.access_method == "hats":
        from xmatch.hats_native import hats_native_crossmatch

        out = hats_native_crossmatch(src1, src2, spec, engine="torchsky")
    else:
        out = sky_match(
            src1,
            src2,
            src1.lazy(),
            src2.lazy(),
            spec,
            engine="torchsky",
        )
        if hasattr(out, "collect"):
            out = out.collect()
    payload = {
        "left": str(left_path),
        "right": str(right_path),
        "n_match": int(out.height),
        "engine": "torchsky",
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    sys.exit(main())
