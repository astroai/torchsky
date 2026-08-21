#!/usr/bin/env python
"""Real-catalog gate: Gaia-subset × 2MASS/unWISE Parquet/HATS.

Set CATALOG_SCALE_LEFT / CATALOG_SCALE_RIGHT, or let the runner look under
CATALOG_SCALE_HATS_ROOT (default /arc/projects/hats). Does not download.

Prefer xmatch engine=torchsky. If xmatch is missing, load a bounded set of
Parquet leaves and call crossmatch_sky. Linux numbers only for claims.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


_LEFT_KEYS = ("gaia",)
_RIGHT_KEYS = ("2mass", "twomass", "unwise")


def _is_hats(path: Path) -> bool:
    return (path / "properties").is_file() or (path / "hats.properties").is_file()


def _scan_hats(root: Path, keys: tuple[str, ...]) -> Path | None:
    if not root.is_dir():
        return None
    lowered = tuple(k.lower() for k in keys)
    candidates: list[Path] = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and any(k in child.name.lower() for k in lowered):
            candidates.append(child)
            candidates.extend(sorted(p for p in child.iterdir() if p.is_dir()))
    for path in candidates:
        if _is_hats(path):
            return path
        nested = path / "dataset"
        if nested.is_dir() and _is_hats(path):
            return path
    return None


def _parquet_leaves(root: Path, limit: int) -> list[Path]:
    files = sorted(root.rglob("*.parquet"))
    return files[:limit]


def _resolve_paths() -> tuple[Path | None, Path | None, str]:
    left = os.environ.get("CATALOG_SCALE_LEFT")
    right = os.environ.get("CATALOG_SCALE_RIGHT")
    if left and right:
        return Path(left), Path(right), "env"
    root = Path(os.environ.get("CATALOG_SCALE_HATS_ROOT", "/arc/projects/hats"))
    return (
        _scan_hats(root, _LEFT_KEYS),
        _scan_hats(root, _RIGHT_KEYS),
        f"scan:{root}",
    )


def _run_xmatch(left_path: Path, right_path: Path) -> dict:
    import polars as pl
    from xmatch.hats_native import hats_native_crossmatch
    from xmatch.matchers import MatchSpec, sky_match
    from xmatch.sources import CatalogueSource

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
    t0 = time.perf_counter()
    if src1.access_method == "hats" or src2.access_method == "hats":
        out = hats_native_crossmatch(src1, src2, spec, engine="torchsky")
    else:
        out = sky_match(src1, src2, src1.lazy(), src2.lazy(), spec, engine="torchsky")
        if hasattr(out, "collect"):
            out = out.collect()
    return {
        "n_match": int(out.height),
        "engine": "torchsky",
        "via": "xmatch",
        "torchsky_s": time.perf_counter() - t0,
    }


def _run_tensor_subset(left_path: Path, right_path: Path, max_files: int) -> dict:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch

    from torchsky.catalogs import crossmatch_sky

    def _load(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
        files = [path] if path.is_file() else _parquet_leaves(path, max_files)
        if not files:
            raise FileNotFoundError(f"no parquet under {path}")
        table = pa.concat_tables([pq.read_table(str(f)) for f in files])
        names = {n.lower(): n for n in table.column_names}
        ra_n = names.get("ra") or names.get("ra_icrs")
        dec_n = names.get("dec") or names.get("de_icrs") or names.get("dec_icrs")
        if ra_n is None or dec_n is None:
            raise ValueError(f"no ra/dec columns in {path}: {table.column_names}")
        ra = np.asarray(table[ra_n].to_numpy(), dtype=np.float64)
        dec = np.asarray(table[dec_n].to_numpy(), dtype=np.float64)
        return torch.from_numpy(ra), torch.from_numpy(dec)

    left_ra, left_dec = _load(left_path)
    right_ra, right_dec = _load(right_path)
    t0 = time.perf_counter()
    result = crossmatch_sky(
        left_ra, left_dec, right_ra, right_dec, radius_arcsec=1.0, find="best"
    )
    return {
        "n_match": int(result.left_index.numel()),
        "engine": "torchsky",
        "via": "tensor_subset",
        "left_rows": int(left_ra.numel()),
        "right_rows": int(right_ra.numel()),
        "torchsky_s": time.perf_counter() - t0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--max-parquet-files", type=int, default=8)
    args = parser.parse_args()
    left_path, right_path, how = _resolve_paths()
    payload: dict = {"source": how}
    if left_path is None or right_path is None:
        payload["skipped"] = True
        payload["reason"] = "no Gaia/2MASS/unWISE HATS or Parquet paths"
        print(json.dumps(payload, indent=2))
        if args.json_out is not None:
            args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
        return 0
    payload["left"] = str(left_path)
    payload["right"] = str(right_path)
    try:
        payload.update(_run_xmatch(left_path, right_path))
    except ImportError:
        payload.update(
            _run_tensor_subset(left_path, right_path, args.max_parquet_files)
        )
    print(json.dumps(payload, indent=2))
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
