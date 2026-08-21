#!/usr/bin/env python
"""Synthetic catalog-scale contract: pair-ID parity vs scipy cKDTree.

Linux CANFAR is the claim machine. Mac numbers are wiring only.

  pixi run -e test -- python scripts/run_catalog_scale_contract.py --left-rows 20000 --engine-fast
  pixi run -e test -- python scripts/run_catalog_scale_contract.py --left-rows 10000000 --engine-fast --find best
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import resource
import time
from pathlib import Path

import numpy as np
import torch

from torchsky.catalogs import crossmatch_sky
from torchsky.sphere import chord_radius, lonlat_to_unit_xyz


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def _synthetic(n_left: int, n_right: int, *, seed: int = 20260821, layout: str = "sparse"):
    rng = np.random.default_rng(seed)
    if layout == "dense":
        left_ra = rng.uniform(150.0, 150.5, n_left)
        left_dec = rng.uniform(1.5, 2.0, n_left)
        right_ra = rng.uniform(150.0, 150.5, n_right)
        right_dec = rng.uniform(1.5, 2.0, n_right)
    else:
        left_ra = rng.uniform(0.0, 360.0, n_left)
        left_dec = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, n_left)))
        right_ra = rng.uniform(0.0, 360.0, n_right)
        right_dec = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, n_right)))
    n_hit = min(n_left, n_right)
    right_ra[:n_hit] = np.mod(left_ra[:n_hit] + 1.0e-7, 360.0)
    right_dec[:n_hit] = left_dec[:n_hit] - 1.0e-7
    return left_ra, left_dec, right_ra, right_dec


def _pair_set(left_idx, right_idx) -> set[tuple[int, int]]:
    return {(int(a), int(b)) for a, b in zip(left_idx, right_idx, strict=True)}


def _unit_xyz(ra, dec) -> np.ndarray:
    return lonlat_to_unit_xyz(
        torch.as_tensor(ra, dtype=torch.float64),
        torch.as_tensor(dec, dtype=torch.float64),
    ).numpy()


def _ckdtree_pairs(left_ra, left_dec, right_ra, right_dec, radius_arcsec: float, find: str):
    from scipy.spatial import cKDTree

    left = _unit_xyz(left_ra, left_dec)
    right = _unit_xyz(right_ra, right_dec)
    chord = float(chord_radius(math.radians(radius_arcsec / 3600.0)))
    tree = cKDTree(right)
    if find == "best":
        dist, idx = tree.query(left, k=1, workers=-1)
        keep = dist <= chord
        left_i = np.nonzero(keep)[0]
        return set(zip(left_i.tolist(), idx[keep].astype(np.int64).tolist(), strict=True))
    pairs: list[tuple[int, int]] = []
    for i, neigh in enumerate(tree.query_ball_point(left, r=chord, workers=-1)):
        for j in neigh:
            pairs.append((i, int(j)))
    return set(pairs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-rows", type=int, default=20_000)
    parser.add_argument("--right-rows", type=int, default=None)
    parser.add_argument("--radius-arcsec", type=float, default=1.0)
    parser.add_argument("--layout", choices=("dense", "sparse"), default="sparse")
    parser.add_argument("--find", choices=("best", "all"), default="all")
    parser.add_argument("--engine-fast", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    n_right = args.right_rows if args.right_rows is not None else args.left_rows
    left_ra, left_dec, right_ra, right_dec = _synthetic(
        args.left_rows, n_right, layout=args.layout
    )
    t0 = time.perf_counter()
    result = crossmatch_sky(
        torch.from_numpy(np.ascontiguousarray(left_ra)),
        torch.from_numpy(np.ascontiguousarray(left_dec)),
        torch.from_numpy(np.ascontiguousarray(right_ra)),
        torch.from_numpy(np.ascontiguousarray(right_dec)),
        radius_arcsec=args.radius_arcsec,
        find=args.find,
    )
    torchsky_s = time.perf_counter() - t0
    ts_pairs = _pair_set(result.left_index.tolist(), result.right_index.tolist())
    payload: dict[str, object] = {
        "left_rows": args.left_rows,
        "right_rows": n_right,
        "radius_arcsec": args.radius_arcsec,
        "layout": args.layout,
        "find": args.find,
        "torchsky_pairs": len(ts_pairs),
        "torchsky_s": torchsky_s,
        "peak_rss_bytes": _peak_rss_bytes(),
        "pair_id_parity": None,
        "platform": platform.platform(),
    }
    if args.engine_fast:
        t1 = time.perf_counter()
        fast_pairs = _ckdtree_pairs(
            left_ra, left_dec, right_ra, right_dec, args.radius_arcsec, args.find
        )
        payload["fast_s"] = time.perf_counter() - t1
        payload["fast_pairs"] = len(fast_pairs)
        payload["pair_id_parity"] = ts_pairs == fast_pairs
        payload["peak_rss_bytes"] = _peak_rss_bytes()
        if not payload["pair_id_parity"]:
            raise SystemExit("pair-ID mismatch vs cKDTree chord query")
    print(json.dumps(payload, indent=2))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
