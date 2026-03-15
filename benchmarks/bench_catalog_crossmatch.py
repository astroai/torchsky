#!/usr/bin/env python3
"""Benchmark catalog crossmatch refinement: declination pruning vs coarse HEALPix index."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the repository root is in sys.path so we can import benchmarks.config
repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
src_root = str(Path(__file__).resolve().parent.parent / "src")
if src_root not in sys.path:
    sys.path.insert(0, src_root)

from benchmarks.config import DEFAULT_OUTPUT_DIR  # noqa: E402


import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Callable  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import torchsky.catalogs.core as catalogs_core  # noqa: E402
from torchsky.catalogs import crossmatch_sky  # noqa: E402


def _time_many(fn: Callable[[], Any], runs: int) -> float:
    fn()
    samples: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples))


def _random_sphere(
    rng: np.random.Generator,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    lon = rng.uniform(0.0, 360.0, size=n)
    lat = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, size=n)))
    return lon.astype(np.float64), lat.astype(np.float64)


def _make_catalogs(
    *,
    n_left: int,
    n_right_band: int,
    n_right_background: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    left_center_ra = 42.0
    left_center_dec = 4.0

    left_ra = left_center_ra + rng.normal(0.0, 0.12, size=n_left)
    left_dec = left_center_dec + rng.normal(0.0, 0.08, size=n_left)
    left_ra = np.remainder(left_ra, 360.0).astype(np.float64)
    left_dec = np.clip(left_dec, -85.0, 85.0).astype(np.float64)

    matched = min(max(16, n_left // 2), n_left)
    matched_ra = np.remainder(
        left_ra[:matched] + rng.normal(0.0, 0.00008, size=matched), 360.0
    )
    matched_dec = np.clip(
        left_dec[:matched] + rng.normal(0.0, 0.00008, size=matched),
        -89.999,
        89.999,
    )

    band_ra = rng.uniform(0.0, 360.0, size=n_right_band)
    band_dec = left_center_dec + rng.normal(0.0, 0.12, size=n_right_band)
    bg_ra, bg_dec = _random_sphere(rng, n_right_background)

    right_ra = np.concatenate((matched_ra, band_ra, bg_ra)).astype(np.float64)
    right_dec = np.concatenate(
        (
            matched_dec,
            np.clip(band_dec, -89.999, 89.999),
            np.clip(bg_dec, -89.999, 89.999),
        )
    ).astype(np.float64)

    return (
        torch.from_numpy(left_ra),
        torch.from_numpy(left_dec),
        torch.from_numpy(right_ra),
        torch.from_numpy(right_dec),
    )


def _assert_same_matches(
    got: catalogs_core.CrossmatchResult,
    expected: catalogs_core.CrossmatchResult,
) -> None:
    if got.left_index.tolist() != expected.left_index.tolist():
        raise RuntimeError(
            "left_index mismatch between coarse and declination crossmatch"
        )
    if got.right_index.tolist() != expected.right_index.tolist():
        raise RuntimeError(
            "right_index mismatch between coarse and declination crossmatch"
        )
    torch.testing.assert_close(
        got.separation_arcsec,
        expected.separation_arcsec,
        atol=1e-9,
        rtol=0.0,
    )


def _recommendation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    decl = next(
        (row for row in rows if row.get("operation") == "crossmatch_declination_pruned"),
        None,
    )
    coarse = next(
        (row for row in rows if row.get("operation") == "crossmatch_coarse_healpix"),
        None,
    )
    if decl is None or coarse is None:
        return []
    recommended = "crossmatch_coarse_healpix"
    note = "Structured coarse HEALPix pruning is faster on this workload"
    if float(coarse["time_s"]) >= float(decl["time_s"]):
        recommended = "crossmatch_declination_pruned"
        note = "Declination pruning is still better on this workload size"
    return [
        {
            "operation": "crossmatch_recommendation",
            "library": "torchsky_recommendation",
            "recommended_operation": recommended,
            "time_s_declination": decl["time_s"],
            "time_s_coarse": coarse["time_s"],
            "candidate_row_reduction_vs_declination": coarse.get(
                "candidate_row_reduction_vs_declination",
                float("nan"),
            ),
            "pair_reduction_vs_declination": coarse.get(
                "pair_reduction_vs_declination",
                float("nan"),
            ),
            "coarse_nside": coarse.get("coarse_nside"),
            "note": note,
        }
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-left", type=int, default=20_000)
    parser.add_argument("--n-right-band", type=int, default=250_000)
    parser.add_argument("--n-right-background", type=int, default=50_000)
    parser.add_argument("--radius-arcsec", type=float, default=2.0)
    parser.add_argument("--left-chunk-size", type=int, default=4096)
    parser.add_argument("--right-chunk-size", type=int, default=8192)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "catalog_crossmatch.json",
    )
    args = parser.parse_args()

    left_ra, left_dec, right_ra, right_dec = _make_catalogs(
        n_left=args.n_left,
        n_right_band=args.n_right_band,
        n_right_background=args.n_right_background,
        seed=args.seed,
    )

    decl_result, decl_stats = catalogs_core._crossmatch_sky_declination_pruned(
        left_ra,
        left_dec,
        right_ra,
        right_dec,
        radius_arcsec=args.radius_arcsec,
        left_chunk_size=args.left_chunk_size,
        right_chunk_size=args.right_chunk_size,
        return_stats=True,
    )
    coarse_result, coarse_stats = catalogs_core._crossmatch_sky_coarse_indexed(
        left_ra,
        left_dec,
        right_ra,
        right_dec,
        radius_arcsec=args.radius_arcsec,
        left_chunk_size=args.left_chunk_size,
        right_chunk_size=args.right_chunk_size,
        return_stats=True,
    )
    coarse_nside = catalogs_core._choose_default_coarse_nside(
        left_ra,
        left_dec,
        right_ra,
        right_dec,
        radius_arcsec=args.radius_arcsec,
    )
    public_result = crossmatch_sky(
        left_ra,
        left_dec,
        right_ra,
        right_dec,
        radius_arcsec=args.radius_arcsec,
        left_chunk_size=args.left_chunk_size,
        right_chunk_size=args.right_chunk_size,
    )

    _assert_same_matches(coarse_result, decl_result)
    _assert_same_matches(public_result, decl_result)

    t_decl = _time_many(
        lambda: catalogs_core._crossmatch_sky_declination_pruned(
            left_ra,
            left_dec,
            right_ra,
            right_dec,
            radius_arcsec=args.radius_arcsec,
            left_chunk_size=args.left_chunk_size,
            right_chunk_size=args.right_chunk_size,
        ),
        runs=args.runs,
    )
    t_coarse = _time_many(
        lambda: catalogs_core._crossmatch_sky_coarse_indexed(
            left_ra,
            left_dec,
            right_ra,
            right_dec,
            radius_arcsec=args.radius_arcsec,
            left_chunk_size=args.left_chunk_size,
            right_chunk_size=args.right_chunk_size,
        ),
        runs=args.runs,
    )

    rows = [
        {
            "operation": "crossmatch_declination_pruned",
            "library": "torchsky",
            "left_points": args.n_left,
            "right_points": args.n_right_band
            + args.n_right_background
            + min(max(16, args.n_left // 2), args.n_left),
            "radius_arcsec": args.radius_arcsec,
            "time_s": t_decl,
            "crossmatches_s": args.n_left / t_decl,
            "matched_rows": decl_stats.matched_rows,
            "candidate_groups": decl_stats.candidate_groups,
            "candidate_rows_considered": decl_stats.candidate_rows_considered,
            "pairs_evaluated": decl_stats.pairs_evaluated,
        },
        {
            "operation": "crossmatch_coarse_healpix",
            "library": "torchsky",
            "left_points": args.n_left,
            "right_points": args.n_right_band
            + args.n_right_background
            + min(max(16, args.n_left // 2), args.n_left),
            "radius_arcsec": args.radius_arcsec,
            "time_s": t_coarse,
            "crossmatches_s": args.n_left / t_coarse,
            "matched_rows": coarse_stats.matched_rows,
            "candidate_groups": coarse_stats.candidate_groups,
            "candidate_rows_considered": coarse_stats.candidate_rows_considered,
            "pairs_evaluated": coarse_stats.pairs_evaluated,
            "coarse_nside": coarse_nside,
            "speedup_vs_declination": t_decl / t_coarse,
            "candidate_row_reduction_vs_declination": decl_stats.candidate_rows_considered
            / max(coarse_stats.candidate_rows_considered, 1),
            "pair_reduction_vs_declination": decl_stats.pairs_evaluated
            / max(coarse_stats.pairs_evaluated, 1),
        },
    ]
    rows.extend(_recommendation_rows(rows))

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(rows, indent=2))

    print("Catalog crossmatch benchmark")
    print(
        f"  declination_pruned: {t_decl:.4f}s "
        f"pairs={decl_stats.pairs_evaluated} "
        f"candidate_rows={decl_stats.candidate_rows_considered}"
    )
    print(
        f"  coarse_healpix:     {t_coarse:.4f}s "
        f"pairs={coarse_stats.pairs_evaluated} "
        f"candidate_rows={coarse_stats.candidate_rows_considered} "
        f"nside={coarse_nside} "
        f"speedup={t_decl / t_coarse:.2f}x"
    )
    recommendation = rows[-1]
    print(
        f"  recommendation:    {recommendation['recommended_operation']} "
        f"({recommendation['note']})"
    )
    print(f"Wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
