#!/usr/bin/env python3
"""Healpy vs torchsky spin SHT replay used by bench_spin_matrix.py."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

repo_root = str(Path(__file__).resolve().parents[2])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from torchsky.sphere.spectral import alm2map_spin, map2alm_spin  # noqa: E402
from torchsky.sphere import healpix as hp  # noqa: E402

DEFAULT_CASES = (
    {"nside": 8, "lmax": 8, "mmax": 8, "nest": False},
    {"nside": 16, "lmax": 12, "mmax": 12, "nest": False},
)
EXTENDED_CASES = DEFAULT_CASES + (
    {"nside": 16, "lmax": 12, "mmax": 12, "nest": True},
    {"nside": 32, "lmax": 20, "mmax": 20, "nest": False},
    {"nside": 64, "lmax": 48, "mmax": 48, "nest": False},
)


def _median_time(fn, runs: int) -> float:
    fn()
    vals: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        vals.append(time.perf_counter() - t0)
    return float(np.median(vals))


def _rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(b))
    if denom == 0.0:
        return float(np.linalg.norm(a))
    return float(np.linalg.norm(a - b) / denom)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n-points", type=int, default=200_000)
    parser.add_argument(
        "--case-set", choices=("default", "extended"), default="extended"
    )
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--disable-gates", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("TORCHSKY_SPIN_MAP2ALM_RING_TORCH", "auto")
    os.environ.setdefault("TORCHSKY_SPIN_MAP2ALM_RING_CONCAT_CPP", "1")
    os.environ.setdefault("TORCHSKY_SPIN_MAP2ALM_RECURRENCE_CPP", "1")
    os.environ.setdefault("TORCHSKY_SPIN_ALM2MAP_RECURRENCE_CPP", "1")

    try:
        import healpy as healpy_hp
    except Exception as exc:
        print(f"healpy is required for this replay: {exc}")
        return 2

    spin = 2
    rng = np.random.default_rng(args.seed)
    cases = DEFAULT_CASES if args.case_set == "default" else EXTENDED_CASES
    rows: list[dict[str, object]] = []

    for case in cases:
        nside = int(case["nside"])
        lmax = int(case["lmax"])
        mmax = int(case["mmax"])
        nest = bool(case["nest"])
        npix = hp.nside2npix(nside)
        qu_np = rng.normal(size=(2, npix)).astype(np.float64)
        qu_t = torch.from_numpy(qu_np.copy())
        qu_hp = qu_np
        if nest:
            qu_hp = np.stack(
                [
                    healpy_hp.reorder(qu_np[0], n2r=True),
                    healpy_hp.reorder(qu_np[1], n2r=True),
                ]
            )

        def _ts_map2alm() -> tuple[torch.Tensor, torch.Tensor]:
            return map2alm_spin(
                qu_t, spin=spin, nside=nside, lmax=lmax, mmax=mmax, nest=nest
            )

        def _hp_map2alm() -> tuple[np.ndarray, np.ndarray]:
            return healpy_hp.map2alm_spin(qu_hp, spin=spin, lmax=lmax, mmax=mmax)

        ae_t, ab_t = _ts_map2alm()
        ae_h, ab_h = _hp_map2alm()
        map2alm_err = max(
            _rel_l2(ae_t.detach().cpu().numpy(), np.asarray(ae_h)),
            _rel_l2(ab_t.detach().cpu().numpy(), np.asarray(ab_h)),
        )
        t_ts = _median_time(_ts_map2alm, args.runs)
        t_hp = _median_time(_hp_map2alm, args.runs)
        rows.append(
            {
                "operation": "map2alm_spin",
                "nside": nside,
                "lmax": lmax,
                "mmax": mmax,
                "nest": nest,
                "spin": spin,
                "n_points": args.n_points,
                "rel_l2_error": map2alm_err,
                "time_torchsky_s": t_ts,
                "time_healpy_s": t_hp,
                "ratio_vs_healpy": t_hp / t_ts if t_ts > 0.0 else 0.0,
            }
        )

        ae_in = torch.from_numpy(np.asarray(ae_h)).to(torch.complex128)
        ab_in = torch.from_numpy(np.asarray(ab_h)).to(torch.complex128)

        def _ts_alm2map() -> torch.Tensor:
            return alm2map_spin(
                (ae_in, ab_in), nside=nside, spin=spin, lmax=lmax, mmax=mmax, nest=nest
            )

        def _hp_alm2map() -> np.ndarray:
            out = healpy_hp.alm2map_spin(
                [np.asarray(ae_h), np.asarray(ab_h)],
                nside=nside,
                spin=spin,
                lmax=lmax,
                mmax=mmax,
            )
            if nest:
                return np.stack(
                    [
                        healpy_hp.reorder(out[0], r2n=True),
                        healpy_hp.reorder(out[1], r2n=True),
                    ]
                )
            return np.asarray(out)

        qu_ts = _ts_alm2map().detach().cpu().numpy()
        qu_ref = _hp_alm2map()
        alm2map_err = _rel_l2(qu_ts, qu_ref)
        t_ts_i = _median_time(_ts_alm2map, args.runs)
        t_hp_i = _median_time(_hp_alm2map, args.runs)
        rows.append(
            {
                "operation": "alm2map_spin",
                "nside": nside,
                "lmax": lmax,
                "mmax": mmax,
                "nest": nest,
                "spin": spin,
                "n_points": args.n_points,
                "rel_l2_error": alm2map_err,
                "time_torchsky_s": t_ts_i,
                "time_healpy_s": t_hp_i,
                "ratio_vs_healpy": t_hp_i / t_ts_i if t_ts_i > 0.0 else 0.0,
            }
        )

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(
            {"case_set": args.case_set, "runs": args.runs, "rows": rows}, indent=2
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} rows to {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
