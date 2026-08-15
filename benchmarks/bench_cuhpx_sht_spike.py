#!/usr/bin/env python3
"""CUDA-only cuHPX scalar SHT spike. Not a core backend and not a science oracle.

Compares scalar round-trip at matched lmax conventions separately:
``2*nside+1`` (cuHPX default) and ``3*nside-1`` (healpy default). cuHPX is
float32, real scalar, no spin-2. Exit 0 always writes a verdict JSON.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from torchsky.sphere.spectral import alm2map, map2alm
from torchsky.sphere import healpix as hp


def _median_time(fn, runs: int) -> float:
    fn()
    vals: list[float] = []
    for _ in range(max(1, runs)):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        vals.append(time.perf_counter() - t0)
    return float(np.median(vals))


def _rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(b))
    if denom == 0.0:
        return float(np.linalg.norm(a))
    return float(np.linalg.norm(a - b) / denom)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nside", type=int, default=32)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("benchmarks_results/cuhpx_sht_spike.json"),
    )
    args = parser.parse_args()

    payload: dict[str, object] = {
        "nside": args.nside,
        "cuda": bool(torch.cuda.is_available()),
        "cuhpx_import": False,
        "has_spin": False,
        "verdict": "no-go",
        "reason": "",
        "rows": [],
        "autograd": False,
    }

    if not torch.cuda.is_available():
        payload["reason"] = "CUDA not available; cuHPX is CUDA-only"
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return 0

    try:
        from cuhpx import SHTCUDA, iSHTCUDA
    except Exception as exc:
        payload["reason"] = f"cuhpx import failed: {exc}"
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return 0

    payload["cuhpx_import"] = True
    nside = int(args.nside)
    npix = hp.nside2npix(nside)
    rng = np.random.default_rng(args.seed)
    m_np = rng.normal(size=npix).astype(np.float64)
    m_cpu = torch.from_numpy(m_np)
    m_cuda32 = torch.from_numpy(m_np.astype(np.float32)).to("cuda")
    healpy_hp = None
    try:
        import healpy as healpy_hp
    except Exception:
        healpy_hp = None

    lmax_cases = (
        ("cuhpx_default", 2 * nside + 1),
        ("healpy_default", 3 * nside - 1),
    )
    rows: list[dict[str, object]] = []
    autograd_ok = False
    roundtrip_ok = True

    for label, lmax in lmax_cases:
        row: dict[str, object] = {"lmax_label": label, "lmax": lmax}
        try:
            sht = SHTCUDA(nside, lmax=lmax, mmax=lmax, quad_weights="ring")
            isht = iSHTCUDA(nside, lmax=lmax, mmax=lmax)
            coeff = sht(m_cuda32)
            rec = isht(coeff)
            rec_np = rec.detach().float().cpu().numpy().reshape(-1)
            row["cuhpx_roundtrip_rel_l2"] = _rel_l2(rec_np, m_np.astype(np.float32))
            row["cuhpx_map2alm_s"] = _median_time(lambda: sht(m_cuda32), args.runs)
            row["cuhpx_alm2map_s"] = _median_time(lambda: isht(coeff), args.runs)
        except Exception as exc:
            row["cuhpx_error"] = str(exc)
            roundtrip_ok = False

        try:
            alm_ts = map2alm(m_cpu, nside=nside, lmax=lmax)
            rec_ts = alm2map(alm_ts, nside=nside, lmax=lmax).detach().cpu().numpy()
            row["torchsky_roundtrip_rel_l2"] = _rel_l2(rec_ts, m_np)
            row["torchsky_map2alm_s"] = _median_time(
                lambda: map2alm(m_cpu, nside=nside, lmax=lmax), args.runs
            )
        except Exception as exc:
            row["torchsky_error"] = str(exc)

        if healpy_hp is not None:
            try:
                alm_hp = healpy_hp.map2alm(m_np, lmax=lmax, iter=0)
                rec_hp = healpy_hp.alm2map(alm_hp, nside=nside, lmax=lmax)
                row["healpy_roundtrip_rel_l2"] = _rel_l2(np.asarray(rec_hp), m_np)
                row["healpy_map2alm_s"] = _median_time(
                    lambda: healpy_hp.map2alm(m_np, lmax=lmax, iter=0), args.runs
                )
            except Exception as exc:
                row["healpy_error"] = str(exc)

        cuhpx_rt = row.get("cuhpx_roundtrip_rel_l2")
        if (
            not isinstance(cuhpx_rt, float)
            or not np.isfinite(cuhpx_rt)
            or cuhpx_rt > 1.0e-2
        ):
            roundtrip_ok = False
        rows.append(row)

    try:
        sht0 = SHTCUDA(
            nside, lmax=2 * nside + 1, mmax=2 * nside + 1, quad_weights="ring"
        )
        sig = m_cuda32.detach().clone().requires_grad_(True)
        out = sht0(sig)
        torch.as_tensor(out).real.sum().backward()
        autograd_ok = sig.grad is not None and bool(torch.isfinite(sig.grad).all())
    except Exception as exc:
        payload["autograd_error"] = str(exc)
        autograd_ok = False

    payload["rows"] = rows
    payload["autograd"] = autograd_ok
    if roundtrip_ok and autograd_ok:
        payload["verdict"] = "go-scalar-cuda"
        payload["reason"] = (
            "Scalar CUDA SHT round-trip and autograd succeeded at matched lmax; "
            "still not a default backend and still not a spin/oracle path"
        )
    else:
        payload["verdict"] = "no-go"
        payload["reason"] = (
            "cuHPX spike failed round-trip, autograd, or import checks; "
            "leave as a documented comparator next to s2fft"
        )

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
