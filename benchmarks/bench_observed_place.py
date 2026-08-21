#!/usr/bin/env python3
"""Pinned-EOP observed-place timing vs ERFA (CIRS/TETE)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import erfa  # noqa: E402

from torchsky.wcs import EarthOrientation, Site, transform  # noqa: E402
from torchsky.wcs.frames import ARCSEC_TO_RAD  # noqa: E402
from torchsky.wcs.iau2006 import cirs_matrix, npb_matrix  # noqa: E402

UTC_MJD = 60000.0
DUT1_S = 0.184
TT_MINUS_UTC_S = 69.184
TT_MJD = UTC_MJD + TT_MINUS_UTC_S / 86400.0
JD0 = 2400000.5
MK_LON_DEG = -155.5763
MK_LAT_DEG = 19.8260
MK_HEIGHT_M = 4205.0


def _lonlat_to_xyz(lon_deg: torch.Tensor, lat_deg: torch.Tensor) -> torch.Tensor:
    lon = lon_deg * (np.pi / 180.0)
    lat = lat_deg * (np.pi / 180.0)
    clat = torch.cos(lat)
    return torch.stack(
        (clat * torch.cos(lon), clat * torch.sin(lon), torch.sin(lat)), dim=-1
    )


def _sep_arcsec(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    chord = torch.linalg.vector_norm(a - b, dim=-1)
    return (
        2.0 * torch.asin((chord / 2.0).clamp(max=1.0)) / ARCSEC_TO_RAD
    ).detach().cpu().numpy()


def _median_ms(fn: Any, runs: int, device: torch.device) -> float:
    fn()
    _sync(device)
    samples: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        samples.append((time.perf_counter() - t0) * 1.0e3)
    return float(np.median(samples))


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if choice == "mps" and (
        not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS requested but unavailable")
    return torch.device(choice)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-points", type=int, default=4096)
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu"
    )
    args = parser.parse_args()

    device = _resolve_device(args.device)
    dtype = torch.float32 if device.type == "mps" else torch.float64
    eop = EarthOrientation(dut1_s=DUT1_S, tt_minus_utc_s=TT_MINUS_UTC_S)
    site = Site(lon_deg=MK_LON_DEG, lat_deg=MK_LAT_DEG, height_m=MK_HEIGHT_M)
    ra = torch.linspace(0.0, 359.0, args.n_points, dtype=dtype, device=device)
    dec = torch.linspace(-40.0, 40.0, args.n_points, dtype=dtype, device=device)
    ra_chk = torch.tensor([10.0, 83.6331, 201.2983, 359.0], dtype=torch.float64)
    dec_chk = torch.tensor([20.0, 22.0145, -43.0191, -20.0], dtype=torch.float64)

    lon_c, lat_c = transform(
        ra_chk,
        dec_chk,
        from_frame="ICRS",
        to_frame="CIRS",
        obstime_mjd_utc=UTC_MJD,
        eop=eop,
    )
    lon_t, lat_t = transform(
        ra_chk,
        dec_chk,
        from_frame="ICRS",
        to_frame="TETE",
        obstime_mjd_utc=UTC_MJD,
        eop=eop,
    )
    xyz = _lonlat_to_xyz(ra_chk, dec_chk)
    r_cirs = torch.as_tensor(erfa.c2i06a(JD0, TT_MJD), dtype=torch.float64)
    r_tete = torch.as_tensor(erfa.pnm06a(JD0, TT_MJD), dtype=torch.float64)
    sep_cirs = float(
        _sep_arcsec(
            _lonlat_to_xyz(lon_c.cpu().double(), lat_c.cpu().double()),
            (r_cirs @ xyz.T).T,
        ).max()
    )
    sep_tete = float(
        _sep_arcsec(
            _lonlat_to_xyz(lon_t.cpu().double(), lat_t.cpu().double()),
            (r_tete @ xyz.T).T,
        ).max()
    )

    def _run_frame(to_frame: str) -> tuple[torch.Tensor, torch.Tensor]:
        kwargs: dict[str, Any] = {
            "from_frame": "ICRS",
            "to_frame": to_frame,
            "obstime_mjd_utc": UTC_MJD,
            "eop": eop,
        }
        if to_frame == "ALTAZ":
            kwargs["site"] = site
            kwargs["pn"] = "iau2006"
        return transform(ra, dec, **kwargs)

    payload: dict[str, Any] = {
        "n_points": args.n_points,
        "device": str(device),
        "utc_mjd": UTC_MJD,
        "dut1_s": DUT1_S,
        "tt_minus_utc_s": TT_MINUS_UTC_S,
        "max_sep_cirs_arcsec": sep_cirs,
        "max_sep_tete_arcsec": sep_tete,
        "npb_resid_absmax": float(
            np.max(
                np.abs(
                    npb_matrix(torch.tensor(TT_MJD, dtype=torch.float64)).cpu().numpy()
                    - erfa.pnm06a(JD0, TT_MJD)
                )
            )
        ),
        "cirs_resid_absmax": float(
            np.max(
                np.abs(
                    cirs_matrix(torch.tensor(TT_MJD, dtype=torch.float64)).cpu().numpy()
                    - erfa.c2i06a(JD0, TT_MJD)
                )
            )
        ),
        "median_ms_gcrs": _median_ms(lambda: _run_frame("GCRS"), args.runs, device),
        "median_ms_cirs": _median_ms(lambda: _run_frame("CIRS"), args.runs, device),
        "median_ms_tete": _median_ms(lambda: _run_frame("TETE"), args.runs, device),
        "median_ms_altaz": _median_ms(lambda: _run_frame("ALTAZ"), args.runs, device),
    }
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
