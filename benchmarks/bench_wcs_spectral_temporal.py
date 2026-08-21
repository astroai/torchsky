#!/usr/bin/env python3
"""Comparator-only spectral/temporal WCS benchmark against Astropy/WCSLIB.

Astropy/WCSLIB is used here only as a reference and timing comparator. The
torchsky runtime must not import or fall back to Astropy/WCSLIB for these paths.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from astropy.time import Time
from astropy.wcs import WCS as AstropyWCS

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmarks.bench_wcs import _resolve_device, _sync  # noqa: E402
from torchsky.wcs.core import WCS as TorchWCS  # noqa: E402
from torchsky.wcs.temporal import (
    SECONDS_PER_DAY,
    TAI_MINUS_GPS_SECONDS,
    convert_time_scale,
)  # noqa: E402


SPECTRAL_CASES = (
    "FREQ",
    "WAVE",
    "WAVN",
    "FREQ-LOG",
    "WAVE-LOG",
    "WAVN-LOG",
    "FREQ-F2F",
    "FREQ-W2F",
    "FREQ-A2F",
    "FREQ-V2F",
    "WAVE-F2W",
    "WAVE-W2W",
    "WAVE-A2W",
    "WAVE-V2W",
    "AWAV-F2A",
    "AWAV-W2A",
    "AWAV-A2A",
    "AWAV-V2A",
    "VELO-F2V",
    "VELO-W2V",
    "VELO-A2V",
    "VELO-V2V",
    "WAVE-GRI",
    "WAVE-GRA",
    "FREQ-GRA",
)
TEMPORAL_CASES = (
    ("UTC", "TAI"),
    ("UTC", "TT"),
    ("TAI", "UTC"),
    ("TT", "UTC"),
    ("GPS", "UTC"),
)


def _time_median(
    fn: Callable[[], Any], runs: int, device: torch.device | None = None
) -> tuple[float, Any]:
    result = fn()
    if device is not None:
        _sync(device)
    timings: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        result = fn()
        if device is not None:
            _sync(device)
        timings.append(time.perf_counter() - start)
    return statistics.median(timings), result


def _spectral_header(ctype: str) -> dict:
    header = {}
    header["NAXIS"] = 1
    header["NAXIS1"] = 4096
    header["CTYPE1"] = ctype
    header["CRPIX1"] = 1.0
    if ctype in {"WAVE-GRI", "WAVE-GRA"}:
        header["CRVAL1"] = 5.0e-7
        header["CDELT1"] = 1.0e-9
        header["CUNIT1"] = "m"
        header["PV1_0"] = 300000.0
        header["PV1_1"] = 1.0
        header["PV1_2"] = 15.0
        header["PV1_3"] = 1.5
        header["PV1_4"] = -1.0e5
        header["PV1_5"] = 0.0
        header["PV1_6"] = 10.0
    elif ctype == "FREQ-GRA":
        header["CRVAL1"] = 6.0e14
        header["CDELT1"] = 1.0e9
        header["CUNIT1"] = "Hz"
        header["PV1_0"] = 300000.0
        header["PV1_1"] = 1.0
        header["PV1_2"] = 15.0
        header["PV1_3"] = 1.5
        header["PV1_4"] = -1.0e5
        header["PV1_5"] = 0.0
        header["PV1_6"] = 10.0
    elif (
        "-" in ctype
        and len(ctype.rsplit("-", 1)[1]) == 3
        and ctype.rsplit("-", 1)[1][1] == "2"
    ):
        stype = ctype.split("-", 1)[0]
        header["RESTFRQ"] = 1.42040575177e9
        if stype == "FREQ":
            header["CRVAL1"] = 1.420e9
            header["CDELT1"] = 1.0e5
            header["CUNIT1"] = "Hz"
        elif stype in {"WAVE", "AWAV"}:
            header["CRVAL1"] = 0.21
            header["CDELT1"] = 1.0e-4
            header["CUNIT1"] = "m"
        elif stype == "VELO":
            header["CRVAL1"] = 10.0
            header["CDELT1"] = 0.1
            header["CUNIT1"] = "m/s"
    else:
        header["CRVAL1"] = 10.0
        header["CDELT1"] = 1.0e-3
        header["CUNIT1"] = "Hz" if ctype.startswith("FREQ") else "m"
        if ctype.startswith("WAVN"):
            header["CUNIT1"] = "m-1"
    return header


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _spectral_pixels(ctype: str, n_points: int) -> np.ndarray:
    upper = 10.0 if ctype.endswith("-LOG") else 1000.0
    return np.linspace(0.0, upper, n_points, dtype=np.float64).reshape(-1, 1)


def _run_spectral_case(
    ctype: str, pixels: np.ndarray, runs: int, device: torch.device
) -> list[dict[str, Any]]:
    header = _spectral_header(ctype)
    torch_wcs = TorchWCS(dict(header)).to(device)
    astropy_wcs = AstropyWCS(header)
    pixel_tensor = torch.from_numpy(pixels).to(device)

    torch_forward_s, torch_world = _time_median(
        lambda: torch_wcs.pixel_to_world(pixel_tensor, origin=0), runs, device
    )
    astropy_forward_s, astropy_world = _time_median(
        lambda: astropy_wcs.all_pix2world(pixels, 0), runs
    )

    torch_world_np = _as_numpy(torch_world).reshape(-1, 1)
    astropy_world_np = np.asarray(astropy_world, dtype=np.float64).reshape(-1, 1)
    world_err = float(np.nanmax(np.abs(torch_world_np - astropy_world_np)))

    world_tensor = torch.from_numpy(astropy_world_np).to(device)
    torch_inverse_s, torch_pixels = _time_median(
        lambda: torch_wcs.world_to_pixel(world_tensor, origin=0), runs, device
    )
    astropy_inverse_s, astropy_pixels = _time_median(
        lambda: astropy_wcs.all_world2pix(astropy_world_np, 0), runs
    )

    torch_pixels_np = _as_numpy(torch_pixels).reshape(-1, 1)
    astropy_pixels_np = np.asarray(astropy_pixels, dtype=np.float64).reshape(-1, 1)
    pixel_err = float(np.nanmax(np.abs(torch_pixels_np - astropy_pixels_np)))

    n = int(pixels.shape[0])
    return [
        {
            "domain": "wcs_spectral",
            "case": ctype,
            "direction": "pixel_to_world",
            "n_points": n,
            "device": str(device),
            "torchsky_s": torch_forward_s,
            "wcslib_s": astropy_forward_s,
            "torchsky_mpts_s": n / torch_forward_s / 1.0e6,
            "wcslib_mpts_s": n / astropy_forward_s / 1.0e6,
            "speedup_vs_wcslib": astropy_forward_s / torch_forward_s,
            "max_abs_error": world_err,
            "error_unit": str(header["CUNIT1"]),
        },
        {
            "domain": "wcs_spectral",
            "case": ctype,
            "direction": "world_to_pixel",
            "n_points": n,
            "device": str(device),
            "torchsky_s": torch_inverse_s,
            "wcslib_s": astropy_inverse_s,
            "torchsky_mpts_s": n / torch_inverse_s / 1.0e6,
            "wcslib_mpts_s": n / astropy_inverse_s / 1.0e6,
            "speedup_vs_wcslib": astropy_inverse_s / torch_inverse_s,
            "max_abs_error": pixel_err,
            "error_unit": "pixel",
        },
    ]


def _time_values(scale: str, base_utc_mjd: np.ndarray) -> np.ndarray:
    if scale == "UTC":
        return base_utc_mjd
    if scale == "GPS":
        tai = Time(base_utc_mjd, format="mjd", scale="utc").tai.mjd
        return np.asarray(
            tai - TAI_MINUS_GPS_SECONDS / SECONDS_PER_DAY, dtype=np.float64
        )
    return np.asarray(
        getattr(Time(base_utc_mjd, format="mjd", scale="utc"), scale.lower()).mjd,
        dtype=np.float64,
    )


def _astropy_time_scale(
    values: np.ndarray, from_scale: str, to_scale: str
) -> np.ndarray:
    if from_scale == "GPS":
        tai_mjd = values + TAI_MINUS_GPS_SECONDS / SECONDS_PER_DAY
        time_value = Time(tai_mjd, format="mjd", scale="tai")
    else:
        time_value = Time(values, format="mjd", scale=from_scale.lower())
    if to_scale == "GPS":
        return np.asarray(
            time_value.tai.mjd - TAI_MINUS_GPS_SECONDS / SECONDS_PER_DAY,
            dtype=np.float64,
        )
    return np.asarray(getattr(time_value, to_scale.lower()).mjd, dtype=np.float64)


def _run_temporal_case(
    from_scale: str,
    to_scale: str,
    base_utc_mjd: np.ndarray,
    runs: int,
    device: torch.device,
) -> dict[str, Any]:
    values = _time_values(from_scale, base_utc_mjd)
    values_tensor = torch.from_numpy(values).to(device)

    torch_s, torch_out = _time_median(
        lambda: convert_time_scale(values_tensor, from_scale, to_scale), runs, device
    )
    astropy_s, astropy_out = _time_median(
        lambda: _astropy_time_scale(values, from_scale, to_scale),
        runs,
    )

    torch_np = _as_numpy(torch_out)
    astropy_np = np.asarray(astropy_out, dtype=np.float64)
    error = float(np.nanmax(np.abs(torch_np - astropy_np)))
    n = int(values.shape[0])
    return {
        "domain": "wcs_temporal",
        "case": f"{from_scale}->{to_scale}",
        "direction": "scale_conversion",
        "n_points": n,
        "device": str(device),
        "torchsky_s": torch_s,
        "wcslib_s": astropy_s,
        "torchsky_mpts_s": n / torch_s / 1.0e6,
        "wcslib_mpts_s": n / astropy_s / 1.0e6,
        "speedup_vs_wcslib": astropy_s / torch_s,
        "max_abs_error": error,
        "error_unit": "day",
    }


def _print_summary(rows: list[dict[str, Any]]) -> None:
    print(
        "| Domain | Case | Direction | N | torchsky | WCSLIB/Astropy | Speedup | Max error |"
    )
    print("|---|---|---|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            "| {domain} | {case} | {direction} | {n_points} | {torchsky_s:.6g}s | "
            "{wcslib_s:.6g}s | {speedup_vs_wcslib:.3g}x | {max_abs_error:.3g} {error_unit} |".format(
                **row
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-points", type=int, default=200_000)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--spectral-cases",
        default=",".join(SPECTRAL_CASES),
        help="Comma-separated spectral CTYPE cases, or empty to skip spectral benchmarks.",
    )
    parser.add_argument(
        "--skip-temporal",
        action="store_true",
        help="Skip temporal scale-conversion benchmarks.",
    )
    args = parser.parse_args()
    device = _resolve_device(args.device)

    rows: list[dict[str, Any]] = []

    spectral_cases = [
        c.strip().upper() for c in args.spectral_cases.split(",") if c.strip()
    ]
    for ctype in spectral_cases:
        pixels = _spectral_pixels(ctype, args.n_points)
        rows.extend(_run_spectral_case(ctype, pixels, args.runs, device))

    if not args.skip_temporal:
        base_utc_mjd = np.linspace(57204.0, 61049.0, args.n_points, dtype=np.float64)
        for from_scale, to_scale in TEMPORAL_CASES:
            rows.append(
                _run_temporal_case(
                    from_scale, to_scale, base_utc_mjd, args.runs, device
                )
            )

    _print_summary(rows)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
