#!/usr/bin/env python3
"""CFHT MegaCam MEF mosaic benchmark: torchsky vs SWarp on full 36-CCD exposures."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

import bench_map_compat  # noqa: E402

from torchsky.map import ImageObservation  # noqa: E402
from torchsky.map.product import plan_map_product, stack_map_product  # noqa: E402

DATA_DIR = ROOT / "data" / "cfht" / "MegaPipe.028.181" / "g"

N_CCD = 36
CCD_H = 4644
CCD_W = 2112


def discover_exposures(data_dir: Path, max_exposures: int) -> list[Path]:
    files = sorted(data_dir.glob("*.fits*"))
    viable = [f for f in files if f.stat().st_size > 10_000_000]
    return viable[:max_exposures]


def exposure_str(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds // 60:.0f}m {seconds % 60:.0f}s"


def estimate_background_offsets(
    exposures: list[Path], n_ccd: int
) -> dict[tuple[str, int], float]:
    import torchsky
    offsets: dict[tuple[str, int], float] = {}
    for exp_path in exposures:
        for ext_idx in range(1, n_ccd + 1):
            img = torchsky.read(str(exp_path), hdu=int(ext_idx))
            offsets[(str(exp_path), ext_idx)] = float(img.median())
    return offsets


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--exposures", type=int, default=10)
    parser.add_argument("--ccds", type=int, default=36)
    parser.add_argument("--skip-swarp", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    data_dir = DATA_DIR
    if not data_dir.is_dir():
        print(f"Data dir not found: {data_dir}")
        return 1

    exposures = discover_exposures(data_dir, max_exposures=args.exposures)
    if not exposures:
        print(f"No exposures found in {data_dir}")
        return 1

    n_exp = len(exposures)
    n_obs_total = n_exp * args.ccds
    src_mpix = n_obs_total * CCD_W * CCD_H / 1e6

    print("=" * 70)
    print("CFHT MegaCam MEF Mosaic Benchmark (full pipeline)")
    print("=" * 70)
    print(f"  Exposures:    {n_exp}")
    print(f"  CCDs/exp:     {args.ccds}")
    print(f"  Total obs:    {n_obs_total}")
    print(f"  Source pix:   {src_mpix:.0f}M  ({CCD_W}x{CCD_H} per CCD)")
    print(f"  Device:       {args.device}")
    for p in exposures:
        print(f"    {p.name}  {p.stat().st_size / 1e6:.0f}MB")

    # ── Step 1: estimate per-CCD background ────────────────────────
    print("\n── Estimating per-CCD background (median) ──")
    t0 = time.perf_counter()
    bg_offsets = estimate_background_offsets(exposures, args.ccds)
    bg_elapsed = time.perf_counter() - t0
    bg_values = list(bg_offsets.values())
    print(f"  Estimated {len(bg_offsets)} CCD backgrounds in {bg_elapsed * 1000:.0f}ms")
    print(f"    min={min(bg_values):.1f}  max={max(bg_values):.1f}  "
          f"median={float(np.median(bg_values)):.1f}")

    # ── Step 2: Build observations with background offsets ──────────
    print("\n── Building observations w/ background_offset ──")
    t0 = time.perf_counter()
    obs_list = []
    for exp_path in exposures:
        for ext_idx in range(1, args.ccds + 1):
            obs_list.append(ImageObservation(
                str(exp_path), hdu=int(ext_idx),
                background_offset=bg_offsets[(str(exp_path), ext_idx)],
            ))
    print(f"  Created {len(obs_list)} observations in {(time.perf_counter() - t0) * 1000:.0f}ms")

    # ── Step 3: Plan product from union footprint ─────────────────
    print("\n── Planning map product (union footprint) ──")
    t0 = time.perf_counter()
    product = plan_map_product(
        obs_list,
        union_footprint=True,
        max_footprint_pixels=None,
    )
    nx, ny = product.shape[1], product.shape[0]
    mpix_tgt = nx * ny / 1e6
    plan_time = time.perf_counter() - t0
    print(f"  Target: {nx} x {ny}  ({mpix_tgt:.1f}M pix)")
    print(f"  Plan:   {plan_time * 1000:.0f}ms")

    # ── Step 4: torchsky coadd ────────────────────────────────────
    print("\n── torchsky coadd (stack_map_product, bilinear) ──")
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    t0 = time.perf_counter()
    result = stack_map_product(
        obs_list,
        product,
        resampling_kernel="bilinear",
        device=args.device,
    )
    ts_elapsed = time.perf_counter() - t0

    print(f"  Elapsed:    {exposure_str(ts_elapsed)}")
    print(f"  Output:     {list(result.data.shape)}  "
          f"range=[{result.data.min():.2f}, {result.data.max():.2f}]")
    print(f"  Weight:     range=[{result.weight.min():.2f}, {result.weight.max():.2f}]")
    print(f"  Throughput: {mpix_tgt / ts_elapsed:.1f} mpix/s (tgt)")
    print(f"              {src_mpix / ts_elapsed:.1f} mpix/s (src)")

    import psutil
    try:
        proc = psutil.Process()
        print(f"  RSS:        {proc.memory_info().rss / 1e9:.2f} GB")
    except ImportError:
        pass
    n_obs = len(obs_list)
    print(f"  Per obs:    {ts_elapsed * 1000 / n_obs:.1f}ms")

    # ── Step 5: SWarp coadd ─────────────────────────────────────────
    if not args.skip_swarp and bench_map_compat._find_swarp_binary() is not None:
        swarp_bin = bench_map_compat._find_swarp_binary()
        swarp_env = bench_map_compat._external_tool_env()

        # Derive SWarp parameters from the planned product WCS
        sw_crval1 = float(product.target_header["CRVAL1"])
        sw_crval2 = float(product.target_header["CRVAL2"])
        sw_pixscale = abs(float(product.target_header.get("CD1_1", product.target_header.get("CDELT1", 5e-5)))) * 3600.0
        sw_nx = product.shape[1]
        sw_ny = product.shape[0]
        proj = "TAN"

        with tempfile.TemporaryDirectory(prefix="ts_swarp_") as tmpdir:
            tmp = Path(tmpdir)

            # Decompress
            decompressed: list[str] = []
            print(f"\n── SWarp: decompressing {n_exp} fpack files ──")
            t0 = time.perf_counter()
            for exp_path in exposures:
                out = tmp / f"{exp_path.stem}.fits"
                r = subprocess.run(
                    ["funpack", "-O", str(out), str(exp_path)],
                    capture_output=True, text=True, env=swarp_env,
                )
                if r.returncode != 0:
                    print(f"  funpack failed: {r.stderr[:200]}")
                    return 1
                decompressed.append(str(out))
            decomp_time = time.perf_counter() - t0
            print(f"  Decompressed {n_exp} files in {decomp_time:.1f}s")

            def _run_swarp(subtract_back: str) -> tuple[float, np.ndarray | None]:
                cfg = tmp / f"swarp_bg{subtract_back}.conf"
                imgout = tmp / f"coadd_bg{subtract_back}.fits"
                lines = [
                    f"IMAGEOUT_NAME          {imgout}",
                    "WEIGHT_TYPE            NONE",
                    "COMBINE                Y",
                    "COMBINE_TYPE           MEDIAN",
                    "CELESTIAL_TYPE         EQUATORIAL",
                    f"PROJECTION_TYPE        {proj}",
                    "PROJECTION_ERR         0.0",
                    "CENTER_TYPE            MANUAL",
                    f"CENTER                 {sw_crval1:.10f}, {sw_crval2:.10f}",
                    "PIXELSCALE_TYPE        MANUAL",
                    f"PIXEL_SCALE            {sw_pixscale:.10f}",
                    f"IMAGE_SIZE             {sw_nx},{sw_ny}",
                    "RESAMPLE               Y",
                    "RESAMPLING_TYPE        BILINEAR",
                    "INTERPOLATE            N",
                    "FSCALASTRO_TYPE        NONE",
                    f"SUBTRACT_BACK          {subtract_back}",
                    "WRITE_XML              N",
                    "VERBOSE_TYPE           QUIET",
                    "NTHREADS               0",
                    "DELETE_TMPFILES        Y",
                ]
                cfg.write_text("\n".join(lines) + "\n")
                cmd = [str(swarp_bin), "-c", str(cfg), *decompressed]
                label = f"SUBTRACT_BACK={subtract_back}"
                t0 = time.perf_counter()
                r = subprocess.run(cmd, capture_output=True, text=True, env=swarp_env)
                elapsed = time.perf_counter() - t0
                if r.returncode != 0:
                    print(f"  SWarp ({label}) failed (exit {r.returncode})")
                    for line in r.stderr.splitlines()[-5:]:
                        print(f"    > {line}")
                    return elapsed, None
                if not imgout.exists():
                    print(f"  SWarp ({label}) no output")
                    return elapsed, None
                from astropy.io import fits as afits
                with afits.open(str(imgout)) as hdul:
                    data = hdul[0].data.astype(np.float32)
                return elapsed, data

            for sb in ("N", "Y"):
                print(f"\n── SWarp coadd (SUBTRACT_BACK={sb}) ──")
                sw_elapsed, sw_data = _run_swarp(sb)
                if sw_data is None:
                    continue
                print(f"  Elapsed:    {exposure_str(sw_elapsed)} (incl decomp {decomp_time:.1f}s)")
                print(f"  Pure:       {exposure_str(sw_elapsed - decomp_time)}")
                print(f"  Output:     {list(sw_data.shape)}  "
                      f"range=[{sw_data.min():.2f}, {sw_data.max():.2f}]")

                if sw_data.shape == result.data.shape[-2:]:
                    mask = (result.weight.cpu().numpy()[0] > 0) & (sw_data > 0)
                    if mask.any():
                        diff = result.data.cpu().numpy() - sw_data
                        rd = diff[mask]
                        print("  Quality:")
                        print(f"    max_diff: {float(np.abs(rd).max()):.2f}")
                        print(f"    rmse:     {float(np.sqrt((rd**2).mean())):.2f}")
                        print(f"    median:   {float(np.median(rd)):.2f}")
                        print(f"    bias:     {float(np.mean(rd)):.2f}")

                pure = sw_elapsed - decomp_time
                ratio = pure / ts_elapsed if ts_elapsed > 0 else float("inf")
                print(f"  Speed:      {ratio:.1f}x SWarp/torchsky (excl decomp)",
                      f"or {sw_elapsed / ts_elapsed:.1f}x (incl)")
    else:
        print("\n── SWarp skipped ──")

    # ── Summary ─────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("Summary")
    print("=" * 70)
    print(f"  Exposures: {n_exp}")
    print(f"  CCDs:      {args.ccds} per exp ({n_obs_total} total)")
    print(f"  Source:    {src_mpix:.0f}M pixels")
    print(f"  Target:    {nx} x {ny} ({mpix_tgt:.1f}M pixels)")
    print(f"  torchsky:  {exposure_str(ts_elapsed)} ({mpix_tgt / ts_elapsed:.1f} mpix/s)")


if __name__ == "__main__":
    sys.exit(main())
