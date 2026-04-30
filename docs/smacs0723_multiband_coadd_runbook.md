# SMACS J0723 Multiband Coadd Runbook

This is the quick path to rerun the full non-INR coadd/stacker baseline once all public downloads have finished.

The commands below assume:
- repo root: `.`
- public data under `data`

**Scratch / shared filesystem:** set `TORCHSKY_DATA` to a directory with the same layout as `data/` (`jwst_smacs0723/`, `updec_webb/`, optional `benchmark_results/`). All `bench_map_smacs0723_*.py` scripts read this root from the environment when set.
- module-`b` public JWST inputs
- published UPdec `iter0` products as the reference artifacts

## 1. Check That Downloads Are Complete

For the current public module-`b` set, the expected counts are:
- `F090W`: `36`
- `F150W`: `36`
- `F200W`: `36`
- `F277W`: `9`
- `F356W`: `9`
- `F444W`: `9`

Run:

```bash
cd "$(git rev-parse --show-toplevel)"
pixi run -e test python - <<'PY'
from pathlib import Path
from benchmarks.bench_map_smacs0723_real import _discover_exposure_paths

base = Path("data/jwst_smacs0723")
expected = {
    "F090W": 36,
    "F150W": 36,
    "F200W": 36,
    "F277W": 9,
    "F356W": 9,
    "F444W": 9,
}

for band, want in expected.items():
    contains = "nrcblong" if band in {"F277W", "F356W", "F444W"} else ""
    paths = _discover_exposure_paths(
        base,
        band=band,
        module="b",
        input_product="cal",
        filename_contains=contains,
    )
    have = len(paths)
    status = "OK" if have == want else "MISSING"
    print(f"{band}: {have}/{want} {status}")
PY
```

If any row says `MISSING`, finish the downloads first.

## 2. Finish Any Missing Downloads

Full multiband public downloader:

```bash
cd "$(git rev-parse --show-toplevel)"
pixi run -e test python scripts/download_smacs0723_multiband_public.py \
  --bands F090W,F150W,F200W,F277W,F356W,F444W \
  --module b \
  --updec-iterations 0 \
  --jwst-output-dir data/jwst_smacs0723 \
  --updec-output-dir data/updec_webb
```

If only one band is incomplete, a targeted fill command is often faster:

```bash
cd "$(git rev-parse --show-toplevel)"
pixi run -e test python scripts/download_jwst_smacs0723_mast.py \
  --band F150W \
  --module b \
  --subgroup CAL \
  --max-exposures 100 \
  --output-dir data/jwst_smacs0723
```

Swap `F150W` for any missing band.

## 3. Run The Full Sequential Non-INR Sweep

This runs the legacy stackers plus the `torchsky` non-INR stackers band by band:

```bash
cd "$(git rev-parse --show-toplevel)"
PYTHONPATH=src:benchmarks pixi run -e map-compat python benchmarks/bench_map_smacs0723_stackers_band_sweep.py \
  --bands F090W,F150W,F200W,F277W,F356W,F444W \
  --module b \
  --input-product cal \
  --crop-size 256 \
  --crop-mode best_overlap \
  --skip-missing \
  --json-out data/benchmark_results/smacs0723_stackers_band_sweep_full_public_256.json \
  --markdown-out data/benchmark_results/smacs0723_stackers_band_sweep_full_public_256.md
```

Notes:
- `best_overlap` is the recommended real-data crop mode across bands.
- `256` is the safer size if you want a stable, low-memory full sweep.
- If you want a more detailed pass later, rerun with `--crop-size 512`.

## 4. Optional Higher-Resolution Pass

```bash
cd "$(git rev-parse --show-toplevel)"
PYTHONPATH=src:benchmarks pixi run -e map-compat python benchmarks/bench_map_smacs0723_stackers_band_sweep.py \
  --bands F090W,F150W,F200W,F277W,F356W,F444W \
  --module b \
  --input-product cal \
  --crop-size 512 \
  --crop-mode best_overlap \
  --skip-missing \
  --json-out data/benchmark_results/smacs0723_stackers_band_sweep_full_public_512.json \
  --markdown-out data/benchmark_results/smacs0723_stackers_band_sweep_full_public_512.md
```

## 5. Expected Output Files

Main outputs:
- `data/benchmark_results/smacs0723_stackers_band_sweep_full_public_256.json`
- `data/benchmark_results/smacs0723_stackers_band_sweep_full_public_256.md`

Optional higher-resolution outputs:
- `data/benchmark_results/smacs0723_stackers_band_sweep_full_public_512.json`
- `data/benchmark_results/smacs0723_stackers_band_sweep_full_public_512.md`

## 6. What This Covers

This sweep is intended to validate the non-INR baseline stack pipeline before multiband INR work.

It covers:
- legacy stackers
- `torchsky` drizzle-like stackers
- projection/config variants already wired into the benchmark harness
- all public module-`b` bands in the current SMACS J0723 setup

It intentionally excludes INR rows.
