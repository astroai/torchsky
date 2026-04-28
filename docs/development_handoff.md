# Development Handoff

This repo checkpoint is intended to be portable to a fresh development machine.

## What Is Intentionally Not Committed

- large downloaded real-data inputs under `data/`
- generated benchmark artifacts under `data/benchmark_results/`
- temporary staging files under `data/_tmp/`

The repo now ignores `data/`, so a fresh clone should recreate those locally.

## Main Local Data To Recreate

### SMACS J0723 public real-data set

Download the public references plus public JWST inputs:

```bash
cd "$(git rev-parse --show-toplevel)"
pixi run -e test python scripts/download_smacs0723_multiband_public.py \
  --bands F090W,F150W,F200W,F277W,F356W,F444W \
  --module b \
  --updec-iterations 0 \
  --jwst-output-dir data/jwst_smacs0723 \
  --updec-output-dir data/updec_webb
```

If a band needs to be refilled individually:

```bash
cd "$(git rev-parse --show-toplevel)"
pixi run -e test python scripts/download_jwst_smacs0723_mast.py \
  --band F277W \
  --module b \
  --subgroup CAL \
  --output-dir data/jwst_smacs0723
```

### UPdec-Webb smoke references

```bash
cd "$(git rev-parse --show-toplevel)"
pixi run -e test python scripts/download_updec_webb.py \
  --bands F090W,F150W,F200W,F277W,F356W,F444W \
  --iterations 0,10 \
  --output-dir data/updec_webb
```

## Main Real-Data Benchmarks

### Non-INR multiband stacker sweep

```bash
cd "$(git rev-parse --show-toplevel)"
PYTHONPATH=src:benchmarks pixi run -e map-compat python benchmarks/bench_map_smacs0723_stackers_band_sweep.py \
  --bands F090W,F150W,F200W,F277W,F356W,F444W \
  --module b \
  --input-product cal \
  --crop-size 256 \
  --crop-mode best_overlap \
  --json-out data/benchmark_results/smacs0723_stackers_band_sweep_full_public_256.json \
  --markdown-out data/benchmark_results/smacs0723_stackers_band_sweep_full_public_256.md
```

### Real single-band field fit

```bash
cd "$(git rev-parse --show-toplevel)"
PYTHONPATH=src:benchmarks pixi run -e test python benchmarks/bench_map_smacs0723_field_fit.py \
  --band-name F277W \
  --input-product cal \
  --filename-contains nrcblong \
  --crop-size 128 \
  --crop-mode best_overlap \
  --steps 4
```

### Real multiband joint field fit

```bash
cd "$(git rev-parse --show-toplevel)"
PYTHONPATH=src:benchmarks pixi run -e test python benchmarks/bench_map_smacs0723_field_fit_joint.py \
  --bands F090W,F277W,F356W,F444W \
  --crop-size 32 \
  --crop-strategy best_overlap \
  --steps 4
```

## Current Known Caveats

- Real-data field-fit debug runs can still be expensive.
- The `exact_wavelength` control mode exists for single-band and band-sweep real-data field-fit benchmarks.
- Temporary aux-file staging for real-data field-fit now avoids repo-external temp paths, but local free disk still matters.
- Real-data benchmark artifacts should be treated as local products, not committed truth.
