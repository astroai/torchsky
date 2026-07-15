#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/arc/projects/ots/jwst_smacs0723}"
INPUT_DIR="${INPUT_DIR:-${PROJECT_ROOT}/input}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_ROOT}/results}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "${INPUT_DIR}" "${RESULTS_DIR}"

# 1. Verify existing CRFs, organize them by band, and download missing public CRFs.
pixi run -e test -- python scripts/prepare_smacs0723_multiband_crf.py \
  --project-root "${PROJECT_ROOT}" \
  --input-dir "${INPUT_DIR}" \
  --results-dir "${RESULTS_DIR}" \
  --bands F090W,F150W,F200W,F277W,F356W,F444W \
  --module b \
  --adopt-mode symlink \
  --strict

# 2. Train the planar-Fourier model and the spherical-encoding comparison model.
pixi run -e test -- python scripts/smacs0723_wcs_aware_multiband_inr.py \
  --exposure-dir "${INPUT_DIR}" \
  --output-dir "${RESULTS_DIR}/wcs_aware_multiband_inr" \
  --bands F090W,F150W,F200W,F277W,F356W,F444W \
  --run-both-encodings \
  --device "${DEVICE}" \
  --steps "${STEPS:-10000}" \
  --batch-size "${BATCH_SIZE:-16384}" \
  --max-pixels-per-exposure "${MAX_PIXELS_PER_EXPOSURE:-200000}" \
  --bandpass-samples "${BANDPASS_SAMPLES:-5}" \
  --psf-mode none
