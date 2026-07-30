#!/usr/bin/env bash
set -euo pipefail

mode=${1:?usage: run_canfar_adass_depth.sh MODE RUN_ID}
run_id=${2:?usage: run_canfar_adass_depth.sh MODE RUN_ID}
commit=${TORCHSKY_COMMIT:?set TORCHSKY_COMMIT to an exact pushed SHA}
container_image=${CONTAINER_IMAGE:-images.canfar.net/astroai/notebook:latest}
artifact_root=/arc/projects/ots/torchsky-runs/${run_id}
source_root=/scratch/${run_id}-src
data_root=/arc/projects/ots/torchsky-data

case "${container_image}" in
  images.canfar.net/astroai/* | astroai/*) ;;
  *)
    echo "CONTAINER_IMAGE must name an AstroAI image, got: ${container_image}" >&2
    exit 2
    ;;
esac

mkdir -p "${artifact_root}"
exec > >(tee "${artifact_root}/session.log") 2>&1

echo "run_id=${run_id}"
echo "mode=${mode}"
echo "requested_commit=${commit}"
echo "image=${container_image}"
echo "started_utc=$(date -u +%FT%TZ)"
nvidia-smi || true

rm -rf "${source_root}"
if command -v gh >/dev/null && gh auth status >/dev/null 2>&1; then
  gh repo clone sfabbro/torchsky "${source_root}" -- --filter=blob:none
else
  git clone --filter=blob:none https://github.com/sfabbro/torchsky.git "${source_root}"
fi
git -C "${source_root}" checkout --detach "${commit}"
test "$(git -C "${source_root}" rev-parse HEAD)" = "${commit}"
git -C "${source_root}" status --short

export PIXI_HOME=/scratch/pixi
export PIXI_CACHE_DIR=/scratch/pixi-cache
export TORCHSKY_DATA="${data_root}"
export PATH="${PIXI_HOME}/bin:${PATH}"
if ! command -v pixi >/dev/null 2>&1; then
  curl -fsSL https://pixi.sh/install.sh | bash
fi
pixi --version

case "${mode}" in
  foam)
    bands=${FOAM_BANDS:-F090W,F150W}
    crop_size=${FOAM_CROP_SIZE:-32}
    steps=${FOAM_STEPS:-2}
    grid_side=${FOAM_GRID_SIDE:-4}
    basis_count=${FOAM_BASIS_COUNT:-3}
    seed=${FOAM_SEED:-101}
    pixi run --manifest-path "${source_root}/pixi.toml" -e smacs-cuda -- \
      python "${source_root}/scripts/run_local_sphere_bench.py" \
      "${source_root}/benchmarks/bench_map_smacs0723_field_fit_joint.py" \
      --bands "${bands}" \
      --exposure-dir "${data_root}/jwst_smacs0723/aligned" \
      --input-product aligned \
      --crop-size "${crop_size}" \
      --crop-strategy best_overlap \
      --steps "${steps}" \
      --seed "${seed}" \
      --foam-grid-side "${grid_side}" \
      --foam-basis-count "${basis_count}" \
      --models spectral_power_foam \
      --json-out "${artifact_root}/shared-foam.json" \
      --markdown-out "${artifact_root}/shared-foam.md" \
      --export-dir "${artifact_root}/products"
    ;;
  native-band)
    band=${SMACS_BAND:?set SMACS_BAND}
    methods=torchsky_drizzle_square_pf0p800,torchsky_drizzle_square_pf1p000
    validation_root=/arc/projects/ots/torchsky-runs/20260729-smacs-validation-32e0940
    for batch in blank validation-000 validation-001 validation-002; do
      batch_root=${artifact_root}/${batch}/${band}
      mkdir -p "${batch_root}/products"
      injection_args=()
      if [[ "${batch}" != blank ]]; then
        injection_args=(
          --injection-manifest "${validation_root}/${batch}/manifest.json"
        )
      fi
      pixi run --manifest-path "${source_root}/pixi.toml" -e map-compat -- \
        python "${source_root}/scripts/run_local_sphere_bench.py" \
        "${source_root}/benchmarks/bench_map_smacs0723_stackers.py" \
        --band "${band}" \
        --module b \
        --exposure-dir "${data_root}/jwst_smacs0723/aligned" \
        --input-product aligned \
        --patt-nums 3,8 \
        --crop-size 512 \
        --crop-mode center \
        --crop-x0 112 \
        --crop-y0 0 \
        --validation-classical \
        --drizzle-pixfracs 0.8,1.0 \
        --methods "${methods}" \
        "${injection_args[@]}" \
        --product-out-dir "${batch_root}/products" \
        --json-out "${batch_root}/stackers.json" \
        --markdown-out "${batch_root}/stackers.md"
    done
    ;;
  native-score)
    pixi run --manifest-path "${source_root}/pixi.toml" -e map-compat -- \
      python "${source_root}/scripts/run_local_sphere_bench.py" \
      "${source_root}/benchmarks/score_smacs0723_native_classical.py" \
      --input-root-pattern \
        "/arc/projects/ots/torchsky-runs/20260729-smacs-native-validation-d8fece0-{band}" \
      --validation-root \
        /arc/projects/ots/torchsky-runs/20260729-smacs-validation-32e0940 \
      --effective-psf-root \
        /arc/projects/ots/torchsky-runs/20260729-smacs-torchsky-drizzle-aa0aa21/effective_psfs \
      --out-dir "${artifact_root}/score"
    ;;
  *)
    echo "unknown mode: ${mode}" >&2
    exit 2
    ;;
esac

echo "completed_utc=$(date -u +%FT%TZ)"
