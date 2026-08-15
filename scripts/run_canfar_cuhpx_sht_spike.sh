#!/usr/bin/env bash
set -euo pipefail

# CANFAR CUDA-only cuHPX scalar SHT spike. Not a core dependency.
# Usage: TORCHSKY_COMMIT=<pushed SHA> run_canfar_cuhpx_sht_spike.sh RUN_ID

run_id=${1:?usage: run_canfar_cuhpx_sht_spike.sh RUN_ID}
commit=${TORCHSKY_COMMIT:?set TORCHSKY_COMMIT to an exact pushed SHA}
container_image=${CONTAINER_IMAGE:-images.canfar.net/astroai/notebook:latest}
artifact_root=/arc/projects/ots/torchsky-runs/${run_id}
source_root=/scratch/${run_id}-src

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

export PIXI_HOME=/scratch/pixi
export PIXI_CACHE_DIR=/scratch/pixi-cache
export PATH="${PIXI_HOME}/bin:${PATH}"
export PYTHONUNBUFFERED=1
if ! command -v pixi >/dev/null 2>&1; then
  curl -fsSL https://pixi.sh/install.sh | bash
fi

cd "${source_root}"
pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
  python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"

pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
  python -m pip install -e "${source_root}" --no-build-isolation

# Source build; leave the env if cuHPX fails so the spike still writes no-go JSON.
pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
  python -m pip install "git+https://github.com/NVlabs/cuHPX.git" || true

pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
  python "${source_root}/benchmarks/bench_cuhpx_sht_spike.py" \
  --nside 32 --runs 3 \
  --json-out "${artifact_root}/cuhpx_sht_spike.json"

pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
  python "${source_root}/scripts/print_benchmark_comparator_versions.py" \
  --json-out "${artifact_root}/VERSIONS_cuhpx.json" || true

echo "completed_utc=$(date -u +%FT%TZ)"
echo "artifacts=${artifact_root}"
