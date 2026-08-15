#!/usr/bin/env bash
set -euo pipefail

mode=${1:?usage: run_canfar_wcs_sphere.sh MODE RUN_ID}
run_id=${2:?usage: run_canfar_wcs_sphere.sh MODE RUN_ID}
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

case "${mode}" in
  cpu | cuda) ;;
  *)
    echo "unknown mode: ${mode} (expected cpu or cuda)" >&2
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
export PATH="${PIXI_HOME}/bin:${PATH}"
export PYTHONUNBUFFERED=1
if ! command -v pixi >/dev/null 2>&1; then
  curl -fsSL https://pixi.sh/install.sh | bash
fi
pixi --version

cd "${source_root}"
echo "git_head=$(git rev-parse HEAD)"
pixi info || true

install_env() {
  local env_name=$1
  pixi run --manifest-path "${source_root}/pixi.toml" -e "${env_name}" -- \
    python -m pip install -e "${source_root}" --no-build-isolation
}

record_versions() {
  local env_name=$1
  pixi run --manifest-path "${source_root}/pixi.toml" -e "${env_name}" -- \
    python "${source_root}/scripts/print_benchmark_comparator_versions.py" \
    --json-out "${artifact_root}/VERSIONS_${env_name}.json"
}

case "${mode}" in
  cpu)
    install_env bench-all
    record_versions bench-all
    pixi run --manifest-path "${source_root}/pixi.toml" wcs-sphere-acceptance-full
    cp -a "${source_root}/benchmarks_results/." "${artifact_root}/" || true
    ;;
  cuda)
    install_env cuda-bench
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      pytest \
      tests/test_healpix_primitives.py \
      tests/test_healpix_parity.py \
      tests/test_wcs_core.py \
      tests/test_wcs_sip.py \
      tests/test_wcs_parity.py \
      -q --tb=line -k cuda
    install_env bench-all
    record_versions bench-all
    pixi run --manifest-path "${source_root}/pixi.toml" -e bench-all -- \
      python "${source_root}/benchmarks/bench_all.py" \
      --profile user --sphere-only \
      --run-id acceptance_sphere_cuda \
      --output-dir "${artifact_root}"
    ;;
esac

echo "completed_utc=$(date -u +%FT%TZ)"
echo "artifacts=${artifact_root}"
