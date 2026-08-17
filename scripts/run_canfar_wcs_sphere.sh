#!/usr/bin/env bash
# In-container CANFAR runner for torchsky WCS + sphere tests, healpy parity, and benches.
# Writes /arc/projects/ots/torchsky-runs/<run_id>/ and copies the same tree to
# vos:sfabbro/torchsky-runs/<run_id>/.
set -euo pipefail

mode=${1:?usage: run_canfar_wcs_sphere.sh MODE RUN_ID}
run_id=${2:?usage: run_canfar_wcs_sphere.sh MODE RUN_ID}
commit=${TORCHSKY_COMMIT:?set TORCHSKY_COMMIT to an exact pushed SHA}
container_image=${CONTAINER_IMAGE:-images.canfar.net/astroai/notebook:latest}
artifact_root=/arc/projects/ots/torchsky-runs/${run_id}
source_root=${TORCHSKY_SOURCE_ROOT:-/scratch/${run_id}-src}
vos_dest=${TORCHSKY_VOS_DEST:-vos:sfabbro/torchsky-runs/${run_id}}

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
echo "started_utc=$(date -u +%FT%TZ)" > "${artifact_root}/HEARTBEAT"
exec > >(tee "${artifact_root}/session.log") 2>&1

echo "run_id=${run_id}"
echo "mode=${mode}"
echo "requested_commit=${commit}"
echo "image=${container_image}"
echo "vos_dest=${vos_dest}"
echo "started_utc=$(date -u +%FT%TZ)"
nvidia-smi || true

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PIP_CACHE_DIR=/scratch/pip-cache
export TMPDIR=/scratch/tmp
mkdir -p "${PIP_CACHE_DIR}" "${TMPDIR}"

# vos/CADC CLIs ship with AstroAI images but are often off PATH.
for d in /opt/astroai/venv/cadc/bin /opt/astroai/bin /opt/conda/bin /usr/local/bin; do
  if [[ -x "${d}/vcp" ]]; then
    export PATH="${d}:${PATH}"
    break
  fi
done

vos_upload() {
  local label=$1
  echo "vos_upload label=${label} utc=$(date -u +%FT%TZ)"
  if ! command -v vcp >/dev/null; then
    echo "vcp not on PATH; cannot copy ${artifact_root} to ${vos_dest}" >&2
    return 1
  fi
  if command -v vmkdir >/dev/null; then
    vmkdir -p "${vos_dest}"
  fi
  vcp "${artifact_root}/." "${vos_dest}/"
  echo "vos_dest=${vos_dest}"
  tar -C "$(dirname "${artifact_root}")" -czf "/scratch/${run_id}.tar.gz" "$(basename "${artifact_root}")"
  vcp "/scratch/${run_id}.tar.gz" "vos:sfabbro/torchsky-runs/${run_id}.tar.gz"
  echo "vos_tarball=vos:sfabbro/torchsky-runs/${run_id}.tar.gz"
}

on_exit() {
  local rc=$?
  set +e
  echo "exit_utc=$(date -u +%FT%TZ) rc=${rc}" | tee -a "${artifact_root}/HEARTBEAT"
  vos_upload "exit_rc${rc}" || echo "VOS upload failed rc=${rc}" >&2
  exit "${rc}"
}
trap on_exit EXIT

vos_upload heartbeat || echo "heartbeat VOS upload failed; continuing job" >&2

clone_source() {
  if [[ -d "${source_root}/.git" ]] \
    && [[ "$(git -C "${source_root}" rev-parse HEAD 2>/dev/null || true)" == "${commit}" ]]; then
    echo "reusing TORCHSKY_SOURCE_ROOT=${source_root}"
    return
  fi
  rm -rf "${source_root}"
  local clone_url="https://github.com/sfabbro/torchsky.git"
  if [[ -n "${GH_TOKEN:-}" ]]; then
    clone_url="https://x-access-token:${GH_TOKEN}@github.com/sfabbro/torchsky.git"
  fi
  if command -v gh >/dev/null && gh auth status >/dev/null 2>&1; then
    gh repo clone sfabbro/torchsky "${source_root}" -- --filter=blob:none
  else
    git clone --filter=blob:none "${clone_url}" "${source_root}"
  fi
  git -C "${source_root}" checkout --detach "${commit}"
}

clone_source
test "$(git -C "${source_root}" rev-parse HEAD)" = "${commit}"
git -C "${source_root}" status --short

export PIXI_HOME=/scratch/pixi
export PIXI_CACHE_DIR=/scratch/pixi-cache
export PATH="${PIXI_HOME}/bin:${PATH}"
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

copy_tree() {
  local src=$1
  if [[ -d "${src}" ]]; then
    mkdir -p "${artifact_root}"
    cp -a "${src}/." "${artifact_root}/"
  fi
}

case "${mode}" in
  cpu)
    install_env bench-all
    record_versions bench-all
    pixi run --manifest-path "${source_root}/pixi.toml" wcs-sphere-acceptance-full
    copy_tree "${source_root}/benchmarks_results"
    copy_tree "${source_root}/benchmark_results"
    pixi run --manifest-path "${source_root}/pixi.toml" -e sphere-bench -- \
      python "${source_root}/benchmarks/bench_sphere_spectral.py" \
      --nside 32 --lmax 32 --runs 7 \
      --json-out "${artifact_root}/sphere_spectral.json"
    pixi run --manifest-path "${source_root}/pixi.toml" -e sphere-bench -- \
      python "${source_root}/benchmarks/bench_sphere_spectral.py" \
      --nside 64 --lmax 63 --runs 5 \
      --json-out "${artifact_root}/sphere_spectral_nside64.json"
    ;;
  cuda)
    install_env cuda-bench
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python -m pip install astropy
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python -m pip install healpy || true
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      pytest \
      tests/test_healpix_primitives.py \
      tests/test_healpix_parity.py \
      tests/test_healpix_image_plane.py \
      tests/test_wcs_core.py \
      tests/test_wcs_sip.py \
      tests/test_wcs_parity.py \
      -q --tb=line -k cuda \
      --junitxml="${artifact_root}/cuda_pytest.xml"
    record_versions cuda-bench
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python "${source_root}/benchmarks/bench_all.py" \
      --profile user --wcs-only --wcs-device cuda \
      --run-id acceptance_wcs_cuda \
      --output-dir "${artifact_root}"
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python "${source_root}/benchmarks/bench_all.py" \
      --profile user --sphere-only \
      --run-id acceptance_sphere_cuda \
      --output-dir "${artifact_root}"
    ;;
esac

echo "completed_utc=$(date -u +%FT%TZ)"
echo "artifacts=${artifact_root}"
