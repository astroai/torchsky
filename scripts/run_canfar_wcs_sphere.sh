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
  cpu | cuda | spectral | sphere | sparse | healsparse | spin | geometry | cuhpx | \
  wcs-paper2 | wcs-tpv | wcs-spectral | wcs-tabular | sht-profile | moc | catalog | \
  observed | geom-cuda | world-cuda) ;;
  *)
    echo "unknown mode: ${mode} (expected cpu, cuda, spectral, sphere, sparse, healsparse, spin, geometry, cuhpx, wcs-paper2, wcs-tpv, wcs-spectral, wcs-tabular, sht-profile, moc, catalog, observed, geom-cuda, or world-cuda)" >&2
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
  if [[ -x "${d}/vcp" || -x "${d}/gh" ]]; then
    export PATH="${d}:${PATH}"
  fi
done

gist_upload() {
  local label=$1
  if ! command -v gh >/dev/null || ! gh auth status >/dev/null 2>&1; then
    echo "gh not authenticated; skip gist fallback" >&2
    return 1
  fi
  local files=()
  local f
  for f in "${artifact_root}/HEARTBEAT" "${artifact_root}/session.log" "${artifact_root}"/*.json; do
    if [[ -f "${f}" && -s "${f}" ]]; then
      files+=("${f}")
    fi
  done
  if [[ "${#files[@]}" -eq 0 ]]; then
    echo "gist_upload: no non-empty artifact files" >&2
    return 1
  fi
  local url
  url="$(gh gist create --desc "torchsky ${run_id} ${label}" "${files[@]}")"
  echo "${url}" | tee "${artifact_root}/GIST_URL"
  echo "gist_url=${url}"
}

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
  local sample="${artifact_root}/session.log"
  if [[ -s "${sample}" ]]; then
    local remote_bytes
    remote_bytes="$(vls -l "${vos_dest}/session.log" 2>/dev/null | awk '{print $5}' | tail -n 1 || true)"
    if [[ "${remote_bytes}" == "0" ]]; then
      echo "VOS stored 0-byte session.log; artifacts remain at ${artifact_root} and GIST_URL" >&2
      return 1
    fi
  fi
}

on_exit() {
  local rc=$?
  set +e
  echo "exit_utc=$(date -u +%FT%TZ) rc=${rc}" | tee -a "${artifact_root}/HEARTBEAT"
  vos_upload "exit_rc${rc}" || echo "VOS upload failed rc=${rc}" >&2
  gist_upload "exit_rc${rc}" || echo "gist fallback failed rc=${rc}" >&2
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

case "${mode}" in
  cpu)
    # ponytail: do not call wcs-sphere-acceptance-full here. That task's WCS
    # path auto-spawns pixi -e bench-legacy (pyast/kapteyn). The last CPU job
    # died there after finishing native WCS, and copy_tree never ran.
    install_env bench-all
    install_env sphere-bench
    record_versions bench-all
    cpu_rc=0
    pixi run --manifest-path "${source_root}/pixi.toml" -e bench-all -- \
      python "${source_root}/benchmarks/bench_all.py" \
      --profile user --wcs-only --no-gpu --no-legacy-wcs-bridge \
      --run-id acceptance_wcs_full \
      --output-dir "${artifact_root}" || cpu_rc=$?
    vos_upload after_wcs || echo "after_wcs VOS upload failed; continuing job" >&2
    pixi run --manifest-path "${source_root}/pixi.toml" -e bench-all -- \
      python "${source_root}/benchmarks/bench_all.py" \
      --profile user --sphere-only --no-gpu \
      --run-id acceptance_sphere_full \
      --output-dir "${artifact_root}" || cpu_rc=$?
    vos_upload after_sphere || echo "after_sphere VOS upload failed; continuing job" >&2
    pixi run --manifest-path "${source_root}/pixi.toml" -e bench-all -- \
      pytest \
      tests/test_wcs_*.py \
      tests/test_sphere_*.py \
      tests/test_healsparse_upstream.py \
      tests/test_healpix_*.py \
      tests/test_spectral.py \
      -q --tb=line \
      --junitxml="${artifact_root}/cpu_pytest.xml" || cpu_rc=$?
    pixi run --manifest-path "${source_root}/pixi.toml" -e sphere-bench -- \
      python "${source_root}/benchmarks/bench_spin_matrix.py" \
      --case-set default --runs 2 --no-throughput-gate \
      --json-out "${artifact_root}/acceptance_spin_matrix.json" || cpu_rc=$?
    pixi run --manifest-path "${source_root}/pixi.toml" -e sphere-bench -- \
      python "${source_root}/benchmarks/bench_sphere_spectral.py" \
      --nside 32 --lmax 32 --runs 7 \
      --json-out "${artifact_root}/sphere_spectral.json" || cpu_rc=$?
    pixi run --manifest-path "${source_root}/pixi.toml" -e sphere-bench -- \
      python "${source_root}/benchmarks/bench_sphere_spectral.py" \
      --nside 64 --lmax 63 --runs 5 \
      --json-out "${artifact_root}/sphere_spectral_nside64.json" || cpu_rc=$?
    if [[ "${cpu_rc}" -ne 0 ]]; then
      echo "cpu job failures rc=${cpu_rc}"
      exit "${cpu_rc}"
    fi
    ;;
  cuda)
    install_env cuda-bench
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python -m pip install astropy
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python -m pip install healpy || true
    # ponytail: pytest -k cuda is a smoke check, not the Linux evidence.
    # Keep benches running after a cuda-named test fails so VOS still gets
    # WCS/sphere JSON. Fail the job after those benches using cuda_rc.
    cuda_rc=0
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      pytest \
      tests/test_healpix_primitives.py \
      tests/test_healpix_parity.py \
      tests/test_healpix_image_plane.py \
      tests/test_wcs_core.py \
      tests/test_wcs_sip.py \
      tests/test_wcs_parity.py \
      tests/test_wcs_extended_projections.py \
      tests/test_wcs_observed_frames.py \
      tests/test_wcs_tabular.py \
      tests/test_wcs_spectral.py \
      tests/test_wcs_temporal.py \
      tests/test_catalogs.py \
      tests/test_sphere_polygons.py \
      tests/test_sphere_sparse.py \
      tests/test_wcs_gwcs_sampling.py \
      -q --tb=line -k cuda \
      --junitxml="${artifact_root}/cuda_pytest.xml" || cuda_rc=$?
    record_versions cuda-bench || cuda_rc=$?
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python "${source_root}/benchmarks/bench_all.py" \
      --profile user --wcs-only --wcs-device cuda --no-legacy-wcs-bridge \
      --run-id acceptance_wcs_cuda \
      --output-dir "${artifact_root}" || cuda_rc=$?
    vos_upload after_wcs || echo "after_wcs VOS upload failed; continuing job" >&2
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python "${source_root}/benchmarks/bench_all.py" \
      --profile user --sphere-only \
      --run-id acceptance_sphere_cuda \
      --output-dir "${artifact_root}" || cuda_rc=$?
    if [[ "${cuda_rc}" -ne 0 ]]; then
      echo "cuda job failures rc=${cuda_rc}"
      exit "${cuda_rc}"
    fi
    ;;
  spectral)
    install_env sphere-bench
    record_versions sphere-bench
    job_rc=0
    pixi run --manifest-path "${source_root}/pixi.toml" -e sphere-bench -- \
      python "${source_root}/benchmarks/bench_sphere_spectral.py" \
      --nside 32 --lmax 32 --runs 7 \
      --json-out "${artifact_root}/sphere_spectral_nside32.json" || job_rc=$?
    pixi run --manifest-path "${source_root}/pixi.toml" -e sphere-bench -- \
      python "${source_root}/benchmarks/bench_sphere_spectral.py" \
      --nside 64 --lmax 63 --runs 5 \
      --json-out "${artifact_root}/sphere_spectral_nside64.json" || job_rc=$?
    pixi run --manifest-path "${source_root}/pixi.toml" -e sphere-bench -- \
      python "${source_root}/benchmarks/bench_sphere_spectral.py" \
      --nside 128 --lmax 128 --runs 3 \
      --json-out "${artifact_root}/sphere_spectral_nside128.json" || job_rc=$?
    if [[ "${job_rc}" -ne 0 ]]; then
      echo "spectral job failures rc=${job_rc}"
      exit "${job_rc}"
    fi
    ;;
  sphere)
    install_env bench-all
    install_env sphere-bench
    record_versions sphere-bench
    pixi run --manifest-path "${source_root}/pixi.toml" -e bench-all -- \
      python "${source_root}/benchmarks/bench_all.py" \
      --profile user --sphere-only --no-gpu \
      --run-id acceptance_sphere_full \
      --output-dir "${artifact_root}"
    ;;
  sparse | healsparse)
    install_env sphere-bench
    record_versions sphere-bench
    job_rc=0
    pixi run --manifest-path "${source_root}/pixi.toml" -e sphere-bench -- \
      python "${source_root}/benchmarks/bench_sphere_sparse.py" \
      --nside 256 --coverage-frac 0.1 --n-queries 200000 --runs 9 \
      --json-out "${artifact_root}/sphere_sparse_nside256.json" || job_rc=$?
    pixi run --manifest-path "${source_root}/pixi.toml" -e sphere-bench -- \
      python "${source_root}/benchmarks/bench_sphere_sparse.py" \
      --nside 512 --coverage-frac 0.1 --n-queries 500000 --runs 7 \
      --json-out "${artifact_root}/sphere_sparse_nside512.json" || job_rc=$?
    if [[ "${job_rc}" -ne 0 ]]; then
      echo "sparse job failures rc=${job_rc}"
      exit "${job_rc}"
    fi
    ;;
  spin)
    install_env sphere-bench
    record_versions sphere-bench
    job_rc=0
    pixi run --manifest-path "${source_root}/pixi.toml" -e sphere-bench -- \
      python "${source_root}/benchmarks/bench_spin_matrix.py" \
      --case-set default --runs 5 --no-throughput-gate \
      --json-out "${artifact_root}/acceptance_spin_matrix.json" || job_rc=$?
    pixi run --manifest-path "${source_root}/pixi.toml" -e sphere-bench -- \
      python "${source_root}/benchmarks/bench_spin_matrix.py" \
      --case-set extended --runs 3 --no-throughput-gate \
      --json-out "${artifact_root}/acceptance_spin_matrix_extended.json" || job_rc=$?
    if [[ "${job_rc}" -ne 0 ]]; then
      echo "spin job failures rc=${job_rc}"
      exit "${job_rc}"
    fi
    ;;
  geometry)
    install_env sphere-bench
    record_versions sphere-bench
    job_rc=0
    pixi run --manifest-path "${source_root}/pixi.toml" -e sphere-bench -- \
      python "${source_root}/benchmarks/bench_sphere_matrix.py" \
      --device cpu --nside 1024 --n-points 200000 --runs 5 --strict-missing \
      --no-throughput-gate \
      --output-dir "${artifact_root}/sphere_matrix" || job_rc=$?
    pixi run --manifest-path "${source_root}/pixi.toml" -e sphere-bench -- \
      python "${source_root}/benchmarks/bench_healpix_advanced.py" \
      --device cpu --nside 1024 --n-points 200000 --runs 5 \
      --json-out "${artifact_root}/healpix_advanced.json" || job_rc=$?
    if [[ "${job_rc}" -ne 0 ]]; then
      echo "geometry job failures rc=${job_rc}"
      exit "${job_rc}"
    fi
    ;;
  cuhpx)
    install_env cuda-bench
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python -m pip install "git+https://github.com/NVlabs/cuHPX.git" || true
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python "${source_root}/benchmarks/bench_cuhpx_sht_spike.py" \
      --nside 32 --runs 3 \
      --json-out "${artifact_root}/cuhpx_sht_spike.json"
    record_versions cuda-bench || true
    ;;
  wcs-paper2)
    install_env bench-all
    record_versions bench-all
    pixi run --manifest-path "${source_root}/pixi.toml" -e bench-all -- \
      python "${source_root}/benchmarks/bench_wcs.py" \
      --cases PCO,XPH,TSC,CSC,QSC,NCP \
      --n-points 100000 --runs 3 --device cpu \
      --json-out "${artifact_root}/wcs_paper2.json"
    TORCHSKY_WCS_FUSED=1 pixi run --manifest-path "${source_root}/pixi.toml" -e bench-all -- \
      python "${source_root}/benchmarks/bench_wcs.py" \
      --cases CSC --n-points 20000 --runs 1 --device cpu \
      --json-out "${artifact_root}/wcs_paper2_fused1.json"
    TORCHSKY_WCS_FUSED=0 pixi run --manifest-path "${source_root}/pixi.toml" -e bench-all -- \
      python "${source_root}/benchmarks/bench_wcs.py" \
      --cases CSC --n-points 20000 --runs 1 --device cpu \
      --json-out "${artifact_root}/wcs_paper2_fused0.json"
    pixi run --manifest-path "${source_root}/pixi.toml" -e bench-all -- \
      python -c "import torchsky._cpp as c; assert hasattr(c,'wcs_csc_x2s')"
    ;;
  wcs-tpv)
    install_env bench-all
    record_versions bench-all
    job_rc=0
    pixi run --manifest-path "${source_root}/pixi.toml" -e bench-all -- \
      python "${source_root}/benchmarks/bench_wcs.py" \
      --cases TPV --n-points 200000 --runs 5 --device cpu \
      --json-out "${artifact_root}/wcs_tpv.json" || job_rc=$?
    pixi run --manifest-path "${source_root}/pixi.toml" -e bench-all -- \
      python "${source_root}/benchmarks/bench_wcs_tpv_inverse.py" \
      --json-out "${artifact_root}/wcs_tpv_inverse.json" || job_rc=$?
    pixi run --manifest-path "${source_root}/pixi.toml" -e bench-all -- \
      python "${source_root}/benchmarks/bench_wcs_small_n_paths.py" \
      --cases TAN,SIN,ARC,TPV --n-points 256 --runs 11 \
      --json-out "${artifact_root}/wcs_small_n_paths.json" || job_rc=$?
    if [[ "${job_rc}" -ne 0 ]]; then
      echo "wcs-tpv job failures rc=${job_rc}"
      exit "${job_rc}"
    fi
    ;;
  wcs-spectral)
    install_env bench-all
    record_versions bench-all
    pixi run --manifest-path "${source_root}/pixi.toml" -e bench-all -- \
      python "${source_root}/benchmarks/bench_wcs_spectral_temporal.py" \
      --n-points 100000 --runs 3 \
      --json-out "${artifact_root}/wcs_spectral_temporal.json"
    ;;
  wcs-tabular)
    install_env bench-all
    record_versions bench-all
    pixi run --manifest-path "${source_root}/pixi.toml" -e bench-all -- \
      python "${source_root}/benchmarks/bench_wcs_tabular.py" \
      --grid-size 8 --n-points 10000 --runs 5 \
      --json-out "${artifact_root}/wcs_tabular.json"
    ;;
  sht-profile)
    install_env sphere-bench
    record_versions sphere-bench
    export TORCHSKY_SHT_PROFILE=1
    pixi run --manifest-path "${source_root}/pixi.toml" -e sphere-bench -- \
      python "${source_root}/benchmarks/bench_spin_matrix.py" \
      --case-set default --runs 5 --no-throughput-gate \
      --json-out "${artifact_root}/sht_profile_spin.json"
    pixi run --manifest-path "${source_root}/pixi.toml" -e sphere-bench -- \
      python "${source_root}/benchmarks/bench_sphere_spectral.py" \
      --nside 128 --lmax 128 --runs 5 \
      --json-out "${artifact_root}/sht_profile_nside128.json"
    ;;
  moc)
    install_env sphere-bench
    record_versions sphere-bench
    pixi run --manifest-path "${source_root}/pixi.toml" -e sphere-bench -- \
      python "${source_root}/benchmarks/bench_sphere_core.py" \
      --moc-only --n-points 200000 --runs 5 --moc-max-order 8 \
      --json-out "${artifact_root}/moc_core.json"
    ;;
  catalog)
    install_env bench-all
    record_versions bench-all
    job_rc=0
    pixi run --manifest-path "${source_root}/pixi.toml" -e bench-all -- \
      python "${source_root}/benchmarks/bench_catalog_compat.py" \
      --n-left 10000 --n-right-noise 80000 --runs 3 \
      --json-out "${artifact_root}/catalog_compat.json" || job_rc=$?
    vos_upload after_catalog_compat || echo "after_catalog_compat VOS upload failed; continuing job" >&2
    pixi run --manifest-path "${source_root}/pixi.toml" -e bench-all -- \
      python "${source_root}/benchmarks/bench_catalog_crossmatch.py" \
      --n-left 1000000 --n-right-band 500000 --n-right-background 500000 --runs 3 \
      --skip-decl-pruned \
      --json-out "${artifact_root}/catalog_n1e6.json" || job_rc=$?
    vos_upload after_catalog_n1e6 || echo "after_catalog_n1e6 VOS upload failed; continuing job" >&2
    pixi run --manifest-path "${source_root}/pixi.toml" -e bench-all -- \
      python "${source_root}/benchmarks/bench_catalog_crossmatch.py" \
      --n-left 10000000 --n-right-band 500000 --n-right-background 500000 \
      --runs 1 --skip-decl-pruned \
      --json-out "${artifact_root}/catalog_n1e7.json" || job_rc=$?
    if [[ "${job_rc}" -ne 0 ]]; then
      echo "catalog job failures rc=${job_rc}"
      exit "${job_rc}"
    fi
    ;;
  observed)
    install_env test
    record_versions test
    job_rc=0
    pixi run --manifest-path "${source_root}/pixi.toml" -e test -- \
      pytest tests/test_wcs_catalog_state.py tests/test_wcs_frames.py \
      tests/test_wcs_observed_frames.py -q --tb=line \
      --junitxml="${artifact_root}/observed_pytest.xml" || job_rc=$?
    pixi run --manifest-path "${source_root}/pixi.toml" -e test -- \
      python "${source_root}/benchmarks/bench_observed_place.py" \
      --json-out "${artifact_root}/observed_place.json" || job_rc=$?
    if [[ "${job_rc}" -ne 0 ]]; then
      echo "observed job failures rc=${job_rc}"
      exit "${job_rc}"
    fi
    ;;
  geom-cuda)
    install_env cuda-bench
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python "${source_root}/benchmarks/bench_sphere_polygons.py" \
      --device cuda --n-points 200000 --runs 3 \
      --json-out "${artifact_root}/geom_cuda.json"
    record_versions cuda-bench || true
    ;;
  world-cuda)
    install_env cuda-bench
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python -m pip install astropy
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python -m pip install healpy || true
    job_rc=0
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      pytest \
      tests/test_wcs_core.py \
      tests/test_wcs_extended_projections.py \
      tests/test_wcs_observed_frames.py \
      tests/test_wcs_tabular.py \
      tests/test_wcs_spectral.py \
      tests/test_wcs_temporal.py \
      tests/test_catalogs.py \
      tests/test_sphere_polygons.py \
      tests/test_sphere_sparse.py \
      tests/test_wcs_gwcs_sampling.py \
      -q --tb=line -k cuda \
      --junitxml="${artifact_root}/world_cuda_pytest.xml" || job_rc=$?
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python "${source_root}/benchmarks/bench_wcs.py" \
      --cases PCO,XPH,TSC,CSC,QSC,NCP \
      --n-points 100000 --runs 3 --device cuda \
      --json-out "${artifact_root}/wcs_paper2_cuda.json" || job_rc=$?
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python "${source_root}/benchmarks/bench_wcs.py" \
      --cases TPV --n-points 200000 --runs 5 --device cuda \
      --json-out "${artifact_root}/wcs_tpv_cuda.json" || job_rc=$?
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python "${source_root}/benchmarks/bench_wcs_tpv_inverse.py" \
      --n-points 200000 --runs 3 --device cuda \
      --json-out "${artifact_root}/wcs_tpv_inverse_cuda.json" \
      --markdown-out "${artifact_root}/wcs_tpv_inverse_cuda.md" || job_rc=$?
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python "${source_root}/benchmarks/bench_wcs_small_n_paths.py" \
      --cases TAN,SIN,ARC,TPV --n-points 256 --runs 11 --device cuda \
      --json-out "${artifact_root}/wcs_small_n_paths_cuda.json" || job_rc=$?
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python "${source_root}/benchmarks/bench_wcs_spectral_temporal.py" \
      --n-points 100000 --runs 3 --device cuda \
      --json-out "${artifact_root}/wcs_spectral_temporal_cuda.json" || job_rc=$?
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python "${source_root}/benchmarks/bench_wcs_tabular.py" \
      --grid-size 8 --n-points 10000 --runs 5 --device cuda \
      --json-out "${artifact_root}/wcs_tabular_cuda.json" || job_rc=$?
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python "${source_root}/benchmarks/bench_catalog_crossmatch.py" \
      --n-left 20000 --n-right-band 50000 --n-right-background 10000 --runs 3 \
      --device cuda \
      --json-out "${artifact_root}/catalog_cuda.json" || job_rc=$?
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python "${source_root}/benchmarks/bench_observed_place.py" \
      --device cuda --n-points 4096 --runs 7 \
      --json-out "${artifact_root}/observed_place_cuda.json" || job_rc=$?
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python "${source_root}/benchmarks/bench_sphere_polygons.py" \
      --device cuda --n-points 200000 --runs 3 \
      --json-out "${artifact_root}/geom_cuda.json" || job_rc=$?
    pixi run --manifest-path "${source_root}/pixi.toml" -e cuda-bench -- \
      python "${source_root}/benchmarks/bench_sphere_sparse.py" \
      --device cuda --nside 256 --coverage-frac 0.1 --n-queries 200000 --runs 5 \
      --json-out "${artifact_root}/sphere_sparse_cuda.json" || job_rc=$?
    record_versions cuda-bench || job_rc=$?
    if [[ "${job_rc}" -ne 0 ]]; then
      echo "world-cuda job failures rc=${job_rc}"
      exit "${job_rc}"
    fi
    ;;
esac

echo "completed_utc=$(date -u +%FT%TZ)"
echo "artifacts=${artifact_root}"
