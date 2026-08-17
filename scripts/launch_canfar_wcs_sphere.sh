#!/usr/bin/env bash
# Launch CANFAR headless CPU and/or CUDA jobs for torchsky WCS + sphere.
# Usage: launch_canfar_wcs_sphere.sh [cpu|cuda|all]
#
# Jobs clone TORCHSKY_COMMIT (default: origin/main HEAD), run
# scripts/run_canfar_wcs_sphere.sh, and copy artifacts to
# vos:sfabbro/torchsky-runs/<run_id>/.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MODE="${1:-all}"
case "${MODE}" in
  cpu | cuda | all) ;;
  *)
    echo "usage: $0 [cpu|cuda|all]" >&2
    exit 2
    ;;
esac

ensure_vcp() {
  if command -v vcp >/dev/null; then
    return 0
  fi
  local d
  for d in \
    "${TORCHSKY_VOS_BIN:-}" \
    /opt/astroai/venv/cadc/bin \
    /opt/astroai/bin; do
    if [[ -n "${d}" && -x "${d}/vcp" ]]; then
      export PATH="${d}:${PATH}"
      return 0
    fi
  done
  return 1
}

COMMIT="${TORCHSKY_COMMIT:-$(git rev-parse HEAD)}"
if ! git cat-file -e "${COMMIT}^{commit}" 2>/dev/null; then
  echo "TORCHSKY_COMMIT=${COMMIT} is not a local commit" >&2
  exit 2
fi
COMMIT="$(git rev-parse "${COMMIT}")"
IMAGE="${CONTAINER_IMAGE:-astroai/notebook:latest}"
CPU="${TORCHSKY_CANFAR_CPU:-8}"
MEMORY="${TORCHSKY_CANFAR_MEMORY:-32}"
STAMP="$(date -u +%Y%m%d-%H%M%S)"
VOS_BIN_URI="vos:sfabbro/torchsky-runs/bin"
RUNNER_VOS_URI="${VOS_BIN_URI}/run_canfar_wcs_sphere.sh"

if ! command -v canfar >/dev/null; then
  echo "canfar CLI not found" >&2
  exit 1
fi

if ! ensure_vcp; then
  echo "vcp not found; install vos (pip install vos) or set TORCHSKY_VOS_BIN" >&2
  exit 1
fi

if command -v vmkdir >/dev/null; then
  vmkdir -p "${VOS_BIN_URI}"
fi
vcp "${ROOT_DIR}/scripts/run_canfar_wcs_sphere.sh" "${RUNNER_VOS_URI}"

launch_one() {
  local job_mode=$1
  local gpu_flag=$2
  local run_id="wcs-sphere-${job_mode}-${STAMP}"
  local vos_dest="vos:sfabbro/torchsky-runs/${run_id}"
  local name
  name="$(printf 'torchsky-wcs-%s-%s' "${job_mode}" "${STAMP}" | tr '_.' '--' | tr -cd '[:alnum:]-')"
  local local_out="${ROOT_DIR}/benchmarks_results/canfar_${run_id}"
  mkdir -p "${local_out}"

  # ponytail: Skaha splits argv on spaces; tabs keep bash -c as one script.
  # Do not put $ or & in the payload (CANFAR expands $name; & splits URLs).
  local remote_plain="PATH=/opt/astroai/venv/cadc/bin:/opt/astroai/bin:/usr/local/bin:/usr/bin:/bin; vcp ${RUNNER_VOS_URI} /scratch/run_canfar_wcs_sphere.sh; bash /scratch/run_canfar_wcs_sphere.sh ${job_mode} ${run_id}"
  local remote_cmd
  remote_cmd="$(printf '%s' "${remote_plain}" | tr ' ' '\t')"
  case "${remote_cmd}" in
    *'$'* | *'&'* | *' '*)
      echo "refusing canfar payload with \$, &, or spaces: ${remote_plain}" >&2
      exit 2
      ;;
  esac

  local create_args=(
    create headless "${IMAGE}"
    --name "${name}"
    --cpu "${CPU}"
    --memory "${MEMORY}"
    --env "TORCHSKY_COMMIT=${COMMIT}"
    --env "TORCHSKY_VOS_DEST=${vos_dest}"
    --env "CONTAINER_IMAGE=${IMAGE}"
    --env "PYTHONUNBUFFERED=1"
  )
  if [[ -n "${gpu_flag}" ]]; then
    create_args+=(--gpu "${gpu_flag}")
  fi
  # Only forward a GitHub token when explicitly requested. The AstroAI
  # notebook image already has session GitHub credentials; do not scrape
  # `gh auth token` into CANFAR argv or logs.
  if [[ -n "${TORCHSKY_GH_TOKEN:-}" ]]; then
    create_args+=(--env "GH_TOKEN=${TORCHSKY_GH_TOKEN}")
  fi

  echo "launching mode=${job_mode} name=${name} commit=${COMMIT} vos=${vos_dest}"
  local create_log="${local_out}/create.log"
  if [[ "${TORCHSKY_CANFAR_DRY_RUN:-0}" == "1" ]]; then
    echo "dry-run: canfar create headless ${IMAGE} --name ${name} --cpu ${CPU} --memory ${MEMORY} gpu=${gpu_flag:-omit}"
    echo "${run_id}" > "${local_out}/run_id.txt"
    echo "${vos_dest}" > "${local_out}/vos_dest.txt"
    echo "${COMMIT}" > "${local_out}/commit.txt"
    return 0
  fi
  set +o pipefail
  canfar "${create_args[@]}" -- bash -c "${remote_cmd}" 2>&1 | tee "${create_log}"
  local create_rc=${PIPESTATUS[0]}
  set -o pipefail
  if [[ "${create_rc}" -ne 0 ]]; then
    echo "canfar create failed (rc=${create_rc}); see ${create_log}" >&2
    exit 1
  fi

  local session_id
  session_id="$(
    python3 - "${create_log}" <<'PY'
import re, sys
from pathlib import Path
text = Path(sys.argv[1]).read_text(encoding="utf-8")
m = re.search(r"ID:\s*([^)\s]+)", text)
print(m.group(1).strip() if m else "")
PY
  )"
  if [[ -z "${session_id}" ]]; then
    echo "could not parse session ID from ${create_log}" >&2
    exit 1
  fi
  echo "${session_id}" > "${local_out}/session_id.txt"
  echo "${run_id}" > "${local_out}/run_id.txt"
  echo "${vos_dest}" > "${local_out}/vos_dest.txt"
  echo "${COMMIT}" > "${local_out}/commit.txt"
  echo "session_id=${session_id} run_id=${run_id} vos=${vos_dest}"
}

if [[ "${MODE}" == "cpu" || "${MODE}" == "all" ]]; then
  launch_one cpu ""
fi
if [[ "${MODE}" == "cuda" || "${MODE}" == "all" ]]; then
  launch_one cuda 1
fi
