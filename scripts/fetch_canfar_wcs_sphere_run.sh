#!/usr/bin/env bash
# Download one CANFAR WCS/sphere run from vos:sfabbro/torchsky-runs/<run_id>/.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

RUN_ID="${1:?usage: $0 <run-id>}"
VOS_BASE="${TORCHSKY_VOS_BASE:-vos:sfabbro/torchsky-runs}"
VOS_URI="${VOS_BASE}/${RUN_ID}"
LOCAL_DIR="${ROOT_DIR}/benchmarks_results/${RUN_ID}"

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

if ! ensure_vcp; then
  cat >&2 <<EOF
vcp not found. Install VOS tools locally:

  pip install vos
  cadc-get-cert -u <user>

Then re-run:
  bash scripts/fetch_canfar_wcs_sphere_run.sh ${RUN_ID}
EOF
  exit 1
fi

mkdir -p "${LOCAL_DIR}"
if vcp "${VOS_URI}/" "${LOCAL_DIR}/" 2>/dev/null; then
  :
elif vcp "${VOS_URI}.tar.gz" "${LOCAL_DIR}/${RUN_ID}.tar.gz"; then
  tar -xzf "${LOCAL_DIR}/${RUN_ID}.tar.gz" -C "${LOCAL_DIR}"
else
  echo "failed to fetch ${VOS_URI} or ${VOS_URI}.tar.gz" >&2
  exit 1
fi
nested="${LOCAL_DIR}/${RUN_ID}"
if [[ -d "${nested}" ]]; then
  shopt -s nullglob dotglob
  mv "${nested}"/* "${LOCAL_DIR}/" 2>/dev/null || true
  rmdir "${nested}" 2>/dev/null || true
  shopt -u nullglob dotglob
fi
echo "fetched ${VOS_URI} -> ${LOCAL_DIR}"
