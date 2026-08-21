#!/usr/bin/env bash
set -euo pipefail

dest=${TORCHSKY_HEALPIX_DATAPATH:-./datapath}
base_url=${HEALPIX_FULL_WEIGHTS_URL:-https://raw.githubusercontent.com/healpy/healpy-data/master/full_weights}
nsides=("$@")
if [[ ${#nsides[@]} -eq 0 ]]; then
  nsides=(32 64 128 256 512 1024 2048 4096)
fi

mkdir -p "${dest}/full_weights"
for nside in "${nsides[@]}"; do
  printf -v filename 'healpix_full_weights_nside_%04d.fits' "${nside}"
  curl --fail --location --remote-name --output-dir "${dest}/full_weights" \
    "${base_url}/${filename}"
done

printf 'Set TORCHSKY_HEALPIX_DATAPATH=%s\n' "${dest}"
