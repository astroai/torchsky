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
export PYTHONPATH="${source_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
if ! command -v pixi >/dev/null 2>&1; then
  curl -fsSL https://pixi.sh/install.sh | bash
fi
pixi --version

foam_geometry_args=()
case "${FOAM_TRAIN_GEOMETRY:-0}" in
  0) ;;
  1) foam_geometry_args=(--foam-train-geometry) ;;
  *) echo "FOAM_TRAIN_GEOMETRY must be 0 or 1" >&2; exit 2 ;;
esac

foam_initialization_args=()
case "${FOAM_INITIALIZE_FROM_COADD:-0}" in
  0) ;;
  1) foam_initialization_args=(--foam-initialize-from-coadd) ;;
  *) echo "FOAM_INITIALIZE_FROM_COADD must be 0 or 1" >&2; exit 2 ;;
esac

foam_footprint_args=()
case "${FOAM_FIT_TARGET_FOOTPRINT:-0}" in
  0) ;;
  1) foam_footprint_args=(--foam-fit-target-footprint) ;;
  *) echo "FOAM_FIT_TARGET_FOOTPRINT must be 0 or 1" >&2; exit 2 ;;
esac

foam_spatial_args=()
case "${FOAM_AFFINE_SPATIAL_BASIS:-0}" in
  0) ;;
  1) foam_spatial_args=(--foam-affine-spatial-basis) ;;
  *) echo "FOAM_AFFINE_SPATIAL_BASIS must be 0 or 1" >&2; exit 2 ;;
esac

foam_spectral_args=()
case "${FOAM_WHITEN_SPECTRAL_BASIS:-0}" in
  0) ;;
  1) foam_spectral_args=(--foam-whiten-spectral-basis) ;;
  *) echo "FOAM_WHITEN_SPECTRAL_BASIS must be 0 or 1" >&2; exit 2 ;;
esac

foam_exact_refinement_args=()
case "${FOAM_EXACT_OBSERVATION_REFINE:-0}" in
  0) ;;
  1) foam_exact_refinement_args=(--foam-exact-observation-refine) ;;
  *) echo "FOAM_EXACT_OBSERVATION_REFINE must be 0 or 1" >&2; exit 2 ;;
esac

operator_calibration_args=()
if [[ -n "${CALIBRATION_EVIDENCE_MANIFEST:-}" || -n "${CALIBRATION_FIT_REPORT:-}" ]]; then
  test -n "${CALIBRATION_EVIDENCE_MANIFEST:-}"
  test -n "${CALIBRATION_FIT_REPORT:-}"
  operator_calibration_args=(
    --operator-calibration-manifest "${CALIBRATION_EVIDENCE_MANIFEST}"
    --operator-calibration-fit-report "${CALIBRATION_FIT_REPORT}"
  )
fi

case "${mode}" in
  calibration-extract)
    calibration_input_root=${CALIBRATION_INPUT_ROOT:-${data_root}/jwst_smacs0723/aligned}
    calibration_pattern=${CALIBRATION_PATTERN:-*_crf_rshift_tweakreg.fits}
    expected_exposure_count=${CALIBRATION_EXPECTED_EXPOSURES:-135}
    exposure_count=$(find -L "${calibration_input_root}" -maxdepth 1 -type f \
      -name "${calibration_pattern}" | wc -l | tr -d ' ')
    echo "calibration_input_root=${calibration_input_root}"
    echo "calibration_pattern=${calibration_pattern}"
    echo "calibration_exposure_count=${exposure_count}"
    test "${exposure_count}" = "${expected_exposure_count}"
    pixi run --manifest-path "${source_root}/pixi.toml" -e default -- \
      python -m pip install -e "${source_root}" --no-build-isolation
    pixi run --manifest-path "${source_root}/pixi.toml" -e default -- \
      python -c 'from torchsky.cli import main; raise SystemExit(main())' \
      calibration-extract \
      "${calibration_input_root}/${calibration_pattern}" \
      --output-dir "${artifact_root}/evidence"
    test -s "${artifact_root}/evidence/calibration_evidence_manifest.json"
    ;;
  calibration-fit)
    calibration_manifest=${CALIBRATION_EVIDENCE_MANIFEST:?set CALIBRATION_EVIDENCE_MANIFEST}
    calibration_steps=${CALIBRATION_FIT_STEPS:-300}
    calibration_lr=${CALIBRATION_FIT_LR:-0.001}
    calibration_device=${CALIBRATION_FIT_DEVICE:-cpu}
    test -s "${calibration_manifest}"
    pixi run --manifest-path "${source_root}/pixi.toml" -e default -- \
      python -m pip install -e "${source_root}" --no-build-isolation
    pixi run --manifest-path "${source_root}/pixi.toml" -e default -- \
      python "${source_root}/benchmarks/fit_smacs0723_operator_calibration.py" \
      --manifest "${calibration_manifest}" \
      --output-dir "${artifact_root}/fit" \
      --steps "${calibration_steps}" \
      --lr "${calibration_lr}" \
      --device "${calibration_device}"
    test -s "${artifact_root}/fit/operator_calibration_fit.json"
    ;;
  calibration-inspect)
    calibration_manifest=${CALIBRATION_EVIDENCE_MANIFEST:?set CALIBRATION_EVIDENCE_MANIFEST}
    calibration_fit_report=${CALIBRATION_FIT_REPORT:?set CALIBRATION_FIT_REPORT}
    calibration_inspection_detector=${CALIBRATION_INSPECTION_DETECTOR:-NRCB4}
    calibration_inspection_per_group=${CALIBRATION_INSPECTION_PER_GROUP:-12}
    test -s "${calibration_manifest}"
    test -s "${calibration_fit_report}"
    pixi run --manifest-path "${source_root}/pixi.toml" -e bench -- \
      python -m pip install -e "${source_root}" --no-build-isolation
    pixi run --manifest-path "${source_root}/pixi.toml" -e bench -- \
      python "${source_root}/benchmarks/export_smacs0723_calibration_inspection.py" \
      --manifest "${calibration_manifest}" \
      --fit-report "${calibration_fit_report}" \
      --output-dir "${artifact_root}/inspection" \
      --detector "${calibration_inspection_detector}" \
      --per-group "${calibration_inspection_per_group}"
    test -s "${artifact_root}/inspection/README.md"
    test -s "${artifact_root}/inspection/model_diagnostics.png"
    test -s "${artifact_root}/inspection/cutouts_science.fits"
    ;;
  foam)
    bands=${FOAM_BANDS:-F090W,F150W}
    crop_size=${FOAM_CROP_SIZE:-32}
    steps=${FOAM_STEPS:-2}
    grid_side=${FOAM_GRID_SIDE:-4}
    exact_max_candidate_pairs=${FOAM_EXACT_MAX_CANDIDATE_PAIRS:-10000000}
    basis_count=${FOAM_BASIS_COUNT:-3}
    extent=${FOAM_EXTENT:-0.55}
    foam_lr=${FOAM_LR:-0.02}
    seed=${FOAM_SEED:-101}
    patt_nums=${FOAM_PATT_NUMS:-}
    exact_operator_storage=${FOAM_EXACT_OPERATOR_STORAGE:-model}
    exact_observation_batch_size=${FOAM_EXACT_OBSERVATION_BATCH_SIZE:-auto}
    exact_max_iterations=${FOAM_EXACT_MAX_ITERATIONS:-50}
    exact_preconditioner=${FOAM_EXACT_PRECONDITIONER:-auto}
    pixi run --manifest-path "${source_root}/pixi.toml" -e smacs-cuda -- \
      python "${source_root}/scripts/run_local_sphere_bench.py" \
      "${source_root}/benchmarks/bench_map_smacs0723_field_fit_joint.py" \
      --bands "${bands}" \
      --exposure-dir "${data_root}/jwst_smacs0723/aligned" \
      --input-product aligned \
      --patt-nums "${patt_nums}" \
      --crop-size "${crop_size}" \
      --crop-strategy best_overlap \
      --steps "${steps}" \
      --seed "${seed}" \
      --foam-grid-side "${grid_side}" \
      --foam-exact-max-candidate-pairs "${exact_max_candidate_pairs}" \
      --foam-basis-count "${basis_count}" \
      --foam-extent "${extent}" \
      --foam-lr "${foam_lr}" \
      --foam-exact-operator-storage "${exact_operator_storage}" \
      --foam-exact-observation-batch-size "${exact_observation_batch_size}" \
      --foam-exact-max-iterations "${exact_max_iterations}" \
      --foam-exact-preconditioner "${exact_preconditioner}" \
      "${foam_geometry_args[@]}" \
      "${foam_initialization_args[@]}" \
      "${foam_footprint_args[@]}" \
      "${foam_spatial_args[@]}" \
      "${foam_spectral_args[@]}" \
      "${foam_exact_refinement_args[@]}" \
      "${operator_calibration_args[@]}" \
      --models spectral_power_foam \
      --fail-on-method-error \
      --json-out "${artifact_root}/shared-foam.json" \
      --markdown-out "${artifact_root}/shared-foam.md" \
      --export-dir "${artifact_root}/products"
    ;;
  foam-validation)
    bands=${FOAM_BANDS:-F090W,F150W,F200W,F277W,F356W,F444W}
    crop_size=${FOAM_CROP_SIZE:-512}
    steps=${FOAM_STEPS:-200}
    grid_side=${FOAM_GRID_SIDE:-32}
    exact_max_candidate_pairs=${FOAM_EXACT_MAX_CANDIDATE_PAIRS:-10000000}
    basis_count=${FOAM_BASIS_COUNT:-4}
    extent=${FOAM_EXTENT:-0.55}
    foam_lr=${FOAM_LR:-0.02}
    seed=${FOAM_SEED:-101}
    patt_nums=${FOAM_PATT_NUMS:-3,8}
    exact_operator_storage=${FOAM_EXACT_OPERATOR_STORAGE:-model}
    exact_observation_batch_size=${FOAM_EXACT_OBSERVATION_BATCH_SIZE:-auto}
    exact_max_iterations=${FOAM_EXACT_MAX_ITERATIONS:-50}
    exact_preconditioner=${FOAM_EXACT_PRECONDITIONER:-auto}
    validation_root=/arc/projects/ots/torchsky-runs/20260729-smacs-validation-32e0940
    target_header_path=${validation_root}/validation-000/common-support-r9-f0p99.fits
    IFS=',' read -r -a validation_batches <<< \
      "${FOAM_VALIDATION_BATCHES:-blank,validation-000,validation-001,validation-002}"
    for batch in "${validation_batches[@]}"; do
      case "${batch}" in
        blank | validation-000 | validation-001 | validation-002) ;;
        *) echo "unsupported Foam validation batch: ${batch}" >&2; exit 2 ;;
      esac
      injection_args=()
      if [[ "${batch}" != blank ]]; then
        injection_args=(
          --injection-manifest "${validation_root}/${batch}/manifest.json"
        )
      fi
      batch_root=${artifact_root}/${batch}
      pixi run --manifest-path "${source_root}/pixi.toml" -e smacs-cuda -- \
        python "${source_root}/scripts/run_local_sphere_bench.py" \
        "${source_root}/benchmarks/bench_map_smacs0723_field_fit_joint.py" \
        --bands "${bands}" \
        --exposure-dir "${data_root}/jwst_smacs0723/aligned" \
        --input-product aligned \
        --patt-nums "${patt_nums}" \
        --crop-size "${crop_size}" \
        --crop-x0 112 \
        --crop-y0 0 \
        --target-header-path "${target_header_path}" \
        --steps "${steps}" \
        --seed "${seed}" \
        --foam-grid-side "${grid_side}" \
        --foam-exact-max-candidate-pairs "${exact_max_candidate_pairs}" \
        --foam-basis-count "${basis_count}" \
        --foam-extent "${extent}" \
        --foam-lr "${foam_lr}" \
        --foam-exact-operator-storage "${exact_operator_storage}" \
        --foam-exact-observation-batch-size "${exact_observation_batch_size}" \
        --foam-exact-max-iterations "${exact_max_iterations}" \
        --foam-exact-preconditioner "${exact_preconditioner}" \
        "${foam_geometry_args[@]}" \
        "${foam_initialization_args[@]}" \
        "${foam_footprint_args[@]}" \
        "${foam_spatial_args[@]}" \
        "${foam_spectral_args[@]}" \
        "${foam_exact_refinement_args[@]}" \
        "${operator_calibration_args[@]}" \
        --models spectral_power_foam \
        --fail-on-method-error \
        "${injection_args[@]}" \
        --json-out "${batch_root}/shared-foam.json" \
        --markdown-out "${batch_root}/shared-foam.md" \
        --export-dir "${batch_root}/products"
    done
    ;;
  foam-score)
    foam_blank_root=${FOAM_BLANK_ROOT:-/arc/projects/ots/torchsky-runs/20260802-smacs-foam-psfinit-g512-e935868}
    foam_injection_root=${FOAM_INJECTION_ROOT:-/arc/projects/ots/torchsky-runs/20260802-smacs-foam-g512-injections-1d055cf}
    validation_root=/arc/projects/ots/torchsky-runs/20260729-smacs-validation-32e0940
    pixi run --manifest-path "${source_root}/pixi.toml" -e sphere-bench -- \
      python "${source_root}/scripts/run_local_sphere_bench.py" \
      "${source_root}/benchmarks/score_smacs0723_foam.py" \
      --blank-root "${foam_blank_root}" \
      --injection-root "${foam_injection_root}" \
      --validation-root "${validation_root}" \
      --out-dir "${artifact_root}/score"
    ;;
  native-band)
    band=${SMACS_BAND:?set SMACS_BAND}
    methods=torchsky_bilinear,torchsky_drizzle_square_pf0p800,torchsky_drizzle_square_pf1p000
    validation_root=/arc/projects/ots/torchsky-runs/20260729-smacs-validation-32e0940
    IFS=',' read -r -a native_batches <<< \
      "${NATIVE_VALIDATION_BATCHES:-blank,validation-000,validation-001,validation-002}"
    for batch in "${native_batches[@]}"; do
      case "${batch}" in
        blank | validation-000 | validation-001 | validation-002) ;;
        *) echo "unsupported native validation batch: ${batch}" >&2; exit 2 ;;
      esac
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
        --target-header-path \
          "${validation_root}/validation-000/common-support-r9-f0p99.fits" \
        --validation-classical \
        --drizzle-pixfracs 0.8,1.0 \
        --methods "${methods}" \
        --fail-on-method-error \
        --product-only \
        "${injection_args[@]}" \
        --product-out-dir "${batch_root}/products" \
        --json-out "${batch_root}/stackers.json" \
        --markdown-out "${batch_root}/stackers.md"
    done
    ;;
  native-score)
    native_input_root_pattern=${NATIVE_INPUT_ROOT_PATTERN-}
    if [[ -z "${native_input_root_pattern}" ]]; then
      native_input_root_pattern='/arc/projects/ots/torchsky-runs/20260729-smacs-native-validation-d8fece0-{band}'
    fi
    pixi run --manifest-path "${source_root}/pixi.toml" -e map-compat -- \
      python "${source_root}/scripts/run_local_sphere_bench.py" \
      "${source_root}/benchmarks/score_smacs0723_native_classical.py" \
      --input-root-pattern "${native_input_root_pattern}" \
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
