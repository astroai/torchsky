#!/usr/bin/env bash
set -euo pipefail

mode=${1:?usage: run_canfar_adass_depth.sh MODE RUN_ID}
run_id=${2:?usage: run_canfar_adass_depth.sh MODE RUN_ID}
commit=${TORCHSKY_COMMIT:?set TORCHSKY_COMMIT to an exact pushed SHA}
container_image=${CONTAINER_IMAGE:-images.canfar.net/astroai/notebook:latest}
artifact_root=${ARTIFACT_ROOT:-/arc/projects/ots/torchsky-runs/${run_id}}
source_root=${SOURCE_ROOT:-/scratch/${run_id}-src}
data_root=${TORCHSKY_DATA_ROOT:-/arc/projects/ots/torchsky-data}
dry_run=${DRY_RUN:-0}

if [[ "${dry_run}" != "1" ]]; then
  case "${container_image}" in
    images.canfar.net/astroai/* | astroai/*) ;;
    *)
      echo "CONTAINER_IMAGE must name an AstroAI image, got: ${container_image}" >&2
      exit 2
      ;;
  esac

  mkdir -p "${artifact_root}"
  exec > >(tee -a "${artifact_root}/session.log") 2>&1
fi

echo "run_id=${run_id}"
echo "mode=${mode}"
echo "requested_commit=${commit}"
echo "image=${container_image}"
echo "started_utc=$(date -u +%FT%TZ)"
nvidia-smi || true

if [[ "${dry_run}" = "1" ]]; then
  test -d "${source_root}"
  echo "dry_run commit=${commit} source_root=${source_root}"
elif [[ -d "${source_root}/.git" ]] \
  && [[ "$(git -C "${source_root}" rev-parse HEAD 2>/dev/null)" == "${commit}" ]]; then
  # full mode chains sub-modes; each re-invocation reuses the existing
  # exact-SHA checkout instead of re-cloning.
  echo "reusing checkout at ${commit}"
else
  rm -rf "${source_root}"
  if command -v gh >/dev/null && gh auth status >/dev/null 2>&1; then
    gh repo clone sfabbro/torchsky "${source_root}" -- --filter=blob:none
  else
    git clone --filter=blob:none https://github.com/sfabbro/torchsky.git "${source_root}"
  fi
  git -C "${source_root}" checkout --detach "${commit}"
  test "$(git -C "${source_root}" rev-parse HEAD)" = "${commit}"
  git -C "${source_root}" status --short
fi

export PIXI_HOME=/scratch/pixi
export PIXI_CACHE_DIR=/scratch/pixi-cache
export TORCHSKY_DATA="${data_root}"
export PATH="${PIXI_HOME}/bin:${PATH}"
export PYTHONPATH="${source_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
# Build/install the local editable torchsky (incl. torchsky._cpp) only when
# it is not already importable -- sub-invocations of the full chain skip it.
ensure_native_build() {
  if pixi run --manifest-path "${source_root}/pixi.toml" -e default -- \
      python -c "import torchsky; import torchsky.cpp" >/dev/null 2>&1; then
    echo "torchsky native build already available; skipping install"
    return 0
  fi
  pixi run --manifest-path "${source_root}/pixi.toml" -e default -- \
    python -m pip install -e "${source_root}" --no-build-isolation
}

if ! command -v pixi >/dev/null 2>&1; then
  curl -fsSL https://pixi.sh/install.sh | bash
fi
pixi --version

# DRY_RUN=1 prints every command without executing and relaxes artifact gates.
run() {
  if [[ "${dry_run}" = "1" ]]; then
    echo "DRY_RUN:" "$*"
  else
    "$@"
  fi
}
gate() {
  if [[ "${dry_run}" != "1" ]]; then
    test -s "$1"
  fi
}

# --- scene-representation zoo (FIELD_MODELS) --------------------------------
field_models=${FIELD_MODELS:-spectral_power_foam}
IFS=',' read -r -a field_model_list <<< "${field_models}"
foam_selected=0
for kind in "${field_model_list[@]+${field_model_list[@]}}"; do
  [[ -n "${kind}" ]] || continue
  if [[ "${kind}" == "spectral_power_foam" ]]; then
    foam_selected=1
  fi
done
# Validate the CSV against the registry (abort with the allowed list).
# The editable install also builds torchsky._cpp, which every later step's
# bare PYTHONPATH import needs.
if [[ "${dry_run}" != "1" ]]; then
ensure_native_build
pixi run --manifest-path "${source_root}/pixi.toml" -e default -- \
  python - "${field_models}" <<'PYVALIDATE'
import sys
from torchsky.map.field import FIELD_MODEL_KINDS
requested = [part.strip() for part in sys.argv[1].split(",") if part.strip()]
unknown = [kind for kind in requested if kind not in set(FIELD_MODEL_KINDS)]
if unknown:
    allowed = ", ".join(sorted(set(FIELD_MODEL_KINDS)))
    sys.exit(f"unknown FIELD_MODELS entries {unknown}; allowed: {allowed}")
print(f"FIELD_MODELS validated: {requested}")
PYVALIDATE
fi

# --- hybrid operator arms ----------------------------------------------------
case "${PSF_MODEL:-fitted_gaussian}" in
  fitted_gaussian) psf_model=fitted_gaussian ;;
  learned_lowrank) psf_model=learned_lowrank ;;
  *)
    echo "PSF_MODEL must be fitted_gaussian or learned_lowrank, got: ${PSF_MODEL}" >&2
    exit 2
    ;;
esac
case "${BACKGROUND_MODEL:-plane}" in
  plane) background_model=plane ;;
  learned) background_model=learned ;;
  *)
    echo "BACKGROUND_MODEL must be plane or learned, got: ${BACKGROUND_MODEL}" >&2
    exit 2
    ;;
esac
operator_arm_args=(--psf-model "${psf_model}" --background-model "${background_model}")

# Foam-specific arguments only apply when foam is in the model CSV.
if [[ "${foam_selected}" = "1" ]]; then
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
else
  foam_geometry_args=()
  foam_initialization_args=()
  foam_footprint_args=()
  foam_spatial_args=()
  foam_spectral_args=()
  foam_exact_refinement_args=()
fi

operator_calibration_args=()
# The calibration-fit stage runs with the manifest only (it PRODUCES the fit
# report), so require both envs only when wiring calibrated operators into
# a scene fit; partial envs are legitimate in earlier chain stages.
if [[ -n "${CALIBRATION_EVIDENCE_MANIFEST:-}" && -n "${CALIBRATION_FIT_REPORT:-}" ]]; then
  operator_calibration_args=(
    --operator-calibration-manifest "${CALIBRATION_EVIDENCE_MANIFEST}"
    --operator-calibration-fit-report "${CALIBRATION_FIT_REPORT}"
  )
elif [[ -n "${CALIBRATION_EVIDENCE_MANIFEST:-}" || -n "${CALIBRATION_FIT_REPORT:-}" ]]; then
  echo "calibration operator envs partially set; skipping operator_calibration_args" >&2
fi

# --- module fan-out ----------------------------------------------------------
IFS=',' read -r -a modules <<< "${MODULES:-b}"
for module in "${modules[@]+${modules[@]}}"; do
  case "${module}" in
    a | b | A | B) ;;
    *) echo "unsupported module: ${module} (use a and/or b)" >&2; exit 2 ;;
  esac
done

fit_psf_model=fitted_gaussian_kernel
if [[ "${psf_model}" == "learned_lowrank" ]]; then
  fit_psf_model=learned_lowrank
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
    ensure_native_build
    pixi run --manifest-path "${source_root}/pixi.toml" -e default -- \
      python -c 'from torchsky.cli import main; raise SystemExit(main())' \
      calibration-extract \
      "${calibration_input_root}/${calibration_pattern}" \
      --output-dir "${artifact_root}/evidence"
    gate "${artifact_root}/evidence/calibration_evidence_manifest.json"
    ;;
  calibration-fit)
    calibration_manifest=${CALIBRATION_EVIDENCE_MANIFEST:?set CALIBRATION_EVIDENCE_MANIFEST}
    calibration_steps=${CALIBRATION_FIT_STEPS:-300}
    calibration_lr=${CALIBRATION_FIT_LR:-0.001}
    calibration_device=${CALIBRATION_FIT_DEVICE:-cpu}
    test -s "${calibration_manifest}"
    ensure_native_build
    pixi run --manifest-path "${source_root}/pixi.toml" -e default -- \
      python "${source_root}/benchmarks/fit_smacs0723_operator_calibration.py" \
      --manifest "${calibration_manifest}" \
      --output-dir "${CALIBRATION_FIT_OUTPUT_DIR:-${artifact_root}/fit}" \
      --steps "${calibration_steps}" \
      --lr "${calibration_lr}" \
      --device "${calibration_device}" \
      --psf-model "${CALIBRATION_FIT_PSF_MODEL:-${fit_psf_model}}"
    gate "${artifact_root}/fit/operator_calibration_fit.json"
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
    gate "${artifact_root}/inspection/README.md"
    gate "${artifact_root}/inspection/model_diagnostics.png"
    gate "${artifact_root}/inspection/cutouts_science.fits"
    ;;
  foam)
    for module in "${modules[@]+${modules[@]}}"; do
      module_root="${artifact_root}/${module}"
      mkdir -p "${module_root}"
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
      run pixi run --manifest-path "${source_root}/pixi.toml" -e smacs-cuda -- \
        python "${source_root}/scripts/run_local_sphere_bench.py" \
        "${source_root}/benchmarks/bench_map_smacs0723_field_fit_joint.py" \
        --bands "${bands}" \
        --module "${module}" \
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
        "${foam_geometry_args[@]+${foam_geometry_args[@]}}" \
        "${foam_initialization_args[@]+${foam_initialization_args[@]}}" \
        "${foam_footprint_args[@]+${foam_footprint_args[@]}}" \
        "${foam_spatial_args[@]+${foam_spatial_args[@]}}" \
        "${foam_spectral_args[@]+${foam_spectral_args[@]}}" \
        "${foam_exact_refinement_args[@]+${foam_exact_refinement_args[@]}}" \
        "${operator_calibration_args[@]+${operator_calibration_args[@]}}" \
        "${operator_arm_args[@]+${operator_arm_args[@]}}" \
        --models "${field_models}" \
        --fail-on-method-error \
        --json-out "${module_root}/shared-foam.json" \
        --markdown-out "${module_root}/shared-foam.md" \
        --export-dir "${module_root}/products"
      gate "${module_root}/shared-foam.json"
    done
    ;;
  foam-validation)
    for module in "${modules[@]+${modules[@]}}"; do
      module_root="${artifact_root}/${module}"
      mkdir -p "${module_root}"
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
      validation_root=${FOAM_VALIDATION_ROOT:-/arc/projects/ots/torchsky-runs/20260729-smacs-validation-32e0940}
      target_header_path=${validation_root}/validation-000/common-support-r9-f0p99.fits
      IFS=',' read -r -a validation_batches <<< \
        "${FOAM_VALIDATION_BATCHES:-blank,validation-000,validation-001,validation-002}"
      for batch in "${validation_batches[@]+${validation_batches[@]}}"; do
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
        batch_root=${module_root}/${batch}
        run pixi run --manifest-path "${source_root}/pixi.toml" -e smacs-cuda -- \
          python "${source_root}/scripts/run_local_sphere_bench.py" \
          "${source_root}/benchmarks/bench_map_smacs0723_field_fit_joint.py" \
          --bands "${bands}" \
          --module "${module}" \
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
          "${foam_geometry_args[@]+${foam_geometry_args[@]}}" \
          "${foam_initialization_args[@]+${foam_initialization_args[@]}}" \
          "${foam_footprint_args[@]+${foam_footprint_args[@]}}" \
          "${foam_spatial_args[@]+${foam_spatial_args[@]}}" \
          "${foam_spectral_args[@]+${foam_spectral_args[@]}}" \
          "${foam_exact_refinement_args[@]+${foam_exact_refinement_args[@]}}" \
          "${operator_calibration_args[@]+${operator_calibration_args[@]}}" \
          "${operator_arm_args[@]+${operator_arm_args[@]}}" \
          --models "${field_models}" \
          --fail-on-method-error \
          "${injection_args[@]+${injection_args[@]}}" \
          --json-out "${batch_root}/shared-foam.json" \
          --markdown-out "${batch_root}/shared-foam.md" \
          --export-dir "${batch_root}/products"
        gate "${batch_root}/shared-foam.json"
      done
    done
    ;;
  foam-score)
    foam_blank_root=${FOAM_BLANK_ROOT:-/arc/projects/ots/torchsky-runs/20260802-smacs-foam-psfinit-g512-e935868}
    foam_injection_root=${FOAM_INJECTION_ROOT:-/arc/projects/ots/torchsky-runs/20260802-smacs-foam-g512-injections-1d055cf}
    validation_root=${FOAM_VALIDATION_ROOT:-/arc/projects/ots/torchsky-runs/20260729-smacs-validation-32e0940}
    pixi run --manifest-path "${source_root}/pixi.toml" -e sphere-bench -- \
      python "${source_root}/scripts/run_local_sphere_bench.py" \
      "${source_root}/benchmarks/score_smacs0723_foam.py" \
      --blank-root "${foam_blank_root}" \
      --injection-root "${foam_injection_root}" \
      --validation-root "${validation_root}" \
      --out-dir "${artifact_root}/score"
    ;;
  field-score)
    score_input_root=${SCORE_INPUT_ROOT:?set SCORE_INPUT_ROOT to a run root containing <module>/<psf_mode>/<model> products}
    score_methods=${SCORE_METHODS:?set SCORE_METHODS to a CSV of psf_mode/model_kind pairs, e.g. fitted_gaussian/spectral_power_foam,learned_lowrank/spectral_power_foam}
    score_validation_root=${SCORE_VALIDATION_ROOT:-/arc/projects/ots/torchsky-runs/20260729-smacs-validation-32e0940}
    score_out_dir=${SCORE_OUT_DIR:-${artifact_root}/model-selection}
    pixi run --manifest-path "${source_root}/pixi.toml" -e sphere-bench -- \
      python "${source_root}/scripts/run_local_sphere_bench.py" \
      "${source_root}/benchmarks/score_field_models.py" \
      --input-root "${score_input_root}" \
      --methods "${score_methods}" \
      --validation-root "${score_validation_root}" \
      --out-dir "${score_out_dir}"
    gate "${score_out_dir}/model_selection.json"
    ;;
  native-band)
    IFS=',' read -r -a native_band_list <<< "${SMACS_BAND:-${FOAM_BANDS:-F090W,F150W,F200W,F277W,F356W,F444W}}"
    methods=torchsky_bilinear,torchsky_drizzle_square_pf0p800,torchsky_drizzle_square_pf1p000
    validation_root=${FOAM_VALIDATION_ROOT:-/arc/projects/ots/torchsky-runs/20260729-smacs-validation-32e0940}
    IFS=',' read -r -a native_batches <<< \
      "${NATIVE_VALIDATION_BATCHES:-blank,validation-000,validation-001,validation-002}"
    for module in "${modules[@]+${modules[@]}}"; do
      module_root="${artifact_root}/${module}"
      for nband in "${native_band_list[@]+${native_band_list[@]}}"; do
      band=${nband}
      for batch in "${native_batches[@]+${native_batches[@]}}"; do
        case "${batch}" in
          blank | validation-000 | validation-001 | validation-002) ;;
          *) echo "unsupported native validation batch: ${batch}" >&2; exit 2 ;;
        esac
        batch_root=${module_root}/${batch}/${band}
        mkdir -p "${batch_root}/products"
        injection_args=()
        if [[ "${batch}" != blank ]]; then
          injection_args=(
            --injection-manifest "${validation_root}/${batch}/manifest.json"
          )
        fi
        run pixi run --manifest-path "${source_root}/pixi.toml" -e map-compat -- \
          python "${source_root}/scripts/run_local_sphere_bench.py" \
          "${source_root}/benchmarks/bench_map_smacs0723_stackers.py" \
          --band "${band}" \
          --module "${module}" \
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
          "${injection_args[@]+${injection_args[@]}}" \
          --product-out-dir "${batch_root}/products" \
          --json-out "${batch_root}/stackers.json" \
          --markdown-out "${batch_root}/stackers.md"
        gate "${batch_root}/stackers.json"
      done
      done
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
  full)
    # Autonomous chain: calibration -> scene zoo x PSF arms -> native band -> selection.
    # Module A waits until its aligned parallel-field products are staged;
    # override with MODULES=a,b once they exist.
    full_modules=${MODULES:-b}
    full_bands=${FOAM_BANDS:-F090W,F150W,F200W,F277W,F356W,F444W}
    echo "full mode: modules=${full_modules} bands=${full_bands} field_models=${field_models} psf=${psf_model} background=${background_model}"

    stage_done() { [[ -s "$1" ]]; }

    if stage_done "${artifact_root}/evidence/calibration_evidence_manifest.json"; then
      echo "skipping calibration-extract (manifest exists)"
    else
      "${BASH_SOURCE[0]}" calibration-extract "${run_id}"
      gate "${artifact_root}/evidence/calibration_evidence_manifest.json"
    fi

    # Dual PSF arms: implicit Fourier-conditioned kernel bank vs learned
    # low-rank chromatic control grid. Each gets its own calibration fit +
    # foam run; the scorer compares them as separate cells.
    for psf_arm in fitted_gaussian_kernel learned_lowrank; do
      arm_dir="${artifact_root}/fit/${psf_arm}"
      if stage_done "${arm_dir}/operator_calibration_fit.json"; then
        echo "skipping calibration-fit ${psf_arm} (report exists)"
      else
        CALIBRATION_EVIDENCE_MANIFEST="${artifact_root}/evidence/calibration_evidence_manifest.json" \
          "${BASH_SOURCE[0]}" calibration-fit "${run_id}"           --output-dir "${arm_dir}" --psf-model "${psf_arm}"
        gate "${arm_dir}/operator_calibration_fit.json"
      fi
    done

    for psf_arm in fitted_gaussian_kernel learned_lowrank; do
      arm_dir="${artifact_root}/fit/${psf_arm}"
      report="${arm_dir}/operator_calibration_fit.json"
      if ! stage_done "${report}"; then
        echo "skipping foam ${psf_arm} (no calibration report)" >&2
        continue
      fi
      CALIBRATION_EVIDENCE_MANIFEST="${artifact_root}/evidence/calibration_evidence_manifest.json" \
      CALIBRATION_FIT_REPORT="${report}" \
      FOAM_BANDS="${full_bands}" \
      MODULES="${full_modules}" \
        "${BASH_SOURCE[0]}" foam "${run_id}"
      gate "${artifact_root}/shared-foam.json"
    done
    for module in ${full_modules//,/ }; do
      gate "${artifact_root}/${module}/shared-foam.json"
    done
    all_val_done=1
    for module in ${full_modules//,/ }; do
      for batch in blank validation-000 validation-001 validation-002; do
        stage_done "${artifact_root}/${module}/${batch}/shared-foam.json" || all_val_done=0
      done
    done
    if [[ "${all_val_done}" = "1" ]]; then
      echo "skipping foam-validation (all batches exist)"
    else
      CALIBRATION_EVIDENCE_MANIFEST="${artifact_root}/evidence/calibration_evidence_manifest.json" \
      CALIBRATION_FIT_REPORT="${artifact_root}/fit/operator_calibration_fit.json" \
      FOAM_BANDS="${full_bands}" \
      MODULES="${full_modules}" \
        "${BASH_SOURCE[0]}" foam-validation "${run_id}"
    fi
    for module in ${full_modules//,/ }; do
      gate "${artifact_root}/${module}/blank/shared-foam.json"
    done
    MODULES="${full_modules}" \
      "${BASH_SOURCE[0]}" native-band "${run_id}"
    # Model selection: every (psf_mode x model_kind) product cell under the run root.
    score_methods=$(python3 - "${field_models}" "${psf_model}" <<'PYMETHODS'
import sys
models = [m.strip() for m in sys.argv[1].split(",") if m.strip()]
if sys.argv[2] == "learned_lowrank":
    psf_modes = ["fitted_gaussian", "learned_lowrank"]
else:
    psf_modes = ["fitted_gaussian"]
print(",".join(f"{psf}/{model}" for psf in psf_modes for model in models))
PYMETHODS
)
    SCORE_INPUT_ROOT="${artifact_root}" \
    SCORE_METHODS="${score_methods}" \
    SCORE_OUT_DIR="${artifact_root}/model-selection" \
      "${BASH_SOURCE[0]}" field-score "${run_id}"
    gate "${artifact_root}/model-selection/model_selection.json"
    ;;
  *)
    echo "unknown mode: ${mode}" >&2
    exit 2
    ;;
esac

echo "completed_utc=$(date -u +%FT%TZ)"
