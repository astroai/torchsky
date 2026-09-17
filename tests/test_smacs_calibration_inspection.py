from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from types import ModuleType

import torch
import pytest


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))


def test_depth_runner_persists_failure_status(tmp_path):
    runner = (ROOT / "scripts/run_canfar_adass_depth.sh").read_text()
    trap = next(line.strip() for line in runner.splitlines() if line.strip().startswith("trap "))
    result = subprocess.run(
        ["bash", "-c", 'artifact_root=$1\n' + trap + '\nexit 17', "check", str(tmp_path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 17
    assert "run_exit_code=17" in result.stdout
    assert (tmp_path / "exit_status.txt").read_text().startswith("exit_code=17\nfinished_utc=")


def test_depth_foam_uses_cuda_build_and_forwards_size_controls():
    runner = (ROOT / "scripts/run_canfar_adass_depth.sh").read_text()
    selection = runner.split("native_build_env=default", 1)[1].split('if [[ "${dry_run}"', 1)[0]
    assert '"${mode}" == "foam"' in selection
    assert "native_build_env=smacs-cuda" in selection
    assert '"${TORCHSKY_DEVICE:-auto}" == "cpu"' in selection
    assert 'run pixi_run "${native_build_env}"' in runner
    launcher = (ROOT / "scripts/launch_canfar_wcs_sphere.sh").read_text()
    for name in ("FOAM_BANDS", "FOAM_CROP_SIZE", "FOAM_STEPS", "FOAM_GRID_SIDE",
                 "FOAM_BASIS_COUNT", "FOAM_INITIALIZE_FROM_COADD",
                 "FOAM_AFFINE_SPATIAL_BASIS", "FOAM_WHITEN_SPECTRAL_BASIS",
                 "FOAM_TARGET_HEADER_PATH", "SCENE_INITIALIZATION",
                 "SCENE_INITIALIZATION_STEPS", "SCENE_INITIALIZATION_LR"):
        assert f'--env "{name}=' in launcher
    assert '--target-header-path "${FOAM_TARGET_HEADER_PATH}"' in runner
    assert '--scene-initialization "${scene_initialization}"' in runner
    assert '--scene-initialization-steps "${scene_initialization_steps}"' in runner
    assert '--scene-initialization-lr "${scene_initialization_lr}"' in runner


def _inspection_module() -> ModuleType:
    path = BENCHMARKS / "export_smacs0723_calibration_inspection.py"
    spec = importlib.util.spec_from_file_location("smacs_calibration_inspection", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import inspection module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_inspection_selects_exact_fit_scope() -> None:
    inspection = _inspection_module()
    legacy = {"detector": "NRCB4", "role": "background", "family": "stationary"}
    scoped = [dict(legacy, exposure_id=exposure, band=band)
              for exposure, band in (("a", "F150W"), ("b", "F150W"), ("a", "F200W"))]
    report = {"groups": [legacy, *scoped]}
    assert inspection._fit_group(report, **legacy) is legacy
    assert inspection._fit_group(report, **legacy, exposure_id="a", band="f150w") is scoped[0]
    with pytest.raises(ValueError, match="expected one"):
        inspection._fit_group(report, **legacy, exposure_id="missing", band="F150W")
    with pytest.raises(ValueError, match="expected one"):
        inspection._fit_group({"groups": scoped}, **legacy)
    with pytest.raises(ValueError, match="expected one"):
        inspection._fit_group({"groups": [dict(legacy, calibration_scope="exposure")]}, **legacy)


@pytest.mark.parametrize("exposure_id,band", [("a", None), (None, "F150W"), ("", "F150W")])
def test_inspection_requires_paired_scope_selectors(tmp_path, exposure_id, band) -> None:
    inspection = _inspection_module()
    with pytest.raises(ValueError, match="exposure_id and band together"):
        inspection.export_inspection_bundle(
            tmp_path / "missing.json", tmp_path / "missing-fit.json", tmp_path / "output",
            exposure_id=exposure_id, band=band,
        )
    assert not (tmp_path / "output").exists()


def test_inspection_exposure_rows_use_calibration_split(monkeypatch) -> None:
    inspection = _inspection_module()
    base = {"detector": "NRCB4", "role": "background", "split": "test", "band": "F150W"}
    evidence = [dict(base, exposure_id=exposure, calibration_split=split,
                     source_id=f"{exposure}-{split}")
                for exposure in ("a", "b") for split in ("train", "validation", "test")]
    evidence.append(dict(evidence[0], band="F200W", source_id="other-band"))
    report = {"groups": [dict(base, exposure_id="a", family=family, validation_pruning={})
                         for family in ("stationary", "adaptive")]}
    calls = []

    def load(manifest, **kwargs):
        calls.append(kwargs)
        return {"target_values": torch.tensor([2.0])}

    monkeypatch.setattr(inspection, "load_operator_calibration_split", load)
    monkeypatch.setattr(inspection, "_load_background_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(inspection, "apply_operator_calibration_pruning", lambda *args: None)
    monkeypatch.setattr(inspection, "_predict", lambda *args: torch.tensor([3.0]))
    rows = inspection._prediction_rows(Path("unused"), {"evidence": evidence}, report,
                                       detector="NRCB4", exposure_id="a", band="F150W")
    assert [row["source_id"] for row in rows] == ["a-train", "a-validation", "a-test"]
    assert [row["split"] for row in rows] == ["train", "validation", "test"]
    assert all(row["exposure_split"] == "test" and row["pruned_residual"] == 1 for row in rows)
    assert all(call["exposure_id"] == "a" and call["band"] == "F150W" for call in calls)


def test_inspection_rejected_rows_remain_exposure_scoped() -> None:
    inspection = _inspection_module()
    exposures = [{"path": name, "resolved_path": f"/data/{name}", "filename": name,
                  "band": band, "detector": "NRCB4", "patt_num": 1, "split": "test",
                  "timing": {"midpoint_mjd_utc": 60000}}
                 for name, band in (("a", "F150W"), ("b", "F150W"), ("c", "F200W"))]
    rejected = [{"exposure": exposure["path"], "role": "background"} for exposure in exposures]
    payload = {"exposures": exposures, "extraction": {"rejected": rejected}}
    rows = inspection._rejected_rows(payload, detector="NRCB4", exposure_id="a", band="F150W")
    assert len(rows) == 1
    assert rows[0]["split"] == "unassigned"
    assert rows[0]["exposure_split"] == "test"
    assert len(inspection._rejected_rows(payload, detector="NRCB4")) == 3


def test_stationary_diagnostics_only_does_not_read_exposures(tmp_path, monkeypatch):
    inspection = _inspection_module()
    evidence = [{"role": "background", "detector": "NRCB4", "exposure_id": "a",
                 "band": "F150W", "split": "train", "calibration_split": split,
                 "source_id": split, "detector_x": 1, "detector_y": 2,
                 "time_mjd_utc": 60000}
                for split in ("train", "validation", "test")]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"evidence": evidence, "exposures": [],
                                    "extraction": {"rejected": []}}))
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"groups": [{"role": "background", "detector": "NRCB4",
        "family": "stationary", "exposure_id": "a", "band": "F150W"}]}))
    monkeypatch.setattr(inspection, "_load_background_model", lambda *a, **kw: None)
    monkeypatch.setattr(inspection, "load_operator_calibration_split",
                        lambda *a, **kw: {"target_values": torch.tensor([2.0])})
    monkeypatch.setattr(inspection, "_predict", lambda *a: torch.tensor([3.0]))
    monkeypatch.setattr(inspection, "_exposure_planes", lambda *a: pytest.fail("read raw exposure"))
    output = tmp_path / "output"
    readme = inspection.export_inspection_bundle(manifest, report, output,
        exposure_id="a", band="F150W", diagnostics_only=True)
    assert "stationary" in readme.read_text()
    assert "adaptive" not in readme.read_text()
    csv = (output / "background_predictions.csv").read_text()
    assert "stationary_prediction" in csv
    assert "adaptive" not in csv and "pruned" not in csv
    assert (output / "model_diagnostics.png").stat().st_size > 0
    assert not list(output.glob("cutouts*"))
    assert not (output / "contact_sheets").exists()
    assert json.loads((output / "provenance.json").read_text())["source_revision"]


def test_psf_validation_diagnostics_match_masked_normalization(tmp_path, monkeypatch):
    inspection = _inspection_module()
    arrays = {"detector_x": torch.tensor([1.0]), "detector_y": torch.tensor([2.0]),
              "wavelength": torch.tensor([1.5]), "time_mjd": torch.tensor([60000.0]),
              "sample_weight": torch.ones(1), "target_kernels": torch.ones(1, 3, 3),
              "pixel_weight": torch.ones(1, 3, 3)}
    arrays["pixel_weight"][0, 0, 0] = 0
    model, _ = inspection.fit_calibration._make_model(arrays, role="psf", family="stationary",
                                                     device=torch.device("cpu"), exposure_scope=True)
    checkpoint = tmp_path / "psf.pt"
    torch.save({"state_dict": model.state_dict()}, checkpoint)
    group = {"detector": "NRCB4", "role": "psf", "family": "stationary",
             "exposure_id": "a", "band": "F150W", "checkpoint": str(checkpoint)}
    payload = {"evidence": [dict(group, source_id="star", calibration_split="validation")]}
    monkeypatch.setattr(inspection, "load_operator_calibration_split", lambda *a, **kw: arrays)
    (tmp_path / "README.md").write_text("")
    inspection._write_psf_diagnostics(tmp_path / "unused.json", payload, {"groups": [group]},
        tmp_path, detector="NRCB4", exposure_id="a", band="F150W", per_group=2, overwrite=False)
    target = inspection.read(str(tmp_path / "psf_validation_measured.fits"))
    predicted = inspection.read(str(tmp_path / "psf_validation_stationary.fits"))
    residual = inspection.read(str(tmp_path / "psf_validation_residual.fits"))
    assert target[0, 0, 0] == predicted[0, 0, 0] == 0
    torch.testing.assert_close(target.sum(), torch.tensor(1.0))
    torch.testing.assert_close(predicted.sum(), torch.tensor(1.0))
    torch.testing.assert_close(residual, predicted - target)
    term = inspection.fit_calibration._term(model, arrays, role="psf", name="reference")
    torch.testing.assert_close((residual.square() * arrays["pixel_weight"]).sum() / 8,
                               term.raw_loss())
    assert (tmp_path / "psf_validation.png").stat().st_size > 0


def test_inspection_selects_worst_residuals_and_writes_png(tmp_path: Path) -> None:
    inspection = _inspection_module()
    accepted = [
        {
            "status": "accepted",
            "split": "test",
            "band": "F150W",
            "source_id": f"accepted-{index}",
            "exposure": "exposure.fits",
            "pruned_residual": float(index),
            "pruned_absolute_error": float(index),
        }
        for index in range(4)
    ]
    rejected = [
        {
            "status": "rejected",
            "split": "test",
            "band": "F150W",
            "source_id": "rejected-0",
            "exposure": "exposure.fits",
            "reason": "insufficient_support",
        }
    ]
    selected = inspection._selected_rows(accepted, rejected, per_group=2)
    assert [row["source_id"] for row in selected[:2]] == [
        "accepted-3",
        "accepted-2",
    ]
    assert selected[-1]["source_id"] == "rejected-0"

    output = tmp_path / "sheet.png"
    inspection._write_contact_sheet(
        output,
        [(selected[0], torch.arange(25, dtype=torch.float32).reshape(5, 5))],
    )
    assert output.stat().st_size > 0


def test_canfar_inspection_uses_visualization_environment() -> None:
    runner = (ROOT / "scripts" / "run_canfar_adass_depth.sh").read_text(
        encoding="utf-8"
    )
    inspection = runner.split("  calibration-inspect)", 1)[1].split("  foam)", 1)[0]
    assert inspection.count("pixi_run bench \\\n") == 2


def test_canfar_full_mosaic_parity_is_strict_and_torchfits_only() -> None:
    runner = (ROOT / "scripts" / "run_canfar_adass_depth.sh").read_text(
        encoding="utf-8"
    )
    parity = runner.split("  mosaic-parity)", 1)[1].split("  native-score)", 1)[0]

    assert "--full-mosaic-parity" in parity
    assert "--expected-exposures" in parity
    assert "bootstrap_map_compat_tools.py" not in parity
    assert "--force-source" not in parity
    assert "astroai/improc:latest" in runner
    assert "parity_env=${TORCHSKY_MAP_COMPAT_ENV:-map-compat}" in parity
    assert parity.count('pixi_run "${parity_env}"') == 3
    assert "torch.cuda.is_available()" in parity
    assert "torch.cuda.get_device_name()" in parity
    assert "import torchfits" in parity
    assert "astropy.io" not in parity
    assert "native_build_env=${TORCHSKY_MAP_COMPAT_ENV:-map-compat}" in runner
    assert 'ensure_native_build "${native_build_env}"' in runner


def test_canfar_multiband_canary_runs_shared_spectral_comparison() -> None:
    runner = (ROOT / "scripts" / "run_canfar_adass_depth.sh").read_text(
        encoding="utf-8"
    )
    canary = runner.split("  multiband-canary)", 1)[1].split(
        "  mosaic-parity)", 1
    )[0]

    assert "--profile classical-smoke" in canary
    assert "--include-foam" in canary
    assert '"${TORCHSKY_DEVICE:-cuda}"' in canary
    assert 'gate "${artifact_root}/multiband-canary.json"' in canary


def test_canfar_depth_runner_accepts_authorized_skaha_compute_images() -> None:
    runner = (ROOT / "scripts" / "run_canfar_adass_depth.sh").read_text(
        encoding="utf-8"
    )

    assert "skaha/base-notebook:latest" in runner
    assert "skaha/astroflow-cuda:25.03" in runner


def test_canfar_depth_runner_replaces_stale_pixi() -> None:
    runner = (ROOT / "scripts" / "run_canfar_adass_depth.sh").read_text(
        encoding="utf-8"
    )

    assert "pixi_version=${TORCHSKY_PIXI_VERSION:-0.78.0}" in runner
    assert '"$(pixi --version 2>/dev/null)" != "pixi ${pixi_version}"' in runner
    assert 'export PIXI_VERSION="${pixi_version}"' in runner


def test_map_compat_cuda_environment_uses_cuda_and_official_comparators() -> None:
    pixi = tomllib.loads((ROOT / "pixi.toml").read_text(encoding="utf-8"))
    environment = pixi["environments"]["map-compat-cuda"]

    assert environment["no-default-feature"] is True
    assert set(environment["features"]) == {"cuda", "map-compat"}


def test_canfar_runtime_is_selected_after_each_environment_activation(tmp_path):
    runner = (ROOT / "scripts/run_canfar_adass_depth.sh").read_text()
    function = (
        "pixi_run() {" + runner.split("pixi_run() {", 1)[1].split("\n}", 1)[0] + "\n}"
    )
    # Simulate activation overwriting the inherited library path. The command
    # must still see the runtime for its own environment, not the previous one.
    mock = """
pixi() {
  while [[ "$1" != "--" ]]; do shift; done
  shift
  LD_LIBRARY_PATH=/wrong/activation PYTHONNOUSERSITE=0 "$@"
}
source_root=$1
shift
pixi_run "$@"
"""
    inherited = dict(os.environ, LD_LIBRARY_PATH="/existing/runtime")
    probe = "import json, os; print(json.dumps([os.environ['LD_LIBRARY_PATH'], os.environ['PYTHONNOUSERSITE']]))"
    for environment in ("default", "bench", "smacs-cuda", "map-compat-cuda"):
        result = subprocess.run(
            [
                "bash",
                "-c",
                function + mock,
                "runtime-check",
                str(tmp_path),
                environment,
                sys.executable,
                "-c",
                probe,
            ],
            check=True,
            capture_output=True,
            text=True,
            env=inherited,
        )
        assert json.loads(result.stdout) == [
            f"{tmp_path}/.pixi/envs/{environment}/lib:/existing/runtime",
            "1",
        ]
    assert runner.count("pixi run --manifest-path") == 1
