from __future__ import annotations

import sys
import json
from pathlib import Path

import pytest


BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

from compare_smacs_grid_mapping import compare_runs, main  # noqa: E402


def _payload(*, footprint: bool, seed: int = 101) -> dict:
    bounds = [-0.0012, 0.0012, -0.0012, 0.0012] if footprint else None
    return {
        "experiment": {
            "input_product": "aligned",
            "exposure_dir": "/arc/data/aligned",
            "bands": ["F150W", "F277W"],
            "module": "b",
            "psf_model": "fitted_gaussian",
            "background_model": "learned",
            "background_reference": "training_mean",
            "independent_frozen_calibration": True,
            "calibration_family": "stationary",
            "measured_passband_dir": "/arc/calibration/svo",
            "optimizer_steps": 600,
            "seed": seed,
            "patt_nums": [1, 2, 3, 4, 5, 7, 8],
            "target_header_path": "/arc/target_128.fits",
            "operator_calibration_manifest": "/arc/calibration/manifest.json",
            "operator_calibration_fit_report": "/arc/calibration/fit.json",
            "grid_fit_target_footprint": footprint,
            "grid_psf_margin_sigma": 4.0,
            "validation_detector_scoring": {
                "scores_by_model": {
                    "explicit_spectral_grid_field": [
                        {
                            "band": "F150W",
                            "gaussian_nll_per_pixel": -2.0 if footprint else -1.8,
                            "chi2_per_pixel": 1.6 if footprint else 1.9,
                        },
                        {
                            "band": "F277W",
                            "gaussian_nll_per_pixel": -1.4 if footprint else -1.3,
                            "chi2_per_pixel": 2.1 if footprint else 2.3,
                        },
                    ]
                }
            },
        },
        "effective_model_configs": {
            "explicit_spectral_grid_field": {
                "model_kwargs": {
                    "spatial_resolution": 256,
                    "spectral_resolution": 4,
                    **({"spatial_bounds_local_deg": bounds} if bounds else {}),
                },
                "fit_kwargs": {"steps": 600, "lr": 0.003},
            }
        },
        "rows": [
            {
                "band": band,
                "model": "explicit_spectral_grid_field",
                "status": "OK",
                "source_contrast_rmse": 0.05 if footprint else 0.06,
                "source_rmse": 0.055 if footprint else 0.065,
                "support_rmse": 0.019 if footprint else 0.021,
                "aligned_rmse": 0.056 if footprint else 0.066,
                "source_contrast_slope": 0.62 if footprint else 0.55,
                "support_flux_ratio": 0.97 if footprint else 0.94,
                "source_flux_ratio": 0.91 if footprint else 0.84,
                "fit_time_ms": 600_000.0 if footprint else 560_000.0,
                "peak_accelerator_memory_bytes": 50_000_000,
                "grid_spatial_mapping": (
                    "target_footprint_linear" if footprint else "legacy_tanh_domain"
                ),
                "grid_spatial_bounds_local_deg": bounds,
                "grid_target_intervals_x": 245.0 if footprint else 118.0,
                "grid_target_intervals_y": 245.0 if footprint else 118.0,
                "grid_mean_cell_arcsec_x": 0.033 if footprint else 0.068,
                "grid_mean_cell_arcsec_y": 0.033 if footprint else 0.068,
            }
            for band in ("F150W", "F277W")
        ],
    }


def test_compare_runs_reports_matched_screening_deltas() -> None:
    report = compare_runs(_payload(footprint=False), _payload(footprint=True))

    assert report["claim_status"] == "screening_only_no_depth_or_resolution_claim"
    assert report["control_mapping"] == "legacy_tanh_domain"
    assert report["treatment_mapping"] == "target_footprint_linear"
    assert report["bands"]["F150W"]["chi2_per_pixel"]["classification"] == "improved"
    assert report["bands"]["F150W"]["source_flux_ratio"]["classification"] == "improved"
    assert report["bands"]["F150W"]["fit_time_ms"]["classification"] == "informational"
    assert report["bands"]["F150W"]["grid_target_intervals_x"]["control"] == 118.0
    assert report["bands"]["F150W"]["grid_target_intervals_x"]["treatment"] == 245.0


def test_compare_runs_rejects_non_mapping_experiment_changes() -> None:
    with pytest.raises(ValueError, match="seed"):
        compare_runs(_payload(footprint=False), _payload(footprint=True, seed=102))


def test_compare_runs_allows_only_opted_in_spatial_resolution_change() -> None:
    control = _payload(footprint=False)
    treatment = _payload(footprint=True)
    treatment["effective_model_configs"]["explicit_spectral_grid_field"][
        "model_kwargs"
    ]["spatial_resolution"] = 512

    with pytest.raises(ValueError, match="model kwargs"):
        compare_runs(control, treatment)

    report = compare_runs(
        control, treatment, allow_spatial_resolution_change=True
    )

    assert report["configuration_provenance"] == {
        "allow_spatial_resolution_change": True,
        "allowed_differences": {
            "model_kwargs.spatial_resolution": {"control": 256, "treatment": 512}
        },
    }


def test_cli_option_records_allowed_spatial_resolution_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    control = _payload(footprint=False)
    treatment = _payload(footprint=True)
    treatment["effective_model_configs"]["explicit_spectral_grid_field"][
        "model_kwargs"
    ]["spatial_resolution"] = 512
    control_path = tmp_path / "control.json"
    treatment_path = tmp_path / "treatment.json"
    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"
    control_path.write_text(json.dumps(control), encoding="utf-8")
    treatment_path.write_text(json.dumps(treatment), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_smacs_grid_mapping.py",
            "--control",
            str(control_path),
            "--treatment",
            str(treatment_path),
            "--json-out",
            str(json_path),
            "--markdown-out",
            str(markdown_path),
            "--allow-spatial-resolution-change",
        ],
    )

    assert main() == 0
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["configuration_provenance"]["allowed_differences"][
        "model_kwargs.spatial_resolution"
    ] == {"control": 256, "treatment": 512}


def test_compare_runs_rejects_reversed_mapping_roles() -> None:
    with pytest.raises(ValueError, match="control must use legacy grid mapping"):
        compare_runs(_payload(footprint=True), _payload(footprint=False))
