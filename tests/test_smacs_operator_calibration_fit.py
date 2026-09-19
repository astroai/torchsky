from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
from types import ModuleType

import torch
import pytest
import torchsky

from torchsky.calibration_evidence import load_operator_calibration_split

from torchsky.map.operators import ConditionedBackgroundField, ConditionedKernelPSF


def _benchmark_module() -> ModuleType:
    path = (
        Path(__file__).parents[1]
        / "benchmarks"
        / "fit_smacs0723_operator_calibration.py"
    )
    spec = importlib.util.spec_from_file_location("smacs_operator_fit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import benchmark module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exposure_selection_preserves_independent_splits_and_provenance(
    tmp_path: Path,
) -> None:
    benchmark = _benchmark_module()
    rows = []
    for exposure, band, value in (
        ("visit/a_cal.fits", "F277W", 2.0),
        ("visit/b_cal.fits", "F277W", 99.0),
        ("visit/a_cal.fits", "F444W", 42.0),
    ):
        for split in ("train", "validation", "test"):
            for role in ("background", "psf"):
                row = {
                    "exposure": exposure,
                    "band": band,
                    "detector": "NRCALONG",
                    "role": role,
                    "split": split,
                    "calibration_split": split,
                    "detector_x": 500.0,
                    "detector_y": 600.0,
                    "time_mjd_utc": 60000.0,
                    "target_value": value,
                }
                if role == "psf":
                    row.update(
                        target_index=0,
                        target_kind="normalized_effective_psf",
                        pixel_response_included=True,
                        pixel_response_provenance="registered detector SCI stamp",
                    )
                rows.append(row)
    kernels = tmp_path / "kernels.fits"
    weights = tmp_path / "weights.fits"
    torchsky.write(str(kernels), torch.ones((1, 3, 3)) / 9.0)
    torchsky.write(str(weights), torch.ones((1, 3, 3)))
    payload = {
        "extraction": {
            "products": {
                "psf_kernels": {
                    "path": str(kernels),
                    "sha256": hashlib.sha256(kernels.read_bytes()).hexdigest(),
                },
                "psf_pixel_weights": {
                    "path": str(weights),
                    "sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                },
            }
        },
        "evidence": rows,
    }
    selections = benchmark._calibration_groups(payload, "exposure")
    assert selections == sorted(selections)
    assert len(selections) == 3
    arrays = load_operator_calibration_split(
        payload,
        split="train",
        role="background",
        detector="NRCALONG",
        exposure_id="visit/a_cal.fits",
        band="F277W",
    )
    assert arrays["target_values"].tolist() == [2.0]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload))
    group = benchmark._fit_group(
        manifest,
        detector="NRCALONG",
        role="background",
        family="stationary",
        steps=1,
        lr=1e-3,
        device=torch.device("cpu"),
        output_dir=tmp_path,
        seed=1,
        manifest_payload=payload,
        minimum_slice_samples=1,
        exposure_id="visit/a_cal.fits",
        band="F277W",
    )
    checkpoint = torch.load(group["checkpoint"], weights_only=True)
    assert checkpoint["exposure_id"] == group["exposure_id"] == "visit/a_cal.fits"
    assert checkpoint["band"] == group["band"] == "F277W"
    assert checkpoint["calibration_scope"] == "exposure"
    saved_group = json.loads(Path(group["checkpoint"]).with_suffix(".json").read_text())
    assert saved_group["exposure_id"] == "visit/a_cal.fits"
    assert saved_group["elapsed_seconds"] > 0
    assert saved_group["device"] == "cpu"
    assert saved_group["validation_report"] == group["validation_report"]
    psf_group = benchmark._fit_group(
        manifest,
        detector="NRCALONG",
        role="psf",
        family="stationary",
        steps=1,
        lr=1e-3,
        device=torch.device("cpu"),
        output_dir=tmp_path,
        seed=1,
        manifest_payload=payload,
        minimum_slice_samples=1,
        exposure_id="visit/a_cal.fits",
        band="F277W",
    )
    psf_checkpoint = torch.load(psf_group["checkpoint"], weights_only=True)
    assert psf_checkpoint["schema_version"] == 2
    assert psf_checkpoint["pixel_response_included"] is True
    assert psf_checkpoint["pixel_response_provenance"] == "registered detector SCI stamp"
    assert psf_group["pixel_response_included"] is True
    assert psf_group["pixel_response_provenance"] == "registered detector SCI stamp"
    invalid_payload = json.loads(json.dumps(payload))
    for row in invalid_payload["evidence"]:
        if row["role"] == "psf":
            row["pixel_response_included"] = 1
    invalid_manifest = tmp_path / "invalid-manifest.json"
    invalid_manifest.write_text(json.dumps(invalid_payload))
    with pytest.raises(ValueError, match="bool"):
        benchmark._fit_group(
            invalid_manifest,
            detector="NRCALONG",
            role="psf",
            family="stationary",
            steps=1,
            lr=1e-3,
            device=torch.device("cpu"),
            output_dir=tmp_path,
            seed=1,
            manifest_payload=invalid_payload,
            minimum_slice_samples=1,
            exposure_id="visit/a_cal.fits",
            band="F277W",
        )
    with pytest.raises(ValueError, match="missing independent calibration evidence"):
        benchmark._calibration_groups(
            {"evidence": [r for r in rows if r["split"] == "train"]}, "exposure"
        )
    with pytest.raises(ValueError, match="exposure_id or exposure provenance"):
        benchmark._calibration_groups(
            {"evidence": [{"detector": "NRCALONG", "band": "F277W"}]}, "exposure"
        )


def test_comparisons_do_not_pool_exposures_or_bands() -> None:
    benchmark = _benchmark_module()
    groups = [
        {
            "detector": "NRCALONG",
            "exposure_id": exposure,
            "band": band,
            "role": "psf",
            "family": family,
            "validation_report": {"terms": [{"raw_loss": loss}]},
            "test_report": {"terms": [{"raw_loss": loss}]},
        }
        for exposure, band in (("a", "F277W"), ("b", "F277W"), ("a", "F444W"))
        for family, loss in (("stationary", 2.0), ("adaptive", 1.0))
    ]
    comparisons = benchmark._held_out_comparisons(groups)
    assert len(comparisons) == 3
    assert {(row["exposure_id"], row["band"]) for row in comparisons} == {
        ("a", "F277W"),
        ("b", "F277W"),
        ("a", "F444W"),
    }


def test_stationary_only_cli_never_constructs_adaptive_models(tmp_path, monkeypatch) -> None:
    import json
    import sys

    benchmark = _benchmark_module()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"evidence": [{"detector": "NRCB1", "split": "train"}]}))
    calls = []

    def fit_group(_manifest, **kwargs):
        assert kwargs["family"] == "stationary"
        group = {key: kwargs[key] for key in ("detector", "role", "family")}
        calls.append(group)
        return group

    monkeypatch.setattr(benchmark, "_fit_group", fit_group)
    monkeypatch.setattr(sys, "argv", ["fit", "--manifest", str(manifest),
                                    "--output-dir", str(tmp_path / "out"),
                                    "--family", "stationary"])
    assert benchmark.main() == 0
    report = json.loads((tmp_path / "out/operator_calibration_fit.json").read_text())
    assert report["families"] == ["stationary"]
    assert {group["role"] for group in calls} == {"background", "psf"}
    assert report["held_out_comparisons"] == []


def test_stationary_controls_are_coordinate_invariant_and_compare_held_out() -> None:
    benchmark = _benchmark_module()
    arrays = {
        "detector_x": torch.tensor([0.0, 4.0, 9.0]),
        "detector_y": torch.tensor([1.0, 5.0, 8.0]),
        "wavelength": torch.tensor([9000.0, 15000.0, 20000.0]),
        "time_mjd": torch.tensor([60000.0, 60001.0, 60003.0]),
        "sample_weight": torch.ones(3),
        "target_values": torch.tensor([2.0, 2.5, 1.5]),
        "target_kernels": torch.ones((3, 5, 5)) / 25.0,
    }
    background, _ = benchmark._make_model(
        arrays,
        role="background",
        family="stationary",
        device=torch.device("cpu"),
    )
    psf, _ = benchmark._make_model(
        arrays,
        role="psf",
        family="stationary",
        device=torch.device("cpu"),
    )
    assert isinstance(background, ConditionedBackgroundField)
    assert isinstance(psf, ConditionedKernelPSF)
    predicted_background = background(
        arrays["detector_x"],
        arrays["detector_y"],
        arrays["wavelength"],
        arrays["time_mjd"],
    )
    assert torch.equal(predicted_background, predicted_background[:1].expand(3))
    assert psf.kernel_basis.n_basis == 1

    def group(family: str, validation: float, test: float) -> dict[str, object]:
        return {
            "detector": "NRCALONG",
            "role": "psf",
            "family": family,
            "validation_report": {"terms": [{"raw_loss": validation}]},
            "test_report": {"terms": [{"raw_loss": test}]},
        }

    comparisons = benchmark._held_out_comparisons(
        [group("stationary", 2.0, 4.0), group("adaptive", 1.0, 3.0)]
    )
    assert comparisons[0]["validation"]["adaptive_improvement_fraction"] == 0.5
    assert comparisons[0]["test"]["adaptive_to_stationary_ratio"] == 0.75

    coordinates = {
        "detector_x": torch.tensor([0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]),
        "detector_y": torch.tensor([0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0]),
        "time_mjd": torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]),
        "sample_weight": torch.ones(8),
    }
    split_arrays = {
        split: {name: value.clone() for name, value in coordinates.items()}
        for split in ("train", "validation", "test")
    }
    split_rows = {
        split: [{"band": "F090W" if index % 2 == 0 else "F150W"} for index in range(8)]
        for split in ("train", "validation", "test")
    }
    included, skipped = benchmark._slice_definitions(
        split_arrays, split_rows, minimum_samples=2
    )
    assert len(included) == 8
    assert skipped == []
    assert all(min(item["sample_counts"].values()) >= 2 for item in included)
