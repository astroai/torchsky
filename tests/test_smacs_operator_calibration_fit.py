from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import torch

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
    assert all(
        min(item["sample_counts"].values()) >= 2 for item in included
    )
