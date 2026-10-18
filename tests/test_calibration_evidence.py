from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
import torch
from torchfits import write

from torchsky.calibration_evidence import load_operator_calibration_split


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _psf_manifest(tmp_path: Path) -> dict[str, Any]:
    kernels = tmp_path / "kernels.fits"
    weights = tmp_path / "weights.fits"
    write(kernels, torch.ones((2, 3, 3)) / 9.0)
    write(weights, torch.ones((2, 3, 3)))
    return {
        "extraction": {
            "products": {
                "psf_kernels": {"path": str(kernels), "sha256": _sha256(kernels)},
                "psf_pixel_weights": {
                    "path": str(weights),
                    "sha256": _sha256(weights),
                },
            }
        },
        "evidence": [
            {
                "role": "psf",
                "split": "test",
                "detector": "NRCALONG",
                "band": "F277W",
                "detector_x": 1.0,
                "detector_y": 2.0,
                "time_mjd_utc": 60000.0,
                "sample_weight": 3.0,
                "target_index": 1,
            }
        ],
    }


def test_load_operator_calibration_split_verifies_and_selects_targets(
    tmp_path: Path,
) -> None:
    arrays = load_operator_calibration_split(
        _psf_manifest(tmp_path), split="test", role="psf", detector="NRCALONG"
    )

    assert arrays["target_kernels"].shape == (1, 3, 3)
    assert arrays["wavelength"].item() == 27760.0
    assert arrays["sample_weight"].item() == 3.0


def test_load_operator_calibration_split_rejects_invalid_split(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="split must be"):
        load_operator_calibration_split(
            _psf_manifest(tmp_path),
            split="development",
            role="psf",
            detector="NRCALONG",
        )


def test_load_operator_calibration_split_rejects_product_hash_mismatch(
    tmp_path: Path,
) -> None:
    manifest = _psf_manifest(tmp_path)
    products = manifest["extraction"]["products"]
    products["psf_kernels"]["sha256"] = "0" * 64

    with pytest.raises(ValueError, match="SHA-256"):
        load_operator_calibration_split(
            manifest, split="test", role="psf", detector="NRCALONG"
        )
