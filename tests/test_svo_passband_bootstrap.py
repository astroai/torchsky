from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from bench_map_smacs0723_real import _field_observation_kwargs_for_band
from bootstrap_svo_passbands import bootstrap_svo_passbands


def _fake_fetcher(*, band_name: str, filter_id: str, **kwargs):
    band_number = float(band_name[1:4])
    return {
        "svo_filter_id": filter_id,
        "transmission_wavelength": torch.tensor(
            [band_number * 100.0, band_number * 110.0, band_number * 120.0]
        ),
        "transmission_throughput": torch.tensor([0.0, 1.0, 0.25]),
    }


def test_bootstrap_writes_deterministic_smacs_passbands_and_manifest(tmp_path: Path) -> None:
    manifest = bootstrap_svo_passbands(
        tmp_path,
        bands=("F150W", "F277W"),
        fetcher=_fake_fetcher,
    )

    assert manifest["schema_version"] == 1
    assert manifest["source"] == "SVO Filter Profile Service"
    assert [entry["band"] for entry in manifest["passbands"]] == ["F150W", "F277W"]
    for entry in manifest["passbands"]:
        path = tmp_path / entry["filename"]
        assert path.is_file()
        assert entry["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as payload:
            assert payload["filter_id"].item() == f"JWST/NIRCam2025.{entry['band']}"
            assert payload["wavelength_angstrom"].dtype == np.float64
            assert payload["transmission"].dtype == np.float64
        kwargs = _field_observation_kwargs_for_band(
            entry["band"], measured_passband_dir=tmp_path
        )
        assert kwargs["transmission_wavelength"] is not None
    first_bytes = {
        path.name: path.read_bytes()
        for path in tmp_path.glob("JWST_NIRCam2025.*.npz")
    }

    def offline_fetcher(**kwargs):
        raise AssertionError("verified artifacts must not contact SVO")

    repeated = bootstrap_svo_passbands(
        tmp_path,
        bands=("F150W", "F277W"),
        fetcher=offline_fetcher,
    )

    assert repeated == manifest
    assert json.loads((tmp_path / "manifest.json").read_text()) == manifest
    assert first_bytes == {
        path.name: path.read_bytes()
        for path in tmp_path.glob("JWST_NIRCam2025.*.npz")
    }


def test_bootstrap_rejects_conflicting_existing_passband(tmp_path: Path) -> None:
    bootstrap_svo_passbands(tmp_path, bands=("F150W",), fetcher=_fake_fetcher)
    path = tmp_path / "JWST_NIRCam2025.F150W.npz"
    path.write_bytes(b"conflict")

    with pytest.raises(ValueError, match="hash mismatch"):
        bootstrap_svo_passbands(tmp_path, bands=("F150W",), fetcher=_fake_fetcher)
