from __future__ import annotations

import importlib
import json
from pathlib import Path

NAMESPACES = (
    "torchsky",
    "torchsky.wcs",
    "torchsky.sphere",
    "torchsky.map",
    "torchsky.scene",
    "torchsky.field",
    "torchsky.catalogs",
    "torchsky.coords",
    "torchsky.encoders",
    "torchsky.filter_profiles",
    "torchsky.map.likelihood",
    "torchsky.map.query",
    "torchsky.map.evidence",
    "torchsky.map.operators.nuisance",
    "torchsky.map.operators.continuous_response",
    "torchsky.map.operators.detector_response",
    "torchsky.map.operators.pixel_calibration",
    "torchsky.map.operators.epsf",
    "torchsky.map.field.native_response",
    "torchsky.map.field.native_integrated_read",
    "torchsky.map.field.gaussian",
    "torchsky.map.field.joint_native_likelihood",
)


def test_public_api_manifest_is_current() -> None:
    manifest_path = Path(__file__).resolve().parents[1] / "docs" / "api-manifest.json"
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    for namespace in NAMESPACES:
        module = importlib.import_module(namespace)
        assert sorted(module.__all__) == expected[namespace], (
            f"{namespace} public API changed; review it, then run "
            "`pixi run api-manifest`"
        )
