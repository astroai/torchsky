from __future__ import annotations

from pathlib import Path

import pytest
import torch

import torchsky
from torchsky.map.plan import MapPlanV2, build_map_input_identity
import torchfits


def _header() -> dict[str, object]:
    return {
        "NAXIS": 2,
        "NAXIS1": 4,
        "NAXIS2": 4,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 2.5,
        "CRPIX2": 2.5,
        "CRVAL1": 10.0,
        "CRVAL2": 20.0,
        "CDELT1": -0.001,
        "CDELT2": 0.001,
    }


def _plan(identity: dict[str, object]) -> MapPlanV2:
    return MapPlanV2(
        {
            "schema_version": 2,
            "method": "coadd",
            "output": "/tmp/out.fits",
            "inputs": [{"identity": identity}],
            "target": {"shape": [4, 4], "header": _header()},
            "execution": {
                "workflow": "native",
                "kernel": "bilinear",
                "stack": "weighted_mean",
                "background_mode": "none",
                "drizzle_pixfrac": 1.0,
                "clip_sigma": 4.0,
                "clip_iterations": 3,
                "value_semantics": "surface_brightness",
            },
            "requested_products": ["SCI", "WHT"],
            "software": {"torchsky": torchsky.__version__},
            "runtime": {"dtype": "float32", "device": "cpu", "seeds": {}},
        }
    )


def test_map_plan_identity_rejects_changed_input(tmp_path: Path) -> None:
    path = tmp_path / "input.fits"
    torchfits.write(str(path), torch.zeros((4, 4)), header=_header(), overwrite=True)
    identity = build_map_input_identity(path, hdu=0)
    plan = _plan(identity)
    plan.verify_inputs()

    torchfits.write(str(path), torch.ones((4, 4)), header=_header(), overwrite=True)

    with pytest.raises(ValueError, match="content changed|size changed|checksum"):
        plan.verify_inputs()


def test_map_plan_v2_rejects_unidentified_input() -> None:
    with pytest.raises(ValueError, match="requires an identity"):
        MapPlanV2(
            {
                "schema_version": 2,
                "method": "coadd",
                "inputs": [{"path": "input.fits"}],
                "target": {"header": _header()},
                "execution": {},
                "requested_products": ["SCI"],
                "software": {},
                "runtime": {},
            }
        )


def test_map_plan_v2_rejects_non_boolean_inverse_variance_marker(
    tmp_path: Path,
) -> None:
    path = tmp_path / "input.fits"
    torchfits.write(str(path), torch.zeros((4, 4)), header=_header(), overwrite=True)
    identity = build_map_input_identity(path, hdu=0)
    payload = _plan(identity).to_dict()
    payload["inputs"][0]["weight_is_inverse_variance"] = "false"

    with pytest.raises(ValueError, match="must be boolean"):
        MapPlanV2(payload)


@pytest.mark.parametrize(
    "shape",
    ([4], [4, 0], [4, -1], [4, 1.5], [4, True]),
)
def test_map_plan_v2_rejects_malformed_target_shape(shape) -> None:
    payload = _plan(
        {
            "path": "/tmp/input.fits",
            "size": 1,
            "hdu": 0,
            "sha256": "0" * 64,
        }
    ).to_dict()
    payload["target"]["shape"] = shape

    with pytest.raises(ValueError, match="target shape"):
        MapPlanV2(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("workflow", "unknown"),
        ("kernel", "unknown"),
        ("stack", "unknown"),
        ("background_mode", "unknown"),
        ("value_semantics", "unknown"),
    ),
)
def test_map_plan_v2_rejects_unknown_execution_modes(field, value) -> None:
    payload = _plan(
        {
            "path": "/tmp/input.fits",
            "size": 1,
            "hdu": 0,
            "sha256": "0" * 64,
        }
    ).to_dict()
    payload["execution"][field] = value

    with pytest.raises(ValueError, match=field):
        MapPlanV2(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("weight", 0.0),
        ("photometric_scale", float("inf")),
        ("background_offset", float("nan")),
    ),
)
def test_map_plan_v2_rejects_invalid_input_numbers(field, value) -> None:
    payload = _plan(
        {
            "path": "/tmp/input.fits",
            "size": 1,
            "hdu": 0,
            "sha256": "0" * 64,
        }
    ).to_dict()
    payload["inputs"][0][field] = value

    with pytest.raises(ValueError):
        MapPlanV2(payload)


def test_map_plan_v2_rejects_invalid_runtime_and_product_types() -> None:
    payload = _plan(
        {
            "path": "/tmp/input.fits",
            "size": 1,
            "hdu": 0,
            "sha256": "0" * 64,
        }
    ).to_dict()
    payload["runtime"]["device"] = "not-a-device"
    with pytest.raises(ValueError, match="runtime device"):
        MapPlanV2(payload)

    payload = _plan(
        {
            "path": "/tmp/input.fits",
            "size": 1,
            "hdu": 0,
            "sha256": "0" * 64,
        }
    ).to_dict()
    payload["requested_products"] = ["SCI", "NOT_A_PLANE"]
    with pytest.raises(ValueError, match="requested_products"):
        MapPlanV2(payload)


def test_map_plan_v2_rejects_unverified_optional_input_path() -> None:
    payload = _plan(
        {
            "path": "/tmp/input.fits",
            "size": 1,
            "hdu": 0,
            "sha256": "0" * 64,
        }
    ).to_dict()
    payload["inputs"][0]["weight_map"] = "/tmp/weights.fits"

    with pytest.raises(ValueError, match="weight_identity"):
        MapPlanV2(payload)


def test_map_plan_v2_rejects_non_object_root() -> None:
    with pytest.raises(ValueError, match="finite JSON object"):
        MapPlanV2(["not", "a", "plan"])
