from __future__ import annotations

import subprocess
import sys
from typing import Any

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode, is_fake

from torchsky import _ops


pytestmark = pytest.mark.skipif(
    not _ops.has_wcs_dispatcher(), reason="native extension unavailable"
)


def _call_operator(
    direction: str,
    input1: torch.Tensor | None = None,
    input2: torch.Tensor | None = None,
    **overrides: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    input1 = torch.tensor([1.0, 2.0], dtype=torch.float64) if input1 is None else input1
    input2 = torch.tensor([3.0, 4.0], dtype=torch.float64) if input2 is None else input2
    idx = torch.empty((0, 3), dtype=torch.int64, device=input1.device)
    coeff = torch.empty(0, dtype=torch.float64, device=input1.device)
    kwargs: dict[str, Any] = {
        "crpix0": 0.0,
        "crpix1": 0.0,
        "lambda_param": 1.0,
        "hpx_H": 4.0,
        "hpx_K": 3.0,
        "east_x": 0.0,
        "east_y": 1.0,
        "north_x": 0.0,
        "north_y": 0.0,
        "north_z": 1.0,
        "radial_x": 1.0,
        "radial_y": 0.0,
        "radial_z": 0.0,
        "ra_p_rad": 0.0,
        "dec_p_rad": 0.0,
        "phi_p_rad": 0.0,
        "proj_code": "CAR",
        "native_type": "center",
        "alpha0": 0.0,
        "delta0": 0.0,
        "is_tpv": False,
        "tpv_idx1": idx,
        "tpv_c1": coeff,
        "tpv_idx2": idx.clone(),
        "tpv_c2": coeff.clone(),
    }
    matrix_prefix = "cd" if direction == "pixel_to_world" else "cdi"
    kwargs.update(
        {
            f"{matrix_prefix}00": 1.0,
            f"{matrix_prefix}01": 0.0,
            f"{matrix_prefix}10": 0.0,
            f"{matrix_prefix}11": 1.0,
        }
    )
    kwargs.update(overrides)
    return getattr(torch.ops.torchsky, f"wcs_{direction}")(input1, input2, **kwargs)


@pytest.mark.parametrize("direction", ["pixel_to_world", "world_to_pixel"])
def test_wcs_operator_default_arguments_return_functional_outputs(direction: str) -> None:
    output1, output2 = _call_operator(direction)
    torch.testing.assert_close(output1, torch.tensor([1.0, 2.0], dtype=torch.float64))
    torch.testing.assert_close(output2, torch.tensor([3.0, 4.0], dtype=torch.float64))


@pytest.mark.parametrize("direction", ["pixel_to_world", "world_to_pixel"])
def test_wcs_operator_accepts_empty_inputs_and_tpv(direction: str) -> None:
    empty = torch.empty((0, 2), dtype=torch.float64)
    output1, output2 = _call_operator(direction, empty, empty.clone())
    assert output1.shape == (0, 2)
    assert output2.shape == (0, 2)

    idx = torch.empty((0, 3), dtype=torch.int64)
    coeff = torch.empty(0, dtype=torch.float64)
    output1, output2 = _call_operator(
        direction,
        is_tpv=True,
        tpv_idx1=idx,
        tpv_c1=coeff,
        tpv_idx2=idx.clone(),
        tpv_c2=coeff.clone(),
    )
    torch.testing.assert_close(output1, torch.tensor([1.0, 2.0], dtype=torch.float64))
    torch.testing.assert_close(output2, torch.tensor([3.0, 4.0], dtype=torch.float64))


@pytest.mark.parametrize("direction", ["pixel_to_world", "world_to_pixel"])
@pytest.mark.parametrize(
    "input1,input2,overrides,match",
    [
        (
            torch.ones(2, dtype=torch.float64),
            torch.ones(3, dtype=torch.float64),
            {},
            "identical shapes",
        ),
        (
            torch.ones(2, dtype=torch.float32),
            torch.ones(2, dtype=torch.float32),
            {},
            "float64",
        ),
        (
            torch.ones(2, dtype=torch.float64),
            torch.ones(2, dtype=torch.float64),
            {"proj_code": "BAD"},
            "unsupported fused WCS projection",
        ),
        (
            torch.ones(2, dtype=torch.float64),
            torch.ones(2, dtype=torch.float64),
            {"native_type": "bad"},
            "unsupported fused WCS native_type",
        ),
        (
            torch.ones(2, dtype=torch.float64),
            torch.ones(2, dtype=torch.float64),
            {"alpha0": float("nan")},
            "alpha0 must be finite",
        ),
    ],
)
def test_wcs_operator_rejects_malformed_inputs_and_parameters(
    direction: str,
    input1: torch.Tensor,
    input2: torch.Tensor,
    overrides: dict[str, Any],
    match: str,
) -> None:
    with pytest.raises(RuntimeError, match=match):
        _call_operator(direction, input1, input2, **overrides)


@pytest.mark.parametrize("direction", ["pixel_to_world", "world_to_pixel"])
@pytest.mark.parametrize(
    "tpv_args,match",
    [
        (
            {
                "tpv_idx1": torch.empty((0, 2), dtype=torch.int64),
                "tpv_c1": torch.empty(0, dtype=torch.float64),
            },
            "shape \\[N, 3\\]",
        ),
        (
            {
                "tpv_idx1": torch.zeros((1, 3), dtype=torch.int32),
                "tpv_c1": torch.ones(1, dtype=torch.float64),
            },
            "int64",
        ),
        (
            {
                "tpv_idx1": torch.zeros((2, 3), dtype=torch.int64),
                "tpv_c1": torch.ones(1, dtype=torch.float64),
            },
            "lengths must match",
        ),
        (
            {
                "tpv_idx1": torch.tensor([[8, 0, 0]], dtype=torch.int64),
                "tpv_c1": torch.ones(1, dtype=torch.float64),
            },
            "exponents must be in \\[0, 7\\]",
        ),
        (
            {
                "tpv_idx1": torch.zeros((1, 3), dtype=torch.int64),
                "tpv_c1": torch.tensor([float("nan")], dtype=torch.float64),
            },
            "must be finite",
        ),
    ],
)
def test_wcs_operator_rejects_malformed_tpv(
    direction: str, tpv_args: dict[str, Any], match: str
) -> None:
    idx = torch.empty((0, 3), dtype=torch.int64)
    coeff = torch.empty(0, dtype=torch.float64)
    arguments: dict[str, Any] = {
        "is_tpv": True,
        "tpv_idx1": idx,
        "tpv_c1": coeff,
        "tpv_idx2": idx.clone(),
        "tpv_c2": coeff.clone(),
    }
    arguments.update(tpv_args)
    with pytest.raises(RuntimeError, match=match):
        _call_operator(direction, **arguments)


def test_wcs_operator_schema_has_functional_outputs() -> None:
    for name in ("wcs_pixel_to_world", "wcs_world_to_pixel"):
        schema = str(getattr(torch.ops.torchsky, name).default._schema)
        assert schema.startswith(f"torchsky::{name}(Tensor")
        assert ") -> (Tensor " in schema
        assert schema.count("Tensor ") >= 8
        assert "out" not in schema


@pytest.mark.parametrize("direction", ["pixel_to_world", "world_to_pixel"])
def test_wcs_operator_fake_tensor_outputs_are_safe(direction: str) -> None:
    mode = FakeTensorMode()
    with mode:
        first = torch.empty_strided((2, 3), (1, 2), dtype=torch.float64)
        second = torch.empty_strided((2, 3), (1, 2), dtype=torch.float64)
        outputs = _call_operator(direction, first, second)
    assert all(is_fake(output) for output in outputs)
    assert all(output.shape == first.shape for output in outputs)
    assert all(output.is_contiguous() for output in outputs)


def test_wcs_operator_dangerous_malformed_calls_raise_in_subprocess() -> None:
    code = """
import torch
import torchsky._ops as ops

assert ops.has_wcs_dispatcher()
common = dict(
    crpix0=0.0, crpix1=0.0, cd00=1.0, cd01=0.0, cd10=0.0, cd11=1.0,
    proj_code="CAR", lambda_param=1.0, hpx_H=4.0, hpx_K=3.0,
    east_x=0.0, east_y=1.0, north_x=0.0, north_y=0.0, north_z=1.0,
    radial_x=1.0, radial_y=0.0, radial_z=0.0, ra_p_rad=0.0, dec_p_rad=0.0,
    phi_p_rad=0.0, native_type="center", alpha0=0.0, delta0=0.0,
)
x = torch.ones(2, dtype=torch.float64)
y = torch.ones(2, dtype=torch.float64)
empty_idx = torch.empty((0, 3), dtype=torch.int64)
empty_coeff = torch.empty(0, dtype=torch.float64)
bad_idx = torch.tensor([[8, 0, 0]], dtype=torch.int64)
cases = [
    dict(first=x, second=torch.ones(1, dtype=torch.float64), is_tpv=False, tpv_idx1=empty_idx, tpv_c1=empty_coeff, tpv_idx2=empty_idx, tpv_c2=empty_coeff),
    dict(first=x, second=y, is_tpv=True, tpv_idx1=bad_idx, tpv_c1=torch.ones(1, dtype=torch.float64), tpv_idx2=bad_idx, tpv_c2=torch.ones(1, dtype=torch.float64)),
]
for case in cases:
    first = case.pop("first")
    second = case.pop("second")
    try:
        torch.ops.torchsky.wcs_pixel_to_world(first, second, **common, **case)
    except RuntimeError:
        pass
    else:
        raise AssertionError("malformed WCS operator call did not raise")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
