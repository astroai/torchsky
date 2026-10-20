from __future__ import annotations

import subprocess
import sys

import pytest
import torch

from torchsky._native import cpp


pytestmark = pytest.mark.skipif(cpp is None, reason="native extension unavailable")


def _call_fused(
    direction: str,
    input1: torch.Tensor | None = None,
    input2: torch.Tensor | None = None,
    output1: torch.Tensor | None = None,
    output2: torch.Tensor | None = None,
    **overrides: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    input1 = torch.tensor([1.0, 2.0], dtype=torch.float64) if input1 is None else input1
    input2 = torch.tensor([3.0, 4.0], dtype=torch.float64) if input2 is None else input2
    output1 = torch.empty_like(input1) if output1 is None else output1
    output2 = torch.empty_like(input1) if output2 is None else output2
    tensors = {
        "pixel_to_world": ("x_tensor", "y_tensor", "ra_tensor", "dec_tensor"),
        "world_to_pixel": ("ra_tensor", "dec_tensor", "x_tensor", "y_tensor"),
    }
    names = tensors[direction]
    kwargs: dict[str, object] = {
        names[0]: input1,
        names[1]: input2,
        names[2]: output1,
        names[3]: output2,
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
    function = getattr(cpp, f"wcs_{direction}_fused_cpu")
    function(**kwargs)
    return output1, output2


@pytest.mark.parametrize("direction", ["pixel_to_world", "world_to_pixel"])
def test_fused_wcs_default_tpv_arguments_mutate_outputs(direction: str) -> None:
    output1, output2 = _call_fused(direction)
    torch.testing.assert_close(output1, torch.tensor([1.0, 2.0], dtype=torch.float64))
    torch.testing.assert_close(output2, torch.tensor([3.0, 4.0], dtype=torch.float64))


@pytest.mark.parametrize("direction", ["pixel_to_world", "world_to_pixel"])
def test_fused_wcs_valid_empty_inputs_and_tpv(direction: str) -> None:
    empty = torch.empty((0, 2), dtype=torch.float64)
    output1, output2 = _call_fused(direction, empty, empty.clone())
    assert output1.shape == (0, 2)
    assert output2.shape == (0, 2)

    idx = torch.empty((0, 3), dtype=torch.int64)
    coeff = torch.empty((0,), dtype=torch.float64)
    output1, output2 = _call_fused(
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
    "mutation,match",
    [
        ({"input2": torch.ones(3, dtype=torch.float64)}, "identical shapes"),
        ({"output1": torch.empty(1, dtype=torch.float64)}, "identical shapes"),
        ({"input1": torch.ones(2, dtype=torch.float32)}, "float64"),
        ({"input1": torch.ones((2, 2), dtype=torch.float64).t()}, "contiguous"),
        ({"proj_code": "BAD"}, "unsupported fused WCS projection"),
        ({"native_type": "bad"}, "unsupported fused WCS native_type"),
        ({"alpha0": float("nan")}, "alpha0 must be finite"),
    ],
)
def test_fused_wcs_rejects_invalid_raw_buffer_contracts(
    direction: str, mutation: dict[str, object], match: str
) -> None:
    tensor_keys = {"input1", "input2", "output1", "output2"}
    tensor_args = {key: value for key, value in mutation.items() if key in tensor_keys}
    overrides = {key: value for key, value in mutation.items() if key not in tensor_keys}
    with pytest.raises(RuntimeError, match=match):
        _call_fused(direction, **tensor_args, **overrides)


@pytest.mark.parametrize("direction", ["pixel_to_world", "world_to_pixel"])
def test_fused_wcs_rejects_output_overlap(direction: str) -> None:
    input1 = torch.tensor([1.0, 2.0], dtype=torch.float64)
    input2 = torch.tensor([3.0, 4.0], dtype=torch.float64)
    with pytest.raises(RuntimeError, match="must not overlap"):
        _call_fused(direction, input1, input2, input1, torch.empty_like(input1))
    output = torch.empty_like(input1)
    with pytest.raises(RuntimeError, match="must not overlap"):
        _call_fused(direction, input1, input2, output, output)


@pytest.mark.parametrize("direction", ["pixel_to_world", "world_to_pixel"])
@pytest.mark.parametrize(
    "tpv_args,match",
    [
        ({}, "TPV index and coefficient tensors are required"),
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
                "tpv_idx1": torch.zeros((3, 2), dtype=torch.int64).t(),
                "tpv_c1": torch.ones(2, dtype=torch.float64),
            },
            "contiguous",
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
def test_fused_wcs_rejects_malformed_tpv(
    direction: str, tpv_args: dict[str, object], match: str
) -> None:
    idx = torch.empty((0, 3), dtype=torch.int64)
    coeff = torch.empty(0, dtype=torch.float64)
    arguments: dict[str, object] = {
        "is_tpv": True,
        "tpv_idx1": idx,
        "tpv_c1": coeff,
        "tpv_idx2": idx.clone(),
        "tpv_c2": coeff.clone(),
    }
    if tpv_args:
        arguments.update(tpv_args)
    else:
        arguments = {"is_tpv": True}
    with pytest.raises(RuntimeError, match=match):
        _call_fused(direction, **arguments)


def test_fused_wcs_dangerous_malformed_calls_raise_in_subprocess() -> None:
    code = """
import torch
from torchsky._native import cpp

common = dict(
    crpix0=0.0, crpix1=0.0, cd00=1.0, cd01=0.0, cd10=0.0, cd11=1.0,
    proj_code="CAR", lambda_param=1.0, hpx_H=4.0, hpx_K=3.0,
    east_x=0.0, east_y=1.0, north_x=0.0, north_y=0.0, north_z=1.0,
    radial_x=1.0, radial_y=0.0, radial_z=0.0, ra_p_rad=0.0, dec_p_rad=0.0,
    phi_p_rad=0.0, native_type="center", alpha0=0.0, delta0=0.0,
)
x = torch.ones(2, dtype=torch.float64)
y = torch.ones(2, dtype=torch.float64)
idx = torch.tensor([[8, 0, 0]], dtype=torch.int64)
coeff = torch.ones(1, dtype=torch.float64)
cases = [
    dict(x_tensor=x, y_tensor=y, ra_tensor=torch.empty(1, dtype=torch.float64), dec_tensor=torch.empty_like(y), is_tpv=False),
    dict(x_tensor=x, y_tensor=y, ra_tensor=torch.empty_like(x), dec_tensor=torch.empty_like(y), is_tpv=True, tpv_idx1=idx, tpv_c1=coeff, tpv_idx2=idx, tpv_c2=coeff),
]
for case in cases:
    try:
        cpp.wcs_pixel_to_world_fused_cpu(**case, **common)
    except RuntimeError:
        pass
    else:
        raise AssertionError("malformed fused WCS call did not raise")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
