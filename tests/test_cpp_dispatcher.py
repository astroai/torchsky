from __future__ import annotations

import importlib
import os
import subprocess
import sys
from typing import Any

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode, is_fake

from torchsky import _ops
from torchsky._native import cpp
from torchsky.wcs import WCS
from torchsky.wcs.functional import _pixel_to_world_fused, _world_to_pixel_fused
import torchsky.wcs.core as wcs_core


pytestmark = pytest.mark.skipif(cpp is None, reason="native extension unavailable")


def _parameters(direction: str) -> dict[str, Any]:
    matrix = (
        {"cd00": 1.0, "cd01": 0.0, "cd10": 0.0, "cd11": 1.0}
        if direction == "pixel_to_world"
        else {"cdi00": 1.0, "cdi01": 0.0, "cdi10": 0.0, "cdi11": 1.0}
    )
    return {
        "crpix0": 0.0,
        "crpix1": 0.0,
        **matrix,
        "proj_code": "CAR",
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
        "native_type": "center",
        "alpha0": 0.0,
        "delta0": 0.0,
    }


def _functional(
    direction: str, first: torch.Tensor, second: torch.Tensor, **overrides: Any
) -> tuple[torch.Tensor, torch.Tensor]:
    parameters = _parameters(direction)
    parameters.update(overrides)
    function = (
        _pixel_to_world_fused if direction == "pixel_to_world" else _world_to_pixel_fused
    )
    return function(first, second, **parameters)


def _operator_args(
    direction: str, first: torch.Tensor, second: torch.Tensor, **overrides: Any
) -> tuple[Any, ...]:
    parameters = _parameters(direction)
    parameters.update(overrides)
    idx = torch.empty((0, 3), dtype=torch.int64, device=first.device)
    coeff = torch.empty(0, dtype=torch.float64, device=first.device)
    matrix_names = (
        ("cd00", "cd01", "cd10", "cd11")
        if direction == "pixel_to_world"
        else ("cdi00", "cdi01", "cdi10", "cdi11")
    )
    return (
        first,
        second,
        parameters["crpix0"],
        parameters["crpix1"],
        *(parameters[name] for name in matrix_names),
        parameters["proj_code"],
        parameters["lambda_param"],
        parameters["hpx_H"],
        parameters["hpx_K"],
        parameters["east_x"],
        parameters["east_y"],
        parameters["north_x"],
        parameters["north_y"],
        parameters["north_z"],
        parameters["radial_x"],
        parameters["radial_y"],
        parameters["radial_z"],
        parameters["ra_p_rad"],
        parameters["dec_p_rad"],
        parameters["phi_p_rad"],
        parameters["native_type"],
        parameters["alpha0"],
        parameters["delta0"],
        False,
        idx,
        coeff,
        idx.clone(),
        coeff.clone(),
    )


@pytest.mark.parametrize("direction", ["pixel_to_world", "world_to_pixel"])
def test_direct_torch_ops_matches_functional_helper(direction: str) -> None:
    first = torch.tensor([-2.0, 0.5, 3.0], dtype=torch.float64)
    second = torch.tensor([1.0, -1.5, 4.0], dtype=torch.float64)
    idx = torch.empty((0, 3), dtype=torch.int64)
    coeff = torch.empty(0, dtype=torch.float64)
    operator = getattr(torch.ops.torchsky, f"wcs_{direction}")
    actual = operator(
        first,
        second,
        **_parameters(direction),
        is_tpv=False,
        tpv_idx1=idx,
        tpv_c1=coeff,
        tpv_idx2=idx.clone(),
        tpv_c2=coeff.clone(),
    )
    expected = _functional(direction, first, second)
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference, equal_nan=True)


@pytest.mark.parametrize("direction", ["pixel_to_world", "world_to_pixel"])
@pytest.mark.parametrize(
    "inputs",
    [
        (torch.empty(0, dtype=torch.float64), torch.empty(0, dtype=torch.float64)),
        (
            torch.tensor(2.0, dtype=torch.float64),
            torch.tensor(3.0, dtype=torch.float64),
        ),
        (
            torch.arange(12, dtype=torch.float64).reshape(3, 4)[:, ::2],
            torch.arange(12, 24, dtype=torch.float64).reshape(3, 4)[:, ::2],
        ),
    ],
)
def test_dispatcher_matches_functional_for_empty_scalar_and_strided(
    direction: str, inputs: tuple[torch.Tensor, torch.Tensor]
) -> None:
    first, second = inputs
    operator = getattr(torch.ops.torchsky, f"wcs_{direction}")
    actual = operator(*_operator_args(direction, first, second))
    expected = _functional(direction, first, second)
    for result, reference in zip(actual, expected, strict=True):
        assert result.shape == first.shape
        assert result.is_contiguous()
        torch.testing.assert_close(result, reference, equal_nan=True)


@pytest.mark.parametrize("direction", ["pixel_to_world", "world_to_pixel"])
def test_dispatcher_outputs_are_fresh_and_do_not_alias(direction: str) -> None:
    first = torch.tensor([1.0, 2.0], dtype=torch.float64)
    second = torch.tensor([3.0, 4.0], dtype=torch.float64)
    output1, output2 = _functional(direction, first, second)
    pointers = {
        first.data_ptr(),
        second.data_ptr(),
        output1.data_ptr(),
        output2.data_ptr(),
    }
    assert len(pointers) == 4


@pytest.mark.parametrize("direction", ["pixel_to_world", "world_to_pixel"])
@pytest.mark.parametrize(
    "first,second,overrides,match",
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
    ],
)
def test_dispatcher_rejects_invalid_inputs(
    direction: str,
    first: torch.Tensor,
    second: torch.Tensor,
    overrides: dict[str, Any],
    match: str,
) -> None:
    with pytest.raises(RuntimeError, match=match):
        _functional(direction, first, second, **overrides)


def test_dispatcher_cpu_capability_is_explicit() -> None:
    assert _ops.has_wcs_dispatcher()
    for name in ("wcs_pixel_to_world", "wcs_world_to_pixel"):
        qualified = f"torchsky::{name}"
        assert torch._C._dispatch_has_kernel_for_dispatch_key(qualified, "CPU")
        assert torch._C._dispatch_has_kernel_for_dispatch_key(qualified, "Meta")
        assert torch._C._dispatch_has_kernel_for_dispatch_key(qualified, "Autograd")


def test_dispatcher_meta_backend_preserves_shape_dtype_and_layout() -> None:
    first = torch.empty(2, dtype=torch.float64, device="meta")
    second = torch.empty(2, dtype=torch.float64, device="meta")
    outputs = _functional("pixel_to_world", first, second)
    assert all(value.device.type == "meta" for value in outputs)
    assert all(value.shape == first.shape for value in outputs)
    assert all(value.dtype == first.dtype for value in outputs)
    assert all(value.is_contiguous() for value in outputs)


@pytest.mark.parametrize("direction", ["pixel_to_world", "world_to_pixel"])
def test_functional_wrappers_support_gradcheck_and_gradgradcheck(
    direction: str,
) -> None:
    first = torch.tensor([0.1, 0.2], dtype=torch.float64, requires_grad=True)
    second = torch.tensor([0.3, 0.4], dtype=torch.float64, requires_grad=True)
    matrix = (
        {"cd00": 0.01, "cd11": 0.01}
        if direction == "pixel_to_world"
        else {"cdi00": 100.0, "cdi11": 100.0}
    )

    def function(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return _functional(direction, a, b, proj_code="ARC", **matrix)

    assert torch.autograd.gradcheck(function, (first, second), fast_mode=True)
    assert torch.autograd.gradgradcheck(function, (first, second), fast_mode=True)


@pytest.mark.parametrize("direction", ["pixel_to_world", "world_to_pixel"])
def test_dispatcher_fake_tensor_has_no_real_storage(direction: str) -> None:
    mode = FakeTensorMode()
    with mode:
        first = torch.empty_strided((2, 3), (1, 2), dtype=torch.float64)
        second = torch.empty_strided((2, 3), (1, 2), dtype=torch.float64)
        outputs = _functional(direction, first, second)
    assert all(is_fake(value) for value in outputs)
    assert all(value.shape == first.shape for value in outputs)
    assert all(value.is_contiguous() for value in outputs)


@pytest.mark.parametrize("direction", ["pixel_to_world", "world_to_pixel"])
def test_dispatcher_opcheck(direction: str) -> None:
    first = torch.tensor([0.1, 0.2], dtype=torch.float64, requires_grad=True)
    second = torch.tensor([0.3, 0.4], dtype=torch.float64, requires_grad=True)
    operator = getattr(torch.ops.torchsky, f"wcs_{direction}").default
    torch.library.opcheck(operator, _operator_args(direction, first, second))


@pytest.mark.parametrize("direction", ["pixel_to_world", "world_to_pixel"])
def test_dispatcher_tpv_tensor_vjps(direction: str) -> None:
    first = torch.tensor([0.1, 0.2], dtype=torch.float64, requires_grad=True)
    second = torch.tensor([0.3, 0.4], dtype=torch.float64, requires_grad=True)
    idx = torch.tensor([[1, 0, 0], [2, 0, 0]], dtype=torch.int64)
    c1 = torch.tensor([1.0, 1e-3], dtype=torch.float64, requires_grad=True)
    c2 = torch.tensor([1.0, -1e-3], dtype=torch.float64, requires_grad=True)
    operator = getattr(torch.ops.torchsky, f"wcs_{direction}").default

    def function(
        a: torch.Tensor,
        b: torch.Tensor,
        coeff1: torch.Tensor,
        coeff2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        args = list(_operator_args(direction, a, b))
        args[26:31] = [True, idx, coeff1, idx, coeff2]
        return operator(*args)

    inputs = (first, second, c1, c2)
    assert torch.autograd.gradcheck(function, inputs, fast_mode=True)
    assert torch.autograd.gradgradcheck(function, inputs, fast_mode=True)


@pytest.mark.parametrize("direction", ["pixel_to_world", "world_to_pixel"])
def test_dispatcher_compile_fullgraph_forward_and_backward(direction: str) -> None:
    operator = getattr(torch.ops.torchsky, f"wcs_{direction}").default

    def loss(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        outputs = operator(*_operator_args(direction, first, second))
        return outputs[0].sum() + outputs[1].sum()

    compiled = torch.compile(loss, fullgraph=True, backend="aot_eager")
    first = torch.tensor([0.1, 0.2], dtype=torch.float64, requires_grad=True)
    second = torch.tensor([0.3, 0.4], dtype=torch.float64, requires_grad=True)
    compiled(first, second).backward()
    assert first.grad is not None
    assert second.grad is not None
    assert torch.isfinite(first.grad).all()
    assert torch.isfinite(second.grad).all()


def test_dispatcher_invalid_projection_domain_returns_nan() -> None:
    first = torch.tensor([100.0], dtype=torch.float64)
    second = torch.tensor([0.0], dtype=torch.float64)
    output = _functional("pixel_to_world", first, second, proj_code="SIN")
    assert all(torch.isnan(value).all() for value in output)


def test_public_wcs_dispatcher_parity_and_gradient_bypass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    header = {
        "NAXIS": 2,
        "CTYPE1": "RA---ARC",
        "CTYPE2": "DEC--ARC",
        "CRPIX1": 8.0,
        "CRPIX2": 8.0,
        "CRVAL1": 180.0,
        "CRVAL2": 10.0,
        "CD1_1": -0.01,
        "CD1_2": 0.001,
        "CD2_1": 0.002,
        "CD2_2": 0.01,
    }
    pixels = torch.tensor([6.5, 7.3, 8.5], dtype=torch.float64)
    wcs_dispatch = WCS(header)
    world_dispatch = wcs_dispatch.pixel_to_world(pixels, pixels, origin=0)
    wcs_dispatch._disable_cpp_world_to_pixel = True
    pixel_dispatch = wcs_dispatch.world_to_pixel(*world_dispatch, origin=0)

    monkeypatch.setenv("TORCHSKY_CPP_DISPATCHER", "0")
    wcs_python = WCS(header)
    world_python = wcs_python.pixel_to_world(pixels, pixels, origin=0)
    wcs_python._disable_cpp_world_to_pixel = True
    pixel_python = wcs_python.world_to_pixel(*world_python, origin=0)
    for actual, expected in zip(world_dispatch, world_python, strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0.0)
    for actual, expected in zip(pixel_dispatch, pixel_python, strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=0.0)

    differentiable = pixels.clone().requires_grad_()
    result = wcs_dispatch.pixel_to_world(differentiable, differentiable, origin=0)
    (result[0].sum() + result[1].sum()).backward()
    assert differentiable.grad is not None
    assert torch.isfinite(differentiable.grad).all()

    python_differentiable = pixels.clone().requires_grad_()
    python_result = wcs_python.pixel_to_world(
        python_differentiable, python_differentiable, origin=0
    )
    (python_result[0].sum() + python_result[1].sum()).backward()
    assert python_differentiable.grad is not None
    torch.testing.assert_close(
        differentiable.grad, python_differentiable.grad, atol=1e-10, rtol=1e-10
    )

    dispatch_ra = world_dispatch[0].detach().requires_grad_()
    dispatch_dec = world_dispatch[1].detach().requires_grad_()
    dispatch_pixel = wcs_dispatch.world_to_pixel(dispatch_ra, dispatch_dec, origin=0)
    (dispatch_pixel[0].sum() + dispatch_pixel[1].sum()).backward()
    python_ra = world_dispatch[0].detach().requires_grad_()
    python_dec = world_dispatch[1].detach().requires_grad_()
    python_pixel = wcs_python.world_to_pixel(python_ra, python_dec, origin=0)
    (python_pixel[0].sum() + python_pixel[1].sum()).backward()
    assert dispatch_ra.grad is not None and dispatch_dec.grad is not None
    assert python_ra.grad is not None and python_dec.grad is not None
    torch.testing.assert_close(dispatch_ra.grad, python_ra.grad, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(dispatch_dec.grad, python_dec.grad, atol=1e-9, rtol=1e-9)


def test_public_wcs_does_not_swallow_dispatcher_kernel_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    header = {
        "NAXIS": 2,
        "CTYPE1": "RA---ARC",
        "CTYPE2": "DEC--ARC",
        "CRPIX1": 1.0,
        "CRPIX2": 1.0,
        "CRVAL1": 0.0,
        "CRVAL2": 0.0,
        "CD1_1": 0.01,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": 0.01,
    }
    wcs = WCS(header)

    def fail(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError("dispatcher numerical failure")

    monkeypatch.setattr(wcs_core, "_pixel_to_world_fused", fail)
    values = torch.tensor([0.0, 1.0], dtype=torch.float64)
    with pytest.raises(RuntimeError, match="dispatcher numerical failure"):
        wcs.pixel_to_world(values, values)


def test_public_wcs_dispatcher_rollback_flag_uses_python_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TORCHSKY_CPP_DISPATCHER", "0")
    header = {
        "NAXIS": 2,
        "CTYPE1": "RA---ARC",
        "CTYPE2": "DEC--ARC",
        "CRPIX1": 1.0,
        "CRPIX2": 1.0,
        "CRVAL1": 0.0,
        "CRVAL2": 0.0,
        "CD1_1": 0.01,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": 0.01,
    }

    def fail(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        raise AssertionError("dispatcher wrapper called while disabled")

    monkeypatch.setattr(wcs_core, "_pixel_to_world_fused", fail)
    values = torch.tensor([0.0, 1.0], dtype=torch.float64)
    result = WCS(header).pixel_to_world(values, values)
    assert all(torch.isfinite(value).all() for value in result)


def test_dispatcher_registration_survives_subprocess_and_reload() -> None:
    script = """
import importlib
import torch
import torchsky._ops as ops
assert ops.has_wcs_dispatcher()
importlib.reload(ops)
assert ops.has_wcs_dispatcher()
assert str(torch.ops.torchsky.wcs_pixel_to_world.default._schema).startswith(
    'torchsky::wcs_pixel_to_world('
)
assert str(torch.ops.torchsky.wcs_world_to_pixel.default._schema).startswith(
    'torchsky::wcs_world_to_pixel('
)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.fspath(os.path.dirname(os.path.dirname(__file__))),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_no_native_subprocess_reports_dispatcher_unavailable() -> None:
    script = """
from torchsky import _ops
assert not _ops.has_wcs_dispatcher()
try:
    _ops.wcs_pixel_to_world()
except RuntimeError as exc:
    assert 'unavailable' in str(exc)
else:
    raise AssertionError('missing dispatcher did not fail')
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.fspath(os.path.dirname(os.path.dirname(__file__))),
        env={**os.environ, "TORCHSKY_DISABLE_CPP": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_ops_module_can_be_reloaded_without_duplicate_schema_registration() -> None:
    assert importlib.reload(_ops).has_wcs_dispatcher()


def test_dispatcher_sources_are_compiled_and_anchored() -> None:
    package_root = os.path.dirname(os.path.dirname(wcs_core.__file__))
    cpp_root = os.path.join(package_root, "cpp_src")
    with open(os.path.join(cpp_root, "CMakeLists.txt"), encoding="utf-8") as stream:
        cmake = stream.read()
    with open(os.path.join(cpp_root, "bindings.cpp"), encoding="utf-8") as stream:
        bindings = stream.read()
    assert cmake.count("operators/wcs_schemas.cpp") == 1
    assert cmake.count("operators/wcs_cpu.cpp") == 1
    assert "ensure_wcs_operator_schemas_linked();" in bindings
    assert "ensure_wcs_cpu_operators_linked();" in bindings
