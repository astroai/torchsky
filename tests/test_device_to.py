"""Device+dtype conversions go through ``torchsky.device._to``.

PyTorch 2.14 on MPS fills the result with zeros when one copy both leaves MPS
and casts to float64/complex128 (``mps_f32.to("cpu", torch.float64)``,
``torch.as_tensor(mps_f32, device="cpu", dtype=torch.float64)``,
``cpu_f64.copy_(mps_f32)``). ``_to`` moves first and casts on the target device.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch
from torch import nn

from torchsky.device import _as_tensor, _to

requires_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS not available"
)

SRC = Path(__file__).resolve().parents[1] / "src" / "torchsky"

# "relative/path.py: source of the call" -> why the source can never be an MPS
# tensor that is cast to float64/complex128 off MPS. Keep empty when possible.
ALLOWLIST: dict[str, str] = {}

_LITERAL_CALLS = {
    "abs",
    "bool",
    "float",
    "int",
    "len",
    "list",
    "max",
    "min",
    "range",
    "round",
    "sorted",
    "sum",
    "tuple",
}


def _literal_data(node: ast.expr) -> bool:
    """Python scalars and containers: never an MPS tensor as a whole."""
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.UnaryOp):
        return isinstance(node.operand, ast.Constant)
    if isinstance(node, (ast.List, ast.Tuple, ast.ListComp, ast.GeneratorExp)):
        return True
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        return node.func.id in _LITERAL_CALLS
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "math"
    )


def _violation(node: ast.Call) -> str | None:
    func = node.func
    if not isinstance(func, ast.Attribute):
        return None
    keywords = {kw.arg for kw in node.keywords}
    if func.attr == "to":
        if None in keywords:
            return ".to(**kwargs) can pass a device and a dtype"
        positional = len(node.args)
        if (
            {"device", "dtype"} <= keywords
            or positional >= 2
            or (positional == 1 and "dtype" in keywords)
        ):
            return ".to(device, dtype) in one call"
        return None
    if func.attr == "type" and node.args:
        return ".type(tensor_type) changes dtype (and maybe device) in one call"
    if (
        isinstance(func.value, ast.Name)
        and func.value.id == "torch"
        and func.attr in {"as_tensor", "asarray"}
        and node.args
        and ("device" in keywords or len(node.args) >= 3)
        and ("dtype" in keywords or len(node.args) >= 2)
        and not _literal_data(node.args[0])
    ):
        return f"torch.{func.attr}(tensor, device=..., dtype=...)"
    return None


def test_no_fused_device_dtype_conversions_outside_helper() -> None:
    found: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        if rel == "device.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            reason = _violation(node)
            if reason is None:
                continue
            key = f"{rel}: {ast.unparse(node)}"
            if key not in ALLOWLIST:
                found.append(f"{rel}:{node.lineno}: {reason}: {ast.unparse(node)}")
    assert not found, (
        "use torchsky.device._to / _as_tensor (MPS float64 copy bug):\n"
        + "\n".join(found)
    )


def test_guard_detects_each_fused_form() -> None:
    sources = [
        "x.to(device=d, dtype=t)",
        "x.to(d, t)",
        "x.to(d, dtype=t)",
        "x.to(**kw)",
        "x.type(torch.DoubleTensor)",
        "torch.as_tensor(x, device=d, dtype=t)",
    ]
    for source in sources:
        call = ast.parse(source, mode="eval").body
        assert isinstance(call, ast.Call)
        assert _violation(call) is not None, source
    for source in (
        "x.to(d)",
        "x.to(dtype=t)",
        "_to(x, d, t)",
        "torch.as_tensor(0.5, device=d, dtype=t)",
        "torch.as_tensor([a, b], device=d, dtype=t)",
    ):
        call = ast.parse(source, mode="eval").body
        assert isinstance(call, ast.Call)
        assert _violation(call) is None, source


def test_to_matches_tensor_to_on_cpu() -> None:
    x = torch.linspace(0.0, 1.0, 5, requires_grad=True)
    y = _to(x, "cpu", torch.float64)
    assert y.dtype == torch.float64
    torch.testing.assert_close(y, x.detach().double())
    y.sum().backward()
    assert x.grad is not None
    torch.testing.assert_close(x.grad, torch.ones(5))
    assert _to(x, None, None) is x
    assert _to(x, device="cpu") is x
    assert _to(x, dtype=torch.float32) is x


def test_to_delegates_to_objects_with_to_method() -> None:
    calls: list[dict[str, object]] = []

    class Movable:
        def to(self, **kwargs: object) -> str:
            calls.append(kwargs)
            return "moved"

    assert _to(Movable(), "cpu", torch.float64) == "moved"
    assert calls == [{"device": "cpu", "dtype": torch.float64}]


def test_as_tensor_accepts_python_and_numpy_data() -> None:
    import numpy as np

    out = _as_tensor([1.0, 2.0], dtype=torch.float64, device="cpu")
    assert out.dtype == torch.float64
    arr = np.arange(3.0)
    torch.testing.assert_close(
        _as_tensor(arr, dtype=torch.float32, device="cpu"),
        torch.tensor([0.0, 1.0, 2.0]),
    )


@requires_mps
@pytest.mark.parametrize(
    "source_dtype", [torch.float32, torch.float16, torch.int64, torch.bool]
)
@pytest.mark.parametrize("target_dtype", [torch.float64, torch.complex128])
def test_to_mps_to_cpu_wide_dtype_is_not_zero(
    source_dtype: torch.dtype, target_dtype: torch.dtype
) -> None:
    base = torch.arange(1, 1001) % 7 + 1
    x = (base > 3) if source_dtype == torch.bool else base.to(source_dtype)
    mps = x.to("mps")
    expected = mps.cpu().to(target_dtype)
    assert bool(expected.abs().sum() > 0)
    for got in (
        _to(mps, "cpu", target_dtype),
        _to(mps, device=torch.device("cpu"), dtype=target_dtype),
        _as_tensor(mps, dtype=target_dtype, device="cpu"),
    ):
        assert got.device.type == "cpu" and got.dtype == target_dtype
        assert torch.equal(got, expected)


@requires_mps
def test_to_mps_to_cpu_keeps_autograd() -> None:
    x = torch.linspace(1.0, 2.0, 4, device="mps", requires_grad=True)
    y = _to(x, "cpu", torch.float64)
    assert torch.equal(y.detach(), x.detach().cpu().double())
    (2.0 * y).sum().backward()
    assert x.grad is not None and x.grad.device.type == "mps"
    torch.testing.assert_close(x.grad.cpu(), torch.full((4,), 2.0))


@requires_mps
def test_to_cpu_to_mps_in_one_step() -> None:
    x = torch.linspace(1.0, 2.0, 5, dtype=torch.float64)
    y = _to(x, "mps", torch.float32)
    assert y.device.type == "mps" and y.dtype == torch.float32
    torch.testing.assert_close(y.cpu(), x.float())


@requires_mps
def test_to_moves_mps_module_to_cpu_float64() -> None:
    module = nn.Linear(3, 2).to("mps")
    reference = {k: v.cpu().double() for k, v in module.state_dict().items()}
    moved = _to(module, "cpu", torch.float64)
    assert moved is module
    for name, value in module.state_dict().items():
        assert value.device.type == "cpu" and value.dtype == torch.float64
        assert torch.equal(value, reference[name])


# --- end-to-end: public paths that leave MPS for CPU float64 -----------------


def _tan_header(size: int = 16) -> dict[str, object]:
    return {
        "NAXIS": 2,
        "NAXIS1": size,
        "NAXIS2": size,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": (size + 1) / 2,
        "CRPIX2": (size + 1) / 2,
        "CRVAL1": 150.0,
        "CRVAL2": 2.0,
        "CDELT1": -1.0 / 3600.0,
        "CDELT2": 1.0 / 3600.0,
    }


@requires_mps
def test_wcs_world_on_mps_moved_to_cpu_float64_matches_cpu() -> None:
    from torchsky.spectral import SpectralAxis
    from torchsky.wcs import WCS
    from torchsky.wcs.gwcs_sampling import IFUSliceWCS

    y, x = torch.meshgrid(
        torch.arange(16, dtype=torch.float64),
        torch.arange(16, dtype=torch.float64),
        indexing="ij",
    )
    ra_cpu, dec_cpu = WCS(_tan_header()).pixel_to_world(x, y)
    ra_mps, dec_mps = (
        WCS(_tan_header())
        .to("mps")
        .pixel_to_world(x.float().to("mps"), y.float().to("mps"))
    )
    wave = 5000.0 + 10.0 * y.float().to("mps")
    ifu = IFUSliceWCS(ra_mps, dec_mps, wave, torch.zeros_like(wave, dtype=torch.int64))
    back = ifu.to("cpu")
    assert back.ra_map.dtype == torch.float64
    torch.testing.assert_close(back.ra_map, ra_cpu, rtol=0.0, atol=1e-4)
    torch.testing.assert_close(back.dec_map, dec_cpu, rtol=0.0, atol=1e-4)
    torch.testing.assert_close(back.wavelength_map, 5000.0 + 10.0 * y)

    axis = SpectralAxis(
        torch.linspace(4000.0, 9000.0, 11, device="mps"), "Angstrom", "WAVE"
    )
    moved = axis.to(device="cpu", dtype=torch.float64)
    torch.testing.assert_close(
        moved.values, torch.linspace(4000.0, 9000.0, 11).double()
    )


@requires_mps
def test_public_to_methods_leave_mps_for_cpu_float64() -> None:
    # The IFU planner (which moves IFUSliceWCS to CPU float64) is covered by
    # tests/test_map_ifu_cube.py::test_ifu_device_dtype_and_memory_bound[mps].
    from torchsky.map.field.types import PSFSpec
    from torchsky.sphere.multiorder import MultiOrderHealpixMap

    values = torch.linspace(1.0, 2.0, 12)
    sky = MultiOrderHealpixMap(torch.arange(4, 16, dtype=torch.int64), values.to("mps"))
    moved = sky.to("cpu", dtype=torch.float64)
    torch.testing.assert_close(moved.values, values.double())

    kernel = torch.rand(5, 5)
    psf = PSFSpec(mode="kernel2d", kernel2d=kernel.to("mps"))
    torch.testing.assert_close(psf.to("cpu", torch.float64).kernel2d, kernel.double())
