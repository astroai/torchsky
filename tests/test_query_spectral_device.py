"""Device contracts for spectral checks in small linear measurement queries."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor
from torch.utils._python_dispatch import TorchDispatchMode

from torchsky.map.likelihood import GaussianLikelihoodFactor, GaussianLikelihoodSystem
from torchsky.map.operators import ObservationOperatorBundle
from torchsky.map.query import fit_linear_query
from torchsky.map.solvers import BoundObservationOperator
from torchsky.map.target import TargetRenderSpec


def _system(data: Tensor, precision):
    def unused(*args):
        raise AssertionError("linear query must not instantiate a field")

    operator = BoundObservationOperator(
        ObservationOperatorBundle(),
        TargetRenderSpec(header={}, shape=(1, 1)),
        apply_operator=unused,
        apply_adjoint=unused,
        apply_noise=precision,
    )
    factor = GaussianLikelihoodFactor(
        operator, data, ("query-samples",), "cal-v1", "raw-v1"
    )
    return GaussianLikelihoodSystem((factor,))


def _dense_reference(
    data: Tensor,
    templates: Tensor,
    weights: Tensor,
    penalty: Tensor | None = None,
) -> dict[str, Tensor]:
    y = data.to(dtype=torch.float64, device="cpu").reshape(-1)
    t = templates.to(dtype=torch.float64, device="cpu").reshape(templates.shape[0], -1)
    w = weights.to(dtype=torch.float64, device="cpu").reshape(-1)
    design = t.T
    information = design.T @ (w[:, None] * design)
    rhs = design.T @ (w * y)
    p = (
        torch.zeros_like(information)
        if penalty is None
        else penalty.to(dtype=torch.float64, device="cpu")
    )
    system = information + p
    coefficients = torch.linalg.solve(system, rhs)
    transfer = torch.linalg.solve(system, information)
    covariance = torch.linalg.solve(system, transfer.T)
    return {
        "information": information,
        "rhs": rhs,
        "penalty": p,
        "coefficients": coefficients,
        "transfer": transfer,
        "covariance": covariance,
    }


def _watch_linalg(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, tuple[int, ...], str]]:
    calls: list[tuple[str, tuple[int, ...], str]] = []
    for name in ("matrix_norm", "eigvalsh", "eigh", "cholesky_ex"):
        original = getattr(torch.linalg, name)

        def wrapped(
            matrix: Tensor,
            *args: object,
            _name: str = name,
            _original=original,
            **kwargs: object,
        ):
            calls.append((_name, tuple(matrix.shape), matrix.device.type))
            return _original(matrix, *args, **kwargs)

        monkeypatch.setattr(torch.linalg, name, wrapped)

    original_solve = torch.cholesky_solve

    def watched_solve(rhs: Tensor, factor: Tensor, *args: object, **kwargs: object):
        calls.append(("cholesky_solve", tuple(factor.shape), factor.device.type))
        return original_solve(rhs, factor, *args, **kwargs)

    monkeypatch.setattr(torch, "cholesky_solve", watched_solve)
    return calls


class _CopyRecorder(TorchDispatchMode):
    """Record tensor device copies so image-sized fallback stays visible."""

    def __init__(self) -> None:
        super().__init__()
        self.copies: list[tuple[tuple[int, ...], str, str]] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        schema = getattr(func, "_schema", None)
        if schema is not None and schema.name == "aten::_to_copy" and args:
            source = args[0]
            target = kwargs.get("device", source.device)
            self.copies.append(
                (tuple(source.shape), source.device.type, torch.device(target).type)
            )
        return func(*args, **kwargs)


def _assert_compact_query_copies(
    copies: list[tuple[tuple[int, ...], str, str]], parameter_count: int
) -> None:
    assert copies
    allowed_shapes = {(), (parameter_count,), (parameter_count, parameter_count)}
    assert all(shape in allowed_shapes for shape, _, _ in copies)
    assert any(
        shape == (parameter_count, parameter_count)
        and source == "mps"
        and target == "cpu"
        for shape, source, target in copies
    )


def _assert_spectral_cpu_and_solves_native(
    calls: list[tuple[str, tuple[int, ...], str]], parameter_count: int
) -> None:
    assert calls
    allowed_shapes = {(), (parameter_count, parameter_count)}
    for name, shape, device in calls:
        if name in {"matrix_norm", "eigvalsh", "eigh"}:
            assert shape in allowed_shapes
            assert device == "cpu"
        elif name in {"cholesky_ex", "cholesky_solve"}:
            assert shape == (parameter_count, parameter_count)
            assert device == "mps"
        else:
            pytest.fail(f"unexpected linalg call: {name}")
    assert any(name == "eigvalsh" for name, _, _ in calls)
    assert any(name == "cholesky_ex" for name, _, _ in calls)
    assert any(name == "cholesky_solve" for name, _, _ in calls)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_forced_mps_query_matches_dense_reference_without_image_copies(monkeypatch):
    templates_cpu = torch.tensor(
        [
            [[1.0, 0.0, 0.0, 2.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        ],
        dtype=torch.float64,
    )
    weights_cpu = torch.tensor(
        [[1.0, 2.0, 1.0, 3.0], [1.0, 2.0, 4.0, 1.0], [2.0, 1.0, 1.0, 1.0]],
        dtype=torch.float64,
    )
    truth = torch.tensor([1.5, -0.7], dtype=torch.float64)
    data_cpu = torch.einsum("p,pij->ij", truth, templates_cpu)
    expected = _dense_reference(data_cpu, templates_cpu, weights_cpu)

    device = torch.device("mps")
    templates = templates_cpu.to(device=device, dtype=torch.float32)
    weights = weights_cpu.to(device=device, dtype=torch.float32)
    data = data_cpu.to(device=device, dtype=torch.float32)
    callback_shapes: list[tuple[tuple[int, ...], str]] = []

    def precision(value: Tensor) -> Tensor:
        callback_shapes.append((tuple(value.shape), value.device.type))
        assert tuple(value.shape) == (3, 4)
        assert value.device.type == "mps"
        return weights * value

    calls = _watch_linalg(monkeypatch)
    copies = _CopyRecorder()
    with copies:
        result = fit_linear_query(_system(data, precision), (templates,))

    assert result.coefficients.device.type == "mps"
    assert result.information.device.type == "mps"
    assert result.data_rhs.device.type == "mps"
    assert result.sampling_covariance.device.type == "mps"
    assert result.mean_response.device.type == "mps"
    assert result.information_eigenvalues.device.type == "mps"
    assert result.rank_cutoff.device.type == "mps"
    torch.testing.assert_close(
        result.coefficients.cpu().double(), truth, rtol=2e-4, atol=2e-4
    )
    torch.testing.assert_close(
        result.information.cpu().double(), expected["information"], rtol=2e-4, atol=2e-4
    )
    torch.testing.assert_close(
        result.data_rhs.cpu().double(), expected["rhs"], rtol=2e-4, atol=2e-4
    )
    torch.testing.assert_close(
        result.mean_response.cpu().double(), expected["transfer"], rtol=3e-4, atol=3e-4
    )
    torch.testing.assert_close(
        result.sampling_covariance.cpu().double(),
        expected["covariance"],
        rtol=3e-4,
        atol=3e-4,
    )
    assert result.data_rank == 2
    assert callback_shapes and set(callback_shapes) == {((3, 4), "mps")}
    _assert_spectral_cpu_and_solves_native(calls, 2)
    _assert_compact_query_copies(copies.copies, 2)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_rank_deficient_query_with_penalty_matches_dense_transfer(monkeypatch):
    base_cpu = torch.tensor(
        [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    templates_cpu = torch.stack((base_cpu, 2 * base_cpu))
    weights_cpu = torch.tensor(
        [[1.0, 2.0, 1.0, 3.0], [1.0, 2.0, 4.0, 1.0], [2.0, 1.0, 1.0, 1.0]],
        dtype=torch.float64,
    )
    truth = torch.tensor([2.0, -0.5], dtype=torch.float64)
    data_cpu = torch.einsum("p,pij->ij", truth, templates_cpu)
    penalty_cpu = torch.tensor([[0.0, 0.0], [0.0, 0.5]], dtype=torch.float64)
    expected = _dense_reference(data_cpu, templates_cpu, weights_cpu, penalty_cpu)
    device = torch.device("mps")
    templates = templates_cpu.to(device=device, dtype=torch.float32)
    weights = weights_cpu.to(device=device, dtype=torch.float32)
    data = data_cpu.to(device=device, dtype=torch.float32)
    penalty = penalty_cpu.to(device=device, dtype=torch.float32)
    callback_devices: list[str] = []

    def precision(value: Tensor) -> Tensor:
        callback_devices.append(value.device.type)
        assert tuple(value.shape) == (3, 4)
        assert value.device.type == "mps"
        return weights * value

    calls = _watch_linalg(monkeypatch)
    copies = _CopyRecorder()
    with copies:
        result = fit_linear_query(
            _system(data, precision), (templates,), penalty_precision=penalty
        )

    assert result.data_rank == 1
    torch.testing.assert_close(
        result.information.cpu().double(), expected["information"], rtol=2e-4, atol=2e-4
    )
    torch.testing.assert_close(
        result.penalty_precision.cpu().double(), penalty_cpu, rtol=1e-6, atol=1e-6
    )
    torch.testing.assert_close(
        result.coefficients.cpu().double(),
        expected["coefficients"],
        rtol=3e-4,
        atol=3e-4,
    )
    torch.testing.assert_close(
        result.mean_response.cpu().double(), expected["transfer"], rtol=3e-4, atol=3e-4
    )
    torch.testing.assert_close(
        result.sampling_covariance.cpu().double(),
        expected["covariance"],
        rtol=3e-4,
        atol=3e-4,
    )
    assert callback_devices and set(callback_devices) == {"mps"}
    _assert_spectral_cpu_and_solves_native(calls, 2)
    _assert_compact_query_copies(copies.copies, 2)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_repairs_only_small_negative_roundoff_before_penalized_solve(monkeypatch):
    templates_cpu = torch.zeros((2, 3, 4), dtype=torch.float64)
    templates_cpu[0, 0, 0] = 1.0
    templates_cpu[1, 0, 1] = 1.0
    data_cpu = torch.ones((3, 4), dtype=torch.float64)
    weights_cpu = torch.zeros((3, 4), dtype=torch.float64)
    weights_cpu[0, 0] = -1e-7
    weights_cpu[0, 1] = 1.0
    penalty_cpu = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float64)

    device = torch.device("mps")
    templates = templates_cpu.to(device=device, dtype=torch.float32)
    data = data_cpu.to(device=device, dtype=torch.float32)
    weights = weights_cpu.to(device=device, dtype=torch.float32)
    penalty = penalty_cpu.to(device=device, dtype=torch.float32)

    def precision(value: Tensor) -> Tensor:
        assert tuple(value.shape) == (3, 4)
        assert value.device.type == "mps"
        return weights * value

    calls = _watch_linalg(monkeypatch)
    copies = _CopyRecorder()
    with copies:
        result = fit_linear_query(
            _system(data, precision),
            (templates,),
            penalty_precision=penalty,
        )

    torch.testing.assert_close(
        result.information.cpu().double(),
        torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=torch.float64),
        rtol=0,
        atol=2e-7,
    )
    assert result.information_eigenvalues[0] < 0
    assert abs(float(result.information_eigenvalues[0].cpu())) < 2e-7
    assert result.data_rank == 1
    assert any(name == "eigh" and device == "cpu" for name, _, device in calls)
    _assert_spectral_cpu_and_solves_native(calls, 2)
    _assert_compact_query_copies(copies.copies, 2)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_large_rank_tolerance_does_not_allow_material_negative_information(
    monkeypatch,
):
    templates_cpu = torch.zeros((2, 3, 4), dtype=torch.float64)
    templates_cpu[0, 0, 0] = 1.0
    templates_cpu[1, 0, 1] = 1.0
    data_cpu = torch.ones((3, 4), dtype=torch.float64)
    weights_cpu = torch.zeros((3, 4), dtype=torch.float64)
    weights_cpu[0, 0] = -1e-3
    weights_cpu[0, 1] = 1.0
    device = torch.device("mps")
    templates = templates_cpu.to(device=device, dtype=torch.float32)
    data = data_cpu.to(device=device, dtype=torch.float32)
    weights = weights_cpu.to(device=device, dtype=torch.float32)

    def precision(value: Tensor) -> Tensor:
        assert value.device.type == "mps"
        return weights * value

    calls = _watch_linalg(monkeypatch)
    copies = _CopyRecorder()
    with copies:
        with pytest.raises(ValueError, match="positive semidefinite"):
            fit_linear_query(
                _system(data, precision),
                (templates,),
                rank_rtol=100.0,
            )

    assert any(
        name == "eigvalsh" and shape == (2, 2) and device == "cpu"
        for name, shape, device in calls
    )
    _assert_compact_query_copies(copies.copies, 2)


def test_cpu_query_spectral_and_cholesky_paths_remain_cpu(monkeypatch):
    templates = torch.zeros((2, 3, 4), dtype=torch.float64)
    templates[0, 0, 0] = 1.0
    templates[0, 1, 1] = 1.0
    templates[1, 0, 1] = 1.0
    templates[1, 2, 3] = 1.0
    data = torch.ones((3, 4), dtype=torch.float64)
    calls = _watch_linalg(monkeypatch)
    copies = _CopyRecorder()
    with copies:
        result = fit_linear_query(_system(data, lambda value: value), (templates,))

    assert result.coefficients.device.type == "cpu"
    assert calls
    assert all(device == "cpu" for _, _, device in calls)
    assert not copies.copies
