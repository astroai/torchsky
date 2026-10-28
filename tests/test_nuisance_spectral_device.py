"""Device contracts for the compact spectral nuisance-system path."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor
from torch.utils._python_dispatch import TorchDispatchMode

from torchsky.map.operators.nuisance import (
    CalibratedNuisancePrecision,
    ProfiledNuisancePrecision,
)


def _mps_available() -> bool:
    return torch.backends.mps.is_available()


def _profile_reference(
    basis: Tensor,
    weights: Tensor,
    residual: Tensor,
    *,
    rank_atol: float,
    rank_rtol: float,
) -> tuple[Tensor, int]:
    """Independent float64 dense reference for the declared rank policy."""
    b = basis.to(dtype=torch.float64, device="cpu")
    w = weights.to(dtype=torch.float64, device="cpu")
    r = residual.to(dtype=torch.float64, device="cpu")
    precision = torch.diag(w)
    gram = b.T @ precision @ b
    eigenvalues = torch.linalg.eigvalsh(gram)
    threshold = max(rank_atol, rank_rtol * float(eigenvalues.abs().max()))
    rank = int((eigenvalues > threshold).sum())
    gram_pinv = torch.linalg.pinv(gram, atol=rank_atol, rtol=rank_rtol, hermitian=True)
    result = precision @ r - precision @ b @ gram_pinv @ b.T @ precision @ r
    return result, rank


def _watch_linalg(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, tuple[int, ...], str]]:
    calls: list[tuple[str, tuple[int, ...], str]] = []
    for name in ("eigvalsh", "pinv", "cholesky"):
        original = getattr(torch.linalg, name)

        def wrapped(
            matrix: Tensor,
            *args: object,
            _name: str = name,
            _original=original,
            **kwargs: object,
        ) -> Tensor:
            calls.append((_name, tuple(matrix.shape), matrix.device.type))
            return _original(matrix, *args, **kwargs)

        monkeypatch.setattr(torch.linalg, name, wrapped)

    original_solve = torch.cholesky_solve

    def watched_solve(
        rhs: Tensor, factor: Tensor, *args: object, **kwargs: object
    ) -> Tensor:
        calls.append(("cholesky_solve", tuple(factor.shape), factor.device.type))
        return original_solve(rhs, factor, *args, **kwargs)

    monkeypatch.setattr(torch, "cholesky_solve", watched_solve)
    return calls


class _CopyRecorder(TorchDispatchMode):
    """Record explicit tensor device copies made during operator use."""

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


def _assert_only_compact_copies(
    copies: list[tuple[tuple[int, ...], str, str]], rank: int
) -> None:
    assert copies
    assert all(shape in {(), (rank, rank)} for shape, _, _ in copies)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
    ],
)
def test_cpu_cuda_keep_spectral_linalg_on_the_data_device(device: str):
    basis = torch.tensor(
        [[1.0, 0.0], [1.0, 1.0], [0.0, 2.0]],
        dtype=torch.float32,
        device=device,
    )
    residual = torch.tensor([0.5, -1.0, 2.0], dtype=basis.dtype, device=device)
    profiled = ProfiledNuisancePrecision(basis, apply_precision=lambda x: 2 * x)
    calibrated = CalibratedNuisancePrecision(
        basis,
        apply_precision=lambda x: 2 * x,
        covariance_factor=torch.eye(2, dtype=basis.dtype, device=device),
    )
    for operator in (profiled, calibrated):
        assert operator.spectral_device == basis.device
        assert operator(residual).device == basis.device


@pytest.mark.skipif(not _mps_available(), reason="MPS unavailable")
def test_mps_full_rank_profile_uses_cpu_spectrum_and_native_cholesky(monkeypatch):
    basis_cpu = torch.tensor(
        [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, -1.0]], dtype=torch.float64
    )
    weights_cpu = torch.tensor([2.0, 0.5, 0.0, 3.0], dtype=torch.float64)
    residual_cpu = torch.tensor([2.0, -1.0, 8.0, 3.0], dtype=torch.float64)
    expected, expected_rank = _profile_reference(
        basis_cpu,
        weights_cpu,
        residual_cpu,
        rank_atol=0.0,
        rank_rtol=2 * torch.finfo(torch.float32).eps,
    )
    device = torch.device("mps")
    basis = basis_cpu.to(device=device, dtype=torch.float32)
    weights = weights_cpu.to(device=device, dtype=torch.float32)
    residual = residual_cpu.to(device=device, dtype=torch.float32)
    callback_devices: list[str] = []

    def precision_action(value: Tensor) -> Tensor:
        callback_devices.append(value.device.type)
        assert value.device.type == "mps"
        return weights * value

    calls = _watch_linalg(monkeypatch)
    copies = _CopyRecorder()
    with copies:
        operator = ProfiledNuisancePrecision(basis, apply_precision=precision_action)
        result = operator(residual)

    assert operator.spectral_device == torch.device("cpu")
    assert operator.nuisance_rank == expected_rank == 2
    assert operator._cholesky is not None and operator._cholesky.device.type == "mps"
    assert result.device.type == "mps"
    torch.testing.assert_close(result.cpu().double(), expected, rtol=5e-5, atol=5e-5)
    assert calls == [
        ("eigvalsh", (2, 2), "cpu"),
        ("cholesky", (2, 2), "mps"),
        ("cholesky_solve", (2, 2), "mps"),
    ]
    assert callback_devices and set(callback_devices) == {"mps"}
    _assert_only_compact_copies(copies.copies, 2)


@pytest.mark.parametrize(
    "basis_values,weights_values,residual_values,rank_atol,expected_rank",
    [
        (
            [[1.0, 2.0, 0.0], [2.0, 4.0, 0.0], [3.0, 6.0, 0.0], [4.0, 8.0, 0.0]],
            [1.0, 2.0, 0.5, 4.0],
            [3.0, -1.0, 20.0, 2.0],
            0.0,
            1,
        ),
        ([[1.0, 0.0], [0.0, 0.01]], [1.0, 1.0], [2.0, -3.0], 1e-3, 1),
    ],
    ids=["redundant", "rank-truncated"],
)
@pytest.mark.skipif(not _mps_available(), reason="MPS unavailable")
def test_mps_singular_profile_matches_dense_reference_and_pinv_is_compact_cpu(
    basis_values, weights_values, residual_values, rank_atol, expected_rank, monkeypatch
):
    basis_cpu = torch.tensor(basis_values, dtype=torch.float64)
    weights_cpu = torch.tensor(weights_values, dtype=torch.float64)
    residual_cpu = torch.tensor(residual_values, dtype=torch.float64)
    rank_rtol = len(basis_values[0]) * torch.finfo(torch.float32).eps
    expected, ref_rank = _profile_reference(
        basis_cpu,
        weights_cpu,
        residual_cpu,
        rank_atol=rank_atol,
        rank_rtol=rank_rtol,
    )
    device = torch.device("mps")
    basis = basis_cpu.to(device=device, dtype=torch.float32)
    weights = weights_cpu.to(device=device, dtype=torch.float32)
    residual = residual_cpu.to(device=device, dtype=torch.float32)
    callback_devices: list[str] = []

    def precision_action(value: Tensor) -> Tensor:
        callback_devices.append(value.device.type)
        assert value.device.type == "mps"
        return weights * value

    calls = _watch_linalg(monkeypatch)
    copies = _CopyRecorder()
    with copies:
        operator = ProfiledNuisancePrecision(
            basis,
            apply_precision=precision_action,
            rank_atol=rank_atol,
        )
        result = operator(residual)

    mode_count = basis.shape[-1]
    assert operator.spectral_device == torch.device("cpu")
    assert operator.nuisance_rank == expected_rank == ref_rank
    assert operator._cholesky is None
    assert operator._pseudoinverse is not None
    assert operator._pseudoinverse.device.type == "mps"
    assert result.device.type == "mps"
    torch.testing.assert_close(result.cpu().double(), expected, rtol=1e-4, atol=1e-4)
    assert calls == [
        ("eigvalsh", (mode_count, mode_count), "cpu"),
        ("pinv", (mode_count, mode_count), "cpu"),
    ]
    assert callback_devices and set(callback_devices) == {"mps"}
    _assert_only_compact_copies(copies.copies, mode_count)


@pytest.mark.skipif(not _mps_available(), reason="MPS unavailable")
def test_mps_calibrated_singular_factor_matches_dense_covariance_and_logdet(
    monkeypatch,
):
    basis_cpu = torch.tensor(
        [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, -1.0]], dtype=torch.float64
    )
    weights_cpu = torch.tensor([2.0, 0.5, 1.0, 3.0], dtype=torch.float64)
    factor_cpu = torch.tensor([[0.5, 0.0, 1.0], [0.2, 0.0, 0.4]], dtype=torch.float64)
    residual_cpu = torch.tensor([0.5, -1.0, 2.0, 0.25], dtype=torch.float64)
    f = basis_cpu @ factor_cpu
    base_covariance = torch.diag(weights_cpu.reciprocal())
    covariance = base_covariance + f @ f.T
    expected_action = torch.linalg.solve(covariance, residual_cpu)
    expected_logdet = (
        torch.linalg.slogdet(covariance).logabsdet
        - torch.linalg.slogdet(base_covariance).logabsdet
    )
    device = torch.device("mps")
    basis = basis_cpu.to(device=device, dtype=torch.float32)
    weights = weights_cpu.to(device=device, dtype=torch.float32)
    factor = factor_cpu.to(device=device, dtype=torch.float32)
    residual = residual_cpu.to(device=device, dtype=torch.float32)
    callback_devices: list[str] = []

    def precision_action(value: Tensor) -> Tensor:
        callback_devices.append(value.device.type)
        assert value.device.type == "mps"
        return weights * value

    calls = _watch_linalg(monkeypatch)
    copies = _CopyRecorder()
    with copies:
        operator = CalibratedNuisancePrecision(
            basis,
            apply_precision=precision_action,
            covariance_factor=factor,
        )
        result = operator(residual)

    assert operator.spectral_device == torch.device("cpu")
    assert operator._cholesky is not None and operator._cholesky.device.type == "mps"
    assert operator.logdet_covariance_correction.device.type == "mps"
    torch.testing.assert_close(
        result.cpu().double(), expected_action, rtol=7e-5, atol=7e-5
    )
    torch.testing.assert_close(
        operator.logdet_covariance_correction.cpu().double(),
        expected_logdet,
        rtol=7e-5,
        atol=7e-5,
    )
    assert calls == [
        ("eigvalsh", (3, 3), "cpu"),
        ("cholesky", (3, 3), "mps"),
        ("cholesky_solve", (3, 3), "mps"),
    ]
    assert callback_devices and set(callback_devices) == {"mps"}
    _assert_only_compact_copies(copies.copies, 3)


@pytest.mark.parametrize("factor_shape", [(2, 0), (2, 2)])
@pytest.mark.skipif(not _mps_available(), reason="MPS unavailable")
def test_mps_zero_calibration_covariance_is_exact_base_limit(factor_shape):
    device = torch.device("mps")
    basis = torch.tensor([[1.0, 0.0], [1.0, 1.0]], device=device)
    weights = torch.tensor([2.0, 0.5], device=device)
    residual = torch.tensor([3.0, -2.0], device=device)
    factor = torch.zeros(factor_shape, device=device)
    operator = CalibratedNuisancePrecision(
        basis,
        apply_precision=lambda x: weights * x,
        covariance_factor=factor,
    )
    torch.testing.assert_close(operator(residual), weights * residual)
    assert operator.logdet_covariance_correction.device.type == "mps"
    assert operator.logdet_covariance_correction == 0.0


@pytest.mark.skipif(not _mps_available(), reason="MPS unavailable")
def test_mps_masked_nonfinite_support_stays_on_device_and_matches_dense_reference():
    device = torch.device("mps")
    valid_cpu = torch.tensor([True, False, True, True])
    basis_cpu = torch.tensor(
        [[1.0, 0.0], [float("nan"), float("nan")], [1.0, 2.0], [1.0, -1.0]],
        dtype=torch.float64,
    )
    weights_cpu = torch.tensor([2.0, 9.0, 0.5, 3.0], dtype=torch.float64)
    residual_cpu = torch.tensor([0.5, float("nan"), -0.3, 2.0], dtype=torch.float64)
    observed_basis = basis_cpu[valid_cpu]
    observed_weights = weights_cpu[valid_cpu]
    observed_residual = residual_cpu[valid_cpu]
    expected_observed, _ = _profile_reference(
        observed_basis,
        observed_weights,
        observed_residual,
        rank_atol=0.0,
        rank_rtol=2 * torch.finfo(torch.float32).eps,
    )
    expected = torch.zeros(4, dtype=torch.float64)
    expected[valid_cpu] = expected_observed
    basis = basis_cpu.to(device=device, dtype=torch.float32)
    valid = valid_cpu.to(device=device)
    weights = weights_cpu.to(device=device, dtype=torch.float32)
    residual = residual_cpu.to(device=device, dtype=torch.float32)
    callback_devices: list[str] = []

    def precision_action(value: Tensor) -> Tensor:
        callback_devices.append(value.device.type)
        assert value.device.type == "mps"
        assert torch.all(value[~valid] == 0)
        return weights * value

    copies = _CopyRecorder()
    with copies:
        operator = ProfiledNuisancePrecision(
            basis, apply_precision=precision_action, valid=valid
        )
        result = operator(residual)

    assert result.device.type == "mps"
    assert torch.all(result[~valid] == 0)
    torch.testing.assert_close(result.cpu().double(), expected, rtol=1e-4, atol=1e-4)
    assert callback_devices and set(callback_devices) == {"mps"}
    _assert_only_compact_copies(copies.copies, 2)


@pytest.mark.parametrize("treatment", ["profiled", "calibrated"])
@pytest.mark.parametrize("invalid_gram", ["negative", "asymmetric"])
@pytest.mark.skipif(not _mps_available(), reason="MPS unavailable")
def test_mps_rejects_invalid_nuisance_gram_independently_of_rank_tolerance(
    treatment: str, invalid_gram: str
):
    device = torch.device("mps")
    basis = torch.eye(2, dtype=torch.float32, device=device)
    if invalid_gram == "negative":
        precision = torch.diag(torch.tensor([-1.0, 1.0], device=device))
        match = "negative"
    else:
        precision = torch.tensor([[2.0, 1.0], [0.0, 2.0]], device=device)
        match = "asymmetric"

    def precision_action(value: Tensor) -> Tensor:
        assert value.device.type == "mps"
        return precision @ value

    with pytest.raises(ValueError, match=match):
        if treatment == "profiled":
            ProfiledNuisancePrecision(
                basis,
                apply_precision=precision_action,
                rank_rtol=100.0,
            )
        else:
            CalibratedNuisancePrecision(
                basis,
                apply_precision=precision_action,
                covariance_factor=torch.eye(2, device=device),
            )
