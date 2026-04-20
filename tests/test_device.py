"""Tests for torchsky.device default resolution."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from torchsky.device import (
    default_compute_device,
    module_compute_device,
    resolve_fit_device,
)


def test_default_compute_device_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TORCHSKY_DEVICE", "cpu")
    assert str(default_compute_device()) == "cpu"


def test_module_compute_device_from_parameter() -> None:
    m = nn.Linear(2, 3)
    assert module_compute_device(m) == m.weight.device


def test_module_compute_device_from_buffer() -> None:
    class BufferModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("buf", torch.zeros(1))

    m = BufferModule()
    assert module_compute_device(m) == m.buf.device


def test_module_compute_device_fallback() -> None:
    m = nn.Module()
    assert module_compute_device(m) == default_compute_device()


def test_resolve_fit_device_explicit() -> None:
    m = nn.Linear(1, 1)
    assert str(resolve_fit_device(m, "cpu")) == "cpu"


def test_resolve_fit_device_keeps_non_cpu_placement() -> None:
    m = nn.Linear(1, 1)
    if not torch.cuda.is_available():
        return
    m = m.cuda()
    assert resolve_fit_device(m, None).type == "cuda"
