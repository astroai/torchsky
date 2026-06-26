from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "reproduce_smacs0723_updc.py"
)
SPEC = importlib.util.spec_from_file_location("reproduce_smacs0723_updc", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_fft_convolver_adjoint_identity() -> None:
    torch.manual_seed(7)
    image = torch.randn(13, 11)
    dual = torch.randn(13, 11)
    kernel = torch.rand(5, 3)
    kernel = kernel / kernel.sum()
    operator = MODULE.FFTSameConvolver(kernel, tuple(image.shape))

    lhs = torch.sum(operator(image) * dual)
    rhs = torch.sum(image * operator.adjoint(dual))
    torch.testing.assert_close(lhs, rhs, rtol=2.0e-5, atol=2.0e-5)


def test_ratio_is_unclipped_by_default() -> None:
    observed = torch.tensor([[2.0, -1.0], [4.0, 8.0]])
    predicted = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    valid = torch.ones_like(observed, dtype=torch.bool)
    ratio = MODULE._ratio(
        observed,
        predicted,
        valid,
        epsilon=1.0e-8,
        ratio_clip=None,
    )
    torch.testing.assert_close(
        ratio,
        torch.tensor([[2.0, -1.0], [2.0, 4.0]]),
    )
