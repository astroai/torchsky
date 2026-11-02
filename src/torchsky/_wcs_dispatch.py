from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def _validate_fake_inputs(
    first: Tensor,
    second: Tensor,
    tpv_idx1: Tensor,
    tpv_c1: Tensor,
    tpv_idx2: Tensor,
    tpv_c2: Tensor,
) -> None:
    torch._check(
        first.shape == second.shape, lambda: "WCS inputs must have identical shapes"
    )
    torch._check(first.dtype == torch.float64, lambda: "WCS inputs must be float64")
    torch._check(second.dtype == torch.float64, lambda: "WCS inputs must be float64")
    torch._check(
        first.device == second.device, lambda: "WCS inputs must share a device"
    )
    for idx in (tpv_idx1, tpv_idx2):
        torch._check(idx.ndim == 2, lambda: "TPV exponents must be rank 2")
        torch._check(idx.shape[1] == 3, lambda: "TPV exponents must have shape [N, 3]")
        torch._check(idx.dtype == torch.int64, lambda: "TPV exponents must be int64")
    for idx, coeff in ((tpv_idx1, tpv_c1), (tpv_idx2, tpv_c2)):
        torch._check(coeff.ndim == 1, lambda: "TPV coefficients must be rank 1")
        torch._check(
            idx.shape[0] == coeff.shape[0], lambda: "TPV metadata lengths must match"
        )
        torch._check(
            coeff.dtype == torch.float64, lambda: "TPV coefficients must be float64"
        )
        torch._check(
            idx.device == first.device,
            lambda: "TPV metadata must share the input device",
        )
        torch._check(
            coeff.device == first.device,
            lambda: "TPV metadata must share the input device",
        )


def _fake_impl(*args: Any) -> tuple[Tensor, Tensor]:
    first, second = args[:2]
    tpv_idx1, tpv_c1, tpv_idx2, tpv_c2 = args[27:31]
    _validate_fake_inputs(first, second, tpv_idx1, tpv_c1, tpv_idx2, tpv_c2)
    return (
        torch.empty_like(first, memory_format=torch.contiguous_format),
        torch.empty_like(first, memory_format=torch.contiguous_format),
    )


def _setup_context(
    ctx: Any, inputs: tuple[Any, ...], output: tuple[Tensor, Tensor]
) -> None:
    del output
    ctx.save_for_backward(inputs[0], inputs[1], *inputs[27:31])
    ctx.config = inputs[2:27]


def _backward(
    ctx: Any, grad_first: Tensor | None, grad_second: Tensor | None
) -> tuple[Any, ...]:
    first, second, tpv_idx1, tpv_c1, tpv_idx2, tpv_c2 = ctx.saved_tensors
    args = (
        first,
        second,
        *ctx.config,
        tpv_idx1,
        tpv_c1,
        tpv_idx2,
        tpv_c2,
    )
    from torchsky.wcs.functional import _wcs_reference

    create_graph = torch.is_grad_enabled()
    with torch.enable_grad():
        output = _wcs_reference(ctx.direction, *args)
        grad_outputs = (
            torch.zeros_like(output[0]) if grad_first is None else grad_first,
            torch.zeros_like(output[1]) if grad_second is None else grad_second,
        )
        differentiable = tuple(
            tensor
            for index, tensor in zip(
                (0, 1, 27, 28, 29, 30),
                (first, second, tpv_idx1, tpv_c1, tpv_idx2, tpv_c2),
                strict=True,
            )
            if ctx.needs_input_grad[index] and tensor.is_floating_point()
        )
        computed = torch.autograd.grad(
            output,
            differentiable,
            grad_outputs,
            create_graph=create_graph,
            allow_unused=True,
        )
    result: list[Tensor | None] = [None] * 31
    grad_iter = iter(computed)
    for index, tensor in zip(
        (0, 1, 27, 28, 29, 30),
        (first, second, tpv_idx1, tpv_c1, tpv_idx2, tpv_c2),
        strict=True,
    ):
        if ctx.needs_input_grad[index] and tensor.is_floating_point():
            result[index] = next(grad_iter)
    return tuple(result)


def _register() -> None:
    for direction in ("pixel_to_world", "world_to_pixel"):
        qualified = f"torchsky::wcs_{direction}"
        torch.library.register_fake(qualified)(_fake_impl)

        def setup_context(
            ctx: Any,
            inputs: tuple[Any, ...],
            output: tuple[Tensor, Tensor],
            direction: str = direction,
        ) -> None:
            _setup_context(ctx, inputs, output)
            ctx.direction = direction

        torch.library.register_autograd(
            qualified,
            _backward,
            setup_context=setup_context,
        )


_register()
