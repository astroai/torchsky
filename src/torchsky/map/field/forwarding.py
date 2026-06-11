"""Context-aware field-model call helpers."""

from __future__ import annotations

import functools
import inspect

from torch import Tensor, nn

from .types import ForwardContext


@functools.lru_cache(maxsize=512)
def type_method_accepts_kwarg(cls: type, method_name: str, kw: str) -> bool:
    fn = getattr(cls, method_name, None)
    if fn is None:
        return False
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    if kw in sig.parameters:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())


def call_scalar_field_forward(
    model: nn.Module,
    ra: Tensor,
    dec: Tensor,
    wavelength: Tensor | float,
    *,
    forward_context: ForwardContext | None,
) -> Tensor:
    if forward_context is None:
        return model(ra, dec, wavelength)
    if type_method_accepts_kwarg(type(model), "forward", "forward_context"):
        return model(ra, dec, wavelength, forward_context=forward_context)
    return model(ra, dec, wavelength)


def call_render_bandpass(
    model: nn.Module,
    ra: Tensor,
    dec: Tensor,
    wavelengths: Tensor,
    weights: Tensor,
    *,
    forward_context: ForwardContext | None,
) -> Tensor:
    rb = model.render_bandpass
    if forward_context is None:
        return rb(ra, dec, wavelengths, weights)
    if type_method_accepts_kwarg(type(model), "render_bandpass", "forward_context"):
        return rb(ra, dec, wavelengths, weights, forward_context=forward_context)
    return rb(ra, dec, wavelengths, weights)


def call_render_bandpass_distribution(
    model: nn.Module,
    ra: Tensor,
    dec: Tensor,
    wavelengths: Tensor,
    weights: Tensor,
    *,
    forward_context: ForwardContext | None,
) -> tuple[Tensor, Tensor]:
    rb = model.render_bandpass_distribution
    if forward_context is None:
        return rb(ra, dec, wavelengths, weights)
    if type_method_accepts_kwarg(
        type(model), "render_bandpass_distribution", "forward_context"
    ):
        return rb(ra, dec, wavelengths, weights, forward_context=forward_context)
    return rb(ra, dec, wavelengths, weights)


def call_forward_distribution(
    model: nn.Module,
    ra: Tensor,
    dec: Tensor,
    wavelength: Tensor | float,
    *,
    forward_context: ForwardContext | None,
) -> tuple[Tensor, Tensor]:
    fd = model.forward_distribution
    if forward_context is None:
        return fd(ra, dec, wavelength)
    if type_method_accepts_kwarg(type(model), "forward_distribution", "forward_context"):
        return fd(ra, dec, wavelength, forward_context=forward_context)
    return fd(ra, dec, wavelength)


__all__ = [
    "call_forward_distribution",
    "call_render_bandpass",
    "call_render_bandpass_distribution",
    "call_scalar_field_forward",
    "type_method_accepts_kwarg",
]
