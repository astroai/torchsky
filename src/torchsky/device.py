"""Default compute device selection for tensor pipelines.

Environment:

* ``TORCHSKY_DEVICE`` — if set (e.g. ``cuda``, ``cuda:1``, ``cpu``), overrides the
  automatic default for APIs that use :func:`default_compute_device`.

Placement rule (:func:`resolve_device`): an explicit ``device=`` always wins;
otherwise a public API computes and returns on the device of its tensor inputs;
:func:`default_compute_device` applies only when there is nothing to follow
(file-path observations, constructors without tensors).
"""

import os
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from typing import Any, TypeVar

import torch
from torch import nn


def default_compute_device() -> torch.device:
    """Prefer CUDA when available; otherwise CPU.

    If ``TORCHSKY_DEVICE`` is set to a non-empty string, it is parsed as a
    :class:`torch.device` and returned.
    """
    raw = os.environ.get("TORCHSKY_DEVICE", "").strip()
    if raw:
        return torch.device(raw)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _collect_devices(obj: object, found: list[torch.device], depth: int = 0) -> None:
    if obj is None or depth > 3:
        return
    if isinstance(obj, torch.Tensor):
        found.append(obj.device)
    elif isinstance(obj, nn.Module):
        for t in list(obj.parameters()) + list(obj.buffers()):
            _collect_devices(t, found, depth + 1)
    elif isinstance(obj, (str, bytes, int, float, torch.device, torch.dtype)):
        return
    elif isinstance(obj, Mapping):
        for v in obj.values():
            _collect_devices(v, found, depth + 1)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for v in obj:
            _collect_devices(v, found, depth + 1)
    elif is_dataclass(obj) and not isinstance(obj, type):
        # Tensor-valued fields only; WCS objects/headers never decide placement
        # for an observation (a path-backed observation has nothing to follow).
        for f in fields(obj):
            v = getattr(obj, f.name, None)
            if isinstance(v, (torch.Tensor, list, tuple)):
                _collect_devices(v, found, depth + 1)
    else:
        dev = getattr(obj, "device", None)
        if isinstance(dev, torch.device):
            found.append(dev)


def _same_device(a: torch.device, b: torch.device) -> bool:
    if a.type != b.type:
        return False
    if a.index is None or b.index is None:
        # ``cuda`` and ``cuda:0`` denote the same current device.
        return a.type != "cuda" or (a.index or 0) == (b.index or 0)
    return a.index == b.index


def resolve_device(
    explicit: torch.device | str | None,
    *sources: object,
) -> torch.device:
    """Resolve where a public API computes and returns.

    Rule (device follows the input):

    * An ``explicit`` device always wins.
    * Otherwise the device of the tensor inputs is used. ``sources`` may be
      tensors, ``nn.Module`` (parameters and buffers), objects with a
      ``.device`` attribute (e.g. ``WCS``), dataclasses such as
      ``ImageObservation`` (their tensor fields), and lists, tuples, or mappings
      of these. CPU tensors are an input like any other: CPU stays CPU even when
      CUDA is available. ``meta`` tensors are ignored unless every input is ``meta``.
    * Tensors on different devices without an explicit device raise
      :class:`ValueError`; data is never moved silently.
    * Only when no source carries a tensor (e.g. observations given as file
      paths) does :func:`default_compute_device` apply.
    """
    if explicit is not None:
        return torch.device(explicit)
    found: list[torch.device] = []
    for src in sources:
        _collect_devices(src, found)
    real = [d for d in found if d.type != "meta"]
    found = real or found
    if not found:
        return default_compute_device()
    first = found[0]
    for other in found[1:]:
        if not _same_device(first, other):
            raise ValueError(
                "inputs live on different devices "
                f"({sorted({str(d) for d in found})}); move them to one device or "
                "pass an explicit device="
            )
    return first


def require_float64(device: torch.device | str | None, *, what: str) -> None:
    """Raise early when a float64-only path is asked to run on MPS.

    MPS has no float64. ``what`` names the entry point, e.g. ``"drizzle_square"``.
    Nothing is moved to CPU on the caller's behalf.
    """
    if device is not None and torch.device(device).type == "mps":
        raise NotImplementedError(
            f"{what} needs float64 coordinates, which MPS does not support; "
            "run on CPU or CUDA (device='cpu')"
        )


def resolve_fit_device(
    model: nn.Module,
    device: torch.device | str | None,
) -> torch.device:
    """Pick where to run training for ``fit_field_model``-style loops.

    Same rule as :func:`resolve_device` with the model as the input: an explicit
    ``device`` wins, otherwise the device of the model's parameters and buffers
    is kept (CPU included), and :func:`default_compute_device` applies only to a
    model without tensors.
    """
    return resolve_device(device, model)


_T = TypeVar("_T")


def _to(
    obj: _T,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> _T:
    """``obj.to(device=device, dtype=dtype)`` that is correct for MPS sources.

    PyTorch 2.14 on MPS returns zeros, without warning, when one ``.to`` call
    both leaves MPS and casts to a dtype that MPS lacks (``float64``,
    ``complex128``): ``mps_f32.to("cpu", torch.float64)`` is all zeros, while
    ``mps_f32.cpu().double()`` is right. The same cast path is behind
    ``.to(cpu_f64_tensor)``, ``.type(torch.DoubleTensor)``,
    ``torch.as_tensor``/``torch.tensor`` with ``device`` and ``dtype``, and
    ``cpu_f64.copy_(mps_tensor)``. Seen with PyTorch 2.14.1 on Apple silicon;
    integer, float32 and float16 targets are unaffected. MPS dtype limits:
    https://docs.pytorch.org/docs/stable/notes/mps.html

    This helper moves an MPS source to the target device first and casts
    there; every other combination is one ``.to`` call. ``obj`` may be a
    tensor, an ``nn.Module`` (its first parameter or buffer gives the source
    device) or any object with a ``to(device=..., dtype=...)`` method, which is
    called unchanged. Autograd is preserved. Every in-package ``.to`` call that
    passes both a device and a dtype goes through here
    (``tests/test_device_to.py`` enforces it).
    """
    mover: Any = obj
    source = _source_device(obj)
    if (
        device is not None
        and dtype is not None
        and source is not None
        and source.type == "mps"
        and torch.device(device).type != "mps"
    ):
        return mover.to(device=device).to(dtype=dtype)
    return mover.to(device=device, dtype=dtype)


def _source_device(obj: object) -> torch.device | None:
    if isinstance(obj, torch.Tensor):
        return obj.device
    if isinstance(obj, nn.Module):
        for tensor in obj.parameters():
            return tensor.device
        for tensor in obj.buffers():
            return tensor.device
    return None


def _as_tensor(
    data: Any,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """``torch.as_tensor(data, dtype=dtype, device=device)`` via :func:`_to`.

    ``torch.as_tensor`` of an MPS tensor with a CPU ``device`` and a float64
    ``dtype`` hits the same zero-filling copy as ``Tensor.to``.
    """
    if isinstance(data, torch.Tensor):
        return _to(data, device=device, dtype=dtype)
    return torch.as_tensor(data, dtype=dtype, device=device)


_INT32_MIN = -(2**31)
_INT32_MAX = 2**31 - 1


def _scatter_reduce_int(
    target: torch.Tensor,
    dim: int,
    index: torch.Tensor,
    src: torch.Tensor,
    reduce: str,
    *,
    include_self: bool = True,
) -> torch.Tensor:
    """Out-of-place ``target.scatter_reduce`` that is portable for int64 on MPS.

    An int64 ``"amin"``/``"amax"`` scatter on MPS compiles a Metal kernel that
    needs 64-bit atomics, which some Apple GPUs (and the virtualised GPU of
    GitHub's macOS runners) lack: pipeline creation fails with
    ``CompilerError``. On MPS, int64 operands are reduced in int32 when every
    value fits, which only needs 32-bit atomics and gives identical results;
    otherwise the reduction runs on the CPU and the result is moved back. Every
    other device and dtype calls ``scatter_reduce`` unchanged.
    """
    if target.device.type != "mps" or target.dtype != torch.int64:
        return target.scatter_reduce(dim, index, src, reduce, include_self=include_self)
    src = src.to(torch.int64)
    fits = True
    for values in (target, src):
        if values.numel():
            lo, hi = torch.aminmax(values)
            fits = fits and int(lo) >= _INT32_MIN and int(hi) <= _INT32_MAX
    if fits:
        reduced = target.to(torch.int32).scatter_reduce(
            dim, index, src.to(torch.int32), reduce, include_self=include_self
        )
        return reduced.to(torch.int64)
    reduced = target.cpu().scatter_reduce(
        dim, index.cpu(), src.cpu(), reduce, include_self=include_self
    )
    return reduced.to(target.device)
