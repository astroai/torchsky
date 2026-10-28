"""Stable structural fingerprints for immutable operator graphs and plans."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
from typing import Mapping, Protocol, runtime_checkable

import torch
from torch import Tensor

from ..domain import DomainSpec


@dataclass(frozen=True, slots=True)
class PlanFingerprint:
    kind: str
    digest: str

    def __post_init__(self) -> None:
        if not self.kind or not self.digest:
            raise ValueError("fingerprint kind and digest must be nonempty")

    def __str__(self) -> str:
        return f"{self.kind}:{self.digest}"


@runtime_checkable
class Fingerprinted(Protocol):
    @property
    def fingerprint(self) -> PlanFingerprint: ...


def _canonical(value: object) -> object:
    if isinstance(value, PlanFingerprint):
        return {"kind": value.kind, "digest": value.digest}
    if isinstance(value, DomainSpec):
        return {
            "shape": value.shape,
            "dtype": None if value.dtype is None else str(value.dtype),
            "axes": value.axes,
            "units": value.units,
        }
    if isinstance(value, Tensor):
        return {
            "shape": tuple(value.shape),
            "dtype": str(value.dtype),
            "device": value.device.type,
        }
    if isinstance(value, torch.dtype):
        return str(value)
    if is_dataclass(value):
        return _canonical(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported fingerprint payload type {type(value).__name__}")


def plan_fingerprint(kind: str, payload: object) -> PlanFingerprint:
    if not isinstance(kind, str) or not kind.strip():
        raise ValueError("fingerprint kind must be a nonempty string")
    encoded = json.dumps(
        _canonical(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return PlanFingerprint(kind, hashlib.sha256(encoded).hexdigest())


__all__ = ["Fingerprinted", "PlanFingerprint", "plan_fingerprint"]
