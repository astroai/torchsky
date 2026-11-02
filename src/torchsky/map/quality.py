"""Explicit detector DQ policies and map quality products."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

import torch
from torch import Tensor

from ..quality import QualityBit


def _validate_bit_mask(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer bit mask or None")
    if value < 0 or value > (1 << 63) - 1:
        raise ValueError(f"{name} must fit a non-negative signed 64-bit bit mask")
    return int(value)


@dataclass(frozen=True)
class DQPolicy:
    """Declare detector DQ rejection, propagation, and cosmic-ray semantics."""

    reject_bits: int | None = None
    propagate_bits: int | None = None
    cosmic_ray_bit: int = int(QualityBit.COSMIC_RAY)
    unknown_bit_policy: Literal["reject", "preserve", "ignore"] = "reject"

    def __post_init__(self) -> None:
        reject = _validate_bit_mask(self.reject_bits, "reject_bits")
        propagate = _validate_bit_mask(self.propagate_bits, "propagate_bits")
        cosmic = _validate_bit_mask(self.cosmic_ray_bit, "cosmic_ray_bit")
        assert cosmic is not None
        if cosmic == 0 or cosmic & (cosmic - 1):
            raise ValueError("cosmic_ray_bit must contain exactly one set bit")
        if self.unknown_bit_policy not in {"reject", "preserve", "ignore"}:
            raise ValueError("unknown_bit_policy must be reject, preserve, or ignore")
        object.__setattr__(self, "reject_bits", reject)
        object.__setattr__(self, "propagate_bits", propagate)
        object.__setattr__(self, "cosmic_ray_bit", cosmic)

    def usability_mask(self, dq: Tensor) -> Tensor:
        value = self._coerce(dq)
        if self.reject_bits is None:
            usable = value == 0
        else:
            usable = (value & self.reject_bits) == 0
        if self.unknown_bit_policy == "reject" and self.propagate_bits is not None:
            known = self.propagate_bits | (self.reject_bits or 0) | self.cosmic_ray_bit
            usable = usable & ((value & ~known) == 0)
        return usable

    def propagated(self, dq: Tensor) -> Tensor:
        value = self._coerce(dq)
        if self.propagate_bits is None or self.unknown_bit_policy == "preserve":
            return value
        return value & (self.propagate_bits | self.cosmic_ray_bit)

    def with_cosmic_rays(self, dq: Tensor, cosmic_rays: Tensor) -> Tensor:
        value = self._coerce(dq).clone()
        mask = torch.as_tensor(cosmic_rays, device=value.device, dtype=torch.bool)
        if mask.shape != value.shape:
            raise ValueError("cosmic-ray mask must match the DQ plane shape")
        return torch.where(mask, value | self.cosmic_ray_bit, value)

    @staticmethod
    def _coerce(dq: Tensor) -> Tensor:
        value = torch.as_tensor(dq)
        if (
            value.dtype.is_floating_point
            or value.dtype.is_complex
            or value.dtype == torch.bool
        ):
            raise TypeError("DQ planes must use a non-boolean integer dtype")
        return value.to(dtype=torch.int64)


@dataclass(frozen=True)
class DQRejectionLog:
    observation_index: int
    primary_count: int
    grown_count: int
    rejected_count: int
    cosmic_ray_bit: int

    def __post_init__(self) -> None:
        if self.observation_index < 0:
            raise ValueError("observation_index must be non-negative")
        if min(self.primary_count, self.grown_count, self.rejected_count) < 0:
            raise ValueError("rejection counts must be non-negative")
        if self.primary_count > self.rejected_count:
            raise ValueError("primary_count cannot exceed rejected_count")
        if self.grown_count != self.rejected_count - self.primary_count:
            raise ValueError("grown_count must count non-primary rejected pixels")

    def to_dict(self) -> dict[str, int]:
        return {
            "observation_index": int(self.observation_index),
            "primary_count": int(self.primary_count),
            "grown_count": int(self.grown_count),
            "rejected_count": int(self.rejected_count),
            "cosmic_ray_bit": int(self.cosmic_ray_bit),
        }


@dataclass(frozen=True)
class QualityProduct:
    dq: Tensor
    usability_mask: Tensor
    rejection_log: tuple[Mapping[str, Any], ...] = ()
    policy: DQPolicy | None = None

    def __post_init__(self) -> None:
        dq = DQPolicy._coerce(self.dq)
        usable = torch.as_tensor(
            self.usability_mask,
            device=dq.device,
            dtype=torch.bool,
        )
        if dq.shape != usable.shape or dq.ndim not in {2, 3}:
            raise ValueError(
                "DQ and usability planes must be matching 2D or 3D tensors"
            )
        object.__setattr__(self, "dq", dq)
        object.__setattr__(self, "usability_mask", usable)
        object.__setattr__(
            self,
            "rejection_log",
            tuple(dict(entry) for entry in self.rejection_log),
        )


__all__ = ["DQPolicy", "DQRejectionLog", "QualityProduct"]
