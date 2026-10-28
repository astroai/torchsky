"""Capability-specific matrix-free covariance and precision adapters."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import math
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

from ..evidence import EvidenceSnapshot


@runtime_checkable
class CovarianceAction(Protocol):
    def apply_covariance(self, value: Tensor) -> Tensor: ...


@runtime_checkable
class PrecisionAction(Protocol):
    def apply_precision(self, value: Tensor) -> Tensor: ...


@runtime_checkable
class WhiteningAction(Protocol):
    def whiten(self, value: Tensor) -> Tensor: ...


@runtime_checkable
class SamplingAction(Protocol):
    def sample(
        self,
        sample_shape: tuple[int, ...] = (),
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor: ...


@runtime_checkable
class LogDetAction(Protocol):
    def logdet(self) -> Tensor: ...


@runtime_checkable
class DiagonalAction(Protocol):
    def diagonal(self) -> Tensor: ...


def _validate_real_floating(tensor: Tensor, *, name: str, ndim: int | None = None) -> None:
    if not isinstance(tensor, Tensor) or not tensor.is_floating_point() or tensor.is_complex():
        raise TypeError(f"{name} must be a real floating tensor")
    if ndim is not None and tensor.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must contain only finite values")


@dataclass(frozen=True, slots=True)
class DiagonalCovariance:
    variance: Tensor = field(repr=False, compare=False)
    _snapshot: EvidenceSnapshot = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _validate_real_floating(self.variance, name="variance")
        if bool((self.variance < 0).any()):
            raise ValueError("variance must be non-negative")
        object.__setattr__(
            self,
            "_snapshot",
            EvidenceSnapshot.capture({"variance": self.variance}),
        )

    def _value(self, value: Tensor) -> Tensor:
        self._snapshot.validate()
        if value.shape != self.variance.shape or value.dtype != self.variance.dtype or value.device != self.variance.device:
            raise ValueError("value must match variance shape, dtype, and device")
        return value

    def _positive(self) -> None:
        if bool((self.variance <= 0).any()):
            raise ValueError("this capability requires strictly positive variance")

    def apply_covariance(self, value: Tensor) -> Tensor:
        return self._value(value) * self.variance

    def apply_precision(self, value: Tensor) -> Tensor:
        self._positive()
        return self._value(value) / self.variance

    def whiten(self, value: Tensor) -> Tensor:
        self._positive()
        return self._value(value) / torch.sqrt(self.variance)

    def sample(
        self,
        sample_shape: tuple[int, ...] = (),
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        self._snapshot.validate()
        shape = tuple(int(size) for size in sample_shape)
        if any(size < 0 for size in shape):
            raise ValueError("sample dimensions must be non-negative")
        noise = torch.randn(
            (*shape, *self.variance.shape),
            dtype=self.variance.dtype,
            device=self.variance.device,
            generator=generator,
        )
        return noise * torch.sqrt(self.variance)

    def logdet(self) -> Tensor:
        self._snapshot.validate()
        self._positive()
        return torch.log(self.variance).sum()

    def diagonal(self) -> Tensor:
        self._snapshot.validate()
        return self.variance

    def query_variance(self, query: Tensor) -> Tensor:
        value = self._value(query)
        return torch.sum(value.square() * self.variance)


@dataclass(frozen=True, slots=True)
class BlockDiagonalCovariance:
    blocks: tuple[CovarianceAction, ...]
    block_shapes: tuple[tuple[int, ...], ...]

    def __init__(
        self,
        blocks: Sequence[CovarianceAction],
        block_shapes: Sequence[tuple[int, ...]] | None = None,
    ) -> None:
        items = tuple(blocks)
        if not items:
            raise ValueError("block covariance requires at least one block")
        if block_shapes is None:
            inferred = []
            for block in items:
                if not isinstance(block, DiagonalAction):
                    raise ValueError("block_shapes are required when a block has no diagonal")
                inferred.append(tuple(block.diagonal().shape))
            shapes = tuple(inferred)
        else:
            shapes = tuple(tuple(int(size) for size in shape) for shape in block_shapes)
        if len(shapes) != len(items) or any(any(size < 0 for size in shape) for shape in shapes):
            raise ValueError("block shapes must be non-negative and align with blocks")
        object.__setattr__(self, "blocks", items)
        object.__setattr__(self, "block_shapes", shapes)

    @property
    def size(self) -> int:
        return sum(math.prod(shape) for shape in self.block_shapes)

    def _parts(self, value: Tensor) -> list[Tensor]:
        if value.ndim != 1 or value.numel() != self.size:
            raise ValueError("block covariance values must be flat with the declared size")
        parts = []
        offset = 0
        for shape in self.block_shapes:
            count = math.prod(shape)
            parts.append(value[offset : offset + count].reshape(shape))
            offset += count
        return parts

    def _map(self, capability: type, method: str, value: Tensor) -> Tensor:
        parts = self._parts(value)
        outputs = []
        for block, part in zip(self.blocks, parts, strict=True):
            if not isinstance(block, capability):
                raise NotImplementedError(f"block does not provide {method}")
            outputs.append(getattr(block, method)(part).reshape(-1))
        return torch.cat(outputs)

    def apply_covariance(self, value: Tensor) -> Tensor:
        return self._map(CovarianceAction, "apply_covariance", value)

    def apply_precision(self, value: Tensor) -> Tensor:
        return self._map(PrecisionAction, "apply_precision", value)

    def whiten(self, value: Tensor) -> Tensor:
        return self._map(WhiteningAction, "whiten", value)

    def sample(
        self,
        sample_shape: tuple[int, ...] = (),
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        outputs = []
        for block in self.blocks:
            if not isinstance(block, SamplingAction):
                raise NotImplementedError("block does not provide sampling")
            outputs.append(block.sample(sample_shape, generator=generator).flatten(start_dim=len(sample_shape)))
        return torch.cat(outputs, dim=-1)

    def logdet(self) -> Tensor:
        values = []
        for block in self.blocks:
            if not isinstance(block, LogDetAction):
                raise NotImplementedError("block does not provide logdet")
            values.append(block.logdet())
        return torch.stack(values).sum()

    def diagonal(self) -> Tensor:
        values = []
        for block in self.blocks:
            if not isinstance(block, DiagonalAction):
                raise NotImplementedError("block does not provide a diagonal")
            values.append(block.diagonal().reshape(-1))
        return torch.cat(values)

    def query_variance(self, query: Tensor) -> Tensor:
        return torch.sum(query * self.apply_covariance(query))


@dataclass(frozen=True, slots=True)
class LowRankCovariance:
    """Covariance ``diag(variance) + factor @ factor.T`` without dense storage."""

    variance: Tensor = field(repr=False, compare=False)
    factor: Tensor = field(repr=False, compare=False)
    _snapshot: EvidenceSnapshot = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _validate_real_floating(self.variance, name="variance")
        _validate_real_floating(self.factor, name="factor")
        if self.factor.shape[:-1] != self.variance.shape:
            raise ValueError("factor must have shape (*variance.shape, rank)")
        if self.factor.dtype != self.variance.dtype or self.factor.device != self.variance.device:
            raise ValueError("factor and variance must share dtype and device")
        if bool((self.variance <= 0).any()):
            raise ValueError("low-rank covariance requires positive diagonal variance")
        object.__setattr__(
            self,
            "_snapshot",
            EvidenceSnapshot.capture({"variance": self.variance, "factor": self.factor}),
        )

    @property
    def rank(self) -> int:
        return int(self.factor.shape[-1])

    def _value(self, value: Tensor) -> Tensor:
        self._snapshot.validate()
        if value.shape != self.variance.shape or value.dtype != self.variance.dtype or value.device != self.variance.device:
            raise ValueError("value must match covariance shape, dtype, and device")
        return value

    def _flat_factor(self) -> Tensor:
        return self.factor.reshape(self.variance.numel(), self.rank)

    def apply_covariance(self, value: Tensor) -> Tensor:
        safe = self._value(value)
        flat = safe.reshape(-1)
        factor = self._flat_factor()
        return (self.variance.reshape(-1) * flat + factor @ (factor.T @ flat)).reshape(self.variance.shape)

    def apply_precision(self, value: Tensor) -> Tensor:
        safe = self._value(value)
        diagonal = self.variance.reshape(-1)
        factor = self._flat_factor()
        weighted = safe.reshape(-1) / diagonal
        if self.rank == 0:
            return weighted.reshape(self.variance.shape)
        weighted_factor = factor / diagonal[:, None]
        system = torch.eye(self.rank, dtype=factor.dtype, device=factor.device) + factor.T @ weighted_factor
        correction = weighted_factor @ torch.linalg.solve(system, factor.T @ weighted)
        return (weighted - correction).reshape(self.variance.shape)

    def sample(
        self,
        sample_shape: tuple[int, ...] = (),
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        self._snapshot.validate()
        shape = tuple(int(size) for size in sample_shape)
        diagonal_noise = torch.randn(
            (*shape, *self.variance.shape),
            dtype=self.variance.dtype,
            device=self.variance.device,
            generator=generator,
        ) * torch.sqrt(self.variance)
        if self.rank == 0:
            return diagonal_noise
        low_rank_noise = torch.randn(
            (*shape, self.rank),
            dtype=self.variance.dtype,
            device=self.variance.device,
            generator=generator,
        )
        low_rank = low_rank_noise @ self._flat_factor().T
        return diagonal_noise + low_rank.reshape(*shape, *self.variance.shape)

    def logdet(self) -> Tensor:
        self._snapshot.validate()
        diagonal_logdet = torch.log(self.variance).sum()
        if self.rank == 0:
            return diagonal_logdet
        factor = self._flat_factor()
        system = torch.eye(self.rank, dtype=factor.dtype, device=factor.device) + factor.T @ (factor / self.variance.reshape(-1, 1))
        sign, logabsdet = torch.linalg.slogdet(system)
        if bool(sign <= 0):
            raise RuntimeError("low-rank determinant lemma produced a non-positive system")
        return diagonal_logdet + logabsdet

    def diagonal(self) -> Tensor:
        self._snapshot.validate()
        return self.variance + self.factor.square().sum(dim=-1)

    def query_variance(self, query: Tensor) -> Tensor:
        safe = self._value(query)
        flat = safe.reshape(-1)
        projected = self._flat_factor().T @ flat
        return torch.sum(flat.square() * self.variance.reshape(-1)) + torch.sum(projected.square())


@dataclass(frozen=True, slots=True)
class NuisancePrecisionAdapter:
    """Precision-only adapter for profiled or calibrated nuisance actions."""

    action: Callable[[Tensor], Tensor] = field(repr=False, compare=False)
    logdet_correction: Tensor | None = None
    diagonal_action: Callable[[], Tensor] | None = field(default=None, repr=False, compare=False)

    @classmethod
    def from_nuisance(cls, nuisance: object) -> NuisancePrecisionAdapter:
        action = getattr(nuisance, "apply", None)
        if not callable(action):
            raise TypeError("nuisance object must expose apply")
        return cls(action, getattr(nuisance, "logdet_covariance_correction", None))

    def apply_precision(self, value: Tensor) -> Tensor:
        result = self.action(value)
        if not isinstance(result, Tensor) or result.shape != value.shape:
            raise ValueError("nuisance precision output must match its input shape")
        return result

    def logdet(self) -> Tensor:
        if self.logdet_correction is None:
            raise NotImplementedError("nuisance logdet correction is not available")
        return self.logdet_correction

    def diagonal(self) -> Tensor:
        if self.diagonal_action is None:
            raise NotImplementedError("nuisance precision diagonal is not available")
        return self.diagonal_action()


NuisanceCovariance = LowRankCovariance


__all__ = [
    "BlockDiagonalCovariance",
    "CovarianceAction",
    "DiagonalAction",
    "DiagonalCovariance",
    "LogDetAction",
    "LowRankCovariance",
    "NuisanceCovariance",
    "NuisancePrecisionAdapter",
    "PrecisionAction",
    "SamplingAction",
    "WhiteningAction",
]
