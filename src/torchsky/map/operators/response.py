"""Deterministic linear response plans and sparse resampling adapters."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
import math
from typing import Mapping, Protocol, runtime_checkable

import torch
from torch import Tensor

from ...device import _scatter_reduce_int
from ...domain import DomainSpec
from ..evidence import EvidenceSnapshot
from ..provenance import PlanFingerprint, plan_fingerprint
from ..quality import DQPolicy


@runtime_checkable
class ResponsePlan(Protocol):
    @property
    def source(self) -> DomainSpec: ...

    @property
    def target(self) -> DomainSpec: ...

    @property
    def fingerprint(self) -> PlanFingerprint: ...

    def apply(self, value: Tensor, *, chunk_size: int | None = None) -> Tensor: ...

    def adjoint(self, value: Tensor, *, chunk_size: int | None = None) -> Tensor: ...

    def apply_squared(
        self, value: Tensor, *, chunk_size: int | None = None
    ) -> Tensor: ...

    def apply_flags(
        self, flags: Tensor, *, chunk_size: int | None = None
    ) -> Tensor: ...

    def contributor_count(self, *, chunk_size: int | None = None) -> Tensor: ...

    def context(self, *, chunk_size: int | None = None) -> Tensor: ...


@dataclass(frozen=True)
class ResponseProducts:
    science: Tensor
    variance_components: Mapping[str, Tensor]
    dq: Tensor | None
    context: Tensor
    nexp: Tensor


def apply_response_products(
    plan: ResponsePlan,
    science: Tensor,
    *,
    variance_components: Mapping[str, Tensor] | None = None,
    dq: Tensor | None = None,
    dq_policy: DQPolicy | None = None,
    chunk_size: int | None = None,
) -> ResponseProducts:
    """Apply one immutable coefficient plan to every linear map product."""
    propagated_dq = None
    if dq is not None:
        policy = dq_policy or DQPolicy()
        propagated_dq = plan.apply_flags(
            policy.propagated(dq),
            chunk_size=chunk_size,
        )
    components = {
        name: plan.apply_squared(value, chunk_size=chunk_size)
        for name, value in (variance_components or {}).items()
    }
    return ResponseProducts(
        science=plan.apply(science, chunk_size=chunk_size),
        variance_components=components,
        dq=propagated_dq,
        context=plan.context(chunk_size=chunk_size),
        nexp=plan.contributor_count(chunk_size=chunk_size),
    )


def _occupied_bits(values: Tensor, width: int) -> list[int]:
    """Bit positions set in any element (on-device reductions, one transfer)."""
    present = torch.stack([((values >> b) & 1).amax() for b in range(width)])
    return [b for b, flag in enumerate(present.tolist()) if flag]


def _from_bit_pattern(packed: Tensor, dtype: torch.dtype) -> Tensor:
    """Reinterpret an unsigned bit pattern held in int64 as ``dtype``."""
    bits = torch.iinfo(dtype).bits
    if dtype == torch.uint8 or bits == 64:
        return packed.to(dtype)
    half = 1 << (bits - 1)
    return torch.where(packed >= half, packed - (1 << bits), packed).to(dtype)


def _chunk_ranges(size: int, chunk_size: int | None) -> Iterator[tuple[int, int]]:
    chunk = size if chunk_size is None else int(chunk_size)
    if chunk <= 0:
        raise ValueError("chunk_size must be positive")
    if size == 0:
        return
    for start in range(0, size, chunk):
        yield start, min(start + chunk, size)


@dataclass(frozen=True, slots=True)
class IndexedResponsePlan:
    """Sparse response in deterministic COO order, reduced by target chunks."""

    source: DomainSpec
    target: DomainSpec
    source_indices: Tensor = field(repr=False, compare=False)
    target_indices: Tensor = field(repr=False, compare=False)
    coefficients: Tensor = field(repr=False, compare=False)
    contributor_ids: Tensor | None = field(default=None, repr=False, compare=False)
    default_chunk_size: int = 65_536
    fingerprint: PlanFingerprint = field(init=False)
    _snapshot: EvidenceSnapshot = field(init=False, repr=False, compare=False)
    _order: Tensor = field(init=False, repr=False, compare=False)
    _bounds: dict[int, list[int]] = field(init=False, repr=False, compare=False)
    _pair_order: list[Tensor] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name, indices, limit in (
            ("source_indices", self.source_indices, self.source.numel),
            ("target_indices", self.target_indices, self.target.numel),
        ):
            if (
                not isinstance(indices, Tensor)
                or indices.dtype != torch.int64
                or indices.ndim != 1
            ):
                raise TypeError(f"{name} must be a one-dimensional int64 tensor")
            if bool(((indices < 0) | (indices >= limit)).any()):
                raise ValueError(f"{name} contains an out-of-domain index")
        if (
            not isinstance(self.coefficients, Tensor)
            or self.coefficients.ndim != 1
            or not self.coefficients.is_floating_point()
        ):
            raise TypeError("coefficients must be a one-dimensional floating tensor")
        size = self.coefficients.numel()
        if self.source_indices.numel() != size or self.target_indices.numel() != size:
            raise ValueError("response COO arrays must have matching lengths")
        if (
            self.coefficients.device != self.source_indices.device
            or self.coefficients.device != self.target_indices.device
        ):
            raise ValueError("response COO arrays must share a device")
        if not bool(torch.isfinite(self.coefficients).all()):
            raise ValueError("response coefficients must be finite")
        contributor_ids = (
            self.source_indices
            if self.contributor_ids is None
            else self.contributor_ids
        )
        if (
            contributor_ids.dtype != torch.int64
            or contributor_ids.shape != self.source_indices.shape
        ):
            raise TypeError("contributor_ids must be int64 and match response entries")
        if contributor_ids.device != self.coefficients.device or bool(
            (contributor_ids < 0).any()
        ):
            raise ValueError(
                "contributor_ids must share the response device and be non-negative"
            )
        chunk = int(self.default_chunk_size)
        if chunk <= 0:
            raise ValueError("default_chunk_size must be positive")
        object.__setattr__(self, "default_chunk_size", chunk)
        object.__setattr__(self, "contributor_ids", contributor_ids)
        # Group entries by target once, O(E log E). A stable sort keeps the
        # original entry order within a target, so per-target reductions add in
        # the same order as the COO order.
        object.__setattr__(
            self, "_order", torch.argsort(self.target_indices, stable=True)
        )
        object.__setattr__(self, "_bounds", {})
        object.__setattr__(self, "_pair_order", [])
        object.__setattr__(
            self,
            "_snapshot",
            EvidenceSnapshot.capture(
                {
                    "source_indices": self.source_indices,
                    "target_indices": self.target_indices,
                    "coefficients": self.coefficients,
                    "contributor_ids": contributor_ids,
                }
            ),
        )
        object.__setattr__(
            self,
            "fingerprint",
            plan_fingerprint(
                "indexed-response",
                {
                    "source": self.source,
                    "target": self.target,
                    "entries": size,
                    "contributors": 0
                    if size == 0
                    else int(contributor_ids.max().item()) + 1,
                    "chunk": chunk,
                },
            ),
        )

    def _chunks(self, chunk_size: int | None) -> Iterator[tuple[int, int]]:
        return _chunk_ranges(
            self.target.numel,
            self.default_chunk_size if chunk_size is None else chunk_size,
        )

    def _entry_slices(self, chunk_size: int | None) -> Iterator[tuple[int, int, int]]:
        """Yield ``(target_start, entry_lo, entry_hi)`` per target chunk.

        Entries ``_order[entry_lo:entry_hi]`` are exactly those whose target lies
        in the chunk. The boundaries come from one ``searchsorted`` over the
        sorted targets (a single host transfer per chunk size, cached).
        """
        size = int(self.default_chunk_size if chunk_size is None else chunk_size)
        chunks = list(self._chunks(size))
        if not chunks:
            return
        bounds = self._bounds.get(size)
        if bounds is None:
            sorted_targets = self.target_indices[self._order]
            edges = torch.tensor(
                [c[0] for c in chunks] + [chunks[-1][1]],
                dtype=torch.int64,
                device=sorted_targets.device,
            )
            bounds = torch.searchsorted(sorted_targets, edges).tolist()
            self._bounds[size] = bounds
        for k, (start, _stop) in enumerate(chunks):
            yield start, bounds[k], bounds[k + 1]

    def _validate_source(self, value: Tensor, name: str) -> None:
        self._snapshot.validate()
        self.source.validate(value, name=name)
        if value.device != self.coefficients.device:
            raise ValueError(f"{name} must share the response device")

    def _validate_target(self, value: Tensor, name: str) -> None:
        self._snapshot.validate()
        self.target.validate(value, name=name)
        if value.device != self.coefficients.device:
            raise ValueError(f"{name} must share the response device")

    def apply(self, value: Tensor, *, chunk_size: int | None = None) -> Tensor:
        self._validate_source(value, "response input")
        source = value.reshape(-1)
        output = value.new_zeros(self.target.numel)
        for start, lo, hi in self._entry_slices(chunk_size):
            entries = self._order[lo:hi]
            stop = min(start + self._chunk_len(chunk_size), self.target.numel)
            local = output.new_zeros(stop - start)
            local.scatter_add_(
                0,
                self.target_indices[entries] - start,
                source[self.source_indices[entries]]
                * self.coefficients[entries].to(dtype=value.dtype),
            )
            output[start:stop] = local
        return output.reshape(self.target.shape)

    def _chunk_len(self, chunk_size: int | None) -> int:
        return int(self.default_chunk_size if chunk_size is None else chunk_size)

    def adjoint(self, value: Tensor, *, chunk_size: int | None = None) -> Tensor:
        self._validate_target(value, "response adjoint input")
        target = value.reshape(-1)
        output = value.new_zeros(self.source.numel)
        for _start, lo, hi in self._entry_slices(chunk_size):
            entries = self._order[lo:hi]
            output.scatter_add_(
                0,
                self.source_indices[entries],
                target[self.target_indices[entries]]
                * self.coefficients[entries].to(dtype=value.dtype),
            )
        return output.reshape(self.source.shape)

    def apply_squared(self, value: Tensor, *, chunk_size: int | None = None) -> Tensor:
        self._validate_source(value, "squared-response input")
        source = value.reshape(-1)
        output = value.new_zeros(self.target.numel)
        for start, lo, hi in self._entry_slices(chunk_size):
            entries = self._order[lo:hi]
            stop = min(start + self._chunk_len(chunk_size), self.target.numel)
            local = output.new_zeros(stop - start)
            coefficient = self.coefficients[entries].to(dtype=value.dtype)
            local.scatter_add_(
                0,
                self.target_indices[entries] - start,
                source[self.source_indices[entries]] * coefficient.square(),
            )
            output[start:stop] = local
        return output.reshape(self.target.shape)

    def apply_flags(self, flags: Tensor, *, chunk_size: int | None = None) -> Tensor:
        self._snapshot.validate()
        if flags.shape != self.source.shape or flags.device != self.coefficients.device:
            raise ValueError("flags must match the source shape and response device")
        if flags.dtype not in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise TypeError("flags must have an integer dtype")
        source = flags.reshape(-1)
        output = flags.new_zeros(self.target.numel)
        if source.numel() == 0:
            return output.reshape(self.target.shape)
        # Bitwise OR has no scatter reduction: reduce each occupied bit with
        # amax, one vectorised pass per bit. The set of occupied bits is the
        # OR of all flags, found with one reduction (one host transfer).
        width = torch.iinfo(flags.dtype).bits
        wide = torch.int64
        values = source.to(wide)
        if flags.dtype != torch.uint8:
            # Reinterpret two's complement as the unsigned bit pattern.
            values = values & ((1 << width) - 1) if width < 64 else values
        occupied = _occupied_bits(values, width)
        for start, lo, hi in self._entry_slices(chunk_size):
            entries = self._order[lo:hi]
            stop = min(start + self._chunk_len(chunk_size), self.target.numel)
            gathered = values[self.source_indices[entries]]
            local_target = self.target_indices[entries] - start
            packed = torch.zeros(stop - start, dtype=wide, device=flags.device)
            for bit in occupied:
                plane = (gathered >> bit) & 1
                bit_plane = _scatter_reduce_int(
                    torch.zeros_like(packed),
                    0,
                    local_target,
                    plane,
                    "amax",
                    include_self=True,
                )
                packed |= bit_plane << bit
            output[start:stop] = _from_bit_pattern(packed, flags.dtype)
        return output.reshape(self.target.shape)

    def contributor_count(self, *, chunk_size: int | None = None) -> Tensor:
        self._snapshot.validate()
        output = torch.zeros(
            self.target.numel,
            dtype=torch.int64,
            device=self.coefficients.device,
        )
        assert self.contributor_ids is not None
        if self._order.numel() == 0:
            return output.reshape(self.target.shape)
        if not self._pair_order:
            # Lexicographic (target, contributor) order from two stable sorts.
            by_contributor = torch.argsort(self.contributor_ids, stable=True)
            by_target = torch.argsort(self.target_indices[by_contributor], stable=True)
            self._pair_order.append(by_contributor[by_target])
        order = self._pair_order[0]
        targets = self.target_indices[order]
        contributors = self.contributor_ids[order]
        first = torch.ones_like(targets, dtype=torch.bool)
        first[1:] = (targets[1:] != targets[:-1]) | (
            contributors[1:] != contributors[:-1]
        )
        output.scatter_add_(0, targets[first], torch.ones_like(targets[first]))
        return output.reshape(self.target.shape)

    def context(self, *, chunk_size: int | None = None) -> Tensor:
        """Per-target contributor bitmask planes (32 contributors per plane)."""
        self._snapshot.validate()
        assert self.contributor_ids is not None
        contributor_count = (
            0
            if self.contributor_ids.numel() == 0
            else int(self.contributor_ids.max().item()) + 1
        )
        planes = math.ceil(contributor_count / 32)
        count = self.target.numel
        output = torch.zeros(
            (planes, count),
            dtype=torch.int64,
            device=self.coefficients.device,
        )
        if planes == 0 or count == 0:
            return output.reshape(planes, *self.target.shape)
        flat = output.reshape(-1)
        plane = self.contributor_ids >> 5
        bit = self.contributor_ids & 31
        slot = plane * count + self.target_indices
        # Setting a bit is idempotent, so duplicate (target, contributor)
        # entries need no deduplication: one vectorised pass per bit position.
        for b in range(32):
            marked = torch.zeros_like(flat)
            marked[slot[bit == b]] = 1
            flat |= marked << b
        return output.reshape(planes, *self.target.shape)


class DirectCopyResponsePlan(IndexedResponsePlan):
    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str = "cpu",
        default_chunk_size: int = 65_536,
    ) -> None:
        domain = DomainSpec(tuple(shape), dtype)
        indices = torch.arange(domain.numel, dtype=torch.int64, device=device)
        coefficients = torch.ones(
            domain.numel,
            dtype=dtype or torch.float32,
            device=device,
        )
        super().__init__(
            domain,
            domain,
            indices,
            indices.clone(),
            coefficients,
            default_chunk_size=default_chunk_size,
        )


class BilinearResponsePlan(IndexedResponsePlan):
    def __init__(
        self,
        source_shape: tuple[int, int],
        x_source: Tensor,
        y_source: Tensor,
        *,
        default_chunk_size: int = 65_536,
    ) -> None:
        if x_source.shape != y_source.shape or x_source.ndim < 1:
            raise ValueError(
                "bilinear coordinates must have matching non-scalar shapes"
            )
        if (
            x_source.device != y_source.device
            or not x_source.is_floating_point()
            or not y_source.is_floating_point()
        ):
            raise TypeError(
                "bilinear coordinates must be floating tensors on one device"
            )
        height, width = int(source_shape[0]), int(source_shape[1])
        if height < 0 or width < 0:
            raise ValueError("source dimensions must be non-negative")
        x = x_source.reshape(-1)
        y = y_source.reshape(-1)
        finite = torch.isfinite(x) & torch.isfinite(y)
        x0 = torch.floor(torch.where(finite, x, torch.zeros_like(x))).to(torch.int64)
        y0 = torch.floor(torch.where(finite, y, torch.zeros_like(y))).to(torch.int64)
        source_entries = []
        target_entries = []
        coefficient_entries = []
        target_base = torch.arange(x.numel(), dtype=torch.int64, device=x.device)
        for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1)):
            xi = x0 + dx
            yi = y0 + dy
            valid = finite & (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
            coefficient = (1.0 - torch.abs(x - xi.to(x.dtype))) * (
                1.0 - torch.abs(y - yi.to(y.dtype))
            )
            valid = valid & (coefficient != 0)
            source_entries.append((yi[valid] * width + xi[valid]).to(torch.int64))
            target_entries.append(target_base[valid])
            coefficient_entries.append(coefficient[valid])
        super().__init__(
            DomainSpec((height, width), x.dtype),
            DomainSpec(tuple(x_source.shape), x.dtype),
            torch.cat(source_entries),
            torch.cat(target_entries),
            torch.cat(coefficient_entries),
            default_chunk_size=default_chunk_size,
        )


def direct_copy_response(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str = "cpu",
    default_chunk_size: int = 65_536,
) -> DirectCopyResponsePlan:
    return DirectCopyResponsePlan(
        shape,
        dtype=dtype,
        device=device,
        default_chunk_size=default_chunk_size,
    )


def bilinear_response(
    source_shape: tuple[int, int],
    x_source: Tensor,
    y_source: Tensor,
    *,
    default_chunk_size: int = 65_536,
) -> BilinearResponsePlan:
    return BilinearResponsePlan(
        source_shape,
        x_source,
        y_source,
        default_chunk_size=default_chunk_size,
    )


ResponseAction = Callable[[Tensor, int | None], Tensor]


@dataclass(frozen=True, slots=True)
class DrizzleResponsePlan:
    """Adapter for existing drizzle kernels that expose all linear actions."""

    source: DomainSpec
    target: DomainSpec
    apply_action: ResponseAction = field(repr=False, compare=False)
    adjoint_action: ResponseAction = field(repr=False, compare=False)
    squared_action: ResponseAction = field(repr=False, compare=False)
    key: str
    flags_action: ResponseAction | None = field(default=None, repr=False, compare=False)
    contributor_action: Callable[[int | None], Tensor] | None = field(
        default=None, repr=False, compare=False
    )
    context_action: Callable[[int | None], Tensor] | None = field(
        default=None, repr=False, compare=False
    )
    fingerprint: PlanFingerprint = field(init=False)

    def __post_init__(self) -> None:
        if not all(
            callable(action)
            for action in (self.apply_action, self.adjoint_action, self.squared_action)
        ):
            raise TypeError("drizzle response actions must be callable")
        if not self.key:
            raise ValueError("drizzle response key must be nonempty")
        object.__setattr__(
            self,
            "fingerprint",
            plan_fingerprint(
                "drizzle-response",
                {"source": self.source, "target": self.target, "key": self.key},
            ),
        )

    def apply(self, value: Tensor, *, chunk_size: int | None = None) -> Tensor:
        self.source.validate(value, name="drizzle input")
        return self.target.validate(
            self.apply_action(value, chunk_size), name="drizzle output"
        )

    def adjoint(self, value: Tensor, *, chunk_size: int | None = None) -> Tensor:
        self.target.validate(value, name="drizzle adjoint input")
        return self.source.validate(
            self.adjoint_action(value, chunk_size), name="drizzle adjoint output"
        )

    def apply_squared(self, value: Tensor, *, chunk_size: int | None = None) -> Tensor:
        self.source.validate(value, name="drizzle squared input")
        return self.target.validate(
            self.squared_action(value, chunk_size), name="drizzle squared output"
        )

    def apply_flags(self, flags: Tensor, *, chunk_size: int | None = None) -> Tensor:
        if self.flags_action is None:
            raise NotImplementedError("drizzle flag propagation is not configured")
        return self.flags_action(flags, chunk_size)

    def contributor_count(self, *, chunk_size: int | None = None) -> Tensor:
        if self.contributor_action is None:
            raise NotImplementedError("drizzle contributor counts are not configured")
        return self.contributor_action(chunk_size)

    def context(self, *, chunk_size: int | None = None) -> Tensor:
        if self.context_action is None:
            raise NotImplementedError("drizzle context is not configured")
        return self.context_action(chunk_size)


__all__ = [
    "BilinearResponsePlan",
    "DirectCopyResponsePlan",
    "DrizzleResponsePlan",
    "IndexedResponsePlan",
    "ResponsePlan",
    "ResponseProducts",
    "apply_response_products",
    "bilinear_response",
    "direct_copy_response",
]
