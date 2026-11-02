"""Internal observation-operator interfaces for mapmaking and field rendering."""

from __future__ import annotations

from abc import ABC
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

from ...domain import DomainSpec, as_domain_spec as as_domain_spec
from ..evidence import EvidenceSnapshot
from ..provenance import PlanFingerprint, plan_fingerprint


@runtime_checkable
class LinearOperator(Protocol):
    """Linear tensor map with an explicitly implemented exact adjoint."""

    def apply(self, value: Tensor) -> Tensor: ...

    def adjoint(self, value: Tensor) -> Tensor: ...


@runtime_checkable
class DifferentiableMapping(Protocol):
    """Possibly nonlinear map exposing point-dependent JVP and VJP actions."""

    def apply(self, value: Tensor) -> Tensor: ...

    def jvp(self, value: Tensor, tangent: Tensor) -> Tensor: ...

    def vjp(self, value: Tensor, cotangent: Tensor) -> Tensor: ...


TensorAction = Callable[[Tensor], Tensor]


def _operator_fingerprint(operator: object) -> PlanFingerprint:
    fingerprint = getattr(operator, "fingerprint", None)
    if not isinstance(fingerprint, PlanFingerprint):
        raise TypeError("composed operators must expose a PlanFingerprint")
    return fingerprint


@dataclass(frozen=True, slots=True)
class CallableLinearOperator:
    domain: DomainSpec
    codomain: DomainSpec
    apply_fn: TensorAction = field(repr=False, compare=False)
    adjoint_fn: TensorAction = field(repr=False, compare=False)
    key: str
    fingerprint: PlanFingerprint = field(init=False)

    def __post_init__(self) -> None:
        if not callable(self.apply_fn) or not callable(self.adjoint_fn):
            raise TypeError("linear operator actions must be callable")
        if not isinstance(self.key, str) or not self.key.strip():
            raise ValueError("linear operator key must be a nonempty string")
        object.__setattr__(
            self,
            "fingerprint",
            plan_fingerprint(
                "linear",
                {"key": self.key, "domain": self.domain, "codomain": self.codomain},
            ),
        )

    def apply(self, value: Tensor) -> Tensor:
        self.domain.validate(value, name="operator input")
        return self.codomain.validate(self.apply_fn(value), name="operator output")

    def adjoint(self, value: Tensor) -> Tensor:
        self.codomain.validate(value, name="adjoint input")
        return self.domain.validate(self.adjoint_fn(value), name="adjoint output")


@dataclass(frozen=True, slots=True)
class AutogradMapping:
    """Differentiable mapping adapter; its VJP is not declared as an adjoint."""

    mapping: TensorAction = field(repr=False, compare=False)

    def apply(self, value: Tensor) -> Tensor:
        return self.mapping(value)

    def jvp(self, value: Tensor, tangent: Tensor) -> Tensor:
        return torch.func.jvp(self.mapping, (value,), (tangent,))[1]

    def vjp(self, value: Tensor, cotangent: Tensor) -> Tensor:
        output, pullback = torch.func.vjp(self.mapping, value)[:2]
        if output.shape != cotangent.shape:
            raise ValueError("VJP cotangent shape must match the mapping output")
        return pullback(cotangent)[0]


@dataclass(frozen=True, slots=True)
class ChainPlan:
    operators: tuple[CallableLinearOperator | LinearOperator, ...]
    fingerprint: PlanFingerprint = field(init=False)

    def __init__(self, operators: Sequence[CallableLinearOperator | LinearOperator]):
        items = tuple(operators)
        if not items:
            raise ValueError("operator chain must not be empty")
        object.__setattr__(self, "operators", items)
        object.__setattr__(
            self,
            "fingerprint",
            plan_fingerprint("chain", tuple(_operator_fingerprint(op) for op in items)),
        )

    def apply(self, value: Tensor) -> Tensor:
        result = value
        for operator in self.operators:
            result = operator.apply(result)
        return result

    def adjoint(self, value: Tensor) -> Tensor:
        result = value
        for operator in reversed(self.operators):
            result = operator.adjoint(result)
        return result


@dataclass(frozen=True, slots=True)
class SumPlan:
    operators: tuple[CallableLinearOperator | LinearOperator, ...]
    fingerprint: PlanFingerprint = field(init=False)

    def __init__(self, operators: Sequence[CallableLinearOperator | LinearOperator]):
        items = tuple(operators)
        if not items:
            raise ValueError("operator sum must not be empty")
        object.__setattr__(self, "operators", items)
        object.__setattr__(
            self,
            "fingerprint",
            plan_fingerprint("sum", tuple(_operator_fingerprint(op) for op in items)),
        )

    def apply(self, value: Tensor) -> Tensor:
        outputs = tuple(operator.apply(value) for operator in self.operators)
        reference = outputs[0]
        if any(output.shape != reference.shape for output in outputs[1:]):
            raise ValueError("summed operator outputs must have matching shapes")
        return torch.stack(outputs, dim=0).sum(dim=0)

    def adjoint(self, value: Tensor) -> Tensor:
        outputs = tuple(operator.adjoint(value) for operator in self.operators)
        reference = outputs[0]
        if any(output.shape != reference.shape for output in outputs[1:]):
            raise ValueError("summed adjoint outputs must have matching shapes")
        return torch.stack(outputs, dim=0).sum(dim=0)


@dataclass(frozen=True, slots=True)
class BlockPlan:
    """Stack operators with a shared input into one flattened observation block."""

    operators: tuple[CallableLinearOperator | LinearOperator, ...]
    output_shapes: tuple[tuple[int, ...], ...]
    fingerprint: PlanFingerprint = field(init=False)

    def __init__(
        self,
        operators: Sequence[CallableLinearOperator | LinearOperator],
        output_shapes: Sequence[tuple[int, ...]],
    ):
        items = tuple(operators)
        shapes = tuple(tuple(int(size) for size in shape) for shape in output_shapes)
        if not items or len(items) != len(shapes):
            raise ValueError(
                "block operators and output shapes must be nonempty and aligned"
            )
        if any(any(size < 0 for size in shape) for shape in shapes):
            raise ValueError("block output dimensions must be non-negative")
        object.__setattr__(self, "operators", items)
        object.__setattr__(self, "output_shapes", shapes)
        object.__setattr__(
            self,
            "fingerprint",
            plan_fingerprint(
                "block",
                {
                    "operators": tuple(_operator_fingerprint(op) for op in items),
                    "output_shapes": shapes,
                },
            ),
        )

    def apply(self, value: Tensor) -> Tensor:
        outputs = []
        for operator, shape in zip(self.operators, self.output_shapes, strict=True):
            output = operator.apply(value)
            if tuple(output.shape) != shape:
                raise ValueError(
                    "block operator output does not match its declared shape"
                )
            outputs.append(output.reshape(-1))
        return torch.cat(outputs)

    def adjoint(self, value: Tensor) -> Tensor:
        if value.ndim != 1:
            raise ValueError("block adjoint input must be flat")
        outputs = []
        offset = 0
        for operator, shape in zip(self.operators, self.output_shapes, strict=True):
            count = DomainSpec(shape).numel
            outputs.append(
                operator.adjoint(value[offset : offset + count].reshape(shape))
            )
            offset += count
        if offset != value.numel():
            raise ValueError("block adjoint input has the wrong number of elements")
        return torch.stack(outputs, dim=0).sum(dim=0)


@dataclass(frozen=True, slots=True)
class RestrictionPlan:
    domain: DomainSpec
    indices: Tensor = field(repr=False, compare=False)
    fingerprint: PlanFingerprint = field(init=False)
    _snapshot: EvidenceSnapshot = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.indices.dtype != torch.int64 or self.indices.ndim != 1:
            raise TypeError(
                "restriction indices must be a one-dimensional int64 tensor"
            )
        if bool(((self.indices < 0) | (self.indices >= self.domain.numel)).any()):
            raise ValueError("restriction index is outside the domain")
        if self.indices.numel() != torch.unique(self.indices).numel():
            raise ValueError("restriction indices must be unique")
        object.__setattr__(
            self,
            "_snapshot",
            EvidenceSnapshot.capture({"restriction_indices": self.indices}),
        )
        object.__setattr__(
            self,
            "fingerprint",
            plan_fingerprint(
                "restriction",
                {"domain": self.domain, "size": self.indices.numel()},
            ),
        )

    def apply(self, value: Tensor) -> Tensor:
        self._snapshot.validate()
        self.domain.validate(value, name="restriction input")
        return value.reshape(-1)[self.indices.to(device=value.device)]

    def adjoint(self, value: Tensor) -> Tensor:
        self._snapshot.validate()
        if value.ndim != 1 or value.numel() != self.indices.numel():
            raise ValueError("restriction adjoint input has the wrong shape")
        output = value.new_zeros(self.domain.numel)
        return output.index_copy(
            0, self.indices.to(device=value.device), value
        ).reshape(self.domain.shape)


@dataclass(frozen=True, slots=True)
class MaskPlan:
    mask: Tensor = field(repr=False, compare=False)
    fingerprint: PlanFingerprint = field(init=False)
    _snapshot: EvidenceSnapshot = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.mask, Tensor) or self.mask.dtype != torch.bool:
            raise TypeError("mask must be a boolean tensor")
        object.__setattr__(
            self,
            "_snapshot",
            EvidenceSnapshot.capture({"mask": self.mask}),
        )
        object.__setattr__(
            self,
            "fingerprint",
            plan_fingerprint("mask", {"shape": tuple(self.mask.shape)}),
        )

    def apply(self, value: Tensor) -> Tensor:
        self._snapshot.validate()
        if value.shape != self.mask.shape or value.device != self.mask.device:
            raise ValueError("mask and input must have matching shape and device")
        return torch.where(self.mask, value, torch.zeros_like(value))

    def adjoint(self, value: Tensor) -> Tensor:
        return self.apply(value)


@dataclass(frozen=True)
class OperatorConditioning:
    """Declared conditioning channels for explicit or learned operators."""

    detector_coordinates: bool = False
    wavelength: bool = False
    time: bool = False
    exposure_metadata_keys: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class ObservationOperator(ABC):
    """Base class for internal operator families."""

    name: str
    learned: bool = False
    enabled: bool = True
    frozen: bool = False
    conditioning: OperatorConditioning = field(default_factory=OperatorConditioning)


@dataclass
class GeometryOp(ObservationOperator):
    pass


@dataclass
class SpectralResponseOp(ObservationOperator):
    pass


@dataclass
class PSFOp(ObservationOperator):
    pixel_response_included: bool | None = None
    pixel_response_provenance: str | None = None

    def __post_init__(self) -> None:
        if self.pixel_response_included is not None and not isinstance(
            self.pixel_response_included, bool
        ):
            raise TypeError("pixel_response_included must be bool or None")
        if self.pixel_response_provenance is not None and (
            not isinstance(self.pixel_response_provenance, str)
            or not self.pixel_response_provenance.strip()
        ):
            raise TypeError(
                "pixel_response_provenance must be a nonempty string or None"
            )


@dataclass
class RadiometryOp(ObservationOperator):
    pass


@dataclass
class BackgroundOp(ObservationOperator):
    pass


@dataclass
class MaskWeightOp(ObservationOperator):
    pass


@dataclass
class NoiseOp(ObservationOperator):
    pass
