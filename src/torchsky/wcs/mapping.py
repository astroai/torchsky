from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections.abc import Iterable, MutableSequence
from typing import Any, Callable, Mapping as TypingMapping, Sequence, cast

import torch
from torch import Tensor, nn

from .semantics import AxisSpec, DomainSpec, UnitSpec, axes_domain


class UnsupportedInverseError(RuntimeError):
    pass


def _mapping_list(items: Iterable[Mapping] = ()) -> MutableSequence[Mapping]:
    """``nn.ModuleList`` of mappings, typed by its element type."""
    return cast("MutableSequence[Mapping]", nn.ModuleList(items))


class Mapping(nn.Module, ABC):
    input_axes: tuple[AxisSpec, ...]
    output_axes: tuple[AxisSpec, ...]
    domain: DomainSpec | None

    def __init__(
        self,
        input_axes: Sequence[AxisSpec],
        output_axes: Sequence[AxisSpec],
        *,
        domain: DomainSpec | None = None,
    ) -> None:
        super().__init__()
        self.input_axes = tuple(input_axes)
        self.output_axes = tuple(output_axes)
        if not self.input_axes or not self.output_axes:
            raise ValueError("mappings require at least one input and one output")
        if domain is not None and domain.n_dim != len(self.input_axes):
            raise ValueError("mapping domain dimensionality must match mapping inputs")
        self.domain = domain

    @property
    def n_inputs(self) -> int:
        return len(self.input_axes)

    @property
    def n_outputs(self) -> int:
        return len(self.output_axes)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim == 0 or inputs.shape[-1] != self.n_inputs:
            raise ValueError(f"mapping expects trailing dimension {self.n_inputs}")
        outputs = self._evaluate(inputs)
        if outputs.ndim == 0 or outputs.shape[-1] != self.n_outputs:
            raise RuntimeError(
                f"mapping returned trailing dimension {outputs.shape[-1] if outputs.ndim else 0}; "
                f"expected {self.n_outputs}"
            )
        return outputs

    @abstractmethod
    def _evaluate(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError

    @property
    def has_exact_inverse(self) -> bool:
        return self._make_exact_inverse() is not None

    @property
    def inverse(self) -> Mapping:
        inverse = self._make_exact_inverse()
        if inverse is None:
            raise UnsupportedInverseError(
                f"{type(self).__name__} does not define an exact inverse"
            )
        return inverse

    def _make_exact_inverse(self) -> Mapping | None:
        return None

    def then(self, other: Mapping) -> SerialMapping:
        return SerialMapping((self, other))

    def parallel(self, other: Mapping) -> ParallelMapping:
        return ParallelMapping((self, other))

    def __or__(self, other: Mapping) -> SerialMapping:
        return self.then(other)

    def __and__(self, other: Mapping) -> ParallelMapping:
        return self.parallel(other)

    def in_domain(self, inputs: Tensor) -> Tensor:
        if inputs.ndim == 0 or inputs.shape[-1] != self.n_inputs:
            raise ValueError(f"mapping expects trailing dimension {self.n_inputs}")
        if self.domain is None:
            return torch.ones(inputs.shape[:-1], dtype=torch.bool, device=inputs.device)
        return self.domain.contains(inputs)

    def jacobian(self, inputs: Tensor) -> Tensor:
        if inputs.ndim == 0 or inputs.shape[-1] != self.n_inputs:
            raise ValueError(f"mapping expects trailing dimension {self.n_inputs}")
        batch_shape = inputs.shape[:-1]
        flat = inputs.reshape(-1, self.n_inputs)

        def evaluate_one(value: Tensor) -> Tensor:
            return self(value.unsqueeze(0)).squeeze(0)

        result = torch.vmap(torch.func.jacrev(evaluate_one))(flat)
        return result.reshape(batch_shape + (self.n_outputs, self.n_inputs))

    def numerical_inverse(
        self,
        *,
        initial: Tensor,
        bounds: DomainSpec,
        max_iterations: int = 16,
        residual_tolerance: float = 1.0e-10,
        relative_tolerance: float = 1.0e-10,
        damping: float = 1.0e-8,
    ) -> NumericalInverseMapping:
        return NumericalInverseMapping(
            self,
            initial=initial,
            bounds=bounds,
            max_iterations=max_iterations,
            residual_tolerance=residual_tolerance,
            relative_tolerance=relative_tolerance,
            damping=damping,
        )

    def to_state(self) -> tuple[dict[str, Any], dict[str, Tensor]]:
        tensors: dict[str, Tensor] = {}
        metadata = _serialize_mapping(self, tensors, "mapping")
        return metadata, tensors

    @classmethod
    def from_state(
        cls, metadata: TypingMapping[str, Any], tensors: TypingMapping[str, Tensor]
    ) -> Mapping:
        return _deserialize_mapping(metadata, tensors)


class IdentityMapping(Mapping):
    def __init__(self, axes: Sequence[AxisSpec]) -> None:
        axes_tuple = tuple(axes)
        super().__init__(axes_tuple, axes_tuple, domain=axes_domain(axes_tuple))

    def _evaluate(self, inputs: Tensor) -> Tensor:
        return inputs

    def _make_exact_inverse(self) -> Mapping:
        return IdentityMapping(self.input_axes)


class AffineMapping(Mapping):
    matrix: Tensor
    offset: Tensor

    def __init__(
        self,
        matrix: Tensor,
        offset: Tensor,
        input_axes: Sequence[AxisSpec],
        output_axes: Sequence[AxisSpec],
        *,
        domain: DomainSpec | None = None,
    ) -> None:
        input_axes_tuple = tuple(input_axes)
        output_axes_tuple = tuple(output_axes)
        matrix_tensor = torch.as_tensor(matrix)
        offset_tensor = torch.as_tensor(offset)
        if matrix_tensor.shape != (len(output_axes_tuple), len(input_axes_tuple)):
            raise ValueError("affine matrix shape must be (n_outputs, n_inputs)")
        if offset_tensor.shape != (len(output_axes_tuple),):
            raise ValueError("affine offset shape must be (n_outputs,)")
        if not matrix_tensor.is_floating_point():
            matrix_tensor = matrix_tensor.to(dtype=torch.float64)
        if not offset_tensor.is_floating_point():
            offset_tensor = offset_tensor.to(dtype=matrix_tensor.dtype)
        offset_tensor = offset_tensor.to(
            device=matrix_tensor.device, dtype=matrix_tensor.dtype
        )
        super().__init__(input_axes_tuple, output_axes_tuple, domain=domain)
        self.register_buffer("matrix", matrix_tensor)
        self.register_buffer("offset", offset_tensor)

    def _evaluate(self, inputs: Tensor) -> Tensor:
        matrix = self.matrix.to(device=inputs.device, dtype=inputs.dtype)
        offset = self.offset.to(device=inputs.device, dtype=inputs.dtype)
        return torch.matmul(inputs, matrix.mT) + offset

    def _make_exact_inverse(self) -> Mapping | None:
        if self.n_inputs != self.n_outputs:
            return None
        inverse_matrix = torch.linalg.inv(self.matrix)
        inverse_offset = -(inverse_matrix @ self.offset)
        return AffineMapping(
            inverse_matrix,
            inverse_offset,
            self.output_axes,
            self.input_axes,
        )


class UnitConversionMapping(Mapping):
    factors: Tensor

    def __init__(
        self, input_axes: Sequence[AxisSpec], output_axes: Sequence[AxisSpec]
    ) -> None:
        source = tuple(input_axes)
        destination = tuple(output_axes)
        if len(source) != len(destination):
            raise ValueError(
                "unit conversion requires equal input and output dimensions"
            )
        factors = []
        for input_axis, output_axis in zip(source, destination, strict=True):
            if not input_axis.compatible_with(output_axis, require_unit=False):
                raise ValueError(
                    f"incompatible unit-conversion axes {input_axis.name!r} and "
                    f"{output_axis.name!r}"
                )
            factors.append(
                input_axis.unit_spec.conversion_factor_to(output_axis.unit_spec)
            )
        super().__init__(source, destination, domain=axes_domain(source))
        self.register_buffer("factors", torch.tensor(factors, dtype=torch.float64))

    def _evaluate(self, inputs: Tensor) -> Tensor:
        return inputs * self.factors.to(device=inputs.device, dtype=inputs.dtype)

    def _make_exact_inverse(self) -> Mapping:
        return UnitConversionMapping(self.output_axes, self.input_axes)


class CallableMapping(Mapping):
    def __init__(
        self,
        function: Callable[[Tensor], Tensor],
        input_axes: Sequence[AxisSpec],
        output_axes: Sequence[AxisSpec],
        *,
        inverse_function: Callable[[Tensor], Tensor] | None = None,
        domain: DomainSpec | None = None,
    ) -> None:
        super().__init__(input_axes, output_axes, domain=domain)
        self.function = function
        self.inverse_function = inverse_function

    def _evaluate(self, inputs: Tensor) -> Tensor:
        return self.function(inputs)

    def _make_exact_inverse(self) -> Mapping | None:
        if self.inverse_function is None:
            return None
        return CallableMapping(
            self.inverse_function,
            self.output_axes,
            self.input_axes,
            inverse_function=self.function,
        )


def _validate_serial_axes(left: Mapping, right: Mapping) -> None:
    if left.n_outputs != right.n_inputs:
        raise ValueError(
            f"serial mappings have incompatible dimensions {left.n_outputs} and "
            f"{right.n_inputs}"
        )
    for source, destination in zip(left.output_axes, right.input_axes, strict=True):
        if not source.compatible_with(destination):
            raise ValueError(
                f"serial axis mismatch {source.name!r} ({source.unit_spec.symbol}) -> "
                f"{destination.name!r} ({destination.unit_spec.symbol})"
            )


class SerialMapping(Mapping):
    def __init__(self, mappings: Sequence[Mapping]) -> None:
        flattened: list[Mapping] = []
        for mapping in mappings:
            if isinstance(mapping, SerialMapping):
                flattened.extend(mapping.mappings)
            else:
                flattened.append(mapping)
        if not flattened:
            raise ValueError("serial mapping requires at least one mapping")
        for left, right in zip(flattened, flattened[1:]):
            _validate_serial_axes(left, right)
        super().__init__(
            flattened[0].input_axes,
            flattened[-1].output_axes,
            domain=flattened[0].domain,
        )
        self.mappings = _mapping_list(flattened)

    def _evaluate(self, inputs: Tensor) -> Tensor:
        result = inputs
        for mapping in self.mappings:
            result = mapping(result)
        return result

    def _make_exact_inverse(self) -> Mapping | None:
        inverses = []
        for mapping in reversed(self.mappings):
            inverse = mapping._make_exact_inverse()
            if inverse is None:
                return None
            inverses.append(inverse)
        return SerialMapping(inverses)


class ParallelMapping(Mapping):
    def __init__(self, mappings: Sequence[Mapping]) -> None:
        mappings_tuple = tuple(mappings)
        if not mappings_tuple:
            raise ValueError("parallel mapping requires at least one mapping")
        input_axes = tuple(
            axis for mapping in mappings_tuple for axis in mapping.input_axes
        )
        output_axes = tuple(
            axis for mapping in mappings_tuple for axis in mapping.output_axes
        )
        domains = [mapping.domain for mapping in mappings_tuple]
        domain = None
        if all(item is not None for item in domains):
            domain = DomainSpec(
                tuple(
                    bound
                    for item in domains
                    if item is not None
                    for bound in item.bounds
                ),
                tuple(
                    flags
                    for item in domains
                    if item is not None
                    for flags in item.closed
                ),
            )
        super().__init__(input_axes, output_axes, domain=domain)
        self.mappings = _mapping_list(mappings_tuple)
        self._input_sizes = tuple(mapping.n_inputs for mapping in mappings_tuple)

    def _evaluate(self, inputs: Tensor) -> Tensor:
        chunks = torch.split(inputs, list(self._input_sizes), dim=-1)
        return torch.cat(
            [
                mapping(chunk)
                for mapping, chunk in zip(self.mappings, chunks, strict=True)
            ],
            dim=-1,
        )

    def _make_exact_inverse(self) -> Mapping | None:
        inverses = []
        for mapping in self.mappings:
            inverse = mapping._make_exact_inverse()
            if inverse is None:
                return None
            inverses.append(inverse)
        return ParallelMapping(inverses)


@dataclass(frozen=True)
class PortRoute:
    mapping: Mapping
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]

    def __init__(
        self, mapping: Mapping, inputs: Sequence[int], outputs: Sequence[int]
    ) -> None:
        object.__setattr__(self, "mapping", mapping)
        object.__setattr__(self, "inputs", tuple(int(index) for index in inputs))
        object.__setattr__(self, "outputs", tuple(int(index) for index in outputs))


class PortRoutingMapping(Mapping):
    def __init__(
        self,
        input_axes: Sequence[AxisSpec],
        output_axes: Sequence[AxisSpec],
        routes: Sequence[PortRoute],
        *,
        domain: DomainSpec | None = None,
    ) -> None:
        input_axes_tuple = tuple(input_axes)
        output_axes_tuple = tuple(output_axes)
        routes_tuple = tuple(routes)
        if not routes_tuple:
            raise ValueError("port routing requires at least one route")
        output_ports: list[int] = []
        for route in routes_tuple:
            if len(route.inputs) != route.mapping.n_inputs:
                raise ValueError("route input ports must match mapping inputs")
            if len(route.outputs) != route.mapping.n_outputs:
                raise ValueError("route output ports must match mapping outputs")
            if any(
                index < 0 or index >= len(input_axes_tuple) for index in route.inputs
            ):
                raise ValueError("route input port is out of range")
            if any(
                index < 0 or index >= len(output_axes_tuple) for index in route.outputs
            ):
                raise ValueError("route output port is out of range")
            for port, axis in zip(route.inputs, route.mapping.input_axes, strict=True):
                if not input_axes_tuple[port].compatible_with(axis):
                    raise ValueError(
                        "route input axis is incompatible with mapping input"
                    )
            for port, axis in zip(
                route.outputs, route.mapping.output_axes, strict=True
            ):
                if not axis.compatible_with(output_axes_tuple[port]):
                    raise ValueError(
                        "route output axis is incompatible with mapping output"
                    )
            output_ports.extend(route.outputs)
        if sorted(output_ports) != list(range(len(output_axes_tuple))):
            raise ValueError("every route output port must be assigned exactly once")
        super().__init__(input_axes_tuple, output_axes_tuple, domain=domain)
        self.routes = routes_tuple
        self.route_mappings = _mapping_list(route.mapping for route in routes_tuple)

    def _evaluate(self, inputs: Tensor) -> Tensor:
        outputs: list[Tensor | None] = [None] * self.n_outputs
        for route, mapping in zip(self.routes, self.route_mappings, strict=True):
            selected = inputs[..., list(route.inputs)]
            result = mapping(selected)
            for local, port in enumerate(route.outputs):
                outputs[port] = result[..., local]
        if any(output is None for output in outputs):
            raise RuntimeError("port route left an output unassigned")
        return torch.stack([output for output in outputs if output is not None], dim=-1)

    def _make_exact_inverse(self) -> Mapping | None:
        input_ports = [port for route in self.routes for port in route.inputs]
        if sorted(input_ports) != list(range(self.n_inputs)):
            return None
        inverse_routes = []
        for route, mapping in zip(self.routes, self.route_mappings, strict=True):
            inverse = mapping._make_exact_inverse()
            if inverse is None:
                return None
            inverse_routes.append(PortRoute(inverse, route.outputs, route.inputs))
        return PortRoutingMapping(self.output_axes, self.input_axes, inverse_routes)


@dataclass(frozen=True)
class NumericalInverseResult:
    values: Tensor
    converged: Tensor
    residual_norm: Tensor
    iterations: int


class NumericalInverseMapping(Mapping):
    initial: Tensor

    def __init__(
        self,
        mapping: Mapping,
        *,
        initial: Tensor,
        bounds: DomainSpec,
        max_iterations: int = 16,
        residual_tolerance: float = 1.0e-10,
        relative_tolerance: float = 1.0e-10,
        damping: float = 1.0e-8,
    ) -> None:
        if mapping.n_inputs != mapping.n_outputs:
            raise UnsupportedInverseError("numerical inverse requires a square mapping")
        initial_tensor = torch.as_tensor(initial)
        if initial_tensor.shape != (mapping.n_inputs,):
            raise ValueError(
                "numerical inverse initial guess must have shape (n_inputs,)"
            )
        if not initial_tensor.is_floating_point():
            initial_tensor = initial_tensor.to(dtype=torch.float64)
        if bounds.n_dim != mapping.n_inputs:
            raise ValueError("numerical inverse bounds must match mapping inputs")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if residual_tolerance < 0.0 or relative_tolerance < 0.0:
            raise ValueError("inverse tolerances must be non-negative")
        if damping < 0.0:
            raise ValueError("inverse damping must be non-negative")
        super().__init__(mapping.output_axes, mapping.input_axes)
        self.mapping = mapping
        self.bounds = bounds
        self.max_iterations = int(max_iterations)
        self.residual_tolerance = float(residual_tolerance)
        self.relative_tolerance = float(relative_tolerance)
        self.damping = float(damping)
        self.register_buffer("initial", initial_tensor)

    def solve(self, targets: Tensor) -> NumericalInverseResult:
        if targets.ndim == 0 or targets.shape[-1] != self.n_inputs:
            raise ValueError(f"inverse expects trailing dimension {self.n_inputs}")
        batch_shape = targets.shape[:-1]
        flat_targets = targets.reshape(-1, self.n_inputs)
        guess = self.initial.to(device=targets.device, dtype=targets.dtype).expand_as(
            flat_targets
        )
        identity = torch.eye(
            self.n_inputs, dtype=targets.dtype, device=targets.device
        ).expand(flat_targets.shape[0], -1, -1)

        def evaluate_one(value: Tensor) -> Tensor:
            return self.mapping(value.unsqueeze(0)).squeeze(0)

        for _ in range(self.max_iterations):
            predicted = self.mapping(guess)
            residual = predicted - flat_targets
            jacobian = torch.vmap(torch.func.jacrev(evaluate_one))(guess)
            normal = jacobian.mT @ jacobian + self.damping * identity
            right = (jacobian.mT @ residual.unsqueeze(-1)).squeeze(-1)
            step = torch.linalg.solve(normal, right)
            guess = self.bounds.clamp(guess - step)
        residual = self.mapping(guess) - flat_targets
        residual_norm = torch.linalg.vector_norm(residual, dim=-1)
        target_norm = torch.linalg.vector_norm(flat_targets, dim=-1)
        converged = residual_norm <= (
            self.residual_tolerance + self.relative_tolerance * target_norm
        )
        return NumericalInverseResult(
            values=guess.reshape(batch_shape + (self.n_outputs,)),
            converged=converged.reshape(batch_shape),
            residual_norm=residual_norm.reshape(batch_shape),
            iterations=self.max_iterations,
        )

    def _evaluate(self, inputs: Tensor) -> Tensor:
        result = self.solve(inputs)
        if not bool(torch.all(result.converged)):
            failed = int((~result.converged).sum().detach().cpu())
            raise RuntimeError(
                f"numerical inverse failed to converge for {failed} input row(s)"
            )
        return result.values


class WCSMapping(Mapping):
    def __init__(self, wcs: Any, *, direction: str = "pixel_to_world") -> None:
        if direction not in {"pixel_to_world", "world_to_pixel"}:
            raise ValueError(
                "WCS mapping direction must be pixel_to_world or world_to_pixel"
            )
        pixel_axes = tuple(
            AxisSpec(f"pixel{axis + 1}", UnitSpec("pix"), "pixel")
            for axis in range(wcs.pixel_n_dim)
        )
        world_axes = tuple(
            AxisSpec(name, UnitSpec(unit), physical_type)
            for name, unit, physical_type in zip(
                wcs.world_axis_names,
                wcs.world_axis_units,
                wcs.world_axis_physical_types,
                strict=True,
            )
        )
        if direction == "pixel_to_world":
            input_axes, output_axes = pixel_axes, world_axes
        else:
            input_axes, output_axes = world_axes, pixel_axes
        super().__init__(input_axes, output_axes)
        self.wcs = wcs
        self.direction = direction

    def _evaluate(self, inputs: Tensor) -> Tensor:
        if self.direction == "pixel_to_world":
            return self.wcs.pixel_to_world_differentiable(inputs, origin=0)
        return self.wcs.world_to_pixel_differentiable(inputs, origin=0)

    def _make_exact_inverse(self) -> Mapping:
        direction = (
            "world_to_pixel" if self.direction == "pixel_to_world" else "pixel_to_world"
        )
        return WCSMapping(self.wcs, direction=direction)


def _axes_to_data(axes: Sequence[AxisSpec]) -> list[dict[str, Any]]:
    return [axis.to_dict() for axis in axes]


def _axes_from_data(data: Sequence[TypingMapping[str, Any]]) -> tuple[AxisSpec, ...]:
    return tuple(AxisSpec.from_dict(axis) for axis in data)


def _domain_to_data(domain: DomainSpec | None) -> dict[str, Any] | None:
    return None if domain is None else domain.to_dict()


def _domain_from_data(data: TypingMapping[str, Any] | None) -> DomainSpec | None:
    return None if data is None else DomainSpec.from_dict(data)


def _serialize_mapping(
    mapping: Mapping, tensors: dict[str, Tensor], prefix: str
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "input_axes": _axes_to_data(mapping.input_axes),
        "output_axes": _axes_to_data(mapping.output_axes),
        "domain": _domain_to_data(mapping.domain),
    }
    if isinstance(mapping, IdentityMapping):
        return {"type": "identity", **base}
    if isinstance(mapping, AffineMapping):
        matrix_key = f"{prefix}.matrix"
        offset_key = f"{prefix}.offset"
        tensors[matrix_key] = mapping.matrix
        tensors[offset_key] = mapping.offset
        return {
            "type": "affine",
            **base,
            "matrix": matrix_key,
            "offset": offset_key,
        }
    if isinstance(mapping, UnitConversionMapping):
        return {"type": "unit_conversion", **base}
    if isinstance(mapping, SerialMapping):
        return {
            "type": "serial",
            **base,
            "mappings": [
                _serialize_mapping(child, tensors, f"{prefix}.{index}")
                for index, child in enumerate(mapping.mappings)
            ],
        }
    if isinstance(mapping, ParallelMapping):
        return {
            "type": "parallel",
            **base,
            "mappings": [
                _serialize_mapping(child, tensors, f"{prefix}.{index}")
                for index, child in enumerate(mapping.mappings)
            ],
        }
    if isinstance(mapping, PortRoutingMapping):
        return {
            "type": "port_routing",
            **base,
            "routes": [
                {
                    "inputs": list(route.inputs),
                    "outputs": list(route.outputs),
                    "mapping": _serialize_mapping(child, tensors, f"{prefix}.{index}"),
                }
                for index, (route, child) in enumerate(
                    zip(mapping.routes, mapping.route_mappings, strict=True)
                )
            ],
        }
    if isinstance(mapping, NumericalInverseMapping):
        initial_key = f"{prefix}.initial"
        tensors[initial_key] = mapping.initial
        return {
            "type": "numerical_inverse",
            **base,
            "mapping": _serialize_mapping(mapping.mapping, tensors, f"{prefix}.base"),
            "initial": initial_key,
            "bounds": mapping.bounds.to_dict(),
            "max_iterations": mapping.max_iterations,
            "residual_tolerance": mapping.residual_tolerance,
            "relative_tolerance": mapping.relative_tolerance,
            "damping": mapping.damping,
        }
    if isinstance(mapping, WCSMapping):
        wcs = mapping.wcs
        if (
            wcs.tabular_axes
            or wcs.coupled_tabular is not None
            or wcs.cpdis1 is not None
            or wcs.cpdis2 is not None
            or wcs.det2im1 is not None
            or wcs.det2im2 is not None
        ):
            raise TypeError(
                "WCS mappings with tabular or lookup state are not serializable"
            )
        return {
            "type": "wcs",
            **base,
            "header": dict(wcs.to_header()),
            "direction": mapping.direction,
        }
    if isinstance(mapping, CallableMapping):
        raise TypeError("callable mappings are not serializable")
    raise TypeError(f"unsupported mapping type {type(mapping).__name__!r}")


def _deserialize_mapping(
    data: TypingMapping[str, Any], tensors: TypingMapping[str, Tensor]
) -> Mapping:
    mapping_type = str(data["type"])
    input_axes = _axes_from_data(data["input_axes"])
    output_axes = _axes_from_data(data["output_axes"])
    domain = _domain_from_data(data.get("domain"))
    if mapping_type == "identity":
        return IdentityMapping(input_axes)
    if mapping_type == "affine":
        return AffineMapping(
            tensors[str(data["matrix"])],
            tensors[str(data["offset"])],
            input_axes,
            output_axes,
            domain=domain,
        )
    if mapping_type == "unit_conversion":
        return UnitConversionMapping(input_axes, output_axes)
    if mapping_type == "serial":
        return SerialMapping(
            [_deserialize_mapping(child, tensors) for child in data["mappings"]]
        )
    if mapping_type == "parallel":
        return ParallelMapping(
            [_deserialize_mapping(child, tensors) for child in data["mappings"]]
        )
    if mapping_type == "port_routing":
        routes = [
            PortRoute(
                _deserialize_mapping(route["mapping"], tensors),
                route["inputs"],
                route["outputs"],
            )
            for route in data["routes"]
        ]
        return PortRoutingMapping(input_axes, output_axes, routes, domain=domain)
    if mapping_type == "numerical_inverse":
        return NumericalInverseMapping(
            _deserialize_mapping(data["mapping"], tensors),
            initial=tensors[str(data["initial"])],
            bounds=DomainSpec.from_dict(data["bounds"]),
            max_iterations=int(data["max_iterations"]),
            residual_tolerance=float(data["residual_tolerance"]),
            relative_tolerance=float(data["relative_tolerance"]),
            damping=float(data["damping"]),
        )
    if mapping_type == "wcs":
        from .core import WCS

        return WCSMapping(WCS(dict(data["header"])), direction=str(data["direction"]))
    raise ValueError(f"unsupported serialized mapping type {mapping_type!r}")


__all__ = [
    "AffineMapping",
    "CallableMapping",
    "IdentityMapping",
    "NumericalInverseMapping",
    "NumericalInverseResult",
    "ParallelMapping",
    "PortRoute",
    "PortRoutingMapping",
    "SerialMapping",
    "UnitConversionMapping",
    "UnsupportedInverseError",
    "WCSMapping",
    "TransformMapping",
]

TransformMapping = Mapping
