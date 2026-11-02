from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping as TypingMapping, Sequence

import torch
from torch import Tensor, nn

from .mapping import (
    IdentityMapping,
    Mapping,
    SerialMapping,
    UnsupportedInverseError,
    _deserialize_mapping,
    _mapping_list,
    _serialize_mapping,
)
from .semantics import AxisSpec, Epoch, FrameSpec, TimeSpec


class UnsupportedPathError(RuntimeError):
    pass


class AmbiguousPathError(RuntimeError):
    pass


@dataclass(frozen=True)
class TransformEdge:
    source: str
    target: str
    mapping: Mapping


def _epoch_to_data(
    epoch: Epoch | None, tensors: dict[str, Tensor], prefix: str
) -> dict[str, Any] | None:
    if epoch is None:
        return None
    key = f"{prefix}.value"
    tensors[key] = epoch.value
    return {"spec": epoch.spec.to_dict(), "value": key}


def _epoch_from_data(
    data: TypingMapping[str, Any] | None, tensors: TypingMapping[str, Tensor]
) -> Epoch | None:
    if data is None:
        return None
    return Epoch(tensors[str(data["value"])], TimeSpec.from_dict(data["spec"]))


def _frame_to_data(
    frame: FrameSpec, tensors: dict[str, Tensor], prefix: str
) -> dict[str, Any]:
    return {
        "name": frame.name,
        "axes": [axis.to_dict() for axis in frame.axes],
        "representation": frame.representation,
        "equinox": _epoch_to_data(frame.equinox, tensors, f"{prefix}.equinox"),
        "obstime": _epoch_to_data(frame.obstime, tensors, f"{prefix}.obstime"),
        "observer": frame.observer,
        "reference_position": frame.reference_position,
    }


def _frame_from_data(
    data: TypingMapping[str, Any], tensors: TypingMapping[str, Tensor]
) -> FrameSpec:
    return FrameSpec(
        str(data["name"]),
        [AxisSpec.from_dict(axis) for axis in data["axes"]],
        representation=str(data["representation"]),
        equinox=_epoch_from_data(data.get("equinox"), tensors),
        obstime=_epoch_from_data(data.get("obstime"), tensors),
        observer=data.get("observer"),
        reference_position=data.get("reference_position"),
    )


class TransformGraph(nn.Module):
    def __init__(
        self,
        frames: Sequence[FrameSpec] = (),
        transforms: Sequence[TransformEdge] = (),
    ) -> None:
        super().__init__()
        self.frames: dict[str, FrameSpec] = {}
        self._sources: list[str] = []
        self._targets: list[str] = []
        self.mappings = _mapping_list()
        for frame in frames:
            self.add_frame(frame)
        for edge in transforms:
            self.add_transform(edge.source, edge.target, edge.mapping)
        self.validate()

    def add_frame(self, frame: FrameSpec) -> TransformGraph:
        if frame.name in self.frames:
            raise ValueError(f"duplicate frame {frame.name!r}")
        self.frames[frame.name] = frame
        return self

    def add_transform(
        self, source: str, target: str, mapping: Mapping
    ) -> TransformGraph:
        if source not in self.frames or target not in self.frames:
            raise ValueError("transform endpoints must be registered frames")
        if source == target:
            raise ValueError("self-loop transforms are not supported")
        source_frame = self.frames[source]
        target_frame = self.frames[target]
        if mapping.n_inputs != source_frame.n_dim:
            raise ValueError("mapping inputs do not match source frame dimensionality")
        if mapping.n_outputs != target_frame.n_dim:
            raise ValueError("mapping outputs do not match target frame dimensionality")
        for frame_axis, mapping_axis in zip(
            source_frame.axes, mapping.input_axes, strict=True
        ):
            if not frame_axis.compatible_with(mapping_axis):
                raise ValueError("mapping input axes do not match source frame axes")
        for mapping_axis, frame_axis in zip(
            mapping.output_axes, target_frame.axes, strict=True
        ):
            if not mapping_axis.compatible_with(frame_axis):
                raise ValueError("mapping output axes do not match target frame axes")
        self._sources.append(source)
        self._targets.append(target)
        self.mappings.append(mapping)
        try:
            self.validate()
        except Exception:
            self._sources.pop()
            self._targets.pop()
            del self.mappings[-1]
            raise
        return self

    @property
    def edges(self) -> tuple[TransformEdge, ...]:
        return tuple(
            TransformEdge(source, target, mapping)
            for source, target, mapping in zip(
                self._sources, self._targets, self.mappings, strict=True
            )
        )

    def validate(self) -> None:
        seen: set[tuple[str, str]] = set()
        adjacency: dict[str, list[str]] = {name: [] for name in self.frames}
        for source, target in zip(self._sources, self._targets, strict=True):
            pair = (source, target)
            if pair in seen:
                raise ValueError(f"duplicate transform edge {source!r}->{target!r}")
            seen.add(pair)
            adjacency[source].append(target)
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node: str) -> None:
            if node in visiting:
                raise ValueError("transform graph must be acyclic")
            if node in visited:
                return
            visiting.add(node)
            for neighbour in adjacency[node]:
                visit(neighbour)
            visiting.remove(node)
            visited.add(node)

        for frame_name in self.frames:
            visit(frame_name)

    def _adjacency(self) -> dict[str, list[tuple[str, Mapping]]]:
        adjacency: dict[str, list[tuple[str, Mapping]]] = {
            name: [] for name in self.frames
        }
        for source, target, mapping in zip(
            self._sources, self._targets, self.mappings, strict=True
        ):
            adjacency[source].append((target, mapping))
            inverse = mapping._make_exact_inverse()
            if inverse is not None:
                adjacency[target].append((source, inverse))
        for neighbours in adjacency.values():
            neighbours.sort(key=lambda item: item[0])
        return adjacency

    def find_path(
        self,
        source: str,
        target: str,
        *,
        path: Sequence[str] | None = None,
    ) -> tuple[str, ...]:
        if source not in self.frames or target not in self.frames:
            raise UnsupportedPathError("path endpoints must be registered frames")
        if path is not None:
            selected = tuple(path)
            if not selected or selected[0] != source or selected[-1] != target:
                raise ValueError("explicit path must start at source and end at target")
            self._mappings_for_path(selected)
            return selected
        if source == target:
            return (source,)
        adjacency = self._adjacency()
        queue: deque[tuple[str, tuple[str, ...]]] = deque([(source, (source,))])
        solutions: list[tuple[str, ...]] = []
        shortest: int | None = None
        while queue:
            node, current = queue.popleft()
            distance = len(current) - 1
            if shortest is not None and distance >= shortest:
                continue
            for neighbour, _mapping in adjacency[node]:
                if neighbour in current:
                    continue
                candidate = current + (neighbour,)
                if neighbour == target:
                    shortest = len(candidate) - 1
                    solutions.append(candidate)
                else:
                    queue.append((neighbour, candidate))
        if not solutions:
            raise UnsupportedPathError(f"no transform path {source!r}->{target!r}")
        shortest_solutions = [
            solution for solution in solutions if len(solution) == len(solutions[0])
        ]
        if len(shortest_solutions) != 1:
            raise AmbiguousPathError(
                f"multiple shortest transform paths {source!r}->{target!r}; "
                "select an explicit path"
            )
        return shortest_solutions[0]

    def _mappings_for_path(self, path: Sequence[str]) -> list[Mapping]:
        adjacency = self._adjacency()
        mappings = []
        for source, target in zip(path, path[1:]):
            matches = [
                mapping
                for neighbour, mapping in adjacency[source]
                if neighbour == target
            ]
            if not matches:
                raise UnsupportedPathError(
                    f"explicit path contains unsupported edge {source!r}->{target!r}"
                )
            if len(matches) != 1:
                raise AmbiguousPathError(
                    f"explicit path edge {source!r}->{target!r} is ambiguous"
                )
            mappings.append(matches[0])
        return mappings

    def get_mapping(
        self,
        source: str,
        target: str,
        *,
        path: Sequence[str] | None = None,
    ) -> Mapping:
        selected = self.find_path(source, target, path=path)
        if len(selected) == 1:
            return IdentityMapping(self.frames[source].axes)
        mappings = self._mappings_for_path(selected)
        return mappings[0] if len(mappings) == 1 else SerialMapping(mappings)

    def to_state(self) -> tuple[dict[str, Any], dict[str, Tensor]]:
        tensors: dict[str, Tensor] = {}
        metadata = {
            "version": 1,
            "frames": [
                _frame_to_data(frame, tensors, f"frames.{index}")
                for index, frame in enumerate(self.frames.values())
            ],
            "edges": [
                {
                    "source": source,
                    "target": target,
                    "mapping": _serialize_mapping(
                        mapping, tensors, f"edges.{index}.mapping"
                    ),
                }
                for index, (source, target, mapping) in enumerate(
                    zip(self._sources, self._targets, self.mappings, strict=True)
                )
            ],
        }
        return metadata, tensors

    @classmethod
    def from_state(
        cls, metadata: TypingMapping[str, Any], tensors: TypingMapping[str, Tensor]
    ) -> TransformGraph:
        if int(metadata.get("version", 0)) != 1:
            raise ValueError("unsupported transform graph serialization version")
        frames = [_frame_from_data(frame, tensors) for frame in metadata["frames"]]
        graph = cls(frames)
        for edge in metadata["edges"]:
            graph.add_transform(
                str(edge["source"]),
                str(edge["target"]),
                _deserialize_mapping(edge["mapping"], tensors),
            )
        return graph


class GeneralizedWCS(nn.Module):
    def __init__(
        self,
        graph: TransformGraph,
        input_frame: str,
        output_frame: str,
        *,
        path: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        selected = graph.find_path(input_frame, output_frame, path=path)
        self.graph = graph
        self.input_frame = input_frame
        self.output_frame = output_frame
        self.path = selected

    @property
    def pixel_n_dim(self) -> int:
        return self.graph.frames[self.input_frame].n_dim

    @property
    def world_n_dim(self) -> int:
        return self.graph.frames[self.output_frame].n_dim

    @property
    def world_axis_names(self) -> tuple[str, ...]:
        return tuple(axis.name for axis in self.graph.frames[self.output_frame].axes)

    @property
    def world_axis_units(self) -> tuple[str, ...]:
        return tuple(
            axis.unit_spec.symbol for axis in self.graph.frames[self.output_frame].axes
        )

    @property
    def world_axis_physical_types(self) -> tuple[str | None, ...]:
        return tuple(
            axis.physical_type for axis in self.graph.frames[self.output_frame].axes
        )

    @property
    def bounding_box(self) -> tuple[tuple[float | None, float | None], ...] | None:
        mapping = self.graph.get_mapping(
            self.input_frame, self.output_frame, path=self.path
        )
        return None if mapping.domain is None else mapping.domain.bounds

    def get_transform(
        self,
        source: str,
        target: str,
        *,
        path: Sequence[str] | None = None,
    ) -> Mapping:
        return self.graph.get_mapping(source, target, path=path)

    def pixel_to_world(self, pixels: Tensor) -> Tensor:
        mapping = self.get_transform(
            self.input_frame, self.output_frame, path=self.path
        )
        return mapping(pixels)

    def world_to_pixel(self, world: Tensor) -> Tensor:
        reverse = tuple(reversed(self.path))
        try:
            mapping = self.get_transform(
                self.output_frame, self.input_frame, path=reverse
            )
        except UnsupportedPathError as exc:
            raise UnsupportedInverseError(
                "GeneralizedWCS world_to_pixel requires exact inverse mappings"
            ) from exc
        return mapping(world)

    @staticmethod
    def _stack_values(values: Sequence[Any], expected: int) -> Tensor:
        if len(values) != expected:
            raise ValueError(f"expected {expected} coordinate arrays")
        tensors = [torch.as_tensor(value) for value in values]
        broadcast = torch.broadcast_tensors(*tensors)
        return torch.stack(broadcast, dim=-1)

    def pixel_to_world_values(self, *pixels: Any) -> tuple[Tensor, ...]:
        output = self.pixel_to_world(self._stack_values(pixels, self.pixel_n_dim))
        return tuple(output[..., axis] for axis in range(self.world_n_dim))

    def world_to_pixel_values(self, *world: Any) -> tuple[Tensor, ...]:
        output = self.world_to_pixel(self._stack_values(world, self.world_n_dim))
        return tuple(output[..., axis] for axis in range(self.pixel_n_dim))

    def to_state(self) -> tuple[dict[str, Any], dict[str, Tensor]]:
        graph_metadata, tensors = self.graph.to_state()
        return {
            "version": 1,
            "input_frame": self.input_frame,
            "output_frame": self.output_frame,
            "path": list(self.path),
            "graph": graph_metadata,
        }, tensors

    @classmethod
    def from_state(
        cls, metadata: TypingMapping[str, Any], tensors: TypingMapping[str, Tensor]
    ) -> GeneralizedWCS:
        if int(metadata.get("version", 0)) != 1:
            raise ValueError("unsupported GeneralizedWCS serialization version")
        return cls(
            TransformGraph.from_state(metadata["graph"], tensors),
            str(metadata["input_frame"]),
            str(metadata["output_frame"]),
            path=metadata["path"],
        )


__all__ = [
    "AmbiguousPathError",
    "GeneralizedWCS",
    "TransformEdge",
    "TransformGraph",
    "UnsupportedPathError",
]
