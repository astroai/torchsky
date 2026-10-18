"""Declared fixed-model dependencies, without tensor copies or content hashes."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import math
from typing import ClassVar

from torch import Tensor


def _revision_token(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, tuple):
        return tuple(_revision_token(item) for item in value)
    raise TypeError("revision must return an immutable scalar or tuple token")


def _version(tensor: Tensor, *, name: str, revision_available: bool) -> int | None:
    try:
        return tensor._version
    except RuntimeError as error:
        if revision_available and tensor.is_inference():
            return None
        raise ValueError(
            f"dependency {name!r} has no mutation version; inference tensors require an explicit revision"
        ) from error


@dataclass(frozen=True)
class EvidenceSnapshot:
    """A snapshot of caller-declared tensors and an optional revision getter.

    Capture response matrices, precision/module parameters, calibration and
    support tensors explicitly. The revision getter must return an immutable
    scalar/tuple token and change whenever non-tensor state, tensor bindings,
    gauge or model conventions change. Identifiers alone do not verify physics.
    Dependencies retain tensors without detaching, copying or scanning them.

    Torch mutation versions detect ordinary in-place writes (including shared
    views); NumPy and ``.data`` alias writes may bypass them. Inference tensors
    have no version counter and require a caller-maintained explicit revision.
    Arbitrary callback closures and module parameter replacement are not
    inspected. Rebuild factors, statistics and queries after a change.
    """

    _tensors: tuple[tuple[str, Tensor, int | None], ...] = field(repr=False)
    _revision: Callable[[], object] | None = field(default=None, repr=False)
    _token: object = field(default=None, repr=False)
    _parents: tuple[EvidenceSnapshot, ...] = field(default=(), repr=False)

    @classmethod
    def capture(
        cls,
        tensors: Mapping[str, Tensor] | None = None,
        *,
        revision: Callable[[], object] | None = None,
    ) -> EvidenceSnapshot:
        if revision is not None and not callable(revision):
            raise TypeError("revision must be callable")
        token = None if revision is None else _revision_token(revision())
        tracked = []
        for name, tensor in ({} if tensors is None else tensors).items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("dependency names must be nonempty strings")
            if not isinstance(tensor, Tensor):
                raise TypeError("declared tensor dependencies must be tensors")
            tracked.append(
                (
                    name,
                    tensor,
                    _version(
                        tensor, name=name, revision_available=revision is not None
                    ),
                )
            )
        return cls(tuple(tracked), revision, token)

    def with_tensors(self, tensors: Mapping[str, Tensor]) -> EvidenceSnapshot:
        """Extend an unchanged snapshot, retaining its explicit revision policy."""
        self.validate()
        local = EvidenceSnapshot.capture(tensors, revision=self._revision)
        return EvidenceSnapshot(local._tensors, local._revision, local._token, (self,))

    def validate(self) -> None:
        for parent in self._parents:
            parent.validate()
        if (
            self._revision is not None
            and _revision_token(self._revision()) != self._token
        ):
            raise RuntimeError(
                "stale evidence revision; rebuild the factor/system/query"
            )
        for name, tensor, version in self._tensors:
            if (
                version is not None
                and _version(tensor, name=name, revision_available=False) != version
            ):
                raise RuntimeError(
                    f"stale evidence dependency {name!r}; rebuild the factor/system/query"
                )


class _GuardedEvidence:
    """Validate declared sources before exposing cached public statistics."""

    _guarded_fields: ClassVar[frozenset[str]] = frozenset()

    def __getattribute__(self, name: str) -> object:
        if name in object.__getattribute__(self, "_guarded_fields"):
            try:
                guard = object.__getattribute__(self, "_evidence_guard")
            except AttributeError:
                guard = None
            if guard is not None:
                guard()
        return object.__getattribute__(self, name)


__all__ = ["EvidenceSnapshot"]
