"""Contract for every public namespace's declared surface."""

from __future__ import annotations

import importlib

import pytest

import torchsky

TYPOGRAPHY_LEAKS = {
    "Any",
    "Dict",
    "Literal",
    "Mapping",
    "Optional",
    "Sequence",
    "Tuple",
    "Type",
    "Union",
    "annotations",
    "dataclass",
    "math",
    "nn",
    "torch",
}


def _namespace(name: str):
    module = importlib.import_module(f"torchsky.{name}")
    names = getattr(module, "__all__", None)
    assert names, f"torchsky.{name} must declare __all__"
    return module, tuple(names)


@pytest.mark.parametrize("name", sorted(torchsky._NAMESPACES))
def test_namespace_declares_a_clean_public_surface(name: str) -> None:
    module, names = _namespace(name)

    assert len(names) == len(set(names)), f"torchsky.{name}.__all__ has duplicates"
    missing = [item for item in names if not hasattr(module, item)]
    assert not missing, f"torchsky.{name}.__all__ names missing: {missing}"
    assert not [item for item in names if item.startswith("_")]

    leaked = sorted(set(names) & TYPOGRAPHY_LEAKS)
    assert not leaked, f"torchsky.{name}.__all__ leaks imports: {leaked}"


@pytest.mark.parametrize("name", sorted(torchsky._NAMESPACES))
def test_star_import_matches_declared_surface(name: str) -> None:
    _, names = _namespace(name)
    namespace: dict[str, object] = {}
    exec(f"from torchsky.{name} import *", namespace)  # noqa: S102 - contract test
    imported = {key for key in namespace if not key.startswith("__")}
    assert imported == set(names)


def test_root_namespaces_are_lazy_and_declared() -> None:
    assert set(torchsky.__all__) == set(torchsky._NAMESPACES) | {"WCS"}
    for name in torchsky._NAMESPACES:
        assert name in torchsky.__all__
