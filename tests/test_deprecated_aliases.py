"""Names renamed after 0.5.0 keep one warning alias until 0.7."""

from __future__ import annotations

import importlib

import pytest

ALIASES = (
    (
        "torchsky.map.field.native_response",
        "NativeFieldResponse",
        "torchsky.map.field.source_quadrature",
        "SourceQuadratureResponse",
    ),
    (
        "torchsky.map.field.native_response",
        "bind_native_field_likelihood",
        "torchsky.map.field.source_quadrature",
        "bind_source_quadrature_likelihood",
    ),
    (
        "torchsky.map.field.native_response",
        "render_native_field_observation",
        "torchsky.map.field.source_quadrature",
        "render_source_quadrature_observation",
    ),
    (
        "torchsky.map.field.native_integrated_read",
        "NativeIntegratedReadLikelihood",
        "torchsky.map.field.source_quadrature_read",
        "SourceQuadratureReadLikelihood",
    ),
    (
        "torchsky.map.field.native_integrated_read",
        "bind_native_integrated_read_likelihood",
        "torchsky.map.field.source_quadrature_read",
        "bind_source_quadrature_read_likelihood",
    ),
    (
        "torchsky.map.field",
        "NativeIntegratedReadLikelihood",
        "torchsky.map.field.source_quadrature_read",
        "SourceQuadratureReadLikelihood",
    ),
    (
        "torchsky.map.field",
        "bind_native_integrated_read_likelihood",
        "torchsky.map.field.source_quadrature_read",
        "bind_source_quadrature_read_likelihood",
    ),
    (
        "torchsky.map.field.joint_native_likelihood",
        "bind_joint_native_likelihood",
        "torchsky.map.field.shared_nuisance_likelihood",
        "bind_shared_nuisance_likelihood",
    ),
    *(
        (module, name, "torchsky.map.operators.lensing", new)
        for module in ("torchsky.map.operators", "torchsky.map.operators.magnification")
        for name, new in (
            ("ConvergenceMagnification", "ConvergenceMagnification"),
            ("MagnificationResidual", "LearnedConvergenceCorrection"),
            ("magnification_bias", "magnification_bias"),
            ("magnify_flux", "magnify_flux"),
            ("scale_source_sizes", "scale_source_sizes"),
        )
    ),
)


@pytest.mark.parametrize(("module", "name", "new_module", "new_name"), ALIASES)
def test_deprecated_alias_warns_and_resolves(module, name, new_module, new_name):
    old = importlib.import_module(module)
    with pytest.warns(DeprecationWarning, match=rf"{name}.*0\.7.*{new_name}"):
        value = getattr(old, name)
    assert value is getattr(importlib.import_module(new_module), new_name)
    assert name not in getattr(old, "__all__", ())


@pytest.mark.parametrize(
    "module",
    [
        "torchsky.map.field.native_response",
        "torchsky.map.field.native_integrated_read",
        "torchsky.map.field.joint_native_likelihood",
        "torchsky.map.operators.magnification",
    ],
)
def test_deprecated_module_rejects_unknown_names(module):
    with pytest.raises(AttributeError):
        getattr(importlib.import_module(module), "not_a_name")
