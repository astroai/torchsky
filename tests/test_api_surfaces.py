from __future__ import annotations

import pytest

import torchsky
import torchsky.map as tmap


def test_api_surface_tiers_are_code_visible() -> None:
    assert torchsky.get_api_surface("torchsky.read").tier is torchsky.SurfaceTier.CORE
    assert (
        torchsky.get_api_surface("torchsky.sphere.compat").tier
        is torchsky.SurfaceTier.COMPAT
    )
    assert (
        torchsky.get_api_surface("torchsky.map.field").tier
        is torchsky.SurfaceTier.EXPERIMENTAL
    )


def test_api_surfaces_by_tier_are_immutable_public_metadata() -> None:
    core_names = {surface.name for surface in torchsky.api_surfaces_by_tier("core")}
    assert {"torchsky.read", "torchsky.write", "torchsky.wcs.WCS"} <= core_names

    with pytest.raises(TypeError):
        torchsky.API_SURFACES["new"] = torchsky.get_api_surface("torchsky.read")  # type: ignore[index]


def test_field_model_recommendations_are_narrower_than_experimental_zoo() -> None:
    assert tmap.RECOMMENDED_FIELD_MODEL_KINDS
    assert len(tmap.RECOMMENDED_FIELD_MODEL_KINDS) < len(tmap.field.FIELD_MODEL_KINDS)
    assert tmap.is_recommended_field_model("sky_lambda_feature_field")
    assert not tmap.is_recommended_field_model("sine_implicit")
