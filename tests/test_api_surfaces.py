from __future__ import annotations

import pytest

import torchsky
import torchsky.map.field as field_models

surfaces = torchsky.surfaces


def test_api_surface_tiers_are_code_visible() -> None:
    assert surfaces.get_api_surface("torchsky.read").tier is surfaces.SurfaceTier.CORE
    assert surfaces.get_api_surface("torchsky.read").pillar is surfaces.SurfacePillar.IO
    assert (
        surfaces.get_api_surface("torchsky.sphere.compat").tier
        is surfaces.SurfaceTier.COMPAT
    )
    assert (
        surfaces.get_api_surface("torchsky.sphere.compat").pillar
        is surfaces.SurfacePillar.WORLD
    )
    assert (
        surfaces.get_api_surface("torchsky.map.field").tier
        is surfaces.SurfaceTier.EXPERIMENTAL
    )
    assert (
        surfaces.get_api_surface("torchsky.map.field").pillar
        is surfaces.SurfacePillar.SCENE
    )
    assert (
        surfaces.get_api_surface("torchsky.field").tier
        is surfaces.SurfaceTier.EXPERIMENTAL
    )
    assert (
        surfaces.get_api_surface("torchsky.field").pillar
        is surfaces.SurfacePillar.SCENE
    )


def test_api_surfaces_by_tier_are_immutable_public_metadata() -> None:
    core_names = {surface.name for surface in surfaces.api_surfaces_by_tier("core")}
    assert {"torchsky.read", "torchsky.write", "torchsky.wcs.WCS"} <= core_names

    with pytest.raises(TypeError):
        surfaces.API_SURFACES["new"] = surfaces.get_api_surface("torchsky.read")  # type: ignore[index]


def test_api_surfaces_by_pillar_declare_modelling_spine() -> None:
    io_names = {surface.name for surface in surfaces.api_surfaces_by_pillar("io")}
    world_names = {surface.name for surface in surfaces.api_surfaces_by_pillar("world")}
    scene_names = {surface.name for surface in surfaces.api_surfaces_by_pillar("scene")}
    viz_names = {surface.name for surface in surfaces.api_surfaces_by_pillar("viz")}

    assert {"torchsky.read", "torchsky.write", "torchsky.table"} <= io_names
    assert {"torchsky.wcs.WCS", "torchsky.sphere"} <= world_names
    assert {
        "torchsky.map.coadd_image_plane",
        "torchsky.map.stack_map_product",
        "torchsky.field",
        "torchsky.field.fit",
        "torchsky.map.field",
    } <= scene_names
    assert {"torchsky.viz"} == viz_names


def test_declared_api_surfaces_have_architecture_pillars() -> None:
    assert all(surface.pillar is not None for surface in surfaces.API_SURFACES.values())


def test_field_model_recommendations_are_narrower_than_experimental_zoo() -> None:
    assert field_models.RECOMMENDED_FIELD_MODEL_KINDS
    assert len(field_models.RECOMMENDED_FIELD_MODEL_KINDS) < len(
        field_models.FIELD_MODEL_KINDS
    )
    assert field_models.is_recommended_field_model("sky_lambda_feature_field")
    assert not field_models.is_recommended_field_model("sine_implicit")
