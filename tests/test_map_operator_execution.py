from __future__ import annotations

import torch

from torchsky.map import ImageObservation
from torchsky.map.operators import (
    ConservativePolygonRemapOp,
    ExactDrizzleGeometryOp,
    ForwardSplatGeometryOp,
    InterpolationGeometryOp,
    prepare_image_observation_data,
    project_image_observation_geometry,
)
from torchsky.map.operators import execution as execution_mod


def test_prepare_image_observation_data_uses_weight_and_calibration_callbacks() -> None:
    calls: dict[str, object] = {}

    def _load_weight_map(obs, image, *, device):
        calls["load"] = (obs, tuple(image.shape), device)
        return torch.full_like(image, 2.0, dtype=torch.float32)

    def _apply_calibration(obs, image, weight):
        calls["calibration"] = (obs, tuple(image.shape), float(weight.mean().item()))
        return image.to(dtype=torch.float32) + 3.0, weight * 5.0

    observation = ImageObservation(
        "mock.fits",
        weight_path="weights.fits",
        photometric_scale=1.5,
    )
    image = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)

    prepared = prepare_image_observation_data(
        observation,
        image,
        device=torch.device("cpu"),
        resampling_kernel="bilinear",
        load_weight_map_fn=_load_weight_map,
        apply_calibration_fn=_apply_calibration,
    )

    assert "load" in calls
    assert "calibration" in calls
    torch.testing.assert_close(prepared.image, image + 3.0)
    torch.testing.assert_close(prepared.source_weight, torch.full_like(image, 10.0))
    assert prepared.operator_bundle.radiometry is not None


def test_prepare_image_observation_data_sanitizes_nan_when_no_calibration_needed() -> None:
    image = torch.tensor([[1.0, float("nan")]], dtype=torch.float32)
    prepared = prepare_image_observation_data(
        ImageObservation("mock.fits"),
        image,
        device=torch.device("cpu"),
        resampling_kernel="bilinear",
        load_weight_map_fn=lambda obs, arr, *, device: torch.ones_like(arr),
        apply_calibration_fn=lambda obs, arr, w: (arr, w),
    )

    torch.testing.assert_close(prepared.image, torch.tensor([[1.0, 0.0]]))
    torch.testing.assert_close(prepared.source_weight, torch.tensor([[1.0, 0.0]]))


def test_project_image_observation_geometry_dispatches_by_operator_family(
    monkeypatch,
) -> None:
    calls: list[str] = []

    def _fake_square(*args, **kwargs):
        calls.append("drizzle_square")
        return "numer", "weight"

    def _fake_point(*args, **kwargs):
        calls.append("drizzle_point")
        return "numer", "weight"

    def _fake_conservative(*args, **kwargs):
        calls.append("conservative")
        return "numer", "weight"

    monkeypatch.setattr(execution_mod, "exact_square_drizzle_project", _fake_square)
    monkeypatch.setattr(execution_mod, "exact_point_drizzle_project", _fake_point)
    monkeypatch.setattr(execution_mod, "conservative_polygon_project", _fake_conservative)

    resample_calls: list[str] = []
    splat_calls: list[str] = []

    def resample_fn(*args, **kwargs):
        resample_calls.append("interp")
        return "n", "w"

    def splat_fn(*args, **kwargs):
        splat_calls.append("splat")
        return "n", "w"

    assert project_image_observation_geometry(
        "image",
        "weight",
        "source_wcs",
        "target",
        geometry=ExactDrizzleGeometryOp(),
        resampling_kernel="drizzle_square",
        value_semantics="surface_brightness",
        drizzle_pixfrac=1.0,
        pixel_area_sr_map_fn="area",
        representative_pixel_area_fn="repr",
        chunk_pixels=256,
        resample_weighted_to_target_fn=resample_fn,
        forward_project_weighted_to_target_fn=splat_fn,
    ) == ("numer", "weight")
    assert project_image_observation_geometry(
        "image",
        "weight",
        "source_wcs",
        "target",
        geometry=ConservativePolygonRemapOp(),
        resampling_kernel="conservative_polygon",
        value_semantics="surface_brightness",
        drizzle_pixfrac=1.0,
        pixel_area_sr_map_fn="area",
        representative_pixel_area_fn="repr",
        chunk_pixels=256,
        resample_weighted_to_target_fn=resample_fn,
        forward_project_weighted_to_target_fn=splat_fn,
    ) == ("numer", "weight")
    assert project_image_observation_geometry(
        "image",
        "weight",
        "source_wcs",
        "target",
        geometry=ForwardSplatGeometryOp(),
        resampling_kernel="splat_box",
        value_semantics="surface_brightness",
        drizzle_pixfrac=1.0,
        pixel_area_sr_map_fn="area",
        representative_pixel_area_fn="repr",
        chunk_pixels=256,
        resample_weighted_to_target_fn=resample_fn,
        forward_project_weighted_to_target_fn=splat_fn,
    ) == ("n", "w")
    assert project_image_observation_geometry(
        "image",
        "weight",
        "source_wcs",
        "target",
        geometry=InterpolationGeometryOp(),
        resampling_kernel="bilinear",
        value_semantics="surface_brightness",
        drizzle_pixfrac=1.0,
        pixel_area_sr_map_fn="area",
        representative_pixel_area_fn="repr",
        chunk_pixels=256,
        resample_weighted_to_target_fn=resample_fn,
        forward_project_weighted_to_target_fn=splat_fn,
    ) == ("n", "w")

    assert calls == ["drizzle_square", "conservative"]
    assert splat_calls == ["splat"]
    assert resample_calls == ["interp"]
