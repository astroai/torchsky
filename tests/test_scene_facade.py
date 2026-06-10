from __future__ import annotations

import torch

import torchsky
from torchsky import scene
from torchsky.map import ImageObservation, MapBandpassSpec, MapProductSpec
from torchsky.map.field import FieldObservation, PSFSpec
from torchsky.map.operators.bundle import (
    field_observation_operator_bundle,
    image_observation_operator_bundle,
)
from torchsky.map.target import TargetRenderSpec


def _simple_header() -> dict:
    return {
        "NAXIS": 2,
        "NAXIS1": 3,
        "NAXIS2": 2,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 2.0,
        "CRPIX2": 1.5,
        "CRVAL1": 180.0,
        "CRVAL2": 0.0,
        "CDELT1": -0.01,
        "CDELT2": 0.01,
    }


def _enabled_slot_types(bundle) -> tuple[tuple[str, type], ...]:
    out = []
    for slot in (
        "geometry",
        "spectral_response",
        "psf",
        "radiometry",
        "background",
        "mask_weight",
        "noise",
    ):
        op = getattr(bundle, slot)
        if op is not None and op.enabled:
            out.append((slot, type(op)))
    return tuple(out)


def test_scene_facade_promotes_existing_concepts() -> None:
    assert torchsky.scene is scene
    assert scene.ImageObservation is ImageObservation
    assert scene.FieldObservation is FieldObservation
    assert scene.MapProductSpec is MapProductSpec
    assert scene.TargetRenderSpec is TargetRenderSpec

    target = scene.target_from_header(_simple_header(), value_semantics="flux")
    assert isinstance(target, TargetRenderSpec)
    assert target.shape == (2, 3)
    assert target.value_semantics == "flux"

    product = scene.product_spec(
        target_header=_simple_header(),
        bands=(MapBandpassSpec(label="r", transmission_center=6500.0),),
    )
    product_target = scene.target_from_product(product)
    assert isinstance(product, MapProductSpec)
    assert product_target.band_label == "r"
    assert product_target.transmission_center == 6500.0

    rendered = scene.render(torch.ones((2, 3)), target)
    torch.testing.assert_close(rendered, torch.ones((2, 3)))


def test_scene_operator_adapter_matches_field_bundle() -> None:
    observation = scene.field_observation(
        image=torch.zeros((2, 3)),
        header=_simple_header(),
        weight=torch.ones((2, 3)),
        mask=torch.ones((2, 3), dtype=torch.bool),
        wavelength=6500.0,
        psf_spec=PSFSpec.gaussian(1.2),
        background_offset=0.5,
    )

    adapter = scene.operator_adapter_for_observation(observation)
    direct = field_observation_operator_bundle(observation)

    assert _enabled_slot_types(adapter.bundle) == _enabled_slot_types(direct)
    assert adapter.enabled_names() == tuple(op.name for op in direct.iter_enabled())
    assert adapter.has("spectral_response")
    assert adapter.has("psf")
    assert adapter.has("background")
    assert adapter.has("mask_weight")
    assert adapter.has("noise")


def test_scene_operator_adapter_matches_image_bundle() -> None:
    observation = scene.image_observation(
        path="science.fits",
        wavelength=6500.0,
        photometric_scale=2.0,
        background_offset=0.5,
        weight_path="weight.fits",
        mask_path="mask.fits",
    )

    bundle = scene.operator_bundle_for_observation(
        observation,
        resampling_kernel="bilinear",
    )
    direct = image_observation_operator_bundle(
        observation,
        resampling_kernel="bilinear",
    )

    assert isinstance(observation, ImageObservation)
    assert _enabled_slot_types(bundle) == _enabled_slot_types(direct)
