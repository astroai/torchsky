from __future__ import annotations

import torch

import torchsky
from torchsky import scene
from torchsky.map import BandpassSpec, ProductSpec
from torchsky.map.observation import ImageObservation
from torchsky.map.field import FieldObservation, PSFSpec
from torchsky.map.operators.bundle import (
    field_observation_operator_bundle,
    image_observation_operator_bundle,
)
from torchsky.map.operators.resampling import sample_image_plane_batched_channels
from torchsky.map.splat import splat_to_native
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
    assert scene.ProductSpec is ProductSpec
    assert scene.BandpassSpec is BandpassSpec
    assert scene.TargetRenderSpec is TargetRenderSpec
    assert (
        scene.sample_image_plane_batched_channels is sample_image_plane_batched_channels
    )
    assert scene.splat_to_native is splat_to_native

    target = scene.target_from_header(_simple_header(), value_semantics="flux")
    assert isinstance(target, TargetRenderSpec)
    assert target.shape == (2, 3)
    assert target.value_semantics == "flux"

    product = scene.product_spec(
        target_header=_simple_header(),
        bands=(BandpassSpec(label="r", transmission_center=6500.0),),
    )
    product_target = scene.target_from_product(product)
    assert isinstance(product, ProductSpec)
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


def test_scene_operator_bundles_for_observations_preserves_order() -> None:
    field_obs = scene.field_observation(
        image=torch.zeros((2, 3)),
        header=_simple_header(),
        wavelength=6500.0,
        background_offset=0.5,
    )
    image_obs = scene.image_observation(
        path="science.fits",
        photometric_scale=2.0,
        weight_path="weight.fits",
    )

    bundles = scene.operator_bundles_for_observations(
        (field_obs, image_obs),
        resampling_kernel="bilinear",
    )

    assert len(bundles) == 2
    assert bundles[0].background is not None
    assert bundles[1].radiometry is not None
    assert bundles[1].mask_weight is not None


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


def test_scene_operator_adapter_binds_exact_adjoint_operator() -> None:
    observation = scene.field_observation(
        image=torch.zeros((2, 3)),
        header=_simple_header(),
        wavelength=6500.0,
    )
    adapter = scene.operator_adapter_for_observation(observation)
    target = scene.target_from_header(_simple_header())
    operator = adapter.bind_target(
        target,
        apply_operator=lambda scene_rep, target_spec: (
            scene_rep.render_to_target(target_spec) * 3.0
        ),
        apply_adjoint=lambda tensor: torch.as_tensor(tensor) * 3.0,
    )

    x = torch.arange(6, dtype=torch.float64).reshape(2, 3)
    y = torch.linspace(-2.0, 1.0, 6, dtype=torch.float64).reshape(2, 3)

    lhs = torch.sum(operator.apply(x) * y)
    rhs = torch.sum(x * operator.adjoint(y))

    torch.testing.assert_close(lhs, rhs)


def test_scene_operator_adapter_normal_equation_probe_and_cg() -> None:
    observation = scene.field_observation(
        image=torch.zeros((2, 3)),
        header=_simple_header(),
        wavelength=6500.0,
    )
    adapter = scene.operator_adapter_for_observation(observation)
    target = scene.target_from_header(_simple_header())

    def _apply_operator(scene_rep, target_spec):
        return scene_rep.render_to_target(target_spec) * 2.0

    def _apply_adjoint(tensor):
        return torch.as_tensor(tensor) * 2.0

    model = torch.ones((2, 3), dtype=torch.float32)
    data = torch.full((2, 3), 6.0, dtype=torch.float32)
    probe = adapter.normal_equation_probe(
        model,
        target,
        data,
        apply_operator=_apply_operator,
        apply_adjoint=_apply_adjoint,
    )

    torch.testing.assert_close(probe.prediction, torch.full((2, 3), 2.0))
    torch.testing.assert_close(probe.lhs_probe, torch.full((2, 3), 4.0))
    torch.testing.assert_close(probe.rhs, torch.full((2, 3), 12.0))

    operator = adapter.bind_target(
        target,
        apply_operator=_apply_operator,
        apply_adjoint=_apply_adjoint,
    )
    result = operator.solve_normal_equation_cg(data, iterations=2)

    torch.testing.assert_close(result.solution, torch.full((2, 3), 3.0))


def test_scene_image_plane_target_binding_uses_projection_and_exact_adjoint() -> None:
    observation = scene.image_observation(path="unused.fits")
    adapter = scene.operator_adapter_for_observation(
        observation,
        resampling_kernel="bilinear",
    )
    target = scene.target_from_header(_simple_header())
    operator = adapter.bind_image_plane_target(
        target,
        observation_header=_simple_header(),
    )

    x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    y = torch.linspace(-1.0, 2.0, 6, dtype=torch.float32).reshape(2, 3)
    prediction = operator.apply(x)

    torch.testing.assert_close(prediction, x, rtol=1.0e-5, atol=1.0e-5)
    lhs = torch.sum(prediction * y)
    rhs = torch.sum(x * operator.adjoint(y))
    torch.testing.assert_close(lhs, rhs, rtol=1.0e-5, atol=1.0e-5)


def test_scene_image_plane_target_binding_normal_equation_recovers_same_grid() -> None:
    observation = scene.image_observation(path="unused.fits")
    adapter = scene.operator_adapter_for_observation(
        observation,
        resampling_kernel="bilinear",
    )
    target = scene.target_from_header(_simple_header())
    operator = adapter.bind_image_plane_target(
        target,
        observation_header=_simple_header(),
    )
    data = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    probe = operator.normal_equation_probe(torch.zeros_like(data), data)
    result = operator.solve_normal_equation_cg(data, iterations=2)

    torch.testing.assert_close(probe.rhs, data, rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(result.solution, data, rtol=1.0e-5, atol=1.0e-5)


def test_scene_image_plane_observation_set_sums_normal_equations() -> None:
    observations = (
        scene.image_observation(path="unused-a.fits"),
        scene.image_observation(path="unused-b.fits"),
    )
    target = scene.target_from_header(_simple_header())
    operator_set = scene.bind_image_plane_observations_to_target(
        observations,
        target,
        observation_headers=(_simple_header(), _simple_header()),
        resampling_kernel="bilinear",
    )
    data_a = torch.ones((2, 3), dtype=torch.float32)
    data_b = torch.full((2, 3), 3.0, dtype=torch.float32)

    probe = operator_set.normal_equation_probe(torch.zeros((2, 3)), (data_a, data_b))
    result = operator_set.solve_normal_equation_cg((data_a, data_b), iterations=2)

    torch.testing.assert_close(probe.rhs, torch.full((2, 3), 4.0))
    torch.testing.assert_close(result.solution, torch.full((2, 3), 2.0))


def test_scene_image_plane_observation_set_applies_scalar_inverse_noise() -> None:
    observations = (
        scene.image_observation(path="unused-a.fits", weight=1.0),
        scene.image_observation(path="unused-b.fits", weight=3.0),
    )
    target = scene.target_from_header(_simple_header())
    operator_set = scene.bind_image_plane_observations_to_target(
        observations,
        target,
        observation_headers=(_simple_header(), _simple_header()),
        resampling_kernel="bilinear",
    )
    data_a = torch.ones((2, 3), dtype=torch.float32)
    data_b = torch.full((2, 3), 3.0, dtype=torch.float32)

    probe = operator_set.normal_equation_probe(torch.zeros((2, 3)), (data_a, data_b))
    result = operator_set.solve_normal_equation_cg((data_a, data_b), iterations=2)

    torch.testing.assert_close(probe.rhs, torch.full((2, 3), 10.0))
    torch.testing.assert_close(result.solution, torch.full((2, 3), 2.5))
