from __future__ import annotations

import pytest
import torch

import torchsky.map as map_api
from torchsky.map.field import FieldObservation, PSFSpec
from torchsky.map.observation import ImageObservation
from torchsky.map.operators import (
    ConservativePolygonRemapOp,
    ExplicitBackgroundOp,
    ExplicitNoiseOp,
    ExplicitPSFOp,
    ForwardSplatGeometryOp,
    ExplicitRadiometryOp,
    ExplicitSpectralResponseOp,
    InterpolationGeometryOp,
    ObservationOperatorBundle,
    bind_image_plane_target_operator,
    conservative_polygon_project,
    field_observation_operator_bundle,
    geometry_op_for_kernel,
    image_observation_operator_bundle,
    make_minimal_learned_operator_scaffold,
    merge_operator_bundles,
    spherical_spectral_operator_bundle,
    swap_bundle_operator_families,
)
from torchsky.map.operators import geometry_conservative as geometry_conservative_mod
from torchsky.map.target import TargetRenderSpec


def _simple_header() -> dict[str, float | int | str]:
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


def test_observation_operator_bundle_iter_enabled_skips_disabled_entries() -> None:
    # The bundle is a canonical operators-module type, not a map-facade alias.
    assert ObservationOperatorBundle.__module__.startswith("torchsky.map.operators")
    assert "ObservationOperatorBundle" not in map_api.__all__
    bundle = ObservationOperatorBundle(
        geometry=InterpolationGeometryOp(),
        spectral_response=ExplicitSpectralResponseOp(),
        psf=ExplicitPSFOp(),
        radiometry=ExplicitRadiometryOp(),
        background=ExplicitBackgroundOp(),
        noise=ExplicitNoiseOp(),
    )
    assert [op.name for op in bundle.iter_enabled()] == [
        "interpolation_geometry",
        "explicit_spectral_response",
        "explicit_psf",
        "explicit_radiometry",
        "explicit_background",
        "explicit_noise",
    ]
    assert bundle.enabled_operator_names() == {
        "geometry": "interpolation_geometry",
        "spectral_response": "explicit_spectral_response",
        "psf": "explicit_psf",
        "radiometry": "explicit_radiometry",
        "background": "explicit_background",
        "noise": "explicit_noise",
    }
    assert bundle.psf is not None
    bundle.psf.enabled = False
    assert [op.name for op in bundle.iter_enabled()] == [
        "interpolation_geometry",
        "explicit_spectral_response",
        "explicit_radiometry",
        "explicit_background",
        "explicit_noise",
    ]
    assert "psf" not in bundle.enabled_operator_names()


def test_minimal_learned_operator_scaffold_declares_shared_conditioning() -> None:
    scaffold = make_minimal_learned_operator_scaffold()
    assert scaffold.geometry.learned
    assert scaffold.background.learned
    assert scaffold.psf.conditioning.detector_coordinates
    assert scaffold.spectral_response.conditioning.wavelength


def test_geometry_op_for_kernel_covers_interpolation_drizzle_splat_and_fft() -> None:
    assert geometry_op_for_kernel("bilinear").name == "interpolation_geometry"
    assert geometry_op_for_kernel("drizzle_square").name == "exact_drizzle_geometry"
    assert (
        geometry_op_for_kernel("conservative_polygon").name
        == "conservative_polygon_geometry"
    )
    assert geometry_op_for_kernel("splat_box").name == "forward_splat_geometry"
    assert geometry_op_for_kernel("fft_shift").name == "restricted_affine_fft_geometry"
    assert (
        geometry_op_for_kernel("spectral_nonuniform").name == "spectral_global_geometry"
    )
    assert (
        geometry_op_for_kernel("spectral_nonuniform_fast").name
        == "spectral_global_geometry"
    )


def test_image_observation_operator_bundle_normalizes_metadata_into_operator_slots() -> (
    None
):
    observation = ImageObservation(
        "mock.fits",
        weight=2.0,
        photometric_scale=1.5,
        background_offset=0.25,
        transmission_center=7000.0,
        transmission_sigma=150.0,
        weight_path="weights.fits",
    )

    bundle = image_observation_operator_bundle(
        observation,
        resampling_kernel="splat_box",
    )

    assert isinstance(bundle.geometry, ForwardSplatGeometryOp)
    assert isinstance(bundle.spectral_response, ExplicitSpectralResponseOp)
    assert isinstance(bundle.radiometry, ExplicitRadiometryOp)
    assert isinstance(bundle.background, ExplicitBackgroundOp)
    assert bundle.mask_weight is not None
    assert isinstance(bundle.noise, ExplicitNoiseOp)


def test_spherical_spectral_operator_bundle_declares_spherical_geometry_mode() -> None:
    bundle = spherical_spectral_operator_bundle(mode="healpix_bridge")
    assert bundle.geometry is not None
    assert bundle.geometry.name == "spherical_spectral_geometry"
    assert getattr(bundle.geometry, "mode", None) == "healpix_bridge"


def test_field_observation_operator_bundle_normalizes_psf_background_and_noise() -> (
    None
):
    observation = FieldObservation(
        image=torch.zeros((4, 4), dtype=torch.float32),
        header={
            "NAXIS": 2,
            "NAXIS1": 4,
            "NAXIS2": 4,
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
            "CRPIX1": 2.0,
            "CRPIX2": 2.0,
            "CRVAL1": 180.0,
            "CRVAL2": 0.0,
            "CDELT1": -0.01,
            "CDELT2": 0.01,
        },
        wavelength=5000.0,
        weight=torch.ones((4, 4), dtype=torch.float32),
        mask=torch.ones((4, 4), dtype=torch.float32),
        psf_spec=PSFSpec.gaussian(1.0),
        background_offset=0.2,
    )

    bundle = field_observation_operator_bundle(observation)

    assert bundle.geometry is None
    assert isinstance(bundle.spectral_response, ExplicitSpectralResponseOp)
    assert isinstance(bundle.psf, ExplicitPSFOp)
    assert isinstance(bundle.background, ExplicitBackgroundOp)
    assert bundle.mask_weight is not None
    assert isinstance(bundle.noise, ExplicitNoiseOp)


def test_field_observation_operator_bundle_detects_structured_background() -> None:
    observation = FieldObservation(
        image=torch.zeros((4, 4), dtype=torch.float32),
        header={
            "NAXIS": 2,
            "NAXIS1": 4,
            "NAXIS2": 4,
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
            "CRPIX1": 2.0,
            "CRPIX2": 2.0,
            "CRVAL1": 180.0,
            "CRVAL2": 0.0,
            "CDELT1": -0.01,
            "CDELT2": 0.01,
        },
        wavelength=5000.0,
        sky_background=torch.ones((4, 4), dtype=torch.float32),
    )

    bundle = field_observation_operator_bundle(observation)

    assert isinstance(bundle.background, ExplicitBackgroundOp)


def test_merge_operator_bundles_prefers_learned_operator_when_present() -> None:
    learned = make_minimal_learned_operator_scaffold()
    merged = merge_operator_bundles(
        ObservationOperatorBundle(background=ExplicitBackgroundOp()),
        ObservationOperatorBundle(background=learned.background),
    )
    assert merged.background is learned.background


def test_swap_bundle_operator_families_replaces_selected_slots_only() -> None:
    learned = make_minimal_learned_operator_scaffold()
    base = ObservationOperatorBundle(
        geometry=InterpolationGeometryOp(),
        spectral_response=ExplicitSpectralResponseOp(),
        background=ExplicitBackgroundOp(),
        noise=ExplicitNoiseOp(),
    )

    swapped = swap_bundle_operator_families(
        base,
        learned,
        families=("background", "noise"),
    )

    assert swapped.geometry is base.geometry
    assert swapped.spectral_response is base.spectral_response
    assert swapped.background is learned.background
    assert swapped.noise is learned.noise


def test_conservative_polygon_op_and_helper_use_full_footprint_contract(
    monkeypatch,
) -> None:
    calls: dict[str, object] = {}

    def _fake_conservative(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return "numer", "weight"

    monkeypatch.setattr(
        geometry_conservative_mod,
        "_exact_conservative_polygon_project",
        _fake_conservative,
    )
    op = ConservativePolygonRemapOp()
    assert op.name == "conservative_polygon_geometry"

    numer, weight = conservative_polygon_project(
        "image",
        "source_weight",
        "source_wcs",
        "target",
        value_semantics="surface_brightness",
        pixel_area_sr_map_fn="pixel_area_fn",
        chunk_pixels=1024,
    )

    assert (numer, weight) == ("numer", "weight")
    assert calls["args"] == ("image", "source_weight", "source_wcs", "target")
    assert calls["kwargs"]["chunk_pixels"] == 1024


def test_image_plane_target_operator_binder_exposes_exact_adjoint() -> None:
    target = TargetRenderSpec.from_header(_simple_header())
    operator = bind_image_plane_target_operator(
        ImageObservation("unused.fits"),
        target,
        observation_header=_simple_header(),
        resampling_kernel="bilinear",
    )
    state = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    dual = torch.linspace(-1.0, 2.0, 6, dtype=torch.float32).reshape(2, 3)

    prediction = operator.apply(state)
    lhs = torch.sum(prediction * dual)
    rhs = torch.sum(state * operator.adjoint(dual))

    torch.testing.assert_close(prediction, state, rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(lhs, rhs, rtol=1.0e-5, atol=1.0e-5)

    with pytest.raises(ValueError, match="adjoint input shape"):
        operator.adjoint(torch.ones((1, 2, 3), dtype=torch.float32))


def test_image_plane_target_operator_noise_is_shape_checked_and_nonnegative() -> None:
    target = TargetRenderSpec.from_header(_simple_header())
    negative_weight_operator = bind_image_plane_target_operator(
        ImageObservation("unused.fits", weight=-2.0),
        target,
        observation_header=_simple_header(),
        resampling_kernel="bilinear",
    )

    data = torch.ones((2, 3), dtype=torch.float32)
    torch.testing.assert_close(
        negative_weight_operator.apply_noise_model(data),
        torch.zeros_like(data),
    )

    with pytest.raises(ValueError, match="at least two image axes"):
        negative_weight_operator.apply_noise_model(torch.ones(3, dtype=torch.float32))

    positive_weight_operator = bind_image_plane_target_operator(
        ImageObservation("unused.fits", weight=2.0),
        target,
        observation_header=_simple_header(),
        resampling_kernel="bilinear",
    )
    torch.testing.assert_close(
        positive_weight_operator.apply_noise_model(torch.ones((1, 3))),
        torch.full((1, 3), 2.0),
    )
    torch.testing.assert_close(
        positive_weight_operator.apply_noise_model(data),
        torch.full_like(data, 2.0),
    )
