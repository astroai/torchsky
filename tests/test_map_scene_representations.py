from __future__ import annotations

import torch

from torchsky.map.field import PSFSpec
from torchsky.map.scenes import (
    ContinuousFieldModel,
    DiscreteTargetImage,
    HarmonicAlmScene,
    HealpixMapScene,
    SpinHarmonicAlmScene,
    as_scene_representation,
    target_to_field_observation,
)
from torchsky.map.target import TargetRenderSpec
from torchsky.map.field import FieldObservation
from torchsky.map.operators import (
    ObservationOperatorBundle,
    spherical_spectral_operator_bundle,
)


class _OnesField(torch.nn.Module):
    def forward(self, ra, dec, wavelength):
        return torch.ones_like(ra, dtype=torch.float32)


class _DistributionOnesField(_OnesField):
    def forward_distribution(self, ra, dec, wavelength):
        return self.forward(ra, dec, wavelength), torch.zeros_like(ra, dtype=torch.float32)


def _simple_header() -> dict:
    return {
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
    }


def test_discrete_target_image_renders_directly_to_matching_target() -> None:
    data = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    target = TargetRenderSpec.from_header(_simple_header())
    rendered = DiscreteTargetImage(data).render_to_target(target)
    torch.testing.assert_close(rendered, data)


def test_continuous_field_model_uses_field_render_path() -> None:
    target = TargetRenderSpec.from_header(_simple_header())
    obs = FieldObservation(
        image=torch.zeros((4, 4), dtype=torch.float32),
        header=_simple_header(),
        wavelength=5000.0,
    )
    rendered = ContinuousFieldModel(_OnesField()).render_to_target(
        target,
        observation=obs,
        num_spectral_samples=1,
    )
    torch.testing.assert_close(rendered, torch.ones((4, 4), dtype=torch.float32))


def test_continuous_field_model_forwards_operator_bundle() -> None:
    target = TargetRenderSpec.from_header(_simple_header())
    obs = FieldObservation(
        image=torch.zeros((4, 4), dtype=torch.float32),
        header=_simple_header(),
        wavelength=5000.0,
        background_offset=3.0,
    )
    scene = ContinuousFieldModel(_OnesField())

    implicit = scene.render_to_target(target, observation=obs, num_spectral_samples=1)
    explicit = scene.render_to_target(
        target,
        observation=obs,
        num_spectral_samples=1,
        operator_bundle=ObservationOperatorBundle(),
    )

    torch.testing.assert_close(implicit, torch.full((4, 4), 4.0))
    torch.testing.assert_close(explicit, torch.ones((4, 4), dtype=torch.float32))


def test_continuous_field_model_distribution_forwards_operator_bundle() -> None:
    target = TargetRenderSpec.from_header(_simple_header())
    obs = FieldObservation(
        image=torch.zeros((4, 4), dtype=torch.float32),
        header=_simple_header(),
        wavelength=5000.0,
        background_offset=3.0,
    )
    scene = ContinuousFieldModel(_DistributionOnesField())

    mean, logvar = scene.render_distribution_to_target(
        target,
        observation=obs,
        num_spectral_samples=1,
        operator_bundle=ObservationOperatorBundle(),
    )

    torch.testing.assert_close(mean, torch.ones((4, 4), dtype=torch.float32))
    torch.testing.assert_close(logvar, torch.zeros((4, 4), dtype=torch.float32))


def test_target_to_field_observation_preserves_bandpass_and_psf_metadata() -> None:
    target = TargetRenderSpec.from_header(
        _simple_header(),
        band_label="f444w",
        spectral_min=42000.0,
        spectral_max=47000.0,
        transmission_center=44400.0,
        transmission_sigma=800.0,
        psf_spec=PSFSpec.gaussian(1.25),
    )

    observation = target_to_field_observation(target)

    assert observation.band_name == "f444w"
    assert observation.wavelength == 44400.0
    assert observation.spectral_min == 42000.0
    assert observation.spectral_max == 47000.0
    assert observation.psf_spec is target.psf_spec


def test_continuous_field_model_can_render_from_target_spec_alone() -> None:
    target = TargetRenderSpec.from_header(
        _simple_header(),
        transmission_center=5000.0,
    )

    rendered = ContinuousFieldModel(_OnesField()).render_to_target(
        target,
        num_spectral_samples=1,
    )

    torch.testing.assert_close(rendered, torch.ones((4, 4), dtype=torch.float32))


def test_as_scene_representation_coerces_module_and_tensor() -> None:
    assert isinstance(as_scene_representation(_OnesField()), ContinuousFieldModel)
    assert isinstance(
        as_scene_representation(torch.ones((4, 4), dtype=torch.float32)),
        DiscreteTargetImage,
    )


def test_healpix_map_scene_renders_constant_map_through_spherical_bundle() -> None:
    target = TargetRenderSpec.from_header(_simple_header(), band_label="mono")
    npix = 12 * 4 * 4
    scene = HealpixMapScene(
        torch.ones(npix, dtype=torch.float32),
        nside=4,
    )

    rendered = scene.render_to_target(
        target,
        operators=spherical_spectral_operator_bundle(mode="harmonic_direct"),
    )

    torch.testing.assert_close(
        rendered,
        torch.ones((4, 4), dtype=torch.float32),
        atol=5.0e-2,
        rtol=0.0,
    )


def test_harmonic_alm_scene_renders_monopole_constant_field() -> None:
    target = TargetRenderSpec.from_header(_simple_header(), band_label="mono")
    alm = torch.zeros(1, dtype=torch.complex64)
    alm[0] = torch.tensor((4.0 * torch.pi) ** 0.5, dtype=torch.float32)

    rendered = HarmonicAlmScene(alm, lmax=0).render_to_target(target)

    torch.testing.assert_close(
        rendered,
        torch.ones((4, 4), dtype=rendered.dtype),
        atol=1.0e-5,
        rtol=0.0,
    )


def test_spin_harmonic_alm_scene_matches_direct_spin_projection() -> None:
    from torchsky.map.healpix_spectral import sample_spin_alm_on_image_header_spectral
    from torchsky.sphere import alm_size

    target = TargetRenderSpec.from_header(_simple_header(), band_label="pol")
    lmax = 3
    alm_e = torch.zeros(alm_size(lmax), dtype=torch.complex64)
    alm_b = torch.zeros(alm_size(lmax), dtype=torch.complex64)
    alm_e[2] = torch.tensor(0.07 + 0.03j, dtype=torch.complex64)
    alm_b[5] = torch.tensor(-0.04 + 0.02j, dtype=torch.complex64)

    rendered = SpinHarmonicAlmScene(
        alm_e,
        alm_b,
        spin=2,
        lmax=lmax,
    ).render_to_target(target)
    direct = sample_spin_alm_on_image_header_spectral(
        (alm_e, alm_b),
        spin=2,
        target_header=target.header,
        lmax=lmax,
    )

    torch.testing.assert_close(rendered, direct)
