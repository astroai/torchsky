"""Tiled WCS rendering, HEALPix-native field cubes, and torch.func vmap evaluation."""

from __future__ import annotations

import torch

from torchsky.map.field import (
    FieldDomain,
    FieldObservation,
    SkyEncoderMLPField,
    evaluate_field_coordinates_vmapped,
    render_field_cube,
    render_field_cube_via_healpix,
    render_field_healpix,
    render_field_observation,
)
from torchsky.spectral import SpectralAxis


def _tan_header(
    size: int,
    *,
    crval1: float,
    crval2: float,
    scale_deg: float,
) -> dict:
    return {
        "NAXIS": 2,
        "NAXIS1": size,
        "NAXIS2": size,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": size / 2.0,
        "CRPIX2": size / 2.0,
        "CRVAL1": crval1,
        "CRVAL2": crval2,
        "CD1_1": -scale_deg,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": scale_deg,
    }


def test_render_field_cube_spatial_tile_matches_full() -> None:
    torch.manual_seed(0)
    domain = FieldDomain(
        center_ra_deg=10.0,
        center_dec_deg=0.0,
        spatial_scale_deg=1.0,
        wavelength_center=5000.0,
        wavelength_scale=100.0,
        coord_mode="sphere_xyz",
    )
    model = SkyEncoderMLPField(
        domain,
        hidden_dim=16,
        hidden_layers=2,
        scale_fourier_to_domain=False,
    )
    header = _tan_header(24, crval1=10.0, crval2=0.0, scale_deg=0.02)
    axis = SpectralAxis(
        values=torch.tensor([5000.0, 5100.0], dtype=torch.float64),
        unit="Angstrom",
        type="WAVE",
    )
    full = render_field_cube(model, target_header=header, spectral_axis=axis)
    tiled = render_field_cube(
        model,
        target_header=header,
        spectral_axis=axis,
        spatial_tile_hw=(7, 11),
    )
    assert full.shape == tiled.shape
    assert torch.allclose(full, tiled, rtol=0.0, atol=1.0e-5)


def test_render_field_cube_via_healpix_tan_shape_and_finite() -> None:
    torch.manual_seed(11)
    domain = FieldDomain(
        center_ra_deg=100.0,
        center_dec_deg=5.0,
        spatial_scale_deg=0.5,
        wavelength_center=5500.0,
        wavelength_scale=200.0,
        coord_mode="sphere_xyz",
    )
    model = SkyEncoderMLPField(
        domain,
        hidden_dim=14,
        hidden_layers=2,
        scale_fourier_to_domain=False,
    )
    header = _tan_header(14, crval1=100.0, crval2=5.0, scale_deg=0.03)
    axis = SpectralAxis(
        values=torch.tensor([5400.0, 5600.0], dtype=torch.float64),
        unit="Angstrom",
        type="WAVE",
    )
    cube = render_field_cube_via_healpix(
        model,
        target_header=header,
        spectral_axis=axis,
        nside=8,
        nest=False,
        healpix_spatial_chunk_pixels=256,
    )
    assert cube.shape == (2, 14, 14)
    assert torch.isfinite(cube).all()


def test_render_field_cube_via_healpix_car_projection() -> None:
    torch.manual_seed(12)
    domain = FieldDomain(
        center_ra_deg=180.0,
        center_dec_deg=0.0,
        spatial_scale_deg=2.0,
        wavelength_center=4000.0,
        wavelength_scale=400.0,
        coord_mode="sphere_xyz",
    )
    model = SkyEncoderMLPField(
        domain,
        hidden_dim=12,
        hidden_layers=2,
        scale_fourier_to_domain=False,
    )
    header = {
        "NAXIS": 2,
        "NAXIS1": 10,
        "NAXIS2": 8,
        "CTYPE1": "RA---CAR",
        "CTYPE2": "DEC--CAR",
        "CRPIX1": 5.0,
        "CRPIX2": 4.0,
        "CRVAL1": 180.0,
        "CRVAL2": 0.0,
        "CDELT1": -0.15,
        "CDELT2": 0.15,
    }
    axis = SpectralAxis(
        values=torch.tensor([4000.0], dtype=torch.float64),
        unit="Angstrom",
        type="WAVE",
    )
    cube = render_field_cube_via_healpix(
        model,
        target_header=header,
        spectral_axis=axis,
        nside=4,
        nest=True,
    )
    assert cube.shape == (1, 8, 10)
    assert torch.isfinite(cube).all()


def test_render_field_healpix_shape_and_finite() -> None:
    torch.manual_seed(1)
    domain = FieldDomain(
        center_ra_deg=0.0,
        center_dec_deg=0.0,
        spatial_scale_deg=30.0,
        wavelength_center=4000.0,
        wavelength_scale=500.0,
        coord_mode="sphere_xyz",
    )
    model = SkyEncoderMLPField(
        domain,
        hidden_dim=12,
        hidden_layers=2,
        scale_fourier_to_domain=False,
    )
    axis = SpectralAxis(
        values=torch.tensor([4000.0, 4500.0], dtype=torch.float64),
        unit="Angstrom",
        type="WAVE",
    )
    cube = render_field_healpix(
        model,
        nside=1,
        spectral_axis=axis,
        nest=False,
        spatial_chunk_pixels=6,
    )
    assert cube.shape == (2, 12)
    assert torch.isfinite(cube).all()


def test_evaluate_field_coordinates_vmapped_matches_batched() -> None:
    torch.manual_seed(2)
    domain = FieldDomain(
        center_ra_deg=50.0,
        center_dec_deg=1.0,
        spatial_scale_deg=0.5,
        wavelength_center=6000.0,
        wavelength_scale=200.0,
        coord_mode="sphere_xyz",
    )
    model = SkyEncoderMLPField(
        domain,
        hidden_dim=20,
        hidden_layers=2,
        scale_fourier_to_domain=False,
    )
    ra = torch.linspace(50.0, 50.2, 32, dtype=torch.float32)
    dec = torch.linspace(1.0, 1.1, 32, dtype=torch.float32)
    wl = torch.full((32,), 6000.0, dtype=torch.float32)
    batched = model(ra, dec, wl)
    vmapped = evaluate_field_coordinates_vmapped(model, ra, dec, wl)
    assert batched.shape == vmapped.shape
    assert torch.allclose(batched, vmapped, rtol=1.0e-4, atol=1.0e-4)


class _BroadcastField(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float64))

    def forward(self, ra, dec, wavelength):
        ra, dec, wavelength = torch.broadcast_tensors(ra, dec, wavelength)
        return self.scale * ra + dec - wavelength / 1000.0


def test_evaluate_field_coordinates_vmapped_broadcasts_arbitrary_shapes() -> None:
    model = _BroadcastField()
    cases = [
        (
            torch.tensor(2.0, dtype=torch.float64),
            torch.tensor(3.0, dtype=torch.float64),
            torch.tensor(4000.0, dtype=torch.float64),
        ),
        (
            torch.arange(5, dtype=torch.float64),
            torch.tensor(1.0, dtype=torch.float64),
            torch.tensor(5000.0, dtype=torch.float64),
        ),
        (
            torch.arange(12, dtype=torch.float64).reshape(3, 4).T,
            torch.arange(4, dtype=torch.float64)[:, None],
            torch.tensor([5000.0, 5100.0, 5200.0])[None, :],
        ),
        (
            torch.arange(6, dtype=torch.float64).reshape(2, 1, 3),
            torch.arange(4, dtype=torch.float64).reshape(1, 4, 1),
            torch.tensor(5000.0, dtype=torch.float64),
        ),
        (
            torch.empty((0, 3), dtype=torch.float64),
            torch.empty((0, 1), dtype=torch.float64),
            torch.tensor(5000.0, dtype=torch.float64),
        ),
    ]
    for ra, dec, wavelength in cases:
        expected = model(ra, dec, wavelength)
        actual = evaluate_field_coordinates_vmapped(model, ra, dec, wavelength)
        assert actual.shape == expected.shape
        torch.testing.assert_close(actual, expected)


def test_evaluate_field_coordinates_vmapped_preserves_parameter_gradients() -> None:
    model = _BroadcastField()
    ra = torch.arange(6, dtype=torch.float64).reshape(2, 3)
    output = evaluate_field_coordinates_vmapped(
        model,
        ra,
        torch.zeros(3, dtype=torch.float64),
        torch.tensor(5000.0, dtype=torch.float64),
    )
    output.sum().backward()
    torch.testing.assert_close(model.scale.grad, ra.sum())


def test_render_field_observation_spatial_tile_matches_full() -> None:
    torch.manual_seed(3)
    domain = FieldDomain(
        center_ra_deg=200.0,
        center_dec_deg=-5.0,
        spatial_scale_deg=0.2,
        wavelength_center=7000.0,
        wavelength_scale=300.0,
        coord_mode="sphere_xyz",
    )
    model = SkyEncoderMLPField(
        domain,
        hidden_dim=14,
        hidden_layers=2,
        scale_fourier_to_domain=False,
    )
    header = _tan_header(18, crval1=200.0, crval2=-5.0, scale_deg=0.01)
    obs = FieldObservation(
        image=torch.zeros((18, 18), dtype=torch.float32),
        header=header,
        wavelength=7000.0,
        spectral_min=6900.0,
        spectral_max=7100.0,
    )
    a = render_field_observation(model, obs, num_spectral_samples=5)
    obs.clear_coordinate_cache()
    b = render_field_observation(
        model,
        obs,
        num_spectral_samples=5,
        spatial_tile_hw=(6, 9),
    )
    assert obs._cached_ra_deg is None
    assert obs._cached_dec_deg is None
    c = render_field_observation(
        model,
        obs,
        num_spectral_samples=5,
        use_func_vmap=True,
        func_vmap_chunk=37,
    )
    assert torch.allclose(a, b, rtol=0.0, atol=1.0e-5)
    assert torch.allclose(a, c, rtol=0.0, atol=1.0e-5)


def test_render_field_observation_tile_matches_full_with_psf() -> None:
    domain = FieldDomain(
        center_ra_deg=0.0,
        center_dec_deg=0.0,
        spatial_scale_deg=1.0,
        wavelength_center=5000.0,
        wavelength_scale=100.0,
        coord_mode="sphere_xyz",
    )
    model = SkyEncoderMLPField(domain, hidden_dim=8, hidden_layers=1)
    header = _tan_header(8, crval1=0.0, crval2=0.0, scale_deg=0.1)
    obs = FieldObservation(
        image=torch.zeros((8, 8), dtype=torch.float32),
        header=header,
        wavelength=5000.0,
        psf_sigma_pix=1.5,
    )
    full = render_field_observation(model, obs, num_spectral_samples=3)
    tiled = render_field_observation(
        model,
        obs,
        num_spectral_samples=3,
        spatial_tile_hw=(4, 4),
    )

    assert torch.allclose(full, tiled, rtol=0.0, atol=1.0e-6)
    full.sum().backward()
    full_grads = [parameter.grad.detach().clone() for parameter in model.parameters()]
    model.zero_grad(set_to_none=True)
    tiled.sum().backward()
    for parameter, expected in zip(model.parameters(), full_grads, strict=True):
        torch.testing.assert_close(parameter.grad, expected, rtol=1.0e-5, atol=1.0e-6)
