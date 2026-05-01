import numpy as np
import pytest
import torch
import astropy.units as u
from astropy.wcs import WCS as AstropyWCS

import torchsky
from torchsky.wcs.spectral import (
    air_wavelength_to_vacuum,
    convert_spectral,
    frequency_to_spectral,
    spectral_to_frequency,
    transform_spectral,
    vacuum_wavelength_to_air,
)


def _wave_1d_header() -> dict[str, float | int | str]:
    return {
        "NAXIS": 1,
        "CTYPE1": "WAVE",
        "CRPIX1": 1.0,
        "CRVAL1": 4000.0,
        "CDELT1": 10.0,
    }


def _cube_header() -> dict[str, float | int | str]:
    return {
        "NAXIS": 3,
        "CTYPE1": "RA---TAN",  # type: ignore
        "CTYPE2": "DEC--TAN",  # type: ignore
        "CTYPE3": "WAVE",
        "CRPIX1": 32.0,
        "CRPIX2": 32.0,
        "CRPIX3": 1.0,
        "CRVAL1": 180.0,
        "CRVAL2": 0.0,
        "CRVAL3": 5000.0,
        "CDELT1": -0.01,  # type: ignore
        "CDELT2": 0.01,
        "CDELT3": 2.0,
    }


@pytest.mark.parametrize("origin", [0, 1])
def test_wcs_spectral_1d_matches_astropy(origin: int) -> None:
    header = _wave_1d_header()  # type: ignore

    twcs = torchsky.WCS(header)  # type: ignore
    awcs = AstropyWCS(header)  # type: ignore

    pixels = np.array([0.0, 1.0, 10.0, 25.0], dtype=np.float64)

    world_t = twcs.pixel_to_world(torch.from_numpy(pixels[:, None]), origin=origin)
    world_a = awcs.all_pix2world(pixels[:, None], origin)

    np.testing.assert_allclose(world_t.cpu().numpy(), world_a, atol=1e-12)  # type: ignore

    pixels_t = twcs.world_to_pixel(torch.from_numpy(world_a), origin=origin)
    pixels_a = awcs.all_world2pix(world_a, origin)

    np.testing.assert_allclose(pixels_t.cpu().numpy(), pixels_a, atol=1e-9)  # type: ignore
    np.testing.assert_allclose(pixels_t.cpu().numpy(), pixels[:, None], atol=1e-9)  # type: ignore


@pytest.mark.parametrize("origin", [0, 1])
def test_wcs_spatial_plus_spectral_3d_matches_astropy(origin: int) -> None:
    header = _cube_header()  # type: ignore

    twcs = torchsky.WCS(header)  # type: ignore
    awcs = AstropyWCS(header)  # type: ignore

    # Keep points around CRPIX for robust inversion across projections.
    x = np.linspace(20.0, 44.0, 7, dtype=np.float64)
    y = np.linspace(20.0, 44.0, 7, dtype=np.float64)
    z = np.array([0.0, 3.0, 11.0], dtype=np.float64)

    xx, yy, zz = np.meshgrid(x, y, z, indexing="xy")
    pixels = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)  # type: ignore

    world_t = twcs.pixel_to_world(torch.from_numpy(pixels), origin=origin)
    world_a = awcs.all_pix2world(pixels, origin)

    np.testing.assert_allclose(world_t.cpu().numpy(), world_a, atol=1e-8)  # type: ignore

    pix_t = twcs.world_to_pixel(torch.from_numpy(world_a), origin=origin)
    pix_a = awcs.all_world2pix(world_a, origin)

    np.testing.assert_allclose(pix_t.cpu().numpy(), pix_a, atol=1e-6)  # type: ignore
    np.testing.assert_allclose(pix_t.cpu().numpy(), pixels, atol=1e-6)  # type: ignore


def test_spectral_frequency_wavelength_energy_match_astropy_units() -> None:
    freq = torch.tensor([1.0e9, 1.42e9, 3.0e14], dtype=torch.float64)
    freq_q = freq.numpy() * u.Hz

    wave = frequency_to_spectral(freq, "WAVE", unit="m")
    energy = frequency_to_spectral(freq, "ENER", unit="eV")

    np.testing.assert_allclose(
        wave.numpy(), freq_q.to(u.m, equivalencies=u.spectral()).value, rtol=1e-12
    )
    np.testing.assert_allclose(
        energy.numpy(), freq_q.to(u.eV, equivalencies=u.spectral()).value, rtol=1e-12
    )
    torch.testing.assert_close(spectral_to_frequency(wave, "WAVE", unit="m"), freq)
    torch.testing.assert_close(spectral_to_frequency(energy, "ENER", unit="eV"), freq)


def test_spectral_wavenumber_matches_astropy_units() -> None:
    freq = torch.tensor([3.0e10, 3.0e13, 6.0e14], dtype=torch.float64)
    freq_q = freq.numpy() * u.Hz

    wavn_cm = frequency_to_spectral(freq, "WAVN", unit="cm-1")
    expected = freq_q.to(1 / u.cm, equivalencies=u.spectral()).value

    np.testing.assert_allclose(wavn_cm.numpy(), expected, rtol=1e-12)
    torch.testing.assert_close(
        spectral_to_frequency(wavn_cm, "WAVN", unit="cm-1"),
        freq,
        rtol=1e-12,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    ("kind", "equiv"),
    [
        ("VRAD", "radio"),
        ("VOPT", "optical"),
        ("VELO", "relativistic"),
    ],
)
def test_spectral_velocity_conventions_match_astropy(kind: str, equiv: str) -> None:
    rest_hz = 1420.40575177e6
    freq = torch.tensor([1.419e9, rest_hz, 1.421e9], dtype=torch.float64)
    rest = rest_hz * u.Hz
    freq_q = freq.numpy() * u.Hz
    equivalency = {
        "radio": u.doppler_radio(rest),
        "optical": u.doppler_optical(rest),
        "relativistic": u.doppler_relativistic(rest),
    }[equiv]

    vel = frequency_to_spectral(freq, kind, unit="m/s", rest_frequency=rest_hz)
    np.testing.assert_allclose(
        vel.numpy(), freq_q.to(u.m / u.s, equivalencies=equivalency).value, rtol=1e-12
    )
    roundtrip = spectral_to_frequency(vel, kind, unit="m/s", rest_frequency=rest_hz)
    torch.testing.assert_close(roundtrip, freq, rtol=1e-12, atol=1e-6)


def test_spectral_redshift_and_rest_wavelength_roundtrip() -> None:
    rest_wave = 656.28e-9
    z = torch.tensor([0.0, 0.1, 1.5], dtype=torch.float64)

    freq = spectral_to_frequency(z, "ZOPT", rest_wavelength=rest_wave)
    z_back = frequency_to_spectral(freq, "ZOPT", rest_wavelength=rest_wave)
    wave = frequency_to_spectral(freq, "WAVE", unit="m")

    torch.testing.assert_close(z_back, z)
    torch.testing.assert_close(wave, torch.as_tensor(rest_wave) * (1.0 + z))


def test_spectral_air_wavelength_roundtrip_and_conversion() -> None:
    air = torch.tensor([5000.0, 6563.0, 10000.0], dtype=torch.float64)
    vac = air_wavelength_to_vacuum(air, unit="Angstrom")
    air_back = vacuum_wavelength_to_air(vac, unit="Angstrom")

    assert torch.all(vac > air)
    torch.testing.assert_close(air_back, air, rtol=1e-12, atol=1e-8)

    freq = spectral_to_frequency(air, "AWAV", unit="Angstrom")
    air_from_freq = frequency_to_spectral(freq, "AWAV", unit="Angstrom")
    torch.testing.assert_close(air_from_freq, air, rtol=1e-12, atol=1e-8)


def test_wcs_spectral_convenience_uses_header_rest_frequency() -> None:
    header = {
        "NAXIS": 1,
        "CTYPE1": "VRAD",
        "CUNIT1": "km/s",
        "CRPIX1": 1.0,
        "CRVAL1": 0.0,
        "CDELT1": 10.0,
        "RESTFRQ": 1420.40575177e6,
    }
    wcs = torchsky.WCS(header)
    velocity = torch.tensor([0.0, 100.0], dtype=torch.float64)
    freq = wcs.spectral_to_frequency(velocity)
    velocity_back = wcs.frequency_to_spectral(freq)

    assert wcs.spectral_axis_indices == (0,)
    torch.testing.assert_close(velocity_back, velocity)


@pytest.mark.parametrize("ctype", ["FREQ-LOG", "WAVE-LOG", "WAVN-LOG"])
def test_wcs_spectral_log_axis_matches_astropy(ctype: str) -> None:
    header = {
        "NAXIS": 1,
        "CTYPE1": ctype,
        "CRPIX1": 1.0,
        "CRVAL1": 10.0,
        "CDELT1": 0.1,
    }
    twcs = torchsky.WCS(header)
    awcs = AstropyWCS(header)
    pixels = np.array([[0.0], [1.0], [2.0], [10.0]], dtype=np.float64)

    world_t = twcs.pixel_to_world(torch.from_numpy(pixels), origin=0)
    world_a = awcs.all_pix2world(pixels, 0)
    pix_t = twcs.world_to_pixel(torch.from_numpy(world_a), origin=0)

    np.testing.assert_allclose(world_t.numpy(), world_a, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(pix_t.numpy(), pixels, atol=1e-9)


@pytest.mark.parametrize("ctype", ["WAVE-F2W", "FREQ-W2F"])
def test_wcs_spectral_compound_reciprocal_axis_matches_astropy(ctype: str) -> None:
    header = {
        "NAXIS": 1,
        "CTYPE1": ctype,
        "CRPIX1": 1.0,
        "CRVAL1": 10.0,
        "CDELT1": 0.1,
    }
    twcs = torchsky.WCS(header)
    awcs = AstropyWCS(header)
    pixels = np.array([[0.0], [1.0], [2.0], [10.0]], dtype=np.float64)

    world_t = twcs.pixel_to_world(torch.from_numpy(pixels), origin=0)
    world_a = awcs.all_pix2world(pixels, 0)
    pix_t = twcs.world_to_pixel(torch.from_numpy(world_a), origin=0)

    np.testing.assert_allclose(world_t.numpy(), world_a, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(pix_t.numpy(), pixels, atol=1e-9)


@pytest.mark.parametrize("ctype", ["WAVE-F2W", "FREQ-W2F"])
def test_wcs_spectral_compound_reciprocal_singular_domain_nan(ctype: str) -> None:
    header = {
        "NAXIS": 1,
        "CTYPE1": ctype,
        "CRPIX1": 1.0,
        "CRVAL1": 10.0,
        "CDELT1": 0.1,
    }
    wcs = torchsky.WCS(header)

    singular_world = wcs.pixel_to_world(
        torch.tensor([[100.0]], dtype=torch.float64), origin=0
    )
    singular_pixel = wcs.world_to_pixel(
        torch.tensor([[0.0]], dtype=torch.float64), origin=0
    )

    assert torch.isnan(singular_world).all()
    assert torch.isnan(singular_pixel).all()


def test_convert_spectral_and_legacy_transform_spectral() -> None:
    rest_hz = 1420.40575177e6
    freq = torch.tensor([rest_hz, 1.419e9], dtype=torch.float64)
    radio_km_s = convert_spectral(
        freq, "FREQ", "VRAD", to_unit="km/s", rest_frequency=rest_hz
    )
    legacy = transform_spectral(freq, rest_hz, "RADIO")
    torch.testing.assert_close(radio_km_s, legacy)
