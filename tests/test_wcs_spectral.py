import pytest
import torch

import torchsky
from torchsky.wcs.spectral import (
    air_wavelength_to_vacuum,
    convert_spectral,
    frequency_to_spectral,
    spectral_to_frequency,
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




@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_wcs_spectral_1d_cuda_matches_cpu() -> None:
    header = _wave_1d_header()
    cpu = torchsky.WCS(header)
    gpu = torchsky.WCS(header).to("cuda")
    pixels = torch.tensor([[0.0], [1.0], [10.0], [25.0]], dtype=torch.float64)
    world_c = cpu.pixel_to_world(pixels, origin=0)
    world_g = gpu.pixel_to_world(pixels.cuda(), origin=0)
    assert world_g.device.type == "cuda"
    torch.testing.assert_close(world_g.cpu(), world_c, atol=1e-12, rtol=0.0)
    pix_g = gpu.world_to_pixel(world_g, origin=0)
    torch.testing.assert_close(pix_g.cpu(), pixels, atol=1e-9, rtol=0.0)










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






_SUPPORTED_COMPOUND_CASES = [
    "FREQ-F2F",
    "FREQ-W2F",
    "FREQ-A2F",
    "FREQ-V2F",
    "WAVE-F2W",
    "WAVE-W2W",
    "WAVE-A2W",
    "WAVE-V2W",
    "AWAV-F2A",
    "AWAV-W2A",
    "AWAV-A2A",
    "AWAV-V2A",
    "VELO-F2V",
    "VELO-W2V",
    "VELO-A2V",
    "VELO-V2V",
]


def _compound_header(ctype: str) -> dict[str, float | int | str]:
    stype = ctype.split("-", 1)[0]
    header: dict[str, float | int | str] = {
        "NAXIS": 1,
        "NAXIS1": 128,
        "CTYPE1": ctype,
        "CRPIX1": 1.0,
        "RESTFRQ": 1.42040575177e9,
    }
    if stype == "FREQ":
        header.update({"CRVAL1": 1.420e9, "CDELT1": 1.0e5, "CUNIT1": "Hz"})
    elif stype in {"WAVE", "AWAV"}:
        header.update({"CRVAL1": 0.21, "CDELT1": 1.0e-4, "CUNIT1": "m"})
    elif stype == "VELO":
        header.update({"CRVAL1": 10.0, "CDELT1": 0.1, "CUNIT1": "m/s"})
    elif stype == "ENER":
        header.update({"CRVAL1": 9.407e-25, "CDELT1": 6.626e-29, "CUNIT1": "J"})
    elif stype == "WAVN":
        header.update({"CRVAL1": 4.737, "CDELT1": 0.01, "CUNIT1": "m-1"})
    elif stype == "ZOPT":
        header.update({"CRVAL1": 0.0, "CDELT1": 1.0e-4})
    elif stype == "BETA":
        header.update({"CRVAL1": 0.0, "CDELT1": 1.0e-6})
    return header




def test_wcs_spectral_zopt_and_beta_compound_roundtrip() -> None:
    for ctype in ("FREQ-Z2F", "FREQ-B2F", "FREQ-N2F", "FREQ-E2F", "ENER-F2E", "WAVN-F2N"):
        header = _compound_header(ctype)
        twcs = torchsky.WCS(header)
        pixels = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float64)
        world = twcs.pixel_to_world(pixels)
        back = twcs.world_to_pixel(world)
        assert torch.isfinite(world).all()
        torch.testing.assert_close(back, pixels, atol=1e-8, rtol=0.0)


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


def _grism_header(ctype: str) -> dict[str, float | int | str]:
    if ctype.startswith("FREQ"):
        crval = 6.0e14
        cdelt = 1.0e9
        cunit = "Hz"
    else:
        crval = 5.0e-7
        cdelt = 1.0e-9
        cunit = "m"
    return {
        "NAXIS": 1,
        "NAXIS1": 128,
        "CTYPE1": ctype,
        "CUNIT1": cunit,
        "CRPIX1": 1.0,
        "CRVAL1": crval,
        "CDELT1": cdelt,
        "PV1_0": 300000.0,
        "PV1_1": 1.0,
        "PV1_2": 15.0,
        "PV1_3": 1.5,
        "PV1_4": -1.0e5,
        "PV1_5": 0.0,
        "PV1_6": 10.0,
    }










def test_specsys_is_preserved_not_applied() -> None:
    base = {
        "NAXIS": 1,
        "CTYPE1": "FREQ",
        "CUNIT1": "Hz",
        "CRPIX1": 1.0,
        "CRVAL1": 1.42e9,
        "CDELT1": 1.0e5,
        "RESTFRQ": 1.42040575177e9,
    }
    with_sys = dict(base)
    with_sys["SPECSYS"] = "LSRK"
    pixels = torch.tensor([[0.0], [3.0], [10.0]], dtype=torch.float64)
    w_plain = torchsky.WCS(base)
    w_sys = torchsky.WCS(with_sys)
    torch.testing.assert_close(
        w_sys.pixel_to_world(pixels, origin=0),
        w_plain.pixel_to_world(pixels, origin=0),
        atol=0.0,
        rtol=0.0,
    )
    assert w_sys.wcs_params.get("SPECSYS") == "LSRK"


def test_unknown_spectral_algorithm_raises() -> None:
    with pytest.raises(ValueError, match="unsupported FITS spectral algorithm"):
        torchsky.WCS(
            {
                "NAXIS": 1,
                "CTYPE1": "WAVE-FOO",
                "CRPIX1": 1.0,
                "CRVAL1": 5000.0,
                "CDELT1": 1.0,
            }
        )




def test_convert_spectral_frequency_to_radio_velocity() -> None:
    rest_hz = 1420.40575177e6
    freq = torch.tensor([rest_hz, 1.419e9], dtype=torch.float64)
    radio_km_s = convert_spectral(
        freq, "FREQ", "VRAD", to_unit="km/s", rest_frequency=rest_hz
    )
    expected = frequency_to_spectral(
        freq, "VRAD", unit="km/s", rest_frequency=rest_hz
    )
    torch.testing.assert_close(radio_km_s, expected)
