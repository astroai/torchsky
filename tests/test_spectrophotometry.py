"""Analytic checks for the torch-native spectrophotometry contract."""

from __future__ import annotations

import math

import pytest
import torch

from torchsky.spectrophotometry import (
    Passband,
    flux_to_magnitude,
    magnitude_to_flux,
    synthetic_photometry,
    synthetic_photometry_bands,
)


def test_constant_f_lambda_has_exact_photon_weighted_mean() -> None:
    band = Passband(
        torch.tensor([4000.0, 5000.0, 6000.0]), torch.tensor([0.0, 1.0, 0.0])
    )
    wavelength = torch.linspace(4000.0, 6000.0, 21)
    result = synthetic_photometry(wavelength, torch.full_like(wavelength, 3.5), band)
    assert result.value.item() == pytest.approx(3.5)
    assert result.coverage.item() == pytest.approx(1.0)
    assert result.valid.item()


def test_constant_f_nu_is_preserved_and_ab_conversion_is_explicit() -> None:
    band = Passband(
        torch.tensor([4000.0, 5000.0, 6000.0]), torch.tensor([0.0, 1.0, 0.0])
    )
    wavelength = torch.linspace(4000.0, 6000.0, 101)
    f_nu_microjy = 12.0
    f_nu_cgs = f_nu_microjy * 1.0e-29
    flux_lambda = f_nu_cgs * 2.99792458e18 / wavelength.square()
    result = synthetic_photometry(wavelength, flux_lambda, band, output="f_nu_microjy")
    assert result.value.item() == pytest.approx(f_nu_microjy, rel=1.0e-4)
    ab = synthetic_photometry(wavelength, flux_lambda, band, output="ab_mag")
    assert ab.value.item() == pytest.approx(
        -2.5 * torch.log10(torch.tensor(f_nu_microjy / 3.631e9)).item()
    )


def test_partial_filter_coverage_fails_closed() -> None:
    band = Passband(
        torch.tensor([4000.0, 5000.0, 6000.0]), torch.tensor([0.0, 1.0, 0.0])
    )
    result = synthetic_photometry(
        torch.linspace(4500.0, 5500.0, 11), torch.ones(11), band
    )
    assert result.coverage.item() < 0.99
    assert not result.valid.item()
    assert torch.isnan(result.value)


def test_disjoint_filter_stays_invalid_even_with_zero_threshold() -> None:
    band = Passband(
        torch.tensor([4000.0, 5000.0, 6000.0]), torch.tensor([0.0, 1.0, 0.0])
    )
    result = synthetic_photometry(
        torch.linspace(7000.0, 8000.0, 11), torch.ones(11), band, min_coverage=0.0
    )
    assert result.coverage.item() == 0.0
    assert not result.valid.item()
    assert torch.isnan(result.value)


def test_overlap_without_a_filter_sample_uses_clipped_boundaries() -> None:
    band = Passband(
        torch.tensor([4000.0, 5000.0, 6000.0]), torch.tensor([0.0, 1.0, 0.0])
    )
    result = synthetic_photometry(
        torch.linspace(4500.0, 4900.0, 11), torch.ones(11), band
    )
    assert result.coverage.item() > 0.0
    assert not result.valid.item()


def test_unsorted_nm_curve_and_magnitude_round_trip() -> None:
    band = Passband.from_samples(
        torch.tensor([600.0, 400.0, 500.0]),
        torch.tensor([0.0, 0.0, 1.0]),
        wavelength_unit="nm",
    )
    assert band.wavelength.tolist() == [4000.0, 5000.0, 6000.0]
    flux = magnitude_to_flux(torch.tensor([22.0]), 3631.0)
    assert flux_to_magnitude(flux, 3631.0).item() == pytest.approx(22.0)


def test_batched_wavelength_matches_looped_1d() -> None:
    band = Passband(
        torch.tensor([4000.0, 5000.0, 6000.0]), torch.tensor([0.0, 1.0, 0.0])
    )
    wave_a = torch.linspace(4000.0, 6000.0, 21)
    wave_b = torch.linspace(4100.0, 5900.0, 21)
    flux_a = torch.full_like(wave_a, 3.5)
    flux_b = torch.full_like(wave_b, 2.0)
    stacked_wave = torch.stack((wave_a, wave_b))
    stacked_flux = torch.stack((flux_a, flux_b))
    batched = synthetic_photometry(stacked_wave, stacked_flux, band)
    looped = torch.stack(
        (
            synthetic_photometry(wave_a, flux_a, band).value,
            synthetic_photometry(wave_b, flux_b, band).value,
        )
    )
    torch.testing.assert_close(batched.value, looped)


def test_synthetic_photometry_bands_trailing_axis() -> None:
    g = Passband(torch.tensor([4000.0, 5000.0, 6000.0]), torch.tensor([0.0, 1.0, 0.0]))
    r = Passband(torch.tensor([5500.0, 6500.0, 7500.0]), torch.tensor([0.0, 1.0, 0.0]))
    wave = torch.linspace(4000.0, 7500.0, 71)
    flux = torch.full_like(wave, 4.0)
    result = synthetic_photometry_bands(wave, flux, [g, r])
    assert result.value.shape == (2,)
    assert result.value[0].item() == pytest.approx(4.0)
    assert result.value[1].item() == pytest.approx(4.0)


def test_mask_does_not_count_as_filter_coverage() -> None:
    band = Passband(
        torch.tensor([4000.0, 5000.0, 6000.0]), torch.tensor([0.0, 1.0, 0.0])
    )
    wavelength = torch.linspace(4000.0, 6000.0, 21)
    flux = torch.ones_like(wavelength)
    mask = torch.zeros_like(wavelength, dtype=torch.bool)
    mask[:3] = True
    result = synthetic_photometry(wavelength, flux, band, mask=mask)
    assert result.coverage.item() < 0.99
    assert not result.valid.item()
    covered = synthetic_photometry(wavelength, flux, band, mask=torch.ones_like(mask))
    assert covered.valid.item()


def test_mask_allows_nan_on_dropped_samples() -> None:
    band = Passband(
        torch.tensor([4000.0, 5000.0, 6000.0]), torch.tensor([0.0, 1.0, 0.0])
    )
    wavelength = torch.linspace(4000.0, 6000.0, 21)
    flux = torch.ones_like(wavelength)
    mask = torch.ones_like(wavelength, dtype=torch.bool)
    mask[10] = False
    flux[10] = torch.nan
    result = synthetic_photometry(wavelength, flux, band, mask=mask, min_coverage=0.0)
    assert torch.isfinite(result.value).item()


def test_mask_hole_is_not_interpolated_as_coverage() -> None:
    band = Passband(
        torch.tensor([4000.0, 5000.0, 6000.0]), torch.tensor([0.0, 1.0, 0.0])
    )
    wavelength = torch.linspace(4000.0, 6000.0, 21)
    flux = torch.ones_like(wavelength)
    mask = torch.ones_like(wavelength, dtype=torch.bool)
    mask[8:13] = False
    holed = synthetic_photometry(wavelength, flux, band, mask=mask, min_coverage=0.0)
    full = synthetic_photometry(wavelength, flux, band, min_coverage=0.0)
    assert holed.coverage.item() < full.coverage.item()
    assert holed.value.item() == pytest.approx(1.0)
    compacted = synthetic_photometry(
        torch.cat((wavelength[:8], wavelength[13:])),
        torch.ones(wavelength.numel() - 5),
        band,
        min_coverage=0.0,
    )
    assert compacted.coverage.item() > holed.coverage.item()


def test_evaluate_throughput_lvf_and_gaussian() -> None:
    from torchsky.spectrophotometry import evaluate_throughput

    wave = torch.tensor([500.0, 510.0, 520.0])
    gauss = evaluate_throughput(wave, center=510.0, sigma=5.0, wavelength_unit="nm")
    assert gauss[1].item() == pytest.approx(1.0)
    assert gauss[0].item() < gauss[1].item()
    lvf = evaluate_throughput(
        wave.unsqueeze(0).expand(2, -1),
        sigma=4.0,
        lambda_center_map=torch.tensor([500.0, 520.0]),
        wavelength_unit="nm",
    )
    assert lvf[0, 0].item() == pytest.approx(1.0)
    assert lvf[1, 2].item() == pytest.approx(1.0)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS is not available on this machine"
)
def test_passband_to_mps_holds_float32_and_round_trips() -> None:
    band = Passband(
        torch.linspace(4000.0, 6000.0, 9),
        torch.tensor([0.0, 0.2, 0.6, 0.9, 1.0, 0.7, 0.3, 0.1, 0.0]),
    )
    on_mps = band.to("mps")
    assert on_mps.wavelength.device.type == "mps"
    assert on_mps.wavelength.dtype == torch.float32
    back = on_mps.to("cpu")
    assert back.wavelength.device.type == "cpu"
    torch.testing.assert_close(
        back.wavelength.double(), band.wavelength, rtol=1e-6, atol=0.0
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS is not available on this machine"
)
@torch.no_grad()
def test_mps_spectrum_matches_the_cpu_integral() -> None:
    """A device without ``float64`` integrates instead of raising, and agrees.

    Two device-specific traps are covered at once. The MPS framework raises on a
    ``float64`` conversion rather than rounding, so a spectrum living there is
    integrated in its own precision. And it returns the *unexpanded* row for an
    expanded index tensor, which turned a gathered passband response into zeros
    with no error -- a band that looks measured and empty. Comparing against the
    CPU double-precision integral catches both: one raises, the other disagrees.
    """
    wave = torch.linspace(4000.0, 6000.0, 41)
    band = Passband(
        torch.linspace(4000.0, 6000.0, 9),
        torch.tensor([0.0, 0.2, 0.6, 0.9, 1.0, 0.7, 0.3, 0.1, 0.0]),
    )
    flux = 3.5 + 0.4 * torch.sin((wave - 4000.0) / 240.0)
    reference = synthetic_photometry(wave, flux, band)
    on_device = synthetic_photometry(wave.to("mps"), flux.to("mps"), band)

    assert on_device.value.device.type == "mps"
    assert bool(on_device.valid.item())
    assert on_device.value.item() == pytest.approx(reference.value.item(), rel=1e-4)
    assert on_device.coverage.item() == pytest.approx(
        reference.coverage.item(), rel=1e-4
    )


def _gaussian_band() -> tuple[torch.Tensor, torch.Tensor]:
    wave = torch.linspace(4000.0, 7000.0, 601, dtype=torch.float64)
    response = torch.exp(-0.5 * ((wave - 5500.0) / 400.0) ** 2)
    response[0] = 0.0
    response[-1] = 0.0
    return wave, response


def test_energy_weighting_is_unweighted_mean_and_matches_speclite_values() -> None:
    from torchsky.spectrophotometry import Passband, synthetic_photometry

    wave, response = _gaussian_band()
    band = Passband(wave, response)
    flux = (wave / 5500.0) ** -2.0
    photon = synthetic_photometry(wave, flux, band).value
    energy = synthetic_photometry(wave, flux, band, weighting="energy").value
    # speclite 1.0 FilterResponse.convolve_with_array ratios on the same grid.
    assert float(photon) == pytest.approx(1.0053597841459898, rel=1e-13)
    assert float(energy) == pytest.approx(1.0162535231088345, rel=1e-13)
    torch.testing.assert_close(
        energy,
        torch.trapezoid(flux * response, wave) / torch.trapezoid(response, wave),
    )
    # Energy weighting equals photon weighting of R / lambda (Bessell & Murphy 2012).
    reweighted = Passband(wave, response / wave)
    for output in ("mean_f_lambda", "f_nu_microjy"):
        torch.testing.assert_close(
            synthetic_photometry(
                wave, flux, band, weighting="energy", output=output
            ).value,
            synthetic_photometry(wave, flux, reweighted, output=output).value,
        )


def test_energy_weighting_recovers_constant_f_nu_and_masks() -> None:
    from torchsky.spectrophotometry import Passband, synthetic_photometry

    wave, response = _gaussian_band()
    band = Passband(wave, response)
    f_nu = 1.0e-28
    flux = f_nu * 2.99792458e18 / wave.square()
    for weighting in ("photon", "energy"):
        mag = synthetic_photometry(
            wave, flux, band, output="ab_mag", weighting=weighting
        )
        assert float(mag.value) == pytest.approx(
            -2.5 * math.log10(f_nu / 3631e-23), abs=1e-10
        )
    mask = torch.ones_like(wave, dtype=torch.bool)
    mask[290:310] = False
    holed = synthetic_photometry(
        wave, flux, band, mask=mask, weighting="energy", min_coverage=0.0
    )
    reweighted = synthetic_photometry(
        wave, flux, Passband(wave, response / wave), mask=mask, min_coverage=0.0
    )
    torch.testing.assert_close(holed.value, reweighted.value)
    torch.testing.assert_close(holed.coverage, reweighted.coverage)
    with pytest.raises(ValueError, match="weighting"):
        synthetic_photometry(wave, flux, band, weighting="bolometric")  # type: ignore[arg-type]


def test_energy_weighting_matches_live_speclite() -> None:
    pytest.importorskip("speclite")
    from astropy import units as u
    import speclite.filters as sf

    from torchsky.spectrophotometry import Passband, synthetic_photometry

    wave, response = _gaussian_band()
    flux = (wave / 5500.0) ** -2.0 + 0.1 * torch.sin(wave / 50.0)
    filt = sf.FilterResponse(
        wavelength=wave.numpy() * u.AA,
        response=response.numpy(),
        meta={"group_name": "torchsky", "band_name": "g"},
    )
    for photon in (True, False):
        ref = filt.convolve_with_array(
            wave.numpy(), flux.numpy(), photon_weighted=photon
        )
        norm = filt.convolve_with_array(
            wave.numpy(), torch.ones_like(wave).numpy(), photon_weighted=photon
        )
        got = synthetic_photometry(
            wave,
            flux,
            Passband(wave, response),
            weighting="photon" if photon else "energy",
        ).value
        assert float(got) == pytest.approx(float(ref / norm), rel=1e-12)


def test_vega_spectrum_and_zero_point() -> None:
    from torchsky.spectrophotometry import (
        Passband,
        flux_to_magnitude,
        synthetic_photometry,
        vega_spectrum,
        vega_zero_point,
    )

    vega_wave, vega_flux = vega_spectrum()
    assert vega_wave.dtype == torch.float64 and vega_wave.shape == (9182,)
    assert bool((vega_wave[1:] > vega_wave[:-1]).all())
    # CALSPEC normalisation: 3.47e-9 erg/s/cm2/A at 5556 A.
    at_5556 = torch.from_numpy(
        __import__("numpy").interp(5556.0, vega_wave.numpy(), vega_flux.numpy())[None]
    )
    assert float(at_5556) == pytest.approx(3.47e-9, rel=2e-3)

    wave, response = _gaussian_band()
    band = Passband(wave, response)
    # synphot 1.8 Observation(...).effstim("flam") / (uJy) of the same spectrum.
    assert float(vega_zero_point(band)) == pytest.approx(
        3.6026233620358613e-09, rel=1e-6
    )
    assert float(vega_zero_point(band, output="f_nu_microjy")) == pytest.approx(
        3615780249.686428, rel=1e-6
    )
    # Vega is magnitude zero in its own system for both weightings.
    for weighting in ("photon", "energy"):
        flux = synthetic_photometry(
            vega_wave, vega_flux, band, weighting=weighting
        ).value
        zero = vega_zero_point(band, weighting=weighting)
        assert float(flux_to_magnitude(flux, zero)) == pytest.approx(0.0, abs=1e-12)
    far_ir = Passband(
        torch.tensor([2.9e6, 3.0e6, 3.1e6], dtype=torch.float64),
        torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="Vega spectrum"):
        vega_zero_point(far_ir)
    with pytest.raises(ValueError, match="output"):
        vega_zero_point(band, output="ab_mag")  # type: ignore[arg-type]


def test_vega_zero_point_matches_live_synphot() -> None:
    pytest.importorskip("synphot")
    from astropy import units as u
    from synphot import Observation, SourceSpectrum, SpectralElement
    from synphot.models import Empirical1D

    from torchsky.spectrophotometry import Passband, vega_spectrum, vega_zero_point

    vega_wave, vega_flux = vega_spectrum()
    vega = SourceSpectrum(
        Empirical1D,
        points=vega_wave.numpy() * u.AA,
        lookup_table=vega_flux.numpy() * u.Unit("erg s-1 cm-2 AA-1"),
        keep_neg=True,
    )
    for center, sigma in ((3600.0, 300.0), (5500.0, 400.0), (12500.0, 1200.0)):
        wave = torch.linspace(
            center - 5 * sigma, center + 5 * sigma, 801, dtype=torch.float64
        )
        response = torch.exp(-0.5 * ((wave - center) / sigma) ** 2)
        response[0] = response[-1] = 0.0
        element = SpectralElement(
            Empirical1D, points=wave.numpy() * u.AA, lookup_table=response.numpy()
        )
        observation = Observation(vega, element, force="taper")
        band = Passband(wave, response)
        assert float(vega_zero_point(band)) == pytest.approx(
            observation.effstim("flam").value, rel=2e-6
        )
        assert float(vega_zero_point(band, output="f_nu_microjy")) == pytest.approx(
            observation.effstim(u.uJy).value, rel=2e-6
        )


def test_weighting_gradient_and_device(compute_device) -> None:
    from torchsky.spectrophotometry import (
        Passband,
        synthetic_photometry,
        vega_zero_point,
    )

    wave, response = _gaussian_band()
    flux = ((wave / 5500.0) ** -2.0).requires_grad_(True)
    band = Passband(wave, response)
    for output in ("mean_f_lambda", "f_nu_microjy"):
        assert torch.autograd.gradcheck(
            lambda f, o=output: (
                synthetic_photometry(wave, f, band, weighting="energy", output=o).value
            ),
            (flux,),
        )
    dtype = torch.float32 if compute_device.type == "mps" else torch.float64
    reference = synthetic_photometry(
        wave, flux.detach(), band, weighting="energy"
    ).value
    moved = Passband(wave.to(compute_device, dtype), response.to(compute_device, dtype))
    on_device = synthetic_photometry(
        wave.to(compute_device, dtype),
        flux.detach().to(compute_device, dtype),
        moved,
        weighting="energy",
    ).value
    assert on_device.device.type == compute_device.type
    assert float(on_device.cpu()) == pytest.approx(float(reference), rel=1e-5)
    zero = vega_zero_point(moved)
    assert zero.device.type == compute_device.type
    assert float(zero.cpu()) == pytest.approx(float(vega_zero_point(band)), rel=1e-5)
