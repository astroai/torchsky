# Synthetic photometry and passbands

`torchsky.spectrophotometry` integrates spectra through explicit passbands and
returns photometry with its units and coverage attached. It is torch-native: a
passband is two tensors, a spectrum is a tensor, and the result carries
`value`, `unit`, `coverage`, and `valid` instead of an unqualified number.

Use this page when you already have an `F_lambda` spectrum on a wavelength grid.
For spectral axes and FITS readers see [spectral.md](spectral.md); for dust
reddening before photometry see [extinction.md](extinction.md).

## Units and conventions

- **Wavelengths are Angstrom-native internally.** `Passband(..., wavelength_unit="nm")`
  converts once at construction and stores Angstrom; `evaluate_throughput(...,
  wavelength_unit="nm")` uses the caller's unit for Gaussian/LVF paths.
- **`mean_f_lambda`** is the photon-weighted mean `F_lambda` in the units you
  supplied (per Angstrom). It is the default output.
- **`f_nu_microjy` and `ab_mag` require cgs `F_lambda`** in
  `erg s^-1 cm^-2 Angstrom^-1`. The AB zero point is `3631 Jy` (`3.631e9` µJy).
- **Coverage is a diagnostic, not a correction.** The integral is evaluated on
  the union of the spectrum and passband support. If less than `min_coverage`
  (default `0.99`) of the photon weight `lambda * R(lambda)` is covered, the
  value is `NaN` and `valid=False`. Nothing is renormalized silently.
- **`mask=True` means "this sample exists".** Masked samples are dropped and do
  not count toward coverage; the remaining runs are integrated separately, so a
  hole is never bridged by interpolation.
- Passband samples are held in `float64` on CPU/CUDA. On a device without
  `float64` (MPS) the passband keeps the tensor's own floating dtype.

## Which entry point?

| Goal | Entry point |
|---|---|
| One spectrum through one band | [`synthetic_photometry`](reference/spectrophotometry.md#torchsky.spectrophotometry.synthetic_photometry) |
| Many spectra or bands at once | [`synthetic_photometry_bands`](reference/spectrophotometry.md#torchsky.spectrophotometry.synthetic_photometry_bands) |
| Build a tabulated band | [`Passband`](reference/spectrophotometry.md#torchsky.spectrophotometry.Passband), `Passband.from_samples` |
| Build a Gaussian or LVF response | [`evaluate_throughput`](reference/spectrophotometry.md#torchsky.spectrophotometry.evaluate_throughput), [`throughput_from_mapping`](reference/spectrophotometry.md#torchsky.spectrophotometry.throughput_from_mapping) |
| Magnitude conversion with an explicit zero point | [`flux_to_magnitude`](reference/spectrophotometry.md#torchsky.spectrophotometry.flux_to_magnitude), [`magnitude_to_flux`](reference/spectrophotometry.md#torchsky.spectrophotometry.magnitude_to_flux) |

## Build a passband

`Passband` validates the curve: strictly increasing wavelengths, matching
shapes, finite values, non-negative throughput with at least one positive
sample. `from_samples` sorts an unsorted curve and averages duplicate
wavelengths, which is what filter-profile tables often need.

<!-- docs: run -->
```python
import torch

from torchsky.spectrophotometry import Passband

band = Passband.from_samples(
    torch.tensor([600.0, 400.0, 500.0], dtype=torch.float64),
    torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
    name="toy",
    wavelength_unit="nm",
)

assert band.wavelength.tolist() == [4000.0, 5000.0, 6000.0]
print(band.name, band.wavelength_unit, band.effective_wavelength.item())
print(band.pivot_wavelength.item())
```

`effective_wavelength` is photon-weighted; `pivot_wavelength` is the pivot used
to convert between mean `F_lambda` and `F_nu`.

## Integrate a spectrum

A flat `F_lambda` through a triangular band recovers the flat value exactly and
reports full coverage.

<!-- docs: run -->
```python
import torch

from torchsky.spectrophotometry import Passband, synthetic_photometry

band = Passband(
    torch.tensor([4000.0, 5000.0, 6000.0], dtype=torch.float64),
    torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
)
wavelength = torch.linspace(4000.0, 6000.0, 21, dtype=torch.float64)
flux = torch.full_like(wavelength, 3.5)

result = synthetic_photometry(wavelength, flux, band)
torch.testing.assert_close(result.value, torch.tensor(3.5, dtype=torch.float64))
torch.testing.assert_close(result.coverage, torch.tensor(1.0, dtype=torch.float64))
assert bool(result.valid)
print(result.unit, result.value.item(), result.coverage.item())
```

## Flux density and AB magnitudes

For `f_nu_microjy` and `ab_mag`, supply cgs `F_lambda`. A constant `F_nu` is
recovered by the same photon-counting quadrature; the AB magnitude uses the
fixed 3631 Jy zero point.

<!-- docs: run -->
```python
import torch

from torchsky.spectrophotometry import Passband, synthetic_photometry

band = Passband(
    torch.tensor([4000.0, 5000.0, 6000.0], dtype=torch.float64),
    torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
)
wavelength = torch.linspace(4000.0, 6000.0, 101, dtype=torch.float64)

f_nu_microjy = torch.tensor(12.0, dtype=torch.float64)
f_nu_cgs = f_nu_microjy * 1.0e-29
c_angstrom_per_s = 2.99792458e18
flux_lambda = f_nu_cgs * c_angstrom_per_s / wavelength.square()

jy = synthetic_photometry(wavelength, flux_lambda, band, output="f_nu_microjy")
ab = synthetic_photometry(wavelength, flux_lambda, band, output="ab_mag")
expected_ab = -2.5 * torch.log10(f_nu_microjy / 3.631e9)
torch.testing.assert_close(jy.value, f_nu_microjy, rtol=1e-4, atol=0.0)
torch.testing.assert_close(ab.value, expected_ab, rtol=1e-4, atol=0.0)
print(jy.unit, jy.value.item())
print(ab.unit, ab.value.item())
```

## Many bands, trailing band axis

`flux_lambda` may have leading batch axes; `wavelength` is either 1D (shared
grid) or the same shape as the flux. `synthetic_photometry_bands` stacks bands
on a trailing axis.

<!-- docs: run -->
```python
import torch

from torchsky.spectrophotometry import Passband, synthetic_photometry_bands

g = Passband(torch.tensor([4000.0, 5000.0, 6000.0]), torch.tensor([0.0, 1.0, 0.0]))
r = Passband(torch.tensor([5500.0, 6500.0, 7500.0]), torch.tensor([0.0, 1.0, 0.0]))
wave = torch.linspace(4000.0, 7500.0, 71)
flux = torch.full_like(wave, 4.0)

result = synthetic_photometry_bands(wave, flux, [g, r])
assert result.value.shape == (2,)
torch.testing.assert_close(
    result.value, torch.full((2,), 4.0, dtype=result.value.dtype)
)
print(result.unit, result.valid.tolist())
```

## Missing samples and coverage

Use the mask for gaps — do not put `NaN` in the flux without masking it.
Masked samples are excluded from both the integral and the coverage
denominator; a masked hole can therefore make a band invalid even when the
remaining samples are finite.

<!-- docs: run -->
```python
import torch

from torchsky.spectrophotometry import Passband, synthetic_photometry

band = Passband(
    torch.tensor([4000.0, 5000.0, 6000.0], dtype=torch.float64),
    torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
)
wavelength = torch.linspace(4000.0, 6000.0, 21, dtype=torch.float64)
flux = torch.ones_like(wavelength)

full = synthetic_photometry(wavelength, flux, band, min_coverage=0.0)
mask = torch.ones_like(wavelength, dtype=torch.bool)
mask[8:13] = False
holed = synthetic_photometry(wavelength, flux, band, mask=mask, min_coverage=0.0)

assert holed.coverage < full.coverage
assert torch.isfinite(holed.value)
print(full.coverage.item(), holed.coverage.item())
```

## Gaussian and position-dependent responses

Gaussian throughput stays in the unit you pass. `lambda_center_map` makes the
centre position-dependent for linear-variable-filter data; the centre map may
be one dimension shorter than the wavelength tensor.

<!-- docs: run -->
```python
import torch

from torchsky.spectrophotometry import evaluate_throughput

wave_nm = torch.tensor([500.0, 510.0, 520.0], dtype=torch.float64)
gauss = evaluate_throughput(wave_nm, center=510.0, sigma=5.0, wavelength_unit="nm")
torch.testing.assert_close(gauss[1], torch.tensor(1.0, dtype=torch.float64))

positions = torch.tensor([[500.0], [520.0]], dtype=torch.float64)
lvf = evaluate_throughput(
    wave_nm.unsqueeze(0).expand(2, -1),
    sigma=4.0,
    lambda_center_map=positions,
    wavelength_unit="nm",
)
assert lvf.shape == (2, 3)
assert lvf[0, 0] > lvf[0, 2]
print(gauss.tolist(), lvf[:, 0].tolist())
```

## Magnitudes and flux with an explicit zero point

`flux_to_magnitude` and `magnitude_to_flux` never assume a band's zero point.
Non-positive fluxes produce `NaN` magnitudes instead of `-inf`.

<!-- docs: run -->
```python
import torch

from torchsky.spectrophotometry import flux_to_magnitude, magnitude_to_flux

flux = magnitude_to_flux(torch.tensor([22.0]), zero_point=3631.0)
round_trip = flux_to_magnitude(flux, zero_point=3631.0)
torch.testing.assert_close(round_trip, torch.tensor([22.0]), rtol=1e-6, atol=0.0)

invalid = flux_to_magnitude(torch.tensor([0.0, -1.0]), zero_point=3631.0)
assert torch.isnan(invalid).all()
print(flux.item(), round_trip.item())
```

## Caveats

- A passband must cover the spectrum where the photon weight is non-zero;
  otherwise coverage fails closed. Extend the passband or the spectrum rather
  than lowering `min_coverage`.
- `mean_f_lambda` is per Angstrom in *your* flux units. It is not a flux
  calibrated to a physical passband unless the throughput is the real one.
- Passband throughput is dimensionless; detector quantum efficiency and
  collecting area belong in the flux calibration, not in the curve.
