# Kinematics

Motion along the line of sight stretches a spectrum. A source that recedes, or
lies at cosmological distance, is seen at longer wavelengths than it emits. The
stretch is the **redshift** \(z\):

\[
1 + z = \frac{\lambda_\mathrm{obs}}{\lambda_\mathrm{emit}}
      = \frac{\nu_\mathrm{emit}}{\nu_\mathrm{obs}} .
\]

`torchsky.redshift` converts between redshift, radial velocity, and the
dimensionless velocity \(\beta = v/c\); shifts wavelength grids; applies the
flux dimming that goes with a shift; and computes the Earth's velocity toward a
source for barycentric correction.
[`relativistic_doppler_shift`](../reference/spectral.md#torchsky.spectral.relativistic_doppler_shift)
shifts a binned spectrum while conserving its integrated flux. All functions are
tensor functions, so a redshift or velocity can be fitted by gradient descent,
one per object in a batch.

Use this page to move spectra between the rest frame and the observed frame,
to compare velocities quoted in different conventions, and to combine
cosmological, peculiar, and barycentric shifts. Distances and ages for a
cosmological redshift are on the [Cosmology](cosmology.md) page.

## How It Works

**Velocity conventions.** A velocity quoted for a spectral line is a
convention for writing the observed frequency relative to a rest frequency
\(\nu_0\), not a measured speed. Three are in use:

| Convention | FITS type | Definition | Typical use |
|---|---|---|---|
| Radio | `VRAD` | \(v = c\,(1 - \nu/\nu_0)\) | Radio cubes, HI and CO lines |
| Optical | `VOPT` | \(v = c\,(\lambda/\lambda_0 - 1) = cz\) | Optical redshift surveys |
| Relativistic | `VELO` | \(1 + z = \sqrt{(1+\beta)/(1-\beta)}\) | A physical line-of-sight speed |

They agree to first order in \(\beta\); radio and optical differ by about
\(v^2/c\), which is 0.3 km/s at 300 km/s and 330 km/s at 10 000 km/s.
Convert axis values between them with
[`convert_spectral`](../reference/wcs.md#torchsky.wcs.convert_spectral). The
`torchsky.redshift` helpers use the relativistic definition by default and the
optical one with `relativistic=False`.

**Composing shifts.** Shifts multiply: the factor seen at a telescope is
\((1+z_\mathrm{cos})(1+z_\mathrm{pec})(1+z_\mathrm{bary})\), for the cosmological
redshift, the source's peculiar velocity, and the observer's motion. Adding
redshifts is wrong beyond first order. The **rapidity**
\(\zeta = \ln(1+z)\) turns the product into a sum, and a shift into a constant
offset on a \(\ln\lambda\) grid:

\[
\ln\lambda_\mathrm{obs} = \ln\lambda_\mathrm{emit} + \zeta_\mathrm{cos}
  + \zeta_\mathrm{pec} + \zeta_\mathrm{bary},
\qquad \zeta_\mathrm{pec} = \operatorname{atanh}\beta .
\]

**Flux under a shift.** A shift moves samples and changes flux densities.
Which factor applies depends on the quantity, so each has its own function:

| Quantity | Factor | Function |
|---|---|---|
| \(F_\lambda\) of a source at fixed luminosity distance | \(1/(1+z)\) | [`dimming_f_lambda`](../reference/redshift.md#torchsky.redshift.dimming_f_lambda) |
| Surface brightness (Tolman dimming) | \((1+z)^{-4}\) | [`dimming_tolman_sb`](../reference/redshift.md#torchsky.redshift.dimming_tolman_sb) |
| Binned spectrum, flux-conserving | integral unchanged | [`relativistic_doppler_shift`](../reference/spectral.md#torchsky.spectral.relativistic_doppler_shift) |

[`apply_redshift`](../reference/redshift.md#torchsky.redshift.apply_redshift)
shifts a sampled rest-frame \(F_\lambda\) spectrum by a velocity or a
redshift in one call, applying both the wavelength stretch and the
\(1/(1+z)\) factor. The inverse-square dimming with distance belongs to the luminosity distance on
the [Cosmology](cosmology.md) page; it is not applied here.

## Conventions and Accuracy

- **Positive is receding.** A positive velocity or \(\beta\) means the source
  and observer move apart (redshift).
- **Units differ by function.** `torchsky.redshift` takes km/s;
  `relativistic_doppler_shift` takes m/s. Both reject \(|v| \ge c\).
- The conversions are closed-form. The Earth velocity comes from the ERFA
  `epv00` series, and its direction is checked against Astropy in the tests.

## A Real Spectrum: Rest Frame and a Line Velocity

The SDSS pipeline gives \(z = 0.04559\) for the quasar on plate 0751, fibre
160. Shift the observed wavelengths to the rest frame and measure the
[O III] 5008 Å line centroid against it:

```python
# fragment: needs spec-0751-52251-0160.fits (SDSS DR16, see spectral-axes.md)
import torch
import torchfits

from torchsky.redshift import shift_wavelength, velocity_kms_from_redshift

path = "spec-0751-52251-0160.fits"
coadd = torchfits.read(path, hdu=1)
z = float(torchfits.read(path, hdu=2)["Z"][0])                    # 0.04559
wave = 10.0 ** coadd["loglam"].to(torch.float64)
flux = coadd["flux"].to(torch.float64)
good = coadd["ivar"] > 0

print(velocity_kms_from_redshift(torch.tensor(z, dtype=torch.float64)))  # 13356 km/s
rest = shift_wavelength(wave, 1.0 / (1.0 + z))

line = 5008.24                                                   # [O III], vacuum
core = good & ((rest - line).abs() < 8.0)
side = good & ((rest - line).abs() > 16.0) & ((rest - line).abs() < 40.0)
weight = (flux[core] - flux[side].median()).clamp(min=0.0)
centre = (rest[core] * weight).sum() / weight.sum()
print(299792.458 * (centre / line - 1.0))                         # about +14 km/s
```

The line sits within 15 km/s of the pipeline redshift, about a fifth of an SDSS
pixel (69 km/s). The full analysis is
[`examples/sdss_spectrum_photometry.py`](https://github.com/astroai/torchsky/blob/main/examples/sdss_spectrum_photometry.py).

## Recipes

### Convert Between Redshift and Velocity

<!-- docs: run -->
```python
import torch

from torchsky.redshift import redshift_from_velocity_kms, velocity_kms_from_redshift

v = torch.tensor([300.0, 30000.0], dtype=torch.float64)          # km/s
z_rel = redshift_from_velocity_kms(v)
z_opt = redshift_from_velocity_kms(v, relativistic=False)        # cz convention
assert abs(float(z_rel[1]) - 0.1056) < 1e-4 and abs(float(z_opt[1]) - 0.1001) < 1e-4
torch.testing.assert_close(velocity_kms_from_redshift(z_rel), v)
```

### Convert a Radio Cube Axis Between Conventions

<!-- docs: run -->
```python
import torch

from torchsky.wcs import convert_spectral

hi_rest = 1.420405751768e9                                       # Hz, HI 21 cm
freq = torch.tensor([1.4200e9, 1.4190e9], dtype=torch.float64)
v_radio = convert_spectral(freq, "FREQ", "VRAD", to_unit="km/s", rest_frequency=hi_rest)
v_optical = convert_spectral(freq, "FREQ", "VOPT", to_unit="km/s", rest_frequency=hi_rest)
v_rel = convert_spectral(freq, "FREQ", "VELO", to_unit="km/s", rest_frequency=hi_rest)
assert bool((v_radio < v_rel).all() and (v_rel < v_optical).all())
print(v_optical - v_radio)                                       # 0.02 and 0.29 km/s
```

### Compose Cosmological and Peculiar Shifts

<!-- docs: run -->
```python
import torch

from torchsky.redshift import (
    compose_rapidities, one_plus_z_from_velocity_kms, one_plus_z_from_zeta,
    zeta_from_redshift, zeta_from_velocity_kms,
)

z_cos = torch.tensor(0.5, dtype=torch.float64)
v_pec = torch.tensor(500.0, dtype=torch.float64)                 # km/s, receding
total = one_plus_z_from_zeta(compose_rapidities(zeta_from_redshift(z_cos), zeta_from_velocity_kms(v_pec)))
torch.testing.assert_close(total, (1.0 + z_cos) * one_plus_z_from_velocity_kms(v_pec))
```

On a \(\ln\lambda\) grid,
[`shift_ln_wavelength`](../reference/redshift.md#torchsky.redshift.shift_ln_wavelength)
applies the summed rapidity as one offset.

### Shift a Binned Spectrum Without Losing Flux

`relativistic_doppler_shift` moves the bin edges by the Doppler factor; with
`output_edges` it rebins the shifted spectrum back onto a common grid, so a
batch of spectra with different velocities shares one grid.

<!-- docs: run -->
```python
import torch

from torchsky.spectral import relativistic_doppler_shift

edges = torch.linspace(5000.0, 5100.0, 101, dtype=torch.float64)
centres = 0.5 * (edges[1:] + edges[:-1])
flux = torch.exp(-0.5 * ((centres - 5050.0) / 2.0) ** 2)
velocities = torch.tensor([[-100_000.0], [100_000.0]], dtype=torch.float64)   # m/s
shifted = relativistic_doppler_shift(flux, edges, velocities.squeeze(-1), output_edges=edges)
assert shifted.flux.shape == (2, 100)
torch.testing.assert_close(shifted.flux.sum(-1), flux.sum().expand(2))
```

### Barycentric Correction

[`zeta_berv_from_sky`](../reference/redshift.md#torchsky.redshift.zeta_berv_from_sky)
returns the rapidity of the Earth's barycentric motion toward an ICRS position
at a TT epoch; [`los_barycentric_beta`](../reference/redshift.md#torchsky.redshift.los_barycentric_beta)
is the batched tensor form.

<!-- docs: run -->
```python
from torchsky.redshift import zeta_berv_from_sky

zeta = zeta_berv_from_sky(83.82, -5.39, 60000.0)                  # Orion, TT MJD 60000
assert abs(zeta * 299792.458) < 31.0                              # |v| below 30 km/s
```

This is the velocity of the geocentre. The diurnal term from Earth rotation
(up to 0.46 km/s) and the light-time and relativistic terms of a precision
radial-velocity reduction are not included. Converting UTC to TT, and site
coordinates, are covered on the [Observed place](../coordinates/observed-place.md)
and [Frames and time](../coordinates/frames.md) pages.

## Pitfalls

- **Air versus vacuum.** Convert air wavelengths to vacuum on the observed grid
  first, then shift. The refractive index depends on wavelength, so converting
  after the shift gives a different answer
  ([Spectral axes](spectral-axes.md#convert-between-wavelength-frequency-and-energy)).
- **Do not add redshifts.** \(z_\mathrm{cos} + z_\mathrm{pec}\) is wrong at the
  \(z\,\beta\) level; multiply the factors or add rapidities.
- **Pick the dimming that matches the quantity.** \(F_\lambda\), \(F_\nu\),
  and surface brightness transform differently.
- **Quote the convention.** A velocity without "radio", "optical", or
  "relativistic" is ambiguous at the km/s level above about 1000 km/s.

## See Also

- [Spectral axes](spectral-axes.md): grids, bins, and flux-conserving rebinning.
- [Cosmology](cosmology.md): distances and ages for a cosmological redshift.
- [Synthetic photometry](photometry.md): magnitudes of a redshifted spectrum.
- Reference: [`torchsky.redshift`](../reference/redshift.md) and
  [`torchsky.spectral`](../reference/spectral.md).
