# Dust extinction

Interstellar dust absorbs and scatters starlight, more strongly in the blue than
in the red. The dimming at wavelength \(\lambda\) is the **extinction**
\(A(\lambda)\), in magnitudes, so the observed flux is

\[
F_\mathrm{obs}(\lambda) = F_\mathrm{int}(\lambda)\, 10^{-0.4\,A(\lambda)} .
\]

An **extinction law** gives the shape of \(A(\lambda)\). Its amount is set by
the extinction in the V band, \(A_V\), or by the **colour excess**
\(E(B-V) = A_B - A_V\), the reddening of the B−V colour. The two are linked by
\(R_V = A_V / E(B-V)\), which describes the grain-size distribution: 3.1 for the
average diffuse Milky Way, larger in dense clouds.

`torchsky.extinction` evaluates these laws on tensors and returns either
\(A(\lambda)\) ([`a_lambda`](../reference/extinction.md#torchsky.extinction.a_lambda))
or the multiplicative factor \(10^{-0.4A}\)
([`transmission`](../reference/extinction.md#torchsky.extinction.transmission)).
\(E(B-V)\), \(A_V\), and \(R_V\) may be tensors with gradients, one per object.

Use it to correct spectra and photometry for Milky Way foreground dust, and to
model dust in the source itself (a host galaxy) as part of a fit.

## How It Works

The Milky Way laws are parameterized by \(R_V\) in inverse wavelength
\(x = 1/\lambda\) (µm⁻¹). For CCM89 and O'Donnell (1994),

\[
\frac{A(\lambda)}{A_V} = a(x) + \frac{b(x)}{R_V},
\]

with \(a\) and \(b\) piecewise polynomials in the infrared, optical, and
ultraviolet. F99 uses the Fitzpatrick and Massa ultraviolet form plus a cubic
spline through optical and infrared anchor points that also depend on \(R_V\).
Calzetti (2000) is an empirical attenuation curve for star-forming galaxies,
which mixes dust and stars rather than placing a screen in front of them.

| `law` | Reference | Use for | Range |
|---|---|---|---|
| `"ccm89"` | Cardelli, Clayton & Mathis (1989) | Milky Way, default | 0.1–3.3 µm (\(x = 0.3\)–10); IR power law extrapolated below, far-UV frozen above |
| `"odonnell94"` | O'Donnell (1994) | Milky Way, improved optical | Optical polynomial replaced; UV and IR as CCM89 |
| `"f99"` | Fitzpatrick (1999) | Milky Way, the law behind SFD/SF11 recalibrations | Calibrated over its spline knots; outside is extrapolation |
| `"calzetti"` | Calzetti et al. (2000) | Star-forming host galaxies, \(R_V = 4.05\) default | 0.12–2.2 µm; clamped outside |

Later laws such as Gordon et al. (2023, G23) are not included.

## Assumptions and Accuracy

- One foreground screen per object: no scattering into the beam, no emission,
  no mixing of dust and stars (Calzetti aside, which is empirical).
- Wavelengths are Angstrom unless `wavelength_unit="nm"`; for a host-galaxy
  screen they are **rest-frame** wavelengths.
- \(A(\lambda)\) is clamped at zero in `transmission`, so a screen never
  brightens a source.
- F99 agrees with the [`dust_extinction`](https://dust-extinction.readthedocs.io)
  package to 2 per cent from 1500 to 8000 Å, and CCM89 matches its V-band
  anchor, in the test suite.

## A Real Spectrum: Galactic Dust Correction

The Schlafly & Finkbeiner (2011) map gives \(E(B-V) = 0.0698\) toward the SDSS
quasar on plate 0751, fibre 160. Removing the dust is a multiplication by
\(10^{+0.4A}\):

```python
# fragment: needs spec-0751-52251-0160.fits, SDSS.g.dat and SDSS.r.dat (see photometry.md)
import torch
import torchfits

from torchsky.extinction import a_lambda
from torchsky.filter_profiles import bandpass_from_transmission_file
from torchsky.spectrophotometry import Passband, synthetic_photometry_bands

coadd = torchfits.read("spec-0751-52251-0160.fits", hdu=1)
wave = 10.0 ** coadd["loglam"].to(torch.float64)
flux = coadd["flux"].to(torch.float64) * 1e-17
good = coadd["ivar"] > 0

dereddened = flux * 10.0 ** (0.4 * a_lambda(wave, ebv=0.0698, law="f99"))

bands = []
for name in "gr":
    curve = bandpass_from_transmission_file(f"SDSS.{name}.dat", band_name=name)
    bands.append(Passband.from_samples(curve["transmission_wavelength"].double(),
                                       curve["transmission_throughput"].double(), name=name))
mags = synthetic_photometry_bands(wave, torch.stack([flux, dereddened]), bands,
                                  output="ab_mag", mask=good.expand(2, -1), min_coverage=0.9)
print(mags.value)   # [[17.470, 16.776], [17.213, 16.597]]: g-r is 0.077 mag bluer
```

The g band brightens by 0.26 mag and r by 0.18 mag. The complete analysis is
[`examples/sdss_spectrum_photometry.py`](https://github.com/astroai/torchsky/blob/main/examples/sdss_spectrum_photometry.py).

## Recipes

### Extinction at Given Wavelengths

<!-- docs: run -->
```python
import torch

from torchsky.extinction import a_lambda

wave = torch.tensor([3500.0, 5510.0, 8000.0], dtype=torch.float64)
a = a_lambda(wave, ebv=0.5, r_v=3.1)                # CCM89 by default
assert abs(float(a[1]) - 1.55) < 0.1               # A_V = R_V E(B-V) at 5510 A
assert a[0] > a[1] > a[2]                          # blue is extinguished more

same = a_lambda(torch.tensor([551.0]), ebv=0.5, r_v=3.1, wavelength_unit="nm")
torch.testing.assert_close(same, a[1:2].float(), atol=1e-5, rtol=0.0)
```

### Redden or Deredden a Batch

A per-object `ebv` of shape `(B,)` broadcasts against a shared `(L,)` grid to
`(B, L)`. [`apply_extinction`](../reference/extinction.md#torchsky.extinction.apply_extinction)
is the one-call form of the multiplication by `transmission`.

<!-- docs: run -->
```python
import torch

from torchsky.extinction import a_lambda, transmission

wave = torch.linspace(4000.0, 8000.0, 5, dtype=torch.float64)
intrinsic = torch.ones(2, wave.numel(), dtype=torch.float64)
ebv = torch.tensor([0.0, 1.0], dtype=torch.float64)

reddened = intrinsic * transmission(wave, ebv, law="f99")
assert reddened.shape == (2, 5) and torch.all(reddened[1] < reddened[0])
restored = reddened * 10.0 ** (0.4 * a_lambda(wave, ebv=ebv, law="f99"))
torch.testing.assert_close(restored, intrinsic)
```

### Fit E(B−V) by Gradient Descent

<!-- docs: run -->
```python
import torch

from torchsky.extinction import a_lambda, transmission

wave = torch.linspace(3500.0, 9000.0, 300, dtype=torch.float64)
intrinsic = (wave / 5500.0) ** -1.0
observed = intrinsic * transmission(wave, 0.3, law="f99")

ebv = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
optimizer = torch.optim.LBFGS([ebv], max_iter=50, line_search_fn="strong_wolfe")


def closure():
    optimizer.zero_grad()
    model = intrinsic * 10.0 ** (-0.4 * a_lambda(wave, ebv=ebv, law="f99"))
    loss = (torch.log(model) - torch.log(observed)).square().sum()
    loss.backward()
    return loss


optimizer.step(closure)
assert abs(float(ebv) - 0.3) < 1e-4
```

### Host-Galaxy Dust Before Redshifting

Evaluate a host screen at rest-frame wavelength, then shift. The normalized
curve [`k_lambda`](../reference/extinction.md#torchsky.extinction.k_lambda),
\(A(\lambda)/A_V\), is convenient as a prior shape.

<!-- docs: run -->
```python
import torch

from torchsky.extinction import calzetti, k_lambda
from torchsky.redshift import shift_wavelength

rest = torch.linspace(1500.0, 9000.0, 200, dtype=torch.float64)
sed = (rest / 5500.0) ** -2.0
attenuated = sed * 10.0 ** (-0.4 * calzetti(rest, a_v=0.8))   # host dust, rest frame
observed_wave = shift_wavelength(rest, 1.0 + 1.5)               # then redshift to z = 1.5

k = k_lambda(torch.tensor([5510.0], dtype=torch.float64), law="ccm89")
assert abs(float(k) - 1.0) < 0.1 and observed_wave[0] == 3750.0
```

### Compare Laws

The laws agree in the optical and diverge in the ultraviolet, where the choice
matters most.

<!-- docs: run -->
```python
import torch

from torchsky.extinction import a_lambda

uv = torch.tensor([2000.0], dtype=torch.float64)
values = {law: float(a_lambda(uv, a_v=1.0, law=law)) for law in ("ccm89", "odonnell94", "f99", "calzetti")}
assert max(values.values()) - min(values.values()) > 0.1
print(values)
```

## Pitfalls

- **Magnitudes, not factors.** `a_lambda` returns magnitudes; the flux factor
  is \(10^{-0.4A}\).
- **Frame.** Milky Way dust acts at observed wavelength; host dust at rest
  wavelength. Mixing them shifts features by \(1+z\).
- **\(E(B-V)\) maps are calibrated for one law.** The Schlafly & Finkbeiner
  (2011) recalibration of the SFD map assumes F99 with \(R_V = 3.1\); use the
  same law when applying it.
- **Range.** Check that your wavelengths lie within the calibrated interval of
  the law before relying on far-UV or infrared corrections.

## See Also

- [Synthetic photometry](photometry.md): magnitudes after dust correction.
- [Kinematics](kinematics.md): shift between rest and observed frames.
- [Spectra and photometry overview](index.md): the full SED observation chain.
- Reference: [`torchsky.extinction`](../reference/extinction.md).
