# Cosmology

In an expanding universe, "distance" depends on how it is measured: by the
dimming of a source, by its angular size, or by the light-travel time.
`torchsky.cosmology` computes these distances for a homogeneous, isotropic
**Friedmann–Lemaître–Robertson–Walker (FLRW)** universe as differentiable tensor
functions of redshift and of the cosmological parameters. It holds no priors,
likelihoods, or learnable parameters: pass tensors (including `nn.Parameter`)
in and the distances differentiate with respect to them.

Use it to turn a redshift into a luminosity distance and distance modulus, an
angular-diameter distance for physical sizes, a comoving volume for number
densities, or a lookback time; and to fit cosmological parameters by gradient
descent.

## How It Works

The expansion rate follows the Friedmann equation. With the present-day
Hubble constant \(H_0\), matter density \(\Omega_m\), curvature \(\Omega_k\),
and dark energy with the Chevallier–Polarski–Linder (CPL) equation of state
\(w(a) = w_0 + w_a(1-a)\),

\[
E(z) = \frac{H(z)}{H_0} = \sqrt{\Omega_m (1+z)^3 + \Omega_k (1+z)^2
       + \Omega_\mathrm{DE}(1+z)^{3(1+w_0+w_a)} e^{-3 w_a z/(1+z)}},
\qquad \Omega_\mathrm{DE} = 1 - \Omega_m - \Omega_k .
\]

Every distance follows from the line-of-sight comoving distance
\(D_C = D_H \int_0^z \mathrm dz'/E(z')\)
([`comoving_distance`](../reference/cosmology.md#torchsky.cosmology.comoving_distance)),
with the Hubble distance \(D_H = c/H_0\):

| Quantity | Definition | Function |
|---|---|---|
| Transverse comoving \(D_M\) | \(D_C\) if flat; \(D_H/\sqrt{\Omega_k}\,\sinh(\sqrt{\Omega_k} D_C/D_H)\) if open, \(\sin\) if closed | [`transverse_comoving_distance`](../reference/cosmology.md#torchsky.cosmology.transverse_comoving_distance) |
| Luminosity \(D_L\) | \((1+z) D_M\): flux \(= L / 4\pi D_L^2\) | [`luminosity_distance`](../reference/cosmology.md#torchsky.cosmology.luminosity_distance) |
| Angular diameter \(D_A\) | \(D_M/(1+z)\): size \(= \theta D_A\) | [`angular_diameter_distance`](../reference/cosmology.md#torchsky.cosmology.angular_diameter_distance) |
| Distance modulus \(\mu\) | \(5 \log_{10}(D_L / 10\,\mathrm{pc})\) | [`distance_modulus`](../reference/cosmology.md#torchsky.cosmology.distance_modulus) |
| Volume element | \(\mathrm dV/\mathrm dz\,\mathrm d\Omega = D_H D_M^2/E(z)\) | [`differential_comoving_volume`](../reference/cosmology.md#torchsky.cosmology.differential_comoving_volume) |
| Expansion rate | \(H(z)\), \(E(z)\) | [`hubble_parameter`](../reference/cosmology.md#torchsky.cosmology.hubble_parameter), [`e_of_z`](../reference/cosmology.md#torchsky.cosmology.e_of_z) |

Parameters are given either as keyword arguments to these functions or once in
an [`FLRW`](../reference/cosmology.md#torchsky.cosmology.FLRW) object, whose
methods have the same names. Lookback time and age are integrals of the same
\(E(z)\) and are computed in a recipe below.

## Assumptions and Accuracy

- Units: distances in Mpc, \(H_0\) in km s⁻¹ Mpc⁻¹, \(\mu\) in magnitudes.
- Defaults are Planck 2018: \(H_0 = 67.66\), \(\Omega_m = 0.3111\), flat,
  \(w_0 = -1\), \(w_a = 0\).
- **Radiation is neglected.** For Planck parameters, adding photons and
  massless neutrinos would shorten \(D_L\) by 0.01 per cent at \(z = 1\),
  0.05 per cent at \(z = 10\), and 0.16 per cent at \(z = 100\). That is a
  modelling choice, not numerical error.
- The integral uses a fixed 64-node Gauss–Legendre rule in \(\ln(1+z)\), per
  point, so a result never depends on the rest of the batch. In float64 the
  flat and curved distances match `astropy.cosmology` (with radiation switched
  off, `Tcmb0=0`) to better than \(10^{-13}\) relative for \(0.01 \le z \le 100\).
- Negative redshifts are treated as zero, so \(D_C(0) = D_L(0) = 0\) and
  \(D_A\) never divides by zero.

## Example: Absolute Magnitude of an SDSS Quasar

The SDSS quasar on plate 0751, fibre 160 has \(z = 0.04559\) and a synthetic
\(r_\mathrm{AB} = 16.776\) from its spectrum (see
[Synthetic photometry](photometry.md#a-real-spectrum-through-sdss-filters)).
Its distance modulus gives the absolute magnitude, before any K-correction:

<!-- docs: run -->
```python
import torch

from torchsky.cosmology import distance_modulus, luminosity_distance

z = torch.tensor(0.04559, dtype=torch.float64)    # SDSS pipeline redshift
r_ab = 16.776                                      # synthetic r from the spectrum

d_l = luminosity_distance(z)                       # Mpc, Planck 2018
mu = distance_modulus(z)
absolute_r = r_ab - mu
assert 205.0 < float(d_l) < 213.0 and abs(float(absolute_r) + 19.82) < 0.01
```

## Recipes

### Distances for a Batch

The Etherington relation \(D_L = (1+z)^2 D_A\) holds by construction.

<!-- docs: run -->
```python
import torch

from torchsky.cosmology import angular_diameter_distance, luminosity_distance

z = torch.tensor([0.1, 1.0, 3.0], dtype=torch.float64)
d_l = luminosity_distance(z, H0=70.0, om0=0.3)
d_a = angular_diameter_distance(z, H0=70.0, om0=0.3)
torch.testing.assert_close(d_l, (1.0 + z) ** 2 * d_a)

kpc_per_arcsec = d_a * 1e3 * torch.pi / (180.0 * 3600.0)
print(kpc_per_arcsec)                              # about 1.8, 8.0, 7.7 kpc per arcsec
```

### Lookback Time and Age

The lookback time is \(t_L(z) = t_H \int_0^z \mathrm dz' / [(1+z')E(z')]\),
with the Hubble time \(t_H = 1/H_0 = 977.79/H_0\) Gyr. In \(u = \ln(1+z)\) the
integrand is \(1/E\). The age of the universe is the same integral to
\(z \to \infty\). This matches the flat ΛCDM closed form and, at
\(z = 0.5\), 1, and 3, the values of `astropy.cosmology` (5.196, 7.936,
11.645 Gyr) for the same parameters.

<!-- docs: run -->
```python
import math

import torch

from torchsky.cosmology import FLRW, e_of_z

HUBBLE_TIME_GYR = 977.792221        # 1/H0 in Gyr for H0 in km/s/Mpc


def lookback_time_gyr(z, cosmology=FLRW(), nodes=8192):
    u = torch.linspace(0.0, 1.0, nodes, dtype=torch.float64) * torch.log1p(z).unsqueeze(-1)
    inv_e = 1.0 / e_of_z(torch.expm1(u), om0=cosmology.Om0, ok0=cosmology.Ok0,
                         w0=cosmology.w0, wa=cosmology.wa)
    return HUBBLE_TIME_GYR / cosmology.H0 * torch.trapezoid(inv_e, u, dim=-1)


planck = FLRW()
lookback = lookback_time_gyr(torch.tensor([0.5, 1.0, 3.0], dtype=torch.float64), planck)
torch.testing.assert_close(lookback, torch.tensor([5.1964, 7.9360, 11.6446], dtype=torch.float64), atol=1e-3, rtol=0.0)

age = lookback_time_gyr(torch.tensor([1e5], dtype=torch.float64), planck)
omega_l = 1.0 - planck.Om0
exact = HUBBLE_TIME_GYR / planck.H0 * 2.0 / (3.0 * math.sqrt(omega_l)) * math.asinh(math.sqrt(omega_l / planck.Om0))
assert abs(float(age) - exact) < 1e-3                # 13.79 Gyr without radiation
```

### Curved and Dynamical Dark Energy Models

`Ok0 > 0` is an open universe, `Ok0 < 0` closed. `e_of_z` refuses parameters
that make \(E(z)^2 \le 0\) instead of returning `NaN` distances.

<!-- docs: run -->
```python
import torch

from torchsky.cosmology import FLRW

z = torch.tensor([0.5, 2.0], dtype=torch.float64)
flat = FLRW(H0=70.0, Om0=0.3)
open_ = FLRW(H0=70.0, Om0=0.3, Ok0=0.05)
thawing = FLRW(H0=70.0, Om0=0.3, w0=-0.9, wa=-0.3)
ratio_open = open_.transverse_comoving_distance(z) / flat.transverse_comoving_distance(z)
ratio_thawing = thawing.luminosity_distance(z) / flat.luminosity_distance(z)
assert torch.all((ratio_open - 1.0).abs() < 0.02) and torch.all((ratio_thawing - 1.0).abs() < 0.02)
print(ratio_open, ratio_thawing)                 # both about 0.99
```

### Comoving Volume of a Redshift Shell

`differential_comoving_volume` is per steradian per unit redshift; integrate it
over \(z\) and multiply by the survey solid angle.

<!-- docs: run -->
```python
import math

import torch

from torchsky.cosmology import FLRW, differential_comoving_volume

z = torch.linspace(0.5, 0.7, 201, dtype=torch.float64)
dv = differential_comoving_volume(z, FLRW())                 # Mpc^3 / sr / dz
area_sr = 100.0 * (math.pi / 180.0) ** 2                     # 100 square degrees
shell = torch.trapezoid(dv, z) * area_sr
assert 9e7 < float(shell) < 1.1e8                            # about 0.1 Gpc^3
```

### Gradients With Respect to Parameters

<!-- docs: run -->
```python
import torch

from torchsky.cosmology import FLRW

om0 = torch.tensor(0.3111, dtype=torch.float64, requires_grad=True)
h0 = torch.tensor(67.66, dtype=torch.float64, requires_grad=True)
mu = FLRW(H0=h0, Om0=om0).distance_modulus(torch.tensor([0.5, 1.0], dtype=torch.float64))
mu.sum().backward()
assert om0.grad is not None and float(h0.grad) < 0.0         # larger H0, closer source
```

Comoving Cartesian positions
([`comoving_position`](../reference/cosmology.md#torchsky.cosmology.comoving_position),
flat only), pair separations
([`comoving_separation`](../reference/cosmology.md#torchsky.cosmology.comoving_separation),
any curvature), and Alcock–Paczynski scalings
([`alcock_paczynski_scaling`](../reference/cosmology.md#torchsky.cosmology.alcock_paczynski_scaling))
build on the same distances.

## Pitfalls

- **\(h\)-scaled units.** Many catalogues quote \(h^{-1}\) Mpc. These functions
  return Mpc for the given \(H_0\).
- **Astropy's Planck18 includes radiation and neutrinos.** Comparisons at
  high redshift differ by that physics, not by numerical error.
- **K-corrections are separate.** The distance modulus converts flux to
  luminosity in the observed band; the shift of the band in rest frame needs
  [synthetic photometry](photometry.md) of a redshifted SED.
- **Peculiar velocities.** A spectroscopic redshift includes the source's own
  motion; at \(z \lesssim 0.01\) it can dominate the distance error
  ([Kinematics](kinematics.md#compose-cosmological-and-peculiar-shifts)).

## See Also

- [Kinematics](kinematics.md): redshift as a shift of the spectrum.
- [Synthetic photometry](photometry.md): observed magnitudes of a model SED.
- [`examples/sed_photometry.py`](https://github.com/astroai/torchsky/blob/main/examples/sed_photometry.py):
  colour tracks of a galaxy SED placed at luminosity distance across redshift.
- Reference: [`torchsky.cosmology`](../reference/cosmology.md).
