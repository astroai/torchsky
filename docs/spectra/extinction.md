# Dust extinction
<!-- TODO(0.5.0 docs): intro -->


`torchsky.extinction` evaluates Milky Way and host-galaxy dust laws on torch
tensors. It returns **attenuation in magnitudes** (`A(λ)`) and multiplicative
**transmission** (`10**(-0.4 A)`), never a reddened spectrum with hidden
assumptions.

Use this page before synthetic photometry ([spectrophotometry.md](photometry.md))
or when building spectral models ([spectral.md](index.md)). The reference is
[`torchsky.extinction`](../reference/extinction.md).

## Units and conventions

- Wavelengths are **Angstrom** by default; pass `wavelength_unit="nm"` at the
  boundary and the conversion happens once.
- `A(λ)` is in **magnitudes**. `transmission = 10**(-0.4 A)` with `A` clamped at
  zero, so a dust screen can never amplify flux.
- Supply **exactly one** of `ebv` or `a_v`; `A_V = R_V E(B-V)`. `R_V` defaults
  to `3.1`, except Calzetti, whose default is `4.05`.
- A per-object `ebv` broadcasts as a **leading batch axis**: a `(B,)` scale
  against an `(L,)` wavelength grid produces `(B, L)`, not a last-axis match.
- `k_lambda` is `A(λ)/A_V` (`a_v=1`), so `k(λ_V) ≈ 1`. Use it for host screens
  evaluated at rest-frame wavelength.

## Laws

| `law` | Reference | Notes |
|---|---|---|
| `"ccm89"` | Cardelli, Clayton & Mathis (1989) | IR power law, optical and UV polynomials; IR extrapolated below `x=1/λ_µm = 0.3`; far-UV frozen at `x=10`. |
| `"odonnell94"` (`"odonnell"`) | O'Donnell (1994) | Substitutes for the CCM89 optical polynomials only; UV and IR follow CCM89. |
| `"f99"` (`"fitzpatrick99"`) | Fitzpatrick (1999) | FM90 UV plus a natural cubic spline through the F99 knots; calibrated over the knot range, extrapolation outside it is not a claim. |
| `"calzetti"` | Calzetti (2000) | Starburst law for `0.12–2.20 µm`; wavelengths outside that interval are clamped. |

No dust emission, geometry mixing, or attenuation-free ("gray") component is
included: a screen is a screen.

## Magnitudes at a wavelength

<!-- docs: run -->
```python
import torch

from torchsky.extinction import a_lambda

wave = torch.tensor([3500.0, 5510.0, 8000.0], dtype=torch.float64)
a_v = a_lambda(wave, ebv=0.5, r_v=3.1)

# A_V = R_V * E(B-V) = 1.55 mag at the V-band anchor.
assert abs(float(a_v[1]) - 1.55) < 0.1
print(a_v.tolist())
```

The same value in nanometres is identical; only the boundary unit changes.

<!-- docs: run -->
```python
import torch

from torchsky.extinction import a_lambda

mag_angstrom = a_lambda(torch.tensor([5510.0]), ebv=1.0, r_v=3.1)
mag_nm = a_lambda(torch.tensor([551.0]), ebv=1.0, r_v=3.1, wavelength_unit="nm")
torch.testing.assert_close(mag_angstrom, mag_nm, atol=1e-6, rtol=0.0)
print(mag_angstrom.item())
```

## Transmission and applying a screen

`transmission` is the multiplicative factor for a flux density; zero `E(B-V)`
is the identity.

<!-- docs: run -->
```python
import torch

from torchsky.extinction import apply_extinction, transmission

wave = torch.tensor([3500.0, 5510.0, 8000.0], dtype=torch.float64)
flux = torch.ones_like(wave)

identity = transmission(wave, 0.0)
torch.testing.assert_close(identity, torch.ones_like(wave))

reddened = apply_extinction(flux, wave, ext_params=torch.tensor(0.5, dtype=torch.float64))
assert torch.all(reddened <= 1.0)
assert reddened[0] < reddened[2]  # blue light is attenuated more
print(identity.tolist(), reddened.tolist())
```

For a batch of objects sharing a wavelength grid, pass one `E(B-V)` per object:

<!-- docs: run -->
```python
import torch

from torchsky.extinction import apply_extinction, transmission

wave = torch.linspace(4000.0, 8000.0, 5, dtype=torch.float64)
flux = torch.ones(2, wave.numel(), dtype=torch.float64)
ebv = torch.tensor([0.0, 1.0], dtype=torch.float64)

attenuated = apply_extinction(flux, wave, ebv)
assert attenuated.shape == flux.shape
torch.testing.assert_close(attenuated[0], flux[0])
assert torch.all(attenuated[1] < attenuated[0])
print(transmission(wave, ebv).shape)
```

`apply_extinction` accepts either `lambda_obs` or its USPM spelling
`lambda_obs_angstrom`, but not both; `wavelength_unit` then applies to
`lambda_obs`.

## Choosing a law

The laws agree in the infrared and diverge in the ultraviolet, which is exactly
where the choice matters for stellar and AGN spectra. The Milky Way laws
(`ccm89`, `odonnell94`, `f99`) describe a foreground screen; `calzetti`
describes a starburst attenuation curve and is the usual choice for host-galaxy
dust at high redshift.

<!-- docs: run -->
```python
import torch

from torchsky.extinction import calzetti, fitzpatrick99

uv = torch.tensor([2000.0], dtype=torch.float64)
f99 = fitzpatrick99(uv, a_v=1.0, r_v=3.1)
cal = calzetti(uv, a_v=1.0)
assert abs(float(f99) - float(cal)) > 0.02 * abs(float(cal))
print(f99.item(), cal.item())
```

Use `k_lambda` when you need the normalized curve `A(λ)/A_V`, for example to
attach a host-screen prior to a spectral model:

<!-- docs: run -->
```python
import torch

from torchsky.extinction import k_lambda

wave = torch.tensor([3000.0, 5510.0, 8000.0], dtype=torch.float64)
k = k_lambda(wave, law="ccm89", r_v=3.1)
assert abs(float(k[1]) - 1.0) < 0.1
print(k.tolist())
```

## Caveats

- An attenuation law returns magnitudes; converting to transmission is
  `10**(-0.4 A)` and the implementation clamps negative `A` to zero. Do not
  reuse `A` as a multiplicative factor.
- `E(B-V)` is not a flux ratio and does not depend on the object's redshift;
  evaluate host-screen curves at **rest-frame** wavelength.
- Wavelengths outside a law's calibrated interval are extrapolated or clamped
  as listed above. Check the interval before publishing a fit that relies on
  the far-UV or far-IR behavior.
