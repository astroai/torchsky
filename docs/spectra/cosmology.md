# Cosmology
<!-- TODO(0.5.0 docs): intro -->


`torchsky.cosmology` provides differentiable FLRW distances and constant-curvature
comoving geometry. It owns the background expansion only: no priors, no
likelihoods, and no learnable parameters. Pass `nn.Parameter` values in and the
distances differentiate with respect to them.

Use this page to convert redshifts into distances, volumes, and Alcock–Paczynski
scalings before feeding spectral or catalog models. The reference is
[`torchsky.cosmology`](../reference/cosmology.md).

## Units and conventions

- Distances are **Mpc**; `H0` is **km s⁻¹ Mpc⁻¹**; `c = 299792.458 km s⁻¹`.
- Distance modulus is **magnitudes**.
- Defaults are Planck 2018: `H0=67.66`, `Om0=0.3111`, `w0=-1`, `wa=0`, `Ok0=0`.
- Dark energy uses the CPL law `w(a) = w0 + wa (1 - a)`; radiation is neglected.
- `Ok0 > 0` is an open universe, `Ok0 < 0` closed; `FlatWCDM` is the flat
  spelling (`Ok0=0` by default).
- Distances are defined for `z >= 0`: negative redshifts are clamped to zero, so
  `D_C(0) = D_L(0) = 0` and `D_A` never divides by zero.
- The line-of-sight integral uses a fixed Gauss-Legendre rule in `ln(1+z)`;
  there is no quadrature-size argument.

## Which function?

| Goal | Entry point |
|---|---|
| Line-of-sight comoving distance `D_C` | [`comoving_distance`](../reference/cosmology.md#torchsky.cosmology.comoving_distance) |
| Transverse comoving distance `D_M` (curvature-aware) | [`transverse_comoving_distance`](../reference/cosmology.md#torchsky.cosmology.transverse_comoving_distance) |
| Luminosity / angular-diameter distance | [`luminosity_distance`](../reference/cosmology.md#torchsky.cosmology.luminosity_distance), [`angular_diameter_distance`](../reference/cosmology.md#torchsky.cosmology.angular_diameter_distance) |
| Distance modulus | [`distance_modulus`](../reference/cosmology.md#torchsky.cosmology.distance_modulus) |
| Expansion rate / `E(z)` | [`hubble_parameter`](../reference/cosmology.md#torchsky.cosmology.hubble_parameter), [`e_of_z`](../reference/cosmology.md#torchsky.cosmology.e_of_z) |
| Volume element per sr per dz | [`differential_comoving_volume`](../reference/cosmology.md#torchsky.cosmology.differential_comoving_volume) |
| Cartesian positions (flat only) | [`comoving_position`](../reference/cosmology.md#torchsky.cosmology.comoving_position) |
| Pair separation (curved or flat) | [`comoving_separation`](../reference/cosmology.md#torchsky.cosmology.comoving_separation) |
| Alcock–Paczynski scaling | [`alcock_paczynski_scaling`](../reference/cosmology.md#torchsky.cosmology.alcock_paczynski_scaling) |

## Distance ladder

The Etherington duality `D_L = (1+z)^2 D_A` holds by construction.

<!-- docs: run -->
```python
import torch

from torchsky.cosmology import angular_diameter_distance, luminosity_distance

z = torch.tensor([0.1, 1.0], dtype=torch.float64)
d_l = luminosity_distance(z)
d_a = angular_diameter_distance(z)
torch.testing.assert_close(d_l, (1.0 + z) ** 2 * d_a, rtol=1e-5, atol=1e-4)
print(d_l.tolist(), d_a.tolist())
```

`luminosity_distance` is `(1+z) D_M`, and `angular_diameter_distance` is
`D_M/(1+z)`; both accept the full parameter set as keyword arguments.

## Expansion rate

`e_of_z` validates the parameters and refuses a non-positive `E(z)^2`
instead of returning `NaN` distances.

<!-- docs: run -->
```python
import torch

from torchsky.cosmology import hubble_parameter

z = torch.tensor([0.0, 1.0], dtype=torch.float64)
h = hubble_parameter(z, H0=70.0, om0=0.3, w0=-1.0)
expected = 70.0 * torch.sqrt(0.3 * (1.0 + z) ** 3 + 0.7)
torch.testing.assert_close(h, expected, rtol=1e-6, atol=0.0)
print(h.tolist())
```

## Volume element

`differential_comoving_volume` returns `c D_M^2 / H(z)` in
`Mpc^3 sr^-1` per unit redshift. It is the measure factor for
observed-coordinate point processes; selection functions are not included.

<!-- docs: run -->
```python
import torch

from torchsky.cosmology import FLRW, differential_comoving_volume

cosmology = FLRW(H0=70.0, Om0=0.3)
z = torch.tensor([0.1, 1.0, 2.0], dtype=torch.float64)
dv = differential_comoving_volume(z, cosmology)
assert torch.all(dv > 0)
assert torch.all(dv[1:] > dv[:-1])
print(dv.tolist())
```

## Comoving positions and separations

`comoving_position` is defined only for a flat universe, where a global
Euclidean Cartesian frame exists. `comoving_separation` supports curvature and
uses cancellation-free half-angle forms, so close float32 pairs remain usable.

<!-- docs: run -->
```python
import torch

from torchsky.cosmology import FLRW, comoving_position, comoving_separation

direction = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
z_a = torch.tensor(0.4, dtype=torch.float64)
z_b = torch.tensor(1.2, dtype=torch.float64)
cosmology = FLRW(H0=70.0, Om0=0.3)

position = comoving_position(direction, z_a, cosmology)
torch.testing.assert_close(
    torch.linalg.vector_norm(position), cosmology.comoving_distance(z_a)
)
radial = comoving_separation(direction, z_a, direction, z_b, cosmology)
expected = torch.abs(
    cosmology.comoving_distance(z_b) - cosmology.comoving_distance(z_a)
)
torch.testing.assert_close(radial, expected, rtol=2e-6, atol=2e-3)
print(radial.item())
```

The unit-vector direction must have trailing shape `[..., 3]` and be normalized;
both functions reject non-unit directions rather than silently projecting.

## Alcock–Paczynski scaling

The `alpha` values are dimensionless ratios relative to a reference cosmology;
`f_ap = D_M H / c` is the absolute AP observable. The function requires
`z > 0`.

<!-- docs: run -->
```python
import torch

from torchsky.cosmology import FLRW, alcock_paczynski_scaling

z = torch.tensor([0.5, 1.0], dtype=torch.float64)
cosmology = FLRW(H0=72.0, Om0=0.28, Ok0=0.02, w0=-0.9, wa=0.1)
reference = FLRW(H0=67.66, Om0=0.3111)

scaling = alcock_paczynski_scaling(z, cosmology, reference)
assert torch.all(scaling.alpha_parallel > 0)
assert torch.all(scaling.alpha_perpendicular > 0)
assert torch.all(scaling.f_ap > 0)
print(scaling.alpha_parallel.tolist(), scaling.f_ap.tolist())
```

Passing the same cosmology as `cosmology` and `reference` yields
`alpha_parallel = alpha_perpendicular = 1` exactly.

## Differentiability

Distances and the volume element are differentiable with respect to `H0`,
`Om0`, `Ok0`, `w0`, and `wa`. Use `float64` for parameter gradients when
finite-difference agreement matters.

<!-- docs: run -->
```python
import torch

from torchsky.cosmology import FLRW

om0 = torch.tensor(0.3111, dtype=torch.float64, requires_grad=True)
cosmology = FLRW(Om0=om0)
z = torch.tensor([0.5, 1.0], dtype=torch.float64)

distance = cosmology.comoving_distance(z)
distance.sum().backward()
assert om0.grad is not None
assert torch.isfinite(om0.grad)
print(om0.grad.item())
```

## Caveats

- Radiation is neglected. At `z` beyond a few, that is a modelling choice, not a
  precision statement.
- `comoving_position` raises for `Ok0 != 0`; use `comoving_separation` on a
  curved slice.
- Distances clamp negative redshifts; model predictions that need the sign of a
  small negative perturbation should handle it before calling these functions.
