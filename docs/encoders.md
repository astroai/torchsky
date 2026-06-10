# `torchsky.encoders`

`torchsky.encoders` contains feature encoders used by field, map, and
coordinate-aware ML workflows.

## What This Namespace Is For

Use encoders when building continuous fields or learned models that need
sky-aware, harmonic, Fourier, or wavelength-aware features. Most users will meet
these through `torchsky.map.field` factories; advanced users can import encoder
pieces directly.

## Core Concepts

- Encoders should preserve tensor-native execution and device placement.
- Sky coordinates and wavelength/frequency features should be scaled to the
  domain being modelled rather than copied blindly from generic vision models.
- Encoder surfaces remain closer to experimental map/field work than to the
  core FITS/WCS API.

## Recipes

Use field factories first when possible:

```python
from torchsky.map.field import estimate_field_domain, make_field_model

domain = estimate_field_domain(field_observations, coord_mode="sphere_xyz")
field = make_field_model("sky_encoder_mlp", domain)
```

`field_observations` should be a sequence of `torchsky.map.field.FieldObservation`
objects, not `torchsky.map.ImageObservation` objects from the image-plane coadd
API.

## Related Guides

- [map.md](map.md) for field/INR workflows.
- [wcs.md](wcs.md) for coordinate conversions.
- [ml.md](ml.md) for datasets and transform pipelines.

## Validation

Encoder-heavy field workflows are exercised through map field tests and
benchmarks such as `benchmarks/bench_map_field_fit.py` and the SMACS field-fit
benchmarks.
