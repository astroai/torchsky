# `torchsky.encoders`

`torchsky.encoders` contains feature encoders used by field, map, and
coordinate-aware ML workflows.

## What This Namespace Is For

Use encoders when building continuous fields or learned models that need
sky-aware, harmonic, Fourier, or wavelength-aware features. Most users will meet
these through `torchsky.field` models; advanced users can import encoder
pieces directly.

## Core Concepts

- Encoders should preserve tensor-native execution and device placement.
- Sky coordinates and wavelength/frequency features should be scaled to the
  domain being modelled rather than copied blindly from generic vision models.
- Encoder surfaces remain closer to experimental map/field work than to the
  core FITS/WCS API.

## Recipes

Use the field namespace first when possible:

```python
from torchsky import field

domain = field.estimate_domain(field_observations, coord_mode="sphere_xyz")
model = field.model("sky_encoder_mlp", domain)
```

`field_observations` should be a sequence of `torchsky.field.Observation`
objects, not `torchsky.map.observation.ImageObservation` objects from the image-plane coadd
API.

## Related Guides

- [map.md](map.md) for field/INR workflows.
- [wcs.md](wcs.md) for coordinate conversions.
- [ml.md](ml.md) for datasets and transform pipelines.

## Validation

Core field tests retain bounded model, gradient, and device contracts.
Scientific model comparisons and real-data field-fit campaigns live in
[torchsky-harness](https://github.com/astroai/torchsky-harness).
