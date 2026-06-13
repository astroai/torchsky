# `torchsky.scene`

`torchsky.scene` is the promoted namespace for joint sky-modelling concepts. It is currently an additive facade over the mature `torchsky.map` and `torchsky.map.field` implementation, so existing map imports keep working.

Use this namespace for the mission-level objects:

- `scene.FieldObservation` and `scene.ImageObservation`
- `scene.TargetRenderSpec`
- `scene.MapProductSpec` and `scene.MapBandpassSpec`
- `scene.SceneRepresentation` plus tensor, HEALPix, harmonic, and field-model adapters
- `scene.ObservationOperatorAdapter` and `scene.operator_bundle_for_observation(...)`

The intent is that users can describe an observation, a target product, and a scene representation without needing to know which lower-level map module owns the current implementation.

## Minimal Example

```python
import torch
from torchsky import scene

# Define a WCS header dictionary
header = {
    "NAXIS": 2,
    "NAXIS1": 32,
    "NAXIS2": 32,
    "CTYPE1": "RA---TAN",
    "CTYPE2": "DEC--TAN",
    "CRPIX1": 16.5,
    "CRPIX2": 16.5,
    "CRVAL1": 150.0,
    "CRVAL2": 2.0,
    "CDELT1": -0.05 / 3600.0,
    "CDELT2": 0.05 / 3600.0,
}

# Construct render target spec and render a zero scene to it
target = scene.target_from_header(header, value_semantics="surface_brightness")
prediction = scene.render(torch.zeros(target.shape), target)
```

For differentiable field models, `scene.predict(...)` and `scene.predict_observation_mean(...)` call the same current field-observation path used by `torchsky.map.field`.

## Operator Adapter

`scene.operator_adapter_for_observation(...)` wraps the existing `ObservationOperatorBundle` used by map and field rendering. It exposes enabled operator names, slot checks, and the deterministic prediction path while keeping the underlying bundle available through `.bundle` for lower-level integrations.
