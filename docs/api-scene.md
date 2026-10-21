# Scene API reference

`torchsky.scene` is the promoted composition facade: observations, scene
representations, targets, operators, and rendering behind one vocabulary. The
[scene guide](scene.md) explains the model; this page lists the promoted names.

---

## Observations and targets

| Name | Contract |
|---|---|
| `image_observation(path=None, **kwargs)` | Build an image-plane observation (the map `Observation` contract). |
| `field_observation(data, **kwargs)` | Build a tensor-backed field observation. |
| `product_spec(header, bands, ...)` | Build a map product specification using scene vocabulary. |
| `target_from_header(header)` / `target_from_product(product, band=None)` | Build a render target from a WCS header or a product band. |
| `target_to_field_observation(target)` | Convert a render target into a field observation. |
| `ImageObservation`, `FieldObservation`, `BandpassSpec`, `ProductSpec`, `PSFSpec`, `ForwardContext`, `TargetRenderSpec` | Object contracts shared with `torchsky.map` and `torchsky.field`. |

## Scene representations

| Name | Contract |
|---|---|
| `SceneRepresentation` | Base protocol: anything that can render onto a target. |
| `Scene` | Abstract base for composed scenes. |
| `DiscreteTargetImage` | Sampled image with explicit value semantics. |
| `ContinuousFieldModel` | Wrap a continuous field model as a scene representation. |
| `HealpixMapScene` | HEALPix map scene (dense or sparse). |
| `HarmonicAlmScene` | Scalar alm scene; `SpinHarmonicAlmScene` is the spin-weighted form. |
| `ObservationOperatorAdapter` | Executable adapter over an observation-operator bundle. |
| `ObservationOperatorBundle`, `ObservationOperatorBundleProtocol` | Composable observation-law components and their structural protocol. |
| `SplatDiagnostics` | Per-batch accounting for bilinear deposits at grid edges. |
| `as_scene_representation(value)` | Coerce a supported object into a scene representation. |

## Supplied-source scenes

| Name | Contract |
|---|---|
| `SourceCatalogScene` | Fixed caller-supplied component identities, positions, morphology, spectral coefficients, supports, and parameter slices. |
| `SourceStamp` | One bounded fixed-template morphology stamp with explicit native-grid placement. |
| `POINT_MORPHOLOGY`, `GAUSSIAN_MORPHOLOGY`, `SERSIC_MORPHOLOGY`, `TEMPLATE_MORPHOLOGY` | Supported known-component morphology identifiers. |

Source scenes never detect, create, delete, or silently exchange components.

## Operators, rendering, and binding

| Name | Contract |
|---|---|
| `operator_bundle_for_observation(observation)` | Operator bundle for one observation. |
| `operator_bundles_for_observations(observations)` | One bundle per observation, order preserved. |
| `operator_adapter_for_observation(observation)` | Promote an observation to the adapter interface. |
| `bind_image_plane_observations_to_target(observations, target, ...)` | Bind several observations to one latent target-grid scene. |
| `predict(representation, observation)` | Deterministic operator-path prediction for one field observation. |
| `render(representation, target, ...)` | Render any supported representation onto a target grid. |
| `splat_to_native(...)` | Deposit canonical-pixel flux onto a native grid with bilinear weights. |
| `sample_image_plane_batched_channels(...)` | Batched image-plane channel sampling helper. |

See also: [scene guide](scene.md), [map reference](api-map.md), and
[field reference](api-field.md).
