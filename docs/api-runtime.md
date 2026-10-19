# Runtime API reference

Runtime-facing namespaces: tensor transforms, feature encoders, and the
map-quality contract. Dataset and DataLoader classes come from
[`torchfits.data`](ml.md); the browser viewer is documented in [viz](viz.md).

---

## Transforms

```python
from torchsky.transforms import Compose, Normalize, RandomFlip

augment = Compose([Normalize(), RandomFlip()])
```

| Group | Names |
|---|---|
| Intensity stretches | `AsinhStretch`, `LogStretch`, `PowerStretch`, `ZScale`, `MinMaxScale`, `RobustScale` |
| Geometry | `CenterCrop`, `RandomCrop`, `RandomFlip`, `RandomRotation` |
| Tensor/device | `Normalize`, `Compose`, `ToDevice` |
| Noise and perturbations | `GaussianNoise`, `PoissonNoise`, `PerturbByError` |
| Factories | `create_display_transform`, `create_training_transform`, `create_validation_transform`, `create_inference_transform` |

Every transform is a callable operating on tensors and preserves the input
dtype/device contract of the factory that built it.

---

## Encoders

| Name | Contract |
|---|---|
| `SkyCoordinateEncoder` | Encode sky coordinates (with optional spectral context) into features. |
| `PhotometryEncoder` | Encode multi-band photometry into a fixed-width feature vector. |
| `BinnedRedshiftHead` | Bin a redshift estimate into class logits and an expectation. |
| `MaskedSetFusion` | Fuse variable-length sets of measurements with masks. |
| `SineLayer`, `SirenImplicitField` | Sinusoidal (SIREN) layer and implicit-field wrapper. |
| `harmonic_sky_encoding` | Harmonic (spherical) sky-coordinate encoding. |
| `log_spectral_encoding` | Logarithmic spectral encoding features. |
| `sky_encoder_fourier_min_frequency` | Minimum Fourier frequency for sky-encoder features. |
| `marginalize_association_logits` | Marginalize association logits over detections. |

The encoders are building blocks for `torchsky.field` models; the field factory
(`field.model(kind, domain)`) wires them for you.

---

## Quality contract

| Name | Contract |
|---|---|
| `QualityBit` | Named bits for the per-pixel quality mask (`IntFlag`). |
| `WeightProduct`, `build_weight_product` | A weight/mask product that carries its own semantics. |
| `quality_bit_names`, `quality_bits_from_names` | Convert between bit values and stable names. |
| `CONFIDENCE_SEMANTICS`, `INVERSE_VARIANCE_SEMANTICS`, `MASK_POLARITY` | Declared semantics for weight/confidence/mask products. |
| `CONTRACT_VERSION` | Version of the quality contract. |
| `DEFAULT_ZERO_WEIGHT_BITS`, `MAX_CONFIDENCE_SAMPLES` | Defaults used when masks are unknown or confidence maps are bounded. |

The point of the contract is that a weight array is never *just* an array: the
semantics, mask polarity, and bit meanings travel with the product.

See also: [ML and runtime](ml.md), [encoders guide](encoders.md), and
[map reference](api-map.md).
