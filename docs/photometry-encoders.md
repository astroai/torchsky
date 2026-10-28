# Shared photometry encoding and fusion

`torchsky.encoders` provides differentiable measurement-set primitives. They
consume already selected observation tensors; catalogue association, release
provenance, training splits, and label policy remain application responsibilities.
There are no pretrained weights or demonstrated astronomical accuracy claims.

## Measurement tokens

`PhotometryEncoder` accepts tensors of shape `(..., measurements)` and returns
`(latent, supported)` with shapes `(..., latent_dim)` and `(...)`. The leading
dimensions can be objects, objects × association hypotheses, or ordinary batches.
Its content representation has no absolute sky coordinates.

```python
import torch
from torchsky.encoders import PhotometryEncoder, BinnedRedshiftHead

encoder = PhotometryEncoder(
    num_passbands=3, num_surveys=2, flux_scale=[10., 10., 20.], latent_dim=32
)
flux = torch.tensor([[8., -2., 30.], [float("nan"), 6., 0.]])
error = torch.tensor([[1., 2., float("nan")], [float("nan"), 1., 0.]])
mask = torch.tensor([[True, True, True], [False, True, False]])
latent, supported = encoder(
    flux, error,
    passband_id=torch.tensor([[0, 1, 2], [-1, 1, -1]]),
    survey_id=torch.tensor([[0, 0, 1], [-1, 0, -1]]),
    status=torch.tensor([[0, 0, 1], [-1, 0, -1]]),
    epoch=torch.tensor([[2016., 2016., float("nan")], [0., 2016., 0.]]),
    mask=mask,
)
head = BinnedRedshiftHead(32, [0., 0.2, 0.5, 1., 2., 4.])
logits = head(latent, supported=supported)
loss = -head.log_prob(logits, torch.tensor([0.4, 1.2])).mean()
loss.backward()
```

The observation contract is explicit:

| Input | Meaning |
|---|---|
| `flux` | Measured signed flux, or an upper-limit threshold. Observed values must be finite. |
| `error` | Positive finite uncertainty in the same unit as flux; NaN or `None` means unknown. |
| `passband_id` | Integer index into the caller's passband vocabulary. |
| `survey_id` | Integer index into the caller's survey/release vocabulary. |
| `method_id` | Integer measurement-method vocabulary; omitted values default to method zero. |
| `status` | Zero for a detection, one for an upper-limit threshold. Default is detection. |
| `epoch` | Decimal observation year; NaN or `None` means unknown. Never substitute coordinate equinox. |
| `mask` | Boolean observation availability. False means padding, not a nondetection or unknown error. |

An upper limit must already have been expressed as a flux threshold. Its
confidence convention must be fixed in the application schema or distinguished
through the measurement-method vocabulary. Unknown errors and epochs each have
their own availability feature; their placeholder zero does not mean zero noise
or the reference epoch. Non-observation reasons such as outside footprint,
masked image, or unavailable processing belong in the catalogue coverage table;
this content encoder excludes those tokens rather than treating them as zero flux.

`flux_scale` supplies one positive value per passband, in that passband's flux
unit. Continuous token features use signed `asinh(flux / scale)`,
`log1p(error / scale)`, error availability, scaled epoch, and epoch availability.
Categorical embeddings retain filter, survey, method, and detection status.
An MLP followed by learned masked set pooling produces the object latent. Pooling
also receives measurement count. Its work is linear in the number of tokens for
a fixed latent size; batching remains the application's responsibility.

Save the passband definitions, physical units, survey release IDs, method and
limit conventions, and vocabulary ordering alongside the model checkpoint.
Scaling values and epoch reference/scale are model buffers saved in `state_dict`.
Fit any data-dependent scaling on training data only. New vocabulary entries need
an explicit checkpoint migration or retraining; integer IDs are not global IDs.

## Fusion and optional sky context

`MaskedSetFusion(latent_dim)` also accepts arbitrary modality embeddings with
shape `(..., items, latent_dim)` and an optional boolean mask `(..., items)`.
This can fuse externally encoded image/spectrum/time-series or per-survey
latents after they have been mapped to a shared dimension. It is permutation
invariant and padding-safe. Its learned attention weights do not establish
counterpart identity and are not calibrated association probabilities.

Empty or fully masked sets produce a finite zero latent and `supported=False`.
Pass support into a prediction head, or select supported rows before prediction;
`BinnedRedshiftHead(...)(latent, supported=supported)` rejects unsupported rows.
An unsupported zero latent is not an astrophysical prior.

For an optional position branch, reuse `harmonic_sky_encoding` or
`SkyCoordinateEncoder` from the same namespace and concatenate projected context
in application code. Preserve original astrometric coordinates and covariances
for matching. Compare content-only, simple positional context, and multiscale
context using spatial and survey holdouts before choosing a positional encoding.
Preserve gradient/device behavior by moving the model and input tensors together;
photometry pooling operates in the model dtype.

## Conditional redshift distributions

`BinnedRedshiftHead(latent_dim, bin_edges)` returns logits representing bin
probability masses. `log_prob(logits, redshift)` evaluates a piecewise-uniform
density, including division by bin width. Bins are left-closed/right-open, except
the final edge is included in the final bin. Finite targets outside the bin
support have log probability `-inf`; reject such targets or expand the scientific
support before training. Probability density can exceed one for narrow bins.
The head supports nonuniform bins and does not collapse multimodal redshift
uncertainty to a point. Model fitting alone does not establish calibration.

`marginalize_association_logits(logits, association_weights, hypothesis_mask=None)`
combines conditional distributions for mutually exclusive association hypotheses.
Its shapes are `(..., hypotheses, bins)`, `(..., hypotheses)`, and an optional
boolean `(..., hypotheses)`; its result is log bin masses `(..., bins)`.

```python
from torchsky.encoders import marginalize_association_logits

conditional_logits = torch.tensor([[[2., 0.], [0., 2.]]])
association_weights = torch.tensor([[0.75, 0.25]])
log_bin_mass = marginalize_association_logits(
    conditional_logits, association_weights
)
assert torch.allclose(log_bin_mass.exp().sum(-1), torch.ones(1))
```

Weights must be finite, nonnegative, and contain positive mass per object; they
are normalized over supplied hypotheses. Include no-match and blend hypotheses
when scientifically required, with their own supported conditional models.
If omitted, the result is conditional on the retained hypotheses. Ranking scores
cannot become calibrated association probabilities through normalization.
Masked or zero-weight hypotheses may contain arbitrary logits; positive-weight
hypotheses must have finite logits. All hypotheses must share the same redshift
bins. Averaging embeddings instead would generally predict a different distribution.

Real-data evaluation still needs reviewed association truth, representative
spectroscopic labels, lineage/spatial/survey splits, missing-modality tests,
out-of-distribution checks, and probability calibration. These primitives do not
provide an automatic scientific training pipeline.

## Runnable catalogue photo-z pilot

[`examples/catalogue_photoz.py`](published-examples/catalogue_photoz.py) is a small CPU
application demonstrating training, prediction and evaluation against a pinned
`xmatch.observations.release.v1` directory. It requires the existing optional
Polars/PyArrow dependencies (`pixi run -e test`) and never imports `xmatch`.

Two application-selected Parquet inputs make identity and supervision explicit:

| File | Required columns |
|---|---|
| Memberships | `source_id`, `object_id`, `split`, `split_group` (strings) |
| Labels | `object_id`, `redshift`, `evidence_id`, `category` |

`source_id` must match the namespaced identity in the observation inventory.
Each native source occurs once, each object has one split and lineage group,
and every lineage group stays within one split. Allowed splits are `train`,
`validation`, and `test`. Assign `split_group` from complete object/association/
blend lineage before exporting memberships; the example validates the supplied
groups but cannot discover missing lineage information. Memberships select one
association hypothesis per source. Ambiguous alternatives need an explicit
application policy or the hypothesis-mixture primitive above.

Labels require finite redshifts, nonempty evidence IDs, and category
`inferred_spectroscopy`. Select one reviewed redshift per object explicitly;
the application rejects duplicate labels and labels outside the membership set.
Photometric predictions are rejected as label evidence for this benchmark.
The evidence ID is an audit reference, not independent proof of spectroscopic
quality: selection, spectral quality and source-population suitability remain
the application's responsibility. An object can have no label; its target mask
then remains false, while its photometry remains available for prediction.

```bash
pixi run -e test -- python examples/catalogue_photoz.py train \
  --release /data/observations-dr1 \
  --memberships /data/memberships.parquet --labels /data/spec-labels.parquet \
  --feature-cache /data/photoz-features \
  --checkpoint /data/photoz.pt --epochs 20 --bin-edges 0,0.2,0.5,1,2,4,6

pixi run -e test -- python examples/catalogue_photoz.py predict \
  --release /data/observations-dr1 --memberships /data/memberships.parquet \
  --feature-cache /data/photoz-features \
  --checkpoint /data/photoz.pt --output /data/predictions.parquet

pixi run -e test -- python examples/catalogue_photoz.py evaluate \
  --release /data/observations-dr1 \
  --memberships /data/memberships.parquet --labels /data/spec-labels.parquet \
  --feature-cache /data/photoz-features \
  --checkpoint /data/photoz.pt --split test --output /data/test-metrics.json
```

The loader verifies manifest shard checksums, schemas and counts, and validates
membership/label identities. Native photometry is the only model input. Raw
inventory fields, redshift/type proxies and label evidence never become feature
columns. Only measured flux and explicit upper limits enter the content encoder;
other availability states remain absent observations. Epoch features are used
only for declared `jyear` times; other time conventions remain unknown in this
pilot. Passband IDs are namespace-qualified, conservatively preserving filters
with identical names in different instruments.

The workflow builds an object-sorted Parquet feature cache once through a lazy
left join of selected memberships with whitelisted photometry columns. The left
join preserves sources with no usable measurements. Cache construction uses
Polars' streaming sink and global object/measurement ordering. Each epoch then
reads that cache once through sequential Arrow batches, reconstructing complete
objects across batch boundaries. Labels are sorted separately and joined through
a sequential cursor; they never enter the feature cache.

Use `--feature-cache` to persist and reuse the cache across train/predict/evaluate
commands. Its manifest records selection/schema conventions, input-release and
membership hashes, row count, schema and content checksum. Reuse requires matching
input identities and an unchanged cache checksum; mismatches fail without replacing
the cache. A new cache is published atomically after successful materialization.
Without the option, a temporary cache is built once per command and reused across
its epochs. Persistent caches are preferable for large releases and exact artifact
reproducibility.

Set `--batch-size` and `--max-measurements` to fit the tensor memory budget. The cap
applies to both source-membership count and active measurement count per object;
exceeded caps fail rather than truncate. Active measurement tensors are bounded
by object batch size times measurement cap; source bookkeeping is also capped.
This is not a process-RSS guarantee: global validation, joins, sorting, codec
buffers and scratch disk still need appropriate resources. No full-sky throughput
or distributed-cache performance has been measured.

The fixed scale `--flux-scale-jy` defaults to `1e-5` Jy per passband, so no
validation/test observations enter scaling estimation. Checkpoints include model
weights, explicit bin edges, scales, deterministic vocabularies, seed, training
NLL history, label policy, and SHA-256 identities of the manifest, memberships and
label snapshot. Version-2 checkpoints also pin the feature-cache logical identity,
content checksum and inference-field selection. Reload requires those frozen
input identities; adding a release
or remapping vocabularies needs a deliberate migration or retraining. Evaluation
uses the same frozen labels and only a validation or test split. Output artifacts
are published atomically without replacing existing files.

Prediction Parquet retains every selected object and stores `supported`, bin
probability masses `p_z`, and the distribution mean `z_mean`. Unsupported objects
have null distributions and null means. Support here means at least one usable
photometry token; it does not establish training-population or out-of-distribution
support. The footer pins the checkpoint hash,
input metadata and bin edges. Evaluation reports mean NLL, normalized residual
bias/RMS, the fraction with `abs(delta_z)/(1+z) > 0.15`, PIT histogram counts, and
central 50/90-percent interval coverage. Metrics explicitly exclude unsupported
and unlabelled objects and report their counts; they do not claim calibration.

The synthetic integration check proves that training can learn a simple signal,
checkpoint reload preserves predictions, token permutation preserves inference,
labels/proxies do not enter the feature tensors, and invalid identities or
leaking split groups fail. Cache checks park the original survey shard after
materialization and verify repeated epochs still work, verify cross-batch group
assembly and persistent reuse, and reject changed cache contents or memberships.
It does not establish real-survey photo-z accuracy.
