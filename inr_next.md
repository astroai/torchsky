# INR Next

Working note for what to build next in `torchsky` after reading
*The Surprising Effectiveness of Noise Pretraining for Implicit Neural Representations*
([arXiv:2603.29034v1](https://arxiv.org/html/2603.29034v1)).

This note is intentionally broader than "small next patch." It records the INR family that looks most worth implementing next, plus the larger lessons from the paper that are still worth taking even when they require substantial new infrastructure.

## Main Decision

If we decide to invest in one substantial INR direction next, it should be:

`STRAINER-style transferable encoder pretraining for torchsky field models`

and not:

- another standalone pure-coordinate MLP variant
- another tiny activation tweak without a reusable pretraining story
- another local benchmark-only optimizer sweep

The key point from the paper is not merely "gaussian noise is good." The key point is that a shared encoder can be pretrained on simple synthetic signals and then reused as a strong initialization for unseen targets, with a fresh decoder head at fit time.

For `torchsky`, the most faithful and worthwhile transfer is therefore:

1. build a reusable encoder/trunk plus resettable decoder-head interface for selected field families
2. add offline pretraining over synthetic signal families
3. evaluate whether that initialization improves early supported-pixel fidelity, not just final global loss

## INR To Implement

### Primary target

`torchsky_strainer_field`

Proposed shape:

- shared spatial-spectral encoder trunk
- lightweight per-instance decoder head
- explicit `reset_decoder_head()` support
- optional frozen, partially frozen, or fully trainable encoder during fit
- optional bandpass-conditioned decoder or spectral readout head

This should first exist for one clean MLP-style family, likely closest to `sky_encoder_mlp` or `sky_encoder_resmlp`, before being generalized.

### Why this is the best next substantial INR

- It is the architecture family that the paper actually validates. Snp is a noise-pretrained Strainer, not a fundamentally new backbone.
- It gives us a principled place to attach synthetic pretraining, real-data pretraining, or mixed pretraining later.
- It aligns with the existing `torchsky` need for faster and better initialization on repeated map-fitting tasks.
- It keeps the public story clean: reusable encoder priors for field fitting.

## Best Things To Take From The Paper

### 1. Initialization is a first-class design surface

The paper is strong evidence that INR performance can change dramatically from initialization alone, even without inventing a brand-new architecture.

Transfer to `torchsky`:

- treat initialization as part of the model family, not a training afterthought
- benchmark initialization choices explicitly
- expose initialization mode in benchmark reports

### 2. Shared encoder plus fresh decoder is the transferable pattern

The most useful pattern is the split between reusable early layers and per-signal late layers.

Transfer to `torchsky`:

- make trunk/head separation explicit in the code
- support restoring a fresh head cheaply
- support storing and loading trunk-only checkpoints

### 3. Synthetic signals can be enough

The paper suggests we do not need rich natural-image corpora to get useful transfer. Simple synthetic signal families can already change INR behavior a lot.

Transfer to `torchsky`:

- invest in synthetic astronomy-shaped pretraining corpora
- do not block on curated real-image datasets before testing transfer

### 4. Different synthetic priors may help different tasks

Their result is roughly:

- unstructured noise helps fitting
- spectrum-shaped noise gives a better fit/prior tradeoff

Transfer to `torchsky`:

- do not assume there is one universal best pretraining corpus
- separate "fast fitting" from "good inverse prior" in our evaluation
- test one pretraining family for supported-pixel fidelity and another for denoising-like regularization behavior

### 5. The idea generalizes beyond one activation

The paper explicitly extends the story beyond vanilla SIREN to FINER-style activations.

Transfer to `torchsky`:

- any pretraining interface should be backbone-agnostic
- the checkpoint format should not assume one activation family forever

### 6. Number and diversity of pretraining signals matter

The paper reports better results when the pretrained encoder sees more signals.

Transfer to `torchsky`:

- treat pretraining-dataset size as a benchmark axis
- do not over-interpret results from one or two synthetic templates

## Secondary INR Methods Worth Pulling In

### `WIRE`

Paper:
[WIRE: Wavelet Implicit Neural Representations](https://arxiv.org/abs/2301.05187)

Why it matters:

- better multiscale locality bias than plain sine MLPs
- more naturally aligned with supported-pixel fidelity and local detail recovery
- still compatible with a trunk/head transfer story

What to take:

- wavelet/Gabor activation bias
- stronger spatial locality in early layers
- robustness framing, not just fit quality framing

Recommended role:

- second architecture to plug into the same Strainer-style pretraining framework

### `FINER`

Paper:
[FINER: Flexible Spectral-Bias Tuning in Implicit NEural Representation by Variableperiodic Activation Functions](https://openreview.net/forum?id=3yCqHiSQ1T)

Why it matters:

- gives a more flexible frequency story than fixed-period sine activations
- may help exact-wavelength and mixed-frequency spatial structure

What to take:

- variable-periodic activation design
- bias-controlled frequency selection
- activation-level control instead of only feature-level control

Recommended role:

- low-to-medium effort activation experiment after the trunk/head interface exists

### `IPC`

Paper family:
Implicit Pattern Composers, cited in the Snp paper as a stronger data-driven INR baseline

Why it matters:

- another example of keeping some INR weights instance-specific while sharing the rest
- points toward structured modulation instead of full-network retraining

What to take:

- instance-specific modulation rather than fully independent full models
- explicit treatment of which layers should stay shared and which should adapt

Recommended role:

- conceptual reference, not the first implementation target

## Torchsky-Specific Synthetic Pretraining Ideas

The paper uses general synthetic noise classes. For `torchsky`, we should go beyond that.

### Worth implementing

- uniform noise
- gaussian noise
- spectral `1/f^alpha` noise
- multiscale edge / dead-leaves style fields
- point-source fields with variable PSF width
- smooth extended-source fields with galaxy-like profiles
- filamentary / ridge-like fields
- mixed band-correlated spectrospatial fields

### High-effort but especially attractive

- synthetic broad-band observations produced through actual bandpass integration
- pretraining on latent sky fields, then rendering through observation operators
- operator-aware pretraining with WCS shifts, PSF blur, masks, and heterogeneous visits

The highest-value version is not "fit random images." It is:

`pretrain on synthetic latent sky fields and synthetic observation operators that resemble our real fitting problem`

## Large-Effort Features Worth Keeping

### 1. Offline trunk pretraining pipeline

Build a real pretraining script and checkpoint format, not only an on-the-fly warm start.

Needs:

- synthetic dataset generator
- checkpoint format for trunk-only and full model states
- trunk/head extraction utilities
- reproducible benchmark harness

### 2. Observation-operator-aware pretraining

Do not pretrain only on latent fields. Pretrain through:

- bandpass integration
- PSF convolution
- subpixel shifts
- varying WCS footprints
- masks and weight maps

This is much closer to `torchsky` than generic 2D image fitting.

### 3. Multiple pretraining objectives

Separate objectives for:

- fast fit initialization
- inverse-problem prior quality
- spectral consistency
- cross-band transfer

The paper strongly suggests that one objective will not dominate all regimes.

### 4. Trunk freezing schedules

Add support for:

- frozen encoder, train head only
- partially unfrozen late encoder blocks
- fully unfrozen fine-tuning

We should measure where the gains actually come from.

### 5. Pretrained structured-field backbones

The paper is MLP-centric, but the same transfer idea may still be adapted to `torchsky`'s more promising structured families.

Longer-term targets:

- pretrained `sky_encoder_resmlp`
- pretrained `sky_lambda_feature_field` decoder blocks
- pretrained operator-constrained residual heads

This is harder, but more aligned with the current `torchsky` representation thesis than staying forever in pure MLP land.

## What To Measure If We Build This

Do not judge success only by final loss.

Measure:

- initial loss after trunk transfer
- loss after a small fixed number of steps
- supported-pixel RMSE
- support fringe / extrapolated region metrics
- per-band fidelity
- calibration of uncertainty, if the model predicts it
- behavior on sparse heterogeneous mosaics

The paper is mainly about convergence and fitting capacity. For `torchsky`, the key question is whether that translates into better astronomy mapmaking metrics.

## Concrete Roadmap

### Phase 1

- formalize trunk/head split for one MLP field family
- add trunk-only checkpointing
- add offline synthetic pretraining script
- support decoder-head reset and fine-tune schedules

### Phase 2

- pretrain on uniform, gaussian, and spectral noise families
- benchmark early fit quality on existing synthetic map cases
- compare frozen-vs-unfrozen trunk strategies

### Phase 3

- replace generic noise with astronomy-shaped synthetic latent fields
- render those through bandpass and observation operators
- benchmark on sparse large-mosaic cases

### Phase 4

- add WIRE and FINER backbones under the same pretraining interface
- decide whether a pretrained activation family beats current structured fields enough to matter

## Current Recommendation

If we spend real effort here, do this next:

`Implement a STRAINER-style pretrained trunk + fresh decoder field for sky_encoder_mlp/sky_encoder_resmlp, with offline synthetic pretraining and benchmarked support-fidelity evaluation.`

If we want the next backup direction after that:

`Port WIRE into the same trunk/head pretraining framework.`

If we want a smaller side experiment:

`Add FINER as another activation option behind the same interface.`

## Things Not To Forget

- the paper's strongest result is about initialization, not necessarily the final best astronomy architecture
- `torchsky` still likely wants structured spatial/spectral fields in the long run
- pretraining should be evaluated against strong coadd and structured-field baselines, not only against vanilla SIREN-like models
- a large effort is justified only if it improves supported-pixel fidelity and sparse heterogeneous map fitting, not merely image-overfitting benchmarks
