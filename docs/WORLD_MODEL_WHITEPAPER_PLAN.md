# World-Model Whitepaper Plan

This document replaces the abandoned root-level `continuous_latent_field_world_model_whitepaper_revised*.md` drafts.

Its purpose is not to be the white paper itself. It is the plan for a full white paper that is aligned with what `torchsky` actually is, what it already implements, and what should move into a separate research layer rather than into the core library.

---

## 1. Working Thesis

The white paper should argue for a specific position:

- `torchsky` is best understood as an **observation-native operator and rendering substrate** for astronomy and cosmology.
- The next major research program should be a **predictive-first continuous latent field world model** built **on top of** that substrate, not folded wholesale into it.
- The latent field should be presented as a **predictive state**, not as a secretly assumed physical field.
- Scientific meaning should be attached later through **probes, decoders, score heads, and misspecification tests**, not hard-coded into the first latent ontology.

That is the center of gravity for the white paper. It should not read like:

- a generic foundation-model pitch,
- a rebrand of existing INR fitting,
- a claim that `torchsky` already is the world model,
- or an argument that predictive reconstruction alone is enough for cosmology.

---

## 2. What The White Paper Must Do

The finished white paper should do five jobs well.

1. State the scientific gap between explicit field-level inference and generic latent world models.
2. Explain why a coordinate-indexed latent field is a better predictive state for cosmology than a global latent vector.
3. Define a clean software split between stable `torchsky` infrastructure and fast-moving world-model research code.
4. Be explicit about the real risks: nuisance entanglement, weak identifiability, and the possibility that predictive sufficiency is not cosmological sufficiency.
5. Commit to concrete empirical tests that could falsify the program.

If the paper does not do all five, it will read as ambition without scientific discipline.

---

## 3. Audience

The target reader is a mixed audience:

- cosmologists who know field-level inference,
- ML researchers familiar with world models and JEPA-style predictive learning,
- scientific software people evaluating whether the architecture split is coherent,
- potential collaborators who need to understand what belongs in `torchsky` versus a future research repo.

The paper should therefore optimize for:

- precise scientific positioning,
- explicit notation,
- engineering realism,
- and falsifiable claims.

It should not optimize for:

- maximal breadth of citations,
- architectural novelty theater,
- or overclaiming external readiness.

---

## 4. Core Claims To Make

The white paper should make the following claims directly.

### 4.1 Predictive-state claim

There may exist a continuous, coordinate-indexed latent state that is better matched to observation-native cosmology than either:

- simulator-tied explicit latent fields, or
- opaque vector/token latents.

### 4.2 Observation-native claim

Training against real or near-real observations and their metadata is a stronger and more relevant standard than training only against simulator-native latent products.

### 4.3 Structure claim

The most important structure should remain explicit in the observation operator:

- geometry,
- WCS,
- PSF/beam,
- bandpass,
- masking,
- noise,
- survey/instrument context.

The latent should not silently absorb these unless forced to.

### 4.4 Software claim

`torchsky` should remain the reusable substrate:

- observations,
- metadata,
- operator bundles,
- field primitives,
- rendering contracts,
- benchmarking and parity machinery.

The world-model training program should live in a separate layer or repository.

### 4.5 Scientific humility claim

Held-out predictive success is necessary but not sufficient. The paper must say that explicitly and repeatedly.

---

## 5. Claims To Avoid

The white paper should explicitly avoid the following overclaims.

- That the latent field is already a physical field.
- That `torchsky` already implements the proposed world model.
- That scale coordinates automatically correspond to physical coarse-graining.
- That uncertainty heads automatically provide calibrated cosmological uncertainty.
- That `cosmapper` should become the core implementation.
- That predictive reconstruction metrics alone establish scientific value.

These are the places where the earlier drafts drifted too easily toward conceptual inflation.

---

## 6. Grounding In The Current Repo

The paper plan should be anchored to what `torchsky` already contains.

### 6.1 Already real in `torchsky`

- Observation metadata surfaces for field models via `FieldObservation`.
- Exposure and survey context via `ForwardContext`.
- Shared operator composition via `ObservationOperatorBundle`.
- Scene-to-target rendering contracts via `SceneRepresentation` and `TargetRenderSpec`.
- Observation-conditioned field rendering and fitting.
- Context-aware decomposition and shared/private component machinery.
- Learned-operator scaffolding.
- Synthetic trunk pretraining for STRAINER-style fields.
- Real-data and parity-oriented benchmark culture.

### 6.2 Still incomplete

- A true encoder-from-context world-model baseline.
- A first-class context/target split API across real observations.
- Real learned nuisance fields rather than slot scaffolding alone.
- Latent stability and misspecification benchmarks.
- Probe-driven scientific evaluation beyond fit quality.
- A clear public observation abstraction that unifies image-plane and field-style inputs.

The white paper must describe the proposal in a way that is consistent with both lists.

---

## 7. Relation To Existing Docs

The new white paper should extend existing docs rather than compete with them.

### 7.1 `docs/ROADMAP.md`

Keep:

- inverse-problem framing,
- sky-away-from-detectors goal,
- pretrained trunk logic,
- multi-survey fusion ambition.

Extend with:

- predictive-state framing,
- stronger operator-versus-latent separation,
- a research-layer split outside `torchsky`,
- explicit falsification criteria.

### 7.2 `docs/map_operator_progress.md`

Treat this as the reality anchor for what is implemented versus scaffolded.

The white paper should inherit its honesty:

- real shared substrate,
- real decomposition machinery,
- incomplete learned nuisance models,
- incomplete identifiability framework.

### 7.3 `docs/AUDIT_MAPPING_INR.md`

Use this as the methodological discipline for evaluation:

- parity,
- real-data benchmarks,
- honest replacement boundaries,
- photometric and operator contracts.

The world-model paper should inherit this standard instead of building a separate research-evaluation culture.

---

## 8. Recommended White Paper Structure

The full white paper should be written in the following order.

### Section 1. Abstract

Short, disciplined, and modest.

It should say:

- what the gap is,
- what is proposed,
- what is not being claimed,
- and how the software split works.

### Section 2. Motivation

Set up the gap between:

- ontology-first field inference,
- generic latent world models,
- and the proposed middle path.

### Section 3. Scientific Positioning

Define the proposal in one sentence.

State clearly what it is not.

### Section 4. Relation To Physical Fields

Keep the field analogy.

Keep the anti-overclaim section.

This is one of the best parts of the revised thinking and should remain central.

### Section 5. Mathematical Formulation

Define:

- observations and metadata,
- context and target sets,
- latent field domain,
- encoder,
- scene adapter,
- observation operator,
- prediction objective,
- optional scale-consistency term,
- optional probabilistic extension.

### Section 6. Failure Modes

This needs to be a full section, not a quick caveat list.

It should include:

- predictive versus cosmological sufficiency,
- nuisance entanglement,
- weak identifiability,
- nonphysical scale axes,
- calibration difficulty,
- weak interventions in astronomy.

### Section 7. Software Architecture

This section should be very concrete.

- What belongs in `torchsky`.
- What belongs in a future `torchsky-world`.
- Why `cosmapper` is a source of ideas and probes, not the core implementation.

### Section 8. Empirical Program

This is the section the earlier drafts needed more of.

It should define the actual tests needed before the program earns confidence.

### Section 9. Implementation Phases

Start with the deterministic predictive baseline.

Do not lead with probabilistic latents or dynamic skies.

### Section 10. Broader Applications

Keep short and secondary.

### Section 11. Conclusion

End on the strongest honest claim:

if explicit field-level inference is best when ontology is trusted, then continuous latent field world models are attractive when ontology itself should remain partly learnable.

---

## 9. Empirical Program The White Paper Must Commit To

This is the most important addition from the current view.

The paper should commit to the following evaluation ladder.

### Phase A. Predictive baseline

Goal:

- infer from context observations,
- render held-out target observations,
- beat strong coadd and field-fit baselines where appropriate.

Metrics:

- RMSE,
- robust residual metrics,
- NLL where probabilistic predictions are used,
- stratification by instrument, band, mask fraction, and PSF regime.

### Phase B. Cross-view generalization

Train on one subset of views and predict another:

- different bands,
- different epochs,
- different PSFs,
- different surveys or instruments.

This is where the world-model claim starts to become real.

### Phase C. Nuisance disentanglement

Test whether the latent changes too much under operator perturbations that should remain in the observation model:

- masks,
- PSF shifts,
- exposure subsets,
- calibration variation.

Use explicit leakage diagnostics where possible.

### Phase D. Probe usefulness

Attach:

- field decoders,
- cosmology score heads,
- parameter heads,
- anomaly or misspecification heads.

Ask whether these probes recover meaningful scientific structure.

### Phase E. Failure sensitivity

The system must identify when it is confidently wrong:

- unseen instrument regimes,
- unusual morphology,
- domain shift,
- bad calibration regimes.

If it cannot do that, it is not scientifically trustworthy.

---

## 10. Software Orientation Beyond The Current Docs

This is the strategic recommendation that should guide repo work.

### 10.1 Keep `torchsky` focused on substrate

`torchsky` should absorb only the pieces that remain useful across many model families:

- unified observation abstractions,
- metadata containers,
- operator bundles,
- rendering contracts,
- scene representations,
- reusable continuous-field backbones,
- benchmark and evaluation surfaces.

### 10.2 Do not turn `torchsky` into the research stack

The following should stay out of the core repo once they become specific:

- context/target world-model training loops,
- latent-state encoders specialized for predictive pretraining,
- paper-specific ablation code,
- cosmology-probe training logic,
- speculative probabilistic latent machinery.

### 10.3 Build the next layer as a thin research system

The future research repo should initially be as small as possible:

- observation-region sampler,
- context/target splitter,
- deterministic latent encoder,
- scene adapter,
- render loss,
- evaluation harness.

It should depend on `torchsky`, not fork it in spirit.

---

## 11. Concrete Repo Tasks Implied By This Plan

Before the full white paper is written, the codebase should move in these directions.

1. Introduce a clearer public observation abstraction spanning image-plane and field-style workflows.
2. Promote scene/operator/target contracts from semi-internal to first-class substrate.
3. Add a real held-out observation prediction benchmark over calibrated exposures.
4. Implement at least one real learned nuisance family on top of the existing learned-operator slots.
5. Add latent-stability and misspecification evaluation alongside reconstruction metrics.
6. Keep decomposition, gauge, and leakage diagnostics tied to real-data settings rather than toy-only examples.

These are the implementation moves that would make the future white paper feel earned.

---

## 12. Writing Plan

The full white paper should be drafted in this order.

1. Write the motivation, positioning, and anti-claims sections first.
2. Write the software architecture and repo-split sections second.
3. Write the empirical-program section third.
4. Write the mathematical section only after the empirical contract is fixed.
5. Write the introduction and conclusion last.

Reason:

the biggest risk is not mathematical incompleteness, but rhetorical drift away from what the project can actually defend.

---

## 13. Deliverable Standard

The finished white paper will be good enough when:

- it can be handed to a skeptical cosmologist without sounding naive,
- it can be handed to an ML researcher without sounding vague,
- it can be handed to a systems engineer without sounding architecturally confused,
- and it gives a concrete answer to the question:
  why is this a new research layer built on `torchsky`, rather than just another name for the current INR code?

That is the bar.
