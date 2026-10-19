# Product architecture and wavelength scope

This note describes torchsky's product direction. It is not a parity matrix,
feature-completion claim, or experiment schedule. Current validation and
measured results are maintained in
[`torchsky-harness`](https://github.com/astroai/torchsky-harness).

## Product mission

Torchsky is building a torch-native astronomy stack for scalable multi-modal
sky data, differentiable observation pipelines, and continuous spatial-spectral
models where those methods are technically coherent. The core should remain a
computational astronomy library rather than a thin wrapper around legacy
packages.

The product divides responsibility along stable data and compute boundaries:

| Responsibility | Owner | Interface |
|---|---|---|
| FITS image/table I/O and table storage | `torchfits` | Tensors, header mappings, Arrow/table batches |
| Coordinates, WCS, spherical geometry, and catalog spatial operations | `torchsky` | Tensor-native geometry and transforms |
| Map products, scene composition, and observation operators | `torchsky` | Explicit products, metadata, masks, weights, and operators |
| Continuous spatial-spectral fields | `torchsky` | Differentiable field, fit, render, and sample APIs |
| Parity campaigns, comparator workflows, scientific scoring, and evidence | `torchsky-harness` | Installed subject plus revision-pinned evaluations |

These responsibilities are documented in the [package roadmap](status.md) and
[API guide](../api.md).

## Multi-wavelength scope

Wavelength-aware products should compose established primitives: spectral axes,
bandpass integration, coordinate transforms, detector/observation operators,
and explicit tensor products. The design should account for the different
sampling, noise, and response contracts of optical, infrared, radio, and other
domains without requiring one model or one reference library to define every
instrument.

Scientific formulation for differentiable reconstruction and spatial-spectral
representations is preserved in
[`projects/inr_foam_problem_formulation.md`](../projects/inr_foam_problem_formulation.md).
Those formulations are research context; they do not assert that a particular
model has passed real-data acceptance.

## Engineering horizon

Future package work should address concrete runtime needs: correctness of
shared geometry and observation operators, survey-scale execution, explicit
uncertainty and calibration contracts, and stable interoperability at data
boundaries. Work proceeds through reviewed implementation changes with bounded
tests and documented API behavior. It does not create speculative runtime
packages or directories before a concrete implementation exists.

For the current implementation priorities and verification rules, see
[`AGENTS.md`](../../AGENTS.md) and [`next.md`](next.md). The harness owns the
comparator methodology, historical evidence, and scientific campaign status.
