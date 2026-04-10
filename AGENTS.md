# AGENTS.md

This file gives repository-specific operating guidance for coding agents and maintainers working in `torchsky`.

## Mission

`torchsky` is building a torch-native astronomy stack for:

- very high performance on large workloads
- multi-wavelength and multi-modal sky data
- scalability to many very large datasets
- continuous, field-like modelling / prediction / generation where that is technically coherent
- differentiable pipelines where that is technically coherent
- cross-device execution across CPU, CUDA, and MPS when practical
- compatibility layers with established astronomy libraries, without forcing primitive-for-primitive mimicry where a higher-level compatibility surface is better

The project should prefer designs that remain viable at survey scale, not just toy examples or notebook-sized inputs.
The core should not collapse into a thin wrapper around legacy astronomy packages.
Geometric primitives, mapmaking, and cross-wavelength field representations are central building blocks, not side surfaces.

## Core Principles

### 1. No heuristics where exact or well-founded methods are available

- Do not introduce ad hoc thresholds, fallback rules, or data-dependent branching unless there is a clear numerical, algorithmic, or benchmark-backed reason.
- If a heuristic is temporarily necessary, isolate it, document why it exists, and add a benchmark or regression test that justifies it.
- Prefer mathematically grounded algorithms, explicit error bounds, or battle-tested upstream semantics.

### 2. Build forward-looking primitives; use established libraries as references, comparators, or compatibility targets

- Core primitives should be designed for the mission goals above: scalability, differentiability, cross-device execution, and high performance on modern workloads.
- Do not shape the internal architecture around legacy astronomy package boundaries if that blocks better primitives.
- Use established astronomy libraries primarily for:
  - semantic comparison
  - regression checks
  - migration and compatibility layers
  - upstream fixtures and validation data
- Use battle-tested non-astronomy libraries where they are the best fit for numerical kernels, systems concerns, storage, or compiler/runtime leverage.
- Compatibility does not require matching another project's internal primitives; it requires matching trusted behavior at a useful API layer when compatibility is an explicit goal.

### 3. Optimize for scalability first

- Assume catalogs, maps, cubes, and tables may be large enough that naive `O(N*M)` behavior is unacceptable.
- Favor structured indexing, chunking, streaming, sparse representations, and algorithmic pruning before native-kernel work.
- Native C++ kernels are justified after algorithmic structure is sound and the remaining bottleneck is real.

### 4. Preserve differentiability and tensor-native execution

- Keep user-facing compute paths in PyTorch unless there is a strong reason not to.
- Avoid unnecessary host transfers, NumPy-only detours, or CPU-only assumptions in public tensor paths.
- When adding fast paths, keep a correct tensor-native fallback unless the project explicitly decides otherwise.

### 5. Cross-device behavior matters

- CPU is the baseline.
- CUDA and MPS support should be preserved where the surrounding module claims it.
- Device-specific optimizations must not silently change semantics.

## API and Compatibility Policy

- Public APIs should stay thin and stable unless there is a strong reason to widen them.
- Prefer compatibility layers that help users migrate from established astronomy tooling:
  - `healpy`
  - `astropy-healpix`
  - `spherical-geometry`
  - `astropy` / `fitsio`
- See `docs/compatibility.md` for how these map onto torchsky modules (e.g. HEALPix/healpy-shaped APIs under `torchsky.sphere.compat`).
- Compatibility work should target behavior users depend on, test fixtures, and operational workflows.
- The compatibility layer should sit on top of torchsky-native primitives where possible, not define the primitive architecture.
- Avoid exposing low-level internal kernels as public API unless repeated use or composability clearly justifies them.

## Performance Policy

- Every performance-sensitive change should answer:
  - what algorithmic bottleneck is being removed
  - why this scales better at large `N`
  - what the benchmark evidence is
- Prefer:
  - algorithmic improvements
  - structured indices
  - data layout improvements
  - chunking/streaming
  - selective native kernels for proven hotspots
- Benchmark both small/medium and large workloads when the optimization could shift crossover points.
- Do not force a native fast path by default if it regresses realistic medium-size workloads.

## Testing Policy

All substantive work should be validated on both simulated and real data when possible.

### Simulated / synthetic data

Use synthetic data to:

- stress corner cases
- isolate numerical behavior
- generate large-scale workloads cheaply
- test randomized parity against brute force or known baselines

### Real data or upstream fixtures

Use packaged fixtures, upstream test data, or representative real astronomy data to:

- confirm semantic compatibility
- catch edge cases synthetic generators miss
- validate interoperability claims

Examples already present in this repo include upstream replay scripts, parity suites, and FITS-based integration tests. Extend those patterns rather than inventing one-off checks.

### For new features or optimizations

Try to include:

- correctness tests
- parity tests against a trusted baseline when available
- benchmark evidence
- at least one realistic-data or upstream-fixture check for user-facing behavior

## Implementation Preferences

- Prefer `torch` operations over Python loops in user-facing hot paths.
- Prefer `rg` for search.
- Use `apply_patch` for manual file edits.
- Keep comments sparse and high signal.
- Do not add broad new dependencies casually.
- Do not replace exact semantics with approximate shortcuts unless the tradeoff is explicit and tested.
- Do not reduce torchsky to a wrapper around `scamp`, `sextractor`, `swarp`, `psfex`, LSST Stack components, `drizzle`, `astropy`, `fitsio`, `healpy`, or similar packages.

## Native Code Guidance

- Add C++ only after confirming the Python/Torch algorithm is already the right one.
- Keep native kernels focused on real hotspots.
- Maintain correctness parity tests against the non-native path.
- Expose native code through narrow bindings; keep orchestration and policy in Python unless there is a compelling reason otherwise.

## Benchmarks and Evidence

Relevant benchmark surfaces live in:

- `benchmarks/`
- `docs/benchmarks.md`
- upstream replay scripts under `benchmarks/replays/`

For sphere and catalog work specifically:

- maintain explicit parity against trusted upstream behavior where available
- report throughput and candidate-reduction effects, not just wall time
- measure both default paths and gated fast paths when a native kernel is optional

## Change Checklist

Before finishing a non-trivial change, check:

1. Is the algorithm scalable enough for large datasets?
2. Did we preserve or improve differentiability and device behavior?
3. Did we avoid unnecessary heuristics?
4. Did we compare against a trusted library or upstream fixture where possible?
5. Did we test on both simulated and realistic data or fixtures?
6. Did we add benchmark evidence if this change claims performance value?
7. Did we avoid widening the public API unnecessarily?

## Documentation Expectations

- Update user-facing docs if public behavior changes.
- Update benchmark docs when benchmark surfaces or benchmark policy change.
- Keep repository guidance in this file aligned with actual practice.

## Agent PR hygiene and lint gates

- Do **not** open PRs that only remove one unused import, reformat a single file, or repeat the same micro-opt across many PRs. Batch mechanical cleanup into **one** commit/PR, or fold it into substantive work.
- Performance-related PRs must satisfy **Performance Policy** above (bottleneck, scaling story, benchmark evidence).
- **Before every push**, run `pixi run lint` (and tests as appropriate for your change). Automated agents must not push if `ruff check` fails.
- Optional: `pre-commit install` to mirror `.pre-commit-config.yaml` locally.
