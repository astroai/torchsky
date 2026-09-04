# Autonomous Refactor Launch Plan

This file is the handoff contract for the scheduled Codex refactor run.

## Launch Contract

- Start date: May 8, 2026 in the `America/Vancouver` timezone.
- Runtime model: Codex with `gpt-5.5`, reasoning effort `xhigh`.
- Target branch: `codex/autonomous-refactor-2026-05`.
- Workspace: `/Users/fabbros/src/sfabbro/torchsky`.
- Restart cadence: once per day for about one month.
- Primary state file: `docs/reviews/autonomous_refactor_log.md`.

If a run starts before May 8, 2026, it must not modify files. It should only report
that the launch gate has not opened.

## North Star

The refactor is not ordinary cleanup. It must keep `torchsky` aimed at becoming
the backbone of a differentiable, any-device, torch-native library and tool stack
for modelling observations of the sky.

The high-level goal is to model any part and any structure of the observable sky:
foregrounds, instruments, backgrounds, explicit physical components, learned
implicit components, and hybrid scenes. The software should extract maximal joint
information from heterogeneous data using smart optimization algorithms, learned
models where data can beat assumption-heavy physical models, and physical models
where they remain unbeatable. It should work across wavelength, instrument type,
and data modality.

Every automated run should use this decision filter:

- Prefer architecture that supports joint observation modelling over isolated
  FITS/WCS/coadd utilities.
- Preserve torch-native, differentiable, cross-device paths wherever the touched
  code claims that behavior.
- Keep the core runtime lean and avoid optional astronomy ecosystem imports in
  `src/torchsky`.
- Make the public API and CLI clean, lean, easy to adopt, and low-jargon.
- Reduce redundant namespaces and long public names only through compatibility
  shims, tests, and staged migrations.
- Build primitives that remain viable at survey scale: chunking, streaming,
  tiling, sparse representations, explicit operators, and benchmarked kernels.
- Use established astronomy packages as references, fixtures, comparators, and
  compatibility targets, not as hidden runtime backends.
- Avoid shallow cleanup that does not materially improve architecture,
  usability, scalability, differentiability, or correctness.

## Daily Bootstrap

Each daily run should begin by running:

```bash
pixi run -e default -- python scripts/autonomous_refactor_status.py --fail-before-start
git status --short --branch
```

Then:

1. Switch to `codex/autonomous-refactor-2026-05`, creating it from the current
   checked-out branch if it does not exist.
2. Read `AGENTS.md`, all files in `docs/reviews/`, and
   `docs/roadmap/status.md` before choosing work.
3. Re-read the North Star above and use it to reject work that is only cosmetic
   or moves the package away from joint differentiable sky modelling.
4. Read `docs/reviews/autonomous_refactor_log.md` and continue the first
   unchecked or incomplete item.
5. Keep changes small enough that focused tests can run in the same session.
6. Update the log with the exact work done, tests run, failures, next step, and
   how the slice supports the North Star.
7. Commit only coherent green slices. Push only after `pixi run ci-local` passes.

Do not wait for the maintainer. Make conservative choices that preserve existing
public behavior and import compatibility.

## Work Queue

The queue is ordered to match the safest common path across
`docs/reviews/arch_review_1.md`, `docs/reviews/arch_review_2.md`, and
`docs/reviews/arch_review_3.md`.

- [ ] A0: Establish branch, baseline, and safety checks.
- [ ] A1: Declare the public modelling spine in docs and API-surface metadata.
- [ ] A2: Add or tighten tests for public API tiers and cold import behavior.
- [ ] B1: Extract root FITS I/O implementation from `src/torchsky/__init__.py`
      into implementation modules without user-visible churn.
- [ ] B2: Move Python-side root I/O caches behind a single cache facade with
      `clear` and `stats` behavior.
- [ ] B3: Make `torchsky.io` call the implementation directly rather than
      routing backward through the package root.
- [ ] B4: Keep existing `torchsky.read`, `torchsky.write`, `torchsky.open`,
      `torchsky.get_header`, and `torchsky.get_wcs` behavior stable.
- [ ] C1: Add a promoted observation/product/scene facade that wraps existing
      `ImageObservation`, `FieldObservation`, `TargetRenderSpec`,
      `MapProductSpec`, and `SceneRepresentation` concepts.
- [ ] C2: Define executable operator protocols or adapters for the existing
      observation operator bundle, with parity tests against current behavior.
- [ ] D1: Move HEALPix ownership toward `torchsky.sphere`.
- [ ] D2: Remove leftover HEALPix, coords, and geom alias surfaces so the public
      API points cleanly at `torchsky.wcs` and `torchsky.sphere`.
- [ ] E1: Add `torchsky info` and `torchsky cat` read-only CLI workflows with
      JSON output.
- [ ] F1: Rationalize field model exposure through presets and factories while
      keeping current expert imports available.
- [ ] G1: Continue roadmap implementation from `docs/roadmap/status.md`, starting
      with WCS, sphere/tiling, map architecture, and INR/field-model next work.

## Engineering Rules

- Use Pixi for all project Python commands.
- Do not add runtime dependencies outside the existing core contract.
- Do not import optional astronomy ecosystem packages from `src/torchsky`.
- Do not remove or break existing public imports without compatibility shims.
- Do not perform sweeping directory renames until facades and tests are in place.
- Do not do tree-wide formatting or unrelated cleanup.
- Preserve CPU behavior first and keep CUDA/MPS semantics where the touched path
  claims support.
- Prefer exact, tested semantics over heuristics.

## Test Policy

For every slice:

- Run the narrowest relevant tests first with `pixi run -e test -- pytest ...`.
- Run `pixi run preflight-push` before committing non-doc code.
- Run `pixi run ci-local` before pushing.
- If a full gate is too slow or fails for unrelated baseline reasons, record the
  command, failure summary, and residual risk in the log.

## Commit Policy

Use clear commits on `codex/autonomous-refactor-2026-05`. A good commit shape is:

```text
refactor(io): extract FITS cache facade
```

Commit messages should explain behavior preservation and tests. Avoid committing
known-broken code unless the log makes the blocker explicit and the next run needs
the checkpoint.

## Completion Criteria

The branch is review-ready when:

- `pixi run ci-local` passes on the branch.
- The log has a final summary of completed queue items and remaining risks.
- Public import compatibility is tested for any moved surface.
- Docs describe the promoted architecture and any compatibility aliases.
