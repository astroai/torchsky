# Autonomous Refactor Log

Target branch: `codex/autonomous-refactor-2026-05`

Start gate: May 8, 2026, `America/Vancouver`

## Current State

- Resumed manually on June 8, 2026 after the scheduled automation did not run.
- Active branch: `codex/autonomous-refactor-2026-05`.
- Completed A0, A1, and A2 from
  `docs/reviews/autonomous_refactor_launch.md`.
- Next recommended slice: B1/B2, extracting root FITS I/O internals and cache
  policy without user-visible churn.

## Daily Entries

### 2026-05-02

- Added the autonomous launch plan and status script.
- Added the North Star and daily decision filter so future runs keep the
  differentiable, any-device, torch-native joint sky-modelling mission ahead of
  cosmetic cleanup.
- No refactor implementation was started before the May 8 launch gate.

### 2026-06-08

- Confirmed the scheduled automation had not made implementation progress.
- Created and switched to `codex/autonomous-refactor-2026-05`.
- Added `torchsky.SurfacePillar` plus
  `torchsky.api_surfaces_by_pillar(...)` so public API metadata can express the
  intended `io` / `world` / `scene` / `viz` architecture spine.
- Assigned current declared surfaces to pillars without moving implementation or
  breaking public import paths.
- Added `docs/architecture.md` and linked it from `README.md`, `docs/README.md`,
  and `docs/api.md`.
- Tests run:
  - `pixi run -e test -- pytest tests/test_api_surfaces.py -q`
  - Result: `4 passed, 2 warnings`.
- North Star connection: this slice turns the high-level joint sky-modelling
  philosophy into code-visible metadata and user-facing documentation, giving
  later refactor slices a stable target before any directory moves.
- Added a focused cold-import regression test proving root `import torchsky`
  keeps `fastapi`, `torchsky.viz`, and `torchsky.map.field` lazy.
- Added an API-surface test requiring every declared public surface to have an
  architecture pillar.
- Tests run:
  - `pixi run -e test -- pytest tests/test_api_surfaces.py tests/test_cold_import.py -q`
  - Result: `6 passed, 2 warnings`.
- North Star connection: this prevents the public facade from accidentally
  pulling experimental field or browser-server surfaces into the core import
  path while the larger I/O extraction proceeds.

## Work Queue Snapshot

- [x] A0: Establish branch, baseline, and safety checks.
- [x] A1: Declare the public modelling spine in docs and API-surface metadata.
- [x] A2: Add or tighten tests for public API tiers and cold import behavior.
- [ ] B1: Extract root FITS I/O implementation without user-visible churn.
- [ ] B2: Move Python-side root I/O caches behind a single cache facade.
- [ ] B3: Make `torchsky.io` call the implementation directly.
- [ ] B4: Preserve root FITS public behavior.
- [ ] C1: Add promoted observation/product/scene facades.
- [ ] C2: Define executable operator protocols or adapters.
- [ ] D1: Move HEALPix ownership toward `torchsky.sphere`.
- [ ] D2: Mark `coords` and `geom` as compatibility aliases.
- [ ] E1: Add `torchsky info` and `torchsky cat` CLI workflows.
- [ ] F1: Rationalize field model exposure through presets and factories.
- [ ] G1: Continue roadmap implementation from `docs/roadmap/status.md`.
