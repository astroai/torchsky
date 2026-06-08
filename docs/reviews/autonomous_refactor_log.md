# Autonomous Refactor Log

Target branch: `codex/autonomous-refactor-2026-05`

Start gate: May 8, 2026, `America/Vancouver`

## Current State

- Resumed manually on June 8, 2026 after the scheduled automation did not run.
- Active branch: `codex/autonomous-refactor-2026-05`.
- Completed A0, A1, and A2 from
  `docs/reviews/autonomous_refactor_launch.md`.
- Next recommended slice: B1/B3, extracting root FITS I/O dispatch internals and
  making `torchsky.io` call implementation modules directly without
  user-visible churn.

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
- Extracted root FITS I/O Python cache state and generic cache operations into
  `torchsky._io_engine.caches`.
- Kept root private aliases and public wrappers (`get_cache_performance`,
  `clear_file_cache`, `_invalidate_path_caches`) intact so read/write dispatch
  behavior and tests remain stable.
- Added a direct private-module test for clearing data, handles, metadata, HDU
  type, auto-mmap, auto-HDU, and stats caches.
- Tests run:
  - `pixi run -e test -- pytest tests/test_io_engine_caches.py tests/test_cache.py::TestCaching::test_cache_performance_tracking tests/test_cache.py::TestCaching::test_cache_clearing tests/test_cache.py::TestCaching::test_multiple_file_caching tests/test_cache.py::TestCaching::test_cold_nommap_heuristic_with_cache_enabled_int16 tests/test_cache.py::TestCaching::test_read_mmap_auto_defaults_to_false_for_compressed tests/test_cache.py::TestCaching::test_read_mmap_true_is_explicit_override tests/test_api.py::TestMainAPI::test_cache_functions -q`
  - Result: `8 passed, 2 warnings`.
  - `pixi run -e test -- pytest tests/test_cold_import.py -q`
  - Result: `1 passed`.
- North Star connection: this moves root I/O policy toward a testable private
  subsystem while preserving the lean public facade and existing FITS behavior.

## Work Queue Snapshot

- [x] A0: Establish branch, baseline, and safety checks.
- [x] A1: Declare the public modelling spine in docs and API-surface metadata.
- [x] A2: Add or tighten tests for public API tiers and cold import behavior.
- [ ] B1: Extract root FITS I/O implementation without user-visible churn.
- [x] B2: Move Python-side root I/O caches behind a single cache facade.
- [ ] B3: Make `torchsky.io` call the implementation directly.
- [ ] B4: Preserve root FITS public behavior.
- [ ] C1: Add promoted observation/product/scene facades.
- [ ] C2: Define executable operator protocols or adapters.
- [ ] D1: Move HEALPix ownership toward `torchsky.sphere`.
- [ ] D2: Mark `coords` and `geom` as compatibility aliases.
- [ ] E1: Add `torchsky info` and `torchsky cat` CLI workflows.
- [ ] F1: Rationalize field model exposure through presets and factories.
- [ ] G1: Continue roadmap implementation from `docs/roadmap/status.md`.
