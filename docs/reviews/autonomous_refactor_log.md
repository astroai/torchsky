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
- Moved `ReadOptions` and `ReadOptionsKwargs` into
  `torchsky._io_engine.options`, leaving root overloads and runtime behavior
  unchanged.
- Tests run:
  - `pixi run -e test -- pytest tests/test_io.py tests/test_api.py::TestMainAPI::test_read_basic tests/test_api.py::TestMainAPI::test_read_mode_image tests/test_api.py::TestMainAPI::test_cache_functions -q`
  - Result: `23 passed, 2 warnings`.
  - `pixi run -e test -- pytest tests/test_api.py -k 'read_mode_table' -q`
  - Result: `1 passed, 31 deselected, 2 warnings`.
- North Star connection: this is a small B1 step toward a private, testable I/O
  engine while keeping user-facing reads stable.
- Moved deterministic low-level `read_image` validation and C++ dispatch into
  `torchsky._io_engine.image`, leaving the root `torchsky.read_image` wrapper
  and return semantics unchanged.
- Tests run:
  - `pixi run -e test -- pytest tests/test_api.py::TestMainAPI::test_read_image_basic tests/test_api.py::TestMainAPI::test_read_image_matches_read tests/test_api.py::TestMainAPI::test_read_image_rejects_auto_hdu tests/test_api.py::TestMainAPI::test_read_image_handle_cache_flag tests/test_api.py::TestMainAPI::test_read_mode_image -q`
  - Result: `5 passed, 2 warnings`.
  - `pixi run -e test -- pytest tests/test_io.py -q`
  - Result: `20 passed, 2 warnings`.
- North Star connection: this moves an expert low-level I/O path behind a
  private engine boundary while preserving deterministic tensor-native image
  reads for performance profiling and downstream tools.
- Moved the specialized `read_table` wrapper into
  `torchsky._io_engine.table_api`, with root `torchsky.read_table` delegating to
  that implementation through the existing unified `read(...)` function.
- Tests run:
  - `pixi run -e test -- pytest tests/test_api.py -k 'read_table or read_mode_table' -q`
  - Result: `8 passed, 24 deselected, 2 warnings`.
  - `pixi run -e test -- pytest tests/test_io.py tests/test_cold_import.py -q`
  - Result: `21 passed, 2 warnings`.
- North Star connection: this keeps table reads on the same behavior path while
  reducing root-module policy weight and preserving the lean user API.
- Moved `read_hdus` and the shared image batch device helper into
  `torchsky._io_engine.image`, keeping root `torchsky.read_hdus` as the public
  wrapper.
- Tests run:
  - `pixi run -e test -- pytest tests/test_api.py::TestMainAPI::test_read_hdus_direct tests/test_io.py -q`
  - Result: `21 passed, 2 warnings`.
  - `pixi run -e test -- pytest tests/test_cold_import.py tests/test_io_engine_caches.py -q`
  - Result: `2 passed, 2 warnings`.
- North Star connection: this keeps direct multi-HDU tensor image reads
  explicit and testable while continuing to reduce the root facade's
  implementation load.
- Moved `read_subset`, `SubsetReader`, and `open_subset_reader` into
  `torchsky._io_engine.subset`, keeping root wrappers and the existing
  Python-side handle-cache policy unchanged.
- Tests run:
  - `pixi run -e test -- pytest tests/test_api.py -k 'subset or read_basic or cache_functions' tests/test_io.py -q`
  - Result: `4 passed, 48 deselected, 2 warnings`.
  - `pixi run -e test -- pytest tests/test_cutout_performance_api.py tests/test_cold_import.py -q`
  - Result: `2 passed, 2 warnings`.
  - `pixi run -e test -- pytest tests/test_io.py -q`
  - Result: `20 passed, 2 warnings`.
- North Star connection: this preserves efficient tensor cutout workflows while
  continuing to separate root API routing from private I/O implementation.
- Moved `open`, `get_header`, and `get_wcs` implementation logic into
  `torchsky._io_engine.hdu_api`, keeping root wrappers and auto-HDU behavior
  intact.
- Tests run:
  - `pixi run -e test -- pytest tests/test_api.py -k 'get_header or get_wcs or top_level_namespaces or read_by_extname or read_hdu_auto or cache_functions' -q`
  - Result: `5 passed, 27 deselected, 2 warnings`.
  - `pixi run -e test -- pytest tests/test_header_cards.py tests/test_io.py tests/test_cold_import.py -q`
  - Result: `27 passed, 2 warnings`.
- North Star connection: this keeps HDU/header/WCS access as a lean public route
  while moving file-format implementation details into the private I/O engine.
- Moved `read_large_table` and `stream_table` implementation logic into
  `torchsky._io_engine.table_streaming`, keeping root convenience wrappers and
  chunking behavior intact.
- Tests run:
  - `pixi run -e test -- pytest tests/test_api.py -k 'large_table or stream_table or read_table' tests/test_table.py -k 'large_table or stream_table' -q`
  - Result: `3 passed, 52 deselected, 2 warnings`.
  - `pixi run -e test -- pytest tests/test_io.py tests/test_cold_import.py -q`
  - Result: `21 passed, 2 warnings`.
- North Star connection: this keeps survey-scale table streaming and chunking as
  an explicit private I/O subsystem while preserving public convenience APIs.
- Moved `read_batch` and `get_batch_info` implementation logic into
  `torchsky._io_engine.batch`, keeping root wrappers and C++ batch fast-path
  behavior intact.
- Tests run:
  - `pixi run -e test -- pytest tests/test_api.py -k 'batch or cache_functions' tests/test_io.py -q`
  - Result: `6 passed, 46 deselected, 2 warnings`.
- North Star connection: this keeps survey-scale multi-file tensor ingestion
  explicit while continuing to shrink root I/O implementation code.
- Updated `torchsky.io` so extracted surfaces (`read_image`, `read_hdus`,
  `read_table`, `open`, `stream_table`, and `read_large_table`) call
  `_io_engine` implementation modules instead of routing back through the root
  wrappers. Root-only surfaces remain delegated until their implementation is
  moved.
- Tests run:
  - `pixi run -e test -- pytest tests/test_io.py tests/test_api.py -k 'top_level_namespaces or read_image or read_hdus or read_table or large_table or stream_table' tests/test_header_cards.py::test_card_exported_on_public_io_surfaces -q`
  - Result: `14 passed, 39 deselected, 2 warnings`.
  - `pixi run -e test -- pytest tests/test_cold_import.py -q`
  - Result: `1 passed`.
- North Star connection: this starts correcting the reverse-alias `torchsky.io`
  architecture while preserving the lean namespace users already import.
- Moved Python-side file signature, CFITSIO handle-cache, and cached HDU-type
  helpers into `torchsky._io_engine.caches`, keeping the existing root private
  names as compatibility aliases for the remaining root dispatch code.
- Tests run:
  - `pixi run -e test -- pytest tests/test_cache.py tests/test_api.py -k 'read_hdu_auto or get_header or read_by_extname or read_hdus or cache_functions or subset' tests/test_io.py tests/test_io_engine_caches.py tests/test_cold_import.py -q`
  - Result: `8 passed, 85 deselected, 2 warnings`.
  - `pixi run -e test -- pytest tests/test_api.py tests/test_io.py tests/test_cache.py tests/test_io_engine_caches.py tests/test_cold_import.py -q`
  - Result: `93 passed, 2 warnings`.
- North Star connection: this concentrates stale-file and open-handle policy in
  one cache module, which makes the remaining FITS root facade smaller without
  changing tensor IO semantics or adding runtime dependencies.
- Moved image metadata parsing and mmap auto-policy helpers into
  `torchsky._io_engine.image_meta`, keeping root wrappers for the existing
  monkeypatchable private names used by tests and remaining root dispatch code.
- Tests run:
  - `pixi run -e test -- pytest tests/test_io.py::TestIORead::test_read_basic tests/test_io.py::TestIORead::test_read_scale_on_device_scaled tests/test_io.py::TestIORead::test_read_scale_on_device_not_scaled tests/test_io.py::TestIORead::test_read_use_cache_raw_scale tests/test_io.py::TestIORead::test_read_use_cache_no_raw_scale tests/test_cache.py::TestCaching::test_cold_nommap_heuristic_with_cache_enabled_int16 -q`
  - Result: `6 passed, 2 warnings`.
  - `pixi run -e test -- pytest tests/test_api.py tests/test_io.py tests/test_cache.py tests/test_io_engine_caches.py tests/test_cold_import.py -q`
  - Result: `93 passed, 2 warnings`.
- North Star connection: this isolates image-read policy from the public facade
  while preserving exact FITS header normalization, mmap behavior, and lean core
  runtime dependencies.
- Moved fast header probing, first-payload HDU discovery, and auto-HDU cache
  resolution into `torchsky._io_engine.hdu_api`, keeping root private aliases
  for the remaining root read paths.
- Tests run:
  - `pixi run -e test -- pytest tests/test_api.py -k 'read_hdu_auto or get_header or get_wcs or read_by_extname or read_hdus or top_level_namespaces' tests/test_header_cards.py tests/test_io.py tests/test_cache.py tests/test_cold_import.py -q`
  - Result: `7 passed, 91 deselected, 2 warnings`.
  - `pixi run -e test -- pytest tests/test_api.py tests/test_io.py tests/test_cache.py tests/test_io_engine_caches.py tests/test_header_cards.py tests/test_cold_import.py -q`
  - Result: `99 passed, 2 warnings`.
- North Star connection: this keeps HDU selection and compressed-image probing
  near the HDU/header API instead of in the package root, while preserving
  payload preference semantics for image-first tensor workflows.
- Moved CPU and generic image fast-path dispatch helpers into
  `torchsky._io_engine.read_dispatch`, keeping root private wrappers for the
  remaining unified `read(...)` orchestrator and mocked C++ test seams.
- Tests run:
  - `pixi run -e test -- pytest tests/test_io.py tests/test_cache.py tests/test_api.py -k 'read_image or read_hdu_auto or read_hdus or cache_functions or read_basic or raw_scale or scale_on_device or no_cache' tests/test_cold_import.py -q`
  - Result: `17 passed, 75 deselected, 2 warnings`.
  - `pixi run -e test -- pytest tests/test_api.py tests/test_io.py tests/test_cache.py tests/test_io_engine_caches.py tests/test_header_cards.py tests/test_cold_import.py -q`
  - Result: `99 passed, 2 warnings`.
- North Star connection: this keeps hot image tensor dispatch isolated and
  testable while preserving device conversion, mmap auto-policy, and cache
  accounting semantics.
- Moved the unified read cache-hit check into `torchsky._io_engine.caches`,
  leaving root `read(...)` with a wrapper while cache counters, stale-file
  detection, and device transfer of cached data are owned by the cache facade.
- Tests run:
  - `pixi run -e test -- pytest tests/test_cache.py tests/test_io.py tests/test_api.py -k 'cache_functions or read_basic or use_cache or no_cache or read_image or read_hdu_auto' tests/test_io_engine_caches.py tests/test_cold_import.py -q`
  - Result: `14 passed, 79 deselected, 2 warnings`.
  - `pixi run -e test -- pytest tests/test_api.py tests/test_io.py tests/test_cache.py tests/test_io_engine_caches.py tests/test_header_cards.py tests/test_cold_import.py -q`
  - Result: `99 passed, 2 warnings`.
- North Star connection: this keeps read-cache policy centralized, preserving
  correctness for stale files and cached CPU/GPU tensor transfer while reducing
  root dispatch state.
- Moved batch-path and multi-HDU read dispatch helpers into
  `torchsky._io_engine.read_dispatch`, keeping root wrappers and recursive
  `read(...)` fallback behavior unchanged.
- Tests run:
  - `pixi run -e test -- pytest tests/test_io.py::TestIORead::test_read_batch_paths tests/test_io.py::TestIORead::test_read_batch_hdus tests/test_api.py -k 'read_hdus or read_image or read_hdu_auto' tests/test_cold_import.py -q`
  - Result: `5 passed, 30 deselected, 2 warnings`.
  - `pixi run -e test -- pytest tests/test_api.py tests/test_io.py tests/test_cache.py tests/test_io_engine_caches.py tests/test_header_cards.py tests/test_cold_import.py -q`
  - Result: `99 passed, 2 warnings`.
- North Star connection: this keeps bulk image ingestion policy in the private
  read engine while preserving batch C++ fast paths and recursive torch-native
  fallback behavior.

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
