# Autonomous Refactor Log

> Archived audit trail. This file is no longer the active roadmap or work
> queue. Use `../roadmap/status.md` for the current status dashboard and
> `../roadmap/history.md` for the maintained low-level record of
> shipped improvements, tests, and follow-ups.

Target branch: `codex/autonomous-refactor-2026-05`

Start gate: May 8, 2026, `America/Vancouver`

## Current State

- Resumed manually on June 8, 2026 after the scheduled automation did not run.
- Active branch: `codex/autonomous-refactor-2026-05`.
- Completed A0, A1, A2, B1-B4, C1-C2, D1, and D2 from
  `docs/reviews/autonomous_refactor_launch.md`.
- Backward compatibility for unused alias namespaces is no longer a goal; the
  public API should stay clean and route users to the owning modules.
- Active slice: G1, continuing roadmap implementation after completing the
  field-model preset surface.

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
- Added `docs/architecture.md` and linked it from `README.md`, `docs/index.md`,
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
- Moved fallback image and table readers into `torchsky._io_engine.read_dispatch`,
  keeping root wrappers for the remaining fallback orchestrator.
- Tests run:
  - `pixi run -e test -- pytest tests/test_io.py tests/test_table.py tests/test_cache.py -k 'read_basic or use_cache or no_cache or read_table or large_table or stream_table or return_header or cache_functions' tests/test_api.py -k 'read_table or read_image or read_hdu_auto or get_header' tests/test_cold_import.py -q`
  - Result: `14 passed, 101 deselected, 2 warnings`.
  - `pixi run -e test -- pytest tests/test_api.py tests/test_io.py tests/test_cache.py tests/test_io_engine_caches.py tests/test_header_cards.py tests/test_table.py tests/test_cold_import.py -q`
  - Result: `122 passed, 2 warnings`.
- North Star connection: this moves image/table fallback I/O policy into the
  private read engine while preserving return-header behavior, table slicing,
  cache writes, and device movement.
- Moved the generic fallback read orchestrator into
  `torchsky._io_engine.read_dispatch`, keeping the root private wrapper used by
  the public `read(...)` dispatcher.
- Tests run:
  - `pixi run -e test -- pytest tests/test_io.py tests/test_table.py tests/test_cache.py -k 'read_basic or use_cache or no_cache or read_table or return_header or cache_functions' tests/test_api.py -k 'read_table or read_image or read_hdu_auto or read_by_extname' tests/test_cold_import.py -q`
  - Result: `14 passed, 101 deselected, 2 warnings`.
  - `pixi run -e test -- pytest tests/test_api.py tests/test_io.py tests/test_cache.py tests/test_io_engine_caches.py tests/test_header_cards.py tests/test_table.py tests/test_cold_import.py -q`
  - Result: `122 passed, 2 warnings`.
- North Star connection: this puts named-HDU resolution, image/table fallback
  selection, and handle closing in the private read engine while leaving the
  public `read(...)` contract unchanged.
- Moved the remaining unified `read(...)` orchestration into
  `torchsky._io_engine.read_dispatch.read_unified`, leaving the root public
  function as an overload-preserving wrapper.
- Tests run:
  - `pixi run -e test -- pytest tests/test_io.py tests/test_api.py tests/test_cache.py tests/test_table.py tests/test_io_engine_caches.py tests/test_cold_import.py -q`
  - Result: `116 passed, 2 warnings`.
  - `pixi run preflight-push`
  - Result: `All checks passed`.
- North Star connection: this makes the root package a thinner public facade
  while keeping the canonical `torchsky.read(...)` contract, option semantics,
  batch dispatch, cache behavior, and device policy stable.
- Updated `torchsky.io.read` and `torchsky.io.read_table` to call the private
  read engine directly instead of routing backward through the package root,
  while preserving the existing `torchsky.cpp` monkeypatch seam used by tests.
- Tests run:
  - `pixi run -e test -- pytest tests/test_io.py::TestIORead -q`
  - Result: `20 passed, 2 warnings`.
  - `pixi run -e test -- pytest tests/test_api.py tests/test_io.py tests/test_cache.py tests/test_table.py tests/test_io_engine_caches.py tests/test_header_cards.py tests/test_cold_import.py -q`
  - Result: `122 passed, 2 warnings`.
  - `pixi run preflight-push`
  - Result: `All checks passed`.
- North Star connection: this advances the canonical `torchsky.io` namespace
  toward implementation ownership, while preserving root compatibility and test
  seams for existing users.
- Updated `torchsky.io.get_header`, `torchsky.io.get_wcs`, `torchsky.io.read_image`
  fallback headers, `torchsky.io.stream_table`, and
  `torchsky.io.read_large_table` to use local direct header/HDU helpers instead
  of importing root `get_header`.
- Tests run:
  - `pixi run -e test -- pytest tests/test_api.py -k 'get_header or get_wcs or top_level_namespaces or read_image or large_table or stream_table' tests/test_header_cards.py tests/test_io.py tests/test_table.py -k 'large_table or stream_table or read_table' tests/test_cold_import.py -q`
  - Result: `10 passed, 72 deselected, 2 warnings`.
  - `pixi run -e test -- pytest tests/test_api.py tests/test_io.py tests/test_cache.py tests/test_table.py tests/test_io_engine_caches.py tests/test_header_cards.py tests/test_cold_import.py -q`
  - Result: `122 passed, 2 warnings`.
  - `pixi run preflight-push`
  - Result: `All checks passed`.
- North Star connection: this removes another reverse dependency from the
  canonical I/O namespace while preserving header/WCS and streaming behavior.
- Updated `torchsky.io.read_subset` to call `torchsky._io_engine.subset`
  directly instead of importing root `read_subset`.
- Tests run:
  - `pixi run -e test -- pytest tests/test_api.py -k 'subset or top_level_namespaces' tests/test_cutout_performance_api.py tests/test_io.py tests/test_cold_import.py -q`
  - Result: `3 passed, 51 deselected, 2 warnings`.
  - `pixi run -e test -- pytest tests/test_api.py tests/test_io.py tests/test_cache.py tests/test_table.py tests/test_io_engine_caches.py tests/test_header_cards.py tests/test_cutout_performance_api.py tests/test_cold_import.py -q`
  - Result: `123 passed, 2 warnings`.
  - `pixi run preflight-push`
  - Result: `All checks passed`.
- North Star connection: this keeps cutout/subset reads in the private I/O
  engine and further reduces reverse routing through the package root.
- Moved the FITS write implementation, table normalization helpers, compressed
  rewrite helpers, and HDU insert/replace/delete implementations into
  `torchsky._io_engine.write_api`, leaving root functions as compatibility
  wrappers and preserving root private helper aliases used by table/HDU code.
  Updated `torchsky.io.write` to call the write engine directly instead of
  importing `torchsky.write`.
- Added a regression test that monkeypatches `torchsky.write` and verifies
  `torchsky.io.write` still writes and reads back a valid image, locking in the
  direct canonical-namespace route.
- Tests run:
  - `pixi run preflight-push`
  - Result: `All checks passed`.
  - `pixi run -e test -- pytest tests/test_writing.py tests/test_hdu_file_ops.py tests/test_table_file_ops.py tests/test_header_cards.py tests/test_checksum.py -q`
  - Result: `52 passed, 1 skipped, 2 warnings`.
  - `pixi run -e test -- pytest tests/test_api.py tests/test_io.py tests/test_cache.py tests/test_table.py tests/test_io_engine_caches.py tests/test_header_cards.py tests/test_writing.py tests/test_hdu_file_ops.py tests/test_table_file_ops.py tests/test_cold_import.py -q`
  - Result: `161 passed, 1 skipped, 2 warnings`.
  - `pixi run ci-local`
  - Result: `1589 passed, 61 skipped, 1 xfailed, 74 warnings`.
- North Star connection: this removes the largest remaining root-owned FITS I/O
  implementation block while keeping the lean public API stable, preserving
  cache invalidation and compressed-write behavior, and making `torchsky.io`
  the canonical I/O implementation facade.
- Moved root checksum helper implementations into
  `torchsky._io_engine.checksum_api`, keeping `torchsky.write_checksums(...)`
  and `torchsky.verify_checksums(...)` as root compatibility wrappers backed by
  `ChecksumVerifier`.
- Tests run:
  - `pixi run -e test -- pytest tests/test_checksum.py tests/test_core.py::test_checksum_verifier_in_memory_methods_fail_closed tests/test_fitsio_upstream_smoke.py::test_fitsio_readme_table_mutation_header_and_checksum_workflows -q`
  - Result: `9 passed, 2 warnings`.
  - `pixi run preflight-push`
  - Result: `All checks passed`.
- North Star connection: this keeps checksum behavior on the established
  CFITSIO-backed verifier while removing another implementation detail from the
  root facade.
- Moved the combined Python/C++ FITS cache clearing routine into
  `torchsky._io_engine.caches.clear_file_cache`, keeping
  `torchsky.clear_file_cache(...)` as a root compatibility wrapper.
- Tests run:
  - `pixi run -e test -- pytest tests/test_api.py::TestMainAPI::test_cache_functions tests/test_cache.py tests/test_io_engine_caches.py -q`
  - Result: `41 passed, 2 warnings`.
  - `pixi run preflight-push`
  - Result: `All checks passed`.
- North Star connection: this keeps cache policy centralized in the private I/O
  cache facade, reducing root orchestration while preserving stale-handle and
  C++ shared-metadata invalidation semantics.
- Removed two unused root-only table-header helper functions
  (`_get_vla_columns`, `_get_scaled_columns`) after verifying they had no
  remaining references.
- Tests run:
  - `pixi run preflight-push`
  - Result: `All checks passed`.
  - `pixi run -e test -- pytest tests/test_api.py -k 'namespace or top_level or api_surface' tests/test_cold_import.py -q`
  - Result: `1 passed, 32 deselected, 2 warnings`.
  - `pixi run -e test -- pytest tests/test_cold_import.py -q`
  - Result: `1 passed`.
- North Star connection: this keeps the root package focused as a public facade
  and removes obsolete private FITS table parsing code that no longer belongs
  in the root namespace.
- Removed two unused root private read helper wrappers
  (`_get_image_meta_from_handle`, `_read_scaled_cpu_fast`) and their unused
  implementation imports after verifying they had no references outside their
  own definitions.
- Tests run:
  - `pixi run preflight-push`
  - Result: `All checks passed`.
  - `pixi run -e test -- pytest tests/test_api.py -k 'read_image or read_table or read_hdus or read_mode or read_policy or namespace' tests/test_io.py::TestIORead tests/test_cold_import.py -q`
  - Result: `17 passed, 36 deselected, 2 warnings`.
- North Star connection: this keeps root read APIs as intentional public
  wrappers while removing obsolete private hooks from the root namespace.
- Expanded `torchsky.io` into the complete canonical FITS I/O namespace by
  adding direct private-engine routes for `open_subset_reader`, HDU
  insert/replace/delete, checksum helpers, batch helpers, and cache helpers.
  Root remains a compatibility facade for existing users.
- Added a routing regression that monkeypatches the root helpers and verifies
  the `torchsky.io` helpers still mutate HDUs, write/verify checksums, report
  batch/cache status, and clear caches without delegating back to root.
- Updated `docs/io.md` to state the completed B-stage API split: root keeps
  compatibility names, while `torchsky.io` is the canonical namespace-oriented
  FITS surface.
- Tests run:
  - `pixi run preflight-push`
  - Result: `All checks passed`.
  - `pixi run -e test -- pytest tests/test_io.py tests/test_writing.py tests/test_hdu_file_ops.py tests/test_checksum.py tests/test_cache.py tests/test_api.py -k 'io or read_batch or cache_functions or namespace or checksum or read_hdus or read_image or read_table' tests/test_cold_import.py -q`
  - Result: `61 passed, 62 deselected, 2 warnings`.
  - `pixi run ci-local`
  - Result: `1590 passed, 61 skipped, 1 xfailed, 88 warnings`.
- North Star connection: this finishes the B-stage architecture direction by
  making the root package a stable public facade and `torchsky.io` the complete
  direct implementation namespace for FITS I/O, without adding runtime
  dependencies or changing tensor/device semantics.
- Added the promoted `torchsky.scene` facade as an additive namespace over the
  existing map/field implementation:
  - `scene.FieldObservation` / `scene.ImageObservation`
  - `scene.TargetRenderSpec`
  - `scene.MapProductSpec` / `scene.MapBandpassSpec`
  - `scene.SceneRepresentation` and existing scene adapters
  - factory helpers such as `target_from_header`, `target_from_product`,
    `product_spec`, `render`, and `predict`.
- Added `torchsky.scene.operators` with `ObservationOperatorAdapter`,
  `ObservationOperatorBundleProtocol`, `operator_adapter_for_observation(...)`,
  and `operator_bundle_for_observation(...)`, preserving the current
  `ObservationOperatorBundle` implementation while exposing an executable
  scene-level adapter.
- Exposed `torchsky.scene` through the root lazy namespace and API surface
  metadata, and added `docs/scene.md`.
- Tests run:
  - `pixi run preflight-push`
  - Result: `All checks passed`.
  - `pixi run -e test -- pytest tests/test_scene_facade.py tests/test_map_target_render_spec.py tests/test_cold_import.py -q`
  - Result: `7 passed, 2 warnings`.
  - `pixi run typecheck`
  - Result: `0 errors, 0 warnings, 0 informations`.
  - `pixi run ci-local`
  - Result: `1593 passed, 61 skipped, 1 xfailed, 102 warnings`.
- North Star connection: this starts the scene pillar without destabilizing
  existing map code: users get clear observation/product/scene vocabulary, while
  parity tests ensure the promoted operator adapter still uses the same
  torch-native map/field operator bundle behavior.
- Moved the native HEALPix pixel implementation from `torchsky.wcs.healpix` to
  `torchsky.sphere.healpix`, making sphere the owning namespace for indexing,
  neighbors, interpolation, ordering conversion, range compression, and query
  primitives.
- Kept `torchsky.wcs.healpix` as a small re-export shim for now because the D2
  backward-compatibility policy is intentionally undecided.
- Updated internal imports in catalogs, coords, encoders, map, sphere, and viz
  code to depend on `torchsky.sphere.healpix` directly, and documented the new
  ownership in `docs/sphere.md`, `docs/coords.md`, and
  `docs/compatibility.md`.
- Added a regression test proving the WCS shim re-exports the sphere-owned
  functions without creating a second implementation.
- Tests run:
  - `pixi run -e test -- pytest tests/test_healpix_primitives.py tests/test_healpix_parity.py tests/test_healpix_spectral.py tests/test_sphere_core.py tests/test_sphere_sparse.py tests/test_sphere_spectral.py tests/test_sphere_compat.py tests/test_sphere_ecosystem.py tests/test_viz_renderers.py tests/test_viz_server.py tests/test_map_sphere_sh_map_smoke.py tests/test_perf_gate.py -q`
  - Result: `266 passed, 9 skipped, 10 warnings`.
  - `pixi run preflight-push`
  - Result: `All checks passed`.
  - `pixi run typecheck`
  - Result: `0 errors, 0 warnings, 0 informations`.
  - `pixi run ci-local`
  - Result: `1594 passed, 61 skipped, 1 xfailed, 102 warnings`.
- North Star connection: this puts spherical sky pixel primitives under the
  sphere pillar where survey-scale map, sparse, MOC, and compatibility layers
  already live, while preserving the torch-native kernels and avoiding a
  premature decision about long-term alias compatibility.
- Removed leftover alias surfaces now that backward compatibility is explicitly
  out of scope for unused APIs:
  - deleted `torchsky.wcs.healpix`; HEALPix is only under
    `torchsky.sphere.healpix`;
  - deleted `torchsky.coords`; WCS, frame, spectral, temporal, alignment, and
    TAB helpers are under `torchsky.wcs`;
  - deleted `torchsky.geom`; spherical geometry is under `torchsky.sphere` and
    `torchsky.sphere.geom`.
- Updated docs and tests so the clean public API points to the owning
  namespaces instead of compatibility aliases.
- Tests run:
  - `pixi run -e test -- pytest tests/test_api.py::test_top_level_namespaces_expose_phase1_modules tests/test_wcs_tabular.py tests/test_healpix_primitives.py tests/test_docs_integrity.py tests/test_cold_import.py -q`
  - Result: `35 passed, 4 skipped, 2 warnings`.
  - `pixi run -e test -- pytest tests/test_catalog_compat.py tests/test_catalogs.py tests/test_map_compat_bench.py tests/test_map_ground_truth.py tests/test_map_multifield_truth.py tests/test_map_image_plane.py::test_coadd_image_plane_clamps_negative_resampled_weights_for_high_order_kernels -q`
  - Result: `86 passed, 1 skipped, 2 warnings`.
  - `pixi run -e test -- pytest tests/test_sphere_polygons.py::test_upstream_polygon_replay_gate_small -q`
  - Result: `1 passed, 2 warnings`.
  - `pixi run preflight-push`
  - Result: `All checks passed`.
  - `pixi run typecheck`
  - Result: `0 errors, 0 warnings, 0 informations`.
  - `pixi run ci-local`
  - Result: `1593 passed, 61 skipped, 1 xfailed, 88 warnings`.
- North Star connection: this reduces jargon-heavy duplicate import paths and
  makes the world pillar easier to learn: WCS owns coordinate transforms and
  sphere owns sky geometry, pixelization, MOC, sparse maps, and harmonics.
- Added read-only CLI inspection workflows:
  - `torchsky info PATH` prints HDU summaries without materializing data;
  - `torchsky info PATH --json` emits automation-friendly JSON, with optional
    `--headers`;
  - `torchsky cat PATH --hdu ... --columns ... --rows ...` prints a bounded
    table sample;
  - `torchsky cat ... --json` emits row dictionaries with JSON-safe tensor
    scalars/lists.
- Added focused CLI tests for JSON and text output on image/table FITS fixtures,
  and updated `docs/cli.md` / `docs/api.md`.
- Tests run:
  - `pixi run -e test -- pytest tests/test_cli_inspect.py tests/test_cli_map.py tests/test_api.py::test_map_cli_exposes_resampling_kernel_option tests/test_docs_integrity.py -q`
  - Result: `12 passed, 2 warnings`.
  - `pixi run preflight-push`
  - Result: `All checks passed`.
  - `pixi run typecheck`
  - Result: `0 errors, 0 warnings, 0 informations`.
- North Star connection: this gives agents and users a low-jargon, read-only
  shell surface for inspecting FITS files and table samples before building
  heavier map or modelling workflows.

### 2026-06-10

- Added `FieldModelPreset`, `FIELD_MODEL_PRESETS`,
  `FIELD_MODEL_PRESET_NAMES`, `get_field_model_preset(...)`, and
  `make_field_model_preset(...)` under `torchsky.map.field`, re-exported
  through `torchsky.map` for the main modelling workflow.
- Defined four low-jargon presets over the current expert model zoo:
  `fast`, `balanced`, `spectral`, and `operator_constrained`.
- Updated `docs/field.md` to recommend presets first while keeping
  `make_field_model(kind, ...)` documented for expert selection.
- Added focused tests for preset names, immutable registry exposure, name
  normalization, override forwarding, and equivalence with the existing factory.
- Tests run:
  - `pixi run -e test -- pytest tests/test_field_model_presets.py tests/test_cli_inspect.py tests/test_cli_map.py tests/test_docs_integrity.py -q`
  - Result: `15 passed, 2 warnings`.
- North Star connection: this lowers the jargon barrier for continuous-field
  modelling while preserving the torch-native expert model families needed for
  ambitious multi-wavelength survey-scale experiments.

- Promoted constrained learned instrument-operator components through
  `torchsky.map.operators`: `ConstrainedKernelPSF`, `AffineRadiometry`,
  `SubPixelShift`, `GatedResidualOperator`, `InstrumentStateEncoder`, and
  `LearnedInstrumentOperator`.
- Kept the operator family constrained by construction: positive unit-sum PSF
  basis kernels, affine radiometry initialized from caller-provided gains,
  zero-gated nonlinear residuals, and staged training controls.
- Added focused tests for flux conservation, conditioning, identity-at-init
  residuals, radiometry initialization, differentiable sub-pixel shifts,
  masked instrument-state pooling, and curriculum gradients.
- Tests run:
  - `pixi run -e test -- pytest tests/test_learned_kernel_operators.py tests/test_map_learned_ops.py tests/test_map_operator_bundle.py tests/test_docs_integrity.py -q`
  - Result: `23 passed, 2 warnings`.
- North Star connection: this gives future field-fitting work a torch-native
  learned-instrument path that can absorb real survey nuisance structure without
  collapsing the scene model into unconstrained black-box corrections.

- Started G1 map-architecture work by moving source-to-target window bbox
  planning out of `map/image_plane.py` into
  `torchsky.map.operators.execution.source_bbox_on_target(...)`, then exposing
  it through `torchsky.map.operators`.
- Kept private target slicing local to `image_plane.py` because it depends on
  the private prepared-target dataclass, but routed the geometry decision
  through the operator execution module.
- Fixed the large-TAN no-overlap branch so it returns an empty projection
  instead of falling through into geometry projection with a `None` target view.
- Added focused tests for clipped source-boundary bbox planning and no-overlap
  window handling.
- Tests run:
  - `pixi run -e test -- pytest tests/test_map_operator_execution.py tests/test_map_operator_bundle.py tests/test_map_image_plane.py::test_project_observation_large_tan_windowing_matches_full_reprojection tests/test_map_image_plane.py::test_project_observation_window_no_overlap_skips_geometry -q`
  - Result: `20 passed, 2 warnings`.
  - `pixi run preflight-push`
  - Result: `All checks passed`.
  - `pixi run typecheck`
  - Result: `0 errors, 0 warnings, 0 informations`.
- North Star connection: this continues moving map projection policy toward the
  shared operator execution substrate, while preserving the large-map windowing
  path needed for survey-scale coadds.

- Added `scripts/audit_refactor_cohesion.py` plus the
  `pixi run audit-refactor-cohesion` task as a cheap, default-env wiring audit
  for the refactor/benchmark surface.
- The audit checks that key map/WCS/sphere benchmark gates remain wired in both
  `pixi.toml` and `pyproject.toml`, that benchmark baselines and docs exist,
  that map/field/architecture docs mention the promoted surfaces, and that
  source exports expose the current public map/operator/scene spine.
- Promoted `ObservationOperatorBundle` through `torchsky.map` because the audit
  caught that docs described it as shared map substrate while only
  `torchsky.map.operators` exported it.
- The audit currently passes with two intentional warnings:
  `src/torchsky/map/image_plane.py` and `src/torchsky/map/field/core.py` remain
  above their cohesion pressure thresholds and should keep guiding future G1
  extraction slices.
- Tests run:
  - `pixi run audit-refactor-cohesion`
  - Result: `0 failures, 2 warnings`.
  - `pixi run -e test -- pytest tests/test_refactor_cohesion_audit.py tests/test_map_operator_bundle.py::test_observation_operator_bundle_iter_enabled_skips_disabled_entries tests/test_docs_integrity.py -q`
  - Result: `4 passed, 2 warnings`.
  - `pixi run typecheck`
  - Result: `0 errors, 0 warnings, 0 informations`.
  - `pixi run preflight-push`
  - Result: `All checks passed`.
- North Star connection: this gives the refactor a recurring, automated audit
  surface that ties API cohesion to benchmark evidence before larger
  architecture moves.

- Moved plain-TAN and no-distortion fast-policy predicates into
  `torchsky.map.operators.execution` as `is_plain_tan_header(...)` and
  `wcs_has_distortion(...)`, then exported them through `torchsky.map.operators`.
- Updated `image_plane.py` to use those shared predicates in both
  surface-brightness area-scaling policy and large-TAN target windowing.
- Extended the cohesion audit so those reusable fast-policy predicates are part
  of the checked operator execution surface.
- Tests run:
  - `pixi run -e test -- pytest tests/test_map_operator_execution.py tests/test_map_image_plane.py::test_project_observation_large_tan_windowing_matches_full_reprojection tests/test_map_image_plane.py::test_surface_brightness_area_scaling_gate_uses_target_vs_harmonic_source_scale tests/test_refactor_cohesion_audit.py -q`
  - Result: `13 passed, 2 warnings`.
  - `pixi run audit-refactor-cohesion`
  - Result: `0 failures, 2 warnings`; `image_plane.py` is now `3212` lines,
    still above the `3000`-line cohesion warning threshold.
  - `pixi run preflight-push`
  - Result: `All checks passed`.
  - `pixi run typecheck`
  - Result: `0 errors, 0 warnings, 0 informations`.
- North Star connection: this keeps fast mapmaking policy reusable and tested
  outside the monolithic image-plane module, so future benchmark-driven
  windowing and scaling changes have one shared decision point.

- Extracted pixel-area math from `image_plane.py` into
  `torchsky.map.operators.pixel_area`, exported as `pixel_area_sr_map(...)`,
  `representative_pixel_area_sr(...)`, and `wrapped_delta_ra_deg(...)`.
- Updated flux-semantics coadds, surface-brightness information-density scaling,
  target preparation, and product flux metrics to use the shared operator
  helpers instead of private image-plane functions.
- Updated tests that previously reached into private image-plane helpers to use
  `torchsky.map.operators` exports.
- Tests run:
  - `pixi run -e test -- pytest tests/test_map_operator_execution.py tests/test_map_image_plane.py::test_coadd_image_plane_flux_semantics_improves_flux_conservation tests/test_map_image_plane.py::test_coadd_image_plane_weighted_mean_scales_surface_brightness_weights_only_for_coarser_targets tests/test_map_product.py tests/test_refactor_cohesion_audit.py -q`
  - Result: `28 passed, 2 warnings`.
  - `pixi run audit-refactor-cohesion`
  - Result: `0 failures, 2 warnings`; `image_plane.py` is now `3162` lines,
    still above the `3000`-line cohesion warning threshold.
  - `pixi run preflight-push`
  - Result: `All checks passed`.
  - `pixi run typecheck`
  - Result: `0 errors, 0 warnings, 0 informations`.
- North Star connection: this centralizes physically meaningful solid-angle
  estimation for projection, flux accounting, and benchmark metrics, reducing
  hidden policy in the image-plane orchestrator.

- Added `torchsky.map.operators.config` for shared map resampling kernels,
  value semantics, stacking methods, and validators.
- Updated `image_plane.py` to delegate its private compatibility validators to
  the shared config module, and updated `product.py` so map-product planning no
  longer imports private image-plane validators.
- Exposed the option tuples and validators through `torchsky.map.operators`, and
  added focused tests for supported values and error reporting.
- Tests run:
  - `pixi run -e test -- pytest tests/test_map_operator_execution.py tests/test_map_image_plane.py::test_coadd_image_plane_rejects_unknown_value_semantics tests/test_map_image_plane.py::test_coadd_image_plane_rejects_unknown_stacking_method tests/test_map_product.py tests/test_refactor_cohesion_audit.py -q`
  - Result: `29 passed, 2 warnings`.
  - `pixi run audit-refactor-cohesion`
  - Result: `0 failures, 2 warnings`; `image_plane.py` is now `3116` lines,
    still above the `3000`-line cohesion warning threshold.
  - `pixi run preflight-push`
  - Result: `All checks passed`.
  - `pixi run typecheck`
  - Result: `0 errors, 0 warnings, 0 informations`.
- North Star connection: this gives user-facing map configuration a single
  reusable validation surface and removes another private dependency from the
  product-planning layer.

- Added `torchsky.map.operators.image_grid` for reusable image-plane
  validation and FITS-like image-grid header helpers:
  `validate_image_plane_2d(...)`, `target_shape_from_header(...)`,
  `normalized_header_for_shape(...)`, and
  `headers_equivalent_for_direct_copy(...)`.
- Updated `image_plane.py`, mask/weight loading, diagnostics, and benchmark
  scripts to use the shared operator helper surface instead of private
  image-plane functions or local duplicate predicates.
- Extended the cohesion audit and operator tests so image-grid helpers are now
  part of the checked operator API.
- Tests run:
  - `pixi run -e test -- pytest tests/test_map_operator_execution.py tests/test_map_image_plane.py::test_project_observation_large_tan_windowing_matches_full_reprojection tests/test_map_operator_bundle.py tests/test_refactor_cohesion_audit.py -q`
  - Result: `25 passed, 2 warnings`.
  - `pixi run audit-refactor-cohesion`
  - Result: `0 failures, 2 warnings`; `image_plane.py` is now `3055` lines,
    still above the `3000`-line cohesion warning threshold.
  - `pixi run preflight-push`
  - Result: `All checks passed`.
  - `pixi run typecheck`
  - Result: `0 errors, 0 warnings, 0 informations`.
  - `pixi run -e test -- pytest tests/test_docs_integrity.py -q`
  - Result: `1 passed`.
- North Star connection: image-grid shape/header semantics now live in a
  reusable operator module, which reduces monolithic map orchestration and
  keeps diagnostics, benchmarks, and production code on one exact policy.

- Added `torchsky.map.operators.resampling` for reusable image-plane sampling,
  custom separable kernels, affine WCS pixel-map inference, and periodic
  resampling finalization helpers.
- Updated `image_plane.py` to import resampling mechanics from the operator
  module while keeping target preparation, distortion policy, quadrature, and
  coadd orchestration local to the image-plane mapmaker.
- Updated tests, diagnostics, and boundary-support benchmarks to instrument the
  new resampling module directly instead of reaching into private image-plane
  sampling helpers.
- Extended the cohesion audit to check the new resampling helper exports.
- Tests run:
  - `pixi run -e test -- pytest tests/test_map_image_plane.py::test_sample_image_plane_batched_channels_matches_separate_sampling tests/test_map_image_plane.py::test_coadd_image_plane_passes_resampling_kernel_to_grid_sample tests/test_map_image_plane.py::test_project_observation_large_tan_windowing_matches_full_reprojection tests/test_map_operator_execution.py tests/test_refactor_cohesion_audit.py -q`
  - Result: `20 passed, 2 warnings`.
  - `pixi run audit-refactor-cohesion`
  - Result: `0 failures, 1 warnings`; `image_plane.py` is now `2610` lines,
    below the `3000`-line cohesion warning threshold. The remaining warning is
    `field/core.py`.
  - `pixi run preflight-push`
  - Result: `All checks passed`.
  - `pixi run typecheck`
  - Result: `0 errors, 0 warnings, 0 informations`.
  - `pixi run -e test -- pytest tests/test_docs_integrity.py -q`
  - Result: `1 passed`.
- North Star connection: interpolation and FFT/affine resampling are now
  reusable, testable operator mechanics rather than private mapmaker internals,
  which keeps future performance work closer to a single benchmarkable surface.

- Added the concise `torchsky.field` namespace for composable continuous-field
  workflows:
  - `field.model(...)` and `field.preset(...)` for construction;
  - `field.fit(...)` for the built-in optimizer loop;
  - `field.render(...)`, `field.sample(...)`, and `field.distribution(...)` for
    explicit evaluation.
- Updated API-surface metadata and docs so new user-facing examples recommend
  `torchsky.field` instead of centering the API on long `make_field_*` /
  `render_field_*` names.
- Extracted field bandpass helpers into `torchsky.map.field.bandpass` and
  observation geometry/cropping/pixel-sampling helpers into
  `torchsky.map.field.observation_ops`.
- Extended the cohesion audit to check the new field namespace and extracted
  helper modules.
- Tests run:
  - `pixi run -e test -- pytest tests/test_api.py::test_field_namespace_uses_short_composable_verbs tests/test_map_field.py::test_bandpass_quadrature_trapezoid_closer_than_uniform_for_quadratic tests/test_map_field.py::test_bandpass_quadrature_tabulated_triangle_weighted_mean tests/test_map_field_render_scale.py tests/test_implicit_forward.py::test_fit_field_model_threads_observation_forward_contexts tests/test_refactor_cohesion_audit.py -q`
  - Result: `13 passed, 2 warnings`.
  - `pixi run -e test -- pytest tests/test_docs_integrity.py tests/test_api.py::test_field_namespace_uses_short_composable_verbs tests/test_api_surfaces.py tests/test_map_field.py::test_bandpass_quadrature_tabulated_triangle_weighted_mean -q`
  - Result: `8 passed, 2 warnings`.
  - `pixi run audit-refactor-cohesion`
  - Result: `0 failures, 1 warnings`; `field/core.py` is now `7310` lines,
    still above the `6000`-line cohesion warning threshold.
  - `pixi run preflight-push`
  - Result: `All checks passed`.
  - `pixi run typecheck`
  - Result: `0 errors, 0 warnings, 0 informations`.
- North Star connection: field workflows now have short, composable verbs and
  clearer primitive names, while reusable observation/bandpass mechanics are no
  longer hidden inside the model-zoo core module.

- Reworked `audit-refactor-cohesion` to stop using line-count thresholds.
  Cohesion is now checked through structural boundaries:
  - required field/map helper modules and concise field namespace exports must
    exist;
  - helpers already extracted from `image_plane.py` and `field/core.py` must not
    move back into those orchestration/model modules.
- Extracted field forward-dispatch helpers into `torchsky.map.field.forwarding`
  so context-aware calls to `forward`, `render_bandpass`, and distribution
  methods are reusable plumbing rather than embedded core-model code.
- Tests run:
  - `pixi run -e test -- pytest tests/test_refactor_cohesion_audit.py tests/test_api.py::test_field_namespace_uses_short_composable_verbs tests/test_implicit_forward.py::test_render_field_observation_passes_forward_context_when_supported tests/test_implicit_forward.py::test_fit_field_model_threads_observation_forward_contexts -q`
  - Result: `5 passed, 2 warnings`.
  - `pixi run audit-refactor-cohesion`
  - Result: `0 failures, 0 warnings`.
  - `pixi run preflight-push`
  - Result: `All checks passed`.
  - `pixi run typecheck`
  - Result: `0 errors, 0 warnings, 0 informations`.
  - `pixi run -e test -- pytest tests/test_docs_integrity.py -q`
  - Result: `1 passed`.
- North Star connection: the audit now encourages clean architecture and API
  boundaries directly instead of optimizing for arbitrary file sizes.

- Extracted hash-grid and feature-plane encoding mechanics into
  `torchsky.map.field.hashgrid`.
- Kept the public user-facing field API lean: the new module is reusable by
  field implementations and tests, while high-level users still compose through
  `torchsky.field.model`, `torchsky.field.fit`, `torchsky.field.render`, and
  `torchsky.field.sample`.
- Extended `audit-refactor-cohesion` so the new hash-grid module is required and
  the old private hash-grid helpers cannot drift back into
  `torchsky.map.field.core`.
- Tests run:
  - `pixi run -e test -- pytest tests/test_map_field.py::test_hashgrid_encoder_rejects_mismatched_log2_sequence tests/test_map_field.py::test_hashgrid_encoder_coarse_dense_fine_hash tests/test_refactor_cohesion_audit.py -q`
  - Result: `4 passed, 2 warnings`.
  - `pixi run audit-refactor-cohesion`
  - Result: `0 failures, 0 warnings`.
  - `pixi run preflight-push`
  - Result: `All checks passed`.
  - `pixi run typecheck`
  - Result: `0 errors, 0 warnings, 0 informations`.
  - `pixi run -e test -- pytest tests/test_docs_integrity.py -q`
  - Result: `1 passed`.
- North Star connection: hash-grid encoding is a performance-sensitive field
  primitive, so it now has a direct module boundary for future benchmarking and
  reuse instead of living inside the overloaded model core.

- Extracted generic field network building blocks into
  `torchsky.map.field.networks`:
  - residual MLP block/stack;
  - shared SiLU MLP construction;
  - shared log-flux readout initialization.
- Extracted sky-coordinate encoder construction helpers into
  `torchsky.map.field.sky_encoding`.
- Kept these as implementation submodules under `torchsky.map.field`; the
  high-level `torchsky.field` surface remains focused on the short composable
  verbs.
- Extended `audit-refactor-cohesion` so those helper boundaries are required and
  the old private helper definitions cannot return to `field/core.py`.
- Tests run:
  - `pixi run -e test -- pytest tests/test_map_field.py::test_sky_encoder_mlp_field_supports_charbonnier_fit tests/test_map_field.py::test_sky_encoder_resmlp_field_supports_charbonnier_fit tests/test_map_field.py::test_hashgrid_resmlp_field_supports_charbonnier_fit tests/test_refactor_cohesion_audit.py -q`
  - Result: `5 passed, 2 warnings`.
  - `pixi run audit-refactor-cohesion`
  - Result: `0 failures, 0 warnings`.
  - `pixi run preflight-push`
  - Result: `All checks passed`.
  - `pixi run typecheck`
  - Result: `0 errors, 0 warnings, 0 informations`.
  - `pixi run -e test -- pytest tests/test_docs_integrity.py -q`
  - Result: `1 passed`.
- North Star connection: field model classes now read more like model
  composition, while reusable network and encoding primitives have direct module
  homes for future reuse and benchmarking.

- Extracted field rendering and coordinate evaluation into
  `torchsky.map.field.rendering`.
- Redirected public rendering exports through the new module while preserving the
  existing `torchsky.map` / `torchsky.map.field` names.
- Removed a stale `observation_builders -> core` helper dependency by importing
  header-shape logic from `torchsky.map.field.observation_ops`.
- Extended `audit-refactor-cohesion` so rendering helpers cannot drift back into
  `field/core.py`.
- Tests run:
  - `pixi run -e test -- pytest tests/test_map_field_render_scale.py tests/test_implicit_forward.py::test_render_field_observation_passes_forward_context_when_supported tests/test_map_background_ops.py tests/test_refactor_cohesion_audit.py -q`
  - Result: `17 passed, 2 warnings`.
  - `pixi run audit-refactor-cohesion`
  - Result: `0 failures, 0 warnings`.
  - `pixi run preflight-push`
  - Result: `All checks passed`.
  - `pixi run typecheck`
  - Result: `0 errors, 0 warnings, 0 informations`.
  - `pixi run -e test -- pytest tests/test_docs_integrity.py -q`
  - Result: `1 passed`.
- North Star connection: rendering is now a reusable execution surface for
  observation, cube, HEALPix, tiled, and vmap evaluation instead of being mixed
  into model definitions.

- Extracted field fitting and noise-pretraining loops into
  `torchsky.map.field.training`.
- Extracted shared residual-loss logic into `torchsky.map.field.losses` and
  synthetic-noise targets into `torchsky.map.field.noise`, so `core.py` and
  training code share the helpers without a circular dependency.
- Redirected public `fit_field_model`, `fit_field_component_decomposition`, and
  `_noise_pretrain_field_model` exports through the new training module.
- Redirected scene-product field rendering to import from
  `torchsky.map.field.rendering` instead of the old core module.
- Extended `audit-refactor-cohesion` so training, loss, and noise helpers are
  required outside `field/core.py`.
- Tests run:
  - `pixi run -e test -- pytest tests/test_map_field.py::test_fit_field_model_reduces_loss_and_renders_cube tests/test_map_field.py::test_noise_pretraining_updates_trunk_but_restores_decoder_head tests/test_map_field.py::test_fit_field_model_supports_pixel_sampled_training tests/test_map_field.py::test_fit_field_model_supports_mixture_mc_training tests/test_map_field_strainer.py tests/test_field_fit_bundle.py tests/test_refactor_cohesion_audit.py -q`
  - Result: `16 passed, 2 warnings`.
  - `pixi run -e test -- pytest tests/test_docs_integrity.py tests/test_api.py::test_field_namespace_uses_short_composable_verbs -q`
  - Result: `2 passed, 2 warnings`.
  - `pixi run -e test -- pytest --lf -q`
  - Result after fixing the stale scene import: `114 passed, 4 skipped, 2 warnings`.
  - `pixi run audit-refactor-cohesion`
  - Result: `0 failures, 0 warnings`.
  - `pixi run preflight-push`
  - Result: `All checks passed`.
  - `pixi run typecheck`
  - Result: `0 errors, 0 warnings, 0 informations`.
  - `pixi run test`
  - Result: `1618 passed, 61 skipped, 1 xfailed, 88 warnings`.
- North Star connection: `field/core.py` is now mostly model definitions,
  factories, and decomposition regularizers, while fitting policy and optimizer
  orchestration live in a module named for the job.

- Added executable target binding to `torchsky.scene.ObservationOperatorAdapter`.
- Routed adapter-bound targets through the existing `build_solver_hooks` /
  `BoundObservationOperator` substrate so promoted scene workflows can assemble
  normal-equation probes and run CG solves without reaching into map internals.
- Kept exact adjoints explicit: solve/probe workflows require an
  `apply_adjoint` hook or target binder for the concrete operator family.
- Added scene-facade tests for the exact adjoint relation and a normal-equation
  recovery solve through the promoted adapter.
- Tests run:
  - `pixi run -e test -- pytest tests/test_scene_facade.py -q`
  - Result: `5 passed, 2 warnings`.
  - `pixi run -e test -- pytest tests/test_map_solvers.py tests/test_map_operator_bundle.py -q`
  - Result: `58 passed, 2 warnings`.
  - `pixi run preflight-push`
  - Result: `All checks passed`.
  - `pixi run typecheck`
  - Result: `0 errors, 0 warnings, 0 informations`.
  - `pixi run -e test -- pytest tests/test_docs_integrity.py -q`
  - Result: `1 passed`.
- North Star connection: `torchsky.scene` can now describe and execute the same
  observation law as the lower-level map solver path, which is the first
  promoted gate for making `ObservationOperatorBundle` the common execution
  substrate.

- Added `ObservationOperatorAdapter.bind_image_plane_target(...)`, which maps a
  target-grid tensor scene into an `ImageObservation` grid using the existing
  image-plane projection helper.
- Implemented the corresponding exact adjoint as the vector-Jacobian product of
  that same projection, so the promoted path has a real `A` / `A^T` pair instead
  of separate approximation code.
- Added `scene.bind_image_plane_observations_to_target(...)` to assemble a
  `BoundObservationOperatorSet` for multiple image-plane observations sharing
  one latent target grid.
- Routed scalar image-observation weights plus explicit weight/mask/primary-beam
  maps through the bound operator inverse-noise hook.
- Added same-grid gates for projection identity, the adjoint dot-product
  relation, single-observation normal-equation recovery, and summed
  multi-observation normal equations, including scalar inverse-noise weighting.
- Tests run:
  - `pixi run -e test -- pytest tests/test_scene_facade.py -q`
  - Result: `9 passed, 2 warnings`.
- North Star connection: the promoted scene facade now has an executable
  image-plane inverse operator over a shared latent target grid, which is the
  first concrete bridge from classical image stacks toward joint scene solving.

- Added `FieldResidualProduct`, `field_image_residual_product(...)`, and
  `render_field_residual_product(...)` so rendered residual products keep
  prediction, target, residual, weight, support, band label, and metrics together.
- Added `field_quality_scorecard(...)` with by-band aggregates and exact extrema
  for worst weighted RMSE, lowest residual SNR, lowest support fraction, and
  lowest effective weight.
- Kept `render_field_quality(...)` backward-compatible by making scorecards and
  residual products opt-in with `include_scorecard` and
  `include_residual_products`.
- Exported the new helpers through `torchsky.map.field`, compatibility lazy
  `torchsky.map` access, and concise `torchsky.field` names.
- Tests run:
  - `pixi run -e test -- pytest tests/test_field_diagnostics.py -q`
  - Result: `10 passed, 2 warnings`.
- North Star connection: field fits can now emit support-aware products and
  band-group diagnostics that are closer to a survey QA artifact than a single
  pixel-RMSE scalar.

- Added `FieldResidualTile` and `select_field_residual_tiles(...)` to rank
  fixed-size residual-product tiles by exact weighted residual energy.
- Exported the selector through `torchsky.map.field`, compatibility lazy
  `torchsky.map`, and concise `torchsky.field.residual_tiles`.
- Added a focused gate for residual-energy ranking, tile bounds, band label
  propagation, and explicit support filtering.
- Tests run:
  - `pixi run -e test -- pytest tests/test_field_diagnostics.py -q`
  - Result: `11 passed, 2 warnings`.
- North Star connection: adaptive local capacity now has a support-aware,
  survey-scale evidence primitive before adding any sparse basis or feature-tile
  model parameters.

- Added `torchsky.map.field.adaptive` with `ResidualTileBasisField`,
  `ResidualTileBasisSpec`, `residual_tile_basis_spec_from_observation(...)`, and
  `make_residual_tile_basis_field(...)`.
- The new field wrapper adds bounded signed local Gaussian correction atoms on
  top of an existing diffuse field. Atom centers/scales come from residual-tile
  WCS geometry; amplitudes initialize from weighted mean residuals.
- Exported the adaptive helper through `torchsky.map.field`, lazy `torchsky.map`,
  and concise `torchsky.field.residual_tile_basis`.
- Added a focused gate that selects the strongest residual tile, initializes the
  local correction amplitude, and verifies stronger response at the tile WCS
  center than away from it.
- Tests run:
  - `pixi run -e test -- pytest tests/test_field_diagnostics.py -q`
  - Result: `12 passed, 2 warnings`.
- North Star connection: the field stack now has its first explicit opt-in
  local-capacity correction layer, tied to residual evidence and bounded by a
  caller-specified tile budget.

- Extracted image-plane inverse operator binding into
  `torchsky.map.operators.inverse`.
- `bind_image_plane_target_operator(...)` and
  `bind_image_plane_observations_to_target_operator_set(...)` now own the
  concrete target-grid scene to image-observation projection, exact VJP adjoint,
  and inverse-noise hook.
- Simplified `torchsky.scene.ObservationOperatorAdapter.bind_image_plane_target`
  and `scene.bind_image_plane_observations_to_target(...)` into facade delegates
  over the shared map-operator implementation.
- Added a direct map-operator gate for projection identity and the adjoint
  dot-product relation, alongside the existing scene facade gates.
- Tests run:
  - `pixi run -e test -- pytest tests/test_map_operator_bundle.py tests/test_scene_facade.py -q`
  - Result: `20 passed, 2 warnings`.
- North Star connection: the executable `A` / `A^T` image-plane inverse path is
  now a map-operator primitive rather than scene-facade policy.

- Promoted `MOC.from_polygon(...)` to the shared general spherical-polygon
  pixelizer so simple non-convex footprints no longer use the convex-only
  HEALPix polygon query path.
- Added `MOC.from_region(...)` for torchsky spherical regions, native/exact
  region wrappers, boolean regions, and `PixelizedRegion` inputs, preserving
  NUNIQ/NESTED semantics and converting RING pixels explicitly at the boundary.
- Added `MOC.to_pixelized_region(...)` so MOC coverage can round-trip back into
  the region boolean API at a requested HEALPix resolution.
- Tests run:
  - `pixi run -e test -- pytest tests/test_sphere_moc.py tests/test_sphere_ecosystem_p3.py tests/test_sphere_polygons.py tests/test_sphere_exact_regions.py -q`
  - Result: `54 passed, 2 warnings`.
  - `pixi run preflight-push`
  - Result: passed.
  - `pixi run typecheck`
  - Result: passed.
  - `pixi run -e test -- pytest tests/test_docs_integrity.py -q`
  - Result: `1 passed`.
- North Star connection: exact/native region work now feeds survey-scale MOC,
  sparse-map, and FITS coverage products directly instead of remaining a
  separate geometry side surface.

- Reworked `HealSparseMap.from_moc(...)` to build sparse nested pixels directly
  from `MOC.to_pixelized_region(...)` instead of allocating a dense full-sky
  boolean mask.
- Fixed `MOC.to_sparse_map(...)` to pass explicit coverage values, sentinel,
  dtype, and coverage resolution through `HealSparseMap.from_moc(...)`; small
  NSIDE maps now default to `nside_coverage=1` instead of an invalid zero.
- Added focused sparse/MOC tests for direct nested-pixel realization, boolean
  coverage values, float coverage-map values, and dense readback.
- Tests run:
  - `pixi run -e test -- pytest tests/test_sphere_moc.py tests/test_sphere_ecosystem_p2.py tests/test_sphere_ecosystem_p3.py tests/test_sphere_sparse.py -q`
  - Result: `49 passed, 2 warnings`.
  - `pixi run preflight-push`
  - Result: passed.
  - `pixi run typecheck`
  - Result: passed.
  - `pixi run -e test -- pytest tests/test_docs_integrity.py -q`
  - Result: `1 passed`.
- North Star connection: MOC-to-sparse conversion now scales with covered
  pixels, not all-sky pixel count, which is the right direction for survey
  footprints and sparse masks.

- Stopped coercing sparse FITS `VALUE` columns to `float32` on write/read.
- Added dtype-preserving sparse FITS roundtrip coverage for boolean, integer,
  and float64 sparse maps while retaining metadata/header roundtrip behavior.
- Tests run:
  - `pixi run -e test -- pytest tests/test_sphere_sparse.py tests/test_sphere_ecosystem_p2.py::test_sparse_fits_io -q`
  - Result: `13 passed, 2 warnings`.
  - `pixi run preflight-push`
  - Result: passed.
  - `pixi run typecheck`
  - Result: passed.
  - `pixi run -e test -- pytest tests/test_docs_integrity.py -q`
  - Result: `1 passed`.
- North Star connection: sparse survey masks and typed coverage products can now
  survive FITS interchange without value-type degradation, which matters for
  bool masks, label-like maps, and higher-precision weights.

- Added `key=` support to file-backed `torchsky.get_wcs(...)` and
  `torchsky.io.get_wcs(...)`, so alternate FITS-WCS versions can be loaded from
  paths without manually opening the HDU.
- Extended file-backed TAB autodiscovery to respect the selected alternate WCS
  key, including `CTYPEiA` and `PSi_jA` metadata.
- Tests run:
  - `pixi run -e test -- pytest tests/test_wcs_api.py::test_get_wcs_from_file_selects_alternate_key tests/test_wcs_api.py::test_wcs_alternate_key_matches_astropy tests/test_wcs_tabular.py -q`
  - Result: `23 passed`.
  - `pixi run preflight-push`
  - Result: passed.
  - `pixi run typecheck`
  - Result: passed.
  - `pixi run -e test -- pytest tests/test_docs_integrity.py -q`
  - Result: `1 passed`.
- North Star connection: alternate WCS and table-driven axes are now usable
  through the same file-level entry point as primary WCS, which matters for
  survey products carrying multiple calibrated coordinate solutions.

- Added `MOC.contains_pixels(nside, pixels, nest=True)` for direct HEALPix
  pixel membership checks in MOC coverage.
- Routed lon/lat `MOC.contains(...)` through the same range-membership logic so
  point and pixel containment use one implementation.
- Fixed `MOC.from_sparse_map(...)` to convert RING-ordered sparse-map pixels to
  NESTED before NUNIQ construction.
- Added nested/RING ordering tests plus an explicit lower-resolution rejection
  gate and a sparse-map RING conversion regression.
- Tests run:
  - `pixi run -e test -- pytest tests/test_sphere_moc.py tests/test_sphere_sparse.py tests/test_sphere_ecosystem.py tests/test_sphere_ecosystem_p2.py tests/test_sphere_ecosystem_p3.py -q`
  - Result: `59 passed, 2 warnings`.
  - `pixi run -e test -- pytest tests/test_wcs_api.py::test_get_wcs_from_file_selects_alternate_key tests/test_wcs_tabular.py -q`
  - Result: `22 passed`.
  - `pixi run preflight-push`
  - Result: passed.
  - `pixi run typecheck`
  - Result: passed.
  - `pixi run -e test -- pytest tests/test_docs_integrity.py -q`
  - Result: `1 passed`.
- North Star connection: high-NSIDE catalogs and sparse maps can now filter by
  MOC coverage without converting already-indexed pixels back through
  coordinates.

- Final full local gate for this continuation:
  - `pixi run ci-local`
  - Result: `1688 passed, 59 skipped, 77 warnings`.

- Enforced sorted-unique sparse-map pixels inside `SparseHealpixMap`.
- `from_pixels(...)`, direct construction, and `update_values_pix(...)` now keep
  the last value supplied for duplicate pixels, matching update semantics and
  preserving correct `searchsorted`-based `get_values(...)` lookups.
- Added single-band, multi-band, and update-overwrite regression tests plus
  upstream-style valid-pixel/get-values parity checks.
- Tests run:
  - `pixi run -e test -- pytest tests/test_sphere_sparse.py tests/test_healsparse_upstream.py::test_healsparse_quickstart_make_empty_update_and_valid_pixels_match tests/test_healsparse_upstream.py::test_healsparse_get_values_matches_upstream -q`
  - Result: `17 passed, 2 warnings`.
- North Star connection: sparse survey masks and catalogs now have deterministic
  indexed lookup behavior even when input chunks arrive unsorted or contain
  repeated pixel updates.

- Fixed `MOC.from_threshold(...)` to select actual sparse-map pixels after
  thresholding instead of reconstructing global pixels from coverage-map block
  offsets.
- Added RING sparse-map conversion in the threshold path so threshold-derived
  NUNIQ coverage is correct for both sparse-map orderings.
- Tests run:
  - `pixi run -e test -- pytest tests/test_sphere_moc.py tests/test_sphere_sparse.py tests/test_sphere_ecosystem_p2.py tests/test_sphere_ecosystem_p3.py -q`
  - Result: `60 passed, 2 warnings`.
- North Star connection: thresholded survey masks now preserve the exact sparse
  footprint users supplied, including sparse/non-contiguous chunks.

- Hardened independent FITS `-TAB` table-column lookup.
- Exact column names still win; when exact lookup fails, coordinate and indexing
  columns can resolve through a unique case-insensitive match, while ambiguous
  case variants fail explicitly.
- Tests run:
  - `pixi run -e test -- pytest tests/test_wcs_tabular.py -q`
  - Result: `23 passed`.
- North Star connection: table-driven WCS ingestion is more tolerant of common
  FITS table naming drift without guessing when a file is ambiguous.

- Extended `MOC.from_ascii(...)` to parse compact inclusive MOC pixel ranges
  such as `3/5-7` and comma-separated range/list mixtures.
- Added optional `MOC.to_ascii(compact=True)` so contiguous pixel runs can be
  emitted as compact ranges while the default output remains unchanged.
- Added fail-closed coverage for descending ranges.
- Tests run:
  - `pixi run -e test -- pytest tests/test_sphere_moc.py tests/test_sphere_ecosystem_p3.py -q`
  - Result: `42 passed, 2 warnings`.
- North Star connection: compact MOC coverage exchange can ingest common ASCII
  serializations without requiring callers to expand large pixel lists first.

- Final full local gate after this continuation:
  - `pixi run ci-local`
  - Result: `1698 passed, 59 skipped, 91 warnings`.

- Hardened `SparseHealpixMap` coverage hierarchy handling.
- Direct construction, `from_dense(...)`, `from_pixels(...)`, updates, and FITS reads now
  materialize a derived `coverage_map` whenever `nside_coverage` is set, and invalid
  coverage NSIDE values are rejected before parent-index math can become invalid.
- RING-ordered sparse maps now compute coverage parents by converting sparse pixels
  through the NESTED hierarchy first, so `coverage_mask` and `coverage_map` match
  HealSparse-style coverage semantics instead of RING integer-division artifacts.
- `SparseHealpixMap.ud_grade(...)` now preserves valid coverage hierarchy metadata,
  retaining the old coverage NSIDE when possible and lowering it to the output NSIDE
  when degrading below the old coverage resolution.
- Added native regression tests plus upstream HealSparse parity coverage for
  coverage-mask ordering and resampled coverage resolution.
- Tests run:
  - `pixi run -e test -- pytest tests/test_sphere_sparse.py tests/test_healsparse_upstream.py::test_healsparse_coverage_mask_matches_upstream tests/test_healsparse_upstream.py::test_healsparse_ud_grade_coverage_resolution_matches_upstream tests/test_sphere_ecosystem.py::test_sparse_map tests/test_sphere_ecosystem_p2.py::test_sparse_fits_io -q`
  - Result: `25 passed, 2 warnings`.
- North Star connection: sparse survey masks now carry stable, ordering-independent
  hierarchical coverage through construction, mutation, persistence, and resolution
  changes.

- Final full local gate after this continuation:
  - Initial `pixi run ci-local` hit a stale TorchInductor precompiled-header cache
    failure in `tests/test_wcs_parity.py::test_projection_inverse_stack_compiled_matches_astropy[1-HPX]`.
  - Exact failing test rerun:
    `pixi run -e test -- pytest 'tests/test_wcs_parity.py::test_projection_inverse_stack_compiled_matches_astropy[1-HPX]' -q`
  - Result: `1 passed, 14 warnings`.
  - Clean full rerun: `pixi run ci-local`
  - Result: `1705 passed, 59 skipped, 77 warnings`.

- Extended FITS `-TAB` parsing beyond independent one-dimensional columns.
- `tabular_axes_from_fits(...)` now accepts `PVi_3` mappings when a
  multidimensional coordinate array is exactly separable along the selected table
  dimension, recovering it into the existing per-axis `TabularAxis` execution
  path.
- Multidimensional coordinate arrays without `PVi_3`, or arrays that vary along
  non-selected dimensions, fail explicitly instead of being flattened or guessed.
- Added a `WCS` roundtrip gate for two separable multidimensional table axes plus
  fail-closed tests for missing `PVi_3` and genuinely coupled coordinate arrays.
- Tests run:
  - `pixi run -e test -- pytest tests/test_wcs_tabular.py -q`
  - Result: `24 passed`.
- North Star connection: table-driven cube axes can now ingest a wider class of
  realistic FITS lookup tables while preserving exact semantics and leaving truly
  coupled interpolation as an explicit future substrate.

- Final full local gate after this continuation:
  - `pixi run ci-local`
  - Result: `1706 passed, 59 skipped, 77 warnings`.

- Added the first torch-native coupled multidimensional FITS `-TAB` lookup
  substrate.
- `CoupledTabularLookupWCS` performs multilinear forward interpolation over
  explicit table-grid pixel axes, supports autograd through the local cell
  weights, exposes metadata names/units, and intentionally fails closed for the
  general inverse instead of guessing one.
- Added `coupled_tabular_wcs_from_fits(...)` to build that forward substrate from
  nonseparable multidimensional `-TAB` coordinate arrays with `PVi_3` mappings,
  including finite/shape validation and default table-index axes.
- Added focused gates for multilinear interpolation, clamp/NaN bounds behavior,
  autograd slope propagation, inverse fail-closed behavior, and FITS helper
  construction from genuinely nonseparable coordinate arrays.
- Tests run:
  - `pixi run -e test -- pytest tests/test_wcs_tabular.py -q`
  - Result: `27 passed`.
  - `pixi run preflight-push`
  - Result: passed.
- North Star connection: torchsky now has a native forward engine for coupled
  table-coordinate cubes, which is the substrate needed before adding honest
  inverse solving and main-`WCS` integration for arbitrary multidimensional
  `-TAB`.

- Final full local gate after this continuation:
  - `pixi run ci-local`
  - Result: `1709 passed, 59 skipped, 63 warnings`.
  - Note: pytest exited cleanly after printing existing nanobind shutdown leak
    diagnostics for `FITSFile`/`TableReader`.

- Autonomous review closeout hardened consistency and edge cases across the
  current roadmap changes.
- Table-WCS fixes now support decreasing pixel grids for independent axes,
  normalize descending coupled-grid axes, reject complex lookup values, and keep
  coupled `-TAB` inverse behavior fail-closed.
- Sparse-map fixes preserve metadata through `ud_grade(...)`, move pixels to the
  value device before canonicalization, build coverage maps deterministically
  without duplicate scatter ordering, and cover multiband duplicate updates.
- Image-plane inverse binding now imports through domain modules, caches
  inverse-noise maps by device and image shape, rejects non-image noise inputs,
  clamps scalar inverse weights to nonnegative values, and checks adjoint input
  shape before autograd can broadcast.
- Field and spectral review fixes aligned support fractions with positive
  effective weights, fixed middle/last spectral-axis y/x indexing in cube
  spectrum extraction and plotting, and added a spectral-cache clear regression.
- Root and map facades remain lean in `__all__` while compatibility aliases are
  still lazily resolvable through `__getattr__` and `dir(...)`.
- Tests run:
  - `pixi run -e test -- pytest tests/test_wcs_tabular.py -q`
  - Result: `29 passed`.
  - `pixi run -e test -- pytest tests/test_sphere_moc.py tests/test_sphere_sparse.py tests/test_healsparse_upstream.py::test_healsparse_ud_grade_coverage_resolution_matches_upstream tests/test_healsparse_upstream.py::test_healsparse_geom_circle_coverage_matches_upstream_sparse_mask -q`
  - Result: `56 passed, 2 warnings`.
  - `pixi run -e test -- pytest tests/test_spectral.py::TestDataCube::test_spectrum_extraction tests/test_spectral.py::TestDataCube::test_spectrum_extraction_preserves_yx_for_middle_spectral_axis tests/test_spectral.py::TestDataCube::test_spectrum_extraction_preserves_yx_for_last_spectral_axis tests/test_viz_plots.py tests/test_field_diagnostics.py::test_field_quality_scorecard_groups_bands_and_tracks_extrema tests/test_field_diagnostics.py::test_select_field_residual_tiles_ranks_weighted_residual_energy tests/test_field_diagnostics.py::test_residual_tile_basis_field_initializes_local_capacity_from_tiles -q`
  - Result: `11 passed, 2 warnings`.
  - `pixi run -e test -- pytest tests/test_api.py::test_top_level_namespaces_expose_phase1_modules tests/test_api_surfaces.py tests/test_cold_import.py -q`
  - Result: `8 passed, 2 warnings`.
  - `pixi run -e test -- pytest tests/test_map_operator_bundle.py::test_image_plane_target_operator_binder_exposes_exact_adjoint tests/test_map_operator_bundle.py::test_image_plane_target_operator_noise_is_shape_checked_and_nonnegative -q`
  - Result: `2 passed, 2 warnings`.
  - `pixi run -e test -- pytest tests/test_sphere_spectral.py::test_clear_spectral_caches_resets_lookup_state -q`
  - Result: `1 passed, 2 warnings`.

- Final full local gate after this autonomous review:
  - `pixi run ci-local`
  - Result: `1718 passed, 59 skipped, 77 warnings`.
  - Note: pytest exited cleanly after printing existing nanobind shutdown leak
    diagnostics for `FITSFile`.

- Extended all-code review pass after the initial closeout.
- Lean-import cleanup now routes dataset, spectral, frame, map I/O, CLI, sphere
  FITS helpers, and viz request handlers through concrete domain modules instead
  of package-root lazy dispatch where practical, leaving table/HDU internals
  unchanged unless they expose a concrete issue.
- Spherical pairwise distances now use the same stable `atan2(cross, dot)`
  formulation as scalar great-circle distances, preserving tiny separations that
  `acos(dot)` can round to zero.
- High-level spectral helpers now normalize FREQ units before wavelength
  conversion, use exact speed-of-light constants, compare spectral type/unit
  names case-insensitively, fail closed for missing FITS `-TAB` coordinate
  tables, and resample `Spectrum1D` with explicit linear interpolation plus
  inverse-variance propagation instead of nearest-neighbor overwrite.
- ML transforms now use both `AsinhStretch` parameters, avoid cached CPU tensor
  constants in `LogStretch`, handle explicit scalar `Normalize(...)` values on
  the input device/dtype, and replace percentile `ZScale` with iterative
  sigma-clipped zscale limits.
- `IterableFITSDataset` now consumes real JSON or newline manifests from local,
  `file://`, `http://`, or `https://` indexes instead of fabricating placeholder
  shard file names.
- Focused tests run:
  - `pixi run -e test -- pytest tests/test_spectral.py tests/test_spectral_nonuniform.py tests/test_transforms.py tests/test_datasets.py tests/test_dataloader.py tests/test_sphere_core.py tests/test_sphere_moc.py tests/test_sphere_sparse.py -q`
  - Result: `147 passed, 1 skipped, 2 warnings`.
- Final full local gate after this extended review:
  - First `pixi run ci-local` found one stale frame test mock after the direct
    `torchsky.io.write` import cleanup; the targeted rerun
    `pixi run -e test -- pytest tests/test_frame_stypes.py::test_write_tensor_frame_stypes -q`
    passed after updating the mock.
  - Clean full rerun: `pixi run ci-local`
  - Result: `1721 passed, 59 skipped, 63 warnings`.
  - Note: pytest exited cleanly after printing existing nanobind shutdown leak
    diagnostics for `FITSFile`.
