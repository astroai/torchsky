# Autonomous Refactor Log

Target branch: `codex/autonomous-refactor-2026-05`

Start gate: May 8, 2026, `America/Vancouver`

## Current State

- Resumed manually on June 8, 2026 after the scheduled automation did not run.
- Active branch: `codex/autonomous-refactor-2026-05`.
- Completed A0, A1, A2, B1-B4, C1-C2, D1, and D2 from
  `docs/reviews/autonomous_refactor_launch.md`.
- Backward compatibility for unused alias namespaces is no longer a goal; the
  public API should stay clean and route users to the owning modules.
- Next recommended slice: E1, adding narrow `torchsky info` and `torchsky cat`
  CLI workflows.

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

## Work Queue Snapshot

- [x] A0: Establish branch, baseline, and safety checks.
- [x] A1: Declare the public modelling spine in docs and API-surface metadata.
- [x] A2: Add or tighten tests for public API tiers and cold import behavior.
- [x] B1: Extract root FITS I/O implementation without user-visible churn.
- [x] B2: Move Python-side root I/O caches behind a single cache facade.
- [x] B3: Make `torchsky.io` call the implementation directly.
- [x] B4: Preserve root FITS public behavior.
- [x] C1: Add promoted observation/product/scene facades.
- [x] C2: Define executable operator protocols or adapters.
- [x] D1: Move HEALPix ownership toward `torchsky.sphere`.
- [x] D2: Remove leftover HEALPix, coords, and geom alias surfaces.
- [ ] E1: Add `torchsky info` and `torchsky cat` CLI workflows.
- [ ] F1: Rationalize field model exposure through presets and factories.
- [ ] G1: Continue roadmap implementation from `docs/roadmap/status.md`.
