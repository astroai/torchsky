# Torchsky Audit Report

> **Date:** April 3, 2026
> **Scope:** Code quality & conventions, Performance & scalability
> **Repository:** `/Users/fabbros/src/torchsky`
> **Version:** 0.4.0.dev0

---

## Executive Summary

Torchsky is a substantial ~22K LOC project building a torch-native astronomy stack for sky data, coordinates, geometry, mapmaking, and catalog workflows. The codebase demonstrates strong engineering in several areas — spherical geometry primitives, HEALPix operations, and catalog crossmatching are well-designed and benchmarked. However, the audit identified **3 critical bugs**, **multiple AGENTS.md principle violations**, **significant performance bottlenecks in hot paths**, and **structural maintainability concerns** that should be addressed before the next release.

### Key Findings at a Glance

| Category | Critical | High | Medium | Low |
|----------|----------|------|--------|-----|
| Bugs | 3 | 4 | 3 | 2 |
| AGENTS.md Violations | 4 | 3 | 2 | 1 |
| Performance | 2 | 5 | 4 | 3 |
| Structure | 1 | 3 | 3 | 2 |
| Benchmarks | 0 | 2 | 3 | 2 |

---

## Table of Contents

1. [Critical Bugs](#1-critical-bugs)
2. [AGENTS.md Principle Violations](#2-agentsmd-principle-violations)
3. [Performance Bottlenecks](#3-performance-bottlenecks)
4. [Code Organization & Structure](#4-code-organization--structure)
5. [API Stability Concerns](#5-api-stability-concerns)
6. [Device-Specific Issues](#6-device-specific-issues)
7. [Benchmark Gaps](#7-benchmark-gaps)
8. [Dead Code & Unused Imports](#8-dead-code--unused-imports)
9. [Recommendations](#9-recommendations)

---

## 1. Critical Bugs

**Resolution note (April 2026 follow-up):** The three issues below are **fixed or hardened** in-tree: (1) `table.py` keeps `_COMPLEX_DTYPE_MAP` as a dtype **dict** and uses a separate **`_COMPLEX_TFORM_CODES` frozenset** for TFORM membership so a set/dict confusion cannot recur; regression test `tests/test_audit_critical_regressions.py::test_complex_tform_map_remains_subscriptable_after_update_rows`. (2) `MOC.flatten` allocates the dense mask on **`pix.device`**; regression tests in the same file for CPU and CUDA. (3) `read()` image fast paths catch **`_READ_EXC_TYPES`** (not bare `Exception`) and log at DEBUG on fallback; see `src/torchsky/__init__.py` near `_READ_EXC_TYPES`.

### 1.1 `_COMPLEX_DTYPE_MAP` Set-vs-Dict Bug

**File:** `src/torchsky/table.py:3356`
**Severity:** Critical — causes crash after `update_rows` is called

```python
global _COMPLEX_DTYPE_MAP
if not _COMPLEX_DTYPE_MAP:
    _COMPLEX_DTYPE_MAP = {"C", "M"}  # This is a SET, not a dict!
```

The module-level `_COMPLEX_DTYPE_MAP` is initialized as a dict mapping FITS type codes to numpy dtypes:

```python
_COMPLEX_DTYPE_MAP = {"C": "complex64", "M": "complex128"}
```

But `update_rows` reassigns it to a **set** `{"C", "M"}`. Any subsequent code that does `_COMPLEX_DTYPE_MAP[code]` will raise `TypeError: 'set' object is not subscriptable`.

There is a second shadowing issue at `table.py:315` where `_normalize_mutation_rows` reassigns the module-level dict with string values, shadowing the original numpy-dtype version.

**Fix:** Use `dataclasses.replace()` or a factory pattern instead of mutating module-level state. Never reassign shared constants.

---

### 1.2 Device Mismatch in `MOC.flatten()`

**File:** `src/torchsky/sphere/moc.py:297`
**Severity:** Critical — crashes on non-CPU devices

```python
def flatten(self, nside: int) -> Tensor:
    mask = torch.zeros(12 * nside**2, dtype=torch.bool)  # Always CPU
    pix = self._flat_pixels(nside)
    if pix.numel() > 0:
        mask[pix] = True  # pix may be on CUDA → crash
    return mask
```

`self.uniq` (and therefore `pix`) may reside on CUDA or MPS, but `mask` is always created on CPU. The indexing operation `mask[pix]` will raise a RuntimeError when devices don't match.

**Fix:** Create the mask on the same device as `pix`:
```python
mask = torch.zeros(12 * nside**2, dtype=torch.bool, device=pix.device)
```

---

### 1.3 Silent Exception Swallowing in Image Read Fast Path

**File:** `src/torchsky/__init__.py:1026-1031`
**Severity:** Critical — masks bugs, makes debugging extremely difficult

```python
if scale_on_device and not raw_scale and device == "cpu" and ...:
    try:
        ...
    except Exception:
        if force_image:
            raise
        skip_generic_image_fast_path = True
        pass
```

This catches **all** exceptions (including `KeyboardInterrupt`, `SystemExit`) in a hot path. If the C++ reader has a bug, it silently falls through to a completely different code path. Users will see incorrect results or degraded performance with no indication of why.

**Fix:** Catch specific exception types (e.g., `RuntimeError`, `ValueError`). Log the exception at DEBUG level before falling back.

---

## 2. AGENTS.md Principle Violations

### 2.1 Principle 1: No Heuristics Where Exact Methods Are Available

AGENTS.md states: *"Do not introduce ad hoc thresholds, fallback rules, or data-dependent branching unless there is a clear numerical, algorithmic, or benchmark-backed reason. If a heuristic is temporarily necessary, isolate it, document why it exists, and add a benchmark or regression test that justifies it."*

**Violations found:**

| Location | Function | Heuristic | Evidence |
|----------|----------|-----------|----------|
| `__init__.py:554-617` | `_should_use_cold_nommap` | 1 MiB file size cutoff, BITPIX-based branching | None |
| `__init__.py:625-669` | `_resolve_image_mmap` | Auto-selects mmap behavior based on file characteristics | None |
| `table.py:1104-1149` | `_can_use_mmap_row_path_for_full_read` | Docstring literally says "Heuristic for full-table Arrow reads" | None |
| `table.py:1152-1194` | `_can_use_torch_table_path_for_full_read` | Docstring literally says "Heuristic for direct C++ torch-table full reads" | None |
| `table.py:1431-1438` | `read` hybrid strategy | `100000` row threshold for VLA, `1000` for non-VLA | Env var override only |
| `catalogs/core.py:145-166` | `_choose_default_coarse_nside` | Thresholds: 25000, 90000, 180000, 1000000 | Comment says "Benchmark-backed" but thresholds are unexplained |

**Acceptable (documented as benchmark-backed):**

| Location | Function | Note |
|----------|----------|------|
| `wcs/core.py:26-42` | `_CENTER_CPP_P2W_THRESHOLDS` / `_CENTER_CPP_W2P_THRESHOLDS` | Comment says "Benchmark-backed small-batch CPU crossover points" — but values are hardcoded per-projection with no mechanism to re-validate |

---

### 2.2 Principle 4: Preserve Differentiability and Tensor-Native Execution

AGENTS.md states: *"Keep user-facing compute paths in PyTorch unless there is a strong reason not to. Avoid unnecessary host transfers, NumPy-only detours, or CPU-only assumptions in public tensor paths."*

**Violations found:**

- **`wcs/core.py:55-90`** — `_wrap_lon360` / `_wrap_lon180` check `requires_grad` and take different code paths. This silently changes behavior based on gradient tracking, which could break autograd chains unexpectedly.

- **`sphere/geom.py:1579-1595`** — `ExactSphericalRegion.contains` uses a Python for-loop over numpy arrays:
  ```python
  [bool(self._sg_poly.contains_lonlat(...)) for lo, la in zip(...)]
  ```
  Completely non-differentiable and not tensor-native.

- **`core.py:232`** — `process_data` takes `np.ndarray` as input:
  ```python
  def process_data(data: np.ndarray, header: Dict[str, Any], ...) -> torch.Tensor:
  ```
  Violates the tensor-native execution principle.

- **`sphere/geom.py:369`** — `_gnomonic_project` returns `.cpu().numpy()`, forcing a host transfer in what is otherwise a torch-native pipeline.

- **`sphere/geom.py:416-478`** — `_triangulate_simple_polygon_2d` runs entirely in Python on numpy arrays with `del indices[k]` list deletion.

---

### 2.3 Principle 5: Cross-Device Behavior Matters

AGENTS.md states: *"CPU is the baseline. CUDA and MPS support should be preserved where the surrounding module claims it. Device-specific optimizations must not silently change semantics."*

**Violations found:**

- **`sphere/spectral.py`** — 20+ module-level cache dicts that are never device-aware. If tensors are cached on CPU and the user expects CUDA execution, this causes silent device mismatches or unnecessary transfers.

- **`wcs/healpix.py:264, 321, 415, 488, 547, 559`** — Multiple functions route through CPU for MPS:
  ```python
  return func(...).cpu()...to(device)
  ```
  This causes full device-host-device roundtrips for every call.

- **`__init__.py:710-770`** — `read_image` has many `hasattr(cpp, ...)` checks that silently fall back to different code paths. If a C++ function is missing, the semantics may change without warning.

---

### 2.4 Principle 3: Optimize for Scalability First

AGENTS.md states: *"Assume catalogs, maps, cubes, and tables may be large enough that naive O(N*M) behavior is unacceptable."*

Multiple O(N*M) patterns exist in hot paths (detailed in [Section 3](#3-performance-bottlenecks)), most notably:

- `wcs/healpix.py:1433` — `query_disc` scans all pixels: O(Npix)
- `wcs/healpix.py:1530` — `query_polygon` scans all pixels: O(Npix)
- `sphere/geom.py:114-117` — `_points_on_edges` creates O(M×N) intermediate tensor

---

## 3. Performance Bottlenecks

### 3.1 O(N×M) Patterns in Hot Paths

| File:Line | Function | Pattern | Impact |
|-----------|----------|---------|--------|
| `sphere/geom.py:114-117` | `_points_on_edges` | `torch.einsum("mj,nj->mn", points, vertices)` creates full M×N distance matrix | 1.6GB for M=1M, N=100 |
| `sphere/geom.py:608-617` | `spherical_polygon_contains` fallback | Full M×N projection matrix + winding number computation | Worst-case for large polygons |
| `sphere/geom.py:727-732` | `spherical_polygons_intersect` | Creates O(E1×E2×3) intermediate tensor for edge-edge intersections | Quadratic memory in edge count |
| `sphere/geom.py:1025-1032` | `SphericalMultiPolygon.intersects` | Nested Python loop: `for p in self.polygons: for q in others:` | No early-exit parallelism |
| `sphere/moc.py:139-144` | `MOC.contains` | Python loop `for ns, p in zip(nsides, pix_nested)` | Should be vectorized |
| `sphere/moc.py:150-155` | `MOC.contains` | Creates Python lists then converts to tensors: `torch.tensor([r[0] for r in ranges])` | Unnecessary Python detour |
| `sphere/sparse.py:191-195` | `_build_coverage_map` | Python loop `for cp in unique_cov:` with individual tensor masking | Should use scatter/index_add |
| `sphere/sparse.py:550-563` | `SkyMaskPipe.combine` | Sequentially converts each stage to dense: `d1 = result.to_dense(...)` | 2GB per map at nside=4096 |
| `catalogs/core.py:96-99` | `_CoarseHealpixIndex.lookup` | Python loop with `.tolist()` host transfer | O(num_matching_pixels) Python overhead |
| `catalogs/core.py:212-229` | `_candidate_coarse_pixels_for_left_pixel` | Python `for _ in range(hops):` with `torch.unique(torch.cat(...))` per iteration | Rebuilds neighbor shells per pixel |

### 3.2 Unnecessary NumPy/Host Detours

| File:Line | Function | Issue |
|-----------|----------|-------|
| `sphere/geom.py:369` | `_gnomonic_project` | Returns `.cpu().numpy()` |
| `sphere/geom.py:372-377` | `_signed_area_2d` | Takes numpy array, does `np.roll` operations |
| `sphere/geom.py:416-478` | `_triangulate_simple_polygon_2d` | Ear-clipping in Python with `del indices[k]` — O(N²) worst-case |
| `table.py:424-432` | `_read_table_for_rewrite` | Python loop converting VLA column items one at a time with `.cpu()` and `.numpy()` |
| `table.py:976-987` | `_chunk_to_record_batch` | Python loop over list items with individual `.cpu()` + `.numpy()` |
| `sphere/moc.py:468-469` | `MOC.boundary._key` | Python function called per-vertex in a loop |
| `sphere/moc.py:499-518` | `MOC.boundary` | Python `while current != start:` loop with dict lookups |

### 3.3 Python Loops in Hot Paths

| File:Line | Function | Issue |
|-----------|----------|-------|
| `sphere/spectral.py:396-405` | `_legendre_l_sequence_loop` | Pure Python loop over Legendre recurrence — extremely slow for lmax>1024 |
| `sphere/spectral.py:471-478` | `_ylm_basis` | Python loop calling `_legendre_l_sequence` per m value |
| `map/image_plane.py:1257-1269` | `_forward_project_weighted_to_target` | Nested Python loops for drizzle: `for off_y, wy in ...: for off_x, wx in ...:` |
| `map/field.py:1065-1073` | `_spatial_fourier_features` | Python loop over scales creating sin/cos phases |
| `map/field.py:1086-1094` | `_spectral_basis` | Python loop over spectral frequencies |
| `wcs/healpix.py:1533-1567` | `query_polygon` | Iterates over all pixels in chunks — O(Npix) brute-force |
| `wcs/healpix.py:1517-1527` | `query_strip` | Same O(Npix) brute-force pattern |

### 3.4 Algorithmic Complexity Concerns

| Location | Current | Should Be | Issue |
|----------|---------|-----------|-------|
| `sphere/geom.py:416` `_triangulate_simple_polygon_2d` | O(N²) worst-case | O(N log N) | Ear clipping with `del indices[k]` is O(N) per deletion |
| `wcs/healpix.py:1433` `query_disc` | O(Npix) brute-force | O(√Npix) | No use of HEALPix ring-range pruning |
| `wcs/healpix.py:1530` `query_polygon` | O(Npix) brute-force | O(√Npix) | No use of HEALPix hierarchical structure |
| `sphere/moc.py:74-93` `_ranges_to_uniq` | O(total_pixels) Python while loop | O(log N) per range | Greedy decomposition in Python |
| `catalogs/core.py:212` `_candidate_coarse_pixels_for_left_pixel` | O(hops × neighbors) Python loop | O(1) with precomputed neighbor shells | Rebuilds per pixel |
| `sphere/geom.py:1025` `SphericalMultiPolygon.intersects` | O(P×Q) sequential | O(P×Q) with early-exit + spatial pruning | No bounding-box or cap pre-filtering |

---

## 4. Code Organization & Structure

### 4.1 `__init__.py` is a God Module (2726 lines)

This single file contains:

- Module-level cache state (8+ `OrderedDict` instances)
- Cache invalidation logic (`_invalidate_path_caches`)
- Handle management (`_get_cached_handle`, `_path_signature`)
- Image metadata caching (`_get_image_meta`, `_get_image_meta_from_handle`)
- HDU type caching (`_get_cached_hdu_type`, `_set_cached_hdu_type`)
- mmap resolution heuristics (`_should_use_cold_nommap`, `_resolve_image_mmap`)
- The entire `read()` function (~630 lines, lines 855-1486)
- The entire `write()` function (~200 lines, lines 1622-1741)
- HDU management (`insert_hdu`, `replace_hdu`, `delete_hdu`)
- Batch reading (`read_batch`, `read_hdus`)
- Streaming (`stream_table`, `read_large_table`)
- Cache management (`clear_file_cache`, `get_cache_performance`)
- Checksum utilities
- `SubsetReader` class

**Recommended split:**

| Module | Responsibility |
|--------|---------------|
| `io_core.py` | `read()`, `write()`, `read_image()` core logic |
| `cache_manager.py` | All cache state, invalidation, handles, metadata |
| `hdu_operations.py` | `insert_hdu`, `replace_hdu`, `delete_hdu`, HDU type caching |
| `batch_io.py` | `read_batch`, `read_hdus`, `stream_table`, `read_large_table` |
| `checksums.py` | Checksum utilities, `SubsetReader` |

### 4.2 `table.py` is Enormous (3508 lines)

Contains Arrow conversion, FITS table reading, writing, mutation (insert/append/delete/update/rename/drop columns), where-clause parsing, DuckDB/Polars/Pandas integration, schema extraction, and more. The mutation operations alone span ~1300 lines.

### 4.3 `sphere/spectral.py` is 4141 lines

Contains the full spherical harmonic transform implementation with 20+ module-level caches. Should be split into `map2alm.py`, `alm2map.py`, `spin.py`, `smoothing.py`, `synthesis.py`, and `caching.py`.

### 4.4 Namespace Inconsistency

- `torchsky.geom` re-exports from `torchsky.sphere` (`geom/__init__.py:1-63`) — users won't know whether to import from `geom` or `sphere`
- `torchsky.coords` re-exports from `torchsky.wcs.healpix` and `torchsky.wcs` — creates a circular dependency surface
- `torchsky.io` is a thin delegation layer that just re-imports from `torchsky` (`io/__init__.py:1-130`) — every function is `from .. import X as _X; return _X(...)`. This adds indirection with no value.

### 4.5 Massive Function Sizes

| Function | File | Lines |
|----------|------|-------|
| `read()` | `__init__.py` | ~630 |
| `_setup_tensors()` | `wcs/core.py` | ~300 |
| `_pixel_to_world_2d_fast()` | `wcs/core.py` | ~170 |
| `read()` | `table.py` | ~190 |
| `_chunk_to_record_batch()` | `table.py` | ~100 |

### 4.6 Deep Nesting

`__init__.py:1079-1190` has 7+ levels of nesting in the image read fast path, making it extremely difficult to reason about control flow.

---

## 5. API Stability Concerns

### 5.1 `__all__` Exports Internal Functions

`__init__.py:141-230` exports many internal cache management functions (`configure_for_environment`, `get_cache_stats`, `clear_cache`, `configure_buffers`, `get_buffer_stats`, `clear_buffers`) that should likely be internal.

### 5.2 Private Functions Accessible at Module Level

`__init__.py:233-248` — `_read_header_fast` has a leading underscore suggesting it's private, but is accessible as `torchsky._read_header_fast`.

### 5.3 Global Mutable State

`sphere/compat.py:31-59` — `_STRICT_MODE` is a global boolean toggled by `set_strict_mode()`. This is thread-unsafe and makes tests non-deterministic.

### 5.4 Inconsistent Error Handling

- Some functions raise `ValueError` for invalid input, others raise `RuntimeError`
- `read_large_table` (`__init__.py:2381-2537`) silently returns `{}` on any exception, masking errors
- `read_batch` (`__init__.py:2239-2247`) silently skips failed files with `continue`

### 5.5 Type Hint Inconsistency

- `__init__.py` uses old-style `Optional[List[str]]` (line 776)
- `sphere/core.py` uses modern `Tensor | float` union syntax (line 24)
- `table.py` mixes both styles
- Many functions in `__init__.py` have no return type annotations (e.g., `read()`, `write()`, `read_table()`)

---

## 6. Device-Specific Issues

### 6.1 MPS (Apple Silicon)

| Location | Issue |
|----------|-------|
| `wcs/healpix.py:58-59` | MPS forced to float32 — acceptable but means `ang2pix` on MPS may have precision issues at high nside (>2048) |
| `sphere/core.py:14` | `_float_dtype` returns float32 for MPS — all sphere core ops lose precision on Apple GPUs |
| `wcs/healpix.py:264, 321, 415, 488, 547, 559` | Multiple functions route through CPU for MPS — full device-host-device roundtrips |

### 6.2 CUDA

- **No CUDA kernels exist** for any HEALPix, sphere geometry, or WCS operations. All compute falls back to CPU or Python.
- `sphere/spectral.py` is explicitly "CPU-first" with no CUDA path.
- `wcs/healpix.py` — `_preferred_real_dtype` returns float32 for non-CPU devices, but actual compute paths all force `.to(device="cpu")` before computation.

### 6.3 CPU

- The C++ extension is well-developed for WCS and HEALPix but **not all Python paths use it**.
- `wcs/core.py:162` — `_enable_fused_wcs` defaults to `"1" == "1"` but the comment says "vectorized PyTorch + torch.compile is typically 2x-4x faster than the scalar C++ loop." The fused C++ path is enabled by default despite being known slower.

---

## 7. Benchmark Gaps

### 7.1 Missing Coverage

| Missing Benchmark | Why It Matters |
|-------------------|----------------|
| `query_polygon` at scale | `bench_sphere_polygons.py` covers contains/intersects/area but not the O(Npix) query operation on large maps |
| `MOC.contains` with large point sets | Containment is a core operation; performance at scale is unmeasured |
| `SphericalMultiPolygon` operations | Intersection/contains for multi-polygon regions |
| `SkyMaskPipe.combine` | The dense materialization cost is unmeasured |
| Cross-device transfers | MPS→CPU→MPS roundtrip costs are not measured |
| Large-scale catalog crossmatch (100M+ sources) | `bench_catalog_crossmatch.py` covers only refinement stages |
| Memory profiling | Peak memory usage is not tracked in any benchmark |
| Heuristic thresholds | `TORCHSKY_TABLE_SCANNER_THRESHOLD`, `_PLANAR_PIP_CPP_MIN_OPS`, `_CENTER_CPP_P2W_THRESHOLDS` — described as "benchmark-backed" but no evidence of systematic threshold optimization |

### 7.2 Benchmark Quality Issues

- Many benchmarks are SMACS0723-specific (`bench_map_smacs0723_*`) which limits generalizability
- Benchmarks measure wall time but not memory — memory regressions can go undetected
- Most sphere benches go up to N=200K points but not 1M+
- No regression tests for memory usage

---

## 8. Dead Code & Unused Imports

### 8.1 Unused Imports

`__init__.py:8-10`:
```python
from typing import Any, Dict, List, Optional, Union, Iterator, Tuple, Sequence
```
`Iterator` and `Sequence` are imported but never used.

### 8.2 Placeholder Checksum Functions

`core.py:139-148`:
```python
@staticmethod
def verify_datasum(data: np.ndarray, expected_datasum: str) -> bool:
    return True

@staticmethod
def verify_checksum(header: Dict[str, Any], data: Optional[np.ndarray] = None) -> bool:
    return True
```
These are stubs that always return `True`. They should either be implemented or removed to avoid giving users false confidence.

### 8.3 Redundant Delegation Layer

`io/__init__.py:1-130` — every function is `from .. import X as _X; return _X(...)`. This adds indirection with no value.

### 8.4 Legacy FITS Dependency

`sphere/sparse.py:483-516` — `write_fits` / `read_fits` use `__import__("fitsio")`, violating the AGENTS.md principle of not reducing torchsky to a wrapper around legacy packages. If FITS I/O is needed, it should go through torchsky's own FITS layer.

### 8.5 Fallback After Fused C++ Failure

`wcs/core.py:1186-1189`:
```python
except TypeError:
    self._disable_fused_pixel_to_world = True
    return self._pixel_to_world_2d_core(x, y, float(origin))
```
After disabling the fused path, the instance state is permanently modified. If the TypeError was transient (e.g., a stale C++ extension during editable install), the fused path is permanently lost for the lifetime of the WCS object.

---

## 9. Recommendations

### Critical Priority

| # | Action | Files | Rationale |
|---|--------|-------|-----------|
| 1 | **Fix `_COMPLEX_DTYPE_MAP` set-vs-dict bug** | `table.py:3356` | Causes crash after `update_rows` is called |
| 2 | **Add device safety to `MOC.flatten()`** | `sphere/moc.py:297` | Crashes on non-CPU devices |
| 3 | **Split `__init__.py` into focused modules** | `__init__.py` | 2726-line file is unmaintainable |
| 4 | **Document or remove all heuristics with benchmark evidence** | `__init__.py:554-669`, `table.py:1104-1194` | Required by AGENTS.md Principle 1 |

### High Priority

| # | Action | Files | Rationale |
|---|--------|-------|-----------|
| 5 | **Vectorize `MOC.contains`** | `sphere/moc.py:139-155` | Replace Python loops with tensor operations |
| 6 | **Vectorize `_CoarseHealpixIndex.lookup`** | `catalogs/core.py:96-99` | Replace `.tolist()` + Python loop with `torch.repeat_interleave` |
| 7 | **Add HEALPix ring-range query** | `wcs/healpix.py:1433, 1530` | Avoid O(Npix) scanning — use O(√Npix) ring-range computation |
| 8 | **Vectorize Fourier feature generation** | `map/field.py:1065-1094` | Replace Python loops with `torch.logspace` + broadcasting |
| 9 | **Fix fused WCS default** | `wcs/core.py:162` | Disable `_enable_fused_wcs` by default or make it adaptive (comment says it's slower) |
| 10 | **Replace broad `except Exception`** | `__init__.py:1026-1031`, `table.py` | Catch specific exception types; log before fallback |
| 11 | **Remove dead code** | `core.py:139-148`, `io/__init__.py` | `ChecksumVerifier` stubs, redundant delegation layer |
| 12 | **Standardize type hints** | All files | Choose PEP 604 `X \| Y` or `Optional[X]` consistently |
| 13 | **Add return type annotations** | `__init__.py` | All public functions (`read()`, `write()`, `read_table()`) lack them |

### Medium Priority

| # | Action | Files | Rationale |
|---|--------|-------|-----------|
| 14 | **Replace ear-clipping triangulation** | `sphere/geom.py:416-478` | Use O(N log N) monotone polygon algorithm or C++ kernel |
| 15 | **Add MPS-native HEALPix kernels** | `wcs/healpix.py` | Avoid CPU roundtrips for Apple Silicon |
| 16 | **Vectorize `_build_coverage_map`** | `sphere/sparse.py:191-195` | Use scatter operations instead of Python loop |
| 17 | **Precompute neighbor shells in crossmatch** | `catalogs/core.py:212-229` | Avoid rebuilding per-pixel |
| 18 | **Add memory benchmarks** | `benchmarks/` | Track peak memory in the benchmark suite |
| 19 | **Consolidate namespace structure** | `geom/__init__.py`, `coords/__init__.py` | Clarify whether users should use `torchsky.geom` or `torchsky.sphere` |
| 20 | **Make `sphere/spectral.py` caches device-aware** | `sphere/spectral.py` | Or document that they are CPU-only |
| 21 | **Replace global `_STRICT_MODE`** | `sphere/compat.py:31-59` | Use context-local or instance-level state |
| 22 | **Add tests for edge cases** | `tests/` | Empty catalogs, RA wraparound in crossmatch, malformed FITS headers |
| 23 | **Remove `fitsio` dependency** | `sphere/sparse.py:483-516` | Use torchsky's own FITS layer |

### Low Priority

| # | Action | Files | Rationale |
|---|--------|-------|-----------|
| 24 | **Add CUDA kernels for HEALPix ang2pix/pix2ang** | `wcs/healpix.py` | Currently all GPU paths fall back to CPU |
| 25 | **Optimize `_gnomonic_project` to stay on-device** | `sphere/geom.py:369` | Avoid unnecessary `.cpu().numpy()` |
| 26 | **Add benchmark for large-N sphere geometry (1M+ points)** | `benchmarks/` | Current benches stop at 200K |
| 27 | **Systematic threshold optimization** | Multiple | Run threshold sweeps and commit optimal values with benchmark evidence |
| 28 | **Add `__slots__` to frequently-created dataclasses** | `catalogs/core.py` | e.g., `CrossmatchResult` |
| 29 | **Audit all `hasattr(cpp, ...)` checks** | Multiple | Ensure fallback paths are tested |
| 30 | **Add docstrings to internal helper functions** | Multiple | Many `_`-prefixed functions have no documentation |

---

## Appendix A: Test Suite Assessment

- **86 test files** — good coverage breadth
- Tests use `torch.testing.assert_close` appropriately for numerical comparisons
- `test_catalogs.py` tests both the public API and internal functions — good practice
- **Missing:** No tests for heuristic functions (`_should_use_cold_nommap`, `_resolve_image_mmap`) despite AGENTS.md requiring benchmark/regression tests for heuristics
- **Missing:** No tests for `_COMPLEX_DTYPE_MAP` shadowing bug
- Many tests import internal functions (e.g., `catalogs_core._crossmatch_sky_coarse_indexed`) which creates tight coupling to implementation details

## Appendix B: Benchmark Suite Assessment

- **75 benchmark files** — extensive coverage
- Benchmarks compare against healpy where available — good practice
- `bench_catalog_crossmatch.py` properly measures both algorithms and reports candidate-reduction effects — good, per AGENTS.md
- **Issue:** Many benchmarks are SMACS0723-specific which limits generalizability
- **Issue:** No benchmark for heuristic thresholds in `_should_use_cold_nommap` or `_resolve_image_mmap`

## Appendix C: File Size Summary

| File | Lines | Notes |
|------|-------|-------|
| `src/torchsky/__init__.py` | 2726 | God module — must be split |
| `src/torchsky/table.py` | 3508 | Contains I/O, mutation, schema, integration |
| `src/torchsky/sphere/spectral.py` | 4141 | Full spherical harmonic transform + 20+ caches |
| `src/torchsky/wcs/core.py` | ~1600 | WCS setup, pixel/world transforms, fused paths |
| `src/torchsky/sphere/geom.py` | ~1600 | Spherical geometry, polygons, triangulation |
| `src/torchsky/map/image_plane.py` | ~1300 | Mapmaking, drizzle, projection |
| `src/torchsky/wcs/healpix.py` | ~1600 | HEALPix operations, queries, nside utilities |
