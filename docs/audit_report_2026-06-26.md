# torchsky Repository Audit Report

**Date:** 2026-06-26
**Auditor:** opencode
**Scope:** All ~127 source files, ~124 test modules, ~103 benchmark scripts, C++ (18 files), build/CI/packaging
**Focus:** Correctness, scientific logic, performance, API design

---

## Executive Summary

| Dimension | Score | Key Findings |
|-----------|-------|-------------|
| **Architecture** | B | Strong design intent; CFITSIO linking and in-tree FITS I/O contradict AGENTS.md |
| **WCS/Geometry** | A | Excellent astropy parity, all projections validated forward+inverse, differentiable |
| **HEALPix/Sphere** | A | Exhaustive healpy parity (300+ tests), correct indexing, device-native |
| **Map Operators** | B | Well-architected but per-pixel polygon loops block survey-scale viability |
| **Field/INR Models** | B | Novel and sophisticated; GaussianBasisField coord_dim bug; SIREN init correct |
| **C++ Code** | B- | Single TU build, CFITSIO contradiction, stale cache.h, unsafe std::thread |
| **Test Quality** | A- | Exceptional parity and policy enforcement; gaps in performance gates, adjoints |
| **Benchmarks** | B+ | Well-structured reproducible baselines; missing map throughput gates |
| **CI/CD** | C+ | Placeholder jobs, no macOS/GPU runners, no coverage tracking |
| **API Cleanliness** | A- | Lazy imports, clean namespaces, well-typed; map `__init__` is 479 lines (verbose) |

---

## 1. Architectural Contradictions

### 1.1 CFITSIO linked in C++ extension — CRITICAL

`src/torchsky/cpp_src/CMakeLists.txt:406-415` links `cfitsio`, `pixi.toml:34-38` lists `cfitsio >=4` as a runtime dependency.

**AGENTS.md states:** *"torchsky C++ must not link CFITSIO"* and *"FITS parser, HDU, checksum, compression, file-cache, and FITS-table implementations belong in torchfits."*

The C++ extension compiles ~11,000 lines of FITS I/O (`fits.cpp`, `table.cpp`, `compression.cpp`) with CFITSIO statically linked. This creates potential symbol conflicts if `torchfits` (also CFITSIO-dependent) is loaded in the same process.

**Resolution:** Remove CFITSIO entirely from torchsky. The C++ extension must not depend on `cfitsio` at build, host, or run time. All FITS I/O functionality moves to `torchfits` (sibling package at `~/src/torchfits`). torchsky will depend on `torchfits` at runtime.

### 1.2 FITS I/O implementation in-tree — ACTIONABLE

`src/torchsky/table.py` (3759 lines), `src/torchsky/hdu.py` (2256 lines), `src/torchsky/_io_engine/` (13 files), `src/torchsky/_table_engine/`, `src/torchsky/io/fits.py` (330 lines) constitute a full FITS I/O implementation. AGENTS.md says this should live in `torchfits`. These are not stubs — they are active, production implementations.

**Resolution:** Extract all FITS I/O code into `torchfits` (already in development at `~/src/torchfits`). torchsky imports from `torchfits` at runtime. Remove all FITS-related code from torchsky source. Update `AGENTS.md` to reflect the new dependency.

### 1.3 Ecosystem packages in default workspace — MAJOR

`pixi.toml:58-62` lists `healpy`, `hpgeom`, `healsparse`, `astropy-healpix`, `spherical-geometry` in workspace `[dependencies]` (always installed for dev). AGENTS.md says they belong only in feature-gated environments. They are already duplicated in `[feature.sphere-bench.dependencies]` (lines 235-243).

**Resolution:** Remove these from `[dependencies]`; they are already present in `[feature.sphere-bench.dependencies]`.

---

## 2. Correctness Issues

### 2.1 Scientific Logic

| Severity | File | Line | Issue |
|----------|------|------|-------|
| **Critical** | `wcs/core.py` | 4050 | Inverse spherical rotation had sign error in `_spherical_rotation` (fixed in this session) |
| **Critical** | `wcs/core.py` | 4475+ | Missing `-` on `xi` in SIN/ARC/ZEA/STG analytic inverses (fixed in this session) |
| **Major** | `map/operators/geometry_drizzle.py` | 553-559 | `flux_scale = (A_t/A_s)²` applied to both numer and weight — cancels to dead code, mathematically misleading |
| **Major** | `sphere/spectral.py` | 244-246 | MC weights reset to uniform `1/N` after sorting, discarding importance sampling density |
| **Major** | `torchsky/interop.py` | 46,136 | `to_pandas`/`to_arrow` call `.numpy()` without `.cpu()` — crashes on CUDA/MPS tensors |
| **Major** | `torchsky/spectral.py` | 672-673 | `SpectralReader` hardcodes `unit="Angstrom"`, ignores `CUNIT1`/`CTYPE1` in header |
| **Medium** | `torchsky/map/exposure_pose.py` | 63,79 | `cos(dec0)` tangent-plane approximation undocumented for large fields or polar regions |
| **Medium** | `torchsky/map/healpix_spectral.py` | 323 | Adjoint Legendre `P_lm` convention unvalidated against upstream; must be cross-checked |
| **Minor** | `torchsky/map/image_plane.py` | 1450 | NaN weights pass validation — IEEE 754: `NaN <= 0.0` is `False` |

### 2.2 Code Correctness

| Severity | File | Line | Issue |
|----------|------|------|-------|
| **Critical** | `map/image_plane.py` | 407-414 | `_tpv_candidate_mask` — dead function, empty body, returns `None` |
| **Major** | `cpp_src/table.h` | 29 | `int num_rows` (header) vs `long num_rows` (impl) — overflows at >2^31 rows |
| **Major** | `cpp_src/compression.cpp` | 262 | `throw` inside `std::thread` worker triggers `std::terminate`, process crash |
| **Major** | `torchsky/map/healpix_image_plane.py` | 50 | `.detach()` kills autograd on every HEALPix→image call — blocks differentiable pipelines |
| **Major** | `torchsky/map/solvers.py` | 2175,2270,2363 | Three dead `isinstance(scene, Tensor)` branches (always reached by earlier branch) |
| **Medium** | `torchsky/map/windowing.py` | 217-226 | try-except circular-import workaround is dead code (locals defined before the block) |
| **Medium** | `cpp_src/cache.h` | entire | Stale `CacheManager` class — `cache.cpp` uses completely different `UnifiedCache` API |
| **Medium** | `torchsky/map/scenes.py` | 139-167 | `HealpixMapScene` rendering uses `.detach()` — non-differentiable |
| **Low** | `cpp_src/fits.cpp` | 2778-2779 | Duplicate `validate_fits_filename` call |
| **Low** | `cpp_src/table.cpp` | 2456 | `const_cast<char*>` on `std::string::c_str()` — technically UB if CFITSIO ever mutates |
| **Low** | `torchsky/map/spectral_nufft.py` | entire | Named "nufft" but is just FFT upsample + bilinear sample — not a real NUFFT (Kaiser-Bessel or min-max gridded) |

---

## 3. Performance Issues

### 3.1 Survey-Scale Blockers

| Severity | File | Issue |
|----------|------|-------|
| **Critical** | `map/operators/geometry_conservative.py:106-141` | Per-pixel Python polygon clipping loops — `O(N × M)` for each CCD |
| **Critical** | `map/operators/geometry_drizzle.py:653-686` | Same per-pixel polygon clipping in the drizzle path |
| **Major** | `map/operators/pixel_area.py:22` | Full-resolution meshgrid for pixel area — 10k×10k → 1.6 GB for two maps |
| **Major** | `map/healpix_image_plane.py:65-68` | Full-grid allocation on GPU — 10k×10k image uses 3.2+ GB |
| **Major** | `map/solvers.py:266-294` | `estimate_normal_operator_diagonal` probes each of 4096 basis vectors individually |
| **Major** | `cpp_src/CMakeLists.txt:376-378` | Single translation unit — no parallel compilation, breaks tooling |
| **Major** | `map/operators/geometry_drizzle.py:161-210` | Golden-section pixmap inversion runs up to 50 iterations per source-pixel corner |

### 3.2 Suboptimal Patterns

| File | Issue |
|------|-------|
| `map/operators/background.py:86-95` | Per-term Python loop for polynomial evaluation |
| `map/operators/resampling.py:109-113` | `index_select` creates large intermediate tensors |
| `map/spectral_nonuniform.py` | `O(n_chunks · H_work · W_work)` — correct but slow at survey scale |
| `map/solvers.py:1947` | Python `for` loop over 4096 `ell` values for TEB precision factors |
| `map/fft_affine_ldu.py` | `_per_col_vertical_shift_fft` does `W` FFTs of length `H` instead of one FFT over the image |

---

## 4. API Design

### 4.1 Strengths
- **Lazy namespace loading** — `import torchsky` does not eagerly load numpy, torch, or any domain module. Cold import time ~0.01s.
- **Consistent `__getattr__` pattern** in `map/__init__.py`, `wcs/__init__.py`, `sphere/__init__.py`
- **Clean root API** — small set of well-named functions (`read`, `write`, `open`, `get_header`, `get_wcs`)
- **`TargetRenderSpec`** — clean cross-module contract
- **`APISurface` stability tiers** (`surfaces.py`) — clear API contracts for downstream consumers
- **`ObservationOperator` class hierarchy** with dataclass slots — clean, extensible
- **Adjoint via autograd** (`inverse.py:121`) — elegant, avoids maintaining separate adjoint implementations

### 4.2 Weaknesses
- **`map/__init__.py`**: 479 lines, 118 `_EXPORTS` entries — too large a facade. Consider splitting.
- **`scene/__init__.py`**: TypeAliases at lines 63-67 (`Observation`, `Scene`, `ProductSpec`, `Target`) add abstraction with no added clarity.
- **`spectral_nufft.py`**: Named "nufft" but is not a real NUFFT — over-promises.
- **`schema.py`-like patterns**: `frame.py:63-99` infers `stype` from dtype alone — int32 columns are always treated as numerical, never categorical. Schema should be required for production use.
- **`map/surfaces.py`**: `BandMetadataTable` and `ImageObservation` exposure in map facade creates confusion with the clean `scene` module.

---

## 5. Build System, CI, Packaging

### 5.1 Critical
- **CFITSIO in C++ extension** (see §1.1)
- **`[package.run-dependencies]`** includes `cfitsio >=4` and `psutil >=5.0` — contradicts the lean runtime principle (AGENTS.md: only `torch` and `numpy`)

### 5.2 Major
- **`pyrightconfig.json`** — covers only `header_parser.py`. Zero type coverage for `wcs/`, `sphere/`, `map/`, `field/`, etc.
- **Three placeholder CI jobs**: `memory-safety`, `performance-benchmark`, `documentation` — all just `echo` statements
- **Single OS/Python in CI** (linux-64, Python 3.12 only) — no macOS or 3.13/3.14 testing despite claims
- **No GPU runners** — CUDA and MPS tests always skipped
- **No coverage tracking** — no `pytest-cov` or Codecov in CI

### 5.3 Minor
- `[dependency-groups]` (PEP 735) duplicates `[project.optional-dependencies]` with slightly different version pins — drift risk
- `memory_tests.yml.backup` — stale file referencing Python 3.9, old APIs
- No `aarch64` Linux wheels (`CIBW_ARCHS_LINUX` only `x86_64`)
- `install.strip = false` — release wheels should be stripped
- ZLIB handling in `cpp_src/CMakeLists.txt:205-231` creates symlinks to work around path mismatches — fragile

---

## 6. Tests

### 6.1 Excellent
- **Cold import tests** (`test_cold_import.py`) — subprocess-based isolation, catches dependency creep before release
- **AST-based policy enforcement** (`test_no_external_fits_backends.py`) — parses every `.py` for forbidden top-level imports
- **Parity testing** (`test_healpix_parity.py`, 727 lines) — exhaustive healpy parity at multiple nsides and sample sizes
- **WCS parity** (`test_wcs_parity.py`) — 8 projections × 2 origins × forward/inverse/stacked/compiled, astropy at 5e-9° tolerance
- **Adjoint verification** (`test_map_adjoints.py`) — dot-product test `<A(x), y> == <x, A^T(y)>` at 1e-4 tolerance
- **Map ground truth** (`test_map_ground_truth.py`, 507 lines) — 8 reconstruction cases with flux conservation validation
- **Cross-device** — CPU/CUDA/MPS matching tests across HEALPix, WCS, sphere

### 6.2 Gaps
- **No adjoint tests** for spherical reprojection or FFT operators
- **No gradient tests** for WCS transforms (differentiability claimed but `torch.autograd.gradcheck` never called)
- **Performance gates** only cover WCS/HEALPix (4 gates) — missing map coadd throughput
- **No property-based testing** (`hypothesis` is a dependency but unused)
- **No `@pytest.mark.slow`** categorization — all tests run identically
- **No error-propagation tests** (what happens when WCS inversion fails? observation has no overlap?)

### 6.3 Benchmarks Gaps
- No map coadd throughput benchmark
- No sphere spectral transform throughput (map2alm/alm2map/synfast)
- No aggregate GPU throughput suite
- No trend tracking / historical archive

---

## 7. Key Positive Observations

1. **Lazy import discipline** — `import torchsky` takes ~0.01s, does not pull in numpy, torch, or any domain module. This is rare and well-executed.

2. **AST-parsed import scanning** — programmatic architectural enforcement at the AST level is an exemplary practice rarely seen in astronomy software.

3. **Subprocess isolation tests** — catches silent dependency creep before it reaches users or CI.

4. **Autograd-based adjoints** — avoids maintaining separate adjoint implementations for each operator. Clever and correct for linear operators.

5. **Constrained learned operators** (`learned_kernel.py`) — softplus + unit-sum + convex mixture for PSF, identity-initialized with gate penalty for residual ops, exponential gain for radiometry. Mathematically principled construction.

6. **Quadrature pixel integration** (`windowing.py`) — principled flux-preserving reprojection without per-pixel polygon overhead.

7. **Reproducible benchmarks** — all scripts accept `--seed`, `--runs`, `--json-out`, use median timing with warmup, and record version info. `PHASE0_WCS_SPHERE.md` documents the full reproducibility protocol.

8. **Device-native design** — core primitives use `torch` tensors throughout, enabling trivial CPU/CUDA/MPS switching.

---

## 8. File-by-File Volume

| Module | Files | Est. Lines | Grade | Key Issue |
|--------|-------|------------|-------|-----------|
| `wcs/` | 13 | ~8,000 | A- | SIP/TPV analytic inverses had sign bugs (fixed) |
| `sphere/` | 12 | ~7,000 | A | Excellent healpy parity, correct indexing |
| `map/operators/` | 24 | ~7,000 | B | Per-pixel Python loops block survey scale |
| `map/field/` | 19 | ~10,000 | B | GaussianBasisField coord_dim bug |
| `map/` (other) | 14 | ~8,000 | B+ | `.detach()` in HEALPix sampler |
| `cpp_src/` | 18 | ~11,000 | B- | Single TU, CFITSIO, stale cache.h |
| `_io_engine/` | 13 | ~4,000 | B | Architecture contradiction (in-tree FITS) |
| `tests/` | 124 | ~50,000 | A- | Exceptional parity; missing perf gates |
| `benchmarks/` | 73 | ~40,000 | B+ | Reproducible but narrow scope |

**Total source:** ~110,000 lines across ~280 files.
