# Deep audit: mapping, mosaicking, and INR (torchsky vs SWarp, Drizzle, Montage, reproject, radio/mm)

**Scope:** `torchsky.map` (image-plane coaddition, spectral cubes, products), `torchsky.map.field` (continuous / INR-style fields), related benchmarks and tests.  
**Goal:** Clarity on **parity** with common optical/IR mosaicking stacks and **gaps** for mm/radio workflows, plus a concrete plan for **tests, real-data coverage, and benchmarks**.

---

## 1. What torchsky implements today

### 1.1 Image-plane mapmaking (`map/image_plane.py`)

- **Inverse mapping (target ← source):** World positions on the output grid are mapped into source pixel space; resampling uses `torch.nn.functional.grid_sample` for most kernels, with optional **windowed** evaluation for large TAN fields.
- **Resampling kernels (public):** `nearest`, `bilinear`, `bicubic`, `lanczos3`, `lanczos5`, `mitchell_netravali`, `catmull_rom`, `fft_shift`, `fft_diagonal_affine`, `fft_linear_affine`, `periodic_linear_affine`, `drizzle_point`, `drizzle_square`.
- **HEALPix on TAN grids (helper):** `sample_healpix_map_on_tan_header` samples a HEALPix map onto a target header using `torchsky.wcs.healpix.get_interp_val` (no `reproject` dependency).
- **Drizzle path:** **Default** coaddition uses the native torch **forward-drizzle** accumulator (CUDA/MPS/differentiable). The external Python **`drizzle`** package is **opt-in** for regression against DrizzlePac-style behavior: set `TORCHSKY_USE_EXTERNAL_DRIZZLE=1` (see `map/image_plane.py`).
- **Value semantics:** `surface_brightness` vs `flux` (with pixel solid-angle scaling where required).
- **Stacking:** `weighted_mean`, `mean`, `median` (median not supported everywhere, e.g. spectral cube path documents limits).
- **Weights / masks / per-observation photometry:** `ImageObservation` supports weights, weight maps, masks, photometric scale, background offset, and **spectral** metadata (wavelength, bandpass bounds, Gaussian transmission approximation).
- **Footprints:** Optional HEALPix **MOC** (`footprint_order`) for coverage bookkeeping.
- **Spectral dimension:** `coadd_image_plane_cube`, `coadd_image_plane_multiband`, planning helpers (`plan_image_plane_cube_target`, …) integrate with `spectral` / `DataCube` for multi-band and cube products.

### 1.2 Product layer (`map/product.py`)

- `MapProductSpec` / `stack_map_product` orchestrate coadds against a declared product (bands, target header, semantics).
- **Uncertainty / evaluation hooks:** inverse-variance estimate, frontier-region metrics, product evaluation utilities (see `evaluate_map_*`).

### 1.3 INR / continuous fields (`map/field/` package)

- Large **experimental** surface: SIREN/WIRE-style implicit fields, Fourier MLP, spectral basis fields, **sky–λ** factorizations (planes, K-planes, feature fields, atlas variants), **hash-grid** spatial encoders, **sky-encoder** bandpass aggregators, operator-constrained features, bandpass-conditioned decoders, etc.
- **`fit_field_model`** trains against `FieldObservation` tensors (images + WCS headers + optional weights/masks/spectral metadata).
- **Rendering:** `render_field_observation`, `render_field_cube`, integration with `MapProductSpec` via `render_field_product`.

### 1.4 WCS dependency

- All mapmaking is **header/WCS-driven** via `torchsky.WCS` (SIP, TPV, etc., as supported elsewhere in the package). Parity with legacy tools is limited by **matching grid, kernel, flux convention, and mask/weight semantics**, not by “same FITS reader.”

---

## 2. Comparator matrix (functionality, not “who is faster”)

| Capability | SWarp | Montage | `drizzle` (std) | DrizzlePac | `reproject` / `dfreproject` | torchsky.map |
|------------|-------|---------|-----------------|------------|------------------------------|--------------|
| Global mosaics, arbitrary projection | Yes | Yes | Partial (typ. small field) | Yes (HST-style) | Yes (interp / GPU df) | Yes (target header grid) |
| Lanczos / cubic family | Yes | Limited / module-dependent | Drizzle kernels ≠ Lanczos | Tweakreg/ AstroDrizzle stack | Interp orders | Lanczos3/5, Mitchell-Netravali, Catmull-Rom, bicubic |
| Flux / SB preservation options | Yes (config) | Yes (reprojection math) | Drizzle-specific | Extensive | Depends on mode | `surface_brightness` / `flux` |
| Median / min-max stack | Extensible | mAdd / mAddExec | No | Comb type | No (single regrid) | `median` / `mean` / `weighted_mean` |
| Weight maps | Yes | Yes | Yes | Yes | User supplies arrays | Per-obs weights + weight HDU |
| Bad-pixel masks | Yes | Yes | Yes | Yes | User | mask HDU |
| True drizzle drop size / pixfrac | N/A | N/A | Yes | Yes | N/A | Yes (native torch drizzle; external `drizzle` pkg opt-in) |
| DFT / spherical harmonic reprojection | No | No | No | No | **`reproject` has spherical FFT path (separate from interp)** | **Not implemented as first-class** |
| Pipeline orchestration (many tiles) | CLI | Montage grid | Library | Pipeline | Library | **Library API** (user loops) |
| Differentiable stack | No | No | No | No | No | **Yes (PyTorch path)** |

**Already wired in-tree for benchmarks:** `bench_map_compat.py` and `bench_map_smacs0723_stackers.py` compare against **`reproject_interp`**, **`dfreproject`**, **`drizzle`**, **MontagePy**, **SWarp**, and **casutools imstack** where semantics align; **drizzlepac** is explicitly noted as optional / often **SKIPPED** with reasons.

---

## 3. mm / radio / interferometry gaps (high level)

The current design is **image-plane, FITS-WCS-first**, similar to **SWarp / Montage / optical drizzle**. It does **not** yet model:

- **Interferometric gridders:** dirty / clean cubes, `tclean`, WSClean-style facets, **primary-beam multiplication per visibility** or per Stokes.
- **PB correction and mosaics in CASA sense:** per-field sensitivity patterns, noise equalization across PB edges, **joint mosaicking** with **direction-dependent beams**.
- **Full Stokes / polarization** bookkeeping in map products (beyond treating extra axes as generic tensor channels if user supplies them).
- **Spectral frames** common in radio: **LSR/BARY**, **velocity** axes, **RESTFRQ/RESTWAV** consistency across mosaics (partially depends on `spectral` + header discipline, not a CASA-style pipeline).
- **Time-domain** mosaics (epochs, proper motion grids).

**Recommendation:** Treat **CASA / tclean / WSClean** outputs as **FITS inputs** for **post-imaging** mosaicking first (same as optical); add **PB-aware weighting** as a **second phase** (per-pixel effective area from PB cube or 2D maps), then consider **deeper** interferometry only if torchsky’s mission explicitly includes synthesis imaging.

---

## 4. INR / field-model gaps vs classical mappers

| Topic | Classical stack | torchsky.map.field |
|-------|------------------|---------------------|
| Objective | Chi-square to data, regularization optional | Many architectures; Charbonnier/Huber/NLL, curricula, MC spectral sampling |
| PSF | Convolution or matched filter per tool | `psf_sigma_pix` hint; **no full 2D PSF convolution in the mapmaker** |
| Joint calibration | Photometric matching, background models | `photometric_scale`, `background_offset`; **no full solver** |
| Uncertainty on field | Rare | Possible via ensemble / NLL; **not standardized like `weight` maps on coadd** |
| Interpretability | Known kernel | Black-box INR; **basis / encoder hybrids** are the migration path |

**Risk:** INR benchmarks can “win” on RMSE on a **fixed grid** while **failing** flux conservation or edge behavior vs drizzle/SWarp unless tests enforce **photometric** and **support** contracts.

---

## 5. Existing tests (inventory)

| Area | Tests |
|------|--------|
| Image-plane kernels, coadd, drizzle bypass | `tests/test_map_image_plane.py` (large) |
| Product / uncertainty API | `tests/test_map_product.py` |
| Field models (unit) | `tests/test_map_field.py`, `tests/test_map_field_fit.py` |
| Ground-truth synthetic truth | `tests/test_map_ground_truth.py` |
| Multifield / cube truth | `tests/test_map_multifield_truth.py` |
| Real JWST SMACS J0723 (conditional on data) | `tests/test_map_smacs0723_*.py` |
| Compat bench wiring | `tests/test_map_compat_bench.py` (Montage header template, suite parsers, etc.) |
| `reproject` / `dfreproject` interp parity (small TAN) | `tests/test_map_reproject_fft_parity.py`, `tests/test_map_dfreproject_parity.py` (need **`torchsky[bench]`**) |
| UPdec-Webb | `benchmarks/bench_updec_webb_real.py` (bench; optional CI) |

**Gaps in tests:**

- **Radio/mm FITS:** synthetic **Stokes cube** + **RESTFRQ** / **SPECSYS** header round-trip is covered in **`tests/test_map_radio_smoke.py`**; a vendored real ALMA/SKA cutout fixture is still optional.
- **DFT / adaptive reproject:** `dfreproject.calculate_reprojection` bilinear parity is in **`tests/test_map_dfreproject_parity.py`**; **`reproject_adaptive`** finiteness smoke is in **`tests/test_map_reproject_fft_parity.py`**. Distinct spherical **FFT** backends in `reproject` (if any) may still need a dedicated gate when the public API is stable.
- **End-to-end SWarp:** optional **`tests/test_map_swarp_optional_parity.py`** compares SWarp **BILINEAR** to torchsky on a tiny shifted pair when the **`SWarp`** binary is installed.
- **`ImageObservation.hdu`** defaults to **`None`** (aligned with auto-payload reads). Optional **`primary_beam_path`** scales per-pixel weights by a primary-beam map (see `map/image_plane.py`).

---

## 6. Existing benchmarks (inventory)

Documented in `docs/benchmarks.md` (map section). Highlights:

- **Synthetic:** `bench_map_image_plane.py`, `bench_map_ground_truth.py`, `bench_map_multifield_truth.py`, `bench_map_field_fit.py`, projection mosaics / TAN-SIP suites.
- **Compat / competitive:** `bench_map_compat.py`, `run_ground_truth_competitive_large_union.py`, `bench_map_tan_sip_suite.py`, `bench_map_projection_mosaic_suite.py`.
- **Real optical/IR:** `bench_map_smacs0723_*`, `bench_updec_webb_real.py`.
- **Diagnostics:** `diagnose_map_large_union_*.py`, `diagnose_map_projection_tail.py`, `bench_map_boundary_support.py`.

**Gaps in benchmarks:**

- **Timed regression gates** in default CI for map (many benches are **optional envs**: `sphere-bench`, `map-compat`).
- **reproject FFT / spherical** row next to `reproject_interp` on **same synthetic truth**.
- **Radio/mm:** small **public** cube + PB (e.g. **ALMA science archive** tiny cutout, or **VLA** Sky Survey patch) — **throughput + max abs error** vs SWarp or Montage on **TP map**.
- **INR:** baseline JSON gates exist for some field-fit benches; **not** uniformly tied to **flux** and **support** metrics on real data.

---

## 7. Recommendations (prioritized)

### P0 — Correctness and “defaults that match the ecosystem”

1. ~~**Align `ImageObservation.hdu` default**~~ **Done:** `None` / auto payload.
2. **Document** kernel + semantics parity in **`docs/mapping_parity.md`** (user-facing summary).
3. **Parity vs `reproject`:** `tests/test_map_reproject_fft_parity.py` locks **bilinear** agreement with `reproject_interp` on small synthetic grids when `reproject` is installed (declared under **`torchsky[bench]`**, not `torchsky[test]`). Public `reproject` 0.19 does not expose a separate FFT reprojection API; HEALPix helpers are interpolation-based (optional skip in the same test module).

### P1 — Real data and radio/mm entry

4. **Curate one tiny public radio/mm FITS mosaic fixture** (TP or primary-beam corrected cube header) + **pytest** that runs `coadd_image_plane` vs **SWarp** or **Montage** on a **2–4 file** subset (skip if tools missing).
5. **PB-aware weights:** optional **`primary_beam_path`** on `ImageObservation` (**implemented**); synthetic coverage in `tests/test_map_radio_smoke.py`.

### P2 — Performance on par with production mappers

6. **Establish wall-time baselines** on fixed hardware: `torchsky` vs **dfreproject** (GPU) vs **SWarp** for **2048²**, **10 inputs**, Lanczos3 — record in `docs/benchmarks.md`.
7. **Large-field memory:** already have windowing / chunking; add **benchmark row** for **10k×10k target** with **streaming** observation loop (memory peak).

### P3 — INR credibility

8. **Standard evaluation bundle** on real SMACS crop: RMSE, **flux in aperture**, **power spectrum ratio**, **support-masked** metrics — same for **drizzle** reference.
9. **Shrink public field-model zoo** in docs to **3 recommended** families (e.g. bilinear coadd baseline, hash-grid spectral, sky-encoder bandpass) with **bench row** each.

---

## 8. Suggested new artifacts (concrete)

| Artifact | Purpose |
|----------|---------|
| `tests/test_map_reproject_fft_parity.py` | `reproject_interp` bilinear vs torchsky on small synthetic (+ optional HEALPix skip) |
| `tests/test_map_radio_smoke.py` | Synthetic radio-style FITS; PB weighting + coadd smoke |
| `benchmarks/bench_map_radio_tp_smoke.py` | Wall-time + max diff vs `reproject_interp` on synthetic TP pair |
| `benchmarks/baselines/map_radio_tp_smoke.json` | Loose baseline (`skip_time_regression` on timings) |
| `docs/mapping_parity.md` | User-facing kernel / semantics parity summary |

---

## 9. Summary

torchsky’s **map** stack is already a **serious** differentiable image-plane mosaic engine with **strong** optical/IR benchmark coverage (including **real JWST** paths) and **explicit** comparators for **SWarp, Montage, drizzle, reproject, dfreproject**. **Gaps** for “on par with everything popular” are mainly: **DFT/spherical reprojection parity**, **radio/interferometry-specific physics (PB, synthesis)**, **tighter real-data + timing regression gates**, and **INR evaluation tied to photometric contracts**. Addressing **P0–P1** above yields the best return before expanding scope into CASA-like imaging.
