# Mapmaking parity: torchsky vs classical stacks

This note summarizes **when** `torchsky.map` image-plane coaddition is expected to agree with common optical/IR tools (SWarp, Montage, `reproject`, DrizzlePac-style drizzle), and where it **intentionally differs**. Exact agreement requires matching **grid**, **kernel**, **flux vs surface-brightness semantics**, and **weight maps**.

## Resampling kernels

| torchsky `MapResamplingKernel` | Typical classical analogue | Notes |
|--------------------------------|----------------------------|--------|
| `nearest` | Nearest-neighbor regrid | Fast; not flux-conserving. |
| `bilinear` | `reproject` bilinear, many SWarp/Montage linear modes | Good default for smooth maps; see tests vs `reproject_interp`. |
| `bicubic`, `catmull_rom`, `mitchell_netravali` | Higher-order interpolation in `reproject` / SWarp cubic family | Not identical polynomial to every implementation; compare on synthetic truth. |
| `lanczos3`, `lanczos5` | Lanczos-style mosaickers | Implemented via `grid_sample`; edge handling differs from all legacy stacks. |
| `fft_shift` | Band-limited subpixel shift (FFT) | **Restricted:** same `NAXIS` as source, linear WCS, **pure pixel translation** between source and target (probe-validated). **Periodic** boundaries; compare interiors to `bilinear` on CRPIX-only offsets. |
| `fft_diagonal_affine` | Separable diagonal scale+shift (FFT upsample + periodic wrap) | **Restricted:** same shape as `fft_shift`, linear WCS, target→source map **separable** as `src_x = s_x·x + b_x`, `src_y = s_y·y + b_y` in pixel indices (5×5 probe fit, cross-terms rejected). **Periodic** boundaries. |
| `fft_linear_affine` | Mixed **FFT column-shear** + **periodic bilinear** upper-triangular map + **FFT translation** (LDU pivot on `a00`) | **Restricted:** same shape / linear frame as other FFT kernels. **Not** a full 2D band-limited NUFFT; use `periodic_linear_affine` if the LDU pivot is degenerate or you need a single torus-bilinear sample of the full affine. |
| `periodic_linear_affine` | Full 2×2 linear map + translation on a **torus** (bilinear, not band-limited FFT) | **Restricted:** same shape / linear frame as the FFT kernels. Fits `src = A·(jx, iy) + b` from WCS probes; use for **rotated CD** targets where diagonal FFT paths reject. |
| `drizzle_point`, `drizzle_square` | Drizzle / AstroDrizzle drop kernels | **Default:** native torch forward accumulation (CUDA/MPS/differentiable). The Python **`drizzle`** package is **opt-in** for regression only: set `TORCHSKY_USE_EXTERNAL_DRIZZLE=1`. |

## Value semantics

- **`surface_brightness`:** Stack is set up so weighted means approximate **per solid-angle** consistency when area scaling is applicable (see `coadd_image_plane` and `_should_apply_surface_brightness_area_scaling`).
- **`flux`:** Per-pixel areas enter the numerator/denominator construction so stacked units follow **flux density × pixel area** conventions on the target grid.

Mismatch with SWarp/Montage almost always traces to **different SB/flux convention** or **different pixel-area treatment**, not to FITS I/O.

## Differentiable execution

All public tensor paths run in **PyTorch** on CPU, CUDA, or MPS (subject to kernel support). torchsky does **not** require SWarp, Montage, or external drizzle for core mapmaking.

## INR / continuous fields

Field models live under `torchsky.map.field` (package: `map/field/`, with `core`, `types`, `registry`). They are **experimental** and orthogonal to classical coadd semantics; validate against chi-square or product metrics rather than assuming identity with SWarp outputs.

## Further reading

- Comparator benchmarks: `benchmarks/bench_map_compat.py`, `docs/benchmarks.md`
- Mapping / INR audit: `docs/AUDIT_MAPPING_INR.md`

## Developers (editable installs)

Scikit-build editable installs record a static module→path map in `_torchsky_editable.py`. After **moving or renaming** Python modules (for example splitting `map/field.py` into a `map/field/` package), run **`uv pip install -e .`** (or your equivalent) so imports resolve to the new layout.

**Optional comparators:** `reproject`, **`dfreproject`**, the Python **`drizzle`** package, and other legacy stack clients are pulled in by **`torchsky[bench]`** for benchmarks and external comparisons, not by **`torchsky[test]`**. Modules are skipped when the dependency is absent: `test_map_reproject_fft_parity` (`reproject`), `test_map_dfreproject_parity` (`dfreproject`), `test_map_drizzle_external_parity` (`drizzle`).

## Gap-closure roadmap (in-repo)

High-level plan for closer parity with **SWarp**, **Montage**, **DFT/FFT/SH reprojection**, and **drizzle** comparators. **Background matching** and full **DrizzlePac** orchestration stay out of scope here.

### Phase 0 — Comparator contracts

- Keep this file and `docs/AUDIT_MAPPING_INR.md` aligned on **grid**, **kernel**, **flux vs SB**, **weights**, and **edge policy**.
- Track `SKIPPED` rows in `benchmarks/bench_map_compat.py` with explicit reasons.

### Phase 1 — SWarp-class semantics

- Expand **synthetic** tests for **subpixel shifts**, **partial overlap**, and **frame borders** (zero-padded `grid_sample` policy vs external tools). **`tests/test_map_image_plane.py`** now includes a **subpixel CRVAL shift** coadd smoke and an **edge partial-overlap** bilinear case.
- **Documented** SWarp comparator default: **`FSCALASTRO_TYPE NONE`** in **`benchmarks/bench_map_compat.py`** ``_coadd_with_swarp`` so hidden astrometric flux scaling does not confound resampling parity (see inline comment there).

### Phase 2 — Montage-class templates

- Treat **Montage header templates** as a first-class round-trip surface: template written for `mProject` should match **`plan_image_plane_target`** / normalized headers on the same grid (see **`tests/test_map_compat_bench.py`**: CD matrix template + **CDELT-only** template).
- Add bench rows for **projection edge cases** where `mProject` / `mProjectPP` and torchsky grids are representable. **`benchmarks/bench_map_compat.py`** includes **`synthetic_montage_sin_grid`** (RA/DEC **SIN**) alongside TAN cases.

### Phase 3 — DFT / FFT / spherical reprojection

- **3a:** `resample_translation_fft2d` plus **`coadd_image_plane(..., resampling_kernel="fft_shift")`** when source/target WCS are a **pure pixel translation** on the same array shape (linear frame; **no SIP/TPV**). Uses **periodic** FFT boundaries; compare interiors to `bilinear` on synthetic CRPIX offsets.
- **3b:** CI pytest **`test_map_dfreproject_parity`** locks small-field **bilinear** agreement between torchsky and **`dfreproject.calculate_reprojection`** (DFT backend when installed). **`test_map_reproject_fft_parity`** covers **`reproject_interp`**, **`reproject_adaptive`** finiteness, and **HEALPix** bilinear round-trips. **`fft_diagonal_affine`** (diagonal FFT), **`fft_linear_affine`** (mixed FFT + periodic bilinear via `resample_fft_ldu_affine_2d`), and **`periodic_linear_affine`** (full affine, periodic **bilinear**) cover the same-shape linear WCS gap; a **single-stage band-limited FFT** for arbitrary 2×2 affine (e.g. full 2D NUFFT) is still not implemented.
- **3c:** **HEALPix → TAN bridge** smoke (**`test_map_healpix_coadd_bridge_smoke`**) and **SH** pipeline finiteness (**`test_map_sphere_sh_map_smoke`**) pin spherical adjacency to `torchsky.map` / `torchsky.sphere.spectral`. **`sample_healpix_map_on_tan_header`** (see `map/healpix_image_plane.py`) bilinearly samples a ring/nest HEALPix map onto a FITS WCS grid using **`torchsky.wcs.healpix.get_interp_val`** only—no **`reproject`**—with parity checked in **`test_map_healpix_native_tan`** when **`reproject`** is installed.

### Phase 4 — Drizzle comparator gaps

- **External `drizzle` regression** with `TORCHSKY_USE_EXTERNAL_DRIZZLE=1` now supports both **`surface_brightness`** (`iscale=1`) and **`flux`** (`iscale` set from the pixel-scale ratio squared), matching the SMACS reproduction / compat conventions.
- **Kernels:** only **`drizzle_point`** / **`drizzle_square`** are public; extra DrizzlePac kernels remain future work.
- **Smoke tests:** `tests/test_map_drizzle_external_parity.py` (optional `drizzle` install) exercises the external drizzle path for both semantics on a tiny grid. Tight native-vs-`drizzle`-package parity stays in `benchmarks/bench_map_compat.py` (different pixmap construction).
- **DrizzlePac pipeline** (distortion tables, CR rejection, blot, etc.) is **not** duplicated inside `coadd_image_plane`.

### Phase 5 — Benchmarks and docs

- Extend **synthetic truth** harness rows for SWarp/Montage/drizzle/FFT where tools are available; record wall-time baselines in `docs/benchmarks.md` when hosts are pinned.
- **Optional wall-time gate:** `benchmarks/bench_map_compat.py` accepts **`--time-baseline-json`** (see **`benchmarks/baselines/map_compat_time_gate_example.json`**) plus **`--time-regression-tolerance`**; default CI does **not** gate on wall time.
- **CI map job** (`.github/workflows/ci.yml`, **map-mosaic-bench**) runs full pytest with **`torchsky[bench]`** so `reproject` / **`dfreproject`** / `drizzle` parity modules execute on every PR when the job is enabled.
