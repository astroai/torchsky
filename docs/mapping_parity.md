# Mapmaking parity: torchsky vs classical stacks

This note summarizes **when** `torchsky.map` image-plane coaddition is expected to agree with common optical/IR tools (SWarp, Montage, `reproject`, DrizzlePac-style drizzle), and where it **intentionally differs**. Exact agreement requires matching **grid**, **kernel**, **flux vs surface-brightness semantics**, and **weight maps**.

## Resampling kernels

| torchsky `MapResamplingKernel` | Typical classical analogue | Notes |
|--------------------------------|----------------------------|--------|
| `nearest` | Nearest-neighbor regrid | Fast; not flux-conserving. |
| `bilinear` | `reproject` bilinear, many SWarp/Montage linear modes | Good default for smooth maps; see tests vs `reproject_interp`. |
| `bicubic`, `catmull_rom`, `mitchell_netravali` | Higher-order interpolation in `reproject` / SWarp cubic family | Not identical polynomial to every implementation; compare on synthetic truth. |
| `lanczos3`, `lanczos5` | Lanczos-style mosaickers | Implemented via `grid_sample`; edge handling differs from all legacy stacks. |
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

**Optional comparators:** `reproject` (and other legacy stack clients) are pulled in by **`torchsky[bench]`** for benchmarks and external comparisons, not by **`torchsky[test]`**. The `test_map_reproject_fft_parity` module is skipped when `reproject` is absent.
