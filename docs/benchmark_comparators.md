# Benchmark comparators: open-source matrix and environments

Torchsky benchmarks are **not** Astropy- or healpy-only. CI and local runs combine several **Pixi environments** because upstream wheels and ABI pins differ (Python 3.10 legacy WCS vs 3.13+ map tools vs HEALPix-heavy stacks).

**Goal (product policy):** on every **comparable** row in the emitted CSV/JSON, **torchsky should rank first** on throughput (and stay within parity tolerances). Rows where a comparator is skipped (`SKIPPED` / different semantics) are **not** counted as losses; expand coverage carefully so skips shrink over time.

---

## Environments (Pixi)

| Environment | Purpose | Typical comparators |
|-------------|---------|---------------------|
| `bench-all` | Default dev + `bench_all.py` WCS/sphere domains | **Astropy** (`astropy_torch` timing in `bench_wcs.py`), **torch**; optional subprocess bridge to legacy WCS |
| `bench-legacy` | **Python 3.10**, `numpy<2`, `starlink-ast` | **PyAST** (`starlink.Ast`), **Kapteyn** (`kapteyn.wcs`) — used by `bench_wcs_legacy_only.py` and the `bench_all` **legacy bridge** |
| `sphere-bench` | HEALPix ecosystem | **healpy**, **hpgeom**, **astropy-healpix**, **healsparse**, **spherical-geometry**; optional extras via `scripts/bootstrap_sphere_bench.py` (**healpix**, **mhealpy**, **skyproj**, harmonic stacks, …) |
| `map-compat` | Optical/IR stackers (narrow grids) | **reproject**, **drizzle**, **MontagePy**, **SWarp**, **casutools**, etc. (see `bench_map_compat.py`) |

Switching envs is normal: e.g. `pixi run -e sphere-bench …` then `pixi run -e bench-all …` for torchsky rebuilds against the same git tree.

---

## WCS domain

| Library | Role | How it is exercised |
|---------|------|---------------------|
| **Astropy** (`wcslib` via `astropy.wcs`) | Primary torchsky comparator in `bench_wcs_suite.py` | Same headers as torchsky; `astropy_torch` timings in `bench_wcs.py` |
| **PyAST** (`starlink-pyast` / `starlink.Ast`) | Classic FITS-WCS library | `bench_competitors.py`, `bench_wcs_legacy_only.py`, merged via `bench_all` legacy **bridge** when `pixi` can spawn `bench-legacy` |
| **Kapteyn** | Python WCS / spectral cube tooling | Same legacy path as PyAST |

Install / run:

```bash
# Primary suite (torchsky + Astropy in-process)
pixi run -e bench-all python benchmarks/bench_all.py --profile user --wcs-only

# Include legacy rows when bench-legacy env is available (cross-env subprocess)
pixi run -e bench-all python benchmarks/bench_all.py --profile user --wcs-only --legacy-wcs
```

Legacy env setup (PyAST + optional Kapteyn):

```bash
pixi install -e bench-legacy   # once, creates .pixi/envs/bench-legacy
pixi run -e bench-legacy pip install -e . --no-build-isolation
# Kapteyn is optional and not a Pixi lock dependency (sdist build needs NumPy at
# build time; some platforms lack wheels). Install when benchmarks should include it:
pixi run -e bench-legacy python -m pip install kapteyn
```

If `kapteyn` fails to build, legacy rows for Kapteyn still appear as **`SKIPPED`** in `bench_wcs_legacy_only.py`; **PyAST** (`starlink-ast`) remains the primary extra WCS comparator in `bench-legacy`.

---

## Sphere / HEALPix domain

| Library | Role | Typical bench surface |
|---------|------|------------------------|
| **healpy** | Reference HEALPix Python bindings | `bench_sphere_suite.py`, replays, `bench_healpix*.py` |
| **hpgeom** | Geometry-heavy HEALPix operations | `bench_sphere_geometry.py`, matrix gates |
| **astropy-healpix** | Astropy-aligned HEALPix API | Same |
| **healsparse** | Sparse maps | `bench_sphere_sparse.py` |
| **spherical-geometry** | Polygon / region parity | `bench_sphere_polygons.py`, replays |
| **healpix** (standalone), **mhealpy**, **skyproj** | Optional ecosystem | `bootstrap_sphere_bench.py` |
| **torch-harmonics**, **s2fft**, … | Harmonic / ML sphere (optional) | `--with-harmonics` bootstrap |

Full HEALPix comparator install:

```bash
pixi run sphere-bench-bootstrap-core
# optional extras:
pixi run sphere-bench-bootstrap --with-harmonics
```

Representative **multi-library** geometry bench (already in `pixi.toml`):

```bash
pixi run sphere-bench-geometry-ecosystem
```

---

## Map / resampling (separate contract)

`bench_map_compat.py` and friends compare torchsky mapmaking against **reproject**, **dfreproject**, **drizzle**, **Montage**, **SWarp**, **casutools**, etc. Use the **`map-compat`** env and `scripts/bootstrap_map_compat_tools.py` — see [`benchmarks.md`](benchmarks.md) and [`install.md`](install.md).

---

## Record versions for a benchmark report

After installing comparators in an env:

```bash
pixi run -e bench-all python scripts/print_benchmark_comparator_versions.py --json-out benchmarks_results/VERSIONS_comparators_bench-all.json
pixi run -e sphere-bench python scripts/print_benchmark_comparator_versions.py --json-out benchmarks_results/VERSIONS_comparators_sphere-bench.json
pixi run -e bench-legacy python scripts/print_benchmark_comparator_versions.py --json-out benchmarks_results/VERSIONS_comparators_bench-legacy.json
```

Or one shot:

```bash
pixi run bench-comparator-versions-all
```

---

## See also

- [`benchmarks/PHASE0_WCS_SPHERE.md`](../benchmarks/PHASE0_WCS_SPHERE.md) — baseline procedure
- [`benchmarks.md`](benchmarks.md) — script index and gates
- [`compatibility.md`](compatibility.md) — user-facing compatibility story
