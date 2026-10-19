# torchsky Architecture & API Review (Deep Pass)

> Generated 2026-05-02 from a deep walk of `src/torchsky/` (≈ 43 k LOC Python, 16 k LOC C++).
> This document supersedes the shorter `docs/ARCH_REVIEW.md` draft. It is a code-grounded
> review against the stated mission: a torch-native, differentiable, any-device library
> to model the observable sky end-to-end (foregrounds, instruments, backgrounds, sources;
> any wavelength, any modality), preferring learned models where data outpace inductive
> biases and physical models where they remain unbeatable.

## TL;DR

The package has succeeded at building a strong torch-native FITS + WCS + sphere + map
foundation. But:

1. **The architecture is shaped like an I/O+coadd library**, not a joint-inference
   stack. The mission-essential concepts (scene, instrument, observation context,
   kinematic state, likelihood, fit) are buried 3 levels deep inside `map.field` and
   `map.operators` and tagged "experimental".
2. **The public surface is heavy and redundant.** Same primitives are exposed under up
   to four namespaces (`torchsky.X`, `torchsky.coords.X`, `torchsky.wcs.X`,
   `torchsky.sphere.X`). Ten distinct `read*` entry points exist for FITS payload reads.
3. **A handful of god modules carry the bulk of the code.** `__init__.py` is 3149 LOC
   (does dispatch, caches, lazy load, fast paths in one file), `map/field/core.py` is
   7713 LOC with 22 `nn.Module` subclasses, `wcs/core.py` is 4108 LOC, `sphere/spectral.py`
   is 4112 LOC, `table.py` is 3760 LOC.
4. **Names leak research jargon and internal axes.** `StrainerSkyEncoderMLPField`,
   `accumulate_image_on_spin_alm_adjoint`, `resample_linear_affine_periodic_bilinear_2d`,
   etc. are unambiguous but encode 4–5 axes into the symbol where a small number of
   parameterised functions would be enough.
5. **The CLI does not match the agent-friendly, lean shape advertised in the mission.**
   Two commands (`viz`, `map`) where `map` carries ~70 flags split across two unrelated
   workflows (drizzle/coadd vs INR fit).

The plan in §6 reorganises the package around four pillars (`io`, `world`, `scene`,
`viz`), promotes the joint-inference engine to first class, and is staged so the first
three phases can ship with **no user-visible churn**.

---

## 1. Method

- Walked the full `src/torchsky/` tree (35 directories, 89 Python files, 17 C++ files).
- Counted LOC for every Python file and every C++ translation unit.
- Read all sub-package `__init__.py` files to map the public surface.
- Read the 3149-line root `__init__.py` end to end (helpers, dispatch, lazy loader).
- Cross-referenced `docs/api.md`, `docs/getting-started.md`, `docs/cli.md`,
  `docs/roadmap/status.md`, and `README.md`.
- Sampled `examples/` to understand idiomatic user code paths.
- Did not run benchmarks; LOC and dependency counts are mechanical, but the mission /
  intuitiveness assessment is structural.

## 2. Package shape (where the weight sits)

LOC by module (Python, top 15):

| LOC   | Path                                          | Role                                           |
|-------|-----------------------------------------------|------------------------------------------------|
| 7713  | `map/field/core.py`                           | 22 `nn.Module` field architectures + training  |
| 4112  | `sphere/spectral.py`                          | Scalar+spin SHT, beams, pixwin, 16+ caches     |
| 4108  | `wcs/core.py`                                 | `WCS` class: parse, projections, slicing, jacobians |
| 3760  | `table.py`                                    | Arrow/FITS table I/O, polars/duckdb interop    |
| 3268  | `map/image_plane.py`                          | Image-plane drizzle/coadd kernels + planning   |
| 3149  | `__init__.py`                                 | Package facade, FITS dispatch, caches, lazy    |
| 2410  | `map/solvers.py`                              | Harmonic priors, scene operators, hooks        |
| 2228  | `wcs/healpix.py`                              | HEALPix indexing/queries (lives under wcs!)    |
| 2225  | `hdu.py`                                      | `Header`, `Card`, `TensorHDU`, `TableHDU`, `HDUList` |
| 1854  | `sphere/geom.py`                              | Sphere polygons, regions, predicates           |
| 1216  | `map/product.py`                              | `MapProductSpec` orchestration                 |
|  802  | `sphere/sparse.py`                            | `HealSparseMap`, `SparseMap`                   |
|  788  | `sphere/compat.py`                            | healpy-shaped wrapper                          |
|  638  | `sphere/moc.py`                               | MOC / MultiOrder Coverage                      |
|  617  | `wcs/sip.py`                                  | SIP distortion                                 |

LOC by C++ translation unit (top 6):

| LOC   | Path                                      | Role                                |
|-------|-------------------------------------------|-------------------------------------|
| 3866  | `cpp_src/fits.cpp`                        | CFITSIO image / header / write      |
| 3438  | `cpp_src/healpix.cpp`                     | HEALPix kernels                     |
| 2775  | `cpp_src/table.cpp`                       | CFITSIO BINTABLE                    |
| 2752  | `cpp_src/bindings.cpp`                    | nanobind bindings (~40 image, 15 table funcs) |
| 2030  | `cpp_src/wcs.cpp`                         | Fused WCS                           |

Total ~32 k LOC concentrated in 8 files. Splitting is necessary purely for
maintainability and parallel work.

## 3. Findings

### 3.1 Directory structure & redundancy

#### 3.1.1 Top-level `__init__.py` is a 3149-line god module

```text
src/torchsky/__init__.py
  - 35 helper functions (cache invalidation, header probing, HDU autodetect,
    cold-mmap policy, image-meta cache, …)
  - 8 module-level OrderedDict caches: _file_cache, _file_handle_cache,
    _file_handle_sig_cache, _image_meta_cache, _hdu_type_cache,
    _cold_nommap_cache, _auto_mmap_cache, _auto_hdu_cache
  - read() with 4 @overload signatures + 1 real definition
  - read_image, read_table, read_subset, read_hdus, read_batch,
    read_large_table, stream_table, read_full_*
  - write, insert_hdu, replace_hdu, delete_hdu
  - open, get_header, get_wcs, write_checksums, verify_checksums
  - clear_file_cache, get_cache_performance, get_batch_info
  - half-implemented __getattr__ lazy loader
```

The file mixes four distinct roles:

- Package façade (re-exports and `__all__`).
- Core FITS I/O dispatch engine.
- Cache policy.
- Lazy loader.

Cold `import torchsky` pays for the entire stack on every entry.

#### 3.1.2 Empty alias namespaces (`coords`, `geom`)

`src/torchsky/coords/__init__.py` (99 LOC) imports nothing of its own; it re-exports
`WCS`, `healpix.*`, `frames.*`, `spectral.*`, `tabular.*`, `temporal.*`, `alignment.*`
from `torchsky.wcs`.

`src/torchsky/geom/__init__.py` (64 LOC) re-exports `MOC`, `SphericalPolygon`,
`query_polygon_general`, `great_circle_distance`, etc. from `torchsky.sphere`.

A user looking up `query_polygon_general` will find it under
`torchsky.geom`, `torchsky.sphere`, `torchsky.sphere.geom`, and (transitively)
`torchsky.coords` (because `coords` re-exports `healpix` which re-exports parts of sphere).
Same primitives, four namespaces.

#### 3.1.3 Reverse-dependent `io` namespace

`src/torchsky/io/__init__.py` defines `read`, `read_image`, `read_table`, `read_hdus`,
`read_subset`, `open`, `write`, `get_header`, `get_wcs`, `stream_table`,
`read_large_table` — each as a single-line wrapper that imports from the parent and
forwards the call:

```40:50:src/torchsky/io/__init__.py
def read_table(*args: Any, **kwargs: Any):
    from .. import read_table as _read_table

    return _read_table(*args, **kwargs)
```

Net effect: `torchsky.io.read` and `torchsky.read` are the same callable. There is no
meaningful boundary; the source of truth is the root `__init__.py`. `torchsky.io` is
also a circular dependency bridge in mock-based tests.

`io.read_fast` aliases `torchsky._fastio.read`, which itself overlaps with
`torchsky.read_image` (both are documented as "fast direct path"). Three "fast" reads.

#### 3.1.4 HEALPix lives under `wcs`, not `sphere`

`src/torchsky/wcs/healpix.py` is 2228 LOC of HEALPix nest/ring conversion, query disc /
polygon / box, neighbour finding, interpolation weights, Fourier features. It is then
re-imported by `sphere/__init__.py`:

```24:32:src/torchsky/sphere/__init__.py
from ..wcs.healpix import (
    ang2pix,
    npix2nside,
    nside2npix,
    pix2ang,
    pix2vec,
    query_disc,
    vec2pix,
)
```

HEALPix is a pixelisation of the sphere, not a WCS projection family. `map/`,
`catalogs/`, `sphere/spectral.py`, `sphere/compat.py`, `sphere/sparse.py`, and
`encoders/__init__.py` all import `..wcs.healpix`, so the import graph crosses the
`wcs ↔ sphere` boundary in both directions. Placement is a historical accident.

#### 3.1.5 Three different `spectral` modules

| Path                           | Role                                              |
|--------------------------------|---------------------------------------------------|
| `torchsky/spectral.py`         | FITS spectral axis + DataCube + 1D spectra dataclasses |
| `torchsky/wcs/spectral.py`     | WCS spectral coordinate translations              |
| `torchsky/sphere/spectral.py`  | Spherical harmonic transforms (4112 LOC)          |

All three appear in `__all__` lists. The `docs/spectral.md` page covers only the
top-level `torchsky.spectral`. A user searching for `spectral` cannot tell which one
they want.

#### 3.1.6 `torchsky.map` is a kitchen-sink namespace

`map/__init__.py` is 316 LOC and exports **~150 symbols** spanning:

- 22 `nn.Module` field architectures (`SineImplicitField`, `WireImplicitField`,
  `GaussianBasisField`, `FourierMLPField`, `SpectralBasisField`,
  `BandpassConditionedField`, `LearnedSpectralBasisField`,
  `SkyEncoderBandpassField`, `ConditionedSpectralField`,
  `HashGridConditionedField`, `SkyEncoderMLPField`,
  `StrainerSkyEncoderMLPField`, `SkyEncoderResMLPField`,
  `HashGridMLPField`, `HashGridResMLPField`, `HashGridSpectralField`,
  `SkyLambdaPlanesField`, `SkyLambdaKPlanesField`,
  `SkyLambdaFeatureField`, `OperatorConstrainedFeatureField`,
  `SkyLambdaAtlasField`, `SkyLambdaFeatureAtlasField`)
- 7 image-plane resamplers (`resample_diagonal_affine_fft2d`,
  `resample_fft_ldu_affine_2d`, `resample_linear_affine_periodic_bilinear_2d`,
  `resample_linear_affine_spectral_2d`, `resample_nonuniform_spectral_fast_2d`,
  `resample_nonuniform_spectral_2d`, `resample_translation_fft2d`)
- 12 HEALPix↔image samplers (`sample_healpix_map_on_image_header`, …
  `sample_healpix_spin_map_on_tan_header_harmonic`)
- 5 coadd entry points (`coadd_image_plane`, `coadd_image_plane_cube`,
  `coadd_image_plane_multiband`, `plan_image_plane_*`)
- 13 harmonic prior / state operator builders (`build_harmonic_cl_prior_operator`,
  `build_state_kronecker_operator`, …)
- Dataclasses: `MapResult`, `MultiBandMapResult`, `SpectralCubeMapResult`,
  `MapProductResult`, `MapProductSpec`, `MapBandpassSpec`, `MapUncertaintyMode`,
  `BandMetadataTable`, `PlannedImagePlaneTarget`, …
- Field-related: `FieldDomain`, `FieldObservation`, `FieldFitResult`, `PSFSpec`,
  `KinematicState`, `SurveyDataTerm`, `ForwardContext`, `FieldSpectralQuantity`,
  `ObservationSpectralSummary`, `SpectralCapacityProfile`, …
- INR helpers: `make_field_model`, `fit_field_model`,
  `fit_field_component_decomposition`, `make_field_component_decomposition`,
  `make_tangent_plane_header`, `field_observations_from_bin_centers`,
  `field_observations_from_bin_edges`, `field_observations_from_ifu_cube`,
  `evaluate_field_coordinates_vmapped`, `render_field_*` (5 variants),
  `pretrain_strainer_trunk_synthetic`, `save_strainer_trunk_checkpoint`,
  `load_strainer_trunk_checkpoint`, `multi_survey_charbonnier_loss`,
  `observation_charbonnier_loss`, `predict_observation_mean`,
  `scale_observation_for_rest_frame_scene`, `spectral_data_budget`,
  `suggest_num_spectral_samples`, `suggest_spectral_model_capacity`

This is at least three subsystems flat-exported:

1. The classical image-plane coadd / drizzle pipeline.
2. The harmonic / Bayesian-prior solver hooks.
3. The implicit (INR) generative-field engine.

The user has no surface-level cue about which is which.

#### 3.1.7 Caches are not unified

| File                                  | Caches                                               |
|---------------------------------------|------------------------------------------------------|
| `__init__.py`                         | `_file_cache`, `_file_handle_cache`, `_file_handle_sig_cache`, `_image_meta_cache`, `_hdu_type_cache`, `_cold_nommap_cache`, `_auto_mmap_cache`, `_auto_hdu_cache` |
| `table.py`                            | `_handle_cache`, `_reader_cache` (own locks, env vars) |
| `cache.py`                            | `CacheConfig` — environment policy class             |
| `buffer.py`                           | `MemoryPool` — tensor buffer pool                    |
| `sphere/spectral.py`                  | `_ANGLE_CACHE`, `_YLM_CACHE`, `_ALM_ELL_CACHE`, `_ALM_M_CACHE`, `_SPIN_YLM_CACHE`, `_SPIN_MAP2ALM_MAT_CACHE`, `_SPIN_ALM2MAP_MAT_CACHE`, `_SCALAR_MAP2ALM_MAT_CACHE`, `_SCALAR_ALM2MAP_MAT_CACHE`, `_RING_*` (7 more), `_SCALAR_RING_AUTO_MIN_BYTES`, `_SPIN_*_TOTAL_BYTES` (4) |
| `wcs/healpix.py`                      | `_FACE_CONST_CACHE`, `_CAP_START_CACHE`, `_MAX_PIXEL_RADIUS_CACHE` |
| C++ side                              | `torchsky::configure_cache(max_files, max_memory_mb)` |

Total: ~25 caches with independent lifecycles, no `clear_all()`, no `stats()` reporter,
no test isolation hook. Some are bounded, some are not. Some respond to env vars
(`TORCHSKY_SPIN_MAX_CACHE_BYTES`, `TORCHSKY_SCALAR_MAX_CACHE_BYTES`,
`TORCHSKY_TABLE_HANDLE_CACHE_SIZE`, `TORCHSKY_CFITSIO_CACHE_MB`,
`TORCHSKY_CFITSIO_CACHE_FILES`), each a separate convention.

### 3.2 API and naming

#### 3.2.1 Read-path zoo

The user has 11 read entry points to choose from:

| Entry point                          | Returns                                | Notes                            |
|--------------------------------------|----------------------------------------|----------------------------------|
| `torchsky.read(...)`                 | tensor or dict, with optional header   | Canonical dispatcher, `mode=auto/image/table` |
| `torchsky.read_image(...)`           | tensor                                 | "Direct low-level path; bypasses runtime policy dispatch" |
| `torchsky.read_table(...)`           | dict / table                           | Wrapper around `read(mode="table")` |
| `torchsky.read_subset(...)`          | tensor                                 | Image cutout                     |
| `torchsky.read_hdus(...)`            | list[tensor]                           | Multi-HDU                        |
| `torchsky.read_batch(...)`           | list[tensor]                           | Multi-file                       |
| `torchsky.read_large_table(...)`     | dict accumulator                       | Chunked, falls back              |
| `torchsky.stream_table(...)`         | generator                              | Chunked iterator                 |
| `torchsky.io.read_fast(...)`         | tensor                                 | Aliases `torchsky._fastio.read`  |
| `torchsky.table.read(...)`           | Arrow Table / dict / polars            | Different return shape           |
| `torchsky.open(path).primary`        | `HDUList[0]`                           | Astropy-shaped object API        |

This is the largest single source of friction. New users cannot guess which one to
call; experienced users carry the table in their heads.

#### 3.2.2 Long descriptive function names

Sample from `torchsky.map`:

```text
accumulate_image_on_healpix_map_adjoint
accumulate_image_on_alm_adjoint
accumulate_image_on_spin_alm_adjoint
sample_healpix_map_on_image_header
sample_healpix_map_on_image_header_harmonic
sample_healpix_map_on_image_header_spectral
sample_healpix_spin_map_on_image_header_harmonic
sample_healpix_map_on_tan_header
sample_healpix_map_on_tan_header_harmonic
sample_healpix_map_on_tan_header_spectral
sample_healpix_spin_map_on_tan_header_harmonic
sample_alm_on_image_header_spectral
sample_alm_on_tan_header_spectral
sample_spin_alm_on_image_header_spectral
sample_spin_alm_on_tan_header_spectral
resample_diagonal_affine_fft2d
resample_fft_ldu_affine_2d
resample_linear_affine_periodic_bilinear_2d
resample_linear_affine_spectral_2d
resample_nonuniform_spectral_fast_2d
resample_nonuniform_spectral_2d
resample_translation_fft2d
```

Each name encodes 4–5 axes (operation × representation × target geometry × kernel ×
dimensionality). A cleaner library carries those axes as parameters:

```python
torchsky.functional.resample(image, kernel="lanczos3", domain="fft", affine=...)
torchsky.functional.sample(map, target=header, mode="harmonic")
torchsky.functional.accumulate(image, target_map, adjoint=True)
```

This shrinks the public surface ~10× without losing any capability.

#### 3.2.3 Research-label class names leaked publicly

```python
StrainerSkyEncoderMLPField     # "Strainer" is a paper name
SkyLambdaKPlanesField           # K-Planes is a paper architecture
SkyLambdaPlanesField            # ditto
SkyLambdaAtlasField             # internal naming
SkyLambdaFeatureAtlasField      # "feature atlas" is internal terminology
SkyLambdaFeatureField           # internal
OperatorConstrainedFeatureField # internal
SkyEncoderBandpassField         # OK but composite
WireImplicitField               # WIRE paper
SineImplicitField               # SIREN
HashGridConditionedField        # Instant-NGP-style
HashGridSpectralField
LearnedSpectralBasisField
ConditionedSpectralField
BandpassConditionedField
GaussianBasisField
FourierMLPField
SpectralBasisField
```

`make_field_model("sine_implicit")` already exists as a registry. Both the registry
and the 22 individual classes are public; users have no signal about which is the
recommended path. The mission requires the **factory** to be the public API and the
class names to be implementation labels.

#### 3.2.4 Two surfaces for the "observation" concept

`ImageObservation` (in `map/image_plane.py`) and `FieldObservation` (in
`map/field/types.py`) describe the same conceptual object — an instance of survey
data with WCS, weights, metadata — for two different downstream paths. The CLI has
to convert between them in `cli/map_command.py:_paths_to_field_observations`. The
mission ("model any data jointly") wants one `Observation` shape that works for
classical coadd, INR fit, harmonic inversion, and future modalities.

#### 3.2.5 CLI surface is small but unbalanced

```text
torchsky viz   ... ~12 flags; fine.
torchsky map   ... ~70 flags; double-purpose:
                       --method coadd  → classical drizzle / coadd workflow
                       --method inr    → INR field fit + render
```

Missing for a torch-native science library:

```text
torchsky info    <fits>                    # HDU summary, headers, shapes, types
torchsky show    <fits>                    # browser preview / PNG dump
torchsky cat     <fits> --columns RA,DEC   # table rows (Polars-shaped) to stdout
torchsky read    <fits> -o out.npy         # bulk read to torch / npy / parquet
torchsky table   head|schema|stats <fits>  # FITS table inspection
torchsky world   inspect <fits>            # WCS summary; pixel⇄world demo
torchsky scene   coadd  <inputs...>        # current `map --method coadd`
torchsky scene   fit    <inputs...>        # current `map --method inr`
torchsky scene   render <model.pt> ...     # future
```

A flat 30-flag `--inr-...` block also fights the `cli-for-agents` skill in this
workspace.

### 3.3 Mission alignment

The roadmap (`docs/roadmap/status.md`) states the actual mission cleanly:

> Survey inference is one inverse problem: find the latent scene and instrument law
> such that predicted data agree with observations.
>
> $Y \sim \mathbb{P}_{\psi}(\,\cdot \mid F_{\theta},\, \zeta,\, c(t)\,)$
>
> - $F_{\theta}$: continuous latent specific intensity (the sky)
> - $\zeta$: kinematic latents (redshift, proper motion, acceleration)
> - $c(t)$: observational context (pointing, platform ephemeris)
> - $\mathbb{P}_{\psi}$: instrument-environment-noise model

The README, in contrast, advertises stable surfaces as:

> `torchsky.read`, `torchsky.write`, `torchsky.open`, `torchsky.table`,
> `torchsky.wcs.WCS`, `torchsky.sphere`, `torchsky.map.coadd_image_plane`,
> `torchsky.map.stack_map_product`.

These are I/O + classical coadd. The mission-essential pieces are buried:

| Mission concept                | Lives at                                           | Visibility           |
|--------------------------------|----------------------------------------------------|----------------------|
| Latent scene $F_\theta$        | `torchsky.map.field.*Field` (22 classes)           | experimental tier    |
| Instrument operator $\psi$     | `torchsky.map.operators.bundle.ObservationOperatorBundle` | "internal scaffolding" docstring |
| Forward context $c(t)$         | `torchsky.map.field.types.ForwardContext`          | dataclass, 3 levels deep |
| Kinematic state $\zeta$        | `torchsky.map.field.implicit_forward.KinematicState` | 3 levels deep    |
| Likelihood $\mathbb{P}_\psi$   | `multi_survey_charbonnier_loss`, `predict_observation_mean` | helper-function shaped |
| Fit                            | `torchsky.map.field.fit_field_model`               | one of 25 entry points in `map.field` |
| Coadd-vs-fit choice            | `--method coadd|inr` flag in CLI                   | flag-shaped          |

The architecture currently treats joint inference as a side feature of `map`.
The mission needs the opposite: I/O + WCS + sphere are the **substrate**, and `scene`
(with model + instrument + observation + fit) is the **product**.

### 3.4 Performance and runtime

#### 3.4.1 Half-implemented lazy imports

```30:53:src/torchsky/__init__.py
from .buffer import clear_buffers, configure_buffers, get_buffer_stats
from .cache import clear_cache, configure_for_environment, get_cache_stats
from .core import ChecksumVerifier, CompressionType, FITSCore
from .dataloader import (
    create_dataloader,
    create_fits_dataloader,
    create_streaming_dataloader,
    create_table_dataloader,
)
from .datasets import FITSDataset, IterableFITSDataset, TableChunkDataset
from .frame import read_tensor_frame, to_tensor_frame, write_tensor_frame
from .hdu import Card, HDUList, Header, TableHDU, TableHDURef, TensorHDU
from .interop import to_arrow, to_pandas, to_polars
from . import table
from . import sphere
from . import io
```

```3141:3149:src/torchsky/__init__.py
_LAZY_NAMESPACES = {"io", "coords", "geom", "map", "encoders", "catalogs"}


def __getattr__(name: str):
    if name in _LAZY_NAMESPACES:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
```

`io` is on the lazy list but eagerly imported on line 45. `sphere` is eagerly
imported, which compiles 4112 LOC of SHT cache machinery on every `import torchsky`.
`table` is eagerly imported, which pulls PyArrow when present. The lazy hook only
fires for `coords`, `geom`, `map`, `encoders`, `catalogs`. Lazy loading is half-done.

#### 3.4.2 C++ surface fragmentation

`bindings.cpp` exposes ~40 image readers and ~15 table readers with overlapping
shapes:

```text
read_full(file, hdu, mmap)                       read_full_cached(...)
read_full_nocache(...)                           read_full_unmapped(...)
read_full_unmapped_raw(...)                      read_full_raw(...)
read_full_raw_with_scale(...)                    read_full_scaled_cpu(...)
read_full_numpy(...)                             read_full_numpy_cached(...)
read_rice_parallel(...)                          read_full(handle, hdu, mmap)

read_fits_table(...)                             read_fits_table_from_handle(...)
read_fits_table_rows(...)                        read_fits_table_rows_from_handle(...)
read_fits_table_rows_numpy(...)                  read_fits_table_rows_numpy_from_handle(...)
read_fits_table_filtered(...)                    read_fits_table(handle, ...)
```

`__init__.py` then probes for capabilities (`hasattr(cpp, "read_full_raw_with_scale")`,
…) and dispatches across these. The Python policy code is ~600 LOC of probe-and-call.
A single C++ entry per domain (`fits.read_image(opts)`, `fits.read_table(opts)`)
taking an options struct would erase most of that.

#### 3.4.3 Strict tier enforcement is absent

`surfaces.py` declares `core` / `compat` / `experimental` tiers, but nothing actually
warns or gates when an experimental surface is imported. `torchsky.map.*` exports
~150 names mixing all three tiers; users see one flat surface.

### 3.5 What is healthy

To balance the picture, the codebase is not in trouble:

- Strict pre-push gate (`pixi run ci-local`) and the agents-memory operating model
  are in place.
- WCS replacement coverage is solid: 1500+ tests passing, broad astropy/wcslib parity,
  differentiable `pixel_to_world`, Jacobian extraction, slicing, alternate keys, SIP,
  TPV, TNX, ZPX, spectral `-LOG` / `WAVE-F2W` / `FREQ-W2F`, temporal scale conversions.
- `torchsky.table` is genuinely Arrow-shaped (predicate pushdown, polars/duckdb
  bridges, streaming).
- `torchsky.sphere.healpy` / `torchsky.sphere.compat` is the right shape for migration.
- `surfaces.py` is the right idea even though it isn't yet enforced.
- The C++ extension scoping (CFITSIO + nanobind, no Python `fitsio` runtime
  dependency) matches the AGENTS.md contract.
- Tests + benchmarks layout is mature.

The architecture is not broken; it has out-grown its initial shape, and now needs to
be re-cut around the four-pillar mission.

---

## 4. Proposed plan

The plan is staged so each phase is shippable on its own.

### Phase 0 — Lock the model: declare the four pillars

Pick four top-level pillars and make them the entire public surface. Everything else
becomes private (`_internal`) or `tier=experimental` and is gated.

```text
torchsky.io       FITS images / tables / HDUs / cutouts / streaming / lazy datasets
torchsky.world    WCS, frames, projections, sphere, HEALPix, MOC, alignment, harmonics
torchsky.scene    latent sky + instrument + noise + observation + fit + coadd + product
torchsky.viz      browser / notebook visualisation
```

Compatibility shims (healpy, astropy.io.fits-shaped) live under `torchsky.compat.*`.

A user reads:

```python
from torchsky import io, world, scene, viz

img, hdr = io.read("file.fits", return_header=True)
wcs = world.WCS(hdr)

obs = [scene.Observation.from_fits(p) for p in glob("*.fits")]
fit = scene.fit(obs, model="sine_implicit")
pred = fit.predict(scene.target_from_header(hdr))

viz.serve(root=".")
```

The phase is purely a documentation + re-export choice. No code moves yet. It anchors
all later phases.

### Phase 1 — Dismantle the 3149-line god module

1. Move all FITS dispatch logic, helper caches, and `read*` family from
   `torchsky/__init__.py` into a new private package:

   ```text
   torchsky/_io_engine/
     __init__.py        # public-to-package surface
     caches.py          # Caches dataclass: file, handle, image_meta, hdu_type, …
     dispatch.py        # read() top-level dispatch
     fast_image.py      # current _read_cpu_fast_path / _read_generic_fast_path
     fast_table.py
     fallback.py        # current _read_fallback / _read_fallback_image / _read_fallback_table
     header.py
     subset.py          # SubsetReader + read_subset
     batch.py           # _read_batch_paths / _read_batch_hdus
     write.py
     hdu_ops.py         # insert/replace/delete
   ```

2. Reduce `torchsky/__init__.py` to a thin re-export router (~150 LOC) with a real
   uniform `__getattr__` lazy loader covering **every** subpackage. Cold
   `import torchsky` should not import PyArrow, FastAPI, `torch_frame`, or
   `sphere/spectral.py`.

3. Unify the 11 module-level caches into a single `Caches` dataclass with `clear()`,
   `stats()`, env-driven config, and a top-level `torchsky.clear_caches()` /
   `torchsky.cache_stats()`.

4. Collapse the read zoo into:

   ```python
   torchsky.read(path, *, hdu="auto", mode="auto", subset=None,
                 device="cpu", mmap="auto", dtype=None,
                 return_header=False, chunks=None) -> Tensor | dict | Iterator
   torchsky.write(path, data, *, header=None, overwrite=False, compress=False)
   torchsky.open(path) -> HDUList
   ```

   Internal helpers (`_fast_image`, `_fast_table`, `_subset`, `_batch`) stay private.
   Keep the existing public names as deprecated aliases for one minor release.

### Phase 2 — Collapse `coords` / `geom`, fold HEALPix into `world`

1. Delete `torchsky/coords/` and `torchsky/geom/` (or keep as deprecated stubs that
   emit `DeprecationWarning` once on import).
2. Move `torchsky/wcs/healpix.py` (2228 LOC) to `torchsky/world/healpix.py`. Fix
   imports in `map`, `catalogs`, `sphere/spectral`, `sphere/compat`,
   `encoders/__init__.py`.
3. Merge `wcs/` and `sphere/` content under `torchsky/world/`:

   ```text
   torchsky/world/
     __init__.py            # re-exports WCS, healpix, regions, harmonics
     wcs/                   # (old wcs/ content, modularised – see Phase 4)
     healpix.py             # was wcs/healpix.py
     frames.py              # was wcs/frames.py
     spectral_axis.py       # was wcs/spectral.py
     temporal.py            # was wcs/temporal.py
     tabular.py             # was wcs/tabular.py
     alignment.py           # was wcs/alignment.py
     regions.py             # was sphere/geom.py + sphere/moc.py
     sparse.py              # was sphere/sparse.py
     harmonics/             # was sphere/spectral.py modularised – see Phase 4
   ```

4. Pick one and only one `spectral` namespace per pillar:

   - `torchsky.io.spectral` — FITS spectral axis + cube I/O dataclasses
     (current `torchsky.spectral`).
   - `torchsky.world.spectral_axis` — WCS spectral coordinate translations
     (current `torchsky.wcs.spectral`).
   - `torchsky.world.harmonics` — SHT (current `torchsky.sphere.spectral`).

5. Rename verbose user-facing functions while keeping aliases:

   | Current                                             | New                                                     |
   |-----------------------------------------------------|---------------------------------------------------------|
   | `accumulate_image_on_healpix_map_adjoint`           | `world.healpix.accumulate(image, header, target)`       |
   | `sample_healpix_map_on_image_header*`               | `world.healpix.sample(map, header, mode="bilinear|harmonic|spectral")` |
   | `resample_linear_affine_periodic_bilinear_2d`       | `torchsky.functional.resample(image, kernel="bilinear", domain="periodic", affine=...)` |
   | `accumulate_image_on_alm_adjoint`                   | `world.harmonics.accumulate(image, header)`             |
   | `sample_alm_on_image_header_spectral`               | `world.harmonics.sample(alm, header, ...)`              |

### Phase 3 — Promote `scene` to first-class

This is the highest-leverage phase for the mission.

1. Create `torchsky.scene` as the top-level joint-inference surface:

   ```text
   torchsky.scene.Scene              # latent sky F_θ (was *Field hierarchy)
   torchsky.scene.Instrument         # ψ: PSF + bandpass + radiometry + noise + background
   torchsky.scene.Observation        # one observation (data + wcs + ψ + c(t) + ζ)
   torchsky.scene.Context            # exposure / survey / instrument metadata (was ForwardContext)
   torchsky.scene.Kinematics         # ζ (was KinematicState)
   torchsky.scene.Likelihood         # multi-survey loss term (was SurveyDataTerm)
   torchsky.scene.fit                # the optimiser (was fit_field_model + fit_*_decomposition)
   torchsky.scene.predict            # forward: scene + observation -> prediction
   torchsky.scene.render             # scene -> map / cube / HEALPix / observation
   torchsky.scene.coadd              # explicit-image coadd (was coadd_image_plane)
   torchsky.scene.product            # was MapProductSpec / stack_map_product
   torchsky.scene.models.*           # registry: scene.models.make("sine_implicit")
   torchsky.scene.operators.*        # was map/operators
   torchsky.scene.priors.*           # was solvers’ harmonic / cl / dense priors
   ```

2. Unify `ImageObservation` and `FieldObservation` into a single `Observation`
   dataclass with optional bandpass/PSF/weight/header fields. The two paths are
   different views of the same generative chain.

3. Hide all 22 named `*Field` classes behind `scene.models.make(kind, ...)` and
   `scene.models.register(kind, factory)`. Keep the classes importable from
   `torchsky.scene.models._<kind>` but stop re-exporting them at top level.

4. Document the $Y \sim \mathbb{P}_\psi(\cdot \mid F_\theta, \zeta, c(t))$ model
   directly in `docs/scene.md` and link from `docs/getting-started.md`. The model is
   already described in `docs/roadmap/status.md`; lift it to the user-facing layer.

5. Keep the old import paths (`torchsky.map.*`) as deprecated shims for one minor
   release.

### Phase 4 — Modularise the monoliths

Mechanical splits, no behaviour change. Public import paths stay via re-export.

| Current file                     | Split target                                                                                |
|----------------------------------|---------------------------------------------------------------------------------------------|
| `__init__.py` (3149)             | `_io_engine/*` (Phase 1)                                                                    |
| `wcs/core.py` (4108)             | `world/wcs/{__init__,projections,distortions,spectral,temporal,cube,jacobian,parse}.py`     |
| `sphere/spectral.py` (4112)      | `world/harmonics/{caches,scalar,spin,beam,pixwin,public}.py`                                |
| `table.py` (3760)                | `io/table/{cache,reader,writer,scan,interop,where,fits_meta}.py`                            |
| `map/image_plane.py` (3268)      | `scene/coadd/{kernels,drizzle,plane,target,quadrature,executor}.py`                         |
| `map/field/core.py` (7713)       | `scene/models/{base,siren,wire,gabor,fourier_mlp,hashgrid,sky_lambda,sky_encoder,strainer,operator_constrained,decomposition,renderers,training,pretrain}.py` |
| `hdu.py` (2225)                  | `io/hdu/{header,card,tensor_hdu,table_hdu,hdu_list,view}.py`                                |
| `cpp_src/bindings.cpp` (2752)    | `cpp_src/bindings/{fits,table,wcs,healpix,misc}.cpp`                                        |
| `cpp_src/fits.cpp` (3866)        | `cpp_src/fits/{io,header,write,checksum,compress}.cpp`                                      |
| `cpp_src/healpix.cpp` (3438)     | `cpp_src/healpix/{nest,ring,query,interp,neighbors}.cpp`                                    |

### Phase 5 — CLI redesign

Replace the single double-purpose `torchsky map ...` with a verb–noun CLI shaped for
both humans and agents:

```text
torchsky info     <fits>                     # HDU summary, headers, shapes, types
torchsky show     <fits>                     # browser preview / PNG dump
torchsky cat      <fits> [--columns ...]     # table sample to stdout (Polars-shaped)
torchsky read     <fits> -o out.npy          # bulk read to torch / npy / parquet
torchsky table    head | schema | stats <fits>
torchsky world    inspect <fits>             # WCS summary; pixel⇄world demo
torchsky scene    coadd  <inputs...> -o out.fits  --config coadd.yaml
torchsky scene    fit    <inputs...> -o out.fits  --config fit.yaml
torchsky scene    render <model.pt> --header h.fits -o pred.fits
torchsky viz      [--root ...]               # current viz server
```

- Replace the 30 `--inr-...` flags with `--config inr.yaml`.
- Add `--json` output mode to `info`, `schema`, `stats`, `world inspect` for agent
  consumption (matches the `cli-for-agents` skill contract).
- Keep `torchsky map ...` as a deprecated alias for one minor release.

### Phase 6 — Performance / runtime cleanup

1. Single `torchsky._io_engine.Caches` with bounded LRU and configurable env knobs.
   Public `torchsky.cache_stats()` returns the full picture.
2. Single C++ `fits.read_image(handle|path, hdu, opts)` taking an options struct;
   deprecate the 11 `read_full_*` variants.
3. Single C++ `fits.read_table(...)` likewise; deprecate the 7 `read_fits_table_*`
   variants.
4. Tier enforcement: importing an `experimental` surface emits one-time
   `FutureWarning` unless `TORCHSKY_EXPERIMENTAL=1` or the import goes through the
   explicit `torchsky.experimental` namespace. CI gate to assert `core` surfaces
   never import `experimental`.
5. Lazy import every subpackage by default. Move every top-level
   `from .X import …` into a uniform lazy `__getattr__`.
6. Add `tests/test_cold_import.py` asserting `import torchsky` does not import
   `pyarrow`, `fastapi`, `torch_frame`, `torchsky.world.harmonics`, or
   `torchsky.scene.models`.
7. Add a `cold_import_seconds` smoke benchmark.

### Phase 7 — Documentation realignment

1. Promote `docs/roadmap/status.md`'s mission paragraph
   ($Y \sim \mathbb{P}_\psi(\cdot \mid F_\theta, \zeta, c(t))$) to a top-level
   `docs/architecture.md` and reference it from `README.md` and
   `docs/getting-started.md`.
2. Reshape `docs/api.md` around the four pillars (io / world / scene / viz) instead
   of nine flat namespaces.
3. Mark `docs/coords.md` and `docs/geom.md` as deprecated alias shims.
4. Add a "Concepts" page that defines `Scene`, `Instrument`, `Observation`,
   `Context`, `Kinematics`, `Likelihood` once; everything else cross-references it.

### Sequencing and risk

| Phase | Effort | Risk   | User-facing change                           |
|-------|--------|--------|----------------------------------------------|
| 0     | small  | none   | none (only declares pillars, all-doc)        |
| 1     | medium | low    | private only                                 |
| 2     | medium | medium | deprecates `coords`, `geom`, renames healpix |
| 3     | large  | medium | renames `map.field` → `scene` (alias kept)   |
| 4     | medium | low    | private only (re-export shims)               |
| 5     | medium | low    | new CLI verbs (old `torchsky map` aliased)   |
| 6     | medium | low    | private only                                 |
| 7     | small  | none   | docs                                         |

Phases 0, 1, 6 can ship first with no user-visible churn. Phase 5 is shippable in
parallel. Phases 2, 3, 7 should ship together as a single minor-bump release with
deprecation aliases.

---

## 5. Concrete first cuts (one week)

If only one week is available before a larger reshuffle, these seven cuts give the
biggest win-per-line:

1. **Move root caches.** Pull the eight `_*_cache` dictionaries out of
   `__init__.py` into `torchsky/_io_engine/caches.py` with a single `Caches` class.
   Add `torchsky.clear_caches()` and `torchsky.cache_stats()`.
2. **Fix lazy loading.** Replace `_LAZY_NAMESPACES` with a uniform `__getattr__`
   covering every subpackage; remove the eager `from . import sphere` /
   `from . import table` / `from . import io` lines. Add `tests/test_cold_import.py`.
3. **Deprecate `coords`, `geom`, `io.read_fast`, `read_image`, `read_table`,
   `read_subset`, `read_hdus`, `read_batch`, `read_large_table`,
   `_fastio.read`.** All become `DeprecationWarning`-emitting aliases for
   `torchsky.read(...)` with the relevant `mode=` / `subset=` kwarg. Keep
   `torchsky.stream_table` as `torchsky.io.scan`.
4. **Rename `torchsky.map.field.StrainerSkyEncoderMLPField` and the 21 sibling
   classes to start with `_`** (private), and document `make_field_model("...")`
   as the only public path. Keep the old names as deprecated re-exports.
5. **Move `torchsky.wcs.healpix` to `torchsky.sphere.healpix`** (the smaller of
   the two renames toward `world/healpix.py`). Keep `torchsky.wcs.healpix` as a
   deprecated shim. This alone removes ~6 cross-pillar imports.
6. **Add `torchsky info` and `torchsky cat` CLI subcommands.** Both read-only.
   Both emit JSON with `--json`. Closes the biggest agent / human discoverability
   gap.
7. **Tighten `surfaces.py`.** Add `tier="internal"` for ~30 obvious internals
   (`map.operators.*`, `_io_engine.*`, `_BaseFieldModel`,
   `accumulate_image_on_*_adjoint`, `_alm_ell_array`, the SHT cache globals).
   Add a `tests/test_no_internal_in_core.py` gate that fails CI if a `core`
   surface imports an `internal` symbol.

These are all backward-compatible. They do not require Phase 2/3/4 to ship and can
land independently behind separate PRs.

---

## 6. Open questions for the maintainer

1. **Pillar names.** The proposed `io / world / scene / viz` is one option;
   `io / coords / model / viz` is another. `world` is unfamiliar to astronomers
   unless presented as "world coordinate system + on-sphere geometry" together.
   `model` is shorter than `scene` but overloads "ML model".
2. **Compat scope.** Should `torchsky.compat` be one namespace (with
   `compat.healpy`, `compat.astropy_io_fits`, `compat.fitsio`, …) or three
   separate top-level packages? One namespace is cleaner; three are easier to
   strip via build flags.
3. **Field model factory shape.** `scene.models.make(kind, ...)` is one design.
   The other is `scene.Scene(kind=..., **opts)` with `kind` selecting the
   internal architecture. The latter hides 22 classes behind one constructor and
   matches the mission better.
4. **CLI extras.** Is `torchsky show` worth implementing as a thin
   browser-launcher that delegates to `viz`, or as a one-shot PNG dumper, or both?
5. **Deprecation horizon.** A single minor release is the proposed window. The
   current `0.4.0.dev0` version means there is room to be aggressive (drop
   aliases at `0.5.0`) — but if any external consumer exists, two releases is
   safer.

---

## 7. Appendix: cross-cutting metrics

### 7.1 Public symbol counts

| Namespace                         | `__all__` size | Notes                                                |
|-----------------------------------|----------------|------------------------------------------------------|
| `torchsky` (root)                 | ~80            | Includes `read*`, `write`, transforms, dataset cls   |
| `torchsky.io`                     | 27             | Mostly aliases of root                               |
| `torchsky.table`                  | (no `__all__`) | ~30 public functions                                 |
| `torchsky.wcs`                    | 23             |                                                      |
| `torchsky.coords`                 | 45             | Re-exports of `wcs`                                  |
| `torchsky.geom`                   | 25             | Re-exports of `sphere`                               |
| `torchsky.sphere`                 | 79             | HEALPix + harmonics + regions + sparse + MOC         |
| `torchsky.sphere.compat`          | ~80            | healpy mirror                                        |
| `torchsky.map`                    | 152            | Three subsystems flat-exported                       |
| `torchsky.map.field`              | 84             |                                                      |
| `torchsky.map.operators`          | 77             |                                                      |
| `torchsky.encoders`               | 6              |                                                      |
| `torchsky.catalogs`               | 2              |                                                      |
| `torchsky.viz`                    | 3              |                                                      |
| `torchsky.spectral`               | (no `__all__`) | ~10 dataclasses + helpers                            |

Roughly **700 publicly listed symbols** at last count. The four-pillar plan reduces
this to ~120 user-visible symbols (with the rest still importable via fully-qualified
private paths).

### 7.2 Environment variables in active use

```text
TORCHSKY_DEBUG_SCALE                       (root __init__)
TORCHSKY_COLD_NOMMAP                       (root __init__)
TORCHSKY_COLD_NOCACHE                      (root __init__)
TORCHSKY_CFITSIO_CACHE_MB                  (root __init__)
TORCHSKY_CFITSIO_CACHE_FILES               (root __init__)
TORCHSKY_TABLE_HANDLE_CACHE                (table.py)
TORCHSKY_TABLE_HANDLE_CACHE_SIZE           (table.py)
TORCHSKY_TABLE_READER_CACHE                (table.py)
TORCHSKY_TABLE_READER_CACHE_SIZE           (table.py)
TORCHSKY_GEOM_PAIRWISE_CHUNK               (sphere/geom.py)
TORCHSKY_SPIN_MAX_CACHE_BYTES              (sphere/spectral.py)
TORCHSKY_SCALAR_MAX_CACHE_BYTES            (sphere/spectral.py)
TORCHSKY_SCALAR_RING_AUTO_MIN_BYTES        (sphere/spectral.py)
TORCHSKY_SCALAR_MAP2ALM_RECURRENCE_CPP     (sphere/spectral.py)
TORCHSKY_SCALAR_ALM2MAP_RECURRENCE_CPP     (sphere/spectral.py)
TORCHSKY_SPIN_RING_AUTO_MIN_BYTES          (sphere/spectral.py)
TORCHSKY_WCS_FUSED                         (wcs/core.py)
TORCHSKY_PERF_GATE_*                       (tests/test_perf_gate.py)
TORCHSKY_EXPERIMENTAL                      (proposed)
```

A single config object (`torchsky.config()`) exposing the same knobs by attribute,
backed by env vars, would cut the convention drift.

### 7.3 Eager vs lazy import audit

After Phase 1+6 changes, `import torchsky` should not transitively import:

- `pyarrow` (only via `torchsky.io.table`, `torchsky.io` only when used)
- `polars`, `pandas`, `duckdb` (interop)
- `fastapi`, `uvicorn` (only via `torchsky.viz.serve()`)
- `torch_frame` (only via `torchsky.io.frame`)
- `torchsky.world.harmonics.*` (only when needed)
- `torchsky.scene.models.*` (only when needed)

Currently all of `pyarrow`, `polars` (when present), and `torch_frame` (when present)
are pulled on cold import via `from .interop import` and `from .frame import` lines
in the root `__init__.py`.

