# Catalog API reference

`torchsky.catalogs` is the sky-domain catalog layer: it owns indexing,
cone queries, crossmatching, and streaming datasets. Table parsing belongs to
[`torchfits.table`](table.md); this namespace consumes tensors and Arrow
batches.

---

## Catalog model

```python
from torchsky.catalogs import Catalog

catalog = Catalog(lon_deg, lat_deg)
```

| Name | Contract |
|---|---|
| `Catalog` | Batched sky rows on one device: `lon`, `lat`, optional proper motion, parallax, radial velocity, validity mask, and extra tensor columns. |
| `CatalogStreamDataset` | Wrap a partition iterator into a `torch.utils.data.IterableDataset` of `Catalog` batches. |
| `CrossmatchResult` | `left_index`, `right_index`, and `separation_arcsec` for matched pairs. |
| `HealpixIndex` | Bucketed NESTED HEALPix index over catalog rows; the engine behind crossmatch and cone queries. |

`Catalog` stores spherical coordinates in degrees and keeps frames/epochs
explicit (`frame`, `epoch`). There is no implicit coordinate conversion in a
crossmatch: pass catalogs already in the same frame and epoch.

---

## Crossmatching

| Name | Contract |
|---|---|
| `crossmatch(left, right, radius_arcsec, *, find="best")` | Unified entry point. `find="best"` returns one best match per left row; `find="all"` returns every pair within the radius. |
| `crossmatch_sky(...)` | Chunked all-pairs crossmatch over two in-memory tiles. |
| `crossmatch_sky_from_arrow(...)` | Crossmatch Arrow inputs by materializing HEALPix tiles. |
| `crossmatch_sky_streamed(...)` | Stream the left catalog in batches; the right side is indexed once. |

```python
from torchsky.catalogs import crossmatch

matched = crossmatch(catalog_a, catalog_b, radius_arcsec=1.0, find="best")
print(matched.left_index[:5], matched.separation_arcsec.max())
```

Chunking and the HEALPix index bound memory; the index resolution is chosen from
the search radius by `nside_for_cap`.

---

## Indexing and coverage helpers

| Name | Contract |
|---|---|
| `cap_covering_pixels(nside, lon, lat, radius_rad)` | HEALPix pixels whose centers lie in a spherical cap. |
| `nside_for_cap(radius_rad)` | Coarsest power-of-two NSIDE whose pixel radius is small enough for the cap. |

These are the primitives a partitioner uses to assign catalog rows to HEALPix
tiles; they are exposed so survey-specific partition schemes can reuse the same
bounds.

---

## Known-source photometry and deblending

These APIs fit a caller-supplied source list. They do not detect sources,
create components, or change component identity.

| Name | Contract |
|---|---|
| `SourceGrouping` | Deterministic connected blend groups built from supplied source supports and the existing sky index. |
| `build_source_groups(...)` | Build bounded source/exposure overlap groups without a global pixels-by-sources matrix. |
| `PhotometrySolver` | Select signed weighted least squares or nonnegative flux fitting. |
| `BackgroundModel` | Declare the local nuisance-background columns included in each group fit. |
| `BlockSparseCovariance` | Conditional flux covariance stored by independent blend group. |
| `BlendFitDiagnostics` | Rank, conditioning, convergence, residual, and active-set diagnostics for forced photometry. |
| `ForwardPhotometryResult` | Catalog coefficients, predictions, residuals, covariance, groups, diagnostics, and provenance. |
| `fit_forced_photometry(...)` | Fit supplied point/template components through native WCS, PSF, response, and noise operators. |
| `FailurePolicy` | Choose fail-fast or diagnostic retention when a supplied component model is inadequate. |
| `KnownComponentFitError` | Raised when a known-component fit cannot satisfy its declared contracts. |
| `FitAdequacyDiagnostics` | Held-out likelihood, residual structure, and model-adequacy evidence without source creation. |
| `ParametricBounds` | Bounds for supplied point, Gaussian, or Sérsic position and shape refinement. |
| `ParametricBlendDiagnostics` | LM/trust-region, variable-projection, rank, and component-swap diagnostics. |
| `ParametricDeblendResult` | Fixed-component parametric scene, fluxes, covariance, predictions, and residuals. |
| `fit_parametric_deblending(...)` | Refine supplied component positions/shapes while projecting linear flux/background parameters. |
| `FactorizationBlendDiagnostics` | Constraint, convergence, rank, and swap diagnostics for morphology-spectrum factorization. |
| `MorphologySpectrumDeblendResult` | Fixed-component nonnegative morphology×spectrum factors and reconstructed observations. |
| `fit_morphology_spectrum_deblending(...)` | Fit supplied components with optional caller-requested monotonic or symmetric morphology constraints. |

```python
from torchsky.catalogs import fit_forced_photometry

result = fit_forced_photometry(observations, supplied_scene)
flux = result.catalog
```

All uncertainty and residual products are conditional on the supplied component
set. A failed adequacy check is reported explicitly; it never triggers hidden
source discovery or component birth/death.

---

See also: [catalogs guide](catalogs.md), [catalogue foundations](catalogue-foundations.md),
[HEALPix reference](api-sphere.md), and the [table boundary](table.md).
