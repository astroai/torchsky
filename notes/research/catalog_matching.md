# Catalog matching: exact covering vs approximate neighbors

Identification of counterparts needs **recall = 1** under a stated metric.
Three communities solved “match points at scale” independently. Steal the
**join algorithm** (partition, prune, exact test). Do not steal their
dataframes, and do not put approximate nearest-neighbor (ANN) indexes on the
spatial join.

torchsky owns the spherical covering and the tensor predicate.
[xmatch](https://github.com/sfabbro/xmatch) and torchfits own Polars / Arrow /
HATS / FITS I/O. `Catalog` is a device tensor batch, not a dataframe.
LanceDB is not a HATS replacement.

## 1. Metric and exactness

On the unit sphere the geodesic is great-circle angle. torchsky uses a stable
atan2 form (`great_circle_distance`): $\theta = \mathrm{atan2}(\|a\times b\|, a\cdot b)$.

Euclidean $k$NN in $\mathbb{R}^3$ on unit vectors is an **exact** spherical-cap
query if and only if the chord radius is $2\sin(\theta/2)$. That is the
xmatch `fast` engine (scipy `cKDTree`) and the same lemma GIS uses on ECEF /
S2. Latitude–longitude kd-trees that ignore RA wrap and the poles are wrong
in astronomy and in GIS.

A covering is **sound** when every true pair survives the prune. HEALPix
pixels have maximum angular radius $\rho$. A pair with $\theta\le r$ can sit
in pixels whose centers are at most $r+2\rho$ apart. Candidate pixels are
that cap; the exact GC (or Mahalanobis) test is applied to the surviving
rows. Dropping a true pair here is a bug, not a speedup.

## 2. Astronomy spatial joins

HEALPix (Górski et al. 2005, *ApJ* 622, 759) is the covering torchsky uses.
The same partition-then-refine pattern appears in CDS X-Match, xmatch `zone`,
and HATS pixel walks (`hats_native.py`). HTM, Q3C, and LSST sphgeom are
other coverings, not a second public API.

Per-row positional error (`skyerr`) must cover **that row’s** radius. A
global `nanmax` radius is only a candidate-generation hack; the exact test
is still pair-wise. Mahalanobis / χ-match (Pineau et al. 2017, *A&A* 597,
A89) is the founded 2-D Gaussian after the scalar cap is sound. Mixed 6D vs
angular proper motion stays the existing xmatch/torchsky contract: never
impute parallax to force 6D (Budavári & Szalay 2008, *ApJ* 679, 301, for
probabilistic scoring, not for the covering).

FoF / union-find already lives in xmatch (Huchra & Geller 1982). Graph neural
nets (Kipf & Welling 2017) stay research.

STMOC covers time windows. Spectral or SED similarity is a **different
metric**, not a HEALPix query. Incremental LSST-alert matching is a streaming
consumer of the same covering.

## 3. Geospatial

The join is the same; the Earth is not.

- Google **S2** and Uber **H3**: spherical (quasi-)equal-area cells; neighbor
  rings are HEALPix $k$-rings. H3 is hexagonal; S2 is Hilbert-ordered quads.
  Neither replaces HEALPix for all-sky equal-area science maps.
- **GEOS** STRtree and PostGIS `&&` then `ST_DWithin`: envelope prune, then
  exact predicate. Analog of HEALPix tile + GC.
- **GeoParquet / GeoArrow**, Apache **Sedona**, DuckDB spatial: columnar
  scan and partition. That is xmatch/torchfits, not torchsky.
- Dual-tree spatial joins (Gray & Moore; Ram, Lee, Silberstein, Deshpande)
  are the CS name for “a tree on both sides.”

Earth positions live on an **ellipsoid**. Catalogues of stars are a **unit
sphere at infinity**. Do not import WGS84 geodesics as catalog separation.

## 4. Point clouds ($\mathbb{R}^3$)

Closest non-astro literature, with public **recall** benchmarks:

- Exact vs approximate $k$NN: nanoflann, FLANN, PCL `KdTreeFLANN`.
- Registration (pose, not counterpart ID): ICP, FGR, TEASER++; Open3D. The
  prune-then-exact inner loop is the same; the scientific claim is not.
- Benchmarks: KITTI odometry, 3DMatch / ScanNet, ModelNet. Report
  recall@radius, not wall time alone — the same discipline as catalog
  pair-ID parity.

These libraries assume Euclidean $L^2$ in a local frame. Catalogues need
$L^2$ on the unit sphere or GC. Open3D/PCL are not dependencies.

## 5. High-D ANN (photometry / SED / spectra — not sky)

[ann-benchmarks](https://github.com/erikbern/ann-benchmarks) (Aumüller,
Bernhardsson, Faithfull) and BIGANN plot recall–QPS for HNSW, FAISS IVF-PQ,
ScaNN, DiskANN, NGT, USearch. **Recall $<1$ is the product.** That cannot
back a claimed astrometric identification.

Optional later extra for “similar SEDs,” never the spatial join.

**LanceDB** stores Arrow on disk with ANN indexes (IVF/PQ and similar). Same
category as FAISS: optional photometric neighbors. It is not a dataframe, not
a HATS replacement, and not a core runtime dependency. HATS Parquet already
partitions the sky.

## What torchsky implements

- `nside_for_cap(radius)` — covering-derived HEALPix NSIDE, not a catalog-size
  lookup table.
- `cap_covering_pixels` — sound inclusive pixel set for a spherical cap.
- `chord_radius` — $2\sin(\theta/2)$ for unit-sphere Euclidean prune.
- `crossmatch_sky` — exact GC on one in-memory tile; per-row radii honored
  in the covering, not only after the fact. Small caps gather batched nested
  neighbour rings (no Python loop over left pixels).

I/O, spill, HATS walks, join types, Bayes/LR/AUF stay in xmatch.
