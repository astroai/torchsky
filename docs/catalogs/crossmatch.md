# Crossmatch

A **crossmatch** pairs rows of one catalog (the *left* table) with rows of
another (the *right* table) whose positions agree on the sky. Two rows match
when their **separation**, the great-circle angle between them, is no larger
than a threshold: a fixed **radius**, a per-row radius, or the sum of the two
rows' positional errors. torchsky returns the pairs as row indices plus
separations, so you gather any column of either table afterwards.

Use it to find counterparts across surveys or bands, to flag duplicates inside
one catalog, to group detections from several catalogs, or to join tables on
an integer key. It is a positional, deterministic match: it does not compute
association probabilities, propagate covariances, or split blends. When you
need a likelihood-ratio or Bayesian association, use the pairs from `find="all"`
as the candidate list and score them yourself.

## How It Works

Testing every left row against every right row costs \(O(NM)\). torchsky
sorts the right catalog once into **zones**, declination bands, and by right
ascension inside each zone (Gray, Szalay and Nieto-Santisteban 2006). The match
cap of a left row of radius \(r\) at declination \(\delta\) meets the zones
covering \([\delta - r, \delta + r]\). Within each zone it covers one contiguous
range of right ascension of half-width \(\arcsin(\sin r / \cos\delta)\): two
pieces at the 0/360° wrap, and the whole zone when the cap holds a pole. A binary
search finds each range. Only those candidates go through the exact test

\[
\theta = \operatorname{atan2}\left(\lVert \mathbf a \times \mathbf b \rVert,\; \mathbf a \cdot \mathbf b\right) \le r,
\]

where \(\mathbf a, \mathbf b\) are unit vectors. This form stays accurate at
sub-milliarcsecond separations, where the arccosine of a dot product does not.

```mermaid
flowchart LR
    R["right catalog"] -->|"sort by<br/>(zone, RA)"| I["SkyIndex"]
    L["left catalog<br/>(chunks)"] --> C["candidate ranges<br/>binary search"]
    I --> C
    C -->|"exact great-circle test"| P["pairs within threshold"]
    P -->|"find= best1, best2,<br/>mutual, symmetric, all"| M["CrossmatchResult"]
```

The cost is \(O((N + M)\log M + K)\) for \(K\) candidate pairs. Candidate
ranges are padded by a few float64 ulp, so they never drop a pair that passes
the exact test. The zone height changes the cost, never the result, and so do
chunk sizes, thread count and device. Build the index once with
[`SkyIndex`](../reference/catalogs.md#torchsky.catalogs.SkyIndex) and pass it
as `index=` to reuse it across calls.

**Assumptions.** Both tables are in the same frame at the same epoch; positions
are degrees and thresholds arcseconds; the threshold is a circle (no error
ellipses). **Validation.** Every `find` mode, `match_within`, `match_nearest` and
`count_within` is tested against a brute-force \(O(NM)\) search at the poles
(including exactly ±90°), across the RA wrap, in a crowded cluster, and with
per-row radii spanning five decades. Separations agree with Astropy `SkyCoord`
to about \(10^{-10}\) arcsec; see [parity](../performance/parity.md).

## Gaia, 2MASS and AllWISE Around M67

The [M67 example](../examples/index.md#gaia-dr3-x-2mass-x-allwise-in-the-field-of-m67)
fetches a 0.25° cone from three public TAP services, propagates Gaia DR3 to the
2MASS epoch, and matches the three catalogs. The core of it:

```python
# gaia and tmass are pyarrow tables fetched by
# examples/crossmatch_gaia_2mass_allwise.py (needs network on the first run).
import torch

from torchsky.catalogs import Catalog

def col(table, name):
    return torch.tensor(table[name].to_numpy(zero_copy_only=False), dtype=torch.float64)

# Two-parameter Gaia solutions have no proper motion; treat it as zero here.
optical = Catalog(
    lon=col(gaia, "ra"), lat=col(gaia, "dec"), epoch=2016.0,
    pm_lon_coslat_mas_yr=col(gaia, "pmra").nan_to_num(),
    pm_lat_mas_yr=col(gaia, "pmdec").nan_to_num(),
)
infrared = Catalog(lon=col(tmass, "ra"), lat=col(tmass, "dec"), epoch=2000.0)

pairs = optical.crossmatch(infrared, radius_arcsec=2.0, find="mutual", target_epoch=2000.0)
k_mag = col(tmass, "k_m")[pairs.right_index]
```

![Gaia x 2MASS colour-magnitude diagram and the 2MASS x AllWISE colour plane](../assets/gallery/examples/crossmatch_gaia_2mass_allwise.png)

On about 2,100 Gaia and 1,300 2MASS rows the `best1` and `mutual` matches take a
few milliseconds, with a median separation of 0.08 arcsec, and a brute-force
search on a subset reproduces the indexed result exactly.

## The Result

Every matcher returns a
[`CrossmatchResult`](../reference/catalogs.md#torchsky.catalogs.CrossmatchResult)
with `left_index`, `right_index`, and `separation_arcsec`. Indices refer to the
original input row order. Exact ties go to the lowest row index, so results do
not depend on chunking, threads or device. An empty input returns empty tensors;
positions must be finite.

<!-- docs: run -->
```python
import torch

from torchsky.catalogs import crossmatch_sky

left_ra = torch.tensor([10.0, 30.0], dtype=torch.float64)
left_dec = torch.zeros(2, dtype=torch.float64)
right_ra = torch.tensor([10.0001, 10.0002, 29.9998, 80.0], dtype=torch.float64)
right_dec = torch.zeros(4, dtype=torch.float64)

best = crossmatch_sky(left_ra, left_dec, right_ra, right_dec, radius_arcsec=2.0)
assert best.left_index.tolist() == [0, 1]
assert best.right_index.tolist() == [0, 2]

every = crossmatch_sky(left_ra, left_dec, right_ra, right_dec, radius_arcsec=2.0, find="all")
assert every.left_index.tolist() == [0, 0, 1]
assert every.right_index.tolist() == [0, 1, 2]
print(best.separation_arcsec)
```

## STILTS-style Matching

`find=` chooses which of the pairs inside the threshold are kept. The names
follow [STILTS](https://www.star.bris.ac.uk/~mbt/stilts/) `tmatch2`, except
that torchsky's `"best"` is STILTS `best1`.

| `find` | Keeps | One-to-one | STILTS |
|---|---|---|---|
| `"best"`, `"best1"` (default) | the nearest right row for each left row | no: right rows may repeat | `find=best1` |
| `"best2"` | the nearest left row for each right row | no: left rows may repeat | `find=best2` |
| `"mutual"` | pairs that are each other's nearest | yes | `find=best` |
| `"symmetric"` | greedy assignment by ascending separation | yes, and maximal | none |
| `"all"` | every pair inside the threshold | no | `find=all` |

`mutual` leaves a row unmatched when its nearest partner prefers another row;
`symmetric` then gives it its next-nearest free partner. `all` is ordered by
left index, separation, right index; the other modes by left, then right
index.

| torchsky | STILTS |
|---|---|
| `crossmatch_sky(..., radius_arcsec=r)` | `tmatch2 matcher=sky params=r` |
| `crossmatch_sky(..., left_err_arcsec=, right_err_arcsec=)` | `tmatch2 matcher=skyerr` (match if `sep <= err1 + err2`) |
| `crossmatch_sky(..., left_distance=, right_distance=, distance_tol=)` | `sky` plus a 1-d cut on distance |
| [`crossmatch_exact(key1, key2)`](../reference/catalogs.md#torchsky.catalogs.crossmatch_exact) | `tmatch2 matcher=exact` |
| [`join_rows(result, n1, n2, join)`](../reference/catalogs.md#torchsky.catalogs.join_rows) | `join=1and2`, `1or2`, `all1`, `all2`, `1not2`, `2not1` |
| [`crossmatch_self(..., group=True)`](../reference/catalogs.md#torchsky.catalogs.crossmatch_self) | `tmatch1 action=identify` |
| [`crossmatch_many(cats, r, mode="pairs")`](../reference/catalogs.md#torchsky.catalogs.crossmatch_many) / `"group"` | `tmatchn multimode=pairs` / `group` |

Elliptical errors (`skyellipse`) and error-normalised ranking are not provided:
ranking always uses the separation in arcseconds.

<!-- docs: run -->
```python
import torch

from torchsky.catalogs import crossmatch_exact, crossmatch_many, crossmatch_self, crossmatch_sky, join_rows

d = torch.float64
ra1 = torch.tensor([10.0, 10.0, 359.9999, 50.0], dtype=d)
ra2 = torch.tensor([10.0002, 0.0001, 80.0], dtype=d)
z1, z2 = torch.zeros(4, dtype=d), torch.zeros(3, dtype=d)

sym = crossmatch_sky(ra1, z1, ra2, z2, radius_arcsec=2.0, find="symmetric")
assert sym.left_index.tolist() == [0, 2] and sym.right_index.tolist() == [0, 1]
left, right = join_rows(sym, 4, 3, "1or2")
assert left.tolist() == [0, 2, 1, 3, -1] and right.tolist() == [0, 1, -1, -1, 2]

# The three pairs are 0.72" apart: 0.5" + 0.3" accepts them, 0.3" + 0.3" does not.
near = crossmatch_sky(ra1, z1, ra2, z2, left_err_arcsec=0.5, right_err_arcsec=0.3, find="all")
assert near.left_index.tolist() == [0, 1, 2]
far = crossmatch_sky(ra1, z1, ra2, z2, left_err_arcsec=0.3, right_err_arcsec=0.3, find="all")
assert far.left_index.numel() == 0

own = crossmatch_self(ra1, z1, 2.0, group=True)
assert own.group_id.tolist() == [0, 0, -1, -1] and own.group_size.tolist() == [2, 2, 1, 1]

keys = crossmatch_exact(torch.tensor([1, 2, 2]), torch.tensor([2, 3]))
assert keys.left_index.tolist() == [1, 2] and keys.right_index.tolist() == [0, 0]

groups = crossmatch_many([(ra1, z1), (ra2, z2)], 2.0, mode="group")
assert groups.index.tolist() == [[0, 0], [2, 1], [3, -1], [-1, 2]]
```

## Recipes

### Match with per-source errors

Pass `left_err_arcsec` and `right_err_arcsec` (scalars or per-row tensors,
`>= 0`, either side optional) instead of `radius_arcsec`. A pair matches when
`separation <= err_left[i] + err_right[j]`. Scale catalog errors to the
confidence you want, for example three times the 1σ position error, as the M67
example does. `Catalog.crossmatch` also accepts a column name for either side.

### Keep unmatched rows: joins

`join_rows(result, n_left, n_right, join)` turns pairs into two index tensors
with `-1` for the absent side: matched pairs first, then unmatched left rows,
then unmatched right rows. Use `"all1"` for a left outer join and `"1not2"` for
the left rows without a counterpart.

### Find duplicates and groups in one catalog

`crossmatch_self(ra, dec, radius)` returns each pair `i < j` once. With
`group=True` it also links rows into **friends-of-friends** groups (connected
components of the pair graph): `group_id` is `-1` for rows without a partner and
`0..n_groups-1` otherwise, numbered by smallest member row; `group_size` is 1
for singletons. Chains can join rows farther apart than the radius.

### Combine more than two catalogs

`crossmatch_many(catalogs, radius, mode=...)` returns one output row per group
and `index[g, c]`, the row of catalog `c` (`-1` if absent).
`mode="pairs"` takes the first catalog as reference and gives each reference row
its `best1` partner in every other catalog; unclaimed rows follow alone.
`mode="group"` links all cross-catalog pairs friends-of-friends style and keeps,
per catalog, the member with the smallest separation to the rest of the group;
`group_size` counts every member.

### Match catalogs at different epochs

Stars move. For catalogs observed years apart, propagate positions to one epoch
before matching: `Catalog.crossmatch(..., target_epoch=2000.0)` moves each row
with the richest motion it has. Rows with proper motion, positive parallax and
radial velocity get full space motion. Rows with proper motion only get linear
proper motion; a non-positive parallax is then ignored, never "fixed". Rows
without proper motion need an explicit
[`MotionPrior`](../reference/catalogs.md#torchsky.catalogs.MotionPrior), for
example the Gaia proper-motion distribution of their sky cell. Missing motion
is never treated as zero, and the call raises instead. See
[Frames and time](../coordinates/frames.md) and
[Catalog containers](containers.md#frames-epochs-and-motion).

### Positional uncertainties

[`Astrometry`](../reference/catalogs.md#torchsky.catalogs.Astrometry) holds one
6x6 covariance per row in the local east/north basis: position, proper motion,
parallax and radial velocity, in mas and mas/yr. Build it from Gaia's archive
columns (`Astrometry.from_gaia`), a 2MASS-style error ellipse
(`from_ellipse`) or plain errors (`from_errors`). Quantities a catalogue does
not measure stay NaN, never zero.
[`propagate_astrometry`](../reference/catalogs.md#torchsky.catalogs.propagate_astrometry)
returns positions and the position covariance at another epoch, \(J C J^T\)
with the exact motion Jacobian. For rows without proper motion the covariance
grows by \(\Delta t^2\) times the prior's covariance.

<!-- docs: run -->
```python
import torch

from torchsky.catalogs import Astrometry, propagate_astrometry
from torchsky.coords import CatalogState

state = CatalogState(
    lon=torch.tensor([269.45]), lat=torch.tensor([4.69]), epoch=2016.0,
    pm_lon_coslat_mas_yr=torch.tensor([-798.6]), pm_lat_mas_yr=torch.tensor([10328.1]),
)
errors = Astrometry.from_gaia({
    "ra_error": torch.tensor([0.03]), "dec_error": torch.tensor([0.03]),
    "parallax_error": torch.tensor([0.04]), "pmra_error": torch.tensor([0.04]),
    "pmdec_error": torch.tensor([0.04]),
    **{f"{a}_{b}_corr": torch.tensor([0.0]) for a, b in [
        ("ra", "dec"), ("ra", "parallax"), ("ra", "pmra"), ("ra", "pmdec"),
        ("dec", "parallax"), ("dec", "pmra"), ("dec", "pmdec"),
        ("parallax", "pmra"), ("parallax", "pmdec"), ("pmra", "pmdec")]},
})
moved, cov = propagate_astrometry(state, errors, 2000.0)
sigma = cov[0].diagonal().sqrt()  # mas: 16 yr of 0.04 mas/yr pm error on 0.03 mas
assert torch.allclose(sigma, torch.full((2,), (0.03**2 + (16 * 0.04) ** 2) ** 0.5, dtype=torch.float64), rtol=1e-3)
```

### Probabilistic counterparts

[`associate`](../reference/catalogs.md#torchsky.catalogs.associate) goes beyond a
fixed radius. It decides counterparts from each pair's uncertainties and the
background density:

1. **Common epoch.** Both catalogs move to one epoch with their covariances
   (`propagate_astrometry`).
2. **Candidates.** A pair is a candidate when its Mahalanobis distance passes a
   \(\chi^2_2\) gate: \(d^2 = \Delta^T (C_l + C_r)^{-1} \Delta \le k^2\), where \(k^2\) is the
   \(1-\alpha\) quantile for `gate_pvalue` \(=\alpha\). Search radii follow from the
   covariances, so no pair inside the gate is ever missed. \(\Delta\) and both
   covariances are expressed at the pair's midpoint (parallel transport), which
   is exact at the poles.
3. **Likelihood ratio.** For each candidate,
   \(\mathrm{LR} = \mathcal N(\Delta; 0, \Sigma)/\nu\), where \(\nu\) is the right
   catalog's surface density over its footprint (a `MOC`; pass the survey footprint).
   Optional photometric terms (`HistogramRatio`) multiply it.
4. **Posterior.** \(P_{ij} = \pi\,\mathrm{LR}_{ij} / ((1-\pi) + \pi \sum_k \mathrm{LR}_{ik})\).
   `completeness="em"` fits \(\pi\).
5. **Assignment.** `assign="optimal"` gives the exact one-to-one assignment that
   maximises \(\sum \log(P_{ij}/P_{i0})\).

<!-- docs: run -->
```python
import torch

from torchsky.catalogs import Astrometry, Catalog, associate

opt = Catalog(lon=torch.tensor([10.0, 10.01]), lat=torch.tensor([0.0, 0.0]), epoch=2016.0,
              astrometry=Astrometry.from_errors(torch.tensor([50.0, 50.0]), torch.tensor([50.0, 50.0])))
ir = Catalog(lon=torch.tensor([10.00003, 10.01, 10.02]), lat=torch.tensor([0.0, 0.00002, 0.0]), epoch=2016.0,
             astrometry=Astrometry.from_ellipse(torch.tensor([300.0] * 3), torch.tensor([100.0] * 3),
                                                torch.tensor([90.0] * 3)))
res = associate(opt, ir, completeness=0.9, max_radius_arcsec=5.0)
assert res.left_index[res.match].tolist() == [0, 1] and res.right_index[res.match].tolist() == [0, 1]
assert torch.all(res.posterior[res.match] > 0.9)
```

### Several catalogs at once

[`associate_many`](../reference/catalogs.md#torchsky.catalogs.associate_many)
scores, for each primary row, every combination of one gated counterpart (or
none) per other catalog. Its weight is the exact Gaussian likelihood that all
present positions come from one true position (Budavári & Szalay 2008, with
full covariances), times each catalog's completeness and inverse density. The
result lists the hypotheses per primary row with posteriors, `p_any` (any
counterpart) and the MAP hypothesis. Primary rows with more than `budget`
combinations are enumerated best-first by an upper bound, and the posterior
mass left out is certified (`residual_mass` \(\le\) `mass_tol`). Nothing is
truncated silently.

### Per-row radii, nearest neighbours and counts

`radius_arcsec` may be a tensor with one radius per left row. Each row's window
uses its own radius, so one large radius does not slow the others.
[`match_within`](../reference/catalogs.md#torchsky.catalogs.match_within) returns
every pair. With `right_radius_arcsec=` it applies a two-sided threshold
`combine(r_left[i], r_right[j])`, with `combine` one of `"sum"` (STILTS `skyerr`),
`"quadrature"` or `"max"`.
[`match_nearest`](../reference/catalogs.md#torchsky.catalogs.match_nearest)
returns the exact `k` nearest right rows per left row, optionally capped by
`max_radius_arcsec`. [`count_within`](../reference/catalogs.md#torchsky.catalogs.count_within)
counts the neighbours of each row. All three accept a prebuilt
[`SkyIndex`](../reference/catalogs.md#torchsky.catalogs.SkyIndex) as `index=`,
and the index has the same three queries as methods.

<!-- docs: run -->
```python
import torch

from torchsky.catalogs import SkyIndex, count_within, match_nearest, match_within

left_ra = torch.tensor([10.0, 10.0], dtype=torch.float64)
left_dec = torch.tensor([0.0, 0.001], dtype=torch.float64)
right_ra = torch.tensor([10.0, 10.0002, 50.0], dtype=torch.float64)
right_dec = torch.zeros(3, dtype=torch.float64)

per_row = match_within(left_ra, left_dec, right_ra, right_dec, torch.tensor([1.0, 0.5]))
assert per_row.left_index.tolist() == [0, 0]

nearest = match_nearest(left_ra, left_dec, right_ra, right_dec, k=2)
assert nearest.index.tolist() == [[0, 1], [0, 1]]

index = SkyIndex.build(right_ra, right_dec)
assert count_within(left_ra, left_dec, right_ra, right_dec, 4.0, index=index).tolist() == [2, 2]
assert index.query_nearest(left_ra, left_dec, 1, max_radius_arcsec=1.0).index.tolist() == [[0], [-1]]
```

`match_nearest` is exact without a radius: search radii double from a guess set
by the mean density until each row has `k` neighbours, so dense fields resolve
in the first rounds and sparse targets far from a small catalog still find
their neighbours.

### Coming from Astropy or SciPy

| Astropy / SciPy | torchsky |
|---|---|
| `SkyCoord.match_to_catalog_sky(c2, nthneighbor=n)` | `match_nearest(..., k=n).index[:, n - 1]` |
| `search_around_sky(c1, c2, seplimit)` | `match_within(..., radius_arcsec)` |
| `cKDTree(xyz).query(xyz, k, distance_upper_bound=chord)` | `match_nearest(..., k, max_radius_arcsec=r)` |
| `cKDTree(xyz).query_ball_point(xyz, chord)` | `match_within(..., radius_arcsec=r)` |
| `query_ball_point(..., return_length=True)` | `count_within(...)` |

Radii are angles in arcseconds rather than chord lengths, inputs are degrees,
and results never depend on the order of the catalogue on disk: ties go to the
lowest row index.

### Scale to large catalogs and other devices

- `left_chunk_size` and `right_chunk_size` bound the working set; they never
  change the result.
- `device="cuda"` moves the inputs and returns results there. Indices agree
  with the CPU; separations agree to floating-point tolerance. On MPS positions
  are float32 (about 0.1 arcsec near RA 360°), so smaller radii are not
  meaningful there.
- For catalogs larger than memory, read them in sky partitions (HEALPix or
  HATS tiles) with torchfits, pyarrow or polars, and let
  [`PartitionPlan`](../reference/catalogs.md#torchsky.catalogs.PartitionPlan)
  decide what to read. `plan.owner(lon, lat)` assigns rows to cells (order-29
  ranges, the HATS `_healpix_29` convention), and `margin_arcsec(radius, ...)`
  bounds how far a counterpart can be: the radius, plus errors, plus motion
  over the epoch gap. `left_plan.neighbors(margin, right_plan)` lists the
  right cells that can hold counterparts, and `left_plan.in_margin(p, ...)` keeps
  the right rows that can reach cell `p`. Match each cell's owned left rows
  against those right rows; the union over cells equals the single-pass
  result for `all`, `best` and `match_nearest`, with no de-duplication.
- The [million-row example](../published-examples/crossmatch_million.py)
  matches \(10^6 \times 10^6\) rows in about 0.5 s per mode on a CPU.

## Pitfalls

- **`best` is not one-to-one.** Two left rows can pick the same right row. Use
  `mutual` or `symmetric` when each source may be used once.
- **Order of arguments matters** for `best1`, `best2`, and `pairs`: the left or
  reference table decides who chooses.
- **Units.** Match radii are arcseconds; cone queries take degrees.
- **Wrap-around and poles** need no special handling, but positions must be
  finite; drop or mask NaN rows first.
- **Crowded fields.** With a radius larger than the typical source spacing,
  `best` picks the nearest neighbour, which may be the wrong source. Inspect
  `find="all"` multiplicities before trusting a one-to-one result.

## See Also

- [Catalog containers](containers.md): `Catalog`, cone queries, Arrow input.
- [Frames and time](../coordinates/frames.md): frames and proper motion.
- Reference: [`torchsky.catalogs`](../reference/catalogs.md).
- Examples: [Gaia, 2MASS and AllWISE](../published-examples/crossmatch_gaia_2mass_allwise.py),
  [a million-row crossmatch](../published-examples/crossmatch_million.py).
- Measurements: [benchmarks](../performance/benchmarks.md#catalogs) and
  [parity](../performance/parity.md).
