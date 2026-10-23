# Migrating from Astropy, healpy, and reproject

`torchsky` is not a drop-in replacement for the legacy astronomy stack. It is a
tensor-native implementation of the same sky-domain operations, with explicit
inputs, explicit units, and no hidden file I/O. This page maps the common
migration paths and, just as importantly, states what does **not** carry over.

Read [compatibility.md](compatibility.md) for the ownership boundary: `torchfits`
owns FITS files and tables, `torchsky` owns coordinates, geometry, maps, and
models, and Astropy/healpy/reproject are references and oracles rather than
runtime backends.

## At a glance

| Legacy call | torchsky replacement | Notes |
|---|---|---|
| `astropy.wcs.WCS(header)` | [`torchsky.wcs.WCS`](reference/wcs.md#torchsky.wcs.WCS) | Header stays a plain dict; pixels are 0-based by default. |
| `wcs.pixel_to_world(x, y)` | `wcs.pixel_to_world(x, y)` | Tensors in, tensors out; dispatch on the header's axes. |
| `SkyCoord(ra, dec, frame=...)` | [`CatalogState`](reference/coords.md#torchsky.coords.CatalogState) | Batched tensors, explicit missing components. |
| `SkyCoord.transform_to(frame)` | [`transform`](reference/coords.md#torchsky.coords.transform) / `CatalogState.to_frame` | No `SkyCoord` graph; observed place needs Earth orientation. |
| `healpy.ang2pix(...)` | [`torchsky.sphere.ang2pix`](reference/sphere.md#torchsky.sphere.ang2pix) | Same arguments and defaults, tensor values. |
| `healpy.query_disc(...)` | [`torchsky.sphere.query_disc`](reference/sphere.md#torchsky.sphere.query_disc) | Takes a unit 3-vector, returns pixel indices. |
| `reproject.reproject_interp(...)` | [`torchsky.map.resample`](reference/map.md#torchsky.map.resample) | Operates on `Observation` objects and a target grid. |
| `astropy.io.fits` | [`torchfits`](io.md) | Never imported by torchsky. |
| `astropy.table.Table` | [`torchfits.table`](table.md) | Arrow/table batches; torchsky consumes tensors. |
| `astropy.cosmology` | [`torchsky.cosmology`](reference/cosmology.md) | Differentiable FLRW container, no parameter priors. |
| `astropy extinction laws` | [`torchsky.extinction`](reference/extinction.md) | Unit-explicit dust laws on tensors. |
| `specutils` Doppler/rebin | [`torchsky.spectral`](spectral.md) | Axis objects and flux-conserving rebin. |

## WCS

The structural differences are small and deliberate:

- **Header source.** Build a `WCS` from a `torchfits` header mapping. Torchsky
  never opens the file. See [the I/O boundary](io.md).
- **Pixel origin.** `pixel_to_world` and `world_to_pixel` use 0-based pixels.
  Use `pixel_to_world_values(..., origin=1)` when porting code that carries the
  FITS 1-based convention.
- **Axis dispatch.** A header with celestial, spectral, and time axes returns
  one tensor per axis; a pure spectral header returns the world array directly.
- **Distortions.** SIP, TPV, paper-IV lookup tables, and coupled `-TAB` axes are
  part of the same object; no separate `all_pix2world` bookkeeping.
- **Devices.** Move the WCS and its cached tables with `wcs.to("cuda")`; pass
  tensors on the device you want.

<!-- docs: run -->
```python
import torch
from torchsky.wcs import WCS

header = {
    "NAXIS": 2,
    "NAXIS1": 256,
    "NAXIS2": 256,
    "CTYPE1": "RA---TAN",
    "CTYPE2": "DEC--TAN",
    "CRPIX1": 128.5,
    "CRPIX2": 128.5,
    "CRVAL1": 210.0,
    "CRVAL2": 54.0,
    "CDELT1": -1.0 / 3600.0,
    "CDELT2": 1.0 / 3600.0,
}

wcs = WCS(header)
x = torch.tensor([100.0, 128.0, 150.0], dtype=torch.float64)
y = torch.tensor([90.0, 128.0, 140.0], dtype=torch.float64)

ra, dec = wcs.pixel_to_world(x, y)
x_back, y_back = wcs.world_to_pixel(ra, dec)

assert torch.allclose(x_back, x, atol=1e-8)
assert torch.allclose(y_back, y, atol=1e-8)
print(f"round trip max error: {float((x_back - x).abs().max()):.2e} px")
```

The [WCS guide](wcs.md) covers `-TAB`, coupled tables, GWCS sampling, and
astrometric calibration. The full surface is in the
[WCS reference](reference/wcs.md).

## SkyCoord-style catalog geometry

There is no `SkyCoord` class, and that is intentional: a catalog is a batch of
tensors, not a graph of representation objects.

- [`CatalogState`](reference/coords.md#torchsky.coords.CatalogState)
  holds `lon`, `lat`, optional proper motion, parallax, radial velocity, epoch,
  frame, and a `valid` mask. Missing values stay `None`.
- `convert_celestial` and `transform` handle geometric frames; `CatalogState.to_frame`
  propagates proper motion through rigid rotations.
- Observed place (`ALTAZ`, `ITRS`, `GCRS`, `CIRS`, `TETE`) requires an explicit
  `EarthOrientation` (DUT1, optional polar motion) and site. IERS values are
  never inferred.
- `pmsafe` distance override, unit-bearing `Quantity` objects, and automatic
  epoch inference do not exist. Supply tensors in the documented units.

<!-- docs: run -->
```python
import torch
from torchsky.coords import CatalogState, convert_celestial

lon = torch.tensor([266.405, 83.822], dtype=torch.float64)
lat = torch.tensor([-28.936, -5.391], dtype=torch.float64)

gal_lon, gal_lat = convert_celestial(
    lon, lat, from_frame="ICRS", to_frame="galactic"
)
state = CatalogState(lon=lon, lat=lat, frame="ICRS").to_frame("galactic")

assert torch.allclose(state.lon, gal_lon, atol=1e-9)
assert torch.allclose(state.lat, gal_lat, atol=1e-9)
print(f"galactic longitude: {state.lon.tolist()}")
```

Frame names follow FITS/AST conventions (`ICRS`, `GALACTIC`, `FK5`, `FK4`,
`ECLIPTIC`, `GALACTOCENTRIC`, `ALTAZ`, …). Conversion between Cartesian
Galactocentric coordinates and ICRS has its own helpers; it is not a
`convert_celestial` direction. See [catalog coordinates](coordinates.md).

## HEALPix and healpy

`torchsky.sphere` reuses healpy's naming and argument order, including the
`nest` and `lonlat` switches, so most calls port by changing the import:

```python
import healpy as hp
pix = hp.ang2pix(64, lon_deg, lat_deg, lonlat=True)

import torchsky.sphere as sphere
pix = sphere.ang2pix(64, lon_deg, lat_deg, lonlat=True)
```

Differences that matter:

- Values are `torch.Tensor`; pass tensors, not lists, in hot paths.
- `query_disc` and related queries take a unit 3-vector, not `(lon, lat)`.
- `UNSEEN` exists as a constant, but map file reading/writing belongs to
  `torchfits`, not the sphere package.
- `nest=False` (RING) remains the default everywhere.
- The healpy-shaped namespace is available as `torchsky.sphere.compat` and as
  the alias `torchsky.sphere.healpy`; new code can use `torchsky.sphere`
  directly. See the [compatibility reference](reference/sphere-compat.md).
- SHTs, smoothing, and spectra follow torchsky's own backend contract; review
  [sphere.md](sphere.md) before assuming healpy's precision or threading model.

<!-- docs: run -->
```python
import torch
import torchsky.sphere as sphere

nside = 64
lon = torch.tensor([10.0, 120.0, 250.0], dtype=torch.float64)
lat = torch.tensor([5.0, -30.0, 60.0], dtype=torch.float64)

pix = sphere.ang2pix(nside, lon, lat, lonlat=True)
lon_back, lat_back = sphere.pix2ang(nside, pix, lonlat=True)
# pix2ang returns the pixel centre, so round trip through ang2pix instead of
# comparing the centre to the original off-centre coordinate.
assert torch.equal(sphere.ang2pix(nside, lon_back, lat_back, lonlat=True), pix)

vec = sphere.lonlat_to_unit_xyz(lon[0], lat[0])
inside = sphere.query_disc(nside, vec, radius=0.05)
assert bool((inside == pix[0]).any())

print(f"pixel indices: {pix.tolist()}")
```

## Reprojection and coaddition

`reproject` maps arrays between WCS grids. `torchsky.map` goes further: it takes
observations with explicit weight, mask, DQ, and spectral metadata and returns a
[`MapResult`](reference/map.md#torchsky.map.MapResult) that carries data,
weight, header, and provenance.

- Use [`resample`](reference/map.md#torchsky.map.resample) to project
  observations onto a target grid without stacking.
- Use [`coadd`](reference/map.md#torchsky.map.coadd) to stack with a declared
  kernel, pixfrac, stacking method, and value semantics.
- Pass a reference FITS WCS/shape or a planned target; the target is never
  inferred silently when several observations disagree.
- Flux versus surface brightness is an explicit `ResampleConfig.semantics`
  choice; there is no implicit area scaling.

See [mapmaking](map.md) for the value-semantics contract and
[HEALPix maps](healpix-maps.md) for sphere-to-grid direction.

## FITS, tables, cosmology, and spectra

- **FITS.** Replace `astropy.io.fits` with `torchfits` reads and writes;
  `torchsky` consumes the resulting tensor and header. Never import
  `astropy.io.fits` in torchsky code. See [I/O boundary](io.md).
- **Tables.** Replace `astropy.table.Table` with `torchfits.table` Arrow or
  tensor reads; `torchsky.catalogs` accepts those batches. See
  [tables](table.md).
- **Cosmology.** Replace `astropy.cosmology` objects with
  [`FLRW`](reference/cosmology.md#torchsky.cosmology.FLRW) and the module
  functions; parameters are plain tensors, so they can be differentiated.
- **Dust.** Replace extinction-law calls with
  [`torchsky.extinction`](reference/extinction.md); wavelength units are
  explicit.
- **Spectra.** Replace `specutils` rebinning and Doppler shifts with
  [`SpectralAxis`](reference/spectral.md#torchsky.spectral.SpectralAxis),
  `flux_conserving_rebin`, and `torchsky.redshift`.

## What does not carry over

- No `SkyCoord` subclass or representation graph.
- No implicit units, no `Quantity` arithmetic, no automatic epoch/EOP values.
- No FITS or table I/O inside `torchsky` (including healpy's `read_map`).
- No Astropy `WCSAxes` plotting, `NDData`, or `Cutout2D` equivalents; use
  `torchfits` cutouts and `torchsky.viz`.
- No guaranteed bit-for-bit parity with healpy/reproject/Astropy outside the
  documented contracts; parity evidence lives in `torchsky-harness`.
- No `astropy.io.fits`/`fitsio` imports anywhere in the compute path.

## Porting checklist

1. Replace file reads with `torchfits` and pass header dicts onward.
2. Convert NumPy arrays to tensors with an explicit dtype and device.
3. Decide the pixel origin once (0-based torchsky vs 1-based FITS values).
4. Name the frame and epoch explicitly; supply EOP values for observed place.
5. Declare map value semantics, weights, and masks instead of relying on
   defaults inherited from another package.
6. Run the equivalent torchsky example and compare against your previous
   pipeline on the same inputs before deleting the old path.
