# Coming from Astropy, healpy, and reproject


torchsky is a tensor-native implementation of sky-domain operations, with
explicit inputs, explicit units, and no hidden file I/O. It is not a drop-in
replacement for Astropy, healpy, or reproject; this page maps the calls you
already know to their torchsky equivalents. For the ownership boundary with
`torchfits`, see [Concepts](concepts.md#ecosystem-and-compatibility).

## Call mapping

| Astropy / healpy / reproject call | torchsky replacement | Notes |
|---|---|---|
| `astropy.wcs.WCS(header)` | [`torchsky.wcs.WCS`](../reference/wcs.md#torchsky.wcs.WCS) | Header stays a plain dict; pixels are 0-based by default. |
| `wcs.pixel_to_world(x, y)` | `wcs.pixel_to_world(x, y)` | Tensors in, tensors out; dispatch on the header's axes. |
| `SkyCoord(ra, dec, frame=...)` | [`CatalogState`](../reference/coords.md#torchsky.coords.CatalogState) | Batched tensors, explicit missing components. |
| `SkyCoord.transform_to(frame)` | [`transform`](../reference/coords.md#torchsky.coords.transform) / `CatalogState.to_frame` | No `SkyCoord` graph; observed place needs Earth orientation. |
| `healpy.ang2pix(...)` | [`torchsky.sphere.ang2pix`](../reference/sphere.md#torchsky.sphere.ang2pix) | Same arguments and defaults, tensor values. |
| `healpy.query_disc(...)` | [`torchsky.sphere.query_disc`](../reference/sphere.md#torchsky.sphere.query_disc) | Takes a unit 3-vector, returns pixel indices. |
| `reproject.reproject_interp(...)` | [`torchsky.map.resample`](../reference/map.md#torchsky.map.resample) | Operates on `Observation` objects and a target grid. |
| `astropy.io.fits` | [`torchfits`](concepts.md#file-boundary) | Never imported by torchsky. |
| `astropy.table.Table` | [`torchfits.table`](concepts.md#file-boundary) | Arrow/table batches; torchsky consumes tensors. |
| `astropy.cosmology` | [`torchsky.cosmology`](../reference/cosmology.md) | Differentiable FLRW container, no parameter priors. |
| `astropy extinction laws` | [`torchsky.extinction`](../reference/extinction.md) | Unit-explicit dust laws on tensors. |
| `specutils` Doppler/rebin | [`torchsky.spectral`](../spectra/index.md) | Spectral axes, flux-conserving rebin and relativistic Doppler shifts (parity-tested); not spectrum I/O or cube readers. |

## Notes

- **Pixel origin.** `pixel_to_world` and `world_to_pixel` use 0-based pixels.
  Use `pixel_to_world_values(..., origin=1)` for the FITS 1-based convention.
- **Headers.** Build a `WCS` from a plain header mapping read with `torchfits`;
  torchsky never opens the file. See [the file boundary](concepts.md#file-boundary).
- **No `SkyCoord`.** A catalog is a batch of tensors. `CatalogState` holds
  `lon`, `lat`, optional proper motion, parallax, radial velocity, epoch, frame,
  and a `valid` mask. Observed place (`ALTAZ`, `ITRS`, `GCRS`, `CIRS`, `TETE`)
  takes an explicit `EarthOrientation` and site; IERS values are never inferred.
- **No implicit units.** Supply tensors in the documented units; there is no
  `Quantity` arithmetic and no automatic epoch inference.
- **healpy.** `torchsky.sphere` uses healpy's naming, argument order, and the
  `nest` and `lonlat` switches (RING is the default). `query_disc` and related
  queries take a unit 3-vector. A healpy-shaped namespace is available as
  `torchsky.sphere.compat`; see the [compatibility reference](../reference/sphere-compat.md).
  Spherical harmonic transforms, smoothing, and spectra follow torchsky's own
  backend contract (see [sphere](../sphere/index.md)).
- **reproject.** `torchsky.map` takes observations with explicit weight, mask,
  DQ, and spectral metadata and returns a
  [`MapResult`](../reference/map.md#torchsky.map.MapResult) with data, weight,
  header, and provenance. Flux versus surface brightness is an explicit
  `ResampleConfig.semantics` choice. See [mapmaking](../maps/index.md).
- **Not provided.** Astropy `WCSAxes`, `NDData`, and `Cutout2D` equivalents
  (use `torchfits` cutouts and `torchsky.viz`), and bit-for-bit parity with
  healpy, reproject, or Astropy outside the documented contracts.

<!-- docs: run -->
```python
import torch
from torchsky.wcs import WCS
import torchsky.sphere as sphere

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

pix = sphere.ang2pix(64, ra, dec, lonlat=True)  # healpy-style call, tensor values
print(pix.tolist())
```

## From torchfits 0.3

FITS I/O lives in `torchfits`; torchsky provides sky geometry (WCS, HEALPix and
MOC, catalogs, mapmaking, fields) on top of the tensors and header mappings it
reads. Install both and read files with `torchfits.read`; pass the header
dict to `torchsky.wcs.WCS`.
