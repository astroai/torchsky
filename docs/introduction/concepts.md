# Concepts: tensors, devices, and gradients
<!-- TODO(0.5.0 docs): intro -->


The ideas that hold the rest of the documentation together: what torchsky owns,
how data crosses the file boundary, where computation runs, and how it relates to
the established astronomy stack.

## Tensors in, tensors out

Every compute path accepts tensors and header dictionaries. Nothing in the core
opens a file, and no object hides a NumPy or Astropy round trip.

<!-- docs: run -->
```python
import torch

from torchsky.wcs import WCS

# A header is a plain dict: the same mapping torchfits returns from a file.
header = {
    "NAXIS": 2, "NAXIS1": 128, "NAXIS2": 128,
    "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
    "CRPIX1": 64.5, "CRPIX2": 64.5, "CRVAL1": 10.0, "CRVAL2": -30.0,
    "CDELT1": -1e-3, "CDELT2": 1e-3,
}
wcs = WCS(header)

x = torch.arange(0.0, 128.0, 16.0, dtype=torch.float64)
ra, dec = wcs.pixel_to_world(x, x)
x_back, y_back = wcs.world_to_pixel(ra, dec)

assert ra.dtype == torch.float64 and ra.device == x.device
torch.testing.assert_close(x_back, x, atol=1e-6, rtol=0.0)
```

Two consequences follow. Dtype and device are decided by your inputs, and
autograd flows through the geometry wherever the underlying maths is smooth.

## File boundary

FITS files and tables are owned by
[`torchfits`](https://github.com/astroai/torchfits), a required dependency that
installs with torchsky. `torchsky` contains no FITS or CFITSIO code and never
imports `astropy.io.fits` or Python `fitsio`.

```python
# fragment: needs a FITS file on disk
import torchfits
from torchsky.wcs import WCS

image, header = torchfits.read("science.fits", hdu="auto", return_header=True)
wcs = WCS(header)
```

Catalogs follow the same rule. `torchfits.table` reads, filters, and streams
FITS tables as Arrow or tensor batches; `torchsky.catalogs` owns the sky-domain
part: spherical indexing, partitioning, and crossmatch.

```python
# fragment: needs a FITS table on disk
import torchfits
from torchsky.catalogs import crossmatch

for batch in torchfits.table.scan("catalog.fits", columns=["RA", "DEC"], batch_size=100_000):
    matches = crossmatch(batch["RA"], batch["DEC"], ref_ra, ref_dec, radius_arcsec=1.5)
```

What must survive the boundary:

- tensor dtype, shape, and device;
- ordered header cards that WCS needs, including alternate keys and `-TAB`
  references;
- mask and weight meaning. torchsky never infers inverse variance or validity;
- spectral and time units and axis order;
- row order and chunk boundaries for streamed tables.

Ambiguous metadata fails closed rather than being guessed from a file name.
`torchfits` documents its own scan and read signatures; consult its
[documentation](https://astroai.github.io/torchfits/) for the exact calls.

## Explicit semantics

Choices that change the answer are arguments, not defaults buried in a function:

- flux versus surface brightness, and the kernel used to resample;
- weights, inverse variance, masks, and data-quality bits;
- pixel origin (0-based tensors versus 1-based FITS values);
- coordinate frame and epoch;
- spectral units and wavelength reference.

Where an input is ambiguous torchsky raises `ValueError` or a typed exception
listed in the [exceptions reference](../reference/exceptions.md).

## Scale

Survey-scale inputs drive the design. Algorithms prune before they compute
(spherical indices, footprint cropping), and products are bounded by
configuration: tiling, chunking, streaming, and sparse HEALPix. The
[scaling guide](../performance/scaling.md) lists the knobs. Native C++ kernels exist
only where profiling showed a real hotspot, and a pure-torch path remains the
correctness baseline. Set `TORCHSKY_DISABLE_CPP=1` to run without the native
accelerator.

## Device placement

Map, field-fitting, and IFU APIs compute and return on the device of their
tensor inputs. CPU tensors stay on CPU even when CUDA is available. An explicit
`device=` wins, and mixed devices without it raise `ValueError`. Only path-backed
inputs use `torchsky.device.default_compute_device()`. MPS has no float64, so
float64-only kernels raise `NotImplementedError` there; run them on CPU or CUDA.
The [map guide](../maps/index.md#device-placement) has the full rules.

## Ecosystem and compatibility

torchsky implements its own coordinates, WCS, spherical geometry, maps, scenes,
and field models. Established packages are references, parity oracles, or opt-in
adapters. They are not hidden backends.

| Area | torchsky owns | Used only as reference or adapter |
|---|---|---|
| FITS files and tables | nothing (uses `torchfits`) | `astropy.io.fits` and `fitsio` are never imported |
| WCS | `torchsky.wcs` | Astropy WCS as a test oracle |
| Sphere and HEALPix | `torchsky.sphere` | healpy, hpgeom, healsparse, astropy-healpix, spherical-geometry |
| Mapmaking | `torchsky.map` | SWarp, Montage, STScI drizzle, reproject as comparators |
| Catalogs | `torchsky.catalogs` | Astropy `SkyCoord` as an oracle |

Compatibility targets are behaviours users depend on, not another package's
internals. The healpy-shaped surface in `torchsky.sphere.compat` sits on top of
the native kernels. To port code, see
[Coming from Astropy, healpy, and reproject](coming-from.md).

## Validation

The repository's tests cover bounded API, regression, adjoint, gradient, and
device contracts. Exhaustive upstream parity, real-data validation, and
scorecards live in the sibling
[`torchsky-harness`](https://github.com/astroai/torchsky-harness) repository;
[Parity evidence](../performance/parity.md) summarizes what it covers. FITS decode
and write throughput belong to `torchfits`.
