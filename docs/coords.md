# `torchsky.coords`

`torchsky.coords` groups coordinate conversion helpers used by WCS, sphere, and
catalog workflows.

## What This Namespace Is For

Use `coords` when the task is coordinate conversion rather than mapmaking or
region geometry. For full FITS-WCS transforms, use [wcs.md](wcs.md). For HEALPix
maps and spherical geometry objects, use [sphere.md](sphere.md).

## Core Concepts

- Celestial helpers cover the current native ICRS/Galactic and FK5 precession
  slice.
- Spectral helpers share semantics with `torchsky.wcs` and FITS spectral units.
- HEALPix-oriented helpers are re-exported here for coordinate-centric code; the
  owning namespace is `torchsky.sphere.healpix`.

## API Surface

Celestial and frame helpers:

- `icrs_to_galactic(...)`
- `galactic_to_icrs(...)`
- `precess_fk5(...)`
- `convert_celestial(...)`

Spectral helpers:

- `convert_spectral(...)`
- `spectral_to_frequency(...)`
- `frequency_to_spectral(...)`
- `air_wavelength_to_vacuum(...)`
- `vacuum_wavelength_to_air(...)`

HEALPix-style coordinate helpers are re-exported for coordinate-centric
pipelines; see [sphere.md](sphere.md) for the full HEALPix surface.

HEALPix and geometry helpers:

- `healpix`
- `ang2pix(...)`, `pix2ang(...)`, `vec2pix(...)`, `pix2vec(...)`
- `query_disc(...)`, `query_polygon(...)`
- `query_strip(...)` through `torchsky.coords.healpix` /
  `torchsky.sphere.healpix`
- `get_all_neighbours(...)`, `get_interp_weights(...)`, `get_interp_val(...)`

Temporal, tabular, and alignment helpers:

- `TemporalWCS`
- `TabularAxis`, `TabularLookupWCS`, `tabular_axes_from_fits(...)`
- `TangentPlaneFrame`
- `fit_affine_2d(...)`, `fit_sky_affine(...)`

For lower-level temporal conversion helpers such as `convert_time_scale(...)`
and `iso_to_mjd(...)`, import from `torchsky.wcs.temporal`.

## Related Guides

- [wcs.md](wcs.md)
- [sphere.md](sphere.md)
- [geom.md](geom.md)
