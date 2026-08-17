# Catalog coordinates

torchsky does not clone Astropy `SkyCoord`. Catalog geometry is batched
tensors with explicit missing data. Observed place is a separate module and
never runs unless Earth orientation is supplied.

## Layout

- `CatalogState` in `torchsky.wcs.catalog_state`: `lon`, `lat`, optional
  proper motion, parallax, radial velocity, epoch, and frame. Missing
  components stay `None`. There is no `pmsafe` distance override.
- Geometric frames: ICRS, Galactic, FK4, FK5, mean ecliptic of J2000.
  These are rotations and precession. They do not take `obstime` or site.
- Angular propagation: `propagate_proper_motion` / `CatalogState.propagate_angular`.
- Physical 6-parameter propagation: `propagate_space_motion` /
  `CatalogState.propagate_physical`. Incomplete state raises.
- Observed place: `torchsky.wcs.observed`. Requires `EarthOrientation.dut1_s`,
  site, and UTC MJD. Polar motion (`xp`/`yp`) raises until a CIRS path exists.
  The current AltAz path is classical hour angle from DUT1 GMST, not IAU 2006
  CIRS/TETE.

Representations are layouts (`lonlat_deg`, `unit_xyz`, `barycentric_au`),
not Astropy representation classes. `barycentric_au` requires a positive
parallax; it is never filled from a default distance.

Oracle: Astropy/ERFA for geometric frames and space motion. Observed-place
parity is only meaningful after the same EOP values are pinned on both sides.
