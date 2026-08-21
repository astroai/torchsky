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
  site, and UTC MJD. Omitting polar motion keeps classical DUT1 GMST hour-angle
  AltAz. Setting `xp_arcsec` / `yp_arcsec` applies CIO/ERA plus SOFA `pom00`
  (`s'=0`) into ITRS geodetic AltAz. `pn='iau2006'` adds Fukushima-Williams
  bias-precession and full IAU 2000A nutation with IAU 2006/P03 adjustments
  (SOFA `nut06a` / `c2i06a`; requires `tt_minus_utc_s`; leaps are never
  inferred). That name is kept; the series is IAU 2000A, not the old 10-term
  truncation. `wcs.transform(..., to_frame='GCRS'|'CIRS'|'TETE')` needs the
  same obstime and EOP as ITRS. GCRS is ICRS plus optional Keplerian
  aberration (no PN). CIRS is GCRS then the CIO-based NPB matrix. TETE is
  GCRS then the equinox-based true equator/equinox of date (same FW + nutation
  as SOFA `pnm06a`). Stellar aberration is Keplerian Earth barycentric velocity and
  is off unless requested. Saemundsson/Meeus refraction is off unless
  `refraction=True` with `site.pressure_mbar` and `site.temperature_c`.
  `icrs_to_itrs_xyz` returns unit ITRS directions. IERS values are never
  inferred. Site `height_m` is WGS84 ellipsoidal height (metres); it does not
  change stellar (infinite-distance) azimuth and altitude.
- Thin façade: `torchsky.wcs.transform(lon, lat, from_frame=..., to_frame=...)`.
  Geometric frames use `CatalogState.to_frame`. `ALTAZ`/`HORIZON`, `ITRS`,
  `GCRS`, `CIRS`, and `TETE` need obstime and EOP. This is not a SkyCoord
  subclass. Catalog tensors stay
  on their device. Observed place and geometric frames use ATen (CUDA float64,
  MPS float32; MPS has no float64). MPS IAU 2006 observed vs CPU float64 is
  a few arcseconds. There is no fused C++ observed kernel.

Representations are layouts (`lonlat_deg`, `unit_xyz`, `barycentric_au`),
not Astropy representation classes. `barycentric_au` requires a positive
parallax; it is never filled from a default distance.

Oracle: Astropy/ERFA for geometric frames and space motion. Observed-place
parity is only meaningful after the same EOP values are pinned on both sides.
