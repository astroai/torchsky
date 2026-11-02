# Observed place
<!-- TODO(0.5.0 docs): intro -->


## Observed place

Altitude and azimuth need a site, a UTC MJD, and Earth orientation. Nothing is
inferred from IERS tables.

<!-- docs: run -->
```python
import torch

from torchsky.catalogs import Catalog
from torchsky.coords import EarthOrientation, Site

cat = Catalog(
    lon=torch.tensor([10.0, 266.4], dtype=torch.float64),
    lat=torch.tensor([-5.0, -28.9], dtype=torch.float64),
    frame="ICRS",
)
altaz = cat.observe(
    "ALTAZ",
    obstime_mjd_utc=torch.full((2,), 60000.5, dtype=torch.float64),
    site=Site(lon_deg=-155.47, lat_deg=19.83),
    eop=EarthOrientation(dut1_s=0.1),
)
assert altaz.frame == "ALTAZ"
assert bool((altaz.lat.abs() <= 90.0).all())
```

- Without polar motion the result is classical DUT1 GMST hour-angle AltAz.
  Setting `xp_arcsec`/`yp_arcsec` applies CIO/ERA plus the SOFA polar-motion
  matrix into ITRS geodetic AltAz.
- `pn="iau2006"` adds Fukushima-Williams bias-precession and the IAU 2000A
  nutation series; it requires `tt_minus_utc_s` and never infers leap seconds.
- `transform(..., to_frame="GCRS"|"CIRS"|"TETE"|"ITRS")` uses the same obstime
  and EOP. Aberration and refraction are off unless requested.
- `aberration=True` gives the apparent place as ERFA `atciq` does: solar light
  deflection plus annual and diurnal aberration (the observer's Earth-rotation
  velocity), with the ERFA `epv00` Earth ephemeris (about 1e-15 au, evaluated in
  float64 on the CPU). Measured against `erfa.atco13` without refraction over
  4000 random sky positions and epochs (2017-2026, polar motion, DUT1, 4.2 km
  height), alt/az agrees to 0.012 mas at the maximum (median 0.009 mas); the
  residual is the TIO locator s' (taken as 0) and TT used for TDB.
- `refraction=True` applies the ERFA `refco` model, `A tan z + B tan^3 z`, to the
  topocentric direction as `atioq` does (`Site.relative_humidity` and
  `Site.wavelength_um` enter `refco`; wavelengths above 100 micrometres select the
  radio formula). Against `erfa.atco13` it agrees to 0.012 mas for zenith
  distance below 85 degrees (optical and radio). `refraction_model="saemundsson"`
  selects the Saemundsson/Meeus altitude formula instead. Refraction needs
  `site.pressure_mbar` and `site.temperature_c`.
- Site `height_m` is the WGS84 ellipsoidal height. It does not change the
  azimuth and altitude of an infinitely distant star.
- Observed place runs on the catalog's device (CUDA float64, MPS float32). MPS
  IAU 2006 results differ from CPU float64 by a few arcseconds.
  There is no fused C++ observed kernel.
