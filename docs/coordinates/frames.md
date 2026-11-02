# Frames and time
<!-- TODO(0.5.0 docs): intro -->


torchsky does not clone Astropy `SkyCoord`. Catalog geometry is a batch of
tensors with explicit missing data, frame conversion is a function, and observed
place is a separate step that never runs unless you supply Earth orientation.

## CatalogState

`CatalogState` (also importable from `torchsky.coords`) carries `lon`, `lat`,
the `frame`, an `epoch`, and optional proper motion, parallax, and radial
velocity. Missing components stay `None`; nothing is imputed, and there is no
`pmsafe`-style distance override.

<!-- docs: run -->
```python
import torch

from torchsky.coords import CatalogState, transform

ra = torch.tensor([10.0, 266.4], dtype=torch.float64)
dec = torch.tensor([-5.0, -28.9], dtype=torch.float64)

# Frame conversion on bare tensors ...
l, b = transform(ra, dec, from_frame="ICRS", to_frame="GALACTIC")
ra_back, _ = transform(l, b, from_frame="GALACTIC", to_frame="ICRS")
torch.testing.assert_close(ra_back, ra, atol=1e-9, rtol=0.0)

# ... or on a state, which rotates proper motion with the position.
state = CatalogState(
    lon=ra, lat=dec, frame="ICRS", epoch=2016.0,
    pm_lon_coslat_mas_yr=torch.tensor([50.0, -20.0], dtype=torch.float64),
    pm_lat_mas_yr=torch.tensor([10.0, 5.0], dtype=torch.float64),
)
gal = state.to_frame("galactic")
assert gal.frame == "GALACTIC"
torch.testing.assert_close(
    torch.hypot(gal.pm_lon_coslat_mas_yr, gal.pm_lat_mas_yr),
    torch.hypot(state.pm_lon_coslat_mas_yr, state.pm_lat_mas_yr),
)
```

Supported geometric frames are ICRS, Galactic, FK4, FK5, mean ecliptic of J2000,
ecliptic of date (TT MJD with IAU 2006 precession-nutation), Galactocentric
(explicit `galcen_distance_kpc` and `z_sun_pc`), and LSR (explicit solar UVW).
Representations are layouts (`lonlat_deg`, `unit_xyz`, `barycentric_au`), not
Astropy representation classes; `barycentric_au` needs a positive parallax.
FK4 proper-motion conversion is rejected because no complete time-dependent
transform is implemented; position-only FK4 conversion works. The conversion is
differentiable.

## Propagating motion

The free functions `propagate_proper_motion` and `propagate_space_motion` do the
same on bare tensors. `CatalogState.propagate_angular` moves positions with proper motion only; it needs both
components. `propagate_physical` is the six-parameter ERFA/SOFA model with
changing light-time and needs proper motion, parallax, and radial velocity. An
incomplete state raises `ValueError`.

<!-- docs: run -->
```python
import torch

from torchsky.coords import CatalogState

ra = torch.tensor([10.0, 266.4], dtype=torch.float64)
dec = torch.tensor([-5.0, -28.9], dtype=torch.float64)
pm = torch.tensor([50.0, -20.0], dtype=torch.float64)
state = CatalogState(
    lon=ra, lat=dec, frame="ICRS", epoch=2016.0,
    pm_lon_coslat_mas_yr=pm, pm_lat_mas_yr=torch.tensor([10.0, 5.0], dtype=torch.float64),
)

earlier = state.propagate_angular(2000.0)
assert earlier.epoch == 2000.0 and not torch.equal(earlier.lon, state.lon)

try:
    state.propagate_physical(2000.0)
except ValueError:
    pass
else:
    raise AssertionError("physical propagation must not impute parallax")

full = CatalogState(
    lon=ra, lat=dec, frame="ICRS", epoch=2016.0,
    pm_lon_coslat_mas_yr=pm, pm_lat_mas_yr=torch.tensor([10.0, 5.0], dtype=torch.float64),
    parallax_mas=torch.tensor([10.0, 5.0], dtype=torch.float64),
    radial_velocity_km_s=torch.tensor([20.0, -30.0], dtype=torch.float64),
)
assert full.propagate_physical(2000.0).lon.shape == (2,)
```

`propagate_proper_motion_with_jacobian` and `propagate_space_motion_with_jacobian`
also return the local Jacobian (2x4 and 6x6) in east/north tangent-plane
offsets, so you can propagate a covariance with `J @ C @ J.mT`. The 6x6 form
lets you select Gaia's correlated astrometric subspace without inventing joint
radial-velocity covariance. Catalog covariance modelling, blends, and
association probabilities are out of scope.

## Optional interop

Opt-in extras are never imported by `torchsky.wcs`: `astropy_interop`
(`from_astropy`, `to_astropy`), `ephemeris.get_body` (`jplephem` and a DE
kernel; the small DE440s kernel covers the Sun, Moon and inner planets, Mars and the
outer planets need the full DE440), `tle.catalog_state_from_tle` (`sgp4`, frame `TEME`), and
`gwcs_live.freeze_live_gwcs`. Astropy and ERFA are the oracles for geometric
frames and space motion; observed-place parity is only meaningful once both
sides pin the same EOP values.

## See also

- [WCS](index.md) for FITS-WCS transforms.
- [Catalogs and crossmatch](../catalogs/index.md) for epoch-aware matching.
- [`torchsky.coords` reference](../reference/coords.md).
