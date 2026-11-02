# Distortions

Optics, filters, and detector mounting bend the focal plane, so a linear `CD`
matrix plus a projection (see [Projections](projections.md)) leaves residuals
of a fraction of an arcsecond to several arcseconds at the edge of a wide
field. FITS headers describe the remainder with a *distortion*: a polynomial
or a sampled table added at a fixed point of the pixel → world chain. torchsky
reads them from the header into the same [`WCS`](../reference/wcs.md#torchsky.wcs.WCS)
object; there is no separate distortion step to call.

## Where Each Convention Acts

```mermaid
flowchart LR
    P["pixel"] -->|"DET2IM<br/>(Paper IV table)"| P2["corrected pixel"]
    P2 -->|"SIP polynomial<br/>+ CPDIS table"| U["distorted offsets<br/>u, v"]
    U -->|"CD"| X["intermediate x, y"]
    X -->|"TPV / TNX / ZPX<br/>polynomial"| X2["corrected x, y"]
    X2 -->|"projection<br/>+ rotation"| W["RA, Dec"]
```

| Convention | Keywords | Acts on | Typical source |
|---|---|---|---|
| SIP (Simple Imaging Polynomial) | `CTYPE = RA---TAN-SIP`, `A_p_q`, `B_p_q`, optional `AP_p_q`, `BP_p_q` | Pixel offsets \(u = p - r\) | HST, JWST, Spitzer, astrometry.net, LSST |
| TPV | `CTYPE = RA---TPV` (or `RA---TAN` with `PVi_m`), `PV1_0`…`PV2_39` | Intermediate coordinates | SCAMP, CFHT MegaCam, DECam |
| TNX / ZPX | `WAT1_*`, `WAT2_*` strings | Intermediate coordinates | IRAF (NOAO Mosaic, older KPNO/CTIO) |
| Paper IV lookup tables | `CPDISj`, `DP.j`, `D2IMDISj` with table extensions | Pixel coordinates | HST ACS/WFC3 detector and filter corrections |
| `-TAB` (Paper III) | `CTYPEi = WAVE-TAB`, `PSi_m`, `PVi_m` | One axis or a coupled block | Irregular spectral or celestial axes |

SIP adds polynomials in pixel offsets:

\[
u' = u + \sum_{p+q \le N} A_{pq}\, u^p v^q, \qquad
v' = v + \sum_{p+q \le N} B_{pq}\, u^p v^q,
\]

and `CD` then multiplies \((u', v')\). TPV applies a polynomial in
\((x, y, r = \sqrt{x^2+y^2})\) to the intermediate coordinates, so an identity
TPV has `PV1_1 = 1` and `PV2_1 = 1`. The forward, inverse, and Jacobian paths
evaluate the polynomials in nested Horner form.

## The Inverse

Distortion polynomials are defined in the forward direction. `world_to_pixel`
solves the forward polynomial for the pixel with Newton iterations:

- **SIP.** `AP`/`BP` coefficients, when the header has them, only seed the
  solve; the result satisfies the forward `A`/`B` polynomial to 1e-6 pixel.
  Without `AP`/`BP` the solve starts from the undistorted position.
- **TPV.** Newton runs up to 20 iterations to 1e-11 degree. Rows that do not
  converge, or whose target is not finite, are NaN.

Check `world_to_pixel(pixel_to_world(p)) ≈ p` over the full image for any
new header: a strongly non-monotonic polynomial outside the fitted area has
no unique inverse, and the round trip is the cheapest test.

## Minimal Example: SIP Round Trip

Synthetic coefficients of the size found in a 2k × 4k CCD at 0.18 arcsec per
pixel; the distortion moves the corners by about 0.9 arcsec.

<!-- docs: run -->
```python
import torch

from torchsky.wcs import WCS

linear = {
    "NAXIS": 2, "NAXIS1": 2048, "NAXIS2": 4096,
    "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
    "CRPIX1": 1024.5, "CRPIX2": 2048.5, "CRVAL1": 150.1, "CRVAL2": 2.2,
    "CD1_1": -5.1e-05, "CD1_2": 0.0, "CD2_1": 0.0, "CD2_2": 5.1e-05,
}
sip = {
    **linear, "CTYPE1": "RA---TAN-SIP", "CTYPE2": "DEC--TAN-SIP",
    "A_ORDER": 2, "A_2_0": 2.0e-6, "A_1_1": -1.0e-6, "A_0_2": 3.0e-7,
    "B_ORDER": 2, "B_2_0": 4.0e-7, "B_1_1": 1.5e-6, "B_0_2": -2.0e-6,
}
wcs, flat = WCS(sip), WCS(linear)

x = torch.tensor([0.0, 1023.5, 2047.0], dtype=torch.float64)   # corner, centre, corner
y = torch.tensor([0.0, 2047.5, 4095.0], dtype=torch.float64)
ra, dec = wcs.pixel_to_world(x, y)
x_back, y_back = wcs.world_to_pixel(ra, dec)
assert (x_back - x).abs().max() < 1e-6 and (y_back - y).abs().max() < 1e-6

shift_arcsec = (dec - flat.pixel_to_world(x, y)[1]).abs() * 3600
assert shift_arcsec[1] < 1e-6 and shift_arcsec[0] > 0.8      # zero at CRPIX, ~0.9" at a corner
assert wcs.to_header()["CTYPE1"] == "RA---TAN-SIP"
```

The [dithered SIP coadd](../published-examples/coadd_dithered_sip.py) renders
eight exposures through SIP solutions and checks the astrometry of the coadd.

## Recipes

### Read a SCAMP TPV Header

<!-- docs: run -->
```python
import torch

from torchsky.wcs import WCS

header = {
    "NAXIS": 2, "NAXIS1": 2048, "NAXIS2": 4612,
    "CTYPE1": "RA---TPV", "CTYPE2": "DEC--TPV",
    "CRPIX1": 1024.5, "CRPIX2": 2306.5, "CRVAL1": 150.1, "CRVAL2": 2.2,
    "CD1_1": -5.16e-05, "CD1_2": 0.0, "CD2_1": 0.0, "CD2_2": 5.16e-05,
    "PV1_0": 0.0, "PV1_1": 1.0, "PV1_2": 0.0, "PV1_4": 2e-3, "PV1_7": 5e-3,
    "PV2_0": 0.0, "PV2_1": 1.0, "PV2_2": 0.0, "PV2_4": -1e-3, "PV2_7": 4e-3,
}
tpv = WCS(header)
same = WCS({**header, "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN"})  # SCAMP's older spelling

x = torch.tensor([0.0, 2047.0], dtype=torch.float64)
y = torch.tensor([0.0, 4611.0], dtype=torch.float64)
ra, dec = tpv.pixel_to_world(x, y)
assert torch.equal(ra, same.pixel_to_world(x, y)[0])
x_back, y_back = tpv.world_to_pixel(ra, dec)
assert (x_back - x).abs().max() < 1e-6 and (y_back - y).abs().max() < 1e-6
```

### Attach Paper IV Lookup Tables

HST headers carry `CPDIS` and `D2IM` corrections as image extensions. Read the
extensions with torchfits, wrap each in a
[`DistortionLookupTable`](../reference/wcs.md#torchsky.wcs.DistortionLookupTable),
and pass them to the constructor. The tables are interpolated bilinearly and
clamped at their edges, as in Astropy.

```python
# fragment: needs an HST FLT/FLC file with WCSDVARR and D2IMARR extensions
import torchfits
from torchsky.wcs import WCS, DistortionLookupTable

path = "science_flc.fits"
header = dict(torchfits.read_header(path, 1))      # SCI extension
dx, dx_header = torchfits.read(path, hdu=4, return_header=True)  # first WCSDVARR
dx_header = dict(dx_header)
table = DistortionLookupTable(
    dx,
    crpix=(dx_header["CRPIX1"], dx_header["CRPIX2"]),
    crval=(dx_header["CRVAL1"], dx_header["CRVAL2"]),
    cdelt=(dx_header["CDELT1"], dx_header["CDELT2"]),
)
wcs = WCS(header, cpdis1=table)   # cpdis2, det2im1, det2im2 likewise
```

### Read `-TAB` Axes

`-TAB` axes need the decoded tables as well as the header. Read both with
torchfits and pass them in:

```python
# fragment: needs a FITS file with -TAB extensions
import torchfits
from torchsky.wcs import WCS

with torchfits.open("cube_with_wave_tab.fits") as hdul:
    wcs = WCS(hdul[0].header, tables=hdul)
```

One-dimensional axes use `PSi_0` (table extension), `PSi_1` (coordinate
column), optional `PSi_2` (index vector), `PVi_1` (`EXTVER`), and `PVi_2`
(`EXTLEVEL`). Missing tables or columns raise rather than fall back to a
linear axis. Coupled multidimensional tables go through
[`coupled_tabular_wcs_from_fits`](../reference/wcs.md#torchsky.wcs.coupled_tabular_wcs_from_fits),
a multilinear interpolator whose inverse is exact for square tables,
least-squares for overdetermined ones, and a minimum-norm particular solution
for underdetermined ones.

### Refine a WCS Against Reference Stars

A header can be right in shape and wrong in position by a fraction of an
arcsecond. [`fit_affine_2d`](../reference/wcs.md#torchsky.wcs.fit_affine_2d)
fits a shift, similarity (`geometry="rscale"`), or general affine between
matched pixel lists, with optional weights and sigma clipping;
`WCS.with_affine` returns a corrected copy and leaves the distortion
untouched.

<!-- docs: run -->
```python
import torch

from torchsky.wcs import WCS, fit_affine_2d

wcs = WCS({
    "NAXIS": 2, "NAXIS1": 512, "NAXIS2": 512,
    "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
    "CRPIX1": 256.5, "CRPIX2": 256.5, "CRVAL1": 150.0, "CRVAL2": 2.0,
    "CD1_1": -1e-5, "CD1_2": 0.0, "CD2_1": 0.0, "CD2_2": 1e-5,
})
# Where the header puts four reference stars, and where they really are.
predicted = torch.tensor([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]], dtype=torch.float64)
measured = predicted + torch.tensor([1.5, -2.0], dtype=torch.float64)

corrected = wcs.with_affine(fit_affine_2d(predicted, measured, geometry="rscale"))

# The corrected WCS puts each star's catalog position on its measured pixel.
ra, dec = wcs.pixel_to_world(predicted[:, 0], predicted[:, 1])
x, y = corrected.world_to_pixel(ra, dec)
torch.testing.assert_close(torch.stack([x, y], -1), measured, atol=1e-8, rtol=0.0)
```

When both lists are sky positions, use
[`fit_sky_affine`](../reference/wcs.md#torchsky.wcs.fit_sky_affine), which fits
in a local gnomonic plane. A full astrometric solution across many exposures
is [`solve_astrometry`](../reference/wcs.md#torchsky.wcs.solve_astrometry).

## When Headers Lie

- **SIP without `-SIP`.** A header with `A_ORDER`/`B_ORDER` gets SIP applied
  even when `CTYPE` ends in plain `TAN`, as Astropy does. Remove the `A_*`
  and `B_*` cards first if they are stale.
- **`TAN` with `PV` cards is TPV.** SCAMP writes `RA---TAN` plus `PVi_m`.
  torchsky reads that as TPV; if the `PV` cards were meant as projection
  parameters, the header is ambiguous and must be fixed upstream.
- **Rank-deficient TPV.** A header with `PV2_2 = 1` and no `PV2_1` is
  rank-deficient; its inverse is a particular solution.
- **Polynomials outside their fit.** SIP and TPV are fitted to stars inside the
  detector. Evaluated far outside it (on a planned mosaic grid, say) they can
  fold back on themselves. Restrict reprojection to the detector footprint.
- **Inverse coefficients from another fit.** `AP`/`BP` that disagree with
  `A`/`B` slow the inverse down but do not change its answer, because the
  forward polynomial is what the Newton solve matches.

## Accuracy and Validation

Distorted headers are compared with Astropy/WCSLIB on upstream fixtures in [torchsky-harness](https://github.com/astroai/torchsky-harness);
see the [parity page](../performance/parity.md). Jacobians of distorted WCS
are checked against finite differences in the test suite.

## See Also

- [Projections](projections.md) and the [coordinates overview](index.md).
- [Coadds](../maps/coadds.md): resampling distorted exposures.
- Reference: [`WCS`](../reference/wcs.md#torchsky.wcs.WCS),
  [`fit_affine_2d`](../reference/wcs.md#torchsky.wcs.fit_affine_2d),
  [`DistortionLookupTable`](../reference/wcs.md#torchsky.wcs.DistortionLookupTable).
- Examples: [WCS tour](../published-examples/wcs_tour.py),
  [dithered SIP coadd](../published-examples/coadd_dithered_sip.py).
