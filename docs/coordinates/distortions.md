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

## Minimal Example: An HST ACS Header (SIP)

The header below is the SIP part of chip 1 (`SCI,1`) of the ACS/WFC exposure
`ja7za3uwq_flc.fits` (programme 11371, NGC 6217, F814W, 2009). It was read from
MAST and its WCS cards are copied verbatim, so the snippet runs offline. The
fifth-order polynomial moves the chip corners by up to 3.1 arcsec relative to
the linear `CD` solution. The full HST solution also has the `CPDIS` and `D2IM`
lookup tables that sit in other extensions of the file (see the recipe below).
torchsky matches Astropy/WCSLIB on this header to better than 1e-6 mas.

<!-- docs: run -->
```python
import torch

from torchsky.wcs import WCS

acs_wfc1 = {   # MAST mast:HST/product/ja7za3uwq_flc.fits, HDU SCI,1 (WCSNAME IDC_0461802dj)
    "NAXIS": 2, "NAXIS1": 4096, "NAXIS2": 2048, "CTYPE1": "RA---TAN-SIP", "CTYPE2": "DEC--TAN-SIP",
    "CRPIX1": 2048.0, "CRPIX2": 1024.0, "CRVAL1": 248.092265756327, "CRVAL2": 78.18901993692644,
    "CD1_1": 1.18249897003846e-05, "CD1_2": 8.07657398228177e-06, "CD2_1": 7.20569251008454e-06,
    "CD2_2": -1.1404356942835e-05, "A_ORDER": 5, "A_0_2": 2.16281011638133e-06,
    "A_0_3": -1.46765064057e-13, "A_0_4": 2.30223358084363e-14, "A_0_5": 6.02562521036014e-18,
    "A_1_1": -5.2158590853721e-06, "A_1_2": -5.3107969261848e-10, "A_1_3": -7.8659474888604e-15,
    "A_1_4": -1.4384073054676e-18, "A_2_0": 8.55616623616636e-06, "A_2_1": -3.3917464109293e-11,
    "A_2_2": 3.78668896735343e-14, "A_2_3": 2.16488764283563e-18, "A_3_0": -4.9291246606641e-10,
    "A_3_1": 6.55012597688049e-15, "A_3_2": 1.79047178737381e-18, "A_4_0": 2.49607276946249e-14,
    "A_4_1": -2.2971527069895e-18, "A_5_0": 3.34341779527238e-18, "B_ORDER": 5,
    "B_0_2": -7.2361867909899e-06, "B_0_3": -3.8365985750577e-10, "B_0_4": 4.81417281373599e-15,
    "B_0_5": -3.0671653630262e-17, "B_1_1": 6.18551929615015e-06, "B_1_2": -7.2701405701748e-11,
    "B_1_3": 1.17031008147548e-14, "B_1_4": 1.2378998215801e-17, "B_2_0": -1.7465651995444e-06,
    "B_2_1": -5.1770808967636e-10, "B_2_2": -2.9745588348931e-14, "B_2_3": -4.0252836502166e-19,
    "B_3_0": 9.84290294757522e-11, "B_3_1": -9.2509958233811e-16, "B_3_2": -6.5707633212806e-19,
    "B_4_0": -1.7429369365811e-14, "B_4_1": 1.26904088898591e-18, "B_5_0": -2.9208753120287e-18,
}
wcs = WCS(acs_wfc1)
linear = WCS({k: v for k, v in acs_wfc1.items() if k[:2] not in ("A_", "B_")}
             | {"CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN"})

x = torch.tensor([0.0, 2047.0, 4095.0, 0.0, 4095.0], dtype=torch.float64)   # corners and CRPIX
y = torch.tensor([0.0, 1023.0, 0.0, 2047.0, 2047.0], dtype=torch.float64)
ra, dec = wcs.pixel_to_world(x, y)
x_back, y_back = wcs.world_to_pixel(ra, dec)
assert (x_back - x).abs().max() < 1e-6 and (y_back - y).abs().max() < 1e-6

ra0, dec0 = linear.pixel_to_world(x, y)
shift = torch.hypot((ra - ra0) * torch.cos(torch.deg2rad(dec)), dec - dec0) * 3600
assert shift[1] < 1e-6 and 2.0 < shift.max() < 3.5        # 0 at CRPIX, ~3 arcsec at the corners
```

The [dithered SIP coadd](../published-examples/coadd_dithered_sip.py) renders
eight exposures through SIP solutions and checks the astrometry of the coadd.

## Recipes

### Read a DECam TPV Header

DECam community-pipeline images (`InstCal`) carry one SCAMP TPV solution per
CCD. The header below is CCD S29 of `c4d_140525_072118_ooi_g_rrr.fits.fz`
(programme 2014A-0321, NGC 6744 field, g band, 2014), read from the NOIRLab
Astro Data Archive with its WCS cards copied verbatim. `CRPIX` points at the
optical axis of the 62-CCD mosaic, some 11 000 pixels off this chip. `PV1_3`
and `PV2_3` are the coefficients of \(r\), here zero. Under `-TPV` they are
polynomial terms, not the `LONPOLE`/`LATPOLE` of Paper II. The polynomial moves
this chip by 3.0–3.5 arcsec relative to the linear `CD` solution; torchsky
matches Astropy/WCSLIB on it to better than 1e-6 mas.

<!-- docs: run -->
```python
import torch

from torchsky.wcs import WCS

decam_s29 = {   # NOIRLab archive c4d_140525_072118_ooi_g_rrr.fits.fz, extension S29
    "NAXIS": 2, "NAXIS1": 2046, "NAXIS2": 4094, "CTYPE1": "RA---TPV", "CTYPE2": "DEC--TPV",
    "CRPIX1": 13422.2, "CRPIX2": 6306.33300000001, "CRVAL1": 287.4314124816,
    "CRVAL2": -63.84949275399, "CD1_1": -4.060086963025e-07, "CD1_2": 7.285880050401e-05,
    "CD2_1": -7.286252566409e-05, "CD2_2": -4.059689173171e-07, "PV1_0": 0.001709600882009,
    "PV1_1": 1.01479314616, "PV1_2": -0.002479541426295, "PV1_3": 0.0, "PV1_4": 0.007835790723642,
    "PV1_5": -0.01976350862627, "PV1_6": 0.001901723338406, "PV1_7": 0.0002087520149181,
    "PV1_8": -0.007273769983855, "PV1_9": 0.007872055622797, "PV1_10": -0.0005311615314207,
    "PV2_0": -0.003103387145331, "PV2_1": 1.013895242054, "PV2_2": -0.0102502558997,
    "PV2_3": 0.0, "PV2_4": -0.01626927460387, "PV2_5": 0.02022513728663, "PV2_6": -0.01122435264465,
    "PV2_7": 0.006422711612, "PV2_8": -0.01015554463106, "PV2_9": 0.008962633753844,
    "PV2_10": -0.0026543549284,
}
tpv = WCS(decam_s29)
linear = WCS({k: v for k, v in decam_s29.items() if not k.startswith("PV")}
             | {"CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN"})

x = torch.tensor([0.0, 2045.0, 0.0, 2045.0, 1022.5], dtype=torch.float64)
y = torch.tensor([0.0, 0.0, 4093.0, 4093.0, 2046.5], dtype=torch.float64)
ra, dec = tpv.pixel_to_world(x, y)
x_back, y_back = tpv.world_to_pixel(ra, dec)
assert (x_back - x).abs().max() < 1e-6 and (y_back - y).abs().max() < 1e-6

ra0, dec0 = linear.pixel_to_world(x, y)
shift = torch.hypot((ra - ra0) * torch.cos(torch.deg2rad(dec)), dec - dec0) * 3600
assert 2.5 < shift.min() and shift.max() < 4.0           # arcsec, across the whole chip
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

`CTYPE` is authoritative, as the FITS standard and the SIP convention
require. A header whose distortion cards contradict it raises `ValueError`
naming the fix; the one compatibility reading torchsky keeps emits
[`WCSHeaderWarning`](../reference/wcs.md#torchsky.wcs.WCSHeaderWarning).

| Header | Astropy / WCSLIB | torchsky |
|---|---|---|
| `A_ORDER`/`B_ORDER` with plain `TAN` (or `TPV`) | applies SIP, logs a message | `ValueError`: append `-SIP`, or drop the stale `A_*`/`B_*` cards (drizzled images keep them) |
| `TAN` with `PVi_m`, some `m >= 5` | TPV (pre-2012 SCAMP) | TPV, `CTYPE` rewritten to `-TPV`, `WCSHeaderWarning` |
| `TAN` with `PV2_m`, all `m <= 4` | standard `TAN`; `PV2_m` ignored, `PV1_1`/`PV1_2` become the fiducial point | `ValueError`: a degree-1 SCAMP fit looks exactly like this, so set `-TPV` or remove the cards |
| `TAN` with only `PV1_0`…`PV1_4` | standard fiducial point and pole | same as WCSLIB |
| `TAN-SIP` with SCAMP `PVi_m` (`m >= 5`) | drops the `PV` cards | `ValueError`: keep one convention |

Make the legacy reading an error with
`warnings.simplefilter("error", WCSHeaderWarning)`.

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
