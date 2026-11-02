# Projections
<!-- TODO(0.5.0 docs): intro -->


## Projections And Devices

Celestial projections include the Paper II zenithal, cylindrical, and conic
families, polyconic (`BON`, `PCO`), HEALPix (`HPX`, `XPH`), quadcube (`TSC`,
`CSC`, `QSC`), and `NCP` (slant `SIN`). `PV1_1` and `PV1_2` set the native
fiducial point `(phi_0, theta_0)` of Paper II section 2.5 as in WCSLIB `celset`:
they fix the Euler angles and the default `LONPOLE` (`phi_0` if `dec_0 >= theta_0`,
else `phi_0 + 180`), and the projection origin is not shifted onto the fiducial
point (WCSLIB `cel.offset = 0`). Headers with a non-default fiducial point use
the tensor path on every device (no fused C++ shortcut). `TAN` with any `PV` card
is read as SCAMP `TPV` distortion, so its `PV1_1`/`PV1_2` are distortion terms.
Fused C++ kernels cover `PCO`, `XPH`, `TSC`, `CSC`, and `QSC` on CPU; the public
methods are the only supported entry points and fall back to tensor code
elsewhere.

`CSC` reproduces WCSLIB's float32 evaluation: float32 Chan-O'Neill polynomials,
the float32 `x/45` plane coordinate with its wrap and face classification, and the
float32 face-offset sum in the forward direction (up to 1e-5 degree on faces away
from the base face). `a*b + c` is always two separately rounded operations, so
results are identical on every platform and device. A baseline x86-64 WCSLIB
build matches exactly (6e-14 degree); AArch64 WCSLIB builds fuse the multiply-add,
so against them CSC differs by one float32 ulp (about 2e-5 degree forward).

### MPS precision

MPS has no float64, so a WCS on MPS runs float32. `wcs.to("mps")` (and the first
MPS tensor passed to a WCS on another device) emits `torchsky.wcs.PrecisionWarning`
once per WCS when the expected float32 error reaches half a pixel, the point at
which the nearest pixel is no longer reliable. The estimate is one float32 ulp of
the largest world value (longitude up to 360 degrees times `cos(dec_0)`, latitude
up to 90) divided by the smallest CD-column norm in degrees per pixel, plus one
ulp of the largest pixel coordinate. At 0.2 arcsec per pixel and `dec_0 = 0` that
is 111 mas, 0.55 pixel, so the warning fires; at 1 arcsec per pixel it is 0.11
pixel and it stays silent. Use CPU or CUDA for such grids.

Identity TPV uses `PV1_1=1` and `PV2_1=1`. A header with `PV2_2=1` and no `PV2_1`
is rank-deficient and its inverse is a particular solution.

GWCS objects are sampled onto regular grid transforms at calibration time and the
runtime never imports `gwcs`. Frozen detector tables keep a consistent Jacobian
sign, including reflected detector frames.
