# PSF conventions

!!! warning "Experimental surface"
    The PSF specifications and learned PSF operators used by
    [fields](fields.md) and [scenes](scenes.md) are experimental. The
    orientation and flux conventions on this page are tested contracts; the
    learned kernel classes, their parameters, and their checkpoint layout may
    change between minor releases.

The **point-spread function (PSF)** is the image of a point source: how the
light of a single sky position spreads over detector pixels. In torchsky a
discrete PSF is given as a small odd-sized array, a **kernel**, and this page
fixes two things that otherwise cause silent errors: which way round the kernel
is read, and what "normalized" does and does not guarantee near detector edges
and for diffuse light.

Read it before you import a PSF model from another package, train a learned
PSF, or interpret photometry from a forward model near the edge of a detector.
It applies to `PSFSpec.kernel(...)`, `ConstrainedKernelPSF`,
`ConditionedKernelPSF`, and `ExtendedResidualKernelPSF`.

## How It Works

**Orientation.** A kernel is an **impulse response stamp**: the entry at
`[cy + dy, cx + dx]` is the response at detector offset \((dx, dy)\) from a
unit source at the kernel centre `[cy, cx]`. It looks exactly like the image of
an isolated star, cut out around the star, which is also what calibration star
stamps look like. Rendering a sky \(x\) through it is a convolution:

\[
y[p] = \sum_q K[p - q]\, x[q] .
\]

PyTorch's `conv2d` computes a cross-correlation, \(\sum_q K[q - p]\,x[q]\), so
the renderer rotates every kernel by 180° at the `conv2d` call. Symmetric
kernels are unaffected; asymmetric ones (coma, trailing, diffraction spikes at
an angle) render in their measured orientation. Adjoints and variance
propagation use the same transport: the adjoint is the transpose including the
zero padding and cropping at the image boundary.

**Source flux versus uniform sky.** For a spatially varying PSF built from
basis kernels \(K_b\) with source-dependent weights \(a_b(q)\),

\[
y[p] = \sum_q \sum_b a_b(q)\, K_b[p - q]\, x[q] .
\]

| Probe | Sums | Equals one when |
|---|---|---|
| A point source \(x = \delta_q\) (flux) | Column sum over \(p\) | Kernels are non-negative and unit-sum, weights are convex, and the whole stamp lies on the detector |
| A uniform sky \(x = 1\) (flat field) | Row sum over \(q\) | The weights do not vary across the stamp; not guaranteed otherwise |

So a unit-sum kernel conserves the flux of an interior source. At a detector
edge, the part of the stamp that falls off the detector is lost and is not
renormalized back. Unit sum also does not make the response to a flat sky
uniform when the PSF varies with position; flat-fielding, pixel area, and
vignetting are separate calibrations.

## Assumptions and Validation

- The kernel's finite support is the whole PSF. Wings beyond it are not
  modelled, so "unit sum" means unit sum of the represented stamp, not total
  throughput or encircled energy at large radius.
- `PSFSpec.kernel` normalizes the kernel to unit sum; odd dimensions are
  required.
- [`tests/test_psf_transport_invariants.py`](https://github.com/astroai/torchsky/blob/main/tests/test_psf_transport_invariants.py)
  checks orientation, calibration/render agreement, boundary loss, interior
  flux, variance, adjoints, and gradients against an independent dense
  transport matrix. It does not certify continuous optics, sub-pixel sampling,
  or omitted wings.

## Orientation and Edge Loss

A kernel with 30 per cent of the light one pixel to the right of the centre
puts it to the right of the star in the rendered image; a star at the right
edge loses that 30 per cent; a flat sky darkens by the same amount in the edge
column.

<!-- docs: run -->
```python
import torch

from torchsky.field import PSFSpec
from torchsky.map.operators import apply_psf_spec

kernel = torch.zeros(3, 3, dtype=torch.float64)
kernel[1, 1] = 0.7
kernel[1, 2] = 0.3                                # response at dx = +1, dy = 0
psf = PSFSpec.kernel(kernel)

star = torch.zeros(9, 9, dtype=torch.float64)
star[4, 4] = 1.0
image = apply_psf_spec(star, psf)
assert float(image[4, 5]) == 0.3                  # the stamp's orientation is kept
torch.testing.assert_close(image.sum(), torch.tensor(1.0, dtype=torch.float64))

edge_star = torch.zeros(9, 9, dtype=torch.float64)
edge_star[4, 8] = 1.0
torch.testing.assert_close(apply_psf_spec(edge_star, psf).sum(), torch.tensor(0.7, dtype=torch.float64))

flat = apply_psf_spec(torch.ones(9, 9, dtype=torch.float64), psf)
assert float(flat[4, 0]) == 0.7 and float(flat[4, 4]) == 1.0
```

## Recipes

### Import a Kernel From Another Package

- **A measured star stamp** or a PSF model image (PSFEx, WebbPSF/STPSF,
  photutils ePSF evaluated on the pixel grid) is already an impulse response:
  pass it to `PSFSpec.kernel` unchanged.
- **A cross-correlation weight**, such as a filter stored for direct use with
  `conv2d` or `scipy.ndimage.correlate`, must be rotated first:

```python
# fragment: weights is a 2-D cross-correlation array from another pipeline
from torchsky.field import PSFSpec

psf = PSFSpec.kernel(weights.flip(-2, -1))
```

Record which convention an imported calibration used alongside the model.

### Learned Kernels and Checkpoints

`ConditionedKernelPSF` learns non-negative unit-sum basis kernels mixed with
convex weights that depend on detector position, wavelength, and time, so it
conserves interior flux by construction. Its `effective_kernels` are returned
in stamp orientation, the same as the normalized star stamps used by
`PSFCalibrationTerm`. A checkpoint trained against images rendered under a
cross-correlation convention needs `raw_kernels` (and, for
`ExtendedResidualKernelPSF`, `baseline_kernel`) rotated by 180° over the last
two dimensions; one calibrated directly on star stamps does not.

### Low-Rank Chromatic PSFs Are Different

`PSFSpec.learned_lowrank` and `LowRankChromaticPSF.filter_image` blend
stationary filtered images by **output** position. Their row response
(uniform sky) is one in the interior, but their column sums (point-source flux)
are not one in general. Do not use them where a source-flux claim is needed;
use a source-conditioned kernel PSF instead.

### Wide Kernels at Constant Cost

A dense stamp that reaches 200 pixels costs `conv2d` work proportional to its area, which is out of reach
for an image of a few thousand pixels. `NestedKernelPSF` takes the PSF as a **kernel field**, a function of
the offset `(dx, dy)` in pixels, and splits it into concentric shells. Shell `l` is sampled on a grid of
pitch `2**l` pixels with a fixed number of nodes, so the core is kept at full resolution and each wing is
carried at a pitch matched to its radius; the cost grows with the number of shells (the logarithm of the
extent), not with the area. It obeys the contract above: the stamp is an impulse response, rendering is the
convolution `y[p] = sum_q K[p - q] x[q]`, the image is zero outside the detector, flux falling off it is
lost, the adjoint is the exact transpose, and the unit-sum kernel conserves the flux of an interior source.

Three things are different from `apply_kernel2d`, and are tested:

- The result matches a dense stamp to an interpolation error, not to round-off: about 4e-5 of the peak
  for a point source or a smooth source and 2e-4 for white noise (the worst case, because its power lies at
  the pixel scale, which coarse shells cannot carry) for a 2-pixel core with a power-law wing to 200 pixels.
  The only sharp structure of the kernel must lie inside half of `core_half_width` pixels.
- The operator is not exactly shift-invariant: the response of a source depends, at the 3e-5 level of the
  peak, on its position modulo the coarsest pitch. `effective_kernels` returns the stamp for each source
  position; the rendered image is exactly the scatter of those stamps.
- Kernels must be non-negative (the field is normalised to unit sum for every region node), and coordinates
  must be a unit-pitch detector grid, because the pooling lattices are anchored to detector coordinates. That
  anchoring makes tiled rendering with the halo `kernel_size // 2` reproduce the untiled image to round-off.

<!-- docs: run -->
```python
import torch

from torchsky.map.operators import NestedKernelPSF


def field(offsets, condition):
    """Density per pixel^2 at offsets [P, 2] = (dx, dy); one row per region node."""
    r2 = (offsets**2).sum(-1)
    core = torch.exp(-r2 / 3.0)  # FWHM about 2 pixels
    wing = 0.02 * (1.0 + r2 / 9.0) ** -1.5  # power-law wing
    return (core + wing)[None]


psf = NestedKernelPSF(field, core_half_width=16, extent=100.0)  # 4 levels: pitch 1, 2, 4, 8
y, x = torch.meshgrid(torch.arange(256.0), torch.arange(256.0), indexing="ij")
kwargs = {"spatial_x": x, "spatial_y": y, "wavelength": 0.5}

star = torch.zeros(256, 256, dtype=torch.float64)
star[128, 128] = 1.0
image = psf(star, **kwargs)
assert psf.num_levels == 4
assert abs(float(image.sum()) - 1.0) < 1e-9  # unit flux, wings included
assert float(image[128, 128]) > float(image[128, 132]) > float(image[128, 178])

edge = torch.zeros(256, 256, dtype=torch.float64)
edge[128, 255] = 1.0
assert 0.5 < float(psf(edge, **kwargs).sum()) < 0.9  # flux off the detector is lost, not renormalized

other = torch.rand(256, 256, dtype=torch.float64)
lhs = (psf(star, **kwargs) * other).sum()
rhs = (star * psf.adjoint(other, **kwargs)).sum()
assert abs(float(lhs - rhs)) < 1e-10 * abs(float(lhs))  # adjoint is the exact transpose
```

## Pitfalls

- **Flipping twice.** Applying `flip(-2, -1)` to a stamp that was already an
  impulse response mirrors the PSF.
- **Edge photometry.** A source within half a kernel width of the edge loses
  flux; mask or model those pixels rather than rescaling them.
- **Kernel support.** A 21-pixel kernel truncates the wings of a bright star;
  the missing light is not in the model.

## See Also

- [Fields](fields.md#learn-the-psf-and-background-in-stages): staged fitting of
  learned PSF and background operators.
- [Likelihoods](likelihoods.md): pixel-integrated PSF responses for native
  star stamps.
- [Formulation](formulation.md#the-observation-model): where the spatial
  response sits in the forward model.
- Reference: [`torchsky.field`](../reference/field.md#torchsky.field.PSFSpec),
  [`torchsky.map.operators.epsf`](../reference/map-operators-epsf.md).
