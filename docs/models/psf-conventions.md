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
