# Detector-grid PSF response conventions

`PSFSpec.kernel2d`, `ConstrainedKernelPSF`, `ConditionedKernelPSF`, and
`ExtendedResidualKernelPSF` use image-shaped impulse responses. A kernel entry
at `[cy + dy, cx + dx]` describes the response at detector offset `(dx, dy)`
from a unit source. The central entry represents zero displacement. Explicit
`PSFSpec` kernels have odd dimensions; learned kernels are odd square arrays.
`effective_kernels` returns these source stamps in the same orientation as
the normalized star stamps consumed by `PSFCalibrationTerm`.

PyTorch `conv2d` computes cross-correlation. The renderer rotates these kernels
by 180 degrees at the `conv2d` boundary to implement source transport. Explicit
variance propagation uses the squared transport weights. The explicit-grid
inverse operators use the same convention in full, tiled, and grouped execution;
their adjoints transpose that transport, including zero-padding and cropping.

## Source flux and uniform sky are different contracts

For source-conditioned mixture weights `a_b(q)` and image-shaped basis kernels
`K_b`, the discrete operator is

```text
y[p] = sum_q sum_b a_b(q) K_b[p - q] x[q].
```

Nonnegative basis kernels and convex source weights give unit column sums when
each source's entire finite support lies on the detector. Thus interior source
flux is conserved. At a detector boundary, the sum is the fraction of the stamp
that lands on the detector: escaped response is lost. The renderer does not
renormalize individual boundary stamps or recover unmodelled aperture tails.
Unit sum describes the represented finite kernel, not independently calibrated
total throughput or encircled energy beyond that support.

A uniform sky probes row sums instead:

```text
y[p] for x[q] = 1 is sum_q sum_b a_b(q) K_b[p - q].
```

Those row sums can vary inside the detector when the source weights vary.
Unit-sum source responses therefore do not establish a flat-field response.
Detector radiometry, pixel area, vignetting, and diffuse-sky calibration require
their own physical conventions and evidence.

The existing `PSFSpec.learned_lowrank` control-cell renderer and
`LowRankChromaticPSF.filter_image` retain their separate output-conditioned tent
blend. They weight each stationary filtered image at its **output** position.
Their kernels are centrosymmetric Gaussian profiles, so reversing orientation
does not affect them. A normalized output blend has unit interior row response;
it does not generally have unit column sums. It must not be substituted for the
source-conditioned operator under a source-flux claim.

## Asymmetric kernels and imported conventions

Image-shaped kernels are applied as impulse responses: the renderer converts
`conv2d` cross-correlation to convolution, so asymmetric stamps render in their
measured orientation. Symmetric kernels are unaffected, and image-shaped
calibration stamps need no conversion.

An array stored as a cross-correlation weight must be converted with
`kernel.flip(-2, -1)` before constructing `PSFSpec`. For learned checkpoints
trained against rendered images under a cross-correlation convention, rotate
`raw_kernels` across their last two dimensions (and the extended model's fixed
`baseline_kernel`). Checkpoints calibrated directly against measured star
stamps already have the intended stamp orientation and should not be rotated.
Record which convention an imported calibration used.

`tests/test_psf_transport_invariants.py` checks these bounded discrete contracts
against independent source scattering and a dense transport matrix. It checks
orientation, calibration/render agreement, boundary losses, interior flux,
source/output mixing differences, variance, adjoints, and parameter gradients.
It does not certify continuous optics, subpixel sampling, omitted wings, or
physical flat-field closure.
