# Exposure Fit

!!! note "Experimental: pieces 1 to 3 of a move"
    `torchsky.map.exposure_fit` is moved, number for number, from a research prototype. It reads calibrated exposure files into
    `FieldObservation` objects (`load_exposures`), fits one sky and an instrument response to them (`fit_sky`), scores exposures
    (`predict_exposure`), and makes the sky image of any bands with its standard deviation (`SkyFit.product`). Catalogue inputs and a command
    line are not available yet; `ReferenceCatalogue` raises `NotImplementedError`. Names may change between minor releases.

## What Goes In

Calibrated exposures, one `FieldObservation` per exposure and detector chip, cut to a square box on the sky, never resampled or coadded.

- **From files:** [`load_exposures`](../reference/map-exposure-fit.md#torchsky.map.exposure_fit.load_exposures) reads JWST NIRCam `_cal`
  files and HST WFC3/IR `_flt` and ACS/WFC `_flc` files through `torchfits`. It applies each instrument's data-quality policy
  ([`instrument_description`](../reference/instrument.md#torchsky.instrument.instrument_description)), converts HST counts to MJy/sr
  from the header cards, and uses the HST a priori (guide-star) astrometric solution, never one fitted to other images.
- **Filter curves:** the fit needs the transmission curve of every filter. Nothing is fetched during a fit. Give the instrument description the
  curves as tensors with `InstrumentDescription.with_filter_curves({"F150W": (wavelength_um, transmission)})` before loading. (A curve cache
  from `torchsky.filter_profiles` can fill those tensors; that facility may use the network when a curve is missing, so it is left to the caller.)
- **Bookkeeping cards:** the observation's header carries a few extra cards, written by `load_exposures` and required on a hand-built
  observation: `TSINST` (`NIRCAM`, `WFC3_IR`, `ACS_WFC`), `DETECTOR`, `TSNAME` (unique exposure name), `TSSPLIT` (`train` or `held`),
  `TSBOXRA`, `TSBOXDE`, `TSBOXHS` (centre in degrees and half side in arcsec of the box), and `TSDETNX`, `TSDETNY` (size of the whole chip in
  pixels). Set `TSSPLIT` to `"held"` to keep an exposure out of the fit and score it afterwards.
- **Header solution:** `torchsky.wcs.WCS` evaluates the header of the cut; positions are projected onto the tangent plane at the box centre
  (X west, Y north, in arcsec).

## What Comes Out

- `fit_sky(observations, config)` returns a `SkyFit` (an `nn.Module`): `.sky` (the items and the spectral basis), `.response` (the kernel
  network, the codes and `noise_terms()`), `.exposure_terms` (affine map, gain, background and kernel code of every exposure), `.config` and
  `.report` (loss log, learned noise floor, warm-start and registration records). `.save(path)` writes `manifest.json` and `state.pt`;
  `SkyFit.load(path)` reads them back and gives identical predictions.
- `predict_exposure(fit, observation, cross_fit=True)` returns `ExposurePrediction(model, sigma, terms, chi2)` for a held-out exposure, or for
  any exposure that was not in the fit (a name, `TSNAME`, the fit does not know) of an instrument, detector and band the fit has. The exposure's own
  terms (background, affine map, gain, kernel code, stripes) are fitted at prediction time with the sky and response held, on a copy of the
  operator; the fit and the global random state are not changed. With cross-fitting each pixel is predicted by terms that were fitted on the
  other half of the interleaved 4 x 4 pixel blocks.
- `fit.product(bands, observations=...)` returns a [`SkyProduct`](../reference/map-exposure-fit.md#torchsky.map.exposure_fit.SkyProduct); see
  [The Sky Product](#the-sky-product).
- `fit.response.kernel(instrument, band, exposure=None)` gives the fitted kernel as a stamp (`KernelStamp`: `stamp [n, n]` in 1 / arcsec^2 and
  its `pitch_arcsec`), unit integral inside the normalisation radius (`config.kernel_norm_radius`) on the model's own quadrature;
  `fit.response.throughput(instrument, band)` gives `(wavelengths [um], curve)`, the published transmission at the `Q` nodes times the learned
  throughput (the reference instrument's mean log-throughput, gains included, is zero at every node, so its curve is the published one up to the
  small gains); `fit.response.noise_terms()` gives the learned fractional floors and scales. `fit.bands` and `fit.instruments` list the names.

## The Model

For pixel \(\boldsymbol\xi\) of exposure \(e\):

\[
d_e(\boldsymbol\xi) = N_e\!\left[\frac1Q\sum_{q=1}^{Q}\alpha_e(\boldsymbol\xi)\,e^{\gamma_e(\boldsymbol\xi,\lambda_q)}
\int d^2u\; k_e(\boldsymbol u;\boldsymbol\xi,\lambda_q)\; s\big(\Omega_e(\boldsymbol\xi,\lambda_q)+\boldsymbol u,\lambda_q\big)\right]
+ B_e(\boldsymbol\xi) + n_e(\boldsymbol\xi)
\]

\(s\) is the item sky (many anisotropic Gaussians with \(K\) spectral coefficients each, positive at every wavelength), \(\lambda_q\) are \(Q\)
wavelength nodes at quantiles of the filter curve, \(k_e\) the learned kernel, \(\gamma_e\) the learned log-throughput including a per-exposure
gain, \(\Omega_e\) the per-exposure geometry (affine map plus translation), \(B_e\) a quadratic background, and \(n_e\) noise with a learned
scale and fractional floor.

## Stated Approximations

- The numerics are those of the research prototype, moved unchanged; internal settings that are not in `SkyFitConfig` keep the prototype's
  defaults (a test pins them). The kernel is the same for every position on the detector within an exposure's learned code; no distortion
  beyond the header solution, no detector pattern, no non-linearity, no per-pixel gain.
- One tile only (a single square box). `reference_instrument` fixes the absolute flux scale with the published curves of that instrument.
- `background_order` must be 2: the moved code fits six Legendre terms.
- The sky half-size is the box half-size minus 1 arcsec plus 2 arcsec when the observations carry the same `TSBOXHS`; otherwise it is derived
  from the pixel positions.
- ACS/WFC look-up-table distortion corrections are not applied by `torchsky.wcs`: positions differ from astropy's by about 1 milliarcsecond
  in the checked file.
- A new exposure given to `predict_exposure` gets terms fitted by the same procedure as a held-out one, from the fit's starting values for a
  new row (the held-out ones of the fit were also registered with the training exposures in the registration pre-pass).

## The Sky Product

`fit.product(bands, pitch_arcsec=None, resolution="native", region=None, *, observations, method="circulant", polish=True, samples=8,
batch_pixels=16000, max_minutes=None, log=None)` returns a frozen `SkyProduct`: `image` and `std` (`[B, H, W]`, float32, MJy/sr for data from
`load_exposures`), `valid` (`[H, W]`), `bands`, `resolution_fwhm_arcsec` per band, `unit`, `header` (a tangent-plane WCS as a dict of cards that
`torchsky.wcs.WCS` evaluates; east to the left), `pitch_arcsec`, `zero_level_convention`, `polished`, `method`, `n_samples`, `truncated` with
`residual`, `n_passes`, and `notes` (the approximations in words).

**The training exposures are given again.** The normal operator below is built from the training pixels, and the fit does not keep them (it
stays small, and a fit read back with `SkyFit.load` works exactly the same way). Pass the observations the fit was made from; held-out ones
are ignored. The median error and the median value of every exposure are checked against what the fit saw.

**Definition.** With the sky grid \(A\) (the item sky rasterised on its nodes), the per-exposure background coefficients \(b\) (six Legendre
terms each), and everything else held at its fitted value, the model of the training pixels is linear: \(m = J_A A + L\,b + \text{held terms}\).
The Gauss-Newton normal operator is

\[
F = \begin{pmatrix} J_A & L \end{pmatrix}^{\mathsf T} W \begin{pmatrix} J_A & L \end{pmatrix} + \operatorname{diag}(\mu I,\,0),\qquad
W = \operatorname{diag}(w_i),\quad w_i = \frac{\min(1,\delta/|z_i|)}{\sigma_i^2},
\]

where \(\sigma_i\) is the learned noise of pixel \(i\) (pipeline error times the learned scale, plus the fractional floor), \(z_i\) the fit's
residual in units of \(\sigma_i\), \(\delta\) the Huber threshold of the fit's loss, and \(\mu\) a weak prior (the median node information over
\(10^2\)). The product of a band with spectral weights \(\phi\) is the sky smoothed with a circular Gaussian \(g_s\) (the tent interpolant of
the grid convolved with it), and its variance is

\[
\sigma_s(x)^2 = k_x^{\mathsf T}\,[F^{-1}]_{AA}\,k_x,\qquad k_x = \phi\otimes (T * g_s)(x-x_\cdot),
\]

with \(T\) the bilinear tent of one sky pitch. \(F^{-1}\) is never formed: conjugate gradients with a Fourier preconditioner apply \(F\) with the
model's own render and its adjoint.

**What is held fixed and what is marginalised.** Held at the fitted values, so not in `std`: the kernel and throughput, the registration, the
per-exposure gains, the stripes, the noise scales and floors. Marginalised: the six background terms of every training exposure. The sky modes
that are exactly degenerate with those backgrounds (a constant, a gradient and a quadratic over the field) are tied to the background convention
(the training mean of each term is held), so they are not in `std` either; on the prototype's known-truth problem they add 0.3 to 0.5 sigma in
quadrature at the supported resolution. The zero level is the median rule of the fit (one number per filter, `zero_level_convention`), and its
uncertainty is not in `std` (on the same problem about 1.4 sigma at 8 exposures; it grows as the square root of the number of exposures).

**The trusted sky is "fit plus one Gauss-Newton step" (`polish=True`).** On a problem with known truth the pulls of that sky have rms 1.0 to 1.1;
the optimiser's own sky has about half the noise of the posterior (positivity and an unfinished optimisation) and a bias of about 1.5 sigma in
empty regions, so `std` is not its error. With `polish=False` the image is the optimiser's sky and `polished` is False; do not quote `std` for it.
The step is not positive: the product can be negative.

**What "native" resolution means.** `resolution` is the full width at half maximum of the smoothing Gaussian. `"native"` is, per band, the median
over covered nodes of the supported resolution \(s^\ast(x)\): with \(\alpha(s,x) = -\partial\ln\sigma_s(x)/\partial\ln s\), the smallest
width from which \(\alpha\le 2\) on a ladder of widths from half to six sky pitches (scanned from coarse to fine; at the finest scales the prior takes
over and \(\alpha\) falls back towards 1, which is prior and not resolution). For a Gaussian response of width \(p\) it is \(\sqrt2\,p\). Finer than
that, `std` depends on the prior rather than the data. One solve serves all bands; each band has its own native resolution, reported in
`resolution_fwhm_arcsec`. When the ladder never settles (a poorly fitted field) `"native"` raises and asks for an explicit resolution.

**Methods and cost.** One pass is one forward and one adjoint render of every training pixel.

| step | passes over the training pixels |
|---|---|
| set-up (weights, coverage, Fourier preconditioner), `method="circulant"` | about 13 at \(K=12\) spectral coefficients (\(K+1\) applications and one forward-only pass) |
| Gauss-Newton step (`polish=True`) | about 61 more |
| each posterior draw (`method="draws"`, `samples`) | about 61 |

`method="circulant"` uses the Fourier model of \(F\) alone (uniform coverage, background terms ignored, scaled by the local relative coverage):
within 1 per cent of draws at the supported resolution on the known-truth problem, but blind to chip gaps and edges. `method="draws"` averages
the variance of `samples` perturb-and-solve draws over a Gaussian window of three times the product width (relative error \(1/\sqrt{2n}\) before
the window). `max_minutes` stops the solve cleanly when the next pass would end after the budget (set-up is not interrupted; a cut step is its last
iterate, a cut draw is dropped; `truncated`, `residual`, `n_samples` and `notes` say what happened). Measured on one CPU thread on the prototype's
truth problem (14,746 pixels, \(K=3\)): about 1.4 s per pass.

**Other grids.** With `pitch_arcsec` or `region` (`(x_min, x_max, y_min, y_max)` in arcsec from the box centre, X west, Y north) the image is
evaluated exactly at the output positions, while `std` and `valid` are interpolated linearly from the node grid.

**Checking `std` on real data.** Fit two disjoint halves of the exposures (for example alternate dither groups), make a product of each with the
same bands and an explicit `resolution`, and call `half_difference(product_a, product_b)`: the pull \((I_a-I_b)/\sqrt{\sigma_a^2+\sigma_b^2}\) should
have mean 0 and rms 0.9 to 1.2 per band. Biases common to the halves (zero level, held response) cancel, so this checks the noise part only.

The runnable block reads a product the way a caller would, on arrays built by hand (a real fit takes minutes); the call itself is a fragment.

<!-- docs: run -->
```python
import dataclasses
import torch
from torchsky.map.exposure_fit import SkyProduct, half_difference
from torchsky.wcs import WCS

H = W = 9
g = torch.Generator().manual_seed(0)
header = {"NAXIS": 2, "NAXIS1": W, "NAXIS2": H, "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN", "CUNIT1": "deg", "CUNIT2": "deg",
          "CRVAL1": 150.0, "CRVAL2": 2.0, "CRPIX1": 5.0, "CRPIX2": 5.0, "CD1_1": -0.05 / 3600, "CD1_2": 0.0, "CD2_1": 0.0, "CD2_2": 0.05 / 3600}
std = torch.full((1, H, W), 0.1)
truth = torch.randn(1, H, W, generator=g)
make = lambda noise: SkyProduct(                                    # noqa: E731
    image=truth + noise * std, std=std, valid=torch.ones(H, W, dtype=torch.bool), bands=("F150W",), resolution_fwhm_arcsec=(0.2,),
    unit="MJy/sr", header=header, pitch_arcsec=0.05, zero_level_convention="median rule", polished=True, method="circulant", n_samples=0,
    truncated=False, residual=1e-4, n_passes=75, notes=())
a, b = make(torch.randn(1, H, W, generator=g)), make(torch.randn(1, H, W, generator=g))
stats = half_difference(a, b)                                       # two halves of the same sky: pull mean 0, rms 1
assert stats["bands"] == ("F150W",) and stats["n_pixels"] == [H * W] and abs(stats["rms"][0] - 1.0) < 0.4
ra, dec = WCS(header=a.header).pixel_to_world_differentiable(torch.tensor([[4.0, 4.0]], dtype=torch.float64), origin=0)[0].tolist()
assert abs(ra - 150.0) < 1e-9 and abs(dec - 2.0) < 1e-9             # the centre pixel is the box centre
try:
    dataclasses.replace(a, unit="Jy").unit == "Jy" and half_difference(a, dataclasses.replace(b, unit="Jy"))
except ValueError:
    pass
else:
    raise AssertionError("products in different units must be refused")
```

```python
# fragment: `observations` and `fit` are those of the Recipe; the result of fit_sky must be given the same observations again
product = fit.product(["F100W"], observations=observations, resolution=0.15, method="circulant", log=print)   # about 75 passes
print(product.bands, product.resolution_fwhm_arcsec, product.polished, product.notes)
sky, sigma = product.image[0], product.std[0]                      # MJy/sr on the grid of product.header; sigma = 0 where not product.valid
native = fit.product("all", observations=observations, method="draws", samples=8, max_minutes=240)   # per-band native resolution, one solve
```

## Recipe

The block below builds a small synthetic observation by hand, shows the cards a fit needs, and checks that a bad configuration and the
placeholder fail closed. A complete fit at the default private settings takes about two minutes on a CPU even for this tiny field, so the call
is shown as a fragment.

<!-- docs: run -->
```python
import torch

from torchsky.map.field.types import FieldObservation

RA0, DEC0, PIX, N = 150.0, 2.0, 0.1, 24            # tangent point [deg], pixel [arcsec], chip size
g = torch.Generator().manual_seed(0)
blobs = torch.rand(6, 3, generator=g) * torch.tensor([2.0, 2.0, 3.0]) - torch.tensor([1.0, 1.0, 0.0])   # x, y [arcsec], flux
lam = torch.linspace(1.0, 1.4, 12).double()


def observation(index, band, shift):
    off = (torch.arange(N) - (N - 1) / 2) * PIX
    Y, X = torch.meshgrid(off, off, indexing="ij")                      # X is west (increases as RA falls)
    image = torch.zeros(N, N)
    for x, y, f in blobs:
        image += f * torch.exp(-((X - x - shift[0]) ** 2 + (Y - y - shift[1]) ** 2) / (2 * 0.15 ** 2))
    image = image + 1.0 + 0.05 * torch.randn(N, N, generator=g)
    s = PIX / 3600.0
    header = {"CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN", "CRVAL1": RA0, "CRVAL2": DEC0, "CRPIX1": (N + 1) / 2, "CRPIX2": (N + 1) / 2,
              "CD1_1": -s, "CD1_2": 0.0, "CD2_1": 0.0, "CD2_2": s,
              "TSINST": "NIRCAM", "DETECTOR": "NRCB1", "TSNAME": f"{band}_{index}", "TSSPLIT": "train",
              "TSBOXRA": RA0, "TSBOXDE": DEC0, "TSBOXHS": 0.8, "TSDETNX": N, "TSDETNY": N}
    return FieldObservation(image=image, header=header, mask=torch.ones(N, N, dtype=torch.bool), err=torch.full((N, N), 0.05),
                            band_name=band, transmission_wavelength=lam, transmission_throughput=torch.ones(12).double(), spectral_unit="um",
                            value_unit="MJy/sr", value_semantics="surface_brightness")

observations = [observation(i, "F100W", (0.02 * i, -0.01 * i)) for i in range(3)]

from torchsky.map.exposure_fit import SkyFitConfig, fit_sky
from torchsky.map.exposure_fit._observations import observations_to_exposures

records, box_half = observations_to_exposures(observations, n_lambda=5)
assert len(records) == 3 and box_half == 0.8 and records[0].d.shape == (N, N)
assert abs(float(records[0].X.mean())) < 0.1                       # the field is centred on the tangent point

config = SkyFitConfig(sky_pitch=0.1, item_count=200, spectral_k=4, kernel_extent=1.0, steps=3, device="cpu")
try:
    SkyFitConfig(background_order=3)
except ValueError:
    pass
else:
    raise AssertionError("background_order other than 2 must be refused")
try:
    fit_sky(observations, config, catalogue=object())
except NotImplementedError:
    pass
else:
    raise AssertionError("a catalogue is not used yet")
```

```python
# fragment: a fit takes about two minutes on a CPU even at this size; `observations` are those built above
from torchsky.map.exposure_fit import SkyFit, fit_sky, predict_exposure

fit = fit_sky(observations, config, log=print)
fit.save("sky_fit")                       # manifest.json (configuration, report) and state.pt (tensors)
fit = SkyFit.load("sky_fit")
print(fit.exposure_terms, fit.response.noise_terms())
# for an observation whose header card TSSPLIT was "held" when it was given to fit_sky:
# prediction = predict_exposure(fit, held_observation)   # model, sigma, terms, chi2
```

From files, replace the construction of `observations` by:

```python
# fragment: needs exposure files on disk
from torchsky.instrument import instrument_description
from torchsky.map.exposure_fit import load_exposures

nircam = instrument_description("NIRCAM").with_filter_curves(curves)     # curves: {"F150W": (wavelength_um, transmission)}
observations = load_exposures(paths, [nircam], (110.8268, -73.4544, 11.24))
```

## What Is Checked

The moved code gives identical bytes to the prototype for every tensor of the fitted state on a synthetic field, and identical held-out
chi-squared, and the sky product (polished sky and standard deviation, circulant and posterior draws) is bit-identical to the prototype's on its
known-truth problem (the checks live outside this repository, with the prototype). The prototype's calibration on that problem: pulls of the
polished sky have rms 0.96 (with a quadratic removed) to 1.10 over the inner region and 0.9 to 1.3 at sources, and the half-difference estimate
agrees with `std` to 1.00 +- 0.03. A small synthetic field here cannot support that precision: the tests assert only what it supports (a field
with zero true sky: pull mean near 0, rms below 1.2 and above 0.4, with the reason in the test). On real files, pixel values, errors, masks, conversion factors and
positions of the loader agree with the prototype's own loader (JWST and WFC3/IR to float32 round-off). The tests here cover the fit, saving
and loading, prediction, configuration validation, the converter, the loader on small synthetic files, the FFT correlation against direct
convolution and its adjoint, the item rasteriser, and one small step of the loss and its gradient.
