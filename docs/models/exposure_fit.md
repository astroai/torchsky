# Exposure Fit

!!! note "Experimental: pieces 1 and 2 of a move"
    `torchsky.map.exposure_fit` is moved, number for number, from a research prototype. It reads calibrated exposure files into
    `FieldObservation` objects (`load_exposures`), fits one sky and an instrument response to them (`fit_sky`), and scores held-out exposures
    (`predict_exposure`). Sky images with uncertainties (`SkyFit.product`), catalogue inputs and a command line are not available yet; the
    placeholders `SkyProduct`, `ReferenceCatalogue` and `SkyFit.product` raise `NotImplementedError`. Names may change between minor releases.

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
- `predict_exposure(fit, observation, cross_fit=True)` returns `ExposurePrediction(model, sigma, terms, chi2)` for a held-out exposure. With
  cross-fitting each pixel is predicted by per-exposure terms that were fitted on the other half of the interleaved 4 x 4 pixel blocks.

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
- A held-out exposure must be in the list given to `fit_sky`; an exposure that was not cannot be scored yet.

## Recipe

The block below builds a small synthetic observation by hand, shows the cards a fit needs, and checks that a bad configuration and the
placeholders fail closed. A complete fit at the default private settings takes about two minutes on a CPU even for this tiny field, so the call
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
chi-squared (the check lives outside this repository, with the prototype). On real files, pixel values, errors, masks, conversion factors and
positions of the loader agree with the prototype's own loader (JWST and WFC3/IR to float32 round-off). The tests here cover the fit, saving
and loading, prediction, configuration validation, the converter, the loader on small synthetic files, the FFT correlation against direct
convolution and its adjoint, the item rasteriser, and one small step of the loss and its gradient.
