# Spectral INRs, Physical Observation Operators, and Likelihoods: Roadmap & Recommendations

This document outlines high-impact architectural additions and API improvements for `torchsky` to natively support continuous rest-frame spectral modeling, high-frequency line resolution, multi-band spectrophotometry, and topocentric atmospheric operators.

## 1. Context & Motivation

Astronomical foundation models and rest-frame SED representations (such as `uspm` and multi-modal sky mappers) require evaluating implicit neural representations (INRs) across vast continuous wavelength domains ($\sim 150\text{ \AA}$ to $>12,000\text{ \AA}$) while simultaneously resolving narrow spectral lines (FWHM $\sim 1\text{--}3\text{ \AA}$, relative width $\Delta\ln\lambda \sim 2\times 10^{-4}$).

Currently, `torchsky` provides foundational coordinate transforms, extinction laws (`torchsky.extinction`), Doppler relations (`torchsky.redshift`), and map-level field operators (`torchsky.map.field`). However, spectral modeling pipelines must often improvise custom 1D INR wrappers, custom Python filter integration loops, and ad-hoc atmospheric layers.

Implementing these primitives directly in `torchsky` provides standardized, C++/CUDA-accelerated, and differentiable building blocks across the AstroAI stack.

---

## 2. Proposed `torchsky` Enhancements

### 2.1 Native 1D Spectral Wavelet INR (`torchsky.spectral.WireSpectralField1D`)

#### Current State
`torchsky.map.field.models_implicit` provides `_GaborLayer` and `WireImplicitField`, but they are tightly coupled to 2D/3D image-plane coordinates:
```python
# Current map implementation is tied to spatial coordinates:
self._encoded_coords(ra_deg, dec_deg, wavelength_angstrom)
```

#### Proposed API
A first-class, batched 1D continuous spectral representation module in `torchsky.spectral`:
```python
class WireSpectralField1D(nn.Module):
    """Batched 1D continuous rest-frame spectral representation with Gabor wavelets.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of conditioning code (z_src).
    hidden_dim : int
        Width of hidden Gabor layers.
    hidden_layers : int
        Number of Gabor layers.
    omega0 : float
        Spatial frequency of Gabor wavelet carrier.
    sigma0 : float
        Gaussian envelope bandwidth.
    coordinate_mode : {"log_wavelength", "wavelength"}
        Coordinate convention (default: dimensionless log-lambda).
    """
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 128,
        hidden_layers: int = 4,
        omega0: float = 30.0,
        sigma0: float = 5.0,
        coordinate_mode: str = "log_wavelength",
    ) -> None: ...

    def forward(
        self,
        coords: Tensor,          # (batch, n_coords, 1) or (1, n_coords, 1)
        condition: Tensor,       # (batch, latent_dim) FiLM conditioning vector
    ) -> Tensor:                 # (batch, n_coords, 1) rest-frame flux
        ...
```

#### Key Advantages
- **Eliminates Spectral Bias & Gibbs Ringing**: Gabor wavelets $\phi(x) = \cos(\omega_0 x) \exp(-\|s_0 x\|^2)$ provide compact localization in both wavelength and frequency space, accurately resolving narrow absorption/emission lines without global continuum ringing.
- **Differentiable FiLM Modulation**: Conditioning via feature-wise linear modulation ($\gamma(z) \odot h + \beta(z)$) enables smooth latent interpolation across spectral types.

---

### 2.2 GPU-Fused Tensor Synthetic Photometry (`torchsky.spectrophotometry.synthetic_photometry`)

#### Current State
Synthetic photometry integration $\int F_\lambda T_b(\lambda) d\lambda$ is frequently implemented via Python loops over filter dictionaries, with dynamic spline resampling on CPU/GPU.

#### Proposed API
A pre-compiled, batched tensor operator in `torchsky.spectrophotometry`:
```python
class FilterBundle:
    """Pre-compiled, tensor-native transmission curves on standard quadrature grids."""
    @classmethod
    def from_svo(cls, band_names: Sequence[str], grid: Tensor) -> FilterBundle: ...

def integrate_photometry(
    flux_spec: Tensor,          # (batch, n_grid) in F_lambda (e.g. erg/s/cm2/A)
    grid_angstrom: Tensor,      # (n_grid,) or (batch, n_grid)
    filters: FilterBundle,      # Pre-interpolated transmission matrix (n_bands, n_grid)
    output_convention: str = "f_nu_microjy",
) -> Tensor:                    # (batch, n_bands) predicted fluxes
    """Fused trapezoidal quadrature across all bands in a single CUDA pass."""
```

#### Key Advantages
- Fully vectorized across batch and bands; zero Python loop overhead.
- Directly supports autograd with respect to input flux and grid coordinates.
- Caches normalized response matrices $\frac{\lambda T_b(\lambda)}{\int \lambda T_b(\lambda) d\lambda}$ to guarantee sub-millimagnitude spectrophotometric precision.

---

### 2.3 Batched Variable-Resolution LSF Convolution (`torchsky.spectral.convolve_lsf`)

#### Current State
Astrophysical spectrographs (e.g. SDSS, DESI, WEAVE, PFS) exhibit wavelength-dependent resolving power:
$$R(\lambda) \equiv \frac{\lambda}{\Delta\lambda_{\text{FWHM}}}$$
Currently, lines are either smoothed with constant-velocity Gaussian kernels or processed through ad-hoc FFT convolutions.

#### Proposed API
```python
def convolve_variable_lsf(
    flux: Tensor,               # (batch, n_lambda) on uniform log-lambda grid
    resolving_power: Tensor,    # (n_lambda,) or scalar R(lambda)
    delta_log_lambda: float,    # Pixel step in ln(lambda)
    kernel_shape: str = "gauss_hermite",
    hermite_h3_h4: Tensor | None = None,
) -> Tensor:
    """Fast, differentiable convolution with local line spread function (LSF)."""
```

#### Key Advantages
- Accurately models asymmetric instrumental profiles (Gauss-Hermite terms $h_3, h_4$).
- Native C++/AVX2 and CUDA implementations ensure high throughput on large batches.

---

### 2.4 Physical Atmospheric Operators (`torchsky.atmosphere`)

#### Current State
Observation-level nuisances (night sky emission and telluric absorption) are often modeled with unconstrained MLPs. Without physical priors, these MLPs risk absorbing intrinsic rest-frame features.

#### Proposed API
A dedicated `torchsky.atmosphere` module:
1. **Telluric Transmission**:
   $$T_{\text{atm}}(\lambda_{\text{obs}}) = \exp\left( -X \cdot \tau_{O_2}(\lambda_{\text{obs}}) - \text{PWV} \cdot \tau_{H_2O}(\lambda_{\text{obs}}) \right)$$
   where $X$ is airmass, $\text{PWV}$ is precipitable water vapor, and $\tau(\lambda)$ are pre-tabulated high-resolution atmospheric templates for major observatories (APO, Paranal, Mauna Kea).
2. **Signed Sky Line Emission Residuals**:
   $$S_{\text{sky}}(\lambda_{\text{obs}}) = \sum_{k=1}^K c_k \cdot \psi_k(\lambda_{\text{obs}})$$
   where $\psi_k(\lambda)$ are empirical eigenspectra of night sky emission lines ([OI] 5577, 6300, 6363 Å; Na I D; OH Meinel bands), and $c_k \in \mathbb{R}$ are **signed** coefficients accounting for both under- and over-subtraction.

---

### 2.5 Likelihoods Free of Truncation (Lutz-Kelker) Bias (`torchsky.stats`)

#### Current State
In faint spectra, low-throughput filter tails, and deep absorption lines, true flux $F \ge 0$, but additive Gaussian detector noise $\epsilon \sim \mathcal{N}(0, \sigma^2)$ frequently yields $F_{\text{obs}} < 0$. Clamping predictions or observations $F \ge 0$ induces an asymmetric positive flux bias (identical to the Lutz-Kelker truncation bias in astrometric parallaxes).

#### Proposed API
`torchsky.stats` utilities for evaluating unconvoluted positive signals under symmetric additive noise:
```python
def untruncated_flux_nll(
    f_pred: Tensor,             # Predicted rest/convolved flux (unclamped)
    f_obs: Tensor,              # Observed flux (allowed to be negative from noise)
    ivar: Tensor,               # Inverse variance
    floor_mode: str = "soft_penalty",
) -> Tensor:
    """Evaluates Gaussian/Student-t NLL with signed residuals around prediction,
    penalizing unphysical negative model flux smoothly without asymmetric noise clipping."""
```

---

## 3. Implementation Priority Matrix

| Feature | Target Module | Complexity | Primary Beneficiaries |
| :--- | :--- | :--- | :--- |
| **1D Gabor Wavelet Field** | `torchsky.spectral` | Medium | `uspm`, 1D continuous SED models |
| **GPU-Fused Synthetic Photometry** | `torchsky.spectrophotometry` | Low | All photometric and SED pipelines |
| **Variable LSF Convolution** | `torchsky.spectral` | Medium | High-resolution spectrographs (DESI, SDSS) |
| **Physical Atmosphere Templates** | `torchsky.atmosphere` | Medium | Ground-based spectroscopic pipelines |
| **Untruncated Noise Likelihood** | `torchsky.stats` | Low | Low-S/N spectra, faint-end photometry |
