"""Constrained learned instrument-operator families.

Building blocks for survey forward modelling where a learned observation
operator must capture instrument effects (PSF, calibration, background,
small astrometric errors, mild nonlinearities) without absorbing scene
information:

- ``ConstrainedKernelPSF``: quasi-convolution core built from a bank of
  positive, unit-sum basis kernels (flux conserving by construction),
  initialised from Gaussian profiles around a nominal FWHM and mixed by
  coefficients that may be conditioned on an instrument-state latent.
- ``GatedResidualOperator``: small nonlinear residual (brighter-fatter and
  unknown effects) whose amplitude is controlled by a single gate that is
  meant to be L2-regularised toward zero by the caller.
- ``AffineRadiometry``: per-sample gain and background, parameterised as a
  multiplicative correction on a caller-provided gain initialisation
  (e.g. from a FITS zero point).
- ``SubPixelShift``: differentiable sub-pixel translation for astrometric
  refinement (regularise the shifts toward zero).
- ``InstrumentStateEncoder``: pools a variable number of evidence stamps
  (neighbouring cutouts, star stamps) into a low-dimensional instrument
  latent. By design the caller must never feed it the target pixels of the
  frame being rendered (counterfactual conditioning).
- ``LearnedInstrumentOperator``: composition of the above with a curriculum
  switch (``frozen`` -> ``core`` -> ``full``).

All modules are plain ``torch.nn`` components with no torchsky-internal
coupling so they can be used directly inside external forward models.
"""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .adaptive_axes import AdaptiveSpatioSpectralTemporalEncoder

__all__ = [
    "AffineRadiometry",
    "ConstrainedKernelPSF",
    "GatedResidualOperator",
    "InstrumentStateEncoder",
    "LearnedInstrumentOperator",
    "LowRankChromaticPSF",
    "SpatioSpectralTemporalEncoder",
    "ConditionedKernelPSF",
    "ExtendedResidualKernelPSF",
    "SubPixelShift",
    "gaussian_kernel2d",
    "gaussian_kernels_from_params",
]


class SpatioSpectralTemporalEncoder(nn.Module):
    """Fourier features with independent spatial, spectral, and time scales."""

    def __init__(
        self,
        *,
        spatial_center: tuple[float, float] = (0.0, 0.0),
        spatial_scale: float | tuple[float, float] = 1.0,
        wavelength_center: float = 0.0,
        spectral_scale: float = 1.0,
        time_center_mjd: float = 0.0,
        temporal_scale_days: float = 1.0,
        spatial_frequencies: int = 4,
        spectral_frequencies: int = 3,
        temporal_frequencies: int = 2,
    ) -> None:
        super().__init__()
        if len(spatial_center) != 2:
            raise ValueError("spatial_center must contain x and y")
        if not isinstance(spatial_scale, (int, float)) and len(spatial_scale) != 2:
            raise ValueError("spatial_scale must be scalar or contain x and y")
        spatial_scales = (
            (float(spatial_scale), float(spatial_scale))
            if isinstance(spatial_scale, (int, float))
            else (float(spatial_scale[0]), float(spatial_scale[1]))
        )
        values = (*spatial_scales, float(spectral_scale), float(temporal_scale_days))
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("conditioning scales must be finite and positive")
        frequencies = (
            int(spatial_frequencies),
            int(spectral_frequencies),
            int(temporal_frequencies),
        )
        if any(value < 0 for value in frequencies):
            raise ValueError("conditioning frequency counts must be non-negative")
        centers = (
            float(spatial_center[0]),
            float(spatial_center[1]),
            float(wavelength_center),
            float(time_center_mjd),
        )
        if not all(math.isfinite(value) for value in centers):
            raise ValueError("conditioning centers must be finite")
        self.spatial_center = centers[:2]
        self.spatial_scale = spatial_scales
        self.wavelength_center = centers[2]
        self.spectral_scale = float(spectral_scale)
        self.time_center_mjd = centers[3]
        self.temporal_scale_days = float(temporal_scale_days)
        self.spatial_frequencies = frequencies[0]
        self.spectral_frequencies = frequencies[1]
        self.temporal_frequencies = frequencies[2]
        self.output_dim = (
            4
            + 4 * self.spatial_frequencies
            + 2 * self.spectral_frequencies
            + 2 * self.temporal_frequencies
        )

    @staticmethod
    def _features(value: Tensor, frequencies: int) -> list[Tensor]:
        features = [value]
        for index in range(frequencies):
            phase = (2.0 * math.pi * (2.0**index)) * value
            features.extend((torch.sin(phase), torch.cos(phase)))
        return features

    def forward(
        self,
        spatial_x: Tensor,
        spatial_y: Tensor,
        wavelength: Tensor | float,
        time_mjd: Tensor | float | None = None,
    ) -> Tensor:
        wavelength_offset = (
            wavelength - self.wavelength_center
            if isinstance(wavelength, Tensor)
            else float(wavelength) - self.wavelength_center
        )
        if time_mjd is None:
            time_offset: Tensor | float = 0.0
        elif isinstance(time_mjd, Tensor):
            time_offset = time_mjd - self.time_center_mjd
        else:
            time_offset = float(time_mjd) - self.time_center_mjd
        wavelength_offset_tensor = torch.as_tensor(
            wavelength_offset, device=spatial_x.device, dtype=spatial_x.dtype
        )
        time_offset_tensor = torch.as_tensor(
            time_offset, device=spatial_x.device, dtype=spatial_x.dtype
        )
        x, y, spectral_offset, temporal_offset = torch.broadcast_tensors(
            spatial_x,
            spatial_y.to(device=spatial_x.device, dtype=spatial_x.dtype),
            wavelength_offset_tensor,
            time_offset_tensor,
        )
        x = (x - self.spatial_center[0]) / self.spatial_scale[0]
        y = (y - self.spatial_center[1]) / self.spatial_scale[1]
        spectral = spectral_offset / self.spectral_scale
        temporal = temporal_offset / self.temporal_scale_days
        features = self._features(x, self.spatial_frequencies)
        features.extend(self._features(y, self.spatial_frequencies))
        features.extend(self._features(spectral, self.spectral_frequencies))
        features.extend(self._features(temporal, self.temporal_frequencies))
        return torch.stack(features, dim=-1)


class ConditionedKernelPSF(nn.Module):
    """Flux-conserving non-stationary PSF from conditioned convex basis weights."""

    def __init__(
        self,
        encoder: SpatioSpectralTemporalEncoder
        | AdaptiveSpatioSpectralTemporalEncoder,
        *,
        n_basis: int = 4,
        kernel_size: int = 15,
        init_fwhm_pix: float = 3.0,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        if int(hidden_dim) < 1:
            raise ValueError("hidden_dim must be positive")
        self.encoder = encoder
        self.kernel_basis = ConstrainedKernelPSF(
            n_basis=n_basis,
            kernel_size=kernel_size,
            init_fwhm_pix=init_fwhm_pix,
        )
        self.mixer = nn.Sequential(
            nn.Linear(encoder.output_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(n_basis)),
        )
        nn.init.zeros_(self.mixer[-1].weight)
        nn.init.zeros_(self.mixer[-1].bias)

    @property
    def kernel_size(self) -> int:
        return self.kernel_basis.kernel_size

    def mixture_weights(
        self,
        spatial_x: Tensor,
        spatial_y: Tensor,
        wavelength: Tensor | float,
        time_mjd: Tensor | float | None = None,
    ) -> Tensor:
        return torch.softmax(
            self.mixer(
                self.encoder(spatial_x, spatial_y, wavelength, time_mjd)
            ),
            dim=-1,
        ).movedim(-1, 0)

    def conditioning_regularization(
        self,
        *,
        gate_weight: float = 1.0,
        scale_shift_weight: float = 1.0,
    ) -> Tensor:
        regularization = getattr(self.encoder, "regularization", None)
        if not callable(regularization):
            return self.kernel_basis.raw_kernels.sum() * 0.0
        return regularization(
            gate_weight=gate_weight,
            scale_shift_weight=scale_shift_weight,
        )

    def forward(
        self,
        image: Tensor,
        *,
        spatial_x: Tensor,
        spatial_y: Tensor,
        wavelength: Tensor | float,
        time_mjd: Tensor | float | None = None,
    ) -> Tensor:
        if image.ndim != 2:
            raise ValueError("ConditionedKernelPSF image must be 2D")
        if spatial_x.shape != image.shape or spatial_y.shape != image.shape:
            raise ValueError("PSF spatial coordinates must match image shape")
        mixture = self.mixture_weights(
            spatial_x, spatial_y, wavelength, time_mjd
        ).to(dtype=image.dtype)
        basis = self.kernel_basis.basis_kernels().to(
            device=image.device, dtype=image.dtype
        )
        weighted_sources = mixture * image.unsqueeze(0)
        convolved = F.conv2d(
            weighted_sources.unsqueeze(0),
            basis.unsqueeze(1),
            padding=self.kernel_size // 2,
            groups=self.kernel_basis.n_basis,
        )
        return convolved.squeeze(0).sum(dim=0)

    def effective_kernels(
        self,
        spatial_x: Tensor,
        spatial_y: Tensor,
        wavelength: Tensor | float,
        time_mjd: Tensor | float | None = None,
    ) -> Tensor:
        """Return normalized effective kernels with sample axes first."""
        mixture = self.mixture_weights(
            spatial_x, spatial_y, wavelength, time_mjd
        ).movedim(0, -1)
        return torch.einsum(
            "...b,bhw->...hw", mixture, self.kernel_basis.basis_kernels()
        )


def _center_pad_kernel(kernel: Tensor, size: int) -> Tensor:
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError("PSF kernels must be square 2D tensors")
    if int(kernel.shape[0]) % 2 != 1 or int(size) % 2 != 1:
        raise ValueError("PSF kernel sizes must be odd")
    if int(kernel.shape[0]) > int(size):
        raise ValueError("cannot pad a PSF kernel to a smaller size")
    pad = (int(size) - int(kernel.shape[0])) // 2
    return F.pad(kernel, (pad, pad, pad, pad))


class ExtendedResidualKernelPSF(nn.Module):
    """Fixed positive core+wing PSF with a small conditioned residual.

    The fixed optical baseline and the learned residual are both positive and
    unit-sum. Residual bases are centrosymmetrized before use, anchoring their
    centroid to the central pixel. A bounded convex fraction prevents the PSF
    residual from becoming an unrestricted image operator.
    """

    def __init__(
        self,
        encoder: SpatioSpectralTemporalEncoder
        | AdaptiveSpatioSpectralTemporalEncoder,
        *,
        core_kernel: Tensor | None = None,
        wing_kernel: Tensor | None = None,
        core_kernel_size: int = 15,
        wing_kernel_size: int = 51,
        core_fwhm_pix: float = 3.0,
        wing_fwhm_pix: float = 15.0,
        baseline_wing_fraction: float = 0.03,
        n_residual_basis: int = 4,
        residual_fraction_init: float = 1.0e-4,
        residual_fraction_max: float = 0.25,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        if not 0.0 <= float(baseline_wing_fraction) <= 1.0:
            raise ValueError("baseline_wing_fraction must lie in [0, 1]")
        if not 0.0 < float(residual_fraction_max) < 1.0:
            raise ValueError("residual_fraction_max must lie in (0, 1)")
        if not 0.0 < float(residual_fraction_init) < float(
            residual_fraction_max
        ):
            raise ValueError(
                "residual_fraction_init must lie in (0, residual_fraction_max)"
            )
        core = (
            gaussian_kernel2d(core_fwhm_pix / 2.354820045, core_kernel_size)
            if core_kernel is None
            else torch.as_tensor(core_kernel).detach().clone()
        )
        wing = (
            gaussian_kernel2d(wing_fwhm_pix / 2.354820045, wing_kernel_size)
            if wing_kernel is None
            else torch.as_tensor(wing_kernel).detach().clone()
        )
        if not core.is_floating_point() or not wing.is_floating_point():
            raise ValueError("PSF kernels must be floating point")
        if not bool(torch.isfinite(core).all()) or not bool(torch.isfinite(wing).all()):
            raise ValueError("PSF kernels must be finite")
        if bool((core < 0).any()) or bool((wing < 0).any()):
            raise ValueError("PSF kernels must be non-negative")
        if float(core.sum()) <= 0.0 or float(wing.sum()) <= 0.0:
            raise ValueError("PSF kernels must have positive sums")
        self.kernel_size = max(int(core.shape[0]), int(wing.shape[0]))
        if self.kernel_size % 2 != 1:
            raise ValueError("PSF kernel sizes must be odd")
        core = _center_pad_kernel(core / core.sum(), self.kernel_size)
        wing = _center_pad_kernel(wing / wing.sum(), self.kernel_size)
        baseline = (1.0 - float(baseline_wing_fraction)) * core + float(
            baseline_wing_fraction
        ) * wing
        self.register_buffer("baseline_kernel", baseline / baseline.sum())
        self.encoder = encoder
        self.kernel_basis = ConstrainedKernelPSF(
            n_basis=int(n_residual_basis),
            kernel_size=self.kernel_size,
            init_fwhm_pix=float(core_fwhm_pix),
        )
        self.mixer = nn.Sequential(
            nn.Linear(encoder.output_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(n_residual_basis)),
        )
        nn.init.zeros_(self.mixer[-1].weight)
        nn.init.zeros_(self.mixer[-1].bias)
        ratio = float(residual_fraction_init) / float(residual_fraction_max)
        self.raw_residual_fraction = nn.Parameter(
            torch.tensor(math.log(ratio / (1.0 - ratio)), dtype=torch.float32)
        )
        self.residual_fraction_max = float(residual_fraction_max)

    @property
    def residual_fraction(self) -> Tensor:
        return self.residual_fraction_max * torch.sigmoid(
            self.raw_residual_fraction
        )

    def mixture_weights(
        self,
        spatial_x: Tensor,
        spatial_y: Tensor,
        wavelength: Tensor | float,
        time_mjd: Tensor | float | None = None,
    ) -> Tensor:
        return torch.softmax(
            self.mixer(self.encoder(spatial_x, spatial_y, wavelength, time_mjd)),
            dim=-1,
        ).movedim(-1, 0)

    def _anchored_residual_basis(self) -> Tensor:
        basis = self.kernel_basis.basis_kernels()
        anchored = 0.5 * (basis + torch.flip(basis, dims=(-2, -1)))
        return anchored / anchored.sum(dim=(-2, -1), keepdim=True)

    def effective_kernels(
        self,
        spatial_x: Tensor,
        spatial_y: Tensor,
        wavelength: Tensor | float,
        time_mjd: Tensor | float | None = None,
    ) -> Tensor:
        mixture = self.mixture_weights(
            spatial_x, spatial_y, wavelength, time_mjd
        ).movedim(0, -1)
        residual = torch.einsum(
            "...b,bhw->...hw", mixture, self._anchored_residual_basis()
        )
        baseline = self.baseline_kernel.to(device=residual.device, dtype=residual.dtype)
        fraction = self.residual_fraction.to(device=residual.device, dtype=residual.dtype)
        return (1.0 - fraction) * baseline + fraction * residual

    def conditioning_regularization(
        self,
        *,
        gate_weight: float = 1.0,
        scale_shift_weight: float = 1.0,
    ) -> Tensor:
        regularization = getattr(self.encoder, "regularization", None)
        penalty = self.residual_fraction.square()
        if callable(regularization):
            penalty = penalty + regularization(
                gate_weight=gate_weight,
                scale_shift_weight=scale_shift_weight,
            )
        return penalty

    def forward(
        self,
        image: Tensor,
        *,
        spatial_x: Tensor,
        spatial_y: Tensor,
        wavelength: Tensor | float,
        time_mjd: Tensor | float | None = None,
    ) -> Tensor:
        if image.ndim != 2:
            raise ValueError("ExtendedResidualKernelPSF image must be 2D")
        if spatial_x.shape != image.shape or spatial_y.shape != image.shape:
            raise ValueError("PSF spatial coordinates must match image shape")
        mixture = self.mixture_weights(
            spatial_x, spatial_y, wavelength, time_mjd
        ).to(dtype=image.dtype)
        basis = self._anchored_residual_basis().to(
            device=image.device, dtype=image.dtype
        )
        residual = F.conv2d(
            (mixture * image.unsqueeze(0)).unsqueeze(0),
            basis.unsqueeze(1),
            padding=self.kernel_size // 2,
            groups=int(basis.shape[0]),
        ).squeeze(0).sum(dim=0)
        baseline_kernel = self.baseline_kernel.to(
            device=image.device, dtype=image.dtype
        )
        baseline = F.conv2d(
            image[None, None],
            baseline_kernel[None, None],
            padding=self.kernel_size // 2,
        )[0, 0]
        fraction = self.residual_fraction.to(device=image.device, dtype=image.dtype)
        return (1.0 - fraction) * baseline + fraction * residual


def gaussian_kernel2d(
    sigma_pix: float, kernel_size: int, *, device: torch.device | None = None
) -> Tensor:
    """Unit-sum 2D Gaussian kernel of odd size ``kernel_size``."""
    if kernel_size % 2 != 1:
        raise ValueError(f"kernel_size must be odd, got {kernel_size}")
    half = kernel_size // 2
    x = torch.arange(-half, half + 1, dtype=torch.float32, device=device)
    sig = max(float(sigma_pix), 1e-3)
    g1 = torch.exp(-0.5 * torch.square(x / sig))
    k = g1[:, None] * g1[None, :]
    return k / k.sum().clamp_min(1e-12)


class ConstrainedKernelPSF(nn.Module):
    """Quasi-convolution PSF core with positive, unit-sum basis kernels.

    The effective kernel is a convex mixture of ``n_basis`` learned basis
    kernels. Each basis kernel is constrained positive (softplus) and
    normalised to unit sum, so the operator conserves flux for any mixture.
    Mixture coefficients come either from a learned global logit vector or,
    when ``cond_dim > 0``, from a linear head over an instrument-state
    latent (spatially varying observing conditions).
    """

    def __init__(
        self,
        *,
        n_basis: int = 4,
        kernel_size: int = 15,
        init_fwhm_pix: float = 3.0,
        cond_dim: int = 0,
    ) -> None:
        super().__init__()
        if n_basis < 1:
            raise ValueError("n_basis must be >= 1")
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        self.n_basis = int(n_basis)
        self.kernel_size = int(kernel_size)
        self.cond_dim = int(cond_dim)

        sigma0 = max(float(init_fwhm_pix), 0.5) / 2.354820045
        # Spread initial widths around the nominal seeing so the mixture can
        # express narrower/wider profiles immediately.
        scales = torch.linspace(0.7, 1.6, self.n_basis)
        raw = []
        for s in scales.tolist():
            k = gaussian_kernel2d(sigma0 * s, self.kernel_size)
            # Inverse-softplus so softplus(raw) ~ k at init.
            raw.append(torch.log(torch.expm1(k.clamp_min(1e-10))))
        self.raw_kernels = nn.Parameter(torch.stack(raw, dim=0))

        self.coeff_logits = nn.Parameter(torch.zeros(self.n_basis))
        self.cond_head: nn.Linear | None = None
        if self.cond_dim > 0:
            self.cond_head = nn.Linear(self.cond_dim, self.n_basis)
            nn.init.zeros_(self.cond_head.weight)
            nn.init.zeros_(self.cond_head.bias)

    def basis_kernels(self) -> Tensor:
        """[K, ks, ks] positive, unit-sum basis kernels."""
        k = F.softplus(self.raw_kernels)
        return k / k.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)

    def mixture_coefficients(self, cond: Tensor | None, batch_size: int) -> Tensor:
        """[B, K] convex mixture coefficients."""
        logits = self.coeff_logits.unsqueeze(0).expand(batch_size, -1)
        if cond is not None and self.cond_head is not None:
            logits = logits + self.cond_head(cond)
        return F.softmax(logits, dim=-1)

    def effective_kernels(self, cond: Tensor | None, batch_size: int) -> Tensor:
        """[B, ks, ks] per-sample unit-sum kernels."""
        coeffs = self.mixture_coefficients(cond, batch_size)
        basis = self.basis_kernels()
        return torch.einsum("bk,kij->bij", coeffs, basis)

    def forward(self, image: Tensor, cond: Tensor | None = None) -> Tensor:
        """Convolve ``image`` [B,1,H,W] with per-sample effective kernels."""
        if image.ndim != 4 or image.shape[1] != 1:
            raise ValueError(f"image must be [B,1,H,W], got {tuple(image.shape)}")
        b = image.shape[0]
        kernels = self.effective_kernels(cond, b).to(dtype=image.dtype)
        pad = self.kernel_size // 2
        # Batched per-sample convolution via grouped conv with B groups.
        x = image.permute(1, 0, 2, 3)  # [1, B, H, W]
        w = kernels.unsqueeze(1)  # [B, 1, ks, ks]
        out = F.conv2d(x, w, padding=pad, groups=b)
        return out.permute(1, 0, 2, 3)


class GatedResidualOperator(nn.Module):
    """Small nonlinear residual with a scalar gate.

    ``out = x + gate * net(x, cond)``. The gate starts at zero so the
    operator is initially the identity; callers should L2-penalise
    :meth:`gate_penalty` so the residual only explains what constrained
    components cannot.
    """

    def __init__(self, *, hidden_channels: int = 16, cond_dim: int = 0) -> None:
        super().__init__()
        self.cond_dim = int(cond_dim)
        in_ch = 1 + (1 if self.cond_dim > 0 else 0)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )
        self.cond_proj: nn.Linear | None = None
        if self.cond_dim > 0:
            self.cond_proj = nn.Linear(self.cond_dim, 1)
        self.gate = nn.Parameter(torch.zeros(1))

    def gate_penalty(self) -> Tensor:
        return torch.square(self.gate).sum()

    def forward(self, image: Tensor, cond: Tensor | None = None) -> Tensor:
        if image.ndim != 4 or image.shape[1] != 1:
            raise ValueError(f"image must be [B,1,H,W], got {tuple(image.shape)}")
        x = image
        if self.cond_proj is not None:
            if cond is None:
                raise ValueError("cond is required when cond_dim > 0")
            ctx = self.cond_proj(cond).view(-1, 1, 1, 1)
            ctx_map = ctx.expand(-1, 1, image.shape[-2], image.shape[-1])
            x = torch.cat([image, ctx_map], dim=1)
        return image + self.gate * self.net(x)


class AffineRadiometry(nn.Module):
    """Per-sample gain and background: ``out = gain_init * exp(d) * x + b``.

    ``d`` and ``b`` come from a linear head over the instrument-state latent
    (or learned globals when ``cond_dim == 0``) and are initialised to zero,
    so the operator starts at the caller-provided gain initialisation
    (e.g. derived from a FITS zero point) with zero background.
    """

    def __init__(self, *, cond_dim: int = 0) -> None:
        super().__init__()
        self.cond_dim = int(cond_dim)
        self.global_params = nn.Parameter(torch.zeros(2))
        self.cond_head: nn.Linear | None = None
        if self.cond_dim > 0:
            self.cond_head = nn.Linear(self.cond_dim, 2)
            nn.init.zeros_(self.cond_head.weight)
            nn.init.zeros_(self.cond_head.bias)

    def gain_and_background(
        self, gain_init: Tensor, cond: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        b = gain_init.shape[0]
        params = self.global_params.unsqueeze(0).expand(b, -1)
        if cond is not None and self.cond_head is not None:
            params = params + self.cond_head(cond)
        gain = gain_init * torch.exp(params[:, 0])
        background = params[:, 1]
        return gain, background

    def forward(
        self, image: Tensor, *, gain_init: Tensor, cond: Tensor | None = None
    ) -> Tensor:
        if image.ndim != 4:
            raise ValueError(f"image must be [B,C,H,W], got {tuple(image.shape)}")
        gain, background = self.gain_and_background(
            gain_init.to(image.dtype),
            cond.to(image.dtype) if cond is not None else None,
        )
        return image * gain.view(-1, 1, 1, 1) + background.view(-1, 1, 1, 1)


class SubPixelShift(nn.Module):
    """Differentiable sub-pixel translation for astrometric refinement.

    Shifts are predicted by the caller (e.g. from an instrument-state
    latent) and should be L2-regularised toward zero; this module only
    applies them via bilinear resampling of the *model* image.
    """

    def forward(self, image: Tensor, shift_pix: Tensor) -> Tensor:
        """Shift ``image`` [B,C,H,W] by ``shift_pix`` [B,2] (dx, dy) pixels."""
        if image.ndim != 4:
            raise ValueError(f"image must be [B,C,H,W], got {tuple(image.shape)}")
        if shift_pix.ndim != 2 or shift_pix.shape[-1] != 2:
            raise ValueError(f"shift_pix must be [B,2], got {tuple(shift_pix.shape)}")
        b, _, h, w = image.shape
        # Normalised translation for affine_grid (align_corners=False).
        tx = -2.0 * shift_pix[:, 0] / max(w, 1)
        ty = -2.0 * shift_pix[:, 1] / max(h, 1)
        theta = torch.zeros(b, 2, 3, device=image.device, dtype=image.dtype)
        theta[:, 0, 0] = 1.0
        theta[:, 1, 1] = 1.0
        theta[:, 0, 2] = tx.to(image.dtype)
        theta[:, 1, 2] = ty.to(image.dtype)
        grid = F.affine_grid(theta, list(image.shape), align_corners=False)
        return F.grid_sample(
            image, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )


class InstrumentStateEncoder(nn.Module):
    """Pool evidence stamps into a low-dimensional instrument latent.

    Input is ``[B, N, C, h, w]`` (N stamps of local evidence: neighbouring
    cutouts of the same band, star stamps). Stamps flagged invalid via
    ``stamp_mask`` [B, N] are excluded from the pooling. Counterfactual
    rule (enforced by the caller): never include the target frame's own
    pixels for the band being rendered.
    """

    def __init__(
        self, *, in_channels: int = 2, latent_dim: int = 16, hidden_channels: int = 32
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1
            ),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(hidden_channels, self.latent_dim)

    def forward(self, stamps: Tensor, stamp_mask: Tensor | None = None) -> Tensor:
        if stamps.ndim != 5:
            raise ValueError(f"stamps must be [B,N,C,h,w], got {tuple(stamps.shape)}")
        b, n, c, h, w = stamps.shape
        feats = self.net(stamps.reshape(b * n, c, h, w)).reshape(b, n, -1)
        latents = self.proj(feats)  # [B, N, D]
        if stamp_mask is None:
            return latents.mean(dim=1)
        m = stamp_mask.to(latents.dtype).unsqueeze(-1)  # [B, N, 1]
        denom = m.sum(dim=1).clamp_min(1e-6)
        return (latents * m).sum(dim=1) / denom


class LearnedInstrumentOperator(nn.Module):
    """Constrained kernel core + gated residual + affine radiometry.

    Composition order: astrometric shift -> PSF core -> nonlinear residual
    -> radiometry. A curriculum ``stage`` controls which parts train:

    - ``"frozen"``: kernel core fixed at its Gaussian initialisation,
      residual gate fixed at zero (operator is Gaussian PSF + radiometry).
    - ``"core"``: kernel core trains, residual stays gated off.
    - ``"full"``: everything trains.
    """

    STAGES: Sequence[str] = ("frozen", "core", "full")

    def __init__(
        self,
        *,
        n_basis: int = 4,
        kernel_size: int = 15,
        init_fwhm_pix: float = 3.0,
        cond_dim: int = 0,
        residual_hidden: int = 16,
        max_shift_pix: float = 2.0,
    ) -> None:
        super().__init__()
        self.cond_dim = int(cond_dim)
        self.max_shift_pix = float(max_shift_pix)
        self.psf = ConstrainedKernelPSF(
            n_basis=n_basis,
            kernel_size=kernel_size,
            init_fwhm_pix=init_fwhm_pix,
            cond_dim=cond_dim,
        )
        self.residual = GatedResidualOperator(
            hidden_channels=residual_hidden, cond_dim=cond_dim
        )
        self.radiometry = AffineRadiometry(cond_dim=cond_dim)
        self.shifter = SubPixelShift()
        self.shift_head: nn.Linear | None = None
        if self.cond_dim > 0:
            self.shift_head = nn.Linear(self.cond_dim, 2)
            nn.init.zeros_(self.shift_head.weight)
            nn.init.zeros_(self.shift_head.bias)
        self._stage = "frozen"
        self.set_stage("frozen")

    @property
    def stage(self) -> str:
        return self._stage

    def set_stage(self, stage: str) -> None:
        if stage not in self.STAGES:
            raise ValueError(
                f"stage must be one of {tuple(self.STAGES)}, got {stage!r}"
            )
        self._stage = stage
        core_trainable = stage in ("core", "full")
        self.psf.raw_kernels.requires_grad_(core_trainable)
        residual_trainable = stage == "full"
        for p in self.residual.parameters():
            p.requires_grad_(residual_trainable)

    def predicted_shift(self, cond: Tensor | None, batch_size: int) -> Tensor:
        if cond is None or self.shift_head is None:
            device = None if cond is None else cond.device
            dtype = None if cond is None else cond.dtype
            return torch.zeros(batch_size, 2, device=device, dtype=dtype)
        return self.max_shift_pix * torch.tanh(self.shift_head(cond))

    def regularization(self, cond: Tensor | None = None) -> Tensor:
        reg = self.residual.gate_penalty()
        if cond is not None and self.shift_head is not None:
            shift = self.predicted_shift(cond, cond.shape[0])
            reg = reg + torch.square(shift / max(self.max_shift_pix, 1e-6)).mean()
        return reg

    def forward(
        self,
        image: Tensor,
        *,
        gain_init: Tensor,
        cond: Tensor | None = None,
    ) -> Tensor:
        """Apply the full operator to a model image [B,1,H,W]."""
        b = image.shape[0]
        out = image
        if cond is not None and self.shift_head is not None:
            shift = self.predicted_shift(cond, b).to(
                device=image.device, dtype=image.dtype
            )
            out = self.shifter(out, shift)
        out = self.psf(out, cond)
        if self._stage == "full":
            out = self.residual(out, cond)
        return self.radiometry(out, gain_init=gain_init, cond=cond)


def _legendre_basis(x: Tensor, degree: int) -> Tensor:
    """Legendre polynomials L_0..L_degree evaluated at ``x`` -> ``[D, ...]``."""
    terms = [torch.ones_like(x), x]
    for k in range(2, degree + 1):
        terms.append(((2 * k - 1) * x * terms[-1] - (k - 1) * terms[-2]) / k)
    return torch.stack(terms[: degree + 1], dim=0)


def gaussian_kernels_from_params(
    params: Tensor, *, size: int | None = None
) -> Tensor:
    """Unit-sum, zero-centroid Gaussian kernels from ``(sigma_x, sigma_y, rho)``.

    ``params`` is ``[..., 3]``; returns ``[..., ky, kx]`` with odd ``ky == kx``.
    Unit integral and zero centroid are enforced by construction: the kernel
    is normalized to unit sum, its discrete centroid measured, and the
    analytic mean re-centered once (residual centroid is second order).
    """
    if params.ndim < 1 or params.shape[-1] != 3:
        raise ValueError("params must have trailing shape (3,)")
    sigma_x = params[..., 0]
    sigma_y = params[..., 1]
    rho = torch.clamp(params[..., 2], -0.99, 0.99)
    if size is None:
        sigma_max = torch.max(sigma_x, sigma_y)
        radius = max(1, int(torch.ceil(3.0 * sigma_max.max()).item()))
        size = 2 * radius + 1
    elif size % 2 == 0:
        raise ValueError("kernel size must be odd")
    center = size // 2
    grid = torch.arange(size, device=params.device, dtype=params.dtype) - center
    shape = params.shape[:-1] + (size, size)
    gx = grid.view(*([1] * (params.ndim - 1)), 1, size).expand(shape)
    gy = grid.view(*([1] * (params.ndim - 1)), size, 1).expand(shape)
    one_minus = (1.0 - rho**2).clamp_min(1e-6)
    sx = sigma_x.unsqueeze(-1).unsqueeze(-1)
    sy = sigma_y.unsqueeze(-1).unsqueeze(-1)
    rho_u = rho.unsqueeze(-1).unsqueeze(-1)

    def _kernel(mx: Tensor, my: Tensor) -> Tensor:
        ex = (gx - mx.unsqueeze(-1).unsqueeze(-1)) / sx
        ey = (gy - my.unsqueeze(-1).unsqueeze(-1)) / sy
        quad = (ex**2 - 2.0 * rho_u * ex * ey + ey**2) / one_minus.unsqueeze(
            -1
        ).unsqueeze(-1)
        kern = torch.exp(-0.5 * quad)
        return kern / kern.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    mean_x = torch.zeros_like(sigma_x)
    mean_y = torch.zeros_like(sigma_y)
    kern = _kernel(mean_x, mean_y)
    for _ in range(8):
        centroid_x = (kern * gx).sum(dim=(-2, -1))
        centroid_y = (kern * gy).sum(dim=(-2, -1))
        mean_x = mean_x + centroid_x
        mean_y = mean_y + centroid_y
        kern = _kernel(mean_x, mean_y)
        if (
            (kern * gx).sum(dim=(-2, -1)).abs().max() < 1e-6
            and (kern * gy).sum(dim=(-2, -1)).abs().max() < 1e-6
        ):
            break
    return kern


class LowRankChromaticPSF(nn.Module):
    """Low-rank chromatic PSF: spatial control grid x Legendre spectral basis.

    Kernel parameters ``(sigma_x, sigma_y, rho)`` at focal-plane position
    ``u_pix`` and normalized wavelength ``lam_norm`` are bilinear interpolants
    of per-node log-sigma / ellipticity / position-angle offsets modulated by
    an in-band Legendre spectral basis. Positivity holds by construction
    (softplus/exponential parameterization); unit integral and zero centroid
    are enforced downstream by :func:`gaussian_kernels_from_params`. At zero
    initialization the model reproduces a stationary Gaussian with width
    ``base_sigma_pix``.
    """

    def __init__(
        self,
        *,
        grid_side: int = 4,
        spectral_degree: int = 2,
        base_sigma_pix: float,
        extent_pix: float = 2048.0,
        kernel_size: int | None = None,
        wavelength_range_um: tuple[float, float] = (0.5, 5.0),
    ) -> None:
        super().__init__()
        if int(grid_side) < 1:
            raise ValueError("grid_side must be >= 1")
        if int(spectral_degree) < 0:
            raise ValueError("spectral_degree must be >= 0")
        if not (math.isfinite(base_sigma_pix) and base_sigma_pix > 0.0):
            raise ValueError("base_sigma_pix must be finite and positive")
        self.grid_side = int(grid_side)
        self.spectral_degree = int(spectral_degree)
        self.base_sigma_pix = float(base_sigma_pix)
        n_nodes = self.grid_side * self.grid_side
        self.raw_log_sigma = nn.Parameter(
            torch.zeros(self.spectral_degree + 1, n_nodes)
        )
        self.raw_ellipticity = nn.Parameter(
            torch.zeros(self.spectral_degree + 1, n_nodes)
        )
        self.raw_pa_rad = nn.Parameter(torch.zeros(self.spectral_degree + 1, n_nodes))
        self.register_buffer("extent_pix", torch.tensor(float(extent_pix)))
        nodes = torch.linspace(-1.0, 1.0, self.grid_side)
        node_y, node_x = torch.meshgrid(nodes, nodes, indexing="ij")
        self.register_buffer(
            "node_uvw",
            torch.stack([node_x, node_y], dim=-1).reshape(n_nodes, 2),
        )
        self._kernel_size = (
            int(kernel_size) if kernel_size is not None else None
        )
        if self._kernel_size is not None and self._kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.register_buffer(
            "wavelength_range_um",
            torch.tensor([float(v) for v in wavelength_range_um]),
        )

    @property
    def n_cells(self) -> int:
        return self.grid_side * self.grid_side

    @property
    def default_kernel_size(self) -> int:
        radius = max(1, math.ceil(3.0 * self.base_sigma_pix))
        return min(2 * radius + 1, 101)

    @property
    def kernel_size(self) -> int:
        if self._kernel_size is None:
            return self.default_kernel_size
        return int(self._kernel_size)

    def normalized_wavelength(self, wavelength_um: Tensor) -> Tensor:
        """Map vacuum wavelengths to the in-band ``lam_norm`` in [0, 1].

        ponytail: accepts microns or Angstroms -- torchsky calibration
        evidence carries ``pivot_wavelength_angstrom``, and any real NIRCam
        wavelength exceeds 100 in Angstroms but not in microns. Upgrade path
        is an explicit unit parameter at the call sites.
        """
        lo, hi = (float(v) for v in self.wavelength_range_um)
        if not (math.isfinite(lo) and math.isfinite(hi) and hi > lo):
            raise ValueError("wavelength_range_um must be a finite increasing pair")
        wl = torch.as_tensor(wavelength_um, dtype=torch.float64)
        if float(wl.median()) > 100.0 * max(hi, 1.0):
            wl = wl / 1.0e4
        wl = wl.clamp(lo, hi)
        return ((wl - lo) / (hi - lo)).to(torch.float32)

    def effective_kernels(
        self,
        spatial_x: Tensor,
        spatial_y: Tensor,
        wavelength: Tensor | float,
        time_mjd: Tensor | float | None = None,
    ) -> Tensor:
        """ConditionedKernelPSF-compatible kernel rendering for calibration."""
        u_pix = torch.stack(
            (
                torch.as_tensor(spatial_x).reshape(-1),
                torch.as_tensor(spatial_y).reshape(-1),
            ),
            dim=-1,
        )
        lam_norm = self.normalized_wavelength(
            torch.as_tensor(wavelength).reshape(-1)
        ).to(u_pix.device)
        params = self(u_pix, lam_norm)
        return gaussian_kernels_from_params(params, size=self.kernel_size)

    def forward(self, u_pix: Tensor, lam_norm: Tensor | float) -> Tensor:
        """Return stacked ``(sigma_x, sigma_y, rho)`` of shape ``[N, 3]``."""
        if u_pix.ndim != 2 or u_pix.shape[-1] != 2:
            raise ValueError("u_pix must have shape [N, 2]")
        lam = torch.as_tensor(
            lam_norm, device=u_pix.device, dtype=u_pix.dtype
        ).expand(u_pix.shape[0])
        # Legendre polynomials are orthogonal on [-1, 1]; lam_norm lives on
        # [0, 1], so rescale before evaluating the spectral basis.
        spec = _legendre_basis(2.0 * lam - 1.0, self.spectral_degree).to(
            device=self.raw_log_sigma.device, dtype=self.raw_log_sigma.dtype
        )
        u_norm = u_pix.to(self.extent_pix.device, self.extent_pix.dtype) / (
            self.extent_pix.clamp_min(1.0)
        )
        grid = u_norm.view(1, 1, -1, 2)[..., [0, 1]].clamp(-1.0, 1.0)

        def _interp(field: Tensor) -> Tensor:
            sampled = F.grid_sample(
                field.view(1, -1, self.grid_side, self.grid_side),
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            return sampled.view(-1, u_pix.shape[0])

        # Spectral mixing happens on the raw offsets; the nonlinearity is
        # applied once to the summed (spatial x spectral) field.
        sigma_value = (_interp(self.raw_log_sigma) * spec).sum(dim=0)
        e_value = (_interp(self.raw_ellipticity) * spec).sum(dim=0)
        theta = (_interp(self.raw_pa_rad) * spec).sum(dim=0)
        # softplus^-1(1) shift: zero offsets give a multiplier of exactly 1.
        sigma = self.base_sigma_pix * F.softplus(
            sigma_value + math.log(math.expm1(1.0))
        )
        e = torch.tanh(e_value)
        sigma_major = sigma * torch.sqrt(1.0 + e)
        sigma_minor = sigma / torch.sqrt((1.0 + e).clamp_min(1e-3))
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        var_x = sigma_major**2 * cos_t**2 + sigma_minor**2 * sin_t**2
        var_y = sigma_major**2 * sin_t**2 + sigma_minor**2 * cos_t**2
        cov_xy = (sigma_major**2 - sigma_minor**2) * sin_t * cos_t
        sigma_x = var_x.clamp_min(1e-8).sqrt()
        sigma_y = var_y.clamp_min(1e-8).sqrt()
        rho = cov_xy / (sigma_x * sigma_y)
        return torch.stack([sigma_x, sigma_y, rho], dim=-1)

    def smoothness_penalty(self) -> Tensor:
        """Discrete Laplacian energy on each control-grid coefficient field."""
        penalty: Tensor | None = None
        for field in (self.raw_log_sigma, self.raw_ellipticity, self.raw_pa_rad):
            grid_map = field.view(-1, self.grid_side, self.grid_side)
            if self.grid_side >= 3:
                lap = (
                    4.0 * grid_map[:, 1:-1, 1:-1]
                    - grid_map[:, :-2, 1:-1]
                    - grid_map[:, 2:, 1:-1]
                    - grid_map[:, 1:-1, :-2]
                    - grid_map[:, 1:-1, 2:]
                )
                total = (lap**2).sum()
            else:
                total = grid_map.sum() * 0.0
            penalty = total if penalty is None else penalty + total
        assert penalty is not None
        return penalty
