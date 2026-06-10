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

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = [
    "AffineRadiometry",
    "ConstrainedKernelPSF",
    "GatedResidualOperator",
    "InstrumentStateEncoder",
    "LearnedInstrumentOperator",
    "SubPixelShift",
    "gaussian_kernel2d",
]


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
            gain_init.to(image.dtype), cond.to(image.dtype) if cond is not None else None
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
            raise ValueError(f"stage must be one of {tuple(self.STAGES)}, got {stage!r}")
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
