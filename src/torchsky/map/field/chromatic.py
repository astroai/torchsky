"""Chromatic PSF fields and continuous-wavelength sky fields for multiband forward models.

These are the building blocks of a forward model in which a single sky
``s(x, y, lambda)`` is observed through each band's transmission curve and a
PSF that varies smoothly with wavelength and detector position:

``m_b(p) = sum_q w_bq sum_k u_k(p; lambda_b) [K_k(.; lambda_bq) * s(., lambda_bq)](T(p))``

* :class:`ChromaticPSFField` -- one network for every band: kernel coordinates
  are expressed in diffraction units ``dx * pixel / (lambda / D)`` so the
  Airy-like structure scales with wavelength by construction, plus pixel-unit
  coordinates for pixel response and jitter. Kernel ``0`` is a normalised,
  non-negative core; kernels ``1..`` are zero-sum perturbations mixed with
  weights that vary smoothly over the detector.
* :class:`LogLambdaRBFBasis` -- a partition-of-unity spectral basis uniform in
  ``log lambda``, optionally with a learned smooth correction.
* :class:`SpectralSkyField` -- ``s(x, y, lambda) = sum_k a_k(x, y) phi_k(lambda)``
  with any coefficient provider (pixel grid or neural field) and band
  quadrature nodes from :func:`torchsky.filter_profiles.photon_quadrature`.

Everything is plain ``torch`` and runs on CPU, CUDA and MPS.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = [
    "ChromaticPSFField",
    "LogLambdaRBFBasis",
    "SpectralSkyField",
    "spectral_coefficients_from_bands",
]

_ARCSEC_PER_RAD = 180.0 / math.pi * 3600.0


def _fourier(x: Tensor, num_frequencies: int) -> Tensor:
    freqs = (
        2.0 ** torch.arange(num_frequencies, device=x.device, dtype=x.dtype)
    ) * math.pi
    xf = x[..., None] * freqs
    return torch.cat([torch.sin(xf), torch.cos(xf)], -1).flatten(-2)


class ChromaticPSFField(nn.Module):
    """PSF basis field smooth in wavelength and detector position.

    Parameters
    ----------
    num_channels:
        Number of optical channels sharing the field (e.g. 2 for NIRCam SW/LW);
        a channel embedding conditions the kernel network.
    num_detectors:
        Detectors with their own (learned) embedding for the spatial weights.
    pixel_scale_arcsec:
        Pixel scale of the grid the kernels are sampled on.
    aperture_m:
        Telescope diameter, used for ``lambda / D``.
    num_kernels:
        Kernel 0 is the normalised core, the others zero-sum perturbations.
    size:
        Odd kernel side in grid pixels.
    max_diffraction_radius:
        Normaliser of the diffraction-unit coordinates (in ``lambda / D``).
    """

    _offsets: Tensor

    def __init__(
        self,
        *,
        num_channels: int,
        num_detectors: int,
        pixel_scale_arcsec: float,
        aperture_m: float = 6.5,
        num_kernels: int = 4,
        size: int = 31,
        hidden: int = 96,
        max_diffraction_radius: float = 24.0,
        lambda_ref_um: float = 2.0,
    ) -> None:
        super().__init__()
        if size % 2 != 1:
            raise ValueError("size must be odd")
        if num_kernels < 1:
            raise ValueError("num_kernels must be positive")
        self.size, self.num_kernels, self.num_channels = (
            int(size),
            int(num_kernels),
            int(num_channels),
        )
        self.pixel_scale_arcsec = float(pixel_scale_arcsec)
        self.aperture_m = float(aperture_m)
        self.max_diffraction_radius = float(max_diffraction_radius)
        self.lambda_ref_um = float(lambda_ref_um)
        cond = 1 + num_channels
        self.kernel_net = nn.Sequential(
            nn.Linear(2 * 2 * 6 + 2 * 2 * 4 + cond, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, num_kernels),
        )
        self.detector_embedding = nn.Embedding(num_detectors, 4)
        self.weight_net = nn.Sequential(
            nn.Linear(2 * 2 * 3 + cond + 4, 48),
            nn.SiLU(),
            nn.Linear(48, 48),
            nn.SiLU(),
            nn.Linear(48, max(num_kernels - 1, 1)),
        )
        weight_out = self.weight_net[-1]
        assert isinstance(weight_out, nn.Linear)  # last layer built above
        nn.init.zeros_(weight_out.weight)
        nn.init.zeros_(weight_out.bias)
        r = torch.arange(self.size, dtype=torch.float32) - self.size // 2
        yy, xx = torch.meshgrid(r, r, indexing="ij")
        self.register_buffer(
            "_offsets", torch.stack([xx, yy], -1).reshape(-1, 2), persistent=False
        )

    def _condition(self, lam_um: Tensor, channel: Tensor) -> Tensor:
        loglam = (torch.log(lam_um) - math.log(self.lambda_ref_um)) / 0.8
        onehot = F.one_hot(channel.long(), self.num_channels).to(lam_um.dtype)
        return torch.cat([loglam[:, None], onehot], 1)

    def basis(self, lam_um: Tensor, channel: Tensor) -> Tensor:
        """Kernels at wavelengths ``lam_um`` (M,) and channels (M,): ``(M, num_kernels, size, size)``."""
        lam_um = torch.as_tensor(
            lam_um, dtype=torch.float32, device=self._offsets.device
        ).flatten()
        channel = torch.as_tensor(channel, device=self._offsets.device).flatten()
        if lam_um.shape != channel.shape:
            raise ValueError("lam_um and channel must have the same length")
        lod_arcsec = lam_um * 1e-6 / self.aperture_m * _ARCSEC_PER_RAD
        pix_per_lod = self.pixel_scale_arcsec / lod_arcsec
        xy = self._offsets
        num_offsets = xy.shape[0]
        cond = self._condition(lam_um, channel)
        rows = []
        for m in range(lam_um.shape[0]):
            rho = xy * pix_per_lod[m] / self.max_diffraction_radius
            rows.append(
                torch.cat(
                    [
                        _fourier(rho, 6),
                        _fourier(xy / self.size, 4),
                        cond[m].expand(num_offsets, -1),
                    ],
                    1,
                )
            )
        k = self.kernel_net(torch.cat(rows, 0)).reshape(
            lam_um.shape[0], num_offsets, self.num_kernels
        )
        k = k.permute(0, 2, 1).reshape(
            lam_um.shape[0], self.num_kernels, self.size, self.size
        )
        core = F.softplus(k[:, 0])
        core = core / core.sum((-2, -1), keepdim=True)
        rest = 0.01 * (k[:, 1:] - k[:, 1:].mean((-2, -1), keepdim=True))
        return torch.cat([core[:, None], rest], 1)

    def weights(
        self, lam_um: Tensor, channel: Tensor, detector: Tensor, u: Tensor, v: Tensor
    ) -> Tensor:
        """Mixing weights ``(N, num_kernels)`` at detector coordinates ``u, v`` in ``[-1, 1]``.

        ``lam_um``, ``channel`` and ``detector`` broadcast against ``u``.
        """
        n = u.shape[0]
        lam_um = torch.as_tensor(lam_um, dtype=u.dtype, device=u.device).expand(n)
        channel = torch.as_tensor(channel, device=u.device).expand(n)
        detector = torch.as_tensor(detector, device=u.device).long().expand(n)
        inp = torch.cat(
            [
                _fourier(torch.stack([u, v], -1) * 0.5, 3),
                self._condition(lam_um, channel),
                self.detector_embedding(detector),
            ],
            1,
        )
        ones = torch.ones(n, 1, device=u.device, dtype=u.dtype)
        if self.num_kernels == 1:
            return ones
        return torch.cat([ones, self.weight_net(inp)], 1)

    def centring_penalty(self, lam_um: Tensor, channel: Tensor) -> Tensor:
        """Sum of squared first moments of every kernel: pins the PSF centroid, which is
        otherwise degenerate with the astrometric solution."""
        b = self.basis(lam_um, channel)
        r = torch.arange(self.size, device=b.device, dtype=b.dtype) - self.size // 2
        mx = (b.sum(-2) * r).sum(-1)
        my = (b.sum(-1) * r).sum(-1)
        return (mx.pow(2) + my.pow(2)).sum()


class LogLambdaRBFBasis(nn.Module):
    """``phi_k(lambda)``: ``K`` normalised Gaussian bumps uniform in ``log lambda`` on
    ``[lam_min, lam_max]`` (a partition of unity), optionally plus a learned smooth
    correction initialised at zero."""

    centres: Tensor

    def __init__(
        self,
        num_basis: int,
        lam_min_um: float,
        lam_max_um: float,
        *,
        learned: bool = False,
    ) -> None:
        super().__init__()
        if num_basis < 2:
            raise ValueError("num_basis must be at least 2")
        if not 0.0 < lam_min_um < lam_max_um:
            raise ValueError("need 0 < lam_min_um < lam_max_um")
        self.num_basis = int(num_basis)
        self.log_min, self.log_max = math.log(lam_min_um), math.log(lam_max_um)
        self.register_buffer("centres", torch.linspace(0.0, 1.0, num_basis))
        self.width = 1.0 / (num_basis - 1)
        self.correction: nn.Sequential | None = None
        if learned:
            self.correction = nn.Sequential(
                nn.Linear(2 * 6, 32), nn.SiLU(), nn.Linear(32, num_basis)
            )
            correction_out = self.correction[-1]
            assert isinstance(correction_out, nn.Linear)  # last layer built above
            nn.init.zeros_(correction_out.weight)
            nn.init.zeros_(correction_out.bias)

    def unit(self, lam_um: Tensor) -> Tensor:
        return (torch.log(lam_um) - self.log_min) / (self.log_max - self.log_min)

    def forward(self, lam_um: Tensor) -> Tensor:
        u = self.unit(
            torch.as_tensor(
                lam_um, dtype=self.centres.dtype, device=self.centres.device
            ).flatten()
        )
        g = torch.exp(-0.5 * ((u[:, None] - self.centres[None]) / self.width) ** 2)
        phi = g / g.sum(1, keepdim=True)
        if self.correction is not None:
            phi = phi + 0.1 * self.correction(_fourier(u[:, None], 6))
        return phi


class SpectralSkyField(nn.Module):
    """Continuous sky ``s(x, y, lambda) = sum_k a_k(x, y) phi_k(lambda)``.

    ``coefficients`` is any module returning ``(K, H, W)`` (a pixel grid or a neural
    field); ``nodes_um`` / ``node_weights`` are ``(B, Q)`` band quadrature nodes, e.g.
    from :func:`torchsky.filter_profiles.photon_quadrature`.

    The spectral prior penalises the second difference of ``a_k`` across ``k``.
    ``smoothness_mode="relative"`` (the default) divides it by each pixel's own
    spectral level (``mean_k |a_k| + floor * scale``), so a colour gradient costs the
    same for a faint galaxy as for a bright one; ``"absolute"`` divides by ``scale``
    only and flattens the spectra of bright sources.
    """

    nodes_um: Tensor
    node_weights: Tensor

    def __init__(
        self,
        coefficients: nn.Module,
        basis: LogLambdaRBFBasis,
        nodes_um: Tensor,
        node_weights: Tensor,
        *,
        spectral_smoothness: float = 0.0,
        scale: float = 1.0,
        smoothness_mode: str = "relative",
        floor: float = 3.0,
    ) -> None:
        super().__init__()
        if nodes_um.shape != node_weights.shape or nodes_um.dim() != 2:
            raise ValueError("nodes_um and node_weights must both be (bands, nodes)")
        self.coefficients, self.basis = coefficients, basis
        self.register_buffer("nodes_um", nodes_um.float())
        self.register_buffer("node_weights", node_weights.float())
        if smoothness_mode not in ("relative", "absolute"):
            raise ValueError("smoothness_mode must be 'relative' or 'absolute'")
        self.spectral_smoothness, self.scale = float(spectral_smoothness), float(scale)
        self.smoothness_mode, self.floor = smoothness_mode, float(floor)

    def at_nodes(self) -> tuple[Tensor, Tensor]:
        """``(B, Q, H, W)`` sky at every quadrature node and ``(B, H, W)`` band averages."""
        a = self.coefficients()
        nb, q = self.nodes_um.shape
        phi = self.basis(self.nodes_um.flatten())
        sn = torch.einsum("mk,khw->mhw", phi, a).view(nb, q, *a.shape[-2:])
        return sn, (sn * self.node_weights[:, :, None, None]).sum(1)

    def forward(self) -> Tensor:
        return self.at_nodes()[1]

    def spectral_cube(self, lam_um: Tensor) -> Tensor:
        return torch.einsum("mk,khw->mhw", self.basis(lam_um), self.coefficients())

    def penalty(self) -> Tensor:
        if self.spectral_smoothness <= 0.0:
            return torch.zeros((), device=self.nodes_um.device)
        a = self.coefficients()
        d2 = a[2:] - 2.0 * a[1:-1] + a[:-2]
        if self.smoothness_mode == "relative":
            level = (a.abs().mean(0, keepdim=True) + self.floor * self.scale).detach()
            return self.spectral_smoothness * (d2 / level).pow(2).mean()
        return self.spectral_smoothness * (d2 / self.scale).pow(2).mean()


def spectral_coefficients_from_bands(
    band_images: Tensor,
    nodes_um: Tensor,
    node_weights: Tensor,
    basis: LogLambdaRBFBasis,
    *,
    smoothness: float = 1e-5,
) -> Tensor:
    """Initial ``a_k`` that reproduce ``band_images`` (B, H, W) through the band
    quadrature: ridge least squares with a second-difference prior across ``k``."""
    with torch.no_grad():
        nb, q = nodes_um.shape
        phi = basis(nodes_um.flatten()).view(nb, q, -1)
        design = (phi * node_weights[:, :, None].to(phi)).sum(1)  # B K
        k = design.shape[1]
        d = torch.zeros(max(k - 2, 0), k, dtype=torch.float64)
        for i in range(k - 2):
            d[i, i : i + 3] = torch.tensor([1.0, -2.0, 1.0], dtype=torch.float64)
        dm = design.double().cpu()
        normal = (
            dm.T @ dm + smoothness * d.T @ d + 1e-6 * torch.eye(k, dtype=torch.float64)
        )
        proj = torch.linalg.solve(normal, dm.T).to(
            device=band_images.device, dtype=band_images.dtype
        )
        return torch.einsum("kb,bhw->khw", proj, band_images)
