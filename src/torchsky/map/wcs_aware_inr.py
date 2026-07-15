"""WCS-aware multi-wavelength INR for native JWST exposure pixels.

The scene is trained directly against native detector pixels.  Each forward
pass evaluates the fixed exposure WCS, maps the pixel into a union TAN canvas,
encodes position (planar Fourier or spherical), applies trainable alignment in
feature space (option A), and decodes a shared wavelength-conditioned scene.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal, Sequence

import torch
from torch import Tensor, nn

from ..wcs import WCS
from ..encoders import harmonic_sky_encoding
from ..sphere.healpix import spherical_fourier_features

EncodingKind = Literal["fourier", "spherical"]


@dataclass(frozen=True)
class UnionCanvas:
    header: dict[str, Any]
    shape: tuple[int, int]
    pixel_scale_arcsec: float


@dataclass(frozen=True)
class BandpassTable:
    wavelength: Tensor  # [E,L]
    weight: Tensor      # [E,L]

    def __post_init__(self) -> None:
        if self.wavelength.ndim != 2 or self.weight.shape != self.wavelength.shape:
            raise ValueError("bandpass tensors must have identical [E,L] shapes")
        if bool((self.wavelength <= 0).any()) or bool((self.weight < 0).any()):
            raise ValueError("invalid bandpass values")

    def normalized(self) -> "BandpassTable":
        return BandpassTable(
            self.wavelength,
            self.weight / self.weight.sum(-1, keepdim=True).clamp_min(1e-20),
        )

    def to(self, device: torch.device | str) -> "BandpassTable":
        return BandpassTable(self.wavelength.to(device), self.weight.to(device))


@dataclass(frozen=True)
class PSFKernelBank:
    kernel: Tensor  # [E,L,K,K]

    def __post_init__(self) -> None:
        if self.kernel.ndim != 4 or self.kernel.shape[-1] != self.kernel.shape[-2]:
            raise ValueError("PSF kernels must have shape [E,L,K,K]")
        if self.kernel.shape[-1] % 2 != 1:
            raise ValueError("PSF kernel size must be odd")

    def normalized(self) -> "PSFKernelBank":
        return PSFKernelBank(
            self.kernel / self.kernel.sum((-1, -2), keepdim=True).clamp_min(1e-20)
        )

    def to(self, device: torch.device | str) -> "PSFKernelBank":
        return PSFKernelBank(self.kernel.to(device))


def _pixel_scale_arcsec(wcs: WCS) -> float:
    det = float(torch.det(wcs.cd.detach().cpu().double()).abs())
    if not math.isfinite(det) or det <= 0:
        raise ValueError("singular spatial WCS")
    return math.sqrt(det) * 3600.0


def _circular_mean_deg(ra: Tensor) -> float:
    angle = torch.deg2rad(ra.double())
    return float(torch.rad2deg(torch.atan2(torch.sin(angle).mean(), torch.cos(angle).mean())) % 360)


def build_union_canvas(
    headers: Sequence[dict[str, Any]],
    *,
    pixel_scale_arcsec: float | None = None,
    padding_pixels: int = 8,
) -> UnionCanvas:
    """Plan a rectangular TAN canvas covering the union of WCS footprints.

    No image values are resampled here.
    """
    if not headers:
        raise ValueError("at least one exposure header is required")
    footprints: list[Tensor] = []
    scales: list[float] = []
    for header in headers:
        wcs = WCS(dict(header))
        fp = torch.as_tensor(wcs.footprint(), dtype=torch.float64)
        fp = fp[torch.isfinite(fp).all(-1)]
        if fp.numel() == 0:
            raise ValueError("non-finite WCS footprint")
        footprints.append(fp)
        scales.append(_pixel_scale_arcsec(wcs))
    world = torch.cat(footprints)
    ra0 = _circular_mean_deg(world[:, 0])
    dec0 = float(world[:, 1].mean())
    scale_arcsec = float(pixel_scale_arcsec) if pixel_scale_arcsec else min(scales)
    scale_deg = scale_arcsec / 3600.0
    provisional = {
        "NAXIS": 2, "NAXIS1": 1, "NAXIS2": 1,
        "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
        "CUNIT1": "deg", "CUNIT2": "deg",
        "CRVAL1": ra0, "CRVAL2": dec0,
        "CRPIX1": 0.5, "CRPIX2": 0.5,
        "CD1_1": -scale_deg, "CD1_2": 0.0,
        "CD2_1": 0.0, "CD2_2": scale_deg,
    }
    wcs = WCS(provisional)
    x, y = wcs.world_to_pixel(world[:, 0], world[:, 1])
    finite = torch.isfinite(x) & torch.isfinite(y)
    x, y = x[finite], y[finite]
    x0 = math.floor(float(x.min())) - padding_pixels
    x1 = math.ceil(float(x.max())) + padding_pixels
    y0 = math.floor(float(y.min())) - padding_pixels
    y1 = math.ceil(float(y.max())) + padding_pixels
    header = dict(provisional)
    header["NAXIS1"] = int(x1 - x0 + 1)
    header["NAXIS2"] = int(y1 - y0 + 1)
    header["CRPIX1"] = provisional["CRPIX1"] - x0
    header["CRPIX2"] = provisional["CRPIX2"] - y0
    return UnionCanvas(header, (header["NAXIS2"], header["NAXIS1"]), scale_arcsec)


class WCSGeometryBank:
    """Fixed per-exposure WCS geometry evaluated inside model.forward."""

    def __init__(self, headers: Sequence[dict[str, Any]], canvas: UnionCanvas) -> None:
        self.headers = tuple(dict(h) for h in headers)
        self.canvas = canvas
        self._cache: dict[str, tuple[tuple[WCS, ...], WCS]] = {}

    def _on(self, device: torch.device) -> tuple[tuple[WCS, ...], WCS]:
        key = str(device)
        if key not in self._cache:
            self._cache[key] = (
                tuple(WCS(h).to(device) for h in self.headers),
                WCS(self.canvas.header).to(device),
            )
        return self._cache[key]

    def native_to_union(self, uv: Tensor, exposure_id: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        source, target = self._on(uv.device)
        uv64 = uv.double()
        xy = torch.empty_like(uv64)
        sky = torch.empty_like(uv64)
        for raw_id in torch.unique(exposure_id).detach().cpu().tolist():
            e = int(raw_id)
            choose = exposure_id == e
            ra, dec = source[e].pixel_to_world(uv64[choose, 0], uv64[choose, 1])
            x, y = target.world_to_pixel(ra, dec)
            sky[choose] = torch.stack((ra, dec), -1)
            xy[choose] = torch.stack((x, y), -1)
        valid = torch.isfinite(xy).all(-1) & torch.isfinite(sky).all(-1)
        return xy, sky, valid


class FourierPlaneEncoder(nn.Module):
    """Planar Fourier features separated into sin/cos pairs for exact phase shifts."""

    def __init__(self, canvas: UnionCanvas, num_frequencies: int = 10) -> None:
        super().__init__()
        self.width = max(canvas.shape[1] - 1, 1)
        self.height = max(canvas.shape[0] - 1, 1)
        frequencies = (2.0 ** torch.arange(num_frequencies)) * math.pi
        self.register_buffer("frequencies", frequencies.float())
        self.output_dim = 4 * num_frequencies + 2

    def normalized_xy(self, xy: Tensor) -> Tensor:
        x = 2.0 * xy[:, 0] / self.width - 1.0
        y = 2.0 * xy[:, 1] / self.height - 1.0
        return torch.stack((x, y), -1)

    def forward(self, xy: Tensor) -> Tensor:
        p = self.normalized_xy(xy).float()
        phase = p[..., None] * self.frequencies
        sx, cx = torch.sin(phase[:, 0]), torch.cos(phase[:, 0])
        sy, cy = torch.sin(phase[:, 1]), torch.cos(phase[:, 1])
        return torch.cat((p, sx, cx, sy, cy), -1)


class FourierFeatureShiftAlignment(nn.Module):
    """Per-exposure trainable translation applied between h and z.

    Translation is implemented as the exact Fourier phase rotation, so the
    alignment remains physically interpretable in union-canvas pixels.
    """

    def __init__(self, encoder: FourierPlaneEncoder, n_exposures: int, anchor: int = 0) -> None:
        super().__init__()
        self.encoder = encoder
        self.shift_pixels = nn.Parameter(torch.zeros(n_exposures, 2))
        mask = torch.ones(n_exposures, 1)
        mask[anchor] = 0
        self.register_buffer("mask", mask)

    def effective_shift_pixels(self) -> Tensor:
        return self.shift_pixels * self.mask

    def forward(self, features: Tensor, exposure_id: Tensor) -> Tensor:
        nfreq = self.encoder.frequencies.numel()
        shift = self.effective_shift_pixels()[exposure_id]
        dx = 2.0 * shift[:, 0] / self.encoder.width
        dy = 2.0 * shift[:, 1] / self.encoder.height
        fx = self.encoder.frequencies[None, :]
        phx, phy = dx[:, None] * fx, dy[:, None] * fx
        start = 2
        sx = features[:, start:start+nfreq]
        cx = features[:, start+nfreq:start+2*nfreq]
        sy = features[:, start+2*nfreq:start+3*nfreq]
        cy = features[:, start+3*nfreq:start+4*nfreq]
        sx2 = sx * torch.cos(phx) + cx * torch.sin(phx)
        cx2 = cx * torch.cos(phx) - sx * torch.sin(phx)
        sy2 = sy * torch.cos(phy) + cy * torch.sin(phy)
        cy2 = cy * torch.cos(phy) - sy * torch.sin(phy)
        p = features[:, :2] + torch.stack((dx, dy), -1)
        return torch.cat((p, sx2, cx2, sy2, cy2), -1)

    def regularization(self) -> Tensor:
        return self.effective_shift_pixels().square().mean()


class SphericalPositionEncoder(nn.Module):
    def __init__(self, num_frequencies: int = 10, min_frequency: float = 1.0) -> None:
        super().__init__()
        self.num_frequencies = num_frequencies
        self.min_frequency = min_frequency
        with torch.no_grad():
            probe = self.forward(torch.zeros(1, 2, dtype=torch.float64))
        self.output_dim = int(probe.shape[-1])

    def forward(self, ra_dec: Tensor) -> Tensor:
        ra, dec = ra_dec[:, 0], ra_dec[:, 1]
        harmonic = harmonic_sky_encoding(ra, dec, include_degree2=True)
        fourier = spherical_fourier_features(
            ra, dec,
            num_frequencies=self.num_frequencies,
            min_frequency=self.min_frequency,
            include_xyz=True,
            include_lonlat=True,
        )
        return torch.cat((harmonic, fourier), -1).float()


class ResidualFeatureAlignment(nn.Module):
    """Exposure-specific low-rank trainable h->z adapter for spherical PE."""

    def __init__(self, n_exposures: int, dim: int, rank: int = 8, anchor: int = 0) -> None:
        super().__init__()
        self.rank = rank
        self.down = nn.Parameter(torch.zeros(n_exposures, dim, rank))
        self.up = nn.Parameter(torch.zeros(n_exposures, rank, dim))
        self.bias = nn.Parameter(torch.zeros(n_exposures, dim))
        nn.init.normal_(self.down, std=1e-3)
        nn.init.normal_(self.up, std=1e-3)
        mask = torch.ones(n_exposures, 1)
        mask[anchor] = 0
        self.register_buffer("mask", mask)

    def forward(self, h: Tensor, exposure_id: Tensor) -> Tensor:
        down = self.down[exposure_id]
        up = self.up[exposure_id]
        delta = torch.bmm(torch.bmm(h[:, None, :], down), up).squeeze(1)
        delta = (delta + self.bias[exposure_id]) * self.mask[exposure_id]
        return h + delta

    def regularization(self) -> Tensor:
        return (self.down.square().mean() + self.up.square().mean() + self.bias.square().mean())


def _mlp(in_dim: int, out_dim: int, hidden: int, depth: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.SiLU()]
        d = hidden
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class WCSAwareMultiWavelengthINR(nn.Module):
    def __init__(
        self,
        headers: Sequence[dict[str, Any]],
        canvas: UnionCanvas,
        *,
        encoding: EncodingKind = "fourier",
        spatial_frequencies: int = 10,
        spherical_min_frequency: float = 1.0,
        spherical_alignment_rank: int = 8,
        spectral_frequencies: int = 6,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        trunk_depth: int = 4,
        decoder_depth: int = 3,
        anchor_exposure: int = 0,
        learn_background: bool = True,
        output_activation: str = "softplus",
    ) -> None:
        super().__init__()
        self.geometry = WCSGeometryBank(headers, canvas)
        self.canvas = canvas
        self.encoding = encoding
        self.spectral_frequencies = spectral_frequencies
        self.output_activation = output_activation
        n_exp = len(headers)
        if encoding == "fourier":
            self.position_encoder = FourierPlaneEncoder(canvas, spatial_frequencies)
            self.feature_alignment = FourierFeatureShiftAlignment(
                self.position_encoder, n_exp, anchor_exposure
            )
        elif encoding == "spherical":
            self.position_encoder = SphericalPositionEncoder(
                spatial_frequencies, spherical_min_frequency
            )
            self.feature_alignment = ResidualFeatureAlignment(
                n_exp, self.position_encoder.output_dim,
                spherical_alignment_rank, anchor_exposure
            )
        else:
            raise ValueError("encoding must be fourier or spherical")
        self.trunk = _mlp(self.position_encoder.output_dim, latent_dim, hidden_dim, trunk_depth)
        spectral_dim = 1 + 2 * spectral_frequencies
        self.decoder = _mlp(latent_dim + spectral_dim, 1, hidden_dim, decoder_depth)
        self.background = nn.Parameter(torch.zeros(n_exp), requires_grad=learn_background)

    def _spectral_features(self, wavelength: Tensor) -> Tensor:
        logw = torch.log(wavelength.clamp_min(1e-12))
        scales = 2.0 ** torch.arange(
            self.spectral_frequencies, device=logw.device, dtype=logw.dtype
        )
        phase = logw[:, None] * scales[None, :]
        return torch.cat((logw[:, None], torch.sin(phase), torch.cos(phase)), -1).float()

    def spatial_latent(self, uv: Tensor, exposure_id: Tensor) -> tuple[Tensor, Tensor]:
        xy, sky, valid = self.geometry.native_to_union(uv, exposure_id)
        raw = self.position_encoder(xy) if self.encoding == "fourier" else self.position_encoder(sky)
        aligned = self.feature_alignment(raw, exposure_id)
        return self.trunk(aligned), valid

    def decode(self, latent: Tensor, wavelength: Tensor, valid: Tensor) -> Tensor:
        value = self.decoder(torch.cat((latent, self._spectral_features(wavelength)), -1)).squeeze(-1)
        if self.output_activation == "softplus":
            value = torch.nn.functional.softplus(value)
        return torch.where(valid, value, torch.zeros_like(value))

    def scene(self, uv: Tensor, exposure_id: Tensor, wavelength: Tensor) -> tuple[Tensor, Tensor]:
        latent, valid = self.spatial_latent(uv, exposure_id)
        return self.decode(latent, wavelength, valid), valid

    def predict_observation(
        self,
        uv: Tensor,
        exposure_id: Tensor,
        bandpass: BandpassTable,
        *,
        psf: PSFKernelBank | None = None,
    ) -> tuple[Tensor, Tensor]:
        wavelength = bandpass.wavelength[exposure_id]
        weight = bandpass.weight[exposure_id]
        weight = weight / weight.sum(-1, keepdim=True).clamp_min(1e-20)
        pred = torch.zeros(uv.shape[0], device=uv.device, dtype=uv.dtype)
        valid_any = torch.zeros(uv.shape[0], device=uv.device, dtype=torch.bool)
        if psf is None:
            latent, valid = self.spatial_latent(uv, exposure_id)
            valid_any = valid
            for j in range(wavelength.shape[1]):
                pred += weight[:, j] * self.decode(latent, wavelength[:, j], valid)
        else:
            k = psf.kernel.shape[-1]
            r = k // 2
            axis = torch.arange(-r, r + 1, device=uv.device, dtype=uv.dtype)
            dy, dx = torch.meshgrid(axis, axis, indexing="ij")
            offsets = torch.stack((dx.flatten(), dy.flatten()), -1)
            q = uv[:, None, :] + offsets[None, :, :]
            qe = exposure_id[:, None].expand(-1, offsets.shape[0])
            latent, valid = self.spatial_latent(q.reshape(-1, 2), qe.reshape(-1))
            latent = latent.reshape(uv.shape[0], offsets.shape[0], -1)
            valid = valid.reshape(uv.shape[0], offsets.shape[0])
            valid_any = valid.any(-1)
            kernels = psf.kernel[exposure_id]
            for j in range(wavelength.shape[1]):
                wave = wavelength[:, j:j+1].expand(-1, offsets.shape[0])
                scene = self.decode(latent.reshape(-1, latent.shape[-1]), wave.reshape(-1), valid.reshape(-1))
                scene = scene.reshape(uv.shape[0], offsets.shape[0])
                pred += weight[:, j] * (scene * kernels[:, j].reshape(uv.shape[0], -1)).sum(-1)
        return pred + self.background[exposure_id], valid_any

    def alignment_regularization(self) -> Tensor:
        return self.feature_alignment.regularization()

    def alignment_summary(self) -> dict[str, Any]:
        if isinstance(self.feature_alignment, FourierFeatureShiftAlignment):
            return {
                "kind": "fourier_phase_shift",
                "shift_pixels_xy": self.feature_alignment.effective_shift_pixels().detach().cpu().tolist(),
            }
        return {"kind": "spherical_low_rank_feature_adapter", "rank": self.feature_alignment.rank}


def weighted_robust_loss(
    prediction: Tensor,
    target: Tensor,
    err: Tensor,
    mask: Tensor,
    *,
    kind: str = "huber",
    delta: float = 0.1,
    max_relative_weight: float = 100.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    valid = mask.bool() & torch.isfinite(prediction) & torch.isfinite(target) & torch.isfinite(err) & (err > 0)
    if not bool(valid.any()):
        raise RuntimeError("batch has no valid pixels")
    residual = prediction[valid] - target[valid]
    if kind == "huber":
        a = residual.abs()
        rho = torch.where(a <= delta, 0.5 * residual.square(), delta * (a - 0.5 * delta))
    elif kind == "charbonnier":
        rho = torch.sqrt(residual.square() + delta * delta) - delta
    else:
        raise ValueError("kind must be huber or charbonnier")
    weight = err[valid].reciprocal().square()
    weight = (weight / weight.mean().clamp_min(1e-20)).clamp_max(max_relative_weight)
    loss = (weight * rho).sum() / weight.sum().clamp_min(1e-20)
    return loss, {
        "rmse": residual.square().mean().sqrt().detach(),
        "mae": residual.abs().mean().detach(),
        "valid_fraction": valid.float().mean().detach(),
    }


def gaussian_psf_kernel(fwhm_pixels: float, size: int = 7) -> Tensor:
    if size % 2 != 1:
        raise ValueError("PSF size must be odd")
    sigma = fwhm_pixels / 2.354820045
    axis = torch.arange(-(size // 2), size // 2 + 1, dtype=torch.float32)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    kernel = torch.exp(-(xx.square() + yy.square()) / (2 * sigma * sigma))
    return kernel / kernel.sum()


__all__ = [
    "BandpassTable", "PSFKernelBank", "UnionCanvas",
    "WCSAwareMultiWavelengthINR", "build_union_canvas",
    "gaussian_psf_kernel", "weighted_robust_loss",
]
