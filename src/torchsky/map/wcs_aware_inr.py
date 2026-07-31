"""WCS-aware multi-wavelength INR for native JWST exposure pixels.

The scene is trained directly against native detector pixels. Each forward pass
uses the fixed exposure WCS, maps the pixel into a union TAN canvas, encodes the
position (planar Fourier or spherical), optionally applies trainable residual
alignment, and decodes one shared wavelength-conditioned scene.  Deterministic
TweakReg alignment is represented by corrected frozen exposure WCS headers.

Bandpass quadrature is intentionally delegated to TorchSky's native
``FieldObservation``/``bandpass_quadrature`` implementation. The observation
operator optionally accepts a wavelength-dependent PSF bank with shape
``[exposure, spectral_sample, kernel_y, kernel_x]`` and applies each PSF *inside*
the wavelength loop before the weighted bandpass sum.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any, Literal, Sequence

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from ..encoders import harmonic_sky_encoding
from ..sphere.healpix import spherical_fourier_features
from ..wcs import WCS
from .field.bandpass import bandpass_quadrature

if TYPE_CHECKING:
    from .field.types import FieldObservation

EncodingKind = Literal["fourier", "spherical"]
AlignmentMode = Literal["original_wcs", "tweakreg_frozen", "trainable"]


@dataclass(frozen=True)
class UnionCanvas:
    """Rectangular TAN canvas covering the union of exposure footprints."""

    header: dict[str, Any]
    shape: tuple[int, int]
    pixel_scale_arcsec: float


@dataclass(frozen=True)
class PSFKernelBank:
    """Per-exposure, per-quadrature-sample detector-plane PSF kernels.

    Parameters
    ----------
    kernel
        Tensor with shape ``[E, L, K, K]``. Kernels must be finite,
        non-negative, square, odd-sized, and have positive total mass.
    wavelength
        Optional quadrature wavelengths with shape ``[E, L]``. Supplying these
        is strongly recommended because it lets the renderer verify that the
        PSFs were generated at exactly the same wavelength samples as the
        native TorchSky bandpass quadrature.
    """

    kernel: Tensor
    wavelength: Tensor | None = None

    def __post_init__(self) -> None:
        if self.kernel.ndim != 4 or self.kernel.shape[-1] != self.kernel.shape[-2]:
            raise ValueError("PSF kernels must have shape [E,L,K,K]")
        if self.kernel.shape[-1] % 2 != 1:
            raise ValueError("PSF kernel size must be odd")
        if not bool(torch.isfinite(self.kernel).all()):
            raise ValueError("PSF kernels must be finite")
        if bool((self.kernel < 0).any()):
            raise ValueError("PSF kernels must be non-negative")
        mass = self.kernel.sum(dim=(-1, -2))
        if bool((mass <= 0).any()):
            raise ValueError("every PSF kernel must have positive total mass")
        if self.wavelength is not None:
            expected = self.kernel.shape[:2]
            if self.wavelength.ndim != 2 or tuple(self.wavelength.shape) != tuple(expected):
                raise ValueError("PSF wavelengths must have shape [E,L]")
            if not bool(torch.isfinite(self.wavelength).all()):
                raise ValueError("PSF wavelengths must be finite")

    @property
    def num_exposures(self) -> int:
        return int(self.kernel.shape[0])

    @property
    def num_samples(self) -> int:
        return int(self.kernel.shape[1])

    @property
    def size(self) -> int:
        return int(self.kernel.shape[-1])

    def normalized(self) -> "PSFKernelBank":
        denom = self.kernel.sum(dim=(-1, -2), keepdim=True).clamp_min(1e-20)
        return PSFKernelBank(self.kernel / denom, self.wavelength)

    def to(self, device: torch.device | str) -> "PSFKernelBank":
        return PSFKernelBank(
            self.kernel.to(device=device),
            None if self.wavelength is None else self.wavelength.to(device=device),
        )

    def validate_against(
        self,
        wavelengths: Tensor,
        *,
        rtol: float = 1e-5,
        atol: float = 1e-3,
    ) -> None:
        if wavelengths.ndim != 2:
            raise ValueError("bandpass wavelengths must have shape [E,L]")
        if tuple(wavelengths.shape) != tuple(self.kernel.shape[:2]):
            raise ValueError(
                "PSF bank [E,L] dimensions do not match bandpass quadrature: "
                f"{tuple(self.kernel.shape[:2])} != {tuple(wavelengths.shape)}"
            )
        if self.wavelength is not None and not torch.allclose(
            self.wavelength.to(device=wavelengths.device, dtype=wavelengths.dtype),
            wavelengths,
            rtol=rtol,
            atol=atol,
        ):
            max_error = float(
                (
                    self.wavelength.to(device=wavelengths.device, dtype=wavelengths.dtype)
                    - wavelengths
                )
                .abs()
                .max()
                .detach()
                .cpu()
            )
            raise ValueError(
                "PSF-bank wavelengths do not match native bandpass quadrature; "
                f"maximum absolute mismatch is {max_error:g}"
            )


def native_bandpass_quadrature_bank(
    observations: Sequence["FieldObservation"],
    *,
    num_spectral_samples: int,
    device: torch.device | str = "cpu",
) -> tuple[Tensor, Tensor]:
    """Build ``[E,L]`` wavelengths and weights using TorchSky's native code.

    This is the only bandpass construction path used by the WCS-aware INR. It
    therefore inherits the native handling of tabulated throughput curves,
    linear interpolation, trapezoidal quadrature, normalization, and the
    uniform/Gaussian fallbacks encoded by ``FieldObservation``.
    """

    if not observations:
        raise ValueError("at least one FieldObservation is required")
    target_device = torch.device(device)
    wavelength_rows: list[Tensor] = []
    weight_rows: list[Tensor] = []
    expected_samples: int | None = None
    for observation in observations:
        wavelength, weight = bandpass_quadrature(
            observation,
            num_spectral_samples=int(num_spectral_samples),
            device=target_device,
        )
        if expected_samples is None:
            expected_samples = int(wavelength.numel())
        elif int(wavelength.numel()) != expected_samples:
            raise ValueError(
                "all observations must yield the same number of quadrature samples"
            )
        wavelength_rows.append(wavelength)
        weight_rows.append(weight)
    wavelengths = torch.stack(wavelength_rows, dim=0)
    weights = torch.stack(weight_rows, dim=0)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    return wavelengths, weights


def _pixel_scale_arcsec(wcs: WCS) -> float:
    det = float(torch.det(wcs.cd.detach().cpu().double()).abs())
    if not math.isfinite(det) or det <= 0:
        raise ValueError("singular spatial WCS")
    return math.sqrt(det) * 3600.0


def _circular_mean_deg(ra: Tensor) -> float:
    angle = torch.deg2rad(ra.double())
    value = torch.rad2deg(
        torch.atan2(torch.sin(angle).mean(), torch.cos(angle).mean())
    ) % 360
    return float(value)


def build_union_canvas(
    headers: Sequence[dict[str, Any]],
    *,
    pixel_scale_arcsec: float | None = None,
    padding_pixels: int = 8,
) -> UnionCanvas:
    """Plan a rectangular TAN canvas covering the union of exposure footprints.

    No science image is resampled or coadded here.
    """

    if not headers:
        raise ValueError("at least one exposure header is required")
    if padding_pixels < 0:
        raise ValueError("padding_pixels must be non-negative")
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
    if not math.isfinite(scale_arcsec) or scale_arcsec <= 0:
        raise ValueError("pixel_scale_arcsec must be finite and positive")
    scale_deg = scale_arcsec / 3600.0
    provisional = {
        "NAXIS": 2,
        "NAXIS1": 1,
        "NAXIS2": 1,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CUNIT1": "deg",
        "CUNIT2": "deg",
        "CRVAL1": ra0,
        "CRVAL2": dec0,
        "CRPIX1": 0.5,
        "CRPIX2": 0.5,
        "CD1_1": -scale_deg,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": scale_deg,
    }
    wcs = WCS(provisional)
    x, y = wcs.world_to_pixel(world[:, 0], world[:, 1])
    finite = torch.isfinite(x) & torch.isfinite(y)
    if not bool(finite.any()):
        raise ValueError("union projection produced no finite pixels")
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
    return UnionCanvas(
        header=header,
        shape=(header["NAXIS2"], header["NAXIS1"]),
        pixel_scale_arcsec=scale_arcsec,
    )


class WCSGeometryBank:
    """Fixed per-exposure WCS geometry evaluated inside ``model.forward``."""

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

    def native_to_union(
        self,
        uv: Tensor,
        exposure_id: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if uv.ndim != 2 or uv.shape[1] != 2:
            raise ValueError(f"uv must have shape [N,2], got {tuple(uv.shape)}")
        if exposure_id.ndim != 1:
            raise ValueError(
                "exposure_id must have shape [N], "
                f"got {tuple(exposure_id.shape)}"
            )
        if uv.shape[0] != exposure_id.shape[0]:
            raise ValueError(
                "uv and exposure_id must contain the same number of rows: "
                f"{uv.shape[0]} != {exposure_id.shape[0]}"
            )

        source, target = self._on(uv.device)
        uv64 = uv.double()
        xy = torch.empty_like(uv64)
        sky = torch.empty_like(uv64)
        for raw_id in torch.unique(exposure_id).detach().cpu().tolist():
            e = int(raw_id)
            if e < 0 or e >= len(source):
                raise IndexError(f"exposure_id {e} is out of range")
            choose = exposure_id == e
            ra, dec = source[e].pixel_to_world(uv64[choose, 0], uv64[choose, 1])
            x, y = target.world_to_pixel(ra, dec)
            sky[choose] = torch.stack((ra, dec), -1)
            xy[choose] = torch.stack((x, y), -1)
        valid = torch.isfinite(xy).all(-1) & torch.isfinite(sky).all(-1)
        valid = (
            valid
            & (xy[:, 0] >= 0)
            & (xy[:, 0] <= self.canvas.shape[1] - 1)
            & (xy[:, 1] >= 0)
            & (xy[:, 1] <= self.canvas.shape[0] - 1)
        )
        return xy, sky, valid


class FourierPlaneEncoder(nn.Module):
    """Planar Fourier features with explicit sine/cosine pairs."""

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
    """Per-exposure trainable translation applied between ``h`` and ``z``."""

    def __init__(
        self,
        encoder: FourierPlaneEncoder,
        n_exposures: int,
        anchor: int = 0,
    ) -> None:
        super().__init__()
        if anchor < 0 or anchor >= n_exposures:
            raise ValueError("anchor exposure is out of range")
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
        freq = self.encoder.frequencies[None, :]
        phx, phy = dx[:, None] * freq, dy[:, None] * freq
        start = 2
        sx = features[:, start : start + nfreq]
        cx = features[:, start + nfreq : start + 2 * nfreq]
        sy = features[:, start + 2 * nfreq : start + 3 * nfreq]
        cy = features[:, start + 3 * nfreq : start + 4 * nfreq]
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
            ra,
            dec,
            num_frequencies=self.num_frequencies,
            min_frequency=self.min_frequency,
            include_xyz=True,
            include_lonlat=True,
        )
        return torch.cat((harmonic, fourier), -1).float()


class ResidualFeatureAlignment(nn.Module):
    """Exposure-specific low-rank trainable ``h -> z`` adapter for spherical PE."""

    def __init__(
        self,
        n_exposures: int,
        dim: int,
        rank: int = 8,
        anchor: int = 0,
    ) -> None:
        super().__init__()
        if anchor < 0 or anchor >= n_exposures:
            raise ValueError("anchor exposure is out of range")
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
        return (
            self.down.square().mean()
            + self.up.square().mean()
            + self.bias.square().mean()
        )


class IdentityFeatureAlignment(nn.Module):
    """No residual feature alignment; geometry is entirely fixed by the WCS."""

    def __init__(self, mode: str) -> None:
        super().__init__()
        self.mode = mode
        self.register_buffer("zero", torch.zeros(()))

    def forward(self, features: Tensor, exposure_id: Tensor) -> Tensor:
        del exposure_id
        return features

    def regularization(self) -> Tensor:
        return self.zero


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
        alignment_mode: AlignmentMode = "trainable",
        learn_background: bool = True,
        output_activation: str = "softplus",
    ) -> None:
        super().__init__()
        self.geometry = WCSGeometryBank(headers, canvas)
        self.canvas = canvas
        self.encoding = encoding
        self.spectral_frequencies = spectral_frequencies
        self.output_activation = output_activation
        self.alignment_mode = alignment_mode
        self.num_exposures = len(headers)
        if alignment_mode not in {"original_wcs", "tweakreg_frozen", "trainable"}:
            raise ValueError(
                "alignment_mode must be original_wcs, tweakreg_frozen, or trainable"
            )
        if encoding == "fourier":
            self.position_encoder = FourierPlaneEncoder(canvas, spatial_frequencies)
            self.feature_alignment = (
                FourierFeatureShiftAlignment(
                    self.position_encoder,
                    self.num_exposures,
                    anchor_exposure,
                )
                if alignment_mode == "trainable"
                else IdentityFeatureAlignment(alignment_mode)
            )
        elif encoding == "spherical":
            self.position_encoder = SphericalPositionEncoder(
                spatial_frequencies,
                spherical_min_frequency,
            )
            self.feature_alignment = (
                ResidualFeatureAlignment(
                    self.num_exposures,
                    self.position_encoder.output_dim,
                    spherical_alignment_rank,
                    anchor_exposure,
                )
                if alignment_mode == "trainable"
                else IdentityFeatureAlignment(alignment_mode)
            )
        else:
            raise ValueError("encoding must be fourier or spherical")
        self.trunk = _mlp(
            self.position_encoder.output_dim,
            latent_dim,
            hidden_dim,
            trunk_depth,
        )
        spectral_dim = 1 + 2 * spectral_frequencies
        self.decoder = _mlp(
            latent_dim + spectral_dim,
            1,
            hidden_dim,
            decoder_depth,
        )
        self.background = nn.Parameter(
            torch.zeros(self.num_exposures),
            requires_grad=learn_background,
        )

    def _spectral_features(self, wavelength: Tensor) -> Tensor:
        logw = torch.log(wavelength.clamp_min(1e-12))
        scales = 2.0 ** torch.arange(
            self.spectral_frequencies,
            device=logw.device,
            dtype=logw.dtype,
        )
        phase = logw[:, None] * scales[None, :]
        return torch.cat(
            (logw[:, None], torch.sin(phase), torch.cos(phase)),
            -1,
        ).float()

    def spatial_latent(
        self,
        uv: Tensor,
        exposure_id: Tensor,
    ) -> tuple[Tensor, Tensor]:
        xy, sky, valid = self.geometry.native_to_union(uv, exposure_id)
        raw = (
            self.position_encoder(xy)
            if self.encoding == "fourier"
            else self.position_encoder(sky)
        )
        aligned = self.feature_alignment(raw, exposure_id)
        return self.trunk(aligned), valid

    def decode(self, latent: Tensor, wavelength: Tensor, valid: Tensor) -> Tensor:
        value = self.decoder(
            torch.cat((latent, self._spectral_features(wavelength)), -1)
        ).squeeze(-1)
        if self.output_activation == "softplus":
            value = torch.nn.functional.softplus(value)
        elif self.output_activation != "linear":
            raise ValueError("output_activation must be softplus or linear")
        return torch.where(valid, value, torch.zeros_like(value))

    def scene(
        self,
        uv: Tensor,
        exposure_id: Tensor,
        wavelength: Tensor,
    ) -> tuple[Tensor, Tensor]:
        latent, valid = self.spatial_latent(uv, exposure_id)
        return self.decode(latent, wavelength, valid), valid

    @staticmethod
    def _validate_bandpass_tensors(
        wavelengths: Tensor,
        weights: Tensor,
        *,
        num_exposures: int,
    ) -> tuple[Tensor, Tensor]:
        if wavelengths.ndim != 2 or weights.ndim != 2:
            raise ValueError("bandpass tensors must have shape [E,L]")
        if wavelengths.shape != weights.shape:
            raise ValueError("bandpass wavelengths and weights must have identical shapes")
        if int(wavelengths.shape[0]) != int(num_exposures):
            raise ValueError("bandpass exposure count does not match the model")
        if not bool(torch.isfinite(wavelengths).all()) or bool((wavelengths <= 0).any()):
            raise ValueError("bandpass wavelengths must be finite and positive")
        if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
            raise ValueError("bandpass weights must be finite and non-negative")
        if bool((weights.sum(dim=-1) <= 0).any()):
            raise ValueError("every exposure must have positive total bandpass weight")
        return wavelengths, weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)

    def predict_observation(
        self,
        uv: Tensor,
        exposure_id: Tensor,
        bandpass_wavelength: Tensor,
        bandpass_weight: Tensor,
        *,
        psf: PSFKernelBank | None = None,
        psf_offset_chunk: int = 256,
        checkpoint_psf_chunks: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Render native pixels through bandpass and optional wavelength PSFs.

        For a PSF bank, the computation is exactly

        ``sum_j w[e,j] * (K[e,j] * S(theta, lambda[e,j]))``.

        The kernel is therefore applied before the spectral sum. The optional
        checkpointing trades extra compute for lower activation memory when the
        kernel support is large.
        """

        wavelengths, weights = self._validate_bandpass_tensors(
            bandpass_wavelength,
            bandpass_weight,
            num_exposures=self.num_exposures,
        )
        wavelengths = wavelengths.to(device=uv.device)[exposure_id]
        weights = weights.to(device=uv.device)[exposure_id]
        prediction = torch.zeros(uv.shape[0], device=uv.device, dtype=uv.dtype)

        if psf is None:
            latent, center_valid = self.spatial_latent(uv, exposure_id)
            for j in range(wavelengths.shape[1]):
                prediction = prediction + weights[:, j] * self.decode(
                    latent,
                    wavelengths[:, j],
                    center_valid,
                )
        else:
            psf.validate_against(
                bandpass_wavelength.to(device=psf.kernel.device, dtype=torch.float32)
            )
            if psf.num_exposures != self.num_exposures:
                raise ValueError("PSF exposure count does not match the model")
            _, _, center_valid = self.geometry.native_to_union(uv, exposure_id)
            radius = psf.size // 2
            axis = torch.arange(
                -radius,
                radius + 1,
                device=uv.device,
                dtype=uv.dtype,
            )
            dy, dx = torch.meshgrid(axis, axis, indexing="ij")
            offsets = torch.stack((dx.flatten(), dy.flatten()), -1)
            total_offsets = int(offsets.shape[0])
            chunk_size = max(1, int(psf_offset_chunk))
            convolved = torch.zeros(
                uv.shape[0],
                wavelengths.shape[1],
                device=uv.device,
                dtype=uv.dtype,
            )

            for start in range(0, total_offsets, chunk_size):
                stop = min(start + chunk_size, total_offsets)
                offset_chunk = offsets[start:stop]
                count = int(offset_chunk.shape[0])
                query_uv = uv[:, None, :] + offset_chunk[None, :, :]
                query_uv_flat = query_uv.reshape(-1, 2)

                # Activation checkpointing recomputes this function during
                # backward, after the outer loop has advanced. Bind every
                # ragged-chunk value explicitly so recomputation cannot combine
                # an earlier query with the final short chunk's metadata.
                def evaluate_chunk(
                    query: Tensor,
                    base_exposure_id: Tensor,
                    chunk_count: int = count,
                    chunk_start: int = start,
                    chunk_stop: int = stop,
                ) -> Tensor:
                    query_exposure = (
                        base_exposure_id[:, None]
                        .expand(-1, chunk_count)
                        .reshape(-1)
                    )
                    latent, valid = self.spatial_latent(query, query_exposure)
                    contributions: list[Tensor] = []
                    for sample_index in range(wavelengths.shape[1]):
                        wave = wavelengths[:, sample_index : sample_index + 1].expand(
                            -1, chunk_count
                        )
                        scene = self.decode(
                            latent,
                            wave.reshape(-1),
                            valid,
                        ).reshape(uv.shape[0], chunk_count)
                        kernel = psf.kernel[
                            base_exposure_id,
                            sample_index,
                        ].reshape(uv.shape[0], total_offsets)[
                            :, chunk_start:chunk_stop
                        ]
                        contributions.append(
                            (
                                scene
                                * kernel.to(
                                    device=scene.device,
                                    dtype=scene.dtype,
                                )
                            ).sum(-1)
                        )
                    return torch.stack(contributions, dim=-1)

                if checkpoint_psf_chunks and self.training and torch.is_grad_enabled():
                    chunk_value = checkpoint(
                        evaluate_chunk,
                        query_uv_flat,
                        exposure_id,
                        use_reentrant=False,
                    )
                else:
                    chunk_value = evaluate_chunk(query_uv_flat, exposure_id)
                convolved = convolved + chunk_value
            prediction = (weights * convolved).sum(dim=-1)

        return prediction + self.background[exposure_id], center_valid

    def alignment_regularization(self) -> Tensor:
        return self.feature_alignment.regularization()

    def alignment_summary(self) -> dict[str, Any]:
        if isinstance(self.feature_alignment, IdentityFeatureAlignment):
            return {"kind": self.alignment_mode, "trainable": False}
        if isinstance(self.feature_alignment, FourierFeatureShiftAlignment):
            return {
                "kind": "fourier_phase_shift",
                "trainable": True,
                "shift_pixels_xy": self.feature_alignment.effective_shift_pixels()
                .detach()
                .cpu()
                .tolist(),
            }
        return {
            "kind": "spherical_low_rank_feature_adapter",
            "trainable": True,
            "rank": self.feature_alignment.rank,
        }


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
    valid = (
        mask.bool()
        & torch.isfinite(prediction)
        & torch.isfinite(target)
        & torch.isfinite(err)
        & (err > 0)
    )
    if not bool(valid.any()):
        raise RuntimeError("batch has no valid pixels")
    residual = prediction[valid] - target[valid]
    if kind == "whitened_l2":
        whitened = residual / err[valid]
        loss = 0.5 * whitened.square().mean()
        return loss, {
            "rmse": residual.square().mean().sqrt().detach(),
            "mae": residual.abs().mean().detach(),
            "whitened_rmse": whitened.square().mean().sqrt().detach(),
            "whitened_mae": whitened.abs().mean().detach(),
            "valid_fraction": valid.float().mean().detach(),
        }
    if kind == "huber":
        absolute = residual.abs()
        rho = torch.where(
            absolute <= delta,
            0.5 * residual.square(),
            delta * (absolute - 0.5 * delta),
        )
    elif kind == "charbonnier":
        rho = torch.sqrt(residual.square() + delta * delta) - delta
    else:
        raise ValueError("kind must be huber, charbonnier, or whitened_l2")
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
    if not math.isfinite(fwhm_pixels) or fwhm_pixels <= 0:
        raise ValueError("PSF FWHM must be finite and positive")
    sigma = fwhm_pixels / 2.354820045
    axis = torch.arange(-(size // 2), size // 2 + 1, dtype=torch.float32)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    kernel = torch.exp(-(xx.square() + yy.square()) / (2 * sigma * sigma))
    return kernel / kernel.sum()


__all__ = [
    "PSFKernelBank",
    "UnionCanvas",
    "WCSAwareMultiWavelengthINR",
    "build_union_canvas",
    "gaussian_psf_kernel",
    "native_bandpass_quadrature_bank",
    "weighted_robust_loss",
]
