"""Neural field-model classes for :mod:`torchsky.map.field`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from typing import Sequence
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ._core_helpers import (
    _BaseFieldModel,
    _fourier_features_vectorized,
    _fourier_features_with_curriculum,
    _point_bandpass_features,
    _reset_xavier_mlp,
    _sample_feature_volume,
)
from .types import (
    FieldDomain,
)
from ...device import require_float64
from ...wcs import WCS


if TYPE_CHECKING:
    from ..product import ProductResult

class ExplicitSpectralGridField(_BaseFieldModel):
    """Explicit multiframe spatial and spectral grid baseline."""

    def __init__(
        self,
        domain: FieldDomain,
        *,
        spatial_resolution: int = 48,
        spectral_resolution: int = 8,
        spatial_bounds_local_deg: Sequence[float] | None = None,
    ) -> None:
        super().__init__(domain)
        if self.coord_mode != "local_tangent":
            raise ValueError("ExplicitSpectralGridField requires local_tangent coordinates")
        self.spatial_resolution = int(spatial_resolution)
        self.spectral_resolution = int(spectral_resolution)
        if self.spatial_resolution < 2 or self.spectral_resolution < 2:
            raise ValueError("grid resolutions must be at least 2")
        self.spatial_bounds_local_deg: tuple[float, float, float, float] | None = None
        if spatial_bounds_local_deg is not None:
            if len(spatial_bounds_local_deg) != 4:
                raise ValueError("spatial_bounds_local_deg must contain four values")
            bounds = tuple(float(value) for value in spatial_bounds_local_deg)
            if not all(math.isfinite(value) for value in bounds):
                raise ValueError("spatial_bounds_local_deg must be finite")
            if bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
                raise ValueError("spatial_bounds_local_deg bounds must be increasing")
            self.spatial_bounds_local_deg = bounds
        self.grid = nn.Parameter(
            1.0e-3
            * torch.randn(
                1,
                self.spectral_resolution,
                self.spatial_resolution,
                self.spatial_resolution,
            )
        )

    def _grid_world_coords(self) -> tuple[Tensor, Tensor, Tensor]:
        """Return world coordinates for the configured grid nodes.

        Legacy spatial and all spectral endpoints use the nearest representable
        value inside (-1, 1), since the exact inverse tanh there is infinite.
        """
        device = self.grid.device
        # Absolute celestial coordinates need more precision than brightness
        # parameters: round only after subtracting the field centre.
        dtype = torch.float32 if device.type == "mps" else torch.float64
        bound = torch.nextafter(
            torch.ones((), device=device, dtype=dtype),
            torch.zeros((), device=device, dtype=dtype),
        )
        if self.spatial_bounds_local_deg is None:
            xs = torch.linspace(
                -1.0, 1.0, steps=self.spatial_resolution, device=device, dtype=dtype
            ).clamp(-bound, bound).atanh()
            ys = torch.linspace(
                -1.0, 1.0, steps=self.spatial_resolution, device=device, dtype=dtype
            ).clamp(-bound, bound).atanh()
            spatial_scale = float(self.domain.spatial_scale_deg)
            xs = xs * spatial_scale
            ys = ys * spatial_scale
        else:
            x_min, x_max, y_min, y_max = self.spatial_bounds_local_deg
            xs = torch.linspace(
                x_min, x_max, steps=self.spatial_resolution, device=device, dtype=dtype
            )
            ys = torch.linspace(
                y_min, y_max, steps=self.spatial_resolution, device=device, dtype=dtype
            )
        ls = torch.linspace(
            -1.0, 1.0, steps=self.spectral_resolution, device=device, dtype=dtype
        ).clamp(-bound, bound).atanh()
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        wavelengths = float(self.domain.wavelength_center) + ls * float(
            self.domain.wavelength_scale
        )
        cos_dec = math.cos(math.radians(float(self.domain.center_dec_deg)))
        dra = xx / max(cos_dec, 1.0e-6)
        dec = yy + float(self.domain.center_dec_deg)
        ra = dra + float(self.domain.center_ra_deg)
        return ra, dec, wavelengths

    def initialize_from_product(self, product: ProductResult) -> None:
        if self.coord_mode != "local_tangent":
            raise ValueError(
                "ExplicitSpectralGridField initialization requires local_tangent"
            )
        target_wcs = WCS(product.header)
        ra_grid, dec_grid, wavelengths = self._grid_world_coords()
        px, py = target_wcs.world_to_pixel(ra_grid, dec_grid)
        target_width = max(int(product.header["NAXIS1"]) - 1, 1)
        target_height = max(int(product.header["NAXIS2"]) - 1, 1)
        x_norm = 2.0 * (px / target_width) - 1.0
        y_norm = 2.0 * (py / target_height) - 1.0
        sample_grid = (
            torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0).to(dtype=self.grid.dtype)
        )
        band_centers = []
        for band in product.product.bands:
            if band.transmission_center is not None:
                band_centers.append(float(band.transmission_center))
            elif band.spectral_min is not None and band.spectral_max is not None:
                band_centers.append(
                    0.5 * (float(band.spectral_min) + float(band.spectral_max))
                )
            else:
                raise ValueError(
                    "ProductSpec band requires center or bounds for initialization"
                )
        centers = torch.tensor(
            band_centers, device=self.grid.device, dtype=self.grid.dtype
        )
        data = product.data.to(device=self.grid.device, dtype=self.grid.dtype)
        sampled_planes = []
        for plane in data:
            sampled = (
                F.grid_sample(
                    plane.unsqueeze(0).unsqueeze(0),
                    sample_grid,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True,
                )
                .squeeze(0)
                .squeeze(0)
            )
            sampled_planes.append(sampled)
        sampled_cube = torch.stack(sampled_planes, dim=0)
        order = torch.argsort(centers)
        centers = centers[order]
        sampled_cube = sampled_cube[order]
        interp_planes = []
        for wavelength in wavelengths:
            if wavelength <= centers[0]:
                interp_planes.append(sampled_cube[0])
                continue
            if wavelength >= centers[-1]:
                interp_planes.append(sampled_cube[-1])
                continue
            upper_idx = int(torch.searchsorted(centers, wavelength).item())
            lower_idx = max(upper_idx - 1, 0)
            lam0 = centers[lower_idx]
            lam1 = centers[upper_idx]
            alpha = (wavelength - lam0) / torch.clamp(lam1 - lam0, min=1.0e-6)
            interp_planes.append(
                (1.0 - alpha) * sampled_cube[lower_idx]
                + alpha * sampled_cube[upper_idx]
            )
        init_cube = torch.stack(interp_planes, dim=0).unsqueeze(0)
        with torch.no_grad():
            init_cube = torch.clamp(init_cube, min=1.0e-4)
            self.grid.copy_(init_cube + torch.log(-torch.expm1(-init_cube)))

    def _coords_unit(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        if self.coord_mode != "local_tangent":
            raise ValueError(
                "ExplicitSpectralGridField requires local_tangent coordinates"
            )
        if self.spatial_bounds_local_deg is None:
            coords = self._unit_box_coords(ra_deg, dec_deg, wavelength_angstrom)
        else:
            encoded = self._encoded_coords(ra_deg, dec_deg, wavelength_angstrom)
            scale = float(self.domain.spatial_scale_deg)
            x_min, x_max, y_min, y_max = self.spatial_bounds_local_deg
            x_unit = (encoded[..., 0] * scale - x_min) / (x_max - x_min)
            y_unit = (encoded[..., 1] * scale - y_min) / (y_max - y_min)
            wavelength_unit = 0.5 * (torch.tanh(encoded[..., 2]) + 1.0)
            coords = torch.stack((x_unit, y_unit, wavelength_unit), dim=-1)
        return coords.to(dtype=self.grid.dtype)

    def forward(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        coords_unit = self._coords_unit(ra_deg, dec_deg, wavelength_angstrom)
        sampled = _sample_feature_volume(self.grid, coords_unit)
        return F.softplus(sampled) + 1.0e-4

    def render_bandpass(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelengths: Tensor,
        weights: Tensor,
    ) -> Tensor:
        norm_weights = weights.to(dtype=self.grid.dtype)
        norm_weights = norm_weights / torch.clamp(norm_weights.sum(), min=1.0e-8)
        rendered = [
            self.forward(
                ra_deg,
                dec_deg,
                torch.as_tensor(
                    float(wavelength.item()),
                    device=ra_deg.device,
                    dtype=self.grid.dtype,
                ),
            )
            for wavelength in wavelengths
        ]
        return torch.sum(
            torch.stack(rendered, dim=0) * norm_weights.view(-1, *([1] * ra_deg.ndim)),
            dim=0,
        )


class SineImplicitField(_BaseFieldModel):
    def __init__(
        self,
        domain: FieldDomain,
        *,
        hidden_dim: int = 48,
        hidden_layers: int = 2,
        omega0: float = 6.0,
    ) -> None:
        super().__init__(domain)
        self.omega0 = float(omega0)
        layers: list[nn.Linear] = [nn.Linear(self.coord_dim, hidden_dim)]
        for _ in range(max(int(hidden_layers) - 1, 0)):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.hidden = nn.ModuleList(layers)
        self.output = nn.Linear(hidden_dim, 1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        first = True
        for layer in self.hidden:
            fan_in = layer.weight.shape[1]
            if first:
                bound = 1.0 / max(fan_in, 1)
                first = False
            else:
                bound = math.sqrt(6.0 / fan_in) / max(self.omega0, 1.0)
            nn.init.uniform_(layer.weight, -bound, bound)
            nn.init.uniform_(layer.bias, -bound, bound)
        nn.init.uniform_(self.output.weight, -1e-2, 1e-2)
        nn.init.constant_(self.output.bias, -3.0)

    def forward(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        x = self._encoded_coords(ra_deg, dec_deg, wavelength_angstrom).to(
            dtype=self.hidden[0].weight.dtype
        )
        for layer in self.hidden:
            x = torch.sin(self.omega0 * layer(x))
        out = self.output(x).squeeze(-1)
        return F.softplus(out) + 1.0e-4


class _GaborLayer(nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int, *, omega0: float, sigma0: float
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.omega0 = float(omega0)
        self.sigma0 = float(sigma0)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        fan_in = self.linear.weight.shape[1]
        bound = math.sqrt(6.0 / max(fan_in, 1)) / max(self.omega0, 1.0)
        nn.init.uniform_(self.linear.weight, -bound, bound)
        nn.init.uniform_(self.linear.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        z = self.linear(x)
        return torch.cos(self.omega0 * z) * torch.exp(-torch.square(self.sigma0 * z))


class WireImplicitField(_BaseFieldModel):
    def __init__(
        self,
        domain: FieldDomain,
        *,
        hidden_dim: int = 48,
        hidden_layers: int = 2,
        omega0: float = 8.0,
        sigma0: float = 2.5,
    ) -> None:
        super().__init__(domain)
        layers: list[_GaborLayer] = [
            _GaborLayer(self.coord_dim, hidden_dim, omega0=omega0, sigma0=sigma0)
        ]
        for _ in range(max(int(hidden_layers) - 1, 0)):
            layers.append(
                _GaborLayer(hidden_dim, hidden_dim, omega0=omega0, sigma0=sigma0)
            )
        self.hidden = nn.ModuleList(layers)
        self.output = nn.Linear(hidden_dim, 1)
        nn.init.uniform_(self.output.weight, -1e-2, 1e-2)
        nn.init.constant_(self.output.bias, -3.0)

    def forward(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        x = self._encoded_coords(ra_deg, dec_deg, wavelength_angstrom).to(
            dtype=self.output.weight.dtype
        )
        for layer in self.hidden:
            x = layer(x)
        out = self.output(x).squeeze(-1)
        return F.softplus(out) + 1.0e-4


class GaussianBasisField(_BaseFieldModel):
    """Gaussian field; opt-in ``splat`` uses additive elliptical sky sources.

    Splat amplitudes integrate brightness over normalized tangent-coordinate
    area (not physical catalog flux; that requires solid-angle conversion),
    with positive linear spectral interpolation and constant extrapolation.
    PSF convolution belongs to the observation operator, not this latent field.
    The default retains the original three-dimensional basis/checkpoint layout.
    """

    def __init__(
        self,
        domain: FieldDomain,
        *,
        num_basis: int = 24,
        init_scale: float = 0.35,
        splat: bool = False,
        spectral_resolution: int = 8,
        query_chunk_size: int = 2048,
    ) -> None:
        super().__init__(domain)
        self.splat = bool(splat)
        if int(num_basis) < 1 or not math.isfinite(init_scale) or init_scale <= 0:
            raise ValueError("num_basis and init_scale must be positive")
        if self.splat:
            if domain.coord_mode != "local_tangent":
                raise ValueError("Gaussian splats require local_tangent coordinates")
            if spectral_resolution < 2 or query_chunk_size < 1:
                raise ValueError("spectral_resolution >= 2 and query_chunk_size >= 1 required")
            self.query_chunk_size = int(query_chunk_size)
            self.centers = nn.Parameter(torch.empty(int(num_basis), 2))
            self.log_scales = nn.Parameter(torch.full((int(num_basis), 2), math.log(init_scale)))
            self.shear = nn.Parameter(torch.zeros(int(num_basis)))
            self.amplitudes = nn.Parameter(torch.full((int(num_basis), int(spectral_resolution)), -5.0))
            self.register_buffer("spectral_knots", torch.linspace(-1.0, 1.0, int(spectral_resolution)))
            nn.init.uniform_(self.centers, -1.0, 1.0)
            return
        self.centers = nn.Parameter(torch.empty(int(num_basis), 3))
        self.log_scales = nn.Parameter(
            torch.full((int(num_basis), 3), math.log(init_scale))
        )
        self.amplitudes = nn.Parameter(torch.empty(int(num_basis)))
        self.bias = nn.Parameter(torch.zeros(()))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.uniform_(self.centers, -1.0, 1.0)
        nn.init.uniform_(self.amplitudes, -0.05, 0.05)

    def _splat_values(self, coords: Tensor) -> Tensor:
        scales = self.log_scales.exp()
        diff = coords[:, None, :2] - self.centers
        x = diff[..., 0] / scales[:, 0]
        y = (diff[..., 1] - self.shear * diff[..., 0]) / scales[:, 1]
        basis = torch.exp(-0.5 * (x.square() + y.square())) / (
            2.0 * math.pi * scales.prod(dim=-1)
        )
        lam = coords[:, 2].clamp(self.spectral_knots[0], self.spectral_knots[-1])
        upper = torch.searchsorted(self.spectral_knots, lam.contiguous()).clamp(1, self.spectral_knots.numel() - 1)
        alpha = (lam - self.spectral_knots[upper - 1]) / (self.spectral_knots[upper] - self.spectral_knots[upper - 1])
        # Sum sources before interpolation to avoid repeated spectral-index
        # scatter-adds over every source during the backward pass.
        spectrum = basis @ F.softplus(self.amplitudes)
        lower_flux = spectrum.gather(1, (upper - 1)[:, None]).squeeze(1)
        upper_flux = spectrum.gather(1, upper[:, None]).squeeze(1)
        return lower_flux * (1 - alpha) + upper_flux * alpha

    @torch.no_grad()
    def initialize_from_product(self, product: ProductResult) -> None:
        if not self.splat:
            raise ValueError("product initialization requires splat=True")
        device, dtype = self.centers.device, self.centers.dtype
        require_float64(device, what="GaussianBasisField.initialize_from_product")
        data = product.data.to(device=device, dtype=dtype)
        if data.ndim != 3 or not torch.isfinite(data).all():
            raise ValueError("splat initialization requires a finite band/image cube")
        bands = []
        for band in product.product.bands:
            if band.transmission_center is not None:
                bands.append(float(band.transmission_center))
            elif band.spectral_min is not None and band.spectral_max is not None:
                bands.append(0.5 * (float(band.spectral_min) + float(band.spectral_max)))
            else:
                raise ValueError("initialization bands require centers or bounds")
        wavelengths = torch.tensor(bands, device=device, dtype=dtype)
        if len(bands) != data.shape[0] or not bands or not torch.isfinite(wavelengths).all():
            raise ValueError("initialization requires one finite wavelength per image")
        wavelengths, order = wavelengths.sort()
        if wavelengths.numel() > 1 and torch.any(wavelengths.diff() <= 0):
            raise ValueError("initialization wavelengths must be distinct")
        count = self.centers.shape[0]
        height, width = data.shape[-2:]
        columns = min(count, max(1, int(math.ceil(math.sqrt(count * width / height)))))
        rows = int(math.ceil(count / columns))
        indices = torch.arange(count, device=device, dtype=dtype)
        px = (indices.remainder(columns) + 0.5) * width / columns - 0.5
        py = (indices.div(columns, rounding_mode="floor") + 0.5) * height / rows - 0.5
        wcs = WCS(product.header)
        ra, dec = wcs.pixel_to_world(px.to(torch.float64), py.to(torch.float64))
        coords = self._encoded_coords(ra, dec, self.domain.wavelength_center).to(dtype)
        self.centers.copy_(coords[:, :2])
        # ponytail: a source lattice seeds the fit, not a deconvolution; the shared
        # detector operator and subsequent optimization recover intrinsic widths.
        ra_x, dec_x = wcs.pixel_to_world((px + width / columns).double(), py.double())
        ra_y, dec_y = wcs.pixel_to_world(px.double(), (py + height / rows).double())
        dx = self._encoded_coords(ra_x, dec_x, self.domain.wavelength_center).to(dtype)[:, :2] - coords[:, :2]
        dy = self._encoded_coords(ra_y, dec_y, self.domain.wavelength_center).to(dtype)[:, :2] - coords[:, :2]
        area = (dx[:, 0] * dy[:, 1] - dx[:, 1] * dy[:, 0]).abs().clamp_min(torch.finfo(dtype).tiny)
        spacing = torch.stack((dx.norm(dim=-1), dy.norm(dim=-1)), dim=-1)
        self.log_scales.copy_((0.5 * spacing).clamp_min(torch.finfo(dtype).eps).log())
        self.shear.zero_()
        grid = torch.stack((2 * (px + 0.5) / width - 1, 2 * (py + 0.5) / height - 1), dim=-1).view(1, 1, count, 2)
        samples = F.grid_sample(data[order].unsqueeze(0), grid, align_corners=False, padding_mode="border")[0, :, 0].T.clamp_min(0)
        knots = torch.linspace(wavelengths[0], wavelengths[-1], self.spectral_knots.numel(), device=device, dtype=dtype)
        if len(bands) == 1:
            sampled = samples.expand(-1, knots.numel())
        else:
            upper = torch.searchsorted(wavelengths, knots).clamp(1, len(bands) - 1)
            alpha = (knots - wavelengths[upper - 1]) / (wavelengths[upper] - wavelengths[upper - 1])
            sampled = samples[:, upper - 1] * (1 - alpha) + samples[:, upper] * alpha
            self.spectral_knots.copy_((knots - self.domain.wavelength_center) / self.domain.wavelength_scale)
        flux = (sampled * area[:, None]).clamp_min(torch.finfo(dtype).eps)
        self.amplitudes.copy_(flux + torch.log(-torch.expm1(-flux)))

    def forward(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        coords = self._encoded_coords(ra_deg, dec_deg, wavelength_angstrom).to(
            dtype=self.centers.dtype
        )
        if self.splat:
            flat = coords.reshape(-1, 3)
            if flat.shape[0] == 0:
                return coords[..., 0]
            # Checkpoint each query chunk to bound saved basis activations in training.
            from torch.utils.checkpoint import checkpoint

            outputs = [
                checkpoint(self._splat_values, chunk, use_reentrant=False)
                if torch.is_grad_enabled() else self._splat_values(chunk)
                for chunk in flat.split(self.query_chunk_size)
            ]
            return torch.cat(outputs).reshape(coords.shape[:-1])
        diff = coords.unsqueeze(-2) - self.centers
        scales = F.softplus(self.log_scales) + 1.0e-3
        # ⚡ Bolt: Element-wise multiplication is faster than torch.square or **2
        scaled_diff = diff / scales
        radial2 = torch.sum(scaled_diff * scaled_diff, dim=-1)
        basis = torch.exp(-0.5 * radial2)
        out = torch.sum(basis * self.amplitudes, dim=-1) + self.bias
        return F.softplus(out) + 1.0e-4


class FourierMLPField(_BaseFieldModel):
    def __init__(
        self,
        domain: FieldDomain,
        *,
        hidden_dim: int = 64,
        hidden_layers: int = 3,
        num_frequencies: int = 6,
        max_frequency: float = 16.0,
    ) -> None:
        super().__init__(domain)
        self.num_frequencies = int(num_frequencies)
        self.max_frequency = float(max_frequency)
        in_dim = self.coord_dim * (1 + 2 * self.num_frequencies)
        layers: list[nn.Module] = []
        current_dim = in_dim
        for _ in range(max(int(hidden_layers), 1)):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.SiLU())
            current_dim = hidden_dim
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(current_dim, 1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        _reset_xavier_mlp(self.hidden, self.output)

    def _fourier_features(self, coords: Tensor) -> Tensor:
        if self.num_frequencies <= 0:
            return coords
        # base ** linspace == logspace; aten::logspace is not implemented on MPS.
        scales = torch.pow(
            2.0,
            torch.linspace(
                0.0,
                math.log2(max(self.max_frequency, 1.0)),
                steps=self.num_frequencies,
                device=coords.device,
                dtype=coords.dtype,
            ),
        )
        # Vectorized phase computation: (..., num_frequencies, coords_dim)
        phase = math.pi * coords.unsqueeze(-2) * scales.unsqueeze(-1)

        sin_feats = torch.sin(phase)
        cos_feats = torch.cos(phase)

        # Interleave sin and cos features by stacking on a new dimension
        # and then flattening the last 3 dimensions (num_frequencies, 2, coords_dim)
        vect_feats = torch.stack([sin_feats, cos_feats], dim=-2).view(
            coords.shape[:-1] + (-1,)
        )

        return torch.cat([coords, vect_feats], dim=-1)

    def forward(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        coords = self._encoded_coords(ra_deg, dec_deg, wavelength_angstrom).to(
            dtype=self.output.weight.dtype
        )
        feats = self._fourier_features(coords)
        out = self.output(self.hidden(feats)).squeeze(-1)
        return F.softplus(out) + 1.0e-4


class SpectralBasisField(_BaseFieldModel):
    def __init__(
        self,
        domain: FieldDomain,
        *,
        hidden_dim: int = 96,
        hidden_layers: int = 2,
        num_spatial_frequencies: int = 6,
        num_spectral_frequencies: int = 4,
    ) -> None:
        super().__init__(domain)
        self.num_spatial_frequencies = int(num_spatial_frequencies)
        self.num_spectral_frequencies = int(num_spectral_frequencies)
        self.spatial_dim = 2 if self.coord_mode == "local_tangent" else 3
        self.basis_dim = 3 + 2 * max(self.num_spectral_frequencies, 0)
        self._curriculum_progress = 1.0
        in_dim = self.spatial_dim * (1 + 2 * max(self.num_spatial_frequencies, 0))
        layers: list[nn.Module] = []
        current_dim = in_dim
        for _ in range(max(int(hidden_layers), 1)):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.SiLU())
            current_dim = hidden_dim
        self.hidden = nn.Sequential(*layers)
        self.coeff_head = nn.Linear(current_dim, self.basis_dim)
        self.bias_head = nn.Linear(current_dim, 1)
        self.spectral_amp_head = nn.Sequential(
            nn.Linear(1, 16), nn.SiLU(), nn.Linear(16, 1)
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.hidden:
            if type(module) is nn.Linear:
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.coeff_head.weight, gain=0.01)
        nn.init.zeros_(self.coeff_head.bias)
        nn.init.uniform_(self.bias_head.weight, -1e-4, 1e-4)
        nn.init.constant_(self.bias_head.bias, -3.0)
        for module in self.spectral_amp_head:
            if type(module) is nn.Linear:
                nn.init.uniform_(module.weight, -1e-3, 1e-3)
                nn.init.zeros_(module.bias)

    def set_curriculum_progress(self, progress: float) -> None:
        self._curriculum_progress = min(max(float(progress), 0.0), 1.0)

    def _active_frequency_count(self, total: int) -> int:
        if total <= 0:
            return 0
        return max(1, int(math.ceil(self._curriculum_progress * total)))

    def _spatial_fourier_features(self, spatial: Tensor) -> Tensor:
        return _fourier_features_with_curriculum(
            spatial,
            self.num_spatial_frequencies,
            self._active_frequency_count(self.num_spatial_frequencies),
        )

    def _spectral_basis(self, lam: Tensor) -> Tensor:
        if self.num_spectral_frequencies > 0:
            scales = torch.arange(
                1,
                self.num_spectral_frequencies + 1,
                device=lam.device,
                dtype=lam.dtype,
            )
            active = self._active_frequency_count(self.num_spectral_frequencies)
            scaled = lam.unsqueeze(-1) * scales
            phase = math.pi * scaled

            sin_phase = torch.sin(phase)
            cos_phase = torch.cos(phase)

            # Mask out inactive frequencies
            mask = (
                torch.arange(self.num_spectral_frequencies, device=lam.device) < active
            )
            sin_phase = sin_phase * mask
            cos_phase = cos_phase * mask

            stacked_fourier = torch.stack([sin_phase, cos_phase], dim=-1).view(
                *lam.shape, -1
            )
            base_feats = torch.stack([torch.ones_like(lam), lam, lam * lam], dim=-1)
            return torch.cat([base_feats, stacked_fourier], dim=-1)
        else:
            return torch.stack([torch.ones_like(lam), lam, lam * lam], dim=-1)

    def forward(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        require_float64(ra_deg.device, what="SpectralBasisField.forward")
        spatial = self._spatial_coords(
            ra_deg.to(dtype=torch.float64),
            dec_deg.to(dtype=torch.float64),
            wavelength_angstrom,
        )
        spatial_feats = self._spatial_fourier_features(spatial).to(
            dtype=self.coeff_head.weight.dtype
        )
        hidden = self.hidden(spatial_feats)
        coeff = self.coeff_head(hidden)
        bias = self.bias_head(hidden).squeeze(-1)
        lam = self._normalized_wavelength(
            wavelength_angstrom,
            device=ra_deg.device,
            dtype=torch.float64,
            shape=ra_deg.shape,
        )
        spectral_basis = self._spectral_basis(lam).to(
            dtype=self.coeff_head.weight.dtype
        )
        out = bias + torch.sum(coeff * spectral_basis, dim=-1)

        # Apply per-band amplitude control
        lam_flat = lam.view(-1, 1).to(dtype=self.coeff_head.weight.dtype)
        log_amp = self.spectral_amp_head(lam_flat).view(*out.shape)

        return torch.exp(log_amp) * F.softplus(out) + 1.0e-4

    def render_bandpass(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelengths: Tensor,
        weights: Tensor,
    ) -> Tensor:
        require_float64(ra_deg.device, what="SpectralBasisField.render_bandpass")
        if (
            wavelengths.ndim != 1
            or weights.ndim != 1
            or wavelengths.shape != weights.shape
        ):
            raise ValueError(
                "wavelengths and weights must be 1D tensors with matching shape"
            )
        if wavelengths.numel() == 0:
            raise ValueError("bandpass render requires at least one wavelength sample")
        if wavelengths.numel() == 1:
            return self.forward(ra_deg, dec_deg, wavelengths.reshape(-1)[0])
        spatial = self._spatial_coords(
            ra_deg.to(dtype=torch.float64),
            dec_deg.to(dtype=torch.float64),
            wavelengths.reshape(-1)[0],
        )
        spatial_feats = self._spatial_fourier_features(spatial).to(
            dtype=self.coeff_head.weight.dtype
        )
        hidden = self.hidden(spatial_feats)
        coeff = self.coeff_head(hidden)
        bias = self.bias_head(hidden).squeeze(-1)

        lam = self._normalized_wavelength(
            wavelengths,
            device=coeff.device,
            dtype=torch.float64,
            shape=wavelengths.shape,
        )
        spectral_basis = self._spectral_basis(lam).to(dtype=coeff.dtype)
        out = bias.unsqueeze(0) + torch.sum(
            coeff.unsqueeze(0)
            * spectral_basis.view(
                spectral_basis.shape[0],
                *([1] * bias.ndim),
                spectral_basis.shape[-1],
            ),
            dim=-1,
        )
        log_amp = self.spectral_amp_head(
            lam.view(-1, 1).to(dtype=self.coeff_head.weight.dtype)
        ).view(-1, *([1] * bias.ndim))
        per_wavelength = torch.exp(log_amp) * F.softplus(out) + 1.0e-4
        norm_weights = weights.to(device=coeff.device, dtype=per_wavelength.dtype)
        norm_weights = norm_weights / torch.clamp(norm_weights.sum(), min=1.0e-8)
        return torch.sum(
            per_wavelength * norm_weights.view(-1, *([1] * bias.ndim)),
            dim=0,
        )


class BandpassConditionedField(_BaseFieldModel):
    def __init__(
        self,
        domain: FieldDomain,
        *,
        hidden_dim: int = 96,
        hidden_layers: int = 2,
        num_spatial_frequencies: int = 6,
        num_bandpass_frequencies: int = 6,
        bandpass_hidden_dim: int = 48,
    ) -> None:
        super().__init__(domain)
        self.num_spatial_frequencies = int(num_spatial_frequencies)
        self.num_bandpass_frequencies = int(num_bandpass_frequencies)
        self.spatial_dim = 2 if self.coord_mode == "local_tangent" else 3
        spatial_in_dim = self.spatial_dim * (
            1 + 2 * max(self.num_spatial_frequencies, 0)
        )
        spatial_layers: list[nn.Module] = []
        current_dim = spatial_in_dim
        for _ in range(max(int(hidden_layers), 1)):
            spatial_layers.append(nn.Linear(current_dim, hidden_dim))
            spatial_layers.append(nn.SiLU())
            current_dim = hidden_dim
        self.spatial_hidden = nn.Sequential(*spatial_layers)
        bandpass_in_dim = 4 + 2 * max(self.num_bandpass_frequencies, 0)
        self.bandpass_hidden = nn.Sequential(
            nn.Linear(bandpass_in_dim, bandpass_hidden_dim),
            nn.SiLU(),
            nn.Linear(bandpass_hidden_dim, bandpass_hidden_dim),
            nn.SiLU(),
        )
        self.output = nn.Linear(hidden_dim + bandpass_hidden_dim, 1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.spatial_hidden:
            if type(module) is nn.Linear:
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for module in self.bandpass_hidden:
            if type(module) is nn.Linear:
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.uniform_(self.output.weight, -1e-2, 1e-2)
        nn.init.constant_(self.output.bias, -3.0)

    def _spatial_fourier_features(self, spatial: Tensor) -> Tensor:
        return _fourier_features_vectorized(
            spatial, self.num_spatial_frequencies
        )

    def _bandpass_features(self, wavelengths: Tensor, weights: Tensor) -> Tensor:
        lam = (
            wavelengths.to(dtype=self.output.weight.dtype)
            - float(self.domain.wavelength_center)
        ) / float(self.domain.wavelength_scale)
        weights = weights.to(device=lam.device, dtype=self.output.weight.dtype)
        weights = weights / torch.clamp(weights.sum(), min=1.0e-8)
        mean = torch.sum(weights * lam)
        centered = lam - mean
        # Element-wise multiplication is faster than torch.square or **2
        centered_sq = centered * centered
        second = torch.sum(weights * centered_sq)
        third = torch.sum(weights * centered_sq * centered)
        fourth = torch.sum(weights * centered_sq * centered_sq)
        feats = [mean, second, third, fourth]
        if self.num_bandpass_frequencies > 0:
            scales = torch.arange(
                1,
                self.num_bandpass_frequencies + 1,
                device=lam.device,
                dtype=lam.dtype,
            )
            phase = math.pi * scales.view(-1, *([1] * lam.ndim)) * lam
            sin_sum = torch.sum(
                weights * torch.sin(phase), dim=tuple(range(1, 1 + lam.ndim))
            )
            cos_sum = torch.sum(
                weights * torch.cos(phase), dim=tuple(range(1, 1 + lam.ndim))
            )
            freq_feats = torch.stack([sin_sum, cos_sum], dim=1).flatten()
            return torch.cat([torch.stack(feats), freq_feats])
        return torch.stack(feats)

    def _spatial_latent(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        spatial = self._spatial_coords(ra_deg, dec_deg, wavelength_angstrom).to(
            dtype=self.output.weight.dtype
        )
        return self.spatial_hidden(self._spatial_fourier_features(spatial))

    def render_bandpass(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelengths: Tensor,
        weights: Tensor,
    ) -> Tensor:
        representative_wavelength = float(wavelengths.detach().mean())
        spatial_latent = self._spatial_latent(
            ra_deg,
            dec_deg,
            representative_wavelength,
        )
        bandpass_latent = self.bandpass_hidden(
            self._bandpass_features(wavelengths, weights)
        )
        bandpass_latent = bandpass_latent.view(*([1] * (spatial_latent.ndim - 1)), -1)
        bandpass_latent = bandpass_latent.expand(*spatial_latent.shape[:-1], -1)
        out = self.output(torch.cat([spatial_latent, bandpass_latent], dim=-1)).squeeze(
            -1
        )
        return F.softplus(out) + 1.0e-4

    def forward(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        spatial_latent = self._spatial_latent(
            ra_deg,
            dec_deg,
            wavelength_angstrom,
        )
        lam = self._normalized_wavelength(
            wavelength_angstrom,
            device=ra_deg.device,
            dtype=self.output.weight.dtype,
            shape=ra_deg.shape,
        )
        bandpass_latent = self.bandpass_hidden(
            _point_bandpass_features(lam, self.num_bandpass_frequencies)
        )
        out = self.output(torch.cat([spatial_latent, bandpass_latent], dim=-1)).squeeze(
            -1
        )
        return F.softplus(out) + 1.0e-4


class LearnedSpectralBasisField(_BaseFieldModel):
    def __init__(
        self,
        domain: FieldDomain,
        *,
        hidden_dim: int = 96,
        hidden_layers: int = 2,
        num_spatial_frequencies: int = 6,
        basis_dim: int = 16,
        spectral_hidden_dim: int = 32,
    ) -> None:
        super().__init__(domain)
        self.num_spatial_frequencies = int(num_spatial_frequencies)
        self.spatial_dim = 2 if self.coord_mode == "local_tangent" else 3
        self.basis_dim = int(basis_dim)
        self._curriculum_progress = 1.0
        in_dim = self.spatial_dim * (1 + 2 * max(self.num_spatial_frequencies, 0))
        layers: list[nn.Module] = []
        current_dim = in_dim
        for _ in range(max(int(hidden_layers), 1)):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.SiLU())
            current_dim = hidden_dim
        self.hidden = nn.Sequential(*layers)
        self.coeff_head = nn.Linear(current_dim, self.basis_dim)
        self.bias_head = nn.Linear(current_dim, 1)
        self.spectral_basis = nn.Sequential(
            nn.Linear(3, spectral_hidden_dim),
            nn.SiLU(),
            nn.Linear(spectral_hidden_dim, self.basis_dim),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.hidden:
            if type(module) is nn.Linear:
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.coeff_head.weight)
        nn.init.zeros_(self.coeff_head.bias)
        nn.init.uniform_(self.bias_head.weight, -1e-2, 1e-2)
        nn.init.constant_(self.bias_head.bias, 0.0)
        for module in self.spectral_basis:
            if type(module) is nn.Linear:
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def set_curriculum_progress(self, progress: float) -> None:
        self._curriculum_progress = min(max(float(progress), 0.0), 1.0)

    def _active_frequency_count(self, total: int) -> int:
        if total <= 0:
            return 0
        return max(1, int(math.ceil(self._curriculum_progress * total)))

    def _spatial_fourier_features(self, spatial: Tensor) -> Tensor:
        return _fourier_features_with_curriculum(
            spatial,
            self.num_spatial_frequencies,
            self._active_frequency_count(self.num_spatial_frequencies),
        )

    def _spectral_features(self, lam: Tensor) -> Tensor:
        return torch.stack([lam, lam * lam, torch.tanh(lam)], dim=-1)

    def forward(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        require_float64(ra_deg.device, what="LearnedSpectralBasisField.forward")
        spatial = self._spatial_coords(
            ra_deg.to(dtype=torch.float64),
            dec_deg.to(dtype=torch.float64),
            wavelength_angstrom,
        )
        spatial_feats = self._spatial_fourier_features(spatial).to(
            dtype=self.coeff_head.weight.dtype
        )
        hidden = self.hidden(spatial_feats)
        coeff = self.coeff_head(hidden)
        bias = self.bias_head(hidden).squeeze(-1)
        lam = self._normalized_wavelength(
            wavelength_angstrom,
            device=ra_deg.device,
            dtype=torch.float64,
            shape=ra_deg.shape,
        )
        basis = self.spectral_basis(
            self._spectral_features(lam).to(dtype=self.coeff_head.weight.dtype)
        )
        out = bias + torch.sum(coeff * basis, dim=-1)
        return F.softplus(out) + 1.0e-4
