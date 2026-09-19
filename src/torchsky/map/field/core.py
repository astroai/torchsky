"""Experimental continuous field models for multiwavelength map fitting."""

from __future__ import annotations

import copy
import math
import time
from typing import TYPE_CHECKING, Any, Literal, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ...device import default_compute_device
from ...wcs import WCS

if TYPE_CHECKING:
    from ..product import MapProductResult

from ..operators.bundle import field_observation_operator_bundle
from ..operators.psf import apply_psf_spec, gaussian_blur as _shared_gaussian_blur
from ..solvers import ComponentDecompositionDiagnostics, SurveyInstrumentComponentMixer

from .bandpass import (
    bandpass_bounds as _bandpass_bounds,
    bandpass_mc_samples as _bandpass_mc_samples,
    bandpass_quadrature as _bandpass_quadrature,
    bandpass_response as _bandpass_response,
    validate_override_quadrature as _validate_override_quadrature,
)
from .forwarding import call_scalar_field_forward as _call_scalar_field_forward
from .foam import (
    FoamAdaptationResult,
    PowerPixelOperator,
    PowerValueSemantics,
    SpectralPowerFoam,
    adapt_spectral_power_foam,
    sparse_bounded_power_pixel_operator,
)
from .hashgrid import HashGridEncoder, sample_feature_plane
from .losses import _field_residual_loss
from .networks import (
    ResidualMLPBlock,
    ResidualStack,
    reset_logflux_readout_linear,
    reset_silu_mlp_linears,
    silu_mlp_sequential,
)
from .noise import _synthetic_noise_target
from .observation_ops import (
    coordinate_grid as _coordinate_grid,
    observation_pixel_to_world as _observation_pixel_to_world,
    observation_support_mask as _observation_support_mask,
    wrapped_delta_ra_deg as _wrapped_delta_ra_deg,
)
from .rendering import (
    render_field_observation,
    render_field_samples,
)
from .sky_encoding import make_sky_coordinate_encoder, sky_encoder_output_dim
from .types import (
    FieldComponentAnchorTerm,
    FieldCentroidAnchorTerm,
    FieldCoordinateMode,
    FieldDomain,
    FieldLossKind,
    FieldModelKind,
    FieldNoisePretrainKind,
    FieldObservation,
    FieldSpectralSampling,
    ForwardContext,
)


def _gaussian_kernel1d(sigma_pix: float, *, device: torch.device) -> Tensor:
    radius = max(1, int(math.ceil(3.0 * float(sigma_pix))))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    kernel = torch.exp(-0.5 * torch.square(x / float(sigma_pix)))
    return kernel / torch.clamp(kernel.sum(), min=1e-8)


def _gaussian_blur(image: Tensor, sigma_pix: float) -> Tensor:
    return _shared_gaussian_blur(image, sigma_pix)


def _smoothness_probe_grid(
    domain: FieldDomain,
    *,
    grid_size: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    if grid_size < 2:
        raise ValueError("spatial_smoothness_grid_size must be at least 2")
    offsets = torch.linspace(
        -1.0, 1.0, steps=int(grid_size), device=device, dtype=torch.float32
    )
    off_y, off_x = torch.meshgrid(offsets, offsets, indexing="ij")
    cos_dec = max(math.cos(math.radians(float(domain.center_dec_deg))), 1.0e-6)
    dra = off_x * float(domain.spatial_scale_deg) / cos_dec
    ddec = off_y * float(domain.spatial_scale_deg)
    ra = torch.remainder(float(domain.center_ra_deg) + dra + 360.0, 360.0)
    dec = torch.clamp(float(domain.center_dec_deg) + ddec, min=-90.0, max=90.0)
    return ra, dec


def _observation_footprint_world_deg(
    observation: FieldObservation,
) -> tuple[list[float], list[float]]:
    """Return detector-edge sky samples, falling back to the reference position."""
    nx = int(observation.header.get("NAXIS1", observation.image.shape[-1]))
    ny = int(observation.header.get("NAXIS2", observation.image.shape[-2]))
    x = torch.tensor([-0.5, nx - 0.5, nx - 0.5, -0.5], dtype=torch.float64)
    y = torch.tensor([-0.5, -0.5, ny - 0.5, ny - 0.5], dtype=torch.float64)
    try:
        ra, dec = _observation_pixel_to_world(observation, x, y)
        ra_values = torch.as_tensor(ra, dtype=torch.float64)
        dec_values = torch.as_tensor(dec, dtype=torch.float64)
        finite = torch.isfinite(ra_values) & torch.isfinite(dec_values)
        if bool(torch.all(finite)):
            return ra_values.tolist(), dec_values.tolist()
    except (RuntimeError, TypeError, ValueError):
        pass
    return (
        [float(observation.header.get("CRVAL1", 0.0))],
        [float(observation.header.get("CRVAL2", 0.0))],
    )


def estimate_field_domain(
    observations: Sequence[FieldObservation],
    *,
    coord_mode: FieldCoordinateMode = "local_tangent",
) -> FieldDomain:
    if not observations:
        raise ValueError("estimate_field_domain requires at least one observation")
    wavelength_lowers: list[float] = []
    wavelength_uppers: list[float] = []
    footprint_ra: list[float] = []
    footprint_dec: list[float] = []
    for observation in observations:
        lower, upper = _bandpass_bounds(observation)
        wavelength_lowers.append(lower)
        wavelength_uppers.append(upper)
        ra, dec = _observation_footprint_world_deg(observation)
        footprint_ra.extend(ra)
        footprint_dec.extend(dec)
    ra_rad = torch.deg2rad(torch.tensor(footprint_ra, dtype=torch.float64))
    dec_rad = torch.deg2rad(torch.tensor(footprint_dec, dtype=torch.float64))
    cos_dec = torch.cos(dec_rad)
    mean_x = float(torch.mean(cos_dec * torch.cos(ra_rad)))
    mean_y = float(torch.mean(cos_dec * torch.sin(ra_rad)))
    mean_z = float(torch.mean(torch.sin(dec_rad)))
    center_ra = math.degrees(math.atan2(mean_y, mean_x)) % 360.0
    center_dec = math.degrees(math.atan2(mean_z, math.hypot(mean_x, mean_y)))
    tangent_cos_dec = max(math.cos(math.radians(center_dec)), 1.0e-6)
    x_offsets = [
        ((ra - center_ra + 180.0) % 360.0 - 180.0) * tangent_cos_dec
        for ra in footprint_ra
    ]
    y_offsets = [dec - center_dec for dec in footprint_dec]
    # Preserve the historical normalization where one detector width maps to
    # roughly one coordinate unit while covering the full footprint union.
    spatial_scale = 2.0 * max(
        max(abs(value) for value in x_offsets),
        max(abs(value) for value in y_offsets),
        1.0e-4,
    )
    lower = min(wavelength_lowers)
    upper = max(wavelength_uppers)
    wavelength_center = 0.5 * (lower + upper)
    wavelength_scale = max(0.5 * (upper - lower), 1.0)
    return FieldDomain(
        center_ra_deg=center_ra,
        center_dec_deg=center_dec,
        spatial_scale_deg=spatial_scale,
        wavelength_center=wavelength_center,
        wavelength_scale=wavelength_scale,
        coord_mode=coord_mode,
    )


class _BaseFieldModel(nn.Module):
    """Latent sky field; subclasses implement ``forward(ra, dec, λ)``.

    Physics layers (extinction, beams, etc.) can be composed by wrapping this
    module or by subclassing and applying deterministic transforms inside
    ``forward`` while keeping the public signature.
    """

    def __init__(self, domain: FieldDomain) -> None:
        super().__init__()
        self.domain = domain
        self.coord_mode = domain.coord_mode

    @property
    def coord_dim(self) -> int:
        if self.coord_mode == "local_tangent":
            return 3
        if self.coord_mode == "sphere_xyz":
            return 4
        raise ValueError(f"unsupported field coordinate mode {self.coord_mode!r}")

    def _encoded_coords(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        wavelength = torch.as_tensor(
            wavelength_angstrom,
            device=ra_deg.device,
            dtype=ra_deg.dtype,
        )
        if wavelength.shape != ra_deg.shape:
            wavelength = torch.broadcast_to(wavelength, ra_deg.shape)
        dlam = wavelength - float(self.domain.wavelength_center)
        if self.coord_mode == "local_tangent":
            dra = _wrapped_delta_ra_deg(ra_deg, self.domain.center_ra_deg)
            dra = dra * math.cos(math.radians(self.domain.center_dec_deg))
            ddec = dec_deg - float(self.domain.center_dec_deg)
            return torch.stack(
                [
                    dra / float(self.domain.spatial_scale_deg),
                    ddec / float(self.domain.spatial_scale_deg),
                    dlam / float(self.domain.wavelength_scale),
                ],
                dim=-1,
            )
        if self.coord_mode == "sphere_xyz":
            ra_rad = torch.deg2rad(ra_deg)
            dec_rad = torch.deg2rad(dec_deg)
            cos_dec = torch.cos(dec_rad)
            return torch.stack(
                [
                    cos_dec * torch.cos(ra_rad),
                    cos_dec * torch.sin(ra_rad),
                    torch.sin(dec_rad),
                    dlam / float(self.domain.wavelength_scale),
                ],
                dim=-1,
            )
        raise ValueError(f"unsupported field coordinate mode {self.coord_mode!r}")

    def _unit_box_coords(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        coords = self._encoded_coords(ra_deg, dec_deg, wavelength_angstrom)
        return 0.5 * (torch.tanh(coords) + 1.0)

    def _spatial_coords(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        coords = self._encoded_coords(ra_deg, dec_deg, wavelength_angstrom)
        return coords[..., :-1]

    def _spatial_unit_box_coords(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        spatial = self._spatial_coords(ra_deg, dec_deg, wavelength_angstrom)
        return 0.5 * (torch.tanh(spatial) + 1.0)

    def _normalized_wavelength(
        self,
        wavelength_angstrom: Tensor | float,
        *,
        device: torch.device,
        dtype: torch.dtype,
        shape: torch.Size,
    ) -> Tensor:
        wavelength = torch.as_tensor(
            wavelength_angstrom,
            device=device,
            dtype=dtype,
        )
        if wavelength.shape != shape:
            wavelength = torch.broadcast_to(wavelength, shape)
        return (wavelength - float(self.domain.wavelength_center)) / float(
            self.domain.wavelength_scale
        )

class SpectralPowerFoamField(_BaseFieldModel):
    """Adapt a spectral Power Foam to the common sky-field forward interface.

    Sites and radii use the normalized local-tangent coordinates defined by
    ``domain``. Spectral coefficients multiply a Legendre basis over normalized
    wavelength, avoiding poorly conditioned raw powers for broad bandpasses.
    """

    def __init__(
        self,
        domain: FieldDomain,
        *,
        sites: Tensor,
        radii: Tensor,
        cell_coefficients: Tensor,
        cell_gradient_coefficients: Tensor | None = None,
        minimum_radius: float = 0.0,
        temperature_start: float = 4.0,
        temperature_end: float = 24.0,
        compact_fixed_geometry: bool = False,
        log_intensity_spectra: bool = False,
    ) -> None:
        super().__init__(domain)
        if self.coord_mode != "local_tangent":
            raise ValueError(
                "SpectralPowerFoamField currently requires local_tangent coordinates"
            )
        start = float(temperature_start)
        end = float(temperature_end)
        if not math.isfinite(start) or not math.isfinite(end) or start <= 0 or end <= 0:
            raise ValueError("Foam temperatures must be finite and positive")
        self.foam = SpectralPowerFoam(
            sites,
            radii,
            cell_coefficients,
            cell_gradient_coefficients=cell_gradient_coefficients,
            minimum_radius=minimum_radius,
            log_intensity_spectra=log_intensity_spectra,
        )
        self.temperature_start = start
        self.temperature_end = end
        self._temperature = end
        self.compact_fixed_geometry = bool(compact_fixed_geometry)
        self.register_buffer(
            "_spectral_basis_transform",
            torch.eye(
                self.foam.basis_count,
                device=self.foam.sites.device,
                dtype=self.foam.sites.dtype,
            ),
        )

    @property
    def sites(self) -> Tensor:
        return self.foam.sites

    @property
    def radii(self) -> Tensor:
        return self.foam.radii

    @property
    def cell_coefficients(self) -> Tensor:
        return self.foam.cell_coefficients

    @property
    def cell_gradient_coefficients(self) -> Tensor | None:
        return self.foam.cell_gradient_coefficients

    @property
    def basis_count(self) -> int:
        return self.foam.basis_count

    def set_curriculum_progress(self, progress: float) -> None:
        bounded = min(max(float(progress), 0.0), 1.0)
        if self.temperature_start == self.temperature_end:
            self._temperature = self.temperature_end
            return
        self._temperature = (
            self.temperature_start
            * (self.temperature_end / self.temperature_start) ** bounded
        )

    def update_adaptation_statistics(
        self,
        cell_indices: Tensor,
        whitened_residuals: Tensor,
        contributions: Tensor,
        positions: Tensor,
        *,
        split: str,
        decay: float = 0.9,
    ) -> None:
        """Accumulate held-out evidence used by deterministic cell adaptation."""
        self.foam.update_adaptation_statistics(
            cell_indices,
            whitened_residuals,
            contributions,
            positions,
            split=split,
            decay=decay,
        )

    def adapt_cells(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        densify_count: int,
        prune_count: int,
        min_updates: int = 1,
    ) -> FoamAdaptationResult:
        """Adapt cell topology while retaining optimizer state for survivors."""
        return adapt_spectral_power_foam(
            self.foam,
            optimizer,
            densify_count=densify_count,
            prune_count=prune_count,
            min_updates=min_updates,
        )

    def _legendre_basis(self, normalized: Tensor) -> Tensor:
        terms = [torch.ones_like(normalized)]
        if self.basis_count > 1:
            terms.append(normalized)
        for degree in range(2, self.basis_count):
            terms.append(
                ((2 * degree - 1) * normalized * terms[-1] - (degree - 1) * terms[-2])
                / degree
            )
        return torch.stack(terms, dim=-1)

    def spectral_basis(self, wavelength: Tensor | float) -> Tensor:
        """Return the Legendre basis shared by smooth and exact rendering."""
        value = torch.as_tensor(
            wavelength,
            device=self.sites.device,
            dtype=self.sites.dtype,
        )
        normalized = (value - float(self.domain.wavelength_center)) / float(
            self.domain.wavelength_scale
        )
        return self._legendre_basis(normalized) @ self._spectral_basis_transform

    def set_spectral_basis_transform(self, transform: Tensor) -> None:
        """Set a fixed right transform of the continuous spectral basis."""
        if (
            not transform.is_floating_point()
            or transform.shape != (self.basis_count, self.basis_count)
            or transform.device != self.sites.device
            or transform.dtype != self.sites.dtype
            or not bool(torch.isfinite(transform).all())
        ):
            raise ValueError(
                "transform must be a finite basis-by-basis tensor matching the model"
            )
        with torch.no_grad():
            self._spectral_basis_transform.copy_(transform)

    def normalized_spatial_coordinates(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
    ) -> Tensor:
        """Map sky coordinates to the Foam's normalized local tangent plane."""
        wavelength = torch.full_like(
            ra_deg,
            float(self.domain.wavelength_center),
        )
        return self._encoded_coords(ra_deg, dec_deg, wavelength)[..., :2].to(
            device=self.sites.device,
            dtype=self.sites.dtype,
        )

    def forward(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength: Tensor | float,
    ) -> Tensor:
        coords = self._encoded_coords(ra_deg, dec_deg, wavelength).to(
            dtype=self.sites.dtype
        )
        return self.foam(
            coords[..., :2],
            self._legendre_basis(coords[..., 2]) @ self._spectral_basis_transform,
            power_temperature=self._temperature,
            support_temperature=self._temperature,
            compact_fixed_geometry=self.compact_fixed_geometry,
        )

    def exact_operator(
        self,
        pixel_polygons: Tensor,
        **operator_kwargs: Any,
    ) -> PowerPixelOperator:
        """Build a hard operator from detached geometry and normalized pixels."""
        return sparse_bounded_power_pixel_operator(
            pixel_polygons,
            self.sites.detach(),
            self.radii.detach(),
            **operator_kwargs,
        )

    def render_exact(
        self,
        pixel_polygons: Tensor,
        wavelengths: Tensor,
        response_weights: Tensor,
        *,
        semantics: PowerValueSemantics = "surface_brightness",
        **operator_kwargs: Any,
    ) -> Tensor:
        """Hard-render convex pixels with the learned spectral coefficients."""
        operator = self.exact_operator(pixel_polygons, **operator_kwargs)
        return self.foam.render_exact(
            operator,
            self.spectral_basis(wavelengths),
            response_weights,
            semantics=semantics,
        )


class FoamInitializedINRField(_BaseFieldModel):
    """Positive INR correction initialized exactly from a fitted Foam field.

    The reference Foam is frozen and copied by default. A signed MLP predicts
    a bounded log-intensity correction whose final layer starts at zero, so
    the initial field is exactly the positive-clipped Foam prediction.
    """

    def __init__(
        self,
        domain: FieldDomain,
        *,
        foam: SpectralPowerFoamField,
        hidden_dim: int = 64,
        hidden_layers: int = 3,
        intensity_floor: float = 1.0e-8,
        max_log_correction: float = 8.0,
        copy_reference: bool = True,
    ) -> None:
        super().__init__(domain)
        if foam.domain != domain:
            raise ValueError("Foam and INR domains must match")
        if int(hidden_dim) < 1 or int(hidden_layers) < 1:
            raise ValueError("hidden_dim and hidden_layers must be positive")
        if not math.isfinite(intensity_floor) or float(intensity_floor) <= 0.0:
            raise ValueError("intensity_floor must be finite and positive")
        if not math.isfinite(max_log_correction) or float(max_log_correction) <= 0.0:
            raise ValueError("max_log_correction must be finite and positive")
        self.reference_foam = copy.deepcopy(foam) if copy_reference else foam
        self.reference_foam.requires_grad_(False)
        self.residual_trunk = silu_mlp_sequential(
            self.coord_dim, int(hidden_dim), int(hidden_layers)
        )
        self.residual_output = nn.Linear(int(hidden_dim), 1)
        reset_silu_mlp_linears(self.residual_trunk)
        nn.init.zeros_(self.residual_output.weight)
        nn.init.zeros_(self.residual_output.bias)
        self.intensity_floor = float(intensity_floor)
        self.max_log_correction = float(max_log_correction)

    def log_intensity_correction(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        coords = self._encoded_coords(ra_deg, dec_deg, wavelength_angstrom)
        parameter = self.residual_output.weight
        hidden = self.residual_trunk(
            coords.to(device=parameter.device, dtype=parameter.dtype)
        )
        raw = self.residual_output(hidden).squeeze(-1)
        return self.max_log_correction * torch.tanh(raw / self.max_log_correction)

    def forward(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        with torch.no_grad():
            reference = self.reference_foam(ra_deg, dec_deg, wavelength_angstrom)
        correction = self.log_intensity_correction(ra_deg, dec_deg, wavelength_angstrom)
        reference = reference.to(device=correction.device, dtype=correction.dtype)
        return reference.clamp_min(self.intensity_floor) * torch.exp(correction)


class DiffuseFoamHybridField(_BaseFieldModel):
    """Low-capacity diffuse field plus localized non-negative spectral Foam.

    Only the summed sky is an identified product. Component values are exposed
    for diagnostics and injection tests, not as separately inferred physical
    skies without additional identifiability evidence.
    """

    _STAGES = {"diffuse": 0, "foam": 1, "joint": 2}

    def __init__(
        self,
        domain: FieldDomain,
        *,
        foam: SpectralPowerFoamField,
        diffuse_hidden_dim: int = 24,
        diffuse_hidden_layers: int = 1,
        diffuse_omega0: float = 3.0,
        training_stage: Literal["diffuse", "foam", "joint"] = "joint",
    ) -> None:
        super().__init__(domain)
        if foam.domain != domain:
            raise ValueError("diffuse and Foam domains must match")
        if not foam.foam.log_intensity_spectra:
            raise ValueError(
                "diffuse/Foam hybrid requires non-negative log-intensity Foam spectra"
            )
        if int(diffuse_hidden_dim) < 1 or int(diffuse_hidden_layers) < 1:
            raise ValueError("diffuse hidden sizes must be positive")
        omega0 = float(diffuse_omega0)
        if not math.isfinite(omega0) or omega0 <= 0.0:
            raise ValueError("diffuse_omega0 must be finite and positive")
        self.diffuse_field = SineImplicitField(
            domain,
            hidden_dim=int(diffuse_hidden_dim),
            hidden_layers=int(diffuse_hidden_layers),
            omega0=omega0,
        )
        self.localized_foam = foam
        self.component_identifiability = "combined_only"
        self.register_buffer("_training_stage_code", torch.tensor(2, dtype=torch.int8))
        self.set_training_stage(training_stage)

    @property
    def training_stage(self) -> str:
        code = int(self._training_stage_code.item())
        for name, value in self._STAGES.items():
            if value == code:
                return name
        raise ValueError(f"invalid stored hybrid training stage code {code}")

    def set_training_stage(
        self,
        stage: Literal["diffuse", "foam", "joint"],
    ) -> None:
        try:
            code = self._STAGES[str(stage)]
        except KeyError as exc:
            raise ValueError(f"unsupported hybrid training stage {stage!r}") from exc
        diffuse_trainable = stage in {"diffuse", "joint"}
        foam_trainable = stage in {"foam", "joint"}
        self.diffuse_field.requires_grad_(diffuse_trainable)
        self.localized_foam.requires_grad_(foam_trainable)
        self._training_stage_code.fill_(code)

    def diagnostic_components(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> tuple[Tensor, Tensor]:
        diffuse = self.diffuse_field(ra_deg, dec_deg, wavelength_angstrom)
        localized = self.localized_foam(ra_deg, dec_deg, wavelength_angstrom).to(
            device=diffuse.device,
            dtype=diffuse.dtype,
        )
        return diffuse, localized

    def component_regularization(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
        *,
        foam_sparsity_weight: float,
        correlation_weight: float,
    ) -> Tensor:
        sparsity_weight = float(foam_sparsity_weight)
        separation_weight = float(correlation_weight)
        if (
            not math.isfinite(sparsity_weight)
            or not math.isfinite(separation_weight)
            or sparsity_weight < 0.0
            or separation_weight < 0.0
        ):
            raise ValueError("hybrid regularization weights must be finite and non-negative")
        diffuse, localized = self.diagnostic_components(
            ra_deg,
            dec_deg,
            wavelength_angstrom,
        )
        diffuse_centered = diffuse - diffuse.mean()
        localized_centered = localized - localized.mean()
        dot = torch.sum(diffuse_centered * localized_centered)
        scale = torch.sum(diffuse_centered.square()) * torch.sum(
            localized_centered.square()
        )
        correlation_squared = dot.square() / torch.clamp(
            scale,
            min=torch.finfo(diffuse.dtype).tiny,
        )
        return sparsity_weight * localized.abs().mean() + separation_weight * correlation_squared

    def forward(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        diffuse, localized = self.diagnostic_components(
            ra_deg,
            dec_deg,
            wavelength_angstrom,
        )
        return diffuse + localized


def _observation_without_background(observation: FieldObservation) -> FieldObservation:
    stripped = copy.deepcopy(observation)
    stripped.sky_background = None
    stripped.instrument_background = None
    stripped.detector_background = None
    stripped.background_offset = 0.0
    return stripped


def _render_field_component_anchor_target(
    term: FieldComponentAnchorTerm,
    *,
    num_spectral_samples: int,
    spectral_sampling: FieldSpectralSampling,
    spectral_sampling_grid_size: int | None,
    generator: torch.Generator | None,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if term.target_image is not None:
        return term.target_image.to(device=device, dtype=dtype)
    if term.target_values is not None:
        return term.target_values.to(device=device, dtype=dtype)
    assert term.reference_model is not None
    base_observation = _observation_without_background(term.observation)
    if isinstance(term.reference_model, FieldComponentDecomposition):
        if term.sample_ra_deg is not None:
            return term.reference_model.render_component_samples(
                base_observation,
                ra_deg=term.sample_ra_deg.to(device=device, dtype=torch.float32),
                dec_deg=term.sample_dec_deg.to(device=device, dtype=torch.float32),
                roles=term.roles,
                num_spectral_samples=num_spectral_samples,
                spectral_sampling=spectral_sampling,
                spectral_sampling_grid_size=spectral_sampling_grid_size,
                generator=generator,
                override_wavelengths=term.override_wavelengths,
                override_weights=term.override_bandpass_weights,
                forward_context=term.forward_context,
            ).to(device=device, dtype=dtype)
        return term.reference_model.render_component_observation(
            base_observation,
            roles=term.roles,
            num_spectral_samples=num_spectral_samples,
            spectral_sampling=spectral_sampling,
            spectral_sampling_grid_size=spectral_sampling_grid_size,
            generator=generator,
            override_wavelengths=term.override_wavelengths,
            override_weights=term.override_bandpass_weights,
            forward_context=term.forward_context,
        ).to(device=device, dtype=dtype)
    if term.sample_ra_deg is not None:
        return render_field_samples(
            term.reference_model,
            base_observation,
            ra_deg=term.sample_ra_deg.to(device=device, dtype=torch.float32),
            dec_deg=term.sample_dec_deg.to(device=device, dtype=torch.float32),
            num_spectral_samples=num_spectral_samples,
            spectral_sampling=spectral_sampling,
            spectral_sampling_grid_size=spectral_sampling_grid_size,
            generator=generator,
            override_wavelengths=term.override_wavelengths,
            override_weights=term.override_bandpass_weights,
            forward_context=term.forward_context,
        ).to(device=device, dtype=dtype)
    return render_field_observation(
        term.reference_model,
        base_observation,
        num_spectral_samples=num_spectral_samples,
        spectral_sampling=spectral_sampling,
        spectral_sampling_grid_size=spectral_sampling_grid_size,
        generator=generator,
        override_wavelengths=term.override_wavelengths,
        override_weights=term.override_bandpass_weights,
        forward_context=term.forward_context,
    ).to(device=device, dtype=dtype)


def _resolve_anchor_sample_world_coordinates(
    term: FieldComponentAnchorTerm,
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    if term.sample_ra_deg is not None and term.sample_dec_deg is not None:
        return (
            term.sample_ra_deg.to(device=device, dtype=torch.float32),
            term.sample_dec_deg.to(device=device, dtype=torch.float32),
        )
    if term.sample_x_pix is None or term.sample_y_pix is None:
        raise ValueError("anchor term does not define sample coordinates")
    x_pix = term.sample_x_pix.to(device=device, dtype=torch.float64)
    y_pix = term.sample_y_pix.to(device=device, dtype=torch.float64)
    ra_deg, dec_deg = _observation_pixel_to_world(term.observation, x_pix, y_pix)
    return ra_deg.to(dtype=torch.float32), dec_deg.to(dtype=torch.float32)


def _resolve_centroid_target_pixel_coordinates(
    term: FieldCentroidAnchorTerm,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    if term.target_x_pix is not None and term.target_y_pix is not None:
        return (
            torch.tensor(float(term.target_x_pix), device=device, dtype=dtype),
            torch.tensor(float(term.target_y_pix), device=device, dtype=dtype),
        )
    if term.observation.detector_to_sky is not None:
        raise ValueError(
            "sky-coordinate centroid anchors require an inverse detector transform; "
            "provide target_x_pix and target_y_pix for forward-only native astrometry"
        )
    assert term.target_ra_deg is not None and term.target_dec_deg is not None
    x_pix, y_pix = WCS(term.observation.header).world_to_pixel(
        torch.tensor(float(term.target_ra_deg), dtype=torch.float64),
        torch.tensor(float(term.target_dec_deg), dtype=torch.float64),
    )
    return (
        x_pix.to(device=device, dtype=dtype),
        y_pix.to(device=device, dtype=dtype),
    )


def _pixel_coordinate_grids(
    shape: tuple[int, int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    y = torch.arange(shape[0], device=device, dtype=dtype)
    x = torch.arange(shape[1], device=device, dtype=dtype)
    y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")
    return x_grid, y_grid


def field_component_anchor_loss(
    model: nn.Module,
    component_anchor_terms: Sequence[FieldComponentAnchorTerm],
    *,
    num_spectral_samples: int = 9,
    spectral_sampling: FieldSpectralSampling = "deterministic_quadrature",
    spectral_sampling_grid_size: int | None = None,
    generator: torch.Generator | None = None,
    loss_kind: FieldLossKind = "mse",
    charbonnier_eps: float = 1.0e-3,
    huber_delta: float = 5.0e-2,
) -> Tensor:
    """Weighted anchor loss for selected field decomposition roles."""
    if loss_kind == "gaussian_nll":
        raise ValueError("field component anchors do not support gaussian_nll")
    total = None
    for term in component_anchor_terms:
        if term.loss_weight == 0.0:
            continue
        base_observation = _observation_without_background(term.observation)
        if (
            term.sample_ra_deg is not None
            or term.sample_dec_deg is not None
            or term.sample_x_pix is not None
            or term.sample_y_pix is not None
        ):
            ra_anchor, dec_anchor = _resolve_anchor_sample_world_coordinates(
                term,
                device=next(
                    (p.device for p in model.parameters()), default_compute_device()
                ),
            )
            if isinstance(model, FieldComponentDecomposition):
                prediction = model.render_component_samples(
                    base_observation,
                    ra_deg=ra_anchor,
                    dec_deg=dec_anchor,
                    roles=term.roles,
                    num_spectral_samples=num_spectral_samples,
                    spectral_sampling=spectral_sampling,
                    spectral_sampling_grid_size=spectral_sampling_grid_size,
                    generator=generator,
                    override_wavelengths=term.override_wavelengths,
                    override_weights=term.override_bandpass_weights,
                    forward_context=term.forward_context,
                )
            else:
                if term.roles != SurveyInstrumentComponentMixer._VALID_ROLES:
                    raise ValueError(
                        "component anchor role selection requires FieldComponentDecomposition"
                    )
                prediction = render_field_samples(
                    model,
                    base_observation,
                    ra_deg=ra_anchor,
                    dec_deg=dec_anchor,
                    num_spectral_samples=num_spectral_samples,
                    spectral_sampling=spectral_sampling,
                    spectral_sampling_grid_size=spectral_sampling_grid_size,
                    generator=generator,
                    override_wavelengths=term.override_wavelengths,
                    override_weights=term.override_bandpass_weights,
                    forward_context=term.forward_context,
                )
        elif isinstance(model, FieldComponentDecomposition):
            prediction = model.render_component_observation(
                base_observation,
                roles=term.roles,
                num_spectral_samples=num_spectral_samples,
                spectral_sampling=spectral_sampling,
                spectral_sampling_grid_size=spectral_sampling_grid_size,
                generator=generator,
                override_wavelengths=term.override_wavelengths,
                override_weights=term.override_bandpass_weights,
                forward_context=term.forward_context,
            )
        else:
            if term.roles != SurveyInstrumentComponentMixer._VALID_ROLES:
                raise ValueError(
                    "component anchor role selection requires FieldComponentDecomposition"
                )
            prediction = render_field_observation(
                model,
                base_observation,
                num_spectral_samples=num_spectral_samples,
                spectral_sampling=spectral_sampling,
                spectral_sampling_grid_size=spectral_sampling_grid_size,
                generator=generator,
                override_wavelengths=term.override_wavelengths,
                override_weights=term.override_bandpass_weights,
                forward_context=term.forward_context,
            )
        target = _render_field_component_anchor_target(
            term,
            num_spectral_samples=num_spectral_samples,
            spectral_sampling=spectral_sampling,
            spectral_sampling_grid_size=spectral_sampling_grid_size,
            generator=generator,
            device=prediction.device,
            dtype=prediction.dtype,
        )
        if (
            term.sample_ra_deg is not None
            or term.sample_dec_deg is not None
            or term.sample_x_pix is not None
            or term.sample_y_pix is not None
        ):
            residual = prediction - target
            weight = (
                term.sample_weight.to(device=prediction.device, dtype=prediction.dtype)
                if term.sample_weight is not None
                else torch.ones_like(prediction)
            )
        else:
            valid = _observation_support_mask(term.observation).to(
                device=prediction.device
            )
            target = torch.where(valid, target, torch.zeros_like(target))
            residual = torch.where(
                valid, prediction - target, torch.zeros_like(prediction)
            )
            weight = (
                term.observation.weight.to(
                    device=prediction.device, dtype=prediction.dtype
                )
                if term.observation.weight is not None
                else torch.ones_like(prediction)
            )
            weight = torch.where(valid, weight, torch.zeros_like(weight))
        weighted_loss = torch.sum(
            weight
            * _field_residual_loss(
                residual,
                loss_kind=loss_kind,
                charbonnier_eps=charbonnier_eps,
                huber_delta=huber_delta,
                logvar=None,
            )
        )
        term_loss = float(term.loss_weight) * (
            weighted_loss / torch.clamp(weight.sum(), min=1.0e-8)
        )
        total = term_loss if total is None else total + term_loss
    if total is None:
        return torch.zeros(
            (),
            device=next(
                (p.device for p in model.parameters()), default_compute_device()
            ),
            dtype=torch.float32,
        )
    return total


def field_component_centroid_anchor_loss(
    model: nn.Module,
    centroid_anchor_terms: Sequence[FieldCentroidAnchorTerm],
    *,
    num_spectral_samples: int = 9,
    spectral_sampling: FieldSpectralSampling = "deterministic_quadrature",
    spectral_sampling_grid_size: int | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Weighted centroid loss for geometry-aware role anchors."""
    total = None
    for term in centroid_anchor_terms:
        if term.loss_weight == 0.0:
            continue
        base_observation = _observation_without_background(term.observation)
        if isinstance(model, FieldComponentDecomposition):
            prediction = model.render_component_observation(
                base_observation,
                roles=term.roles,
                num_spectral_samples=num_spectral_samples,
                spectral_sampling=spectral_sampling,
                spectral_sampling_grid_size=spectral_sampling_grid_size,
                generator=generator,
                forward_context=term.forward_context,
            )
        else:
            if term.roles != SurveyInstrumentComponentMixer._VALID_ROLES:
                raise ValueError(
                    "centroid anchor role selection requires FieldComponentDecomposition"
                )
            prediction = render_field_observation(
                model,
                base_observation,
                num_spectral_samples=num_spectral_samples,
                spectral_sampling=spectral_sampling,
                spectral_sampling_grid_size=spectral_sampling_grid_size,
                generator=generator,
                forward_context=term.forward_context,
            )
        target_x, target_y = _resolve_centroid_target_pixel_coordinates(
            term,
            device=prediction.device,
            dtype=prediction.dtype,
        )
        x_grid, y_grid = _pixel_coordinate_grids(
            tuple(prediction.shape),
            device=prediction.device,
            dtype=prediction.dtype,
        )
        sigma_pix = torch.tensor(
            float(term.window_sigma_pix),
            device=prediction.device,
            dtype=prediction.dtype,
        )
        valid = _observation_support_mask(term.observation).to(device=prediction.device)
        prediction_scale = torch.clamp(prediction.detach().abs().amax(), min=1.0)
        logits = 16.0 * (prediction / prediction_scale)
        logits = torch.where(valid, logits, torch.full_like(logits, -1.0e9))
        energy = torch.softmax(logits.reshape(-1), dim=0).reshape_as(prediction)
        centroid_x = (energy * x_grid).sum()
        centroid_y = (energy * y_grid).sum()
        term_loss = float(term.loss_weight) * (
            ((centroid_x - target_x) / sigma_pix).square()
            + ((centroid_y - target_y) / sigma_pix).square()
        )
        total = term_loss if total is None else total + term_loss
    if total is None:
        return torch.zeros(
            (),
            device=next(
                (p.device for p in model.parameters()), default_compute_device()
            ),
            dtype=torch.float32,
        )
    return total


def _infer_context_axis_labels(
    observation_forward_contexts: Sequence[ForwardContext | None] | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if observation_forward_contexts is None:
        return (), ()
    surveys: list[str] = []
    instruments: list[str] = []
    for context in observation_forward_contexts:
        if context is None:
            continue
        if context.survey is not None and context.survey not in surveys:
            surveys.append(context.survey)
        if context.instrument is not None and context.instrument not in instruments:
            instruments.append(context.instrument)
    return tuple(surveys), tuple(instruments)


class FieldComponentDecomposition(_BaseFieldModel):
    """Explicit shared/private decomposition over multiple field models."""

    def __init__(
        self,
        component_models: Sequence[nn.Module],
        *,
        component_roles: Sequence[str],
        surveys: Sequence[str] = (),
        instruments: Sequence[str] = (),
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
    ) -> None:
        models = tuple(component_models)
        roles = tuple(str(role) for role in component_roles)
        if not models:
            raise ValueError("component_models must be non-empty")
        if len(models) != len(roles):
            raise ValueError("component_roles length must match component_models")
        domain = getattr(models[0], "domain", None)
        if not isinstance(domain, FieldDomain):
            raise ValueError(
                "component_models[0] must expose a FieldDomain via .domain"
            )
        for idx, model in enumerate(models[1:], start=1):
            if getattr(model, "domain", None) != domain:
                raise ValueError(
                    f"component_models[{idx}] domain must match component_models[0].domain"
                )
        super().__init__(domain)
        self.component_roles = roles
        self.component_models = nn.ModuleList(models)
        self.component_gains = nn.Parameter(
            torch.ones(len(self.component_roles), dtype=dtype, device=device)
        )
        self.mixer = SurveyInstrumentComponentMixer(
            component_roles=self.component_roles,
            surveys=surveys,
            instruments=instruments,
            dtype=dtype,
            device=device,
        )

    def _role_indices(self, roles: str | Sequence[str]) -> tuple[int, ...]:
        role_names = (
            (roles,) if isinstance(roles, str) else tuple(str(role) for role in roles)
        )
        valid_roles = set(SurveyInstrumentComponentMixer._VALID_ROLES)
        unknown = tuple(role for role in role_names if role not in valid_roles)
        if unknown:
            raise ValueError(f"unsupported component roles: {unknown!r}")
        role_set = set(role_names)
        return tuple(
            idx for idx, role in enumerate(self.component_roles) if role in role_set
        )

    def _context_labels(
        self,
        forward_context: ForwardContext | None,
    ) -> tuple[str | None, str | None]:
        if forward_context is None:
            return None, None
        return forward_context.survey, forward_context.instrument

    def component_gain_magnitudes(self) -> Tensor:
        return torch.abs(self.component_gains.detach())

    def fix_component_scale_gauge_(
        self,
        *,
        target_gain: float = 1.0,
        min_gain: float = 1.0e-12,
    ) -> Tensor:
        """Canonicalize explicit component gains without changing rendered predictions."""
        if target_gain <= 0.0:
            raise ValueError("target_gain must be positive")
        if min_gain <= 0.0:
            raise ValueError("min_gain must be positive")
        gains = self.component_gains.detach()
        safe = torch.abs(gains) > float(min_gain)
        scale = torch.where(
            safe,
            gains / float(target_gain),
            torch.ones_like(gains),
        )
        with torch.no_grad():
            self.component_gains.div_(scale)
        nonzero_scale = scale[safe]
        if int(nonzero_scale.numel()) > 0:
            full_scale = torch.where(safe, scale, torch.ones_like(scale))
            self.mixer.scale_component_columns_(full_scale)
        return self.component_gains.detach().clone()

    def component_prediction(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
        *,
        roles: str | Sequence[str],
        forward_context: ForwardContext | None = None,
        aggregate: bool = True,
    ) -> Tensor:
        selected = self._role_indices(roles)
        if not selected:
            if aggregate:
                return torch.zeros_like(ra_deg)
            return torch.zeros(
                (0, *ra_deg.shape),
                dtype=ra_deg.dtype,
                device=ra_deg.device,
            )
        survey, instrument = self._context_labels(forward_context)
        weights = self.mixer(survey=survey, instrument=instrument)
        outputs = []
        for idx in selected:
            plane = _call_scalar_field_forward(
                self.component_models[idx],
                ra_deg,
                dec_deg,
                wavelength_angstrom,
                forward_context=forward_context,
            )
            gain = self.component_gains[idx].to(dtype=plane.dtype, device=plane.device)
            outputs.append(
                weights[idx].to(dtype=plane.dtype, device=plane.device) * gain * plane
            )
        stacked = torch.stack(outputs, dim=0)
        if aggregate:
            return stacked.sum(dim=0)
        return stacked

    def forward(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
        *,
        forward_context: ForwardContext | None = None,
    ) -> Tensor:
        return self.component_prediction(
            ra_deg,
            dec_deg,
            wavelength_angstrom,
            roles=SurveyInstrumentComponentMixer._VALID_ROLES,
            forward_context=forward_context,
            aggregate=True,
        )

    def render_component_observation(
        self,
        observation: FieldObservation,
        *,
        roles: str | Sequence[str],
        num_spectral_samples: int = 9,
        spectral_sampling: FieldSpectralSampling = "deterministic_quadrature",
        spectral_sampling_grid_size: int | None = None,
        generator: torch.Generator | None = None,
        override_wavelengths: Tensor | None = None,
        override_weights: Tensor | None = None,
        forward_context: ForwardContext | None = None,
        aggregate: bool = True,
    ) -> Tensor:
        base_observation = _observation_without_background(observation)
        image = base_observation.image
        if image.ndim != 2:
            raise ValueError("FieldObservation.image must be 2D")
        device = next((p.device for p in self.parameters()), default_compute_device())
        ra_deg, dec_deg = _coordinate_grid(
            base_observation,
            shape=tuple(image.shape),
            device=device,
        )
        if override_wavelengths is not None or override_weights is not None:
            if override_wavelengths is None or override_weights is None:
                raise ValueError(
                    "override_wavelengths and override_weights must be provided together"
                )
            wavelengths, weights = _validate_override_quadrature(
                override_wavelengths,
                override_weights,
                device=device,
                dtype=torch.float32,
            )
        elif spectral_sampling == "deterministic_quadrature":
            wavelengths, weights = _bandpass_quadrature(
                base_observation,
                num_spectral_samples=num_spectral_samples,
                device=device,
            )
        elif spectral_sampling == "bandpass_mc":
            wavelengths, weights = _bandpass_mc_samples(
                base_observation,
                num_spectral_samples=num_spectral_samples,
                device=device,
                proposal_grid_size=spectral_sampling_grid_size,
                generator=generator,
            )
        else:
            raise ValueError(f"unsupported spectral_sampling {spectral_sampling!r}")
        selected = self._role_indices(roles)
        if not selected:
            shape = (0, *image.shape) if not aggregate else tuple(image.shape)
            return torch.zeros(shape, dtype=image.dtype, device=device)
        rendered_shape = (
            tuple(image.shape) if aggregate else (len(selected), *image.shape)
        )
        rendered = torch.zeros(rendered_shape, device=device, dtype=torch.float32)
        ra32 = ra_deg.to(dtype=torch.float32)
        dec32 = dec_deg.to(dtype=torch.float32)
        for i, wavelength in enumerate(wavelengths):
            plane = self.component_prediction(
                ra32,
                dec32,
                wavelength,
                roles=roles,
                forward_context=forward_context,
                aggregate=aggregate,
            )
            weight = weights[i].to(dtype=plane.dtype, device=plane.device)
            if aggregate:
                rendered = rendered + plane * weight
            else:
                rendered = rendered + plane * weight.view(1, 1, 1)
        if (
            base_observation.psf_spec is not None
            and base_observation.psf_spec.mode != "none"
        ):
            if aggregate:
                rendered = apply_psf_spec(rendered, base_observation.psf_spec)
            else:
                rendered = torch.stack(
                    [
                        apply_psf_spec(rendered[idx], base_observation.psf_spec)
                        for idx in range(int(rendered.shape[0]))
                    ],
                    dim=0,
                )
        return rendered.to(dtype=image.dtype)

    def render_component_samples(
        self,
        observation: FieldObservation,
        *,
        ra_deg: Tensor,
        dec_deg: Tensor,
        roles: str | Sequence[str],
        num_spectral_samples: int = 9,
        spectral_sampling: FieldSpectralSampling = "deterministic_quadrature",
        spectral_sampling_grid_size: int | None = None,
        generator: torch.Generator | None = None,
        override_wavelengths: Tensor | None = None,
        override_weights: Tensor | None = None,
        forward_context: ForwardContext | None = None,
        aggregate: bool = True,
    ) -> Tensor:
        base_observation = _observation_without_background(observation)
        operator_bundle = field_observation_operator_bundle(base_observation)
        if operator_bundle.psf is not None:
            raise ValueError("component sample rendering does not support PSF blurring")
        if override_wavelengths is not None or override_weights is not None:
            if override_wavelengths is None or override_weights is None:
                raise ValueError(
                    "override_wavelengths and override_weights must be provided together"
                )
            wavelengths, weights = _validate_override_quadrature(
                override_wavelengths,
                override_weights,
                device=ra_deg.device,
                dtype=torch.float32,
            )
        elif spectral_sampling == "deterministic_quadrature":
            wavelengths, weights = _bandpass_quadrature(
                base_observation,
                num_spectral_samples=num_spectral_samples,
                device=ra_deg.device,
            )
        elif spectral_sampling == "bandpass_mc":
            wavelengths, weights = _bandpass_mc_samples(
                base_observation,
                num_spectral_samples=num_spectral_samples,
                device=ra_deg.device,
                proposal_grid_size=spectral_sampling_grid_size,
                generator=generator,
            )
        else:
            raise ValueError(f"unsupported spectral_sampling {spectral_sampling!r}")
        selected = self._role_indices(roles)
        if not selected:
            shape = (0, *ra_deg.shape) if not aggregate else tuple(ra_deg.shape)
            return torch.zeros(shape, dtype=ra_deg.dtype, device=ra_deg.device)
        rendered_shape = (
            tuple(ra_deg.shape) if aggregate else (len(selected), *ra_deg.shape)
        )
        rendered = torch.zeros(
            rendered_shape, device=ra_deg.device, dtype=torch.float32
        )
        ra32 = ra_deg.to(dtype=torch.float32)
        dec32 = dec_deg.to(dtype=torch.float32)
        for i, wavelength in enumerate(wavelengths):
            sample = self.component_prediction(
                ra32,
                dec32,
                wavelength,
                roles=roles,
                forward_context=forward_context,
                aggregate=aggregate,
            )
            weight = weights[i].to(dtype=sample.dtype, device=sample.device)
            if aggregate:
                rendered = rendered + sample * weight
            else:
                rendered = rendered + sample * weight.view(
                    1, *([1] * (sample.ndim - 1))
                )
        return rendered

    def decomposition_diagnostics(
        self,
        observations: Sequence[FieldObservation],
        truth_model: "FieldComponentDecomposition",
        *,
        observation_forward_contexts: Sequence[ForwardContext | None] | None = None,
        num_spectral_samples: int = 9,
        spectral_sampling: FieldSpectralSampling = "deterministic_quadrature",
        spectral_sampling_grid_size: int | None = None,
        generator: torch.Generator | None = None,
        calibration_anchor_model: nn.Module | None = None,
        calibration_anchor_roles: str | Sequence[str] = "shared",
    ) -> ComponentDecompositionDiagnostics:
        if not observations:
            raise ValueError("observations must be non-empty")
        if observation_forward_contexts is not None:
            if len(observation_forward_contexts) != len(observations):
                raise ValueError(
                    "observation_forward_contexts must match observations length"
                )
            contexts = tuple(observation_forward_contexts)
        else:
            contexts = tuple(None for _ in observations)
        private_basis_count = len(
            truth_model._role_indices(("survey", "instrument", "interaction"))
        )
        private_basis_chunks: list[list[Tensor]] = (
            [[] for _ in range(private_basis_count)] if private_basis_count > 0 else []
        )
        shared_est_chunks: list[Tensor] = []
        private_est_chunks: list[Tensor] = []
        shared_truth_chunks: list[Tensor] = []
        private_truth_chunks: list[Tensor] = []
        calibration_anchor_chunks: list[Tensor] = []

        def _relative_norm(error: Tensor, reference: Tensor) -> float:
            error_norm = torch.linalg.vector_norm(torch.as_tensor(error).reshape(-1))
            ref_norm = torch.linalg.vector_norm(torch.as_tensor(reference).reshape(-1))
            if float(ref_norm.item()) == 0.0:
                return (
                    0.0 if float(error_norm.item()) == 0.0 else float(error_norm.item())
                )
            return float((error_norm / ref_norm).item())

        def _subspace_leakage(error: Tensor, basis: Tensor, reference: Tensor) -> float:
            basis_t = torch.as_tensor(basis)
            if int(basis_t.numel()) == 0:
                return 0.0
            basis_rows = basis_t.reshape(int(basis_t.shape[0]), -1)
            nonzero = torch.any(basis_rows != 0, dim=1)
            basis_rows = basis_rows[nonzero]
            if int(basis_rows.shape[0]) == 0:
                return 0.0
            vector = torch.as_tensor(error).reshape(-1)
            dtype = torch.promote_types(vector.dtype, basis_rows.dtype)
            matrix = basis_rows.to(dtype=dtype, device=vector.device).transpose(0, 1)
            rhs = vector.to(dtype=dtype, device=vector.device)
            coeffs = torch.linalg.lstsq(matrix, rhs).solution
            projected = matrix @ coeffs
            proj_norm = torch.linalg.vector_norm(projected)
            ref_norm = torch.linalg.vector_norm(torch.as_tensor(reference).reshape(-1))
            if float(ref_norm.item()) == 0.0:
                return (
                    0.0 if float(proj_norm.item()) == 0.0 else float(proj_norm.item())
                )
            return float((proj_norm / ref_norm).item())

        for observation, forward_context in zip(observations, contexts, strict=True):
            render_kwargs = dict(
                num_spectral_samples=num_spectral_samples,
                spectral_sampling=spectral_sampling,
                spectral_sampling_grid_size=spectral_sampling_grid_size,
                generator=generator,
                forward_context=forward_context,
            )
            shared_est = self.render_component_observation(
                observation,
                roles="shared",
                **render_kwargs,
            )
            private_est = self.render_component_observation(
                observation,
                roles=("survey", "instrument", "interaction"),
                **render_kwargs,
            )
            shared_truth = truth_model.render_component_observation(
                observation,
                roles="shared",
                **render_kwargs,
            )
            private_truth = truth_model.render_component_observation(
                observation,
                roles=("survey", "instrument", "interaction"),
                **render_kwargs,
            )
            private_truth_basis = truth_model.render_component_observation(
                observation,
                roles=("survey", "instrument", "interaction"),
                aggregate=False,
                **render_kwargs,
            )
            shared_est_chunks.append(shared_est.reshape(-1))
            private_est_chunks.append(private_est.reshape(-1))
            shared_truth_chunks.append(shared_truth.reshape(-1))
            private_truth_chunks.append(private_truth.reshape(-1))
            for basis_idx in range(int(private_truth_basis.shape[0])):
                private_basis_chunks[basis_idx].append(
                    private_truth_basis[basis_idx].reshape(-1)
                )
            if calibration_anchor_model is not None:
                base_observation = _observation_without_background(observation)
                if isinstance(calibration_anchor_model, FieldComponentDecomposition):
                    anchor = calibration_anchor_model.render_component_observation(
                        base_observation,
                        roles=calibration_anchor_roles,
                        **render_kwargs,
                    )
                else:
                    anchor = render_field_observation(
                        calibration_anchor_model,
                        base_observation,
                        **render_kwargs,
                    )
                calibration_anchor_chunks.append(anchor.reshape(-1))

        shared_est_vec = torch.cat(shared_est_chunks, dim=0)
        private_est_vec = torch.cat(private_est_chunks, dim=0)
        shared_truth_vec = torch.cat(shared_truth_chunks, dim=0)
        private_truth_vec = torch.cat(private_truth_chunks, dim=0)
        if private_basis_chunks:
            private_truth_basis_vec = torch.stack(
                [torch.cat(chunks, dim=0) for chunks in private_basis_chunks],
                dim=0,
            )
        else:
            private_truth_basis_vec = torch.zeros(
                (0, int(shared_truth_vec.numel())),
                dtype=shared_truth_vec.dtype,
                device=shared_truth_vec.device,
            )
        calibration_transfer_stability = None
        if calibration_anchor_chunks:
            calibration_anchor_vec = torch.cat(calibration_anchor_chunks, dim=0)
            calibration_transfer_stability = _relative_norm(
                shared_est_vec - calibration_anchor_vec,
                calibration_anchor_vec,
            )
        return ComponentDecompositionDiagnostics(
            shared_to_private_leakage=_subspace_leakage(
                shared_est_vec - shared_truth_vec,
                private_truth_basis_vec,
                private_truth_vec,
            ),
            private_to_shared_leakage=_subspace_leakage(
                private_est_vec - private_truth_vec,
                shared_truth_vec.unsqueeze(0),
                shared_truth_vec,
            ),
            shared_relative_error=_relative_norm(
                shared_est_vec - shared_truth_vec,
                shared_truth_vec,
            ),
            private_relative_error=_relative_norm(
                private_est_vec - private_truth_vec,
                private_truth_vec,
            ),
            calibration_transfer_stability=calibration_transfer_stability,
        )


def make_field_component_decomposition(
    *,
    component_roles: Sequence[str],
    domain: FieldDomain | None = None,
    component_models: Sequence[nn.Module] | None = None,
    component_model_kind: FieldModelKind = "sine_implicit",
    component_model_kwargs: dict[str, object] | None = None,
    surveys: Sequence[str] | None = None,
    instruments: Sequence[str] | None = None,
    observation_forward_contexts: Sequence[ForwardContext | None] | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> FieldComponentDecomposition:
    """Build an explicit field decomposition model from component specs or modules."""
    roles = tuple(str(role) for role in component_roles)
    if not roles:
        raise ValueError("component_roles must be non-empty")
    if component_models is not None:
        models = tuple(component_models)
        if len(models) != len(roles):
            raise ValueError("component_models length must match component_roles")
    else:
        if domain is None:
            raise ValueError(
                "domain is required when component_models are not provided"
            )
        model_kwargs = dict(component_model_kwargs or {})
        models = tuple(
            make_field_model(component_model_kind, domain, **model_kwargs)
            for _ in roles
        )
    inferred_surveys, inferred_instruments = _infer_context_axis_labels(
        observation_forward_contexts
    )
    return FieldComponentDecomposition(
        models,
        component_roles=roles,
        surveys=tuple(str(name) for name in surveys)
        if surveys is not None
        else inferred_surveys,
        instruments=tuple(str(name) for name in instruments)
        if instruments is not None
        else inferred_instruments,
        dtype=dtype,
        device=device,
    )


def _sample_feature_volume(volume: Tensor, coords_unit: Tensor) -> Tensor:
    flat = coords_unit.reshape(-1, 3).to(dtype=volume.dtype, device=volume.device)
    grid = torch.stack(
        [
            2.0 * flat[:, 0] - 1.0,
            2.0 * flat[:, 1] - 1.0,
            2.0 * flat[:, 2] - 1.0,
        ],
        dim=-1,
    ).view(1, -1, 1, 1, 3)
    sampled = F.grid_sample(
        volume.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.squeeze(0).squeeze(0).reshape(*coords_unit.shape[:-1])


class ExplicitSpectralGridField(_BaseFieldModel):
    """Explicit multiframe spatial and spectral grid baseline."""

    def __init__(
        self,
        domain: FieldDomain,
        *,
        spatial_resolution: int = 48,
        spectral_resolution: int = 8,
    ) -> None:
        super().__init__(domain)
        if self.coord_mode != "local_tangent":
            raise ValueError("ExplicitSpectralGridField requires local_tangent coordinates")
        self.spatial_resolution = int(spatial_resolution)
        self.spectral_resolution = int(spectral_resolution)
        if self.spatial_resolution < 2 or self.spectral_resolution < 2:
            raise ValueError("grid resolutions must be at least 2")
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
        device = self.grid.device
        dtype = self.grid.dtype
        xs = torch.linspace(
            -1.0, 1.0, steps=self.spatial_resolution, device=device, dtype=dtype
        )
        ys = torch.linspace(
            -1.0, 1.0, steps=self.spatial_resolution, device=device, dtype=dtype
        )
        ls = torch.linspace(
            -1.0, 1.0, steps=self.spectral_resolution, device=device, dtype=dtype
        )
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        wavelengths = float(self.domain.wavelength_center) + ls * float(
            self.domain.wavelength_scale
        )
        cos_dec = math.cos(math.radians(float(self.domain.center_dec_deg)))
        dra = xx * float(self.domain.spatial_scale_deg) / max(cos_dec, 1.0e-6)
        dec = yy * float(self.domain.spatial_scale_deg) + float(
            self.domain.center_dec_deg
        )
        ra = dra + float(self.domain.center_ra_deg)
        return ra, dec, wavelengths

    def initialize_from_product(self, product: MapProductResult) -> None:
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
                    "MapProductSpec band requires center or bounds for initialization"
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
            self.grid.copy_(
                torch.log(
                    torch.clamp(
                        torch.expm1(torch.clamp(init_cube, min=1.0e-4)), min=1.0e-6
                    )
                )
            )

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
        return self._unit_box_coords(ra_deg, dec_deg, wavelength_angstrom).to(
            dtype=self.grid.dtype
        )

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
    def __init__(
        self,
        domain: FieldDomain,
        *,
        num_basis: int = 24,
        init_scale: float = 0.35,
    ) -> None:
        super().__init__(domain)
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

    def forward(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        coords = self._encoded_coords(ra_deg, dec_deg, wavelength_angstrom).to(
            dtype=self.centers.dtype
        )
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
        for module in self.hidden:
            if type(module) is nn.Linear:
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.uniform_(self.output.weight, -1e-2, 1e-2)
        nn.init.constant_(self.output.bias, -3.0)

    def _fourier_features(self, coords: Tensor) -> Tensor:
        if self.num_frequencies <= 0:
            return coords
        scales = torch.logspace(
            0.0,
            math.log2(max(self.max_frequency, 1.0)),
            steps=self.num_frequencies,
            base=2.0,
            device=coords.device,
            dtype=coords.dtype,
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
        if self.num_spatial_frequencies <= 0:
            return spatial
        scales = torch.arange(
            1,
            self.num_spatial_frequencies + 1,
            device=spatial.device,
            dtype=spatial.dtype,
        )
        active = self._active_frequency_count(self.num_spatial_frequencies)
        feats = [spatial]
        for idx, scale in enumerate(scales):
            phase = math.pi * scale * spatial
            sin_phase = torch.sin(phase)
            cos_phase = torch.cos(phase)
            if idx >= active:
                sin_phase = torch.zeros_like(sin_phase)
                cos_phase = torch.zeros_like(cos_phase)
            feats.append(sin_phase)
            feats.append(cos_phase)
        return torch.cat(feats, dim=-1)

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

    def _spectral_basis_bandpass_average(
        self,
        lower_wavelength: float,
        upper_wavelength: float,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        lower = (
            float(lower_wavelength) - float(self.domain.wavelength_center)
        ) / float(self.domain.wavelength_scale)
        upper = (
            float(upper_wavelength) - float(self.domain.wavelength_center)
        ) / float(self.domain.wavelength_scale)
        if upper < lower:
            raise ValueError("bandpass upper wavelength must be >= lower wavelength")
        if upper == lower:
            lam = torch.tensor([lower], device=device, dtype=dtype)
            return self._spectral_basis(lam).squeeze(0).to(dtype=dtype)
        delta = upper - lower
        basis_terms: list[Tensor] = [
            torch.tensor(1.0, device=device, dtype=dtype),
            torch.tensor(0.5 * (lower + upper), device=device, dtype=dtype),
            torch.tensor(
                (lower * lower + lower * upper + upper * upper) / 3.0,
                device=device,
                dtype=dtype,
            ),
        ]
        if self.num_spectral_frequencies > 0:
            for scale_idx in range(1, self.num_spectral_frequencies + 1):
                omega = math.pi * float(scale_idx)
                sin_avg = (math.cos(omega * lower) - math.cos(omega * upper)) / (
                    omega * delta
                )
                cos_avg = (math.sin(omega * upper) - math.sin(omega * lower)) / (
                    omega * delta
                )
                basis_terms.append(torch.tensor(sin_avg, device=device, dtype=dtype))
                basis_terms.append(torch.tensor(cos_avg, device=device, dtype=dtype))
        return torch.stack(basis_terms, dim=0)

    def forward(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
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
        if self.num_spatial_frequencies <= 0:
            return spatial
        scales = torch.arange(
            1,
            self.num_spatial_frequencies + 1,
            device=spatial.device,
            dtype=spatial.dtype,
        )
        scaled = spatial.unsqueeze(-1) * scales
        phase = math.pi * scaled

        sin_feats = torch.sin(phase).transpose(-1, -2)
        cos_feats = torch.cos(phase).transpose(-1, -2)

        stacked = torch.stack([sin_feats, cos_feats], dim=-2)
        flattened = stacked.reshape(*spatial.shape[:-1], -1)

        return torch.cat([spatial, flattened], dim=-1)

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
        wavelength = torch.as_tensor(
            wavelength_angstrom,
            device=ra_deg.device,
            dtype=self.output.weight.dtype,
        )
        if wavelength.ndim == 0:
            wavelengths = wavelength.view(1)
        else:
            wavelengths = wavelength.reshape(-1)
        weights = torch.ones_like(wavelengths)
        return self.render_bandpass(ra_deg, dec_deg, wavelengths, weights)


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
        if self.num_spatial_frequencies <= 0:
            return spatial
        scales = torch.arange(
            1,
            self.num_spatial_frequencies + 1,
            device=spatial.device,
            dtype=spatial.dtype,
        )
        active = self._active_frequency_count(self.num_spatial_frequencies)
        feats = [spatial]
        for idx, scale in enumerate(scales):
            phase = math.pi * scale * spatial
            sin_phase = torch.sin(phase)
            cos_phase = torch.cos(phase)
            if idx >= active:
                sin_phase = torch.zeros_like(sin_phase)
                cos_phase = torch.zeros_like(cos_phase)
            feats.append(sin_phase)
            feats.append(cos_phase)
        return torch.cat(feats, dim=-1)

    def _spectral_features(self, lam: Tensor) -> Tensor:
        return torch.stack([lam, lam * lam, torch.tanh(lam)], dim=-1)

    def forward(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
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


class SkyEncoderBandpassField(_BaseFieldModel):
    def __init__(
        self,
        domain: FieldDomain,
        *,
        hidden_dim: int = 128,
        hidden_layers: int = 3,
        num_fourier_frequencies: int = 8,
        spectral_frequencies: int = 6,
        num_bandpass_frequencies: int = 6,
        bandpass_hidden_dim: int = 48,
        scale_fourier_to_domain: bool = True,
    ) -> None:
        super().__init__(domain)
        self.num_bandpass_frequencies = int(num_bandpass_frequencies)
        self.encoder = make_sky_coordinate_encoder(
            domain,
            num_fourier_frequencies=num_fourier_frequencies,
            spectral_frequencies=spectral_frequencies,
            scale_fourier_to_domain=scale_fourier_to_domain,
        )
        encoded_dim = sky_encoder_output_dim(self.encoder)
        bandpass_in_dim = 4 + 2 * max(self.num_bandpass_frequencies, 0)
        self.bandpass_hidden = nn.Sequential(
            nn.Linear(bandpass_in_dim, bandpass_hidden_dim),
            nn.SiLU(),
            nn.Linear(bandpass_hidden_dim, bandpass_hidden_dim),
            nn.SiLU(),
        )
        layers: list[nn.Module] = []
        current_dim = encoded_dim + bandpass_hidden_dim
        for _ in range(max(int(hidden_layers), 1)):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.SiLU())
            current_dim = hidden_dim
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(current_dim, 1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.bandpass_hidden:
            if type(module) is nn.Linear:
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)
        reset_silu_mlp_linears(self.hidden)
        reset_logflux_readout_linear(self.output)

    def _bandpass_features(self, wavelengths: Tensor, weights: Tensor) -> Tensor:
        lam = (
            wavelengths.to(dtype=self.hidden[0].weight.dtype)
            - float(self.domain.wavelength_center)
        ) / float(self.domain.wavelength_scale)
        weights = weights.to(dtype=self.hidden[0].weight.dtype)
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

    def _encoded_planes(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelengths: Tensor,
    ) -> Tensor:
        encoded = []
        ra64 = ra_deg.to(dtype=torch.float64)
        dec64 = dec_deg.to(dtype=torch.float64)
        for wavelength in wavelengths:
            wavelength_grid = torch.full_like(
                ra64, float(wavelength.item()), dtype=torch.float64
            )
            encoded.append(
                self.encoder(
                    ra64,
                    dec64,
                    wavelength=wavelength_grid,
                ).to(dtype=self.output.weight.dtype)
            )
        return torch.stack(encoded, dim=0)

    def render_bandpass(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelengths: Tensor,
        weights: Tensor,
    ) -> Tensor:
        encoded = self._encoded_planes(ra_deg, dec_deg, wavelengths)
        norm_weights = weights.to(dtype=encoded.dtype)
        norm_weights = norm_weights / torch.clamp(norm_weights.sum(), min=1.0e-8)
        bandpass_latent = self.bandpass_hidden(
            self._bandpass_features(wavelengths, weights)
        )
        bandpass_latent = bandpass_latent.view(1, *([1] * (encoded.ndim - 2)), -1)
        bandpass_latent = bandpass_latent.expand(
            encoded.shape[0], *encoded.shape[1:-1], -1
        )
        hidden_planes = self.hidden(torch.cat([encoded, bandpass_latent], dim=-1))
        mixed_hidden = torch.sum(
            hidden_planes * norm_weights.view(-1, *([1] * (hidden_planes.ndim - 1))),
            dim=0,
        )
        out = self.output(mixed_hidden).squeeze(-1)
        return F.softplus(out) + 1.0e-4

    def forward(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        wavelength = torch.as_tensor(
            wavelength_angstrom,
            device=ra_deg.device,
            dtype=torch.float32,
        )
        if wavelength.ndim == 0:
            wavelengths = wavelength.view(1)
        else:
            wavelengths = wavelength.reshape(-1)
        weights = torch.ones_like(wavelengths)
        return self.render_bandpass(ra_deg, dec_deg, wavelengths, weights)


class ConditionedSpectralField(_BaseFieldModel):
    def __init__(
        self,
        domain: FieldDomain,
        *,
        hidden_dim: int = 96,
        hidden_layers: int = 3,
        num_spatial_frequencies: int = 6,
        num_spectral_frequencies: int = 6,
        spectral_hidden_dim: int = 48,
    ) -> None:
        super().__init__(domain)
        self.hidden_dim = int(hidden_dim)
        self.hidden_layers = max(int(hidden_layers), 1)
        self.num_spatial_frequencies = int(num_spatial_frequencies)
        self.num_spectral_frequencies = int(num_spectral_frequencies)
        self.spatial_dim = 2 if self.coord_mode == "local_tangent" else 3
        in_dim = self.spatial_dim * (1 + 2 * max(self.num_spatial_frequencies, 0))
        self.input = nn.Linear(in_dim, self.hidden_dim)
        self.hidden = nn.ModuleList(
            nn.Linear(self.hidden_dim, self.hidden_dim)
            for _ in range(self.hidden_layers - 1)
        )
        spectral_in_dim = 3 + 2 * max(self.num_spectral_frequencies, 0)
        self.spectral_mlp = nn.Sequential(
            nn.Linear(spectral_in_dim, spectral_hidden_dim),
            nn.SiLU(),
            nn.Linear(
                spectral_hidden_dim,
                2 * self.hidden_dim * self.hidden_layers,
            ),
        )
        self.output = nn.Linear(self.hidden_dim, 1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input.weight)
        nn.init.zeros_(self.input.bias)
        for layer in self.hidden:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        for module in self.spectral_mlp:
            if type(module) is nn.Linear:
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.uniform_(self.output.weight, -1e-2, 1e-2)
        nn.init.constant_(self.output.bias, -3.0)

    def _spatial_fourier_features(self, spatial: Tensor) -> Tensor:
        if self.num_spatial_frequencies <= 0:
            return spatial
        scales = torch.arange(
            1,
            self.num_spatial_frequencies + 1,
            device=spatial.device,
            dtype=spatial.dtype,
        )
        scaled = spatial.unsqueeze(-1) * scales
        phase = math.pi * scaled

        sin_feats = torch.sin(phase).transpose(-1, -2)
        cos_feats = torch.cos(phase).transpose(-1, -2)

        stacked = torch.stack([sin_feats, cos_feats], dim=-2)
        flattened = stacked.reshape(*spatial.shape[:-1], -1)

        return torch.cat([spatial, flattened], dim=-1)

    def _spectral_features(self, lam: Tensor) -> Tensor:
        if self.num_spectral_frequencies > 0:
            scales = torch.arange(
                1,
                self.num_spectral_frequencies + 1,
                device=lam.device,
                dtype=lam.dtype,
            )
            scaled = lam.unsqueeze(-1) * scales
            phase = math.pi * scaled

            sin_feats = torch.sin(phase)
            cos_feats = torch.cos(phase)

            stacked_fourier = torch.stack([sin_feats, cos_feats], dim=-1).view(
                *lam.shape, -1
            )
            base_feats = torch.stack([lam, lam * lam, torch.tanh(lam)], dim=-1)
            return torch.cat([base_feats, stacked_fourier], dim=-1)
        else:
            return torch.stack([lam, lam * lam, torch.tanh(lam)], dim=-1)

    def _film_params(self, lam: Tensor) -> tuple[list[Tensor], list[Tensor]]:
        spec = self._spectral_features(lam)
        params = self.spectral_mlp(spec)
        params = params.view(*lam.shape, self.hidden_layers, 2, self.hidden_dim)
        gammas = [
            params[..., layer_idx, 0, :] for layer_idx in range(self.hidden_layers)
        ]
        betas = [
            params[..., layer_idx, 1, :] for layer_idx in range(self.hidden_layers)
        ]
        return gammas, betas

    def forward(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        lam = self._normalized_wavelength(
            wavelength_angstrom,
            device=ra_deg.device,
            dtype=self.output.weight.dtype,
            shape=ra_deg.shape,
        )
        spatial = self._spatial_coords(ra_deg, dec_deg, wavelength_angstrom).to(
            dtype=self.output.weight.dtype
        )
        x = self._spatial_fourier_features(spatial)
        gammas, betas = self._film_params(lam)
        x = self.input(x)
        x = F.silu(x * (1.0 + gammas[0]) + betas[0])
        for layer_idx, layer in enumerate(self.hidden, start=1):
            x = layer(x)
            x = F.silu(x * (1.0 + gammas[layer_idx]) + betas[layer_idx])
        out = self.output(x).squeeze(-1)
        return F.softplus(out) + 1.0e-4


class HashGridConditionedField(_BaseFieldModel):
    def __init__(
        self,
        domain: FieldDomain,
        *,
        hidden_dim: int = 96,
        hidden_layers: int = 3,
        num_levels: int = 6,
        features_per_level: int = 4,
        base_resolution: int = 8,
        finest_resolution: int = 128,
        num_spectral_frequencies: int = 8,
        log2_hashmap_size: int | Sequence[int] = 12,
        hash_interpolation: Literal["linear", "smoothstep"] = "linear",
        hash_dense_coarse: bool = True,
        hash_max_dense_cells: int = 4096,
        hash_level_gain: bool = False,
    ) -> None:
        super().__init__(domain)
        self.spatial_dim = 2 if self.coord_mode == "local_tangent" else 3
        self.num_spectral_frequencies = int(num_spectral_frequencies)
        self.encoder = HashGridEncoder(
            self.spatial_dim,
            num_levels=num_levels,
            features_per_level=features_per_level,
            base_resolution=base_resolution,
            finest_resolution=finest_resolution,
            log2_hashmap_size=log2_hashmap_size,
            interpolation=hash_interpolation,
            dense_coarse=hash_dense_coarse,
            max_dense_cells=hash_max_dense_cells,
            learnable_level_gain=hash_level_gain,
        )
        spectral_dim = 3 + 2 * max(self.num_spectral_frequencies, 0)
        in_dim = self.spatial_dim + num_levels * features_per_level + spectral_dim
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
        for module in self.hidden:
            if type(module) is nn.Linear:
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.uniform_(self.output.weight, -1e-2, 1e-2)
        nn.init.constant_(self.output.bias, -3.0)

    def _spectral_features(self, lam: Tensor) -> Tensor:
        if self.num_spectral_frequencies > 0:
            scales = torch.arange(
                1,
                self.num_spectral_frequencies + 1,
                device=lam.device,
                dtype=lam.dtype,
            )
            scaled = lam.unsqueeze(-1) * scales
            phase = math.pi * scaled

            sin_feats = torch.sin(phase)
            cos_feats = torch.cos(phase)

            stacked_fourier = torch.stack([sin_feats, cos_feats], dim=-1).view(
                *lam.shape, -1
            )
            base_feats = torch.stack([lam, lam * lam, torch.tanh(lam)], dim=-1)
            return torch.cat([base_feats, stacked_fourier], dim=-1)
        else:
            return torch.stack([lam, lam * lam, torch.tanh(lam)], dim=-1)

    def forward(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        spatial_raw = self._spatial_coords(
            ra_deg,
            dec_deg,
            wavelength_angstrom,
        ).to(dtype=self.output.weight.dtype)
        spatial_unit = self._spatial_unit_box_coords(
            ra_deg,
            dec_deg,
            wavelength_angstrom,
        ).to(dtype=self.output.weight.dtype)
        lam = self._normalized_wavelength(
            wavelength_angstrom,
            device=ra_deg.device,
            dtype=self.output.weight.dtype,
            shape=ra_deg.shape,
        )
        spectral = self._spectral_features(lam)
        grid_feats = self.encoder(spatial_unit)
        feats = torch.cat([spatial_raw, grid_feats, spectral], dim=-1)
        out = self.output(self.hidden(feats)).squeeze(-1)
        return F.softplus(out) + 1.0e-4


class SkyEncoderMLPField(_BaseFieldModel):
    def __init__(
        self,
        domain: FieldDomain,
        *,
        hidden_dim: int = 96,
        hidden_layers: int = 3,
        num_fourier_frequencies: int = 8,
        spectral_frequencies: int = 5,
        scale_fourier_to_domain: bool = True,
    ) -> None:
        super().__init__(domain)
        self.encoder = make_sky_coordinate_encoder(
            domain,
            num_fourier_frequencies=num_fourier_frequencies,
            spectral_frequencies=spectral_frequencies,
            scale_fourier_to_domain=scale_fourier_to_domain,
        )
        in_dim = sky_encoder_output_dim(self.encoder)
        self.hidden = silu_mlp_sequential(in_dim, hidden_dim, hidden_layers)
        self.output = nn.Linear(hidden_dim, 1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        reset_silu_mlp_linears(self.hidden)
        reset_logflux_readout_linear(self.output)

    def forward(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        wavelength = torch.as_tensor(
            wavelength_angstrom,
            device=ra_deg.device,
            dtype=torch.float64,
        )
        if wavelength.shape != ra_deg.shape:
            wavelength = torch.broadcast_to(wavelength, ra_deg.shape)
        feats = self.encoder(
            ra_deg.to(dtype=torch.float64),
            dec_deg.to(dtype=torch.float64),
            wavelength=wavelength,
        )
        output_dtype = self.output.weight.dtype
        feats = feats.to(dtype=output_dtype)
        out = self.output(self.hidden(feats)).squeeze(-1)
        return F.softplus(out) + 1.0e-4


class StrainerSkyEncoderMLPField(_BaseFieldModel):
    """Sky encoder with explicit trunk (`encoder` + `body`) and `decoder_head` for transfer learning.

    Matches ``SkyEncoderMLPField`` capacity; adds ``reset_decoder_head``, trunk state-dict helpers,
    and an ``output`` property alias pointing at ``decoder_head`` for noise-pretrain compatibility.
    """

    def __init__(
        self,
        domain: FieldDomain,
        *,
        hidden_dim: int = 96,
        hidden_layers: int = 3,
        num_fourier_frequencies: int = 8,
        spectral_frequencies: int = 5,
        scale_fourier_to_domain: bool = True,
    ) -> None:
        super().__init__(domain)
        self.encoder = make_sky_coordinate_encoder(
            domain,
            num_fourier_frequencies=num_fourier_frequencies,
            spectral_frequencies=spectral_frequencies,
            scale_fourier_to_domain=scale_fourier_to_domain,
        )
        in_dim = sky_encoder_output_dim(self.encoder)
        self.body = silu_mlp_sequential(in_dim, hidden_dim, hidden_layers)
        self.decoder_head = nn.Linear(hidden_dim, 1)
        self._reset_parameters()

    @property
    def output(self) -> nn.Linear:
        return self.decoder_head

    def _reset_parameters(self) -> None:
        reset_silu_mlp_linears(self.body)
        reset_logflux_readout_linear(self.decoder_head)

    def reset_decoder_head(self, *, bias_init: float = -3.0) -> None:
        nn.init.uniform_(self.decoder_head.weight, -1e-2, 1e-2)
        nn.init.constant_(self.decoder_head.bias, float(bias_init))

    def trunk_state_dict(self) -> dict[str, Tensor]:
        state = self.state_dict()
        return {
            key: value
            for key, value in state.items()
            if key.startswith("encoder.") or key.startswith("body.")
        }

    def load_trunk_state_dict(
        self,
        state_dict: dict[str, Tensor],
        *,
        strict: bool = True,
    ):
        incompatible = self.load_state_dict(state_dict, strict=False)
        if strict:
            for key in state_dict:
                if key in incompatible.missing_keys:
                    raise RuntimeError(f"failed to load trunk tensor for key {key!r}")
        return incompatible

    def forward(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        wavelength = torch.as_tensor(
            wavelength_angstrom,
            device=ra_deg.device,
            dtype=torch.float64,
        )
        if wavelength.shape != ra_deg.shape:
            wavelength = torch.broadcast_to(wavelength, ra_deg.shape)
        feats = self.encoder(
            ra_deg.to(dtype=torch.float64),
            dec_deg.to(dtype=torch.float64),
            wavelength=wavelength,
        )
        output_dtype = self.decoder_head.weight.dtype
        feats = feats.to(dtype=output_dtype)
        out = self.decoder_head(self.body(feats)).squeeze(-1)
        return F.softplus(out) + 1.0e-4


def save_strainer_trunk_checkpoint(
    model: nn.Module,
    path: str,
    *,
    meta: dict[str, object] | None = None,
) -> None:
    trunk_fn = getattr(model, "trunk_state_dict", None)
    if not callable(trunk_fn):
        raise TypeError(
            "model must implement trunk_state_dict() (e.g. StrainerSkyEncoderMLPField)"
        )
    payload: dict[str, object] = {
        "format": _STRAINER_TRUNK_FORMAT,
        "trunk": {k: v.detach().cpu() for k, v in trunk_fn().items()},
        "meta": dict(meta) if meta else {},
    }
    torch.save(payload, path)


_STRAINER_TRUNK_FORMAT = "torchsky.strainer_trunk.v1"
STRAINER_TRUNK_CHECKPOINT_FORMAT = _STRAINER_TRUNK_FORMAT


def load_strainer_trunk_checkpoint(
    model: nn.Module,
    path: str,
    *,
    map_location: torch.device | str | None = None,
    strict: bool = True,
    require_known_format: bool = True,
) -> nn.Module:
    load_fn = getattr(model, "load_trunk_state_dict", None)
    if not callable(load_fn):
        raise TypeError(
            "model must implement load_trunk_state_dict() (e.g. StrainerSkyEncoderMLPField)"
        )
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or "trunk" not in payload:
        raise ValueError("checkpoint must be a dict with a 'trunk' key")
    fmt = payload.get("format")
    if require_known_format and fmt is not None and fmt != _STRAINER_TRUNK_FORMAT:
        raise ValueError(
            f"unexpected strainer trunk format {fmt!r}; expected {_STRAINER_TRUNK_FORMAT!r}"
        )
    trunk = payload["trunk"]
    if not isinstance(trunk, dict):
        raise ValueError("checkpoint['trunk'] must be a state dict")
    load_fn(trunk, strict=strict)
    return model


def pretrain_strainer_trunk_synthetic(
    model: nn.Module,
    *,
    steps: int,
    lr: float,
    noise_kind: FieldNoisePretrainKind = "gaussian",
    noise_kinds: Sequence[FieldNoisePretrainKind] | None = None,
    batch_size: int = 4096,
    num_terms: int = 48,
    max_frequency: float = 8.0,
    seed: int = 42,
) -> dict[str, Any] | None:
    if not isinstance(model, _BaseFieldModel):
        raise TypeError("model must be a torchsky map field _BaseFieldModel subclass")
    steps = max(int(steps), 0)
    if steps == 0:
        return None
    if noise_kinds is not None:
        kinds = tuple(k for k in noise_kinds if k != "none")
        if not kinds:
            raise ValueError(
                "noise_kinds must include at least one kind other than 'none'"
            )
    else:
        if noise_kind == "none":
            raise ValueError(
                "noise_kind must not be 'none' for synthetic trunk pretraining"
            )
        kinds = (noise_kind,)
    domain = model.domain
    device = next((p.device for p in model.parameters()), default_compute_device())
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("pretrain_strainer_trunk_synthetic: no trainable parameters")
    optimizer = torch.optim.Adam(trainable, lr=float(lr))
    model.train()
    training_gen = torch.Generator(device="cpu")
    training_gen.manual_seed(int(seed))
    noise_gen = torch.Generator(device="cpu")
    noise_gen.manual_seed(int(seed) + 17)
    batch = max(int(batch_size), 1)
    num_kinds = len(kinds)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    for step_idx in range(steps):
        step_noise = kinds[step_idx % num_kinds]
        optimizer.zero_grad(set_to_none=True)
        dra = (
            torch.rand(
                batch, generator=training_gen, dtype=torch.float64, device=device
            )
            - 0.5
        ) * (2.0 * float(domain.spatial_scale_deg))
        ddec = (
            torch.rand(
                batch, generator=training_gen, dtype=torch.float64, device=device
            )
            - 0.5
        ) * (2.0 * float(domain.spatial_scale_deg))
        ra_deg = torch.remainder(float(domain.center_ra_deg) + dra, 360.0)
        dec_deg = torch.clamp(
            float(domain.center_dec_deg) + ddec,
            min=-90.0,
            max=90.0,
        )
        lam_offset = (
            torch.rand(
                batch, generator=training_gen, dtype=torch.float64, device=device
            )
            - 0.5
        ) * (2.0 * float(domain.wavelength_scale))
        wavelength = (
            torch.as_tensor(float(domain.wavelength_center), device=device) + lam_offset
        )
        coords_unit = model._unit_box_coords(ra_deg, dec_deg, wavelength)
        target = _synthetic_noise_target(
            coords_unit,
            noise_kind=step_noise,
            num_terms=num_terms,
            max_frequency=max_frequency,
            generator=noise_gen,
        ).to(device=device, dtype=torch.float32)
        pred = model(
            ra_deg.to(dtype=torch.float32),
            dec_deg.to(dtype=torch.float32),
            wavelength.to(dtype=torch.float32),
        )
        # ⚡ Bolt: Element-wise multiplication is faster than torch.square or **2
        diff_pt = pred - target
        loss = torch.mean(diff_pt * diff_pt)
        loss.backward()
        optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    duration = end_time - start_time

    max_mem = 0
    if device.type == "cuda":
        max_mem = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

    step_duration_mean_ms = (duration / steps * 1000.0) if steps > 0 else 0.0
    throughput = (steps * batch) / duration if duration > 0 else 0.0

    return {
        "total_duration_sec": duration,
        "step_duration_mean_ms": step_duration_mean_ms,
        "throughput_samples_per_sec": throughput,
        "max_memory_allocated_bytes": max_mem,
    }


class SkyEncoderResMLPField(_BaseFieldModel):
    def __init__(
        self,
        domain: FieldDomain,
        *,
        hidden_dim: int = 128,
        num_blocks: int = 4,
        num_fourier_frequencies: int = 8,
        spectral_frequencies: int = 6,
        multiscale_readout: bool = False,
        readout_hidden_dim: int | None = None,
        scale_fourier_to_domain: bool = True,
    ) -> None:
        super().__init__(domain)
        self.multiscale_readout = bool(multiscale_readout)
        self.encoder = make_sky_coordinate_encoder(
            domain,
            num_fourier_frequencies=num_fourier_frequencies,
            spectral_frequencies=spectral_frequencies,
            scale_fourier_to_domain=scale_fourier_to_domain,
        )
        in_dim = sky_encoder_output_dim(self.encoder)
        self.input = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(hidden_dim) for _ in range(max(int(num_blocks), 1))]
        )
        if self.multiscale_readout:
            stacked_dim = hidden_dim * (1 + len(self.blocks))
            readout_width = (
                int(readout_hidden_dim)
                if readout_hidden_dim is not None
                else max(hidden_dim // 2, 64)
            )
            self.output = nn.Sequential(
                nn.Linear(stacked_dim, readout_width),
                nn.SiLU(),
                nn.Linear(readout_width, 1),
            )
        else:
            self.output = nn.Linear(hidden_dim, 1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.input.weight, nonlinearity="relu")
        nn.init.zeros_(self.input.bias)
        if isinstance(self.output, nn.Sequential):
            for module in self.output:
                if type(module) is nn.Linear:
                    nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                    nn.init.zeros_(module.bias)
            final = self.output[-1]
            assert isinstance(final, nn.Linear)
            nn.init.uniform_(final.weight, -1e-2, 1e-2)
            nn.init.constant_(final.bias, 0.0)
        else:
            nn.init.uniform_(self.output.weight, -1e-2, 1e-2)
            nn.init.constant_(self.output.bias, -3.0)

    def forward(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        wavelength = torch.as_tensor(
            wavelength_angstrom,
            device=ra_deg.device,
            dtype=torch.float64,
        )
        if wavelength.shape != ra_deg.shape:
            wavelength = torch.broadcast_to(wavelength, ra_deg.shape)
        feats = self.encoder(
            ra_deg.to(dtype=torch.float64),
            dec_deg.to(dtype=torch.float64),
            wavelength=wavelength,
        )
        output_dtype = (
            self.output[0].weight.dtype
            if isinstance(self.output, nn.Sequential)
            else self.output.weight.dtype
        )
        feats = feats.to(dtype=output_dtype)
        x = F.silu(self.input(feats))
        readout_states = [x] if self.multiscale_readout else None
        for block in self.blocks:
            x = block(x)
            if readout_states is not None:
                readout_states.append(x)
        if readout_states is not None:
            x = torch.cat(readout_states, dim=-1)
        out = self.output(x).squeeze(-1)
        return F.softplus(out) + 1.0e-4


class HashGridMLPField(_BaseFieldModel):
    def __init__(
        self,
        domain: FieldDomain,
        *,
        hidden_dim: int = 64,
        hidden_layers: int = 2,
        num_levels: int = 6,
        features_per_level: int = 4,
        base_resolution: int = 8,
        finest_resolution: int = 128,
        log2_hashmap_size: int | Sequence[int] = 12,
        hash_interpolation: Literal["linear", "smoothstep"] = "linear",
        hash_dense_coarse: bool = True,
        hash_max_dense_cells: int = 4096,
        hash_level_gain: bool = False,
    ) -> None:
        super().__init__(domain)
        self.encoder = HashGridEncoder(
            self.coord_dim,
            num_levels=num_levels,
            features_per_level=features_per_level,
            base_resolution=base_resolution,
            finest_resolution=finest_resolution,
            log2_hashmap_size=log2_hashmap_size,
            interpolation=hash_interpolation,
            dense_coarse=hash_dense_coarse,
            max_dense_cells=hash_max_dense_cells,
            learnable_level_gain=hash_level_gain,
        )
        in_dim = self.coord_dim + num_levels * features_per_level
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
        for module in self.hidden:
            if type(module) is nn.Linear:
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.uniform_(self.output.weight, -1e-2, 1e-2)
        nn.init.constant_(self.output.bias, -3.0)

    def forward(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        coords_unit = self._unit_box_coords(ra_deg, dec_deg, wavelength_angstrom)
        grid_feats = self.encoder(coords_unit.to(dtype=self.output.weight.dtype))
        raw = coords_unit.to(dtype=self.output.weight.dtype)
        feats = torch.cat([raw, grid_feats], dim=-1)
        out = self.output(self.hidden(feats)).squeeze(-1)
        return F.softplus(out) + 1.0e-4


class HashGridResMLPField(_BaseFieldModel):
    """Hash-grid spatial encoder + residual MLP decoder (explicit features + refinement)."""

    def __init__(
        self,
        domain: FieldDomain,
        *,
        hidden_dim: int = 96,
        num_res_blocks: int = 3,
        num_levels: int = 6,
        features_per_level: int = 4,
        base_resolution: int = 8,
        finest_resolution: int = 128,
        log2_hashmap_size: int | Sequence[int] = 12,
        hash_interpolation: Literal["linear", "smoothstep"] = "linear",
        hash_dense_coarse: bool = True,
        hash_max_dense_cells: int = 4096,
        hash_level_gain: bool = False,
    ) -> None:
        super().__init__(domain)
        self.encoder = HashGridEncoder(
            self.coord_dim,
            num_levels=num_levels,
            features_per_level=features_per_level,
            base_resolution=base_resolution,
            finest_resolution=finest_resolution,
            log2_hashmap_size=log2_hashmap_size,
            interpolation=hash_interpolation,
            dense_coarse=hash_dense_coarse,
            max_dense_cells=hash_max_dense_cells,
            learnable_level_gain=hash_level_gain,
        )
        in_dim = self.coord_dim + num_levels * features_per_level
        self.input = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(hidden_dim) for _ in range(max(int(num_res_blocks), 1))]
        )
        self.output = nn.Linear(hidden_dim, 1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input.weight)
        nn.init.zeros_(self.input.bias)
        nn.init.uniform_(self.output.weight, -1e-2, 1e-2)
        nn.init.constant_(self.output.bias, -3.0)

    def forward(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        coords_unit = self._unit_box_coords(ra_deg, dec_deg, wavelength_angstrom)
        grid_feats = self.encoder(coords_unit.to(dtype=self.output.weight.dtype))
        raw = coords_unit.to(dtype=self.output.weight.dtype)
        feats = torch.cat([raw, grid_feats], dim=-1)
        x = F.silu(self.input(feats))
        for block in self.blocks:
            x = block(x)
        out = self.output(x).squeeze(-1)
        return F.softplus(out) + 1.0e-4


class HashGridSpectralField(_BaseFieldModel):
    def __init__(
        self,
        domain: FieldDomain,
        *,
        hidden_dim: int = 64,
        hidden_layers: int = 2,
        num_levels: int = 6,
        features_per_level: int = 4,
        base_resolution: int = 8,
        finest_resolution: int = 128,
        num_spectral_frequencies: int = 4,
        log2_hashmap_size: int | Sequence[int] = 12,
        hash_interpolation: Literal["linear", "smoothstep"] = "linear",
        hash_dense_coarse: bool = True,
        hash_max_dense_cells: int = 4096,
        hash_level_gain: bool = False,
    ) -> None:
        super().__init__(domain)
        self.spatial_dim = 2 if self.coord_mode == "local_tangent" else 3
        self.num_spectral_frequencies = int(num_spectral_frequencies)
        self.basis_dim = 3 + 2 * max(self.num_spectral_frequencies, 0)
        self.encoder = HashGridEncoder(
            self.spatial_dim,
            num_levels=num_levels,
            features_per_level=features_per_level,
            base_resolution=base_resolution,
            finest_resolution=finest_resolution,
            log2_hashmap_size=log2_hashmap_size,
            interpolation=hash_interpolation,
            dense_coarse=hash_dense_coarse,
            max_dense_cells=hash_max_dense_cells,
            learnable_level_gain=hash_level_gain,
        )
        in_dim = self.spatial_dim + num_levels * features_per_level
        layers: list[nn.Module] = []
        current_dim = in_dim
        for _ in range(max(int(hidden_layers), 1)):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.SiLU())
            current_dim = hidden_dim
        self.hidden = nn.Sequential(*layers)
        self.coeff_head = nn.Linear(current_dim, self.basis_dim)
        self.bias_head = nn.Linear(current_dim, 1)
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

    def _spectral_basis(self, lam: Tensor) -> Tensor:
        if self.num_spectral_frequencies > 0:
            scales = torch.arange(
                1,
                self.num_spectral_frequencies + 1,
                device=lam.device,
                dtype=lam.dtype,
            )
            scaled = lam.unsqueeze(-1) * scales
            phase = math.pi * scaled

            sin_feats = torch.sin(phase)
            cos_feats = torch.cos(phase)

            stacked_fourier = torch.stack([sin_feats, cos_feats], dim=-1).view(
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
        spatial_unit = self._spatial_unit_box_coords(
            ra_deg,
            dec_deg,
            wavelength_angstrom,
        ).to(dtype=self.coeff_head.weight.dtype)
        spatial_raw = self._spatial_coords(
            ra_deg,
            dec_deg,
            wavelength_angstrom,
        ).to(dtype=self.coeff_head.weight.dtype)
        grid_feats = self.encoder(spatial_unit)
        hidden = self.hidden(torch.cat([spatial_raw, grid_feats], dim=-1))
        coeff = self.coeff_head(hidden)
        bias = self.bias_head(hidden).squeeze(-1)
        lam = self._normalized_wavelength(
            wavelength_angstrom,
            device=ra_deg.device,
            dtype=self.coeff_head.weight.dtype,
            shape=ra_deg.shape,
        )
        spectral_basis = self._spectral_basis(lam)
        out = bias + torch.sum(coeff * spectral_basis, dim=-1)
        return F.softplus(out) + 1.0e-4


class SkyLambdaPlanesField(_BaseFieldModel):
    def __init__(
        self,
        domain: FieldDomain,
        *,
        plane_resolution: int = 32,
        features_per_plane: int = 8,
        hidden_dim: int = 64,
        hidden_layers: int = 2,
    ) -> None:
        super().__init__(domain)
        self.spatial_dim = 2 if self.coord_mode == "local_tangent" else 3
        self.plane_resolution = int(plane_resolution)
        self.features_per_plane = int(features_per_plane)
        plane_names = ["xy", "xl", "yl"]
        if self.spatial_dim == 3:
            plane_names.extend(["xz", "yz", "zl"])
        self.planes = nn.ParameterDict(
            {
                name: nn.Parameter(
                    1.0e-3
                    * torch.randn(
                        self.features_per_plane,
                        self.plane_resolution,
                        self.plane_resolution,
                    )
                )
                for name in plane_names
            }
        )
        in_dim = len(plane_names) * self.features_per_plane + self.spatial_dim + 1
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
        for module in self.hidden:
            if type(module) is nn.Linear:
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.uniform_(self.output.weight, -1e-2, 1e-2)
        nn.init.constant_(self.output.bias, -3.0)

    def forward(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        spatial_raw = self._spatial_coords(
            ra_deg,
            dec_deg,
            wavelength_angstrom,
        ).to(dtype=self.output.weight.dtype)
        spatial_unit = self._spatial_unit_box_coords(
            ra_deg,
            dec_deg,
            wavelength_angstrom,
        ).to(dtype=self.output.weight.dtype)
        lam = self._normalized_wavelength(
            wavelength_angstrom,
            device=ra_deg.device,
            dtype=self.output.weight.dtype,
            shape=ra_deg.shape,
        )
        lam_unit = 0.5 * (torch.tanh(lam) + 1.0)
        x = spatial_unit[..., 0]
        y = spatial_unit[..., 1]
        sampled = [
            sample_feature_plane(self.planes["xy"], x, y),
            sample_feature_plane(self.planes["xl"], x, lam_unit),
            sample_feature_plane(self.planes["yl"], y, lam_unit),
        ]
        if self.spatial_dim == 3:
            z = spatial_unit[..., 2]
            sampled.extend(
                [
                    sample_feature_plane(self.planes["xz"], x, z),
                    sample_feature_plane(self.planes["yz"], y, z),
                    sample_feature_plane(self.planes["zl"], z, lam_unit),
                ]
            )
        plane_feats = torch.cat(sampled, dim=-1)
        feats = torch.cat([spatial_raw, lam.unsqueeze(-1), plane_feats], dim=-1)
        out = self.output(self.hidden(feats)).squeeze(-1)
        return F.softplus(out) + 1.0e-4


class SkyLambdaKPlanesField(_BaseFieldModel):
    def __init__(
        self,
        domain: FieldDomain,
        *,
        base_resolution: int = 16,
        num_scales: int = 3,
        features_per_plane: int = 8,
        hidden_dim: int = 64,
        hidden_layers: int = 2,
        refine_hidden_dim: int = 32,
    ) -> None:
        super().__init__(domain)
        self.spatial_dim = 2 if self.coord_mode == "local_tangent" else 3
        self.base_resolution = int(base_resolution)
        self.num_scales = int(num_scales)
        self.features_per_plane = int(features_per_plane)
        plane_names = ["xy", "xl", "yl"]
        if self.spatial_dim == 3:
            plane_names.extend(["xz", "yz", "zl"])
        self._plane_names = tuple(plane_names)
        self.planes = nn.ParameterDict()
        for scale_idx in range(self.num_scales):
            resolution = self.base_resolution * (2**scale_idx)
            for name in self._plane_names:
                self.planes[f"s{scale_idx}_{name}"] = nn.Parameter(
                    1.0e-3
                    * torch.randn(
                        self.features_per_plane,
                        resolution,
                        resolution,
                    )
                )
        in_dim = (
            len(self._plane_names) * self.num_scales * self.features_per_plane
            + self.spatial_dim
            + 1
        )
        layers: list[nn.Module] = []
        current_dim = in_dim
        for _ in range(max(int(hidden_layers), 1)):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.SiLU())
            current_dim = hidden_dim
        self.hidden = nn.Sequential(*layers)
        self.base_output = nn.Linear(current_dim, 1)
        self.refine = nn.Sequential(
            nn.Linear(current_dim, refine_hidden_dim),
            nn.SiLU(),
            nn.Linear(refine_hidden_dim, 1),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.hidden:
            if type(module) is nn.Linear:
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.uniform_(self.base_output.weight, -1e-2, 1e-2)
        nn.init.constant_(self.base_output.bias, -3.0)
        for module in self.refine:
            if type(module) is nn.Linear:
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def _hidden_features(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> tuple[Tensor, Tensor]:
        spatial_raw = self._spatial_coords(
            ra_deg,
            dec_deg,
            wavelength_angstrom,
        ).to(dtype=self.base_output.weight.dtype)
        spatial_unit = self._spatial_unit_box_coords(
            ra_deg,
            dec_deg,
            wavelength_angstrom,
        ).to(dtype=self.base_output.weight.dtype)
        lam = self._normalized_wavelength(
            wavelength_angstrom,
            device=ra_deg.device,
            dtype=self.base_output.weight.dtype,
            shape=ra_deg.shape,
        )
        lam_unit = 0.5 * (torch.tanh(lam) + 1.0)
        x = spatial_unit[..., 0]
        y = spatial_unit[..., 1]
        sampled: list[Tensor] = []
        for scale_idx in range(self.num_scales):
            sampled.extend(
                [
                    sample_feature_plane(self.planes[f"s{scale_idx}_xy"], x, y),
                    sample_feature_plane(self.planes[f"s{scale_idx}_xl"], x, lam_unit),
                    sample_feature_plane(self.planes[f"s{scale_idx}_yl"], y, lam_unit),
                ]
            )
            if self.spatial_dim == 3:
                z = spatial_unit[..., 2]
                sampled.extend(
                    [
                        sample_feature_plane(self.planes[f"s{scale_idx}_xz"], x, z),
                        sample_feature_plane(self.planes[f"s{scale_idx}_yz"], y, z),
                        sample_feature_plane(
                            self.planes[f"s{scale_idx}_zl"], z, lam_unit
                        ),
                    ]
                )
        plane_feats = torch.cat(sampled, dim=-1)
        feats = torch.cat([spatial_raw, lam.unsqueeze(-1), plane_feats], dim=-1)
        hidden = self.hidden(feats)
        return hidden, lam

    def _decode_hidden(
        self,
        hidden: Tensor,
    ) -> Tensor:
        out = self.base_output(hidden).squeeze(-1) + self.refine(hidden).squeeze(-1)
        return F.softplus(out) + 1.0e-4

    def render_bandpass(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelengths: Tensor,
        weights: Tensor,
    ) -> Tensor:
        hidden_planes = []
        for wavelength in wavelengths:
            hidden, _ = self._hidden_features(ra_deg, dec_deg, wavelength)
            hidden_planes.append(hidden)
        stacked_hidden = torch.stack(hidden_planes, dim=0)
        norm_weights = weights.to(dtype=stacked_hidden.dtype)
        norm_weights = norm_weights / torch.clamp(norm_weights.sum(), min=1.0e-8)
        mixed_hidden = torch.sum(
            stacked_hidden * norm_weights.view(-1, *([1] * (stacked_hidden.ndim - 1))),
            dim=0,
        )
        return self._decode_hidden(mixed_hidden)

    def forward(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        hidden, _ = self._hidden_features(
            ra_deg,
            dec_deg,
            wavelength_angstrom,
        )
        return self._decode_hidden(hidden)


class SkyLambdaFeatureField(_BaseFieldModel):
    def __init__(
        self,
        domain: FieldDomain,
        *,
        base_resolution: int = 16,
        num_scales: int = 3,
        features_per_plane: int = 8,
        scale_hidden_dim: int = 64,
        scale_num_blocks: int = 2,
        decoder_hidden_dim: int = 64,
        decoder_num_blocks: int = 2,
        refine_hidden_dim: int = 32,
        bandpass_hidden_dim: int = 32,
        global_hidden_dim: int = 0,
        global_num_blocks: int = 1,
        aggregate_hidden_moments: bool = False,
        coarse_fine_readout: bool = False,
        predict_variance: bool = False,
    ) -> None:
        super().__init__(domain)
        self.spatial_dim = 2 if self.coord_mode == "local_tangent" else 3
        self.base_resolution = int(base_resolution)
        self.num_scales = int(num_scales)
        self.features_per_plane = int(features_per_plane)
        self.scale_hidden_dim = int(scale_hidden_dim)
        self.bandpass_hidden_dim = int(bandpass_hidden_dim)
        self.global_hidden_dim = max(int(global_hidden_dim), 0)
        self.aggregate_hidden_moments = bool(aggregate_hidden_moments)
        self.coarse_fine_readout = bool(coarse_fine_readout)
        self.predict_variance = bool(predict_variance)
        self.predictive_distribution_available = self.predict_variance
        self._curriculum_progress = 1.0
        plane_names = ["xy", "xl", "yl"]
        if self.spatial_dim == 3:
            plane_names.extend(["xz", "yz", "zl"])
        self._plane_names = tuple(plane_names)
        self.planes = nn.ParameterDict()
        for scale_idx in range(self.num_scales):
            resolution = self.base_resolution * (2**scale_idx)
            for name in self._plane_names:
                self.planes[f"s{scale_idx}_{name}"] = nn.Parameter(
                    1.0e-3
                    * torch.randn(
                        self.features_per_plane,
                        resolution,
                        resolution,
                    )
                )
        per_scale_dim = (
            len(self._plane_names) * self.features_per_plane + self.spatial_dim + 1
        )
        self.scale_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(per_scale_dim, self.scale_hidden_dim),
                    nn.SiLU(),
                    ResidualStack(self.scale_hidden_dim, int(scale_num_blocks)),
                )
                for _ in range(self.num_scales)
            ]
        )
        self.scale_gate = nn.Sequential(
            nn.Linear(self.spatial_dim + 1, self.scale_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.scale_hidden_dim, self.num_scales),
        )
        self.bandpass_context = nn.Sequential(
            nn.Linear(4 + 8, self.bandpass_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.bandpass_hidden_dim, self.bandpass_hidden_dim),
            nn.SiLU(),
        )
        if self.global_hidden_dim > 0:
            self.global_encoder = nn.Sequential(
                nn.Linear(self.spatial_dim + 1, self.global_hidden_dim),
                nn.SiLU(),
                ResidualStack(self.global_hidden_dim, int(global_num_blocks)),
            )
        else:
            self.global_encoder = None
        hidden_multiplier = 2 if self.aggregate_hidden_moments else 1
        coarse_multiplier = hidden_multiplier if self.coarse_fine_readout else 0
        self.decoder = nn.Sequential(
            nn.Linear(
                hidden_multiplier * self.scale_hidden_dim
                + coarse_multiplier * self.scale_hidden_dim
                + self.bandpass_hidden_dim
                + hidden_multiplier * self.global_hidden_dim,
                int(decoder_hidden_dim),
            ),
            nn.SiLU(),
            ResidualStack(int(decoder_hidden_dim), int(decoder_num_blocks)),
        )
        self.base_output = nn.Linear(int(decoder_hidden_dim), 1)
        self.refine = nn.Sequential(
            nn.Linear(int(decoder_hidden_dim), int(refine_hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(refine_hidden_dim), 1),
        )
        if self.predict_variance:
            self.logvar_output = nn.Linear(int(decoder_hidden_dim), 1)
            self.logvar_refine = nn.Sequential(
                nn.Linear(int(decoder_hidden_dim), int(refine_hidden_dim)),
                nn.SiLU(),
                nn.Linear(int(refine_hidden_dim), 1),
            )
        else:
            self.logvar_output = None
            self.logvar_refine = None
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for mlp in self.scale_mlps:
            for module in mlp:
                if type(module) is nn.Linear:
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
        for module in self.scale_gate:
            if type(module) is nn.Linear:
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for module in self.bandpass_context:
            if type(module) is nn.Linear:
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        if self.global_encoder is not None:
            for module in self.global_encoder:
                if type(module) is nn.Linear:
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
        for module in self.decoder:
            if type(module) is nn.Linear:
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.uniform_(self.base_output.weight, -1e-2, 1e-2)
        nn.init.constant_(self.base_output.bias, -3.0)
        for module in self.refine:
            if type(module) is nn.Linear:
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        if self.logvar_output is not None:
            nn.init.uniform_(self.logvar_output.weight, -1e-2, 1e-2)
            nn.init.constant_(self.logvar_output.bias, -2.0)
        if self.logvar_refine is not None:
            for module in self.logvar_refine:
                if type(module) is nn.Linear:
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)

    def set_curriculum_progress(self, progress: float) -> None:
        self._curriculum_progress = min(max(float(progress), 0.0), 1.0)

    def _curriculum_scale_weights(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        eased_progress = math.sqrt(self._curriculum_progress)
        active_extent = 1.0 + (self.num_scales - 1) * eased_progress
        scale_index = torch.arange(
            self.num_scales,
            device=device,
            dtype=dtype,
        )
        return torch.clamp(active_extent - scale_index, min=0.0, max=1.0)

    def _per_scale_hidden_from_components(
        self,
        spatial_raw: Tensor,
        spatial_unit: Tensor,
        lam: Tensor,
    ) -> tuple[Tensor, Tensor]:
        lam_unit = 0.5 * (torch.tanh(lam) + 1.0)
        x = spatial_unit[..., 0]
        y = spatial_unit[..., 1]
        context = torch.cat([spatial_raw, lam.unsqueeze(-1)], dim=-1)
        per_scale_hidden: list[Tensor] = []
        for scale_idx, mlp in enumerate(self.scale_mlps):
            sampled = [
                sample_feature_plane(self.planes[f"s{scale_idx}_xy"], x, y),
                sample_feature_plane(self.planes[f"s{scale_idx}_xl"], x, lam_unit),
                sample_feature_plane(self.planes[f"s{scale_idx}_yl"], y, lam_unit),
            ]
            if self.spatial_dim == 3:
                z = spatial_unit[..., 2]
                sampled.extend(
                    [
                        sample_feature_plane(self.planes[f"s{scale_idx}_xz"], x, z),
                        sample_feature_plane(self.planes[f"s{scale_idx}_yz"], y, z),
                        sample_feature_plane(
                            self.planes[f"s{scale_idx}_zl"], z, lam_unit
                        ),
                    ]
                )
            scale_feats = torch.cat([context, torch.cat(sampled, dim=-1)], dim=-1)
            per_scale_hidden.append(mlp(scale_feats))
        scale_hidden = torch.stack(per_scale_hidden, dim=-2)
        return scale_hidden, context

    def _aggregate_scale_stack(
        self,
        scale_hidden: Tensor,
        context: Tensor,
    ) -> Tensor:
        gate_logits = self.scale_gate(context)
        scale_weights = self._curriculum_scale_weights(
            device=gate_logits.device,
            dtype=gate_logits.dtype,
        )
        gate_logits = gate_logits + torch.log(torch.clamp(scale_weights, min=1.0e-6))
        gate = torch.softmax(gate_logits, dim=-1).unsqueeze(-1)
        return torch.sum(scale_hidden * gate, dim=-2)

    def _scale_hidden_from_components(
        self,
        spatial_raw: Tensor,
        spatial_unit: Tensor,
        lam: Tensor,
    ) -> Tensor:
        scale_hidden, context = self._per_scale_hidden_from_components(
            spatial_raw,
            spatial_unit,
            lam,
        )
        return self._aggregate_scale_stack(scale_hidden, context)

    def _scale_hidden(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        spatial_raw = self._spatial_coords(
            ra_deg,
            dec_deg,
            wavelength_angstrom,
        ).to(dtype=self.base_output.weight.dtype)
        spatial_unit = self._spatial_unit_box_coords(
            ra_deg,
            dec_deg,
            wavelength_angstrom,
        ).to(dtype=self.base_output.weight.dtype)
        lam = self._normalized_wavelength(
            wavelength_angstrom,
            device=ra_deg.device,
            dtype=self.base_output.weight.dtype,
            shape=ra_deg.shape,
        )
        return self._scale_hidden_from_components(spatial_raw, spatial_unit, lam)

    def _global_hidden_from_components(
        self,
        spatial_raw: Tensor,
        lam: Tensor,
    ) -> Tensor | None:
        if self.global_encoder is None:
            return None
        context = torch.cat([spatial_raw, lam.unsqueeze(-1)], dim=-1)
        return self.global_encoder(context)

    def _global_hidden(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor | None:
        spatial_raw = self._spatial_coords(
            ra_deg,
            dec_deg,
            wavelength_angstrom,
        ).to(dtype=self.base_output.weight.dtype)
        lam = self._normalized_wavelength(
            wavelength_angstrom,
            device=ra_deg.device,
            dtype=self.base_output.weight.dtype,
            shape=ra_deg.shape,
        )
        return self._global_hidden_from_components(spatial_raw, lam)

    def _bandpass_features(self, wavelengths: Tensor, weights: Tensor) -> Tensor:
        lam = (
            wavelengths.to(dtype=self.base_output.weight.dtype)
            - float(self.domain.wavelength_center)
        ) / float(self.domain.wavelength_scale)
        norm_weights = weights.to(dtype=self.base_output.weight.dtype)
        norm_weights = norm_weights / torch.clamp(norm_weights.sum(), min=1.0e-8)
        mean = torch.sum(norm_weights * lam)
        centered = lam - mean
        # Element-wise multiplication is faster than torch.square or **2
        centered_sq = centered * centered
        second = torch.sum(norm_weights * centered_sq)
        third = torch.sum(norm_weights * centered_sq * centered)
        fourth = torch.sum(norm_weights * centered_sq * centered_sq)
        feats = [mean, second, third, fourth]
        scales = torch.arange(
            1,
            5,
            device=lam.device,
            dtype=lam.dtype,
        )
        phase = math.pi * scales.view(-1, *([1] * lam.ndim)) * lam
        sin_sum = torch.sum(
            norm_weights * torch.sin(phase), dim=tuple(range(1, 1 + lam.ndim))
        )
        cos_sum = torch.sum(
            norm_weights * torch.cos(phase), dim=tuple(range(1, 1 + lam.ndim))
        )
        freq_feats = torch.stack([sin_sum, cos_sum], dim=1).flatten()
        return torch.cat([torch.stack(feats), freq_feats])

    def _decode_hidden(
        self,
        hidden: Tensor,
        bandpass_latent: Tensor,
        global_hidden: Tensor | None = None,
        coarse_hidden: Tensor | None = None,
    ) -> Tensor:
        if bandpass_latent.ndim < hidden.ndim:
            bandpass_latent = bandpass_latent.view(
                *([1] * (hidden.ndim - 1)),
                bandpass_latent.shape[-1],
            )
            bandpass_latent = bandpass_latent.expand(*hidden.shape[:-1], -1)
        parts = [hidden]
        if coarse_hidden is not None:
            parts.append(coarse_hidden)
        parts.append(bandpass_latent)
        if global_hidden is not None:
            parts.append(global_hidden)
        decoded = self.decoder(torch.cat(parts, dim=-1))
        out = self.base_output(decoded).squeeze(-1) + self.refine(decoded).squeeze(-1)
        return F.softplus(out) + 1.0e-4

    def _decode_distribution(
        self,
        hidden: Tensor,
        bandpass_latent: Tensor,
        global_hidden: Tensor | None = None,
        coarse_hidden: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if bandpass_latent.ndim < hidden.ndim:
            bandpass_latent = bandpass_latent.view(
                *([1] * (hidden.ndim - 1)),
                bandpass_latent.shape[-1],
            )
            bandpass_latent = bandpass_latent.expand(*hidden.shape[:-1], -1)
        parts = [hidden]
        if coarse_hidden is not None:
            parts.append(coarse_hidden)
        parts.append(bandpass_latent)
        if global_hidden is not None:
            parts.append(global_hidden)
        decoded = self.decoder(torch.cat(parts, dim=-1))
        mean = self.base_output(decoded).squeeze(-1) + self.refine(decoded).squeeze(-1)
        mean = F.softplus(mean) + 1.0e-4
        if self.logvar_output is None or self.logvar_refine is None:
            raise ValueError("predictive variance is disabled for this field model")
        logvar = self.logvar_output(decoded).squeeze(-1) + self.logvar_refine(
            decoded
        ).squeeze(-1)
        logvar = torch.clamp(logvar, min=-8.0, max=6.0)
        return mean, logvar

    def _aggregate_hidden_stack(
        self,
        stacked_hidden: Tensor,
        norm_weights: Tensor,
    ) -> Tensor:
        mean_hidden = torch.sum(
            stacked_hidden * norm_weights.view(-1, *([1] * (stacked_hidden.ndim - 1))),
            dim=0,
        )
        if not self.aggregate_hidden_moments:
            return mean_hidden
        centered = stacked_hidden - mean_hidden.unsqueeze(0)
        # ⚡ Bolt: Element-wise multiplication is faster than torch.square or **2
        variance = torch.sum(
            (centered * centered)
            * norm_weights.view(-1, *([1] * (stacked_hidden.ndim - 1))),
            dim=0,
        )
        std_hidden = torch.sqrt(torch.clamp(variance, min=1.0e-8))
        return torch.cat([mean_hidden, std_hidden], dim=-1)

    def render_bandpass(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelengths: Tensor,
        weights: Tensor,
    ) -> Tensor:
        hidden_planes = []
        coarse_hidden_planes = []
        global_hiddens = []
        for wavelength in wavelengths:
            spatial_raw = self._spatial_coords(
                ra_deg,
                dec_deg,
                wavelength,
            ).to(dtype=self.base_output.weight.dtype)
            spatial_unit = self._spatial_unit_box_coords(
                ra_deg,
                dec_deg,
                wavelength,
            ).to(dtype=self.base_output.weight.dtype)
            lam = self._normalized_wavelength(
                wavelength,
                device=ra_deg.device,
                dtype=self.base_output.weight.dtype,
                shape=ra_deg.shape,
            )
            scale_hidden, context = self._per_scale_hidden_from_components(
                spatial_raw,
                spatial_unit,
                lam,
            )
            hidden_planes.append(self._aggregate_scale_stack(scale_hidden, context))
            if self.coarse_fine_readout:
                coarse_hidden_planes.append(scale_hidden[..., 0, :])
            global_hidden = self._global_hidden(ra_deg, dec_deg, wavelength)
            if global_hidden is not None:
                global_hiddens.append(global_hidden)
        stacked_hidden = torch.stack(hidden_planes, dim=0)
        norm_weights = weights.to(dtype=stacked_hidden.dtype)
        norm_weights = norm_weights / torch.clamp(norm_weights.sum(), min=1.0e-8)
        mixed_hidden = self._aggregate_hidden_stack(stacked_hidden, norm_weights)
        bandpass_latent = self.bandpass_context(
            self._bandpass_features(wavelengths, weights)
        )
        mixed_coarse_hidden = None
        if coarse_hidden_planes:
            stacked_coarse_hidden = torch.stack(coarse_hidden_planes, dim=0)
            mixed_coarse_hidden = self._aggregate_hidden_stack(
                stacked_coarse_hidden,
                norm_weights,
            )
        mixed_global_hidden = None
        if global_hiddens:
            stacked_global_hidden = torch.stack(global_hiddens, dim=0)
            mixed_global_hidden = self._aggregate_hidden_stack(
                stacked_global_hidden,
                norm_weights,
            )
        return self._decode_hidden(
            mixed_hidden,
            bandpass_latent,
            mixed_global_hidden,
            mixed_coarse_hidden,
        )

    def render_bandpass_distribution(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelengths: Tensor,
        weights: Tensor,
    ) -> tuple[Tensor, Tensor]:
        hidden_planes = []
        coarse_hidden_planes = []
        global_hiddens = []
        for wavelength in wavelengths:
            spatial_raw = self._spatial_coords(
                ra_deg,
                dec_deg,
                wavelength,
            ).to(dtype=self.base_output.weight.dtype)
            spatial_unit = self._spatial_unit_box_coords(
                ra_deg,
                dec_deg,
                wavelength,
            ).to(dtype=self.base_output.weight.dtype)
            lam = self._normalized_wavelength(
                wavelength,
                device=ra_deg.device,
                dtype=self.base_output.weight.dtype,
                shape=ra_deg.shape,
            )
            scale_hidden, context = self._per_scale_hidden_from_components(
                spatial_raw,
                spatial_unit,
                lam,
            )
            hidden_planes.append(self._aggregate_scale_stack(scale_hidden, context))
            if self.coarse_fine_readout:
                coarse_hidden_planes.append(scale_hidden[..., 0, :])
            global_hidden = self._global_hidden(ra_deg, dec_deg, wavelength)
            if global_hidden is not None:
                global_hiddens.append(global_hidden)
        stacked_hidden = torch.stack(hidden_planes, dim=0)
        norm_weights = weights.to(dtype=stacked_hidden.dtype)
        norm_weights = norm_weights / torch.clamp(norm_weights.sum(), min=1.0e-8)
        mixed_hidden = self._aggregate_hidden_stack(stacked_hidden, norm_weights)
        bandpass_latent = self.bandpass_context(
            self._bandpass_features(wavelengths, weights)
        )
        mixed_coarse_hidden = None
        if coarse_hidden_planes:
            stacked_coarse_hidden = torch.stack(coarse_hidden_planes, dim=0)
            mixed_coarse_hidden = self._aggregate_hidden_stack(
                stacked_coarse_hidden,
                norm_weights,
            )
        mixed_global_hidden = None
        if global_hiddens:
            stacked_global_hidden = torch.stack(global_hiddens, dim=0)
            mixed_global_hidden = self._aggregate_hidden_stack(
                stacked_global_hidden,
                norm_weights,
            )
        return self._decode_distribution(
            mixed_hidden,
            bandpass_latent,
            mixed_global_hidden,
            mixed_coarse_hidden,
        )

    def forward(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        wavelength = torch.as_tensor(
            wavelength_angstrom,
            device=ra_deg.device,
            dtype=torch.float32,
        )
        if wavelength.ndim == 0:
            wavelengths = wavelength.view(1)
        else:
            wavelengths = wavelength.reshape(-1)
        weights = torch.ones_like(wavelengths)
        spatial_raw = self._spatial_coords(
            ra_deg,
            dec_deg,
            wavelength_angstrom,
        ).to(dtype=self.base_output.weight.dtype)
        spatial_unit = self._spatial_unit_box_coords(
            ra_deg,
            dec_deg,
            wavelength_angstrom,
        ).to(dtype=self.base_output.weight.dtype)
        lam = self._normalized_wavelength(
            wavelength_angstrom,
            device=ra_deg.device,
            dtype=self.base_output.weight.dtype,
            shape=ra_deg.shape,
        )
        scale_hidden, context = self._per_scale_hidden_from_components(
            spatial_raw,
            spatial_unit,
            lam,
        )
        hidden = self._aggregate_scale_stack(scale_hidden, context)
        coarse_hidden = scale_hidden[..., 0, :] if self.coarse_fine_readout else None
        global_hidden = self._global_hidden(ra_deg, dec_deg, wavelength_angstrom)
        bandpass_latent = self.bandpass_context(
            self._bandpass_features(wavelengths, weights)
        )
        return self._decode_hidden(
            hidden, bandpass_latent, global_hidden, coarse_hidden
        )

    def forward_distribution(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> tuple[Tensor, Tensor]:
        wavelength = torch.as_tensor(
            wavelength_angstrom,
            device=ra_deg.device,
            dtype=torch.float32,
        )
        if wavelength.ndim == 0:
            wavelengths = wavelength.view(1)
        else:
            wavelengths = wavelength.reshape(-1)
        weights = torch.ones_like(wavelengths)
        spatial_raw = self._spatial_coords(
            ra_deg,
            dec_deg,
            wavelength_angstrom,
        ).to(dtype=self.base_output.weight.dtype)
        spatial_unit = self._spatial_unit_box_coords(
            ra_deg,
            dec_deg,
            wavelength_angstrom,
        ).to(dtype=self.base_output.weight.dtype)
        lam = self._normalized_wavelength(
            wavelength_angstrom,
            device=ra_deg.device,
            dtype=self.base_output.weight.dtype,
            shape=ra_deg.shape,
        )
        scale_hidden, context = self._per_scale_hidden_from_components(
            spatial_raw,
            spatial_unit,
            lam,
        )
        hidden = self._aggregate_scale_stack(scale_hidden, context)
        coarse_hidden = scale_hidden[..., 0, :] if self.coarse_fine_readout else None
        global_hidden = self._global_hidden(ra_deg, dec_deg, wavelength_angstrom)
        bandpass_latent = self.bandpass_context(
            self._bandpass_features(wavelengths, weights)
        )
        return self._decode_distribution(
            hidden,
            bandpass_latent,
            global_hidden,
            coarse_hidden,
        )


class OperatorConstrainedFeatureField(_BaseFieldModel):
    def __init__(
        self,
        domain: FieldDomain,
        *,
        base_resolution: int = 16,
        num_scales: int = 4,
        features_per_plane: int = 12,
        scale_hidden_dim: int = 96,
        scale_num_blocks: int = 2,
        spectral_basis_dim: int = 8,
        decoder_hidden_dim: int = 96,
        decoder_num_blocks: int = 2,
        refine_hidden_dim: int = 48,
        bandpass_hidden_dim: int = 48,
        spectral_context_dim: int | None = None,
        residual_hidden_dim: int = 48,
    ) -> None:
        super().__init__(domain)
        self.spatial_dim = 2 if self.coord_mode == "local_tangent" else 3
        self.base_resolution = int(base_resolution)
        self.num_scales = int(num_scales)
        self.features_per_plane = int(features_per_plane)
        self.scale_hidden_dim = int(scale_hidden_dim)
        self.spectral_basis_dim = int(spectral_basis_dim)
        self.bandpass_hidden_dim = int(bandpass_hidden_dim)
        self.spectral_context_dim = (
            int(spectral_context_dim)
            if spectral_context_dim is not None
            else self.scale_hidden_dim
        )
        self._spectral_observability_cache: dict[
            tuple[tuple[object, ...], ...], Tensor
        ] = {}
        spatial_plane_names = ["xy"] if self.spatial_dim == 2 else ["xy", "xz", "yz"]
        self._spatial_plane_names = tuple(spatial_plane_names)
        self.planes = nn.ParameterDict()
        for scale_idx in range(self.num_scales):
            resolution = self.base_resolution * (2**scale_idx)
            for name in self._spatial_plane_names:
                self.planes[f"s{scale_idx}_{name}"] = nn.Parameter(
                    1.0e-3
                    * torch.randn(
                        self.features_per_plane,
                        resolution,
                        resolution,
                    )
                )
        per_scale_dim = (
            len(self._spatial_plane_names) * self.features_per_plane + self.spatial_dim
        )
        self.scale_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(per_scale_dim, self.scale_hidden_dim),
                    nn.SiLU(),
                    ResidualStack(self.scale_hidden_dim, int(scale_num_blocks)),
                )
                for _ in range(self.num_scales)
            ]
        )
        self.scale_gate = nn.Sequential(
            nn.Linear(self.spatial_dim, self.scale_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.scale_hidden_dim, self.num_scales),
        )
        self.coefficient_head = nn.Sequential(
            nn.Linear(self.scale_hidden_dim, self.scale_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.scale_hidden_dim, self.spectral_basis_dim),
        )
        self.spectral_context = nn.Sequential(
            nn.Linear(2 * self.spectral_basis_dim, self.spectral_context_dim),
            nn.SiLU(),
            nn.Linear(self.spectral_context_dim, self.spectral_context_dim),
            nn.SiLU(),
        )
        self.spectral_affine = nn.Sequential(
            nn.Linear(self.spectral_context_dim, self.scale_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.scale_hidden_dim, 2 * self.scale_hidden_dim),
        )
        self.bandpass_context = nn.Sequential(
            nn.Linear(4 + 8, self.bandpass_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.bandpass_hidden_dim, self.bandpass_hidden_dim),
            nn.SiLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(
                2 * self.scale_hidden_dim
                + self.spectral_context_dim
                + self.bandpass_hidden_dim,
                int(decoder_hidden_dim),
            ),
            nn.SiLU(),
            ResidualStack(int(decoder_hidden_dim), int(decoder_num_blocks)),
        )
        self.base_output = nn.Linear(int(decoder_hidden_dim), 1)
        self.refine = nn.Sequential(
            nn.Linear(int(decoder_hidden_dim), int(refine_hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(refine_hidden_dim), 1),
        )
        self.residual_affine = nn.Sequential(
            nn.Linear(
                self.bandpass_hidden_dim
                + self.spectral_context_dim
                + self.scale_hidden_dim,
                int(residual_hidden_dim),
            ),
            nn.SiLU(),
            nn.Linear(int(residual_hidden_dim), 2 * self.scale_hidden_dim),
        )
        self.residual_decoder = nn.Sequential(
            nn.Linear(
                2 * self.scale_hidden_dim
                + self.spectral_context_dim
                + self.bandpass_hidden_dim,
                int(residual_hidden_dim),
            ),
            nn.SiLU(),
            nn.Linear(int(residual_hidden_dim), 1),
        )
        self._register_spectral_basis()
        self._reset_parameters()

    def _register_spectral_basis(self) -> None:
        lower = max(
            float(self.domain.wavelength_center) - float(self.domain.wavelength_scale),
            1.0,
        )
        upper = max(
            float(self.domain.wavelength_center) + float(self.domain.wavelength_scale),
            lower + 1.0,
        )
        log_center = math.log(float(self.domain.wavelength_center))
        log_scale = max(0.5 * (math.log(upper) - math.log(lower)), 1.0e-3)
        coarse_count = max(1, self.spectral_basis_dim // 3)
        medium_count = max(1, self.spectral_basis_dim // 3)
        fine_count = max(self.spectral_basis_dim - coarse_count - medium_count, 1)
        coarse_centers = torch.linspace(
            -1.0, 1.0, steps=coarse_count, dtype=torch.float32
        )
        medium_centers = torch.linspace(
            -1.0, 1.0, steps=medium_count, dtype=torch.float32
        )
        fine_centers = torch.linspace(-1.0, 1.0, steps=fine_count, dtype=torch.float32)
        centers = torch.cat([coarse_centers, medium_centers, fine_centers], dim=0)[
            : self.spectral_basis_dim
        ]
        widths = torch.cat(
            [
                torch.full((coarse_count,), 0.95, dtype=torch.float32),
                torch.full((medium_count,), 0.45, dtype=torch.float32),
                torch.full((fine_count,), 0.18, dtype=torch.float32),
            ],
            dim=0,
        )[: self.spectral_basis_dim]
        self.register_buffer("spectral_basis_centers", centers, persistent=False)
        self.register_buffer("spectral_basis_widths", widths, persistent=False)
        self.register_buffer(
            "spectral_log_center",
            torch.tensor(log_center, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "spectral_log_scale",
            torch.tensor(log_scale, dtype=torch.float32),
            persistent=False,
        )

    def _reset_parameters(self) -> None:
        for mlp in self.scale_mlps:
            for module in mlp:
                if type(module) is nn.Linear:
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
        for module in self.scale_gate:
            if type(module) is nn.Linear:
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for module in self.coefficient_head:
            if type(module) is nn.Linear:
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for module in self.spectral_context:
            if type(module) is nn.Linear:
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for module in self.spectral_affine:
            if type(module) is nn.Linear:
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for module in self.bandpass_context:
            if type(module) is nn.Linear:
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for module in self.decoder:
            if type(module) is nn.Linear:
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for module in self.refine:
            if type(module) is nn.Linear:
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for module in self.residual_affine:
            if type(module) is nn.Linear:
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for module in self.residual_decoder[:-1]:
            if type(module) is nn.Linear:
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.uniform_(self.base_output.weight, -1.0e-2, 1.0e-2)
        nn.init.constant_(self.base_output.bias, -3.0)
        final_refine = self.refine[-1]
        assert isinstance(final_refine, nn.Linear)
        nn.init.zeros_(final_refine.weight)
        nn.init.zeros_(final_refine.bias)
        residual_out = self.residual_decoder[-1]
        assert isinstance(residual_out, nn.Linear)
        nn.init.zeros_(residual_out.weight)
        nn.init.zeros_(residual_out.bias)

    def _spatial_feature_stack(
        self,
        spatial_raw: Tensor,
        spatial_unit: Tensor,
    ) -> Tensor:
        x = spatial_unit[..., 0]
        y = spatial_unit[..., 1]
        per_scale_hidden: list[Tensor] = []
        for scale_idx, mlp in enumerate(self.scale_mlps):
            sampled = [sample_feature_plane(self.planes[f"s{scale_idx}_xy"], x, y)]
            if self.spatial_dim == 3:
                z = spatial_unit[..., 2]
                sampled.extend(
                    [
                        sample_feature_plane(self.planes[f"s{scale_idx}_xz"], x, z),
                        sample_feature_plane(self.planes[f"s{scale_idx}_yz"], y, z),
                    ]
                )
            feats = torch.cat([spatial_raw, torch.cat(sampled, dim=-1)], dim=-1)
            per_scale_hidden.append(mlp(feats))
        return torch.stack(per_scale_hidden, dim=-2)

    def _shared_hidden_components_from_spatial(
        self,
        spatial_raw: Tensor,
        spatial_unit: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        scale_hidden = self._spatial_feature_stack(spatial_raw, spatial_unit)
        gate_logits = self.scale_gate(spatial_raw)
        gate = torch.softmax(gate_logits, dim=-1).unsqueeze(-1)
        shared_hidden = torch.sum(scale_hidden * gate, dim=-2)
        if self.num_scales == 1:
            return shared_hidden, shared_hidden, torch.zeros_like(shared_hidden)
        scale_positions = torch.linspace(
            0.0,
            1.0,
            steps=self.num_scales,
            device=scale_hidden.device,
            dtype=scale_hidden.dtype,
        ).view(*([1] * (gate.ndim - 2)), self.num_scales, 1)
        coarse_gate = gate * (1.0 - scale_positions)
        fine_gate = gate * scale_positions
        coarse_gate = coarse_gate / torch.clamp(
            coarse_gate.sum(dim=-2, keepdim=True), min=1.0e-8
        )
        fine_gate = fine_gate / torch.clamp(
            fine_gate.sum(dim=-2, keepdim=True), min=1.0e-8
        )
        coarse_hidden = torch.sum(scale_hidden * coarse_gate, dim=-2)
        fine_hidden = torch.sum(scale_hidden * fine_gate, dim=-2)
        return shared_hidden, coarse_hidden, fine_hidden

    def _normalized_log_wavelength(
        self,
        wavelength_angstrom: Tensor | float,
        *,
        device: torch.device,
        dtype: torch.dtype,
        shape: torch.Size,
    ) -> Tensor:
        wavelength = torch.as_tensor(
            wavelength_angstrom,
            device=device,
            dtype=dtype,
        )
        if wavelength.shape != shape:
            wavelength = torch.broadcast_to(wavelength, shape)
        wavelength = torch.clamp(wavelength, min=1.0)
        return (
            torch.log(wavelength)
            - self.spectral_log_center.to(device=device, dtype=dtype)
        ) / (self.spectral_log_scale.to(device=device, dtype=dtype))

    def _spectral_basis(
        self,
        wavelength_angstrom: Tensor | float,
        *,
        device: torch.device,
        dtype: torch.dtype,
        shape: torch.Size,
    ) -> Tensor:
        lam = self._normalized_log_wavelength(
            wavelength_angstrom,
            device=device,
            dtype=dtype,
            shape=shape,
        )
        centers = self.spectral_basis_centers.to(device=device, dtype=dtype)
        widths = self.spectral_basis_widths.to(device=device, dtype=dtype)
        diff = lam.unsqueeze(-1) - centers
        # ⚡ Bolt: Element-wise multiplication is faster than torch.square or **2
        scaled_diff = diff / torch.clamp(widths, min=1.0e-3)
        basis = torch.exp(-0.5 * (scaled_diff * scaled_diff))
        return basis / torch.clamp(basis.sum(dim=-1, keepdim=True), min=1.0e-8)

    def _bandpass_features(self, wavelengths: Tensor, weights: Tensor) -> Tensor:
        lam = self._normalized_log_wavelength(
            wavelengths.to(dtype=self.base_output.weight.dtype),
            device=wavelengths.device,
            dtype=self.base_output.weight.dtype,
            shape=wavelengths.shape,
        )
        norm_weights = weights.to(dtype=self.base_output.weight.dtype)
        norm_weights = norm_weights / torch.clamp(norm_weights.sum(), min=1.0e-8)
        mean = torch.sum(norm_weights * lam)
        centered = lam - mean
        # Element-wise multiplication is faster than torch.square or **2
        centered_sq = centered * centered
        second = torch.sum(norm_weights * centered_sq)
        third = torch.sum(norm_weights * centered_sq * centered)
        fourth = torch.sum(norm_weights * centered_sq * centered_sq)
        feats = [mean, second, third, fourth]
        scales = torch.arange(1, 5, device=lam.device, dtype=lam.dtype)
        phase = math.pi * scales.view(-1, *([1] * lam.ndim)) * lam
        sin_sum = torch.sum(
            norm_weights * torch.sin(phase), dim=tuple(range(1, 1 + lam.ndim))
        )
        cos_sum = torch.sum(
            norm_weights * torch.cos(phase), dim=tuple(range(1, 1 + lam.ndim))
        )
        freq_feats = torch.stack([sin_sum, cos_sum], dim=1).flatten()
        return torch.cat([torch.stack(feats), freq_feats])

    def _spectral_context_from_basis(
        self,
        coeffs: Tensor,
        basis: Tensor,
    ) -> Tensor:
        return self.spectral_context(torch.cat([coeffs * basis, basis], dim=-1))

    def _bandpass_support_scale(self, wavelengths: Tensor, weights: Tensor) -> Tensor:
        lam = self._normalized_log_wavelength(
            wavelengths.to(dtype=self.base_output.weight.dtype),
            device=wavelengths.device,
            dtype=self.base_output.weight.dtype,
            shape=wavelengths.shape,
        )
        norm_weights = weights.to(dtype=lam.dtype)
        norm_weights = norm_weights / torch.clamp(norm_weights.sum(), min=1.0e-8)
        mean = torch.sum(norm_weights * lam)
        # ⚡ Bolt: Element-wise multiplication is faster than torch.square or **2
        lam_diff = lam - mean
        variance = torch.sum(norm_weights * (lam_diff * lam_diff))
        # Keep the continuous field exact for delta-like bands while allowing
        # broader passbands to use a small support-refinement correction.
        return 1.0 - torch.exp(-12.0 * variance)

    def _decode_with_bandpass(
        self,
        shared_hidden: Tensor,
        fine_hidden: Tensor,
        spectral_context: Tensor,
        bandpass_latent: Tensor,
        *,
        support_scale: Tensor | float = 0.0,
    ) -> Tensor:
        if spectral_context.ndim < shared_hidden.ndim:
            spectral_context = spectral_context.view(
                *([1] * (shared_hidden.ndim - 1)),
                spectral_context.shape[-1],
            )
            spectral_context = spectral_context.expand(*shared_hidden.shape[:-1], -1)
        if bandpass_latent.ndim < shared_hidden.ndim:
            bandpass_latent = bandpass_latent.view(
                *([1] * (shared_hidden.ndim - 1)),
                bandpass_latent.shape[-1],
            )
            bandpass_latent = bandpass_latent.expand(*shared_hidden.shape[:-1], -1)
        spectral_gamma_beta = self.spectral_affine(spectral_context)
        spectral_gamma, spectral_beta = torch.chunk(spectral_gamma_beta, 2, dim=-1)
        spectral_hidden = shared_hidden * (1.0 + spectral_gamma) + spectral_beta
        decoded = self.decoder(
            torch.cat(
                [shared_hidden, spectral_hidden, spectral_context, bandpass_latent],
                dim=-1,
            )
        )
        base = self.base_output(decoded).squeeze(-1) + self.refine(decoded).squeeze(-1)
        detail_hidden = fine_hidden - shared_hidden
        residual_context = torch.cat(
            [detail_hidden, spectral_context, bandpass_latent], dim=-1
        )
        gamma_beta = self.residual_affine(residual_context)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        mod_hidden = (spectral_hidden + detail_hidden) * (1.0 + gamma) + beta
        residual = self.residual_decoder(
            torch.cat(
                [mod_hidden, detail_hidden, spectral_context, bandpass_latent], dim=-1
            )
        ).squeeze(-1)
        support_scale_tensor = torch.as_tensor(
            support_scale,
            device=base.device,
            dtype=base.dtype,
        )
        if support_scale_tensor.ndim < base.ndim:
            support_scale_tensor = support_scale_tensor.view(
                *([1] * base.ndim),
            )
        residual = residual * support_scale_tensor
        return F.softplus(base + residual) + 1.0e-4

    def _shared_hidden_and_coeffs(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        spatial_raw = self._spatial_coords(
            ra_deg,
            dec_deg,
            float(self.domain.wavelength_center),
        ).to(dtype=self.base_output.weight.dtype)
        spatial_unit = self._spatial_unit_box_coords(
            ra_deg,
            dec_deg,
            float(self.domain.wavelength_center),
        ).to(dtype=self.base_output.weight.dtype)
        shared_hidden, _coarse_hidden, fine_hidden = (
            self._shared_hidden_components_from_spatial(
                spatial_raw,
                spatial_unit,
            )
        )
        coeffs = self.coefficient_head(shared_hidden)
        return shared_hidden, coeffs, fine_hidden

    def forward(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        wavelength = torch.as_tensor(
            wavelength_angstrom,
            device=ra_deg.device,
            dtype=torch.float32,
        )
        wavelengths = (
            wavelength.reshape(-1) if wavelength.ndim > 0 else wavelength.view(1)
        )
        weights = torch.ones_like(wavelengths)
        shared_hidden, coeffs, fine_hidden = self._shared_hidden_and_coeffs(
            ra_deg, dec_deg
        )
        basis = self._spectral_basis(
            wavelength_angstrom,
            device=ra_deg.device,
            dtype=shared_hidden.dtype,
            shape=ra_deg.shape,
        )
        spectral_context = self._spectral_context_from_basis(coeffs, basis)
        bandpass_latent = self.bandpass_context(
            self._bandpass_features(wavelengths, weights)
        )
        return self._decode_with_bandpass(
            shared_hidden,
            fine_hidden,
            spectral_context,
            bandpass_latent,
            support_scale=self._bandpass_support_scale(wavelengths, weights),
        )

    def render_bandpass(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelengths: Tensor,
        weights: Tensor,
    ) -> Tensor:
        shared_hidden, coeffs, fine_hidden = self._shared_hidden_and_coeffs(
            ra_deg, dec_deg
        )
        norm_weights = weights.to(dtype=shared_hidden.dtype)
        norm_weights = norm_weights / torch.clamp(norm_weights.sum(), min=1.0e-8)
        basis_stack = torch.stack(
            [
                self._spectral_basis(
                    wavelength,
                    device=ra_deg.device,
                    dtype=shared_hidden.dtype,
                    shape=ra_deg.shape,
                )
                for wavelength in wavelengths
            ],
            dim=0,
        )
        mixed_basis = torch.sum(
            basis_stack * norm_weights.view(-1, *([1] * (basis_stack.ndim - 1))),
            dim=0,
        )
        spectral_context = self._spectral_context_from_basis(coeffs, mixed_basis)
        bandpass_latent = self.bandpass_context(
            self._bandpass_features(wavelengths, weights)
        )
        return self._decode_with_bandpass(
            shared_hidden,
            fine_hidden,
            spectral_context,
            bandpass_latent,
            support_scale=self._bandpass_support_scale(wavelengths, weights),
        )

    def _observability_mode_weights(
        self,
        observations: Sequence[FieldObservation],
        *,
        device: torch.device,
        dtype: torch.dtype,
        grid_size: int = 65,
    ) -> Tensor:
        cache_key = tuple(
            sorted(
                {
                    (
                        observation.band_name,
                        observation.spectral_min,
                        observation.spectral_max,
                        observation.transmission_center,
                        observation.transmission_sigma,
                    )
                    for observation in observations
                }
            )
        )
        cached = self._spectral_observability_cache.get(cache_key)
        if cached is not None:
            return cached.to(device=device, dtype=dtype)
        lower = max(
            min(_bandpass_bounds(observation)[0] for observation in observations),
            1.0,
        )
        upper = max(_bandpass_bounds(observation)[1] for observation in observations)
        wavelengths = torch.linspace(
            lower,
            upper,
            steps=max(int(grid_size), 9),
            device=device,
            dtype=dtype,
        )
        basis = self._spectral_basis(
            wavelengths,
            device=device,
            dtype=dtype,
            shape=wavelengths.shape,
        )
        responses = []
        for observation in observations:
            response = _bandpass_response(observation, wavelengths)
            response = response / torch.clamp(response.sum(), min=1.0e-8)
            responses.append(torch.sum(response.unsqueeze(-1) * basis, dim=0))
        response_matrix = torch.stack(responses, dim=0)
        # ⚡ Bolt: Element-wise multiplication is faster than torch.square or **2
        visibility = torch.sqrt(torch.sum(response_matrix * response_matrix, dim=0))
        max_visibility = torch.clamp(visibility.max(), min=1.0e-8)
        weights = torch.clamp(
            max_visibility / torch.clamp(visibility, min=1.0e-4) - 1.0, min=0.0
        )
        weights = weights / torch.clamp(weights.mean(), min=1.0e-8)
        self._spectral_observability_cache[cache_key] = weights.detach().cpu()
        return weights

    def spectral_observability_penalty(
        self,
        observations: Sequence[FieldObservation],
        *,
        probe_grid_size: int = 8,
    ) -> Tensor:
        device = self.base_output.weight.device
        dtype = self.base_output.weight.dtype
        probe_ra, probe_dec = _smoothness_probe_grid(
            self.domain,
            grid_size=max(int(probe_grid_size), 2),
            device=device,
        )
        _shared_hidden, coeffs, _fine_hidden = self._shared_hidden_and_coeffs(
            probe_ra,
            probe_dec,
        )
        mode_weights = self._observability_mode_weights(
            observations,
            device=device,
            dtype=dtype,
        )
        return torch.mean(
            torch.square(coeffs) * mode_weights.view(*([1] * (coeffs.ndim - 1)), -1)
        )

    def residual_head_penalty(self) -> Tensor:
        penalty = torch.tensor(
            0.0,
            device=self.base_output.weight.device,
            dtype=self.base_output.weight.dtype,
        )
        for param in self.residual_affine.parameters():
            penalty = penalty + torch.mean(param * param)
        residual_out = self.residual_decoder[-1]
        assert isinstance(residual_out, nn.Linear)
        penalty = penalty + torch.mean(residual_out.weight * residual_out.weight)
        penalty = penalty + torch.mean(residual_out.bias * residual_out.bias)
        return penalty


class SkyLambdaAtlasField(_BaseFieldModel):
    def __init__(
        self,
        domain: FieldDomain,
        *,
        atlas_side: int = 2,
        atlas_sigma: float = 0.3,
        topk_experts: int | None = None,
        base_resolution: int = 16,
        num_scales: int = 3,
        features_per_plane: int = 8,
        hidden_dim: int = 64,
        hidden_layers: int = 2,
        refine_hidden_dim: int = 32,
    ) -> None:
        super().__init__(domain)
        self.atlas_side = int(atlas_side)
        self.atlas_sigma = float(atlas_sigma)
        self.topk_experts = None if topk_experts is None else max(int(topk_experts), 1)
        centers_1d = torch.linspace(
            0.5 / self.atlas_side,
            1.0 - 0.5 / self.atlas_side,
            self.atlas_side,
            dtype=torch.float32,
        )
        yy, xx = torch.meshgrid(centers_1d, centers_1d, indexing="ij")
        centers = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
        self.register_buffer("atlas_centers", centers, persistent=False)
        self.experts = nn.ModuleList(
            [
                SkyLambdaKPlanesField(
                    domain,
                    base_resolution=base_resolution,
                    num_scales=num_scales,
                    features_per_plane=features_per_plane,
                    hidden_dim=hidden_dim,
                    hidden_layers=hidden_layers,
                    refine_hidden_dim=refine_hidden_dim,
                )
                for _ in range(self.atlas_side * self.atlas_side)
            ]
        )

    def _atlas_local_xy(
        self,
        xy: Tensor,
    ) -> Tensor:
        local = []
        tile_size = 1.0 / float(self.atlas_side)
        for center in self.atlas_centers:
            lower = center - 0.5 * tile_size
            shifted = (xy - lower.to(device=xy.device, dtype=xy.dtype)) / tile_size
            local.append(torch.clamp(shifted, 0.0, 1.0))
        return torch.stack(local, dim=-2)

    def _atlas_expert_forward(
        self,
        expert: SkyLambdaKPlanesField,
        spatial_raw: Tensor,
        spatial_unit: Tensor,
        lam: Tensor,
        lam_unit: Tensor,
        local_xy: Tensor,
    ) -> Tensor:
        x = local_xy[..., 0]
        y = local_xy[..., 1]
        sampled: list[Tensor] = []
        for scale_idx in range(expert.num_scales):
            sampled.extend(
                [
                    sample_feature_plane(expert.planes[f"s{scale_idx}_xy"], x, y),
                    sample_feature_plane(
                        expert.planes[f"s{scale_idx}_xl"], x, lam_unit
                    ),
                    sample_feature_plane(
                        expert.planes[f"s{scale_idx}_yl"], y, lam_unit
                    ),
                ]
            )
            if expert.spatial_dim == 3:
                z = spatial_unit[..., 2]
                sampled.extend(
                    [
                        sample_feature_plane(expert.planes[f"s{scale_idx}_xz"], x, z),
                        sample_feature_plane(expert.planes[f"s{scale_idx}_yz"], y, z),
                        sample_feature_plane(
                            expert.planes[f"s{scale_idx}_zl"], z, lam_unit
                        ),
                    ]
                )
        plane_feats = torch.cat(sampled, dim=-1)
        local_xy_raw = 2.0 * local_xy - 1.0
        if expert.spatial_dim == 3:
            spatial_feats = torch.cat([local_xy_raw, spatial_raw[..., 2:3]], dim=-1)
        else:
            spatial_feats = local_xy_raw
        feats = torch.cat([spatial_feats, lam.unsqueeze(-1), plane_feats], dim=-1)
        hidden = expert.hidden(feats)
        return hidden

    def _atlas_weights(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        spatial_unit = self._spatial_unit_box_coords(
            ra_deg,
            dec_deg,
            wavelength_angstrom,
        )
        xy = spatial_unit[..., :2]
        diff = xy.unsqueeze(-2) - self.atlas_centers.to(
            device=xy.device, dtype=xy.dtype
        )
        logits = -0.5 * torch.sum(
            torch.square(diff / max(self.atlas_sigma, 1.0e-3)), dim=-1
        )
        weights = torch.softmax(logits, dim=-1)
        if self.topk_experts is not None and self.topk_experts < weights.shape[-1]:
            topk_vals, topk_idx = torch.topk(weights, k=self.topk_experts, dim=-1)
            sparse = torch.zeros_like(weights)
            sparse.scatter_(-1, topk_idx, topk_vals)
            weights = sparse / torch.clamp(sparse.sum(dim=-1, keepdim=True), min=1.0e-8)
        return weights

    def forward(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        spatial_raw = self._spatial_coords(
            ra_deg,
            dec_deg,
            wavelength_angstrom,
        ).to(dtype=self.experts[0].base_output.weight.dtype)
        spatial_unit = self._spatial_unit_box_coords(
            ra_deg,
            dec_deg,
            wavelength_angstrom,
        ).to(dtype=self.experts[0].base_output.weight.dtype)
        lam = self._normalized_wavelength(
            wavelength_angstrom,
            device=ra_deg.device,
            dtype=self.experts[0].base_output.weight.dtype,
            shape=ra_deg.shape,
        )
        lam_unit = 0.5 * (torch.tanh(lam) + 1.0)
        weights = self._atlas_weights(ra_deg, dec_deg, wavelength_angstrom)
        local_xys = self._atlas_local_xy(spatial_unit[..., :2])
        preds = torch.stack(
            [
                expert._decode_hidden(
                    self._atlas_expert_forward(
                        expert,
                        spatial_raw,
                        spatial_unit,
                        lam,
                        lam_unit,
                        local_xys[..., expert_idx, :],
                    )
                )
                for expert_idx, expert in enumerate(self.experts)
            ],
            dim=-1,
        )
        return torch.sum(weights * preds, dim=-1)

    def render_bandpass(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelengths: Tensor,
        weights: Tensor,
    ) -> Tensor:
        norm_weights = weights.to(dtype=self.experts[0].base_output.weight.dtype)
        norm_weights = norm_weights / torch.clamp(norm_weights.sum(), min=1.0e-8)
        spatial_raw = self._spatial_coords(
            ra_deg,
            dec_deg,
            float(wavelengths[0].item()),
        ).to(dtype=self.experts[0].base_output.weight.dtype)
        spatial_unit = self._spatial_unit_box_coords(
            ra_deg,
            dec_deg,
            float(wavelengths[0].item()),
        ).to(dtype=self.experts[0].base_output.weight.dtype)
        local_xys = self._atlas_local_xy(spatial_unit[..., :2])
        atlas_weights = self._atlas_weights(
            ra_deg,
            dec_deg,
            float(wavelengths[0].item()),
        ).to(dtype=self.experts[0].base_output.weight.dtype)
        mixed_expert_hiddens: list[Tensor] = []
        for expert_idx, expert in enumerate(self.experts):
            per_lambda_hidden = []
            for wavelength in wavelengths:
                lam = expert._normalized_wavelength(
                    wavelength,
                    device=ra_deg.device,
                    dtype=expert.base_output.weight.dtype,
                    shape=ra_deg.shape,
                )
                lam_unit = 0.5 * (torch.tanh(lam) + 1.0)
                per_lambda_hidden.append(
                    self._atlas_expert_forward(
                        expert,
                        spatial_raw,
                        spatial_unit,
                        lam,
                        lam_unit,
                        local_xys[..., expert_idx, :],
                    )
                )
            stacked_hidden = torch.stack(per_lambda_hidden, dim=0)
            mixed_expert_hiddens.append(
                torch.sum(
                    stacked_hidden
                    * norm_weights.view(-1, *([1] * (stacked_hidden.ndim - 1))),
                    dim=0,
                )
            )
        preds = torch.stack(
            [
                expert._decode_hidden(hidden)
                for expert, hidden in zip(
                    self.experts, mixed_expert_hiddens, strict=True
                )
            ],
            dim=-1,
        )
        return torch.sum(atlas_weights * preds, dim=-1)


class SkyLambdaFeatureAtlasField(_BaseFieldModel):
    def __init__(
        self,
        domain: FieldDomain,
        *,
        atlas_side: int = 2,
        atlas_sigma: float = 0.2,
        topk_experts: int | None = None,
        base_resolution: int = 16,
        num_scales: int = 3,
        features_per_plane: int = 8,
        scale_hidden_dim: int = 64,
        scale_num_blocks: int = 2,
        decoder_hidden_dim: int = 64,
        decoder_num_blocks: int = 2,
        refine_hidden_dim: int = 32,
        bandpass_hidden_dim: int = 32,
        global_hidden_dim: int = 0,
        global_num_blocks: int = 1,
        aggregate_hidden_moments: bool = False,
    ) -> None:
        super().__init__(domain)
        self.atlas_side = int(atlas_side)
        self.atlas_sigma = float(atlas_sigma)
        self.topk_experts = None if topk_experts is None else max(int(topk_experts), 1)
        centers_1d = torch.linspace(
            0.5 / self.atlas_side,
            1.0 - 0.5 / self.atlas_side,
            self.atlas_side,
            dtype=torch.float32,
        )
        yy, xx = torch.meshgrid(centers_1d, centers_1d, indexing="ij")
        centers = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
        self.register_buffer("atlas_centers", centers, persistent=False)
        self.experts = nn.ModuleList(
            [
                SkyLambdaFeatureField(
                    domain,
                    base_resolution=base_resolution,
                    num_scales=num_scales,
                    features_per_plane=features_per_plane,
                    scale_hidden_dim=scale_hidden_dim,
                    scale_num_blocks=scale_num_blocks,
                    decoder_hidden_dim=decoder_hidden_dim,
                    decoder_num_blocks=decoder_num_blocks,
                    refine_hidden_dim=refine_hidden_dim,
                    bandpass_hidden_dim=bandpass_hidden_dim,
                    global_hidden_dim=global_hidden_dim,
                    global_num_blocks=global_num_blocks,
                    aggregate_hidden_moments=aggregate_hidden_moments,
                )
                for _ in range(self.atlas_side * self.atlas_side)
            ]
        )

    def _atlas_local_xy(self, xy: Tensor) -> Tensor:
        local = []
        tile_size = 1.0 / float(self.atlas_side)
        for center in self.atlas_centers:
            lower = center - 0.5 * tile_size
            shifted = (xy - lower.to(device=xy.device, dtype=xy.dtype)) / tile_size
            local.append(torch.clamp(shifted, 0.0, 1.0))
        return torch.stack(local, dim=-2)

    def _atlas_weights(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        spatial_unit = self._spatial_unit_box_coords(
            ra_deg,
            dec_deg,
            wavelength_angstrom,
        )
        xy = spatial_unit[..., :2]
        diff = xy.unsqueeze(-2) - self.atlas_centers.to(
            device=xy.device, dtype=xy.dtype
        )
        logits = -0.5 * torch.sum(
            torch.square(diff / max(self.atlas_sigma, 1.0e-3)),
            dim=-1,
        )
        weights = torch.softmax(logits, dim=-1)
        if self.topk_experts is not None and self.topk_experts < weights.shape[-1]:
            topk_vals, topk_idx = torch.topk(weights, k=self.topk_experts, dim=-1)
            sparse = torch.zeros_like(weights)
            sparse.scatter_(-1, topk_idx, topk_vals)
            weights = sparse / torch.clamp(
                sparse.sum(dim=-1, keepdim=True),
                min=1.0e-8,
            )
        return weights

    def _expert_local_components(
        self,
        expert: SkyLambdaFeatureField,
        spatial_raw: Tensor,
        spatial_unit: Tensor,
        lam: Tensor,
        local_xy: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        local_xy_raw = 2.0 * local_xy - 1.0
        if expert.spatial_dim == 3:
            local_spatial_raw = torch.cat([local_xy_raw, spatial_raw[..., 2:3]], dim=-1)
            local_spatial_unit = torch.cat([local_xy, spatial_unit[..., 2:3]], dim=-1)
        else:
            local_spatial_raw = local_xy_raw
            local_spatial_unit = local_xy
        return local_spatial_raw, local_spatial_unit, lam

    def _expert_prediction(
        self,
        expert: SkyLambdaFeatureField,
        spatial_raw: Tensor,
        spatial_unit: Tensor,
        local_xy: Tensor,
        wavelengths: Tensor,
        weights: Tensor,
    ) -> Tensor:
        hidden_planes = []
        global_hiddens = []
        dtype = expert.base_output.weight.dtype
        for wavelength in wavelengths:
            lam = expert._normalized_wavelength(
                wavelength,
                device=spatial_raw.device,
                dtype=dtype,
                shape=spatial_raw.shape[:-1],
            )
            local_spatial_raw, local_spatial_unit, lam = self._expert_local_components(
                expert,
                spatial_raw,
                spatial_unit,
                lam,
                local_xy,
            )
            hidden_planes.append(
                expert._scale_hidden_from_components(
                    local_spatial_raw,
                    local_spatial_unit,
                    lam,
                )
            )
            global_hidden = expert._global_hidden_from_components(
                local_spatial_raw, lam
            )
            if global_hidden is not None:
                global_hiddens.append(global_hidden)
        stacked_hidden = torch.stack(hidden_planes, dim=0)
        norm_weights = weights.to(dtype=stacked_hidden.dtype)
        norm_weights = norm_weights / torch.clamp(norm_weights.sum(), min=1.0e-8)
        mixed_hidden = expert._aggregate_hidden_stack(stacked_hidden, norm_weights)
        bandpass_latent = expert.bandpass_context(
            expert._bandpass_features(wavelengths, weights)
        )
        mixed_global_hidden = None
        if global_hiddens:
            stacked_global_hidden = torch.stack(global_hiddens, dim=0)
            mixed_global_hidden = expert._aggregate_hidden_stack(
                stacked_global_hidden,
                norm_weights,
            )
        return expert._decode_hidden(mixed_hidden, bandpass_latent, mixed_global_hidden)

    def render_bandpass(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelengths: Tensor,
        weights: Tensor,
    ) -> Tensor:
        reference_wavelength = float(wavelengths[0].item())
        spatial_raw = self._spatial_coords(
            ra_deg,
            dec_deg,
            reference_wavelength,
        ).to(dtype=self.experts[0].base_output.weight.dtype)
        spatial_unit = self._spatial_unit_box_coords(
            ra_deg,
            dec_deg,
            reference_wavelength,
        ).to(dtype=self.experts[0].base_output.weight.dtype)
        local_xys = self._atlas_local_xy(spatial_unit[..., :2])
        atlas_weights = self._atlas_weights(
            ra_deg,
            dec_deg,
            reference_wavelength,
        ).to(dtype=self.experts[0].base_output.weight.dtype)
        preds = torch.stack(
            [
                self._expert_prediction(
                    expert,
                    spatial_raw,
                    spatial_unit,
                    local_xys[..., expert_idx, :],
                    wavelengths,
                    weights,
                )
                for expert_idx, expert in enumerate(self.experts)
            ],
            dim=-1,
        )
        return torch.sum(atlas_weights * preds, dim=-1)

    def forward(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        wavelength = torch.as_tensor(
            wavelength_angstrom,
            device=ra_deg.device,
            dtype=torch.float32,
        )
        if wavelength.ndim == 0:
            wavelengths = wavelength.view(1)
        else:
            wavelengths = wavelength.reshape(-1)
        weights = torch.ones_like(wavelengths)
        return self.render_bandpass(ra_deg, dec_deg, wavelengths, weights)


def make_field_model(
    kind: FieldModelKind,
    domain: FieldDomain,
    **kwargs: object,
) -> nn.Module:
    if kind == "spectral_power_foam":
        return SpectralPowerFoamField(domain, **kwargs)
    if kind == "foam_initialized_inr":
        return FoamInitializedINRField(domain, **kwargs)
    if kind == "diffuse_foam_hybrid":
        return DiffuseFoamHybridField(domain, **kwargs)
    if kind == "sine_implicit":
        return SineImplicitField(domain, **kwargs)
    if kind == "wire_implicit":
        return WireImplicitField(domain, **kwargs)
    if kind == "explicit_spectral_grid_field":
        return ExplicitSpectralGridField(domain, **kwargs)
    if kind == "sasnet":
        from .frequency_adaptive import SASNetField

        return SASNetField(domain, **kwargs)
    if kind == "flair_bla":
        from .frequency_adaptive import FlairField

        return FlairField(domain, **kwargs)
    if kind == "gaussian_basis":
        return GaussianBasisField(domain, **kwargs)
    if kind == "fourier_mlp":
        return FourierMLPField(domain, **kwargs)
    if kind == "spectral_basis_field":
        return SpectralBasisField(domain, **kwargs)
    if kind == "bandpass_conditioned_field":
        return BandpassConditionedField(domain, **kwargs)
    if kind == "learned_spectral_basis_field":
        return LearnedSpectralBasisField(domain, **kwargs)
    if kind == "sky_encoder_bandpass_field":
        return SkyEncoderBandpassField(domain, **kwargs)
    if kind == "sky_lambda_planes_field":
        return SkyLambdaPlanesField(domain, **kwargs)
    if kind == "sky_lambda_kplanes_field":
        return SkyLambdaKPlanesField(domain, **kwargs)
    if kind == "sky_lambda_feature_field":
        return SkyLambdaFeatureField(domain, **kwargs)
    if kind == "operator_constrained_feature_field":
        return OperatorConstrainedFeatureField(domain, **kwargs)
    if kind == "sky_lambda_feature_field_hetero":
        return SkyLambdaFeatureField(domain, predict_variance=True, **kwargs)
    if kind == "sky_lambda_feature_atlas_field":
        return SkyLambdaFeatureAtlasField(domain, **kwargs)
    if kind == "sky_lambda_atlas_field":
        return SkyLambdaAtlasField(domain, **kwargs)
    if kind == "conditioned_spectral_field":
        return ConditionedSpectralField(domain, **kwargs)
    if kind == "hashgrid_conditioned_field":
        return HashGridConditionedField(domain, **kwargs)
    if kind == "sky_encoder_mlp":
        return SkyEncoderMLPField(domain, **kwargs)
    if kind == "strainer_sky_encoder_mlp":
        return StrainerSkyEncoderMLPField(domain, **kwargs)
    if kind == "sky_encoder_resmlp":
        return SkyEncoderResMLPField(domain, **kwargs)
    if kind == "hashgrid_mlp":
        return HashGridMLPField(domain, **kwargs)
    if kind == "hashgrid_spectral_field":
        return HashGridSpectralField(domain, **kwargs)
    if kind == "hashgrid_resmlp":
        return HashGridResMLPField(domain, **kwargs)
    raise ValueError(f"unknown field model kind {kind!r}")
