"""Neural field-model classes for :mod:`torchsky.map.field`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from typing import Any, Literal
import copy
import math

import torch
import torch.nn as nn
from torch import Tensor

from ._core_helpers import (
    _BaseFieldModel,
)
from .foam_common import PowerValueSemantics
from .foam_field import (
    FoamAdaptationResult,
    SpectralPowerFoam,
    adapt_spectral_power_foam,
)
from .foam_operators import sparse_bounded_power_pixel_operator
from .foam_solvers import PowerPixelOperator
from .models_implicit import SineImplicitField
from .networks import (
    reset_silu_mlp_linears,
    silu_mlp_sequential,
)
from .types import (
    FieldDomain,
)


if TYPE_CHECKING:
    pass


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
            raise ValueError(
                "hybrid regularization weights must be finite and non-negative"
            )
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
        return (
            sparsity_weight * localized.abs().mean()
            + separation_weight * correlation_squared
        )

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
