"""Bounded power-diagram primitives for experimental sky-field models.

These tensor-native kernels implement the geometry underlying Power Foam.  They
are representation primitives, not a complete renderer: exact rendering also
needs sparse candidate construction and pixel/cell intersection integration.
"""

from __future__ import annotations

from .foam_common import PowerValueSemantics
from .foam_geometry import (
    _validate_sites_and_radii,
    smooth_bounded_power_weights,
    soft_bounded_power_weights,
)
from .foam_solvers import (
    PowerPixelOperator,
    render_power_spectral_affine_basis,
    render_power_spectral_basis,
    render_power_spectral_samples,
)

from dataclasses import dataclass
import math
from typing import Any, Sequence

import torch
from torch import Tensor

@dataclass(frozen=True)
class PersistentFoamSiteInitialization:
    """Deterministic sites selected from coherent multi-exposure evidence."""

    sites: Tensor
    radii: Tensor
    scores: Tensor
    exposure_counts: Tensor
    bin_ids: Tensor


@dataclass(frozen=True)
class FoamAdaptationResult:
    """Cell-identity mapping from one deterministic Foam adaptation step."""

    densified_cell_ids: tuple[int, ...]
    pruned_cell_ids: tuple[int, ...]
    old_to_new: tuple[int, ...]


def persistent_foam_site_initialization(
    exposure_coordinates: Sequence[Tensor],
    exposure_whitened_residuals: Sequence[Tensor],
    *,
    num_sites: int,
    min_exposures: int,
    bin_shape: tuple[int, int],
    lower: tuple[float, float],
    upper: tuple[float, float],
    initial_radius: float,
    exposure_contributions: Sequence[Tensor] | None = None,
) -> PersistentFoamSiteInitialization:
    """Select sites from positive residual evidence coherent across exposures.

    Each exposure contributes one weighted mean whitened residual per spatial
    bin, so dense exposures cannot dominate sparse ones. Candidate score is the
    number of covered exposures times the squared coherent positive mean.
    ``min_exposures`` is an explicit persistence gate; ties resolve by bin id.
    """
    coordinates = tuple(exposure_coordinates)
    residuals = tuple(exposure_whitened_residuals)
    if not coordinates or len(coordinates) != len(residuals):
        raise ValueError("coordinates and residuals require matching exposures")
    if exposure_contributions is None:
        contributions: tuple[Tensor | None, ...] = (None,) * len(coordinates)
    else:
        contributions = tuple(exposure_contributions)
        if len(contributions) != len(coordinates):
            raise ValueError("exposure_contributions must match exposures")
    requested_sites = int(num_sites)
    required_exposures = int(min_exposures)
    bins_y, bins_x = map(int, bin_shape)
    if requested_sites < 1:
        raise ValueError("num_sites must be positive")
    if required_exposures < 1 or required_exposures > len(coordinates):
        raise ValueError("min_exposures must be within the exposure count")
    if bins_y < 1 or bins_x < 1:
        raise ValueError("bin_shape dimensions must be positive")
    radius = float(initial_radius)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("initial_radius must be finite and positive")
    first = coordinates[0]
    if not first.is_floating_point():
        raise ValueError("exposure coordinates must be floating tensors")
    device, dtype = first.device, first.dtype
    lower_t = torch.tensor(lower, device=device, dtype=dtype)
    upper_t = torch.tensor(upper, device=device, dtype=dtype)
    if (
        lower_t.shape != (2,)
        or upper_t.shape != (2,)
        or not bool(torch.isfinite(lower_t).all())
        or not bool(torch.isfinite(upper_t).all())
        or bool(torch.any(upper_t <= lower_t))
    ):
        raise ValueError("lower and upper must define a finite 2-D box")

    bin_count = bins_y * bins_x
    signed_sum = torch.zeros(bin_count, device=device, dtype=dtype)
    covered_count = torch.zeros(bin_count, device=device, dtype=torch.int64)
    positive_count = torch.zeros_like(covered_count)
    for exposure_index, (coords, values, weights) in enumerate(
        zip(coordinates, residuals, contributions, strict=True)
    ):
        flat_coords = coords.reshape(-1, 2)
        flat_values = values.reshape(-1).to(device=device, dtype=dtype)
        if (
            coords.device != device
            or coords.dtype != dtype
            or coords.ndim < 2
            or coords.shape[-1] != 2
            or flat_values.numel() != flat_coords.shape[0]
            or not bool(torch.isfinite(flat_coords).all())
            or not bool(torch.isfinite(flat_values).all())
        ):
            raise ValueError(
                f"exposure {exposure_index} coordinates/residuals are incompatible"
            )
        if bool(torch.any(flat_coords < lower_t)) or bool(
            torch.any(flat_coords > upper_t)
        ):
            raise ValueError("exposure coordinates must lie within lower/upper")
        if weights is None:
            flat_weights = torch.ones_like(flat_values)
        else:
            flat_weights = weights.reshape(-1).to(device=device, dtype=dtype)
            if (
                flat_weights.numel() != flat_values.numel()
                or not bool(torch.isfinite(flat_weights).all())
                or bool(torch.any(flat_weights < 0.0))
            ):
                raise ValueError(
                    f"exposure {exposure_index} contributions must be finite and non-negative"
                )
        unit = (flat_coords - lower_t) / (upper_t - lower_t)
        bin_xy = torch.floor(
            unit * torch.tensor([bins_x, bins_y], device=device, dtype=dtype)
        ).to(dtype=torch.int64)
        bin_xy[:, 0].clamp_(0, bins_x - 1)
        bin_xy[:, 1].clamp_(0, bins_y - 1)
        bin_ids = bin_xy[:, 1] * bins_x + bin_xy[:, 0]
        weighted_sum = torch.zeros_like(signed_sum).index_add(
            0,
            bin_ids,
            flat_values * flat_weights,
        )
        weight_sum = torch.zeros_like(signed_sum).index_add(
            0,
            bin_ids,
            flat_weights,
        )
        covered = weight_sum > 0.0
        exposure_mean = torch.where(
            covered,
            weighted_sum / torch.clamp(weight_sum, min=torch.finfo(dtype).tiny),
            torch.zeros_like(weighted_sum),
        )
        signed_sum.add_(exposure_mean)
        covered_count.add_(covered.to(dtype=torch.int64))
        positive_count.add_((covered & (exposure_mean > 0.0)).to(dtype=torch.int64))

    coherent_mean = signed_sum / torch.clamp(covered_count, min=1).to(dtype=dtype)
    scores = (
        covered_count.to(dtype=dtype) * torch.clamp_min(coherent_mean, 0.0).square()
    )
    valid_ids = torch.nonzero(
        (positive_count >= required_exposures) & (scores > 0.0),
        as_tuple=False,
    ).flatten()
    if valid_ids.numel() < requested_sites:
        raise ValueError(
            "insufficient persistent positive-evidence bins for requested num_sites"
        )
    order = torch.argsort(scores[valid_ids], descending=True, stable=True)
    selected = valid_ids[order[:requested_sites]]
    selected_y = torch.div(selected, bins_x, rounding_mode="floor")
    selected_x = torch.remainder(selected, bins_x)
    step = (upper_t - lower_t) / torch.tensor(
        [bins_x, bins_y], device=device, dtype=dtype
    )
    sites = (
        lower_t
        + (torch.stack([selected_x, selected_y], dim=-1).to(dtype=dtype) + 0.5) * step
    )
    return PersistentFoamSiteInitialization(
        sites=sites,
        radii=torch.full((requested_sites,), radius, device=device, dtype=dtype),
        scores=scores[selected],
        exposure_counts=positive_count[selected],
        bin_ids=selected,
    )


class SpectralPowerFoam(torch.nn.Module):
    """Trainable shared geometry with caller-defined spectral basis values.

    Smooth point queries are the optimization path. :meth:`render_exact` reuses
    the learned appearance coefficients with a hard ``PowerPixelOperator`` built
    from detached geometry for scientific inference.
    """

    _minimum_radius: Tensor
    heldout_residual_ema: Tensor
    contribution_ema: Tensor
    residual_position_ema: Tensor
    evidence_updates: Tensor

    def __init__(

        self,
        sites: Tensor,
        radii: Tensor,
        cell_coefficients: Tensor,
        *,
        cell_gradient_coefficients: Tensor | None = None,
        minimum_radius: float = 0.0,
        log_intensity_spectra: bool = False,
    ) -> None:
        super().__init__()
        _validate_sites_and_radii(sites, radii, allow_empty=False)
        if sites.shape[1] != 2:
            raise ValueError("SpectralPowerFoam requires 2-D sites")
        if (
            not cell_coefficients.is_floating_point()
            or cell_coefficients.ndim != 2
            or cell_coefficients.shape[0] != sites.shape[0]
            or cell_coefficients.shape[1] == 0
        ):
            raise ValueError("cell_coefficients must have shape (cells, basis>=1)")
        if (
            cell_coefficients.device != sites.device
            or cell_coefficients.dtype != sites.dtype
        ):
            raise ValueError(
                "sites, radii, and cell_coefficients must share dtype and device"
            )
        if not bool(torch.isfinite(cell_coefficients).all()):
            raise ValueError("cell_coefficients must be finite")
        if cell_gradient_coefficients is not None and (
            not cell_gradient_coefficients.is_floating_point()
            or cell_gradient_coefficients.shape != cell_coefficients.shape + (2,)
            or cell_gradient_coefficients.device != sites.device
            or cell_gradient_coefficients.dtype != sites.dtype
            or not bool(torch.isfinite(cell_gradient_coefficients).all())
        ):
            raise ValueError(
                "cell_gradient_coefficients must match coefficients with trailing 2"
            )
        if log_intensity_spectra and cell_gradient_coefficients is not None:
            raise ValueError(
                "log-intensity Foam does not yet support exact affine spatial spectra"
            )
        radius_floor = float(minimum_radius)
        if not math.isfinite(radius_floor) or radius_floor < 0.0:
            raise ValueError("minimum_radius must be finite and non-negative")
        shifted = radii.detach() - radius_floor
        if bool(torch.any(shifted < 0)):
            raise ValueError("initial radii must be at least minimum_radius")
        scale = torch.clamp(radii.detach().abs().max(), min=1.0)
        shifted = torch.clamp(
            shifted,
            min=torch.finfo(radii.dtype).eps * scale,
        )
        raw_radii = shifted + torch.log(-torch.expm1(-shifted))

        self.sites = torch.nn.Parameter(sites.detach().clone())
        self.raw_radii = torch.nn.Parameter(raw_radii)
        self.cell_coefficients = torch.nn.Parameter(cell_coefficients.detach().clone())
        self.cell_gradient_coefficients = (
            torch.nn.Parameter(cell_gradient_coefficients.detach().clone())
            if cell_gradient_coefficients is not None
            else None
        )
        self.log_intensity_spectra = bool(log_intensity_spectra)
        self.register_buffer(
            "_minimum_radius",
            radii.new_tensor(radius_floor),
        )
        self.register_buffer(
            "heldout_residual_ema",
            torch.zeros_like(radii),
        )
        self.register_buffer(
            "contribution_ema",
            torch.zeros_like(radii),
        )
        self.register_buffer(
            "residual_position_ema",
            sites.detach().clone(),
        )
        self.register_buffer(
            "evidence_updates",
            torch.zeros(sites.shape[0], device=sites.device, dtype=torch.int64),
        )

    @property
    def radii(self) -> Tensor:
        """Strictly positive trainable radii in the geometry coordinate unit."""
        return self._minimum_radius + torch.nn.functional.softplus(self.raw_radii)

    @property
    def basis_count(self) -> int:
        return int(self.cell_coefficients.shape[1])

    @torch.no_grad()
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
        """Update per-cell held-out residual, contribution, and position EMAs."""
        if split != "held_out":
            raise ValueError("Foam adaptation statistics require split='held_out'")
        rate = float(decay)
        if not math.isfinite(rate) or rate < 0.0 or rate >= 1.0:
            raise ValueError("decay must be finite in [0, 1)")
        indices = cell_indices.reshape(-1).to(
            device=self.sites.device,
            dtype=torch.int64,
        )
        residual = whitened_residuals.reshape(-1).to(
            device=self.sites.device,
            dtype=self.sites.dtype,
        )
        contribution = contributions.reshape(-1).to(
            device=self.sites.device,
            dtype=self.sites.dtype,
        )
        coords = positions.reshape(-1, 2).to(
            device=self.sites.device,
            dtype=self.sites.dtype,
        )
        sample_count = indices.numel()
        if (
            sample_count == 0
            or residual.numel() != sample_count
            or contribution.numel() != sample_count
            or coords.shape[0] != sample_count
        ):
            raise ValueError(
                "adaptation statistic samples must be non-empty and aligned"
            )
        if (
            bool(torch.any(indices < 0))
            or bool(torch.any(indices >= self.sites.shape[0]))
            or not bool(torch.isfinite(residual).all())
            or not bool(torch.isfinite(contribution).all())
            or bool(torch.any(contribution < 0.0))
            or not bool(torch.isfinite(coords).all())
        ):
            raise ValueError("adaptation statistic values are invalid")

        cell_count = int(self.sites.shape[0])
        counts = torch.zeros(
            cell_count,
            device=self.sites.device,
            dtype=self.sites.dtype,
        ).index_add_(0, indices, torch.ones_like(residual))
        residual_energy = residual.square()
        residual_sum = torch.zeros_like(counts).index_add_(0, indices, residual_energy)
        contribution_sum = torch.zeros_like(counts).index_add_(0, indices, contribution)
        weighted_position = torch.zeros_like(self.residual_position_ema).index_add_(
            0,
            indices,
            coords * residual_energy.unsqueeze(-1),
        )
        ordinary_position = torch.zeros_like(self.residual_position_ema).index_add_(
            0,
            indices,
            coords,
        )
        active = counts > 0.0
        batch_residual = residual_sum / torch.clamp(
            counts, min=torch.finfo(counts.dtype).tiny
        )
        batch_contribution = contribution_sum / torch.clamp(
            counts, min=torch.finfo(counts.dtype).tiny
        )
        batch_position = torch.where(
            (residual_sum > 0.0).unsqueeze(-1),
            weighted_position
            / torch.clamp(
                residual_sum.unsqueeze(-1), min=torch.finfo(counts.dtype).tiny
            ),
            ordinary_position
            / torch.clamp(counts.unsqueeze(-1), min=torch.finfo(counts.dtype).tiny),
        )
        first = self.evidence_updates == 0
        blend = active & ~first
        initialize = active & first
        self.heldout_residual_ema[initialize] = batch_residual[initialize]
        self.contribution_ema[initialize] = batch_contribution[initialize]
        self.residual_position_ema[initialize] = batch_position[initialize]
        self.heldout_residual_ema[blend] = (
            rate * self.heldout_residual_ema[blend]
            + (1.0 - rate) * batch_residual[blend]
        )
        self.contribution_ema[blend] = (
            rate * self.contribution_ema[blend]
            + (1.0 - rate) * batch_contribution[blend]
        )
        self.residual_position_ema[blend] = (
            rate * self.residual_position_ema[blend]
            + (1.0 - rate) * batch_position[blend]
        )
        self.evidence_updates[active] += 1

    def forward(
        self,
        points: Tensor,
        basis_values: Tensor,
        *,
        power_temperature: float = 10.0,
        support_temperature: float = 10.0,
        compact_fixed_geometry: bool = False,
    ) -> Tensor:
        """Evaluate point samples; uncovered union mass contributes zero.

        ``compact_fixed_geometry`` uses exact bounded support and is intended for
        appearance-only optimization. It avoids the dense point-by-all-cells
        matrix, but deliberately rejects trainable sites or radii.
        """
        if (
            not basis_values.is_floating_point()
            or basis_values.ndim < 1
            or basis_values.shape[-1] != self.basis_count
        ):
            raise ValueError("basis_values must have shape (..., basis)")
        if (
            basis_values.device != self.sites.device
            or basis_values.dtype != self.sites.dtype
        ):
            raise ValueError("basis_values must match model dtype and device")
        if basis_values.ndim > 1 and basis_values.shape[:-1] != points.shape[:-1]:
            raise ValueError("non-scalar basis_values leading shape must match points")
        selected: Tensor | None = None
        if compact_fixed_geometry:
            if self.sites.requires_grad or self.raw_radii.requires_grad:
                raise ValueError(
                    "compact_fixed_geometry requires frozen sites and radii"
                )
            support_scale = float(support_temperature)
            if not math.isfinite(support_scale) or support_scale <= 0:
                raise ValueError("support_temperature must be finite and positive")
            flat_points = points.reshape(-1, points.shape[-1])
            if flat_points.shape[0] == 0:
                return points[..., 0] * self.cell_coefficients.sum()
            with torch.no_grad():
                lower = flat_points.amin(dim=0)
                upper = flat_points.amax(dim=0)
                # ponytail: this O(cells) scan is cheaper than maintaining a
                # spatial index here; switch to cached tile CSR if profiling says so.
                closest = torch.minimum(torch.maximum(self.sites, lower), upper)
                selected = torch.nonzero(
                    torch.sum((self.sites - closest) ** 2, dim=1) <= self.radii**2,
                    as_tuple=False,
                ).flatten()
            if selected.numel() == 0:
                return points[..., 0] * 0.0 + self.cell_coefficients.sum() * 0.0
            weights = soft_bounded_power_weights(
                points,
                self.sites[selected],
                self.radii[selected],
                temperature=power_temperature,
            )
        else:
            weights = smooth_bounded_power_weights(
                points,
                self.sites,
                self.radii,
                power_temperature=power_temperature,
                support_temperature=support_temperature,
            )
        coefficients = (
            self.cell_coefficients
            if selected is None
            else self.cell_coefficients[selected]
        )
        cell_values = torch.einsum(
            "...k,ck->...c",
            basis_values,
            coefficients,
        )
        if self.cell_gradient_coefficients is not None:
            gradient_coefficients = (
                self.cell_gradient_coefficients
                if selected is None
                else self.cell_gradient_coefficients[selected]
            )
            gradients = torch.einsum(
                "...k,ckd->...cd",
                basis_values,
                gradient_coefficients,
            )
            selected_sites = self.sites if selected is None else self.sites[selected]
            cell_values = cell_values + torch.sum(
                gradients * (points[..., None, :] - selected_sites),
                dim=-1,
            )
        if self.log_intensity_spectra:
            cell_values = torch.exp(cell_values)
        return torch.sum(weights * cell_values, dim=-1)

    def render_exact(
        self,
        operator: PowerPixelOperator,
        basis_samples: Tensor,
        response_weights: Tensor,
        *,
        semantics: PowerValueSemantics = "surface_brightness",
    ) -> Tensor:
        """Render hard pixels with an operator built from detached model geometry."""
        if operator.cell_count != self.sites.shape[0]:
            raise ValueError("operator cell count must match the model")
        if self.cell_gradient_coefficients is not None:
            if semantics != "surface_brightness":
                raise ValueError(
                    "affine Foam exact rendering requires surface_brightness semantics"
                )
            return render_power_spectral_affine_basis(
                operator,
                self.cell_coefficients,
                self.cell_gradient_coefficients,
                basis_samples,
                response_weights,
            )
        if self.log_intensity_spectra:
            cell_samples = torch.exp(
                torch.einsum(
                    "ck,sk->sc",
                    self.cell_coefficients,
                    basis_samples,
                )
            )
            return render_power_spectral_samples(
                operator,
                cell_samples,
                response_weights,
                semantics=semantics,
            )
        return render_power_spectral_basis(
            operator,
            self.cell_coefficients,
            basis_samples,
            response_weights,
            semantics=semantics,
        )


def _rank_cell_ids(scores: Tensor, eligible: Tensor, *, descending: bool) -> Tensor:
    cell_ids = torch.nonzero(eligible, as_tuple=False).flatten()
    return cell_ids[torch.argsort(scores[cell_ids], descending=descending, stable=True)]


def _replace_optimizer_parameter(
    optimizer: torch.optim.Optimizer,
    old: torch.nn.Parameter,
    new: torch.nn.Parameter,
    *,
    keep_ids: Tensor,
    child_count: int,
) -> None:
    found = False
    for group in optimizer.param_groups:
        parameters = group["params"]
        for index, parameter in enumerate(parameters):
            if parameter is old:
                parameters[index] = new
                found = True
    old_state = optimizer.state.pop(old, None)
    if old_state is None:
        return
    if not found:
        raise ValueError("optimizer state contains a parameter outside its groups")
    new_state: dict[Any, Any] = {}
    for name, value in old_state.items():
        if isinstance(value, Tensor) and value.shape == old.shape:
            retained = value.index_select(0, keep_ids.to(device=value.device))
            children = torch.zeros(
                (child_count, *value.shape[1:]),
                device=value.device,
                dtype=value.dtype,
            )
            new_state[name] = torch.cat([retained, children], dim=0)
        elif isinstance(value, Tensor):
            new_state[name] = value.clone()
        else:
            new_state[name] = value
    optimizer.state[new] = new_state


def _replace_cell_parameter(
    model: SpectralPowerFoam,
    optimizer: torch.optim.Optimizer,
    name: str,
    values: Tensor,
    *,
    keep_ids: Tensor,
    child_count: int,
) -> None:
    old = getattr(model, name)
    if not isinstance(old, torch.nn.Parameter):
        raise TypeError(f"Foam {name} is not a parameter")
    new = torch.nn.Parameter(values, requires_grad=old.requires_grad)
    setattr(model, name, new)
    _replace_optimizer_parameter(
        optimizer,
        old,
        new,
        keep_ids=keep_ids,
        child_count=child_count,
    )


def _inverse_radius_parameter(radii: Tensor, minimum_radius: Tensor) -> Tensor:
    shifted = radii - minimum_radius
    if bool(torch.any(shifted <= 0.0)):
        raise ValueError("adapted radii must remain above minimum_radius")
    return shifted + torch.log(-torch.expm1(-shifted))


@torch.no_grad()
def adapt_spectral_power_foam(
    model: SpectralPowerFoam,
    optimizer: torch.optim.Optimizer,
    *,
    densify_count: int,
    prune_count: int,
    min_updates: int = 1,
) -> FoamAdaptationResult:
    """Deterministically split high-error cells and prune low contributors.

    Densification proposals are the held-out residual-position EMAs. A split
    copies appearance, assigns equal-area radii to parent and child, preserves
    optimizer state for retained cells, and starts child optimizer moments at
    zero. Pruning and densification counts are explicit—there are no hidden
    score thresholds.
    """
    add_count = int(densify_count)
    remove_count = int(prune_count)
    required_updates = int(min_updates)
    cell_count = int(model.sites.shape[0])
    if add_count < 0 or remove_count < 0:
        raise ValueError("densify_count and prune_count must be non-negative")
    if required_updates < 1:
        raise ValueError("min_updates must be positive")
    if remove_count >= cell_count:
        raise ValueError("pruning must leave at least one Foam cell")
    if add_count == 0 and remove_count == 0:
        return FoamAdaptationResult((), (), tuple(range(cell_count)))

    updated = model.evidence_updates >= required_updates
    split_radii = model.radii.detach() / math.sqrt(2.0)
    movable = torch.any(model.residual_position_ema != model.sites.detach(), dim=-1) & (
        split_radii > model._minimum_radius
    )
    densify_ranked = _rank_cell_ids(
        model.heldout_residual_ema,
        updated & movable,
        descending=True,
    )
    if densify_ranked.numel() < add_count:
        raise ValueError("insufficient held-out evidence for requested densification")
    densified = densify_ranked[:add_count]
    densify_mask = torch.zeros(cell_count, dtype=torch.bool, device=model.sites.device)
    densify_mask[densified] = True

    prune_ranked = _rank_cell_ids(
        model.contribution_ema,
        updated & ~densify_mask,
        descending=False,
    )
    if prune_ranked.numel() < remove_count:
        raise ValueError("insufficient held-out evidence for requested pruning")
    pruned = prune_ranked[:remove_count]
    keep_mask = torch.ones(cell_count, dtype=torch.bool, device=model.sites.device)
    keep_mask[pruned] = False
    keep_ids = torch.nonzero(keep_mask, as_tuple=False).flatten()
    keep_list = keep_ids.tolist()
    kept_position = {old_id: new_id for new_id, old_id in enumerate(keep_list)}

    old_sites = model.sites.detach()
    old_radii = model.radii.detach()
    old_coefficients = model.cell_coefficients.detach()
    new_sites = torch.cat(
        [old_sites[keep_ids], model.residual_position_ema[densified]], dim=0
    )
    new_radii = torch.cat([old_radii[keep_ids], split_radii[densified]], dim=0)
    for cell_id in densified.tolist():
        new_radii[kept_position[cell_id]] = split_radii[cell_id]
    new_coefficients = torch.cat(
        [old_coefficients[keep_ids], old_coefficients[densified]], dim=0
    )
    new_raw_radii = _inverse_radius_parameter(new_radii, model._minimum_radius)

    _replace_cell_parameter(
        model,
        optimizer,
        "sites",
        new_sites,
        keep_ids=keep_ids,
        child_count=add_count,
    )
    _replace_cell_parameter(
        model,
        optimizer,
        "raw_radii",
        new_raw_radii,
        keep_ids=keep_ids,
        child_count=add_count,
    )
    _replace_cell_parameter(
        model,
        optimizer,
        "cell_coefficients",
        new_coefficients,
        keep_ids=keep_ids,
        child_count=add_count,
    )
    if model.cell_gradient_coefficients is not None:
        old_gradients = model.cell_gradient_coefficients.detach()
        new_gradients = torch.cat(
            [old_gradients[keep_ids], old_gradients[densified]], dim=0
        )
        _replace_cell_parameter(
            model,
            optimizer,
            "cell_gradient_coefficients",
            new_gradients,
            keep_ids=keep_ids,
            child_count=add_count,
        )

    residual_stats = torch.cat(
        [model.heldout_residual_ema[keep_ids], model.heldout_residual_ema[densified]]
    )
    contribution_stats = torch.cat(
        [model.contribution_ema[keep_ids], model.contribution_ema[densified]], dim=0
    )
    for child_index, cell_id in enumerate(densified.tolist()):
        parent_index = kept_position[cell_id]
        half = 0.5 * model.contribution_ema[cell_id]
        contribution_stats[parent_index] = half
        contribution_stats[len(keep_ids) + child_index] = half
    position_stats = torch.cat(
        [
            model.residual_position_ema[keep_ids],
            model.residual_position_ema[densified],
        ],
        dim=0,
    )
    update_stats = torch.cat(
        [model.evidence_updates[keep_ids], model.evidence_updates[densified]], dim=0
    )
    model.heldout_residual_ema = residual_stats
    model.contribution_ema = contribution_stats
    model.residual_position_ema = position_stats
    model.evidence_updates = update_stats

    old_to_new = [-1] * cell_count
    for new_id, old_id in enumerate(keep_list):
        old_to_new[old_id] = new_id
    return FoamAdaptationResult(
        densified_cell_ids=tuple(int(value) for value in densified.tolist()),
        pruned_cell_ids=tuple(int(value) for value in pruned.tolist()),
        old_to_new=tuple(old_to_new),
    )
