"""Tensor-native pseudo-C_ell estimation on regular CAR grids.

Spectra use IAU ``(E, B)`` ordering.  Spin-0 x spin-2 blocks are ``(TE,
TB)`` and spin-2 x spin-2 blocks are ``(EE, EB, BE, BB)``.  Coupling acts on
the final multipole axis and never retains an unbinned ell-by-ell matrix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor

from . import spectral as _spectral
from .sht import CARGeometry, SHTBackend, _wigner_sequence, analyze, synthesize


class ModeCouplingError(RuntimeError):
    """Raised when a mode-coupling system cannot be inverted safely."""


_VALIDATED_MASK_TOKEN = object()


def _require_real_floating(value: Tensor, name: str) -> None:
    if value.dtype not in {torch.float32, torch.float64}:
        raise TypeError(f"{name} must use float32 or float64")


def _validate_window(
    value: Tensor | None, field_value: Tensor, lmax: int, name: str
) -> Tensor:
    if value is None:
        return torch.ones(lmax + 1, dtype=field_value.dtype, device=field_value.device)
    result = torch.as_tensor(value, dtype=field_value.dtype, device=field_value.device)
    if result.ndim != 1 or int(result.shape[0]) != lmax + 1:
        raise ValueError(f"{name} must have shape ({lmax + 1},)")
    if not bool(torch.all(torch.isfinite(result)).item()):
        raise ValueError(f"{name} must be finite")
    return result


def _pixel_weights(geometry: CARGeometry, value: Tensor) -> Tensor:
    ring = geometry.ring_weights(dtype=value.dtype, device=value.device)
    ring = ring * (2.0 * math.pi / geometry.nlon)
    shape = [1, 1]
    shape[geometry.latitude_axis] = geometry.nlat
    return ring.reshape(shape).expand(geometry.shape)


def _condition_diagnostics(
    matrix: Tensor, maximum_condition_number: float | None
) -> tuple[int, float, float, Tensor]:
    diagnostic_matrix = matrix.to("cpu") if matrix.device.type == "mps" else matrix
    singular_values = torch.linalg.svdvals(diagnostic_matrix)
    largest = singular_values[0]
    smallest = singular_values[-1]
    eps = torch.finfo(matrix.dtype).eps
    rank_tolerance = eps * max(matrix.shape) * largest
    rank = int(
        torch.count_nonzero(singular_values > rank_tolerance).detach().cpu().item()
    )
    condition = (
        float((largest / smallest).detach().cpu().item())
        if bool((smallest > 0).item())
        else math.inf
    )
    condition_limit = (
        eps**-0.5
        if maximum_condition_number is None
        else float(maximum_condition_number)
    )
    if not math.isfinite(condition_limit) or condition_limit < 1.0:
        raise ValueError("maximum_condition_number must be finite and at least one")
    return rank, condition, condition_limit, singular_values


@dataclass(frozen=True, slots=True)
class ApodizedMask:
    """A scalar mask and its spin-1/spin-2 raising derivatives.

    ``first_derivative`` and ``second_derivative`` are two-component spin maps
    representing ``eth W`` and ``eth eth W`` in the geometry's declared spin
    convention.  Construction verifies harmonic derivative identities and the
    pure-estimator boundary conditions ``W = eth W = 0`` at both polar rings.
    """

    values: Tensor
    first_derivative: Tensor
    second_derivative: Tensor
    geometry: CARGeometry
    lmax: int
    _validation_token: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._validation_token is not _VALIDATED_MASK_TOKEN:
            raise ValueError("construct apodized masks with ApodizedMask.from_mask")

    @classmethod
    def from_mask(
        cls,
        values: Tensor,
        geometry: CARGeometry,
        lmax: int,
        *,
        backend: str | SHTBackend | None = None,
        latitude_block_size: int = 64,
    ) -> ApodizedMask:
        mask = torch.as_tensor(values)
        _require_real_floating(mask, "values")
        if tuple(mask.shape) != geometry.shape:
            raise ValueError(f"values must have shape {geometry.shape}")
        if int(lmax) != lmax or int(lmax) < 2:
            raise ValueError("pure E/B masks require integer lmax >= 2")
        if geometry.nlat < int(lmax) + 1 or geometry.nlon < 2 * int(lmax) + 1:
            raise ValueError("mask geometry does not sample the requested lmax")
        if not bool(torch.all(torch.isfinite(mask)).item()):
            raise ValueError("apodized mask must be finite")
        if bool(torch.any((mask < 0.0) | (mask > 1.0)).item()):
            raise ValueError("apodized mask values must lie in [0, 1]")
        mask_alm = analyze(
            mask,
            geometry,
            lmax=int(lmax),
            backend=backend,
            latitude_block_size=latitude_block_size,
        ).values
        ell = _spectral._alm_ell_array(int(lmax), int(lmax)).to(mask.device)
        real_ell = ell.to(mask.dtype)
        first_factor = torch.sqrt(real_ell * (real_ell + 1.0))
        second_factor = torch.sqrt(
            torch.clamp(
                (real_ell - 1.0) * real_ell * (real_ell + 1.0) * (real_ell + 2.0),
                min=0.0,
            )
        )
        zero = torch.zeros_like(mask_alm)
        first = synthesize(
            torch.stack((mask_alm * first_factor, zero)),
            geometry,
            lmax=int(lmax),
            spin=1,
            backend=backend,
            latitude_block_size=latitude_block_size,
        ).values
        second = synthesize(
            torch.stack((mask_alm * second_factor, zero)),
            geometry,
            lmax=int(lmax),
            spin=2,
            backend=backend,
            latitude_block_size=latitude_block_size,
        ).values
        reconstructed = synthesize(
            mask_alm,
            geometry,
            lmax=int(lmax),
            backend=backend,
            latitude_block_size=latitude_block_size,
        ).values
        eps = torch.finfo(mask.dtype).eps
        scale = torch.amax(torch.abs(mask)).clamp_min(1.0)
        tolerance = 128.0 * eps * max(1, geometry.pixel_count) * scale
        if bool(torch.amax(torch.abs(reconstructed - mask)) > tolerance):
            raise ValueError("pure E/B mask must be band-limited at lmax")
        canonical_mask = mask.transpose(-2, -1) if geometry.latitude_axis == 1 else mask
        canonical_first = (
            first.transpose(-2, -1) if geometry.latitude_axis == 1 else first
        )
        boundary_mask = torch.stack((canonical_mask[0], canonical_mask[-1]))
        boundary_first = torch.stack(
            (canonical_first[..., 0, :], canonical_first[..., -1, :])
        )
        derivative_scale = torch.amax(torch.abs(first)).clamp_min(1.0)
        derivative_tolerance = (
            256.0 * eps * max(1, geometry.pixel_count) * derivative_scale
        )
        if bool(torch.amax(torch.abs(boundary_mask)) > tolerance):
            raise ValueError("pure E/B mask must vanish on both boundary rings")
        if bool(torch.amax(torch.abs(boundary_first)) > derivative_tolerance):
            raise ValueError("first spin derivative must vanish on both boundary rings")
        if bool(torch.any(canonical_mask[1:-1] <= 0.0).item()):
            raise ValueError(
                "pure E/B currently supports only masks with polar boundaries and positive interiors"
            )
        return cls(mask, first, second, geometry, int(lmax), _VALIDATED_MASK_TOKEN)


@dataclass(frozen=True, slots=True)
class PseudoClField:
    """A scalar or spin-2 CAR map, mask, and harmonic transfer contract."""

    values: Tensor
    mask: Tensor
    geometry: CARGeometry
    lmax: int
    spin: Literal[0, 2] = 0
    beam: Tensor | None = None
    pixel_window: Tensor | None = None
    templates: Tensor | None = None
    maximum_template_condition_number: float | None = None
    apodized_mask: ApodizedMask | None = None
    purify_e: bool = False
    purify_b: bool = False
    backend: str | SHTBackend | None = field(default=None, repr=False, compare=False)
    latitude_block_size: int = 64

    def __post_init__(self) -> None:
        values = torch.as_tensor(self.values)
        _require_real_floating(values, "values")
        if not isinstance(self.geometry, CARGeometry):
            raise TypeError("PseudoClField requires CARGeometry")
        if int(self.lmax) != self.lmax or int(self.lmax) < 0:
            raise ValueError("lmax must be a non-negative integer")
        if self.spin not in {0, 2}:
            raise ValueError("spin must be 0 or 2")
        if int(self.latitude_block_size) < 1:
            raise ValueError("latitude_block_size must be positive")
        component_axes = 0 if self.spin == 0 else 1
        required_ndim = 2 + component_axes
        if (
            values.ndim < required_ndim
            or tuple(values.shape[-2:]) != self.geometry.shape
        ):
            if self.spin == 0:
                raise ValueError(
                    f"scalar values must end in shape {self.geometry.shape}"
                )
            raise ValueError(
                f"spin-2 values must end in shape (2, {self.geometry.shape[0]}, {self.geometry.shape[1]})"
            )
        if self.spin == 2 and int(values.shape[-3]) != 2:
            raise ValueError("spin-2 values require Q/U component axis of length 2")
        if self.geometry.nlat < int(self.lmax) + 1:
            raise ValueError("CAR latitude sampling requires nlat >= lmax + 1")
        mask = torch.as_tensor(self.mask, dtype=values.dtype, device=values.device)
        if tuple(mask.shape) != self.geometry.shape:
            raise ValueError(f"mask must have shape {self.geometry.shape}")
        if not bool(torch.all(torch.isfinite(mask)).item()):
            raise ValueError("mask must be finite")
        beam = _validate_window(self.beam, values, int(self.lmax), "beam")
        pixel_window = _validate_window(
            self.pixel_window, values, int(self.lmax), "pixel_window"
        )
        templates = None
        if self.templates is not None:
            templates = torch.as_tensor(
                self.templates, dtype=values.dtype, device=values.device
            )
            map_shape = geometry_shape = self.geometry.shape
            if self.spin == 2:
                map_shape = (2, *geometry_shape)
            if (
                templates.ndim != len(map_shape) + 1
                or tuple(templates.shape[1:]) != map_shape
            ):
                raise ValueError(
                    f"templates must have shape (ntemplate, {', '.join(str(v) for v in map_shape)})"
                )
            if int(templates.shape[0]) < 1:
                raise ValueError("templates must contain at least one template")
            if not bool(torch.all(torch.isfinite(templates)).item()):
                raise ValueError("templates must be finite")
        if self.purify_e or self.purify_b:
            if self.spin != 2:
                raise ValueError("pure E/B estimators require a spin-2 field")
            if self.apodized_mask is None:
                raise ValueError("pure E/B estimators require apodized_mask")
        if self.apodized_mask is not None:
            if self.apodized_mask.geometry != self.geometry:
                raise ValueError("apodized mask geometry must match field geometry")
            if self.apodized_mask.lmax != int(self.lmax):
                raise ValueError("apodized mask lmax must match field lmax")
            if not bool(torch.equal(self.apodized_mask.values, mask)):
                raise ValueError("field mask must equal apodized_mask.values")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "beam", beam)
        object.__setattr__(self, "pixel_window", pixel_window)
        object.__setattr__(self, "templates", templates)
        object.__setattr__(self, "lmax", int(self.lmax))
        object.__setattr__(self, "latitude_block_size", int(self.latitude_block_size))

    @property
    def transfer(self) -> Tensor:
        if self.beam is None or self.pixel_window is None:
            raise RuntimeError("validated field has no transfer windows")
        return self.beam * self.pixel_window

    def deprojected_values(self) -> Tensor:
        if self.templates is None:
            return self.values
        templates = self.templates
        component_ndim = 2 if self.spin == 0 else 3
        feature_count = math.prod(self.values.shape[-component_ndim:])
        template_flat = templates.reshape(int(templates.shape[0]), feature_count)
        weights = _pixel_weights(self.geometry, self.values) * self.mask.square()
        if self.spin == 2:
            weights = weights.unsqueeze(0).expand(2, *self.geometry.shape)
        weight_flat = weights.reshape(feature_count)
        gram = (template_flat * weight_flat) @ template_flat.transpose(0, 1)
        rank, condition, condition_limit, _ = _condition_diagnostics(
            gram, self.maximum_template_condition_number
        )
        if rank != int(gram.shape[0]):
            raise ModeCouplingError(
                f"template Gram matrix is rank deficient: rank {rank} < {gram.shape[0]}"
            )
        if not math.isfinite(condition) or condition > condition_limit:
            raise ModeCouplingError(
                f"template Gram matrix condition number {condition:.6g} exceeds {condition_limit:.6g}"
            )
        batch_shape = self.values.shape[:-component_ndim]
        values_flat = self.values.reshape(-1, feature_count)
        rhs = (values_flat * weight_flat) @ template_flat.transpose(0, 1)
        coefficients = torch.linalg.solve(gram, rhs.transpose(0, 1)).transpose(0, 1)
        cleaned = values_flat - coefficients @ template_flat
        return cleaned.reshape(*batch_shape, *self.values.shape[-component_ndim:])

    def masked_alms(self) -> Tensor:
        values = self.deprojected_values()
        if self.spin == 2 and (self.purify_e or self.purify_b):
            pure = pure_alms(self, values=values)
            standard = analyze(
                values * self.mask.unsqueeze(-3),
                self.geometry,
                lmax=self.lmax,
                spin=2,
                backend=self.backend,
                latitude_block_size=self.latitude_block_size,
            ).values
            return torch.stack(
                (
                    pure[..., 0, :] if self.purify_e else standard[..., 0, :],
                    pure[..., 1, :] if self.purify_b else standard[..., 1, :],
                ),
                dim=-2,
            )
        weighting = self.mask if self.spin == 0 else self.mask.unsqueeze(-3)
        return analyze(
            values * weighting,
            self.geometry,
            lmax=self.lmax,
            spin=self.spin,
            backend=self.backend,
            latitude_block_size=self.latitude_block_size,
        ).values


def _spin_product(left: Tensor, right: Tensor, *, conjugate_left: bool) -> Tensor:
    left_complex = torch.complex(left[..., 0, :, :], left[..., 1, :, :])
    right_complex = torch.complex(right[..., 0, :, :], right[..., 1, :, :])
    if conjugate_left:
        left_complex = torch.conj(left_complex)
    product = left_complex * right_complex
    if conjugate_left:
        product = -product
    return torch.stack((product.real, product.imag), dim=-3)


def pure_alms(field: PseudoClField, *, values: Tensor | None = None) -> Tensor:
    """Return pure ``(E, B)`` alms for a validated apodized spin-2 field.

    The Smith--Zaldarriaga scalar ``chi_E/chi_B`` form is used.  It is exactly
    the integration-by-parts form containing ``eth W`` and ``eth eth W`` when
    the validated boundary conditions hold.  Multipoles below two are zero.
    """

    if field.spin != 2 or field.apodized_mask is None:
        raise ValueError("pure_alms requires a spin-2 field with apodized_mask")
    maps = (
        field.values
        if values is None
        else torch.as_tensor(
            values, dtype=field.values.dtype, device=field.values.device
        )
    )
    if tuple(maps.shape[-3:]) != (2, *field.geometry.shape):
        raise ValueError(
            f"values must end in shape (2, {field.geometry.shape[0]}, {field.geometry.shape[1]})"
        )
    apodized = field.apodized_mask
    source_alms = analyze(
        maps,
        field.geometry,
        lmax=field.lmax,
        spin=2,
        backend=field.backend,
        latitude_block_size=field.latitude_block_size,
    ).values
    ell = _spectral._alm_ell_array(field.lmax, field.lmax).to(maps.device)
    real_ell = ell.to(maps.dtype)
    normalization_two = torch.sqrt(
        torch.clamp(
            (real_ell - 1.0) * real_ell * (real_ell + 1.0) * (real_ell + 2.0),
            min=0.0,
        )
    )
    denominator = torch.where(
        ell >= 2, normalization_two, torch.ones_like(normalization_two)
    )
    chi_maps = torch.stack(
        tuple(
            synthesize(
                source_alms[..., component, :] * normalization_two,
                field.geometry,
                lmax=field.lmax,
                backend=field.backend,
                latitude_block_size=field.latitude_block_size,
            ).values
            for component in range(2)
        ),
        dim=-3,
    )
    result = torch.stack(
        tuple(
            analyze(
                chi_maps[..., component, :, :] * apodized.values,
                field.geometry,
                lmax=field.lmax,
                backend=field.backend,
                latitude_block_size=field.latitude_block_size,
            ).values
            / denominator
            for component in range(2)
        ),
        dim=-2,
    )
    return torch.where((ell >= 2).view(1, -1), result, torch.zeros_like(result))


def template_deprojection_bias(
    field_a: PseudoClField,
    signal_realizations_a: Tensor,
    field_b: PseudoClField | None = None,
    signal_realizations_b: Tensor | None = None,
) -> Tensor:
    """Estimate the additive template-deprojection bias from signal realizations.

    Realizations are interpreted as already beam/pixel-convolved maps and must
    have a leading realization axis.  The returned coupled-spectrum bias is the
    ensemble mean of deprojected minus undeprojected estimates in the standard
    block ordering.
    """

    other = field_a if field_b is None else field_b
    simulations_a = torch.as_tensor(
        signal_realizations_a, dtype=field_a.values.dtype, device=field_a.values.device
    )
    map_ndim_a = 2 if field_a.spin == 0 else 3
    if simulations_a.ndim != map_ndim_a + 1:
        raise ValueError("signal_realizations_a must have one leading realization axis")
    simulations_b = simulations_a
    if field_b is not None:
        if signal_realizations_b is None:
            raise ValueError("signal_realizations_b is required for a cross-field bias")
        simulations_b = torch.as_tensor(
            signal_realizations_b, dtype=other.values.dtype, device=other.values.device
        )
    elif signal_realizations_b is not None:
        raise ValueError("signal_realizations_b requires field_b")
    map_ndim_b = 2 if other.spin == 0 else 3
    if simulations_b.ndim != map_ndim_b + 1:
        raise ValueError("signal_realizations_b must have one leading realization axis")
    if int(simulations_a.shape[0]) != int(simulations_b.shape[0]):
        raise ValueError("signal realization counts must match")

    def with_values(
        source: PseudoClField, values: Tensor, templates: Tensor | None
    ) -> PseudoClField:
        return PseudoClField(
            values,
            source.mask,
            source.geometry,
            source.lmax,
            spin=source.spin,
            beam=source.beam,
            pixel_window=source.pixel_window,
            templates=templates,
            maximum_template_condition_number=source.maximum_template_condition_number,
            apodized_mask=source.apodized_mask,
            purify_e=source.purify_e,
            purify_b=source.purify_b,
            backend=source.backend,
            latitude_block_size=source.latitude_block_size,
        )

    cleaned_a = with_values(field_a, simulations_a, field_a.templates)
    cleaned_b = (
        cleaned_a
        if field_b is None
        else with_values(other, simulations_b, other.templates)
    )
    raw_a = with_values(field_a, simulations_a, None)
    raw_b = raw_a if field_b is None else with_values(other, simulations_b, None)
    cleaned = coupled_spectra(cleaned_a, None if field_b is None else cleaned_b)
    raw = coupled_spectra(raw_a, None if field_b is None else raw_b)
    return (cleaned - raw).mean(dim=0)


def _packed_cross_spectrum(left: Tensor, right: Tensor, lmax: int) -> Tensor:
    if int(left.shape[-1]) != _spectral.alm_size(lmax):
        raise ValueError("left alm shape does not match lmax")
    if int(right.shape[-1]) != _spectral.alm_size(lmax):
        raise ValueError("right alm shape does not match lmax")
    left, right = torch.broadcast_tensors(left, right)
    ell = _spectral._alm_ell_array(lmax, lmax).to(left.device)
    m = _spectral._alm_m_array(lmax, lmax).to(left.device)
    gain = torch.where(m == 0, 1.0, 2.0).to(dtype=left.real.dtype)
    product = torch.real(left * torch.conj(right)) * gain
    index = ell.view(*((1,) * (product.ndim - 1)), -1).expand_as(product)
    result = torch.zeros(
        (*product.shape[:-1], lmax + 1), dtype=product.dtype, device=product.device
    ).scatter_add(-1, index, product)
    normalization = 2 * torch.arange(lmax + 1, device=product.device) + 1
    return result / normalization.to(product.dtype)


def coupled_spectra(
    field_a: PseudoClField, field_b: PseudoClField | None = None
) -> Tensor:
    """Return scalar, ``(E, B)``, or ``(EE, EB, BE, BB)`` coupled spectra."""

    other = field_a if field_b is None else field_b
    if field_a.geometry != other.geometry:
        raise ValueError("fields must use the same CAR geometry")
    if field_a.lmax != other.lmax:
        raise ValueError("fields must use the same lmax")
    left = field_a.masked_alms()
    right = left if field_b is None else other.masked_alms()
    ll = field_a.lmax
    if field_a.spin == 0 and other.spin == 0:
        return _packed_cross_spectrum(left, right, ll)
    if field_a.spin == 0:
        return torch.stack(
            tuple(
                _packed_cross_spectrum(left, right[..., component, :], ll)
                for component in range(2)
            ),
            dim=-2,
        )
    if other.spin == 0:
        return torch.stack(
            tuple(
                _packed_cross_spectrum(left[..., component, :], right, ll)
                for component in range(2)
            ),
            dim=-2,
        )
    return torch.stack(
        tuple(
            _packed_cross_spectrum(left[..., first, :], right[..., second, :], ll)
            for first, second in ((0, 0), (0, 1), (1, 0), (1, 1))
        ),
        dim=-2,
    )


@dataclass(frozen=True, slots=True)
class BandpowerBinning:
    """Paired ell-to-bandpower and bandpower-to-ell linear operators."""

    binning_matrix: Tensor
    unbinning_matrix: Tensor

    def __post_init__(self) -> None:
        binning = torch.as_tensor(self.binning_matrix)
        unbinning = torch.as_tensor(
            self.unbinning_matrix, dtype=binning.dtype, device=binning.device
        )
        _require_real_floating(binning, "binning_matrix")
        if binning.ndim != 2 or unbinning.ndim != 2:
            raise ValueError("binning matrices must be two-dimensional")
        if int(binning.shape[0]) < 1 or int(binning.shape[1]) < 1:
            raise ValueError(
                "binning matrices must contain at least one band and multipole"
            )
        if int(binning.shape[1]) != int(unbinning.shape[0]):
            raise ValueError("binning and unbinning ell dimensions differ")
        if int(binning.shape[0]) != int(unbinning.shape[1]):
            raise ValueError("binning and unbinning band dimensions differ")
        identity = binning @ unbinning
        expected = torch.eye(
            int(binning.shape[0]), dtype=binning.dtype, device=binning.device
        )
        tolerance = (
            32.0 * torch.finfo(binning.dtype).eps * max(1, int(binning.shape[1]))
        )
        if not bool(torch.allclose(identity, expected, rtol=tolerance, atol=tolerance)):
            raise ValueError("binning_matrix @ unbinning_matrix must be identity")
        object.__setattr__(self, "binning_matrix", binning)
        object.__setattr__(self, "unbinning_matrix", unbinning)

    @property
    def lmax(self) -> int:
        return int(self.binning_matrix.shape[1]) - 1

    @property
    def nband(self) -> int:
        return int(self.binning_matrix.shape[0])

    @classmethod
    def from_edges(
        cls,
        edges: list[int] | tuple[int, ...],
        lmax: int,
        *,
        weighting: Literal["uniform", "2ell+1"] = "uniform",
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> BandpowerBinning:
        """Construct non-overlapping half-open bands from integer edges."""

        normalized = tuple(int(edge) for edge in edges)
        if len(normalized) < 2 or any(
            stop <= start
            for start, stop in zip(normalized[:-1], normalized[1:], strict=True)
        ):
            raise ValueError("edges must be a strictly increasing sequence")
        if normalized[0] < 0 or normalized[-1] > int(lmax) + 1:
            raise ValueError("edges must lie within [0, lmax + 1]")
        if dtype not in {torch.float32, torch.float64}:
            raise TypeError("dtype must be float32 or float64")
        if weighting not in {"uniform", "2ell+1"}:
            raise ValueError("weighting must be 'uniform' or '2ell+1'")
        nband = len(normalized) - 1
        binning = torch.zeros((nband, int(lmax) + 1), dtype=dtype, device=device)
        unbinning = torch.zeros((int(lmax) + 1, nband), dtype=dtype, device=device)
        ell = torch.arange(int(lmax) + 1, dtype=dtype, device=device)
        base = torch.ones_like(ell) if weighting == "uniform" else 2.0 * ell + 1.0
        for band, (start, stop) in enumerate(
            zip(normalized[:-1], normalized[1:], strict=True)
        ):
            weights = base[start:stop]
            binning[band, start:stop] = weights / weights.sum()
            unbinning[start:stop, band] = 1.0
        return cls(binning, unbinning)

    def bin(self, spectra: Tensor) -> Tensor:
        if int(spectra.shape[-1]) != self.lmax + 1:
            raise ValueError(f"spectra must end in shape ({self.lmax + 1},)")
        matrix = self.binning_matrix.to(dtype=spectra.dtype, device=spectra.device)
        return torch.einsum("bl,...l->...b", matrix, spectra)

    def unbin(self, bandpowers: Tensor) -> Tensor:
        if int(bandpowers.shape[-1]) != self.nband:
            raise ValueError(f"bandpowers must end in shape ({self.nband},)")
        matrix = self.unbinning_matrix.to(
            dtype=bandpowers.dtype, device=bandpowers.device
        )
        return torch.einsum("lb,...b->...l", matrix, bandpowers)


@dataclass(frozen=True, slots=True)
class ModeCouplingDiagnostics:
    rank: int
    size: int
    condition_number: float
    maximum_condition_number: float
    singular_values: Tensor


def _gauss_legendre(
    order: int, *, dtype: torch.dtype, device: torch.device
) -> tuple[Tensor, Tensor]:
    index = torch.arange(order, dtype=dtype, device=device)
    nodes = torch.cos(math.pi * (index + 0.75) / (order + 0.5))
    previous = torch.ones_like(nodes)
    current = nodes
    derivative = torch.ones_like(nodes)
    for _ in range(12):
        previous = torch.ones_like(nodes)
        current = nodes
        if order == 1:
            current = nodes
        else:
            for ell in range(2, order + 1):
                following = (
                    (2 * ell - 1) * nodes * current - (ell - 1) * previous
                ) / ell
                previous, current = current, following
        derivative = order * (nodes * current - previous) / (nodes * nodes - 1.0)
        nodes = nodes - current / derivative
    previous = torch.ones_like(nodes)
    current = nodes
    if order > 1:
        for ell in range(2, order + 1):
            following = ((2 * ell - 1) * nodes * current - (ell - 1) * previous) / ell
            previous, current = current, following
    derivative = order * (nodes * current - previous) / (nodes * nodes - 1.0)
    weights = 2.0 / ((1.0 - nodes * nodes) * derivative * derivative)
    return nodes, weights


def _legendre_values(lmax: int, nodes: Tensor) -> Tensor:
    values = [torch.ones_like(nodes)]
    if lmax == 0:
        return torch.stack(values)
    values.append(nodes)
    for ell in range(2, lmax + 1):
        values.append(
            ((2 * ell - 1) * nodes * values[-1] - (ell - 1) * values[-2]) / ell
        )
    return torch.stack(values)


def _wigner_values(
    lmax: int, nodes: Tensor, first_spin: int, second_spin: int
) -> Tensor:
    values = torch.zeros(
        (lmax + 1, nodes.numel()), dtype=nodes.dtype, device=nodes.device
    )
    ell_min = max(abs(first_spin), abs(second_spin))
    if ell_min <= lmax:
        values[ell_min:] = _wigner_sequence(
            ell_min,
            lmax,
            first_spin,
            second_spin,
            torch.acos(nodes),
        )
    return values


def _basis_values(lmax: int, nodes: Tensor, channel: str) -> Tensor:
    if channel == "scalar":
        return _legendre_values(lmax, nodes)
    if channel == "mixed":
        return _wigner_values(lmax, nodes, 0, 2)
    if channel == "spin-plus":
        return _wigner_values(lmax, nodes, 2, 2)
    if channel == "spin-minus":
        return _wigner_values(lmax, nodes, 2, -2)
    raise ValueError(f"unknown coupling channel {channel!r}")


def _channel_action(
    spectra: Tensor,
    nodes: Tensor,
    weights: Tensor,
    mask_correlation: Tensor,
    transfer: Tensor,
    node_block_size: int,
    channel: str,
) -> Tensor:
    lmax = int(transfer.shape[0]) - 1
    if int(spectra.shape[-1]) != lmax + 1:
        raise ValueError(f"spectra must end in shape ({lmax + 1},)")
    values = spectra.to(dtype=transfer.dtype, device=transfer.device)
    ell = torch.arange(lmax + 1, dtype=values.dtype, device=values.device)
    scaled = values * transfer * (ell + 0.5)
    result = torch.zeros_like(values)
    for start in range(0, int(nodes.numel()), node_block_size):
        stop = min(start + node_block_size, int(nodes.numel()))
        basis = _basis_values(lmax, nodes[start:stop], channel)
        correlation = mask_correlation[start:stop] * weights[start:stop]
        synthesized = torch.einsum("...l,ln->...n", scaled, basis)
        result = result + torch.einsum(
            "...n,ln->...l", synthesized * correlation, basis
        )
    return result


def _channel_adjoint(
    spectra: Tensor,
    nodes: Tensor,
    weights: Tensor,
    mask_correlation: Tensor,
    transfer: Tensor,
    node_block_size: int,
    channel: str,
) -> Tensor:
    lmax = int(transfer.shape[0]) - 1
    if int(spectra.shape[-1]) != lmax + 1:
        raise ValueError(f"spectra must end in shape ({lmax + 1},)")
    values = spectra.to(dtype=transfer.dtype, device=transfer.device)
    result = torch.zeros_like(values)
    for start in range(0, int(nodes.numel()), node_block_size):
        stop = min(start + node_block_size, int(nodes.numel()))
        basis = _basis_values(lmax, nodes[start:stop], channel)
        correlation = mask_correlation[start:stop] * weights[start:stop]
        synthesized = torch.einsum("...l,ln->...n", values, basis)
        result = result + torch.einsum(
            "...n,ln->...l", synthesized * correlation, basis
        )
    ell = torch.arange(lmax + 1, dtype=values.dtype, device=values.device)
    return result * transfer * (ell + 0.5)


def _block_size(spins: tuple[int, int]) -> int:
    if spins == (0, 0):
        return 1
    if 0 in spins:
        return 2
    return 4


def _coupling_action(
    spectra: Tensor,
    nodes: Tensor,
    weights: Tensor,
    mask_correlation: Tensor,
    transfer: Tensor,
    node_block_size: int,
    spins: tuple[int, int],
) -> Tensor:
    block_size = _block_size(spins)
    if block_size == 1:
        return _channel_action(
            spectra,
            nodes,
            weights,
            mask_correlation,
            transfer,
            node_block_size,
            "scalar",
        )
    if spectra.ndim < 2 or int(spectra.shape[-2]) != block_size:
        raise ValueError(
            f"spectra must end in shape ({block_size}, {transfer.shape[0]})"
        )
    if block_size == 2:
        return _channel_action(
            spectra,
            nodes,
            weights,
            mask_correlation,
            transfer,
            node_block_size,
            "mixed",
        )
    ee, eb, be, bb = spectra.unbind(dim=-2)
    plus_even = _channel_action(
        ee + bb,
        nodes,
        weights,
        mask_correlation,
        transfer,
        node_block_size,
        "spin-plus",
    )
    minus_even = _channel_action(
        ee - bb,
        nodes,
        weights,
        mask_correlation,
        transfer,
        node_block_size,
        "spin-minus",
    )
    plus_odd = _channel_action(
        be - eb,
        nodes,
        weights,
        mask_correlation,
        transfer,
        node_block_size,
        "spin-plus",
    )
    minus_odd = _channel_action(
        be + eb,
        nodes,
        weights,
        mask_correlation,
        transfer,
        node_block_size,
        "spin-minus",
    )
    return torch.stack(
        (
            0.5 * (plus_even + minus_even),
            0.5 * (minus_odd - plus_odd),
            0.5 * (plus_odd + minus_odd),
            0.5 * (plus_even - minus_even),
        ),
        dim=-2,
    )


def _coupling_adjoint(
    spectra: Tensor,
    nodes: Tensor,
    weights: Tensor,
    mask_correlation: Tensor,
    transfer: Tensor,
    node_block_size: int,
    spins: tuple[int, int],
) -> Tensor:
    block_size = _block_size(spins)
    if block_size == 1:
        return _channel_adjoint(
            spectra,
            nodes,
            weights,
            mask_correlation,
            transfer,
            node_block_size,
            "scalar",
        )
    if spectra.ndim < 2 or int(spectra.shape[-2]) != block_size:
        raise ValueError(
            f"spectra must end in shape ({block_size}, {transfer.shape[0]})"
        )
    if block_size == 2:
        return _channel_adjoint(
            spectra,
            nodes,
            weights,
            mask_correlation,
            transfer,
            node_block_size,
            "mixed",
        )
    out_ee, out_eb, out_be, out_bb = spectra.unbind(dim=-2)

    def plus(value: Tensor) -> Tensor:
        return _channel_adjoint(
            value,
            nodes,
            weights,
            mask_correlation,
            transfer,
            node_block_size,
            "spin-plus",
        )

    def minus(value: Tensor) -> Tensor:
        return _channel_adjoint(
            value,
            nodes,
            weights,
            mask_correlation,
            transfer,
            node_block_size,
            "spin-minus",
        )

    mplus_ee = 0.5 * (plus(out_ee) + minus(out_ee))
    mminus_ee = 0.5 * (plus(out_ee) - minus(out_ee))
    mplus_eb = 0.5 * (plus(out_eb) + minus(out_eb))
    mminus_eb = 0.5 * (plus(out_eb) - minus(out_eb))
    mplus_be = 0.5 * (plus(out_be) + minus(out_be))
    mminus_be = 0.5 * (plus(out_be) - minus(out_be))
    mplus_bb = 0.5 * (plus(out_bb) + minus(out_bb))
    mminus_bb = 0.5 * (plus(out_bb) - minus(out_bb))
    return torch.stack(
        (
            mplus_ee + mminus_bb,
            mplus_eb - mminus_be,
            mplus_be - mminus_eb,
            mminus_ee + mplus_bb,
        ),
        dim=-2,
    )


@dataclass(frozen=True, slots=True)
class ModeCouplingWorkspace:
    """Exact correlation-space block coupling and binned inversion."""

    lmax: int
    spins: tuple[int, int]
    spectrum_block_size: int
    mask_spectrum: Tensor
    transfer: Tensor
    binning: BandpowerBinning
    quadrature_nodes: Tensor
    quadrature_weights: Tensor
    mask_correlation: Tensor
    coupling_matrix: Tensor
    diagnostics: ModeCouplingDiagnostics
    node_block_size: int

    @classmethod
    def from_fields(
        cls,
        field_a: PseudoClField,
        field_b: PseudoClField | None = None,
        binning: BandpowerBinning | None = None,
        *,
        maximum_condition_number: float | None = None,
        node_block_size: int = 256,
    ) -> ModeCouplingWorkspace:
        """Build a workspace without materializing an ell-by-ell kernel."""

        other = field_a if field_b is None else field_b
        if binning is None:
            raise ValueError("binning is required")
        if field_a.geometry != other.geometry:
            raise ValueError("fields must use the same CAR geometry")
        if field_a.lmax != other.lmax:
            raise ValueError("fields must use the same lmax")
        if binning.lmax != field_a.lmax:
            raise ValueError("binning lmax must match field lmax")
        if node_block_size < 1:
            raise ValueError("node_block_size must be positive")
        if field_a.purify_e or field_a.purify_b or other.purify_e or other.purify_b:
            raise NotImplementedError(
                "pure-estimator coupling workspaces require derivative-mask kernels"
            )
        lmax = field_a.lmax
        spins = (field_a.spin, other.spin)
        spectrum_block_size = _block_size(spins)
        ell_min = max(spins)
        if ell_min:
            low_binning = binning.binning_matrix[:, :ell_min]
            low_unbinning = binning.unbinning_matrix[:ell_min]
            if bool(torch.any(low_binning != 0).item()) or bool(
                torch.any(low_unbinning != 0).item()
            ):
                raise ValueError(
                    f"spin coupling binning must exclude multipoles below {ell_min}"
                )
        mask_lmax = 2 * lmax
        geometry = field_a.geometry
        if geometry.nlat < mask_lmax + 1 or geometry.nlon < 2 * mask_lmax + 1:
            raise ValueError(
                "exact scalar coupling requires nlat >= 2*lmax + 1 and nlon >= 4*lmax + 1"
            )
        mask_alm_a = analyze(
            field_a.mask,
            geometry,
            lmax=mask_lmax,
            backend=field_a.backend,
            latitude_block_size=field_a.latitude_block_size,
        ).values
        mask_alm_b = (
            mask_alm_a
            if field_b is None
            else analyze(
                other.mask,
                geometry,
                lmax=mask_lmax,
                backend=other.backend,
                latitude_block_size=other.latitude_block_size,
            ).values
        )
        mask_spectrum = _packed_cross_spectrum(mask_alm_a, mask_alm_b, mask_lmax)
        dtype = field_a.values.dtype
        device = field_a.values.device
        degree = 4 * lmax
        quadrature_order = max(1, (degree + 2) // 2)
        nodes, weights = _gauss_legendre(quadrature_order, dtype=dtype, device=device)
        mask_basis = _legendre_values(mask_lmax, nodes)
        mask_ell = torch.arange(mask_lmax + 1, dtype=dtype, device=device)
        mask_correlation = torch.einsum(
            "l,ln->n",
            mask_spectrum * (2.0 * mask_ell + 1.0) / (4.0 * math.pi),
            mask_basis,
        )
        transfer = field_a.transfer * other.transfer
        binning_matrix = binning.binning_matrix.to(dtype=dtype, device=device)
        unbinning_matrix = binning.unbinning_matrix.to(dtype=dtype, device=device)
        if spectrum_block_size == 1:
            unbinned_columns = unbinning_matrix.transpose(0, 1)
        else:
            identity = torch.eye(
                spectrum_block_size * binning.nband, dtype=dtype, device=device
            ).reshape(
                spectrum_block_size * binning.nband,
                spectrum_block_size,
                binning.nband,
            )
            unbinned_columns = torch.einsum("lb,icb->icl", unbinning_matrix, identity)
        coupled_columns = _coupling_action(
            unbinned_columns,
            nodes,
            weights,
            mask_correlation,
            transfer,
            int(node_block_size),
            spins,
        )
        if spectrum_block_size == 1:
            binned_columns = torch.einsum("bl,il->ib", binning_matrix, coupled_columns)
        else:
            binned_columns = torch.einsum(
                "bl,icl->icb", binning_matrix, coupled_columns
            ).reshape(spectrum_block_size * binning.nband, -1)
        coupling_matrix = binned_columns.transpose(0, 1)
        rank, condition, condition_limit, singular_values = _condition_diagnostics(
            coupling_matrix, maximum_condition_number
        )
        diagnostics = ModeCouplingDiagnostics(
            rank=rank,
            size=int(coupling_matrix.shape[0]),
            condition_number=condition,
            maximum_condition_number=condition_limit,
            singular_values=singular_values.detach().to(device=device).clone(),
        )
        if rank != diagnostics.size:
            raise ModeCouplingError(
                f"binned coupling matrix is rank deficient: rank {rank} < {diagnostics.size}"
            )
        if not math.isfinite(condition) or condition > condition_limit:
            raise ModeCouplingError(
                f"binned coupling matrix condition number {condition:.6g} exceeds {condition_limit:.6g}"
            )
        return cls(
            lmax=lmax,
            spins=spins,
            spectrum_block_size=spectrum_block_size,
            mask_spectrum=mask_spectrum,
            transfer=transfer,
            binning=BandpowerBinning(binning_matrix, unbinning_matrix),
            quadrature_nodes=nodes,
            quadrature_weights=weights,
            mask_correlation=mask_correlation,
            coupling_matrix=coupling_matrix,
            diagnostics=diagnostics,
            node_block_size=int(node_block_size),
        )

    def couple_theory(self, theory_spectrum: Tensor) -> Tensor:
        """Apply mask, beam, and pixel-window coupling to an unbinned theory spectrum."""

        return _coupling_action(
            theory_spectrum,
            self.quadrature_nodes,
            self.quadrature_weights,
            self.mask_correlation,
            self.transfer,
            self.node_block_size,
            self.spins,
        )

    def couple_bandpowers(self, theory_bandpowers: Tensor) -> Tensor:
        """Couple piecewise-unbinned theory bandpowers and return binned pseudo-spectra."""

        return self.binning.bin(
            self.couple_theory(self.binning.unbin(theory_bandpowers))
        )

    def couple_noise(self, noise_spectrum: Tensor) -> Tensor:
        """Propagate an unbinned sky-noise spectrum through all coupling blocks."""

        return self.couple_theory(noise_spectrum)

    def decouple(
        self,
        coupled_spectrum: Tensor,
        *,
        noise_bias: Tensor | None = None,
        deprojection_bias: Tensor | None = None,
    ) -> Tensor:
        """Subtract an unbinned coupled noise bias and solve for bandpowers."""

        expected = (self.lmax + 1,)
        if self.spectrum_block_size > 1:
            expected = (self.spectrum_block_size, self.lmax + 1)
        if tuple(coupled_spectrum.shape[-len(expected) :]) != expected:
            raise ValueError(f"coupled_spectrum must end in shape {expected}")
        residual = coupled_spectrum
        for name, supplied_bias in (
            ("noise_bias", noise_bias),
            ("deprojection_bias", deprojection_bias),
        ):
            if supplied_bias is None:
                continue
            bias = torch.as_tensor(
                supplied_bias,
                dtype=coupled_spectrum.dtype,
                device=coupled_spectrum.device,
            )
            if bias.shape != coupled_spectrum.shape:
                raise ValueError(f"{name} must have the same shape as coupled_spectrum")
            residual = residual - bias
        binned = self.binning.bin(residual)
        matrix = self.coupling_matrix.to(dtype=binned.dtype, device=binned.device)
        output_size = self.spectrum_block_size * self.binning.nband
        batch_shape = binned.shape[:-1]
        if self.spectrum_block_size > 1:
            batch_shape = binned.shape[:-2]
        flat = binned.reshape(-1, output_size)
        solved = torch.linalg.solve(matrix, flat.transpose(0, 1)).transpose(0, 1)
        if self.spectrum_block_size == 1:
            return solved.reshape(*batch_shape, self.binning.nband)
        return solved.reshape(
            *batch_shape, self.spectrum_block_size, self.binning.nband
        )

    def bandpower_windows(self) -> Tensor:
        """Return the exact linear map from unbinned theory C_ell to estimates."""

        probes: Tensor
        if self.spectrum_block_size == 1:
            probes = self.binning.binning_matrix
        else:
            identity = torch.eye(
                self.spectrum_block_size,
                dtype=self.binning.binning_matrix.dtype,
                device=self.binning.binning_matrix.device,
            )
            probes = torch.einsum(
                "oc,bl->obcl", identity, self.binning.binning_matrix
            ).reshape(
                self.spectrum_block_size * self.binning.nband,
                self.spectrum_block_size,
                self.lmax + 1,
            )
        rows = _coupling_adjoint(
            probes,
            self.quadrature_nodes,
            self.quadrature_weights,
            self.mask_correlation,
            self.transfer,
            self.node_block_size,
            self.spins,
        )
        flat_rows = rows.reshape(
            self.spectrum_block_size * self.binning.nband,
            self.spectrum_block_size * (self.lmax + 1),
        )
        solved = torch.linalg.solve(self.coupling_matrix, flat_rows)
        if self.spectrum_block_size == 1:
            return solved
        return solved.reshape(
            self.spectrum_block_size,
            self.binning.nband,
            self.spectrum_block_size,
            self.lmax + 1,
        )


__all__ = [
    "ApodizedMask",
    "BandpowerBinning",
    "ModeCouplingDiagnostics",
    "ModeCouplingError",
    "ModeCouplingWorkspace",
    "PseudoClField",
    "coupled_spectra",
    "pure_alms",
    "template_deprojection_bias",
]
