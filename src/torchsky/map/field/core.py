"""Experimental continuous field models for multiwavelength map fitting."""

from __future__ import annotations

import copy
import functools
import inspect
import math
from collections.abc import Iterator
from typing import Literal, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ... import WCS
from ...device import default_compute_device, resolve_fit_device
from ...encoders import SkyCoordinateEncoder, sky_encoder_fourier_min_frequency
from ...spectral import SpectralAxis
from ...wcs import healpix as _healpix_wcs

from ..healpix_image_plane import sample_healpix_map_on_image_header
from ..operators.bundle import field_observation_operator_bundle
from ..operators.background import (
    apply_background_components,
    observation_background_components,
)
from ..operators.psf import apply_psf_spec, gaussian_blur as _shared_gaussian_blur
from ..operators.spectral import (
    bandpass_bounds as _shared_bandpass_bounds,
    bandpass_mc_samples as _shared_bandpass_mc_samples,
    bandpass_quadrature as _shared_bandpass_quadrature,
    bandpass_response as _shared_bandpass_response,
    mixture_importance_weights as _shared_mixture_importance_weights,
    mixture_mc_samples as _shared_mixture_mc_samples,
)
from ..solvers import ComponentDecompositionDiagnostics, SurveyInstrumentComponentMixer

from .types import (
    FieldComponentAnchorTerm,
    FieldCentroidAnchorTerm,
    FieldCoordinateMode,
    FieldCurriculum,
    FieldDomain,
    FieldFitResult,
    FieldLossKind,
    FieldLossReduction,
    FieldLRSchedule,
    FieldModelKind,
    FieldNoisePretrainKind,
    FieldObservation,
    FieldObservationSampling,
    FieldSpectralSampling,
    ForwardContext,
)


@functools.lru_cache(maxsize=512)
def _type_method_accepts_kwarg(cls: type, method_name: str, kw: str) -> bool:
    fn = getattr(cls, method_name, None)
    if fn is None:
        return False
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    if kw in sig.parameters:
        return True
    return any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )


def _call_scalar_field_forward(
    model: nn.Module,
    ra: Tensor,
    dec: Tensor,
    wavelength: Tensor | float,
    *,
    forward_context: ForwardContext | None,
) -> Tensor:
    if forward_context is None:
        return model(ra, dec, wavelength)
    if _type_method_accepts_kwarg(type(model), "forward", "forward_context"):
        return model(ra, dec, wavelength, forward_context=forward_context)
    return model(ra, dec, wavelength)


def _call_render_bandpass(
    model: nn.Module,
    ra: Tensor,
    dec: Tensor,
    wavelengths: Tensor,
    weights: Tensor,
    *,
    forward_context: ForwardContext | None,
) -> Tensor:
    rb = model.render_bandpass
    if forward_context is None:
        return rb(ra, dec, wavelengths, weights)
    if _type_method_accepts_kwarg(type(model), "render_bandpass", "forward_context"):
        return rb(ra, dec, wavelengths, weights, forward_context=forward_context)
    return rb(ra, dec, wavelengths, weights)


def _call_render_bandpass_distribution(
    model: nn.Module,
    ra: Tensor,
    dec: Tensor,
    wavelengths: Tensor,
    weights: Tensor,
    *,
    forward_context: ForwardContext | None,
) -> tuple[Tensor, Tensor]:
    rb = model.render_bandpass_distribution
    if forward_context is None:
        return rb(ra, dec, wavelengths, weights)
    if _type_method_accepts_kwarg(
        type(model), "render_bandpass_distribution", "forward_context"
    ):
        return rb(ra, dec, wavelengths, weights, forward_context=forward_context)
    return rb(ra, dec, wavelengths, weights)


def _call_forward_distribution(
    model: nn.Module,
    ra: Tensor,
    dec: Tensor,
    wavelength: Tensor | float,
    *,
    forward_context: ForwardContext | None,
) -> tuple[Tensor, Tensor]:
    fd = model.forward_distribution
    if forward_context is None:
        return fd(ra, dec, wavelength)
    if _type_method_accepts_kwarg(type(model), "forward_distribution", "forward_context"):
        return fd(ra, dec, wavelength, forward_context=forward_context)
    return fd(ra, dec, wavelength)


def _wrapped_delta_ra_deg(ra_a_deg: Tensor, ra_b_deg: float) -> Tensor:
    return ((ra_a_deg - float(ra_b_deg) + 180.0) % 360.0) - 180.0


def _header_shape(header: dict, *, fallback: tuple[int, int]) -> tuple[int, int]:
    naxis1 = header.get("NAXIS1")
    naxis2 = header.get("NAXIS2")
    if naxis1 is None or naxis2 is None:
        return fallback
    return int(naxis2), int(naxis1)


def _copy_header_with_crop(
    header: dict,
    *,
    x0: int,
    y0: int,
    width: int,
    height: int,
) -> dict:
    cropped = dict(header)
    cropped["NAXIS1"] = int(width)
    cropped["NAXIS2"] = int(height)
    if "CRPIX1" in cropped:
        cropped["CRPIX1"] = float(cropped["CRPIX1"]) - float(x0)
    if "CRPIX2" in cropped:
        cropped["CRPIX2"] = float(cropped["CRPIX2"]) - float(y0)
    return cropped


def _observation_band_key(observation: FieldObservation) -> tuple[object, ...]:
    return (
        observation.band_name,
        observation.spectral_min,
        observation.spectral_max,
        observation.wavelength,
        observation.transmission_center,
    )


def _observation_support_mask(observation: FieldObservation) -> Tensor:
    support = torch.isfinite(observation.image)
    if observation.weight is not None:
        support = support & torch.isfinite(observation.weight) & (observation.weight > 0)
    if observation.mask is not None:
        support = support & torch.isfinite(observation.mask) & (observation.mask > 0)
    return support


def _observation_weight_sum_scalar(observation: FieldObservation) -> float:
    if observation.weight is not None:
        weight = observation.weight.detach().to(dtype=torch.float32)
    else:
        weight = torch.ones_like(observation.image, dtype=torch.float32)
    if observation.mask is not None:
        valid = observation.mask.detach() > 0
        weight = torch.where(valid, weight, torch.zeros_like(weight))
    return float(weight.sum().item())


def _observation_support_count_scalar(observation: FieldObservation) -> int:
    support = _observation_support_mask(observation)
    return int(torch.count_nonzero(support).item())


def _sample_observation_indices(
    observations: Sequence[FieldObservation],
    *,
    observations_per_step: int | None,
    observation_sampling: FieldObservationSampling,
    generator: torch.Generator,
) -> list[int]:
    total = len(observations)
    if observations_per_step is None or observations_per_step <= 0 or observations_per_step >= total:
        return list(range(total))
    target = int(observations_per_step)
    if observation_sampling == "uniform":
        return torch.randperm(total, generator=generator)[:target].tolist()
    if observation_sampling != "band_balanced":
        raise ValueError(f"unsupported observation_sampling {observation_sampling!r}")
    band_groups: dict[tuple[object, ...], list[int]] = {}
    for idx, observation in enumerate(observations):
        band_groups.setdefault(_observation_band_key(observation), []).append(idx)
    ordered_groups = list(band_groups.values())
    if not ordered_groups:
        return list(range(total))
    selected: list[int] = []
    per_group = max(target // len(ordered_groups), 1)
    remainders: list[int] = []
    for group in ordered_groups:
        perm = torch.randperm(len(group), generator=generator).tolist()
        take = min(len(group), per_group)
        selected.extend(group[i] for i in perm[:take])
        remainders.extend(group[i] for i in perm[take:])
    if len(selected) < target and remainders:
        extra = torch.randperm(len(remainders), generator=generator)[: target - len(selected)].tolist()
        selected.extend(remainders[i] for i in extra)
    elif len(selected) > target:
        keep = torch.randperm(len(selected), generator=generator)[:target].tolist()
        selected = [selected[i] for i in keep]
    return selected


def _sample_patch_origin(
    observation: FieldObservation,
    *,
    patch_size: int,
    generator: torch.Generator,
) -> tuple[int, int]:
    image = observation.image
    height, width = int(image.shape[-2]), int(image.shape[-1])
    patch = min(int(patch_size), height, width)
    max_y = max(height - patch, 0)
    max_x = max(width - patch, 0)
    support = _observation_support_mask(observation)
    support_indices = torch.nonzero(support, as_tuple=False)
    if support_indices.numel() > 0:
        anchor_index = int(torch.randint(support_indices.shape[0], (1,), generator=generator).item())
        anchor_y = int(support_indices[anchor_index, 0].item())
        anchor_x = int(support_indices[anchor_index, 1].item())
        y0 = min(max(anchor_y - patch // 2, 0), max_y)
        x0 = min(max(anchor_x - patch // 2, 0), max_x)
        return x0, y0
    x0 = int(torch.randint(max_x + 1, (1,), generator=generator).item()) if max_x > 0 else 0
    y0 = int(torch.randint(max_y + 1, (1,), generator=generator).item()) if max_y > 0 else 0
    return x0, y0


def _crop_field_observation(
    observation: FieldObservation,
    *,
    x0: int,
    y0: int,
    width: int,
    height: int,
) -> FieldObservation:
    y_slice = slice(int(y0), int(y0) + int(height))
    x_slice = slice(int(x0), int(x0) + int(width))
    return FieldObservation(
        image=observation.image[y_slice, x_slice],
        header=_copy_header_with_crop(
            observation.header,
            x0=int(x0),
            y0=int(y0),
            width=int(width),
            height=int(height),
        ),
        weight=(
            observation.weight[y_slice, x_slice]
            if observation.weight is not None
            else None
        ),
        mask=(
            observation.mask[y_slice, x_slice]
            if observation.mask is not None
            else None
        ),
        wavelength=observation.wavelength,
        spectral_min=observation.spectral_min,
        spectral_max=observation.spectral_max,
        transmission_center=observation.transmission_center,
        transmission_sigma=observation.transmission_sigma,
        transmission_wavelength=observation.transmission_wavelength,
        transmission_throughput=observation.transmission_throughput,
        psf_sigma_pix=observation.psf_sigma_pix,
        psf_spec=observation.psf_spec,
        background_offset=observation.background_offset,
        band_name=observation.band_name,
        spectral_quantity=observation.spectral_quantity,
    )


def _crop_field_observation_to_shared_sky_patch(
    observation: FieldObservation,
    *,
    reference_observation: FieldObservation,
    x0: int,
    y0: int,
    patch_size: int,
) -> FieldObservation | None:
    patch = max(int(patch_size), 1)
    ref_width = int(reference_observation.image.shape[-1])
    ref_height = int(reference_observation.image.shape[-2])
    x1 = min(int(x0) + patch, ref_width)
    y1 = min(int(y0) + patch, ref_height)
    x0 = max(min(int(x0), x1 - 1), 0)
    y0 = max(min(int(y0), y1 - 1), 0)
    if observation is reference_observation:
        return _crop_field_observation(
            observation,
            x0=x0,
            y0=y0,
            width=x1 - x0,
            height=y1 - y0,
        )
    ref_x = torch.tensor(
        [x0, x1 - 1, x0, x1 - 1, 0.5 * (x0 + x1 - 1)],
        dtype=torch.float64,
    )
    ref_y = torch.tensor(
        [y0, y0, y1 - 1, y1 - 1, 0.5 * (y0 + y1 - 1)],
        dtype=torch.float64,
    )
    ref_wcs = WCS(reference_observation.header)
    ra_deg, dec_deg = ref_wcs.pixel_to_world(ref_x, ref_y)
    obs_wcs = WCS(observation.header)
    px, py = obs_wcs.world_to_pixel(ra_deg, dec_deg)
    if not (torch.isfinite(px).any() and torch.isfinite(py).any()):
        return None
    valid = torch.isfinite(px) & torch.isfinite(py)
    if not torch.any(valid):
        return None
    px = px[valid]
    py = py[valid]
    height = int(observation.image.shape[-2])
    width = int(observation.image.shape[-1])
    crop_x0 = max(int(torch.floor(px.min()).item()), 0)
    crop_y0 = max(int(torch.floor(py.min()).item()), 0)
    crop_x1 = min(int(torch.ceil(px.max()).item()) + 1, width)
    crop_y1 = min(int(torch.ceil(py.max()).item()) + 1, height)
    if crop_x1 <= crop_x0 or crop_y1 <= crop_y0:
        return None
    cropped = _crop_field_observation(
        observation,
        x0=crop_x0,
        y0=crop_y0,
        width=crop_x1 - crop_x0,
        height=crop_y1 - crop_y0,
    )
    if not torch.any(_observation_support_mask(cropped)):
        return None
    return cropped


def _sample_observation_pixels(
    observation: FieldObservation,
    *,
    num_samples: int,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor, int, float] | None:
    support = _observation_support_mask(observation)
    support_indices = torch.nonzero(support, as_tuple=False)
    if support_indices.numel() == 0:
        return None
    support_count = int(support_indices.shape[0])
    sample_count = max(int(num_samples), 1)
    if support_count >= sample_count:
        sample_ids = torch.randperm(
            support_count,
            generator=generator,
        )[:sample_count]
    else:
        sample_ids = torch.randint(
            support_count,
            (sample_count,),
            generator=generator,
        )
    sampled = support_indices[sample_ids]
    y = sampled[:, 0].to(dtype=torch.float64, device=device)
    x = sampled[:, 1].to(dtype=torch.float64, device=device)
    ra_deg, dec_deg = WCS(observation.header).pixel_to_world(x, y)
    image = observation.image.to(device=device)[sampled[:, 0], sampled[:, 1]]
    if observation.weight is not None:
        weight = observation.weight.to(device=device)[sampled[:, 0], sampled[:, 1]]
    else:
        weight = torch.ones_like(image)
    return (
        ra_deg.to(dtype=torch.float32),
        dec_deg.to(dtype=torch.float32),
        image,
        weight.to(dtype=image.dtype),
        support_count,
        _observation_weight_sum_scalar(observation),
    )


def _avg_pool_map(image: Tensor, *, factor: int) -> Tensor:
    pool = max(int(factor), 1)
    if pool <= 1 or min(int(image.shape[-2]), int(image.shape[-1])) < pool:
        return image
    return F.avg_pool2d(
        image.unsqueeze(0).unsqueeze(0),
        kernel_size=pool,
        stride=pool,
    ).squeeze(0).squeeze(0)


def _coordinate_grid(
    header: dict,
    *,
    shape: tuple[int, int],
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    height, width = _header_shape(header, fallback=shape)
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float64, device=device),
        torch.arange(width, dtype=torch.float64, device=device),
        indexing="ij",
    )
    ra_deg, dec_deg = WCS(header).pixel_to_world(x, y)
    return ra_deg, dec_deg


def _gaussian_kernel1d(sigma_pix: float, *, device: torch.device) -> Tensor:
    radius = max(1, int(math.ceil(3.0 * float(sigma_pix))))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    kernel = torch.exp(-0.5 * torch.square(x / float(sigma_pix)))
    return kernel / torch.clamp(kernel.sum(), min=1e-8)


def _gaussian_blur(image: Tensor, sigma_pix: float) -> Tensor:
    return _shared_gaussian_blur(image, sigma_pix)


def _bandpass_bounds(observation: FieldObservation) -> tuple[float, float]:
    return _shared_bandpass_bounds(observation)


def _has_tabulated_transmission(observation: FieldObservation) -> bool:
    return (
        observation.transmission_wavelength is not None
        and observation.transmission_throughput is not None
    )


def _sorted_tabulated_transmission(
    observation: FieldObservation,
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Move the tabulated curve to ``device`` as float32, sorted by wavelength (GPU-friendly).

    Always applies :func:`torch.sort` so we never call Python ``bool`` on tensor reductions
    (which would force a device sync). Sorting a few hundred to a few thousand nodes is cheap
    relative to field rendering.
    """
    wl = observation.transmission_wavelength
    tt = observation.transmission_throughput
    assert wl is not None and tt is not None
    wl_d = wl.to(device=device, dtype=torch.float32).reshape(-1)
    tt_d = tt.to(device=device, dtype=torch.float32).reshape(-1)
    wl_s, perm = torch.sort(wl_d)
    tt_s = torch.clamp(tt_d[perm], min=0.0)
    return wl_s, tt_s


def _linear_interp_transmission(
    wl_tab: Tensor,
    tt_tab: Tensor,
    wl_query: Tensor,
) -> Tensor:
    """Linear interpolation of throughput; zero outside ``[wl_tab[0], wl_tab[-1]]``.

    ``wl_tab`` must be sorted ascending on the same device as ``wl_query`` (see
    :func:`_sorted_tabulated_transmission`). Stays on the query tensor's device for CUDA/MPS.
    """
    xq = wl_query.to(dtype=torch.float32)
    x = wl_tab
    y = tt_tab
    if x.dtype != torch.float32:
        x = x.to(dtype=torch.float32)
    if y.dtype != torch.float32:
        y = y.to(dtype=torch.float32)
    lo = x[0]
    hi = x[-1]
    clamped = torch.clamp(xq, lo, hi)
    idx = torch.searchsorted(x, clamped, right=False) - 1
    idx = torch.clamp(idx, 0, x.numel() - 2)
    x0 = x[idx]
    x1 = x[idx + 1]
    denom = x1 - x0
    t = torch.where(
        denom > 0.0,
        (clamped - x0) / denom,
        torch.zeros_like(clamped),
    )
    y0 = y[idx]
    y1 = y[idx + 1]
    out = torch.lerp(y0, y1, t)
    outside = (xq < lo) | (xq > hi)
    return torch.where(outside, torch.zeros_like(out), out)


def _tabulated_response_or_uniform(
    response: Tensor,
    *,
    mass: Tensor,
    mass_eps: float = 1e-30,
) -> Tensor:
    """If ``mass`` is non-positive, return ones like ``response`` (fully tensor, no host sync)."""
    return torch.where(
        mass > mass_eps,
        response,
        torch.ones_like(response),
    )


def _bandpass_quadrature(
    observation: FieldObservation,
    *,
    num_spectral_samples: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    return _shared_bandpass_quadrature(
        observation,
        num_spectral_samples=num_spectral_samples,
        device=device,
    )


def _bandpass_response(
    observation: FieldObservation,
    wavelengths: Tensor,
) -> Tensor:
    return _shared_bandpass_response(observation, wavelengths)


def _bandpass_mc_samples(
    observation: FieldObservation,
    *,
    num_spectral_samples: int,
    device: torch.device,
    proposal_grid_size: int | None = None,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    return _shared_bandpass_mc_samples(
        observation,
        num_spectral_samples=num_spectral_samples,
        device=device,
        proposal_grid_size=proposal_grid_size,
        generator=generator,
    )


def _mixture_mc_samples(
    observations: Sequence[FieldObservation],
    *,
    num_spectral_samples: int,
    device: torch.device,
    proposal_grid_size: int | None = None,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    return _shared_mixture_mc_samples(
        observations,
        num_spectral_samples=num_spectral_samples,
        device=device,
        proposal_grid_size=proposal_grid_size,
        generator=generator,
    )


def _mixture_importance_weights(
    observation: FieldObservation,
    wavelengths: Tensor,
    proposal_probs: Tensor,
) -> Tensor:
    return _shared_mixture_importance_weights(observation, wavelengths, proposal_probs)


def _smoothness_probe_grid(
    domain: FieldDomain,
    *,
    grid_size: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    if grid_size < 2:
        raise ValueError("spatial_smoothness_grid_size must be at least 2")
    offsets = torch.linspace(-1.0, 1.0, steps=int(grid_size), device=device, dtype=torch.float32)
    off_y, off_x = torch.meshgrid(offsets, offsets, indexing="ij")
    cos_dec = max(math.cos(math.radians(float(domain.center_dec_deg))), 1.0e-6)
    dra = off_x * float(domain.spatial_scale_deg) / cos_dec
    ddec = off_y * float(domain.spatial_scale_deg)
    ra = torch.remainder(float(domain.center_ra_deg) + dra + 360.0, 360.0)
    dec = torch.clamp(float(domain.center_dec_deg) + ddec, min=-90.0, max=90.0)
    return ra, dec


def _header_spatial_scale_terms_deg(header: dict) -> list[float]:
    scale_terms: list[float] = []
    for key in ("CD1_1", "CD1_2", "CD2_1", "CD2_2"):
        value = header.get(key)
        if value is not None:
            scale_terms.append(abs(float(value)))
    if scale_terms:
        return scale_terms
    d1 = header.get("CDELT1")
    d2 = header.get("CDELT2")
    if d1 is None or d2 is None:
        return scale_terms
    cdelt1 = abs(float(d1))
    cdelt2 = abs(float(d2))
    pc11 = float(header.get("PC1_1", 1.0))
    pc12 = float(header.get("PC1_2", 0.0))
    pc21 = float(header.get("PC2_1", 0.0))
    pc22 = float(header.get("PC2_2", 1.0))
    scale_terms.extend(
        [
            abs(pc11 * cdelt1) + abs(pc12 * cdelt2),
            abs(pc21 * cdelt1) + abs(pc22 * cdelt2),
        ]
    )
    return scale_terms


def estimate_field_domain(
    observations: Sequence[FieldObservation],
    *,
    coord_mode: FieldCoordinateMode = "local_tangent",
) -> FieldDomain:
    if not observations:
        raise ValueError("estimate_field_domain requires at least one observation")
    center_ra = float(observations[0].header.get("CRVAL1", 0.0))
    center_dec = float(observations[0].header.get("CRVAL2", 0.0))
    wavelength_lowers: list[float] = []
    wavelength_uppers: list[float] = []
    spatial_scales: list[float] = []
    for observation in observations:
        lower, upper = _bandpass_bounds(observation)
        wavelength_lowers.append(lower)
        wavelength_uppers.append(upper)
        scale_terms = _header_spatial_scale_terms_deg(observation.header)
        spatial_scale = max(scale_terms) if scale_terms else 1.0 / 3600.0
        nx = int(observation.header.get("NAXIS1", observation.image.shape[-1]))
        ny = int(observation.header.get("NAXIS2", observation.image.shape[-2]))
        spatial_scales.append(spatial_scale * max(nx, ny))
    lower = min(wavelength_lowers)
    upper = max(wavelength_uppers)
    wavelength_center = 0.5 * (lower + upper)
    wavelength_scale = max(0.5 * (upper - lower), 1.0)
    return FieldDomain(
        center_ra_deg=center_ra,
        center_dec_deg=center_dec,
        spatial_scale_deg=max(max(spatial_scales), 1e-4),
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
        return (
            wavelength - float(self.domain.wavelength_center)
        ) / float(self.domain.wavelength_scale)

    def forward_distribution(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> tuple[Tensor, Tensor]:
        mean = self.forward(ra_deg, dec_deg, wavelength_angstrom)
        logvar = torch.zeros_like(mean)
        return mean, logvar


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
    ra_deg, dec_deg = WCS(term.observation.header).pixel_to_world(x_pix, y_pix)
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
                device=next((p.device for p in model.parameters()), default_compute_device()),
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
            valid = _observation_support_mask(term.observation).to(device=prediction.device)
            target = torch.where(valid, target, torch.zeros_like(target))
            residual = torch.where(valid, prediction - target, torch.zeros_like(prediction))
            weight = (
                term.observation.weight.to(device=prediction.device, dtype=prediction.dtype)
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
            device=next((p.device for p in model.parameters()), default_compute_device()),
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
            tuple(int(v) for v in prediction.shape),
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
            device=next((p.device for p in model.parameters()), default_compute_device()),
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
            raise ValueError("component_models[0] must expose a FieldDomain via .domain")
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
        role_names = (roles,) if isinstance(roles, str) else tuple(str(role) for role in roles)
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
            base_observation.header,
            shape=tuple(image.shape),
            device=device,
        )
        if override_wavelengths is not None or override_weights is not None:
            if override_wavelengths is None or override_weights is None:
                raise ValueError(
                    "override_wavelengths and override_weights must be provided together"
                )
            wavelengths = override_wavelengths.to(device=device, dtype=torch.float32)
            weights = override_weights.to(device=device, dtype=torch.float32)
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
        rendered_shape = tuple(image.shape) if aggregate else (len(selected), *image.shape)
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
        if base_observation.psf_spec is not None and base_observation.psf_spec.mode != "none":
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
            wavelengths = override_wavelengths.to(device=ra_deg.device, dtype=torch.float32)
            weights = override_weights.to(device=ra_deg.device, dtype=torch.float32)
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
        rendered_shape = tuple(ra_deg.shape) if aggregate else (len(selected), *ra_deg.shape)
        rendered = torch.zeros(rendered_shape, device=ra_deg.device, dtype=torch.float32)
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
                rendered = rendered + sample * weight.view(1, *([1] * (sample.ndim - 1)))
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
                return 0.0 if float(error_norm.item()) == 0.0 else float(error_norm.item())
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
                return 0.0 if float(proj_norm.item()) == 0.0 else float(proj_norm.item())
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
            raise ValueError("domain is required when component_models are not provided")
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
    def __init__(self, in_dim: int, out_dim: int, *, omega0: float, sigma0: float) -> None:
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
        return torch.cos(self.omega0 * z) * torch.exp(
            -torch.square(self.sigma0 * z)
        )


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
        self.log_scales = nn.Parameter(torch.full((int(num_basis), 3), math.log(init_scale)))
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
        radial2 = torch.sum(torch.square(diff / scales), dim=-1)
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
        vect_feats = torch.stack([sin_feats, cos_feats], dim=-2).view(coords.shape[:-1] + (-1,))

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
            nn.Linear(1, 16),
            nn.SiLU(),
            nn.Linear(16, 1)
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
        feats = [torch.ones_like(lam), lam, torch.square(lam)]
        if self.num_spectral_frequencies > 0:
            scales = torch.arange(
                1,
                self.num_spectral_frequencies + 1,
                device=lam.device,
                dtype=lam.dtype,
            )
            active = self._active_frequency_count(self.num_spectral_frequencies)
            for idx, scale in enumerate(scales):
                phase = math.pi * scale * lam
                sin_phase = torch.sin(phase)
                cos_phase = torch.cos(phase)
                if idx >= active:
                    sin_phase = torch.zeros_like(sin_phase)
                    cos_phase = torch.zeros_like(cos_phase)
                feats.append(sin_phase)
                feats.append(cos_phase)
        return torch.stack(feats, dim=-1)

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
        spectral_basis = self._spectral_basis(lam).to(dtype=self.coeff_head.weight.dtype)
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
        if wavelengths.ndim != 1 or weights.ndim != 1 or wavelengths.shape != weights.shape:
            raise ValueError("wavelengths and weights must be 1D tensors with matching shape")
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
        spatial_in_dim = self.spatial_dim * (1 + 2 * max(self.num_spatial_frequencies, 0))
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
        feats = [spatial]
        for scale in scales:
            phase = math.pi * scale * spatial
            feats.append(torch.sin(phase))
            feats.append(torch.cos(phase))
        return torch.cat(feats, dim=-1)

    def _bandpass_features(self, wavelengths: Tensor, weights: Tensor) -> Tensor:
        lam = (
            wavelengths.to(dtype=self.output.weight.dtype) - float(self.domain.wavelength_center)
        ) / float(self.domain.wavelength_scale)
        weights = weights.to(dtype=self.output.weight.dtype)
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
            sin_sum = torch.sum(weights * torch.sin(phase), dim=tuple(range(1, 1 + lam.ndim)))
            cos_sum = torch.sum(weights * torch.cos(phase), dim=tuple(range(1, 1 + lam.ndim)))
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
        spatial_latent = self._spatial_latent(ra_deg, dec_deg, float(wavelengths.mean()))
        bandpass_latent = self.bandpass_hidden(self._bandpass_features(wavelengths, weights))
        bandpass_latent = bandpass_latent.view(*([1] * (spatial_latent.ndim - 1)), -1)
        bandpass_latent = bandpass_latent.expand(*spatial_latent.shape[:-1], -1)
        out = self.output(torch.cat([spatial_latent, bandpass_latent], dim=-1)).squeeze(-1)
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
        return torch.stack([lam, torch.square(lam), torch.tanh(lam)], dim=-1)

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
        basis = self.spectral_basis(self._spectral_features(lam).to(dtype=self.coeff_head.weight.dtype))
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
        self.encoder = _make_sky_coordinate_encoder(
            domain,
            num_fourier_frequencies=num_fourier_frequencies,
            spectral_frequencies=spectral_frequencies,
            scale_fourier_to_domain=scale_fourier_to_domain,
        )
        encoded_dim = _sky_encoder_output_dim(self.encoder)
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
        _reset_silu_mlp_linears(self.hidden)
        _reset_logflux_readout_linear(self.output)

    def _bandpass_features(self, wavelengths: Tensor, weights: Tensor) -> Tensor:
        lam = (
            wavelengths.to(dtype=self.hidden[0].weight.dtype) - float(self.domain.wavelength_center)
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
            sin_sum = torch.sum(weights * torch.sin(phase), dim=tuple(range(1, 1 + lam.ndim)))
            cos_sum = torch.sum(weights * torch.cos(phase), dim=tuple(range(1, 1 + lam.ndim)))
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
            wavelength_grid = torch.full_like(ra64, float(wavelength.item()), dtype=torch.float64)
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
        bandpass_latent = self.bandpass_hidden(self._bandpass_features(wavelengths, weights))
        bandpass_latent = bandpass_latent.view(1, *([1] * (encoded.ndim - 2)), -1)
        bandpass_latent = bandpass_latent.expand(encoded.shape[0], *encoded.shape[1:-1], -1)
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
        feats = [spatial]
        for scale in scales:
            phase = math.pi * scale * spatial
            feats.append(torch.sin(phase))
            feats.append(torch.cos(phase))
        return torch.cat(feats, dim=-1)

    def _spectral_features(self, lam: Tensor) -> Tensor:
        feats = [lam, torch.square(lam), torch.tanh(lam)]
        if self.num_spectral_frequencies > 0:
            scales = torch.arange(
                1,
                self.num_spectral_frequencies + 1,
                device=lam.device,
                dtype=lam.dtype,
            )
            for scale in scales:
                phase = math.pi * scale * lam
                feats.append(torch.sin(phase))
                feats.append(torch.cos(phase))
        return torch.stack(feats, dim=-1)

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
        self.encoder = _HashGridEncoder(
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
        feats = [lam, torch.square(lam), torch.tanh(lam)]
        if self.num_spectral_frequencies > 0:
            scales = torch.arange(
                1,
                self.num_spectral_frequencies + 1,
                device=lam.device,
                dtype=lam.dtype,
            )
            for scale in scales:
                phase = math.pi * scale * lam
                feats.append(torch.sin(phase))
                feats.append(torch.cos(phase))
        return torch.stack(feats, dim=-1)

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


class _ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.SiLU()
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return self.act(self.norm(residual + x))


class _ResidualStack(nn.Module):
    def __init__(self, hidden_dim: int, num_blocks: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [_ResidualMLPBlock(hidden_dim) for _ in range(max(int(num_blocks), 0))]
        )

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return x


def _make_sky_coordinate_encoder(
    domain: FieldDomain,
    *,
    num_fourier_frequencies: int,
    spectral_frequencies: int,
    include_degree2: bool = True,
    scale_fourier_to_domain: bool = True,
) -> SkyCoordinateEncoder:
    min_freq = 1.0
    if scale_fourier_to_domain:
        min_freq = sky_encoder_fourier_min_frequency(
            patch_extent_deg=float(domain.spatial_scale_deg),
        )
    return SkyCoordinateEncoder(
        num_fourier_frequencies=num_fourier_frequencies,
        spectral_frequencies=spectral_frequencies,
        include_degree2=include_degree2,
        min_frequency=min_freq,
    )


def _sky_encoder_output_dim(encoder: SkyCoordinateEncoder) -> int:
    with torch.no_grad():
        tensor = encoder(
            torch.zeros(1, dtype=torch.float64),
            torch.zeros(1, dtype=torch.float64),
            wavelength=torch.ones(1, dtype=torch.float64),
        )
    return int(tensor.shape[-1])


def _silu_mlp_sequential(in_dim: int, hidden_dim: int, hidden_layers: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = in_dim
    for _ in range(max(int(hidden_layers), 1)):
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(nn.SiLU())
        current_dim = hidden_dim
    return nn.Sequential(*layers)


def _reset_silu_mlp_linears(sequence: nn.Sequential) -> None:
    for module in sequence:
        if type(module) is nn.Linear:
            nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            nn.init.zeros_(module.bias)


def _reset_logflux_readout_linear(linear: nn.Linear) -> None:
    nn.init.uniform_(linear.weight, -1e-2, 1e-2)
    nn.init.constant_(linear.bias, -3.0)


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
        self.encoder = _make_sky_coordinate_encoder(
            domain,
            num_fourier_frequencies=num_fourier_frequencies,
            spectral_frequencies=spectral_frequencies,
            scale_fourier_to_domain=scale_fourier_to_domain,
        )
        in_dim = _sky_encoder_output_dim(self.encoder)
        self.hidden = _silu_mlp_sequential(in_dim, hidden_dim, hidden_layers)
        self.output = nn.Linear(hidden_dim, 1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        _reset_silu_mlp_linears(self.hidden)
        _reset_logflux_readout_linear(self.output)

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
        self.encoder = _make_sky_coordinate_encoder(
            domain,
            num_fourier_frequencies=num_fourier_frequencies,
            spectral_frequencies=spectral_frequencies,
            scale_fourier_to_domain=scale_fourier_to_domain,
        )
        in_dim = _sky_encoder_output_dim(self.encoder)
        self.body = _silu_mlp_sequential(in_dim, hidden_dim, hidden_layers)
        self.decoder_head = nn.Linear(hidden_dim, 1)
        self._reset_parameters()

    @property
    def output(self) -> nn.Linear:
        return self.decoder_head

    def _reset_parameters(self) -> None:
        _reset_silu_mlp_linears(self.body)
        _reset_logflux_readout_linear(self.decoder_head)

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
) -> None:
    if not isinstance(model, _BaseFieldModel):
        raise TypeError("model must be a torchsky map field _BaseFieldModel subclass")
    steps = max(int(steps), 0)
    if steps == 0:
        return
    if noise_kinds is not None:
        kinds = tuple(k for k in noise_kinds if k != "none")
        if not kinds:
            raise ValueError("noise_kinds must include at least one kind other than 'none'")
    else:
        if noise_kind == "none":
            raise ValueError("noise_kind must not be 'none' for synthetic trunk pretraining")
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
    for step_idx in range(steps):
        step_noise = kinds[step_idx % num_kinds]
        optimizer.zero_grad(set_to_none=True)
        dra = (
            torch.rand(batch, generator=training_gen, dtype=torch.float64, device=device)
            - 0.5
        ) * (2.0 * float(domain.spatial_scale_deg))
        ddec = (
            torch.rand(batch, generator=training_gen, dtype=torch.float64, device=device)
            - 0.5
        ) * (2.0 * float(domain.spatial_scale_deg))
        ra_deg = torch.remainder(float(domain.center_ra_deg) + dra, 360.0)
        dec_deg = torch.clamp(
            float(domain.center_dec_deg) + ddec,
            min=-90.0,
            max=90.0,
        )
        lam_offset = (
            torch.rand(batch, generator=training_gen, dtype=torch.float64, device=device)
            - 0.5
        ) * (2.0 * float(domain.wavelength_scale))
        wavelength = torch.as_tensor(float(domain.wavelength_center), device=device) + lam_offset
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
        loss = torch.mean(torch.square(pred - target))
        loss.backward()
        optimizer.step()


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
        self.encoder = _make_sky_coordinate_encoder(
            domain,
            num_fourier_frequencies=num_fourier_frequencies,
            spectral_frequencies=spectral_frequencies,
            scale_fourier_to_domain=scale_fourier_to_domain,
        )
        in_dim = _sky_encoder_output_dim(self.encoder)
        self.input = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [_ResidualMLPBlock(hidden_dim) for _ in range(max(int(num_blocks), 1))]
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


def _hash_grid_smoothstep(t: Tensor) -> Tensor:
    t = torch.clamp(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _hash_grid_ravel_voxels(corner: Tensor, resolution: int, coord_dim: int) -> Tensor:
    idx = corner[..., -1]
    mult = resolution
    for axis in range(coord_dim - 2, -1, -1):
        idx = idx + corner[..., axis] * mult
        mult *= resolution
    return idx


class _HashGridEncoder(nn.Module):
    """Multiresolution spatial encoding: dense voxels when small, else hashed (Instant-NGP-style).

    Aligns with repo INR notes: multiscale explicit features, reduced hash collisions on coarse
    scales (dense lattice), optional larger tables on fine levels, smoothstep for smoother
    reconstruction (Zip-NeRF-style footprint heuristic).
    """

    _PRIMES = (
        1,
        2654435761,
        805459861,
        3674653429,
    )

    def __init__(
        self,
        coord_dim: int,
        *,
        num_levels: int = 6,
        features_per_level: int = 4,
        log2_hashmap_size: int | Sequence[int] = 12,
        base_resolution: int = 8,
        finest_resolution: int = 128,
        interpolation: Literal["linear", "smoothstep"] = "linear",
        dense_coarse: bool = True,
        max_dense_cells: int = 4096,
        learnable_level_gain: bool = False,
    ) -> None:
        super().__init__()
        self.coord_dim = int(coord_dim)
        self.num_levels = int(num_levels)
        self.features_per_level = int(features_per_level)
        self.base_resolution = int(base_resolution)
        self.finest_resolution = int(finest_resolution)
        self.interpolation = interpolation
        self.dense_coarse = bool(dense_coarse)
        self.max_dense_cells = int(max_dense_cells)
        if interpolation not in {"linear", "smoothstep"}:
            raise ValueError(
                f"interpolation must be 'linear' or 'smoothstep', got {interpolation!r}"
            )
        if self.num_levels <= 0:
            raise ValueError("num_levels must be positive")
        if self.features_per_level <= 0:
            raise ValueError("features_per_level must be positive")
        if self.coord_dim <= 0 or self.coord_dim > len(self._PRIMES):
            raise ValueError("unsupported hash-grid coordinate dimension")
        if isinstance(log2_hashmap_size, int):
            log2_seq = (int(log2_hashmap_size),) * self.num_levels
        else:
            log2_seq = tuple(int(x) for x in log2_hashmap_size)
            if len(log2_seq) != self.num_levels:
                raise ValueError(
                    "len(log2_hashmap_size) must equal num_levels when a sequence is passed"
                )
        self._log2_hashmap_per_level = log2_seq
        if self.num_levels == 1:
            resolutions = [self.base_resolution]
        else:
            growth = math.exp(
                math.log(self.finest_resolution / self.base_resolution)
                / max(self.num_levels - 1, 1)
            )
            resolutions = [
                max(2, int(round(self.base_resolution * (growth**level))))
                for level in range(self.num_levels)
            ]
        self.resolutions = tuple(resolutions)
        self._is_dense: tuple[bool, ...] = tuple(
            self.dense_coarse
            and (int(res) ** self.coord_dim) <= self.max_dense_cells
            for res in self.resolutions
        )
        self._table_sizes: tuple[int, ...] = tuple(
            int(res) ** self.coord_dim if dense else (1 << log2_seq[level])
            for level, (res, dense) in enumerate(
                zip(self.resolutions, self._is_dense, strict=True)
            )
        )
        self.tables = nn.ParameterList(
            [
                nn.Parameter(
                    1.0e-4
                    * torch.randn(int(self._table_sizes[level]), self.features_per_level)
                )
                for level in range(self.num_levels)
            ]
        )
        self.level_gain: nn.Parameter | None
        if learnable_level_gain:
            self.level_gain = nn.Parameter(torch.ones(self.num_levels))
        else:
            self.register_parameter("level_gain", None)

    def _hash(self, corner: Tensor, table_size: int) -> Tensor:
        hashed = corner[..., 0].to(dtype=torch.int64)
        for dim in range(1, self.coord_dim):
            hashed = torch.bitwise_xor(
                hashed,
                corner[..., dim].to(dtype=torch.int64) * int(self._PRIMES[dim]),
            )
        return torch.remainder(hashed, int(table_size)).to(dtype=torch.long)

    def forward(self, coords_unit: Tensor) -> Tensor:
        flat = coords_unit.reshape(-1, self.coord_dim)
        outputs: list[Tensor] = []
        for level_idx, resolution in enumerate(self.resolutions):
            table = self.tables[level_idx]
            table_size = int(self._table_sizes[level_idx])
            is_dense = self._is_dense[level_idx]
            pos = flat * float(resolution - 1)
            base = torch.floor(pos).to(dtype=torch.int64)
            frac = pos - base.to(dtype=pos.dtype)
            if self.interpolation == "smoothstep":
                frac_w = _hash_grid_smoothstep(frac)
            else:
                frac_w = frac
            level_feat = torch.zeros(
                flat.shape[0],
                self.features_per_level,
                device=flat.device,
                dtype=table.dtype,
            )
            for mask in range(1 << self.coord_dim):
                weight = torch.ones(
                    flat.shape[0],
                    device=flat.device,
                    dtype=table.dtype,
                )
                corner = base.clone()
                for dim in range(self.coord_dim):
                    use_upper = bool((mask >> dim) & 1)
                    frac_dim = frac_w[:, dim].to(dtype=table.dtype)
                    if use_upper:
                        corner[:, dim] = corner[:, dim] + 1
                        weight = weight * frac_dim
                    else:
                        weight = weight * (1.0 - frac_dim)
                corner = torch.clamp(corner, min=0, max=resolution - 1)
                if is_dense:
                    idx = _hash_grid_ravel_voxels(
                        corner, int(resolution), self.coord_dim
                    ).to(dtype=torch.long)
                    sample = table[idx]
                else:
                    sample = table[self._hash(corner, table_size)]
                level_feat = level_feat + weight.unsqueeze(-1) * sample
            if self.level_gain is not None:
                g = self.level_gain[level_idx].to(
                    device=level_feat.device, dtype=level_feat.dtype
                )
                level_feat = level_feat * g
            outputs.append(level_feat)
        encoded = torch.cat(outputs, dim=-1)
        return encoded.reshape(*coords_unit.shape[:-1], -1)


def _sample_feature_plane(plane: Tensor, coord_u: Tensor, coord_v: Tensor) -> Tensor:
    flat_u = coord_u.reshape(-1)
    flat_v = coord_v.reshape(-1)
    grid = torch.stack([2.0 * flat_u - 1.0, 2.0 * flat_v - 1.0], dim=-1)
    grid = grid.view(1, -1, 1, 2).to(dtype=plane.dtype, device=plane.device)
    sampled = F.grid_sample(
        plane.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    sampled = sampled.squeeze(0).squeeze(-1).transpose(0, 1)
    return sampled.reshape(*coord_u.shape, plane.shape[0])


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
        self.encoder = _HashGridEncoder(
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
        self.encoder = _HashGridEncoder(
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
            [
                _ResidualMLPBlock(hidden_dim)
                for _ in range(max(int(num_res_blocks), 1))
            ]
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
        self.encoder = _HashGridEncoder(
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
        feats = [torch.ones_like(lam), lam, torch.square(lam)]
        if self.num_spectral_frequencies > 0:
            scales = torch.arange(
                1,
                self.num_spectral_frequencies + 1,
                device=lam.device,
                dtype=lam.dtype,
            )
            for scale in scales:
                phase = math.pi * scale * lam
                feats.append(torch.sin(phase))
                feats.append(torch.cos(phase))
        return torch.stack(feats, dim=-1)

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
            _sample_feature_plane(self.planes["xy"], x, y),
            _sample_feature_plane(self.planes["xl"], x, lam_unit),
            _sample_feature_plane(self.planes["yl"], y, lam_unit),
        ]
        if self.spatial_dim == 3:
            z = spatial_unit[..., 2]
            sampled.extend(
                [
                    _sample_feature_plane(self.planes["xz"], x, z),
                    _sample_feature_plane(self.planes["yz"], y, z),
                    _sample_feature_plane(self.planes["zl"], z, lam_unit),
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
                    _sample_feature_plane(self.planes[f"s{scale_idx}_xy"], x, y),
                    _sample_feature_plane(self.planes[f"s{scale_idx}_xl"], x, lam_unit),
                    _sample_feature_plane(self.planes[f"s{scale_idx}_yl"], y, lam_unit),
                ]
            )
            if self.spatial_dim == 3:
                z = spatial_unit[..., 2]
                sampled.extend(
                    [
                        _sample_feature_plane(self.planes[f"s{scale_idx}_xz"], x, z),
                        _sample_feature_plane(self.planes[f"s{scale_idx}_yz"], y, z),
                        _sample_feature_plane(self.planes[f"s{scale_idx}_zl"], z, lam_unit),
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
        per_scale_dim = len(self._plane_names) * self.features_per_plane + self.spatial_dim + 1
        self.scale_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(per_scale_dim, self.scale_hidden_dim),
                    nn.SiLU(),
                    _ResidualStack(self.scale_hidden_dim, int(scale_num_blocks)),
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
                _ResidualStack(self.global_hidden_dim, int(global_num_blocks)),
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
            _ResidualStack(int(decoder_hidden_dim), int(decoder_num_blocks)),
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
                _sample_feature_plane(self.planes[f"s{scale_idx}_xy"], x, y),
                _sample_feature_plane(self.planes[f"s{scale_idx}_xl"], x, lam_unit),
                _sample_feature_plane(self.planes[f"s{scale_idx}_yl"], y, lam_unit),
            ]
            if self.spatial_dim == 3:
                z = spatial_unit[..., 2]
                sampled.extend(
                    [
                        _sample_feature_plane(self.planes[f"s{scale_idx}_xz"], x, z),
                        _sample_feature_plane(self.planes[f"s{scale_idx}_yz"], y, z),
                        _sample_feature_plane(self.planes[f"s{scale_idx}_zl"], z, lam_unit),
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
        sin_sum = torch.sum(norm_weights * torch.sin(phase), dim=tuple(range(1, 1 + lam.ndim)))
        cos_sum = torch.sum(norm_weights * torch.cos(phase), dim=tuple(range(1, 1 + lam.ndim)))
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
            logvar = torch.zeros_like(mean)
        else:
            logvar = self.logvar_output(decoded).squeeze(-1) + self.logvar_refine(decoded).squeeze(-1)
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
        variance = torch.sum(
            torch.square(centered)
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
        bandpass_latent = self.bandpass_context(self._bandpass_features(wavelengths, weights))
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
        bandpass_latent = self.bandpass_context(self._bandpass_features(wavelengths, weights))
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
        bandpass_latent = self.bandpass_context(self._bandpass_features(wavelengths, weights))
        return self._decode_hidden(hidden, bandpass_latent, global_hidden, coarse_hidden)

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
        bandpass_latent = self.bandpass_context(self._bandpass_features(wavelengths, weights))
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
        self._spectral_observability_cache: dict[tuple[tuple[object, ...], ...], Tensor] = {}
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
        per_scale_dim = len(self._spatial_plane_names) * self.features_per_plane + self.spatial_dim
        self.scale_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(per_scale_dim, self.scale_hidden_dim),
                    nn.SiLU(),
                    _ResidualStack(self.scale_hidden_dim, int(scale_num_blocks)),
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
                2 * self.scale_hidden_dim + self.spectral_context_dim + self.bandpass_hidden_dim,
                int(decoder_hidden_dim),
            ),
            nn.SiLU(),
            _ResidualStack(int(decoder_hidden_dim), int(decoder_num_blocks)),
        )
        self.base_output = nn.Linear(int(decoder_hidden_dim), 1)
        self.refine = nn.Sequential(
            nn.Linear(int(decoder_hidden_dim), int(refine_hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(refine_hidden_dim), 1),
        )
        self.residual_affine = nn.Sequential(
            nn.Linear(
                self.bandpass_hidden_dim + self.spectral_context_dim + self.scale_hidden_dim,
                int(residual_hidden_dim),
            ),
            nn.SiLU(),
            nn.Linear(int(residual_hidden_dim), 2 * self.scale_hidden_dim),
        )
        self.residual_decoder = nn.Sequential(
            nn.Linear(
                2 * self.scale_hidden_dim + self.spectral_context_dim + self.bandpass_hidden_dim,
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
        coarse_centers = torch.linspace(-1.0, 1.0, steps=coarse_count, dtype=torch.float32)
        medium_centers = torch.linspace(-1.0, 1.0, steps=medium_count, dtype=torch.float32)
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
            sampled = [_sample_feature_plane(self.planes[f"s{scale_idx}_xy"], x, y)]
            if self.spatial_dim == 3:
                z = spatial_unit[..., 2]
                sampled.extend(
                    [
                        _sample_feature_plane(self.planes[f"s{scale_idx}_xz"], x, z),
                        _sample_feature_plane(self.planes[f"s{scale_idx}_yz"], y, z),
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
        coarse_gate = coarse_gate / torch.clamp(coarse_gate.sum(dim=-2, keepdim=True), min=1.0e-8)
        fine_gate = fine_gate / torch.clamp(fine_gate.sum(dim=-2, keepdim=True), min=1.0e-8)
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
        return (torch.log(wavelength) - self.spectral_log_center.to(device=device, dtype=dtype)) / (
            self.spectral_log_scale.to(device=device, dtype=dtype)
        )

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
        basis = torch.exp(-0.5 * torch.square(diff / torch.clamp(widths, min=1.0e-3)))
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
        sin_sum = torch.sum(norm_weights * torch.sin(phase), dim=tuple(range(1, 1 + lam.ndim)))
        cos_sum = torch.sum(norm_weights * torch.cos(phase), dim=tuple(range(1, 1 + lam.ndim)))
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
        variance = torch.sum(norm_weights * torch.square(lam - mean))
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
            torch.cat([shared_hidden, spectral_hidden, spectral_context, bandpass_latent], dim=-1)
        )
        base = self.base_output(decoded).squeeze(-1) + self.refine(decoded).squeeze(-1)
        detail_hidden = fine_hidden - shared_hidden
        residual_context = torch.cat([detail_hidden, spectral_context, bandpass_latent], dim=-1)
        gamma_beta = self.residual_affine(residual_context)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        mod_hidden = (spectral_hidden + detail_hidden) * (1.0 + gamma) + beta
        residual = self.residual_decoder(
            torch.cat([mod_hidden, detail_hidden, spectral_context, bandpass_latent], dim=-1)
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
        shared_hidden, _coarse_hidden, fine_hidden = self._shared_hidden_components_from_spatial(
            spatial_raw,
            spatial_unit,
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
        wavelengths = wavelength.reshape(-1) if wavelength.ndim > 0 else wavelength.view(1)
        weights = torch.ones_like(wavelengths)
        shared_hidden, coeffs, fine_hidden = self._shared_hidden_and_coeffs(ra_deg, dec_deg)
        basis = self._spectral_basis(
            wavelength_angstrom,
            device=ra_deg.device,
            dtype=shared_hidden.dtype,
            shape=ra_deg.shape,
        )
        spectral_context = self._spectral_context_from_basis(coeffs, basis)
        bandpass_latent = self.bandpass_context(self._bandpass_features(wavelengths, weights))
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
        shared_hidden, coeffs, fine_hidden = self._shared_hidden_and_coeffs(ra_deg, dec_deg)
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
        bandpass_latent = self.bandpass_context(self._bandpass_features(wavelengths, weights))
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
        visibility = torch.sqrt(torch.sum(torch.square(response_matrix), dim=0))
        max_visibility = torch.clamp(visibility.max(), min=1.0e-8)
        weights = torch.clamp(max_visibility / torch.clamp(visibility, min=1.0e-4) - 1.0, min=0.0)
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
        return torch.mean(torch.square(coeffs) * mode_weights.view(*([1] * (coeffs.ndim - 1)), -1))

    def residual_head_penalty(self) -> Tensor:
        penalty = torch.tensor(0.0, device=self.base_output.weight.device, dtype=self.base_output.weight.dtype)
        for param in self.residual_affine.parameters():
            penalty = penalty + torch.mean(torch.square(param))
        residual_out = self.residual_decoder[-1]
        assert isinstance(residual_out, nn.Linear)
        penalty = penalty + torch.mean(torch.square(residual_out.weight))
        penalty = penalty + torch.mean(torch.square(residual_out.bias))
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
        self.topk_experts = (
            None if topk_experts is None else max(int(topk_experts), 1)
        )
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
                    _sample_feature_plane(expert.planes[f"s{scale_idx}_xy"], x, y),
                    _sample_feature_plane(expert.planes[f"s{scale_idx}_xl"], x, lam_unit),
                    _sample_feature_plane(expert.planes[f"s{scale_idx}_yl"], y, lam_unit),
                ]
            )
            if expert.spatial_dim == 3:
                z = spatial_unit[..., 2]
                sampled.extend(
                    [
                        _sample_feature_plane(expert.planes[f"s{scale_idx}_xz"], x, z),
                        _sample_feature_plane(expert.planes[f"s{scale_idx}_yz"], y, z),
                        _sample_feature_plane(expert.planes[f"s{scale_idx}_zl"], z, lam_unit),
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
        diff = (xy.unsqueeze(-2) - self.atlas_centers.to(device=xy.device, dtype=xy.dtype))
        logits = -0.5 * torch.sum(torch.square(diff / max(self.atlas_sigma, 1.0e-3)), dim=-1)
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
                for expert, hidden in zip(self.experts, mixed_expert_hiddens, strict=True)
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
        self.topk_experts = (
            None if topk_experts is None else max(int(topk_experts), 1)
        )
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
        diff = (
            xy.unsqueeze(-2)
            - self.atlas_centers.to(device=xy.device, dtype=xy.dtype)
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
            global_hidden = expert._global_hidden_from_components(local_spatial_raw, lam)
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


def _field_noise_pretrain_head_modules(model: nn.Module) -> tuple[nn.Module, ...] | None:
    if isinstance(
        model,
        (
            SineImplicitField,
            WireImplicitField,
            FourierMLPField,
            BandpassConditionedField,
            SkyEncoderBandpassField,
            SkyEncoderMLPField,
            SkyEncoderResMLPField,
            StrainerSkyEncoderMLPField,
            HashGridMLPField,
            HashGridResMLPField,
            HashGridConditionedField,
        ),
    ):
        output = getattr(model, "output", None)
        if isinstance(output, nn.Module):
            return (output,)
    return None


def _noise_pretrain_reference_wavelength(observation: FieldObservation) -> float:
    if observation.wavelength is not None:
        return float(observation.wavelength)
    if observation.transmission_center is not None:
        return float(observation.transmission_center)
    lower, upper = _bandpass_bounds(observation)
    return 0.5 * (lower + upper)


def _synthetic_noise_target(
    coords: Tensor,
    *,
    noise_kind: FieldNoisePretrainKind,
    num_terms: int,
    max_frequency: float,
    generator: torch.Generator,
) -> Tensor:
    flat = coords.reshape(-1, coords.shape[-1]).to(dtype=torch.float32)
    device = flat.device
    coord_dim = flat.shape[-1]
    term_count = max(int(num_terms), 1)
    freq_direction = torch.randn(term_count, coord_dim, generator=generator)
    freq_direction = freq_direction / torch.clamp(
        torch.linalg.norm(freq_direction, dim=-1, keepdim=True),
        min=1.0e-6,
    )
    radial = torch.rand(term_count, 1, generator=generator) * float(max_frequency)
    freqs = (1.0 + radial) * freq_direction
    phases = 2.0 * math.pi * torch.rand(term_count, generator=generator)
    if noise_kind == "uniform":
        amplitudes = 2.0 * torch.rand(term_count, generator=generator) - 1.0
    elif noise_kind in {"gaussian", "spectrum"}:
        amplitudes = torch.randn(term_count, generator=generator)
    else:
        raise ValueError(f"unsupported noise pretraining kind {noise_kind!r}")
    if noise_kind == "spectrum":
        spectral_slope = torch.pow(
            1.0 + torch.linalg.norm(freqs, dim=-1),
            1.5,
        )
        amplitudes = amplitudes / spectral_slope
    freqs = freqs.to(device=device, dtype=flat.dtype)
    phases = phases.to(device=device, dtype=flat.dtype)
    amplitudes = amplitudes.to(device=device, dtype=flat.dtype)
    signal = torch.sin(flat @ freqs.transpose(0, 1) + phases) @ amplitudes
    signal = signal.reshape(coords.shape[:-1])
    centered = signal - signal.mean()
    scaled = centered / torch.clamp(centered.std(unbiased=False), min=1.0e-5)
    return torch.clamp(0.5 + 0.18 * scaled, min=1.0e-4)


def _noise_pretrain_field_model(
    observations: Sequence[FieldObservation],
    model: nn.Module,
    *,
    noise_kind: FieldNoisePretrainKind,
    steps: int,
    lr: float,
    observations_per_step: int,
    pixel_samples_per_observation: int,
    num_terms: int,
    max_frequency: float,
) -> None:
    if steps <= 0 or noise_kind == "none":
        return
    head_modules = _field_noise_pretrain_head_modules(model)
    if head_modules is None:
        raise ValueError(
            "noise pretraining is only supported for MLP-style field models with a separable output head"
        )
    preserved_heads = [copy.deepcopy(module.state_dict()) for module in head_modules]
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    sampling_generator = torch.Generator(device="cpu")
    sampling_generator.manual_seed(314159)
    model.train()
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        selected_indices = _sample_observation_indices(
            observations,
            observations_per_step=max(int(observations_per_step), 1),
            observation_sampling="band_balanced",
            generator=sampling_generator,
        )
        total_loss = None
        for index in selected_indices:
            observation = observations[index]
            sampled = _sample_observation_pixels(
                observation,
                num_samples=max(int(pixel_samples_per_observation), 1),
                generator=sampling_generator,
                device=next((p.device for p in model.parameters()), default_compute_device()),
            )
            if sampled is None:
                continue
            sample_ra_deg, sample_dec_deg, _image, sample_weight, _support_count, _ = sampled
            sample_wavelength = torch.full_like(
                sample_ra_deg,
                _noise_pretrain_reference_wavelength(observation),
            )
            assert isinstance(model, _BaseFieldModel)
            coords = model._unit_box_coords(
                sample_ra_deg.to(dtype=torch.float64),
                sample_dec_deg.to(dtype=torch.float64),
                sample_wavelength.to(dtype=torch.float64),
            )
            target = _synthetic_noise_target(
                coords,
                noise_kind=noise_kind,
                num_terms=num_terms,
                max_frequency=max_frequency,
                generator=sampling_generator,
            ).to(device=sample_ra_deg.device)
            pred = model(
                sample_ra_deg,
                sample_dec_deg,
                sample_wavelength,
            )
            weight = sample_weight.to(device=pred.device, dtype=pred.dtype)
            loss = torch.sum(weight * torch.square(pred - target.to(dtype=pred.dtype)))
            loss = loss / torch.clamp(weight.sum(), min=1.0e-8)
            total_loss = loss if total_loss is None else total_loss + loss
        if total_loss is None:
            continue
        total_loss.backward()
        optimizer.step()
    for module, state_dict in zip(head_modules, preserved_heads, strict=True):
        module.load_state_dict(state_dict)


def make_field_model(
    kind: FieldModelKind,
    domain: FieldDomain,
    **kwargs: object,
) -> nn.Module:
    if kind == "sine_implicit":
        return SineImplicitField(domain, **kwargs)
    if kind == "wire_implicit":
        return WireImplicitField(domain, **kwargs)
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


def _field_residual_loss(
    residual: Tensor,
    *,
    loss_kind: FieldLossKind,
    charbonnier_eps: float,
    huber_delta: float,
    logvar: Tensor | None = None,
) -> Tensor:
    if loss_kind == "mse":
        return torch.square(residual)
    if loss_kind == "charbonnier":
        eps = torch.as_tensor(
            float(charbonnier_eps),
            device=residual.device,
            dtype=residual.dtype,
        )
        return torch.sqrt(torch.square(residual) + torch.square(eps))
    if loss_kind == "huber":
        return F.huber_loss(
            residual,
            torch.zeros_like(residual),
            reduction="none",
            delta=float(huber_delta),
        )
    if loss_kind == "gaussian_nll":
        if logvar is None:
            raise ValueError("gaussian_nll requires predictive log-variance")
        clipped = torch.clamp(logvar, min=-8.0, max=6.0)
        return 0.5 * (torch.exp(-clipped) * torch.square(residual) + clipped)
    raise ValueError(f"unsupported field loss kind {loss_kind!r}")


def render_field_observation_distribution(
    model: nn.Module,
    observation: FieldObservation,
    *,
    num_spectral_samples: int = 9,
    spectral_sampling: FieldSpectralSampling = "deterministic_quadrature",
    spectral_sampling_grid_size: int | None = None,
    generator: torch.Generator | None = None,
    override_wavelengths: Tensor | None = None,
    override_weights: Tensor | None = None,
    forward_context: ForwardContext | None = None,
) -> tuple[Tensor, Tensor]:
    image = observation.image
    if image.ndim != 2:
        raise ValueError("FieldObservation.image must be 2D")
    device = next((p.device for p in model.parameters()), default_compute_device())
    ra_deg, dec_deg = _coordinate_grid(
        observation.header,
        shape=tuple(image.shape),
        device=device,
    )
    if override_wavelengths is not None or override_weights is not None:
        if override_wavelengths is None or override_weights is None:
            raise ValueError(
                "override_wavelengths and override_weights must be provided together"
            )
        wavelengths = override_wavelengths.to(device=device, dtype=torch.float32)
        weights = override_weights.to(device=device, dtype=torch.float32)
    elif spectral_sampling == "deterministic_quadrature":
        wavelengths, weights = _bandpass_quadrature(
            observation,
            num_spectral_samples=num_spectral_samples,
            device=device,
        )
    elif spectral_sampling == "bandpass_mc":
        wavelengths, weights = _bandpass_mc_samples(
            observation,
            num_spectral_samples=num_spectral_samples,
            device=device,
            proposal_grid_size=spectral_sampling_grid_size,
            generator=generator,
        )
    elif spectral_sampling == "mixture_mc":
        raise ValueError(
            "mixture_mc sampling must be driven from fit_field_model with shared wavelengths"
        )
    else:
        raise ValueError(f"unsupported spectral_sampling {spectral_sampling!r}")
    render_bandpass_distribution = getattr(model, "render_bandpass_distribution", None)
    if callable(render_bandpass_distribution):
        mean, logvar = _call_render_bandpass_distribution(
            model,
            ra_deg.to(dtype=torch.float32),
            dec_deg.to(dtype=torch.float32),
            wavelengths,
            weights,
            forward_context=forward_context,
        )
    else:
        forward_distribution = getattr(model, "forward_distribution", None)
        if callable(forward_distribution):
            means = []
            variances = []
            for wavelength in wavelengths:
                mean, logvar = _call_forward_distribution(
                    model,
                    ra_deg.to(dtype=torch.float32),
                    dec_deg.to(dtype=torch.float32),
                    wavelength,
                    forward_context=forward_context,
                )
                means.append(mean)
                variances.append(torch.exp(logvar))
            stacked_mean = torch.stack(means, dim=0)
            stacked_var = torch.stack(variances, dim=0)
            mean = torch.sum(
                stacked_mean * weights.view(-1, 1, 1).to(dtype=stacked_mean.dtype),
                dim=0,
            )
            var = torch.sum(
                stacked_var * torch.square(weights).view(-1, 1, 1).to(dtype=stacked_var.dtype),
                dim=0,
            )
            logvar = torch.log(torch.clamp(var, min=1.0e-8))
        else:
            mean = render_field_observation(
                model,
                observation,
                num_spectral_samples=num_spectral_samples,
                spectral_sampling=spectral_sampling,
                spectral_sampling_grid_size=spectral_sampling_grid_size,
                generator=generator,
                override_wavelengths=override_wavelengths,
                override_weights=override_weights,
                forward_context=forward_context,
            )
            logvar = torch.zeros_like(mean)
    operator_bundle = field_observation_operator_bundle(observation)
    if operator_bundle.psf is not None:
        mean = apply_psf_spec(mean, observation.psf_spec)
        variance = apply_psf_spec(torch.exp(logvar), observation.psf_spec)
        logvar = torch.log(torch.clamp(variance, min=1.0e-8))
    if operator_bundle.background is not None:
        mean = apply_background_components(
            mean,
            **observation_background_components(
                observation,
                shape=tuple(mean.shape),
                device=mean.device,
                dtype=mean.dtype,
            ),
        )
    return mean.to(dtype=image.dtype), logvar.to(dtype=image.dtype)


def render_field_samples_distribution(
    model: nn.Module,
    observation: FieldObservation,
    *,
    ra_deg: Tensor,
    dec_deg: Tensor,
    num_spectral_samples: int = 9,
    spectral_sampling: FieldSpectralSampling = "deterministic_quadrature",
    spectral_sampling_grid_size: int | None = None,
    generator: torch.Generator | None = None,
    override_wavelengths: Tensor | None = None,
    override_weights: Tensor | None = None,
    forward_context: ForwardContext | None = None,
) -> tuple[Tensor, Tensor]:
    operator_bundle = field_observation_operator_bundle(observation)
    if operator_bundle.psf is not None:
        raise ValueError("pixel-sampled field rendering does not support PSF blurring")
    if operator_bundle.background is not None:
        raise ValueError("pixel-sampled field rendering does not support background terms")
    if override_wavelengths is not None or override_weights is not None:
        if override_wavelengths is None or override_weights is None:
            raise ValueError(
                "override_wavelengths and override_weights must be provided together"
            )
        wavelengths = override_wavelengths.to(device=ra_deg.device, dtype=torch.float32)
        weights = override_weights.to(device=ra_deg.device, dtype=torch.float32)
    elif spectral_sampling == "deterministic_quadrature":
        wavelengths, weights = _bandpass_quadrature(
            observation,
            num_spectral_samples=num_spectral_samples,
            device=ra_deg.device,
        )
    elif spectral_sampling == "bandpass_mc":
        wavelengths, weights = _bandpass_mc_samples(
            observation,
            num_spectral_samples=num_spectral_samples,
            device=ra_deg.device,
            proposal_grid_size=spectral_sampling_grid_size,
            generator=generator,
        )
    elif spectral_sampling == "mixture_mc":
        raise ValueError(
            "mixture_mc sampling must be driven from fit_field_model with shared wavelengths"
        )
    else:
        raise ValueError(f"unsupported spectral_sampling {spectral_sampling!r}")
    render_bandpass_distribution = getattr(model, "render_bandpass_distribution", None)
    if callable(render_bandpass_distribution):
        return _call_render_bandpass_distribution(
            model,
            ra_deg.to(dtype=torch.float32),
            dec_deg.to(dtype=torch.float32),
            wavelengths,
            weights,
            forward_context=forward_context,
        )
    forward_distribution = getattr(model, "forward_distribution", None)
    if callable(forward_distribution):
        means = []
        variances = []
        for wavelength in wavelengths:
            mean, logvar = _call_forward_distribution(
                model,
                ra_deg.to(dtype=torch.float32),
                dec_deg.to(dtype=torch.float32),
                wavelength,
                forward_context=forward_context,
            )
            means.append(mean)
            variances.append(torch.exp(logvar))
        stacked_mean = torch.stack(means, dim=0)
        stacked_var = torch.stack(variances, dim=0)
        mean = torch.sum(
            stacked_mean * weights.view(-1, 1, 1).to(dtype=stacked_mean.dtype),
            dim=0,
        )
        var = torch.sum(
            stacked_var * torch.square(weights).view(-1, 1, 1).to(dtype=stacked_var.dtype),
            dim=0,
        )
        return mean, torch.log(torch.clamp(var, min=1.0e-8))
    mean = render_field_samples(
        model,
        observation,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        num_spectral_samples=num_spectral_samples,
        spectral_sampling=spectral_sampling,
        spectral_sampling_grid_size=spectral_sampling_grid_size,
        generator=generator,
        override_wavelengths=override_wavelengths,
        override_weights=override_weights,
        forward_context=forward_context,
    )
    return mean, torch.zeros_like(mean)



def render_field_observation(
    model: nn.Module,
    observation: FieldObservation,
    *,
    num_spectral_samples: int = 9,
    spectral_sampling: FieldSpectralSampling = "deterministic_quadrature",
    spectral_sampling_grid_size: int | None = None,
    generator: torch.Generator | None = None,
    override_wavelengths: Tensor | None = None,
    override_weights: Tensor | None = None,
    spatial_tile_hw: tuple[int, int] | None = None,
    use_func_vmap: bool = False,
    func_vmap_chunk: int = 65536,
    forward_context: ForwardContext | None = None,
) -> Tensor:
    image = observation.image
    if image.ndim != 2:
        raise ValueError("FieldObservation.image must be 2D")
    device = next((p.device for p in model.parameters()), default_compute_device())
    ra_deg, dec_deg = _coordinate_grid(
        observation.header,
        shape=tuple(image.shape),
        device=device,
    )
    operator_bundle = field_observation_operator_bundle(observation)
    if override_wavelengths is not None or override_weights is not None:
        if override_wavelengths is None or override_weights is None:
            raise ValueError(
                "override_wavelengths and override_weights must be provided together"
            )
        wavelengths = override_wavelengths.to(device=device, dtype=torch.float32)
        weights = override_weights.to(device=device, dtype=torch.float32)
    elif spectral_sampling == "deterministic_quadrature":
        wavelengths, weights = _bandpass_quadrature(
            observation,
            num_spectral_samples=num_spectral_samples,
            device=device,
        )
    elif spectral_sampling == "bandpass_mc":
        wavelengths, weights = _bandpass_mc_samples(
            observation,
            num_spectral_samples=num_spectral_samples,
            device=device,
            proposal_grid_size=spectral_sampling_grid_size,
            generator=generator,
        )
    elif spectral_sampling == "mixture_mc":
        raise ValueError(
            "mixture_mc sampling must be driven from fit_field_model with shared wavelengths"
        )
    else:
        raise ValueError(f"unsupported spectral_sampling {spectral_sampling!r}")
    if spatial_tile_hw is not None and operator_bundle.psf is not None:
        raise ValueError(
            "render_field_observation(..., spatial_tile_hw=...) cannot be used with "
            "a non-trivial PSF because the PSF couples pixels across tiles"
        )
    if forward_context is not None and use_func_vmap:
        raise ValueError(
            "render_field_observation(..., use_func_vmap=True) does not support "
            "forward_context=... (torch.func vmap path is context-free)"
        )
    render_bandpass = getattr(model, "render_bandpass", None)
    if callable(render_bandpass):
        if spatial_tile_hw is not None or use_func_vmap:
            raise ValueError(
                "spatial_tile_hw and use_func_vmap are not supported when model.render_bandpass is defined"
            )
        rendered = _call_render_bandpass(
            model,
            ra_deg.to(dtype=torch.float32),
            dec_deg.to(dtype=torch.float32),
            wavelengths,
            weights,
            forward_context=forward_context,
        )
        if operator_bundle.psf is not None:
            rendered = apply_psf_spec(rendered, observation.psf_spec)
        if operator_bundle.background is not None:
            rendered = apply_background_components(
                rendered,
                **observation_background_components(
                    observation,
                    shape=tuple(rendered.shape),
                    device=rendered.device,
                    dtype=rendered.dtype,
                ),
            )
        return rendered.to(dtype=image.dtype)
    H, W = int(image.shape[-2]), int(image.shape[-1])
    ra32 = ra_deg.to(dtype=torch.float32)
    dec32 = dec_deg.to(dtype=torch.float32)
    if spatial_tile_hw is None:
        tiles = [(0, H, 0, W)]
    else:
        th, tw = int(spatial_tile_hw[0]), int(spatial_tile_hw[1])
        tiles = list(_iter_spatial_tiles(H, W, th, tw))
    rendered = torch.zeros((H, W), device=device, dtype=torch.float32)
    for y0, y1, x0, x1 in tiles:
        patch = torch.zeros((y1 - y0, x1 - x0), device=device, dtype=torch.float32)
        ra_p = ra32[y0:y1, x0:x1]
        dec_p = dec32[y0:y1, x0:x1]
        for i, wavelength in enumerate(wavelengths):
            if use_func_vmap:
                plane = _evaluate_field_plane_func_vmap(
                    model,
                    ra_p,
                    dec_p,
                    wavelength=wavelength,
                    chunk=func_vmap_chunk,
                )
            else:
                plane = _call_scalar_field_forward(
                    model,
                    ra_p,
                    dec_p,
                    wavelength,
                    forward_context=forward_context,
                )
            patch += plane * weights[i].to(dtype=plane.dtype)
        rendered[y0:y1, x0:x1] = patch
    if operator_bundle.psf is not None:
        rendered = apply_psf_spec(rendered, observation.psf_spec)
    if operator_bundle.background is not None:
        rendered = apply_background_components(
            rendered,
            **observation_background_components(
                observation,
                shape=tuple(rendered.shape),
                device=rendered.device,
                dtype=rendered.dtype,
            ),
        )
    return rendered.to(dtype=image.dtype)


def render_field_samples(
    model: nn.Module,
    observation: FieldObservation,
    *,
    ra_deg: Tensor,
    dec_deg: Tensor,
    num_spectral_samples: int = 9,
    spectral_sampling: FieldSpectralSampling = "deterministic_quadrature",
    spectral_sampling_grid_size: int | None = None,
    generator: torch.Generator | None = None,
    override_wavelengths: Tensor | None = None,
    override_weights: Tensor | None = None,
    forward_context: ForwardContext | None = None,
) -> Tensor:
    operator_bundle = field_observation_operator_bundle(observation)
    if operator_bundle.psf is not None:
        raise ValueError("pixel-sampled field rendering does not support PSF blurring")
    if operator_bundle.background is not None:
        raise ValueError("pixel-sampled field rendering does not support background terms")
    if override_wavelengths is not None or override_weights is not None:
        if override_wavelengths is None or override_weights is None:
            raise ValueError(
                "override_wavelengths and override_weights must be provided together"
            )
        wavelengths = override_wavelengths.to(device=ra_deg.device, dtype=torch.float32)
        weights = override_weights.to(device=ra_deg.device, dtype=torch.float32)
    elif spectral_sampling == "deterministic_quadrature":
        wavelengths, weights = _bandpass_quadrature(
            observation,
            num_spectral_samples=num_spectral_samples,
            device=ra_deg.device,
        )
    elif spectral_sampling == "bandpass_mc":
        wavelengths, weights = _bandpass_mc_samples(
            observation,
            num_spectral_samples=num_spectral_samples,
            device=ra_deg.device,
            proposal_grid_size=spectral_sampling_grid_size,
            generator=generator,
        )
    elif spectral_sampling == "mixture_mc":
        raise ValueError(
            "mixture_mc sampling must be driven from fit_field_model with shared wavelengths"
        )
    else:
        raise ValueError(f"unsupported spectral_sampling {spectral_sampling!r}")
    render_bandpass = getattr(model, "render_bandpass", None)
    if callable(render_bandpass):
        return _call_render_bandpass(
            model,
            ra_deg.to(dtype=torch.float32),
            dec_deg.to(dtype=torch.float32),
            wavelengths,
            weights,
            forward_context=forward_context,
        )
    spectral_samples = []
    for wavelength in wavelengths:
        spectral_samples.append(
            _call_scalar_field_forward(
                model,
                ra_deg.to(dtype=torch.float32),
                dec_deg.to(dtype=torch.float32),
                wavelength,
                forward_context=forward_context,
            )
        )
    stacked = torch.stack(spectral_samples, dim=0)
    return torch.sum(
        stacked * weights.view(-1, 1).to(dtype=stacked.dtype),
        dim=0,
    )


def _iter_spatial_tiles(
    height: int, width: int, tile_h: int, tile_w: int
) -> Iterator[tuple[int, int, int, int]]:
    th = max(int(tile_h), 1)
    tw = max(int(tile_w), 1)
    for y0 in range(0, height, th):
        for x0 in range(0, width, tw):
            y1 = min(y0 + th, height)
            x1 = min(x0 + tw, width)
            yield y0, y1, x0, x1


def evaluate_field_coordinates_vmapped(
    model: nn.Module,
    ra_deg: Tensor,
    dec_deg: Tensor,
    wavelength_angstrom: Tensor,
) -> Tensor:
    """Evaluate the field with ``torch.func.vmap`` over flattened coordinates.

    This uses ``functional_call`` so the forward is expressed as a pure function
    of parameters/buffers plus per-sample inputs—useful for ``func`` transforms
    (JAX-style batching semantics). Standard field ``nn.Module`` instances already
    support batched ``(ra, dec, λ)`` tensors; prefer a single vectorized forward unless you
    need this composition.
    """
    try:
        from torch.func import functional_call, vmap
    except ImportError as e:
        raise RuntimeError(
            "evaluate_field_coordinates_vmapped requires PyTorch 2.x torch.func"
        ) from e
    ra = torch.as_tensor(ra_deg).reshape(-1).to(dtype=torch.float32)
    dec = torch.as_tensor(dec_deg).reshape(-1).to(dtype=torch.float32)
    wl = torch.as_tensor(wavelength_angstrom)
    if wl.ndim == 0:
        wl = wl.expand(ra.shape[0]).to(dtype=torch.float32, device=ra.device)
    else:
        wl = wl.reshape(-1).to(dtype=torch.float32, device=ra.device)
        if wl.shape[0] != ra.shape[0]:
            raise ValueError(
                "wavelength_angstrom must be scalar or match flattened ra/dec length"
            )
    state = {**dict(model.named_parameters()), **dict(model.named_buffers())}

    def one_point(
        ra_s: Tensor,
        dec_s: Tensor,
        wl_s: Tensor,
    ) -> Tensor:
        out = functional_call(
            model,
            state,
            (ra_s.unsqueeze(0), dec_s.unsqueeze(0), wl_s.unsqueeze(0)),
        )
        return out.reshape(())

    return vmap(one_point)(ra, dec, wl)


def _evaluate_field_plane_func_vmap(
    model: nn.Module,
    ra_grid: Tensor,
    dec_grid: Tensor,
    *,
    wavelength: Tensor | float,
    chunk: int,
) -> Tensor:
    """One wavelength slice; ``ra_grid``/``dec_grid`` share shape (outputs that shape)."""
    shape = tuple(ra_grid.shape)
    ra_f = ra_grid.to(dtype=torch.float32).reshape(-1)
    dec_f = dec_grid.to(dtype=torch.float32).reshape(-1)
    n = int(ra_f.shape[0])
    dev = ra_f.device
    wl0 = torch.as_tensor(wavelength, dtype=torch.float32, device=dev)
    cc = max(int(chunk), 1)
    parts: list[Tensor] = []
    for i in range(0, n, cc):
        sl = slice(i, min(i + cc, n))
        m = sl.stop - sl.start
        wlb = wl0.expand(m) if wl0.ndim == 0 else wl0[sl]
        parts.append(
            evaluate_field_coordinates_vmapped(model, ra_f[sl], dec_f[sl], wlb)
        )
    return torch.cat(parts, dim=0).view(shape)


def _field_hypercube_spectral_stack(
    model: nn.Module,
    ra_grid: Tensor,
    dec_grid: Tensor,
    wavelengths: Tensor,
    *,
    spectral_chunk: int,
    use_func_vmap: bool,
    func_vmap_chunk: int,
) -> Tensor:
    """Stack wavelength planes for one spatial patch (``ra_grid`` shape = ``dec_grid`` shape)."""
    nw = int(wavelengths.shape[0])
    hw_shape = tuple(ra_grid.shape)
    ra_f = ra_grid.to(dtype=torch.float32)
    dec_f = dec_grid.to(dtype=torch.float32)
    chunk = int(spectral_chunk)
    if chunk <= 0 or chunk >= nw:
        chunk = nw
    planes_out: list[Tensor] = []
    for start in range(0, nw, chunk):
        end = min(start + chunk, nw)
        lam_chunk = wavelengths[start:end]
        c = int(lam_chunk.shape[0])
        if use_func_vmap:
            if c == 1:
                planes_out.append(
                    _evaluate_field_plane_func_vmap(
                        model,
                        ra_f,
                        dec_f,
                        wavelength=lam_chunk[0],
                        chunk=func_vmap_chunk,
                    ).unsqueeze(0)
                )
            else:
                for j in range(c):
                    planes_out.append(
                        _evaluate_field_plane_func_vmap(
                            model,
                            ra_f,
                            dec_f,
                            wavelength=lam_chunk[j],
                            chunk=func_vmap_chunk,
                        ).unsqueeze(0)
                    )
            continue
        if c == 1:
            plane = model(ra_f, dec_f, lam_chunk[0])
            planes_out.append(plane.unsqueeze(0))
            continue
        nflat = c * int(ra_f.numel())
        ra_flat = ra_f.unsqueeze(0).expand(c, *ra_f.shape).reshape(nflat)
        dec_flat = dec_f.unsqueeze(0).expand(c, *dec_f.shape).reshape(nflat)
        lam_flat = lam_chunk.view(c, *([1] * len(hw_shape))).expand(c, *hw_shape).reshape(
            nflat
        )
        out_flat = model(ra_flat, dec_flat, lam_flat)
        if out_flat.ndim != 1 or out_flat.shape[0] != nflat:
            raise ValueError(
                "field render expects one scalar output per (ra, dec, λ) sample; "
                f"got shape {tuple(out_flat.shape)} for batch size {nflat}"
            )
        planes_out.append(out_flat.view(c, *hw_shape))
    return torch.cat(planes_out, dim=0)


def render_field_cube(
    model: nn.Module,
    *,
    target_header: dict,
    spectral_axis: SpectralAxis,
    shape: tuple[int, int] | None = None,
    spectral_chunk: int = 1,
    spatial_tile_hw: tuple[int, int] | None = None,
    use_func_vmap: bool = False,
    func_vmap_chunk: int = 65536,
) -> Tensor:
    """Render a (nλ, H, W) cube on the target WCS grid.

    ``spectral_chunk`` controls how many wavelength planes are evaluated per
    ``model`` forward. The default ``1`` matches the legacy one-λ loop (lowest
    peak memory). Set to ``0`` or any value ``>= nλ`` to fuse all wavelengths
    into a single batched forward (usually faster on GPU for moderate grids).

    ``spatial_tile_hw`` optionally processes the map in ``(tile_h, tile_w)`` spatial
    windows to cap peak activation memory on large TAN grids.

    ``use_func_vmap=True`` routes each wavelength slice through
    :func:`evaluate_field_coordinates_vmapped` in sub-batches of ``func_vmap_chunk``
    pixels (``torch.func`` / JAX-style vectorization). Default field models are
    already batch-vectorized; this path is for ``func`` composition and debugging.
    """
    fallback = (
        (int(target_header["NAXIS2"]), int(target_header["NAXIS1"]))
        if shape is None
        else shape
    )
    device = next((p.device for p in model.parameters()), default_compute_device())
    ra_deg, dec_deg = _coordinate_grid(target_header, shape=fallback, device=device)
    wavelengths = spectral_axis.to_wavelength().to(device=device, dtype=torch.float32)
    if wavelengths.ndim != 1:
        raise ValueError("spectral_axis must yield a 1D wavelength tensor")
    nw = int(wavelengths.shape[0])
    if nw == 0:
        return torch.empty((0, *tuple(ra_deg.shape)), device=device, dtype=torch.float32)
    ra_f = ra_deg.to(dtype=torch.float32)
    dec_f = dec_deg.to(dtype=torch.float32)
    H, W = int(ra_f.shape[0]), int(ra_f.shape[1])
    tiles: list[tuple[int, int, int, int]]
    if spatial_tile_hw is None:
        tiles = [(0, H, 0, W)]
    else:
        th, tw = int(spatial_tile_hw[0]), int(spatial_tile_hw[1])
        tiles = list(_iter_spatial_tiles(H, W, th, tw))
    out = torch.empty((nw, H, W), device=device, dtype=torch.float32)
    for y0, y1, x0, x1 in tiles:
        sub = _field_hypercube_spectral_stack(
            model,
            ra_f[y0:y1, x0:x1],
            dec_f[y0:y1, x0:x1],
            wavelengths,
            spectral_chunk=spectral_chunk,
            use_func_vmap=use_func_vmap,
            func_vmap_chunk=func_vmap_chunk,
        )
        out[:, y0:y1, x0:x1] = sub
    return out


def render_field_healpix(
    model: nn.Module,
    *,
    nside: int,
    spectral_axis: SpectralAxis,
    nest: bool = False,
    spectral_chunk: int = 1,
    spatial_chunk_pixels: int | None = None,
    use_func_vmap: bool = False,
    func_vmap_chunk: int = 65536,
    device: torch.device | None = None,
) -> Tensor:
    """Render ``(nλ, npix)`` on the native HEALPix pixel ordering (RING or NEST).

    Pixels are visited in increasing ``ipix`` order via :func:`torchsky.wcs.healpix.pix2ang`.
    Use ``spatial_chunk_pixels`` to limit the number of sky samples per ``model`` call
    (memory). Combine with ``spectral_chunk`` and optional ``use_func_vmap`` like
    :func:`render_field_cube`.
    """
    nside_i = int(nside)
    npix = _healpix_wcs.nside2npix(nside_i)
    dev = device if device is not None else next((p.device for p in model.parameters()), default_compute_device())
    wavelengths = spectral_axis.to_wavelength().to(device=dev, dtype=torch.float32)
    nw = int(wavelengths.shape[0])
    if nw == 0:
        return torch.empty((0, npix), device=dev, dtype=torch.float32)
    chunk_sp = spatial_chunk_pixels or min(1_048_576, max(npix, 1))
    chunk_sp = max(int(chunk_sp), 1)
    ipix_full = torch.arange(npix, device=dev, dtype=torch.int64)
    out = torch.empty((nw, npix), device=dev, dtype=torch.float32)
    for i0 in range(0, npix, chunk_sp):
        i1 = min(i0 + chunk_sp, npix)
        ipix = ipix_full[i0:i1]
        ra, dec = _healpix_wcs.pix2ang(nside_i, ipix, nest=nest, lonlat=True)
        ra = ra.to(device=dev, dtype=torch.float32)
        dec = dec.to(device=dev, dtype=torch.float32)
        sub = _field_hypercube_spectral_stack(
            model,
            ra,
            dec,
            wavelengths,
            spectral_chunk=spectral_chunk,
            use_func_vmap=use_func_vmap,
            func_vmap_chunk=func_vmap_chunk,
        )
        out[:, i0:i1] = sub
    return out


def render_field_cube_via_healpix(
    model: nn.Module,
    *,
    target_header: dict,
    spectral_axis: SpectralAxis,
    nside: int,
    nest: bool = False,
    shape: tuple[int, int] | None = None,
    spectral_chunk: int = 1,
    healpix_spatial_chunk_pixels: int | None = None,
    use_func_vmap: bool = False,
    func_vmap_chunk: int = 65536,
    device: torch.device | None = None,
) -> Tensor:
    """Render ``(nλ, H, W)`` by evaluating on HEALPix, then sampling onto the WCS grid.

    The target projection is whatever ``target_header`` encodes (TAN, CAR, SIN, …).
    Spherical bilinear interpolation uses :func:`sample_healpix_map_on_image_header`.

    Unlike :func:`render_field_cube`, this path does **not** evaluate the field exactly
    at image pixel centers; it uses HEALPix cell centers (plus interpolation), which
    is useful for spherical pipelines and for reusing one healpix cube under several
    target headers.

    Grid size ``(H, W)`` follows the same ``shape`` / ``NAXIS`` rules as
    :func:`render_field_cube` via :func:`_header_shape`.

    Spherical resampling detaches from autograd (see
    :func:`sample_healpix_map_on_image_header`); use :func:`render_field_cube` when
    you need gradients through pixel-center evaluation.
    """
    dev = device if device is not None else next((p.device for p in model.parameters()), default_compute_device())
    hp_cube = render_field_healpix(
        model,
        nside=nside,
        spectral_axis=spectral_axis,
        nest=nest,
        spectral_chunk=spectral_chunk,
        spatial_chunk_pixels=healpix_spatial_chunk_pixels,
        use_func_vmap=use_func_vmap,
        func_vmap_chunk=func_vmap_chunk,
        device=dev,
    )
    fallback_hw = (
        (int(target_header["NAXIS2"]), int(target_header["NAXIS1"]))
        if shape is None
        else (int(shape[0]), int(shape[1]))
    )
    h, w = _header_shape(target_header, fallback=fallback_hw)
    hdr = dict(target_header)
    hdr["NAXIS1"] = int(w)
    hdr["NAXIS2"] = int(h)
    nw = int(hp_cube.shape[0])
    if nw == 0:
        return torch.empty((0, h, w), device=dev, dtype=torch.float32)
    out = torch.empty((nw, h, w), device=dev, dtype=hp_cube.dtype)
    for i in range(nw):
        out[i] = sample_healpix_map_on_image_header(
            hp_cube[i],
            nside=int(nside),
            nest=nest,
            target_header=hdr,
            device=dev,
        )
    return out


def fit_field_component_decomposition(
    observations: Sequence[FieldObservation],
    *,
    component_roles: Sequence[str],
    model: FieldComponentDecomposition | None = None,
    domain: FieldDomain | None = None,
    component_models: Sequence[nn.Module] | None = None,
    component_model_kind: FieldModelKind = "sine_implicit",
    component_model_kwargs: dict[str, object] | None = None,
    surveys: Sequence[str] | None = None,
    instruments: Sequence[str] | None = None,
    observation_forward_contexts: Sequence[ForwardContext | None] | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
    **fit_kwargs: object,
) -> FieldFitResult:
    """Construct and fit a field decomposition model on context-labeled observations."""
    if model is not None:
        if component_models is not None:
            raise ValueError("component_models cannot be provided when model is provided")
        if domain is not None:
            raise ValueError("domain cannot be provided when model is provided")
        if component_model_kwargs is not None:
            raise ValueError(
                "component_model_kwargs cannot be provided when model is provided"
            )
    else:
        build_domain = domain if domain is not None else estimate_field_domain(observations)
        model = make_field_component_decomposition(
            component_roles=component_roles,
            domain=build_domain,
            component_models=component_models,
            component_model_kind=component_model_kind,
            component_model_kwargs=component_model_kwargs,
            surveys=surveys,
            instruments=instruments,
            observation_forward_contexts=observation_forward_contexts,
            dtype=dtype,
            device=torch.device(device) if isinstance(device, str) else device,
        )
    return fit_field_model(
        observations,
        model,
        device=device,
        observation_forward_contexts=observation_forward_contexts,
        **fit_kwargs,
    )


def fit_field_model(
    observations: Sequence[FieldObservation],
    model: nn.Module,
    *,
    steps: int = 200,
    lr: float = 5.0e-3,
    weight_decay: float = 0.0,
    ema_decay: float = 0.0,
    num_spectral_samples: int = 9,
    loss_kind: FieldLossKind = "mse",
    charbonnier_eps: float = 1.0e-3,
    huber_delta: float = 5.0e-2,
    loss_reduction: FieldLossReduction = "per_observation",
    lr_schedule: FieldLRSchedule = "constant",
    grad_clip_norm: float | None = None,
    curriculum: FieldCurriculum = "none",
    training_spectral_sampling: FieldSpectralSampling = "deterministic_quadrature",
    training_spectral_sampling_grid_size: int | None = None,
    spectral_anchor_observations_per_step: int = 0,
    spectral_anchor_num_samples: int | None = None,
    spectral_anchor_loss_weight: float = 0.0,
    component_anchor_terms: Sequence[FieldComponentAnchorTerm] | None = None,
    component_anchor_num_spectral_samples: int | None = None,
    component_anchor_loss_kind: FieldLossKind | None = None,
    component_centroid_anchor_terms: Sequence[FieldCentroidAnchorTerm] | None = None,
    lr_warmup_steps: int = 0,
    spatial_smoothness_weight: float = 0.0,
    spatial_smoothness_grid_size: int = 8,
    observations_per_step: int | None = None,
    observation_sampling: FieldObservationSampling = "uniform",
    training_patch_size: int | None = None,
    training_patches_per_observation: int = 1,
    pixel_samples_per_observation: int | None = None,
    shared_sky_patch_sampling: bool = False,
    streaming_accumulation: bool = False,
    full_observations_per_step: int = 0,
    context_observations_per_step: int = 0,
    context_downsample_factor: int = 1,
    context_loss_weight: float = 0.0,
    spectral_observability_weight: float = 0.0,
    spectral_observability_probe_grid_size: int = 8,
    residual_head_penalty_weight: float = 0.0,
    noise_pretrain_kind: FieldNoisePretrainKind = "none",
    noise_pretrain_steps: int = 0,
    noise_pretrain_lr: float | None = None,
    noise_pretrain_observations_per_step: int | None = None,
    noise_pretrain_pixel_samples_per_observation: int = 1024,
    noise_pretrain_num_terms: int = 48,
    noise_pretrain_max_frequency: float = 8.0,
    freeze_field_encoder: bool = False,
    freeze_field_body: bool = False,
    device: torch.device | str | None = None,
    observation_forward_contexts: Sequence[ForwardContext | None] | None = None,
) -> FieldFitResult:
    if not observations:
        raise ValueError("fit_field_model requires at least one observation")
    train_device = resolve_fit_device(model, device)
    model = model.to(device=train_device)
    if observation_forward_contexts is not None:
        if len(observation_forward_contexts) != len(observations):
            raise ValueError(
                "observation_forward_contexts must have the same length as observations"
            )
        obs_forward_ctx: tuple[ForwardContext | None, ...] = tuple(
            observation_forward_contexts
        )
    else:
        obs_forward_ctx = tuple(None for _ in observations)
    if steps <= 0:
        raise ValueError("steps must be positive")
    pretrain_steps = max(int(noise_pretrain_steps), 0)
    if noise_pretrain_kind != "none" and pretrain_steps > 0:
        _noise_pretrain_field_model(
            observations,
            model,
            noise_kind=noise_pretrain_kind,
            steps=pretrain_steps,
            lr=(
                float(noise_pretrain_lr)
                if noise_pretrain_lr is not None
                else float(lr)
            ),
            observations_per_step=(
                int(noise_pretrain_observations_per_step)
                if noise_pretrain_observations_per_step is not None
                else min(len(observations), 4)
            ),
            pixel_samples_per_observation=noise_pretrain_pixel_samples_per_observation,
            num_terms=noise_pretrain_num_terms,
            max_frequency=noise_pretrain_max_frequency,
        )
    if freeze_field_encoder or freeze_field_body:
        for name, param in model.named_parameters():
            if freeze_field_encoder and name.startswith("encoder."):
                param.requires_grad = False
            if freeze_field_body and name.startswith("body."):
                param.requires_grad = False
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError(
            "fit_field_model: no trainable parameters (check freeze_field_encoder / freeze_field_body)"
        )
    base_lr = float(lr)
    optimizer = torch.optim.Adam(
        trainable_params,
        lr=base_lr,
        weight_decay=max(float(weight_decay), 0.0),
    )
    model.train()
    loss_history: list[float] = []
    anchor_count = max(int(spectral_anchor_observations_per_step), 0)
    anchor_num_samples = (
        int(spectral_anchor_num_samples)
        if spectral_anchor_num_samples is not None
        else max(int(num_spectral_samples), 17)
    )
    anchor_loss_weight = max(float(spectral_anchor_loss_weight), 0.0)
    component_anchor_num = (
        int(component_anchor_num_spectral_samples)
        if component_anchor_num_spectral_samples is not None
        else int(num_spectral_samples)
    )
    component_anchor_kind = (
        loss_kind if component_anchor_loss_kind is None else component_anchor_loss_kind
    )
    warmup_steps = max(int(lr_warmup_steps), 0)
    smoothness_weight = max(float(spatial_smoothness_weight), 0.0)
    smoothness_grid_size = max(int(spatial_smoothness_grid_size), 2)
    patches_per_observation = max(int(training_patches_per_observation), 1)
    pixel_sample_count = (
        max(int(pixel_samples_per_observation), 1)
        if pixel_samples_per_observation is not None
        else 0
    )
    uncropped_count = max(int(full_observations_per_step), 0)
    context_count = max(int(context_observations_per_step), 0)
    context_pool = max(int(context_downsample_factor), 1)
    context_weight = max(float(context_loss_weight), 0.0)
    spectral_penalty_weight = max(float(spectral_observability_weight), 0.0)
    spectral_penalty_grid = max(int(spectral_observability_probe_grid_size), 2)
    residual_penalty_weight = max(float(residual_head_penalty_weight), 0.0)
    smoothness_domain = getattr(model, "domain", None)
    ema_rate = float(ema_decay)
    use_ema = 0.0 < ema_rate < 1.0
    sampling_generator = torch.Generator(device="cpu")
    sampling_generator.manual_seed(12345)
    ema_params: list[Tensor] | None = None
    if use_ema:
        ema_params = [
            param.detach().clone()
            for param in model.parameters()
            if param.requires_grad
        ]
    for step_idx in range(int(steps)):
        progress = float(step_idx + 1) / max(int(steps), 1)
        set_progress = getattr(model, "set_curriculum_progress", None)
        if callable(set_progress):
            if curriculum == "none":
                set_progress(1.0)
            elif curriculum in {"spectral_frequency_ramp", "feature_scale_ramp"}:
                set_progress(progress)
            else:
                raise ValueError(f"unsupported curriculum {curriculum!r}")
        if lr_schedule == "constant":
            step_lr = base_lr
        elif lr_schedule == "cosine":
            frac = float(step_idx) / max(int(steps) - 1, 1)
            step_lr = base_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * frac)))
        else:
            raise ValueError(f"unsupported lr_schedule {lr_schedule!r}")
        if warmup_steps > 0 and step_idx < warmup_steps:
            warm_frac = float(step_idx + 1) / float(warmup_steps)
            step_lr = step_lr * max(warm_frac, 1.0e-3)
        for group in optimizer.param_groups:
            group["lr"] = step_lr
        optimizer.zero_grad(set_to_none=True)
        total_loss = None
        global_weighted_loss = None
        global_weight_sum = None
        band_weighted_loss = {}
        band_weight_sum = {}
        mixture_wavelengths = None
        mixture_probs = None
        if training_spectral_sampling == "mixture_mc":
            mixture_wavelengths, mixture_probs = _mixture_mc_samples(
                observations,
                num_spectral_samples=num_spectral_samples,
                device=next((p.device for p in model.parameters()), default_compute_device()),
                proposal_grid_size=training_spectral_sampling_grid_size,
            )
        selected_indices = _sample_observation_indices(
            observations,
            observations_per_step=observations_per_step,
            observation_sampling=observation_sampling,
            generator=sampling_generator,
        )
        selected_pairs: list[tuple[FieldObservation, ForwardContext | None]] = []
        if pixel_sample_count > 0:
            selected_pairs = [
                (observations[idx], obs_forward_ctx[idx]) for idx in selected_indices
            ]
        elif (
            shared_sky_patch_sampling
            and training_patch_size is not None
            and len(selected_indices) > uncropped_count
        ):
            uncropped_indices = selected_indices[:uncropped_count]
            cropped_indices = selected_indices[uncropped_count:]
            if uncropped_indices:
                selected_pairs.extend(
                    (observations[i], obs_forward_ctx[i]) for i in uncropped_indices
                )
            for _ in range(patches_per_observation):
                ref_position = int(
                    torch.randint(
                        len(cropped_indices),
                        (1,),
                        generator=sampling_generator,
                    ).item()
                )
                reference_observation = observations[cropped_indices[ref_position]]
                patch = min(
                    int(training_patch_size),
                    int(reference_observation.image.shape[-2]),
                    int(reference_observation.image.shape[-1]),
                )
                if patch <= 0:
                    continue
                ref_x0, ref_y0 = _sample_patch_origin(
                    reference_observation,
                    patch_size=patch,
                    generator=sampling_generator,
                )
                for observation_index in cropped_indices:
                    cropped = _crop_field_observation_to_shared_sky_patch(
                        observations[observation_index],
                        reference_observation=reference_observation,
                        x0=ref_x0,
                        y0=ref_y0,
                        patch_size=patch,
                    )
                    if cropped is not None:
                        selected_pairs.append(
                            (cropped, obs_forward_ctx[observation_index])
                        )
            if not selected_pairs:
                selected_pairs = [
                    (observations[idx], obs_forward_ctx[idx])
                    for idx in selected_indices
                ]
        else:
            for selected_rank, observation_index in enumerate(selected_indices):
                base_observation = observations[observation_index]
                base_ctx = obs_forward_ctx[observation_index]
                if training_patch_size is None or selected_rank < uncropped_count:
                    selected_pairs.append((base_observation, base_ctx))
                    continue
                patch = min(
                    int(training_patch_size),
                    int(base_observation.image.shape[-2]),
                    int(base_observation.image.shape[-1]),
                )
                if patch <= 0 or (
                    patch >= int(base_observation.image.shape[-2])
                    and patch >= int(base_observation.image.shape[-1])
                ):
                    selected_pairs.append((base_observation, base_ctx))
                    continue
                for _ in range(patches_per_observation):
                    x0, y0 = _sample_patch_origin(
                        base_observation,
                        patch_size=patch,
                        generator=sampling_generator,
                    )
                    selected_pairs.append(
                        (
                            _crop_field_observation(
                                base_observation,
                                x0=x0,
                                y0=y0,
                                width=patch,
                                height=patch,
                            ),
                            base_ctx,
                        )
                    )
        selected_observations = [o for o, _ in selected_pairs]
        context_pairs: list[tuple[FieldObservation, ForwardContext | None]] = []
        if context_count > 0 and context_weight > 0.0:
            context_indices = _sample_observation_indices(
                observations,
                observations_per_step=context_count,
                observation_sampling=observation_sampling,
                generator=sampling_generator,
            )
            context_pairs = [
                (observations[idx], obs_forward_ctx[idx]) for idx in context_indices
            ]

        streamed_total_loss = 0.0
        selected_global_weight_sum = None
        selected_band_weight_sum: dict[tuple[object, ...], float] = {}
        if streaming_accumulation:
            if loss_reduction in {"global", "band_global_mix"}:
                selected_global_weight_sum = sum(
                    _observation_weight_sum_scalar(observation)
                    for observation in selected_observations
                )
            if loss_reduction in {"per_band", "band_global_mix"}:
                for observation in selected_observations:
                    band_key = (
                        observation.spectral_min,
                        observation.spectral_max,
                        observation.wavelength,
                        observation.band_name,
                    )
                    selected_band_weight_sum[band_key] = (
                        selected_band_weight_sum.get(band_key, 0.0)
                        + _observation_weight_sum_scalar(observation)
                    )
        for observation, forward_context in selected_pairs:
            render_kwargs = {}
            if training_spectral_sampling == "mixture_mc":
                assert mixture_wavelengths is not None and mixture_probs is not None
                render_kwargs["override_wavelengths"] = mixture_wavelengths
                render_kwargs["override_weights"] = _mixture_importance_weights(
                    observation,
                    mixture_wavelengths,
                    mixture_probs,
                )
                render_kwargs["spectral_sampling"] = "deterministic_quadrature"
            else:
                render_kwargs["spectral_sampling"] = training_spectral_sampling
                render_kwargs["spectral_sampling_grid_size"] = (
                    training_spectral_sampling_grid_size
                )
            render_kwargs["forward_context"] = forward_context
            if (
                pixel_sample_count > 0
                and (
                    observation.psf_spec is None
                    or observation.psf_spec.mode == "none"
                )
            ):
                sampled = _sample_observation_pixels(
                    observation,
                    num_samples=pixel_sample_count,
                    generator=sampling_generator,
                    device=next((p.device for p in model.parameters()), default_compute_device()),
                )
                if sampled is None:
                    continue
                (
                    sample_ra_deg,
                    sample_dec_deg,
                    image,
                    weight,
                    support_count,
                    full_weight_sum,
                ) = sampled
                if loss_kind == "gaussian_nll":
                    pred, pred_logvar = render_field_samples_distribution(
                        model,
                        observation,
                        ra_deg=sample_ra_deg,
                        dec_deg=sample_dec_deg,
                        num_spectral_samples=num_spectral_samples,
                        **render_kwargs,
                    )
                else:
                    pred = render_field_samples(
                        model,
                        observation,
                        ra_deg=sample_ra_deg,
                        dec_deg=sample_dec_deg,
                        num_spectral_samples=num_spectral_samples,
                        **render_kwargs,
                    )
                    pred_logvar = None
                image = image.to(device=pred.device, dtype=pred.dtype)
                weight = weight.to(device=pred.device, dtype=pred.dtype)
                residual = pred - image
                sample_element_loss = weight * _field_residual_loss(
                    residual,
                    loss_kind=loss_kind,
                    charbonnier_eps=charbonnier_eps,
                    huber_delta=huber_delta,
                    logvar=pred_logvar,
                )
                weighted_loss = (
                    torch.mean(sample_element_loss)
                    * float(max(support_count, 1))
                )
                weight_sum = torch.as_tensor(
                    max(float(full_weight_sum), 1.0e-8),
                    device=pred.device,
                    dtype=pred.dtype,
                )
            elif loss_kind == "gaussian_nll":
                pred, pred_logvar = render_field_observation_distribution(
                    model,
                    observation,
                    num_spectral_samples=num_spectral_samples,
                    **render_kwargs,
                )
            else:
                pred = render_field_observation(
                    model,
                    observation,
                    num_spectral_samples=num_spectral_samples,
                    **render_kwargs,
                )
                pred_logvar = None
                image = observation.image.to(device=pred.device, dtype=pred.dtype)
                valid = _observation_support_mask(observation).to(device=pred.device)
                image = torch.where(valid, image, torch.zeros_like(image))
                residual = torch.where(valid, pred - image, torch.zeros_like(pred))
                weight = (
                    observation.weight.to(device=pred.device, dtype=pred.dtype)
                    if observation.weight is not None
                    else torch.ones_like(pred)
                )
                weight = torch.where(valid, weight, torch.zeros_like(weight))
                weighted_loss = torch.sum(
                    weight
                    * _field_residual_loss(
                        residual,
                        loss_kind=loss_kind,
                        charbonnier_eps=charbonnier_eps,
                        huber_delta=huber_delta,
                        logvar=pred_logvar,
                    )
                )
                weight_sum = weight.sum()
            band_key = (
                observation.spectral_min,
                observation.spectral_max,
                observation.wavelength,
                observation.band_name,
            )
            if streaming_accumulation:
                if loss_reduction == "per_observation":
                    obs_loss = weighted_loss / torch.clamp(weight_sum, min=1.0e-8)
                elif loss_reduction == "per_band":
                    denom = max(selected_band_weight_sum.get(band_key, 0.0), 1.0e-8)
                    obs_loss = weighted_loss / denom
                elif loss_reduction == "global":
                    denom = max(float(selected_global_weight_sum or 0.0), 1.0e-8)
                    obs_loss = weighted_loss / denom
                elif loss_reduction == "band_global_mix":
                    band_denom = max(selected_band_weight_sum.get(band_key, 0.0), 1.0e-8)
                    global_denom = max(float(selected_global_weight_sum or 0.0), 1.0e-8)
                    obs_loss = 0.5 * (
                        (weighted_loss / band_denom)
                        + (weighted_loss / global_denom)
                    )
                else:
                    raise ValueError(f"unsupported loss_reduction {loss_reduction!r}")
                streamed_total_loss += float(obs_loss.detach().cpu())
                obs_loss.backward()
            elif loss_reduction == "per_observation":
                denom = torch.clamp(weight_sum, min=1.0e-8)
                obs_loss = weighted_loss / denom
                total_loss = obs_loss if total_loss is None else total_loss + obs_loss
            elif loss_reduction == "per_band":
                band_weighted_loss[band_key] = (
                    weighted_loss
                    if band_key not in band_weighted_loss
                    else band_weighted_loss[band_key] + weighted_loss
                )
                band_weight_sum[band_key] = (
                    weight_sum
                    if band_key not in band_weight_sum
                    else band_weight_sum[band_key] + weight_sum
                )
            elif loss_reduction == "global":
                global_weighted_loss = (
                    weighted_loss
                    if global_weighted_loss is None
                    else global_weighted_loss + weighted_loss
                )
                global_weight_sum = (
                    weight_sum
                    if global_weight_sum is None
                    else global_weight_sum + weight_sum
                )
            elif loss_reduction == "band_global_mix":
                band_weighted_loss[band_key] = (
                    weighted_loss
                    if band_key not in band_weighted_loss
                    else band_weighted_loss[band_key] + weighted_loss
                )
                band_weight_sum[band_key] = (
                    weight_sum
                    if band_key not in band_weight_sum
                    else band_weight_sum[band_key] + weight_sum
                )
                global_weighted_loss = (
                    weighted_loss
                    if global_weighted_loss is None
                    else global_weighted_loss + weighted_loss
                )
                global_weight_sum = (
                    weight_sum
                    if global_weight_sum is None
                    else global_weight_sum + weight_sum
                )
            else:
                raise ValueError(f"unsupported loss_reduction {loss_reduction!r}")
        if streaming_accumulation:
            total_loss = torch.tensor(
                streamed_total_loss,
                dtype=torch.float32,
                device=next((p.device for p in model.parameters()), default_compute_device()),
            )
        elif loss_reduction == "global":
            assert global_weighted_loss is not None and global_weight_sum is not None
            total_loss = global_weighted_loss / torch.clamp(global_weight_sum, min=1.0e-8)
        elif loss_reduction == "per_band":
            total_loss = torch.stack(
                [
                    weighted / torch.clamp(band_weight_sum[band_key], min=1.0e-8)
                    for band_key, weighted in band_weighted_loss.items()
                ]
            ).sum()
        elif loss_reduction == "band_global_mix":
            assert global_weighted_loss is not None and global_weight_sum is not None
            band_loss = torch.stack(
                [
                    weighted / torch.clamp(band_weight_sum[band_key], min=1.0e-8)
                    for band_key, weighted in band_weighted_loss.items()
                ]
            ).sum()
            global_loss = global_weighted_loss / torch.clamp(global_weight_sum, min=1.0e-8)
            total_loss = 0.5 * (band_loss + global_loss)
        if (
            training_spectral_sampling == "mixture_mc"
            and anchor_count > 0
            and anchor_loss_weight > 0.0
        ):
            anchor_losses = []
            max_anchor_count = min(anchor_count, len(selected_observations))
            if max_anchor_count > 0:
                anchor_perm = torch.randperm(
                    len(selected_observations),
                    generator=sampling_generator,
                )[:max_anchor_count].tolist()
                for anchor_index in anchor_perm:
                    anchor_observation, anchor_fctx = selected_pairs[anchor_index]
                    anchor_pred = render_field_observation(
                        model,
                        anchor_observation,
                        num_spectral_samples=anchor_num_samples,
                        spectral_sampling="deterministic_quadrature",
                        forward_context=anchor_fctx,
                    )
                    anchor_image = anchor_observation.image.to(
                        device=anchor_pred.device,
                        dtype=anchor_pred.dtype,
                    )
                    anchor_weight = (
                        anchor_observation.weight.to(
                            device=anchor_pred.device,
                            dtype=anchor_pred.dtype,
                        )
                        if anchor_observation.weight is not None
                        else torch.ones_like(anchor_pred)
                    )
                    if anchor_observation.mask is not None:
                        anchor_valid = anchor_observation.mask.to(device=anchor_pred.device) > 0
                        anchor_image = torch.where(anchor_valid, anchor_image, torch.zeros_like(anchor_image))
                        anchor_weight = torch.where(anchor_valid, anchor_weight, torch.zeros_like(anchor_weight))
                    else:
                        anchor_valid = torch.ones_like(anchor_image, dtype=torch.bool)

                    anchor_residual = torch.where(anchor_valid, anchor_pred - anchor_image, torch.zeros_like(anchor_pred))

                    anchor_weighted_loss = torch.sum(
                        anchor_weight
                        * _field_residual_loss(
                            anchor_residual,
                            loss_kind=loss_kind,
                            charbonnier_eps=charbonnier_eps,
                            huber_delta=huber_delta,
                            logvar=None,
                        )
                    )
                    anchor_losses.append(
                        anchor_weighted_loss
                        / torch.clamp(anchor_weight.sum(), min=1.0e-8)
                    )
            if anchor_losses:
                anchor_loss = torch.stack(anchor_losses).mean()
                if streaming_accumulation:
                    scaled_anchor_loss = anchor_loss_weight * anchor_loss
                    streamed_total_loss += float(scaled_anchor_loss.detach().cpu())
                    scaled_anchor_loss.backward()
                    total_loss = torch.tensor(
                        streamed_total_loss,
                        dtype=torch.float32,
                        device=next((p.device for p in model.parameters()), default_compute_device()),
                    )
                else:
                    total_loss = total_loss + anchor_loss_weight * anchor_loss
        if component_anchor_terms:
            anchor_component_loss = field_component_anchor_loss(
                model,
                component_anchor_terms,
                num_spectral_samples=component_anchor_num,
                spectral_sampling="deterministic_quadrature",
                loss_kind=component_anchor_kind,
                charbonnier_eps=charbonnier_eps,
                huber_delta=huber_delta,
            )
            if streaming_accumulation:
                streamed_total_loss += float(anchor_component_loss.detach().cpu())
                anchor_component_loss.backward()
                total_loss = torch.tensor(
                    streamed_total_loss,
                    dtype=torch.float32,
                    device=next((p.device for p in model.parameters()), default_compute_device()),
                )
            else:
                total_loss = total_loss + anchor_component_loss
        if component_centroid_anchor_terms:
            centroid_anchor_loss = field_component_centroid_anchor_loss(
                model,
                component_centroid_anchor_terms,
                num_spectral_samples=component_anchor_num,
                spectral_sampling="deterministic_quadrature",
            )
            if streaming_accumulation:
                streamed_total_loss += float(centroid_anchor_loss.detach().cpu())
                centroid_anchor_loss.backward()
                total_loss = torch.tensor(
                    streamed_total_loss,
                    dtype=torch.float32,
                    device=next((p.device for p in model.parameters()), default_compute_device()),
                )
            else:
                total_loss = total_loss + centroid_anchor_loss
        if context_pairs and context_weight > 0.0:
            scaled_context_weight = context_weight / max(len(context_pairs), 1)
            for context_observation, ctx_f in context_pairs:
                context_render_kwargs = {}
                if training_spectral_sampling == "mixture_mc":
                    assert mixture_wavelengths is not None and mixture_probs is not None
                    context_render_kwargs["override_wavelengths"] = mixture_wavelengths
                    context_render_kwargs["override_weights"] = _mixture_importance_weights(
                        context_observation,
                        mixture_wavelengths,
                        mixture_probs,
                    )
                    context_render_kwargs["spectral_sampling"] = "deterministic_quadrature"
                else:
                    context_render_kwargs["spectral_sampling"] = training_spectral_sampling
                    context_render_kwargs["spectral_sampling_grid_size"] = (
                        training_spectral_sampling_grid_size
                    )
                context_render_kwargs["forward_context"] = ctx_f
                context_pred = render_field_observation(
                    model,
                    context_observation,
                    num_spectral_samples=num_spectral_samples,
                    **context_render_kwargs,
                )
                context_image = context_observation.image.to(
                    device=context_pred.device,
                    dtype=context_pred.dtype,
                )
                context_weight_map = (
                    context_observation.weight.to(
                        device=context_pred.device,
                        dtype=context_pred.dtype,
                    )
                    if context_observation.weight is not None
                    else torch.ones_like(context_pred)
                )
                if context_observation.mask is not None:
                    context_valid = context_observation.mask.to(device=context_pred.device) > 0
                    context_image = torch.where(context_valid, context_image, torch.zeros_like(context_image))
                    context_weight_map = torch.where(
                        context_valid,
                        context_weight_map,
                        torch.zeros_like(context_weight_map),
                    )
                else:
                    context_valid = torch.ones_like(context_image, dtype=torch.bool)

                pooled_pred = _avg_pool_map(context_pred, factor=context_pool)
                pooled_image = _avg_pool_map(context_image, factor=context_pool)
                pooled_weight = _avg_pool_map(context_weight_map, factor=context_pool)

                # Use pooled validity approximation: if weight is > 0, it is somewhat valid
                pooled_valid = pooled_weight > 0.0
                context_residual = torch.where(pooled_valid, pooled_pred - pooled_image, torch.zeros_like(pooled_pred))

                context_loss = torch.sum(
                    pooled_weight
                    * _field_residual_loss(
                        context_residual,
                        loss_kind=loss_kind,
                        charbonnier_eps=charbonnier_eps,
                        huber_delta=huber_delta,
                        logvar=None,
                    )
                ) / torch.clamp(pooled_weight.sum(), min=1.0e-8)
                scaled_context_loss = scaled_context_weight * context_loss
                if streaming_accumulation:
                    streamed_total_loss += float(scaled_context_loss.detach().cpu())
                    scaled_context_loss.backward()
                    total_loss = torch.tensor(
                        streamed_total_loss,
                        dtype=torch.float32,
                        device=next((p.device for p in model.parameters()), default_compute_device()),
                    )
                else:
                    total_loss = total_loss + scaled_context_loss
        if smoothness_weight > 0.0 and isinstance(smoothness_domain, FieldDomain):
            probe_ra, probe_dec = _smoothness_probe_grid(
                smoothness_domain,
                grid_size=smoothness_grid_size,
                device=next((p.device for p in model.parameters()), default_compute_device()),
            )
            probe = model(
                probe_ra,
                probe_dec,
                torch.full_like(
                    probe_ra,
                    float(smoothness_domain.wavelength_center),
                ),
            )
            diff_x = probe[:, 1:] - probe[:, :-1]
            diff_y = probe[1:, :] - probe[:-1, :]
            smoothness_loss = 0.5 * (
                torch.mean(torch.square(diff_x)) + torch.mean(torch.square(diff_y))
            )
            if streaming_accumulation:
                scaled_smoothness_loss = smoothness_weight * smoothness_loss
                streamed_total_loss += float(scaled_smoothness_loss.detach().cpu())
                scaled_smoothness_loss.backward()
                total_loss = torch.tensor(
                    streamed_total_loss,
                    dtype=torch.float32,
                    device=next((p.device for p in model.parameters()), default_compute_device()),
                )
            else:
                total_loss = total_loss + smoothness_weight * smoothness_loss
        if spectral_penalty_weight > 0.0:
            spectral_penalty_fn = getattr(model, "spectral_observability_penalty", None)
            if callable(spectral_penalty_fn):
                spectral_penalty = spectral_penalty_fn(
                    observations,
                    probe_grid_size=spectral_penalty_grid,
                )
                scaled_spectral_penalty = spectral_penalty_weight * spectral_penalty
                if streaming_accumulation:
                    streamed_total_loss += float(scaled_spectral_penalty.detach().cpu())
                    scaled_spectral_penalty.backward()
                    total_loss = torch.tensor(
                        streamed_total_loss,
                        dtype=torch.float32,
                        device=next((p.device for p in model.parameters()), default_compute_device()),
                    )
                else:
                    total_loss = total_loss + scaled_spectral_penalty
        if residual_penalty_weight > 0.0:
            residual_penalty_fn = getattr(model, "residual_head_penalty", None)
            if callable(residual_penalty_fn):
                residual_penalty = residual_penalty_fn()
                scaled_residual_penalty = residual_penalty_weight * residual_penalty
                if streaming_accumulation:
                    streamed_total_loss += float(scaled_residual_penalty.detach().cpu())
                    scaled_residual_penalty.backward()
                    total_loss = torch.tensor(
                        streamed_total_loss,
                        dtype=torch.float32,
                        device=next((p.device for p in model.parameters()), default_compute_device()),
                    )
                else:
                    total_loss = total_loss + scaled_residual_penalty
        assert total_loss is not None
        loss_history.append(float(total_loss.detach().cpu()))
        if not streaming_accumulation:
            total_loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
        optimizer.step()
        if use_ema and ema_params is not None:
            ema_index = 0
            with torch.no_grad():
                for param in model.parameters():
                    if not param.requires_grad:
                        continue
                    ema_params[ema_index].mul_(ema_rate).add_(
                        param.detach(),
                        alpha=1.0 - ema_rate,
                    )
                    ema_index += 1
    if use_ema and ema_params is not None:
        ema_index = 0
        with torch.no_grad():
            for param in model.parameters():
                if not param.requires_grad:
                    continue
                param.copy_(ema_params[ema_index])
                ema_index += 1
    model.eval()
    with torch.no_grad():
        final_losses = []
        global_weighted_loss = None
        global_weight_sum = None
        band_weighted_loss = {}
        band_weight_sum = {}
        for obs_idx, observation in enumerate(observations):
            fctx = obs_forward_ctx[obs_idx]
            if loss_kind == "gaussian_nll":
                pred, pred_logvar = render_field_observation_distribution(
                    model,
                    observation,
                    num_spectral_samples=num_spectral_samples,
                    forward_context=fctx,
                )
            else:
                pred = render_field_observation(
                    model,
                    observation,
                    num_spectral_samples=num_spectral_samples,
                    forward_context=fctx,
                )
                pred_logvar = None
            image = observation.image.to(device=pred.device, dtype=pred.dtype)
            valid = _observation_support_mask(observation).to(device=pred.device)
            image = torch.where(valid, image, torch.zeros_like(image))
            residual = torch.where(valid, pred - image, torch.zeros_like(pred))
            weight = (
                observation.weight.to(device=pred.device, dtype=pred.dtype)
                if observation.weight is not None
                else torch.ones_like(pred)
            )
            weight = torch.where(valid, weight, torch.zeros_like(weight))
            weighted_loss = torch.sum(
                weight
                * _field_residual_loss(
                    residual,
                    loss_kind=loss_kind,
                    charbonnier_eps=charbonnier_eps,
                    huber_delta=huber_delta,
                    logvar=pred_logvar,
                )
            )
            weight_sum = weight.sum()
            if loss_reduction == "per_observation":
                denom = torch.clamp(weight_sum, min=1.0e-8)
                final_losses.append(float(weighted_loss / denom))
            elif loss_reduction == "per_band":
                band_key = (
                    observation.spectral_min,
                    observation.spectral_max,
                    observation.wavelength,
                    observation.band_name,
                )
                band_weighted_loss[band_key] = (
                    weighted_loss
                    if band_key not in band_weighted_loss
                    else band_weighted_loss[band_key] + weighted_loss
                )
                band_weight_sum[band_key] = (
                    weight_sum
                    if band_key not in band_weight_sum
                    else band_weight_sum[band_key] + weight_sum
                )
            elif loss_reduction == "global":
                global_weighted_loss = (
                    weighted_loss
                    if global_weighted_loss is None
                    else global_weighted_loss + weighted_loss
                )
                global_weight_sum = (
                    weight_sum
                    if global_weight_sum is None
                    else global_weight_sum + weight_sum
                )
            elif loss_reduction == "band_global_mix":
                band_key = (
                    observation.spectral_min,
                    observation.spectral_max,
                    observation.wavelength,
                    observation.band_name,
                )
                band_weighted_loss[band_key] = (
                    weighted_loss
                    if band_key not in band_weighted_loss
                    else band_weighted_loss[band_key] + weighted_loss
                )
                band_weight_sum[band_key] = (
                    weight_sum
                    if band_key not in band_weight_sum
                    else band_weight_sum[band_key] + weight_sum
                )
                global_weighted_loss = (
                    weighted_loss
                    if global_weighted_loss is None
                    else global_weighted_loss + weighted_loss
                )
                global_weight_sum = (
                    weight_sum
                    if global_weight_sum is None
                    else global_weight_sum + weight_sum
                )
            else:
                raise ValueError(f"unsupported loss_reduction {loss_reduction!r}")
        if loss_reduction == "global":
            assert global_weighted_loss is not None and global_weight_sum is not None
            final_losses = [
                float(global_weighted_loss / torch.clamp(global_weight_sum, min=1.0e-8))
            ]
        elif loss_reduction == "per_band":
            final_losses = [
                float(weighted / torch.clamp(band_weight_sum[band_key], min=1.0e-8))
                for band_key, weighted in band_weighted_loss.items()
            ]
        elif loss_reduction == "band_global_mix":
            assert global_weighted_loss is not None and global_weight_sum is not None
            band_losses = [
                float(weighted / torch.clamp(band_weight_sum[band_key], min=1.0e-8))
                for band_key, weighted in band_weighted_loss.items()
            ]
            global_loss = float(global_weighted_loss / torch.clamp(global_weight_sum, min=1.0e-8))
            final_losses = [0.5 * (sum(band_losses) + global_loss)]
    return FieldFitResult(
        model=model,
        loss_history=tuple(loss_history),
        final_loss=float(sum(final_losses)),
    )
