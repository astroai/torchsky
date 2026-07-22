"""Rendering and evaluation helpers for continuous field models."""

from __future__ import annotations

from collections.abc import Iterator

import torch
import torch.nn as nn
from torch import Tensor

from ...device import default_compute_device
from ...spectral import SpectralAxis
from ...sphere import healpix as _healpix_wcs
from ...wcs import WCS

from ..healpix_image_plane import sample_healpix_map_on_image_header
from ..operators.background import (
    apply_background_components,
    observation_background_components,
)
from ..operators.bundle import (
    ObservationOperatorBundle,
    field_observation_operator_bundle,
)
from ..operators.psf import apply_psf_spec
from ..exposure_pose import ExposurePose

from .bandpass import (
    bandpass_mc_samples as _bandpass_mc_samples,
    bandpass_quadrature as _bandpass_quadrature,
)
from .forwarding import (
    call_forward_distribution as _call_forward_distribution,
    call_render_bandpass as _call_render_bandpass,
    call_render_bandpass_distribution as _call_render_bandpass_distribution,
    call_scalar_field_forward as _call_scalar_field_forward,
)
from .observation_ops import (
    apply_exposure_pose_to_sky_grid as _apply_exposure_pose_to_sky_grid,
    coordinate_grid as _coordinate_grid,
    header_shape as _header_shape,
)
from .types import (
    FieldObservation,
    FieldSpatialSampling,
    FieldSpectralSampling,
    ForwardContext,
)


def _spatial_quadrature_nodes(
    spatial_sampling: FieldSpatialSampling,
) -> tuple[tuple[float, float, float], ...]:
    if spatial_sampling == "center":
        return ((0.0, 0.0, 1.0),)
    if spatial_sampling == "gauss_legendre_2x2":
        offset = 0.5 / (3.0**0.5)
        return tuple(
            (offset_x, offset_y, 0.25)
            for offset_y in (-offset, offset)
            for offset_x in (-offset, offset)
        )
    raise ValueError(f"unsupported spatial_sampling {spatial_sampling!r}")


def _tile_world_coordinate_grid(
    wcs: WCS,
    *,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    offset_x: float,
    offset_y: float,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    grid_device = torch.device("cpu") if device.type == "mps" else device
    y, x = torch.meshgrid(
        torch.arange(y0, y1, dtype=torch.float64, device=grid_device) + offset_y,
        torch.arange(x0, x1, dtype=torch.float64, device=grid_device) + offset_x,
        indexing="ij",
    )
    ra_deg, dec_deg = wcs.pixel_to_world(x, y)
    output_dtype = torch.float32 if device.type == "mps" else ra_deg.dtype
    return (
        ra_deg.to(device=device, dtype=output_dtype),
        dec_deg.to(device=device, dtype=output_dtype),
    )


def _resolve_operator_bundle(
    observation: FieldObservation,
    operator_bundle: ObservationOperatorBundle | None,
) -> ObservationOperatorBundle:
    if operator_bundle is not None:
        return operator_bundle
    return field_observation_operator_bundle(observation)


def _render_spatially_integrated_distribution(
    model: nn.Module,
    observation: FieldObservation,
    *,
    wavelengths: Tensor,
    weights: Tensor,
    spatial_nodes: tuple[tuple[float, float, float], ...],
    spatial_tile_hw: tuple[int, int] | None,
    forward_context: ForwardContext | None,
    exposure_pose: ExposurePose | None,
    operator_bundle: ObservationOperatorBundle,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    image = observation.image
    height, width = _header_shape(
        observation.header,
        fallback=(int(image.shape[-2]), int(image.shape[-1])),
    )
    if spatial_tile_hw is not None and operator_bundle.psf is not None:
        raise ValueError(
            "spatially tiled distribution rendering cannot be used with a "
            "non-trivial PSF because the PSF couples pixels across tiles"
        )
    if spatial_tile_hw is None:
        tiles = [(0, height, 0, width)]
    else:
        tile_h, tile_w = _validate_spatial_tile_hw(spatial_tile_hw)
        tiles = list(_iter_spatial_tiles(height, width, tile_h, tile_w))
    sampling_wcs = WCS(observation.header)
    grid_device = torch.device("cpu") if device.type == "mps" else device
    sampling_wcs.to(grid_device)
    output_mean = torch.zeros((height, width), device=device, dtype=torch.float32)
    output_var = torch.zeros_like(output_mean)
    render_distribution = getattr(model, "render_bandpass_distribution", None)
    forward_distribution = getattr(model, "forward_distribution", None)
    for y0, y1, x0, x1 in tiles:
        patch_mean = torch.zeros(
            (y1 - y0, x1 - x0), device=device, dtype=torch.float32
        )
        patch_var = torch.zeros_like(patch_mean)
        for offset_x, offset_y, spatial_weight in spatial_nodes:
            ra_deg, dec_deg = _tile_world_coordinate_grid(
                sampling_wcs,
                y0=y0,
                y1=y1,
                x0=x0,
                x1=x1,
                offset_x=offset_x,
                offset_y=offset_y,
                device=device,
            )
            ra_deg, dec_deg = _apply_exposure_pose_to_sky_grid(
                ra_deg,
                dec_deg,
                observation,
                exposure_pose,
            )
            ra32 = ra_deg.to(dtype=torch.float32)
            dec32 = dec_deg.to(dtype=torch.float32)
            if callable(render_distribution):
                node_mean, node_logvar = _call_render_bandpass_distribution(
                    model,
                    ra32,
                    dec32,
                    wavelengths,
                    weights,
                    forward_context=forward_context,
                )
                node_var = torch.exp(node_logvar)
            elif callable(forward_distribution):
                means: list[Tensor] = []
                variances: list[Tensor] = []
                for wavelength in wavelengths:
                    mean, logvar = _call_forward_distribution(
                        model,
                        ra32,
                        dec32,
                        wavelength,
                        forward_context=forward_context,
                    )
                    means.append(mean)
                    variances.append(torch.exp(logvar))
                stacked_mean = torch.stack(means, dim=0)
                stacked_var = torch.stack(variances, dim=0)
                node_mean = torch.sum(
                    stacked_mean
                    * weights.view(-1, 1, 1).to(dtype=stacked_mean.dtype),
                    dim=0,
                )
                node_var = torch.sum(
                    stacked_var
                    * torch.square(weights).view(-1, 1, 1).to(
                        dtype=stacked_var.dtype
                    ),
                    dim=0,
                )
            else:
                node_mean = torch.zeros_like(patch_mean)
                for index, wavelength in enumerate(wavelengths):
                    plane = _call_scalar_field_forward(
                        model,
                        ra32,
                        dec32,
                        wavelength,
                        forward_context=forward_context,
                    )
                    node_mean += plane * weights[index].to(dtype=plane.dtype)
                node_var = torch.ones_like(node_mean)
            patch_mean += float(spatial_weight) * node_mean
            patch_var += float(spatial_weight) ** 2 * node_var
        output_mean[y0:y1, x0:x1] = patch_mean
        output_var[y0:y1, x0:x1] = patch_var
    if operator_bundle.psf is not None:
        output_mean = apply_psf_spec(output_mean, observation.psf_spec)
        output_var = apply_psf_spec(output_var, observation.psf_spec)
    if operator_bundle.background is not None:
        output_mean = apply_background_components(
            output_mean,
            **observation_background_components(
                observation,
                shape=tuple(output_mean.shape),
                device=output_mean.device,
                dtype=output_mean.dtype,
            ),
        )
    return (
        output_mean.to(dtype=image.dtype),
        torch.log(torch.clamp(output_var, min=1.0e-8)).to(dtype=image.dtype),
    )


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
    spatial_sampling: FieldSpatialSampling = "center",
    spatial_tile_hw: tuple[int, int] | None = None,
    forward_context: ForwardContext | None = None,
    exposure_pose: ExposurePose | None = None,
    operator_bundle: ObservationOperatorBundle | None = None,
) -> tuple[Tensor, Tensor]:
    image = observation.image
    if image.ndim != 2:
        raise ValueError("FieldObservation.image must be 2D")
    spatial_nodes = _spatial_quadrature_nodes(spatial_sampling)
    operator_bundle = _resolve_operator_bundle(observation, operator_bundle)
    device = next((p.device for p in model.parameters()), default_compute_device())
    ra_deg, dec_deg = _coordinate_grid(
        observation,
        shape=tuple(image.shape),
        device=device,
    )
    ra_deg, dec_deg = _apply_exposure_pose_to_sky_grid(
        ra_deg, dec_deg, observation, exposure_pose
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
    if spatial_sampling != "center":
        return _render_spatially_integrated_distribution(
            model,
            observation,
            wavelengths=wavelengths,
            weights=weights,
            spatial_nodes=spatial_nodes,
            spatial_tile_hw=spatial_tile_hw,
            forward_context=forward_context,
            exposure_pose=exposure_pose,
            operator_bundle=operator_bundle,
            device=device,
        )
    render_bandpass_distribution = getattr(model, "render_bandpass_distribution", None)
    operators_applied = False
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
            # ⚡ Bolt: Element-wise multiplication is faster than torch.square or **2
            var = torch.sum(
                stacked_var
                * torch.square(weights).view(-1, 1, 1).to(dtype=stacked_var.dtype),
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
                spatial_sampling=spatial_sampling,
                forward_context=forward_context,
                exposure_pose=exposure_pose,
                operator_bundle=operator_bundle,
            )
            logvar = torch.zeros_like(mean)
            operators_applied = True
    if operators_applied:
        return mean.to(dtype=image.dtype), logvar.to(dtype=image.dtype)
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
    exposure_pose: ExposurePose | None = None,
    operator_bundle: ObservationOperatorBundle | None = None,
) -> tuple[Tensor, Tensor]:
    ra_deg, dec_deg = _apply_exposure_pose_to_sky_grid(
        ra_deg, dec_deg, observation, exposure_pose
    )
    operator_bundle = _resolve_operator_bundle(observation, operator_bundle)
    if operator_bundle.psf is not None:
        raise ValueError("pixel-sampled field rendering does not support PSF blurring")
    if operator_bundle.background is not None:
        raise ValueError(
            "pixel-sampled field rendering does not support background terms"
        )
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
        # ⚡ Bolt: Element-wise multiplication is faster than torch.square or **2
        var = torch.sum(
            stacked_var
            * torch.square(weights).view(-1, 1, 1).to(dtype=stacked_var.dtype),
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
        operator_bundle=operator_bundle,
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
    spatial_sampling: FieldSpatialSampling = "center",
    spatial_tile_hw: tuple[int, int] | None = None,
    use_func_vmap: bool = False,
    func_vmap_chunk: int = 65536,
    forward_context: ForwardContext | None = None,
    exposure_pose: ExposurePose | None = None,
    operator_bundle: ObservationOperatorBundle | None = None,
) -> Tensor:
    image = observation.image
    if image.ndim != 2:
        raise ValueError("FieldObservation.image must be 2D")
    operator_bundle = _resolve_operator_bundle(observation, operator_bundle)
    device = next((p.device for p in model.parameters()), default_compute_device())
    spatial_nodes = _spatial_quadrature_nodes(spatial_sampling)
    center_coordinates: tuple[Tensor, Tensor] | None = None
    if spatial_sampling == "center":
        ra_deg, dec_deg = _coordinate_grid(
            observation,
            shape=tuple(image.shape),
            device=device,
        )
        center_coordinates = _apply_exposure_pose_to_sky_grid(
            ra_deg, dec_deg, observation, exposure_pose
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
    if callable(render_bandpass) and use_func_vmap:
        raise ValueError(
            "use_func_vmap is not supported when model.render_bandpass is defined"
        )
    if center_coordinates is not None:
        center_ra32 = center_coordinates[0].to(dtype=torch.float32)
        center_dec32 = center_coordinates[1].to(dtype=torch.float32)
        H, W = int(center_ra32.shape[-2]), int(center_ra32.shape[-1])
    else:
        center_ra32 = center_dec32 = None
        H, W = _header_shape(
            observation.header,
            fallback=(int(image.shape[-2]), int(image.shape[-1])),
        )
    sampling_wcs = WCS(observation.header)
    grid_device = torch.device("cpu") if device.type == "mps" else device
    sampling_wcs.to(grid_device)
    if spatial_tile_hw is None:
        tiles = [(0, H, 0, W)]
    else:
        th, tw = _validate_spatial_tile_hw(spatial_tile_hw)
        tiles = list(_iter_spatial_tiles(H, W, th, tw))
    rendered = torch.zeros((H, W), device=device, dtype=torch.float32)
    for y0, y1, x0, x1 in tiles:
        patch = torch.zeros((y1 - y0, x1 - x0), device=device, dtype=torch.float32)
        for offset_x, offset_y, spatial_weight in spatial_nodes:
            if center_ra32 is not None and center_dec32 is not None:
                ra_p = center_ra32[y0:y1, x0:x1]
                dec_p = center_dec32[y0:y1, x0:x1]
            else:
                ra_p, dec_p = _tile_world_coordinate_grid(
                    sampling_wcs,
                    y0=y0,
                    y1=y1,
                    x0=x0,
                    x1=x1,
                    offset_x=offset_x,
                    offset_y=offset_y,
                    device=device,
                )
                ra_p, dec_p = _apply_exposure_pose_to_sky_grid(
                    ra_p,
                    dec_p,
                    observation,
                    exposure_pose,
                )
                ra_p = ra_p.to(dtype=torch.float32)
                dec_p = dec_p.to(dtype=torch.float32)
            if callable(render_bandpass):
                node_plane = _call_render_bandpass(
                    model,
                    ra_p,
                    dec_p,
                    wavelengths,
                    weights,
                    forward_context=forward_context,
                )
            else:
                node_plane = torch.zeros_like(patch)
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
                    node_plane += plane * weights[i].to(dtype=plane.dtype)
            patch += node_plane * spatial_weight
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
    exposure_pose: ExposurePose | None = None,
    operator_bundle: ObservationOperatorBundle | None = None,
) -> Tensor:
    ra_deg, dec_deg = _apply_exposure_pose_to_sky_grid(
        ra_deg, dec_deg, observation, exposure_pose
    )
    operator_bundle = _resolve_operator_bundle(observation, operator_bundle)
    if operator_bundle.psf is not None:
        raise ValueError("pixel-sampled field rendering does not support PSF blurring")
    if operator_bundle.background is not None:
        raise ValueError(
            "pixel-sampled field rendering does not support background terms"
        )
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
    th, tw = int(tile_h), int(tile_w)
    for y0 in range(0, height, th):
        for x0 in range(0, width, tw):
            y1 = min(y0 + th, height)
            x1 = min(x0 + tw, width)
            yield y0, y1, x0, x1


def _validate_spatial_tile_hw(
    spatial_tile_hw: tuple[int, int],
) -> tuple[int, int]:
    th, tw = int(spatial_tile_hw[0]), int(spatial_tile_hw[1])
    if th <= 0 or tw <= 0:
        raise ValueError("spatial_tile_hw entries must be positive")
    return th, tw


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
        lam_flat = (
            lam_chunk.view(c, *([1] * len(hw_shape)))
            .expand(c, *hw_shape)
            .reshape(nflat)
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
        return torch.empty(
            (0, *tuple(ra_deg.shape)), device=device, dtype=torch.float32
        )
    ra_f = ra_deg.to(dtype=torch.float32)
    dec_f = dec_deg.to(dtype=torch.float32)
    H, W = int(ra_f.shape[0]), int(ra_f.shape[1])
    tiles: list[tuple[int, int, int, int]]
    if spatial_tile_hw is None:
        tiles = [(0, H, 0, W)]
    else:
        th, tw = _validate_spatial_tile_hw(spatial_tile_hw)
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

    Pixels are visited in increasing ``ipix`` order via :func:`torchsky.sphere.healpix.pix2ang`.
    Use ``spatial_chunk_pixels`` to limit the number of sky samples per ``model`` call
    (memory). Combine with ``spectral_chunk`` and optional ``use_func_vmap`` like
    :func:`render_field_cube`.
    """
    nside_i = int(nside)
    npix = _healpix_wcs.nside2npix(nside_i)
    dev = (
        device
        if device is not None
        else next((p.device for p in model.parameters()), default_compute_device())
    )
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
    dev = (
        device
        if device is not None
        else next((p.device for p in model.parameters()), default_compute_device())
    )
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
