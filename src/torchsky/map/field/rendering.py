"""Rendering and evaluation helpers for continuous field models.

Sky coordinates retain WCS precision until the model encodes local coordinates;
models should cast their encoded features to their parameter dtype. MPS grids
remain float32 because MPS cannot represent absolute sky coordinates in float64.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from ...device import default_compute_device
from ...spectral import SpectralAxis
from ...sphere import healpix as _healpix_wcs

from ..healpix_image_plane import sample_healpix_map_on_image_header
from ..operators.background import (
    LearnedBackgroundOp,
    apply_background_components,
    observation_background_components,
)
from ..operators.bundle import (
    ObservationOperatorBundle,
    field_observation_operator_bundle,
)
from ..operators.psf import (
    LearnedPSFOp,
    apply_psf_spec,
    apply_psf_variance_spec,
    psf_support_radius,
)
from ..exposure_pose import ExposurePose

from .bandpass import (
    bandpass_mc_samples as _bandpass_mc_samples,
    bandpass_quadrature as _bandpass_quadrature,
    validate_override_quadrature as _validate_override_quadrature,
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
    crop_field_observation as _crop_field_observation,
    header_shape as _header_shape,
    observation_pixel_to_world as _observation_pixel_to_world,
)
from .types import (
    FieldObservation,
    FieldSpatialSampling,
    FieldSpectralSampling,
    ForwardContext,
)


def field_model_has_predictive_distribution(model: nn.Module) -> bool:
    """Return whether a model provides an explicit predictive variance."""
    declared = getattr(model, "predictive_distribution_available", None)
    if declared is not None:
        return bool(declared)
    return callable(getattr(model, "render_bandpass_distribution", None)) or callable(
        getattr(model, "forward_distribution", None)
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


def _validate_pixel_response(
    observation: FieldObservation, spatial_sampling: FieldSpatialSampling,
    operator_bundle: ObservationOperatorBundle | None = None,
) -> None:
    bundle = _resolve_operator_bundle(observation, operator_bundle)
    if bundle.psf is None or spatial_sampling == "center":
        return
    pixel_response_included = bundle.psf.pixel_response_included
    if pixel_response_included is True:
        raise ValueError(
            "PSF already includes pixel response; use spatial_sampling='center'"
        )
    learned_spec = (
        observation.psf_spec is not None
        and observation.psf_spec.mode == "learned_lowrank"
    )
    if (
        isinstance(bundle.psf, LearnedPSFOp) or learned_spec
    ) and pixel_response_included is None:
        raise ValueError(
            "learned PSF pixel response convention is unknown; record whether the "
            "calibrated kernel includes detector pixel response"
        )


def iter_field_observation_tiles(
    model: nn.Module,
    observation: FieldObservation,
    *,
    spatial_tile_hw: tuple[int, int],
    **render_kwargs: Any,
) -> Iterator[tuple[slice, slice, Tensor]]:
    """Yield native detector cores rendered with their exact finite PSF margins.

    Keyword arguments match :func:`render_field_observation`. Spectral draws are
    shared across tiles. Consume and backward each core before requesting the
    next to bound saved activations; assembling a differentiable full image still
    retains every core's graph. Input/output storage itself remains proportional
    to image size. Discrete convolution is unchanged, not a certification of an
    optical PSF or subpixel accuracy.
    """
    tile_h, tile_w = _validate_spatial_tile_hw(spatial_tile_hw)
    bundle = _resolve_operator_bundle(observation, render_kwargs.get("operator_bundle"))
    if isinstance(bundle.psf, LearnedPSFOp):
        size = getattr(bundle.psf.model, "kernel_size", None)
        if size is None or int(size) <= 0 or int(size) % 2 != 1:
            raise ValueError("tiled learned PSF requires an explicit odd kernel_size")
        halo_y = halo_x = int(size) // 2
    elif bundle.psf is None:
        halo_y = halo_x = 0
    else:
        if (
            observation.psf_spec is not None
            and observation.psf_spec.mode == "learned_lowrank"
        ):
            raise ValueError(
                "tiled learned_lowrank PSF requires globally indexed cell geometry"
            )
        halo_y, halo_x = psf_support_radius(observation.psf_spec)
    device = next((p.device for p in model.parameters()), default_compute_device())
    if (
        render_kwargs.get("override_wavelengths") is None
        and render_kwargs.get("override_weights") is None
    ):
        sampling = render_kwargs.get("spectral_sampling", "deterministic_quadrature")
        count = render_kwargs.get("num_spectral_samples", 9)
        if sampling == "deterministic_quadrature":
            wavelengths, weights = _bandpass_quadrature(
                observation, num_spectral_samples=count, device=device
            )
        elif sampling == "bandpass_mc":
            wavelengths, weights = _bandpass_mc_samples(
                observation,
                num_spectral_samples=count,
                device=device,
                proposal_grid_size=render_kwargs.get("spectral_sampling_grid_size"),
                generator=render_kwargs.get("generator"),
            )
        else:
            raise ValueError(
                "mixture_mc tiles require shared override wavelengths and weights"
            )
        render_kwargs.update(override_wavelengths=wavelengths, override_weights=weights)
    height, width = map(int, observation.image.shape)
    for y0, y1, x0, x1 in _iter_spatial_tiles(height, width, tile_h, tile_w):
        hy0, hy1 = max(y0 - halo_y, 0), min(y1 + halo_y, height)
        hx0, hx1 = max(x0 - halo_x, 0), min(x1 + halo_x, width)
        cropped = _crop_field_observation(
            observation,
            x0=hx0,
            y0=hy0,
            width=hx1 - hx0,
            height=hy1 - hy0,
        )
        prediction = render_field_observation(model, cropped, **render_kwargs)
        yield (
            slice(y0, y1),
            slice(x0, x1),
            prediction[y0 - hy0 : y1 - hy0, x0 - hx0 : x1 - hx0],
        )
        del prediction, cropped


def _tile_world_coordinate_grid(
    observation: FieldObservation,
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
    ra_deg, dec_deg = _observation_pixel_to_world(observation, x, y)
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


def _detector_coordinate_tile(
    observation: FieldObservation,
    *,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    y, x = torch.meshgrid(
        torch.arange(y0, y1, device=device, dtype=torch.float32)
        + float(observation.detector_y0_pix),
        torch.arange(x0, x1, device=device, dtype=torch.float32)
        + float(observation.detector_x0_pix),
        indexing="ij",
    )
    return x, y


def _apply_learned_psf(
    image: Tensor,
    observation: FieldObservation,
    operator: LearnedPSFOp,
    *,
    wavelength: Tensor | float,
    forward_context: ForwardContext | None,
    spatial_tile_hw: tuple[int, int] | None,
) -> Tensor:
    psf_model = operator.model
    if psf_model is None:
        raise ValueError("LearnedPSFOp requires a model for rendering")
    height, width = map(int, image.shape)
    time_mjd = None if forward_context is None else forward_context.time_mjd
    if spatial_tile_hw is None or not hasattr(psf_model, "kernel_size"):
        x, y = _detector_coordinate_tile(
            observation,
            y0=0,
            y1=height,
            x0=0,
            x1=width,
            device=image.device,
        )
        return psf_model(
            image,
            spatial_x=x,
            spatial_y=y,
            wavelength=wavelength,
            time_mjd=time_mjd,
        )
    tile_h, tile_w = _validate_spatial_tile_hw(spatial_tile_hw)
    radius = int(getattr(psf_model, "kernel_size")) // 2
    output = torch.empty_like(image)
    for y0, y1, x0, x1 in _iter_spatial_tiles(height, width, tile_h, tile_w):
        halo_y0 = max(y0 - radius, 0)
        halo_y1 = min(y1 + radius, height)
        halo_x0 = max(x0 - radius, 0)
        halo_x1 = min(x1 + radius, width)
        x, y = _detector_coordinate_tile(
            observation,
            y0=halo_y0,
            y1=halo_y1,
            x0=halo_x0,
            x1=halo_x1,
            device=image.device,
        )
        filtered = psf_model(
            image[halo_y0:halo_y1, halo_x0:halo_x1],
            spatial_x=x,
            spatial_y=y,
            wavelength=wavelength,
            time_mjd=time_mjd,
        )
        output[y0:y1, x0:x1] = filtered[
            y0 - halo_y0 : y1 - halo_y0,
            x0 - halo_x0 : x1 - halo_x0,
        ]
    return output


def _conditioned_background_map(
    observation: FieldObservation,
    operator: LearnedBackgroundOp,
    *,
    wavelengths: Tensor,
    weights: Tensor,
    forward_context: ForwardContext | None,
    exposure_pose: ExposurePose | None,
    spatial_tile_hw: tuple[int, int] | None,
    device: torch.device,
) -> Tensor:
    background_model = operator.model
    if background_model is None:
        raise ValueError("LearnedBackgroundOp requires a model for rendering")
    height, width = map(int, observation.image.shape[-2:])
    if spatial_tile_hw is None:
        tiles = [(0, height, 0, width)]
    else:
        tile_h, tile_w = _validate_spatial_tile_hw(spatial_tile_hw)
        tiles = list(_iter_spatial_tiles(height, width, tile_h, tile_w))
    output = torch.zeros((height, width), device=device, dtype=torch.float32)
    time_mjd = None if forward_context is None else forward_context.time_mjd
    for y0, y1, x0, x1 in tiles:
        if operator.coordinate_frame == "detector":
            spatial_x, spatial_y = _detector_coordinate_tile(
                observation,
                y0=y0,
                y1=y1,
                x0=x0,
                x1=x1,
                device=device,
            )
        else:
            spatial_x, spatial_y = _tile_world_coordinate_grid(
                observation,
                y0=y0,
                y1=y1,
                x0=x0,
                x1=x1,
                offset_x=0.0,
                offset_y=0.0,
                device=device,
            )
            spatial_x, spatial_y = _apply_exposure_pose_to_sky_grid(
                spatial_x,
                spatial_y,
                observation,
                exposure_pose,
            )
        patch = torch.zeros_like(spatial_x, dtype=torch.float32)
        for wavelength, weight in zip(wavelengths, weights, strict=True):
            patch += background_model(
                spatial_x,
                spatial_y,
                wavelength,
                time_mjd,
            ).to(dtype=torch.float32) * weight.to(dtype=torch.float32)
        output[y0:y1, x0:x1] = patch
    return output


def _apply_observation_background(
    image: Tensor,
    observation: FieldObservation,
    operator_bundle: ObservationOperatorBundle,
    *,
    wavelengths: Tensor,
    weights: Tensor,
    forward_context: ForwardContext | None,
    exposure_pose: ExposurePose | None,
    spatial_tile_hw: tuple[int, int] | None,
) -> Tensor:
    output = image
    if isinstance(operator_bundle.background, LearnedBackgroundOp):
        output = output + _conditioned_background_map(
            observation,
            operator_bundle.background,
            wavelengths=wavelengths,
            weights=weights,
            forward_context=forward_context,
            exposure_pose=exposure_pose,
            spatial_tile_hw=spatial_tile_hw,
            device=image.device,
        ).to(dtype=image.dtype)
    if operator_bundle.background is not None:
        output = apply_background_components(
            output,
            **observation_background_components(
                observation,
                shape=tuple(output.shape),
                device=output.device,
                dtype=output.dtype,
            ),
        )
    return output


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
    if spatial_tile_hw is None:
        tiles = [(0, height, 0, width)]
    else:
        tile_h, tile_w = _validate_spatial_tile_hw(spatial_tile_hw)
        tiles = list(_iter_spatial_tiles(height, width, tile_h, tile_w))
    output_mean = torch.zeros((height, width), device=device, dtype=torch.float32)
    output_var = torch.zeros_like(output_mean)
    render_distribution = getattr(model, "render_bandpass_distribution", None)
    forward_distribution = getattr(model, "forward_distribution", None)
    for y0, y1, x0, x1 in tiles:
        patch_mean = torch.zeros((y1 - y0, x1 - x0), device=device, dtype=torch.float32)
        patch_var = torch.zeros_like(patch_mean)
        for offset_x, offset_y, spatial_weight in spatial_nodes:
            ra_deg, dec_deg = _tile_world_coordinate_grid(
                observation,
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
            if callable(render_distribution):
                node_mean, node_logvar = _call_render_bandpass_distribution(
                    model,
                    ra_deg,
                    dec_deg,
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
                        ra_deg,
                        dec_deg,
                        wavelength,
                        forward_context=forward_context,
                    )
                    means.append(mean)
                    variances.append(torch.exp(logvar))
                stacked_mean = torch.stack(means, dim=0)
                stacked_var = torch.stack(variances, dim=0)
                node_mean = torch.sum(
                    stacked_mean * weights.view(-1, 1, 1).to(dtype=stacked_mean.dtype),
                    dim=0,
                )
                node_var = torch.sum(
                    stacked_var
                    * torch.square(weights).view(-1, 1, 1).to(dtype=stacked_var.dtype),
                    dim=0,
                )
            else:
                raise ValueError(
                    "field model does not provide an explicit predictive distribution"
                )
            patch_mean += float(spatial_weight) * node_mean
            patch_var += float(spatial_weight) ** 2 * node_var
        output_mean[y0:y1, x0:x1] = patch_mean
        output_var[y0:y1, x0:x1] = patch_var
    if operator_bundle.psf is not None:
        output_mean = apply_psf_spec(output_mean, observation.psf_spec)
        output_var = apply_psf_variance_spec(output_var, observation.psf_spec)
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
    _validate_pixel_response(observation, spatial_sampling, operator_bundle)
    image = observation.image
    if image.ndim != 2:
        raise ValueError("FieldObservation.image must be 2D")
    spatial_nodes = _spatial_quadrature_nodes(spatial_sampling)
    operator_bundle = _resolve_operator_bundle(observation, operator_bundle)
    if isinstance(operator_bundle.psf, LearnedPSFOp) or isinstance(
        operator_bundle.background, LearnedBackgroundOp
    ):
        raise ValueError(
            "distribution rendering with learned PSF/background operators is not "
            "implemented; use a value loss until covariance propagation is enabled"
        )
    if not field_model_has_predictive_distribution(model):
        raise ValueError(
            "field model does not provide an explicit predictive distribution"
        )
    device = next((p.device for p in model.parameters()), default_compute_device())
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
    if spatial_sampling != "center" or spatial_tile_hw is not None:
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
    ra_deg, dec_deg = _coordinate_grid(
        observation,
        shape=tuple(image.shape),
        device=device,
    )
    ra_deg, dec_deg = _apply_exposure_pose_to_sky_grid(
        ra_deg, dec_deg, observation, exposure_pose
    )
    render_bandpass_distribution = getattr(model, "render_bandpass_distribution", None)
    if callable(render_bandpass_distribution):
        mean, logvar = _call_render_bandpass_distribution(
            model,
            ra_deg,
            dec_deg,
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
                    ra_deg,
                    dec_deg,
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
            raise ValueError(
                "field model does not provide an explicit predictive distribution"
            )
    if operator_bundle.psf is not None:
        mean = apply_psf_spec(mean, observation.psf_spec)
        variance = apply_psf_variance_spec(torch.exp(logvar), observation.psf_spec)
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
    if not field_model_has_predictive_distribution(model):
        raise ValueError(
            "field model does not provide an explicit predictive distribution"
        )
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
            ra_deg,
            dec_deg,
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
                ra_deg,
                dec_deg,
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
    raise ValueError("field model does not provide an explicit predictive distribution")


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
    _validate_pixel_response(observation, spatial_sampling, operator_bundle)
    if spatial_tile_hw is not None:
        rendered = None
        for ys, xs, prediction in iter_field_observation_tiles(
            model,
            observation,
            spatial_tile_hw=spatial_tile_hw,
            num_spectral_samples=num_spectral_samples,
            spectral_sampling=spectral_sampling,
            spectral_sampling_grid_size=spectral_sampling_grid_size,
            generator=generator,
            override_wavelengths=override_wavelengths,
            override_weights=override_weights,
            spatial_sampling=spatial_sampling,
            use_func_vmap=use_func_vmap,
            func_vmap_chunk=func_vmap_chunk,
            forward_context=forward_context,
            exposure_pose=exposure_pose,
            operator_bundle=operator_bundle,
        ):
            if rendered is None:
                rendered = prediction.new_empty(observation.image.shape)
            rendered[ys, xs] = prediction
        if rendered is None:
            raise ValueError("field observation must have nonempty spatial dimensions")
        return rendered
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
        wavelengths, weights = _validate_override_quadrature(
            override_wavelengths,
            override_weights,
            device=device,
            dtype=torch.float32,
        )
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
        center_ra = center_coordinates[0]
        center_dec = center_coordinates[1]
        H, W = int(center_ra.shape[-2]), int(center_ra.shape[-1])
    else:
        center_ra = center_dec = None
        H, W = _header_shape(
            observation.header,
            fallback=(int(image.shape[-2]), int(image.shape[-1])),
        )
    if spatial_tile_hw is None:
        tiles = [(0, H, 0, W)]
    else:
        th, tw = _validate_spatial_tile_hw(spatial_tile_hw)
        tiles = list(_iter_spatial_tiles(H, W, th, tw))
    if isinstance(operator_bundle.psf, LearnedPSFOp):
        rendered = torch.zeros((H, W), device=device, dtype=torch.float32)
        for wavelength, spectral_weight in zip(wavelengths, weights, strict=True):
            monochromatic = torch.zeros_like(rendered)
            for y0, y1, x0, x1 in tiles:
                patch = torch.zeros(
                    (y1 - y0, x1 - x0), device=device, dtype=torch.float32
                )
                for offset_x, offset_y, spatial_weight in spatial_nodes:
                    if center_ra is not None and center_dec is not None:
                        ra_p = center_ra[y0:y1, x0:x1]
                        dec_p = center_dec[y0:y1, x0:x1]
                    else:
                        ra_p, dec_p = _tile_world_coordinate_grid(
                            observation,
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
                    patch += plane * spatial_weight
                monochromatic[y0:y1, x0:x1] = patch
            rendered += _apply_learned_psf(
                monochromatic,
                observation,
                operator_bundle.psf,
                wavelength=wavelength,
                forward_context=forward_context,
                spatial_tile_hw=spatial_tile_hw,
            ) * spectral_weight.to(dtype=rendered.dtype)
        rendered = _apply_observation_background(
            rendered,
            observation,
            operator_bundle,
            wavelengths=wavelengths,
            weights=weights,
            forward_context=forward_context,
            exposure_pose=exposure_pose,
            spatial_tile_hw=spatial_tile_hw,
        )
        return rendered.to(dtype=image.dtype)
    rendered = torch.zeros((H, W), device=device, dtype=torch.float32)
    for y0, y1, x0, x1 in tiles:
        patch = torch.zeros((y1 - y0, x1 - x0), device=device, dtype=torch.float32)
        for offset_x, offset_y, spatial_weight in spatial_nodes:
            if center_ra is not None and center_dec is not None:
                ra_p = center_ra[y0:y1, x0:x1]
                dec_p = center_dec[y0:y1, x0:x1]
            else:
                ra_p, dec_p = _tile_world_coordinate_grid(
                    observation,
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
    rendered = _apply_observation_background(
        rendered,
        observation,
        operator_bundle,
        wavelengths=wavelengths,
        weights=weights,
        forward_context=forward_context,
        exposure_pose=exposure_pose,
        spatial_tile_hw=spatial_tile_hw,
    )
    return rendered.to(dtype=image.dtype)


def render_spectral_power_foam_observation_exact(
    model: nn.Module,
    observation: FieldObservation,
    *,
    additive_model: nn.Module | None = None,
    additive_spatial_sampling: FieldSpatialSampling = "center",
    num_spectral_samples: int = 9,
    operator_kwargs: dict[str, object] | None = None,
    forward_context: ForwardContext | None = None,
    operator_bundle: ObservationOperatorBundle | None = None,
    spatial_tile_hw: tuple[int, int] | None = None,
) -> Tensor:
    """Hard-render a spectral Power Foam through target pixels and common operators."""
    if observation.image.ndim != 2:
        raise ValueError("FieldObservation.image must be 2D")
    _validate_pixel_response(
        observation,
        "gauss_legendre_2x2",
        operator_bundle=operator_bundle,
    )
    normalized_spatial_coordinates = getattr(
        model,
        "normalized_spatial_coordinates",
        None,
    )
    render_exact = getattr(model, "render_exact", None)
    if not callable(normalized_spatial_coordinates) or not callable(render_exact):
        raise TypeError(
            "model must provide normalized_spatial_coordinates and render_exact"
        )
    device = next(
        (parameter.device for parameter in model.parameters()),
        default_compute_device(),
    )
    height, width = _header_shape(
        observation.header,
        fallback=(int(observation.image.shape[-2]), int(observation.image.shape[-1])),
    )
    grid_device = torch.device("cpu") if device.type == "mps" else device
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float64, device=grid_device),
        torch.arange(width, dtype=torch.float64, device=grid_device),
        indexing="ij",
    )
    offsets = torch.tensor(
        [
            [-0.5, -0.5],
            [0.5, -0.5],
            [0.5, 0.5],
            [-0.5, 0.5],
        ],
        dtype=torch.float64,
        device=grid_device,
    )
    corner_x = x[..., None] + offsets[:, 0]
    corner_y = y[..., None] + offsets[:, 1]
    ra_deg, dec_deg = _observation_pixel_to_world(observation, corner_x, corner_y)
    polygons = normalized_spatial_coordinates(
        ra_deg.to(device=device),
        dec_deg.to(device=device),
    ).reshape(height * width, 4, 2)
    wavelengths, weights = _bandpass_quadrature(
        observation,
        num_spectral_samples=num_spectral_samples,
        device=device,
    )
    next_vertices = torch.roll(polygons, shifts=-1, dims=-2)
    pixel_areas = 0.5 * torch.abs(
        torch.sum(
            polygons[..., 0] * next_vertices[..., 1]
            - polygons[..., 1] * next_vertices[..., 0],
            dim=-1,
        )
    )
    if bool(torch.any(pixel_areas <= 0)):
        raise ValueError("target WCS produced a degenerate pixel polygon")
    operator_bundle = _resolve_operator_bundle(observation, operator_bundle)
    empty_operator_bundle = ObservationOperatorBundle()
    if isinstance(operator_bundle.psf, LearnedPSFOp):
        rendered = torch.zeros((height, width), device=device, dtype=torch.float32)
        for wavelength, weight in zip(wavelengths, weights, strict=True):
            pixel_integrals = render_exact(
                polygons,
                wavelength.reshape(1),
                torch.ones_like(wavelength).reshape(1),
                semantics="surface_brightness",
                **dict(operator_kwargs or {}),
            )
            monochromatic = (pixel_integrals / pixel_areas).reshape(height, width)
            if additive_model is not None:
                monochromatic = monochromatic + render_field_observation(
                    additive_model,
                    observation,
                    num_spectral_samples=1,
                    override_wavelengths=wavelength.reshape(1),
                    override_weights=torch.ones_like(wavelength).reshape(1),
                    spatial_sampling=additive_spatial_sampling,
                    spatial_tile_hw=spatial_tile_hw,
                    forward_context=forward_context,
                    operator_bundle=empty_operator_bundle,
                ).to(device=device, dtype=monochromatic.dtype)
            rendered += _apply_learned_psf(
                monochromatic,
                observation,
                operator_bundle.psf,
                wavelength=wavelength,
                forward_context=forward_context,
                spatial_tile_hw=spatial_tile_hw,
            ) * weight.to(dtype=rendered.dtype)
    else:
        pixel_integrals = render_exact(
            polygons,
            wavelengths,
            weights,
            semantics="surface_brightness",
            **dict(operator_kwargs or {}),
        )
        rendered = (pixel_integrals / pixel_areas).reshape(height, width)
        if additive_model is not None:
            rendered = rendered + render_field_observation(
                additive_model,
                observation,
                num_spectral_samples=num_spectral_samples,
                override_wavelengths=wavelengths,
                override_weights=weights,
                spatial_sampling=additive_spatial_sampling,
                spatial_tile_hw=spatial_tile_hw,
                forward_context=forward_context,
                operator_bundle=empty_operator_bundle,
            ).to(device=device, dtype=rendered.dtype)
    if operator_bundle.psf is not None and not isinstance(
        operator_bundle.psf, LearnedPSFOp
    ):
        rendered = apply_psf_spec(rendered, observation.psf_spec)
    rendered = _apply_observation_background(
        rendered,
        observation,
        operator_bundle,
        wavelengths=wavelengths,
        weights=weights,
        forward_context=forward_context,
        exposure_pose=None,
        spatial_tile_hw=spatial_tile_hw,
    )
    return rendered.to(dtype=observation.image.dtype)


def render_diffuse_foam_hybrid_observation_exact(
    model: nn.Module,
    observation: FieldObservation,
    *,
    num_spectral_samples: int = 9,
    additive_spatial_sampling: FieldSpatialSampling = "center",
    operator_kwargs: dict[str, object] | None = None,
    forward_context: ForwardContext | None = None,
    operator_bundle: ObservationOperatorBundle | None = None,
    spatial_tile_hw: tuple[int, int] | None = None,
) -> Tensor:
    """Render the combined diffuse sky plus exact hard-cell Foam component."""
    diffuse_model = getattr(model, "diffuse_field", None)
    localized_foam = getattr(model, "localized_foam", None)
    if not isinstance(diffuse_model, nn.Module) or not isinstance(
        localized_foam, nn.Module
    ):
        raise TypeError("model must provide diffuse_field and localized_foam modules")
    if getattr(model, "component_identifiability", None) != "combined_only":
        raise ValueError("hybrid model must declare combined-only identifiability")
    return render_spectral_power_foam_observation_exact(
        localized_foam,
        observation,
        additive_model=diffuse_model,
        additive_spatial_sampling=additive_spatial_sampling,
        num_spectral_samples=num_spectral_samples,
        operator_kwargs=operator_kwargs,
        forward_context=forward_context,
        operator_bundle=operator_bundle,
        spatial_tile_hw=spatial_tile_hw,
    )


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
        wavelengths, weights = _validate_override_quadrature(
            override_wavelengths,
            override_weights,
            device=ra_deg.device,
            dtype=torch.float32,
        )
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
            ra_deg,
            dec_deg,
            wavelengths,
            weights,
            forward_context=forward_context,
        )
    spectral_samples = []
    for wavelength in wavelengths:
        spectral_samples.append(
            _call_scalar_field_forward(
                model,
                ra_deg,
                dec_deg,
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
    ra = torch.as_tensor(ra_deg).reshape(-1)
    dec = torch.as_tensor(dec_deg).reshape(-1)
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
    ra_f = ra_grid.reshape(-1)
    dec_f = dec_grid.reshape(-1)
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
    ra_f = ra_grid
    dec_f = dec_grid
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
    ra_f = ra_deg
    dec_f = dec_deg
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
        coordinate_dtype = torch.float32 if dev.type == "mps" else ra.dtype
        ra = ra.to(device=dev, dtype=coordinate_dtype)
        dec = dec.to(device=dev, dtype=coordinate_dtype)
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
