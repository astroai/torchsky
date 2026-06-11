"""Observation geometry, cropping, and pixel sampling helpers for field fitting."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from ... import WCS
from ..exposure_pose import ExposurePose
from .types import FieldObservation, FieldObservationSampling


def wrapped_delta_ra_deg(ra_a_deg: Tensor, ra_b_deg: float) -> Tensor:
    return ((ra_a_deg - float(ra_b_deg) + 180.0) % 360.0) - 180.0


def header_shape(header: dict, *, fallback: tuple[int, int]) -> tuple[int, int]:
    naxis1 = header.get("NAXIS1")
    naxis2 = header.get("NAXIS2")
    if naxis1 is None or naxis2 is None:
        return fallback
    return int(naxis2), int(naxis1)


def copy_header_with_crop(
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


def observation_band_key(observation: FieldObservation) -> tuple[object, ...]:
    return (
        observation.band_name,
        observation.spectral_min,
        observation.spectral_max,
        observation.wavelength,
        observation.transmission_center,
    )


def observation_support_mask(observation: FieldObservation) -> Tensor:
    support = torch.isfinite(observation.image)
    if observation.weight is not None:
        support = support & torch.isfinite(observation.weight) & (observation.weight > 0)
    if observation.mask is not None:
        support = support & torch.isfinite(observation.mask) & (observation.mask > 0)
    return support


def observation_weight_sum_scalar(observation: FieldObservation) -> float:
    if observation.weight is not None:
        weight = observation.weight.detach().to(dtype=torch.float32)
    else:
        weight = torch.ones_like(observation.image, dtype=torch.float32)
    if observation.mask is not None:
        valid = observation.mask.detach() > 0
        weight = torch.where(valid, weight, torch.zeros_like(weight))
    return float(weight.sum().item())


def observation_support_count_scalar(observation: FieldObservation) -> int:
    support = observation_support_mask(observation)
    return int(torch.count_nonzero(support).item())


def sample_observation_indices(
    observations: Sequence[FieldObservation],
    *,
    observations_per_step: int | None,
    observation_sampling: FieldObservationSampling,
    generator: torch.Generator,
) -> list[int]:
    total = len(observations)
    if (
        observations_per_step is None
        or observations_per_step <= 0
        or observations_per_step >= total
    ):
        return list(range(total))
    target = int(observations_per_step)
    if observation_sampling == "uniform":
        return torch.randperm(total, generator=generator)[:target].tolist()
    if observation_sampling != "band_balanced":
        raise ValueError(f"unsupported observation_sampling {observation_sampling!r}")
    band_groups: dict[tuple[object, ...], list[int]] = {}
    for idx, observation in enumerate(observations):
        band_groups.setdefault(observation_band_key(observation), []).append(idx)
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
        extra = torch.randperm(len(remainders), generator=generator)[
            : target - len(selected)
        ].tolist()
        selected.extend(remainders[i] for i in extra)
    elif len(selected) > target:
        keep = torch.randperm(len(selected), generator=generator)[:target].tolist()
        selected = [selected[i] for i in keep]
    return selected


def sample_patch_origin(
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
    support = observation_support_mask(observation)
    support_indices = torch.nonzero(support, as_tuple=False)
    if support_indices.numel() > 0:
        anchor_index = int(
            torch.randint(support_indices.shape[0], (1,), generator=generator).item()
        )
        anchor_y = int(support_indices[anchor_index, 0].item())
        anchor_x = int(support_indices[anchor_index, 1].item())
        y0 = min(max(anchor_y - patch // 2, 0), max_y)
        x0 = min(max(anchor_x - patch // 2, 0), max_x)
        return x0, y0
    x0 = (
        int(torch.randint(max_x + 1, (1,), generator=generator).item())
        if max_x > 0
        else 0
    )
    y0 = (
        int(torch.randint(max_y + 1, (1,), generator=generator).item())
        if max_y > 0
        else 0
    )
    return x0, y0


def crop_field_observation(
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
        header=copy_header_with_crop(
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
            observation.mask[y_slice, x_slice] if observation.mask is not None else None
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


def crop_field_observation_to_shared_sky_patch(
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
        return crop_field_observation(
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
    cropped = crop_field_observation(
        observation,
        x0=crop_x0,
        y0=crop_y0,
        width=crop_x1 - crop_x0,
        height=crop_y1 - crop_y0,
    )
    if not torch.any(observation_support_mask(cropped)):
        return None
    return cropped


def sample_observation_pixels(
    observation: FieldObservation,
    *,
    num_samples: int,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor, int, float] | None:
    support = observation_support_mask(observation)
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
        observation_weight_sum_scalar(observation),
    )


def avg_pool_map(image: Tensor, *, factor: int) -> Tensor:
    pool = max(int(factor), 1)
    if pool <= 1 or min(int(image.shape[-2]), int(image.shape[-1])) < pool:
        return image
    return (
        F.avg_pool2d(
            image.unsqueeze(0).unsqueeze(0),
            kernel_size=pool,
            stride=pool,
        )
        .squeeze(0)
        .squeeze(0)
    )


def coordinate_grid(
    observation_or_header: FieldObservation | dict,
    *,
    shape: tuple[int, int],
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    if isinstance(observation_or_header, FieldObservation):
        cached_ra = observation_or_header._cached_ra_deg
        cached_dec = observation_or_header._cached_dec_deg
        if cached_ra is not None and cached_dec is not None:
            if tuple(cached_ra.shape) == tuple(shape):
                return cached_ra.to(device=device), cached_dec.to(device=device)
            observation_or_header.clear_coordinate_cache()

    header = (
        observation_or_header.header
        if isinstance(observation_or_header, FieldObservation)
        else observation_or_header
    )
    height, width = header_shape(header, fallback=shape)
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float64, device=device),
        torch.arange(width, dtype=torch.float64, device=device),
        indexing="ij",
    )
    ra_deg, dec_deg = WCS(header).pixel_to_world(x, y)

    if isinstance(observation_or_header, FieldObservation):
        observation_or_header._cached_ra_deg = ra_deg.cpu()
        observation_or_header._cached_dec_deg = dec_deg.cpu()

    return ra_deg, dec_deg


def apply_exposure_pose_to_sky_grid(
    ra_deg: Tensor,
    dec_deg: Tensor,
    observation: FieldObservation,
    exposure_pose: ExposurePose | None,
) -> tuple[Tensor, Tensor]:
    if exposure_pose is None:
        return ra_deg, dec_deg
    center_ra = float(observation.header.get("CRVAL1", 0.0))
    center_dec = float(observation.header.get("CRVAL2", 0.0))
    return exposure_pose.apply_to_world(
        ra_deg, dec_deg, center_ra_deg=center_ra, center_dec_deg=center_dec
    )


__all__ = [
    "apply_exposure_pose_to_sky_grid",
    "avg_pool_map",
    "coordinate_grid",
    "copy_header_with_crop",
    "crop_field_observation",
    "crop_field_observation_to_shared_sky_patch",
    "header_shape",
    "observation_band_key",
    "observation_support_count_scalar",
    "observation_support_mask",
    "observation_weight_sum_scalar",
    "sample_observation_indices",
    "sample_observation_pixels",
    "sample_patch_origin",
    "wrapped_delta_ra_deg",
]
