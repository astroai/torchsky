"""Image-plane-first mapmaking primitives."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from .. import WCS, read, write


@dataclass
class ImageObservation:
    path: str
    hdu: int | str | None = 0
    weight: float = 1.0
    wavelength: float | None = None


@dataclass
class MapResult:
    data: Tensor
    weight: Tensor
    header: dict


@dataclass
class _PreparedImagePlaneTarget:
    header: dict
    wcs: WCS
    shape: tuple[int, int]
    ra_deg: Tensor
    dec_deg: Tensor


def _validate_image_plane(image: Tensor) -> Tensor:
    if image.ndim != 2:
        raise ValueError("Image-plane mapmaking currently supports 2D images only")
    return image


def _target_shape_from_header(
    header: dict,
    *,
    fallback: tuple[int, int],
) -> tuple[int, int]:
    naxis1 = header.get("NAXIS1")
    naxis2 = header.get("NAXIS2")
    if naxis1 is None or naxis2 is None:
        return fallback
    width = int(naxis1)
    height = int(naxis2)
    if width <= 0 or height <= 0:
        raise ValueError("Target header must define positive NAXIS1/NAXIS2")
    return height, width


def _build_target_world_grid(
    target_wcs: WCS, height: int, width: int, device: torch.device
) -> tuple[Tensor, Tensor]:
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float64, device=device),
        torch.arange(width, dtype=torch.float64, device=device),
        indexing="ij",
    )
    ra, dec = target_wcs.pixel_to_world(x, y)
    return ra, dec


def _resample_to_target(
    image: Tensor,
    source_wcs: WCS,
    target: _PreparedImagePlaneTarget,
) -> tuple[Tensor, Tensor]:
    height, width = target.shape
    image = _validate_image_plane(image).to(dtype=torch.float32)

    x_src, y_src = source_wcs.world_to_pixel(target.ra_deg, target.dec_deg)
    h_src, w_src = image.shape

    x_norm = (2.0 * x_src / max(w_src - 1, 1)) - 1.0
    y_norm = (2.0 * y_src / max(h_src - 1, 1)) - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1).to(dtype=torch.float32)

    finite_mask = torch.isfinite(image).to(dtype=torch.float32)
    image = torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

    sampled = (
        F.grid_sample(
            image.unsqueeze(0).unsqueeze(0),
            grid.unsqueeze(0),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        .squeeze(0)
        .squeeze(0)
    )
    sampled_mask = (
        F.grid_sample(
            finite_mask.unsqueeze(0).unsqueeze(0),
            grid.unsqueeze(0),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        .squeeze(0)
        .squeeze(0)
    )
    valid = (
        (x_src >= 0.0)
        & (x_src <= (w_src - 1))
        & (y_src >= 0.0)
        & (y_src <= (h_src - 1))
    ).to(dtype=sampled.dtype)
    valid = valid * (sampled_mask > 1e-6).to(dtype=sampled.dtype)
    sampled = torch.where(
        sampled_mask > 1e-6,
        sampled / torch.clamp(sampled_mask, min=1e-6),
        torch.zeros_like(sampled),
    )
    return sampled, valid


def _prepare_target(
    *,
    header: dict,
    target_wcs: WCS,
    target_shape: tuple[int, int],
    device: torch.device,
) -> _PreparedImagePlaneTarget:
    ra_deg, dec_deg = _build_target_world_grid(
        target_wcs,
        target_shape[0],
        target_shape[1],
        device=device,
    )
    return _PreparedImagePlaneTarget(
        header=header,
        wcs=target_wcs,
        shape=target_shape,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
    )


def coadd_image_plane(
    observations: Sequence[ImageObservation],
    *,
    target_header: Optional[dict] = None,
    output_path: Optional[str] = None,
) -> MapResult:
    if not observations:
        raise ValueError("At least one observation is required")

    first = observations[0]
    first_image, first_header = read(first.path, hdu=first.hdu, return_header=True)
    first_image = _validate_image_plane(first_image)
    header_dict = dict(first_header if target_header is None else target_header)
    target_wcs = WCS(header_dict)
    target_shape = _target_shape_from_header(
        header_dict,
        fallback=tuple(int(v) for v in first_image.shape[-2:]),
    )
    device = first_image.device
    prepared_target = _prepare_target(
        header=header_dict,
        target_wcs=target_wcs,
        target_shape=target_shape,
        device=device,
    )

    numer = torch.zeros(target_shape, dtype=torch.float32, device=device)
    denom = torch.zeros(target_shape, dtype=torch.float32, device=device)

    for obs in observations:
        if not torch.isfinite(torch.tensor(obs.weight, dtype=torch.float64)):
            raise ValueError("Observation weights must be finite")
        if obs.weight <= 0.0:
            raise ValueError("Observation weights must be positive")
        image, header = read(obs.path, hdu=obs.hdu, return_header=True)
        image = _validate_image_plane(image).to(device=device)
        header_dict_obs = dict(header)
        if (
            header_dict_obs == header_dict
            and tuple(int(v) for v in image.shape) == target_shape
        ):
            sampled = torch.nan_to_num(
                image.to(dtype=torch.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            valid = torch.isfinite(image).to(dtype=torch.float32)
        else:
            sampled, valid = _resample_to_target(
                image,
                WCS(header_dict_obs),
                prepared_target,
            )
        weight = float(obs.weight)
        numer = numer + sampled * valid * weight
        denom = denom + valid * weight

    coadd = numer / torch.clamp(denom, min=1e-8)
    result = MapResult(data=coadd, weight=denom, header=header_dict)
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        write(output_path, coadd, header=header_dict, overwrite=True)
    return result


class ImagePlaneMapMaker:
    """Differentiable image-plane coaddition baseline."""

    def __init__(self, *, target_header: Optional[dict] = None) -> None:
        self.target_header = target_header

    def run(
        self,
        observations: Sequence[ImageObservation],
        *,
        output_path: Optional[str] = None,
    ) -> MapResult:
        return coadd_image_plane(
            observations,
            target_header=self.target_header,
            output_path=output_path,
        )
