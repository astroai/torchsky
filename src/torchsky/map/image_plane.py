"""Image-plane-first mapmaking primitives."""

from __future__ import annotations

from dataclasses import dataclass
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
    target_wcs: WCS,
    target_shape: tuple[int, int],
) -> tuple[Tensor, Tensor]:
    height, width = target_shape
    image = image.to(dtype=torch.float32)
    if image.ndim != 2:
        raise ValueError("Image-plane mapmaking currently supports 2D images only")

    device = image.device
    ra, dec = _build_target_world_grid(target_wcs, height, width, device=device)
    x_src, y_src = source_wcs.world_to_pixel(ra, dec)
    h_src, w_src = image.shape

    x_norm = (2.0 * x_src / max(w_src - 1, 1)) - 1.0
    y_norm = (2.0 * y_src / max(h_src - 1, 1)) - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1).to(dtype=torch.float32)

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
    valid = (
        (x_src >= 0.0)
        & (x_src <= (w_src - 1))
        & (y_src >= 0.0)
        & (y_src <= (h_src - 1))
    ).to(dtype=sampled.dtype)
    return sampled, valid


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
    header_dict = dict(first_header if target_header is None else target_header)
    target_wcs = WCS(header_dict)
    target_shape = tuple(int(v) for v in first_image.shape[-2:])

    numer = torch.zeros(target_shape, dtype=torch.float32)
    denom = torch.zeros(target_shape, dtype=torch.float32)

    for obs in observations:
        image, header = read(obs.path, hdu=obs.hdu, return_header=True)
        header_dict_obs = dict(header)
        if (
            header_dict_obs == header_dict
            and tuple(int(v) for v in image.shape[-2:]) == target_shape
        ):
            sampled = image.to(dtype=torch.float32)
            valid = torch.ones_like(sampled, dtype=torch.float32)
        else:
            sampled, valid = _resample_to_target(
                image,
                WCS(header_dict_obs),
                target_wcs,
                target_shape,
            )
        weight = float(obs.weight)
        numer = numer + sampled * valid * weight
        denom = denom + valid * weight

    coadd = numer / torch.clamp(denom, min=1e-8)
    result = MapResult(data=coadd, weight=denom, header=header_dict)
    if output_path is not None:
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
