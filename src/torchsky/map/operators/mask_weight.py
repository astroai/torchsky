"""Weight/mask loading helpers shared across mapmaking paths."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from ...io import read
from .base import MaskWeightOp
from .image_grid import validate_image_plane_2d
from .noise import sanitize_inverse_variance


@dataclass
class ExplicitMaskWeightOp(MaskWeightOp):
    def __init__(self, *, name: str = "explicit_mask_weight") -> None:
        super().__init__(name=name, learned=False)


def load_image_observation_weight_map(
    obs: object,
    image: Tensor,
    *,
    device: torch.device,
) -> Tensor:
    weight_map = torch.isfinite(image).to(dtype=torch.float32, device=device)

    if getattr(obs, "weight_path", None) is not None:
        source_weight = read(
            getattr(obs, "weight_path"),
            hdu=getattr(obs, "weight_hdu"),
            device=str(device),
        )
        source_weight = validate_image_plane_2d(source_weight)
        if tuple(source_weight.shape) != tuple(image.shape):
            raise ValueError("Weight map shape must match the observation image shape")
        source_weight = source_weight.to(device=device, dtype=torch.float32)
        source_weight = torch.where(
            torch.isfinite(source_weight) & (source_weight > 0.0),
            source_weight,
            torch.zeros_like(source_weight),
        )
        weight_map = weight_map * source_weight

    if getattr(obs, "mask_path", None) is not None:
        mask = read(
            getattr(obs, "mask_path"), hdu=getattr(obs, "mask_hdu"), device=str(device)
        )
        mask = validate_image_plane_2d(mask)
        if tuple(mask.shape) != tuple(image.shape):
            raise ValueError("Mask shape must match the observation image shape")
        mask = mask.to(device=device, dtype=torch.float32)
        mask = torch.where(
            torch.isfinite(mask) & (mask > 0.0),
            torch.ones_like(mask),
            torch.zeros_like(mask),
        )
        weight_map = weight_map * mask

    if getattr(obs, "primary_beam_path", None) is not None:
        pb = read(
            getattr(obs, "primary_beam_path"),
            hdu=getattr(obs, "primary_beam_hdu"),
            device=str(device),
        )
        pb = validate_image_plane_2d(pb)
        if tuple(pb.shape) != tuple(image.shape):
            raise ValueError(
                "primary_beam map shape must match the observation image shape"
            )
        pb = pb.to(device=device, dtype=torch.float32)
        pb = torch.where(torch.isfinite(pb) & (pb > 0.0), pb, torch.zeros_like(pb))
        weight_map = weight_map * pb

    scalar_weight = float(getattr(obs, "weight", 1.0))
    if not math.isfinite(scalar_weight):
        raise ValueError("observation weight must be finite")
    scalar = torch.tensor(scalar_weight, dtype=weight_map.dtype, device=device)
    return weight_map * sanitize_inverse_variance(scalar)
