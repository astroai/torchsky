"""Weight/mask loading helpers shared across mapmaking paths."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from torchfits import read
from .base import MaskWeightOp
from .image_grid import validate_image_plane_2d
from .noise import sanitize_inverse_variance


@dataclass
class ExplicitMaskWeightOp(MaskWeightOp):
    def __init__(self, *, name: str = "explicit_mask_weight") -> None:
        super().__init__(name=name, learned=False)


def _load_or_validate_aux_plane(
    tensor_or_none: Tensor | None,
    path_or_none: str | None,
    hdu: int | str | None,
    *,
    image_shape: tuple[int, int],
    device: torch.device,
    label: str,
    preserve_integer: bool = False,
) -> Tensor | None:
    if tensor_or_none is not None:
        plane = validate_image_plane_2d(torch.as_tensor(tensor_or_none))
    elif path_or_none is not None:
        plane = validate_image_plane_2d(
            read(path_or_none, hdu=hdu, device=str(device))
        )
    else:
        return None
    if tuple(plane.shape) != tuple(image_shape):
        raise ValueError(f"{label} shape must match the observation image shape")
    if preserve_integer and not plane.is_floating_point():
        return plane.to(device=device)
    return plane.to(device=device, dtype=torch.float32)


def load_image_observation_weight_map(
    obs: object,
    image: Tensor,
    *,
    device: torch.device,
) -> Tensor:
    image_shape = (int(image.shape[0]), int(image.shape[1]))
    weight_map = torch.isfinite(image).to(dtype=torch.float32, device=device)

    weight_type = str(getattr(obs, "weight_type", "weight") or "weight").strip().lower()
    if weight_type not in {
        "weight",
        "ivm",
        "rms",
        "variance",
        "exptime",
        "expsq",
        "background",
    }:
        raise ValueError(
            f"Unsupported weight_type={weight_type!r}; expected one of "
            "'weight', 'ivm', 'rms', 'variance', 'exptime', 'expsq', 'background'"
        )
    weight_thresh = getattr(obs, "weight_thresh", None)
    if weight_thresh is not None:
        weight_thresh = float(weight_thresh)
        if not math.isfinite(weight_thresh) or weight_thresh < 0.0:
            raise ValueError("weight_thresh must be a finite non-negative float")

    source_weight = _load_or_validate_aux_plane(
        getattr(obs, "weight_map", None),
        getattr(obs, "weight_path", None),
        getattr(obs, "weight_hdu", 0),
        image_shape=image_shape,
        device=device,
        label="Weight map",
    )
    if source_weight is None:
        var_planes = [
            getattr(obs, attr, None)
            for attr in ("var_poisson", "var_rnoise", "var_flat")
            if getattr(obs, attr, None) is not None
        ]
        if var_planes:
            total_var = torch.zeros(image_shape, dtype=torch.float32, device=device)
            valid_var = torch.ones(image_shape, dtype=torch.bool, device=device)
            for vp in var_planes:
                vp_t = validate_image_plane_2d(torch.as_tensor(vp)).to(
                    device=device, dtype=torch.float32
                )
                if tuple(vp_t.shape) != image_shape:
                    raise ValueError(
                        "Variance component shape must match the observation image shape"
                    )
                valid_var = valid_var & torch.isfinite(vp_t) & (vp_t >= 0.0)
                total_var = total_var + torch.where(
                    torch.isfinite(vp_t) & (vp_t >= 0.0),
                    vp_t,
                    torch.zeros_like(vp_t),
                )
            source_weight = torch.where(
                valid_var & (total_var > 0.0),
                total_var.reciprocal(),
                torch.zeros_like(total_var),
            )

    if source_weight is not None:
        valid_w = torch.isfinite(source_weight) & (source_weight > 0.0)
        if weight_type == "rms":
            if weight_thresh is not None:
                valid_w = valid_w & (source_weight <= weight_thresh)
            source_weight = torch.where(
                valid_w,
                torch.square(source_weight).reciprocal(),
                torch.zeros_like(source_weight),
            )
        elif weight_type == "variance":
            if weight_thresh is not None:
                valid_w = valid_w & (source_weight <= weight_thresh)
            source_weight = torch.where(
                valid_w,
                source_weight.reciprocal(),
                torch.zeros_like(source_weight),
            )
        else:
            if weight_thresh is not None:
                valid_w = valid_w & (source_weight >= weight_thresh)
            source_weight = torch.where(
                valid_w,
                source_weight,
                torch.zeros_like(source_weight),
            )
        weight_map = weight_map * source_weight

    if weight_type in {"exptime", "expsq"}:
        exptime_val = getattr(obs, "exposure_time", None)
        if exptime_val is None:
            hdr = getattr(obs, "header", None)
            kw = getattr(obs, "exposure_time_keyword", None) or "EXPTIME"
            if isinstance(hdr, dict) and kw in hdr:
                exptime_val = float(hdr[kw])
        if exptime_val is not None:
            exptime_f = float(exptime_val)
            if not math.isfinite(exptime_f) or exptime_f <= 0.0:
                raise ValueError("exposure_time must be finite and positive")
            factor = exptime_f * exptime_f if weight_type == "expsq" else exptime_f
            weight_map = weight_map * float(factor)
    elif weight_type == "background":
        finite_vals = image[torch.isfinite(image)].to(dtype=torch.float32)
        if finite_vals.numel() >= 2:
            med = torch.median(finite_vals)
            mad = torch.median(torch.abs(finite_vals - med))
            sigma = float(mad.item()) * 1.482602218505602
            if math.isfinite(sigma) and sigma > 1e-12:
                weight_map = weight_map * float(1.0 / (sigma * sigma))

    mask = _load_or_validate_aux_plane(
        getattr(obs, "mask", None),
        getattr(obs, "mask_path", None),
        getattr(obs, "mask_hdu", 0),
        image_shape=image_shape,
        device=device,
        label="Mask",
        preserve_integer=True,
    )
    if mask is not None:
        good_bits = getattr(obs, "good_bits", None)
        if good_bits is not None:
            dq = mask.to(device=device, dtype=torch.int64)
            keep = (dq & (~int(good_bits))) == 0
            weight_map = weight_map * keep.to(dtype=torch.float32)
        elif mask.dtype == torch.bool:
            weight_map = weight_map * mask.to(device=device, dtype=torch.float32)
        else:
            mask_f = mask.to(device=device, dtype=torch.float32)
            mask_f = torch.where(
                torch.isfinite(mask_f) & (mask_f > 0.0),
                torch.ones_like(mask_f),
                torch.zeros_like(mask_f),
            )
            weight_map = weight_map * mask_f

    pb = _load_or_validate_aux_plane(
        getattr(obs, "primary_beam", None),
        getattr(obs, "primary_beam_path", None),
        getattr(obs, "primary_beam_hdu", None),
        image_shape=image_shape,
        device=device,
        label="primary_beam map",
    )
    if pb is not None:
        pb = torch.where(torch.isfinite(pb) & (pb > 0.0), pb, torch.zeros_like(pb))
        weight_map = weight_map * pb

    scalar_weight = float(getattr(obs, "weight", 1.0))
    if not math.isfinite(scalar_weight):
        raise ValueError("observation weight must be finite")
    scalar = torch.tensor(scalar_weight, dtype=weight_map.dtype, device=device)
    return weight_map * sanitize_inverse_variance(scalar)
