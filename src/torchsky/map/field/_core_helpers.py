"""Shared helpers and the base class for :mod:`torchsky.map.field` models.

Extracted from ``map/field/core.py`` so the model zoo and the helper layer can
grow independently; ``core`` re-exports every name here for compatibility.
"""

from __future__ import annotations

import copy
import math
import time
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ...device import _as_tensor, resolve_device, _to
from ...wcs import WCS

if TYPE_CHECKING:
    pass


from .bandpass import (
    bandpass_bounds as _bandpass_bounds,
)
from .noise import _synthetic_noise_target
from .observation_ops import (
    observation_pixel_to_world as _observation_pixel_to_world,
    wrapped_delta_ra_deg as _wrapped_delta_ra_deg,
)
from .types import (
    FieldComponentAnchorTerm,
    FieldCentroidAnchorTerm,
    FieldCoordinateMode,
    FieldDomain,
    FieldNoisePretrainKind,
    FieldObservation,
    ForwardContext,
)


_STRAINER_TRUNK_FORMAT = "torchsky.strainer_trunk.v1"


def _point_bandpass_features(lam: Tensor, num_frequencies: int) -> Tensor:
    zeros = torch.zeros_like(lam)
    feats = [lam, zeros, zeros, zeros]
    for scale_idx in range(1, max(int(num_frequencies), 0) + 1):
        phase = (math.pi * float(scale_idx)) * lam
        feats.append(torch.sin(phase))
        feats.append(torch.cos(phase))
    return torch.stack(feats, dim=-1)


def _smoothness_probe_grid(
    domain: FieldDomain,
    *,
    grid_size: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    if grid_size < 2:
        raise ValueError("spatial_smoothness_grid_size must be at least 2")
    offsets = torch.linspace(
        -1.0, 1.0, steps=int(grid_size), device=device, dtype=torch.float32
    )
    off_y, off_x = torch.meshgrid(offsets, offsets, indexing="ij")
    cos_dec = max(math.cos(math.radians(float(domain.center_dec_deg))), 1.0e-6)
    dra = off_x * float(domain.spatial_scale_deg) / cos_dec
    ddec = off_y * float(domain.spatial_scale_deg)
    ra = torch.remainder(float(domain.center_ra_deg) + dra + 360.0, 360.0)
    dec = torch.clamp(float(domain.center_dec_deg) + ddec, min=-90.0, max=90.0)
    return ra, dec


def _observation_footprint_world_deg(
    observation: FieldObservation,
) -> tuple[list[float], list[float]]:
    """Return detector-edge sky samples, falling back to the reference position."""
    nx = int(observation.header.get("NAXIS1", observation.image.shape[-1]))
    ny = int(observation.header.get("NAXIS2", observation.image.shape[-2]))
    x = torch.tensor([-0.5, nx - 0.5, nx - 0.5, -0.5], dtype=torch.float64)
    y = torch.tensor([-0.5, -0.5, ny - 0.5, ny - 0.5], dtype=torch.float64)
    try:
        ra, dec = _observation_pixel_to_world(observation, x, y)
        ra_values = torch.as_tensor(ra, dtype=torch.float64)
        dec_values = torch.as_tensor(dec, dtype=torch.float64)
        finite = torch.isfinite(ra_values) & torch.isfinite(dec_values)
        if bool(torch.all(finite)):
            return ra_values.tolist(), dec_values.tolist()
    except (RuntimeError, TypeError, ValueError):
        pass
    return (
        [float(observation.header.get("CRVAL1", 0.0))],
        [float(observation.header.get("CRVAL2", 0.0))],
    )


def estimate_field_domain(
    observations: Sequence[FieldObservation],
    *,
    coord_mode: FieldCoordinateMode = "local_tangent",
) -> FieldDomain:
    """Estimate a normalized spatial-spectral domain covering observations."""
    if not observations:
        raise ValueError("estimate_field_domain requires at least one observation")
    wavelength_lowers: list[float] = []
    wavelength_uppers: list[float] = []
    footprint_ra: list[float] = []
    footprint_dec: list[float] = []
    for observation in observations:
        lower, upper = _bandpass_bounds(observation)
        wavelength_lowers.append(lower)
        wavelength_uppers.append(upper)
        ra, dec = _observation_footprint_world_deg(observation)
        footprint_ra.extend(ra)
        footprint_dec.extend(dec)
    ra_rad = torch.deg2rad(torch.tensor(footprint_ra, dtype=torch.float64))
    dec_rad = torch.deg2rad(torch.tensor(footprint_dec, dtype=torch.float64))
    cos_dec = torch.cos(dec_rad)
    mean_x = float(torch.mean(cos_dec * torch.cos(ra_rad)))
    mean_y = float(torch.mean(cos_dec * torch.sin(ra_rad)))
    mean_z = float(torch.mean(torch.sin(dec_rad)))
    center_ra = math.degrees(math.atan2(mean_y, mean_x)) % 360.0
    center_dec = math.degrees(math.atan2(mean_z, math.hypot(mean_x, mean_y)))
    tangent_cos_dec = max(math.cos(math.radians(center_dec)), 1.0e-6)
    x_offsets = [
        ((ra - center_ra + 180.0) % 360.0 - 180.0) * tangent_cos_dec
        for ra in footprint_ra
    ]
    y_offsets = [dec - center_dec for dec in footprint_dec]
    # Preserve the historical normalization where one detector width maps to
    # roughly one coordinate unit while covering the full footprint union.
    spatial_scale = 2.0 * max(
        max(abs(value) for value in x_offsets),
        max(abs(value) for value in y_offsets),
        1.0e-4,
    )
    lower = min(wavelength_lowers)
    upper = max(wavelength_uppers)
    wavelength_center = 0.5 * (lower + upper)
    wavelength_scale = max(0.5 * (upper - lower), 1.0)
    return FieldDomain(
        center_ra_deg=center_ra,
        center_dec_deg=center_dec,
        spatial_scale_deg=spatial_scale,
        wavelength_center=wavelength_center,
        wavelength_scale=wavelength_scale,
        coord_mode=coord_mode,
    )


def _fourier_features_with_curriculum(
    spatial: Tensor, num_frequencies: int, active: int
) -> Tensor:
    """Spatial Fourier features with a curriculum over active frequencies."""
    if num_frequencies <= 0:
        return spatial
    scales = torch.arange(
        1, num_frequencies + 1, device=spatial.device, dtype=spatial.dtype
    )
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


def _fourier_features_vectorized(spatial: Tensor, num_frequencies: int) -> Tensor:
    """Vectorized spatial Fourier features (no curriculum)."""
    if num_frequencies <= 0:
        return spatial
    scales = torch.arange(
        1, num_frequencies + 1, device=spatial.device, dtype=spatial.dtype
    )
    phase = math.pi * spatial.unsqueeze(-1) * scales
    sin_feats = torch.sin(phase).transpose(-1, -2)
    cos_feats = torch.cos(phase).transpose(-1, -2)
    stacked = torch.stack([sin_feats, cos_feats], dim=-2)
    flattened = stacked.reshape(*spatial.shape[:-1], -1)
    return torch.cat([spatial, flattened], dim=-1)


def _atlas_expert_weights(
    xy: Tensor,
    centers: Tensor,
    sigma: float,
    topk_experts: int | None,
) -> Tensor:
    """Gaussian softmax weights over atlas experts, optionally top-k sparsified."""
    diff = xy.unsqueeze(-2) - _to(centers, device=xy.device, dtype=xy.dtype)
    logits = -0.5 * torch.sum(torch.square(diff / max(sigma, 1.0e-3)), dim=-1)
    weights = torch.softmax(logits, dim=-1)
    if topk_experts is not None and topk_experts < weights.shape[-1]:
        topk_vals, topk_idx = torch.topk(weights, k=topk_experts, dim=-1)
        sparse = torch.zeros_like(weights)
        sparse.scatter_(-1, topk_idx, topk_vals)
        weights = sparse / torch.clamp(sparse.sum(dim=-1, keepdim=True), min=1.0e-8)
    return weights


def _reset_xavier_mlp(
    hidden: "nn.Sequential | list[nn.Module] | nn.ModuleList", output: nn.Linear
) -> None:
    """Xavier-init every Linear in ``hidden`` and a small final linear readout."""
    for module in hidden:
        if type(module) is nn.Linear:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    nn.init.uniform_(output.weight, -1e-2, 1e-2)
    nn.init.constant_(output.bias, -3.0)


def _atlas_local_coords(xy: Tensor, centers: Tensor, atlas_side: int) -> Tensor:
    """Tile-local coordinates for every atlas expert, clamped to ``[0, 1]``."""
    local = []
    tile_size = 1.0 / float(atlas_side)
    for center in centers:
        lower = center - 0.5 * tile_size
        shifted = (xy - _to(lower, device=xy.device, dtype=xy.dtype)) / tile_size
        local.append(torch.clamp(shifted, 0.0, 1.0))
    return torch.stack(local, dim=-2)


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
        wavelength = _as_tensor(
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
        wavelength = _as_tensor(
            wavelength_angstrom,
            device=device,
            dtype=dtype,
        )
        if wavelength.shape != shape:
            wavelength = torch.broadcast_to(wavelength, shape)
        return (wavelength - float(self.domain.wavelength_center)) / float(
            self.domain.wavelength_scale
        )


def _observation_without_background(observation: FieldObservation) -> FieldObservation:
    if observation.additive_components:
        raise NotImplementedError(
            "component anchors do not yet support typed additive components"
        )
    stripped = copy.deepcopy(observation)
    stripped.sky_background = None
    stripped.instrument_background = None
    stripped.detector_background = None
    stripped.background_offset = 0.0
    return stripped


def _resolve_anchor_sample_world_coordinates(
    term: FieldComponentAnchorTerm,
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    if term.sample_ra_deg is not None and term.sample_dec_deg is not None:
        return (
            _to(term.sample_ra_deg, device=device, dtype=torch.float32),
            _to(term.sample_dec_deg, device=device, dtype=torch.float32),
        )
    if term.sample_x_pix is None or term.sample_y_pix is None:
        raise ValueError("anchor term does not define sample coordinates")
    x_pix = _to(term.sample_x_pix, device=device, dtype=torch.float64)
    y_pix = _to(term.sample_y_pix, device=device, dtype=torch.float64)
    ra_deg, dec_deg = _observation_pixel_to_world(term.observation, x_pix, y_pix)
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
    if term.observation.detector_to_sky is not None:
        raise ValueError(
            "sky-coordinate centroid anchors require an inverse detector transform; "
            "provide target_x_pix and target_y_pix for forward-only native astrometry"
        )
    assert term.target_ra_deg is not None and term.target_dec_deg is not None
    x_pix, y_pix = WCS(term.observation.header).world_to_pixel(
        torch.tensor(float(term.target_ra_deg), dtype=torch.float64),
        torch.tensor(float(term.target_dec_deg), dtype=torch.float64),
    )
    return (
        _to(x_pix, device=device, dtype=dtype),
        _to(y_pix, device=device, dtype=dtype),
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


def _sample_feature_volume(volume: Tensor, coords_unit: Tensor) -> Tensor:
    flat = _to(coords_unit.reshape(-1, 3), dtype=volume.dtype, device=volume.device)
    grid = torch.stack(
        [
            2.0 * flat[:, 0] - 1.0,
            2.0 * flat[:, 1] - 1.0,
            2.0 * flat[:, 2] - 1.0,
        ],
        dim=-1,
    ).view(1, -1, 1, 1, 3)
    sampled = F.grid_sample(
        volume.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.squeeze(0).squeeze(0).reshape(*coords_unit.shape[:-1])


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
    trunk_fn = cast(
        Callable[[], Mapping[str, Tensor]], trunk_fn
    )  # duck-typed trunk API

    payload: dict[str, object] = {
        "format": _STRAINER_TRUNK_FORMAT,
        "trunk": {k: v.detach().cpu() for k, v in trunk_fn().items()},
        "meta": dict(meta) if meta else {},
    }
    torch.save(payload, path)


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
) -> dict[str, Any] | None:
    if not isinstance(model, _BaseFieldModel):
        raise TypeError("model must be a torchsky map field _BaseFieldModel subclass")
    steps = max(int(steps), 0)
    if steps == 0:
        return None
    kinds: tuple[FieldNoisePretrainKind, ...]
    if noise_kinds is not None:
        kinds = tuple(k for k in noise_kinds if k != "none")
        if not kinds:
            raise ValueError(
                "noise_kinds must include at least one kind other than 'none'"
            )
    else:
        if noise_kind == "none":
            raise ValueError(
                "noise_kind must not be 'none' for synthetic trunk pretraining"
            )
        kinds = (noise_kind,)
    domain = model.domain
    device = resolve_device(None, model)
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

    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    for step_idx in range(steps):
        step_noise = kinds[step_idx % num_kinds]
        optimizer.zero_grad(set_to_none=True)
        dra = (
            torch.rand(
                batch, generator=training_gen, dtype=torch.float64, device=device
            )
            - 0.5
        ) * (2.0 * float(domain.spatial_scale_deg))
        ddec = (
            torch.rand(
                batch, generator=training_gen, dtype=torch.float64, device=device
            )
            - 0.5
        ) * (2.0 * float(domain.spatial_scale_deg))
        ra_deg = torch.remainder(float(domain.center_ra_deg) + dra, 360.0)
        dec_deg = torch.clamp(
            float(domain.center_dec_deg) + ddec,
            min=-90.0,
            max=90.0,
        )
        lam_offset = (
            torch.rand(
                batch, generator=training_gen, dtype=torch.float64, device=device
            )
            - 0.5
        ) * (2.0 * float(domain.wavelength_scale))
        wavelength = (
            torch.as_tensor(float(domain.wavelength_center), device=device) + lam_offset
        )
        coords_unit = model._unit_box_coords(ra_deg, dec_deg, wavelength)
        target = _to(
            _synthetic_noise_target(
                coords_unit,
                noise_kind=step_noise,
                num_terms=num_terms,
                max_frequency=max_frequency,
                generator=noise_gen,
            ),
            device=device,
            dtype=torch.float32,
        )
        pred = model(
            ra_deg.to(dtype=torch.float32),
            dec_deg.to(dtype=torch.float32),
            wavelength.to(dtype=torch.float32),
        )
        # ⚡ Bolt: Element-wise multiplication is faster than torch.square or **2
        diff_pt = pred - target
        loss = torch.mean(diff_pt * diff_pt)
        loss.backward()
        optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    duration = end_time - start_time

    max_mem = 0
    if device.type == "cuda":
        max_mem = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

    step_duration_mean_ms = (duration / steps * 1000.0) if steps > 0 else 0.0
    throughput = (steps * batch) / duration if duration > 0 else 0.0

    return {
        "total_duration_sec": duration,
        "step_duration_mean_ms": step_duration_mean_ms,
        "throughput_samples_per_sec": throughput,
        "max_memory_allocated_bytes": max_mem,
    }
