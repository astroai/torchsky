"""Sky matching, outlier detection, and background rectification for map products."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from .observation import (
    MapResult,
    Observation,
    OutlierConfig,
    SkyConfig,
    StackConfig,
    Target,
)
from .sky import CrossExposureSkyCorrector, MeshSkyEstimator
from .quality import DQPolicy, DQRejectionLog
from .operators.mask_weight import load_image_observation_dq
from .calibration import (
    _apply_observation_calibration,
    _effective_observation_photometric_scale,
    _load_observation_weight_map,
)
from .image_plane_common import (
    _coerce_target_header,
    _convert_world_between_wcs,
    _resolve_map_device,
    load_observation_image_and_header,
)
from .planning import plan
from .windowing import _build_target_world_grid
from ..wcs import WCS


def _load_existing_observation_valid_mask(
    obs: Observation,
    image_shape: tuple[int, int],
    device: torch.device,
) -> Tensor | None:
    from .operators.mask_weight import _load_or_validate_aux_plane

    mask = _load_or_validate_aux_plane(
        obs.mask,
        obs.mask_path,
        obs.mask_hdu,
        image_shape=image_shape,
        device=device,
        label="Mask",
        preserve_integer=True,
    )
    if mask is None:
        return None
    if obs.good_bits is not None:
        dq = mask.to(device=device, dtype=torch.int64)
        return (dq & (~int(obs.good_bits))) == 0
    if mask.dtype == torch.bool:
        return mask.to(device=device, dtype=torch.bool)
    return mask.to(device=device, dtype=torch.float32) > 0.0


def detect_outliers(
    observations: Sequence[Observation],
    coadd_result: MapResult | None = None,
    *,
    config: OutlierConfig | None = None,
    snr_primary: float | None = None,
    snr_neighbor: float | None = None,
    scale_primary: float | None = None,
    scale_neighbor: float | None = None,
    min_floor: float | None = None,
    grow: int | None = None,
    kernel: Literal[
        "bilinear", "bicubic", "lanczos2", "lanczos3", "lanczos4", "nearest"
    ]
    | None = None,
    update_masks: bool = True,
    return_logs: bool = False,
    device: torch.device | str | None = None,
) -> (
    tuple[list[Tensor], list[Observation]]
    | tuple[list[Tensor], list[Observation], tuple[dict[str, int], ...]]
):
    """Detect cosmic rays and transient outliers via AstroDrizzle / JWST blot + spatial derivative test.

    Returns ``(outlier_masks, updated_observations)`` where ``outlier_masks[i]`` is a boolean
    tensor with ``True`` at detected cosmic-ray/outlier pixels in exposure ``i``.
    """
    if not observations:
        raise ValueError("At least one Observation is required")
    cfg = config or OutlierConfig()
    s_prim = float(snr_primary if snr_primary is not None else cfg.snr[0])
    s_neigh = float(snr_neighbor if snr_neighbor is not None else cfg.snr[1])
    c_prim = float(scale_primary if scale_primary is not None else cfg.scale[0])
    c_neigh = float(scale_neighbor if scale_neighbor is not None else cfg.scale[1])
    floor_val = float(min_floor if min_floor is not None else cfg.min_floor)
    grow_px = int(grow if grow is not None else cfg.grow)
    blot_kernel = kernel if kernel is not None else cfg.blot_kernel
    should_update = bool(update_masks and cfg.mode != "flag_only")
    if cfg.parity_mode and cfg.mode == "flag_only":
        raise ValueError("certified parity mode requires a final usability mask")
    if grow_px < 0:
        raise ValueError("outlier grow radius must be non-negative")

    from .image_plane_coadd import _blot_spatial_derivative, blot, coadd_image_plane

    map_device = _resolve_map_device(device, observations, coadd_result)
    if coadd_result is None:
        coadd_result = coadd_image_plane(
            observations,
            stacking_method="median" if len(observations) >= 3 else "weighted_mean",
            resampling_kernel="bilinear",
            value_semantics="surface_brightness",
            device=map_device,
        )

    blotted_list = blot(
        coadd_result,
        list(observations),
        kernel=blot_kernel,
        semantics="surface_brightness",
        device=map_device,
    )
    assert isinstance(blotted_list, list)

    outlier_masks: list[Tensor] = []
    updated_observations: list[Observation] = []
    rejection_logs: list[DQRejectionLog] = []
    read_dev = str(map_device)
    for observation_index, (obs, blotted) in enumerate(
        zip(observations, blotted_list, strict=True)
    ):
        raw_img, hdr = load_observation_image_and_header(obs, device=read_dev)
        raw_img_dev = raw_img.to(device=map_device, dtype=torch.float32)
        src_w = _load_observation_weight_map(obs, raw_img_dev, device=map_device)
        cal_img, cal_w = _apply_observation_calibration(obs, raw_img_dev, src_w)
        s_eff = _effective_observation_photometric_scale(obs, hdr)
        grad = _blot_spatial_derivative(blotted)
        diff = torch.abs(cal_img - blotted)

        # Estimate per-pixel 1-sigma noise in calibrated units from variance planes, weight map, or robust MAD
        noise_sigma: Tensor | None = None
        if (
            obs.var_poisson is not None
            or obs.var_rnoise is not None
            or obs.var_flat is not None
        ):
            var_tot = torch.zeros_like(cal_img)
            if isinstance(obs.var_poisson, Tensor):
                var_tot = var_tot + obs.var_poisson.to(
                    device=map_device, dtype=torch.float32
                )
            if isinstance(obs.var_rnoise, Tensor):
                var_tot = var_tot + obs.var_rnoise.to(
                    device=map_device, dtype=torch.float32
                )
            if isinstance(obs.var_flat, Tensor):
                var_tot = var_tot + obs.var_flat.to(
                    device=map_device, dtype=torch.float32
                )
            valid_variance = torch.isfinite(var_tot) & (var_tot > 0.0)
            noise_sigma = torch.where(
                valid_variance,
                float(s_eff) * torch.sqrt(torch.clamp(var_tot, min=1e-12)),
                torch.full_like(var_tot, float("inf")),
            )
        elif (
            (obs.weight_map is not None or obs.weight_path is not None)
            and obs.weight_type in ("ivm", "rms", "variance")
        ) or obs.weight_is_inverse_variance:
            noise_sigma = torch.where(
                cal_w > 0.0,
                torch.rsqrt(torch.clamp(cal_w, min=1e-12)),
                torch.full_like(cal_img, float("inf")),
            )
        if noise_sigma is None and cfg.parity_mode:
            raise ValueError(
                "certified outlier parity requires declared detector variance"
            )
        if noise_sigma is None:
            finite_diff = diff[torch.isfinite(diff)]
            if finite_diff.numel() > 0:
                mad_sigma = torch.clamp(1.4826 * torch.median(finite_diff), min=1e-6)
            else:
                mad_sigma = torch.tensor(1.0, device=map_device, dtype=torch.float32)
            noise_sigma = torch.clamp(
                torch.sqrt(torch.clamp(torch.abs(blotted), min=0.0) + mad_sigma**2),
                min=float(mad_sigma.item()),
            )

        thresh_primary = c_prim * grad + s_prim * noise_sigma + floor_val
        thresh_neighbor = c_neigh * grad + s_neigh * noise_sigma + floor_val
        primary_cr = (diff > thresh_primary) & torch.isfinite(cal_img)
        if grow_px > 0 and bool(primary_cr.any()):
            k_size = 2 * grow_px + 1
            dilated = (
                F.max_pool2d(
                    primary_cr.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0),
                    kernel_size=k_size,
                    stride=1,
                    padding=grow_px,
                )
                .squeeze(0)
                .squeeze(0)
                > 0.0
            )
            cr_mask = primary_cr | (
                dilated & (diff > thresh_neighbor) & torch.isfinite(cal_img)
            )
        else:
            cr_mask = primary_cr
        outlier_masks.append(cr_mask)
        rejection_logs.append(
            DQRejectionLog(
                observation_index=observation_index,
                primary_count=int(primary_cr.sum().item()),
                grown_count=int((cr_mask & ~primary_cr).sum().item()),
                rejected_count=int(cr_mask.sum().item()),
                cosmic_ray_bit=int(
                    (cfg.dq_policy or obs.dq_policy or DQPolicy()).cosmic_ray_bit
                ),
            )
        )
        if should_update:
            obs_shape = (int(raw_img_dev.shape[0]), int(raw_img_dev.shape[1]))
            prev_valid = _load_existing_observation_valid_mask(
                obs, obs_shape, map_device
            )
            valid_mask = (~cr_mask) if prev_valid is None else (prev_valid & (~cr_mask))
            new_wmap = obs.weight_map
            if cfg.mode == "weight_zero" and isinstance(obs.weight_map, Tensor):
                new_wmap = torch.where(
                    ~cr_mask,
                    obs.weight_map.to(device=map_device),
                    torch.zeros_like(obs.weight_map.to(device=map_device)),
                )
            if cfg.parity_mode:
                policy = cfg.dq_policy or obs.dq_policy or DQPolicy()
                existing_dq = load_image_observation_dq(
                    obs,
                    raw_img_dev,
                    device=map_device,
                )
                if existing_dq is None:
                    existing_dq = torch.zeros_like(raw_img_dev, dtype=torch.int64)
                updated_dq = policy.with_cosmic_rays(existing_dq, cr_mask)
                valid_mask = valid_mask & policy.usability_mask(updated_dq)
                updated_observations.append(
                    dataclasses.replace(
                        obs,
                        mask=valid_mask,
                        good_bits=None,
                        weight_map=new_wmap,
                        dq=updated_dq,
                        dq_path=None,
                        dq_policy=policy,
                    )
                )
            else:
                updated_observations.append(
                    dataclasses.replace(
                        obs,
                        mask=valid_mask,
                        good_bits=None,
                        weight_map=new_wmap,
                    )
                )
        else:
            updated_observations.append(obs)
    logs = tuple(log.to_dict() for log in rejection_logs)
    if return_logs:
        return outlier_masks, updated_observations, logs
    return outlier_masks, updated_observations


def _robust_scalar_sky(
    image: Tensor,
    bad_mask: Tensor | None,
    *,
    clip_sigma: float = 3.0,
    clip_iterations: int = 3,
) -> float:
    valid = torch.isfinite(image)
    if bad_mask is not None:
        valid = valid & (~bad_mask.to(device=image.device, dtype=torch.bool))
    vals = image[valid].to(dtype=torch.float32)
    if vals.numel() == 0:
        return 0.0
    for _ in range(max(int(clip_iterations), 1)):
        if vals.numel() < 3:
            break
        med = torch.median(vals)
        mad = torch.median(torch.abs(vals - med))
        sigma = 1.482602218505602 * mad
        if float(sigma.item()) <= 1e-12:
            break
        keep = torch.abs(vals - med) <= float(clip_sigma) * sigma
        if not bool(keep.any()) or int(keep.sum().item()) == int(vals.numel()):
            break
        vals = vals[keep]
    return float(torch.median(vals).item())


def match_sky(
    observations: Sequence[Observation],
    *,
    target: Target | Mapping[str, Any] | WCS | None = None,
    config: SkyConfig | None = None,
    mode: Literal[
        "constant",
        "planar",
        "mesh",
        "per_exposure",
        "local",
        "global",
        "match",
        "global+match",
        "overlap_constant",
        "overlap_planar",
        "overlap_mesh",
        "none",
    ]
    | None = None,
    mesh_size: int | None = None,
    regularization: float | None = None,
    apply: bool | None = None,
    device: torch.device | str | None = None,
) -> tuple[list[Observation], dict[str, Any]]:
    """Solve and optionally subtract sky background across exposures.

    Supports single-exposure mesh background estimation (``"per_exposure"`` / ``"local"``),
    JWST/Roman ``stcal.skymatch`` global modes (``"global"``, ``"match"``, ``"global+match"``),
    and global overlap-graph rectification (``"constant"``, ``"planar"``, ``"mesh"``).
    """
    if not observations:
        raise ValueError("At least one Observation is required")
    cfg = config or SkyConfig(mode="overlap_constant")
    raw_mode = mode if mode is not None else cfg.mode
    if raw_mode == "none":
        return list(observations), {"mode": "none"}
    should_apply = bool(cfg.subtract if apply is None else apply)
    resolved_mesh = int(mesh_size if mesh_size is not None else cfg.mesh_size)
    resolved_reg = float(
        regularization if regularization is not None else cfg.regularization
    )
    from .image_plane_coadd import blot, resample

    map_device = _resolve_map_device(device, observations)
    read_dev = str(map_device)

    if raw_mode in ("per_exposure", "local"):
        estimator = MeshSkyEstimator(
            block_size=resolved_mesh,
            statistic=cfg.stat,
            mesh_filter_size=cfg.smooth_scale,
            self_mask_sigma=cfg.clip_sigma,
        )
        updated: list[Observation] = []
        backgrounds: list[Tensor] = []
        for obs in observations:
            img, hdr = load_observation_image_and_header(obs, device=read_dev)
            img_dev = img.to(device=map_device, dtype=torch.float32)
            obs_shape = (int(img_dev.shape[0]), int(img_dev.shape[1]))
            bad_mask = ~torch.isfinite(img_dev)
            prev_valid = _load_existing_observation_valid_mask(
                obs, obs_shape, map_device
            )
            if prev_valid is not None:
                bad_mask = bad_mask | (~prev_valid)
            bg = estimator(img_dev, bad_mask if bool(bad_mask.any()) else None)
            backgrounds.append(bg)
            if should_apply:
                updated.append(dataclasses.replace(obs, data=img_dev - bg, header=hdr))
            else:
                updated.append(obs)
        return updated, {"mode": raw_mode, "backgrounds": backgrounds}

    if raw_mode == "global":
        sky_levels: list[float] = []
        scales: list[float] = []
        for obs in observations:
            img, hdr = load_observation_image_and_header(obs, device=read_dev)
            img_dev = img.to(device=map_device, dtype=torch.float32)
            src_w = _load_observation_weight_map(obs, img_dev, device=map_device)
            cal_img, cal_w = _apply_observation_calibration(obs, img_dev, src_w)
            s_eff = _effective_observation_photometric_scale(obs, hdr)
            scales.append(s_eff)
            sky_levels.append(
                _robust_scalar_sky(
                    cal_img,
                    ~(cal_w > 0.0),
                    clip_sigma=cfg.clip_sigma,
                    clip_iterations=cfg.clip_iterations,
                )
            )
        global_sky = min(sky_levels) if sky_levels else 0.0
        offsets_t = torch.full(
            (len(observations),),
            float(global_sky),
            dtype=torch.float32,
            device=map_device,
        )
        updated_global: list[Observation] = []
        for obs, s_eff in zip(observations, scales):
            if should_apply:
                updated_global.append(
                    dataclasses.replace(
                        obs,
                        background_offset=float(obs.background_offset)
                        + float(global_sky) / float(s_eff),
                    )
                )
            else:
                updated_global.append(obs)
        return updated_global, {
            "mode": "global",
            "global_sky": float(global_sky),
            "per_exposure_sky": sky_levels,
            "offsets": offsets_t,
        }

    # Global overlap-graph sky matching ("constant"/"match"/"global+match", "planar", or "mesh")
    corrector_mode: Literal["constant", "planar", "mesh"]
    if raw_mode in ("constant", "overlap_constant", "match", "global+match"):
        corrector_mode = "constant"
    elif raw_mode in ("planar", "overlap_planar"):
        corrector_mode = "planar"
    elif raw_mode in ("mesh", "overlap_mesh"):
        corrector_mode = "mesh"
    else:
        raise ValueError(f"Unsupported sky matching mode: {raw_mode!r}")

    target_hdr = _coerce_target_header(target)
    if target_hdr is None:
        target_hdr = dict(plan(observations, device=map_device).header)
    resampled_list = resample(
        observations,
        target_hdr,
        kernel="bilinear",
        semantics="surface_brightness",
        device=map_device,
    )
    assert isinstance(resampled_list, list)
    images = [r.data for r in resampled_list]
    masks = [~(r.weight > 0.0) for r in resampled_list]
    corrector = CrossExposureSkyCorrector(
        mesh_size=resolved_mesh,
        regularization=resolved_reg,
        mode=corrector_mode,
    )

    if corrector_mode == "constant":
        offsets = corrector.fit_offsets(images, masks)
        if raw_mode == "global+match":
            equalized_skies = [
                _robust_scalar_sky(
                    img - off,
                    msk,
                    clip_sigma=cfg.clip_sigma,
                    clip_iterations=cfg.clip_iterations,
                )
                for img, msk, off in zip(images, masks, offsets)
            ]
            global_base = min(equalized_skies) if equalized_skies else 0.0
            offsets = offsets + float(global_base)
        updated_obs: list[Observation] = []
        for obs, off in zip(observations, offsets.tolist()):
            if should_apply:
                s_eff = _effective_observation_photometric_scale(obs)
                updated_obs.append(
                    dataclasses.replace(
                        obs,
                        background_offset=float(obs.background_offset)
                        + float(off) / float(s_eff),
                    )
                )
            else:
                updated_obs.append(obs)
        return updated_obs, {
            "mode": "global+match" if raw_mode == "global+match" else "constant",
            "offsets": offsets,
        }

    if corrector_mode == "planar":
        planes = corrector.fit_planes(images, masks)
        coadd_wcs = WCS(target_hdr)
        coadd_shape = (int(images[0].shape[0]), int(images[0].shape[1]))
        H_c, W_c = coadd_shape
        updated_obs = []
        for idx, obs in enumerate(observations):
            if should_apply:
                img, hdr = load_observation_image_and_header(obs, device=read_dev)
                img_dev = img.to(device=map_device, dtype=torch.float32)
                s_eff = _effective_observation_photometric_scale(obs, hdr)
                obs_wcs = WCS(hdr)
                obs_shape = (int(img_dev.shape[0]), int(img_dev.shape[1]))
                ra_grid, dec_grid = _build_target_world_grid(
                    obs_wcs, obs_shape[0], obs_shape[1], device=map_device
                )
                flat_ra = ra_grid.reshape(-1)
                flat_dec = dec_grid.reshape(-1)
                flat_ra, flat_dec = _convert_world_between_wcs(
                    flat_ra, flat_dec, obs_wcs, coadd_wcs
                )
                cx_flat, cy_flat = coadd_wcs.world_to_pixel(flat_ra, flat_dec)
                cx = cx_flat.reshape(obs_shape)
                cy = cy_flat.reshape(obs_shape)
                u = (cx / max(W_c - 1, 1)) * 2.0 - 1.0
                v = (cy / max(H_c - 1, 1)) * 2.0 - 1.0
                a_i = planes[idx, 0]
                bx_i = planes[idx, 1]
                by_i = planes[idx, 2]
                plane_obs = (a_i + bx_i * u + by_i * v) / float(s_eff)
                updated_obs.append(
                    dataclasses.replace(obs, data=img_dev - plane_obs, header=hdr)
                )
            else:
                updated_obs.append(obs)
        return updated_obs, {"mode": "planar", "planes": planes}

    corrections = corrector.fit_corrections(images, masks)
    updated_obs = []
    for obs, corr in zip(observations, corrections):
        if should_apply:
            img, hdr = load_observation_image_and_header(obs, device=read_dev)
            img_dev = img.to(device=map_device, dtype=torch.float32)
            s_eff = _effective_observation_photometric_scale(obs, hdr)
            corr_obs = blot(
                corr,
                target_hdr,
                target=obs,
                kernel="bilinear",
                semantics="surface_brightness",
                device=map_device,
            )
            assert isinstance(corr_obs, Tensor)
            updated_obs.append(
                dataclasses.replace(
                    obs, data=img_dev - corr_obs / float(s_eff), header=hdr
                )
            )
        else:
            updated_obs.append(obs)
    return updated_obs, {"mode": "mesh", "corrections": corrections}


def _resolve_tile_hw(
    tile_size: int | tuple[int, int] | None,
    spatial_tile_hw: tuple[int, int] | None,
    s_cfg: StackConfig,
) -> tuple[int, int] | None:
    raw = (
        tile_size
        if tile_size is not None
        else (
            spatial_tile_hw
            if spatial_tile_hw is not None
            else (s_cfg.tile_size if s_cfg.tile_size is not None else s_cfg.tile_hw)
        )
    )
    if raw is None:
        return None
    if isinstance(raw, int):
        return (int(raw), int(raw))
    return (int(raw[0]), int(raw[1]))
