"""Direct-to-HEALPix coaddition and HEALPix-to-image resampling."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import torch
from torch import Tensor

from ..sphere.moc import MOC
from ..wcs import WCS
from .image_plane import (
    _apply_observation_astrometric_shifts,
    _apply_observation_calibration,
    _coerce_target_header,
    _load_observation_weight_map,
    _resolve_map_device,
    detect_outliers,
    load_observation_image_and_header,
    match_sky,
)
from .observation import (
    HealpixResult,
    MapResult,
    Observation,
    OutlierConfig,
    SkyConfig,
    Target,
)
from .operators.image_grid import normalized_header_for_shape, target_shape_from_header
from .operators.resampling import sample_image_plane_batched_channels
from .windowing import _build_target_world_grid


def _sorted_chunk(
    pixels: Tensor,
    numer: Tensor,
    denom: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    ordered = pixels.numel() < 2 or bool((pixels[1:] > pixels[:-1]).all())
    if ordered:
        sorted_pixels = pixels
        sorted_numer = numer
        sorted_denom = denom
    else:
        order = torch.argsort(pixels, stable=True)
        sorted_pixels = pixels[order]
        sorted_numer = numer[order]
        sorted_denom = denom[order]
    if sorted_pixels.numel() > 1 and bool(
        (sorted_pixels[1:] == sorted_pixels[:-1]).any()
    ):
        raise ValueError("a contribution chunk must contain unique HEALPix pixels")
    return (
        sorted_pixels,
        sorted_numer,
        sorted_denom,
        torch.ones_like(sorted_pixels, dtype=torch.int32),
    )


def _sorted_alignment(
    source_pixels: Tensor,
    target_pixels: Tensor,
) -> tuple[Tensor, Tensor]:
    positions = torch.searchsorted(source_pixels, target_pixels)
    safe_positions = positions.clamp_max(source_pixels.numel() - 1)
    present = (positions < source_pixels.numel()) & (
        source_pixels[safe_positions] == target_pixels
    )
    return safe_positions, present


def _aligned_values(
    source_values: Tensor,
    safe_positions: Tensor,
    present: Tensor,
) -> Tensor:
    return torch.where(
        present,
        source_values[safe_positions],
        torch.zeros((), dtype=source_values.dtype, device=source_values.device),
    )


class _SortedContributionAccumulator:
    def __init__(self, *, device: torch.device) -> None:
        self.pixels = torch.empty(0, dtype=torch.int64, device=device)
        self.numer = torch.empty(0, dtype=torch.float32, device=device)
        self.denom = torch.empty(0, dtype=torch.float32, device=device)
        self.overlap = torch.empty(0, dtype=torch.int32, device=device)

    def add(self, pixels: Tensor, numer: Tensor, denom: Tensor) -> None:
        if pixels.numel() == 0:
            return
        chunk_pixels, chunk_numer, chunk_denom, chunk_overlap = _sorted_chunk(
            pixels, numer, denom
        )
        if self.pixels.numel() == 0:
            self.pixels = chunk_pixels
            self.numer = chunk_numer
            self.denom = chunk_denom
            self.overlap = chunk_overlap
            return

        if torch.equal(self.pixels, chunk_pixels):
            self.numer = self.numer + chunk_numer
            self.denom = self.denom + chunk_denom
            self.overlap = self.overlap + chunk_overlap
            return

        merged_pixels = torch.unique(
            torch.cat((self.pixels, chunk_pixels)), sorted=True
        )
        old_positions, old_present = _sorted_alignment(self.pixels, merged_pixels)
        chunk_positions, chunk_present = _sorted_alignment(
            chunk_pixels, merged_pixels
        )
        self.numer = _aligned_values(
            self.numer, old_positions, old_present
        ) + _aligned_values(chunk_numer, chunk_positions, chunk_present)
        self.denom = _aligned_values(
            self.denom, old_positions, old_present
        ) + _aligned_values(chunk_denom, chunk_positions, chunk_present)
        self.overlap = _aligned_values(
            self.overlap, old_positions, old_present
        ) + _aligned_values(chunk_overlap, chunk_positions, chunk_present)
        self.pixels = merged_pixels


def coadd_healpix(
    observations: Sequence[Observation],
    nside: int,
    *,
    nest: bool = False,
    sparse: bool = False,
    nside_coverage: int = 32,
    kernel: Literal[
        "bilinear", "nearest", "bicubic", "lanczos2", "lanczos3", "lanczos4"
    ] = "bilinear",
    sky: SkyConfig | None = None,
    outliers: OutlierConfig | None = None,
    device: torch.device | str | None = None,
    reduction_chunk_size: int = 262_144,
) -> HealpixResult:
    """Coadd 2D WCS observations directly onto a dense or sparse HEALPix grid.

    ``reduction_chunk_size`` bounds sampled contributions retained before each
    incremental reduction. Sparse accumulation retains merged output support;
    dense accumulation writes each chunk directly into the output tensors.
    """
    from ..sphere import healpix as _hp
    from ..sphere.sparse import HealSparseMap

    if not observations:
        raise ValueError("At least one Observation is required")
    if (
        isinstance(reduction_chunk_size, bool)
        or not isinstance(reduction_chunk_size, int)
        or reduction_chunk_size < 1
    ):
        raise ValueError("reduction_chunk_size must be a positive integer")
    nside_i = int(nside)
    npix = _hp.nside2npix(nside_i)
    order = nside_i.bit_length() - 1
    map_device = _resolve_map_device(device)
    read_dev = str(map_device)

    obs_list = _apply_observation_astrometric_shifts(observations, map_device)
    if sky is not None and sky.mode != "none" and sky.subtract:
        obs_list, _ = match_sky(obs_list, config=sky, apply=True, device=map_device)
    if outliers is not None and outliers.enabled:
        _, obs_list = detect_outliers(
            obs_list, config=outliers, update_masks=True, device=map_device
        )

    union_moc = MOC.empty()
    sparse_accumulator = (
        _SortedContributionAccumulator(device=map_device) if sparse else None
    )
    numer_dense = (
        None
        if sparse
        else torch.zeros(npix, dtype=torch.float32, device=map_device)
    )
    denom_dense = (
        None
        if sparse
        else torch.zeros(npix, dtype=torch.float32, device=map_device)
    )
    overlap_dense = (
        None if sparse else torch.zeros(npix, dtype=torch.int32, device=map_device)
    )

    for obs in obs_list:
        if obs.data is not None:
            raw_img = obs.data.to(device=map_device, dtype=torch.float32)
            hdr_obs = dict(obs.header) if obs.header is not None else {}
        else:
            raw_img, hdr_obs = load_observation_image_and_header(obs, device=read_dev)
            raw_img = raw_img.to(device=map_device, dtype=torch.float32)
        src_w = _load_observation_weight_map(obs, raw_img, device=map_device)
        cal_img, cal_w = _apply_observation_calibration(obs, raw_img, src_w)

        obs_wcs = (
            obs.wcs
            if isinstance(obs.wcs, WCS)
            else WCS(obs.wcs if isinstance(obs.wcs, dict) else hdr_obs)
        ).to(map_device)
        H_s, W_s = int(cal_img.shape[-2]), int(cal_img.shape[-1])
        obs_moc = MOC.from_wcs(obs_wcs, shape=(H_s, W_s), max_order=order)
        union_moc = union_moc.union(obs_moc)
        obs_pix = obs_moc.to_pixelized_region(nside_i, nest=nest).pixels.to(
            device=map_device, dtype=torch.int64
        )
        if obs_pix.numel() == 0:
            continue

        valid_src = (cal_w > 0.0) & torch.isfinite(cal_img) & torch.isfinite(cal_w)
        channels = torch.stack(
            [
                torch.where(valid_src, cal_img * cal_w, torch.zeros_like(cal_img)),
                torch.where(valid_src, cal_w, torch.zeros_like(cal_w)),
                valid_src.to(dtype=torch.float32),
            ],
            dim=0,
        )
        for pixel_chunk in obs_pix.split(reduction_chunk_size):
            ra_p, dec_p = _hp.pix2ang(nside_i, pixel_chunk, nest=nest, lonlat=True)
            x_src, y_src = obs_wcs.world_to_pixel(ra_p, dec_p)
            x_2d = x_src.to(dtype=torch.float32).unsqueeze(0)
            y_2d = y_src.to(dtype=torch.float32).unsqueeze(0)
            x_norm = (2.0 * x_2d / max(W_s - 1, 1)) - 1.0
            y_norm = (2.0 * y_2d / max(H_s - 1, 1)) - 1.0
            grid = torch.stack([x_norm, y_norm], dim=-1).to(dtype=torch.float32)
            sampled = sample_image_plane_batched_channels(
                channels,
                x_src=x_2d,
                y_src=y_2d,
                grid=grid,
                kernel=kernel,
            )
            s_num = sampled[0, 0]
            s_den = sampled[1, 0]
            s_sup = sampled[2, 0]
            in_b = (
                torch.isfinite(x_2d[0])
                & torch.isfinite(y_2d[0])
                & (x_2d[0] >= 0.0)
                & (x_2d[0] <= (W_s - 1))
                & (y_2d[0] >= 0.0)
                & (y_2d[0] <= (H_s - 1))
            )
            hit = in_b & (s_sup > 0.0) & (s_den > 0.0) & torch.isfinite(s_num)
            if bool(hit.any()):
                hit_pixels = pixel_chunk[hit]
                if sparse_accumulator is not None:
                    sparse_accumulator.add(hit_pixels, s_num[hit], s_den[hit])
                else:
                    assert (
                        numer_dense is not None
                        and denom_dense is not None
                        and overlap_dense is not None
                    )
                    numer_dense.index_add_(0, hit_pixels, s_num[hit])
                    denom_dense.index_add_(0, hit_pixels, s_den[hit])
                    overlap_dense.index_add_(
                        0,
                        hit_pixels,
                        torch.ones_like(hit_pixels, dtype=torch.int32),
                    )

    if not sparse:
        assert (
            numer_dense is not None
            and denom_dense is not None
            and overlap_dense is not None
        )
        nz = denom_dense > 0.0
        data_dense = torch.where(
            nz,
            numer_dense / denom_dense.clamp_min(1e-12),
            torch.zeros_like(numer_dense),
        )
        return HealpixResult(
            data=data_dense,
            weight=denom_dense,
            nside=nside_i,
            nest=nest,
            sparse=False,
            overlap_count=overlap_dense,
            coverage_moc=union_moc,
        )

    assert sparse_accumulator is not None
    nside_cov = min(int(nside_coverage), nside_i)
    if sparse_accumulator.pixels.numel() == 0:
        empty_data = HealSparseMap.from_pixels(
            sparse_accumulator.pixels,
            sparse_accumulator.numer,
            nside_sparse=nside_i,
            nest=nest,
            nside_coverage=nside_cov,
            sentinel=0.0,
        )
        empty_weight = HealSparseMap.from_pixels(
            sparse_accumulator.pixels,
            sparse_accumulator.denom,
            nside_sparse=nside_i,
            nest=nest,
            nside_coverage=nside_cov,
            sentinel=0.0,
        )
        empty_overlap = HealSparseMap.from_pixels(
            sparse_accumulator.pixels,
            sparse_accumulator.overlap,
            nside_sparse=nside_i,
            nest=nest,
            nside_coverage=nside_cov,
            sentinel=0.0,
        )
        return HealpixResult(
            data=empty_data,
            weight=empty_weight,
            nside=nside_i,
            nest=nest,
            sparse=True,
            overlap_count=empty_overlap,
            coverage_moc=union_moc,
        )

    u_data = sparse_accumulator.numer / sparse_accumulator.denom.clamp_min(1e-12)

    data_smap = HealSparseMap.from_pixels(
        sparse_accumulator.pixels,
        u_data,
        nside_sparse=nside_i,
        nest=nest,
        nside_coverage=nside_cov,
        sentinel=0.0,
    )
    weight_smap = HealSparseMap.from_pixels(
        sparse_accumulator.pixels,
        sparse_accumulator.denom,
        nside_sparse=nside_i,
        nest=nest,
        nside_coverage=nside_cov,
        sentinel=0.0,
    )
    ovl_smap = HealSparseMap.from_pixels(
        sparse_accumulator.pixels,
        sparse_accumulator.overlap,
        nside_sparse=nside_i,
        nest=nest,
        nside_coverage=nside_cov,
        sentinel=0.0,
    )
    return HealpixResult(
        data=data_smap,
        weight=weight_smap,
        nside=nside_i,
        nest=nest,
        sparse=True,
        overlap_count=ovl_smap,
        coverage_moc=union_moc,
    )


def resample_healpix(
    healpix_map: Tensor | HealpixResult | Any,
    target: Target | Mapping[str, Any] | WCS,
    *,
    nside: int | None = None,
    nest: bool = False,
    method: Literal["bilinear", "nearest", "harmonic"] = "bilinear",
    device: torch.device | str | None = None,
) -> MapResult:
    """Project a dense or sparse HEALPix map (or ``HealpixResult``) onto a 2D target WCS."""
    from ..sphere import healpix as _hp
    from ..sphere.core import sample_healpix_map
    from ..sphere.sparse import HealSparseMap
    from .healpix_spectral import sample_healpix_map_on_image_header_harmonic

    map_device = _resolve_map_device(device)
    raw_hdr = _coerce_target_header(target)
    if raw_hdr is None:
        raise ValueError("target must provide a 2D WCS or header")
    H, W = target_shape_from_header(raw_hdr, fallback=(64, 64))
    header_dict = normalized_header_for_shape(dict(raw_hdr), (H, W), overrides=None)
    target_wcs = WCS(header_dict).to(map_device)

    weight_smap: HealSparseMap | None = None
    weight_dense: Tensor | None = None
    if isinstance(healpix_map, HealpixResult):
        nest = healpix_map.nest
        nside = healpix_map.nside
        if healpix_map.sparse:
            weight_smap = healpix_map.weight
            healpix_map = healpix_map.data
        else:
            weight_dense = healpix_map.weight
            healpix_map = healpix_map.data

    if isinstance(healpix_map, HealSparseMap):
        ra_g, dec_g = _build_target_world_grid(target_wcs, H, W, device=map_device)
        interp_mode: Literal["bilinear", "nearest"] = (
            "nearest" if method == "nearest" else "bilinear"
        )
        sampled = healpix_map.interpolate(ra_g, dec_g, method=interp_mode).to(
            device=map_device, dtype=torch.float32
        )
        valid = torch.isfinite(sampled) & (
            sampled != float(healpix_map.fill_value)
        )
        data_2d = torch.where(valid, sampled, torch.zeros_like(sampled))
        if weight_smap is not None:
            w_2d = weight_smap.interpolate(ra_g, dec_g, method=interp_mode).to(
                device=map_device, dtype=torch.float32
            )
            w_2d = torch.where(valid & (w_2d > 0.0), w_2d, torch.zeros_like(data_2d))
        else:
            w_2d = valid.to(dtype=torch.float32)
        return MapResult(data=data_2d, weight=w_2d, header=header_dict)

    map_t = torch.as_tensor(healpix_map, device=map_device, dtype=torch.float32)
    resolved_nside = int(nside) if nside is not None else _hp.npix2nside(int(map_t.shape[-1]))
    if method == "harmonic":
        data_2d = sample_healpix_map_on_image_header_harmonic(
            map_t, header_dict, nest=nest
        ).to(device=map_device, dtype=torch.float32)
        w_2d = torch.isfinite(data_2d).to(dtype=torch.float32)
        return MapResult(data=data_2d, weight=w_2d, header=header_dict)

    ra_g, dec_g = _build_target_world_grid(target_wcs, H, W, device=map_device)
    interp_mode_d: Literal["bilinear", "nearest"] = (
        "nearest" if method == "nearest" else "bilinear"
    )
    if weight_dense is not None:
        w_t = weight_dense.to(device=map_device, dtype=torch.float32)
        good_hp = torch.isfinite(map_t) & (~_hp.mask_bad(map_t)) & (w_t > 0.0)
        safe_map = torch.where(good_hp, map_t, torch.zeros_like(map_t))
        safe_w = torch.where(good_hp, w_t, torch.zeros_like(w_t))
        num_2d = sample_healpix_map(
            safe_map * safe_w,
            nside=resolved_nside,
            lon_deg=ra_g,
            lat_deg=dec_g,
            nest=nest,
            interpolation=interp_mode_d,
        ).to(device=map_device, dtype=torch.float32)
        w_2d = sample_healpix_map(
            safe_w,
            nside=resolved_nside,
            lon_deg=ra_g,
            lat_deg=dec_g,
            nest=nest,
            interpolation=interp_mode_d,
        ).to(device=map_device, dtype=torch.float32)
        valid = torch.isfinite(num_2d) & torch.isfinite(w_2d) & (w_2d > 0.0)
        data_2d = torch.where(
            valid, num_2d / w_2d.clamp_min(1e-12), torch.zeros_like(num_2d)
        )
        w_2d = torch.where(valid, w_2d, torch.zeros_like(w_2d))
        return MapResult(data=data_2d, weight=w_2d, header=header_dict)

    data_2d = sample_healpix_map(
        map_t,
        nside=resolved_nside,
        lon_deg=ra_g,
        lat_deg=dec_g,
        nest=nest,
        interpolation=interp_mode_d,
    ).to(device=map_device, dtype=torch.float32)
    valid = torch.isfinite(data_2d) & (~_hp.mask_bad(data_2d))
    data_2d = torch.where(valid, data_2d, torch.zeros_like(data_2d))
    w_2d = valid.to(dtype=torch.float32)
    return MapResult(data=data_2d, weight=w_2d, header=header_dict)
