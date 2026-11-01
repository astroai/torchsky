"""Spectral-cube coaddition: filter-band cubes and 3D IFU cubes.

``coadd_cube`` dispatches to the 3D IFU path when observations carry a per-pixel
wavelength (``wavelength_map``/``ifu_wcs``) and to the 2D filter-band path
otherwise. The projection engine and the 2D image-plane coadd live in dedicated
map implementation modules.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, cast

import torch
from torch import Tensor

from ..spectral import SpectralAxis
from ..wcs import WCS
from .box_overlap import box_overlap_taps
from .calibration import (
    _apply_observation_calibration,
    _load_observation_weight_map,
)
from .image_plane_coadd import (
    _apply_observation_astrometric_shifts,
    coadd_image_plane_cube,
)
from .image_plane_common import (
    _coerce_target_header,
    _resolve_map_device,
    load_observation_image_and_header,
)
from .observation import (
    BandMetadataTable,
    CubeResult,
    CubeTarget,
    IFUObservation,
    Observation,
    PlannedImagePlaneCubeTarget,
    ResampleConfig,
    StackConfig,
    Target,
)
from .operators.config import (
    MapResamplingKernel,
    MapStackingMethod,
    MapValueSemantics,
)
from .operators.image_grid import normalized_header_for_shape, target_shape_from_header
from .spectral_projection import (
    _spectral_fractional_index,
    _validated_spectral_axis,
    _validated_spectral_bin_edges,
)


def _has_3d_ifu_observations(
    observations: Sequence[Observation | IFUObservation],
) -> bool:
    from ..wcs.gwcs_sampling import IFUSliceWCS

    return any(
        isinstance(obs, IFUObservation)
        or obs.wavelength_map is not None
        or obs.ifu_wcs is not None
        or isinstance(obs.wcs, IFUSliceWCS)
        for obs in observations
    )


def _coadd_ifu_3d_cube(
    observations: Sequence[Observation],
    target: CubeTarget | Target | Mapping[str, Any] | WCS | None,
    *,
    spectral_axis: SpectralAxis | Sequence[float] | Tensor | None,
    spectral_bin_edges: Sequence[float] | Tensor | None,
    spectral_unit: str,
    spectral_type: str,
    spectral_kernel: Literal["linear", "drizzle", "emsm", "msm", "nearest"],
    pixfrac: float,
    spectral_pixfrac: float,
    emsm_scale: float,
    device: torch.device | str | None,
) -> CubeResult:
    """Build a 3D cube from IFU detector images.

    ``inverse_variance`` contains diagonal voxel inverse variances only when every
    source weight is declared to be an inverse variance. Resampling one detector
    sample into multiple voxels induces cross-voxel covariance that is not stored;
    ``correlation_factor`` reports the local first-to-squared coefficient ratio as
    an aperture-noise diagnostic, not a covariance matrix.
    """
    from ..wcs.gwcs_sampling import IFUSliceWCS
    from ..wcs.iau2006 import apply_dcr, apply_ephemeris_offset

    map_device = _resolve_map_device(device)
    read_dev = str(map_device)

    if isinstance(target, (PlannedImagePlaneCubeTarget, CubeTarget)):
        header_dict = dict(target.header)
        H, W = int(target.shape[0]), int(target.shape[1])
        if spectral_axis is None:
            spectral_axis = target.spectral_axis
        if spectral_bin_edges is None:
            spectral_bin_edges = target.spectral_bin_edges
    else:
        raw_hdr = _coerce_target_header(target)
        if raw_hdr is None:
            raise ValueError(
                "target header or CubeTarget is required for 3D IFU cube building"
            )
        H, W = target_shape_from_header(raw_hdr, fallback=(32, 32))
        header_dict = normalized_header_for_shape(dict(raw_hdr), (H, W), overrides=None)

    if spectral_axis is None:
        raise ValueError(
            "spectral_axis (or spectral_bin_edges / a CubeTarget carrying one) must "
            "be provided for 3D IFU cube building; no default spectral sampling is assumed"
        )
    s_axis = _validated_spectral_axis(
        spectral_axis, default_unit=spectral_unit, default_type=spectral_type
    )
    wave_centers = s_axis.values.to(device=map_device, dtype=torch.float32)
    N_wave = int(wave_centers.numel())

    target_wcs = WCS(header_dict).to(map_device)
    numer = torch.zeros((N_wave, H, W), dtype=torch.float32, device=map_device)
    denom = torch.zeros((N_wave, H, W), dtype=torch.float32, device=map_device)
    variance_numer = torch.zeros_like(denom)
    overlap = torch.zeros((N_wave, H, W), dtype=torch.int32, device=map_device)
    band_obs_counts = torch.zeros((N_wave,), dtype=torch.int64)

    flat_numer = numer.reshape(-1)
    flat_denom = denom.reshape(-1)
    flat_variance_numer = variance_numer.reshape(-1)
    inverse_variance_available = bool(observations) and all(
        obs.weight_is_inverse_variance for obs in observations
    )
    stride_z = H * W
    stride_y = W

    for obs in observations:
        if obs.data is not None:
            raw_img = obs.data.to(device=map_device, dtype=torch.float32)
            hdr_obs = dict(obs.header) if obs.header is not None else {}
        else:
            raw_img, hdr_obs = load_observation_image_and_header(obs, device=read_dev)
            raw_img = raw_img.to(device=map_device, dtype=torch.float32)
        src_w = _load_observation_weight_map(obs, raw_img, device=map_device)
        cal_img, cal_w = _apply_observation_calibration(obs, raw_img, src_w)

        valid = torch.isfinite(cal_img) & torch.isfinite(cal_w) & (cal_w > 0.0)
        ifu_obj = (
            obs.ifu_wcs
            if obs.ifu_wcs is not None
            else (obs.wcs if isinstance(obs.wcs, IFUSliceWCS) else None)
        )
        if ifu_obj is not None and getattr(ifu_obj, "valid_mask", None) is not None:
            valid = valid & ifu_obj.valid_mask.to(device=map_device, dtype=torch.bool)
        if not bool(valid.any()):
            continue

        yy, xx = torch.nonzero(valid, as_tuple=True)
        val_v = cal_img[yy, xx].to(dtype=torch.float32)
        w_v = cal_w[yy, xx].to(dtype=torch.float32)

        w_dtype = torch.float32 if map_device.type == "mps" else torch.float64
        x_det = xx.to(dtype=w_dtype)
        y_det = yy.to(dtype=w_dtype)
        if ifu_obj is not None:
            ra_d, dec_d, lam_d = ifu_obj.to(map_device).pixel_to_world(x_det, y_det)
        else:
            obs_wcs = (
                obs.wcs
                if isinstance(obs.wcs, WCS)
                else WCS(obs.wcs if isinstance(obs.wcs, dict) else hdr_obs)
            ).to(map_device)
            ra_d, dec_d = obs_wcs.pixel_to_world(x_det, y_det)
            if obs.wavelength_map is not None:
                lam_d = obs.wavelength_map.to(device=map_device, dtype=w_dtype)[yy, xx]
            elif obs.wavelength is not None:
                lam_d = torch.full_like(ra_d, float(obs.wavelength))
            else:
                raise ValueError(
                    "Observation must define ifu_wcs, wavelength_map, or wavelength"
                )

        if obs.dcr_zenith_deg is not None:
            ra_d, dec_d = apply_dcr(
                ra_d,
                dec_d,
                lam_d,
                zenith_angle_deg=float(obs.dcr_zenith_deg),
                parallactic_angle_deg=float(obs.dcr_parallactic_deg),
                reference_wavelength_angstrom=float(obs.dcr_ref_wavelength),
                pressure_hpa=float(obs.dcr_pressure_hpa),
                temperature_c=float(obs.dcr_temperature_c),
            )
        if (
            obs.tracking_rate is not None
            and obs.mjd_obs is not None
            and obs.reference_mjd is not None
        ):
            ra_d, dec_d = apply_ephemeris_offset(
                ra_d,
                dec_d,
                d_ra_cosdec_arcsec_per_hr=float(obs.tracking_rate[0]),
                d_dec_arcsec_per_hr=float(obs.tracking_rate[1]),
                mjd_obs=float(obs.mjd_obs),
                mjd_ref=float(obs.reference_mjd),
            )

        xt, yt = target_wcs.world_to_pixel(ra_d, dec_d)
        xt = xt.to(dtype=torch.float32)
        yt = yt.to(dtype=torch.float32)
        lam_f = lam_d.to(dtype=torch.float32)
        zt = _spectral_fractional_index(wave_centers, lam_f)

        ok = torch.isfinite(xt) & torch.isfinite(yt) & torch.isfinite(zt)
        if not bool(ok.any()):
            continue
        xt, yt, zt, val_v, w_v = xt[ok], yt[ok], zt[ok], val_v[ok], w_v[ok]
        obs_hit_mask = torch.zeros_like(flat_denom, dtype=torch.bool)

        if spectral_kernel == "nearest":
            ix = torch.round(xt).to(torch.int64)
            iy = torch.round(yt).to(torch.int64)
            iz = torch.round(zt).to(torch.int64)
            in_b = (
                (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H) & (iz >= 0) & (iz < N_wave)
            )
            if bool(in_b.any()):
                idx = iz[in_b] * stride_z + iy[in_b] * stride_y + ix[in_b]
                flat_numer.scatter_add_(0, idx, val_v[in_b] * w_v[in_b])
                flat_denom.scatter_add_(0, idx, w_v[in_b])
                if inverse_variance_available:
                    flat_variance_numer.scatter_add_(0, idx, w_v[in_b])
                obs_hit_mask[idx] = True
        elif spectral_kernel == "emsm":
            sigma = max(float(emsm_scale), 1e-3)
            xc = torch.round(xt).to(torch.int64)
            yc = torch.round(yt).to(torch.int64)
            zc = torch.round(zt).to(torch.int64)
            # Compute normalized 3x3x3 Shepard exponential weights
            w_neighbors: list[tuple[int, int, int, Tensor]] = []
            w_norm = torch.zeros_like(xt)
            for dz in (-1, 0, 1):
                iz = zc + dz
                z_ok = (iz >= 0) & (iz < N_wave)
                dist_z2 = torch.square(iz.to(torch.float32) - zt)
                for dy in (-1, 0, 1):
                    iy = yc + dy
                    zy_ok = z_ok & (iy >= 0) & (iy < H)
                    dist_zy2 = dist_z2 + torch.square(iy.to(torch.float32) - yt)
                    for dx in (-1, 0, 1):
                        ix = xc + dx
                        in_b = zy_ok & (ix >= 0) & (ix < W)
                        dist2 = dist_zy2 + torch.square(ix.to(torch.float32) - xt)
                        wk = torch.where(
                            in_b,
                            torch.exp(-dist2 / (sigma * sigma)),
                            torch.zeros_like(dist2),
                        )
                        w_neighbors.append((dz, dy, dx, wk))
                        w_norm = w_norm + wk
            w_norm_safe = w_norm.clamp_min(1e-12)
            for dz, dy, dx, wk in w_neighbors:
                frac = wk / w_norm_safe
                pos = frac > 1e-6
                if not bool(pos.any()):
                    continue
                iz = (zc + dz)[pos]
                iy = (yc + dy)[pos]
                ix = (xc + dx)[pos]
                idx = iz * stride_z + iy * stride_y + ix
                eff_w = w_v[pos] * frac[pos]
                flat_numer.scatter_add_(0, idx, val_v[pos] * eff_w)
                flat_denom.scatter_add_(0, idx, eff_w)
                if inverse_variance_available:
                    flat_variance_numer.scatter_add_(
                        0, idx, w_v[pos] * torch.square(frac[pos])
                    )
                obs_hit_mask[idx] = True
        else:
            # 3D voxel drizzle / trilinear ("drizzle" or "linear")
            pf_xy = max(float(pixfrac), 1e-6) if spectral_kernel == "drizzle" else 1.0
            pf_z = (
                max(float(spectral_pixfrac), 1e-6)
                if spectral_kernel == "drizzle"
                else 1.0
            )
            if spectral_kernel == "drizzle":
                # Exact box/voxel overlap; footprints are in output voxels.
                x0, wxs = box_overlap_taps(xt, pf_xy)
                y0, wys = box_overlap_taps(yt, pf_xy)
                z0, wzs = box_overlap_taps(zt, pf_z)
            else:
                # Trilinear splat.
                x0, y0, z0 = (torch.floor(v).to(torch.int64) for v in (xt, yt, zt))
                dx, dy, dz = xt - x0, yt - y0, zt - z0
                wxs = torch.stack([1.0 - dx, dx])
                wys = torch.stack([1.0 - dy, dy])
                wzs = torch.stack([1.0 - dz, dz])
            for oz, wz in enumerate(wzs):
                iz = z0 + oz
                for oy, wy in enumerate(wys):
                    iy = y0 + oy
                    for ox, wx in enumerate(wxs):
                        ix = x0 + ox
                        frac = wz * wy * wx
                        in_b = (
                            (iz >= 0)
                            & (iz < N_wave)
                            & (iy >= 0)
                            & (iy < H)
                            & (ix >= 0)
                            & (ix < W)
                            & (frac > 0.0)
                        )
                        if not bool(in_b.any()):
                            continue
                        idx = iz[in_b] * stride_z + iy[in_b] * stride_y + ix[in_b]
                        eff_w = w_v[in_b] * frac[in_b]
                        flat_numer.scatter_add_(0, idx, val_v[in_b] * eff_w)
                        flat_denom.scatter_add_(0, idx, eff_w)
                        if inverse_variance_available:
                            flat_variance_numer.scatter_add_(
                                0, idx, w_v[in_b] * torch.square(frac[in_b])
                            )
                        obs_hit_mask[idx] = True

        obs_hit_3d = obs_hit_mask.reshape(N_wave, H, W)
        overlap.add_(obs_hit_3d.to(torch.int32))
        band_obs_counts.add_(obs_hit_3d.any(dim=(-2, -1)).cpu().to(torch.int64))

    nz = denom > 0.0
    data = torch.where(nz, numer / denom.clamp_min(1e-12), torch.zeros_like(numer))
    inverse_variance = None
    correlation_factor = None
    if inverse_variance_available:
        has_variance = variance_numer > 0.0
        inverse_variance = torch.where(
            has_variance,
            torch.square(denom) / variance_numer.clamp_min(1e-12),
            torch.zeros_like(denom),
        )
        correlation_factor = torch.where(
            has_variance,
            denom / variance_numer.clamp_min(1e-12),
            torch.zeros_like(denom),
        )
    centers_cpu = s_axis.values.to(dtype=torch.float64).cpu()
    if spectral_bin_edges is not None:
        edges_t = _validated_spectral_bin_edges(
            torch.as_tensor(spectral_bin_edges, dtype=torch.float64), n_band=N_wave
        )
        lower_c = edges_t[:-1].clone()
        upper_c = edges_t[1:].clone()
    else:
        lower_c = centers_cpu.clone()
        upper_c = centers_cpu.clone()
    band_meta = BandMetadataTable(
        band_index=torch.arange(N_wave, dtype=torch.int64),
        center=centers_cpu,
        lower=lower_c,
        upper=upper_c,
        observation_count=band_obs_counts,
        spectral_weight_sum=band_obs_counts.to(torch.float64),
        band_names=tuple(None for _ in range(N_wave)),
    )
    return CubeResult(
        spectral_axis=s_axis,
        band_metadata=band_meta,
        data=data,
        weight=denom,
        header=header_dict,
        spatial_wcs=WCS(header_dict),
        inverse_variance=inverse_variance,
        correlation_factor=correlation_factor,
        overlap_count=overlap,
    )


def coadd_cube(
    observations: Sequence[Observation | IFUObservation],
    target: CubeTarget | Target | Mapping[str, Any] | WCS | None = None,
    *,
    resample: ResampleConfig | None = None,
    stack: StackConfig | None = None,
    kernel: MapResamplingKernel | None = None,
    pixfrac: float | None = None,
    semantics: MapValueSemantics | None = None,
    method: MapStackingMethod | None = None,
    spectral_kernel: Literal["linear", "drizzle", "emsm", "msm", "nearest"]
    | None = None,
    spectral_pixfrac: float | None = None,
    emsm_scale: float | None = None,
    spatial_roi: Any | None = None,
    spectral_roi: Any | None = None,
    resolving_power: Any | None = None,
    response_chunk_size: int | None = None,
    max_response_entries: int | None = None,
    footprint_tolerance: float | None = None,
    footprint_max_depth: int | None = None,
    footprint_order: int | None = None,
    spectral_axis: SpectralAxis | Sequence[float] | Tensor | None = None,
    wavelengths: SpectralAxis | Sequence[float] | Tensor | None = None,
    spectral_bin_edges: Sequence[float] | Tensor | None = None,
    spectral_unit: str = "Angstrom",
    spectral_type: str = "WAVE",
    background_estimator: Callable[[Tensor, Tensor | None], Tensor] | None = None,
    device: torch.device | str | None = None,
) -> CubeResult:
    """Coadd filter images or IFU slices into an image-plane spectral cube.

    Args:
        observations: Filter-image or IFU observations to combine.
        target: Spatial and optional spectral target geometry.
        resample: Structured spatial and spectral resampling configuration.
        stack: Structured stacking configuration.
        spectral_axis: Output spectral-axis definition.
        spectral_bin_edges: Explicit output spectral-bin edges.
        spectral_unit: Unit label for a constructed output spectral axis.
        spectral_type: FITS-style spectral coordinate type.
        device: Optional compute device.

    Returns:
        Coadded cube, weights, coverage, mask, and target metadata.

    Examples:
        >>> from torchsky.map import coadd_cube
        >>> callable(coadd_cube)
        True
    """
    if spectral_axis is None and wavelengths is not None:
        spectral_axis = wavelengths
    r_cfg = resample or ResampleConfig()
    s_cfg = stack or StackConfig()
    resolved_spec_kernel = (
        spectral_kernel if spectral_kernel is not None else r_cfg.spectral_kernel
    )
    if any(isinstance(observation, IFUObservation) for observation in observations):
        if not all(
            isinstance(observation, IFUObservation) for observation in observations
        ):
            raise TypeError(
                "physical IFU cube building cannot mix IFUObservation and Observation"
            )
        if not isinstance(target, CubeTarget):
            raise TypeError(
                "IFUObservation requires a CubeTarget with spectral bin edges"
            )
        if resolved_spec_kernel not in {"drizzle", "emsm", "msm"}:
            resolved_spec_kernel = "drizzle"
        from .ifu import build_ifu_cube

        return build_ifu_cube(
            cast(Sequence[IFUObservation], observations),
            target,
            kernel=resolved_spec_kernel,
            pixfrac=pixfrac if pixfrac is not None else r_cfg.pixfrac,
            spectral_pixfrac=(
                spectral_pixfrac
                if spectral_pixfrac is not None
                else r_cfg.spectral_pixfrac
            ),
            emsm_scale=emsm_scale if emsm_scale is not None else r_cfg.emsm_scale,
            spatial_roi=spatial_roi if spatial_roi is not None else r_cfg.spatial_roi,
            spectral_roi=(
                spectral_roi if spectral_roi is not None else r_cfg.spectral_roi
            ),
            resolving_power=(
                resolving_power
                if resolving_power is not None
                else r_cfg.resolving_power
            ),
            chunk_size=(
                response_chunk_size
                if response_chunk_size is not None
                else r_cfg.response_chunk_size
            ),
            max_response_entries=(
                max_response_entries
                if max_response_entries is not None
                else r_cfg.max_response_entries
            ),
            footprint_tolerance=(
                footprint_tolerance
                if footprint_tolerance is not None
                else r_cfg.footprint_tolerance
            ),
            footprint_max_depth=(
                footprint_max_depth
                if footprint_max_depth is not None
                else r_cfg.footprint_max_depth
            ),
            device=device,
        )
    legacy_observations = cast(Sequence[Observation], observations)
    if _has_3d_ifu_observations(legacy_observations) or resolved_spec_kernel == "emsm":
        return _coadd_ifu_3d_cube(
            legacy_observations,
            target,
            spectral_axis=spectral_axis,
            spectral_bin_edges=spectral_bin_edges,
            spectral_unit=spectral_unit,
            spectral_type=spectral_type,
            spectral_kernel=resolved_spec_kernel,
            pixfrac=pixfrac if pixfrac is not None else r_cfg.pixfrac,
            spectral_pixfrac=(
                spectral_pixfrac
                if spectral_pixfrac is not None
                else r_cfg.spectral_pixfrac
            ),
            emsm_scale=emsm_scale if emsm_scale is not None else r_cfg.emsm_scale,
            device=device,
        )

    obs_shifted = _apply_observation_astrometric_shifts(observations, device)
    if (
        isinstance(target, PlannedImagePlaneCubeTarget)
        and spectral_axis is None
        and spectral_bin_edges is None
    ):
        return coadd_image_plane_cube(
            obs_shifted,
            target=target,
            footprint_order=footprint_order
            if footprint_order is not None
            else s_cfg.footprint_order,
            resampling_kernel=kernel if kernel is not None else r_cfg.kernel,
            value_semantics=semantics if semantics is not None else r_cfg.semantics,
            stacking_method=method if method is not None else s_cfg.method,
            drizzle_pixfrac=pixfrac if pixfrac is not None else r_cfg.pixfrac,
            spectral_unit=spectral_unit,
            spectral_type=spectral_type,
            device=device,
            background_estimator=background_estimator,
        )
    if isinstance(target, (PlannedImagePlaneCubeTarget, CubeTarget)):
        target_hdr = dict(target.header)
        if spectral_axis is None:
            spectral_axis = target.spectral_axis
        if spectral_bin_edges is None:
            spectral_bin_edges = target.spectral_bin_edges
    else:
        target_hdr = _coerce_target_header(target)
    return coadd_image_plane_cube(
        obs_shifted,
        target_header=target_hdr,
        footprint_order=footprint_order
        if footprint_order is not None
        else s_cfg.footprint_order,
        resampling_kernel=kernel if kernel is not None else r_cfg.kernel,
        value_semantics=semantics if semantics is not None else r_cfg.semantics,
        stacking_method=method if method is not None else s_cfg.method,
        drizzle_pixfrac=pixfrac if pixfrac is not None else r_cfg.pixfrac,
        spectral_axis=spectral_axis,
        spectral_bin_edges=spectral_bin_edges,
        spectral_unit=spectral_unit,
        spectral_type=spectral_type,
        device=device,
        background_estimator=background_estimator,
    )
