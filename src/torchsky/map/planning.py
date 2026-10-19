"""Target planning for image-plane map products.

Split out of :mod:`torchsky.map.image_plane`; the planners read observation
metadata and WCS headers, so the image-plane primitives they need are imported
lazily to avoid a module import cycle.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from .observation import (
    ImageObservation,
    PlannedImagePlaneCubeTarget,
    PlannedImagePlaneTarget,
)
from ..spectral import SpectralAxis, spectral_axis_from_header, spectral_bin_edges_from_header
from .spectral_projection import (
    _build_band_metadata,
    _build_spectral_projection,
    _observation_spectral_columns,
    _validated_spectral_axis,
    _validated_spectral_bin_edges,
)
from .operators.image_grid import normalized_header_for_shape, target_shape_from_header
from ..wcs import WCS


def _image_plane():
    from . import image_plane
    return image_plane


def _spherical_union_center(footprints: Tensor) -> tuple[float, float]:
    """Return a seam-safe mean direction for celestial footprint points."""
    ra = torch.deg2rad(footprints[:, 0].to(dtype=torch.float64))
    dec = torch.deg2rad(footprints[:, 1].to(dtype=torch.float64))
    cos_dec = torch.cos(dec)
    mean = torch.stack(
        (cos_dec * torch.cos(ra), cos_dec * torch.sin(ra), torch.sin(dec)),
        dim=-1,
    ).mean(dim=0)
    norm = torch.linalg.vector_norm(mean)
    if not bool(torch.isfinite(norm)) or float(norm) <= 1.0e-12:
        raise RuntimeError(
            "Union footprint has no unique spherical centre; provide a reference "
            "target or partition the mosaic"
        )
    mean = mean / norm
    center_ra = torch.rad2deg(torch.atan2(mean[1], mean[0])) % 360.0
    center_dec = torch.rad2deg(torch.asin(torch.clamp(mean[2], -1.0, 1.0)))
    return float(center_ra), float(center_dec)


def _world_to_tangent_plane(
    footprint: Tensor,
    *,
    center_ra_deg: float,
    center_dec_deg: float,
) -> tuple[Tensor, Tensor]:
    """Project celestial coordinates onto the exact gnomonic tangent plane."""
    ra = torch.deg2rad(footprint[:, 0].to(dtype=torch.float64))
    dec = torch.deg2rad(footprint[:, 1].to(dtype=torch.float64))
    ra0 = math.radians(center_ra_deg)
    dec0 = math.radians(center_dec_deg)
    dra = torch.remainder(ra - ra0 + math.pi, 2.0 * math.pi) - math.pi
    sin_dec, cos_dec = torch.sin(dec), torch.cos(dec)
    sin_dec0, cos_dec0 = math.sin(dec0), math.cos(dec0)
    denom = sin_dec0 * sin_dec + cos_dec0 * cos_dec * torch.cos(dra)
    if bool(torch.any(~torch.isfinite(denom))) or bool(torch.any(denom <= 0.0)):
        raise RuntimeError(
            "Union footprint crosses the TAN projection horizon; provide a "
            "reference target or partition the mosaic"
        )
    radians_to_degrees = 180.0 / math.pi
    xi = cos_dec * torch.sin(dra) / denom * radians_to_degrees
    eta = (
        (cos_dec0 * sin_dec - sin_dec0 * cos_dec * torch.cos(dra))
        / denom
        * radians_to_degrees
    )
    if bool(torch.any(~torch.isfinite(xi))) or bool(torch.any(~torch.isfinite(eta))):
        raise RuntimeError("Union footprint produced non-finite TAN coordinates")
    return xi, eta


def _adaptive_tangent_boundary(
    wcs: WCS,
    *,
    height: int,
    width: int,
    center_ra_deg: float,
    center_dec_deg: float,
    pixel_scale_deg: float,
    tolerance_px: float,
    max_depth: int,
) -> Tensor:
    def project(pixel: tuple[float, float]) -> Tensor:
        pixel_tensor = torch.tensor([pixel], dtype=torch.float64)
        world = wcs.pixel_to_world(pixel_tensor)
        if isinstance(world, tuple):
            world = torch.stack(world[:2], dim=-1)
        world = world.detach().to(device="cpu", dtype=torch.float64)
        xi, eta = _world_to_tangent_plane(
            world,
            center_ra_deg=center_ra_deg,
            center_dec_deg=center_dec_deg,
        )
        point = torch.stack((xi[0], eta[0]))
        if not bool(torch.isfinite(point).all()):
            raise RuntimeError("Observation boundary produced non-finite coordinates")
        return point

    output: list[Tensor] = []

    def sample_segment(
        p0: tuple[float, float],
        p1: tuple[float, float],
        q0: Tensor,
        q1: Tensor,
        depth: int,
    ) -> None:
        fractions = (0.25, 0.5, 0.75)
        pixels = [
            (
                p0[0] + fraction * (p1[0] - p0[0]),
                p0[1] + fraction * (p1[1] - p0[1]),
            )
            for fraction in fractions
        ]
        projected = [project(pixel) for pixel in pixels]
        errors = [
            torch.linalg.vector_norm(value - ((1.0 - fraction) * q0 + fraction * q1))
            / float(pixel_scale_deg)
            for fraction, value in zip(fractions, projected, strict=True)
        ]
        if max(float(error) for error in errors) <= float(tolerance_px):
            output.append(q1)
            return
        if depth >= int(max_depth):
            raise RuntimeError(
                "Could not bound curved observation footprint within the requested tolerance"
            )
        midpoint = pixels[1]
        q_midpoint = projected[1]
        sample_segment(p0, midpoint, q0, q_midpoint, depth + 1)
        sample_segment(midpoint, p1, q_midpoint, q1, depth + 1)

    corners = (
        (-0.5, -0.5),
        (float(width) - 0.5, -0.5),
        (float(width) - 0.5, float(height) - 0.5),
        (-0.5, float(height) - 0.5),
    )
    for edge_index in range(4):
        p0 = corners[edge_index]
        p1 = corners[(edge_index + 1) % 4]
        q0 = project(p0)
        q1 = project(p1)
        if not output:
            output.append(q0)
        sample_segment(p0, p1, q0, q1, 0)
    return torch.stack(output, dim=0)


def _resolve_plan_frame_prefixes(
    frame: str,
) -> tuple[str, str, str, str | None, float | None]:
    norm = str(frame).strip().lower()
    if norm in {"icrs", "equatorial", "j2000"}:
        return "ICRS", "RA--", "DEC-", "ICRS", None
    if norm == "fk5":
        return "FK5", "RA--", "DEC-", "FK5", 2000.0
    if norm in {"fk4", "b1950"}:
        return "FK4", "RA--", "DEC-", "FK4", 1950.0
    if norm in {"galactic", "gal"}:
        return "GALACTIC", "GLON", "GLAT", None, None
    if norm in {"ecliptic", "ecl"}:
        return "ECLIPTIC", "ELON", "ELAT", None, None
    raise ValueError(
        f"Unsupported coordinate frame {frame!r}; expected 'icrs', 'fk5', 'fk4', 'galactic', or 'ecliptic'"
    )


def _plan_union_footprint_target(
    observations: Sequence[ImageObservation],
    *,
    device: torch.device | str | None = None,
    padding: float = 1.05,
    tolerance_px: float = 0.25,
    max_depth: int = 20,
    projection: str = "TAN",
    frame: str = "icrs",
    center: str | tuple[float, float] = "union",
    pixel_scale: str | float = "median",
    pixel_scale_arcsec: float | None = None,
    shape: tuple[int, int] | None = None,
    rotation_deg: float = 0.0,
) -> tuple[dict, tuple[int, int]]:
    """Plan one rotation- and RA-seam-safe grid for all observations."""
    if not math.isfinite(float(padding)) or float(padding) < 1.0:
        raise ValueError("footprint_padding must be finite and at least 1.0")
    if not math.isfinite(float(tolerance_px)) or float(tolerance_px) <= 0.0:
        raise ValueError("footprint_tolerance_px must be finite and positive")
    if int(max_depth) < 1:
        raise ValueError("footprint_max_depth must be positive")
    proj_code = str(projection).strip().upper()
    target_frame, ctype1_prefix, ctype2_prefix, radesys, equinox = (
        _resolve_plan_frame_prefixes(frame)
    )
    is_default_tan = (
        proj_code == "TAN"
        and target_frame == "ICRS"
        and center == "union"
        and pixel_scale == "median"
        and pixel_scale_arcsec is None
        and shape is None
        and float(rotation_deg) == 0.0
    )

    footprints: list[Tensor] = []
    obs_centers: list[tuple[float, float]] = []
    pixel_scales: list[float] = []
    footprint_records: list[tuple[WCS, int, int]] = []
    for observation in observations:
        header = _image_plane()._observation_header_for_footprint(observation, device=device)
        wcs = WCS(header)
        height, width = _image_plane()._true_image_shape(header)
        pixel_edges = torch.tensor(
            [
                [-0.5, -0.5],
                [float(width) - 0.5, -0.5],
                [float(width) - 0.5, float(height) - 0.5],
                [-0.5, float(height) - 0.5],
            ],
            dtype=torch.float64,
        )
        footprint = wcs.pixel_to_world(pixel_edges)
        if isinstance(footprint, tuple):
            footprint = torch.stack(footprint[:2], dim=-1)
        footprint = footprint.detach().to(device="cpu", dtype=torch.float64)
        if footprint.shape != (4, 2) or not bool(torch.all(torch.isfinite(footprint))):
            raise RuntimeError(
                f"Could not determine finite footprint for {observation.path}"
            )
        if getattr(wcs, "celestial_frame", "ICRS") != target_frame or (
            equinox is not None and getattr(wcs, "celestial_equinox", None) != equinox
        ):
            lon_c, lat_c = wcs.convert_celestial(
                footprint[:, 0],
                footprint[:, 1],
                target_frame,
                to_equinox=equinox,
            )
            footprint = torch.stack((lon_c, lat_c), dim=-1)
        determinant = torch.linalg.det(
            wcs.cd.detach().to(device="cpu", dtype=torch.float64)
        )
        obs_scale = math.sqrt(abs(float(determinant)))
        if not math.isfinite(obs_scale) or obs_scale <= 0.0:
            raise RuntimeError(
                f"Could not determine a non-singular spatial pixel scale for {observation.path}"
            )
        footprints.append(footprint)
        obs_centers.append(_spherical_union_center(footprint))
        pixel_scales.append(obs_scale)
        footprint_records.append((wcs, height, width))
    if not footprints:
        raise RuntimeError("Could not determine any observation footprints")

    footprint = torch.cat(footprints, dim=0)
    if isinstance(center, tuple):
        center_ra = float(center[0]) % 360.0
        center_dec = float(center[1])
    elif str(center).strip().lower() == "most":
        best_idx = 0
        best_score = -1.0
        for idx_i, (ra_i, dec_i) in enumerate(obs_centers):
            score = 0.0
            for idx_j, (ra_j, dec_j) in enumerate(obs_centers):
                sep = _image_plane()._angular_separation_deg(ra_i, dec_i, ra_j, dec_j)
                score += 1.0 / (1.0 + (sep / max(pixel_scales[idx_j] * 512.0, 1e-3)) ** 2)
            if score > best_score:
                best_score = score
                best_idx = idx_i
        center_ra, center_dec = obs_centers[best_idx]
    elif str(center).strip().lower() == "union":
        center_ra, center_dec = _spherical_union_center(footprint)
    else:
        raise ValueError(
            f"Unsupported center={center!r}; expected 'union', 'most', or (lon_deg, lat_deg)"
        )

    if pixel_scale_arcsec is not None:
        resolved_scale = float(pixel_scale_arcsec) / 3600.0
    elif isinstance(pixel_scale, (int, float)):
        resolved_scale = float(pixel_scale)
    else:
        scale_mode = str(pixel_scale).strip().lower()
        if scale_mode == "median":
            resolved_scale = float(
                torch.tensor(pixel_scales, dtype=torch.float64).median()
            )
        elif scale_mode == "min":
            resolved_scale = float(min(pixel_scales))
        elif scale_mode == "max":
            resolved_scale = float(max(pixel_scales))
        else:
            raise ValueError(
                f"Unsupported pixel_scale={pixel_scale!r}; expected 'median', 'min', 'max', or a positive float"
            )
    if not math.isfinite(resolved_scale) or resolved_scale <= 0.0:
        raise ValueError("pixel_scale must be finite and positive")

    if is_default_tan:
        tangent_boundaries = [
            _adaptive_tangent_boundary(
                wcs,
                height=height,
                width=width,
                center_ra_deg=center_ra,
                center_dec_deg=center_dec,
                pixel_scale_deg=resolved_scale,
                tolerance_px=float(tolerance_px),
                max_depth=int(max_depth),
            )
            for wcs, height, width in footprint_records
        ]
        tangent_boundary = torch.cat(tangent_boundaries, dim=0)
        xi = tangent_boundary[:, 0]
        eta = tangent_boundary[:, 1]
        xi_mid = 0.5 * float(xi.min() + xi.max())
        eta_mid = 0.5 * float(eta.min() + eta.max())
        chord_margin_deg = float(tolerance_px) * resolved_scale
        xi_half = (0.5 * float(xi.max() - xi.min()) + chord_margin_deg) * float(padding)
        eta_half = (0.5 * float(eta.max() - eta.min()) + chord_margin_deg) * float(
            padding
        )
        xi_min, xi_max = xi_mid - xi_half, xi_mid + xi_half
        eta_min, eta_max = eta_mid - eta_half, eta_mid + eta_half
        naxis1 = max(1, int(math.ceil((xi_max - xi_min) / resolved_scale)))
        naxis2 = max(1, int(math.ceil((eta_max - eta_min) / resolved_scale)))
        header: dict = {
            "NAXIS": 2,
            "NAXIS1": naxis1,
            "NAXIS2": naxis2,
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
            "CUNIT1": "deg",
            "CUNIT2": "deg",
            "CRVAL1": center_ra,
            "CRVAL2": center_dec,
            "CRPIX1": 0.5 + xi_max / resolved_scale,
            "CRPIX2": 0.5 - eta_min / resolved_scale,
            "CD1_1": -resolved_scale,
            "CD1_2": 0.0,
            "CD2_1": 0.0,
            "CD2_2": resolved_scale,
        }
        return header, (naxis2, naxis1)

    rot_rad = math.radians(float(rotation_deg))
    cos_r = math.cos(rot_rad)
    sin_r = math.sin(rot_rad)
    cd1_1 = -resolved_scale * cos_r
    cd1_2 = -resolved_scale * sin_r
    cd2_1 = -resolved_scale * sin_r
    cd2_2 = resolved_scale * cos_r

    probe_header: dict[str, Any] = {
        "NAXIS": 2,
        "NAXIS1": 100,
        "NAXIS2": 100,
        "CTYPE1": f"{ctype1_prefix}-{proj_code}",
        "CTYPE2": f"{ctype2_prefix}-{proj_code}",
        "CUNIT1": "deg",
        "CUNIT2": "deg",
        "CRVAL1": center_ra,
        "CRVAL2": center_dec,
        "CRPIX1": 1.0,
        "CRPIX2": 1.0,
        "CD1_1": cd1_1,
        "CD1_2": cd1_2,
        "CD2_1": cd2_1,
        "CD2_2": cd2_2,
    }
    if radesys is not None:
        probe_header["RADESYS"] = radesys
    if equinox is not None:
        probe_header["Equinox"] = equinox
        probe_header["EQUINOX"] = equinox
        del probe_header["Equinox"]
    if proj_code in {"COE", "COO", "COD", "COP", "BON"}:
        probe_header["PV2_1"] = center_dec if abs(center_dec) >= 1.0 else 45.0
    elif proj_code == "HPX":
        probe_header["PV2_1"] = 4.0
        probe_header["PV2_2"] = 3.0
    probe_wcs = WCS(probe_header)

    if shape is not None:
        naxis2, naxis1 = int(shape[0]), int(shape[1])
        if naxis1 <= 0 or naxis2 <= 0:
            raise ValueError("shape dimensions must be positive")
        probe_header["NAXIS1"] = naxis1
        probe_header["NAXIS2"] = naxis2
        probe_header["CRPIX1"] = 0.5 * (naxis1 + 1)
        probe_header["CRPIX2"] = 0.5 * (naxis2 + 1)
        return probe_header, (naxis2, naxis1)

    boundary_world: list[Tensor] = []
    t_steps = torch.linspace(0.0, 1.0, 33, dtype=torch.float64)
    for wcs, height, width in footprint_records:
        w_lim = float(width) - 0.5
        h_lim = float(height) - 0.5
        e0 = torch.stack((-0.5 + (w_lim + 0.5) * t_steps, torch.full_like(t_steps, -0.5)), dim=-1)
        e1 = torch.stack((torch.full_like(t_steps, w_lim), -0.5 + (h_lim + 0.5) * t_steps), dim=-1)
        e2 = torch.stack((w_lim - (w_lim + 0.5) * t_steps, torch.full_like(t_steps, h_lim)), dim=-1)
        e3 = torch.stack((torch.full_like(t_steps, -0.5), h_lim - (h_lim + 0.5) * t_steps), dim=-1)
        edges_pix = torch.cat((e0, e1, e2, e3), dim=0)
        lon_e, lat_e = wcs.pixel_to_world(edges_pix[:, 0], edges_pix[:, 1])
        lon_e = lon_e.detach().to(device="cpu", dtype=torch.float64)
        lat_e = lat_e.detach().to(device="cpu", dtype=torch.float64)
        if getattr(wcs, "celestial_frame", "ICRS") != target_frame or (
            equinox is not None and getattr(wcs, "celestial_equinox", None) != equinox
        ):
            lon_e, lat_e = wcs.convert_celestial(
                lon_e, lat_e, target_frame, to_equinox=equinox
            )
        boundary_world.append(torch.stack((lon_e, lat_e), dim=-1))

    all_world = torch.cat(boundary_world, dim=0)
    px, py = probe_wcs.world_to_pixel(all_world[:, 0], all_world[:, 1])
    finite = torch.isfinite(px) & torch.isfinite(py)
    if not bool(finite.any()):
        raise RuntimeError(
            f"Observation footprints could not be projected into {proj_code}"
        )
    px = px[finite]
    py = py[finite]
    px_mid = 0.5 * float(px.min() + px.max())
    py_mid = 0.5 * float(py.min() + py.max())
    px_half = (0.5 * float(px.max() - px.min()) + float(tolerance_px)) * float(padding)
    py_half = (0.5 * float(py.max() - py.min()) + float(tolerance_px)) * float(padding)
    px_min, px_max = px_mid - px_half, px_mid + px_half
    py_min, py_max = py_mid - py_half, py_mid + py_half
    naxis1 = max(1, int(math.ceil(px_max - px_min)))
    naxis2 = max(1, int(math.ceil(py_max - py_min)))
    probe_header["NAXIS1"] = naxis1
    probe_header["NAXIS2"] = naxis2
    probe_header["CRPIX1"] = 1.0 - px_min + 0.5
    probe_header["CRPIX2"] = 1.0 - py_min + 0.5
    return probe_header, (naxis2, naxis1)


def plan_image_plane_target(
    observations: Sequence[ImageObservation],
    *,
    reference_index: int | None = 0,
    header_overrides: dict | None = None,
    device: torch.device | str | None = None,
    union_footprint: bool = False,
    footprint_padding: float = 1.05,
    footprint_tolerance_px: float = 0.25,
    footprint_max_depth: int = 20,
    max_footprint_pixels: int | None = None,
    projection: str = "TAN",
    frame: str = "icrs",
    center: str | tuple[float, float] = "union",
    pixel_scale: str | float = "median",
    pixel_scale_arcsec: float | None = None,
    shape: tuple[int, int] | None = None,
    rotation_deg: float = 0.0,
) -> PlannedImagePlaneTarget:
    if not observations:
        raise ValueError("At least one observation is required")
    if max_footprint_pixels is not None and int(max_footprint_pixels) <= 0:
        raise ValueError("max_footprint_pixels must be positive")

    if union_footprint or reference_index is None:
        header, planned_shape = _plan_union_footprint_target(
            observations,
            device=device,
            padding=footprint_padding,
            tolerance_px=footprint_tolerance_px,
            max_depth=footprint_max_depth,
            projection=projection,
            frame=frame,
            center=center,
            pixel_scale=pixel_scale,
            pixel_scale_arcsec=pixel_scale_arcsec,
            shape=shape,
            rotation_deg=rotation_deg,
        )
        if (
            max_footprint_pixels is not None
            and planned_shape[0] * planned_shape[1] > max_footprint_pixels
        ):
            raise RuntimeError(
                f"Union footprint output {planned_shape[0]}×{planned_shape[1]} "
                f"({planned_shape[0] * planned_shape[1] / 1e6:.1f} MPix) exceeds --max-footprint-pixels="
                f"{max_footprint_pixels}"
            )
        final_shape = target_shape_from_header(
            header_overrides or {},
            fallback=planned_shape,
        )
        return PlannedImagePlaneTarget(
            header=normalized_header_for_shape(
                header,
                final_shape,
                overrides=header_overrides,
            ),
            shape=final_shape,
        )

    if reference_index < 0 or reference_index >= len(observations):
        raise IndexError("reference_index is out of range")

    ref = observations[reference_index]
    read_dev = str(_image_plane()._resolve_map_device(device))
    image, header = _image_plane().load_observation_image_and_header(ref, device=read_dev)
    shape_out = tuple(image.shape) if shape is None else (int(shape[0]), int(shape[1]))
    shape_out = target_shape_from_header(header_overrides or {}, fallback=shape_out)
    normalized_header = normalized_header_for_shape(
        dict(header),
        shape_out,
        overrides=header_overrides,
    )
    return PlannedImagePlaneTarget(header=normalized_header, shape=shape_out)


def plan_image_plane_cube_target(
    observations: Sequence[ImageObservation],
    *,
    reference_index: int | None = 0,
    header_overrides: dict | None = None,
    spectral_axis: SpectralAxis | None = None,
    spectral_bin_edges: Tensor | None = None,
    spectral_unit: str = "Angstrom",
    spectral_type: str = "WAVE",
    device: torch.device | str | None = None,
    union_footprint: bool = False,
    footprint_padding: float = 1.05,
    footprint_tolerance_px: float = 0.25,
    footprint_max_depth: int = 20,
    max_footprint_pixels: int | None = None,
) -> PlannedImagePlaneCubeTarget:
    spatial_target = plan_image_plane_target(
        observations,
        reference_index=reference_index,
        header_overrides=header_overrides,
        device=device,
        union_footprint=union_footprint,
        footprint_padding=footprint_padding,
        footprint_tolerance_px=footprint_tolerance_px,
        footprint_max_depth=footprint_max_depth,
        max_footprint_pixels=max_footprint_pixels,
    )
    (
        observation_center,
        observation_lower,
        observation_upper,
        observation_filter_center,
        observation_filter_sigma,
        observation_names,
    ) = _observation_spectral_columns(observations)
    if spectral_axis is None:
        if spectral_bin_edges is not None:
            raise ValueError(
                "spectral_axis must be provided when spectral_bin_edges are specified"
            )
        spectral_axis = SpectralAxis(
            values=torch.unique(observation_center, sorted=True),
            unit=spectral_unit,
            type=spectral_type,
        )
    spectral_axis = _validated_spectral_axis(spectral_axis)
    validated_edges = None
    if spectral_bin_edges is not None:
        validated_edges = _validated_spectral_bin_edges(
            spectral_bin_edges,
            n_band=int(spectral_axis.values.numel()),
        )
    projection = _build_spectral_projection(
        observation_center,
        observation_lower,
        observation_upper,
        observation_filter_center,
        observation_filter_sigma,
        target_axis=spectral_axis,
        target_edges=validated_edges,
    )
    return PlannedImagePlaneCubeTarget(
        spatial_target=spatial_target,
        spectral_axis=spectral_axis,
        spectral_bin_edges=validated_edges,
        band_metadata=_build_band_metadata(
            spectral_axis,
            validated_edges,
            projection,
            observation_names,
        ),
    )


def plan_image_plane_cube_target_from_header(
    observations: Sequence[ImageObservation],
    spectral_header: dict,
    *,
    reference_index: int | None = 0,
    header_overrides: dict | None = None,
    use_header_bin_edges: bool = True,
    device: torch.device | str | None = None,
) -> PlannedImagePlaneCubeTarget:
    spectral_axis, _ = spectral_axis_from_header(spectral_header)
    spectral_bin_edges = None
    if use_header_bin_edges:
        spectral_bin_edges, _ = spectral_bin_edges_from_header(spectral_header)
    return plan_image_plane_cube_target(
        observations,
        reference_index=reference_index,
        header_overrides=header_overrides,
        spectral_axis=spectral_axis,
        spectral_bin_edges=spectral_bin_edges,
        device=device,
    )


def plan(
    observations: Sequence[ImageObservation],
    *,
    projection: str = "TAN",
    frame: str = "icrs",
    center: str | tuple[float, float] = "union",
    pixel_scale: str | float = "median",
    pixel_scale_arcsec: float | None = None,
    shape: tuple[int, int] | None = None,
    rotation_deg: float = 0.0,
    padding: float = 1.05,
    reference_index: int | None = None,
    header_overrides: dict | None = None,
    max_pixels: int | None = None,
    device: torch.device | str | None = None,
) -> PlannedImagePlaneTarget:
    """Plan a 2D celestial target grid from a set of observations."""
    return plan_image_plane_target(
        observations,
        reference_index=reference_index,
        header_overrides=header_overrides,
        device=device,
        union_footprint=(reference_index is None),
        footprint_padding=padding,
        max_footprint_pixels=max_pixels,
        projection=projection,
        frame=frame,
        center=center,
        pixel_scale=pixel_scale,
        pixel_scale_arcsec=pixel_scale_arcsec,
        shape=shape,
        rotation_deg=rotation_deg,
    )


def plan_cube(
    observations: Sequence[ImageObservation],
    *,
    spectral_axis: SpectralAxis | None = None,
    spectral_bin_edges: Tensor | None = None,
    spectral_header: dict | None = None,
    spectral_unit: str = "Angstrom",
    spectral_type: str = "WAVE",
    reference_index: int | None = None,
    header_overrides: dict | None = None,
    padding: float = 1.05,
    max_pixels: int | None = None,
    device: torch.device | str | None = None,
) -> PlannedImagePlaneCubeTarget:
    """Plan a 3D spatial-spectral target cube from a set of observations."""
    if spectral_header is not None:
        return plan_image_plane_cube_target_from_header(
            observations,
            spectral_header,
            reference_index=reference_index,
            header_overrides=header_overrides,
            device=device,
        )
    return plan_image_plane_cube_target(
        observations,
        reference_index=reference_index,
        header_overrides=header_overrides,
        spectral_axis=spectral_axis,
        spectral_bin_edges=spectral_bin_edges,
        spectral_unit=spectral_unit,
        spectral_type=spectral_type,
        device=device,
        union_footprint=(reference_index is None),
        footprint_padding=padding,
        max_footprint_pixels=max_pixels,
    )


def _obs_target_pixel_bbox(
    source_wcs: WCS,
    source_shape: tuple[int, int],
    full_target_wcs: WCS,
    *,
    margin: float = 8.0,
) -> tuple[float, float, float, float] | None:
    """Return (y_min, y_max, x_min, x_max) of source_wcs on full_target_wcs, or None if unbounded."""
    hs, ws = int(source_shape[0]), int(source_shape[1])
    if hs <= 0 or ws <= 0:
        return (0.0, -1.0, 0.0, -1.0)
    try:
        t = torch.linspace(0.0, 1.0, 9, dtype=source_wcs.crval.dtype, device=source_wcs.device)
        x0, x1 = -0.5, float(ws) - 0.5
        y0, y1 = -0.5, float(hs) - 0.5
        xs = torch.cat([x0 + t * (x1 - x0), torch.full_like(t, x1), x1 - t * (x1 - x0), torch.full_like(t, x0)])
        ys = torch.cat([torch.full_like(t, y0), y0 + t * (y1 - y0), torch.full_like(t, y1), y1 - t * (y1 - y0)])
        ra, dec = source_wcs.pixel_to_world(xs, ys)
        ra, dec = _image_plane()._convert_world_between_wcs(ra, dec, source_wcs, full_target_wcs)
        tx, ty = full_target_wcs.world_to_pixel(ra, dec)
        if not bool((torch.isfinite(tx) & torch.isfinite(ty)).all()):
            return None
        return (
            float(ty.min().item()) - margin,
            float(ty.max().item()) + margin,
            float(tx.min().item()) - margin,
            float(tx.max().item()) + margin,
        )
    except Exception:
        return None

