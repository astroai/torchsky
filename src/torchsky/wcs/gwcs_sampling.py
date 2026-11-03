"""Optional calibration-time sampling of GWCS-like detector transforms.

This module accepts a callable rather than importing GWCS. The resulting
runtime object contains only torch tensors and JSON-compatible provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping

import numpy as np
import torch
from torch import Tensor

from .detector_astrometry import (
    AngularCoordinates,
    RegularGridTransform2D,
    SampledTransformValidation,
    validate_sampled_transform,
)


def _reference_values(reference: Callable[..., Any], points: Tensor) -> Tensor:
    x = points[..., 0].detach().cpu().numpy()
    y = points[..., 1].detach().cpu().numpy()
    try:
        result = reference(x, y)
    except TypeError:
        result = reference(np.stack((x, y), axis=-1))
    if isinstance(result, tuple):
        if len(result) != 2:
            raise ValueError("reference tuple must contain two coordinate arrays")
        values = np.stack((np.asarray(result[0]), np.asarray(result[1])), axis=-1)
    else:
        values = np.asarray(result)
    output = torch.as_tensor(values, device=points.device, dtype=points.dtype)
    if output.shape != points.shape or not bool(torch.isfinite(output).all()):
        raise ValueError("reference must return finite coordinates matching (..., 2)")
    return output


def _validation_points(
    bounds: tuple[tuple[float, float], tuple[float, float]],
    *,
    random_points: int,
    seed: int,
    chip_boundaries_x: tuple[float, ...],
    dtype: torch.dtype,
) -> dict[str, Tensor]:
    (x0, x1), (y0, y1) = bounds
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    random = torch.rand(
        (max(int(random_points), 1), 2), generator=generator, dtype=dtype
    )
    random[:, 0] = x0 + (x1 - x0) * random[:, 0]
    random[:, 1] = y0 + (y1 - y0) * random[:, 1]
    edge_samples = max(int(random_points) // 8, 8)
    t = torch.linspace(0.0, 1.0, edge_samples, dtype=dtype)
    edges = torch.cat(
        (
            torch.stack((x0 + (x1 - x0) * t, torch.full_like(t, y0)), dim=-1),
            torch.stack((x0 + (x1 - x0) * t, torch.full_like(t, y1)), dim=-1),
            torch.stack((torch.full_like(t, x0), y0 + (y1 - y0) * t), dim=-1),
            torch.stack((torch.full_like(t, x1), y0 + (y1 - y0) * t), dim=-1),
        )
    )
    groups = {"random": random, "edges": edges}
    if chip_boundaries_x:
        line_y = torch.linspace(y0, y1, edge_samples, dtype=dtype)
        groups["chip_boundaries"] = torch.cat(
            [
                torch.stack((torch.full_like(line_y, x), line_y), dim=-1)
                for x in chip_boundaries_x
            ]
        )
    return groups


@dataclass(frozen=True)
class GWCSGridSamplingResult:
    """Frozen torch table, validation groups, and calibration provenance."""

    transform: RegularGridTransform2D
    validation: Mapping[str, SampledTransformValidation]
    grid_shape: tuple[int, int]
    detector_bounds: tuple[tuple[float, float], tuple[float, float]]
    reference_name: str
    calibration_provenance: Mapping[str, str]

    @property
    def passed(self) -> bool:
        return all(item.passed for item in self.validation.values())

    def provenance_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "reference_name": self.reference_name,
            "grid_shape_yx": list(self.grid_shape),
            "detector_bounds_xy": [list(axis) for axis in self.detector_bounds],
            "calibration": dict(self.calibration_provenance),
            "validation": {
                name: {
                    "p99_mas": result.p99_mas,
                    "max_mas": result.max_mas,
                    "p99_limit_mas": result.p99_limit_mas,
                    "max_limit_mas": result.max_limit_mas,
                    "passed": result.passed,
                }
                for name, result in self.validation.items()
            },
        }


def sample_gwcs_detector_to_tangent(
    reference: Callable[..., Any],
    *,
    detector_bounds: tuple[tuple[float, float], tuple[float, float]],
    initial_grid_shape: tuple[int, int] = (9, 9),
    maximum_grid_shape: tuple[int, int] = (257, 257),
    interpolation: Literal["bilinear", "bicubic"] = "bicubic",
    coordinates: AngularCoordinates = "plane_arcsec",
    random_validation_points: int = 4096,
    seed: int = 0,
    p99_limit_mas: float = 0.01,
    max_limit_mas: float = 0.05,
    chip_boundaries_x: tuple[float, ...] = (),
    reference_name: str = "gwcs_detector_to_v2v3",
    calibration_provenance: Mapping[str, str] | None = None,
    dtype: torch.dtype = torch.float64,
) -> GWCSGridSamplingResult:
    """Refine a detector table until independent points satisfy mas gates."""
    (x0, x1), (y0, y1) = detector_bounds
    if not all(np.isfinite((x0, x1, y0, y1))) or x1 <= x0 or y1 <= y0:
        raise ValueError("detector_bounds must contain finite increasing intervals")
    height, width = map(int, initial_grid_shape)
    max_height, max_width = map(int, maximum_grid_shape)
    minimum = 4 if interpolation == "bicubic" else 2
    if min(height, width) < minimum or height > max_height or width > max_width:
        raise ValueError("invalid initial/maximum grid shapes")
    groups = _validation_points(
        detector_bounds,
        random_points=random_validation_points,
        seed=seed,
        chip_boundaries_x=chip_boundaries_x,
        dtype=dtype,
    )
    while True:
        x = torch.linspace(x0, x1, width, dtype=dtype)
        y = torch.linspace(y0, y1, height, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        points = torch.stack((xx, yy), dim=-1)
        table = _reference_values(reference, points)
        transform = RegularGridTransform2D(
            table,
            origin=(x0, y0),
            spacing=((x1 - x0) / (width - 1), (y1 - y0) / (height - 1)),
            mode=interpolation,
        )
        validation = {
            name: validate_sampled_transform(
                transform,
                lambda p: _reference_values(reference, p),
                samples,
                coordinates=coordinates,
                p99_limit_mas=p99_limit_mas,
                max_limit_mas=max_limit_mas,
            )
            for name, samples in groups.items()
        }
        if all(result.passed for result in validation.values()):
            return GWCSGridSamplingResult(
                transform=transform,
                validation=validation,
                grid_shape=(height, width),
                detector_bounds=detector_bounds,
                reference_name=str(reference_name),
                calibration_provenance=dict(calibration_provenance or {}),
            )
        next_height = min(2 * (height - 1) + 1, max_height)
        next_width = min(2 * (width - 1) + 1, max_width)
        if (next_height, next_width) == (height, width):
            worst = max(validation.items(), key=lambda item: item[1].max_mas)
            raise ValueError(
                "GWCS sampling did not meet tolerance at maximum grid: "
                f"{worst[0]} max={worst[1].max_mas:.8g} mas"
            )
        height, width = next_height, next_width


@dataclass(frozen=True)
class FrozenDetectorEpochTables:
    """Time-dependent detector tables as frozen samples, not a live GWCS graph."""

    epochs: tuple[float, ...]
    samples: tuple[GWCSGridSamplingResult, ...]

    def transform_at(self, epoch: float) -> RegularGridTransform2D:
        if not self.epochs:
            raise ValueError("no frozen detector epochs")
        diffs = torch.tensor(self.epochs, dtype=torch.float64) - float(epoch)
        index = int(torch.argmin(diffs.abs()).item())
        return self.samples[index].transform

    def to(self, device: Any) -> "FrozenDetectorEpochTables":
        """Move frozen tables to ``device``. Sampling still happened on host."""
        device_t = torch.device(device)
        dtype = torch.float32 if device_t.type == "mps" else None
        samples = tuple(
            GWCSGridSamplingResult(
                transform=sample.transform.to(device=device_t, dtype=dtype),
                validation=sample.validation,
                grid_shape=sample.grid_shape,
                detector_bounds=sample.detector_bounds,
                reference_name=sample.reference_name,
                calibration_provenance=sample.calibration_provenance,
            )
            for sample in self.samples
        )
        return FrozenDetectorEpochTables(epochs=self.epochs, samples=samples)

    def evaluate(self, detector_xy: Any, epoch: float) -> Tensor:
        """Device-native table eval. Runtime does not import gwcs."""
        pts = torch.as_tensor(detector_xy)
        table = self.transform_at(epoch)
        return table(pts.to(device=table.table.device, dtype=table.table.dtype))

    def inverse(self, sky_xy: Any, epoch: float, **kwargs: Any) -> Tensor:
        """Newton inverse of the frozen table at the nearest epoch."""
        table = self.transform_at(epoch)
        return table.inverse(sky_xy, **kwargs)


def sample_gwcs_detector_epochs(
    references: Mapping[float, Callable[..., Any]],
    *,
    detector_bounds: tuple[tuple[float, float], tuple[float, float]],
    initial_grid_shape: tuple[int, int] = (9, 9),
    maximum_grid_shape: tuple[int, int] = (257, 257),
    interpolation: Literal["bilinear", "bicubic"] = "bicubic",
    coordinates: AngularCoordinates = "plane_arcsec",
    random_validation_points: int = 4096,
    seed: int = 0,
    p99_limit_mas: float = 0.01,
    max_limit_mas: float = 0.05,
    chip_boundaries_x: tuple[float, ...] = (),
    reference_name: str = "gwcs_detector_to_v2v3",
    calibration_provenance: Mapping[str, str] | None = None,
    dtype: torch.dtype = torch.float64,
) -> FrozenDetectorEpochTables:
    """Sample one frozen table per epoch. Runtime never imports ``gwcs``."""
    if not references:
        raise ValueError("references must contain at least one epoch")
    by_epoch = {float(epoch): fn for epoch, fn in references.items()}
    epochs = tuple(sorted(by_epoch))
    samples = tuple(
        sample_gwcs_detector_to_tangent(
            by_epoch[epoch],
            detector_bounds=detector_bounds,
            initial_grid_shape=initial_grid_shape,
            maximum_grid_shape=maximum_grid_shape,
            interpolation=interpolation,
            coordinates=coordinates,
            random_validation_points=random_validation_points,
            seed=seed,
            p99_limit_mas=p99_limit_mas,
            max_limit_mas=max_limit_mas,
            chip_boundaries_x=chip_boundaries_x,
            reference_name=f"{reference_name}_epoch_{epoch}",
            calibration_provenance=calibration_provenance,
            dtype=dtype,
        )
        for epoch in epochs
    )
    return FrozenDetectorEpochTables(epochs=epochs, samples=samples)


@dataclass(frozen=True)
class IFUSliceWCS:
    """Piecewise-disjoint 2D detector -> 3D ``(ra_deg, dec_deg, wavelength)`` IFU slice WCS.

    Prevents cross-slice interpolation bleed at IFU slice boundaries by restricting
    sub-pixel bilinear interpolation strictly to neighbors sharing the same ``slice_id >= 0``.
    """

    ra_map: Tensor
    dec_map: Tensor
    wavelength_map: Tensor
    slice_id_map: Tensor

    def __post_init__(self) -> None:
        if self.ra_map.ndim != 2:
            raise ValueError(
                "ra_map, dec_map, wavelength_map, and slice_id_map must be 2D (H, W)"
            )
        shape = self.ra_map.shape
        if (
            self.dec_map.shape != shape
            or self.wavelength_map.shape != shape
            or self.slice_id_map.shape != shape
        ):
            raise ValueError("all IFUSliceWCS maps must have identical (H, W) shape")

    @property
    def detector_shape(self) -> tuple[int, int]:
        return (int(self.ra_map.shape[0]), int(self.ra_map.shape[1]))

    @property
    def valid_mask(self) -> Tensor:
        return (
            (self.slice_id_map >= 0)
            & torch.isfinite(self.ra_map)
            & torch.isfinite(self.dec_map)
            & torch.isfinite(self.wavelength_map)
        )

    def to(self, device: Any) -> "IFUSliceWCS":
        dev = torch.device(device)
        dt = torch.float32 if dev.type == "mps" else torch.float64

        def move(value: Tensor, dtype: torch.dtype) -> Tensor:
            # Never a fused cross-device dtype copy: MPS float32 -> CPU float64 in
            # one ``.to`` returns zeros in PyTorch 2.14. Cast where the dtype exists.
            if dev.type == "mps":
                return value.to(dtype=dtype).to(device=dev)
            return value.to(device=dev).to(dtype=dtype)

        return IFUSliceWCS(
            ra_map=move(self.ra_map, dt),
            dec_map=move(self.dec_map, dt),
            wavelength_map=move(self.wavelength_map, dt),
            slice_id_map=move(self.slice_id_map, torch.int64),
        )

    @classmethod
    def from_slices(
        cls,
        detector_shape_or_slices: Any,
        slices: list[Any] | tuple[Any, ...] | None = None,
        *,
        device: Any = "cpu",
    ) -> "IFUSliceWCS":
        """Build an :class:`IFUSliceWCS` from ``(slice_mask, callable_or_tuple)`` pairs or slice dicts."""
        if slices is None:
            slice_list = list(detector_shape_or_slices)
            if not slice_list:
                raise ValueError(
                    "slices must not be empty when detector_shape is omitted"
                )
            first = slice_list[0]
            first_mask = first["mask"] if isinstance(first, dict) else first[0]
            m0 = torch.as_tensor(first_mask)
            h, w = int(m0.shape[-2]), int(m0.shape[-1])
        else:
            h, w = int(detector_shape_or_slices[0]), int(detector_shape_or_slices[1])
            slice_list = list(slices)

        dev = torch.device(device)
        dt = torch.float32 if dev.type == "mps" else torch.float64
        ra_full = torch.full((h, w), float("nan"), dtype=dt, device=dev)
        dec_full = torch.full((h, w), float("nan"), dtype=dt, device=dev)
        wave_full = torch.full((h, w), float("nan"), dtype=dt, device=dev)
        sid_full = torch.full((h, w), -1, dtype=torch.int64, device=dev)

        yy, xx = torch.meshgrid(
            torch.arange(h, dtype=dt, device=dev),
            torch.arange(w, dtype=dt, device=dev),
            indexing="ij",
        )
        spec: Any
        for slice_idx, item in enumerate(slice_list):
            if isinstance(item, dict):
                mask = item["mask"]
                if "wcs" in item:
                    wcs_obj = item["wcs"]
                    wave_spec = item["wavelength"]

                    def _eval_dict_slice(
                        x_pts: Tensor,
                        y_pts: Tensor,
                        _w: Any = wcs_obj,
                        _lam: Any = wave_spec,
                    ) -> tuple[Tensor, Tensor, Tensor]:
                        r_s, d_s = _w.pixel_to_world(x_pts, y_pts)
                        l_s: Any
                        if callable(_lam):
                            l_s = _lam(x_pts, y_pts)
                        else:
                            l_t = torch.as_tensor(_lam, dtype=dt, device=dev)
                            l_s = l_t[m] if l_t.shape == (h, w) else l_t
                        return r_s, d_s, l_s

                    spec = _eval_dict_slice
                else:
                    spec = (item["ra"], item["dec"], item["wavelength"])
            else:
                mask, spec = item
            m = torch.as_tensor(mask, dtype=torch.bool, device=dev)
            if m.shape != (h, w):
                raise ValueError(
                    f"slice {slice_idx} mask shape {tuple(m.shape)} != {(h, w)}"
                )
            if not bool(m.any()):
                continue
            if callable(spec):
                sampled: Any = spec(xx[m], yy[m])
                r_s, d_s, lam_s = sampled
                ra_full[m] = torch.as_tensor(r_s, dtype=dt, device=dev)
                dec_full[m] = torch.as_tensor(d_s, dtype=dt, device=dev)
                wave_full[m] = torch.as_tensor(lam_s, dtype=dt, device=dev)
            else:
                r_s, d_s, lam_s = spec
                r_t = torch.as_tensor(r_s, dtype=dt, device=dev)
                d_t = torch.as_tensor(d_s, dtype=dt, device=dev)
                l_t = torch.as_tensor(lam_s, dtype=dt, device=dev)
                if r_t.shape == (h, w):
                    ra_full[m] = r_t[m]
                    dec_full[m] = d_t[m]
                    wave_full[m] = l_t[m]
                else:
                    ra_full[m] = r_t
                    dec_full[m] = d_t
                    wave_full[m] = l_t
            sid_full[m] = int(slice_idx)
        return cls(
            ra_map=ra_full,
            dec_map=dec_full,
            wavelength_map=wave_full,
            slice_id_map=sid_full,
        )

    @classmethod
    def from_callable(
        cls,
        reference: Callable[..., Any],
        detector_shape: tuple[int, int],
        *,
        slice_id_map: Tensor | None = None,
        device: Any = "cpu",
    ) -> "IFUSliceWCS":
        """Sample a 2D -> 3D ``(ra, dec, wavelength)`` callable across the detector grid."""
        h, w = int(detector_shape[0]), int(detector_shape[1])
        dev = torch.device(device)
        dt = torch.float32 if dev.type == "mps" else torch.float64
        yy, xx = torch.meshgrid(
            torch.arange(h, dtype=dt, device=dev),
            torch.arange(w, dtype=dt, device=dev),
            indexing="ij",
        )
        ra, dec, wave = reference(xx, yy)
        ra_t = torch.as_tensor(ra, dtype=dt, device=dev).reshape(h, w)
        dec_t = torch.as_tensor(dec, dtype=dt, device=dev).reshape(h, w)
        wave_t = torch.as_tensor(wave, dtype=dt, device=dev).reshape(h, w)
        valid = torch.isfinite(ra_t) & torch.isfinite(dec_t) & torch.isfinite(wave_t)
        if slice_id_map is None:
            sid = torch.where(
                valid,
                torch.zeros((h, w), dtype=torch.int64, device=dev),
                torch.full((h, w), -1, dtype=torch.int64, device=dev),
            )
        else:
            sid = torch.as_tensor(slice_id_map, dtype=torch.int64, device=dev).reshape(
                h, w
            )
            sid = torch.where(valid, sid, torch.full_like(sid, -1))
        return cls(
            ra_map=ra_t,
            dec_map=dec_t,
            wavelength_map=wave_t,
            slice_id_map=sid,
        )

    def pixel_to_world(
        self, x: Tensor | float, y: Tensor | float
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Evaluate ``(ra_deg, dec_deg, wavelength)`` at sub-pixel ``(x, y)`` without cross-slice bleed."""
        dev = self.ra_map.device
        dt = self.ra_map.dtype
        xt = torch.as_tensor(x, dtype=dt, device=dev)
        yt = torch.as_tensor(y, dtype=dt, device=dev)
        xt, yt = torch.broadcast_tensors(xt, yt)
        orig_shape = xt.shape
        xf = xt.reshape(-1)
        yf = yt.reshape(-1)
        h, w = self.detector_shape

        in_bounds = (
            torch.isfinite(xf)
            & torch.isfinite(yf)
            & (xf >= -0.5)
            & (xf <= float(w) - 0.5)
            & (yf >= -0.5)
            & (yf <= float(h) - 0.5)
        )
        xi_nn = xf.round().to(torch.int64).clamp(0, w - 1)
        yi_nn = yf.round().to(torch.int64).clamp(0, h - 1)
        target_sid = torch.where(
            in_bounds,
            self.slice_id_map[yi_nn, xi_nn],
            torch.full_like(xi_nn, -1),
        )
        valid_query = target_sid >= 0

        x0 = torch.floor(xf).to(torch.int64).clamp(0, w - 1)
        y0 = torch.floor(yf).to(torch.int64).clamp(0, h - 1)
        x1 = (x0 + 1).clamp(0, w - 1)
        y1 = (y0 + 1).clamp(0, h - 1)
        fx_raw = xf - x0.to(dt)
        fy_raw = yf - y0.to(dt)
        fx = torch.where(
            (fx_raw >= 0.0) & (fx_raw <= 1.0), fx_raw, fx_raw.clamp(0.0, 1.0)
        )
        fy = torch.where(
            (fy_raw >= 0.0) & (fy_raw <= 1.0), fy_raw, fy_raw.clamp(0.0, 1.0)
        )

        w00 = (1.0 - fx) * (1.0 - fy)
        w10 = fx * (1.0 - fy)
        w01 = (1.0 - fx) * fy
        w11 = fx * fy

        m00 = (self.slice_id_map[y0, x0] == target_sid) & valid_query
        m10 = (self.slice_id_map[y0, x1] == target_sid) & valid_query
        m01 = (self.slice_id_map[y1, x0] == target_sid) & valid_query
        m11 = (self.slice_id_map[y1, x1] == target_sid) & valid_query

        w00 = torch.where(m00, w00, torch.zeros_like(w00))
        w10 = torch.where(m10, w10, torch.zeros_like(w10))
        w01 = torch.where(m01, w01, torch.zeros_like(w01))
        w11 = torch.where(m11, w11, torch.zeros_like(w11))
        w_sum = w00 + w10 + w01 + w11
        fallback_nn = valid_query & (w_sum <= 0.0)
        if bool(fallback_nn.any()):
            w00 = torch.where(fallback_nn, torch.ones_like(w00), w00)
            x0 = torch.where(fallback_nn, xi_nn, x0)
            y0 = torch.where(fallback_nn, yi_nn, y0)
            w_sum = w00 + w10 + w01 + w11

        safe_w = w_sum.clamp_min(1e-30)
        w00 = w00 / safe_w
        w10 = w10 / safe_w
        w01 = w01 / safe_w
        w11 = w11 / safe_w

        def _safe_gather(
            m_2d: Tensor, iy: Tensor, ix: Tensor, mask_b: Tensor
        ) -> Tensor:
            vals = m_2d[iy, ix]
            return torch.where(
                mask_b & torch.isfinite(vals), vals, torch.zeros_like(vals)
            )

        ra_ref = _safe_gather(self.ra_map, yi_nn, xi_nn, valid_query)

        def _interp_ra_delta(iy: Tensor, ix: Tensor, mask_b: Tensor) -> Tensor:
            r_v = _safe_gather(self.ra_map, iy, ix, mask_b)
            delta = torch.remainder(r_v - ra_ref + 180.0, 360.0) - 180.0
            return torch.where(mask_b, delta, torch.zeros_like(delta))

        m00_eff = m00 | fallback_nn
        ra_out = torch.remainder(
            ra_ref
            + w00 * _interp_ra_delta(y0, x0, m00_eff)
            + w10 * _interp_ra_delta(y0, x1, m10)
            + w01 * _interp_ra_delta(y1, x0, m01)
            + w11 * _interp_ra_delta(y1, x1, m11),
            360.0,
        )
        dec_out = (
            w00 * _safe_gather(self.dec_map, y0, x0, m00_eff)
            + w10 * _safe_gather(self.dec_map, y0, x1, m10)
            + w01 * _safe_gather(self.dec_map, y1, x0, m01)
            + w11 * _safe_gather(self.dec_map, y1, x1, m11)
        )
        wave_out = (
            w00 * _safe_gather(self.wavelength_map, y0, x0, m00_eff)
            + w10 * _safe_gather(self.wavelength_map, y0, x1, m10)
            + w01 * _safe_gather(self.wavelength_map, y1, x0, m01)
            + w11 * _safe_gather(self.wavelength_map, y1, x1, m11)
        )
        nan_v = torch.full_like(ra_out, float("nan"))
        ra_out = torch.where(valid_query, ra_out, nan_v).reshape(orig_shape)
        dec_out = torch.where(valid_query, dec_out, nan_v).reshape(orig_shape)
        wave_out = torch.where(valid_query, wave_out, nan_v).reshape(orig_shape)
        return ra_out, dec_out, wave_out


__all__ = [
    "FrozenDetectorEpochTables",
    "GWCSGridSamplingResult",
    "IFUSliceWCS",
    "sample_gwcs_detector_epochs",
    "sample_gwcs_detector_to_tangent",
]
