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


__all__ = [
    "FrozenDetectorEpochTables",
    "GWCSGridSamplingResult",
    "sample_gwcs_detector_epochs",
    "sample_gwcs_detector_to_tangent",
]
