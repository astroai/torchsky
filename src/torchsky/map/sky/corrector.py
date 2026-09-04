from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Protocol

import torch
from torch import Tensor

from .estimator import MeshSkyEstimator


class _CompactDifferenceModel(Protocol):
    def render_window(self, y0: int, y1: int, x0: int, x1: int) -> Tensor: ...


_DifferenceTileFactory = Callable[
    [], Iterable[tuple[int, int, int, int, Tensor, Tensor]]
]


def _valid_pixels(
    images: Sequence[Tensor],
    masks: Sequence[Tensor | None] | None,
) -> list[Tensor]:
    if masks is not None and len(masks) != len(images):
        raise ValueError("masks count must match images count")
    if not images:
        return []
    shape = images[0].shape
    device = images[0].device
    valid: list[Tensor] = []
    for index, image in enumerate(images):
        if image.shape != shape:
            raise ValueError("all images must have the same shape")
        if image.device != device:
            raise ValueError("all images must be on the same device")
        support = torch.isfinite(image)
        mask = None if masks is None else masks[index]
        if mask is not None:
            if mask.shape != shape:
                raise ValueError("mask shape must match image shape")
            support = support & ~mask.to(device=device, dtype=torch.bool)
        valid.append(support)
    return valid


def _overlap_edges(valid: Sequence[Tensor]) -> list[tuple[int, int, Tensor, int]]:
    edges: list[tuple[int, int, Tensor, int]] = []
    # ponytail: O(n^2) is bounded for one exposure group; use a footprint index
    # before applying this to survey-wide plate graphs.
    for left in range(len(valid)):
        for right in range(left + 1, len(valid)):
            overlap = valid[left] & valid[right]
            count = int(overlap.sum().item())
            if count > 0:
                edges.append((left, right, overlap, count))
    return edges


def _require_anchor_connected(
    n_images: int,
    edges: Sequence[tuple[int, int, Tensor, int]],
) -> None:
    if n_images <= 1:
        return
    reached = {0}
    changed = True
    while changed:
        changed = False
        for left, right, _, _ in edges:
            if left in reached and right not in reached:
                reached.add(right)
                changed = True
            elif right in reached and left not in reached:
                reached.add(left)
                changed = True
    if len(reached) != n_images:
        raise ValueError("overlap graph is disconnected from anchor observation 0")


def _solve_anchored_graph(
    n_images: int,
    edges: Sequence[tuple[int, int, Tensor, int]],
    differences: Tensor,
) -> Tensor:
    if n_images == 0:
        return differences.new_empty((0, *differences.shape[1:]))
    if n_images == 1:
        return differences.new_zeros((1, *differences.shape[1:]))
    _require_anchor_connected(n_images, edges)
    if differences.shape[0] != len(edges):
        raise ValueError("difference count must match overlap graph edges")

    dtype = (
        torch.float64
        if differences.dtype == torch.float64 and differences.device.type != "mps"
        else torch.float32
    )
    incidence = torch.zeros(
        (len(edges), n_images), device=differences.device, dtype=dtype
    )
    counts = torch.tensor(
        [edge[3] for edge in edges], device=differences.device, dtype=dtype
    )
    weights = counts / counts.max()
    for row, (left, right, _, _) in enumerate(edges):
        incidence[row, left] = -1.0
        incidence[row, right] = 1.0
    weighted_incidence = weights[:, None] * incidence
    laplacian = incidence.T @ weighted_incidence
    flat_differences = differences.to(dtype=dtype).reshape(len(edges), -1)
    rhs = incidence.T @ (weights[:, None] * flat_differences)
    free = torch.linalg.solve(laplacian[1:, 1:], rhs[1:])
    solution = torch.cat(
        [torch.zeros((1, free.shape[1]), device=free.device, dtype=dtype), free],
        dim=0,
    )
    return solution.reshape(n_images, *differences.shape[1:]).to(
        dtype=differences.dtype
    )


class CrossExposureSkyCorrector:
    """Rectify overlapping exposures with an anchored weighted graph solve.

    Pairwise differences are measured only where both projected exposures have
    valid support. Observation zero is the explicit gauge anchor. ``correct``
    fits a smooth difference plane for every overlap edge and solves the same
    graph at every output pixel, preserving structure shared by the exposures.
    """

    def __init__(
        self,
        estimator: Callable[..., Tensor] | None = None,
    ) -> None:
        self._estimator = estimator

    def _get_estimator(self) -> Callable[..., Tensor]:
        if self._estimator is not None:
            return self._estimator
        return MeshSkyEstimator(block_size=128, statistic="median")

    def fit_offsets(
        self,
        images: list[Tensor],
        masks: list[Tensor | None] | None = None,
    ) -> Tensor:
        """Return overlap-derived additive offsets relative to observation zero."""
        if not images:
            return torch.empty(0)
        valid = _valid_pixels(images, masks)
        edges = _overlap_edges(valid)
        _require_anchor_connected(len(images), edges)
        if len(images) == 1:
            return images[0].new_zeros(1)
        differences = torch.stack(
            [
                torch.median((images[right] - images[left])[overlap])
                for left, right, overlap, _ in edges
            ]
        )
        return _solve_anchored_graph(len(images), edges, differences)

    def fit_difference_model(
        self,
        difference: Tensor,
        overlap: Tensor,
    ) -> Tensor:
        """Fit one smooth pairwise difference model on explicit overlap support."""
        if difference.ndim != 2:
            raise ValueError("overlap difference must be a 2D image")
        if overlap.shape != difference.shape:
            raise ValueError("overlap support must match difference shape")
        support = overlap.to(device=difference.device, dtype=torch.bool)
        if not bool(support.any()):
            raise ValueError("overlap support must contain at least one pixel")
        model = self._get_estimator()(difference, ~support)
        if model.shape != difference.shape:
            raise ValueError("overlap background model must match difference shape")
        if not bool(torch.isfinite(model).all()):
            raise ValueError("overlap background model contains non-finite values")
        return model

    def fit_difference_model_compact(
        self,
        tile_factory: _DifferenceTileFactory,
        *,
        image_shape: tuple[int, int],
        scratch: Tensor,
    ) -> _CompactDifferenceModel | None:
        """Fit a compact model when the configured estimator supports it."""

        fit_compact = getattr(self._get_estimator(), "fit_compact", None)
        if not callable(fit_compact):
            return None
        try:
            model = fit_compact(
                tile_factory,
                image_shape=image_shape,
                scratch=scratch,
            )
        except NotImplementedError:
            return None
        render_window = getattr(model, "render_window", None)
        if not callable(render_window):
            raise ValueError("compact overlap model must render bounded windows")
        return model

    def correct(
        self,
        images: list[Tensor],
        masks: list[Tensor | None] | None = None,
    ) -> list[Tensor]:
        """Subtract smooth overlap-difference corrections from all exposures."""
        if not images:
            return []
        valid = _valid_pixels(images, masks)
        edges = _overlap_edges(valid)
        _require_anchor_connected(len(images), edges)
        if len(images) == 1:
            return [images[0]]
        difference_models = torch.stack(
            [
                self.fit_difference_model(
                    images[right] - images[left],
                    overlap,
                )
                for left, right, overlap, _ in edges
            ]
        )
        corrections = _solve_anchored_graph(
            len(images), edges, difference_models
        )
        return [image - corrections[index] for index, image in enumerate(images)]
