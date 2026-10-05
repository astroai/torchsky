from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Protocol

import torch
from torch import Tensor

from .estimator import MeshSkyEstimator
from .polynomial import CompactPolynomialSkyModel, PolynomialSkyModel


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


def _support_bbox(mask_2d: Tensor) -> tuple[int, int, int, int] | None:
    row_any = mask_2d.any(dim=1)
    if not bool(row_any.any()):
        return None
    col_any = mask_2d.any(dim=0)
    rows = torch.nonzero(row_any, as_tuple=False).squeeze(-1)
    cols = torch.nonzero(col_any, as_tuple=False).squeeze(-1)
    return (
        int(rows[0].item()),
        int(rows[-1].item()) + 1,
        int(cols[0].item()),
        int(cols[-1].item()) + 1,
    )


def _overlap_edges_with_bboxes(
    valid: Sequence[Tensor],
) -> list[tuple[int, int, Tensor, int, tuple[int, int, int, int]]]:
    bboxes = [_support_bbox(v) for v in valid]
    edges: list[tuple[int, int, Tensor, int, tuple[int, int, int, int]]] = []
    for left in range(len(valid)):
        box_l = bboxes[left]
        if box_l is None:
            continue
        ly0, ly1, lx0, lx1 = box_l
        for right in range(left + 1, len(valid)):
            box_r = bboxes[right]
            if box_r is None:
                continue
            ry0, ry1, rx0, rx1 = box_r
            iy0 = max(ly0, ry0)
            iy1 = min(ly1, ry1)
            ix0 = max(lx0, rx0)
            ix1 = min(lx1, rx1)
            if iy1 <= iy0 or ix1 <= ix0:
                continue
            sub_overlap = valid[left][iy0:iy1, ix0:ix1] & valid[right][iy0:iy1, ix0:ix1]
            count = int(sub_overlap.sum().item())
            if count > 0:
                overlap = torch.zeros_like(valid[left], dtype=torch.bool)
                overlap[iy0:iy1, ix0:ix1] = sub_overlap
                edges.append((left, right, overlap, count, (iy0, iy1, ix0, ix1)))
    return edges


def _overlap_edges(valid: Sequence[Tensor]) -> list[tuple[int, int, Tensor, int]]:
    return [
        (left, right, overlap, count)
        for left, right, overlap, count, _ in _overlap_edges_with_bboxes(valid)
    ]


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


def _fit_planar_coefficients_on_bbox(
    diff_sub: Tensor,
    support_sub: Tensor,
    *,
    full_shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
) -> Tensor:
    """Fit Chebyshev order-1 plane [c0, cx, cy] on a supported bbox slice."""
    height, width = int(full_shape[0]), int(full_shape[1])
    iy0, iy1, ix0, ix1 = bbox
    device = diff_sub.device
    dtype = torch.float32 if device.type == "mps" else torch.float64
    ys, xs = torch.nonzero(support_sub, as_tuple=True)
    vals = diff_sub[ys, xs].to(dtype=dtype)
    if vals.numel() == 0:
        return torch.zeros(3, device=device, dtype=diff_sub.dtype)
    med = torch.median(vals)
    abs_dev = torch.abs(vals - med)
    mad = torch.median(abs_dev)
    sigma = 1.482602218505602 * mad
    if float(sigma.item()) > 1e-12:
        u = abs_dev / (1.345 * sigma)
        w = torch.where(u <= 1.0, torch.ones_like(u), 1.0 / u)
    else:
        w = torch.ones_like(vals)
    if vals.numel() < 3:
        out = torch.zeros(3, device=device, dtype=diff_sub.dtype)
        out[0] = med.to(dtype=diff_sub.dtype)
        return out
    gx = (2.0 * (xs.to(dtype=dtype) + float(ix0)) / max(width - 1, 1)) - 1.0
    gy = (2.0 * (ys.to(dtype=dtype) + float(iy0)) / max(height - 1, 1)) - 1.0
    basis = torch.stack((torch.ones_like(gx), gx, gy), dim=-1)
    sqrt_w = torch.sqrt(w).unsqueeze(-1)
    sol = torch.linalg.lstsq(
        basis * sqrt_w,
        (vals * sqrt_w.squeeze(-1)).unsqueeze(-1),
        rcond=None,
    ).solution.squeeze(-1)
    if not bool(torch.isfinite(sol).all()):
        out = torch.zeros(3, device=device, dtype=diff_sub.dtype)
        out[0] = med.to(dtype=diff_sub.dtype)
        return out
    return sol.to(dtype=diff_sub.dtype)


class CrossExposureSkyCorrector:
    """Rectify overlapping exposures with an anchored weighted graph solve.

    Pairwise differences are measured only where both projected exposures have
    valid support. Observation zero is the explicit gauge anchor. ``correct``
    fits a smooth difference model (mesh, planar ``Ax + By + C``, or constant
    offset) for every overlap edge and solves the anchored graph, preserving
    structure shared by the exposures.
    """

    def __init__(
        self,
        estimator: Callable[..., Tensor] | None = None,
        *,
        mode: str = "mesh",
        mesh_size: int = 128,
        regularization: float = 1e-6,
    ) -> None:
        if mode not in ("mesh", "constant", "planar"):
            raise ValueError(
                f"unknown CrossExposureSkyCorrector mode {mode!r}; "
                "choose from 'mesh', 'constant', 'planar'"
            )
        self._estimator = estimator
        self.mode = mode
        self.mesh_size = int(mesh_size)
        self.regularization = float(regularization)

    def _get_estimator(self) -> Callable[..., Tensor]:
        if self._estimator is not None:
            return self._estimator
        if self.mode == "constant":
            return PolynomialSkyModel(order=0, statistic="huber")
        if self.mode == "planar":
            return PolynomialSkyModel(order=1, statistic="huber")
        return MeshSkyEstimator(block_size=self.mesh_size, statistic="median")

    def fit_offsets(
        self,
        images: list[Tensor],
        masks: list[Tensor | None] | None = None,
    ) -> Tensor:
        """Return overlap-derived additive offsets relative to observation zero."""
        if not images:
            return torch.empty(0)
        valid = _valid_pixels(images, masks)
        edges_bbox = _overlap_edges_with_bboxes(valid)
        edges = [(left, right, ov, c) for left, right, ov, c, _ in edges_bbox]
        _require_anchor_connected(len(images), edges)
        if len(images) == 1:
            return images[0].new_zeros(1)
        differences = torch.stack(
            [
                torch.median(
                    (
                        images[right][iy0:iy1, ix0:ix1]
                        - images[left][iy0:iy1, ix0:ix1]
                    )[ov[iy0:iy1, ix0:ix1]]
                )
                for left, right, ov, _, (iy0, iy1, ix0, ix1) in edges_bbox
            ]
        )
        return _solve_anchored_graph(len(images), edges, differences)

    def fit_planes(
        self,
        images: list[Tensor],
        masks: list[Tensor | None] | None = None,
    ) -> Tensor:
        """Return (N, 3) Chebyshev order-1 plane coefficients [c0, cx, cy] relative to obs 0."""
        if not images:
            return torch.empty((0, 3))
        valid = _valid_pixels(images, masks)
        edges_bbox = _overlap_edges_with_bboxes(valid)
        edges = [(left, right, ov, c) for left, right, ov, c, _ in edges_bbox]
        _require_anchor_connected(len(images), edges)
        if len(images) == 1:
            return images[0].new_zeros((1, 3))
        full_shape = (int(images[0].shape[0]), int(images[0].shape[1]))
        edge_coeffs = torch.stack(
            [
                _fit_planar_coefficients_on_bbox(
                    images[right][iy0:iy1, ix0:ix1] - images[left][iy0:iy1, ix0:ix1],
                    ov[iy0:iy1, ix0:ix1],
                    full_shape=full_shape,
                    bbox=(iy0, iy1, ix0, ix1),
                )
                for left, right, ov, _, (iy0, iy1, ix0, ix1) in edges_bbox
            ],
            dim=0,
        )
        return _solve_anchored_graph(len(images), edges, edge_coeffs)

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
        if self._estimator is None and self.mode == "constant":
            offsets = self.fit_offsets(images, masks)
            return [
                image - offsets[index].to(dtype=image.dtype)
                for index, image in enumerate(images)
            ]
        if self._estimator is None and self.mode == "planar":
            coeffs = self.fit_planes(images, masks)
            h, w = int(images[0].shape[0]), int(images[0].shape[1])
            out: list[Tensor] = []
            for index, image in enumerate(images):
                plane = CompactPolynomialSkyModel(
                    coefficients=coeffs[index].to(dtype=image.dtype),
                    image_shape=(h, w),
                    order=1,
                ).render_window(0, h, 0, w)
                out.append(image - plane)
            return out
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

    def fit_corrections(
        self,
        images: list[Tensor],
        masks: list[Tensor | None] | None = None,
    ) -> list[Tensor]:
        """Return additive 2D sky correction maps for each exposure."""
        if not images:
            return []
        corrected = self.correct(images, masks)
        return [img - corr for img, corr in zip(images, corrected)]
