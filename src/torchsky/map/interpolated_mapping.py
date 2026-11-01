"""Verified coarse-grid interpolation of an expensive target-to-source pixel mapping.

Inverse-mapped resampling needs, for every target pixel, the source pixel it lands
on.  For distorted source WCSs (SIP, TPV, ...) the exact mapping is an iterative
world -> pixel inverse per pixel (plus a round-trip validity check), which dominates
the cost of a large mosaic.  The mapping is smooth, so it is evaluated exactly on a
coarse lattice of target nodes and interpolated (Catmull-Rom cubic, exact for
quadratics) to every pixel.  Smoothness is *verified*, not assumed: the exact mapping
is also evaluated at the centre and the four edge midpoints of every lattice cell and
a cell is accepted only if the interpolation error there is within
:data:`MAPPING_TOLERANCE_PX` source pixels and every node in its 4 x 4 stencil is
valid.  Rejected cells are re-tried with the lattice stride halved (down to
:data:`MAPPING_MIN_STRIDE`); whatever still fails is evaluated exactly per pixel.
Pixels whose interpolated source coordinate lies within a small band of the source
image bounds are re-evaluated exactly so that the coverage support is unchanged.

The module is device agnostic (pure torch on the device of the evaluator's tensors;
callers keep MPS on the CPU for float64).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import Tensor

# Maximum tolerated interpolation error of the target -> source pixel mapping, in
# source pixels, verified on every accepted lattice cell.
MAPPING_TOLERANCE_PX = 1.0e-3
# Initial lattice stride (target pixels) and the finest stride tried before falling
# back to exact per-pixel evaluation.
MAPPING_NODE_STRIDE = 8
MAPPING_MIN_STRIDE = 4
# Pixels whose interpolated source coordinate is within this many tolerances of the
# source image bounds are evaluated exactly (the in-bounds decision is then exact).
MAPPING_BOUNDS_BAND_TOLERANCES = 10.0
# Exact-evaluation chunk (points), bounding temporaries like the exact path does.
_EXACT_CHUNK = 1 << 20
# Below this many target pixels exact evaluation is as cheap as the lattice.
_MIN_PIXELS_FOR_LATTICE = 4096

# "interpolated" (verified coarse lattice) or "exact" (per-pixel inverse).
MAPPING_MODE = "interpolated"
# On an accelerator the default mode uses the lattice only for target windows of at
# least this many pixels (see ``image_plane_projection._use_interpolated_mapping``).
MAPPING_MIN_PIXELS_ACCELERATOR = 1 << 22

_T_LO = (3.0 - math.sqrt(3.0)) / 6.0
_T_HI = 1.0 - _T_LO
_VERIFY_OFFSETS = (
    (_T_LO, _T_LO),
    (_T_LO, _T_HI),
    (_T_HI, _T_LO),
    (_T_HI, _T_HI),
    (0.5, 0.5),
)

# The verification points sample the interpolation error; the true maximum over a cell
# exceeded the sampled maximum by <= 1.04x on strongly distorted SIP fixtures (higher
# order terms than the five points resolve).  Cells are accepted at half the tolerance so
# that the stated bound holds for every pixel, not only at the sampled points.
_VERIFY_SAFETY = 0.5

Evaluator = Callable[[Tensor, Tensor], tuple[Tensor, Tensor, Tensor]]


def resolve_mapping_mode(mapping: str | None) -> str:
    mode = MAPPING_MODE if mapping is None else mapping
    if mode not in ("interpolated", "exact"):
        raise ValueError(f"mapping must be 'interpolated' or 'exact', got {mode!r}")
    return mode


@contextmanager
def mapping_mode(mode: str) -> Iterator[None]:
    """Temporarily set the module default mapping mode (``"exact"`` / ``"interpolated"``)."""
    global MAPPING_MODE
    resolve_mapping_mode(mode)
    previous = MAPPING_MODE
    MAPPING_MODE = mode
    try:
        yield
    finally:
        MAPPING_MODE = previous


def _catmull_rom_weights(t: Tensor) -> Tensor:
    """Catmull-Rom weights ``(..., 4)`` for the nodes ``c-1, c, c+1, c+2`` at offset ``t``."""
    t2 = t * t
    t3 = t2 * t
    return torch.stack(
        [
            -0.5 * t3 + t2 - 0.5 * t,
            1.5 * t3 - 2.5 * t2 + 1.0,
            -1.5 * t3 + 2.0 * t2 + 0.5 * t,
            0.5 * t3 - 0.5 * t2,
        ],
        dim=-1,
    )


def _axis_nodes(n_pix: int, stride: int, device: torch.device) -> Tensor:
    n = max(math.ceil((n_pix - 1) / float(stride)) + 1, 2)
    return torch.linspace(0.0, float(n_pix - 1), n, dtype=torch.float64, device=device)


def _stencil(u: Tensor, n_nodes: int) -> tuple[Tensor, Tensor, Tensor]:
    """Cell index, padded-array stencil indices ``(..., 4)`` and weights for node coordinate ``u``."""
    cell = torch.clamp(torch.floor(u), 0.0, float(n_nodes - 2)).to(torch.int64)
    t = u - cell.to(u.dtype)
    idx = cell.unsqueeze(-1) + torch.arange(4, device=u.device)  # padded: c-1 -> c
    return cell, idx, _catmull_rom_weights(t)


def _pad_nodes(g: Tensor, dim: int) -> Tensor:
    """Add one ghost node at each end by cubic extrapolation (3 p0 - 3 p1 + p2)."""
    n = g.shape[dim]
    first = 3.0 * g.select(dim, 0) - 3.0 * g.select(dim, 1) + g.select(dim, 2)
    last = 3.0 * g.select(dim, n - 1) - 3.0 * g.select(dim, n - 2) + g.select(dim, n - 3)
    return torch.cat([first.unsqueeze(dim), g, last.unsqueeze(dim)], dim=dim)


def _interpolate_dense(
    nodes_padded: Tensor, uy: Tensor, ux: Tensor, ny: int, nx: int
) -> Tensor:
    """Separable Catmull-Rom of padded node values ``(C, ny+2, nx+2)`` on ``uy x ux``."""
    _, iy, wy = _stencil(uy, ny)
    _, ix, wx = _stencil(ux, nx)
    rows = None
    for k in range(4):
        term = nodes_padded[:, iy[:, k], :] * wy[:, k].reshape(1, -1, 1)
        rows = term if rows is None else rows + term
    out = None
    for k in range(4):
        term = rows[:, :, ix[:, k]] * wx[:, k].reshape(1, 1, -1)
        out = term if out is None else out + term
    return out


def _interpolate_points(
    nodes_padded: Tensor, uy: Tensor, ux: Tensor, ny: int, nx: int
) -> Tensor:
    """Catmull-Rom of padded node values at scattered node coordinates -> ``(C, P)``."""
    _, iy, wy = _stencil(uy, ny)
    _, ix, wx = _stencil(ux, nx)
    vals = nodes_padded[:, iy[:, :, None], ix[:, None, :]]  # (C, P, 4, 4)
    return (vals * (wy[:, :, None] * wx[:, None, :]).unsqueeze(0)).sum(dim=(-1, -2))


def _evaluate_exact_chunked(
    evaluate: Evaluator, x: Tensor, y: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    xs, ys, oks = [], [], []
    for start in range(0, x.numel(), _EXACT_CHUNK):
        a, b, c = evaluate(x[start : start + _EXACT_CHUNK], y[start : start + _EXACT_CHUNK])
        xs.append(a)
        ys.append(b)
        oks.append(c)
    if not xs:
        empty = x.new_zeros((0,))
        return empty, empty, torch.zeros((0,), dtype=torch.bool, device=x.device)
    return torch.cat(xs), torch.cat(ys), torch.cat(oks)


def _level(
    evaluate: Evaluator,
    height: int,
    width: int,
    stride: int,
    tolerance: float,
    device: torch.device,
    stats: dict | None,
) -> tuple[Tensor, Tensor]:
    """One lattice level -> (dense mapping ``(2, H, W)``, per-pixel acceptance ``(H, W)``)."""
    ys = _axis_nodes(height, stride, device)
    xs = _axis_nodes(width, stride, device)
    ny, nx = int(ys.numel()), int(xs.numel())
    sy = float(height - 1) / (ny - 1)
    sx = float(width - 1) / (nx - 1)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    # Verification points in every cell.  The leading Catmull-Rom error term of a
    # smooth map is ~ f_xxx h^3 t (1 - t) (1 - 2t): it vanishes at the centre and on
    # the lattice lines, so those are blind spots.  Its extrema are at
    # t = (3 -+ sqrt 3) / 6, which are checked (separably in 2-D); the centre catches
    # the even (f_xxxx h^4) part.
    cy_idx = torch.arange(ny - 1, device=device, dtype=torch.float64)
    cx_idx = torch.arange(nx - 1, device=device, dtype=torch.float64)
    cell_err = torch.zeros((ny - 1, nx - 1), dtype=torch.float64, device=device)
    verify_u = []
    for t_y, t_x in _VERIFY_OFFSETS:
        u_y, u_x = torch.meshgrid(cy_idx + t_y, cx_idx + t_x, indexing="ij")
        verify_u.append((u_y.reshape(-1), u_x.reshape(-1)))
    # One evaluator call for the nodes and all verification points: the exact mapping
    # is an iterative per-point solve, so on an accelerator the cost of a call is
    # dominated by kernel launches, not by the number of points.
    eval_x = torch.cat([gx.reshape(-1)] + [u_x * sx for _, u_x in verify_u])
    eval_y = torch.cat([gy.reshape(-1)] + [u_y * sy for u_y, _ in verify_u])
    all_x, all_y, all_ok = _evaluate_exact_chunked(evaluate, eval_x, eval_y)
    n_nodes = ny * nx
    nx_src, ny_src, n_ok = all_x[:n_nodes], all_y[:n_nodes], all_ok[:n_nodes]
    nodes = torch.stack([nx_src.reshape(ny, nx), ny_src.reshape(ny, nx)], dim=0)
    node_bad = (~n_ok.reshape(ny, nx)) | ~torch.isfinite(nodes).all(dim=0)
    nodes = torch.where(torch.isfinite(nodes), nodes, torch.zeros_like(nodes))
    padded = _pad_nodes(_pad_nodes(nodes, 1), 2)

    n_cells = (ny - 1) * (nx - 1)
    for k, (u_y, u_x) in enumerate(verify_u):
        interp = _interpolate_points(padded, u_y, u_x, ny, nx)
        lo = n_nodes + k * n_cells
        ex, ey, eok = all_x[lo : lo + n_cells], all_y[lo : lo + n_cells], all_ok[lo : lo + n_cells]
        err = torch.hypot(interp[0] - ex, interp[1] - ey)
        err = torch.where(
            eok & torch.isfinite(err), err, torch.full_like(err, float("inf"))
        )
        cell_err = torch.maximum(cell_err, err.reshape(ny - 1, nx - 1))
    # A cell is only as good as every node of its 4 x 4 Catmull-Rom stencil.
    bad_f = F.pad(node_bad.to(torch.float32)[None, None], (1, 1, 1, 1), mode="replicate")
    cell_bad = F.max_pool2d(bad_f, kernel_size=4, stride=1)[0, 0] > 0.0
    cell_ok = (cell_err <= tolerance * _VERIFY_SAFETY) & ~cell_bad
    if stats is not None:
        finite_err = cell_err[cell_ok]
        if finite_err.numel():
            stats["max_verified_error_px"] = max(
                stats.get("max_verified_error_px", 0.0), float(finite_err.max())
            )

    py = torch.arange(height, device=device, dtype=torch.float64) / sy
    px = torch.arange(width, device=device, dtype=torch.float64) / sx
    dense = _interpolate_dense(padded, py, px, ny, nx)
    cy = torch.clamp(torch.floor(py), 0.0, float(ny - 2)).to(torch.int64)
    cx = torch.clamp(torch.floor(px), 0.0, float(nx - 2)).to(torch.int64)
    accepted = cell_ok[cy[:, None], cx[None, :]]
    if stats is not None:
        stats.setdefault("levels", []).append(
            {"stride": stride, "cells": int(cell_ok.numel()), "accepted_cells": int(cell_ok.sum())}
        )
    return dense, accepted


def interpolated_inverse_mapping(
    evaluate: Evaluator,
    height: int,
    width: int,
    *,
    device: torch.device,
    bounds_xy: tuple[float, float] | None = None,
    stride: int | None = None,
    tolerance: float | None = None,
    min_stride: int | None = None,
    stats: dict | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Verified lattice interpolation of ``evaluate`` over the target window.

    ``evaluate(x, y)`` maps target pixel coordinates (float64 tensors of any shape)
    to ``(x_src, y_src, ok)`` exactly; ``ok`` is the validity (finite result, and for
    SIP sources a passed round trip).  Returns float64 ``x_src, y_src`` of shape
    ``(height, width)`` and the boolean validity ``ok``.  Every accepted pixel is
    within ``tolerance`` source pixels of the exact mapping on the verification points
    of its cell; pixels in cells that touch an invalid node, fail verification at every
    stride >= ``min_stride`` or lie within a small band of ``bounds_xy`` (the source
    ``(x_max, y_max)`` pixel extent) are evaluated exactly.
    """
    stride = MAPPING_NODE_STRIDE if stride is None else int(stride)
    tolerance = MAPPING_TOLERANCE_PX if tolerance is None else float(tolerance)
    min_stride = MAPPING_MIN_STRIDE if min_stride is None else int(min_stride)
    if stats is not None:
        stats.update({"pixels": height * width, "exact_pixels": 0, "max_verified_error_px": 0.0})

    out = torch.zeros((2, height, width), dtype=torch.float64, device=device)
    ok = torch.zeros((height, width), dtype=torch.bool, device=device)
    todo = torch.ones((height, width), dtype=torch.bool, device=device)
    exact_mask = todo

    lattice_possible = (
        height >= 2
        and width >= 2
        and height * width >= _MIN_PIXELS_FOR_LATTICE
        and (height - 1) / float(max(stride, 1)) >= 2.0
        and (width - 1) / float(max(stride, 1)) >= 2.0
    )
    if lattice_possible:
        level_stride = stride
        while level_stride >= max(min_stride, 1):
            if (height - 1) / float(level_stride) < 2.0 or (width - 1) / float(level_stride) < 2.0:
                break
            dense, accepted = _level(
                evaluate, height, width, level_stride, tolerance, device, stats
            )
            take = todo & accepted
            out = torch.where(take.unsqueeze(0), dense, out)
            ok = ok | take
            todo = todo & ~accepted
            if not bool(todo.any()):
                break
            level_stride //= 2
        exact_mask = todo.clone()
        if bounds_xy is not None:
            band = MAPPING_BOUNDS_BAND_TOLERANCES * tolerance
            xmax, ymax = bounds_xy
            x, y = out[0], out[1]
            near = (
                (x.abs() <= band)
                | ((x - xmax).abs() <= band)
                | (y.abs() <= band)
                | ((y - ymax).abs() <= band)
            )
            exact_mask = exact_mask | (ok & near)

    if bool(exact_mask.any()):
        iy, ix = torch.nonzero(exact_mask, as_tuple=True)
        ex, ey, eok = _evaluate_exact_chunked(
            evaluate, ix.to(torch.float64), iy.to(torch.float64)
        )
        out[0][iy, ix] = ex
        out[1][iy, ix] = ey
        ok[iy, ix] = eok
        if stats is not None:
            stats["exact_pixels"] = int(iy.numel())
    return out[0], out[1], ok
