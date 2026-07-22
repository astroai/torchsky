"""Bounded power-diagram primitives for experimental sky-field models.

These tensor-native kernels implement the geometry underlying Power Foam.  They
are representation primitives, not a complete renderer: exact rendering also
needs sparse candidate construction and pixel/cell intersection integration.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def _validate_power_inputs(points: Tensor, sites: Tensor, radii: Tensor) -> None:
    if not points.is_floating_point() or not sites.is_floating_point():
        raise TypeError("points and sites must be floating-point tensors")
    if points.ndim < 1 or sites.ndim != 2:
        raise ValueError("points must end in coordinates and sites must have shape (N, D)")
    if points.shape[-1] != sites.shape[-1]:
        raise ValueError("points and sites must have the same coordinate dimension")
    if sites.shape[0] == 0:
        raise ValueError("at least one power site is required")
    if radii.ndim != 1 or radii.shape[0] != sites.shape[0]:
        raise ValueError("radii must have shape (N,) matching sites")
    if points.device != sites.device or sites.device != radii.device:
        raise ValueError("points, sites, and radii must be on the same device")
    if points.dtype != sites.dtype or sites.dtype != radii.dtype:
        raise ValueError("points, sites, and radii must have the same dtype")
    if not bool(torch.isfinite(points).all()):
        raise ValueError("points must be finite")
    if not bool(torch.isfinite(sites).all()):
        raise ValueError("sites must be finite")
    if not bool(torch.isfinite(radii).all()) or bool(torch.any(radii < 0)):
        raise ValueError("radii must be finite and non-negative")


def power_distance(points: Tensor, sites: Tensor, radii: Tensor) -> Tensor:
    """Return ``||x - p_i||² - r_i²`` for every point/site pair.

    ``points`` has shape ``(..., D)``, ``sites`` has shape ``(N, D)``, and the
    result has shape ``(..., N)``.  Use :func:`bounded_power_ownership` for
    bounded-memory ownership queries over many points.
    """
    _validate_power_inputs(points, sites, radii)
    delta = points.unsqueeze(-2) - sites
    return torch.sum(delta * delta, dim=-1) - radii * radii


def bounded_power_ownership(
    points: Tensor,
    sites: Tensor,
    radii: Tensor,
    *,
    point_chunk_size: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Return exact bounded-power-cell ids and minimum power distances.

    Cell ``i`` is its ordinary power cell intersected with its radius-``r_i``
    ball.  Points outside the union of those balls receive owner ``-1``.  The
    optional chunk size bounds the temporary point-by-site distance matrix.
    """
    _validate_power_inputs(points, sites, radii)
    flat = points.reshape(-1, points.shape[-1])
    if flat.shape[0] == 0:
        shape = points.shape[:-1]
        return (
            torch.empty(shape, dtype=torch.long, device=points.device),
            torch.empty(shape, dtype=points.dtype, device=points.device),
        )
    chunk_size = flat.shape[0] if point_chunk_size is None else int(point_chunk_size)
    if chunk_size <= 0:
        raise ValueError("point_chunk_size must be positive")

    owner_parts: list[Tensor] = []
    distance_parts: list[Tensor] = []
    for start in range(0, flat.shape[0], chunk_size):
        distances = power_distance(flat[start : start + chunk_size], sites, radii)
        minimum, owners = torch.min(distances, dim=-1)
        owners = torch.where(minimum <= 0, owners, torch.full_like(owners, -1))
        owner_parts.append(owners)
        distance_parts.append(minimum)
    shape = points.shape[:-1]
    return torch.cat(owner_parts).reshape(shape), torch.cat(distance_parts).reshape(shape)


def soft_bounded_power_weights(
    points: Tensor,
    sites: Tensor,
    radii: Tensor,
    *,
    temperature: float = 10.0,
) -> Tensor:
    """Return a differentiable, compactly supported power-ownership surrogate.

    Weights are a softmax of negative power distance over balls containing each
    query.  They sum to one inside the union of bounds and to zero outside it.
    This is useful during optimization but is not an exact power-cell renderer.
    """
    _validate_power_inputs(points, sites, radii)
    if not math.isfinite(float(temperature)) or float(temperature) <= 0:
        raise ValueError("temperature must be finite and positive")
    distances = power_distance(points, sites, radii)
    support = distances <= 0
    logits = -float(temperature) * distances
    masked_logits = torch.where(support, logits, torch.full_like(logits, -torch.inf))
    any_support = support.any(dim=-1, keepdim=True)
    safe_logits = torch.where(any_support, masked_logits, torch.zeros_like(masked_logits))
    weights = torch.softmax(safe_logits, dim=-1)
    return torch.where(any_support, weights, torch.zeros_like(weights))


def cech_adjacency(
    sites: Tensor,
    radii: Tensor,
    *,
    site_chunk_size: int = 1024,
) -> Tensor:
    """Return undirected edges between all overlapping bounding balls.

    The result has shape ``(2, E)`` with each edge stored once as ``i < j``.
    This exact, bounded-memory reference still performs quadratic pair tests;
    survey-scale callers should use a spatial candidate index before this test.
    """
    _validate_power_inputs(sites[:1], sites, radii)
    chunk_size = int(site_chunk_size)
    if chunk_size <= 0:
        raise ValueError("site_chunk_size must be positive")
    edges: list[Tensor] = []
    count = sites.shape[0]
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        delta = sites[start:stop, None, :] - sites[None, :, :]
        distance_squared = torch.sum(delta * delta, dim=-1)
        radius_sum = radii[start:stop, None] + radii[None, :]
        local_i = torch.arange(start, stop, device=sites.device)[:, None]
        global_j = torch.arange(count, device=sites.device)[None, :]
        overlap = (local_i < global_j) & (distance_squared <= radius_sum * radius_sum)
        ij = torch.nonzero(overlap, as_tuple=False)
        if ij.numel() != 0:
            ij[:, 0] += start
            edges.append(ij)
    if not edges:
        return torch.empty((2, 0), dtype=torch.long, device=sites.device)
    return torch.cat(edges, dim=0).transpose(0, 1).contiguous()


def power_face_planes(
    sites: Tensor,
    radii: Tensor,
    edges: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return radical planes ``normal · x = offset`` for site-pair edges."""
    _validate_power_inputs(sites[:1], sites, radii)
    if edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype != torch.long:
        raise ValueError("edges must be a torch.long tensor with shape (2, E)")
    if edges.device != sites.device:
        raise ValueError("edges and sites must be on the same device")
    if edges.numel() and (
        bool(torch.any(edges < 0)) or bool(torch.any(edges >= sites.shape[0]))
    ):
        raise ValueError("edges contain an out-of-range site index")
    left, right = edges[0], edges[1]
    left_sites = sites[left]
    right_sites = sites[right]
    normals = right_sites - left_sites
    offsets = 0.5 * (
        torch.sum(right_sites * right_sites, dim=-1)
        - torch.sum(left_sites * left_sites, dim=-1)
        + radii[left] * radii[left]
        - radii[right] * radii[right]
    )
    return normals, offsets


__all__ = [
    "bounded_power_ownership",
    "cech_adjacency",
    "power_distance",
    "power_face_planes",
    "soft_bounded_power_weights",
]
