"""Batched anisotropic Gaussian items rasterised onto regular tangent-plane grids.

A sky made of many Gaussian "items" (continuous position, two widths, a position
angle, and ``K`` linear coefficients each) is rasterised onto ``K`` coefficient
planes.  The pixel kernel is private and WCS-free (:func:`_rasterize_gaussians`);
the public class :class:`GaussianItemScene` holds the item parameters in sky units
and maps them to pixel centres and pixel covariances through a WCS.

Pixel convention (as :mod:`torchsky.map.splat`): 0-indexed pixel centres, ``x``
along columns, ``y`` along rows; a grid of shape ``(H, W)`` has nodes
``x = 0..W-1`` and ``y = 0..H-1``.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd.function import once_differentiable

from ..device import _to
from ..map.scenes import SceneRepresentation
from ..map.target import TargetRenderSpec
from ..wcs import WCS

# Patch half-size, in units of the larger principal width of the sampled covariance.
# The fraction of a 2-d Gaussian's flux beyond 4 widths is exp(-8) = 3.4e-4.
_TRUNCATION_SIGMA = 4.0
# Patch half-size buckets [nodes]: items sharing a bucket are processed as one batch.
# Above the table the half-size is rounded up to a multiple of ``_BUCKET_STEP``.
_HALF_BUCKETS = (2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128)
_BUCKET_STEP = 32
# Elements (items x patch nodes x K) per rasterisation chunk.
_DEFAULT_CHUNK_ELEMENTS = 4_000_000
# Variance of a unit pixel box along one axis: the cell integral of a Gaussian is
# approximated by the point sample of the Gaussian whose covariance is Sigma + I / 12.
_PIXEL_VARIANCE = 1.0 / 12.0
# Pyramid: the cascade from level l to l - 1 blurs with s_l = _PYR_BLUR * pitch_l, an
# item goes to the coarsest level whose remaining smaller principal variance is at least
# (_PYR_MIN * pitch_l)^2, and the cascade kernel reaches _PYR_TAP * s_l (truncating it
# changes its variance by 1.5e-5 relative).
_PYR_BLUR = 1.0
_PYR_MIN = 1.0
_PYR_TAP = 5.0
_MODES = ("direct", "pyramid")


# ------------------------------------------------------------------------ validation
def _check_shape(shape: Any) -> tuple[int, int]:
    try:
        height, width = (int(v) for v in shape)
        n_axes = len(tuple(shape))
    except (TypeError, ValueError) as exc:
        raise ValueError("shape must be two positive integers (H, W)") from exc
    if n_axes != 2 or height <= 0 or width <= 0:
        raise ValueError("shape must be two positive integers (H, W)")
    return height, width


def _check_chunk(chunk_elements: Any) -> int:
    if isinstance(chunk_elements, bool) or int(chunk_elements) != chunk_elements:
        raise ValueError("chunk_elements must be a positive integer")
    if int(chunk_elements) <= 0:
        raise ValueError("chunk_elements must be a positive integer")
    return int(chunk_elements)


def _covariance_entries(shape_pix: Tensor) -> Tensor:
    """Validate item shapes and return ``(C, 3)`` rows ``[Sxx, Syy, Sxy]``."""
    if not isinstance(shape_pix, Tensor) or not shape_pix.is_floating_point():
        raise ValueError("shape_pix must be a floating tensor")
    if not bool(torch.isfinite(shape_pix).all()):
        raise ValueError("shape_pix must be finite")
    if shape_pix.ndim == 3 and tuple(shape_pix.shape[1:]) == (2, 2):
        sxx, syy = shape_pix[:, 0, 0], shape_pix[:, 1, 1]
        sxy = 0.5 * (shape_pix[:, 0, 1] + shape_pix[:, 1, 0])
        asym = (shape_pix[:, 0, 1] - shape_pix[:, 1, 0]).abs()
        scale = (sxx.abs() + syy.abs()).clamp_min(torch.finfo(shape_pix.dtype).tiny)
        if bool((asym > 1e-6 * scale).any()):
            raise ValueError("covariance_pix must be symmetric")
        cov = torch.stack((sxx, syy, sxy), dim=-1)
        tr = sxx + syy
        lam_min = 0.5 * tr - torch.sqrt((0.5 * (sxx - syy)).square() + sxy.square())
        if bool((tr <= 0).any()) or bool((lam_min <= 0).any()):
            raise ValueError("covariance_pix must be positive definite")
        return cov
    if shape_pix.ndim == 2 and shape_pix.shape[1] == 3:
        major, minor, angle = shape_pix.unbind(-1)
        if bool((major <= 0).any()) or bool((minor <= 0).any()):
            raise ValueError("major and minor widths must be positive")
        if bool((minor > major).any()):
            raise ValueError("minor width cannot exceed major width")
        c, s = torch.cos(angle), torch.sin(angle)
        a2, b2 = major.square(), minor.square()
        return torch.stack(
            (c * c * a2 + s * s * b2, s * s * a2 + c * c * b2, c * s * (a2 - b2)), -1
        )
    raise ValueError(
        "shape_pix must have shape (C, 2, 2) (covariance) or (C, 3) "
        "(major, minor, angle in radians)"
    )


# ---------------------------------------------------------------------------- levels
@dataclass(frozen=True)
class _Level:
    """One regular grid of the rasteriser: node ``(j, i)`` is at ``org + pitch * (i, j)``."""

    ny: int
    nx: int
    pitch: float
    org: float
    qoff: float  # variance added to Sigma before point sampling (pixel term minus cascade blur)
    margin: int = 0  # nodes beyond the fine grid kept on every side
    radius: int = 0  # taps either side of the cascade kernel to the next finer level
    sigma: float = 0.0  # cascade blur width to the next finer level
    threshold: float = (
        0.0  # smaller principal variance of Sigma + q needed to be sent here
    )


def _pyramid_levels(height: int, width: int, q: float, tmin_max: float) -> list[_Level]:
    """Level 0 plus every coarse level some item can reach (``tmin_max`` = largest T_min).

    Going from level l to l - 1 every level-l node deposits a discretely normalised
    Gaussian of width ``s_l = _PYR_BLUR * pitch_l`` (separable, stride 2).  Level l has a
    margin of ``E_l = ceil(reach_l / pitch_l) + 1`` nodes beyond the fine grid, where
    ``reach_l`` is the farthest a level-l node can deposit on level 0, so that everything
    that can reach the fine grid is kept and the result equals that of unbounded grids.
    """
    levels = [_Level(height, width, 1.0, 0.0, q)]
    blur2 = 0.0
    reach = 0.0
    level = 0
    while True:
        level += 1
        pitch = 2.0**level
        sigma = _PYR_BLUR * pitch
        blur2 += sigma * sigma
        threshold = (_PYR_MIN * pitch) ** 2 + blur2
        if threshold > tmin_max:
            break
        radius = math.ceil(_PYR_TAP * sigma / (pitch / 2))
        reach += radius * pitch / 2
        margin = math.ceil(reach / pitch) + 1
        levels.append(
            _Level(
                ny=math.ceil((height - 1) / pitch) + 2 * margin + 1,
                nx=math.ceil((width - 1) / pitch) + 2 * margin + 1,
                pitch=pitch,
                org=-margin * pitch,
                qoff=q - blur2,
                margin=margin,
                radius=radius,
                sigma=sigma,
                threshold=threshold,
            )
        )
    return levels


def _upsample(grid: Tensor, levels: list[_Level], level: int) -> Tensor:
    """Level-l density ``(K, ny_l, nx_l)`` -> its deposit on level l - 1 (ordinary autograd).

    1-d weights ``2 g_t / sum g`` make the 2-d weights sum to 4, the ratio of the node
    areas, so the flux ``pitch_l^2 * density`` is conserved.
    """
    lv, prev = levels[level], levels[level - 1]
    taps = torch.arange(-lv.radius, lv.radius + 1, dtype=torch.float64) * (lv.pitch / 2)
    kernel = torch.exp(-0.5 * (taps / lv.sigma) ** 2)
    kernel = _to(2.0 * kernel / kernel.sum(), grid.device, grid.dtype)
    t = kernel.numel()
    out = F.conv_transpose2d(grid[:, None], kernel.view(1, 1, 1, t), stride=(1, 2))
    out = F.conv_transpose2d(out, kernel.view(1, 1, t, 1), stride=(2, 1))[:, 0]
    # Output index o is level-(l-1) node o + c - r, c = E_{l-1} - 2 E_l, r = lv.radius.
    start = lv.radius - (prev.margin - 2 * lv.margin)
    return F.pad(
        out,
        (
            -start,
            start + prev.nx - out.shape[-1],
            -start,
            start + prev.ny - out.shape[-2],
        ),
    )


# --------------------------------------------------------------------------- patches
def _principal_variances(cov: Tensor, offset: float) -> tuple[Tensor, Tensor]:
    """Larger and smaller eigenvalue of ``Sigma + offset * I`` for ``(C, 3)`` rows (float64)."""
    c = cov.detach().double()
    mean = 0.5 * (c[:, 0] + c[:, 1]) + offset
    spread = torch.sqrt((0.5 * (c[:, 0] - c[:, 1])).square() + c[:, 2].square())
    return mean + spread, mean - spread


def _patch_half_sizes(cov: Tensor, level: _Level, truncation_sigma: float) -> Tensor:
    """Patch half-size [nodes of ``level``] of each item: bucketed ``ceil(R sqrt(lambda_max) / pitch)``."""
    lam_max, _ = _principal_variances(cov, level.qoff)
    need = (
        torch.ceil(truncation_sigma * torch.sqrt(lam_max.clamp_min(0.0)) / level.pitch)
        .long()
        .clamp_min(1)
    )
    table = torch.tensor(_HALF_BUCKETS, device=need.device)
    bucketed = table[torch.searchsorted(table, need.clamp(max=_HALF_BUCKETS[-1]))]
    stepped = ((need + _BUCKET_STEP - 1) // _BUCKET_STEP) * _BUCKET_STEP
    return torch.where(need > _HALF_BUCKETS[-1], stepped, bucketed)


def _chunks(
    pos: Tensor,
    cov: Tensor,
    k: int,
    level: _Level,
    chunk_elements: int,
    truncation_sigma: float,
) -> Iterator[tuple[int, Tensor]]:
    """Batches ``(half, item indices)`` sharing a patch size, at most ``chunk_elements`` elements.

    Items whose whole patch lies off the grid contribute exactly nothing and are skipped.
    A single item larger than ``chunk_elements`` forms a chunk of its own.
    """
    half = _patch_half_sizes(cov, level, truncation_sigma)
    pd = pos.detach().double()
    jx = torch.round((pd[:, 0] - level.org) / level.pitch).long()
    jy = torch.round((pd[:, 1] - level.org) / level.pitch).long()
    inside = (
        (jx + half >= 0) & (jx - half <= level.nx - 1)
        & (jy + half >= 0) & (jy - half <= level.ny - 1)
    )  # fmt: skip
    half = torch.where(inside, half, torch.zeros_like(half))
    for h in torch.unique(half).tolist():
        if h == 0:
            continue
        idx = torch.nonzero(half == h).squeeze(1)
        per_item = (2 * h + 1) ** 2 * max(k, 1)
        step = max(1, chunk_elements // per_item)
        for start in range(0, idx.numel(), step):
            yield h, idx[start : start + step]


def _patch(
    pos: Tensor, cov: Tensor, half: int, level: _Level, dtype: torch.dtype
) -> tuple[Tensor, Tensor]:
    """Normalised patches ``g`` ``(m, P*P)`` of items on +-half nodes and flat node indices.

    ``g`` is the Gaussian with covariance ``Sigma + qoff * I`` point-sampled at the nodes and
    normalised so that ``pitch^2 * sum_patch g = 1`` over the WHOLE patch (nodes off the
    grid included): the flux is exact, and what falls off the grid is dropped.  The index
    of a node off the grid is ``ny * nx`` (the sink row).
    """
    x, y = pos[:, 0].double(), pos[:, 1].double()
    txx, tyy = cov[:, 0].double() + level.qoff, cov[:, 1].double() + level.qoff
    txy = cov[:, 2].double()
    det = txx * tyy - txy * txy
    ixx, iyy, ixy = (
        v.to(dtype)[:, None, None] for v in (tyy / det, txx / det, -txy / det)
    )
    off = torch.arange(-half, half + 1, device=pos.device)
    jx = torch.round((x.detach() - level.org) / level.pitch).long()[:, None] + off
    jy = torch.round((y.detach() - level.org) / level.pitch).long()[:, None] + off
    dx = ((level.org + jx.double() * level.pitch) - x[:, None]).to(dtype)[:, None, :]
    dy = ((level.org + jy.double() * level.pitch) - y[:, None]).to(dtype)[:, :, None]
    arg = 0.5 * (ixx * dx * dx + iyy * dy * dy) + ixy * dx * dy  # (m, Py, Px)
    arg = arg - arg.detach().amin(
        (1, 2), keepdim=True
    )  # exact: cancels in the normalisation
    g = torch.exp(-arg)
    g = g / (g.sum((1, 2), keepdim=True) * (level.pitch * level.pitch))
    okx = (jx >= 0) & (jx < level.nx)
    oky = (jy >= 0) & (jy < level.ny)
    flat = torch.where(
        oky[:, :, None] & okx[:, None, :],
        jy[:, :, None] * level.nx + jx[:, None, :],
        torch.full_like(jy[:, :, None] * jx[:, None, :], level.ny * level.nx),
    )
    m = g.shape[0]
    return g.reshape(m, -1), flat.reshape(m, -1)


def _scatter_forward(
    pos: Tensor,
    cov: Tensor,
    coef: Tensor,
    level: _Level,
    chunk_elements: int,
    truncation_sigma: float,
) -> Tensor:
    """``(ny * nx + 1, K)`` density of every item (last row: sink). No graph is kept."""
    k = coef.shape[1]
    out = torch.zeros(level.ny * level.nx + 1, k, device=pos.device, dtype=coef.dtype)
    for half, idx in _chunks(pos, cov, k, level, chunk_elements, truncation_sigma):
        g, flat = _patch(pos[idx], cov[idx], half, level, coef.dtype)
        out.index_add_(
            0, flat.reshape(-1), (g[:, :, None] * coef[idx, None, :]).reshape(-1, k)
        )
    return out


def _scatter_vjp(
    pos: Tensor,
    cov: Tensor,
    coef: Tensor,
    grad_out: Tensor,
    level: _Level,
    chunk_elements: int,
    truncation_sigma: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Gradient of ``sum(grad_out * raster)`` with respect to ``(pos, cov, coef)``.

    Each chunk's patches are recomputed with a graph, differentiated and freed, so the
    memory held is one chunk (``chunk_elements``), not items x patch nodes x K.
    """
    grads = [torch.zeros_like(t) for t in (pos, cov, coef)]
    k = coef.shape[1]
    for half, idx in _chunks(pos, cov, k, level, chunk_elements, truncation_sigma):
        with torch.enable_grad():
            leaves = [t[idx].detach().requires_grad_(True) for t in (pos, cov, coef)]
            g, flat = _patch(leaves[0], leaves[1], half, level, coef.dtype)
            weight = (grad_out[flat] * leaves[2][:, None, :]).sum(-1)
            found = torch.autograd.grad((g * weight).sum(), leaves)
        for grad, part in zip(grads, found, strict=True):
            grad[idx] = part
    return grads[0], grads[1], grads[2]


class _Scatter(torch.autograd.Function):
    """Item scatter on one level with a recompute-in-chunks backward (first order only)."""

    @staticmethod
    def forward(ctx, pos, cov, coef, level, chunk_elements, truncation_sigma):  # type: ignore[override]
        ctx.config = (level, chunk_elements, truncation_sigma)
        ctx.save_for_backward(pos, cov, coef)
        return _scatter_forward(pos, cov, coef, level, chunk_elements, truncation_sigma)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):  # type: ignore[override]
        pos, cov, coef = ctx.saved_tensors
        level, chunk_elements, truncation_sigma = ctx.config
        grads = _scatter_vjp(
            pos, cov, coef, grad_out, level, chunk_elements, truncation_sigma
        )
        return (*grads, None, None, None)


def _scatter_level(
    pos: Tensor,
    cov: Tensor,
    coef: Tensor,
    level: _Level,
    chunk_elements: int,
    truncation_sigma: float,
) -> Tensor:
    out = _Scatter.apply(pos, cov, coef, level, chunk_elements, truncation_sigma)
    return out[:-1].T.reshape(coef.shape[1], level.ny, level.nx)


# ---------------------------------------------------------------------------- kernel
def _rasterize_gaussians(
    centers_pix: Tensor,
    shape_pix: Tensor,
    coefficients: Tensor,
    shape: tuple[int, int],
    *,
    mode: str = "direct",
    pixel_integration: bool = True,
    chunk_elements: int = _DEFAULT_CHUNK_ELEMENTS,
    truncation_sigma: float = _TRUNCATION_SIGMA,
) -> Tensor:
    """Rasterise Gaussian items onto ``K`` planes of a regular ``(H, W)`` pixel grid.

    Item ``n`` contributes ``coefficients[n, k] * g_n(node)`` to plane ``k``, where ``g_n``
    is a Gaussian density per pixel (``sum_nodes g_n = 1`` for an item inside the grid), so
    the flux of item ``n`` in plane ``k`` is its coefficient.  Flux falling off the grid is
    dropped, not renormalised.

    Args:
        centers_pix: ``(C, 2)`` item centres ``(x, y)``, 0-indexed pixel centres (``x`` along
            columns, ``y`` along rows).
        shape_pix: ``(C, 2, 2)`` covariance in pixels squared (symmetric positive definite),
            or ``(C, 3)`` rows ``(major_sigma, minor_sigma, angle)`` in pixels and radians,
            with the major axis along ``(cos angle, sin angle)`` in ``(x, y)``.
        coefficients: ``(C, K)`` coefficients (fluxes); sets the output dtype.
        shape: ``(H, W)`` of the grid.
        mode: ``"direct"`` samples every item on the fine grid on a patch of
            ``+-truncation_sigma`` widths.  ``"pyramid"`` samples items wider than about
            2.8 nodes on coarser grids with the covariance left after a fixed Gaussian
            cascade, so the cost per item does not grow with its size.
        pixel_integration: if true the pixel cell integral is approximated by adding
            ``1/12`` pixel^2 to the covariance before point sampling; if false the item is
            point sampled with its own covariance.
        chunk_elements: elements (items x patch nodes x K) held per chunk, in the forward
            and, recomputed, in the backward pass.
        truncation_sigma: patch half-size in larger principal widths.

    Returns:
        ``(K, H, W)`` tensor in ``coefficients.dtype``, differentiable (first order) in
        ``centers_pix``, ``shape_pix`` and ``coefficients``.

    Raises:
        ValueError: on malformed shapes, non-finite values, non-positive or ordered-wrongly
            widths, a non-positive-definite covariance, an unknown mode or a non-positive
            ``chunk_elements`` / ``truncation_sigma``.
    """
    height, width = _check_shape(shape)
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
    chunk_elements = _check_chunk(chunk_elements)
    if not math.isfinite(truncation_sigma) or truncation_sigma <= 0:
        raise ValueError("truncation_sigma must be positive and finite")
    if not isinstance(centers_pix, Tensor) or not centers_pix.is_floating_point():
        raise ValueError("centers_pix must be a floating tensor")
    if centers_pix.ndim != 2 or centers_pix.shape[1] != 2:
        raise ValueError("centers_pix must have shape (C, 2)")
    if (
        not isinstance(coefficients, Tensor)
        or not coefficients.is_floating_point()
        or coefficients.ndim != 2
        or coefficients.shape[1] < 1
    ):
        raise ValueError(
            "coefficients must be a floating tensor of shape (C, K), K >= 1"
        )
    n_items, k = coefficients.shape
    if centers_pix.shape[0] != n_items or shape_pix.shape[0] != n_items:
        raise ValueError("centers_pix, shape_pix and coefficients must share C items")
    if not bool(torch.isfinite(centers_pix).all()) or not bool(
        torch.isfinite(coefficients).all()
    ):
        raise ValueError("centers_pix and coefficients must be finite")
    if not (centers_pix.device == shape_pix.device == coefficients.device):
        raise ValueError("inputs must be on one device")
    cov = _covariance_entries(shape_pix)  # (C, 3)
    dtype = coefficients.dtype
    q = _PIXEL_VARIANCE if pixel_integration else 0.0
    zero = torch.zeros(k, height, width, device=coefficients.device, dtype=dtype)
    if n_items == 0:
        return zero

    tmin = _principal_variances(cov, q)[1]
    levels = (
        _pyramid_levels(height, width, q, float(tmin.max()))
        if mode == "pyramid"
        else [_Level(height, width, 1.0, 0.0, q)]
    )
    if len(levels) > 1:
        thresholds = torch.tensor(
            [lv.threshold for lv in levels[1:]], dtype=torch.float64, device=tmin.device
        )
        assigned = torch.searchsorted(thresholds, tmin, right=True)
    else:
        assigned = torch.zeros(n_items, dtype=torch.long, device=tmin.device)

    total: Tensor | None = None
    for lev in range(len(levels) - 1, -1, -1):
        idx = torch.nonzero(assigned == lev).squeeze(1)
        if idx.numel():
            plane = _scatter_level(
                centers_pix[idx],
                cov[idx],
                coefficients[idx],
                levels[lev],
                chunk_elements,
                truncation_sigma,
            )
            total = plane if total is None else total + plane
        if lev > 0 and total is not None:
            total = _upsample(total, levels, lev)
    if total is None:  # every item is off the grid: keep the graph connected
        return zero + 0.0 * (centers_pix.sum() + cov.sum() + coefficients.sum())
    return total


# ------------------------------------------------------------------------ the scene
def _as_float_rows(value: Any, name: str, ndim: int) -> Tensor:
    tensor = torch.as_tensor(value)
    if not tensor.is_floating_point() or tensor.is_complex():
        raise TypeError(f"{name} must be a real floating tensor")
    if tensor.ndim != ndim:
        raise ValueError(
            f"{name} must have {ndim} dimension(s), got shape {tuple(tensor.shape)}"
        )
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be finite")
    return tensor


class GaussianItemScene(SceneRepresentation, nn.Module):
    """A sky of anisotropic Gaussian items with linear coefficients, rasterised in batch.

    Item ``n`` is a Gaussian at ``(lon_n, lat_n)`` with principal widths
    ``(major_n, minor_n)`` and position angle ``angle_n`` (east of north: the major axis
    lies along ``(sin a, cos a)`` in ``(east, north)``), with ``K`` coefficients, the
    conventions of :class:`~torchsky.scene.SourceCatalogScene` Gaussian components.
    Rendering produces ``K`` coefficient planes on a tangent-plane target: plane ``k``
    holds, in each pixel, ``sum_n coefficients[n, k] * (fraction of item n in the pixel)``,
    i.e. flux per pixel, so the sum of a plane is the total coefficient of the items inside
    the grid.  Flux falling off the grid is dropped.  The scene does not apply a PSF.

    All item parameters are :class:`torch.nn.Parameter` and differentiable in float32 or
    float64 on any device; the WCS geometry is evaluated in float64.  Use float64 for
    ``lon`` / ``lat`` where sub-pixel positions matter (float32 resolves about 0.05 arcsec
    at 180 degrees).

    Args:
        lon: ``(C,)`` longitudes [degrees].
        lat: ``(C,)`` latitudes [degrees], in ``[-90, 90]``.
        gaussian_parameters: ``(C, 3)`` rows ``[major_sigma_arcsec, minor_sigma_arcsec,
            angle_deg]`` with ``0 < minor <= major``.
        coefficients: ``(C, K)`` coefficients (fluxes per plane).
        min_sigma_arcsec: optional width floor [arcsec]; each width is replaced by
            ``sqrt(floor**2 + sigma**2)`` when rendering.  ``None`` means no floor.
        mode: ``"direct"`` or ``"pyramid"`` (see :func:`_rasterize_gaussians`).
        pixel_integration: approximate the pixel cell integral by adding ``1/12``
            pixel^2 to the covariance (default); ``False`` point-samples the Gaussian.
        chunk_elements: elements (items x patch nodes x K) held per chunk.
        truncation_sigma: patch half-size in larger principal widths.

    Raises:
        ValueError: on malformed or non-finite parameters, a latitude outside
            ``[-90, 90]``, non-positive widths or ``minor > major``.
        TypeError: if the parameters are not real floating tensors.

    Examples:
        >>> import torch
        >>> from torchsky.scene.gaussians import GaussianItemScene
        >>> header = {"NAXIS": 2, "NAXIS1": 16, "NAXIS2": 12, "CTYPE1": "RA---TAN",
        ...           "CTYPE2": "DEC--TAN", "CRPIX1": 8.5, "CRPIX2": 6.5, "CRVAL1": 10.0,
        ...           "CRVAL2": 0.0, "CDELT1": -1 / 3600, "CDELT2": 1 / 3600}
        >>> scene = GaussianItemScene(
        ...     torch.tensor([10.0], dtype=torch.float64),
        ...     torch.tensor([0.0], dtype=torch.float64),
        ...     torch.tensor([[1.0, 0.5, 30.0]], dtype=torch.float64),
        ...     torch.tensor([[5.0, 1.0]], dtype=torch.float64),
        ... )
        >>> planes = scene.render_coefficients(header)
        >>> tuple(planes.shape)
        (2, 12, 16)
        >>> bool(torch.allclose(planes.sum((1, 2)), torch.tensor([5.0, 1.0], dtype=torch.float64), atol=1e-6))
        True
    """

    def __init__(
        self,
        lon: Tensor,
        lat: Tensor,
        gaussian_parameters: Tensor,
        coefficients: Tensor,
        *,
        min_sigma_arcsec: float | None = None,
        mode: str = "direct",
        pixel_integration: bool = True,
        chunk_elements: int = _DEFAULT_CHUNK_ELEMENTS,
        truncation_sigma: float = _TRUNCATION_SIGMA,
    ) -> None:
        super().__init__()
        lon_t = _as_float_rows(lon, "lon", 1)
        lat_t = _to(_as_float_rows(lat, "lat", 1), lon_t.device, lon_t.dtype)
        count = int(lon_t.numel())
        if lat_t.numel() != count:
            raise ValueError("lon and lat must have the same number of items")
        if bool((lat_t.abs() > 90.0).any()):
            raise ValueError("lat must be in [-90, 90] degrees")
        gauss = _as_float_rows(gaussian_parameters, "gaussian_parameters", 2)
        if tuple(gauss.shape) != (count, 3):
            raise ValueError("gaussian_parameters must have shape (items, 3)")
        coef = _as_float_rows(coefficients, "coefficients", 2)
        if coef.shape[0] != count or coef.shape[1] < 1:
            raise ValueError("coefficients must have shape (items, K) with K >= 1")
        if bool((gauss[:, :2] <= 0).any()):
            raise ValueError("major/minor sigma must be positive arcseconds")
        if bool((gauss[:, 1] > gauss[:, 0]).any()):
            raise ValueError("minor sigma cannot exceed major sigma")
        if min_sigma_arcsec is not None and (
            not math.isfinite(min_sigma_arcsec) or min_sigma_arcsec < 0
        ):
            raise ValueError("min_sigma_arcsec must be None or finite and non-negative")
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
        if not math.isfinite(truncation_sigma) or truncation_sigma <= 0:
            raise ValueError("truncation_sigma must be positive and finite")
        self.min_sigma_arcsec = (
            None if min_sigma_arcsec is None else float(min_sigma_arcsec)
        )
        self.mode = mode
        self.pixel_integration = bool(pixel_integration)
        self.chunk_elements = _check_chunk(chunk_elements)
        self.truncation_sigma = float(truncation_sigma)
        device = lon_t.device
        self.lon = nn.Parameter(torch.remainder(lon_t, 360.0).clone())
        self.lat = nn.Parameter(lat_t.clone())
        self.sigma_arcsec = nn.Parameter(_to(gauss[:, :2], device, lon_t.dtype).clone())
        self.angle_deg = nn.Parameter(_to(gauss[:, 2], device, lon_t.dtype).clone())
        self.coefficients = nn.Parameter(coef.to(device).clone())

    @property
    def component_count(self) -> int:
        """Number of Gaussian items."""
        return int(self.lon.numel())

    @property
    def coefficient_count(self) -> int:
        """Number of coefficient planes ``K``."""
        return int(self.coefficients.shape[1])

    @property
    def gaussian_parameters(self) -> Tensor:
        """``(C, 3)`` rows ``[major_sigma_arcsec, minor_sigma_arcsec, angle_deg]``."""
        return torch.cat((self.sigma_arcsec, self.angle_deg[:, None]), dim=1)

    def _pixel_geometry(self, wcs: WCS) -> tuple[Tensor, Tensor, Tensor]:
        """Pixel centres ``(V, 2)``, covariances ``(V, 2, 2)`` and the indices of the V valid items.

        An item is valid when its centre has a finite pixel position on the target
        projection; the others (for example behind the tangent point) are dropped.
        Differentiable in lon, lat, widths and angle; evaluated in float64.
        """
        device = self.lon.device
        wide = torch.float32 if device.type == "mps" else torch.float64
        lon = self.lon.to(wide)
        lat = self.lat.to(wide)
        with torch.no_grad():
            probe = wcs.world_to_pixel_differentiable(torch.stack((lon, lat), -1))
            valid = torch.isfinite(probe).all(-1) & torch.isfinite(
                self.sigma_arcsec
            ).all(-1)
        index = torch.nonzero(valid).squeeze(1)
        lon, lat = lon[index], lat[index]
        sigma = self.sigma_arcsec[index].to(wide)
        angle = torch.deg2rad(self.angle_deg[index].to(wide))
        if self.min_sigma_arcsec is not None:
            sigma = torch.sqrt(self.min_sigma_arcsec**2 + sigma.square())
        if bool((sigma <= 0).any()):
            raise ValueError("item widths must be positive (after the floor)")

        def to_pixel(lo: Tensor, la: Tensor) -> Tensor:
            return wcs.world_to_pixel_differentiable(torch.stack((lo, la), -1))

        centre = to_pixel(lon, lat)
        # Jacobian d(pixel)/d(east, north arcsec) by central differences over +-1/2 arcsec
        # along the local tangent directions (the lon step is divided by cos(lat)).
        step = 0.5 / 3600.0
        cos_lat = torch.cos(torch.deg2rad(lat)).abs().clamp_min(1e-8)
        d_east = step / cos_lat
        east = (
            to_pixel(torch.remainder(lon + d_east, 360.0), lat)
            - to_pixel(torch.remainder(lon - d_east, 360.0), lat)
        ) / (2 * 3600.0 * step)
        north = (to_pixel(lon, lat + step) - to_pixel(lon, lat - step)) / (
            2 * 3600.0 * step
        )
        jac = torch.stack((east, north), dim=-1)  # (V, 2 pixel axes, 2 sky axes)
        s, c = torch.sin(angle), torch.cos(angle)
        rot = torch.stack(
            (torch.stack((s, -c), -1), torch.stack((c, s), -1)), dim=-2
        )  # columns: major axis (s, c), minor axis (-c, s) in (east, north)
        lin = jac @ rot * sigma[:, None, :]  # J R diag(sigma)
        cov = lin @ lin.transpose(-1, -2)
        return centre, cov, index

    def render_coefficients(
        self, wcs: WCS | dict[str, Any], shape: tuple[int, int] | None = None
    ) -> Tensor:
        """Render the ``K`` coefficient planes onto a tangent-plane grid.

        Args:
            wcs: a :class:`~torchsky.wcs.WCS` or a plain FITS-like header dict.  A dict
                must carry ``NAXIS1`` / ``NAXIS2`` unless ``shape`` is given.
            shape: ``(H, W)`` of the grid; defaults to the header's ``(NAXIS2, NAXIS1)``.

        Returns:
            ``(K, H, W)`` tensor in the dtype of ``coefficients`` on the scene's device:
            flux per pixel (0-indexed pixel centres, ``x`` along columns).  Differentiable
            in every parameter.

        Raises:
            ValueError: if the grid size cannot be determined, or an item width is not
                positive after the floor.
        """
        wcs_obj, grid = self._resolve_grid(wcs, shape)
        centre, cov, index = self._pixel_geometry(wcs_obj)
        return _rasterize_gaussians(
            centre,
            cov,
            self.coefficients[index],
            grid,
            mode=self.mode,
            pixel_integration=self.pixel_integration,
            chunk_elements=self.chunk_elements,
            truncation_sigma=self.truncation_sigma,
        ) + self._graph_anchor(index)

    def _graph_anchor(self, index: Tensor) -> Tensor:
        """Zero that keeps dropped items' parameters in the graph (zero gradient, not None)."""
        dropped = self.lon.numel() - index.numel()
        if dropped == 0:
            return self.coefficients.new_zeros(())
        return 0.0 * (
            self.lon.sum()
            + self.lat.sum()
            + self.sigma_arcsec.sum()
            + self.angle_deg.sum()
        ).to(self.coefficients.dtype)

    def _resolve_grid(
        self, wcs: WCS | dict[str, Any], shape: tuple[int, int] | None
    ) -> tuple[WCS, tuple[int, int]]:
        if isinstance(wcs, WCS):
            if shape is None:
                raise ValueError("shape is required when wcs is a WCS object")
            return wcs.to(self.lon.device), _check_shape(shape)
        if not isinstance(wcs, dict):
            raise TypeError("wcs must be a WCS or a header dict")
        if shape is None:
            if "NAXIS1" not in wcs or "NAXIS2" not in wcs:
                raise ValueError("header has no NAXIS1/NAXIS2 and no shape was given")
            shape = (int(wcs["NAXIS2"]), int(wcs["NAXIS1"]))
        return WCS(wcs).to(self.lon.device), _check_shape(shape)

    def render_to_target(self, target: TargetRenderSpec, **kwargs: Any) -> Tensor:
        """Render one image on a target grid (the ``SceneRepresentation`` contract).

        Args:
            target: output grid; only ``target.header`` and ``target.shape`` are used.
                ``value_semantics`` is not converted: the image is flux per pixel.
            **kwargs: ``band_weights``, ``(K,)`` weights combining the coefficient planes
                into one image (``sum_k w_k plane_k``, computed by folding the weights into
                the coefficients, so a single plane is rasterised).  Required when
                ``K > 1``; optional when ``K == 1``.

        Returns:
            ``(H, W)`` tensor, ``H, W = target.shape``.

        Raises:
            TypeError: for an unknown keyword.
            ValueError: if ``K > 1`` and no ``band_weights`` are given, or the weights
                are not finite with shape ``(K,)``.
        """
        unknown = set(kwargs) - {"band_weights"}
        if unknown:
            raise TypeError(f"unexpected keyword arguments {sorted(unknown)}")
        weights = kwargs.get("band_weights")
        k = self.coefficient_count
        if weights is None:
            if k != 1:
                raise ValueError(
                    "GaussianItemScene has K > 1 coefficient planes: pass band_weights "
                    "or call render_coefficients"
                )
            planes = self.render_coefficients(target.header, target.shape)
            return planes[0]
        w = torch.as_tensor(weights, device=self.coefficients.device).to(
            self.coefficients.dtype
        )
        if tuple(w.shape) != (k,) or not bool(torch.isfinite(w).all()):
            raise ValueError(f"band_weights must be finite with shape ({k},)")
        wcs_obj, grid = self._resolve_grid(target.header, target.shape)
        centre, cov, index = self._pixel_geometry(wcs_obj)
        folded = (self.coefficients[index] @ w)[:, None]
        image = _rasterize_gaussians(
            centre,
            cov,
            folded,
            grid,
            mode=self.mode,
            pixel_integration=self.pixel_integration,
            chunk_elements=self.chunk_elements,
            truncation_sigma=self.truncation_sigma,
        )[0]
        return image + self._graph_anchor(index)


__all__ = ["GaussianItemScene"]
