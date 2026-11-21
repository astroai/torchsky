"""Convolution with a wide, spatially varying kernel at a cost independent of its extent.

A point-spread function with a sharp core and wings that reach hundreds of
pixels is expensive as a dense stamp: the work of a direct convolution grows as
the stamp area.  :class:`NestedKernelPSF` splits the kernel into concentric
shells.  Shell ``l`` is sampled on a grid of pitch ``2**l`` pixels with a fixed
number of nodes, so its radius doubles with each level while its cost stays
that of a small kernel.  The image is brought to each pitch, convolved with the
shell by FFT, and brought back to full resolution; the shells are summed.
The cost grows with the number of levels, that is, with the logarithm of the
extent.

Conventions follow ``docs/models/psf-conventions.md``: a kernel is an impulse
response stamp, rendering is the convolution ``y[p] = sum_q K[p - q] x[q]``,
the image is zero outside the detector, flux that lands off the detector is
lost, and the adjoint is the exact transpose.  The operator is linear in the
image and source-conditioned: a source at ``q`` is rendered with the kernel
mixed from the nodes surrounding ``q`` (the same flux-conserving convention as
:class:`~torchsky.map.operators.learned_kernel.ConditionedKernelPSF`).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ...device import _as_tensor, _to

__all__ = ["NestedKernelPSF"]

# Radial windows: level ``l`` owns the radii from ``_INNER_FRACTION`` of the
# previous level's reach to its own reach; the two windows cross over smoothly
# (C1) inside that band.  Moving this fraction towards one makes the crossover
# sharper and the coarse levels start at smaller ``r / pitch``; the accuracy as
# a function of ``core_half_width`` (which sets ``r / pitch`` at the crossover)
# is measured in ``tests/test_nested_kernel_psf.py``.
_INNER_FRACTION = 0.5
_GROWTH = 2
_INTERPOLATION_REACH = {"linear": 1, "cubic": 2}


def _smooth_down(r: Tensor, inner: float, outer: float) -> Tensor:
    """One for ``r <= inner``, zero for ``r >= outer``, cubic Hermite (C1) between."""
    t = ((r - inner) / (outer - inner)).clamp(0.0, 1.0)
    return 1.0 - t * t * (3.0 - 2.0 * t)


@lru_cache(maxsize=256)
def _fast_len(n: int) -> int:
    """Smallest ``2**a 3**b 5**c >= n``: an efficient FFT length."""
    m = max(int(n), 1)
    while True:
        k = m
        for p in (2, 3, 5):
            while k % p == 0:
                k //= p
        if k == 1:
            return m
        m += 1


def _interpolation_taps(
    pitch: int, kind: str, *, dtype: torch.dtype, device: torch.device
) -> Tensor:
    """Interpolant sampled at the pixels around a node: ``phi(m / pitch)``.

    ``phi`` is the hat function (``"linear"``) or the Keys cubic convolution
    kernel with ``a = -1/2`` (``"cubic"``).  Both are a partition of unity on
    the node lattice, so ``sum_nodes phi = 1`` at every pixel and a unit flux
    pooled to nodes keeps unit flux.
    """
    reach = _INTERPOLATION_REACH[kind]
    m = torch.arange(
        -reach * pitch, reach * pitch + 1, dtype=torch.float64, device="cpu"
    )
    t = m.abs() / pitch
    if kind == "linear":
        phi = (1.0 - t).clamp_min(0.0)
    else:
        a = -0.5
        near = (a + 2.0) * t**3 - (a + 3.0) * t**2 + 1.0
        far = a * t**3 - 5.0 * a * t**2 + 8.0 * a * t - 4.0 * a
        phi = torch.where(
            t <= 1.0, near, torch.where(t < 2.0, far, torch.zeros_like(t))
        )
    return _to(phi, device, dtype)


def _pool_axis(x: Tensor, pitch: int, taps: Tensor, left: int, n_nodes: int) -> Tensor:
    """``U^T x`` along the last axis: node ``J`` collects ``sum_m phi(m) x[pitch J + m]``."""
    length = (n_nodes - 1) * pitch + int(taps.numel())
    right = length - left - int(x.shape[-1])
    padded = F.pad(x, (left, right))
    return F.conv1d(padded.unsqueeze(1), taps.view(1, 1, -1), stride=pitch).squeeze(1)


def _expand_axis(z: Tensor, pitch: int, taps: Tensor, left: int, size: int) -> Tensor:
    """``U z`` along the last axis: pixel ``j`` reads ``sum_J phi(j - pitch J) z[J]``."""
    full = F.conv_transpose1d(z.unsqueeze(1), taps.view(1, 1, -1), stride=pitch)
    return full.squeeze(1)[..., left : left + size]


def _pool2d(x: Tensor, ty: Tensor, tx: Tensor, level: _LevelFrame) -> Tensor:
    z = _pool_axis(x, level.pitch, tx, level.left_x, level.n_x)
    z = _pool_axis(z.transpose(0, 1), level.pitch, ty, level.left_y, level.n_y)
    return z.transpose(0, 1)


def _expand2d(
    z: Tensor, ty: Tensor, tx: Tensor, level: _LevelFrame, h: int, w: int
) -> Tensor:
    out = _expand_axis(z, level.pitch, tx, level.left_x, w)
    out = _expand_axis(out.transpose(0, 1), level.pitch, ty, level.left_y, h)
    return out.transpose(0, 1)


def _fft_convolve_same(z: Tensor, kernel: Tensor) -> Tensor:
    """``out[I] = sum_d kernel[d + c] z[I - d]`` with ``z`` zero outside (``c`` = centre).

    Linear convolution by zero-padded FFT, cropped to the shape of ``z``.
    """
    ny, nx = int(z.shape[-2]), int(z.shape[-1])
    ky, kx = int(kernel.shape[-2]), int(kernel.shape[-1])
    size = (_fast_len(ny + ky - 1), _fast_len(nx + kx - 1))
    spectrum = torch.fft.rfft2(z, s=size) * torch.fft.rfft2(kernel, s=size)
    full = torch.fft.irfft2(spectrum, s=size)
    return full[ky // 2 : ky // 2 + ny, kx // 2 : kx // 2 + nx]


def _node_positions(lo: float, hi: float, count: int) -> list[float]:
    if count == 1:
        return [0.5 * (lo + hi)]
    return [lo + (hi - lo) * i / (count - 1) for i in range(count)]


@dataclass(frozen=True)
class _Level:
    """One shell of the kernel: nodes of pitch ``pitch`` on a ``(kh, kw)`` grid."""

    pitch: int
    shape: tuple[int, int]
    offsets: Tensor  # [P, 2] float64 (dx, dy) in pixels, nodes with a nonzero window
    window: Tensor  # [P] float64 window times pitch**2 (the node's area)
    flat: Tensor  # [P] int64 index into the flattened (kh, kw) grid
    inner_nodes: float  # nodes closer than this (in node units) have zero window
    reach_nodes: int  # largest |index| of a node with nonzero window


@dataclass(frozen=True)
class _LevelFrame:
    """Coarse-grid geometry of one level for one image."""

    pitch: int
    n_y: int
    n_x: int
    left_y: int
    left_x: int
    crop_y: int
    crop_x: int


def _build_levels(
    core_half_width: int, extent: float, taper_fraction: float
) -> list[_Level]:
    n_h = int(core_half_width)
    n_levels = 1
    while n_h * _GROWTH ** (n_levels - 1) < extent * (1.0 - 1e-9):
        n_levels += 1
    last = n_levels - 1
    pitches = [_GROWTH**level for level in range(n_levels)]
    reach = [float(n_h * p) for p in pitches]
    index = torch.arange(-n_h, n_h + 1, dtype=torch.float64)
    iy, ix = torch.meshgrid(index, index, indexing="ij")

    def cumulative(r: Tensor, level: int) -> Tensor:
        taper = _smooth_down(r, taper_fraction * extent, extent)
        if level < 0:
            return torch.zeros_like(r)
        if level >= last:
            return taper
        return _smooth_down(r, _INNER_FRACTION * reach[level], reach[level]) * taper

    levels = []
    for level, pitch in enumerate(pitches):
        r = torch.hypot(ix, iy) * pitch
        window = cumulative(r, level) - cumulative(r, level - 1)
        mask = window > 0.0
        levels.append(
            _Level(
                pitch=pitch,
                shape=(2 * n_h + 1, 2 * n_h + 1),
                offsets=torch.stack([ix[mask] * pitch, iy[mask] * pitch], dim=-1),
                window=window[mask] * float(pitch) ** 2,
                flat=torch.nonzero(mask.reshape(-1)).squeeze(-1),
                inner_nodes=0.0
                if level == 0
                else _INNER_FRACTION * reach[level - 1] / pitch,
                reach_nodes=int(torch.maximum(ix[mask].abs(), iy[mask].abs()).max()),
            )
        )
    return levels


class _StampField(nn.Module):
    """Kernel field of one dense stamp: looks the stamp up at integer offsets."""

    def __init__(self, stamp: Tensor) -> None:
        super().__init__()
        self.stamp = nn.Parameter(stamp.clone(), requires_grad=False)

    def forward(self, offsets: Tensor, condition: Tensor | None) -> Tensor:
        ky, kx = self.stamp.shape
        col = offsets[:, 0].round().long() + kx // 2
        row = offsets[:, 1].round().long() + ky // 2
        return self.stamp.reshape(-1)[row * kx + col].unsqueeze(0)


class NestedKernelPSF(nn.Module):
    """Wide spatially varying PSF applied at a cost that does not grow with its extent.

    The PSF is a kernel field ``kernel(offsets, condition)``, a non-negative
    density per square pixel of the offset ``(dx, dy)`` between a source and the
    pixel it lights.  It is split into shells sampled at pitches ``1, 2, 4, ...``
    pixels with ``(2 core_half_width + 1)**2`` nodes each, so the core is
    resolved at full resolution while the wings are carried at a coarse pitch
    matched to their radius.  Level ``l`` pools the image to nodes of pitch
    ``2**l`` (the transpose of the interpolant), convolves by FFT with the
    shell's weights, and reads the result back with the interpolant; the levels
    are summed.  Each node kernel has unit sum over all shells, so the response
    to a point source whose stamp lies on the detector carries its flux.

    The kernel may vary across the detector through a regular grid of region
    nodes.  The kernel field is evaluated at each node with a ``condition``
    built from the node position, wavelength and time, and a source is rendered
    with the tent-weighted mixture of the node kernels at the source position
    (partition of unity; constant beyond the outermost nodes).

    Conventions (see ``docs/models/psf-conventions.md``): ``dx`` runs along
    columns and ``dy`` along rows; the response of a unit source at ``q`` is
    ``K[p - q]`` at pixel ``p``; the image is zero outside the detector and the
    part of a response falling off it is lost; :meth:`adjoint` is the exact
    transpose of :meth:`forward`.  The pooling lattices are anchored to
    **detector coordinates** (node ``J`` of level ``l`` sits at coordinate
    ``2**l J``), not to the image array, so rendering a cropped tile with its
    own coordinates gives the same values as the full image (up to round-off)
    when the crop includes a halo of :attr:`kernel_size` ``// 2`` pixels.

    Args:
        kernel: Kernel field, callable or module, ``kernel(offsets, condition)``
            with ``offsets`` of shape ``[P, 2]`` (columns ``dx``, ``dy``, pixels),
            ``condition`` of shape ``[N, D]`` (``N`` region nodes), returning a
            non-negative density of shape ``[N, P]`` (``[P]`` or ``[1, P]`` is
            accepted for a field that does not depend on the condition).  Only its relative values matter: each node kernel is
            normalised to unit sum.
        core_half_width: Half-width, in pixels, of the full-resolution core
            grid (``>= 2``).  Every level has ``2 core_half_width + 1`` nodes per
            side; larger values lower the interpolation error and raise the cost
            of each level.  The finest level resolves radii up to
            ``_INNER_FRACTION * core_half_width`` exactly.
        extent: Radius, in pixels, at which the kernel is cut to zero
            (a finite support, as for any dense stamp).  The number of levels is
            the smallest for which ``core_half_width * 2**(levels - 1) >= extent``.
        taper_fraction: The kernel falls smoothly (C1) to zero between
            ``taper_fraction * extent`` and ``extent``; in ``(0, 1)``.
        nodes: ``(ny, nx)`` region nodes over the detector.  ``(1, 1)`` is a
            stationary kernel (evaluated once, at the centre of ``bounds``).
        bounds: ``(x_min, x_max, y_min, y_max)`` detector coordinates of the
            outermost nodes; required when ``nodes`` has more than one node
            along an axis, so that tiles of a detector share one node grid.
        encoder: Optional ``encoder(spatial_x, spatial_y, wavelength, time_mjd)``
            returning ``[..., D]`` features (for example
            :class:`~torchsky.map.operators.learned_kernel.SpatioSpectralTemporalEncoder`).
            Default: the raw features ``[x, y, wavelength, time_mjd or 0]``.
        interpolation: ``"linear"`` (hat function, non-negative responses,
            second-order accurate) or ``"cubic"`` (Keys cubic convolution,
            third-order accurate, small negative lobes in the response).

    Attributes:
        kernel_size: Odd integer; the dense stamp that contains the whole
            response is ``kernel_size x kernel_size`` (halo of ``kernel_size // 2``
            pixels for tiled rendering).  It includes the reach of the
            interpolants and so exceeds ``2 * extent + 1``.
        num_levels: Number of nested levels.

    Raises:
        ValueError: On invalid geometry (see the argument ranges above), a
            kernel field that returns negative, non-finite or mis-shaped values,
            or coordinates that are not a unit-pitch detector grid.

    Examples:
        >>> import torch
        >>> def field(offsets, condition):
        ...     r2 = (offsets**2).sum(-1)
        ...     return (0.8 * torch.exp(-r2 / 2.0) + 0.2 / (1.0 + r2 / 25.0) ** 1.5)[None]
        >>> psf = NestedKernelPSF(field, core_half_width=8, extent=64.0)
        >>> image = torch.zeros(128, 128, dtype=torch.float64)
        >>> image[64, 64] = 1.0
        >>> y, x = torch.meshgrid(torch.arange(128.0), torch.arange(128.0), indexing="ij")
        >>> out = psf(image, spatial_x=x, spatial_y=y, wavelength=0.5)
        >>> bool(abs(float(out.sum()) - 1.0) < 1e-9)
        True
    """

    def __init__(
        self,
        kernel: Callable[[Tensor, Tensor | None], Tensor],
        *,
        core_half_width: int = 16,
        extent: float = 128.0,
        taper_fraction: float = 0.85,
        nodes: tuple[int, int] = (1, 1),
        bounds: tuple[float, float, float, float] | None = None,
        encoder: Callable[..., Tensor] | None = None,
        interpolation: str = "linear",
    ) -> None:
        super().__init__()
        if not callable(kernel):
            raise TypeError("kernel must be a callable kernel field")
        if int(core_half_width) != core_half_width or core_half_width < 2:
            raise ValueError("core_half_width must be an integer >= 2")
        if not (math.isfinite(extent) and extent > 0.0):
            raise ValueError("extent must be finite and positive")
        if not (0.0 < taper_fraction < 1.0):
            raise ValueError("taper_fraction must be in (0, 1)")
        if interpolation not in _INTERPOLATION_REACH:
            raise ValueError("interpolation must be 'linear' or 'cubic'")
        self._setup_nodes(nodes, bounds)
        self.kernel = kernel
        self.encoder = encoder
        self.interpolation = interpolation
        self.core_half_width = int(core_half_width)
        self.extent = float(extent)
        self.taper_fraction = float(taper_fraction)
        self._levels = _build_levels(
            self.core_half_width, self.extent, self.taper_fraction
        )
        self._finish_geometry()

    @classmethod
    def from_stamp(
        cls,
        stamp: Tensor,
    ) -> NestedKernelPSF:
        """Stationary operator from one dense odd-sized impulse-response stamp.

        A single level of pitch one pixel: the operator equals
        :func:`~torchsky.map.operators.psf.apply_kernel2d` with the stamp (the
        limit of the nested scheme with one level) up to the FFT round-off.  The
        stamp is normalised to unit sum and must be non-negative.

        Args:
            stamp: ``[ky, kx]`` odd-sized, finite, non-negative stamp with
                positive sum; the entry ``[ky // 2 + dy, kx // 2 + dx]`` is the
                response at offset ``(dx, dy)``.

        Returns:
            A :class:`NestedKernelPSF` with ``kernel_size = max(ky, kx)``.

        Raises:
            ValueError: If the stamp is not 2-D, odd-sized, finite, non-negative
                with positive sum.

        Examples:
            >>> import torch
            >>> psf = NestedKernelPSF.from_stamp(torch.ones(3, 3, dtype=torch.float64))
            >>> psf.kernel_size
            3
        """
        if stamp.ndim != 2 or stamp.numel() == 0:
            raise ValueError("stamp must be a non-empty 2-D tensor")
        if any(int(n) % 2 == 0 for n in stamp.shape):
            raise ValueError("stamp dimensions must be odd")
        if not bool(torch.isfinite(stamp).all()):
            raise ValueError("stamp must contain only finite values")
        if bool((stamp < 0).any()):
            raise ValueError("stamp must be non-negative")
        if not bool(stamp.sum() > 0):
            raise ValueError("stamp must have positive sum")
        self = cls.__new__(cls)
        nn.Module.__init__(self)
        self._setup_nodes((1, 1), None)
        ky, kx = (int(n) for n in stamp.shape)
        self.kernel = _StampField(stamp.detach())
        self.encoder = None
        self.interpolation = "linear"
        self.core_half_width = max(ky, kx) // 2
        self.extent = float(max(ky, kx) // 2)
        self.taper_fraction = 1.0
        cy, cx = ky // 2, kx // 2
        iy, ix = torch.meshgrid(
            torch.arange(-cy, cy + 1, dtype=torch.float64),
            torch.arange(-cx, cx + 1, dtype=torch.float64),
            indexing="ij",
        )
        self._levels = [
            _Level(
                pitch=1,
                shape=(ky, kx),
                offsets=torch.stack([ix.reshape(-1), iy.reshape(-1)], dim=-1),
                window=torch.ones(ky * kx, dtype=torch.float64),
                flat=torch.arange(ky * kx),
                inner_nodes=0.0,
                reach_nodes=max(cy, cx),
            )
        ]
        self._finish_geometry()
        return self

    # ------------------------------------------------------------------ setup

    def _setup_nodes(
        self,
        nodes: tuple[int, int],
        bounds: tuple[float, float, float, float] | None,
    ) -> None:
        if len(nodes) != 2 or any(int(n) != n or n < 1 for n in nodes):
            raise ValueError("nodes must be two positive integers (ny, nx)")
        self.nodes = (int(nodes[0]), int(nodes[1]))
        if bounds is None:
            if max(self.nodes) > 1:
                raise ValueError("bounds are required when there is more than one node")
            bounds = (0.0, 0.0, 0.0, 0.0)
        if len(bounds) != 4 or not all(math.isfinite(b) for b in bounds):
            raise ValueError(
                "bounds must be four finite numbers (x_min, x_max, y_min, y_max)"
            )
        x_min, x_max, y_min, y_max = (float(b) for b in bounds)
        if (self.nodes[1] > 1 and not x_max > x_min) or (
            self.nodes[0] > 1 and not y_max > y_min
        ):
            raise ValueError(
                "bounds must increase along every axis that has several nodes"
            )
        self.bounds = (x_min, x_max, y_min, y_max)
        self._node_x = _node_positions(x_min, x_max, self.nodes[1])
        self._node_y = _node_positions(y_min, y_max, self.nodes[0])

    def _finish_geometry(self) -> None:
        reach = _INTERPOLATION_REACH[self.interpolation]
        half = 0
        for level in self._levels:
            spread = 0 if level.pitch == 1 else reach * level.pitch - 1
            half = max(half, level.reach_nodes * level.pitch + 2 * spread)
        self.kernel_size = 2 * half + 1
        self.num_levels = len(self._levels)

    # ----------------------------------------------------------------- helpers

    @property
    def n_nodes(self) -> int:
        """Number of region nodes ``ny * nx``."""
        return self.nodes[0] * self.nodes[1]

    def _frame(
        self, image: Tensor, spatial_x: Tensor, spatial_y: Tensor
    ) -> tuple[int, int, Tensor, Tensor, list[_LevelFrame]]:
        if image.ndim != 2 or not image.dtype.is_floating_point:
            raise ValueError("NestedKernelPSF image must be a 2-D floating tensor")
        if spatial_x.shape != image.shape or spatial_y.shape != image.shape:
            raise ValueError("PSF spatial coordinates must match image shape")
        height, width = int(image.shape[0]), int(image.shape[1])
        # Geometry is float64 on the host (it is O(H + W)); only the regularity
        # check touches the full coordinate arrays, in their own dtype and device.
        xs = _to(spatial_x[0].detach(), "cpu", torch.float64)
        ys = _to(spatial_y[:, 0].detach(), "cpu", torch.float64)
        x_ref = xs[0] + torch.arange(width, dtype=torch.float64)
        y_ref = ys[0] + torch.arange(height, dtype=torch.float64)
        for coords, ref, expected in (
            (spatial_x, x_ref, x_ref[None, :]),
            (spatial_y, y_ref, y_ref[:, None]),
        ):
            # Absolute 1e-4 pixel plus eight units in the last place of the
            # coordinates' own dtype: a unit-pitch grid stored in float32 far from
            # the origin is still one.
            tol = 1e-4 + 8.0 * torch.finfo(coords.dtype).eps * float(ref.abs().max())
            reference = _to(expected, coords.device, coords.dtype)
            if float((coords.detach() - reference).abs().max()) > tol:
                raise ValueError(
                    "spatial_x and spatial_y must be a unit-pitch detector grid: "
                    "x = x0 + column, y = y0 + row"
                )
        ix0, iy0 = round(float(xs[0])), round(float(ys[0]))
        reach = _INTERPOLATION_REACH[self.interpolation]
        frames = []
        for level in self._levels:
            s = level.pitch
            r = 0 if s == 1 else reach
            lo_x, lo_y = ix0 // s - max(r - 1, 0), iy0 // s - max(r - 1, 0)
            hi_x = -((-(ix0 + width - 1)) // s) + max(r - 1, 0)
            hi_y = -((-(iy0 + height - 1)) // s) + max(r - 1, 0)
            n_x, n_y = (hi_x - lo_x + 1, hi_y - lo_y + 1) if s > 1 else (width, height)
            frames.append(
                _LevelFrame(
                    pitch=s,
                    n_y=n_y,
                    n_x=n_x,
                    left_y=iy0 - s * lo_y + r * s if s > 1 else 0,
                    left_x=ix0 - s * lo_x + r * s if s > 1 else 0,
                    crop_y=min(level.shape[0] // 2, n_y - 1),
                    crop_x=min(level.shape[1] // 2, n_x - 1),
                )
            )
        return ix0, iy0, x_ref, y_ref, frames

    def _conditions(
        self,
        wavelength: Tensor | float,
        time_mjd: Tensor | float | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        gy, gx = torch.meshgrid(
            torch.tensor(self._node_y, dtype=torch.float64),
            torch.tensor(self._node_x, dtype=torch.float64),
            indexing="ij",
        )
        nx = _to(gx.reshape(-1), device, dtype)
        ny = _to(gy.reshape(-1), device, dtype)
        if self.encoder is not None:
            cond = self.encoder(nx, ny, wavelength, time_mjd)
        else:
            wl = _as_tensor(wavelength, dtype, device).expand_as(nx)
            tm = _as_tensor(0.0 if time_mjd is None else time_mjd, dtype, device)
            cond = torch.stack([nx, ny, wl, tm.expand_as(nx)], dim=-1)
        if cond.shape[0] != self.n_nodes:
            raise ValueError("encoder must return one condition row per region node")
        return cond

    def _node_kernels(
        self,
        wavelength: Tensor | float,
        time_mjd: Tensor | float | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> list[Tensor]:
        """Unit-sum discrete weights of every level for every node: ``[N, kh, kw]`` each."""
        param = (
            next(self.kernel.parameters(), None)
            if isinstance(self.kernel, nn.Module)
            else None
        )
        field_dtype = dtype if param is None else param.dtype
        cond = self._conditions(wavelength, time_mjd, device, field_dtype)
        offsets = _to(
            torch.cat([lv.offsets for lv in self._levels]), device, field_dtype
        )
        density = self.kernel(offsets, cond)
        if density.ndim == 1:
            density = density.unsqueeze(0)
        if density.ndim == 2 and density.shape[0] == 1 and self.n_nodes > 1:
            density = density.expand(
                self.n_nodes, -1
            )  # a field that ignores the condition
        if tuple(density.shape) != (self.n_nodes, int(offsets.shape[0])):
            raise ValueError(
                f"kernel field must return [{self.n_nodes}, {int(offsets.shape[0])}] "
                f"(nodes, offsets); got {tuple(density.shape)}"
            )
        if not bool(torch.isfinite(density).all()):
            raise ValueError("kernel field returned non-finite values")
        if bool((density < 0).any()):
            raise ValueError("kernel field must be non-negative")
        density = density.to(dtype)
        area = _to(torch.cat([lv.window for lv in self._levels]), device, dtype)
        weights = density * area
        total = weights.sum(dim=-1, keepdim=True)
        if not bool((total > 0).all()):
            raise ValueError("kernel field has zero total weight on a region node")
        weights = weights / total
        out, start = [], 0
        for level in self._levels:
            count = int(level.flat.numel())
            grid = weights.new_zeros(self.n_nodes, level.shape[0] * level.shape[1])
            grid = grid.index_copy(
                1, level.flat.to(device), weights[:, start : start + count]
            )
            out.append(grid.reshape(self.n_nodes, *level.shape))
            start += count
        return out

    @staticmethod
    def _hat(coords: Tensor, nodes: list[float], index: int) -> Tensor:
        """Tent weight of node ``index`` at ``coords``; constant beyond the outer nodes."""
        if len(nodes) == 1:
            return torch.ones_like(coords)
        step = nodes[1] - nodes[0]
        clamped = coords.clamp(nodes[0], nodes[-1])
        return (1.0 - (clamped - nodes[index]).abs() / step).clamp_min(0.0)

    def _apply(
        self,
        image: Tensor,
        spatial_x: Tensor,
        spatial_y: Tensor,
        wavelength: Tensor | float,
        time_mjd: Tensor | float | None,
        *,
        adjoint: bool,
    ) -> Tensor:
        _, _, x_cols, y_rows, frames = self._frame(image, spatial_x, spatial_y)
        height, width = int(image.shape[0]), int(image.shape[1])
        dtype, device = image.dtype, image.device
        kernels = self._node_kernels(wavelength, time_mjd, device, dtype)
        taps = {
            lv.pitch: _interpolation_taps(
                lv.pitch, self.interpolation, dtype=dtype, device=device
            )
            for lv in self._levels
            if lv.pitch > 1
        }
        out = torch.zeros_like(image)
        for jy in range(self.nodes[0]):
            wy = self._hat(y_rows, self._node_y, jy)
            if not bool((wy > 0).any()):
                continue
            for jx in range(self.nodes[1]):
                wx = self._hat(x_cols, self._node_x, jx)
                if not bool((wx > 0).any()):
                    continue
                node = jy * self.nodes[1] + jx
                weight = (
                    _to(wy, device, dtype)[:, None] * _to(wx, device, dtype)[None, :]
                )
                single = self.n_nodes == 1
                source = image if (adjoint or single) else image * weight
                acc = torch.zeros_like(image)
                for level, frame, grid in zip(
                    self._levels, frames, kernels, strict=True
                ):
                    if (
                        level.pitch > 1
                        and math.hypot(frame.n_y - 1, frame.n_x - 1)
                        <= level.inner_nodes
                    ):
                        continue  # every node of this shell is farther than the coarse grid
                    cy, cx = level.shape[0] // 2, level.shape[1] // 2
                    k = grid[
                        node,
                        cy - frame.crop_y : cy + frame.crop_y + 1,
                        cx - frame.crop_x : cx + frame.crop_x + 1,
                    ]
                    k = k / float(level.pitch) ** 2
                    if adjoint:
                        k = k.flip(-2, -1)
                    if level.pitch == 1:
                        acc = acc + _fft_convolve_same(source, k)
                    else:
                        t = taps[level.pitch]
                        z = _pool2d(source, t, t, frame)
                        z = _fft_convolve_same(z, k)
                        acc = acc + _expand2d(z, t, t, frame, height, width)
                out = out + (acc if (not adjoint or single) else acc * weight)
        return out

    # -------------------------------------------------------------- public API

    def forward(
        self,
        image: Tensor,
        *,
        spatial_x: Tensor,
        spatial_y: Tensor,
        wavelength: Tensor | float,
        time_mjd: Tensor | float | None = None,
    ) -> Tensor:
        """Render ``image`` through the PSF: ``y[p] = sum_q K_q[p - q] image[q]``.

        Args:
            image: ``[H, W]`` floating tensor (float32 or float64), pixel
                values per pixel; zero outside the image.
            spatial_x: ``[H, W]`` detector x coordinate of each pixel
                (pixels, increasing by one per column).
            spatial_y: ``[H, W]`` detector y coordinate (increasing by one
                per row).
            wavelength: Scalar passed to the condition (for example micrometres).
            time_mjd: Optional scalar epoch passed to the condition.

        Returns:
            ``[H, W]`` rendered image, same dtype and device as ``image``.
            Flux that falls off the detector is lost.

        Raises:
            ValueError: If the coordinates are not a unit-pitch grid or the
                kernel field is invalid.

        Examples:
            >>> import torch
            >>> psf = NestedKernelPSF.from_stamp(torch.ones(3, 3, dtype=torch.float64))
            >>> y, x = torch.meshgrid(torch.arange(5.0), torch.arange(5.0), indexing="ij")
            >>> psf(torch.ones(5, 5, dtype=torch.float64), spatial_x=x, spatial_y=y, wavelength=0.0).shape
            torch.Size([5, 5])
        """
        return self._apply(
            image, spatial_x, spatial_y, wavelength, time_mjd, adjoint=False
        )

    def adjoint(
        self,
        image: Tensor,
        *,
        spatial_x: Tensor,
        spatial_y: Tensor,
        wavelength: Tensor | float,
        time_mjd: Tensor | float | None = None,
    ) -> Tensor:
        """Exact transpose of :meth:`forward`: ``<A x, y> = <x, A^T y>`` to round-off.

        Args:
            image: ``[H, W]`` floating tensor in the output space of ``forward``.
            spatial_x: As for :meth:`forward`.
            spatial_y: As for :meth:`forward`.
            wavelength: As for :meth:`forward`.
            time_mjd: As for :meth:`forward`.

        Returns:
            ``[H, W]`` tensor in the input space of ``forward``.

        Raises:
            ValueError: As for :meth:`forward`.

        Examples:
            >>> import torch
            >>> psf = NestedKernelPSF.from_stamp(torch.ones(3, 3, dtype=torch.float64))
            >>> y, x = torch.meshgrid(torch.arange(5.0), torch.arange(5.0), indexing="ij")
            >>> psf.adjoint(torch.ones(5, 5, dtype=torch.float64), spatial_x=x, spatial_y=y, wavelength=0.0).shape
            torch.Size([5, 5])
        """
        return self._apply(
            image, spatial_x, spatial_y, wavelength, time_mjd, adjoint=True
        )

    def effective_kernels(
        self,
        spatial_x: Tensor,
        spatial_y: Tensor,
        wavelength: Tensor | float,
        time_mjd: Tensor | float | None = None,
    ) -> Tensor:
        """Dense stamps the operator is equivalent to for sources at given positions.

        The stamp of a source at ``(x, y)`` is the response of the operator
        (on a detector large enough to hold it) to a unit source there, in
        impulse-response orientation: entry ``[c + dy, c + dx]`` with
        ``c = kernel_size // 2`` is the response at offset ``(dx, dy)``.  The
        nested scheme is not exactly shift-invariant, so the stamp depends on
        the position of the source modulo the pitch of the coarsest level, and
        the operator on a detector is exactly the scatter of these stamps with
        zero outside the detector.  Meant for inspection and tests: it renders
        a ``kernel_size x kernel_size`` image per position.

        Args:
            spatial_x: Source x coordinates (pixels), any shape ``S``; rounded to
                the nearest pixel for the lattice phase, used as is for the
                node weights.
            spatial_y: Source y coordinates, same shape.
            wavelength: Scalar passed to the condition.
            time_mjd: Optional scalar epoch.

        Returns:
            Tensor ``[*S, kernel_size, kernel_size]`` in float64 unless the kernel
            field is single precision.

        Raises:
            ValueError: If the shapes of ``spatial_x`` and ``spatial_y`` differ.

        Examples:
            >>> import torch
            >>> psf = NestedKernelPSF.from_stamp(torch.ones(3, 3, dtype=torch.float64))
            >>> psf.effective_kernels(torch.zeros(2), torch.zeros(2), 0.0).shape
            torch.Size([2, 3, 3])
        """
        if spatial_x.shape != spatial_y.shape:
            raise ValueError("spatial_x and spatial_y must have the same shape")
        param = next(self.parameters(), None)
        dtype = (
            torch.float64
            if param is None or param.dtype == torch.float64
            else param.dtype
        )
        device = spatial_x.device if param is None else param.device
        half = self.kernel_size // 2
        size = 2 * half + 1
        axis = torch.arange(-half, half + 1, dtype=torch.float64, device=device)
        stamps = []
        for x, y in zip(
            spatial_x.reshape(-1).tolist(), spatial_y.reshape(-1).tolist(), strict=True
        ):
            impulse = torch.zeros(size, size, dtype=dtype, device=device)
            impulse[half, half] = 1.0
            gx = (x + axis).expand(size, size)
            gy = (y + axis)[:, None].expand(size, size)
            # The lattice is anchored to the rounded coordinates, so a local grid
            # whose origin is an integer shift of the detector grid is consistent.
            stamps.append(
                self._apply(impulse, gx, gy, wavelength, time_mjd, adjoint=False)
            )
        out = torch.stack(stamps) if stamps else torch.zeros(0, size, size, dtype=dtype)
        return out.reshape(*spatial_x.shape, size, size)
