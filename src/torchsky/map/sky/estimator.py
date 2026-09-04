from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .polynomial import DifferenceTileFactory, _validate_row_major_tiling
from .stats import huber_location, tukey_location


_STATISTIC_REGISTRY = {
    "median": lambda x, dim: torch.sort(x, dim=dim).values[..., x.shape[dim] // 2],
    "huber": lambda x, dim: huber_location(x, dim=dim),
    "tukey": lambda x, dim: tukey_location(x, dim=dim),
}


def _gaussian_kernel_2d(
    kernel_size: int,
    device: torch.device,
    sigma: float | None = None,
) -> Tensor:
    if sigma is None:
        sigma = kernel_size / 3.0
    k = kernel_size // 2
    x = torch.arange(-k, k + 1, dtype=torch.float32, device=device)
    g = torch.exp(-0.5 * (x / sigma) ** 2)
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None, :, :]


@dataclass(frozen=True)
class CompactMeshSkyModel:
    """Filtered background mesh that renders bounded output windows."""

    mesh: Tensor
    image_shape: tuple[int, int]
    interpolate: str

    def render_window(self, y0: int, y1: int, x0: int, x1: int) -> Tensor:
        height, width = self.image_shape
        y0, y1, x0, x1 = int(y0), int(y1), int(x0), int(x1)
        if not (0 <= y0 < y1 <= height and 0 <= x0 < x1 <= width):
            raise ValueError("mesh render window is out of bounds")
        y = (
            2.0
            * (torch.arange(y0, y1, device=self.mesh.device, dtype=self.mesh.dtype) + 0.5)
            / height
            - 1.0
        )
        x = (
            2.0
            * (torch.arange(x0, x1, device=self.mesh.device, dtype=self.mesh.dtype) + 0.5)
            / width
            - 1.0
        )
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack((xx, yy), dim=-1).unsqueeze(0)
        return F.grid_sample(
            self.mesh[None, None],
            grid,
            mode=self.interpolate,
            padding_mode="border",
            align_corners=False,
        )[0, 0]


class MeshSkyEstimator:
    """Differentiable mesh-based 2D sky background estimator.

    Replaces SWarp's ``BACK_SIZE`` + ``BACK_FILTERSIZE`` + sigma-clip
    heuristic with a single-shot robust estimator on GPU.

    Parameters
    ----------
    block_size
        Side length in pixels of each mesh cell (equivalent to SWarp
        ``BACK_SIZE``).
    block_stride
        Stride between mesh blocks. ``None`` means ``block_size`` (no
        overlap). Use ``block_size // 2`` for 50% overlap (SWarp
        default).
    statistic
        Per-cell robust statistic: ``"median"``, ``"huber"``, or
        ``"tukey"``.  ``"median"`` is not differentiable; ``"huber"``
        and ``"tukey"`` are one-step IRLS estimators that *are*
        differentiable.
    mesh_filter
        Post-filter applied to the coarse mesh: ``"median"``,
        ``"gaussian"``, or ``None`` (no filter).  Legacy strings
        ``"median_3"`` and ``"gaussian_3"`` are also accepted.
    mesh_filter_size
        Kernel size for the post-filter (default 3).  Ignored when
        *mesh_filter* is ``None``.
    interpolate
        Interpolation mode for upsampling the mesh to full image
        resolution: ``"bilinear"`` or ``"bicubic"``.
    self_mask_sigma
        If set (e.g. ``3.0``), a single-pass self-masking step is
        applied: pixels with ``|residual| > self_mask_sigma * sigma``
        after an initial estimate are masked and the sky is
        re-estimated.  ``None`` disables self-masking.
    mesh_polynomial_order
        If set (e.g. ``4``), replaces interpolation with a polynomial
        surface of the given total degree fitted to the mesh cell
        values.
    """

    def __init__(
        self,
        *,
        block_size: int = 128,
        block_stride: int | None = None,
        statistic: str = "huber",
        mesh_filter: str | None = "median",
        mesh_filter_size: int = 3,
        interpolate: str = "bicubic",
        self_mask_sigma: float | None = None,
        mesh_polynomial_order: int | None = None,
    ):
        if statistic not in _STATISTIC_REGISTRY:
            raise ValueError(
                f"unknown statistic '{statistic}'; "
                f"choose from {set(_STATISTIC_REGISTRY)}"
            )

        self._filter_type: str | None
        self._filter_kernel: int | None
        if mesh_filter in ("median", "median_3"):
            self._filter_type = "median"
            self._filter_kernel = mesh_filter_size if mesh_filter == "median" else 3
        elif mesh_filter in ("gaussian", "gaussian_3"):
            self._filter_type = "gaussian"
            self._filter_kernel = mesh_filter_size if mesh_filter == "gaussian" else 3
        elif mesh_filter is None:
            self._filter_type = None
            self._filter_kernel = None
        else:
            raise ValueError(
                f"unknown mesh_filter '{mesh_filter}'; "
                f"choose from 'median', 'gaussian', 'median_3', "
                f"'gaussian_3', None"
            )

        if interpolate not in ("bilinear", "bicubic"):
            raise ValueError(
                f"unknown interpolate '{interpolate}'; "
                f"choose from 'bilinear', 'bicubic'"
            )

        if block_stride is not None and block_stride > block_size:
            raise ValueError(
                f"block_stride ({block_stride}) must be <= block_size ({block_size})"
            )

        if mesh_polynomial_order is not None and mesh_polynomial_order < 1:
            raise ValueError(
                f"mesh_polynomial_order must be >= 1 or None, "
                f"got {mesh_polynomial_order}"
            )

        self.block_size = block_size
        self.block_stride = block_size if block_stride is None else block_stride
        self.statistic = statistic
        self.mesh_filter = mesh_filter
        self.mesh_filter_size = mesh_filter_size
        self.interpolate = interpolate
        self.self_mask_sigma = self_mask_sigma
        self.mesh_polynomial_order = mesh_polynomial_order

    def fit_compact(
        self,
        tile_factory: DifferenceTileFactory,
        *,
        image_shape: tuple[int, int],
        scratch: Tensor,
    ) -> CompactMeshSkyModel:
        """Fit and retain only the filtered mesh for bounded tile rendering."""

        if self.self_mask_sigma is not None or self.mesh_polynomial_order is not None:
            raise NotImplementedError(
                "compact mesh fitting does not yet support self masking or "
                "mesh-polynomial interpolation"
            )
        height, width = int(image_shape[0]), int(image_shape[1])
        if (
            height <= 0
            or width <= 0
            or tuple(scratch.shape) != (height, width)
            or not scratch.is_floating_point()
        ):
            raise ValueError(
                "compact mesh scratch must be a positive floating image-sized tensor"
            )
        scratch.fill_(float("nan"))
        layout: list[tuple[int, int, int, int]] = []
        for row in tile_factory():
            if len(row) != 6:
                raise ValueError("compact mesh tile rows require six values")
            y0, y1, x0, x1, difference, support = row
            window = (int(y0), int(y1), int(x0), int(x1))
            y0, y1, x0, x1 = window
            if not (0 <= y0 < y1 <= height and 0 <= x0 < x1 <= width):
                raise ValueError("compact mesh tile is out of bounds")
            window_shape = (y1 - y0, x1 - x0)
            if (
                tuple(difference.shape) != window_shape
                or tuple(support.shape) != window_shape
            ):
                raise ValueError("compact mesh tile data must match its window")
            if (
                difference.device != scratch.device
                or support.device != scratch.device
                or difference.dtype != scratch.dtype
            ):
                raise ValueError(
                    "compact mesh tiles and scratch must share device/dtype"
                )
            valid = support.to(dtype=torch.bool)
            if bool((~torch.isfinite(difference[valid])).any()):
                raise ValueError("compact mesh supported differences must be finite")
            scratch[y0:y1, x0:x1].copy_(
                torch.where(
                    valid,
                    difference,
                    torch.full_like(difference, float("nan")),
                )
            )
            layout.append(window)
        _validate_row_major_tiling(layout, (height, width))
        mesh = self._reduce_mesh_compact(scratch)
        if self._filter_type is not None:
            mesh = self._filter_mesh(mesh)
        return CompactMeshSkyModel(
            mesh=mesh,
            image_shape=(height, width),
            interpolate=self.interpolate,
        )

    def _reduce_mesh_compact(self, image: Tensor) -> Tensor:
        """Reduce one mesh cell at a time without a padded image-sized tensor."""

        height, width = image.shape
        block_size = self.block_size
        stride = self.block_stride
        n_height = (height + stride - 1) // stride
        n_width = (width + stride - 1) // stride
        pad_height = max(0, (n_height - 1) * stride + block_size - height)
        pad_width = max(0, (n_width - 1) * stride + block_size - width)
        reflect = pad_height < height and pad_width < width
        statistic = _STATISTIC_REGISTRY[self.statistic]
        mesh = torch.empty(
            (n_height, n_width), dtype=image.dtype, device=image.device
        )
        for row in range(n_height):
            y_index = torch.arange(
                row * stride,
                row * stride + block_size,
                device=image.device,
            )
            y_index = (
                torch.where(y_index < height, y_index, 2 * height - y_index - 2)
                if reflect
                else y_index.clamp(max=height - 1)
            )
            for column in range(n_width):
                x_index = torch.arange(
                    column * stride,
                    column * stride + block_size,
                    device=image.device,
                )
                x_index = (
                    torch.where(x_index < width, x_index, 2 * width - x_index - 2)
                    if reflect
                    else x_index.clamp(max=width - 1)
                )
                values = image[y_index[:, None], x_index].reshape(-1)
                if bool(values.isnan().any()):
                    fill = torch.nanmedian(values)
                    values = torch.where(values.isnan(), fill, values)
                mesh[row, column] = statistic(values, dim=-1)
        return self._fill_empty_mesh_cells(mesh)

    def __call__(
        self,
        image: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        dtype = image.dtype
        sky = self._estimate(image, mask)

        if self.self_mask_sigma is not None:
            residual = image - sky
            mad = (residual - residual.median()).abs().median() * 1.4826
            source_mask = residual.abs() > self.self_mask_sigma * mad.clamp(min=1e-10)
            if mask is not None:
                source_mask = source_mask | mask
            sky = self._estimate(image, source_mask)

        return sky.to(dtype=dtype)

    def _estimate(self, image: Tensor, mask: Tensor | None) -> Tensor:
        H, W = image.shape[-2:]
        blocks = self._extract_blocks(image)
        flat = blocks.reshape(*blocks.shape[:2], -1)

        if mask is not None:
            mask_blocks = self._extract_blocks(mask.to(image.dtype))
            flat_mask = mask_blocks.reshape(*mask_blocks.shape[:2], -1) > 0.5
            flat = flat.clone()
            flat[flat_mask] = torch.nan

        mesh = self._reduce_mesh(flat)

        if self._filter_type is not None:
            mesh = self._filter_mesh(mesh)

        if self.mesh_polynomial_order is not None:
            sky = self._polynomial_to_image(mesh, (H, W))
        else:
            sky = self._interpolate_to_image(mesh, (H, W))

        return sky

    def _extract_blocks(self, x: Tensor) -> Tensor:
        bh = bw = self.block_size
        stride = self.block_stride
        H, W = x.shape[-2:]

        nH = (H + stride - 1) // stride
        nW = (W + stride - 1) // stride

        last_end_h = (nH - 1) * stride + bh
        last_end_w = (nW - 1) * stride + bw
        pad_h = max(0, last_end_h - H)
        pad_w = max(0, last_end_w - W)

        if pad_h > 0 or pad_w > 0:
            mode = "reflect" if (pad_h < H and pad_w < W) else "replicate"
            xp = F.pad(x[None, None], (0, pad_w, 0, pad_h), mode=mode)[0, 0]
        else:
            xp = x

        return xp.unfold(0, bh, stride).unfold(1, bw, stride)

    def _reduce_mesh(self, flat: Tensor) -> Tensor:
        has_nan = flat.isnan().any()
        if has_nan:
            fill = torch.nanmedian(flat, dim=-1).values[..., None]
            flat = torch.where(flat.isnan(), fill, flat)
        stat_fn = _STATISTIC_REGISTRY[self.statistic]
        mesh = stat_fn(flat, dim=-1)
        return self._fill_empty_mesh_cells(mesh)

    @staticmethod
    def _fill_empty_mesh_cells(mesh: Tensor) -> Tensor:
        valid = torch.isfinite(mesh)
        if bool(valid.all()):
            return mesh
        if not bool(valid.any()):
            raise ValueError("mask excludes every sky mesh cell")
        filled = torch.where(valid, mesh, 0.0)
        kernel = torch.ones((1, 1, 3, 3), dtype=mesh.dtype, device=mesh.device)
        for _ in range(max(mesh.shape)):
            support = valid.to(mesh.dtype)
            count = F.conv2d(support[None, None], kernel, padding=1)[0, 0]
            neighbor_sum = F.conv2d(filled[None, None], kernel, padding=1)[0, 0]
            can_fill = ~valid & (count > 0)
            filled = torch.where(
                can_fill,
                neighbor_sum / count.clamp_min(1),
                filled,
            )
            valid = valid | can_fill
            if bool(valid.all()):
                return filled
        raise RuntimeError("failed to fill disconnected sky mesh cells")

    def _filter_mesh(self, mesh: Tensor) -> Tensor:
        ft = self._filter_type
        fk = self._filter_kernel
        if ft == "median":
            assert fk is not None
            k = fk
            pad = k // 2
            mp = F.pad(mesh[None, None], (pad, pad, pad, pad), mode="reflect")
            patches = mp.unfold(2, k, 1).unfold(3, k, 1)
            nH, nW = mesh.shape
            flat = patches.reshape(1, 1, nH, nW, -1)
            sorted_vals = torch.sort(flat, dim=-1).values
            return sorted_vals[0, 0, ..., flat.shape[-1] // 2]

        if ft == "gaussian":
            assert fk is not None
            k = fk
            sigma = k / 3.0
            kernel = _gaussian_kernel_2d(k, device=mesh.device, sigma=sigma)
            pad = k // 2
            return F.conv2d(mesh[None, None], kernel, padding=pad)[0, 0]

        return mesh

    def _interpolate_to_image(self, mesh: Tensor, image_shape: tuple[int, int]) -> Tensor:
        H, W = image_shape
        mesh_2d = mesh[None, None, :, :]
        up = F.interpolate(
            mesh_2d,
            size=(H, W),
            mode=self.interpolate,
            align_corners=False,
        )
        return up[0, 0]

    def _polynomial_to_image(self, mesh: Tensor, image_shape: tuple[int, int]) -> Tensor:
        H, W = image_shape
        nH, nW = mesh.shape
        order = self.mesh_polynomial_order
        assert order is not None
        device = mesh.device
        dtype = mesh.dtype

        n_terms = (order + 1) * (order + 2) // 2
        if nH * nW < n_terms:
            return self._interpolate_to_image(mesh, (H, W))

        yn = torch.linspace(-1, 1, nH, device=device, dtype=dtype)
        xn = torch.linspace(-1, 1, nW, device=device, dtype=dtype)
        ynv, xnv = torch.meshgrid(yn, xn, indexing="ij")

        terms = []
        for i in range(order + 1):
            for j in range(order + 1 - i):
                terms.append(xnv**i * ynv**j)
        A = torch.stack(terms, dim=-1).reshape(-1, n_terms)

        mesh_flat = mesh.ravel()
        ATA = A.T @ A
        ATb = A.T @ mesh_flat
        coeffs = torch.linalg.solve(ATA, ATb)

        yi = torch.linspace(-1, 1, H, device=device, dtype=dtype)
        xi = torch.linspace(-1, 1, W, device=device, dtype=dtype)
        yv, xv = torch.meshgrid(yi, xi, indexing="ij")

        sky = torch.zeros(H, W, device=device, dtype=dtype)
        x_pow = torch.ones_like(xv)
        y_pow = torch.ones_like(yv)
        idx = 0
        for i in range(order + 1):
            if i > 0:
                x_pow = x_pow * xv
            y_pow = torch.ones_like(yv)
            for j in range(order + 1 - i):
                if j > 0:
                    y_pow = y_pow * yv
                sky.add_(coeffs[idx] * x_pow * y_pow)
                idx += 1

        return sky
