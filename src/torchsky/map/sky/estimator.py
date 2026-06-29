from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .stats import huber_location, tukey_location


_STATISTIC_REGISTRY = {
    "median": lambda x, dim: torch.sort(x, dim=dim).values[..., x.shape[dim] // 2],
    "huber": lambda x, dim: huber_location(x, dim=dim),
    "tukey": lambda x, dim: tukey_location(x, dim=dim),
}


def _gaussian_kernel_2d(sigma: float, kernel_size: int, device: torch.device) -> Tensor:
    k = kernel_size // 2
    x = torch.arange(-k, k + 1, dtype=torch.float32, device=device)
    g = torch.exp(-0.5 * (x / sigma) ** 2)
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None, :, :]


class MeshSkyEstimator:
    """Differentiable mesh-based 2D sky background estimator.

    Replaces SWarp's ``BACK_SIZE`` + ``BACK_FILTERSIZE`` + sigma-clip
    heuristic with a single-shot robust estimator on GPU.

    Parameters
    ----------
    block_size
        Side length in pixels of each mesh cell (equivalent to SWarp
        ``BACK_SIZE``).
    statistic
        Per-cell robust statistic: ``"median"``, ``"huber"``, or
        ``"tukey"``.  ``"median"`` is not differentiable; ``"huber"``
        and ``"tukey"`` are one-step IRLS estimators that *are*
        differentiable.
    mesh_filter
        Post-filter applied to the coarse mesh.  ``"median_3"`` (3×3
        median), ``"gaussian_3"`` (3×3 Gaussian, sigma=1.0), or
        ``None`` (no filter).
    interpolate
        Interpolation mode for upsampling the mesh to full image
        resolution: ``"bilinear"`` or ``"bicubic"``.
    """

    def __init__(
        self,
        *,
        block_size: int = 128,
        statistic: str = "huber",
        mesh_filter: str | None = "median_3",
        interpolate: str = "bicubic",
    ):
        if statistic not in _STATISTIC_REGISTRY:
            raise ValueError(
                f"unknown statistic '{statistic}'; "
                f"choose from {set(_STATISTIC_REGISTRY)}"
            )
        if mesh_filter not in ("median_3", "gaussian_3", None):
            raise ValueError(
                f"unknown mesh_filter '{mesh_filter}'; "
                f"choose from 'median_3', 'gaussian_3', None"
            )
        if interpolate not in ("bilinear", "bicubic"):
            raise ValueError(
                f"unknown interpolate '{interpolate}'; "
                f"choose from 'bilinear', 'bicubic'"
            )
        self.block_size = block_size
        self.statistic = statistic
        self.mesh_filter = mesh_filter
        self.interpolate = interpolate

    def __call__(
        self,
        image: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Estimate 2D sky background from *image*.

        Parameters
        ----------
        image
            ``[H, W]`` tensor.
        mask
            Optional ``[H, W]`` boolean mask — ``True`` pixels are
            **excluded** from the per-cell statistic (e.g.  source
            regions).

        Returns
        -------
        sky_model
            ``[H, W]`` tensor on the same device and dtype as *image*.
        """
        dtype = image.dtype
        H, W = image.shape[-2:]

        blocks = self._extract_blocks(image)  # [nH, nW, bh, bw]
        flat = blocks.reshape(*blocks.shape[:2], -1)

        if mask is not None:
            mask_blocks = self._extract_blocks(mask.to(image.dtype))
            flat_mask = mask_blocks.reshape(*mask_blocks.shape[:2], -1) > 0.5
            flat = flat.clone()
            flat[flat_mask] = torch.nan

        mesh = self._reduce_mesh(flat)

        if self.mesh_filter is not None:
            mesh = self._filter_mesh(mesh)

        sky = self._interpolate_to_image(mesh, (H, W))
        return sky.to(dtype=dtype)

    def _extract_blocks(self, x: Tensor) -> Tensor:
        bh = bw = self.block_size
        H, W = x.shape[-2:]
        nH = (H + bh - 1) // bh
        nW = (W + bw - 1) // bw
        pad_h = nH * bh - H
        pad_w = nW * bw - W
        if pad_h > 0 or pad_w > 0:
            mode = "reflect" if (pad_h < H and pad_w < W) else "replicate"
            xp = F.pad(x[None, None], (0, pad_w, 0, pad_h), mode=mode)[0, 0]
        else:
            xp = x
        return xp.unfold(0, bh, bh).unfold(1, bw, bw)

    def _reduce_mesh(self, flat: Tensor) -> Tensor:
        has_nan = flat.isnan().any()
        if has_nan:
            fill = torch.nanmedian(flat, dim=-1).values[..., None]
            flat = torch.where(flat.isnan(), fill, flat)
        stat_fn = _STATISTIC_REGISTRY[self.statistic]
        return stat_fn(flat, dim=-1)

    def _filter_mesh(self, mesh: Tensor) -> Tensor:
        if self.mesh_filter == "median_3":
            k = 3
            pad = k // 2
            mp = F.pad(mesh[None, None], (pad, pad, pad, pad), mode="reflect")
            patches = mp.unfold(2, k, 1).unfold(3, k, 1)
            nH, nW = mesh.shape
            flat = patches.reshape(1, 1, nH, nW, -1)
            sorted_vals = torch.sort(flat, dim=-1).values
            return sorted_vals[0, 0, ..., flat.shape[-1] // 2]

        if self.mesh_filter == "gaussian_3":
            kernel = _gaussian_kernel_2d(sigma=1.0, kernel_size=3, device=mesh.device)
            return F.conv2d(mesh[None, None], kernel, padding=1)[0, 0]

        return mesh

    def _interpolate_to_image(self, mesh: Tensor, image_shape: tuple[int, int]) -> Tensor:
        H, W = image_shape
        nH, nW = mesh.shape
        mesh_2d = mesh[None, None, :, :]
        up = F.interpolate(
            mesh_2d,
            size=(H, W),
            mode=self.interpolate,
            align_corners=False,
        )
        return up[0, 0]
