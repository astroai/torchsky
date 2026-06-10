"""
HiPS tile renderer.

Converts any torchsky-readable FITS image into HiPS (Hierarchical Progressive
Survey) PNG tiles suitable for Aladin Lite v3.

Pipeline per tile:
    1. Compute (lon, lat) grid covering the HEALPix tile using torchsky.sphere.healpix
    2. Project (lon, lat) → FITS pixel coords via torchsky.WCS.world_to_pixel_2d
    3. Extract sub-image via torch.nn.functional.grid_sample (bilinear, GPU if avail)
    4. Apply stretch (ZScale/Asinh/Log/…) via renderers.image
    5. Encode to PNG via Pillow
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from .image import apply_stretch, to_png, zscale_limits

# Delay-import torchsky sub-modules to avoid circular imports
_healpix = None
_WCS = None


def _lazy_import() -> None:
    global _healpix, _WCS
    if _healpix is None:
        from torchsky.sphere import healpix as _hp

        _healpix = _hp
    if _WCS is None:
        from torchsky.wcs import WCS as _W

        _WCS = _W


# ---------------------------------------------------------------------------
# Tile size constant (HiPS standard: 512)
# ---------------------------------------------------------------------------
TILE_SIZE = 512


def compute_hips_order(wcs, image_shape: tuple[int, int]) -> int:
    """
    Estimate the maximum useful HiPS order for a given WCS + image shape.

    Returns the order where HEALPix pixel scale ≈ image pixel scale.
    """
    _lazy_import()
    h, w = image_shape
    try:
        cd = wcs.cd_full[:2, :2].double()
        # Pixel scale in degrees per pixel (approx)
        pixel_scale_deg = float(torch.sqrt(torch.det(cd).abs()).item())
    except Exception:
        return 5  # safe fallback

    if pixel_scale_deg < 1e-10:
        return 10

    # HEALPix pixel solid angle at order k: Ω = 4π / (12 * 4^k) sr
    # Angular scale ~ sqrt(Ω) in radians ~ sqrt(π / (3 * 4^k))
    # We want this to equal pixel_scale_deg * π/180
    pixel_rad = pixel_scale_deg * math.pi / 180.0
    k = math.log2(math.sqrt(math.pi / 3.0) / pixel_rad) / 2.0
    # Clamp to reasonable range
    max_order = max(1, min(13, int(k) + 1))
    return max_order


def _tile_lon_lat(
    order: int, pix: int, tile_size: int = TILE_SIZE
) -> tuple[Tensor, Tensor]:
    """
    Compute (lon, lat) coordinates for each pixel of a HiPS tile.

    Uses the HEALPix NESTED subdivision: each tile at (order, pix)
    corresponds to NESTED sub-pixels at order+log2(tile_size).
    Returns (lon, lat) tensors of shape (tile_size, tile_size), in degrees.
    """
    _lazy_import()
    nside_tile = 1 << order
    # HiPS tiles use nested sub-pixels; tile_size must be a power of 2
    # (adds int(log2(tile_size)) extra HEALPix orders — e.g. 9 for 512)
    nside_sub = nside_tile * tile_size  # 2^(order+log2(tile_size))

    # Sub-pixel indices in NESTED order within this tile
    start = pix * (tile_size * tile_size)
    sub_pix = torch.arange(start, start + tile_size * tile_size, dtype=torch.int64)

    lon, lat = _healpix.pix2ang(nside_sub, sub_pix, nest=True, lonlat=True)
    # Standard HiPS tile: reshape so rows run South→North, cols West→East
    # The NESTED space-filling curve gives a specific 2D layout
    lon = lon.reshape(tile_size, tile_size)
    lat = lat.reshape(tile_size, tile_size)
    return lon, lat


def _world_to_sample_grid(
    lon: Tensor,
    lat: Tensor,
    wcs,
    image_h: int,
    image_w: int,
) -> tuple[Tensor, Tensor]:
    """
    Project (lon, lat) → normalized grid sample coordinates [-1, 1].

    Returns (grid_x, grid_y) ready for F.grid_sample, and a validity mask.
    """
    # WCS world_to_pixel: (ra, dec) tensors → (x, y) pixel coords (FITS 1-based)
    x_pix, y_pix = wcs.world_to_pixel_2d(lon.reshape(-1), lat.reshape(-1))

    # Normalize to [-1, 1] for grid_sample (converts from FITS 1-based)
    # FITS pixel convention: pixel 1 = image edge
    grid_x = (x_pix - 1) / (image_w - 1) * 2.0 - 1.0
    grid_y = (y_pix - 1) / (image_h - 1) * 2.0 - 1.0

    # Validity: pixels that fall inside the image
    valid = (
        torch.isfinite(grid_x)
        & torch.isfinite(grid_y)
        & (grid_x >= -1.0)
        & (grid_x <= 1.0)
        & (grid_y >= -1.0)
        & (grid_y <= 1.0)
    )
    # Clamp to avoid grid_sample complaints
    grid_x = grid_x.clamp(-1.0, 1.0)
    grid_y = grid_y.clamp(-1.0, 1.0)

    return grid_x, grid_y, valid


def render_hips_tile(
    image: Tensor,
    wcs,
    order: int,
    pix: int,
    stretch: str = "zscale",
    cmap: str = "gray",
    vmin: float | None = None,
    vmax: float | None = None,
    tile_size: int = TILE_SIZE,
) -> bytes | None:
    """
    Render one HiPS tile as PNG bytes.

    Args:
        image:   2D float tensor (H×W), already read from FITS.
        wcs:     torchsky.WCS instance for this image.
        order:   HiPS order (HEALPix nside = 2^order).
        pix:     HEALPix NESTED pixel index.
        stretch: Stretch algorithm name.
        cmap:    Colormap name.
        vmin/vmax: Display limits. If None, computed from a sample of the full image.
        tile_size: Output tile size in pixels (default 512).

    Returns:
        PNG bytes, or None if the tile is entirely outside the image footprint.
    """
    _lazy_import()
    image_h, image_w = image.shape[-2], image.shape[-1]

    # Compute display limits if not provided
    if vmin is None or vmax is None:
        sample = image.flatten()
        sample = sample[torch.isfinite(sample)]
        if sample.numel() == 0:
            return None
        _vmin, _vmax = zscale_limits(sample)
        if vmin is None:
            vmin = _vmin
        if vmax is None:
            vmax = _vmax

    # Compute (lon, lat) for each pixel of this tile
    try:
        lon, lat = _tile_lon_lat(order, pix, tile_size)
    except Exception:
        return None

    lon = lon.to(torch.float64)
    lat = lat.to(torch.float64)

    # Project to FITS pixel coordinates
    try:
        grid_x, grid_y, valid = _world_to_sample_grid(lon, lat, wcs, image_h, image_w)
    except Exception:
        return None

    if not valid.any():
        return None  # tile entirely outside image

    # Build grid for F.grid_sample: (1, 1, tile_size^2, 2)
    grid = torch.stack([grid_x, grid_y], dim=-1).float()
    grid = grid.reshape(1, tile_size * tile_size, 1, 2)

    # Extract pixel values using bilinear interpolation
    src = image.float().unsqueeze(0).unsqueeze(0)  # 1×1×H×W
    sampled = F.grid_sample(
        src, grid, mode="bilinear", align_corners=True, padding_mode="zeros"
    )
    # sampled: 1×1×(tile^2)×1
    sampled = sampled.squeeze().reshape(tile_size, tile_size)

    # Mask out-of-bounds pixels
    valid_2d = valid.reshape(tile_size, tile_size)
    sampled = sampled.masked_fill(~valid_2d, float("nan"))

    # Apply stretch and encode
    stretched = apply_stretch(sampled, vmin, vmax, stretch)
    alpha_mask = valid_2d

    return to_png(stretched, cmap=cmap, alpha_mask=alpha_mask)


def render_preview(
    image: Tensor,
    max_pixels: int = 1024,
    stretch: str = "zscale",
    cmap: str = "gray",
    vmin: float | None = None,
    vmax: float | None = None,
) -> bytes:
    """
    Render a full-image preview PNG, downsampled to at most max_pixels on the long axis.

    Used for quick previews and Aladin FITS image loading.
    """
    h, w = image.shape[-2], image.shape[-1]
    scale = min(1.0, max_pixels / max(h, w))
    if scale < 1.0:
        new_h = max(1, int(h * scale))
        new_w = max(1, int(w * scale))
        img = F.interpolate(
            image.float().unsqueeze(0).unsqueeze(0),
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze()
    else:
        img = image.float()

    if vmin is None or vmax is None:
        sample = img.flatten()
        sample = sample[torch.isfinite(sample)]
        _vmin, _vmax = zscale_limits(sample) if sample.numel() > 0 else (0.0, 1.0)
        if vmin is None:
            vmin = _vmin
        if vmax is None:
            vmax = _vmax

    stretched = apply_stretch(img, vmin, vmax, stretch)
    return to_png(stretched, cmap=cmap)
