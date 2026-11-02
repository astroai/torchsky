"""Quadrature and pixel-weight helpers for HEALPix map2alm."""

import os
from pathlib import Path

import torch
from torch import Tensor

from torchfits import read_tensor as read_image
from torchfits.table import read_torch as read_table

from . import healpix as _healpix
from ._spectral_cache import SpectralCacheNamespace

from ._spectral_config import (
    _SPECTRAL_CACHE,
)
from ._spectral_kernels import (
    _ring_pix2ring_index,
)


_RING_WEIGHT_CACHE: SpectralCacheNamespace[Tensor] = _SPECTRAL_CACHE.namespace(
    "ring_weight", device="cpu", dtype=torch.float64
)


_PIXEL_WEIGHT_CACHE: SpectralCacheNamespace[Tensor] = _SPECTRAL_CACHE.namespace(
    "pixel_weight", device="cpu", dtype=torch.float64
)


_HEALPIX_FULL_WEIGHT_REL = "full_weights/healpix_full_weights_nside_%04d.fits"


def _even_legendre_p(z: Tensor, n_even: int) -> Tensor:
    """``P_0, P_2, ..., P_{2*(n_even-1)}`` via Bonnet recurrence, shape ``(n_even, N)``."""
    if n_even < 1:
        raise ValueError("n_even must be positive")
    one = torch.ones_like(z)
    rows = [one]
    if n_even == 1:
        return torch.stack(rows, dim=0)
    p0 = one
    p1 = z
    for n in range(1, 2 * (n_even - 1) + 1):
        p2 = ((2 * n + 1) * z * p1 - n * p0) / (n + 1)
        p0 = p1
        p1 = p2
        if n % 2 == 1:
            rows.append(p2)
    return torch.stack(rows, dim=0)


def _healpix_ring_quadrature_weights(nside: int) -> Tensor:
    """Per-pixel RING quadrature weights with mean 1.

    Tikhonov-regularized even-Legendre moment matching through
    ``l = 2*min(nside, 128)-2`` (Gorski et al. 2005 equal-area rings).
    ``use_weights`` applies this quadrature. Pixel-weight FITS files are a
    separate product loaded only when ``use_pixel_weights=True``.
    """
    nside_i = int(nside)
    n_even = min(nside_i, 128)
    key = (nside_i, n_even)

    def build() -> Tensor:
        npix = _healpix.nside2npix(nside_i)
        pix = torch.arange(npix, dtype=torch.int64)
        ring = _ring_pix2ring_index(nside_i)
        nring = int(ring.max().item()) + 1
        counts = torch.bincount(ring, minlength=nring).to(torch.float64)
        theta, _ = _healpix.pix2ang(nside_i, pix, nest=False)
        z = torch.cos(theta.to(dtype=torch.float64))
        z_sum = torch.zeros(nring, dtype=torch.float64)
        z_sum.scatter_add_(0, ring, z)
        z_ring = z_sum / counts.clamp_min(1.0)
        design = _even_legendre_p(z_ring, n_even) * counts.unsqueeze(0)
        target = torch.zeros(n_even, dtype=torch.float64)
        target[0] = float(npix)
        ones = torch.ones(nring, dtype=torch.float64)
        ridge = 1.0e-6 * float(npix)
        gram = design @ design.T + ridge * torch.eye(n_even, dtype=torch.float64)
        coeff = torch.linalg.solve(gram, target - design @ ones)
        ring_w = ones + design.T @ coeff
        pix_w = ring_w[ring]
        return pix_w / pix_w.mean().clamp_min(1.0e-30)

    return _RING_WEIGHT_CACHE.get_or_build(key, build)


def _resolve_healpix_datapath(
    datapath: str | os.PathLike[str] | None,
) -> Path | None:
    if datapath is not None:
        return Path(datapath)
    env = os.environ.get("TORCHSKY_HEALPIX_DATAPATH")
    if env:
        return Path(env)
    return None


def _healpix_full_weight_path(nside: int, datapath: Path) -> Path:
    if datapath.is_file():
        return datapath
    return datapath / (_HEALPIX_FULL_WEIGHT_REL % int(nside))


def _read_healpix_full_weight_vector(path: str, nside: int) -> Tensor:
    """Load HEALPix full-pixel weight *corrections* (applied as ``1 + w``)."""
    npix = _healpix.nside2npix(nside)
    candidates: list[Tensor] = []
    for hdu in (1, 0):
        try:
            data = torch.as_tensor(read_image(path, hdu=hdu, mmap=False)).reshape(-1)
        except (OSError, ValueError, RuntimeError, IndexError, KeyError):
            continue
        if data.numel() > 0:
            candidates.append(data)
    if not candidates:
        try:
            table = read_table(path)
        except (OSError, ValueError, RuntimeError, IndexError, KeyError) as exc:
            raise ValueError(
                f"could not read HEALPix pixel weights from {path!r}"
            ) from exc
        if not hasattr(table, "keys"):
            raise ValueError(f"pixel weight table in {path!r} has no columns")
        first = next(iter(table.keys()))
        candidates.append(torch.as_tensor(table[first]).reshape(-1))
    data = candidates[0].to(dtype=torch.float64)
    if data.numel() == npix:
        return data
    if data.numel() == 3 * npix:
        return data.reshape(3, npix)[0]
    raise ValueError(
        f"pixel weights in {path!r} have {data.numel()} values; expected {npix}"
    )


def _healpix_full_pixel_weight_corrections(
    nside: int, datapath: str | os.PathLike[str] | None
) -> Tensor:
    root = _resolve_healpix_datapath(datapath)
    if root is None:
        raise FileNotFoundError(
            "use_pixel_weights=True requires datapath or TORCHSKY_HEALPIX_DATAPATH "
            "pointing at a healpy-data tree "
            "(full_weights/healpix_full_weights_nside_NNNN.fits). "
            "torchsky does not download or vendor those files"
        )
    path = _healpix_full_weight_path(nside, root)
    key = (int(nside), str(path.resolve()) if path.exists() else str(path))

    def build() -> Tensor:
        if not path.is_file():
            raise FileNotFoundError(
                f"pixel weights file missing at {path}. "
                "Set datapath to a local healpy-data checkout or pass use_pixel_weights=False"
            )
        weights = _read_healpix_full_weight_vector(str(path), int(nside))
        if not bool(torch.isfinite(weights).all().item()):
            raise ValueError(f"pixel weights in {path} are not finite")
        return weights

    return _PIXEL_WEIGHT_CACHE.get_or_build(key, build)


def _gal_cut_keep_mask(
    nside: int, gal_cut_deg: float, *, nest: bool, device: torch.device
) -> Tensor:
    cut = float(gal_cut_deg)
    if cut < 0.0:
        raise ValueError("gal_cut must be non-negative")
    npix = _healpix.nside2npix(nside)
    pix = torch.arange(npix, device=device, dtype=torch.int64)
    _lon, lat = _healpix.pix2ang(nside, pix, nest=nest, lonlat=True)
    return torch.abs(lat) >= cut


def _apply_map2alm_weights(
    rows: Tensor,
    *,
    nside: int,
    gal_cut: float,
    use_weights: bool,
    use_pixel_weights: bool,
    datapath: str | os.PathLike[str] | None = None,
) -> Tensor:
    """Apply galactic cut, RING quadrature, or HEALPix full-pixel weights."""
    if use_weights and use_pixel_weights:
        raise ValueError("use_weights and use_pixel_weights are mutually exclusive")
    out = rows
    if float(gal_cut) > 0.0:
        keep = _gal_cut_keep_mask(nside, float(gal_cut), nest=False, device=rows.device)
        out = out * keep.to(dtype=out.dtype)
    if use_pixel_weights:
        corr = _healpix_full_pixel_weight_corrections(nside, datapath).to(
            device=rows.device, dtype=out.dtype
        )
        out = out * (1.0 + corr).unsqueeze(0)
    elif use_weights:
        w = _healpix_ring_quadrature_weights(nside).to(
            device=rows.device, dtype=out.dtype
        )
        out = out * w.unsqueeze(0)
    return out
