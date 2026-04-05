"""Spherical harmonic map synthesis / analysis smoke (torchsky sphere stack)."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("torchsky.cpp")

from torchsky.sphere.spectral import alm2map, map2alm  # noqa: E402
from torchsky.wcs import healpix as hp  # noqa: E402


def test_map2alm_alm2map_pipeline_finite_torch_backend() -> None:
    """Band-limit analysis/synthesis runs on CPU without NaNs (exact round-trip is backend-specific)."""
    nside = 16
    lmax = 3 * nside - 1
    torch.manual_seed(0)
    m = torch.randn(hp.nside2npix(nside), dtype=torch.float64)
    alm = map2alm(m, nside=nside, lmax=lmax, iter=0, backend="torch")
    m2 = alm2map(alm, nside=nside, lmax=lmax, backend="torch")
    assert torch.isfinite(alm).all()
    assert torch.isfinite(m2).all()
    assert m2.shape == m.shape
