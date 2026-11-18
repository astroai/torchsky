"""Astrometric covariance containers, motion priors, and propagation."""

from __future__ import annotations

import math

import pytest
import torch

from torchsky.catalogs import Astrometry, Catalog, MotionPrior, propagate_astrometry
from torchsky.wcs.catalog_state import CatalogState

D = torch.float64


def _gaia_columns(n: int, seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    cols = {
        f"{k}_error": 0.02 + 0.2 * torch.rand(n, generator=g, dtype=D)
        for k in ("ra", "dec", "parallax", "pmra", "pmdec")
    }
    names = ("ra", "dec", "parallax", "pmra", "pmdec")
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            cols[f"{a}_{b}_corr"] = 0.6 * (torch.rand(n, generator=g, dtype=D) - 0.5)
    return cols


def test_gaia_covariance_layout_and_levels() -> None:
    cols = _gaia_columns(4)
    cols["parallax_error"][3] = math.nan  # two-parameter solution
    cols["pmra_error"][3] = math.nan
    ast = Astrometry.from_gaia(
        cols, radial_velocity_error=torch.tensor([1.0, math.nan, 2.0, 1.0], dtype=D)
    )
    c = ast.cov
    assert torch.allclose(c[:, 0, 0], cols["ra_error"] ** 2)
    assert torch.allclose(
        c[0, 4, 2],
        cols["parallax_pmra_corr"][0]
        * cols["parallax_error"][0]
        * cols["pmra_error"][0],
    )
    assert torch.allclose(
        c[0, 0, 1], cols["ra_dec_corr"][0] * cols["ra_error"][0] * cols["dec_error"][0]
    )
    assert ast.level.tolist() == [6, 5, 6, 2]
    assert torch.isnan(c[3, 2:, :]).all() and torch.isfinite(c[3, :2, :2]).all()


def test_ellipse_covariance_axes() -> None:
    ast = Astrometry.from_ellipse(
        torch.tensor([3.0, 3.0]), torch.tensor([1.0, 1.0]), torch.tensor([0.0, 90.0])
    )
    pos = ast.position_cov()
    torch.testing.assert_close(
        pos[0], torch.tensor([[1.0, 0.0], [0.0, 9.0]], dtype=D)
    )  # major axis north
    torch.testing.assert_close(
        pos[1], torch.tensor([[9.0, 0.0], [0.0, 1.0]], dtype=D)
    )  # major axis east


def test_invalid_covariances_fail_closed() -> None:
    with pytest.raises(ValueError, match="positive definite"):
        Astrometry.from_errors(1.0, 1.0, 1.5, n=1)
    with pytest.raises(ValueError, match="finite position"):
        Astrometry.from_errors(torch.tensor([math.nan]), 1.0)
    cov = torch.full((1, 6, 6), math.nan, dtype=D)
    cov[0, :2, :2] = torch.tensor([[1.0, 0.2], [0.3, 1.0]])
    with pytest.raises(ValueError, match="symmetric"):
        Astrometry(cov)


def test_propagated_covariance_matches_monte_carlo() -> None:
    # Gaia-like 5-parameter star, 26 yr back: J C J^T vs the sample covariance
    # of exactly propagated draws (Wishart 99.9% band on each entry).
    cols = _gaia_columns(1, seed=3)
    for k in ("pmra", "pmdec"):
        cols[f"{k}_error"] = torch.tensor([0.5], dtype=D)
    ast = Astrometry.from_gaia(cols)
    state = CatalogState(
        lon=torch.tensor([40.0], dtype=D),
        lat=torch.tensor([89.9], dtype=D),
        epoch=2016.0,
        pm_lon_coslat_mas_yr=torch.tensor([-798.6], dtype=D),
        pm_lat_mas_yr=torch.tensor([10328.1], dtype=D),
        parallax_mas=torch.tensor([547.0], dtype=D),
    )
    moved, cov = propagate_astrometry(state, ast, 1990.0)
    n = 40_000
    g = torch.Generator().manual_seed(4)
    c4 = ast.cov[0, :4, :4]
    draws = torch.randn(n, 4, generator=g, dtype=D) @ torch.linalg.cholesky(c4).mT
    mas = math.radians(1.0 / 3.6e6)
    lat0 = torch.deg2rad(state.lat)
    lon0 = torch.deg2rad(state.lon)
    p = torch.stack(
        [
            torch.cos(lat0) * torch.cos(lon0),
            torch.cos(lat0) * torch.sin(lon0),
            torch.sin(lat0),
        ],
        -1,
    )
    e = torch.stack([-torch.sin(lon0), torch.cos(lon0), torch.zeros_like(lon0)], -1)
    nn = torch.linalg.cross(p, e)
    q = p + mas * (draws[:, :1] * e + draws[:, 1:2] * nn)
    q = q / q.norm(dim=-1, keepdim=True)
    lon_s = torch.rad2deg(torch.atan2(q[:, 1], q[:, 0]))
    lat_s = torch.rad2deg(torch.asin(q[:, 2]))
    sample = CatalogState(
        lon=lon_s,
        lat=lat_s,
        epoch=2016.0,
        pm_lon_coslat_mas_yr=-798.6 + draws[:, 2],
        pm_lat_mas_yr=10328.1 + draws[:, 3],
    ).propagate_angular(1990.0)
    lon1, lat1 = torch.deg2rad(moved.lon), torch.deg2rad(moved.lat)
    p1 = torch.stack(
        [
            torch.cos(lat1) * torch.cos(lon1),
            torch.cos(lat1) * torch.sin(lon1),
            torch.sin(lat1),
        ],
        -1,
    )
    e1 = torch.stack([-torch.sin(lon1), torch.cos(lon1), torch.zeros_like(lon1)], -1)
    n1 = torch.linalg.cross(p1, e1)
    ls, bs = torch.deg2rad(sample.lon), torch.deg2rad(sample.lat)
    qs = torch.stack(
        [torch.cos(bs) * torch.cos(ls), torch.cos(bs) * torch.sin(ls), torch.sin(bs)],
        -1,
    )
    off = torch.stack([(qs * e1).sum(-1), (qs * n1).sum(-1)], -1) / mas
    emp = torch.cov(off.T)
    sd = torch.sqrt((cov[0] ** 2 + torch.outer(cov[0].diag(), cov[0].diag())) / n)
    assert ((emp - cov[0]).abs() <= 3.3 * sd).all()


def test_rows_without_motion_need_a_prior_and_widen_by_dt_squared() -> None:
    ast = Astrometry.from_errors(torch.tensor([10.0, 10.0]), torch.tensor([10.0, 10.0]))
    state = CatalogState(
        lon=torch.tensor([10.0, 20.0], dtype=D),
        lat=torch.zeros(2, dtype=D),
        epoch=2000.0,
    )
    with pytest.raises(ValueError, match="prior"):
        propagate_astrometry(state, ast, 2016.0)
    prior = MotionPrior.constant(
        (2.0, -1.0), torch.tensor([[25.0, 0.0], [0.0, 16.0]], dtype=D)
    )
    moved, cov = propagate_astrometry(state, ast, 2016.0, motion_prior=prior)
    torch.testing.assert_close(
        cov[0],
        torch.tensor([[100.0 + 256 * 25.0, 0.0], [0.0, 100.0 + 256 * 16.0]], dtype=D),
    )
    assert moved.lon[0] > 10.0  # moved east by the prior mean


def test_prior_from_reference_pools_sparse_cells() -> None:
    g = torch.Generator().manual_seed(5)
    n = 5000
    lon = torch.rand(n, generator=g, dtype=D) * 20.0
    lat = torch.rand(n, generator=g, dtype=D) * 20.0
    pe = 3.0 + 2.0 * torch.randn(n, generator=g, dtype=D)
    pn = -1.0 + 1.0 * torch.randn(n, generator=g, dtype=D)
    prior = MotionPrior.from_reference(lon, lat, pe, pn, order=3, min_count=50)
    mean, cov = prior.at(
        torch.tensor([10.0, 200.0], dtype=D), torch.tensor([10.0, -60.0], dtype=D)
    )
    assert torch.allclose(mean[0], torch.tensor([3.0, -1.0], dtype=D), atol=0.5)
    # An empty sky cell falls back to the whole-sky moments.
    torch.testing.assert_close(
        mean[1], torch.stack([pe.mean(), pn.mean()]), rtol=0, atol=1e-12
    )
    assert torch.allclose(cov[1].diag(), torch.tensor([4.0, 1.0], dtype=D), rtol=0.1)


def test_catalog_carries_astrometry_through_take() -> None:
    ast = Astrometry.from_errors(
        torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 3.0])
    )
    cat = Catalog(lon=torch.tensor([1.0, 2.0, 3.0]), lat=torch.zeros(3), astrometry=ast)
    sub = cat.take(torch.tensor([2, 0]))
    assert sub.astrometry is not None
    assert sub.astrometry.cov[:, 0, 0].tolist() == [9.0, 1.0]
    with pytest.raises(ValueError, match="one covariance per row"):
        Catalog(lon=torch.tensor([1.0]), lat=torch.zeros(1), astrometry=ast)
