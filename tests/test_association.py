"""Probabilistic association: gate soundness, calibration, posteriors, assignment."""

from __future__ import annotations

import math

import pytest
import torch

from torchsky.catalogs import (
    Astrometry,
    Catalog,
    HistogramRatio,
    assign_one_to_one,
    associate,
)
from torchsky.catalogs._zones import unit_vectors
from torchsky.sphere import MOC
from torchsky.catalogs.association import gate_k2, mahalanobis2, pair_frame

D = torch.float64
MAS = math.radians(1.0 / 3.6e6)


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _offset(lon, lat, de_mas, dn_mas):
    """Exact tangent offsets (pole-safe) of unit vectors, returned in degrees."""
    v = unit_vectors(lon, lat)
    lam, phi = torch.deg2rad(lon), torch.deg2rad(lat)
    e = torch.stack((-torch.sin(lam), torch.cos(lam), torch.zeros_like(lam)), -1)
    n = torch.stack(
        (
            -torch.sin(phi) * torch.cos(lam),
            -torch.sin(phi) * torch.sin(lam),
            torch.cos(phi),
        ),
        -1,
    )
    w = v + MAS * (de_mas[:, None] * e + dn_mas[:, None] * n)
    w = w / w.norm(dim=-1, keepdim=True)
    return torch.rad2deg(torch.atan2(w[:, 1], w[:, 0])) % 360.0, torch.rad2deg(
        torch.asin(w[:, 2].clamp(-1, 1))
    )


def _random_cov(n: int, g: torch.Generator, lo: float, hi: float) -> torch.Tensor:
    a = lo * (hi / lo) ** torch.rand(n, generator=g, dtype=D)
    b = a * (0.2 + 0.8 * torch.rand(n, generator=g, dtype=D))
    pa = torch.rand(n, generator=g, dtype=D) * 180.0
    return Astrometry.from_ellipse(a, b, pa).cov


def test_pair_frame_arc_length_and_eigenvalues() -> None:
    g = _gen(1)
    lon = torch.tensor([10.0, 359.9999, 45.0, 120.0, 0.0], dtype=D)
    lat = torch.tensor([0.0, 0.0, 89.99999, -89.9999, 90.0], dtype=D)
    lon2, lat2 = _offset(
        lon,
        lat,
        torch.randn(5, generator=g, dtype=D) * 500,
        torch.randn(5, generator=g, dtype=D) * 500,
    )
    lx, rx = unit_vectors(lon, lat), unit_vectors(lon2, lat2)
    delta, tl, tr, psi = pair_frame(lx, rx)
    torch.testing.assert_close(delta.norm(dim=-1), psi / MAS, rtol=1e-10, atol=1e-9)
    # Transports are rotations: orthogonal 2x2 matrices.
    eye = torch.eye(2, dtype=D).expand(5, 2, 2)
    torch.testing.assert_close(tl @ tl.mT, eye, atol=1e-12, rtol=0)
    torch.testing.assert_close(tr @ tr.mT, eye, atol=1e-12, rtol=0)
    # Coincident points: identity transports and zero offset.
    d0, t0, _, p0 = pair_frame(lx, lx)
    assert float(p0.max()) < 1e-15 and float(d0.abs().max()) < 1e-6  # mas
    assert torch.allclose(t0[:4], eye[:4], atol=1e-12)


def test_mahalanobis_in_small_offsets_matches_local_tangent_plane() -> None:
    # For tiny separations the midpoint metric equals the left tangent-plane metric.
    g = _gen(2)
    n = 200
    lon = torch.rand(n, generator=g, dtype=D) * 360
    lat = torch.rand(n, generator=g, dtype=D) * 170 - 85
    de, dn = (
        torch.randn(n, generator=g, dtype=D) * 50,
        torch.randn(n, generator=g, dtype=D) * 50,
    )
    lon2, lat2 = _offset(lon, lat, de, dn)
    cov = _random_cov(n, g, 10.0, 100.0)[:, :2, :2]
    delta, tl, tr, _ = pair_frame(unit_vectors(lon, lat), unit_vectors(lon2, lat2))
    d2, _ = mahalanobis2(delta, tl @ cov @ tl.mT + tr @ cov @ tr.mT)
    ref, _ = mahalanobis2(torch.stack((de, dn), -1), 2 * cov)
    # The reference treats the right covariance in the left basis; the bases differ
    # by a rotation of ~psi tan(dec), hence a 1e-5 tolerance up to |dec| = 85 deg.
    torch.testing.assert_close(d2, ref, rtol=1e-5, atol=1e-9)


def _scene(
    seed: int, n_true: int, n_bg: int, *, region: str = "field", sigma=(20.0, 400.0)
):
    g = _gen(seed)
    if region == "pole":
        lon = torch.rand(n_true, generator=g, dtype=D) * 360
        lat = 90 - torch.rand(n_true, generator=g, dtype=D) * 0.05
    elif region == "wrap":
        lon = (torch.rand(n_true, generator=g, dtype=D) - 0.5) * 0.1 % 360
        lat = (torch.rand(n_true, generator=g, dtype=D) - 0.5) * 0.1
    else:
        lon = 150 + torch.rand(n_true, generator=g, dtype=D) * 0.1
        lat = 2 + torch.rand(n_true, generator=g, dtype=D) * 0.1
    cl = _random_cov(n_true, g, *sigma)
    cr = _random_cov(n_true + n_bg, g, *sigma)

    # Draw observed positions from the true ones with each side's covariance.
    def draw(lon0, lat0, cov):
        z = torch.randn(len(lon0), 2, generator=g, dtype=D)
        off = (torch.linalg.cholesky(cov[:, :2, :2]) @ z[:, :, None])[:, :, 0]
        return _offset(lon0, lat0, off[:, 0], off[:, 1])

    llon, llat = draw(lon, lat, cl)
    bg_lon = (
        lon.min() + (lon.max() - lon.min()) * torch.rand(n_bg, generator=g, dtype=D)
        if region != "pole"
        else torch.rand(n_bg, generator=g, dtype=D) * 360
    )
    bg_lat = lat.min() + (lat.max() - lat.min()) * torch.rand(
        n_bg, generator=g, dtype=D
    )
    rlon0 = torch.cat((lon, bg_lon % 360))
    rlat0 = torch.cat((lat, bg_lat))
    rlon, rlat = draw(rlon0, rlat0, cr)
    left = Catalog(lon=llon, lat=llat, epoch=2016.0, astrometry=Astrometry(cl))
    right = Catalog(lon=rlon, lat=rlat, epoch=2016.0, astrometry=Astrometry(cr))
    return left, right


@pytest.mark.parametrize("region", ["field", "pole", "wrap"])
def test_gate_is_sound_against_brute_force(region: str) -> None:
    left, right = _scene(3, 300, 300, region=region)
    res = associate(
        left,
        right,
        gate_pvalue=1e-3,
        max_radius_arcsec=30.0,
        completeness=0.9,
        assign="none",
    )
    lx, rx = unit_vectors(left.lon, left.lat), unit_vectors(right.lon, right.lat)
    n, m = len(left), len(right)
    li = torch.arange(n).repeat_interleave(m)
    rj = torch.arange(m).repeat(n)
    delta, tl, tr, psi = pair_frame(lx[li], rx[rj])
    cl, cr = left.astrometry.cov[:, :2, :2], right.astrometry.cov[:, :2, :2]  # type: ignore[union-attr]
    d2, _ = mahalanobis2(delta, tl @ cl[li] @ tl.mT + tr @ cr[rj] @ tr.mT)
    inside = (d2 <= gate_k2(1e-3)) & (psi * 206264.80624709636 <= 30.0)
    want = set(zip(li[inside].tolist(), rj[inside].tolist()))
    got = set(zip(res.left_index.tolist(), res.right_index.tolist()))
    assert want == got
    assert len(want) > 100


def test_true_pairs_follow_chi2_and_gate_loss_equals_pvalue() -> None:
    left, right = _scene(5, 6000, 0)
    res = associate(
        left,
        right,
        gate_pvalue=0.05,
        max_radius_arcsec=60.0,
        completeness=1.0,
        assign="none",
    )
    true = res.left_index == res.right_index
    found = int(true.sum())
    # Fraction of true pairs lost by the gate ~ 5% (binomial 4-sigma band).
    lost = 1.0 - found / 6000
    assert abs(lost - 0.05) < 4 * math.sqrt(0.05 * 0.95 / 6000)
    # Gated true d^2 follows chi^2_2 truncated at k^2: P(d^2 <= x) = (1 - e^{-x/2}) / 0.95.
    d2 = res.d2[true].sort().values
    cdf = (1 - torch.exp(-d2 / 2)) / 0.95
    emp = torch.arange(1, found + 1, dtype=D) / found
    assert float((cdf - emp).abs().max()) < 1.63 / math.sqrt(found)  # KS 99%


def test_posteriors_are_calibrated_and_em_recovers_completeness() -> None:
    # 70% of left rows have a counterpart; background is dense enough to confuse.
    g = _gen(7)
    left, right = _scene(7, 4000, 12000, sigma=(150.0, 600.0))
    keep = torch.rand(4000, generator=g) < 0.7
    drop = torch.nonzero(~keep, as_tuple=True)[0]
    mask = torch.ones(len(right), dtype=torch.bool)
    mask[drop] = False
    right_rows = torch.nonzero(mask, as_tuple=True)[0]
    right = right.take(right_rows)
    # The survey footprint (the box the background was drawn in) sets the density.
    box = MOC.from_polygon(
        [150.0, 150.1, 150.1, 150.0], [2.0, 2.0, 2.1, 2.1], max_order=16
    )
    res = associate(
        left,
        right,
        gate_pvalue=1e-4,
        max_radius_arcsec=30.0,
        completeness="em",
        assign="none",
        right_footprint=box,
    )
    assert abs(res.completeness - 0.7) < 0.03
    truth = right_rows[res.right_index] == res.left_index
    # Reliability: in posterior bins, the fraction of true pairs equals the mean posterior.
    p = res.posterior.detach()
    bins = torch.clamp((p * 10).long(), max=9)
    ece = 0.0
    for b in range(10):
        sel = bins == b
        if int(sel.sum()) >= 30:
            ece += float(sel.float().mean()) * abs(
                float(p[sel].mean()) - float(truth[sel].float().mean())
            )
    assert ece < 0.03
    # Every left row's probabilities sum to one.
    total = torch.zeros(len(left), dtype=D).index_add(0, res.left_index, p) + res.p_none
    torch.testing.assert_close(total, torch.ones(len(left), dtype=D))


def test_magnitude_prior_improves_purity() -> None:
    g = _gen(9)
    left, right = _scene(9, 3000, 9000, sigma=(300.0, 900.0))
    n_true = 3000
    mag = torch.cat(
        (
            16 + torch.randn(n_true, generator=g, dtype=D) * 0.5,
            20 + torch.randn(9000, generator=g, dtype=D) * 1.0,
        )
    )
    edges = [torch.linspace(12, 26, 29, dtype=D)]
    plain = associate(
        left, right, gate_pvalue=1e-4, max_radius_arcsec=30.0, assign="optimal"
    )
    photo = associate(
        left,
        right,
        gate_pvalue=1e-4,
        max_radius_arcsec=30.0,
        assign="optimal",
        terms=(HistogramRatio(mag, edges),),
    )

    def purity(res):
        m = res.match
        return float((res.left_index[m] == res.right_index[m]).float().mean()), int(
            m.sum()
        )

    p0, n0 = purity(plain)
    p1, n1 = purity(photo)
    assert p1 > p0 + 0.02 and n1 >= 0.9 * n0


def test_optimal_assignment_matches_scipy() -> None:
    scipy_opt = pytest.importorskip("scipy.optimize")
    g = _gen(11)
    for trial in range(20):
        nl, nr = 7, 6
        dense = (
            torch.rand(nl, nr, generator=g, dtype=D) * 4 - 1
        )  # some negative weights
        present = torch.rand(nl, nr, generator=g) < 0.6
        li, rj = torch.nonzero(present, as_tuple=True)
        w = dense[li, rj]
        sel = assign_one_to_one(li, rj, w, n_left=nl, n_right=nr)
        got = float(w[sel].sum())
        # scipy: absent and non-positive edges cost 0 (stay unmatched).
        cost = -torch.where(
            present & (dense > 0), dense, torch.zeros_like(dense)
        ).numpy()
        r, c = scipy_opt.linear_sum_assignment(cost)
        assert abs(got + float(cost[r, c].sum())) < 1e-9
        # One-to-one.
        assert (
            len(set(li[sel].tolist())) == int(sel.sum()) == len(set(rj[sel].tolist()))
        )


def test_assignment_star_components_take_best_edge() -> None:
    li = torch.tensor([0, 0, 0, 1, 2, 3])
    rj = torch.tensor([0, 1, 2, 3, 3, 4])
    w = torch.tensor([0.5, 2.0, 1.0, 0.3, 0.9, -1.0], dtype=D)
    sel = assign_one_to_one(li, rj, w)
    assert sel.tolist() == [False, True, False, False, True, False]


def test_associate_requires_astrometry() -> None:
    cat = Catalog(lon=torch.tensor([1.0]), lat=torch.tensor([0.0]))
    with pytest.raises(ValueError, match="astrometry"):
        associate(cat, cat)
