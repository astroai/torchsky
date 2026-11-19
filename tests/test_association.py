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


# ------------------------------------------------------------------- N-way


def _triple_scene(seed: int, n: int, n_bg: int, keep=(0.8, 0.6)):
    g = _gen(seed)
    lon = 150 + torch.rand(n, generator=g, dtype=D) * 0.1
    lat = 2 + torch.rand(n, generator=g, dtype=D) * 0.1
    cats, truth = [], []
    for c, frac in enumerate((1.0,) + tuple(keep)):
        cov = _random_cov(n + n_bg, g, 80.0, 500.0)
        has = torch.rand(n, generator=g) < frac
        src = torch.nonzero(has, as_tuple=True)[0]
        bg_lon = 150 + torch.rand(n_bg if c else 0, generator=g, dtype=D) * 0.1
        bg_lat = 2 + torch.rand(n_bg if c else 0, generator=g, dtype=D) * 0.1
        lon0 = torch.cat((lon[src], bg_lon))
        lat0 = torch.cat((lat[src], bg_lat))
        cov = cov[: len(lon0)]
        z = torch.randn(len(lon0), 2, generator=g, dtype=D)
        off = (torch.linalg.cholesky(cov[:, :2, :2]) @ z[:, :, None])[:, :, 0]
        olon, olat = _offset(lon0, lat0, off[:, 0], off[:, 1])
        cats.append(
            Catalog(lon=olon, lat=olat, epoch=2016.0, astrometry=Astrometry(cov))
        )
        t = torch.full((n,), -1, dtype=torch.int64)
        t[src] = torch.arange(len(src))
        truth.append(t)
    box = MOC.from_polygon(
        [150.0, 150.1, 150.1, 150.0], [2.0, 2.0, 2.1, 2.1], max_order=16
    )
    return cats, truth, box


def test_nway_with_two_catalogues_equals_pairwise() -> None:
    from torchsky.catalogs import associate_many

    cats, _, box = _triple_scene(21, 800, 2000, keep=(0.7,))
    pair = associate(
        cats[0],
        cats[1],
        completeness=0.7,
        assign="none",
        right_footprint=box,
        max_radius_arcsec=30,
    )
    many = associate_many(
        cats, completeness=[0.7], footprints=[box, box], max_radius_arcsec=30
    )
    present = many.members[:, 1] >= 0
    key_many = many.members[present, 0] * 10**7 + many.members[present, 1]
    key_pair = pair.left_index * 10**7 + pair.right_index
    order_m, order_p = torch.argsort(key_many), torch.argsort(key_pair)
    assert torch.equal(key_many[order_m], key_pair[order_p])
    torch.testing.assert_close(
        many.posterior[present][order_m], pair.posterior[order_p], rtol=1e-9, atol=1e-12
    )
    torch.testing.assert_close(1 - many.p_any, pair.p_none, rtol=1e-9, atol=1e-12)


def test_nway_marginal_matches_numerical_integration() -> None:
    from torchsky.catalogs.association import _hypothesis_log_weight

    g = _gen(23)
    covs = [_random_cov(1, g, 50.0, 200.0)[0, :2, :2] for _ in range(3)]
    xs = [
        torch.zeros(2, dtype=D),
        torch.tensor([60.0, -40.0], dtype=D),
        torch.tensor([-30.0, 90.0], dtype=D),
    ]
    members = torch.tensor([[0, 0, 0]])
    lw = _hypothesis_log_weight(
        members,
        [x[None] for x in xs],
        [torch.linalg.inv(c)[None] for c in covs],
        [torch.logdet(c)[None] for c in covs],
        torch.zeros(3, dtype=D),
        torch.zeros(3, dtype=D),
        [torch.zeros(1, dtype=D)] * 3,
    )
    # M = integral over y of prod_k N(x_k; y, C_k) dy / N(x_0; y, C_0) normalisation: the primary
    # density integrates the flat prior, so M = int prod_k N(x_k; y, C_k) dy.
    grid = torch.linspace(-600, 600, 1201, dtype=D)
    yy = torch.stack(torch.meshgrid(grid, grid, indexing="ij"), -1).reshape(-1, 2)
    dens = torch.ones(yy.shape[0], dtype=D)
    for x, c in zip(xs, covs):
        d = x - yy
        ci = torch.linalg.inv(c)
        dens = (
            dens
            * torch.exp(-0.5 * (d @ ci * d).sum(-1))
            / (2 * math.pi * torch.sqrt(torch.det(c)))
        )
    integral = float(dens.sum()) * float(grid[1] - grid[0]) ** 2
    assert abs(float(lw[0]) - math.log(integral)) < 1e-4


def test_nway_best_first_matches_exact_within_certified_residual() -> None:
    from torchsky.catalogs import associate_many

    cats, _, box = _triple_scene(25, 200, 4000)
    exact = associate_many(
        cats, completeness=[0.8, 0.6], footprints=[box] * 3, max_radius_arcsec=30
    )
    approx = associate_many(
        cats,
        completeness=[0.8, 0.6],
        footprints=[box] * 3,
        max_radius_arcsec=30,
        budget=2,
        mass_tol=1e-8,
    )
    assert float(exact.residual_mass.max()) == 0.0
    assert float(approx.residual_mass.max()) <= 1e-6
    # The MAP hypothesis is the same and its posterior agrees within the residual.
    for i in range(200):
        a = exact.members[exact.best[i]]
        b = approx.members[approx.best[i]]
        assert torch.equal(a, b)
        diff = abs(
            float(exact.posterior[exact.best[i]])
            - float(approx.posterior[approx.best[i]])
        )
        assert diff <= float(approx.residual_mass[i]) + 1e-9


def test_nway_recovers_injected_triples() -> None:
    from torchsky.catalogs import associate_many

    cats, truth, box = _triple_scene(27, 1500, 3000)
    res = associate_many(cats, footprints=[box] * 3, max_radius_arcsec=30)
    assert (
        abs(res.completeness[1] - 0.8) < 0.05 and abs(res.completeness[2] - 0.6) < 0.05
    )
    best = res.members[res.best]
    correct = (best[:, 1] == truth[1]) & (best[:, 2] == truth[2])
    assert float(correct.float().mean()) > 0.9
    # p_any is calibrated: the mean p_any matches the true fraction with any counterpart.
    has_any = (truth[1] >= 0) | (truth[2] >= 0)
    assert abs(float(res.p_any.mean()) - float(has_any.float().mean())) < 0.03


# ------------------------------------------------- pair epochs, calibration


def test_pair_epochs_are_sound_and_recover_fast_movers() -> None:
    from torchsky.catalogs import propagate_astrometry
    from torchsky.wcs.catalog_state import CatalogState

    g = _gen(31)
    n = 300
    lon = 150 + torch.rand(n, generator=g, dtype=D) * 0.05
    lat = 2 + torch.rand(n, generator=g, dtype=D) * 0.05
    pm_e = torch.randn(n, generator=g, dtype=D) * 50
    pm_n = torch.randn(n, generator=g, dtype=D) * 50
    pm_e[0], pm_n[0] = -798.6, 10328.1  # Barnard-like
    cov = Astrometry.from_errors(
        torch.full((n,), 0.1, dtype=D), torch.full((n,), 0.1, dtype=D)
    ).cov
    cov[:, 2, 2] = cov[:, 3, 3] = 0.05**2
    cov[:, 2:4, :2] = cov[:, :2, 2:4] = 0.0
    cov[:, 2, 3] = cov[:, 3, 2] = 0.0
    left = Catalog(
        lon=lon,
        lat=lat,
        epoch=2016.0,
        pm_lon_coslat_mas_yr=pm_e,
        pm_lat_mas_yr=pm_n,
        astrometry=Astrometry(cov),
    )
    t_r = 1997.0 + 4.0 * torch.rand(n, generator=g, dtype=D)
    state = CatalogState(
        lon=lon, lat=lat, epoch=2016.0, pm_lon_coslat_mas_yr=pm_e, pm_lat_mas_yr=pm_n
    )
    true_pos = state.propagate(t_r)
    err = torch.randn(n, 2, generator=g, dtype=D) * 70.0
    rlon, rlat = _offset(true_pos.lon, true_pos.lat, err[:, 0], err[:, 1])
    right = Catalog(
        lon=rlon,
        lat=rlat,
        epoch=t_r,
        astrometry=Astrometry.from_errors(
            torch.full((n,), 80.0, dtype=D), torch.full((n,), 80.0, dtype=D)
        ),
    )
    res = associate(
        left,
        right,
        gate_pvalue=1e-3,
        max_radius_arcsec=120.0,
        completeness=0.9,
        assign="optimal",
    )
    assert bool(res.match[(res.left_index == 0) & (res.right_index == 0)].any())
    # Brute force at pair epochs: every left row propagated to every right epoch.
    li = torch.arange(n).repeat_interleave(n)
    rj = torch.arange(n).repeat(n)
    gathered = CatalogState(
        lon=lon[li],
        lat=lat[li],
        epoch=2016.0,
        pm_lon_coslat_mas_yr=pm_e[li],
        pm_lat_mas_yr=pm_n[li],
    )
    moved, lc = propagate_astrometry(gathered, Astrometry(cov[li]), t_r[rj])
    delta, tl, tr, psi = pair_frame(
        unit_vectors(moved.lon, moved.lat), unit_vectors(rlon[rj], rlat[rj])
    )
    d2, _ = mahalanobis2(
        delta, tl @ lc @ tl.mT + tr @ right.astrometry.cov[rj, :2, :2] @ tr.mT
    )  # type: ignore[union-attr]
    inside = (d2 <= gate_k2(1e-3)) & (psi * 206264.80624709636 <= 120.0)
    assert set(zip(li[inside].tolist(), rj[inside].tolist())) == set(
        zip(res.left_index.tolist(), res.right_index.tolist())
    )


def test_calibration_recovers_error_scale_and_floor() -> None:
    g = _gen(33)
    n = 6000
    lon = 150 + torch.rand(n, generator=g, dtype=D) * 0.2
    lat = 2 + torch.rand(n, generator=g, dtype=D) * 0.2
    left_cov = Astrometry.from_errors(
        torch.full((n,), 20.0, dtype=D), torch.full((n,), 20.0, dtype=D)
    )
    pub = 100.0 + 100.0 * torch.rand(n, generator=g, dtype=D)  # published right errors
    true_sigma = 2.0 * pub  # truth: errors underestimated 2x ...
    # ... plus a systematic floor. It must be large enough to be identifiable
    # next to 200-400 mas errors: the statistical error on floor^2 from 6000
    # pairs is about 2e3 mas^2.
    floor = 150.0
    z = torch.randn(n, 2, generator=g, dtype=D)
    off = z * torch.sqrt(true_sigma**2 + floor**2 + 20.0**2)[:, None]
    rlon, rlat = _offset(lon, lat, off[:, 0], off[:, 1])
    left = Catalog(lon=lon, lat=lat, epoch=2016.0, astrometry=left_cov)
    right = Catalog(
        lon=rlon, lat=rlat, epoch=2016.0, astrometry=Astrometry.from_errors(pub, pub)
    )
    raw = associate(
        left,
        right,
        gate_pvalue=1e-4,
        max_radius_arcsec=30.0,
        completeness=1.0,
        assign="none",
    )
    cal = associate(
        left,
        right,
        gate_pvalue=1e-4,
        max_radius_arcsec=30.0,
        completeness=1.0,
        assign="none",
        calibrate="scale_floor",
    )
    assert abs(cal.right_scale - 2.0) < 0.1
    assert (
        cal.floor_mas2 is not None
        and abs(float(cal.floor_mas2.diagonal().mean()) ** 0.5 - floor) < 25.0
    )
    true_raw = raw.left_index == raw.right_index
    true_cal = cal.left_index == cal.right_index
    assert int(true_cal.sum()) > int(true_raw.sum())  # the raw gate rejects true pairs
    assert abs(float(cal.d2[true_cal].mean()) - 2.0) < 0.15


def test_binned_calibration_recovers_per_population_scales() -> None:
    g = _gen(35)
    n = 8000
    lon = 150 + torch.rand(n, generator=g, dtype=D) * 0.3
    lat = 2 + torch.rand(n, generator=g, dtype=D) * 0.3
    pub = 100.0 + 100.0 * torch.rand(n, generator=g, dtype=D)
    blended = torch.rand(n, generator=g) < 0.4
    true_scale = torch.where(blended, 3.0, 1.0).to(D)
    z = torch.randn(n, 2, generator=g, dtype=D)
    off = z * torch.sqrt((true_scale * pub) ** 2 + 20.0**2)[:, None]
    rlon, rlat = _offset(lon, lat, off[:, 0], off[:, 1])
    left = Catalog(
        lon=lon,
        lat=lat,
        epoch=2016.0,
        astrometry=Astrometry.from_errors(
            torch.full((n,), 20.0, dtype=D), torch.full((n,), 20.0, dtype=D)
        ),
    )
    right = Catalog(
        lon=rlon, lat=rlat, epoch=2016.0, astrometry=Astrometry.from_errors(pub, pub)
    )
    res = associate(
        left,
        right,
        gate_pvalue=1e-4,
        max_radius_arcsec=30.0,
        completeness=1.0,
        assign="none",
        calibrate="scale_floor",
        calibrate_by=blended.to(D),
        calibrate_edges=torch.tensor([0.0, 0.5, 1.5]),
    )
    scale = res.right_scale
    assert isinstance(scale, torch.Tensor)
    assert abs(float(scale[0]) - 1.0) < 0.1 and abs(float(scale[1]) - 3.0) < 0.2
    true = res.left_index == res.right_index
    assert abs(float(res.d2[true].mean()) - 2.0) < 0.15


def test_pair_colour_term_resolves_blends() -> None:
    g = _gen(37)
    n = 3000
    lon = 150 + torch.rand(n, generator=g, dtype=D) * 0.3
    lat = 2 + torch.rand(n, generator=g, dtype=D) * 0.3
    gmag_true = 14 + 4 * torch.rand(n, generator=g, dtype=D)
    # A faint Gaia neighbour ~1.5" from every true counterpart (a blend).
    ang = torch.rand(n, generator=g, dtype=D) * 2 * math.pi
    nlon, nlat = _offset(lon, lat, 1500 * torch.cos(ang), 1500 * torch.sin(ang))
    gaia_lon = torch.cat((lon, nlon))
    gaia_lat = torch.cat((lat, nlat))
    gaia_g = torch.cat(
        (gmag_true, gmag_true + 2 + 2 * torch.rand(n, generator=g, dtype=D))
    )
    w1 = (
        gmag_true - 1.0 + 0.2 * torch.randn(n, generator=g, dtype=D)
    )  # W1 - G ~ -1 for the true star
    off = torch.randn(n, 2, generator=g, dtype=D) * 800.0  # large WISE errors
    wlon, wlat = _offset(lon, lat, off[:, 0], off[:, 1])
    gaia = Catalog(
        lon=gaia_lon,
        lat=gaia_lat,
        epoch=2016.0,
        astrometry=Astrometry.from_errors(
            torch.full((2 * n,), 30.0, dtype=D), torch.full((2 * n,), 30.0, dtype=D)
        ),
    )
    wise = Catalog(
        lon=wlon,
        lat=wlat,
        epoch=2016.0,
        astrometry=Astrometry.from_errors(
            torch.full((n,), 800.0, dtype=D), torch.full((n,), 800.0, dtype=D)
        ),
    )
    box = MOC.from_polygon(
        [150.0, 150.3, 150.3, 150.0], [2.0, 2.0, 2.3, 2.3], max_order=16
    )
    common = dict(
        gate_pvalue=1e-4, max_radius_arcsec=30.0, right_footprint=box, assign="optimal"
    )
    plain = associate(gaia, wise, **common)
    colour = associate(
        gaia,
        wise,
        terms=(
            HistogramRatio(
                w1, [torch.linspace(-6, 4, 41, dtype=D)], left_values=gaia_g
            ),
        ),
        **common,
    )

    def purity(res):
        m = res.match
        return float((res.left_index[m] == res.right_index[m]).float().mean())

    assert purity(colour) > purity(plain) + 0.1
    assert purity(colour) > 0.9


def test_nway_per_row_epoch_catalogue_recovers_fast_mover() -> None:
    from torchsky.catalogs import associate_many
    from torchsky.wcs.catalog_state import CatalogState

    g = _gen(41)
    n = 400
    lon = 150 + torch.rand(n, generator=g, dtype=D) * 0.05
    lat = 2 + torch.rand(n, generator=g, dtype=D) * 0.05
    pm_e = torch.randn(n, generator=g, dtype=D) * 30
    pm_n = torch.randn(n, generator=g, dtype=D) * 30
    pm_e[0], pm_n[0] = -798.6, 10328.1
    cov = Astrometry.from_errors(
        torch.full((n,), 0.1, dtype=D), torch.full((n,), 0.1, dtype=D)
    ).cov
    cov[:, 2, 2] = cov[:, 3, 3] = 0.05**2
    cov[:, 2:4, :2] = cov[:, :2, 2:4] = 0.0
    cov[:, 2, 3] = cov[:, 3, 2] = 0.0
    primary = Catalog(
        lon=lon,
        lat=lat,
        epoch=2016.0,
        pm_lon_coslat_mas_yr=pm_e,
        pm_lat_mas_yr=pm_n,
        astrometry=Astrometry(cov),
    )
    state = CatalogState(
        lon=lon, lat=lat, epoch=2016.0, pm_lon_coslat_mas_yr=pm_e, pm_lat_mas_yr=pm_n
    )
    cats = [primary]
    for seed, (t_lo, t_hi) in ((43, (1997.0, 2001.0)), (44, (2010.0, 2011.0))):
        gg = _gen(seed)
        t = t_lo + (t_hi - t_lo) * torch.rand(n, generator=gg, dtype=D)
        true_pos = state.propagate(t)
        err = torch.randn(n, 2, generator=gg, dtype=D) * 80.0
        rlon, rlat = _offset(true_pos.lon, true_pos.lat, err[:, 0], err[:, 1])
        cats.append(
            Catalog(
                lon=rlon,
                lat=rlat,
                epoch=t,
                astrometry=Astrometry.from_errors(
                    torch.full((n,), 80.0, dtype=D), torch.full((n,), 80.0, dtype=D)
                ),
            )
        )
    res = associate_many(cats, completeness=[0.95, 0.95], max_radius_arcsec=120.0)
    best = res.members[res.best]
    assert best[0, 1] == 0 and best[0, 2] == 0  # Barnard-like star found in both
    assert float((best[:, 1] == torch.arange(n)).float().mean()) > 0.97
    assert float((best[:, 2] == torch.arange(n)).float().mean()) > 0.97


# ------------------------------------------------------------ exclusivity


def _exact_matching_marginals(li, rj, log_w, log_a):
    """Brute-force marginals of the partial one-to-one matching model."""
    import itertools

    n = int(log_a.numel())
    edges = list(zip(li.tolist(), rj.tolist()))
    options = [
        [None] + [e for e, (i, _) in enumerate(edges) if i == r] for r in range(n)
    ]
    z = 0.0
    marg = [0.0] * len(edges)
    none = [0.0] * n
    for combo in itertools.product(*options):
        used = [edges[e][1] for e in combo if e is not None]
        if len(used) != len(set(used)):
            continue
        w = math.exp(
            sum(
                float(log_w[e]) if e is not None else float(log_a[i])
                for i, e in enumerate(combo)
            )
        )
        z += w
        for i, e in enumerate(combo):
            if e is None:
                none[i] += w
            else:
                marg[e] += w
    return torch.tensor(marg, dtype=D) / z, torch.tensor(none, dtype=D) / z


def _random_graph(g: torch.Generator, n_left: int, n_right: int, tree: bool):
    if tree:  # forest: each left joins at most one existing right
        li, rj, r = [], [], 0
        for i in range(n_left):
            if r and float(torch.rand(1, generator=g)) < 0.7:
                li.append(i)
                rj.append(int(torch.randint(0, r, (1,), generator=g)))
            for _ in range(int(torch.randint(0, 3, (1,), generator=g))):
                li.append(i)
                rj.append(r)
                r += 1
    else:
        m = torch.rand(n_left, n_right, generator=g) < 0.5
        li, rj = (x.tolist() for x in torch.nonzero(m, as_tuple=True))
    li_t, rj_t = (
        torch.tensor(li, dtype=torch.int64),
        torch.tensor(rj, dtype=torch.int64),
    )
    order = torch.argsort(li_t * 1000 + rj_t)
    return li_t[order], rj_t[order]


def test_exclusive_marginals_exact_on_trees_and_bounded_on_loops() -> None:
    from torchsky.catalogs._exclusive import pairwise_exclusive

    g = _gen(41)
    worst_loop = 0.0
    for trial in range(120):
        tree = trial % 2 == 0
        li, rj = _random_graph(g, 5, 4, tree)
        if li.numel() == 0:
            continue
        log_w = torch.randn(li.numel(), generator=g, dtype=D) * 2.0
        log_a = torch.log(torch.rand(5, generator=g, dtype=D))
        post, p_none, _ = pairwise_exclusive(li, rj, log_w, log_a)
        ref, ref_none = _exact_matching_marginals(li, rj, log_w, log_a)
        err = max(
            float((post - ref).abs().max()), float((p_none - ref_none).abs().max())
        )
        if tree:
            assert err < 1e-12  # Bethe is exact on cycle-free graphs
        else:
            worst_loop = max(worst_loop, err)
        # Both sides are normalised: rows sum to one, columns to at most one.
        rows = torch.zeros(5, dtype=D).index_add(0, li, post) + p_none
        torch.testing.assert_close(rows, torch.ones(5, dtype=D))
        cols = torch.zeros(int(rj.max()) + 1, dtype=D).index_add(0, rj, post)
        assert float(cols.max()) <= 1.0 + 1e-12
    # Dense 5x4 random graphs are the hard case for the Bethe approximation.
    assert worst_loop < 0.15


def test_exclusive_without_shared_rows_is_the_independent_posterior() -> None:
    from torchsky.catalogs._exclusive import pairwise_exclusive

    g = _gen(42)
    li = torch.tensor([0, 0, 1, 2, 2, 2], dtype=torch.int64)
    rj = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int64)
    log_w = torch.randn(6, generator=g, dtype=D)
    log_a = torch.log(torch.rand(3, generator=g, dtype=D))
    post, p_none, rho = pairwise_exclusive(li, rj, log_w, log_a)
    z = torch.exp(log_a).index_add(0, li, torch.exp(log_w))
    torch.testing.assert_close(post, torch.exp(log_w) / z[li])
    torch.testing.assert_close(p_none, torch.exp(log_a) / z)
    assert float(rho.abs().max()) == 0.0


def test_exclusive_gradient_matches_finite_differences() -> None:
    from torchsky.catalogs._exclusive import pairwise_exclusive

    g = _gen(43)
    li, rj = _random_graph(g, 5, 4, tree=False)
    log_a = torch.log(torch.rand(5, generator=g, dtype=D))

    def f(log_w):
        post, p_none, _ = pairwise_exclusive(li, rj, log_w, log_a)
        return torch.cat((post, p_none))

    log_w = (torch.randn(li.numel(), generator=g, dtype=D) * 2.0).requires_grad_(True)
    assert torch.autograd.gradcheck(f, (log_w,), eps=1e-6, atol=1e-6, rtol=1e-4)


def _crowded_scene(seed: int, n_left: int, frac: float, side: float = 0.05):
    """Dense precise left catalogue; shallow right one whose published errors
    are 2x too small (CatWISE-like in a bulge field)."""
    g = _gen(seed)
    lon = 150 + torch.rand(n_left, generator=g, dtype=D) * side
    lat = 2 + torch.rand(n_left, generator=g, dtype=D) * side
    src = torch.nonzero(torch.rand(n_left, generator=g) < frac, as_tuple=True)[0]
    n_bg = 300
    pub = 150.0 + 250.0 * torch.rand(len(src) + n_bg, generator=g, dtype=D)
    lon0 = torch.cat((lon[src], 150 + torch.rand(n_bg, generator=g, dtype=D) * side))
    lat0 = torch.cat((lat[src], 2 + torch.rand(n_bg, generator=g, dtype=D) * side))
    z = torch.randn(len(lon0), 2, generator=g, dtype=D) * (2.0 * pub)[:, None]
    rlon, rlat = _offset(lon0, lat0, z[:, 0], z[:, 1])
    truth = torch.full((len(lon0),), -1, dtype=torch.int64)
    truth[: len(src)] = src
    lerr = torch.full((n_left,), 3.0, dtype=D)
    left = Catalog(
        lon=lon, lat=lat, epoch=2016.0, astrometry=Astrometry.from_errors(lerr, lerr)
    )
    right = Catalog(
        lon=rlon, lat=rlat, epoch=2016.0, astrometry=Astrometry.from_errors(pub, pub)
    )
    box = MOC.from_polygon(
        [150, 150 + side, 150 + side, 150], [2, 2, 2 + side, 2 + side], max_order=18
    )
    return left, right, truth, box, len(src) / n_left


def test_exclusive_model_beats_independent_in_a_crowded_field() -> None:
    left, right, truth, box, frac = _crowded_scene(4, 5000, 0.3)
    out = {}
    for exclusive in (False, True):
        res = associate(
            left,
            right,
            gate_pvalue=1e-4,
            max_radius_arcsec=15.0,
            right_footprint=box,
            completeness="em",
            assign="none",
            calibrate="scale_floor",
            exclusive=exclusive,
        )
        ok = truth[res.right_index] == res.left_index
        p = res.posterior
        bins = torch.clamp((p * 10).long(), max=9)
        ece = sum(
            float((bins == b).double().mean())
            * abs(float(p[bins == b].mean()) - float(ok[bins == b].double().mean()))
            for b in range(10)
            if bool((bins == b).any())
        )
        out[exclusive] = (res.completeness, res.right_scale, ece, float(p[ok].sum()))
    comp, scale, ece, true_mass = out[True]
    assert abs(comp - frac) < 0.02
    assert abs(scale - 2.0) < 0.15
    assert ece < 0.015
    # Independent posteriors hand shared right rows to several left rows:
    # calibration is worse and less mass goes to the true pairs.
    assert abs(out[False][1] - 2.0) < 0.15  # both fit the error scale
    assert ece < out[False][2]
    assert true_mass > out[False][3]


def test_nway_exclusive_shares_no_source_mass() -> None:
    from torchsky.catalogs import associate_many

    cats, _, box = _triple_scene(27, 1500, 3000)
    res = associate_many(
        cats, footprints=[box] * 3, max_radius_arcsec=30, completeness=[0.8, 0.6]
    )
    for c in (1, 2):
        rows = res.members[:, c]
        on = rows >= 0
        mass = torch.zeros(len(cats[c]), dtype=D).index_add(
            0, rows[on], res.posterior[on]
        )
        assert float(mass.max()) <= 1.0 + 1e-9
    prim = torch.repeat_interleave(
        torch.arange(len(cats[0])), res.offsets[1:] - res.offsets[:-1]
    )
    total = torch.zeros(len(cats[0]), dtype=D).index_add(0, prim, res.posterior)
    torch.testing.assert_close(total, torch.ones(len(cats[0]), dtype=D))


def test_native_segment_lse_matches_torch_reference() -> None:
    from torchsky.catalogs._exclusive import _Segments, lse_excluding_self

    seg_obj = _Segments(torch.zeros(1, dtype=torch.int64), 1)
    if seg_obj.ext is None:
        pytest.skip("native extension not built")
    g = _gen(44)
    n_seg = 500
    seg = torch.randint(0, n_seg, (5000,), generator=g)
    x = torch.randn(5000, generator=g, dtype=D) * 30.0
    x[::17] = -math.inf  # absent weights
    x[1::23] = x[0]  # ties
    slack = torch.randn(n_seg, generator=g, dtype=D)
    segs = _Segments(seg, n_seg)
    for sl in (None, slack):
        ref = lse_excluding_self(x, seg, n_seg, sl)
        got = segs.lse_excl(x, sl)
        torch.testing.assert_close(got, ref, rtol=1e-13, atol=0.0, equal_nan=True)


def test_fixed_calibration_transfers_from_a_fit() -> None:
    g = _gen(45)
    n = 3000
    lon = 150 + torch.rand(n, generator=g, dtype=D) * 0.2
    lat = 2 + torch.rand(n, generator=g, dtype=D) * 0.2
    pub = 100.0 + 100.0 * torch.rand(n, generator=g, dtype=D)
    z = torch.randn(n, 2, generator=g, dtype=D) * (2.0 * pub)[:, None]
    rlon, rlat = _offset(lon, lat, z[:, 0], z[:, 1])
    lerr = torch.full((n,), 20.0, dtype=D)
    left = Catalog(
        lon=lon, lat=lat, epoch=2016.0, astrometry=Astrometry.from_errors(lerr, lerr)
    )
    right = Catalog(
        lon=rlon, lat=rlat, epoch=2016.0, astrometry=Astrometry.from_errors(pub, pub)
    )
    kw = dict(gate_pvalue=1e-4, max_radius_arcsec=30.0, completeness=1.0, assign="none")
    fit = associate(left, right, calibrate="scale_floor", **kw)
    fixed = associate(
        left, right, right_scale=fit.right_scale, floor_mas2=fit.floor_mas2, **kw
    )
    assert fixed.right_scale == pytest.approx(fit.right_scale)
    torch.testing.assert_close(fixed.posterior, fit.posterior, rtol=1e-6, atol=1e-8)
    with pytest.raises(ValueError):
        associate(left, right, right_scale=-1.0, **kw)
