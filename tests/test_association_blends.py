"""Blend-aware association (``associate(blends="pairs")``): solver, likelihood, fits."""

from __future__ import annotations

import itertools
import math

import pytest
import torch

from torchsky.catalogs import Astrometry, Catalog, associate, match_within
from torchsky.catalogs._blends import (
    _BlendGraph,
    blend_log_density,
    dimer_graph,
    edge_geometry,
)
from torchsky.catalogs._zones import unit_vectors
from torchsky.sphere import MOC

D = torch.float64
MAS = math.radians(1.0 / 3.6e6)


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _offset(lon, lat, de_mas, dn_mas):
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


# ------------------------------------------------------------------- solver


def _graph(hyps, n_owner):
    """``_BlendGraph`` from ``[(owner, (members...), log_w)]``."""
    edges: dict = {}
    e1, e2, own = [], [], []
    for o, mem, _ in hyps:
        ids = [edges.setdefault((o, m), len(edges)) for m in mem]
        e1.append(ids[0])
        e2.append(ids[1] if len(ids) > 1 else -1)
        own.append(o)
    keys = list(edges)
    prims = sorted({m for _, m in keys})
    graph = _BlendGraph(
        torch.tensor(own),
        torch.tensor(e1),
        torch.tensor(e2),
        torch.tensor([o for o, _ in keys]),
        torch.tensor([prims.index(m) for _, m in keys]),
        n_owner,
        len(prims),
    )
    return graph, torch.tensor([h[2] for h in hyps], dtype=D)


def _brute(hyps, n_owner):
    """Exact hypothesis marginals and log Z: every owner picks background or
    one hypothesis, and no primary is used twice."""
    options = [
        [None] + [h for h, (o, _, _) in enumerate(hyps) if o == j]
        for j in range(n_owner)
    ]
    z = 0.0
    marg = [0.0] * len(hyps)
    for combo in itertools.product(*options):
        used = [m for h in combo if h is not None for m in hyps[h][1]]
        if len(used) != len(set(used)):
            continue
        w = math.exp(sum(hyps[h][2] for h in combo if h is not None))
        z += w
        for h in combo:
            if h is not None:
                marg[h] += w
    return torch.tensor(marg, dtype=D) / z, math.log(z)


def _tree_hyps(g, n_owner: int, scale: float):
    """The owner-primary graph is a forest: each owner reuses at most one
    earlier primary."""
    hyps, n_prim = [], 0
    for j in range(n_owner):
        cands = []
        if n_prim and float(torch.rand(1, generator=g)) < 0.7:
            cands.append(int(torch.randint(0, n_prim, (1,), generator=g)))
        for _ in range(int(torch.randint(1, 3, (1,), generator=g))):
            cands.append(n_prim)
            n_prim += 1
        for m in cands:
            hyps.append((j, (m,), float(torch.randn(1, generator=g, dtype=D) * scale)))
        for a, b in itertools.combinations(sorted(cands), 2):
            hyps.append(
                (j, (a, b), float(torch.randn(1, generator=g, dtype=D) * scale))
            )
    return hyps


def _loopy_hyps(g, n_owner: int, n_prim: int):
    hyps = []
    for j in range(n_owner):
        k = int(torch.randint(1, 4, (1,), generator=g))
        cands = torch.randperm(n_prim, generator=g)[:k].tolist()
        for m in cands:
            hyps.append((j, (m,), float(torch.randn(1, generator=g, dtype=D) * 2)))
        for a, b in itertools.combinations(sorted(cands), 2):
            if float(torch.rand(1, generator=g)) < 0.6:
                hyps.append(
                    (j, (a, b), float(torch.randn(1, generator=g, dtype=D) * 2))
                )
    return hyps


def test_blend_marginals_exact_on_trees_and_bounded_on_loops() -> None:
    g = _gen(51)
    for trial in range(120):
        n_owner = 1 + trial % 4
        # Large weights drive owner probabilities to 1 (the direct-sum path).
        hyps = _tree_hyps(g, n_owner, 12.0 if trial % 3 == 0 else 2.0)
        graph, lw = _graph(hyps, n_owner)
        x = graph.messages(lw, None)
        p_h, p_bg, p_e, p_free = graph.marginals(x, lw)
        ref, log_z = _brute(hyps, n_owner)
        assert float((p_h - ref).abs().max()) < 1e-10
        assert float(graph.log_partition(x, lw)) == pytest.approx(log_z, rel=1e-10)
        # Owners and primaries are normalised.
        owners = p_bg.index_add(0, graph.hyp_owner, p_h)
        torch.testing.assert_close(owners, torch.ones_like(owners))
        prim = p_free.index_add(0, graph.edge_primary, p_e)
        torch.testing.assert_close(prim, torch.ones_like(prim))
    worst = 0.0
    for _ in range(80):
        hyps = _loopy_hyps(g, 3, 5)
        graph, lw = _graph(hyps, 3)
        x = graph.messages(lw, None)
        p_h = graph.marginals(x, lw)[0]
        worst = max(worst, float((p_h - _brute(hyps, 3)[0]).abs().max()))
    # Dense loopy graphs are the Bethe approximation's hard case.
    assert worst < 0.15


def test_blend_gradient_matches_finite_differences() -> None:
    g = _gen(52)
    hyps = _loopy_hyps(g, 3, 5)
    graph, lw0 = _graph(hyps, 3)

    def f(lw):
        x = graph.messages(lw, None)
        p_h, p_bg, _, _ = graph.marginals(x, lw)
        return torch.cat((p_h, p_bg))

    lw = lw0.clone().requires_grad_(True)
    assert torch.autograd.gradcheck(f, (lw,), eps=1e-6, atol=1e-6, rtol=1e-4)


def test_prior_graph_is_the_monomer_dimer_partition_function() -> None:
    # A path and a triangle with a tail: dimers on primary pairs.
    for edges in ([(0, 1), (1, 2), (2, 3)], [(0, 1), (1, 2), (0, 2), (2, 3)]):
        g = _gen(53)
        lw = torch.randn(len(edges), generator=g, dtype=D)
        pairs = torch.tensor(edges)
        graph = dimer_graph(pairs[:, 0], pairs[:, 1])
        assert graph.dimer
        lwt = lw
        x = graph.messages(lwt, None)
        # The closed-form dimer sweep equals the general one.
        general, _ = _graph(
            [(d, e, float(lw[d])) for d, e in enumerate(edges)], len(edges)
        )
        assert not general.dimer
        xg = general.messages(lw, None)
        assert float(general.log_partition(xg, lw)) == pytest.approx(
            float(graph.log_partition(x, lwt)), rel=1e-9
        )
        exact = math.log(
            sum(
                math.exp(sum(float(lw[d]) for d in m))
                for r in range(len(edges) + 1)
                for m in itertools.combinations(range(len(edges)), r)
                if len({v for d in m for v in edges[d]}) == 2 * len(m)
            )
        )
        bethe = float(graph.log_partition(x, lwt))
        if len(edges) == 3:  # a path is a tree: exact
            assert bethe == pytest.approx(exact, rel=1e-10)
        else:
            assert abs(bethe - exact) < 0.05


# --------------------------------------------------------------- likelihood


def test_blend_density_matches_numerical_integration() -> None:
    g = _gen(54)
    p = 40
    ua = torch.randn(p, 2, generator=g, dtype=D) * 1500.0
    ub = torch.randn(p, 2, generator=g, dtype=D) * 1500.0
    a = 100.0 + 400.0 * torch.rand(p, generator=g, dtype=D)
    s = Astrometry.from_ellipse(
        a, 0.6 * a, 180.0 * torch.rand(p, generator=g, dtype=D)
    ).cov[:, :2, :2]
    zero = torch.zeros(p, 2, 2, dtype=D)
    nodes = torch.linspace(0.0, 1.0, 20001, dtype=D)

    def gauss(r, cov):
        inv = torch.linalg.inv(cov)
        q = (r[..., None, :] @ inv @ r[..., :, None])[..., 0, 0]
        return torch.exp(-0.5 * q) / (2 * math.pi * torch.sqrt(torch.linalg.det(cov)))

    # Uniform flux fraction, exact primaries: closed form against quadrature.
    log_b, d2, mean = blend_log_density(
        ua, ub, zero, zero, s, torch.full((p,), math.nan, dtype=D)
    )
    f = nodes[None, :, None]
    r = ub[:, None, :] + f * (ua - ub)[:, None, :]
    dens = gauss(r, s[:, None].expand(-1, nodes.numel(), -1, -1))
    integral = torch.trapezoid(dens, nodes, dim=1)
    torch.testing.assert_close(log_b, torch.log(integral), rtol=1e-6, atol=1e-6)
    post_mean = torch.trapezoid(dens * nodes, nodes, dim=1) / integral
    torch.testing.assert_close(mean, post_mean, rtol=1e-5, atol=1e-6)
    q = (r[..., None, :] @ torch.linalg.inv(s)[:, None] @ r[..., :, None])[..., 0, 0]
    torch.testing.assert_close(d2, q.amin(1), rtol=1e-6, atol=1e-6)
    # Given fraction: one Gaussian with the primaries' errors at that fraction.
    ca = Astrometry.from_errors(
        torch.full((p,), 30.0, dtype=D), torch.full((p,), 20.0, dtype=D)
    ).cov[:, :2, :2]
    frac = torch.rand(p, generator=g, dtype=D)
    log_f, d2_f, frac_out = blend_log_density(ua, ub, ca, 2 * ca, s, frac)
    fr = frac[:, None, None]
    cov = s + fr**2 * ca + (1 - fr) ** 2 * (2 * ca)
    centre = ub + frac[:, None] * (ua - ub)
    torch.testing.assert_close(log_f, torch.log(gauss(centre, cov)))
    torch.testing.assert_close(frac_out, frac)
    # Coincident primaries: the blend is a single Gaussian.
    log_c, _, mean_c = blend_log_density(
        ua, ua, zero, zero, s, torch.full((p,), math.nan, dtype=D)
    )
    torch.testing.assert_close(log_c, torch.log(gauss(ua, s)))
    torch.testing.assert_close(mean_c, torch.full((p,), 0.5, dtype=D))


def test_edge_geometry_is_the_log_map_at_the_secondary() -> None:
    # Near the pole the local bases of nearby points differ by large rotations.
    g = _gen(55)
    n = 50
    rlon = torch.rand(n, generator=g, dtype=D) * 360
    rlat = 90 - torch.rand(n, generator=g, dtype=D) * 0.01
    de = torch.randn(n, generator=g, dtype=D) * 3000.0
    dn = torch.randn(n, generator=g, dtype=D) * 3000.0
    # Exact exponential map at the secondary (unit vectors: no lon/lat round
    # trip, which loses precision at the pole).
    v = unit_vectors(rlon, rlat)
    lam, phi = torch.deg2rad(rlon), torch.deg2rad(rlat)
    east = torch.stack((-torch.sin(lam), torch.cos(lam), torch.zeros_like(lam)), -1)
    north = torch.stack(
        (
            -torch.sin(phi) * torch.cos(lam),
            -torch.sin(phi) * torch.sin(lam),
            torch.cos(phi),
        ),
        -1,
    )
    theta = MAS * torch.hypot(de, dn)[:, None]
    w = v * torch.cos(theta) + (de[:, None] * east + dn[:, None] * north) * (
        MAS * torch.sin(theta) / theta
    )
    lcov = torch.tensor([[25.0, 5.0], [5.0, 16.0]], dtype=D).expand(n, 2, 2)
    u, c = edge_geometry(w, v, lcov)
    torch.testing.assert_close(u, torch.stack((de, dn), -1), rtol=0, atol=1e-6)
    # The primary basis is rotated against the secondary's; transport keeps the
    # eigenvalues and turns the ellipse by the angle between the bases.
    torch.testing.assert_close(
        torch.linalg.eigvalsh(c), torch.linalg.eigvalsh(lcov), rtol=1e-12, atol=1e-9
    )


# ------------------------------------------------------------- associate


def _blend_scene(
    seed: int,
    n_left: int,
    *,
    density: float = 0.1,
    frac_det: float = 0.3,
    frac_blend: float = 0.3,
    blend_radius: float = 4.0,
    true_scale: float = 2.0,
):
    """Deep precise primaries (3 mas). A shallow secondary detects a fraction
    of them alone and, with probability ``frac_blend``, a detected primary
    together with a random neighbour within ``blend_radius``: one detection at
    the flux-weighted centroid. 5 % background; published errors (100-250
    mas) are ``true_scale`` too small. Returns the catalogues, the true
    members per secondary row (``[M, 2]``, ascending, ``-1`` padded), the
    footprint and the primaries' fluxes."""
    g = _gen(seed)
    side = math.sqrt(n_left / density) / 3600.0
    lon = 150 + torch.rand(n_left, generator=g, dtype=D) * side
    lat = 2 + torch.rand(n_left, generator=g, dtype=D) * side
    a = 0.35 * math.log(10)
    u = torch.rand(n_left, generator=g, dtype=D)
    mag = torch.log(math.exp(a * 12) + u * (math.exp(a * 18) - math.exp(a * 12))) / a
    flux = 10 ** (-0.4 * mag)
    n_det = int(frac_det * n_left)
    seeds = torch.randperm(n_left, generator=g)[:n_det]
    want_pair = torch.rand(n_det, generator=g) < frac_blend
    nb = match_within(lon, lat, lon, lat, blend_radius)
    li, rj = nb.left_index, nb.right_index
    li, rj = li[li != rj], rj[li != rj]
    start = torch.searchsorted(li, torch.arange(n_left))
    count = torch.bincount(li, minlength=n_left)
    used = torch.zeros(n_left, dtype=torch.bool)
    used[seeds] = True
    members = []
    for s, wp in zip(seeds.tolist(), want_pair.tolist()):
        partner = -1
        if wp and count[s] > 0:
            cand = int(
                rj[start[s] + int(torch.randint(0, int(count[s]), (1,), generator=g))]
            )
            if not used[cand]:
                partner = cand
                used[cand] = True
        members.append((s, partner))
    m = torch.tensor(members, dtype=torch.int64)
    xa = unit_vectors(lon[m[:, 0]], lat[m[:, 0]])
    pb = m[:, 1].clamp(min=0)
    xb = unit_vectors(lon[pb], lat[pb])
    fa = flux[m[:, 0]]
    fb = torch.where(m[:, 1] >= 0, flux[pb], torch.zeros_like(fa))
    c = (fa / (fa + fb))[:, None] * xa + (fb / (fa + fb))[:, None] * xb
    c = c / c.norm(dim=-1, keepdim=True)
    n_bg = n_det // 20
    clon = torch.cat(
        (
            torch.rad2deg(torch.atan2(c[:, 1], c[:, 0])) % 360,
            150 + torch.rand(n_bg, generator=g, dtype=D) * side,
        )
    )
    clat = torch.cat(
        (
            torch.rad2deg(torch.asin(c[:, 2])),
            2 + torch.rand(n_bg, generator=g, dtype=D) * side,
        )
    )
    truth = torch.cat((m, torch.full((n_bg, 2), -1, dtype=torch.int64)))
    perr = 100.0 + 150.0 * torch.rand(len(clon), generator=g, dtype=D)
    z = torch.randn(len(clon), 2, generator=g, dtype=D) * (true_scale * perr)[:, None]
    rlon, rlat = _offset(clon, clat, z[:, 0], z[:, 1])
    swap = (truth[:, 1] >= 0) & (truth[:, 1] < truth[:, 0])
    truth = torch.where(swap[:, None], truth.flip(-1), truth)
    lerr = torch.full((n_left,), 3.0, dtype=D)
    left = Catalog(
        lon=lon, lat=lat, epoch=2016.0, astrometry=Astrometry.from_errors(lerr, lerr)
    )
    right = Catalog(
        lon=rlon, lat=rlat, epoch=2016.0, astrometry=Astrometry.from_errors(perr, perr)
    )
    box = MOC.from_polygon(
        [150, 150 + side, 150 + side, 150], [2, 2, 2 + side, 2 + side], max_order=18
    )
    return left, right, truth, box, flux


def _state(res, n_right: int) -> torch.Tensor:
    if res.blend is not None:
        return res.blend.map_members
    st = torch.full((n_right, 2), -1, dtype=torch.int64)
    st[res.right_index[res.match], 0] = res.left_index[res.match]
    return st


def _membership(
    res, truth: torch.Tensor, n_left: int, thr: float
) -> tuple[float, float]:
    owner = torch.full((n_left,), -1, dtype=torch.int64)
    rows = torch.arange(truth.shape[0])
    for c in (0, 1):
        ok = truth[:, c] >= 0
        owner[truth[ok, c]] = rows[ok]
    correct = owner[res.left_index] == res.right_index
    claim = res.posterior >= thr
    hits = float((claim & correct).sum())
    return hits / int((owner >= 0).sum()), hits / max(int(claim.sum()), 1)


COMMON = dict(gate_pvalue=1e-4, max_radius_arcsec=15.0, completeness="em")


def test_blends_are_opt_in() -> None:
    left, right, _, box, _ = _blend_scene(56, 400)
    res = associate(left, right, right_footprint=box, **COMMON)
    assert res.blend is None
    for bad in (
        dict(blends="pairs"),  # no blend radius
        dict(blends="pairs", blend_radius_arcsec=4.0, exclusive=False),
        dict(blends="triples", blend_radius_arcsec=4.0),
        dict(
            blends="pairs",
            blend_radius_arcsec=4.0,
            blend_flux=-torch.ones(len(left), dtype=D),
        ),
        dict(
            blends="pairs", blend_radius_arcsec=4.0, blend_flux=torch.ones(3, dtype=D)
        ),
    ):
        with pytest.raises(ValueError):
            associate(left, right, right_footprint=box, **COMMON, **bad)


@pytest.mark.parametrize("completeness", ["em", 0.25])
def test_without_pair_sources_blends_equal_the_exclusive_model(completeness) -> None:
    left, right, _, box, _ = _blend_scene(57, 1500, frac_blend=0.0)
    kw = dict(COMMON, completeness=completeness, right_footprint=box, assign="none")
    ref = associate(left, right, **kw)
    # A blend radius below the primaries' spacing: no pair sources at all.
    res = associate(left, right, **kw, blends="pairs", blend_radius_arcsec=1e-6)
    assert res.blend is not None and res.blend.n_pair_sources == 0
    assert torch.equal(res.left_index, ref.left_index)
    assert torch.equal(res.right_index, ref.right_index)
    torch.testing.assert_close(res.posterior, ref.posterior, rtol=0, atol=1e-9)
    torch.testing.assert_close(res.p_none, ref.p_none, rtol=0, atol=1e-9)
    torch.testing.assert_close(res.log_lr, ref.log_lr)
    torch.testing.assert_close(res.d2, ref.d2)
    assert res.completeness == pytest.approx(ref.completeness, abs=1e-7)
    single = res.blend.p_single
    torch.testing.assert_close(single, res.posterior)  # every edge is a single
    p_bg = res.blend.p_background
    taken = torch.zeros(len(right), dtype=D).index_add(
        0, res.right_index, res.posterior
    )
    torch.testing.assert_close(p_bg + taken, torch.ones(len(right), dtype=D))


def test_no_injected_blends_fit_no_blends() -> None:
    left, right, _, box, _ = _blend_scene(58, 2000, frac_blend=0.0)
    kw = dict(COMMON, right_footprint=box, assign="none", calibrate="scale_floor")
    ref = associate(left, right, **kw)
    res = associate(left, right, **kw, blends="pairs", blend_radius_arcsec=4.0)
    assert res.blend is not None and res.blend.n_pair_sources > 1000
    assert res.blend.blend_rate < 1e-3
    assert float(res.blend.posterior.sum()) < 1.0
    assert abs(res.completeness - ref.completeness) < 0.01
    assert abs(res.right_scale - ref.right_scale) < 0.03


def test_blend_model_recovers_blends_and_the_error_scale() -> None:
    left, right, truth, box, flux = _blend_scene(59, 2500)
    n_left, n_right = len(left), len(right)
    kw = dict(COMMON, right_footprint=box, calibrate="scale_floor")
    excl = associate(left, right, **kw)
    blend = associate(left, right, **kw, blends="pairs", blend_radius_arcsec=4.0)
    known = associate(
        left, right, **kw, blends="pairs", blend_radius_arcsec=4.0, blend_flux=flux
    )
    is_blend = truth[:, 1] >= 0
    for res in (blend, known):
        assert res.blend is not None
        # The blended positions no longer inflate the error scale (true 2).
        assert abs(res.right_scale - 2.0) < abs(excl.right_scale - 2.0)
        assert abs(res.right_scale - 2.0) < 0.1
        # Blends are counted about right, and the secondary states improve.
        n_blends = int(is_blend.sum())
        assert abs(float(res.blend.posterior.sum()) - n_blends) < 0.25 * n_blends
        acc = (_state(res, n_right) == truth).all(-1).double()
        acc_excl = (_state(excl, n_right) == truth).all(-1).double()
        assert float(acc.mean()) > float(acc_excl.mean()) + 0.02
        assert float(acc[is_blend].mean()) > 0.15
        # More true members are confident, at the same purity.
        comp, pur = _membership(res, truth, n_left, 0.9)
        comp_x, pur_x = _membership(excl, truth, n_left, 0.9)
        assert comp > comp_x and pur > pur_x - 0.01
        # Every left row's probabilities sum to one; members are ordered.
        rows = res.p_none.index_add(0, res.left_index, res.posterior)
        torch.testing.assert_close(rows, torch.ones(n_left, dtype=D))
        mem = res.blend.members
        assert bool((mem[:, 0] < mem[:, 1]).all())
        # The selection uses each left row once.
        sel = res.left_index[res.match]
        assert sel.unique().numel() == sel.numel()
    # Known fluxes fix the centroid: fewer, sharper pair hypotheses.
    assert known.blend.right_index.numel() < blend.blend.right_index.numel()
    assert float(
        ((known.blend.flux_fraction >= 0) & (known.blend.flux_fraction <= 1)).all()
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_blends_run_on_device(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    # MPS has no float64; associate as a whole does not run there.
    left, right, truth, box, _ = _blend_scene(60, 600)
    kw = dict(COMMON, right_footprint=box, blends="pairs", blend_radius_arcsec=4.0)
    ref = associate(left, right, **kw)
    res = associate(left.to(device), right.to(device), **kw)
    assert res.posterior.device.type == device
    assert res.blend is not None and res.blend.map_members.device.type == device
    torch.testing.assert_close(res.posterior.cpu(), ref.posterior, rtol=1e-6, atol=1e-8)
