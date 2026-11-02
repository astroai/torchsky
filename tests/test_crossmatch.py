"""STILTS-style crossmatch primitives against brute-force oracles."""

from __future__ import annotations

import pytest
import torch

from torchsky.catalogs import (
    Catalog,
    CrossmatchResult,
    crossmatch_exact,
    crossmatch_many,
    crossmatch_self,
    crossmatch_sky,
    join_rows,
)
from torchsky.sphere import great_circle_distance

D = torch.float64


def _sky(n: int, seed: int, *, wrap: bool = True, pole: bool = True):
    g = torch.Generator().manual_seed(seed)
    ra = torch.rand(n, generator=g, dtype=D) * 360.0
    dec = torch.rand(n, generator=g, dtype=D) * 180.0 - 90.0
    if wrap:  # dense clump across RA 0/360
        k = n // 3
        ra[:k] = (torch.rand(k, generator=g, dtype=D) - 0.5) * 0.4 % 360.0
        dec[:k] = (torch.rand(k, generator=g, dtype=D) - 0.5) * 0.4
    if pole:
        k = n // 4
        dec[n - k :] = 89.9 + torch.rand(k, generator=g, dtype=D) * 0.1
        dec[n - k : n - k // 2] *= -1
    return ra, dec


def _sep_matrix(la, lb, ra, rb):
    return (
        great_circle_distance(
            la[:, None], lb[:, None], ra[None, :], rb[None, :], degrees=True
        )
        * 3600.0
    )


def _pairs(res: CrossmatchResult) -> set[tuple[int, int]]:
    return set(zip(res.left_index.tolist(), res.right_index.tolist(), strict=True))


def _data(seed=0, n=300, m=350):
    la, lb = _sky(n, seed)
    ra, rb = _sky(m, seed + 100)
    # make sure there are many near neighbours
    ra[: m // 2] = la[: m // 2] + (torch.rand(m // 2, dtype=D) - 0.5) * 8 / 3600
    rb[: m // 2] = lb[: m // 2] + (torch.rand(m // 2, dtype=D) - 0.5) * 8 / 3600
    return la, lb, ra, rb


def test_skyerr_all_matches_brute_force_with_wrap_and_poles() -> None:
    la, lb, ra, rb = _data()
    g = torch.Generator().manual_seed(5)
    el = torch.rand(la.numel(), generator=g, dtype=D) * 4.0
    er = torch.rand(ra.numel(), generator=g, dtype=D) * 3.0
    res = crossmatch_sky(
        la, lb, ra, rb, left_err_arcsec=el, right_err_arcsec=er, find="all"
    )
    sep = _sep_matrix(la, lb, ra, rb)
    li, ri = torch.nonzero(sep <= el[:, None] + er[None, :], as_tuple=True)
    assert _pairs(res) == set(zip(li.tolist(), ri.tolist(), strict=True))
    assert len(_pairs(res)) > 50
    # one-sided error and scalar errors
    res1 = crossmatch_sky(la, lb, ra, rb, left_err_arcsec=el, find="all")
    li, ri = torch.nonzero(sep <= el[:, None], as_tuple=True)
    assert _pairs(res1) == set(zip(li.tolist(), ri.tolist(), strict=True))
    res2 = crossmatch_sky(
        la, lb, ra, rb, left_err_arcsec=1.0, right_err_arcsec=2.0, find="all"
    )
    li, ri = torch.nonzero(sep <= 3.0, as_tuple=True)
    assert _pairs(res2) == set(zip(li.tolist(), ri.tolist(), strict=True))


def test_find_best1_best2_symmetric_oracle_and_ties() -> None:
    la, lb, ra, rb = _data(seed=3)
    r = 6.0
    sep = _sep_matrix(la, lb, ra, rb)
    inside = sep <= r
    # best1 / "best"
    for name in ("best", "best1"):
        res = crossmatch_sky(la, lb, ra, rb, radius_arcsec=r, find=name)
        masked = torch.where(inside, sep, torch.full_like(sep, float("inf")))
        best = masked.argmin(1)  # first minimum = lowest right index
        rows = torch.nonzero(inside.any(1), as_tuple=True)[0]
        assert res.left_index.tolist() == rows.tolist()
        assert res.right_index.tolist() == best[rows].tolist()
    # best2
    res = crossmatch_sky(la, lb, ra, rb, radius_arcsec=r, find="best2")
    masked = torch.where(inside, sep, torch.full_like(sep, float("inf")))
    best = masked.argmin(0)
    cols = torch.nonzero(inside.any(0), as_tuple=True)[0]
    assert _pairs(res) == {(int(best[c]), int(c)) for c in cols}
    assert res.left_index.tolist() == sorted(res.left_index.tolist())
    # symmetric: sequential greedy oracle
    res = crossmatch_sky(la, lb, ra, rb, radius_arcsec=r, find="symmetric")
    li, ri = torch.nonzero(inside, as_tuple=True)
    s = sep[li, ri]
    order = sorted(
        range(li.numel()), key=lambda k: (s[k].item(), li[k].item(), ri[k].item())
    )
    used_l, used_r, want = set(), set(), set()
    for k in order:
        a, b = int(li[k]), int(ri[k])
        if a not in used_l and b not in used_r:
            used_l.add(a), used_r.add(b), want.add((a, b))
    assert _pairs(res) == want
    assert len(set(res.left_index.tolist())) == res.left_index.numel()
    assert len(set(res.right_index.tolist())) == res.right_index.numel()


def test_symmetric_tie_rule_lowest_indices() -> None:
    # Two left rows equidistant from one right row: lowest left row wins.
    res = crossmatch_sky(
        torch.tensor([10.0, 10.0], dtype=D),
        torch.tensor([0.001, -0.001], dtype=D),
        torch.tensor([10.0], dtype=D),
        torch.tensor([0.0], dtype=D),
        radius_arcsec=10.0,
        find="symmetric",
    )
    assert res.left_index.tolist() == [0] and res.right_index.tolist() == [0]
    # best1 keeps both left rows; best2 keeps the lowest left row.
    kw = dict(radius_arcsec=10.0)
    args = (
        torch.tensor([10.0, 10.0], dtype=D),
        torch.tensor([0.001, -0.001], dtype=D),
        torch.tensor([10.0], dtype=D),
        torch.tensor([0.0], dtype=D),
    )
    assert crossmatch_sky(*args, find="best1", **kw).left_index.tolist() == [0, 1]
    assert crossmatch_sky(*args, find="best2", **kw).left_index.tolist() == [0]


def test_chunking_does_not_change_results() -> None:
    la, lb, ra, rb = _data(seed=9)
    for find in ("all", "best1", "best2", "symmetric"):
        a = crossmatch_sky(la, lb, ra, rb, radius_arcsec=5.0, find=find)
        b = crossmatch_sky(
            la,
            lb,
            ra,
            rb,
            radius_arcsec=5.0,
            find=find,
            left_chunk_size=17,
            right_chunk_size=64,
        )
        assert _pairs(a) == _pairs(b)


def test_distance_cut() -> None:
    la, lb, ra, rb = _data(seed=4)
    g = torch.Generator().manual_seed(1)
    dl = torch.rand(la.numel(), generator=g, dtype=D)
    dr = torch.rand(ra.numel(), generator=g, dtype=D)
    dl[0] = float("nan")
    res = crossmatch_sky(
        la,
        lb,
        ra,
        rb,
        radius_arcsec=6.0,
        find="all",
        left_distance=dl,
        right_distance=dr,
        distance_tol=0.2,
    )
    sep = _sep_matrix(la, lb, ra, rb)
    ok = (sep <= 6.0) & ((dl[:, None] - dr[None, :]).abs() <= 0.2)
    li, ri = torch.nonzero(ok, as_tuple=True)
    assert _pairs(res) == set(zip(li.tolist(), ri.tolist(), strict=True))
    with pytest.raises(ValueError):
        crossmatch_sky(la, lb, ra, rb, radius_arcsec=6.0, left_distance=dl)


def test_argument_validation() -> None:
    x = torch.zeros(2, dtype=D)
    with pytest.raises(ValueError):
        crossmatch_sky(x, x, x, x)
    with pytest.raises(ValueError):
        crossmatch_sky(x, x, x, x, radius_arcsec=1.0, left_err_arcsec=1.0)
    with pytest.raises(ValueError):
        crossmatch_sky(x, x, x, x, radius_arcsec=1.0, find="nope")
    with pytest.raises(ValueError):
        crossmatch_sky(x, x, x, x, left_err_arcsec=-1.0)


def test_join_rows_semantics() -> None:
    res = CrossmatchResult(
        torch.tensor([0, 2]), torch.tensor([1, 0]), torch.zeros(2, dtype=D)
    )
    expect = {
        "1and2": ([0, 2], [1, 0]),
        "all1": ([0, 2, 1, 3], [1, 0, -1, -1]),
        "all2": ([0, 2, -1], [1, 0, 2]),
        "1or2": ([0, 2, 1, 3, -1], [1, 0, -1, -1, 2]),
        "1not2": ([1, 3], [-1, -1]),
        "2not1": ([-1], [2]),
    }
    for join, (lrows, r) in expect.items():
        got = join_rows(res, 4, 3, join)
        assert (got[0].tolist(), got[1].tolist()) == (lrows, r), join
    with pytest.raises(ValueError):
        join_rows(res, 4, 3, "bogus")


def test_exact_matcher() -> None:
    kl = torch.tensor([5, 7, 5, 9, 1])
    kr = torch.tensor([7, 5, 5, 2])
    res = crossmatch_exact(kl, kr, find="all")
    assert _pairs(res) == {(0, 1), (0, 2), (1, 0), (2, 1), (2, 2)}
    assert _pairs(crossmatch_exact(kl, kr, find="best1")) == {(0, 1), (1, 0), (2, 1)}
    assert _pairs(crossmatch_exact(kl, kr, find="best2")) == {(0, 1), (1, 0), (0, 2)}
    assert _pairs(crossmatch_exact(kl, kr, find="symmetric")) == {
        (0, 1),
        (1, 0),
        (2, 2),
    }
    f = crossmatch_exact(
        torch.tensor([1.0, float("nan")]), torch.tensor([float("nan"), 1.0])
    )
    assert _pairs(f) == {(0, 1)}


def _oracle_groups(n, edges):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        parent[max(find(a), find(b))] = min(find(a), find(b))
    return [find(i) for i in range(n)]


def test_self_match_pairs_and_groups() -> None:
    ra, dec = _sky(400, 11)
    ra[:200] = ra[0] + (torch.rand(200, dtype=D) - 0.5) * 60 / 3600
    dec[:200] = dec[0] + (torch.rand(200, dtype=D) - 0.5) * 60 / 3600
    r = 5.0
    out = crossmatch_self(ra, dec, r, group=True)
    sep = _sep_matrix(ra, dec, ra, dec)
    iu = torch.triu(sep <= r, diagonal=1)
    li, ri = torch.nonzero(iu, as_tuple=True)
    assert _pairs(out) == set(zip(li.tolist(), ri.tolist(), strict=True))
    roots = _oracle_groups(400, zip(li.tolist(), ri.tolist(), strict=True))
    size = {}
    for x in roots:
        size[x] = size.get(x, 0) + 1
    assert out.group_size.tolist() == [size[x] for x in roots]
    assert out.n_groups == sum(1 for v in size.values() if v >= 2)
    # same partition and ids ordered by smallest member
    gid = out.group_id.tolist()
    assert all((g == -1) == (size[x] == 1) for g, x in zip(gid, roots, strict=True))
    seen = {}
    for g, x in zip(gid, roots, strict=True):
        if g >= 0:
            assert seen.setdefault(g, x) == x
    firsts = [min(i for i, g in enumerate(gid) if g == k) for k in range(out.n_groups)]
    assert firsts == sorted(firsts)
    # skyerr self-match
    e = torch.rand(400, dtype=D) * 3
    o2 = crossmatch_self(ra, dec, err_arcsec=e)
    li, ri = torch.nonzero(
        torch.triu(sep <= e[:, None] + e[None, :], diagonal=1), as_tuple=True
    )
    assert _pairs(o2) == set(zip(li.tolist(), ri.tolist(), strict=True))
    assert crossmatch_self(ra[:1], dec[:1], 1.0, group=True).group_id.tolist() == [-1]


def test_connected_components_long_chain() -> None:
    n = 5000
    ra = torch.arange(n, dtype=D) * (0.9 / 3600)
    out = crossmatch_self(ra, torch.zeros(n, dtype=D), 1.0, group=True)
    assert out.n_groups == 1 and int(out.group_size.min()) == n


def test_many_pairs_mode() -> None:
    cats = []
    base_ra, base_dec = _sky(120, 21)
    for k in range(3):
        g = torch.Generator().manual_seed(30 + k)
        jit = (torch.rand(120, generator=g, dtype=D) - 0.5) * 2 / 3600
        keep = torch.rand(120, generator=g) < 0.8
        cats.append(((base_ra + jit)[keep], (base_dec + jit)[keep]))
    out = crossmatch_many(cats, 3.0, mode="pairs", reference=1)
    n_ref = cats[1][0].numel()
    assert out.index[:n_ref, 1].tolist() == list(range(n_ref))
    for k in (0, 2):
        ref = crossmatch_sky(*cats[1], *cats[k], radius_arcsec=3.0, find="symmetric")
        got = out.index[:n_ref, k]
        assert {(int(i), int(j)) for i, j in enumerate(got) if j >= 0} == _pairs(ref)
        col = out.index[:, k]
        assert sorted(col[col >= 0].tolist()) == list(range(cats[k][0].numel()))
    assert out.index.shape[0] >= n_ref
    assert (out.n_members >= 1).all()
    assert (out.index[n_ref:, 1] == -1).all()


def test_many_group_mode() -> None:
    a = (
        torch.tensor([10.0, 20.0, 359.9999], dtype=D),
        torch.tensor([0.0, 0.0, 0.0], dtype=D),
    )
    b = (torch.tensor([10.0003, 10.0006, 0.0001], dtype=D), torch.zeros(3, dtype=D))
    c = (torch.tensor([10.0009, 50.0], dtype=D), torch.zeros(2, dtype=D))
    out = crossmatch_many([a, b, c], 1.2, mode="group")
    assert int(out.group_size.sum()) == 8
    # a2 (359.9999) and b2 (0.0001) are 0.72" apart across RA wrap
    assert any(r[0] == 2 and r[1] == 2 for r in out.index.tolist())
    # full-brute-force FoF check
    allra = torch.cat([a[0], b[0], c[0]])
    sep = _sep_matrix(allra, torch.zeros(8, dtype=D), allra, torch.zeros(8, dtype=D))
    cat = [0, 0, 0, 1, 1, 1, 2, 2]
    edges = [
        (i, j)
        for i in range(8)
        for j in range(i + 1, 8)
        if cat[i] != cat[j] and sep[i, j] <= 1.2
    ]
    roots = _oracle_groups(8, edges)
    assert out.index.shape[0] == len(set(roots))
    assert sorted(out.group_size.tolist()) == sorted(
        [roots.count(x) for x in set(roots)]
    )


def test_many_group_mode_representative_is_nearest_link() -> None:
    # catalogue 1 has two rows chained to one catalogue-0 row; nearer one is chosen.
    a = (torch.tensor([10.0], dtype=D), torch.zeros(1, dtype=D))
    b = (torch.tensor([10.0004, 10.0002], dtype=D), torch.zeros(2, dtype=D))
    out = crossmatch_many([a, b], 2.0, mode="group")
    assert out.index.tolist() == [[0, 1]]
    assert out.group_size.tolist() == [3] and out.n_members.tolist() == [2]


def test_catalog_crossmatch_options() -> None:
    la, lb, ra, rb = _data(seed=2, n=100, m=120)
    el = torch.full((100,), 2.0, dtype=D)
    left = Catalog(lon=la, lat=lb, columns={"e": el})
    right = Catalog(lon=ra, lat=rb)
    res = left.crossmatch(right, find="all", left_err_arcsec="e", right_err_arcsec=1.0)
    ref = crossmatch_sky(
        la, lb, ra, rb, left_err_arcsec=el, right_err_arcsec=1.0, find="all"
    )
    assert _pairs(res) == _pairs(ref)
    sym = left.crossmatch(right, 5.0, find="symmetric")
    assert sym.left_index.numel() == len(set(sym.right_index.tolist()))


def test_astropy_parity() -> None:
    pytest.importorskip("astropy")
    from astropy import units as u
    from astropy.coordinates import SkyCoord, search_around_sky

    la, lb, ra, rb = _data(seed=6)
    c1 = SkyCoord(la.numpy() * u.deg, lb.numpy() * u.deg)
    c2 = SkyCoord(ra.numpy() * u.deg, rb.numpy() * u.deg)
    i1, i2, _, _ = search_around_sky(c1, c2, 6 * u.arcsec)
    res = crossmatch_sky(la, lb, ra, rb, radius_arcsec=6.0, find="all")
    assert _pairs(res) == set(zip(i1.tolist(), i2.tolist(), strict=True))
    idx, d2d, _ = c1.match_to_catalog_sky(c2)
    best = crossmatch_sky(la, lb, ra, rb, radius_arcsec=6.0)
    expect = {(i, int(idx[i])) for i in range(la.numel()) if d2d[i].arcsec <= 6.0}
    assert _pairs(best) == expect


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_mps_options_run_float32() -> None:
    dev = torch.device("mps")
    x = torch.tensor([10.0, 10.0005], dtype=torch.float32, device=dev)
    z = torch.zeros(2, dtype=torch.float32, device=dev)
    res = crossmatch_sky(
        x, z, x, z, left_err_arcsec=2.0, right_err_arcsec=1.0, find="symmetric"
    )
    assert res.left_index.device.type == "mps"
    assert res.left_index.tolist() == [0, 1]
    assert crossmatch_self(x, z, 5.0, group=True).group_size.tolist() == [2, 2]
