"""Crossmatch candidate engine against an exact brute-force oracle.

Every public within-radius and nearest-match path must reproduce the
``O(N M)`` oracle exactly, outside a tie band of float64 rounding around
the threshold. Geometry covers the poles (including exactly +-90 deg), the RA
0/360 wrap, a crowded cluster, duplicated positions, and per-row radii that span
five decades.
"""

from __future__ import annotations

import math

import pytest
import torch

from torchsky.catalogs import crossmatch_sky
from torchsky.catalogs import _zones

D = torch.float64
# Pairs whose separation lies within this band of their threshold are
# float64-ambiguous and excluded from set comparisons (they are counted).
TIE_BAND_ARCSEC = 1e-8


def _unit(ra_deg: torch.Tensor, dec_deg: torch.Tensor) -> torch.Tensor:
    ra, dec = torch.deg2rad(ra_deg), torch.deg2rad(dec_deg)
    return torch.stack(
        [
            torch.cos(dec) * torch.cos(ra),
            torch.cos(dec) * torch.sin(ra),
            torch.sin(dec),
        ],
        dim=-1,
    )


def oracle_separations(la, lb, ra, rb) -> torch.Tensor:
    """Dense ``[N, M]`` great-circle separations in arcsec (atan2 form)."""
    a, b = _unit(la, lb), _unit(ra, rb)
    out = torch.empty(a.shape[0], b.shape[0], dtype=D)
    for s in range(0, a.shape[0], 1024):
        aa = a[s : s + 1024, None, :]
        cross = torch.linalg.cross(
            aa.expand(-1, b.shape[0], -1), b[None].expand(aa.shape[0], -1, -1), dim=-1
        )
        dot = (aa * b[None]).sum(-1)
        out[s : s + 1024] = torch.rad2deg(torch.atan2(cross.norm(dim=-1), dot)) * 3600.0
    return out


def _threshold(
    n: int, m: int, radius, left_err, right_err, combine: str
) -> torch.Tensor:
    if radius is not None:
        r = torch.as_tensor(radius, dtype=D)
        return (r.expand(n) if r.ndim == 0 else r)[:, None].expand(n, m)
    le = torch.as_tensor(0.0 if left_err is None else left_err, dtype=D)
    re = torch.as_tensor(0.0 if right_err is None else right_err, dtype=D)
    le = (le.expand(n) if le.ndim == 0 else le)[:, None]
    re = (re.expand(m) if re.ndim == 0 else re)[None, :]
    if combine == "sum":
        return le + re
    return torch.sqrt(le * le + re * re)


def oracle_all(sep: torch.Tensor, thr: torch.Tensor):
    """Return (sure pairs, ambiguous pairs) as Python sets."""
    inside = sep <= thr
    band = (sep - thr).abs() <= TIE_BAND_ARCSEC + 1e-12 * thr
    sure = inside & ~band
    li, ri = torch.nonzero(sure, as_tuple=True)
    bi, bj = torch.nonzero(band, as_tuple=True)
    return set(zip(li.tolist(), ri.tolist())), set(zip(bi.tolist(), bj.tolist()))


def oracle_best(
    sep: torch.Tensor, thr: torch.Tensor
) -> tuple[dict[int, int], set[int]]:
    """Nearest right row per left row (ties to lowest right index).

    Returns the expected mapping for unambiguous left rows and the set of left
    rows whose answer is float64-ambiguous (threshold or runner-up tie band).
    """
    masked = torch.where(sep <= thr, sep, torch.full_like(sep, math.inf))
    expected: dict[int, int] = {}
    ambiguous: set[int] = set()
    if masked.shape[1] == 0:
        return expected, ambiguous
    best, arg = masked.min(dim=1)  # min returns the first (lowest) index on ties
    band = (sep - thr).abs() <= TIE_BAND_ARCSEC + 1e-12 * thr
    for i in range(masked.shape[0]):
        if band[i].any():
            ambiguous.add(i)
            continue
        if not math.isfinite(float(best[i])):
            continue
        near = (masked[i] - best[i]).abs() <= TIE_BAND_ARCSEC
        near[arg[i]] = False
        if bool(near.any()) and bool((masked[i][near] != best[i]).any()):
            ambiguous.add(i)
            continue
        expected[i] = int(arg[i])
    return expected, ambiguous


@pytest.fixture(params=["native", "torch"], autouse=True)
def engine_path(request, monkeypatch):
    """Run every oracle test on the native kernel and on the torch path."""
    if request.param == "native":
        if _zones._native_ext() is None:
            pytest.skip("native crossmatch kernel not built")
    else:
        monkeypatch.setattr(_zones, "_native_ext", lambda: None)
    return request.param


# --------------------------------------------------------------------- geometry


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def uniform_sky(n: int, seed: int):
    g = _gen(seed)
    ra = torch.rand(n, generator=g, dtype=D) * 360.0
    dec = torch.rad2deg(torch.asin(torch.rand(n, generator=g, dtype=D) * 2.0 - 1.0))
    return ra, dec


def jitter(ra, dec, sigma_arcsec: float, seed: int):
    """Offset each point by a random tangent-plane vector (pole-safe)."""
    g = _gen(seed)
    v = _unit(ra, dec)
    east = torch.stack(
        [
            -torch.sin(torch.deg2rad(ra)),
            torch.cos(torch.deg2rad(ra)),
            torch.zeros_like(ra),
        ],
        -1,
    )
    north = torch.linalg.cross(v, east, dim=-1)
    pole = v[:, 2].abs() > 1.0 - 1e-15  # exactly at a pole: any tangent basis works
    east[pole] = torch.tensor([1.0, 0.0, 0.0], dtype=D)
    north[pole] = torch.tensor([0.0, 1.0, 0.0], dtype=D)
    s = math.radians(sigma_arcsec / 3600.0)
    dx = torch.randn(ra.shape[0], generator=g, dtype=D) * s
    dy = torch.randn(ra.shape[0], generator=g, dtype=D) * s
    w = v + dx[:, None] * east + dy[:, None] * north
    w = w / w.norm(dim=-1, keepdim=True)
    out_ra = torch.rad2deg(torch.atan2(w[:, 1], w[:, 0])) % 360.0
    out_dec = torch.rad2deg(torch.asin(w[:, 2].clamp(-1.0, 1.0)))
    return out_ra, out_dec


def pole_field(n: int, seed: int, *, south: bool = False):
    g = _gen(seed)
    ra = torch.rand(n, generator=g, dtype=D) * 360.0
    dec = 90.0 - torch.rand(n, generator=g, dtype=D) * 0.02
    dec[: max(1, n // 50)] = 90.0  # exactly on the pole, arbitrary RA
    return (ra, -dec) if south else (ra, dec)


def wrap_field(n: int, seed: int):
    g = _gen(seed)
    ra = (torch.rand(n, generator=g, dtype=D) - 0.5) * 0.02 % 360.0
    dec = (torch.rand(n, generator=g, dtype=D) - 0.5) * 0.02
    return ra, dec


def cluster_field(n: int, seed: int, *, radius_arcsec: float = 10.0):
    g = _gen(seed)
    r = torch.sqrt(torch.rand(n, generator=g, dtype=D)) * radius_arcsec / 3600.0
    t = torch.rand(n, generator=g, dtype=D) * 2.0 * math.pi
    return 150.0 + r * torch.cos(t) / math.cos(math.radians(2.0)), 2.0 + r * torch.sin(
        t
    )


def _case(name: str):
    if name == "uniform":
        la, lb = uniform_sky(1500, 1)
        ra, rb = uniform_sky(1500, 2)
        return la, lb, ra, rb, 3600.0
    if name == "north_pole":
        la, lb = pole_field(800, 3)
        ra, rb = jitter(la, lb, 5.0, 4)
        return la, lb, ra, rb, 30.0
    if name == "south_pole":
        la, lb = pole_field(800, 5, south=True)
        ra, rb = jitter(la, lb, 5.0, 6)
        return la, lb, ra, rb, 30.0
    if name == "ra_wrap":
        la, lb = wrap_field(1200, 7)
        ra, rb = jitter(la, lb, 2.0, 8)
        return la, lb, ra, rb, 5.0
    if name == "crowded_cluster":
        la, lb = cluster_field(2000, 9)
        ra, rb = cluster_field(2000, 10)
        return la, lb, ra, rb, 0.3
    raise AssertionError(name)


CASES = ["uniform", "north_pole", "south_pole", "ra_wrap", "crowded_cluster"]


def _pairs(res) -> list[tuple[int, int]]:
    return list(zip(res.left_index.tolist(), res.right_index.tolist(), strict=True))


def _check_all(pairs: list[tuple[int, int]], sure: set, band: set) -> None:
    got = set(pairs)
    assert len(got) == len(pairs), "duplicate pairs returned"
    missing = sure - got
    extra = got - sure - band
    assert not missing, f"{len(missing)} true pairs dropped, e.g. {sorted(missing)[:5]}"
    assert not extra, (
        f"{len(extra)} pairs outside the threshold, e.g. {sorted(extra)[:5]}"
    )


def _check_best(
    pairs: list[tuple[int, int]], expected: dict[int, int], ambiguous: set[int]
) -> None:
    got: dict[int, int] = {}
    for i, j in pairs:
        assert i not in got, f"left row {i} returned twice"
        got[i] = j
    for i, j in expected.items():
        assert got.get(i) == j, f"left {i}: expected right {j}, got {got.get(i)}"
    extra = set(got) - set(expected) - ambiguous
    assert not extra, f"unexpected matches for left rows {sorted(extra)[:5]}"


# ------------------------------------------------------------------------ tests


@pytest.mark.parametrize("case", CASES)
def test_all_pairs_match_oracle(case: str) -> None:
    la, lb, ra, rb, r = _case(case)
    sep = oracle_separations(la, lb, ra, rb)
    sure, band = oracle_all(
        sep, _threshold(la.numel(), ra.numel(), r, None, None, "sum")
    )
    assert len(sure) > 50, "geometry must produce matches"
    res = crossmatch_sky(la, lb, ra, rb, radius_arcsec=r, find="all")
    _check_all(_pairs(res), sure, band)
    torch.testing.assert_close(
        res.separation_arcsec,
        sep[res.left_index, res.right_index],
        rtol=1e-12,
        atol=1e-9,
    )


@pytest.mark.parametrize("case", CASES)
def test_best_matches_oracle(case: str) -> None:
    la, lb, ra, rb, r = _case(case)
    sep = oracle_separations(la, lb, ra, rb)
    expected, ambiguous = oracle_best(
        sep, _threshold(la.numel(), ra.numel(), r, None, None, "sum")
    )
    res = crossmatch_sky(la, lb, ra, rb, radius_arcsec=r, find="best")
    _check_best(_pairs(res), expected, ambiguous)


def test_per_row_radii_spanning_five_decades() -> None:
    la, lb = uniform_sky(1500, 11)
    ra, rb = jitter(la, lb, 30.0, 12)
    g = _gen(13)
    # 1.8 mas .. 180 arcsec, log-uniform
    radius = 1.8e-3 * 10.0 ** (torch.rand(la.numel(), generator=g, dtype=D) * 5.0)
    radius[:5] = 5.0 * 3600.0  # a few 5-degree windows
    sep = oracle_separations(la, lb, ra, rb)
    thr = _threshold(la.numel(), ra.numel(), radius, None, None, "sum")
    sure, band = oracle_all(sep, thr)
    _check_all(
        _pairs(crossmatch_sky(la, lb, ra, rb, radius_arcsec=radius, find="all")),
        sure,
        band,
    )
    expected, ambiguous = oracle_best(sep, thr)
    _check_best(
        _pairs(crossmatch_sky(la, lb, ra, rb, radius_arcsec=radius, find="best")),
        expected,
        ambiguous,
    )


@pytest.mark.parametrize("case", ["north_pole", "ra_wrap", "crowded_cluster"])
def test_two_sided_error_radii_sum(case: str) -> None:
    la, lb, ra, rb, r = _case(case)
    g = _gen(17)
    le = torch.rand(la.numel(), generator=g, dtype=D) * r
    re = torch.rand(ra.numel(), generator=g, dtype=D) * r
    re[:3] = 20.0 * r  # one large right error must not break anyone else
    sep = oracle_separations(la, lb, ra, rb)
    sure, band = oracle_all(
        sep, _threshold(la.numel(), ra.numel(), None, le, re, "sum")
    )
    res = crossmatch_sky(
        la, lb, ra, rb, left_err_arcsec=le, right_err_arcsec=re, find="all"
    )
    _check_all(_pairs(res), sure, band)


def test_duplicates_break_ties_to_lowest_right_index() -> None:
    la, lb = uniform_sky(300, 19)
    ra = torch.cat([la, la, la])
    rb = torch.cat([lb, lb, lb])
    res = crossmatch_sky(la, lb, ra, rb, radius_arcsec=1.0, find="best")
    assert res.left_index.tolist() == list(range(300))
    assert res.right_index.tolist() == list(range(300))
    every = crossmatch_sky(la, lb, ra, rb, radius_arcsec=1.0, find="all")
    assert every.left_index.numel() == 900


def test_exact_pole_points_with_arbitrary_ra_match_each_other() -> None:
    la = torch.tensor([0.0, 123.0, 271.5], dtype=D)
    lb = torch.full((3,), 90.0, dtype=D)
    ra = torch.tensor([45.0, 300.0], dtype=D)
    rb = torch.tensor([90.0, 90.0 - 1.0 / 3600.0], dtype=D)
    res = crossmatch_sky(la, lb, ra, rb, radius_arcsec=2.0, find="all")
    assert sorted(_pairs(res)) == [(i, j) for i in range(3) for j in range(2)]


def test_empty_inputs() -> None:
    e = torch.empty(0, dtype=D)
    one = torch.tensor([10.0], dtype=D)
    for args in [(e, e, one, one), (one, one, e, e), (e, e, e, e)]:
        for find in ("best", "all"):
            res = crossmatch_sky(*args, radius_arcsec=1.0, find=find)
            assert res.left_index.numel() == 0
            assert res.right_index.numel() == 0


@pytest.mark.parametrize("chunk", [1, 7, 100_000])
def test_chunk_size_never_changes_results(chunk: int) -> None:
    la, lb = wrap_field(150, 21)
    ra, rb = jitter(la, lb, 2.0, 22)
    ref = crossmatch_sky(la, lb, ra, rb, radius_arcsec=5.0, find="all")
    got = crossmatch_sky(
        la,
        lb,
        ra,
        rb,
        radius_arcsec=5.0,
        find="all",
        left_chunk_size=chunk,
        right_chunk_size=max(chunk, 2),
    )
    assert _pairs(got) == _pairs(ref)
    torch.testing.assert_close(
        got.separation_arcsec, ref.separation_arcsec, rtol=0.0, atol=0.0
    )


@pytest.mark.parametrize("threads", [1, 4])
def test_thread_count_never_changes_results(threads: int) -> None:
    la, lb, ra, rb, r = _case("crowded_cluster")
    ref = crossmatch_sky(la, lb, ra, rb, radius_arcsec=r, find="all")
    old = torch.get_num_threads()
    try:
        torch.set_num_threads(threads)
        got = crossmatch_sky(la, lb, ra, rb, radius_arcsec=r, find="all")
    finally:
        torch.set_num_threads(old)
    assert _pairs(got) == _pairs(ref)
    torch.testing.assert_close(
        got.separation_arcsec, ref.separation_arcsec, rtol=0.0, atol=0.0
    )


def test_self_match_per_row_radius_is_symmetric() -> None:
    from torchsky.catalogs import crossmatch_self

    ra, dec = cluster_field(600, 23, radius_arcsec=20.0)
    g = _gen(24)
    radius = 0.1 + torch.rand(600, generator=g, dtype=D) * 1.5
    sep = oracle_separations(ra, dec, ra, dec)
    thr = torch.maximum(radius[:, None], radius[None, :])
    sure, band = oracle_all(sep, thr)
    sure = {(i, j) for i, j in sure if i < j}
    band = {(i, j) for i, j in band if i < j}
    res = crossmatch_self(ra, dec, radius)
    _check_all(_pairs(res), sure, band)
    # Reversing the row order gives the same pairs (the old rule used radius[i] only).
    rev = torch.arange(599, -1, -1)
    back = crossmatch_self(ra[rev], dec[rev], radius[rev])
    mapped = {tuple(sorted((599 - i, 599 - j))) for i, j in _pairs(back)}
    assert mapped == set(_pairs(res))


# ------------------------------------------------------------ SkyIndex API


def oracle_nearest(sep: torch.Tensor, k: int, cap: float = math.inf):
    """Expected ``[N, k]`` neighbour indices (-1 past the cap), ties to lowest index."""
    m = sep.shape[1]
    idx = torch.arange(m).expand_as(sep)
    # Stable lexicographic (sep, index): stable sort on sep keeps index order.
    s, order = torch.sort(sep, dim=1, stable=True)
    take = min(k, m)
    out = torch.full((sep.shape[0], k), -1, dtype=torch.int64)
    out_s = torch.full((sep.shape[0], k), math.inf, dtype=D)
    inside = s[:, :take] <= cap
    out[:, :take] = torch.where(inside, idx.gather(1, order)[:, :take], -1)
    out_s[:, :take] = torch.where(inside, s[:, :take], math.inf)
    return out, out_s


@pytest.mark.parametrize("combine", ["sum", "quadrature", "max"])
def test_match_within_two_sided_combines(combine: str) -> None:
    from torchsky.catalogs import match_within

    la, lb, ra, rb, r = _case("crowded_cluster")
    g = _gen(31)
    rl = torch.rand(la.numel(), generator=g, dtype=D) * r
    rr = torch.rand(ra.numel(), generator=g, dtype=D) * r
    sep = oracle_separations(la, lb, ra, rb)
    sure, band = oracle_all(
        sep,
        _threshold(la.numel(), ra.numel(), None, rl, rr, combine)
        if combine != "max"
        else torch.maximum(rl[:, None], rr[None, :]),
    )
    res = match_within(la, lb, ra, rb, rl, right_radius_arcsec=rr, combine=combine)
    _check_all(_pairs(res), sure, band)


def test_count_within_matches_oracle_and_reused_index() -> None:
    from torchsky.catalogs import SkyIndex, count_within

    la, lb, ra, rb, r = _case("north_pole")
    sep = oracle_separations(la, lb, ra, rb)
    expected = (sep <= r).sum(dim=1)
    index = SkyIndex.build(ra, rb)
    got = count_within(la, lb, ra, rb, r, index=index)
    band = ((sep - r).abs() <= TIE_BAND_ARCSEC).any(dim=1)
    assert torch.equal(got[~band], expected[~band])
    assert torch.equal(index.count_within(la, lb, r), got)


@pytest.mark.parametrize(
    "case", ["uniform", "north_pole", "ra_wrap", "crowded_cluster"]
)
@pytest.mark.parametrize("k", [1, 3])
def test_match_nearest_exact_k(case: str, k: int) -> None:
    from torchsky.catalogs import match_nearest

    la, lb, ra, rb, _ = _case(case)
    sep = oracle_separations(la, lb, ra, rb)
    want_i, want_s = oracle_nearest(sep, k)
    got = match_nearest(la, lb, ra, rb, k=k)
    # Rows whose k-th and (k+1)-th neighbours tie within float64 are ambiguous.
    srt = torch.sort(sep, dim=1).values
    amb = (
        (srt[:, k] - srt[:, k - 1]).abs() <= TIE_BAND_ARCSEC
        if sep.shape[1] > k
        else torch.zeros(sep.shape[0], dtype=torch.bool)
    )
    assert torch.equal(got.index[~amb], want_i[~amb])
    torch.testing.assert_close(
        got.separation_arcsec[~amb], want_s[~amb], rtol=1e-12, atol=1e-9
    )


def test_match_nearest_cap_and_far_away_catalogue() -> None:
    from torchsky.catalogs import match_nearest

    # Sparse targets far from a small right footprint: radii must grow to it.
    la, lb = uniform_sky(50, 41)
    ra, rb = wrap_field(400, 42)
    sep = oracle_separations(la, lb, ra, rb)
    want_i, want_s = oracle_nearest(sep, 2)
    got = match_nearest(la, lb, ra, rb, k=2)
    assert torch.equal(got.index, want_i)
    capped = match_nearest(la, lb, ra, rb, k=2, max_radius_arcsec=30.0 * 3600.0)
    cap_i, _ = oracle_nearest(sep, 2, cap=30.0 * 3600.0)
    assert torch.equal(capped.index, cap_i)
    # More neighbours requested than right rows exist.
    few = match_nearest(la[:3], lb[:3], ra[:2], rb[:2], k=4)
    assert (few.index[:, 2:] == -1).all() and torch.isinf(
        few.separation_arcsec[:, 2:]
    ).all()


def test_match_nearest_agrees_with_astropy_nthneighbor() -> None:
    coords = pytest.importorskip("astropy.coordinates")
    units = pytest.importorskip("astropy.units")
    from torchsky.catalogs import match_nearest

    la, lb = uniform_sky(400, 51)
    ra, rb = uniform_sky(900, 52)
    left = coords.SkyCoord(la.numpy() * units.deg, lb.numpy() * units.deg)
    right = coords.SkyCoord(ra.numpy() * units.deg, rb.numpy() * units.deg)
    got = match_nearest(la, lb, ra, rb, k=2)
    for nth in (1, 2):
        idx, d2d, _ = left.match_to_catalog_sky(right, nthneighbor=nth)
        assert got.index[:, nth - 1].tolist() == idx.tolist()
        torch.testing.assert_close(
            got.separation_arcsec[:, nth - 1],
            torch.from_numpy(d2d.arcsec),
            rtol=0.0,
            atol=1e-6,
        )


def test_separations_are_differentiable() -> None:
    from torchsky.catalogs import match_within

    la, lb, ra, rb, r = _case("ra_wrap")
    la = la[:60].clone().requires_grad_(True)
    lb = lb[:60].clone().requires_grad_(True)
    res = match_within(la, lb, ra, rb, r)
    assert res.separation_arcsec.requires_grad
    res.separation_arcsec.sum().backward()
    assert la.grad is not None and torch.isfinite(la.grad).all()

    def f(x, y):
        return match_within(x, y, ra, rb, r).separation_arcsec

    torch.autograd.gradcheck(
        f,
        (
            la.detach()[:5].clone().requires_grad_(True),
            lb.detach()[:5].clone().requires_grad_(True),
        ),
    )


def test_tiny_candidate_batches_split_single_windows() -> None:
    from torchsky.catalogs import SkyIndex

    la, lb, ra, rb, r = _case("crowded_cluster")
    index = SkyIndex.build(ra, rb, zone_height_deg=1.0)  # one coarse zone: huge windows
    ref = index.query_within(la[:50], lb[:50], r)
    got = index.query_within(la[:50], lb[:50], r, max_pairs=3)
    assert _pairs(got) == _pairs(ref)


def test_invalid_inputs_fail_closed() -> None:
    from torchsky.catalogs import match_nearest, match_within

    one = torch.tensor([10.0], dtype=D)
    with pytest.raises(ValueError, match="finite"):
        match_within(torch.tensor([math.nan], dtype=D), one * 0, one, one * 0, 1.0)
    with pytest.raises(ValueError, match="latitudes"):
        match_within(one, torch.tensor([91.0], dtype=D), one, one * 0, 1.0)
    with pytest.raises(ValueError, match="positive"):
        match_within(one, one * 0, one, one * 0, 0.0)
    with pytest.raises(ValueError, match="k must"):
        match_nearest(one, one * 0, one, one * 0, k=0)


def test_native_and_torch_paths_agree_at_scale(monkeypatch) -> None:
    from torchsky.catalogs import match_nearest, match_within

    if _zones._native_ext() is None:
        pytest.skip("native crossmatch kernel not built")
    la, lb = uniform_sky(40_000, 61)
    ra, rb = jitter(la, lb, 20.0, 62)
    g = _gen(63)
    radius = 1.0 + 59.0 * torch.rand(40_000, generator=g, dtype=D)
    native_all = match_within(la, lb, ra, rb, radius)
    native_knn = match_nearest(la, lb, ra, rb, k=3)
    monkeypatch.setattr(_zones, "_native_ext", lambda: None)
    torch_all = match_within(la, lb, ra, rb, radius)
    torch_knn = match_nearest(la, lb, ra, rb, k=3)
    assert torch.equal(native_all.left_index, torch_all.left_index)
    assert torch.equal(native_all.right_index, torch_all.right_index)
    torch.testing.assert_close(
        native_all.separation_arcsec, torch_all.separation_arcsec, rtol=0, atol=0
    )
    assert torch.equal(native_knn.index, torch_knn.index)
    torch.testing.assert_close(
        native_knn.separation_arcsec, torch_knn.separation_arcsec, rtol=0, atol=0
    )


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
@pytest.mark.parametrize("case", CASES)
def test_mps_results_are_exact_float64(case: str) -> None:
    from torchsky.catalogs import match_nearest, match_within

    la, lb, ra, rb, r = _case(case)
    sep = oracle_separations(la, lb, ra, rb)
    sure, band = oracle_all(
        sep, _threshold(la.numel(), ra.numel(), r, None, None, "sum")
    )
    res = crossmatch_sky(la, lb, ra, rb, radius_arcsec=r, find="all", device="mps")
    assert res.left_index.device.type == "mps"
    _check_all(_pairs(res), sure, band)
    within = match_within(la, lb, ra, rb, r, device="mps")
    assert within.left_index.device.type == "mps"
    _check_all(_pairs(within), sure, band)
    near = match_nearest(la, lb, ra, rb, k=1, device="mps")
    assert near.index.device.type == "mps"
    torch.testing.assert_close(
        near.index.cpu(), match_nearest(la, lb, ra, rb, k=1).index
    )
