"""Partitioned matching equals single-pass matching (owner = left row)."""

from __future__ import annotations

import pytest
import torch

from torchsky.catalogs import crossmatch_sky, match_nearest
from torchsky.catalogs.partition import PartitionPlan, healpix29, margin_arcsec

D = torch.float64


def _sky(seed: int, n: int):
    g = torch.Generator().manual_seed(seed)
    lon = torch.rand(n, generator=g, dtype=D) * 360
    lat = torch.rad2deg(torch.asin(torch.rand(n, generator=g, dtype=D) * 2 - 1))
    # Clusters at both poles, across RA 0/360 and around a dense blob.
    k = n // 8
    lon[:k] = torch.rand(k, generator=g, dtype=D) * 360
    lat[:k] = 89.5 + torch.rand(k, generator=g, dtype=D) * 0.5
    lon[k : 2 * k] = torch.rand(k, generator=g, dtype=D) * 360
    lat[k : 2 * k] = -89.5 - torch.rand(k, generator=g, dtype=D) * 0.5
    lon[2 * k : 3 * k] = (torch.rand(k, generator=g, dtype=D) - 0.5) % 360
    lat[2 * k : 3 * k] = torch.rand(k, generator=g, dtype=D) - 0.5
    lon[3 * k : 5 * k] = 150 + torch.rand(2 * k, generator=g, dtype=D) * 0.2
    lat[3 * k : 5 * k] = 2 + torch.rand(2 * k, generator=g, dtype=D) * 0.2
    return lon, lat


def _jitter(lon, lat, arcsec, seed):
    g = torch.Generator().manual_seed(seed)
    dlat = torch.randn(lon.shape, generator=g, dtype=D) * arcsec / 3600
    dlon = (
        torch.randn(lon.shape, generator=g, dtype=D)
        * arcsec
        / 3600
        / torch.cos(torch.deg2rad(lat)).clamp(min=1e-3)
    )
    return (lon + dlon) % 360, (lat + dlat).clamp(-90, 90)


def test_owner_is_unique_and_matches_order29_ranges() -> None:
    lon, lat = _sky(1, 20000)
    plan = PartitionPlan.from_counts(lon, lat, max_rows=500, max_order=10)
    owner = plan.owner(lon, lat)
    assert bool((owner >= 0).all())
    h = healpix29(lon, lat)
    assert bool(((h >= plan.start[owner]) & (h < plan.stop[owner])).all())
    counts = torch.bincount(owner, minlength=len(plan))
    assert int(counts[plan.orders < 10].max()) <= 500


@pytest.mark.parametrize("find", ["all", "best"])
def test_partitioned_crossmatch_equals_single_pass(find: str) -> None:
    llon, llat = _sky(2, 12000)
    rlon, rlat = _jitter(llon, llat, 3.0, 3)
    radius = 10.0
    want = crossmatch_sky(llon, llat, rlon, rlat, radius_arcsec=radius, find=find)
    left_plan = PartitionPlan.from_counts(llon, llat, max_rows=400, max_order=9)
    right_plan = PartitionPlan.from_counts(
        rlon, rlat, max_rows=700, max_order=8
    )  # different plan
    l_owner = left_plan.owner(llon, llat)
    r_owner = right_plan.owner(rlon, rlat)
    margin = margin_arcsec(radius)
    p_idx, q_idx = left_plan.neighbors(margin, right_plan)
    got_l, got_r = [], []
    for p in range(len(left_plan)):
        lrows = torch.nonzero(l_owner == p, as_tuple=True)[0]
        if lrows.numel() == 0:
            continue
        qs = q_idx[p_idx == p]
        cand = torch.nonzero(torch.isin(r_owner, qs), as_tuple=True)[0]
        cand = cand[left_plan.in_margin(p, rlon[cand], rlat[cand], margin)]
        res = crossmatch_sky(
            llon[lrows],
            llat[lrows],
            rlon[cand],
            rlat[cand],
            radius_arcsec=radius,
            find=find,
        )
        got_l.append(lrows[res.left_index])
        got_r.append(cand[res.right_index])
    got = set(zip(torch.cat(got_l).tolist(), torch.cat(got_r).tolist()))
    assert got == set(zip(want.left_index.tolist(), want.right_index.tolist()))
    assert len(got) > 10000


def test_partitioned_nearest_equals_single_pass_with_cap() -> None:
    llon, llat = _sky(4, 8000)
    rlon, rlat = _sky(5, 8000)
    cap = 600.0
    want = match_nearest(llon, llat, rlon, rlat, k=2, max_radius_arcsec=cap)
    # Left rows must be owned: plan over both catalogues.
    plan = PartitionPlan.from_counts(
        torch.cat((llon, rlon)), torch.cat((llat, rlat)), max_rows=300, max_order=9
    )
    l_owner = plan.owner(llon, llat)
    r_owner = plan.owner(rlon, rlat)
    assert bool((l_owner >= 0).all())
    p_idx, q_idx = plan.neighbors(cap)
    got = torch.full_like(want.index, -1)
    for p in range(len(plan)):
        lrows = torch.nonzero(l_owner == p, as_tuple=True)[0]
        if lrows.numel() == 0:
            continue
        cand = torch.nonzero(torch.isin(r_owner, q_idx[p_idx == p]), as_tuple=True)[0]
        cand = cand[plan.in_margin(p, rlon[cand], rlat[cand], cap)]
        if cand.numel() == 0:
            continue  # nothing within the cap: rows stay -1
        res = match_nearest(
            llon[lrows], llat[lrows], rlon[cand], rlat[cand], k=2, max_radius_arcsec=cap
        )
        got[lrows] = torch.where(
            res.index >= 0, cand[res.index.clamp(min=0)], res.index
        )
    assert torch.equal(got, want.index)


def test_neighbors_are_sound_for_adjacent_cells_across_orders() -> None:
    plan = PartitionPlan.from_pixels([3, 5, 5, 0], [10, 4 * 4 * 11, 4 * 4 * 11 + 1, 11])
    p, q = plan.neighbors(1.0)
    pairs = set(zip(p.tolist(), q.tolist()))
    assert all((i, i) in pairs for i in range(len(plan)))
    with pytest.raises(ValueError, match="overlap"):
        PartitionPlan.from_pixels([0, 1], [0, 0])
