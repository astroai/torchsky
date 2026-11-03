"""Exact-key, self, and multi-catalogue matching built on the strip index."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, cast

import torch
from torch import Tensor

from .._dtype import work_dtype as _work_dtype
from ._pairs import (
    connected_components,
    join_indices,
    label_groups,
    normalize_find,
    select_pairs,
)
from .core import (
    CrossmatchResult,
    _empty_crossmatch,
    _error_radii,
    _normalize_radius_arcsec,
    sky_pairs,
)
from ..device import _as_tensor, _to


@dataclass
class SelfMatchResult:
    """Pairs ``i < j`` within one catalogue, optionally with FoF groups.

    ``group_id`` / ``group_size`` are per row (``None`` unless ``group=True``).
    Rows with no partner get ``group_id = -1`` and ``group_size = 1``; groups
    of two or more rows are numbered ``0..n_groups-1`` by smallest member row.
    """

    left_index: Tensor
    right_index: Tensor
    separation_arcsec: Tensor
    group_id: Tensor | None = None
    group_size: Tensor | None = None
    n_groups: int = 0


@dataclass
class ManyMatchResult:
    """One output row per match group across ``C`` catalogues.

    ``index[g, c]`` is the row of catalogue ``c`` in output row ``g`` (``-1`` if
    absent). ``n_members`` counts present entries; ``group_size`` counts every
    member of the underlying group (equal to ``n_members`` in ``pairs`` mode).
    ``separation_arcsec[g, c]`` is the separation to the reference (``pairs``)
    or the nearest link in the group (``group``); NaN when absent or unlinked.
    """

    index: Tensor
    n_members: Tensor
    group_size: Tensor
    separation_arcsec: Tensor


def join_rows(
    result: CrossmatchResult,
    n_left: int,
    n_right: int,
    join: str = "1and2",
) -> tuple[Tensor, Tensor]:
    """Row-index pairs for a STILTS join type, ``-1`` where a side is absent.

    ``join`` is one of ``1and2`` (matched pairs), ``1or2`` (matched pairs plus
    every unmatched row of both), ``all1`` / ``all2`` (matched pairs plus the
    unmatched rows of table 1 / 2), ``1not2`` / ``2not1`` (unmatched rows only).
    Output order: matched pairs as given, then unmatched left rows ascending,
    then unmatched right rows ascending. Costs ``O(K + N + M)``.

    Returns:
        ``(left_rows, right_rows)`` int64 tensors of equal length.
    """
    return join_indices(result.left_index, result.right_index, n_left, n_right, join)


def crossmatch_exact(
    left_key: Tensor,
    right_key: Tensor,
    *,
    find: str = "all",
) -> CrossmatchResult:
    """Equal-key join (STILTS ``exact``): ``O((N + M) log M + K)``.

    Keys are 1-d tensors (integers or floats; NaN never matches). Separations
    are returned as zeros. ``find`` follows :func:`crossmatch_sky`; every pair
    scores equally, so ties resolve by lowest row index (``"symmetric"`` pairs
    the lowest rows of each key first).
    """
    find = normalize_find(find)
    kl = torch.as_tensor(left_key).reshape(-1)
    kr = torch.as_tensor(right_key, device=kl.device).reshape(-1)
    device = kl.device
    dtype = _work_dtype(device)
    if kl.numel() == 0 or kr.numel() == 0:
        return _empty_crossmatch(device)
    order = torch.argsort(kr, stable=True)
    if kr.is_floating_point():  # NaN breaks searchsorted: drop those right rows
        order = order[~torch.isnan(kr[order])]
    sorted_keys = kr[order]
    if sorted_keys.numel() == 0:
        return _empty_crossmatch(device)
    lo = torch.searchsorted(sorted_keys, kl, right=False)
    hi = torch.searchsorted(sorted_keys, kl, right=True)
    counts = hi - lo
    if kl.is_floating_point():
        counts = counts * (~torch.isnan(kl))
    left = torch.repeat_interleave(torch.arange(kl.numel(), device=device), counts)
    base = torch.repeat_interleave(lo, counts)
    first = torch.cumsum(counts, 0) - counts
    offs = torch.arange(left.numel(), device=device) - torch.repeat_interleave(
        first, counts
    )
    right = order[base + offs]
    sep = torch.zeros(left.numel(), dtype=dtype, device=device)
    return CrossmatchResult(*select_pairs(left, right, sep, find))


def crossmatch_self(
    lon_deg: Tensor,
    lat_deg: Tensor,
    radius_arcsec: float | Tensor | None = None,
    *,
    err_arcsec: float | Tensor | None = None,
    group: bool = False,
    left_chunk_size: int = 65536,
    right_chunk_size: int = 65536,
) -> SelfMatchResult:
    """Pairs within one catalogue (STILTS ``tmatch1``).

    Returns every pair ``i < j`` (no self pairs, each pair once) with
    ``separation <= radius_arcsec`` or, with ``err_arcsec`` (scalar or per row),
    ``separation <= err[i] + err[j]``. With ``group=True`` also returns
    friends-of-friends groups (connected components of the pair graph) as
    per-row ``group_id`` and ``group_size`` (STILTS ``action=identify``).
    Candidate generation is the declination-strip index, so cost is
    ``O(N log N + K)``; components use scatter-min union-find.
    """
    ra = torch.as_tensor(lon_deg).reshape(-1)
    device = ra.device
    dtype = _work_dtype(device)
    ra = ra.to(dtype)
    dec = _as_tensor(lat_deg, dtype=dtype, device=device).reshape(-1)
    n = int(ra.numel())
    res = _self_pairs(
        ra, dec, radius_arcsec, err_arcsec, left_chunk_size, right_chunk_size
    )
    if not group:
        return SelfMatchResult(res.left_index, res.right_index, res.separation_arcsec)
    root = connected_components(n, res.left_index, res.right_index)
    gid, sizes, per_row = label_groups(root)
    in_group = per_row >= 2
    # Renumber groups of size >= 2 contiguously, keeping smallest-member order.
    multi = sizes >= 2
    remap = torch.cumsum(multi.to(torch.int64), 0) - 1
    group_id = torch.where(in_group, remap[gid], torch.full_like(gid, -1))
    return SelfMatchResult(
        res.left_index,
        res.right_index,
        res.separation_arcsec,
        group_id,
        per_row,
        int(multi.sum().item()),
    )


def _self_pairs(
    ra: Tensor,
    dec: Tensor,
    radius_arcsec: float | Tensor | None,
    err_arcsec: float | Tensor | None,
    left_chunk_size: int,
    right_chunk_size: int,
) -> CrossmatchResult:
    n, device = int(ra.numel()), ra.device
    if (radius_arcsec is None) == (err_arcsec is None):
        raise ValueError("pass exactly one of radius_arcsec and err_arcsec")
    if radius_arcsec is not None:
        rl, _ = _normalize_radius_arcsec(radius_arcsec, n, device)
        rr = None
    else:
        rl = _error_radii(err_arcsec, n, device, "err_arcsec")  # type: ignore[assignment]
        rr = rl
    if n < 2:
        return _empty_crossmatch(device)
    return sky_pairs(
        ra,
        dec,
        ra,
        dec,
        radius_left=cast(Tensor, rl),
        radius_right=rr,
        find="all",
        accept=lambda lrows, rrows: lrows < rrows,
        left_chunk_size=left_chunk_size,
        right_chunk_size=right_chunk_size,
    )


def _as_lonlat(cat: Any, device: torch.device | None) -> tuple[Tensor, Tensor]:
    if hasattr(cat, "lon") and hasattr(cat, "lat"):
        lon, lat = cat.lon, cat.lat
    else:
        lon, lat = cat
    lon = torch.as_tensor(lon).reshape(-1)
    dev = device if device is not None else lon.device
    dtype = _work_dtype(dev)
    return _to(lon, device=dev, dtype=dtype), _as_tensor(
        lat, dtype=dtype, device=dev
    ).reshape(-1)


def crossmatch_many(
    catalogs: Sequence[Any],
    radius_arcsec: float | None = None,
    *,
    mode: str = "pairs",
    reference: int = 0,
    err_arcsec: Sequence[float | Tensor | None] | None = None,
) -> ManyMatchResult:
    """Match ``C >= 2`` catalogues (STILTS ``tmatchn``).

    Each catalogue is a ``Catalog`` or a ``(lon_deg, lat_deg)`` pair. A pair of
    rows matches when ``separation <= radius_arcsec`` or, with ``err_arcsec``
    (one scalar/tensor/``None`` per catalogue), ``<= err_a + err_b``.

    ``mode="pairs"`` (``multimode=pairs``): every other catalogue is matched to
    ``catalogs[reference]`` with ``find="best1"``: each reference row takes its
    nearest partner (ties by lowest row) in every other catalogue, so one
    entry may serve several reference rows. Output rows are all reference rows
    in order (``-1`` where no partner), then, for each other catalogue in
    order, its rows that are no reference row's partner (alone in their row).
    Two non-reference rows are never linked directly.

    ``mode="group"`` (``multimode=group``): friends-of-friends over *all*
    cross-catalogue pairs; one output row per group, ordered by the group's
    smallest stacked row (catalogue order, then row). A group may hold several
    rows of one catalogue (chains); the representative of catalogue ``c`` is
    the member with the smallest separation to any linked member of another
    catalogue, ties by lowest row index. Same-catalogue pairs are not links.
    """
    if mode not in ("pairs", "group"):
        raise ValueError("mode must be 'pairs' or 'group'")
    n_cat = len(catalogs)
    if n_cat < 2:
        raise ValueError("crossmatch_many needs at least two catalogues")
    if not 0 <= reference < n_cat:
        raise ValueError("reference must index a catalogue")
    if (radius_arcsec is None) == (err_arcsec is None):
        raise ValueError("pass exactly one of radius_arcsec and err_arcsec")
    if err_arcsec is not None and len(err_arcsec) != n_cat:
        raise ValueError("err_arcsec needs one entry per catalogue")
    first = catalogs[0]
    device = (
        first.lon.device if hasattr(first, "lon") else torch.as_tensor(first[0]).device
    )
    coords = [_as_lonlat(c, device) for c in catalogs]
    sizes = [int(lon.numel()) for lon, _ in coords]
    errs: list[Tensor | None] = [None] * n_cat
    if err_arcsec is not None:
        errs = [
            _error_radii(0.0 if e is None else e, sizes[i], device, "err_arcsec")
            for i, e in enumerate(err_arcsec)
        ]

    def pairs(a: int, b: int, find: str) -> CrossmatchResult:
        if sizes[a] == 0 or sizes[b] == 0:
            return _empty_crossmatch(device)
        if radius_arcsec is not None:
            rl, _ = _normalize_radius_arcsec(radius_arcsec, sizes[a], device)
            rr = None
        else:
            rl, rr = errs[a], errs[b]
        return sky_pairs(
            *coords[a],
            *coords[b],
            radius_left=rl,  # type: ignore[arg-type]
            radius_right=rr,
            find=find,
        )

    nan = float("nan")
    dtype = _work_dtype(device)
    if mode == "pairs":
        n_ref = sizes[reference]
        matched = {
            k: pairs(reference, k, "best1") for k in range(n_cat) if k != reference
        }
        extra = {}
        for k, res in matched.items():
            seen = torch.zeros(sizes[k], dtype=torch.bool, device=device)
            seen[res.right_index] = True
            extra[k] = torch.nonzero(~seen, as_tuple=True)[0]
        total = n_ref + sum(int(v.numel()) for v in extra.values())
        index = torch.full((total, n_cat), -1, dtype=torch.int64, device=device)
        sep = torch.full((total, n_cat), nan, dtype=dtype, device=device)
        index[:n_ref, reference] = torch.arange(n_ref, device=device)
        sep[:n_ref, reference] = 0.0
        cursor = n_ref
        for k in range(n_cat):
            if k == reference:
                continue
            res = matched[k]
            index[res.left_index, k] = res.right_index
            sep[res.left_index, k] = res.separation_arcsec
            m = int(extra[k].numel())
            index[cursor : cursor + m, k] = extra[k]
            cursor += m
        count = (index >= 0).sum(1)
        return ManyMatchResult(index, count, count, sep)

    offsets = [0]
    for s in sizes:
        offsets.append(offsets[-1] + s)
    total = offsets[-1]
    us: list[Tensor] = []
    vs: list[Tensor] = []
    ss: list[Tensor] = []
    for a in range(n_cat):
        for b in range(a + 1, n_cat):
            res = pairs(a, b, "all")
            us.append(res.left_index + offsets[a])
            vs.append(res.right_index + offsets[b])
            ss.append(res.separation_arcsec)
    u, v, s = torch.cat(us), torch.cat(vs), torch.cat(ss)
    root = connected_components(total, u, v)
    gid, gsizes, _ = label_groups(root)
    link = torch.full((total,), float("inf"), dtype=dtype, device=device)
    link.scatter_reduce_(0, u, s, "amin")
    link.scatter_reduce_(0, v, s, "amin")
    cat_of = torch.repeat_interleave(
        torch.arange(n_cat, device=device),
        torch.tensor(sizes, device=device),
    )
    # Representative: smallest (group, catalogue, link separation, row).
    order = torch.argsort(link, stable=True)
    order = order[torch.argsort(gid[order] * n_cat + cat_of[order], stable=True)]
    cell = (gid * n_cat + cat_of)[order]
    keep = torch.ones_like(cell, dtype=torch.bool)
    keep[1:] = cell[1:] != cell[:-1]
    rep = order[keep]
    n_groups = int(gsizes.numel())
    index = torch.full((n_groups, n_cat), -1, dtype=torch.int64, device=device)
    sep = torch.full((n_groups, n_cat), nan, dtype=dtype, device=device)
    rep_cat = cat_of[rep]
    rep_off = torch.tensor(offsets[:-1], device=device)[rep_cat]
    index[gid[rep], rep_cat] = rep - rep_off
    rep_link = link[rep]
    sep[gid[rep], rep_cat] = torch.where(
        torch.isfinite(rep_link), rep_link, torch.full_like(rep_link, nan)
    )
    return ManyMatchResult(index, (index >= 0).sum(1), gsizes, sep)
