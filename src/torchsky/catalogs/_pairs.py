"""Pair-list algebra shared by the crossmatch entry points.

Everything here works on index/separation tensors of candidate pairs, so each
step is ``O(K log K)`` (or linear) in the number of pairs ``K``, never in
``N * M``. Index tensors are int64 and run on any device, MPS included.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..device import _scatter_reduce_int

FIND_MODES = ("best", "best1", "best2", "all", "symmetric", "mutual")
JOIN_MODES = ("1and2", "1or2", "all1", "all2", "1not2", "2not1")


def normalize_find(find: str) -> str:
    if find not in FIND_MODES:
        raise ValueError(f"find must be one of {FIND_MODES}, got {find!r}")
    return "best1" if find == "best" else find


def lexsort_pairs(left: Tensor, right: Tensor, sep: Tensor) -> Tensor:
    """Permutation ordering pairs by ``(sep, left, right)`` ascending."""
    order = torch.argsort(right, stable=True)
    order = order[torch.argsort(left[order], stable=True)]
    return order[torch.argsort(sep[order], stable=True)]


def sort_by_left(left: Tensor, right: Tensor, sep: Tensor) -> Tensor:
    """Permutation ordering pairs by ``(left, right)`` ascending."""
    order = torch.argsort(right, stable=True)
    return order[torch.argsort(left[order], stable=True)]


def best_per_key(key: Tensor, other: Tensor, sep: Tensor) -> Tensor:
    """Indices keeping, for each ``key`` value, the minimum ``(sep, other)`` pair."""
    if key.numel() <= 1:
        return torch.arange(key.numel(), device=key.device)
    order = torch.argsort(other, stable=True)
    order = order[torch.argsort(sep[order], stable=True)]
    order = order[torch.argsort(key[order], stable=True)]
    k = key[order]
    keep = torch.ones(k.shape[0], dtype=torch.bool, device=key.device)
    keep[1:] = k[1:] != k[:-1]
    return order[keep]


def greedy_one_to_one(left: Tensor, right: Tensor, sep: Tensor) -> Tensor:
    """Indices of the greedy one-to-one assignment (not STILTS ``find=best``; see ``mutual``).

    Pairs are ranked by ``(sep, left, right)``; the best pair is taken, every
    pair touching a used row is discarded, and the process repeats. The result
    equals the sequential greedy pass, but rounds take all *locally dominant*
    pairs (best-ranked at both of their rows) at once, so typical inputs need a
    handful of ``O(K)`` rounds. A synthetic chain with strictly monotone
    separations needs ``O(K)`` rounds; that is the only slow case.
    """
    n = int(left.numel())
    if n <= 1:
        return torch.arange(n, device=left.device)
    device = left.device
    order = lexsort_pairs(left, right, sep)
    _, lc = torch.unique(left[order], return_inverse=True)
    _, rc = torch.unique(right[order], return_inverse=True)
    n_l, n_r = int(lc.max().item()) + 1, int(rc.max().item()) + 1
    alive = torch.arange(n, device=device)  # ranks into `order`
    chosen: list[Tensor] = []
    big = torch.tensor(n, dtype=torch.int64, device=device)
    while alive.numel() > 0:
        lcur, rcur = lc[alive], rc[alive]
        min_l = _scatter_reduce_int(big.expand(n_l).clone(), 0, lcur, alive, "amin")
        min_r = _scatter_reduce_int(big.expand(n_r).clone(), 0, rcur, alive, "amin")
        take = (min_l[lcur] == alive) & (min_r[rcur] == alive)
        chosen.append(alive[take])
        used_l = torch.zeros(n_l, dtype=torch.bool, device=device)
        used_r = torch.zeros(n_r, dtype=torch.bool, device=device)
        used_l[lcur[take]] = True
        used_r[rcur[take]] = True
        alive = alive[~(used_l[lcur] | used_r[rcur])]
    return order[torch.cat(chosen)]


def select_pairs(
    left: Tensor, right: Tensor, sep: Tensor, find: str
) -> tuple[Tensor, Tensor, Tensor]:
    """Apply a ``find`` mode to *all* candidate pairs; result sorted by ``(left, right)``."""
    if find == "best1":
        idx = best_per_key(left, right, sep)
    elif find == "best2":
        idx = best_per_key(right, left, sep)
    elif find == "symmetric":
        idx = greedy_one_to_one(left, right, sep)
    elif find == "mutual":
        keep = torch.zeros(left.numel(), dtype=torch.bool, device=left.device)
        keep[best_per_key(left, right, sep)] = True
        keep2 = torch.zeros_like(keep)
        keep2[best_per_key(right, left, sep)] = True
        idx = torch.nonzero(keep & keep2, as_tuple=True)[0]
    else:
        idx = torch.arange(left.numel(), device=left.device)
    left, right, sep = left[idx], right[idx], sep[idx]
    order = sort_by_left(left, right, sep)
    return left[order], right[order], sep[order]


def join_indices(
    left_index: Tensor,
    right_index: Tensor,
    n_left: int,
    n_right: int,
    join: str,
) -> tuple[Tensor, Tensor]:
    """Row-index pairs for a STILTS join type; ``-1`` marks the absent side."""
    if join not in JOIN_MODES:
        raise ValueError(f"join must be one of {JOIN_MODES}, got {join!r}")
    device = left_index.device
    no_l = _unmatched(left_index, n_left)
    no_r = _unmatched(right_index, n_right)
    neg_l = torch.full_like(no_l, -1)
    neg_r = torch.full_like(no_r, -1)
    parts_l: list[Tensor] = []
    parts_r: list[Tensor] = []
    if join in ("1and2", "1or2", "all1", "all2"):
        parts_l.append(left_index)
        parts_r.append(right_index)
    if join in ("1or2", "all1", "1not2"):
        parts_l.append(no_l)
        parts_r.append(neg_l)
    if join in ("1or2", "all2", "2not1"):
        parts_l.append(neg_r)
        parts_r.append(no_r)
    if not parts_l:  # pragma: no cover - guarded by JOIN_MODES
        empty = torch.empty(0, dtype=torch.int64, device=device)
        return empty, empty
    return torch.cat(parts_l), torch.cat(parts_r)


def _unmatched(index: Tensor, n: int) -> Tensor:
    seen = torch.zeros(n, dtype=torch.bool, device=index.device)
    seen[index] = True
    return torch.nonzero(~seen, as_tuple=True)[0]


def connected_components(n: int, u: Tensor, v: Tensor) -> Tensor:
    """Smallest member index of each node's component (friends-of-friends).

    Hook-and-compress union-find: each round hooks the larger root of every
    edge onto the smaller one with a scatter-min, then fully compresses by
    pointer jumping. ``O(E log N)`` work with ``O(log N)`` rounds in practice.
    """
    device = u.device
    parent = torch.arange(n, dtype=torch.int64, device=device)
    if u.numel() == 0:
        return parent
    while True:
        pu, pv = parent[u], parent[v]
        differ = pu != pv
        if not bool(differ.any()):
            return parent
        pu, pv = pu[differ], pv[differ]
        lo, hi = torch.minimum(pu, pv), torch.maximum(pu, pv)
        parent = _scatter_reduce_int(parent, 0, hi, lo, "amin")
        while True:
            jumped = parent[parent]
            if bool((jumped == parent).all()):
                break
            parent = jumped


def label_groups(root: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """``(group_id, group_size_per_group, group_size_per_row)`` from component roots.

    Ids are contiguous and ordered by each group's smallest member row.
    """
    _, gid = torch.unique(root, return_inverse=True)
    sizes = torch.bincount(gid)
    return gid, sizes, sizes[gid]
