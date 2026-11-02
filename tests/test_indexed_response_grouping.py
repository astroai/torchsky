"""Sort-based grouping in IndexedResponsePlan matches a brute-force reference."""

import pytest
import torch

from torchsky.map.operators import IndexedResponsePlan
from torchsky.map.operators.base import DomainSpec


def _plan(source=37, target=53, entries=400, chunk=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    src = torch.randint(0, source, (entries,), generator=g)
    tgt = torch.randint(0, target, (entries,), generator=g)
    coef = torch.rand(entries, generator=g, dtype=torch.float64)
    ids = torch.randint(0, 70, (entries,), generator=g)
    plan = IndexedResponsePlan(
        DomainSpec((source,), torch.float64),
        DomainSpec((target,), torch.float64),
        src,
        tgt,
        coef,
        contributor_ids=ids,
        default_chunk_size=chunk,
    )
    return plan, src, tgt, coef, ids


@pytest.mark.parametrize("chunk", (None, 1, 7, 53, 1000))
def test_apply_adjoint_squared_match_dense(chunk):
    plan, src, tgt, coef, _ = _plan()
    dense = torch.zeros(53, 37, dtype=torch.float64)
    dense.index_put_((tgt, src), coef, accumulate=True)
    squared = torch.zeros(53, 37, dtype=torch.float64)
    squared.index_put_((tgt, src), coef.square(), accumulate=True)
    x = torch.randn(37, dtype=torch.float64)
    y = torch.randn(53, dtype=torch.float64)
    torch.testing.assert_close(plan.apply(x, chunk_size=chunk), dense @ x)
    torch.testing.assert_close(plan.adjoint(y, chunk_size=chunk), dense.T @ y)
    torch.testing.assert_close(plan.apply_squared(x, chunk_size=chunk), squared @ x)


@pytest.mark.parametrize("dtype", (torch.uint8, torch.int16, torch.int32, torch.int64))
@pytest.mark.parametrize("chunk", (3, None))
def test_flags_and_contributors_match_loops(dtype, chunk):
    plan, src, tgt, _, ids = _plan(seed=3)
    info = torch.iinfo(dtype)
    g = torch.Generator().manual_seed(4)
    flags = torch.randint(info.min, info.max, (37,), generator=g, dtype=dtype)
    expected = torch.zeros(53, dtype=dtype)
    for s, t in zip(src.tolist(), tgt.tolist()):
        expected[t] |= flags[s]
    assert torch.equal(plan.apply_flags(flags, chunk_size=chunk), expected)
    pairs = {}
    context = torch.zeros(3, 53, dtype=torch.int64)
    for t, c in zip(tgt.tolist(), ids.tolist()):
        pairs.setdefault(t, set()).add(c)
        context[c // 32, t] |= 1 << (c % 32)
    counts = torch.tensor([len(pairs.get(t, ())) for t in range(53)])
    assert torch.equal(plan.contributor_count(chunk_size=chunk), counts)
    assert torch.equal(plan.context(chunk_size=chunk), context)


def test_empty_plan_and_no_host_sync_per_chunk():
    plan = IndexedResponsePlan(
        DomainSpec((4,), torch.float32),
        DomainSpec((6,), torch.float32),
        torch.empty(0, dtype=torch.int64),
        torch.empty(0, dtype=torch.int64),
        torch.empty(0),
        default_chunk_size=2,
    )
    assert plan.apply(torch.ones(4)).abs().sum() == 0
    assert plan.contributor_count().sum() == 0
    # Chunk boundaries are computed once per chunk size and cached.
    assert len(plan._bounds) == 1
