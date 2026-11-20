"""One-to-one (exclusive) association marginals by belief propagation.

Model. Each primary row ``i`` chooses one hypothesis ``h`` among its own
(``owner[h] == i``) with weight ``W_h``. A hypothesis fills one *slot* per
secondary catalogue: a candidate source of that catalogue, or the slot
"absent". The constraint is exclusivity: a secondary source is the counterpart
of at most one primary. The joint weight of a configuration is the product of
the chosen ``W_h``, and is zero when two primaries share a source.

The exact marginals are a sum over constrained configurations (a matching
polynomial, #P-hard in general). Loopy belief propagation gives the Bethe
approximation: exact when the candidate graph has no cycles, and equal to the
independent per-primary posteriors when no source is shared. Messages from a
source's exclusivity factor to a primary's slot, as log ratios
``log rho = log m(selected) - log m(not selected)``, satisfy

``rho_{s -> i} = 1 / (1 + sum_{l != i} r_{l -> s})``

``r_{i -> s} = M_i(slot s, without rho_{s -> i}) / M_i(slot != s)``

where ``M_i(slot v)`` is the primary's hypothesis mass whose slot in the
catalogue of ``s`` is ``v``, every hypothesis weighted by
``W_h prod_{slots of h} rho``. The posterior of ``h`` is that weight
normalised within its primary. For one secondary catalogue this is the
sum-product algorithm for bipartite matching with unmatched slack on both
sides (Bayati, Shah & Sharma 2008; Huang & Jebara 2009). Its Bethe free
energy has a unique stationary point (Vontobel 2013), reached here by
Anderson-accelerated fixed-point iteration.

All steps are segment operations, so one iteration is ``O(H * C)``. Only
primaries connected to a shared source take part: elsewhere ``rho = 1``
exactly and the result is the independent posterior.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from ..device import _to

_F64 = torch.float64


def lse_excluding_self(
    x: Tensor, seg: Tensor, n_seg: int, slack: Tensor | None
) -> Tensor:
    """``log(exp(slack[seg_e]) + sum_{f in seg_e, f != e} exp(x_f))`` per element.

    The sum without the element is taken directly for each segment's largest
    element, and as ``total - own`` for the others. The latter is never a
    cancellation: the total then also holds the largest term, so it is at
    least twice ``own``.
    """
    dev = x.device
    m = torch.full((n_seg,), -math.inf, dtype=_F64, device=dev).scatter_reduce(
        0, seg, x, "amax", include_self=True
    )
    big = m if slack is None else torch.maximum(m, slack)
    big = torch.where(torch.isfinite(big), big, torch.zeros_like(big))
    ex = torch.exp(x - big[seg])
    base = (
        torch.zeros(n_seg, dtype=_F64, device=dev)
        if slack is None
        else torch.exp(slack - big)
    )
    total = base.index_add(0, seg, ex)
    idx = torch.arange(x.numel(), device=dev)
    first = torch.where(x == m[seg], idx, torch.full_like(idx, x.numel()))
    arg = torch.full((n_seg,), x.numel(), dtype=torch.int64, device=dev).scatter_reduce(
        0, seg, first, "amin", include_self=True
    )
    is_arg = idx == arg[seg]
    total_wo_arg = base.index_add(0, seg, torch.where(is_arg, torch.zeros_like(ex), ex))
    rest = torch.where(is_arg, total_wo_arg[seg], total[seg] - ex)
    return big[seg] + torch.log(rest)


def _native_ext():
    from ._zones import _native_ext as zones_ext

    ext = zones_ext()
    return ext if ext is not None and hasattr(ext, "xmatch_segment_lse_excl") else None


class _Segments:
    """Segment layout for repeated :func:`lse_excluding_self` calls.

    On CPU with the native extension, elements are permuted once into
    segment order and each call runs the torch-free kernel
    ``xmatch_segment_lse_excl`` (same arithmetic as the torch reference, which
    serves other devices and autograd).
    """

    def __init__(self, seg: Tensor, n_seg: int) -> None:
        self.seg, self.n_seg = seg, n_seg
        self.ext = _native_ext() if seg.device.type == "cpu" else None
        if self.ext is not None:
            self.perm = torch.argsort(seg, stable=True)
            self.sorted = bool(
                torch.equal(self.perm, torch.arange(seg.numel(), device=seg.device))
            )
            off = torch.zeros(n_seg + 1, dtype=torch.int64)
            off[1:] = torch.cumsum(torch.bincount(seg, minlength=n_seg), 0)
            self.offsets = off
            self.threads = max(1, torch.get_num_threads())

    def lse_excl(self, x: Tensor, slack: Tensor | None) -> Tensor:
        grad = torch.is_grad_enabled() and (
            x.requires_grad or (slack is not None and slack.requires_grad)
        )
        if self.ext is None or grad:
            return lse_excluding_self(x, self.seg, self.n_seg, slack)
        xs = (x if self.sorted else x[self.perm]).to(_F64).contiguous()
        out = torch.empty_like(xs)
        sl = (
            torch.empty(0, dtype=_F64) if slack is None else slack.to(_F64).contiguous()
        )
        self.ext.xmatch_segment_lse_excl(self.offsets, xs, sl, out, self.threads)
        if self.sorted:
            return out
        return torch.empty_like(out).index_copy_(0, self.perm, out)


def _segment_lse(x: Tensor, seg: Tensor, n_seg: int) -> Tensor:
    m = torch.full((n_seg,), -math.inf, dtype=_F64, device=x.device).scatter_reduce(
        0, seg, x, "amax", include_self=True
    )
    shift = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    s = torch.zeros(n_seg, dtype=_F64, device=x.device).index_add(
        0, seg, torch.exp(x - shift[seg])
    )
    return shift + torch.log(s)


def exclusive_log_rho(
    owner: Tensor,
    log_weight: Tensor,
    hyp_slot: Tensor,
    slot_owner: Tensor,
    slot_group: Tensor,
    slot_source: Tensor,
    *,
    init: Tensor | None = None,
    tol: float = 1e-9,
    max_iter: int = 2000,
    strict: bool = True,
    cache: dict | None = None,
) -> Tensor:
    """Fixed-point log messages ``log rho`` per slot (0 for "absent" slots).

    ``owner [H]`` primary of each hypothesis, ``log_weight [H]``,
    ``hyp_slot [H, C]`` slot of each hypothesis per secondary catalogue,
    ``slot_owner [S]`` the primary of each slot, ``slot_group [S]`` the (primary, catalogue) group of each slot (dense ids),
    ``slot_source [S]`` the global secondary-source id (``-1`` for absent).
    ``init`` warm-starts the messages. The fixed point is found by Anderson
    acceleration of the message update (exact after a few sweeps on trees);
    gradients with respect to ``log_weight`` use the implicit function
    theorem at the fixed point. Raises if the largest message change is not
    below ``tol`` within ``max_iter`` updates, unless ``strict=False`` (used
    for the warm-started partial solves inside an outer EM loop, which ends
    with a strict solve). ``cache`` (a dict) keeps the graph preparation
    between calls on the same graph with different weights.
    """
    dev = log_weight.device
    n_slots = int(slot_group.numel())
    log_rho = (
        torch.zeros(n_slots, dtype=_F64, device=dev)
        if init is None
        else _to(init, device=dev, dtype=_F64).clone()
    )
    if cache is not None and "prep" in cache:
        prep = cache["prep"]
    else:
        prep = _prepare(owner, hyp_slot, slot_owner, slot_group, slot_source)
        if cache is not None:
            cache["prep"] = prep
    if prep is None:
        return torch.zeros_like(log_rho)
    (
        hyp,
        slots,
        a_hslot,
        msg_idx,
        n_active,
        n_cat,
        flat_a,
        inv_slot,
        groups,
        sources,
        zero_src,
        base,
    ) = prep
    a_logw = log_weight[hyp]

    def update(x: Tensor, logw: Tensor) -> Tensor:
        rho = base.index_copy(0, msg_idx, x)
        u = logw + rho[a_hslot].sum(-1)
        if inv_slot is not None:
            slot_mass = u[inv_slot]
        else:
            slot_mass = _segment_lse(u.repeat_interleave(n_cat), flat_a, n_active)
        uses = slot_mass - rho
        others = groups.lse_excl(slot_mass, None)
        log_r = (uses - others)[msg_idx]
        log_r = torch.where(
            torch.isnan(log_r), torch.full_like(log_r, -math.inf), log_r
        )
        return -sources.lse_excl(log_r, zero_src)

    x0 = log_rho[slots][msg_idx]
    x = _FixedPoint.apply(update, a_logw, x0, tol, max_iter, strict)
    out = torch.zeros_like(log_rho)
    out[slots[msg_idx]] = x
    return out


def _prepare(owner, hyp_slot, slot_owner, slot_group, slot_source):
    """Active (contested) subgraph and segment layouts; ``None`` if nothing is shared."""
    dev = owner.device
    n_slots = int(slot_group.numel())
    present = slot_source >= 0
    if not bool(present.any()):
        return None
    # Restrict to the part of the graph where a source is shared: there rho != 1.
    src = slot_source[present]
    n_src = int(src.max()) + 1
    shared_src = torch.bincount(src, minlength=n_src) >= 2
    contested = torch.zeros(n_slots, dtype=torch.bool, device=dev)
    contested[present] = shared_src[src]
    if not bool(contested.any()):
        return None
    n_cat = int(hyp_slot.shape[1])
    n_owner = int(max(int(owner.max()), int(slot_owner.max()))) + 1
    active_owner = torch.zeros(n_owner, dtype=torch.bool, device=dev)
    active_owner[slot_owner[contested]] = True
    hyp = torch.nonzero(active_owner[owner], as_tuple=True)[0]
    slots = torch.nonzero(active_owner[slot_owner], as_tuple=True)[0]
    remap = torch.full((n_slots,), -1, dtype=torch.int64, device=dev)
    remap[slots] = torch.arange(slots.numel(), device=dev)
    a_hslot = remap[hyp_slot[hyp]]  # [Ha, C] active slot ids
    a_group = torch.unique(slot_group[slots], return_inverse=True)[1]
    n_group = int(a_group.max()) + 1
    a_src_raw = slot_source[slots]
    # Only contested slots carry messages; the others stay at rho = 1.
    msg = contested[slots]
    msg_idx = torch.nonzero(msg, as_tuple=True)[0]
    msg_src = torch.unique(a_src_raw[msg_idx], return_inverse=True)[1]
    n_msg_src = int(msg_src.max()) + 1
    n_active = int(slots.numel())
    flat_a = a_hslot.reshape(-1)
    # One hypothesis per slot (pairwise association): the slot mass is a gather.
    inv_slot = None
    if n_cat == 1 and int(flat_a.numel()) == n_active:
        inv_slot = torch.empty(n_active, dtype=torch.int64, device=dev)
        inv_slot[flat_a] = torch.arange(n_active, device=dev)
    zero_src = torch.zeros(n_msg_src, dtype=_F64, device=dev)
    base = torch.zeros(n_active, dtype=_F64, device=dev)
    groups = _Segments(a_group, n_group)
    sources = _Segments(msg_src, n_msg_src)

    return (
        hyp,
        slots,
        a_hslot,
        msg_idx,
        n_active,
        n_cat,
        flat_a,
        inv_slot,
        groups,
        sources,
        zero_src,
        base,
    )


def _anderson(
    g, x: Tensor, tol: float, max_iter: int, strict: bool = True, memory: int = 10
) -> Tensor:
    """Fixed point ``x = g(x)`` by Anderson acceleration (Walker & Ni 2011).

    Type-II Anderson mixing over the last ``memory`` iterates (differences
    kept in a ring buffer), restarted whenever the residual grows past twice
    the best one or stops improving for ``memory`` steps. Raises when ``max |g(x) - x|`` is not below
    ``tol`` within ``max_iter`` evaluations, unless ``strict=False``.
    """
    n = int(x.numel())
    dev = x.device
    df = torch.empty(memory, n, dtype=_F64, device=dev)
    dg = torch.empty(memory, n, dtype=_F64, device=dev)
    gx = g(x)
    f = gx - x
    used, slot, stale = 0, 0, 0
    best = float(f.abs().max()) if n else 0.0
    for _ in range(max_iter):
        res = float(f.abs().max()) if n else 0.0
        if res <= tol:
            return gx
        stale = 0 if res < 0.99 * best else stale + 1
        if res > 2.0 * best or stale > memory:
            used, stale = 0, 0  # restart a diverging or stagnating history
        best = min(best, res)
        if used:
            a_f, a_g = df[:used], dg[:used]
            gram = a_f @ a_f.mT
            reg = 1e-12 * float(gram.diagonal().max().clamp_min(1e-300))
            gram = gram + reg * torch.eye(used, dtype=_F64, device=dev)
            gamma = torch.linalg.solve(gram, a_f @ f)
            x = (gx - gamma @ a_g).clamp(max=0.0)
        else:
            x = gx
        gx_new = g(x)
        f_new = gx_new - x
        torch.sub(f_new, f, out=df[slot])
        torch.sub(gx_new, gx, out=dg[slot])
        slot = (slot + 1) % memory
        used = min(used + 1, memory)
        gx, f = gx_new, f_new
    if not strict:
        return gx
    raise RuntimeError(
        "exclusive association did not converge "
        f"(max message change {float(f.abs().max()):.3g} after {max_iter} "
        "iterations); pass exclusive=False for independent posteriors"
    )


class _FixedPoint(torch.autograd.Function):
    """``x* = update(x*, logw)`` with implicit-function gradients in ``logw``."""

    @staticmethod
    def forward(ctx, update, logw, x0, tol, max_iter, strict):  # type: ignore[override]
        w = logw.detach()
        with torch.no_grad():
            x = _anderson(lambda v: update(v, w), x0.detach(), tol, max_iter, strict)
        ctx.update = update
        ctx.tol, ctx.max_iter = tol, max_iter
        ctx.save_for_backward(w, x)
        return x

    @staticmethod
    def backward(ctx, grad_x):  # type: ignore[override]
        w, x = ctx.saved_tensors
        with torch.enable_grad():
            xv = x.detach().requires_grad_(True)
            wv = w.detach().requires_grad_(True)
            gx = ctx.update(xv, wv)

            def adjoint(v: Tensor) -> Tensor:
                # Fixed point of v = grad_x + J_x^T v.
                (jt,) = torch.autograd.grad(gx, xv, v, retain_graph=True)
                return grad_x + jt

            with torch.no_grad():
                v = _anderson_linear(adjoint, grad_x, ctx.tol, ctx.max_iter)
            (gw,) = torch.autograd.grad(gx, wv, v, retain_graph=False)
        return None, gw, None, None, None, None


def _anderson_linear(g, x: Tensor, tol: float, max_iter: int) -> Tensor:
    scale = max(float(x.abs().max()) if x.numel() else 0.0, 1e-300)
    gx = g(x)
    f = gx - x
    xs_g, fs = [gx], [f]
    for _ in range(max_iter):
        if float(f.abs().max()) <= tol * scale:
            return gx
        if len(fs) > 1:
            df = torch.stack([fs[i + 1] - fs[i] for i in range(len(fs) - 1)], 1)
            dg = torch.stack([xs_g[i + 1] - xs_g[i] for i in range(len(xs_g) - 1)], 1)
            gram = df.mT @ df
            reg = 1e-12 * float(gram.diagonal().max().clamp_min(1e-300))
            gram = gram + reg * torch.eye(gram.shape[0], dtype=_F64, device=x.device)
            x = gx - dg @ torch.linalg.solve(gram, df.mT @ f)
        else:
            x = gx
        gx = g(x)
        f = gx - x
        xs_g.append(gx)
        fs.append(f)
        if len(fs) > 7:
            xs_g.pop(0)
            fs.pop(0)
    raise RuntimeError("exclusive association gradient did not converge")


def pairwise_exclusive(
    li: Tensor,
    rj: Tensor,
    log_w: Tensor,
    log_absent: Tensor,
    *,
    init: Tensor | None = None,
    max_iter: int = 2000,
    strict: bool = True,
    cache: dict | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Exclusive marginals for one secondary catalogue.

    ``log_w`` is ``log(pi_i LR_ij)`` per candidate pair, ``log_absent`` is
    ``log(1 - pi_i)`` per left row. Returns ``(posterior [K], p_none [N],
    log_rho [K])``.
    """
    n_left = int(log_absent.numel())
    k = int(li.numel())
    dev = log_w.device
    if k == 0:
        return (
            torch.empty(0, dtype=_F64, device=dev),
            torch.ones(n_left, dtype=_F64, device=dev),
            torch.empty(0, dtype=_F64, device=dev),
        )
    # Hypotheses: the K pairs, then one "absent" per left row; one slot each.
    owner = torch.cat((li, torch.arange(n_left, device=dev)))
    logw = torch.cat((log_w, log_absent))
    slot = torch.arange(k + n_left, device=dev)[:, None]
    source = torch.cat((rj, torch.full((n_left,), -1, dtype=torch.int64, device=dev)))
    init_full = None
    if init is not None:
        init_full = torch.cat((init, torch.zeros(n_left, dtype=_F64, device=dev)))
    rho = exclusive_log_rho(
        owner,
        logw,
        slot,
        owner,
        owner,
        source,
        init=init_full,
        max_iter=max_iter,
        strict=strict,
        cache=cache,
    )[:k]
    u = torch.cat((log_w + rho, log_absent))
    z = _segment_lse(u, owner, n_left)
    return torch.exp(u[:k] - z[li]), torch.exp(log_absent - z), rho


__all__ = ["exclusive_log_rho", "lse_excluding_self", "pairwise_exclusive"]
