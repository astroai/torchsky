"""Blend-aware association: a secondary row may be the image of two primaries.

Low-resolution catalogues (WISE, a 6" PSF) often report one source where a
high-resolution primary catalogue (Gaia) has two. One-to-one models then
either drop a true counterpart or explain the blended position, the
flux-weighted centroid, by inflating the error model. This module adds the
blend hypothesis to the exclusive (one-to-one) model of :mod:`._exclusive`.

**Model.** Two kinds of sources emit secondary detections: each primary
``i`` alone, with probability ``pi_i`` (completeness), and each pair
``{a, b}`` of primaries closer than the secondary's resolution
(``blend_radius_arcsec``), with probability ``eta`` (the blend rate), at the
pair's flux-weighted centroid. The sources are independent, conditioned on
each primary contributing to at most one detection. Every other secondary row
is background (Poisson, density ``nu``). The marginal likelihood (up to the
background terms of the exclusive model) is ``Z_data / Z_prior``:

- ``Z_data`` sums, over the states of every secondary row (background, one
  primary, one pair) that use each primary at most once, the products of
  ``o_i N(x_j; x_i, Sigma_ij) / nu_j`` (the exclusive model's weight,
  ``o = pi / (1 - pi)``) and ``o_eta B_jab``;
- ``Z_prior`` is the same sum without data: the monomer-dimer partition
  function of the primary pair graph,
  ``prod_i (1 - pi_i)^-1 sum_M prod_{ab in M} o_eta (1 - pi_a)(1 - pi_b)``.

With ``eta = 0`` this is the exclusive model. The conditioning is what makes
``eta`` a probability: raising it also raises ``Z_prior``, so the fit cannot
explain every secondary by blends. ``B_jab = int N(x_j; f x_a + (1 - f) x_b,
Sigma(f)) p(f) df / nu_j``, with ``f`` the flux fraction of ``a`` in the
secondary's band:

- given primary fluxes (``blend_flux``), ``f = F_a / (F_a + F_b)`` and ``B`` is
  one Gaussian (exact);
- otherwise ``f ~ Uniform(0, 1)`` and the integral along the segment is in
  closed form (an error function). Primary covariances enter ``Sigma(f) =
  s^2 C_j + F + f^2 C_a + (1 - f)^2 C_b`` at the most probable ``f``; exact
  when they are negligible against the secondary's errors, as for Gaia
  against WISE.

``eta`` is fitted by maximum marginal likelihood with completeness, the error
calibration and the background density.

**Solver.** The secondaries own the hypotheses (background, singles, pairs);
each primary carries an at-most-one constraint on its edges ``(j, i)``. Loopy
belief propagation on that factor graph gives the Bethe marginals and
``log Z`` (exact on cycle-free graphs, as the exclusive model), with the same
Anderson-accelerated fixed point and implicit-function gradients;
``Z_prior`` uses the same solver with one owner per primary pair (BP on
monomer-dimer models converges to a unique fixed point). One sweep is
``O(H)`` segment operations for ``H`` hypotheses, so memory is linear in the
gated pair hypotheses and in the primary pairs within the blend radius: the
pair search runs in chunks of primary pairs and keeps only pairs whose
centroid passes the chi^2 gate.
An exact solve on small connected components was not chosen: in crowded
fields the components percolate (one component spans the field), so a
per-component enumeration has no useful bound.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from ._exclusive import (
    _MEMORY,
    _FixedPoint,
    _Segments,
    _anderson,
    _segment_lse,
)

_F64 = torch.float64
_LOG_2PI = math.log(2.0 * math.pi)
# Segment length (in units of the error ellipse) below which the closed form
# is replaced by the midpoint value: the relative error is a / 24 < 1e-9.
_A_SMALL = 1e-8
# Edges whose owner-side probability exceeds 1 - 1e-6 get a direct sum for
# the "edge off" mass (the subtraction elsewhere loses at most 1e-10).
_LOG_CRIT = math.log1p(-1e-6)


@dataclass
class BlendAssociation:
    """Blend posteriors of a :func:`associate` run with ``blends="pairs"``.

    Per pair hypothesis (``B`` of them): ``right_index`` (secondary row),
    ``members [B, 2]`` (the two primary rows, ascending), ``posterior`` (P that
    the secondary row is the blend of exactly these two), ``flux_fraction``
    (flux share of ``members[:, 0]``: given by ``blend_flux``, else its
    posterior mean under the uniform prior) and ``log_lr`` (natural log of
    ``B_jab``, the blend likelihood ratio against the background, without
    priors). Per candidate pair of the parent :class:`Association`:
    ``p_single`` (P that the right row is the image of that left row alone).
    Per right row: ``p_background``, ``map_members [M, 2]`` (members of the
    most probable hypothesis, ``-1`` padded; ``[-1, -1]`` for background) and
    ``map_posterior``. ``blend_rate`` is the fitted probability ``eta`` that
    a pair of left rows closer than the blend radius is detected as one
    blended right row; ``n_pair_sources`` counts those pairs.
    """

    right_index: Tensor
    members: Tensor
    posterior: Tensor
    flux_fraction: Tensor
    log_lr: Tensor
    p_single: Tensor
    p_background: Tensor
    map_members: Tensor
    map_posterior: Tensor
    blend_rate: float
    n_pair_sources: int


# ------------------------------------------------------------ blend likelihood


def _inv2(c: Tensor) -> tuple[Tensor, Tensor]:
    """Inverse and log determinant of ``[..., 2, 2]`` symmetric matrices."""
    a, b, d = c[..., 0, 0], c[..., 0, 1], c[..., 1, 1]
    det = a * d - b * b
    inv = (
        torch.stack((torch.stack((d, -b), -1), torch.stack((-b, a), -1)), -2)
        / det[..., None, None]
    )
    return inv, torch.log(det)


def _quad(x: Tensor, p: Tensor, y: Tensor) -> Tensor:
    return (x[..., :, None] * p * y[..., None, :]).sum((-2, -1))


def _log_ndtr_diff(hi: Tensor, lo: Tensor) -> Tensor:
    """``log(Phi(hi) - Phi(lo))`` for ``hi > lo``, using the far tail."""
    flip = lo > 0
    top = torch.where(flip, -lo, hi)
    bot = torch.where(flip, -hi, lo)
    lt, lb = torch.special.log_ndtr(top), torch.special.log_ndtr(bot)
    return lt + torch.log(-torch.expm1(lb - lt))


def blend_log_density(
    ua: Tensor,
    ub: Tensor,
    ca: Tensor,
    cb: Tensor,
    s: Tensor,
    frac: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Log density of a secondary at the origin given a blend of two primaries.

    ``ua, ub [P, 2]`` are the primaries' offsets from the secondary (mas, its
    tangent plane), ``ca, cb [P, 2, 2]`` their covariances there, ``s`` the
    secondary covariance (calibrated, floor included) and ``frac [P]`` the
    flux fraction of ``a`` (NaN: uniform prior on ``[0, 1]``). Returns
    ``(log density, d2 gate statistic, flux fraction)``: ``d2`` is the
    Mahalanobis distance of the secondary to the centroid (at the given
    fraction, or the closest point of the segment), the fraction is the given
    one or its posterior mean.
    """
    fixed = torch.isfinite(frac)
    d = ua - ub
    # Most probable fraction with the primaries' errors averaged in, then the
    # covariance at that fraction (exact when the primaries' errors are small).
    p0, _ = _inv2(s + 0.5 * (ca + cb))
    a0 = _quad(d, p0, d)
    f0 = torch.where(a0 > 0, -_quad(d, p0, ub) / a0.clamp_min(1e-300), 0.5)
    f0 = torch.where(fixed, frac, f0.clamp(0.0, 1.0))
    f0_ = f0[:, None, None]
    sig = s + f0_ * f0_ * ca + (1.0 - f0_) * (1.0 - f0_) * cb
    p, logdet = _inv2(sig)
    a = _quad(d, p, d)
    b = _quad(d, p, ub)
    c = _quad(ub, p, ub)
    # Given fraction: one Gaussian.
    r_fix = ub + f0[:, None] * d
    q_fix = _quad(r_fix, p, r_fix)
    log_fix = -_LOG_2PI - 0.5 * logdet - 0.5 * q_fix
    # Uniform fraction: Gaussian in f along the segment, integrated over [0, 1].
    small = a < _A_SMALL
    a_s = torch.where(small, torch.ones_like(a), a)
    fh = -b / a_s
    qmin = (c - b * b / a_s).clamp_min(0.0)
    ra = torch.sqrt(a_s)
    hi, lo = ra * (1.0 - fh), -ra * fh
    log_mass = _log_ndtr_diff(hi, lo)
    log_uni = (
        -_LOG_2PI
        - 0.5 * logdet
        - 0.5 * qmin
        + 0.5 * (_LOG_2PI - torch.log(a_s))
        + log_mass
    )
    fc = fh.clamp(0.0, 1.0)
    r_c = ub + fc[:, None] * d
    q_seg = _quad(r_c, p, r_c)
    # Posterior mean of a truncated normal on [0, 1].
    log_phi = lambda t: -0.5 * t * t - 0.5 * _LOG_2PI  # noqa: E731
    mean = (
        fh
        + (torch.exp(log_phi(lo) - log_mass) - torch.exp(log_phi(hi) - log_mass)) / ra
    )
    r_mid = ub + 0.5 * d
    log_mid = -_LOG_2PI - 0.5 * logdet - 0.5 * _quad(r_mid, p, r_mid)
    log_uni = torch.where(small, log_mid, log_uni)
    mean = torch.where(small, torch.full_like(mean, 0.5), mean.clamp(0.0, 1.0))
    q_seg = torch.where(small, _quad(r_mid, p, r_mid), q_seg)
    return (
        torch.where(fixed, log_fix, log_uni),
        torch.where(fixed, q_fix, q_seg),
        torch.where(fixed, frac, mean),
    )


def edge_geometry(lx: Tensor, rx: Tensor, lcov: Tensor) -> tuple[Tensor, Tensor]:
    """Offsets of primaries from secondaries in the secondary's tangent plane.

    ``lx, rx [E, 3]`` unit vectors, ``lcov [E, 2, 2]`` the primary covariances
    in their own local basis. Returns ``(u [E, 2] mas, C [E, 2, 2])`` in the
    secondary's (east, north) basis: ``u`` is the exact log map (direction
    and great-circle distance) and ``C`` is parallel transported.
    """
    from .association import pair_frame

    delta, tl, tr, _ = pair_frame(lx, rx)
    trt = tr.mT
    u = -(trt @ delta[..., None])[..., 0]
    m = trt @ tl
    return u, m @ lcov @ m.mT


# ------------------------------------------------------------------- solver


class _BlendGraph:
    """Secondary-owned hypotheses with at-most-one constraints on primaries.

    Hypotheses ``h`` (background excluded, it is each owner's slack of weight
    1) have an owner ``hyp_owner`` and one or two edges ``e1``, ``e2`` (``-1``
    for none). Edges ``e`` join an owner and a primary. Messages ``x_e`` are
    ``log rho``: the log probability, from the primary's constraint, that no
    other owner takes the primary. One sweep:

    - ``u_h = log W_h + x_e1 + x_e2``;
    - ``r_e = log M_j(e on, without x_e) - log M_j(e off)`` from the owner;
    - ``x_e = -log(1 + sum_{e' of i, e' != e} exp r_e')`` from the primary.

    ``M_j(e off)`` is ``Z_j - M_j(e on)`` unless ``e`` holds almost all of
    the owner's mass; then it is summed directly (no cancellation).
    """

    def __init__(
        self,
        hyp_owner: Tensor,
        e1: Tensor,
        e2: Tensor,
        edge_owner: Tensor,
        edge_primary: Tensor,
        n_owner: int,
        n_primary: int,
    ) -> None:
        dev = hyp_owner.device
        self.hyp_owner, self.e1, self.e2 = hyp_owner, e1, e2
        self.edge_owner, self.edge_primary = edge_owner, edge_primary
        self.n_owner, self.n_primary = n_owner, n_primary
        self.n_edge = int(edge_owner.numel())
        self.has2 = e2 >= 0
        h_all = torch.arange(int(hyp_owner.numel()), device=dev)
        self.inc_h = torch.cat((h_all, h_all[self.has2]))
        self.inc_e = torch.cat((e1, e2[self.has2]))
        self.e2c = e2.clamp(min=0)
        self.owner_slack = torch.zeros(n_owner, dtype=_F64, device=dev)
        self.primaries = _Segments(edge_primary, n_primary)
        self.zero_p = torch.zeros(n_primary, dtype=_F64, device=dev)
        self.last_sweeps = 0
        # Hypotheses of each owner (CSR) for the direct sums.
        self.h_sorted = torch.argsort(hyp_owner, stable=True)
        self.h_cnt = torch.bincount(hyp_owner, minlength=n_owner)
        self.h_off = torch.cumsum(self.h_cnt, 0) - self.h_cnt
        # Monomer-dimer layout (the prior graph): owner h has the single
        # hypothesis h with edges h and h + n_owner.
        ar = torch.arange(n_owner, device=dev)
        self.dimer = bool(
            int(hyp_owner.numel()) == n_owner
            and self.n_edge == 2 * n_owner
            and torch.equal(hyp_owner, ar)
            and torch.equal(e1, ar)
            and torch.equal(e2, ar + n_owner)
        )

    def utility(self, x: Tensor, logw: Tensor) -> Tensor:
        return logw + x[self.e1] + torch.where(self.has2, x[self.e2c], 0.0)

    def owner_lse(self, u: Tensor) -> Tensor:
        """``log(1 + sum_h exp u_h)`` per owner (background slack 1)."""
        return _lse_slack(u, self.hyp_owner, self.n_owner, self.owner_slack)

    def edge_ratios(self, x: Tensor, logw: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """``(u, log Z_j, r_e)`` at messages ``x``."""
        if self.dimer:
            # One owner per pair, one hypothesis, edges h and h + n: the owner
            # sums are closed form, r_e = log W + x(partner edge).
            n = self.n_owner
            u = logw + x[:n] + x[n:]
            r = torch.cat((logw + x[n:], logw + x[:n]))
            return u, torch.nn.functional.softplus(u), r
        u = self.utility(x, logw)
        log_z = self.owner_lse(u)
        on = _segment_lse(u[self.inc_h], self.inc_e, self.n_edge)
        lz = log_z[self.edge_owner]
        diff = (on - lz).clamp(max=0.0)
        off = lz + torch.log(-torch.expm1(diff))
        # Z_j - M_j(e on) loses eps / P(e off) relatively: at most 1e-10
        # unless P(e on) > 1 - 1e-6. Those few edges are summed directly.
        crit = torch.nonzero(diff.detach() > _LOG_CRIT, as_tuple=True)[0]
        if crit.numel():
            off = off.index_put((crit,), self._off_direct(u, crit))
        r = on - x - off
        r = torch.where(torch.isnan(r), torch.full_like(r, -math.inf), r)
        return u, log_z, r

    def _off_direct(self, u: Tensor, edges: Tensor) -> Tensor:
        """``log(1 + sum of exp u_h over the owner's hypotheses without e)``
        for the given edges, from the owner's hypothesis list."""
        own = self.edge_owner[edges]
        cnt = self.h_cnt[own]
        seg = torch.repeat_interleave(
            torch.arange(int(edges.numel()), device=u.device), cnt
        )
        first = torch.cumsum(cnt, 0) - cnt
        pos = torch.arange(int(seg.numel()), device=u.device) - first[seg]
        hh = self.h_sorted[self.h_off[own][seg] + pos]
        e = edges[seg]
        hold = (self.e1[hh] == e) | (self.e2[hh] == e)
        uu = torch.where(hold, torch.full_like(u[hh], -math.inf), u[hh])
        return _lse_slack(
            uu,
            seg,
            int(edges.numel()),
            torch.zeros(int(edges.numel()), dtype=_F64, device=u.device),
        )

    def sweep(self, x: Tensor, logw: Tensor, *, native: bool | None = None) -> Tensor:
        _, _, r = self.edge_ratios(x, logw)
        return -self.primaries.lse_excl(r, self.zero_p)

    def solve(
        self,
        x0: Tensor,
        logw: Tensor,
        tol: float,
        max_iter: int,
        strict: bool,
        *,
        native: bool | None = None,
    ) -> Tensor:
        def g(v: Tensor) -> Tensor:
            return self.sweep(v, logw)

        x, self.last_sweeps = _anderson(g, x0, tol, max_iter, strict, memory=_MEMORY)
        return x

    def messages(
        self,
        logw: Tensor,
        init: Tensor | None,
        *,
        tol: float = 1e-9,
        max_iter: int = 2000,
        strict: bool = True,
    ) -> Tensor:
        x0 = (
            torch.zeros(self.n_edge, dtype=_F64, device=logw.device)
            if init is None
            else init.to(_F64)
        )
        if self.n_edge == 0:
            return x0
        return _FixedPoint.apply(self, logw, x0, tol, max_iter, strict)

    def marginals(
        self, x: Tensor, logw: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """``(P(h), P(background) per owner, P(edge), P(primary unused))``.

        Hypothesis posteriors are normalised per owner; edge and primary
        posteriors per primary (they agree with the owner's at the fixed
        point), so each primary's probabilities sum to one.
        """
        u, log_z, r = self.edge_ratios(x, logw)
        p_h = torch.exp(u - log_z[self.hyp_owner])
        p_bg = torch.exp(-log_z)
        log_den = _lse_slack(r, self.edge_primary, self.n_primary, self.zero_p)
        p_e = torch.exp(r - log_den[self.edge_primary])
        return p_h, p_bg, p_e, torch.exp(-log_den)

    def log_partition(self, x: Tensor, logw: Tensor) -> Tensor:
        """Bethe ``log Z`` at messages ``x`` (held fixed: envelope gradients).

        ``sum_j log Z_j + sum_i log(1 + sum_e exp r_e) - sum_e log(1 + exp(r_e + x_e))``;
        only the owners' term depends on the weights directly, and at a fixed
        point the Bethe free energy is stationary in the messages, so the
        gradient in ``logw`` is the posterior-weighted one.
        """
        x = x.detach()
        u = self.utility(x, logw)
        owners = self.owner_lse(u).sum()
        with torch.no_grad():
            _, _, r = self.edge_ratios(x, logw.detach())
            prim = _lse_slack(r, self.edge_primary, self.n_primary, self.zero_p).sum()
            edges = torch.nn.functional.softplus(r + x).sum()
        return owners + (prim - edges)


def dimer_graph(a: Tensor, b: Tensor) -> _BlendGraph:
    """Monomer-dimer graph of primary pairs ``(a, b)``: one owner per pair
    with one hypothesis covering both (edges ``p`` and ``p + n``)."""
    n = int(a.numel())
    pid, prim = torch.unique(torch.cat((a, b)), return_inverse=True)
    ar = torch.arange(n, device=a.device)
    return _BlendGraph(ar, ar, ar + n, torch.cat((ar, ar)), prim, n, int(pid.numel()))


def _lse_slack(x: Tensor, seg: Tensor, n_seg: int, slack: Tensor) -> Tensor:
    """``log(exp(slack) + sum_{e in seg} exp(x_e))`` per segment."""
    m = torch.full((n_seg,), -math.inf, dtype=_F64, device=x.device).scatter_reduce(
        0, seg, x.detach(), "amax", include_self=True
    )
    m = torch.maximum(m, slack.detach())
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    s_ = torch.exp(slack - m).index_add(0, seg, torch.exp(x - m[seg]))
    return m + torch.log(s_)


# ---------------------------------------------------------------- candidates


def primary_pairs(
    llon: Tensor, llat: Tensor, ok: Tensor, blend_radius_arcsec: float
) -> tuple[Tensor, Tensor, Tensor]:
    """Pairs ``(a, b, sep_arcsec)``, ``a < b``, of rows of ``ok`` closer than the
    blend radius (exact zone-index search)."""
    from ._zones import (
        ZoneIndex,
        default_zone_height_deg,
        occupied_spacing_deg,
        zone_query,
    )

    dev = llon.device
    rows = torch.nonzero(ok, as_tuple=True)[0]
    n = int(rows.numel())
    if n < 2:
        e = torch.empty(0, dtype=torch.int64, device=dev)
        return e, e, torch.empty(0, dtype=_F64, device=dev)
    lon, lat = llon[rows], llat[rows]
    idx = ZoneIndex(
        lon,
        lat,
        zone_height_deg=default_zone_height_deg(
            n, blend_radius_arcsec, occupied_spacing_deg(lon, lat)
        ),
    )
    br = torch.full((n,), float(blend_radius_arcsec), dtype=_F64, device=dev)
    a, b, sep = zone_query(idx, lon, lat, br, self_upper=True, find="all")
    keep = b > a
    return rows[a[keep]], rows[b[keep]], sep[keep]


def pair_candidates(
    pairs: tuple[Tensor, Tensor, Tensor],
    lx: Tensor,
    rlon: Tensor,
    rlat: Tensor,
    right_ok: Tensor,
    rl: Tensor,
    rr: Tensor,
    accept,
    *,
    chunk: int = 1 << 18,
) -> tuple[Tensor, Tensor, Tensor]:
    """Gated blend triples ``(a, b, j)`` from primary ``pairs`` (``a < b``).

    Per chunk of primary pairs, the secondaries within
    ``h + max(rl_a, rl_b) + rr_j`` of the pair's midpoint (``h`` its half
    separation). That window holds every secondary whose distance to some
    point of the segment is at most ``max(rl) + rr``, the bound implied by
    the chi^2 gate. ``accept(a, b, j)`` applies the exact gate to a chunk and
    returns a boolean mask; only accepted triples are kept, so peak memory is
    one chunk. Triples are ordered by ``(j, a, b)``.
    """
    from ._zones import (
        ZoneIndex,
        default_zone_height_deg,
        occupied_spacing_deg,
        zone_query,
    )

    a, b, sep = pairs
    dev = lx.device
    e = torch.empty(0, dtype=torch.int64, device=dev)
    rows = torch.nonzero(right_ok, as_tuple=True)[0]
    if a.numel() == 0 or rows.numel() == 0:
        return e, e, e
    rr_ok = rr[rows]
    ridx = ZoneIndex(
        rlon[rows],
        rlat[rows],
        zone_height_deg=default_zone_height_deg(
            int(rows.numel()),
            float(rl.median()) + float(rr_ok.max()) + float(sep.max()),
            occupied_spacing_deg(rlon[rows], rlat[rows]),
        ),
    )
    out_a, out_b, out_j = [], [], []
    for s in range(0, int(a.numel()), chunk):
        aa, bb = a[s : s + chunk], b[s : s + chunk]
        mid = lx[aa] + lx[bb]
        mid = mid / mid.norm(dim=-1, keepdim=True)
        mlon = torch.rad2deg(torch.atan2(mid[:, 1], mid[:, 0])) % 360.0
        mlat = torch.rad2deg(torch.asin(mid[:, 2].clamp(-1.0, 1.0)))
        # Half separation, plus 1e-6" for the plane-sphere difference of the
        # centroid (below 1e-9" at 10").
        h = 0.5 * sep[s : s + chunk] + 1e-6
        rad = h + torch.maximum(rl[aa], rl[bb])
        p, jj, _ = zone_query(
            ridx, mlon, mlat, rad, radius_right=rr_ok, combine="sum", find="all"
        )
        ta, tb, tj = aa[p], bb[p], rows[jj]
        if ta.numel() == 0:
            continue
        ok = accept(ta, tb, tj)
        out_a.append(ta[ok])
        out_b.append(tb[ok])
        out_j.append(tj[ok])
    if not out_a:
        return e, e, e
    ta, tb, tj = torch.cat(out_a), torch.cat(out_b), torch.cat(out_j)
    order = _lexsort(tj, ta, tb)
    return ta[order], tb[order], tj[order]


def _lexsort(*keys: Tensor) -> Tensor:
    order = torch.argsort(keys[-1], stable=True)
    for k in reversed(keys[:-1]):
        order = order[torch.argsort(k[order], stable=True)]
    return order


def greedy_hypotheses(
    hyp_owner: Tensor,
    m1: Tensor,
    m2: Tensor,
    p_h: Tensor,
    p_bg: Tensor,
    n_primary: int,
) -> Tensor:
    """Selected hypotheses: each owner's most probable one (if not background),
    accepted greedily by posterior so that no primary is used twice."""
    nh = int(p_h.numel())
    dev = p_h.device
    sel = torch.zeros(nh, dtype=torch.bool, device=dev)
    if nh == 0:
        return sel
    n_owner = int(p_bg.numel())
    best = torch.zeros(n_owner, dtype=_F64, device=dev).scatter_reduce(
        0, hyp_owner, p_h, "amax", include_self=False
    )
    cand = (p_h == best[hyp_owner]) & (p_h > p_bg[hyp_owner])
    idx = torch.nonzero(cand, as_tuple=True)[0]
    # One per owner (ties: lowest index).
    _, first = torch.unique(hyp_owner[idx], return_inverse=True)
    keep = torch.zeros(
        int(first.max()) + 1 if idx.numel() else 0, dtype=torch.int64, device=dev
    )
    keep = keep.fill_(nh).scatter_reduce(0, first, idx, "amin")
    idx = keep
    order = torch.argsort(-p_h[idx], stable=True)
    idx = idx[order]
    used = torch.zeros(n_primary, dtype=torch.bool)
    for h, a, b in zip(idx.tolist(), m1[idx].tolist(), m2[idx].tolist()):
        if used[a] or (b >= 0 and used[b]):
            continue
        used[a] = True
        if b >= 0:
            used[b] = True
        sel[h] = True
    return sel


# ------------------------------------------------------------------- driver


def associate_blends(
    *,
    llon: Tensor,
    llat: Tensor,
    lcov: Tensor,
    lam_l: Tensor,
    rlon: Tensor,
    rlat: Tensor,
    rcov: Tensor,
    lam_r: Tensor,
    rbin: Tensor,
    scale: Tensor,
    floor: Tensor,
    dens,
    inside: Tensor,
    k2: float,
    gate_pvalue: float,
    max_radius_arcsec: float,
    completeness,
    calibrate: str,
    calibrate_min_pairs: float,
    assign: str,
    blend_radius_arcsec: float,
    blend_flux: Tensor | None,
    pi_start: float | None = None,
):
    """:func:`~torchsky.catalogs.associate` with ``blends="pairs"``.

    Same rounds as the exclusive model (gate, candidates, maximum marginal
    likelihood fit, regenerate until the calibration is stable), with pair
    hypotheses gated at each round's calibration and the blend prior
    ``kappa`` fitted with completeness, calibration and background density.
    """
    from . import association as A
    from ..device import _as_tensor
    from ._zones import unit_vectors

    dev = llon.device
    k = math.sqrt(k2)
    n_left, n_right = int(llon.numel()), int(rlon.numel())
    n_bins = int(scale.numel())
    lx, rx = unit_vectors(llon, llat), unit_vectors(rlon, rlat)
    right_inside = dens.w_in > 0
    log_nu_cell = torch.log(dens.cell_nu().clamp_min(1e-300))
    flux: Tensor | None = None
    if blend_flux is not None:
        flux = _as_tensor(blend_flux, dtype=_F64, device=dev).reshape(-1)
        if flux.numel() != n_left:
            raise ValueError("blend_flux needs one value per left row")
        if bool((flux < 0).any()):
            raise ValueError("blend_flux must be non-negative (NaN for unknown)")
    fit_pi = completeness == "em"
    if not fit_pi and not 0.0 <= float(completeness) <= 1.0:
        raise ValueError("completeness must lie in [0, 1] or be 'em'")
    pi_c = (
        float(completeness) if not fit_pi else (0.5 if pi_start is None else pi_start)
    )
    rounds = 1 if calibrate == "none" else 8
    outside_log = torch.where(
        inside > 0, torch.zeros_like(inside), torch.full_like(inside, -math.inf)
    )
    prev: tuple[Tensor, Tensor, Tensor] | None = None
    prev_state: dict = {}
    # Pair sources: left rows in the footprint closer than the blend radius.
    # They do not depend on the calibration, so the prior (monomer-dimer)
    # graph is built once.
    dimers = primary_pairs(llon, llat, inside > 0, blend_radius_arcsec)
    da, db = dimers[0], dimers[1]
    n_dim = int(da.numel())
    prior_graph = dimer_graph(da, db) if n_dim else None
    # Start the blend rate at a tenth of the singles' expected rate per pair.
    eta0 = min(max(0.1 * pi_c * float(inside.sum()) / max(n_dim, 1), 1e-6), 0.5)
    logit_eta = torch.tensor(
        [math.log(eta0) - math.log1p(-eta0)], dtype=_F64, device=dev
    )

    def frac_of(a: Tensor, b: Tensor) -> Tensor:
        if flux is None:
            return torch.full(a.shape, math.nan, dtype=_F64, device=dev)
        fa, fb = flux[a], flux[b]
        tot = fa + fb
        ok = torch.isfinite(tot) & (tot > 0)
        return torch.where(ok, fa / tot.clamp_min(1e-300), math.nan)

    def pair_geom(a: Tensor, b: Tensor, j: Tensor):
        ua, ca = edge_geometry(lx[a], rx[j], lcov[a])
        ub, cb = edge_geometry(lx[b], rx[j], lcov[b])
        return ua, ub, ca, cb

    for _round in range(rounds):
        lam_f = float(A._lambda_max(floor).max())
        rl_raw = k * (torch.sqrt(lam_l) + math.sqrt(lam_f)) / 1000.0
        rr_raw = k * scale[rbin] * torch.sqrt(lam_r) / 1000.0
        rl = rl_raw.clamp(max=max_radius_arcsec)
        rr = rr_raw.clamp(max=max_radius_arcsec)
        n_capped = int(
            (rl_raw > max_radius_arcsec).sum() + (rr_raw > max_radius_arcsec).sum()
        )
        # Single hypotheses: the exclusive model's gated candidates.
        li, rj = A._sound_candidates(llon, llat, rlon, rlat, rl, rr)
        delta, tl, tr, psi = A.pair_frame(lx[li], rx[rj])
        cl = tl @ lcov[li] @ tl.mT
        cr = tr @ rcov[rj] @ tr.mT
        pbs = rbin[rj]
        d2, _ = A.mahalanobis2(
            delta, cl + (scale[pbs] ** 2)[:, None, None] * cr + floor[pbs]
        )
        keep = (
            (d2 <= k2)
            & (psi * A._ARCSEC_PER_RAD <= max_radius_arcsec)
            & right_inside[rj]
        )
        li, rj, delta, cl, cr = li[keep], rj[keep], delta[keep], cl[keep], cr[keep]
        pbs = rbin[rj]
        # Pair hypotheses, gated at this round's calibration.
        sc_now, fl_now = scale.clone(), floor.clone()

        def accept(a: Tensor, b: Tensor, j: Tensor) -> Tensor:
            ua, ub, ca, cb = pair_geom(a, b, j)
            bj = rbin[j]
            s = (sc_now[bj] ** 2)[:, None, None] * rcov[j] + fl_now[bj]
            _, q, _ = blend_log_density(ua, ub, ca, cb, s, frac_of(a, b))
            return q <= k2

        pa, pb, pj = pair_candidates(
            dimers, lx, rlon, rlat, right_inside, rl, rr, accept
        )
        ua, ub, ca, cb = pair_geom(pa, pb, pj)
        frac = frac_of(pa, pb)
        rc_p, pbin = rcov[pj], rbin[pj]
        # Edges (left, right) of all hypotheses, ordered by left then right.
        n_s, n_p = int(li.numel()), int(pa.numel())
        keys = torch.cat((li * n_right + rj, pa * n_right + pj, pb * n_right + pj))
        ekeys, inv = torch.unique(keys, return_inverse=True)
        e_s, e_a, e_b = inv[:n_s], inv[n_s : n_s + n_p], inv[n_s + n_p :]
        edge_i, edge_j = ekeys // n_right, ekeys % n_right
        oid, hyp_owner = torch.unique(torch.cat((rj, pj)), return_inverse=True)
        pid, edge_primary = torch.unique(edge_i, return_inverse=True)
        graph = _BlendGraph(
            hyp_owner,
            torch.cat((e_s, e_a)),
            torch.cat((torch.full((n_s,), -1, dtype=torch.int64, device=dev), e_b)),
            torch.searchsorted(oid, edge_j),
            edge_primary,
            int(oid.numel()),
            int(pid.numel()),
        )
        m1 = torch.cat((li, pa))
        m2 = torch.cat((torch.full((n_s,), -1, dtype=torch.int64, device=dev), pb))
        state: dict = {"extra": logit_eta}
        if "y" in prev_state:
            state["y"] = prev_state["y"]
        if prev is not None:
            from .association import _carry_messages

            state["x"] = _carry_messages(*prev, edge_i, edge_j, n_right)

        def weights(pi_c, sc: Tensor, fl: Tensor, lnu: Tensor, extra: Tensor):
            sig = cl + (sc[pbs] ** 2)[:, None, None] * cr + fl[pbs]
            d2_, logdet_ = A.mahalanobis2(delta, sig)
            log_nu_r = dens.log_right(lnu)
            ls = -_LOG_2PI - 0.5 * logdet_ - 0.5 * d2_ - log_nu_r[rj]
            s = (sc[pbin] ** 2)[:, None, None] * rc_p + fl[pbin]
            lb, _, fbar = blend_log_density(ua, ub, ca, cb, s, frac)
            lp = lb - log_nu_r[pj]
            pi_eff = _as_tensor(pi_c, dtype=_F64, device=dev) * (1.0 - gate_pvalue)
            pi_i = pi_eff * inside
            # log(pi) + outside_log, not log(pi * inside): finite gradients.
            log_odds = torch.log(pi_eff) + outside_log - torch.log1p(-pi_i)
            # eta = sigmoid(z) (1 - gate_pvalue); in logs, finite as eta -> 0.
            log_eta = torch.nn.functional.logsigmoid(extra[0]) + math.log1p(
                -gate_pvalue
            )
            log_odds_eta = log_eta - torch.log(-torch.expm1(log_eta))
            lw = torch.cat((log_odds[li] + ls, log_odds_eta + lp))
            log1m = torch.log1p(-pi_i)
            lw_prior = log_odds_eta + log1m[da] + log1m[db]
            return lw, log1m.sum(), lw_prior, ls, lp, fbar

        def objective(pi_c, sc, fl, lnu, extra):
            # log Z_data - log Z_prior: the sources (single left rows, pairs)
            # are conditioned on each left row contributing at most once.
            lw, log1m, lw_prior, _, _, _ = weights(pi_c, sc, fl, lnu, extra)
            x = graph.messages(lw.detach(), state.get("x"), strict=False)
            state["x"] = x.detach()
            ll = log1m + graph.log_partition(x, lw)
            if prior_graph is not None:
                y = prior_graph.messages(
                    lw_prior.detach(), state.get("y"), strict=False
                )
                state["y"] = y.detach()
                ll = ll - prior_graph.log_partition(y, lw_prior)
            return ll

        def posterior(pi_c, sc, fl, lnu):
            lw, _, _, ls, lp, fbar = weights(pi_c, sc, fl, lnu, state["extra"])
            x = graph.messages(lw, state.get("x"))
            state["x"] = x.detach()
            return graph.marginals(x, lw), ls, lp, fbar

        fit_cal = calibrate != "none"
        own = None
        if fit_cal and n_bins > 1:
            (p_h0, _, _, _), _, _, _ = posterior(pi_c, scale, floor, log_nu_cell)
            mass = torch.zeros(n_bins, dtype=_F64, device=dev).index_add(
                0, rbin[torch.cat((rj, pj))], p_h0.detach()
            )
            own = mass >= calibrate_min_pairs
        elif fit_cal:
            own = torch.zeros(1, dtype=torch.bool, device=dev)
        typical = (
            float(torch.median(0.5 * (cl + cr).diagonal(dim1=-2, dim2=-1).sum(-1)))
            if li.numel()
            else 1.0
        )
        fixed = (scale.clone(), floor.clone())

        def fit(pi0, sc0, fl0, nu0, group):
            return A._fit_marginal(
                None,
                li,
                rj,
                n_left,
                True,
                state,
                pi0,
                sc0,
                fl0,
                nu0,
                dens,
                fit_pi=fit_pi,
                calibrate=calibrate,
                group=group,
                typical=typical,
                objective=objective,
                extra0=state["extra"],
            )

        pi_c, scale, floor, log_nu_cell, rejected = A._fit_all(
            fit, pi_c, scale, floor, log_nu_cell, calibrate=calibrate, own=own
        )
        logit_eta = state["extra"]
        prev_state = state
        (p_h, p_bg, p_e, p_free), ls, lp, fbar = posterior(
            pi_c, scale, floor, log_nu_cell
        )
        prev = (edge_i, edge_j, state["x"])
        if not fit_cal:
            break
        moved_by = float((scale - fixed[0]).abs().max()) + float(
            (floor - fixed[1]).abs().max()
        ) / max(1.0, float(floor.abs().max()))
        if moved_by < 1e-4 or rejected:
            break

    # Per edge: single-hypothesis metric at the final calibration.
    delta_e, tl_e, tr_e, psi_e = A.pair_frame(lx[edge_i], rx[edge_j])
    be = rbin[edge_j]
    sig_e = (
        tl_e @ lcov[edge_i] @ tl_e.mT
        + (scale[be] ** 2)[:, None, None] * (tr_e @ rcov[edge_j] @ tr_e.mT)
        + floor[be]
    )
    d2_e, logdet_e = A.mahalanobis2(delta_e, sig_e)
    log_lr_e = (
        -_LOG_2PI - 0.5 * logdet_e - 0.5 * d2_e - dens.log_right(log_nu_cell)[edge_j]
    )
    n_e = int(edge_i.numel())
    p_single = torch.zeros(n_e, dtype=_F64, device=dev).index_copy(0, e_s, p_h[:n_s])
    p_none = torch.ones(n_left, dtype=_F64, device=dev).index_copy(0, pid, p_free)
    p_background = torch.ones(n_right, dtype=_F64, device=dev).index_copy(0, oid, p_bg)
    # Each right row's most probable hypothesis.
    nh = int(p_h.numel())
    best = torch.zeros(int(oid.numel()), dtype=_F64, device=dev).scatter_reduce(
        0, hyp_owner, p_h.detach(), "amax", include_self=False
    )
    hit = (p_h.detach() == best[hyp_owner]) & (p_h.detach() > p_bg.detach()[hyp_owner])
    from ..device import _scatter_reduce_int

    arg = _scatter_reduce_int(
        torch.full((int(oid.numel()),), nh, dtype=torch.int64, device=dev),
        0,
        hyp_owner,
        torch.where(hit, torch.arange(nh, device=dev), nh),
        "amin",
    )
    found = arg < nh
    argc = arg.clamp(max=max(nh - 1, 0))
    map_members = torch.full((n_right, 2), -1, dtype=torch.int64, device=dev)
    map_posterior = p_background.detach().clone()
    if nh:
        mm = torch.stack((m1[argc], m2[argc]), -1)
        map_members[oid] = torch.where(found[:, None], mm, -1)
        map_posterior[oid] = torch.where(found, p_h.detach()[argc], p_bg.detach())
    match = torch.zeros(n_e, dtype=torch.bool, device=dev)
    if assign != "none" and nh:
        sel = greedy_hypotheses(hyp_owner, m1, m2, p_h.detach(), p_bg.detach(), n_left)
        match[e_s[sel[:n_s]]] = True
        match[e_a[sel[n_s:]]] = True
        match[e_b[sel[n_s:]]] = True
    pi = (pi_c * (1.0 - gate_pvalue)) * inside
    blend = BlendAssociation(
        right_index=pj,
        members=torch.stack((pa, pb), -1),
        posterior=p_h[n_s:],
        flux_fraction=fbar.detach(),
        log_lr=lp.detach(),
        p_single=p_single,
        p_background=p_background,
        map_members=map_members,
        map_posterior=map_posterior,
        blend_rate=float(torch.sigmoid(logit_eta[0])),
        n_pair_sources=n_dim,
    )
    return A.Association(
        left_index=edge_i,
        right_index=edge_j,
        separation_arcsec=psi_e * A._ARCSEC_PER_RAD,
        d2=d2_e.detach(),
        log_lr=log_lr_e,
        posterior=p_e,
        match=match,
        p_none=p_none,
        prior=pi,
        completeness=pi_c,
        n_capped=n_capped,
        right_scale=float(scale[0]) if n_bins == 1 else scale,
        floor_mas2=floor[0] if n_bins == 1 else floor,
        density_mas2=dens.log_left(log_nu_cell).exp(),
        blend=blend,
    )


__all__ = ["BlendAssociation", "blend_log_density", "edge_geometry"]
