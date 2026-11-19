"""Probabilistic catalogue association: Mahalanobis gate, likelihood ratios, posteriors.

Pipeline of :func:`associate` (left = primary, right = candidate catalogue):

1. **Epoch.** Both catalogues move to one epoch with :func:`propagate_astrometry`
   (per-row motion router, ``J C J^T`` covariances, :class:`MotionPrior` for rows
   without proper motion).
2. **Sound candidates.** A pair passes the gate when
   ``d^2 = delta^T (C_l + C_r)^-1 delta <= k^2``, with ``k^2`` the chi^2 (2 dof)
   quantile of ``1 - gate_pvalue``. That implies
   ``sep <= k sqrt(lmax(C_l)) + k sqrt(lmax(C_r))``. Each side searches its own
   radius, capped by ``max_radius_arcsec``, through the exact zone index
   (two-sided ``sum`` rule). Right rows are bucketed by powers of two in radius,
   so one large error does not widen every window. Recall is 1 for every pair
   inside the gate and the ceiling.
3. **Pair metric.** ``delta`` is the offset in the tangent plane at the pair
   midpoint, scaled so that ``|delta|`` is the exact great-circle arc. Both
   covariances are parallel-transported to the midpoint (a rotation about
   ``l x r``), which is exact in eigenvalues and pole-safe.
4. **Likelihood ratio.** ``LR_ij = N(delta; 0, Sigma) / nu(x_i) * prod rho_t``
   (Sutherland & Saunders 1992; equal to the Budavari & Szalay 2008 Bayes factor
   against a background of density ``nu``). ``nu`` is the right catalogue's
   surface density, global or per HEALPix cell, over its footprint (:class:`MOC`).
5. **Posterior.** With prior ``pi_i = completeness * [x_i in right footprint]
   * (1 - gate_pvalue)``: ``P_ij = pi LR_ij / ((1 - pi) + pi sum_k LR_ik)``,
   ``P_i0 = 1 - sum_j P_ij``. ``completeness="em"`` maximises the marginal
   likelihood. :class:`HistogramRatio` terms ``rho = q(m) / n(m)`` use explicit
   bin edges, with ``q`` learnt by EM responsibilities.
6. **Assignment.** ``assign="none"`` keeps all posteriors. ``"greedy"`` and
   ``"optimal"`` give one-to-one matches maximising ``sum log(P_ij / P_i0)``.
   ``"optimal"`` is exact: star-shaped components in one vectorised step, the
   rest by the Hungarian algorithm per connected component.

Every step is a segment operation over the ``K`` candidate pairs, ``O(K)``
after candidate generation. Posteriors are differentiable in positions and
covariances for a fixed candidate set.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor

from ..sphere.moc import MOC
from ..sphere import healpix as _healpix
from ._zones import (
    ZoneIndex,
    default_zone_height_deg,
    occupied_spacing_deg,
    unit_vectors,
    zone_query,
)
from .astrometry import MotionPrior, propagate_astrometry
from .catalog import Catalog

_F64 = torch.float64
_MAS_PER_RAD = 180.0 * 3600.0 * 1000.0 / math.pi
_ARCSEC_PER_RAD = 180.0 * 3600.0 / math.pi


# ------------------------------------------------------------------ geometry


def _basis(xyz: Tensor) -> tuple[Tensor, Tensor]:
    """Local (east, north) unit vectors at unit vectors ``xyz`` ([..., 3])."""
    lon = torch.atan2(xyz[..., 1], xyz[..., 0])
    lat = torch.asin(xyz[..., 2].clamp(-1.0, 1.0))
    east = torch.stack((-torch.sin(lon), torch.cos(lon), torch.zeros_like(lon)), -1)
    north = torch.stack(
        (
            -torch.sin(lat) * torch.cos(lon),
            -torch.sin(lat) * torch.sin(lon),
            torch.cos(lat),
        ),
        -1,
    )
    return east, north


def _rotate(
    v: Tensor, axis: Tensor, s: Tensor, c: Tensor, one_minus_c: Tensor
) -> Tensor:
    """Rodrigues rotation of ``v`` by the rotation with ``k sin t = s * axis`` etc.

    ``axis`` is unnormalised; ``s`` multiplies ``axis x v`` and ``one_minus_c``
    multiplies ``axis (axis . v)``, so the near-zero angle limit is stable.
    """
    return (
        v * c[..., None]
        + torch.linalg.cross(axis, v, dim=-1) * s[..., None]
        + axis * (axis * v).sum(-1, keepdim=True) * one_minus_c[..., None]
    )


def pair_frame(
    left_xyz: Tensor, right_xyz: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Midpoint offset and transports for unit-vector pairs.

    Returns ``(delta_mas [K, 2], T_left [K, 2, 2], T_right [K, 2, 2], sep_rad [K])``.
    ``delta`` is ``right - left`` in the midpoint (east, north) basis, scaled so
    ``|delta|`` equals the great-circle arc. ``T`` maps a covariance in a row's
    own local basis to the midpoint basis: ``C_mid = T C T^T``, by parallel
    transport along the great circle (exact rotation, so eigenvalues are kept).
    """
    a = torch.linalg.cross(left_xyz, right_xyz, dim=-1)  # |a| = sin(psi)
    dot = (left_xyz * right_xyz).sum(-1).clamp(-1.0, 1.0)
    psi = torch.atan2(a.norm(dim=-1), dot)
    mid = left_xyz + right_xyz
    mid = mid / mid.norm(dim=-1, keepdim=True).clamp_min(1e-300)
    em, nm = _basis(mid)
    chord = right_xyz - left_xyz  # perpendicular to mid, |chord| = 2 sin(psi/2)
    scale = psi / (2.0 * torch.sin(0.5 * psi)).clamp_min(1e-300)
    scale = torch.where(psi > 0, scale, torch.ones_like(scale))
    delta = (
        torch.stack(((chord * em).sum(-1), (chord * nm).sum(-1)), -1)
        * (scale * _MAS_PER_RAD)[..., None]
    )
    # Rotation by psi/2 about a: with k = a / sin(psi), k sin(psi/2) = a / (2 cos(psi/2))
    # and (1 - cos(psi/2)) k k^T = a a^T / (4 cos^2(psi/2) (1 + cos(psi/2))).
    ch = torch.cos(0.5 * psi)
    s = 1.0 / (2.0 * ch)
    omc = 1.0 / (4.0 * ch * ch * (1.0 + ch))

    def transport(xyz: Tensor, sign: float) -> Tensor:
        e, n = _basis(xyz)
        re = _rotate(e, a, sign * s, ch, omc)
        rn = _rotate(n, a, sign * s, ch, omc)
        # Columns: images of the row's east/north, expressed in the midpoint basis.
        return torch.stack(
            (
                torch.stack(((re * em).sum(-1), (rn * em).sum(-1)), -1),
                torch.stack(((re * nm).sum(-1), (rn * nm).sum(-1)), -1),
            ),
            -2,
        )

    return delta, transport(left_xyz, 1.0), transport(right_xyz, -1.0), psi


def mahalanobis2(delta: Tensor, cov: Tensor) -> tuple[Tensor, Tensor]:
    """``(d^2, log det cov)`` for ``[K, 2]`` offsets and ``[K, 2, 2]`` covariances."""
    a, b, d = cov[..., 0, 0], cov[..., 0, 1], cov[..., 1, 1]
    det = a * d - b * b
    x, y = delta[..., 0], delta[..., 1]
    return (d * x * x - 2.0 * b * x * y + a * y * y) / det, torch.log(det)


def gate_k2(pvalue: float) -> float:
    """chi^2 (2 dof) quantile: ``P(d^2 > k^2) = pvalue`` for true pairs."""
    if not 0.0 < pvalue < 1.0:
        raise ValueError("gate_pvalue must lie in (0, 1)")
    return -2.0 * math.log(pvalue)


# ------------------------------------------------------------ likelihood terms


@dataclass
class HistogramRatio:
    """Photometric likelihood-ratio term ``rho(m) = q(m) / n(m)`` on fixed bins.

    ``right_values`` is ``[M]`` or ``[M, D]`` (magnitudes or colours of the right
    catalogue); ``edges`` is one tensor of bin edges per dimension. ``n`` is the
    normalised histogram of all right rows in the footprint; ``q`` (counterparts)
    starts equal to ``n`` and is re-estimated by EM from posterior
    responsibilities. Rows with a NaN value get ``rho = 1`` (marginalised).
    ``pseudocount`` (default one count per bin) regularises empty bins; it is the
    only smoothing and is explicit.
    """

    right_values: Tensor
    edges: list[Tensor]
    pseudocount: float = 1.0
    _bins: Tensor | None = field(default=None, init=False, repr=False)
    _n: Tensor | None = field(default=None, init=False, repr=False)
    _q: Tensor | None = field(default=None, init=False, repr=False)

    def setup(self, in_footprint: Tensor) -> None:
        v = torch.as_tensor(self.right_values, dtype=_F64)
        v = v[:, None] if v.ndim == 1 else v
        if v.shape[1] != len(self.edges):
            raise ValueError("one edge tensor per value dimension")
        sizes = [int(e.numel()) - 1 for e in self.edges]
        flat = torch.zeros(v.shape[0], dtype=torch.int64)
        ok = torch.isfinite(v).all(-1)
        stride = 1
        for d in range(v.shape[1] - 1, -1, -1):
            e = torch.as_tensor(self.edges[d], dtype=_F64)
            idx = torch.bucketize(v[:, d].contiguous(), e, right=True) - 1
            ok &= (idx >= 0) & (idx < sizes[d])
            flat = flat + idx.clamp(0, sizes[d] - 1) * stride
            stride *= sizes[d]
        self._bins = torch.where(ok, flat, torch.full_like(flat, -1))
        use = ok & in_footprint.cpu()
        n = torch.bincount(flat[use], minlength=stride).to(_F64) + self.pseudocount
        self._n = n / n.sum()
        self._q = self._n.clone()

    def log_rho(self, right_rows: Tensor) -> Tensor:
        assert self._bins is not None and self._q is not None and self._n is not None
        b = self._bins.to(right_rows.device)[right_rows]
        lr = torch.log(self._q.to(right_rows.device)) - torch.log(
            self._n.to(right_rows.device)
        )
        return torch.where(b >= 0, lr[b.clamp(min=0)], torch.zeros_like(b, dtype=_F64))

    def update(self, right_rows: Tensor, weight: Tensor) -> None:
        assert self._bins is not None and self._n is not None
        b = self._bins.to(right_rows.device)[right_rows]
        ok = b >= 0
        q = torch.zeros_like(self._n).index_add_(
            0, b[ok].cpu(), weight.detach()[ok].cpu()
        )
        q = q + self.pseudocount
        self._q = q / q.sum()


# ------------------------------------------------------------------ results


@dataclass
class Association:
    """Pairwise association of a primary (left) catalogue with a right catalogue.

    Pair arrays (``K`` gated candidates, ordered by left then right index):
    ``left_index``, ``right_index``, ``separation_arcsec``, ``d2`` (Mahalanobis),
    ``log_lr`` (natural log likelihood ratio), ``posterior`` (P that the pair is the
    counterpart), and ``match`` (bool, the selected one-to-one or best pairs; all
    False when ``assign="none"``). Per left row: ``p_none`` (no counterpart)
    and ``prior`` (``pi_i``). ``completeness`` is the fitted or given value;
    ``n_capped`` counts rows whose search radius hit ``max_radius_arcsec``.
    """

    left_index: Tensor
    right_index: Tensor
    separation_arcsec: Tensor
    d2: Tensor
    log_lr: Tensor
    posterior: Tensor
    match: Tensor
    p_none: Tensor
    prior: Tensor
    completeness: float
    n_capped: int


# ----------------------------------------------------------------- pipeline


def _epoch_positions(
    cat: Catalog, epoch: Tensor | None, prior: MotionPrior | None
) -> tuple[Tensor, Tensor, Tensor]:
    if cat.astrometry is None:
        raise ValueError("associate needs Catalog.astrometry on both catalogues")
    if epoch is None:
        return cat.lon.to(_F64), cat.lat.to(_F64), cat.astrometry.position_cov()
    moved, cov = propagate_astrometry(
        cat.to_state(), cat.astrometry, epoch, motion_prior=prior
    )
    return moved.lon, moved.lat, cov


def _lambda_max(cov: Tensor) -> Tensor:
    a, b, d = cov[..., 0, 0], cov[..., 0, 1], cov[..., 1, 1]
    return 0.5 * (a + d) + torch.sqrt((0.5 * (a - d)) ** 2 + b * b)


def _sound_candidates(
    llon: Tensor, llat: Tensor, rlon: Tensor, rlat: Tensor, rl: Tensor, rr: Tensor
) -> tuple[Tensor, Tensor]:
    """All pairs with ``sep <= rl[i] + rr[j]``, bucketing right rows by radius class."""
    cls = torch.floor(torch.log2(rr.clamp_min(1e-12))).to(torch.int64)
    out_l, out_r = [], []
    spacing = occupied_spacing_deg(rlon, rlat)
    for c in torch.unique(cls).tolist():
        rows = torch.nonzero(cls == c, as_tuple=True)[0]
        rr_c = rr[rows]
        window = float(rl.median()) + float(rr_c.max())
        idx = ZoneIndex(
            rlon[rows],
            rlat[rows],
            zone_height_deg=default_zone_height_deg(int(rows.numel()), window, spacing),
        )
        li, rj, _ = zone_query(
            idx, llon, llat, rl, radius_right=rr_c, combine="sum", find="all"
        )
        out_l.append(li)
        out_r.append(rows[rj])
    if not out_l:
        e = torch.empty(0, dtype=torch.int64, device=llon.device)
        return e, e
    li, rj = torch.cat(out_l), torch.cat(out_r)
    order = torch.argsort(rj, stable=True)
    order = order[torch.argsort(li[order], stable=True)]
    return li[order], rj[order]


def _density(
    rlon: Tensor,
    rlat: Tensor,
    llon: Tensor,
    llat: Tensor,
    footprint: MOC,
    mode: str,
    order: int | None,
) -> Tensor:
    """Right-catalogue surface density at each left row, per mas^2."""
    mas2_per_sr = _MAS_PER_RAD**2
    inside = footprint.contains(rlon, rlat)
    if mode == "global":
        area = footprint.area
        if area <= 0.0:
            raise ValueError("right footprint has zero area")
        nu = float(inside.sum()) / (area * mas2_per_sr)
        return torch.full_like(llon, nu, dtype=_F64)
    if mode != "healpix" or order is None:
        raise ValueError("density must be 'global' or 'healpix' with density_order")
    nside = 2**order
    pix_r = _healpix.ang2pix(
        nside, rlon[inside], rlat[inside], nest=True, lonlat=True
    ).to(torch.int64)
    counts = torch.bincount(pix_r, minlength=12 * nside * nside).to(_F64)
    pix_l = _healpix.ang2pix(nside, llon, llat, nest=True, lonlat=True).to(torch.int64)
    # Footprint-clipped cell area: fraction of each cell inside the right footprint.
    from ..sphere.moc import nest2uniq

    frac = footprint.overlap_fraction(nest2uniq(nside, pix_l)).to(_F64)
    area = frac * (4.0 * math.pi / (12.0 * nside * nside)) * mas2_per_sr
    return torch.where(
        area > 0, counts[pix_l] / area.clamp_min(1e-300), torch.zeros_like(area)
    )


def associate(
    left: Catalog,
    right: Catalog,
    *,
    gate_pvalue: float = 1e-4,
    max_radius_arcsec: float = 10.0,
    epoch: float | Tensor | Literal["right", "left"] | None = "right",
    motion_prior: MotionPrior
    | tuple[MotionPrior | None, MotionPrior | None]
    | None = None,
    right_footprint: MOC | None = None,
    footprint_order: int = 10,
    density: Literal["global", "healpix"] = "global",
    density_order: int | None = None,
    completeness: float | Literal["em"] = "em",
    terms: tuple[HistogramRatio, ...] = (),
    assign: Literal["none", "best", "greedy", "optimal"] = "optimal",
    max_iter: int = 50,
    tol: float = 1e-6,
) -> Association:
    """Probabilistic counterparts of ``left`` rows in ``right`` (see module docs).

    Both catalogues need ``astrometry`` (position covariance at least).
    ``epoch="right"`` propagates both catalogues to the right catalogue's scalar epoch. A
    number propagates both to it; ``None`` uses the stored positions as they are.
    ``motion_prior`` serves rows without proper motion (one prior, or one
    per side). ``right_footprint`` (MOC) defaults to the right catalogue's occupied
    cells at ``footprint_order``. Pass the survey MOC when you have it, so that density and
    priors use the true coverage.
    """
    if left.frame != right.frame:
        right = right.to_frame(left.frame)
    priors = (
        motion_prior
        if isinstance(motion_prior, tuple)
        else (motion_prior, motion_prior)
    )
    if epoch == "right" or epoch == "left":
        ref = right if epoch == "right" else left
        ep = torch.as_tensor(ref.epoch)
        if ep.numel() != 1:
            raise ValueError(
                f"the {epoch} catalogue has per-row epochs; pass a scalar epoch"
            )
        target: Tensor | None = ep.reshape(())
    else:
        target = None if epoch is None else torch.as_tensor(epoch, dtype=_F64)
    llon, llat, lcov = _epoch_positions(left, target, priors[0])
    rlon, rlat, rcov = _epoch_positions(right, target, priors[1])
    dev = llon.device
    k2 = gate_k2(gate_pvalue)
    k = math.sqrt(k2)
    rl_raw = k * torch.sqrt(_lambda_max(lcov)) / 1000.0
    rr_raw = k * torch.sqrt(_lambda_max(rcov)) / 1000.0
    rl = rl_raw.clamp(max=max_radius_arcsec)
    rr = rr_raw.clamp(max=max_radius_arcsec)
    n_capped = int(
        (rl_raw > max_radius_arcsec).sum() + (rr_raw > max_radius_arcsec).sum()
    )
    li, rj = _sound_candidates(llon, llat, rlon, rlat, rl, rr)

    rfoot = (
        right_footprint
        if right_footprint is not None
        else MOC.from_lonlat(rlon, rlat, footprint_order)
    )
    lx, rx = unit_vectors(llon, llat), unit_vectors(rlon, rlat)
    delta, tl, tr, psi = pair_frame(lx[li], rx[rj])
    sigma = tl @ lcov[li] @ tl.mT + tr @ rcov[rj] @ tr.mT
    d2, logdet = mahalanobis2(delta, sigma)
    sep_arcsec = psi * _ARCSEC_PER_RAD
    keep = (d2 <= k2) & (sep_arcsec <= max_radius_arcsec)
    li, rj, d2, logdet, sep_arcsec = (
        li[keep],
        rj[keep],
        d2[keep],
        logdet[keep],
        sep_arcsec[keep],
    )
    nu = _density(rlon, rlat, llon, llat, rfoot, density, density_order)
    log_gauss = -math.log(2.0 * math.pi) - 0.5 * logdet - 0.5 * d2
    base_log_lr = log_gauss - torch.log(nu[li].clamp_min(1e-300))
    inside = rfoot.contains(llon, llat).to(_F64)
    n_left = int(llon.numel())
    for term in terms:
        term.setup(rfoot.contains(rlon, rlat))

    def posterior(pi_c: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        log_lr = base_log_lr.clone()
        for term in terms:
            log_lr = log_lr + term.log_rho(rj)
        pi = (pi_c * (1.0 - gate_pvalue)) * inside
        # P_ij = pi LR / ((1 - pi) + pi sum LR), in log space per left segment.
        lse_max = torch.full(
            (n_left,), -math.inf, dtype=_F64, device=dev
        ).scatter_reduce(0, li, log_lr, "amax", include_self=True)
        shift = torch.where(torch.isfinite(lse_max), lse_max, torch.zeros_like(lse_max))
        s = torch.zeros(n_left, dtype=_F64, device=dev).index_add(
            0, li, torch.exp(log_lr - shift[li])
        )
        sum_lr = s * torch.exp(shift)
        denom = (1.0 - pi) + pi * sum_lr
        post = pi[li] * torch.exp(log_lr) / denom[li]
        p_none = (1.0 - pi) / denom
        return post, p_none, pi, log_lr

    if completeness == "em":
        pi_c = 0.5
        for _ in range(max_iter):
            post, p_none, _, _ = posterior(pi_c)
            for term in terms:
                term.update(rj, post)
            in_rows = inside > 0
            new = (
                float((1.0 - p_none[in_rows]).mean()) / (1.0 - gate_pvalue)
                if bool(in_rows.any())
                else 0.0
            )
            new = min(max(new, 1e-6), 1.0 - 1e-9)
            if abs(new - pi_c) < tol:
                pi_c = new
                break
            pi_c = new
    else:
        pi_c = float(completeness)
        if not 0.0 <= pi_c <= 1.0:
            raise ValueError("completeness must lie in [0, 1] or be 'em'")
    post, p_none, pi, log_lr = posterior(pi_c)
    match = _assign(li, rj, post, p_none, n_left, int(rlon.numel()), assign)
    return Association(
        left_index=li,
        right_index=rj,
        separation_arcsec=sep_arcsec,
        d2=d2,
        log_lr=log_lr,
        posterior=post,
        match=match,
        p_none=p_none,
        prior=pi,
        completeness=pi_c,
        n_capped=n_capped,
    )


# ---------------------------------------------------------------- assignment


def _assign(
    li: Tensor,
    rj: Tensor,
    post: Tensor,
    p_none: Tensor,
    n_left: int,
    n_right: int,
    how: Literal["none", "best", "greedy", "optimal"],
) -> Tensor:
    k = int(li.numel())
    match = torch.zeros(k, dtype=torch.bool, device=li.device)
    if how == "none" or k == 0:
        return match
    w = torch.log(post.detach().clamp_min(1e-300)) - torch.log(
        p_none.detach()[li].clamp_min(1e-300)
    )
    if how == "best":
        best = torch.full(
            (n_left,), -math.inf, dtype=_F64, device=li.device
        ).scatter_reduce(0, li, w, "amax")
        return (w == best[li]) & (w > 0)
    weight = torch.where(w > 0, w, torch.full_like(w, -math.inf))
    method: Literal["greedy", "optimal"] = "greedy" if how == "greedy" else "optimal"
    return assign_one_to_one(
        li, rj, weight, n_left=n_left, n_right=n_right, method=method
    )


def assign_one_to_one(
    left: Tensor,
    right: Tensor,
    weight: Tensor,
    *,
    n_left: int | None = None,
    n_right: int | None = None,
    method: Literal["greedy", "optimal"] = "optimal",
) -> Tensor:
    """Select pairs so each row is used at most once, maximising total weight.

    Pairs with weight ``<= 0`` (or ``-inf``) are never selected. Leaving a row
    unmatched scores 0. ``"greedy"`` takes pairs by descending weight.
    ``"optimal"`` is exact (maximum-weight bipartite matching). Connected
    components with one left or one right row are solved in one vectorised step
    (the best positive edge); larger components run the Hungarian algorithm.
    Returns a boolean mask over the pairs.
    """
    from ._pairs import connected_components, greedy_one_to_one

    k = int(left.numel())
    dev = left.device
    sel = torch.zeros(k, dtype=torch.bool, device=dev)
    pos = torch.nonzero(weight > 0, as_tuple=True)[0]
    if pos.numel() == 0:
        return sel
    li, rj, w = left[pos], right[pos], weight[pos]
    n_left = int(li.max()) + 1 if n_left is None else n_left
    n_right = int(rj.max()) + 1 if n_right is None else n_right
    if method == "greedy":
        keep = greedy_one_to_one(li, rj, -w)
        sel[pos[keep]] = True
        return sel
    if method != "optimal":
        raise ValueError("method must be 'greedy' or 'optimal'")
    root = connected_components(n_left + n_right, li, rj + n_left)
    comp = root[li]
    _, comp_id = torch.unique(comp, return_inverse=True)
    n_comp = int(comp_id.max()) + 1
    # Distinct left and right rows per component.
    lkey = torch.unique(torch.stack((comp_id, li)), dim=1)
    rkey = torch.unique(torch.stack((comp_id, rj)), dim=1)
    lefts = torch.bincount(lkey[0], minlength=n_comp)
    rights = torch.bincount(rkey[0], minlength=n_comp)
    star = (lefts == 1) | (rights == 1)
    in_star = star[comp_id]
    if bool(in_star.any()):
        best = torch.full(
            (n_comp,), -math.inf, dtype=w.dtype, device=dev
        ).scatter_reduce(0, comp_id[in_star], w[in_star], "amax")
        cand = in_star & (w == best[comp_id])
        # Exact ties: keep the lowest pair position per component.
        idx = torch.arange(int(w.numel()), device=dev)
        first = torch.full(
            (n_comp,), int(w.numel()), dtype=torch.int64, device=dev
        ).scatter_reduce(0, comp_id[cand], idx[cand], "amin")
        chosen = first[(first < int(w.numel()))]
        sel[pos[chosen]] = True
    hard = torch.nonzero(~star, as_tuple=True)[0]
    for c in hard.tolist():
        m = torch.nonzero(comp_id == c, as_tuple=True)[0]
        chosen = _hungarian(li[m], rj[m], w[m])
        sel[pos[m[chosen]]] = True
    return sel


def _hungarian(li: Tensor, rj: Tensor, w: Tensor) -> Tensor:
    """Exact maximum-weight matching on one small dense component (rows unmatched allowed)."""
    lu, lidx = torch.unique(li, return_inverse=True)
    ru, ridx = torch.unique(rj, return_inverse=True)
    nl, nr = int(lu.numel()), int(ru.numel())
    # Square cost with dummy columns/rows so any row may stay unmatched (score 0).
    size = nl + nr
    cost = torch.zeros(size, size, dtype=_F64)
    big = float(w.abs().max()) * 4.0 + 1.0
    cost[:nl, :nr] = big  # absent edges: worse than staying unmatched
    cost[lidx.cpu(), ridx.cpu()] = -w.detach().cpu().to(_F64)
    assignment = _lsa(cost)
    chosen = torch.zeros(int(w.numel()), dtype=torch.bool)
    pair_of = {
        (int(a), int(b)): i
        for i, (a, b) in enumerate(zip(lidx.tolist(), ridx.tolist()))
    }
    for r, c in enumerate(assignment):
        if r < nl and c < nr and (r, c) in pair_of:
            chosen[pair_of[(r, c)]] = True
    return torch.nonzero(chosen.to(li.device), as_tuple=True)[0]


def _lsa(cost: Tensor) -> list[int]:
    """Minimum-cost perfect assignment on a square matrix (Jonker-Volgenant style, O(n^3))."""
    n = int(cost.shape[0])
    c = cost.tolist()
    inf = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [inf] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], inf, 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = c[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    out = [0] * n
    for j in range(1, n + 1):
        if p[j]:
            out[p[j] - 1] = j - 1
    return out


__all__ = [
    "Association",
    "HistogramRatio",
    "assign_one_to_one",
    "associate",
    "gate_k2",
    "mahalanobis2",
    "pair_frame",
]
