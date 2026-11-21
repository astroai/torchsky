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
   * (1 - gate_pvalue)``. Independent model (``exclusive=False``, nway):
   ``P_ij = pi LR_ij / ((1 - pi) + pi sum_k LR_ik)``, ``P_i0 = 1 - sum_j P_ij``.
   Exclusive model (default): a right row is the counterpart of at most one
   left row, and ``LR_ij`` is discounted by ``rho_ij``, the probability that no
   competing left row takes ``j``, from belief propagation (Bethe marginals,
   :mod:`._exclusive`); ``rho = 1`` where ``j`` is not shared.
   ``completeness="em"``, ``calibrate`` and (exclusive) the background density
   maximise the marginal likelihood directly (L-BFGS on the exact or Bethe
   log partition function). :class:`HistogramRatio` terms ``rho = q(m) / n(m)``
   use explicit bin edges, with ``q`` learnt by EM responsibilities.
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

from ..device import _as_tensor, _scatter_reduce_int
from ..sphere.moc import MOC
from ..sphere import healpix as _healpix
from ._zones import (
    ZoneIndex,
    default_zone_height_deg,
    occupied_spacing_deg,
    unit_vectors,
    zone_query,
)
from ._blends import BlendAssociation
from ._exclusive import exclusive_log_rho, pairwise_exclusive
from .astrometry import Astrometry, MotionPrior, propagate_astrometry
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
    """Photometric likelihood-ratio term ``rho = q / n`` on fixed bins.

    Two forms:

    - **Right magnitudes** (``left_values=None``): the feature is
      ``right_values[j]`` (``[M]`` or ``[M, D]``). ``n`` is the normalised
      histogram of all right rows in the footprint (Sutherland & Saunders).
      ``q`` (counterparts) starts equal to ``n`` and is re-estimated by EM from
      the posteriors.
    - **Pair colours** (``left_values`` given): the feature is
      ``right_values[j] - left_values[i]`` (for example W1 - G). It separates the
      true counterpart of a blend from its neighbours. Both ``q`` and ``n`` are
      learnt by EM from the candidate pairs: ``q`` weighted by the posteriors,
      ``n`` by ``1 - posterior``.

    ``edges`` holds one tensor of bin edges per dimension. Rows with a NaN or
    out-of-range feature get ``rho = 1`` (marginalised). ``pseudocount`` (one
    count per bin by default) regularises empty bins; it is the only smoothing
    and is explicit.
    """

    right_values: Tensor
    edges: list[Tensor]
    left_values: Tensor | None = None
    pseudocount: float = 1.0
    _bins: Tensor | None = field(default=None, init=False, repr=False)
    _n: Tensor | None = field(default=None, init=False, repr=False)
    _q: Tensor | None = field(default=None, init=False, repr=False)
    _size: int = field(default=0, init=False, repr=False)

    def _bin(self, v: Tensor) -> Tensor:
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
        self._size = stride
        return torch.where(ok, flat, torch.full_like(flat, -1))

    def _pair_bins(self, li: Tensor, rj: Tensor) -> Tensor:
        if self.left_values is None:
            assert self._bins is not None
            return self._bins.to(rj.device)[rj]
        r = torch.as_tensor(self.right_values, dtype=_F64)
        left = torch.as_tensor(self.left_values, dtype=_F64)
        return self._bin(r[rj.cpu()] - left[li.cpu()]).to(rj.device)

    def setup(self, in_footprint: Tensor, li: Tensor, rj: Tensor) -> None:
        if self.left_values is None:
            self._bins = self._bin(torch.as_tensor(self.right_values, dtype=_F64))
            use = (self._bins >= 0) & in_footprint.cpu()
            n = (
                torch.bincount(self._bins[use], minlength=self._size).to(_F64)
                + self.pseudocount
            )
        else:
            b = self._pair_bins(li, rj).cpu()
            n = (
                torch.bincount(b[b >= 0], minlength=self._size).to(_F64)
                + self.pseudocount
            )
        self._n = n / n.sum()
        self._q = self._n.clone()

    def log_rho(self, li: Tensor, rj: Tensor) -> Tensor:
        assert self._q is not None and self._n is not None
        b = self._pair_bins(li, rj)
        lr = torch.log(self._q.to(rj.device)) - torch.log(self._n.to(rj.device))
        return torch.where(b >= 0, lr[b.clamp(min=0)], torch.zeros_like(b, dtype=_F64))

    def update(self, li: Tensor, rj: Tensor, weight: Tensor) -> None:
        assert self._n is not None
        b = self._pair_bins(li, rj).cpu()
        ok = b >= 0
        w = weight.detach().cpu()
        q = torch.zeros_like(self._n).index_add_(0, b[ok], w[ok]) + self.pseudocount
        self._q = q / q.sum()
        if self.left_values is not None:
            n = (
                torch.zeros_like(self._n).index_add_(0, b[ok], (1.0 - w)[ok])
                + self.pseudocount
            )
            self._n = n / n.sum()


# ------------------------------------------------------------------ results


@dataclass
class Association:
    """Pairwise association of a primary (left) catalogue with a right catalogue.

    Pair arrays (``K`` gated candidates, ordered by left then right index):
    ``left_index``, ``right_index``, ``separation_arcsec``, ``d2`` (Mahalanobis),
    ``log_lr`` (natural log likelihood ratio), ``posterior`` (P that the pair is the
    counterpart), and ``match`` (bool, the selected one-to-one or best pairs; all
    False when ``assign="none"``). Per left row: ``p_none`` (no counterpart),
    ``prior`` (``pi_i``) and ``density_mas2`` (the right background density
    in the left row's cell, per mas^2; with ``exclusive=True`` and
    ``density="healpix"`` the likelihood ratio uses the right row's cell,
    usually the same). ``completeness`` is the fitted or
    given value; ``n_capped`` counts rows whose search radius hit
    ``max_radius_arcsec``. ``blend`` holds the blend posteriors when
    ``associate(blends="pairs")`` (else ``None``).
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
    right_scale: float | Tensor = 1.0
    floor_mas2: Tensor | None = None
    density_mas2: Tensor | None = None
    blend: BlendAssociation | None = None


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


class _Density:
    """Right-catalogue background density, piecewise constant on cells.

    ``mode="global"``: one cell, the footprint. ``"healpix"``: HEALPix cells
    at ``order``, clipped to the footprint. Cells holding right rows carry a
    free density ``nu_c`` (per mas^2); empty cells have ``nu = 0``.
    :meth:`nu` gives the density at each left row; :meth:`log_lik` is the
    inhomogeneous Poisson log likelihood of the background
    ``sum_j (1 - m_j) log nu(x_j) - sum_c nu_c A_c`` (``m_j`` counterpart mass)
    plus a Gamma(3/2) prior on each cell's background count. The maximiser is
    ``(right rows - expected counterparts + 1/2) / area`` per cell: the half
    count keeps the density positive when every right row is a counterpart,
    where the likelihood alone sends it to zero and the likelihood ratios to
    infinity.
    """

    def __init__(self, rlon, rlat, llon, llat, footprint: MOC, mode: str, order):
        mas2_per_sr = _MAS_PER_RAD**2
        dev = llon.device
        inside = footprint.contains(rlon, rlat)
        self.w_in = inside.to(_F64)
        if mode == "global":
            area = footprint.area
            if area <= 0.0:
                raise ValueError("right footprint has zero area")
            self.area = torch.tensor([area * mas2_per_sr], dtype=_F64, device=dev)
            self.left_cell = torch.zeros(llon.shape, dtype=torch.int64, device=dev)
            self.right_cell = torch.zeros(rlon.shape, dtype=torch.int64, device=dev)
        elif mode == "healpix" and order is not None:
            from ..sphere.moc import nest2uniq

            nside = 2**order
            pix_r = _healpix.ang2pix(nside, rlon, rlat, nest=True, lonlat=True).to(
                torch.int64
            )
            pix_l = _healpix.ang2pix(nside, llon, llat, nest=True, lonlat=True).to(
                torch.int64
            )
            cells, inv = torch.unique(torch.cat((pix_l, pix_r)), return_inverse=True)
            frac = footprint.overlap_fraction(nest2uniq(nside, cells)).to(_F64)
            self.area = frac * (4.0 * math.pi / (12.0 * nside * nside)) * mas2_per_sr
            self.left_cell = inv[: llon.numel()]
            self.right_cell = inv[llon.numel() :]
        else:
            raise ValueError("density must be 'global' or 'healpix' with density_order")
        self.n_cells = int(self.area.numel())
        self.count = torch.zeros(self.n_cells, dtype=_F64, device=dev).index_add(
            0, self.right_cell, self.w_in
        )
        # Free cells: some right rows and a positive area.
        self.free = (self.count > 0) & (self.area > 0)

    def cell_nu(self, matched: Tensor | None = None) -> Tensor:
        """Closed-form MAP density per cell given counterpart masses ``matched``
        (``matched=None``: the total density, every right row as background)."""
        w = self.w_in if matched is None else self.w_in * (1.0 - matched)
        c = torch.zeros(self.n_cells, dtype=_F64, device=w.device).index_add(
            0, self.right_cell, w
        )
        if matched is not None:
            c = c + 0.5
        return torch.where(
            self.free, c / self.area.clamp_min(1e-300), torch.zeros_like(c)
        )

    def nu(self, matched: Tensor | None = None) -> Tensor:
        return self.cell_nu(matched)[self.left_cell]

    def log_left(self, log_nu_cell: Tensor) -> Tensor:
        """Per-left log density from per-cell log densities (``-690`` when empty)."""
        lc = torch.where(
            self.free, log_nu_cell, torch.full_like(log_nu_cell, math.log(1e-300))
        )
        return lc[self.left_cell]

    def log_right(self, log_nu_cell: Tensor) -> Tensor:
        """Per-right-row log density (``-690`` in empty cells)."""
        lc = torch.where(
            self.free, log_nu_cell, torch.full_like(log_nu_cell, math.log(1e-300))
        )
        return lc[self.right_cell]

    def log_lik(self, log_nu_cell: Tensor) -> Tensor:
        """Background Poisson log likelihood (all right rows in the footprint;
        the counterpart share is removed by the matching term)."""
        free = self.free
        lc = torch.where(free, log_nu_cell, torch.zeros_like(log_nu_cell))
        return ((self.count + 0.5) * lc).sum() - (
            torch.where(free, torch.exp(lc), torch.zeros_like(lc)) * self.area
        ).sum()


def _pm_speed(state, prior: MotionPrior | None) -> Tensor:
    """|proper motion| (mas/yr) per row; prior mean speed for rows without motion."""
    n = int(state.lon.numel())
    speed = torch.zeros(n, dtype=_F64, device=state.lon.device)
    if state.pm_lon_coslat_mas_yr is not None and state.pm_lat_mas_yr is not None:
        speed = torch.hypot(state.pm_lon_coslat_mas_yr, state.pm_lat_mas_yr).nan_to_num(
            0.0
        )
    if prior is not None:
        mean, _ = prior.at(state.lon, state.lat)
        no_pm = state.motion_level() == 2
        speed = torch.where(no_pm, mean.norm(dim=-1), speed)
    return speed


def _segment_logaddexp(x: Tensor, seg: Tensor, n_seg: int, slack: Tensor) -> Tensor:
    """``log(exp(slack) + sum_{e in seg} exp(x_e))`` per segment."""
    m = torch.full((n_seg,), -math.inf, dtype=_F64, device=x.device).scatter_reduce(
        0, seg, x.detach(), "amax", include_self=True
    )
    m = torch.maximum(m, slack.detach())
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    s_ = torch.exp(slack - m).index_add(0, seg, torch.exp(x - m[seg]))
    return m + torch.log(s_)


def _carry_messages(
    li_old: Tensor, rj_old: Tensor, rho_old: Tensor, li: Tensor, rj: Tensor, m: int
) -> Tensor:
    """Messages of pairs ``(li, rj)`` that were in the old candidate set (else 0)."""
    key_old = li_old * m + rj_old
    order = torch.argsort(key_old)
    key_old, rho_old = key_old[order], rho_old[order]
    key = li * m + rj
    if key_old.numel() == 0:
        return torch.zeros(key.shape, dtype=_F64, device=key.device)
    pos = torch.searchsorted(key_old, key).clamp(max=key_old.numel() - 1)
    hit = key_old[pos] == key
    return torch.where(hit, rho_old[pos], torch.zeros_like(rho_old[pos]))


def _log_partition(
    li: Tensor,
    rj: Tensor,
    log_w: Tensor,
    log_a: Tensor,
    n_left: int,
    exclusive: bool,
    state: dict,
) -> Tensor:
    """Marginal log likelihood ``log Z`` of the association model (up to constants).

    Independent model: ``sum_i log(a_i + sum_j w_ij)`` (exact). Exclusive model:
    the Bethe approximation of the matching partition function from the
    belief-propagation messages,
    ``sum_i log(a_i + sum_j w rho) + sum_j log(1 + sum_i w lambda)
    - sum_ij log(1 + w lambda rho)``. Messages are held fixed: at the BP fixed
    point the Bethe free energy is stationary in them, so the gradient in the
    weights is exact (the posterior-weighted one).
    """
    if not exclusive:
        return _segment_logaddexp(log_w, li, n_left, log_a).sum()
    _, _, rho = pairwise_exclusive(
        li,
        rj,
        log_w.detach(),
        log_a.detach(),
        init=state.get("rho"),
        cache=state.setdefault("graph", {}),  # type: ignore[arg-type]
        # Line-search trial points need not converge; the final posterior
        # solve is strict.
        strict=False,
    )
    state["rho"] = rho
    from ._exclusive import lse_excluding_self

    log_lam = -lse_excluding_self((log_w + rho).detach(), li, n_left, log_a.detach())
    n_right = int(rj.max()) + 1 if rj.numel() else 0
    left_term = _segment_logaddexp(log_w + rho, li, n_left, log_a).sum()
    right_term = _segment_logaddexp(
        log_w + log_lam,
        rj,
        n_right,
        torch.zeros(n_right, dtype=_F64, device=log_w.device),
    ).sum()
    edge_term = torch.nn.functional.softplus(log_w + log_lam + rho).sum()
    return left_term + right_term - edge_term


def _fit_marginal(
    log_weights,
    li: Tensor,
    rj: Tensor,
    n_left: int,
    exclusive: bool,
    state: dict[str, Tensor],
    pi_c: float,
    scale: Tensor,
    floor: Tensor,
    log_nu_cell: Tensor,
    dens: "_Density | None",
    *,
    fit_pi: bool,
    calibrate: str,
    group: Tensor | None,
    typical: float,
    objective=None,
    extra0: Tensor | None = None,
) -> tuple[float, Tensor, Tensor, Tensor, Tensor]:
    """One maximum marginal likelihood fit (see :func:`_fit_all`).

    Parameters: ``logit(completeness)`` (``fit_pi``); per calibration group
    ``log s`` (the right error scale, ``"scale_floor"`` only) and the
    Cholesky factor of the 2x2 floor ``F`` (log diagonal), where
    ``group[b]`` is the group of bin ``b`` (``-1``: held at the given
    ``scale``/``floor``; ``group=None``: no calibration parameters); with
    ``dens`` (exclusive model) the log background density per free cell, with
    the background Poisson term in the likelihood. ``Sigma = C_l + s^2 C_r + F``
    per pair. L-BFGS on the exact (independent) or Bethe (exclusive) log
    likelihood, whose gradient is the posterior-weighted one, so stationary
    points are those of EM; EM itself crawls in crowded fields, where most
    of the information about the errors is missing.

    ``objective(pi, scale, floor, log_nu_cell, extra)`` replaces the
    association log likelihood (the blend model), with free parameters
    ``extra`` starting at ``extra0``; their fitted values are left in
    ``state["extra"]``.

    Returns ``(completeness, scale, floor, log_nu_cell, bad)``. ``bad[g]``
    flags groups outside physical bounds (scale within ``e^+-8``, floor below
    100x the typical pair variance) or non-finite; their bins keep the given
    values. A failed line search returns the inputs with every group bad.
    """
    dev = scale.device
    fit_cal = group is not None and bool((group >= 0).any())
    fit_scale = calibrate == "scale_floor"
    n_groups = int(group.max()) + 1 if fit_cal and group is not None else 0
    eye = torch.eye(2, dtype=_F64, device=dev)
    if fit_cal:
        assert group is not None
        first = [int((group == g).nonzero()[0, 0]) for g in range(n_groups)]
        rep = floor[first]
        rep_scale = scale[first]
        # Start the floor at a tenth of the typical pair variance when it is ~0
        # (a log-parametrised floor at 0 has no gradient).
        small = rep.diagonal(dim1=-2, dim2=-1).amax(-1) <= 1e-3 * typical
        start = torch.where(small[:, None, None], 0.1 * typical * eye, rep)
        chol = torch.linalg.cholesky(start + 1e-9 * typical * eye)
        cal0 = torch.stack(
            (
                torch.log(rep_scale),
                torch.log(chol[:, 0, 0]),
                chol[:, 1, 0],
                torch.log(chol[:, 1, 1]),
            ),
            -1,
        ).reshape(-1)
    else:
        cal0 = torch.empty(0, dtype=_F64, device=dev)
    pi0 = min(max(pi_c, 1e-6), 1.0 - 1e-9)
    eta0 = torch.tensor(
        [math.log(pi0) - math.log1p(-pi0)] if fit_pi else [], dtype=_F64, device=dev
    )
    nu0 = (
        log_nu_cell[dens.free]
        if dens is not None
        else torch.empty(0, dtype=_F64, device=dev)
    )
    ex0 = (
        torch.empty(0, dtype=_F64, device=dev)
        if extra0 is None
        else extra0.detach().to(_F64)
    )
    params = torch.cat((eta0, cal0, nu0, ex0)).clone().requires_grad_(True)
    n_pi = int(eta0.numel())
    n_cal = int(cal0.numel())
    n_nu = int(nu0.numel())
    all_bad = torch.ones(n_groups, dtype=torch.bool, device=dev)

    def unpack_nu(p: Tensor) -> Tensor:
        if dens is None:
            return log_nu_cell
        return log_nu_cell.masked_scatter(
            dens.free, p[n_pi + n_cal : n_pi + n_cal + n_nu]
        )

    def unpack_extra(p: Tensor) -> Tensor:
        return p[n_pi + n_cal + n_nu :]

    def unpack(p: Tensor) -> tuple[Tensor | float, Tensor, Tensor]:
        pi = torch.sigmoid(p[0]) if fit_pi else pi_c
        if not fit_cal:
            return pi, scale, floor
        assert group is not None
        q = p[n_pi : n_pi + n_cal].reshape(n_groups, 4)
        sc = (
            torch.exp(q[:, 0])
            if fit_scale
            else torch.ones(n_groups, dtype=_F64, device=dev)
        )
        zero = torch.zeros(n_groups, dtype=_F64, device=dev)
        lmat = torch.stack(
            (
                torch.stack((torch.exp(q[:, 1]), zero), -1),
                torch.stack((q[:, 2], torch.exp(q[:, 3])), -1),
            ),
            -2,
        )
        on = group >= 0
        g = group.clamp(min=0)
        sc_b = torch.where(on, sc[g], scale)
        fl_b = torch.where(on[:, None, None], (lmat @ lmat.mT)[g], floor)
        return pi, sc_b, fl_b

    opt = torch.optim.LBFGS(
        [params], max_iter=100, line_search_fn="strong_wolfe", tolerance_grad=1e-9
    )

    def closure():
        opt.zero_grad()
        pi, sc, fl = unpack(params)
        lnu = unpack_nu(params)
        if objective is not None:
            ll = objective(pi, sc, fl, lnu, unpack_extra(params))
        else:
            log_w, log_a, _ = log_weights(pi, sc, fl, lnu)
            ll = _log_partition(li, rj, log_w, log_a, n_left, exclusive, state)
        if dens is not None:
            ll = ll + dens.log_lik(lnu)
        loss = -ll / n_left
        loss.backward()
        return loss

    try:
        opt.step(closure)
    except RuntimeError as exc:  # line-search failure: keep the given values
        if "did not converge" in str(exc):
            raise
        return pi_c, scale, floor, log_nu_cell, all_bad
    with torch.no_grad():
        p = params.detach()
        if not bool(torch.isfinite(p).all()):
            return pi_c, scale, floor, log_nu_cell, all_bad
        if extra0 is not None:
            state["extra"] = unpack_extra(p).clone()
        pi, sc, fl = unpack(p)
        lnu = unpack_nu(p).detach().clone()
        new_pi = min(max(float(pi), 1e-6), 1.0 - 1e-9) if fit_pi else pi_c
        if not fit_cal:
            return new_pi, scale, floor, lnu, all_bad[:0]
        assert group is not None
        q = p[n_pi : n_pi + n_cal].reshape(n_groups, 4)
        bound_floor = 0.5 * math.log(100.0 * typical)
        bad = (q[:, [1, 3]].amax(-1) > bound_floor) | (
            q[:, 2].abs() > math.exp(bound_floor)
        )
        if fit_scale:
            bad |= q[:, 0].abs() > 8.0
        keep = (group >= 0) & ~bad[group.clamp(min=0)]
        sc_out = torch.where(keep, sc, scale).clone()
        fl_out = torch.where(keep[:, None, None], fl, floor).clone()
        return new_pi, sc_out, fl_out, lnu, bad


def _fit_all(
    fit,
    pi_c: float,
    scale: Tensor,
    floor: Tensor,
    log_nu_cell: Tensor,
    *,
    calibrate: str,
    own: Tensor | None,
) -> tuple[float, Tensor, Tensor, Tensor, bool]:
    """Completeness, calibration and density by maximum marginal likelihood.

    Calibration is fitted in two stages: one shared group over every pair,
    then, holding the other bins at the shared values, one group per bin with
    enough matched weight (``own``). A group outside physical bounds keeps
    its starting values, and completeness and density are then refitted with
    the calibration held. Returns ``(..., rejected)``: ``True`` when the shared
    fit was rejected (no calibration progress is possible).
    """
    n_bins = int(scale.numel())
    if calibrate == "none":
        pi_c, scale, floor, log_nu_cell, _ = fit(
            pi_c, scale, floor, log_nu_cell, group=None
        )
        return pi_c, scale, floor, log_nu_cell, False
    dev = scale.device
    shared = torch.zeros(n_bins, dtype=torch.int64, device=dev)
    p1, s1, f1, n1, bad = fit(pi_c, scale, floor, log_nu_cell, group=shared)
    if bool(bad.any()):
        p0, _, _, n0, _ = fit(pi_c, scale, floor, log_nu_cell, group=None)
        return p0, scale, floor, n0, True
    if own is None or n_bins == 1 or not bool(own.any()):
        return p1, s1, f1, n1, False
    per_bin = torch.full((n_bins,), -1, dtype=torch.int64, device=dev)
    per_bin[own] = torch.arange(int(own.sum()), device=dev)
    p2, s2, f2, n2, bad2 = fit(p1, s1, f1, n1, group=per_bin)
    if bool(bad2.any()):
        p2, _, _, n2, _ = fit(p1, s2, f2, n2, group=None)
    return p2, s2, f2, n2, False


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
    calibrate: Literal["none", "floor", "scale_floor"] = "none",
    calibrate_by: Tensor | None = None,
    calibrate_edges: Tensor | None = None,
    calibrate_min_pairs: float = 50.0,
    right_scale: float | Tensor | None = None,
    floor_mas2: Tensor | None = None,
    exclusive: bool = True,
    blends: Literal["none", "pairs"] = "none",
    blend_radius_arcsec: float | None = None,
    blend_flux: Tensor | None = None,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> Association:
    """Probabilistic counterparts of ``left`` rows in ``right`` (see module docs).

    Both catalogues need ``astrometry`` (position covariance at least).

    **Epochs.** ``epoch="right"`` matches at the right catalogue's epoch. When
    the right catalogue has per-row epochs (e.g. 2MASS observation dates), each
    candidate pair is evaluated with the left row propagated to *that right
    row's* epoch, with its covariance (pair epochs). The search runs at the
    median epoch, with each left row's radius widened by its motion over the
    epoch span. Its covariance bound uses the span's end points (``lambda_max``
    is convex in time). A number propagates both catalogues to it, and ``None``
    uses stored positions. ``motion_prior`` serves rows without proper motion
    (one prior, or one per side).

    **Calibration.** Published errors are often off: CatWISE positions, for
    instance, are underestimated. ``calibrate="floor"`` fits a 2x2 systematic
    covariance ``F`` added to every pair; ``"scale_floor"`` also scales the right
    catalogue's errors by ``s``. Both maximise the marginal likelihood jointly
    with completeness (and the background density), and are reported on the
    result. Search radii include the fitted terms and candidates are
    regenerated until the calibration is stable, so the gate stays sound.
    ``calibrate_by`` (one value per right row, e.g. magnitude, or a blending
    count from :func:`count_within`) with ``calibrate_edges`` fits the
    terms per bin: first one shared fit over every pair, then one per bin
    whose matched weight reaches ``calibrate_min_pairs``; the other bins keep
    the shared fit. A fit outside physical bounds (scale within ``e^+-8``,
    floor below 100x the typical pair variance) is not applied. Scale and
    floor are separately identifiable only when the right errors vary within
    a bin.

    ``right_scale`` (scalar or one per bin) and ``floor_mas2`` (``[2, 2]`` or
    ``[n_bins, 2, 2]``) set the calibration: held fixed with
    ``calibrate="none"``, starting values otherwise. This transfers a
    calibration fitted where it is identifiable (an uncrowded field, per
    magnitude bin) to a crowded field. There a single Gaussian error model
    cannot separate larger errors from blending: Gaia stars cluster around
    every blended CatWISE detection, and the fitted scale grows to absorb
    them.

    **Crowding.** With ``exclusive=True`` (default) a right row is the
    counterpart of at most one left row. Posteriors are then the Bethe
    (belief-propagation) marginals of that one-to-one model: each pair's
    likelihood ratio is discounted by the probability that its right row is
    not taken by a competing left row. They are exact when the candidate graph
    has no cycles and equal to the independent posteriors (``exclusive=False``,
    as nway and Sutherland & Saunders) when no right row is shared. In crowded
    fields, where one right source lies in the gates of many left rows, the
    independent model hands it to all of them; completeness, photometric terms
    and error calibration then learn from that multiple counting.

    The background density ``nu`` is the total right density for
    ``exclusive=False`` (nway's convention). For ``exclusive=True`` it is a
    parameter: the density of right rows that are not counterparts, fitted
    with the rest (a Gamma(3/2) prior on each cell's background count keeps it
    positive when every right row is matched). Completeness, calibration and
    background density are fitted by maximum marginal likelihood (L-BFGS on
    the exact or Bethe log likelihood), with :class:`HistogramRatio` terms
    updated by EM in between. With ``exclusive=True`` and calibration, the
    calibration and gate rounds first run with the independent model (no
    belief propagation) until stable; the exclusive rounds start from that
    calibration and stop by the same rule.

    ``right_footprint`` (MOC) defaults to the right catalogue's occupied cells at
    ``footprint_order``. Pass the survey MOC when you have it, so that density
    and priors use the true coverage. With ``exclusive=True``, right rows
    outside it take no part.

    **Blends.** ``blends="pairs"`` (exclusive model only) lets a right row be
    the blend of two left rows closer than ``blend_radius_arcsec`` (the right
    catalogue's resolution, required): its position is then the
    flux-weighted centroid of the two. ``blend_flux`` (one value per left
    row, linear flux in the right catalogue's band, NaN where unknown) fixes
    each pair's flux ratio; otherwise the ratio has a uniform prior and is
    marginalised. Single left rows and left pairs are independent sources
    conditioned on each left row contributing at most once; the blend rate
    (the probability that a left pair is detected as one right row) is
    fitted with the other parameters, starting, with calibration, from the
    independent model's calibration. Pair
    arrays then hold every (left, right) edge of a single or blend
    hypothesis, ``posterior`` is the probability that the left row is part
    of the right row's counterpart (alone or in a blend), ``match`` is the
    greedy selection of each right row's most probable hypothesis, and
    ``Association.blend`` holds the blend posteriors and member lists
    (:class:`~torchsky.catalogs.BlendAssociation`). Not available with
    photometric ``terms`` or per-row right epochs.
    """
    if left.astrometry is None or right.astrometry is None:
        raise ValueError("associate needs Catalog.astrometry on both catalogues")
    if blends not in ("none", "pairs"):
        raise ValueError("blends must be 'none' or 'pairs'")
    if blends == "pairs":
        if not exclusive:
            raise ValueError("blends='pairs' needs exclusive=True")
        if terms:
            raise ValueError("blends='pairs' does not support photometric terms")
        if blend_radius_arcsec is None or not blend_radius_arcsec > 0.0:
            raise ValueError(
                "blends='pairs' needs blend_radius_arcsec > 0 (the right "
                "catalogue's resolution, about 6 for WISE)"
            )
    if left.frame != right.frame:
        right = right.to_frame(left.frame)
    priors = (
        motion_prior
        if isinstance(motion_prior, tuple)
        else (motion_prior, motion_prior)
    )
    pair_epoch = False
    if epoch == "right" or epoch == "left":
        ref = right if epoch == "right" else left
        ep = torch.as_tensor(ref.epoch, dtype=_F64).reshape(-1)
        if ep.numel() != 1 and epoch == "left":
            raise ValueError(
                "the left catalogue has per-row epochs; pass a scalar epoch"
            )
        pair_epoch = ep.numel() != 1 and not bool((ep == ep[0]).all())
        target: Tensor | None = ep.median() if pair_epoch else ep[0]
    else:
        target = None if epoch is None else torch.as_tensor(epoch, dtype=_F64)
    llon, llat, lcov = _epoch_positions(left, target, priors[0])
    dev = llon.device
    lam_l = _lambda_max(lcov)
    drift_mas = torch.zeros_like(lam_l)
    if pair_epoch:
        assert target is not None
        rlon, rlat = right.lon.to(_F64), right.lat.to(_F64)
        right_ast = right.astrometry
        assert right_ast is not None
        rcov = right_ast.position_cov().to(dev)
        t_right = torch.broadcast_to(
            _as_tensor(right.epoch, dtype=_F64, device=dev), rlon.shape
        )
        t_lo, t_hi = t_right.min(), t_right.max()
        state = left.to_state()
        span = float(torch.maximum((t_lo - target).abs(), (t_hi - target).abs()))
        speeds = [_pm_speed(state, priors[0])]
        for t_end in (t_lo, t_hi):
            moved, cov_end = propagate_astrometry(
                state, left.astrometry, t_end, motion_prior=priors[0]
            )
            lam_l = torch.maximum(lam_l, _lambda_max(cov_end))
            speeds.append(_pm_speed(moved, priors[0]))
        drift_mas = torch.stack(speeds).amax(0) * span
    else:
        rlon, rlat, rcov = _epoch_positions(right, target, priors[1])
        t_right = None
    lam_r = _lambda_max(rcov)
    k2 = gate_k2(gate_pvalue)
    k = math.sqrt(k2)
    rfoot = (
        right_footprint
        if right_footprint is not None
        else MOC.from_lonlat(rlon, rlat, footprint_order)
    )
    dens = _Density(rlon, rlat, llon, llat, rfoot, density, density_order)
    right_inside = dens.w_in > 0
    log_nu_cell = torch.log(dens.cell_nu().clamp_min(1e-300))
    inside = rfoot.contains(llon, llat).to(_F64)
    n_left = int(llon.numel())
    lx, rx = unit_vectors(llon, llat), unit_vectors(rlon, rlat)
    if calibrate_by is not None:
        if calibrate_edges is None:
            raise ValueError("calibrate_by needs calibrate_edges")
        by = _as_tensor(calibrate_by, dtype=_F64, device=dev).reshape(-1)
        edges = _as_tensor(calibrate_edges, dtype=_F64, device=dev).reshape(-1)
        if by.numel() != rlon.numel() or edges.numel() < 2:
            raise ValueError(
                "calibrate_by needs one value per right row and >= 2 edges"
            )
        n_bins = int(edges.numel()) - 1
        rbin = (
            torch.bucketize(
                by.nan_to_num(float(edges[0])).contiguous(), edges, right=True
            )
            - 1
        ).clamp(0, n_bins - 1)
    else:
        n_bins = 1
        rbin = torch.zeros(rlon.shape, dtype=torch.int64, device=dev)
    scale = torch.ones(n_bins, dtype=_F64, device=dev)
    floor = torch.zeros(n_bins, 2, 2, dtype=_F64, device=dev)
    if right_scale is not None:
        given = _as_tensor(right_scale, dtype=_F64, device=dev).reshape(-1)
        if given.numel() not in (1, n_bins) or not bool((given > 0).all()):
            raise ValueError("right_scale must be positive, scalar or one per bin")
        scale = given.expand(n_bins).clone()
    if floor_mas2 is not None:
        given_f = _as_tensor(floor_mas2, dtype=_F64, device=dev)
        if given_f.shape not in ((2, 2), (n_bins, 2, 2)):
            raise ValueError("floor_mas2 must be [2, 2] or [n_bins, 2, 2]")
        if bool((torch.linalg.eigvalsh(0.5 * (given_f + given_f.mT)) < 0).any()):
            raise ValueError("floor_mas2 must be positive semi-definite")
        floor = given_f.expand(n_bins, 2, 2).clone()
    if blends == "pairs":
        if pair_epoch:
            raise ValueError("blends='pairs' does not support per-row right epochs")
        from ._blends import associate_blends

        assert blend_radius_arcsec is not None
        pi_start = None
        if calibrate != "none":
            # Start from the independent model's calibration (cheap, no belief
            # propagation), as the exclusive model does. Started from
            # underestimated errors, blends would absorb the offsets instead.
            pre = associate(
                left,
                right,
                gate_pvalue=gate_pvalue,
                max_radius_arcsec=max_radius_arcsec,
                epoch=epoch,
                motion_prior=motion_prior,
                right_footprint=rfoot,
                density=density,
                density_order=density_order,
                completeness=completeness,
                assign="none",
                calibrate=calibrate,
                calibrate_by=calibrate_by,
                calibrate_edges=calibrate_edges,
                calibrate_min_pairs=calibrate_min_pairs,
                right_scale=right_scale,
                floor_mas2=floor_mas2,
                exclusive=False,
                max_iter=max_iter,
                tol=tol,
            )
            scale = _as_tensor(pre.right_scale, dtype=_F64, device=dev).reshape(-1)
            scale = scale.expand(n_bins).clone()
            assert pre.floor_mas2 is not None
            floor = pre.floor_mas2.to(dev).expand(n_bins, 2, 2).clone()
            pi_start = pre.completeness
        return associate_blends(
            llon=llon,
            llat=llat,
            lcov=lcov,
            lam_l=lam_l,
            rlon=rlon,
            rlat=rlat,
            rcov=rcov,
            lam_r=lam_r,
            rbin=rbin,
            scale=scale,
            floor=floor,
            dens=dens,
            inside=inside,
            k2=k2,
            gate_pvalue=gate_pvalue,
            max_radius_arcsec=max_radius_arcsec,
            completeness=completeness,
            calibrate=calibrate,
            calibrate_min_pairs=calibrate_min_pairs,
            assign=assign,
            blend_radius_arcsec=float(blend_radius_arcsec),
            blend_flux=blend_flux,
            pi_start=pi_start,
        )
    rounds = 1 if calibrate == "none" else 8
    outside_log = torch.where(
        inside > 0, torch.zeros_like(inside), torch.full_like(inside, -math.inf)
    )
    pi_c = 0.5
    prev_bp: tuple[Tensor, Tensor, Tensor] | None = None
    # Exclusive model with calibration: the calibration and gate rounds first
    # run with the independent model (cheap, no belief propagation) until
    # stable. That only sets the starting calibration and gate; the exclusive
    # rounds then run to the same stopping rule, so their result is the same
    # self-consistent exclusive fit.
    pre_left = 8 if exclusive and calibrate != "none" else 0
    use_excl = exclusive and pre_left == 0
    for _round in range(rounds + pre_left):
        lam_f = float(_lambda_max(floor).max())
        # sqrt(lmax(A + s^2 B + F)) <= sqrt(lmax A) + s sqrt(lmax B) + sqrt(lmax F)
        rl_raw = (k * (torch.sqrt(lam_l) + math.sqrt(lam_f)) + drift_mas) / 1000.0
        rr_raw = k * scale[rbin] * torch.sqrt(lam_r) / 1000.0
        rl = rl_raw.clamp(max=max_radius_arcsec)
        rr = rr_raw.clamp(max=max_radius_arcsec)
        n_capped = int(
            (rl_raw > max_radius_arcsec).sum() + (rr_raw > max_radius_arcsec).sum()
        )
        li, rj = _sound_candidates(llon, llat, rlon, rlat, rl, rr)
        if pair_epoch:
            assert t_right is not None
            gathered = _gather_state(left.to_state(), li)
            moved, lc = propagate_astrometry(
                gathered, left.astrometry.take(li), t_right[rj], motion_prior=priors[0]
            )
            lxp = unit_vectors(moved.lon, moved.lat)
        else:
            lxp, lc = lx[li], lcov[li]
        delta, tl, tr, psi = pair_frame(lxp, rx[rj])
        cl = tl @ lc @ tl.mT
        cr = tr @ rcov[rj] @ tr.mT
        pb = rbin[rj]
        d2, logdet = mahalanobis2(
            delta, cl + (scale[pb] ** 2)[:, None, None] * cr + floor[pb]
        )
        sep_arcsec = psi * _ARCSEC_PER_RAD
        keep = (d2 <= k2) & (sep_arcsec <= max_radius_arcsec)
        if use_excl:
            keep &= right_inside[rj]
        li, rj, d2, logdet, sep_arcsec = (
            li[keep],
            rj[keep],
            d2[keep],
            logdet[keep],
            sep_arcsec[keep],
        )
        delta, cl, cr = delta[keep], cl[keep], cr[keep]
        pb = rbin[rj]
        # Background density: the total right density, or (exclusive model) the
        # density of right rows that are not counterparts (closed-form ML).
        bp_state: dict = {}
        if prev_bp is not None:
            # Warm-start the messages of pairs kept from the previous round (the
            # fixed point is unique, so this changes only the sweeps needed).
            bp_state["rho"] = _carry_messages(*prev_bp, li, rj, int(rlon.numel()))
        fixed = (scale.clone(), floor.clone())

        def log_n_of(sc: Tensor, fl: Tensor) -> tuple[Tensor, Tensor, Tensor]:
            sig = cl + (sc[pb] ** 2)[:, None, None] * cr + fl[pb]
            d2_, logdet_ = mahalanobis2(delta, sig)
            return -math.log(2.0 * math.pi) - 0.5 * logdet_ - 0.5 * d2_, d2_, logdet_

        def log_weights(pi_c: Tensor | float, sc: Tensor, fl: Tensor, lnu: Tensor):
            log_n, _, _ = log_n_of(sc, fl)
            # Exclusive model: background density at the right row (its cell's
            # expected matches never exceed its count); independent model: at
            # the left row, as nway.
            log_lr = log_n - (
                dens.log_right(lnu)[rj] if use_excl else dens.log_left(lnu)[li]
            )
            for term in terms:
                log_lr = log_lr + term.log_rho(li, rj)
            pi_eff = _as_tensor(pi_c, dtype=_F64, device=dev) * (1.0 - gate_pvalue)
            log_pi = torch.log(pi_eff) + outside_log
            return log_pi[li] + log_lr, torch.log1p(-pi_eff * inside), log_lr

        def posterior(
            pi_c: float,
        ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
            log_w, log_a, log_lr = log_weights(pi_c, scale, floor, log_nu_cell)
            pi = (pi_c * (1.0 - gate_pvalue)) * inside
            if use_excl:
                post_x, p_none_x, rho = pairwise_exclusive(
                    li,
                    rj,
                    log_w,
                    log_a,
                    init=bp_state.get("rho"),
                    cache=bp_state.setdefault("graph", {}),  # type: ignore[arg-type]
                )
                bp_state["rho"] = rho.detach()
                return post_x, p_none_x, pi, log_lr
            z = _segment_logaddexp(log_w, li, n_left, log_a)
            return torch.exp(log_w - z[li]), torch.exp(log_a - z), pi, log_lr

        fit_pi = completeness == "em"
        if not fit_pi and not 0.0 <= float(completeness) <= 1.0:  # type: ignore[arg-type]
            raise ValueError("completeness must lie in [0, 1] or be 'em'")
        if _round == 0:
            pi_c = 0.5 if fit_pi else float(completeness)  # type: ignore[arg-type]
        for term in terms:
            term.setup(rfoot.contains(rlon, rlat), li, rj)
        fit_cal = calibrate != "none"
        rejected = False
        if fit_cal:
            own = torch.zeros(1, dtype=torch.bool, device=dev)
            if n_bins > 1:  # per-bin fits need the matched weight per bin
                post0, _, _, _ = posterior(pi_c)
                mass = torch.zeros(n_bins, dtype=_F64, device=dev).index_add(
                    0, pb, post0.detach()
                )
                own = mass >= calibrate_min_pairs
        typical = (
            float(torch.median(0.5 * (cl + cr).diagonal(dim1=-2, dim2=-1).sum(-1)))
            if li.numel()
            else 1.0
        )
        prev = None
        for _ in range(max_iter):
            # Maximum marginal likelihood in completeness, calibration and (exclusive
            # model) background density; then EM updates of photometric terms.
            if fit_pi or fit_cal or use_excl:

                def fit(pi0, sc0, fl0, nu0, group):
                    return _fit_marginal(
                        log_weights,
                        li,
                        rj,
                        n_left,
                        use_excl,
                        bp_state,
                        pi0,
                        sc0,
                        fl0,
                        nu0,
                        dens if use_excl else None,
                        fit_pi=fit_pi,
                        calibrate=calibrate,
                        group=group,
                        typical=typical,
                    )

                pi_c, scale, floor, log_nu_cell, rejected = _fit_all(
                    fit,
                    pi_c,
                    scale,
                    floor,
                    log_nu_cell,
                    calibrate=calibrate,
                    own=own if fit_cal else None,
                )
            if not terms:
                break
            post, _, _, _ = posterior(pi_c)
            for term in terms:
                term.update(li, rj, post)
            done = prev is not None and (
                post.numel() == 0 or float((post - prev).abs().max()) < tol
            )
            prev = post.detach()
            if done:
                break
        post, p_none, pi, log_lr = posterior(pi_c)
        _, d2, logdet = log_n_of(scale, floor)
        d2 = d2.detach()
        if not fit_cal:
            break
        if use_excl and "rho" in bp_state:
            prev_bp = (li, rj, bp_state["rho"])
        moved_by = float((scale - fixed[0]).abs().max()) + float(
            (floor - fixed[1]).abs().max()
        ) / max(1.0, float(floor.abs().max()))
        # The gate grew or shrank with the calibration: regenerate candidates
        # until the calibration is stable (or the shared fit is out of bounds).
        if pre_left and not use_excl:
            pre_left -= 1
            if moved_by < 1e-4 or rejected or pre_left == 0:
                # The exclusive rounds start here. Their background density
                # (non-counterparts) starts at its closed-form maximum given
                # the independent counterpart masses (at most 1 per right row).
                use_excl = True
                matched = (
                    torch.zeros(int(rlon.numel()), dtype=_F64, device=dev)
                    .index_add(0, rj, post.detach())
                    .clamp(max=1.0)
                )
                log_nu_cell = torch.log(dens.cell_nu(matched).clamp_min(1e-300))
            continue
        if moved_by < 1e-4 or rejected:
            break
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
        right_scale=float(scale[0]) if n_bins == 1 else scale,
        floor_mas2=floor[0] if n_bins == 1 else floor,
        density_mas2=dens.log_left(log_nu_cell).exp(),
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
    # Distinct left / right rows per component (1-D unique on packed keys).
    lkey = torch.unique(comp_id * n_left + li)
    rkey = torch.unique(comp_id * n_right + rj)
    lefts = torch.bincount(lkey // n_left, minlength=n_comp)
    rights = torch.bincount(rkey // n_right, minlength=n_comp)
    star = (lefts == 1) | (rights == 1)
    in_star = star[comp_id]
    if bool(in_star.any()):
        best = torch.full(
            (n_comp,), -math.inf, dtype=w.dtype, device=dev
        ).scatter_reduce(0, comp_id[in_star], w[in_star], "amax")
        cand = in_star & (w == best[comp_id])
        # Exact ties: keep the lowest pair position per component.
        idx = torch.arange(int(w.numel()), device=dev)
        first = _scatter_reduce_int(
            torch.full((n_comp,), int(w.numel()), dtype=torch.int64, device=dev),
            0,
            comp_id[cand],
            idx[cand],
            "amin",
        )
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


# ------------------------------------------------------------------- N-way


def _to_left_frame(left_xyz: Tensor, right_xyz: Tensor) -> tuple[Tensor, Tensor]:
    """Offset of ``right`` in ``left``'s tangent basis (mas, arc length) and the
    parallel transport ``T`` of ``right``'s basis to ``left``'s (``C_l = T C_r T^T``)."""
    a = torch.linalg.cross(left_xyz, right_xyz, dim=-1)
    cpsi = (left_xyz * right_xyz).sum(-1).clamp(-1.0, 1.0)
    psi = torch.atan2(a.norm(dim=-1), cpsi)
    el, nl = _basis(left_xyz)
    chord_perp = (
        right_xyz - left_xyz * cpsi[..., None]
    )  # tangential part, |.| = sin psi
    sinpsi = a.norm(dim=-1)
    scale = torch.where(
        sinpsi > 0, psi / sinpsi.clamp_min(1e-300), torch.ones_like(psi)
    )
    delta = (
        torch.stack(((chord_perp * el).sum(-1), (chord_perp * nl).sum(-1)), -1)
        * (scale * _MAS_PER_RAD)[..., None]
    )
    # Rotation taking right -> left: axis -a, angle psi. With k = -a / sin psi:
    # k sin psi = -a and (1 - cos psi) k k^T = a a^T / (1 + cos psi).
    s = -torch.ones_like(psi)
    omc = 1.0 / (1.0 + cpsi)
    er, nr = _basis(right_xyz)
    re = _rotate(er, a, s, cpsi, omc)
    rn = _rotate(nr, a, s, cpsi, omc)
    t = torch.stack(
        (
            torch.stack(((re * el).sum(-1), (rn * el).sum(-1)), -1),
            torch.stack(((re * nl).sum(-1), (rn * nl).sum(-1)), -1),
        ),
        -2,
    )
    return delta, t


@dataclass
class NWayAssociation:
    """Hypotheses for each primary row, in CSR layout.

    ``members[h, c]`` is the row of catalogue ``c`` in hypothesis ``h`` (``-1``
    when absent; column 0 is the primary row). Hypotheses of primary ``i`` are
    ``offsets[i]:offsets[i + 1]``. ``log_weight`` is the unnormalised log
    weight and ``posterior`` its normalisation within the primary.
    ``p_any[i]`` is the probability that primary ``i`` has at least one counterpart.
    ``residual_mass[i]`` is a certified upper bound on the posterior mass of
    hypotheses that were not enumerated (0 when all were). ``completeness[c]``
    and ``best`` (the MAP hypothesis per primary) are also returned.
    """

    members: Tensor
    offsets: Tensor
    log_weight: Tensor
    posterior: Tensor
    p_any: Tensor
    residual_mass: Tensor
    completeness: list[float]
    best: Tensor


def _hypothesis_log_weight(
    members: Tensor,
    pos: list[Tensor],
    cov_inv: list[Tensor],
    logdet: list[Tensor],
    log_prior_present: Tensor,
    log_prior_absent: Tensor,
    log_nu: list[Tensor],
) -> Tensor:
    """Exact log weight of hypotheses ``members [H, K]`` (information form)."""
    h, k = members.shape
    dev = members.device
    w = torch.zeros(h, 2, 2, dtype=_F64, device=dev)
    b = torch.zeros(h, 2, dtype=_F64, device=dev)
    q = torch.zeros(h, dtype=_F64, device=dev)
    log_w = torch.zeros(h, dtype=_F64, device=dev)
    present = torch.zeros(h, dtype=_F64, device=dev)
    for c in range(k):
        rows = members[:, c]
        on = rows >= 0
        idx = rows.clamp(min=0)
        ci = cov_inv[c][idx] * on[:, None, None]
        x = pos[c][idx] * on[:, None]
        w = w + ci
        cx = (ci @ x[:, :, None])[:, :, 0]
        b = b + cx
        q = q + (x * cx).sum(-1)
        log_w = log_w - 0.5 * logdet[c][idx] * on
        if c > 0:
            present = present + on
            log_w = log_w + torch.where(
                on, log_prior_present[c] - log_nu[c][idx], log_prior_absent[c].expand(h)
            )
    wa, wb, wd = w[:, 0, 0], w[:, 0, 1], w[:, 1, 1]
    det_w = wa * wd - wb * wb
    winv_b = (
        torch.stack((wd * b[:, 0] - wb * b[:, 1], -wb * b[:, 0] + wa * b[:, 1]), -1)
        / det_w[:, None]
    )
    chi2 = q - (b * winv_b).sum(-1)
    return (
        log_w - present * math.log(2.0 * math.pi) - 0.5 * torch.log(det_w) - 0.5 * chi2
    )


def _gather_state(sub, idx: Tensor):
    """Rows ``idx`` of a CatalogState (per-row epochs kept)."""

    def pick(v):
        return None if v is None else v[idx]

    return type(sub)(
        lon=sub.lon[idx],
        lat=sub.lat[idx],
        frame=sub.frame,
        epoch=torch.broadcast_to(torch.as_tensor(sub.epoch, dtype=_F64), sub.lon.shape)[
            idx
        ],
        epoch_scale=sub.epoch_scale,
        epoch_format=sub.epoch_format,
        pm_lon_coslat_mas_yr=pick(sub.pm_lon_coslat_mas_yr),
        pm_lat_mas_yr=pick(sub.pm_lat_mas_yr),
        parallax_mas=pick(sub.parallax_mas),
        radial_velocity_km_s=pick(sub.radial_velocity_km_s),
    )


def _psd_part(m: Tensor) -> Tensor:
    """Positive semi-definite part of symmetric ``[..., 2, 2]`` matrices."""
    w, v = torch.linalg.eigh(0.5 * (m + m.mT))
    return v @ torch.diag_embed(w.clamp(min=0.0)) @ v.mT


def _per_row_epochs(cat: Catalog) -> Tensor | None:
    ep = torch.as_tensor(cat.epoch, dtype=_F64).reshape(-1)
    if ep.numel() == 1 or bool((ep == ep[0]).all()):
        return None
    return torch.broadcast_to(ep, cat.lon.shape)


def associate_many(
    catalogs: list[Catalog],
    *,
    primary: int = 0,
    gate_pvalue: float = 1e-4,
    max_radius_arcsec: float = 10.0,
    epoch: float | Tensor | None = None,
    motion_prior: MotionPrior | None = None,
    footprints: list[MOC | None] | None = None,
    footprint_order: int = 10,
    completeness: list[float] | Literal["em"] = "em",
    terms: list[tuple[HistogramRatio, ...]] | None = None,
    budget: int = 4096,
    mass_tol: float = 1e-6,
    exclusive: bool = True,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> NWayAssociation:
    """N-way association (nway-style) of a primary catalogue with several others.

    For each primary row, the hypotheses are every combination of, per secondary
    catalogue, one counterpart that passes the pairwise gate against the primary,
    or none. A hypothesis with present set ``S`` has weight

    ``prod_{c in S} pi_c rho_c / nu_c * prod_{c not in S} (1 - pi_c) * M_S``

    where ``M_S`` is the exact Gaussian marginal likelihood that all present
    positions (primary included) come from one true position with a flat prior.
    This is Budavari & Szalay (2008) eq. 18 for isotropic errors, here with full
    covariances in the primary's tangent plane. ``pi_c`` is catalogue ``c``'s
    completeness (pairwise EM when ``"em"``, also counting gate losses);
    ``nu_c`` is its surface density at the primary over its footprint;
    ``rho_c`` is the product of the member's photometric ratios (1 without
    ``terms``).

    Primaries with at most ``budget`` hypotheses are enumerated exactly.
    Larger ones are enumerated best-first by an upper bound (``M`` gains at most
    ``1 / (2 pi sqrt(det C))`` per added member) until the unenumerated mass,
    certified through that bound, is below ``mass_tol``. The certified residual
    is reported per primary.

    ``terms`` adds photometric likelihood ratios: one tuple of
    :class:`HistogramRatio` per secondary catalogue (in ``catalogs`` order,
    the primary skipped), with the semantics of :func:`associate`. Right
    magnitudes are that catalogue's values; pair colours
    (``left_values``) are relative to the primary. Each present member's
    factor is multiplied by its ``rho = q / n``. The terms are first learnt
    by each pairwise association (which also fits ``pi_c`` and ``nu_c`` with
    them), then by EM from the hypothesis posteriors: a candidate's
    counterpart weight is the posterior mass of the hypotheses holding it.
    ``max_iter`` and ``tol`` bound that EM (and the pairwise ones), as in
    :func:`associate`. The best-first bound includes each candidate's
    ``rho``, so the residual stays certified. The given
    :class:`HistogramRatio` objects hold the learnt ``q`` afterwards.

    With ``exclusive=True`` (default) each secondary source is the counterpart
    of at most one primary. Hypothesis posteriors are then the Bethe
    (belief-propagation) marginals of that constraint across all primaries and
    catalogues (see :func:`associate`); ``log_weight`` stays the unconstrained
    weight, and the background densities ``nu_c`` are the EM-fitted ones of the
    pairwise associations. ``exclusive=False`` gives nway's independent
    per-primary posteriors.
    """
    if len(catalogs) < 2:
        raise ValueError("need a primary and at least one other catalogue")
    order = [primary] + [i for i in range(len(catalogs)) if i != primary]
    cats = [catalogs[i] for i in order]
    feet: list[MOC | None] = (
        [None] * len(cats) if footprints is None else [footprints[i] for i in order]
    )
    k = len(cats)
    term_sets: list[tuple[HistogramRatio, ...]] = [()] * k
    if terms is not None:
        if len(terms) != k - 1:
            raise ValueError("terms needs one tuple per secondary catalogue")
        term_sets = [()] + [tuple(t) for t in terms]
    ref = cats[0]
    if epoch is None:
        ep = torch.as_tensor(ref.epoch)
        if ep.numel() != 1:
            raise ValueError("the primary has per-row epochs; pass a scalar epoch")
        epoch = ep.reshape(())
    pos_lonlat, covs = [], []
    for c_idx, cat in enumerate(cats):
        cat = cat if cat.frame == ref.frame else cat.to_frame(ref.frame)
        if c_idx > 0 and _per_row_epochs(cat) is not None:
            # Evaluated at pair epochs below; stored positions serve the density.
            if cat.astrometry is None:
                raise ValueError(
                    "associate_many needs Catalog.astrometry on every catalogue"
                )
            lon, lat, cov = (
                cat.lon.to(_F64),
                cat.lat.to(_F64),
                cat.astrometry.position_cov(),
            )
        else:
            lon, lat, cov = _epoch_positions(
                cat, torch.as_tensor(epoch, dtype=_F64), motion_prior
            )
        pos_lonlat.append((lon, lat))
        covs.append(cov)
    n0 = int(pos_lonlat[0][0].numel())
    dev = pos_lonlat[0][0].device
    xyz = [unit_vectors(lon, lat) for lon, lat in pos_lonlat]
    # Pairwise gated candidates and completeness against the primary.
    pairs: list[tuple[Tensor, Tensor]] = [
        (torch.arange(n0, device=dev), torch.arange(n0, device=dev))
    ]
    pi = [1.0]
    log_nu: list[Tensor] = [torch.zeros(n0, dtype=_F64, device=dev)]
    pair_geometry: dict[int, tuple[Tensor, Tensor]] = {}
    for c in range(1, k):
        left = Catalog(
            lon=pos_lonlat[0][0],
            lat=pos_lonlat[0][1],
            epoch=epoch,
            astrometry=_cov_only(covs[0]),
        )
        right = Catalog(
            lon=pos_lonlat[c][0],
            lat=pos_lonlat[c][1],
            epoch=epoch,
            astrometry=_cov_only(covs[c]),
        )
        comp = "em" if completeness == "em" else float(completeness[c - 1])
        t_rows = _per_row_epochs(cats[c])
        if t_rows is None:
            res = associate(
                left,
                right,
                gate_pvalue=gate_pvalue,
                max_radius_arcsec=max_radius_arcsec,
                epoch=None,
                right_footprint=feet[c],
                footprint_order=footprint_order,
                completeness=comp,
                terms=term_sets[c],
                assign="none",
                exclusive=exclusive,
                max_iter=max_iter,
                tol=tol,
            )
        else:
            # Per-row epochs (e.g. 2MASS dates): gate at pair epochs, then express each
            # candidate relative to the primary propagated to that candidate's epoch.
            res = associate(
                cats[0],
                cats[c],
                gate_pvalue=gate_pvalue,
                max_radius_arcsec=max_radius_arcsec,
                epoch="right",
                motion_prior=motion_prior,
                right_footprint=feet[c],
                footprint_order=footprint_order,
                completeness=comp,
                terms=term_sets[c],
                assign="none",
                exclusive=exclusive,
                max_iter=max_iter,
                tol=tol,
            )
            li_c, rj_c = res.left_index, res.right_index
            ast0 = cats[0].astrometry
            ast_c = cats[c].astrometry
            assert ast0 is not None and ast_c is not None
            moved, cov0_t = propagate_astrometry(
                _gather_state(cats[0].to_state(), li_c),
                ast0.take(li_c),
                t_rows.to(dev)[rj_c],
                motion_prior=motion_prior,
            )
            delta_c, t_c = _to_left_frame(
                unit_vectors(moved.lon, moved.lat),
                unit_vectors(cats[c].lon.to(_F64)[rj_c], cats[c].lat.to(_F64)[rj_c]),
            )
            cc = t_c @ ast_c.position_cov().to(dev)[rj_c] @ t_c.mT + _psd_part(
                cov0_t - covs[0][li_c]
            )
            pair_geometry[c] = (delta_c, cc)
        pairs.append((res.left_index, res.right_index))
        pi.append(res.completeness * (1.0 - gate_pvalue))
        rlon_c, rlat_c = pos_lonlat[c]
        llon0, llat0 = pos_lonlat[0]
        given = feet[c]
        rfoot: MOC = (
            given
            if given is not None
            else MOC.from_lonlat(rlon_c, rlat_c, footprint_order)
        )
        assert res.density_mas2 is not None
        nu = res.density_mas2.to(dev)
        inside = rfoot.contains(llon0, llat0)
        log_nu.append(
            torch.where(
                inside, torch.log(nu.clamp_min(1e-300)), torch.full_like(nu, math.inf)
            )
        )
    # A secondary with no gated candidate is absent from every hypothesis: its
    # factor (1 - pi_c) is common to all of a primary's hypotheses, so the
    # posteriors are those of the association without it. Solve without such
    # catalogues and report them as absent (-1), keeping log_weight exact.
    no_cand = [c for c in range(1, k) if int(pairs[c][0].numel()) == 0]
    if no_cand:
        back_order = torch.tensor(order, device=dev)
        inv_order = torch.argsort(back_order)
        log_absent = sum(math.log1p(-pi[c]) for c in no_cand)
        comp_all = [pi[c] / (1.0 - gate_pvalue) if c else 1.0 for c in range(k)]
        keep = [c for c in range(k) if c not in no_cand]
        if len(keep) == 1:
            members_int = torch.full((n0, k), -1, dtype=torch.int64, device=dev)
            members_int[:, 0] = torch.arange(n0, device=dev)
            return NWayAssociation(
                members=members_int[:, inv_order],
                offsets=torch.arange(n0 + 1, dtype=torch.int64, device=dev),
                log_weight=torch.full((n0,), log_absent, dtype=_F64, device=dev),
                posterior=torch.ones(n0, dtype=_F64, device=dev),
                p_any=torch.zeros(n0, dtype=_F64, device=dev),
                residual_mass=torch.zeros(n0, dtype=_F64, device=dev),
                completeness=[comp_all[int(i)] for i in inv_order.tolist()],
                best=torch.arange(n0, dtype=torch.int64, device=dev),
            )
        sub = associate_many(
            [cats[c] for c in keep],
            primary=0,
            gate_pvalue=gate_pvalue,
            max_radius_arcsec=max_radius_arcsec,
            epoch=epoch,
            motion_prior=motion_prior,
            footprints=None if footprints is None else [feet[c] for c in keep],
            footprint_order=footprint_order,
            completeness=(
                "em"
                if completeness == "em"
                else [completeness[c - 1] for c in keep[1:]]
            ),
            terms=None if terms is None else [term_sets[c] for c in keep[1:]],
            budget=budget,
            mass_tol=mass_tol,
            exclusive=exclusive,
            max_iter=max_iter,
            tol=tol,
        )
        n_h = int(sub.members.shape[0])
        members_int = torch.full((n_h, k), -1, dtype=torch.int64, device=dev)
        members_int[:, keep] = sub.members
        for pos, c in enumerate(keep):
            comp_all[c] = sub.completeness[pos]
        return NWayAssociation(
            members=members_int[:, inv_order],
            offsets=sub.offsets,
            log_weight=sub.log_weight + log_absent,
            posterior=sub.posterior,
            p_any=sub.p_any,
            residual_mass=sub.residual_mass,
            completeness=[comp_all[int(i)] for i in inv_order.tolist()],
            best=sub.best,
        )
    # Per-candidate geometry in each primary's tangent frame.
    cov_inv: list[Tensor] = [torch.linalg.inv(covs[0])]
    logdet: list[Tensor] = [torch.logdet(covs[0])]
    empty = torch.empty(0, dtype=_F64, device=dev)
    cand_pos: list[Tensor] = [empty]
    cand_cinv: list[Tensor] = [empty]
    cand_ld: list[Tensor] = [empty]
    for c in range(1, k):
        li, rj = pairs[c]
        if c in pair_geometry:
            delta, cc = pair_geometry[c]
        else:
            delta, t = _to_left_frame(xyz[0][li], xyz[c][rj])
            cc = t @ covs[c][rj] @ t.mT
        cand_pos.append(delta)
        cand_cinv.append(torch.linalg.inv(cc))
        cand_ld.append(torch.logdet(cc))
    # Hypothesis members refer to candidate slots; geometry is indexed by slot.
    log_pp = torch.tensor([0.0] + [math.log(p) for p in pi[1:]], dtype=_F64, device=dev)
    log_pa = torch.tensor(
        [0.0] + [math.log1p(-p) for p in pi[1:]], dtype=_F64, device=dev
    )
    # Per primary, candidate slots per catalogue (CSR by primary, already sorted by left).
    counts = [torch.bincount(pairs[c][0], minlength=n0) for c in range(1, k)]
    starts = [torch.cumsum(cnt, 0) - cnt for cnt in counts]
    n_hyp = torch.ones(n0, dtype=torch.int64, device=dev)
    for cnt in counts:
        n_hyp = n_hyp * (cnt + 1)
    exact = n_hyp <= budget
    residual = torch.zeros(n0, dtype=_F64, device=dev)
    log_remaining = torch.full((n0,), -math.inf, dtype=_F64, device=dev)

    def slot_geometry(
        c: int, slots: Tensor, prim: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        # The background density is evaluated at the primary's position.
        return (
            cand_pos[c][slots],
            cand_cinv[c][slots],
            cand_ld[c][slots],
            log_nu[c][prim],
        )

    def geo_weights(prim: Tensor, slots: Tensor) -> Tensor:
        """Positional log weight; ``slots [H, K-1]`` candidate slot per catalogue (-1 absent)."""
        h = int(prim.numel())
        geo_pos = [torch.zeros(h, 2, dtype=_F64, device=dev)]
        geo_ci = [cov_inv[0][prim]]
        geo_ld = [logdet[0][prim]]
        geo_nu = [torch.zeros(h, dtype=_F64, device=dev)]
        mem = torch.zeros(h, k, dtype=torch.int64, device=dev)
        mem[:, 0] = torch.arange(h, device=dev)  # geometry lists are per hypothesis
        for c in range(1, k):
            s = slots[:, c - 1]
            on = s >= 0
            p, ci, ld, nu = slot_geometry(c, s.clamp(min=0), prim)
            geo_pos.append(p)
            geo_ci.append(ci)
            geo_ld.append(ld)
            geo_nu.append(nu)
            mem[:, c] = torch.where(
                on, torch.arange(h, device=dev), torch.full_like(s, -1)
            )
        return _hypothesis_log_weight(
            mem, geo_pos, geo_ci, geo_ld, log_pp, log_pa, geo_nu
        )

    # Photometric terms: one log ratio per candidate slot (sum over the
    # catalogue's terms), multiplying each present member's factor.
    def slot_log_rho(c: int) -> Tensor:
        li_c, rj_c = pairs[c]
        out = torch.zeros(int(li_c.numel()), dtype=_F64, device=dev)
        for term in term_sets[c]:
            out = out + term.log_rho(li_c, rj_c)
        return out

    cand_lrho: list[Tensor] = [empty] + [slot_log_rho(c) for c in range(1, k)]

    def term_weights(slots: Tensor) -> Tensor:
        out = torch.zeros(int(slots.shape[0]), dtype=_F64, device=dev)
        for c in range(1, k):
            if not term_sets[c] or cand_lrho[c].numel() == 0:
                continue
            s = slots[:, c - 1]
            out = out + torch.where(
                s >= 0, cand_lrho[c][s.clamp(min=0)], torch.zeros_like(out)
            )
        return out

    def weights(prim: Tensor, slots: Tensor) -> Tensor:
        return geo_weights(prim, slots) + term_weights(slots)

    # Exact enumeration, vectorised: mixed-radix decode of hypothesis ids.
    prim_exact = torch.nonzero(exact, as_tuple=True)[0]
    slots_exact = torch.empty(0, k - 1, dtype=torch.int64, device=dev)
    prim_e = torch.empty(0, dtype=torch.int64, device=dev)
    if prim_exact.numel():
        nh = n_hyp[prim_exact]
        prim_e = torch.repeat_interleave(prim_exact, nh)
        local = torch.arange(int(nh.sum()), device=dev) - torch.repeat_interleave(
            torch.cumsum(nh, 0) - nh, nh
        )
        slots_exact = torch.empty(
            int(prim_e.numel()), k - 1, dtype=torch.int64, device=dev
        )
        rem = local
        for c in range(1, k):
            base = counts[c - 1][prim_e] + 1
            choice = rem % base
            rem = rem // base
            slots_exact[:, c - 1] = torch.where(
                choice == 0,
                torch.full_like(choice, -1),
                starts[c - 1][prim_e] + choice - 1,
            )
    geo_exact = geo_weights(prim_e, slots_exact)
    big = torch.nonzero(~exact, as_tuple=True)[0].tolist()

    def best_first_all() -> list[tuple]:
        """Best-first enumeration with a certified residual for oversized primaries."""
        return [
            _best_first(
                i,
                k,
                counts,
                starts,
                cand_ld,
                cand_lrho,
                log_nu,
                log_pp,
                log_pa,
                weights,
                mass_tol,
                budget,
                dev,
            )
            for i in big
        ]

    def gather(bf: list[tuple]) -> tuple[Tensor, Tensor, Tensor]:
        """Hypothesis slots, primaries and positional log weights, by primary."""
        sl = torch.cat([slots_exact] + [b[0] for b in bf])
        pr = torch.cat([prim_e] + [b[1] for b in bf])
        geo = torch.cat([geo_exact] + [geo_weights(b[1], b[0]) for b in bf])
        order_h = torch.argsort(pr, stable=True)
        return sl[order_h], pr[order_h], geo[order_h]

    def posterior_of(pr: Tensor, sl: Tensor, lw: Tensor):
        u = lw
        if exclusive and k > 1:
            u = lw + _nway_exclusive_rho(pr, lw, sl, pairs, cats, n0, k)
        mx_ = torch.full((n0,), -math.inf, dtype=_F64, device=dev).scatter_reduce(
            0, pr, u, "amax"
        )
        z_ = torch.zeros(n0, dtype=_F64, device=dev).index_add(
            0, pr, torch.exp(u - mx_[pr])
        )
        return torch.exp(u - mx_[pr]) / z_[pr], mx_, z_

    bf = best_first_all()
    slots, prim, geo = gather(bf)
    if any(term_sets[1:]):
        # EM for the photometric terms from the hypothesis posteriors: each
        # candidate's counterpart weight is the posterior mass of the
        # hypotheses holding it. Completeness and densities stay at the
        # pairwise fits, which learnt the terms pairwise.
        prev = None
        for _ in range(max_iter):
            post, _, _ = posterior_of(prim, slots, geo + term_weights(slots))
            if prev is not None and (
                post.numel() == 0 or float((post - prev).abs().max()) < tol
            ):
                break
            for c in range(1, k):
                if not term_sets[c]:
                    continue
                s = slots[:, c - 1]
                on = s >= 0
                mass = torch.zeros(
                    int(pairs[c][0].numel()), dtype=_F64, device=dev
                ).index_add(0, s[on], post[on])
                for term in term_sets[c]:
                    term.update(pairs[c][0], pairs[c][1], mass)
                cand_lrho[c] = slot_log_rho(c)
            prev = post.detach()
        if big:
            # Re-enumerate with the final terms so the residual is certified
            # against the weights actually reported.
            bf = best_first_all()
            slots, prim, geo = gather(bf)
    for i, b in zip(big, bf):
        residual[i] = b[3]
        log_remaining[i] = b[4]
    logw = geo + term_weights(slots)
    members = torch.full((int(prim.numel()), k), -1, dtype=torch.int64, device=dev)
    members[:, 0] = prim
    for c in range(1, k):
        s = slots[:, c - 1]
        members[:, c] = torch.where(
            s >= 0, pairs[c][1][s.clamp(min=0)], torch.full_like(s, -1)
        )
    post, mx, z = posterior_of(prim, slots, logw)
    if exclusive and bool(torch.isfinite(log_remaining).any()):
        # Certified residual against the exclusive weights (rho <= 1 only lowers
        # the unenumerated mass, so the bound still holds).
        rem = torch.exp(log_remaining - mx)
        residual = torch.where(
            torch.isfinite(log_remaining), rem / (z + rem), torch.zeros_like(rem)
        )
    none = (members[:, 1:] < 0).all(-1)
    p_none = torch.zeros(n0, dtype=_F64, device=dev).index_add(
        0, prim[none], post[none]
    )
    offsets = torch.zeros(n0 + 1, dtype=torch.int64, device=dev)
    offsets[1:] = torch.cumsum(torch.bincount(prim, minlength=n0), 0)
    best_post = torch.full((n0,), -1.0, dtype=_F64, device=dev).scatter_reduce(
        0, prim, post, "amax"
    )
    is_best = post == best_post[prim]
    idx = torch.arange(int(prim.numel()), device=dev)
    best = _scatter_reduce_int(
        torch.full((n0,), int(prim.numel()), dtype=torch.int64, device=dev),
        0,
        prim[is_best],
        idx[is_best],
        "amin",
    )
    back = torch.tensor(order, device=dev)
    inv = torch.argsort(back)
    return NWayAssociation(
        members=members[:, inv],
        offsets=offsets,
        log_weight=logw,
        posterior=post,
        p_any=1.0 - p_none,
        residual_mass=residual,
        completeness=[pi[c] / (1.0 - gate_pvalue) if c else 1.0 for c in range(k)],
        best=best,
    )


def _nway_exclusive_rho(
    prim: Tensor,
    logw: Tensor,
    slots: Tensor,
    pairs: list[tuple[Tensor, Tensor]],
    cats,
    n0: int,
    k: int,
) -> Tensor:
    """Sum of exclusivity log messages over each hypothesis' members."""
    dev = prim.device
    n_pairs = [int(pairs[c][0].numel()) for c in range(1, k)]
    pair_off = [0]
    for n in n_pairs:
        pair_off.append(pair_off[-1] + n)
    src_off = [0]
    for c in range(1, k):
        src_off.append(src_off[-1] + int(cats[c].lon.numel()))
    n_present = pair_off[-1]
    cat_ids = torch.arange(k - 1, device=dev)
    # Present slots (catalogue c, pair p), then one absent slot per (primary, c).
    slot_owner = torch.cat(
        [pairs[c][0] for c in range(1, k)]
        + [torch.arange(n0, device=dev).repeat_interleave(k - 1)]
    )
    slot_cat = torch.cat(
        [
            torch.full((n_pairs[c - 1],), c - 1, dtype=torch.int64, device=dev)
            for c in range(1, k)
        ]
        + [cat_ids.repeat(n0)]
    )
    slot_group = slot_owner * (k - 1) + slot_cat
    slot_source = torch.cat(
        [pairs[c][1] + src_off[c - 1] for c in range(1, k)]
        + [torch.full((n0 * (k - 1),), -1, dtype=torch.int64, device=dev)]
    )
    off = torch.tensor(pair_off[:-1], dtype=torch.int64, device=dev)
    absent = n_present + prim[:, None] * (k - 1) + cat_ids[None, :]
    hyp_slot = torch.where(slots >= 0, slots + off[None, :], absent)
    rho = exclusive_log_rho(
        prim,
        logw,
        hyp_slot,
        slot_owner,
        slot_group,
        slot_source,
    )
    return rho[hyp_slot].sum(-1)


def _cov_only(pos_cov: Tensor) -> Astrometry:
    full = torch.full(
        (pos_cov.shape[0], 6, 6), math.nan, dtype=_F64, device=pos_cov.device
    )
    full[:, :2, :2] = pos_cov
    return Astrometry(full)


def _best_first(
    i,
    k,
    counts,
    starts,
    cand_ld,
    cand_lrho,
    log_nu,
    log_pp,
    log_pa,
    weights,
    mass_tol,
    budget,
    dev,
):
    """Enumerate one primary's hypotheses by decreasing upper bound.

    Bound per catalogue choice: absent -> (1 - pi_c); candidate j ->
    pi_c rho_j / nu_j / (2 pi sqrt(det C_j)), with ``rho_j`` the candidate's
    photometric factor (exact: a term multiplies its member's factor alone,
    so its histogram ratio at ``j`` is its own bound). The product of the
    chosen factors bounds
    the exact weight relative to the primary-only hypothesis. The hypotheses
    enumerated in bound order stop when the remaining bound mass (closed form
    total minus enumerated bounds) is below ``mass_tol`` of the enumerated
    exact mass.
    """
    import heapq

    options: list[list[tuple[float, int]]] = []
    for c in range(1, k):
        lo = int(starts[c - 1][i])
        cnt = int(counts[c - 1][i])
        slots = torch.arange(lo, lo + cnt, device=dev)
        lb_t = (
            log_pp[c] - log_nu[c][i] - math.log(2.0 * math.pi) - 0.5 * cand_ld[c][slots]
        )
        if cand_lrho[c].numel():
            lb_t = lb_t + cand_lrho[c][slots]
        lb = lb_t.tolist()
        opts = [(float(log_pa[c]), -1)] + [
            (b, int(s)) for b, s in zip(lb, slots.tolist())
        ]
        opts.sort(key=lambda t: -t[0])
        options.append(opts)
    log_total = sum(math.log(sum(math.exp(b) for b, _ in o)) for o in options)
    start = tuple(0 for _ in options)
    heap = [(-sum(o[0][0] for o in options), start)]
    seen = {start}
    chosen, bounds = [], []
    exact_mass = 0.0
    shift = None
    while heap:
        nb, combo = heapq.heappop(heap)
        chosen.append([options[c][j][1] for c, j in enumerate(combo)])
        bounds.append(-nb)
        if len(chosen) % 64 == 0 or not heap:
            sl = torch.tensor(chosen, dtype=torch.int64, device=dev)
            w = weights(
                torch.full((len(chosen),), i, dtype=torch.int64, device=dev), sl
            )
            shift = float(w.max())
            exact_mass = float(torch.exp(w - shift).sum())
            enumerated_bound = math.fsum(math.exp(b - shift) for b in bounds)
            remaining = max(math.exp(log_total - shift) - enumerated_bound, 0.0)
            if remaining <= mass_tol * exact_mass or len(chosen) >= 64 * budget:
                break
        for c in range(len(options)):
            nxt = list(combo)
            nxt[c] += 1
            if nxt[c] < len(options[c]):
                t = tuple(nxt)
                if t not in seen:
                    seen.add(t)
                    heapq.heappush(
                        heap, (-sum(options[cc][jj][0] for cc, jj in enumerate(t)), t)
                    )
    sl = torch.tensor(chosen, dtype=torch.int64, device=dev)
    w = weights(torch.full((len(chosen),), i, dtype=torch.int64, device=dev), sl)
    shift = float(w.max())
    exact_mass = float(torch.exp(w - shift).sum())
    remaining = max(
        math.exp(log_total - shift) - math.fsum(math.exp(b - shift) for b in bounds),
        0.0,
    )
    residual = remaining / (exact_mass + remaining)
    log_rem = math.log(remaining) + shift if remaining > 0 else -math.inf
    return (
        sl,
        torch.full((len(chosen),), i, dtype=torch.int64, device=dev),
        w,
        residual,
        log_rem,
    )


__all__ = [
    "Association",
    "HistogramRatio",
    "NWayAssociation",
    "assign_one_to_one",
    "associate",
    "associate_many",
    "gate_k2",
    "mahalanobis2",
    "pair_frame",
]
