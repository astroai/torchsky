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

from ..device import _as_tensor
from ..sphere.moc import MOC
from ..sphere import healpix as _healpix
from ._zones import (
    ZoneIndex,
    default_zone_height_deg,
    occupied_spacing_deg,
    unit_vectors,
    zone_query,
)
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
    right_scale: float | Tensor = 1.0
    floor_mas2: Tensor | None = None


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


def _fit_calibration(
    delta: Tensor,
    cl: Tensor,
    cr: Tensor,
    weight: Tensor,
    fit_scale: bool,
    scale: float,
    floor: Tensor,
) -> tuple[float, Tensor]:
    """Posterior-weighted ML of ``Sigma = C_l + s^2 C_r + F`` over matched pairs.

    ``F = L L^T`` (2x2, PSD) is a systematic floor in the local tangent frame,
    ``s`` scales the right catalogue's published errors. L-BFGS on
    ``(log s, log L11, L21, log L22)``.
    """
    w = weight.detach()
    if float(w.sum()) <= 0:
        return scale, floor
    d, a, b = delta.detach(), cl.detach(), cr.detach()
    # Start the floor at a tenth of the typical pair variance (a log-parametrised
    # floor at ~0 has no gradient); a true zero floor is then driven down.
    eye = torch.eye(2, dtype=_F64, device=d.device)
    typical = float(torch.median(0.5 * (a + b).diagonal(dim1=-2, dim2=-1).sum(-1)))
    start = floor if float(floor.abs().max()) > 1e-3 * typical else 0.1 * typical * eye
    chol = torch.linalg.cholesky(start + 1e-9 * typical * eye)
    params = torch.tensor(
        [
            math.log(scale),
            math.log(float(chol[0, 0])),
            float(chol[1, 0]),
            math.log(float(chol[1, 1])),
        ],
        dtype=_F64,
        device=d.device,
        requires_grad=True,
    )
    opt = torch.optim.LBFGS(
        [params], max_iter=200, line_search_fn="strong_wolfe", tolerance_grad=1e-10
    )

    def closure():
        opt.zero_grad()
        log_s = params[0] if fit_scale else params[0].detach() * 0.0 + math.log(scale)
        lmat = torch.stack(
            (
                torch.stack((torch.exp(params[1]), params[1] * 0.0)),
                torch.stack((params[2], torch.exp(params[3]))),
            )
        )
        sig = a + torch.exp(2.0 * log_s) * b + lmat @ lmat.mT
        d2, logdet = mahalanobis2(d, sig)
        loss = (w * (0.5 * d2 + 0.5 * logdet)).sum() / w.sum()
        loss.backward()
        return loss

    try:
        opt.step(closure)
    except RuntimeError:  # line search failure: keep the previous estimate
        return scale, floor
    with torch.no_grad():
        p = params.detach()
        # Physical bounds: scale within e^+-8, floor at most 100x the typical pair
        # variance. Divergent fits (outside them) keep the previous estimate.
        bound_floor = 0.5 * math.log(100.0 * typical)
        ok = (
            bool(torch.isfinite(p).all())
            and abs(float(p[0])) <= 8.0
            and max(float(p[1]), float(p[3])) <= bound_floor
            and abs(float(p[2])) <= math.exp(bound_floor)
        )
        if not ok:
            return scale, floor
        s_new = math.exp(float(params[0])) if fit_scale else scale
        lmat = torch.tensor(
            [
                [math.exp(float(params[1])), 0.0],
                [float(params[2]), math.exp(float(params[3]))],
            ],
            dtype=_F64,
            device=d.device,
        )
        return s_new, lmat @ lmat.T


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
    catalogue's errors by ``s``. Both are posterior-weighted maximum likelihood,
    alternated with the association until stable, and reported on the
    result. Search radii include the fitted terms, so the gate stays sound.
    ``calibrate_by`` (one value per right row, e.g. magnitude, or a blending
    count from :func:`count_within`) with ``calibrate_edges`` fits the
    terms per bin. Bins whose matched weight is below ``calibrate_min_pairs``
    use the global fit. Scale and floor are separately identifiable only
    when the right errors vary within a bin.

    ``right_footprint`` (MOC) defaults to the right catalogue's occupied cells at
    ``footprint_order``. Pass the survey MOC when you have it, so that density
    and priors use the true coverage.
    """
    if left.astrometry is None or right.astrometry is None:
        raise ValueError("associate needs Catalog.astrometry on both catalogues")
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
    nu = _density(rlon, rlat, llon, llat, rfoot, density, density_order)
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
    rounds = 1 if calibrate == "none" else 8
    for _round in range(rounds):
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
            sub = left.to_state()
            gathered = type(sub)(
                lon=sub.lon[li],
                lat=sub.lat[li],
                frame=sub.frame,
                epoch=torch.broadcast_to(
                    torch.as_tensor(sub.epoch, dtype=_F64), sub.lon.shape
                )[li],
                epoch_scale=sub.epoch_scale,
                epoch_format=sub.epoch_format,
                pm_lon_coslat_mas_yr=None
                if sub.pm_lon_coslat_mas_yr is None
                else sub.pm_lon_coslat_mas_yr[li],
                pm_lat_mas_yr=None
                if sub.pm_lat_mas_yr is None
                else sub.pm_lat_mas_yr[li],
                parallax_mas=None if sub.parallax_mas is None else sub.parallax_mas[li],
                radial_velocity_km_s=None
                if sub.radial_velocity_km_s is None
                else sub.radial_velocity_km_s[li],
            )
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
        li, rj, d2, logdet, sep_arcsec = (
            li[keep],
            rj[keep],
            d2[keep],
            logdet[keep],
            sep_arcsec[keep],
        )
        delta, cl, cr = delta[keep], cl[keep], cr[keep]
        base_log_lr = (
            -math.log(2.0 * math.pi)
            - 0.5 * logdet
            - 0.5 * d2
            - torch.log(nu[li].clamp_min(1e-300))
        )

        def posterior(pi_c: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
            log_lr = base_log_lr.clone()
            for term in terms:
                log_lr = log_lr + term.log_rho(li, rj)
            pi = (pi_c * (1.0 - gate_pvalue)) * inside
            # P_ij = pi LR / ((1 - pi) + pi sum LR), in log space per left segment.
            lse_max = torch.full(
                (n_left,), -math.inf, dtype=_F64, device=dev
            ).scatter_reduce(0, li, log_lr, "amax", include_self=True)
            shift = torch.where(
                torch.isfinite(lse_max), lse_max, torch.zeros_like(lse_max)
            )
            s_ = torch.zeros(n_left, dtype=_F64, device=dev).index_add(
                0, li, torch.exp(log_lr - shift[li])
            )
            denom = (1.0 - pi) + pi * s_ * torch.exp(shift)
            return (
                pi[li] * torch.exp(log_lr) / denom[li],
                (1.0 - pi) / denom,
                pi,
                log_lr,
            )

        fit_pi = completeness == "em"
        if not fit_pi and not 0.0 <= float(completeness) <= 1.0:  # type: ignore[arg-type]
            raise ValueError("completeness must lie in [0, 1] or be 'em'")
        pi_c = 0.5 if fit_pi else float(completeness)  # type: ignore[arg-type]
        for term in terms:
            term.setup(rfoot.contains(rlon, rlat), li, rj)
        if fit_pi or terms:
            # EM: completeness and photometric terms from posterior responsibilities.
            prev = None
            for _ in range(max_iter):
                post, p_none, _, _ = posterior(pi_c)
                for term in terms:
                    term.update(li, rj, post)
                done = prev is not None and (
                    post.numel() == 0 or float((post - prev).abs().max()) < tol
                )
                prev = post
                if fit_pi:
                    in_rows = inside > 0
                    new = (
                        float((1.0 - p_none[in_rows]).mean()) / (1.0 - gate_pvalue)
                        if bool(in_rows.any())
                        else 0.0
                    )
                    new = min(max(new, 1e-6), 1.0 - 1e-9)
                    done = done and abs(new - pi_c) < tol
                    pi_c = new
                if done:
                    break
        post, p_none, pi, log_lr = posterior(pi_c)
        if calibrate == "none":
            break
        pb = rbin[rj]
        fit_scale = calibrate == "scale_floor"
        g_scale, g_floor = _fit_calibration(
            delta, cl, cr, post, fit_scale, float(scale.mean()), floor.mean(0)
        )
        new_scale = torch.full_like(scale, g_scale)
        new_floor = g_floor.expand_as(floor).clone()
        if n_bins > 1:
            for b in range(n_bins):
                sel = pb == b
                if float(post[sel].sum()) < calibrate_min_pairs:
                    continue  # too few matches: the global fit applies
                sb, fb = _fit_calibration(
                    delta[sel],
                    cl[sel],
                    cr[sel],
                    post[sel],
                    fit_scale,
                    float(scale[b]),
                    floor[b],
                )
                new_scale[b], new_floor[b] = sb, fb
        moved_by = float((new_scale - scale).abs().max()) + float(
            (new_floor - floor).abs().max()
        ) / max(1.0, float(new_floor.abs().max()))
        scale, floor = new_scale, new_floor
        if moved_by < 1e-4:
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
    budget: int = 4096,
    mass_tol: float = 1e-6,
) -> NWayAssociation:
    """N-way association (nway-style) of a primary catalogue with several others.

    For each primary row, the hypotheses are every combination of, per secondary
    catalogue, one counterpart that passes the pairwise gate against the primary,
    or none. A hypothesis with present set ``S`` has weight

    ``prod_{c in S} pi_c / nu_c * prod_{c not in S} (1 - pi_c) * M_S``

    where ``M_S`` is the exact Gaussian marginal likelihood that all present
    positions (primary included) come from one true position with a flat prior.
    This is Budavari & Szalay (2008) eq. 18 for isotropic errors, here with full
    covariances in the primary's tangent plane. ``pi_c`` is catalogue ``c``'s
    completeness (pairwise EM when ``"em"``, also counting gate losses);
    ``nu_c`` is its surface density at the primary over its footprint.

    Primaries with at most ``budget`` hypotheses are enumerated exactly.
    Larger ones are enumerated best-first by an upper bound (``M`` gains at most
    ``1 / (2 pi sqrt(det C))`` per added member) until the unenumerated mass,
    certified through that bound, is below ``mass_tol``. The certified residual
    is reported per primary.
    """
    if len(catalogs) < 2:
        raise ValueError("need a primary and at least one other catalogue")
    order = [primary] + [i for i in range(len(catalogs)) if i != primary]
    cats = [catalogs[i] for i in order]
    feet: list[MOC | None] = (
        [None] * len(cats) if footprints is None else [footprints[i] for i in order]
    )
    k = len(cats)
    ref = cats[0]
    if epoch is None:
        ep = torch.as_tensor(ref.epoch)
        if ep.numel() != 1:
            raise ValueError("the primary has per-row epochs; pass a scalar epoch")
        epoch = ep.reshape(())
    pos_lonlat, covs = [], []
    for cat in cats:
        cat = cat if cat.frame == ref.frame else cat.to_frame(ref.frame)
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
        res = associate(
            left,
            right,
            gate_pvalue=gate_pvalue,
            max_radius_arcsec=max_radius_arcsec,
            epoch=None,
            right_footprint=feet[c],
            footprint_order=footprint_order,
            completeness=comp,
            assign="none",
        )
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
        nu = _density(rlon_c, rlat_c, llon0, llat0, rfoot, "global", None)
        inside = rfoot.contains(llon0, llat0)
        log_nu.append(
            torch.where(
                inside, torch.log(nu.clamp_min(1e-300)), torch.full_like(nu, math.inf)
            )
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
    members_out, prim_out, logw_out = [], [], []
    residual = torch.zeros(n0, dtype=_F64, device=dev)

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

    def weights(prim: Tensor, slots: Tensor) -> Tensor:
        """``slots [H, K-1]`` candidate slot per catalogue (-1 absent)."""
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

    # Exact enumeration, vectorised: mixed-radix decode of hypothesis ids.
    prim_exact = torch.nonzero(exact, as_tuple=True)[0]
    if prim_exact.numel():
        nh = n_hyp[prim_exact]
        prim = torch.repeat_interleave(prim_exact, nh)
        local = torch.arange(int(nh.sum()), device=dev) - torch.repeat_interleave(
            torch.cumsum(nh, 0) - nh, nh
        )
        slots = torch.empty(int(prim.numel()), k - 1, dtype=torch.int64, device=dev)
        rem = local
        for c in range(1, k):
            base = counts[c - 1][prim] + 1
            choice = rem % base
            rem = rem // base
            slots[:, c - 1] = torch.where(
                choice == 0,
                torch.full_like(choice, -1),
                starts[c - 1][prim] + choice - 1,
            )
        members_out.append(slots)
        prim_out.append(prim)
        logw_out.append(weights(prim, slots))
    # Best-first enumeration with a certified residual for oversized primaries.
    for i in torch.nonzero(~exact, as_tuple=True)[0].tolist():
        s, p, w, r = _best_first(
            i,
            k,
            counts,
            starts,
            cand_ld,
            log_nu,
            pairs,
            log_pp,
            log_pa,
            weights,
            mass_tol,
            budget,
            dev,
        )
        members_out.append(s)
        prim_out.append(p)
        logw_out.append(w)
        residual[i] = r
    slots = torch.cat(members_out)
    prim = torch.cat(prim_out)
    logw = torch.cat(logw_out)
    order_h = torch.argsort(prim, stable=True)
    slots, prim, logw = slots[order_h], prim[order_h], logw[order_h]
    members = torch.full((int(prim.numel()), k), -1, dtype=torch.int64, device=dev)
    members[:, 0] = prim
    for c in range(1, k):
        s = slots[:, c - 1]
        members[:, c] = torch.where(
            s >= 0, pairs[c][1][s.clamp(min=0)], torch.full_like(s, -1)
        )
    mx = torch.full((n0,), -math.inf, dtype=_F64, device=dev).scatter_reduce(
        0, prim, logw, "amax"
    )
    z = torch.zeros(n0, dtype=_F64, device=dev).index_add(
        0, prim, torch.exp(logw - mx[prim])
    )
    post = torch.exp(logw - mx[prim]) / z[prim]
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
    best = torch.full(
        (n0,), int(prim.numel()), dtype=torch.int64, device=dev
    ).scatter_reduce(0, prim[is_best], idx[is_best], "amin")
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
    log_nu,
    pairs,
    log_pp,
    log_pa,
    weights,
    mass_tol,
    budget,
    dev,
):
    """Enumerate one primary's hypotheses by decreasing upper bound.

    Bound per catalogue choice: absent -> (1 - pi_c); candidate j ->
    pi_c / nu_j / (2 pi sqrt(det C_j)). The product of the chosen factors bounds
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
        lb = (
            log_pp[c] - log_nu[c][i] - math.log(2.0 * math.pi) - 0.5 * cand_ld[c][slots]
        ).tolist()
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
    return sl, torch.full((len(chosen),), i, dtype=torch.int64, device=dev), w, residual


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
