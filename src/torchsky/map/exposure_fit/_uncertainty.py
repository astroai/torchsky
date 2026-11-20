"""Uncertainty of the fitted sky (moved from the research prototype, numerics unchanged; the prototype report is UNCERTAINTY_2026-11-18.md).

Runs AFTER a fit; nothing here changes the fit. ``SkyFit.product`` is the public entry point; the command line, the FITS writer and the file-based
difference of the prototype were left out (core code opens no files). Wording below that mentions ``--unc-*`` options refers to the prototype's command
line; the equivalent here is an argument of ``SkyFit.product``.


Product. For a band with spectral weights phi (a filter's bandpass-averaged basis, model.Phi_band[f], or a unit vector for one spectral
coefficient), the sky is s_phi(x) = sum_j (phi . A_j) T(x - x_j) (A the coefficient grid, T the bilinear tent at the sky pitch). The product at
resolution s is the continuous sky convolved with a unit-integral circular Gaussian g_s, evaluated on an output grid:
    I_s(x) = sum_j (phi . A_j) (T * g_s)(x - x_j)          (separable; weights sum to 1, so a constant sky is unchanged)

Analytic estimator (linearised, Gauss-Newton posterior). The model of the training pixels is linear in A and in the per-exposure background
coefficients b (6 Legendre terms per exposure) once the response, registration, gain and noise terms are held at their fitted values:
    m = J_A A + L b + (terms held fixed),   weights W = diag(w_i),  w_i = min(1, delta / |z_i|) / sigma_i^2  (Huber IRLS weight at the fit)
    F = [J_A L]^T W [J_A L] + diag(mu I, 0)   restricted to the background gauge (per filter, the training mean of each of the 6
                                               background terms held; see SkyNormalOperator)
    Cov(I_s(x)) = k_x^T [F^{-1}]_AA k_x,  k_x = phi (x) (T * g_s)(x - .)
F^{-1} is never formed. Two ways, both by preconditioned conjugate gradients with the model's own renders (J_A v by a forward render with the
sky coefficients replaced by v, J_A^T r by the backward pass of that render):
  * samples: x = F^{-1} ([J_A L]^T W^{1/2} z + [sqrt(mu) z', 0]), z, z' standard normal, has covariance F^{-1} exactly ("perturb and solve");
    smoothing the A part of n such draws at any s gives sigma_s(x) for every output pixel and scale (relative error 1 / sqrt(2 n)).
  * exact: one solve F x = k_x per requested pixel and scale (sigma^2 = k_x . x_A), for a subsample of pixels (cross-check of the samples).
mu = (median node information) / rho^2 is a weak Gaussian prior on each coefficient (rho = --unc-prior, default 10 node-noise units); it only
matters at scales finer than the supported resolution, where the data do not constrain the sky (checked in the report).

Supported resolution. alpha(s, x) = -d ln sigma_s(x) / d ln s. For white noise alpha = 1 (the noise per element falls as its linear size);
for a sky that is a deconvolution, alpha > 1 at small s. For a Gaussian response of width p, sigma_s^2 is proportional to 1 / (s^2 - p^2) and
alpha = s^2 / (s^2 - p^2) = the variance of the element divided by the variance it would have without the blur. The supported resolution
s*(x) is the smallest s with alpha(s, x) <= 2 (for a Gaussian response s* = sqrt(2) p).

Empirical estimators. h = (I_s[fit on half a] - I_s[fit on half b]) / 2 for two fits on disjoint halves of the training exposures (alternate
dither groups, --train-half 0 / 1): its variance equals the full fit's if the noise variance scales as 1 / (number of exposures), and biases
common to both halves cancel. For two fits that differ only by the seed, sigma_opt = rms(I_a - I_b) / sqrt(2).
"""

from __future__ import annotations

import contextlib
import math
import time
from typing import Literal, overload

import numpy as np
import torch

from ...device import _to

from ._response import legendre6


def _silent(msg):
    """Default log callable: discards the line."""


# ============================================================================= product: band vector, smoothing operator
SCALES_DEFAULT = (
    0.0,
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    2.0,
    2.5,
    3.0,
    4.0,
    6.0,
)  # in units of the sky pitch


def band_vector(model, band):
    """Spectral weights phi (K,): an int or a filter name -> model.Phi_band[f]; 'coef:k' -> unit vector on spectral coefficient k."""
    K = model.sky.K
    if isinstance(band, str) and band.startswith("coef:"):
        v = torch.zeros(K, device=model.Phi_band.device)
        v[int(band.split(":")[1])] = 1.0
        return v
    if isinstance(band, str):
        band = (
            model.fset.names.index(band) if hasattr(model.fset, "names") else int(band)
        )
    return model.Phi_band[int(band)].detach().clone()


def tent_gauss_1d(x_out, x_nodes, pitch, s, n_quad=48):
    """(n_out, N) matrix of (T * g_s)(x_out - x_j): tent of half-width pitch convolved with a unit Gaussian of width s (s = 0: the tent)."""
    x_out = torch.as_tensor(x_out, dtype=torch.float64)
    x_nodes = torch.as_tensor(x_nodes, dtype=torch.float64)
    u = x_out[:, None] - x_nodes[None]
    if s <= 0:
        return (1 - u.abs() / pitch).clamp(min=0)
    t, wq = np.polynomial.legendre.leggauss(
        n_quad
    )  # integral over the tent support [-p, p] in two halves
    out = torch.zeros_like(u)
    for a, b in ((-pitch, 0.0), (0.0, pitch)):
        tt = torch.tensor(0.5 * (b - a) * t + 0.5 * (b + a))
        ww = torch.tensor(0.5 * (b - a) * wq)
        tent = 1 - tt.abs() / pitch
        g = torch.exp(-0.5 * ((u[..., None] - tt) / s) ** 2) / (
            math.sqrt(2 * math.pi) * s
        )
        out += (g * (tent * ww)).sum(-1)
    return out  # sum over nodes = 1 (the tent is a partition of unity)


class Smoother:
    """Separable linear map from a node image (N, N) to the product at resolution s on an output axis (default: the node positions)."""

    def __init__(self, N, pitch, s, x_out=None, dtype=torch.float32, device="cpu"):
        x_nodes = (torch.arange(N, dtype=torch.float64) - (N - 1) / 2) * pitch
        self.x_out = (
            x_nodes if x_out is None else torch.as_tensor(x_out, dtype=torch.float64)
        )
        self.s, self.pitch = float(s), float(pitch)
        self.M = _to(
            tent_gauss_1d(self.x_out, x_nodes, pitch, s), device=device, dtype=dtype
        )  # (n_out, N)

    def __call__(self, img):
        """img (..., N, N) rows = Y, columns = X -> (..., n_out, n_out)."""
        M = self.M.to(img.dtype)
        return M @ img @ M.T

    def weights_at(self, ix, iy):
        """Node weights (N, N) of output pixel (row iy, column ix): k = M[iy] (outer) M[ix]."""
        return self.M[iy][:, None] * self.M[ix][None, :]


# ============================================================================= the linear operator of the fit
@contextlib.contextmanager
def frozen(model):
    """Every model parameter without gradient while the operator renders (only the sky coefficients passed in carry one)."""
    flags = [(p, p.requires_grad) for p in model.parameters()]
    for p, _ in flags:
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, f in flags:
            p.requires_grad_(f)


@contextlib.contextmanager
def sky_override(sky, A):
    """Render with the coefficient grid A in place of the fitted sky (instance attribute shadows the method; removed on exit)."""
    sky.coeffs = lambda view=None: A
    try:
        yield
    finally:
        del sky.coeffs


def default_tile(e, cfg):
    J = np.asarray(e.extra.get("J", [[cfg.pix_default, 0], [0, cfg.pix_default]]))
    return max(
        32, int(round(cfg.patch_arcsec / max(abs(np.linalg.det(J)) ** 0.5, 1e-6)))
    )


DEFAULT_BATCH_PX = 16000  # pixels per batched render on a CUDA device (about the fit's batch: 5 patches of ~2,500 pixels)
DEFAULT_TILE_CUDA = 64  # detector pixels per tile edge on a CUDA device (CPU: --region-px, the fastest per pixel on one thread)
LEGACY_CHUNK = 32  # tiles per batched render when no pixel budget is in force (CPU default, the validated one)


def plan_passes(
    K, method="draws", polish=True, n_samples=8, maxiter=60, n_exact=0, n_bands=1
):
    """Planned cost in applications of F (one application = one forward and one adjoint render of every training pixel; at most: a conjugate-
    gradient solve stops earlier when it reaches --unc-tol). The set-up also has one forward-only pass (the weights), counted separately."""
    setup = K + 1  # coverage (J^T w) + K delta responses of the Fourier preconditioner
    pol = (1 + maxiter) if polish else 0  # right-hand side + iterations
    dr = n_samples * (1 + maxiter) if (method == "draws" and n_samples > 0) else 0
    ex = (
        n_exact * n_bands * 41
    )  # typical (measured on the truth problem); the cap is 600 per solve
    return {
        "setup": setup,
        "forward_only": 1,
        "polish": pol,
        "draws": dr,
        "exact": ex,
        "total": setup + pol + dr + ex,
    }


def resolve_batching(device_type, region_px, tile=None, chunk=None, batch_px=None):
    """(tile, batch_px, chunk). Explicit tile / chunk win. CUDA: tile 64 and a budget of 16000 pixels per batched render unless given. CPU: tile =
    region_px and 32 tiles per render (the validated default) unless a budget or a chunk is given."""
    cuda = device_type == "cuda"
    if batch_px is None and not chunk and cuda:
        batch_px = DEFAULT_BATCH_PX
    tile = tile or (DEFAULT_TILE_CUDA if cuda else region_px)
    batch_px = None if chunk else batch_px
    return tile, batch_px, (chunk or (None if batch_px else LEGACY_CHUNK))


def pack_batches(sizes, batch_px=None, chunk=None):
    """Indices of tiles grouped into batched renders: fixed number of tiles (chunk), or greedily up to batch_px pixels (a tile larger than the
    budget goes alone). Order is kept."""
    out, cur, s = [], [], 0
    for i, n in enumerate(sizes):
        if chunk:
            if len(cur) == chunk:
                out.append(cur)
                cur = []
        elif cur and s + n > batch_px:
            out.append(cur)
            cur, s = [], 0
        cur.append(i)
        s += n
    if cur:
        out.append(cur)
    return out


class Tile:
    """One detector tile of a training exposure. Holds ONLY what cannot be recomputed cheaply: the exposure, the slice, the validity mask
    (bool) and the weights w and w r (float32): about 9 bytes per pixel. The pixel dictionary (positions, data, 10 more arrays) and the
    background basis are rebuilt for each batch and freed after it (keys 'pix' and 'Lb' build them on access, for the tests)."""

    __slots__ = ("op", "eid", "row", "e", "sl", "n", "mask", "w", "wr")

    def __init__(self, op, e, sl, mask):
        self.op, self.e, self.sl, self.mask = op, e, sl, mask
        self.eid, self.row = int(e.eid), op.row_of[int(e.eid)]
        self.n = (sl[1] - sl[0]) * (sl[3] - sl[2])
        self.w = self.wr = None

    def __getitem__(self, k):
        if k == "pix":
            return self.e.pix(self.sl)
        if k == "Lb":
            return self.op._lb(self, self.e.pix(self.sl))
        return getattr(self, k) if k != "e" else self.e


class Monitor:
    """Progress lines and the time budget of a run. stop_now() is asked before every application of F: True when the application (estimated
    from the last one, plus 15 per cent and the reserve for writing the outputs) would end after the deadline. Once it said True, .hit is set."""

    def __init__(
        self,
        op,
        log=_silent,
        max_minutes=0.0,
        t0=None,
        planned_apps=None,
        every=5,
        reserve=60.0,
        clock=time.perf_counter,
    ):
        self.op, self.log, self.every, self.reserve, self.clock = (
            op,
            log,
            every,
            reserve,
            clock,
        )
        self.t0 = clock() if t0 is None else t0
        self.deadline = (
            self.t0 + 60.0 * max_minutes if max_minutes and max_minutes > 0 else None
        )
        self.planned = planned_apps
        self.hit, self.where, self.residual = False, "", float("nan")

    def elapsed(self):
        return self.clock() - self.t0

    def eta(self):
        if not self.planned or self.op.n_apply == 0:
            return float("nan")
        return (
            max(self.planned - self.op.n_apply, 0) * self.op.t_apply / self.op.n_apply
        )

    def stop_now(self, where=""):
        if self.deadline is None:
            return False
        if (
            self.clock() + 1.15 * getattr(self.op, "t_last", 0.0) + self.reserve
            > self.deadline
        ):
            self.hit, self.where = True, where
            return True
        return False

    def report(self, label, it, rel):
        self.residual = rel
        if it and it % self.every == 0:
            self.log(
                f"  {label}: iteration {it}, relative residual {rel:.2e}, {self.op.n_apply} passes so far, elapsed {self.elapsed() / 60:.1f} min, "
                f"estimated time left (planned, upper bound) {self.eta() / 60:.1f} min"
            )

    def tail(self):
        return f"{self.op.n_apply} passes, elapsed {self.elapsed() / 60:.1f} min, estimated time left (upper bound) {self.eta() / 60:.1f} min"


def _eigh_rows(fh, fn, rows=32):
    """Apply fn(block of fh (R, N, K, K)) over blocks of rows (bounds the memory of the batched eigendecomposition)."""
    # On the CPU in every case: the GPU's batched eigensolver refused the real field's batch (cusolver INVALID_VALUE, 18 Nov), and
    # N * N matrices of K x K take seconds on the host. A non-finite entry is reported here rather than as a solver error.
    if not bool(torch.isfinite(fh.real).all() and torch.isfinite(fh.imag).all()):
        raise RuntimeError(
            f"Fourier blocks of the normal operator hold {int((~torch.isfinite(fh.real)).sum())} non-finite entries"
        )
    dev = fh.device
    return torch.cat(
        [fn(fh[r : r + rows].cpu()) for r in range(0, fh.shape[0], rows)]
    ).to(dev)


class SkyNormalOperator:
    """F = [J_A L]^T W [J_A L] + diag(mu, 0) on the flat vector x = [A (K N N), b (n_train_rows x 6)], restricted to a background gauge.

    Gauge. A sky polynomial of order <= 2 over the field is exactly degenerate with the per-exposure quadratic backgrounds (a sky gradient g x
    seen by an exposure dithered by c_e is g x_det + g c_e: a linear plus a constant background term). The fit fixes only the mean constant
    (pedestal gauge); the other five directions are left to the optimiser and the warm start. bkg_gauge="all" (default) holds the training mean
    of each of the 6 terms at its fitted value, so those sky modes are NOT in sigma (listed in the report); "constant" uses the pedestal gauge
    only, and the five degenerate sky modes then get the prior variance 1 / mu (sigma then depends on --unc-prior at every scale).

    Held at the fitted values (not marginalised): response (kernel, throughput, distortion), registration, gain u_e, pixel gain, detector
    pattern, stripes, noise scale and floor (the weights). Marginalised: every training exposure's 6 background coefficients.

    Batching (real-data adaptation). Every render goes through self.batches (lists of tiles): up to batch_px pixels per batched render (default
    16000 on a CUDA device with tiles of 64 detector pixels), or chunk tiles per render when chunk is given; on CPU without either, tiles of
    cfg.region_px and 32 per render (the validated default). Pixel dictionaries, background bases and every render output are built for one batch
    and released after it; what stays on the device is the weights (two float32 and one bool per training pixel) plus fixed-size arrays."""

    def __init__(
        self,
        model,
        exps,
        cfg,
        P=None,
        chunk=None,
        rho=10.0,
        tile=None,
        huber=True,
        precond_mode="fourier",
        bkg_gauge="all",
        log=_silent,
        batch_px=None,
        plan=None,
    ):
        self.model, self.cfg = model, cfg
        self.bkg_gauge = bkg_gauge  # "constant": the fit's pedestal gauge only; "all": the training mean of each of the 6 terms held
        self.log = log
        sky = model.sky
        self.K, self.N = sky.K, sky.N
        self.nA = self.K * self.N * self.N
        dev = model.dev
        tile, self.batch_px, self.chunk = resolve_batching(
            torch.device(dev).type, int(cfg.region_px), tile, chunk, batch_px
        )
        self.tile = tile
        with torch.no_grad():
            self.P = (
                P
                if P is not None
                else {
                    k: (v.detach() if torch.is_tensor(v) else v)
                    for k, v in model.prepare().items()
                }
            )
        tr = [e for e in exps if e.split == "train"]
        self.rows = sorted({int(e.eid) for e in tr})
        self.row_of = {r: i for i, r in enumerate(self.rows)}
        self.nb = 6 * len(self.rows)
        # pedestal gauge groups: per filter, the training rows (sum of their b_e0 fixed)
        gf = {}
        for e in tr:
            gf.setdefault(int(e.filt), set()).add(self.row_of[int(e.eid)])
        self.gauge_groups = (
            [torch.tensor(sorted(v), device=dev) for v in gf.values()]
            if getattr(cfg, "pedestal_gauge", 1)
            else []
        )
        h = sky.half_extent - cfg.pred_margin
        self.tiles = []
        A_fit = sky.coeffs().detach()
        n_pix, n_down = 0, 0
        with torch.no_grad():
            for e in tr:
                ny, nx = e.shape
                for y0 in range(0, ny, tile):
                    for x0 in range(0, nx, tile):
                        sl = (y0, min(ny, y0 + tile), x0, min(nx, x0 + tile))
                        pix = e.pix(sl)
                        inside = ((pix["X"] + e.shift0[0]).abs() < h) & (
                            (pix["Y"] + e.shift0[1]).abs() < h
                        )
                        mask = pix["valid"] & inside
                        k = int(mask.sum())
                        if k == 0:
                            continue
                        n_pix += k
                        self.tiles.append(Tile(self, e, sl, mask))
        self.batches = pack_batches(
            [t.n for t in self.tiles], self.batch_px, self.chunk
        )
        self.n_pix = n_pix
        self.A_fit = A_fit
        self.dev = dev
        self.n_apply = 0
        self.t_apply = 0.0
        self.t_last = 0.0
        self.n_forward = 0
        self.t_forward = 0.0
        K = self.K
        pl = plan_passes(K, **plan) if plan else None
        px_render = sum(t.n for t in self.tiles)
        big = max((sum(self.tiles[i].n for i in b) for b in self.batches), default=0)
        log(
            f"uncertainty plan: {len(self.tiles)} tiles of up to {tile} x {tile} detector pixels, {n_pix} training pixels ({px_render} rendered), "
            f"{len(self.batches)} batches per pass (largest {big} pixels; budget {self.batch_px if self.batch_px else 'none'}, "
            f"{self.chunk if self.chunk else 'by budget'} tiles per batch)"
            + (
                f"; planned passes (one pass = forward + adjoint over all training pixels): set-up {pl['setup']} + 1 forward-only, "
                f"Gauss-Newton step {pl['polish']}, draws {pl['draws']}, exact solves ~{pl['exact']}; total up to {pl['total']}"
                if pl
                else ""
            )
        )
        # weights at the fit (forward only): w, w r per tile, the 6 x 6 background blocks (exact, block-Jacobi)
        Bb = torch.zeros(len(self.rows), 6, 6, device=dev)
        t0 = time.perf_counter()
        with torch.no_grad(), sky_override(sky, A_fit):
            for ch in self._batches():
                pixs = [t["pix"] for t in ch]
                outs = model.render_batch(
                    [(t.eid, p) for t, p in zip(ch, pixs)], self.P
                )
                for t, pix, o in zip(ch, pixs, outs):
                    et = model.et(t.eid)
                    S, m = o["S"], o["m"]
                    sig = model.sigma(et, S, pix["sig"])
                    z = (pix["d"] - m) / sig
                    w = 1.0 / sig**2
                    if huber and cfg.loss == "huber":
                        hw = torch.clamp(
                            cfg.huber_delta / z.abs().clamp(min=1e-12), max=1.0
                        )
                        n_down += int(((hw < 1) & t.mask).sum())
                        w = w * hw
                    t.w = torch.where(t.mask, w, torch.zeros_like(w))
                    t.wr = t.w * (
                        pix["d"] - m
                    )  # weighted residual at the fit (Gauss-Newton step)
                    Lb = self._lb(t, pix)
                    Bb[t.row] += Lb.T @ (t.w[:, None] * Lb)
                del outs, pixs
        self.n_huber = n_down
        self.n_forward += 1
        self.t_forward += time.perf_counter() - t0
        log(
            f"first pass (forward only, the weights): {self.t_forward:.1f} s for {len(self.batches)} batches; a full application "
            f"(forward + adjoint) is expected to take about {1.8 * self.t_forward:.1f} s"
        )
        # preconditioner of the sky block (F_AA + mu): Fourier (circulant) model of F_AA from its response to a delta at the centre node,
        # scaled by the relative coverage d_j = (J^T w)_j / (J^T w)_centre; Jacobi on the nodes with d < 0.05 (edges, uncovered)
        t0 = time.perf_counter()
        self.cov = self._JT_w().sum(0)  # (N, N): sum_k (J_A^T w)
        log(
            f"first full application (forward + adjoint): {self.t_last:.1f} s"
            + (
                f"; planned total up to {pl['total']} passes = about "
                f"{pl['total'] * self.t_last / 3600:.2f} h"
                if pl
                else ""
            )
        )
        c = (self.N - 1) // 2
        cols = []
        for k in range(self.K):
            v = torch.zeros(self.K, self.N, self.N, device=dev)
            v[k, c, c] = 1.0
            cols.append(
                self._data_normal(v.reshape(-1), torch.zeros(self.nb, device=dev))[
                    0
                ].view(self.K, self.N, self.N)
            )
        self.fc = fc = torch.stack(cols, 1)  # (K', K, N, N): F[(k', j), (k, centre)]
        dcen = torch.stack(
            [fc[k, k, c, c] for k in range(self.K)]
        )  # (K,) diagonal entries at the centre
        rel = (self.cov / self.cov[c, c].clamp(min=1e-30)).clamp(min=0)
        self.covered = rel > 1e-3
        dA = dcen[:, None, None] * rel[None]
        dmed = float(
            torch.stack([dA[k][self.covered].median() for k in range(self.K)]).max()
        )
        self.mu = dmed / rho**2
        self.rho = rho
        self.node_noise = 1.0 / math.sqrt(dmed)
        self.dA = dA + self.mu
        self.precond_mode = precond_mode
        self.fmask = rel > 0.05
        self.fscale = torch.where(self.fmask, rel, torch.ones_like(rel)).rsqrt()
        fh = torch.fft.fft2(
            torch.roll(fc, shifts=(-c, -c), dims=(-2, -1))
        )  # (K', K, N, N)
        fh = fh.permute(2, 3, 0, 1)  # (N, N, K', K)

        def _inv(b):
            b = 0.5 * (b + b.conj().transpose(-1, -2))
            ev, V = torch.linalg.eigh(b)
            ev = ev.clamp(min=0) + self.mu
            return (V * (1.0 / ev)[..., None, :]) @ V.conj().transpose(-1, -2)

        self.finv = _eigh_rows(fh, _inv)  # (N, N, K, K)
        del fh
        self._qcache = {}
        self.kappa = (dcen / self.cov[c, c].clamp(min=1e-30)).tolist()
        kap = self.kappa
        self.Bb_inv = torch.linalg.inv(Bb + 1e-9 * torch.eye(6, device=dev))
        self.t_setup_precond = time.perf_counter() - t0
        log(
            f"uncertainty operator: {len(self.tiles)} tiles, {n_pix} training pixels ({n_down} with Huber weight < 1), {len(self.rows)} exposures; "
            f"coefficients {self.nA}, background terms {self.nb}; node noise (median, coefficient units) {self.node_noise:.4g}; prior rho {rho} -> mu {self.mu:.4g}; "
            f"kappa {np.round(kap, 5).tolist()}; preconditioner {self.t_setup_precond:.1f} s ({self.n_apply} applications + {self.n_forward} forward-only)"
        )

    def _batches(self):
        for b in self.batches:
            yield [self.tiles[i] for i in b]

    def _lb(self, t, pix):
        """Background basis of a tile (Legendre terms times the exposure gain), no gradient."""
        with torch.no_grad():
            return legendre6(pix["xi"])

    # ---------------------------------------------------------------- fast approximation (no solve)
    def _circ_q(self, phi):
        """q(freq) = sum over the eigenmodes of the Fourier model of |<v, phi>|^2 / (lambda + mu), double precision, in blocks of rows; cached
        per band vector (the eigendecomposition does not depend on the scale)."""
        key = tuple(float(x) for x in phi.double().cpu().tolist())
        if key not in self._qcache:
            c = (self.N - 1) // 2
            fh = torch.fft.fft2(
                torch.roll(self.fc.double(), (-c, -c), (-2, -1))
            ).permute(2, 3, 0, 1)
            ph = phi.double().to(fh.device)

            def _q(b):
                b = 0.5 * (b + b.conj().transpose(-1, -2))
                ev, V = torch.linalg.eigh(b)
                lam = ev.clamp(min=0) + self.mu
                proj = (
                    V.conj().transpose(-1, -2) @ _to(ph, device=V.device, dtype=V.dtype)
                ).abs() ** 2
                return (proj / lam).sum(-1)

            self._qcache[key] = _eigh_rows(fh, _q)
        return self._qcache[key]

    def circulant_sigma(self, phi, s):
        """sigma of the product at scale s from the Fourier model of F alone (uniform-coverage approximation): sigma_c(s) at the centre's
        coverage, scaled by 1 / sqrt(relative coverage) elsewhere. Costs nothing beyond the preconditioner's K + 1 applications; ignores the
        background terms, the edges and coverage gradients. Returns (N, N) (NaN where the relative coverage is below 0.05)."""
        N, c = self.N, (self.N - 1) // 2
        q = self._circ_q(phi)
        g = Smoother(
            N, self.model.sky.pitch, s, dtype=torch.float64, device=q.device
        ).weights_at(c, c)
        gh = torch.fft.fft2(torch.roll(g, (-c, -c), (0, 1)))
        sc = math.sqrt(float((gh.abs() ** 2 * q).sum() / N**2))
        rel = (self.cov / self.cov[c, c]).double()
        return torch.where(
            self.fmask,
            sc / rel.clamp(min=1e-12).sqrt(),
            torch.full_like(rel, float("nan")),
        ).cpu()

    # ---------------------------------------------------------------- pieces
    def gauge(self, xb):
        """Project the background part onto the gauge (per filter, sum over training rows of b_e0, or of every term, = 0)."""
        xb = xb.view(-1, 6).clone()
        cols = slice(0, 6) if self.bkg_gauge == "all" else slice(0, 1)
        for g in self.gauge_groups:
            xb[g, cols] -= xb[g, cols].mean(0)
        return xb.reshape(-1)

    @overload
    def _render_sum(
        self, xA, rfun, want_gb: Literal[False] = False
    ) -> tuple[torch.Tensor, None]: ...

    @overload
    def _render_sum(
        self, xA, rfun, want_gb: Literal[True]
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def _render_sum(self, xA, rfun, want_gb=False):
        """One application: forward render of every tile with sky coefficients xA (requires grad), batch by batch, and backward of
        sum_i r_i S_i with r = rfun(tile, S detached, Lb). Each batch's graph is freed by its own backward and its tensors are deleted before the
        next batch. Returns J_A^T r (K, N, N) and, if want_gb, L^T r per training row (n_rows, 6) accumulated per batch."""
        model = self.model
        t0 = time.perf_counter()
        xA = xA.detach().requires_grad_(True)
        gb = torch.zeros(len(self.rows), 6, device=self.dev) if want_gb else None
        with torch.enable_grad(), frozen(model), sky_override(model.sky, xA):
            for ch in self._batches():
                pixs = [t["pix"] for t in ch]
                outs = model.render_batch(
                    [(t.eid, p) for t, p in zip(ch, pixs)], self.P
                )
                Ss, Rs = [], []
                for t, p, o in zip(ch, pixs, outs):
                    Lb = self._lb(t, p)
                    r = rfun(t, o["S"].detach(), Lb)
                    Ss.append(o["S"])
                    Rs.append(r)
                    if gb is not None:  # == want_gb
                        gb[t.row] += Lb.T @ r
                torch.autograd.backward(Ss, Rs)
                del outs, Ss, Rs, pixs
        self.n_apply += 1
        self.t_last = time.perf_counter() - t0
        self.t_apply += self.t_last
        g = xA.grad if xA.grad is not None else torch.zeros_like(xA)
        return g.detach(), gb

    def _JT_w(self):
        g, _ = self._render_sum(
            torch.zeros(self.K, self.N, self.N, device=self.dev), lambda t, S, Lb: t.w
        )
        return g

    def _data_normal(self, xA_flat, xb_flat):
        xA = xA_flat.view(self.K, self.N, self.N)
        xb = xb_flat.view(-1, 6)
        gA, gb = self._render_sum(
            xA, lambda t, S, Lb: t.w * (S + Lb @ xb[t.row]), want_gb=True
        )
        return gA.reshape(-1), gb.reshape(-1)

    # ---------------------------------------------------------------- operator, preconditioner, right-hand sides
    def apply(self, x):
        xA, xb = x[: self.nA], self.gauge(x[self.nA :])
        gA, gb = self._data_normal(xA, xb)
        return torch.cat([gA + self.mu * xA, self.gauge(gb)])

    def precond(self, r):
        rA, rb = r[: self.nA], r[self.nA :].view(-1, 6)
        zb = torch.einsum("rij,rj->ri", self.Bb_inv, rb).reshape(-1)
        if self.precond_mode == "jacobi":
            return torch.cat([rA / self.dA.reshape(-1), self.gauge(zb)])
        R = rA.view(self.K, self.N, self.N)
        u = torch.where(self.fmask, R * self.fscale, torch.zeros_like(R))
        Uh = torch.fft.fft2(u).permute(1, 2, 0)  # (N, N, K)
        z = torch.fft.ifft2(
            torch.einsum("hwij,hwj->hwi", self.finv, Uh).permute(2, 0, 1)
        ).real
        zA = torch.where(self.fmask, z * self.fscale, R / self.dA)
        return torch.cat([zA.reshape(-1), self.gauge(zb)])

    def sample_rhs(self, gen):
        """[J_A L]^T W^{1/2} z + [sqrt(mu) z', 0] (z, z' standard normal from generator gen; z is drawn tile by tile, in tile order, on the host)."""
        gA, gb = self._render_sum(
            torch.zeros(self.K, self.N, self.N, device=self.dev),
            lambda t, S, Lb: (
                t.w.sqrt() * torch.randn((t.n,), generator=gen).to(self.dev)
            ),
            want_gb=True,
        )
        zA = torch.randn(self.nA, generator=gen).to(self.dev)
        return torch.cat(
            [gA.reshape(-1) + math.sqrt(self.mu) * zA, self.gauge(gb.reshape(-1))]
        )

    def residual_rhs(self):
        """Gradient of the log posterior at the fit: [J_A^T W r - mu A_fit, L^T W r] (gauge-projected)."""
        gA, gb = self._render_sum(
            torch.zeros(self.K, self.N, self.N, device=self.dev),
            lambda t, S, Lb: t.wr,
            want_gb=True,
        )
        return torch.cat(
            [
                gA.reshape(-1) - self.mu * self.A_fit.reshape(-1),
                self.gauge(gb.reshape(-1)),
            ]
        )

    def unit_rhs(self, kA):
        """Right-hand side [k (A part), 0] for an exact variance solve."""
        return torch.cat(
            [kA.reshape(-1).to(self.dev), torch.zeros(self.nb, device=self.dev)]
        )


def pcg(
    op, b, tol=1e-3, maxiter=300, x0=None, snapshots=None, monitor=None, label="solve"
):
    """Preconditioned conjugate gradients for op.apply (symmetric positive definite on the gauge subspace). Returns x, info.
    monitor (optional): progress line every few iterations and the time budget; when it asks to stop, the iteration ends cleanly before the next
    application and info['truncated'] is True (x is the last iterate; info['rel_residual'] the residual reached)."""
    with torch.no_grad():
        x = torch.zeros_like(b) if x0 is None else x0.clone()
        r = b - op.apply(x) if x0 is not None else b.clone()
        z = op.precond(r)
        p = z.clone()
        rz = torch.dot(r, z)
        bn = float(b.norm())
        hist = []
        it, truncated = 0, False
        if monitor is not None:
            monitor.residual = 1.0
        for it in range(1, maxiter + 1):
            if monitor is not None and monitor.stop_now(label):
                truncated, it = True, it - 1
                break
            Ap = op.apply(p)
            a = rz / torch.dot(p, Ap)
            x += a * p
            r -= a * Ap
            rel = float(r.norm()) / max(bn, 1e-30)
            hist.append(rel)
            if monitor is not None:
                monitor.report(label, it, rel)
            if snapshots is not None and it in snapshots:
                snapshots[it] = x.clone()
            if rel < tol:
                break
            z = op.precond(r)
            rz_new = torch.dot(r, z)
            p = z + (rz_new / rz) * p
            rz = rz_new
    return x, {
        "iterations": it,
        "rel_residual": hist[-1] if hist else 1.0,
        "history": hist[::10],
        "truncated": truncated,
    }


# ============================================================================= estimators
def posterior_samples(op, n, seed=0, tol=1e-3, maxiter=300, log=_silent, monitor=None):
    """n draws of the coefficient error from the linearised posterior: (n, K, N, N) and the solver records. With a monitor, a draw that the
    time budget cuts is DISCARDED (an unconverged draw has too small a variance); the draws completed so far are returned (None if none) and
    monitor.hit is set."""
    gen = torch.Generator().manual_seed(seed)
    out, recs = [], []
    with torch.no_grad():
        for i in range(n):
            if monitor is not None and monitor.stop_now(f"draw {i + 1}/{n}"):
                break
            t0 = time.perf_counter()
            x, info = pcg(
                op,
                op.sample_rhs(gen),
                tol=tol,
                maxiter=maxiter,
                monitor=monitor,
                label=f"draw {i + 1}/{n}",
            )
            if info["truncated"]:
                log(
                    f"  posterior draw {i + 1}/{n}: stopped by the time budget at iteration {info['iterations']} (relative residual "
                    f"{info['rel_residual']:.2e}); this draw is discarded"
                )
                break
            out.append(x[: op.nA].view(op.K, op.N, op.N).cpu())
            info["seconds"] = time.perf_counter() - t0
            recs.append(info)
            tail = f"; {monitor.tail()}" if monitor is not None else ""
            log(
                f"  posterior draw {i + 1}/{n}: {info['iterations']} iterations, relative residual {info['rel_residual']:.2e}, {info['seconds']:.1f} s{tail}"
            )
    return (torch.stack(out) if out else None), recs


def gauss_newton_polish(op, tol=1e-4, maxiter=60, monitor=None):
    """One Gauss-Newton step from the fit with the response held: A_pol = A_fit + [F^{-1} g]_A, g = residual_rhs(). For the grid sky (linear
    in A) this is the minimiser of the weighted least squares with the weak prior, i.e. the estimator the linearised posterior describes;
    positivity is not imposed. The backgrounds move too (returned) within the gauge. Returns (A_pol (K, N, N), delta_b (n_rows, 6), info).
    With a monitor the step may be truncated (info['truncated']): the last conjugate-gradient iterate is used and info['rel_residual'] says how far
    the solve got (1.0 if not even the right-hand side was computed: A_pol = A_fit)."""
    if monitor is not None and monitor.stop_now("Gauss-Newton step"):
        return (
            op.A_fit.clone().detach(),
            torch.zeros(len(op.rows), 6),
            {"iterations": 0, "rel_residual": 1.0, "history": [], "truncated": True},
        )
    with torch.no_grad():
        x, info = pcg(
            op,
            op.residual_rhs(),
            tol=tol,
            maxiter=maxiter,
            monitor=monitor,
            label="Gauss-Newton step",
        )
        return (
            (op.A_fit + x[: op.nA].view(op.K, op.N, op.N)).detach(),
            x[op.nA :].view(-1, 6).detach(),
            info,
        )


def exact_variance(op, phi, smoother, pts, tol=1e-4, maxiter=600, monitor=None):
    """Exact sigma of the product at output pixels pts [(ix, iy), ...]: one solve F x = k per pixel (fewer if the time budget stops them)."""
    out = []
    for ix, iy in pts:
        k = phi[:, None, None].cpu() * smoother.weights_at(ix, iy)[None].to(
            torch.float32
        )
        rhs = op.unit_rhs(k)
        x, info = pcg(
            op,
            rhs,
            tol=tol,
            maxiter=maxiter,
            monitor=monitor,
            label=f"exact ({ix}, {iy})",
        )
        if info["truncated"]:
            break
        out.append((float(torch.dot(rhs, x)) ** 0.5, info["iterations"]))
    return out


def band_of(A, phi):
    return torch.einsum("k,...khw->...hw", phi.to(A.dtype).to(A.device), A)


def sigma_maps(samples, phi, pitch, scales):
    """sigma_s (n_scales, N, N) at the node positions, from posterior draws (n, K, N, N)."""
    B = band_of(samples, phi)  # (n, N, N)
    N = B.shape[-1]
    out = []
    for s in scales:
        sm = Smoother(N, pitch, s)
        out.append(sm(B.double()).pow(2).mean(0).sqrt())
    return torch.stack(out)


def supported_resolution(sig, scales, alpha_max=2.0):
    """s*(x): smallest scale with alpha = -d ln sigma / d ln s <= alpha_max (log-linear between ladder points; scales > 0 only).
    Returns s* (N, N) (NaN where never reached on the ladder) and alpha (n_scales - 1, N, N) at the mid-points."""
    sc = np.asarray(scales, dtype=np.float64)
    keep = sc > 0
    sc, sg = sc[keep], sig[torch.as_tensor(keep)]
    ls = torch.tensor(np.log(sc))
    lsg = torch.log(sg.clamp(min=1e-30))
    alpha = -(lsg[1:] - lsg[:-1]) / (ls[1:] - ls[:-1])[:, None, None]
    mids = 0.5 * (ls[1:] + ls[:-1])
    # scan from the largest scale down: s* = the scale below which alpha first exceeds alpha_max (at the finest scales the prior takes over and
    # alpha falls again towards 1, the white-noise rate of the prior: that regime is not resolution)
    n = alpha.shape[0]
    s_star = torch.full(sig.shape[1:], float("nan"), dtype=torch.float64)
    ok = alpha[n - 1] <= alpha_max
    s_star = torch.where(
        ok, torch.full_like(s_star, math.exp(float(mids[n - 1]))), s_star
    )
    alive = ok.clone()
    for i in range(n - 2, -1, -1):
        a_lo, a_hi = alpha[i], alpha[i + 1]
        stop = alive & (a_lo > alpha_max)
        f = ((a_lo - alpha_max) / (a_lo - a_hi).clamp(min=1e-12)).clamp(0, 1)
        val = torch.exp(mids[i] + f * (mids[i + 1] - mids[i]))
        s_star = torch.where(stop, val, s_star)
        cont = alive & ~stop
        s_star = torch.where(
            cont, torch.full_like(s_star, math.exp(float(mids[i]))), s_star
        )
        alive = cont
    return s_star, alpha


def correlation_length(samples, phi, pitch, s, covered, max_lag=None):
    """Azimuthally averaged correlation of the product errors at scale s over the covered nodes; length = lag where it falls to 1/2 [arcsec]."""
    B = band_of(samples, phi).double()
    N = B.shape[-1]
    E = Smoother(N, pitch, s)(B)
    m = covered.double().cpu()
    E = E / E.pow(2).mean(0).sqrt().clamp(min=1e-30) * m
    F = torch.fft.rfft2(E, s=(2 * N, 2 * N))
    ac = torch.fft.irfft2((F * F.conj()).mean(0), s=(2 * N, 2 * N))
    Fm = torch.fft.rfft2(m, s=(2 * N, 2 * N))
    nm = torch.fft.irfft2(Fm * Fm.conj(), s=(2 * N, 2 * N))
    ac = ac / nm.clamp(min=1.0)
    L = max_lag or N // 2
    yy, xx = np.meshgrid(np.r_[0:L], np.r_[0:L], indexing="ij")
    r = np.hypot(xx, yy)
    prof = []
    for k in range(L):
        sel = (r >= k - 0.5) & (r < k + 0.5)
        prof.append(float(ac[:L, :L].numpy()[sel].mean()))
    prof = np.array(prof) / prof[0]
    below = np.nonzero(prof < 0.5)[0]
    if below.size == 0:
        return float("nan"), prof
    k = below[0]
    lag = (k - 1) + (prof[k - 1] - 0.5) / max(prof[k - 1] - prof[k], 1e-12)
    return float(lag * pitch), prof


def difference_map(
    img_a, img_b, phi, pitch, s, mode="halves", window=None, covered=None
):
    """Empirical estimator from two coefficient grids (K, N, N) (or band images (N, N)): the product difference at scale s and the implied
    sigma of the full fit. mode 'halves': h = (I_a - I_b) / 2 (fits on disjoint halves; Var h = Var of the full fit when the noise variance
    scales as 1 / exposures); mode 'seeds': h = (I_a - I_b) / sqrt(2) (two fits differing by the seed; Var h = optimisation scatter of one fit).
    Implied sigma: global rms of h over the covered nodes and a local map sqrt(g_window * h^2) (window default 6 s)."""

    def band(x):
        x = torch.as_tensor(x).double()
        return band_of(x, phi.double()) if x.dim() == 3 else x

    a, b = band(img_a), band(img_b)
    N = a.shape[-1]
    sm = Smoother(N, pitch, s, dtype=torch.float64)
    fac = 0.5 if mode == "halves" else 1 / math.sqrt(2)
    h = fac * (sm(a) - sm(b))
    cov = (
        torch.ones(N, N, dtype=torch.bool)
        if covered is None
        else torch.as_tensor(covered).bool()
    )
    glob = float(h[cov].pow(2).mean().sqrt())
    w = window or max(6 * s, 3 * pitch)
    sw = Smoother(N, pitch, w, dtype=torch.float64)
    num = sw(h.pow(2) * cov.double())
    den = sw(cov.double()).clamp(min=1e-6)
    local = (num / den).sqrt() * cov.double()
    return {
        "h": h,
        "sigma_global": glob,
        "sigma_local": local,
        "window": w,
        "mode": mode,
    }


# ============================================================================= half split of the training exposures
def alternate_groups(groups, half):
    """Training dither groups kept by --train-half: groups sorted by name, rank parity == half. Returns the kept set."""
    gs = sorted(set(groups))
    return {g for i, g in enumerate(gs) if i % 2 == half}


# ============================================================================= one call: the whole map
def solve_shared(
    op,
    n_samples=8,
    method="draws",
    polish=True,
    tol=1e-4,
    maxiter=60,
    seed=0,
    log=_silent,
    monitor=None,
):
    """The part of the map that does not depend on the band: the Gauss-Newton step (all K spectral coefficients at once) and the posterior
    draws (K coefficients each). Done once for any number of bands. The step comes first (it is the product), the draws after it. With a
    monitor, the time budget can cut either: the step is then the last iterate, a cut draw is discarded, and no further draw is started.
    Returns a dict: A_pol (None without the step), db, polish (info), samples (None without draws), recs, seconds_samples, truncated, where,
    residual (of the solve that was cut, else of the step), n_samples_requested."""
    sh = {
        "A_pol": None,
        "db": None,
        "polish": None,
        "samples": None,
        "recs": [],
        "seconds_samples": 0.0,
        "truncated": False,
        "where": "",
        "residual": float("nan"),
        "n_samples_requested": n_samples if method == "draws" else 0,
        "method": method,
    }
    with torch.no_grad():
        if polish:
            A_pol, db, info = gauss_newton_polish(
                op, tol=tol, maxiter=maxiter, monitor=monitor
            )
            sh.update(
                A_pol=A_pol.cpu(),
                db=db.cpu(),
                polish=info,
                residual=info["rel_residual"],
            )
            log(
                f"Gauss-Newton step from the fit: {info['iterations']} iterations, relative residual {info['rel_residual']:.2e}"
                + (" (TRUNCATED by the time budget)" if info["truncated"] else "")
                + f"; background constants moved by {db[:, 0].abs().max().item():.4g} at most (gauge-preserving)"
            )
        t0 = time.perf_counter()
        if (
            method == "draws"
            and n_samples > 0
            and not (monitor is not None and monitor.hit)
        ):
            samp, recs = posterior_samples(
                op,
                n_samples,
                seed=seed,
                tol=tol,
                maxiter=maxiter,
                log=log,
                monitor=monitor,
            )
            sh.update(samples=samp, recs=recs)
        sh["seconds_samples"] = time.perf_counter() - t0
    if monitor is not None and monitor.hit:
        sh.update(truncated=True, where=monitor.where, residual=monitor.residual)
        log(
            f"time budget reached in '{monitor.where}' (relative residual reached {monitor.residual:.2e}); {len(sh['recs'])} of "
            f"{sh['n_samples_requested']} draws complete; outputs are written from what was solved"
        )
    return sh


def band_readout(
    model,
    op,
    shared,
    band,
    resolution: str | float = "auto",
    alpha_max=2.0,
    scales_pitch=SCALES_DEFAULT,
    seed=0,
    n_exact=0,
    log=_silent,
    var_window=3.0,
    monitor=None,
    rho=None,
    tol=1e-4,
):
    """Everything that depends on the band, from the shared solve: the circulant sigma, sigma from the draws, supported resolution, the product at
    its resolution, the error correlation length and (optional) exact solves. No application of F except for the exact solves."""
    sky = model.sky
    pitch, N = sky.pitch, sky.N
    phi = band_vector(model, band).float().cpu()
    scales = [f * pitch for f in scales_pitch]
    cov = op.covered.cpu()
    samp, recs, method = shared["samples"], shared["recs"], shared["method"]
    circ = torch.stack(
        [
            op.circulant_sigma(phi, sc)
            if sc > 0
            else torch.full((N, N), float("nan"), dtype=torch.float64)
            for sc in scales
        ]
    )
    sig = sigma_maps(samp, phi, pitch, scales) if samp is not None else circ
    s_star, alpha = supported_resolution(sig, scales, alpha_max)
    s_med = (
        float(np.nanmedian(s_star[cov].numpy())) if bool(cov.any()) else float("nan")
    )
    s_use = s_med if resolution == "auto" else float(resolution)
    sm = Smoother(N, pitch, s_use, dtype=torch.float64)
    A_fit = op.A_fit.cpu().double()
    img_fit = sm(band_of(A_fit, phi.double()))
    pol_info = shared["polish"]
    img = (
        sm(band_of(shared["A_pol"].double(), phi.double()))
        if shared["A_pol"] is not None
        else img_fit
    )
    sig_c = op.circulant_sigma(phi, s_use)
    if samp is not None:
        var = sm(band_of(samp, phi).double()).pow(2).mean(0)
        w = Smoother(N, pitch, var_window * s_use, dtype=torch.float64)
        m = cov.double()
        sigma = (w(var * m) / w(m).clamp(min=1e-6)).sqrt() * m
        ell, prof = correlation_length(samp, phi, pitch, s_use, cov)
    else:
        sigma = torch.nan_to_num(sig_c, nan=0.0)
        ell, prof = float("nan"), None
    exact = []
    if n_exact:
        rng = np.random.default_rng(seed)
        idx = torch.nonzero(cov)
        pick = idx[
            torch.as_tensor(
                rng.choice(len(idx), size=min(n_exact, len(idx)), replace=False)
            )
        ]
        ex = exact_variance(
            op, phi, sm, [(int(p[1]), int(p[0])) for p in pick], monitor=monitor
        )
        exact = [
            {
                "row": int(p[0]),
                "col": int(p[1]),
                "sigma_exact": v,
                "iterations": it,
                "sigma_map": float(sigma[p[0], p[1]]),
                "sigma_circulant": float(sig_c[p[0], p[1]]),
            }
            for p, (v, it) in zip(pick, ex)
        ]
    ok = cov & torch.isfinite(sig_c) & (sigma > 0)
    ratio = float((sigma[ok] / sig_c[ok]).median()) if bool(ok.any()) else float("nan")
    iters = [r["iterations"] for r in recs]
    return {
        "image": img,
        "image_fit": img_fit,
        "sigma": sigma,
        "sigma_circulant": sig_c,
        "draws_over_circulant": ratio,
        "s_star": s_star,
        "alpha": alpha,
        "sigma_scales": sig,
        "scales": scales,
        "resolution": s_use,
        "resolution_median": s_med,
        "corr_length": ell,
        "corr_profile": prof,
        "covered": cov,
        "samples": samp,
        "phi": phi,
        "exact": exact,
        "cg_iterations": iters,
        "seconds_samples": shared["seconds_samples"],
        "n_apply": op.n_apply,
        "seconds_per_apply": op.t_apply / max(op.n_apply, 1),
        "mu": op.mu,
        "rho": op.rho if rho is None else rho,
        "node_noise": op.node_noise,
        "n_pix": op.n_pix,
        "n_huber_downweighted": op.n_huber,
        "polish": pol_info,
        "method": method,
        "op": op,
        "truncated": shared["truncated"],
        "truncated_in": shared["where"],
        "residual": shared["residual"],
        "n_samples_requested": shared["n_samples_requested"],
        "sigma_source": "draws" if samp is not None else "circulant",
    }


def uncertainty_map(
    model,
    exps,
    cfg,
    band,
    n_samples=8,
    rho=10.0,
    scales_pitch=SCALES_DEFAULT,
    resolution="auto",
    alpha_max=2.0,
    tol=1e-4,
    maxiter=60,
    seed=0,
    n_exact=0,
    log=_silent,
    P=None,
    op=None,
    method="draws",
    polish=True,
    var_window=3.0,
    monitor=None,
):
    """Sky product, sigma, supported resolution and correlation length for one band (node grid). Returns a dict of tensors / numbers.
    method 'draws': sigma from n_samples posterior draws, the variance map averaged locally over a Gaussian window of var_window x the product
    resolution (reduces the sampling error of few draws; assumes sigma varies slowly on that scale); the Fourier (circulant) sigma is computed
    too and written alongside. method 'circulant': no draws (cost: the preconditioner's K + 1 applications only).
    polish: the product SKY is the fit plus one Gauss-Newton step (the estimator the posterior describes; calibrated on the known-truth
    problem); SKY_FIT is the fit's own sky at the same resolution (its sigma is NOT the posterior's: see the report).
    Several bands: call solve_shared once and band_readout per band (what the command line does); this function is the two for one band."""
    op = op or SkyNormalOperator(model, exps, cfg, P=P, rho=rho, log=log)
    shared = solve_shared(
        op,
        n_samples=n_samples,
        method=method,
        polish=polish,
        tol=tol,
        maxiter=maxiter,
        seed=seed,
        log=log,
        monitor=monitor,
    )
    return band_readout(
        model,
        op,
        shared,
        band,
        resolution=resolution,
        alpha_max=alpha_max,
        scales_pitch=scales_pitch,
        seed=seed,
        n_exact=n_exact,
        log=log,
        var_window=var_window,
        monitor=monitor,
        rho=rho,
        tol=tol,
    )
