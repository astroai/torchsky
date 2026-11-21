import math
from typing import Any, Dict, Tuple

import torch
from torch import Tensor
from ..device import _to
from ._math import _hypot

PV1_KEYS = tuple(f"PV1_{i}" for i in range(40))
PV2_KEYS = tuple(f"PV2_{i}" for i in range(40))
PV1_KEYS_SET = frozenset(PV1_KEYS)
PV2_KEYS_SET = frozenset(PV2_KEYS)


class TPV:
    """
    Tangent PV (TPV) distortion correction.
    """

    _CONDITION_EPS = 64.0 * torch.finfo(torch.float64).eps

    def __init__(self, header: Dict[str, Any]):
        self.power_map = self._build_power_map()
        self.idx1, self.c1 = self._parse_pv(header, 1)
        self.idx2, self.c2 = self._parse_pv(header, 2)
        self._validate_metadata(self.idx1, self.c1, "PV1")
        self._validate_metadata(self.idx2, self.c2, "PV2")
        self.terms1 = self._build_terms(self.idx1, self.c1)
        self.terms2 = self._build_terms(self.idx2, self.c2)

        (
            self.terms1_dx_poly,
            self.terms1_dy_poly,
            self.terms1_rad,
        ) = self._build_derivative_terms(self.terms1)
        (
            self.terms2_dx_poly,
            self.terms2_dy_poly,
            self.terms2_rad,
        ) = self._build_derivative_terms(self.terms2)

        (
            self._affine_seed_b1,
            self._affine_seed_b2,
            self._affine_seed_inv00,
            self._affine_seed_inv01,
            self._affine_seed_inv10,
            self._affine_seed_inv11,
            self._has_affine_seed_inverse,
            self._affine_seed_is_identity,
            self._affine_only,
        ) = self._build_affine_seed_params(self.terms1, self.terms2)

        self._invert_trace_enabled = False
        self._last_invert_trace = None
        self._plans: Dict[Any, Any] = {}
        # Host-side (row, u power, v power, r power, coefficient) terms for the
        # traced evaluator, built eagerly so tracing never mutates state.
        self._host_terms = {
            jacobian: self._plan_entries(jacobian) for jacobian in (False, True)
        }

    def set_invert_trace(self, enabled: bool):
        self._invert_trace_enabled = enabled

    def get_last_invert_trace(self) -> Dict[str, Any] | None:
        return self._last_invert_trace

    def _distort_and_jacobian(self, u: Tensor, v: Tensor):
        return self._distort_and_jacobian_impl(u, v)

    def _build_power_map(self):
        mapping = {0: (0, 0, 0)}
        idx = 1
        for deg in range(1, 8):
            for k in range(deg + 1):
                mapping[idx] = (deg - k, k, 0)
                idx += 1
            if deg % 2 == 1:
                mapping[idx] = (0, 0, deg)
                idx += 1
        return mapping

    def _parse_pv(self, header: Dict[str, Any], axis: int):
        indices = []
        coeffs = []
        keys = PV1_KEYS if axis == 1 else PV2_KEYS
        keys_set = PV1_KEYS_SET if axis == 1 else PV2_KEYS_SET

        # Intersect keys to avoid looking up 40 keys individually in large headers
        present_keys = keys_set.intersection(header.keys())

        for j, key in enumerate(keys):
            if key in present_keys:
                val = float(header[key])
                if not math.isfinite(val):
                    raise ValueError(f"{key} must be finite")
                if val != 0 and j in self.power_map:
                    indices.append(self.power_map[j])
                    coeffs.append(val)
        if not indices:
            return torch.empty((0, 3), dtype=torch.long), torch.empty(
                (0), dtype=torch.float64
            )
        return torch.tensor(indices, dtype=torch.long), torch.tensor(
            coeffs, dtype=torch.float64
        )

    @staticmethod
    def _validate_metadata(idx: Tensor, coeffs: Tensor, name: str) -> None:
        if idx.ndim != 2 or idx.shape[1] != 3:
            raise ValueError(f"{name} exponents must have shape [N, 3]")
        if coeffs.ndim != 1 or idx.shape[0] != coeffs.shape[0]:
            raise ValueError(f"{name} exponent and coefficient lengths must match")
        if idx.numel() and bool(((idx < 0) | (idx > 7)).any().item()):
            raise ValueError(f"{name} exponents must be in [0, 7]")
        if not bool(torch.isfinite(coeffs).all().item()):
            raise ValueError(f"{name} coefficients must be finite")

    def _build_terms(
        self, idx: Tensor, coeffs: Tensor
    ) -> "list[tuple[int, int, int, float]]":
        return [
            (int(idx[k, 0]), int(idx[k, 1]), int(idx[k, 2]), float(coeffs[k]))
            for k in range(coeffs.numel())
        ]

    def _build_derivative_terms(self, terms):
        dx, dy, dr = [], [], []
        for px, py, pr, c in terms:
            if px > 0:
                dx.append((px - 1, py, pr, c * px))
            if py > 0:
                dy.append((px, py - 1, pr, c * py))
            if pr > 0:
                dr.append((px, py, pr - 1, c * pr))
        return dx, dy, dr

    @staticmethod
    def _build_affine_seed_params(terms1, terms2):
        b1, b2, a11, a12, a21, a22 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        affine_only = True
        for px, py, pr, c in terms1:
            if c == 0.0:
                continue
            if pr != 0:
                affine_only = False
                continue
            if px == 0 and py == 0:
                b1 += c
            elif px == 1 and py == 0:
                a11 += c
            elif px == 0 and py == 1:
                a12 += c
            else:
                affine_only = False
        for px, py, pr, c in terms2:
            if c == 0.0:
                continue
            if pr != 0:
                affine_only = False
                continue
            if px == 0 and py == 0:
                b2 += c
            elif px == 1 and py == 0:
                a22 += c
            elif px == 0 and py == 1:
                a21 += c
            else:
                affine_only = False
        det = a11 * a22 - a12 * a21
        scale = max(abs(a11), abs(a12), abs(a21), abs(a22))
        if (
            not math.isfinite(det)
            or not math.isfinite(scale)
            or scale == 0.0
            or abs(det) <= TPV._CONDITION_EPS * scale * scale
        ):
            return b1, b2, 1.0, 0.0, 0.0, 1.0, False, True, affine_only
        is_id = (
            abs(b1) < 1e-15
            and abs(b2) < 1e-15
            and abs(a11 - 1.0) < 1e-15
            and abs(a12) < 1e-15
            and abs(a21) < 1e-15
            and abs(a22 - 1.0) < 1e-15
        )
        return (
            b1,
            b2,
            a22 / det,
            -a12 / det,
            -a21 / det,
            a11 / det,
            True,
            is_id,
            affine_only,
        )

    def to(self, device: torch.device) -> "TPV":
        self.idx1, self.idx2 = self.idx1.to(device), self.idx2.to(device)
        coeff_dtype = torch.float32 if device.type == "mps" else self.c1.dtype
        self.c1 = _to(self.c1, device=device, dtype=coeff_dtype)
        self.c2 = _to(self.c2, device=device, dtype=coeff_dtype)
        return self

    def distort(self, u: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        if u.shape != v.shape:
            raise ValueError("TPV distortion inputs must have identical shapes")
        if u.numel() == 0:
            return u, v
        if not self.terms1 and not self.terms2:
            valid = torch.isfinite(u) & torch.isfinite(v)
            nan = torch.full_like(u, torch.nan)
            return torch.where(valid, u, nan), torch.where(valid, v, nan)
        xi, eta = self._distort_impl(u, v)
        valid = (
            torch.isfinite(u)
            & torch.isfinite(v)
            & torch.isfinite(xi)
            & torch.isfinite(eta)
        )
        nan = torch.full_like(xi, torch.nan)
        return torch.where(valid, xi, nan), torch.where(valid, eta, nan)

    # Both polynomials and their Jacobians are evaluated as one product of
    # monomials: M[k] = u**a_k * v**b_k * r**c_k over the distinct monomials k,
    # then a single (rows, K) @ (K, N) matmul. This replaces a few hundred
    # per-term kernels per call (the TPV inverse calls it once per Newton sweep)
    # with about twenty, whatever the number of PV terms.
    _MONOMIAL_BUDGET = 1 << 22  # elements of M per chunk (32 MiB in float64)

    def _plan_entries(self, jacobian: bool) -> "list[tuple[int, int, int, int, float]]":
        # (row, u power, v power, r power, coefficient); poly 2 has the roles
        # of its x and y exponents swapped (it is evaluated in (v, u)).
        entries: list[tuple[int, int, int, int, float]] = []
        for px, py, pr, c in self.terms1:
            entries.append((0, px, py, pr, c))
        for px, py, pr, c in self.terms2:
            entries.append((4 if jacobian else 1, py, px, pr, c))
        if jacobian:
            for px, py, pr, c in self.terms1_dx_poly:
                entries.append((1, px, py, pr, c))  # d xi / du
            for px, py, pr, c in self.terms1_dy_poly:
                entries.append((2, px, py, pr, c))  # d xi / dv
            for px, py, pr, c in self.terms1_rad:
                entries.append((3, px, py, pr, c))  # d xi / dr
            for px, py, pr, c in self.terms2_dx_poly:
                entries.append((5, py, px, pr, c))  # d eta / dv
            for px, py, pr, c in self.terms2_dy_poly:
                entries.append((6, py, px, pr, c))  # d eta / du
            for px, py, pr, c in self.terms2_rad:
                entries.append((7, py, px, pr, c))  # d eta / dr
        return entries

    def _monomial_plan(self, jacobian: bool, device: torch.device, dtype: torch.dtype):
        key = (jacobian, device, dtype)
        plan = self._plans.get(key)
        if plan is not None:
            return plan
        entries = self._host_terms[jacobian]
        monomials: dict[tuple[int, int, int], int] = {}
        for _, a, b, c, _ in entries:
            monomials.setdefault((a, b, c), len(monomials))
        rows = 8 if jacobian else 2
        coeff = torch.zeros(rows, max(len(monomials), 1), dtype=torch.float64)
        for row, a, b, c, value in entries:
            coeff[row, monomials[(a, b, c)]] += value
        powers = torch.tensor(
            list(monomials) or [(0, 0, 0)], dtype=torch.long
        ).T.contiguous()
        plan = (
            _to(powers, device=device),
            _to(coeff, device=device, dtype=dtype),
            any(c > 0 for (_, _, c) in monomials),
        )
        self._plans[key] = plan
        return plan

    def _evaluate_traced(self, u: Tensor, v: Tensor, jacobian: bool) -> Tensor:
        """:meth:`_evaluate` as one elementwise expression per row.

        Used under ``torch.compile``: host coefficients and Python power
        tables fuse into a single kernel, with no ``(monomials, N)`` buffer
        (so no chunk loop over a possibly symbolic ``N``) and no plan cache.
        """
        rows = 8 if jacobian else 2
        entries = self._host_terms[jacobian]
        max_power = max((max(a, b, c) for _, a, b, c, _ in entries), default=0)
        up = [torch.ones_like(u), u]
        vp = [torch.ones_like(v), v]
        for _ in range(2, max_power + 1):
            up.append(up[-1] * u)
            vp.append(vp[-1] * v)
        rp: list[Tensor] = []
        if any(c > 0 for _, _, _, c, _ in entries):
            r = _hypot(u, v)
            rp = [torch.ones_like(r), r]
            for _ in range(2, max_power + 1):
                rp.append(rp[-1] * r)
        out: list[Tensor] = [torch.zeros_like(u) for _ in range(rows)]
        for row, a, b, c, value in entries:
            term = up[a] * vp[b]
            if c > 0:
                term = term * rp[c]
            out[row] = out[row] + value * term
        return torch.stack(out)

    def _evaluate(self, u: Tensor, v: Tensor, jacobian: bool) -> Tensor:
        """Rows of ``coeff @ monomials`` for flattened ``u, v``: ``(rows, N)``."""
        if torch.compiler.is_compiling():
            return self._evaluate_traced(u, v, jacobian)
        powers, coeff, uses_r = self._monomial_plan(jacobian, u.device, u.dtype)
        n = u.numel()
        k = powers.shape[1]
        chunk = max(1, self._MONOMIAL_BUDGET // k)
        out = []
        for start in range(0, n, chunk):
            uc = u[start : start + chunk]
            vc = v[start : start + chunk]
            up = [torch.ones_like(uc), uc]
            vp = [torch.ones_like(vc), vc]
            for _ in range(2, 8):
                up.append(up[-1] * uc)
                vp.append(vp[-1] * vc)
            mono = torch.stack(up)[powers[0]] * torch.stack(vp)[powers[1]]
            if uses_r:
                rc = _hypot(uc, vc)
                rp = [torch.ones_like(rc), rc]
                for _ in range(2, 8):
                    rp.append(rp[-1] * rc)
                mono = mono * torch.stack(rp)[powers[2]]
            out.append(coeff @ mono)
        return out[0] if len(out) == 1 else torch.cat(out, dim=1)

    def _distort_impl(self, u: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        shape = u.shape
        rows = self._evaluate(u.reshape(-1), v.reshape(-1), jacobian=False)
        return rows[0].reshape(shape), rows[1].reshape(shape)

    def _distort_and_jacobian_impl(
        self, u: Tensor, v: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        if not self.terms1 and not self.terms2:
            return (
                u,
                v,
                torch.ones_like(u),
                torch.zeros_like(u),
                torch.zeros_like(u),
                torch.ones_like(u),
            )
        shape = u.shape
        uf, vf = u.reshape(-1), v.reshape(-1)
        rows = self._evaluate(uf, vf, jacobian=True)
        r = _hypot(uf, vf)
        r_safe = torch.where(r > 0, r, torch.ones_like(r))
        dr_du = torch.where(r > 0, uf / r_safe, torch.zeros_like(uf))
        dr_dv = torch.where(r > 0, vf / r_safe, torch.zeros_like(vf))
        xi = rows[0]
        dxi_du = rows[1] + rows[3] * dr_du
        dxi_dv = rows[2] + rows[3] * dr_dv
        eta = rows[4]
        deta_dv = rows[5] + rows[7] * dr_dv
        deta_du = rows[6] + rows[7] * dr_du
        return tuple(
            t.reshape(shape) for t in (xi, eta, dxi_du, dxi_dv, deta_du, deta_dv)
        )  # type: ignore[return-value]

    def invert(
        self, xi_t: Tensor, eta_t: Tensor, max_iter: int = 20, tol: float = 1e-11
    ) -> Tuple[Tensor, Tensor]:
        if xi_t.shape != eta_t.shape:
            raise ValueError("TPV inverse inputs must have identical shapes")
        if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError("TPV max_iter must be a positive integer")
        if not math.isfinite(tol) or tol <= 0.0:
            raise ValueError("TPV tolerance must be finite and positive")
        if xi_t.numel() == 0:
            return xi_t, eta_t

        shape = xi_t.shape
        finite_target = torch.isfinite(xi_t) & torch.isfinite(eta_t)
        if not self.terms1 and not self.terms2:
            nan = torch.full_like(xi_t, torch.nan)
            return torch.where(finite_target, xi_t, nan), torch.where(
                finite_target, eta_t, nan
            )
        if torch.compiler.is_compiling():
            return self._invert_traced(xi_t, eta_t, max_iter, tol)
        native = self._native_invert(xi_t, eta_t, max_iter, tol)
        if native is not None:
            return native

        xi_f = xi_t.reshape(-1)
        eta_f = eta_t.reshape(-1)

        u_curr, v_curr = self._initial_guess_affine(xi_f, eta_f)
        tol_pt = self._tolerance_floor(xi_f, eta_f, tol)
        active_counts = []
        u, v = u_curr, v_curr
        viable = (
            torch.isfinite(xi_f)
            & torch.isfinite(eta_f)
            & torch.isfinite(u)
            & torch.isfinite(v)
        )
        active = viable
        # Newton sweeps on the points still moving: after two or three sweeps
        # almost every point has converged, so later sweeps touch only the
        # stragglers (gathered by index; out-of-place, so autograd still flows).
        rows = torch.arange(u.numel(), device=u.device)
        for _ in range(max_iter):
            if rows.numel() == 0:
                break
            u_a, v_a = u[rows], v[rows]
            xi, eta, dxi_du, dxi_dv, deta_du, deta_dv = self._distort_and_jacobian_impl(
                u_a, v_a
            )
            fx0 = xi - xi_f[rows]
            fx1 = eta - eta_f[rows]
            res_norm = _hypot(fx0, fx1)
            det = dxi_du * deta_dv - dxi_dv * deta_du
            conditioned = self._well_conditioned(dxi_du, dxi_dv, deta_du, deta_dv)
            good = (
                torch.isfinite(res_norm)
                & torch.isfinite(u_a)
                & torch.isfinite(v_a)
                & conditioned
            )
            act = active[rows]
            viable = viable.index_put((rows,), viable[rows] & (~act | good))
            update = act & good & (res_norm > tol_pt[rows])
            if not bool(update.any()):
                break
            if self._invert_trace_enabled:
                active_counts.append(int(update.sum().item()))
            det_safe = torch.where(update, det, torch.ones_like(det))
            du = (deta_dv * fx0 - dxi_dv * fx1) / det_safe
            dv = (-deta_du * fx0 + dxi_du * fx1) / det_safe
            u_new = torch.where(update, u_a - du.clamp(-1.0, 1.0), u_a)
            v_new = torch.where(update, v_a - dv.clamp(-1.0, 1.0), v_a)
            u = u.index_put((rows,), u_new)
            v = v.index_put((rows,), v_new)
            still = update & torch.isfinite(u_new) & torch.isfinite(v_new)
            viable = viable.index_put((rows,), viable[rows] & (~update | still))
            active = torch.zeros_like(active).index_put((rows,), still)
            rows = rows[still]

        xi, eta, dxi_du, dxi_dv, deta_du, deta_dv = self._distort_and_jacobian_impl(
            u, v
        )
        final_residual = _hypot(xi - xi_f, eta - eta_f)
        valid = (
            viable
            & torch.isfinite(final_residual)
            & (final_residual <= tol_pt)
            & self._well_conditioned(dxi_du, dxi_dv, deta_du, deta_dv)
        )
        nan = torch.full_like(u, torch.nan)
        u_out = torch.where(valid, u, nan)
        v_out = torch.where(valid, v, nan)

        if self._invert_trace_enabled:
            final_active = int((~valid).sum().item())
            n_points = xi_t.numel()
            self._last_invert_trace = {
                "n_points": n_points,
                "converged": n_points - final_active,
                "final_active": final_active,
                "active_counts": active_counts,
                "active_counts_sum": int(sum(active_counts)),
                "iterations": len(active_counts),
                "iterations_max": len(active_counts),
                "mean_point_iterations": (
                    float(sum(active_counts)) / float(n_points) if n_points > 0 else 0.0
                ),
                "backend": "analytic_jacobian",
            }

        return u_out.reshape(shape), v_out.reshape(shape)

    def _tolerance_floor(self, xi_f: Tensor, eta_f: Tensor, tol: float) -> Tensor:
        # Per-point convergence tolerance: ``tol``, but never below what the
        # working dtype can resolve. Evaluating the polynomial (n terms) at a
        # target of size |xi| + |eta| carries ~n * eps * max(1, |xi| + |eta|)
        # rounding, so in float32 (MPS) a 1e-11 deg residual is unreachable and
        # converged points were reported as NaN. In float64 the floor is
        # ~1e-14 deg and ``tol`` (1e-11) is unchanged.
        n_terms = max(len(self.terms1), len(self.terms2), 1) + 2
        floor = (
            n_terms
            * torch.finfo(xi_f.dtype).eps
            * torch.clamp(xi_f.abs() + eta_f.abs(), min=1.0)
        )
        return torch.clamp(floor.nan_to_num(nan=tol), min=tol)

    def _invert_traced(
        self, xi_t: Tensor, eta_t: Tensor, max_iter: int, tol: float
    ) -> Tuple[Tensor, Tensor]:
        """Fixed-iteration masked form of the Newton sweeps in :meth:`invert`.

        Same seed, step clamp, conditioning, tolerance and fail-closed mask as
        the eager loop, which gathers the still-moving rows and stops when none
        moves; here every sweep runs over all rows and a per-row ``active``
        mask freezes converged ones, so ``torch.compile`` traces one graph
        (no host syncs, no data-dependent shapes). Converged rows are never
        touched again in either form, so the results agree.
        """
        shape = xi_t.shape
        xi_f = xi_t.reshape(-1)
        eta_f = eta_t.reshape(-1)
        u, v = self._initial_guess_affine(xi_f, eta_f)
        tol_pt = self._tolerance_floor(xi_f, eta_f, tol)
        viable = (
            torch.isfinite(xi_f)
            & torch.isfinite(eta_f)
            & torch.isfinite(u)
            & torch.isfinite(v)
        )
        active = viable
        for _ in range(max_iter):
            xi, eta, dxi_du, dxi_dv, deta_du, deta_dv = self._distort_and_jacobian_impl(
                u, v
            )
            fx0 = xi - xi_f
            fx1 = eta - eta_f
            res_norm = _hypot(fx0, fx1)
            det = dxi_du * deta_dv - dxi_dv * deta_du
            conditioned = self._well_conditioned(dxi_du, dxi_dv, deta_du, deta_dv)
            good = (
                torch.isfinite(res_norm)
                & torch.isfinite(u)
                & torch.isfinite(v)
                & conditioned
            )
            viable = viable & (~active | good)
            update = active & good & (res_norm > tol_pt)
            det_safe = torch.where(update, det, torch.ones_like(det))
            du = (deta_dv * fx0 - dxi_dv * fx1) / det_safe
            dv = (-deta_du * fx0 + dxi_du * fx1) / det_safe
            u_new = torch.where(update, u - du.clamp(-1.0, 1.0), u)
            v_new = torch.where(update, v - dv.clamp(-1.0, 1.0), v)
            still = update & torch.isfinite(u_new) & torch.isfinite(v_new)
            viable = viable & (~update | still)
            active = still
            u, v = u_new, v_new

        xi, eta, dxi_du, dxi_dv, deta_du, deta_dv = self._distort_and_jacobian_impl(
            u, v
        )
        final_residual = _hypot(xi - xi_f, eta - eta_f)
        valid = (
            viable
            & torch.isfinite(final_residual)
            & (final_residual <= tol_pt)
            & self._well_conditioned(dxi_du, dxi_dv, deta_du, deta_dv)
        )
        nan = torch.full_like(u, torch.nan)
        return (
            torch.where(valid, u, nan).reshape(shape),
            torch.where(valid, v, nan).reshape(shape),
        )

    # Per-point Newton in the native core (same seed, step clamp, conditioning
    # and tolerance contract) for CPU float64 inputs that need no gradient. Set
    # to False to force the torch sweeps (tests compare the two).
    _USE_NATIVE_INVERT = True

    def _native_invert(
        self, xi_t: Tensor, eta_t: Tensor, max_iter: int, tol: float
    ) -> Tuple[Tensor, Tensor] | None:
        if (
            not self._USE_NATIVE_INVERT
            or self._invert_trace_enabled
            or xi_t.device.type != "cpu"
            or eta_t.device.type != "cpu"
            or xi_t.dtype != torch.float64
            or eta_t.dtype != torch.float64
            or (torch.is_grad_enabled() and (xi_t.requires_grad or eta_t.requires_grad))
        ):
            return None
        from .._native import cpp

        if cpp is None:
            return None
        from .. import _cpp_ext

        kernel = getattr(_cpp_ext, "wcs_tpv_invert", None)
        if kernel is None:
            return None
        plan = self._plans.get("native_invert")
        if plan is None:
            terms1 = torch.tensor(
                [list(term) for term in self.terms1] or [], dtype=torch.float64
            ).reshape(-1, 4)
            terms2 = torch.tensor(
                [list(term) for term in self.terms2] or [], dtype=torch.float64
            ).reshape(-1, 4)
            seed = torch.tensor(
                [
                    self._affine_seed_b1,
                    self._affine_seed_b2,
                    self._affine_seed_inv00,
                    self._affine_seed_inv01,
                    self._affine_seed_inv10,
                    self._affine_seed_inv11,
                    1.0 if self._has_affine_seed_inverse else 0.0,
                ],
                dtype=torch.float64,
            )
            plan = (terms1, terms2, seed)
            self._plans["native_invert"] = plan
        u, v = kernel(
            torch.get_num_threads(),
            xi_t.detach().contiguous(),
            eta_t.detach().contiguous(),
            *plan,
            int(max_iter),
            float(tol),
        )
        return u, v

    @classmethod
    def _well_conditioned(
        cls, j11: Tensor, j12: Tensor, j21: Tensor, j22: Tensor
    ) -> Tensor:
        scale = torch.maximum(
            torch.maximum(j11.abs(), j12.abs()),
            torch.maximum(j21.abs(), j22.abs()),
        )
        det = j11 * j22 - j12 * j21
        return (
            torch.isfinite(scale)
            & torch.isfinite(det)
            & (scale > 0.0)
            & (det.abs() > cls._CONDITION_EPS * scale * scale)
        )

    def _initial_guess_affine(self, xi_t, eta_t):
        if not self._has_affine_seed_inverse:
            return xi_t.clone(), eta_t.clone()
        xi0, eta0 = xi_t - self._affine_seed_b1, eta_t - self._affine_seed_b2
        u = self._affine_seed_inv00 * xi0 + self._affine_seed_inv01 * eta0
        v = self._affine_seed_inv10 * xi0 + self._affine_seed_inv11 * eta0
        return u, v

    def invert_affine_exact(self, xi_t: Tensor, eta_t: Tensor) -> Tuple[Tensor, Tensor]:
        if not self._affine_only:
            raise RuntimeError("invert_affine_exact requires affine-only TPV")
        if xi_t.shape != eta_t.shape:
            raise ValueError("TPV inverse inputs must have identical shapes")
        finite = torch.isfinite(xi_t) & torch.isfinite(eta_t)
        if not self.terms1 and not self.terms2:
            valid = finite
            u, v = xi_t, eta_t
        elif not self._has_affine_seed_inverse:
            valid = torch.zeros_like(finite)
            u, v = xi_t, eta_t
        else:
            u, v = self._initial_guess_affine(xi_t, eta_t)
            xi, eta = self._distort_impl(u, v)
            valid = (
                finite
                & torch.isfinite(u)
                & torch.isfinite(v)
                & (_hypot(xi - xi_t, eta - eta_t) <= 1e-11)
            )
        nan = torch.full_like(u, torch.nan)
        u, v = torch.where(valid, u, nan), torch.where(valid, v, nan)
        if self._invert_trace_enabled:
            n_points = int(xi_t.numel())
            converged = int(valid.sum().item())
            self._last_invert_trace = {
                "n_points": n_points,
                "converged": converged,
                "final_active": n_points - converged,
                "active_counts": [],
                "active_counts_sum": 0,
                "iterations": 0,
                "iterations_max": 0,
                "mean_point_iterations": 0.0,
                "backend": "affine_exact",
            }
        return u, v
