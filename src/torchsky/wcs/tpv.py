import math
from typing import Any, Dict, Tuple

import torch
from torch import Tensor

PV1_KEYS = tuple(f"PV1_{i}" for i in range(40))
PV2_KEYS = tuple(f"PV2_{i}" for i in range(40))
PV1_KEYS_SET = frozenset(PV1_KEYS)
PV2_KEYS_SET = frozenset(PV2_KEYS)


class TPV:
    """
    Tangent PV (TPV) distortion correction.
    """

    _CPP_INVERT_MAX_POINTS = 65536
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

    def set_invert_trace(self, enabled: bool):
        self._invert_trace_enabled = enabled

    def get_last_invert_trace(self) -> Dict[str, Any] | None:
        return self._last_invert_trace

    def set_cpp_invert_max_points(self, n: int):
        self._CPP_INVERT_MAX_POINTS = n

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
        self.c1 = self.c1.to(device=device, dtype=coeff_dtype)
        self.c2 = self.c2.to(device=device, dtype=coeff_dtype)
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
        valid = torch.isfinite(u) & torch.isfinite(v) & torch.isfinite(xi) & torch.isfinite(eta)
        nan = torch.full_like(xi, torch.nan)
        return torch.where(valid, xi, nan), torch.where(valid, eta, nan)

    def _distort_impl(self, u: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        r = torch.hypot(u, v)
        xp = [torch.ones_like(u), u]
        yp = [torch.ones_like(v), v]
        rp = [torch.ones_like(r), r]
        for _ in range(2, 8):
            xp.append(xp[-1] * u)
            yp.append(yp[-1] * v)
            rp.append(rp[-1] * r)

        xi = torch.zeros_like(u)
        for px, py, pr, c in self.terms1:
            xi.add_(xp[px] * yp[py] * rp[pr], alpha=c)
        eta = torch.zeros_like(v)
        for px, py, pr, c in self.terms2:
            eta.add_(yp[px] * xp[py] * rp[pr], alpha=c)
        return xi, eta

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
        r = torch.hypot(u, v)
        xp, yp, rp = (
            [torch.ones_like(u), u],
            [torch.ones_like(v), v],
            [torch.ones_like(r), r],
        )
        for _ in range(2, 8):
            xp.append(xp[-1] * u)
            yp.append(yp[-1] * v)
            rp.append(rp[-1] * r)

        r_safe = torch.where(r > 0, r, torch.ones_like(r))
        dr_du = torch.where(r > 0, u / r_safe, torch.zeros_like(u))
        dr_dv = torch.where(r > 0, v / r_safe, torch.zeros_like(v))

        def eval_poly_jac(terms, dx_p, dy_p, dr_p, xc, yc, dr_da, dr_db):
            out = torch.zeros_like(u)
            for px, py, pr, c in terms:
                out.add_(xc[px] * yc[py] * rp[pr], alpha=c)
            da = torch.zeros_like(u)
            for px, py, pr, c in dx_p:
                da.add_(xc[px] * yc[py] * rp[pr], alpha=c)
            db = torch.zeros_like(u)
            for px, py, pr, c in dy_p:
                db.add_(xc[px] * yc[py] * rp[pr], alpha=c)
            if dr_p:
                pref = torch.zeros_like(u)
                for px, py, pr, c in dr_p:
                    pref.add_(xc[px] * yc[py] * rp[pr], alpha=c)
                da.add_(pref * dr_da)
                db.add_(pref * dr_db)
            return out, da, db

        xi, dxi_du, dxi_dv = eval_poly_jac(
            self.terms1,
            self.terms1_dx_poly,
            self.terms1_dy_poly,
            self.terms1_rad,
            xp,
            yp,
            dr_du,
            dr_dv,
        )
        eta, deta_dv, deta_du = eval_poly_jac(
            self.terms2,
            self.terms2_dx_poly,
            self.terms2_dy_poly,
            self.terms2_rad,
            yp,
            xp,
            dr_dv,
            dr_du,
        )
        return xi, eta, dxi_du, dxi_dv, deta_du, deta_dv

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

        xi_f = xi_t.reshape(-1)
        eta_f = eta_t.reshape(-1)

        u_curr, v_curr = self._initial_guess_affine(xi_f, eta_f)
        active_counts = []
        u, v = u_curr, v_curr
        viable = (
            torch.isfinite(xi_f)
            & torch.isfinite(eta_f)
            & torch.isfinite(u)
            & torch.isfinite(v)
        )
        active = viable
        for _ in range(max_iter):
            xi, eta, dxi_du, dxi_dv, deta_du, deta_dv = (
                self._distort_and_jacobian_impl(u, v)
            )
            fx0 = xi - xi_f
            fx1 = eta - eta_f
            res_norm = torch.hypot(fx0, fx1)
            det = dxi_du * deta_dv - dxi_dv * deta_du
            conditioned = self._well_conditioned(
                dxi_du, dxi_dv, deta_du, deta_dv
            )
            good = (
                torch.isfinite(res_norm)
                & torch.isfinite(u)
                & torch.isfinite(v)
                & conditioned
            )
            viable = viable & (~active | good)
            update = active & good & (res_norm > tol)
            if self._invert_trace_enabled:
                active_counts.append(int(update.sum().item()))
            det_safe = torch.where(update, det, torch.ones_like(det))
            du = (deta_dv * fx0 - dxi_dv * fx1) / det_safe
            dv = (-deta_du * fx0 + dxi_du * fx1) / det_safe
            u = torch.where(update, u - du.clamp(-1.0, 1.0), u)
            v = torch.where(update, v - dv.clamp(-1.0, 1.0), v)
            active = update & torch.isfinite(u) & torch.isfinite(v)
            viable = viable & (~update | active)

        xi, eta, dxi_du, dxi_dv, deta_du, deta_dv = (
            self._distort_and_jacobian_impl(u, v)
        )
        final_residual = torch.hypot(xi - xi_f, eta - eta_f)
        valid = (
            viable
            & torch.isfinite(final_residual)
            & (final_residual <= tol)
            & self._well_conditioned(dxi_du, dxi_dv, deta_du, deta_dv)
        )
        nan = torch.full_like(u, torch.nan)
        u_out = torch.where(valid, u, nan)
        v_out = torch.where(valid, v, nan)
        final_active = int((~valid).sum().item())

        if self._invert_trace_enabled:
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
            valid = finite & torch.isfinite(u) & torch.isfinite(v) & (
                torch.hypot(xi - xi_t, eta - eta_t) <= 1e-11
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
