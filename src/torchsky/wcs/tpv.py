import torch
from torch import Tensor
from typing import Dict, Any, Tuple

from torchsky._native import cpp as _cpp


PV1_KEYS = tuple(f"PV1_{i}" for i in range(40))
PV2_KEYS = tuple(f"PV2_{i}" for i in range(40))
PV1_KEYS_SET = frozenset(PV1_KEYS)
PV2_KEYS_SET = frozenset(PV2_KEYS)


class TPV:
    """
    Tangent PV (TPV) distortion correction.
    """

    _CPP_INVERT_MAX_POINTS = 65536

    def __init__(self, header: Dict[str, Any]):
        self.power_map = self._build_power_map()
        self.idx1, self.c1 = self._parse_pv(header, 1)
        self.idx2, self.c2 = self._parse_pv(header, 2)
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
        if abs(det) < 1e-15:
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
        if u.numel() == 0:
            return u, v
        if self._can_use_cpp_distort(u, v):
            return _cpp.wcs_tpv_distort(
                u.contiguous(),
                v.contiguous(),
                self.idx1,
                self.c1,
                self.idx2,
                self.c2,
            )
        return self._distort_impl(u, v)

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
        if xi_t.numel() == 0:
            return xi_t, eta_t

        shape = xi_t.shape

        # Use C++ path if requested and available
        if self._can_use_cpp_invert(xi_t, eta_t) and not self._invert_trace_enabled:
            u_out, v_out = _cpp.wcs_tpv_invert(
                xi_t.reshape(-1).contiguous(),
                eta_t.reshape(-1).contiguous(),
                self.idx1,
                self.c1,
                self.idx2,
                self.c2,
                int(max_iter),
                float(tol),
            )
            return u_out.reshape(shape), v_out.reshape(shape)

        xi_f = xi_t.reshape(-1)
        eta_f = eta_t.reshape(-1)

        u_curr, v_curr = self._initial_guess_affine(xi_f, eta_f)
        active_counts = []
        if u_curr.requires_grad or v_curr.requires_grad:
            u = u_curr
            v = v_curr
            for _ in range(max_iter):
                xi, eta, dxi_du, dxi_dv, deta_du, deta_dv = (
                    self._distort_and_jacobian_impl(u, v)
                )
                fx0 = xi - xi_f
                fx1 = eta - eta_f
                res_norm = torch.sqrt(fx0 * fx0 + fx1 * fx1)
                if self._invert_trace_enabled:
                    active_counts.append(int((res_norm > tol).sum().item()))
                det = dxi_du * deta_dv - dxi_dv * deta_du
                det_sign = torch.where(
                    det < 0.0, -torch.ones_like(det), torch.ones_like(det)
                )
                det = torch.where(det.abs() < 1e-18, det_sign * 1e-18, det)
                du = (deta_dv * fx0 - dxi_dv * fx1) / det
                dv = (-deta_du * fx0 + dxi_du * fx1) / det
                u = u - du.clamp(-1.0, 1.0)
                v = v - dv.clamp(-1.0, 1.0)
            u_out = u
            v_out = v
            final_active = active_counts[-1] if active_counts else int(xi_t.numel())
        else:
            u = u_curr.clone()
            v = v_curr.clone()
            active = torch.ones_like(u, dtype=torch.bool)
            for _ in range(max_iter):
                active_idx = torch.nonzero(active, as_tuple=False).squeeze(1)
                if active_idx.numel() == 0:
                    break
                xi, eta, dxi_du, dxi_dv, deta_du, deta_dv = (
                    self._distort_and_jacobian_impl(u[active_idx], v[active_idx])
                )
                fx0 = xi - xi_f[active_idx]
                fx1 = eta - eta_f[active_idx]
                res_norm = torch.sqrt(fx0 * fx0 + fx1 * fx1)
                still_active = res_norm > tol
                if self._invert_trace_enabled:
                    active_counts.append(int(still_active.sum().item()))
                det = dxi_du * deta_dv - dxi_dv * deta_du
                det_sign = torch.where(
                    det < 0.0, -torch.ones_like(det), torch.ones_like(det)
                )
                det = torch.where(det.abs() < 1e-18, det_sign * 1e-18, det)
                du = (deta_dv * fx0 - dxi_dv * fx1) / det
                dv = (-deta_du * fx0 + dxi_du * fx1) / det
                u[active_idx] = u[active_idx] - du.clamp(-1.0, 1.0)
                v[active_idx] = v[active_idx] - dv.clamp(-1.0, 1.0)
                next_active = torch.zeros_like(active)
                next_active[active_idx[still_active]] = True
                active = next_active
            u_out = u
            v_out = v
            final_active = int(active.sum().item())

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
        u, v = self._initial_guess_affine(xi_t, eta_t)
        if self._invert_trace_enabled:
            n_points = int(xi_t.numel())
            self._last_invert_trace = {
                "n_points": n_points,
                "converged": n_points,
                "final_active": 0,
                "active_counts": [],
                "active_counts_sum": 0,
                "iterations": 0,
                "iterations_max": 0,
                "mean_point_iterations": 0.0,
                "backend": "affine_exact",
            }
        return u, v

    def _can_use_cpp_invert(self, xi_t, eta_t):
        return bool(
            _cpp is not None
            and hasattr(_cpp, "wcs_tpv_invert")
            and xi_t.device.type == "cpu"
            and eta_t.device.type == "cpu"
            and xi_t.dtype == torch.float64
            and eta_t.dtype == torch.float64
            and self.idx1.device.type == "cpu"
            and self.idx2.device.type == "cpu"
            and self.c1.device.type == "cpu"
            and self.c2.device.type == "cpu"
        )

    def _can_use_cpp_distort(self, u_t, v_t):
        return bool(
            _cpp is not None
            and hasattr(_cpp, "wcs_tpv_distort")
            and u_t.device.type == "cpu"
            and v_t.device.type == "cpu"
            and u_t.dtype == torch.float64
            and v_t.dtype == torch.float64
            and not u_t.requires_grad
            and not v_t.requires_grad
            and self.idx1.device.type == "cpu"
            and self.idx2.device.type == "cpu"
            and self.c1.device.type == "cpu"
            and self.c2.device.type == "cpu"
        )
