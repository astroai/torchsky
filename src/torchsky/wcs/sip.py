import torch
from torch import Tensor
from typing import Dict, Any

class SIP:
    """
    Simple Imaging Polynomial (SIP) distortion correction.

    This class handles the parsing of SIP coefficients from a FITS header
    and applies the distortion correction to pixel coordinates.

    References:
    - Shupe et al. (2005): "The SIP Convention for Representing Distortion in FITS Image Headers"
    """

    def __init__(self, header: Dict[str, Any]):
        self.a_order = int(header.get("A_ORDER", 0))
        self.b_order = int(header.get("B_ORDER", 0))
        self.ap_order = int(header.get("AP_ORDER", 0))
        self.bp_order = int(header.get("BP_ORDER", 0))

        # Parse A/B coefficients (Forward: Pixel -> Focal Plane)
        self.a_coeffs = self._parse_coeffs(header, "A", self.a_order)
        self.b_coeffs = self._parse_coeffs(header, "B", self.b_order)

        # Parse AP/BP coefficients (Inverse: Focal Plane -> Pixel)
        self.ap_coeffs = self._parse_coeffs(header, "AP", self.ap_order)
        self.bp_coeffs = self._parse_coeffs(header, "BP", self.bp_order)
        self.a_terms = self._coeff_terms(self.a_coeffs)
        self.b_terms = self._coeff_terms(self.b_coeffs)
        self.ap_terms = self._coeff_terms(self.ap_coeffs)
        self.bp_terms = self._coeff_terms(self.bp_coeffs)
        self.a_du_terms, self.a_dv_terms = self._derivative_terms(self.a_terms)
        self.b_du_terms, self.b_dv_terms = self._derivative_terms(self.b_terms)
        self._a_pack = self._build_term_pack(self.a_terms)
        self._b_pack = self._build_term_pack(self.b_terms)
        self._ap_pack = self._build_term_pack(self.ap_terms)
        self._bp_pack = self._build_term_pack(self.bp_terms)
        self._a_du_pack = self._build_term_pack(self.a_du_terms)
        self._a_dv_pack = self._build_term_pack(self.a_dv_terms)
        self._b_du_pack = self._build_term_pack(self.b_du_terms)
        self._b_dv_pack = self._build_term_pack(self.b_dv_terms)
        # Pre-build tensors for vectorized evaluation
        self._a_c = torch.tensor([t[2] for t in self.a_terms], dtype=torch.float64)
        self._a_pq = torch.tensor(
            [[t[0], t[1]] for t in self.a_terms], dtype=torch.long
        )
        self._b_c = torch.tensor([t[2] for t in self.b_terms], dtype=torch.float64)
        self._b_pq = torch.tensor(
            [[t[0], t[1]] for t in self.b_terms], dtype=torch.long
        )

        self._ap_c = torch.tensor([t[2] for t in self.ap_terms], dtype=torch.float64)
        self._ap_pq = torch.tensor(
            [[t[0], t[1]] for t in self.ap_terms], dtype=torch.long
        )
        self._bp_c = torch.tensor([t[2] for t in self.bp_terms], dtype=torch.float64)
        self._bp_pq = torch.tensor(
            [[t[0], t[1]] for t in self.bp_terms], dtype=torch.long
        )

        self.has_forward = bool(self.a_coeffs) or bool(self.b_coeffs)
        self.has_inverse = bool(self.ap_coeffs) or bool(self.bp_coeffs)
        self._quad_forward_fast = False
        self._quad_a20 = 0.0
        self._quad_a11 = 0.0
        self._quad_a02 = 0.0
        self._quad_b20 = 0.0
        self._quad_b11 = 0.0
        self._quad_b02 = 0.0
        self._quad_inverse_fast = False
        self._init_quadratic_forward_fastpath()

    def to(self, device: torch.device) -> "SIP":
        """Move SIP tensors to device."""
        coeff_dtype = torch.float32 if device.type == "mps" else None

        def _move(tensor: Tensor) -> Tensor:
            if tensor.is_floating_point() and coeff_dtype is not None:
                return tensor.to(device=device, dtype=coeff_dtype)
            return tensor.to(device)

        self._a_c = _move(self._a_c)
        self._a_pq = _move(self._a_pq)
        self._b_c = _move(self._b_c)
        self._b_pq = _move(self._b_pq)
        self._ap_c = _move(self._ap_c)
        self._ap_pq = _move(self._ap_pq)
        self._bp_c = _move(self._bp_c)
        self._bp_pq = _move(self._bp_pq)
        for pack_name in (
            "_a_pack",
            "_b_pack",
            "_ap_pack",
            "_bp_pack",
            "_a_du_pack",
            "_a_dv_pack",
            "_b_du_pack",
            "_b_dv_pack",
        ):
            pack = getattr(self, pack_name)
            pack["p"] = _move(pack["p"])
            pack["q"] = _move(pack["q"])
            pack["c"] = _move(pack["c"])
        return self

    def _parse_coeffs(
        self, header: Dict[str, Any], prefix: str, order: int
    ) -> Dict[str, float]:
        """
        Parse coefficients for a given prefix and order.
        Example: A_2_0, A_0_2, etc.
        Returns a dictionary {(p, q): value}
        """
        coeffs = {}
        for p in range(order + 1):
            for q in range(order + 1):
                if p + q > order:
                    continue
                # Skip 0th and 1st order terms if they are typically part of CD matrix?
                # SIP convention says A/B are deviations from the linear term.
                # However, all A_p_q are valid.

                key = f"{prefix}_{p}_{q}"
                if key in header:
                    coeffs[(p, q)] = float(header[key])
        return coeffs

    def _init_quadratic_forward_fastpath(self) -> None:
        """Enable a fast path when forward SIP distortion is purely quadratic."""
        if not self.has_forward:
            return
        for p, q, _ in self.a_terms:
            if p + q > 2:
                return
        for p, q, _ in self.b_terms:
            if p + q > 2:
                return
        self._quad_a20 = float(self.a_coeffs.get((2, 0), 0.0))
        self._quad_a11 = float(self.a_coeffs.get((1, 1), 0.0))
        self._quad_a02 = float(self.a_coeffs.get((0, 2), 0.0))
        self._quad_b20 = float(self.b_coeffs.get((2, 0), 0.0))
        self._quad_b11 = float(self.b_coeffs.get((1, 1), 0.0))
        self._quad_b02 = float(self.b_coeffs.get((0, 2), 0.0))
        self._quad_forward_fast = True
        # Inverse fast path uses Newton on the same quadratic forward map.
        self._quad_inverse_fast = True

    @staticmethod
    def _coeff_terms(
        coeffs: Dict[tuple[int, int], float],
    ) -> "list[tuple[int, int, float]]":
        return [(int(p), int(q), float(c)) for (p, q), c in coeffs.items()]

    @staticmethod
    def _derivative_terms(
        terms: "list[tuple[int, int, float]]",
    ) -> "tuple[list[tuple[int, int, float]], list[tuple[int, int, float]]]":
        du_terms = []
        dv_terms = []
        for p, q, coeff in terms:
            if p > 0:
                du_terms.append((p - 1, q, coeff * p))
            if q > 0:
                dv_terms.append((p, q - 1, coeff * q))
        return du_terms, dv_terms

    @staticmethod
    def _build_term_pack(terms: "list[tuple[int, int, float]]") -> Dict[str, Tensor]:
        if not terms:
            return {
                "p": torch.empty((0,), dtype=torch.long),
                "q": torch.empty((0,), dtype=torch.long),
                "c": torch.empty((0,), dtype=torch.float64),
            }
        p = torch.tensor([t[0] for t in terms], dtype=torch.long)
        q = torch.tensor([t[1] for t in terms], dtype=torch.long)
        c = torch.tensor([t[2] for t in terms], dtype=torch.float64)
        return {"p": p, "q": q, "c": c}

    @staticmethod
    def _sum_terms_from_pack(
        pack: Dict[str, Tensor], u_stack: Tensor, v_stack: Tensor, out_like: Tensor
    ) -> Tensor:
        c = pack["c"]
        if c.device != out_like.device or c.dtype != out_like.dtype:
            c = c.to(device=out_like.device, dtype=out_like.dtype)
        if c.numel() == 0:
            return torch.zeros_like(out_like)
        p = pack["p"]
        q = pack["q"]
        if p.device != out_like.device:
            p = p.to(out_like.device)
            q = q.to(out_like.device)
        vals = u_stack.index_select(0, p) * v_stack.index_select(0, q)
        coeffs = c.reshape((-1,) + (1,) * out_like.ndim)
        return torch.sum(coeffs * vals, dim=0)

    @staticmethod
    def _make_pow_cache(base: Tensor, order: int) -> "list[Tensor]":
        pows = [torch.ones_like(base)]
        if order >= 1:
            pows.append(base)
        curr = base
        for _ in range(2, order + 1):
            curr = curr * base
            pows.append(curr)
        return pows

    @staticmethod
    def _make_pow_stack(base: Tensor, order: int) -> Tensor:
        pows = [torch.ones_like(base)]
        if order >= 1:
            pows.append(base)
        curr = base
        for _ in range(2, order + 1):
            curr = curr * base
            pows.append(curr)
        return torch.stack(pows, dim=0)

    @staticmethod
    def _horner_grid(terms: "list[tuple[int, int, float]]") -> "list[list[float]]":
        """Coefficients c[p][q] of sum_{p,q} c_pq u^p v^q, as a dense triangle for 2-D Horner evaluation."""
        if not terms:
            return []
        P = max(t[0] for t in terms)
        Q = max(t[1] for t in terms)
        grid = [[0.0] * (Q + 1) for _ in range(P + 1)]
        for p, q, c in terms:
            grid[p][q] += float(c)
        for row in grid:  # trim trailing zero coefficients of each row
            while len(row) > 1 and row[-1] == 0.0:
                row.pop()
        return grid

    @staticmethod
    def _horner2d(grid: "list[list[float]]", u: Tensor, v: Tensor) -> Tensor:
        """sum_p u^p (sum_q c_pq v^q) by nested Horner: no power tables, no (terms, N) intermediates.
        In place when no gradient is needed, out of place otherwise (autograd safe)."""
        if not grid:
            return torch.zeros_like(u)
        inplace = not (torch.is_grad_enabled() and (u.requires_grad or v.requires_grad))
        out = None
        for row in reversed(grid):
            inner = torch.full_like(v, row[-1])
            for c in reversed(row[:-1]):
                if inplace:
                    inner.mul_(v)
                    if c != 0.0:
                        inner.add_(c)
                else:
                    inner = inner * v + c
            if out is None:
                out = inner
            elif inplace:
                out.mul_(u).add_(inner)
            else:
                out = out * u + inner
        return out

    def _horner_grids(self) -> "dict[str, list[list[float]]]":
        g = getattr(self, "_hgrids", None)
        if g is None:
            g = {
                "a": self._horner_grid(self.a_terms),
                "b": self._horner_grid(self.b_terms),
                "a_du": self._horner_grid(self.a_du_terms),
                "a_dv": self._horner_grid(self.a_dv_terms),
                "b_du": self._horner_grid(self.b_du_terms),
                "b_dv": self._horner_grid(self.b_dv_terms),
                "ap": self._horner_grid(getattr(self, "ap_terms", [])),
                "bp": self._horner_grid(getattr(self, "bp_terms", [])),
            }
            self._hgrids = g
        return g

    def distort(self, u: Tensor, v: Tensor) -> "tuple[Tensor, Tensor]":
        """
        Apply forward distortion using vectorized matrix-vector multiplication.
        """
        if u.numel() == 0 or not self.has_forward:
            return u, v
        if self._quad_forward_fast:
            uv = u * v
            u2 = u * u
            v2 = v * v
            f_uv = self._quad_a20 * u2 + self._quad_a11 * uv + self._quad_a02 * v2
            g_uv = self._quad_b20 * u2 + self._quad_b11 * uv + self._quad_b02 * v2
            return u + f_uv, v + g_uv

        g = self._horner_grids()
        return u + self._horner2d(g["a"], u, v), v + self._horner2d(g["b"], u, v)

    def _forward_and_jacobian(
        self, u: Tensor, v: Tensor
    ) -> "tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]":
        """
        Evaluate SIP forward map and Jacobian in one pass.

        Returns:
            xd, yd, dxd_du, dxd_dv, dyd_du, dyd_dv
        """
        if self._quad_forward_fast:
            uv = u * v
            u2 = u * u
            v2 = v * v
            xd = u + self._quad_a20 * u2 + self._quad_a11 * uv + self._quad_a02 * v2
            yd = v + self._quad_b20 * u2 + self._quad_b11 * uv + self._quad_b02 * v2
            dxd_du = 1.0 + 2.0 * self._quad_a20 * u + self._quad_a11 * v
            dxd_dv = self._quad_a11 * u + 2.0 * self._quad_a02 * v
            dyd_du = 2.0 * self._quad_b20 * u + self._quad_b11 * v
            dyd_dv = 1.0 + self._quad_b11 * u + 2.0 * self._quad_b02 * v
            return xd, yd, dxd_du, dxd_dv, dyd_du, dyd_dv

        g = self._horner_grids()
        xd = u + self._horner2d(g["a"], u, v)
        yd = v + self._horner2d(g["b"], u, v)
        dxd_du = 1.0 + self._horner2d(g["a_du"], u, v)
        dxd_dv = self._horner2d(g["a_dv"], u, v)
        dyd_du = self._horner2d(g["b_du"], u, v)
        dyd_dv = 1.0 + self._horner2d(g["b_dv"], u, v)
        return xd, yd, dxd_du, dxd_dv, dyd_du, dyd_dv

    def undistort(self, u: Tensor, v: Tensor) -> "tuple[Tensor, Tensor]":
        """
        Apply inverse distortion using matrix-vectorized evaluation.
        """
        if u.numel() == 0 or not self.has_inverse:
            return u, v

        g = self._horner_grids()
        return u + self._horner2d(g["ap"], u, v), v + self._horner2d(g["bp"], u, v)

    def invert_distortion(
        self,
        u_dist: Tensor,
        v_dist: Tensor,
        max_iter: int = 6,
        tol: float = 1e-6,
    ) -> "tuple[Tensor, Tensor]":
        """
        Invert SIP forward distortion map (u, v) -> distort(u, v).
        Uses a vectorized Newton solver with manual Jacobian for high performance.
        """
        if u_dist.numel() == 0 or not self.has_forward:
            return u_dist, v_dist
        if self._quad_inverse_fast:
            u_f = u_dist.reshape(-1)
            v_f = v_dist.reshape(-1)
            u_curr = u_f.clone()
            v_curr = v_f.clone()
            for _ in range(max_iter):
                uv = u_curr * v_curr
                u2 = u_curr * u_curr
                v2 = v_curr * v_curr
                xd = (
                    u_curr
                    + self._quad_a20 * u2
                    + self._quad_a11 * uv
                    + self._quad_a02 * v2
                )
                yd = (
                    v_curr
                    + self._quad_b20 * u2
                    + self._quad_b11 * uv
                    + self._quad_b02 * v2
                )
                du_res = xd - u_f
                dv_res = yd - v_f
                if bool(
                    torch.all(du_res.abs() < tol) and torch.all(dv_res.abs() < tol)
                ):
                    break

                dxd_du = 1.0 + 2.0 * self._quad_a20 * u_curr + self._quad_a11 * v_curr
                dxd_dv = self._quad_a11 * u_curr + 2.0 * self._quad_a02 * v_curr
                dyd_du = 2.0 * self._quad_b20 * u_curr + self._quad_b11 * v_curr
                dyd_dv = 1.0 + self._quad_b11 * u_curr + 2.0 * self._quad_b02 * v_curr

                det = dxd_du * dyd_dv - dxd_dv * dyd_du
                det_sign = torch.sign(det)
                det_sign = torch.where(
                    det_sign == 0, torch.ones_like(det_sign), det_sign
                )
                det = torch.where(det.abs() < 1e-15, det_sign * 1e-15, det)

                du_step = (dyd_dv * du_res - dxd_dv * dv_res) / det
                dv_step = (-dyd_du * du_res + dxd_du * dv_res) / det
                u_curr = u_curr - du_step
                v_curr = v_curr - dv_step
            return u_curr.reshape(u_dist.shape), v_curr.reshape(v_dist.shape)

        u_f = u_dist.reshape(-1)
        v_f = v_dist.reshape(-1)

        # Initial guess from inverse coefficients if available
        if self.has_inverse:
            u_curr, v_curr = self.undistort(u_f, v_f)
            # Most points are already within tolerance after AP/BP inversion.
            xd0, yd0 = self.distort(u_curr, v_curr)
            active_mask = (xd0 - u_f).abs() >= tol
            active_mask |= (yd0 - v_f).abs() >= tol
            if not bool(active_mask.any()):
                return u_curr.reshape(u_dist.shape), v_curr.reshape(v_dist.shape)
            active_idx = None
            if not bool(active_mask.all()):
                active_idx = torch.nonzero(active_mask, as_tuple=False).squeeze(1)
        else:
            u_curr, v_curr = u_f.clone(), v_f.clone()
            active_idx = None

        if active_idx is None:
            # Dense Newton path.
            for _ in range(max_iter):
                xd, yd, dxd_du, dxd_dv, dyd_du, dyd_dv = self._forward_and_jacobian(
                    u_curr, v_curr
                )

                du_res = xd - u_f
                dv_res = yd - v_f
                if torch.all(du_res.abs() < tol) and torch.all(dv_res.abs() < tol):
                    break

                det = dxd_du * dyd_dv - dxd_dv * dyd_du
                det_sign = torch.sign(det)
                det_sign = torch.where(
                    det_sign == 0, torch.ones_like(det_sign), det_sign
                )
                det = torch.where(det.abs() < 1e-15, det_sign * 1e-15, det)

                du_step = (dyd_dv * du_res - dxd_dv * dv_res) / det
                dv_step = (-dyd_du * du_res + dxd_du * dv_res) / det
                u_curr = u_curr - du_step
                v_curr = v_curr - dv_step
        else:
            # Sparse active-set Newton path.
            for _ in range(max_iter):
                if active_idx.numel() == 0:
                    break

                u_a = u_curr.index_select(0, active_idx)
                v_a = v_curr.index_select(0, active_idx)
                u_t = u_f.index_select(0, active_idx)
                v_t = v_f.index_select(0, active_idx)

                xd, yd, dxd_du, dxd_dv, dyd_du, dyd_dv = self._forward_and_jacobian(
                    u_a, v_a
                )

                du_res = xd - u_t
                dv_res = yd - v_t

                keep_active = du_res.abs() >= tol
                keep_active |= dv_res.abs() >= tol
                if not bool(keep_active.any()):
                    break

                local_idx = torch.nonzero(keep_active, as_tuple=False).squeeze(1)
                u_k = u_a.index_select(0, local_idx)
                v_k = v_a.index_select(0, local_idx)
                du_k = du_res.index_select(0, local_idx)
                dv_k = dv_res.index_select(0, local_idx)
                dxd_du_k = dxd_du.index_select(0, local_idx)
                dxd_dv_k = dxd_dv.index_select(0, local_idx)
                dyd_du_k = dyd_du.index_select(0, local_idx)
                dyd_dv_k = dyd_dv.index_select(0, local_idx)

                det = dxd_du_k * dyd_dv_k - dxd_dv_k * dyd_du_k
                det_sign = torch.sign(det)
                det_sign = torch.where(
                    det_sign == 0, torch.ones_like(det_sign), det_sign
                )
                det = torch.where(det.abs() < 1e-15, det_sign * 1e-15, det)

                du_step = (dyd_dv_k * du_k - dxd_dv_k * dv_k) / det
                dv_step = (-dyd_du_k * du_k + dxd_du_k * dv_k) / det

                global_idx = active_idx.index_select(0, local_idx)
                u_curr.index_copy_(0, global_idx, u_k - du_step)
                v_curr.index_copy_(0, global_idx, v_k - dv_step)
                active_idx = global_idx

        return u_curr.reshape(u_dist.shape), v_curr.reshape(v_dist.shape)
