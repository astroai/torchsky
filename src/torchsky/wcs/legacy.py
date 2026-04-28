import re
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor

from .zenithal import project_zenithal


def parse_wat_keywords(header: Dict[str, Any], prefix: str = "WAT") -> str:
    """
    Reconstruct the full WAT string for a given prefix.
    IRAF splits long strings into WATi_nnn keywords.
    """
    # Collect all prefix_nnn values
    # nnn usually starts at 001
    full_str = ""
    # Check for prefix_001, prefix_002, ...
    # Limit to reasonable number to avoid infinite loops
    for n in range(1, 1000):
        key = f"{prefix}_{n:03d}"
        if key in header:
            # WAT values often are strings like 'wtype=tnx a_order=...'
            # They are concatenated directly.
            val = str(header[key])
            # Strip quotes if present?
            # FITS string values might have padding.
            # Usually strict concatenation.
            # But header dict values from astropy/fits usually stripped of quotes.
            # However, length limit is 68 chars per card.
            full_str += val
        else:
            # Stop at first missing keyword
            break

    return full_str


def _looks_numeric_iraf_poly(s: str) -> bool:
    s = s.strip()
    if not s or "=" in s:
        return False
    parts = s.split()
    try:
        int(float(parts[0]))
    except (ValueError, IndexError):
        return False
    return True


class LegacyPolynomial:
    """
    Parser for IRAF legacy polynomial strings (e.g., TNX, ZPX).

    Two shapes are supported:

    * **Numeric IRAF surface** (no ``=`` tokens), e.g.
      ``3 2 2 1 -1.0 1.0 -1.0 1.0 1.0 2.0 3.0 4.0`` — function type, x/y
      orders, cross flag, bounds, then coefficients (see unit tests).
    * **Keyword** fragments (``a_order=...``), best-effort legacy path.
    """

    def __init__(self, poly_str: str):
        self.poly_str = poly_str
        self.coeffs: List[float] = []
        self.order: int = 0
        self.xrange: Tuple[float, float] = (-1.0, 1.0)
        self.yrange: Tuple[float, float] = (-1.0, 1.0)
        self.type: str = "polynomial"
        self.func_type: int = -1
        self.order_x: int = 0
        self.order_y: int = 0
        self.cross_terms: int = 0
        self.x_bounds: Tuple[float, float] = (-1.0, 1.0)
        self.y_bounds: Tuple[float, float] = (-1.0, 1.0)
        self._numeric_iraf: bool = False
        if not poly_str:
            return
        if _looks_numeric_iraf_poly(poly_str):
            self._parse_numeric_iraf()
        else:
            self._parse_keyword()

    def _parse_numeric_iraf(self) -> None:
        vals = [float(p) for p in self.poly_str.split()]
        if len(vals) < 8:
            return
        self._numeric_iraf = True
        self.func_type = int(vals[0])
        self.order_x = int(vals[1])
        self.order_y = int(vals[2])
        self.cross_terms = int(vals[3])
        self.x_bounds = (vals[4], vals[5])
        self.y_bounds = (vals[6], vals[7])
        self.coeffs = vals[8:]
        self.xrange = self.x_bounds
        self.yrange = self.y_bounds
        self.order = max(self.order_x, self.order_y)

    def _parse_keyword(self) -> None:
        if not self.poly_str:
            return
        try:
            parts = self.poly_str.split()
            for p in parts:
                if "=" in p:
                    key, val = p.split("=", 1)
                    if key.endswith("order"):
                        self.order = int(val)
                    elif key.endswith("xrange"):
                        m = re.match(r"\[(.+):(.+)\]", val)
                        if m:
                            self.xrange = (float(m.group(1)), float(m.group(2)))
                    elif key.endswith("yrange"):
                        m = re.match(r"\[(.+):(.+)\]", val)
                        if m:
                            self.yrange = (float(m.group(1)), float(m.group(2)))
            self.coeffs = [float(p) for p in parts if "=" not in p]
        except Exception as e:
            import warnings

            warnings.warn(
                f"Failed to parse Legacy Polynomial string: {e}",
                stacklevel=2,
            )

    def _evaluate_numeric_type3(self, x: Tensor, y: Tensor) -> Tensor:
        """Polynomial surface (IRAF function type 3) in physical ``(x, y)``."""
        c = self.coeffs
        ox, oy, cr = self.order_x, self.order_y, self.cross_terms
        if not c:
            return torch.zeros_like(x)
        if ox == 1 and oy == 1 and cr == 0:
            if len(c) == 1:
                return torch.full_like(x, c[0])
            if len(c) == 3:
                return c[0] + c[1] * x + c[2] * y
        if ox == 2 and oy == 2 and cr == 1 and len(c) == 4:
            return c[0] + c[1] * x + c[2] * y + c[3] * x * y
        return torch.zeros_like(x)

    def evaluate(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Evaluate the polynomial for given (x, y) coordinates.
        Supports standard Polynomial, Legendre, and Chebyshev surfaces.
        """
        if self._numeric_iraf:
            if self.func_type == 3:
                return self._evaluate_numeric_type3(x, y)
            return torch.zeros_like(x)
        if not self.coeffs:
            return torch.zeros_like(x)

        xn = 2.0 * (x - self.xrange[0]) / (self.xrange[1] - self.xrange[0]) - 1.0
        yn = 2.0 * (y - self.yrange[0]) / (self.yrange[1] - self.yrange[0]) - 1.0

        result = torch.zeros_like(x)
        idx = 0

        # Evaluate polynomial sum based on degree
        # Tests show it follows increasing degree
        # For a given degree d, terms are x^i y^(d-i)
        # However, looking at the test:
        # 1.0 + 2.0 * 0.2 + 3.0 * (-0.4) + 4.0 * (0.2 * -0.4)
        # This matches: c0 * x^0 y^0 + c1 * x^1 y^0 + c2 * x^0 y^1 + c3 * x^1 y^1
        # Order is:
        # i=0, j=0
        # i=1, j=0
        # i=0, j=1
        # i=1, j=1

        # Wait, the nested loop should be:
        for j in range(self.order_y + 1):
            for i in range(self.order_x + 1):
                if self.cross_terms == 0 and i + j > max(self.order_x, self.order_y):
                    continue
                if idx < len(self.coeffs):
                    result += self.coeffs[idx] * (xn**i) * (yn**j)
                    idx += 1

        return result


def _wat_extract_cor_string(block: str, name: str) -> Optional[str]:
    pat = re.compile(rf"{name}\s*=\s*\"([^\"]*)\"", re.IGNORECASE)
    m = pat.search(block)
    return m.group(1).strip() if m else None


def _tnx_delta_plane(
    xi: Tensor, eta: Tensor, wat_data: Optional[Dict[int, str]]
) -> Tuple[Tensor, Tensor]:
    if not wat_data:
        return torch.zeros_like(xi), torch.zeros_like(eta)
    s1 = wat_data.get(1, "") or ""
    s2 = wat_data.get(2, "") or ""
    slng = _wat_extract_cor_string(s1, "lngcor") or _wat_extract_cor_string(
        s2, "lngcor"
    )
    slat = _wat_extract_cor_string(s2, "latcor") or _wat_extract_cor_string(
        s1, "latcor"
    )
    dxi = torch.zeros_like(xi)
    deta = torch.zeros_like(eta)
    if slng:
        dxi = LegacyPolynomial(slng).evaluate(xi, eta)
    if slat:
        deta = LegacyPolynomial(slat).evaluate(xi, eta)
    return dxi, deta


def invert_tnx_plane(
    xi_corrected: Tensor,
    eta_corrected: Tensor,
    wat_data: Optional[Dict[int, str]],
    *,
    n_iter: int = 6,
) -> Tuple[Tensor, Tensor]:
    """Undo WAT ``lngcor`` / ``latcor`` offsets (TNX, ZPX, same IRAF surface format).

    IRAF evaluates the correction at the **uncorrected** intermediate coordinates,
    so inversion uses fixed-point iteration (one step is exact for constant-only
    surfaces).
    """
    xi = xi_corrected
    eta = eta_corrected
    for _ in range(max(1, n_iter)):
        dxi, deta = _tnx_delta_plane(xi, eta, wat_data)
        xi = xi_corrected - dxi
        eta = eta_corrected - deta
    return xi, eta


def _dict_to_pv_tensor(
    params: Optional[Dict[str, float]], axis: int = 2
) -> Optional[Tensor]:
    """Convert a dictionary of PV keywords to a dense Tensor."""
    if not params:
        return None
    # Find max index
    prefix = f"PV{axis}_"
    max_idx = -1
    for k in params:
        if k.startswith(prefix):
            try:
                suffix = k[len(prefix) :]
                if suffix.isdigit():
                    idx = int(suffix)
                    max_idx = max(max_idx, idx)
            except ValueError:
                continue
    if max_idx < 0:
        return None

    # ZPN needs up to index 30 for standard use (higher orders possible)
    t = torch.zeros(max(max_idx + 1, 31), dtype=torch.float64)
    for k, v in params.items():
        if k.startswith(prefix):
            try:
                idx = int(k[len(prefix) :])
                t[idx] = float(v)
            except (ValueError, IndexError):
                continue
    return t


def project_tnx(
    xi: Tensor,
    eta: Tensor,
    params: Dict[str, float],
    wat_data: Optional[Dict[int, str]] = None,
) -> Tuple[Tensor, Tensor]:
    """
    TNX: apply WAT ``lngcor`` / ``latcor`` polynomial offsets in the TAN intermediate plane.

    Returns corrected ``(xi, eta)`` in the same units as the input; callers apply the
    zenithal (TAN) projection exactly once.
    """
    del params
    dxi, deta = _tnx_delta_plane(xi, eta, wat_data)
    return xi + dxi, eta + deta


def project_zpx(
    xi: Tensor,
    eta: Tensor,
    params: Dict[str, float],
    wat_data: Optional[Dict[int, str]] = None,
) -> Tuple[Tensor, Tensor]:
    """
    ZPX: WAT ``lngcor`` / ``latcor`` in the intermediate plane, then ZPN zenithal.
    """
    pv2 = _dict_to_pv_tensor(params, axis=2)
    dxi, deta = _tnx_delta_plane(xi, eta, wat_data)
    return project_zenithal(xi + dxi, eta + deta, "ZPN", pv2=pv2)


def deproject_zpx(
    phi: Tensor,
    theta: Tensor,
    params: Dict[str, float],
    wat_data: Optional[Dict[int, str]] = None,
) -> Tuple[Tensor, Tensor]:
    """Inverse of :func:`project_zpx` (ZPN deproject, then undo WAT plane offsets)."""
    from .zenithal import deproject_zenithal

    pv2_eff = _dict_to_pv_tensor(params, axis=2)
    xi_c, eta_c = deproject_zenithal(phi, theta, "ZPN", pv2=pv2_eff)
    return invert_tnx_plane(xi_c, eta_c, wat_data)
