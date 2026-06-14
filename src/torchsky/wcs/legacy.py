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
    for n in range(1, 100):
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
    Base class for IRAF polynomials (Chebyshev, Legendre, etc.)
    """

    def __init__(self, param_str: str):
        self.func_type = 0
        self.order_x = 0
        self.order_y = 0
        self.cross_terms = 0
        self.x_bounds = (0.0, 0.0)
        self.y_bounds = (0.0, 0.0)
        self.coeffs: List[float] = []

        self._parse(param_str)

    def _parse(self, s: str):
        tokens = s.strip().split()
        if not tokens:
            return

        try:
            self.func_type = int(tokens[0])
            self.order_x = int(tokens[1])
            self.order_y = int(tokens[2])
            self.cross_terms = int(tokens[3])

            self.x_bounds = (float(tokens[4]), float(tokens[5]))
            self.y_bounds = (float(tokens[6]), float(tokens[7]))

            self.coeffs = [float(x) for x in tokens[8:]]

        except (ValueError, IndexError) as e:
            import warnings

            warnings.warn(
                f"Failed to parse Legacy Polynomial string: {e}",
                stacklevel=2,
            )

    def evaluate(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Evaluate polynomial P(x, y).
        """
        nx = (2.0 * x - (self.x_bounds[1] + self.x_bounds[0])) / (
            self.x_bounds[1] - self.x_bounds[0]
        )
        ny = (2.0 * y - (self.y_bounds[1] + self.y_bounds[0])) / (
            self.y_bounds[1] - self.y_bounds[0]
        )

        bx = self._compute_basis(nx, self.order_x, self.func_type)
        by = self._compute_basis(ny, self.order_y, self.func_type)

        result = torch.zeros_like(x)
        k = 0

        for j in range(self.order_y):
            for i in range(self.order_x):
                if self.cross_terms == 0 and i > 0 and j > 0:
                    continue
                if self.cross_terms == 2 and i + j >= max(self.order_x, self.order_y):
                    continue

                if k < len(self.coeffs):
                    term = self.coeffs[k] * bx[i] * by[j]
                    result += term
                    k += 1

        return result

    def _compute_basis(self, u: Tensor, order: int, ftype: int) -> List[Tensor]:
        """Generate basis polynomials [P_0(u), ..., P_n(u)]."""
        basis: List[Tensor] = []
        if order < 1:
            return basis

        basis.append(torch.ones_like(u))
        if order == 1:
            return basis

        basis.append(u)

        for n in range(2, order):
            if ftype == 1:  # Chebyshev
                bn = 2.0 * u * basis[-1] - basis[-2]
            elif ftype == 2:  # Legendre
                k = n
                bn = ((2 * k - 1) * u * basis[-1] - (k - 1) * basis[-2]) / k
            else:
                bn = u * basis[-1]
            basis.append(bn)

        return basis


_ZPX_PATTERN = re.compile(r"projp(\d+)\s*=\s*(\S+)", re.IGNORECASE)


def extract_zpx_params(wat_data: Dict[int, str]) -> Dict[str, float]:
    """
    Extract ZPN coefficients (projp_i) from WAT strings and return as PV2_i.
    """
    params = {}
    for axis, wstr in wat_data.items():
        matches = _ZPX_PATTERN.findall(wstr)
        for m in matches:
            idx = int(m[0])
            val = float(m[1])
            params[f"PV2_{idx}"] = val
    return params


def extract_tnx_coeffs(wat_str: str, key: str) -> Optional[str]:
    """
    Extract value for 'lngcor' or 'latcor' from WAT string.
    Format: key = " value " or key = value
    """
    pat = re.compile(rf'{key}\s*=\s*(?:"([^"]*)"|([^"\s]+))', re.IGNORECASE)
    match = pat.search(wat_str)
    if match:
        val_quoted = match.group(1)
        val_unquoted = match.group(2)
        return val_quoted if val_quoted is not None else val_unquoted
    return None


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
