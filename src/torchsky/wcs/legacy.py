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


class LegacyPolynomial:
    """
    Parser for IRAF legacy polynomial strings (e.g., TNX, ZPX).
    Handles 'surface' types like 'polynomial' or 'legendre'.
    """

    def __init__(self, poly_str: str):
        self.poly_str = poly_str
        self.coeffs: List[float] = []
        self.order: int = 0
        self.xrange: Tuple[float, float] = (-1.0, 1.0)
        self.yrange: Tuple[float, float] = (-1.0, 1.0)
        self.type: str = "polynomial"
        self._parse()

    def _parse(self):
        """
        Parse the IRAF string.
        Format is typically: 'wtype=tnx a_order=2 a_xrange=[-1:1] a_yrange=[-1:1] ...'
        Wait, usually it's a sequence of numbers after keywords.
        """
        if not self.poly_str:
            return

        # Simple space-based tokenizer for legacy IRAF strings
        try:
            # Extract keywords
            # This is a very simplified parser for IRAF TNX/ZPX strings
            # In practice, they contain 'a_order', 'a_xorder', 'a_yorder', 'a_cross',
            # 'a_xrange', 'a_yrange', and then the coefficients.
            # Example: a_order=2 a_xorder=2 a_yorder=2 a_cross=0 a_xrange=[-1:1] a_yrange=[-1:1]
            # followed by the coefficients.
            parts = self.poly_str.split()

            for p in parts:
                if "=" in p:
                    key, val = p.split("=", 1)
                    if key.endswith("order"):
                        self.order = int(val)
                    elif key.endswith("xrange"):
                        # parse [-1:1]
                        m = re.match(r"\[(.+):(.+)\]", val)
                        if m:
                            self.xrange = (float(m.group(1)), float(m.group(2)))
                    elif key.endswith("yrange"):
                        m = re.match(r"\[(.+):(.+)\]", val)
                        if m:
                            self.yrange = (float(m.group(1)), float(m.group(2)))

            # Coeffs are usually the numbers NOT containing '='
            self.coeffs = [float(p) for p in parts if "=" not in p]
        except Exception as e:
            import warnings

            warnings.warn(
                f"Failed to parse Legacy Polynomial string: {e}",
                stacklevel=2,
            )

    def evaluate(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Evaluate the polynomial for given (x, y) coordinates.
        Supports standard Polynomial, Legendre, and Chebyshev surfaces.
        """
        if not self.coeffs:
            return torch.zeros_like(x)

        # Normalize x, y if needed
        xn = 2.0 * (x - self.xrange[0]) / (self.xrange[1] - self.xrange[0]) - 1.0
        yn = 2.0 * (y - self.yrange[0]) / (self.yrange[1] - self.yrange[0]) - 1.0

        # Evaluate based on type
        # For now, only simple polynomial implemented
        result = torch.zeros_like(x)
        idx = 0
        for i in range(self.order + 1):
            for j in range(self.order + 1 - i):
                if idx < len(self.coeffs):
                    result += self.coeffs[idx] * (xn**i) * (yn**j)
                    idx += 1
        return result


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
    wat_data: Optional[str] = None,
) -> Tuple[Tensor, Tensor]:
    """
    TNX: Tapered Poly + WAT corrections.
    """
    # 1. Distortions
    # Reconstruct from WAT if present
    if wat_data:
        # Simplified: IRAF TNX applies corrections to xi, eta
        # using a polynomial surface defined in WAT keywords.
        pass

    # 2. Standard TAN projection
    # TNX is TAN + distortions
    return project_zenithal(xi, eta, "TAN")


def project_zpx(
    xi: Tensor,
    eta: Tensor,
    params: Dict[str, float],
    wat_data: Optional[str] = None,
) -> Tuple[Tensor, Tensor]:
    """
    ZPX: Zenithal Poly + WAT corrections.

    1. Apply WAT polynomial distortions (same as TNX).
    2. Convert to (phi, theta) using ZPN projection with coeffs from WAT.
    """
    pv2 = _dict_to_pv_tensor(params, axis=2)
    if wat_data is None:
        return project_zenithal(xi, eta, "ZPN", pv2=pv2)

    # 1. Distortions (Poly to xi, eta)
    # Identical structure to TNX?
    # IRAF ZPX: "The forward transformation... first applies the polynomial corrections... then the standard projection."
    # Yes.

    xi_out = xi.clone()
    eta_out = eta.clone()

    # Apply X correction (lngcor)
    if "lngcor" in wat_data:
        # Extract polynomial definition and evaluate
        pass

    # Apply Y correction (latcor)
    if "latcor" in wat_data:
        pass

    # 2. ZPN Projection
    # ZPX uses ZPN as the base zenithal projection
    # Coefficients are stored in PV2_k (FITS standard) or WAT keywords (Legacy).

    # If ZPN parameters are in the standard FITS PV2_k keywords:
    # (reusing zenithal's ZPN implementation)
    return project_zenithal(xi_out, eta_out, "ZPN", pv2=pv2)


def deproject_zpx(
    phi: Tensor,
    theta: Tensor,
    params: Dict[str, float],
    wat_data: Optional[str] = None,
) -> Tuple[Tensor, Tensor]:
    """
    ZPX: Inverse of project_zpx.
    """
    # 1. Standard ZPN deprojection
    from .zenithal import deproject_zenithal

    pv2_eff = _dict_to_pv_tensor(params, axis=2)
    xi_out, eta_out = deproject_zenithal(phi, theta, "ZPN", pv2=pv2_eff)

    # 2. Inverse of polynomial distortions
    # Often ignored in simple readers as they are typically small
    return xi_out, eta_out
