"""Module-level helpers shared by :class:`torchsky.wcs.WCS` and its mixins."""

from __future__ import annotations

import torch
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
)


if TYPE_CHECKING:
    pass


_HAS_TORCH_SINCOS = hasattr(torch, "sincos")


# Benchmark-backed small-batch CPU crossover points for the center-equator C++
# helpers. Above these sizes, the vectorized Torch path is typically as fast or
# faster, so we keep the native path for throughput-oriented workloads.
_CENTER_CPP_P2W_THRESHOLDS = {
    "AIT": 4096,
    "CAR": 1024,
    "CEA": 512,
    "HPX": 1024,
    "MER": 512,
    "SFL": 1024,
}


_CENTER_CPP_W2P_THRESHOLDS = {
    "AIT": 1024,
    "CAR": 512,
    "CEA": 2048,
    "HPX": 1024,
    "MER": 1024,
    "SFL": 2048,
}


# Fused-first only for zenithal pole two-arg p2w below this N. Vectorized
# cpp_direct stays primary above it so CEA/CAR and N>=10K are not regressable.
_ZENITHAL_FUSED_P2W_MAX = 1536


def _ctype_without_projection(ctype: str) -> str:
    """Strip FITS projection / SIP suffixes, leaving the coordinate type."""
    text = str(ctype).strip()
    upper = text.upper()
    if upper.endswith("-SIP"):
        text = text[:-4]
        upper = upper[:-4]
    if len(upper) >= 5 and upper[-4] == "-":
        return text[:-4].rstrip("-")
    return text


_CELESTIAL_LONGITUDE_PREFIXES = ("RA--", "GLON", "ELON", "HLON", "SLON")


_CELESTIAL_LATITUDE_PREFIXES = ("DEC-", "GLAT", "ELAT", "HLAT", "SLAT")


def _celestial_axis_role(ctype: str) -> str | None:
    upper = str(ctype).strip().upper()
    if upper.startswith(_CELESTIAL_LONGITUDE_PREFIXES):
        return "longitude"
    if upper.startswith(_CELESTIAL_LATITUDE_PREFIXES):
        return "latitude"
    return None


def _ctype_projection_code(ctype: str) -> str:
    upper = str(ctype).strip().upper()
    if upper.endswith("-SIP"):
        upper = upper[:-4]
    return upper[-3:] if len(upper) >= 3 else ""


def _bound_cpp(fn: Callable[..., Any] | None) -> Callable[..., Any]:
    # Direct-dispatch kinds are only selected when the matching binding exists.
    assert fn is not None
    return fn


# Half a pixel is where rounding a coordinate to the nearest pixel centre starts to
# pick the wrong pixel, so float32 can no longer represent the grid.
_MPS_PIXEL_IDENTIFICATION_LIMIT_PX = 0.5


def _float32_ulp(x: float) -> float:
    """Spacing of float32 values at ``x`` (> 0)."""
    t = torch.tensor(float(x), dtype=torch.float32)
    return float(torch.nextafter(t, torch.tensor(float("inf"))) - t)
