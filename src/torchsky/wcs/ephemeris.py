"""Optional JPL DE barycentric body geometry. Not imported by ``torchsky.wcs``."""

from __future__ import annotations

import os
from typing import Any

import torch

from .catalog_state import CatalogState
from .frames import unit_to_radec

try:
    from jplephem.spk import SPK
except ImportError:  # pragma: no cover - optional extra
    SPK = None

# DE430/DE440 segments from the solar-system barycenter. Planets are
# barycenter then planet; Earth/Moon go through EMB (NAIF 3).
_BODY_CHAIN = {
    "sun": ((0, 10),),
    "mercury": ((0, 1), (1, 199)),
    "venus": ((0, 2), (2, 299)),
    "earth": ((0, 3), (3, 399)),
    "moon": ((0, 3), (3, 301)),
    "mars": ((0, 4), (4, 499)),
    "jupiter": ((0, 5), (5, 599)),
    "saturn": ((0, 6), (6, 699)),
    "uranus": ((0, 7), (7, 799)),
    "neptune": ((0, 8), (8, 899)),
}


def _de_path(path: str | None) -> str:
    if path:
        return path
    env = os.environ.get("TORCHSKY_DE440_PATH") or os.environ.get("TORCHSKY_DE_PATH")
    if env:
        return env
    raise FileNotFoundError(
        "JPL DE kernel missing; set TORCHSKY_DE440_PATH or pass path= to get_body"
    )


def _spk_compute(kernel: Any, chain: tuple[tuple[int, int], ...], jd: float):
    xyz = None
    for center, target in chain:
        try:
            part = kernel[center, target].compute(jd)
        except KeyError as exc:
            raise ValueError(f"kernel has no SPK segment {center}→{target}") from exc
        xyz = part if xyz is None else xyz + part
    return xyz


def get_body(
    name: str,
    tt_mjd: Any,
    *,
    path: str | None = None,
) -> CatalogState:
    """Geometric SSB direction of a solar-system body from a JPL DE kernel.

    Requires the ``ephem`` extra (``jplephem``) and a DE430/DE440 SPK file.
    This is not apparent place and does not apply light-time or aberration.
    """
    if SPK is None:
        raise ImportError("get_body requires jplephem; install torchsky[ephem]")
    body = str(name).strip().lower()
    if body not in _BODY_CHAIN:
        raise ValueError(f"unsupported get_body target {name!r}")
    kernel_path = _de_path(path)
    if not os.path.isfile(kernel_path):
        raise FileNotFoundError(f"JPL DE kernel not found: {kernel_path}")

    tt = torch.as_tensor(tt_mjd, dtype=torch.float64)
    if tt.numel() != 1:
        raise ValueError("get_body requires a scalar tt_mjd")
    jd = float(tt.reshape(())) + 2400000.5
    kernel = SPK.open(kernel_path)
    try:
        xyz_km = _spk_compute(kernel, _BODY_CHAIN[body], jd)
    finally:
        kernel.close()
    vec = torch.as_tensor(xyz_km, dtype=torch.float64)
    lon, lat = unit_to_radec(vec)
    return CatalogState(lon=lon, lat=lat, frame="ICRS", epoch=2000.0)
