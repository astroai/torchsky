"""Round-trip coverage for every supported FITS projection family.

The harness compares each projection against astropy; these package tests pin the
internal pixel<->world contract so a regression fails in the fast lane too. The
reference latitude is deliberately non-zero: center-native projections
(cylindrical, all-sky, pseudocylindrical, polyconic) rotate the native frame,
and an inverse that assumes a pure latitude shift looks correct at delta0 = 0
and fails everywhere else.
"""

from __future__ import annotations

import math

import pytest
import torch

from torchsky.wcs import WCS

ZENITHAL = ("TAN", "SIN", "ARC", "STG", "ZEA", "AIR", "ZPN", "SZP", "AZP")
CYLINDRICAL = ("CEA", "MER", "CYP", "CAR")
ALLSKY = ("AIT", "MOL", "HPX", "SFL", "XPH")
CONIC = ("COP", "COE", "COD", "COO")
POLYCONIC = ("BON", "PCO")
QUADCUBE = ("TSC", "CSC", "QSC")
PSEUDOCYLINDRICAL = ("PAR",)

ALL_CODES = (
    ZENITHAL
    + CYLINDRICAL
    + ALLSKY
    + CONIC
    + POLYCONIC
    + QUADCUBE
    + PSEUDOCYLINDRICAL
)


def _header(code: str, *, crval2: float, size: int = 64, cdelt: float = 1.0) -> dict:
    header = {
        "NAXIS": 2,
        "NAXIS1": size,
        "NAXIS2": size,
        "CTYPE1": f"RA---{code}",
        "CTYPE2": f"DEC--{code}",
        "CRPIX1": (size + 1) / 2.0,
        "CRPIX2": (size + 1) / 2.0,
        "CRVAL1": 120.0,
        "CRVAL2": crval2,
        "CDELT1": -cdelt,
        "CDELT2": cdelt,
    }
    if code == "ZPN":
        header.update({"PV2_1": 0.0, "PV2_2": 1.0, "PV2_3": 0.0, "PV2_4": 0.0})
    elif code == "SZP":
        header.update({"PV2_1": 0.0, "PV2_2": 0.0, "PV2_3": 1.0})
    elif code == "AZP":
        header.update({"PV2_1": 0.0, "PV2_2": 0.0})
    elif code == "CYP":
        header["PV2_1"] = 0.5
    elif code in {"COD", "COE", "COO", "COP"}:
        header.update({"PV2_1": 45.0, "PV2_2": 0.5})
    elif code == "BON":
        header["PV2_1"] = 45.0
    elif code == "PCO":
        header.update({"PV2_1": 45.0, "PV2_2": 0.5})
    return header


def _mean_error_at_reference(code: str, crval2: float) -> float:
    header = _header(code, crval2=crval2)
    wcs = WCS(header)
    x = torch.tensor([header["CRPIX1"] - 1.0], dtype=torch.float64)
    y = torch.tensor([header["CRPIX2"] - 1.0], dtype=torch.float64)
    lon, lat = wcs.pixel_to_world(x, y)
    x_back, y_back = wcs.world_to_pixel(lon, lat)
    return float(torch.hypot(x_back - x, y_back - y).max())


@pytest.mark.parametrize("crval2", [0.0, 20.0])
@pytest.mark.parametrize("code", ALL_CODES)
def test_reference_point_roundtrips(code: str, crval2: float) -> None:
    error = _mean_error_at_reference(code, crval2)
    assert error < 1e-6, f"{code} at CRVAL2={crval2}: {error:.3e} px"


@pytest.mark.parametrize("crval2", [0.0, 20.0])
@pytest.mark.parametrize("code", ALL_CODES)
def test_projection_grid_roundtrips(code: str, crval2: float) -> None:
    header = _header(code, crval2=crval2)
    wcs = WCS(header)
    size = int(header["NAXIS1"])
    grid = torch.arange(size, dtype=torch.float64)
    x, y = torch.meshgrid(grid, grid, indexing="xy")
    x = x.reshape(-1)
    y = y.reshape(-1)

    lon, lat = wcs.pixel_to_world(x, y)
    finite = torch.isfinite(lon) & torch.isfinite(lat)
    assert int(finite.sum()) > size * size // 2, f"{code}: too few finite pixels"

    x_back, y_back = wcs.world_to_pixel(lon[finite], lat[finite])
    error = torch.hypot(x_back - x[finite], y_back - y[finite])
    max_error = float(error.max())

    if code == "CSC":
        # Quad-cube seam pixels are conditioned, identically to astropy; the
        # harness exempts them from round-trip thresholds.
        assert max_error < 0.05, f"CSC seam error {max_error:.3e} px"
    else:
        assert max_error < 1e-6, f"{code} at CRVAL2={crval2}: {max_error:.3e} px"


def test_native_rotation_maps_reference_to_native_equator() -> None:
    """The center-native inverse must put (alpha0, delta0) at native (0, 0)."""
    header = _header("MER", crval2=20.0)
    wcs = WCS(header)
    phi, theta = wcs._inverse_spherical_rotation(
        torch.tensor([120.0], dtype=torch.float64),
        torch.tensor([20.0], dtype=torch.float64),
        "center",
    )
    assert math.isclose(float(phi) % 360.0, 0.0, abs_tol=1e-9)
    assert math.isclose(float(theta), 0.0, abs_tol=1e-9)
