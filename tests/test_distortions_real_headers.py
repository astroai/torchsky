"""Real SIP (HST ACS) and TPV (DECam) headers from the distortions guide against Astropy/WCSLIB.

The header dicts are parsed from ``docs/coordinates/distortions.md`` so the
documented snippets and this oracle cannot drift apart. Astropy is a test-only
parity oracle here.
"""

from __future__ import annotations

import ast
from pathlib import Path
import re
import warnings

import numpy as np
import pytest
import torch

from torchsky.wcs import WCS

astropy_wcs = pytest.importorskip("astropy.wcs")

GUIDE = Path(__file__).resolve().parents[1] / "docs" / "coordinates" / "distortions.md"
MAS_PER_DEG = 3.6e6


def _header_from_guide(name: str) -> dict:
    text = GUIDE.read_text(encoding="utf-8")
    match = re.search(rf"^{name} = (\{{.*?^\}})", text, flags=re.S | re.M)
    assert match, f"{name} not found in {GUIDE.name}"
    body = re.sub(r"#.*$", "", match.group(1), flags=re.M)
    return ast.literal_eval(body)


@pytest.mark.parametrize("name", ["acs_wfc1", "decam_s29"])
def test_real_distortion_header_matches_astropy(name: str) -> None:
    header = _header_from_guide(name)
    nx, ny = header["NAXIS1"], header["NAXIS2"]
    xs, ys = np.meshgrid(np.linspace(0, nx - 1, 41), np.linspace(0, ny - 1, 81))
    x, y = xs.ravel(), ys.ravel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        oracle = astropy_wcs.WCS(header)
    ra_ref, dec_ref = oracle.all_pix2world(x, y, 0)

    wcs = WCS(header)
    ra, dec = wcs.pixel_to_world(torch.from_numpy(x), torch.from_numpy(y))
    dra = (ra.numpy() - ra_ref) * np.cos(np.radians(dec_ref))
    separation_mas = np.hypot(dra, dec.numpy() - dec_ref) * MAS_PER_DEG
    assert separation_mas.max() < 1e-3  # sub-microarcsecond in practice

    x_back, y_back = wcs.world_to_pixel(
        torch.from_numpy(ra_ref), torch.from_numpy(dec_ref)
    )
    assert np.abs(x_back.numpy() - x).max() < 1e-6
    assert np.abs(y_back.numpy() - y).max() < 1e-6
