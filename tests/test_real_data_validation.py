from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from astropy.wcs import WCS as AstropyWCS

import torchsky
import torchfits


astropy = pytest.importorskip("astropy")
ASTROPY_ROOT = Path(astropy.__path__[0])
SIP_WCS_FIXTURE = ASTROPY_ROOT / "nddata" / "tests" / "data" / "sip-wcs.fits"


def test_local_sip_wcs_fixture_matches_astropy_world_coordinates() -> None:
    awcs = AstropyWCS(torchfits.read_header(SIP_WCS_FIXTURE.as_posix()))
    twcs = torchsky.get_wcs(SIP_WCS_FIXTURE.as_posix(), hdu=0)
    xy = np.array([[0.0, 0.0], [20.0, 10.0], [80.0, 40.0]], dtype=np.float64)

    ra_ast, dec_ast = awcs.all_pix2world(xy[:, 0], xy[:, 1], 0)
    ra_t, dec_t = twcs.pixel_to_world(
        torch.from_numpy(xy[:, 0]), torch.from_numpy(xy[:, 1]), origin=0
    )

    np.testing.assert_allclose(ra_t.cpu().numpy(), ra_ast, atol=5e-6, rtol=0.0)
    np.testing.assert_allclose(dec_t.cpu().numpy(), dec_ast, atol=5e-6, rtol=0.0)
