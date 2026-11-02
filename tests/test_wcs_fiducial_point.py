"""Native fiducial point (PV1_1/PV1_2, Paper II 2.5) parity against WCSLIB."""

import warnings

import numpy as np
import pytest
import torch

astropy_wcs = pytest.importorskip("astropy.wcs")

from torchsky.wcs import WCS  # noqa: E402

CASES = [
    ("SIN", {}, {"PV1_1": 0.0, "PV1_2": 85.0}),
    ("ARC", {}, {"PV1_1": 30.0, "PV1_2": 70.0}),
    ("STG", {}, {"PV1_1": -20.0, "PV1_2": 75.0}),
    ("ZEA", {}, {"PV1_1": 10.0, "PV1_2": 60.0}),
    ("CAR", {}, {"PV1_1": 10.0, "PV1_2": 20.0}),
    ("CAR", {}, {"PV1_1": 0.0, "PV1_2": 15.0}),
    ("MER", {}, {"PV1_1": -15.0, "PV1_2": 25.0}),
    ("CEA", {"PV2_1": 1.0}, {"PV1_1": 12.0, "PV1_2": -10.0}),
    ("MOL", {}, {"PV1_1": 8.0, "PV1_2": 10.0}),
    ("AIT", {}, {"PV1_1": 8.0, "PV1_2": 10.0}),
    ("COE", {"PV2_1": 30.0, "PV2_2": 15.0}, {"PV1_1": 10.0, "PV1_2": 40.0}),
    ("COD", {"PV2_1": 30.0, "PV2_2": 15.0}, {"PV1_1": 10.0, "PV1_2": 40.0}),
    ("COP", {"PV2_1": 45.0, "PV2_2": 10.0}, {"PV1_1": 0.0, "PV1_2": 55.0}),
]


def _hdr(proj, pv2, pv1, crval, lonpole=None):
    h = {
        "NAXIS": 2,
        "CRPIX1": 50.0,
        "CRPIX2": 40.0,
        "CDELT1": -0.2,
        "CDELT2": 0.2,
        "CTYPE1": "RA---" + proj,
        "CTYPE2": "DEC--" + proj,
        "CRVAL1": crval[0],
        "CRVAL2": crval[1],
    }
    h.update(pv2)
    h.update(pv1)
    if lonpole is not None:
        h["LONPOLE"] = lonpole
    return h


def _sep(r1, d1, r2, d2):
    dr = ((r1 - r2 + 180.0) % 360.0 - 180.0) * np.cos(np.deg2rad(d2))
    return np.hypot(dr, d1 - d2)


@pytest.mark.parametrize("crval", [(150.0, 20.0), (30.0, -35.0)])
@pytest.mark.parametrize("proj,pv2,pv1", CASES)
def test_fiducial_point_matches_astropy(proj, pv2, pv1, crval):
    h = _hdr(proj, pv2, pv1, crval)
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 100, 150)
    y = rng.uniform(0, 80, 150)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = astropy_wcs.WCS(h)
        ra, dec = a.all_pix2world(x, y, 0)
    ok = np.isfinite(ra)
    assert ok.sum() > 100
    t = WCS(h)
    pix = torch.tensor(np.c_[x, y])
    two = t.pixel_to_world(torch.tensor(x), torch.tensor(y))
    stacked = t.pixel_to_world(pix).numpy()
    diff = t.pixel_to_world_differentiable(pix).detach().numpy()
    for r, d in (
        (two[0].numpy(), two[1].numpy()),
        (stacked[:, 0], stacked[:, 1]),
        (diff[:, 0], diff[:, 1]),
    ):
        assert np.nanmax(_sep(r, d, ra, dec)[ok]) < 1e-9
    # inverse, 2-arg and stacked and differentiable
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        px, py = a.all_world2pix(ra[ok], dec[ok], 0)
    q = t.world_to_pixel(torch.tensor(ra[ok]), torch.tensor(dec[ok]))
    assert np.max(np.hypot(q[0].numpy() - px, q[1].numpy() - py)) < 1e-6
    w = torch.tensor(np.c_[ra[ok], dec[ok]])
    qs = t.world_to_pixel(w).numpy()
    qd = t.world_to_pixel_differentiable(w).detach().numpy()
    for arr in (qs, qd):
        assert np.max(np.hypot(arr[:, 0] - px, arr[:, 1] - py)) < 1e-6


def test_fiducial_point_with_lonpole_latpole():
    h = _hdr("CAR", {}, {"PV1_1": 10.0, "PV1_2": 20.0}, (150.0, 20.0), lonpole=170.0)
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 100, 80)
    y = rng.uniform(0, 80, 80)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ra, dec = astropy_wcs.WCS(h).all_pix2world(x, y, 0)
    r, d = WCS(h).pixel_to_world(torch.tensor(x), torch.tensor(y))
    assert np.max(_sep(r.numpy(), d.numpy(), ra, dec)) < 1e-9
