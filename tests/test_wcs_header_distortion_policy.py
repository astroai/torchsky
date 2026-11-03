"""CTYPE decides which header distortion applies; contradictions fail closed.

Astropy/WCSLIB 8.x is the oracle for the one compatibility rule torchsky keeps:
``TAN`` with ``PVi_m`` for some ``m >= 5`` is pre-2012 SCAMP TPV, while ``TAN``
with only ``PV1_0..4`` keeps its standard (fiducial point / pole) meaning.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch
from astropy.wcs import WCS as AstropyWCS

from torchsky.wcs import WCS, WCSHeaderWarning

BASE = {
    "NAXIS": 2,
    "NAXIS1": 100,
    "NAXIS2": 100,
    "CRPIX1": 50.0,
    "CRPIX2": 50.0,
    "CRVAL1": 10.0,
    "CRVAL2": 20.0,
    "CD1_1": -1.0e-3,
    "CD1_2": 0.0,
    "CD2_1": 0.0,
    "CD2_2": 1.0e-3,
}
SIP = {"A_ORDER": 2, "A_2_0": 1.0e-4, "B_ORDER": 2, "B_0_2": 1.0e-4}
SCAMP = {
    "PV1_0": 0.0,
    "PV1_1": 1.0,
    "PV1_4": 1.0e-2,
    "PV1_7": 5.0e-3,
    "PV2_0": 0.0,
    "PV2_1": 1.0,
    "PV2_4": -1.0e-2,
    "PV2_7": 4.0e-3,
}
PIXELS = np.array([[0.0, 0.0], [49.0, 49.0], [98.0, 3.0], [7.0, 91.0]])


def _ctype(code: str) -> dict[str, str]:
    return {"CTYPE1": f"RA---{code}", "CTYPE2": f"DEC--{code}"}


def _torch_world(wcs: WCS) -> np.ndarray:
    pix = torch.as_tensor(PIXELS, dtype=torch.float64)
    ra, dec = wcs.pixel_to_world(pix[:, 0], pix[:, 1])
    return np.stack([ra.numpy(), dec.numpy()], axis=1)


def _astropy_world(header: dict) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return AstropyWCS(header).all_pix2world(PIXELS, 0)


def test_sip_without_sip_ctype_raises() -> None:
    with pytest.raises(ValueError, match="-SIP"):
        WCS({**BASE, **_ctype("TAN"), **SIP})


def test_sip_on_tpv_ctype_raises() -> None:
    with pytest.raises(ValueError, match="-SIP"):
        WCS({**BASE, **_ctype("TPV"), **SCAMP, **SIP})


def test_declared_sip_matches_astropy() -> None:
    header = {**BASE, **_ctype("TAN-SIP"), **SIP}
    np.testing.assert_allclose(
        _torch_world(WCS(header)), _astropy_world(header), rtol=0, atol=1e-10
    )


def test_sip_plus_scamp_pv_raises() -> None:
    with pytest.raises(ValueError, match="SIP and also carries SCAMP"):
        WCS({**BASE, **_ctype("TAN-SIP"), **SIP, **SCAMP})


def test_tan_with_high_order_pv_is_legacy_tpv_with_warning() -> None:
    header = {**BASE, **_ctype("TAN"), **SCAMP}
    with pytest.warns(WCSHeaderWarning, match="pre-2012 SCAMP TPV"):
        legacy = WCS(header)
    with warnings.catch_warnings():
        warnings.simplefilter("error", WCSHeaderWarning)
        explicit = WCS({**BASE, **_ctype("TPV"), **SCAMP})
    world = _torch_world(legacy)
    np.testing.assert_array_equal(world, _torch_world(explicit))
    # WCSLIB reads the same TAN header as TPV.
    np.testing.assert_allclose(world, _astropy_world(header), rtol=0, atol=1e-10)
    out = legacy.to_header()
    assert out["CTYPE1"] == "RA---TPV" and out["CTYPE2"] == "DEC--TPV"


def test_legacy_tpv_warning_can_be_made_an_error() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", WCSHeaderWarning)
        with pytest.raises(WCSHeaderWarning):
            WCS({**BASE, **_ctype("TAN"), **SCAMP})


def test_tan_with_low_order_pv_on_latitude_axis_raises() -> None:
    # SCAMP degree-1 output: PV1_0..2 and PV2_0..2 only. Ambiguous on TAN.
    header = {**BASE, **_ctype("TAN"), "PV1_1": 1.0, "PV2_1": 1.0, "PV2_2": 1e-3}
    with pytest.raises(ValueError, match="TAN does not define"):
        WCS(header)
    # The same cards under an explicit TPV CTYPE are accepted.
    WCS({**header, **_ctype("TPV")})


@pytest.mark.parametrize(
    "pv",
    [{"PV1_3": 170.0}, {"PV1_1": 0.5, "PV1_2": 89.5}],
    ids=["lonpole", "fiducial"],
)
def test_tan_longitude_pv_keeps_standard_meaning(pv: dict[str, float]) -> None:
    header = {**BASE, **_ctype("TAN"), **pv}
    with warnings.catch_warnings():
        warnings.simplefilter("error", WCSHeaderWarning)
        wcs = WCS(header)
    assert wcs.tpv is None
    np.testing.assert_allclose(
        _torch_world(wcs), _astropy_world(header), rtol=0, atol=1e-9
    )


# DECam InstCal-style TPV header (CCD S29 layout, coefficients of DECam size).
DECAM_TPV = {
    "NAXIS": 2,
    "NAXIS1": 2046,
    "NAXIS2": 4094,
    "CTYPE1": "RA---TPV",
    "CTYPE2": "DEC--TPV",
    "CRVAL1": 52.1123,
    "CRVAL2": -28.4511,
    "CRPIX1": -9120.8,
    "CRPIX2": 4177.6,
    "CD1_1": 1.0e-8,
    "CD1_2": 7.2858e-05,
    "CD2_1": -7.2860e-05,
    "CD2_2": 1.5e-8,
    "PV1_0": -0.00334,
    "PV1_1": 1.01177,
    "PV1_2": -0.00756,
    "PV1_3": 0.0,
    "PV1_4": -0.01363,
    "PV1_5": 0.00849,
    "PV1_6": -0.00627,
    "PV1_7": -0.00954,
    "PV1_8": 0.00296,
    "PV1_9": -0.00664,
    "PV1_10": 0.00347,
    "PV2_0": 0.00552,
    "PV2_1": 1.01608,
    "PV2_2": -0.01362,
    "PV2_3": 0.0,
    "PV2_4": -0.01159,
    "PV2_5": 0.00824,
    "PV2_6": -0.00569,
    "PV2_7": -0.00727,
    "PV2_8": 0.00309,
    "PV2_9": -0.00598,
    "PV2_10": 0.00266,
}


def test_tpv_pv1_3_and_pv1_4_are_not_lonpole_latpole() -> None:
    # PV1_3 / PV1_4 are TPV coefficients; with PV1_3 = 0 the old reading set
    # LONPOLE = 0 and mirrored the field about CRVAL.
    linear = {**BASE, **_ctype("TAN")}
    tpv = {**BASE, **_ctype("TPV"), "PV1_1": 1.0, "PV2_1": 1.0, "PV1_3": 0.0}
    np.testing.assert_allclose(
        _torch_world(WCS(tpv)), _torch_world(WCS(linear)), rtol=0, atol=1e-12
    )
    with_pv4 = {**tpv, "PV1_4": 0.0, "PV2_4": 0.0}
    np.testing.assert_allclose(
        _torch_world(WCS(with_pv4)), _torch_world(WCS(linear)), rtol=0, atol=1e-12
    )


def test_decam_tpv_header_matches_astropy() -> None:
    wcs = WCS(DECAM_TPV)
    pix = np.array([[0.0, 0.0], [2045.0, 0.0], [1022.5, 2046.5], [2045.0, 4093.0]])
    t = torch.as_tensor(pix, dtype=torch.float64)
    ra, dec = wcs.pixel_to_world(t[:, 0], t[:, 1])
    world = np.stack([ra.numpy(), dec.numpy()], axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = AstropyWCS(DECAM_TPV).all_pix2world(pix, 0)
    np.testing.assert_allclose(world, ref, rtol=0, atol=1e-9)
    x, y = wcs.world_to_pixel(ra, dec)
    np.testing.assert_allclose(np.stack([x.numpy(), y.numpy()], 1), pix, atol=1e-6)
