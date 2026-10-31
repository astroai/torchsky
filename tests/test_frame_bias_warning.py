import warnings

import pytest
import torch

from torchsky.map.image_plane_common import _convert_world_between_wcs
from torchsky.wcs import WCS, CelestialFrameBiasWarning


def _tan(**extra):
    hdr = {"NAXIS": 2, "NAXIS1": 10, "NAXIS2": 10, "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
           "CRVAL1": 0.0, "CRVAL2": 24.5, "CRPIX1": 5.0, "CRPIX2": 5.0,
           "CD1_1": -5e-5, "CD1_2": 0.0, "CD2_1": 0.0, "CD2_2": 5e-5}
    hdr.update(extra)
    return WCS(hdr)


def test_fk5_j2000_to_icrs_warns_and_applies_frame_bias():
    src, dst = _tan(RADECSYS="FK5", EQUINOX=2000.0), _tan(RADESYS="ICRS")
    lon = torch.tensor([0.0], dtype=torch.float64)
    lat = torch.tensor([24.5], dtype=torch.float64)
    with pytest.warns(CelestialFrameBiasWarning):
        lon2, lat2 = _convert_world_between_wcs(lon, lat, src, dst)
    dlon = torch.remainder(lon2 - lon + 180.0, 360.0) - 180.0
    shift_mas = float(torch.hypot(dlon * torch.cos(torch.deg2rad(lat)), lat2 - lat)) * 3.6e6
    assert 10.0 < shift_mas < 40.0


def test_same_frame_and_explicit_icrs_do_not_warn():
    lon = torch.tensor([0.0], dtype=torch.float64)
    lat = torch.tensor([24.5], dtype=torch.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("error", CelestialFrameBiasWarning)
        _convert_world_between_wcs(lon, lat, _tan(RADESYS="ICRS"), _tan(RADESYS="ICRS"))
        _convert_world_between_wcs(lon, lat, _tan(RADESYS="FK5", EQUINOX=2000.0), _tan(RADESYS="FK5", EQUINOX=2000.0))


def test_fk5_other_equinox_does_not_warn():
    lon = torch.tensor([10.0], dtype=torch.float64)
    lat = torch.tensor([24.5], dtype=torch.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("error", CelestialFrameBiasWarning)
        _convert_world_between_wcs(lon, lat, _tan(RADESYS="FK5", EQUINOX=1975.0), _tan(RADESYS="ICRS"))
