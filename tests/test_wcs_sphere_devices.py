"""CPU / CUDA / MPS residency for public WCS and sphere tensor paths."""

from __future__ import annotations

import pytest
import torch

from torchsky.sphere import MOC, STMOC, SparseHealpixMap
from torchsky.sphere.healpix import get_interp_weights, query_disc, query_polygon
from torchsky.sphere.moc import nest2uniq
from torchsky.wcs.core import WCS
from torchsky.wcs.spectral import air_wavelength_to_vacuum
from torchsky.wcs.tabular import TabularAxis
from torchsky.wcs.temporal import convert_time_scale


def _require_device(device_type: str) -> torch.device:
    if device_type == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return torch.device("cuda")
    if device_type == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            pytest.skip("MPS not available")
        return torch.device("mps")
    return torch.device("cpu")


def _dtype(device: torch.device) -> torch.dtype:
    return torch.float32 if device.type == "mps" else torch.float64


def _tan_header() -> dict[str, float | int | str]:
    return {
        "NAXIS": 2,
        "NAXIS1": 256,
        "NAXIS2": 256,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 128.5,
        "CRPIX2": 128.5,
        "CRVAL1": 180.0,
        "CRVAL2": 0.0,
        "CD1_1": -0.01,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": 0.01,
    }


def _identity_tpv_header() -> dict[str, float | int | str]:
    header = dict(_tan_header())
    header["CTYPE1"] = "RA---TPV"
    header["CTYPE2"] = "DEC--TPV"
    header["PV1_1"] = 1.0
    header["PV2_1"] = 1.0
    return header


def _sip_header() -> dict[str, float | int | str]:
    header = dict(_tan_header())
    header["CTYPE1"] = "RA---TAN-SIP"
    header["CTYPE2"] = "DEC--TAN-SIP"
    header["A_ORDER"] = 2
    header["B_ORDER"] = 2
    header["A_2_0"] = 3.0e-6
    header["B_0_2"] = 2.5e-6
    header["AP_ORDER"] = 1
    header["BP_ORDER"] = 1
    return header


def _csc_header() -> dict[str, float | int | str]:
    header = dict(_tan_header())
    header["CTYPE1"] = "RA---CSC"
    header["CTYPE2"] = "DEC--CSC"
    header["CD1_1"] = -0.3
    header["CD2_2"] = 0.3
    return header


def _wave_header() -> dict[str, float | int | str]:
    return {
        "NAXIS": 1,
        "CTYPE1": "WAVE",
        "CRPIX1": 1.0,
        "CRVAL1": 4000.0,
        "CDELT1": 10.0,
    }


def _gra_header() -> dict[str, float | int | str]:
    return {
        "NAXIS": 1,
        "NAXIS1": 128,
        "CTYPE1": "FREQ-GRA",
        "CUNIT1": "Hz",
        "CRPIX1": 1.0,
        "CRVAL1": 6.0e14,
        "CDELT1": 1.0e9,
        "PV1_0": 300000.0,
        "PV1_1": 1.0,
        "PV1_2": 15.0,
        "PV1_3": 1.5,
        "PV1_4": -1.0e5,
        "PV1_5": 0.0,
        "PV1_6": 10.0,
    }


@pytest.mark.parametrize("device_type", ["cpu", "cuda", "mps"])
@pytest.mark.parametrize(
    "header",
    [_tan_header(), _identity_tpv_header(), _sip_header(), _csc_header()],
)
def test_celestial_wcs_stays_on_device(
    device_type: str, header: dict[str, float | int | str]
) -> None:
    device = _require_device(device_type)
    dtype = _dtype(device)
    wcs = WCS(header).to(device)
    x = torch.tensor([120.0, 140.0], dtype=dtype, device=device)
    y = torch.tensor([120.0, 140.0], dtype=dtype, device=device)
    ra, dec = wcs.pixel_to_world(x, y, origin=0)
    assert ra.device.type == device_type
    assert dec.device.type == device_type
    assert torch.isfinite(ra).all()
    px, py = wcs.world_to_pixel(ra, dec, origin=0)
    assert px.device.type == device_type
    assert py.device.type == device_type
    fp = wcs.footprint(origin=0)
    assert fp.device.type == device_type
    inside = wcs.in_image(torch.stack([x, y], dim=-1), origin=0)
    assert inside.device.type == device_type
    ctype = str(header["CTYPE1"])
    if "CSC" in ctype:
        atol = 1.0e-2
    elif device_type == "mps":
        atol = 0.2
    else:
        atol = 1.0e-8
    torch.testing.assert_close(px, x, atol=atol, rtol=0.0)
    torch.testing.assert_close(py, y, atol=atol, rtol=0.0)


@pytest.mark.parametrize("device_type", ["cpu", "cuda", "mps"])
@pytest.mark.parametrize("header", [_wave_header(), _gra_header()])
def test_spectral_wcs_stays_on_device(
    device_type: str, header: dict[str, float | int | str]
) -> None:
    device = _require_device(device_type)
    dtype = _dtype(device)
    wcs = WCS(header).to(device)
    pix = torch.tensor([[0.0], [1.0], [10.0]], dtype=dtype, device=device)
    world = wcs.pixel_to_world(pix, origin=0)
    assert world.device.type == device_type
    back = wcs.world_to_pixel(world, origin=0)
    assert back.device.type == device_type
    atol = 0.2 if device_type == "mps" else 1e-8
    torch.testing.assert_close(back, pix, atol=atol, rtol=0.0)


@pytest.mark.parametrize("device_type", ["cpu", "cuda", "mps"])
def test_tabular_axis_stays_on_device(device_type: str) -> None:
    device = _require_device(device_type)
    dtype = _dtype(device)
    axis = TabularAxis(
        pixel=torch.tensor([1.0, 2.0, 4.0], dtype=dtype),
        world=torch.tensor([10.0, 20.0, 80.0], dtype=dtype),
        unit="nm",
    ).to(device=device, dtype=dtype)
    pixel = torch.tensor([1.5, 3.0], dtype=dtype, device=device)
    world = axis.pixel_to_world(pixel)
    assert world.device.type == device_type
    back = axis.world_to_pixel(world)
    torch.testing.assert_close(
        back, pixel, atol=1e-5 if device_type == "mps" else 1e-12, rtol=0.0
    )


@pytest.mark.parametrize("device_type", ["cpu", "cuda", "mps"])
def test_air_wavelength_and_time_scale_stay_on_device(device_type: str) -> None:
    device = _require_device(device_type)
    dtype = _dtype(device)
    wave = torch.tensor([5.0e-7, 6.0e-7], dtype=dtype, device=device)
    vac = air_wavelength_to_vacuum(wave)
    assert vac.device.type == device_type
    mjd = torch.tensor([59000.0, 59000.5], dtype=dtype, device=device)
    tt = convert_time_scale(mjd, "UTC", "TT")
    assert tt.device.type == device_type
    assert torch.isfinite(tt).all()


@pytest.mark.parametrize("device_type", ["cpu", "cuda", "mps"])
def test_moc_healsparse_stmoc_stay_on_device(device_type: str) -> None:
    device = _require_device(device_type)
    dtype = _dtype(device)
    nside = 8
    moc = MOC(nest2uniq(nside, torch.tensor([0, 1, 2], dtype=torch.int64)))
    lon = torch.tensor([45.0, 180.0], dtype=dtype, device=device)
    lat = torch.tensor([0.0, 0.0], dtype=dtype, device=device)
    mask = moc.contains(lon, lat)
    assert mask.device.type == device_type
    assert mask.dtype == torch.bool

    pix = torch.tensor([0, 1, 2], dtype=torch.int64, device=device)
    vals = torch.tensor([1.0, 2.0, 3.0], dtype=dtype, device=device)
    smap = SparseHealpixMap.from_pixels(
        pix, vals, nside_sparse=nside, nside_coverage=1, sentinel=0.0
    )
    got = smap.get_values(pix)
    assert got.device.type == device_type
    torch.testing.assert_close(got, vals, atol=0.0, rtol=0.0)

    stmoc = STMOC.from_spatial_coverages(59000.0, 59001.0, moc, time_depth=40)
    t = torch.tensor([59000.5, 58900.0], dtype=dtype, device=device)
    st_mask = stmoc.contains(t, lon, lat)
    assert st_mask.device.type == device_type


@pytest.mark.parametrize("device_type", ["cpu", "cuda", "mps"])
def test_healpix_query_and_interp_stay_on_device(device_type: str) -> None:
    device = _require_device(device_type)
    dtype = _dtype(device)
    vec = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device)
    empty = query_disc(8, vec, 0.0, nest=False)
    assert empty.device.type == device_type
    assert empty.numel() == 0
    pix = query_disc(8, vec, 0.2, nest=False)
    assert pix.device.type == device_type
    assert pix.numel() > 0

    ra = torch.tensor([10.0, 45.0, 180.0], dtype=dtype, device=device)
    dec = torch.tensor([-20.0, 0.0, 35.0], dtype=dtype, device=device)
    ip, w = get_interp_weights(32, ra, dec, nest=False, lonlat=True)
    assert ip.device.type == device_type
    assert w.device.type == device_type
    torch.testing.assert_close(
        w.sum(dim=0), torch.ones(3, dtype=dtype, device=device), atol=1e-5, rtol=0.0
    )

    verts = torch.tensor(
        [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
        dtype=dtype,
        device=device,
    )
    poly = query_polygon(8, verts, lonlat=True, nest=True)
    assert poly.device.type == device_type
    assert poly.numel() > 0
