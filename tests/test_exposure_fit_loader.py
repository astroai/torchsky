"""Instrument descriptions and load_exposures on small synthetic exposure files written with torchfits (no real data)."""

import dataclasses
import math

import pytest
import torch
import torchfits

from torchsky.instrument import (
    ACSWFCDataQuality,
    InstrumentDescription,
    JWSTDataQuality,
    WFC3IRDataQuality,
    instrument_description,
)
from torchsky.map.exposure_fit import load_exposures
from torchsky.map.exposure_fit._bandpass import box_filter
from torchsky.map.exposure_fit._observations import observation_to_record
from torchsky.map.quality import DQPolicy

RA, DEC, N, SCALE = (
    150.0,
    2.0,
    60,
    0.1,
)  # tangent point [deg], chip size [pixel], pixel scale [arcsec]


def _wcs_cards(crval=(RA, DEC), suffix=""):
    s = SCALE / 3600.0
    return {
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        f"CRVAL1{suffix}": crval[0],
        f"CRVAL2{suffix}": crval[1],
        f"CRPIX1{suffix}": N / 2 + 0.5,
        f"CRPIX2{suffix}": N / 2 + 0.5,
        f"CD1_1{suffix}": -s,
        f"CD1_2{suffix}": 0.0,
        f"CD2_1{suffix}": 0.0,
        f"CD2_2{suffix}": s,
    }


def _write(path, primary, sci_header, data, err, dq):
    hdus = [
        torchfits.TensorHDU(data=torch.zeros(1), header=primary),
        torchfits.TensorHDU(data=data, header={**sci_header, "EXTNAME": "SCI"}),
        torchfits.TensorHDU(data=err, header={"EXTNAME": "ERR"}),
        torchfits.TensorHDU(data=dq, header={"EXTNAME": "DQ"}),
    ]
    torchfits.HDUList(hdus).write(str(path))


def _data():
    g = torch.Generator().manual_seed(1)
    data = torch.rand(N, N, generator=g) + 1.0
    err = torch.full((N, N), 0.25)
    return data, err


@pytest.fixture
def jwst_file(tmp_path):
    data, err = _data()
    dq = torch.zeros(N, N, dtype=torch.int32)
    dq[25, 25] = 1  # DO_NOT_USE
    dq[35, 28] = 2  # SATURATED
    path = tmp_path / "jw_test_cal.fits"
    _write(
        path,
        {
            "INSTRUME": "NIRCAM",
            "DETECTOR": "NRCB1",
            "FILTER": "F150W",
            "PUPIL": "CLEAR",
            "EFFEXPTM": 100.0,
        },
        {**_wcs_cards(), "BUNIT": "MJy/sr"},
        data,
        err,
        dq,
    )
    return path, data, err


@pytest.fixture
def hst_file(tmp_path):
    data, err = _data()
    dq = torch.zeros(N, N, dtype=torch.int16)
    dq[25, 25] = 4  # bad detector pixel: rejects
    dq[35, 28] = 16  # hot pixel: ignored by default
    path = tmp_path / "ic_test_flt.fits"
    primary = {
        "INSTRUME": "WFC3",
        "DETECTOR": "IR",
        "FILTER": "F140W",
        "EXPTIME": 100.0,
    }
    sci = {
        **_wcs_cards((RA + 0.5, DEC)),
        **_wcs_cards((RA, DEC), "B"),
        "WCSNAME": "IDC_x-FIT_REL_GAIAeDR3",
        "WCSNAMEB": "IDC_x-GSC240-1",
        "BUNIT": "ELECTRONS/S",
        "PHOTFNU": 1.5e-7,
        "IDCSCALE": SCALE,
    }
    _write(path, primary, sci, data, err, dq)
    return path, data, err


def _small(desc):
    return dataclasses.replace(desc, min_valid_pixels=10)


def test_descriptions_facts_and_errors():
    n, w, a = (instrument_description(k) for k in ("NIRCAM", "WFC3_IR", "ACS_WFC"))
    assert isinstance(n, InstrumentDescription)
    assert (n.diameter_m, w.diameter_m, a.diameter_m) == (6.5, 2.4, 2.4)
    assert (
        n.channel_of("NRCBLONG") == "NIRCAM_LW"
        and n.channel_of("NRCA3") == "NIRCAM_SW"
        and w.channel_of("x") == "WFC3_IR"
    )
    assert (
        n.pixel_scale_arcsec["NIRCAM_LW"] == 0.063
        and a.chip_extvers == (1, 2)
        and n.detector_terms == ("stripes",)
        and not w.detector_terms
    )
    assert (
        n.bandpass_id("F150W") == "JWST/NIRCam.F150W"
        and w.bandpass_id("F140W") == "HST/WFC3_IR.F140W"
    )
    assert instrument_description("wfc3/ir") is w and instrument_description("ACS") is a
    with pytest.raises(KeyError):
        instrument_description("MIRI")
    assert (
        n.dq_flags is JWSTDataQuality
        and w.dq_flags is WFC3IRDataQuality
        and a.dq_flags is ACSWFCDataQuality
    )
    assert (
        w.dq_ignore_bits
        == WFC3IRDataQuality.HOT_PIXEL | WFC3IRDataQuality.BAD_FLAT
        == 528
        and a.dq_ignore_bits == 80
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        n.diameter_m = 1.0


def test_dq_policies_match_the_prototype_rule():
    q = torch.tensor([0, 1, 2, 16, 528, 4, 32768, 65536 + 16], dtype=torch.int64)
    jw = instrument_description("NIRCAM").dq_policy().usability_mask(q).tolist()
    assert jw == [True, False, False, False, False, False, False, False]  # strict
    hst = (
        instrument_description("WFC3_IR")
        .dq_policy()
        .usability_mask(q.to(torch.int32))
        .tolist()
    )
    assert hst == [
        True,
        False,
        False,
        True,
        True,
        False,
        False,
        True,
    ]  # low 16 bits only; 16 and 512 ignored
    assert (
        instrument_description("NIRCAM")
        .dq_policy(ignore_bits=2)
        .usability_mask(q)
        .tolist()[2]
        is True
    )


def test_sky_factor_formulas():
    w = instrument_description("WFC3_IR")
    f, how = w.sky_factor(
        {"EXPTIME": 50.0},
        {
            "BUNIT": "ELECTRONS",
            "PHOTFLAM": 1e-20,
            "PHOTPLAM": 14000.0,
            "IDCSCALE": 0.13,
        },
    )
    phot = 1e23 * 1e-20 * 14000.0**2 / 2.99792458e18
    assert f == pytest.approx(
        phot * 1e-6 / ((0.13**2 * (math.pi / 648000) ** 2) * 50.0), rel=1e-14
    ) and how["photfnu_source"].startswith("1e23")
    f2, _ = w.sky_factor(
        {}, {"BUNIT": "ELECTRONS/S", "PHOTFNU": 2e-7}, nominal_scale_arcsec=0.13
    )
    assert f2 == pytest.approx(
        2e-7 * 1e-6 / (0.13**2 * (math.pi / 648000) ** 2), rel=1e-14
    )
    assert (
        instrument_description("NIRCAM").sky_factor({}, {"BUNIT": "MJy/sr"})[0] == 1.0
    )
    for hdr in ({"BUNIT": "DN"}, {"BUNIT": "ELECTRONS"}, {"BUNIT": "ELECTRONS/S"}):
        with pytest.raises(ValueError):
            w.sky_factor({}, hdr)
    with pytest.raises(ValueError):
        instrument_description("NIRCAM").sky_factor({}, {"BUNIT": "ELECTRONS/S"})


def test_filter_curves_are_attached_not_fetched(jwst_file):
    path, _, _ = jwst_file
    lam, T = torch.linspace(1.2, 1.8, 20), torch.ones(20)
    desc = _small(
        instrument_description("NIRCAM").with_filter_curves({"f150w": (lam, T)})
    )
    (obs,) = load_exposures([path], desc, (RA, DEC, 2.0))
    assert (
        torch.equal(obs.transmission_wavelength, lam.double())
        and obs.spectral_unit == "um"
    )
    (bare,) = load_exposures(
        [path], _small(instrument_description("NIRCAM")), (RA, DEC, 2.0)
    )
    assert bare.transmission_wavelength is None
    with pytest.raises(ValueError):
        instrument_description("NIRCAM").with_filter_curves({"F150W": (lam, T[:5])})


def test_load_jwst_like_file(jwst_file):
    path, data, err = jwst_file
    desc = _small(instrument_description("NIRCAM"))
    (obs,) = load_exposures([path], desc, (RA, DEC, 2.0))
    assert (
        obs.band_name == "F150W"
        and obs.value_unit == "MJy/sr"
        and obs.value_semantics == "surface_brightness"
    )
    h = obs.header
    assert (h["TSINST"], h["DETECTOR"], h["TSSPLIT"], h["TSNAME"]) == (
        "NIRCAM",
        "NRCB1",
        "train",
        "jw_test",
    ) and h["TSTOSKY"] == 1.0
    assert (h["TSDETNX"], h["TSDETNY"]) == (N, N) and h["TSBOXHS"] == 2.0
    ny, nx = obs.image.shape
    assert 38 <= ny <= 42 and 38 <= nx <= 42
    y0, x0 = int(obs.detector_y0_pix), int(obs.detector_x0_pix)
    assert torch.equal(obs.image[obs.mask], data[y0 : y0 + ny, x0 : x0 + nx][obs.mask])
    full = torch.zeros(N, N, dtype=torch.bool)
    full[y0 : y0 + ny, x0 : x0 + nx] = obs.mask
    assert (
        not bool(full[25, 25]) and not bool(full[35, 28]) and bool(full[30, 30])
    )  # DO_NOT_USE and SATURATED both reject (strict policy)
    # DQ policy argument replaces the instrument's own
    (loose,) = load_exposures(
        [path], desc, (RA, DEC, 2.0), dq_policy=DQPolicy.accept_all()
    )
    assert int(loose.mask.sum()) > int(obs.mask.sum())
    # no overlap -> nothing; a missing file raises; an unknown camera raises
    assert load_exposures([path], desc, (RA + 1.0, DEC, 2.0)) == []
    with pytest.raises(FileNotFoundError):
        load_exposures([path.with_name("nope.fits")], desc, (RA, DEC, 2.0))
    with pytest.raises(ValueError, match="no instrument description"):
        load_exposures([path], instrument_description("WFC3_IR"), (RA, DEC, 2.0))


def test_load_hst_like_file_units_dq_and_apriori_solution(hst_file):
    path, data, err = hst_file
    desc = _small(instrument_description("WFC3_IR"))
    (obs,) = load_exposures([path], desc, (RA, DEC, 2.0))
    factor = 1.5e-7 * 1e-6 / (SCALE**2 * (math.pi / 648000) ** 2)
    assert (
        obs.header["TSTOSKY"] == pytest.approx(factor, rel=1e-14)
        and obs.header["TSNAME"] == "ic_test"
    )
    y0, x0 = int(obs.detector_y0_pix), int(obs.detector_x0_pix)
    ny, nx = obs.image.shape
    f32 = torch.tensor(factor, dtype=torch.float32)
    assert torch.equal(
        obs.image[obs.mask], (data[y0 : y0 + ny, x0 : x0 + nx] * f32)[obs.mask]
    )
    assert torch.equal(obs.err, torch.full_like(obs.err, 0.25) * f32)
    m = torch.zeros(N, N, dtype=torch.bool)
    m[y0 : y0 + ny, x0 : x0 + nx] = obs.mask
    assert not bool(m[25, 25]) and bool(
        m[35, 28]
    )  # bad pixel rejects; hot pixel ignored by the HST default
    # the a priori (alternate B) solution is used, not the primary one (0.5 deg away: the box would be empty with it)
    assert obs.header["CRVAL1"] == RA and obs.header["WCSNAME"] == "IDC_x-GSC240-1"
    # positions: the tangent plane at the box centre; the middle pixel is at the centre
    rec = observation_to_record(
        obs,
        filt=0,
        inst=0,
        chan=0,
        det=0,
        bandpass=box_filter("F140W", "WFC3_IR", 1.3, 1.5, 3),
    )
    iy, ix = N // 2 - y0, N // 2 - x0
    assert (
        abs(float(rec.X[iy, ix])) < 0.06 + 1e-3
        and abs(float(rec.Y[iy, ix])) < 0.06 + 1e-3
    )
    assert float(rec.X[iy, ix + 1] - rec.X[iy, ix]) == pytest.approx(
        SCALE, abs=1e-5
    )  # CD1_1 < 0: RA falls with the column, so X (west) rises
    assert rec.det_shape == (N, N) and rec.cut0 == (y0, x0) and not rec.stripes


def test_hst_without_apriori_solution_fails_closed(tmp_path):
    data, err = _data()
    path = tmp_path / "ic_bad_flt.fits"
    sci = {
        **_wcs_cards(),
        "WCSNAME": "IDC_x-FIT_REL_GAIAeDR3",
        "BUNIT": "ELECTRONS/S",
        "PHOTFNU": 1.5e-7,
        "IDCSCALE": SCALE,
    }
    _write(
        path,
        {"INSTRUME": "WFC3", "DETECTOR": "IR", "FILTER": "F140W"},
        sci,
        data,
        err,
        torch.zeros(N, N, dtype=torch.int16),
    )
    with pytest.raises(ValueError, match="a priori"):
        load_exposures(
            [path], _small(instrument_description("WFC3_IR")), (RA, DEC, 2.0)
        )
