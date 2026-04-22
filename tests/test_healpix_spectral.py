from __future__ import annotations

import torch

from torchsky.map import (
    rebandlimit_healpix_map_via_sht,
    sample_alm_on_image_header_spectral,
    sample_healpix_spin_map_on_image_header_harmonic,
    sample_healpix_map_on_image_header,
    sample_healpix_map_on_image_header_harmonic,
    sample_healpix_map_on_image_header_spectral,
    sample_spin_alm_on_image_header_spectral,
)
from torchsky.sphere import alm2lonlat, alm2lonlat_spin, alm2map, alm2map_spin, alm_size
from torchsky.wcs import healpix as hp


def _tan_header(size: int) -> dict:
    return {
        "NAXIS": 2,
        "NAXIS1": size,
        "NAXIS2": size,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRVAL1": 45.0,
        "CRVAL2": -20.0,
        "CRPIX1": size / 2.0,
        "CRPIX2": size / 2.0,
        "CDELT1": -0.8,
        "CDELT2": 0.8,
    }


def test_rebandlimit_healpix_map_via_sht_identity_same_nside() -> None:
    nside = 8
    lmax = 6
    alm = torch.zeros(alm_size(lmax), dtype=torch.complex128)
    alm[0] = 1.0 + 0.0j
    alm[1] = 0.1 + 0.0j
    alm[5] = -0.08 + 0.04j
    m = alm2map(alm, nside=nside, lmax=lmax).to(torch.float64)
    out = rebandlimit_healpix_map_via_sht(
        m,
        nside=nside,
        lmax=lmax,
        target_nside=nside,
    )
    torch.testing.assert_close(out, m, atol=8e-3, rtol=3e-2)


def test_rebandlimit_healpix_map_via_sht_smoothing_reduces_peak() -> None:
    nside = 8
    npix = 12 * nside * nside
    m = torch.zeros(npix, dtype=torch.float64)
    m[npix // 3] = 1.0
    out = rebandlimit_healpix_map_via_sht(
        m,
        nside=nside,
        fwhm_rad=0.15,
    )
    assert float(out.max()) < float(m.max())
    assert torch.isfinite(out).all()


def test_sample_healpix_map_on_image_header_spectral_matches_native_bridge_without_filter() -> (
    None
):
    nside = 8
    npix = 12 * nside * nside
    size = 24
    hdr = _tan_header(size)
    m = torch.linspace(0.0, 1.0, npix, dtype=torch.float64)
    native = sample_healpix_map_on_image_header(
        m,
        nside=nside,
        nest=False,
        target_header=hdr,
    )
    spectral = sample_healpix_map_on_image_header_spectral(
        m,
        nside=nside,
        nest=False,
        target_header=hdr,
    )
    torch.testing.assert_close(spectral, native, atol=3e-2, rtol=3e-2)


def test_sample_healpix_map_on_image_header_spectral_smoothed_is_finite() -> None:
    nside = 8
    npix = 12 * nside * nside
    size = 24
    hdr = _tan_header(size)
    m = torch.zeros(npix, dtype=torch.float64)
    m[npix // 2] = 1.0
    spectral = sample_healpix_map_on_image_header_spectral(
        m,
        nside=nside,
        nest=False,
        target_header=hdr,
        fwhm_rad=0.2,
    )
    assert spectral.shape == (size, size)
    assert torch.isfinite(spectral).all()
    assert float(spectral.max()) < 1.0


def test_alm2lonlat_matches_alm2map_at_healpix_pixel_centers() -> None:
    nside = 8
    lmax = 6
    alm = torch.zeros(alm_size(lmax), dtype=torch.complex128)
    alm[0] = 1.0 + 0.0j
    alm[1] = 0.1 + 0.0j
    alm[5] = -0.08 + 0.04j
    m = alm2map(alm, nside=nside, lmax=lmax).to(torch.float64)
    npix = hp.nside2npix(nside)
    ipix = torch.arange(npix, dtype=torch.int64)
    lon, lat = hp.pix2ang(nside, ipix, nest=False, lonlat=True)
    direct = alm2lonlat(alm, lon, lat, lmax=lmax)
    torch.testing.assert_close(direct, m, atol=8e-3, rtol=3e-2)


def test_sample_alm_on_image_header_spectral_matches_harmonic_map_bridge() -> None:
    nside = 8
    lmax = 6
    size = 24
    hdr = _tan_header(size)
    alm = torch.zeros(alm_size(lmax), dtype=torch.complex128)
    alm[0] = 1.0 + 0.0j
    alm[1] = 0.1 + 0.0j
    alm[5] = -0.08 + 0.04j
    hp_map = alm2map(alm, nside=nside, lmax=lmax).to(torch.float64)
    direct = sample_alm_on_image_header_spectral(
        alm,
        target_header=hdr,
        lmax=lmax,
    )
    harmonic = sample_healpix_map_on_image_header_harmonic(
        hp_map,
        nside=nside,
        nest=False,
        target_header=hdr,
        lmax=lmax,
    )
    torch.testing.assert_close(direct, harmonic, atol=2e-2, rtol=5e-2)


def test_alm2lonlat_spin_matches_alm2map_spin_at_healpix_pixel_centers() -> None:
    nside = 8
    lmax = 6
    a_e = torch.zeros(alm_size(lmax), dtype=torch.complex128)
    a_b = torch.zeros(alm_size(lmax), dtype=torch.complex128)
    a_e[2] = 0.07 + 0.03j
    a_b[5] = -0.04 + 0.02j
    qu = alm2map_spin((a_e, a_b), nside=nside, spin=2, lmax=lmax).to(torch.float64)
    npix = hp.nside2npix(nside)
    ipix = torch.arange(npix, dtype=torch.int64)
    lon, lat = hp.pix2ang(nside, ipix, nest=False, lonlat=True)
    direct = alm2lonlat_spin((a_e, a_b), 2, lon, lat, lmax=lmax)
    torch.testing.assert_close(direct, qu, atol=8e-3, rtol=3e-2)


def test_sample_spin_alm_on_image_header_spectral_matches_spin_harmonic_bridge() -> (
    None
):
    nside = 8
    lmax = 6
    size = 24
    hdr = _tan_header(size)
    a_e = torch.zeros(alm_size(lmax), dtype=torch.complex128)
    a_b = torch.zeros(alm_size(lmax), dtype=torch.complex128)
    a_e[2] = 0.07 + 0.03j
    a_b[5] = -0.04 + 0.02j
    qu = alm2map_spin((a_e, a_b), nside=nside, spin=2, lmax=lmax).to(torch.float64)
    direct = sample_spin_alm_on_image_header_spectral(
        (a_e, a_b),
        spin=2,
        target_header=hdr,
        lmax=lmax,
    )
    harmonic = sample_healpix_spin_map_on_image_header_harmonic(
        qu,
        nside=nside,
        spin=2,
        nest=False,
        target_header=hdr,
        lmax=lmax,
    )
    torch.testing.assert_close(direct, harmonic, atol=2e-2, rtol=5e-2)
