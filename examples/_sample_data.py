"""Download, verify and cache small public samples for the real-data examples.

Every sample is a named entry of :data:`SAMPLES` (URL, expected size and
sha256 where stable, container magic bytes, credit). Files land in
``~/.cache/torchsky/samples`` (override with ``TORCHSKY_SAMPLE_CACHE``) and are
re-validated on every cache hit; a damaged entry is evicted and fetched again.

Network policy: ``TORCHSKY_EXAMPLE_FAST=1`` (set by the runner under CI)
forbids downloads, so :func:`fetch_sample` raises :class:`SampleUnavailable`
for a sample that is not already cached. Examples translate that into exit
status 77 with :func:`skip_if_unavailable`, which the runner reports as SKIP.
Large samples (``large=True``) are only fetched on explicit request.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
from pathlib import Path
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from typing import Any

SKIP = 77
CONNECT_TIMEOUT_S = 30.0
DOWNLOAD_ATTEMPTS = 3
_FITS = (b"SIMPLE  =",)
_GZIP = (b"\x1f\x8b",)
_ASCII = tuple(str(d).encode() for d in range(10))
_SVO = "http://svo2.cab.inta-csic.es/theory/fps/getdata.php?format=ascii&id="
_CADC = "https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/CFHTSG"
_LEGACY = "https://www.legacysurvey.org/viewer/fits-cutout"
_LEGACY_CREDIT = (
    "Legacy Surveys DR10 (Dey et al. 2019), public data; acknowledge the Legacy "
    "Surveys, DECam/DES and NOIRLab per legacysurvey.org/acknowledgment"
)
_PLA = "https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/component-maps"
# Pinned query snapshots: a synchronous TAP / SkyServer GET whose ORDER BY makes
# the CSV byte-stable for an immutable release, so its sha256 can be pinned.
_GAIA_TAP = "https://irsa.ipac.caltech.edu/TAP/sync?FORMAT=csv&QUERY="
_SKYSERVER = (
    "https://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch?format=csv&cmd="
)
_IRSA_HIPS = "https://irsa.ipac.caltech.edu/data/hips/CDS"
_GAIA_CREDIT = "Gaia DR3 (ESA/Gaia/DPAC, CC BY-SA 3.0 IGO) via the IRSA TAP mirror; cite Gaia Collaboration 2023"
_HIPS_MOC_CREDIT = "HiPS coverage MOC built by CDS (Strasbourg), mirrored by IRSA"


def _query_url(base: str, query: str) -> str:
    return base + urllib.parse.quote(" ".join(query.split()), safe="")


class SampleUnavailable(RuntimeError):
    """A sample cannot be obtained (offline, FAST mode, HTTP error, bad bytes)."""


# bytes: exact expected size (None when the server regenerates the file).
# sha256: pinned digest when the file is immutable; None otherwise (the
# MegaPipe cutout embeds a generation timestamp in its FITS header, so only
# its size is stable).
SAMPLES: dict[str, dict[str, Any]] = {
    "horsehead": {
        "url": "http://data.astropy.org/tutorials/FITS-images/HorseHead.fits",
        "bytes": 1650240,
        "sha256": "c4bcc0bccbadf1488e4e361c0d8f04255dc344ff401fd4827139b3354eafe863",
        "magic": _FITS,
        "suffix": ".fits",
        "large": False,
        "description": "Horsehead Nebula DSS image with a TAN WCS (astropy tutorial).",
        "license": "astropy-data tutorial file; DSS credit STScI/DSS",
    },
    "chandra_events": {
        "url": "http://data.astropy.org/tutorials/FITS-tables/chandra_events.fits",
        "bytes": 31078080,
        "sha256": "ed14061f2146b82b78410afef8a620915acaed9e9804e461a0865e236e9ae5df",
        "magic": _FITS,
        "suffix": ".fits",
        "large": False,
        "description": "Chandra ACIS event list (binary table).",
        "license": "astropy-data tutorial file; Chandra X-ray Center public data",
    },
    "sdss_spectrum": {
        "url": (
            "https://data.sdss.org/sas/dr16/sdss/spectro/redux/26/spectra/"
            "0751/spec-0751-52251-0160.fits"
        ),
        "bytes": 604800,
        "sha256": "9bb725b4cf74cc5b547bdffa11d8cf3bcb4949cdb21b9173163fcb500eb85fb6",
        "magic": _FITS,
        "suffix": ".fits",
        "large": False,
        "description": "SDSS DR16 optical spectrum, plate 0751 MJD 52251 fiber 0160.",
        "license": "SDSS public data (CC BY-style; cite SDSS DR16)",
    },
    "cadc_megapipe_cutout": {
        "url": (
            f"{_CADC}/MegaPipe.712.165.G.MP9401.fits"
            "?SUB=%5B0%5D%5B8000:8511,4000:4511%5D"
        ),
        "bytes": 1056960,
        "sha256": None,
        "magic": _FITS,
        "suffix": ".fits",
        "large": False,
        "description": (
            "512x512 g-band cutout (pixels 8000:8511, 4000:4511) of the CFHT "
            "MegaPipe stack MegaPipe.712.165 from the CADC cutout service; "
            "TAN WCS, float32."
        ),
        "license": "CFHT/CADC MegaPipe (Gwyn 2008), open data; cite CFHT and CADC",
    },
    "megapipe_mosaic_a": {
        "url": (
            f"{_CADC}/MegaPipe.712.165.G.MP9401.fits"
            "?SUB=%5B0%5D%5B9489:10000,9489:10000%5D"
        ),
        "bytes": 1056960,
        "sha256": None,
        "magic": _FITS,
        "suffix": ".fits",
        "large": False,
        "description": (
            "512x512 g-band cutout of the CFHT MegaPipe stack, pixels 9489:10000 x 9489:10000 (the north-west corner on the sky) of tile 712.165. "
            "The four cutouts share one sky window near RA 358.82, Dec -7.25 "
            "where tiles 712.165, 711.165 and 712.166 overlap (each tile has "
            "its own TAN reference point)."
        ),
        "license": "CFHT/CADC MegaPipe (Gwyn 2008), open data; cite CFHT and CADC",
    },
    "megapipe_mosaic_b": {
        "url": (
            f"{_CADC}/MegaPipe.711.165.G.MP9401.fits?SUB=%5B0%5D%5B1:512,9489:10000%5D"
        ),
        "bytes": 1056960,
        "sha256": None,
        "magic": _FITS,
        "suffix": ".fits",
        "large": False,
        "description": (
            "512x512 g-band cutout of the CFHT MegaPipe stack, pixels 1:512 x 9489:10000 of tile 711.165, the western neighbour of tile 712.165 in the same row. "
            "The four cutouts share one sky window near RA 358.82, Dec -7.25 "
            "where tiles 712.165, 711.165 and 712.166 overlap (each tile has "
            "its own TAN reference point)."
        ),
        "license": "CFHT/CADC MegaPipe (Gwyn 2008), open data; cite CFHT and CADC",
    },
    "megapipe_mosaic_c": {
        "url": (
            f"{_CADC}/MegaPipe.712.166.G.MP9401.fits?SUB=%5B0%5D%5B1930:2442,1:512%5D"
        ),
        "bytes": 1056960,
        "sha256": None,
        "magic": _FITS,
        "suffix": ".fits",
        "large": False,
        "description": (
            "512x512 g-band cutout of the CFHT MegaPipe stack, pixels 1930:2442 x 1:512 of tile 712.166, the next row north (the rows are staggered in RA). "
            "The four cutouts share one sky window near RA 358.82, Dec -7.25 "
            "where tiles 712.165, 711.165 and 712.166 overlap (each tile has "
            "its own TAN reference point)."
        ),
        "license": "CFHT/CADC MegaPipe (Gwyn 2008), open data; cite CFHT and CADC",
    },
    "megapipe_mosaic_d": {
        "url": (
            f"{_CADC}/MegaPipe.712.166.G.MP9401.fits?SUB=%5B0%5D%5B2186:2697,1:512%5D"
        ),
        "bytes": 1056960,
        "sha256": None,
        "magic": _FITS,
        "suffix": ".fits",
        "large": False,
        "description": (
            "512x512 g-band cutout of the CFHT MegaPipe stack, pixels 2186:2697 x 1:512 of tile 712.166, shifted 256 px from the previous cutout of the same stack. "
            "The four cutouts share one sky window near RA 358.82, Dec -7.25 "
            "where tiles 712.165, 711.165 and 712.166 overlap (each tile has "
            "its own TAN reference point)."
        ),
        "license": "CFHT/CADC MegaPipe (Gwyn 2008), open data; cite CFHT and CADC",
    },
    "svo_sdss_u": {
        "url": f"{_SVO}SLOAN/SDSS.u",
        "bytes": 940,
        "sha256": "008fafdbc2a65b897fd56627142d266269275caa0d28622ac79c57a39ce24cfa",
        "magic": _ASCII,
        "suffix": ".dat",
        "large": False,
        "description": "SDSS u-band filter transmission (wavelength [A], photon response) from the SVO Filter Profile Service.",
        "license": "SVO Filter Profile Service (Rodrigo, Solano & Bayo 2012); cite SVO and Doi et al. 2010",
    },
    "svo_sdss_g": {
        "url": f"{_SVO}SLOAN/SDSS.g",
        "bytes": 1780,
        "sha256": "1efa583c9ba880e3e793b4b42bd2ba7c69d70f7c9aae49a9e6a13a2997413f42",
        "magic": _ASCII,
        "suffix": ".dat",
        "large": False,
        "description": "SDSS g-band filter transmission (wavelength [A], photon response) from the SVO Filter Profile Service.",
        "license": "SVO Filter Profile Service (Rodrigo, Solano & Bayo 2012); cite SVO and Doi et al. 2010",
    },
    "svo_sdss_r": {
        "url": f"{_SVO}SLOAN/SDSS.r",
        "bytes": 1500,
        "sha256": "269e13371b9b65376345d0ebe4ba6b5455ef1aba2a71b310966f1a1c2b662316",
        "magic": _ASCII,
        "suffix": ".dat",
        "large": False,
        "description": "SDSS r-band filter transmission (wavelength [A], photon response) from the SVO Filter Profile Service.",
        "license": "SVO Filter Profile Service (Rodrigo, Solano & Bayo 2012); cite SVO and Doi et al. 2010",
    },
    "svo_sdss_i": {
        "url": f"{_SVO}SLOAN/SDSS.i",
        "bytes": 1780,
        "sha256": "8a2a6ce51d9c56f657ff7e4d846cbcae51fdda7b6619fb487b0e52569983a622",
        "magic": _ASCII,
        "suffix": ".dat",
        "large": False,
        "description": "SDSS i-band filter transmission (wavelength [A], photon response) from the SVO Filter Profile Service.",
        "license": "SVO Filter Profile Service (Rodrigo, Solano & Bayo 2012); cite SVO and Doi et al. 2010",
    },
    "svo_sdss_z": {
        "url": f"{_SVO}SLOAN/SDSS.z",
        "bytes": 2870,
        "sha256": "8caaaf3777e9d4a7a6337cd48800a0bbfcaae8bf63b89dd9124689a84b3078ee",
        "magic": _ASCII,
        "suffix": ".dat",
        "large": False,
        "description": "SDSS z-band filter transmission (wavelength [A], photon response) from the SVO Filter Profile Service.",
        "license": "SVO Filter Profile Service (Rodrigo, Solano & Bayo 2012); cite SVO and Doi et al. 2010",
    },
    "healpy_full_weights_256": {
        "url": (
            "https://raw.githubusercontent.com/healpy/healpy-data/master/"
            "full_weights/healpix_full_weights_nside_0256.fits"
        ),
        "bytes": 403200,
        "sha256": "291d28271c572e0d6edb2c98d02411181457d1d4f3fd606e9cd999b4fd188638",
        "magic": _FITS,
        "suffix": ".fits",
        "large": False,
        "description": "HEALPix ring-weight file for nside=256 (healpy-data).",
        "license": "healpy-data repository (healpy licence)",
    },
    "planck_commander_sync_256": {
        "url": f"{_PLA}/foregrounds/COM_CompMap_Synchrotron-commander_0256_R2.00.fits",
        "bytes": 9455040,
        "sha256": "39a82ff5ced27b552fbf561641e183743d186316140bbbe64ba2702b58cb85bb",
        "magic": _FITS,
        "suffix": ".fits",
        "large": False,
        "description": "Planck PR2 Commander synchrotron component, HEALPix nside=256.",
        "license": "Planck Legacy Archive (ESA); cite Planck Collaboration 2015 X",
    },
    "planck_commander_dust_256": {
        "url": f"{_PLA}/foregrounds/COM_CompMap_dust-commander_0256_R2.00.fits",
        "bytes": 28324800,
        "sha256": "710ddec54b3e867e9138d6687170dfc431a9080ba693b7dbb41a3bc7f66d08c1",
        "magic": _FITS,
        "suffix": ".fits",
        "large": False,
        "description": "Planck PR2 Commander thermal dust component, HEALPix nside=256.",
        "license": "Planck Legacy Archive (ESA); cite Planck Collaboration 2015 X",
    },
    "planck_smica_1024": {
        "url": f"{_PLA}/cmb/COM_CMB_IQU-smica_1024_R2.02_full.fits",
        "bytes": 176209920,
        "sha256": None,
        "magic": _FITS,
        "suffix": ".fits",
        "large": True,
        "description": (
            "Planck PR2 SMICA CMB I/Q/U, nside=1024 (smallest official CMB map; "
            "the PR3 SMICA map is 2 GB at nside=2048)."
        ),
        "license": "Planck Legacy Archive (ESA); cite Planck Collaboration 2015 IX",
    },
    "manga_logcube": {
        "url": (
            "https://data.sdss.org/sas/dr17/manga/spectro/redux/v3_1_1/7443/"
            "stack/manga-7443-12703-LOGCUBE.fits.gz"
        ),
        "bytes": 212028256,
        "sha256": None,
        "magic": _GZIP,
        "suffix": ".fits.gz",
        "large": True,
        "description": "SDSS-IV MaNGA DR17 LOGCUBE, plate-IFU 7443-12703.",
        "license": "SDSS public data; cite SDSS DR17 and MaNGA (Bundy et al. 2015)",
    },
    "manga_8485_1901": {
        "url": (
            "https://data.sdss.org/sas/dr17/manga/spectro/redux/v3_1_1/8485/"
            "stack/manga-8485-1901-LOGCUBE.fits.gz"
        ),
        "bytes": 38872704,
        "sha256": "eb152722718b2420b4a50081abe66a8cee0cdb1b23e84b9eaf0ad83ea14cd567",
        "magic": _GZIP,
        "suffix": ".fits.gz",
        "large": False,
        "description": (
            "SDSS-IV MaNGA DR17 LOGCUBE, plate-IFU 8485-1901 (19-fibre bundle, "
            "34 x 34 spaxels of 0.5 arcsec, 4563 log-lambda channels, 3622-10354 A); "
            "the smallest MaNGA cube format, a star-forming disc galaxy at z = 0.037."
        ),
        "license": "SDSS public data; cite SDSS DR17 and MaNGA (Bundy et al. 2015)",
    },
    "manga_8485_1901_rss": {
        "url": (
            "https://data.sdss.org/sas/dr17/manga/spectro/redux/v3_1_1/8485/"
            "stack/manga-8485-1901-LOGRSS.fits.gz"
        ),
        "bytes": 15762876,
        "sha256": "ff0b6e1700ba3ec578779632690d266884e2bb6352320ff6c1bcfcb21a3cb4a8",
        "magic": _GZIP,
        "suffix": ".fits.gz",
        "large": False,
        "description": (
            "SDSS-IV MaNGA DR17 LOGRSS (row-stacked spectra) of plate-IFU 8485-1901: "
            "171 fibre spectra (19 fibres x 9 dithered exposures) on the cube's "
            "log-lambda grid, with inverse variance, mask and per-wavelength fibre "
            "positions (XPOS/YPOS)."
        ),
        "license": "SDSS public data; cite SDSS DR17 and MaNGA (Bundy et al. 2015)",
    },
    "legacy_cartwheel_griz": {
        "url": (
            f"{_LEGACY}?ra=9.4213&dec=-33.7163&layer=ls-dr10&pixscale=0.524"
            "&bands=griz&size=128&invvar"
        ),
        "bytes": 535680,
        "sha256": "02775297fa6f9daa52f004620276e63d2e4f34afc9bfeaa974c9e77436d0e534",
        "magic": _FITS,
        "suffix": ".fits",
        "large": False,
        "description": (
            "Legacy Surveys DR10 g/r/i/z cutout of the Cartwheel galaxy, 128 x 128 px "
            "at 0.524 arcsec/px (TAN), image cube in HDU 0 and inverse variance in "
            "HDU 1, nanomaggies."
        ),
        "license": _LEGACY_CREDIT,
    },
    "legacy_cartwheel_gz_fine": {
        "url": (
            f"{_LEGACY}?ra=9.4223&dec=-33.7170&layer=ls-dr10&pixscale=0.35"
            "&bands=gz&size=160&invvar"
        ),
        "bytes": 420480,
        "sha256": "0555ef7a087d61860b20d71aefa85cb9fdcc6b344c5f89de5d9109f1f06b8084",
        "magic": _FITS,
        "suffix": ".fits",
        "large": False,
        "description": (
            "Legacy Surveys DR10 g/z cutout of the Cartwheel on a second grid: "
            "160 x 160 px at 0.35 arcsec/px, centre offset by 3 and 2.5 arcsec."
        ),
        "license": _LEGACY_CREDIT,
    },
    "legacy_ngc288_grz": {
        "url": (
            f"{_LEGACY}?ra=13.12&dec=-26.5525&layer=ls-dr10&pixscale=0.262"
            "&bands=grz&size=384&invvar"
        ),
        "bytes": 3548160,
        "sha256": "330711c8ea53077530b756c11c1af929098274c3f2eeff6df3bb3ced0328d41a",
        "magic": _FITS,
        "suffix": ".fits",
        "large": False,
        "description": (
            "Legacy Surveys DR10 g/r/z cutout 4 arcmin west of the globular cluster "
            "NGC 288, 384 x 384 px at the native 0.262 arcsec/px, image cube in "
            "HDU 0 and inverse variance in HDU 1, nanomaggies."
        ),
        "license": _LEGACY_CREDIT,
    },
    "legacy_ngc288_catalog": {
        "url": (
            "https://www.legacysurvey.org/viewer/ls-dr10/cat.fits"
            "?ralo=13.105&rahi=13.135&declo=-26.565&dechi=-26.540"
        ),
        "bytes": 429120,
        "sha256": "a519b6d4192e32a833548551c6606726a752a9d685a0da4c2d0a0b65aa894675",
        "magic": _FITS,
        "suffix": ".fits",
        "large": False,
        "description": (
            "Legacy Surveys DR10 Tractor catalogue of the same field (143 Gaia "
            "stars with Tractor g/r/z fluxes and inverse variances)."
        ),
        "license": _LEGACY_CREDIT + "; Gaia DR3 (ESA/DPAC) astrometry",
    },
    "gaia_pleiades_cone": {
        "url": _query_url(
            _GAIA_TAP,
            """SELECT source_id, ra, dec, parallax, parallax_error, pmra, pmdec,
            radial_velocity, phot_g_mean_mag, bp_rp FROM gaia_dr3_source
            WHERE 1=CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',56.75,24.1167,1.0))
            AND phot_g_mean_mag < 16 AND pmra IS NOT NULL ORDER BY source_id""",
        ),
        "bytes": 270095,
        "sha256": "21ab4d8c92f99dc8c7a6af4c53aa3f96fddd36cfe46f57d7c7c68e6ba88b28a3",
        "magic": (b"source_id",),
        "suffix": ".csv",
        "large": False,
        "description": (
            "Gaia DR3 snapshot from the IRSA TAP service (CSV): sources with G < 16 and a proper motion "
            "within 1 degree of the Pleiades (RA 56.75, Dec +24.12), ordered by "
            "source_id."
        ),
        "license": _GAIA_CREDIT,
    },
    "gaia_taurus_cone": {
        "url": _query_url(
            _GAIA_TAP,
            """SELECT source_id, ra, dec, parallax, phot_g_mean_mag,
            phot_bp_mean_mag, phot_rp_mean_mag FROM gaia_dr3_source
            WHERE 1=CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',64.5,27.0,1.0))
            AND phot_g_mean_mag < 17 AND phot_bp_mean_mag IS NOT NULL
            AND phot_rp_mean_mag IS NOT NULL AND parallax_over_error > 5
            ORDER BY source_id""",
        ),
        "bytes": 415482,
        "sha256": "5ff9b0b160052c4f49d453628701d0bdd34990a6be91dc5eb27f37c5a0341ea4",
        "magic": (b"source_id",),
        "suffix": ".csv",
        "large": False,
        "description": (
            "Gaia DR3 snapshot from the IRSA TAP service (CSV): G < 17 sources with BP/RP photometry and "
            "parallax_over_error > 5 within 1 degree of (RA 64.5, Dec +27.0) in "
            "the Taurus dark clouds, ordered by source_id."
        ),
        "license": _GAIA_CREDIT,
    },
    "sfd_ebv_healpix_512": {
        "url": "https://lambda.gsfc.nasa.gov/data/foregrounds/SFD/lambda_sfd_ebv.fits",
        "bytes": 25174080,
        "sha256": "a4a8b81dbc6da0c8a6e3562f04e07cf358cdae1862d2c68ae6eb44aa6bd71076",
        "magic": _FITS,
        "suffix": ".fits",
        "large": False,
        "description": (
            "Schlegel, Finkbeiner & Davis (1998) E(B-V) dust map resampled by LAMBDA "
            "to HEALPix nside=512, NESTED, Galactic coordinates (the smallest "
            "official HEALPix form of SFD)."
        ),
        "license": "NASA LAMBDA (public domain, US Government work); cite Schlegel et al. 1998",
    },
    "sdss_dr17_photoz_sample": {
        "url": _query_url(
            _SKYSERVER,
            """SELECT p.objid, p.ra, p.dec, s.z AS specz, p.modelFlux_u,
            p.modelFlux_g, p.modelFlux_r, p.modelFlux_i, p.modelFlux_z,
            p.modelFluxIvar_u, p.modelFluxIvar_g, p.modelFluxIvar_r,
            p.modelFluxIvar_i, p.modelFluxIvar_z, p.extinction_r, pz.z AS photoz
            FROM SpecObj AS s JOIN PhotoObj AS p ON p.objid = s.bestobjid
            JOIN Photoz AS pz ON pz.objid = p.objid
            WHERE s.class = 'GALAXY' AND s.zWarning = 0
            AND p.ra BETWEEN 150 AND 156 AND p.dec BETWEEN 0 AND 6
            ORDER BY p.objid""",
        ),
        "bytes": 1253554,
        "sha256": "fc7c2bcdadf2f79d5c60e4251423d72ec67b8575550e9c338219f8825c6e6db5",
        "magic": (b"#Table1",),
        "suffix": ".csv",
        "large": False,
        "description": (
            "SDSS DR17 SkyServer snapshot (CSV): every spectroscopic galaxy with "
            "zWarning = 0 in RA 150-156, Dec 0-6, with ugriz model fluxes and "
            "inverse variances (nanomaggies) and the DR17 Photoz (Beck et al. 2016) "
            "estimate."
        ),
        "license": "SDSS public data; cite SDSS DR17 (Abdurro'uf et al. 2022) and Beck et al. 2016",
    },
    "moc_sdss_dr9": {
        "url": f"{_IRSA_HIPS}/SDSS/DR9/color/Moc.fits",
        "bytes": 2998080,
        "sha256": "1b7327f91612db66086f75217e27638481d193b849d87d4249ae570baf260993",
        "magic": _FITS,
        "suffix": ".fits",
        "large": False,
        "description": "Coverage MOC (order 12) of the SDSS DR9 imaging HiPS.",
        "license": _HIPS_MOC_CREDIT + "; cite SDSS DR9 (Ahn et al. 2012)",
    },
    "moc_galex_ais_nuv": {
        "url": f"{_IRSA_HIPS}/GALEX/GR6-03-2014/AIS-ND/Moc.fits",
        "bytes": 293760,
        "sha256": "59f2dfdcaa8a98355dd7a33c8aa84992a7d44c259ff923ba03adac7e31dc90c7",
        "magic": _FITS,
        "suffix": ".fits",
        "large": False,
        "description": "Coverage MOC of the GALEX GR6 All-sky Imaging Survey NUV HiPS.",
        "license": _HIPS_MOC_CREDIT + "; GALEX (NASA), cite Bianchi et al. 2017",
    },
    "moc_erosita_efeds": {
        "url": "https://skies.esac.esa.int/eROSITA/Hips_rate_eFEDS_single_band/Moc.fits",
        "bytes": 8640,
        "sha256": "3b45b7689513149a1e9fa1491e65156eb3aaf93fae5584a625326635eb380709",
        "magic": _FITS,
        "suffix": ".fits",
        "large": False,
        "description": "Coverage MOC (order 7) of the eROSITA eFEDS field HiPS from ESASky.",
        "license": "ESASky HiPS (ESA); eROSITA EDR data (MPE, CC BY 4.0), cite Brunner et al. 2022",
    },
    **{
        f"svo_decam_{band}": {
            "url": f"{_SVO}CTIO/DECam.{band}",
            "bytes": size,
            "sha256": digest,
            "magic": _ASCII,
            "suffix": ".dat",
            "large": False,
            "description": f"DECam {band}-band total throughput (wavelength [A], transmission) from the SVO Filter Profile Service.",
            "license": "SVO Filter Profile Service (Rodrigo, Solano & Bayo 2012); DECam throughputs from CTIO/DES",
        }
        for band, size, digest in (
            (
                "g",
                4565,
                "35936fc027e8e88576c863ac9fe329fb13d0422c57552120dcee0543bf9bca3f",
            ),
            (
                "r",
                5017,
                "15c0d56823563a9829233a70d19a229593d7eb516560177dabbfd8a93d083956",
            ),
            (
                "i",
                5036,
                "f975476f14a3ef227a351b481576bd0d2d713dc9c113080e992a7a525c2bc97f",
            ),
            (
                "z",
                4937,
                "c5e6ab3c3806b0a058bfb97bd4758a4b30ade19096e3085231edf1a2ddbba96e",
            ),
        )
    },
    # JWST products on MAST are 100 MB to GBs and MAST has no server-side FITS
    # cutout for them; examples/_mast_cutout.py builds small cutouts by HTTP
    # range reads (registry: _mast_cutout.CUTOUTS).
}


def cache_dir() -> Path:
    """Sample cache root (``$TORCHSKY_SAMPLE_CACHE`` or ``~/.cache/torchsky/samples``)."""
    override = os.environ.get("TORCHSKY_SAMPLE_CACHE", "").strip()
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "torchsky" / "samples"


def fast_mode() -> bool:
    return os.environ.get("TORCHSKY_EXAMPLE_FAST", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_path(name: str, registry: dict[str, dict[str, Any]] | None = None) -> Path:
    reg = SAMPLES if registry is None else registry
    if name not in reg:
        raise KeyError(f"unknown sample {name!r}; choose from {sorted(reg)}")
    return cache_dir() / f"{name}{reg[name].get('suffix', '.fits')}"


def check_sample_file(spec: dict[str, Any], path: Path) -> str | None:
    """Return ``None`` when ``path`` is a valid copy of ``spec``, else the reason."""
    try:
        size = path.stat().st_size
        magics = tuple(spec.get("magic") or ())
        with path.open("rb") as fh:
            head = fh.read(max((len(m) for m in magics), default=0))
    except OSError as exc:
        return f"unreadable ({exc})"
    expected = spec.get("bytes")
    if size == 0:
        return "empty file"
    if expected is not None and size != expected:
        return f"size {size} != expected {expected}"
    if magics and not any(head.startswith(m) for m in magics):
        return f"unexpected leading bytes {head[:12]!r}"
    pinned = spec.get("sha256")
    if pinned and sha256_file(path) != pinned:
        return "sha256 mismatch"
    return None


def fetch_sample(
    name: str,
    *,
    registry: dict[str, dict[str, Any]] | None = None,
    allow_download: bool | None = None,
) -> Path:
    """Return the cached path of sample ``name``, downloading it once if needed.

    Raises :class:`SampleUnavailable` when the file is not cached and downloads
    are disabled (``TORCHSKY_EXAMPLE_FAST=1`` or ``allow_download=False``) or the
    transfer fails or yields bytes that fail validation.
    """
    reg = SAMPLES if registry is None else registry
    spec = reg[name] if name in reg else None
    if spec is None:
        raise KeyError(f"unknown sample {name!r}; choose from {sorted(reg)}")
    dest = sample_path(name, reg)
    dest.parent.mkdir(parents=True, exist_ok=True)
    note = ""
    if dest.is_file():
        problem = check_sample_file(spec, dest)
        if problem is None:
            return dest
        dest.unlink(missing_ok=True)
        note = f"; evicted a corrupt cached copy ({problem})"
    if allow_download is None:
        allow_download = not fast_mode()
    if not allow_download:
        raise SampleUnavailable(
            f"sample {name!r} is not cached at {dest}{note} and downloads are "
            "disabled (TORCHSKY_EXAMPLE_FAST); run `pixi run fetch-example-samples`"
        )
    url = spec["url"]
    tmp = dest.with_name(dest.name + ".partial")
    last: Exception | None = None
    for attempt in range(DOWNLOAD_ATTEMPTS):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "torchsky-examples"}
            )
            with urllib.request.urlopen(req, timeout=CONNECT_TIMEOUT_S) as response:  # noqa: S310 - registry URLs
                with tmp.open("wb") as fh:
                    shutil.copyfileobj(response, fh)
            problem = check_sample_file(spec, tmp)
            if problem is not None:
                raise OSError(f"downloaded file failed validation: {problem}")
            tmp.replace(dest)
            return dest
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            tmp.unlink(missing_ok=True)
            last = exc
            if isinstance(exc, urllib.error.HTTPError) and exc.code < 500:
                break
    raise SampleUnavailable(
        f"could not download sample {name!r} from {url}: {last}"
    ) from last


@contextlib.contextmanager
def skip_if_unavailable() -> Iterator[None]:
    """Exit with status 77 (runner SKIP) if a sample fetch inside the block fails.

    Usage: ``with skip_if_unavailable(): path = fetch_sample("horsehead")``.
    """
    try:
        yield
    except SampleUnavailable as exc:
        print(f"SKIP: {exc}")
        raise SystemExit(SKIP) from exc


def try_fetch_sample(name: str) -> Path | None:
    try:
        return fetch_sample(name)
    except SampleUnavailable:
        return None


if __name__ == "__main__":
    for key, meta in SAMPLES.items():
        print(f"{key:28s} {meta['bytes'] or '?':>12}  {meta['url']}", file=sys.stdout)
