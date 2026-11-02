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
import urllib.request
from collections.abc import Iterator
from typing import Any

SKIP = 77
CONNECT_TIMEOUT_S = 30.0
DOWNLOAD_ATTEMPTS = 3
_FITS = (b"SIMPLE  =",)
_GZIP = (b"\x1f\x8b",)
_CADC = "https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/CFHTSG"
_PLA = "https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/component-maps"


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
    # Follow-up: the JWST SMACS0723 NIRCam i2d products on MAST are ~1.7 GB each
    # and MAST offers no stable server-side FITS cutout URL; the mini-cutouts are
    # built by torchsky-harness scripts/build_smacs0723_minicutout.py.
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
