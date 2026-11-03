"""Small sky-centred cutouts of large JWST products on MAST, by HTTP range reads.

JWST calibrated exposures (``*_cal.fits``, 118 MB for one NIRCam long-wave
detector) and level-3 mosaics (``*_i2d.fits``, hundreds of MB) are far too large
for an example, and MAST offers no server-side FITS cutout for JWST. The MAST
download endpoint does honour HTTP ``Range`` requests, so this helper reads the
FITS headers block by block to locate each extension, then fetches only the
band of image rows that contains the cutout (rows of a FITS image are
contiguous on disk) and crops the columns locally. The cutout centre is a sky
position; the detector pixel is found with the SIP approximation that the JWST
pipeline writes into the ``SCI`` header.

Each cutout is written once, with torchfits, to the example sample cache
(:func:`_sample_data.cache_dir`) as a small multi-extension FITS file whose
``SCI`` header carries the shifted WCS and the provenance cards
(``ORIGFILE``, ``CAL_VER``, ``CRDS_CTX``, ``CUTX0``/``CUTY0``). It obeys the
same network policy as :func:`_sample_data.fetch_sample`: under
``TORCHSKY_EXAMPLE_FAST=1`` an uncached cutout raises
:class:`_sample_data.SampleUnavailable`.

MAST reprocesses JWST data when the calibration pipeline improves, so the
remote bytes are not immutable and no sha256 is pinned; each cutout instead
checks the extension shapes and types against the registry and records the
pipeline version it was built from. Only the 2880-byte header blocks are parsed
here to find byte offsets; the cutout itself is written and read with torchfits.
"""

from __future__ import annotations

from math import prod
from pathlib import Path
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import torch

from _sample_data import (
    CONNECT_TIMEOUT_S,
    DOWNLOAD_ATTEMPTS,
    SampleUnavailable,
    cache_dir,
    fast_mode,
)

MAST_FILE = "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:JWST/product/"
BLOCK = 2880
_SMACS_LICENCE = (
    "JWST ERO programme 2736 (SMACS J0723.3-7327), public via MAST (STScI); "
    "cite Pontoppidan et al. 2022 ApJL 936 L14 and acknowledge MAST/STScI"
)
SMACS_CENTRE = (110.8349, -73.4545)  # deg, ICRS: east of the cluster BCG, on NRCBLONG

# One entry per cutout: MAST product, extensions to copy, sky centre, size.
CUTOUTS: dict[str, dict[str, Any]] = {
    **{
        f"smacs0723_f444w_cal_{k}": {
            "product": f"jw02736001001_02105_0000{k}_nrcblong_cal.fits",
            "extensions": ("SCI", "ERR", "VAR_RNOISE"),
            "centre": SMACS_CENTRE,
            "size": 256,
            "license": _SMACS_LICENCE,
            "description": (
                f"JWST NIRCam F444W calibrated exposure {k} of the 9-point "
                "INTRAMODULEBOX dither (NRCBLONG, 837 s), 256 x 256 px cutout of "
                "SCI, ERR and VAR_RNOISE (MJy/sr) with the pipeline SIP WCS; pixels flagged "
                "DO_NOT_USE are NaN in SCI"
            ),
        }
        for k in (1, 2, 3, 4)
    },
    "smacs0723_f444w_i2d": {
        "product": "jw02736-o001_t001_nircam_clear-f444w_i2d.fits",
        "extensions": ("SCI",),
        "centre": SMACS_CENTRE,
        "size": 280,
        "license": _SMACS_LICENCE,
        "description": (
            "JWST NIRCam F444W level-3 pipeline mosaic of all 9 dithers and both "
            "modules (drizzle, 0.063 arcsec/px), 280 x 280 px cutout of SCI"
        ),
    },
}


def _get(url: str, start: int, stop: int) -> bytes:
    """Bytes ``[start, stop)`` of ``url`` with an HTTP range request."""
    last: Exception | None = None
    for _ in range(DOWNLOAD_ATTEMPTS):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "torchsky-examples",
                    "Range": f"bytes={start}-{stop - 1}",
                },
            )
            with urllib.request.urlopen(req, timeout=CONNECT_TIMEOUT_S) as resp:  # noqa: S310 - MAST URL
                if resp.status != 206:
                    raise OSError(
                        f"server ignored the range request (HTTP {resp.status})"
                    )
                body = resp.read()
            if len(body) != stop - start:
                raise OSError(f"short range read: {len(body)} of {stop - start} bytes")
            return body
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last = exc
            if isinstance(exc, urllib.error.HTTPError) and exc.code < 500:
                break
    raise SampleUnavailable(f"range read of {url} failed: {last}") from last


def _card_value(text: str) -> Any:
    """Value of one header card (string, logical, integer or float)."""
    text = text.strip()
    if text.startswith("'"):
        out, i = [], 1
        while i < len(text):
            if text[i] == "'":
                if text[i + 1 : i + 2] == "'":
                    out.append("'")
                    i += 2
                    continue
                break
            out.append(text[i])
            i += 1
        return "".join(out).rstrip()
    text = text.split("/", 1)[0].strip()
    if text in ("T", "F"):
        return text == "T"
    try:
        return int(text)
    except ValueError:
        return float(text.replace("D", "E"))


def _read_header(url: str, offset: int) -> tuple[dict[str, Any], int]:
    """Parse the header starting at byte ``offset``; return it and its length."""
    raw = b""
    while True:
        raw += _get(url, offset + len(raw), offset + len(raw) + 8 * BLOCK)
        cards = [
            raw[i : i + 80].decode("ascii", "replace") for i in range(0, len(raw), 80)
        ]
        end = next((i for i, c in enumerate(cards) if c.startswith("END     ")), None)
        if end is not None:
            break
    header: dict[str, Any] = {}
    for card in cards[:end]:
        key = card[:8].strip()
        if card[8:10] != "= " or not key or key in header:
            continue
        try:
            header[key] = _card_value(card[10:])
        except ValueError:
            continue
    length = -(-(end + 1) * 80 // BLOCK) * BLOCK
    return header, length


def _layout(
    url: str, wanted: tuple[str, ...]
) -> tuple[dict[str, Any], dict[str, tuple[dict[str, Any], int]]]:
    """Primary header and ``{EXTNAME: (header, data offset)}`` for ``wanted``."""
    primary, offset = _read_header(url, 0)
    found: dict[str, tuple[dict[str, Any], int]] = {}
    while len(found) < len(wanted):
        header, length = _read_header(url, offset)
        data_offset = offset + length
        naxis = int(header.get("NAXIS", 0))
        n = prod(int(header[f"NAXIS{i}"]) for i in range(1, naxis + 1)) if naxis else 0
        nbytes = (
            abs(int(header["BITPIX"]))
            // 8
            * int(header.get("GCOUNT", 1))
            * (n + int(header.get("PCOUNT", 0)))
        )
        name = str(header.get("EXTNAME", "")).strip()
        if name in wanted:
            found[name] = (header, data_offset)
        offset = data_offset + -(-nbytes // BLOCK) * BLOCK
    return primary, found


_DTYPES = {8: ">u1", 16: ">i2", 32: ">i4", 64: ">i8", -32: ">f4", -64: ">f8"}
_WCS_PREFIXES = (
    "CTYPE",
    "CRVAL",
    "CRPIX",
    "CD1_",
    "CD2_",
    "PC1_",
    "PC2_",
    "CDELT",
    "CUNIT",
    "A_",
    "B_",
    "AP_",
    "BP_",
)
_KEEP = (
    "BUNIT",
    "PHOTMJSR",
    "PIXAR_SR",
    "PIXAR_A2",
    "RADESYS",
    "EQUINOX",
    "XPOSURE",
    "PIXFRAC",
    "RESWHT",
)
_PRIMARY_KEEP = (
    "TELESCOP",
    "INSTRUME",
    "DETECTOR",
    "FILTER",
    "PUPIL",
    "EXPSTART",
    "EFFEXPTM",
    "CAL_VER",
    "CRDS_CTX",
    "PATTTYPE",
    "PATT_NUM",
    "NUMDTHPT",
)


def _rows(
    url: str, header: dict[str, Any], offset: int, y0: int, y1: int, x0: int, x1: int
) -> torch.Tensor:
    """Rows ``y0:y1``, columns ``x0:x1`` (0-based) of one image extension."""
    width = int(header["NAXIS1"])
    bitpix = int(header["BITPIX"])
    nb = abs(bitpix) // 8
    raw = _get(url, offset + y0 * width * nb, offset + y1 * width * nb)
    arr = np.frombuffer(raw, dtype=_DTYPES[bitpix]).reshape(y1 - y0, width)[:, x0:x1]
    arr = arr.astype(arr.dtype.newbyteorder("="))
    bzero, bscale = header.get("BZERO", 0), header.get("BSCALE", 1)
    if bzero or bscale != 1:
        arr = arr * bscale + bzero
    return torch.from_numpy(np.ascontiguousarray(arr))


def _build(spec: dict[str, Any], dest: Path) -> None:
    import torchfits
    from torchsky.wcs import WCS

    url = MAST_FILE + urllib.parse.quote(spec["product"])
    wanted = tuple(spec["extensions"])
    primary, exts = _layout(url, wanted)
    sci, _ = exts["SCI"]
    height, width = int(sci["NAXIS2"]), int(sci["NAXIS1"])
    wcs_cards = {
        k: v
        for k, v in sci.items()
        if k.startswith(_WCS_PREFIXES)
        or k in ("A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER")
    }
    centre = torch.tensor([list(spec["centre"])], dtype=torch.float64)
    px = WCS(
        {"NAXIS": 2, "NAXIS1": width, "NAXIS2": height, **wcs_cards}
    ).world_to_pixel(centre, origin=0)[0]
    size = int(spec["size"])
    x0 = int(round(float(px[0]))) - size // 2
    y0 = int(round(float(px[1]))) - size // 2
    if x0 < 0 or y0 < 0 or x0 + size > width or y0 + size > height:
        raise SampleUnavailable(
            f"{spec['product']}: cutout at pixel ({x0}, {y0}) falls off the {width} x {height} image"
        )
    hdus = []
    for name in wanted:
        header, offset = exts[name]
        if (int(header["NAXIS1"]), int(header["NAXIS2"])) != (width, height):
            raise SampleUnavailable(f"{spec['product']}: {name} shape differs from SCI")
        if int(header["BITPIX"]) != -32:
            raise SampleUnavailable(f"{spec['product']}: {name} is not a float32 image")
        data = _rows(url, header, offset, y0, y0 + size, x0, x0 + size)
        card: dict[str, Any] = {"EXTNAME": name}
        if name == "SCI":
            card.update({k: primary[k] for k in _PRIMARY_KEEP if k in primary})
            card.update({k: sci[k] for k in _KEEP if k in sci})
            card.update(wcs_cards)
            card["CRPIX1"] = float(sci["CRPIX1"]) - x0
            card["CRPIX2"] = float(sci["CRPIX2"]) - y0
            card.update({"ORIGFILE": spec["product"], "CUTX0": x0, "CUTY0": y0})
        hdus.append(torchfits.TensorHDU(data, torchfits.Header(card)))
    tmp = dest.with_name(dest.name + ".partial")
    torchfits.HDUList(hdus).write(str(tmp), overwrite=True)
    tmp.replace(dest)


def _valid(spec: dict[str, Any], path: Path) -> bool:
    import torchfits

    try:
        if torchfits.read_num_hdus(str(path)) != len(spec["extensions"]):
            return False
        header = torchfits.read_header(str(path), 0)
        size = int(spec["size"])
        return (header["NAXIS1"], header["NAXIS2"]) == (
            size,
            size,
        ) and "CAL_VER" in header
    except Exception:  # noqa: BLE001 - any unreadable cache entry is rebuilt
        return False


def fetch_cutout(name: str, *, allow_download: bool | None = None) -> Path:
    """Return the cached cutout ``name``, building it from MAST once if needed."""
    spec = CUTOUTS[name]
    dest = cache_dir() / "mast" / f"{name}.fits"
    if dest.is_file() and _valid(spec, dest):
        return dest
    dest.unlink(missing_ok=True)
    if allow_download is None:
        allow_download = not fast_mode()
    if not allow_download:
        raise SampleUnavailable(
            f"MAST cutout {name!r} is not cached at {dest} and downloads are "
            "disabled (TORCHSKY_EXAMPLE_FAST); run `pixi run fetch-example-samples`"
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    _build(spec, dest)
    if not _valid(spec, dest):
        dest.unlink(missing_ok=True)
        raise SampleUnavailable(
            f"MAST cutout {name!r} failed validation after the build"
        )
    return dest
