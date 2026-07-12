"""CFITSIO-backed FITS I/O surface — delegated to torchfits.

Only read_fast and I/O capability flags are kept locally; torchfits owns
everything else.
"""

from __future__ import annotations

from typing import Any

import torchfits
import torchsky.cpp as cpp

# ── public API — re-exported from torchfits ──────────────────────────
read = torchfits.read
write = torchfits.write
open = torchfits.open
get_header = torchfits.get_header
def get_wcs(path: str, hdu: int | str = 0, key: str = "", tables: Any = None) -> Any:
    """Read header and construct a torchsky-native WCS."""
    from torchsky.wcs import WCS

    if tables is None:
        with torchfits.open(path) as hdul:
            hdr = torchfits.get_header(path, _resolve_hdu(hdu, path))
            return WCS(dict(hdr), key=key, tables=hdul)

    hdr = torchfits.get_header(path, _resolve_hdu(hdu, path))
    return WCS(dict(hdr), key=key, tables=tables)


def _resolve_hdu(hdu: int | str, path: str) -> int:
    """Normalize ``hdu`` to an integer index."""
    if hdu != "auto":
        return int(hdu)
    for candidate in (0, 1):
        hdr = torchfits.get_header(path, candidate)
        if any(k in dict(hdr) for k in ("CTYPE1", "WCSAXES", "CRPIX1")):
            return candidate
    return 0
read_image = torchfits.read_tensor
read_table = torchfits.read_table
read_hdus = torchfits.read_hdus
read_subset = torchfits.read_subset
read_batch = torchfits.read_batch
# read_large_table in the pre-boundary torchfits API is replaced by read_table_rows;
# wrap it with the convenience surface torchsky callers expect.


def read_large_table(
    file_path: str,
    hdu: int = 1,
    max_memory_mb: int = 100,
    streaming: bool = False,
    return_iterator: bool = False,
) -> Any:
    return torchfits.read_table_rows(
        file_path,
        hdu=hdu, max_memory_mb=max_memory_mb,
        streaming=streaming, return_iterator=return_iterator,
    )

open_subset_reader = torchfits.open_subset_reader
stream_table = torchfits.stream_table
delete_hdu = torchfits.delete_hdu
insert_hdu = torchfits.insert_hdu
replace_hdu = torchfits.replace_hdu
get_batch_info = torchfits.get_batch_info
get_cache_performance = torchfits.get_cache_performance
clear_file_cache = torchfits.clear_file_cache
verify_checksums = torchfits.verify_checksums
write_checksums = torchfits.write_checksums


def read_fast(*args: Any, **kwargs: Any):
    """Fast-path C++ read — delegated to torchfits.read."""
    return torchfits.read(*args, **kwargs)


# ponytail: capability flags — kept for diagnostics; derived from cpp module
_HAS_READ_HDUS_BATCH = hasattr(cpp, "read_hdus_batch")
_HAS_READ_FULL_RAW_WITH_SCALE = hasattr(cpp, "read_full_raw_with_scale")
_HAS_READ_FULL_RAW = hasattr(cpp, "read_full_raw")
_HAS_READ_FULL_UNMAPPED_RAW = hasattr(cpp, "read_full_unmapped_raw")
_HAS_READ_FULL_UNMAPPED = hasattr(cpp, "read_full_unmapped")
_HAS_READ_FULL_NOCACHE = hasattr(cpp, "read_full_nocache")


__all__ = [
    "clear_file_cache",
    "delete_hdu",
    "get_batch_info",
    "get_cache_performance",
    "get_header",
    "get_wcs",
    "insert_hdu",
    "open",
    "open_subset_reader",
    "read",
    "read_batch",
    "read_fast",
    "read_hdus",
    "read_image",
    "read_large_table",
    "read_subset",
    "read_table",
    "replace_hdu",
    "stream_table",
    "verify_checksums",
    "write",
    "write_checksums",
]
