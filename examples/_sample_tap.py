"""Small catalog cones from public TAP services, cached with a sha256 manifest.

Stdlib HTTP only; results are requested as CSV and parsed with ``pyarrow``.
Each cone first runs ``SELECT COUNT(*)`` and fails if the delivered row count
differs (a silent server-side truncation). Cached CSVs live next to the sample
files, with a ``tap_manifest.json`` (query, url, rows, columns, sha256) that is
checked on every hit. ``TORCHSKY_EXAMPLE_FAST=1`` forbids network access.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from _sample_data import (
    CONNECT_TIMEOUT_S,
    SampleUnavailable,
    cache_dir,
    fast_mode,
    sha256_file,
)

SERVICES: dict[str, tuple[str, str, str, str]] = {
    # name: (sync endpoint, table, ra column, dec column)
    "gaia_dr3": (
        "https://gea.esac.esa.int/tap-server/tap/sync",
        "gaiadr3.gaia_source",
        "ra",
        "dec",
    ),
    "twomass": ("https://irsa.ipac.caltech.edu/TAP/sync", "fp_psc", "ra", "dec"),
    "allwise": (
        "https://irsa.ipac.caltech.edu/TAP/sync",
        "allwise_p3as_psd",
        "ra",
        "dec",
    ),
}
DEFAULT_COLUMNS: dict[str, tuple[str, ...]] = {
    "gaia_dr3": (
        "source_id", "ra", "dec", "parallax", "pmra", "pmdec", "phot_g_mean_mag",
    ),
    "twomass": ("designation", "ra", "dec", "j_m", "h_m", "k_m"),
    "allwise": ("designation", "ra", "dec", "w1mpro", "w2mpro"),
}
_MANIFEST = "tap_manifest.json"


def cone_query(
    service: str,
    ra: float,
    dec: float,
    radius_deg: float,
    columns: tuple[str, ...] | None = None,
    where: str | None = None,
    table: str | None = None,
) -> tuple[str, str]:
    """Return ``(select_query, count_query)`` for a circular cone."""
    _, svc_table, ra_col, dec_col = SERVICES[service]
    cols = ", ".join(columns or DEFAULT_COLUMNS[service])
    cond = (
        f"1=CONTAINS(POINT('ICRS',{ra_col},{dec_col}),"
        f"CIRCLE('ICRS',{ra!r},{dec!r},{radius_deg!r}))"
    )
    if where:
        cond += f" AND ({where})"
    tbl = table or svc_table
    return (
        f"SELECT {cols} FROM {tbl} WHERE {cond}",
        f"SELECT COUNT(*) AS n FROM {tbl} WHERE {cond}",
    )


def _post_csv(url: str, query: str) -> bytes:
    data = urllib.parse.urlencode(
        {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": query}
    ).encode()
    last: Exception | None = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                url, data=data, headers={"User-Agent": "torchsky-examples"}
            )
            with urllib.request.urlopen(req, timeout=CONNECT_TIMEOUT_S) as resp:  # noqa: S310 - public TAP endpoints
                return resp.read()
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last = exc
            time.sleep(2 * (attempt + 1))
    raise SampleUnavailable(f"TAP query to {url} failed: {last}")


def _manifest_path() -> Path:
    return cache_dir() / _MANIFEST


def _read_manifest() -> dict[str, Any]:
    try:
        return json.loads(_manifest_path().read_text())
    except (OSError, ValueError):
        return {}


def fetch_cone(
    service: str,
    ra: float,
    dec: float,
    radius_deg: float,
    *,
    columns: tuple[str, ...] | None = None,
    where: str | None = None,
    endpoint: str | None = None,
    table: str | None = None,
):
    """Return a ``pyarrow.Table`` for a cone, from cache or one TAP round trip.

    Raises :class:`SampleUnavailable` offline / in FAST mode with a cold cache,
    and ``RuntimeError`` when the row count differs from ``COUNT(*)``.
    """
    import pyarrow.csv as pacsv

    url = endpoint or SERVICES[service][0]
    select, count = cone_query(service, ra, dec, radius_deg, columns, where, table)
    key = hashlib.sha256(f"{url}\n{select}".encode()).hexdigest()[:16]
    path = cache_dir() / f"tap_{service}_{key}.csv"
    manifest = _read_manifest()
    entry = manifest.get(path.name)
    if entry and path.is_file() and entry.get("sha256") == sha256_file(path):
        return pacsv.read_csv(path)
    if fast_mode():
        raise SampleUnavailable(
            f"TAP cone {service} ({ra}, {dec}, r={radius_deg}) is not cached and "
            "downloads are disabled (TORCHSKY_EXAMPLE_FAST)"
        )
    n_expected = int(next(csv.DictReader(io.StringIO(_post_csv(url, count).decode())))["n"])
    body = _post_csv(url, select)
    table_ = pacsv.read_csv(io.BytesIO(body))
    if table_.num_rows != n_expected:
        raise RuntimeError(
            f"{service} cone returned {table_.num_rows} rows but COUNT(*) is "
            f"{n_expected} (truncated)"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    manifest[path.name] = {
        "service": service,
        "url": url,
        "query": select,
        "rows": n_expected,
        "columns": table_.column_names,
        "sha256": sha256_file(path),
        "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _manifest_path().write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return table_
