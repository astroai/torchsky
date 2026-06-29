#!/usr/bin/env python3
"""CFHT MegaCam mosaic download and torchsky coadd pipeline.

Downloads Elixir-processed individual exposures from a MegaPipe stacked
mosaic via the CADC TAP + caom2ops services, then coadds with torchsky::

  # List available MegaPipe mosaics
  pixi run -e test -- python scripts/download_cfht_megapipe.py list

  # Show info for one mosaic
  pixi run -e test -- python scripts/download_cfht_megapipe.py info MegaPipe.712.165

  # Download input exposures for a band
  pixi run -e test -- python scripts/download_cfht_megapipe.py download \\
      MegaPipe.712.165 --band g --data-dir ./data/cfht

  # Process downloaded exposures
  pixi run -e test -- python scripts/download_cfht_megapipe.py process \\
      ./data/cfht/MegaPipe.712.165/g/ --output mosaic_g.fits --device cuda

  # Or all at once
  pixi run -e test -- python scripts/download_cfht_megapipe.py run \\
      MegaPipe.712.165 --band g --data-dir ./data/cfht --output mosaic_g.fits
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

CADC_TAP = "https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/argus/sync"
CADC_DOWNLOAD = "https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/caom2ops/pkg"

# MegaPipe filter name mapping (MegaPipe → common name)
FILTER_NAMES: dict[str, str] = {
    "g.MP9401": "g",
    "r.MP9601": "r",
    "i.MP9701": "i",
    "z.MP9801": "z",
    "u.MP9301": "u",
    "Y.MP9201": "y",
}

NS = {"v": "http://www.ivoa.net/xml/VOTable/v1.3"}


# ---------------------------------------------------------------------------
# CADC TAP helpers
# ---------------------------------------------------------------------------

def _tap_query(query: str) -> list[list[str | None]]:
    """Execute synchronous ADQL query against CADC TAP, return rows."""
    data = urllib.parse.urlencode(
        {"REQUEST": "doQuery", "LANG": "ADQL", "QUERY": query},
        safe="':,",
    ).encode("utf-8")
    req = urllib.request.Request(CADC_TAP, data=data, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            tree = ET.parse(r)
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        if "ERROR" in body:
            # Try to extract error message from VOTable
            try:
                et = ET.fromstring(body)
                info = et.find(".//v:INFO[@name='QUERY_STATUS']", NS)
                if info is not None and info.get("value") == "ERROR":
                    print(f"TAP error: {info.text}", file=sys.stderr)
            except ET.ParseError:
                pass
        raise
    table = tree.find(".//v:TABLEDATA", NS)
    if table is None:
        return []
    return [[c.text for c in row] for row in table.findall("v:TR", NS)]


def list_mosaics(
    *, limit: int = 30, target_filter: str | None = None
) -> list[dict[str, Any]]:
    """List available MegaPipe mosaics."""
    extra = ""
    if target_filter:
        extra = f" AND LOWER(o.target_name) LIKE LOWER('%{target_filter}%')"
    query = (
        "SELECT o.observationID, o.target_name, o.metaRelease, "
        "  p.energy_bandpassName, p.provenance_inputs, p.calibrationLevel, "
        "  o.obsID "
        "FROM caom2.Plane AS p "
        "JOIN caom2.Observation AS o ON p.obsID=o.obsID "
        "WHERE o.collection='CFHTMEGAPIPE' "
        "  AND p.dataProductType='image' "
        f"  AND p.calibrationLevel=3{extra} "
        "ORDER BY o.metaRelease DESC"
    )
    rows = _tap_query(query)
    seen: dict[str, dict[str, Any]] = {}
    for r in rows:
        if len(r) < 6:
            continue
        obs_id, target, release, band, inputs, cal = r[:6]
        if not obs_id or not target:
            continue
        key = f"{obs_id}"
        if key not in seen:
            seen[key] = {
                "observationID": obs_id,
                "target_name": target,
                "metaRelease": release,
                "bands": {},
            }
        if band:
            n_inputs = len((inputs or "").split())
            seen[key]["bands"][band] = n_inputs
        if len(seen) >= limit:
            break
    return list(seen.values())[:limit]


def get_mosaic_info(obs_id: str) -> dict[str, Any]:
    """Get detailed info for a single MegaPipe mosaic."""
    query = (
        "SELECT o.observationID, o.target_name, o.metaRelease, "
        "  p.energy_bandpassName, p.provenance_inputs, p.publisherID, "
        "  p.planeURI "
        "FROM caom2.Plane AS p "
        "JOIN caom2.Observation AS o ON p.obsID=o.obsID "
        "WHERE o.collection='CFHTMEGAPIPE' "
        f"  AND o.observationID='{obs_id}' "
        "  AND p.dataProductType='image' "
        "  AND p.calibrationLevel=3"
    )
    rows = _tap_query(query)
    info: dict[str, Any] = {
        "observationID": obs_id,
        "target_name": None,
        "metaRelease": None,
        "bands": {},
    }
    for r in rows:
        if len(r) < 7:
            continue
        _, target, release, band, inputs, pub_id, _ = r[:7]
        info["target_name"] = info["target_name"] or target
        info["metaRelease"] = info["metaRelease"] or release
        if band:
            urns = (inputs or "").split()
            info["bands"][band] = {
                "n_inputs": len(urns),
                "input_urns": urns,
                "publisherID": pub_id,
            }
    return info


def resolve_cfht_exposures(
    caom_urns: list[str],
) -> list[dict[str, Any]]:
    """Resolve CAOM URIs (caom:CFHT/.../...) to CFHT plane metadata.

    Returns list of dicts with keys: obsID, productID, publisherID,
    dataRelease, calibrationLevel.
    """
    exposures: list[dict[str, Any]] = []
    for urn in caom_urns:
        # urn format: caom:CFHT/{obsID}/{productID}
        parts = urn.split("/")
        if len(parts) < 3:
            continue
        obs_id_part = parts[-2]
        product_id = parts[-1]
        query = (
            "SELECT o.observationID, p.productID, p.publisherID, "
            "  p.dataRelease, p.calibrationLevel, p.planeURI "
            "FROM caom2.Plane AS p "
            "JOIN caom2.Observation AS o ON p.obsID=o.obsID "
            "WHERE o.collection='CFHT' "
            f"  AND o.observationID='{obs_id_part}' "
            f"  AND p.productID='{product_id}'"
        )
        rows = _tap_query(query)
        for r in rows:
            if len(r) < 6:
                continue
            obs_id, prod_id, pub_id, data_release, cal_level, plane_uri = r[:6]
            if pub_id:
                exposures.append(
                    {
                        "obsID": obs_id,
                        "productID": prod_id,
                        "publisherID": pub_id,
                        "dataRelease": data_release,
                        "calibrationLevel": int(cal_level) if cal_level else None,
                        "download_url": f"{CADC_DOWNLOAD}?ID={urllib.parse.quote(pub_id, safe=':/?')}",
                        "filename": _filename_from_urn(obs_id, prod_id, pub_id),
                    }
                )
    return exposures


def _filename_from_urn(
    obs_id: str | None, prod_id: str | None, pub_id: str | None
) -> str:
    if pub_id:
        stem = pub_id.rsplit("/", 1)[-1].rsplit("?", 1)[-1]
    elif prod_id:
        stem = prod_id
    elif obs_id:
        stem = obs_id
    else:
        stem = "unknown"
    return f"{stem}.fits.fz"


def _download_file(url: str, dest: Path, *, retries: int = 3) -> bool:
    """Download a file with retry. Returns True on success.

    CADC caom2ops returns a tar archive wrapping the actual .fits.fz file
    (tar entry path ``CFHT/{filename}``). This function extracts the inner
    FITS file automatically.
    """
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url)
            # Download full response into memory (tar archives are small enough)
            with urllib.request.urlopen(req, timeout=300) as src:
                raw = src.read()
                if not raw:
                    print(f"  empty response, retry {attempt}/{retries}", file=sys.stderr)
                    continue

            # Check if response is a tar archive (CADC caom2ops) or raw FITS
            if raw[:5] == b"CFHT/":
                # Tar archive: extract inner file
                with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tf:
                    tf.write(raw)
                    tar_path = tf.name
                try:
                    with tarfile.open(tar_path, "r") as tar:
                        members = tar.getmembers()
                        # Find the .fits.fz entry inside the archive
                        fits_member = None
                        for m in members:
                            if m.name.endswith(".fits.fz") or m.name.endswith(".fits"):
                                fits_member = m
                                break
                        if fits_member is None:
                            print(f"  no FITS entry in tar, retry {attempt}/{retries}", file=sys.stderr)
                            continue
                        fobj = tar.extractfile(fits_member)
                        if fobj is None:
                            print(f"  tar entry {fits_member.name} is not a file, retry {attempt}/{retries}", file=sys.stderr)
                            continue
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        with open(dest, "wb") as f:
                            f.write(fobj.read())
                finally:
                    os.unlink(tar_path)
            else:
                # Raw FITS: write directly
                dest.parent.mkdir(parents=True, exist_ok=True)
                with open(dest, "wb") as f:
                    f.write(raw)

            # Verify it looks like a FITS file
            with open(dest, "rb") as f:
                header = f.read(100)
            if header[:6] not in (b"CFHT/ ", b"SIMPLE"):
                dest.unlink()
                print(
                    f"  bad FITS signature, retry {attempt}/{retries}",
                    file=sys.stderr,
                )
                continue
            size_mb = dest.stat().st_size / (1024 * 1024)
            print(f"  {dest.name} ({size_mb:.0f} MB)", file=sys.stderr)
            return True
        except Exception as exc:
            print(f"  download error: {exc}, retry {attempt}/{retries}", file=sys.stderr)
            if attempt < retries:
                time.sleep(2 ** attempt)
    return False


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


def cmd_list(args: argparse.Namespace) -> int:
    mosaics = list_mosaics(
        limit=int(args.limit), target_filter=args.target
    )
    if not mosaics:
        print("No MegaPipe mosaics found.", file=sys.stderr)
        return 1
    for m in mosaics:
        bands = ", ".join(
            f"{b}({n})" for b, n in sorted(m["bands"].items())
        )
        print(
            f"{m['observationID']:25s}  {m['target_name']:20s}  "
            f"{m['metaRelease'] or '?':19s}  [{bands}]"
        )
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    info = get_mosaic_info(args.observationID)
    if not info["target_name"]:
        print(f"Mosaic {args.observationID} not found.", file=sys.stderr)
        return 1
    print(f"Observation: {info['observationID']}")
    print(f"Target:      {info['target_name']}")
    print(f"Release:     {info['metaRelease']}")
    print("Bands:")
    for band, bd in sorted(info["bands"].items()):
        common = FILTER_NAMES.get(band, band)
        print(f"  {band:15s} ({common:2s})  {bd['n_inputs']} exposures")
        if args.verbose:
            for urn in bd["input_urns"]:
                print(f"    {urn}")
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    """Download input exposures for a given mosaic+band."""
    info = get_mosaic_info(args.observationID)
    if not info["target_name"]:
        print(f"Mosaic {args.observationID} not found.", file=sys.stderr)
        return 1

    # Find the requested band
    band = args.band
    matched = None
    for key, bd in info["bands"].items():
        if key == band or key.startswith(band + ".") or key.rsplit(".", 1)[0] == band:
            matched = (key, bd)
            break
    if matched is None:
        available = ", ".join(sorted(info["bands"].keys()))
        print(
            f"Band '{band}' not found. Available: {available}",
            file=sys.stderr,
        )
        return 1
    band_key, band_data = matched
    common = FILTER_NAMES.get(band_key, band_key)

    data_dir = Path(args.data_dir)
    out_dir = data_dir / info["observationID"] / common
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve CFHT exposure URLs
    urns = band_data["input_urns"]
    print(
        f"Resolving {len(urns)} exposures for {info['observationID']} "
        f"band {band_key} ...",
        file=sys.stderr,
    )
    exposures = resolve_cfht_exposures(urns)
    if not exposures:
        print("No CFHT exposures resolved.", file=sys.stderr)
        return 1

    public = all(
        e["dataRelease"] and e["dataRelease"] < "2026-07-01" for e in exposures
    )
    if not public:
        print(
            "Warning: some exposures may still be proprietary "
            "(need CADC certificate).",
            file=sys.stderr,
        )

    if args.dry_run:
        print(f"Would download {len(exposures)} exposures to {out_dir}:")
        for e in exposures:
            print(f"  {e['filename']}  [{e['download_url']}]")
        return 0

    print(f"Downloading {len(exposures)} exposures to {out_dir} ...")
    n_ok = 0
    for e in exposures:
        dest = out_dir / e["filename"]
        if dest.exists() and dest.stat().st_size > 1_000_000:
            print(f"  {dest.name} exists, skipping", file=sys.stderr)
            n_ok += 1
            continue
        # Respect rate limiting
        if n_ok > 0:
            time.sleep(0.5)
        ok = _download_file(e["download_url"], dest)
        if ok:
            n_ok += 1

    print(f"Downloaded {n_ok}/{len(exposures)} files to {out_dir}", file=sys.stderr)
    return 0 if n_ok == len(exposures) else 1


def cmd_process(args: argparse.Namespace) -> int:
    """Run torchsky map coadd --sky on a directory of exposures."""
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        return 1

    fits_files = sorted(input_dir.glob("*.fits*"))
    if not fits_files:
        print(f"No FITS files in {input_dir}", file=sys.stderr)
        return 1

    # Exclude the output file if it lives in the input directory
    out_path = Path(args.output).resolve()
    exposure_files = [f for f in fits_files if f.resolve() != out_path]

    if len(exposure_files) < int(args.min_exposures):
        print(
            f"Only {len(exposure_files)} exposures (need at least {args.min_exposures})",
            file=sys.stderr,
        )
        return 1

    # Cap to --max-exposures
    max_exp = int(args.max_exposures)
    if len(exposure_files) > max_exp:
        print(
            f"Capping to first {max_exp} of {len(exposure_files)} exposures "
            f"(use --max-exposures to change)",
            file=sys.stderr,
        )
        exposure_files = exposure_files[:max_exp]

    print(
        f"Processing {len(exposure_files)} exposures with torchsky map coadd --sky ..."
    )

    # Build the torchsky map command
    cmd = [
        sys.executable, "-c",
        "from torchsky.cli import main; main()",
        "map",
        str(args.output),
    ] + [str(f) for f in exposure_files]

    # Optional flags
    if args.reference_header:
        cmd.extend(["--reference", str(args.reference_header)])
    cmd.extend(["--hdu", "0"])
    cmd.extend(["--kernel", args.kernel])
    cmd.extend(["--stack", args.stack])
    cmd.extend(["--sky"])
    cmd.extend(["--sky-method", args.sky_method])
    cmd.extend(["--sky-statistic", args.sky_statistic])
    cmd.extend(["--sky-block-size", str(args.sky_block_size)])
    if args.device:
        cmd.extend(["--device", args.device])
    if args.dry_run:
        cmd.append("--dry-run")

    print(f"Running: {' '.join(cmd)}", file=sys.stderr)
    if args.dry_run:
        return 0

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"torchsky map exited with code {result.returncode}", file=sys.stderr)
        return result.returncode

    print(f"Wrote coadd to {args.output}", file=sys.stderr)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Download + process in one step."""
    # Setup args for download
    dl_args = argparse.Namespace(
        observationID=args.observationID,
        band=args.band,
        data_dir=args.data_dir,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )
    rc = cmd_download(dl_args)
    if rc != 0:
        return rc

    # Determine input dir
    info = get_mosaic_info(args.observationID)
    band = args.band
    matched = None
    for key in info.get("bands", {}):
        if key == band or key.startswith(band + ".") or key.rsplit(".", 1)[0] == band:
            matched = key
            break
    common = FILTER_NAMES.get(matched or band, matched or band)
    input_dir = Path(args.data_dir) / args.observationID / common

    # Setup args for process
    proc_args = argparse.Namespace(
        input_dir=str(input_dir),
        output=args.output,
        reference_header=args.reference_header,
        kernel=args.kernel,
        stack=args.stack,
        sky_method=args.sky_method,
        sky_statistic=args.sky_statistic,
        sky_block_size=args.sky_block_size,
        device=args.device,
        dry_run=args.dry_run,
        min_exposures=1,
        max_exposures=args.max_exposures,
    )
    return cmd_process(proc_args)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CFHT MegaCam mosaic download and torchsky coadd pipeline"
    )
    sub = p.add_subparsers(dest="command", required=True)

    # list
    lp = sub.add_parser("list", help="List available MegaPipe mosaics")
    lp.add_argument("--limit", default=30, type=int, help="Max mosaics to show")
    lp.add_argument("--target", default=None, help="Filter by target name (substring)")
    lp.set_defaults(handler=cmd_list)

    # info
    ip = sub.add_parser("info", help="Show details for a MegaPipe mosaic")
    ip.add_argument("observationID", help="e.g. MegaPipe.146.158")
    ip.add_argument("--verbose", "-v", action="store_true")
    ip.set_defaults(handler=cmd_info)

    # download
    dp = sub.add_parser("download", help="Download input exposures")
    dp.add_argument("observationID", help="e.g. MegaPipe.146.158")
    dp.add_argument("--band", default="g", help="Band to download (g, r, i, z, u, Y)")
    dp.add_argument("--data-dir", default="./data/cfht", help="Download directory root")
    dp.add_argument("--dry-run", action="store_true", help="Print URLs, don't download")
    dp.add_argument("--verbose", "-v", action="store_true")
    dp.set_defaults(handler=cmd_download)

    # process
    pp = sub.add_parser("process", help="Run torchsky coadd on downloaded exposures")
    pp.add_argument("input_dir", help="Directory with downloaded .fits.fz files")
    pp.add_argument("--output", "-o", default="mosaic.fits", help="Output FITS path")
    pp.add_argument("--reference-header", default=None, help="Reference FITS for WCS grid")
    pp.add_argument("--kernel", default="bilinear", help="Resampling kernel")
    pp.add_argument("--stack", default="weighted_mean", help="Stacking method")
    pp.add_argument("--sky-method", default="mesh", choices=("mesh", "polynomial"))
    pp.add_argument("--sky-statistic", default="huber", choices=("median", "huber", "tukey"))
    pp.add_argument("--sky-block-size", type=int, default=128)
    pp.add_argument("--device", default=None, help="torch device")
    pp.add_argument("--dry-run", action="store_true")
    pp.add_argument("--min-exposures", type=int, default=1)
    pp.add_argument("--max-exposures", type=int, default=10)
    pp.set_defaults(handler=cmd_process)

    # run (download + process)
    rp = sub.add_parser("run", help="Download and process in one step")
    rp.add_argument("observationID", help="e.g. MegaPipe.146.158")
    rp.add_argument("--band", default="g", help="Band to download (g, r, i, z, u, Y)")
    rp.add_argument("--data-dir", default="./data/cfht", help="Download directory root")
    rp.add_argument("--output", "-o", default="mosaic.fits", help="Output FITS path")
    rp.add_argument("--reference-header", default=None)
    rp.add_argument("--kernel", default="bilinear")
    rp.add_argument("--stack", default="weighted_mean")
    rp.add_argument("--sky-method", default="mesh")
    rp.add_argument("--sky-statistic", default="huber")
    rp.add_argument("--sky-block-size", type=int, default=128)
    rp.add_argument("--device", default=None)
    rp.add_argument("--dry-run", action="store_true")
    rp.add_argument("--verbose", "-v", action="store_true")
    rp.add_argument("--max-exposures", type=int, default=10)
    rp.set_defaults(handler=cmd_run)

    return p


def main() -> int:
    p = _build_parser()
    args = p.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
