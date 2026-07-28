#!/usr/bin/env python3
"""Stage the complete UPdec/SMACS J0723 detector set on CANFAR."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
from urllib.request import urlopen


OBSERVATION_GROUPS = ("02101", "02103", "02105")
DETECTORS = ("nrcb1", "nrcb2", "nrcb3", "nrcb4", "nrcblong")
MAST_DOWNLOAD = "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:JWST/product/"


def expected_cal_filenames() -> tuple[str, ...]:
    """Return the frozen 3 x 9 x 5 calibrated-exposure filename grid."""

    return tuple(
        f"jw02736001001_{group}_{pattern:05d}_{detector}_cal.fits"
        for group in OBSERVATION_GROUPS
        for pattern in range(1, 10)
        for detector in DETECTORS
    )


def is_complete_fits(path: Path) -> bool:
    """Check the minimum atomic-staging contract without a FITS dependency."""

    try:
        if path.stat().st_size < 2880:
            return False
        with path.open("rb") as stream:
            return stream.read(8) == b"SIMPLE  "
    except OSError:
        return False


def link_existing_crfs(source: Path, destination: Path) -> list[dict[str, object]]:
    """Link existing CRFs into the canonical data root without copying them."""

    destination.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for source_path in sorted(source.glob("*_crf.fits")):
        link = destination / source_path.name
        if link.is_symlink():
            if link.resolve() != source_path.resolve():
                raise RuntimeError(f"existing CRF link targets the wrong file: {link}")
        elif link.exists():
            raise RuntimeError(f"refusing to replace existing CRF path: {link}")
        else:
            link.symlink_to(source_path.resolve())
        records.append(
            {
                "name": source_path.name,
                "source": str(source_path.resolve()),
                "link": str(link),
                "bytes": source_path.stat().st_size,
            }
        )
    return records


def _download_cal(filename: str, destination: Path) -> dict[str, object]:
    path = destination / filename
    if is_complete_fits(path):
        return {"name": filename, "status": "reused", "bytes": path.stat().st_size}
    part = path.with_suffix(path.suffix + ".part")
    part.unlink(missing_ok=True)
    # ponytail: retries restart one 113 MB file; add HTTP Range only if MAST
    # transfer failures make whole-file retry a measured staging bottleneck.
    with urlopen(MAST_DOWNLOAD + filename, timeout=300) as response:
        with part.open("wb") as stream:
            shutil.copyfileobj(response, stream, length=8 * 1024 * 1024)
    if not is_complete_fits(part):
        part.unlink(missing_ok=True)
        raise RuntimeError(f"downloaded product is not a complete FITS file: {filename}")
    os.replace(part, path)
    return {"name": filename, "status": "downloaded", "bytes": path.stat().st_size}


def stage(
    *,
    root: Path,
    crf_source: Path,
    workers: int,
) -> dict[str, object]:
    """Complete the CAL set, link CRFs, and atomically write its manifest."""

    if workers < 1:
        raise ValueError("workers must be positive")
    cal_dir = root / "cal"
    crf_dir = root / "crf"
    cal_dir.mkdir(parents=True, exist_ok=True)
    filenames = expected_cal_filenames()
    records: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_download_cal, filename, cal_dir): filename
            for filename in filenames
        }
        for future in as_completed(futures):
            records.append(future.result())
    records.sort(key=lambda record: str(record["name"]))
    crfs = link_existing_crfs(crf_source, crf_dir)
    missing = [
        filename
        for filename in filenames
        if not is_complete_fits(cal_dir / filename)
    ]
    manifest = {
        "root": str(root.resolve()),
        "cal": {
            "expected": len(filenames),
            "complete": len(filenames) - len(missing),
            "downloaded": sum(record["status"] == "downloaded" for record in records),
            "reused": sum(record["status"] == "reused" for record in records),
            "missing": missing,
            "files": records,
        },
        "crf": {
            "linked": len(crfs),
            "source": str(crf_source.resolve()),
            "files": crfs,
        },
    }
    temporary = root / "manifest.json.part"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, root / "manifest.json")
    if missing:
        raise RuntimeError(f"{len(missing)} calibrated exposures remain missing")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/arc/projects/ots/torchsky-data/jwst_smacs0723"),
    )
    parser.add_argument(
        "--crf-source",
        type=Path,
        default=Path("/arc/projects/ots/jwst_smacs0723"),
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    manifest = stage(
        root=args.root,
        crf_source=args.crf_source,
        workers=args.workers,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
