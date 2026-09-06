#!/usr/bin/env python3
"""Query and download public JWST SMACS J0723 products from MAST."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests


MAST_INVOKE = "https://mast.stsci.edu/api/v0/invoke"
MAST_DOWNLOAD = "https://mast.stsci.edu/api/v0.1/Download/file?uri={uri}"
DEFAULT_PROPOSAL_ID = "2736"
DEFAULT_TARGET = "SMACS-J0723.3-7327"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "jwst_smacs0723"


def _detector_module_tokens(band: str, module: str | None) -> tuple[str, ...]:
    if module is None:
        return ()
    module_lower = str(module).lower()
    if module_lower not in {"a", "b", "any"}:
        raise ValueError("module must be one of {'a', 'b', 'any'}")
    if module_lower == "any":
        return ()
    band_upper = str(band).upper()
    if band_upper in {"F277W", "F356W", "F444W"}:
        return (f"nrc{module_lower}long",)
    return tuple(f"nrc{module_lower}{idx}" for idx in range(1, 5))


@dataclass(frozen=True)
class MastProduct:
    parent_obsid: int
    product_filename: str
    data_uri: str
    product_type: str
    subgroup: str | None
    size: int
    filters: str | None

    @property
    def download_url(self) -> str:
        return MAST_DOWNLOAD.format(uri=quote(self.data_uri, safe=":"))


def _invoke(request: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(
        MAST_INVOKE, data={"request": json.dumps(request)}, timeout=120
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") != "COMPLETE":
        raise RuntimeError(f"MAST request failed: {payload}")
    return payload


def query_nircam_observations(
    *,
    proposal_id: str = DEFAULT_PROPOSAL_ID,
    target_name: str = DEFAULT_TARGET,
) -> list[dict[str, Any]]:
    payload = _invoke(
        {
            "service": "Mast.Caom.Filtered",
            "params": {
                "columns": "obsid,obs_collection,proposal_id,target_name,instrument_name,filters,dataRights,obs_title",
                "filters": [
                    {"paramName": "obs_collection", "values": ["JWST"]},
                    {"paramName": "proposal_id", "values": [proposal_id]},
                    {"paramName": "target_name", "values": [target_name]},
                    {"paramName": "instrument_name", "values": ["NIRCAM/IMAGE"]},
                ],
            },
            "format": "json",
            "pagesize": 200,
            "page": 1,
        }
    )
    return list(payload["data"])


def query_products(obsid: int) -> list[MastProduct]:
    payload = _invoke(
        {
            "service": "Mast.Caom.Products",
            "params": {"obsid": str(obsid)},
            "format": "json",
            "pagesize": 2000,
            "page": 1,
        }
    )
    products: list[MastProduct] = []
    for row in payload["data"]:
        products.append(
            MastProduct(
                parent_obsid=int(row["parent_obsid"]),
                product_filename=row["productFilename"],
                data_uri=row["dataURI"],
                product_type=row["productType"],
                subgroup=row.get("productSubGroupDescription"),
                size=int(row["size"]),
                filters=row.get("filters"),
            )
        )
    return products


def select_products(
    *,
    band: str,
    subgroup: str,
    max_exposures: int,
    require_detector_split: bool = True,
    module: str | None = None,
    filename_contains: str | None = None,
    proposal_id: str = DEFAULT_PROPOSAL_ID,
    target_name: str = DEFAULT_TARGET,
) -> list[MastProduct]:
    if max_exposures < 1:
        raise ValueError("max_exposures must be positive")
    band = band.upper()
    observations = query_nircam_observations(
        proposal_id=proposal_id, target_name=target_name
    )
    matches = [
        obs for obs in observations if str(obs.get("filters", "")).upper() == band
    ]
    if not matches:
        raise LookupError(
            f"No observations found for band={band}, proposal={proposal_id}"
        )
    # A physical product can appear in several MAST observations.
    products = {
        product.data_uri: product
        for observation in matches
        for product in query_products(int(observation["obsid"]))
    }
    subgroup_upper = subgroup.upper()
    detector_tokens = _detector_module_tokens(band, module)
    chosen = [
        product
        for product in products.values()
        if product.product_type == "SCIENCE"
        and (
            (product.subgroup or "").upper() == subgroup_upper
            or (product.subgroup or "").upper().endswith(f"_{subgroup_upper}")
        )
        and str(product.filters).upper() == band
        and (
            not require_detector_split
            or (
                detector_tokens
                and any(
                    token in product.product_filename.lower()
                    for token in detector_tokens
                )
            )
            or (
                not detector_tokens
                and (
                    "nrcalong" in product.product_filename
                    or "nrcblong" in product.product_filename
                )
            )
        )
        and (
            filename_contains is None
            or filename_contains.lower() in product.product_filename.lower()
        )
    ]
    chosen.sort(key=lambda product: product.product_filename)
    return chosen[:max_exposures]


def download_product(
    product: MastProduct, output_dir: Path, *, overwrite: bool
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / product.product_filename
    if dest.exists() and not overwrite and dest.stat().st_size == product.size:
        return dest
    part = dest.with_suffix(dest.suffix + ".part")
    if part.exists():
        part.unlink()
    with requests.get(product.download_url, stream=True, timeout=240) as response:
        response.raise_for_status()
        with part.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
    if part.stat().st_size != product.size:
        part.unlink(missing_ok=True)
        raise RuntimeError(
            f"downloaded size mismatch for {product.product_filename}: "
            f"{part.stat().st_size if part.exists() else 0} != {product.size}"
        )
    part.replace(dest)
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--band", default="F277W")
    parser.add_argument(
        "--module",
        default="b",
        help="Detector module family to download: a, b, or any",
    )
    parser.add_argument(
        "--subgroup", default="CAL", help="Product subgroup, e.g. CAL or RATE"
    )
    parser.add_argument("--max-exposures", type=int, default=2)
    parser.add_argument(
        "--allow-mosaic-products",
        action="store_true",
        help="Do not require detector-split filenames like nrcalong/nrcblong; useful for I2D products",
    )
    parser.add_argument(
        "--filename-contains",
        default="",
        help="Optional case-insensitive filename substring filter, e.g. nrcblong",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    products = select_products(
        band=args.band.upper(),
        subgroup=args.subgroup.upper(),
        max_exposures=args.max_exposures,
        require_detector_split=not args.allow_mosaic_products,
        module=args.module,
        filename_contains=args.filename_contains or None,
    )
    if args.list:
        for product in products:
            print(product.product_filename, product.size, product.download_url)
        return 0
    for product in products:
        print(download_product(product, args.output_dir, overwrite=args.overwrite))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
