#!/usr/bin/env python3
"""Download SMACS J0723 products across every matching MAST observation.

This leaves TorchSky's original ``scripts/download_jwst_smacs0723_mast.py``
unchanged. It reuses that module's MAST client and download implementation,
but collects products from every observation matching the requested band.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_SCRIPTS = ROOT / "scripts"
if str(UPSTREAM_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_SCRIPTS))

from download_jwst_smacs0723_mast import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROPOSAL_ID,
    DEFAULT_TARGET,
    MastProduct,
    _detector_module_tokens,
    download_product,
    query_nircam_observations,
    query_products,
)


def select_products_all_observations(
    *,
    band: str,
    subgroup: str,
    max_exposures: int,
    module: str | None,
    filename_contains: str | None = None,
    proposal_id: str = DEFAULT_PROPOSAL_ID,
    target_name: str = DEFAULT_TARGET,
) -> list[MastProduct]:
    band_upper = str(band).upper()
    subgroup_upper = str(subgroup).upper()
    observations = query_nircam_observations(
        proposal_id=proposal_id,
        target_name=target_name,
    )
    matches = [
        row
        for row in observations
        if str(row.get("filters", "")).upper() == band_upper
    ]
    if not matches:
        raise LookupError(
            f"No observations found for band={band_upper}, "
            f"proposal={proposal_id}, target={target_name}"
        )

    # One physical product can appear through more than one CAOM observation.
    # data_uri is the stable identifier used for deduplication.
    products_by_uri: dict[str, MastProduct] = {}
    for observation in matches:
        for product in query_products(int(observation["obsid"])):
            products_by_uri[product.data_uri] = product

    detector_tokens = _detector_module_tokens(band_upper, module)
    selected = [
        product
        for product in products_by_uri.values()
        if product.product_type == "SCIENCE"
        and (
            (product.subgroup or "").upper() == subgroup_upper
            or (product.subgroup or "").upper().endswith(f"_{subgroup_upper}")
        )
        and str(product.filters).upper() == band_upper
        and (
            not detector_tokens
            or any(
                token in product.product_filename.lower()
                for token in detector_tokens
            )
        )
        and (
            filename_contains is None
            or filename_contains.lower() in product.product_filename.lower()
        )
    ]
    selected.sort(key=lambda product: product.product_filename)
    return selected[: int(max_exposures)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--band", default="F277W")
    parser.add_argument("--module", default="b", choices=("a", "b", "any"))
    parser.add_argument("--subgroup", default="CAL")
    parser.add_argument("--max-exposures", type=int, default=100)
    parser.add_argument("--filename-contains", default="")
    parser.add_argument("--proposal-id", default=DEFAULT_PROPOSAL_ID)
    parser.add_argument("--target-name", default=DEFAULT_TARGET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    products = select_products_all_observations(
        band=args.band,
        subgroup=args.subgroup,
        max_exposures=int(args.max_exposures),
        module=args.module,
        filename_contains=args.filename_contains or None,
        proposal_id=args.proposal_id,
        target_name=args.target_name,
    )
    print(f"selected {len(products)} products")
    for product in products:
        if args.list:
            print(
                product.product_filename,
                product.size,
                product.download_url,
            )
        else:
            print(
                download_product(
                    product,
                    args.output_dir,
                    overwrite=bool(args.overwrite),
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
