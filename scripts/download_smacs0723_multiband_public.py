#!/usr/bin/env python3
"""Sequentially download public SMACS J0723 references and exposures for multiple bands."""

from __future__ import annotations

import argparse
from pathlib import Path

from download_jwst_smacs0723_mast import DEFAULT_OUTPUT_DIR as DEFAULT_JWST_DIR
from download_jwst_smacs0723_mast import download_product as download_jwst_product
from download_jwst_smacs0723_mast import select_products
from download_updec_webb import DEFAULT_OUTPUT_DIR as DEFAULT_UPDEC_DIR
from download_updec_webb import download_spec as download_updec_spec
from download_updec_webb import resolve_specs


ALL_BANDS = ("F090W", "F150W", "F200W", "F277W", "F356W", "F444W")
FULL_PUBLIC_MAX_EXPOSURES = 100


def _parse_csv(value: str) -> list[str]:
    return [part.strip().upper() for part in str(value).split(",") if part.strip()]


def _default_max_exposures_for_band(band: str) -> int:
    del band
    return FULL_PUBLIC_MAX_EXPOSURES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bands", default=",".join(ALL_BANDS))
    parser.add_argument("--module", default="b")
    parser.add_argument("--subgroup", default="CAL")
    parser.add_argument("--jwst-output-dir", type=Path, default=DEFAULT_JWST_DIR)
    parser.add_argument("--updec-output-dir", type=Path, default=DEFAULT_UPDEC_DIR)
    parser.add_argument("--updec-iterations", default="0")
    parser.add_argument("--updec-attempts", type=int, default=4)
    parser.add_argument("--updec-timeout-s", type=float, default=240.0)
    parser.add_argument("--updec-backoff-s", type=float, default=1.0)
    parser.add_argument(
        "--short-band-max-exposures", type=int, default=FULL_PUBLIC_MAX_EXPOSURES
    )
    parser.add_argument(
        "--long-band-max-exposures", type=int, default=FULL_PUBLIC_MAX_EXPOSURES
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    bands = _parse_csv(args.bands)
    iterations = [int(value) for value in _parse_csv(args.updec_iterations)]

    updec_specs = resolve_specs(bands, iterations, modules=(args.module,))
    if args.list:
        for spec in updec_specs:
            print(
                f"UPDEC {spec.band}{spec.module} iter{spec.iteration} "
                f"{spec.filename}"
            )
        for band in bands:
            max_exposures = (
                args.long_band_max_exposures
                if band in {"F277W", "F356W", "F444W"}
                else args.short_band_max_exposures
            )
            for product in select_products(
                band=band,
                subgroup=args.subgroup.upper(),
                max_exposures=max_exposures,
                module=args.module,
            ):
                print(f"JWST {band} {product.product_filename}")
        return 0

    for spec in updec_specs:
        print(
            download_updec_spec(
                spec,
                args.updec_output_dir,
                overwrite=args.overwrite,
                attempts=int(args.updec_attempts),
                timeout_s=float(args.updec_timeout_s),
                backoff_s=float(args.updec_backoff_s),
            )
        )

    for band in bands:
        max_exposures = (
            args.long_band_max_exposures
            if band in {"F277W", "F356W", "F444W"}
            else args.short_band_max_exposures
        )
        if max_exposures <= 0:
            max_exposures = _default_max_exposures_for_band(band)
        products = select_products(
            band=band,
            subgroup=args.subgroup.upper(),
            max_exposures=max_exposures,
            module=args.module,
        )
        for product in products:
            print(
                download_jwst_product(
                    product, args.jwst_output_dir, overwrite=args.overwrite
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
