#!/usr/bin/env python3
"""Generate or check the frozen torchsky public-API manifest.

The manifest records the names each public namespace exports through ``__all__``.
It is intentionally a name-level contract: it does not import private helpers or
certify behavior.

Run from the repository root through Pixi:
    pixi run api-manifest
    pixi run api-manifest-check
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

NAMESPACES = (
    "torchsky",
    "torchsky.wcs",
    "torchsky.sphere",
    "torchsky.spectral",
    "torchsky.spectrophotometry",
    "torchsky.redshift",
    "torchsky.log_lambda",
    "torchsky.extinction",
    "torchsky.cosmology",
    "torchsky.transforms",
    "torchsky.quality",
    "torchsky.viz",
    "torchsky.map",
    "torchsky.scene",
    "torchsky.field",
    "torchsky.catalogs",
    "torchsky.coords",
    "torchsky.encoders",
    "torchsky.map.likelihood",
    "torchsky.map.query",
    "torchsky.map.evidence",
    "torchsky.map.operators.nuisance",
    "torchsky.map.operators.continuous_response",
    "torchsky.map.operators.detector_response",
    "torchsky.map.operators.pixel_calibration",
    "torchsky.map.operators.epsf",
    "torchsky.map.field.native_response",
    "torchsky.map.field.native_integrated_read",
    "torchsky.map.field.gaussian",
    "torchsky.map.field.joint_native_likelihood",
)


def collect() -> dict[str, list[str]]:
    manifest: dict[str, list[str]] = {}
    for name in NAMESPACES:
        module = importlib.import_module(name)
        exported = getattr(module, "__all__", None)
        if exported is None:
            raise SystemExit(f"{name} does not define __all__")
        manifest[name] = sorted(str(item) for item in exported)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", type=Path, metavar="PATH")
    mode.add_argument("--check", type=Path, metavar="PATH")
    args = parser.parse_args(argv)

    current = collect()
    if args.write is not None:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(
            json.dumps(current, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.write}")
        return 0

    expected = json.loads(args.check.read_text(encoding="utf-8"))
    if current == expected:
        print(f"API manifest matches {args.check}")
        return 0

    failed = False
    for namespace in sorted(set(current) | set(expected)):
        old = set(expected.get(namespace, []))
        new = set(current.get(namespace, []))
        added = sorted(new - old)
        removed = sorted(old - new)
        if added or removed:
            failed = True
            print(f"{namespace}:")
            if added:
                print(f"  added: {added}")
            if removed:
                print(f"  removed: {removed}")
    if failed:
        print(
            "API manifest mismatch; review the change, then run "
            "`pixi run api-manifest`."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
