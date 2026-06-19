#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

BANDS = ["F090W", "F150W", "F277W", "F444W"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile", required=True, choices=["center", "right", "down", "right_down"])
    parser.add_argument("--root", type=Path, default=Path("data/smacs0723_tiles"))
    args = parser.parse_args()

    tile_dir = args.root / args.tile
    outputs = []
    crop_size = None
    data_root = None

    for band in BANDS:
        manifest_path = tile_dir / band / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if crop_size is None:
            crop_size = manifest.get("crop_size")
        if data_root is None:
            data_root = manifest.get("data_root")

        band_outputs = [
            item for item in manifest.get("outputs", [])
            if str(item.get("band", "")).upper() == band
        ]
        if len(band_outputs) != 1:
            raise ValueError(f"Expected one output for {band} in {manifest_path}")

        outputs.append(band_outputs[0])

    merged = {
        "bands": BANDS,
        "tile": args.tile,
        "crop_size": crop_size,
        "data_root": data_root,
        "output_dir": str(tile_dir),
        "outputs": outputs,
    }

    out_path = tile_dir / "manifest.json"
    out_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())