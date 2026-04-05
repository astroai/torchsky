#!/usr/bin/env python3
"""Download selected public UPdec-Webb products from NADC."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve


BASE_URL = "https://nadc.china-vo.org/res/file_upload/download?id={download_id}"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "updec_webb"


@dataclass(frozen=True)
class UpdecWebbFile:
    band: str
    iteration: int
    filename: str
    download_id: int

    @property
    def url(self) -> str:
        return BASE_URL.format(download_id=self.download_id)


UPDEC_WEBB_FILES: dict[tuple[str, int], UpdecWebbFile] = {
    ("F090W", 0): UpdecWebbFile("F090W", 0, "F090Wb_UPDC-v2_iter0_i2d.fits.fz", 47945),
    ("F090W", 10): UpdecWebbFile("F090W", 10, "F090Wb_UPDC-v2_iter10_i2d.fits.fz", 47947),
    ("F150W", 0): UpdecWebbFile("F150W", 0, "F150Wb_UPDC-v2_iter0_i2d.fits.fz", 47958),
    ("F150W", 10): UpdecWebbFile("F150W", 10, "F150Wb_UPDC-v2_iter10_i2d.fits.fz", 47959),
    ("F200W", 0): UpdecWebbFile("F200W", 0, "F200Wb_UPDC-v2_iter0_i2d.fits.fz", 47969),
    ("F200W", 10): UpdecWebbFile("F200W", 10, "F200Wb_UPDC-v2_iter10_i2d.fits.fz", 47970),
    ("F277W", 0): UpdecWebbFile("F277W", 0, "F277Wb_UPDC-v2_iter0_i2d.fits.fz", 47980),
    ("F277W", 10): UpdecWebbFile("F277W", 10, "F277Wb_UPDC-v2_iter10_i2d.fits.fz", 47981),
    ("F356W", 0): UpdecWebbFile("F356W", 0, "F356Wb_UPDC-v2_iter0_i2d.fits.fz", 47991),
    ("F356W", 10): UpdecWebbFile("F356W", 10, "F356Wb_UPDC-v2_iter10_i2d.fits.fz", 47992),
    ("F444W", 0): UpdecWebbFile("F444W", 0, "F444Wb_UPDC-v2_iter0_i2d.fits.fz", 48002),
    ("F444W", 10): UpdecWebbFile("F444W", 10, "F444Wb_UPDC-v2_iter10_i2d.fits.fz", 48003),
}


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def resolve_specs(bands: list[str], iterations: list[int]) -> list[UpdecWebbFile]:
    specs: list[UpdecWebbFile] = []
    for band in bands:
        for iteration in iterations:
            key = (band.upper(), int(iteration))
            if key not in UPDEC_WEBB_FILES:
                raise KeyError(f"No UPdec-Webb public mapping for {key}")
            specs.append(UPDEC_WEBB_FILES[key])
    return specs


def download_spec(spec: UpdecWebbFile, output_dir: Path, *, overwrite: bool) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / spec.filename
    if dest.exists() and not overwrite:
        return dest
    part = dest.with_suffix(dest.suffix + ".part")
    if part.exists():
        part.unlink()
    urlretrieve(spec.url, part)
    part.replace(dest)
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bands", default="F277W", help="Comma-separated band list")
    parser.add_argument(
        "--iterations",
        default="0,10",
        help="Comma-separated iteration list, usually 0 and 10",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--list", action="store_true", help="List known public files and exit")
    args = parser.parse_args()

    if args.list:
        for spec in UPDEC_WEBB_FILES.values():
            print(spec.band, spec.iteration, spec.filename, spec.url)
        return 0

    bands = [band.upper() for band in parse_csv(args.bands)]
    iterations = [int(value) for value in parse_csv(args.iterations)]
    specs = resolve_specs(bands, iterations)
    for spec in specs:
        dest = download_spec(spec, args.output_dir, overwrite=args.overwrite)
        print(dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
