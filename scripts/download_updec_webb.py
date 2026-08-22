#!/usr/bin/env python3
"""Download selected public UPdec-Webb products from NADC."""

from __future__ import annotations

import argparse
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


BASE_URL = "https://nadc.china-vo.org/res/file_upload/download?id={download_id}"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "updec_webb"


@dataclass(frozen=True)
class UpdecWebbFile:
    band: str
    module: str
    iteration: int
    filename: str
    download_id: int

    @property
    def url(self) -> str:
        return BASE_URL.format(download_id=self.download_id)


# Module B (cluster core): UPDC-v2 i2d products, compressed.
# Module A (parallel field): UPDC-v1.1 SCI_clear products. Module-A ids were
# resolved from the NADC release downloads on 2026-08-21 by reading each id's
# Content-Disposition header; ids marked "contiguous" are inferred from their
# verified neighbors within one contiguous per-band block.
UPDEC_WEBB_FILES: dict[tuple[str, str, int], UpdecWebbFile] = {
    # --- module b (v2) ---

    ("F090W", "b", 0): UpdecWebbFile(
        "F090W", "b", 0, "F090Wb_UPDC-v2_iter0_i2d.fits.fz", 47945
    ),
    ("F090W", "b", 10): UpdecWebbFile(
        "F090W", "b", 10, "F090Wb_UPDC-v2_iter10_i2d.fits.fz", 47947
    ),
    ("F150W", "b", 0): UpdecWebbFile(
        "F150W", "b", 0, "F150Wb_UPDC-v2_iter0_i2d.fits.fz", 47958
    ),
    ("F150W", "b", 10): UpdecWebbFile(
        "F150W", "b", 10, "F150Wb_UPDC-v2_iter10_i2d.fits.fz", 47959
    ),
    ("F200W", "b", 0): UpdecWebbFile(
        "F200W", "b", 0, "F200Wb_UPDC-v2_iter0_i2d.fits.fz", 47969
    ),
    ("F200W", "b", 10): UpdecWebbFile(
        "F200W", "b", 10, "F200Wb_UPDC-v2_iter10_i2d.fits.fz", 47970
    ),
    ("F277W", "b", 0): UpdecWebbFile(
        "F277W", "b", 0, "F277Wb_UPDC-v2_iter0_i2d.fits.fz", 47980
    ),
    ("F277W", "b", 10): UpdecWebbFile(
        "F277W", "b", 10, "F277Wb_UPDC-v2_iter10_i2d.fits.fz", 47981
    ),
    ("F356W", "b", 0): UpdecWebbFile(
        "F356W", "b", 0, "F356Wb_UPDC-v2_iter0_i2d.fits.fz", 47991
    ),
    ("F356W", "b", 10): UpdecWebbFile(
        "F356W", "b", 10, "F356Wb_UPDC-v2_iter10_i2d.fits.fz", 47992
    ),
    ("F444W", "b", 0): UpdecWebbFile(
        "F444W", "b", 0, "F444Wb_UPDC-v2_iter0_i2d.fits.fz", 48002
    ),
    ("F444W", "b", 10): UpdecWebbFile(
        "F444W", "b", 10, "F444Wb_UPDC-v2_iter10_i2d.fits.fz", 48003
    ),
}

_F090WA_ITER_IDS = {
    0: 46475, 1: 46478, 2: 46480, 3: 46482, 4: 46484,
    5: 46486, 6: 46487, 7: 46488, 8: 46489, 9: 46490, 10: 46491,
}


def _contiguous_a_block(band: str, first_id: int) -> dict[int, int]:
    """Contiguous per-band block: iter00 at ``first_id``, stride 1."""
    return {iteration: first_id + iteration for iteration in range(11)}


_MODULE_A_ITER_IDS: dict[str, dict[int, int]] = {
    "F090W": _F090WA_ITER_IDS,
    # contiguous blocks (verified anchors inside each block)
    "F150W": _contiguous_a_block("F150W", 46503),   # iter03=46506, iter05=46508
    "F200W": _contiguous_a_block("F200W", 46525),   # iter04=46529, iter10=46535
    # F277W: non-contiguous; iter00 uses the release-page-grounded duplicate
    # id 46474, iters 01-10 verified in the 46547..46556 block.
    "F277W": {
        **{0: 46474},
        **{i: 46546 + i for i in range(1, 11)},
    },
    "F356W": _contiguous_a_block("F356W", 46568),   # iter01=46569, iter09=46577
    "F444W": _contiguous_a_block("F444W", 46590),   # iter01=46591, iter05=46595
}

for _band, _ids in _MODULE_A_ITER_IDS.items():
    for _iteration, _download_id in _ids.items():
        _filename = (
            f"JWST_NIRCAM_{_band}a_UPDC-v1.1_SCI_clear_iter{_iteration:02d}.fits"
        )
        UPDEC_WEBB_FILES[(_band, "a", _iteration)] = UpdecWebbFile(
            _band, "a", _iteration, _filename, _download_id
        )
del _band, _ids, _iteration, _download_id, _filename


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def resolve_specs(
    bands: list[str],
    iterations: list[int],
    modules: list[str] | tuple[str, ...] = ("b",),
) -> list[UpdecWebbFile]:
    specs: list[UpdecWebbFile] = []
    for module in modules:
        module = str(module).lower()
        if module not in {"a", "b"}:
            raise ValueError(f"module must be a or b, got {module!r}")
        for band in bands:
            for iteration in iterations:
                key = (band.upper(), module, int(iteration))
                if key not in UPDEC_WEBB_FILES:
                    raise KeyError(f"No UPdec-Webb public mapping for {key}")
                specs.append(UPDEC_WEBB_FILES[key])
    return specs


def _download_with_retries(
    url: str,
    dest: Path,
    *,
    attempts: int,
    timeout_s: float,
    backoff_s: float,
) -> None:
    last_error: BaseException | None = None
    for attempt in range(1, max(1, int(attempts)) + 1):
        try:
            with urlopen(url, timeout=float(timeout_s)) as response:
                with dest.open("wb") as fh:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        fh.write(chunk)
            return
        except HTTPError:
            dest.unlink(missing_ok=True)
            raise
        except (
            URLError,
            OSError,
            ConnectionResetError,
            TimeoutError,
            socket.timeout,
        ) as exc:
            dest.unlink(missing_ok=True)
            last_error = exc
            if attempt >= max(1, int(attempts)):
                break
            time.sleep(float(backoff_s) * float(2 ** (attempt - 1)))
    assert last_error is not None
    raise RuntimeError(
        f"failed to download {url} after {max(1, int(attempts))} attempts"
    ) from last_error


def download_spec(
    spec: UpdecWebbFile,
    output_dir: Path,
    *,
    overwrite: bool,
    attempts: int = 4,
    timeout_s: float = 240.0,
    backoff_s: float = 1.0,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / spec.filename
    if dest.exists() and not overwrite:
        return dest
    part = dest.with_suffix(dest.suffix + ".part")
    if part.exists():
        part.unlink()
    _download_with_retries(
        spec.url,
        part,
        attempts=attempts,
        timeout_s=timeout_s,
        backoff_s=backoff_s,
    )
    part.replace(dest)
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bands", default="F277W", help="Comma-separated band list")
    parser.add_argument(
        "--module",
        "--modules",
        dest="modules",
        default="b",
        help="Comma-separated NIRCam module list (a and/or b)",
    )
    parser.add_argument(
        "--iterations",
        default="0,10",
        help="Comma-separated iteration list, usually 0 and 10",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--timeout-s", type=float, default=240.0)
    parser.add_argument("--backoff-s", type=float, default=1.0)
    parser.add_argument(
        "--list", action="store_true", help="List known public files and exit"
    )
    args = parser.parse_args()

    bands = [band.upper() for band in parse_csv(args.bands)]
    iterations = [int(value) for value in parse_csv(args.iterations)]
    modules = [module.lower() for module in parse_csv(args.modules)]

    if args.list:
        specs = resolve_specs(bands, iterations, modules)
        for spec in specs:
            print(spec.band, spec.module, spec.iteration, spec.filename, spec.url)
        return 0
    specs = resolve_specs(bands, iterations, modules)
    for spec in specs:
        dest = download_spec(
            spec,
            args.output_dir,
            overwrite=args.overwrite,
            attempts=int(args.attempts),
            timeout_s=float(args.timeout_s),
            backoff_s=float(args.backoff_s),
        )
        print(dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
