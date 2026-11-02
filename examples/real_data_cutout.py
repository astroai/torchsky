"""Coadd real survey cutouts (opt-in; skips cleanly when no data is present).

Reads 2D FITS cutouts with ``torchfits`` from a data directory (``--data`` or
``$TORCHSKY_DATA``; default pattern ``*.fits``), builds one target grid with
``plan`` and combines them with a Lanczos3 ``coadd``. Prints per-file and
coadd statistics and saves one figure. No network access happens unless you
pass ``--fetch URL ...``, which downloads the given FITS URLs (for example a
MAST ``Download/file?uri=...`` link or a CADC cutout service URL) into the data
directory first. The SMACS J0723 JWST mini-cutouts used by the harness
notebooks are produced by ``scripts/build_smacs0723_minicutout.py --download``
in ``torchsky-harness`` and are picked up with
``--data $TORCHSKY_DATA/smacs0723_minicutout --glob 'smacs0723_*_cutout_*.fits'``.
Exits with status 77 (skip) when no FITS files are found. Runtime a few
seconds for small cutouts.

    pixi run -e test -- python examples/real_data_cutout.py --data DIR [--fetch URL ...]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile
import urllib.request

import torch

SKIP = 77


def fetch(urls: list[str], dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for url in urls:
        name = Path(url.split("?")[0]).name or "download.fits"
        if "uri=" in url:
            name = Path(url.split("uri=")[-1]).name
        target = dest / name
        if not target.exists():
            print(f"fetching {url} -> {target}")
            urllib.request.urlretrieve(url, target)  # noqa: S310 - user-supplied URL


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--data", type=Path, default=None, help="directory with FITS cutouts"
    )
    ap.add_argument("--glob", default="*.fits")
    ap.add_argument("--fetch", nargs="*", default=None, metavar="URL")
    ap.add_argument("--hdu", type=int, default=0)
    ap.add_argument("--kernel", default="lanczos3")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    data_dir = args.data or (
        Path(os.environ["TORCHSKY_DATA"]) if os.environ.get("TORCHSKY_DATA") else None
    )
    if args.fetch:
        if data_dir is None:
            data_dir = Path(tempfile.mkdtemp(prefix="torchsky-real-data-"))
        fetch(args.fetch, data_dir)
    files = sorted(data_dir.glob(args.glob)) if data_dir and data_dir.is_dir() else []
    if not files:
        print(
            "SKIP: no FITS cutouts found. Pass --data DIR (or set TORCHSKY_DATA), "
            "or --fetch URL ... to download some; see the module docstring."
        )
        return SKIP

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torchfits

    from torchsky.map import Observation, coadd, plan

    observations = []
    for path in files:
        image = torchfits.read(str(path), args.hdu).to(torch.float32)
        header = dict(torchfits.read_header(str(path), args.hdu))
        if image.ndim != 2 or "CTYPE1" not in header:
            print(f"  skipping {path.name}: need a 2D image with a celestial WCS")
            continue
        finite = torch.isfinite(image)
        image = torch.where(finite, image, torch.zeros_like(image))
        print(
            f"  {path.name}: shape {tuple(image.shape)}, median {float(image[finite].median()):.4g}"
        )
        observations.append(
            Observation(data=image, wcs=header, weight_map=finite.float())
        )
    if not observations:
        print("SKIP: no usable 2D celestial images among the files found.")
        return SKIP

    target = plan(observations, projection="TAN")
    result = coadd(observations, target, kernel=args.kernel, device=args.device)
    data = result.data.cpu()
    covered = result.weight.cpu() > 0
    print(
        f"coadd of {len(observations)} cutouts: shape {tuple(data.shape)}, "
        f"{int(covered.sum())} covered pixels, mean {float(data[covered].mean()):.4g}, device={args.device}"
    )

    out = args.out or Path(tempfile.mkdtemp(prefix="torchsky-real-cutout-"))
    out.mkdir(parents=True, exist_ok=True)
    lo, hi = torch.quantile(data[covered], torch.tensor([0.05, 0.995]))
    fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
    ax.imshow(data, origin="lower", cmap="gray", vmin=float(lo), vmax=float(hi))
    ax.axis("off")
    fig.savefig(out / "real_data_cutout.png", dpi=100)
    print(f"figure: {out / 'real_data_cutout.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
