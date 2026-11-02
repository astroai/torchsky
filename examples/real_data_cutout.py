"""Coadd overlapping pieces of a real CFHT MegaPipe cutout onto one TAN grid.

Fetches the registry sample ``cadc_megapipe_cutout`` (a 512x512 g-band cutout of
the CFHT MegaPipe stack MegaPipe.712.165 from the CADC cutout service, cached in
``~/.cache/torchsky/samples``), splits it into two overlapping sub-images with
their own shifted WCS, plans a target grid, coadds with a Lanczos3 kernel and
compares the result to the original pixels. Prints statistics and saves one
figure. Exits with status 77 (skip) when the sample is not cached and the
network is unavailable or ``TORCHSKY_EXAMPLE_FAST=1``. Pre-fetch with
``pixi run fetch-example-samples``. Pass ``--data DIR`` to coadd your own
``*.fits`` cutouts instead. Runtime a few seconds.

    pixi run -e test -- python examples/real_data_cutout.py [--data DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile

import torch

from _sample_data import SKIP, fetch_sample, skip_if_unavailable


def pieces(path: Path, hdu: int):
    """Return ``[(image, header, valid)]``; the sample is split into two overlapping column ranges."""
    import torchfits

    image = torchfits.read(str(path), hdu).to(torch.float32)
    header = dict(torchfits.read_header(str(path), hdu))
    if image.ndim != 2 or "CTYPE1" not in header:
        print(f"  skipping {path.name}: need a 2D image with a celestial WCS")
        return []
    valid = torch.isfinite(image)
    image = torch.where(valid, image, torch.zeros_like(image))
    print(
        f"  {path.name}: shape {tuple(image.shape)}, median {float(image[valid].median()):.4g}"
    )
    return [(image, header, valid)]


def split_columns(image, header, valid, bounds):
    out = []
    for x0, x1 in bounds:
        sub = dict(header)
        sub["CRPIX1"] = float(header["CRPIX1"]) - x0
        sub["NAXIS1"] = x1 - x0
        out.append((image[:, x0:x1], sub, valid[:, x0:x1]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--data", type=Path, default=None, help="directory of your own cutouts"
    )
    ap.add_argument("--glob", default="*.fits")
    ap.add_argument("--hdu", type=int, default=0)
    ap.add_argument("--kernel", default="lanczos3")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.data is not None:
        files = sorted(args.data.glob(args.glob)) if args.data.is_dir() else []
        if not files:
            print(f"SKIP: no files matching {args.glob} in {args.data}")
            return SKIP
        items = [p for f in files for p in pieces(f, args.hdu)]
        reference = None
    else:
        with skip_if_unavailable():
            path = fetch_sample("cadc_megapipe_cutout")
        ((image, header, valid),) = pieces(path, 0)
        reference = image
        items = split_columns(image, header, valid, [(0, 320), (192, 512)])
    if not items:
        print("SKIP: no usable 2D celestial images among the files found.")
        return SKIP

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from torchsky.map import Observation, coadd, plan

    observations = [
        Observation(data=img, wcs=hdr, weight_map=ok.float()) for img, hdr, ok in items
    ]
    target = plan(observations, projection="TAN")
    result = coadd(observations, target, kernel=args.kernel, device=args.device)
    data = result.data.cpu()
    covered = result.weight.cpu() > 0
    print(
        f"coadd of {len(observations)} images: shape {tuple(data.shape)}, "
        f"{int(covered.sum())} covered pixels, mean {float(data[covered].mean()):.4g}, device={args.device}"
    )
    if reference is not None and data.shape == reference.shape:
        both = covered & (reference != 0)
        resid = (data - reference)[both]
        print(
            f"coadd vs original: rms residual {float(resid.pow(2).mean().sqrt()):.3g} "
            f"(original std {float(reference.std()):.3g})"
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
