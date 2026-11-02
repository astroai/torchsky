"""Render slide-ready all-sky figures with ``torchsky.viz``.

Builds a synthetic HEALPix sky (a galactic-plane-like band, a large-scale cloud
field and a compact source) and writes a small gallery in the styles used for
talks: a dark Mollweide map, a paper-toned Hammer map with a MOC footprint and
catalog overlay, and a north-polar view.

Four PNGs are written to a temporary directory unless ``--out`` is given.
Plotting is CPU-side. Runtime about 15 s.

    pixi run -e test -- python examples/allsky_figures.py [--out DIR] [--nside 128]
"""

from __future__ import annotations

import argparse
import math
import tempfile
from pathlib import Path

import matplotlib
import torch

# A gallery script only ever saves files; do not require a display.
matplotlib.use("Agg")

from torchsky.sphere import MOC, healpix, lonlat_to_unit_xyz, nest2uniq
from torchsky.viz import plot_healpix_sky


def synthetic_sky(
    nside: int, *, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(values, lon_deg, lat_deg)`` for a smooth synthetic sky."""
    npix = healpix.nside2npix(nside)
    pix = torch.arange(npix, dtype=torch.int64)
    lon, lat = healpix.pix2ang(nside, pix, nest=False, lonlat=True)

    band = 0.55 * torch.exp(-((lat / 12.0) ** 2))

    # Large-scale structure: a sum of wide Gaussian clouds on the sphere.
    g = torch.Generator().manual_seed(seed)
    clouds = torch.zeros_like(lon)
    for _ in range(24):
        centre_lon = torch.rand(1, generator=g).item() * 360.0
        centre_lat = (torch.rand(1, generator=g).item() - 0.5) * 140.0
        sigma = 8.0 + torch.rand(1, generator=g).item() * 22.0
        amp = 0.05 + torch.rand(1, generator=g).item() * 0.32
        dlon = torch.remainder(lon - centre_lon + 180.0, 360.0) - 180.0
        clouds = clouds + amp * torch.exp(
            -((dlon / sigma) ** 2) - ((lat - centre_lat) / sigma) ** 2
        )

    bright = 1.10 * torch.exp(
        -(((lon - 265.0 + 180.0) % 360.0 - 180.0) ** 2) / 6.0
        - ((lat - 12.0) / 5.0) ** 2
    )
    values = band + clouds + bright + 0.02
    return values, lon, lat


def footprint(
    nside: int, *, lon: float = 80.0, lat: float = -5.0, radius: float = 60.0
):
    """A circular survey footprint as a MOC, plus its nested pixels."""
    pixels = healpix.query_disc(
        nside, lonlat_to_unit_xyz(lon, lat), math.radians(radius), nest=True
    )
    # MOC cells are addressed by uniq, which is built from *nested* pixel indices.
    return MOC(nest2uniq(nside, pixels)), pixels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory (default: a fresh temporary directory)",
    )
    parser.add_argument("--nside", type=int, default=128, help="HEALPix nside")
    args = parser.parse_args()

    outdir = Path(args.out or tempfile.mkdtemp(prefix="torchsky-allsky-"))
    outdir.mkdir(parents=True, exist_ok=True)
    values, lon, lat = synthetic_sky(args.nside)
    moc, pixels = footprint(args.nside)

    plot_healpix_sky(
        values,
        nside=args.nside,
        title="Whole-sky surface brightness",
        subtitle=(
            f"Synthetic band, clouds and one compact source \u00b7 nside {args.nside}"
        ),
        unit="counts s$^{-1}$ deg$^{-2}$",
        projection="mollweide",
        theme="dark",
        cmap="viridis",
        save=outdir / "allsky_mollweide_dark.png",
    )

    g = torch.Generator().manual_seed(3)
    picked = pixels[torch.randperm(pixels.numel(), generator=g)[:150]]
    # ``pixels`` are nested, so ask for their coordinates directly rather than
    # indexing the ring-ordered arrays built above.
    picked_lon, picked_lat = healpix.pix2ang(args.nside, picked, nest=True, lonlat=True)

    plot_healpix_sky(
        values,
        nside=args.nside,
        title="Survey footprint on the sky",
        subtitle="MOC outline with 150 catalog sources \u00b7 Hammer projection",
        unit="counts s$^{-1}$ deg$^{-2}$",
        projection="hammer",
        theme="paper",
        cmap="magma",
        moc=moc,
        moc_label="Spectral coverage",
        points=(picked_lon, picked_lat),
        point_label="Catalog sources",
        point_color="#c64600",
        save=outdir / "allsky_hammer_paper.png",
    )

    plot_healpix_sky(
        values,
        nside=args.nside,
        title="North-polar view",
        subtitle="Azimuthal view \u00b7 right ascension around the rim",
        unit="counts s$^{-1}$ deg$^{-2}$",
        projection="polar",
        theme="dark",
        cmap="cividis",
        save=outdir / "allsky_polar_dark.png",
    )

    plot_healpix_sky(
        values,
        nside=args.nside,
        title="High dynamic range",
        subtitle="Aitoff projection \u00b7 log normalization \u00b7 decadal colour bar",
        unit="counts s$^{-1}$ deg$^{-2}$",
        projection="aitoff",
        theme="dark",
        cmap="turbo",
        stretch="log",
        moc=moc,
        moc_label="Spectral coverage",
        save=outdir / "allsky_aitoff_log.png",
    )

    print(f"wrote {len(list(outdir.glob('*.png')))} figures to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
