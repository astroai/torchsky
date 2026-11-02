"""Coadd overlapping multiband tiles directly onto a HEALPix grid.

Twelve TAN tiles (three bands x four overlapping tiles) sample a smooth
synthetic sky whose colour varies across the field. ``coadd_healpix`` resamples
each band's tiles straight onto an NSIDE=1024 nested sparse map without an
intermediate flat mosaic. Each band is validated against the analytic sky at
the HEALPix pixel centres, and overlap counts show the tiling depth.
Runtime about 3 s on CPU.

    pixi run -e test -- python examples/coadd_to_healpix.py [--device cuda] [--out DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import torchsky.sphere as sphere
from torchsky.map import Observation, coadd_healpix
from torchsky.wcs import WCS

NSIDE, TILE, SCALE = 1024, 128, 0.02  # 0.02 deg/pix tiles vs ~0.057 deg HEALPix pixels
BANDS = {"g": 1.0, "r": 1.6, "i": 2.1}
CENTRE = (150.0, 2.0)


def sky_model(lon, lat, gain: float):
    """Smooth band-dependent sky: gradient plus two Gaussian clumps."""
    x = (lon - CENTRE[0]) * torch.cos(torch.deg2rad(lat))
    y = lat - CENTRE[1]
    clump = torch.exp(-((x - 0.6) ** 2 + (y + 0.4) ** 2) / 0.18) + 0.6 * torch.exp(
        -((x + 0.7) ** 2 + (y - 0.5) ** 2) / 0.1
    )
    return gain * (1.0 + 0.1 * x + 0.05 * y + clump)


def tile_header(dx: float, dy: float) -> dict:
    return {
        "NAXIS": 2,
        "NAXIS1": TILE,
        "NAXIS2": TILE,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": TILE / 2 + 0.5,
        "CRPIX2": TILE / 2 + 0.5,
        "CRVAL1": CENTRE[0] + dx,
        "CRVAL2": CENTRE[1] + dy,
        "CDELT1": -SCALE,
        "CDELT2": SCALE,
        "CUNIT1": "deg",
        "CUNIT2": "deg",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or Path(tempfile.mkdtemp(prefix="torchsky-coadd-healpix-"))
    out.mkdir(parents=True, exist_ok=True)

    offsets = [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)]
    ax1 = torch.arange(TILE, dtype=torch.float64)
    yy, xx = torch.meshgrid(ax1, ax1, indexing="ij")
    pix = torch.stack([xx.ravel(), yy.ravel()], dim=-1)

    maps, counts = {}, None
    for band, gain in BANDS.items():
        observations = []
        for dx, dy in offsets:
            h = tile_header(dx * 0.9, dy * 0.9)
            world = WCS(h).pixel_to_world(pix, origin=0)
            data = sky_model(world[:, 0], world[:, 1], gain).reshape(TILE, TILE).float()
            observations.append(
                Observation(data=data, wcs=h, weight_map=torch.ones_like(data))
            )
        res = coadd_healpix(
            observations,
            NSIDE,
            nest=True,
            sparse=True,
            kernel="bilinear",
            device=args.device,
        )
        dense = res.data.to_dense() if hasattr(res.data, "to_dense") else res.data
        w = res.weight.to_dense() if hasattr(res.weight, "to_dense") else res.weight
        covered = torch.nonzero(w.cpu() > 0).flatten()
        lon, lat = sphere.pix2ang(NSIDE, covered, nest=True, lonlat=True)
        truth = sky_model(lon, lat, gain)
        rel = (dense.cpu()[covered].double() - truth).abs() / truth
        inner = rel.sort().values[: int(0.95 * rel.numel())]
        print(
            f"band {band}: {covered.numel()} pixels covered, median |rel err| = "
            f"{float(rel.median()):.4f}, 95th pct = {float(inner[-1]):.4f}"
        )
        assert float(rel.median()) < 0.02
        maps[band] = (lon, lat, dense.cpu()[covered])
        counts = res.overlap_count
    depth = counts.to_dense() if hasattr(counts, "to_dense") else counts
    print(
        f"overlap depth up to {int(depth.max())} tiles per pixel; NSIDE={NSIDE}, "
        f"pixel size {sphere.nside2resol(NSIDE, arcmin=True):.2f} arcmin, device={args.device}"
    )

    fig, axes = plt.subplots(
        1, 3, figsize=(10, 3.3), constrained_layout=True, sharey=True
    )
    for ax, (band, (lon, lat, val)) in zip(axes, maps.items()):
        ax.scatter(lon, lat, c=val, s=1.2, cmap="viridis", marker="s")
        ax.set_title(f"band {band}", fontsize=9)
        ax.invert_xaxis()
        ax.set_xlabel("lon [deg]", fontsize=8)
    axes[0].set_ylabel("lat [deg]", fontsize=8)
    fig.savefig(out / "coadd_to_healpix.png", dpi=100)
    print(f"figure: {out / 'coadd_to_healpix.png'}")


if __name__ == "__main__":
    main()
