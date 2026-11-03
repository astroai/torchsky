"""Footprint algebra on real survey coverage: SDSS, GALEX and eROSITA eFEDS MOCs.

Science question: how much sky do the SDSS DR9 imaging, the GALEX All-sky
Imaging Survey (NUV) and the eROSITA eFEDS field share, so that a joint
optical, ultraviolet and X-ray sample can be planned? The three published
coverage MOCs (IVOA MOC FITS files from the CDS HiPS of each survey, mirrored
by IRSA and ESASky; deepest cells at orders 12, 8 and 7) are read with
torchfits into ``torchsky.sphere.MOC`` and combined exactly: union,
intersection and difference are range operations on NESTED cells, and
``area_sq_deg`` counts cells, so inclusion-exclusion |A u B| = |A| + |B| - |A
n B| holds exactly. A circle of radius 1 degree checks the MOC area
against the exact cap area 2 pi (1 - cos r). Each footprint is then degraded
to order 8 (nside 256) and summed with ``sum_union`` into a
``SparseHealpixMap`` of survey depth (number of surveys covering each pixel);
the pixelized areas are compared with the exact ones, the difference being the
partially covered boundary cells that ``degrade`` keeps. The figure shows the
all-sky depth map and a zoom on the eFEDS field.

Data: coverage MOCs of the SDSS9 colour HiPS (cite Ahn et al. 2012) and the
GALEX GR6 AIS NUV HiPS (cite Bianchi et al. 2017) built by CDS and served by
IRSA (3.0 MB and 0.3 MB), and of the eROSITA eFEDS HiPS served by ESASky
(9 kB; eROSITA EDR, CC BY 4.0, cite Brunner et al. 2022); all pinned by sha256.
Runtime about 3 s on CPU once the samples are cached.

    pixi run -e test -- python examples/survey_footprints_moc.py [--out DIR]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from _sample_data import fetch_sample, skip_if_unavailable

import torchfits
import torchsky.sphere as sphere
from torchsky.sphere import MOC, sum_union
from torchsky.viz import healpix_sky_grid

SURVEYS = {
    "SDSS DR9": "moc_sdss_dr9",
    "GALEX AIS NUV": "moc_galex_ais_nuv",
    "eROSITA eFEDS": "moc_erosita_efeds",
}
DEPTH_ORDER = 8
SQ_DEG_SKY = 4.0 * math.pi * (180.0 / math.pi) ** 2


def read_moc(path: Path) -> MOC:
    """Read the UNIQ column of an IVOA MOC FITS file (first table extension).

    ``MOC.read_fits`` requires ``EXTNAME = 'MOC'``, which the IVOA standard does
    not mandate and these CDS-built files do not carry, so the table is read here.
    """
    header = dict(torchfits.read_header(str(path), 1))
    assert header["ORDERING"] == "NUNIQ" and header.get("COORDSYS", "C") == "C"
    table = torchfits.read(str(path), hdu=1)
    return MOC(table["UNIQ"].to(torch.int64))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    with skip_if_unavailable():
        paths = {name: fetch_sample(key) for name, key in SURVEYS.items()}
    mocs = {name: read_moc(path) for name, path in paths.items()}
    for name, moc in mocs.items():
        print(
            f"{name:14s} {moc.uniq.numel():7d} cells, max order {moc.max_order:2d}, "
            f"{moc.area_sq_deg:9.1f} deg^2 ({100 * moc.area_sq_deg / SQ_DEG_SKY:.1f}% of the sky)"
        )
    sdss, galex, efeds = mocs.values()

    both = sdss & galex
    either = sdss | galex
    identity = sdss.area_sq_deg + galex.area_sq_deg - both.area_sq_deg
    print(
        f"SDSS n GALEX {both.area_sq_deg:.1f} deg^2, SDSS u GALEX {either.area_sq_deg:.1f} deg^2, "
        f"SDSS - GALEX {(sdss - galex).area_sq_deg:.1f} deg^2; "
        f"inclusion-exclusion residual {abs(either.area_sq_deg - identity):.1e} deg^2"
    )
    assert abs(either.area_sq_deg - identity) < 1e-6 * either.area_sq_deg
    triple = both & efeds
    print(
        f"eFEDS {efeds.area_sq_deg:.1f} deg^2: {100 * (efeds & sdss).area_sq_deg / efeds.area_sq_deg:.1f}% in SDSS, "
        f"{100 * (efeds & galex).area_sq_deg / efeds.area_sq_deg:.1f}% in GALEX AIS, "
        f"{triple.area_sq_deg:.1f} deg^2 in all three"
    )
    assert triple.area_sq_deg <= min(m.area_sq_deg for m in mocs.values())
    assert efeds.contains(torch.tensor([136.0]), torch.tensor([1.5])).item()

    cap = MOC.from_circle(140.0, 2.0, 1.0, max_order=13)
    exact_cap = (
        2.0 * math.pi * (1.0 - math.cos(math.radians(1.0))) * (180.0 / math.pi) ** 2
    )
    print(
        f"1 deg cap at order 13: MOC {cap.area_sq_deg:.4f} deg^2 vs exact {exact_cap:.4f} deg^2 "
        f"({100 * (cap.area_sq_deg / exact_cap - 1):+.2f}%, boundary cells)"
    )
    assert 1.0 <= cap.area_sq_deg / exact_cap < 1.02

    # Survey depth as one sparse map at order 8: degrade (a cell counts when any
    # part of it is covered), list covered pixels, count surveys per pixel.
    nside = 1 << DEPTH_ORDER
    coverage = [m.degrade(DEPTH_ORDER).to_sparse_map(nside) for m in mocs.values()]
    depth_map = sum_union(coverage)
    pix_area = sphere.nside2pixarea(nside, degrees=True)
    for k in (1, 2, 3):
        n_k = int((depth_map.values >= k).sum())
        print(
            f"order {DEPTH_ORDER} depth >= {k}: {n_k:7d} pixels = {n_k * pix_area:9.1f} deg^2"
        )
    both_pix = int(torch.isin(coverage[1].pixels, coverage[0].pixels).sum())
    print(
        f"SDSS n GALEX at order {DEPTH_ORDER}: {both_pix * pix_area:.1f} deg^2 vs exact "
        f"{both.area_sq_deg:.1f} deg^2 (+{100 * (both_pix * pix_area / both.area_sq_deg - 1):.1f}% from boundary cells)"
    )
    assert both_pix * pix_area >= both.area_sq_deg

    out = args.out or Path(tempfile.mkdtemp(prefix="torchsky-footprints-"))
    out.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12, 5.0), constrained_layout=True)
    ax = fig.add_subplot(1, 2, 1, projection="mollweide")
    grid = healpix_sky_grid(depth_map.to_dense(0.0), nside=nside, nest=True)
    lon = torch.deg2rad(0.5 * (grid.lon_edges[1:] + grid.lon_edges[:-1])).flip(0)
    lat = torch.deg2rad(0.5 * (grid.lat_edges[1:] + grid.lat_edges[:-1]))
    shown = torch.where(grid.data > 0, grid.data, torch.nan).flip(1)
    cmap = plt.get_cmap("viridis", 3)
    im = ax.pcolormesh(
        -lon, lat, shown, shading="auto", cmap=cmap, vmin=0.5, vmax=3.5, rasterized=True
    )
    ax.set_xticklabels([])
    ax.grid(alpha=0.3)
    ax.set_title(
        "Survey depth (SDSS DR9, GALEX AIS NUV, eFEDS), ICRS", fontsize=9, pad=12
    )
    cbar = fig.colorbar(
        im, ax=ax, orientation="horizontal", shrink=0.6, ticks=[1, 2, 3]
    )
    cbar.set_label("number of surveys")

    # Zoom on eFEDS: membership of each footprint on a 0.05 degree grid.
    ax = fig.add_subplot(1, 2, 2)
    gra = torch.arange(124.0, 148.0, 0.05, dtype=torch.float64)
    gdec = torch.arange(-6.0, 9.0, 0.05, dtype=torch.float64)
    mra, mdec = torch.meshgrid(gra, gdec, indexing="xy")
    inside = {
        name: moc.contains(mra.flatten(), mdec.flatten()).reshape(mra.shape)
        for name, moc in mocs.items()
    }
    zoom_depth = sum(v.to(torch.float32) for v in inside.values())
    extent = (float(gra[0]), float(gra[-1]), float(gdec[0]), float(gdec[-1]))
    ax.imshow(
        torch.where(zoom_depth > 0, zoom_depth, torch.nan),
        origin="lower",
        extent=extent,
        cmap=cmap,
        vmin=0.5,
        vmax=3.5,
        interpolation="nearest",
    )
    for (name, mask), colour in zip(inside.items(), ("w", "C1", "C3")):
        ax.contour(
            mra,
            mdec,
            mask.to(torch.float32),
            levels=[0.5],
            colors=colour,
            linewidths=1.0,
        )
        ax.plot([], [], color=colour, label=name)
    ax.invert_xaxis()
    ax.set(
        xlabel="RA [deg]",
        ylabel="Dec [deg]",
        aspect="equal",
        title=f"eFEDS field: {triple.area_sq_deg:.0f} deg$^2$ in all three surveys",
    )
    ax.title.set_fontsize(9)
    ax.legend(fontsize=7, loc="lower left", facecolor="0.85")
    target = out / "survey_footprints_moc.png"
    fig.savefig(target, dpi=110)
    print(f"figure: {target}")


if __name__ == "__main__":
    main()
