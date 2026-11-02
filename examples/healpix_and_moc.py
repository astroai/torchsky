"""HEALPix pixelization, region queries, sparse maps and MOC set algebra.

Round-trips coordinates through NESTED pixels, runs disc and polygon queries,
bins a synthetic catalogue into a dense map and a sparse map (checked equal),
degrades the hit counts with sum-preserving semantics, and builds MOC
footprints (cone, polygon) with union/intersection/difference, containment
and uniform random sampling. Runtime about 3 s on CPU.

    pixi run -e test -- python examples/healpix_and_moc.py [--device cuda] [--out DIR]
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
from torchsky.sphere import MOC, SparseHealpixMap, lonlat_to_unit_xyz


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    dev = args.device
    out = args.out or Path(tempfile.mkdtemp(prefix="torchsky-healpix-"))
    out.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(7)

    # 1. Pixelization round trip.
    nside = 64
    lon = torch.rand(100_000, generator=g, dtype=torch.float64) * 360.0
    lat = torch.rad2deg(
        torch.asin(2 * torch.rand(100_000, generator=g, dtype=torch.float64) - 1)
    )
    lon, lat = lon.to(dev), lat.to(dev)
    pix = sphere.ang2pix(nside, lon, lat, nest=True, lonlat=True)
    lon_c, lat_c = sphere.pix2ang(nside, pix, nest=True, lonlat=True)
    sep = sphere.great_circle_distance(lon, lat, lon_c, lat_c)
    max_sep = float(sep.max())
    print(
        f"nside={nside} npix={sphere.nside2npix(nside)}: max pixel-centre offset "
        f"{max_sep:.3f} deg (resolution {sphere.nside2resol(nside, arcmin=True) / 60:.3f} deg)"
    )

    # 2. Region queries.
    centre = lonlat_to_unit_xyz(
        torch.tensor(30.0, dtype=torch.float64), torch.tensor(20.0, dtype=torch.float64)
    )
    disc = sphere.query_disc(
        32, centre, torch.deg2rad(torch.tensor(15.0)).item(), nest=True
    )
    poly_lon = [20.0, 50.0, 50.0, 20.0]
    poly_lat = [10.0, 10.0, 40.0, 40.0]
    poly = sphere.query_polygon(32, poly_lon, poly_lat, nest=True, lonlat=True)
    both = disc[torch.isin(disc, poly)]
    print(
        f"disc: {disc.numel()} px, polygon: {poly.numel()} px, overlap: {both.numel()} px"
    )

    # 3. Catalogue -> dense and sparse maps, then sum-preserving degrade.
    ns = 16
    cat_lon = torch.cat(
        [torch.rand(5000, generator=g) * 20 + 25, torch.rand(2000, generator=g) * 360]
    ).double()
    cat_lat = torch.cat(
        [
            torch.rand(5000, generator=g) * 20 + 10,
            torch.rand(2000, generator=g) * 100 - 50,
        ]
    ).double()
    hits = sphere.ang2pix(ns, cat_lon, cat_lat, nest=True, lonlat=True)
    dense = torch.bincount(hits, minlength=sphere.nside2npix(ns)).double()
    occ = torch.nonzero(dense).flatten()
    sparse = SparseHealpixMap.from_pixels(
        occ, dense[occ], nside_sparse=ns, nside_coverage=4, nest=True, sentinel=0.0
    )
    torch.testing.assert_close(sparse.to_dense(), dense)
    coarse = sphere.ud_grade(
        dense, nside_out=8, order_in="NESTED", order_out="NESTED", power=-2.0
    )
    assert float(coarse.sum()) == float(dense.sum())
    print(
        f"catalogue {cat_lon.numel()} rows -> {occ.numel()}/{dense.numel()} occupied pixels "
        f"(sparse == dense); degraded sum preserved: {float(coarse.sum()):.0f}"
    )

    # 4. MOC footprints and set algebra.
    cone = MOC.from_circle(30.0, 20.0, 15.0, max_order=7)
    box = MOC.from_polygon(poly_lon, poly_lat, max_order=7)
    union, inter, diff = cone.union(box), cone.intersection(box), cone.difference(box)
    a_u, a_i, a_d = union.area_sq_deg, inter.area_sq_deg, diff.area_sq_deg
    assert abs(a_u - (cone.area_sq_deg + box.area_sq_deg - a_i)) < 1e-6 * a_u
    assert abs(a_d - (cone.area_sq_deg - a_i)) < 1e-6 * a_u
    rl, rb = inter.sample_randoms(2000, generator=torch.Generator().manual_seed(1))
    assert bool(inter.contains(rl, rb).all())
    print(
        f"MOC areas [deg^2] cone={cone.area_sq_deg:.1f} box={box.area_sq_deg:.1f} "
        f"union={a_u:.1f} intersection={a_i:.1f} difference={a_d:.1f}; "
        f"2000 randoms drawn inside the intersection"
    )

    fig, ax = plt.subplots(figsize=(5.2, 3.6), constrained_layout=True)
    for px, color, label in [
        (disc, "#4c78a8", "disc"),
        (poly, "#f58518", "polygon"),
        (both, "#54a24b", "overlap"),
    ]:
        lo, la = sphere.pix2ang(32, px, nest=True, lonlat=True)
        ax.scatter(lo.cpu(), la.cpu(), s=6, c=color, label=label)
    ax.scatter(rl.cpu(), rb.cpu(), s=1, c="k", alpha=0.4, label="MOC randoms")
    ax.set(
        xlabel="lon [deg]", ylabel="lat [deg]", title="HEALPix queries and MOC sampling"
    )
    ax.legend(fontsize=7, markerscale=2)
    fig.savefig(out / "healpix_and_moc.png", dpi=100)
    print(f"device={dev}  figure: {out / 'healpix_and_moc.png'}")


if __name__ == "__main__":
    main()
