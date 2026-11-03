"""Chandra ACIS events of Sgr A* as a sparse HEALPix counts map and a MOC.

Science question: what does the X-ray sky around the Galactic centre look like
when every photon is placed on the sphere instead of in a detector image? The
ACIS-I event list of Chandra observation 5950 (Sgr A*, 48.5 ks, 484k events) is
read with torchfits. Sky pixel coordinates are converted to ICRS with the
event table's own TAN WCS (taken from its ``TC*`` keywords) using
``torchsky.wcs.WCS``. Events in 0.5-7 keV are binned into a NESTED
``SparseHealpixMap`` at nside 2**14 (12.9 arcsec cells; only about 8000 of the
3.2e9 sphere cells are stored), the exposure footprint becomes a MOC, and the
map is smoothed with an exact-conserving 3 x 3 neighbour kernel applied on the
sparse pixel set (``neighbors``), equivalent to a Gaussian of sigma = 0.82
sqrt(n) cells. The example checks that binning conserves the event count, that
the MOC covers every event, that smoothing conserves counts, and compares the
brightest cell with the Sgr A* position.

Data: Chandra X-ray Center ObsID 5950 as distributed in the astropy tutorial
data (public Chandra archive data; credit NASA/CXC). Runtime about 6 s on CPU
once the 31 MB file is cached.

    pixi run -e test -- python examples/chandra_events_on_sphere.py [--device cuda] [--out DIR]
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
from torchsky.sphere import MOC, SparseHealpixMap, ang2pix, neighbors, pix2ang
from torchsky.wcs import WCS

NSIDE = 2**14
MOC_ORDER = 12
E_BAND_EV = (500.0, 7000.0)
SMOOTH_STEPS = 6
SGR_A_STAR = (266.41683, -29.00781)  # ICRS, deg (Reid & Brunthaler 2004)


def event_wcs(header: dict) -> dict:
    """Celestial WCS of the ``x``/``y`` sky columns (columns 11 and 12, 1-based pixels)."""
    return {
        "NAXIS": 2,
        "CTYPE1": header["TCTYP11"],
        "CTYPE2": header["TCTYP12"],
        "CRPIX1": header["TCRPX11"],
        "CRPIX2": header["TCRPX12"],
        "CRVAL1": header["TCRVL11"],
        "CRVAL2": header["TCRVL12"],
        "CDELT1": header["TCDLT11"],
        "CDELT2": header["TCDLT12"],
        "CUNIT1": "deg",
        "CUNIT2": "deg",
    }


def smooth_sparse(pixels: torch.Tensor, counts: torch.Tensor, nside: int, steps: int):
    """3 x 3 box smoothing on a sparse NESTED pixel set, grown by one ring per step."""
    for _ in range(steps):
        nb = neighbors(nside, pixels, nest=True)
        pixels_new = torch.unique(torch.cat([pixels, nb[nb >= 0]]))
        pos = torch.searchsorted(pixels_new, pixels)
        values = torch.zeros(
            pixels_new.numel(), dtype=counts.dtype, device=counts.device
        )
        # each cell spreads value/9 to itself and to its (up to 8) neighbours
        nb_count = (nb >= 0).sum(dim=1).to(counts.dtype)
        share = counts / (1.0 + nb_count)
        values.index_add_(0, pos, share)
        nb_pos = torch.searchsorted(pixels_new, nb.clamp(min=0))
        values.index_add_(0, nb_pos[nb >= 0], share.unsqueeze(1).expand_as(nb)[nb >= 0])
        pixels, counts = pixels_new, values
    return pixels, counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    dev = args.device

    with skip_if_unavailable():
        path = fetch_sample("chandra_events")
    header = dict(torchfits.read_header(str(path), 1))
    events = torchfits.read(str(path), hdu=1)
    energy = events["energy"].to(dev)
    xy = torch.stack([events["x"], events["y"]], dim=-1).to(torch.float64).to(dev)
    wcs = WCS(event_wcs(header)).to(dev)
    ra, dec = wcs.pixel_to_world(xy[:, 0], xy[:, 1], origin=1)
    keep = (energy >= E_BAND_EV[0]) & (energy <= E_BAND_EV[1])
    ra, dec = ra[keep], dec[keep]
    print(
        f"ObsID {header['OBS_ID']} ({header['OBJECT']}, {header['EXPOSURE'] / 1e3:.1f} ks): "
        f"{energy.numel()} events, {ra.numel()} in {E_BAND_EV[0] / 1e3:.1f}-{E_BAND_EV[1] / 1e3:.0f} keV; "
        f"field centre {float(ra.mean()):.3f}, {float(dec.mean()):.3f} deg"
    )

    # Events -> sparse NESTED counts map.
    pix = ang2pix(NSIDE, ra, dec, nest=True, lonlat=True)
    cells, counts = torch.unique(pix, return_counts=True)
    counts = counts.to(torch.float64)
    cmap = SparseHealpixMap.from_pixels(cells.cpu(), counts.cpu(), NSIDE, nest=True)
    assert int(counts.sum()) == ra.numel()
    assert bool((cmap.get_values(cells.cpu()) == counts.cpu()).all())
    print(
        f"sparse map nside {NSIDE} ({3600 * 58.6 / NSIDE:.1f} arcsec cells): {cells.numel()} occupied cells "
        f"of {12 * NSIDE**2:.2e}, {counts.sum():.0f} events conserved, peak {int(counts.max())} counts/cell"
    )

    # Exposure footprint as a MOC and a containment check.
    moc = MOC.from_lonlat(ra.cpu(), dec.cpu(), max_order=MOC_ORDER)
    inside = moc.contains(ra.cpu(), dec.cpu())
    print(
        f"MOC order {MOC_ORDER}: {moc.uniq.numel()} cells, {moc.area_sq_deg * 3600:.0f} arcmin^2 "
        f"(five chips, edges inflated by the 51 arcsec cells); contains {int(inside.sum())}/{ra.numel()} events"
    )
    assert bool(inside.all())

    # Smoothing on the sparse pixel set.
    sm_cells, sm_counts = smooth_sparse(cells, counts, NSIDE, SMOOTH_STEPS)
    sigma_arcsec = (
        math.sqrt(2 * SMOOTH_STEPS / 3)
        * 3600
        * math.degrees(math.sqrt(math.pi / 3))
        / NSIDE
    )
    print(
        f"smoothed {SMOOTH_STEPS} steps (sigma ~ {sigma_arcsec:.0f} arcsec): {sm_cells.numel()} cells, "
        f"counts {float(sm_counts.sum()):.1f} (conserved: {abs(float(sm_counts.sum()) - float(counts.sum())) < 1e-6 * float(counts.sum())})"
    )
    assert abs(float(sm_counts.sum()) - float(counts.sum())) < 1e-6 * float(
        counts.sum()
    )
    top = sm_cells[int(sm_counts.argmax())]
    t_lon, t_lat = pix2ang(NSIDE, top.cpu(), nest=True, lonlat=True)
    sep = (
        math.hypot(
            (float(t_lon) - SGR_A_STAR[0]) * math.cos(math.radians(SGR_A_STAR[1])),
            float(t_lat) - SGR_A_STAR[1],
        )
        * 60
    )
    sgr = ang2pix(
        NSIDE,
        torch.tensor(SGR_A_STAR[0]),
        torch.tensor(SGR_A_STAR[1]),
        nest=True,
        lonlat=True,
    )
    at_sgr = float(sm_counts[sm_cells.cpu() == sgr.cpu()].sum())
    print(
        f"brightest smoothed cell ({float(sm_counts.max()):.0f} counts) lies {sep:.1f} arcmin from Sgr A*, "
        f"a point source; the cell at Sgr A* holds {at_sgr:.0f} counts"
    )

    # Figure: raw counts and smoothed map in tangent offsets (arcmin).
    out = args.out or Path(tempfile.mkdtemp(prefix="torchsky-chandra-"))
    out.mkdir(parents=True, exist_ok=True)
    ra0, dec0 = float(ra.mean()), float(dec.mean())
    fig, axes = plt.subplots(
        1, 2, figsize=(9.5, 4.5), constrained_layout=True, sharex=True, sharey=True
    )
    for ax, (c, v), title in zip(
        axes,
        ((cells, counts), (sm_cells, sm_counts)),
        ('counts per 12.9" cell', "smoothed"),
    ):
        lon, lat = pix2ang(NSIDE, c.cpu(), nest=True, lonlat=True)
        x = (lon - ra0) * math.cos(math.radians(dec0)) * 60
        y = (lat - dec0) * 60
        order = torch.argsort(v.cpu())
        sc = ax.scatter(
            x[order],
            y[order],
            c=torch.log10(v.cpu()[order].clamp(min=1e-2)),
            s=1.2,
            marker="s",
            cmap="inferno",
            rasterized=True,
        )
        ax.set(
            title=f"0.5-7 keV, {title}",
            xlabel="east offset [arcmin]",
            facecolor="black",
        )
        ax.set_aspect("equal")
        fig.colorbar(sc, ax=ax, shrink=0.8, label="log10 counts")
    axes[0].invert_xaxis()
    axes[0].set_ylabel("north offset [arcmin]")
    fig.savefig(out / "chandra_events_on_sphere.png", dpi=100)
    print(f"figure: {out / 'chandra_events_on_sphere.png'}; device={dev}")


if __name__ == "__main__":
    main()
