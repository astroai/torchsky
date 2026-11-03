"""Coadd Legacy Surveys g/r/i/z cutouts of the Cartwheel onto sparse HEALPix maps.

Science question: can multi-band images on different pixel grids go straight
onto one HEALPix pixelization, one sparse map per band, without an
intermediate flat mosaic, and keep the same photometry as a conventional
image-plane coadd? Two Legacy Surveys DR10 cutouts of the Cartwheel galaxy
are used: g/r/i/z at 0.524 arcsec/px (128 x 128 px) and g/z on a second grid
at 0.35 arcsec/px (160 x 160 px, centre offset by a few arcsec), with their
inverse-variance maps. Their values are fluxes per pixel (nanomaggies), so
each image is first divided by its exact pixel solid angle to give surface
brightness, the quantity that resampling preserves. ``coadd_healpix`` then
builds one nested sparse map per band at NSIDE = 2^19 (0.40 arcsec cells),
combining the two grids in g and z with inverse-variance weights.

Each band is verified against ``coadd`` onto a planned TAN grid at the same
0.40 arcsec scale: the total flux in a 25 arcsec radius disc on the galaxy
(surface brightness times the exact area of every HEALPix cell or image
pixel), and the image-plane coadd sampled at the HEALPix cell centres, pixel
by pixel. Runtime about 10 s on CPU once the two 0.5 MB cutouts are cached.

Data: Legacy Surveys DR10 (Dey et al. 2019, AJ 157, 168), public; acknowledge
the Legacy Surveys, DECam/DES and NOIRLab.

    pixi run -e test -- python examples/legacy_healpix_multiband.py [--device cuda] [--out DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from _sample_data import fetch_sample, skip_if_unavailable

import torchfits
import torchsky.sphere as sphere
from torchsky.map import Observation, coadd, coadd_healpix, plan
from torchsky.wcs import WCS

NSIDE = 2**19
GRIDS = (("legacy_cartwheel_griz", "griz"), ("legacy_cartwheel_gz_fine", "gz"))
CENTRE = (9.4213, -33.7163)  # deg, the Cartwheel
DISC = 25.0  # arcsec
WCS_KEYS = (
    "CTYPE1",
    "CTYPE2",
    "CRVAL1",
    "CRVAL2",
    "CRPIX1",
    "CRPIX2",
    "CD1_1",
    "CD1_2",
    "CD2_1",
    "CD2_2",
)


def load() -> dict[str, list[Observation]]:
    """Per band, one surface-brightness observation per grid (nmgy / arcsec^2)."""
    with skip_if_unavailable():
        paths = {name: fetch_sample(name) for name, _ in GRIDS}
    per_band: dict[str, list[Observation]] = {}
    for name, bands in GRIDS:
        path = str(paths[name])
        images = torchfits.read(path, hdu=0).to(
            torch.float64
        )  # (band, y, x), nmgy per pixel
        invvar = torchfits.read(path, hdu=1).to(torch.float64)
        card = dict(torchfits.read_header(path, 0))
        header = {key: card[key] for key in WCS_KEYS}
        header.update(NAXIS=2, NAXIS1=images.shape[2], NAXIS2=images.shape[1])
        area = WCS(header).pixel_area_map(images.shape[1:], unit="arcsec2")
        print(
            f"{name}: bands {bands}, {images.shape[2]} x {images.shape[1]} px at {float(area.mean()) ** 0.5:.3f} arcsec"
        )
        for index, band in enumerate(bands):
            good = invvar[index] > 0
            per_band.setdefault(band, []).append(
                Observation(
                    data=(images[index] / area).float(),
                    wcs=header,
                    mask=good,
                    # sigma of surface brightness = sigma of flux / area
                    weight_map=(invvar[index] * area**2).float(),
                    weight_is_inverse_variance=True,
                )
            )
    return per_band


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    dev = args.device

    per_band = load()
    cell = sphere.nside2resol(NSIDE, arcmin=True) * 60.0
    cell_area = 4.0 * torch.pi * (180.0 * 3600.0 / torch.pi) ** 2 / (12 * NSIDE**2)
    print(
        f"NSIDE = 2^19: {cell:.3f} arcsec cells, {12 * NSIDE**2:.2e} on the full sphere"
    )

    maps = {}
    for band, observations in per_band.items():
        res = coadd_healpix(
            observations, NSIDE, nest=True, sparse=True, kernel="lanczos3", device=dev
        )
        pix = res.data.valid_pixels.cpu()
        val = res.data.get_values(pix.to(res.data.pixels.device)).cpu().double()
        wht = res.weight.get_values(pix.to(res.weight.pixels.device)).cpu().double()
        keep = wht > 0
        pix, val, wht = pix[keep], val[keep], wht[keep]
        lon, lat = sphere.pix2ang(NSIDE, pix, nest=True, lonlat=True)

        # Image-plane reference on a TAN grid at the HEALPix cell scale.
        target = plan(
            observations, projection="TAN", pixel_scale_arcsec=cell, device=dev
        )
        flat = coadd(
            observations,
            target,
            kernel="lanczos3",
            semantics="surface_brightness",
            device=dev,
        )
        fdata, fw = flat.data.cpu().double(), flat.weight.cpu()
        wcs = WCS(target.header)
        farea = wcs.pixel_area_map(fdata.shape, unit="arcsec2")

        # (1) Total flux in a disc on the galaxy.
        cosd = torch.cos(torch.deg2rad(torch.tensor(CENTRE[1], dtype=torch.float64)))
        r_hp = 3600.0 * torch.hypot((lon - CENTRE[0]) * cosd, lat - CENTRE[1])
        f_hp = float((val[r_hp < DISC]).sum() * cell_area)
        h, w = fdata.shape
        yy, xx = torch.meshgrid(
            torch.arange(h, dtype=torch.float64),
            torch.arange(w, dtype=torch.float64),
            indexing="ij",
        )
        world = wcs.pixel_to_world(torch.stack([xx.ravel(), yy.ravel()], -1), origin=0)
        r_fl = (
            3600.0
            * torch.hypot((world[:, 0] - CENTRE[0]) * cosd, world[:, 1] - CENTRE[1])
        ).reshape(h, w)
        inside = (r_fl < DISC) & (fw > 0)
        f_fl = float((fdata * farea)[inside].sum())

        # (2) The image-plane coadd sampled at the HEALPix cell centres.
        px = wcs.world_to_pixel(torch.stack([lon, lat], -1), origin=0)
        grid = torch.stack([2 * px[:, 0] / (w - 1) - 1, 2 * px[:, 1] / (h - 1) - 1], -1)
        sampled = F.grid_sample(
            fdata[None, None], grid[None, None], align_corners=True
        )[0, 0, 0]
        interior = (
            (px[:, 0] > 8) & (px[:, 0] < w - 9) & (px[:, 1] > 8) & (px[:, 1] < h - 9)
        )
        sigma = wht.clamp_min(1e-30) ** -0.5
        z = ((val - sampled) / sigma)[interior]
        mag = 22.5 - 2.5 * torch.log10(torch.tensor(f_hp))
        print(
            f"band {band}: {len(observations)} grid(s), {pix.numel()} cells; {DISC:.0f} arcsec disc "
            f"{f_hp:7.2f} nmgy (AB {float(mag):.2f}) vs image plane {f_fl:7.2f} nmgy "
            f"(ratio {f_hp / f_fl:.4f}); cell-by-cell difference {float(z.median()):+.3f} sigma median, "
            f"{1.4826 * float((z - z.median()).abs().median()):.3f} sigma robust rms"
        )
        assert abs(f_hp / f_fl - 1.0) < 0.01
        maps[band] = (pix, val)

    total = sum(int(m[0].numel()) for m in maps.values())
    dense_bytes = 12 * NSIDE**2 * 4
    print(
        f"sparse storage: {total} cells in 4 maps; a dense float32 map at this NSIDE would be {dense_bytes / 1e12:.1f} TB"
    )

    out = args.out or Path(tempfile.mkdtemp(prefix="torchsky-legacy-healpix-"))
    out.mkdir(parents=True, exist_ok=True)
    # Display: look up the cell under each pixel of a north-up TAN view (no resampling).
    n, scale = 150, 0.4 / 3600.0
    view = WCS(
        {
            "NAXIS": 2,
            "NAXIS1": n,
            "NAXIS2": n,
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
            "CRPIX1": (n + 1) / 2,
            "CRPIX2": (n + 1) / 2,
            "CRVAL1": CENTRE[0],
            "CRVAL2": CENTRE[1],
            "CDELT1": -scale,
            "CDELT2": scale,
        }
    )
    ax1 = torch.arange(n, dtype=torch.float64)
    yy, xx = torch.meshgrid(ax1, ax1, indexing="ij")
    world = view.pixel_to_world(torch.stack([xx.ravel(), yy.ravel()], -1), origin=0)
    cells = sphere.ang2pix(NSIDE, world[:, 0], world[:, 1], nest=True, lonlat=True)
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.5), constrained_layout=True)
    extent = [n * scale * 1800, -n * scale * 1800, -n * scale * 1800, n * scale * 1800]
    for ax, (band, (pix, val)) in zip(axes, maps.items()):
        order = pix.argsort()
        idx = torch.searchsorted(pix[order], cells).clamp_max(pix.numel() - 1)
        hit = pix[order][idx] == cells
        image = torch.where(hit, val[order][idx], torch.nan).reshape(n, n)
        lo, hi = torch.nanquantile(
            image, torch.tensor([0.05, 0.998], dtype=image.dtype)
        ).tolist()
        ax.imshow(
            torch.asinh((image - lo) / (0.03 * (hi - lo))),
            origin="lower",
            cmap="magma",
            extent=extent,
        )
        ax.set_title(f"{band}: {pix.numel()} cells, NSIDE 2^19", fontsize=9)
        ax.set_xlabel("east offset [arcsec]", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("north offset [arcsec]", fontsize=8)
    fig.savefig(out / "legacy_healpix_multiband.png", dpi=100)
    print(f"figure: {out / 'legacy_healpix_multiband.png'}; device={dev}")


if __name__ == "__main__":
    main()
