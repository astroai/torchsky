"""Power spectrum of the Planck Commander synchrotron map on a masked sky.

Science question: how steep is the angular power spectrum of Galactic
synchrotron emission, and does the usual masked-sky pseudo-C_ell with an f_sky
correction recover the full-sky spectrum? The Planck 2015 (PR2) Commander
synchrotron component (nside 256, 408 MHz, 60 arcmin FWHM, Galactic
coordinates, NESTED) is read from the sample cache. A Galactic cut |b| > 20
degrees is built with ``query_strip`` and turned into a MOC, whose area gives
f_sky exactly. ``anafast`` measures the full-sky and masked spectra. For a
statistically isotropic Gaussian simulation with the fitted power-law spectrum
the f_sky-corrected masked spectrum agrees with the input to a few percent;
for the real map it falls 10-40 times lower, because the synchrotron power is
concentrated in the Galactic plane: the f_sky correction presumes a
homogeneous field, so the high-latitude sky is far smoother than the all-sky
average suggests.

Data: Planck Legacy Archive (ESA), COM_CompMap_Synchrotron-commander_0256_R2.00;
cite Planck Collaboration 2015, A&A 594, A10. Runtime about 5 s on CPU once the
map (9 MB) is cached.

    pixi run -e test -- python examples/planck_power_spectrum.py [--device cuda] [--out DIR]
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
from torchsky.sphere import MOC, nest2uniq
from torchsky.viz import healpix_sky_grid

B_CUT_DEG = 20.0
LMAX = 383
BANDS = ((10, 30), (30, 100), (100, 250))


def band_ratio(a: torch.Tensor, b: torch.Tensor) -> list[float]:
    return [float(a[lo:hi].mean() / b[lo:hi].mean()) for lo, hi in BANDS]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    dev = args.device
    torch.manual_seed(0)

    with skip_if_unavailable():
        path = fetch_sample("planck_commander_sync_256")
    table = torchfits.read(str(path), hdu=1)
    header = dict(torchfits.read_header(str(path), 1))
    nside = int(header["NSIDE"])
    assert header["ORDERING"] == "NESTED" and header["COORDSYS"] == "GALACTIC"
    sky = table["I_MEAN"].to(torch.float64).to(dev)
    print(
        f"{header['EXTNAME']}: nside {nside}, {sky.numel()} pixels, "
        f"unit {header['TUNIT2']}, median {float(sky.median()):.3g}"
    )

    # Galactic cut: keep colatitude < 90-b_cut and > 90+b_cut (pixels are NESTED).
    cut = math.radians(B_CUT_DEG)
    keep = torch.cat(
        [
            sphere.query_strip(nside, 0.0, math.pi / 2 - cut, nest=True),
            sphere.query_strip(nside, math.pi / 2 + cut, math.pi, nest=True),
        ]
    ).to(dev)
    mask = torch.zeros_like(sky)
    mask[keep] = 1.0
    footprint = MOC(nest2uniq(nside, keep.cpu()))
    fsky_pixels = float(mask.mean())
    fsky_moc = footprint.area_sq_deg / 41252.96124941928
    print(
        f"mask |b| > {B_CUT_DEG:.0f} deg: f_sky = {fsky_pixels:.4f} (pixels), "
        f"{fsky_moc:.4f} (MOC area), analytic {1 - math.sin(cut):.4f}"
    )
    assert abs(fsky_pixels - fsky_moc) < 1e-9

    cl_full = sphere.anafast(sky, nest=True, lmax=LMAX).to(torch.float64)
    cl_cut = sphere.anafast(sky * mask, nest=True, lmax=LMAX).to(torch.float64)
    cl_corr = cl_cut / fsky_pixels
    ell = torch.arange(LMAX + 1, dtype=torch.float64, device=cl_full.device)

    # Power-law fit of the full-sky spectrum (ell 10..150) and a masked simulation.
    sel = slice(10, 150)
    design = torch.stack([torch.ones_like(ell[sel]), ell[sel].log()], dim=1)
    coef = torch.linalg.lstsq(design, cl_full[sel].log().unsqueeze(1)).solution.squeeze()
    slope = float(coef[1])
    cl_model = torch.zeros_like(ell)
    cl_model[1:] = (coef[0] + coef[1] * ell[1:].log()).exp()
    sim = sphere.reorder(
        sphere.synfast(cl_model, nside=nside, lmax=LMAX).to(dev), r2n=True
    )
    cl_sim = sphere.anafast(sim * mask, nest=True, lmax=LMAX).to(torch.float64) / fsky_pixels
    cl_sim_full = sphere.anafast(sim, nest=True, lmax=LMAX).to(torch.float64)
    print(f"full-sky spectrum: C_ell ~ ell^{slope:.2f} over ell 10-150")
    print("band          masked/fsky vs full (data)   masked/fsky vs full (simulation)")
    for (lo, hi), r_data, r_sim in zip(
        BANDS, band_ratio(cl_corr, cl_full), band_ratio(cl_sim, cl_sim_full)
    ):
        print(f"ell {lo:3d}-{hi:3d}   {r_data:8.3f}                     {r_sim:8.3f}")

    out = args.out or Path(tempfile.mkdtemp(prefix="torchsky-planck-"))
    out.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(9, 3.8), constrained_layout=True)
    ax = fig.add_subplot(1, 2, 1, projection="mollweide")
    grid = healpix_sky_grid(sky.cpu(), nside=nside, nest=True)
    inside = healpix_sky_grid(mask.cpu(), nside=nside, nest=True).data > 0.5
    lon = torch.deg2rad(0.5 * (grid.lon_edges[1:] + grid.lon_edges[:-1])).flip(0)
    lat = torch.deg2rad(0.5 * (grid.lat_edges[1:] + grid.lat_edges[:-1]))
    shown = torch.where(inside, torch.log10(grid.data.clamp(min=1e-3)), torch.nan)
    im = ax.pcolormesh(-lon, lat, shown.flip(1), shading="auto", cmap="magma", rasterized=True)
    ax.set_xticklabels([])
    ax.set_title(f"Synchrotron, |b|>{B_CUT_DEG:.0f} deg (Galactic)", fontsize=9, pad=14)
    fig.colorbar(im, ax=ax, orientation="horizontal", shrink=0.7, label="log10 uK_RJ")
    ax = fig.add_subplot(1, 2, 2)
    x = ell[2:].cpu()
    for cl, style, label in (
        (cl_full, "k-", "full sky"),
        (cl_corr, "C3.", r"masked / $f_{sky}$"),
        (cl_sim, "C0.", r"simulation, masked / $f_{sky}$"),
    ):
        ax.loglog(x, cl[2:].cpu(), style, ms=3, lw=1.2, label=label)
    ax.set(xlabel=r"multipole $\ell$", ylabel=r"$C_\ell$ [$\mu K_{RJ}^2$]", title="Pseudo-$C_\\ell$")
    ax.legend(fontsize=8)
    fig.savefig(out / "planck_power_spectrum.png", dpi=110)
    print(f"figure: {out / 'planck_power_spectrum.png'}; device={dev}")


if __name__ == "__main__":
    main()
