"""Power spectrum of the Planck Commander synchrotron map with an apodised Galactic mask.

Science question: what is the angular power spectrum of Galactic synchrotron
emission away from the Galactic plane, and how do you measure it correctly on a
cut sky? The Planck 2015 (PR2) Commander synchrotron component (nside 256,
408 MHz, 60 arcmin FWHM, Galactic coordinates, NESTED) is read from the sample
cache. The mask keeps |b| > 20 degrees with a 10 degree cosine taper, evaluated
exactly from each pixel's latitude; its MOC gives the observed area. Before
masking, the monopole and dipole are fitted on the unmasked sky and removed
with ``remove_monopole_dipole``: the map has a median of 1.6e7 uK, and through
a sharp cut that monopole leaks into every even multipole. The pseudo-C_ell
from ``anafast`` is divided by w2, the mean of the squared mask.

Four checks: (1) a Gaussian simulation drawn from the measured spectrum and
masked the same way returns its input to a few percent above ell 30; below
that, mode coupling from the steep ell < 10 power biases the w2-scaled estimate
about 20 percent high, which only a coupling-matrix (MASTER) inversion removes;
(2) the |b| > 20 and |b| > 30 spectra agree within 25 percent above ell 30;
(3) the even/odd multipole alternation of the sharp, unapodised cut is gone;
(4) ``anafast`` agrees with healpy to 1e-12 (checked outside the example,
since healpy is test-only). The high-latitude spectrum falls as ell^-2.9 over
ell 10-50 (La Porta et al. 2008 found -2.5 to -3 at high latitude on the
408 MHz Haslam map) and is 30 to 60 times below the full-sky spectrum at
ell > 10. That ratio is physical, not a normalisation error: the full-sky
power comes from the Galactic plane, so the full-sky spectrum is not what a
masked estimate should reproduce.

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

B_CUT_DEG, TAPER_DEG = 20.0, 10.0
LMAX = 383
BANDS = ((10, 30), (30, 60), (60, 150))
FULL_SKY_SQ_DEG = 41252.96124941928


def galactic_mask(abs_b: torch.Tensor, cut: float, taper: float) -> torch.Tensor:
    """0 below ``cut``, 1 above ``cut + taper``, a cosine ramp in |b| between."""
    if taper == 0.0:
        return (abs_b > cut).to(abs_b.dtype)
    ramp = 0.5 - 0.5 * torch.cos(math.pi * (abs_b - cut) / taper)
    return torch.where(abs_b < cut, 0.0, torch.where(abs_b > cut + taper, 1.0, ramp))


def masked_spectrum(
    sky: torch.Tensor, weight: torch.Tensor, nside: int
) -> torch.Tensor:
    clean = sphere.remove_monopole_dipole(sky, nside, nest=True, valid_mask=weight > 0)
    w2 = float((weight * weight).mean())
    return sphere.anafast(clean * weight, nest=True, lmax=LMAX).to(torch.float64) / w2


def band_means(cl: torch.Tensor) -> list[float]:
    return [float(cl[lo:hi].mean()) for lo, hi in BANDS]


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

    pix = torch.arange(sky.numel(), device=dev)
    _, lat = sphere.pix2ang(nside, pix, nest=True, lonlat=True)
    abs_b = lat.abs().to(torch.float64)
    weight = galactic_mask(abs_b, B_CUT_DEG, TAPER_DEG)
    observed = torch.nonzero(weight > 0).squeeze(1)
    footprint = MOC(nest2uniq(nside, observed.cpu()))
    fsky = footprint.area_sq_deg / FULL_SKY_SQ_DEG
    w2 = float((weight * weight).mean())
    print(
        f"mask |b| > {B_CUT_DEG:.0f} deg, {TAPER_DEG:.0f} deg cosine taper: f_sky = {fsky:.4f} "
        f"(MOC area; analytic {1 - math.sin(math.radians(B_CUT_DEG)):.4f}), w2 = {w2:.4f}"
    )

    ell = torch.arange(LMAX + 1, dtype=torch.float64, device=dev)
    cl_full = sphere.anafast(sky, nest=True, lmax=LMAX).to(torch.float64)
    cl_20 = masked_spectrum(sky, weight, nside)
    cl_30 = masked_spectrum(sky, galactic_mask(abs_b, 30.0, TAPER_DEG), nside)
    sharp = galactic_mask(abs_b, B_CUT_DEG, 0.0)
    cl_sharp_raw = sphere.anafast(sky * sharp, nest=True, lmax=LMAX).to(
        torch.float64
    ) / float(sharp.mean())

    def even_odd(cl: torch.Tensor) -> float:
        return float((cl[100:200:2] / cl[101:201:2]).mean())

    print(
        f"even/odd multipole ratio at ell 100-200: sharp cut, monopole kept {even_odd(cl_sharp_raw):.2f}; "
        f"apodised, monopole+dipole removed {even_odd(cl_20):.3f}"
    )
    assert abs(even_odd(cl_20) - 1.0) < 0.05

    # Power law of the high-latitude spectrum, and a masked Gaussian simulation
    # drawn from the measured spectrum to test the estimator on this mask.
    sel = slice(10, 50)
    design = torch.stack([torch.ones_like(ell[sel]), ell[sel].log()], dim=1)
    coef = torch.linalg.lstsq(design, cl_20[sel].log().unsqueeze(1)).solution.squeeze()
    slope = float(coef[1])
    cl_model = cl_20.clone()  # simulate the measured spectrum itself
    cl_model[:2] = 0.0
    sim = sphere.reorder(
        sphere.synfast(cl_model.cpu(), nside=nside, lmax=LMAX).to(dev), r2n=True
    )
    cl_sim = masked_spectrum(sim, weight, nside)
    print(f"high-latitude spectrum: C_ell ~ ell^{slope:.2f} over ell 10-50")
    print("band          |b|>20 / full sky   |b|>30 / |b|>20   simulation out / in")
    sim_ratios = []
    for (lo, hi), d20, d30, full, s_out, s_in in zip(
        BANDS,
        band_means(cl_20),
        band_means(cl_30),
        band_means(cl_full),
        band_means(cl_sim),
        band_means(cl_model),
    ):
        sim_ratios.append(s_out / s_in)
        print(
            f"ell {lo:3d}-{hi:3d}   {d20 / full:12.4f}       {d30 / d20:12.3f}      {s_out / s_in:12.3f}"
        )
    assert all(abs(r - 1.0) < 0.1 for r in sim_ratios[1:])
    assert all(
        abs(a / b - 1.0) < 0.35
        for a, b in zip(band_means(cl_30)[1:], band_means(cl_20)[1:])
    )

    out = args.out or Path(tempfile.mkdtemp(prefix="torchsky-planck-"))
    out.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10, 4.0), constrained_layout=True)
    ax = fig.add_subplot(1, 2, 1, projection="mollweide")
    grid = healpix_sky_grid(sky.cpu(), nside=nside, nest=True)
    wgrid = healpix_sky_grid(weight.cpu(), nside=nside, nest=True).data
    lon = torch.deg2rad(0.5 * (grid.lon_edges[1:] + grid.lon_edges[:-1])).flip(0)
    lat_c = torch.deg2rad(0.5 * (grid.lat_edges[1:] + grid.lat_edges[:-1]))
    shown = torch.where(wgrid > 0, torch.log10(grid.data.clamp(min=1e-3)), torch.nan)
    im = ax.pcolormesh(
        -lon, lat_c, shown.flip(1), shading="auto", cmap="magma", rasterized=True
    )
    ax.contour(
        -lon, lat_c, wgrid.flip(1), levels=[0.5, 0.99], colors="w", linewidths=0.6
    )
    ax.set_xticklabels([])
    ax.set_title(
        f"Synchrotron 408 MHz, |b| > {B_CUT_DEG:.0f} deg, {TAPER_DEG:.0f} deg taper",
        fontsize=9,
        pad=14,
    )
    fig.colorbar(im, ax=ax, orientation="horizontal", shrink=0.7, label="log10 uK_RJ")
    ax = fig.add_subplot(1, 2, 2)
    x = ell[2:].cpu()
    for cl, style, label in (
        (cl_full, "k-", "full sky (Galactic plane dominated)"),
        (cl_sharp_raw, "C7.", r"sharp cut, monopole kept, / $f_{sky}$"),
        (cl_20, "C3-", r"apodised $|b|>20°$, / $w_2$"),
        (cl_30, "C1--", r"apodised $|b|>30°$, / $w_2$"),
        (cl_sim, "C0:", "simulation of the apodised estimate"),
    ):
        ax.loglog(x, cl[2:].cpu(), style, ms=2, lw=1.1, label=label)
    ax.set(
        xlabel=r"multipole $\ell$",
        ylabel=r"$C_\ell$ [$\mu K_{RJ}^2$]",
        title="Pseudo-$C_\\ell$ (60 arcmin beam not removed)",
        xlim=(2, 250),
    )
    ax.legend(fontsize=7)
    fig.savefig(out / "planck_power_spectrum.png", dpi=110)
    print(f"figure: {out / 'planck_power_spectrum.png'}; device={dev}")


if __name__ == "__main__":
    main()
