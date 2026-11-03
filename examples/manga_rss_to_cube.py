"""Build a MaNGA data cube from its 171 dithered fibre spectra with ``coadd_cube``.

Science question: starting from the row-stacked fibre spectra of a MaNGA IFU
observation, rather than from the released cube, can ``coadd_cube`` build a
cube that reproduces the SDSS pipeline cube? MaNGA plate-IFU 8485-1901 is a
19-fibre bundle observed in 9 exposures (3 dither positions, 8101 s in total):
171 spectra of 2 arcsec fibres, each on the common log-lambda grid, with a
mask and the sky position of the fibre at every wavelength (XPOS/YPOS, which
include differential atmospheric refraction).
The spectra become one IFU frame (wavelength x fibre) whose ``IFUSliceWCS``
carries RA, Dec and wavelength per sample, each fibre being its own slice.
Fibre fluxes are divided by the fibre area to give surface brightness, and the
6400-7000 A window (H-alpha, [N II] and [S II] at z = 0.037) is resampled
onto 1 arcsec spaxels with the exponential modified-Shepard kernel (``emsm``,
the JWST ``cube_build`` weighting), whose 1-spaxel width matches the 1 arcsec
fibre radius. Every good sample has the same weight, as in the pipeline:
inverse-variance weights would favour the fainter fibres in each spaxel,
because their Poisson noise is lower, and bias the cube low (by 17 percent
here).

The result is compared with the released DRP LOGCUBE (Law et al. 2016), built
from the same spectra with a Gaussian Shepard kernel on 0.5 arcsec spaxels and
summed here in 2 x 2 blocks onto the same 1 arcsec grid: the total flux of the
window (agreement to about 1 percent), the image centroid, the continuum and
H-alpha images spaxel by spaxel, and the integrated spectrum. Runtime about 10 s on CPU once the 16 MB fibre file and the 39 MB
cube are cached.

Data: SDSS-IV MaNGA DR17 (Bundy et al. 2015, ApJ 798, 7; Abdurro'uf et al.
2022, ApJS 259, 35), public; cite SDSS DR17 and MaNGA.

    pixi run -e test -- python examples/manga_rss_to_cube.py [--device cuda] [--out DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from _sample_data import fetch_sample, skip_if_unavailable

import torchfits
from torchsky.map import CubeTarget, Observation, coadd_cube
from torchsky.wcs import WCS, IFUSliceWCS

WINDOW = (6400.0, 7000.0)  # observed Å
HALPHA_OBS = (6795.0, 6815.0)  # H-alpha at z = 0.037, observed Å
CONTINUUM = ((6700.0, 6750.0), (6880.0, 6930.0))
FIBRE_AREA = torch.pi * 1.0**2  # arcsec^2, 2 arcsec diameter
SPAXEL = 1.0  # arcsec


def band(
    wave: torch.Tensor, cube: torch.Tensor, lo_hi: tuple[float, float]
) -> torch.Tensor:
    sel = (wave >= lo_hi[0]) & (wave < lo_hi[1])
    return cube[sel].sum(0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    dev = args.device

    with skip_if_unavailable():
        rss_path = str(fetch_sample("manga_8485_1901_rss"))
        cube_path = str(fetch_sample("manga_8485_1901"))

    primary = dict(torchfits.read_header(rss_path, 0))
    wave = torchfits.read(rss_path, hdu="WAVE").to(torch.float64)
    sel = (wave >= WINDOW[0]) & (wave < WINDOW[1])
    wave = wave[sel]
    flux = torchfits.read(rss_path, hdu="FLUX").to(torch.float64)[
        :, sel
    ]  # (fibre, wave)
    ivar = torchfits.read(rss_path, hdu="IVAR").to(torch.float64)[:, sel]
    mask = torchfits.read(rss_path, hdu="MASK").to(torch.int64)[:, sel]
    xpos = torchfits.read(rss_path, hdu="XPOS").to(torch.float64)[
        :, sel
    ]  # arcsec, + west (see below)
    ypos = torchfits.read(rss_path, hdu="YPOS").to(torch.float64)[
        :, sel
    ]  # arcsec, + north
    n_fib, n_wave = flux.shape
    ra0, dec0 = float(primary["IFURA"]), float(primary["IFUDEC"])
    dar = (xpos[:, -1] - xpos[:, 0]).hypot(ypos[:, -1] - ypos[:, 0])
    print(
        f"{primary['PLATEIFU']}: {n_fib} fibre spectra ({primary['NEXP']} exposures), "
        f"{n_wave} channels in {WINDOW[0]:.0f}-{WINDOW[1]:.0f} A; refraction moves the fibres "
        f"by {float(dar.median()):.3f} arcsec across the window"
    )

    # One IFU frame: rows = wavelength, columns = fibres (each fibre is a slice).
    good = (mask == 0) & (ivar > 0) & torch.isfinite(flux)
    cosd = torch.cos(torch.deg2rad(torch.tensor(dec0, dtype=torch.float64)))
    ifu = IFUSliceWCS(
        # +XPOS runs along the +x axis of the east-left DRP cube, i.e. towards the
        # west: with the opposite sign the image centroids differ by 0.1 spaxel.
        ra_map=(ra0 - xpos / 3600.0 / cosd).T.contiguous(),
        dec_map=(dec0 + ypos / 3600.0).T.contiguous(),
        wavelength_map=wave[:, None].expand(n_wave, n_fib).contiguous(),
        slice_id_map=torch.arange(n_fib)[None, :].expand(n_wave, n_fib).contiguous(),
    )
    sb = torch.where(good, flux / FIBRE_AREA, 0.0).T.float()
    frame = Observation(
        data=sb,
        ifu_wcs=ifu,
        mask=good.T,
        # Equal weights for every good sample, as in the DRP: inverse-variance
        # weights would favour the fainter fibres of each spaxel (Poisson noise
        # rises with flux) and bias the cube low by about 17 percent here.
        weight=good.T.float(),
    )

    # The DRP cube on 0.5 arcsec spaxels, summed 2 x 2 onto 1 arcsec spaxels.
    drp_header = dict(torchfits.read_header(cube_path, "FLUX"))
    drp_wave = torchfits.read(cube_path, hdu="WAVE").to(torch.float64)
    dsel = (drp_wave >= WINDOW[0]) & (drp_wave < WINDOW[1])
    assert torch.allclose(drp_wave[dsel], wave), (
        "RSS and cube share one log-lambda grid"
    )
    drp = torchfits.read(cube_path, hdu="FLUX").to(torch.float64)[
        dsel
    ]  # per 0.5" spaxel
    nz, ny, nx = drp.shape
    drp = drp.reshape(nz, ny // 2, 2, nx // 2, 2).sum((2, 4))  # per 1" spaxel

    # Target: the DRP tangent point and orientation, 1 arcsec spaxels aligned with the blocks.
    n = nx // 2
    header = {
        "NAXIS": 2,
        "NAXIS1": n,
        "NAXIS2": n,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRVAL1": drp_header["CRVAL1"],
        "CRVAL2": drp_header["CRVAL2"],
        # DRP pixel p (0-based) lies at block coordinate (p - 0.5) / 2.
        "CRPIX1": (drp_header["CRPIX1"] - 1 - 0.5) / 2 + 1,
        "CRPIX2": (drp_header["CRPIX2"] - 1 - 0.5) / 2 + 1,
        "CDELT1": -SPAXEL / 3600.0,
        "CDELT2": SPAXEL / 3600.0,
    }
    log_w = torch.log(wave)
    edges = torch.exp(
        torch.cat(
            [
                log_w[:1] - 0.5 * (log_w[1] - log_w[0]),
                0.5 * (log_w[1:] + log_w[:-1]),
                log_w[-1:] + 0.5 * (log_w[-1] - log_w[-2]),
            ]
        )
    )
    target = CubeTarget(spatial_wcs=WCS(header), spectral_bin_edges=edges)
    result = coadd_cube(
        [frame], target, spectral_kernel="emsm", emsm_scale=1.0, device=dev
    )
    ours = result.data.cpu().double() * SPAXEL**2  # per 1" spaxel, like the binned DRP
    covered = (result.weight.cpu() > 0).all(0)
    print(
        f"cube {tuple(ours.shape)} (wave, y, x) at {SPAXEL:.0f} arcsec; {int(covered.sum())} spaxels covered in every channel"
    )

    both = covered & (drp != 0).all(0)
    f_ours = float(ours[:, both].sum())
    f_drp = float(drp[:, both].sum())
    print(
        f"total flux in the window over {int(both.sum())} spaxels: ours / DRP = {f_ours / f_drp:.4f}"
    )
    assert abs(f_ours / f_drp - 1.0) < 0.03
    grid = torch.arange(n, dtype=torch.float64)

    def centroid(image: torch.Tensor) -> tuple[float, float]:
        m = image * both
        return float((m.sum(0) * grid).sum() / m.sum()), float(
            (m.sum(1) * grid).sum() / m.sum()
        )

    (xo, yo), (xd, yd) = centroid(ours.sum(0)), centroid(drp.sum(0))
    print(f"image centroid ours - DRP: ({xo - xd:+.3f}, {yo - yd:+.3f}) spaxels")
    assert abs(xo - xd) < 0.05 and abs(yo - yd) < 0.05

    cont_o = 0.5 * (
        band(wave, ours, CONTINUUM[0])
        / ((wave >= CONTINUUM[0][0]) & (wave < CONTINUUM[0][1])).sum()
        + band(wave, ours, CONTINUUM[1])
        / ((wave >= CONTINUUM[1][0]) & (wave < CONTINUUM[1][1])).sum()
    )
    cont_d = 0.5 * (
        band(wave, drp, CONTINUUM[0])
        / ((wave >= CONTINUUM[0][0]) & (wave < CONTINUUM[0][1])).sum()
        + band(wave, drp, CONTINUUM[1])
        / ((wave >= CONTINUUM[1][0]) & (wave < CONTINUUM[1][1])).sum()
    )
    n_line = int(((wave >= HALPHA_OBS[0]) & (wave < HALPHA_OBS[1])).sum())
    ha_o = band(wave, ours, HALPHA_OBS) - n_line * cont_o
    ha_d = band(wave, drp, HALPHA_OBS) - n_line * cont_d
    for name, a, b in (("continuum", cont_o, cont_d), ("H-alpha", ha_o, ha_d)):
        bright = both & (b > 0.1 * b[both].max())
        corr = float(torch.corrcoef(torch.stack([a[bright], b[bright]]))[0, 1])
        ratio = a[bright] / b[bright]
        print(
            f"{name:9s} image: {int(bright.sum())} spaxels above 10 % of peak, correlation {corr:.3f}, "
            f"ours / DRP median {float(ratio.median()):.3f} (16-84 %: {float(ratio.quantile(0.16)):.3f}-{float(ratio.quantile(0.84)):.3f})"
        )
        assert corr > 0.9
    spec_o, spec_d = ours[:, both].sum(1), drp[:, both].sum(1)
    rel = (spec_o - spec_d) / spec_d.abs().clamp_min(1e-6)
    print(
        f"integrated spectrum: ours / DRP median {float((spec_o / spec_d).median()):.4f}, rms of the relative difference {float(rel.std()):.4f}"
    )

    out = args.out or Path(tempfile.mkdtemp(prefix="torchsky-manga-rss-"))
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(13.0, 3.4),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1, 1, 1, 1.6]},
    )
    off = torch.stack([-xpos[:, n_wave // 2], ypos[:, n_wave // 2]])  # east, north
    axes[0].scatter(
        off[0], off[1], s=60, c=torch.arange(n_fib) // 19, cmap="tab10", alpha=0.6
    )
    axes[0].invert_xaxis()
    axes[0].set_aspect("equal")
    axes[0].set_title("171 fibre centres (colour: exposure)", fontsize=9)
    axes[0].set_xlabel("east offset [arcsec]", fontsize=8)
    axes[0].set_ylabel("north offset [arcsec]", fontsize=8)
    vmax = float(ha_d[both].max())
    for ax, image, title in (
        (axes[1], ha_o, "H-alpha, coadd_cube from fibres"),
        (axes[2], ha_d, "H-alpha, DRP cube (2x2 binned)"),
    ):
        ax.imshow(
            torch.where(both, image, torch.nan),
            origin="lower",
            cmap="magma",
            vmin=0,
            vmax=vmax,
        )
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[3].plot(wave, spec_d, lw=1.2, label="DRP cube")
    axes[3].plot(wave, spec_o, lw=0.8, label="coadd_cube")
    axes[3].set_xlabel("observed wavelength [A]", fontsize=8)
    axes[3].set_ylabel("1e-17 erg/s/cm2/A", fontsize=8)
    axes[3].set_title("spectrum summed over the bundle", fontsize=9)
    axes[3].legend(fontsize=8)
    fig.savefig(out / "manga_rss_to_cube.png", dpi=100)
    print(f"figure: {out / 'manga_rss_to_cube.png'}; device={dev}")


if __name__ == "__main__":
    main()
