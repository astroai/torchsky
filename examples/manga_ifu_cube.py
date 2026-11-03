"""Hα map, velocity field and spectrum of a MaNGA IFU cube.

Science question: where does a star-forming disc galaxy form stars, and how
does it rotate? The SDSS-IV MaNGA cube 8485-1901 (a 19-fibre bundle, 34 x 34
spaxels of 0.5 arcsec, 4563 logarithmic wavelength channels) is read with
torchfits. Its 3D header goes to ``torchsky.wcs.WCS``, which evaluates the
``WAVE-LOG`` spectral axis; the result is checked against the cube's own
``WAVE`` extension. The cube becomes a ``torchsky.spectral.DataCube``. The
redshift comes from the Hα line of the central spectrum, and every spaxel is
rebinned onto a 1 Å rest-frame grid around Hα and [N II] with
``flux_conserving_rebin``, which also propagates the inverse variance. A
continuum-subtracted narrow-band Hα map and an Hα velocity field are measured
on that grid. The rest-frame cube is then reprojected with ``coadd_cube`` onto
a rotated, 0.25 arcsec, north-up-at-30-degrees TAN grid, and the integrated Hα
flux is checked to be conserved. The figure shows the white-light image, the
Hα map, the velocity field and the central spectrum.

Data: SDSS DR17 MaNGA LOGCUBE manga-8485-1901 (Bundy et al. 2015, ApJ 798, 7;
Abdurro'uf et al. 2022, ApJS 259, 35), SDSS public data. Wavelengths are in
vacuum; flux in 1e-17 erg/s/cm^2/Å per spaxel. Runtime about 5 s on CPU once
the 39 MB cube is cached.

    pixi run -e test -- python examples/manga_ifu_cube.py [--device cuda] [--out DIR]
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
from torchsky.map import Observation, ResampleConfig, coadd_cube, plan, plan_cube
from torchsky.spectral import DataCube, SpectralAxis, flux_conserving_rebin
from torchsky.wcs import WCS

C_KMS = 299792.458
HALPHA = 6564.61  # vacuum rest wavelengths, Å
NII = (6549.86, 6585.27)
DONOTUSE = 1 << 10  # MANGA_DRP3PIXMASK bit 10
REST_EDGES = torch.arange(6500.0, 6631.0, 1.0, dtype=torch.float64)  # 1 Å bins
LINE = (HALPHA - 8.0, HALPHA + 8.0)
SIDEBANDS = ((6510.0, 6535.0), (6600.0, 6625.0))


def window(centres: torch.Tensor, lo_hi: tuple[float, float]) -> torch.Tensor:
    return (centres >= lo_hi[0]) & (centres < lo_hi[1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    dev = args.device

    with skip_if_unavailable():
        path = fetch_sample("manga_8485_1901")
    header = dict(torchfits.read_header(str(path), "FLUX"))
    flux = torchfits.read(str(path), hdu="FLUX").to(torch.float64)
    ivar = torchfits.read(str(path), hdu="IVAR").to(torch.float64)
    bad = (torchfits.read(str(path), hdu="MASK").to(torch.int64) & DONOTUSE) != 0
    wave_ext = torchfits.read(str(path), hdu="WAVE").to(torch.float64)
    nz, ny, nx = flux.shape
    print(
        f"{path.name}: {nx} x {ny} spaxels x {nz} channels, {header['CTYPE3']}, {header['BUNIT']}"
    )

    # Spectral axis from the 3D WCS (Paper III WAVE-LOG), checked against WAVE.
    wcs = WCS(header)
    chan = torch.arange(nz, dtype=torch.float64)
    centre_px = torch.full_like(chan, (nx - 1) / 2)
    world = wcs.pixel_to_world(torch.stack([centre_px, centre_px, chan], dim=-1))
    edge_px = torch.arange(nz + 1, dtype=torch.float64) - 0.5
    edge_world = wcs.pixel_to_world(
        torch.stack(
            [edge_px * 0 + (nx - 1) / 2, edge_px * 0 + (ny - 1) / 2, edge_px], -1
        )
    )
    wave, edges = world[:, 2], edge_world[:, 2]
    dwave = float((wave - wave_ext).abs().max())
    print(
        f"WAVE-LOG axis {float(wave[0]):.2f}-{float(wave[-1]):.2f} Å; max |WCS - WAVE HDU| = {dwave:.1e} Å"
    )
    assert dwave < 1e-3
    cube = DataCube(
        data=flux,
        spectral_axis=SpectralAxis(wave, "Angstrom", "WAVE"),
        ivar=ivar,
        mask=bad,
        header=header,
    )

    # Redshift from the Hα peak of the central 3 x 3 spaxels.
    cy, cx = ny // 2, nx // 2
    central = sum(
        cube.extract_spectrum(cy + dy, cx + dx).flux
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
    )
    search = (wave > 6600.0) & (wave < 7000.0)
    peak = int(torch.nonzero(search)[0]) + int(central[search].argmax())
    sl = slice(peak - 3, peak + 4)
    cont = torch.median(central[(wave > wave[peak] - 60) & (wave < wave[peak] - 30)])
    w = (central[sl] - cont).clamp(min=0)
    z = float((w * wave[sl]).sum() / w.sum()) / HALPHA - 1.0
    print(f"systemic redshift from central Hα: z = {z:.5f} (cz = {z * C_KMS:.0f} km/s)")

    # Rebin every spaxel onto a rest-frame 1 Å grid (flux density scales by 1+z).
    spaxels = flux.permute(1, 2, 0).reshape(-1, nz).to(dev)
    rest = flux_conserving_rebin(
        spaxels * (1.0 + z),
        (edges / (1.0 + z)).to(dev),
        REST_EDGES.to(dev),
        ivar=ivar.permute(1, 2, 0).reshape(-1, nz).to(dev) / (1.0 + z) ** 2,
        mask=~bad.permute(1, 2, 0).reshape(-1, nz).to(dev),
    )
    rest_centres = 0.5 * (REST_EDGES[1:] + REST_EDGES[:-1]).to(dev)
    rest_flux = rest.flux.reshape(ny, nx, -1)
    rest_ivar = rest.ivar.reshape(ny, nx, -1)
    valid = rest.mask.reshape(ny, nx, -1)

    # Narrow-band Hα: line window minus a linear continuum from two sidebands.
    side = window(rest_centres, SIDEBANDS[0]) | window(rest_centres, SIDEBANDS[1])
    sx, sy = rest_centres[side], rest_flux[..., side]
    slope = ((sx - sx.mean()) * (sy - sy.mean(-1, keepdim=True))).sum(-1) / (
        (sx - sx.mean()) ** 2
    ).sum()
    continuum = sy.mean(-1, keepdim=True) + slope.unsqueeze(-1) * (
        rest_centres - sx.mean()
    )
    line = window(rest_centres, LINE)
    excess = (rest_flux - continuum)[..., line]
    halpha = excess.sum(-1)  # 1 Å bins: sum of flux density = flux per spaxel
    halpha_err = (1.0 / rest_ivar[..., line].clamp(min=1e-30)).sum(-1).sqrt()
    good = valid[..., line].all(-1) & (rest_ivar[..., line] > 0).all(-1)
    detected = good & (halpha > 5.0 * halpha_err)
    weight = excess.clamp(min=0)
    centroid = (weight * rest_centres[line]).sum(-1) / weight.sum(-1).clamp(min=1e-30)
    velocity = torch.where(detected, (centroid / HALPHA - 1.0) * C_KMS, torch.nan)
    v = velocity[detected]
    print(
        f"Hα detected at 5 sigma in {int(detected.sum())} spaxels; total Hα flux "
        f"{float(halpha[detected].sum()):.0f} x 1e-17 erg/s/cm^2; velocity range "
        f"{float(v.quantile(0.05)):+.0f} to {float(v.quantile(0.95)):+.0f} km/s"
    )
    nii = (rest_flux - continuum)[
        ..., window(rest_centres, (NII[1] - 4.0, NII[1] + 4.0))
    ].sum(-1)
    ratio = float(nii[detected].sum() / halpha[detected].sum())
    print(
        f"integrated [N II] 6585 / Hα = {ratio:.2f} (log {math.log10(ratio):+.2f}: star formation, not AGN)"
    )
    assert int(detected.sum()) > 100 and 0.05 < ratio < 0.6

    # Reproject the continuum-subtracted rest-frame cube onto a rotated grid.
    spatial = {
        k: header[k]
        for k in (
            "CTYPE1",
            "CTYPE2",
            "CRVAL1",
            "CRVAL2",
            "CRPIX1",
            "CRPIX2",
            "CD1_1",
            "CD2_2",
        )
    }
    spatial.update(NAXIS=2, NAXIS1=nx, NAXIS2=ny, CD1_2=0.0, CD2_1=0.0)
    planes = (rest_flux - continuum).permute(2, 0, 1)[line.cpu()]
    observations = [
        Observation(data=plane.float(), wcs=spatial, wavelength=float(lam), weight=1.0)
        for plane, lam in zip(planes, rest_centres[line].cpu())
    ]
    grid2d = plan(
        observations[:1], pixel_scale_arcsec=0.25, rotation_deg=30.0, device=dev
    )
    target = plan_cube(
        observations,
        spectral_axis=rest_centres[line].cpu(),
        header_overrides=dict(grid2d.header),
        device=dev,
    )
    reprojected = coadd_cube(
        observations,
        target,
        resample=ResampleConfig(semantics="flux", kernel="lanczos3"),
        device=dev,
    )
    total_in = float(planes.sum())
    total_out = float(torch.nan_to_num(reprojected.data).sum())
    print(
        f"reprojected cube {tuple(reprojected.data.shape)} at 0.25 arcsec, rotated 30 deg: "
        f"Hα total {total_out:.0f} vs {total_in:.0f} ({total_out / total_in:.4f})"
    )
    assert abs(total_out / total_in - 1.0) < 0.02

    out = args.out or Path(tempfile.mkdtemp(prefix="torchsky-manga-"))
    out.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(11, 6.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 4)
    white = cube.collapse_spectral("mean", wavelength_range=(4000.0, 9000.0))
    panels = [
        ("g-to-i white light", white, "gray_r", None),
        (
            "Hα flux (1e-17 erg/s/cm²)",
            torch.where(good.cpu(), halpha.cpu(), torch.nan),
            "magma",
            None,
        ),
        (
            "Hα velocity (km/s)",
            (velocity - torch.nanmedian(velocity)).cpu(),
            "RdBu_r",
            (-120, 120),
        ),
        ("Hα, reprojected (30° grid)", reprojected.data.sum(0).cpu(), "magma", None),
    ]
    for i, (title, img, cmap, lim) in enumerate(panels):
        ax = fig.add_subplot(grid[0, i])
        kw = {"vmin": lim[0], "vmax": lim[1]} if lim else {}
        im = ax.imshow(img, origin="lower", cmap=cmap, **kw)
        ax.set(title=title, xticks=[], yticks=[])
        fig.colorbar(im, ax=ax, shrink=0.75)
    ax = fig.add_subplot(grid[1, :])
    rest_wave = wave / (1.0 + z)
    flagged = bad[:, cy - 1 : cy + 2, cx - 1 : cx + 2].any(-1).any(-1)
    shown = torch.where(flagged, torch.nan, central * (1.0 + z))
    top = 1.25 * float(
        shown[(rest_wave > 6500) & (rest_wave < 6600)].nan_to_num().max()
    )
    ax.plot(rest_wave, shown, "k-", lw=0.6)
    lines = (
        (3728.5, "[O II]"),
        (4862.68, "Hβ"),
        (5008.24, "[O III]"),
        (HALPHA, "Hα"),
        (NII[1], "[N II]"),
        (6718.29, "[S II]"),
    )
    for k, (lam, name) in enumerate(lines):
        ax.axvline(lam, color="C3", lw=0.6, alpha=0.5)
        ax.text(lam + 15, top * (0.92 - 0.08 * (k % 2)), name, color="C3", fontsize=8)
    ax.set(
        xlabel="rest wavelength (vacuum, Å)",
        ylabel="flux density, central 3x3",
        xlim=(3600, 9300),
        ylim=(0, top),
        title="Central 3x3 spectrum in the rest frame (DONOTUSE pixels masked)",
    )
    fig.savefig(out / "manga_ifu_cube.png", dpi=100)
    print(f"figure: {out / 'manga_ifu_cube.png'}; device={dev}")


if __name__ == "__main__":
    main()
