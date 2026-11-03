"""SFD dust sampled at Gaia star positions in Taurus, then dereddened with G23.

Science question: how much of the colour spread of stars behind the Taurus
dark clouds is Galactic reddening, and does a 2D dust map remove it? The
Schlegel, Finkbeiner & Davis (1998) E(B-V) map, as a HEALPix nside=512 NESTED
map in Galactic coordinates, is read with torchfits and sampled with bilinear
``torchsky.sphere.sample_healpix_map`` at the positions of 4831 Gaia DR3 stars
(G < 17, parallax_over_error > 5) within 1 degree of (RA 64.5, Dec +27.0),
after an ICRS-to-Galactic ``transform``. The map is rescaled by 0.86 (Schlafly
& Finkbeiner 2011) and converted to A_G, A_BP and A_RP with the Gordon et al.
(2023) law (``a_lambda(law="g23")``, R_V = 3.1) at the Gaia DR3 pivot
wavelengths. The SFD map integrates dust along the whole line of sight, so
the correction applies to stars behind the clouds (parallax < 5 mas, beyond
200 pc; the clouds are at about 140 pc). Because every step is a torch
operation, the dust-map scale is then fitted by gradient descent: one scale
factor s on the map and a cubic main-sequence locus (BP-RP)_0(M_G,0) are fitted
together by minimising a Huber loss of the dereddened colours of
main-sequence stars (4 < M_G,0 < 8, a smooth window). The check is the width
of the main sequence: no correction, the nominal map (s = 1) and the fitted s.

Caveats: the band extinction of a broad Gaia band depends on the stellar
spectrum and on A_V itself (10-20% lower E(BP-RP) for K and M dwarfs at high
A_V); pivot wavelengths ignore that, so s absorbs both this and any SFD
calibration error. SFD is known to over-predict reddening in Taurus (Arce &
Goodman 1999) and is unreliable inside the densest cores.

Data: SFD E(B-V) HEALPix map from NASA LAMBDA (public domain, 25 MB; cite
Schlegel et al. 1998) and Gaia DR3 (ESA/Gaia/DPAC, CC BY-SA 3.0 IGO) via the
IRSA TAP mirror, both pinned by sha256. Runtime about 5 s on CPU once the
samples are cached.

    pixi run -e test -- python examples/dust_extinction_taurus.py [--device cuda] [--out DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyarrow.csv as pacsv
import torch

from _sample_data import fetch_sample, skip_if_unavailable

import torchfits
from torchsky.coords import transform
from torchsky.extinction import a_lambda
from torchsky.sphere import sample_healpix_map

CENTER = (64.5, 27.0)
SF11_SCALE = 0.86  # Schlafly & Finkbeiner 2011: E(B-V)_true = 0.86 E(B-V)_SFD
R_V = 3.1
# Gaia DR3 passband pivot wavelengths [Angstrom] (SVO Filter Profile Service).
PIVOT = {"G": 6217.59, "BP": 5109.71, "RP": 7769.02}
BEHIND_PLX = 5.0  # mas: beyond 200 pc
MS_RANGE = (4.0, 8.0)  # dereddened M_G window of the main-sequence fit
N_STEPS = 1500


def column(table, name: str, device: str) -> torch.Tensor:
    values = table[name].to_numpy(zero_copy_only=False).astype("float64")
    return torch.from_numpy(values).to(device)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    dev = args.device

    with skip_if_unavailable():
        map_path = fetch_sample("sfd_ebv_healpix_512")
        cat_path = fetch_sample("gaia_taurus_cone")
    header = dict(torchfits.read_header(str(map_path), 1))
    nside = int(header["NSIDE"])
    assert header["ORDERING"] == "NESTED" and header["PIXTYPE"] == "HEALPIX"
    sfd = torchfits.read(str(map_path), hdu=1)["TEMPERATURE"].to(torch.float64).to(dev)
    print(
        f"SFD E(B-V): nside {nside}, {sfd.numel()} pixels, NESTED, Galactic; "
        f"all-sky median {float(sfd.median()):.3f} mag"
    )

    table = pacsv.read_csv(cat_path)
    ra, dec = column(table, "ra", dev), column(table, "dec", dev)
    plx = column(table, "parallax", dev)
    g, bp, rp = (column(table, f"phot_{b}_mean_mag", dev) for b in ("g", "bp", "rp"))
    glon, glat = transform(ra, dec, from_frame="ICRS", to_frame="GALACTIC")
    ebv = SF11_SCALE * sample_healpix_map(
        sfd, nside, glon, glat, nest=True, interpolation="bilinear"
    )
    q = torch.quantile(
        ebv, torch.tensor([0.05, 0.5, 0.95], dtype=ebv.dtype, device=dev)
    )
    print(
        f"{table.num_rows} Gaia stars at l = {float(glon.mean()):.1f}, b = {float(glat.mean()):.1f}: "
        f"E(B-V) 5/50/95% = {float(q[0]):.2f} / {float(q[1]):.2f} / {float(q[2]):.2f} mag"
    )

    wave = torch.tensor(
        [PIVOT["G"], PIVOT["BP"], PIVOT["RP"]], dtype=torch.float64, device=dev
    )
    per_ebv = a_lambda(wave, ebv=1.0, r_v=R_V, law="g23")
    print(
        f"G23, R_V = {R_V}: A_G, A_BP, A_RP per E(B-V) = "
        f"{float(per_ebv[0]):.3f}, {float(per_ebv[1]):.3f}, {float(per_ebv[2]):.3f}; "
        f"E(BP-RP)/E(B-V) = {float(per_ebv[1] - per_ebv[2]):.3f}"
    )
    colour = bp - rp
    abs_g = g + 5.0 * torch.log10(plx / 100.0)
    behind = plx < BEHIND_PLX
    k_g, k_col = per_ebv[0].detach(), (per_ebv[1] - per_ebv[2]).detach()

    def fit(scale: float | None) -> tuple[float, torch.Tensor, float]:
        """Fit the MS locus (and the map scale when ``scale`` is None); return s, coef, NMAD."""
        s = torch.tensor(
            1.0 if scale is None else scale, dtype=torch.float64, device=dev
        )
        s.requires_grad_(scale is None)
        coef = torch.zeros(4, dtype=torch.float64, device=dev, requires_grad=True)
        opt = torch.optim.Adam([coef, s] if scale is None else [coef], lr=0.03)
        for _ in range(N_STEPS):
            resid, weight = ms_residual(s, coef)
            loss = (
                weight
                * torch.nn.functional.huber_loss(
                    resid, torch.zeros_like(resid), reduction="none", delta=0.1
                )
            ).sum() / weight.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            resid, weight = ms_residual(s, coef)
            core = resid[weight > 0.5]
            nmad = float(1.4826 * (core - core.median()).abs().median())
        return float(s.detach()), coef.detach(), nmad

    def ms_residual(s, coef):
        colour0 = colour - s * ebv * k_col
        abs_g0 = abs_g - s * ebv * k_g
        weight = torch.sigmoid(8.0 * (abs_g0 - MS_RANGE[0])) * torch.sigmoid(
            8.0 * (MS_RANGE[1] - abs_g0)
        )
        x = (abs_g0 - 6.0) / 2.0
        locus = coef[0] + coef[1] * x + coef[2] * x**2 + coef[3] * x**3
        return colour0 - locus, weight * behind

    _, _, nmad_none = fit(0.0)
    _, _, nmad_nominal = fit(1.0)
    s_fit, coef, nmad_fit = fit(None)
    print(
        f"main-sequence colour width (NMAD) behind the clouds ({int(behind.sum())} stars): "
        f"{nmad_none:.3f} mag uncorrected, {nmad_nominal:.3f} with 0.86 x SFD, "
        f"{nmad_fit:.3f} with the fitted scale s = {s_fit:.2f}"
    )
    print(
        f"fitted reddening is {s_fit:.2f} x the Schlafly-Finkbeiner-scaled SFD "
        f"({s_fit * SF11_SCALE:.2f} x raw SFD) for these stars"
    )
    assert nmad_fit < nmad_none and 0.3 < s_fit < 1.2
    colour0 = colour - s_fit * ebv * k_col
    abs_g0 = abs_g - s_fit * ebv * k_g

    out = args.out or Path(tempfile.mkdtemp(prefix="torchsky-dust-taurus-"))
    out.mkdir(parents=True, exist_ok=True)
    cpu = lambda t: t.detach().cpu()  # noqa: E731
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), constrained_layout=True)
    # Dust map around the field, sampled on a grid in (RA, Dec).
    gra = torch.linspace(
        CENTER[0] - 1.4, CENTER[0] + 1.4, 240, dtype=torch.float64, device=dev
    )
    gdec = torch.linspace(
        CENTER[1] - 1.25, CENTER[1] + 1.25, 220, dtype=torch.float64, device=dev
    )
    mra, mdec = torch.meshgrid(gra, gdec, indexing="xy")
    ml, mb = transform(mra, mdec, from_frame="ICRS", to_frame="GALACTIC")
    image = SF11_SCALE * sample_healpix_map(sfd, nside, ml, mb, nest=True)
    ax = axes[0]
    im = ax.imshow(
        cpu(image),
        origin="lower",
        cmap="magma_r",
        vmin=0.0,
        vmax=1.6,
        extent=(float(gra[0]), float(gra[-1]), float(gdec[0]), float(gdec[-1])),
        aspect="auto",
    )
    ax.scatter(
        cpu(ra[behind]),
        cpu(dec[behind]),
        s=0.6,
        c="C0",
        alpha=0.6,
        label="stars behind the clouds (plx < 5 mas)",
    )
    ax.invert_xaxis()
    ax.set(
        xlabel="RA [deg]", ylabel="Dec [deg]", title="0.86 x SFD E(B-V) and Gaia stars"
    )
    ax.legend(fontsize=7, loc="lower left", markerscale=4)
    fig.colorbar(im, ax=ax, label="E(B-V) [mag]", shrink=0.85)
    ax = axes[1]
    sc = ax.scatter(
        cpu(colour[behind]),
        cpu(g[behind]),
        s=3,
        c=cpu(ebv[behind]),
        cmap="magma_r",
        vmin=0,
        vmax=1.6,
    )
    ax.invert_yaxis()
    ax.set(
        xlabel="BP - RP [mag]",
        ylabel="G [mag]",
        title="Observed (behind the clouds)",
        xlim=(0, 4.5),
    )
    ax = axes[2]
    ax.scatter(
        cpu(colour0[behind]),
        cpu(abs_g0[behind]),
        s=3,
        c=cpu(ebv[behind]),
        cmap="magma_r",
        vmin=0,
        vmax=1.6,
    )
    ax.invert_yaxis()
    mgrid = torch.linspace(
        MS_RANGE[0], MS_RANGE[1], 50, dtype=torch.float64, device=dev
    )
    xg = (mgrid - 6.0) / 2.0
    ax.plot(
        cpu(coef[0] + coef[1] * xg + coef[2] * xg**2 + coef[3] * xg**3),
        cpu(mgrid),
        "C0-",
        lw=1.5,
        label="fitted locus",
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.set(
        xlabel="(BP - RP)$_0$ [mag]",
        ylabel="M$_{G,0}$ [mag]",
        xlim=(-0.5, 3.5),
        title=f"Dereddened: G23, {s_fit:.2f} x 0.86 x SFD",
    )
    fig.colorbar(sc, ax=axes[1:], label="E(B-V) [mag]", shrink=0.85)
    target = out / "dust_extinction_taurus.png"
    fig.savefig(target, dpi=110)
    print(f"figure: {target}; device={dev}")


if __name__ == "__main__":
    main()
