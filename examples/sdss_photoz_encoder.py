"""Photometric redshifts of SDSS galaxies with the experimental torchsky encoders.

Science question: how well do five SDSS broad-band fluxes alone constrain the
redshift of a spectroscopic galaxy, with a small model trained from scratch in
seconds? The sample is every SDSS DR17 spectroscopic galaxy (zWarning = 0) in a
6 x 6 degree patch (RA 150-156, Dec 0-6): 7145 objects, mostly BOSS luminous
red galaxies (median z = 0.31) plus the SDSS main sample. Model fluxes
(nanomaggies) are corrected for Galactic dust with the Gordon et al. (2023)
law at the SDSS effective wavelengths, using E(B-V) = extinction_r / 2.751
from the SDSS catalogue. Each galaxy becomes a set of five measurement tokens
(flux, error, passband id) for ``torchsky.encoders.PhotometryEncoder``; a
``BinnedRedshiftHead`` with 40 bins on 0 < z < 1 gives p(z). The model is
trained by maximum likelihood on a fixed random 70% of the galaxies and
evaluated on the other 30%. Reported on the test set: sigma_NMAD of
dz / (1 + z), the outlier fraction |dz| / (1 + z) > 0.15, the mean bias, and the
mean log density of the true redshift; the SDSS DR17 Photoz table (Beck et al.
2016, a k-nearest-neighbour fit trained on far more spectra, which may include
these galaxies) is scored on the same objects as a reference, not as a fair
competitor. The figure shows photometric against spectroscopic redshift for both
estimates.

This is a small demonstration, not a production photo-z: no hyperparameter
search, no colour-space reweighting, and the training set is the
spectroscopic selection itself, so the metrics do not transfer to a
photometric sample. The 300 training epochs are a runtime budget: on this
split sigma_NMAD is 0.031 after 150 epochs and 0.027 after 300 (batch 1024),
and 0.020 after 600 epochs at batch 512 and learning rate 3e-3 (90 s).

Data: SDSS DR17 (public data release; cite Abdurro'uf et al. 2022 and
Beck et al. 2016) from a SkyServer SQL query pinned by sha256 (1.25 MB CSV).
Runtime about 35 s on CPU once the sample is cached.

    pixi run -e test -- python examples/sdss_photoz_encoder.py [--device cuda] [--out DIR]
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

from torchsky.encoders import BinnedRedshiftHead, PhotometryEncoder
from torchsky.extinction import a_lambda

BANDS = "ugriz"
# SDSS effective wavelengths [Angstrom] (Fukugita et al. 1996).
WAVE = (3551.0, 4686.0, 6166.0, 7480.0, 8932.0)
SFD_R_COEFF = 2.751  # SDSS extinction_r = 2.751 E(B-V)_SFD (Schlegel et al. 1998)
Z_EDGES = torch.linspace(0.0, 1.0, 41)
TRAIN_FRACTION = 0.7
LATENT = 48
EPOCHS = 300
OUTLIER = 0.15


def column(table, name: str) -> torch.Tensor:
    return torch.from_numpy(
        table[name].to_numpy(zero_copy_only=False).astype("float32")
    )


def scores(z_phot: torch.Tensor, z_spec: torch.Tensor) -> dict[str, float]:
    dz = (z_phot - z_spec) / (1.0 + z_spec)
    return {
        "nmad": float(1.4826 * (dz - dz.median()).abs().median()),
        "outlier": float((dz.abs() > OUTLIER).float().mean()),
        "bias": float(dz.mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    dev = args.device
    torch.manual_seed(0)

    with skip_if_unavailable():
        path = fetch_sample("sdss_dr17_photoz_sample")
    table = pacsv.read_csv(path, read_options=pacsv.ReadOptions(skip_rows=1))
    flux = torch.stack([column(table, f"modelFlux_{b}") for b in BANDS], dim=1)
    ivar = torch.stack([column(table, f"modelFluxIvar_{b}") for b in BANDS], dim=1)
    z_spec = column(table, "specz")
    z_sdss = column(table, "photoz")
    n = flux.shape[0]
    print(
        f"SDSS DR17 spectroscopic galaxies: {n}; z_spec 10/50/90% = "
        + " / ".join(
            f"{float(v):.3f}"
            for v in torch.quantile(z_spec, torch.tensor([0.1, 0.5, 0.9]))
        )
    )

    # Galactic dust: per-object E(B-V) from the catalogue, G23 at the band wavelengths.
    ebv = column(table, "extinction_r").double() / SFD_R_COEFF
    a_band = a_lambda(
        torch.tensor(WAVE, dtype=torch.float64), ebv=ebv, r_v=3.1, law="g23"
    )
    flux = flux * torch.pow(10.0, 0.4 * a_band).float()
    error = torch.where(ivar > 0, ivar.clamp_min(1e-30).rsqrt(), torch.nan)
    print(
        f"median E(B-V) {float(ebv.median()):.3f}; A_u = {float(a_band[:, 0].median()):.3f} mag"
    )

    perm = torch.randperm(n, generator=torch.Generator().manual_seed(1))
    n_train = int(TRAIN_FRACTION * n)
    train, test = perm[:n_train], perm[n_train:]
    passband = torch.arange(len(BANDS)).expand(n, -1)
    survey = torch.zeros_like(passband)
    scale = flux[train].median(dim=0).values.clamp_min(1e-3)

    encoder = PhotometryEncoder(
        num_passbands=len(BANDS), num_surveys=1, flux_scale=scale, latent_dim=LATENT
    ).to(dev)
    head = BinnedRedshiftHead(LATENT, Z_EDGES).to(dev)
    params = list(encoder.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=5e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    flux_d, error_d, z_d = flux.to(dev), error.to(dev), z_spec.to(dev)
    passband_d, survey_d = passband.to(dev), survey.to(dev)

    def predict(index: torch.Tensor) -> torch.Tensor:
        latent, supported = encoder(
            flux_d[index], error_d[index], passband_d[index], survey_d[index]
        )
        return head(latent, supported=supported)

    batch = 1024
    for epoch in range(EPOCHS):
        order = train[torch.randperm(n_train)]
        for start in range(0, n_train, batch):
            idx = order[start : start + batch].to(dev)
            loss = -head.log_prob(predict(idx), z_d[idx]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        if epoch in (0, EPOCHS // 2, EPOCHS - 1):
            print(f"epoch {epoch + 1:4d}: train -log p(z) = {float(loss.detach()):.3f}")

    with torch.no_grad():
        test_d = test.to(dev)
        logits = predict(test_d)
        log_density = head.log_prob(logits, z_d[test_d])
        prob = torch.softmax(logits, dim=-1).cpu()
    centres = 0.5 * (Z_EDGES[1:] + Z_EDGES[:-1])
    cdf = prob.cumsum(dim=-1)
    median_bin = (cdf < 0.5).sum(dim=-1).clamp_max(len(centres) - 1)
    # Median of the piecewise-uniform p(z): interpolate inside the median bin.
    lo = Z_EDGES[median_bin]
    width = Z_EDGES[1] - Z_EDGES[0]
    below = torch.where(
        median_bin > 0,
        cdf.gather(1, (median_bin - 1).clamp_min(0)[:, None]).squeeze(1),
        0.0,
    )
    mass = prob.gather(1, median_bin[:, None]).squeeze(1)
    z_phot = lo + width * ((0.5 - below) / mass.clamp_min(1e-12)).clamp(0.0, 1.0)

    ours = scores(z_phot, z_spec[test])
    has_ref = z_sdss[test] > -1.0  # the Photoz table flags failures with -9999
    ref = scores(z_sdss[test][has_ref], z_spec[test][has_ref])
    finite = torch.isfinite(log_density)
    print(f"test set: {test.numel()} galaxies (train {n_train})")
    print(
        f"torchsky encoder : sigma_NMAD {ours['nmad']:.4f}, outliers {100 * ours['outlier']:.1f}%, "
        f"bias {ours['bias']:+.4f}, mean log p(z_spec) {float(log_density[finite].mean()):.2f}"
    )
    print(
        f"SDSS DR17 Photoz: sigma_NMAD {ref['nmad']:.4f}, outliers {100 * ref['outlier']:.1f}%, "
        f"bias {ref['bias']:+.4f} on {int(has_ref.sum())} galaxies with an estimate (reference; trained on a much larger spectroscopic set)"
    )
    assert ours["nmad"] < 0.1 and ours["outlier"] < 0.2

    out = args.out or Path(tempfile.mkdtemp(prefix="torchsky-photoz-"))
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        1, 2, figsize=(9.5, 4.6), constrained_layout=True, sharey=True
    )
    for ax, est, sc, title in (
        (axes[0], z_phot, ours, "torchsky PhotometryEncoder + BinnedRedshiftHead"),
        (
            axes[1],
            z_sdss[test].clamp_min(-1.0),
            ref,
            "SDSS DR17 Photoz (Beck et al. 2016)",
        ),
    ):
        ax.scatter(z_spec[test], est, s=2, c="C0", alpha=0.5)
        line = torch.tensor([0.0, 1.0])
        ax.plot(line, line, "k-", lw=0.8)
        for sign in (1, -1):
            ax.plot(line, line + sign * OUTLIER * (1 + line), "k:", lw=0.8)
        ax.set(
            xlim=(0, 0.9),
            ylim=(0, 0.9),
            aspect="equal",
            xlabel="spectroscopic z",
            title=title,
        )
        ax.text(
            0.04,
            0.84,
            f"$\\sigma_{{NMAD}}$ = {sc['nmad']:.3f}\noutliers {100 * sc['outlier']:.1f}%",
            transform=ax.transAxes,
            fontsize=9,
        )
        ax.title.set_fontsize(9)
    axes[0].set_ylabel("photometric z")
    target = out / "sdss_photoz_encoder.png"
    fig.savefig(target, dpi=110)
    print(f"figure: {target}; device={dev}")


if __name__ == "__main__":
    main()
