"""Rest frame, dust correction and synthetic ugriz photometry of an SDSS spectrum.

Science question: does synthetic photometry computed with torchsky from the
calibrated SDSS spectrum of a low-redshift broad-line quasar reproduce the
magnitudes that the SDSS pipeline derives from the same spectrum, and how much
do Galactic dust and the redshift change the colours? The DR16 spectrum of
plate 0751, MJD 52251, fibre 160 (class QSO, z = 0.0456) is read with
torchfits. Wavelengths are shifted to the rest frame with
``torchsky.redshift.shift_wavelength``; the narrow [O III] lines give an
independent velocity check of the pipeline redshift. The spectrum is corrected
for Galactic reddening with the F99 law (E(B-V) = 0.0698 mag from the
Schlafly & Finkbeiner 2011 map at the object position, IRSA DUST service) and
integrated through the SDSS u, g, r, i and z filter curves from the SVO Filter
Profile Service (loaded with ``filter_profiles.bandpass_from_transmission_file``)
with ``synthetic_photometry_bands``. The spectrum covers 3800-9200 A, so u (12
per cent of the response) and z (69 per cent) fall below the 90 per cent
coverage requirement and are reported as invalid instead of extrapolated; g, r
and i agree with the SDSS synthetic magnitudes to 0.03 mag.

Data: SDSS DR16 spectrum (SDSS Collaboration, public; cite SDSS DR16, Ahumada
et al. 2020) and SVO filter profiles (Rodrigo et al. 2012). Runtime about
2 s on CPU once the files are cached.

    pixi run -e test -- python examples/sdss_spectrum_photometry.py [--device cuda] [--out DIR]
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
from torchsky.extinction import a_lambda
from torchsky.filter_profiles import bandpass_from_transmission_file
from torchsky.redshift import shift_wavelength, velocity_kms_from_redshift
from torchsky.spectrophotometry import Passband, synthetic_photometry_bands

C_KMS = 299792.458
FLUX_UNIT = 1e-17  # erg s^-1 cm^-2 A^-1
EBV = 0.0698  # Schlafly & Finkbeiner (2011) at the plug position, via IRSA DUST
BANDS = "ugriz"
REST_LINES = {"[OIII]4960": 4960.2949, "[OIII]5008": 5008.2396}


def line_centroid(lam, flux, valid, rest, half_width=8.0):
    """Continuum-subtracted flux-weighted centroid in a +-8 A window (rest frame)."""
    core = valid & ((lam - rest).abs() < half_width)
    side = valid & ((lam - rest).abs() > 2 * half_width) & ((lam - rest).abs() < 5 * half_width)
    cont = flux[side].median()
    w = (flux[core] - cont).clamp(min=0)
    return float((lam[core] * w).sum() / w.sum())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    dev = args.device

    with skip_if_unavailable():
        path = fetch_sample("sdss_spectrum")
        filters = {b: fetch_sample(f"svo_sdss_{b}") for b in BANDS}
    coadd = torchfits.read(str(path), hdu=1)
    info = torchfits.read(str(path), hdu=2)
    z = float(info["Z"][0])
    cls = bytes(info["CLASS"][0].tolist()).decode().strip()
    nmgy = info["SPECTROSYNFLUX"][0].to(torch.float64)
    lam = (10.0 ** coadd["loglam"].to(torch.float64)).to(dev)
    flux = (coadd["flux"].to(torch.float64) * FLUX_UNIT).to(dev)
    valid = (coadd["ivar"] > 0).to(dev)
    print(
        f"plate 751 MJD 52251 fibre 160: class {cls}, z = {z:.5f} "
        f"(v = {float(velocity_kms_from_redshift(torch.tensor(z, dtype=torch.float64))):.0f} km/s), "
        f"{int(valid.sum())}/{lam.numel()} good pixels, {float(lam[valid].min()):.0f}-{float(lam[valid].max()):.0f} A"
    )

    # Rest frame and a velocity check against the narrow [O III] lines.
    lam_rest = shift_wavelength(lam, 1.0 / (1.0 + z))
    offsets = []
    for name, rest in REST_LINES.items():
        centre = line_centroid(lam_rest, flux / FLUX_UNIT, valid, rest)
        v = C_KMS * (centre / rest - 1.0)
        offsets.append(v)
        print(f"{name}: rest-frame centroid {centre:.2f} A -> {v:+.0f} km/s from the pipeline redshift")

    # Dust correction and synthetic AB photometry through the SVO filters.
    a_lam = a_lambda(lam, ebv=EBV, law="f99")
    flux_dered = flux * 10.0 ** (0.4 * a_lam)
    passbands = []
    for b in BANDS:
        spec = bandpass_from_transmission_file(filters[b], band_name=b)
        passbands.append(
            Passband.from_samples(
                spec["transmission_wavelength"].to(torch.float64),
                spec["transmission_throughput"].to(torch.float64),
                name=b,
            ).to(dev)
        )
    kw = dict(output="ab_mag", min_coverage=0.9, mask=valid)
    mags = synthetic_photometry_bands(lam, flux, passbands, **kw)
    mags_dered = synthetic_photometry_bands(lam, flux_dered, passbands, **kw)
    cat = 22.5 - 2.5 * torch.log10(nmgy)
    print("band  coverage  torchsky AB  SDSS synthetic  diff    dust-corrected  A_lambda(pivot)")
    diffs = []
    for k, b in enumerate(BANDS):
        cov = float(mags.coverage[k])
        a_piv = float(a_lambda(passbands[k].pivot_wavelength.reshape(1), ebv=EBV, law="f99")[0])
        if bool(mags.valid[k]):
            m, mc = float(mags.value[k]), float(mags_dered.value[k])
            diffs.append(m - float(cat[k]))
            print(f"  {b}   {cov:6.2f}   {m:10.3f}  {float(cat[k]):13.3f}  {m - float(cat[k]):+.3f}   {mc:13.3f}  {a_piv:12.3f}")
        else:
            print(f"  {b}   {cov:6.2f}   {'invalid':>10s}  {float(cat[k]):13.3f}")
    rms = float(torch.tensor(diffs).pow(2).mean().sqrt())
    colour = float(mags_dered.value[1] - mags_dered.value[2])
    print(f"rms difference to the SDSS synthetic magnitudes (griz): {rms:.3f} mag; dust-corrected g-r = {colour:.3f}")
    assert rms < 0.1

    out = args.out or Path(tempfile.mkdtemp(prefix="torchsky-sdss-spectrum-"))
    out.mkdir(parents=True, exist_ok=True)
    fig, (ax, bx) = plt.subplots(
        2, 1, figsize=(8, 5.4), sharex=False, constrained_layout=True,
        gridspec_kw={"height_ratios": [3, 1.3]},
    )
    ok = valid.cpu()
    ax.plot(lam_rest.cpu()[ok], (flux_dered.cpu() / FLUX_UNIT)[ok], lw=0.5, c="C0", label="dust-corrected")
    ax.plot(lam_rest.cpu()[ok], (flux.cpu() / FLUX_UNIT)[ok], lw=0.5, c="0.5", label="observed")
    for name, rest in REST_LINES.items():
        ax.axvline(rest, c="C3", lw=0.5, ls=":")
    ax.set(xlabel="rest wavelength [A]", ylabel=r"$F_\lambda$ [1e-17 erg/s/cm$^2$/A]",
           title=f"SDSS 0751-52251-0160, {cls}, z = {z:.4f}")
    ax.legend(fontsize=8)
    for k, b in enumerate(BANDS):
        w = passbands[k].wavelength.cpu() / (1 + z)
        bx.fill_between(w, passbands[k].throughput.cpu(), alpha=0.5, label=b)
    bx.set(xlabel="rest wavelength [A]", ylabel="throughput", xlim=ax.get_xlim())
    bx.legend(ncol=5, fontsize=8)
    fig.savefig(out / "sdss_spectrum_photometry.png", dpi=110)
    print(f"figure: {out / 'sdss_spectrum_photometry.png'}; device={dev}")


if __name__ == "__main__":
    main()
