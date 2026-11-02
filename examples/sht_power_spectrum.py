"""Spherical harmonic transforms: synthesis, power spectrum recovery, smoothing.

Draws a Gaussian random sky from a power-law C_ell with ``synfast``, measures
it back with ``anafast`` (compared with the input spectrum and the cosmic
variance expectation), then smooths the map with a Gaussian beam in harmonic
space. Runtime about 3 s on CPU (NSIDE=128).

    pixi run -e test -- python examples/sht_power_spectrum.py [--device cuda] [--out DIR]
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    dev = args.device
    out = args.out or Path(tempfile.mkdtemp(prefix="torchsky-sht-"))
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    nside, lmax = 128, 256
    ell = torch.arange(lmax + 1, dtype=torch.float64, device=dev)
    cl_in = 1.0 / (ell.clamp(min=1) * (ell.clamp(min=1) + 1.0)) ** 1.1
    cl_in[0] = 0.0

    sky = sphere.synfast(cl_in, nside=nside, lmax=lmax)
    cl_out = sphere.anafast(sky, lmax=lmax)
    # Per-multipole ratio is chi^2 distributed with 2l+1 degrees of freedom:
    # average over bands and compare against the expected sqrt(2/(2l+1)) scatter.
    lo, hi = 20, 200
    ratio = (cl_out[lo:hi] / cl_in[lo:hi])
    band_mean = float(ratio.mean())
    expected_sigma = float((2.0 / (2 * ell[lo:hi] + 1)).sum().sqrt() / (hi - lo))
    assert abs(band_mean - 1.0) < 5 * expected_sigma + 0.02, band_mean
    print(f"NSIDE={nside} lmax={lmax}: mean C_l(out)/C_l(in) over l={lo}..{hi} = "
          f"{band_mean:.3f} (cosmic-variance sigma {expected_sigma:.3f})")

    fwhm = torch.deg2rad(torch.tensor(2.0)).item()
    bl = sphere.gaussian_beam(fwhm, lmax=lmax).to(dev)
    alm = sphere.map2alm(sky, lmax=lmax)
    smooth = sphere.alm2map(sphere.almxfl(alm, bl), nside=nside, lmax=lmax)
    cl_s = sphere.anafast(smooth, lmax=lmax)
    ratio_s = float((cl_s[lo:hi] / (cl_out[lo:hi] * bl[lo:hi] ** 2)).mean())
    print(f"map std {float(sky.std()):.3e} -> {float(smooth.std()):.3e} after 2 deg beam; "
          f"smoothed/(C_l b_l^2) = {ratio_s:.3f}")
    assert float(smooth.std()) < float(sky.std()) and abs(ratio_s - 1.0) < 0.1

    fig, ax = plt.subplots(figsize=(5.2, 3.6), constrained_layout=True)
    x = ell.cpu()[2:]
    ax.loglog(x, (cl_in * ell * (ell + 1))[2:].cpu(), "k-", lw=1.5, label="input $C_\\ell$")
    ax.loglog(x, (cl_out * ell * (ell + 1))[2:].cpu(), ".", ms=3, c="#4c78a8", label="anafast(synfast)")
    ax.loglog(x, (cl_s * ell * (ell + 1))[2:].cpu(), ".", ms=3, c="#f58518", label="after 2° beam")
    ax.set(xlabel=r"multipole $\ell$", ylabel=r"$\ell(\ell+1)C_\ell$", title="Power spectrum recovery")
    ax.legend(fontsize=8)
    fig.savefig(out / "sht_power_spectrum.png", dpi=100)
    print(f"device={dev}  figure: {out / 'sht_power_spectrum.png'}")


if __name__ == "__main__":
    main()
