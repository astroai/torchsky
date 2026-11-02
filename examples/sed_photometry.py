"""Redshift, dust extinction, and synthetic photometry of a galaxy SED.

Places a rest-frame galaxy-like SED (4000 A break) at redshift z using
``torchsky.cosmology`` luminosity distances, reddens it with a CCM89 dust law
from ``torchsky.extinction``, and integrates it through g/r/i/z-like passbands
with ``torchsky.spectrophotometry`` to get AB magnitudes. Validates the
pipeline against the closed-form flat-f_nu result m = M + DM - 2.5 log10(1+z)
and the dust law at the band pivot, then shows the g-r colour track as the
break redshifts through the bands. Runtime about 2 s on CPU.

    pixi run -e test -- python examples/sed_photometry.py [--device cuda] [--out DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from torchsky.cosmology import angular_diameter_distance, distance_modulus, luminosity_distance
from torchsky.extinction import a_lambda, apply_extinction
from torchsky.spectrophotometry import Passband, synthetic_photometry_bands

C_ANGSTROM_S = 2.99792458e18
ABS_MAG = -21.0
PC_CM = 3.0856775814913673e18
D_L_10PC_MPC = 1.0e-5


def make_bands(dev: str) -> list[Passband]:
    spec = {"g": (3900, 4300, 5400, 5700), "r": (5400, 5700, 6600, 6900),
            "i": (6700, 7000, 7900, 8200), "z": (8000, 8300, 9100, 9400)}
    f64 = dict(dtype=torch.float64, device=dev)
    return [Passband.from_samples(torch.tensor(w, **f64), torch.tensor([0.0, 1.0, 1.0, 0.0], **f64), name=n)
            for n, w in spec.items()]


def rest_sed(lam: torch.Tensor, *, flat: bool) -> torch.Tensor:
    """F_lambda at 10 pc [erg/s/cm^2/A] for absolute AB magnitude ABS_MAG at 5000 A."""
    f_nu = 10 ** (-0.4 * (ABS_MAG + 48.6))
    shape = torch.ones_like(lam) if flat else (0.45 + 0.55 * torch.sigmoid((lam - 4000.0) / 60.0)) * (lam / 5000.0) ** 0.3
    return f_nu * shape * C_ANGSTROM_S / lam**2


def observe(lam_obs, z: float, *, flat: bool, ebv: float, dev: str):
    zt = torch.tensor(z, dtype=torch.float64, device=dev)
    dl = luminosity_distance(zt)
    flux = rest_sed(lam_obs / (1 + z), flat=flat) / (1 + z) * (D_L_10PC_MPC / dl) ** 2
    if ebv > 0:
        flux = apply_extinction(flux, lam_obs, torch.tensor(ebv, dtype=torch.float64, device=dev))
    return flux


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    dev = args.device
    out = args.out or Path(tempfile.mkdtemp(prefix="torchsky-sed-"))
    out.mkdir(parents=True, exist_ok=True)
    lam = torch.linspace(3000.0, 10000.0, 3000, dtype=torch.float64, device=dev)
    bands = make_bands(dev)

    # Cosmology.
    z = torch.tensor([0.1, 0.5, 1.0, 2.0], dtype=torch.float64, device=dev)
    dl, da, dm = luminosity_distance(z), angular_diameter_distance(z), distance_modulus(z)
    assert torch.allclose(dl, da * (1 + z) ** 2)
    print("z      D_L [Mpc]  D_A [Mpc]  DM [mag]")
    for i in range(len(z)):
        print(f"{float(z[i]):4.1f} {float(dl[i]):10.1f} {float(da[i]):10.1f} {float(dm[i]):9.3f}")

    # Closed-form check with a flat-f_nu SED, no dust.
    zz = 0.5
    mags = synthetic_photometry_bands(lam, observe(lam, zz, flat=True, ebv=0.0, dev=dev), bands, output="ab_mag").value
    expect = ABS_MAG + float(distance_modulus(torch.tensor(zz, dtype=torch.float64, device=dev))) - 2.5 * torch.log10(torch.tensor(1 + zz)).item()
    err = float((mags - expect).abs().max())
    print(f"flat-f_nu m(z={zz}) = {[round(float(v), 3) for v in mags]} vs closed form {expect:.3f}: max err {err:.4f} mag")
    assert err < 0.02

    # Dust: reddening of the r band vs the CCM89 curve at its pivot.
    ebv = 0.1
    clean = synthetic_photometry_bands(lam, observe(lam, 0.3, flat=False, ebv=0.0, dev=dev), bands, output="ab_mag").value
    dusty = synthetic_photometry_bands(lam, observe(lam, 0.3, flat=False, ebv=ebv, dev=dev), bands, output="ab_mag").value
    dm_band = (dusty - clean).cpu()
    a_pivot = a_lambda(torch.tensor([b.pivot_wavelength.item() for b in bands], dtype=torch.float64), ebv=ebv).cpu()
    print("band dust dimming [mag]:", {b.name: round(float(v), 3) for b, v in zip(bands, dm_band)},
          " CCM89 A_lambda at pivot:", [round(float(v), 3) for v in a_pivot])
    assert float((dm_band - a_pivot).abs().max()) < 0.03

    # g-r colour track through the 4000 A break.
    zs = torch.linspace(0.02, 1.2, 25, dtype=torch.float64)
    colours = []
    for zi in zs:
        m = synthetic_photometry_bands(lam, observe(lam, float(zi), flat=False, ebv=0.0, dev=dev), bands, output="ab_mag").value
        colours.append((float(m[0] - m[1]), float(m[1] - m[2])))
    colours = torch.tensor(colours)
    print(f"g-r: {colours[0, 0]:.2f} at z=0.02, peaks at {colours[:, 0].max():.2f} (z={float(zs[colours[:, 0].argmax()]):.2f}) as the 4000 A break crosses g and r")
    assert colours[:, 0].max() > colours[0, 0] + 0.3

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), constrained_layout=True)
    for zi, c in [(0.0, "#4c78a8"), (0.5, "#f58518")]:
        f = observe(lam, zi, flat=False, ebv=0.0, dev=dev).cpu()
        axes[0].plot(lam.cpu(), f / f.max(), c=c, label=f"z={zi}")
    for b in bands:
        axes[0].fill_between(b.wavelength.cpu(), 0, b.throughput.cpu() * 0.3, alpha=0.25, lw=0)
    axes[0].set(xlabel="observed wavelength [A]", ylabel="relative F_lambda", title="SED and passbands")
    axes[0].legend(fontsize=8)
    axes[1].plot(zs, colours[:, 0], label="g-r", c="#4c78a8")
    axes[1].plot(zs, colours[:, 1], label="r-i", c="#f58518")
    axes[1].set(xlabel="redshift", ylabel="AB colour", title="colour track")
    axes[1].legend(fontsize=8)
    fig.savefig(out / "sed_photometry.png", dpi=100)
    print(f"device={dev}  figure: {out / 'sed_photometry.png'}")


if __name__ == "__main__":
    main()
