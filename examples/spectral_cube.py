"""Build a spectral cube from IFU-like detector frames with ``coadd_cube``.

Sixteen synthetic IFU exposures (16 slices x 40 wavelength rows each) step
across a field. Every detector pixel carries its own sky position and
wavelength through an ``IFUSliceWCS``. A galaxy-like Gaussian emission-line
source (line at 5040 A on a flat continuum) is sampled by the detectors and
resampled onto an explicit ``CubeTarget`` (16x16 spatial grid, 4 A spectral
bins) with the spectral ``drizzle`` kernel. The cube is checked for the line
position, continuum level and spatial centroid. Runtime about 2 s on CPU.

    pixi run -e test -- python examples/spectral_cube.py [--device cuda] [--out DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from torchsky.map import CubeTarget, Observation, ResampleConfig, coadd_cube
from torchsky.wcs import WCS, IFUSliceWCS

N_SLICE, N_WAVE, N_EXP = 16, 40, 16
PIX = 0.001  # deg
RA0, DEC0 = 150.0, 2.0
CONT, LINE, LAM0, SIGMA_L = 1.0, 8.0, 5040.0, 3.0
SRC_X, SRC_Y, SIGMA_S = 6.0, 9.0, 2.2  # cube pixel coordinates (0-based)

HEADER = {
    "NAXIS": 2, "NAXIS1": N_SLICE, "NAXIS2": N_EXP, "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
    "CRPIX1": (N_SLICE + 1) / 2, "CRPIX2": (N_EXP + 1) / 2, "CRVAL1": RA0, "CRVAL2": DEC0,
    "CDELT1": -PIX, "CDELT2": PIX,
}


def cube_truth(x: torch.Tensor, y: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    spatial = torch.exp(-((x - SRC_X) ** 2 + (y - SRC_Y) ** 2) / (2 * SIGMA_S**2))
    return CONT + LINE * spatial * torch.exp(-0.5 * ((lam - LAM0) / SIGMA_L) ** 2)


def frame(row: int) -> Observation:
    """One IFU exposure that scans cube row ``row`` (constant Dec)."""
    cols = torch.arange(N_SLICE, dtype=torch.float64)
    waves = torch.arange(N_WAVE, dtype=torch.float64)
    x = cols.unsqueeze(0).expand(N_WAVE, N_SLICE)
    lam = (5000.0 + 2.0 * waves).unsqueeze(1).expand(N_WAVE, N_SLICE)
    # Increasing RA is toward -x in the cube header (CDELT1 < 0).
    ra = RA0 - (x - (N_SLICE - 1) / 2) * PIX
    dec = torch.full((N_WAVE, N_SLICE), DEC0 + (row - (N_EXP - 1) / 2) * PIX, dtype=torch.float64)
    ifu = IFUSliceWCS(ra_map=ra.clone(), dec_map=dec, wavelength_map=lam.clone(),
                      slice_id_map=cols.to(torch.int64).unsqueeze(0).expand(N_WAVE, N_SLICE).clone())
    data = cube_truth(x, torch.full_like(x, float(row)), lam).float()
    return Observation(data=data, ifu_wcs=ifu, weight=torch.ones_like(data), weight_is_inverse_variance=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or Path(tempfile.mkdtemp(prefix="torchsky-cube-"))
    out.mkdir(parents=True, exist_ok=True)

    edges = torch.arange(5000.0, 5080.1, 4.0, dtype=torch.float64)
    target = CubeTarget(spatial_wcs=WCS(HEADER), spectral_bin_edges=edges)
    cube = coadd_cube([frame(r) for r in range(N_EXP)], target,
                      resample=ResampleConfig(spectral_kernel="drizzle", spectral_pixfrac=0.8),
                      device=args.device)
    data = cube.data.cpu()
    centres = 0.5 * (edges[:-1] + edges[1:])
    print(f"cube shape {tuple(data.shape)} (wave, y, x); spectral range "
          f"{float(centres[0]):.0f}-{float(centres[-1]):.0f} A")

    spec = data[:, int(SRC_Y), int(SRC_X)]
    peak_lam = float(centres[int(spec.argmax())])
    cont = float(data[0][torch.isfinite(data[0])].median())
    band = (centres > LAM0 - 12) & (centres < LAM0 + 12)
    mom0 = (data[band] - cont).sum(dim=0)
    yy, xx = torch.meshgrid(torch.arange(N_EXP, dtype=torch.float64), torch.arange(N_SLICE, dtype=torch.float64), indexing="ij")
    cx = float((mom0 * xx).sum() / mom0.sum())
    cy = float((mom0 * yy).sum() / mom0.sum())
    print(f"line peak at {peak_lam:.0f} A (truth {LAM0:.0f}); continuum {cont:.3f} (truth {CONT}); "
          f"moment-0 centroid ({cx:.2f}, {cy:.2f}) (truth {SRC_X}, {SRC_Y})")
    assert abs(peak_lam - LAM0) <= 2.0 and abs(cont - CONT) < 0.05
    assert abs(cx - SRC_X) < 0.2 and abs(cy - SRC_Y) < 0.2

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2), constrained_layout=True)
    axes[0].imshow(mom0, origin="lower", cmap="magma")
    axes[0].set_title("line moment 0", fontsize=9)
    axes[1].step(centres, spec, where="mid", c="#4c78a8")
    axes[1].set(xlabel="wavelength [A]", ylabel="flux", title="spectrum at source")
    fig.savefig(out / "spectral_cube.png", dpi=100)
    print(f"device={args.device}  figure: {out / 'spectral_cube.png'}")


if __name__ == "__main__":
    main()
