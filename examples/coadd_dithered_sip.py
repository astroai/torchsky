"""Coadd dithered, rotated TAN-SIP exposures with Lanczos3 and drizzle.

Eight noisy exposures of one synthetic star field are rendered through
distinct WCS solutions (sub-pixel dithers, small rotations, a shared SIP
distortion). ``plan`` builds a common target grid, ``coadd_image_plane``
combines them with the ``lanczos3`` interpolation kernel and the flux-conserving
``drizzle_square`` kernel, and both products are checked for photometry
(aperture flux of bright stars vs truth), astrometry (centroid vs the true
sky position mapped through the target WCS) and noise reduction.
Runtime about 3 s on CPU.

    pixi run -e test -- python examples/coadd_dithered_sip.py [--device cuda] [--out DIR]
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

from torchsky.map import ImageObservation, coadd_image_plane, plan
from torchsky.wcs import WCS

SIZE, SIGMA_PSF, SKY_NOISE = 96, 1.6, 0.02


def header(dx: float, dy: float, theta_deg: float) -> dict:
    c, s = math.cos(math.radians(theta_deg)), math.sin(math.radians(theta_deg))
    scale = 0.2 / 3600.0
    return {
        "NAXIS": 2,
        "NAXIS1": SIZE,
        "NAXIS2": SIZE,
        "CTYPE1": "RA---TAN-SIP",
        "CTYPE2": "DEC--TAN-SIP",
        "CRPIX1": SIZE / 2 + 0.5 + dx,
        "CRPIX2": SIZE / 2 + 0.5 + dy,
        "CRVAL1": 150.0,
        "CRVAL2": 2.0,
        "CUNIT1": "deg",
        "CUNIT2": "deg",
        "CD1_1": -scale * c,
        "CD1_2": scale * s,
        "CD2_1": scale * s,
        "CD2_2": scale * c,
        "A_ORDER": 2,
        "B_ORDER": 2,
        "A_2_0": 2e-5,
        "A_1_1": -1e-5,
        "A_0_2": 1e-5,
        "B_2_0": -1e-5,
        "B_1_1": 1.5e-5,
        "B_0_2": 2e-5,
    }


def render(wcs: WCS, ra, dec, flux, noise: torch.Generator) -> torch.Tensor:
    """Render Gaussian PSFs at the pixel positions of the sources in ``wcs``."""
    pos = wcs.world_to_pixel(torch.stack([ra, dec], dim=-1), origin=0)
    yy, xx = torch.meshgrid(
        *[torch.arange(SIZE, dtype=torch.float64)] * 2, indexing="ij"
    )
    img = torch.zeros(SIZE, SIZE, dtype=torch.float64)
    for (x0, y0), f in zip(pos, flux):
        img += (
            f
            / (2 * math.pi * SIGMA_PSF**2)
            * torch.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * SIGMA_PSF**2))
        )
    return (
        img + SKY_NOISE * torch.randn(SIZE, SIZE, generator=noise, dtype=torch.float64)
    ).float()


def aperture_flux(img: torch.Tensor, x: float, y: float, r: float = 7.0) -> float:
    yy, xx = torch.meshgrid(
        *[torch.arange(n, dtype=torch.float64) for n in img.shape], indexing="ij"
    )
    m = ((xx - x) ** 2 + (yy - y) ** 2) <= r * r
    return float(img.double()[m].sum())


def centroid(
    img: torch.Tensor, x: float, y: float, r: float = 4.0
) -> tuple[float, float]:
    yy, xx = torch.meshgrid(
        *[torch.arange(n, dtype=torch.float64) for n in img.shape], indexing="ij"
    )
    m = ((xx - x) ** 2 + (yy - y) ** 2) <= r * r
    w = img.double()[m].clamp(min=0)
    return float((w * xx[m]).sum() / w.sum()), float((w * yy[m]).sum() / w.sum())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or Path(tempfile.mkdtemp(prefix="torchsky-coadd-sip-"))
    out.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(3)

    # Stars on a jittered 4x3 grid (~16 pix spacing) so that none blend.
    n_star = 12
    gi, gj = torch.meshgrid(
        torch.arange(4.0) - 1.5, torch.arange(3.0) - 1.0, indexing="ij"
    )
    jit = (torch.rand(2, n_star, generator=g, dtype=torch.float64) - 0.5) * 2e-4
    ra = 150.0 + gi.ravel().double() * 9e-4 + jit[0]
    dec = 2.0 + gj.ravel().double() * 9e-4 + jit[1]
    flux = 20.0 + 80.0 * torch.rand(n_star, generator=g, dtype=torch.float64)

    dithers = [
        (0.0, 0.0, 0.0),
        (0.37, -0.52, 0.4),
        (-0.61, 0.28, -0.3),
        (0.83, 0.71, 0.8),
        (-0.24, -0.88, -0.6),
        (0.52, 0.15, 0.2),
        (-0.77, 0.47, 1.0),
        (0.11, 0.93, -0.9),
    ]
    observations, singles = [], []
    for dx, dy, theta in dithers:
        h = header(dx, dy, theta)
        img = render(WCS(h), ra, dec, flux, g)
        singles.append(img)
        observations.append(
            ImageObservation(
                data=img, wcs=h, weight=1.0, weight_map=torch.ones_like(img)
            )
        )
    target = plan(observations, projection="TAN")
    target_header = dict(target.header)
    target_wcs = WCS(target_header)
    truth_xy = target_wcs.world_to_pixel(torch.stack([ra, dec], dim=-1), origin=0)

    # Bright, well-separated, interior stars only.
    h_t, w_t = target.shape
    sel = [
        i
        for i in range(n_star)
        if 12 < truth_xy[i, 0] < w_t - 12 and 12 < truth_xy[i, 1] < h_t - 12
    ]
    products = {}
    for name, kernel, semantics in [
        ("lanczos3", "lanczos3", "surface_brightness"),
        ("drizzle", "drizzle_square", "flux"),
    ]:
        res = coadd_image_plane(
            observations,
            target_header=target_header,
            resampling_kernel=kernel,
            value_semantics=semantics,
            stacking_method="weighted_mean",
            drizzle_pixfrac=0.8,
            device=args.device,
        )
        products[name] = res.data.cpu()
        ferr, aerr = [], []
        for i in sel:
            x, y = float(truth_xy[i, 0]), float(truth_xy[i, 1])
            ferr.append(aperture_flux(res.data.cpu(), x, y) / float(flux[i]) - 1.0)
            cx, cy = centroid(res.data.cpu(), x, y)
            aerr.append(math.hypot(cx - x, cy - y))
        med_f, max_a = float(torch.tensor(ferr).abs().median()), max(aerr)
        print(
            f"{name:9s}: shape {tuple(res.data.shape)}, {len(sel)} stars: median |dF/F| = "
            f"{med_f:.3f}, max centroid error = {max_a:.3f} pix"
        )
        assert med_f < 0.05 and max_a < 0.3, (name, med_f, max_a)

    corner = products["lanczos3"][:20, :20]
    noise_single = float(singles[0][:20, :20].std())
    print(
        f"background rms: single exposure {noise_single:.4f} -> coadd {float(corner.std()):.4f} "
        f"(expect ~{noise_single / math.sqrt(len(dithers)):.4f} before correlated-noise inflation)"
    )

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), constrained_layout=True)
    for ax, (title, im) in zip(
        axes,
        [
            ("single exposure", singles[0]),
            ("lanczos3 coadd", products["lanczos3"]),
            ("drizzle coadd", products["drizzle"]),
        ],
    ):
        ax.imshow(
            im,
            origin="lower",
            cmap="magma",
            vmin=-0.05,
            vmax=0.6 * float(products["lanczos3"].max()),
        )
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.savefig(out / "coadd_dithered_sip.png", dpi=100)
    print(f"device={args.device}  figure: {out / 'coadd_dithered_sip.png'}")


if __name__ == "__main__":
    main()
