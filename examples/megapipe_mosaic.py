"""Mosaic four overlapping CFHT MegaPipe cutouts with Lanczos3 and drizzle.

Science question: do independent, differently exposed stacks of the same sky
agree where they overlap once they are resampled onto one grid? Four 512 x 512
px cutouts (g band, 0.186 arcsec/px) come from the CADC cutout service, all
around RA 358.82, Dec -7.25 where three MegaPipe tiles meet: 712.165 and
711.165 (one row, 240 s each), and two windows of tile 712.166 (the next row,
1224 s). The rows of tiles are staggered, and every tile has its own TAN
reference point, so the resampling must account for four different
projections. ``plan`` builds one common TAN grid, each cutout is resampled
alone with ``coadd`` (Lanczos3 and flux-conserving ``drizzle_square``), and the
pairwise overlaps between different tiles are compared: median offset, and the
rms difference against the noise two independent images would show. Adjacent
tiles are built from the same input exposures, so the overlap noise is shared
and the ratio is far below one; what the test checks is registration and
zero point (a misregistration of a fraction of a pixel would raise the ratio
towards or above one). Thefour cutouts are then combined with inverse-variance weights.

Data: CFHT/MegaCam MegaPipe stacks (Gwyn 2008, PASP 120, 212) served by the
CADC; open data, cite CFHT and CADC. Runtime about 10 s on CPU once the four
1 MB cutouts are cached.

    pixi run -e test -- python examples/megapipe_mosaic.py [--device cuda] [--out DIR]
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from _sample_data import fetch_sample, skip_if_unavailable

import torchfits
from torchsky.map import Observation, coadd, plan

TILES = ("712.165", "711.165", "712.166a", "712.166b")
SAMPLES = [f"megapipe_mosaic_{k}" for k in "abcd"]
KERNELS = {"lanczos3": {}, "drizzle_square": {"pixfrac": 1.0}}


def pixel_noise(image: torch.Tensor) -> float:
    """Robust per-pixel noise from adjacent-pixel differences (MAD)."""
    d = (image[:, 1:] - image[:, :-1]).flatten()
    return float(1.4826 * (d - d.median()).abs().median() / 2**0.5)


def robust_rms(x: torch.Tensor) -> float:
    return float(1.4826 * (x - x.median()).abs().median())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    dev = args.device

    with skip_if_unavailable():
        paths = [fetch_sample(name) for name in SAMPLES]
    images, headers = [], []
    for path in paths:
        images.append(torchfits.read(str(path), 0).to(torch.float32))
        headers.append(dict(torchfits.read_header(str(path), 0)))
    sigma = [pixel_noise(im) for im in images]
    for tile, h, s in zip(TILES, headers, sigma):
        print(
            f"tile {tile}: CRVAL ({h['CRVAL1']:.3f}, {h['CRVAL2']:.3f}), exposure {h['EXPTIME']:.0f} s, noise {s:.2f} counts/px"
        )

    def make(i: int, **kw) -> Observation:
        ones = torch.ones_like(images[i])
        return Observation(
            data=images[i].to(dev),
            wcs=headers[i],
            weight_map=(ones / sigma[i] ** 2).to(dev),
            weight_is_inverse_variance=True,
            **kw,
        )

    obs = [make(i) for i in range(4)]
    target = plan(obs, projection="TAN", device=dev)
    print(f"common TAN grid: {tuple(target.shape)} px")

    products = {}
    for kernel, extra in KERNELS.items():
        single = [
            coadd(
                [o],
                target,
                kernel=kernel,
                semantics="surface_brightness",
                device=dev,
                **extra,
            )
            for o in obs
        ]
        data = [r.data.cpu() for r in single]
        cover = [r.weight.cpu() > 0 for r in single]
        print(f"{kernel}: pairwise overlaps (interior, 8 px eroded)")
        pair_ratio = []
        for i, j in combinations(range(4), 2):
            if {i, j} == {2, 3}:
                continue  # two windows of one stack: identical pixels, no information
            both = cover[i] & cover[j]
            both[:8] = both[-8:] = False
            both[:, :8] = both[:, -8:] = False
            n = int(both.sum())
            if n < 5000:
                continue
            diff = (data[i] - data[j])[both]
            expected = (sigma[i] ** 2 + sigma[j] ** 2) ** 0.5
            ratio = robust_rms(diff) / expected
            pair_ratio.append(ratio)
            print(
                f"  {TILES[i]} - {TILES[j]}: {n:6d} px, median offset {float(diff.median()):+.3f}, rms {robust_rms(diff):.2f} (noise-only {expected:.2f}, ratio {ratio:.2f})"
            )
        full = coadd(
            obs,
            target,
            kernel=kernel,
            semantics="surface_brightness",
            method="weighted_mean",
            device=dev,
            **extra,
        )
        products[kernel] = full
        w = full.weight.cpu()
        n_obs = sum((r.weight.cpu() > 0).to(torch.int32) for r in single)
        print(
            f"  coadd: {int((w > 0).sum())} covered px; pixels seen by 1/2/3/4 cutouts: "
            + "/".join(str(int((n_obs == k).sum())) for k in (1, 2, 3, 4))
        )
        assert pair_ratio and max(pair_ratio) < 3.0

    lan, dri = (products[k].data.cpu() for k in KERNELS)
    both = (products["lanczos3"].weight.cpu() > 0) & (
        products["drizzle_square"].weight.cpu() > 0
    )
    corr = float(torch.corrcoef(torch.stack([lan[both], dri[both]]))[0, 1])
    print(
        f"Lanczos3 vs drizzle coadd: pixel correlation {corr:.4f} over {int(both.sum())} px"
    )

    out = args.out or Path(tempfile.mkdtemp(prefix="torchsky-megapipe-"))
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9), constrained_layout=True)
    noise = pixel_noise(torch.where(both, lan, 0.0)[100:600, 100:700])
    lo, hi = -2.0 * noise, 12.0 * noise
    for ax, (title, im, cover) in zip(
        axes[:2],
        (
            ("Lanczos3 coadd", lan, products["lanczos3"].weight),
            ("drizzle coadd", dri, products["drizzle_square"].weight),
        ),
    ):
        ax.imshow(
            torch.where(cover.cpu() > 0, im, torch.nan),
            origin="lower",
            cmap="gray_r",
            vmin=lo,
            vmax=hi,
        )
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    w = products["lanczos3"].weight.cpu()
    im = axes[2].imshow(
        torch.where(w > 0, w, torch.nan), origin="lower", cmap="viridis"
    )
    axes[2].set_title("inverse-variance weight", fontsize=9)
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], shrink=0.8)
    fig.savefig(out / "megapipe_mosaic.png", dpi=100)
    print(f"figure: {out / 'megapipe_mosaic.png'}; device={dev}")


if __name__ == "__main__":
    main()
