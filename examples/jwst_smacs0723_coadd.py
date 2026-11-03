"""Coadd four dithered JWST NIRCam exposures of SMACS 0723 with Lanczos3 and drizzle.

Science question: does a coadd of calibrated JWST exposures conserve the flux
of the sources it resamples, does its uncertainty map predict the noise that is
actually there, and does it agree with the STScI pipeline mosaic? Four F444W
exposures of the galaxy cluster SMACS J0723.3-7327 (Early Release Observations,
programme 2736, NRCBLONG detector, 837 s each, steps of about 6 arcsec in a
9-point INTRAMODULEBOX dither) are cut down to 256 x 256 px around the same sky
position, east of the brightest cluster galaxy. Each cutout keeps its
calibrated surface brightness (MJy/sr), its ERR and VAR_RNOISE arrays, and the
SIP approximation of the distortion that the pipeline writes into the header.
The weights are a background-limited inverse variance (read noise per pixel
plus the sky Poisson term), which does not depend on source flux: 1/ERR^2
would down-weight the bright pixels of every source and bias the coadd.
``plan`` builds a TAN grid aligned with the detector rows around the union of
the four footprints, and ``coadd`` combines them with cosmic-ray outlier
rejection twice: with a Lanczos3 interpolation kernel and with exact square
drizzle at pixfrac 0.8.

Three checks follow. Flux: the background-subtracted flux of the brightest
isolated sources in a soft-edged 1 arcsec aperture, summed as surface
brightness times the exact solid angle of each pixel, is compared between the
coadds and the median of the input exposures; both kernels conserve it to a
few tenths of a percent on average. Noise: the robust rms of source-free
pixels is compared with the per-pixel prediction ``1/sqrt(inverse_variance)``
(they agree to a few percent, as they do for each input exposure). Pipeline:
the four exposures are also drizzled onto the grid of the MAST level-3 mosaic
(all nine dithers, both modules) and compared with it pixel by pixel, in the
same apertures, and in source centroids.

The cutouts are cut from the MAST products by HTTP range reads
(``examples/_mast_cutout.py``): about 31 MB is transferred once and 3.6 MB is
cached. Runtime about 5 s on CPU once they are cached.

Data: JWST ERO programme 2736 (Pontoppidan et al. 2022, ApJL 936, L14), NIRCam
products from MAST (STScI), public; acknowledge MAST and STScI.

    pixi run -e test -- python examples/jwst_smacs0723_coadd.py [--device cuda] [--out DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import torch
import torch.nn.functional as F

from _mast_cutout import fetch_cutout
from _sample_data import skip_if_unavailable

import torchfits
from torchsky.map import Observation, OutlierConfig, coadd, plan
from torchsky.wcs import WCS

EXPOSURES = (1, 2, 3, 4)
KERNELS = {"lanczos3": {}, "drizzle_square": {"pixfrac": 0.8}}
APERTURE = (0.6, 1.0)  # arcsec: aperture weight 1 inside, cosine taper to 0 at the end
ANNULUS = ((1.2, 1.6), (1.8, 2.2))  # arcsec: inner and outer tapers of the sky ring
UJY_PER_MJY_SR = 1e12  # MJy/sr times sr in microjansky
MIN_FLUX_UJY = (
    1.0  # aperture noise is about 0.01 uJy; keep sources measured to better than 1 %
)


def robust_rms(x: torch.Tensor) -> float:
    return float(1.4826 * (x - x.median()).abs().median())


def background(image: torch.Tensor, block: int = 16) -> torch.Tensor:
    """Smooth background: NaN-ignoring median of ``block`` x ``block`` cells, bilinearly interpolated."""
    h, w = image.shape
    ph, pw = -h % block, -w % block
    padded = F.pad(image, (0, pw, 0, ph), value=float("nan"))
    cells = padded.reshape((h + ph) // block, block, (w + pw) // block, block)
    med = (
        cells.permute(0, 2, 1, 3)
        .reshape(cells.shape[0], cells.shape[2], -1)
        .nanmedian(-1)
        .values
    )
    med = torch.where(torch.isfinite(med), med, torch.nanmedian(med))
    up = F.interpolate(
        med[None, None], scale_factor=block, mode="bilinear", align_corners=False
    )
    return up[0, 0, :h, :w]


def centroid(image: torch.Tensor, x: int, y: int, half: int = 4) -> tuple[float, float]:
    """Background-subtracted first moment in a (2 half + 1)^2 window, pixels."""
    win = image[y - half : y + half + 1, x - half : x + half + 1].double()
    win = (win - win.median()).clamp_min(0.0)
    off = torch.arange(-half, half + 1, dtype=torch.float64)
    return float((win.sum(0) * off).sum() / win.sum()), float(
        (win.sum(1) * off).sum() / win.sum()
    )


def quiet_pixels(snr: torch.Tensor) -> torch.Tensor:
    """Pixels at least 4 px from any pixel above 3 sigma."""
    source = (snr.abs() > 3.0).to(torch.float32)[None, None]
    return (F.max_pool2d(source, 9, 1, 4)[0, 0] == 0) & (snr != 0)


class Photometer:
    """Soft-edged aperture photometry of surface-brightness images.

    The aperture weight is 1 inside ``APERTURE[0]`` and falls to 0 at
    ``APERTURE[1]`` with a cosine taper, so the pixel sum converges on any grid
    instead of depending on which pixel centres fall inside a hard circle. Each
    pixel counts with its exact solid angle. Isolated flagged pixels (NaN in the
    calibrated exposures) in the aperture are filled with the mean of their
    valid neighbours and left out of the sky ring; an aperture with a flagged
    pixel in its core, or whose sky ring runs off the image, is rejected.
    """

    def __init__(self, image: torch.Tensor, header: dict, valid: torch.Tensor) -> None:
        wcs = WCS(header)
        h, w = image.shape
        yy, xx = torch.meshgrid(
            torch.arange(h, dtype=torch.float64),
            torch.arange(w, dtype=torch.float64),
            indexing="ij",
        )
        world = wcs.pixel_to_world(torch.stack([xx.ravel(), yy.ravel()], -1), origin=0)
        self.ra = world[:, 0].reshape(h, w)
        self.dec = world[:, 1].reshape(h, w)
        self.area = wcs.pixel_area_map((h, w), unit="sr")
        v = valid.double()[None, None]
        img = torch.where(valid, image.double(), 0.0)[None, None]
        ones = torch.ones(1, 1, 3, 3, dtype=torch.float64)
        n = F.conv2d(v, ones, padding=1)[0, 0]
        mean = F.conv2d(img, ones, padding=1)[0, 0] / n.clamp_min(1.0)
        self.image = torch.where(valid, image.double(), mean)
        self.valid = valid
        self.fillable = valid | (n >= 4)

    def flux(self, ra: float, dec: float) -> float | None:
        """Background-subtracted flux in microjansky, or None if incomplete."""
        cosd = torch.cos(torch.deg2rad(torch.tensor(dec, dtype=torch.float64)))
        r = 3600.0 * torch.hypot((self.ra - ra) * cosd, self.dec - dec)
        aperture = soft_disc(r, *APERTURE)
        ring = soft_disc(r, *ANNULUS[1]) - soft_disc(r, *ANNULUS[0])
        edge = torch.cat([r[0], r[-1], r[:, 0], r[:, -1]])
        covered = (ring * self.valid * self.area).sum() / (ring * self.area).sum()
        if (
            float(edge.min()) < ANNULUS[1][1]  # the sky ring runs off the image
            or float(covered) < 0.95  # or off the covered area of a coadd
            or not bool(self.valid[r < 0.3].all())
            or not bool(self.fillable[r < APERTURE[1]].all())
        ):
            return None
        # The sky is the area-weighted mean surface brightness in the ring: a
        # linear statistic, so it is conserved by resampling like the flux is.
        ring = ring * self.valid
        sky = (self.image * ring * self.area).sum() / (ring * self.area).sum()
        sb = self.image - sky
        return float((sb * aperture * self.area).sum()) * UJY_PER_MJY_SR


def soft_disc(r: torch.Tensor, start: float, end: float) -> torch.Tensor:
    """1 inside ``start``, 0 beyond ``end``, cosine-squared taper in between."""
    return (
        torch.sin(0.5 * torch.pi * torch.clamp((end - r) / (end - start), 0.0, 1.0))
        ** 2
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    dev = args.device

    with skip_if_unavailable():
        paths = [fetch_cutout(f"smacs0723_f444w_cal_{k}") for k in EXPOSURES]
        i2d_path = fetch_cutout("smacs0723_f444w_i2d")

    observations, inputs, exposures = [], [], []
    for k, path in zip(EXPOSURES, paths):
        sci = torchfits.read(str(path), "SCI").to(torch.float32)
        err = torchfits.read(str(path), "ERR").to(torch.float32)
        rnoise = torchfits.read(str(path), "VAR_RNOISE").to(torch.float32)
        header = dict(torchfits.read_header(str(path), "SCI"))
        good = (
            torch.isfinite(sci)
            & torch.isfinite(err)
            & torch.isfinite(rnoise)
            & (rnoise > 0)
        )
        sci = torch.where(good, sci, 0.0)
        # Background-limited variance: read noise per pixel plus the sky Poisson
        # and flat terms, taken as the median of ERR^2 - VAR_RNOISE (sky dominated).
        # Source Poisson noise is left out so that weights do not depend on flux.
        sky_var = float((err[good] ** 2 - rnoise[good]).median())
        ivar = torch.where(good, 1.0 / (rnoise + sky_var).clamp_min(1e-30), 0.0)
        observations.append(
            Observation(
                data=sci.to(dev),
                wcs=header,
                mask=good.to(dev),
                weight_map=ivar.to(dev),
                weight_is_inverse_variance=True,
            )
        )
        inputs.append(Photometer(sci, header, good))
        exposures.append((sci, ivar, good))
        print(
            f"exposure {k} (dither {header['PATT_NUM']}): pixel origin "
            f"({header['CUTX0']}, {header['CUTY0']}) on {header['DETECTOR']}, "
            f"background sigma {float(ivar[good].median() ** -0.5):.4f} MJy/sr, {int((~good).sum())} flagged px"
        )
    print(
        f"pipeline {header['CAL_VER']} ({header['CRDS_CTX']}); filter {header['FILTER']}"
    )

    # Align the output rows with the detector rows: the smallest grid that
    # holds the dithered cutouts (north-up would need a 356 x 356 grid).
    angle = WCS(observations[0].wcs).position_angle()
    target = plan(observations, projection="TAN", rotation_deg=angle, device=dev)
    scale = 3600.0 * WCS(target.header).pixel_scales()[1]
    print(
        f"planned TAN grid: {tuple(target.shape)} px at {scale:.4f} arcsec/px, rotated {angle:.1f} deg"
    )

    results = {
        kernel: coadd(
            observations,
            target,
            kernel=kernel,
            semantics="surface_brightness",
            method="weighted_mean",
            outliers=OutlierConfig(),
            device=dev,
            **extra,
        )
        for kernel, extra in KERNELS.items()
    }
    out_header = dict(target.header)
    phot = {}
    for kernel, res in results.items():
        data, weight = res.data.cpu(), res.weight.cpu()
        valid = (weight > 0) & torch.isfinite(data)
        phot[kernel] = Photometer(
            torch.where(valid, data, torch.nan), out_header, valid
        )

    # Bright, isolated sources detected on the drizzle coadd.
    drz = results["drizzle_square"]
    data = torch.where(drz.weight.cpu() > 0, drz.data.cpu(), torch.nan)
    sky = float(torch.nanmedian(data))
    resid = data - background(data)
    sigma_pred = drz.inverse_variance.cpu().clamp_min(1e-30) ** -0.5
    snr = torch.nan_to_num(resid / sigma_pred, nan=0.0)
    peaks = (snr == F.max_pool2d(snr[None, None], 25, 1, 12)[0, 0]) & (snr > 10.0)
    ys, xs = torch.nonzero(peaks, as_tuple=True)
    order = snr[ys, xs].argsort(descending=True)
    wcs_out = WCS(out_header)
    rows = []
    for y, x in zip(ys[order].tolist(), xs[order].tolist()):
        ra, dec = (
            float(v)
            for v in wcs_out.pixel_to_world(
                torch.tensor([[x, y]], dtype=torch.float64), origin=0
            )[0]
        )
        f_in = [p.flux(ra, dec) for p in inputs]
        f_in = [f for f in f_in if f is not None]
        f_out = {k: p.flux(ra, dec) for k, p in phot.items()}
        bright = len(f_in) >= 3 and sorted(f_in)[1] > MIN_FLUX_UJY
        if bright and all(v is not None for v in f_out.values()):
            f_ref = torch.tensor(f_in).median().item()
            rows.append(
                (ra, dec, f_ref, robust_rms(torch.tensor(f_in)), f_out, len(f_in))
            )
    assert len(rows) >= 5, f"only {len(rows)} usable sources"
    n_in = sum(row[5] for row in rows)
    print(
        f"flux conservation, {len(rows)} sources above {MIN_FLUX_UJY:.0f} uJy, 1 arcsec soft aperture, coadd / median of "
        f"the inputs ({n_in} source-exposure pairs):"
    )
    ratios = {}
    for kernel in KERNELS:
        r = torch.tensor([row[4][kernel] / row[2] for row in rows])
        ratios[kernel] = r
        print(
            f"  {kernel:15s} median {float(r.median()):.4f}, range {float(r.min()):.4f}-{float(r.max()):.4f}"
        )
        assert abs(float(r.median()) - 1.0) < 0.01
    spread = torch.tensor([row[3] / row[2] for row in rows])
    print(
        f"  input-to-input scatter of the same apertures (robust): median {100 * float(spread.median()):.2f} %"
    )

    # Noise: predicted per-pixel sigma against the robust rms of source-free pixels.
    print(
        "noise in source-free pixels (robust rms / predicted 1/sqrt(inverse_variance)):"
    )
    in_ratio = []
    for sci, ivar, good in exposures:
        d = torch.where(good, sci, torch.nan)
        r = d - background(d)
        sig = ivar.clamp_min(1e-30) ** -0.5
        quiet = quiet_pixels(torch.nan_to_num(r / sig)) & good
        in_ratio.append(robust_rms(r[quiet]) / float(sig[quiet].median()))
    print(f"  {'each exposure':15s} ratio {min(in_ratio):.2f}-{max(in_ratio):.2f}")
    for kernel, res in results.items():
        w = res.weight.cpu()
        d = torch.where(w > 0, res.data.cpu(), torch.nan)
        r = d - background(d)
        sig = res.inverse_variance.cpu().clamp_min(1e-30) ** -0.5
        quiet = quiet_pixels(torch.nan_to_num(r / sig)) & (w > 0)
        rms = robust_rms(r[quiet])
        pred = float(sig[quiet].median())
        corr = float(res.correlation_factor.cpu()[quiet].median())
        print(
            f"  {kernel:15s} rms {rms:.5f}, predicted {pred:.5f} MJy/sr (ratio {rms / pred:.2f}); large-aperture variance factor {corr:.2f}"
        )

    # The STScI pipeline mosaic, on its own grid.
    mast = torchfits.read(str(i2d_path), "SCI").to(torch.float32)
    mast_header = dict(torchfits.read_header(str(i2d_path), "SCI"))
    on_mast = coadd(
        observations,
        mast_header,
        kernel="drizzle_square",
        pixfrac=0.8,
        semantics="surface_brightness",
        method="weighted_mean",
        outliers=OutlierConfig(),
        device=dev,
    )
    ours = on_mast.data.cpu()
    both = (on_mast.weight.cpu() > 0) & torch.isfinite(mast) & (mast != 0)
    offset = float((ours - mast)[both].median())
    diff = torch.where(both, ours - mast - offset, torch.nan)
    cc = float(torch.corrcoef(torch.stack([ours[both], mast[both]]))[0, 1])
    print(
        f"vs MAST i2d ({tuple(mast.shape)} px, 9 dithers x 2 modules): background offset "
        f"{offset:+.4f} MJy/sr ({100 * offset / sky:+.2f} % of the sky), pixel correlation {cc:.4f}, "
        f"robust rms of the difference {robust_rms(diff[both]):.4f} MJy/sr"
    )
    mast_phot = Photometer(torch.where(both, mast, torch.nan), mast_header, both)
    ours_phot = Photometer(torch.where(both, ours, torch.nan), mast_header, both)
    pairs = [(ours_phot.flux(ra, dec), mast_phot.flux(ra, dec)) for ra, dec, *_ in rows]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    mast_ratio = torch.tensor([a / b for a, b in pairs])
    print(
        f"  aperture flux ours / MAST over {len(pairs)} sources: median {float(mast_ratio.median()):.4f}, "
        f"range {float(mast_ratio.min()):.4f}-{float(mast_ratio.max()):.4f}"
    )
    assert abs(float(mast_ratio.median()) - 1.0) < 0.03
    # Registration: first-moment centroids of the same sources on the same grid.
    wcs_mast = WCS(mast_header)
    shifts = []
    for ra, dec, *_ in rows:
        px = wcs_mast.world_to_pixel(
            torch.tensor([[ra, dec]], dtype=torch.float64), origin=0
        )[0]
        x, y = int(round(float(px[0]))), int(round(float(px[1])))
        if 4 <= x < mast.shape[1] - 4 and 4 <= y < mast.shape[0] - 4:
            a, b = centroid(ours, x, y), centroid(mast, x, y)
            shifts.append((a[0] - b[0], a[1] - b[1]))
    shift = torch.tensor(shifts).median(dim=0).values
    mast_scale = 3600e3 * wcs_mast.pixel_scales()[1]
    print(
        f"  centroid offset ours - MAST (x, y) on the MAST grid: ({float(shift[0]) * mast_scale:+.1f}, "
        f"{float(shift[1]) * mast_scale:+.1f}) mas over {len(shifts)} sources"
    )

    out = args.out or Path(tempfile.mkdtemp(prefix="torchsky-jwst-smacs-"))
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.6), constrained_layout=True)
    stretch = dict(origin="lower", cmap="gray_r")

    def show(ax, image, title, lo, hi, cmap="gray_r"):
        im = ax.imshow(
            torch.asinh((image - lo) / (0.02 * (hi - lo))), origin="lower", cmap=cmap
        )
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        return im

    for ax, kernel, title in zip(
        axes[0, :2], KERNELS, ("Lanczos3 coadd", "drizzle coadd (pixfrac 0.8)")
    ):
        res = results[kernel]
        show(
            ax,
            torch.where(res.weight.cpu() > 0, res.data.cpu(), torch.nan),
            f"{title}, planned TAN grid",
            sky,
            sky + 2.0,
        )
    show(
        axes[0, 2],
        torch.where(both, mast + offset, torch.nan),
        "MAST pipeline i2d (its own grid)",
        sky,
        sky + 2.0,
    )
    lim = 5 * robust_rms(diff[both])
    im = axes[1, 0].imshow(
        diff,
        vmin=-lim,
        vmax=lim,
        cmap="RdBu_r",
        **{k: v for k, v in stretch.items() if k != "cmap"},
    )
    axes[1, 0].set_title("drizzle - MAST, offset removed [MJy/sr]", fontsize=9)
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])
    fig.colorbar(im, ax=axes[1, 0], shrink=0.8)
    sig = drz.inverse_variance.cpu().clamp_min(1e-30) ** -0.5
    med = float(sig[drz.weight.cpu() > 0].median())
    im = axes[1, 1].imshow(
        torch.where(drz.weight.cpu() > 0, sig, torch.nan),
        origin="lower",
        cmap="viridis",
        vmin=0.9 * med,
        vmax=1.6 * med,
    )
    axes[1, 1].set_title("drizzle: predicted sigma per pixel [MJy/sr]", fontsize=9)
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])
    fig.colorbar(im, ax=axes[1, 1], shrink=0.8)
    f_mean = torch.tensor([row[2] for row in rows])
    ax = axes[1, 2]
    for kernel, marker in zip(KERNELS, ("o", "s")):
        ax.scatter(f_mean, ratios[kernel], s=18, marker=marker, label=kernel)
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xscale("log")
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylim(0.97, 1.03)
    ax.set_xlabel("median input flux in the 1 arcsec aperture [uJy]", fontsize=8)
    ax.set_ylabel("coadd / input flux", fontsize=8)
    ax.set_title("flux conservation", fontsize=9)
    ax.legend(fontsize=8)
    fig.savefig(out / "jwst_smacs0723_coadd.png", dpi=100)
    print(f"figure: {out / 'jwst_smacs0723_coadd.png'}; device={dev}")


if __name__ == "__main__":
    main()
