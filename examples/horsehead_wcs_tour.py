"""Pixel to sky on a real photographic-plate image: the Horsehead Nebula.

Science question: where on the sky is this image, how large is a pixel, and what
survives when it is reprojected onto other grids? The Digitized Sky Survey
cutout of the Horsehead Nebula (891 x 893 px, TAN WCS, UK Schmidt red plate)
is read with torchfits and its header is handed to ``torchsky.wcs.WCS`` as a
plain dict. The example checks the pixel/world round trip, measures the pixel
scale and rotation from the autograd Jacobian, converts the footprint corners
and the nebula position from ICRS to Galactic with ``torchsky.wcs.coords``,
compares the exact spherical polygon area of the footprint with its MOC,
reprojects the image onto a rotated TAN grid and a CAR grid with
``torchsky.map.coadd`` (Lanczos3), and reprojects the CAR image back onto the
original grid to measure the interpolation loss. The figure overlays the
Galactic graticule and the boundary of the footprint MOC (degraded to order 12,
52 arcsec cells, so its staircase is visible) on the image next to the CAR
version.

Data: DSS image from the astropy tutorial data (STScI/MAST Digitized Sky Survey;
plate credit AAO/ROE, copyright the Royal Observatory Edinburgh and AAO). Runtime
about 4 s on CPU once the 1.6 MB file is cached.

    pixi run -e test -- python examples/horsehead_wcs_tour.py [--device cuda] [--out DIR]
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
from torchsky.map import Observation, coadd, plan
from torchsky.sphere import MOC, spherical_polygon_area
from torchsky.wcs import WCS
from torchsky.wcs.coords import transform

HORSEHEAD_ICRS = (85.2458, -2.4583)  # Barnard 33, from the header OBJCTRA/OBJCTDEC
MOC_ORDER = 15
DRAW_ORDER = 12  # 52 arcsec cells for the figure


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    dev = args.device

    with skip_if_unavailable():
        path = fetch_sample("horsehead")
    header = dict(torchfits.read_header(str(path), 0))
    image = torchfits.read(str(path), 0).to(torch.float32)
    ny, nx = image.shape
    wcs = WCS(header).to(dev)
    print(
        f"{path.name}: {nx} x {ny} px, {header['CTYPE1']}/{header['CTYPE2']}, {header['RADESYS']}"
    )

    # Round trip over a pixel grid.
    ax = torch.linspace(0.0, nx - 1.0, 11, dtype=torch.float64, device=dev)
    ay = torch.linspace(0.0, ny - 1.0, 11, dtype=torch.float64, device=dev)
    yy, xx = torch.meshgrid(ay, ax, indexing="ij")
    pix = torch.stack([xx.ravel(), yy.ravel()], dim=-1)
    err = float((wcs.world_to_pixel(wcs.pixel_to_world(pix)) - pix).abs().max())
    print(f"pixel -> world -> pixel round trip: max |dpix| = {err:.1e}")
    assert err < 1e-6

    # Pixel scale and rotation from the autograd Jacobian at the image centre.
    centre = torch.tensor([(nx - 1) / 2, (ny - 1) / 2], dtype=torch.float64, device=dev)
    jac = wcs.pixel_to_world_jacobian(centre).cpu()  # d(lon, lat) / d(x, y), deg/px
    cosd = math.cos(math.radians(float(wcs.pixel_to_world(centre)[1])))
    sx = math.hypot(float(jac[0, 0]) * cosd, float(jac[1, 0])) * 3600
    sy = math.hypot(float(jac[0, 1]) * cosd, float(jac[1, 1])) * 3600
    rot = math.degrees(math.atan2(float(jac[0, 1]) * cosd, float(jac[1, 1])))
    print(
        f"pixel scale {sx:.3f} x {sy:.3f} arcsec, field {sx * nx / 60:.1f} x {sy * ny / 60:.1f} arcmin, rotation {rot:+.3f} deg"
    )

    # Footprint corners, Galactic coordinates and the nebula position.
    corners_pix = torch.tensor(
        [[-0.5, -0.5], [nx - 0.5, -0.5], [nx - 0.5, ny - 0.5], [-0.5, ny - 0.5]],
        dtype=torch.float64,
        device=dev,
    )
    corners = wcs.pixel_to_world(corners_pix).cpu()
    gl, gb = transform(
        corners[:, 0], corners[:, 1], from_frame="ICRS", to_frame="GALACTIC"
    )
    area_exact = (
        float(spherical_polygon_area(corners[:, 0], corners[:, 1]))
        * (180 / math.pi) ** 2
    )
    nebula = torch.tensor(HORSEHEAD_ICRS, dtype=torch.float64, device=dev)
    px, py = (float(v) for v in wcs.world_to_pixel(nebula))
    nl, nb = transform(nebula[0], nebula[1], from_frame="ICRS", to_frame="GALACTIC")
    print(
        f"footprint centre l, b = {float(gl.mean()):.2f}, {float(gb.mean()):.2f} deg; exact polygon area {area_exact * 3600:.0f} arcmin^2"
    )
    print(
        f"Horsehead at pixel ({px:.1f}, {py:.1f}), Galactic l, b = {float(nl):.2f}, {float(nb):.2f} deg"
    )

    # MOC of the footprint and its area.
    moc = MOC.from_wcs(wcs, shape=(ny, nx), max_order=MOC_ORDER)
    print(
        f"MOC order {MOC_ORDER}: {moc.uniq.numel()} cells, area {moc.area_sq_deg * 3600:.0f} arcmin^2 ({moc.area_sq_deg / area_exact:.3f} x polygon)"
    )
    assert abs(moc.area_sq_deg / area_exact - 1.0) < 0.05
    coarse = moc.degrade(DRAW_ORDER)  # coarser cells make the MOC staircase visible
    loops = coarse.boundary()
    print(
        f"MOC degraded to order {DRAW_ORDER}: {coarse.uniq.numel()} cells, {len(loops)} boundary loop(s), area {coarse.area_sq_deg * 3600:.0f} arcmin^2"
    )

    # Reprojection: rotated TAN grid, CAR grid, and CAR back to the original grid.
    obs = Observation(
        data=image.to(dev), wcs=header, weight_map=torch.ones_like(image).to(dev)
    )
    tan = plan([obs], projection="TAN", rotation_deg=30.0, device=dev)
    car = plan([obs], projection="CAR", device=dev)
    res_tan = coadd([obs], tan, kernel="lanczos3", device=dev)
    res_car = coadd([obs], car, kernel="lanczos3", device=dev)
    obs_car = Observation(
        data=res_car.data, wcs=car.header, weight_map=(res_car.weight > 0).float()
    )
    back = coadd(
        [obs], plan([obs], reference_index=0, device=dev), kernel="lanczos3", device=dev
    )
    back_from_car = coadd(
        [obs_car],
        plan([obs], reference_index=0, device=dev),
        kernel="lanczos3",
        device=dev,
    )
    inner = (slice(40, -40), slice(40, -40))
    ref = back.data.cpu()[inner]
    rt = back_from_car.data.cpu()[inner]
    ok = torch.isfinite(rt) & (back_from_car.weight.cpu()[inner] > 0)
    rel = float((rt - ref)[ok].pow(2).mean().sqrt() / ref[ok].std())
    print(
        f"rotated TAN grid {tuple(res_tan.data.shape)}, CAR grid {tuple(res_car.data.shape)}; "
        f"original -> CAR -> original residual rms = {rel:.3f} of the image std"
    )
    mean_in, mean_out = (
        float(image.mean()),
        float(res_car.data[res_car.weight > 0].mean()),
    )
    print(
        f"mean surface brightness: original {mean_in:.1f}, CAR {mean_out:.1f} (ratio {mean_out / mean_in:.4f})"
    )

    # Figure: Galactic graticule on the original, and the CAR reprojection.
    out = args.out or Path(tempfile.mkdtemp(prefix="torchsky-horsehead-"))
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6), constrained_layout=True)
    lo, hi = (
        float(v)
        for v in torch.quantile(image[::4, ::4].flatten(), torch.tensor([0.02, 0.995]))
    )
    axes[0].imshow(image, origin="lower", cmap="gray_r", vmin=lo, vmax=hi)
    l0, b0 = float(gl.mean()), float(gb.mean())
    line = torch.linspace(-0.25, 0.25, 50, dtype=torch.float64)
    for dl in (-0.2, -0.1, 0.0, 0.1, 0.2):
        ra, dec = transform(
            torch.full_like(line, l0 + dl),
            b0 + line,
            from_frame="GALACTIC",
            to_frame="ICRS",
        )
        xy = wcs.world_to_pixel(torch.stack([ra, dec], -1).to(dev)).cpu()
        axes[0].plot(xy[:, 0], xy[:, 1], c="C1", lw=0.7)
    for db in (-0.2, -0.1, 0.0, 0.1, 0.2):
        ra, dec = transform(
            l0 + line,
            torch.full_like(line, b0 + db),
            from_frame="GALACTIC",
            to_frame="ICRS",
        )
        xy = wcs.world_to_pixel(torch.stack([ra, dec], -1).to(dev)).cpu()
        axes[0].plot(xy[:, 0], xy[:, 1], c="C0", lw=0.7)
    for loop in loops:
        xy = wcs.world_to_pixel(loop.to(torch.float64).to(dev)).cpu()
        xy = torch.cat([xy, xy[:1]])
        axes[0].plot(xy[:, 0], xy[:, 1], c="C3", lw=1.0)
    axes[0].plot([px], [py], "r+", ms=14)
    pad = 120
    axes[0].set(
        xlim=(-pad, nx + pad),
        ylim=(-pad, ny + pad),
        xticks=[],
        yticks=[],
        title=f"DSS: l (orange), b (blue), MOC order {DRAW_ORDER} (red)",
    )
    axes[1].imshow(res_car.data.cpu(), origin="lower", cmap="gray_r", vmin=lo, vmax=hi)
    axes[1].set(title="Lanczos3 reprojection on a CAR grid", xticks=[], yticks=[])
    fig.savefig(out / "horsehead_wcs_tour.png", dpi=100)
    print(f"figure: {out / 'horsehead_wcs_tour.png'}; device={dev}")


if __name__ == "__main__":
    main()
