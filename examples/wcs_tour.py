"""A tour of torchsky WCS: TAN, TAN-SIP, TPV and extended projections.

Builds in-memory header dicts (no files), checks pixel/world round trips for
every family, compares the autograd pixel-to-world Jacobian of a SIP WCS with
an independent central finite difference, and draws the sky footprint of a
pixel grid under four projections. Network-free; runtime about 3 s on CPU.

    pixi run -e test -- python examples/wcs_tour.py [--device cuda] [--out DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from torchsky.wcs import WCS


def base_header(code: str, *, scale: float = 2.8e-4, crval=(150.0, 2.0)) -> dict:
    return {
        "NAXIS": 2,
        "NAXIS1": 256,
        "NAXIS2": 256,
        "CTYPE1": f"RA---{code}",
        "CTYPE2": f"DEC--{code}",
        "CRPIX1": 128.5,
        "CRPIX2": 128.5,
        "CRVAL1": crval[0],
        "CRVAL2": crval[1],
        "CUNIT1": "deg",
        "CUNIT2": "deg",
        "CD1_1": -scale,
        "CD1_2": 1.0e-6,
        "CD2_1": 8.0e-7,
        "CD2_2": scale,
    }


def sip_header() -> dict:
    h = base_header("TAN-SIP")
    h.update(
        A_ORDER=2,
        B_ORDER=2,
        A_2_0=3e-6,
        A_1_1=-2e-6,
        A_0_2=1e-6,
        B_2_0=-2.5e-6,
        B_1_1=2e-6,
        B_0_2=2.5e-6,
    )
    return h


def tpv_header() -> dict:
    h = base_header("TPV")
    h.update(
        PV1_0=0.0,
        PV1_1=1.0,
        PV1_4=2e-4,
        PV1_5=-3e-4,
        PV1_6=1.5e-4,
        PV2_0=0.0,
        PV2_1=1.0,
        PV2_2=2e-4,
        PV2_4=-1e-4,
        PV2_5=2.5e-4,
        PV2_6=-2e-4,
    )
    return h


def grid(device: str, n: int = 9) -> torch.Tensor:
    ax = torch.linspace(16.0, 240.0, n, dtype=torch.float64, device=device)
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    return torch.stack([xx.ravel(), yy.ravel()], dim=-1)


def roundtrip(header: dict, device: str) -> tuple[WCS, torch.Tensor, float]:
    wcs = WCS(header).to(device)
    pix = grid(device)
    back = wcs.world_to_pixel(wcs.pixel_to_world(pix))
    return wcs, pix, float((back - pix).abs().max())


def finite_difference(wcs: WCS, p: torch.Tensor, step: float = 1e-3) -> torch.Tensor:
    cols = []
    for axis in range(2):
        d = torch.zeros_like(p)
        d[axis] = step
        plus = wcs.pixel_to_world_differentiable(p + d)
        minus = wcs.pixel_to_world_differentiable(p - d)
        cols.append((plus - minus) / (2.0 * step))
    return torch.stack(cols, dim=-1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or Path(tempfile.mkdtemp(prefix="torchsky-wcs-tour-"))
    out.mkdir(parents=True, exist_ok=True)

    families = {
        "TAN": base_header("TAN"),
        "TAN-SIP": sip_header(),
        "TPV": tpv_header(),
        "AZP mu=2": {**base_header("AZP", scale=0.02), "PV2_1": 2.0},
        "AIR": {**base_header("AIR", scale=0.02), "PV2_1": 90.0},
        "COE": {
            **base_header("COE", scale=0.02, crval=(150.0, 45.0)),
            "PV2_1": 45.0,
            "PV2_2": 10.0,
        },
        "BON": {**base_header("BON", scale=0.5), "PV2_1": 45.0},
        "PAR": base_header("PAR", scale=0.5),
    }
    worlds = {}
    print(f"{'projection':10s} round-trip max |dpix|")
    for name, header in families.items():
        wcs, pix, err = roundtrip(header, args.device)
        assert err < 1e-5, (name, err)
        worlds[name] = wcs.pixel_to_world(pix).cpu()
        print(f"{name:10s} {err:.2e}")

    sip = WCS(sip_header()).to(args.device)
    p = torch.tensor([140.25, 120.75], dtype=torch.float64, device=args.device)
    jac = sip.pixel_to_world_jacobian(p)
    jerr = float((jac - finite_difference(sip, p)).abs().max())
    assert jerr < 1e-9, jerr
    print(f"SIP autograd Jacobian vs finite difference: {jerr:.2e} deg/pix")

    fig, axes = plt.subplots(1, 4, figsize=(11, 2.8), constrained_layout=True)
    for ax, name in zip(axes, ["TAN-SIP", "TPV", "AIR", "COE"]):
        w = worlds[name]
        ax.scatter(w[:, 0], w[:, 1], s=8, c=torch.arange(len(w)), cmap="viridis")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("lon [deg]", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.invert_xaxis()
    axes[0].set_ylabel("lat [deg]", fontsize=8)
    fig.savefig(out / "wcs_tour.png", dpi=100)
    print(f"device={args.device}  figure: {out / 'wcs_tour.png'}")


if __name__ == "__main__":
    main()
