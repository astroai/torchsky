"""Demonstrate extended WCS projections: conic, zenithal, polyconic, pseudocylindrical.

Requires: ``pip install astropy`` (for comparison headers and reference pixels).
"""

from __future__ import annotations

import numpy as np
import torch
from astropy.io import fits
from astropy.wcs import WCS as AstropyWCS

from torchsky.wcs import WCS as TorchWCS


def make_header(
    projection: str,
    *,
    crval1: float = 180.0,
    crval2: float = 0.0,
    naxis1: int = 256,
    naxis2: int = 256,
    pv2_1: float | None = None,
    pv2_2: float | None = None,
    cd_scale: float = 0.01,
) -> fits.Header:
    """Create a minimal FITS header for a 2D WCS with a given projection code."""
    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = naxis1
    header["NAXIS2"] = naxis2
    header["CTYPE1"] = f"RA---{projection}"
    header["CTYPE2"] = f"DEC--{projection}"
    header["CRPIX1"] = (naxis1 / 2.0) + 0.5
    header["CRPIX2"] = (naxis2 / 2.0) + 0.5
    header["CRVAL1"] = crval1
    header["CRVAL2"] = crval2
    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"
    header["CD1_1"] = -cd_scale
    header["CD1_2"] = 0.0
    header["CD2_1"] = 0.0
    header["CD2_2"] = cd_scale
    if pv2_1 is not None:
        header["PV2_1"] = pv2_1
    if pv2_2 is not None:
        header["PV2_2"] = pv2_2
    return header


def sample_pixels(header: fits.Header, n_side: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Return (x, y) pixel coordinate arrays spanning the image."""
    x = np.linspace(96.0, header["NAXIS1"] - 96.0, n_side, dtype=np.float64)
    y = np.linspace(96.0, header["NAXIS2"] - 96.0, n_side, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return xx.ravel(), yy.ravel()


def roundtrip_report(
    name: str,
    twcs: TorchWCS,
    awcs: AstropyWCS,
    x: np.ndarray,
    y: np.ndarray,
) -> None:
    """Compare torchsky forward/inverse against Astropy and print diagnostics."""
    # Forward: pixel → world
    ra_ast, dec_ast = awcs.all_pix2world(x, y, 1)
    ra_t, dec_t = twcs.pixel_to_world(
        torch.from_numpy(x), torch.from_numpy(y), origin=1,
    )
    ra_tn = ra_t.cpu().numpy()
    dec_tn = dec_t.cpu().numpy()

    valid = (
        np.isfinite(ra_ast)
        & np.isfinite(dec_ast)
        & np.isfinite(ra_tn)
        & np.isfinite(dec_tn)
    )
    n_valid = int(np.sum(valid))
    if n_valid > 0:
        dra = ((ra_tn[valid] - ra_ast[valid] + 180.0) % 360.0) - 180.0
        ddec = dec_tn[valid] - dec_ast[valid]
        max_dra = float(np.max(np.abs(dra)))
        max_ddec = float(np.max(np.abs(ddec)))
    else:
        max_dra = max_ddec = float("nan")

    # Inverse: world → pixel
    if n_valid > 0:
        x_ast, y_ast = awcs.all_world2pix(ra_ast[valid], dec_ast[valid], 1)
        x_t, y_t = twcs.world_to_pixel(
            torch.from_numpy(ra_ast[valid]),
            torch.from_numpy(dec_ast[valid]),
            origin=1,
        )
        dx = x_t.cpu().numpy() - x_ast
        dy = y_t.cpu().numpy() - y_ast
        max_dx = float(np.max(np.abs(dx)))
        max_dy = float(np.max(np.abs(dy)))
    else:
        max_dx = max_dy = float("nan")

    print(
        f"  {name:30s}  forward Δ≤{max(max_dra,max_ddec):.1e}°"
        f"  inverse Δ≤{max(max_dx,max_dy):.1e} px"
        f"  ({n_valid}/{len(x)} valid)"
    )


def main() -> None:
    print("Extended WCS Projections — parity vs Astropy/WCSLIB")
    print("=" * 70)

    # ── Zenithal extended ──────────────────────────────────────────
    print("\n▶ Zenithal extended (AZP, SZP, AIR)")
    for proj, pv21, pv22, cd in [
        ("AZP", 0.0, None, 0.02),
        ("AZP", 2.0, None, 0.02),
        ("AZP", 0.0, 20.0, 0.02),
        ("SZP", 0.0, None, 0.02),
        ("SZP", 3.0, None, 0.02),
        ("AIR", 90.0, None, 0.02),
        ("AIR", 60.0, None, 0.02),
    ]:
        hdr = make_header(proj, pv2_1=pv21, pv2_2=pv22, cd_scale=cd)
        x, y = sample_pixels(hdr)
        label = f"{proj}(pv2_1={pv21}"
        label += f", pv2_2={pv22})" if pv22 is not None else ")"
        roundtrip_report(label, TorchWCS(dict(hdr)), AstropyWCS(hdr), x, y)

    # ── Pseudocylindrical ─────────────────────────────────────────
    print("\n▶ Pseudocylindrical (PAR)")
    hdr = make_header("PAR", cd_scale=0.5)
    x, y = sample_pixels(hdr)
    roundtrip_report("PAR", TorchWCS(dict(hdr)), AstropyWCS(hdr), x, y)

    # ── Polyconic (Bonne) ─────────────────────────────────────────
    print("\n▶ Polyconic (BON)")
    for pv21 in [30.0, 45.0, 60.0]:
        hdr = make_header("BON", pv2_1=pv21, cd_scale=0.5)
        x, y = sample_pixels(hdr)
        roundtrip_report(
            f"BON(pv2_1={pv21})", TorchWCS(dict(hdr)), AstropyWCS(hdr), x, y
        )

    # ── Conic ─────────────────────────────────────────────────────
    print("\n▶ Conic (COP, COE, COD, COO)")
    for proj, pv21, pv22 in [
        ("COP", 45.0, 0.0),
        ("COP", 30.0, 10.0),
        ("COE", 45.0, 0.0),
        ("COE", 30.0, 10.0),
        ("COD", 45.0, 0.0),
        ("COD", 30.0, 10.0),
        ("COO", 45.0, 0.0),
        ("COO", 30.0, 10.0),
    ]:
        hdr = make_header(
            proj, crval2=pv21, pv2_1=pv21, pv2_2=pv22, cd_scale=0.02,
        )
        x, y = sample_pixels(hdr)
        label = f"{proj}(pv2_1={pv21}, pv2_2={pv22})"
        roundtrip_report(label, TorchWCS(dict(hdr)), AstropyWCS(hdr), x, y)

    # ── Device coverage ───────────────────────────────────────────
    print("\n▶ Device check")
    hdr = make_header("TAN", cd_scale=0.02)
    twcs = TorchWCS(dict(hdr))
    x = torch.tensor([128.5], dtype=torch.float64)
    y = torch.tensor([128.5], dtype=torch.float64)
    ra, dec = twcs.pixel_to_world(x, y)
    print(f"  CPU: ra={ra.item():.6f}°, dec={dec.item():.6f}°")
    if torch.cuda.is_available():
        twcs_cuda = TorchWCS(dict(hdr))
        xc = x.to("cuda")
        yc = y.to("cuda")
        ra_c, dec_c = twcs_cuda.pixel_to_world(xc, yc)
        print(f"  CUDA: ra={ra_c.item():.6f}°, dec={dec_c.item():.6f}°")

    print("\nAll extended projection checks complete.")


if __name__ == "__main__":
    main()
