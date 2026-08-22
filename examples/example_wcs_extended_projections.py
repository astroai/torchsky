"""Demonstrate extended WCS projections: conic, zenithal, polyconic, pseudocylindrical."""

from __future__ import annotations

import numpy as np
import torch

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
) -> dict:
    """Create a minimal FITS header for a 2D WCS with a given projection code."""
    header = {
        "NAXIS": 2,
        "NAXIS1": naxis1,
        "NAXIS2": naxis2,
        "CTYPE1": f"RA---{projection}",
        "CTYPE2": f"DEC--{projection}",
        "CRPIX1": (naxis1 / 2.0) + 0.5,
        "CRPIX2": (naxis2 / 2.0) + 0.5,
        "CRVAL1": crval1,
        "CRVAL2": crval2,
        "CUNIT1": "deg",
        "CUNIT2": "deg",
        "CD1_1": -cd_scale,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": cd_scale,
    }
    if pv2_1 is not None:
        header["PV2_1"] = pv2_1
    if pv2_2 is not None:
        header["PV2_2"] = pv2_2
    return header


def sample_pixels(header: dict, n_side: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Return (x, y) pixel coordinate arrays spanning the image."""
    x = np.linspace(96.0, header["NAXIS1"] - 96.0, n_side, dtype=np.float64)
    y = np.linspace(96.0, header["NAXIS2"] - 96.0, n_side, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return xx.ravel(), yy.ravel()


def roundtrip_report(name: str, twcs: TorchWCS, x: np.ndarray, y: np.ndarray) -> None:
    """Round-trip pixel → world → pixel with torchsky WCS."""
    ra_t, dec_t = twcs.pixel_to_world(
        torch.from_numpy(x), torch.from_numpy(y), origin=1
    )
    ra_n = ra_t.cpu().numpy()
    dec_n = dec_t.cpu().numpy()
    valid = np.isfinite(ra_n) & np.isfinite(dec_n)
    n_valid = int(np.sum(valid))
    if n_valid == 0:
        print(f"  {name:30s}  no finite world samples")
        return
    x_back, y_back = twcs.world_to_pixel(
        torch.from_numpy(ra_n[valid]),
        torch.from_numpy(dec_n[valid]),
        origin=1,
    )
    dx = x_back.cpu().numpy() - x[valid]
    dy = y_back.cpu().numpy() - y[valid]
    print(
        f"  {name:30s}  roundtrip Δ≤{max(float(np.max(np.abs(dx))), float(np.max(np.abs(dy)))):.1e} px"
        f"  ({n_valid}/{len(x)} valid)"
    )


def main() -> None:
    print("Extended WCS Projections — torchsky roundtrip")
    print("=" * 70)

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
        roundtrip_report(label, TorchWCS(dict(hdr)), x, y)

    print("\n▶ Pseudocylindrical (PAR)")
    hdr = make_header("PAR", cd_scale=0.5)
    x, y = sample_pixels(hdr)
    roundtrip_report("PAR", TorchWCS(dict(hdr)), x, y)

    print("\n▶ Polyconic (BON)")
    for pv21 in [30.0, 45.0, 60.0]:
        hdr = make_header("BON", pv2_1=pv21, cd_scale=0.5)
        x, y = sample_pixels(hdr)
        roundtrip_report(f"BON(pv2_1={pv21})", TorchWCS(dict(hdr)), x, y)

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
        roundtrip_report(label, TorchWCS(dict(hdr)), x, y)

    print("\n▶ Device check")
    hdr = make_header("TAN", cd_scale=0.02)
    twcs = TorchWCS(dict(hdr))
    x = torch.tensor([128.5], dtype=torch.float64)
    y = torch.tensor([128.5], dtype=torch.float64)
    ra, dec = twcs.pixel_to_world(x, y)
    print(f"  CPU: ra={ra.item():.6f}°, dec={dec.item():.6f}°")
    if torch.cuda.is_available():
        xc = x.to("cuda")
        yc = y.to("cuda")
        ra_c, dec_c = twcs.pixel_to_world(xc, yc)
        print(f"  CUDA: ra={ra_c.item():.6f}°, dec={dec_c.item():.6f}°")

    print("\nAll extended projection checks complete.")


if __name__ == "__main__":
    main()
