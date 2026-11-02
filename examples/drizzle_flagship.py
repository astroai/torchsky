"""Exact square drizzle of two sub-pixel-shifted detector images.

Renders two compact detector images of one source onto a declared target WCS
with the native ``drizzle_square`` kernel and flux semantics, and checks total
flux conservation (to 2e-3), the centroid (to 0.02 pixel) and bounded support
without an external comparator. Runtime under 1 s on CPU.

    pixi run -e test -- python examples/drizzle_flagship.py [--device cuda]
"""

from __future__ import annotations

import argparse

import torch

from torchsky.map import Observation, coadd


def _header(crpix1: float, crpix2: float) -> dict[str, float | int | str]:
    return {
        "NAXIS": 2,
        "NAXIS1": 32,
        "NAXIS2": 32,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": crpix1,
        "CRPIX2": crpix2,
        "CRVAL1": 42.0,
        "CRVAL2": -12.0,
        "CUNIT1": "deg",
        "CUNIT2": "deg",
        "CDELT1": -0.001,
        "CDELT2": 0.001,
    }


def _detector_source(x0: float, y0: float) -> torch.Tensor:
    y, x = torch.meshgrid(
        torch.arange(32, dtype=torch.float32),
        torch.arange(32, dtype=torch.float32),
        indexing="ij",
    )
    radius_squared = (x - x0) ** 2 + (y - y0) ** 2
    image = torch.where(
        radius_squared <= 16.0,
        torch.exp(-0.5 * radius_squared / 1.4**2),
        torch.zeros_like(radius_squared),
    )
    return image / image.sum()


def _centroid(image: torch.Tensor) -> tuple[float, float]:
    y, x = torch.meshgrid(
        torch.arange(image.shape[0], dtype=image.dtype),
        torch.arange(image.shape[1], dtype=image.dtype),
        indexing="ij",
    )
    total = image.sum()
    return float((image * x).sum().item() / total.item()), float(
        (image * y).sum().item() / total.item()
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    device = ap.parse_args().device
    target = _header(16.5, 16.5)
    shifts = ((-0.35, 0.2), (0.4, -0.25))
    observations = []
    for dx, dy in shifts:
        detector = _detector_source(15.5 + dx, 15.5 + dy)
        observations.append(
            Observation(
                data=detector,
                wcs=_header(16.5 + dx, 16.5 + dy),
                weight_map=torch.ones_like(detector),
                weight_is_inverse_variance=True,
            )
        )

    result = coadd(
        observations,
        target,
        kernel="drizzle_square",
        pixfrac=1.0,
        semantics="flux",
        method="weighted_mean",
        device=device,
    )
    result_data, result_weight = result.data.cpu(), result.weight.cpu()
    support = result_weight > 0.0
    measured_flux = float(result_data[support].sum().item())
    centroid_x, centroid_y = _centroid(torch.where(support, result_data, 0.0))
    support_pixels = int(support.sum().item())
    assert abs(measured_flux - 1.0) < 2.0e-3
    assert abs(centroid_x - 15.5) < 2.0e-2
    assert abs(centroid_y - 15.5) < 2.0e-2
    assert 0 < support_pixels <= result_data.numel()

    print(f"flux {measured_flux:.5f} (truth 1), centroid ({centroid_x:.3f}, {centroid_y:.3f}) "
          f"(truth 15.5, 15.5), support {support_pixels} pixels, device={device}")


if __name__ == "__main__":
    main()
