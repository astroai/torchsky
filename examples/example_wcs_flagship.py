"""Validate in-memory celestial WCS families, inverses, and derivatives.

This network-free flagship exercises TAN with a CD matrix, TAN-SIP, and TPV
headers, checks pixel/world round-trips, and compares a Torch autograd Jacobian
with an independent central finite difference. It demonstrates only header
terms embedded below and makes no D2IM or CPDIS support claim.
"""

from __future__ import annotations

import json

import torch

from torchsky.wcs import WCS


def _base_header(ctype: str) -> dict[str, float | int | str]:
    return {
        "NAXIS": 2,
        "NAXIS1": 96,
        "NAXIS2": 80,
        "CTYPE1": f"RA---{ctype}",
        "CTYPE2": f"DEC--{ctype}",
        "CRPIX1": 48.5,
        "CRPIX2": 40.5,
        "CRVAL1": 150.0,
        "CRVAL2": 2.0,
        "CUNIT1": "deg",
        "CUNIT2": "deg",
        "CD1_1": -2.8e-4,
        "CD1_2": 1.0e-6,
        "CD2_1": 8.0e-7,
        "CD2_2": 2.8e-4,
    }


def _headers() -> dict[str, dict[str, float | int | str]]:
    tan = _base_header("TAN")
    sip = _base_header("TAN-SIP")
    sip.update(
        {
            "A_ORDER": 2,
            "B_ORDER": 2,
            "A_2_0": 3.0e-6,
            "A_1_1": -2.0e-6,
            "A_0_2": 1.0e-6,
            "B_2_0": -2.5e-6,
            "B_1_1": 2.0e-6,
            "B_0_2": 2.5e-6,
        }
    )
    tpv = _base_header("TPV")
    tpv.update(
        {
            "PV1_0": 0.0,
            "PV1_1": 1.0,
            "PV1_2": 0.0,
            "PV1_4": 2.0e-4,
            "PV1_5": -3.0e-4,
            "PV1_6": 1.5e-4,
            "PV2_0": 0.0,
            "PV2_1": 1.0,
            "PV2_2": 2.0e-4,
            "PV2_4": -1.0e-4,
            "PV2_5": 2.5e-4,
            "PV2_6": -2.0e-4,
        }
    )
    return {"tan_cd": tan, "tan_sip": sip, "tpv": tpv}


def _finite_difference(wcs: WCS, pixel: torch.Tensor, step: float) -> torch.Tensor:
    columns = []
    for axis in range(2):
        delta = torch.zeros_like(pixel)
        delta[axis] = step
        plus = wcs.pixel_to_world_differentiable(pixel + delta)
        minus = wcs.pixel_to_world_differentiable(pixel - delta)
        columns.append((plus - minus) / (2.0 * step))
    return torch.stack(columns, dim=-1)


def main() -> None:
    pixels = torch.tensor(
        [[12.25, 18.75], [47.5, 39.5], [73.125, 61.875]], dtype=torch.float64
    )
    roundtrips: dict[str, float] = {}
    wcs_by_name: dict[str, WCS] = {}
    for name, header in _headers().items():
        wcs = WCS(header)
        world = wcs.pixel_to_world(pixels)
        recovered = wcs.world_to_pixel(world)
        error = float(torch.max(torch.abs(recovered - pixels)).item())
        assert error < 2.0e-6
        roundtrips[name] = error
        wcs_by_name[name] = wcs

    point = torch.tensor([53.25, 44.75], dtype=torch.float64)
    autograd_jacobian = wcs_by_name["tan_sip"].pixel_to_world_jacobian(point)
    finite_difference_jacobian = _finite_difference(wcs_by_name["tan_sip"], point, 1.0e-3)
    jacobian_error = float(
        torch.max(torch.abs(autograd_jacobian - finite_difference_jacobian)).item()
    )
    assert jacobian_error < 1.0e-9

    print(
        json.dumps(
            {
                "example": "wcs_flagship",
                "families": sorted(roundtrips),
                "jacobian_max_abs_deg_per_pixel": jacobian_error,
                "not_demonstrated": ["D2IM", "CPDIS"],
                "roundtrip_max_abs_pixel": roundtrips,
                "status": "pass",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
