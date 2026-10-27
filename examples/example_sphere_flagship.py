"""Run a compact native sphere workflow from catalog points to harmonics.

This deterministic, network-free flagship builds a MOC footprint and checks
containment, bins a synthetic catalog into matching sparse and dense nested
HEALPix maps, degrades hit counts with explicitly sum-preserving semantics, and
reconstructs a low-order synthetic full sky through native spherical harmonics.
"""

from __future__ import annotations

import json

import torch

from torchsky.sphere import (
    SparseHealpixMap,
    MOC,
    alm2map,
    alm_index,
    alm_size,
    ang2pix,
    map2alm,
    nest2uniq,
    nside2npix,
    ud_grade,
)


def main() -> None:
    nside = 16
    lon = torch.tensor([12.0, 12.2, 12.4, 45.0, 120.0, 240.0], dtype=torch.float64)
    lat = torch.tensor([-3.0, -2.8, -3.2, 20.0, -35.0, 55.0], dtype=torch.float64)
    pixels = ang2pix(nside, lon, lat, nest=True, lonlat=True)
    dense = torch.bincount(pixels, minlength=nside2npix(nside)).to(torch.float64)
    occupied = torch.nonzero(dense, as_tuple=False).flatten()
    sparse = SparseHealpixMap.from_pixels(
        occupied,
        dense[occupied],
        nside_sparse=nside,
        nside_coverage=4,
        nest=True,
        sentinel=0.0,
    )
    torch.testing.assert_close(sparse.to_dense(), dense)

    footprint = MOC(nest2uniq(nside, occupied))
    contained = footprint.contains(lon, lat)
    outside = footprint.contains(torch.tensor([300.0]), torch.tensor([-70.0]))
    assert bool(contained.all()) and not bool(outside.any())

    degraded = ud_grade(
        dense,
        nside_out=8,
        order_in="NESTED",
        order_out="NESTED",
        power=-2.0,
    )
    assert float(degraded.sum().item()) == float(dense.sum().item())

    harmonic_nside = 8
    lmax = 3
    truth = torch.zeros(alm_size(lmax), dtype=torch.complex128)
    truth[alm_index(0, 0, lmax)] = 1.0
    truth[alm_index(1, 0, lmax)] = 0.15
    truth[alm_index(2, 1, lmax)] = 0.08 - 0.03j
    full_sky = alm2map(truth, nside=harmonic_nside, lmax=lmax)
    recovered = map2alm(full_sky, nside=harmonic_nside, lmax=lmax, iter=3)
    reconstruction = alm2map(recovered, nside=harmonic_nside, lmax=lmax)
    relative_rmse = float(
        (torch.linalg.vector_norm(reconstruction - full_sky) / torch.linalg.vector_norm(full_sky)).item()
    )
    assert relative_rmse < 2.0e-3

    print(
        json.dumps(
            {
                "catalog_rows": int(lon.numel()),
                "degrade_semantics": "sum_preserving_hits_power_minus_2",
                "example": "sphere_flagship",
                "harmonic_lmax": lmax,
                "harmonic_relative_rmse": relative_rmse,
                "moc_cells": int(footprint.uniq.numel()),
                "occupied_pixels": int(occupied.numel()),
                "status": "pass",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
