"""Overlap-integrated HEALPix coadds for cells larger than the image pixels.

The oracle is independent of the lattice-chart clipping: each image pixel is
split into 24 x 24 sub-pixels, each sub-pixel is assigned whole to the cell
holding its centre (``ang2pix``) with its spherical-quad solid angle.
"""

from __future__ import annotations

import math

import pytest
import torch

from torchsky.map import Observation, coadd_healpix
from torchsky.map._healpix_overlap import _spherical_quad_area, _world
from torchsky.sphere import healpix as hp
from torchsky.wcs import WCS

SUB = 24


def _header(n: int, cd: float, ra: float, dec: float, rot: float = 17.0) -> dict:
    c, s = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    return {
        "NAXIS": 2, "NAXIS1": n, "NAXIS2": n,
        "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
        "CRPIX1": (n + 1) / 2, "CRPIX2": (n + 1) / 2,
        "CRVAL1": ra, "CRVAL2": dec,
        "CD1_1": -cd * c, "CD1_2": cd * s, "CD2_1": cd * s, "CD2_2": cd * c,
    }  # fmt: skip


def _brute_force(header: dict, image: torch.Tensor, nside: int, nest: bool):
    n = image.shape[0]
    wcs = WCS(header)
    edges = (torch.arange(n * SUB + 1, dtype=torch.float64) / SUB) - 0.5
    yy, xx = torch.meshgrid(edges, edges, indexing="ij")
    lon, lat = _world(wcs, xx.reshape(-1), yy.reshape(-1))
    lon, lat = lon.view(n * SUB + 1, -1), lat.view(n * SUB + 1, -1)

    def quads(a: torch.Tensor) -> torch.Tensor:
        return torch.stack((a[:-1, :-1], a[:-1, 1:], a[1:, 1:], a[1:, :-1]), -1)

    area = _spherical_quad_area(quads(lon).reshape(-1, 4), quads(lat).reshape(-1, 4))
    centres = (torch.arange(n * SUB, dtype=torch.float64) + 0.5) / SUB - 0.5
    cy, cx = torch.meshgrid(centres, centres, indexing="ij")
    clon, clat = _world(wcs, cx.reshape(-1), cy.reshape(-1))
    cells = hp.ang2pix(nside, clon, clat, nest=nest, lonlat=True)
    values = image.to(torch.float64).repeat_interleave(SUB, 0).repeat_interleave(SUB, 1)
    omega = 4.0 * math.pi / (12 * nside**2)
    npix = 12 * nside**2
    weight = torch.zeros(npix, dtype=torch.float64).index_add(0, cells, area / omega)
    numer = torch.zeros(npix, dtype=torch.float64).index_add(
        0, cells, area / omega * values.reshape(-1)
    )
    return weight, numer


@pytest.mark.parametrize(
    "ra,dec",
    [
        (45.0, 10.0),  # equatorial
        (45.0, 41.81),  # straddles z = 2/3 (equatorial / north polar charts)
        (90.0, 60.0),  # straddles the polar interruption at 90 deg
        (10.0, 89.9),  # contains the north pole
        (0.0, -41.81),  # south boundary and the RA = 0 wrap
    ],
)
@pytest.mark.parametrize("nest", [False, True])
def test_overlap_weights_and_means_match_subpixel_oracle(
    ra: float, dec: float, nest: bool
) -> None:
    nside, n, cd = 64, 24, 0.05  # 0.92 deg cells, 1.2 deg image: partial cells
    header = _header(n, cd, ra, dec)
    yy, xx = torch.meshgrid(
        torch.arange(n, dtype=torch.float32),
        torch.arange(n, dtype=torch.float32),
        indexing="ij",
    )
    image = 2.0 + 0.5 * xx - 0.25 * yy
    result = coadd_healpix(
        [Observation(data=image, wcs=header, weight=torch.ones(n, n))],
        nside=nside,
        nest=nest,
    )
    weight_ref, numer_ref = _brute_force(header, image, nside, nest)
    covered = weight_ref > 0.0
    got = result.weight.to(torch.float64)
    # Sub-pixel assignment misplaces at most a boundary strip 1/24 pixel wide.
    assert float((got - weight_ref).abs().max()) < 2e-3
    assert abs(float(got.sum() - weight_ref.sum())) < 1e-5 * float(weight_ref.sum())
    solid = covered & (weight_ref > 0.05)
    mean_ref = numer_ref[solid] / weight_ref[solid]
    assert float((result.data[solid].double() - mean_ref).abs().max()) < 0.05
    assert bool((result.overlap_count[got > 0] == 1).all())


def test_partial_cells_weight_by_covered_fraction_across_exposures() -> None:
    # Two exposures of different values: the cell value is weighted by the
    # area each exposure covers inside the cell.
    nside = 32
    big = _header(64, 0.05, 45.0, 10.0, rot=0.0)  # 3.2 deg, covers whole cells
    small = _header(8, 0.05, 45.0, 10.0, rot=0.0)  # 0.4 deg, a fraction of one
    obs = [
        Observation(data=torch.full((64, 64), 1.0), wcs=big, weight=1.0),
        Observation(data=torch.full((8, 8), 5.0), wcs=small, weight=1.0),
    ]
    result = coadd_healpix(obs, nside=nside, nest=True)
    alone = coadd_healpix(obs[1:], nside=nside, nest=True)
    cells = torch.nonzero(alone.weight > 0).view(-1)
    frac = alone.weight[cells].double()
    assert float(frac.sum()) < 1.0  # the small image is a fraction of one cell
    full = coadd_healpix(obs[:1], nside=nside, nest=True).weight[cells].double()
    expected = (full * 1.0 + frac * 5.0) / (full + frac)
    torch.testing.assert_close(result.data[cells].double(), expected, atol=1e-5, rtol=0)
    assert bool((result.overlap_count[cells] == 2).all())


def test_overlap_path_is_differentiable_in_data() -> None:
    header = _header(16, 0.05, 45.0, 10.0)
    data = torch.full((16, 16), 2.0, requires_grad=True)
    result = coadd_healpix([Observation(data=data, wcs=header)], nside=32)
    (result.data * result.weight).sum().backward()
    assert data.grad is not None and bool(torch.isfinite(data.grad).all())
    # sum_p w_p M_p = sum_i A_i: the gradient is each pixel's area in cells.
    assert float(data.grad.sum()) == pytest.approx(float(result.weight.sum()), rel=1e-5)


def test_fine_cells_keep_centre_sampling() -> None:
    # Cells smaller than the kernel support use the centre-sampled path, so
    # changing the kernel changes the map; on coarse cells it does not.
    header = _header(24, 0.05, 45.0, 10.0)
    torch.manual_seed(0)
    image = torch.rand(24, 24)
    obs = [Observation(data=image, wcs=header)]
    fine_a = coadd_healpix(obs, nside=2048, kernel="nearest")
    fine_b = coadd_healpix(obs, nside=2048, kernel="bilinear")
    assert not torch.equal(fine_a.data, fine_b.data)
    coarse_a = coadd_healpix(obs, nside=64, kernel="nearest")
    coarse_b = coadd_healpix(obs, nside=64, kernel="bilinear")
    torch.testing.assert_close(coarse_a.data, coarse_b.data, rtol=0, atol=0)
