"""Vectorised IFU response planner: loop parity, fibre apertures, fail-closed footprints."""

from __future__ import annotations

import math

import pytest
import torch

from ifu_loop_oracle import loop_plan_ifu_cube_response
from torchsky.map import CubeTarget, IFUObservation, coadd_cube, plan_ifu_cube_response
from torchsky.map.ifu import _disk_polygon_area
from torchsky.wcs import WCS
from torchsky.wcs.gwcs_sampling import IFUSliceWCS


def _wcs(
    shape: tuple[int, int], scale_arcsec: float = 1.0, rotation_deg: float = 0.0
) -> WCS:
    height, width = shape
    c, s = math.cos(math.radians(rotation_deg)), math.sin(math.radians(rotation_deg))
    scale = scale_arcsec / 3600.0
    return WCS(
        {
            "NAXIS": 2,
            "NAXIS1": width,
            "NAXIS2": height,
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
            "CRPIX1": (width + 1) / 2,
            "CRPIX2": (height + 1) / 2,
            "CRVAL1": 150.0,
            "CRVAL2": 2.0,
            "CD1_1": -scale * c,
            "CD1_2": scale * s,
            "CD2_1": scale * s,
            "CD2_2": scale * c,
        }
    )


class _CurvedDetector:
    """Smooth non-linear detector -> (ra, dec, wavelength) map (forces subdivision)."""

    def __init__(self, shape: tuple[int, int], wave_step: float = 7.0) -> None:
        self.shape = shape
        self.wave_step = wave_step

    def pixel_to_world(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.as_tensor(x, dtype=torch.float64)
        y = torch.as_tensor(y, dtype=torch.float64)
        u = 0.73 * x + 0.21 * y + 0.04 * x * x - 0.5
        v = -0.18 * x + 0.81 * y + 0.03 * y * y + 0.3
        ra = 150.0 - u / 3600.0 / math.cos(math.radians(2.0))
        dec = 2.0 + v / 3600.0
        wavelength = 5000.0 + self.wave_step * y + 0.4 * x
        return ra, dec, wavelength


def _slice_transform(shape: tuple[int, int], *, columns_are_slices: bool = False):
    height, width = shape
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float64),
        torch.arange(width, dtype=torch.float64),
        indexing="ij",
    )
    ra, dec = _wcs(shape, rotation_deg=17.0).pixel_to_world(0.9 * x, 1.1 * y)
    slice_id = (
        x.to(torch.int64)
        if columns_are_slices
        else torch.zeros_like(x, dtype=torch.int64)
    )
    return IFUSliceWCS(ra, dec, 5000.0 + 10.0 * y + 0.5 * x, slice_id)


def _assert_same_plan(new, old) -> None:
    torch.testing.assert_close(new.plan.source_indices, old.plan.source_indices)
    torch.testing.assert_close(new.plan.target_indices, old.plan.target_indices)
    torch.testing.assert_close(new.plan.contributor_ids, old.plan.contributor_ids)
    torch.testing.assert_close(
        new.plan.coefficients, old.plan.coefficients, rtol=1.0e-12, atol=0.0
    )
    torch.testing.assert_close(new.weight, old.weight, rtol=1.0e-12, atol=0.0)
    assert new.plan.coefficients.numel() > 0


def _observations(semantics: str, *, curved: bool) -> list[IFUObservation]:
    generator = torch.Generator().manual_seed(3)
    out = []
    for index, shape in enumerate(((6, 5), (4, 7))):
        science = torch.rand(shape, generator=generator, dtype=torch.float64)
        dq = torch.zeros(shape, dtype=torch.int32)
        dq[1, 2] = 1
        science[2, 3] = float("nan")
        transform = _CurvedDetector(shape) if curved else _slice_transform(shape)
        out.append(
            IFUObservation(
                science,
                transform,
                dq=dq,
                variance_components={"read": 0.1 + science.nan_to_num().abs()},
                value_semantics=semantics,
                weight=None if index == 0 else 2.0,
            )
        )
    return out


def _target(semantics: str, *, rotation: float = 8.0) -> CubeTarget:
    return CubeTarget(
        _wcs((7, 8), scale_arcsec=0.8, rotation_deg=rotation),
        torch.linspace(4990.0, 5080.0, 11, dtype=torch.float64),
        shape=(7, 8),
        value_semantics=semantics,
    )


@pytest.mark.parametrize("semantics", ["flux", "surface_brightness"])
@pytest.mark.parametrize("curved", [False, True])
@pytest.mark.parametrize(
    "options",
    [
        {},
        {"pixfrac": 0.6, "spectral_pixfrac": 0.7},
        {"footprint_tolerance": 1.0e-4},
    ],
)
def test_vectorised_drizzle_matches_loop_planner(
    semantics: str, curved: bool, options: dict
) -> None:
    observations = _observations(semantics, curved=curved)
    target = _target(semantics)
    new = plan_ifu_cube_response(observations, target, kernel="drizzle", **options)
    old = loop_plan_ifu_cube_response(observations, target, kernel="drizzle", **options)
    _assert_same_plan(new, old)


@pytest.mark.parametrize("kernel", ["emsm", "msm"])
@pytest.mark.parametrize(
    "options",
    [
        {},
        {"emsm_scale": 0.7},
        {
            "spatial_roi": lambda w: 1.2 + 0.0002 * (w - 5000.0),
            "resolving_power": lambda w: w.new_tensor(600.0),
        },
        {"spatial_roi": 1.5, "spectral_roi": 12.0},
    ],
)
def test_vectorised_distance_kernels_match_loop_planner(
    kernel: str, options: dict
) -> None:
    observations = _observations("surface_brightness", curved=True)
    target = _target("surface_brightness")
    new = plan_ifu_cube_response(observations, target, kernel=kernel, **options)
    old = loop_plan_ifu_cube_response(observations, target, kernel=kernel, **options)
    _assert_same_plan(new, old)


def test_vectorised_supplied_footprints_and_resolving_power_fallback() -> None:
    shape = (3, 4)
    transform = _slice_transform(shape)
    y, x = torch.meshgrid(
        torch.arange(3, dtype=torch.float64),
        torch.arange(4, dtype=torch.float64),
        indexing="ij",
    )
    hexagon = torch.stack(
        [
            torch.tensor([math.cos(a), math.sin(a)], dtype=torch.float64) * 0.45
            for a in torch.linspace(0, 2 * math.pi, 7)[:-1].tolist()
        ]
    )
    detector = torch.stack((x, y), dim=-1)[:, :, None, :] + hexagon
    target = _target("flux")
    science = torch.arange(12, dtype=torch.float64).reshape(shape) + 1.0
    for footprint in (
        detector,
        torch.cat(
            (
                torch.stack(
                    transform.pixel_to_world(detector[..., 0], detector[..., 1])[:2],
                    dim=-1,
                ),
                # One wavelength per footprint: the spectral width falls back to
                # the resolving power.
                (5000.0 + 9.0 * y)[:, :, None, None].expand(3, 4, 6, 1),
            ),
            dim=-1,
        ),
    ):
        observation = IFUObservation(
            science,
            transform,
            detector_pixel_footprint=footprint,
            value_semantics="flux",
        )
        options = {"resolving_power": lambda w: w / 4.0}
        new = plan_ifu_cube_response([observation], target, **options)
        old = loop_plan_ifu_cube_response([observation], target, **options)
        _assert_same_plan(new, old)


def test_vectorised_planner_bounds_blocks_and_entries(monkeypatch) -> None:
    import torchsky.map.ifu as ifu

    observations = _observations("flux", curved=True)
    target = _target("flux")
    reference = plan_ifu_cube_response(observations, target)
    monkeypatch.setattr(ifu, "_BLOCK_ELEMENTS", 7)
    monkeypatch.setattr(ifu, "_SAMPLE_BLOCK", 5)
    small = plan_ifu_cube_response(observations, target)
    torch.testing.assert_close(small.plan.coefficients, reference.plan.coefficients)
    torch.testing.assert_close(small.plan.target_indices, reference.plan.target_indices)
    with pytest.raises(MemoryError, match="max_response_entries"):
        plan_ifu_cube_response(observations, target, max_response_entries=10)


@pytest.mark.parametrize("radius", [0.3, 1.0, 2.7])
def test_disk_polygon_area_is_exact(radius: float) -> None:
    # A disk tiled by unit squares: the overlaps sum to its area; one quadrant
    # square holds exactly a quarter.
    cells = torch.arange(-4, 4, dtype=torch.float64)
    yy, xx = torch.meshgrid(cells, cells, indexing="ij")
    x = xx.reshape(-1, 1) + torch.tensor([0.0, 1.0, 1.0, 0.0], dtype=torch.float64)
    y = yy.reshape(-1, 1) + torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float64)
    area = _disk_polygon_area(x, y, torch.full((x.shape[0],), radius, dtype=x.dtype))
    torch.testing.assert_close(
        area.sum(), area.new_tensor(math.pi * radius**2), rtol=1e-13, atol=0.0
    )
    big = torch.tensor([[0.0, 10.0, 10.0, 0.0]], dtype=torch.float64)
    quarter = _disk_polygon_area(big, big[:, [0, 0, 1, 1]], area.new_tensor([radius]))
    torch.testing.assert_close(quarter, area.new_tensor([math.pi * radius**2 / 4]))
    # Clockwise polygons give the negative area.
    torch.testing.assert_close(
        _disk_polygon_area(
            big.flip(1), big[:, [0, 0, 1, 1]].flip(1), area.new_tensor([radius])
        ),
        -quarter,
    )


def _fibre_bundle(
    n_wave: int = 12, dither: float = 0.0
) -> tuple[torch.Tensor, IFUSliceWCS, torch.Tensor, torch.Tensor]:
    """A hexagonal 19-fibre bundle (2 arcsec fibres, 2.5 arcsec pitch) x 3 dithers."""
    centres = [(0.0, 0.0)]
    for ring in (1, 2):
        for side in range(6):
            a0, a1 = math.radians(60 * side), math.radians(60 * (side + 1))
            for step in range(ring):
                t = step / ring
                centres.append(
                    (
                        ring * ((1 - t) * math.cos(a0) + t * math.cos(a1)),
                        ring * ((1 - t) * math.sin(a0) + t * math.sin(a1)),
                    )
                )
    base = torch.tensor(centres, dtype=torch.float64) * 2.5
    offsets = torch.tensor([[0.0, 0.0], [1.25, 0.72], [0.0, 1.44]], dtype=torch.float64)
    xy = (base[None] + offsets[:, None] + dither).reshape(-1, 2)  # arcsec east, north
    n_fibre = xy.shape[0]
    cosd = math.cos(math.radians(2.0))
    ra = (150.0 + xy[:, 0] / 3600.0 / cosd)[None].expand(n_wave, n_fibre)
    dec = (2.0 + xy[:, 1] / 3600.0)[None].expand(n_wave, n_fibre)
    wave = (5000.0 + 2.0 * torch.arange(n_wave, dtype=torch.float64))[:, None]
    transform = IFUSliceWCS(
        ra.contiguous(),
        dec.contiguous(),
        wave.expand(n_wave, n_fibre).contiguous(),
        torch.arange(n_fibre)[None].expand(n_wave, n_fibre).contiguous(),
    )
    return xy, transform, wave.squeeze(1), ra


def test_fibre_columns_fail_closed_without_explicit_footprint() -> None:
    _, transform, wave, _ = _fibre_bundle()
    observation = IFUObservation(
        torch.ones(transform.detector_shape, dtype=torch.float64), transform
    )
    edges = torch.cat((wave - 1.0, wave[-1:] + 1.0))
    target = CubeTarget(_wcs((15, 15)), edges, shape=(15, 15))
    with pytest.raises(ValueError, match="same-slice neighbour along detector x"):
        plan_ifu_cube_response([observation], target, kernel="drizzle")
    # Point kernels need no footprint.
    plan_ifu_cube_response([observation], target, kernel="emsm")
    with pytest.raises(ValueError, match="either detector_pixel_footprint"):
        IFUObservation(
            observation.sci,
            transform,
            detector_pixel_footprint=torch.zeros((*observation.shape, 4, 2)),
            aperture_diameter_arcsec=2.0,
        )
    with pytest.raises(ValueError, match="finite and positive"):
        IFUObservation(observation.sci, transform, aperture_diameter_arcsec=-1.0)


@pytest.mark.parametrize("rotation", [0.0, 23.0])
def test_fibre_aperture_conserves_flux_and_centroid(rotation: float) -> None:
    xy, transform, wave, _ = _fibre_bundle()
    # A Gaussian source at (0.6, -0.4) arcsec, sigma 1.5 arcsec, integrated
    # exactly over each 2 arcsec fibre to first order (fibre flux = SB x area).
    source = torch.tensor([0.6, -0.4], dtype=torch.float64)
    sigma = 1.5
    sb = torch.exp(-0.5 * ((xy - source).square().sum(-1)) / sigma**2)
    fibre_area = math.pi * 1.0**2
    flux = (sb * fibre_area)[None].expand(transform.detector_shape).contiguous()
    edges = torch.cat((wave - 1.0, wave[-1:] + 1.0))
    target = CubeTarget(
        _wcs((41, 41), scale_arcsec=0.5, rotation_deg=rotation),
        edges,
        shape=(41, 41),
        value_semantics="flux",
    )
    observation = IFUObservation(
        flux, transform, value_semantics="flux", aperture_diameter_arcsec=2.0
    )
    result = coadd_cube([observation], target, spectral_kernel="drizzle")
    # Every fibre lies inside the target: the drizzled flux is the fibre flux.
    torch.testing.assert_close(
        result.data.sum((1, 2)), flux.sum(1), rtol=1e-12, atol=0.0
    )
    # Flux-weighted centroid on the sky equals the fibres' flux-weighted centre.
    y, x = torch.meshgrid(
        torch.arange(41, dtype=torch.float64),
        torch.arange(41, dtype=torch.float64),
        indexing="ij",
    )
    image = result.data[0]
    cx = float((image * x).sum() / image.sum())
    cy = float((image * y).sum() / image.sum())
    ra_c, dec_c = target.spatial_wcs.pixel_to_world(
        torch.tensor([cx], dtype=torch.float64), torch.tensor([cy], dtype=torch.float64)
    )
    east = (float(ra_c) - 150.0) * 3600.0 * math.cos(math.radians(2.0))
    north = (float(dec_c) - 2.0) * 3600.0
    expected = (flux[0, :, None] * xy).sum(0) / flux[0].sum()
    assert abs(east - float(expected[0])) < 1e-3
    assert abs(north - float(expected[1])) < 1e-3
    # One fibre alone covers pi r^2 / 0.25 arcsec^2 spaxels' worth of area.
    single = IFUObservation(
        torch.ones((1, 1), dtype=torch.float64),
        IFUSliceWCS(
            transform.ra_map[:1, :1],
            transform.dec_map[:1, :1],
            transform.wavelength_map[:1, :1],
            torch.zeros((1, 1), dtype=torch.int64),
        ),
        value_semantics="flux",
        aperture_diameter_arcsec=2.0,
    )
    response = plan_ifu_cube_response([single], target, resolving_power=2500.0)
    torch.testing.assert_close(
        response.weight.sum(), response.weight.new_tensor(1.0), rtol=1e-12, atol=0.0
    )
    assert response.plan.coefficients.numel() >= 12


def test_fibre_aperture_surface_brightness_is_a_weighted_mean() -> None:
    xy, transform, wave, _ = _fibre_bundle()
    edges = torch.cat((wave - 1.0, wave[-1:] + 1.0))
    target = CubeTarget(_wcs((21, 21), scale_arcsec=0.5), edges, shape=(21, 21))
    observation = IFUObservation(
        torch.full(transform.detector_shape, 3.0, dtype=torch.float64),
        transform,
        aperture_diameter_arcsec=2.0,
    )
    result = coadd_cube([observation], target, spectral_kernel="drizzle")
    covered = result.weight > 0
    torch.testing.assert_close(
        result.data[covered], torch.full_like(result.data[covered], 3.0)
    )


@pytest.mark.parametrize(
    "device",
    [
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason="MPS unavailable"
            ),
        ),
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
    ],
)
def test_vectorised_planner_on_gpu_matches_cpu(device: str) -> None:
    dtype = torch.float32 if device == "mps" else torch.float64
    xy, transform, wave, _ = _fibre_bundle()
    edges = torch.cat((wave - 1.0, wave[-1:] + 1.0)).to(dtype)
    target = CubeTarget(_wcs((21, 21), scale_arcsec=0.5), edges, shape=(21, 21))
    science = torch.rand(transform.detector_shape, dtype=dtype)
    for kernel, footprint in (("drizzle", 2.0), ("emsm", None)):
        observation = IFUObservation(
            science, transform, aperture_diameter_arcsec=footprint
        )
        cpu = coadd_cube([observation], target, spectral_kernel=kernel)
        gpu = coadd_cube([observation], target, spectral_kernel=kernel, device=device)
        assert gpu.data.device.type == device
        assert gpu.response_plan.coefficients.device.type == device
        torch.testing.assert_close(
            gpu.data.cpu().to(cpu.data.dtype), cpu.data, rtol=1e-4, atol=1e-5
        )
    grid = _observations("flux", curved=False)
    target = _target("flux")
    for kernel in ("drizzle", "msm"):
        options = {"footprint_tolerance": 0.2} if device == "mps" else {}
        cpu = plan_ifu_cube_response(grid, target, kernel=kernel, **options)
        gpu = plan_ifu_cube_response(
            [
                IFUObservation(
                    o.sci.to(dtype),
                    o.detector_to_world,
                    dq=o.dq,
                    variance_components=o.variance_components,
                    value_semantics="flux",
                    weight=o.weight,
                )
                for o in grid
            ],
            target,
            kernel=kernel,
            device=device,
            **options,
        )
        assert gpu.plan.coefficients.device.type == device
        torch.testing.assert_close(
            gpu.weight.cpu().to(torch.float64), cpu.weight, rtol=1e-4, atol=1e-5
        )


def test_legacy_drizzle_footprint_stays_inside_slices() -> None:
    from torchsky.map import Observation, Target

    # Two slices side by side, each two pixels wide: the last pixel of the first
    # slice differences backwards, not into the second slice 3 arcsec away.
    shape = (3, 4)
    y, x = torch.meshgrid(
        torch.arange(3, dtype=torch.float64),
        torch.arange(4, dtype=torch.float64),
        indexing="ij",
    )
    slice_id = (x >= 2).to(torch.int64)
    east = x * 0.5 + 3.0 * slice_id  # arcsec; 0.5 arcsec pixels within a slice
    ifu = IFUSliceWCS(
        150.0 + east / 3600.0 / math.cos(math.radians(2.0)),
        2.0 + y * 0.5 / 3600.0,
        5000.0 + 2.0 * y,
        slice_id,
    )
    result = coadd_cube(
        [Observation(data=torch.ones(shape), ifu_wcs=ifu)],
        Target(wcs=_wcs((21, 21), scale_arcsec=0.5), shape=(21, 21)),
        wavelengths=5000.0 + 2.0 * torch.arange(3, dtype=torch.float64),
        spectral_kernel="drizzle",
        semantics="flux",
    )
    # Each 0.5 arcsec sample covers one 0.5 arcsec spaxel: no spaxel is hit by a
    # footprint stretched across the 3 arcsec gap between the slices.
    assert int((result.weight[1] > 0).sum()) <= 2 * shape[1]
