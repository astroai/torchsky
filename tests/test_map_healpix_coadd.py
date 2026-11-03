from __future__ import annotations

import math

import importlib
from typing import Any, cast

import pytest
import torch

from torchsky.map import Observation, coadd_healpix
from torchsky.map.healpix_coadd import _SortedContributionAccumulator
from torchsky.wcs import WCS


_DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA not available"
        ),
    ),
    pytest.param(
        "mps",
        marks=pytest.mark.skipif(
            not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
            reason="MPS not available",
        ),
    ),
]


def _header(
    shape: tuple[int, int] = (24, 24),
    *,
    crval: tuple[float, float] = (45.0, 10.0),
    cdelt: float = 0.2,
) -> dict[str, Any]:
    height, width = shape
    return {
        "NAXIS": 2,
        "NAXIS1": width,
        "NAXIS2": height,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": (width + 1.0) / 2.0,
        "CRPIX2": (height + 1.0) / 2.0,
        "CRVAL1": crval[0],
        "CRVAL2": crval[1],
        "CDELT1": -cdelt,
        "CDELT2": cdelt,
    }


def _observation(
    value: float,
    weight: float,
    *,
    shape: tuple[int, int] = (24, 24),
    scale: float = 1.0,
    inverse_variance: bool = False,
    mask: torch.Tensor | None = None,
    requires_grad: bool = False,
    cdelt: float = 0.2,
) -> Observation:
    header = _header(shape, cdelt=cdelt)
    return Observation(
        data=torch.full(shape, value, dtype=torch.float32, requires_grad=requires_grad),
        header=header,
        wcs=WCS(header),
        weight_map=torch.full(shape, weight, dtype=torch.float32),
        weight_is_inverse_variance=inverse_variance,
        photometric_scale=scale,
        mask=mask,
    )


def test_sorted_accumulator_matches_retain_all_for_many_repeated_pixels() -> None:
    generator = torch.Generator().manual_seed(1042)
    support = 2048
    observations = 41
    accumulator = _SortedContributionAccumulator(device=torch.device("cpu"))
    all_pixels = []
    all_numer = []
    all_denom = []

    for _ in range(observations):
        pixels = torch.randperm(support, generator=generator)[:1536]
        denom = torch.rand(1536, generator=generator) + 0.25
        values = torch.randn(1536, generator=generator)
        numer = values * denom
        for pixel_chunk, numer_chunk, denom_chunk in zip(
            pixels.split(113), numer.split(113), denom.split(113), strict=True
        ):
            accumulator.add(pixel_chunk, numer_chunk, denom_chunk)
        all_pixels.append(pixels)
        all_numer.append(numer)
        all_denom.append(denom)

    pixels = torch.cat(all_pixels)
    numer = torch.zeros(support).index_add(0, pixels, torch.cat(all_numer))
    denom = torch.zeros(support).index_add(0, pixels, torch.cat(all_denom))
    overlap = torch.zeros(support, dtype=torch.int32).index_add(
        0, pixels, torch.ones_like(pixels, dtype=torch.int32)
    )
    expected_pixels = torch.nonzero(denom > 0.0).flatten()

    torch.testing.assert_close(accumulator.pixels, expected_pixels)
    torch.testing.assert_close(
        accumulator.numer, numer[expected_pixels], rtol=0, atol=0
    )
    torch.testing.assert_close(
        accumulator.denom, denom[expected_pixels], rtol=0, atol=0
    )
    torch.testing.assert_close(accumulator.overlap, overlap[expected_pixels])


def test_sorted_accumulator_storage_is_bounded_by_distinct_support() -> None:
    generator = torch.Generator().manual_seed(9)
    accumulator = _SortedContributionAccumulator(device=torch.device("cpu"))
    support = 4096
    chunk_size = 4096
    total_hits = 131_072

    for _ in range(total_hits // chunk_size):
        pixels = torch.randperm(support, generator=generator)[:chunk_size]
        denom = torch.rand(chunk_size, generator=generator) + 0.1
        accumulator.add(pixels, denom, denom)
        assert accumulator.pixels.numel() <= support
        assert all(
            value.ndim == 1 and value.numel() <= support
            for value in vars(accumulator).values()
        )

    assert set(vars(accumulator)) == {"pixels", "numer", "denom", "overlap"}
    assert accumulator.overlap.sum().item() == total_hits


@pytest.mark.parametrize("device", _DEVICES)
def test_sorted_accumulator_preserves_device_dtype_and_gradients(device: str) -> None:
    torch_device = torch.device(device)
    first_numer = torch.tensor(
        [2.0, 6.0, 12.0], device=torch_device, requires_grad=True
    )
    second_numer = torch.tensor(
        [20.0, 30.0, 42.0], device=torch_device, requires_grad=True
    )
    accumulator = _SortedContributionAccumulator(device=torch_device)
    accumulator.add(
        torch.tensor([7, 3, 11], dtype=torch.int64, device=torch_device),
        first_numer,
        torch.tensor([1.0, 2.0, 3.0], device=torch_device),
    )
    accumulator.add(
        torch.tensor([3, 13, 7], dtype=torch.int64, device=torch_device),
        second_numer,
        torch.tensor([4.0, 5.0, 6.0], device=torch_device),
    )
    accumulator.numer.sum().backward()

    assert accumulator.pixels.device.type == device
    assert accumulator.numer.device.type == device
    assert accumulator.numer.dtype == torch.float32
    assert accumulator.overlap.dtype == torch.int32
    torch.testing.assert_close(first_numer.grad, torch.ones_like(first_numer))
    torch.testing.assert_close(second_numer.grad, torch.ones_like(second_numer))


@pytest.mark.parametrize("nest", [False, True])
def test_coadd_chunking_matches_single_chunk_and_bounds_sampling(
    monkeypatch: pytest.MonkeyPatch,
    nest: bool,
) -> None:
    module = importlib.import_module("torchsky.map.healpix_coadd")
    original = module.sample_image_plane_batched_channels
    sampled_sizes: list[int] = []

    def tracked_sample(*args: Any, **kwargs: Any) -> torch.Tensor:
        x_src = kwargs["x_src"]
        assert isinstance(x_src, torch.Tensor)
        sampled_sizes.append(x_src.numel())
        return original(*args, **kwargs)

    observations = [
        _observation(float(index + 1), float(index + 2)) for index in range(4)
    ]
    reference = coadd_healpix(
        observations,
        nside=512,
        nest=nest,
        sparse=False,
        kernel="nearest",
        reduction_chunk_size=1_000_000,
    )
    monkeypatch.setattr(module, "sample_image_plane_batched_channels", tracked_sample)
    chunked = coadd_healpix(
        observations,
        nside=512,
        nest=nest,
        sparse=False,
        kernel="nearest",
        reduction_chunk_size=17,
    )
    sparse_result = coadd_healpix(
        observations,
        nside=512,
        nest=nest,
        sparse=True,
        nside_coverage=16,
        kernel="nearest",
        reduction_chunk_size=17,
    )

    assert sampled_sizes and max(sampled_sizes) <= 17
    torch.testing.assert_close(chunked.data, reference.data, rtol=0, atol=0)
    torch.testing.assert_close(chunked.weight, reference.weight, rtol=0, atol=0)
    torch.testing.assert_close(chunked.overlap_count, reference.overlap_count)
    pixels = sparse_result.data.valid_pixels
    torch.testing.assert_close(
        sparse_result.data.get_values(pixels), chunked.data[pixels], rtol=0, atol=0
    )
    torch.testing.assert_close(
        sparse_result.weight.get_values(pixels),
        chunked.weight[pixels],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        sparse_result.overlap_count.get_values(pixels), chunked.overlap_count[pixels]
    )


def test_coadd_many_observations_preserves_weight_and_calibration_contracts() -> None:
    values = torch.arange(1, 5, dtype=torch.float32)
    weights = torch.arange(2, 6, dtype=torch.float32)
    observations = [
        _observation(
            float(value),
            float(weight),
            scale=2.0,
            inverse_variance=index % 2 == 0,
        )
        for index, (value, weight) in enumerate(zip(values, weights, strict=True))
    ]
    result = coadd_healpix(
        observations,
        nside=512,
        nest=True,
        sparse=False,
        kernel="nearest",
        reduction_chunk_size=11,
    )
    covered = result.weight > 0.0
    expected_value = (values * weights).sum() / weights.sum() * 2.0
    expected_weight = weights.sum() / 4.0

    assert bool(covered.any())
    torch.testing.assert_close(
        result.data[covered], torch.full_like(result.data[covered], expected_value)
    )
    torch.testing.assert_close(
        result.weight[covered], torch.full_like(result.weight[covered], expected_weight)
    )
    assert bool((result.overlap_count[covered] == len(observations)).all())


@pytest.mark.parametrize("nest", [False, True])
def test_empty_sparse_coadd_preserves_order_device_and_dtypes(nest: bool) -> None:
    observation = _observation(
        1.0,
        1.0,
        mask=torch.zeros((24, 24), dtype=torch.bool),
    )
    result = coadd_healpix(
        [observation],
        nside=128,
        nest=nest,
        sparse=True,
        nside_coverage=16,
        reduction_chunk_size=7,
    )

    assert result.data.nest is nest
    assert result.weight.nest is nest
    assert result.overlap_count.nest is nest
    assert result.data.values.dtype == torch.float32
    assert result.weight.values.dtype == torch.float32
    assert result.overlap_count.values.dtype == torch.int32
    assert result.data.values.device.type == observation.data.device.type
    assert result.data.valid_pixels.numel() == 0


def test_sparse_high_nside_retains_only_observed_support() -> None:
    result = coadd_healpix(
        [
            _observation(
                3.0,
                2.0,
                shape=(8, 8),
                inverse_variance=True,
                cdelt=0.005,
            )
        ],
        nside=4096,
        nest=True,
        sparse=True,
        nside_coverage=32,
        kernel="nearest",
        reduction_chunk_size=19,
    )

    pixels = result.data.valid_pixels
    assert 0 < pixels.numel() < 12 * 4096 * 4096
    assert result.data.values.numel() == pixels.numel()
    assert result.weight.values.numel() == pixels.numel()
    assert result.overlap_count.values.numel() == pixels.numel()


def test_coadd_chunked_reduction_preserves_data_gradient() -> None:
    observation = _observation(3.0, 4.0, shape=(12, 12), requires_grad=True)
    result = coadd_healpix(
        [observation],
        nside=32,
        nest=True,
        sparse=False,
        kernel="nearest",
        reduction_chunk_size=5,
    )
    covered = result.weight > 0.0
    result.data[covered].sum().backward()

    assert observation.data is not None
    assert observation.data.grad is not None
    assert observation.data.grad.dtype == torch.float32
    assert observation.data.grad.device.type == "cpu"
    assert bool(torch.isfinite(observation.data.grad).all())


@pytest.mark.parametrize("chunk_size", [0, -1, True, 1.5])
def test_coadd_rejects_invalid_reduction_chunk_size(chunk_size: object) -> None:
    with pytest.raises(ValueError, match="reduction_chunk_size"):
        coadd_healpix(
            [_observation(1.0, 1.0)],
            nside=16,
            reduction_chunk_size=cast(int, chunk_size),
        )


def test_coadd_linear_ramp_is_exact_on_every_covered_cell() -> None:
    # Bilinear sampling is only defined between the outermost pixel centres;
    # a half-pixel window would extrapolate zero-padded edge values.
    from torchsky.sphere import pix2ang

    shape = (16, 16)
    header = _header(shape, cdelt=0.1)
    wcs = WCS(header)
    yy, xx = torch.meshgrid(
        torch.arange(16, dtype=torch.float64),
        torch.arange(16, dtype=torch.float64),
        indexing="ij",
    )
    image = (2.0 + 0.5 * xx + 0.25 * yy).to(torch.float32)
    obs = Observation(data=image, wcs=wcs, weight=torch.ones_like(image))
    result = coadd_healpix([obs], nside=512, nest=True, sparse=True)
    pixels = result.data.valid_pixels.to(torch.int64)
    assert pixels.numel() > 0
    values = result.data.get_values(pixels).to(torch.float64)
    ra, dec = pix2ang(512, pixels, nest=True, lonlat=True)
    x, y = wcs.world_to_pixel(ra, dec)
    assert float(x.min()) >= -1e-4 and float(x.max()) <= 15 + 1e-4
    assert float(y.min()) >= -1e-4 and float(y.max()) <= 15 + 1e-4
    truth = 2.0 + 0.5 * x.to(torch.float64) + 0.25 * y.to(torch.float64)
    assert float((values - truth).abs().max()) < 1e-3


def test_coadd_healpix_cells_larger_than_image_integrate_overlap() -> None:
    # A 0.64 deg image inside 1.83 deg cells (nside 32): no cell centre falls in
    # the image, so each cell gets the overlap integral, not an error.
    from torchsky.map._healpix_overlap import _spherical_quad_area, _world

    header = _header((32, 32), cdelt=0.02)
    obs = Observation(data=torch.full((32, 32), 3.0), header=header, wcs=WCS(header))
    dense = coadd_healpix([obs], nside=32)
    sparse = coadd_healpix([obs], nside=32, sparse=True, nest=True)
    covered = dense.weight > 0.0
    assert 1 <= int(covered.sum()) <= 4
    torch.testing.assert_close(
        dense.data[covered], torch.full_like(dense.data[covered], 3.0)
    )
    # Weights are covered fractions of each cell: they sum to the image area.
    edges = torch.arange(33, dtype=torch.float64) - 0.5
    yy, xx = torch.meshgrid(edges, edges, indexing="ij")
    lon, lat = _world(WCS(header), xx.reshape(-1), yy.reshape(-1))
    lon, lat = lon.view(33, 33), lat.view(33, 33)

    def quads(a: torch.Tensor) -> torch.Tensor:
        return torch.stack((a[:-1, :-1], a[:-1, 1:], a[1:, 1:], a[1:, :-1]), -1)

    area = _spherical_quad_area(quads(lon).reshape(-1, 4), quads(lat).reshape(-1, 4))
    cell = 4.0 * math.pi / (12 * 32**2)
    assert abs(float(dense.weight.sum()) - float(area.sum()) / cell) < 1e-6
    pixels = sparse.weight.valid_pixels
    assert int(pixels.numel()) == int(covered.sum())
    total = float(sparse.weight.get_values(pixels).sum())
    assert abs(total - float(dense.weight.sum())) < 1e-6


def _fk5_obs(radesys: str | None) -> Observation:
    import warnings

    header = _header((24, 24), cdelt=0.01)
    if radesys is not None:
        header["RADESYS"] = radesys
        header["EQUINOX"] = 2000.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yy, xx = torch.meshgrid(torch.arange(24.0), torch.arange(24.0), indexing="ij")
        blob = torch.exp(-((xx - 11.5) ** 2 + (yy - 11.5) ** 2) / (2 * 3.0**2))
        return Observation(data=blob, header=header, wcs=WCS(header))


def test_coadd_healpix_fk5_header_gets_frame_bias() -> None:
    from torchsky.sphere import healpix as hp
    from torchsky.wcs import CelestialFrameBiasWarning

    nside = 2**15
    with pytest.warns(CelestialFrameBiasWarning):
        fk5 = coadd_healpix([_fk5_obs("FK5")], nside=nside, sparse=True, nest=True)
    icrs = coadd_healpix([_fk5_obs("ICRS")], nside=nside, sparse=True, nest=True)

    def centroid(res: Any) -> tuple[torch.Tensor, torch.Tensor]:
        pix = res.data.valid_pixels
        w = res.data.values.double()
        ra, dec = hp.pix2ang(nside, pix, nest=True, lonlat=True)
        return (ra.double() * w).sum() / w.sum(), (dec.double() * w).sum() / w.sum()

    ra0, dec0 = centroid(icrs)
    ra1, dec1 = centroid(fk5)
    offset_mas = (
        torch.hypot((ra1 - ra0) * torch.cos(torch.deg2rad(dec0)), dec1 - dec0) * 3.6e6
    )
    assert 10.0 < float(offset_mas) < 40.0
