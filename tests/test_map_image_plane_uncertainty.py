from __future__ import annotations
from pathlib import Path
import pytest
import torch
from conftest import assert_close_on_device
import torchfits
import torchsky
import torchsky.map.image_plane_coadd as image_plane_coadd_mod
import torchsky.map.operators.resampling as resampling_mod
from torchsky.map import aperture_uncertainty
from torchsky.map.image_plane_coadd import (
    coadd_image_plane,
)
from torchsky.map.observation import ImageObservation

pytest.importorskip("torchsky._cpp")
from map_image_plane_support import (
    _simple_header,
)


@pytest.mark.parametrize("kernel", ["nearest", "bilinear", "bicubic"])
def test_coadd_image_plane_passes_resampling_kernel_to_grid_sample(
    tmp_path, monkeypatch, kernel: str
) -> None:
    path_a = tmp_path / f"a_{kernel}.fits"
    path_b = tmp_path / f"b_{kernel}.fits"
    header_a = _simple_header()
    header_b = dict(header_a)
    header_b["CRVAL1"] = float(header_b["CRVAL1"]) + 0.02
    data = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    torchfits.write(str(path_a), data, header=header_a, overwrite=True)
    torchfits.write(str(path_b), data + 1.0, header=header_b, overwrite=True)

    calls: list[str] = []
    original = resampling_mod.F.grid_sample

    def recorded(*args, **kwargs):
        calls.append(kwargs["mode"])
        return original(*args, **kwargs)

    monkeypatch.setattr(resampling_mod.F, "grid_sample", recorded)

    result = coadd_image_plane(
        [ImageObservation(str(path_a)), ImageObservation(str(path_b))],
        resampling_kernel=kernel,
    )

    assert result.data.shape == (8, 8)
    assert calls
    assert set(calls) == {kernel}


@pytest.mark.parametrize("kernel", ["bilinear", "catmull_rom"])
def test_sample_image_plane_batched_channels_matches_separate_sampling(
    kernel: str,
) -> None:
    image = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    weight = 1.0 + torch.flip(image, dims=(0,)) / 64.0
    support = torch.ones_like(image, dtype=torch.float32)
    support[:, :2] = 0.0
    x_src = torch.linspace(0.25, 6.75, 8, dtype=torch.float32).repeat(8, 1)
    y_src = torch.linspace(0.1, 6.9, 8, dtype=torch.float32).unsqueeze(1).repeat(1, 8)
    grid = torch.stack(
        [
            (2.0 * x_src / 7.0) - 1.0,
            (2.0 * y_src / 7.0) - 1.0,
        ],
        dim=-1,
    )
    stacked = torch.stack([image, weight, support], dim=0)

    combined = resampling_mod.sample_image_plane_batched_channels(
        stacked,
        x_src=x_src,
        y_src=y_src,
        grid=grid,
        kernel=kernel,
    )
    separate = torch.stack(
        [
            resampling_mod.sample_image_plane_batched(
                plane,
                x_src=x_src,
                y_src=y_src,
                grid=grid,
                kernel=kernel,
            )
            for plane in (image, weight, support)
        ],
        dim=0,
    )

    assert_close_on_device(combined, separate, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("kernel", ["lanczos3", "lanczos5"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_lanczos_sampling_preserves_constant_on_complete_support(
    kernel: str, dtype
) -> None:
    image = torch.ones((16, 16), dtype=dtype)
    x_src = torch.tensor([[5.25, 8.6], [6.7, 9.1]], dtype=dtype)
    y_src = torch.tensor([[6.4, 7.2], [8.3, 5.8]], dtype=dtype)
    grid = torch.stack(
        [
            (2.0 * x_src / 15.0) - 1.0,
            (2.0 * y_src / 15.0) - 1.0,
        ],
        dim=-1,
    )

    sampled = resampling_mod.sample_image_plane(
        image,
        x_src=x_src,
        y_src=y_src,
        grid=grid,
        kernel=kernel,
    )

    # The dense tensor reduction rounds each signed stencil term. Its forward
    # error bound is gamma_n * sum(abs(weights)), not a platform-specific ULP.
    support = resampling_mod.kernel_support(kernel)
    offsets = torch.arange(-support + 1, support + 1, dtype=dtype)
    norms = []
    for coordinate in (x_src, y_src):
        delta = (coordinate - coordinate.floor()).unsqueeze(-1) - offsets
        coefficients = resampling_mod.kernel_weight(kernel, delta)
        coefficients = coefficients / coefficients.sum(-1, keepdim=True)
        norms.append(coefficients.abs().sum(-1))
    operations = (2 * support) ** 2 + 4 * support
    error = operations * torch.finfo(dtype).eps
    bound = error / (1 - error) * float((norms[0] * norms[1]).max())
    assert_close_on_device(sampled, torch.ones_like(sampled), atol=bound, rtol=0.0)


@pytest.mark.parametrize(
    "kernel",
    ["nearest", "bilinear", "bicubic", "lanczos3", "catmull_rom"],
)
def test_sample_image_plane_diagonal_variance_matches_basis_oracle(kernel) -> None:
    height, width = 5, 6
    source_variance = torch.arange(
        1,
        height * width + 1,
        dtype=torch.float32,
    ).reshape(height, width)
    x_src = torch.tensor([[1.25, 3.6], [0.2, 4.75]], dtype=torch.float32)
    y_src = torch.tensor([[2.4, 1.1], [3.7, 0.3]], dtype=torch.float32)
    grid = torch.stack(
        [
            (2.0 * x_src / (width - 1)) - 1.0,
            (2.0 * y_src / (height - 1)) - 1.0,
        ],
        dim=-1,
    )
    basis = torch.eye(height * width, dtype=torch.float32).reshape(
        height * width,
        height,
        width,
    )
    coefficients = resampling_mod.sample_image_plane_batched_channels(
        basis,
        x_src=x_src,
        y_src=y_src,
        grid=grid,
        kernel=kernel,
    )
    expected = (torch.square(coefficients) * source_variance.reshape(-1, 1, 1)).sum(
        dim=0
    )

    actual = resampling_mod.sample_image_plane_diagonal_variance(
        source_variance,
        x_src=x_src,
        y_src=y_src,
        kernel=kernel,
    )

    assert_close_on_device(actual, expected, atol=2.0e-6, rtol=2.0e-6)


def test_batched_quadrature_variance_squares_mean_coefficients() -> None:
    height, width = 5, 6
    source_variance = torch.arange(
        1,
        height * width + 1,
        dtype=torch.float32,
    ).reshape(height, width)
    x_src = torch.tensor(
        [
            [[1.2, 3.4], [2.1, 4.2]],
            [[1.7, 3.8], [2.6, 4.6]],
        ],
        dtype=torch.float32,
    )
    y_src = torch.tensor(
        [
            [[2.3, 1.2], [3.1, 0.4]],
            [[2.7, 1.6], [3.5, 0.8]],
        ],
        dtype=torch.float32,
    )
    sample_valid = torch.tensor(
        [
            [[True, True], [True, False]],
            [[True, False], [True, True]],
        ]
    )
    grid = torch.stack(
        [
            (2.0 * x_src / (width - 1)) - 1.0,
            (2.0 * y_src / (height - 1)) - 1.0,
        ],
        dim=-1,
    )
    basis = torch.eye(height * width, dtype=torch.float32).reshape(
        height * width,
        height,
        width,
    )
    sampled_basis = resampling_mod.sample_image_plane_batched_channels(
        basis,
        x_src=x_src,
        y_src=y_src,
        grid=grid,
        kernel="bilinear",
    )
    mean_coefficients = (
        sampled_basis * sample_valid.unsqueeze(1).to(dtype=torch.float32)
    ).mean(dim=0)
    expected = (
        torch.square(mean_coefficients) * source_variance.reshape(-1, 1, 1)
    ).sum(dim=0)

    actual = resampling_mod.sample_image_plane_diagonal_variance(
        source_variance,
        x_src=x_src,
        y_src=y_src,
        kernel="bilinear",
        average_coordinate_batch=True,
        sample_valid=sample_valid,
    )

    assert_close_on_device(actual, expected, atol=2.0e-6, rtol=2.0e-6)


def test_bilinear_diagonal_variance_matches_monte_carlo() -> None:
    generator = torch.Generator().manual_seed(20260831)
    draws = 20_000
    height, width = 4, 5
    source_variance = torch.linspace(
        0.5,
        2.0,
        height * width,
        dtype=torch.float32,
    ).reshape(height, width)
    x_src = torch.full((1, 1), 2.3, dtype=torch.float32)
    y_src = torch.full((1, 1), 1.4, dtype=torch.float32)
    expected = resampling_mod.sample_image_plane_diagonal_variance(
        source_variance,
        x_src=x_src,
        y_src=y_src,
        kernel="bilinear",
    ).squeeze()
    noise = torch.randn(
        (draws, height, width),
        generator=generator,
        dtype=torch.float32,
    ) * torch.sqrt(source_variance)
    x_batch = x_src.expand(draws, 1, 1)
    y_batch = y_src.expand(draws, 1, 1)
    grid = torch.stack(
        [
            (2.0 * x_batch / (width - 1)) - 1.0,
            (2.0 * y_batch / (height - 1)) - 1.0,
        ],
        dim=-1,
    )
    samples = resampling_mod.sample_image_plane_batched_channels(
        noise.unsqueeze(1),
        x_src=x_batch,
        y_src=y_batch,
        grid=grid,
        kernel="bilinear",
    ).reshape(draws)

    empirical = torch.var(samples, unbiased=True)
    assert float(empirical) == pytest.approx(float(expected), rel=0.03)


def test_bilinear_local_correlation_factor_predicts_large_aperture_variance() -> None:
    generator = torch.Generator().manual_seed(20260901)
    draws = 10_000
    width = 64
    aperture_pixels = width - 1
    x_src = torch.arange(0.5, width - 0.5, dtype=torch.float32).unsqueeze(0)
    y_src = torch.zeros_like(x_src)
    source_variance = torch.ones((2, width), dtype=torch.float32)
    diagonal = resampling_mod.sample_image_plane_diagonal_variance(
        source_variance,
        x_src=x_src,
        y_src=y_src,
        kernel="bilinear",
    )
    local_factor = diagonal.reciprocal()
    assert_close_on_device(local_factor, torch.full_like(local_factor, 2.0))

    noise = torch.randn(
        (draws, 2, width),
        generator=generator,
        dtype=torch.float32,
    )
    x_batch = x_src.expand(draws, 1, aperture_pixels)
    y_batch = y_src.expand(draws, 1, aperture_pixels)
    grid = torch.stack(
        [
            (2.0 * x_batch / (width - 1)) - 1.0,
            y_batch * 2.0 - 1.0,
        ],
        dim=-1,
    )
    samples = resampling_mod.sample_image_plane_batched_channels(
        noise.unsqueeze(1),
        x_src=x_batch,
        y_src=y_batch,
        grid=grid,
        kernel="bilinear",
    ).squeeze(1)
    aperture_sums = samples.sum(dim=(-2, -1))
    empirical_factor = torch.var(aperture_sums, unbiased=True) / diagonal.sum()
    exact_finite_aperture_factor = 2.0 - 1.0 / float(aperture_pixels)

    assert float(empirical_factor) == pytest.approx(
        exact_finite_aperture_factor,
        rel=0.03,
    )


def test_finite_aperture_uncertainty_keeps_covariance_across_tiles(
    tmp_path: Path,
) -> None:
    source_header = _simple_header()
    target_header = {**source_header, "CRPIX1": source_header["CRPIX1"] + 0.5}
    path = tmp_path / "half-pixel-noise.fits"
    torchfits.write(
        str(path),
        torch.zeros((8, 8), dtype=torch.float32),
        header=source_header,
        overwrite=True,
    )
    observations = [
        ImageObservation(
            str(path),
            weight=1.0,
            weight_is_inverse_variance=True,
        )
    ]
    result = coadd_image_plane(
        observations,
        target_header=target_header,
        resampling_kernel="bilinear",
        spatial_tile_hw=(2, 3),
    )
    aperture = torch.zeros_like(result.data)
    aperture[3, 1:7] = 1.0

    with torch.no_grad():
        uncertainty = aperture_uncertainty(
            observations,
            result,
            aperture,
            kernel="bilinear",
            spatial_tile_hw=(2, 3),
        )

    aperture_pixels = 6
    expected_factor = 2.0 - 1.0 / aperture_pixels
    assert float(uncertainty.diagonal_variance) == pytest.approx(
        0.5 * aperture_pixels,
        rel=0.0,
        abs=2.0e-5,
    )
    assert float(uncertainty.variance) == pytest.approx(
        aperture_pixels - 0.5,
        rel=0.0,
        abs=3.0e-5,
    )
    assert float(uncertainty.correlation_factor) == pytest.approx(
        expected_factor,
        rel=0.0,
        abs=2.0e-5,
    )

    generator = torch.Generator().manual_seed(20260901)
    draws = 10_000
    target_x = torch.arange(1, 7, dtype=torch.float32)
    target_y = torch.full_like(target_x, 3.0)
    ra, dec = torchsky.WCS(target_header).pixel_to_world(target_x, target_y)
    source_x, source_y = torchsky.WCS(source_header).world_to_pixel(ra, dec)
    x_batch = source_x.expand(draws, 1, -1)
    y_batch = source_y.expand(draws, 1, -1)
    grid = torch.stack(
        [
            (2.0 * x_batch / 7.0) - 1.0,
            (2.0 * y_batch / 7.0) - 1.0,
        ],
        dim=-1,
    )
    detector_noise = torch.randn(
        (draws, 8, 8), generator=generator, dtype=torch.float32
    )
    resampled = resampling_mod.sample_image_plane_batched_channels(
        detector_noise.unsqueeze(1),
        x_src=x_batch,
        y_src=y_batch,
        grid=grid,
        kernel="bilinear",
    )
    empirical_variance = torch.var(resampled.sum(dim=(-2, -1)), unbiased=True)
    assert float(empirical_variance) == pytest.approx(
        float(uncertainty.variance), rel=0.03
    )


def test_aperture_uncertainty_reuses_call_scoped_target_tiles(monkeypatch) -> None:
    source_header = _simple_header()
    target_header = {**source_header, "CRPIX1": source_header["CRPIX1"] + 0.25}
    observations = [
        ImageObservation(
            data=torch.zeros((8, 8), dtype=torch.float32),
            header=source_header,
            weight=1.0,
            weight_is_inverse_variance=True,
        )
        for _ in range(2)
    ]
    result = coadd_image_plane(
        observations,
        target_header=target_header,
        resampling_kernel="bilinear",
        spatial_tile_hw=(4, 4),
    )
    aperture = (result.weight > 0.0).to(dtype=result.data.dtype)
    original_prepare = image_plane_coadd_mod._prepare_canvas_tile
    prepare_calls = 0

    def counted_prepare(*args, **kwargs):
        nonlocal prepare_calls
        prepare_calls += 1
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(image_plane_coadd_mod, "_prepare_canvas_tile", counted_prepare)
    cached = aperture_uncertainty(
        observations,
        result,
        aperture,
        kernel="bilinear",
        spatial_tile_hw=(4, 4),
        target_cache_capacity=4,
    )
    assert prepare_calls == 4

    prepare_calls = 0
    uncached = aperture_uncertainty(
        observations,
        result,
        aperture,
        kernel="bilinear",
        spatial_tile_hw=(4, 4),
        target_cache_capacity=0,
    )
    assert prepare_calls == 8
    assert_close_on_device(cached.variance, uncached.variance)
    assert_close_on_device(cached.diagonal_variance, uncached.diagonal_variance)


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_finite_aperture_uncertainty_device_parity(
    tmp_path: Path,
    device: str,
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    source_header = _simple_header()
    target_header = {**source_header, "CRPIX1": source_header["CRPIX1"] + 0.5}
    path = tmp_path / f"aperture-{device}.fits"
    torchfits.write(
        str(path), torch.zeros((8, 8)), header=source_header, overwrite=True
    )
    observations = [
        ImageObservation(str(path), weight=1.0, weight_is_inverse_variance=True)
    ]
    cpu_result = coadd_image_plane(
        observations,
        target_header=target_header,
        resampling_kernel="bilinear",
        device="cpu",
    )
    accelerator_result = coadd_image_plane(
        observations,
        target_header=target_header,
        resampling_kernel="bilinear",
        device=device,
    )
    aperture = torch.zeros((8, 8), dtype=torch.float32)
    aperture[3, 1:7] = 1.0

    cpu = aperture_uncertainty(
        observations,
        cpu_result,
        aperture,
        kernel="bilinear",
        spatial_tile_hw=(2, 3),
        device="cpu",
    )
    accelerator = aperture_uncertainty(
        observations,
        accelerator_result,
        aperture.to(device),
        kernel="bilinear",
        spatial_tile_hw=(2, 3),
        device=device,
    )

    assert_close_on_device(
        accelerator.variance.cpu(),
        cpu.variance.to(dtype=accelerator.variance.dtype),
        rtol=5.0e-5,
        atol=5.0e-6,
    )


@pytest.mark.parametrize(
    "kernel",
    ["bilinear", "drizzle_square", "conservative_polygon", "splat_box"],
)
@pytest.mark.needs_float64("conservative_polygon", "drizzle_square", "splat_box")
def test_finite_aperture_uncertainty_matches_source_basis_oracle(
    tmp_path: Path,
    kernel: str,
) -> None:
    source_header = {**_simple_header(), "NAXIS1": 4, "NAXIS2": 4}
    target_header = {**source_header, "CRPIX1": source_header["CRPIX1"] + 0.35}
    path = tmp_path / f"aperture-{kernel}.fits"
    image = torch.zeros((4, 4), dtype=torch.float32)
    torchfits.write(str(path), image, header=source_header, overwrite=True)
    observations = [
        ImageObservation(
            str(path),
            weight=1.0,
            weight_is_inverse_variance=True,
        )
    ]
    result = coadd_image_plane(
        observations,
        target_header=target_header,
        resampling_kernel=kernel,
        drizzle_pixfrac=0.8,
    )
    aperture = torch.zeros_like(result.data)
    aperture[1:3, 1:3] = torch.tensor([[1.0, 0.75], [0.5, -0.25]], dtype=aperture.dtype)

    uncertainty = aperture_uncertainty(
        observations,
        result,
        aperture,
        kernel=kernel,
        pixfrac=0.8,
        spatial_tile_hw=(2, 2),
    )

    responses = []
    for source_index in range(image.numel()):
        basis = torch.zeros_like(image)
        basis.reshape(-1)[source_index] = 1.0
        torchfits.write(str(path), basis, header=source_header, overwrite=True)
        basis_result = coadd_image_plane(
            observations,
            target_header=target_header,
            resampling_kernel=kernel,
            drizzle_pixfrac=0.8,
        )
        responses.append(torch.sum(aperture * basis_result.data))
    expected = torch.sum(torch.square(torch.stack(responses)))

    assert_close_on_device(
        uncertainty.variance,
        expected.to(dtype=uncertainty.variance.dtype),
        rtol=2.0e-5,
        atol=2.0e-6,
    )


@pytest.mark.parametrize(
    "kernel",
    sorted(image_plane_coadd_mod._FINITE_APERTURE_COVARIANCE_KERNELS),
)
@pytest.mark.needs_float64(
    "conservative_polygon",
    "drizzle_gaussian",
    "drizzle_lanczos2",
    "drizzle_lanczos3",
    "drizzle_point",
    "drizzle_square",
    "drizzle_turbo",
    "splat_box",
    "splat_point",
)
def test_finite_aperture_uncertainty_single_pixel_matches_diagonal_plane(
    tmp_path: Path,
    kernel: str,
) -> None:
    source_header = _simple_header()
    target_header = {**source_header, "CRPIX1": source_header["CRPIX1"] + 0.2}
    path = tmp_path / f"single-pixel-{kernel}.fits"
    torchfits.write(
        str(path), torch.zeros((8, 8)), header=source_header, overwrite=True
    )
    observations = [
        ImageObservation(str(path), weight=2.0, weight_is_inverse_variance=True)
    ]
    result = coadd_image_plane(
        observations,
        target_header=target_header,
        resampling_kernel=kernel,
        drizzle_pixfrac=0.7,
    )
    aperture = torch.zeros_like(result.data)
    aperture[3, 3] = 1.0

    uncertainty = aperture_uncertainty(
        observations,
        result,
        aperture,
        kernel=kernel,
        pixfrac=0.7,
        spatial_tile_hw=(3, 3),
    )

    assert result.inverse_variance is not None
    expected = result.inverse_variance[3, 3].reciprocal()
    assert_close_on_device(
        uncertainty.variance,
        expected.to(dtype=uncertainty.variance.dtype),
        rtol=3.0e-5,
        atol=3.0e-6,
    )


def test_finite_aperture_uncertainty_matches_distorted_wcs_basis_oracle(
    tmp_path: Path,
) -> None:
    source_header = {
        **_simple_header(),
        "NAXIS1": 4,
        "NAXIS2": 4,
        "CTYPE1": "RA---TAN-SIP",
        "CTYPE2": "DEC--TAN-SIP",
        "A_ORDER": 2,
        "B_ORDER": 2,
        "A_2_0": 2.0e-2,
        "A_1_1": -1.0e-2,
        "B_0_2": -1.5e-2,
        "B_1_1": 1.0e-2,
    }
    target_header = {**source_header, "CRPIX1": source_header["CRPIX1"] + 0.3}
    path = tmp_path / "aperture-sip.fits"
    image = torch.zeros((4, 4), dtype=torch.float32)
    torchfits.write(str(path), image, header=source_header, overwrite=True)
    observations = [
        ImageObservation(str(path), weight=1.0, weight_is_inverse_variance=True)
    ]
    result = coadd_image_plane(
        observations,
        target_header=target_header,
        resampling_kernel="bilinear",
    )
    aperture = torch.zeros_like(result.data)
    aperture[1:3, 1:3] = 1.0
    uncertainty = aperture_uncertainty(
        observations,
        result,
        aperture,
        kernel="bilinear",
        spatial_tile_hw=(2, 2),
    )

    responses = []
    for source_index in range(image.numel()):
        basis = torch.zeros_like(image)
        basis.reshape(-1)[source_index] = 1.0
        torchfits.write(str(path), basis, header=source_header, overwrite=True)
        basis_result = coadd_image_plane(
            observations,
            target_header=target_header,
            resampling_kernel="bilinear",
        )
        responses.append(torch.sum(aperture * basis_result.data))
    expected = torch.sum(torch.square(torch.stack(responses)))

    assert_close_on_device(
        uncertainty.variance,
        expected.to(dtype=uncertainty.variance.dtype),
        rtol=3.0e-5,
        atol=3.0e-6,
    )


def test_distorted_surface_brightness_area_scaling_is_tile_invariant(
    tmp_path: Path,
) -> None:
    source_header = {
        **_simple_header(),
        "CTYPE1": "RA---TAN-SIP",
        "CTYPE2": "DEC--TAN-SIP",
        "A_ORDER": 2,
        "B_ORDER": 2,
        "A_2_0": 2.0e-3,
        "B_0_2": -1.5e-3,
    }
    target_header = {**source_header, "CRPIX1": source_header["CRPIX1"] + 0.3}
    path = tmp_path / "tile-area-sip.fits"
    torchfits.write(
        str(path),
        torch.arange(64, dtype=torch.float32).reshape(8, 8),
        header=source_header,
        overwrite=True,
    )
    observations = [
        ImageObservation(str(path), weight=2.0, weight_is_inverse_variance=True)
    ]

    full = coadd_image_plane(
        observations,
        target_header=target_header,
        resampling_kernel="bilinear",
    )
    tiled = coadd_image_plane(
        observations,
        target_header=target_header,
        resampling_kernel="bilinear",
        spatial_tile_hw=(3, 3),
    )

    assert_close_on_device(tiled.data, full.data, rtol=2.0e-5, atol=2.0e-5)
    assert_close_on_device(tiled.weight, full.weight, rtol=2.0e-5, atol=2.0e-5)
    assert tiled.inverse_variance is not None
    assert full.inverse_variance is not None
    assert_close_on_device(
        tiled.inverse_variance,
        full.inverse_variance,
        rtol=3.0e-5,
        atol=3.0e-5,
    )


def test_flux_aperture_uncertainty_matches_source_basis_oracle(
    tmp_path: Path,
) -> None:
    source_header = {**_simple_header(), "NAXIS1": 4, "NAXIS2": 4}
    target_header = {**source_header, "CRPIX1": source_header["CRPIX1"] + 0.25}
    path = tmp_path / "aperture-flux.fits"
    image = torch.zeros((4, 4), dtype=torch.float32)
    torchfits.write(str(path), image, header=source_header, overwrite=True)
    observations = [
        ImageObservation(str(path), weight=3.0, weight_is_inverse_variance=True)
    ]
    result = coadd_image_plane(
        observations,
        target_header=target_header,
        resampling_kernel="bilinear",
        value_semantics="flux",
    )
    aperture = torch.zeros_like(result.data)
    aperture[1:3, 1:3] = 1.0
    uncertainty = aperture_uncertainty(
        observations,
        result,
        aperture,
        kernel="bilinear",
        semantics="flux",
        spatial_tile_hw=(2, 2),
    )

    responses = []
    for source_index in range(image.numel()):
        basis = torch.zeros_like(image)
        basis.reshape(-1)[source_index] = 1.0
        torchfits.write(str(path), basis, header=source_header, overwrite=True)
        basis_result = coadd_image_plane(
            observations,
            target_header=target_header,
            resampling_kernel="bilinear",
            value_semantics="flux",
        )
        responses.append(torch.sum(aperture * basis_result.data))
    expected = torch.sum(torch.square(torch.stack(responses))) / 3.0

    assert_close_on_device(
        uncertainty.variance,
        expected.to(dtype=uncertainty.variance.dtype),
        rtol=3.0e-5,
        atol=3.0e-6,
    )


def test_aperture_uncertainty_matches_spatial_noise_map_basis_oracle(
    tmp_path: Path,
) -> None:
    source_header = {**_simple_header(), "NAXIS1": 4, "NAXIS2": 4}
    target_header = {**source_header, "CRPIX2": source_header["CRPIX2"] + 0.4}
    image_path = tmp_path / "aperture-spatial-noise.fits"
    weight_path = tmp_path / "aperture-spatial-ivar.fits"
    image = torch.zeros((4, 4), dtype=torch.float32)
    source_inverse_variance = torch.linspace(0.5, 2.0, 16).reshape(4, 4)
    torchfits.write(str(image_path), image, header=source_header, overwrite=True)
    torchfits.write(
        str(weight_path),
        source_inverse_variance,
        header=source_header,
        overwrite=True,
    )
    observations = [
        ImageObservation(
            str(image_path),
            weight_path=str(weight_path),
            weight_is_inverse_variance=True,
        )
    ]
    result = coadd_image_plane(
        observations,
        target_header=target_header,
        resampling_kernel="bilinear",
    )
    aperture = torch.zeros_like(result.data)
    aperture[1:3, 1:3] = torch.tensor([[1.0, -0.5], [0.25, 0.75]], dtype=aperture.dtype)
    uncertainty = aperture_uncertainty(
        observations,
        result,
        aperture,
        kernel="bilinear",
        spatial_tile_hw=(2, 2),
    )

    contributions = []
    for source_index in range(image.numel()):
        basis = torch.zeros_like(image)
        basis.reshape(-1)[source_index] = 1.0
        torchfits.write(str(image_path), basis, header=source_header, overwrite=True)
        basis_result = coadd_image_plane(
            observations,
            target_header=target_header,
            resampling_kernel="bilinear",
        )
        response = torch.sum(aperture * basis_result.data)
        contributions.append(
            torch.square(response) / source_inverse_variance.reshape(-1)[source_index]
        )
    expected = torch.sum(torch.stack(contributions))

    assert_close_on_device(
        uncertainty.variance,
        expected.to(dtype=uncertainty.variance.dtype),
        rtol=3.0e-5,
        atol=3.0e-6,
    )


def test_finite_aperture_uncertainty_rejects_missing_noise_and_global_kernels(
    tmp_path: Path,
) -> None:
    header = _simple_header()
    path = tmp_path / "no-noise.fits"
    torchfits.write(str(path), torch.zeros((8, 8)), header=header, overwrite=True)
    result = coadd_image_plane([ImageObservation(str(path))])
    aperture = torch.ones_like(result.data)

    with pytest.raises(ValueError, match="explicit detector inverse variance"):
        aperture_uncertainty(
            [ImageObservation(str(path))],
            result,
            aperture,
        )

    noisy = [ImageObservation(str(path), weight=1.0, weight_is_inverse_variance=True)]
    noisy_result = coadd_image_plane(noisy)
    changed_noise = [
        ImageObservation(str(path), weight=2.0, weight_is_inverse_variance=True)
    ]
    with pytest.raises(ValueError, match="observation-operator provenance"):
        aperture_uncertainty(
            changed_noise,
            noisy_result,
            aperture,
        )
    with pytest.raises(ValueError, match="execution provenance does not match"):
        aperture_uncertainty(
            noisy,
            noisy_result,
            aperture,
            kernel="nearest",
        )
    with pytest.raises(ValueError, match="finite-support"):
        aperture_uncertainty(
            noisy,
            noisy_result,
            aperture,
            kernel="fft_shift",
        )

    background_result = coadd_image_plane(
        noisy,
        background_estimator=lambda image, _mask: torch.full_like(image, image.mean()),
    )
    with pytest.raises(ValueError, match="data-dependent background"):
        aperture_uncertainty(
            noisy,
            background_result,
            aperture,
        )
