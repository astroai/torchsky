"""Verification tests for exact mathematical adjoints of the map projection operators."""

from __future__ import annotations

import pytest
import torch

from torchsky import WCS
from torchsky.map.healpix_image_plane import (
    accumulate_image_on_healpix_map_adjoint,
    sample_healpix_map_on_image_header,
)
from torchsky.map.operators import (
    ConservativePolygonRemapOp,
    ExactDrizzleGeometryOp,
    ForwardSplatGeometryOp,
    InterpolationGeometryOp,
    apply_kernel2d,
    gaussian_blur,
    polynomial_background,
    project_image_observation_geometry,
    pixel_area_sr_map,
    representative_pixel_area_sr,
)
from torchsky.map.operators.radiometry import apply_image_observation_calibration
from torchsky.map.windowing import _prepare_target
from torchsky.map.image_plane import (
    _resample_weighted_to_target,
    _forward_project_weighted_to_target,
)

pytest.importorskip("torchsky.cpp")

_DEVICES = ["cpu"]
if torch.cuda.is_available():
    _DEVICES.append("cuda")
# MPS excluded — float64 required by WCS grid arithmetic and gradcheck


def _simple_header(
    nx=8, ny=8, crval1=12.0, crval2=-3.0, cdelt1=-0.01, cdelt2=0.01
) -> dict:
    return {
        "NAXIS": 2,
        "NAXIS1": nx,
        "NAXIS2": ny,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": float(nx + 1) / 2.0,
        "CRPIX2": float(ny + 1) / 2.0,
        "CRVAL1": crval1,
        "CRVAL2": crval2,
        "CDELT1": cdelt1,
        "CDELT2": cdelt2,
    }


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("kernel_name", ["nearest", "bilinear", "lanczos3"])
@pytest.mark.parametrize("value_semantics", ["surface_brightness", "flux"])
def test_interpolation_resampler_adjoint(
    kernel_name: str, value_semantics: str, device: str
) -> None:
    dev = torch.device(device)
    src_h, src_w = 8, 8
    tgt_h, tgt_w = 6, 6

    # Slightly shifted WCS to ensure actual sub-pixel interpolation occurs
    src_header = _simple_header(nx=src_w, ny=src_h, crval1=12.00, crval2=-3.00)
    tgt_header = _simple_header(nx=tgt_w, ny=tgt_h, crval1=12.01, crval2=-2.99)

    src_wcs = WCS(src_header)
    tgt_wcs = WCS(tgt_header)

    target = _prepare_target(
        header=tgt_header,
        target_wcs=tgt_wcs,
        target_shape=(tgt_h, tgt_w),
        device=dev,
        include_pixel_area=(value_semantics == "flux"),
        include_representative_pixel_area=True,
    )

    # x represents the source image pixel values, requires_grad=True to compute adjoint via autograd
    x = torch.randn(
        (src_h, src_w), dtype=torch.float32, device=dev, requires_grad=True
    )
    source_weight = torch.ones((src_h, src_w), dtype=torch.float32, device=dev)

    y_pred, _ = project_image_observation_geometry(
        x,
        source_weight,
        src_wcs,
        target,
        geometry=InterpolationGeometryOp(),
        resampling_kernel=kernel_name,
        value_semantics=value_semantics,
        drizzle_pixfrac=1.0,
        pixel_area_sr_map_fn=pixel_area_sr_map,
        representative_pixel_area_fn=representative_pixel_area_sr,
        chunk_pixels=256,
        resample_weighted_to_target_fn=_resample_weighted_to_target,
        forward_project_weighted_to_target_fn=_forward_project_weighted_to_target,
    )

    # Ensure y_pred is not trivially zero
    assert torch.any(y_pred != 0.0), (
        f"y_pred is completely zero for {kernel_name} ({value_semantics})"
    )

    # y is a random target-space dual vector
    y = torch.randn_like(y_pred)

    # Dot product in target space: <A(x), y>
    val1 = torch.sum(y_pred * y)

    # Compute gradient (adjoint): A^T(y)
    val1.backward()

    # x.grad is A^T(y). Dot product in source space: <x, A^T(y)>
    val2 = torch.sum(x * x.grad)

    # Verify <A(x), y> == <x, A^T(y)>
    torch.testing.assert_close(val1, val2, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("value_semantics", ["surface_brightness", "flux"])
def test_conservative_polygon_remapper_adjoint(value_semantics: str, device: str) -> None:
    dev = torch.device(device)
    src_h, src_w = 8, 8
    tgt_h, tgt_w = 6, 6

    src_header = _simple_header(nx=src_w, ny=src_h, crval1=12.00, crval2=-3.00)
    tgt_header = _simple_header(nx=tgt_w, ny=tgt_h, crval1=12.01, crval2=-2.99)

    src_wcs = WCS(src_header)
    tgt_wcs = WCS(tgt_header)

    target = _prepare_target(
        header=tgt_header,
        target_wcs=tgt_wcs,
        target_shape=(tgt_h, tgt_w),
        device=dev,
        include_pixel_area=(value_semantics == "flux"),
        include_representative_pixel_area=True,
    )

    x = torch.randn(
        (src_h, src_w), dtype=torch.float32, device=dev, requires_grad=True
    )
    source_weight = torch.ones((src_h, src_w), dtype=torch.float32, device=dev)

    y_pred, _ = project_image_observation_geometry(
        x,
        source_weight,
        src_wcs,
        target,
        geometry=ConservativePolygonRemapOp(),
        resampling_kernel="conservative_polygon",
        value_semantics=value_semantics,
        drizzle_pixfrac=1.0,
        pixel_area_sr_map_fn=pixel_area_sr_map,
        representative_pixel_area_fn=representative_pixel_area_sr,
        chunk_pixels=256,
        resample_weighted_to_target_fn=_resample_weighted_to_target,
        forward_project_weighted_to_target_fn=_forward_project_weighted_to_target,
    )

    assert torch.any(y_pred != 0.0), (
        f"y_pred is completely zero for conservative_polygon ({value_semantics})"
    )

    y = torch.randn_like(y_pred)
    val1 = torch.sum(y_pred * y)
    val1.backward()

    val2 = torch.sum(x * x.grad)

    torch.testing.assert_close(val1, val2, rtol=1e-4, atol=1e-4)


def _project_and_assert_adjoint(
    geometry,
    *,
    resampling_kernel: str,
    value_semantics: str,
    device: str = "cpu",
    src_h: int = 16,
    src_w: int = 16,
    tgt_h: int = 12,
    tgt_w: int = 12,
) -> None:
    dev = torch.device(device)
    src_header = _simple_header(nx=src_w, ny=src_h, crval1=12.00, crval2=-3.00)
    tgt_header = _simple_header(nx=tgt_w, ny=tgt_h, crval1=12.01, crval2=-2.99)
    src_wcs = WCS(src_header)
    tgt_wcs = WCS(tgt_header)

    target = _prepare_target(
        header=tgt_header,
        target_wcs=tgt_wcs,
        target_shape=(tgt_h, tgt_w),
        device=dev,
        include_pixel_area=(value_semantics == "flux"),
        include_representative_pixel_area=True,
    )

    x = torch.randn((src_h, src_w), dtype=torch.float32, device=dev, requires_grad=True)
    source_weight = torch.ones((src_h, src_w), dtype=torch.float32, device=dev)

    y_pred, _ = project_image_observation_geometry(
        x,
        source_weight,
        src_wcs,
        target,
        geometry=geometry,
        resampling_kernel=resampling_kernel,
        value_semantics=value_semantics,
        drizzle_pixfrac=1.0,
        pixel_area_sr_map_fn=pixel_area_sr_map,
        representative_pixel_area_fn=representative_pixel_area_sr,
        chunk_pixels=256,
        resample_weighted_to_target_fn=_resample_weighted_to_target,
        forward_project_weighted_to_target_fn=_forward_project_weighted_to_target,
    )

    assert torch.any(y_pred != 0.0), (
        f"y_pred is completely zero for {type(geometry).__name__}/{resampling_kernel} ({value_semantics})"
    )

    y = torch.randn_like(y_pred)
    val1 = torch.sum(y_pred * y)
    val1.backward()
    val2 = torch.sum(x * x.grad)
    torch.testing.assert_close(val1, val2, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("value_semantics", ["surface_brightness", "flux"])
def test_exact_drizzle_square_adjoint(value_semantics: str, device: str) -> None:
    _project_and_assert_adjoint(
        ExactDrizzleGeometryOp(),
        resampling_kernel="drizzle_square",
        value_semantics=value_semantics,
        device=device,
    )


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("value_semantics", ["surface_brightness", "flux"])
def test_exact_drizzle_point_adjoint(value_semantics: str, device: str) -> None:
    _project_and_assert_adjoint(
        ExactDrizzleGeometryOp(),
        resampling_kernel="drizzle_point",
        value_semantics=value_semantics,
        device=device,
    )


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("value_semantics", ["surface_brightness", "flux"])
def test_forward_splat_point_adjoint(value_semantics: str, device: str) -> None:
    _project_and_assert_adjoint(
        ForwardSplatGeometryOp(),
        resampling_kernel="splat_point",
        value_semantics=value_semantics,
        device=device,
    )


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("value_semantics", ["surface_brightness", "flux"])
def test_forward_splat_box_adjoint(value_semantics: str, device: str) -> None:
    _project_and_assert_adjoint(
        ForwardSplatGeometryOp(),
        resampling_kernel="splat_box",
        value_semantics=value_semantics,
        device=device,
    )


_HEALPIX_HEADER = {
    "NAXIS": 2,
    "NAXIS1": 10,
    "NAXIS2": 10,
    "CTYPE1": "RA---TAN",
    "CTYPE2": "DEC--TAN",
    "CRVAL1": 0.0,
    "CRVAL2": 0.0,
    "CRPIX1": 5.0,
    "CRPIX2": 5.0,
    "CDELT1": -0.1,
    "CDELT2": 0.1,
}


def _healpix_adjoint_correct(nest: bool) -> None:
    nside = 2
    npix = 12 * nside**2
    h = int(_HEALPIX_HEADER["NAXIS2"])
    w = int(_HEALPIX_HEADER["NAXIS1"])

    x = torch.randn(npix, dtype=torch.float64, requires_grad=True)
    y_pred = sample_healpix_map_on_image_header(
        x, nside=nside, nest=nest, target_header=_HEALPIX_HEADER
    )
    assert torch.any(y_pred != 0.0), (
        f"y_pred is completely zero for nest={nest}"
    )

    y = torch.randn(h, w, dtype=torch.float64)
    val1 = torch.sum(y_pred * y)
    val1.backward()
    assert x.grad is not None

    x_adj_explicit = accumulate_image_on_healpix_map_adjoint(
        y, nside=nside, nest=nest, source_header=_HEALPIX_HEADER
    )
    torch.testing.assert_close(x.grad, x_adj_explicit, rtol=1e-4, atol=1e-4)


def test_healpix_adjoint_ring() -> None:
    _healpix_adjoint_correct(nest=False)


def test_healpix_adjoint_nest() -> None:
    _healpix_adjoint_correct(nest=True)


@pytest.mark.parametrize("nest", [False, True])
def test_healpix_gradcheck(nest: bool) -> None:
    nside = 2
    npix = 12 * nside**2

    def f(x: torch.Tensor) -> torch.Tensor:
        return sample_healpix_map_on_image_header(
            x, nside=nside, nest=nest, target_header=_HEALPIX_HEADER
        )

    x = torch.randn(npix, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(f, (x,), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("nest", [False, True])
def test_healpix_gradgradcheck(nest: bool) -> None:
    nside = 2
    npix = 12 * nside**2

    def f(x: torch.Tensor) -> torch.Tensor:
        return sample_healpix_map_on_image_header(
            x, nside=nside, nest=nest, target_header=_HEALPIX_HEADER
        ).sum()

    x = torch.randn(npix, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradgradcheck(f, (x,), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("device", _DEVICES)
def test_gaussian_blur_gradcheck(device: str) -> None:
    dev = torch.device(device)
    image = torch.randn(5, 7, dtype=torch.float64, device=dev, requires_grad=True)

    def f(x: torch.Tensor) -> torch.Tensor:
        return gaussian_blur(x, sigma_pix=1.5)

    assert torch.autograd.gradcheck(f, (image,), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("device", _DEVICES)
def test_gaussian_blur_gradgradcheck(device: str) -> None:
    dev = torch.device(device)
    image = torch.randn(5, 7, dtype=torch.float64, device=dev, requires_grad=True)

    def f(x: torch.Tensor) -> torch.Tensor:
        return gaussian_blur(x, sigma_pix=1.5).sum()

    assert torch.autograd.gradgradcheck(f, (image,), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("device", _DEVICES)
def test_apply_kernel2d_gradcheck(device: str) -> None:
    dev = torch.device(device)
    image = torch.randn(5, 7, dtype=torch.float64, device=dev, requires_grad=True)
    kernel = torch.randn(3, 3, dtype=torch.float64, device=dev)

    def f(x: torch.Tensor) -> torch.Tensor:
        return apply_kernel2d(x, kernel)

    assert torch.autograd.gradcheck(f, (image,), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("device", _DEVICES)
def test_apply_kernel2d_gradgradcheck(device: str) -> None:
    dev = torch.device(device)
    image = torch.randn(5, 7, dtype=torch.float64, device=dev, requires_grad=True)
    kernel = torch.randn(3, 3, dtype=torch.float64, device=dev)

    def f(x: torch.Tensor) -> torch.Tensor:
        return apply_kernel2d(x, kernel).sum()

    assert torch.autograd.gradgradcheck(f, (image,), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("device", _DEVICES)
def test_apply_kernel2d_kernel_gradcheck(device: str) -> None:
    dev = torch.device(device)
    image = torch.randn(5, 7, dtype=torch.float64, device=dev)
    kernel = torch.randn(3, 3, dtype=torch.float64, device=dev, requires_grad=True)

    def f(k: torch.Tensor) -> torch.Tensor:
        return apply_kernel2d(image, k)

    assert torch.autograd.gradcheck(f, (kernel,), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("device", _DEVICES)
def test_apply_kernel2d_kernel_gradgradcheck(device: str) -> None:
    dev = torch.device(device)
    image = torch.randn(5, 7, dtype=torch.float64, device=dev)
    kernel = torch.randn(3, 3, dtype=torch.float64, device=dev, requires_grad=True)

    def f(k: torch.Tensor) -> torch.Tensor:
        return apply_kernel2d(image, k).sum()

    assert torch.autograd.gradgradcheck(f, (kernel,), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("device", _DEVICES)
def test_polynomial_background_gradcheck(device: str) -> None:
    coeffs = torch.tensor(
        [1.0, 0.5, -0.3, 0.1], dtype=torch.float64, requires_grad=True
    )

    def f(c: torch.Tensor) -> torch.Tensor:
        return polynomial_background(
            (4, 5),
            coeffs=c,
            device=torch.device(device),
            normalized_coordinates=True,
        )

    assert torch.autograd.gradcheck(f, (coeffs,), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("device", _DEVICES)
def test_polynomial_background_gradgradcheck(device: str) -> None:
    coeffs = torch.tensor(
        [1.0, 0.5, -0.3, 0.1], dtype=torch.float64, requires_grad=True
    )

    def f(c: torch.Tensor) -> torch.Tensor:
        return polynomial_background(
            (4, 5),
            coeffs=c,
            device=torch.device(device),
            normalized_coordinates=True,
        ).sum()

    assert torch.autograd.gradgradcheck(f, (coeffs,), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("device", _DEVICES)
def test_apply_calibration_gradcheck(device: str) -> None:
    dev = torch.device(device)
    image = torch.randn(6, 6, dtype=torch.float64, device=dev, requires_grad=True)
    weight = torch.ones(6, 6, dtype=torch.float64, device=dev, requires_grad=True)

    def f(img: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        cal_img, cal_w = apply_image_observation_calibration(
            photometric_scale=2.0,
            background_offset=0.5,
            image=img,
            source_weight=w,
        )
        return cal_img + cal_w

    assert torch.autograd.gradcheck(f, (image, weight), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("device", _DEVICES)
def test_apply_calibration_gradgradcheck(device: str) -> None:
    dev = torch.device(device)
    image = torch.randn(6, 6, dtype=torch.float64, device=dev, requires_grad=True)
    weight = torch.ones(6, 6, dtype=torch.float64, device=dev, requires_grad=True)

    def f(img: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        cal_img, cal_w = apply_image_observation_calibration(
            photometric_scale=2.0,
            background_offset=0.5,
            image=img,
            source_weight=w,
        )
        return (cal_img + cal_w).sum()

    assert torch.autograd.gradgradcheck(f, (image, weight), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("device", _DEVICES)
def test_learned_lowrank_psf_render_gradcheck(device: str) -> None:
    """Learned low-rank chromatic PSF render is differentiable end to end."""
    from torchsky.map.field.types import PSFSpec
    from torchsky.map.operators import apply_psf_spec, LowRankChromaticPSF

    dev = torch.device(device)
    model = LowRankChromaticPSF(
        grid_side=2,
        spectral_degree=1,
        base_sigma_pix=1.5,
        extent_pix=64.0,
        kernel_size=7,
    ).to(dev)
    for parameter in model.parameters():
        parameter.data.normal_(0.0, 0.05).to(dev)

    image = torch.randn(16, 16, dtype=torch.float64, device=dev, requires_grad=True)
    cell_u = torch.stack(
        torch.meshgrid(
            torch.linspace(0.0, 63.0, 2),
            torch.linspace(0.0, 63.0, 2),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 2)
    spec = PSFSpec.learned(
        model,
        base_sigma_pix=1.5,
        cell_u_pix=cell_u,
        cell_lam_norm=0.5,
    )

    def f(img: torch.Tensor) -> torch.Tensor:
        return apply_psf_spec(img, spec).square().sum()

    assert torch.autograd.gradcheck(f, (image,), atol=1e-4, rtol=1e-4)


def test_learned_lowrank_kernel_gauges() -> None:
    """Rendered kernels integrate to 1 and are zero-centroid by construction."""
    from torchsky.map.operators import (
        LowRankChromaticPSF,
        gaussian_kernels_from_params,
    )
    from torchsky.map.operators.psf import psf_support_radius
    from torchsky.map.field.types import PSFSpec

    model = LowRankChromaticPSF(base_sigma_pix=2.3)
    with torch.no_grad():
        model.raw_log_sigma.normal_(0.0, 0.2)
        model.raw_ellipticity.normal_(0.0, 0.05)
        model.raw_pa_rad.normal_(0.0, 0.05)
    u = torch.tensor([[100.0, 1500.0], [1800.0, 300.0]])
    kernels = gaussian_kernels_from_params(model(u, 0.6), size=31)
    size = kernels.shape[-1]
    gy, gx = torch.meshgrid(
        torch.arange(size, dtype=torch.float64) - size // 2,
        torch.arange(size, dtype=torch.float64) - size // 2,
        indexing="ij",
    )
    assert (kernels.sum(dim=(-2, -1)) - 1.0).abs().max() < 1e-6
    centroid_x = (kernels * gx).sum(dim=(-2, -1))
    centroid_y = (kernels * gy).sum(dim=(-2, -1))
    assert (
        torch.stack([centroid_x, centroid_y], dim=-1).norm(dim=-1).max() < 1e-3
    )

    spec = PSFSpec.learned(model, base_sigma_pix=2.3)
    radius = max(1, round(3.0 * 2.3))
    assert psf_support_radius(spec) == (2 * radius, 2 * radius)


def test_learned_lowrank_psf_dot_product() -> None:
    """<A x, y> == <x, A^T y> for the blended learned chromatic PSF."""
    from torchsky.map.field.types import PSFSpec
    from torchsky.map.operators import (
        LowRankChromaticPSF,
        apply_learned_psf_spec_adjoint,
        apply_psf_spec,
    )

    torch.manual_seed(11)
    model = LowRankChromaticPSF(
        grid_side=2,
        spectral_degree=1,
        base_sigma_pix=0.8,
        extent_pix=48.0,
        kernel_size=5,
    )
    with torch.no_grad():
        model.raw_log_sigma.normal_(0.0, 0.08)
        model.raw_pa_rad.normal_(0.0, 0.2)
        model.raw_ellipticity.normal_(0.0, 0.05)
    u = torch.stack(
        torch.meshgrid(torch.linspace(2, 46, 2), torch.linspace(2, 46, 2), indexing="ij"),
        dim=-1,
    ).reshape(-1, 2)
    spec = PSFSpec.learned(model, base_sigma_pix=0.8, cell_u_pix=u, cell_lam_norm=0.5)
    x = torch.randn(24, 24, dtype=torch.float64)
    y = torch.randn(24, 24, dtype=torch.float64)
    lhs = float((apply_psf_spec(x, spec) * y).sum())
    rhs = float((x * apply_learned_psf_spec_adjoint(y, spec)).sum())
    assert abs(lhs - rhs) / max(abs(lhs), 1e-30) < 1e-12
