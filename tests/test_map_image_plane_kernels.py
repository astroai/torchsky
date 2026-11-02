from __future__ import annotations
import math
from pathlib import Path
import pytest
import torch
from conftest import assert_close_on_device
import torchfits
import torchsky
import torchsky.map.calibration as calibration_mod
import torchsky.map.image_plane_coadd as image_plane_coadd_mod
import torchsky.map.image_plane_projection as image_plane_projection_mod
import torchsky.map.operators.resampling as resampling_mod
import torchsky.map.windowing as windowing_mod
from torchsky.map.image_plane_coadd import (
    coadd_image_plane,
)
from torchsky.map.observation import ImageObservation
from torchsky.map.operators import (
    pixel_area_sr_map,
    representative_pixel_area_sr,
    validate_image_plane_2d,
)

pytest.importorskip("torchsky._cpp")
from map_image_plane_support import (
    _simple_header,
)


@pytest.mark.parametrize(
    "kernel",
    [
        "lanczos3",
        "lanczos5",
        "mitchell_netravali",
        "catmull_rom",
        "drizzle_point",
        "drizzle_square",
        "splat_point",
        "splat_box",
    ],
)
@pytest.mark.needs_float64(
    "drizzle_point", "drizzle_square", "splat_box", "splat_point"
)
def test_coadd_image_plane_accepts_custom_high_accuracy_kernels(
    tmp_path, kernel: str
) -> None:
    path_a = tmp_path / f"a_{kernel}.fits"
    path_b = tmp_path / f"b_{kernel}.fits"
    header_a = _simple_header()
    header_b = dict(header_a)
    header_b["CRVAL1"] = float(header_b["CRVAL1"]) + 0.02
    data = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    torchfits.write(str(path_a), data, header=header_a, overwrite=True)
    torchfits.write(str(path_b), data + 1.0, header=header_b, overwrite=True)

    result = coadd_image_plane(
        [ImageObservation(str(path_a)), ImageObservation(str(path_b))],
        resampling_kernel=kernel,
    )

    assert result.data.shape == (8, 8)
    assert torch.isfinite(result.data).all()


@pytest.mark.parametrize(
    "kernel",
    [
        "drizzle_point",
        "drizzle_square",
        "conservative_polygon",
        "splat_point",
        "splat_box",
    ],
)
@pytest.mark.needs_float64
def test_coadd_image_plane_forward_geometry_kernels_bypass_grid_sample(
    tmp_path: Path,
    monkeypatch,
    kernel: str,
) -> None:
    path_a = tmp_path / f"drizzle_a_{kernel}.fits"
    path_b = tmp_path / f"drizzle_b_{kernel}.fits"
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
    assert not calls


@pytest.mark.needs_float64
def test_coadd_image_plane_conservative_polygon_preserves_shifted_support_and_finite_output(
    tmp_path: Path,
) -> None:
    path = tmp_path / "conservative_polygon_identity.fits"
    header = _simple_header()
    target_header = dict(header)
    target_header["CRVAL1"] = float(target_header["CRVAL1"]) + 0.01
    data = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    torchfits.write(str(path), data, header=header, overwrite=True)

    result = coadd_image_plane(
        [ImageObservation(str(path))],
        target_header=target_header,
        resampling_kernel="conservative_polygon",
        value_semantics="surface_brightness",
    )

    assert torch.isfinite(result.weight).all()
    assert torch.isfinite(result.data).all()
    support = result.weight > 0.0
    assert bool(support.any())


@pytest.mark.needs_float64
def test_coadd_image_plane_exact_drizzle_square_preserves_flux_under_shift(
    tmp_path: Path,
) -> None:
    path = tmp_path / "drizzle_identity.fits"
    header = _simple_header()
    target_header = dict(header)
    target_header["CRVAL1"] = float(target_header["CRVAL1"]) + 0.01
    data = torch.ones((8, 8), dtype=torch.float32)
    torchfits.write(str(path), data, header=header, overwrite=True)

    result = coadd_image_plane(
        [ImageObservation(str(path))],
        target_header=target_header,
        resampling_kernel="drizzle_square",
        value_semantics="surface_brightness",
        drizzle_pixfrac=1.0,
    )

    assert torch.isfinite(result.weight).all()
    assert torch.isfinite(result.data).all()
    support = result.weight > 0.0
    assert bool(support.any())
    assert_close_on_device(
        result.data[support],
        torch.ones_like(result.data[support]),
        atol=1e-2,
        rtol=1e-2,
    )


def test_coadd_fft_shift_matches_bilinear_for_pure_crpix_offset_interior(
    tmp_path: Path,
) -> None:
    """fft_shift uses periodic FFT; interior should track bilinear for CRPIX-only offset."""
    ny, nx = 48, 48
    base = {
        "NAXIS": 2,
        "NAXIS1": nx,
        "NAXIS2": ny,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": nx / 2.0,
        "CRPIX2": ny / 2.0,
        "CRVAL1": 180.0,
        "CRVAL2": 0.0,
        "CDELT1": -0.02,
        "CDELT2": 0.02,
    }
    ha = dict(base)
    ht = dict(base)
    ht["CRPIX1"] = float(ht["CRPIX1"]) + 0.31
    ht["CRPIX2"] = float(ht["CRPIX2"]) - 0.19
    yy, xx = torch.meshgrid(
        torch.arange(ny, dtype=torch.float32),
        torch.arange(nx, dtype=torch.float32),
        indexing="ij",
    )
    blob = torch.exp(-((xx - nx / 2.0) ** 2 + (yy - ny / 2.0) ** 2) / (2.0 * 8.0**2))
    path = tmp_path / "crpix.fits"
    torchfits.write(str(path), blob, header=ha, overwrite=True)
    obs = [ImageObservation(str(path))]
    r_fft = coadd_image_plane(obs, target_header=ht, resampling_kernel="fft_shift")
    r_bi = coadd_image_plane(obs, target_header=ht, resampling_kernel="bilinear")
    inner_y = slice(10, ny - 10)
    inner_x = slice(10, nx - 10)
    mask = (r_fft.weight[inner_y, inner_x] > 0.0) & (
        r_bi.weight[inner_y, inner_x] > 0.0
    )
    assert int(mask.sum().item()) > 50
    assert_close_on_device(
        r_fft.data[inner_y, inner_x][mask],
        r_bi.data[inner_y, inner_x][mask],
        atol=0.06,
        rtol=0.06,
    )


def test_coadd_fft_diagonal_affine_tracks_bilinear_scaled_cdelta_interior(
    tmp_path: Path,
) -> None:
    """Diagonal affine FFT path: uniform CDELT scale should track bilinear in the interior."""
    ny, nx = 56, 56
    base = {
        "NAXIS": 2,
        "NAXIS1": nx,
        "NAXIS2": ny,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": nx / 2.0,
        "CRPIX2": ny / 2.0,
        "CRVAL1": 100.0,
        "CRVAL2": 8.0,
        "CDELT1": -0.018,
        "CDELT2": 0.018,
    }
    ha = dict(base)
    ht = dict(base)
    f = 1.05
    ht["CDELT1"] = float(ht["CDELT1"]) * f
    ht["CDELT2"] = float(ht["CDELT2"]) * f
    yy, xx = torch.meshgrid(
        torch.arange(ny, dtype=torch.float32),
        torch.arange(nx, dtype=torch.float32),
        indexing="ij",
    )
    blob = torch.exp(-((xx - nx / 2.0) ** 2 + (yy - ny / 2.0) ** 2) / (2.0 * 9.0**2))
    path = tmp_path / "diag_cdelta.fits"
    torchfits.write(str(path), blob, header=ha, overwrite=True)
    obs = [ImageObservation(str(path))]
    r_fft = coadd_image_plane(
        obs, target_header=ht, resampling_kernel="fft_diagonal_affine"
    )
    r_bi = coadd_image_plane(obs, target_header=ht, resampling_kernel="bilinear")
    inner_y = slice(12, ny - 12)
    inner_x = slice(12, nx - 12)
    mask = (r_fft.weight[inner_y, inner_x] > 0.0) & (
        r_bi.weight[inner_y, inner_x] > 0.0
    )
    assert int(mask.sum().item()) > 80
    assert_close_on_device(
        r_fft.data[inner_y, inner_x][mask],
        r_bi.data[inner_y, inner_x][mask],
        atol=0.08,
        rtol=0.08,
    )


@pytest.mark.needs_float64
def test_coadd_periodic_linear_affine_matches_bilinear_skewed_cd_interior(
    tmp_path: Path,
) -> None:
    """Full linear pixel map (rotated CD) on a torus bilinear path vs ``grid_sample``."""
    ny, nx = 40, 40
    base = {
        "NAXIS": 2,
        "NAXIS1": nx,
        "NAXIS2": ny,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": nx / 2.0,
        "CRPIX2": ny / 2.0,
        "CRVAL1": 50.0,
        "CRVAL2": 1.0,
        "CDELT1": -0.02,
        "CDELT2": 0.02,
    }
    ha = dict(base)
    th_rad = math.radians(12.0)
    sc = 0.02
    ht = {
        "NAXIS": 2,
        "NAXIS1": nx,
        "NAXIS2": ny,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": nx / 2.0,
        "CRPIX2": ny / 2.0,
        "CRVAL1": base["CRVAL1"],
        "CRVAL2": base["CRVAL2"],
        "CD1_1": -sc * math.cos(th_rad),
        "CD1_2": sc * math.sin(th_rad),
        "CD2_1": sc * math.sin(th_rad),
        "CD2_2": sc * math.cos(th_rad),
    }
    yy, xx = torch.meshgrid(
        torch.arange(ny, dtype=torch.float32),
        torch.arange(nx, dtype=torch.float32),
        indexing="ij",
    )
    blob = torch.exp(-((xx - nx / 2.0) ** 2 + (yy - ny / 2.0) ** 2) / (2.0 * 6.0**2))
    path = tmp_path / "skew_pla.fits"
    torchfits.write(str(path), blob, header=ha, overwrite=True)
    obs = [ImageObservation(str(path))]
    r_p = coadd_image_plane(
        obs,
        target_header=ht,
        resampling_kernel="periodic_linear_affine",
    )
    r_b = coadd_image_plane(obs, target_header=ht, resampling_kernel="bilinear")
    inner_y = slice(8, ny - 8)
    inner_x = slice(8, nx - 8)
    mask = (r_p.weight[inner_y, inner_x] > 0.0) & (r_b.weight[inner_y, inner_x] > 0.0)
    assert_close_on_device(
        r_p.data[inner_y, inner_x][mask],
        r_b.data[inner_y, inner_x][mask],
        atol=1e-4,
        rtol=1e-4,
    )


@pytest.mark.needs_float64
def test_coadd_fft_linear_affine_matches_periodic_linear_affine_scaled_cdelta_interior(
    tmp_path: Path,
) -> None:
    """LDU FFT path reduces to periodic bilinear + translation when CD is diagonal-scaled."""
    ny, nx = 56, 56
    base = {
        "NAXIS": 2,
        "NAXIS1": nx,
        "NAXIS2": ny,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": nx / 2.0,
        "CRPIX2": ny / 2.0,
        "CRVAL1": 100.0,
        "CRVAL2": 8.0,
        "CDELT1": -0.018,
        "CDELT2": 0.018,
    }
    ha = dict(base)
    ht = dict(base)
    f = 1.05
    ht["CDELT1"] = float(ht["CDELT1"]) * f
    ht["CDELT2"] = float(ht["CDELT2"]) * f
    yy, xx = torch.meshgrid(
        torch.arange(ny, dtype=torch.float32),
        torch.arange(nx, dtype=torch.float32),
        indexing="ij",
    )
    blob = torch.exp(-((xx - nx / 2.0) ** 2 + (yy - ny / 2.0) ** 2) / (2.0 * 9.0**2))
    path = tmp_path / "diag_ldu.fits"
    torchfits.write(str(path), blob, header=ha, overwrite=True)
    obs = [ImageObservation(str(path))]
    r_pla = coadd_image_plane(
        obs, target_header=ht, resampling_kernel="periodic_linear_affine"
    )
    r_fl = coadd_image_plane(
        obs, target_header=ht, resampling_kernel="fft_linear_affine"
    )
    inner_y = slice(12, ny - 12)
    inner_x = slice(12, nx - 12)
    mask = (r_pla.weight[inner_y, inner_x] > 0.0) & (
        r_fl.weight[inner_y, inner_x] > 0.0
    )
    assert int(mask.sum().item()) > 80
    assert_close_on_device(
        r_fl.data[inner_y, inner_x][mask],
        r_pla.data[inner_y, inner_x][mask],
        atol=0.004,
        rtol=0.02,
    )


@pytest.mark.needs_float64
def test_coadd_spectral_linear_affine_tracks_bilinear_skewed_cd_interior(
    tmp_path: Path,
) -> None:
    """Global spectral affine remapper should track bilinear on linear rotated grids."""
    ny, nx = 40, 40
    base = {
        "NAXIS": 2,
        "NAXIS1": nx,
        "NAXIS2": ny,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": nx / 2.0,
        "CRPIX2": ny / 2.0,
        "CRVAL1": 50.0,
        "CRVAL2": 1.0,
        "CDELT1": -0.02,
        "CDELT2": 0.02,
    }
    ha = dict(base)
    th_rad = math.radians(12.0)
    sc = 0.02
    ht = {
        "NAXIS": 2,
        "NAXIS1": nx,
        "NAXIS2": ny,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": nx / 2.0,
        "CRPIX2": ny / 2.0,
        "CRVAL1": base["CRVAL1"],
        "CRVAL2": base["CRVAL2"],
        "CD1_1": -sc * math.cos(th_rad),
        "CD1_2": sc * math.sin(th_rad),
        "CD2_1": sc * math.sin(th_rad),
        "CD2_2": sc * math.cos(th_rad),
    }
    yy, xx = torch.meshgrid(
        torch.arange(ny, dtype=torch.float32),
        torch.arange(nx, dtype=torch.float32),
        indexing="ij",
    )
    blob = torch.exp(-((xx - nx / 2.0) ** 2 + (yy - ny / 2.0) ** 2) / (2.0 * 6.0**2))
    path = tmp_path / "skew_spectral.fits"
    torchfits.write(str(path), blob, header=ha, overwrite=True)
    obs = [ImageObservation(str(path))]
    r_spec = coadd_image_plane(
        obs,
        target_header=ht,
        resampling_kernel="spectral_linear_affine",
    )
    r_bi = coadd_image_plane(obs, target_header=ht, resampling_kernel="bilinear")
    inner_y = slice(8, ny - 8)
    inner_x = slice(8, nx - 8)
    mask = (r_spec.weight[inner_y, inner_x] > 0.0) & (
        r_bi.weight[inner_y, inner_x] > 0.0
    )
    assert int(mask.sum().item()) > 100
    assert_close_on_device(
        r_spec.data[inner_y, inner_x][mask],
        r_bi.data[inner_y, inner_x][mask],
        atol=0.05,
        rtol=0.05,
    )


@pytest.mark.needs_float64
def test_coadd_spectral_linear_affine_zero_boundary_does_not_wrap_corner_source(
    tmp_path: Path,
) -> None:
    ny, nx = 32, 32
    base = {
        "NAXIS": 2,
        "NAXIS1": nx,
        "NAXIS2": ny,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": nx / 2.0,
        "CRPIX2": ny / 2.0,
        "CRVAL1": 180.0,
        "CRVAL2": 0.0,
        "CDELT1": -0.02,
        "CDELT2": 0.02,
    }
    ha = dict(base)
    ht = dict(base)
    ht["CRPIX1"] = float(ht["CRPIX1"]) + 2.0
    image = torch.zeros((ny, nx), dtype=torch.float32)
    image[2, 2] = 1.0
    path = tmp_path / "spectral_edge.fits"
    torchfits.write(str(path), image, header=ha, overwrite=True)
    obs = [ImageObservation(str(path))]
    r_spec = coadd_image_plane(
        obs,
        target_header=ht,
        resampling_kernel="spectral_linear_affine",
    )
    r_bi = coadd_image_plane(obs, target_header=ht, resampling_kernel="bilinear")
    assert float(r_spec.data[:, -1].abs().max()) < 1e-5
    assert_close_on_device(r_spec.data, r_bi.data, atol=0.03, rtol=0.03)


@pytest.mark.needs_float64
def test_coadd_spectral_nonuniform_tracks_bilinear_on_subpixel_crval_shift(
    tmp_path: Path,
) -> None:
    size = 36
    ha = {
        "NAXIS": 2,
        "NAXIS1": size,
        "NAXIS2": size,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": size / 2.0,
        "CRPIX2": size / 2.0,
        "CRVAL1": 180.0,
        "CRVAL2": 2.0,
        "CDELT1": -0.015,
        "CDELT2": 0.015,
    }
    ht = dict(ha)
    ht["CRVAL1"] = float(ht["CRVAL1"]) + 0.25 * abs(float(ht["CDELT1"]))
    ht["CRVAL2"] = float(ht["CRVAL2"]) - 0.2 * abs(float(ht["CDELT2"]))
    yy, xx = torch.meshgrid(
        torch.arange(size, dtype=torch.float32),
        torch.arange(size, dtype=torch.float32),
        indexing="ij",
    )
    image = (
        torch.exp(-((xx - 18.0) ** 2 + (yy - 16.0) ** 2) / (2.0 * 4.5**2))
        + 0.15 * torch.sin(0.4 * xx)
        + 0.1 * torch.cos(0.35 * yy)
    ).to(dtype=torch.float32)
    path = tmp_path / "tpv_shift_spectral.fits"
    torchfits.write(str(path), image, header=ha, overwrite=True)
    obs = [ImageObservation(str(path))]
    r_spec = coadd_image_plane(
        obs,
        target_header=ht,
        resampling_kernel="spectral_nonuniform",
    )
    r_bi = coadd_image_plane(obs, target_header=ht, resampling_kernel="bilinear")
    inner = (slice(8, size - 8), slice(8, size - 8))
    mask = (r_spec.weight[inner] > 0.0) & (r_bi.weight[inner] > 0.0)
    assert int(mask.sum().item()) > 100
    assert_close_on_device(
        r_spec.data[inner][mask],
        r_bi.data[inner][mask],
        atol=0.05,
        rtol=0.05,
    )


@pytest.mark.needs_float64
def test_coadd_spectral_nonuniform_fast_tracks_direct_nonuniform_on_subpixel_shift(
    tmp_path: Path,
) -> None:
    size = 36
    ha = {
        "NAXIS": 2,
        "NAXIS1": size,
        "NAXIS2": size,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": size / 2.0,
        "CRPIX2": size / 2.0,
        "CRVAL1": 180.0,
        "CRVAL2": 2.0,
        "CDELT1": -0.015,
        "CDELT2": 0.015,
    }
    ht = dict(ha)
    ht["CRVAL1"] = float(ht["CRVAL1"]) + 0.25 * abs(float(ht["CDELT1"]))
    ht["CRVAL2"] = float(ht["CRVAL2"]) - 0.2 * abs(float(ht["CDELT2"]))
    yy, xx = torch.meshgrid(
        torch.arange(size, dtype=torch.float32),
        torch.arange(size, dtype=torch.float32),
        indexing="ij",
    )
    image = (
        torch.exp(-((xx - 18.0) ** 2 + (yy - 16.0) ** 2) / (2.0 * 4.5**2))
        + 0.15 * torch.sin(0.4 * xx)
        + 0.1 * torch.cos(0.35 * yy)
    ).to(dtype=torch.float32)
    path = tmp_path / "tan_fast_spectral.fits"
    torchfits.write(str(path), image, header=ha, overwrite=True)
    obs = [ImageObservation(str(path))]
    r_fast = coadd_image_plane(
        obs,
        target_header=ht,
        resampling_kernel="spectral_nonuniform_fast",
    )
    r_direct = coadd_image_plane(
        obs,
        target_header=ht,
        resampling_kernel="spectral_nonuniform",
    )
    inner = (slice(8, size - 8), slice(8, size - 8))
    mask = (r_fast.weight[inner] > 0.0) & (r_direct.weight[inner] > 0.0)
    assert int(mask.sum().item()) > 100
    assert_close_on_device(
        r_fast.data[inner][mask],
        r_direct.data[inner][mask],
        atol=0.04,
        rtol=0.04,
    )


def test_coadd_fft_diagonal_affine_rejects_skewed_cd_matrix(tmp_path: Path) -> None:
    """Off-diagonal CD terms induce iy/jx cross-terms; diagonal FFT path must reject."""
    ny, nx = 24, 24
    base = {
        "NAXIS": 2,
        "NAXIS1": nx,
        "NAXIS2": ny,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": nx / 2.0,
        "CRPIX2": ny / 2.0,
        "CRVAL1": 50.0,
        "CRVAL2": 1.0,
        "CDELT1": -0.02,
        "CDELT2": 0.02,
    }
    ha = dict(base)
    th = math.radians(17.0)
    sc = 0.02
    ht = {
        "NAXIS": 2,
        "NAXIS1": nx,
        "NAXIS2": ny,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": nx / 2.0,
        "CRPIX2": ny / 2.0,
        "CRVAL1": base["CRVAL1"],
        "CRVAL2": base["CRVAL2"],
        "CD1_1": -sc * math.cos(th),
        "CD1_2": sc * math.sin(th),
        "CD2_1": sc * math.sin(th),
        "CD2_2": sc * math.cos(th),
    }
    path = tmp_path / "skew.fits"
    torchfits.write(str(path), torch.ones((ny, nx), dtype=torch.float32), header=ha)
    with pytest.raises(ValueError, match="fft_diagonal_affine"):
        coadd_image_plane(
            [ImageObservation(str(path))],
            target_header=ht,
            resampling_kernel="fft_diagonal_affine",
        )


def test_coadd_fft_shift_rejects_crval_only_offset(
    tmp_path: Path,
) -> None:
    ny, nx = 16, 16
    ha = _simple_header()
    ha["NAXIS1"] = nx
    ha["NAXIS2"] = ny
    ha["CRPIX1"] = nx / 2.0
    ha["CRPIX2"] = ny / 2.0
    ht = dict(ha)
    ht["CRVAL1"] = float(ht["CRVAL1"]) + 0.05
    path = tmp_path / "a.fits"
    torchfits.write(str(path), torch.ones((ny, nx), dtype=torch.float32), header=ha)
    with pytest.raises(ValueError, match="fft_shift"):
        coadd_image_plane(
            [ImageObservation(str(path))],
            target_header=ht,
            resampling_kernel="fft_shift",
        )


def test_coadd_bilinear_subpixel_crval_shift_preserves_finite_support(
    tmp_path: Path,
) -> None:
    """SWarp/reproject-class subpixel sky shifts should stay finite on a smooth blob."""
    ny, nx = 40, 40
    base = {
        "NAXIS": 2,
        "NAXIS1": nx,
        "NAXIS2": ny,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": nx / 2.0,
        "CRPIX2": ny / 2.0,
        "CRVAL1": 200.0,
        "CRVAL2": 10.0,
        "CDELT1": -0.015,
        "CDELT2": 0.015,
    }
    ha = dict(base)
    hb = dict(base)
    hb["CRVAL1"] = float(hb["CRVAL1"]) + 0.4 * float(hb["CDELT1"])
    hb["CRVAL2"] = float(hb["CRVAL2"]) + 0.3 * float(hb["CDELT2"])
    yy, xx = torch.meshgrid(
        torch.arange(ny, dtype=torch.float32),
        torch.arange(nx, dtype=torch.float32),
        indexing="ij",
    )
    blob = torch.exp(-((xx - nx / 2.0) ** 2 + (yy - ny / 2.0) ** 2) / (2.0 * 6.0**2))
    pa = tmp_path / "sub_a.fits"
    pb = tmp_path / "sub_b.fits"
    torchfits.write(str(pa), blob, header=ha, overwrite=True)
    torchfits.write(str(pb), blob * 0.95 + 0.02, header=hb, overwrite=True)
    result = coadd_image_plane(
        [ImageObservation(str(pa)), ImageObservation(str(pb))],
        resampling_kernel="bilinear",
        stacking_method="weighted_mean",
        value_semantics="surface_brightness",
    )
    assert torch.isfinite(result.data).all()
    assert torch.isfinite(result.weight).all()
    assert float(result.weight.max().item()) > 0.0


def test_coadd_bilinear_partial_overlap_at_frame_edge_is_finite(
    tmp_path: Path,
) -> None:
    """SWarp-class stacks often have partial footprint support at tile edges."""
    ny, nx = 32, 32
    base = {
        "NAXIS": 2,
        "NAXIS1": nx,
        "NAXIS2": ny,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": nx / 2.0,
        "CRPIX2": ny / 2.0,
        "CRVAL1": 180.0,
        "CRVAL2": 0.0,
        "CDELT1": -0.02,
        "CDELT2": 0.02,
    }
    path_a = tmp_path / "edge_a.fits"
    path_b = tmp_path / "edge_b.fits"
    ha = dict(base)
    hb = dict(base)
    hb["CRVAL1"] = float(hb["CRVAL1"]) + 0.45 * float(hb["CDELT1"])
    data = torch.ones((ny, nx), dtype=torch.float32)
    torchfits.write(str(path_a), data, header=ha, overwrite=True)
    torchfits.write(str(path_b), data * 1.2, header=hb, overwrite=True)

    result = coadd_image_plane(
        [ImageObservation(str(path_a)), ImageObservation(str(path_b))],
        resampling_kernel="bilinear",
        stacking_method="weighted_mean",
        value_semantics="surface_brightness",
    )
    assert torch.isfinite(result.data).all()
    assert torch.isfinite(result.weight).all()
    assert result.overlap_count is not None
    assert int((result.overlap_count > 0).sum().item()) > 0
    assert int((result.overlap_count > 1).sum().item()) > 0


def test_coadd_image_plane_rejects_non_positive_drizzle_pixfrac(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pixfrac_invalid.fits"
    torchfits.write(
        str(path),
        torch.ones((4, 4), dtype=torch.float32),
        header=_simple_header(),
        overwrite=True,
    )

    with pytest.raises(ValueError, match="drizzle_pixfrac"):
        coadd_image_plane(
            [ImageObservation(str(path))],
            resampling_kernel="drizzle_square",
            drizzle_pixfrac=0.0,
        )


def test_coadd_image_plane_flux_semantics_improves_flux_conservation(tmp_path) -> None:
    source_path = tmp_path / "flux_source.fits"
    target_header = _simple_header()
    source_header = dict(target_header)
    source_header["CDELT1"] = -0.02
    source_header["CDELT2"] = 0.02

    source_wcs = torchsky.WCS(source_header)
    target_wcs = torchsky.WCS(target_header)
    source_area = pixel_area_sr_map(
        source_wcs,
        (8, 8),
        device=torch.device("cpu"),
    )
    target_area = pixel_area_sr_map(
        target_wcs,
        (8, 8),
        device=torch.device("cpu"),
    )
    area_norm = float(torch.median(target_area).item())
    brightness = torch.full((8, 8), 2.0, dtype=torch.float32)
    flux_source = brightness * (source_area / area_norm)
    expected_target_flux = brightness * (target_area / area_norm)
    torchfits.write(str(source_path), flux_source, header=source_header, overwrite=True)

    surface_result = coadd_image_plane(
        [ImageObservation(str(source_path))],
        target_header=target_header,
        value_semantics="surface_brightness",
    )
    flux_result = coadd_image_plane(
        [ImageObservation(str(source_path))],
        target_header=target_header,
        value_semantics="flux",
    )

    surface_flux_err = (
        torch.abs(
            surface_result.data - expected_target_flux.to(surface_result.data.device)
        )
        .mean()
        .item()
    )
    flux_flux_err = (
        torch.abs(flux_result.data - expected_target_flux.to(flux_result.data.device))
        .mean()
        .item()
    )
    assert flux_flux_err < surface_flux_err


def test_coadd_image_plane_weighted_mean_scales_surface_brightness_weights_only_for_coarser_targets(
    tmp_path,
) -> None:
    image = torch.ones((8, 8), dtype=torch.float32)
    weight = torch.ones((8, 8), dtype=torch.float32)

    fine_target = _simple_header()
    coarse_target = dict(fine_target)
    coarse_target["CDELT1"] = -0.02
    coarse_target["CDELT2"] = 0.02

    fine_source = dict(fine_target)
    fine_source_path = tmp_path / "fine_source.fits"
    fine_weight_path = tmp_path / "fine_source_wht.fits"
    torchfits.write(str(fine_source_path), image, header=fine_source, overwrite=True)
    torchfits.write(str(fine_weight_path), weight, header=fine_source, overwrite=True)

    coarse_source = dict(coarse_target)
    coarse_source_path = tmp_path / "coarse_source.fits"
    coarse_weight_path = tmp_path / "coarse_source_wht.fits"
    torchfits.write(
        str(coarse_source_path), image, header=coarse_source, overwrite=True
    )
    torchfits.write(
        str(coarse_weight_path), weight, header=coarse_source, overwrite=True
    )

    finer_target_result = coadd_image_plane(
        [
            ImageObservation(
                str(coarse_source_path), weight_path=str(coarse_weight_path)
            )
        ],
        target_header=fine_target,
        value_semantics="surface_brightness",
        stacking_method="weighted_mean",
    )
    coarser_target_result = coadd_image_plane(
        [ImageObservation(str(fine_source_path), weight_path=str(fine_weight_path))],
        target_header=coarse_target,
        value_semantics="surface_brightness",
        stacking_method="weighted_mean",
    )

    fine_area = representative_pixel_area_sr(
        torchsky.WCS(fine_target),
        (8, 8),
        device=torch.device("cpu"),
    )
    coarse_area = representative_pixel_area_sr(
        torchsky.WCS(coarse_target),
        (8, 8),
        device=torch.device("cpu"),
    )

    assert float(finer_target_result.data[4, 4].item()) == pytest.approx(
        1.0, abs=1.0e-4
    )
    assert float(finer_target_result.weight[4, 4].item()) == pytest.approx(
        1.0, abs=1.0e-4
    )
    assert float(coarser_target_result.data[4, 4].item()) == pytest.approx(
        1.0, abs=1.0e-4
    )
    assert float(coarser_target_result.weight[4, 4].item()) == pytest.approx(
        coarse_area / fine_area,
        rel=5.0e-2,
        abs=1.0e-4,
    )


def test_coadd_image_plane_builds_target_grid_once_per_run(
    tmp_path, monkeypatch
) -> None:
    path_a = tmp_path / "a.fits"
    path_b = tmp_path / "b.fits"
    header_a = _simple_header()
    header_b = dict(header_a)
    header_b["CRVAL1"] = header_b["CRVAL1"] + 0.02
    data = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    torchfits.write(str(path_a), data, header=header_a, overwrite=True)
    torchfits.write(str(path_b), data + 1.0, header=header_b, overwrite=True)

    calls = 0
    original = windowing_mod._prepare_target

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(image_plane_coadd_mod, "_prepare_target", counted)

    result = coadd_image_plane(
        [ImageObservation(str(path_a)), ImageObservation(str(path_b))]
    )

    assert calls == 1
    assert result.data.shape == (8, 8)


def test_project_observation_large_tan_windowing_matches_full_reprojection(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "source.fits"
    source_header = _simple_header()
    source_header["NAXIS1"] = 64
    source_header["NAXIS2"] = 64
    source_header["CRPIX1"] = 32.0
    source_header["CRPIX2"] = 32.0
    data = torch.arange(64 * 64, dtype=torch.float32).reshape(64, 64)
    torchfits.write(str(path), data, header=source_header, overwrite=True)

    target_header = dict(source_header)
    target_header["NAXIS1"] = 128
    target_header["NAXIS2"] = 96
    target_header["CRPIX1"] = 64.0
    target_header["CRPIX2"] = 48.0
    target_shape = (96, 128)
    device = data.device
    prepared_target = windowing_mod._prepare_target(
        header=target_header,
        target_wcs=torchsky.WCS(target_header),
        target_shape=target_shape,
        device=device,
        include_pixel_area=False,
        include_representative_pixel_area=True,
        include_quadrature=False,
    )
    obs = ImageObservation(str(path))
    image, header = torchfits.read(str(path), return_header=True)
    image = validate_image_plane_2d(image)
    source_wcs = torchsky.WCS(dict(header))

    original_threshold = windowing_mod._LARGE_TAN_WINDOWING_MIN_TARGET_PIXELS
    original_resample = image_plane_projection_mod._resample_weighted_to_target
    try:
        forced_shapes: list[tuple[int, int]] = []

        def wrapped_resample(*args, **kwargs):
            target = args[3]
            forced_shapes.append(tuple(int(v) for v in target.shape))
            return original_resample(*args, **kwargs)

        monkeypatch.setattr(
            image_plane_projection_mod, "_resample_weighted_to_target", wrapped_resample
        )
        monkeypatch.setattr(windowing_mod, "_LARGE_TAN_WINDOWING_MIN_TARGET_PIXELS", 0)
        forced_numer, forced_weight, _, _ = (
            image_plane_projection_mod._project_observation_to_target(
                obs,
                image=image,
                header_dict_obs=dict(header),
                source_wcs=source_wcs,
                prepared_target=prepared_target,
                target_header=target_header,
                target_shape=target_shape,
                device=device,
                footprint_order=None,
                resampling_kernel="bilinear",
                value_semantics="surface_brightness",
            )
        )
        assert forced_shapes
        assert (
            forced_shapes[0][0] < target_shape[0]
            or forced_shapes[0][1] < target_shape[1]
        )

        monkeypatch.setattr(
            windowing_mod,
            "_LARGE_TAN_WINDOWING_MIN_TARGET_PIXELS",
            10**9,
        )
        full_numer, full_weight, _, _ = (
            image_plane_projection_mod._project_observation_to_target(
                obs,
                image=image,
                header_dict_obs=dict(header),
                source_wcs=source_wcs,
                prepared_target=prepared_target,
                target_header=target_header,
                target_shape=target_shape,
                device=device,
                footprint_order=None,
                resampling_kernel="bilinear",
                value_semantics="surface_brightness",
            )
        )
    finally:
        monkeypatch.setattr(
            windowing_mod,
            "_LARGE_TAN_WINDOWING_MIN_TARGET_PIXELS",
            original_threshold,
        )
        monkeypatch.setattr(
            image_plane_projection_mod,
            "_resample_weighted_to_target",
            original_resample,
        )

    assert_close_on_device(forced_numer, full_numer, atol=0.0, rtol=0.0)
    assert_close_on_device(forced_weight, full_weight, atol=0.0, rtol=0.0)
    assert forced_weight.sum().item() > 0.0


def test_project_observation_window_no_overlap_skips_geometry(monkeypatch) -> None:
    source_header = _simple_header()
    image = torch.ones((8, 8), dtype=torch.float32)
    target_header = dict(source_header)
    target_header["CRVAL1"] = target_header["CRVAL1"] + 10.0
    target_shape = (8, 8)
    prepared_target = windowing_mod._prepare_target(
        header=target_header,
        target_wcs=torchsky.WCS(target_header),
        target_shape=target_shape,
        device=torch.device("cpu"),
        include_pixel_area=False,
        include_representative_pixel_area=True,
        include_quadrature=False,
    )

    def no_overlap_window(**kwargs):
        return None, (0, 0, 0, 0)

    def fail_projection(*args, **kwargs):
        raise AssertionError("no-overlap window must not call geometry projection")

    monkeypatch.setattr(
        image_plane_projection_mod, "_window_large_tan_target", no_overlap_window
    )
    monkeypatch.setattr(
        image_plane_projection_mod,
        "project_image_observation_geometry",
        fail_projection,
    )

    projected = image_plane_projection_mod._project_observation_to_target_chunk(
        ImageObservation("mock.fits"),
        image=image,
        header_dict_obs=source_header,
        source_wcs=torchsky.WCS(source_header),
        prepared_target=prepared_target,
        target_header=target_header,
        target_shape=target_shape,
        device=torch.device("cpu"),
        footprint_order=None,
        resampling_kernel="bilinear",
        value_semantics="surface_brightness",
        drizzle_pixfrac=1.0,
    )

    assert projected.target_window == (0, 0, 0, 0)
    assert projected.numer.shape == (0, 0)
    assert projected.weight.shape == (0, 0)
    assert projected.overlap.shape == (0, 0)


def test_surface_brightness_area_scaling_gate_uses_target_vs_harmonic_source_scale(
    tmp_path: Path,
) -> None:
    data = torch.ones((8, 8), dtype=torch.float32)
    path_a = tmp_path / "a.fits"
    path_b = tmp_path / "b.fits"
    header_a = _simple_header()
    header_b = _simple_header()
    header_a["CDELT1"] = -0.006
    header_a["CDELT2"] = 0.006
    header_b["CDELT1"] = -0.014
    header_b["CDELT2"] = 0.014
    torchfits.write(str(path_a), data, header=header_a, overwrite=True)
    torchfits.write(str(path_b), data, header=header_b, overwrite=True)
    observations = [ImageObservation(str(path_a)), ImageObservation(str(path_b))]

    coarse_target = _simple_header()
    coarse_target["CDELT1"] = -0.010
    coarse_target["CDELT2"] = 0.010
    fine_target = _simple_header()
    fine_target["CDELT1"] = -0.004
    fine_target["CDELT2"] = 0.004

    assert calibration_mod._should_apply_surface_brightness_area_scaling(
        observations,
        target_header=coarse_target,
        target_shape=(8, 8),
    )
    assert not calibration_mod._should_apply_surface_brightness_area_scaling(
        observations,
        target_header=fine_target,
        target_shape=(8, 8),
    )


def test_coadd_image_plane_skips_target_grid_for_identical_wcs_without_footprints(
    tmp_path, monkeypatch
) -> None:
    path_a = tmp_path / "a.fits"
    path_b = tmp_path / "b.fits"
    header = _simple_header()
    data = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    torchfits.write(str(path_a), data, header=header, overwrite=True)
    torchfits.write(str(path_b), data + 1.0, header=header, overwrite=True)

    calls = 0
    original = windowing_mod._build_target_world_grid

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(image_plane_coadd_mod, "_build_target_world_grid", counted)

    result = coadd_image_plane(
        [ImageObservation(str(path_a)), ImageObservation(str(path_b))]
    )

    assert calls == 0
    assert result.data.shape == (8, 8)
