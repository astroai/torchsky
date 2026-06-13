"""Verification tests for exact mathematical adjoints of the map projection operators."""

from __future__ import annotations

import pytest
import torch

from torchsky import WCS
from torchsky.map.operators import (
    InterpolationGeometryOp,
    ConservativePolygonRemapOp,
    project_image_observation_geometry,
    pixel_area_sr_map,
    representative_pixel_area_sr,
)
from torchsky.map.windowing import _prepare_target
from torchsky.map.image_plane import (
    _resample_weighted_to_target,
    _forward_project_weighted_to_target,
)

pytest.importorskip("torchsky.cpp")


def _simple_header(nx=8, ny=8, crval1=12.0, crval2=-3.0, cdelt1=-0.01, cdelt2=0.01) -> dict:
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


@pytest.mark.parametrize("kernel_name", ["nearest", "bilinear", "lanczos3"])
@pytest.mark.parametrize("value_semantics", ["surface_brightness", "flux"])
def test_interpolation_resampler_adjoint(kernel_name: str, value_semantics: str) -> None:
    device = torch.device("cpu")
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
        device=device,
        include_pixel_area=(value_semantics == "flux"),
        include_representative_pixel_area=True,
    )

    # x represents the source image pixel values, requires_grad=True to compute adjoint via autograd
    x = torch.randn((src_h, src_w), dtype=torch.float32, device=device, requires_grad=True)
    source_weight = torch.ones((src_h, src_w), dtype=torch.float32, device=device)

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
    assert torch.any(y_pred != 0.0), f"y_pred is completely zero for {kernel_name} ({value_semantics})"

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


@pytest.mark.parametrize("value_semantics", ["surface_brightness", "flux"])
def test_conservative_polygon_remapper_adjoint(value_semantics: str) -> None:
    device = torch.device("cpu")
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
        device=device,
        include_pixel_area=(value_semantics == "flux"),
        include_representative_pixel_area=True,
    )

    x = torch.randn((src_h, src_w), dtype=torch.float32, device=device, requires_grad=True)
    source_weight = torch.ones((src_h, src_w), dtype=torch.float32, device=device)

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

    assert torch.any(y_pred != 0.0), f"y_pred is completely zero for conservative_polygon ({value_semantics})"

    y = torch.randn_like(y_pred)
    val1 = torch.sum(y_pred * y)
    val1.backward()

    val2 = torch.sum(x * x.grad)

    torch.testing.assert_close(val1, val2, rtol=1e-4, atol=1e-4)
