"""Flux-semantics source pixel areas are evaluated only on the kernel-tap window."""

from __future__ import annotations

import pytest
import torch

import torchsky
import torchsky.map.image_plane_projection as projection_mod
import torchsky.map.operators.pixel_area as pixel_area_mod
import torchsky.map.windowing as windowing_mod

pytest.importorskip("torchsky._cpp")

CPU = torch.device("cpu")


def _tpv_header(size: int) -> dict:
    return {
        "NAXIS": 2,
        "NAXIS1": size,
        "NAXIS2": size,
        "CTYPE1": "RA---TPV",
        "CTYPE2": "DEC--TPV",
        "CRPIX1": size / 2 + 0.5,
        "CRPIX2": size / 2 + 0.5,
        "CRVAL1": 50.0,
        "CRVAL2": -20.0,
        "CD1_1": -2.0e-5,
        "CD1_2": 1.0e-6,
        "CD2_1": 1.0e-6,
        "CD2_2": 2.0e-5,
        "PV1_0": 0.0,
        "PV1_1": 1.0,
        "PV1_2": 0.0,
        "PV1_4": 3.0,
        "PV1_5": -1.0,
        "PV1_6": 2.0,
        "PV2_0": 0.0,
        "PV2_1": 1.0,
        "PV2_2": 0.0,
        "PV2_4": -2.0,
        "PV2_5": 1.5,
        "PV2_6": 1.0,
    }


def _target_header(size: int, scale_deg: float, *, dra: float, ddec: float) -> dict:
    return {
        "NAXIS": 2,
        "NAXIS1": size,
        "NAXIS2": size,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": size / 2 + 0.5,
        "CRPIX2": size / 2 + 0.5,
        "CRVAL1": 50.0 + dra,
        "CRVAL2": -20.0 + ddec,
        "CDELT1": -scale_deg,
        "CDELT2": scale_deg,
    }


def _resample(source_header, target_header, image, weight, *, kernel, mapping):
    shape = (int(target_header["NAXIS2"]), int(target_header["NAXIS1"]))
    target = windowing_mod._prepare_target(
        header=target_header,
        target_wcs=torchsky.WCS(target_header),
        target_shape=shape,
        device=CPU,
        include_pixel_area=True,
        include_quadrature=False,
        include_world_grid=False,
    )
    return projection_mod._resample_weighted_to_target(
        image,
        weight,
        torchsky.WCS(source_header),
        target,
        resampling_kernel=kernel,
        value_semantics="flux",
        return_variance_weight=True,
        mapping=mapping,
    )


def test_pixel_area_window_equals_slice_of_full_map() -> None:
    wcs = torchsky.WCS(_tpv_header(96))
    full = pixel_area_mod.pixel_area_sr_map(wcs, (96, 96), device=CPU, chunk_size=16)
    window = (7, 61, 30, 95)
    part = pixel_area_mod.pixel_area_sr_map(
        wcs, (96, 96), device=CPU, chunk_size=16, window=window
    )
    assert part.shape == (54, 65)
    torch.testing.assert_close(part, full[7:61, 30:95], rtol=1e-6, atol=0.0)
    empty = pixel_area_mod.pixel_area_sr_map(
        wcs, (96, 96), device=CPU, window=(10, 10, 0, 96)
    )
    assert empty.shape == (0, 96)


def test_source_tap_window_covers_taps_and_ignores_nonfinite() -> None:
    x = torch.tensor([[10.2, 20.7], [float("nan"), -3.0]], dtype=torch.float64)
    y = torch.tensor([[5.5, 30.1], [1.0, 2.0]], dtype=torch.float64)
    # x clamps to [0, 20.7], y to [2.0, 30.1] (the NaN sample is ignored).
    assert projection_mod._source_tap_window(x, y, (40, 50), radius=4) == (
        0,
        35,
        0,
        25,
    )
    nan = torch.full((2, 2), float("nan"), dtype=torch.float64)
    assert projection_mod._source_tap_window(nan, nan, (40, 50), radius=4) == (
        0,
        0,
        0,
        0,
    )


@pytest.mark.parametrize("mapping", ["interpolated", "exact"])
@pytest.mark.parametrize("kernel", ["lanczos3", "bilinear", "bicubic"])
def test_windowed_flux_area_matches_full_area_map(monkeypatch, kernel, mapping) -> None:
    # The target covers only a corner of the source: the windowed path evaluates a
    # fraction of the source area map and must reproduce the full-map result.
    src = 160
    source_header = _tpv_header(src)
    target_header = _target_header(64, 2.0e-5, dra=6.0e-4, ddec=-8.0e-4)
    gen = torch.Generator().manual_seed(11)
    image = torch.randn((src, src), generator=gen, dtype=torch.float32) + 10.0
    weight = torch.rand((src, src), generator=gen, dtype=torch.float32) + 0.5
    weight[40:44, :] = 0.0  # a masked band inside the window

    evaluated: list[int] = []
    real_map = projection_mod.pixel_area_sr_map

    def counting_map(*args, **kwargs):
        out = real_map(*args, **kwargs)
        evaluated.append(int(out.numel()))
        return out

    monkeypatch.setattr(projection_mod, "pixel_area_sr_map", counting_map)
    windowed = _resample(
        source_header, target_header, image, weight, kernel=kernel, mapping=mapping
    )
    assert evaluated and max(evaluated) < src * src // 2

    monkeypatch.setattr(projection_mod, "_SOURCE_TAP_RADIUS", {})
    evaluated.clear()
    full = _resample(
        source_header, target_header, image, weight, kernel=kernel, mapping=mapping
    )
    assert evaluated == [src * src]

    assert float(full[1].sum()) > 0.0
    assert torch.equal(full[1] > 0, windowed[1] > 0), (
        "coverage support must be identical"
    )
    for a, b in zip(full, windowed):
        assert a is not None and b is not None
        torch.testing.assert_close(b, a, rtol=1e-6, atol=0.0)
