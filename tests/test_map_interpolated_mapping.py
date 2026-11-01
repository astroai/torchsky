"""Verified coarse-lattice interpolation of the target -> source pixel mapping."""

from __future__ import annotations

import math

import pytest
import torch

import torchsky
import torchsky.map.image_plane_projection as projection_mod
import torchsky.map.interpolated_mapping as im
import torchsky.map.windowing as windowing_mod

pytest.importorskip("torchsky._cpp")

CPU = torch.device("cpu")


def _strong_sip_header(size: int) -> dict:
    # Cubic/quintic terms displace the edge by several pixels; the negative cubic
    # makes the forward map non-monotone ~400 px from the reference pixel, where the
    # Newton inverse (and hence the round-trip gate) fails.
    return {
        "NAXIS": 2,
        "NAXIS1": size,
        "NAXIS2": size,
        "CTYPE1": "RA---TAN-SIP",
        "CTYPE2": "DEC--TAN-SIP",
        "CRPIX1": size / 2 + 0.5,
        "CRPIX2": size / 2 + 0.5,
        "CRVAL1": 50.0,
        "CRVAL2": -20.0,
        "CD1_1": -2.0e-5,
        "CD1_2": 3.0e-6,
        "CD2_1": 3.0e-6,
        "CD2_2": 2.0e-5,
        "A_ORDER": 5,
        "B_ORDER": 5,
        "A_2_0": 3.0e-5,
        "A_1_1": -2.0e-5,
        "A_3_0": -2.0e-6,
        "A_1_2": 1.0e-6,
        "A_5_0": -2.0e-11,
        "B_0_2": -2.5e-5,
        "B_1_1": 1.5e-5,
        "B_0_3": -2.0e-6,
        "B_2_1": 1.0e-6,
        "B_0_5": 1.0e-11,
    }


def _target_header(size: int, scale_deg: float, *, dra: float = 0.0) -> dict:
    return {
        "NAXIS": 2,
        "NAXIS1": size,
        "NAXIS2": size,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": size / 2 + 0.5,
        "CRPIX2": size / 2 + 0.5,
        "CRVAL1": 50.0 + dra,
        "CRVAL2": -20.0,
        "CDELT1": -scale_deg,
        "CDELT2": scale_deg,
    }


def _exact(evaluate, h: int, w: int):
    yy, xx = torch.meshgrid(
        torch.arange(h, dtype=torch.float64),
        torch.arange(w, dtype=torch.float64),
        indexing="ij",
    )
    return evaluate(xx, yy)


def test_strongly_distorted_sip_error_within_tolerance_and_validity_preserved() -> None:
    src = 200
    source_wcs = torchsky.WCS(_strong_sip_header(src))
    # Target 5x larger than the detector: most of it is outside the source domain and
    # the outermost part is where the SIP inverse breaks down.
    n = 1000
    target_wcs = torchsky.WCS(_target_header(n, 2.0e-5))
    evaluate = projection_mod._make_target_to_source_evaluator(
        source_wcs, target_wcs, (src, src)
    )
    stats: dict = {}
    xi, yi, oki = im.interpolated_inverse_mapping(
        evaluate,
        n,
        n,
        device=CPU,
        bounds_xy=(src - 1.0, src - 1.0),
        stats=stats,
    )
    xe, ye, oke = _exact(evaluate, n, n)
    assert not bool(oke.all()), "fixture must contain invalid (non-invertible) pixels"
    inside = (xe >= 0) & (xe <= src - 1) & (ye >= 0) & (ye <= src - 1)
    assert int(inside.sum()) > 10_000 and int((~inside).sum()) > 10_000

    err = torch.hypot(xi - xe, yi - ye)
    valid_both = oki & oke & torch.isfinite(xe)
    assert float(err[valid_both].max()) <= im.MAPPING_TOLERANCE_PX
    # Validity (round trip) and in-bounds decisions are identical to the exact path.
    xi_in = (xi >= 0) & (xi <= src - 1) & (yi >= 0) & (yi <= src - 1)
    assert torch.equal(xi_in & oki, inside & oke)
    # The exact fallback is a small fraction of the window.
    # Extreme fixture (most of the window is far outside the domain where the
    # distortion is wild): still a non-trivial share is interpolated.
    assert stats["exact_pixels"] < 0.8 * n * n
    assert stats["max_verified_error_px"] <= im.MAPPING_TOLERANCE_PX


def test_tolerance_violation_triggers_refinement_and_exact_fallback() -> None:
    h, w = 90, 120

    def bump(x: torch.Tensor, y: torch.Tensor):
        # Smooth shear plus a localised 0.05 px bump (width ~3 px) that a stride-16
        # lattice cannot represent.
        gx = 0.05 * torch.exp(-((x - 40.0) ** 2 + (y - 30.0) ** 2) / (2 * 3.0**2))
        xs = 1.0e-3 * x + 2.0e-5 * y * y / 100.0 + gx
        ys = y + 3.0e-4 * x * y / 100.0
        return xs, ys, torch.ones_like(xs, dtype=torch.bool)

    stats: dict = {}
    xi, yi, ok = im.interpolated_inverse_mapping(
        bump, h, w, device=CPU, stride=16, tolerance=1.0e-3, min_stride=4, stats=stats
    )
    xe, ye, _ = _exact(bump, h, w)
    assert bool(ok.all())
    assert float(torch.hypot(xi - xe, yi - ye).max()) <= 1.0e-3
    # The bump forced at least one cell to be refined or evaluated exactly, while the
    # smooth remainder was still interpolated.
    assert len(stats["levels"]) > 1 or stats["exact_pixels"] > 0
    assert stats["exact_pixels"] < 0.5 * h * w


def test_invalid_nodes_fall_back_to_exact_and_keep_validity() -> None:
    h, w = 80, 100

    def half_plane(x: torch.Tensor, y: torch.Tensor):
        xs = x * 1.0003 + 1.0e-6 * y * y
        ys = y * 0.9998 + 2.0e-6 * x * x
        # Invalid (diverged inverse) wherever x + y > 120, and NaN on part of it.
        ok = (x + y) <= 120.0
        xs = torch.where((x + y) > 150.0, torch.full_like(xs, float("nan")), xs)
        return xs, ys, ok

    xi, yi, oki = im.interpolated_inverse_mapping(
        half_plane, h, w, device=CPU, stride=8, tolerance=1.0e-3
    )
    xe, ye, oke = _exact(half_plane, h, w)
    assert torch.equal(oki, oke)
    good = oke & torch.isfinite(xe)
    assert float(torch.hypot(xi - xe, yi - ye)[good].max()) <= 1.0e-3
    assert bool(torch.isnan(xi[(torch.arange(h)[:, None] + torch.arange(w)[None, :]) > 150]).all())


def test_bounds_band_makes_in_bounds_decision_exact() -> None:
    h, w = 64, 64

    def affine(x: torch.Tensor, y: torch.Tensor):
        return 0.5 * x + 1.0e-4 * y * y, 0.5 * y, torch.ones_like(x, dtype=torch.bool)

    xi, yi, ok = im.interpolated_inverse_mapping(
        affine, h, w, device=CPU, bounds_xy=(20.0, 20.0), stride=8
    )
    xe, ye, _ = _exact(affine, h, w)
    inb_i = (xi >= 0) & (xi <= 20.0) & (yi >= 0) & (yi <= 20.0)
    inb_e = (xe >= 0) & (xe <= 20.0) & (ye >= 0) & (ye <= 20.0)
    assert torch.equal(inb_i, inb_e)


def test_small_window_uses_exact_evaluation() -> None:
    calls: list[int] = []

    def ident(x: torch.Tensor, y: torch.Tensor):
        calls.append(int(x.numel()))
        return x + 0.0, y + 0.0, torch.ones_like(x, dtype=torch.bool)

    stats: dict = {}
    xi, yi, _ok = im.interpolated_inverse_mapping(ident, 20, 30, device=CPU, stats=stats)
    assert calls == [600] and stats["exact_pixels"] == 600


def test_mapping_mode_validation_and_context() -> None:
    assert im.resolve_mapping_mode(None) == im.MAPPING_MODE
    with im.mapping_mode("exact"):
        assert im.resolve_mapping_mode(None) == "exact"
    assert im.MAPPING_MODE == "interpolated"
    with pytest.raises(ValueError):
        im.resolve_mapping_mode("nearest")


def _resample(source_header, target_header, image, *, kernel, semantics, mapping, window=None):
    shape = (int(target_header["NAXIS2"]), int(target_header["NAXIS1"]))
    target = windowing_mod._prepare_target(
        header=target_header,
        target_wcs=torchsky.WCS(target_header),
        target_shape=shape,
        device=CPU,
        include_pixel_area=semantics == "flux",
        include_representative_pixel_area=semantics != "flux",
        include_quadrature=semantics == "flux",
        include_world_grid=False,
    )
    weight = torch.ones_like(image)
    return projection_mod._resample_weighted_to_target(
        image,
        weight,
        torchsky.WCS(source_header),
        target,
        resampling_kernel=kernel,
        value_semantics=semantics,
        return_variance_weight=True,
        mapping=mapping,
    )


@pytest.mark.parametrize("semantics", ["surface_brightness", "flux"])
@pytest.mark.parametrize("kernel", ["bilinear", "lanczos3"])
def test_resample_interpolated_matches_exact_for_sip_source(
    kernel, semantics, monkeypatch
) -> None:
    # White-noise image: data error ~ gradient (~2 sigma/px) x mapping error, so a
    # 1e-4 sigma parity needs a mapping tolerance of ~1e-5 px.
    monkeypatch.setattr(im, "MAPPING_TOLERANCE_PX", 1.0e-5)
    src = 200
    source_header = _strong_sip_header(src)
    target_header = _target_header(300, 2.5e-5, dra=2.0e-4)
    gen = torch.Generator().manual_seed(3)
    image = torch.randn((src, src), generator=gen, dtype=torch.float32) + 10.0
    ex = _resample(source_header, target_header, image, kernel=kernel, semantics=semantics, mapping="exact")
    ip = _resample(source_header, target_header, image, kernel=kernel, semantics=semantics, mapping="interpolated")
    assert float(ex[1].sum()) > 0.0
    assert torch.equal(ex[1] > 0, ip[1] > 0), "coverage support must be identical"
    scale = float(ex[0][ex[1] > 0].std())
    assert float((ex[0] - ip[0]).abs().max()) / scale <= 1.0e-4
    assert float((ex[1] - ip[1]).abs().max()) / float(ex[1].max()) <= 1.0e-4
    assert ex[2] is not None and ip[2] is not None
    assert float((ex[2] - ip[2]).abs().max()) / float(ex[2].max()) <= 1.0e-4


def test_resample_interpolated_matches_exact_for_tpv_source(monkeypatch) -> None:
    monkeypatch.setattr(im, "MAPPING_TOLERANCE_PX", 1.0e-5)
    src = 160
    source_header = {
        "NAXIS": 2,
        "NAXIS1": src,
        "NAXIS2": src,
        "CTYPE1": "RA---TPV",
        "CTYPE2": "DEC--TPV",
        "CRPIX1": src / 2 + 0.5,
        "CRPIX2": src / 2 + 0.5,
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
    assert projection_mod._use_interpolated_mapping(torchsky.WCS(source_header), None)
    target_header = _target_header(240, 2.5e-5, dra=1.0e-4)
    gen = torch.Generator().manual_seed(5)
    image = torch.randn((src, src), generator=gen, dtype=torch.float32) + 10.0
    for semantics in ("surface_brightness", "flux"):
        ex = _resample(source_header, target_header, image, kernel="lanczos3", semantics=semantics, mapping="exact")
        ip = _resample(source_header, target_header, image, kernel="lanczos3", semantics=semantics, mapping="interpolated")
        assert float(ex[1].sum()) > 0.0
        assert torch.equal(ex[1] > 0, ip[1] > 0)
        scale = float(ex[0][ex[1] > 0].std())
        assert float((ex[0] - ip[0]).abs().max()) / scale <= 1.0e-4


def test_interpolated_mapping_default_is_used_for_sip_and_skipped_for_tan() -> None:
    sip_wcs = torchsky.WCS(_strong_sip_header(64))
    tan_wcs = torchsky.WCS(_target_header(64, 2.0e-5))
    assert projection_mod._use_interpolated_mapping(sip_wcs, None)
    assert not projection_mod._use_interpolated_mapping(tan_wcs, None)
    assert not projection_mod._use_interpolated_mapping(sip_wcs, "exact")
    assert math.isfinite(im.MAPPING_TOLERANCE_PX)
