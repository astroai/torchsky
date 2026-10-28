"""
Unit tests for torchsky.viz renderers.

Tests stretch algorithms, colormap LUTs, PNG encoding, HiPS tile generation,
MOC GeoJSON, and validation diagnostics — all without a running HTTP server.
"""

from __future__ import annotations

import math
from typing import Any, cast

import pytest
import torch


# ---------------------------------------------------------------------------
# image.py
# ---------------------------------------------------------------------------


class TestZScale:
    def test_basic(self):
        from torchsky.viz.renderers.image import zscale_limits

        t = torch.linspace(0, 1000, 10_000)
        z1, z2 = zscale_limits(t)
        assert z1 < z2
        assert 0 <= z1 < 100
        assert 900 < z2 <= 1000

    def test_uniform_image(self):
        from torchsky.viz.renderers.image import zscale_limits

        t = torch.full((100, 100), 42.0)
        z1, z2 = zscale_limits(t.flatten())
        # Should not crash; z1 <= z2
        assert z1 <= z2

    def test_nan_robust(self):
        from torchsky.viz.renderers.image import zscale_limits

        t = torch.linspace(0, 1000, 1000)
        t[::10] = float("nan")
        z1, z2 = zscale_limits(t)
        assert math.isfinite(z1) and math.isfinite(z2)

    def test_high_dr(self):
        """ZScale should stay within signal on a high-DR image."""
        from torchsky.viz.renderers.image import zscale_limits

        t = torch.cat([torch.linspace(1, 10, 900), torch.tensor([1e6])])  # one spike
        z1, z2 = zscale_limits(t)
        assert z2 < 1e5  # spike should not dominate


class TestApplyStretch:
    @pytest.mark.parametrize(
        "stretch", ["linear", "log", "sqrt", "asinh", "power", "histeq"]
    )
    def test_output_range(self, stretch):
        from torchsky.viz.renderers.image import apply_stretch

        t = torch.linspace(0, 1000, 500)
        out = apply_stretch(t, 0.0, 1000.0, stretch)
        assert out.shape == t.shape
        assert float(out.min().item()) >= -1e-6
        assert float(out.max().item()) <= 1.0 + 1e-6

    def test_log_monotone(self):
        from torchsky.viz.renderers.image import apply_stretch

        t = torch.linspace(0, 1000, 200)
        out = apply_stretch(t, 0.0, 1000.0, "log")
        diffs = out[1:] - out[:-1]
        assert (diffs >= -1e-6).all()

    def test_nan_passthrough(self):
        from torchsky.viz.renderers.image import apply_stretch

        t = torch.tensor([0.0, float("nan"), 1000.0])
        out = apply_stretch(t, 0.0, 1000.0, "linear")
        assert float(out[0].item()) == pytest.approx(0.0)
        assert float(out[2].item()) == pytest.approx(1.0)

    def test_zscale_is_not_a_low_level_stretch(self):
        from torchsky.viz.renderers.image import apply_stretch

        with pytest.raises(ValueError, match="Unknown stretch"):
            cast(Any, apply_stretch)(torch.arange(3), 0.0, 2.0, "zscale")


class TestColormaps:
    def test_all_cmaps_have_256_entries(self):
        from torchsky.viz.renderers.image import COLORMAP_NAMES, get_lut

        for name in COLORMAP_NAMES:
            lut = get_lut(name)
            assert lut.shape == (256, 3), f"{name}: shape {lut.shape}"
            assert lut.dtype.name == "uint8"

    def test_unknown_cmap_raises(self):
        from torchsky.viz.renderers.image import get_lut

        with pytest.raises(ValueError):
            get_lut("nonexistent_cmap")

    def test_gray_monotone(self):
        from torchsky.viz.renderers.image import get_lut

        lut = get_lut("gray")
        assert (lut[1:, 0] >= lut[:-1, 0]).all()

    def test_rdbu_r_midpoint(self):
        """RdBu_r should be near-white at index 128."""
        from torchsky.viz.renderers.image import get_lut

        lut = get_lut("rdbu_r")
        mid = lut[128]
        assert int(mid.min()) > 200  # all channels bright at center


class TestPngEncode:
    def test_returns_valid_png(self):
        from torchsky.viz.renderers.image import to_png

        t = torch.rand(64, 64)
        png = to_png(t, cmap="gray")
        assert isinstance(png, bytes)
        assert png[:4] == b"\x89PNG"

    def test_with_alpha_mask(self):
        from torchsky.viz.renderers.image import to_png

        t = torch.rand(32, 32)
        mask = torch.ones(32, 32, dtype=torch.bool)
        mask[:16, :16] = False
        png = to_png(t, cmap="viridis", alpha_mask=mask)
        assert png[:4] == b"\x89PNG"

    def test_symmetric_png(self):
        from torchsky.viz.renderers.image import to_png_symmetric

        t = torch.linspace(-5, 5, 100).reshape(10, 10)
        png = to_png_symmetric(t, vmax=5.0, cmap="rdbu_r")
        assert png[:4] == b"\x89PNG"


class TestDynamicRange:
    def test_high_dr(self):
        from torchsky.viz.renderers.image import detect_dynamic_range

        # True high-DR: positive values spanning >3 decades (p1=1, p99=10000+)
        t = torch.cat(
            [
                torch.linspace(1, 10, 500),  # 1-10 (low end)
                torch.linspace(100, 50000, 500),  # 100-50000 (high end)
            ]
        )
        dr = detect_dynamic_range(t.reshape(1, -1))
        assert dr["dr_ratio"] > 1000
        assert "high_dynamic_range" in dr["warnings"]

    def test_negatives(self):
        from torchsky.viz.renderers.image import detect_dynamic_range

        t = torch.linspace(-100, 100, 1000)
        dr = detect_dynamic_range(t.reshape(1, -1))
        assert dr["neg_fraction"] > 0.4
        assert "negatives" in dr["warnings"]

    def test_clean_image(self):
        from torchsky.viz.renderers.image import detect_dynamic_range

        t = torch.linspace(0, 100, 1000)
        dr = detect_dynamic_range(t.reshape(1, -1))
        assert dr["recommended_stretch"] == "linear"


# ---------------------------------------------------------------------------
# hips_tile.py
# ---------------------------------------------------------------------------


class TestHipsTile:
    def _make_wcs(self, h=256, w=256):
        """Create a minimal torchsky.WCS for a square image."""
        from torchsky.wcs import WCS

        header = {
            "NAXIS1": w,
            "NAXIS2": h,
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
            "CRPIX1": w // 2,
            "CRPIX2": h // 2,
            "CRVAL1": 150.0,
            "CRVAL2": 2.0,
            "CD1_1": -5e-5,
            "CD1_2": 0.0,
            "CD2_1": 0.0,
            "CD2_2": 5e-5,
        }
        return WCS(header)

    def test_tile_rows_are_north_up(self):
        """HiPS PNG tiles store scanlines top-down, so row 0 must be north."""
        from torchsky.sphere import healpix
        from torchsky.viz.renderers.hips_tile import _tile_lon_lat

        pix = int(
            healpix.ang2pix(
                8, torch.tensor([150.0]), torch.tensor([2.0]), nest=True, lonlat=True
            ).item()
        )
        _, lat = _tile_lon_lat(3, pix, tile_size=64)
        assert float(lat[0].mean()) > float(lat[-1].mean())

    def test_png_tile_is_north_up(self):
        """A north-bright image must render bright rows at the top of the tile."""
        import io

        import numpy as np
        from PIL import Image

        from torchsky.sphere import healpix
        from torchsky.viz.renderers.hips_tile import TILE_SIZE, render_hips_tile
        from torchsky.wcs import WCS

        h = w = 256
        # FITS y=0 is the southern row, so values increase northward.
        image = (
            torch.arange(h, dtype=torch.float32)
            .reshape(h, 1)
            .expand(h, w)
            .contiguous()
        )
        header = {
            "NAXIS1": w,
            "NAXIS2": h,
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
            "CRPIX1": w // 2,
            "CRPIX2": h // 2,
            "CRVAL1": 150.0,
            "CRVAL2": 2.0,
            "CD1_1": -1.0e-2,
            "CD1_2": 0.0,
            "CD2_1": 0.0,
            "CD2_2": 1.0e-2,
        }
        # Order 6 (~0.9 deg) fits entirely inside the 2.56 deg image.
        order = 6
        pix = int(
            healpix.ang2pix(
                1 << order,
                torch.tensor([150.0]),
                torch.tensor([2.0]),
                nest=True,
                lonlat=True,
            ).item()
        )
        png = render_hips_tile(
            image, WCS(header), order=order, pix=pix, vmin=0.0, vmax=h - 1
        )
        assert png is not None
        arr = np.asarray(Image.open(io.BytesIO(png)).convert("L"), dtype=float)
        top = arr[: TILE_SIZE // 4].mean()
        bottom = arr[-TILE_SIZE // 4 :].mean()
        assert top > bottom + 10.0

    def test_render_preview(self):
        from torchsky.viz.renderers.hips_tile import render_preview

        image = torch.rand(128, 128) * 1000
        png = render_preview(
            image,
            max_pixels=64,
            stretch="linear",
            limit_policy="zscale",
            cmap="gray",
        )
        assert png[:4] == b"\x89PNG"

    def test_compute_hips_order(self):
        from torchsky.viz.renderers.hips_tile import compute_hips_order

        wcs = self._make_wcs(256, 256)
        order = compute_hips_order(wcs, (256, 256))
        assert 1 <= order <= 13


# ---------------------------------------------------------------------------
# moc_geojson.py
# ---------------------------------------------------------------------------


class TestMocGeoJson:
    def test_empty_moc(self):
        from torchsky.viz.renderers.moc_geojson import moc_to_geojson
        from torchsky.sphere.moc import MOC

        moc = MOC(torch.empty(0, dtype=torch.int64))
        geojson = moc_to_geojson(moc)
        assert geojson["type"] == "FeatureCollection"
        assert isinstance(geojson["features"], list)

    def test_synthetic_moc(self):
        from torchsky.viz.renderers.moc_geojson import moc_to_geojson
        from torchsky.sphere.moc import MOC
        from torchsky.sphere.healpix import query_disc
        from torchsky.sphere.core import lonlat_to_unit_xyz
        import torch
        import math

        nside = 32  # order 5
        center_lon, center_lat = 150.0, 2.0
        vec = lonlat_to_unit_xyz(center_lon, center_lat)  # unit XYZ vector
        radius_rad = math.radians(1.0)
        pix = query_disc(nside, vec, radius_rad, nest=True)
        uniq = (4 * nside * nside + pix).to(torch.int64)
        moc = MOC(uniq)
        geojson = moc_to_geojson(moc)
        assert geojson["type"] == "FeatureCollection"
        assert geojson["properties"]["n_cells"] > 0


# ---------------------------------------------------------------------------
# validation.py
# ---------------------------------------------------------------------------


class TestValidationRenderers:
    def _make_pair(self, size=64):
        truth = torch.rand(size, size) * 1000
        model = truth + torch.randn(size, size) * 10
        sigma = torch.full((size, size), 10.0)
        return truth, model, sigma

    def test_residual_panel_pngs(self):
        from torchsky.viz.renderers.validation import residual_panel_pngs

        truth, model, sigma = self._make_pair()
        pngs = residual_panel_pngs(truth, model, sigma)
        assert set(pngs.keys()) == {"truth", "model", "residual", "pull"}
        for k, v in pngs.items():
            assert isinstance(v, bytes), f"{k} not bytes"
            assert v[:4] == b"\x89PNG", f"{k} not valid PNG"

    def test_pull_histogram(self):
        from torchsky.viz.renderers.validation import pull_histogram

        truth, model, sigma = self._make_pair()
        hist = pull_histogram(truth, model, sigma)
        assert "bins" in hist and "counts" in hist
        assert len(hist["bins"]) == len(hist["counts"])
        assert hist["fit"]["mu"] == pytest.approx(0.0, abs=0.5)
        assert hist["fit"]["sigma"] == pytest.approx(1.0, abs=0.3)

    def test_chi2_map(self):
        from torchsky.viz.renderers.validation import chi2_map_png

        truth, model, sigma = self._make_pair()
        png = chi2_map_png(truth, model, sigma)
        assert png[:4] == b"\x89PNG"

    def test_chi2_stats(self):
        from torchsky.viz.renderers.validation import chi2_stats

        truth, model, sigma = self._make_pair()
        stats = chi2_stats(truth, model, sigma)
        assert "mean_chi2" in stats
        # With Gaussian noise σ=10 and sigma=10, mean_chi2 ≈ 1
        assert 0.5 < stats["mean_chi2"] < 2.0


# ---------------------------------------------------------------------------
# LRU cache
# ---------------------------------------------------------------------------


class TestLRUCache:
    def test_basic(self):
        from torchsky.viz.cache import LRUTileCache

        cache = LRUTileCache(max_mb=1)
        cache.put("a", b"hello")
        assert cache.get("a") == b"hello"
        assert cache.get("missing") is None

    def test_eviction(self):
        from torchsky.viz.cache import LRUTileCache

        cache = LRUTileCache(max_mb=1)  # 1 MB max
        # Put 600 KB of data twice → should evict first entry
        cache.put("a", b"x" * 600_000)
        cache.put("b", b"y" * 600_000)
        assert cache.get("a") is None  # evicted
        assert cache.get("b") is not None

    def test_stats(self):
        from torchsky.viz.cache import LRUTileCache

        cache = LRUTileCache(max_mb=10)
        cache.put("k", b"v" * 1000)
        cache.get("k")
        cache.get("missing")
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
