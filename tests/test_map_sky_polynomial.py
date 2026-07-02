from __future__ import annotations

import pytest
import torch

from torchsky.map.sky import PolynomialSkyModel


class TestPolynomialSkyModel:
    def test_constant_image(self):
        img = torch.full((64, 64), 100.0, dtype=torch.float32)
        est = PolynomialSkyModel(order=1, statistic="huber")
        sky = est(img)
        assert sky.shape == (64, 64)
        assert torch.allclose(sky, img, atol=1e-4)

    def test_smooth_gradient(self):
        y = torch.linspace(0, 10, 64)
        x = torch.linspace(0, 10, 64)
        gy, gx = torch.meshgrid(y, x, indexing="ij")
        img = 50.0 + 2.0 * gx + 3.0 * gy
        est = PolynomialSkyModel(order=1, statistic="huber")
        sky = est(img)
        assert torch.allclose(sky, img, atol=1e-3)

    def test_quadratic_recovery(self):
        y = torch.linspace(-1, 1, 32)
        x = torch.linspace(-1, 1, 32)
        gy, gx = torch.meshgrid(y, x, indexing="ij")
        img = 10.0 + 5.0 * gx**2 - 3.0 * gy * gx
        est = PolynomialSkyModel(order=2)
        sky = est(img)
        assert torch.allclose(sky, img, atol=1e-3)

    def test_outlier_rejection(self):
        y = torch.linspace(-1, 1, 32)
        x = torch.linspace(-1, 1, 32)
        gy, gx = torch.meshgrid(y, x, indexing="ij")
        img = 100.0 + 10.0 * gx
        img_noisy = img.clone()
        img_noisy[8:12, 8:12] = 500.0
        est = PolynomialSkyModel(order=1, statistic="huber")
        sky = est(img_noisy)
        assert torch.allclose(sky, img, atol=2.0)

    def test_mask_excludes_sources(self):
        img = torch.full((32, 32), 50.0, dtype=torch.float32)
        img[10:15, 10:15] = 500.0
        mask = torch.zeros((32, 32), dtype=torch.bool)
        mask[10:15, 10:15] = True
        est = PolynomialSkyModel(order=0, statistic="huber")
        sky = est(img, mask)
        assert torch.allclose(sky, torch.full_like(sky, 50.0), atol=1e-3)

    def test_deterministic(self):
        torch.manual_seed(42)
        img = torch.randn(64, 64, dtype=torch.float32)
        est = PolynomialSkyModel(order=2, statistic="huber")
        sky_a = est(img)
        sky_b = est(img)
        assert torch.equal(sky_a, sky_b)

    def test_grad_through_sky(self):
        img = torch.randn(16, 16, dtype=torch.float64, requires_grad=True)
        est = PolynomialSkyModel(order=1, statistic="huber")
        sky = est(img)
        loss = sky.sum()
        loss.backward()
        assert img.grad is not None
        assert img.grad.isfinite().any()

    def test_validation_order(self):
        with pytest.raises(ValueError, match="order"):
            PolynomialSkyModel(order=-1)

    def test_validation_statistic(self):
        with pytest.raises(ValueError, match="statistic"):
            PolynomialSkyModel(statistic="invalid")

    def test_insufficient_pixels(self):
        img = torch.ones((2, 2), dtype=torch.float32)
        est = PolynomialSkyModel(order=3)
        with pytest.raises(ValueError, match="not enough valid pixels"):
            est(img)

    def test_gpu(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        img = torch.randn(32, 32, dtype=torch.float32, device="cuda")
        est = PolynomialSkyModel(order=1)
        sky = est(img)
        assert sky.device.type == "cuda"
