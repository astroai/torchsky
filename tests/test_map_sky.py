from __future__ import annotations

import pytest
import torch

from torchsky.map.sky import MeshSkyEstimator, huber_location, tukey_location


class TestRobustStats:
    def test_huber_location_on_constant(self):
        x = torch.full((100,), 5.0, dtype=torch.float64)
        result = huber_location(x, dim=0)
        assert result.shape == ()
        assert float(result) == pytest.approx(5.0, abs=1e-10)

    def test_huber_location_on_gaussian(self):
        torch.manual_seed(42)
        x = 3.0 + torch.randn(1000, dtype=torch.float64)
        result = huber_location(x, dim=0)
        assert float(result) == pytest.approx(3.0, abs=0.15)

    def test_huber_location_rejects_outliers(self):
        x = torch.cat([torch.full((90,), 0.0), torch.full((10,), 100.0)])
        result = huber_location(x, dim=0)
        assert float(result) == pytest.approx(0.0, abs=0.5)

    def test_tukey_location_on_constant(self):
        x = torch.full((100,), 5.0, dtype=torch.float64)
        result = tukey_location(x, dim=0)
        assert float(result) == pytest.approx(5.0, abs=1e-10)

    def test_tukey_location_rejects_outliers(self):
        x = torch.cat([torch.full((90,), 0.0), torch.full((10,), 100.0)])
        result = tukey_location(x, dim=0)
        assert float(result) == pytest.approx(0.0, abs=0.5)

    def test_batched_statistics(self):
        torch.manual_seed(42)
        a = torch.randn(5, 3, 100, dtype=torch.float64)
        result = huber_location(a, dim=-1)
        assert result.shape == (5, 3)

    def test_median_fallback(self):
        x = torch.tensor([1.0, 2.0, 3.0, 100.0])
        result = torch.median(x, dim=0).values
        assert float(result) == pytest.approx(2.0)  # torch returns lower median


class TestMeshSkyEstimator:
    def test_constant_image(self):
        image = torch.full((64, 64), 100.0, dtype=torch.float32)
        est = MeshSkyEstimator(block_size=16, statistic="median")
        sky = est(image)
        assert sky.shape == (64, 64)
        assert sky.dtype == torch.float32
        assert sky.abs().max() < 101.0
        assert sky.abs().min() > 99.0

    def test_smooth_gradient_preserved(self):
        H, W = 64, 64
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing="ij"
        )
        image = 10.0 + 5.0 * xx + 3.0 * yy
        est = MeshSkyEstimator(block_size=16, statistic="median")
        sky = est(image)
        residual = (image - sky).abs().mean()
        assert float(residual) < 2.0

    def test_outlier_rejection(self):
        image = torch.zeros((64, 64), dtype=torch.float32)
        image[30:34, 30:34] = 1000.0  # Compact source
        est = MeshSkyEstimator(block_size=16, statistic="huber")
        sky = est(image)
        max_sky = float(sky.max())
        assert max_sky < 50.0, f"huber failed to reject outliers (max_sky={max_sky})"
        est_med = MeshSkyEstimator(block_size=16, statistic="median")
        sky_med = est_med(image)
        max_sky_med = float(sky_med.max())
        assert max_sky_med < 50.0, (
            f"median failed to reject outliers (max_sky={max_sky_med})"
        )

    def test_mask_excludes_sources(self):
        image = torch.zeros((64, 64), dtype=torch.float32)
        image[30:34, 30:34] = 1000.0
        mask = torch.zeros((64, 64), dtype=torch.bool)
        mask[30:34, 30:34] = True
        for stat in ("median", "huber", "tukey"):
            est = MeshSkyEstimator(block_size=16, statistic=stat)
            sky_masked = est(image, mask=mask)
            assert float(sky_masked.max()) < 10.0, f"stat={stat}"

    def test_fully_masked_mesh_cells_do_not_poison_valid_background(self):
        yy, xx = torch.meshgrid(
            torch.arange(64, dtype=torch.float32),
            torch.arange(64, dtype=torch.float32),
            indexing="ij",
        )
        image = 5.0 + 0.01 * xx + 0.02 * yy
        mask = torch.zeros_like(image, dtype=torch.bool)
        mask[:, :32] = True
        est = MeshSkyEstimator(
            block_size=16,
            block_stride=16,
            statistic="median",
            mesh_filter="median",
        )

        sky = est(image, mask)

        assert torch.isfinite(sky).all()
        assert float((sky[:, 32:] - image[:, 32:]).abs().mean()) < 0.25

    def test_entirely_masked_image_fails_closed(self):
        image = torch.ones((32, 32), dtype=torch.float32)
        est = MeshSkyEstimator(block_size=16, statistic="median")

        with pytest.raises(ValueError, match="excludes every sky mesh cell"):
            est(image, torch.ones_like(image, dtype=torch.bool))

    def test_non_divisible_dimensions(self):
        image = torch.randn(70, 50, dtype=torch.float32)
        est = MeshSkyEstimator(block_size=16, statistic="median")
        sky = est(image)
        assert sky.shape == (70, 50)

    def test_all_statistics(self):
        image = torch.randn(32, 32, dtype=torch.float32) + 5.0
        for stat in ("median", "huber", "tukey"):
            est = MeshSkyEstimator(block_size=8, statistic=stat)
            sky = est(image)
            assert sky.shape == (32, 32)
            assert torch.isfinite(sky).all()

    def test_mesh_filter_options(self):
        image = torch.randn(64, 64, dtype=torch.float32) + 5.0
        for mf in ("median", "gaussian", None):
            est = MeshSkyEstimator(block_size=16, statistic="median", mesh_filter=mf)
            sky = est(image)
            assert sky.shape == (64, 64)

    def test_interpolation_modes(self):
        image = torch.randn(64, 64, dtype=torch.float32) + 5.0
        for ip in ("bilinear", "bicubic"):
            est = MeshSkyEstimator(block_size=16, statistic="median", interpolate=ip)
            sky = est(image)
            assert sky.shape == (64, 64)

    def test_gpu_compatible(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        image = torch.randn(64, 64, dtype=torch.float32, device="cuda")
        est = MeshSkyEstimator(block_size=16, statistic="huber")
        sky = est(image)
        assert sky.device.type == "cuda"
        assert sky.shape == (64, 64)

    def test_validation_errors(self):
        with pytest.raises(ValueError, match="unknown statistic"):
            MeshSkyEstimator(statistic="invalid")
        with pytest.raises(ValueError, match="unknown mesh_filter"):
            MeshSkyEstimator(mesh_filter="invalid")
        with pytest.raises(ValueError, match="unknown interpolate"):
            MeshSkyEstimator(interpolate="invalid")

    def test_deterministic(self):
        torch.manual_seed(42)
        image = torch.randn(64, 64, dtype=torch.float32)
        est = MeshSkyEstimator(block_size=16, statistic="huber")
        sky_a = est(image)
        sky_b = est(image)
        assert torch.equal(sky_a, sky_b)

    def test_grad_through_sky(self):
        image = torch.randn(32, 32, dtype=torch.float64, requires_grad=True)
        est = MeshSkyEstimator(block_size=8, statistic="huber")
        sky = est(image)
        loss = sky.sum()
        loss.backward()
        assert image.grad is not None
        assert image.grad.isfinite().any()

    def test_gradcheck_huber_sky(self):
        torch.manual_seed(7)
        x = torch.randn(4, 4, 16, dtype=torch.float64, requires_grad=True)

        def f(x):
            return huber_location(x, dim=-1)

        torch.autograd.gradcheck(f, (x,), atol=0.5, rtol=0.5, fast_mode=True)
        loss = f(x).sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.isfinite().all()


class TestMeshFilterNewApi:
    def test_median_with_size(self):
        image = torch.randn(64, 64, dtype=torch.float32) + 5.0
        est = MeshSkyEstimator(block_size=16, mesh_filter="median", mesh_filter_size=5)
        sky = est(image)
        assert sky.shape == (64, 64)

    def test_gaussian_with_size(self):
        image = torch.randn(64, 64, dtype=torch.float32) + 5.0
        est = MeshSkyEstimator(
            block_size=16, mesh_filter="gaussian", mesh_filter_size=5
        )
        sky = est(image)
        assert sky.shape == (64, 64)

    @pytest.mark.parametrize("legacy", ["median_3", "gaussian_3"])
    def test_legacy_filter_strings_are_rejected(self, legacy):
        with pytest.raises(ValueError, match="unknown mesh_filter"):
            MeshSkyEstimator(block_size=16, mesh_filter=legacy)

    @pytest.mark.parametrize("size", [0, 2])
    def test_filter_size_must_be_positive_and_odd(self, size):
        with pytest.raises(ValueError, match="positive odd"):
            MeshSkyEstimator(mesh_filter="median", mesh_filter_size=size)

    def test_larger_filter_gives_smoother(self):
        torch.manual_seed(42)
        image = torch.randn(64, 64, dtype=torch.float32)
        est_small = MeshSkyEstimator(
            block_size=16,
            statistic="median",
            mesh_filter="gaussian",
            mesh_filter_size=3,
        )
        est_large = MeshSkyEstimator(
            block_size=16,
            statistic="median",
            mesh_filter="gaussian",
            mesh_filter_size=7,
        )
        sky_small = est_small(image)
        sky_large = est_large(image)
        assert sky_large.std() < sky_small.std()


class TestBlockStride:
    def test_half_stride_coverage(self):
        image = torch.full((64, 64), 100.0, dtype=torch.float32)
        est = MeshSkyEstimator(block_size=16, block_stride=8, statistic="median")
        sky = est(image)
        assert sky.shape == (64, 64)
        assert torch.allclose(sky, torch.full_like(sky, 100.0), atol=1e-4)

    def test_half_stride_more_detail_than_no_overlap(self):
        torch.manual_seed(42)
        image = torch.randn(64, 64, dtype=torch.float32) + 5.0
        est_no = MeshSkyEstimator(block_size=16, statistic="median", mesh_filter=None)
        est_over = MeshSkyEstimator(
            block_size=16, block_stride=8, statistic="median", mesh_filter=None
        )
        sky_no = est_no(image)
        sky_over = est_over(image)
        assert sky_no.shape == sky_over.shape == (64, 64)
        assert sky_over.std() > sky_no.std()

    def test_non_divisible_with_overlap(self):
        image = torch.randn(70, 50, dtype=torch.float32)
        est = MeshSkyEstimator(block_size=16, block_stride=8, statistic="median")
        sky = est(image)
        assert sky.shape == (70, 50)

    def test_stride_equals_block_size(self):
        image = torch.randn(64, 64, dtype=torch.float32)
        est = MeshSkyEstimator(block_size=16, block_stride=16, statistic="median")
        sky = est(image)
        assert sky.shape == (64, 64)

    def test_stride_greater_than_block_raises(self):
        with pytest.raises(ValueError, match="block_stride"):
            MeshSkyEstimator(block_size=16, block_stride=32)


class TestSelfMaskSigma:
    def test_self_mask_rejects_outliers(self):
        image = torch.zeros((64, 64), dtype=torch.float32)
        image[30:34, 30:34] = 1000.0
        est = MeshSkyEstimator(block_size=16, statistic="median", self_mask_sigma=3.0)
        sky = est(image)
        assert float(sky.max()) < 10.0

    def test_self_mask_with_user_mask(self):
        image = torch.zeros((64, 64), dtype=torch.float32)
        image[30:34, 30:34] = 1000.0
        image[10:12, 10:12] = 500.0
        user_mask = torch.zeros((64, 64), dtype=torch.bool)
        user_mask[10:12, 10:12] = True
        est = MeshSkyEstimator(block_size=16, statistic="median", self_mask_sigma=3.0)
        sky = est(image, mask=user_mask)
        assert float(sky.max()) < 10.0

    def test_self_mask_default_disabled(self):
        image = torch.randn(64, 64, dtype=torch.float32) + 5.0
        est = MeshSkyEstimator(block_size=16, statistic="median")
        assert est.self_mask_sigma is None
        sky = est(image)
        assert sky.shape == (64, 64)

    def test_self_mask_deterministic(self):
        torch.manual_seed(42)
        image = torch.randn(64, 64, dtype=torch.float32) + 5.0
        est = MeshSkyEstimator(block_size=16, statistic="huber", self_mask_sigma=3.0)
        sky_a = est(image)
        sky_b = est(image)
        assert torch.equal(sky_a, sky_b)


class TestPolynomialOnMesh:
    def test_linear_surface_synthetic(self):
        H, W = 64, 64
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing="ij"
        )
        image = 10.0 + 5.0 * xx + 3.0 * yy
        est = MeshSkyEstimator(
            block_size=16,
            statistic="median",
            mesh_filter=None,
            mesh_polynomial_order=1,
        )
        sky = est(image)
        residual = (image - sky).abs().mean()
        assert float(residual) < 0.5

    def test_quadratic_surface(self):
        H, W = 64, 64
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij"
        )
        image = 5.0 + 2.0 * xx + 3.0 * yy + 0.5 * xx * yy + 1.0 * xx**2
        est = MeshSkyEstimator(
            block_size=16,
            statistic="median",
            mesh_filter=None,
            mesh_polynomial_order=2,
        )
        sky = est(image)
        residual = (image - sky).abs().mean()
        assert float(residual) < 0.5

    def test_fallback_when_too_few_mesh_points(self):
        image = torch.randn(24, 24, dtype=torch.float32)
        est = MeshSkyEstimator(
            block_size=16,
            statistic="median",
            mesh_polynomial_order=4,  # 15 terms, but only 4 mesh cells
        )
        sky = est(image)
        assert sky.shape == (24, 24)

    def test_polynomial_with_filter(self):
        image = torch.randn(64, 64, dtype=torch.float32) + 5.0
        est = MeshSkyEstimator(
            block_size=16,
            statistic="median",
            mesh_filter="median",
            mesh_filter_size=3,
            mesh_polynomial_order=2,
        )
        sky = est(image)
        assert sky.shape == (64, 64)

    def test_polynomial_differentiable(self):
        image = torch.randn(32, 32, dtype=torch.float64, requires_grad=True)
        est = MeshSkyEstimator(
            block_size=8,
            statistic="huber",
            mesh_filter=None,
            mesh_polynomial_order=2,
        )
        sky = est(image)
        loss = sky.sum()
        loss.backward()
        assert image.grad is not None
        assert image.grad.isfinite().any()

    def test_polynomial_order_one(self):
        image = torch.randn(64, 64, dtype=torch.float32) + 5.0
        est = MeshSkyEstimator(
            block_size=16,
            statistic="median",
            mesh_filter=None,
            mesh_polynomial_order=1,
        )
        sky = est(image)
        assert sky.shape == (64, 64)

    def test_invalid_order_raises(self):
        with pytest.raises(ValueError, match="mesh_polynomial_order"):
            MeshSkyEstimator(mesh_polynomial_order=0)


class TestValidation:
    def test_unknown_statistic(self):
        with pytest.raises(ValueError, match="unknown statistic"):
            MeshSkyEstimator(statistic="invalid")

    def test_unknown_mesh_filter(self):
        with pytest.raises(ValueError, match="unknown mesh_filter"):
            MeshSkyEstimator(mesh_filter="invalid")

    def test_unknown_interpolate(self):
        with pytest.raises(ValueError, match="unknown interpolate"):
            MeshSkyEstimator(interpolate="invalid")
