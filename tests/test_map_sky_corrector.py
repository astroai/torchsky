from __future__ import annotations

import pytest
import torch

from torchsky.map.sky import CrossExposureSkyCorrector, MeshSkyEstimator, PolynomialSkyModel


class TestCrossExposureSkyCorrector:
    def test_empty(self):
        corrector = CrossExposureSkyCorrector()
        off = corrector.fit_offsets([])
        assert off.shape == (0,)

    def test_single_image_zero_offset(self):
        img = torch.randn(64, 64)
        corrector = CrossExposureSkyCorrector(
            estimator=MeshSkyEstimator(block_size=32, statistic="median")
        )
        off = corrector.fit_offsets([img])
        assert off.shape == (1,)
        assert float(off[0]) == pytest.approx(0.0, abs=1e-6)

    def test_two_images_with_known_offset(self):
        base = torch.randn(64, 64)
        img0 = base + 10.0
        img1 = base + 25.0
        corrector = CrossExposureSkyCorrector(
            estimator=MeshSkyEstimator(block_size=32, statistic="median")
        )
        off = corrector.fit_offsets([img0, img1])
        assert off.shape == (2,)
        diff = float(off[1] - off[0])
        assert diff == pytest.approx(15.0, abs=2.0)

    def test_three_images_relative_offsets(self):
        base = torch.randn(64, 64)
        imgs = [base + v for v in (5.0, 15.0, 30.0)]
        corrector = CrossExposureSkyCorrector(
            estimator=MeshSkyEstimator(block_size=32, statistic="median")
        )
        off = corrector.fit_offsets(imgs)
        assert off.shape == (3,)
        d01 = float(off[1] - off[0])
        d12 = float(off[2] - off[1])
        assert d01 == pytest.approx(10.0, abs=3.0)
        assert d12 == pytest.approx(15.0, abs=3.0)

    def test_with_polynomial_estimator(self):
        base = torch.randn(64, 64)
        img0 = base + 10.0
        img1 = base + 20.0
        corrector = CrossExposureSkyCorrector(
            estimator=PolynomialSkyModel(order=1, statistic="huber")
        )
        off = corrector.fit_offsets([img0, img1])
        assert off.shape == (2,)
        diff = float(off[1] - off[0])
        assert diff == pytest.approx(10.0, abs=3.0)

    def test_correct_subtracts_offsets(self):
        base = torch.randn(64, 64)
        img0 = base + 10.0
        img1 = base + 30.0
        corrector = CrossExposureSkyCorrector(
            estimator=MeshSkyEstimator(block_size=32, statistic="median")
        )
        corrected = corrector.correct([img0, img1])
        assert len(corrected) == 2
        # Both corrected images should have the same mean after matching
        assert float(corrected[0].mean()) == pytest.approx(float(corrected[1].mean()), abs=2.0)

    def test_sum_offsets_zero(self):
        base = torch.randn(64, 64)
        imgs = [base + v for v in (5.0, 10.0, 15.0)]
        corrector = CrossExposureSkyCorrector(
            estimator=MeshSkyEstimator(block_size=32, statistic="median")
        )
        off = corrector.fit_offsets(imgs)
        assert float(off.sum()) == pytest.approx(0.0, abs=1e-5)

    def test_gpu(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        base = torch.randn(64, 64, device="cuda")
        imgs = [base + 5.0, base + 15.0]
        corrector = CrossExposureSkyCorrector(
            estimator=MeshSkyEstimator(block_size=32, statistic="median")
        )
        off = corrector.fit_offsets(imgs)
        assert off.device.type == "cuda"
        diff = float(off[1] - off[0])
        assert diff == pytest.approx(10.0, abs=3.0)
