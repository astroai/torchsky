from __future__ import annotations

import pytest
import torch
import torchsky

from torchsky.map.sky import CrossExposureSkyCorrector, MeshSkyEstimator, PolynomialSkyModel
from torchsky.map import ImageObservation, coadd_image_plane


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

    @staticmethod
    def _simple_header(n: int = 16) -> dict:
        return {
            "NAXIS": 2,
            "NAXIS1": n,
            "NAXIS2": n,
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
            "CRPIX1": n / 2.0,
            "CRPIX2": n / 2.0,
            "CRVAL1": 12.0,
            "CRVAL2": -3.0,
            "CDELT1": -0.01,
            "CDELT2": 0.01,
        }

    def test_with_median_coadd(self, tmp_path):
        header = self._simple_header(32)
        base = torch.ones(32, 32, dtype=torch.float32)
        offsets = [0.0, 15.0, 30.0]
        paths = []
        for i, off in enumerate(offsets):
            p = tmp_path / f"obs_{i}.fits"
            torchsky.write(str(p), base + off, header=header, overwrite=True)
            paths.append(p)
        corrector = CrossExposureSkyCorrector(
            estimator=MeshSkyEstimator(block_size=16, statistic="median")
        )
        result = coadd_image_plane(
            [ImageObservation(str(p)) for p in paths],
            stacking_method="median",
            cross_exposure_sky_corrector=corrector,
        )
        assert torch.isfinite(result.data).all()
        assert result.weight is not None

    def test_with_weighted_mean_coadd(self, tmp_path):
        header = self._simple_header(32)
        base = torch.ones(32, 32, dtype=torch.float32)
        offsets = [0.0, 15.0, 30.0]
        paths = []
        for i, off in enumerate(offsets):
            p = tmp_path / f"obs_{i}.fits"
            torchsky.write(str(p), base + off, header=header, overwrite=True)
            paths.append(p)
        corrector = CrossExposureSkyCorrector(
            estimator=MeshSkyEstimator(block_size=16, statistic="median")
        )
        result = coadd_image_plane(
            [ImageObservation(str(p)) for p in paths],
            stacking_method="weighted_mean",
            cross_exposure_sky_corrector=corrector,
        )
        assert torch.isfinite(result.data).all()
        assert result.weight is not None

    def test_corrected_no_nan(self, tmp_path):
        """Cross-exposure correction should produce finite coadds."""
        header = self._simple_header(32)
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, 32, dtype=torch.float32),
            torch.linspace(-1, 1, 32, dtype=torch.float32),
            indexing="ij",
        )
        structure = 0.5 * torch.exp(-(xx**2 + yy**2) / 0.5)
        offsets = [0.0, 20.0, 40.0]
        paths = []
        for i, off in enumerate(offsets):
            p = tmp_path / f"obs_{i}.fits"
            torchsky.write(str(p), 1.0 + structure + off, header=header, overwrite=True)
            paths.append(p)
        obs = [ImageObservation(str(p)) for p in paths]
        corrector = CrossExposureSkyCorrector(
            estimator=MeshSkyEstimator(block_size=16, statistic="median")
        )
        for method in ("median", "weighted_mean"):
            result = coadd_image_plane(
                obs, stacking_method=method,
                cross_exposure_sky_corrector=corrector,
            )
            assert torch.isfinite(result.data).all()
            assert result.weight is not None
            assert float(result.weight.max()) > 0.0

    def test_through_mapmaker(self, tmp_path):
        """Cross-exposure corrector should work through ImagePlaneMapMaker."""
        header = self._simple_header(32)
        paths = []
        for i, off in enumerate([0.0, 10.0, 20.0]):
            p = tmp_path / f"obs_{i}.fits"
            torchsky.write(str(p), torch.ones(32, 32, dtype=torch.float32) + off,
                           header=header, overwrite=True)
            paths.append(p)
        corrector = CrossExposureSkyCorrector(
            estimator=MeshSkyEstimator(block_size=16, statistic="median")
        )
        from torchsky.map import ImagePlaneMapMaker
        maker = ImagePlaneMapMaker(
            stacking_method="median",
            cross_exposure_sky_corrector=corrector,
        )
        result = maker.run([ImageObservation(str(p)) for p in paths])
        assert torch.isfinite(result.data).all()
        assert float(result.data.max()) > 0.0

    def test_cube_raises(self, tmp_path):
        header = self._simple_header()
        p = tmp_path / "obs.fits"
        torchsky.write(str(p), torch.ones(16, 16, dtype=torch.float32), header=header, overwrite=True)
        corrector = CrossExposureSkyCorrector()
        from torchsky.map.image_plane import coadd_image_plane_cube
        import pytest
        with pytest.raises(NotImplementedError):
            coadd_image_plane_cube(
                [ImageObservation(str(p), wavelength=5000.0)],
                cross_exposure_sky_corrector=corrector,
            )

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
