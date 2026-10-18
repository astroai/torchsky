from __future__ import annotations

import pytest
import torch

from torchsky.map.sky import CrossExposureSkyCorrector, MeshSkyEstimator, PolynomialSkyModel
from torchsky.map import ImageObservation, coadd_image_plane
import torchfits


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

    def test_first_exposure_is_explicit_gauge_anchor(self):
        base = torch.randn(64, 64)
        imgs = [base + v for v in (5.0, 10.0, 15.0)]
        corrector = CrossExposureSkyCorrector(
            estimator=MeshSkyEstimator(block_size=32, statistic="median")
        )
        off = corrector.fit_offsets(imgs)
        torch.testing.assert_close(off, torch.tensor([0.0, 5.0, 10.0]))

    def test_pairwise_overlap_graph_recovers_chain_offsets(self):
        base = torch.arange(64, dtype=torch.float32).reshape(8, 8)
        images = [base + 2.0, base + 7.0, base + 17.0]
        masks = [
            torch.arange(8).expand(8, 8) >= 4,
            torch.zeros((8, 8), dtype=torch.bool),
            torch.arange(8).expand(8, 8) < 4,
        ]
        corrector = CrossExposureSkyCorrector(
            estimator=PolynomialSkyModel(order=0, statistic="huber")
        )

        offsets = corrector.fit_offsets(images, masks)

        torch.testing.assert_close(offsets, torch.tensor([0.0, 5.0, 15.0]))

    def test_disconnected_overlap_graph_fails_closed(self):
        images = [torch.zeros((4, 4)), torch.ones((4, 4))]
        masks = [
            torch.arange(4).expand(4, 4) >= 2,
            torch.arange(4).expand(4, 4) < 2,
        ]

        with pytest.raises(ValueError, match="overlap graph is disconnected"):
            CrossExposureSkyCorrector().fit_offsets(images, masks)

    def test_pairwise_difference_plane_preserves_shared_scene(self):
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, 32),
            torch.linspace(-1.0, 1.0, 32),
            indexing="ij",
        )
        scene = torch.exp(-4.0 * (x.square() + y.square()))
        background_difference = 3.0 + 0.5 * x - 0.25 * y
        corrector = CrossExposureSkyCorrector(
            estimator=PolynomialSkyModel(order=1, statistic="huber")
        )

        corrected = corrector.correct([scene, scene + background_difference])

        torch.testing.assert_close(corrected[0], scene, atol=1.0e-5, rtol=0.0)
        torch.testing.assert_close(corrected[1], scene, atol=2.0e-4, rtol=0.0)

    def test_polynomial_compact_fit_matches_full_model_by_tiles(self):
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, 31),
            torch.linspace(-1.0, 1.0, 27),
            indexing="ij",
        )
        difference = 3.0 + 0.5 * x - 0.25 * y + 0.03 * torch.sin(5.0 * x)
        support = torch.ones_like(difference, dtype=torch.bool)
        support[4:9, 8:15] = False
        estimator = PolynomialSkyModel(order=1, statistic="huber")

        def tiles():
            for y0 in range(0, difference.shape[0], 7):
                for x0 in range(0, difference.shape[1], 6):
                    y1 = min(y0 + 7, difference.shape[0])
                    x1 = min(x0 + 6, difference.shape[1])
                    yield (
                        y0,
                        y1,
                        x0,
                        x1,
                        difference[y0:y1, x0:x1],
                        support[y0:y1, x0:x1],
                    )

        compact = estimator.fit_compact(
            tiles,
            image_shape=tuple(difference.shape),
            scratch=torch.empty_like(difference),
        )
        rendered = torch.empty_like(difference)
        for y0, y1, x0, x1, _, _ in tiles():
            rendered[y0:y1, x0:x1] = compact.render_window(y0, y1, x0, x1)

        expected = estimator(difference, ~support)
        torch.testing.assert_close(rendered, expected, atol=2.0e-5, rtol=2.0e-5)

    def test_mesh_compact_fit_matches_full_model_by_tiles(self):
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, 31),
            torch.linspace(-1.0, 1.0, 27),
            indexing="ij",
        )
        difference = 3.0 + 0.5 * x - 0.25 * y + 0.03 * torch.sin(5.0 * x)
        support = torch.ones_like(difference, dtype=torch.bool)
        support[4:9, 8:15] = False
        estimator = MeshSkyEstimator(
            block_size=8,
            statistic="median",
            mesh_filter="median",
            interpolate="bicubic",
        )

        def tiles():
            for y0 in range(0, difference.shape[0], 7):
                for x0 in range(0, difference.shape[1], 6):
                    y1 = min(y0 + 7, difference.shape[0])
                    x1 = min(x0 + 6, difference.shape[1])
                    yield (
                        y0,
                        y1,
                        x0,
                        x1,
                        difference[y0:y1, x0:x1],
                        support[y0:y1, x0:x1],
                    )

        compact = estimator.fit_compact(
            tiles,
            image_shape=tuple(difference.shape),
            scratch=torch.empty_like(difference),
        )
        rendered = torch.empty_like(difference)
        for y0, y1, x0, x1, _, _ in tiles():
            rendered[y0:y1, x0:x1] = compact.render_window(y0, y1, x0, x1)

        expected = estimator(difference, ~support)
        torch.testing.assert_close(rendered, expected, atol=5.0e-5, rtol=5.0e-5)

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
            torchfits.write(str(p), base + off, header=header, overwrite=True)
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
            torchfits.write(str(p), base + off, header=header, overwrite=True)
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
            torchfits.write(str(p), 1.0 + structure + off, header=header, overwrite=True)
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
            torchfits.write(str(p), torch.ones(32, 32, dtype=torch.float32) + off,
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
        torchfits.write(str(p), torch.ones(16, 16, dtype=torch.float32), header=header, overwrite=True)
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
