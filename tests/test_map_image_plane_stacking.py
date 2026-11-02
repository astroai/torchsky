from __future__ import annotations
from pathlib import Path
import sys
import pytest
import torch
from conftest import assert_close_on_device
import torchfits
import torchsky.map.image_plane_common as image_plane_common_mod
import torchsky.map.operators.execution as execution_mod
from torchsky.map import stack_product
from torchsky.map.image_plane_coadd import (
    coadd_image_plane,
    coadd_image_plane_cube,
)
from torchsky.map.observation import ImageObservation
from torchsky.map.product import plan_map_product
from torchsky.map.sky import (
    CrossExposureSkyCorrector,
    MeshSkyEstimator,
    PolynomialSkyModel,
)
from torchsky.spectral import SpectralAxis

pytest.importorskip("torchsky._cpp")
from map_image_plane_support import (
    _simple_header,
)


def test_cli_map_coadd_writes_output(tmp_path, monkeypatch) -> None:
    from torchsky.cli import main

    path_a = tmp_path / "cli_a.fits"
    path_b = tmp_path / "cli_b.fits"
    header = _simple_header()
    data_a = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    data_b = data_a + 2.0
    torchfits.write(str(path_a), data_a, header=header, overwrite=True)
    torchfits.write(str(path_b), data_b, header=header, overwrite=True)
    out = tmp_path / "coadd_cli.fits"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "torchsky",
            "map",
            str(out),
            str(path_a),
            str(path_b),
            "--no-auto-tile",
            "--no-union-footprint",
        ],
    )
    assert main() == 0
    got, _ = torchfits.read(str(out), return_header=True)
    assert_close_on_device(
        got[1:-1, 1:-1], (data_a + data_b)[1:-1, 1:-1] / 2.0, atol=1e-4, rtol=1e-4
    )


def test_coadd_image_plane_mean_stacking_averages_observations_equally(
    tmp_path,
) -> None:
    path_a = tmp_path / "mean_a.fits"
    path_b = tmp_path / "mean_b.fits"
    header = _simple_header()
    data_a = torch.full((8, 8), 1.0, dtype=torch.float32)
    data_b = torch.full((8, 8), 9.0, dtype=torch.float32)
    torchfits.write(str(path_a), data_a, header=header, overwrite=True)
    torchfits.write(str(path_b), data_b, header=header, overwrite=True)

    weighted = coadd_image_plane(
        [
            ImageObservation(str(path_a), weight=1.0),
            ImageObservation(str(path_b), weight=9.0),
        ],
        stacking_method="weighted_mean",
    )
    mean = coadd_image_plane(
        [
            ImageObservation(str(path_a), weight=1.0),
            ImageObservation(str(path_b), weight=9.0),
        ],
        stacking_method="mean",
    )

    assert_close_on_device(
        weighted.data,
        torch.full((8, 8), 8.2, dtype=torch.float32),
        atol=1e-5,
        rtol=1e-5,
    )
    assert weighted.inverse_variance is None
    assert_close_on_device(
        mean.data,
        torch.full((8, 8), 5.0, dtype=torch.float32),
        atol=1e-5,
        rtol=1e-5,
    )
    assert_close_on_device(
        mean.weight,
        torch.full((8, 8), 2.0, dtype=torch.float32),
        atol=1e-5,
        rtol=1e-5,
    )


def test_coadd_image_plane_median_stacking_uses_observation_median(tmp_path) -> None:
    path_a = tmp_path / "median_a.fits"
    path_b = tmp_path / "median_b.fits"
    path_c = tmp_path / "median_c.fits"
    header = _simple_header()
    data_a = torch.full((8, 8), 1.0, dtype=torch.float32)
    data_b = torch.full((8, 8), 9.0, dtype=torch.float32)
    data_c = torch.full((8, 8), 100.0, dtype=torch.float32)
    torchfits.write(str(path_a), data_a, header=header, overwrite=True)
    torchfits.write(str(path_b), data_b, header=header, overwrite=True)
    torchfits.write(str(path_c), data_c, header=header, overwrite=True)

    weighted = coadd_image_plane(
        [
            ImageObservation(str(path_a), weight=1.0),
            ImageObservation(str(path_b), weight=1.0),
            ImageObservation(str(path_c), weight=20.0),
        ],
        stacking_method="weighted_mean",
    )
    median = coadd_image_plane(
        [
            ImageObservation(str(path_a), weight=1.0),
            ImageObservation(str(path_b), weight=1.0),
            ImageObservation(str(path_c), weight=20.0),
        ],
        stacking_method="median",
    )

    assert float(weighted.data[0, 0].item()) > 80.0
    assert_close_on_device(
        median.data,
        torch.full((8, 8), 9.0, dtype=torch.float32),
        atol=1e-5,
        rtol=1e-5,
    )
    assert_close_on_device(
        median.weight,
        torch.full((8, 8), 3.0, dtype=torch.float32),
        atol=1e-5,
        rtol=1e-5,
    )


def test_spilled_exact_median_matches_memory_and_cleans_scratch(tmp_path) -> None:
    values = [
        torch.tensor([[1.0, 20.0, 7.0], [4.0, 8.0, 3.0]]),
        torch.tensor([[5.0, 10.0, 9.0], [2.0, 6.0, 1.0]]),
        torch.tensor([[3.0, 30.0, 8.0], [6.0, 7.0, 5.0]]),
    ]
    support = [
        torch.tensor([[True, True, False], [True, True, True]]),
        torch.tensor([[True, False, True], [True, True, True]]),
        torch.tensor([[True, True, True], [False, True, True]]),
    ]
    expected = execution_mod.masked_median_stack(values, support)
    scratch = tmp_path / "scratch"

    actual = execution_mod.spilled_exact_median_stack(
        zip(values, support, strict=True),
        n_observations=3,
        shape=(2, 3),
        scratch_dir=scratch,
        working_memory_bytes=32,
    )

    assert_close_on_device(actual[0], expected[0])
    assert_close_on_device(actual[1], expected[1])
    assert scratch.is_dir()
    assert list(scratch.iterdir()) == []


def test_spilled_exact_median_rejects_insufficient_scratch_before_projection(
    tmp_path, monkeypatch
) -> None:
    consumed = False

    def contributions():
        nonlocal consumed
        consumed = True
        yield torch.ones((4, 4)), torch.ones((4, 4), dtype=torch.bool)

    class _Usage:
        free = 1

    monkeypatch.setattr(execution_mod.shutil, "disk_usage", lambda _path: _Usage())

    with pytest.raises(OSError, match="scratch.*requires"):
        execution_mod.spilled_exact_median_stack(
            contributions(),
            n_observations=2,
            shape=(4, 4),
            scratch_dir=tmp_path / "too-small",
        )

    assert consumed is False


def test_spilled_exact_median_cleans_scratch_after_projection_failure(
    tmp_path,
) -> None:
    scratch = tmp_path / "failed-scratch"

    def contributions():
        yield torch.ones((2, 2)), torch.ones((2, 2), dtype=torch.bool)
        raise RuntimeError("projection failed")

    with pytest.raises(RuntimeError, match="projection failed"):
        execution_mod.spilled_exact_median_stack(
            contributions(),
            n_observations=2,
            shape=(2, 2),
            scratch_dir=scratch,
        )

    assert list(scratch.iterdir()) == []


def test_coadd_image_plane_tiled_median_uses_and_cleans_scratch(tmp_path) -> None:
    header = _simple_header()
    observations = []
    for index, level in enumerate((1.0, 5.0, 30.0)):
        path = tmp_path / f"median_tiled_{index}.fits"
        torchfits.write(
            str(path),
            torch.full((8, 8), level, dtype=torch.float32),
            header=header,
            overwrite=True,
        )
        observations.append(ImageObservation(str(path)))
    scratch = tmp_path / "median-scratch"

    full = coadd_image_plane(
        observations,
        stacking_method="median",
        scratch_dir=scratch,
    )
    tiled = coadd_image_plane(
        observations,
        stacking_method="median",
        spatial_tile_hw=(3, 5),
        scratch_dir=scratch,
    )

    assert_close_on_device(tiled.data, full.data)
    assert_close_on_device(tiled.weight, full.weight)
    assert tiled.context is not None and full.context is not None
    assert_close_on_device(tiled.context, full.context)
    assert tiled.overlap_count is not None and full.overlap_count is not None
    assert_close_on_device(tiled.overlap_count, full.overlap_count)
    assert list(scratch.iterdir()) == []


def test_coadd_image_plane_sigma_clipped_rejects_outlier_and_tracks_context(
    tmp_path,
) -> None:
    header = _simple_header()
    observations = []
    for index, level in enumerate((1.0, 1.1, 0.9, 1.05, 100.0)):
        path = tmp_path / f"clip_{index}.fits"
        torchfits.write(
            str(path),
            torch.full((8, 8), level, dtype=torch.float32),
            header=header,
            overwrite=True,
        )
        observations.append(
            ImageObservation(
                str(path),
                weight=4.0,
                weight_is_inverse_variance=True,
            )
        )
    scratch = tmp_path / "clip-scratch"

    clipped = coadd_image_plane(
        observations,
        stacking_method="sigma_clipped",
        clip_sigma=4.0,
        clip_iterations=3,
        scratch_dir=scratch,
    )

    assert_close_on_device(
        clipped.data,
        torch.full((8, 8), 1.0125),
        rtol=0.0,
        atol=1.0e-6,
    )
    assert_close_on_device(clipped.weight, torch.full((8, 8), 16.0))
    assert clipped.inverse_variance is not None
    assert_close_on_device(
        clipped.inverse_variance,
        torch.full((8, 8), 16.0),
    )
    assert clipped.overlap_count is not None
    assert_close_on_device(
        clipped.overlap_count,
        torch.full((8, 8), 4, dtype=torch.int32),
    )
    assert clipped.context is not None
    assert_close_on_device(
        clipped.context[0],
        torch.full((8, 8), 0b01111, dtype=torch.int64),
    )
    assert list(scratch.iterdir()) == []


def test_coadd_image_plane_tiled_sigma_clipped_matches_full(tmp_path) -> None:
    header = _simple_header()
    observations = []
    for index, level in enumerate((2.0, 2.1, 1.9, 2.05, 80.0)):
        path = tmp_path / f"clip_tiled_{index}.fits"
        torchfits.write(
            str(path),
            torch.full((8, 8), level, dtype=torch.float32),
            header=header,
            overwrite=True,
        )
        observations.append(ImageObservation(str(path)))
    scratch = tmp_path / "clip-tiled-scratch"
    kwargs = {
        "stacking_method": "sigma_clipped",
        "clip_sigma": 4.0,
        "clip_iterations": 3,
        "scratch_dir": scratch,
    }

    full = coadd_image_plane(observations, **kwargs)
    tiled = coadd_image_plane(observations, spatial_tile_hw=(3, 5), **kwargs)

    assert_close_on_device(tiled.data, full.data)
    assert_close_on_device(tiled.weight, full.weight)
    assert full.inverse_variance is None
    assert tiled.inverse_variance is None
    assert tiled.context is not None and full.context is not None
    assert_close_on_device(tiled.context, full.context)
    assert list(scratch.iterdir()) == []


def test_spilled_sigma_clipped_rejects_scratch_shortage_before_projection(
    tmp_path, monkeypatch
) -> None:
    consumed = False

    def contributions():
        nonlocal consumed
        consumed = True
        yield (
            torch.ones((2, 2)),
            torch.ones((2, 2)),
            torch.ones((2, 2)),
            None,
        )

    class _Usage:
        free = 1

    monkeypatch.setattr(execution_mod.shutil, "disk_usage", lambda _path: _Usage())

    with pytest.raises(OSError, match="sigma-clipped scratch requires"):
        execution_mod.spilled_sigma_clipped_stack(
            contributions(),
            n_observations=2,
            shape=(2, 2),
            scratch_dir=tmp_path / "too-small-clipped",
        )

    assert consumed is False


def test_spilled_sigma_clipping_uses_explicit_propagated_noise(tmp_path) -> None:
    values = [0.0, 0.0, 10.0]

    without_noise = execution_mod.spilled_sigma_clipped_stack(
        (
            (
                torch.tensor([[value]]),
                torch.ones((1, 1)),
                None,
                None,
            )
            for value in values
        ),
        n_observations=3,
        shape=(1, 1),
        scratch_dir=tmp_path / "relative-weights",
    )
    with_noise = execution_mod.spilled_sigma_clipped_stack(
        (
            (
                torch.tensor([[value]]),
                torch.ones((1, 1)),
                torch.tensor([[variance]]),
                None,
            )
            for value, variance in zip(values, (1.0, 1.0, 10_000.0))
        ),
        n_observations=3,
        shape=(1, 1),
        scratch_dir=tmp_path / "explicit-noise",
    )

    assert_close_on_device(without_noise.data, torch.zeros((1, 1)))
    assert_close_on_device(
        without_noise.overlap_count,
        torch.full((1, 1), 2, dtype=torch.int32),
    )
    assert_close_on_device(
        with_noise.data,
        torch.full((1, 1), 10.0 / 3.0),
    )
    assert_close_on_device(
        with_noise.overlap_count,
        torch.full((1, 1), 3, dtype=torch.int32),
    )
    assert with_noise.variance_weight is not None
    assert_close_on_device(
        with_noise.variance_weight,
        torch.full((1, 1), 10_002.0),
    )


def test_coadd_image_plane_sigma_clipped_rejects_overlap_background(tmp_path) -> None:
    path = tmp_path / "clip_overlap.fits"
    torchfits.write(
        str(path),
        torch.ones((8, 8)),
        header=_simple_header(),
        overwrite=True,
    )

    with pytest.raises(NotImplementedError, match="per-exposure"):
        coadd_image_plane(
            [ImageObservation(str(path))],
            stacking_method="sigma_clipped",
            cross_exposure_sky_corrector=CrossExposureSkyCorrector(),
        )


def test_tiled_overlap_background_uses_one_read_per_exposure_and_cleans_scratch(
    tmp_path,
    monkeypatch,
) -> None:
    from torchsky.map.sky import PolynomialSkyModel

    header = _simple_header()
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, 8),
        torch.linspace(-1.0, 1.0, 8),
        indexing="ij",
    )
    scene = torch.exp(-3.0 * (xx.square() + yy.square()))
    backgrounds = (
        torch.zeros_like(scene),
        4.0 + 0.5 * xx - 0.25 * yy,
        -3.0 + 0.2 * xx + 0.4 * yy,
    )
    observations = []
    for index, background in enumerate(backgrounds):
        path = tmp_path / f"overlap_stream_{index}.fits"
        torchfits.write(
            str(path),
            scene + background,
            header=header,
            overwrite=True,
        )
        observations.append(ImageObservation(str(path)))
    corrector = CrossExposureSkyCorrector(
        estimator=PolynomialSkyModel(order=1, statistic="huber")
    )
    expected = coadd_image_plane(
        observations,
        cross_exposure_sky_corrector=corrector,
    )

    read_calls: list[str] = []
    original_read = torchfits.read

    def counted_read(path, **kwargs):
        read_calls.append(str(path))
        return original_read(path, **kwargs)

    monkeypatch.setattr(image_plane_common_mod, "read", counted_read)
    scratch = tmp_path / "overlap-scratch"
    tiled = coadd_image_plane(
        observations,
        spatial_tile_hw=(3, 5),
        scratch_dir=scratch,
        cross_exposure_sky_corrector=corrector,
    )

    assert read_calls == [str(observation.path) for observation in observations]
    assert_close_on_device(tiled.data, expected.data, atol=2.0e-4, rtol=2.0e-4)
    assert_close_on_device(tiled.weight, expected.weight, atol=2.0e-4, rtol=2.0e-4)
    assert tiled.context is not None and expected.context is not None
    assert_close_on_device(tiled.context, expected.context)
    assert tiled.overlap_count is not None and expected.overlap_count is not None
    assert_close_on_device(tiled.overlap_count, expected.overlap_count)
    assert scratch.is_dir()
    assert list(scratch.iterdir()) == []


@pytest.mark.parametrize(
    "estimator",
    [
        pytest.param(
            PolynomialSkyModel(order=1, statistic="huber"),
            id="polynomial",
        ),
        pytest.param(
            MeshSkyEstimator(
                block_size=4,
                statistic="median",
                mesh_filter="median",
                interpolate="bicubic",
            ),
            id="mesh",
        ),
    ],
)
def test_tiled_overlap_background_renders_compact_models(
    tmp_path,
    monkeypatch,
    estimator,
) -> None:
    header = _simple_header()
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, 8),
        torch.linspace(-1.0, 1.0, 8),
        indexing="ij",
    )
    scene = torch.exp(-3.0 * (xx.square() + yy.square()))
    observations = []
    for index, background in enumerate(
        (torch.zeros_like(scene), 4.0 + 0.5 * xx - 0.25 * yy)
    ):
        path = tmp_path / f"overlap_compact_{index}.fits"
        torchfits.write(str(path), scene + background, header=header, overwrite=True)
        observations.append(ImageObservation(str(path)))
    corrector = CrossExposureSkyCorrector(estimator=estimator)
    expected = coadd_image_plane(
        observations,
        cross_exposure_sky_corrector=corrector,
    )

    def reject_full_plane(*_args, **_kwargs):
        raise AssertionError("full difference-model rendering was used")

    monkeypatch.setattr(corrector, "fit_difference_model", reject_full_plane)
    result = coadd_image_plane(
        observations,
        spatial_tile_hw=(3, 5),
        scratch_dir=tmp_path / "overlap-compact-scratch",
        cross_exposure_sky_corrector=corrector,
    )

    assert_close_on_device(result.data, expected.data, atol=2.0e-4, rtol=2.0e-4)


def test_tiled_overlap_background_rejects_noncompact_estimators(tmp_path) -> None:
    header = _simple_header()
    observations = []
    for index in range(2):
        path = tmp_path / f"overlap_noncompact_{index}.fits"
        torchfits.write(
            str(path),
            torch.ones((8, 8), dtype=torch.float32) + index,
            header=header,
            overwrite=True,
        )
        observations.append(ImageObservation(str(path)))

    with pytest.raises(ValueError, match="compact tile rendering"):
        coadd_image_plane(
            observations,
            spatial_tile_hw=(3, 5),
            scratch_dir=tmp_path / "overlap-noncompact-scratch",
            cross_exposure_sky_corrector=CrossExposureSkyCorrector(
                estimator=lambda image, _mask: torch.zeros_like(image)
            ),
        )


def test_tiled_overlap_background_uses_one_global_chain_graph(tmp_path) -> None:
    from torchsky.map.sky import PolynomialSkyModel

    header = _simple_header()
    base = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    observations = []
    for index, offset in enumerate((2.0, 7.0, 17.0)):
        image_path = tmp_path / f"overlap_chain_{index}.fits"
        mask_path = tmp_path / f"overlap_chain_mask_{index}.fits"
        mask = torch.ones((8, 8), dtype=torch.float32)
        if index == 0:
            mask[:, :4] = 0.0
        elif index == 2:
            mask[:, 4:] = 0.0
        torchfits.write(
            str(image_path),
            base + offset,
            header=header,
            overwrite=True,
        )
        torchfits.write(str(mask_path), mask, header=header, overwrite=True)
        observations.append(ImageObservation(str(image_path), mask_path=str(mask_path)))
    corrector = CrossExposureSkyCorrector(
        estimator=PolynomialSkyModel(order=0, statistic="huber")
    )

    expected = coadd_image_plane(
        observations,
        cross_exposure_sky_corrector=corrector,
    )
    tiled = coadd_image_plane(
        observations,
        spatial_tile_hw=(8, 4),
        scratch_dir=tmp_path / "chain-scratch",
        cross_exposure_sky_corrector=corrector,
    )

    assert_close_on_device(tiled.data, expected.data, atol=2.0e-4, rtol=0.0)
    assert_close_on_device(tiled.weight, expected.weight)


def test_spilled_overlap_background_preserves_propagated_inverse_variance(
    tmp_path,
) -> None:
    from torchsky.map.sky import PolynomialSkyModel

    header = _simple_header()
    observations = []
    for index, offset in enumerate((0.0, 6.0)):
        path = tmp_path / f"overlap_variance_{index}.fits"
        torchfits.write(
            str(path),
            torch.arange(64, dtype=torch.float32).reshape(8, 8) + offset,
            header=header,
            overwrite=True,
        )
        observations.append(
            ImageObservation(
                str(path),
                weight=4.0,
                weight_is_inverse_variance=True,
            )
        )

    result = coadd_image_plane(
        observations,
        spatial_tile_hw=(3, 5),
        scratch_dir=tmp_path / "overlap-variance-scratch",
        cross_exposure_sky_corrector=CrossExposureSkyCorrector(
            estimator=PolynomialSkyModel(order=0, statistic="huber")
        ),
    )

    assert result.inverse_variance is not None
    assert_close_on_device(
        result.inverse_variance,
        torch.full((8, 8), 8.0),
    )


def test_tiled_overlap_background_rejects_scratch_shortage_before_projection(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "overlap-capacity.fits"
    torchfits.write(
        str(path),
        torch.ones((8, 8)),
        header=_simple_header(),
        overwrite=True,
    )
    prepared = False
    original_prepare = execution_mod.prepare_image_observation_data

    def counted_prepare(*args, **kwargs):
        nonlocal prepared
        prepared = True
        return original_prepare(*args, **kwargs)

    class _Usage:
        free = 1

    monkeypatch.setattr(
        execution_mod,
        "prepare_image_observation_data",
        counted_prepare,
    )
    monkeypatch.setattr(execution_mod.shutil, "disk_usage", lambda _path: _Usage())

    with pytest.raises(OSError, match="overlap background scratch requires"):
        coadd_image_plane(
            [ImageObservation(str(path)), ImageObservation(str(path))],
            spatial_tile_hw=(4, 4),
            scratch_dir=tmp_path / "too-small-overlap",
            cross_exposure_sky_corrector=CrossExposureSkyCorrector(),
        )

    assert prepared is False


def test_stack_map_product_passes_mean_stacking_to_single_band_coadd(tmp_path) -> None:
    path_a = tmp_path / "product_mean_a.fits"
    path_b = tmp_path / "product_mean_b.fits"
    header = _simple_header()
    data_a = torch.full((8, 8), 2.0, dtype=torch.float32)
    data_b = torch.full((8, 8), 10.0, dtype=torch.float32)
    torchfits.write(str(path_a), data_a, header=header, overwrite=True)
    torchfits.write(str(path_b), data_b, header=header, overwrite=True)

    observations = [
        ImageObservation(str(path_a), weight=1.0),
        ImageObservation(str(path_b), weight=5.0),
    ]
    product = plan_map_product(observations)

    result = stack_product(
        observations,
        product,
        stacking_method="mean",
    )

    assert_close_on_device(
        result.data.squeeze(0),
        torch.full((8, 8), 6.0, dtype=torch.float32),
        atol=1e-5,
        rtol=1e-5,
    )
    assert_close_on_device(
        result.weight.squeeze(0),
        torch.full((8, 8), 2.0, dtype=torch.float32),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize("stacking_method", ["median", "sigma_clipped"])
def test_coadd_image_plane_cube_rejects_order_statistic_stacking(
    tmp_path, stacking_method
) -> None:
    path = tmp_path / "cube_median.fits"
    header = _simple_header()
    data = torch.ones((8, 8), dtype=torch.float32)
    torchfits.write(str(path), data, header=header, overwrite=True)

    observations = [ImageObservation(str(path), wavelength=5000.0)]
    spectral_axis = SpectralAxis(
        values=torch.tensor([5000.0], dtype=torch.float64),
        unit="Angstrom",
        type="WAVE",
    )

    with pytest.raises(ValueError, match="single-band"):
        coadd_image_plane_cube(
            observations,
            spectral_axis=spectral_axis,
            stacking_method=stacking_method,
        )


def test_coadd_image_plane_respects_target_header_shape(tmp_path) -> None:
    path = tmp_path / "a.fits"
    header = _simple_header()
    data = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    torchfits.write(str(path), data, header=header, overwrite=True)

    target_header = dict(header)
    target_header["NAXIS1"] = 6
    target_header["NAXIS2"] = 5

    result = coadd_image_plane(
        [ImageObservation(str(path))],
        target_header=target_header,
    )

    assert tuple(result.data.shape) == (5, 6)
    assert tuple(result.weight.shape) == (5, 6)


@pytest.mark.parametrize("kernel", ("bilinear", "drizzle_square"))
def test_coadd_image_plane_ignores_non_finite_pixels(
    tmp_path,
    kernel: str,
) -> None:
    path_a = tmp_path / "a.fits"
    path_b = tmp_path / "b.fits"
    header = _simple_header()
    data_a = torch.ones((8, 8), dtype=torch.float32)
    data_b = 3.0 * torch.ones((8, 8), dtype=torch.float32)
    data_a[2, 3] = torch.nan
    torchfits.write(str(path_a), data_a, header=header, overwrite=True)
    torchfits.write(str(path_b), data_b, header=header, overwrite=True)

    result = coadd_image_plane(
        [ImageObservation(str(path_a)), ImageObservation(str(path_b))],
        resampling_kernel=kernel,
    )

    assert torch.isfinite(result.data).all()
    assert result.data[2, 3].item() == 3.0
    assert result.weight[2, 3].item() == 1.0


def test_coadd_image_plane_applies_known_photometric_scale_and_background(
    tmp_path,
) -> None:
    path_ref = tmp_path / "ref.fits"
    path_raw = tmp_path / "raw.fits"
    header = _simple_header()
    truth = torch.full((8, 8), 2.0, dtype=torch.float32)
    raw = torch.full((8, 8), 6.0, dtype=torch.float32)
    torchfits.write(str(path_ref), truth, header=header, overwrite=True)
    torchfits.write(str(path_raw), raw, header=header, overwrite=True)

    result = coadd_image_plane(
        [
            ImageObservation(str(path_ref)),
            ImageObservation(
                str(path_raw),
                photometric_scale=2.0,
                background_offset=5.0,
            ),
        ]
    )

    assert_close_on_device(result.data, truth, atol=1e-5, rtol=1e-5)
    assert_close_on_device(
        result.weight,
        torch.full((8, 8), 1.25, dtype=torch.float32),
        atol=1e-5,
        rtol=1e-5,
    )


def test_coadd_image_plane_rescales_weight_map_with_photometric_scale(tmp_path) -> None:
    path = tmp_path / "scaled.fits"
    weight_path = tmp_path / "scaled_wht.fits"
    header = _simple_header()
    raw = torch.full((8, 8), 7.0, dtype=torch.float32)
    weight = torch.full((8, 8), 8.0, dtype=torch.float32)
    torchfits.write(str(path), raw, header=header, overwrite=True)
    torchfits.write(str(weight_path), weight, header=header, overwrite=True)

    result = coadd_image_plane(
        [
            ImageObservation(
                str(path),
                weight_path=str(weight_path),
                photometric_scale=2.0,
                background_offset=5.0,
            )
        ]
    )

    assert_close_on_device(
        result.data,
        torch.full((8, 8), 4.0, dtype=torch.float32),
        atol=1e-5,
        rtol=1e-5,
    )
    assert_close_on_device(
        result.weight,
        torch.full((8, 8), 2.0, dtype=torch.float32),
        atol=1e-5,
        rtol=1e-5,
    )


def test_coadd_image_plane_rejects_non_2d_images(tmp_path) -> None:
    path = tmp_path / "cube.fits"
    header = _simple_header()
    header["NAXIS"] = 3
    header["NAXIS3"] = 2
    data = torch.ones((2, 8, 8), dtype=torch.float32)
    torchfits.write(str(path), data, header=header, overwrite=True)

    with pytest.raises(ValueError, match="2D images only"):
        coadd_image_plane([ImageObservation(str(path))])


def test_coadd_image_plane_rejects_unknown_resampling_kernel(tmp_path) -> None:
    path = tmp_path / "a.fits"
    header = _simple_header()
    data = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    torchfits.write(str(path), data, header=header, overwrite=True)

    with pytest.raises(ValueError, match="resampling_kernel"):
        coadd_image_plane(
            [ImageObservation(str(path))],
            target_header={**header, "CRVAL1": float(header["CRVAL1"]) + 0.01},
            resampling_kernel="spline5",
        )


def test_coadd_image_plane_rejects_unknown_value_semantics(tmp_path) -> None:
    path = tmp_path / "a.fits"
    header = _simple_header()
    data = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    torchfits.write(str(path), data, header=header, overwrite=True)

    with pytest.raises(ValueError, match="value_semantics"):
        coadd_image_plane(
            [ImageObservation(str(path))],
            value_semantics="counts",
        )


def test_coadd_image_plane_rejects_unknown_stacking_method(tmp_path) -> None:
    path = tmp_path / "stacking_unknown.fits"
    header = _simple_header()
    data = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    torchfits.write(str(path), data, header=header, overwrite=True)

    with pytest.raises(ValueError, match="stacking_method"):
        coadd_image_plane(
            [ImageObservation(str(path))],
            stacking_method="trimmed_mean",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("photometric_scale", 0.0),
        ("photometric_scale", -1.0),
        ("background_offset", float("nan")),
    ],
)
def test_coadd_image_plane_rejects_invalid_calibration_metadata(
    tmp_path: Path,
    field: str,
    value: float,
) -> None:
    path = tmp_path / "invalid_calibration.fits"
    header = _simple_header()
    data = torch.ones((8, 8), dtype=torch.float32)
    torchfits.write(str(path), data, header=header, overwrite=True)

    kwargs = {field: value}
    with pytest.raises(ValueError, match=field):
        coadd_image_plane([ImageObservation(str(path), **kwargs)])
