from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

from bench_map_smacs0723_field_fit_joint import _format_markdown, _summarize_by_model  # noqa: E402
from bench_map_smacs0723_field_fit_joint import _background_subtracted_source_stats  # noqa: E402
from bench_map_smacs0723_field_fit_joint import _joint_real_fit_kwargs  # noqa: E402
from bench_map_smacs0723_field_fit_joint import _preview_uint8  # noqa: E402
from bench_map_smacs0723_field_fit_joint import _source_metric_mask  # noqa: E402
from bench_map_smacs0723_field_fit_joint import _write_export_products  # noqa: E402


def test_joint_field_fit_markdown_lists_summary_and_band_rows() -> None:
    text = _format_markdown(
        [
            {
                "band": "F090W",
                "model": "spectral_basis_field",
                "status": "OK",
                "n_exposures": 36,
                "source_contrast_rmse": 0.04,
                "source_rmse": 0.08,
                "support_rmse": 0.1,
                "uniform_rmse": 0.2,
                "aligned_rmse": 0.05,
                "affine_slope": 1.0,
                "source_contrast_slope": 0.9,
                "support_flux_ratio": 1.01,
                "source_flux_ratio": 0.95,
                "support_frac": 0.9,
                "source_frac": 0.2,
                "fit_time_ms": 12.3,
            }
        ]
    )
    assert "Model Summary" in text
    assert "F090W" in text
    assert "spectral_basis_field" in text


def test_joint_field_fit_summary_averages_rows_by_model() -> None:
    summary = _summarize_by_model(
        [
            {
                "band": "F090W",
                "model": "m",
                "status": "OK",
                "source_contrast_rmse": 0.04,
                "source_rmse": 0.08,
                "support_rmse": 0.1,
                "uniform_rmse": 0.3,
                "aligned_rmse": 0.05,
                "affine_slope": 1.0,
                "source_contrast_slope": 0.9,
                "support_flux_ratio": 1.0,
                "source_flux_ratio": 0.95,
                "fit_time_ms": 5.0,
            },
            {
                "band": "F277W",
                "model": "m",
                "status": "OK",
                "source_contrast_rmse": 0.10,
                "source_rmse": 0.18,
                "support_rmse": 0.3,
                "uniform_rmse": 0.5,
                "aligned_rmse": 0.15,
                "affine_slope": 0.8,
                "source_contrast_slope": 0.7,
                "support_flux_ratio": 0.8,
                "source_flux_ratio": 0.75,
                "fit_time_ms": 7.0,
            },
        ]
    )
    assert len(summary) == 1
    assert summary[0]["model"] == "m"
    assert abs(summary[0]["mean_support_rmse"] - 0.2) < 1.0e-6
    assert abs(summary[0]["mean_source_rmse"] - 0.13) < 1.0e-6
    assert abs(summary[0]["mean_source_contrast_rmse"] - 0.07) < 1.0e-6
    assert summary[0]["bands"] == 2
    assert abs(summary[0]["mean_aligned_rmse"] - 0.1) < 1.0e-6
    assert abs(summary[0]["mean_affine_slope"] - 0.9) < 1.0e-6


def test_joint_real_fit_kwargs_use_pixel_sampling() -> None:
    kwargs = _joint_real_fit_kwargs(
        "spectral_basis_field",
        crop_size=32,
        steps=4,
        num_observations=20,
    )
    assert kwargs["observations_per_step"] == 12
    assert kwargs["training_patch_size"] is None
    assert kwargs["pixel_samples_per_observation"] >= 256


def test_source_metric_mask_prefers_bright_reference_structure() -> None:
    pred = np.ones((4, 4), dtype=np.float32)
    reference = np.zeros((4, 4), dtype=np.float32)
    reference[1, 1] = 10.0
    reference[2, 2] = 8.0
    support = np.ones((4, 4), dtype=bool)

    mask = _source_metric_mask(pred, reference, support)

    assert mask[1, 1]
    assert mask[2, 2]
    assert mask.sum() <= support.sum()


def test_background_subtracted_source_stats_remove_pedestal() -> None:
    reference = np.zeros((6, 6), dtype=np.float32)
    reference[2:4, 2:4] = np.array([[3.0, 5.0], [4.0, 6.0]], dtype=np.float32)
    pred = reference + 2.0
    support = np.ones((6, 6), dtype=bool)

    stats = _background_subtracted_source_stats(pred, reference, support)

    assert stats["source_rmse"] > 1.0
    assert stats["source_contrast_rmse"] < 1.0e-6


def test_preview_uint8_returns_byte_image() -> None:
    image = np.array([[0.0, 1.0], [2.0, np.nan]], dtype=np.float32)
    preview = _preview_uint8(image)
    assert preview.dtype == np.uint8
    assert preview.shape == image.shape


def test_write_export_products_writes_fits_and_png() -> None:
    reference = np.arange(16, dtype=np.float32).reshape(4, 4)
    prediction = reference + 1.0
    support_mask = np.ones((4, 4), dtype=bool)
    source_mask = np.zeros((4, 4), dtype=bool)
    source_mask[1:3, 1:3] = True
    header = {"NAXIS": 2, "NAXIS1": 4, "NAXIS2": 4}

    with tempfile.TemporaryDirectory() as tmp:
        _write_export_products(
            Path(tmp),
            band="F090W",
            model="spectral_basis_field",
            header=header,
            reference=reference,
            prediction=prediction,
            support_mask=support_mask,
            source_mask=source_mask,
        )
        out = Path(tmp) / "f090w" / "spectral_basis_field"
        assert (out / "reference.fits").exists()
        assert (out / "prediction.fits").exists()
        assert (out / "residual.fits").exists()
        assert (out / "support_mask.fits").exists()
        assert (out / "source_mask.fits").exists()
        assert (out / "reference.png").exists()
        assert (out / "prediction.png").exists()
        assert (out / "residual.png").exists()
