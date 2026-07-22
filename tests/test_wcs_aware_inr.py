from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from torchsky.map.wcs_aware_calibration import (
    ExposureCalibrationInput,
    build_bandpass_calibration,
    build_psf_calibration,
)
from torchsky.map.wcs_aware_inr import (
    WCSAwareMultiWavelengthINR,
    build_union_canvas,
    weighted_robust_loss,
)
from torchsky.map.wcs_aware_training import (
    ExposurePool,
    effective_batch_size,
    sample_balanced_batch,
)

BAND_SPECS = {
    "F090W": (9000.0, 8000.0, 10000.0),
    "F150W": (15000.0, 13000.0, 17000.0),
}


def _header(*, crpix1: float = 8.5, crpix2: float = 8.5) -> dict:
    return {
        "NAXIS": 2,
        "NAXIS1": 16,
        "NAXIS2": 16,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CUNIT1": "deg",
        "CUNIT2": "deg",
        "CRVAL1": 110.0,
        "CRVAL2": -73.0,
        "CRPIX1": crpix1,
        "CRPIX2": crpix2,
        "CD1_1": -1.0e-5,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": 1.0e-5,
    }


def _exposures(tmp_path: Path) -> list[ExposureCalibrationInput]:
    return [
        ExposureCalibrationInput(
            path=tmp_path / "a_crf.fits",
            band="F090W",
            detector="NRCB1",
            header=_header(),
            shape=(16, 16),
        ),
        ExposureCalibrationInput(
            path=tmp_path / "b_crf.fits",
            band="F150W",
            detector="NRCB2",
            header=_header(crpix1=9.5, crpix2=7.5),
            shape=(16, 16),
        ),
    ]


def test_official_bandpass_uses_full_tabulated_support(tmp_path: Path) -> None:
    throughput_dir = tmp_path / "throughput"
    throughput_dir.mkdir()
    f090 = throughput_dir / "F090W_NRCB1_throughput.txt"
    f150 = throughput_dir / "F150W_NRCB2_throughput.txt"
    np.savetxt(
        f090,
        np.array(
            [[0.8, 0.0], [0.85, 0.5], [0.9, 1.0], [0.95, 0.5], [1.0, 0.0]]
        ),
    )
    np.savetxt(
        f150,
        np.array(
            [[1.3, 0.0], [1.4, 0.5], [1.5, 1.0], [1.6, 0.5], [1.7, 0.0]]
        ),
    )
    manifest = tmp_path / "throughput.json"
    manifest.write_text(
        json.dumps({"NRCB1/F090W": str(f090), "NRCB2/F150W": str(f150)}),
        encoding="utf-8",
    )

    result = build_bandpass_calibration(
        _exposures(tmp_path),
        mode="official",
        num_samples=5,
        band_specs=BAND_SPECS,
        throughput_dir=throughput_dir,
        throughput_manifest=manifest,
    )

    assert result.wavelength.shape == (2, 5)
    assert result.wavelength[0, 0] < result.wavelength[0, -1]
    torch.testing.assert_close(result.weight.sum(-1), torch.ones(2))
    assert result.metadata[0]["source"] == str(f090)


def test_basic_bandpass_modes_are_retained(tmp_path: Path) -> None:
    exposures = _exposures(tmp_path)
    center = build_bandpass_calibration(
        exposures,
        mode="center",
        num_samples=9,
        band_specs=BAND_SPECS,
    )
    top_hat = build_bandpass_calibration(
        exposures,
        mode="top_hat",
        num_samples=5,
        band_specs=BAND_SPECS,
    )
    gaussian = build_bandpass_calibration(
        exposures,
        mode="gaussian",
        num_samples=5,
        band_specs=BAND_SPECS,
    )
    assert center.wavelength.shape == (2, 1)
    assert top_hat.wavelength.shape == (2, 5)
    assert gaussian.wavelength.shape == (2, 5)
    assert not torch.allclose(top_hat.weight, gaussian.weight)


def test_basic_psf_modes_are_retained(tmp_path: Path) -> None:
    exposures = _exposures(tmp_path)
    wavelength = torch.tensor(
        [[8000.0, 9000.0, 10000.0], [13000.0, 15000.0, 17000.0]]
    )
    none = build_psf_calibration(
        exposures,
        wavelength,
        mode="none",
        kernel_size=5,
    )
    assert none.bank is None

    gaussian = build_psf_calibration(
        exposures,
        wavelength,
        mode="gaussian",
        kernel_size=5,
        gaussian_fwhm_by_band={"F090W": 1.2, "F150W": 1.5},
    )
    assert gaussian.bank is not None
    assert gaussian.bank.kernel.shape == (2, 3, 5, 5)
    torch.testing.assert_close(
        gaussian.bank.kernel.sum((-1, -2)), torch.ones((2, 3))
    )

    kernel_dir = tmp_path / "kernels"
    kernel_dir.mkdir()
    base = np.ones((3, 3), dtype=np.float32)
    np.save(kernel_dir / "F090W.npy", base)
    np.save(kernel_dir / "F150W.npy", base)
    loaded = build_psf_calibration(
        exposures,
        wavelength,
        mode="npy",
        kernel_size=3,
        kernel_dir=kernel_dir,
    )
    assert loaded.bank is not None
    assert loaded.bank.kernel.shape == (2, 3, 3, 3)


def test_model_runs_with_wavelength_dependent_psf(tmp_path: Path) -> None:
    exposures = _exposures(tmp_path)
    headers = [exposure.header for exposure in exposures]
    canvas = build_union_canvas(headers, padding_pixels=3)
    model = WCSAwareMultiWavelengthINR(
        headers,
        canvas,
        encoding="fourier",
        spatial_frequencies=2,
        spectral_frequencies=2,
        hidden_dim=16,
        latent_dim=8,
        trunk_depth=1,
        decoder_depth=1,
        output_activation="linear",
    )
    wavelength = torch.tensor(
        [[8000.0, 9000.0], [14000.0, 15000.0]], dtype=torch.float32
    )
    weight = torch.full((2, 2), 0.5)
    psf = build_psf_calibration(
        exposures,
        wavelength,
        mode="gaussian",
        kernel_size=3,
        gaussian_fwhm_by_band={"F090W": 1.2, "F150W": 1.5},
    ).bank
    assert psf is not None
    uv = torch.tensor([[7.0, 7.0], [8.0, 6.0]], dtype=torch.float32)
    exposure_id = torch.tensor([0, 1], dtype=torch.long)
    prediction, valid = model.predict_observation(
        uv,
        exposure_id,
        wavelength,
        weight,
        psf=psf,
        checkpoint_psf_chunks=False,
    )
    assert prediction.shape == (2,)
    assert valid.all()
    assert torch.isfinite(prediction).all()


def test_fourier_and_spherical_models_run_without_psf(tmp_path: Path) -> None:
    headers = [exposure.header for exposure in _exposures(tmp_path)]
    canvas = build_union_canvas(headers, padding_pixels=2)
    uv = torch.tensor([[7.0, 7.0], [8.0, 6.0]], dtype=torch.float32)
    exposure_id = torch.tensor([0, 1], dtype=torch.long)
    wavelength = torch.tensor(
        [[8000.0, 9000.0], [14000.0, 15000.0]], dtype=torch.float32
    )
    weight = torch.full((2, 2), 0.5)
    for encoding in ("fourier", "spherical"):
        model = WCSAwareMultiWavelengthINR(
            headers,
            canvas,
            encoding=encoding,
            spatial_frequencies=2,
            spectral_frequencies=2,
            hidden_dim=16,
            latent_dim=8,
            trunk_depth=1,
            decoder_depth=1,
            output_activation="linear",
        )
        prediction, valid = model.predict_observation(
            uv, exposure_id, wavelength, weight
        )
        assert prediction.shape == (2,)
        assert valid.all()


def test_weighted_robust_loss_respects_mask() -> None:
    prediction = torch.tensor([1.0, 2.0, 100.0])
    target = torch.tensor([0.0, 2.0, 0.0])
    err = torch.tensor([1.0, 0.5, 1.0])
    mask = torch.tensor([True, True, False])
    loss, diagnostics = weighted_robust_loss(
        prediction, target, err, mask, kind="huber", delta=1.0
    )
    assert torch.isfinite(loss)
    torch.testing.assert_close(diagnostics["rmse"], torch.sqrt(torch.tensor(0.5)))
    torch.testing.assert_close(diagnostics["mae"], torch.tensor(0.5))


def test_whitened_l2_matches_sigma_normalized_formula() -> None:
    prediction = torch.tensor([2.0, 4.0])
    target = torch.tensor([1.0, 2.0])
    err = torch.tensor([0.5, 2.0])
    loss, diagnostics = weighted_robust_loss(
        prediction,
        target,
        err,
        torch.ones(2, dtype=torch.bool),
        kind="whitened_l2",
    )
    expected_whitened = torch.tensor([2.0, 1.0])
    torch.testing.assert_close(loss, 0.5 * expected_whitened.square().mean())
    torch.testing.assert_close(
        diagnostics["whitened_rmse"], expected_whitened.square().mean().sqrt()
    )


def test_short_long_batch_spec_is_exact_per_exposure(tmp_path: Path) -> None:
    def pool(index: int, band: str) -> ExposurePool:
        count = 32
        return ExposurePool(
            path=tmp_path / f"{index}.fits",
            band=band,
            detector="NRCB1" if band == "F090W" else "NRCB5",
            header=_header(),
            shape=(16, 16),
            uv=torch.arange(count * 2, dtype=torch.float32).reshape(count, 2),
            value=torch.arange(count, dtype=torch.float32),
            err=torch.ones(count),
        )

    pools = [pool(0, "F090W"), pool(1, "F090W"), pool(2, "F444W")]
    batch = sample_balanced_batch(
        pools,
        batch_size=[3, 7],
        generator=torch.Generator().manual_seed(4),
    )
    _, exposure_id, _, _, _ = batch
    assert effective_batch_size(pools, [3, 7]) == 13
    assert torch.bincount(exposure_id, minlength=3).tolist() == [3, 3, 7]


def test_frozen_alignment_has_no_trainable_alignment_parameters(
    tmp_path: Path,
) -> None:
    headers = [exposure.header for exposure in _exposures(tmp_path)]
    canvas = build_union_canvas(headers, padding_pixels=2)
    for mode in ("original_wcs", "tweakreg_frozen"):
        model = WCSAwareMultiWavelengthINR(
            headers,
            canvas,
            encoding="fourier",
            spatial_frequencies=2,
            spectral_frequencies=2,
            hidden_dim=16,
            latent_dim=8,
            trunk_depth=1,
            decoder_depth=1,
            alignment_mode=mode,
            output_activation="linear",
        )
        assert list(model.feature_alignment.parameters()) == []
        assert model.alignment_summary() == {"kind": mode, "trainable": False}
        torch.testing.assert_close(model.alignment_regularization(), torch.tensor(0.0))
