from __future__ import annotations

from pathlib import Path

import pytest
import torch

import torchsky
from torchsky.map import (
    FieldDomain,
    FieldObservation,
    GaussianBasisField,
    ImageObservation,
    MapBandpassSpec,
    check_map_benchmark_regressions,
    evaluate_map_product,
    make_field_model,
    plan_map_product,
    render_field_observation,
    render_field_product,
    stack_map_product,
)
from torchsky.spectral import SpectralAxis

pytest.importorskip("torchsky.cpp")


def _simple_header() -> dict:
    return {
        "NAXIS": 2,
        "NAXIS1": 6,
        "NAXIS2": 6,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 3.0,
        "CRPIX2": 3.0,
        "CRVAL1": 12.0,
        "CRVAL2": -3.0,
        "CDELT1": -0.01,
        "CDELT2": 0.01,
    }


def test_plan_map_product_single_band_stack_matches_image_plane(tmp_path: Path) -> None:
    path_a = tmp_path / "a.fits"
    path_b = tmp_path / "b.fits"
    header = _simple_header()
    data_a = torch.arange(36, dtype=torch.float32).reshape(6, 6)
    data_b = data_a + 4.0
    torchsky.write(str(path_a), data_a, header=header, overwrite=True)
    torchsky.write(str(path_b), data_b, header=header, overwrite=True)

    observations = [ImageObservation(str(path_a)), ImageObservation(str(path_b))]
    product = plan_map_product(observations)
    result = stack_map_product(observations, product)

    assert product.n_bands == 1
    assert result.data.shape == (1, 6, 6)
    torch.testing.assert_close(result.data[0], (data_a + data_b) / 2.0)


def test_plan_map_product_cube_tracks_requested_bands(tmp_path: Path) -> None:
    path_blue = tmp_path / "blue.fits"
    path_red = tmp_path / "red.fits"
    header = _simple_header()
    blue = torch.full((6, 6), 2.0, dtype=torch.float32)
    red = torch.full((6, 6), 5.0, dtype=torch.float32)
    torchsky.write(str(path_blue), blue, header=header, overwrite=True)
    torchsky.write(str(path_red), red, header=header, overwrite=True)

    observations = [
        ImageObservation(
            str(path_blue),
            wavelength=5000.0,
            spectral_min=4800.0,
            spectral_max=5200.0,
            band_name="blue",
        ),
        ImageObservation(
            str(path_red),
            wavelength=6500.0,
            spectral_min=6200.0,
            spectral_max=6800.0,
            band_name="red",
        ),
    ]
    product = plan_map_product(
        observations,
        spectral_axis=SpectralAxis(
            values=torch.tensor([5000.0, 6500.0], dtype=torch.float64),
            unit="Angstrom",
            type="WAVE",
        ),
        spectral_bin_edges=torch.tensor([4800.0, 5200.0, 6800.0], dtype=torch.float64),
    )
    result = stack_map_product(observations, product)

    assert tuple(band.label for band in product.bands) == ("blue", "red")
    assert result.data.shape == (2, 6, 6)
    torch.testing.assert_close(result.data[0], blue)
    torch.testing.assert_close(result.data[1], red)


def test_render_field_product_matches_product_shape() -> None:
    product = torchsky.map.MapProductSpec(
        target_header=_simple_header(),
        bands=(
            MapBandpassSpec(label="g", spectral_min=4700.0, spectral_max=5200.0),
            MapBandpassSpec(label="r", spectral_min=6100.0, spectral_max=6800.0),
        ),
    )
    model = GaussianBasisField(
        FieldDomain(
            center_ra_deg=12.0,
            center_dec_deg=-3.0,
            spatial_scale_deg=0.05,
            wavelength_center=5750.0,
            wavelength_scale=1200.0,
        )
    )
    result = render_field_product(model, product)

    assert result.data.shape == (2, 6, 6)
    assert result.weight.shape == (2, 6, 6)
    assert bool(torch.all(result.weight == 1.0))


def test_render_field_product_matches_exact_transmission_per_band_stacks(
    tmp_path: Path,
) -> None:
    torch.manual_seed(2026)
    domain = FieldDomain(
        center_ra_deg=12.0,
        center_dec_deg=-3.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=1000.0,
        coord_mode="sphere_xyz",
    )
    model = make_field_model("sky_encoder_mlp", domain)
    header = _simple_header()
    wavelengths = (4800.0, 5600.0, 6400.0)
    labels = ("blue", "green", "red")
    observations: list[ImageObservation] = []
    expected_planes: list[torch.Tensor] = []
    for wavelength, label in zip(wavelengths, labels, strict=True):
        template = FieldObservation(
            image=torch.zeros((6, 6), dtype=torch.float32),
            header=header,
            wavelength=wavelength,
            band_name=label,
        )
        rendered = render_field_observation(
            model,
            template,
            num_spectral_samples=1,
        ).detach()
        expected_planes.append(rendered)
        for replica, scalar_weight in enumerate((1.0, 2.5)):
            path = tmp_path / f"{label}_{replica}.fits"
            torchsky.write(str(path), rendered, header=header, overwrite=True)
            observations.append(
                ImageObservation(
                    str(path),
                    wavelength=wavelength,
                    band_name=label,
                    weight=scalar_weight,
                )
            )
    product = plan_map_product(
        observations,
        spectral_axis=SpectralAxis(
            values=torch.tensor(wavelengths, dtype=torch.float64),
            unit="Angstrom",
            type="WAVE",
        ),
    )
    stacked = stack_map_product(
        observations,
        product,
        resampling_kernel="bilinear",
        stacking_method="weighted_mean",
    )
    rendered_product = render_field_product(
        model,
        product,
        num_spectral_samples=1,
    )

    torch.testing.assert_close(
        rendered_product.data,
        torch.stack(expected_planes, dim=0),
        rtol=0.0,
        atol=1.0e-5,
    )
    torch.testing.assert_close(
        stacked.data,
        rendered_product.data,
        rtol=0.0,
        atol=1.0e-5,
    )
    assert tuple(band.label for band in product.bands) == labels
    assert bool(torch.all(stacked.weight > 0.0))


def test_evaluate_map_product_uses_product_level_metrics() -> None:
    product = torchsky.map.MapProductSpec(
        target_header=_simple_header(),
        bands=(MapBandpassSpec(label="single"),),
    )
    truth = torch.ones((1, 6, 6), dtype=torch.float32)
    pred = truth.clone()
    pred[0, 0, 0] = 3.0
    metrics = evaluate_map_product(
        torchsky.map.MapProductResult(
            product=product,
            data=pred,
            weight=torch.ones_like(pred),
            header=_simple_header(),
        ),
        truth,
    )

    assert metrics["reconstruction_rmse"] is not None
    assert float(metrics["reconstruction_rmse"]) > 0.0
    assert float(metrics["coverage_fraction"]) == pytest.approx(1.0)
    assert float(metrics["total_flux_relative_error"]) > 0.0
    assert metrics["band_reconstruction_rmse"] == [pytest.approx(float(metrics["reconstruction_rmse"]))]
    assert len(metrics["band_flux_relative_error"]) == 1
    assert float(metrics["band_flux_relative_error"][0]) > 0.0


def test_check_map_benchmark_regressions_respects_skip_time_flag() -> None:
    baseline = [
        {
            "case": "case",
            "library": "torchsky",
            "method": "coadd",
            "variant": "bilinear",
            "reconstruction_rmse": 1.0,
            "signal_to_residual_db": 10.0,
            "total_flux_relative_error": 0.1,
            "time_ms": 1.0,
            "skip_time_regression": True,
        }
    ]
    current = [
        {
            "case": "case",
            "library": "torchsky",
            "method": "coadd",
            "variant": "bilinear",
            "reconstruction_rmse": 1.05,
            "signal_to_residual_db": 10.0,
            "total_flux_relative_error": 0.105,
            "time_ms": 100.0,
        }
    ]
    errors = check_map_benchmark_regressions(current, baseline)
    assert errors == []
