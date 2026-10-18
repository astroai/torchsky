from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

import torchsky
import torchsky.map.product as map_product_mod
from torchsky.map import (
    FieldDomain,
    FieldObservation,
    GaussianBasisField,
    ImageObservation,
    MapBandpassSpec,
    estimate_map_inverse_variance,
    make_field_model,
    plan_map_product,
    render_field_observation,
    render_field_ensemble_product,
    render_field_product,
    stack_map_product,
)
from torchsky.map.scenes import ContinuousFieldModel, DiscreteTargetImage
from torchsky.map.scenes import HealpixMapScene
from torchsky.map.operators import spherical_spectral_operator_bundle
from torchsky.map.solvers import build_solver_hooks
from torchsky.spectral import SpectralAxis
import torchfits

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


class _ConstantProductField(torch.nn.Module):
    def __init__(self, value: float, *, variance: float | None = None) -> None:
        super().__init__()
        self.domain = FieldDomain(
            center_ra_deg=12.0,
            center_dec_deg=-3.0,
            spatial_scale_deg=0.05,
            wavelength_center=5500.0,
            wavelength_scale=1000.0,
        )
        self.value = torch.nn.Parameter(torch.tensor(float(value)))
        self.predictive_distribution_available = variance is not None
        self.register_buffer(
            "log_variance",
            torch.tensor(0.0 if variance is None else math.log(float(variance))),
        )

    def forward(self, ra, dec, wavelength):
        del dec, wavelength
        return torch.zeros_like(ra) + self.value

    def forward_distribution(self, ra, dec, wavelength):
        if not self.predictive_distribution_available:
            raise ValueError("predictive variance is unavailable")
        mean = self.forward(ra, dec, wavelength)
        return mean, torch.zeros_like(mean) + self.log_variance


def test_plan_map_product_single_band_stack_matches_image_plane(tmp_path: Path) -> None:
    path_a = tmp_path / "a.fits"
    path_b = tmp_path / "b.fits"
    header = _simple_header()
    data_a = torch.arange(36, dtype=torch.float32).reshape(6, 6)
    data_b = data_a + 4.0
    torchfits.write(str(path_a), data_a, header=header, overwrite=True)
    torchfits.write(str(path_b), data_b, header=header, overwrite=True)

    observations = [ImageObservation(str(path_a)), ImageObservation(str(path_b))]
    product = plan_map_product(observations)
    result = stack_map_product(observations, product)

    assert product.n_bands == 1
    assert result.data.shape == (1, 6, 6)
    torch.testing.assert_close(result.data[0], (data_a + data_b) / 2.0)
    assert result.uncertainty is None


def test_explicit_unit_inverse_variance_is_not_confused_with_default_weight(
    tmp_path: Path,
) -> None:
    path = tmp_path / "unit-ivar.fits"
    header = _simple_header()
    torchfits.write(str(path), torch.ones((6, 6)), header=header, overwrite=True)
    default = [ImageObservation(str(path), weight=1.0)]
    explicit = [
        ImageObservation(
            str(path),
            weight=1.0,
            weight_is_inverse_variance=True,
        )
    ]

    default_result = stack_map_product(default, plan_map_product(default))
    explicit_result = stack_map_product(explicit, plan_map_product(explicit))

    assert default_result.uncertainty is None
    assert explicit_result.uncertainty is not None
    torch.testing.assert_close(
        explicit_result.uncertainty.inverse_variance,
        torch.ones((1, 6, 6)),
    )


def test_map_mask_without_noise_does_not_invent_inverse_variance(
    tmp_path: Path,
) -> None:
    path = tmp_path / "masked.fits"
    mask_path = tmp_path / "mask.fits"
    header = _simple_header()
    torchfits.write(str(path), torch.ones((6, 6)), header=header, overwrite=True)
    torchfits.write(
        str(mask_path),
        torch.ones((6, 6)),
        header=header,
        overwrite=True,
    )
    observations = [ImageObservation(str(path), mask_path=str(mask_path))]

    result = stack_map_product(observations, plan_map_product(observations))

    assert result.weight is not None
    assert result.uncertainty is None


def test_plan_map_product_single_band_honors_union_footprint(tmp_path: Path) -> None:
    path_a = tmp_path / "a.fits"
    path_b = tmp_path / "b.fits"
    header_a = _simple_header()
    header_b = {**header_a, "CRVAL1": float(header_a["CRVAL1"]) + 0.04}
    data = torch.ones((6, 6), dtype=torch.float32)
    torchfits.write(str(path_a), data, header=header_a, overwrite=True)
    torchfits.write(str(path_b), data, header=header_b, overwrite=True)

    observations = [ImageObservation(str(path_a)), ImageObservation(str(path_b))]
    reference = plan_map_product(observations, union_footprint=False)
    union = plan_map_product(
        observations,
        union_footprint=True,
        footprint_padding=1.0,
    )

    assert reference.shape == (6, 6)
    assert union.shape[1] > reference.shape[1]


def test_plan_map_product_spectral_honors_union_footprint(tmp_path: Path) -> None:
    path_a = tmp_path / "a.fits"
    path_b = tmp_path / "b.fits"
    header_a = _simple_header()
    header_b = {**header_a, "CRVAL1": float(header_a["CRVAL1"]) + 0.04}
    data = torch.ones((6, 6), dtype=torch.float32)
    torchfits.write(str(path_a), data, header=header_a, overwrite=True)
    torchfits.write(str(path_b), data, header=header_b, overwrite=True)
    observations = [
        ImageObservation(str(path_a), wavelength=5000.0),
        ImageObservation(str(path_b), wavelength=5000.0),
    ]

    union = plan_map_product(
        observations,
        union_footprint=True,
        footprint_padding=1.0,
    )

    assert union.shape[1] > 6


def test_plan_map_product_header_override_owns_output_shape(tmp_path: Path) -> None:
    path = tmp_path / "input.fits"
    torchfits.write(
        str(path),
        torch.ones((6, 6), dtype=torch.float32),
        header=_simple_header(),
        overwrite=True,
    )
    target = {**_simple_header(), "NAXIS1": 9, "NAXIS2": 4, "CRVAL1": 33.0}

    product = plan_map_product(
        [ImageObservation(str(path))],
        header_overrides=target,
        union_footprint=True,
    )

    assert product.shape == (4, 9)
    assert product.target_header["CRVAL1"] == 33.0


def test_plan_map_product_cube_tracks_requested_bands(tmp_path: Path) -> None:
    path_blue = tmp_path / "blue.fits"
    path_red = tmp_path / "red.fits"
    header = _simple_header()
    blue = torch.full((6, 6), 2.0, dtype=torch.float32)
    red = torch.full((6, 6), 5.0, dtype=torch.float32)
    torchfits.write(str(path_blue), blue, header=header, overwrite=True)
    torchfits.write(str(path_red), red, header=header, overwrite=True)

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
    assert result.weight is None
    assert result.uncertainty is None


def test_render_field_product_labels_likelihood_variance_as_detector() -> None:
    product = torchsky.map.MapProductSpec(
        target_header=_simple_header(),
        bands=(MapBandpassSpec(label="g", transmission_center=5000.0),),
    )

    result = render_field_product(
        _ConstantProductField(2.0, variance=4.0),
        product,
        num_spectral_samples=1,
        spatial_tile_hw=(4, 3),
    )

    assert result.uncertainty is not None
    assert result.uncertainty.kind == "model_predictive"
    torch.testing.assert_close(
        result.uncertainty.components["detector"],
        torch.full_like(result.data, 4.0),
    )


def test_render_field_ensemble_product_separates_uncertainty_components() -> None:
    product = torchsky.map.MapProductSpec(
        target_header=_simple_header(),
        bands=(MapBandpassSpec(label="g", transmission_center=5000.0),),
    )
    calibration_draws = (
        torch.ones((1, 6, 6)),
        torch.full((1, 6, 6), 3.0),
    )

    result = render_field_ensemble_product(
        (
            _ConstantProductField(1.0, variance=4.0),
            _ConstantProductField(3.0, variance=4.0),
        ),
        product,
        member_seeds=(11, 29),
        calibration_draws=calibration_draws,
        calibration_seeds=(101, 103),
        num_spectral_samples=1,
        spatial_tile_hw=(4, 3),
    )

    assert result.uncertainty is not None
    assert result.uncertainty.kind == "ensemble_predictive"
    torch.testing.assert_close(result.data, torch.full_like(result.data, 2.0))
    torch.testing.assert_close(
        result.uncertainty.components["epistemic"],
        torch.full_like(result.data, 2.0),
    )
    torch.testing.assert_close(
        result.uncertainty.components["detector"],
        torch.full_like(result.data, 4.0),
    )
    torch.testing.assert_close(
        result.uncertainty.components["calibration"],
        torch.full_like(result.data, 2.0),
    )
    torch.testing.assert_close(
        result.uncertainty.variance,
        torch.full_like(result.data, 8.0),
    )
    assert result.provenance is not None
    assert result.provenance["uncertainty"]["member_seeds"] == [11, 29]


def test_render_field_ensemble_product_rejects_mixed_detector_evidence() -> None:
    product = torchsky.map.MapProductSpec(
        target_header=_simple_header(),
        bands=(MapBandpassSpec(label="g", transmission_center=5000.0),),
    )

    with pytest.raises(ValueError, match="all provide detector"):
        render_field_ensemble_product(
            (
                _ConstantProductField(1.0, variance=4.0),
                _ConstantProductField(3.0),
            ),
            product,
            member_seeds=(11, 29),
            num_spectral_samples=1,
        )


def test_render_field_product_supports_spatial_quadrature() -> None:
    class ConstantField(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.value = torch.nn.Parameter(torch.tensor(2.5))

        def forward(self, ra, dec, wavelength):
            del dec, wavelength
            return torch.zeros_like(ra) + self.value

    product = torchsky.map.MapProductSpec(
        target_header=_simple_header(),
        bands=(MapBandpassSpec(label="g", transmission_center=5000.0),),
    )

    result = map_product_mod.render_field_product(
        ConstantField(),
        product,
        num_spectral_samples=1,
        spatial_sampling="gauss_legendre_2x2",
        spatial_tile_hw=(4, 3),
    )

    torch.testing.assert_close(result.data, torch.full((1, 6, 6), 2.5))


def test_tiled_field_product_allocates_on_model_device(monkeypatch) -> None:
    class MetaField(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.value = torch.nn.Parameter(torch.empty((), device="meta"))

    product = torchsky.map.MapProductSpec(
        target_header=_simple_header(),
        bands=(MapBandpassSpec(label="g", transmission_center=5000.0),),
    )

    def fake_render(model, observation, **kwargs):
        del kwargs
        return torch.zeros(
            observation.image.shape,
            dtype=torch.float32,
            device=next(model.parameters()).device,
        )

    monkeypatch.setattr(
        "torchsky.map.field.rendering.render_field_observation", fake_render
    )
    result = map_product_mod.render_field_product(
        MetaField(), product, spatial_tile_hw=(3, 3)
    )

    assert result.data.device.type == "meta"


def test_square_drizzle_reports_squared_coefficient_inverse_variance(
    tmp_path: Path,
) -> None:
    source_header = _simple_header()
    target_header = dict(source_header)
    target_header["CRPIX1"] = float(target_header["CRPIX1"]) - 0.5
    path = tmp_path / "variance_drizzle.fits"
    torchfits.write(
        str(path),
        torch.ones((6, 6), dtype=torch.float32),
        header=source_header,
        overwrite=True,
    )
    product = torchsky.map.MapProductSpec(
        target_header=target_header,
        bands=(MapBandpassSpec(label="mono"),),
    )

    result = stack_map_product(
        [
            ImageObservation(
                str(path),
                weight=4.0,
                weight_is_inverse_variance=True,
            )
        ],
        product,
        resampling_kernel="drizzle_square",
    )

    assert result.uncertainty is not None
    assert result.uncertainty.kind == "propagated_linear"
    assert result.weight is not None
    assert float(result.weight[0, 3, 3]) == pytest.approx(4.0, rel=1.0e-5)
    assert float(result.uncertainty.inverse_variance[0, 3, 3]) == pytest.approx(
        8.0, rel=1.0e-5
    )
    assert result.uncertainty.correlation_factor is not None
    assert float(result.uncertainty.correlation_factor[0, 3, 3]) == pytest.approx(
        2.0, rel=1.0e-5
    )


@pytest.mark.parametrize(
    "kernel",
    ["bilinear", "conservative_polygon", "splat_box"],
)
@pytest.mark.parametrize("value_semantics", ["surface_brightness", "flux"])
def test_half_pixel_projection_reports_squared_coefficient_inverse_variance(
    tmp_path: Path,
    kernel: str,
    value_semantics: str,
) -> None:
    path = tmp_path / f"{kernel}-{value_semantics}.fits"
    source_header = _simple_header()
    target_header = {
        **source_header,
        "CRPIX1": float(source_header["CRPIX1"]) + 0.5,
    }
    torchfits.write(
        str(path),
        torch.ones((6, 6), dtype=torch.float32),
        header=source_header,
        overwrite=True,
    )
    observations = [
        ImageObservation(
            str(path),
            weight=1.0,
            weight_is_inverse_variance=True,
        )
    ]
    product = torchsky.map.MapProductSpec(
        target_header=target_header,
        bands=(MapBandpassSpec(label="science"),),
        value_semantics=value_semantics,
    )

    result = stack_map_product(
        observations,
        product,
        resampling_kernel=kernel,
    )

    assert result.uncertainty is not None
    assert float(result.uncertainty.inverse_variance[0, 3, 3]) == pytest.approx(
        2.0,
        rel=0.0,
        abs=2.0e-5,
    )
    assert result.uncertainty.correlation_factor is not None
    assert float(result.uncertainty.correlation_factor[0, 3, 3]) == pytest.approx(
        2.0,
        rel=0.0,
        abs=2.0e-5,
    )


def test_map_product_context_uses_multiple_words_and_records_observations(
    tmp_path: Path,
) -> None:
    header = _simple_header()
    path = tmp_path / "context.fits"
    torchfits.write(str(path), torch.ones((6, 6)), header=header, overwrite=True)
    observations = [ImageObservation(str(path)) for _ in range(33)]
    product = torchsky.map.MapProductSpec(
        target_header=header,
        bands=(MapBandpassSpec(label="mono"),),
    )

    result = stack_map_product(observations, product)

    assert result.context is not None
    assert result.context.shape == (1, 2, 6, 6)
    assert int(result.context[0, 0, 3, 3]) == 2**32 - 1
    assert int(result.context[0, 1, 3, 3]) == 1
    assert result.provenance is not None
    assert len(result.provenance["context_observations"]) == 33


def test_map_product_result_exposes_target_render_specs() -> None:
    product = torchsky.map.MapProductSpec(
        target_header=_simple_header(),
        bands=(
            MapBandpassSpec(label="g", transmission_center=5000.0),
            MapBandpassSpec(label="r", transmission_center=6500.0),
        ),
        value_semantics="flux",
    )
    result = torchsky.map.MapProductResult(
        product=product,
        data=torch.zeros((2, 6, 6), dtype=torch.float32),
        weight=torch.ones((2, 6, 6), dtype=torch.float32),
        header=_simple_header(),
    )

    specs = result.target_render_specs(psf_specs=("psf-g", "psf-r"))

    assert tuple(spec.band_label for spec in specs) == ("g", "r")
    assert tuple(spec.psf_spec for spec in specs) == ("psf-g", "psf-r")
    assert all(spec.value_semantics == "flux" for spec in specs)


def test_map_uncertainty_product_exposes_honest_inverse_variance() -> None:
    variance = torch.tensor([[[4.0, 0.0]]], dtype=torch.float32)
    uncertainty = torchsky.map.MapUncertaintyProduct(
        variance=variance,
        kind="propagated_linear",
        components={"detector": variance},
        correlation_factor=torch.tensor([[[2.0, 0.0]]]),
    )

    torch.testing.assert_close(
        uncertainty.inverse_variance,
        torch.tensor([[[0.25, 0.0]]], dtype=torch.float32),
    )
    with pytest.raises(ValueError, match="component shapes"):
        torchsky.map.MapUncertaintyProduct(
            variance=variance,
            kind="propagated_linear",
            components={"detector": torch.ones((2, 2))},
        )
    with pytest.raises(ValueError, match="positive wherever variance is positive"):
        torchsky.map.MapUncertaintyProduct(
            variance=variance,
            kind="propagated_linear",
            correlation_factor=torch.zeros_like(variance),
        )


def test_internal_scene_product_render_supports_discrete_multiband_cube() -> None:
    product = torchsky.map.MapProductSpec(
        target_header=_simple_header(),
        bands=(
            MapBandpassSpec(label="g", transmission_center=5000.0),
            MapBandpassSpec(label="r", transmission_center=6500.0),
        ),
    )
    scene = DiscreteTargetImage(
        torch.stack(
            [
                torch.full((6, 6), 2.0, dtype=torch.float32),
                torch.full((6, 6), 5.0, dtype=torch.float32),
            ],
            dim=0,
        )
    )

    result = map_product_mod.render_scene_product(scene, product)

    torch.testing.assert_close(result.data[0], torch.full((6, 6), 2.0))
    torch.testing.assert_close(result.data[1], torch.full((6, 6), 5.0))
    assert result.weight is None
    assert result.uncertainty is None
    assert result.mask is not None and bool(torch.all(result.mask))


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
            torchfits.write(str(path), rendered, header=header, overwrite=True)
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


def test_render_field_product_uses_shared_scene_product_renderer() -> None:
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

    direct = render_field_product(model, product, num_spectral_samples=3)
    shared = map_product_mod.render_scene_product(
        ContinuousFieldModel(model),
        product,
        num_spectral_samples=3,
    )

    torch.testing.assert_close(direct.data, shared.data)
    torch.testing.assert_close(direct.weight, shared.weight)


def test_render_scene_product_accepts_plain_module_scene() -> None:
    product = torchsky.map.MapProductSpec(
        target_header=_simple_header(),
        bands=(MapBandpassSpec(label="g", transmission_center=5000.0),),
    )
    model = GaussianBasisField(
        FieldDomain(
            center_ra_deg=12.0,
            center_dec_deg=-3.0,
            spatial_scale_deg=0.05,
            wavelength_center=5000.0,
            wavelength_scale=500.0,
        )
    )

    result = map_product_mod.render_scene_product(
        model,
        product,
        num_spectral_samples=1,
    )

    assert result.data.shape == (1, 6, 6)
    assert result.weight is None
    assert result.mask is not None and result.mask.shape == (1, 6, 6)


def test_render_scene_product_renders_healpix_scene_through_solver_hooks() -> None:
    product = torchsky.map.MapProductSpec(
        target_header=_simple_header(),
        bands=(MapBandpassSpec(label="mono", transmission_center=5000.0),),
    )
    scene = HealpixMapScene(
        torch.ones(12 * 4 * 4, dtype=torch.float32),
        nside=4,
    )

    result = map_product_mod.render_scene_product(
        scene,
        product,
        solver_hooks=build_solver_hooks(
            spherical_spectral_operator_bundle(mode="harmonic_direct")
        ),
    )

    torch.testing.assert_close(
        result.data[0],
        torch.ones((6, 6), dtype=torch.float32),
        atol=5.0e-2,
        rtol=0.0,
    )
    assert result.weight is None
    assert result.mask is not None and bool(torch.all(result.mask))




def test_estimate_map_inverse_variance_support_proxy_flags_extrapolated_regions() -> (
    None
):
    support_weight = torch.zeros((1, 7, 7), dtype=torch.float32)
    support_weight[:, 2:5, 2:5] = 4.0
    ivar = estimate_map_inverse_variance(
        support_weight,
        mode="support_density_proxy",
        kernel_size=3,
    )
    sigma = torch.rsqrt(ivar)

    assert ivar.shape == support_weight.shape
    assert float(sigma[0, 3, 3]) < float(sigma[0, 0, 0])
    assert float(sigma[0, 1, 1]) < float(sigma[0, 0, 0])


def test_estimate_map_inverse_variance_never_invents_field_weights() -> None:
    product = torchsky.map.MapProductSpec(
        target_header=_simple_header(),
        bands=(MapBandpassSpec(label="single"),),
    )
    result = torchsky.map.MapProductResult(
        product=product,
        data=torch.ones((1, 6, 6)),
        weight=None,
        header=_simple_header(),
    )

    with pytest.raises(ValueError, match="inverse variance is unavailable"):
        estimate_map_inverse_variance(result)

    variance = torch.full_like(result.data, 4.0)
    result.uncertainty = torchsky.map.MapUncertaintyProduct(
        variance=variance,
        kind="model_predictive",
        components={"epistemic": variance},
    )
    torch.testing.assert_close(
        estimate_map_inverse_variance(result),
        torch.full_like(result.data, 0.25),
    )




