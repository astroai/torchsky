from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from torchsky.catalogs import build_source_groups, fit_forced_photometry
from torchsky.map.field import FieldObservation, PSFSpec
from torchsky.scene import SourceCatalogScene
from torchsky.sphere import MOC
from torchsky.wcs import WCS


def _header(
    *,
    size: int = 25,
    crval_ra: float = 180.0,
    crpix_shift: float = 0.0,
) -> dict:
    center = 0.5 * (size + 1)
    return {
        "NAXIS": 2,
        "NAXIS1": size,
        "NAXIS2": size,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": center + crpix_shift,
        "CRPIX2": center,
        "CRVAL1": crval_ra,
        "CRVAL2": 0.0,
        "CDELT1": -1.0 / 3600.0,
        "CDELT2": 1.0 / 3600.0,
    }


def _positions(
    header: dict, pixels: list[tuple[float, float]]
) -> tuple[torch.Tensor, torch.Tensor]:
    coordinates = torch.tensor(pixels, dtype=torch.float64)
    return WCS(header).pixel_to_world(coordinates[:, 0], coordinates[:, 1])


def _blank_observation(
    header: dict,
    *,
    band: str | None = None,
    psf_sigma: float = 1.0,
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
) -> FieldObservation:
    shape = (int(header["NAXIS2"]), int(header["NAXIS1"]))
    image = torch.zeros(shape, dtype=dtype, device=device)
    return FieldObservation(
        image=image,
        header=header,
        weight=torch.ones_like(image),
        band_name=band,
        psf_spec=PSFSpec.gaussian(psf_sigma),
    )


def _render_observations(
    scene: SourceCatalogScene,
    observations: list[FieldObservation],
) -> list[FieldObservation]:
    return [
        replace(item, image=scene.render_observation(item)) for item in observations
    ]


def _dense_design(
    scene: SourceCatalogScene,
    observations: list[FieldObservation],
) -> torch.Tensor:
    rows = []
    for observation in observations:
        columns = []
        for parameter in range(scene.parameter_count):
            state = torch.zeros(
                scene.parameter_count, dtype=scene.dtype, device=scene.device
            )
            state[parameter] = 1.0
            columns.append(
                scene.render_observation(observation, parameter_values=state).reshape(
                    -1
                )
            )
        rows.append(torch.stack(columns, dim=1))
    return torch.cat(rows, dim=0)


def test_source_catalog_scene_accepts_tensor_boundary_and_fixed_morphologies() -> None:
    header = _header()
    lon, lat = _positions(header, [(12.0, 12.0), (8.0, 12.0), (17.0, 12.0)])
    templates = torch.zeros((3, 5, 5), dtype=torch.float64)
    templates[2, 1:4, 1:4] = torch.tensor(
        [[1.0, 2.0, 1.0], [2.0, 5.0, 2.0], [1.0, 2.0, 1.0]],
        dtype=torch.float64,
    )
    supports = tuple(
        MOC.from_circle(
            float(lon[index]), float(lat[index]), 3.0 / 3600.0, max_order=13
        )
        for index in range(3)
    )
    scene = SourceCatalogScene(
        lon,
        lat,
        component_ids=torch.tensor([11, 12, 13]),
        morphology=("point", "gaussian", "template"),
        gaussian_parameters=torch.tensor(
            [[0.0, 0.0, 0.0], [1.2, 0.6, 30.0], [0.0, 0.0, 0.0]],
            dtype=torch.float64,
        ),
        fixed_templates=templates,
        template_pixel_scale_arcsec=0.25,
        coefficients=torch.ones((3, 1), dtype=torch.float64),
        support_footprints=supports,
    )
    observation = _blank_observation(header, psf_sigma=0.7)

    stamps = [scene.render_component(index, observation) for index in range(3)]
    catalog = scene.tensor_catalog()

    assert all(stamp is not None for stamp in stamps)
    for stamp in stamps:
        assert stamp is not None
        torch.testing.assert_close(
            stamp.data.sum(),
            torch.tensor(1.0, dtype=torch.float64),
            rtol=2e-3,
            atol=2e-3,
        )
    assert all(isinstance(value, torch.Tensor) for value in catalog.values())
    torch.testing.assert_close(scene.support_footprint(2).uniq, supports[2].uniq)


def test_source_groups_use_exposure_mocs_and_connect_crowded_supports() -> None:
    header_a = _header(crval_ra=180.0)
    header_b = _header(crval_ra=181.0)
    lon_a, lat_a = _positions(header_a, [(12.0, 12.0), (13.0, 12.0)])
    lon = torch.cat((lon_a, torch.tensor([181.0], dtype=torch.float64)))
    lat = torch.cat((lat_a, torch.tensor([0.0], dtype=torch.float64)))
    scene = SourceCatalogScene(
        lon,
        lat,
        coefficients=torch.zeros((3, 1), dtype=torch.float64),
        support_radius_arcsec=torch.tensor([2.0, 2.0, 1.0], dtype=torch.float64),
    )
    observations = [
        _blank_observation(header_a),
        _blank_observation(header_b),
    ]

    grouping = build_source_groups(scene, observations)

    assert grouping.exposure_sources[0].tolist() == [0, 1]
    assert grouping.exposure_sources[1].tolist() == [2]
    assert [group.tolist() for group in grouping.blend_groups] == [[0, 1], [2]]
    assert [value.tolist() for value in grouping.group_exposures] == [[0], [1]]


def test_signed_multiband_blend_matches_dense_flux_and_covariance_oracle() -> None:
    header = _header()
    lon, lat = _positions(header, [(11.5, 12.0), (13.0, 12.0)])
    truth = SourceCatalogScene(
        lon,
        lat,
        coefficients=torch.tensor([[4.0, 2.0], [1.5, 3.5]], dtype=torch.float64),
        band_labels=("g", "r"),
    )
    blank = [
        _blank_observation(header, band="g", psf_sigma=1.2),
        _blank_observation(header, band="r", psf_sigma=0.8),
    ]
    blank[0].weight = torch.linspace(0.5, 2.0, 25 * 25, dtype=torch.float64).reshape(
        25, 25
    )
    blank[1].weight = torch.linspace(1.8, 0.7, 25 * 25, dtype=torch.float64).reshape(
        25, 25
    )
    observations = _render_observations(truth, blank)
    fit_scene = SourceCatalogScene(
        lon,
        lat,
        coefficients=torch.zeros((2, 2), dtype=torch.float64),
        band_labels=("g", "r"),
    )

    result = fit_forced_photometry(observations, fit_scene, background="none")
    design = _dense_design(fit_scene, blank)
    dense_weight = torch.cat(
        [item.weight.reshape(-1) for item in blank if item.weight is not None]
    )
    expected_covariance = torch.linalg.inv(design.T @ (design * dense_weight[:, None]))

    torch.testing.assert_close(
        result.flux, truth.tensor_catalog()["coefficients"], rtol=1e-9, atol=1e-9
    )
    torch.testing.assert_close(
        result.covariance.to_dense(), expected_covariance, rtol=1e-8, atol=1e-8
    )
    query = torch.tensor([1.0, -0.5, 0.25, 2.0], dtype=torch.float64)
    torch.testing.assert_close(
        result.covariance.query_variance(query),
        query @ expected_covariance @ query,
    )
    for prediction, observation in zip(result.predictions, observations, strict=True):
        torch.testing.assert_close(prediction, observation.image, rtol=1e-9, atol=1e-9)
    assert len(result.grouping.blend_groups) == 1
    assert result.diagnostics[0].data_rank == 4
    assert not result.diagnostics[0].gauge_detected


def test_spectral_basis_recovers_flux_and_color_with_missing_band_diagnostic() -> None:
    header = _header()
    lon, lat = _positions(header, [(12.0, 12.0)])
    basis = torch.tensor([[1.0, -1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
    truth = SourceCatalogScene(
        lon,
        lat,
        coefficients=torch.tensor([[5.0, 1.25]], dtype=torch.float64),
        band_labels=("g", "r", "i"),
        spectral_basis=basis,
    )
    observations = _render_observations(
        truth,
        [
            _blank_observation(header, band="g"),
            _blank_observation(header, band="i"),
        ],
    )
    fit_scene = SourceCatalogScene(
        lon,
        lat,
        coefficients=torch.zeros((1, 2), dtype=torch.float64),
        band_labels=("g", "r", "i"),
        spectral_basis=basis,
    )

    recovered = fit_forced_photometry(observations, fit_scene, background="none")
    torch.testing.assert_close(
        recovered.flux, torch.tensor([[5.0, 1.25]], dtype=torch.float64)
    )

    per_band_scene = SourceCatalogScene(
        lon,
        lat,
        coefficients=torch.zeros((1, 3), dtype=torch.float64),
        band_labels=("g", "r", "i"),
    )
    missing = fit_forced_photometry(observations[:1], per_band_scene, background="none")
    assert torch.isfinite(missing.flux[0, 0])
    assert torch.isnan(missing.flux[0, 1:]).all()
    assert missing.diagnostics[0].gauge_parameter_indices.tolist() == [1, 2]


def test_local_background_column_and_rank_gauge_diagnostics() -> None:
    header = _header()
    lon, lat = _positions(header, [(12.0, 12.0)])
    truth = SourceCatalogScene(
        lon,
        lat,
        coefficients=torch.tensor([[3.0]], dtype=torch.float64),
    )
    blank = _blank_observation(header)
    source = truth.render_observation(blank)
    stamp = truth.render_component(0, blank)
    assert stamp is not None
    image = source.clone()
    image[stamp.y_slice, stamp.x_slice] += 0.4
    observation = replace(blank, image=image)
    fit_scene = SourceCatalogScene(
        lon,
        lat,
        coefficients=torch.zeros((1, 1), dtype=torch.float64),
    )

    result = fit_forced_photometry([observation], fit_scene, background="constant")

    torch.testing.assert_close(
        result.coefficients, torch.tensor([3.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        result.background_coefficients[0], torch.tensor([0.4], dtype=torch.float64)
    )
    torch.testing.assert_close(result.predictions[0], image)

    coincident = SourceCatalogScene(
        lon.repeat(2),
        lat.repeat(2),
        coefficients=torch.zeros((2, 1), dtype=torch.float64),
    )
    degenerate = fit_forced_photometry([observation], coincident, background="none")
    assert degenerate.diagnostics[0].rank_deficient
    assert degenerate.diagnostics[0].gauge_parameter_indices.tolist() == [0, 1]
    assert torch.isnan(degenerate.covariance.to_dense()).all()


def test_nnls_active_system_is_nonnegative_and_reports_conditional_covariance() -> None:
    header = _header()
    lon, lat = _positions(header, [(11.0, 12.0), (13.0, 12.0)])
    generating = SourceCatalogScene(
        lon,
        lat,
        coefficients=torch.tensor([[2.0], [-0.8]], dtype=torch.float64),
    )
    blank = _blank_observation(header, psf_sigma=1.0)
    observation = replace(blank, image=generating.render_observation(blank))
    fit_scene = SourceCatalogScene(
        lon,
        lat,
        coefficients=torch.zeros((2, 1), dtype=torch.float64),
    )

    signed = fit_forced_photometry(
        [observation], fit_scene, solver="signed", background="none"
    )
    constrained = fit_forced_photometry(
        [observation], fit_scene, solver="nnls", background="none"
    )

    torch.testing.assert_close(
        signed.coefficients, torch.tensor([2.0, -0.8], dtype=torch.float64)
    )
    assert bool((constrained.coefficients >= 0).all())
    assert constrained.coefficients[1] == 0
    assert constrained.covariance.constrained
    assert torch.isfinite(constrained.covariance.diagonal()[0])
    assert torch.isnan(constrained.covariance.diagonal()[1])


def test_fixed_template_chunking_is_numerically_identical() -> None:
    header = _header(size=31)
    lon, lat = _positions(header, [(15.2, 14.7)])
    template = torch.arange(1, 82, dtype=torch.float64).reshape(1, 9, 9)
    scene = SourceCatalogScene(
        lon,
        lat,
        morphology="template",
        fixed_templates=template,
        template_pixel_scale_arcsec=0.2,
        coefficients=torch.tensor([[2.5]], dtype=torch.float64),
    )
    blank = _blank_observation(header, psf_sigma=0.9)

    unchunked = scene.render_observation(blank)
    chunked = scene.render_observation(blank, chunk_size=7)
    observation = replace(blank, image=unchunked)
    fit = fit_forced_photometry(
        [observation],
        scene,
        background="none",
        chunk_size=5,
    )

    torch.testing.assert_close(chunked, unchunked)
    torch.testing.assert_close(
        fit.coefficients, torch.tensor([2.5], dtype=torch.float64)
    )


def test_psf_and_wcs_mismatch_leave_larger_native_residuals() -> None:
    header = _header()
    lon, lat = _positions(header, [(12.3, 11.8)])
    scene = SourceCatalogScene(
        lon,
        lat,
        coefficients=torch.tensor([[5.0]], dtype=torch.float64),
    )
    truth_observation = _blank_observation(header, psf_sigma=0.8)
    image = scene.render_observation(truth_observation)
    data = replace(truth_observation, image=image)

    matched = fit_forced_photometry([data], scene, background="none")
    psf_mismatch = fit_forced_photometry(
        [replace(data, psf_spec=PSFSpec.gaussian(1.5))],
        scene,
        background="none",
    )
    shifted_data = replace(data, header=_header(crpix_shift=0.35))
    wcs_mismatch = fit_forced_photometry([shifted_data], scene, background="none")

    matched_norm = torch.linalg.vector_norm(matched.residuals[0])
    assert torch.linalg.vector_norm(psf_mismatch.residuals[0]) > matched_norm + 1e-5
    assert torch.linalg.vector_norm(wcs_mismatch.residuals[0]) > matched_norm + 1e-5


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason="MPS unavailable"
            ),
        ),
    ],
)
def test_signed_fit_preserves_device_and_image_gradient(device: str) -> None:
    header = _header(size=17)
    lon, lat = _positions(header, [(8.0, 8.0)])
    dtype = torch.float32 if device == "mps" else torch.float64
    scene = SourceCatalogScene(
        lon.to(device=device, dtype=dtype),
        lat.to(device=device, dtype=dtype),
        coefficients=torch.zeros((1, 1), dtype=dtype, device=device),
    )
    blank = _blank_observation(header, dtype=dtype, device=device)
    template = scene.render_observation(
        blank, parameter_values=torch.ones(1, dtype=dtype, device=device)
    )
    image = (2.0 * template).detach().requires_grad_(True)
    observation = replace(blank, image=image, weight=torch.ones_like(image))

    result = fit_forced_photometry([observation], scene, background="none")
    loss = result.coefficients.sum()
    gradient = torch.autograd.grad(loss, image)[0]
    expected = template / torch.sum(template.square())

    assert result.coefficients.device.type == device
    torch.testing.assert_close(
        result.coefficients,
        torch.tensor([2.0], dtype=dtype, device=device),
        rtol=2e-4,
        atol=2e-5,
    )
    torch.testing.assert_close(gradient, expected, rtol=2e-4, atol=2e-5)


def test_separated_catalog_scales_as_single_source_blocks() -> None:
    header = _header(size=129)
    pixels = [(8.0 + 15.0 * index, 64.0) for index in range(8)]
    lon, lat = _positions(header, pixels)
    scene = SourceCatalogScene(
        lon,
        lat,
        coefficients=torch.arange(1, 9, dtype=torch.float64)[:, None],
    )
    blank = _blank_observation(header, psf_sigma=0.6)
    observation = replace(blank, image=scene.render_observation(blank))
    fit_scene = SourceCatalogScene(
        lon,
        lat,
        coefficients=torch.zeros((8, 1), dtype=torch.float64),
    )

    result = fit_forced_photometry([observation], fit_scene, background="none")

    assert len(result.grouping.blend_groups) == 8
    assert all(group.numel() == 1 for group in result.grouping.blend_groups)
    assert all(block.shape == (1, 1) for block in result.covariance.blocks)
    torch.testing.assert_close(
        result.coefficients, torch.arange(1, 9, dtype=torch.float64)
    )
    assert result.provenance["selection"] == "caller-supplied-known-sources"
