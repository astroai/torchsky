from __future__ import annotations

import pytest
import torch

from torchsky.wcs import (
    AstrometricMatchGraph,
    AstrometricObservation,
    AstrometryConfig,
    CoupledCalibrationConfig,
    CoupledCalibrationGraph,
    PhotometricMeasurementBlock,
    PhotometricReference,
    TangentPlaneFrame,
    photometric_log_residuals,
)
from torchsky.wcs.calibration import _PhotometrySolver


DTYPE = torch.float64


def _world(frame: TangentPlaneFrame, plane: torch.Tensor) -> torch.Tensor:
    ra, dec = frame.plane_to_world(plane[:, 0] / 3600.0, plane[:, 1] / 3600.0)
    return torch.stack((ra, dec), dim=-1)


def _coupled_problem(
    *,
    outlier: bool = False,
    variable: bool = False,
) -> tuple[
    CoupledCalibrationGraph,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    frame = TangentPlaneFrame(35.0, -12.0)
    source_ids = torch.arange(10)
    source = torch.stack(
        (
            torch.linspace(-18.0, 21.0, 10, dtype=DTYPE),
            torch.tensor(
                [-9.0, 6.0, 13.0, -15.0, 3.0, 18.0, -4.0, 9.0, -17.0, 1.0], dtype=DTYPE
            ),
        ),
        dim=-1,
    )
    log_flux = torch.stack(
        (
            torch.linspace(3.2, 4.1, 10, dtype=DTYPE),
            torch.linspace(3.8, 3.35, 10, dtype=DTYPE),
        ),
        dim=-1,
    )
    color_scale = -2.5 / torch.log(torch.tensor(10.0, dtype=DTYPE))
    color = color_scale * (log_flux[:, 0] - log_flux[:, 1])
    shifts = torch.tensor(
        [[0.4, -0.3], [-0.8, 0.5], [0.2, 0.7], [0.9, -0.4]], dtype=DTYPE
    )
    dcr = torch.tensor([0.14, -0.09, 0.19, -0.12], dtype=DTYPE)
    dcr_basis = torch.tensor(
        [[0.6, 0.8], [-0.8, 0.6], [0.3, 0.9539392], [-0.5, 0.8660254]],
        dtype=DTYPE,
    )
    lateral = torch.tensor([0.045, -0.035], dtype=DTYPE)
    lateral_basis = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=DTYPE)
    exposure_ids = ("g0", "r0", "g1", "r1")
    detector_ids = ("ccd0", "ccd1", "ccd0", "ccd1")
    observations = []
    for exposure, (exposure_id, detector_id) in enumerate(
        zip(exposure_ids, detector_ids, strict=True)
    ):
        detector = 0 if detector_id == "ccd0" else 1
        observed = (
            source
            - shifts[exposure]
            - dcr[exposure] * color[:, None] * dcr_basis[exposure]
            - lateral[detector] * color[:, None] * lateral_basis[detector]
        )
        observations.append(
            AstrometricObservation(
                exposure_id,
                observed,
                source_ids,
                torch.eye(2, dtype=DTYPE).expand(10, 2, 2) * 0.015**2,
                sky_position=_world(frame, observed),
                covariance_space="tangent",
                detector_id=detector_id,
                epoch=2020.0 + exposure,
                dcr_basis=dcr_basis[exposure],
                lateral_color_basis=lateral_basis[detector],
            )
        )
    astrometry = AstrometricMatchGraph(
        observations,
        reference_source_ids=source_ids,
        reference_positions=_world(frame, source),
        reference_covariance=torch.eye(2, dtype=DTYPE) * 1.0e-8,
        tangent_frame=frame,
    )
    zeropoint = torch.tensor([0.12, -0.08, 0.04, 0.17], dtype=DTYPE)
    extinction = torch.tensor([0.025, -0.018, 0.031, -0.022], dtype=DTYPE)
    cloud = torch.tensor([-0.015, 0.012, 0.008, -0.01], dtype=DTYPE)
    star_flat = torch.tensor([0.021, -0.017], dtype=DTYPE)
    color_term = torch.tensor([0.028, -0.019], dtype=DTYPE)
    x = torch.linspace(-1.0, 1.0, 10, dtype=DTYPE)
    extinction_basis = torch.tensor(
        [-0.8, 0.3, 0.9, -0.4, 0.6, -0.7, 0.1, 0.75, -0.2, -0.55],
        dtype=DTYPE,
    )
    extinction_basis -= extinction_basis.mean()
    cloud_basis = x.square() - x.square().mean()
    flat_basis = (x**3 - (x**3).mean()).unsqueeze(-1)
    blocks = []
    for exposure, (exposure_id, detector_id) in enumerate(
        zip(exposure_ids, detector_ids, strict=True)
    ):
        band = exposure % 2
        detector = 0 if detector_id == "ccd0" else 1
        model = (
            log_flux[:, band]
            + zeropoint[exposure]
            + extinction[exposure] * extinction_basis
            + cloud[exposure] * cloud_basis
            + star_flat[detector] * flat_basis[:, 0]
            + color_term[band] * color
        )
        if variable and exposure == 2:
            model = model.clone()
            model[0] += 0.7
        if outlier and exposure == 3:
            model = model.clone()
            model[-1] += 1.5
        flux = torch.exp(model)
        sigma = 0.006 * flux
        covariance = torch.diag(sigma.square()) + 1.0e-6 * torch.outer(flux, flux)
        blocks.append(
            PhotometricMeasurementBlock(
                exposure_id,
                source_ids,
                ("g", "r")[band],
                flux,
                covariance,
                detector_id=detector_id,
                extinction_basis=extinction_basis,
                cloud_basis=cloud_basis,
                star_flat_basis=flat_basis,
            )
        )
    reference = PhotometricReference(
        source_ids,
        ("g", "r"),
        torch.exp(log_flux),
        torch.exp(2.0 * log_flux) * 1.0e-8,
    )
    return (
        CoupledCalibrationGraph(astrometry, blocks, reference=reference),
        log_flux,
        shifts,
        dcr,
        lateral,
        zeropoint,
    )


def _config(**kwargs: object) -> CoupledCalibrationConfig:
    values: dict[str, object] = {
        "astrometry": AstrometryConfig(
            pose_model="shift",
            fit_dcr=True,
            fit_lateral_color=True,
            rank_policy="drop",
            huber_delta=5.0,
            convergence_tolerance=1.0e-10,
        ),
        "color_bands": ("g", "r"),
        "max_iterations": 4,
        "photometry_max_iterations": 40,
        "convergence_tolerance": 1.0e-8,
        "photometry_convergence_tolerance": 1.0e-9,
        "huber_delta": 4.0,
    }
    values.update(kwargs)
    return CoupledCalibrationConfig(**values)


def test_dense_reference_recovers_photometry_color_dcr_and_lateral_color() -> None:
    graph, log_flux, shifts, dcr, lateral, zeropoint = _coupled_problem()
    solution = graph.solve(_config())

    torch.testing.assert_close(
        solution.photometry.source_band_log_flux, log_flux, atol=2.0e-3, rtol=0
    )
    torch.testing.assert_close(
        solution.photometry.exposure_log_zeropoint, zeropoint, atol=3.0e-3, rtol=0
    )
    torch.testing.assert_close(
        solution.photometry.exposure_extinction,
        torch.tensor([0.025, -0.018, 0.031, -0.022], dtype=DTYPE),
        atol=3.0e-3,
        rtol=0,
    )
    torch.testing.assert_close(
        solution.photometry.exposure_cloud,
        torch.tensor([-0.015, 0.012, 0.008, -0.01], dtype=DTYPE),
        atol=3.0e-3,
        rtol=0,
    )
    torch.testing.assert_close(
        solution.photometry.star_flat_coefficients[:, 0],
        torch.tensor([0.021, -0.017], dtype=DTYPE),
        atol=3.0e-3,
        rtol=0,
    )
    torch.testing.assert_close(
        solution.photometry.band_color_terms,
        torch.tensor([0.028, -0.019], dtype=DTYPE),
        atol=3.0e-3,
        rtol=0,
    )
    torch.testing.assert_close(
        solution.astrometry.pose_parameters, shifts, atol=3.0e-5, rtol=0
    )
    assert solution.astrometry.dcr_coefficients is not None
    assert solution.astrometry.lateral_color_coefficients is not None
    torch.testing.assert_close(
        solution.astrometry.dcr_coefficients, dcr, atol=3.0e-5, rtol=0
    )
    torch.testing.assert_close(
        solution.astrometry.lateral_color_coefficients,
        lateral,
        atol=3.0e-5,
        rtol=0,
    )
    assert solution.photometry.global_covariance is not None
    covariance = solution.photometry.global_covariance
    torch.testing.assert_close(covariance, covariance.mT)
    assert bool((torch.diagonal(covariance) > 0).all())
    assert bool(torch.isfinite(solution.photometry.source_coefficient_variance).all())
    assert solution.converged
    assert solution.provenance["detection"] is False
    assert solution.provenance["associations"] == "caller-supplied-immutable"


def test_variable_sources_and_outliers_do_not_bias_global_or_astrometry() -> None:
    graph, _, shifts, _, _, zeropoint = _coupled_problem(outlier=True, variable=True)
    robust = graph.solve(
        _config(
            fit_variability=True,
            variable_source_ids=(0,),
            huber_delta=2.0,
        )
    )

    assert robust.photometry.variability_offsets[20] > 0.5
    assert robust.photometry.robust_weights[-1] < 0.05
    torch.testing.assert_close(
        robust.photometry.exposure_log_zeropoint, zeropoint, atol=8.0e-3, rtol=0
    )
    torch.testing.assert_close(
        robust.astrometry.pose_parameters, shifts, atol=4.0e-5, rtol=0
    )
    assert robust.provenance["variability_enters_astrometry"] is False


def test_missing_band_holdouts_disconnected_gauges_and_association_rejection() -> None:
    frame = TangentPlaneFrame(120.0, 20.0)
    source_ids = torch.tensor([0, 1, 10, 11])
    source = torch.tensor(
        [[-4.0, 1.0], [2.0, 5.0], [30.0, -3.0], [36.0, 7.0]], dtype=DTYPE
    )
    observations = []
    for exposure_id, selected in (("a", slice(0, 2)), ("b", slice(2, 4))):
        observations.append(
            AstrometricObservation(
                exposure_id,
                source[selected],
                source_ids[selected],
                torch.eye(2, dtype=DTYPE).expand(2, 2, 2) * 0.01**2,
                sky_position=_world(frame, source[selected]),
                covariance_space="tangent",
            )
        )
    astrometry = AstrometricMatchGraph(observations, tangent_frame=frame)
    blocks = [
        PhotometricMeasurementBlock(
            "a",
            source_ids[:2],
            "g",
            torch.tensor([10.0, 14.0], dtype=DTYPE),
            torch.full((2,), 0.01, dtype=DTYPE),
        ),
        PhotometricMeasurementBlock(
            "b",
            source_ids[2:],
            "r",
            torch.tensor([20.0, 24.0], dtype=DTYPE),
            torch.full((2,), 0.01, dtype=DTYPE),
        ),
    ]
    graph = CoupledCalibrationGraph(astrometry, blocks)
    solution = graph.solve(
        CoupledCalibrationConfig(
            astrometry=AstrometryConfig(pose_model="shift"),
            color_bands=("g", "r"),
            missing_color="drop",
            holdout_source_ids=(11,),
            holdout_exposure_ids=("a",),
            fit_extinction=False,
            fit_cloud=False,
            fit_star_flat=False,
            fit_color_terms=False,
        )
    )

    assert torch.isnan(solution.photometry.colors).all()
    assert torch.isnan(solution.photometry.source_band_flux[-1]).all()
    assert solution.photometry.diagnostics.gauge_exposures == ("b",)
    assert solution.diagnostics.source_holdout_count == 1
    assert solution.diagnostics.exposure_holdout_count == 1
    with pytest.raises(ValueError, match="association"):
        CoupledCalibrationGraph(
            astrometry,
            [
                PhotometricMeasurementBlock(
                    "a",
                    torch.tensor([10]),
                    "g",
                    torch.tensor([2.0], dtype=DTYPE),
                    torch.tensor([0.01], dtype=DTYPE),
                )
            ],
        )


def test_photometric_components_match_order_under_many_row_permutations() -> None:
    frame = TangentPlaneFrame(120.0, 20.0)
    component_count = 48
    source_count = 3 * component_count
    source_ids = torch.arange(source_count)
    source = torch.stack(
        (
            torch.arange(source_count, dtype=DTYPE),
            torch.arange(source_count, dtype=DTYPE).remainder(11),
        ),
        dim=-1,
    )
    permutation = torch.tensor([2, 0, 1])

    def make_graph(permute: bool) -> CoupledCalibrationGraph:
        observations = []
        blocks = []
        for component in range(component_count):
            selected = source_ids[3 * component : 3 * component + 3]
            if permute:
                selected = selected[permutation]
            band = ("g", "r")[component % 2]
            for suffix in ("a", "b"):
                exposure_id = f"c{component}{suffix}"
                observations.append(
                    AstrometricObservation(
                        exposure_id,
                        source[selected],
                        selected,
                        torch.eye(2, dtype=DTYPE).expand(3, 2, 2) * 0.01**2,
                        sky_position=_world(frame, source[selected]),
                        covariance_space="tangent",
                    )
                )
                blocks.append(
                    PhotometricMeasurementBlock(
                        exposure_id,
                        selected,
                        band,
                        torch.full((3,), 10.0 + component, dtype=DTYPE),
                        torch.full((3,), 0.01, dtype=DTYPE),
                    )
                )
        astrometry = AstrometricMatchGraph(observations, tangent_frame=frame)
        referenced = source_ids[::9]
        reference_flux = torch.full((referenced.numel(), 2), float("nan"), dtype=DTYPE)
        reference_variance = torch.full_like(reference_flux, float("nan"))
        for row, component in enumerate(range(0, component_count, 3)):
            band = component % 2
            reference_flux[row, band] = 10.0 + component
            reference_variance[row, band] = 0.01
        reference = PhotometricReference(
            referenced,
            ("g", "r"),
            reference_flux,
            reference_variance,
        )
        return CoupledCalibrationGraph(astrometry, blocks, reference=reference)

    config = CoupledCalibrationConfig(
        fit_extinction=False,
        fit_cloud=False,
        fit_star_flat=False,
        fit_color_terms=False,
    )
    baseline = _PhotometrySolver(make_graph(False), config)
    permuted = _PhotometrySolver(make_graph(True), config)
    expected_exposures = tuple(
        (2 * component, 2 * component + 1) for component in range(component_count)
    )
    expected_components = tuple(
        tuple(
            2 * source_index + component % 2
            for source_index in range(3 * component, 3 * component + 3)
        )
        for component in range(component_count)
    )
    expected_anchored = tuple(
        component % 3 == 0 for component in range(component_count)
    )

    assert baseline.components == expected_components
    assert baseline.component_exposures == expected_exposures
    assert baseline.component_anchored == expected_anchored
    assert baseline.gauge_exposure_indices == tuple(
        exposures[0]
        for exposures, anchored in zip(
            expected_exposures, expected_anchored, strict=True
        )
        if not anchored
    )
    assert permuted.components == baseline.components
    assert permuted.component_exposures == baseline.component_exposures
    assert permuted.component_anchored == baseline.component_anchored
    assert permuted.gauge_exposure_indices == baseline.gauge_exposure_indices


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_full_covariance_residual_gradient_dtype_and_scaling_structure(
    dtype: torch.dtype,
) -> None:
    measured = torch.tensor([2.0, 3.0], dtype=dtype, requires_grad=True)
    predicted = torch.tensor([1.8, 3.2], dtype=dtype, requires_grad=True)
    covariance = torch.tensor([[0.04, 0.01], [0.01, 0.09]], dtype=dtype)
    residual = photometric_log_residuals(measured, predicted, covariance)
    residual.square().sum().backward()

    assert measured.grad is not None
    assert predicted.grad is not None
    assert residual.dtype == dtype
    graph, _, _, _, _, _ = _coupled_problem()
    solution = graph.solve(_config(covariance_max_parameters=1))
    assert solution.photometry.global_covariance is None
    assert solution.photometry.diagnostics.matrix_free
    assert not solution.photometry.diagnostics.assembled_measurement_matrix
    assert solution.astrometry.diagnostics.matrix_free_reduced


@pytest.mark.parametrize(
    "device",
    ["cpu"]
    + (["cuda"] if torch.cuda.is_available() else [])
    + (["mps"] if torch.backends.mps.is_available() else []),
)
def test_coupled_solver_preserves_float32_device(device: str) -> None:
    dtype = torch.float32
    frame = TangentPlaneFrame(10.0, 5.0)
    source_ids = torch.arange(3, device=device)
    plane = torch.tensor(
        [[-2.0, 1.0], [1.0, 3.0], [4.0, -1.0]],
        dtype=dtype,
        device=device,
    )
    observation = AstrometricObservation(
        "e",
        plane,
        source_ids,
        torch.eye(2, dtype=dtype, device=device).expand(3, 2, 2) * 0.01,
        sky_position=_world(frame, plane),
        covariance_space="tangent",
    )
    astrometry = AstrometricMatchGraph([observation], tangent_frame=frame)
    flux = torch.tensor([3.0, 5.0, 7.0], dtype=dtype, device=device)
    block = PhotometricMeasurementBlock(
        "e", source_ids, "g", flux, torch.full_like(flux, 0.01)
    )
    solution = CoupledCalibrationGraph(astrometry, [block]).solve(
        CoupledCalibrationConfig(
            astrometry=AstrometryConfig(pose_model="shift"),
            fit_extinction=False,
            fit_cloud=False,
            fit_star_flat=False,
            fit_color_terms=False,
            max_iterations=2,
            photometry_max_iterations=8,
            convergence_tolerance=1.0e-5,
            photometry_convergence_tolerance=1.0e-5,
            cg_rtol=1.0e-5,
        )
    )

    assert solution.photometry.source_band_flux.dtype == dtype
    assert solution.photometry.source_band_flux.device.type == device
    torch.testing.assert_close(
        solution.photometry.source_band_flux[:, 0], flux, atol=2.0e-4, rtol=2.0e-4
    )


def test_compact_sed_basis_and_proper_motion_color_confounding_are_separated() -> None:
    graph, log_flux, _, _, _, _ = _coupled_problem()
    mean = log_flux.mean(dim=1)
    slope = 0.5 * (log_flux[:, 0] - log_flux[:, 1])
    sed_basis = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=DTYPE)
    compact = CoupledCalibrationGraph(
        graph.astrometry_graph,
        graph.blocks,
        reference=graph.reference,
        sed_basis=sed_basis,
    )
    solution = compact.solve(
        _config(
            astrometry=AstrometryConfig(
                pose_model="shift",
                fit_dcr=True,
                fit_lateral_color=True,
                fit_proper_motion=True,
                reference_epoch=2020.0,
                rank_policy="drop",
                convergence_tolerance=1.0e-10,
            )
        )
    )

    torch.testing.assert_close(
        solution.photometry.source_band_log_flux, log_flux, atol=2.0e-3, rtol=0
    )
    recovered = torch.stack((mean, slope), dim=-1)
    inferred = solution.photometry.source_band_log_flux @ torch.linalg.inv(sed_basis).mT
    torch.testing.assert_close(inferred, recovered, atol=2.0e-3, rtol=0)
    assert solution.astrometry.proper_motion_mas_per_year is not None
    assert (
        torch.linalg.vector_norm(solution.astrometry.proper_motion_mas_per_year)
        < 1.0e-2
    )
