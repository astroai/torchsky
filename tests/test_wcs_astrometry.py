from __future__ import annotations

import math

import pytest
import torch

from torchsky.wcs import (
    AstrometricMatchGraph,
    AstrometricObservation,
    AstrometryConfig,
    TangentPlaneFrame,
    match_approximate_wcs,
    tangent_plane_block_residuals,
)
from torchsky.wcs.astrometry import _AstrometrySolver


DTYPE = torch.float64
DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")
if torch.backends.mps.is_available():
    DEVICES.append("mps")


def _world(frame: TangentPlaneFrame, plane_arcsec: torch.Tensor) -> torch.Tensor:
    ra, dec = frame.plane_to_world(
        plane_arcsec[:, 0] / 3600.0, plane_arcsec[:, 1] / 3600.0
    )
    return torch.stack((ra, dec), dim=-1)


def _observation(
    exposure_id: str,
    frame: TangentPlaneFrame,
    observed_plane: torch.Tensor,
    source_ids: torch.Tensor,
    *,
    sigma: float = 0.02,
    epoch: float | None = None,
    color: torch.Tensor | None = None,
    dcr_basis: torch.Tensor | None = None,
    parallax_factor: torch.Tensor | None = None,
    detector_xy: torch.Tensor | None = None,
) -> AstrometricObservation:
    if detector_xy is None:
        detector_xy = observed_plane
    covariance = torch.eye(2, dtype=DTYPE).expand(source_ids.numel(), 2, 2)
    return AstrometricObservation(
        exposure_id,
        detector_xy,
        source_ids,
        covariance * sigma**2,
        sky_position=_world(frame, observed_plane),
        covariance_space="tangent",
        epoch=epoch,
        color=color,
        dcr_basis=dcr_basis,
        parallax_factor=parallax_factor,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", DEVICES)
def test_tangent_plane_block_residuals_use_full_covariance_and_autograd(
    dtype: torch.dtype, device: str
) -> None:
    if device == "mps" and dtype == torch.float64:
        pytest.skip("MPS does not support float64")
    observed = torch.tensor(
        [[2.0, -1.0]], dtype=dtype, device=device, requires_grad=True
    )
    predicted = torch.tensor(
        [[1.0, 1.0]], dtype=dtype, device=device, requires_grad=True
    )
    centroid = torch.tensor([[[4.0, 1.0], [1.0, 3.0]]], dtype=dtype, device=device)
    reference = torch.tensor([[[1.0, -0.2], [-0.2, 2.0]]], dtype=dtype, device=device)

    residual = tangent_plane_block_residuals(observed, predicted, centroid, reference)
    expected = torch.linalg.solve_triangular(
        torch.linalg.cholesky(centroid + reference),
        (observed - predicted).unsqueeze(-1),
        upper=False,
    ).squeeze(-1)
    torch.testing.assert_close(residual, expected)
    residual.square().sum().backward()
    assert observed.grad is not None
    assert predicted.grad is not None
    assert residual.dtype == observed.dtype
    assert residual.device == observed.device


def test_global_shift_fit_dense_reference_and_small_block_covariance() -> None:
    frame = TangentPlaneFrame(42.0, -18.0)
    generator = torch.Generator().manual_seed(4)
    sources = 30.0 * torch.randn((48, 2), generator=generator, dtype=DTYPE)
    source_ids = torch.arange(sources.shape[0])
    shifts = torch.tensor([[0.8, -0.4], [-1.2, 0.6], [0.3, 1.1]], dtype=DTYPE)
    observations = [
        _observation(f"e{index}", frame, sources - shift, source_ids)
        for index, shift in enumerate(shifts)
    ]
    reference_covariance = torch.tensor(
        [[[0.01**2, 2.0e-5], [2.0e-5, 0.015**2]]], dtype=DTYPE
    ).expand(sources.shape[0], 2, 2)
    graph = AstrometricMatchGraph(
        observations,
        reference_source_ids=source_ids,
        reference_positions=_world(frame, sources),
        reference_covariance=reference_covariance,
        tangent_frame=frame,
    )

    solution = graph.solve(
        AstrometryConfig(
            pose_model="shift",
            huber_delta=10.0,
            convergence_tolerance=1.0e-10,
        )
    )

    torch.testing.assert_close(solution.pose_parameters, shifts, atol=2.0e-5, rtol=0)
    torch.testing.assert_close(
        solution.source_tangent_positions, sources, atol=2.0e-5, rtol=0
    )
    assert solution.converged
    assert solution.diagnostics.matrix_free_reduced
    assert not solution.diagnostics.assembled_observation_matrix
    assert solution.diagnostics.connected_components == (("e0", "e1", "e2"),)
    assert solution.diagnostics.gauge_exposures == ()
    assert solution.diagnostics.reduced_rank == solution.diagnostics.reduced_dimension
    assert set(solution.exposure_covariance) == {"e0", "e1", "e2"}
    assert all(value.shape == (2, 2) for value in solution.exposure_covariance.values())


def test_rscale_pose_and_huber_irls_refit_outlier() -> None:
    frame = TangentPlaneFrame(120.0, 35.0)
    generator = torch.Generator().manual_seed(9)
    sources = 50.0 * torch.randn((36, 2), generator=generator, dtype=DTYPE)
    source_ids = torch.arange(sources.shape[0])
    theta = 4.0e-3
    log_scale = -2.0e-3
    shift = torch.tensor([0.7, -1.1], dtype=DTYPE)
    rotation = torch.tensor(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
        dtype=DTYPE,
    )
    observed = ((sources - shift) @ rotation) / math.exp(log_scale)
    observed[0] += torch.tensor([20.0, -15.0], dtype=DTYPE)
    observation = _observation("exp", frame, observed, source_ids, sigma=0.05)
    graph = AstrometricMatchGraph(
        [observation],
        reference_source_ids=source_ids,
        reference_positions=_world(frame, sources),
        reference_covariance=torch.eye(2, dtype=DTYPE) * 0.01**2,
        tangent_frame=frame,
    )

    robust = graph.solve(
        AstrometryConfig(
            pose_model="rscale",
            huber_delta=2.5,
            convergence_tolerance=1.0e-10,
            max_iterations=20,
        )
    )
    nonrobust = graph.solve(
        AstrometryConfig(
            pose_model="rscale",
            huber_delta=1.0e9,
            convergence_tolerance=1.0e-10,
            max_iterations=20,
        )
    )
    truth = torch.tensor([shift[0], shift[1], theta, log_scale], dtype=DTYPE)
    robust_error = torch.linalg.vector_norm(robust.pose_parameters[0] - truth)
    nonrobust_error = torch.linalg.vector_norm(nonrobust.pose_parameters[0] - truth)
    assert robust_error < nonrobust_error * 0.1
    assert robust.robust_weights[0] < 0.02
    assert robust.diagnostics.iterations > 1


def test_disconnected_components_receive_explicit_exposure_gauges() -> None:
    frame = TangentPlaneFrame(250.0, 15.0)
    source_a = torch.tensor([[-4.0, 2.0], [3.0, 7.0]], dtype=DTYPE)
    source_b = torch.tensor([[30.0, -2.0], [35.0, 8.0]], dtype=DTYPE)
    observations = [
        _observation("a0", frame, source_a, torch.tensor([0, 1])),
        _observation("a1", frame, source_a - 1.0, torch.tensor([0, 1])),
        _observation("b0", frame, source_b, torch.tensor([10, 11])),
        _observation("b1", frame, source_b + 2.0, torch.tensor([10, 11])),
    ]
    solution = AstrometricMatchGraph(observations, tangent_frame=frame).solve(
        AstrometryConfig(pose_model="shift")
    )

    assert solution.diagnostics.connected_components == (("a0", "a1"), ("b0", "b1"))
    assert solution.diagnostics.gauge_exposures == ("a0", "b0")
    torch.testing.assert_close(solution.pose_for("a0"), torch.zeros(2, dtype=DTYPE))
    torch.testing.assert_close(solution.pose_for("b0"), torch.zeros(2, dtype=DTYPE))


def test_component_materialization_matches_order_under_many_row_permutations() -> None:
    frame = TangentPlaneFrame(250.0, 15.0)
    component_count = 48
    source_count = 3 * component_count
    source = torch.stack(
        (
            torch.arange(source_count, dtype=DTYPE),
            torch.arange(source_count, dtype=DTYPE).remainder(7),
        ),
        dim=-1,
    )
    permutation = torch.tensor([2, 0, 1])

    def make_graph(permute: bool) -> AstrometricMatchGraph:
        observations = []
        for component in range(component_count):
            selected = torch.arange(3 * component, 3 * component + 3)
            if permute:
                selected = selected[permutation]
            for suffix in ("a", "b"):
                observations.append(
                    _observation(
                        f"c{component}{suffix}",
                        frame,
                        source[selected],
                        selected,
                    )
                )
        referenced = torch.arange(0, source_count, 9)
        return AstrometricMatchGraph(
            observations,
            reference_source_ids=referenced,
            reference_positions=_world(frame, source[referenced]),
            reference_covariance=torch.eye(2, dtype=DTYPE) * 0.01**2,
            tangent_frame=frame,
        )

    config = AstrometryConfig(pose_model="shift")
    baseline = _AstrometrySolver(make_graph(False), config)
    permuted = _AstrometrySolver(make_graph(True), config)
    expected_exposures = tuple(
        (2 * component, 2 * component + 1) for component in range(component_count)
    )
    expected_components = tuple(
        exposures
        + tuple(
            2 * component_count + source_index
            for source_index in range(3 * component, 3 * component + 3)
        )
        for component, exposures in enumerate(expected_exposures)
    )
    expected_anchored = tuple(
        component % 3 == 0 for component in range(component_count)
    )

    assert baseline.fit_components == expected_components
    assert baseline.fit_component_exposures == expected_exposures
    assert baseline.fit_component_reference_anchored == expected_anchored
    assert baseline.gauge_exposure_indices == tuple(
        exposures[0]
        for exposures, anchored in zip(
            expected_exposures, expected_anchored, strict=True
        )
        if not anchored
    )
    assert permuted.fit_components == baseline.fit_components
    assert permuted.fit_component_exposures == baseline.fit_component_exposures
    assert (
        permuted.fit_component_reference_anchored
        == baseline.fit_component_reference_anchored
    )
    assert permuted.gauge_exposure_indices == baseline.gauge_exposure_indices


def test_motion_rank_policy_and_identifiable_pm_parallax() -> None:
    frame = TangentPlaneFrame(5.0, 70.0)
    source_ids = torch.arange(6)
    source = torch.stack(
        (torch.linspace(-10.0, 10.0, 6), torch.linspace(5.0, -7.0, 6)), dim=-1
    )
    one_epoch = _observation("only", frame, source, source_ids, epoch=2020.0)
    graph = AstrometricMatchGraph([one_epoch], tangent_frame=frame)
    with pytest.raises(ValueError, match="not identifiable"):
        graph.solve(AstrometryConfig(fit_proper_motion=True, rank_policy="error"))
    dropped = graph.solve(AstrometryConfig(fit_proper_motion=True, rank_policy="drop"))
    assert dropped.diagnostics.dropped_source_blocks == tuple(range(6))

    proper_motion = torch.tensor(
        [
            [0.012, -0.006],
            [0.008, 0.003],
            [-0.004, 0.009],
            [0.0, -0.01],
            [0.005, 0.004],
            [-0.007, -0.002],
        ],
        dtype=DTYPE,
    )
    parallax = torch.linspace(0.001, 0.006, 6, dtype=DTYPE)
    observations = []
    for index, (epoch, factor) in enumerate(
        [
            (2019.0, (1.0, 0.0)),
            (2020.0, (0.0, 1.0)),
            (2021.0, (-1.0, 0.0)),
            (2022.0, (0.0, -1.0)),
        ]
    ):
        dt = epoch - 2020.0
        factor_tensor = torch.tensor(factor, dtype=DTYPE).expand(6, 2)
        observed = source + dt * proper_motion + parallax[:, None] * factor_tensor
        observations.append(
            _observation(
                f"e{index}",
                frame,
                observed,
                source_ids,
                epoch=epoch,
                parallax_factor=factor_tensor,
            )
        )
    moving_graph = AstrometricMatchGraph(
        observations,
        reference_source_ids=source_ids,
        reference_positions=_world(frame, source),
        reference_covariance=torch.eye(2, dtype=DTYPE) * 1.0e-6,
        reference_proper_motion_mas_per_year=proper_motion * 1000.0,
        reference_proper_motion_covariance=torch.eye(2, dtype=DTYPE) * 1.0e-4,
        reference_parallax_mas=parallax * 1000.0,
        reference_parallax_variance=torch.full((6,), 1.0e-4, dtype=DTYPE),
        tangent_frame=frame,
    )
    moving = moving_graph.solve(
        AstrometryConfig(
            pose_model="shift",
            fit_proper_motion=True,
            fit_parallax=True,
            reference_epoch=2020.0,
            huber_delta=100.0,
            convergence_tolerance=1.0e-10,
        )
    )
    assert moving.proper_motion_mas_per_year is not None
    assert moving.parallax_mas is not None
    torch.testing.assert_close(
        moving.proper_motion_mas_per_year, proper_motion * 1000.0, atol=2.0e-3, rtol=0
    )
    torch.testing.assert_close(
        moving.parallax_mas, parallax * 1000.0, atol=2.0e-3, rtol=0
    )


def test_color_dcr_policy_and_fit() -> None:
    frame = TangentPlaneFrame(180.0, -5.0)
    sources = torch.tensor(
        [[-8.0, -3.0], [-2.0, 5.0], [4.0, -6.0], [9.0, 4.0]], dtype=DTYPE
    )
    source_ids = torch.arange(4)
    color = torch.tensor([-1.0, -0.2, 0.7, 1.5], dtype=DTYPE)
    basis = torch.tensor([0.6, 0.8], dtype=DTYPE).expand(4, 2)
    coefficient = 0.35
    observed = sources - coefficient * color[:, None] * basis
    observation = _observation(
        "dcr", frame, observed, source_ids, color=color, dcr_basis=basis
    )
    graph = AstrometricMatchGraph(
        [observation],
        reference_source_ids=source_ids,
        reference_positions=_world(frame, sources),
        reference_covariance=torch.eye(2, dtype=DTYPE) * 1.0e-6,
        tangent_frame=frame,
    )
    solution = graph.solve(
        AstrometryConfig(
            pose_model="shift",
            fit_dcr=True,
            color_pivot=0.0,
            convergence_tolerance=1.0e-10,
        )
    )
    assert solution.dcr_coefficients is not None
    torch.testing.assert_close(
        solution.dcr_coefficients,
        torch.tensor([coefficient], dtype=DTYPE),
        atol=2.0e-5,
        rtol=0,
    )

    missing = _observation("missing", frame, sources, source_ids, dcr_basis=basis)
    missing_graph = AstrometricMatchGraph([missing], tangent_frame=frame)
    with pytest.raises(ValueError, match="color is required"):
        missing_graph.solve(AstrometryConfig(fit_dcr=True, missing_color="error"))
    dropped = missing_graph.solve(
        AstrometryConfig(fit_dcr=True, missing_color="drop", rank_policy="drop")
    )
    assert dropped.diagnostics.dropped_dcr_exposures == ("missing",)


@pytest.mark.parametrize("center", [(359.999, 20.0), (45.0, 89.8)])
def test_ra_wrap_and_polar_tangent_plane_fit(center: tuple[float, float]) -> None:
    frame = TangentPlaneFrame(*center)
    sources = torch.tensor(
        [[-5.0, -3.0], [4.0, -2.0], [-2.0, 6.0], [7.0, 5.0]], dtype=DTYPE
    )
    source_ids = torch.arange(4)
    shift = torch.tensor([0.4, -0.25], dtype=DTYPE)
    graph = AstrometricMatchGraph(
        [_observation("exp", frame, sources - shift, source_ids)],
        reference_source_ids=source_ids,
        reference_positions=_world(frame, sources),
        reference_covariance=torch.eye(2, dtype=DTYPE) * 1.0e-6,
        tangent_frame=frame,
    )
    solution = graph.solve(AstrometryConfig(pose_model="shift"))
    torch.testing.assert_close(solution.pose_parameters[0], shift, atol=2.0e-5, rtol=0)
    assert bool(
        (
            (solution.source_positions[:, 0] >= 0)
            & (solution.source_positions[:, 0] < 360)
        ).all()
    )


def test_approximate_wcs_matching_uses_supplied_healpix_catalog() -> None:
    reference = torch.tensor(
        [[359.9998, 1.0], [0.0002, 1.0], [20.0, -30.0]], dtype=DTYPE
    )
    detector = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=DTYPE)
    sky = torch.tensor([[359.99979, 1.0], [0.00021, 1.0]], dtype=DTYPE)
    observation = AstrometricObservation(
        "match",
        detector,
        None,
        torch.eye(2, dtype=DTYPE),
        sky_position=sky,
        covariance_space="tangent",
    )

    result = match_approximate_wcs(observation, reference, radius_arcsec=1.0)
    torch.testing.assert_close(result.reference_indices, torch.tensor([0, 1]))
    assert bool(result.matched.all())
    assert bool((result.separation_arcsec < 0.1).all())


def test_affine_free_detector_grid_has_zero_constant_and_linear_modes() -> None:
    frame = TangentPlaneFrame(80.0, 10.0)
    y, x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, 6, dtype=DTYPE),
        torch.linspace(-1.0, 1.0, 6, dtype=DTYPE),
        indexing="ij",
    )
    detector = torch.stack((x.reshape(-1), y.reshape(-1)), dim=-1)
    source_ids = torch.arange(detector.shape[0])
    nonlinear = 0.2 * torch.stack((x.square() - y.square(), x * y), dim=-1).reshape(
        -1, 2
    )
    sources = detector + nonlinear
    graph = AstrometricMatchGraph(
        [_observation("grid", frame, detector, source_ids, detector_xy=detector)],
        reference_source_ids=source_ids,
        reference_positions=_world(frame, sources),
        reference_covariance=torch.eye(2, dtype=DTYPE) * 1.0e-6,
        tangent_frame=frame,
    )
    solution = graph.solve(
        AstrometryConfig(
            pose_model="affine",
            fit_detector_grid=True,
            detector_grid_shape=(4, 4),
            detector_grid_regularization=1.0e-12,
            convergence_tolerance=1.0e-10,
        )
    )
    assert solution.detector_residual_control is not None
    control = solution.detector_residual_control[0]
    gy, gx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, 4, dtype=DTYPE),
        torch.linspace(-1.0, 1.0, 4, dtype=DTYPE),
        indexing="ij",
    )
    affine = torch.stack((torch.ones_like(gx), gx, gy), dim=-1).reshape(-1, 3)
    torch.testing.assert_close(
        affine.mT @ control.reshape(-1, 2),
        torch.zeros((3, 2), dtype=DTYPE),
        atol=2.0e-10,
        rtol=0,
    )
    assert solution.residuals_arcsec.shape == detector.shape
