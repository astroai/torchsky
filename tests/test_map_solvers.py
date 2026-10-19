from __future__ import annotations

import math

import torch

from torchsky.map.solvers import build_spherical_harmonic_solver_hooks, build_spherical_healpix_solver_hooks, build_spherical_spin_harmonic_solver_hooks
from torchsky.map.operators import (
    InterpolationGeometryOp,
    ObservationOperatorBundle,
    spherical_spectral_operator_bundle,
)
from torchsky.map.scenes import DiscreteTargetImage, HealpixMapScene
from torchsky.map.solvers import (
    ComponentDecompositionDiagnostics,
    HarmonicComponentDecomposition,
    SurveyInstrumentComponentMixer,
    build_harmonic_cl_prior_operator,
    build_harmonic_dense_prior_operator,
    build_harmonic_degree_prior_operator,
    build_conditioned_component_mixture_operator,
    build_spin_harmonic_kronecker_prior_operator,
    build_state_component_mixture_operator,
    build_state_subspace_operator,
    build_state_matrix_operator,
    build_state_low_rank_operator,
    build_teb_cls_prior_operator,
    build_solver_hooks,
    build_state_diagonal_operator,
    build_state_identity_operator,
    build_state_kronecker_operator,
    compose_bound_operators,
    harmonic_dense_covariance,
    harmonic_cl_diagonal,
    harmonic_degree_diagonal,
    make_diagonal_preconditioner,
    teb_cls_precision_factors,
)
from torchsky.map.target import TargetRenderSpec


def _simple_header() -> dict[str, float | int | str]:
    return {
        "NAXIS": 2,
        "NAXIS1": 4,
        "NAXIS2": 3,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 2.0,
        "CRPIX2": 1.5,
        "CRVAL1": 180.0,
        "CRVAL2": 0.0,
        "CDELT1": -0.01,
        "CDELT2": 0.01,
    }


def _offset_header(
    ra_offset_deg: float, dec_offset_deg: float
) -> dict[str, float | int | str]:
    header = dict(_simple_header())
    header["CRVAL1"] = float(header["CRVAL1"]) + float(ra_offset_deg)
    header["CRVAL2"] = float(header["CRVAL2"]) + float(dec_offset_deg)
    return header


def test_solver_hooks_default_to_scene_render_contract() -> None:
    hooks = build_solver_hooks(
        ObservationOperatorBundle(geometry=InterpolationGeometryOp()),
    )
    target = TargetRenderSpec.from_header(_simple_header())
    scene = DiscreteTargetImage(torch.ones((3, 4), dtype=torch.float32))

    rendered = hooks.render_scene(scene, target)
    noised = hooks.apply_noise_model(rendered)

    torch.testing.assert_close(rendered, torch.ones((3, 4), dtype=torch.float32))
    torch.testing.assert_close(noised, rendered)


def test_solver_hooks_accept_custom_operator_and_noise() -> None:
    hooks = build_solver_hooks(
        ObservationOperatorBundle(geometry=InterpolationGeometryOp()),
        apply_operator=lambda scene, target: scene.render_to_target(target) * 2.0,
        apply_noise=lambda tensor: tensor + 3.0,
    )
    target = TargetRenderSpec.from_header(_simple_header())
    scene = DiscreteTargetImage(torch.ones((3, 4), dtype=torch.float32))

    rendered = hooks.render_scene(scene, target)
    noised = hooks.apply_noise_model(rendered)

    torch.testing.assert_close(rendered, torch.full((3, 4), 2.0, dtype=torch.float32))
    torch.testing.assert_close(noised, torch.full((3, 4), 5.0, dtype=torch.float32))


def test_solver_hooks_require_configured_adjoint_for_adjoint_application() -> None:
    hooks = build_solver_hooks(
        ObservationOperatorBundle(geometry=InterpolationGeometryOp()),
    )

    try:
        hooks.apply_adjoint_model(torch.ones((3, 4), dtype=torch.float32))
    except NotImplementedError as exc:
        assert "adjoint operator hook is not configured" in str(exc)
    else:
        raise AssertionError("expected missing adjoint hook to raise")


def test_solver_hooks_assemble_normal_equation_probe() -> None:
    hooks = build_solver_hooks(
        ObservationOperatorBundle(geometry=InterpolationGeometryOp()),
        apply_noise=lambda tensor: tensor * 0.5,
        apply_adjoint=lambda tensor: tensor.sum(dim=0),
    )
    target = TargetRenderSpec.from_header(_simple_header())
    scene = DiscreteTargetImage(torch.full((3, 4), 2.0, dtype=torch.float32))
    data = torch.full((3, 4), 6.0, dtype=torch.float32)

    probe = hooks.normal_equation_probe(scene, target, data)

    torch.testing.assert_close(probe.prediction, torch.full((3, 4), 2.0))
    torch.testing.assert_close(probe.weighted_prediction, torch.full((3, 4), 1.0))
    torch.testing.assert_close(probe.weighted_data, torch.full((3, 4), 3.0))
    torch.testing.assert_close(probe.lhs_probe, torch.full((4,), 3.0))
    torch.testing.assert_close(probe.rhs, torch.full((4,), 9.0))


def test_solver_hooks_bind_target_provides_shared_operator_interface() -> None:
    hooks = build_solver_hooks(
        ObservationOperatorBundle(geometry=InterpolationGeometryOp()),
        apply_noise=lambda tensor: tensor + 1.0,
        apply_adjoint=lambda tensor: tensor.sum(dim=1),
    )
    target = TargetRenderSpec.from_header(_simple_header())
    operator = hooks.bind_target(target)

    prediction = operator.apply(torch.ones((3, 4), dtype=torch.float32))
    weighted = operator.apply_noise_model(prediction)
    adjoint = operator.adjoint(weighted)

    torch.testing.assert_close(prediction, torch.ones((3, 4), dtype=torch.float32))
    torch.testing.assert_close(weighted, torch.full((3, 4), 2.0))
    torch.testing.assert_close(adjoint, torch.full((3,), 8.0))


def test_bound_operator_normal_equation_probe_accepts_module_scene() -> None:
    class _OnesField(torch.nn.Module):
        def forward(self, ra, dec, wavelength):
            return torch.ones_like(ra, dtype=torch.float32)

    hooks = build_solver_hooks(
        ObservationOperatorBundle(geometry=InterpolationGeometryOp()),
        apply_noise=lambda tensor: tensor * 2.0,
        apply_adjoint=lambda tensor: tensor.sum(dim=0),
    )
    operator = hooks.bind_target(
        TargetRenderSpec.from_header(
            _simple_header(),
            transmission_center=5000.0,
        )
    )
    data = torch.full((3, 4), 5.0, dtype=torch.float32)

    probe = operator.normal_equation_probe(_OnesField(), data)

    torch.testing.assert_close(
        probe.prediction, torch.ones((3, 4), dtype=torch.float32)
    )
    torch.testing.assert_close(probe.weighted_prediction, torch.full((3, 4), 2.0))
    torch.testing.assert_close(probe.weighted_data, torch.full((3, 4), 10.0))


def test_bound_operator_weighted_least_squares_solver_recovers_identity_scene() -> None:
    hooks = build_solver_hooks(
        ObservationOperatorBundle(geometry=InterpolationGeometryOp()),
        apply_adjoint=lambda tensor: tensor,
    )
    operator = hooks.bind_target(TargetRenderSpec.from_header(_simple_header()))
    data = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    result = operator.solve_weighted_least_squares(data, iterations=1, step_size=1.0)

    torch.testing.assert_close(result.solution, data)
    assert len(result.residual_norm_history) == 1


def test_bound_operator_weighted_least_squares_solver_handles_scaled_forward_model() -> (
    None
):
    hooks = build_solver_hooks(
        ObservationOperatorBundle(geometry=InterpolationGeometryOp()),
        apply_operator=lambda scene, target: scene.render_to_target(target) * 2.0,
        apply_adjoint=lambda tensor: tensor * 0.5,
    )
    operator = hooks.bind_target(TargetRenderSpec.from_header(_simple_header()))
    truth = torch.full((3, 4), 7.0, dtype=torch.float32)
    data = truth * 2.0

    result = operator.solve_weighted_least_squares(data, iterations=1, step_size=1.0)

    torch.testing.assert_close(result.solution, truth)


def test_bound_operator_auto_step_estimate_is_positive_and_stable() -> None:
    hooks = build_solver_hooks(
        ObservationOperatorBundle(geometry=InterpolationGeometryOp()),
        apply_adjoint=lambda tensor: tensor,
    )
    operator = hooks.bind_target(TargetRenderSpec.from_header(_simple_header()))
    data = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    step_size = operator.recommended_step_size(data, iterations=3, safety=0.9)
    result = operator.solve_weighted_least_squares(
        data,
        iterations=4,
        step_size=None,
        scale_iterations=3,
        step_safety=0.9,
    )

    assert 0.0 < step_size <= 1.0
    assert result.residual_norm_history[-1] < result.residual_norm_history[0]


def test_solver_hooks_render_healpix_scene_through_spherical_bundle() -> None:
    hooks = build_solver_hooks(
        spherical_spectral_operator_bundle(mode="harmonic_direct"),
    )
    operator = hooks.bind_target(
        TargetRenderSpec.from_header(_simple_header(), band_label="mono")
    )
    scene = HealpixMapScene(
        torch.ones(12 * 4 * 4, dtype=torch.float32),
        nside=4,
    )

    rendered = operator.apply(scene)

    torch.testing.assert_close(
        rendered,
        torch.ones((3, 4), dtype=torch.float32),
        atol=5.0e-2,
        rtol=0.0,
    )


def test_spherical_healpix_solver_hooks_satisfy_adjoint_relation() -> None:
    hooks = build_spherical_healpix_solver_hooks(
        nside=4,
        mode="pixel_bilinear",
    )
    operator = hooks.bind_target(
        TargetRenderSpec.from_header(_simple_header(), band_label="mono")
    )
    coeffs = torch.randn(12 * 4 * 4, dtype=torch.float32)
    image = torch.randn((3, 4), dtype=torch.float32)

    lhs = torch.sum(operator.apply(coeffs) * image)
    rhs = torch.sum(coeffs * operator.adjoint(image))

    torch.testing.assert_close(lhs, rhs, atol=1.0e-4, rtol=1.0e-4)


def test_spherical_healpix_weighted_least_squares_reduces_image_residual() -> None:
    hooks = build_spherical_healpix_solver_hooks(
        nside=4,
        mode="pixel_bilinear",
    )
    operator = hooks.bind_target(
        TargetRenderSpec.from_header(_simple_header(), band_label="mono")
    )
    truth = torch.randn(12 * 4 * 4, dtype=torch.float32)
    data = operator.apply(truth)

    result = operator.solve_weighted_least_squares(
        data,
        initial_solution=torch.zeros(12 * 4 * 4, dtype=torch.float32),
        iterations=6,
        step_size=0.2,
    )

    assert len(result.residual_norm_history) == 6
    assert result.residual_norm_history[-1] < result.residual_norm_history[0]


def test_spherical_healpix_multi_observation_cg_reduces_combined_residual() -> None:
    hooks = build_spherical_healpix_solver_hooks(
        nside=4,
        mode="pixel_bilinear",
    )
    targets = (
        TargetRenderSpec.from_header(_offset_header(0.0, 0.0), band_label="mono"),
        TargetRenderSpec.from_header(_offset_header(0.8, -0.5), band_label="mono"),
    )
    operator_set = hooks.bind_targets(targets)
    truth = torch.randn(12 * 4 * 4, dtype=torch.float32)
    data = operator_set.apply(truth)

    result = operator_set.solve_normal_equation_cg(
        data,
        initial_solution=torch.zeros_like(truth),
        iterations=10,
        damping=1.0e-4,
    )

    assert len(result.residual_norm_history) >= 1
    assert min(result.residual_norm_history) < result.residual_norm_history[0]


def test_spherical_harmonic_solver_hooks_satisfy_adjoint_relation() -> None:
    hooks = build_spherical_harmonic_solver_hooks(
        lmax=2,
    )
    operator = hooks.bind_target(
        TargetRenderSpec.from_header(_simple_header(), band_label="mono")
    )
    coeffs = torch.randn(6, dtype=torch.complex64)
    image = torch.randn((3, 4), dtype=torch.float32)

    lhs = torch.sum(operator.apply(coeffs) * image).to(torch.float64)
    rhs = torch.sum(
        torch.conj(coeffs.to(torch.complex128))
        * operator.adjoint(image).to(torch.complex128)
    ).real

    torch.testing.assert_close(lhs, rhs, atol=1.0e-4, rtol=1.0e-4)


def test_spherical_harmonic_weighted_least_squares_reduces_image_residual() -> None:
    hooks = build_spherical_harmonic_solver_hooks(
        lmax=2,
    )
    operator = hooks.bind_target(
        TargetRenderSpec.from_header(_simple_header(), band_label="mono")
    )
    truth = torch.randn(6, dtype=torch.complex64) * 0.1
    data = operator.apply(truth)
    initial = torch.zeros_like(truth)

    result = operator.solve_weighted_least_squares(
        data,
        initial_solution=initial,
        iterations=12,
        step_size=1.0e-2,
    )
    final_prediction = operator.apply(result.solution)
    final_rmse = math.sqrt(
        float(torch.mean(torch.square(final_prediction - data)).item())
    )

    assert len(result.residual_norm_history) == 12
    assert result.residual_norm_history[-1] < result.residual_norm_history[0]
    assert final_rmse < 0.2


def test_spherical_spin_harmonic_solver_hooks_satisfy_adjoint_relation() -> None:
    hooks = build_spherical_spin_harmonic_solver_hooks(
        spin=2,
        lmax=3,
    )
    operator = hooks.bind_target(
        TargetRenderSpec.from_header(_simple_header(), band_label="pol")
    )
    coeffs = torch.randn((2, 10), dtype=torch.complex64)
    maps = torch.randn((2, 3, 4), dtype=torch.float32)

    lhs = torch.sum(operator.apply(coeffs) * maps).to(torch.float64)
    rhs = torch.sum(
        torch.conj(coeffs.to(torch.complex128))
        * operator.adjoint(maps).to(torch.complex128)
    ).real

    torch.testing.assert_close(lhs, rhs, atol=1.0e-4, rtol=1.0e-4)


def test_spherical_spin_harmonic_weighted_least_squares_reduces_map_residual() -> None:
    hooks = build_spherical_spin_harmonic_solver_hooks(
        spin=2,
        lmax=3,
    )
    operator = hooks.bind_target(
        TargetRenderSpec.from_header(_simple_header(), band_label="pol")
    )
    truth = torch.randn((2, 10), dtype=torch.complex64) * 0.05
    data = operator.apply(truth)

    result = operator.solve_weighted_least_squares(
        data,
        initial_solution=torch.zeros_like(truth),
        iterations=20,
        step_size=1.0e-4,
    )
    final_prediction = operator.apply(result.solution)
    final_rmse = torch.sqrt(torch.mean(torch.square(final_prediction - data))).item()

    assert len(result.residual_norm_history) == 20
    assert result.residual_norm_history[-1] < result.residual_norm_history[0]
    assert final_rmse < 0.2


def test_spherical_harmonic_multi_observation_solver_reduces_combined_residual() -> (
    None
):
    hooks = build_spherical_harmonic_solver_hooks(lmax=2)
    targets = (
        TargetRenderSpec.from_header(_offset_header(0.0, 0.0), band_label="mono"),
        TargetRenderSpec.from_header(_offset_header(1.5, 0.7), band_label="mono"),
    )
    operator_set = hooks.bind_targets(targets)
    truth = torch.randn(6, dtype=torch.complex64) * 0.05
    data = operator_set.apply(truth)

    step_size = operator_set.recommended_step_size(data, iterations=4, safety=0.5)
    result = operator_set.solve_weighted_least_squares(
        data,
        initial_solution=torch.zeros_like(truth),
        iterations=16,
        step_size=None,
        scale_iterations=4,
        step_safety=0.5,
    )

    assert step_size > 0.0
    assert len(result.residual_norm_history) == 16
    assert result.residual_norm_history[-1] < result.residual_norm_history[0]


def test_spherical_harmonic_multi_observation_cg_reduces_combined_residual() -> None:
    hooks = build_spherical_harmonic_solver_hooks(lmax=2)
    targets = (
        TargetRenderSpec.from_header(_offset_header(0.0, 0.0), band_label="mono"),
        TargetRenderSpec.from_header(_offset_header(1.2, 0.6), band_label="mono"),
    )
    operator_set = hooks.bind_targets(targets)
    truth = torch.randn(6, dtype=torch.complex64) * 0.05
    data = operator_set.apply(truth)

    result = operator_set.solve_normal_equation_cg(
        data,
        initial_solution=torch.zeros_like(truth),
        iterations=10,
        damping=1.0e-4,
    )

    assert len(result.residual_norm_history) >= 1
    assert result.residual_norm_history[-1] < result.residual_norm_history[0]


def test_spherical_harmonic_multi_observation_pcg_with_exact_diagonal_reduces_combined_residual() -> (
    None
):
    hooks = build_spherical_harmonic_solver_hooks(lmax=2)
    targets = (
        TargetRenderSpec.from_header(_offset_header(0.0, 0.0), band_label="mono"),
        TargetRenderSpec.from_header(_offset_header(1.2, 0.6), band_label="mono"),
    )
    operator_set = hooks.bind_targets(targets)
    truth = torch.randn(6, dtype=torch.complex64) * 0.05
    data = operator_set.apply(truth)
    diagonal = operator_set.estimate_normal_operator_diagonal(
        data,
        initial_probe=torch.zeros_like(truth),
        damping=1.0e-4,
    )

    result = operator_set.solve_normal_equation_cg(
        data,
        initial_solution=torch.zeros_like(truth),
        iterations=8,
        damping=1.0e-4,
        preconditioner=make_diagonal_preconditioner(diagonal),
    )

    assert diagonal.shape == truth.shape
    assert len(result.residual_norm_history) >= 1
    assert result.residual_norm_history[-1] < result.residual_norm_history[0]


def test_spherical_spin_harmonic_multi_observation_cg_reduces_combined_residual() -> (
    None
):
    hooks = build_spherical_spin_harmonic_solver_hooks(
        spin=2,
        lmax=3,
    )
    targets = (
        TargetRenderSpec.from_header(_offset_header(0.0, 0.0), band_label="pol"),
        TargetRenderSpec.from_header(_offset_header(-1.0, 0.8), band_label="pol"),
    )
    operator_set = hooks.bind_targets(targets)
    truth = torch.randn((2, 10), dtype=torch.complex64) * 0.10
    data = operator_set.apply(truth)

    result = operator_set.solve_normal_equation_cg(
        data,
        initial_solution=torch.zeros_like(truth),
        iterations=12,
        damping=1.0e-4,
    )

    assert len(result.residual_norm_history) >= 1
    assert result.residual_norm_history[-1] < result.residual_norm_history[0]


def test_composed_operator_blocks_support_mixed_image_and_state_constraints() -> None:
    hooks = build_spherical_harmonic_solver_hooks(lmax=2)
    targets = (
        TargetRenderSpec.from_header(_offset_header(0.0, 0.0), band_label="mono"),
        TargetRenderSpec.from_header(_offset_header(1.0, -0.4), band_label="mono"),
    )
    image_block = hooks.bind_targets(targets)
    truth = torch.randn(6, dtype=torch.complex64) * 0.05
    anchor_operator = build_state_identity_operator(
        truth,
        apply_noise=lambda tensor: tensor * 0.05,
    )
    operator_set = compose_bound_operators(image_block, anchor_operator)
    data = (*image_block.apply(truth), truth)
    diagonal = operator_set.estimate_normal_operator_diagonal(
        data,
        initial_probe=torch.zeros_like(truth),
        damping=1.0e-4,
    )

    result = operator_set.solve_normal_equation_cg(
        data,
        initial_solution=torch.zeros_like(truth),
        iterations=8,
        damping=1.0e-4,
        preconditioner=make_diagonal_preconditioner(diagonal),
    )
    coefficient_error = torch.linalg.vector_norm(result.solution - truth).item()

    assert len(result.residual_norm_history) >= 1
    assert result.residual_norm_history[-1] < result.residual_norm_history[0]
    assert coefficient_error < torch.linalg.vector_norm(truth).item()


def test_harmonic_degree_diagonal_matches_expected_ell_ordering() -> None:
    diagonal = harmonic_degree_diagonal(
        lmax=2,
        power=1.0,
        include_monopole=False,
        dtype=torch.float32,
    )
    expected = torch.tensor([0.0, 1.0, 2.0, 1.0, 2.0, 2.0], dtype=torch.float32)
    torch.testing.assert_close(diagonal, expected)


def test_state_diagonal_operator_scales_forward_and_adjoint() -> None:
    state = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    diagonal = torch.tensor([2.0, 4.0, 8.0], dtype=torch.float32)
    operator = build_state_diagonal_operator(state, diagonal)

    forward = operator.apply(state)
    adjoint = operator.adjoint(torch.ones_like(state))

    torch.testing.assert_close(forward, torch.tensor([2.0, 8.0, 24.0]))
    torch.testing.assert_close(adjoint, diagonal)


def test_state_matrix_operator_satisfies_adjoint_relation() -> None:
    state = torch.tensor([1.0 + 0.0j, 2.0 - 1.0j], dtype=torch.complex64)
    matrix = torch.tensor(
        [[2.0 + 0.0j, 1.0 - 1.0j], [0.5 + 0.25j, 3.0 + 0.0j]],
        dtype=torch.complex64,
    )
    operator = build_state_matrix_operator(state, matrix)
    image = torch.tensor([0.4 - 0.1j, -0.3 + 0.2j], dtype=torch.complex64)

    lhs = torch.sum(torch.conj(operator.apply(state)) * image)
    rhs = torch.sum(torch.conj(state) * operator.adjoint(image))

    torch.testing.assert_close(lhs, rhs)


def test_state_subspace_operator_embeds_adjoint_into_joint_state() -> None:
    hooks = build_spherical_harmonic_solver_hooks(lmax=2)
    base = hooks.bind_target(
        TargetRenderSpec.from_header(_simple_header(), band_label="mono")
    )
    joint_state = torch.zeros((3, 6), dtype=torch.complex64)
    subspace = build_state_subspace_operator(joint_state, base, selector=0)
    coeffs = torch.randn((3, 6), dtype=torch.complex64)
    image = torch.randn((3, 4), dtype=torch.float32)

    lhs = torch.sum(subspace.apply(coeffs) * image).to(torch.float64)
    rhs = torch.sum(
        torch.conj(coeffs.to(torch.complex128))
        * subspace.adjoint(image).to(torch.complex128)
    ).real

    torch.testing.assert_close(lhs, rhs, atol=1.0e-4, rtol=1.0e-4)


def test_state_component_mixture_operator_satisfies_adjoint_relation() -> None:
    hooks = build_spherical_harmonic_solver_hooks(lmax=2)
    base = hooks.bind_target(
        TargetRenderSpec.from_header(_simple_header(), band_label="mono")
    )
    joint_state = torch.zeros((3, 6), dtype=torch.complex64)
    mixture = build_state_component_mixture_operator(
        joint_state,
        base,
        components=(0, 2),
        weights=torch.tensor([1.0 + 0.0j, -0.4 + 0.1j], dtype=torch.complex64),
    )
    coeffs = torch.randn((3, 6), dtype=torch.complex64)
    image = torch.randn((3, 4), dtype=torch.float32)

    lhs = torch.sum(mixture.apply(coeffs) * image).to(torch.float64)
    rhs = torch.sum(
        torch.conj(coeffs.to(torch.complex128))
        * mixture.adjoint(image).to(torch.complex128)
    ).real

    torch.testing.assert_close(lhs, rhs, atol=1.0e-4, rtol=1.0e-4)


def test_survey_instrument_component_mixer_respects_component_roles() -> None:
    mixer = SurveyInstrumentComponentMixer(
        component_roles=("shared", "survey", "instrument", "interaction"),
        surveys=("A", "B"),
        instruments=("cam1", "cam2"),
    )
    with torch.no_grad():
        mixer.shared_weights[:] = torch.tensor([2.0, 2.0, 2.0, 2.0])
        mixer.survey_weights[1] = torch.tensor([3.0, 3.0, 3.0, 3.0])
        mixer.instrument_weights[1] = torch.tensor([4.0, 4.0, 4.0, 4.0])
        mixer.interaction_weights[1, 1] = torch.tensor([5.0, 5.0, 5.0, 5.0])

    weights = mixer(survey="A", instrument="cam1")

    torch.testing.assert_close(weights, torch.tensor([2.0, 3.0, 4.0, 5.0]))


def test_conditioned_component_mixture_operator_satisfies_adjoint_relation() -> None:
    hooks = build_spherical_harmonic_solver_hooks(lmax=2)
    base = hooks.bind_target(
        TargetRenderSpec.from_header(_simple_header(), band_label="mono")
    )
    joint_state = torch.zeros((4, 6), dtype=torch.complex64)
    mixer = SurveyInstrumentComponentMixer(
        component_roles=("shared", "survey", "instrument", "interaction"),
        surveys=("A",),
        instruments=("cam1",),
    )
    with torch.no_grad():
        mixer.shared_weights[:] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        mixer.survey_weights[1] = torch.tensor([0.0, 0.8, 0.0, 0.0])
        mixer.instrument_weights[1] = torch.tensor([0.0, 0.0, -0.5, 0.0])
        mixer.interaction_weights[1, 1] = torch.tensor([0.0, 0.0, 0.0, 0.25])

    conditioned = build_conditioned_component_mixture_operator(
        joint_state,
        base,
        mixer,
        survey="A",
        instrument="cam1",
    )
    coeffs = torch.randn((4, 6), dtype=torch.complex64)
    image = torch.randn((3, 4), dtype=torch.float32)

    lhs = torch.sum(conditioned.apply(coeffs) * image).to(torch.float64)
    rhs = torch.sum(
        torch.conj(coeffs.to(torch.complex128))
        * conditioned.adjoint(image).to(torch.complex128)
    ).real

    torch.testing.assert_close(lhs, rhs, atol=1.0e-4, rtol=1.0e-4)


def test_harmonic_component_decomposition_exposes_state_and_regularization() -> None:
    model = HarmonicComponentDecomposition(
        lmax=2,
        component_roles=("shared", "survey", "instrument"),
        surveys=("A",),
        instruments=("cam1",),
        init_scale=0.02,
    )
    state = model.state()
    diag = harmonic_degree_diagonal(
        lmax=2,
        power=2.0,
        include_monopole=False,
        dtype=torch.float32,
    )
    reg = model.regularization_loss(
        private_harmonic_diagonal=diag,
        private_weight=0.5,
        mixer_weight=0.1,
    )

    assert state.shape == (3, 6)
    assert state.dtype.is_complex
    assert float(reg.item()) >= 0.0


def test_harmonic_component_decomposition_fix_component_scale_gauge_preserves_predictions() -> (
    None
):
    model = HarmonicComponentDecomposition(
        lmax=2,
        component_roles=("shared", "survey", "instrument", "interaction"),
        surveys=("A",),
        instruments=("cam1",),
        init_scale=0.0,
    )
    with torch.no_grad():
        model.coeff_real.copy_(
            torch.tensor(
                [
                    [3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 4.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 5.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 6.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
            )
        )
        model.coeff_imag.zero_()
        model.mixer.shared_weights[:] = torch.tensor([2.0, 0.0, 0.0, 0.0])
        model.mixer.survey_weights[1] = torch.tensor([0.0, 3.0, 0.0, 0.0])
        model.mixer.instrument_weights[1] = torch.tensor([0.0, 0.0, 4.0, 0.0])
        model.mixer.interaction_weights[1, 1] = torch.tensor([0.0, 0.0, 0.0, 5.0])

    hooks = build_spherical_harmonic_solver_hooks(lmax=2)
    targets = (
        TargetRenderSpec.from_header(_simple_header(), band_label="mono"),
        TargetRenderSpec.from_header(_offset_header(0.8, -0.4), band_label="mono"),
    )
    operator = model.conditioned_operator(
        hooks.bind_targets(targets),
        survey="A",
        instrument="cam1",
    )
    state_before = model.state().detach()
    prediction_before = operator.apply(state_before)

    norms_after = model.fix_component_scale_gauge_()

    operator_after = model.conditioned_operator(
        hooks.bind_targets(targets),
        survey="A",
        instrument="cam1",
    )
    prediction_after = operator_after.apply(model.state())

    torch.testing.assert_close(prediction_after, prediction_before)
    torch.testing.assert_close(norms_after, torch.ones_like(norms_after))


def test_harmonic_component_decomposition_fix_component_scale_gauge_skips_zero_components() -> (
    None
):
    model = HarmonicComponentDecomposition(
        lmax=2,
        component_roles=("shared", "survey"),
        surveys=("A",),
        init_scale=0.0,
    )
    with torch.no_grad():
        model.coeff_real[0, 0] = 2.0
        model.coeff_imag.zero_()
        model.mixer.shared_weights[:] = torch.tensor([3.0, 0.0])
        model.mixer.survey_weights[1] = torch.tensor([0.0, 4.0])
    norms_after = model.fix_component_scale_gauge_()

    torch.testing.assert_close(norms_after[0], torch.tensor(1.0))
    torch.testing.assert_close(norms_after[1], torch.tensor(0.0))
    torch.testing.assert_close(model.mixer.shared_weights[0], torch.tensor(6.0))
    torch.testing.assert_close(model.mixer.survey_weights[1, 1], torch.tensor(4.0))


def test_harmonic_component_decomposition_component_state_aggregates_roles() -> None:
    model = HarmonicComponentDecomposition(
        lmax=2,
        component_roles=("shared", "survey", "instrument"),
        surveys=("A",),
        instruments=("cam1",),
        init_scale=0.0,
    )
    state = torch.tensor(
        [
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 2.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 3.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        ],
        dtype=torch.complex64,
    )

    shared = model.component_state("shared", state=state)
    private = model.component_state(("survey", "instrument"), state=state)
    private_components = model.component_state(
        ("survey", "instrument"),
        state=state,
        aggregate=False,
    )

    torch.testing.assert_close(shared, state[0])
    torch.testing.assert_close(private, state[1] + state[2])
    torch.testing.assert_close(private_components, state[1:])


def test_harmonic_component_decomposition_decomposition_diagnostics_measure_exact_leakage() -> (
    None
):
    model = HarmonicComponentDecomposition(
        lmax=2,
        component_roles=("shared", "survey", "instrument"),
        surveys=("A",),
        instruments=("cam1",),
        init_scale=0.0,
    )
    truth = torch.tensor(
        [
            [4.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 2.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        ],
        dtype=torch.complex64,
    )
    estimate = truth.clone()
    estimate[0] = estimate[0] + 0.5 * truth[1]
    estimate[1] = estimate[1] + 0.25 * truth[0]

    diagnostics = model.decomposition_diagnostics(
        truth,
        state=estimate,
        calibration_anchor=truth[0],
        calibration_selector=[0],
    )

    assert isinstance(diagnostics, ComponentDecompositionDiagnostics)
    torch.testing.assert_close(
        torch.tensor(diagnostics.shared_to_private_leakage),
        torch.tensor(0.5),
    )
    torch.testing.assert_close(
        torch.tensor(diagnostics.private_to_shared_leakage),
        torch.tensor(0.25),
    )
    torch.testing.assert_close(
        torch.tensor(diagnostics.calibration_transfer_stability),
        torch.tensor(0.0),
    )


def test_state_low_rank_operator_satisfies_adjoint_relation() -> None:
    state = torch.tensor([1.0 + 0.0j, 2.0 - 1.0j, -0.5 + 0.3j], dtype=torch.complex64)
    basis = torch.tensor(
        [[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 1.0j, 1.0 + 0.0j], [1.0 - 0.5j, -0.2 + 0.3j]],
        dtype=torch.complex64,
    )
    weights = torch.tensor([2.0 + 0.0j, 0.5 + 0.0j], dtype=torch.complex64)
    operator = build_state_low_rank_operator(state, basis, weights=weights)
    image = torch.tensor([0.4 - 0.1j, -0.3 + 0.2j], dtype=torch.complex64)

    lhs = torch.sum(torch.conj(operator.apply(state)) * image)
    rhs = torch.sum(torch.conj(state) * operator.adjoint(image))

    torch.testing.assert_close(lhs, rhs)


def test_state_kronecker_operator_satisfies_adjoint_relation() -> None:
    state = torch.tensor(
        [[1.0 + 0.0j, 2.0 - 1.0j], [0.5 + 0.2j, -0.3 + 0.1j]],
        dtype=torch.complex64,
    )
    left = torch.tensor([[1.0 + 0.0j, 0.2 - 0.1j]], dtype=torch.complex64)
    right = torch.tensor(
        [[2.0 + 0.0j, 0.0 + 0.0j], [0.1 + 0.3j, 1.0 + 0.0j]], dtype=torch.complex64
    )
    operator = build_state_kronecker_operator(state, left, right)
    image = torch.tensor([[0.3 - 0.1j, -0.2 + 0.4j]], dtype=torch.complex64)

    lhs = torch.sum(torch.conj(operator.apply(state)) * image)
    rhs = torch.sum(torch.conj(state) * operator.adjoint(image))

    torch.testing.assert_close(lhs, rhs)


def test_harmonic_dense_covariance_is_positive_definite_shape() -> None:
    covariance = harmonic_dense_covariance(
        lmax=2,
        ell_correlation_length=1.5,
        diagonal_jitter=1.0e-3,
        amplitude=2.0,
        dtype=torch.float64,
    )
    eigvals = torch.linalg.eigvalsh(covariance)

    assert covariance.shape == (6, 6)
    assert torch.all(eigvals > 0)


def test_harmonic_cl_diagonal_tracks_cls_and_beam_transfer() -> None:
    cls = torch.tensor([4.0, 1.0, 0.25], dtype=torch.float64)
    beam = torch.tensor([1.0, 0.5, 0.25], dtype=torch.float64)
    diagonal = harmonic_cl_diagonal(
        lmax=2,
        cls=cls,
        beam_transfer=beam,
        dtype=torch.float64,
    )
    expected = torch.tensor(
        [
            1.0 / (4.0**0.5),
            1.0 / ((1.0 * 0.5 * 0.5) ** 0.5),
            1.0 / ((0.25 * 0.25 * 0.25) ** 0.5),
            1.0 / ((1.0 * 0.5 * 0.5) ** 0.5),
            1.0 / ((0.25 * 0.25 * 0.25) ** 0.5),
            1.0 / ((0.25 * 0.25 * 0.25) ** 0.5),
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(diagonal, expected)


def test_teb_cls_precision_factors_shape_and_positive_blocks() -> None:
    factors = teb_cls_precision_factors(
        lmax=2,
        cls_tt=torch.tensor([10.0, 2.0, 1.0]),
        cls_ee=torch.tensor([5.0, 1.0, 0.5]),
        cls_bb=torch.tensor([5.0, 1.0, 0.5]),
        cls_te=torch.tensor([0.0, 0.2, 0.1]),
    )
    assert factors.shape == (6, 3, 3)
    for idx in range(int(factors.shape[0])):
        block = factors[idx]
        eigvals = torch.linalg.eigvalsh(torch.conj(block).transpose(-2, -1) @ block)
        assert torch.all(eigvals.real > 0)


def test_harmonic_degree_prior_operator_stabilizes_mixed_block_solve() -> None:
    hooks = build_spherical_harmonic_solver_hooks(lmax=2)
    targets = (
        TargetRenderSpec.from_header(_offset_header(0.0, 0.0), band_label="mono"),
        TargetRenderSpec.from_header(_offset_header(1.0, 0.2), band_label="mono"),
    )
    image_block = hooks.bind_targets(targets)
    truth = torch.randn(6, dtype=torch.complex64) * 0.05
    prior_block = build_harmonic_degree_prior_operator(
        truth,
        lmax=2,
        power=2.0,
        include_monopole=False,
        apply_noise=lambda tensor: tensor * 0.2,
    )
    operator_set = compose_bound_operators(image_block, prior_block)
    data = (*image_block.apply(truth), torch.zeros_like(truth))
    diagonal = operator_set.estimate_normal_operator_diagonal(
        data,
        initial_probe=torch.zeros_like(truth),
        damping=1.0e-4,
    )

    result = operator_set.solve_normal_equation_cg(
        data,
        initial_solution=torch.zeros_like(truth),
        iterations=8,
        damping=1.0e-4,
        preconditioner=make_diagonal_preconditioner(diagonal),
    )

    assert len(result.residual_norm_history) >= 1
    assert min(result.residual_norm_history) < result.residual_norm_history[0]


def test_healpix_mixed_image_and_anchor_constraints_reduce_residual() -> None:
    hooks = build_spherical_healpix_solver_hooks(
        nside=4,
        mode="pixel_bilinear",
    )
    targets = (
        TargetRenderSpec.from_header(_offset_header(0.0, 0.0), band_label="mono"),
        TargetRenderSpec.from_header(_offset_header(0.6, -0.4), band_label="mono"),
    )
    image_block = hooks.bind_targets(targets)
    truth = torch.randn(12 * 4 * 4, dtype=torch.float32)
    anchor = truth + 0.05 * torch.randn_like(truth)
    anchor_block = build_state_identity_operator(
        truth,
        apply_noise=lambda tensor: tensor * 0.1,
    )
    operator_set = compose_bound_operators(image_block, anchor_block)
    data = (*image_block.apply(truth), anchor)

    result = operator_set.solve_normal_equation_cg(
        data,
        initial_solution=torch.zeros_like(truth),
        iterations=8,
        damping=1.0e-4,
    )

    assert len(result.residual_norm_history) >= 1
    assert result.residual_norm_history[-1] < result.residual_norm_history[0]


def test_harmonic_dense_prior_operator_reduces_correlated_mode_penalty() -> None:
    hooks = build_spherical_harmonic_solver_hooks(lmax=2)
    targets = (
        TargetRenderSpec.from_header(_offset_header(0.0, 0.0), band_label="mono"),
        TargetRenderSpec.from_header(_offset_header(1.0, 0.2), band_label="mono"),
    )
    image_block = hooks.bind_targets(targets)
    truth = torch.randn(6, dtype=torch.complex64) * 0.05
    covariance = harmonic_dense_covariance(
        lmax=2,
        ell_correlation_length=1.2,
        diagonal_jitter=1.0e-3,
        amplitude=1.0,
        dtype=torch.complex128,
    )
    prior_block = build_harmonic_dense_prior_operator(
        truth,
        covariance,
        apply_noise=lambda tensor: tensor * 0.1,
    )
    operator_set = compose_bound_operators(image_block, prior_block)
    data = (*image_block.apply(truth), torch.zeros_like(truth))
    image_only_result = image_block.solve_normal_equation_cg(
        image_block.apply(truth),
        initial_solution=torch.zeros_like(truth),
        iterations=8,
        damping=1.0e-4,
    )
    diagonal = operator_set.estimate_normal_operator_diagonal(
        data,
        initial_probe=torch.zeros_like(truth),
        damping=1.0e-4,
    )

    result = operator_set.solve_normal_equation_cg(
        data,
        initial_solution=torch.zeros_like(truth),
        iterations=8,
        damping=1.0e-4,
        preconditioner=make_diagonal_preconditioner(diagonal),
    )
    prior_norm_with_prior = torch.linalg.vector_norm(
        prior_block.apply(result.solution)
    ).item()
    prior_norm_image_only = torch.linalg.vector_norm(
        prior_block.apply(image_only_result.solution)
    ).item()

    assert prior_norm_with_prior < prior_norm_image_only


def test_harmonic_cl_prior_operator_reduces_power_spectrum_penalty() -> None:
    hooks = build_spherical_harmonic_solver_hooks(lmax=2)
    targets = (
        TargetRenderSpec.from_header(_offset_header(0.0, 0.0), band_label="mono"),
        TargetRenderSpec.from_header(_offset_header(1.0, 0.2), band_label="mono"),
    )
    image_block = hooks.bind_targets(targets)
    truth = torch.randn(6, dtype=torch.complex64) * 0.05
    cl_prior = torch.tensor([10.0, 1.0, 0.2], dtype=torch.float64)
    beam = torch.tensor([1.0, 0.8, 0.5], dtype=torch.float64)
    prior_block = build_harmonic_cl_prior_operator(
        truth,
        lmax=2,
        cls=cl_prior,
        beam_transfer=beam,
        apply_noise=lambda tensor: tensor * 0.1,
    )
    operator_set = compose_bound_operators(image_block, prior_block)
    data = (*image_block.apply(truth), torch.zeros_like(truth))
    image_only_result = image_block.solve_normal_equation_cg(
        image_block.apply(truth),
        initial_solution=torch.zeros_like(truth),
        iterations=8,
        damping=1.0e-4,
    )
    diagonal = operator_set.estimate_normal_operator_diagonal(
        data,
        initial_probe=torch.zeros_like(truth),
        damping=1.0e-4,
    )
    result = operator_set.solve_normal_equation_cg(
        data,
        initial_solution=torch.zeros_like(truth),
        iterations=8,
        damping=1.0e-4,
        preconditioner=make_diagonal_preconditioner(diagonal),
    )

    prior_norm_with_prior = torch.linalg.vector_norm(
        prior_block.apply(result.solution)
    ).item()
    prior_norm_image_only = torch.linalg.vector_norm(
        prior_block.apply(image_only_result.solution)
    ).item()

    assert prior_norm_with_prior < prior_norm_image_only


def test_teb_cl_prior_joint_temperature_polarization_reduce_cross_spectrum_penalty() -> (
    None
):
    temp_hooks = build_spherical_harmonic_solver_hooks(lmax=2)
    pol_hooks = build_spherical_spin_harmonic_solver_hooks(spin=2, lmax=2)
    joint_truth = torch.randn((3, 6), dtype=torch.complex64) * 0.05
    temp_block = build_state_subspace_operator(
        joint_truth,
        temp_hooks.bind_targets(
            (
                TargetRenderSpec.from_header(
                    _offset_header(0.0, 0.0), band_label="temp"
                ),
                TargetRenderSpec.from_header(
                    _offset_header(1.0, 0.2), band_label="temp"
                ),
            )
        ),
        selector=0,
    )
    pol_block = build_state_subspace_operator(
        joint_truth,
        pol_hooks.bind_targets(
            (
                TargetRenderSpec.from_header(
                    _offset_header(0.0, 0.0), band_label="pol"
                ),
                TargetRenderSpec.from_header(
                    _offset_header(-0.8, 0.5), band_label="pol"
                ),
            )
        ),
        selector=[1, 2],
    )
    prior_block = build_teb_cls_prior_operator(
        joint_truth,
        lmax=2,
        cls_tt=torch.tensor([10.0, 1.5, 0.4], dtype=torch.float64),
        cls_ee=torch.tensor([5.0, 1.0, 0.3], dtype=torch.float64),
        cls_bb=torch.tensor([5.0, 1.0, 0.3], dtype=torch.float64),
        cls_te=torch.tensor([0.0, 0.2, 0.1], dtype=torch.float64),
        apply_noise=lambda tensor: tensor * 0.08,
    )
    operator_set = compose_bound_operators(temp_block, pol_block, prior_block)
    data = (
        *temp_block.apply(joint_truth),
        *pol_block.apply(joint_truth),
        torch.zeros_like(joint_truth),
    )
    image_only_set = compose_bound_operators(temp_block, pol_block)
    image_only_result = image_only_set.solve_normal_equation_cg(
        (*temp_block.apply(joint_truth), *pol_block.apply(joint_truth)),
        initial_solution=torch.zeros_like(joint_truth),
        iterations=8,
        damping=1.0e-4,
    )
    diagonal = operator_set.estimate_normal_operator_diagonal(
        data,
        initial_probe=torch.zeros_like(joint_truth),
        damping=1.0e-4,
    )
    result = operator_set.solve_normal_equation_cg(
        data,
        initial_solution=torch.zeros_like(joint_truth),
        iterations=8,
        damping=1.0e-4,
        preconditioner=make_diagonal_preconditioner(diagonal),
    )

    prior_norm_with_prior = torch.linalg.vector_norm(
        prior_block.apply(result.solution)
    ).item()
    prior_norm_image_only = torch.linalg.vector_norm(
        prior_block.apply(image_only_result.solution)
    ).item()

    assert prior_norm_with_prior < prior_norm_image_only


def test_harmonic_survey_split_reduces_shared_component_leakage() -> None:
    hooks = build_spherical_harmonic_solver_hooks(lmax=3)
    survey_a_targets = (
        TargetRenderSpec.from_header(_offset_header(0.0, 0.0), band_label="survey_a"),
        TargetRenderSpec.from_header(_offset_header(3.0, -1.2), band_label="survey_a"),
    )
    survey_b_targets = (
        TargetRenderSpec.from_header(_offset_header(-2.0, 1.2), band_label="survey_b"),
        TargetRenderSpec.from_header(_offset_header(-4.2, -0.3), band_label="survey_b"),
    )
    shared_truth = torch.zeros(10, dtype=torch.complex64)
    shared_truth[0] = complex(math.sqrt(4.0 * math.pi) * 0.25, 0.0)
    shared_truth[1] = complex(0.08, 0.0)
    shared_truth[2] = complex(-0.05, 0.015)
    shared_truth[4] = complex(0.03, -0.01)
    private_a_truth = torch.zeros(10, dtype=torch.complex64)
    private_a_truth[3] = complex(0.05, -0.01)
    private_a_truth[6] = complex(-0.02, 0.01)
    private_b_truth = torch.zeros(10, dtype=torch.complex64)
    private_b_truth[1] = complex(-0.04, 0.01)
    private_b_truth[7] = complex(0.05, -0.02)
    split_truth = torch.stack([shared_truth, private_a_truth, private_b_truth], dim=0)

    survey_a_block = build_state_component_mixture_operator(
        split_truth,
        hooks.bind_targets(survey_a_targets),
        components=(0, 1),
    )
    survey_b_block = build_state_component_mixture_operator(
        split_truth,
        hooks.bind_targets(survey_b_targets),
        components=(0, 2),
    )
    private_a_prior = build_state_subspace_operator(
        split_truth,
        build_harmonic_degree_prior_operator(
            split_truth[1],
            lmax=3,
            power=2.0,
            include_monopole=False,
            apply_noise=lambda tensor: tensor * 0.08,
        ),
        selector=1,
    )
    private_b_prior = build_state_subspace_operator(
        split_truth,
        build_harmonic_degree_prior_operator(
            split_truth[2],
            lmax=3,
            power=2.0,
            include_monopole=False,
            apply_noise=lambda tensor: tensor * 0.08,
        ),
        selector=2,
    )
    split_set = compose_bound_operators(
        survey_a_block,
        survey_b_block,
        private_a_prior,
        private_b_prior,
    )

    clean_a = survey_a_block.apply(split_truth)
    clean_b = survey_b_block.apply(split_truth)
    observed_a = tuple(cutout + 0.006 * torch.randn_like(cutout) for cutout in clean_a)
    observed_b = tuple(cutout + 0.006 * torch.randn_like(cutout) for cutout in clean_b)
    split_data = (
        *observed_a,
        *observed_b,
        torch.zeros_like(split_truth[1]),
        torch.zeros_like(split_truth[2]),
    )
    split_diagonal = split_set.estimate_normal_operator_diagonal(
        split_data,
        initial_probe=torch.zeros_like(split_truth),
        damping=1.0e-4,
    )
    split_result = split_set.solve_normal_equation_cg(
        split_data,
        initial_solution=torch.zeros_like(split_truth),
        iterations=12,
        damping=1.0e-4,
        preconditioner=make_diagonal_preconditioner(split_diagonal),
    )

    shared_only_block = hooks.bind_targets((*survey_a_targets, *survey_b_targets))
    shared_only_data = (*observed_a, *observed_b)
    shared_only_diagonal = shared_only_block.estimate_normal_operator_diagonal(
        shared_only_data,
        initial_probe=torch.zeros_like(shared_truth),
        damping=1.0e-4,
    )
    shared_only_result = shared_only_block.solve_normal_equation_cg(
        shared_only_data,
        initial_solution=torch.zeros_like(shared_truth),
        iterations=12,
        damping=1.0e-4,
        preconditioner=make_diagonal_preconditioner(shared_only_diagonal),
    )

    shared_error_split = torch.linalg.vector_norm(
        split_result.solution[0] - shared_truth
    ).item()
    shared_error_only = torch.linalg.vector_norm(
        shared_only_result.solution - shared_truth
    ).item()
    private_norm = torch.linalg.vector_norm(split_result.solution[1:]).item()

    assert shared_error_split < shared_error_only
    assert private_norm > 0.0


def test_healpix_low_rank_template_prior_reduces_template_amplitude() -> None:
    hooks = build_spherical_healpix_solver_hooks(
        nside=4,
        mode="pixel_bilinear",
    )
    targets = (
        TargetRenderSpec.from_header(_offset_header(0.0, 0.0), band_label="mono"),
        TargetRenderSpec.from_header(_offset_header(0.6, -0.4), band_label="mono"),
    )
    image_block = hooks.bind_targets(targets)
    npix = 12 * 4 * 4
    truth = torch.randn(npix, dtype=torch.float32)
    basis = torch.stack(
        [
            torch.ones(npix, dtype=torch.float32),
            torch.linspace(-1.0, 1.0, npix, dtype=torch.float32),
        ],
        dim=1,
    )
    low_rank_block = build_state_low_rank_operator(
        truth,
        basis,
        weights=torch.tensor([0.05, 0.05], dtype=torch.float32),
    )
    operator_set = compose_bound_operators(image_block, low_rank_block)
    data = (*image_block.apply(truth), torch.zeros(2, dtype=torch.float32))
    image_only_result = image_block.solve_normal_equation_cg(
        image_block.apply(truth),
        initial_solution=torch.zeros_like(truth),
        iterations=8,
        damping=1.0e-4,
    )
    result = operator_set.solve_normal_equation_cg(
        data,
        initial_solution=torch.zeros_like(truth),
        iterations=8,
        damping=1.0e-4,
    )

    prior_norm_with_prior = torch.linalg.vector_norm(
        low_rank_block.apply(result.solution)
    ).item()
    prior_norm_image_only = torch.linalg.vector_norm(
        low_rank_block.apply(image_only_result.solution)
    ).item()

    assert prior_norm_with_prior < prior_norm_image_only


def test_spin_harmonic_prior_operator_stabilizes_mixed_block_solve() -> None:
    hooks = build_spherical_spin_harmonic_solver_hooks(
        spin=2,
        lmax=3,
    )
    targets = (
        TargetRenderSpec.from_header(_offset_header(0.0, 0.0), band_label="pol"),
        TargetRenderSpec.from_header(_offset_header(-0.8, 0.5), band_label="pol"),
    )
    image_block = hooks.bind_targets(targets)
    truth = torch.randn((2, 10), dtype=torch.complex64) * 0.10
    prior_block = build_harmonic_degree_prior_operator(
        truth,
        lmax=3,
        power=2.0,
        include_monopole=False,
        apply_noise=lambda tensor: tensor * 0.03,
    )
    operator_set = compose_bound_operators(image_block, prior_block)
    data = (*image_block.apply(truth), torch.zeros_like(truth))
    image_only_result = image_block.solve_normal_equation_cg(
        image_block.apply(truth),
        initial_solution=torch.zeros_like(truth),
        iterations=8,
        damping=1.0e-4,
    )
    diagonal = operator_set.estimate_normal_operator_diagonal(
        data,
        initial_probe=torch.zeros_like(truth),
        damping=1.0e-4,
    )

    result = operator_set.solve_normal_equation_cg(
        data,
        initial_solution=torch.zeros_like(truth),
        iterations=8,
        damping=1.0e-4,
        preconditioner=make_diagonal_preconditioner(diagonal),
    )
    prior_norm_with_prior = torch.linalg.vector_norm(
        prior_block.apply(result.solution)
    ).item()
    prior_norm_image_only = torch.linalg.vector_norm(
        prior_block.apply(image_only_result.solution)
    ).item()

    assert len(result.residual_norm_history) >= 1
    assert prior_norm_with_prior < prior_norm_image_only
    assert result.residual_norm_history[-1] < 2.0 * result.residual_norm_history[0]


def test_spin_harmonic_kronecker_prior_reduces_separable_penalty() -> None:
    hooks = build_spherical_spin_harmonic_solver_hooks(
        spin=2,
        lmax=3,
    )
    targets = (
        TargetRenderSpec.from_header(_offset_header(0.0, 0.0), band_label="pol"),
        TargetRenderSpec.from_header(_offset_header(-0.8, 0.5), band_label="pol"),
    )
    image_block = hooks.bind_targets(targets)
    truth = torch.randn((2, 10), dtype=torch.complex64) * 0.05
    harmonic_factor = torch.diag(torch.linspace(1.0, 2.0, 10, dtype=torch.complex64))
    eb_coupling = torch.tensor(
        [[1.0 + 0.0j, 0.15 + 0.0j], [0.0 + 0.0j, 0.7 + 0.0j]],
        dtype=torch.complex64,
    )
    prior_block = build_spin_harmonic_kronecker_prior_operator(
        truth,
        harmonic_factor=harmonic_factor,
        eb_coupling=eb_coupling,
        apply_noise=lambda tensor: tensor * 0.15,
    )
    operator_set = compose_bound_operators(image_block, prior_block)
    data = (*image_block.apply(truth), torch.zeros_like(prior_block.apply(truth)))
    image_only_result = image_block.solve_normal_equation_cg(
        image_block.apply(truth),
        initial_solution=torch.zeros_like(truth),
        iterations=8,
        damping=1.0e-4,
    )
    diagonal = operator_set.estimate_normal_operator_diagonal(
        data,
        initial_probe=torch.zeros_like(truth),
        damping=1.0e-4,
    )
    result = operator_set.solve_normal_equation_cg(
        data,
        initial_solution=torch.zeros_like(truth),
        iterations=8,
        damping=1.0e-4,
        preconditioner=make_diagonal_preconditioner(diagonal),
    )

    prior_norm_with_prior = torch.linalg.vector_norm(
        prior_block.apply(result.solution)
    ).item()
    prior_norm_image_only = torch.linalg.vector_norm(
        prior_block.apply(image_only_result.solution)
    ).item()

    assert prior_norm_with_prior < prior_norm_image_only
