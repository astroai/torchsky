"""Software checks only; these fixtures provide no scientific evidence."""

import math

import pytest
import torch


def test_neural_pcg_matches_dense_linear_normal_equations() -> None:
    from torchsky.map.field.neural_inverse import gauss_newton_pcg_step

    matrix = torch.tensor([[1.0, 2.0], [3.0, -1.0], [0.5, 1.0]], dtype=torch.float64)
    target = torch.tensor([1.0, 2.0, -1.0], dtype=torch.float64)
    initial = torch.tensor([0.3, -0.1], dtype=torch.float64)
    updated, report = gauss_newton_pcg_step(
        [lambda state: matrix @ state - target],
        initial,
        cg_iterations=4,
    )
    normal = matrix.T @ matrix + report["effective_damping"] * torch.eye(2)
    expected = initial + torch.linalg.solve(
        normal, matrix.T @ (target - matrix @ initial)
    )
    torch.testing.assert_close(updated, expected)
    assert report["final_loss"] < report["initial_loss"]
    assert report["accepted_step_scale"] == 1.0


def test_neural_pcg_backtracks_a_nonlinear_step() -> None:
    from torchsky.map.field.neural_inverse import gauss_newton_pcg_step

    initial = torch.tensor([0.1], dtype=torch.float64)
    updated, report = gauss_newton_pcg_step(
        [lambda state: state.square() - 1.0],
        initial,
        cg_iterations=2,
    )
    assert 0.0 < report["accepted_step_scale"] < 1.0
    assert report["final_loss"] < report["initial_loss"]
    assert updated.item() > initial.item()


def test_neural_pcg_sums_blocks_and_rejects_nonfinite_input() -> None:
    from torchsky.map.field.neural_inverse import gauss_newton_pcg_step

    matrix = torch.tensor([[1.0, 2.0], [3.0, -1.0], [0.5, 1.0]], dtype=torch.float64)
    target = torch.tensor([1.0, 2.0, -1.0], dtype=torch.float64)
    state = torch.tensor([0.3, -0.1], dtype=torch.float64)
    blocks = [
        lambda flat: matrix[:1] @ flat - target[:1],
        lambda flat: matrix[1:] @ flat - target[1:],
    ]
    updated, report = gauss_newton_pcg_step(blocks, state, cg_iterations=4)
    normal = matrix.T @ matrix + report["effective_damping"] * torch.eye(2)
    expected = state + torch.linalg.solve(normal, matrix.T @ (target - matrix @ state))
    torch.testing.assert_close(updated, expected)
    assert report["cg_residual_norm"] < 1.0e-12
    with pytest.raises(ValueError, match="finite"):
        gauss_newton_pcg_step([lambda flat: flat * float("nan")], state)
    unchanged, stationary = gauss_newton_pcg_step(
        [lambda flat: flat * 0.0 + 1.0], state
    )
    torch.testing.assert_close(unchanged, state)
    assert stationary["accepted_step_scale"] == 0.0


class _SimpleDetectorToSky:
    def __init__(
        self,
        *,
        center_ra_deg: float,
        center_dec_deg: float,
        bounds: tuple[float, float, float, float],
        shape: tuple[int, int],
    ) -> None:
        self.center_ra_deg = center_ra_deg
        self.center_dec_deg = center_dec_deg
        self.bounds = bounds
        self.shape = shape

    def pixel_to_world(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        import math

        x_min, x_max, y_min, y_max = self.bounds
        height, width = self.shape
        x_local = x_min + x * (x_max - x_min) / max(width - 1, 1)
        y_local = y_min + y * (y_max - y_min) / max(height - 1, 1)
        ra = self.center_ra_deg + x_local / math.cos(
            math.radians(self.center_dec_deg)
        )
        dec = self.center_dec_deg + y_local
        return ra, dec


def test_learn2foam_proposal_and_preconditioned_recurrent_optimizer_monotone_and_differentiable() -> None:
    from torchsky.map.field import (
        CechSetProposalNetwork,
        FactorizedObjectSceneState,
        FieldDomain,
        FieldObservation,
        LocalObjectINRDecoder,
        PreconditionedLearn2FoamOptimizer,
        bind_linear_band_grid_observations_to_operator_set,
        propose_factorized_objects_from_coadd,
        render_factorized_scene_grid,
    )

    shape = (15, 15)
    bounds = (-0.07, 0.07, -0.07, 0.07)
    domain = FieldDomain(
        center_ra_deg=10.0,
        center_dec_deg=20.0,
        spatial_scale_deg=0.14,
        wavelength_center=20_000.0,
        wavelength_scale=5_000.0,
    )
    weight = torch.full(shape, 50.0, dtype=torch.float64)

    def _make_band_op_set(band_name: str):
        obs_tuple = tuple(
            FieldObservation(
                image=torch.zeros(shape, dtype=torch.float64),
                header={"NAXIS1": shape[1], "NAXIS2": shape[0]},
                weight=weight.clone(),
                detector_to_sky=_SimpleDetectorToSky(
                    center_ra_deg=domain.center_ra_deg,
                    center_dec_deg=domain.center_dec_deg,
                    bounds=bounds,
                    shape=shape,
                ),
                band_name=band_name,
                value_semantics="surface_brightness",
            )
            for _ in range(3)
        )
        return bind_linear_band_grid_observations_to_operator_set(
            obs_tuple,
            domain,
            spatial_resolution=15,
            spatial_bounds_local_deg=bounds,
        )

    op_b0 = _make_band_op_set("F150W")
    op_b1 = _make_band_op_set("F277W")

    true_decoder = LocalObjectINRDecoder(latent_dim=8, hidden_dim=16).double()
    true_state = FactorizedObjectSceneState(
        positions=torch.tensor([[-0.02, 0.01], [0.03, -0.02]], dtype=torch.float64),
        log_fluxes=torch.log(torch.tensor([12.0, 8.0], dtype=torch.float64)),
        log_radii=torch.log(torch.tensor([0.025, 0.025], dtype=torch.float64)),
        sed_coeffs=torch.tensor([[0.6, 1.1], [1.2, 0.5]], dtype=torch.float64),
        latents=torch.zeros((2, 8), dtype=torch.float64),
        diffuse_grids=torch.full((2, 15, 15), 0.5, dtype=torch.float64),
    )
    true_sky = render_factorized_scene_grid(
        true_state,
        true_decoder,
        spatial_resolution=15,
        spatial_bounds_local_deg=bounds,
    )
    data_b0 = list(op_b0.apply(true_sky[0]))
    data_b1 = list(op_b1.apply(true_sky[1]))

    proposal_net = CechSetProposalNetwork(num_bands=2, latent_dim=8, hidden_dim=16).double()
    proposed_state, coadds, sens = propose_factorized_objects_from_coadd(
        (op_b0, op_b1),
        (data_b0, data_b1),
        spatial_resolution=15,
        spatial_bounds_local_deg=bounds,
        max_objects=4,
        snr_threshold=3.0,
        latent_dim=8,
        proposal_net=proposal_net,
    )
    assert proposed_state.num_objects >= 2
    assert coadds.shape == (2, 15, 15)
    assert sens.shape == (2, 15, 15)

    # Perturb true state and verify PreconditionedLearn2FoamOptimizer monotonically lowers loss
    perturbed = FactorizedObjectSceneState(
        positions=true_state.positions + torch.tensor([[0.005, -0.004], [-0.004, 0.005]], dtype=torch.float64),
        log_fluxes=true_state.log_fluxes - 0.35,
        log_radii=true_state.log_radii + 0.1,
        sed_coeffs=true_state.sed_coeffs * 0.8,
        latents=torch.zeros((2, 8), dtype=torch.float64),
        diffuse_grids=torch.full((2, 15, 15), 0.3, dtype=torch.float64),
    )
    rim_opt = PreconditionedLearn2FoamOptimizer(
        true_decoder,
        num_bands=2,
        latent_dim=8,
        hidden_dim=16,
        step_scale=0.6,
    ).double()
    rim_opt.eval()
    res = rim_opt(
        perturbed,
        (op_b0, op_b1),
        (data_b0, data_b1),
        spatial_resolution=15,
        spatial_bounds_local_deg=bounds,
        steps=5,
    )
    assert res.loss_history[-1] < 0.25 * res.loss_history[0]
    for prev_l, next_l in zip(res.loss_history[:-1], res.loss_history[1:], strict=True):
        assert next_l <= prev_l + 1.0e-9

    # Verify end-to-end differentiability wrt learned optimizer + decoder weights in training mode
    rim_opt.train()
    train_res = rim_opt(
        perturbed,
        (op_b0, op_b1),
        (data_b0, data_b1),
        spatial_resolution=15,
        spatial_bounds_local_deg=bounds,
        steps=2,
        backtracking=1,
    )
    meta_loss = torch.mean(torch.square(train_res.sky_grids - true_sky))
    meta_loss.backward()
    assert rim_opt.delta_head.weight.grad is not None
    assert torch.isfinite(rim_opt.delta_head.weight.grad).all()
    assert rim_opt.decoder.out_head.weight.grad is not None
    assert torch.isfinite(rim_opt.decoder.out_head.weight.grad).all()


def test_learn2foam_detector_irls_and_topological_adaptation() -> None:
    from torchsky.map.field import (
        FactorizedObjectSceneState,
        FieldDomain,
        FieldObservation,
        IRLSConfig,
        LocalObjectINRDecoder,
        PreconditionedLearn2FoamOptimizer,
        adapt_factorized_object_scene,
        bind_linear_band_grid_observations_to_operator_set,
        render_factorized_scene_grid,
    )

    # 1. Test split, merge, and prune on Cech graph
    bounds = (-0.05, 0.05, -0.05, 0.05)
    init_state = FactorizedObjectSceneState(
        positions=torch.tensor(
            [
                [-0.02, -0.02],  # 0: high residual + high grad -> split
                [0.02, 0.02],    # 1: coincident with 2 -> merge
                [0.021, 0.02],   # 2: coincident with 1 -> absorbed
                [0.0, -0.03],    # 3: tiny flux -> pruned
            ],
            dtype=torch.float64,
        ),
        log_fluxes=torch.log(torch.tensor([10.0, 5.0, 5.0, 1.0e-7], dtype=torch.float64)),
        log_radii=torch.log(torch.tensor([0.015, 0.015, 0.015, 0.015], dtype=torch.float64)),
        sed_coeffs=torch.ones((4, 2), dtype=torch.float64),
        latents=torch.zeros((4, 4), dtype=torch.float64),
    )
    pos_grad_norm = torch.tensor([0.8, 0.01, 0.01, 0.0], dtype=torch.float64)
    local_snr = torch.tensor([8.0, 1.0, 1.0, 0.1], dtype=torch.float64)
    adapted, counts = adapt_factorized_object_scene(
        init_state,
        pos_grad_norm,
        local_snr,
        spatial_bounds_local_deg=bounds,
    )
    assert counts == {"split": 1, "merged": 1, "pruned": 1}
    # Started with 4: -1 pruned, -1 merged, +1 split = 3 objects
    assert adapted.num_objects == 3
    # Total flux is conserved across split (5 + 5 = 10) and merge (5 + 5 = 10)
    torch.testing.assert_close(
        adapted.fluxes.sum(),
        torch.tensor(20.0, dtype=torch.float64),
        rtol=1.0e-5,
        atol=1.0e-5,
    )

    # 2. Verify detector-space Tukey IRLS inside PreconditionedLearn2FoamOptimizer rejects cosmic ray
    shape = (11, 11)
    domain = FieldDomain(
        center_ra_deg=10.0,
        center_dec_deg=20.0,
        spatial_scale_deg=0.1,
        wavelength_center=20_000.0,
        wavelength_scale=5_000.0,
    )
    obs_tuple = tuple(
        FieldObservation(
            image=torch.zeros(shape, dtype=torch.float64),
            header={"NAXIS1": shape[1], "NAXIS2": shape[0]},
            weight=torch.full(shape, 100.0, dtype=torch.float64),
            detector_to_sky=_SimpleDetectorToSky(
                center_ra_deg=domain.center_ra_deg,
                center_dec_deg=domain.center_dec_deg,
                bounds=bounds,
                shape=shape,
            ),
            band_name="F150W",
            value_semantics="surface_brightness",
        )
        for _ in range(4)
    )
    op_set = bind_linear_band_grid_observations_to_operator_set(
        obs_tuple,
        domain,
        spatial_resolution=11,
        spatial_bounds_local_deg=bounds,
    )
    dec = LocalObjectINRDecoder(latent_dim=4, hidden_dim=8).double()
    truth_state = FactorizedObjectSceneState(
        positions=torch.tensor([[0.0, 0.0]], dtype=torch.float64),
        log_fluxes=torch.log(torch.tensor([15.0], dtype=torch.float64)),
        log_radii=torch.log(torch.tensor([0.02], dtype=torch.float64)),
        sed_coeffs=torch.full((1, 1), math.log(math.expm1(math.log(2.0))), dtype=torch.float64),
        latents=torch.zeros((1, 4), dtype=torch.float64),
        diffuse_grids=torch.ones((1, 11, 11), dtype=torch.float64),
    )
    truth_sky = render_factorized_scene_grid(
        truth_state, dec, spatial_resolution=11, spatial_bounds_local_deg=bounds
    )
    corrupted = [p.clone() for p in op_set.apply(truth_sky[0])]
    corrupted[0][1, 9] = 400.0  # Cosmic ray in exposure 0

    opt = PreconditionedLearn2FoamOptimizer(
        dec, num_bands=1, latent_dim=4, hidden_dim=8, step_scale=0.5
    ).double()
    opt.eval()
    start_state = FactorizedObjectSceneState(
        positions=truth_state.positions.clone(),
        log_fluxes=truth_state.log_fluxes - 0.2,
        log_radii=truth_state.log_radii.clone(),
        sed_coeffs=truth_state.sed_coeffs.clone(),
        latents=truth_state.latents.clone(),
        diffuse_grids=truth_state.diffuse_grids.clone(),
    )
    res_tukey = opt(
        start_state,
        (op_set,),
        (corrupted,),
        spatial_resolution=11,
        spatial_bounds_local_deg=bounds,
        steps=4,
        data_weight_fn="tukey",
        irls_config=IRLSConfig(weight_fn="tukey", use_mad_scaling=True),
    )
    assert abs(float(res_tukey.sky_grids[0, 1, 9].item()) - 1.0) < 0.15

