#!/usr/bin/env python
"""Learned survey/instrument split over shared spherical harmonic components.

This example jointly optimizes:

- a shared astrophysical harmonic component
- survey-private latent components
- instrument-private latent components
- a learned mechanistic mixer that decides which contexts see which components

The mixer is explicit rather than a black box: component roles are fixed as
shared, survey, or instrument, but the per-context weights are learned from the
image data.
"""

from __future__ import annotations

import math

import torch

from torchsky.map import (
    build_spherical_harmonic_solver_hooks,
    HarmonicComponentDecomposition,
)
from torchsky.map.solvers import harmonic_degree_diagonal
from torchsky.map.target import TargetRenderSpec


def _tan_header(
    *,
    crval1: float,
    crval2: float,
) -> dict[str, float | int | str]:
    return {
        "NAXIS": 2,
        "NAXIS1": 72,
        "NAXIS2": 72,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 36.5,
        "CRPIX2": 36.5,
        "CRVAL1": crval1,
        "CRVAL2": crval2,
        "CDELT1": -0.14,
        "CDELT2": 0.14,
    }


def main() -> None:
    torch.manual_seed(71)
    lmax = 3
    nalm = (lmax + 1) * (lmax + 2) // 2
    roles = ("shared", "survey", "survey", "instrument", "instrument")

    shared_truth = torch.zeros(nalm, dtype=torch.complex64)
    shared_truth[0] = complex(math.sqrt(4.0 * math.pi) * 0.28, 0.0)
    shared_truth[1] = complex(0.08, 0.0)
    shared_truth[2] = complex(-0.04, 0.02)
    shared_truth[4] = complex(0.03, -0.01)

    survey_a_truth = torch.zeros(nalm, dtype=torch.complex64)
    survey_a_truth[3] = complex(0.05, -0.01)
    survey_a_truth[6] = complex(-0.02, 0.01)

    survey_b_truth = torch.zeros(nalm, dtype=torch.complex64)
    survey_b_truth[1] = complex(-0.04, 0.01)
    survey_b_truth[7] = complex(0.05, -0.02)

    inst_cam1_truth = torch.zeros(nalm, dtype=torch.complex64)
    inst_cam1_truth[5] = complex(0.03, 0.01)

    inst_cam2_truth = torch.zeros(nalm, dtype=torch.complex64)
    inst_cam2_truth[8] = complex(-0.025, 0.015)

    joint_truth = torch.stack(
        [
            shared_truth,
            survey_a_truth,
            survey_b_truth,
            inst_cam1_truth,
            inst_cam2_truth,
        ],
        dim=0,
    )
    truth_model = HarmonicComponentDecomposition(
        lmax=lmax,
        component_roles=roles,
        surveys=("A", "B"),
        instruments=("cam1", "cam2"),
        init_scale=0.0,
    )
    with torch.no_grad():
        truth_model.coeff_real.copy_(joint_truth.real)
        truth_model.coeff_imag.copy_(joint_truth.imag)
        truth_model.mixer.shared_weights[:] = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
        truth_model.mixer.survey_weights[1] = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0])
        truth_model.mixer.survey_weights[2] = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0])
        truth_model.mixer.instrument_weights[1] = torch.tensor(
            [0.0, 0.0, 0.0, 1.0, 0.0]
        )
        truth_model.mixer.instrument_weights[2] = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 1.0]
        )

    hooks = build_spherical_harmonic_solver_hooks(lmax=lmax)
    contexts = (
        (
            "A",
            "cam1",
            (
                TargetRenderSpec.from_header(
                    _tan_header(crval1=210.0, crval2=54.0), band_label="survey_a"
                ),
                TargetRenderSpec.from_header(
                    _tan_header(crval1=213.0, crval2=52.8), band_label="survey_a"
                ),
            ),
        ),
        (
            "B",
            "cam2",
            (
                TargetRenderSpec.from_header(
                    _tan_header(crval1=208.0, crval2=55.2), band_label="survey_b"
                ),
                TargetRenderSpec.from_header(
                    _tan_header(crval1=205.8, crval2=53.7), band_label="survey_b"
                ),
            ),
        ),
    )

    observed_blocks: list[
        tuple[str, str, tuple[TargetRenderSpec, ...], tuple[torch.Tensor, ...]]
    ] = []
    for survey, instrument, targets in contexts:
        operator = truth_model.conditioned_operator(
            hooks.bind_targets(targets),
            survey=survey,
            instrument=instrument,
        )
        clean = operator.apply(truth_model.state())
        observed = tuple(
            (cutout + 0.006 * torch.randn_like(cutout)).detach() for cutout in clean
        )
        observed_blocks.append((survey, instrument, targets, observed))

    fit_model = HarmonicComponentDecomposition(
        component_roles=roles,
        lmax=lmax,
        surveys=("A", "B"),
        instruments=("cam1", "cam2"),
        init_scale=0.01,
    )
    with torch.no_grad():
        fit_model.mixer.shared_weights[0] = 1.0
        fit_model.mixer.survey_weights[1, 1] = 0.2
        fit_model.mixer.survey_weights[2, 2] = 0.2
        fit_model.mixer.instrument_weights[1, 3] = 0.2
        fit_model.mixer.instrument_weights[2, 4] = 0.2
    fit_model.mixer.shared_weights.requires_grad_(False)
    optimizer = torch.optim.Adam(fit_model.parameters(), lr=0.08)
    degree_diag = harmonic_degree_diagonal(
        lmax=lmax,
        power=2.0,
        include_monopole=False,
        dtype=torch.float32,
    )

    loss_history: list[float] = []
    for _step in range(160):
        optimizer.zero_grad()
        current_state = fit_model.state()
        loss = torch.zeros((), dtype=torch.float32)
        for survey, instrument, targets, observed in observed_blocks:
            operator = fit_model.conditioned_operator(
                hooks.bind_targets(targets),
                survey=survey,
                instrument=instrument,
            )
            predictions = operator.apply(current_state)
            for pred, obs in zip(predictions, observed, strict=True):
                loss = loss + torch.mean(torch.square(pred - obs))
        loss = loss + fit_model.regularization_loss(
            private_harmonic_diagonal=degree_diag,
            private_weight=0.01,
            mixer_weight=0.01,
        )
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.item()))

    fit_model.fix_component_scale_gauge_()
    fitted_state = fit_model.state().detach()
    diagnostics = fit_model.decomposition_diagnostics(
        truth_model.state().detach(),
        state=fitted_state,
        calibration_anchor=shared_truth,
        calibration_selector=[0, 1],
    )
    shared_error_learned = float(
        torch.linalg.vector_norm(fitted_state[0] - shared_truth).item()
    )

    shared_only_real = torch.nn.Parameter(0.01 * torch.randn_like(shared_truth.real))
    shared_only_imag = torch.nn.Parameter(0.01 * torch.randn_like(shared_truth.imag))
    shared_optimizer = torch.optim.Adam([shared_only_real, shared_only_imag], lr=0.08)
    for _step in range(160):
        shared_optimizer.zero_grad()
        shared_state = torch.complex(shared_only_real, shared_only_imag)
        loss = torch.zeros((), dtype=torch.float32)
        shared_block = hooks.bind_targets(
            tuple(target for _, _, targets, _ in observed_blocks for target in targets)
        )
        shared_observed = tuple(
            obs for _, _, _, observed in observed_blocks for obs in observed
        )
        shared_predictions = shared_block.apply(shared_state)
        for pred, obs in zip(shared_predictions, shared_observed, strict=True):
            loss = loss + torch.mean(torch.square(pred - obs))
        loss.backward()
        shared_optimizer.step()
    shared_only_state = torch.complex(
        shared_only_real.detach(), shared_only_imag.detach()
    )
    shared_error_naive = float(
        torch.linalg.vector_norm(shared_only_state - shared_truth).item()
    )

    print("Learned survey/instrument split harmonic decomposition example")
    print(
        f"lmax={lmax} nalm={nalm} components={len(roles)} contexts={len(observed_blocks)}"
    )
    print(f"loss history (first 5): {[round(v, 5) for v in loss_history[:5]]}")
    print(f"loss history (last 5): {[round(v, 5) for v in loss_history[-5:]]}")
    print(
        f"shared-component error with learned split model: {shared_error_learned:.6f}"
    )
    print(
        f"shared-component error with naive shared-only fit: {shared_error_naive:.6f}"
    )
    print(
        f"component norms after gauge fix: {[round(float(v), 4) for v in fit_model.component_norms().tolist()]}"
    )
    print(f"shared-to-private leakage: {diagnostics.shared_to_private_leakage:.6f}")
    print(f"private-to-shared leakage: {diagnostics.private_to_shared_leakage:.6f}")
    print(
        f"calibration transfer stability: {diagnostics.calibration_transfer_stability:.6f}"
    )
    print(
        f"learned weights for survey A / cam1: {[round(float(v), 4) for v in fit_model.mixer(survey='A', instrument='cam1').detach().tolist()]}"
    )
    print(
        f"learned weights for survey B / cam2: {[round(float(v), 4) for v in fit_model.mixer(survey='B', instrument='cam2').detach().tolist()]}"
    )
    print(
        "useful when survey/instrument residual structure should be decomposed from a shared astrophysical field without hard-coding the exact nuisance pattern"
    )


if __name__ == "__main__":
    main()
