#!/usr/bin/env python
"""Train a shared-versus-survey-private field decomposition on synthetic images."""

from __future__ import annotations

import torch

from torchsky.map import (
    FieldComponentAnchorTerm,
    FieldComponentDecomposition,
    FieldDomain,
    FieldObservation,
    ForwardContext,
    fit_field_component_decomposition,
    make_field_component_decomposition,
    render_field_observation,
)
class ConstantField(torch.nn.Module):
    def __init__(self, domain: FieldDomain, value: float) -> None:
        super().__init__()
        self.domain = domain
        self.value = torch.nn.Parameter(torch.tensor(float(value), dtype=torch.float32))

    def forward(
        self,
        ra_deg: torch.Tensor,
        dec_deg: torch.Tensor,
        wavelength_angstrom: torch.Tensor | float,
        *,
        forward_context: ForwardContext | None = None,
    ) -> torch.Tensor:
        del dec_deg, wavelength_angstrom, forward_context
        return torch.zeros_like(ra_deg) + self.value


def _tan_header() -> dict[str, float | int | str]:
    return {
        "NAXIS": 2,
        "NAXIS1": 8,
        "NAXIS2": 8,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 4.5,
        "CRPIX2": 4.5,
        "CRVAL1": 150.0,
        "CRVAL2": 2.0,
        "CDELT1": -1.0 / 3600.0,
        "CDELT2": 1.0 / 3600.0,
    }


def _blank_observation() -> FieldObservation:
    return FieldObservation(
        image=torch.zeros((8, 8), dtype=torch.float32),
        header=_tan_header(),
        spectral_min=5600.0,
        spectral_max=5600.0,
        transmission_center=5600.0,
        transmission_sigma=None,
    )


def main() -> None:
    torch.manual_seed(11)
    domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=1000.0,
    )
    roles = ("shared", "survey")
    contexts = (
        ForwardContext(survey="A", instrument="cam1"),
        ForwardContext(survey="B", instrument="cam1"),
    )

    truth_model = FieldComponentDecomposition(
        [
            ConstantField(domain, 3.0),
            ConstantField(domain, 1.0),
        ],
        component_roles=roles,
        surveys=("A", "B"),
        instruments=("cam1",),
    )
    with torch.no_grad():
        truth_model.mixer.shared_weights[:] = torch.tensor([1.0, 0.0])
        truth_model.mixer.survey_weights[1] = torch.tensor([0.0, 1.0])
        truth_model.mixer.survey_weights[2] = torch.tensor([0.0, -1.0])

    observation_templates = (_blank_observation(), _blank_observation())
    observations = []
    for template, context in zip(observation_templates, contexts, strict=True):
        rendered = render_field_observation(
            truth_model,
            template,
            num_spectral_samples=1,
            forward_context=context,
        ).detach()
        observations.append(
            FieldObservation(
                image=rendered,
                header=template.header,
                spectral_min=template.spectral_min,
                spectral_max=template.spectral_max,
                transmission_center=template.transmission_center,
                transmission_sigma=template.transmission_sigma,
            )
        )

    fit_model = make_field_component_decomposition(
        component_roles=roles,
        domain=domain,
        component_models=[
            ConstantField(domain, 1.0),
            ConstantField(domain, 1.0),
        ],
        observation_forward_contexts=contexts,
    )
    with torch.no_grad():
        fit_model.component_gains[:] = torch.tensor([2.0, -1.5])
        fit_model.mixer.shared_weights[0] = 0.8
        fit_model.mixer.survey_weights[1, 1] = 0.2
        fit_model.mixer.survey_weights[2, 1] = -0.2

    shared_anchor = FieldComponentAnchorTerm(
        observation=_blank_observation(),
        target_values=torch.tensor([3.0], dtype=torch.float32),
        roles="shared",
        sample_x_pix=torch.tensor([3.5], dtype=torch.float32),
        sample_y_pix=torch.tensor([3.5], dtype=torch.float32),
        override_wavelengths=torch.tensor([6000.0], dtype=torch.float32),
        override_bandpass_weights=torch.tensor([1.0], dtype=torch.float32),
        forward_context=contexts[0],
        loss_weight=1.0,
    )
    result = fit_field_component_decomposition(
        observations,
        component_roles=roles,
        model=fit_model,
        observation_forward_contexts=contexts,
        component_anchor_terms=[shared_anchor],
        steps=160,
        lr=1.0e-1,
        num_spectral_samples=1,
    )
    gains_before_fix = result.model.component_gains.detach().clone()
    result.model.fix_component_scale_gauge_()

    diagnostics = result.model.decomposition_diagnostics(
        observations,
        truth_model,
        observation_forward_contexts=contexts,
        num_spectral_samples=1,
        calibration_anchor_model=truth_model,
        calibration_anchor_roles="shared",
    )
    pred_a = render_field_observation(
        result.model,
        observations[0],
        num_spectral_samples=1,
        forward_context=contexts[0],
    )
    pred_b = render_field_observation(
        result.model,
        observations[1],
        num_spectral_samples=1,
        forward_context=contexts[1],
    )

    print("Field shared-versus-survey-private decomposition example")
    print(f"final loss: {result.final_loss:.6f}")
    print(
        "predicted survey means:",
        round(float(pred_a.mean().item()), 4),
        round(float(pred_b.mean().item()), 4),
    )
    print(
        "truth survey means:",
        round(float(observations[0].image.mean().item()), 4),
        round(float(observations[1].image.mean().item()), 4),
    )
    print(f"shared-to-private leakage: {diagnostics.shared_to_private_leakage:.6f}")
    print(f"private-to-shared leakage: {diagnostics.private_to_shared_leakage:.6f}")
    print(f"shared relative error: {diagnostics.shared_relative_error:.6f}")
    print(f"private relative error: {diagnostics.private_relative_error:.6f}")
    print(
        "component gains before/after gauge fix:",
        [round(float(v), 4) for v in gains_before_fix.tolist()],
        [round(float(v), 4) for v in result.model.component_gains.detach().tolist()],
    )
    print(
        "learned context weights:",
        [round(float(v), 4) for v in result.model.mixer(survey="A").detach().tolist()],
        [round(float(v), 4) for v in result.model.mixer(survey="B").detach().tolist()],
    )


if __name__ == "__main__":
    main()
