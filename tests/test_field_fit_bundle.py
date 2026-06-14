"""Tests for fitting ExposurePose alongside a FieldModel."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from torchsky.map import (
    ExposurePose,
    FieldObservation,
    fit_field_model,
)


def _tan_header(
    *,
    h: int = 16,
    w: int = 16,
    ra: float = 150.0,
    dec: float = 2.0,
) -> dict[str, float | int | str]:
    return {
        "NAXIS": 2,
        "NAXIS1": w,
        "NAXIS2": h,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": (w + 1) / 2,
        "CRPIX2": (h + 1) / 2,
        "CRVAL1": ra,
        "CRVAL2": dec,
        "CDELT1": -0.1 / 3600.0,
        "CDELT2": 0.1 / 3600.0,
        "CUNIT1": "deg",
        "CUNIT2": "deg",
    }


def test_fit_field_model_with_exposure_pose_shift() -> None:
    # 1. Create a synthetic "true" model
    # A simple model that just predicts a gradient
    class ConstantGradModel(nn.Module):
        def forward(self, ra, dec, lam):
            return (ra - 150.0) * 100.0 + (dec - 2.0) * 100.0

    true_model = ConstantGradModel()

    # 2. Create an observation with a known WCS shift
    header = _tan_header(h=16, w=16)
    pose_true = ExposurePose()
    # Shift by 0.5 arcsec
    shift_ra = 0.5 / 3600.0
    shift_dec = 0.3 / 3600.0
    torch.nn.init.constant_(pose_true.delta_ra_deg, shift_ra)
    torch.nn.init.constant_(pose_true.delta_dec_deg, shift_dec)

    h, w = 16, 16
    y, x = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="ij",
    )
    from torchsky.wcs import WCS

    wcs = WCS(header)
    ra, dec = wcs.pixel_to_world(x, y)

    # Apply true shift to get "observed" RA/Dec
    ra_obs, dec_obs = pose_true.apply_to_world(
        ra,
        dec,
        center_ra_deg=150.0,
        center_dec_deg=2.0,
    )
    image = true_model(ra_obs, dec_obs, 5000.0).detach()

    obs = FieldObservation(
        image=image,
        header=header,
        wavelength=5000.0,
        spectral_min=4500.0,
        spectral_max=5500.0,
        transmission_center=5000.0,
        transmission_sigma=100.0,
    )

    # 3. Fit with an initial identity pose and a learnable model
    class LearnableGradModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.w_ra = nn.Parameter(torch.zeros((), dtype=torch.float32))
            self.w_dec = nn.Parameter(torch.zeros((), dtype=torch.float32))

        def forward(self, ra, dec, lam):
            return (ra - 150.0) * self.w_ra + (dec - 2.0) * self.w_dec

    fit_model = LearnableGradModel()
    pose_fit = ExposurePose()

    # We expect fit_field_model to optimize both fit_model.parameters() and pose_fit.parameters()
    result = fit_field_model(
        [obs],
        fit_model,
        observation_exposure_poses=[pose_fit],
        steps=50,
        lr=1.0e-2,
    )

    # 4. Check if it converged (even partially)
    assert result.final_loss < result.initial_loss

    # And the pose parameters should have moved toward the true shift
    assert abs(float(pose_fit.delta_ra_deg.detach())) > 0
    assert abs(float(pose_fit.delta_dec_deg.detach())) > 0


class _BiasModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros((), dtype=torch.float32))

    def forward(self, ra, dec, lam):
        return self.bias.expand_as(ra)


def _pose_api_observation(
    *,
    ra: float = 150.0,
    dec: float = 2.0,
) -> FieldObservation:
    return FieldObservation(
        image=torch.zeros((6, 6), dtype=torch.float32),
        header=_tan_header(h=6, w=6, ra=ra, dec=dec),
        wavelength=5000.0,
        spectral_min=4500.0,
        spectral_max=5500.0,
    )


def test_fit_field_model_refine_poses_creates_anchored_result_poses() -> None:
    observations = [
        _pose_api_observation(ra=150.0),
        _pose_api_observation(ra=150.0001),
    ]

    result = fit_field_model(
        observations,
        _BiasModel(),
        refine_poses=True,
        steps=1,
        lr=1.0e-3,
    )

    assert result.exposure_poses is not None
    assert len(result.exposure_poses) == 2
    assert result.exposure_poses[0] is not None
    assert result.exposure_poses[1] is not None
    assert not any(
        param.requires_grad for param in result.exposure_poses[0].parameters()
    )
    assert any(param.requires_grad for param in result.exposure_poses[1].parameters())

    diagnostics = result.diagnostics
    assert diagnostics is not None
    pose_diag = diagnostics["pose_refinement"]
    assert pose_diag["enabled"] is True
    assert pose_diag["auto_created"] is True
    assert pose_diag["anchor_first_pose"] is True
    assert pose_diag["num_poses"] == 2
    assert pose_diag["num_trainable_poses"] == 1
    assert len(pose_diag["parameters"]) == 2


def test_fit_field_model_refine_poses_rejects_explicit_pose_sequence() -> None:
    with pytest.raises(ValueError, match="refine_poses"):
        fit_field_model(
            [_pose_api_observation()],
            _BiasModel(),
            refine_poses=True,
            observation_exposure_poses=[ExposurePose()],
            steps=1,
        )
