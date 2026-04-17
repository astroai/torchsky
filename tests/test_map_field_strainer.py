from __future__ import annotations

import copy
import tempfile

import pytest
import torch

import torch.nn as nn

from torchsky.map import (
    STRAINER_TRUNK_CHECKPOINT_FORMAT,
    FieldDomain,
    FieldFitResult,
    FieldObservation,
    StrainerSkyEncoderMLPField,
    estimate_field_domain,
    fit_field_model,
    load_strainer_trunk_checkpoint,
    make_field_model,
    pretrain_strainer_trunk_synthetic,
    render_field_observation,
    save_strainer_trunk_checkpoint,
)


def _tan_header(
    size: int,
    *,
    crval1: float,
    crval2: float,
    scale_deg: float,
) -> dict:
    return {
        "NAXIS": 2,
        "NAXIS1": size,
        "NAXIS2": size,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": size / 2.0,
        "CRPIX2": size / 2.0,
        "CRVAL1": crval1,
        "CRVAL2": crval2,
        "CD1_1": -scale_deg,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": scale_deg,
    }


def test_strainer_reset_decoder_head_changes_weights() -> None:
    domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=900.0,
        coord_mode="sphere_xyz",
    )
    model = StrainerSkyEncoderMLPField(domain)
    w0 = model.decoder_head.weight.detach().clone()
    model.reset_decoder_head()
    assert not torch.allclose(w0, model.decoder_head.weight)


def test_strainer_trunk_checkpoint_roundtrip() -> None:
    domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=900.0,
        coord_mode="sphere_xyz",
    )
    m1 = StrainerSkyEncoderMLPField(domain)
    m2 = StrainerSkyEncoderMLPField(domain)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        path = tmp.name
    try:
        save_strainer_trunk_checkpoint(m1, path, meta={"test": True})
        payload = torch.load(path, weights_only=True)
        assert payload["format"] == STRAINER_TRUNK_CHECKPOINT_FORMAT
        assert payload["meta"]["test"] is True
        load_strainer_trunk_checkpoint(m2, path, strict=True)
        for key in m1.trunk_state_dict():
            assert torch.allclose(m1.state_dict()[key], m2.state_dict()[key])
    finally:
        import os

        os.unlink(path)


def test_fit_field_model_freeze_encoder_trains_head_only() -> None:
    torch.manual_seed(42)
    truth_domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=900.0,
        coord_mode="sphere_xyz",
    )
    truth = make_field_model("strainer_sky_encoder_mlp", truth_domain)
    header = _tan_header(12, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0)
    template = FieldObservation(
        image=torch.zeros((12, 12), dtype=torch.float32),
        header=header,
        wavelength=5600.0,
    )
    image = render_field_observation(truth, template, num_spectral_samples=1).detach()
    obs = FieldObservation(
        image=image,
        header=header,
        weight=torch.ones_like(image),
        wavelength=5600.0,
    )
    estimate_field_domain([obs], coord_mode="sphere_xyz")
    model = copy.deepcopy(truth)
    with torch.no_grad():
        model.decoder_head.weight.add_(0.35)
    body0 = next(model.body.parameters()).detach().clone()
    fit = fit_field_model(
        [obs],
        model,
        steps=35,
        lr=2.0e-2,
        num_spectral_samples=1,
        freeze_field_encoder=True,
        freeze_field_body=True,
    )
    assert torch.allclose(body0, next(model.body.parameters()))
    assert fit.final_loss < fit.initial_loss


def test_pretrain_strainer_trunk_synthetic_runs() -> None:
    domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=900.0,
        coord_mode="sphere_xyz",
    )
    model = StrainerSkyEncoderMLPField(domain)
    pretrain_strainer_trunk_synthetic(
        model,
        steps=3,
        lr=1.0e-2,
        noise_kind="gaussian",
        batch_size=256,
        seed=0,
    )
    ra = torch.tensor([150.0], dtype=torch.float32)
    dec = torch.tensor([2.0], dtype=torch.float32)
    lam = torch.tensor([5600.0], dtype=torch.float32)
    y = model(ra, dec, lam)
    assert y.shape == (1,)
    assert torch.isfinite(y).all()


def test_fit_field_model_all_frozen_raises() -> None:
    domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=900.0,
        coord_mode="sphere_xyz",
    )
    header = _tan_header(8, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0)
    obs = FieldObservation(
        image=torch.zeros((8, 8), dtype=torch.float32),
        header=header,
        wavelength=5600.0,
    )
    model = StrainerSkyEncoderMLPField(domain)
    for param in model.decoder_head.parameters():
        param.requires_grad = False
    with pytest.raises(ValueError, match="no trainable parameters"):
        fit_field_model(
            [obs],
            model,
            steps=1,
            lr=1.0e-3,
            num_spectral_samples=1,
            freeze_field_encoder=True,
            freeze_field_body=True,
        )


def test_load_strainer_trunk_checkpoint_rejects_unknown_format() -> None:
    domain = FieldDomain(
        center_ra_deg=0.0,
        center_dec_deg=0.0,
        spatial_scale_deg=1.0,
        wavelength_center=5000.0,
        wavelength_scale=100.0,
    )
    model = StrainerSkyEncoderMLPField(domain)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        path = tmp.name
    try:
        torch.save(
            {
                "format": "not-a-strainer-format",
                "trunk": model.trunk_state_dict(),
            },
            path,
        )
        with pytest.raises(ValueError, match="unexpected strainer trunk format"):
            load_strainer_trunk_checkpoint(model, path, strict=True)
    finally:
        import os

        os.unlink(path)


def test_pretrain_strainer_noise_kinds_round_robin() -> None:
    domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=900.0,
        coord_mode="sphere_xyz",
    )
    model = StrainerSkyEncoderMLPField(domain)
    pretrain_strainer_trunk_synthetic(
        model,
        steps=6,
        lr=1.0e-2,
        noise_kinds=("gaussian", "uniform", "spectrum"),
        batch_size=128,
        seed=0,
    )
    y = model(
        torch.tensor([150.0], dtype=torch.float32),
        torch.tensor([2.0], dtype=torch.float32),
        torch.tensor([5600.0], dtype=torch.float32),
    )
    assert torch.isfinite(y).all()


def test_field_fit_result_training_loss_after_step() -> None:
    m = nn.Identity()
    fr = FieldFitResult(model=m, loss_history=(1.0, 0.5, 0.25), final_loss=0.2)
    assert fr.training_loss_after_step(1) == 1.0
    assert fr.training_loss_after_step(3) == 0.25
    with pytest.raises(IndexError):
        fr.training_loss_after_step(4)
    with pytest.raises(ValueError):
        fr.training_loss_after_step(0)


def test_make_field_model_strainer_alias() -> None:
    domain = FieldDomain(
        center_ra_deg=0.0,
        center_dec_deg=0.0,
        spatial_scale_deg=1.0,
        wavelength_center=5000.0,
        wavelength_scale=100.0,
    )
    m = make_field_model("strainer_sky_encoder_mlp", domain)
    assert isinstance(m, StrainerSkyEncoderMLPField)
