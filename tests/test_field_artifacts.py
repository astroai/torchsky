from __future__ import annotations

from pathlib import Path

import pytest
import torch

from torchsky.map.field import (
    FieldDomain,
    FieldFitResult,
    FieldObservation,
    DiffuseFoamHybridField,
    FoamInitializedINRField,
    SpectralPowerFoamField,
    fit_field_model,
    load_field_artifact,
    make_field_model,
    save_field_artifact,
)
from torchsky.map.field.core import ExplicitSpectralGridField
from torchsky.map.plan import build_map_input_identity
import torchfits


def _header() -> dict[str, object]:
    return {
        "NAXIS": 2,
        "NAXIS1": 6,
        "NAXIS2": 6,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 3.5,
        "CRPIX2": 3.5,
        "CRVAL1": 12.0,
        "CRVAL2": -3.0,
        "CDELT1": -0.001,
        "CDELT2": 0.001,
    }


def _domain() -> FieldDomain:
    return FieldDomain(
        center_ra_deg=12.0,
        center_dec_deg=-3.0,
        spatial_scale_deg=0.02,
        wavelength_center=5000.0,
        wavelength_scale=500.0,
        coord_mode="sphere_xyz",
    )


def _foam_domain() -> FieldDomain:
    return FieldDomain(
        center_ra_deg=12.0,
        center_dec_deg=-3.0,
        spatial_scale_deg=0.02,
        wavelength_center=5000.0,
        wavelength_scale=500.0,
        coord_mode="local_tangent",
    )


def _foam() -> SpectralPowerFoamField:
    model = make_field_model(
        "spectral_power_foam",
        _foam_domain(),
        sites=torch.tensor([[-0.25, 0.0], [0.25, 0.0]], dtype=torch.float64),
        radii=torch.tensor([0.8, 0.9], dtype=torch.float64),
        cell_coefficients=torch.tensor([[0.5, 0.1], [0.8, -0.1]], dtype=torch.float64),
        cell_gradient_coefficients=torch.tensor(
            [[[0.01, 0.02], [0.03, 0.04]], [[-0.01, 0.0], [0.02, -0.03]]],
            dtype=torch.float64,
        ),
        minimum_radius=0.1,
        temperature_start=3.0,
        temperature_end=12.0,
    )
    assert isinstance(model, SpectralPowerFoamField)
    model.set_curriculum_progress(0.5)
    return model


def test_field_artifact_roundtrip_uses_weights_only_load(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "input.fits"
    torchfits.write(
        str(input_path), torch.ones((6, 6)), header=_header(), overwrite=True
    )
    model = make_field_model("sine_implicit", _domain(), hidden_dim=8, hidden_layers=1)
    artifact_path = tmp_path / "artifact"
    save_field_artifact(
        artifact_path,
        model=model,
        model_kind="sine_implicit",
        model_kwargs={"hidden_dim": 8, "hidden_layers": 1},
        fit_config={"steps": 2, "validation_split": [0]},
        input_identities=[build_map_input_identity(input_path)],
    )

    calls: list[bool | None] = []
    original_load = torch.load

    def watched_load(*args, **kwargs):
        calls.append(kwargs.get("weights_only"))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", watched_load)
    loaded = load_field_artifact(artifact_path, device="cpu")

    assert calls and all(value is True for value in calls)
    assert loaded.manifest.model_kind == "sine_implicit"
    assert loaded.manifest.payload["selected_checkpoint"] == "weights.pt"
    for name, parameter in model.state_dict().items():
        torch.testing.assert_close(parameter, loaded.model.state_dict()[name])


def test_field_artifact_roundtrips_foam_without_tensor_json(tmp_path: Path) -> None:
    model = _foam()
    artifact_path = tmp_path / "foam-artifact"
    manifest = save_field_artifact(
        artifact_path,
        model=model,
        model_kind="spectral_power_foam",
    )
    kwargs = manifest.model_kwargs
    assert kwargs["artifact_constructor"] == "spectral_power_foam_v1"
    assert kwargs["current_temperature"] == 6.0
    assert "sites" not in kwargs
    assert "radii" not in kwargs
    assert "cell_coefficients" not in kwargs

    loaded = load_field_artifact(artifact_path, device="cpu")
    assert isinstance(loaded.model, SpectralPowerFoamField)
    assert loaded.model._temperature == 6.0
    for name, parameter in model.state_dict().items():
        torch.testing.assert_close(parameter, loaded.model.state_dict()[name])
    ra = torch.tensor([11.999, 12.001], dtype=torch.float64)
    dec = torch.tensor([-3.001, -2.999], dtype=torch.float64)
    wavelength = torch.tensor([4800.0, 5200.0], dtype=torch.float64)
    torch.testing.assert_close(
        loaded.model(ra, dec, wavelength),
        model(ra, dec, wavelength),
    )


def test_field_artifact_roundtrips_explicit_grid_spatial_bounds(tmp_path: Path) -> None:
    kwargs = {
        "spatial_resolution": 5,
        "spectral_resolution": 3,
        "spatial_bounds_local_deg": (-0.01, 0.02, -0.03, 0.04),
    }
    model = make_field_model("explicit_spectral_grid_field", _foam_domain(), **kwargs)
    assert isinstance(model, ExplicitSpectralGridField)
    artifact_path = tmp_path / "grid-artifact"
    manifest = save_field_artifact(
        artifact_path,
        model=model,
        model_kind="explicit_spectral_grid_field",
        model_kwargs=kwargs,
    )

    assert manifest.model_kwargs["spatial_bounds_local_deg"] == [
        -0.01,
        0.02,
        -0.03,
        0.04,
    ]
    loaded = load_field_artifact(artifact_path, device="cpu")
    assert isinstance(loaded.model, ExplicitSpectralGridField)
    assert loaded.model.spatial_bounds_local_deg == (-0.01, 0.02, -0.03, 0.04)
    ra = torch.tensor([11.999, 12.001], dtype=torch.float64)
    dec = torch.tensor([-3.001, -2.999], dtype=torch.float64)
    wavelength = torch.tensor([4800.0, 5200.0], dtype=torch.float64)
    torch.testing.assert_close(
        loaded.model(ra, dec, wavelength),
        model(ra, dec, wavelength),
    )


def test_field_artifact_resumes_adapted_foam_optimizer_state(tmp_path: Path) -> None:
    model = _foam()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    points = torch.tensor([[-0.2, 0.1], [0.2, -0.1]], dtype=torch.float64)
    basis = torch.tensor([[1.0, -0.2], [1.0, 0.3]], dtype=torch.float64)
    model.foam(points, basis).sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model.update_adaptation_statistics(
        torch.tensor([0, 1]),
        torch.tensor([5.0, 1.0], dtype=torch.float64),
        torch.tensor([2.0, 0.5], dtype=torch.float64),
        torch.tensor([[-0.1, 0.1], [0.25, 0.0]], dtype=torch.float64),
        split="held_out",
    )
    model.adapt_cells(
        optimizer,
        densify_count=1,
        prune_count=0,
    )
    fit_result = FieldFitResult(
        model=model,
        loss_history=(1.0,),
        final_loss=1.0,
        optimizer_state=optimizer.state_dict(),
        completed_steps=1,
    )

    artifact_path = tmp_path / "adapted-foam-artifact"
    save_field_artifact(
        artifact_path,
        model=model,
        model_kind="spectral_power_foam",
        fit_result=fit_result,
    )
    loaded = load_field_artifact(artifact_path, device="cpu")

    assert isinstance(loaded.model, SpectralPowerFoamField)
    assert loaded.model.sites.shape == (3, 2)
    assert loaded.model.foam.evidence_updates.tolist() == [1, 1, 1]
    resumed_optimizer = torch.optim.Adam(loaded.model.parameters(), lr=1.0e-3)
    assert loaded.optimizer_state is not None
    resumed_optimizer.load_state_dict(loaded.optimizer_state)
    loaded.model.foam(points, basis).sum().backward()
    resumed_optimizer.step()
    assert torch.isfinite(loaded.model.sites).all()


def test_field_artifact_roundtrips_foam_initialized_inr(tmp_path: Path) -> None:
    model = make_field_model(
        "foam_initialized_inr",
        _foam_domain(),
        foam=_foam(),
        hidden_dim=12,
        hidden_layers=2,
        intensity_floor=1.0e-7,
        max_log_correction=5.0,
    )
    assert isinstance(model, FoamInitializedINRField)
    with torch.no_grad():
        model.residual_output.bias.fill_(0.2)
    artifact_path = tmp_path / "foam-inr-artifact"
    manifest = save_field_artifact(
        artifact_path,
        model=model,
        model_kind="foam_initialized_inr",
    )
    assert manifest.model_kwargs["artifact_constructor"] == "foam_initialized_inr_v1"
    assert manifest.model_kwargs["hidden_dim"] == 12
    assert manifest.model_kwargs["hidden_layers"] == 2

    loaded = load_field_artifact(artifact_path, device="cpu")
    assert isinstance(loaded.model, FoamInitializedINRField)
    for name, parameter in model.state_dict().items():
        torch.testing.assert_close(parameter, loaded.model.state_dict()[name])
    ra = torch.tensor([11.999, 12.001], dtype=torch.float64)
    dec = torch.tensor([-3.001, -2.999], dtype=torch.float64)
    wavelength = torch.tensor([4800.0, 5200.0], dtype=torch.float64)
    torch.testing.assert_close(
        loaded.model(ra, dec, wavelength),
        model(ra, dec, wavelength),
    )


def test_field_artifact_roundtrips_diffuse_foam_hybrid(tmp_path: Path) -> None:
    foam = make_field_model(
        "spectral_power_foam",
        _foam_domain(),
        sites=torch.tensor([[-0.25, 0.0], [0.25, 0.0]], dtype=torch.float64),
        radii=torch.tensor([0.8, 0.9], dtype=torch.float64),
        cell_coefficients=torch.tensor(
            [[-0.5, 0.1], [-0.8, -0.1]], dtype=torch.float64
        ),
        minimum_radius=0.1,
        log_intensity_spectra=True,
    )
    assert isinstance(foam, SpectralPowerFoamField)
    model = make_field_model(
        "diffuse_foam_hybrid",
        _foam_domain(),
        foam=foam,
        diffuse_hidden_dim=10,
        diffuse_hidden_layers=1,
        diffuse_omega0=2.0,
        training_stage="foam",
    )
    assert isinstance(model, DiffuseFoamHybridField)
    artifact_path = tmp_path / "diffuse-foam-artifact"
    manifest = save_field_artifact(
        artifact_path,
        model=model,
        model_kind="diffuse_foam_hybrid",
    )

    assert manifest.model_kwargs["artifact_constructor"] == "diffuse_foam_hybrid_v1"
    assert manifest.model_kwargs["training_stage"] == "foam"
    assert "sites" not in manifest.model_kwargs["localized_foam"]
    loaded = load_field_artifact(artifact_path, device="cpu")

    assert isinstance(loaded.model, DiffuseFoamHybridField)
    assert loaded.model.training_stage == "foam"
    for name, parameter in model.state_dict().items():
        torch.testing.assert_close(parameter, loaded.model.state_dict()[name])
    ra = torch.tensor([11.999, 12.001], dtype=torch.float64)
    dec = torch.tensor([-3.001, -2.999], dtype=torch.float64)
    wavelength = torch.tensor([4800.0, 5200.0], dtype=torch.float64)
    torch.testing.assert_close(
        loaded.model(ra, dec, wavelength),
        model(ra, dec, wavelength),
    )


def test_field_artifact_rejects_non_json_metadata_before_writing(
    tmp_path: Path,
) -> None:
    model = make_field_model("sine_implicit", _domain(), hidden_dim=8, hidden_layers=1)
    artifact_path = tmp_path / "invalid-artifact"
    with pytest.raises(ValueError, match="finite JSON"):
        save_field_artifact(
            artifact_path,
            model=model,
            model_kind="sine_implicit",
            model_kwargs={"unsafe_tensor": torch.ones(1)},
        )
    assert not artifact_path.exists()


def test_field_artifact_resume_matches_uninterrupted_training(tmp_path: Path) -> None:
    input_path = tmp_path / "input.fits"
    image = torch.linspace(0.0, 1.0, 36).reshape(6, 6)
    torchfits.write(str(input_path), image, header=_header(), overwrite=True)
    observation = FieldObservation(
        image=image,
        header=_header(),
        weight=torch.ones_like(image),
        wavelength=5000.0,
    )
    model_kwargs = {"hidden_dim": 8, "hidden_layers": 1}
    torch.manual_seed(2026)
    uninterrupted = make_field_model("sine_implicit", _domain(), **model_kwargs)
    split = make_field_model("sine_implicit", _domain(), **model_kwargs)
    split.load_state_dict(uninterrupted.state_dict())
    fit_kwargs = {
        "lr": 1.0e-3,
        "num_spectral_samples": 1,
        "training_patch_size": 3,
        "training_patches_per_observation": 1,
        "observations_per_step": 1,
        "device": "cpu",
    }

    fit_field_model([observation], uninterrupted, steps=4, **fit_kwargs)
    first = fit_field_model([observation], split, steps=2, **fit_kwargs)
    artifact_path = tmp_path / "resume-artifact"
    save_field_artifact(
        artifact_path,
        model=first.model,
        model_kind="sine_implicit",
        model_kwargs=model_kwargs,
        fit_config={"steps": 4, **fit_kwargs, "device": "cpu"},
        input_identities=[build_map_input_identity(input_path)],
        fit_result=first,
    )
    loaded = load_field_artifact(artifact_path, device="cpu")
    resumed = fit_field_model(
        [observation],
        loaded.model,
        steps=2,
        optimizer_state=loaded.optimizer_state,
        sampling_rng_state=loaded.sampling_rng_state,
        **fit_kwargs,
    )

    for name, parameter in uninterrupted.state_dict().items():
        torch.testing.assert_close(
            parameter,
            resumed.model.state_dict()[name],
            atol=1.0e-7,
            rtol=1.0e-6,
        )
