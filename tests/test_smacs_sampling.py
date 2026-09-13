"""Calibration splits must constrain every manifest-based reconstruction."""

from pathlib import Path
import math
import sys
import os
import subprocess
from dataclasses import replace

import pytest
import torch

from torchsky.map.field import FieldObservation
from torchsky.map import ForwardContext
from torchsky.map.operators import LearnedBackgroundOp, ObservationOperatorBundle

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import bench_map_smacs0723_field_fit_joint as joint
from bench_map_smacs0723_field_fit_joint import _training_calibration_exposure_paths


def test_joint_command_runs_without_gpu_memory_logging():
    env = dict(os.environ)
    env.pop("TORCHSKY_GPU_MEM_LOG", None)
    result = subprocess.run(
        [sys.executable, joint.__file__, "--help"],
        env=env, capture_output=True, text=True, check=True,
    )
    assert "--independent-frozen-calibration" in result.stdout
    invalid = subprocess.run(
        [sys.executable, joint.__file__, "--not-a-valid-option"],
        env=env, capture_output=True, text=True,
    )
    assert invalid.returncode == 2
    assert "unrecognized arguments" in invalid.stderr


def test_manifest_training_selection_preserves_grouped_split():
    splits = {1: "train", 2: "train", 3: "validation", 4: "train",
              5: "train", 6: "test", 7: "train", 8: "validation", 9: "test"}
    paths = [Path(f"exposure_{index}.fits") for index in splits]
    lookup = {path.name: {"split": split} for path, split in zip(paths, splits.values())}
    selected, omitted = _training_calibration_exposure_paths(paths, lookup)
    assert selected == [paths[index - 1] for index in (1, 2, 4, 5, 7)]
    assert omitted == {"validation": 2, "test": 2}


@pytest.mark.parametrize("operator_args", [
    ["--forward-context"],
    ["--operator-calibration-manifest", "missing.json",
     "--operator-calibration-fit-report", "missing-fit.json"],
])
def test_exact_refinement_rejects_conditioned_operators_before_reading_data(operator_args):
    result = subprocess.run(
        [sys.executable, joint.__file__, "--foam-exact-observation-refine",
         "--foam-affine-spatial-basis", *operator_args],
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "does not yet support conditioned operators" in result.stderr


def test_joint_inr_configuration_runs_with_psf_and_pixel_integration():
    from test_map_field_streamed_tiles import Scene, observation
    from torchsky.map.field.training import fit_field_model

    kwargs = joint._joint_real_fit_kwargs(
        "spectral_basis_field", crop_size=9, steps=1, num_observations=1,
    )
    kwargs.update(steps=1, render_spatial_tile_hw=(5, 5), device="cpu")
    result = fit_field_model([observation(9)], Scene(), **kwargs)
    assert math.isfinite(result.final_loss)


def test_fixed_foam_keeps_training_patch_but_bounds_render_work():
    kwargs = joint._joint_real_fit_kwargs(
        "spectral_power_foam", crop_size=128, steps=200,
        num_observations=50, foam_grid_side=64, foam_fixed_geometry=True,
    )
    assert kwargs["training_patch_size"] == 128
    assert kwargs["render_spatial_tile_hw"] == (16, 16)
    assert kwargs["diagnostic_spatial_tile_hw"] == (16, 16)


def test_compact_foam_render_tiles_preserve_psf_prediction_and_gradient():
    from test_field_artifacts import _foam, _header
    from test_map_field_streamed_tiles import observation
    from torchsky.map.field.rendering import iter_field_observation_tiles

    model = _foam().float()
    joint._set_foam_geometry_trainable(model, False)
    model.compact_fixed_geometry = True
    header = {**_header(), "NAXIS1": 36, "NAXIS2": 33,
              "CRPIX1": 18.5, "CRPIX2": 17.0}
    obs = replace(observation(33), header=header,
                  spectral_min=4500., spectral_max=5500.)
    kwargs = dict(num_spectral_samples=5, spatial_sampling="gauss_legendre_2x2")
    full = joint.render_field_observation(model, obs, **kwargs)
    full.square().sum().backward()
    parameters = [p for p in model.parameters() if p.requires_grad]
    expected = [p.grad.clone() for p in parameters]
    model.zero_grad()
    assembled = torch.empty_like(full)
    for ys, xs, prediction in iter_field_observation_tiles(
        model, obs, spatial_tile_hw=(16, 16), **kwargs,
    ):
        assembled[ys, xs] = prediction.detach()
        prediction.square().sum().backward()
    torch.testing.assert_close(assembled, full, rtol=2e-5, atol=2e-6)
    for parameter, gradient in zip(parameters, expected, strict=True):
        torch.testing.assert_close(parameter.grad, gradient, rtol=3e-5, atol=2e-5)


def test_manifest_training_selection_fails_without_training():
    path = Path("heldout.fits")
    with pytest.raises(ValueError, match="no training exposures"):
        _training_calibration_exposure_paths([path], {path.name: {"split": "test"}})


def test_manifest_training_selection_rejects_unknown_split():
    path = Path("unknown.fits")
    with pytest.raises(ValueError, match="invalid calibration split"):
        _training_calibration_exposure_paths([path], {path.name: {"split": "typo"}})


def test_validation_scores_use_fixed_prediction_full_mask_and_original_units(monkeypatch):
    model = torch.nn.Linear(1, 1)
    observation = FieldObservation(
        image=torch.tensor([[1., 2.], [100., 4.]]), header={},
        err=torch.full((2, 2), 2.), weight=torch.ones((2, 2)),
        mask=torch.tensor([[1., 1.], [0., 1.]]), value_unit="MJy/sr",
    )
    context = ForwardContext(survey="F090W", extra={"split": "validation"})
    bundle = object()
    original = model.weight.detach().clone()

    def render(model, observation, **kwargs):
        assert not torch.is_grad_enabled()
        assert not model.training
        assert kwargs["operator_bundle"] is bundle
        assert kwargs["forward_context"] is context
        return torch.zeros_like(observation.image)

    monkeypatch.setattr(joint, "render_field_observation", render)
    rows = joint._validation_detector_scores(
        model, [observation], [context], [bundle], intensity_scale=3., fit_kwargs={}
    )
    assert rows[0]["pixels"] == 3
    assert rows[0]["exposures"] == 1
    budget = rows[0]["per_exposure"][0]
    assert budget["observed_mean"] == pytest.approx(7.)
    assert budget["fixed_detector_mean"] == 0.
    assert budget["observed_minus_fixed_median"] == 6.
    assert budget["observed_minus_fixed_negative_fraction"] == 0.
    assert rows[0]["gaussian_nll_per_pixel"] == pytest.approx(
        .5 * (7. / 4. + math.log(2. * math.pi * 36.))
    )
    assert torch.equal(model.weight, original)


def test_validation_scores_reject_test_before_rendering(monkeypatch):
    def render(*args, **kwargs):
        pytest.fail("test exposures must never be rendered")
    monkeypatch.setattr(joint, "render_field_observation", render)
    with pytest.raises(ValueError, match="validation split"):
        joint._validation_detector_scores(
            torch.nn.Linear(1, 1), [None],
            [ForwardContext(survey="F090W", extra={"split": "test"})], [None],
            intensity_scale=1., fit_kwargs={},
        )


def test_validation_moves_frozen_operator_without_training_exposure(monkeypatch):
    class Background(torch.nn.Linear):
        def to(self, *args, **kwargs):
            self.moved_device = kwargs["device"]
            return super().to(*args, **kwargs)

    model = torch.nn.Linear(1, 1)
    background = Background(1, 1).requires_grad_(False).eval()
    original = {key: value.clone() for key, value in background.state_dict().items()}
    bundle = ObservationOperatorBundle(background=LearnedBackgroundOp(background))
    observation = FieldObservation(
        image=torch.ones((2, 2)), header={}, err=torch.ones((2, 2)),
        mask=torch.ones((2, 2)),
    )

    def render(model, observation, **kwargs):
        assert background.moved_device == next(model.parameters()).device
        assert not background.training
        assert not any(parameter.requires_grad for parameter in background.parameters())
        return torch.ones_like(observation.image)

    monkeypatch.setattr(joint, "render_field_observation", render)
    scores = joint._validation_detector_scores(
        model, [observation], [ForwardContext(survey="F090W", extra={"split": "validation"})],
        [bundle], intensity_scale=1., fit_kwargs={},
    )
    assert scores[0]["chi2_per_pixel"] == 0.
    for key, value in background.state_dict().items():
        torch.testing.assert_close(value, original[key])


def test_validation_scores_render_real_detector_tiles():
    from test_map_field_streamed_tiles import Scene, observation
    from torchsky.map.operators import field_observation_operator_bundle

    obs = observation(9)
    obs = replace(obs, err=torch.ones_like(obs.image))
    scores = joint._validation_detector_scores(
        Scene(), [obs], [ForwardContext(survey="F090W", extra={"split": "validation"})],
        [field_observation_operator_bundle(obs)], intensity_scale=1.,
        fit_kwargs={"num_spectral_samples": 2, "spatial_sampling": "gauss_legendre_2x2",
                    "render_spatial_tile_hw": (5, 5)},
    )
    assert scores[0]["pixels"] == int(obs.mask.sum())
    assert math.isfinite(scores[0]["gaussian_nll_per_pixel"])


def test_plain_report_leads_with_validation_and_labels_missing_scores():
    scores = {"example": [{
        "band": "F090W", "exposures": 2, "pixels": 100,
        "gaussian_nll_per_pixel": -1.5, "chi2_per_pixel": 1.1,
        "value_unit": "MJy/sr",
    }]}
    report = joint._format_markdown([], validation_scores=scores)
    assert "| example | F090W | 2 | 100 | -1.500000 | 1.100000 | MJy/sr |" in report
    assert report.index("excluded from training") < report.index("reference-image")
    assert "final test exposures were not evaluated" in report
    assert "not measured" in joint._format_markdown([])


@pytest.mark.parametrize("unfrozen,refined", [(False, False), (True, False), (False, True)])
def test_joint_artifact_replays_fitted_foam_and_records_resume_limits(tmp_path, unfrozen, refined):
    from test_field_artifacts import _foam
    from torchsky.map.field import load_field_artifact
    from torchsky.map.field.types import FieldFitResult

    model = _foam()
    ra = torch.tensor([11.999, 12.001], dtype=torch.float64)
    dec = torch.tensor([-3.001, -2.999], dtype=torch.float64)
    wavelength = torch.tensor([4800., 5200.], dtype=torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.e-3)
    model(ra, dec, wavelength).square().sum().backward()
    optimizer.step()
    result = FieldFitResult(
        model=model, loss_history=(1.,), final_loss=1.,
        optimizer_state=optimizer.state_dict(), completed_steps=1,
        sampling_rng_state=torch.Generator().manual_seed(42).get_state(),
    )
    calibration = tmp_path / "frozen.pt"
    calibration.write_bytes(b"frozen calibration checkpoint")
    identity = {"path": str(calibration), "size": calibration.stat().st_size,
                "sha256": joint.calibration_fit._sha256(calibration)}
    context = ForwardContext(
        survey="F090W", exposure_id="train-1",
        extra={"split": "train", "measured_passband_sha256": "passband-hash"},
    )
    audit = {"models": [{"checkpoint": str(calibration),
                         "checkpoint_sha256": identity["sha256"]}]}
    path = tmp_path / "artifact"
    status = joint._save_joint_field_artifact(
        path, result=result, model_kind="spectral_power_foam", model_kwargs={},
        fit_kwargs={"lr": 1.e-3, "observation_operator_bundles": (object(),),
                    "observation_forward_contexts": (context,)},
        run_config={"seed": 42}, contexts=[context], input_identities=[identity],
        calibration_audit=audit, intensity_scale=3., normalization_training_count=1,
        unfrozen_extras=unfrozen, exact_refined=refined,
    )
    loaded = load_field_artifact(path)
    torch.testing.assert_close(loaded.model(ra, dec, wavelength), model(ra, dec, wavelength))
    assert loaded.completed_steps == 1
    torch.testing.assert_close(loaded.sampling_rng_state, result.sampling_rng_state)
    assert (loaded.optimizer_state is None) == refined
    if not refined:
        torch.testing.assert_close(loaded.optimizer_state, optimizer.state_dict())
    manifest = loaded.manifest.payload
    assert manifest["fit_config"]["normalization"]["intensity_scale"] == 3.
    assert manifest["fit_config"]["fit_kwargs"] == {"lr": 1.e-3}
    assert manifest["operator_manifest"]["contexts"][0]["extra"] == context.extra
    assert manifest["calibration_manifest"] == audit
    assert status["detector_replay_available"] == (not unfrozen)
    assert status["training_resume_available"] == (not unfrozen and not refined)
    calibration.write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="input size changed"):
        load_field_artifact(path)


def test_joint_artifact_explicitly_skips_per_exposure_wrapper(tmp_path):
    from test_field_artifacts import _foam
    from torchsky.map.field.types import FieldFitResult

    result = FieldFitResult(
        model=joint._PerExposureScalarBias(_foam(), 2), loss_history=(1.,), final_loss=1.,
    )
    status = joint._save_joint_field_artifact(
        tmp_path / "unsupported", result=result, model_kind="spectral_power_foam",
        model_kwargs={}, fit_kwargs={}, run_config={}, contexts=[], input_identities=[],
        calibration_audit=None, intensity_scale=1., normalization_training_count=2,
    )
    assert status["status"] == "unsupported"
    assert not status["training_resume_available"]
    assert not (tmp_path / "unsupported").exists()
