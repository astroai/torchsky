"""Calibration splits must constrain every manifest-based reconstruction."""

from pathlib import Path
import math
import sys
from dataclasses import replace

import pytest
import torch

from torchsky.map.field import FieldObservation
from torchsky.map import ForwardContext
from torchsky.map.operators import LearnedBackgroundOp, ObservationOperatorBundle

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import bench_map_smacs0723_field_fit_joint as joint
from bench_map_smacs0723_field_fit_joint import _training_calibration_exposure_paths


def test_manifest_training_selection_preserves_grouped_split():
    splits = {1: "train", 2: "train", 3: "validation", 4: "train",
              5: "train", 6: "test", 7: "train", 8: "validation", 9: "test"}
    paths = [Path(f"exposure_{index}.fits") for index in splits]
    lookup = {path.name: {"split": split} for path, split in zip(paths, splits.values())}
    selected, omitted = _training_calibration_exposure_paths(paths, lookup)
    assert selected == [paths[index - 1] for index in (1, 2, 4, 5, 7)]
    assert omitted == {"validation": 2, "test": 2}


def test_joint_inr_configuration_runs_with_psf_and_pixel_integration():
    from test_map_field_streamed_tiles import Scene, observation
    from torchsky.map.field.training import fit_field_model

    kwargs = joint._joint_real_fit_kwargs(
        "spectral_basis_field", crop_size=9, steps=1, num_observations=1,
    )
    kwargs.update(steps=1, render_spatial_tile_hw=(5, 5), device="cpu")
    result = fit_field_model([observation(9)], Scene(), **kwargs)
    assert math.isfinite(result.final_loss)


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
