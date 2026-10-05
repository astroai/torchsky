"""Measurement-set invariants and probability-density checks."""

import pytest
import torch

import torchsky.encoders as encoders


def test_public_photometry_primitives_exist() -> None:
    for name in (
        "PhotometryEncoder",
        "MaskedSetFusion",
        "BinnedRedshiftHead",
        "marginalize_association_logits",
    ):
        assert hasattr(encoders, name), f"Missing measurement primitive: {name}"


def _encoder():
    return encoders.PhotometryEncoder(
        num_passbands=3,
        num_surveys=2,
        num_methods=2,
        flux_scale=[1.0, 2.0, 3.0],
        latent_dim=8,
    ).double()


def _measurements():
    return {
        "flux": torch.tensor([[-2.0, 1.0, 3.0]], dtype=torch.float64),
        "error": torch.tensor([[0.2, 0.3, 0.4]], dtype=torch.float64),
        "passband_id": torch.tensor([[0, 1, 2]]),
        "survey_id": torch.tensor([[0, 1, 0]]),
        "method_id": torch.tensor([[0, 1, 0]]),
        "status": torch.tensor([[0, 1, 0]]),
        "epoch": torch.tensor([[2000.0, 2010.0, 2020.0]], dtype=torch.float64),
    }


def test_encoder_permutation_invariance_and_nan_padding() -> None:
    model = _encoder()
    batch = _measurements()
    original, supported = model(**batch)
    permutation = torch.tensor([2, 0, 1])
    shuffled, _ = model(**{key: value[:, permutation] for key, value in batch.items()})
    torch.testing.assert_close(original, shuffled)
    padded = {
        key: torch.cat(
            [
                value,
                torch.full_like(
                    value[:, :1], float("nan") if value.is_floating_point() else -1
                ),
            ],
            dim=-1,
        )
        for key, value in batch.items()
    }
    padded["mask"] = torch.tensor([[True, True, True, False]])
    result, _ = model(**padded)
    torch.testing.assert_close(original, result)
    assert supported.tolist() == [True]


def test_encoder_missing_data_is_finite_and_explicitly_unsupported() -> None:
    batch = _measurements()
    for key in ("flux", "error", "epoch"):
        batch[key].fill_(float("nan"))
    latent, supported = _encoder()(**batch, mask=torch.zeros((1, 3), dtype=torch.bool))
    assert not supported.any()
    assert torch.equal(latent, torch.zeros_like(latent))
    head = encoders.BinnedRedshiftHead(8, [0.0, 0.5, 2.0]).double()
    with pytest.raises(ValueError, match="unsupported"):
        head(latent, supported=supported)
    empty = {key: value[:, :0] for key, value in batch.items()}
    latent, supported = _encoder()(**empty)
    assert torch.equal(latent, torch.zeros_like(latent))
    assert not supported.any()


def test_unknown_error_and_epoch_are_distinct_from_known_values() -> None:
    model = _encoder()
    batch = _measurements()
    known, _ = model(**batch)
    unknown, _ = model(**(batch | {"error": None, "epoch": None}))
    assert torch.isfinite(unknown).all()
    assert not torch.allclose(known, unknown)
    nan_unknown, _ = model(
        **(
            batch
            | {"error": torch.full_like(batch["error"], float("nan")), "epoch": None}
        )
    )
    torch.testing.assert_close(unknown, nan_unknown)


def test_limits_and_negative_flux_are_encoded_without_losing_gradients() -> None:
    model = _encoder()
    batch = _measurements()
    batch["flux"].requires_grad_()
    latent, _ = model(**batch)
    detection, _ = model(**(batch | {"status": torch.zeros_like(batch["status"])}))
    assert not torch.allclose(latent, detection)
    latent.square().sum().backward()
    assert batch["flux"].grad is not None
    assert torch.isfinite(batch["flux"].grad).all()
    assert torch.isfinite(model.passband_embedding.weight.grad).all()


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("flux", float("nan"), "flux"),
        ("error", 0.0, "error"),
        ("error", float("inf"), "error"),
        ("epoch", float("inf"), "epoch"),
        ("passband_id", 3, "passband_id"),
        ("survey_id", -1, "survey_id"),
        ("method_id", 2, "method_id"),
        ("status", 2, "status"),
    ],
)
def test_invalid_observed_measurements_fail(name, value, message) -> None:
    batch = _measurements()
    batch[name][0, 0] = value
    with pytest.raises(ValueError, match=message):
        _encoder()(**batch)


def test_encoder_configuration_and_shape_boundaries() -> None:
    with pytest.raises(ValueError, match="flux_scale"):
        encoders.PhotometryEncoder(2, 1, flux_scale=[1.0, -1.0])
    with pytest.raises(ValueError, match="flux_scale"):
        encoders.PhotometryEncoder(2, 1, flux_scale=[1.0])
    with pytest.raises(ValueError, match="epoch_scale"):
        encoders.PhotometryEncoder(2, 1, flux_scale=[1.0, 1.0], epoch_scale=0.0)
    batch = _measurements()
    with pytest.raises(ValueError, match="shape"):
        _encoder()(**(batch | {"error": batch["error"][:, :1]}))
    with pytest.raises(ValueError, match="integer"):
        _encoder()(**(batch | {"passband_id": batch["passband_id"].double()}))


def test_masked_modality_fusion_ignores_padding_and_supports_empty_sets() -> None:
    fusion = encoders.MaskedSetFusion(4).double()
    tokens = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[True, False, True], [False, False, False]])
    latent, supported = fusion(tokens, mask)
    assert supported.tolist() == [True, False]
    assert torch.equal(latent[1], torch.zeros_like(latent[1]))
    contaminated = torch.where(
        mask.unsqueeze(-1), tokens, torch.full_like(tokens, float("nan"))
    )
    equivalent, _ = fusion(contaminated, mask)
    torch.testing.assert_close(latent, equivalent)
    latent.sum().backward()
    assert torch.equal(tokens.grad[~mask], torch.zeros_like(tokens.grad[~mask]))


def test_binned_head_density_matches_manual_nonuniform_bins() -> None:
    head = encoders.BinnedRedshiftHead(3, [0.0, 0.5, 2.0]).double()
    logits = torch.tensor([[0.0, 0.0]] * 5, dtype=torch.float64, requires_grad=True)
    z = torch.tensor([0.25, 0.5, 2.0, -0.1, 2.1], dtype=torch.float64)
    logp = head.log_prob(logits, z)
    torch.testing.assert_close(
        logp[:3],
        torch.tensor(
            [0.0, -1.0986122886681098, -1.0986122886681098], dtype=torch.float64
        ),
    )
    assert torch.isneginf(logp[3:]).all()
    (-logp[:3].mean()).backward()
    assert torch.isfinite(logits.grad).all()
    prediction = head(torch.zeros(2, 3, dtype=torch.float64))
    assert prediction.shape == (2, 2)


def test_binned_head_rejects_invalid_density_inputs() -> None:
    with pytest.raises(ValueError, match="bin_edges"):
        encoders.BinnedRedshiftHead(2, [0.0, 1.0, 1.0])
    head = encoders.BinnedRedshiftHead(2, [0.0, 1.0])
    with pytest.raises(ValueError, match="finite"):
        head.log_prob(torch.tensor([[float("nan")]]), torch.tensor([0.5]))
    with pytest.raises(ValueError, match="shape"):
        head.log_prob(torch.zeros(2, 1), torch.zeros(1))


def test_association_marginalization_is_probability_mixture_and_differentiable() -> (
    None
):
    probabilities = torch.tensor([[[0.8, 0.2], [0.1, 0.9]]], dtype=torch.float64)
    logits = probabilities.log().requires_grad_()
    weights = torch.tensor([[0.75, 0.25]], dtype=torch.float64, requires_grad=True)
    result = encoders.marginalize_association_logits(logits, weights)
    torch.testing.assert_close(
        result.exp(), torch.tensor([[0.625, 0.375]], dtype=torch.float64)
    )
    result.sum().backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.isfinite(weights.grad).all()
    zero = encoders.marginalize_association_logits(
        logits.detach(), torch.tensor([[1.0, 0.0]])
    )
    torch.testing.assert_close(zero.exp(), probabilities[:, 0])


def test_association_marginalization_rejects_empty_mass_and_ignores_masked_nan() -> (
    None
):
    logits = torch.tensor([[[0.0, 0.0], [float("nan"), float("nan")]]])
    mask = torch.tensor([[True, False]])
    result = encoders.marginalize_association_logits(
        logits, torch.tensor([[2.0, float("nan")]]), mask
    )
    torch.testing.assert_close(result.exp(), torch.tensor([[0.5, 0.5]]))
    with pytest.raises(ValueError, match="positive"):
        encoders.marginalize_association_logits(torch.zeros(1, 2, 2), torch.zeros(1, 2))
    with pytest.raises(ValueError, match="nonnegative"):
        encoders.marginalize_association_logits(
            torch.zeros(1, 2, 2), torch.tensor([[1.0, -1.0]])
        )
