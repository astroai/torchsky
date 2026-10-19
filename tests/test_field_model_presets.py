import pytest

from torchsky.map.field import (
    FIELD_MODEL_PRESETS,
    FIELD_MODEL_PRESET_NAMES,
    FieldDomain,
    get_field_model_preset,
    make_field_model_preset,
)
from torchsky.map.field import make_field_model


def _domain() -> FieldDomain:
    return FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.01,
        wavelength_center=5000.0,
        wavelength_scale=1000.0,
    )


def test_field_model_presets_declare_small_user_facing_set() -> None:
    assert FIELD_MODEL_PRESET_NAMES == (
        "fast",
        "balanced",
        "spectral",
        "operator_constrained",
    )
    assert FIELD_MODEL_PRESETS["balanced"].kind == "sky_lambda_feature_field"
    with pytest.raises(TypeError):
        FIELD_MODEL_PRESETS["new"] = FIELD_MODEL_PRESETS["fast"]  # type: ignore[index]


def test_make_field_model_preset_builds_same_kind_as_underlying_factory() -> None:
    domain = _domain()
    preset = get_field_model_preset("balanced")
    model = make_field_model_preset("balanced", domain)
    direct = make_field_model(preset.kind, domain)
    assert type(model) is type(direct)


def test_make_field_model_preset_accepts_overrides() -> None:
    model = make_field_model_preset("fast", _domain(), hidden_dim=12, hidden_layers=1)
    assert model.output.in_features == 12


def test_get_field_model_preset_normalizes_names_and_reports_choices() -> None:
    assert get_field_model_preset("operator-constrained").name == "operator_constrained"
    with pytest.raises(ValueError, match="choose one of: fast, balanced"):
        get_field_model_preset("unknown")
