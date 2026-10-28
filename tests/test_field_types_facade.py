from __future__ import annotations

from dataclasses import replace
from types import ModuleType
from typing import cast

import pytest
import torch

import torchsky.map.field as field_api
from torchsky.map.field.types import (
    FieldCoordinateMode,
    FieldDomain,
    FieldObservation,
    PSFSpec,
)


def _domain() -> FieldDomain:
    return FieldDomain(10.0, 20.0, 1.0, 5000.0, 100.0)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("center_ra_deg", float("nan")),
        ("center_dec_deg", float("inf")),
        ("wavelength_center", float("-inf")),
        ("spatial_scale_deg", 0.0),
        ("spatial_scale_deg", float("inf")),
        ("wavelength_scale", -1.0),
        ("wavelength_scale", float("nan")),
    ],
)
def test_field_domain_rejects_nonfinite_centers_and_invalid_scales(
    field_name: str, value: float
) -> None:
    with pytest.raises(ValueError):
        replace(_domain(), **{field_name: value})


@pytest.mark.parametrize("declination", [-90.1, 90.1])
def test_field_domain_rejects_declination_outside_sphere(declination: float) -> None:
    with pytest.raises(ValueError, match="center_dec_deg"):
        replace(_domain(), center_dec_deg=declination)


def test_field_domain_rejects_unknown_coordinate_mode() -> None:
    with pytest.raises(ValueError, match="coord_mode"):
        replace(_domain(), coord_mode=cast(FieldCoordinateMode, "invalid"))


@pytest.mark.parametrize(
    "name",
    [
        "adaptive",
        "artifacts",
        "bandpass",
        "forwarding",
        "hashgrid",
        "inverse",
        "losses",
        "networks",
        "noise",
        "observation_ops",
        "quality",
        "rendering",
        "sky_encoding",
        "training",
    ],
)
def test_field_lazy_pseudo_module_exports_are_modules(name: str) -> None:
    value = getattr(field_api, name)
    assert isinstance(value, ModuleType)
    assert value.__name__ == f"torchsky.map.field.{name}"


def test_field_facade_does_not_advertise_future_annotations() -> None:
    assert "annotations" not in field_api.__all__
    with pytest.raises(AttributeError):
        getattr(field_api, "annotations")


class _LearnedPSF(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))


class _DetectorToSky(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.offset = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def pixel_to_world(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return x + self.offset, y + self.offset


def test_psf_spec_to_copies_model_without_mutating_source() -> None:
    source_model = _LearnedPSF()
    source = PSFSpec.learned(source_model, base_sigma_pix=1.0)

    moved = source.to(dtype=torch.float64)

    assert moved.model is not None
    assert moved.model is not source_model
    assert next(source_model.parameters()).dtype == torch.float32
    assert next(moved.model.parameters()).dtype == torch.float64


def test_field_observation_to_copies_nested_modules_without_mutating_source() -> None:
    psf_model = _LearnedPSF()
    transform = _DetectorToSky()
    source = FieldObservation(
        image=torch.ones((2, 2), dtype=torch.float32),
        header={},
        psf_spec=PSFSpec.learned(psf_model, base_sigma_pix=1.0),
        detector_to_sky=transform,
    )

    moved = source.to(dtype=torch.float64)

    assert moved.psf_spec is not None
    assert moved.psf_spec.model is not None
    assert moved.psf_spec.model is not psf_model
    assert moved.detector_to_sky is not transform
    assert next(psf_model.parameters()).dtype == torch.float32
    assert next(transform.parameters()).dtype == torch.float32
    assert next(moved.psf_spec.model.parameters()).dtype == torch.float64
    moved_transform = cast(torch.nn.Module, moved.detector_to_sky)
    assert next(moved_transform.parameters()).dtype == torch.float64
