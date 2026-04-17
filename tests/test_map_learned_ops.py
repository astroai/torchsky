from __future__ import annotations

import pytest

from torchsky.map.operators import (
    ExplicitBackgroundOp,
    ExplicitNoiseOp,
    InterpolationGeometryOp,
    LEARNED_OPERATOR_FAMILIES,
    LEARNED_OPERATOR_REGISTRY,
    ObservationOperatorBundle,
    OperatorConditioning,
    learned_operator_manifest,
    make_minimal_learned_operator_scaffold,
    swap_bundle_operator_families,
    validate_learned_operator_families,
)
from torchsky.map.target import TargetRenderSpec


def _target_header() -> dict[str, object]:
    return {
        "NAXIS": 2,
        "NAXIS1": 4,
        "NAXIS2": 3,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 2.0,
        "CRPIX2": 2.0,
        "CRVAL1": 180.0,
        "CRVAL2": 0.0,
        "CDELT1": -0.01,
        "CDELT2": 0.01,
    }


def test_learned_operator_registry_covers_declared_families() -> None:
    assert tuple(LEARNED_OPERATOR_REGISTRY) == LEARNED_OPERATOR_FAMILIES
    assert LEARNED_OPERATOR_REGISTRY["background"]().__class__.__name__ == (
        "LearnedBackgroundOp"
    )


def test_validate_learned_operator_families_rejects_unknown_family() -> None:
    assert validate_learned_operator_families(("background", "noise")) == (
        "background",
        "noise",
    )
    with pytest.raises(ValueError, match="unsupported learned operator families"):
        validate_learned_operator_families(("background", "detector_response"))


def test_learned_operator_manifest_reports_slots_and_conditioning_union() -> None:
    conditioning = OperatorConditioning(
        detector_coordinates=True,
        wavelength=False,
        time=True,
        exposure_metadata_keys=("exposure_id", "thermal_state"),
    )
    scaffold = make_minimal_learned_operator_scaffold(conditioning=conditioning)
    scaffold.psf.enabled = False
    scaffold.noise.frozen = True

    manifest = learned_operator_manifest(scaffold, families=("psf", "noise"))

    assert manifest.families == ("psf", "noise")
    assert tuple(slot.operator_name for slot in manifest.slots) == (
        "learned_psf",
        "learned_noise",
    )
    assert tuple(slot.enabled for slot in manifest.slots) == (False, True)
    assert tuple(slot.frozen for slot in manifest.slots) == (False, True)
    assert manifest.conditioning_union.detector_coordinates
    assert manifest.conditioning_union.time
    assert not manifest.conditioning_union.wavelength
    assert manifest.conditioning_union.exposure_metadata_keys == (
        "exposure_id",
        "thermal_state",
    )


def test_swapping_learned_ops_preserves_target_render_contract() -> None:
    target = TargetRenderSpec.from_header(
        _target_header(),
        value_semantics="flux",
        band_label="native",
        transmission_center=5500.0,
    )
    bundle = ObservationOperatorBundle(
        geometry=InterpolationGeometryOp(),
        background=ExplicitBackgroundOp(),
        noise=ExplicitNoiseOp(),
    )
    scaffold = make_minimal_learned_operator_scaffold()

    swapped = swap_bundle_operator_families(
        bundle,
        scaffold,
        families=("background", "noise"),
    )

    assert swapped.geometry is bundle.geometry
    assert swapped.background is scaffold.background
    assert swapped.noise is scaffold.noise
    assert target == TargetRenderSpec.from_header(
        _target_header(),
        value_semantics="flux",
        band_label="native",
        transmission_center=5500.0,
    )
