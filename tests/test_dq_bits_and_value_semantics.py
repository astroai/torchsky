"""Regression tests for the data-quality bit contract, instrument bit tables,
``FieldObservation.value_semantics`` validation, ``FieldObservation.to`` precision and
shared-sky cropping of far-away exposures."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from torchsky.instrument import (
    ACSWFCDataQuality,
    JWSTDataQuality,
    WFC3IRDataQuality,
)
from torchsky.map.field.observation_ops import (
    crop_field_observation_to_shared_sky_patch,
)
from torchsky.map.field.types import (
    VALUE_SEMANTICS,
    FieldObservation,
    FieldValueSemantics,
)
from torchsky.map.quality import DQPolicy, QualityProduct

# ------------------------------------------------------------ 1. reject_bits meaning


def test_reject_bits_none_is_strict_and_zero_accepts_everything() -> None:
    dq = torch.tensor([0, 1, 2, 4, 2**20], dtype=torch.int64)
    assert DQPolicy().usability_mask(dq).tolist() == [True, False, False, False, False]
    assert DQPolicy(reject_bits=None).usability_mask(dq).all().item() is False
    assert DQPolicy(reject_bits=0).usability_mask(dq).all().item() is True
    assert DQPolicy(reject_bits=1).usability_mask(dq).tolist() == [
        True,
        False,
        True,
        True,
        True,
    ]


def test_named_constructors_spell_both_meanings() -> None:
    dq = torch.tensor([0, 1, 8], dtype=torch.int64)
    strict = DQPolicy.strict()
    accept = DQPolicy.accept_all()
    assert strict == DQPolicy() and strict.reject_bits is None
    assert accept.reject_bits == 0
    assert strict.usability_mask(dq).tolist() == [True, False, False]
    assert accept.usability_mask(dq).tolist() == [True, True, True]
    assert (
        DQPolicy.accept_all(unknown_bit_policy="preserve").unknown_bit_policy
        == "preserve"
    )
    assert "strict" in (DQPolicy.__doc__ or "") and "accept_all" in (
        DQPolicy.__doc__ or ""
    )


# ------------------------------------------------------- 2. signed int32 is unsigned


def test_signed_int32_plane_is_read_as_an_unsigned_bit_field() -> None:
    top = torch.tensor(
        [-(2**31), 0, 1, -1], dtype=torch.int32
    )  # bit 31, none, bit 0, all
    coerced = DQPolicy._coerce(top)
    assert coerced.dtype == torch.int64
    assert coerced.tolist() == [2**31, 0, 1, 2**32 - 1]
    # A high-bit mask above bit 31 must not see the sign extension.
    high = DQPolicy(reject_bits=1 << 40)
    assert high.usability_mask(top).tolist() == [True, True, True, True]
    # The same bit pattern as uint32 gives the same answer.
    as_uint32 = torch.tensor([2**31, 0, 1, 2**32 - 1], dtype=torch.uint32)
    assert torch.equal(DQPolicy._coerce(as_uint32), coerced)
    assert DQPolicy(reject_bits=1 << 31).usability_mask(top).tolist() == [
        False,
        True,
        True,
        False,
    ]
    assert DQPolicy(propagate_bits=1 << 31).propagated(top).tolist() == [
        2**31,
        0,
        0,
        2**31 + 4,  # the cosmic-ray bit always propagates
    ]


@pytest.mark.parametrize(
    "dtype, bits", [(torch.int8, 8), (torch.int16, 16), (torch.int32, 32)]
)
def test_every_signed_width_masks_to_its_own_width(
    dtype: torch.dtype, bits: int
) -> None:
    value = torch.tensor([-1, -(2 ** (bits - 1))], dtype=dtype)
    assert DQPolicy._coerce(value).tolist() == [2**bits - 1, 2 ** (bits - 1)]
    assert DQPolicy._coerce(torch.tensor([5, 0], dtype=dtype)).tolist() == [5, 0]


def test_quality_product_stores_the_unsigned_pattern() -> None:
    dq = torch.tensor([[-(2**31), 3]], dtype=torch.int32)
    product = QualityProduct(dq=dq, usability_mask=torch.ones(1, 2, dtype=torch.bool))
    assert product.dq.tolist() == [[2**31, 3]]


# ---------------------------------------------------------- 3. instrument bit tables


@pytest.mark.parametrize(
    "table", [JWSTDataQuality, WFC3IRDataQuality, ACSWFCDataQuality]
)
def test_bit_tables_are_single_distinct_bits(table: type) -> None:
    values = [int(member) for member in table]
    assert all(v > 0 and v & (v - 1) == 0 for v in values)
    assert len(values) == len(set(values))
    assert (
        table.__doc__
        and "Source" in table.__doc__
        or "handbook" in table.__doc__.lower()
    )


def test_jwst_table_matches_the_documented_bits() -> None:
    assert JWSTDataQuality.DO_NOT_USE == 1
    assert JWSTDataQuality.SATURATED == 2
    assert JWSTDataQuality.JUMP_DET == 4
    assert JWSTDataQuality.OUTLIER == 16
    assert JWSTDataQuality.NON_SCIENCE == 512
    assert JWSTDataQuality.REFERENCE_PIXEL == 2**31
    assert not any(int(m) == 128 for m in JWSTDataQuality)  # reserved, not named


def test_hst_tables_name_only_the_certain_bits() -> None:
    assert WFC3IRDataQuality.BAD_DETECTOR_PIXEL == 4
    assert WFC3IRDataQuality.FULL_WELL_SATURATION == 256
    assert ACSWFCDataQuality.HOT_PIXEL == 16
    assert ACSWFCDataQuality.FULL_WELL_SATURATION == 256
    assert ACSWFCDataQuality.AD_SATURATION == 2048


def test_tables_compose_into_a_dq_policy() -> None:
    reject = JWSTDataQuality.DO_NOT_USE | JWSTDataQuality.SATURATED
    policy = DQPolicy(reject_bits=reject)
    assert policy.reject_bits == 3 and type(policy.reject_bits) is int
    dq = torch.tensor([0, 1, 2, 4, 2**31, 6], dtype=torch.uint32)
    assert policy.usability_mask(dq).tolist() == [True, False, False, True, True, False]
    hst = DQPolicy(reject_bits=ACSWFCDataQuality.FULL_WELL_SATURATION | 4096)
    assert hst.usability_mask(
        torch.tensor([256, 4096, 16], dtype=torch.int16)
    ).tolist() == [
        False,
        False,
        True,
    ]


# ------------------------------------------------------------ 5. value_semantics


def _observation(**kwargs) -> FieldObservation:
    return FieldObservation(image=torch.zeros(4, 5), header={}, **kwargs)


@pytest.mark.parametrize(
    "literal",
    [
        "surface_brightness",
        "single_integrated_read",
        "detector_counts",
        "integrated_flux_per_pixel",
        "integrated_flux",
        "flux",
        "count_rate_per_pixel",
    ],
)
def test_every_literal_used_in_the_tree_is_accepted(literal: str) -> None:
    assert literal in VALUE_SEMANTICS
    assert _observation(value_semantics=literal).value_semantics == literal


def test_enum_member_is_stored_as_a_plain_string() -> None:
    obs = _observation(value_semantics=FieldValueSemantics.COUNT_RATE_PER_PIXEL)
    assert obs.value_semantics == "count_rate_per_pixel"
    assert type(obs.value_semantics) is str


def test_unknown_value_semantics_fails_at_construction_with_the_choices() -> None:
    with pytest.raises(ValueError, match="surface_brigthness") as caught:
        _observation(value_semantics="surface_brigthness")
    assert "surface_brightness" in str(caught.value)
    with pytest.raises(ValueError, match="non-empty"):
        _observation(value_semantics="")
    assert _observation().value_semantics is None


def test_value_semantics_survives_to_and_replace() -> None:
    obs = _observation(value_semantics="count_rate_per_pixel").to(dtype=torch.float64)
    assert obs.value_semantics == "count_rate_per_pixel"


# ------------------------------------------------------------ 7. .to keeps precision


class _Table(nn.Module):
    """Detector-to-sky stub with a float64 table and a float32 buffer."""

    def __init__(self) -> None:
        super().__init__()
        self.table = nn.Parameter(torch.tensor([1.0 + 1e-12, 2.0], dtype=torch.float64))
        self.register_buffer("scale", torch.tensor([0.5], dtype=torch.float32))
        self.register_buffer("index", torch.tensor([3], dtype=torch.int64))

    def pixel_to_world(self, x, y):
        return x * self.table[0], y * self.table[1]


def test_to_float32_keeps_a_float64_detector_table() -> None:
    obs = _observation(detector_to_sky=_Table())
    moved = obs.to(dtype=torch.float32)
    assert moved.image.dtype == torch.float32
    transform = moved.detector_to_sky
    assert transform.table.dtype == torch.float64
    assert float(transform.table[0].detach()) == 1.0 + 1e-12
    assert transform.index.dtype == torch.int64
    assert obs.detector_to_sky.table.dtype == torch.float64  # source untouched
    assert transform is not obs.detector_to_sky


def test_to_float64_still_widens_a_float32_transform() -> None:
    moved = _observation(detector_to_sky=_Table()).to(dtype=torch.float64)
    assert moved.detector_to_sky.scale.dtype == torch.float64
    assert moved.detector_to_sky.table.dtype == torch.float64


def test_device_move_still_applies_to_the_transform() -> None:
    moved = _observation(detector_to_sky=_Table()).to(
        device="meta", dtype=torch.float32
    )
    assert moved.detector_to_sky.table.device.type == "meta"
    assert moved.detector_to_sky.table.dtype == torch.float64
    assert moved.image.device.type == "meta"


# ----------------------------------------------- 6. shared-sky crop far from detector

_WIDTH = 32


class _FlatSky(nn.Module):
    """Pixel (x, y) maps to (lon0 + 1e-3 x, lat0 + 1e-3 y) degrees."""

    def __init__(self, lon0: float, *, fail_closed_keyword: bool = True) -> None:
        super().__init__()
        self.lon0 = lon0
        if not fail_closed_keyword:
            self.world_to_pixel = self._plain  # type: ignore[method-assign]

    def pixel_to_world(self, x, y):
        return self.lon0 + 1e-3 * x, 0.0 + 1e-3 * y

    def world_to_pixel(self, lon, lat, *, fail_closed: bool = True):
        x = (lon - self.lon0) / 1e-3
        y = lat / 1e-3
        outside = (x < -0.5) | (x > _WIDTH - 0.5) | (y < -0.5) | (y > _WIDTH - 0.5)
        if fail_closed and bool(outside.any()):
            raise ValueError("world_to_pixel did not converge")
        nan = torch.full_like(x, float("nan"))
        return torch.where(outside, nan, x), torch.where(outside, nan, y)

    def _plain(self, lon, lat):
        x = (lon - self.lon0) / 1e-3
        return x, lat / 1e-3


def _flat_observation(lon0: float, **kwargs) -> FieldObservation:
    return FieldObservation(
        image=torch.ones(_WIDTH, _WIDTH),
        header={},
        weight=torch.ones(_WIDTH, _WIDTH),
        mask=torch.ones(_WIDTH, _WIDTH, dtype=torch.bool),
        detector_to_sky=_FlatSky(lon0, **kwargs),
    )


def test_crop_far_from_the_detector_reports_none_instead_of_raising() -> None:
    reference = _flat_observation(10.0)
    far = _flat_observation(14.0)  # the reference patch is 4 degrees away
    cropped = crop_field_observation_to_shared_sky_patch(
        far, reference_observation=reference, x0=4, y0=4, patch_size=8, psf_halo=(0, 0)
    )
    assert cropped is None


def test_crop_with_overlapping_exposure_is_unchanged() -> None:
    reference = _flat_observation(10.0)
    shifted = _flat_observation(10.0 - 2e-3)  # two pixels to the right of the reference
    cropped = crop_field_observation_to_shared_sky_patch(
        shifted,
        reference_observation=reference,
        x0=4,
        y0=4,
        patch_size=8,
        psf_halo=(0, 0),
    )
    assert cropped is not None
    assert cropped.detector_x0_pix == 6.0 and cropped.detector_y0_pix == 4.0
    assert tuple(cropped.image.shape) == (8, 8)


def test_crop_still_works_when_world_to_pixel_has_no_fail_closed_keyword() -> None:
    reference = _flat_observation(10.0)
    plain = _flat_observation(10.0, fail_closed_keyword=False)
    cropped = crop_field_observation_to_shared_sky_patch(
        plain,
        reference_observation=reference,
        x0=4,
        y0=4,
        patch_size=8,
        psf_halo=(0, 0),
    )
    assert cropped is not None and cropped.image.shape[0] >= 8
