"""Sample partitions retain independent full-detector calibration evidence."""

from copy import deepcopy

import pytest

from torchsky.calibration_partitions import prepare_exposure_calibration_partitions


def manifest():
    return {
        "extraction": {"settings": {"psf_radius_px": 2, "background_radius_px": 3}},
        "evidence": [
            {
                "exposure": "a.fits",
                "band": "F090W",
                "role": role,
                "source_id": f"{role}{i}",
                "detector_x": i * 100,
                "detector_y": y,
                "split": "test",
            }
            for role, y in (("psf", 0), ("background", 100))
            for i in range(3)
        ],
    }


def test_independent_partitions_preserve_science_and_order():
    original = manifest()
    before = deepcopy(original)
    prepared = prepare_exposure_calibration_partitions(original)
    assert original == before
    assert all(row["split"] == "test" for row in prepared["evidence"])
    for role in ("psf", "background"):
        assert {
            r["calibration_split"] for r in prepared["evidence"] if r["role"] == role
        } == {"train", "validation", "test"}
    original["evidence"].reverse()
    reversed_result = prepare_exposure_calibration_partitions(original)
    assert sorted(prepared["evidence"], key=lambda r: r["source_id"]) == sorted(
        reversed_result["evidence"], key=lambda r: r["source_id"]
    )
    assert (
        prepared["calibration_partitioning"]
        == reversed_result["calibration_partitioning"]
    )
    assert prepared["evidence"][0]["calibration_bounds_xy"] == [-3, 3, -3, 3]


def test_overlap_source_identity_and_transitive_cross_role_groups():
    payload = manifest()
    rows = payload["evidence"]
    rows.extend(
        [
            {**rows[0], "role": "background", "source_id": "near", "detector_x": 6},
            {**rows[0], "source_id": "near", "detector_x": 500},
            {**rows[0], "source_id": "last", "detector_x": 506},
        ]
    )
    result = prepare_exposure_calibration_partitions(payload)
    connected = [result["evidence"][i] for i in (0, 6, 7, 8)]
    assert len({r["calibration_group_id"] for r in connected}) == 1
    assert len({r["calibration_split"] for r in connected}) == 1


def test_sparse_exposure_reported_without_poisoning_supported_exposure():
    payload = manifest()
    payload["evidence"].append({**payload["evidence"][0], "exposure": "sparse.fits"})
    summary = prepare_exposure_calibration_partitions(payload)[
        "calibration_partitioning"
    ]
    assert summary["supported_exposures"] == [{"exposure": "a.fits", "band": "F090W"}]
    assert summary["excluded_exposures"][0]["exposure"] == "sparse.fits"
    assert all(
        r["reason"] == "insufficient_independent_groups"
        for r in summary["coverage"]
        if r["exposure"] == "sparse.fits"
    )


def test_exposure_and_band_scoping_and_exposure_id_alias():
    payload = manifest()
    row = payload["evidence"][0]
    payload["evidence"].extend(
        [{**row, "band": "F200W"}, {**row, "exposure": "b.fits"}]
    )
    del payload["evidence"][-1]["exposure"]
    payload["evidence"][-1]["exposure_id"] = "b.fits"
    result = prepare_exposure_calibration_partitions(payload)
    assert len({result["evidence"][i]["calibration_group_id"] for i in (0, 6, 7)}) == 3


def test_invalid_coordinates_and_sample_ceiling():
    payload = manifest()
    payload["evidence"][0]["detector_x"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        prepare_exposure_calibration_partitions(payload)
    payload = manifest()
    payload["evidence"] *= 22
    with pytest.raises(ValueError, match="128"):
        prepare_exposure_calibration_partitions(payload)


def test_explicit_identity_precedes_product_path_and_band_is_normalized():
    payload = manifest()
    first = payload["evidence"][0]
    first["exposure_id"] = "exact-id"
    payload["evidence"].append(
        {**first, "exposure": "alternate-path.fits", "band": "f090w"}
    )
    result = prepare_exposure_calibration_partitions(payload)
    assert (
        result["evidence"][0]["calibration_group_id"]
        == result["evidence"][-1]["calibration_group_id"]
    )
    coverage = result["calibration_partitioning"]["coverage"]
    assert all(r["band"] == "F090W" for r in coverage)
    assert any(r["exposure"] == "exact-id" and r["sample_count"] == 2 for r in coverage)
