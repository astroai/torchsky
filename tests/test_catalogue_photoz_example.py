"""Small application benchmark checks; no real-survey accuracy claim."""

import hashlib
import importlib.util
import json
from pathlib import Path

import polars as pl
import pytest
import torch


def _example():
    path = Path(__file__).parents[1] / "examples" / "catalogue_photoz.py"
    assert path.exists(), "Missing runnable catalogue photo-z example"
    spec = importlib.util.spec_from_file_location("catalogue_photoz", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path):
    release = tmp_path / "release"
    release.mkdir()
    memberships, labels, measurements, inventory = [], [], [], []
    for index in range(25):
        object_id = f"object-{index:03}"
        split = "train" if index < 16 else "validation" if index < 20 else "test"
        high = index % 2
        memberships.append(
            {
                "source_id": f"s{index}",
                "object_id": object_id,
                "split": split,
                "split_group": f"group-{index}",
            }
        )
        inventory.append({"source_id": f"s{index}"})
        labels.append(
            {
                "object_id": object_id,
                "redshift": 1.25 if high else 0.25,
                "evidence_id": f"spectrum-{index}",
                "category": "inferred_spectroscopy",
            }
        )
        for band in ("g", "r"):
            missing = index == 24
            measurements.append(
                {
                    "source_id": f"s{index}",
                    "release_namespace": "survey/dr1",
                    "measurement_id": f"measurement-{index}-{band}",
                    "passband": band,
                    "flux_jy": None
                    if missing
                    else (5e-5 if high else (-1e-6 if band == "g" else 5e-6)),
                    "flux_error_jy": None if missing else 1e-6,
                    "upper_limit_jy": None,
                    "status": "missing" if missing else "measured",
                    "measurement_method": "psf",
                    "observation_time": None,
                    "time_format": None,
                    "time_scale": None,
                    "redshift_proxy_that_must_not_be_used": 1000.0 + index,
                }
            )
    entry = {
        "release_namespace": "survey/dr1",
        "photometry": [{"passband": band, "method": "psf"} for band in ("g", "r")],
    }
    for kind, rows in (
        ("inventory", inventory),
        ("photometry", measurements),
        ("evidence", [{"source_id": "s0"}]),
    ):
        path = release / f"{kind}.parquet"
        pl.DataFrame(rows).write_parquet(path)
        entry[f"{kind}_path"] = path.name
        entry[f"{kind}_sha256"] = "sha256:" + _sha(path)
        entry[f"{kind}_count"] = len(rows)
        entry[f"{kind}_schema"] = {
            key: str(dtype)
            for key, dtype in pl.scan_parquet(path).collect_schema().items()
        }
    (release / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "xmatch.observations.release.v1",
                "release_id": "synthetic-example-only",
                "sources": [entry],
            }
        )
    )
    member_path, label_path = (
        tmp_path / "memberships.parquet",
        tmp_path / "labels.parquet",
    )
    pl.DataFrame(memberships).write_parquet(member_path)
    pl.DataFrame(labels).write_parquet(label_path)
    return release, member_path, label_path


def test_train_predict_evaluate_learns_and_preserves_unsupported_objects(
    tmp_path,
) -> None:
    example = _example()
    release, memberships, labels = _fixture(tmp_path)
    checkpoint, predictions, metrics = (
        tmp_path / "model.pt",
        tmp_path / "predictions.parquet",
        tmp_path / "metrics.json",
    )
    common = [
        "--release",
        str(release),
        "--memberships",
        str(memberships),
        "--checkpoint",
        str(checkpoint),
        "--batch-size",
        "8",
    ]
    example.main(
        [
            "train",
            *common,
            "--labels",
            str(labels),
            "--epochs",
            "25",
            "--learning-rate",
            "0.02",
            "--bin-edges",
            "0,0.5,1.5",
            "--latent-dim",
            "16",
        ]
    )
    saved = torch.load(checkpoint, weights_only=True)
    assert saved["training_nll"][-1] < saved["training_nll"][0] - 0.15
    assert saved["metadata"]["label_policy"] == "inferred_spectroscopy"
    example.main(["predict", *common, "--output", str(predictions)])
    result = pl.read_parquet(predictions)
    assert result.height == 25
    unsupported = result.filter(~pl.col("supported"))
    assert unsupported["object_id"].to_list() == ["object-024"]
    assert unsupported["p_z"].to_list() == [None]
    for probability in result.filter(pl.col("supported"))["p_z"].to_list():
        assert sum(probability) == pytest.approx(1.0, abs=1e-6)
    example.main(
        [
            "evaluate",
            *common,
            "--labels",
            str(labels),
            "--output",
            str(metrics),
            "--split",
            "test",
        ]
    )
    measured = json.loads(metrics.read_text())
    assert measured["evaluated_objects"] == 4
    assert measured["unsupported_objects"] == 1
    assert measured["catastrophic_fraction"] == 0.0
    assert 0 <= measured["central_90_coverage"] <= 1
    assert sum(measured["pit_histogram_counts"]) == 4
    assert measured["pit_histogram_edges"][0] == 0.0
    assert measured["pit_histogram_edges"][-1] == 1.0
    second = tmp_path / "predictions-again.parquet"
    example.main(["predict", *common, "--output", str(second)])
    assert pl.read_parquet(second).equals(result)
    with example.ReleaseDataset(release, memberships, labels, batch_size=2) as dataset:
        model, _ = example.load_model(dataset, checkpoint)
        batch = next(dataset.batches("test"))
        permuted = {
            key: value[:, torch.tensor([1, 0])]
            for key, value in batch["features"].items()
        }
        original, supported = model["encoder"](**batch["features"])
        shuffled, _ = model["encoder"](**permuted)
        torch.testing.assert_close(
            model["head"](original, supported=supported),
            model["head"](shuffled, supported=supported),
        )
    pl.read_parquet(labels).with_columns(
        (pl.col("redshift") + 0.01).alias("redshift")
    ).write_parquet(labels)
    with pytest.raises(ValueError, match="label snapshot"):
        example.main(
            [
                "evaluate",
                *common,
                "--labels",
                str(labels),
                "--output",
                str(tmp_path / "altered-metrics.json"),
                "--split",
                "test",
            ]
        )


def test_loader_is_batched_and_features_do_not_contain_labels(tmp_path) -> None:
    example = _example()
    release, memberships, labels = _fixture(tmp_path)
    with example.ReleaseDataset(
        release, memberships, labels, batch_size=2, max_measurements=3
    ) as dataset:
        batches = list(dataset.batches("test"))
        assert [len(batch["object_ids"]) for batch in batches] == [2, 2, 1]
        first = batches[0]
        assert set(first["features"]) == {
            "flux",
            "error",
            "passband_id",
            "survey_id",
            "method_id",
            "status",
            "epoch",
            "mask",
        }
        assert first["features"]["flux"].shape == (2, 2)
        assert first["label_mask"].all()
        baseline = {key: value.clone() for key, value in first["features"].items()}
    altered = tmp_path / "altered-labels.parquet"
    pl.read_parquet(labels).with_columns(
        (pl.col("redshift") + 0.01).alias("redshift")
    ).write_parquet(altered)
    with example.ReleaseDataset(
        release, memberships, altered, batch_size=2, max_measurements=3
    ) as dataset:
        changed = next(dataset.batches("test"))
        for key, value in baseline.items():
            torch.testing.assert_close(value, changed["features"][key], equal_nan=True)


@pytest.mark.parametrize(
    "problem",
    ["source_duplicate", "group_leak", "object_leak", "unknown_source", "bad_category"],
)
def test_split_identity_and_label_boundaries_fail_closed(tmp_path, problem) -> None:
    example = _example()
    release, memberships, labels = _fixture(tmp_path)
    members = pl.read_parquet(memberships)
    if problem == "source_duplicate":
        members = pl.concat([members, members.head(1)])
    elif problem == "group_leak":
        members = members.with_columns(
            pl.when(pl.col("object_id") == "object-020")
            .then(pl.lit("group-0"))
            .otherwise(pl.col("split_group"))
            .alias("split_group")
        )
    elif problem == "object_leak":
        members = members.with_columns(
            pl.when(pl.col("object_id") == "object-020")
            .then(pl.lit("object-000"))
            .otherwise(pl.col("object_id"))
            .alias("object_id")
        )
    elif problem == "unknown_source":
        members = members.with_columns(
            pl.when(pl.col("source_id") == "s0")
            .then(pl.lit("unknown-source"))
            .otherwise(pl.col("source_id"))
            .alias("source_id")
        )
    else:
        pl.read_parquet(labels).with_columns(
            pl.lit("prediction").alias("category")
        ).write_parquet(labels)
    members.write_parquet(memberships)
    with pytest.raises(ValueError):
        example.ReleaseDataset(release, memberships, labels)


def test_measurement_cap_and_release_checksum_fail_closed(tmp_path) -> None:
    example = _example()
    release, memberships, labels = _fixture(tmp_path)
    with example.ReleaseDataset(
        release, memberships, labels, max_measurements=1
    ) as dataset:
        with pytest.raises(ValueError, match="max_measurements"):
            next(dataset.batches("train"))
    with (release / "photometry.parquet").open("ab") as stream:
        stream.write(b"altered")
    with pytest.raises(ValueError, match="checksum"):
        example.ReleaseDataset(release, memberships, labels)


def test_missing_label_keeps_photometry_and_false_target_mask(tmp_path) -> None:
    example = _example()
    release, memberships, labels = _fixture(tmp_path)
    pl.read_parquet(labels).filter(pl.col("object_id") != "object-020").write_parquet(
        labels
    )
    with example.ReleaseDataset(release, memberships, labels, batch_size=2) as dataset:
        batch = next(dataset.batches("test"))
        assert batch["label_mask"].tolist() == [False, True]
        assert torch.isnan(batch["redshift"][0])
        assert batch["features"]["mask"].all()


def test_feature_cache_reuse_and_epochs_do_not_reopen_survey_shards(tmp_path) -> None:
    example = _example()
    release, memberships, labels = _fixture(tmp_path)
    cache = tmp_path / "feature-cache"
    with example.ReleaseDataset(
        release, memberships, labels, feature_cache=cache, batch_size=1
    ) as dataset:
        first = list(dataset.batches("test"))
        assert len(first) == 5
        assert first[0]["features"]["flux"].shape == (1, 2)
        identity = dataset.metadata["feature_cache"]
        cache_stat = dataset.feature_path.stat()
        original = release / "photometry.parquet"
        parked = release / "photometry-parked.parquet"
        original.rename(parked)
        try:
            second = list(dataset.batches("test"))
        finally:
            parked.rename(original)
        for before, after in zip(first, second, strict=True):
            for key, value in before["features"].items():
                torch.testing.assert_close(
                    value, after["features"][key], equal_nan=True
                )
    with example.ReleaseDataset(
        release, memberships, labels, feature_cache=cache, batch_size=2
    ) as dataset:
        assert dataset.feature_path.stat().st_mtime_ns == cache_stat.st_mtime_ns
        assert dataset.metadata["feature_cache"] == identity
        assert sum(len(batch["object_ids"]) for batch in dataset.batches()) == 25


def test_feature_cache_rejects_changed_content_or_membership(tmp_path) -> None:
    example = _example()
    release, memberships, labels = _fixture(tmp_path)
    cache = tmp_path / "feature-cache"
    with example.ReleaseDataset(
        release, memberships, labels, feature_cache=cache
    ) as dataset:
        feature_path = dataset.feature_path
    original = feature_path.read_bytes()
    feature_path.write_bytes(original + b"changed")
    with pytest.raises(ValueError, match="cache.*checksum"):
        example.ReleaseDataset(release, memberships, labels, feature_cache=cache)
    feature_path.write_bytes(original)
    pl.read_parquet(memberships).with_columns(
        pl.when(pl.col("object_id") == "object-000")
        .then(pl.lit("revised-group"))
        .otherwise(pl.col("split_group"))
        .alias("split_group")
    ).write_parquet(memberships)
    with pytest.raises(ValueError, match="cache.*identity"):
        example.ReleaseDataset(release, memberships, labels, feature_cache=cache)
