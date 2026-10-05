"""Pinned-catalogue photometric-redshift pilot; requires optional Polars/Arrow.

Application-selected memberships and spectroscopic labels are explicit inputs.
An immutable object-sorted Parquet cache supplies sequential bounded batches.
Global cache construction/validation still needs suitable memory and scratch disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import cast

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn

from torchsky.encoders import BinnedRedshiftHead, PhotometryEncoder


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect(frame):
    return frame.collect(engine="streaming")


def reject_rows(frame, condition, message):
    if collect(frame.filter(condition).limit(1)).height:
        raise ValueError(message)


class ReleaseDataset:
    """Immutable object-sorted feature cache with sequential labelled batches."""

    def __init__(
        self,
        release,
        memberships,
        labels=None,
        *,
        batch_size=128,
        max_measurements=1024,
        feature_cache=None,
    ):
        if batch_size <= 0 or max_measurements <= 0:
            raise ValueError("batch_size and max_measurements must be positive")
        root = Path(release).resolve()
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest.get(
            "schema_version"
        ) != "xmatch.observations.release.v1" or not manifest.get("sources"):
            raise ValueError("unsupported or empty observation release")
        self.photometry, inventories = [], []
        vocabularies = {"passbands": set(), "surveys": set(), "methods": set()}
        for source in manifest["sources"]:
            namespace = source["release_namespace"]
            if namespace in vocabularies["surveys"]:
                raise ValueError("duplicate release namespace")
            vocabularies["surveys"].add(namespace)
            for mapping in source["photometry"]:
                vocabularies["passbands"].add((namespace, mapping["passband"]))
                vocabularies["methods"].add(
                    (namespace, mapping.get("method", "unknown"))
                )
            for kind in ("inventory", "photometry", "evidence"):
                relative = Path(source[f"{kind}_path"])
                path = (root / relative).resolve()
                if relative.is_absolute() or not path.is_relative_to(root):
                    raise ValueError("release shard path escapes release directory")
                if sha256(path) != source[f"{kind}_sha256"].removeprefix("sha256:"):
                    raise ValueError("release shard checksum mismatch")
                frame = pl.scan_parquet(path)
                if (
                    collect(frame.select(pl.len())).item() != source[f"{kind}_count"]
                    or {
                        key: str(value) for key, value in frame.collect_schema().items()
                    }
                    != source[f"{kind}_schema"]
                ):
                    raise ValueError("release shard schema/count mismatch")
                if kind == "photometry":
                    self.photometry.append(frame)
                elif kind == "inventory":
                    inventories.append(frame.select("source_id"))
        if not vocabularies["passbands"]:
            raise ValueError("release has no declared photometry passbands")
        self.vocabularies = {key: sorted(value) for key, value in vocabularies.items()}
        self.indices = {
            key: {value: index for index, value in enumerate(values)}
            for key, values in self.vocabularies.items()
        }
        members = pl.scan_parquet(memberships).select(
            "source_id", "object_id", "split", "split_group"
        )
        for name, dtype in members.collect_schema().items():
            if dtype != pl.String:
                raise ValueError(f"membership {name} must be a string")
        reject_rows(
            members,
            pl.any_horizontal(pl.all().is_null() | (pl.all().str.len_chars() == 0)),
            "membership identities must be nonempty",
        )
        reject_rows(
            members,
            ~pl.col("split").is_in(["train", "validation", "test"]),
            "unknown split",
        )
        if collect(members.select(pl.len() != pl.col("source_id").n_unique())).item():
            raise ValueError("native source memberships must be unique")
        reject_rows(
            members.group_by("object_id").agg(
                pl.col("split").n_unique().alias("splits"),
                pl.col("split_group").n_unique().alias("groups"),
            ),
            (pl.col("splits") != 1) | (pl.col("groups") != 1),
            "object crosses split or lineage group",
        )
        reject_rows(
            members.group_by("split_group").agg(
                pl.col("split").n_unique().alias("splits")
            ),
            pl.col("splits") != 1,
            "lineage group crosses splits",
        )
        if collect(
            members.join(pl.concat(inventories), on="source_id", how="anti").limit(1)
        ).height:
            raise ValueError("membership source absent from release inventory")
        self.labels = None
        if labels is not None:
            self.labels = pl.scan_parquet(labels).select(
                "object_id",
                pl.col("redshift").cast(pl.Float64),
                "evidence_id",
                "category",
            )
            reject_rows(
                self.labels,
                pl.col("object_id").is_null()
                | pl.col("evidence_id").is_null()
                | (pl.col("evidence_id").str.len_chars() == 0)
                | pl.col("category").is_null()
                | (pl.col("category") != "inferred_spectroscopy")
                | ~pl.col("redshift").is_finite()
                | pl.col("redshift").is_null(),
                "labels require finite spectroscopic redshift and evidence identity",
            )
            if collect(
                self.labels.select(pl.len() != pl.col("object_id").n_unique())
            ).item():
                raise ValueError("select one label evidence per object explicitly")
            if collect(
                self.labels.join(
                    members.select("object_id").unique(), on="object_id", how="anti"
                ).limit(1)
            ).height:
                raise ValueError("label object absent from memberships")
        self.metadata = {
            "release_id": manifest["release_id"],
            "manifest_sha256": sha256(manifest_path),
            "memberships_sha256": sha256(memberships),
            "labels_sha256": sha256(labels) if labels else None,
            "label_policy": "inferred_spectroscopy",
            "vocabularies": self.vocabularies,
        }
        self.batch_size, self.max_measurements = batch_size, max_measurements
        self.workspace = tempfile.TemporaryDirectory(prefix="catalogue-photoz-")
        self.feature_cache = (
            Path(feature_cache)
            if feature_cache
            else Path(self.workspace.name) / "features"
        )
        self.feature_path = self.feature_cache / "features.parquet"
        self._prepare_feature_cache(members)
        self.sorted_labels = None
        if self.labels is not None:
            self.sorted_labels = Path(self.workspace.name) / "labels.parquet"
            self.labels.sort("object_id").sink_parquet(
                self.sorted_labels, engine="streaming"
            )

    def _prepare_feature_cache(self, members):
        columns = [
            "source_id",
            "release_namespace",
            "measurement_id",
            "passband",
            "flux_jy",
            "flux_error_jy",
            "upper_limit_jy",
            "status",
            "measurement_method",
            "observation_time",
            "time_format",
        ]
        selection = {
            "columns": columns,
            "statuses": ["measured", "upper_limit"],
            "epoch_format": "jyear",
            "units": "Jy",
            "schema_version": "torchsky.catalogue-photoz.features.v1",
        }
        inputs = {
            "manifest_sha256": self.metadata["manifest_sha256"],
            "memberships_sha256": self.metadata["memberships_sha256"],
            "selection": selection,
        }
        identity = hashlib.sha256(
            json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        manifest_path = self.feature_cache / "manifest.json"
        if not self.feature_cache.exists():
            self.feature_cache.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix=f".{self.feature_cache.name}.", dir=self.feature_cache.parent
            ) as directory:
                stage = Path(directory)
                data = stage / "features.parquet"
                photometry = pl.concat(
                    [
                        frame.filter(
                            pl.col("status").is_in(selection["statuses"])
                        ).select(columns)
                        for frame in self.photometry
                    ],
                    how="vertical_relaxed",
                )
                members.join(photometry, on="source_id", how="left").sort(
                    "object_id", "measurement_id", "source_id"
                ).sink_parquet(data, engine="streaming")
                frame = pl.scan_parquet(data)
                metadata = {
                    "identity": identity,
                    "inputs": inputs,
                    "data_sha256": sha256(data),
                    "count": collect(frame.select(pl.len())).item(),
                    "schema": {
                        key: str(value) for key, value in frame.collect_schema().items()
                    },
                }
                (stage / "manifest.json").write_text(
                    json.dumps(metadata, sort_keys=True, indent=2) + "\n"
                )
                os.rename(stage, self.feature_cache)
        if not manifest_path.is_file() or self.feature_path.is_symlink():
            raise ValueError("invalid feature cache manifest/data")
        metadata = json.loads(manifest_path.read_text())
        if metadata.get("identity") != identity or metadata.get("inputs") != inputs:
            raise ValueError(
                "feature cache input identity mismatch; choose a new directory"
            )
        if sha256(self.feature_path) != metadata.get("data_sha256"):
            raise ValueError("feature cache checksum mismatch")
        frame = pl.scan_parquet(self.feature_path)
        if collect(frame.select(pl.len())).item() != metadata.get("count") or {
            key: str(value) for key, value in frame.collect_schema().items()
        } != metadata.get("schema"):
            raise ValueError("feature cache schema/count mismatch")
        self.metadata["feature_cache"] = {
            "identity": identity,
            "data_sha256": metadata["data_sha256"],
            "selection": selection,
        }

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.workspace.cleanup()

    def batches(self, split=None):
        objects, current = [], []
        sources, measurements = set(), set()
        label_rows = (
            iter(())
            if self.sorted_labels is None
            else (
                row
                for batch in pq.ParquetFile(self.sorted_labels).iter_batches(
                    batch_size=self.batch_size
                )
                for row in batch.to_pylist()
            )
        )
        label = next(label_rows, None)

        def assemble(groups):
            nonlocal label
            targets = []
            for rows in groups:
                object_id = rows[0]["object_id"]
                while label is not None and label["object_id"] < object_id:
                    label = next(label_rows, None)
                targets.append(
                    label["redshift"]
                    if label is not None and label["object_id"] == object_id
                    else float("nan")
                )
            return self._batch(groups, targets)

        for arrow_batch in pq.ParquetFile(self.feature_path).iter_batches(
            batch_size=self.batch_size
        ):
            for row in arrow_batch.to_pylist():
                if current and row["object_id"] != current[0]["object_id"]:
                    objects.append(current)
                    current = []
                    sources, measurements = set(), set()
                    if len(objects) == self.batch_size:
                        yield assemble(objects)
                        objects = []
                if split is None or row["split"] == split:
                    sources.add(row["source_id"])
                    if len(sources) > self.max_measurements:
                        raise ValueError("object source count exceeds max_measurements")
                    if row["measurement_id"] is not None:
                        if row["measurement_id"] in measurements:
                            raise ValueError("duplicate measurement identity")
                        measurements.add(row["measurement_id"])
                        if len(measurements) > self.max_measurements:
                            raise ValueError(
                                "object photometry exceeds max_measurements; no truncation performed"
                            )
                    current.append(row)
        if current:
            objects.append(current)
        if objects:
            yield assemble(objects)

    def _batch(self, objects, targets):
        object_ids = [rows[0]["object_id"] for rows in objects]
        token_rows = [
            [row for row in rows if row["measurement_id"] is not None]
            for rows in objects
        ]
        width = max(map(len, token_rows), default=0)
        features = {
            key: torch.full((len(objects), width), float("nan"))
            for key in ("flux", "error", "epoch")
        }
        features.update(
            {
                key: torch.zeros((len(objects), width), dtype=torch.long)
                for key in ("passband_id", "survey_id", "method_id", "status")
            }
        )
        features["mask"] = torch.zeros((len(objects), width), dtype=torch.bool)
        for index, rows in enumerate(token_rows):
            for column, row in enumerate(
                sorted(rows, key=lambda value: value["measurement_id"])
            ):
                namespace = row["release_namespace"]
                limit = row["status"] == "upper_limit"
                features["flux"][index, column] = (
                    row["upper_limit_jy"] if limit else row["flux_jy"]
                )
                features["error"][index, column] = (
                    row["flux_error_jy"]
                    if row["flux_error_jy"] is not None
                    else float("nan")
                )
                features["epoch"][index, column] = (
                    row["observation_time"]
                    if row["time_format"] == "jyear"
                    and row["observation_time"] is not None
                    else float("nan")
                )
                features["passband_id"][index, column] = self.indices["passbands"][
                    (namespace, row["passband"])
                ]
                features["survey_id"][index, column] = self.indices["surveys"][
                    namespace
                ]
                features["method_id"][index, column] = self.indices["methods"][
                    (namespace, row["measurement_method"])
                ]
                features["status"][index, column] = int(limit)
                features["mask"][index, column] = True
        redshift = torch.tensor(targets)
        return {
            "object_ids": object_ids,
            "splits": [rows[0]["split"] for rows in objects],
            "features": features,
            "redshift": redshift,
            "label_mask": torch.isfinite(redshift),
        }


def make_model(configuration, vocabularies):
    return nn.ModuleDict(
        {
            "encoder": PhotometryEncoder(
                len(vocabularies["passbands"]),
                len(vocabularies["surveys"]),
                num_methods=len(vocabularies["methods"]),
                flux_scale=configuration["flux_scale_jy"],
                latent_dim=configuration["latent_dim"],
            ),
            "head": BinnedRedshiftHead(
                configuration["latent_dim"], configuration["bin_edges"]
            ),
        }
    )


def atomic_write(path, writer):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        writer(temporary)
        os.link(
            temporary, path
        )  # atomic publication without replacing an existing artifact
    finally:
        temporary.unlink(missing_ok=True)


def train(dataset, args):
    if (
        dataset.labels is None
        or args.epochs <= 0
        or args.latent_dim <= 0
        or not math.isfinite(args.learning_rate)
        or args.learning_rate <= 0
        or not math.isfinite(args.flux_scale_jy)
        or args.flux_scale_jy <= 0
    ):
        raise ValueError("training needs labels and positive finite configuration")
    torch.manual_seed(args.seed)
    configuration = {
        "latent_dim": args.latent_dim,
        "bin_edges": [float(value) for value in args.bin_edges.split(",")],
        "flux_scale_jy": [args.flux_scale_jy] * len(dataset.vocabularies["passbands"]),
    }
    model = make_model(configuration, dataset.vocabularies)
    head = cast(BinnedRedshiftHead, model["head"])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history = []
    for _ in range(args.epochs):
        loss_sum, count = 0.0, 0
        for batch in dataset.batches("train"):
            latent, supported = model["encoder"](**batch["features"])
            selected = supported & batch["label_mask"]
            if not selected.any():
                continue
            logits = model["head"](latent[selected], supported=supported[selected])
            loss = -head.log_prob(logits, batch["redshift"][selected]).mean()
            if not torch.isfinite(loss):
                raise ValueError(
                    "training label outside redshift bins or nonfinite loss"
                )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * selected.sum().item()
            count += selected.sum().item()
        if not count:
            raise ValueError("no supported spectroscopically labelled training objects")
        history.append(loss_sum / count)
    checkpoint = {
        "schema_version": "torchsky.catalogue-photoz.v2",
        "configuration": configuration,
        "metadata": dataset.metadata,
        "seed": args.seed,
        "training_nll": history,
        "state_dict": model.state_dict(),
    }
    atomic_write(args.checkpoint, lambda path: torch.save(checkpoint, path))
    return checkpoint


def load_model(dataset, checkpoint):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved.get("schema_version") != "torchsky.catalogue-photoz.v2":
        raise ValueError("unsupported model checkpoint")
    for name in (
        "manifest_sha256",
        "memberships_sha256",
        "vocabularies",
        "feature_cache",
    ):
        if saved["metadata"][name] != dataset.metadata[name]:
            raise ValueError(f"checkpoint input identity mismatch: {name}")
    if (
        dataset.labels is not None
        and saved["metadata"]["labels_sha256"] != dataset.metadata["labels_sha256"]
    ):
        raise ValueError("checkpoint label snapshot mismatch")
    model = make_model(saved["configuration"], dataset.vocabularies)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model, saved


def predict(dataset, model, checkpoint, output, split=None):
    schema = pa.schema(
        [
            ("object_id", pa.string()),
            ("split", pa.string()),
            ("supported", pa.bool_()),
            ("p_z", pa.list_(pa.float64())),
            ("z_mean", pa.float64()),
        ],
        metadata={
            b"checkpoint_sha256": sha256(checkpoint).encode(),
            b"bin_edges": json.dumps(model["head"].bin_edges.tolist()).encode(),
            b"input_release": json.dumps(dataset.metadata, sort_keys=True).encode(),
        },
    )

    def write(path):
        with pq.ParquetWriter(path, schema) as writer, torch.no_grad():
            for batch in dataset.batches(split):
                latent, supported = model["encoder"](**batch["features"])
                probabilities = model["head"](
                    latent[supported], supported=supported[supported]
                ).softmax(-1)
                centers = (
                    model["head"].bin_edges[:-1] + model["head"].bin_edges[1:]
                ) / 2
                estimates = probabilities @ centers
                rows, selected = [], 0
                for index, object_id in enumerate(batch["object_ids"]):
                    rows.append(
                        {
                            "object_id": object_id,
                            "split": batch["splits"][index],
                            "supported": bool(supported[index]),
                            "p_z": probabilities[selected].tolist()
                            if supported[index]
                            else None,
                            "z_mean": estimates[selected].item()
                            if supported[index]
                            else None,
                        }
                    )
                    selected += int(supported[index])
                writer.write_table(pa.Table.from_pylist(rows, schema=schema))

    atomic_write(output, write)


def evaluate(dataset, model, split):
    totals = {
        "nll": 0.0,
        "residual": 0.0,
        "residual_square": 0.0,
        "catastrophic": 0,
        "central_50": 0,
        "central_90": 0,
    }
    count, unsupported, unlabelled = 0, 0, 0
    pit_histogram = torch.zeros(10, dtype=torch.long)
    with torch.no_grad():
        for batch in dataset.batches(split):
            latent, supported = model["encoder"](**batch["features"])
            unsupported += int((~supported).sum())
            unlabelled += int((supported & ~batch["label_mask"]).sum())
            selected = supported & batch["label_mask"]
            if not selected.any():
                continue
            redshift = batch["redshift"][selected]
            if (redshift <= -1).any():
                raise ValueError("normalized photo-z residual needs redshift > -1")
            logits = model["head"](latent[selected], supported=supported[selected])
            log_density = model["head"].log_prob(logits, redshift)
            if not torch.isfinite(log_density).all():
                raise ValueError("evaluation label outside redshift bin support")
            probability, edges = logits.softmax(-1), model["head"].bin_edges
            residual = (probability @ ((edges[:-1] + edges[1:]) / 2) - redshift) / (
                1 + redshift
            )
            bins = (
                torch.bucketize(redshift.contiguous(), edges, right=True) - 1
            ).clamp(0, probability.shape[-1] - 1)
            cdf_left = probability.cumsum(-1) - probability
            pit = cdf_left.gather(-1, bins[:, None]).squeeze(-1) + probability.gather(
                -1, bins[:, None]
            ).squeeze(-1) * ((redshift - edges[bins]) / edges.diff()[bins])
            pit_histogram += torch.histc(pit.clamp(0, 1), bins=10, min=0, max=1).long()
            totals["nll"] += float(-log_density.sum())
            totals["residual"] += float(residual.sum())
            totals["residual_square"] += float(residual.square().sum())
            totals["catastrophic"] += int((residual.abs() > 0.15).sum())
            totals["central_50"] += int(((pit >= 0.25) & (pit <= 0.75)).sum())
            totals["central_90"] += int(((pit >= 0.05) & (pit <= 0.95)).sum())
            count += len(redshift)
    if not count:
        raise ValueError("no supported labelled evaluation objects")
    return {
        "split": split,
        "evaluated_objects": count,
        "unsupported_objects": unsupported,
        "unlabelled_supported_objects": unlabelled,
        "mean_nll": totals["nll"] / count,
        "mean_normalized_residual": totals["residual"] / count,
        "rms_normalized_residual": math.sqrt(totals["residual_square"] / count),
        "catastrophic_fraction": totals["catastrophic"] / count,
        "catastrophic_threshold": 0.15,
        "central_50_coverage": totals["central_50"] / count,
        "central_90_coverage": totals["central_90"] / count,
        "pit_histogram_edges": [index / 10 for index in range(11)],
        "pit_histogram_counts": pit_histogram.tolist(),
        "calibration_claim": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "predict", "evaluate"])
    for name in ("release", "memberships", "checkpoint"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split", choices=["train", "validation", "test"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--max-measurements", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--flux-scale-jy", type=float, default=1e-5)
    parser.add_argument("--bin-edges", default="0,0.1,0.2,0.4,0.8,1.6,3.2,6.4")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.command in ("train", "evaluate") and args.labels is None:
        parser.error("train/evaluate requires --labels")
    if args.command != "train" and args.output is None:
        parser.error("predict/evaluate requires --output")
    if args.command == "evaluate" and args.split not in ("validation", "test"):
        parser.error("evaluate requires --split validation or test")
    torch.set_num_threads(1)
    with ReleaseDataset(
        args.release,
        args.memberships,
        args.labels,
        batch_size=args.batch_size,
        max_measurements=args.max_measurements,
        feature_cache=args.feature_cache,
    ) as dataset:
        if args.command == "train":
            return train(dataset, args)
        model, _ = load_model(dataset, args.checkpoint)
        if args.command == "predict":
            predict(dataset, model, args.checkpoint, args.output, args.split)
            return None
        metrics = evaluate(dataset, model, args.split)
        metrics.update(
            {"input": dataset.metadata, "checkpoint_sha256": sha256(args.checkpoint)}
        )
        atomic_write(
            args.output,
            lambda path: path.write_text(
                json.dumps(metrics, sort_keys=True, indent=2, allow_nan=False) + "\n"
            ),
        )
        return metrics


if __name__ == "__main__":
    main()
