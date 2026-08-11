"""Export human-readable NRCB4 background calibration diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import matplotlib
import torch
from torch import Tensor

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import fit_smacs0723_operator_calibration as fit_calibration  # noqa: E402
from torchsky.calibration_evidence import load_operator_calibration_split  # noqa: E402
from torchsky.io import read, write  # noqa: E402
from torchsky.map.operators import (  # noqa: E402
    ConditionedBackgroundField,
    apply_operator_calibration_pruning,
)
from torchsky.viz.renderers.image import zscale_limits  # noqa: E402


def _fit_group(
    report: dict[str, Any], *, detector: str, role: str, family: str
) -> dict[str, Any]:
    matches = [
        group
        for group in report["groups"]
        if group["detector"] == detector
        and group["role"] == role
        and group["family"] == family
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one {detector}/{role}/{family} fit group")
    return matches[0]


def _load_background_model(
    manifest: Path,
    group: dict[str, Any],
    *,
    detector: str,
) -> ConditionedBackgroundField:
    train = load_operator_calibration_split(
        manifest,
        split="train",
        role="background",
        detector=detector,
        verify_hashes=True,
    )
    model, _ = fit_calibration._make_model(
        train,
        role="background",
        family=str(group["family"]),
        device=torch.device("cpu"),
    )
    if not isinstance(model, ConditionedBackgroundField):
        raise TypeError("fit checkpoint did not describe a background model")
    checkpoint = torch.load(group["checkpoint"], map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def _predict(model: ConditionedBackgroundField, arrays: dict[str, Tensor]) -> Tensor:
    parameter = next(model.parameters())
    with torch.no_grad():
        return model(
            arrays["detector_x"].to(dtype=parameter.dtype),
            arrays["detector_y"].to(dtype=parameter.dtype),
            arrays["wavelength"],
            arrays["time_mjd"],
        ).to(dtype=torch.float64)


def _prediction_rows(
    manifest: Path,
    payload: dict[str, Any],
    report: dict[str, Any],
    *,
    detector: str,
) -> list[dict[str, Any]]:
    stationary_group = _fit_group(
        report, detector=detector, role="background", family="stationary"
    )
    adaptive_group = _fit_group(
        report, detector=detector, role="background", family="adaptive"
    )
    stationary = _load_background_model(
        manifest, stationary_group, detector=detector
    )
    adaptive = _load_background_model(manifest, adaptive_group, detector=detector)
    pruned = _load_background_model(manifest, adaptive_group, detector=detector)
    apply_operator_calibration_pruning(pruned, adaptive_group["validation_pruning"])
    output: list[dict[str, Any]] = []
    for split in ("train", "validation", "test"):
        arrays = load_operator_calibration_split(
            manifest,
            split=split,
            role="background",
            detector=detector,
            verify_hashes=True,
        )
        rows = [
            row
            for row in payload["evidence"]
            if row["split"] == split
            and row["role"] == "background"
            and str(row["detector"]).upper() == detector
        ]
        if len(rows) != arrays["target_values"].numel():
            raise RuntimeError("manifest row order/count does not match loaded evidence")
        predictions = {
            "stationary": _predict(stationary, arrays),
            "adaptive": _predict(adaptive, arrays),
            "pruned": _predict(pruned, arrays),
        }
        for index, source in enumerate(rows):
            target = float(arrays["target_values"][index])
            item = dict(source)
            item.update({"status": "accepted", "target_value": target})
            for family, values in predictions.items():
                prediction = float(values[index])
                item[f"{family}_prediction"] = prediction
                item[f"{family}_residual"] = prediction - target
                item[f"{family}_absolute_error"] = abs(prediction - target)
            item["pruned_vs_stationary_absolute_error"] = (
                item["pruned_absolute_error"] - item["stationary_absolute_error"]
            )
            output.append(item)
    return output


def _exposure_lookup(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for exposure in payload["exposures"]:
        lookup[str(exposure["path"])] = exposure
        lookup[str(exposure["resolved_path"])] = exposure
        lookup[str(exposure["filename"])] = exposure
    return lookup


def _rejected_rows(
    payload: dict[str, Any], *, detector: str
) -> list[dict[str, Any]]:
    lookup = _exposure_lookup(payload)
    output = []
    for source in payload["extraction"]["rejected"]:
        if source["role"] != "background":
            continue
        exposure = lookup.get(str(source["exposure"])) or lookup.get(
            Path(str(source["exposure"])).name
        )
        if exposure is None or str(exposure["detector"]).upper() != detector:
            continue
        item = dict(source)
        item.update(
            {
                "status": "rejected",
                "band": exposure["band"],
                "detector": exposure["detector"],
                "patt_num": exposure["patt_num"],
                "split": exposure["split"],
                "time_mjd_utc": exposure["timing"]["midpoint_mjd_utc"],
            }
        )
        output.append(item)
    return output


def _selected_rows(
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    *,
    per_group: int,
) -> list[dict[str, Any]]:
    if per_group < 1:
        raise ValueError("per_group must be positive")
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in accepted + rejected:
        groups[(row["status"], row["split"], row["band"])].append(row)
    selected = []
    for (status, _split, _band), rows in sorted(groups.items()):
        if status == "accepted":
            rows.sort(
                key=lambda row: (
                    -float(row["pruned_absolute_error"]), str(row["source_id"])
                )
            )
        else:
            rows.sort(key=lambda row: (str(row["reason"]), str(row["source_id"])))
        selected.extend(rows[:per_group])
    return selected


@lru_cache(maxsize=2)
def _exposure_planes(path: str) -> tuple[Tensor, Tensor, Tensor]:
    return (
        read(path, hdu="SCI").detach().cpu(),
        read(path, hdu="ERR").detach().cpu(),
        read(path, hdu="DQ").detach().cpu(),
    )


def _cutout(
    row: dict[str, Any], exposure: dict[str, Any], *, radius: int
) -> tuple[Tensor, Tensor, Tensor]:
    science, error, dq = _exposure_planes(str(exposure["resolved_path"]))
    xmin, _xmax, ymin, _ymax = map(int, exposure["detector_bounds_xy"])
    x = round(float(row["detector_x"]) - xmin)
    y = round(float(row["detector_y"]) - ymin)
    ys = slice(y - radius, y + radius + 1)
    xs = slice(x - radius, x + radius + 1)
    planes = science[ys, xs], error[ys, xs], dq[ys, xs]
    expected = (2 * radius + 1, 2 * radius + 1)
    if any(tuple(plane.shape) != expected for plane in planes):
        raise ValueError(f"{row['source_id']}: inspection cutout crosses image boundary")
    return planes


def _write_contact_sheet(
    path: Path,
    samples: list[tuple[dict[str, Any], Tensor]],
) -> None:
    columns = 4
    rows = math.ceil(len(samples) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 3.4 * rows))
    flat_axes = list(axes.flat) if hasattr(axes, "flat") else [axes]
    for axis, (item, cutout) in zip(flat_axes, samples, strict=False):
        z1, z2 = zscale_limits(cutout)
        axis.imshow(cutout.numpy(), origin="lower", cmap="gray", vmin=z1, vmax=z2)
        if item["status"] == "accepted":
            detail = f"pruned residual={item['pruned_residual']:.4g}"
        else:
            detail = str(item["reason"])
        axis.set_title(
            f"{item['split']} {item['band']}\n{Path(str(item['exposure'])).stem}\n{detail}",
            fontsize=7,
        )
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in flat_axes[len(samples) :]:
        axis.set_visible(False)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _write_model_diagnostics(path: Path, rows: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 10))
    colors = {"train": "#4c78a8", "validation": "#f58518", "test": "#e45756"}
    for split, color in colors.items():
        split_rows = [row for row in rows if row["split"] == split]
        target = [row["target_value"] for row in split_rows]
        axes[0, 0].scatter(
            target,
            [row["pruned_prediction"] for row in split_rows],
            s=9,
            alpha=0.55,
            label=split,
            color=color,
        )
        axes[0, 1].scatter(
            [row["time_mjd_utc"] for row in split_rows],
            [row["pruned_residual"] for row in split_rows],
            s=9,
            alpha=0.55,
            label=split,
            color=color,
        )
    bounds = axes[0, 0].get_xlim()
    axes[0, 0].plot(bounds, bounds, "k--", linewidth=1)
    axes[0, 0].set(xlabel="measured background", ylabel="pruned prediction")
    axes[0, 0].legend()
    axes[0, 1].axhline(0.0, color="black", linewidth=1)
    axes[0, 1].set(xlabel="MJD UTC", ylabel="pruned residual")
    test = [row for row in rows if row["split"] == "test"]
    scatter = axes[1, 0].scatter(
        [row["detector_x"] for row in test],
        [row["detector_y"] for row in test],
        c=[row["pruned_residual"] for row in test],
        cmap="coolwarm",
        s=18,
    )
    figure.colorbar(scatter, ax=axes[1, 0], label="pruned residual")
    axes[1, 0].set(xlabel="detector x", ylabel="detector y", title="test spatial residual")
    axes[1, 1].scatter(
        [row["stationary_residual"] for row in test],
        [row["pruned_residual"] for row in test],
        c=[row["time_mjd_utc"] for row in test],
        cmap="viridis",
        s=18,
    )
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 1].axvline(0.0, color="black", linewidth=0.8)
    axes[1, 1].set(
        xlabel="stationary residual",
        ylabel="pruned residual",
        title="test residual comparison (colour = MJD)",
    )
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _weighted_metrics(rows: list[dict[str, Any]], family: str) -> dict[str, float]:
    residual = torch.tensor([row[f"{family}_residual"] for row in rows])
    weight = torch.tensor([float(row.get("sample_weight", 1.0)) for row in rows])
    weight = weight / weight.sum()
    return {
        "rmse": float(torch.sqrt((weight * residual.square()).sum())),
        "mae": float((weight * residual.abs()).sum()),
        "bias": float((weight * residual).sum()),
    }


def _write_readme(
    path: Path,
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    *,
    detector: str,
) -> None:
    lines = [
        f"# {detector} background inspection",
        "",
        "This bundle compares the stationary, full adaptive, and validation-pruned "
        "background models against automatic CRF evidence. Model selection never uses "
        "the test split.",
        "",
        f"Accepted targets: **{len(accepted)}**. Rejected measurements: **{len(rejected)}**. "
        f"Displayed cutouts: **{len(selected)}**.",
        "",
        "## Held-out metrics",
        "",
        "| split | model | weighted RMSE | weighted MAE | weighted bias | n |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for split in ("train", "validation", "test"):
        rows = [row for row in accepted if row["split"] == split]
        for family in ("stationary", "adaptive", "pruned"):
            metrics = _weighted_metrics(rows, family)
            lines.append(
                f"| {split} | {family} | {metrics['rmse']:.6g} | "
                f"{metrics['mae']:.6g} | {metrics['bias']:.6g} | {len(rows)} |"
            )
    lines.extend(
        [
            "",
            "## Test metrics by band",
            "",
            "| band | model | weighted RMSE | weighted MAE | weighted bias | n |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for band in sorted({str(row["band"]) for row in accepted}):
        rows = [
            row
            for row in accepted
            if row["split"] == "test" and row["band"] == band
        ]
        for family in ("stationary", "adaptive", "pruned"):
            metrics = _weighted_metrics(rows, family)
            lines.append(
                f"| {band} | {family} | {metrics['rmse']:.6g} | "
                f"{metrics['mae']:.6g} | {metrics['bias']:.6g} | {len(rows)} |"
            )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `model_diagnostics.png`: measured/predicted, temporal, and spatial residual plots.",
            "- `contact_sheets/`: zscale-rendered CRF cutouts grouped by split, band, and acceptance.",
            "- `inspection_index.csv`: compact provenance and model-residual index for displayed cutouts.",
            "- `background_predictions.csv`: all accepted NRCB4 background measurements and predictions.",
            "- `cutouts_*.fits`: matching SCI, ERR, and DQ cubes written through Torchfits.",
            "- `regions/`: one-indexed image-coordinate DS9 boxes, one file per exposure.",
            "",
            "The FITS cube plane order is exactly the row order in `inspection_index.csv`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def export_inspection_bundle(
    manifest: Path,
    fit_report: Path,
    output_dir: Path,
    *,
    detector: str = "NRCB4",
    per_group: int = 12,
    overwrite: bool = False,
) -> Path:
    detector = detector.upper()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    report = json.loads(fit_report.read_text(encoding="utf-8"))
    accepted = _prediction_rows(manifest, payload, report, detector=detector)
    rejected = _rejected_rows(payload, detector=detector)
    selected = _selected_rows(accepted, rejected, per_group=per_group)
    lookup = _exposure_lookup(payload)
    radius = int(payload["extraction"]["settings"]["background_radius_px"])
    samples: list[tuple[dict[str, Any], Tensor, Tensor, Tensor]] = []
    for index, row in enumerate(selected):
        exposure = lookup.get(str(row["exposure"])) or lookup.get(
            Path(str(row["exposure"])).name
        )
        if exposure is None:
            raise ValueError(f"unknown exposure for {row['source_id']}")
        science, error, dq = _cutout(row, exposure, radius=radius)
        row["inspection_index"] = index
        row["resolved_exposure"] = exposure["resolved_path"]
        samples.append((row, science, error, dq))

    _write_csv(output_dir / "background_predictions.csv", accepted)
    _write_csv(output_dir / "inspection_index.csv", selected)
    write(
        str(output_dir / "cutouts_science.fits"),
        torch.stack([sample[1].to(torch.float32) for sample in samples]),
        header={"DETECTOR": detector, "ROLE": "BACKGROUND"},
        overwrite=overwrite,
    )
    write(
        str(output_dir / "cutouts_error.fits"),
        torch.stack([sample[2].to(torch.float32) for sample in samples]),
        header={"DETECTOR": detector, "ROLE": "BACKGROUND"},
        overwrite=overwrite,
    )
    write(
        str(output_dir / "cutouts_dq.fits"),
        torch.stack([sample[3].to(torch.int32) for sample in samples]),
        header={"DETECTOR": detector, "ROLE": "BACKGROUND"},
        overwrite=overwrite,
    )

    sheets = output_dir / "contact_sheets"
    sheets.mkdir(exist_ok=True)
    grouped: dict[tuple[str, str, str], list[tuple[dict[str, Any], Tensor]]] = (
        defaultdict(list)
    )
    for row, science, _error, _dq in samples:
        grouped[(row["split"], row["band"], row["status"])].append((row, science))
    for (split, band, status), group in sorted(grouped.items()):
        _write_contact_sheet(sheets / f"{split}_{band}_{status}.png", group)

    regions = output_dir / "regions"
    regions.mkdir(exist_ok=True)
    by_exposure: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_exposure[str(row["exposure"])].append(row)
    for exposure_token, rows in by_exposure.items():
        exposure = lookup.get(exposure_token) or lookup.get(Path(exposure_token).name)
        if exposure is None:
            raise ValueError(f"unknown exposure {exposure_token}")
        xmin, _xmax, ymin, _ymax = map(float, exposure["detector_bounds_xy"])
        lines = ["# Region file format: DS9 version 4.1", "image"]
        for row in rows:
            x = float(row["detector_x"]) - xmin + 1.0
            y = float(row["detector_y"]) - ymin + 1.0
            color = "cyan" if row["status"] == "accepted" else "red"
            label = f"{row['status']} {row['split']} {row['band']}"
            lines.append(
                f"box({x:.3f},{y:.3f},{2 * radius + 1},{2 * radius + 1},0) "
                f"# color={color} text={{{label}}}"
            )
        (regions / f"{Path(exposure_token).stem}.reg").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    _write_model_diagnostics(output_dir / "model_diagnostics.png", accepted)
    _write_readme(
        output_dir / "README.md",
        accepted,
        rejected,
        selected,
        detector=detector,
    )
    return output_dir / "README.md"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fit-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--detector", default="NRCB4")
    parser.add_argument("--per-group", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    readme = export_inspection_bundle(
        args.manifest.resolve(),
        args.fit_report.resolve(),
        args.output_dir,
        detector=args.detector,
        per_group=args.per_group,
        overwrite=args.overwrite,
    )
    print(readme)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
