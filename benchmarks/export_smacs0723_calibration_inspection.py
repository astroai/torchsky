"""Export human-readable detector or exposure background calibration diagnostics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import textwrap
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
from torchsky.calibration_evidence import (  # noqa: E402
    calibration_exposure_id,
    load_operator_calibration_split,
)
from torchsky.io import read, write  # noqa: E402
from torchsky.map.operators import (  # noqa: E402
    ConditionedBackgroundField,
    apply_operator_calibration_pruning,
)
from torchsky.viz.renderers.image import zscale_limits  # noqa: E402


def _fit_group(
    report: dict[str, Any], *, detector: str, role: str, family: str,
    exposure_id: str | None = None, band: str | None = None,
    optional: bool = False,
) -> dict[str, Any] | None:
    matches = [
        group
        for group in report["groups"]
        if group["detector"] == detector
        and group["role"] == role
        and group["family"] == family
        and group.get("calibration_scope", "exposure" if group.get("exposure_id") else "detector")
        == ("exposure" if exposure_id is not None else "detector")
        and group.get("exposure_id") == exposure_id
        and (band is None or str(group.get("band", "")).upper() == band.upper())
    ]
    if not matches and optional:
        return None
    if len(matches) != 1:
        raise ValueError(
            f"expected one {detector}/{role}/{family} fit group"
            f" (exposure_id={exposure_id!r}, band={band!r})"
        )
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
        exposure_id=group.get("exposure_id"),
        band=group.get("band"),
        verify_hashes=True,
    )
    model, _ = fit_calibration._make_model(
        train,
        role="background",
        family=str(group["family"]),
        device=torch.device("cpu"),
        exposure_scope=group.get("exposure_id") is not None,
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
    exposure_id: str | None = None,
    band: str | None = None,
) -> list[dict[str, Any]]:
    stationary_group = _fit_group(
        report, detector=detector, role="background", family="stationary",
        exposure_id=exposure_id, band=band,
    )
    assert stationary_group is not None
    adaptive_group = _fit_group(
        report, detector=detector, role="background", family="adaptive",
        exposure_id=exposure_id, band=band,
        optional=True,
    )
    models = {"stationary": _load_background_model(manifest, stationary_group, detector=detector)}
    if adaptive_group is not None:
        models["adaptive"] = _load_background_model(manifest, adaptive_group, detector=detector)
        models["pruned"] = _load_background_model(manifest, adaptive_group, detector=detector)
        apply_operator_calibration_pruning(models["pruned"], adaptive_group["validation_pruning"])
    output: list[dict[str, Any]] = []
    for split in ("train", "validation", "test"):
        arrays = load_operator_calibration_split(
            manifest,
            split=split,
            role="background",
            detector=detector,
            exposure_id=exposure_id,
            band=band,
            verify_hashes=True,
        )
        rows = [
            row
            for row in payload["evidence"]
            if row.get("calibration_split" if exposure_id is not None else "split") == split
            and row["role"] == "background"
            and str(row["detector"]).upper() == detector
            and (exposure_id is None or calibration_exposure_id(row) == exposure_id)
            and (band is None or str(row.get("band", "")).upper() == band.upper())
        ]
        if len(rows) != arrays["target_values"].numel():
            raise RuntimeError("manifest row order/count does not match loaded evidence")
        predictions = {family: _predict(model, arrays) for family, model in models.items()}
        for index, source in enumerate(rows):
            target = float(arrays["target_values"][index])
            item = dict(source)
            if exposure_id is not None:
                item["exposure_split"] = source["split"]
                item["split"] = split
            item.update({"status": "accepted", "target_value": target})
            for family, values in predictions.items():
                prediction = float(values[index])
                item[f"{family}_prediction"] = prediction
                item[f"{family}_residual"] = prediction - target
                item[f"{family}_absolute_error"] = abs(prediction - target)
            if "pruned" in predictions:
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
    payload: dict[str, Any], *, detector: str,
    exposure_id: str | None = None, band: str | None = None,
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
        if exposure_id is not None and calibration_exposure_id(source) != exposure_id:
            continue
        if band is not None and str(exposure["band"]).upper() != band.upper():
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
        if exposure_id is not None:
            item["exposure_split"] = exposure["split"]
            item["split"] = source.get("calibration_split", "unassigned")
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
                    -float(row[f"{_diagnostic_family(row)}_absolute_error"]), str(row["source_id"])
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
            family = _diagnostic_family(item)
            detail = f"{family} residual={item[f'{family}_residual']:.4g}"
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


def _diagnostic_family(row: dict[str, Any]) -> str:
    return "pruned" if "pruned_residual" in row else "stationary"


def _write_model_diagnostics(
    path: Path, rows: list[dict[str, Any]], *, title: str = "Background calibration"
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 10))
    figure.suptitle(textwrap.fill(title, width=100), fontsize=10)
    family = _diagnostic_family(rows[0])
    colors = {"train": "#4c78a8", "validation": "#f58518", "test": "#e45756"}
    for split, color in colors.items():
        split_rows = [row for row in rows if row["split"] == split]
        target = [row["target_value"] for row in split_rows]
        axes[0, 0].scatter(
            target,
            [row[f"{family}_prediction"] for row in split_rows],
            s=9,
            alpha=0.55,
            label=split,
            color=color,
        )
        axes[0, 1].scatter(
            [row["time_mjd_utc"] for row in split_rows],
            [row[f"{family}_residual"] for row in split_rows],
            s=9,
            alpha=0.55,
            label=split,
            color=color,
        )
    bounds = axes[0, 0].get_xlim()
    axes[0, 0].plot(bounds, bounds, "k--", linewidth=1)
    axes[0, 0].set(xlabel="measured background", ylabel=f"{family} prediction")
    axes[0, 0].legend()
    axes[0, 1].axhline(0.0, color="black", linewidth=1)
    axes[0, 1].set(xlabel="MJD UTC", ylabel=f"{family} residual")
    test = [row for row in rows if row["split"] == "test"]
    scatter = axes[1, 0].scatter(
        [row["detector_x"] for row in test],
        [row["detector_y"] for row in test],
        c=[row[f"{family}_residual"] for row in test],
        cmap="coolwarm",
        s=18,
    )
    figure.colorbar(scatter, ax=axes[1, 0], label=f"{family} residual")
    axes[1, 0].set(xlabel="detector x", ylabel="detector y", title="test spatial residual")
    axes[1, 1].scatter(
        [row["stationary_residual"] if family == "pruned" else row["target_value"] for row in test],
        [row[f"{family}_residual"] for row in test],
        c=[row["time_mjd_utc"] for row in test],
        cmap="viridis",
        s=18,
    )
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 1].axvline(0.0, color="black", linewidth=0.8)
    axes[1, 1].set(
        xlabel="stationary residual" if family == "pruned" else "measured background",
        ylabel=f"{family} residual",
        title="test residuals (colour = MJD)",
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
    exposure_id: str | None = None,
    band: str | None = None,
    diagnostics_only: bool = False,
) -> None:
    families = [family for family in ("stationary", "adaptive", "pruned")
                if f"{family}_residual" in accepted[0]]
    lines = [
        f"# {detector} background inspection",
        "",
        f"This bundle evaluates {', '.join(families)} "
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
    if exposure_id is not None:
        lines[0] = f"# {detector} / {exposure_id} / {band} background inspection"
        lines[2:2] = [
            f"Exposure: `{exposure_id}`. Band: `{band}`. Metrics use `calibration_split`; "
            "the original exposure holdout is retained as `exposure_split`. "
            "Rejected measurements without a calibration split are marked `unassigned`.",
            "",
        ]
    for split in ("train", "validation", "test"):
        rows = [row for row in accepted if row["split"] == split]
        for family in families:
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
        for family in families:
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
            "- `background_predictions.csv`: accepted background measurements and predictions for the selected scope.",
            "- `cutouts_*.fits`: matching SCI, ERR, and DQ cubes written through Torchfits.",
            "- `regions/`: one-indexed image-coordinate DS9 boxes, one file per exposure.",
            "",
            "The FITS cube plane order is exactly the row order in `inspection_index.csv`.",
            "",
        ]
    )
    if diagnostics_only:
        lines = [line for line in lines if not any(token in line for token in
                 ("contact_sheets/", "inspection_index.csv", "cutouts_*.fits", "regions/", "FITS cube plane order"))]
        lines.extend(["Diagnostics-only export: raw exposure cutouts were not read.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_psf_diagnostics(
    manifest: Path, payload: dict[str, Any], report: dict[str, Any], output_dir: Path,
    *, detector: str, exposure_id: str | None, band: str | None,
    per_group: int, overwrite: bool,
) -> None:
    """Show stationary validation kernels using the calibration loss support."""
    group = _fit_group(report, detector=detector, role="psf", family="stationary",
                       exposure_id=exposure_id, band=band, optional=True)
    if group is None:
        return
    selection = dict(role="psf", detector=detector, exposure_id=exposure_id, band=band)
    train = load_operator_calibration_split(manifest, split="train", **selection)
    arrays = load_operator_calibration_split(manifest, split="validation", **selection)
    model, _ = fit_calibration._make_model(train, role="psf", family="stationary",
        device=torch.device("cpu"), exposure_scope=exposure_id is not None)
    checkpoint = torch.load(group["checkpoint"], map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    term = fit_calibration._term(model, arrays, role="psf", name="inspection_validation")
    with torch.no_grad():
        predicted = term.predicted_kernels()
        weight = term.pixel_weight.to(dtype=predicted.dtype)
        target = term.target_kernels.to(dtype=predicted.dtype)
        predicted = torch.where(weight > 0, predicted, torch.zeros_like(predicted))
        predicted = predicted / predicted.sum(dim=(-2, -1), keepdim=True).clamp_min(
            torch.finfo(predicted.dtype).tiny)
        residual = predicted - target
    rows = [dict(row) for row in payload["evidence"]
            if row.get("calibration_split" if exposure_id is not None else "split") == "validation"
            and row["role"] == "psf" and str(row["detector"]).upper() == detector
            and (exposure_id is None or calibration_exposure_id(row) == exposure_id)
            and (band is None or str(row["band"]).upper() == band.upper())]
    if len(rows) != len(target):
        raise RuntimeError("PSF manifest row order/count does not match loaded evidence")
    for index, row in enumerate(rows):
        row["kernel_index"] = index
        row["pixel_weighted_mse"] = float((residual[index].square() * weight[index]).sum() / weight[index].sum())
    _write_csv(output_dir / "psf_validation_index.csv", rows)
    for name, values in (("measured", target), ("stationary", predicted),
                         ("residual", residual), ("pixel_weight", weight)):
        write(str(output_dir / f"psf_validation_{name}.fits"), values.detach().cpu(),
              header={"DETECTOR": detector, "ROLE": "PSF", "SPLIT": "validation",
                      "BUNIT": "RELATIVE" if name == "pixel_weight" else "NORMALIZED",
                      "FILTER": band or "MULTIPLE"},
              overwrite=overwrite)
    count = min(len(rows), per_group)
    figure, axes = plt.subplots(count, 3, squeeze=False, figsize=(10, 3 * count))
    identity = Path(exposure_id).name if exposure_id else "detector scope"
    metrics = group.get("validation_report", {}).get("terms", [{}])[0]
    def measurement(key: str) -> str:
        value = metrics.get(key)
        return "not measured" if value is None else f"{value:.4g}"
    figure.suptitle(
        f"{detector} / {band or 'all bands'}\n{identity}\n"
        f"Stationary PSF: normalized flux; detector x right, y up; 1 pixel/sample\n"
        f"validation MSE {measurement('raw_loss')}; centroid RMS "
        f"{measurement('centroid_rmse_pixels')} px; max enclosed-flux error "
        f"{measurement('maximum_encircled_energy_error')}\n"
        "FWHM / photometric bias / memory: not measured", fontsize=9,
    )
    lower = float(torch.minimum(target.min(), predicted.min()))
    upper = float(torch.maximum(target.max(), predicted.max()))
    limit = max(float(residual.abs().max()), torch.finfo(residual.dtype).eps)
    for index in range(count):
        for column, (name, values) in enumerate((("measured", target), ("stationary", predicted), ("residual", residual))):
            axis = axes[index, column]
            artist = axis.imshow(values[index].detach().cpu().numpy(), origin="lower",
                cmap="coolwarm" if column == 2 else "viridis",
                vmin=-limit if column == 2 else lower, vmax=limit if column == 2 else upper)
            axis.set_title(f"{name}: kernel {index}", fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])
            figure.colorbar(artist, ax=axis, shrink=0.7)
    figure.tight_layout(h_pad=2.5)
    figure.savefig(output_dir / "psf_validation.png", dpi=150)
    plt.close(figure)
    with (output_dir / "README.md").open("a", encoding="utf-8") as handle:
        handle.write("\nStationary PSF validation: `psf_validation.png` uses a common measured/predicted "
                     "scale and a symmetric residual scale. FITS cubes contain measured, stationary, "
                     "residual, and pixel-weight arrays in `psf_validation_index.csv` order. Kernels "
                     "are masked and flux-normalized on the calibration loss support. These are "
                     "extracted stellar kernels, not independent PSF source-physics validation.\n")


def export_inspection_bundle(
    manifest: Path,
    fit_report: Path,
    output_dir: Path,
    *,
    detector: str = "NRCB4",
    exposure_id: str | None = None,
    band: str | None = None,
    per_group: int = 12,
    overwrite: bool = False,
    diagnostics_only: bool = False,
) -> Path:
    detector = detector.upper()
    if per_group < 1:
        raise ValueError("per_group must be positive")
    if (exposure_id is None) != (band is None) or (
        exposure_id is not None and (not exposure_id.strip() or not band or not band.strip())
    ):
        raise ValueError("exposure selection requires nonempty exposure_id and band together")
    band = band.upper() if band is not None else None
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    report = json.loads(fit_report.read_text(encoding="utf-8"))
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
                              check=True, capture_output=True, text=True).stdout.strip()
    provenance = {"source_revision": revision,
        "exporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "manifest": str(manifest.resolve()), "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "fit_report": str(fit_report.resolve()), "fit_report_sha256": hashlib.sha256(fit_report.read_bytes()).hexdigest(),
        "detector": detector, "exposure_id": exposure_id, "band": band,
        "diagnostics_only": diagnostics_only}
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    accepted = _prediction_rows(
        manifest, payload, report, detector=detector, exposure_id=exposure_id, band=band
    )
    rejected = _rejected_rows(payload, detector=detector, exposure_id=exposure_id, band=band)
    if diagnostics_only:
        _write_csv(output_dir / "background_predictions.csv", accepted)
        title = f"{detector} / {exposure_id} / {band} background calibration" if exposure_id else f"{detector} background calibration"
        _write_model_diagnostics(output_dir / "model_diagnostics.png", accepted, title=title)
        _write_readme(output_dir / "README.md", accepted, rejected, [], detector=detector,
                      exposure_id=exposure_id, band=band, diagnostics_only=True)
        _write_psf_diagnostics(manifest, payload, report, output_dir, detector=detector,
            exposure_id=exposure_id, band=band, per_group=per_group, overwrite=overwrite)
        return output_dir / "README.md"
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

    title = f"{detector} background calibration"
    if exposure_id is not None:
        title = f"{detector} / {exposure_id} / {band} background calibration"
    _write_model_diagnostics(output_dir / "model_diagnostics.png", accepted, title=title)
    _write_readme(
        output_dir / "README.md",
        accepted,
        rejected,
        selected,
        detector=detector,
        exposure_id=exposure_id,
        band=band,
    )
    return output_dir / "README.md"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fit-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--detector", default="NRCB4")
    parser.add_argument("--exposure-id", help="Exact exposure identity from the fit report; requires --band")
    parser.add_argument("--band", help="Band for exposure-scoped fits; requires --exposure-id")
    parser.add_argument("--per-group", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--diagnostics-only", action="store_true", help="Export predictions and plots without reading raw exposure cutouts")
    args = parser.parse_args()
    readme = export_inspection_bundle(
        args.manifest.resolve(),
        args.fit_report.resolve(),
        args.output_dir,
        detector=args.detector,
        exposure_id=args.exposure_id,
        band=args.band,
        per_group=args.per_group,
        overwrite=args.overwrite,
        diagnostics_only=args.diagnostics_only,
    )
    print(readme)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
