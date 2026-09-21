#!/usr/bin/env python3
"""Render reproducible SMACS J0723 product quicklooks.

The original cutout-manifest and legacy cube arguments remain supported.  New
run products can be supplied with ``--products-json`` using entries such as
``{"method": "linear", "band": "F090W", "path": "linear_F090W.fits"}``.
Entries may also point to a cube and include a ``bands`` list.

FITS I/O is owned by :mod:`torchfits`.  Display limits are measured once from
the baseline and persisted so every method uses the same scale.  Supplied
metrics are displayed as supplied; this script does not estimate science
metrics from pixels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import textwrap
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib
import numpy as np
import torch
import torchfits

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
MISSING_COLOR = "#4d4d4d"
_WCS_KEYS = (
    "CTYPE1", "CTYPE2", "CUNIT1", "CUNIT2", "CRPIX1", "CRPIX2",
    "CRVAL1", "CRVAL2", "CDELT1", "CDELT2", "CD1_1", "CD1_2", "CD2_1",
    "CD2_2", "PC1_1", "PC1_2", "PC2_1", "PC2_2", "CROTA1", "CROTA2",
)


@dataclass(frozen=True)
class ProductPlane:
    """One rendered band from one method/run."""

    method: str
    band: str
    data: np.ndarray
    header: Mapping[str, Any]
    path: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _default_manifest_path() -> Path:
    return ROOT / "data" / "smacs0723_minicutout" / "manifest.json"


def _resolve_manifest_path(path_text: str, *, manifest_dir: Path) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    root_path = (ROOT / path).resolve()
    if root_path.exists():
        return root_path
    return (manifest_dir / path).resolve()


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"missing {manifest_path}; run scripts/build_smacs0723_minicutout.py first"
        )
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"manifest {manifest_path} must contain a JSON object")
    return value


def _read_fits(path: Path, *, hdu: int | str = 0) -> tuple[np.ndarray, dict[str, Any]]:
    """Read an image or cube through torchfits and return a NumPy view."""

    try:
        tensor, header = torchfits.read_tensor(
            str(path), hdu=hdu, mmap=False, return_header=True
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"could not read FITS image {path}: {exc}") from exc
    array = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(array), dict(header)


def _read_image(path: Path) -> np.ndarray:
    data, _ = _read_fits(path)
    if data.ndim != 2:
        raise ValueError(f"{path} must contain a 2D image, got shape {data.shape}")
    return data


def _read_cube(path: Path) -> np.ndarray:
    data, _ = _read_fits(path)
    if data.ndim != 3:
        raise ValueError(f"{path} must contain a 3D cube, got shape {data.shape}")
    return data


def _finite_values(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    return values[np.isfinite(values)]


def _display_limits(image: np.ndarray) -> tuple[float, float]:
    """Return robust limits for one reference image."""

    finite = _finite_values(image)
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        med = float(np.nanmedian(finite))
        return med - 1.0, med + 1.0
    return float(lo), float(hi)


def _shared_display_limits(
    images: Mapping[str, np.ndarray],
    *,
    persisted: Mapping[str, Any] | None = None,
) -> dict[str, tuple[float, float]]:
    """Compute or restore one frozen scale per reference band."""

    result: dict[str, tuple[float, float]] = {}
    persisted = persisted or {}
    for band, image in images.items():
        prior = persisted.get(band)
        if isinstance(prior, (list, tuple)) and len(prior) == 2:
            lo, hi = float(prior[0]), float(prior[1])
            if math.isfinite(lo) and math.isfinite(hi) and hi > lo:
                result[band] = (lo, hi)
                continue
        result[band] = _display_limits(image)
    return result


def _symmetric_residual_limit(residuals: Iterable[np.ndarray]) -> float:
    values_list = [_finite_values(np.asarray(residual)) for residual in residuals]
    values = np.concatenate(values_list, axis=0) if values_list else np.empty(0)
    if values.size == 0:
        return 1.0
    limit = float(np.percentile(np.abs(values), 99.0))
    if not math.isfinite(limit) or limit <= 0.0:
        limit = float(np.max(np.abs(values), initial=0.0))
    return max(limit, 1.0e-8)


def _normalize_for_rgb(
    image: np.ndarray,
    limits: tuple[float, float] | None = None,
) -> np.ndarray:
    lo, hi = _display_limits(image) if limits is None else limits
    scaled = (np.asarray(image, dtype=np.float32) - lo) / max(hi - lo, 1.0e-8)
    return np.clip(scaled, 0.0, 1.0)


def _header_value(header: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        for key in (name, name.upper(), name.lower()):
            if key in header and header[key] not in (None, ""):
                return header[key]
    return None


def _format_value(value: Any, *, suffix: str = "") -> str:
    if value is None or value == "":
        return "not measured"
    if isinstance(value, (float, int)):
        value = float(value)
        if not math.isfinite(value):
            return "not measured"
        return f"{value:g}{suffix}"
    return f"{value}{suffix}"


def _units(header: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    value = _header_value(metadata, "units", "unit", "bunit")
    return str(value) if value is not None else str(_header_value(header, "BUNIT") or "not measured")


def _angular_scale(header: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    value = _header_value(metadata, "angular_scale", "pixel_scale_arcsec")
    if value is not None:
        return _format_value(value, suffix='"/pix')
    if any(str(header.get(f"CUNIT{i}", "deg")).lower() not in {"deg", "degree"} for i in (1, 2)):
        return "not measured"
    if any(str(k).startswith(("CD1_", "CD2_", "PC1_", "PC2_")) for k in header):
        return "not measured"
    values = [
        abs(float(value))
        for value in (_header_value(header, "CDELT1"), _header_value(header, "CDELT2"))
        if value is not None
    ]
    return _format_value(float(np.mean(values)) * 3600.0, suffix='"/pix') if values else "not measured"


def _orientation(header: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    value = _header_value(metadata, "orientation", "orientation_deg")
    if value is not None:
        return _format_value(value, suffix=" deg" if isinstance(value, (float, int)) else "")
    # A rotation card alone does not specify handedness or projection convergence.
    return "not measured"


def _wcs_fingerprint(header: Mapping[str, Any]) -> tuple[Any, ...] | None:
    required = ("CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2")
    if any(_header_value(header, key) is None for key in required):
        return None
    if not (("CDELT1" in header and "CDELT2" in header) or any(str(k).startswith("CD1_") for k in header)):
        return None
    distortion = re.compile(r"^(?:(?:A|B|AP|BP)_(?:ORDER|\d+_\d+)|PV[12]_\d+|LONPOLE|LATPOLE|RADESYS|EQUINOX)$")
    return tuple(sorted((str(key), str(value)) for key, value in header.items() if key in _WCS_KEYS or distortion.match(str(key))))


def _support_fingerprint(data: np.ndarray) -> str:
    mask = np.isfinite(np.asarray(data))
    return hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest()


def _validate_shared_grid(products: Iterable[ProductPlane]) -> tuple[bool, str]:
    """Require an explicit common image grid and WCS."""

    items = list(products)
    if not items:
        return False, "no bands supplied"
    if len({tuple(item.data.shape) for item in items}) != 1:
        return False, "different image shapes/resolution"
    wcs = {_wcs_fingerprint(item.header) for item in items}
    if None in wcs:
        return False, "WCS metadata is missing"
    if len(wcs) != 1:
        return False, "WCS metadata differs"

    return True, "shared WCS and shape"


def _validate_shared_support(products: Iterable[ProductPlane]) -> tuple[bool, str]:
    """Require explicit common grid, WCS, and physical units."""

    items = list(products)
    valid_grid, reason = _validate_shared_grid(items)
    if not valid_grid:
        return valid_grid, reason
    units = {_units(item.header, item.metadata) for item in items}
    if len(units) != 1 or "not measured" in units:
        return False, "common physical units are missing or differ"
    return True, "shared WCS, shape, units; intersection of band coverage"


def _metrics_for(metrics: Mapping[str, Any], *, method: str, band: str) -> Mapping[str, Any]:
    """Read flat, method-grouped, and method/band metric JSON shapes."""

    candidates: list[Any] = [metrics.get(f"{method}:{band}"), metrics.get(method)]
    for root_name in ("methods", "runs", "products"):
        root = metrics.get(root_name)
        if isinstance(root, Mapping):
            candidates.extend((root.get(f"{method}:{band}"), root.get(method)))
        elif isinstance(root, list):
            candidates.extend(
                item.get("metrics")
                for item in root
                if isinstance(item, Mapping)
                and str(item.get("method", "")) == method
                and str(item.get("band", band)) == band
            )
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            bands = candidate.get("bands")
            if isinstance(bands, Mapping) and isinstance(bands.get(band), Mapping):
                return bands[band]
            return candidate
    return {}


def _metric_text(metrics: Mapping[str, Any], *names: str) -> str:
    for name in names:
        if name in metrics:
            return _format_value(metrics[name])
    return "not measured"


def _legend_text(product: ProductPlane, metrics: Mapping[str, Any]) -> str:
    if "validation_nll_per_pixel" in metrics or "validation_chi2_per_pixel" in metrics:
        prediction_text = (
            f"validation NLL/pixel={_metric_text(metrics, 'validation_nll_per_pixel')}  "
            f"chi2/pixel={_metric_text(metrics, 'validation_chi2_per_pixel')}"
        )
    else:
        prediction_text = f"prediction={_metric_text(metrics, 'prediction', 'prediction_rmse')}"
    render_details = ""
    render_mode = product.metadata.get("render_mode")
    product_interpretation = product.metadata.get("product_interpretation")
    if render_mode is not None or product_interpretation is not None:
        render_details = (
            f"\nrender={render_mode or 'not measured'}"
        )
        if product_interpretation is not None:
            render_details += "\ninterpretation=" + "\n".join(
                textwrap.wrap(str(product_interpretation), width=42)
            )
        else:
            render_details += "\ninterpretation=not measured"
    return (
        f"method={product.method}\n"
        f"band={product.band}  units={_units(product.header, product.metadata)}{render_details}\n"
        f"scale={_angular_scale(product.header, product.metadata)}  orientation={_orientation(product.header, product.metadata)}\n"
        f"{prediction_text}  "
        f"PSF width={_metric_text(metrics, 'psf_width', 'psf_width_pix')}\n"
        f"flux bias={_metric_text(metrics, 'flux_bias', 'flux_bias_fraction')}  "
        f"color bias={_metric_text(metrics, 'color_bias', 'color_bias_mag')}\n"
        f"runtime={_metric_text(metrics, 'runtime', 'runtime_s')}  "
        f"memory={_metric_text(metrics, 'memory', 'memory_mb')}"
    )


def _grid_shape(n_panel: int) -> tuple[int, int]:
    cols = int(np.ceil(np.sqrt(max(n_panel, 1))))
    return int(np.ceil(max(n_panel, 1) / cols)), cols


def _wrap_title(title: str, *, width: int = 48) -> str:
    """Keep multiline panel titles inside their axes."""

    return "\n".join(
        "\n".join(textwrap.wrap(line, width=width, break_long_words=False, break_on_hyphens=False))
        if line
        else ""
        for line in title.splitlines()
    )


def _cmap_with_missing(name: str) -> Any:
    return plt.get_cmap(name).with_extremes(bad=MISSING_COLOR)


def _save_image_grid(
    images: list[np.ndarray],
    titles: list[str],
    output_path: Path,
    *,
    max_panels: int,
    cmap: str = "gray",
    dpi: int = 140,
    limits: list[tuple[float, float]] | None = None,
) -> None:
    if not images:
        return
    shown_images, shown_titles = images[:max_panels], titles[:max_panels]
    shown_limits = (limits or [])[:max_panels]
    rows, cols = _grid_shape(len(shown_images))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 4.0 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    image_cmap = _cmap_with_missing(cmap)
    for index, (ax, image, title) in enumerate(zip(axes.ravel()[:len(shown_images)], shown_images, shown_titles, strict=True)):
        lo, hi = shown_limits[index] if index < len(shown_limits) else _display_limits(image)
        ax.imshow(image, origin="lower", cmap=image_cmap, vmin=lo, vmax=hi)
        ax.set_title(_wrap_title(title), fontsize=7.5, linespacing=1.0)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _save_false_color(
    cube: np.ndarray,
    output_path: Path,
    *,
    dpi: int = 140,
    limits: list[tuple[float, float]] | None = None,
    title: str = "Approximate false color",
) -> None:
    if cube.shape[0] == 1:
        rgb = np.repeat(_normalize_for_rgb(cube[0], limits[0] if limits else None)[..., None], 3, axis=-1)
    else:
        scale = limits or [_display_limits(cube[i]) for i in range(cube.shape[0])]
        red = _normalize_for_rgb(cube[-1], scale[-1])
        blue = _normalize_for_rgb(cube[0], scale[0])
        middle = cube.shape[0] // 2
        green = _normalize_for_rgb(cube[middle], scale[middle])
        rgb = np.stack([red, green, blue], axis=-1)
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.imshow(rgb, origin="lower")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _lupton_rgb(red: np.ndarray, green: np.ndarray, blue: np.ndarray, *, stretch: float = 0.5, q: float = 8.0) -> np.ndarray:
    rgb = torchfits.transforms.lupton_rgb(
        torch.as_tensor(red, dtype=torch.float32),
        torch.as_tensor(green, dtype=torch.float32),
        torch.as_tensor(blue, dtype=torch.float32),
        Q=float(q),
        stretch=float(stretch),
    )
    return rgb.detach().cpu().numpy()


def _six_band_rgb(
    products: Mapping[str, ProductPlane],
    limits: Mapping[str, tuple[float, float]],
    *,
    stretch: float = 0.5,
    q: float = 8.0,
) -> tuple[np.ndarray | None, str]:
    """Return fixed B/G/R Lupton mix, or a pending reason."""

    groups = {"B": ("F090W", "F150W"), "G": ("F200W", "F277W"), "R": ("F356W", "F444W")}
    required = sum(groups.values(), ())
    missing = [band for band in required if band not in products]
    if missing:
        return None, f"missing bands: {', '.join(missing)}"
    chosen = [products[band] for band in required]
    ok, reason = _validate_shared_support(chosen)
    if not ok:
        return None, reason
    channels = {}
    for channel, (first, second) in groups.items():
        mean = 0.5 * (products[first].data + products[second].data)
        lo, hi = np.mean([limits[first], limits[second]], axis=0)
        # Average calibrated brightness before stretching; keep bright values
        # unclipped so Lupton's joint stretch can retain their color ratios.
        channels[channel] = np.maximum((mean - lo) / (hi - lo), 0)
    red, green, blue = channels["R"], channels["G"], channels["B"]
    valid = np.logical_and.reduce([np.isfinite(products[band].data) for band in required])
    rgb = _lupton_rgb(np.nan_to_num(red), np.nan_to_num(green), np.nan_to_num(blue), stretch=stretch, q=q)
    rgb[~valid] = 0.3
    return rgb, reason


def _save_comparison_grid(
    baseline: np.ndarray,
    comparison: np.ndarray,
    output_path: Path,
    *,
    max_panels: int,
    dpi: int = 140,
    limits: list[tuple[float, float]] | None = None,
    method: str = "comparison",
    bands: list[str] | None = None,
) -> dict[str, Any]:
    if baseline.shape != comparison.shape:
        raise ValueError(f"comparison cube shape {comparison.shape} must match baseline shape {baseline.shape}")
    n_plane = min(int(baseline.shape[0]), int(max_panels))
    residuals = [comparison[i] - baseline[i] for i in range(n_plane)]
    residual_limit = _symmetric_residual_limit(residuals)
    fig, axes = plt.subplots(n_plane, 3, figsize=(12.0, 3.5 * n_plane), squeeze=False)
    gray, diverging = _cmap_with_missing("gray"), _cmap_with_missing("coolwarm")
    for i in range(n_plane):
        band = bands[i] if bands and i < len(bands) else f"plane {i}"
        image_limits = limits[i] if limits and i < len(limits) else _display_limits(baseline[i])
        panels = (
            (baseline[i], image_limits, f"baseline | {band}"),
            (comparison[i], image_limits, f"{method} | {band}"),
            (residuals[i], (-residual_limit, residual_limit), f"signed residual | {band}"),
        )
        for ax, (image, panel_limits, title) in zip(axes[i], panels, strict=True):
            ax.imshow(image, origin="lower", cmap=diverging if "residual" in title else gray, vmin=panel_limits[0], vmax=panel_limits[1])
            ax.set_title(title, fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return {"residual_limit": residual_limit}


def _manifest_cutout_paths(manifest: Mapping[str, Any], *, manifest_dir: Path) -> tuple[list[Path], list[str]]:
    paths, titles = [], []
    for item in manifest.get("outputs", []):
        if not isinstance(item, Mapping):
            continue
        paths.append(_resolve_manifest_path(str(item["path"]), manifest_dir=manifest_dir))
        band, origin = str(item.get("band", "unknown")), item.get("crop_origin_xy")
        titles.append(f"{band}{f' origin={origin}' if origin is not None else ''}")
    return paths, titles


def _parse_product_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"--product expects METHOD=PATH, got {value!r}")
    method, path = value.split("=", 1)
    if not method.strip() or not path.strip():
        raise ValueError(f"--product expects METHOD=PATH, got {value!r}")
    return method.strip(), Path(path.strip())


def _product_entries(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, Mapping)]
    if isinstance(payload, Mapping):
        value = payload.get("products", payload.get("runs", []))
        if isinstance(value, list):
            return [item for item in value if isinstance(item, Mapping)]
    raise ValueError("product JSON must contain a list or a 'products'/'runs' list")


def _joint_export_slug(value: Any) -> str:
    """Match the benchmark's safe export directory component convention."""

    text = str(value).strip().lower()
    return "".join(char if char.isalnum() else "_" for char in text).strip("_") or "item"


def _joint_products(report_path: Path, export_dir: Path | None):
    """Adapt the joint benchmark report and exports to product manifests."""

    report = _load_json(report_path)
    experiment = report.get("experiment")
    if not isinstance(experiment, Mapping):
        raise ValueError("joint report must contain an experiment object")
    export_value = export_dir if export_dir is not None else experiment.get("export_dir")
    if not export_value:
        raise ValueError("joint report has no export_dir; supply --joint-export-dir")
    export_root = Path(export_value).expanduser()
    if export_dir is None and not export_root.is_absolute():
        export_root = report_path.parent / export_root
    else:
        export_root = export_root.resolve()

    psf_model = experiment.get("psf_model")
    if psf_model in (None, ""):
        raise ValueError("joint report experiment is missing psf_model")
    rows = report.get("rows")
    if not isinstance(rows, list):
        raise ValueError("joint report must contain a rows list")
    scoring = experiment.get("validation_detector_scoring")
    scores: Mapping[str, Any] = {}
    if isinstance(scoring, Mapping) and scoring.get("split") == "validation":
        value = scoring.get("scores_by_model", {})
        if not isinstance(value, Mapping):
            raise ValueError("joint validation scoring scores_by_model must be an object")
        scores = value

    entries: list[dict[str, Any]] = []
    metrics: dict[str, dict[str, Any]] = {}
    references: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("joint report rows must be objects")
        if row.get("status") != "OK":
            continue
        method = str(row.get("model", "")).strip()
        band = str(row.get("band", "")).strip().upper()
        if not method or not band:
            raise ValueError("successful joint report rows require model and band")
        directory = export_root / _joint_export_slug(psf_model) / _joint_export_slug(band) / _joint_export_slug(method)
        common = {"band": band, "support_path": str(directory / "support_mask.fits")}
        row_metadata = {
            key: row[key]
            for key in ("render_mode", "product_interpretation")
            if row.get(key) is not None
        }
        entries.append({**common, **row_metadata, "method": method, "path": str(directory / "prediction.fits")})
        if band not in references:
            entries.append({**common, "method": "baseline", "path": str(directory / "reference.fits")})
            references.add(band)

        supplied: dict[str, Any] = {}
        model_scores = scores.get(method, [])
        if isinstance(model_scores, list):
            for score in model_scores:
                if isinstance(score, Mapping) and str(score.get("band", "")).upper() == band:
                    supplied["validation_nll_per_pixel"] = score.get("gaussian_nll_per_pixel")
                    supplied["validation_chi2_per_pixel"] = score.get("chi2_per_pixel")
                    break
        if row.get("fit_time_ms") is not None:
            supplied["runtime"] = f"joint fit {_format_value(row['fit_time_ms'] / 1000)} s"
        memory_bytes = row.get("peak_accelerator_memory_bytes")
        if isinstance(memory_bytes, (float, int)) and math.isfinite(float(memory_bytes)) and memory_bytes > 0:
            supplied["memory"] = f"peak accelerator {_format_value(memory_bytes / 1024**2)} MiB"
        metrics[f"{method}:{band}"] = supplied
    if not references:
        raise ValueError("joint report has no successful exported products")
    return entries, metrics


def _load_product_specs(entries: Iterable[Mapping[str, Any]], *, manifest_dir: Path, default_bands: list[str]) -> list[ProductPlane]:
    result: list[ProductPlane] = []
    for entry in entries:
        if "path" not in entry:
            raise ValueError("product entry is missing path")
        method = str(entry.get("method", entry.get("run", "unknown")))
        path = _resolve_manifest_path(str(entry["path"]), manifest_dir=manifest_dir)
        data, header = _read_fits(path, hdu=entry.get("hdu", 0))
        if entry.get("support_path"):
            support_path = _resolve_manifest_path(str(entry["support_path"]), manifest_dir=manifest_dir)
            support, support_header = _read_fits(support_path)
            if support.ndim != 2:
                raise ValueError(f"support mask {support_path} must be a 2D image, got {support.shape}")
            if not np.isfinite(support).all():
                raise ValueError(f"support mask {support_path} contains nonfinite values")
            if not np.any(support > 0):
                raise ValueError(f"support mask {support_path} has no supported pixels")
            ok, reason = _validate_shared_grid([
                ProductPlane(method, "support", support, support_header, str(support_path)),
                ProductPlane(method, "image", data, header, str(path)),
            ])
            if not ok:
                raise ValueError(f"support mask for {path}: {reason}")
            data = np.where(np.isfinite(support) & (support > 0), data, np.nan)
        bands = [str(value).upper() for value in entry.get("bands", default_bands)]
        band = str(entry.get("band", "")).upper()
        if data.ndim == 2:
            if not band:
                if len(bands) == 1:
                    band = bands[0]
                else:
                    raise ValueError(f"individual product {path} must specify band")
            result.append(ProductPlane(method, band, data, header, str(path), entry))
        elif data.ndim == 3:
            if len(bands) != data.shape[0]:
                raise ValueError(f"cube {path} has {data.shape[0]} planes but {len(bands)} bands were supplied")
            result.extend(ProductPlane(method, item_band, data[i], header, str(path), entry) for i, item_band in enumerate(bands))
        else:
            raise ValueError(f"product {path} must contain a 2D image or 3D cube, got {data.shape}")
    return result


def _group_products(products: Iterable[ProductPlane]) -> dict[str, dict[str, ProductPlane]]:
    grouped: dict[str, dict[str, ProductPlane]] = {}
    for product in products:
        group = grouped.setdefault(product.method, {})
        if product.band in group:
            raise ValueError(f"duplicate product for {product.method}/{product.band}")
        group[product.band] = product
    return grouped


def _save_method_report(products, reference, limits, metrics, path, *, region=None, dpi=140):
    """Fixed reference scales in physical units, with signed image differences."""
    from matplotlib.colors import AsinhNorm

    bands = [band for band in products if band in reference]
    if not bands:
        return
    fig, axes = plt.subplots(len(bands), 3, figsize=(14, 4 * len(bands)), squeeze=False)
    for row, band in enumerate(bands):
        product, base = products[band], reference[band]
        valid_grid, reason = _validate_shared_support([product, base])
        if not valid_grid:
            plt.close(fig)
            raise ValueError(f"cannot compare {product.method}/{band}: {reason}")
        data, ref = product.data, base.data
        if region is not None:
            x0, y0, width, height = region
            if x0 < 0 or y0 < 0 or width <= 0 or height <= 0 or x0 + width > data.shape[1] or y0 + height > data.shape[0]:
                plt.close(fig)
                raise ValueError("region must fit inside the common image grid")
            data, ref = data[y0:y0+height, x0:x0+width], ref[y0:y0+height, x0:x0+width]
        lo, hi = limits[band]
        residual_limit = hi - lo
        panels = (data, data, data - ref)
        for col, image in enumerate(panels):
            ax = axes[row, col]
            cmap = _cmap_with_missing("coolwarm" if col == 2 else "gray")
            options = {"norm": AsinhNorm(linear_width=(hi-lo)/10, vmin=lo, vmax=hi)} if col == 1 else {"vmin": -residual_limit if col == 2 else lo, "vmax": residual_limit if col == 2 else hi}
            artist = ax.imshow(image, origin="lower", cmap=cmap, **options)
            ax.set_title(("linear", "asinh", "signed difference from reference")[col])
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(artist, ax=ax, fraction=0.04, label=_units(product.header, product.metadata))
        axes[row, 0].set_ylabel(_legend_text(product, _metrics_for(metrics, method=product.method, band=band)), rotation=0, ha="right", va="center", fontsize=8)
    fig.suptitle("Gray marks missing coverage; image differences are not detector residuals")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _load_json(path: Path | None) -> Mapping[str, Any]:
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"metrics JSON {path} must contain an object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def _save_synthetic_check(output_path: Path, *, dpi: int = 140) -> None:
    y, x = np.mgrid[-1.0:1.0:128j, -1.0:1.0:128j]
    image = np.exp(-((x + 0.3) ** 2 + (y - 0.2) ** 2) / 0.06) + 0.35 * np.exp(-((x - 0.35) ** 2 + (y + 0.25) ** 2) / 0.02)
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.imshow(image, origin="lower", cmap=_cmap_with_missing("gray"), vmin=0.0, vmax=1.0)
    ax.set_title("Synthetic input check | renderer only")
    ax.set_xlabel("Synthetic coordinates; no science measurement")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _default_bands(manifest: Mapping[str, Any]) -> list[str]:
    values = manifest.get("bands")
    if isinstance(values, list):
        return [str(value).upper() for value in values]
    return [str(item.get("band", "unknown")).upper() for item in manifest.get("outputs", []) if isinstance(item, Mapping)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=_default_manifest_path())
    parser.add_argument("--baseline-cube", type=Path, default=None, help="Default: smacs0723_stack_baseline.fits next to the manifest.")
    parser.add_argument("--comparison-cube", type=Path, default=None, help="Optional INR/reconstruction cube to compare against the baseline cube.")
    parser.add_argument("--products-json", type=Path, default=None, help="JSON list of individual-band or cube products with method/band/path metadata.")
    parser.add_argument("--joint-report", type=Path, help="Joint field-fit JSON report; use its per-band references and validation scores.")
    parser.add_argument("--joint-export-dir", type=Path, help="Joint export directory when absent from the report.")
    parser.add_argument("--product", action="append", default=[], metavar="METHOD=PATH", help="Add a product FITS file; repeat for run methods.")
    parser.add_argument("--metrics-json", type=Path, default=None, help="JSON containing supplied per-method/per-band metrics.")
    parser.add_argument("--regions-json", type=Path, help="Fixed named pixel regions: {name: [x0, y0, width, height]} on the common grid.")
    parser.add_argument("--display-limits", type=Path, help="Reuse display_limits.json from the frozen classical reference report.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: quicklooks/ next to the manifest.")
    parser.add_argument("--max-panels", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=140)
    parser.add_argument("--no-synthetic-check", action="store_true", help="Skip the renderer-only synthetic diagnostic panel.")
    args = parser.parse_args()

    manifest_path = (args.joint_report or args.manifest).expanduser().resolve()
    manifest = {} if args.joint_report else _load_manifest(manifest_path)
    manifest_dir = manifest_path.parent
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir is not None else manifest_dir / "quicklooks"
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "run_metadata.json").exists():
        raise FileExistsError("completed report already exists; choose a new output directory")
    max_panels, dpi = max(int(args.max_panels), 1), int(args.dpi)
    bands, metrics = _default_bands(manifest), _load_json(args.metrics_json)
    joint_entries = []
    if args.joint_report:
        joint_entries, joint_metrics = _joint_products(manifest_path, args.joint_export_dir)
        metrics = {**joint_metrics, **metrics}
        bands = list(dict.fromkeys(entry["band"] for entry in joint_entries))

    cutout_paths, cutout_titles = _manifest_cutout_paths(manifest, manifest_dir=manifest_dir)
    cutout_images = [_read_image(path) for path in cutout_paths]
    cutouts_png = output_dir / "cutouts.png"
    _save_image_grid(cutout_images, cutout_titles, cutouts_png, max_panels=max_panels, dpi=dpi)

    baseline_path = args.baseline_cube.expanduser().resolve() if args.baseline_cube is not None else manifest_dir / "smacs0723_stack_baseline.fits"
    summary: dict[str, Any] = {"manifest": str(manifest_path), "output_dir": str(output_dir), "cutouts_png": str(cutouts_png), "cutout_count": len(cutout_images), "synthetic_input": "renderer-only check" if not args.no_synthetic_check else "disabled"}
    products = _load_product_specs(joint_entries, manifest_dir=manifest_dir, default_bands=bands)
    baseline: np.ndarray | None = None
    if baseline_path.is_file() and not args.joint_report:
        baseline, baseline_header = _read_fits(baseline_path)
        if baseline.ndim != 3:
            raise ValueError(f"baseline cube {baseline_path} must be 3D, got {baseline.shape}")
        baseline_bands = bands[: baseline.shape[0]] if len(bands) >= baseline.shape[0] else [f"plane-{i}" for i in range(baseline.shape[0])]
        products.extend(ProductPlane("baseline", band, baseline[i], baseline_header, str(baseline_path), {}) for i, band in enumerate(baseline_bands))
    references = [product for product in products if product.method == "baseline"]
    if references:
        valid_references, reference_reason = _validate_shared_support(references)
        if not valid_references:
            raise ValueError(f"classical reference bands do not share a common grid: {reference_reason}")
        baseline_bands = [product.band for product in references]
        scale_path = output_dir / "display_limits.json"
        persisted: Mapping[str, Any] = {}
        prior_path = args.display_limits or scale_path
        prior: Mapping[str, Any] = {}
        if args.display_limits and not prior_path.is_file():
            raise FileNotFoundError(f"display limits file does not exist: {prior_path}")
        if prior_path.is_file():
            loaded_prior = json.loads(prior_path.read_text(encoding="utf-8"))
            if isinstance(loaded_prior, Mapping):
                prior = loaded_prior
            if isinstance(prior, Mapping) and isinstance(prior.get("limits"), Mapping):
                persisted = prior["limits"]
        limits = _shared_display_limits({product.band: product.data for product in products if product.method == "baseline"}, persisted=persisted)
        reference_hashes = []
        reference_paths = sorted({product.path for product in references})
        for path in reference_paths:
            with Path(path).open("rb") as stream:
                reference_hashes.append(hashlib.file_digest(stream, "sha256").hexdigest())
        reference_sha256 = reference_hashes[0] if len(reference_hashes) == 1 else hashlib.sha256("".join(reference_hashes).encode()).hexdigest()
        if args.display_limits and prior.get("reference_sha256") != reference_sha256:
            raise ValueError("display scales belong to a different classical reference")
        reference_value: str | list[str] = reference_paths[0] if len(reference_paths) == 1 else reference_paths
        _write_json(scale_path, {"limits": {band: list(value) for band, value in limits.items()}, "reference": reference_value, "reference_sha256": reference_sha256, "policy": "baseline frozen per band"})
        baseline_png = output_dir / "baseline_cube.png"
        _save_image_grid([product.data for product in references], [f"baseline | {band}" for band in baseline_bands], baseline_png, max_panels=max_panels, dpi=dpi, limits=[limits[band] for band in baseline_bands])
        summary.update({"baseline_references": [product.path for product in references], "baseline_png": str(baseline_png), "display_limits": str(scale_path)})
    else:
        limits = {}
        summary["baseline_missing"] = str(baseline_path)

    entries: list[Mapping[str, Any]] = []
    products_payload = args.products_json or manifest.get("products_json")
    if products_payload:
        product_json_path = _resolve_manifest_path(str(products_payload), manifest_dir=manifest_dir)
        entries.extend(_product_entries(json.loads(product_json_path.read_text(encoding="utf-8"))))
    if isinstance(manifest.get("products"), list):
        entries.extend(item for item in manifest["products"] if isinstance(item, Mapping))
    for value in args.product:
        method, path = _parse_product_argument(value)
        entries.append({"method": method, "path": str(path), "bands": bands})
    products.extend(_load_product_specs(entries, manifest_dir=manifest_dir, default_bands=bands))

    comparison_result: dict[str, Any] = {}
    if args.comparison_cube is not None and baseline is not None:
        comparison_path = args.comparison_cube.expanduser().resolve()
        comparison, comparison_header = _read_fits(comparison_path)
        if comparison.ndim != 3:
            raise ValueError(f"comparison cube {comparison_path} must be 3D, got {comparison.shape}")
        comparison_bands = bands[: comparison.shape[0]] if len(bands) >= comparison.shape[0] else [f"plane-{i}" for i in range(comparison.shape[0])]
        products.extend(ProductPlane("comparison", band, comparison[i], comparison_header, str(comparison_path), {}) for i, band in enumerate(comparison_bands))
        comparison_png = output_dir / "comparison_residuals.png"
        comparison_result = _save_comparison_grid(baseline, comparison, comparison_png, max_panels=max_panels, dpi=dpi, limits=[limits[band] for band in comparison_bands], bands=comparison_bands)
        summary.update({"comparison_cube": str(comparison_path), "comparison_shape": list(comparison.shape), "comparison_residuals_png": str(comparison_png)})

    grouped = _group_products(products)
    if products and "baseline" not in grouped:
        raise ValueError("a classical baseline cube is required to fix display scales")
    regions = _load_json(args.regions_json)
    reports = {}
    for method, method_products in grouped.items():
        slug = re.sub(r"[^A-Za-z0-9_-]", "_", method)
        # Add a digest so different labels cannot silently overwrite a report.
        slug += "_" + hashlib.sha256(method.encode()).hexdigest()[:8]
        reports[method] = []
        for region_name, region in {"full": None, **regions}.items():
            if region is not None and (not isinstance(region, list) or len(region) != 4 or any(type(v) is not int for v in region)):
                raise ValueError("regions must be integer [x0, y0, width, height] lists")
            region_slug = re.sub(r"[^A-Za-z0-9_-]", "_", str(region_name))
            path = output_dir / f"{slug}_{region_slug}.png"
            _save_method_report(method_products, grouped["baseline"], limits, metrics, path, region=region, dpi=dpi)
            reports[method].append(str(path))
    summary["method_reports"] = reports
    report_methods = [method for method in grouped if method != "baseline"]
    if report_methods and limits:
        report_images, report_titles, report_limits = [], [], []
        for method in report_methods:
            for band, product in grouped[method].items():
                if band not in limits:
                    continue
                report_images.append(product.data)
                report_titles.append(_legend_text(product, _metrics_for(metrics, method=method, band=band)))
                report_limits.append(limits[band])
        if report_images:
            report_png = output_dir / "product_report.png"
            _save_image_grid(report_images, report_titles, report_png, max_panels=max_panels, dpi=dpi, limits=report_limits)
            summary["product_report_png"] = str(report_png)

    rgb_summary: dict[str, Any] = {}
    for method, method_products in grouped.items():
        if not limits:
            continue
        rgb, status = _six_band_rgb(method_products, limits)
        if rgb is None:
            rgb_summary[method] = {"status": "pending", "reason": status}
            continue
        slug = re.sub(r"[^A-Za-z0-9_-]", "_", method) + "_" + hashlib.sha256(method.encode()).hexdigest()[:8]
        rgb_path = output_dir / f"{slug}_six_band_lupton_rgb.png"
        fig, ax = plt.subplots(figsize=(5.0, 5.0))
        ax.imshow(rgb, origin="lower")
        ax.set_title(f"{method}\nFalse color | native resolution\nB: F090W/F150W; G: F200W/F277W; R: F356W/F444W\nFixed stretch; gray marks missing coverage", fontsize=9)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(rgb_path, dpi=dpi)
        plt.close(fig)
        rgb_summary[method] = {"status": "rendered", "path": str(rgb_path), "mix": {"B": ["F090W", "F150W"], "G": ["F200W", "F277W"], "R": ["F356W", "F444W"]}}
    if rgb_summary:
        summary["six_band_rgb"] = rgb_summary
        summary["resolution_matched_rgb"] = "not measured; requires independent PSF matching kernels"

    if not args.no_synthetic_check:
        synthetic_path = output_dir / "synthetic_input_check.png"
        _save_synthetic_check(synthetic_path, dpi=dpi)
        summary["synthetic_input_check_png"] = str(synthetic_path)

    metadata = {"generated_at_utc": datetime.now(UTC).isoformat(), "manifest": str(manifest_path), "baseline": str(baseline_path) if baseline_path.is_file() else None, "methods": sorted(grouped), "bands": sorted({product.band for product in products}), "display_limits": {band: list(value) for band, value in limits.items()}, "residual_limits": comparison_result, "metrics_json": str(args.metrics_json) if args.metrics_json else None, "products": [{"method": product.method, "band": product.band, "path": product.path, "shape": list(product.data.shape)} for product in products]}
    metadata_path = output_dir / "run_metadata.json"
    _write_json(metadata_path, metadata)
    summary["run_metadata"] = str(metadata_path)
    summary_path = output_dir / "quicklook_summary.json"
    _write_json(summary_path, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
