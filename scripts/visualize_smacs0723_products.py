#!/usr/bin/env python3
"""Write quicklook PNGs for SMACS J0723 cutouts, stack baselines, and INR cubes.

This is intentionally a script, not a notebook, so a student can check products
right after a command-line run. It writes:

- ``cutouts.png``: one panel per input band from ``manifest.json``
- ``baseline_cube.png``: one panel per baseline cube wavelength plane
- ``baseline_false_color.png``: approximate color composite from the cube
- ``comparison_residuals.png``: optional baseline / model / residual panels
- ``quicklook_summary.json``: file paths and product shapes

Examples::

  pixi run -e test -- python scripts/visualize_smacs0723_products.py
  pixi run -e test -- python scripts/visualize_smacs0723_products.py --comparison-cube hyperspectral_reconstruction.fits
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from astropy.io import fits

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


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
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _read_image(path: Path) -> np.ndarray:
    data = fits.getdata(path, ext=0)
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{path} must contain a 2D image, got shape {arr.shape}")
    return arr


def _read_cube(path: Path) -> np.ndarray:
    data = fits.getdata(path, ext=0)
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"{path} must contain a 3D cube, got shape {arr.shape}")
    return arr


def _display_limits(image: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(image[np.isfinite(image)], dtype=np.float32)
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        med = float(np.nanmedian(finite))
        return med - 1.0, med + 1.0
    return float(lo), float(hi)


def _normalize_for_rgb(image: np.ndarray) -> np.ndarray:
    lo, hi = _display_limits(image)
    scaled = (np.asarray(image, dtype=np.float32) - lo) / max(hi - lo, 1.0e-8)
    return np.clip(scaled, 0.0, 1.0)


def _grid_shape(n_panel: int) -> tuple[int, int]:
    cols = int(np.ceil(np.sqrt(max(n_panel, 1))))
    rows = int(np.ceil(max(n_panel, 1) / cols))
    return rows, cols


def _save_image_grid(
    images: list[np.ndarray],
    titles: list[str],
    output_path: Path,
    *,
    max_panels: int,
    cmap: str = "gray",
    dpi: int = 140,
) -> None:
    if not images:
        return
    shown_images = images[:max_panels]
    shown_titles = titles[:max_panels]
    rows, cols = _grid_shape(len(shown_images))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 4.0 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, image, title in zip(axes.ravel(), shown_images, shown_titles, strict=True):
        lo, hi = _display_limits(image)
        ax.imshow(image, origin="lower", cmap=cmap, vmin=lo, vmax=hi)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _save_false_color(cube: np.ndarray, output_path: Path, *, dpi: int = 140) -> None:
    if cube.shape[0] == 1:
        rgb = np.repeat(_normalize_for_rgb(cube[0])[..., None], 3, axis=-1)
    else:
        red = _normalize_for_rgb(cube[-1])
        blue = _normalize_for_rgb(cube[0])
        green = _normalize_for_rgb(cube[cube.shape[0] // 2])
        rgb = np.stack([red, green, blue], axis=-1)
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.imshow(rgb, origin="lower")
    ax.set_title("Approximate false color")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _save_comparison_grid(
    baseline: np.ndarray,
    comparison: np.ndarray,
    output_path: Path,
    *,
    max_panels: int,
    dpi: int = 140,
) -> None:
    if baseline.shape != comparison.shape:
        raise ValueError(
            f"comparison cube shape {comparison.shape} must match baseline shape {baseline.shape}"
        )
    n_plane = min(int(baseline.shape[0]), int(max_panels))
    fig, axes = plt.subplots(n_plane, 3, figsize=(12.0, 3.5 * n_plane), squeeze=False)
    for i in range(n_plane):
        residual = comparison[i] - baseline[i]
        panels = (
            (baseline[i], f"baseline plane {i}"),
            (comparison[i], f"comparison plane {i}"),
            (residual, f"comparison - baseline {i}"),
        )
        for ax, (image, title) in zip(axes[i], panels, strict=True):
            lo, hi = _display_limits(image)
            ax.imshow(image, origin="lower", cmap="gray", vmin=lo, vmax=hi)
            ax.set_title(title)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _manifest_cutout_paths(
    manifest: dict[str, Any], *, manifest_dir: Path
) -> tuple[list[Path], list[str]]:
    paths: list[Path] = []
    titles: list[str] = []
    for item in manifest.get("outputs", []):
        paths.append(_resolve_manifest_path(str(item["path"]), manifest_dir=manifest_dir))
        band = str(item.get("band", "unknown"))
        origin = item.get("crop_origin_xy")
        suffix = f" origin={origin}" if origin is not None else ""
        titles.append(f"{band}{suffix}")
    return paths, titles


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--manifest", type=Path, default=_default_manifest_path())
    parser.add_argument(
        "--baseline-cube",
        type=Path,
        default=None,
        help="Default: smacs0723_stack_baseline.fits next to the manifest.",
    )
    parser.add_argument(
        "--comparison-cube",
        type=Path,
        default=None,
        help="Optional INR/reconstruction cube to compare against the baseline cube.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: quicklooks/ next to the manifest.",
    )
    parser.add_argument("--max-panels", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=140)
    args = parser.parse_args()

    manifest_path = args.manifest.expanduser().resolve()
    manifest = _load_manifest(manifest_path)
    manifest_dir = manifest_path.parent
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else manifest_dir / "quicklooks"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cutout_paths, cutout_titles = _manifest_cutout_paths(
        manifest, manifest_dir=manifest_dir
    )
    cutout_images = [_read_image(path) for path in cutout_paths]
    cutouts_png = output_dir / "cutouts.png"
    _save_image_grid(
        cutout_images,
        cutout_titles,
        cutouts_png,
        max_panels=max(int(args.max_panels), 1),
        dpi=int(args.dpi),
    )

    baseline_path = (
        args.baseline_cube.expanduser().resolve()
        if args.baseline_cube is not None
        else manifest_dir / "smacs0723_stack_baseline.fits"
    )
    summary: dict[str, Any] = {
        "manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "cutouts_png": str(cutouts_png),
        "cutout_count": len(cutout_images),
    }
    if baseline_path.is_file():
        baseline = _read_cube(baseline_path)
        baseline_png = output_dir / "baseline_cube.png"
        _save_image_grid(
            [baseline[i] for i in range(int(baseline.shape[0]))],
            [f"baseline plane {i}" for i in range(int(baseline.shape[0]))],
            baseline_png,
            max_panels=max(int(args.max_panels), 1),
            dpi=int(args.dpi),
        )
        false_color_png = output_dir / "baseline_false_color.png"
        _save_false_color(baseline, false_color_png, dpi=int(args.dpi))
        summary.update(
            {
                "baseline_cube": str(baseline_path),
                "baseline_shape": list(baseline.shape),
                "baseline_png": str(baseline_png),
                "baseline_false_color_png": str(false_color_png),
            }
        )
        if args.comparison_cube is not None:
            comparison_path = args.comparison_cube.expanduser().resolve()
            comparison = _read_cube(comparison_path)
            comparison_png = output_dir / "comparison_residuals.png"
            _save_comparison_grid(
                baseline,
                comparison,
                comparison_png,
                max_panels=max(int(args.max_panels), 1),
                dpi=int(args.dpi),
            )
            summary.update(
                {
                    "comparison_cube": str(comparison_path),
                    "comparison_shape": list(comparison.shape),
                    "comparison_residuals_png": str(comparison_png),
                }
            )
    else:
        summary["baseline_missing"] = str(baseline_path)

    summary_path = output_dir / "quicklook_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
