#!/usr/bin/env python3
"""Plot diagnostic figures from WCS-aware INR held-out validation samples."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


BANDS = ("F090W", "F150W", "F200W", "F277W", "F356W", "F444W")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples",
        type=Path,
        required=True,
        help="validation_residual_samples.npz",
    )
    parser.add_argument(
        "--per-exposure-csv",
        type=Path,
        required=True,
        help="validation_per_exposure.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--gridsize",
        type=int,
        default=75,
        help="Hexbin resolution.",
    )
    parser.add_argument(
        "--max-points-per-band",
        type=int,
        default=200000,
        help="Maximum plotted points per band; sampling is deterministic.",
    )
    return parser


def read_exposure_metadata(path: Path) -> dict[int, dict[str, str]]:
    rows: dict[int, dict[str, str]] = {}

    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            exposure_index = int(row["exposure_index"])
            rows[exposure_index] = row

    return rows


def deterministic_subsample(
    indices: np.ndarray,
    *,
    maximum: int,
    seed: int,
) -> np.ndarray:
    if maximum <= 0 or indices.size <= maximum:
        return indices

    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=maximum, replace=False))


def finite_log_values(
    values: np.ndarray,
    *,
    floor_quantile: float = 0.001,
) -> tuple[np.ndarray, np.ndarray]:
    """Return log10 positive values and the corresponding finite mask."""

    values = np.asarray(values, dtype=np.float64)

    valid = np.isfinite(values) & (values > 0.0)
    positive = values[valid]

    if positive.size == 0:
        raise ValueError("No finite positive values available.")

    floor = float(np.quantile(positive, floor_quantile))
    floor = max(floor, np.finfo(np.float64).tiny)

    clipped = np.maximum(values, floor)

    valid &= np.isfinite(clipped)
    logged = np.full(values.shape, np.nan, dtype=np.float64)
    logged[valid] = np.log10(clipped[valid])

    return logged, valid


def robust_plot_limits(
    values: np.ndarray,
    *,
    lower: float = 0.002,
    upper: float = 0.998,
    padding: float = 0.04,
) -> tuple[float, float]:
    values = np.asarray(values)
    values = values[np.isfinite(values)]

    lo = float(np.quantile(values, lower))
    hi = float(np.quantile(values, upper))

    if hi <= lo:
        hi = lo + 1.0

    margin = padding * (hi - lo)
    return lo - margin, hi + margin


def add_binned_trend(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    bins: int = 24,
) -> None:
    """Overlay median and 90th-percentile trend lines."""

    x = np.asarray(x)
    y = np.asarray(y)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if x.size < 100:
        return

    lo, hi = np.quantile(x, [0.01, 0.99])
    edges = np.linspace(lo, hi, bins + 1)

    centers: list[float] = []
    medians: list[float] = []
    q90s: list[float] = []

    for left, right in zip(edges[:-1], edges[1:]):
        selected = (x >= left) & (x < right)

        if selected.sum() < 30:
            continue

        centers.append(0.5 * (left + right))
        medians.append(float(np.median(y[selected])))
        q90s.append(float(np.quantile(y[selected], 0.90)))

    if centers:
        ax.plot(centers, medians, linewidth=2.0, label="median |z|")
        ax.plot(
            centers,
            q90s,
            linewidth=1.5,
            linestyle="--",
            label="90th percentile |z|",
        )


def plot_abs_z_vs_quantity(
    *,
    quantity: np.ndarray,
    abs_z: np.ndarray,
    band_id: np.ndarray,
    band_names: np.ndarray,
    output_path: Path,
    x_label: str,
    gridsize: int,
    max_points_per_band: int,
) -> None:
    """Make one 2x3 hexbin figure for all six bands."""

    log_quantity, valid_quantity = finite_log_values(quantity)
    log_abs_z, valid_z = finite_log_values(abs_z)

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(17, 10),
        sharey=True,
    )

    last_hexbin = None

    for panel_index, band in enumerate(BANDS):
        ax = axes.flat[panel_index]

        matching_band_ids = np.flatnonzero(band_names == band)
        if matching_band_ids.size != 1:
            ax.set_visible(False)
            continue

        current_band_id = int(matching_band_ids[0])

        indices = np.flatnonzero(
            (band_id == current_band_id)
            & valid_quantity
            & valid_z
        )

        indices = deterministic_subsample(
            indices,
            maximum=max_points_per_band,
            seed=20260718 + panel_index,
        )

        x = log_quantity[indices]
        y = log_abs_z[indices]

        x_limits = robust_plot_limits(x)
        y_limits = robust_plot_limits(
            y,
            lower=0.001,
            upper=0.995,
        )

        last_hexbin = ax.hexbin(
            x,
            y,
            gridsize=gridsize,
            bins="log",
            mincnt=1,
        )

        add_binned_trend(ax, x, y)

        ax.axhline(np.log10(1.0), linestyle=":", linewidth=1.0)
        ax.axhline(np.log10(3.0), linestyle=":", linewidth=1.0)
        ax.axhline(np.log10(5.0), linestyle=":", linewidth=1.0)

        ax.text(
            0.02,
            np.log10(1.0),
            "1σ",
            transform=ax.get_yaxis_transform(),
            va="bottom",
        )
        ax.text(
            0.02,
            np.log10(3.0),
            "3σ",
            transform=ax.get_yaxis_transform(),
            va="bottom",
        )
        ax.text(
            0.02,
            np.log10(5.0),
            "5σ",
            transform=ax.get_yaxis_transform(),
            va="bottom",
        )

        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.set_title(f"{band}   n={indices.size:,}")
        ax.grid(alpha=0.2)
        ax.legend(loc="upper left", fontsize=8)

        if panel_index >= 3:
            ax.set_xlabel(x_label)

        if panel_index % 3 == 0:
            ax.set_ylabel(r"$\log_{10}|z|$")

    if last_hexbin is not None:
        colorbar = figure.colorbar(
            last_hexbin,
            ax=axes.ravel().tolist(),
            shrink=0.88,
        )
        colorbar.set_label("log10(number of validation pixels per hexagon)")

    figure.suptitle(
        r"$|z| = |\hat y-y|/\mathrm{ERR}$",
        fontsize=15,
    )

    figure.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)


def sky_coordinates_for_samples(
    *,
    x: np.ndarray,
    y: np.ndarray,
    exposure_index: np.ndarray,
    metadata: dict[int, dict[str, str]],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert every validation sample from detector coordinates to RA/Dec."""

    ra = np.full(x.shape, np.nan, dtype=np.float64)
    dec = np.full(y.shape, np.nan, dtype=np.float64)

    for current_exposure in np.unique(exposure_index):
        current_exposure = int(current_exposure)

        if current_exposure not in metadata:
            raise KeyError(
                f"Exposure {current_exposure} missing from CSV metadata."
            )

        fits_path = Path(metadata[current_exposure]["path"])

        with fits.open(fits_path, memmap=False) as hdul:
            if "SCI" not in hdul:
                raise ValueError(f"{fits_path} has no SCI extension.")

            wcs = WCS(hdul["SCI"].header).celestial

        selected = exposure_index == current_exposure

        current_ra, current_dec = wcs.pixel_to_world_values(
            x[selected].astype(np.float64),
            y[selected].astype(np.float64),
        )

        ra[selected] = np.asarray(current_ra, dtype=np.float64)
        dec[selected] = np.asarray(current_dec, dtype=np.float64)

        print(
            f"WCS transformed exposure {current_exposure}: "
            f"{selected.sum():,} samples"
        )

    return ra, dec


def plot_spatial_maps(
    *,
    ra: np.ndarray,
    dec: np.ndarray,
    z: np.ndarray,
    band_id: np.ndarray,
    band_names: np.ndarray,
    output_dir: Path,
    gridsize: int,
    max_points_per_band: int,
) -> None:
    finite = (
        np.isfinite(ra)
        & np.isfinite(dec)
        & np.isfinite(z)
    )

    # ---------------------------------------------------------
    # Signed median residual map
    # ---------------------------------------------------------
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(17, 10),
    )

    last_hexbin = None

    for panel_index, band in enumerate(BANDS):
        ax = axes.flat[panel_index]

        matching_band_ids = np.flatnonzero(band_names == band)
        if matching_band_ids.size != 1:
            ax.set_visible(False)
            continue

        current_band_id = int(matching_band_ids[0])

        indices = np.flatnonzero(
            finite & (band_id == current_band_id)
        )

        indices = deterministic_subsample(
            indices,
            maximum=max_points_per_band,
            seed=20260818 + panel_index,
        )

        last_hexbin = ax.hexbin(
            ra[indices],
            dec[indices],
            C=np.clip(z[indices], -10.0, 10.0),
            reduce_C_function=np.median,
            gridsize=gridsize,
            mincnt=3,
            cmap="coolwarm",
            vmin=-3.0,
            vmax=3.0,
        )

        ax.set_title(f"{band}: median signed z")
        ax.set_xlabel("RA [deg]")
        ax.set_ylabel("Dec [deg]")
        ax.invert_xaxis()
        ax.set_aspect("equal", adjustable="box")

    if last_hexbin is not None:
        colorbar = figure.colorbar(
            last_hexbin,
            ax=axes.ravel().tolist(),
            shrink=0.88,
        )
        colorbar.set_label(
            r"median signed $z=(\hat y-y)/\mathrm{ERR}$"
        )

    figure.suptitle(
        "Spatial median standardized residual",
        fontsize=15,
    )

    figure.savefig(
        output_dir / "spatial_median_signed_z.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)

    # ---------------------------------------------------------
    # Fraction of samples exceeding 3 sigma
    # ---------------------------------------------------------
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(17, 10),
    )

    last_hexbin = None

    for panel_index, band in enumerate(BANDS):
        ax = axes.flat[panel_index]

        matching_band_ids = np.flatnonzero(band_names == band)
        if matching_band_ids.size != 1:
            ax.set_visible(False)
            continue

        current_band_id = int(matching_band_ids[0])

        indices = np.flatnonzero(
            finite & (band_id == current_band_id)
        )

        indices = deterministic_subsample(
            indices,
            maximum=max_points_per_band,
            seed=20260918 + panel_index,
        )

        outlier = (np.abs(z[indices]) > 3.0).astype(np.float64)

        last_hexbin = ax.hexbin(
            ra[indices],
            dec[indices],
            C=outlier,
            reduce_C_function=np.mean,
            gridsize=gridsize,
            mincnt=3,
            cmap="magma",
            vmin=0.0,
            vmax=0.5,
        )

        ax.set_title(f"{band}: fraction |z| > 3σ")
        ax.set_xlabel("RA [deg]")
        ax.set_ylabel("Dec [deg]")
        ax.invert_xaxis()
        ax.set_aspect("equal", adjustable="box")

    if last_hexbin is not None:
        colorbar = figure.colorbar(
            last_hexbin,
            ax=axes.ravel().tolist(),
            shrink=0.88,
        )
        colorbar.set_label("fraction of validation pixels with |z| > 3")

    figure.suptitle(
        "Spatial concentration of >3σ validation residuals",
        fontsize=15,
    )

    figure.savefig(
        output_dir / "spatial_fraction_abs_z_gt_3.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> int:
    args = build_parser().parse_args()

    samples_path = args.samples.expanduser().resolve()
    exposure_csv = args.per_exposure_csv.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(samples_path, allow_pickle=False)

    exposure_index = np.asarray(
        data["exposure_index"],
        dtype=np.int64,
    )
    band_id = np.asarray(data["band_id"], dtype=np.int64)
    band_names = np.asarray(data["band_names"]).astype(str)

    x = np.asarray(data["x"], dtype=np.int32)
    y = np.asarray(data["y"], dtype=np.int32)

    target = np.asarray(
        data["target_physical"],
        dtype=np.float64,
    )
    residual = np.asarray(
        data["residual_physical"],
        dtype=np.float64,
    )
    z = np.asarray(data["z"], dtype=np.float64)

    abs_z = np.abs(z)

    # CRF values can be negative after background subtraction.
    # We use absolute target intensity to measure brightness magnitude.
    target_brightness = np.abs(target)

    # Since z = residual / ERR:
    #
    #     ERR = |residual| / |z|
    #
    # z=0 and residual=0 cannot determine ERR, so those points are omitted.
    err = np.full(z.shape, np.nan, dtype=np.float64)

    reconstructable_err = (
        np.isfinite(residual)
        & np.isfinite(z)
        & (np.abs(z) > 1.0e-12)
    )

    err[reconstructable_err] = (
        np.abs(residual[reconstructable_err])
        / np.abs(z[reconstructable_err])
    )

    print(f"Loaded validation samples: {z.size:,}")
    print(f"Finite reconstructed ERR: {np.isfinite(err).sum():,}")

    plot_abs_z_vs_quantity(
        quantity=target_brightness,
        abs_z=abs_z,
        band_id=band_id,
        band_names=band_names,
        output_path=output_dir / "abs_z_vs_target_brightness.png",
        x_label=r"$\log_{10}|\mathrm{target\ intensity}|$",
        gridsize=args.gridsize,
        max_points_per_band=args.max_points_per_band,
    )

    plot_abs_z_vs_quantity(
        quantity=err,
        abs_z=abs_z,
        band_id=band_id,
        band_names=band_names,
        output_path=output_dir / "abs_z_vs_err.png",
        x_label=r"$\log_{10}(\mathrm{ERR})$",
        gridsize=args.gridsize,
        max_points_per_band=args.max_points_per_band,
    )

    metadata = read_exposure_metadata(exposure_csv)

    ra, dec = sky_coordinates_for_samples(
        x=x,
        y=y,
        exposure_index=exposure_index,
        metadata=metadata,
    )

    np.savez_compressed(
        output_dir / "validation_samples_with_sky_coordinates.npz",
        exposure_index=exposure_index,
        band_id=band_id,
        band_names=band_names,
        x=x,
        y=y,
        ra=ra,
        dec=dec,
        target_physical=target,
        residual_physical=residual,
        err_physical=err,
        z=z,
    )

    plot_spatial_maps(
        ra=ra,
        dec=dec,
        z=z,
        band_id=band_id,
        band_names=band_names,
        output_dir=output_dir,
        gridsize=args.gridsize,
        max_points_per_band=args.max_points_per_band,
    )

    print("\nCreated:")
    for path in sorted(output_dir.glob("*.png")):
        print(f"  {path}")

    print(
        "  "
        + str(
            output_dir
            / "validation_samples_with_sky_coordinates.npz"
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
