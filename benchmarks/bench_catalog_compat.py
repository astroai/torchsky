#!/usr/bin/env python3
"""Benchmark torchsky catalog crossmatch against astropy SkyCoord."""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure the repository root is in sys.path so we can import benchmarks.config
repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
src_root = str(Path(__file__).resolve().parent.parent / "src")
if src_root not in sys.path:
    sys.path.insert(0, src_root)

from benchmarks.config import DEFAULT_OUTPUT_DIR  # noqa: E402


import argparse  # noqa: E402
import json  # noqa: E402
from typing import Any, Callable  # noqa: E402

import astropy.units as u  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from astropy.coordinates import SkyCoord  # noqa: E402
from astropy.table import Table  # noqa: E402

import torchsky  # noqa: E402
from torchsky.catalogs import crossmatch_sky  # noqa: E402
import tempfile  # noqa: E402


def _time_many(fn: Callable[[], Any], runs: int) -> float:
    fn()
    samples: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples))


def _has_native_table_io() -> bool:
    try:
        import torchsky.cpp  # type: ignore  # noqa: F401
    except Exception:
        return False
    return True


def _astropy_crossmatch(
    left_ra: torch.Tensor,
    left_dec: torch.Tensor,
    right_ra: torch.Tensor,
    right_dec: torch.Tensor,
    *,
    radius_arcsec: float,
) -> tuple[list[int], list[int], np.ndarray]:
    if left_ra.numel() == 0 or right_ra.numel() == 0:
        return [], [], np.empty(0, dtype=np.float64)

    left = SkyCoord(
        ra=left_ra.cpu().numpy() * u.deg, dec=left_dec.cpu().numpy() * u.deg
    )
    right = SkyCoord(
        ra=right_ra.cpu().numpy() * u.deg,
        dec=right_dec.cpu().numpy() * u.deg,
    )
    right_idx, sep2d, _ = left.match_to_catalog_sky(right)
    keep = sep2d.arcsec <= radius_arcsec
    return (
        np.nonzero(keep)[0].tolist(),
        right_idx[keep].tolist(),
        np.asarray(sep2d.arcsec[keep], dtype=np.float64),
    )


def _assert_same_matches(
    left_ra: torch.Tensor,
    left_dec: torch.Tensor,
    right_ra: torch.Tensor,
    right_dec: torch.Tensor,
    *,
    radius_arcsec: float,
) -> tuple[int, float]:
    torchsky_result = crossmatch_sky(
        left_ra,
        left_dec,
        right_ra,
        right_dec,
        radius_arcsec=radius_arcsec,
    )
    astropy_left, astropy_right, astropy_sep = _astropy_crossmatch(
        left_ra,
        left_dec,
        right_ra,
        right_dec,
        radius_arcsec=radius_arcsec,
    )
    if torchsky_result.left_index.tolist() != astropy_left:
        raise RuntimeError("left_index mismatch between torchsky and astropy")
    if torchsky_result.right_index.tolist() != astropy_right:
        raise RuntimeError("right_index mismatch between torchsky and astropy")
    max_sep_diff = 0.0
    if astropy_sep.size > 0:
        diff = np.abs(torchsky_result.separation_arcsec.cpu().numpy() - astropy_sep)
        max_sep_diff = float(diff.max())
        np.testing.assert_allclose(
            torchsky_result.separation_arcsec.cpu().numpy(),
            astropy_sep,
            atol=1e-6,
            rtol=0.0,
        )
    return int(torchsky_result.left_index.numel()), max_sep_diff


def _random_sphere(
    rng: np.random.Generator,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    lon = rng.uniform(0.0, 360.0, size=n)
    lat = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, size=n)))
    return lon.astype(np.float64), lat.astype(np.float64)


def _make_synthetic_catalogs(
    *,
    n_left: int,
    n_right_noise: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    left_ra = np.remainder(150.0 + rng.normal(0.0, 0.25, size=n_left), 360.0)
    left_dec = np.clip(-2.0 + rng.normal(0.0, 0.18, size=n_left), -89.999, 89.999)

    matched = min(max(16, n_left // 2), n_left)
    matched_ra = np.remainder(
        left_ra[:matched] + rng.normal(0.0, 0.00008, size=matched), 360.0
    )
    matched_dec = np.clip(
        left_dec[:matched] + rng.normal(0.0, 0.00008, size=matched),
        -89.999,
        89.999,
    )

    noise_ra, noise_dec = _random_sphere(rng, n_right_noise)
    right_ra = np.concatenate((matched_ra, noise_ra)).astype(np.float64)
    right_dec = np.concatenate((matched_dec, noise_dec)).astype(np.float64)
    return (
        torch.from_numpy(left_ra.astype(np.float64)),
        torch.from_numpy(left_dec.astype(np.float64)),
        torch.from_numpy(right_ra),
        torch.from_numpy(right_dec),
    )


def _catalog_fixture_dir() -> Path:
    return repo_root_path() / "tests" / "data" / "upstream_astropy_catalogs"


def repo_root_path() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_real_catalog(
    filename: str,
    *,
    ra_col: str,
    dec_col: str,
) -> tuple[Table, torch.Tensor, torch.Tensor]:
    table = Table.read(_catalog_fixture_dir() / filename, format="votable")
    ra = torch.from_numpy(np.asarray(table[ra_col], dtype=np.float64))
    dec = torch.from_numpy(np.asarray(table[dec_col], dtype=np.float64))
    return table, ra, dec


def _make_real_catalogs() -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    _, left_ra, left_dec = _load_real_catalog(
        "irsa-nph-m31.xml",
        ra_col="ra",
        dec_col="dec",
    )
    _, vizier_ra, vizier_dec = _load_real_catalog(
        "vizier_b2_votable.xml",
        ra_col="RAJ2000",
        dec_col="DEJ2000",
    )
    _, gaia_ra, gaia_dec = _load_real_catalog(
        "binary2_masked_strings.xml",
        ra_col="ra",
        dec_col="dec",
    )

    perm = torch.tensor([6, 1, 11, 3, 14, 9, 0, 7, 12, 5, 15, 8, 2, 16, 4, 10, 13, 17])
    right_ra = torch.cat((left_ra[perm], vizier_ra, gaia_ra))
    right_dec = torch.cat((left_dec[perm], vizier_dec, gaia_dec))
    return left_ra, left_dec, right_ra, right_dec


def _make_real_fits_tables() -> tuple[Table, Table]:
    irsa_table, irsa_ra, irsa_dec = _load_real_catalog(
        "irsa-nph-m31.xml",
        ra_col="ra",
        dec_col="dec",
    )
    vizier_table, vizier_ra, vizier_dec = _load_real_catalog(
        "vizier_b2_votable.xml",
        ra_col="RAJ2000",
        dec_col="DEJ2000",
    )
    gaia_table, gaia_ra, gaia_dec = _load_real_catalog(
        "binary2_masked_strings.xml",
        ra_col="ra",
        dec_col="dec",
    )

    perm = np.array([6, 1, 11, 3, 14, 9, 0, 7, 12, 5, 15, 8, 2, 16, 4, 10, 13, 17])
    left_table = Table(
        {
            "RA": irsa_ra.numpy(),
            "DEC": irsa_dec.numpy(),
            "SOURCE": np.full(len(irsa_table), "irsa_2mass_psc", dtype=object),
            "DESIGNATION": np.asarray(irsa_table["designation"]).astype(str),
            "J_MAG": np.asarray(irsa_table["j_m"], dtype=np.float64),
        }
    )
    right_table = Table(
        {
            "RA": np.concatenate(
                (
                    irsa_ra.numpy()[perm],
                    vizier_ra.numpy(),
                    gaia_ra.numpy(),
                )
            ),
            "DEC": np.concatenate(
                (
                    irsa_dec.numpy()[perm],
                    vizier_dec.numpy(),
                    gaia_dec.numpy(),
                )
            ),
            "SOURCE": np.concatenate(
                (
                    np.full(len(perm), "irsa_2mass_psc", dtype=object),
                    np.full(len(vizier_table), "vizier_mash", dtype=object),
                    np.full(len(gaia_table), "gaia_dr2", dtype=object),
                )
            ),
            "LABEL": np.concatenate(
                (
                    np.asarray(irsa_table["designation"]).astype(str)[perm],
                    np.asarray(vizier_table["PNG"]).astype(str),
                    np.asarray(gaia_table["source_id"]).astype(str),
                )
            ),
        }
    )
    return left_table, right_table


def _load_torchsky_catalog_from_fits(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    table = torchsky.table.read(str(path), columns=["RA", "DEC"])
    return table["RA"].to(torch.float64), table["DEC"].to(torch.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-left", type=int, default=10_000)
    parser.add_argument("--n-right-noise", type=int, default=80_000)
    parser.add_argument("--radius-arcsec", type=float, default=2.0)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "catalog_compat.json",
    )
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    left_ra, left_dec, right_ra, right_dec = _make_synthetic_catalogs(
        n_left=args.n_left,
        n_right_noise=args.n_right_noise,
        seed=args.seed,
    )

    matched_rows, max_sep_diff = _assert_same_matches(
        left_ra,
        left_dec,
        right_ra,
        right_dec,
        radius_arcsec=args.radius_arcsec,
    )
    t_torchsky = _time_many(
        lambda: crossmatch_sky(
            left_ra,
            left_dec,
            right_ra,
            right_dec,
            radius_arcsec=args.radius_arcsec,
        ),
        runs=args.runs,
    )
    t_astropy = _time_many(
        lambda: _astropy_crossmatch(
            left_ra,
            left_dec,
            right_ra,
            right_dec,
            radius_arcsec=args.radius_arcsec,
        ),
        runs=max(2, args.runs),
    )
    rows.extend(
        [
            {
                "case": "synthetic_clustered",
                "library": "torchsky",
                "status": "OK",
                "left_points": int(left_ra.numel()),
                "right_points": int(right_ra.numel()),
                "matched_rows": matched_rows,
                "radius_arcsec": args.radius_arcsec,
                "time_s": t_torchsky,
                "crossmatches_s": left_ra.numel() / t_torchsky,
                "max_sep_diff_arcsec_vs_astropy": max_sep_diff,
            },
            {
                "case": "synthetic_clustered",
                "library": "astropy",
                "status": "OK",
                "left_points": int(left_ra.numel()),
                "right_points": int(right_ra.numel()),
                "matched_rows": matched_rows,
                "radius_arcsec": args.radius_arcsec,
                "time_s": t_astropy,
                "crossmatches_s": left_ra.numel() / t_astropy,
                "speed_ratio_torchsky_vs_astropy": t_astropy / t_torchsky,
            },
        ]
    )

    real_left_ra, real_left_dec, real_right_ra, real_right_dec = _make_real_catalogs()
    matched_rows_real, max_sep_diff_real = _assert_same_matches(
        real_left_ra,
        real_left_dec,
        real_right_ra,
        real_right_dec,
        radius_arcsec=0.1,
    )
    t_torchsky_real = _time_many(
        lambda: crossmatch_sky(
            real_left_ra,
            real_left_dec,
            real_right_ra,
            real_right_dec,
            radius_arcsec=0.1,
        ),
        runs=args.runs,
    )
    t_astropy_real = _time_many(
        lambda: _astropy_crossmatch(
            real_left_ra,
            real_left_dec,
            real_right_ra,
            real_right_dec,
            radius_arcsec=0.1,
        ),
        runs=max(2, args.runs),
    )
    rows.extend(
        [
            {
                "case": "real_astropy_votable_catalogs",
                "library": "torchsky",
                "status": "OK",
                "left_points": int(real_left_ra.numel()),
                "right_points": int(real_right_ra.numel()),
                "matched_rows": matched_rows_real,
                "radius_arcsec": 0.1,
                "time_s": t_torchsky_real,
                "crossmatches_s": real_left_ra.numel() / t_torchsky_real,
                "max_sep_diff_arcsec_vs_astropy": max_sep_diff_real,
            },
            {
                "case": "real_astropy_votable_catalogs",
                "library": "astropy",
                "status": "OK",
                "left_points": int(real_left_ra.numel()),
                "right_points": int(real_right_ra.numel()),
                "matched_rows": matched_rows_real,
                "radius_arcsec": 0.1,
                "time_s": t_astropy_real,
                "crossmatches_s": real_left_ra.numel() / t_astropy_real,
                "speed_ratio_torchsky_vs_astropy": t_astropy_real / t_torchsky_real,
            },
        ]
    )

    if _has_native_table_io():
        left_table, right_table = _make_real_fits_tables()
        with tempfile.TemporaryDirectory() as tmpdir:
            left_path = Path(tmpdir) / "left_catalog.fits"
            right_path = Path(tmpdir) / "right_catalog.fits"
            left_table.write(left_path, format="fits", overwrite=True)
            right_table.write(right_path, format="fits", overwrite=True)

            left_ra_fit, left_dec_fit = _load_torchsky_catalog_from_fits(left_path)
            right_ra_fit, right_dec_fit = _load_torchsky_catalog_from_fits(right_path)

            matched_rows_fit, max_sep_diff_fit = _assert_same_matches(
                left_ra_fit,
                left_dec_fit,
                right_ra_fit,
                right_dec_fit,
                radius_arcsec=args.radius_arcsec,
            )
            t_torchsky_fit = _time_many(
                lambda: crossmatch_sky(
                    left_ra_fit,
                    left_dec_fit,
                    right_ra_fit,
                    right_dec_fit,
                    radius_arcsec=args.radius_arcsec,
                ),
                runs=args.runs,
            )
            t_astropy_fit = _time_many(
                lambda: _astropy_crossmatch(
                    left_ra_fit,
                    left_dec_fit,
                    right_ra_fit,
                    right_dec_fit,
                    radius_arcsec=args.radius_arcsec,
                ),
                runs=max(2, args.runs),
            )
            rows.extend(
                [
                    {
                        "case": "real_astropy_votable_fits_roundtrip",
                        "library": "torchsky",
                        "status": "OK",
                        "left_points": int(left_ra_fit.numel()),
                        "right_points": int(right_ra_fit.numel()),
                        "matched_rows": matched_rows_fit,
                        "radius_arcsec": args.radius_arcsec,
                        "time_s": t_torchsky_fit,
                        "crossmatches_s": left_ra_fit.numel() / t_torchsky_fit,
                        "max_sep_diff_arcsec_vs_astropy": max_sep_diff_fit,
                    },
                    {
                        "case": "real_astropy_votable_fits_roundtrip",
                        "library": "astropy",
                        "status": "OK",
                        "left_points": int(left_ra_fit.numel()),
                        "right_points": int(right_ra_fit.numel()),
                        "matched_rows": matched_rows_fit,
                        "radius_arcsec": args.radius_arcsec,
                        "time_s": t_astropy_fit,
                        "crossmatches_s": left_ra_fit.numel() / t_astropy_fit,
                        "speed_ratio_torchsky_vs_astropy": t_astropy_fit
                        / t_torchsky_fit,
                    },
                ]
            )
    else:
        rows.append(
            {
                "case": "real_astropy_votable_fits_roundtrip",
                "library": "torchsky",
                "status": "SKIPPED",
                "reason": "torchsky native FITS table extension unavailable",
            }
        )

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(rows, indent=2))

    for row in rows:
        if row.get("status") != "OK":
            print(
                f"case={row['case']}, library={row['library']}, status={row['status']}, reason={row.get('reason', '')}"
            )
            continue
        print(
            f"case={row['case']}, library={row['library']}, "
            f"time_s={row['time_s']:.4f}, crossmatches_s={row['crossmatches_s']:.2f}"
        )
    print(f"JSON: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
