#!/usr/bin/env python3
"""Benchmark image-plane mapmaking on synthetic and real FITS inputs."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import astropy  # noqa: E402

import torchsky  # noqa: E402
from torchsky.map import (  # noqa: E402
    ImageObservation,
    coadd_image_plane,
    coadd_image_plane_cube,
    coadd_image_plane_multiband,
    plan_image_plane_cube_target_from_header,
    plan_image_plane_target,
)

try:  # noqa: E402
    import torchsky.cpp  # noqa: F401
except Exception as exc:  # pragma: no cover - native bench env requirement
    raise RuntimeError(
        "bench_map_image_plane.py requires torchsky.cpp; "
        "run it from a native-extension environment"
    ) from exc


def _simple_header(size: int) -> dict[str, Any]:
    return {
        "NAXIS": 2,
        "NAXIS1": size,
        "NAXIS2": size,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": float(size // 2),
        "CRPIX2": float(size // 2),
        "CRVAL1": 12.0,
        "CRVAL2": -3.0,
        "CDELT1": -0.01,
        "CDELT2": 0.01,
    }


def _rotated_cd(scale_deg: float, rotation_deg: float) -> tuple[float, float, float, float]:
    theta = torch.deg2rad(torch.tensor(float(rotation_deg), dtype=torch.float64)).item()
    cos_t = float(torch.cos(torch.tensor(theta)).item())
    sin_t = float(torch.sin(torch.tensor(theta)).item())
    return (
        -scale_deg * cos_t,
        scale_deg * sin_t,
        scale_deg * sin_t,
        scale_deg * cos_t,
    )


def _tpv_header(
    size: int,
    *,
    crval1: float,
    crval2: float,
    rotation_deg: float = 0.0,
    scale_deg: float = 0.01,
) -> fits.Header:
    h = fits.Header()
    h["NAXIS"] = 2
    h["NAXIS1"] = size
    h["NAXIS2"] = size
    h["CTYPE1"] = "RA---TPV"
    h["CTYPE2"] = "DEC--TPV"
    h["CRPIX1"] = size / 2.0
    h["CRPIX2"] = size / 2.0
    h["CRVAL1"] = crval1
    h["CRVAL2"] = crval2
    cd11, cd12, cd21, cd22 = _rotated_cd(scale_deg, rotation_deg)
    h["CD1_1"] = cd11
    h["CD1_2"] = cd12
    h["CD2_1"] = cd21
    h["CD2_2"] = cd22
    h["PV1_0"] = 0.0
    h["PV1_1"] = 1.0
    h["PV1_4"] = 2.0e-4
    h["PV1_5"] = -3.0e-4
    h["PV1_6"] = 1.5e-4
    h["PV1_7"] = 2.0e-6
    h["PV1_39"] = 2.0e-11
    h["PV2_0"] = 0.0
    h["PV2_1"] = 0.0
    h["PV2_2"] = 1.0
    h["PV2_4"] = -1.0e-4
    h["PV2_5"] = 2.5e-4
    h["PV2_6"] = -2.0e-4
    h["PV2_7"] = -1.5e-6
    h["PV2_39"] = -1.0e-11
    return h


def _astropy_sip_fixture() -> Path:
    return (
        Path(astropy.__file__).resolve().parent
        / "nddata"
        / "tests"
        / "data"
        / "sip-wcs.fits"
    )


def _clean_header(header: dict[str, Any]) -> fits.Header:
    out = fits.Header()
    for key, value in header.items():
        if str(key).endswith("_COMMENT"):
            continue
        out[key] = value
    return out


def _write_real_sip_mef(tmpdir: Path, *, shifted: bool) -> Path:
    image, header = torchsky.read(str(_astropy_sip_fixture()), return_header=True)
    data = image.to(dtype=torch.float32).cpu().numpy()
    sci1 = _clean_header(dict(header))
    sci1["EXTNAME"] = "SCI"
    sci1["EXTVER"] = 1
    sci2 = _clean_header(dict(header))
    sci2["EXTNAME"] = "SCI"
    sci2["EXTVER"] = 2
    if shifted:
        sci2["CRVAL1"] = float(sci2["CRVAL1"]) + 0.01
        sci2["CRVAL2"] = float(sci2["CRVAL2"]) - 0.01
    err1 = _clean_header(dict(header))
    err1["EXTNAME"] = "ERR"
    err1["EXTVER"] = 1
    err2 = _clean_header(dict(header))
    err2["EXTNAME"] = "ERR"
    err2["EXTVER"] = 2
    dq1 = _clean_header(dict(header))
    dq1["EXTNAME"] = "DQ"
    dq1["EXTVER"] = 1
    dq2 = _clean_header(dict(header))
    dq2["EXTNAME"] = "DQ"
    dq2["EXTVER"] = 2

    weight1 = torch.ones_like(image, dtype=torch.float32).cpu().numpy()
    weight2 = torch.ones_like(image, dtype=torch.float32).cpu().numpy()
    weight1[:, : image.shape[1] // 3] = 2.0
    weight2[:, (2 * image.shape[1]) // 3 :] = 3.0
    mask1 = torch.ones_like(image, dtype=torch.float32).cpu().numpy()
    mask2 = torch.ones_like(image, dtype=torch.float32).cpu().numpy()
    mask1[:, (2 * image.shape[1]) // 3 :] = 0.0
    mask2[:, : image.shape[1] // 3] = 0.0
    path = tmpdir / ("real_sip_shifted_mef.fits" if shifted else "real_sip_mef.fits")
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(data=data, header=sci1, name="SCI"),
            fits.ImageHDU(data=weight1, header=err1, name="ERR"),
            fits.ImageHDU(data=mask1, header=dq1, name="DQ"),
            fits.ImageHDU(data=data, header=sci2, name="SCI"),
            fits.ImageHDU(data=weight2, header=err2, name="ERR"),
            fits.ImageHDU(data=mask2, header=dq2, name="DQ"),
        ]
    ).writeto(path, overwrite=True)
    return path


def _write_synthetic_tpv_mef(tmpdir: Path, *, size: int) -> Path:
    base_header = _clean_header(_simple_header(size))
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, size, dtype=torch.float32),
        torch.linspace(-1.0, 1.0, size, dtype=torch.float32),
        indexing="ij",
    )
    image = (
        0.7 * torch.exp(-((xx + 0.2) ** 2 + (yy - 0.15) ** 2) / 0.08)
        + 0.9 * torch.exp(-((xx - 0.35) ** 2 + (yy + 0.25) ** 2) / 0.03)
        + 0.2 * torch.sin(5.0 * xx)
        + 0.1 * torch.cos(4.0 * yy)
    ).to(dtype=torch.float32)
    sci1 = base_header.copy()
    sci1["EXTNAME"] = "SCI"
    sci1["EXTVER"] = 1
    sci2 = _tpv_header(
        size,
        crval1=float(base_header["CRVAL1"]) + 0.42,
        crval2=float(base_header["CRVAL2"]) - 0.18,
        rotation_deg=14.0,
        scale_deg=0.01,
    )
    sci2["EXTNAME"] = "SCI"
    sci2["EXTVER"] = 2
    err1 = base_header.copy()
    err1["EXTNAME"] = "ERR"
    err1["EXTVER"] = 1
    err2 = sci2.copy()
    err2["EXTNAME"] = "ERR"
    err2["EXTVER"] = 2
    dq1 = base_header.copy()
    dq1["EXTNAME"] = "DQ"
    dq1["EXTVER"] = 1
    dq2 = sci2.copy()
    dq2["EXTNAME"] = "DQ"
    dq2["EXTVER"] = 2

    weight1 = (0.7 + 1.3 * (xx + 1.0) / 2.0).to(dtype=torch.float32)
    weight2 = (0.6 + 1.6 * (yy + 1.0) / 2.0).to(dtype=torch.float32)
    mask1 = torch.ones_like(image, dtype=torch.float32)
    mask2 = torch.ones_like(image, dtype=torch.float32)
    mask1[:, size // 3 : size // 3 + max(2, size // 16)] = 0.0
    mask2[size // 2 - max(2, size // 20) : size // 2 + max(2, size // 20), :] = 0.0
    path = tmpdir / f"synthetic_tpv_mef_{size}.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(data=image.cpu().numpy(), header=sci1, name="SCI"),
            fits.ImageHDU(data=weight1.cpu().numpy(), header=err1, name="ERR"),
            fits.ImageHDU(data=mask1.cpu().numpy(), header=dq1, name="DQ"),
            fits.ImageHDU(data=image.cpu().numpy(), header=sci2, name="SCI"),
            fits.ImageHDU(data=weight2.cpu().numpy(), header=err2, name="ERR"),
            fits.ImageHDU(data=mask2.cpu().numpy(), header=dq2, name="DQ"),
        ]
    ).writeto(path, overwrite=True)
    return path


def _time_many(fn, runs: int) -> float:
    fn()
    samples = []
    for _ in range(max(1, runs)):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return float(samples[len(samples) // 2])


def _prepare_synthetic_case(
    tmpdir: Path, *, size: int, n_obs: int, shifted: bool
) -> tuple[list[ImageObservation], tuple[int, int]]:
    header = _simple_header(size)
    base = torch.linspace(0.0, 1.0, size * size, dtype=torch.float32).reshape(size, size)
    observations: list[ImageObservation] = []
    for i in range(n_obs):
        image_path = tmpdir / f"synthetic_{i}.fits"
        weight_path = tmpdir / f"synthetic_{i}_wht.fits"
        obs_header = dict(header)
        if shifted:
            obs_header["CRVAL1"] = float(obs_header["CRVAL1"]) + 0.0025 * i
            obs_header["CRVAL2"] = float(obs_header["CRVAL2"]) - 0.0015 * i
        image = base + float(i)
        weight = torch.ones_like(image)
        weight[:, : size // 4] = 2.0 + float(i)
        torchsky.write(str(image_path), image, header=obs_header, overwrite=True)
        torchsky.write(str(weight_path), weight, header=obs_header, overwrite=True)
        observations.append(
            ImageObservation(
                str(image_path),
                weight_path=str(weight_path),
            )
        )
    return observations, (size, size)


def _prepare_real_case(tmpdir: Path) -> tuple[list[ImageObservation], tuple[int, int]]:
    image = torchsky.read(str(_astropy_sip_fixture()))
    image = image.to(dtype=torch.float32)
    mask_a = tmpdir / "mask_a.fits"
    mask_b = tmpdir / "mask_b.fits"

    left = torch.zeros_like(image)
    left[:, : image.shape[1] // 2] = 1.0
    right = torch.zeros_like(image)
    right[:, image.shape[1] // 2 :] = 1.0

    torchsky.write(str(mask_a), left, overwrite=True)
    torchsky.write(str(mask_b), right, overwrite=True)

    return (
        [
            ImageObservation(str(_astropy_sip_fixture()), mask_path=str(mask_a)),
            ImageObservation(str(_astropy_sip_fixture()), mask_path=str(mask_b)),
        ],
        tuple(int(v) for v in image.shape),
    )


def _prepare_real_shifted_sip_case(
    tmpdir: Path,
) -> tuple[list[ImageObservation], tuple[int, int]]:
    image, header = torchsky.read(str(_astropy_sip_fixture()), return_header=True)
    image = image.to(dtype=torch.float32)
    shifted_header = dict(header)
    shifted_header["CRVAL1"] = float(shifted_header["CRVAL1"]) + 0.01
    shifted_header["CRVAL2"] = float(shifted_header["CRVAL2"]) - 0.01
    shifted_path = tmpdir / "sip_shifted.fits"
    torchsky.write(str(shifted_path), image, header=shifted_header, overwrite=True)
    return (
        [
            ImageObservation(str(_astropy_sip_fixture())),
            ImageObservation(str(shifted_path)),
        ],
        tuple(int(v) for v in image.shape),
    )


def _prepare_real_sip_mef_case(
    tmpdir: Path,
    *,
    shifted: bool,
) -> tuple[list[ImageObservation], tuple[int, int]]:
    mef_path = _write_real_sip_mef(tmpdir, shifted=shifted)
    image = torchsky.read(str(_astropy_sip_fixture())).to(dtype=torch.float32)
    return (
        [
            ImageObservation(str(mef_path), hdu=1, weight_path=str(mef_path), weight_hdu=2, mask_path=str(mef_path), mask_hdu=3),
            ImageObservation(str(mef_path), hdu=4, weight_path=str(mef_path), weight_hdu=5, mask_path=str(mef_path), mask_hdu=6),
        ],
        tuple(int(v) for v in image.shape),
    )


def _prepare_synthetic_tpv_mef_case(
    tmpdir: Path,
    *,
    size: int,
) -> tuple[list[ImageObservation], tuple[int, int]]:
    mef_path = _write_synthetic_tpv_mef(tmpdir, size=size)
    return (
        [
            ImageObservation(str(mef_path), hdu=1, weight_path=str(mef_path), weight_hdu=2, mask_path=str(mef_path), mask_hdu=3),
            ImageObservation(str(mef_path), hdu=4, weight_path=str(mef_path), weight_hdu=5, mask_path=str(mef_path), mask_hdu=6),
        ],
        (size, size),
    )


def _bench_case(
    name: str,
    observations: list[ImageObservation],
    shape: tuple[int, int],
    runs: int,
    *,
    footprint_order: int | None,
) -> dict[str, Any]:
    dt = _time_many(
        lambda: coadd_image_plane(observations, footprint_order=footprint_order),
        runs,
    )
    result = coadd_image_plane(observations, footprint_order=footprint_order)
    pixels = float(shape[0] * shape[1])
    return {
        "case": name,
        "n_obs": len(observations),
        "shape": list(shape),
        "footprint_order": footprint_order,
        "time_ms": dt * 1000.0,
        "mpix_s": (pixels / dt) / 1e6,
        "mean_weight": float(result.weight.mean().item()),
        "coverage_frac": float((result.weight > 0).to(torch.float32).mean().item()),
        "max_overlap": int(result.overlap_count.max().item())
        if result.overlap_count is not None
        else 0,
        "coverage_area_sq_deg": (
            float(result.coverage_moc.area_sq_deg) if result.coverage_moc is not None else 0.0
        ),
    }


def _bench_multiband_case(
    name: str,
    observations: list[ImageObservation],
    runs: int,
    *,
    footprint_order: int | None,
) -> dict[str, Any]:
    planned = plan_image_plane_target(observations, reference_index=0)
    dt = _time_many(
        lambda: coadd_image_plane_multiband(
            observations,
            target_header=planned.header,
            footprint_order=footprint_order,
        ),
        runs,
    )
    result = coadd_image_plane_multiband(
        observations,
        target_header=planned.header,
        footprint_order=footprint_order,
    )
    pixels = float(result.data.shape[0] * result.data.shape[1] * result.data.shape[2])
    return {
        "case": name,
        "n_obs": len(observations),
        "shape": list(result.data.shape),
        "footprint_order": footprint_order,
        "time_ms": dt * 1000.0,
        "mpix_s": (pixels / dt) / 1e6,
        "mean_weight": float(result.weight.mean().item()),
        "coverage_frac": float((result.weight > 0).to(torch.float32).mean().item()),
        "max_overlap": int(result.overlap_count.max().item())
        if result.overlap_count is not None
        else 0,
        "coverage_area_sq_deg": float(
            sum(
                moc.area_sq_deg
                for moc in (result.coverage_mocs or ())
                if moc is not None
            )
        ),
        "n_bands": int(result.data.shape[0]),
        "spectral_unit": result.spectral_axis.unit,
        "spectral_type": result.spectral_axis.type,
    }


def _bench_cube_case(
    name: str,
    observations: list[ImageObservation],
    runs: int,
    *,
    footprint_order: int | None,
    spectral_header: dict[str, Any],
) -> dict[str, Any]:
    planned = plan_image_plane_cube_target_from_header(
        observations,
        spectral_header,
        reference_index=0,
    )
    dt = _time_many(
        lambda: coadd_image_plane_cube(
            observations,
            target=planned,
            footprint_order=footprint_order,
        ),
        runs,
    )
    result = coadd_image_plane_cube(
        observations,
        target=planned,
        footprint_order=footprint_order,
    )
    pixels = float(result.data.shape[0] * result.data.shape[1] * result.data.shape[2])
    return {
        "case": name,
        "n_obs": len(observations),
        "shape": list(result.data.shape),
        "footprint_order": footprint_order,
        "time_ms": dt * 1000.0,
        "mpix_s": (pixels / dt) / 1e6,
        "mean_weight": float(result.weight.mean().item()),
        "coverage_frac": float((result.weight > 0).to(torch.float32).mean().item()),
        "max_overlap": int(result.overlap_count.max().item())
        if result.overlap_count is not None
        else 0,
        "coverage_area_sq_deg": float(
            sum(
                moc.area_sq_deg
                for moc in (result.coverage_mocs or ())
                if moc is not None
            )
        ),
        "n_bands": int(result.data.shape[0]),
        "spectral_unit": result.spectral_axis.unit,
        "spectral_type": result.spectral_axis.type,
        "spectral_weight_sum": float(result.band_metadata.spectral_weight_sum.sum().item()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--n-obs", type=int, default=4)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--footprint-order", type=int, default=8)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="torchsky_map_bench_") as tmp:
        tmpdir = Path(tmp)
        synth_obs, synth_shape = _prepare_synthetic_case(
            tmpdir, size=args.size, n_obs=args.n_obs, shifted=False
        )
        rows.append(
            _bench_case(
                "synthetic_identical_wcs",
                synth_obs,
                synth_shape,
                args.runs,
                footprint_order=args.footprint_order,
            )
        )

        shifted_obs, shifted_shape = _prepare_synthetic_case(
            tmpdir, size=args.size, n_obs=args.n_obs, shifted=True
        )
        rows.append(
            _bench_case(
                "synthetic_shifted_wcs",
                shifted_obs,
                shifted_shape,
                args.runs,
                footprint_order=args.footprint_order,
            )
        )

        real_obs, real_shape = _prepare_real_case(tmpdir)
        rows.append(
            _bench_case(
                "real_astropy_sip_fixture",
                real_obs,
                real_shape,
                args.runs,
                footprint_order=args.footprint_order,
            )
        )
        shifted_sip_obs, shifted_sip_shape = _prepare_real_shifted_sip_case(tmpdir)
        rows.append(
            _bench_case(
                "real_astropy_sip_shifted_wcs",
                shifted_sip_obs,
                shifted_sip_shape,
                args.runs,
                footprint_order=args.footprint_order,
            )
        )

        mef_obs, mef_shape = _prepare_real_sip_mef_case(tmpdir, shifted=False)
        rows.append(
            _bench_case(
                "real_astropy_sip_mef",
                mef_obs,
                mef_shape,
                args.runs,
                footprint_order=args.footprint_order,
            )
        )
        shifted_mef_obs, shifted_mef_shape = _prepare_real_sip_mef_case(tmpdir, shifted=True)
        rows.append(
            _bench_case(
                "real_astropy_sip_mef_shifted_wcs",
                shifted_mef_obs,
                shifted_mef_shape,
                args.runs,
                footprint_order=args.footprint_order,
            )
        )

        tpv_mef_obs, tpv_mef_shape = _prepare_synthetic_tpv_mef_case(tmpdir, size=args.size)
        rows.append(
            _bench_case(
                "synthetic_tpv_mef_low_overlap",
                tpv_mef_obs,
                tpv_mef_shape,
                args.runs,
                footprint_order=args.footprint_order,
            )
        )

        multiband_obs = [
            ImageObservation(
                obs.path,
                hdu=obs.hdu,
                weight=obs.weight,
                wavelength=4800.0,
                spectral_min=4700.0,
                spectral_max=4900.0,
                band_name="g",
                weight_path=obs.weight_path,
                weight_hdu=obs.weight_hdu,
                mask_path=obs.mask_path,
                mask_hdu=obs.mask_hdu,
            )
            for obs in synth_obs
        ] + [
            ImageObservation(
                obs.path,
                hdu=obs.hdu,
                weight=obs.weight,
                wavelength=6500.0,
                spectral_min=6400.0,
                spectral_max=6600.0,
                band_name="r",
                weight_path=obs.weight_path,
                weight_hdu=obs.weight_hdu,
                mask_path=obs.mask_path,
                mask_hdu=obs.mask_hdu,
            )
            for obs in shifted_obs
        ]
        rows.append(
            _bench_multiband_case(
                "synthetic_multiband_cube",
                multiband_obs,
                args.runs,
                footprint_order=args.footprint_order,
            )
        )
        rows.append(
            _bench_cube_case(
                "synthetic_planned_spectral_cube",
                multiband_obs,
                args.runs,
                footprint_order=args.footprint_order,
                spectral_header={
                    "NAXIS": 3,
                    "NAXIS1": args.size,
                    "NAXIS2": args.size,
                    "NAXIS3": 2,
                    "CTYPE3": "WAVE",
                    "CUNIT3": "Angstrom",
                    "CRVAL3": 4800.0,
                    "CDELT3": 1700.0,
                    "CRPIX3": 1.0,
                },
            )
        )

    for row in rows:
        print(
            f"{row['case']:>28s}"
            f"  n_obs={row['n_obs']:>2d}"
            f"  shape={tuple(row['shape'])}"
            f"  time_ms={row['time_ms']:.3f}"
            f"  mpix_s={row['mpix_s']:.3f}"
            f"  coverage={row['coverage_frac']:.3f}"
            f"  mean_weight={row['mean_weight']:.3f}"
            f"  max_overlap={row['max_overlap']}"
            f"  area_sq_deg={row['coverage_area_sq_deg']:.3f}"
            + (
                f"  spectral_weight_sum={row['spectral_weight_sum']:.3f}"
                if "spectral_weight_sum" in row
                else ""
            )
        )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
