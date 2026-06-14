"""Tests for the SMACS J0723 student-data preparation helpers."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from torchsky.spectral import SpectralReader


def _load_script_module():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "build_smacs0723_minicutout.py"
    )
    spec = importlib.util.spec_from_file_location("build_smacs0723_minicutout", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_baseline_module():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "build_smacs0723_stack_baseline.py"
    )
    spec = importlib.util.spec_from_file_location(
        "build_smacs0723_stack_baseline", script
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_visualize_module():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "visualize_smacs0723_products.py"
    )
    spec = importlib.util.spec_from_file_location(
        "visualize_smacs0723_products", script
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_manifest_path_is_relative_inside_repo() -> None:
    module = _load_script_module()
    root = Path("/tmp/torchsky-root").resolve()
    path = root / "data" / "smacs0723_minicutout" / "cutout.fits"
    assert (
        module._manifest_path(path, root=root)
        == "data/smacs0723_minicutout/cutout.fits"
    )


def test_manifest_path_allows_external_data_root() -> None:
    module = _load_script_module()
    root = Path("/tmp/torchsky-root").resolve()
    external = Path("/tmp/torchsky-data/smacs0723_minicutout/cutout.fits").resolve()
    assert module._manifest_path(external, root=root) == str(external)


def test_stack_baseline_resolves_manifest_paths(tmp_path: Path) -> None:
    module = _load_baseline_module()
    manifest_dir = tmp_path / "manifest-dir"
    manifest_dir.mkdir()
    local = manifest_dir / "cutout.fits"
    local.write_bytes(b"")
    assert (
        module._resolve_manifest_path("cutout.fits", manifest_dir=manifest_dir)
        == local.resolve()
    )
    external = tmp_path / "external" / "cutout.fits"
    assert (
        module._resolve_manifest_path(str(external), manifest_dir=manifest_dir)
        == external
    )


def test_cutout_builder_validates_crop_size() -> None:
    module = _load_script_module()
    with pytest.raises(ValueError, match="positive"):
        module._validate_crop_size(0)


def test_cutout_builder_records_crop_metadata() -> None:
    module = _load_script_module()
    header = fits.Header()
    header["CRPIX1"] = 5.0
    header["CRPIX2"] = 6.0
    data = np.zeros((10, 12), dtype=np.float32)
    patch, hdr, meta = module._cutout_2d(
        data,
        header,
        crop_size=4,
        center_x_pix=8.0,
        center_y_pix=3.0,
    )
    assert patch.shape == (4, 4)
    assert hdr["CRPIX1"] == pytest.approx(-1.0)
    assert hdr["CRPIX2"] == pytest.approx(5.0)
    assert meta["source_shape"] == [10, 12]
    assert meta["crop_origin_xy"] == [6, 1]
    assert meta["crop_bounds_xy"] == [6, 1, 10, 5]


def test_stack_baseline_observations_from_manifest(tmp_path: Path) -> None:
    module = _load_baseline_module()
    cutout = tmp_path / "smacs0723_F090W_cutout_16.fits"
    cutout.write_bytes(b"placeholder")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"outputs": [{"band": "F090W", "path": str(cutout)}]}),
        encoding="utf-8",
    )
    observations = module._observations_from_manifest(manifest)
    assert len(observations) == 1
    assert observations[0].band_name == "F090W"
    assert observations[0].wavelength == 9000.0


def test_stack_baseline_missing_manifest_error(tmp_path: Path) -> None:
    module = _load_baseline_module()
    with pytest.raises(FileNotFoundError, match="build_smacs0723_minicutout"):
        module._observations_from_manifest(tmp_path / "manifest.json")


def test_minicutout_and_stack_baseline_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_dir = tmp_path / "updec_webb"
    source_dir.mkdir()
    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = 32
    header["NAXIS2"] = 32
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CRPIX1"] = 16.5
    header["CRPIX2"] = 16.5
    header["CRVAL1"] = 150.0
    header["CRVAL2"] = 2.0
    header["CDELT1"] = -0.05 / 3600.0
    header["CDELT2"] = 0.05 / 3600.0
    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"
    for band, offset in (("F090W", 1.0), ("F150W", 2.0)):
        data = np.arange(32 * 32, dtype=np.float32).reshape(32, 32) + offset
        filename = _load_script_module().REFERENCE_NAMES[band]
        fits.HDUList(
            [
                fits.PrimaryHDU(),
                fits.ImageHDU(data=data, header=header, name="SCI"),
            ]
        ).writeto(source_dir / filename)

    cutout_module = _load_script_module()
    monkeypatch.setenv("TORCHSKY_DATA", str(tmp_path))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_smacs0723_minicutout.py",
            "--bands",
            "F090W,F150W",
            "--crop-size",
            "16",
        ],
    )
    assert cutout_module.main() == 0

    manifest = tmp_path / "smacs0723_minicutout" / "manifest.json"
    assert manifest.is_file()
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    assert len(manifest_data["outputs"]) == 2
    first = manifest_data["outputs"][0]
    assert first["source_shape"] == [32, 32]
    assert first["crop_origin_xy"] == [8, 8]
    assert first["crop_bounds_xy"] == [8, 8, 24, 24]

    baseline_module = _load_baseline_module()
    output = tmp_path / "baseline.fits"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_smacs0723_stack_baseline.py",
            "--manifest",
            str(manifest),
            "--output",
            str(output),
        ],
    )
    assert baseline_module.main() == 0
    cube = SpectralReader.read_data_cube(str(output), hdu=0)
    assert cube.data.shape == (2, 16, 16)
    assert cube.spectral_axis.values.tolist() == [9000.0, 15000.0]

    drizzle_output = tmp_path / "baseline_drizzle.fits"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_smacs0723_stack_baseline.py",
            "--manifest",
            str(manifest),
            "--output",
            str(drizzle_output),
            "--resampling-kernel",
            "drizzle_square",
        ],
    )
    assert baseline_module.main() == 0
    drizzle_cube = SpectralReader.read_data_cube(str(drizzle_output), hdu=0)
    assert drizzle_cube.data.shape == (2, 16, 16)

    visualize_module = _load_visualize_module()
    quicklook_dir = tmp_path / "quicklooks"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "visualize_smacs0723_products.py",
            "--manifest",
            str(manifest),
            "--baseline-cube",
            str(output),
            "--output-dir",
            str(quicklook_dir),
        ],
    )
    assert visualize_module.main() == 0
    assert (quicklook_dir / "cutouts.png").is_file()
    assert (quicklook_dir / "baseline_cube.png").is_file()
    assert (quicklook_dir / "baseline_false_color.png").is_file()
    summary = json.loads(
        (quicklook_dir / "quicklook_summary.json").read_text(encoding="utf-8")
    )
    assert summary["baseline_shape"] == [2, 16, 16]
