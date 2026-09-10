"""Check calibrated FITS quicklooks and shared display scales with synthetic inputs."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import torchfits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import visualize_smacs0723_products as viz


def test_fits_report_uses_reference_scales_and_preserves_completed_output(tmp_path, monkeypatch):
    bands = ["F090W", "F150W", "F200W", "F277W", "F356W", "F444W"]
    cube = torch.linspace(0, 1, 6 * 12 * 13).reshape(6, 12, 13)
    header = {"BUNIT": "MJy/sr", "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
              "CRPIX1": 1., "CRPIX2": 1., "CRVAL1": 1., "CRVAL2": 0.,
              "CDELT1": -0.00001, "CDELT2": 0.00001}
    baseline = tmp_path / "smacs0723_stack_baseline.fits"
    comparison = tmp_path / "synthetic.fits"
    torchfits.write(baseline, cube, header)
    torchfits.write(comparison, cube * 2, header)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"bands": bands, "products": [{"method": "synthetic", "path": str(comparison), "bands": bands}]}))
    monkeypatch.setattr(sys, "argv", ["visualize", "--manifest", str(manifest), "--dpi", "35", "--no-synthetic-check"])
    assert viz.main() == 0
    output = tmp_path / "quicklooks"
    scales = json.loads((output / "display_limits.json").read_text())["limits"]
    np.testing.assert_allclose(scales[bands[0]], np.percentile(cube[0].numpy(), [1, 99]))
    summary = json.loads((output / "quicklook_summary.json").read_text())
    assert summary["six_band_rgb"]["synthetic"]["status"] == "rendered"
    assert Path(summary["method_reports"]["synthetic"][0]).is_file()
    assert "not measured" in viz._legend_text(viz.ProductPlane("x", "F090W", cube[0].numpy(), header, "synthetic"), {})
    original = (output / "run_metadata.json").read_bytes()
    with pytest.raises(FileExistsError):
        viz.main()
    assert (output / "run_metadata.json").read_bytes() == original


def test_rgb_uses_torchfits_and_intersection_coverage(monkeypatch):
    bands = ["F090W", "F150W", "F200W", "F277W", "F356W", "F444W"]
    header = {"CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN", "BUNIT": "MJy/sr",
              "CRVAL1": 1., "CRVAL2": 0., "CRPIX1": 1., "CRPIX2": 1.,
              "CDELT1": -0.001, "CDELT2": 0.001}
    products = {band: viz.ProductPlane("x", band, np.full((3, 4), i / 10.), header, "synthetic") for i, band in enumerate(bands)}
    products["F090W"].data[0, 0] = np.nan
    called = []
    def capture(red, green, blue, **kwargs):
        called.append((red, green, blue))
        return torch.stack((red, green, blue), -1)
    monkeypatch.setattr(torchfits.transforms, "lupton_rgb", capture)
    rgb, _ = viz._six_band_rgb(products, {band: (0., 1.) for band in bands})
    np.testing.assert_allclose(rgb[1, 1], [.45, .25, .05])
    np.testing.assert_allclose(rgb[0, 0], [.3, .3, .3])
    assert len(called) == 1
    with pytest.raises(ValueError, match="duplicate"):
        viz._group_products([products["F090W"]] * 2)


def test_non_square_panel_count_does_not_drop_images(tmp_path):
    viz._save_image_grid([np.zeros((3, 3))] * 5, ["synthetic"] * 5,
                         tmp_path / "five.png", max_panels=5, dpi=30)
    assert (tmp_path / "five.png").is_file()


def _write_joint_fixture(tmp_path):
    bands = ["F090W", "F150W", "F200W", "F277W", "F356W", "F444W"]
    export_dir = tmp_path / "joint_exports"
    header = {
        "BUNIT": "MJy/sr",
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 1.0,
        "CRPIX2": 1.0,
        "CRVAL1": 1.0,
        "CRVAL2": 0.0,
        "CDELT1": -0.001,
        "CDELT2": 0.001,
    }
    support_header = {key: value for key, value in header.items() if key != "BUNIT"}
    scores = []
    rows = []
    for index, band in enumerate(bands):
        directory = export_dir / "fitted_gaussian" / band.lower() / "spectral_power_foam"
        directory.mkdir(parents=True)
        reference = torch.full((5, 6), 1.0 + index)
        prediction = reference + 0.05
        support = torch.ones((5, 6))
        support[0, 0] = 0.0
        torchfits.write(directory / "reference.fits", reference, header)
        torchfits.write(directory / "prediction.fits", prediction, header)
        torchfits.write(directory / "support_mask.fits", support, support_header)
        rows.append(
            {
                "band": band,
                "model": "spectral_power_foam",
                "status": "OK",
                "fit_time_ms": 1250.0 + index,
                "peak_accelerator_memory_bytes": 2 * 1024**2,
                "render_mode": "hard_pixel_footprint",
                "product_interpretation": "target-grid scene with assumed target PSF",
            }
        )
        scores.append(
            {
                "band": band,
                "exposures": 2,
                "pixels": 29,
                "gaussian_nll_per_pixel": 1.25 + index,
                "chi2_per_pixel": 0.75 + index,
            }
        )
    report = tmp_path / "joint.json"
    report.write_text(
        json.dumps(
            {
                "experiment": {
                    "bands": bands,
                    "psf_model": "fitted_gaussian",
                    "validation_detector_scoring": {
                        "split": "validation",
                        "scores_by_model": {"spectral_power_foam": scores},
                    },
                },
                "rows": rows,
            }
        ),
        encoding="utf-8",
    )
    return report, export_dir, bands, header


def test_joint_report_loads_real_export_layout_and_renders_six_band_rgb(tmp_path, monkeypatch):
    report, export_dir, bands, _header = _write_joint_fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    entries, metrics = viz._joint_products(report, Path("joint_exports"))

    assert len(entries) == 12
    assert all("fitted_gaussian/f090w/spectral_power_foam" in entry["path"] for entry in entries if entry["band"] == "F090W")
    assert metrics["spectral_power_foam:F090W"]["validation_nll_per_pixel"] == pytest.approx(1.25)
    assert metrics["spectral_power_foam:F090W"]["validation_chi2_per_pixel"] == pytest.approx(0.75)
    products = viz._load_product_specs(entries, manifest_dir=report.parent, default_bands=bands)
    grouped = viz._group_products(products)
    method_products = grouped["spectral_power_foam"]
    assert np.isnan(method_products["F090W"].data[0, 0])
    assert not np.isnan(method_products["F090W"].data[1, 1])
    rgb, status = viz._six_band_rgb(method_products, {band: (0.0, 8.0) for band in bands})
    assert status == "shared WCS, shape, units; intersection of band coverage"
    assert rgb is not None and rgb.shape == (5, 6, 3)
    np.testing.assert_allclose(rgb[0, 0], [0.3, 0.3, 0.3])
    legend = viz._legend_text(method_products["F090W"], metrics["spectral_power_foam:F090W"])
    assert "validation NLL/pixel=1.25" in legend
    assert "chi2/pixel=0.75" in legend
    assert "runtime=joint fit 1.25 s" in legend
    assert "memory=peak accelerator 2 MiB" in legend
    assert "render=hard_pixel_footprint" in legend
    assert "interpretation=target-grid scene with assumed target PSF" in legend

    output_dir = tmp_path / "quicklooks"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "visualize",
            "--joint-report",
            str(report),
            "--joint-export-dir",
            "joint_exports",
            "--output-dir",
            str(output_dir),
            "--dpi",
            "20",
            "--no-synthetic-check",
        ],
    )
    assert viz.main() == 0
    summary = json.loads((output_dir / "quicklook_summary.json").read_text())
    assert summary["six_band_rgb"]["spectral_power_foam"]["status"] == "rendered"


def test_joint_report_rejects_malformed_support_masks(tmp_path):
    report, export_dir, bands, header = _write_joint_fixture(tmp_path)
    entries, _metrics = viz._joint_products(report, export_dir)
    support_path = Path(entries[0]["support_path"])

    torchfits.write(support_path, torch.ones((4, 6)), header, overwrite=True)
    with pytest.raises(ValueError, match="different image shapes/resolution"):
        viz._load_product_specs(entries[:1], manifest_dir=report.parent, default_bands=bands)

    torchfits.write(support_path, torch.full((5, 6), float("nan")), header, overwrite=True)
    with pytest.raises(ValueError, match="contains nonfinite values"):
        viz._load_product_specs(entries[:1], manifest_dir=report.parent, default_bands=bands)

    torchfits.write(support_path, torch.zeros((5, 6)), header, overwrite=True)
    with pytest.raises(ValueError, match="has no supported pixels"):
        viz._load_product_specs(entries[:1], manifest_dir=report.parent, default_bands=bands)
