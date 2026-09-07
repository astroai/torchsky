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
