"""Optional SWarp CLI parity vs torchsky on a tiny weighted stack (same grid)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("torchsky.cpp")

import torchsky  # noqa: E402
from torchsky.map import ImageObservation, coadd_image_plane  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

import bench_map_compat as bench_map_compat  # noqa: E402


@pytest.mark.skipif(
    bench_map_compat._find_swarp_binary() is None,
    reason="SWarp binary not installed or not on PATH",
)
def test_swarp_bilinear_stack_near_torchsky_on_tiny_shifted_pair(tmp_path: Path) -> None:
    ny, nx = 24, 24
    base = {
        "NAXIS": 2,
        "NAXIS1": nx,
        "NAXIS2": ny,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": nx / 2.0,
        "CRPIX2": ny / 2.0,
        "CRVAL1": 200.0,
        "CRVAL2": 15.0,
        "CDELT1": -0.02,
        "CDELT2": 0.02,
    }
    ha = dict(base)
    hb = dict(base)
    hb["CRVAL1"] = float(hb["CRVAL1"]) + 0.5 * float(hb["CDELT1"])
    yy, xx = torch.meshgrid(
        torch.arange(ny, dtype=torch.float32),
        torch.arange(nx, dtype=torch.float32),
        indexing="ij",
    )
    blob = torch.exp(
        -((xx - nx / 2.0) ** 2 + (yy - ny / 2.0) ** 2) / (2.0 * 4.0**2)
    )
    pa = tmp_path / "a.fits"
    pb = tmp_path / "b.fits"
    torchsky.write(str(pa), blob, header=ha, overwrite=True)
    torchsky.write(str(pb), blob * 0.95, header=hb, overwrite=True)
    obs = [ImageObservation(str(pa)), ImageObservation(str(pb))]
    target_header = dict(base)
    ts = coadd_image_plane(
        obs,
        target_header=target_header,
        resampling_kernel="bilinear",
    )
    sw_data, sw_w = bench_map_compat._coadd_with_swarp(
        obs,
        target_header=target_header,
        resampling_type="BILINEAR",
    )
    mask = (ts.weight > 0.0) & torch.from_numpy(sw_w > 0.0).to(ts.weight.device)
    inner_y = slice(4, ny - 4)
    inner_x = slice(4, nx - 4)
    mask = mask[inner_y, inner_x]
    td = ts.data[inner_y, inner_x][mask].cpu()
    sd = torch.from_numpy(sw_data[inner_y, inner_x]).to(td.device)[mask]
    assert float((td - sd).abs().max().item()) < 0.05
