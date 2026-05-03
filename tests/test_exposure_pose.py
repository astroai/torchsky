"""Tests for torchsky.map.exposure_pose.ExposurePose."""

from __future__ import annotations

import pytest
import torch

from torchsky.map import (
    ExposurePose,
    FieldDomain,
    FieldObservation,
    ForwardContext,
    freeze_pose,
    make_field_model,
    render_field_observation,
)
from torchsky.wcs import WCS


def _tan_header(*, h: int = 16, w: int = 16) -> dict[str, float | int | str]:
    return {
        "NAXIS": 2,
        "NAXIS1": w,
        "NAXIS2": h,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": (w + 1) / 2,
        "CRPIX2": (h + 1) / 2,
        "CRVAL1": 150.0,
        "CRVAL2": 2.0,
        "CDELT1": -0.2 / 3600.0,
        "CDELT2": 0.2 / 3600.0,
        "CUNIT1": "deg",
        "CUNIT2": "deg",
    }


def test_exposure_pose_identity_world_to_pixel_roundtrip() -> None:
    header = _tan_header()
    h, w = int(header["NAXIS2"]), int(header["NAXIS1"])
    device = torch.device("cpu")
    y, x = torch.meshgrid(
        torch.arange(h, dtype=torch.float64, device=device),
        torch.arange(w, dtype=torch.float64, device=device),
        indexing="ij",
    )
    wcs = WCS(header)
    ra, dec = wcs.pixel_to_world(x, y)
    pose = ExposurePose()
    center_ra = float(header["CRVAL1"])
    center_dec = float(header["CRVAL2"])
    ra2, dec2 = pose.apply_to_world(ra, dec, center_ra_deg=center_ra, center_dec_deg=center_dec)
    with torch.no_grad():
        xp, yp = wcs.world_to_pixel(ra2.detach(), dec2.detach())
    torch.testing.assert_close(xp, x, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(yp, y, rtol=1e-5, atol=1e-5)


def test_exposure_pose_compose_with_wcs_shift() -> None:
    header = _tan_header()
    wcs = WCS(header)
    ra = torch.tensor([150.01, 150.02], dtype=torch.float64)
    dec = torch.tensor([2.0, 2.01], dtype=torch.float64)
    pose = ExposurePose()
    torch.nn.init.constant_(pose.delta_ra_deg, 0.1 / 3600.0)
    x0, y0 = pose.compose_with_wcs(
        wcs,
        ra,
        dec,
        center_ra_deg=float(header["CRVAL1"]),
        center_dec_deg=float(header["CRVAL2"]),
    )
    x1, y1 = wcs.world_to_pixel(ra, dec)
    assert not torch.allclose(x0, x1)


def test_exposure_pose_compose_with_wcs_preserves_gradients() -> None:
    header = _tan_header()
    wcs = WCS(header)
    ra = torch.tensor([150.01, 150.02], dtype=torch.float64)
    dec = torch.tensor([2.0, 2.01], dtype=torch.float64)
    pose = ExposurePose()
    x_pix, y_pix = pose.compose_with_wcs(
        wcs,
        ra,
        dec,
        center_ra_deg=float(header["CRVAL1"]),
        center_dec_deg=float(header["CRVAL2"]),
    )
    loss = (x_pix.square() + y_pix.square()).mean()
    loss.backward()
    assert pose.delta_ra_deg.grad is not None
    assert pose.delta_dec_deg.grad is not None


def test_freeze_pose_disables_gradients() -> None:
    pose = ExposurePose()
    freeze_pose(pose)
    assert all(not p.requires_grad for p in pose.parameters())


def test_render_field_observation_with_pose_and_forward_context() -> None:
    header = _tan_header(h=8, w=8)
    obs = FieldObservation(
        image=torch.zeros(8, 8, dtype=torch.float32),
        header=header,
        spectral_min=5000.0,
        spectral_max=5000.0,
        transmission_center=5000.0,
    )
    domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.02,
        wavelength_center=5000.0,
        wavelength_scale=500.0,
    )
    model = make_field_model("sine_implicit", domain, hidden_dim=32, hidden_layers=2)
    pose = ExposurePose()
    ctx = ForwardContext(survey="JWST", instrument="NIRCam")
    pred = render_field_observation(
        model,
        obs,
        num_spectral_samples=2,
        forward_context=ctx,
        exposure_pose=pose,
    )
    assert pred.shape == (8, 8)


def test_exposure_pose_pole_raises() -> None:
    pose = ExposurePose()
    ra = torch.tensor([0.0])
    dec = torch.tensor([89.99])
    with pytest.raises(ValueError, match="pole"):
        pose.apply_to_world(ra, dec, center_ra_deg=0.0, center_dec_deg=90.0)
