"""Tests for torchsky.map.exposure_pose.fit_pose_correction."""

from __future__ import annotations

import math

import pytest
import torch

from torchsky.map.exposure_pose import fit_pose_correction


def _random_pairs(n: int = 200, seed: int = 7) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    ra = torch.rand(n, generator=generator, dtype=torch.float64) * 4.0 - 2.0 + 10.0
    dec = torch.rand(n, generator=generator, dtype=torch.float64) * 4.0 - 2.0 - 30.0
    return ra, dec


def _apply_shift_rotation(
    ra: torch.Tensor,
    dec: torch.Tensor,
    *,
    dxi_deg: float,
    deta_deg: float,
    theta_rad: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inject a rigid tangent-plane transform about the anchor (row 0)."""
    ra0 = float(ra[0])
    dec0 = float(dec[0])
    xi = ((ra - ra0 + 180.0) % 360.0 - 180.0) * torch.cos(torch.deg2rad(dec))
    eta = dec - dec0
    c = math.cos(theta_rad)
    s = math.sin(theta_rad)
    xi_out = c * xi - s * eta + dxi_deg
    eta_out = s * xi + c * eta + deta_deg
    dec_out = dec0 + eta_out
    ra_out = ra0 + xi_out / torch.cos(torch.deg2rad(dec_out))
    return ra_out, dec_out


@pytest.mark.parametrize(
    ("model", "dxi", "deta", "theta_deg"),
    [
        ("shift", 1e-3, 0.0, 0.0),
        ("rshift", 1e-3, 5e-4, 0.5),
    ],
)
def test_fit_recovers_known_rigid_correction(
    model: str, dxi: float, deta: float, theta_deg: float
) -> None:
    src_ra, src_dec = _random_pairs()
    dst_ra, dst_dec = _apply_shift_rotation(
        src_ra, src_dec, dxi_deg=dxi, deta_deg=deta, theta_rad=math.radians(theta_deg)
    )
    pose = fit_pose_correction(
        src_ra, src_dec, dst_ra, dst_dec, model=model, anchor_index=0
    )
    out_ra, out_dec = pose.apply_to_world(
        src_ra, src_dec, center_ra_deg=float(src_ra[0]), center_dec_deg=float(src_dec[0])
    )
    assert float((out_ra - dst_ra).abs().max()) < 1e-6
    assert float((out_dec - dst_dec).abs().max()) < 1e-6


def test_anchor_self_pairs_yield_identity_gauge() -> None:
    src_ra, src_dec = _random_pairs()
    pose = fit_pose_correction(
        src_ra, src_dec, src_ra.clone(), src_dec.clone(), model="rshift"
    )
    assert float(pose.delta_theta_rad) == 0.0
    assert abs(float(pose.delta_ra_deg)) < 1e-12
    assert abs(float(pose.delta_dec_deg)) < 1e-12
    out_ra, out_dec = pose.apply_to_world(
        src_ra, src_dec, center_ra_deg=float(src_ra[0]), center_dec_deg=float(src_dec[0])
    )
    assert float((out_ra - src_ra).abs().max()) < 1e-12
    assert float((out_dec - src_dec).abs().max()) < 1e-12


def test_invalid_model_raises() -> None:
    src_ra, src_dec = _random_pairs(n=4)
    with pytest.raises(ValueError, match="unsupported pose-correction model"):
        fit_pose_correction(src_ra, src_dec, src_ra, src_dec, model="rscale")


def test_anchor_index_bounds_and_empty_input() -> None:
    src_ra, src_dec = _random_pairs(n=4)
    with pytest.raises(ValueError, match="anchor_index"):
        fit_pose_correction(src_ra, src_dec, src_ra, src_dec, anchor_index=99)
    empty = torch.empty(0, dtype=torch.float64)
    with pytest.raises(ValueError, match="at least one matched pair"):
        fit_pose_correction(empty, empty, empty, empty)
