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
    center_ra_deg: float,
    center_dec_deg: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent analytic gnomonic injection about the renderer center."""
    ra0 = math.radians(center_ra_deg)
    dec0 = math.radians(center_dec_deg)
    ra_rad = torch.deg2rad(ra)
    dec_rad = torch.deg2rad(dec)
    dra = torch.remainder(ra_rad - ra0 + math.pi, 2.0 * math.pi) - math.pi
    cosc = math.sin(dec0) * torch.sin(dec_rad) + math.cos(dec0) * torch.cos(
        dec_rad
    ) * torch.cos(dra)
    xi = torch.rad2deg(torch.cos(dec_rad) * torch.sin(dra) / cosc)
    eta = torch.rad2deg(
        (
            math.cos(dec0) * torch.sin(dec_rad)
            - math.sin(dec0) * torch.cos(dec_rad) * torch.cos(dra)
        )
        / cosc
    )
    c = math.cos(theta_rad)
    s = math.sin(theta_rad)
    xi_out = c * xi - s * eta + dxi_deg
    eta_out = s * xi + c * eta + deta_deg
    x = torch.deg2rad(xi_out)
    y = torch.deg2rad(eta_out)
    rho = torch.sqrt(x.square() + y.square())
    angle = torch.atan(rho)
    rho_safe = torch.clamp(rho, min=1e-30)
    dec_out = torch.asin(
        torch.clamp(
            torch.cos(angle) * math.sin(dec0)
            + y * torch.sin(angle) * math.cos(dec0) / rho_safe,
            -1.0,
            1.0,
        )
    )
    ra_out = ra0 + torch.atan2(
        x * torch.sin(angle),
        rho * math.cos(dec0) * torch.cos(angle)
        - y * math.sin(dec0) * torch.sin(angle),
    )
    return torch.remainder(torch.rad2deg(ra_out), 360.0), torch.rad2deg(dec_out)


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
    center_ra, center_dec = 10.25, -30.2
    dst_ra, dst_dec = _apply_shift_rotation(
        src_ra,
        src_dec,
        dxi_deg=dxi,
        deta_deg=deta,
        theta_rad=math.radians(theta_deg),
        center_ra_deg=center_ra,
        center_dec_deg=center_dec,
    )
    pose = fit_pose_correction(
        src_ra,
        src_dec,
        dst_ra,
        dst_dec,
        center_ra_deg=center_ra,
        center_dec_deg=center_dec,
        model=model,
    )
    out_ra, out_dec = pose.apply_to_world(
        src_ra, src_dec, center_ra_deg=center_ra, center_dec_deg=center_dec
    )
    assert float((out_ra - dst_ra).abs().max().detach()) < 1e-6
    assert float((out_dec - dst_dec).abs().max().detach()) < 1e-6


def test_anchor_self_pairs_yield_identity_gauge() -> None:
    src_ra, src_dec = _random_pairs()
    pose = fit_pose_correction(
        src_ra,
        src_dec,
        src_ra.clone(),
        src_dec.clone(),
        center_ra_deg=10.25,
        center_dec_deg=-30.2,
        model="rshift",
    )
    assert float(pose.delta_theta_rad.detach()) == 0.0
    assert abs(float(pose.delta_ra_deg.detach())) < 1e-12
    assert abs(float(pose.delta_dec_deg.detach())) < 1e-12
    out_ra, out_dec = pose.apply_to_world(
        src_ra, src_dec, center_ra_deg=10.25, center_dec_deg=-30.2
    )
    assert float((out_ra - src_ra).abs().max().detach()) < 1e-12
    assert float((out_dec - src_dec).abs().max().detach()) < 1e-12


def test_invalid_model_raises() -> None:
    src_ra, src_dec = _random_pairs(n=4)
    with pytest.raises(ValueError, match="unsupported pose-correction model"):
        fit_pose_correction(
            src_ra,
            src_dec,
            src_ra,
            src_dec,
            center_ra_deg=10.0,
            center_dec_deg=-30.0,
            model="rscale",
        )


def test_empty_nonfinite_and_degenerate_inputs_fail_closed() -> None:
    empty = torch.empty(0, dtype=torch.float64)
    with pytest.raises(ValueError, match="at least one matched pair"):
        fit_pose_correction(
            empty,
            empty,
            empty,
            empty,
            center_ra_deg=10.0,
            center_dec_deg=-30.0,
        )
    ra, dec = _random_pairs(n=4)
    ra[1] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        fit_pose_correction(
            ra,
            dec,
            ra,
            dec,
            center_ra_deg=10.0,
            center_dec_deg=-30.0,
        )
    repeated = torch.full((2,), 10.0, dtype=torch.float64)
    repeated_dec = torch.full((2,), -30.0, dtype=torch.float64)
    with pytest.raises(ValueError, match="degenerate"):
        fit_pose_correction(
            repeated,
            repeated_dec,
            repeated,
            repeated_dec,
            center_ra_deg=10.0,
            center_dec_deg=-30.0,
        )
