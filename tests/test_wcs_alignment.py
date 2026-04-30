import math

import torch

from torchsky.wcs import TangentPlaneFrame, fit_affine_2d, fit_sky_affine


def _points() -> torch.Tensor:
    return torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [2.0, 1.0], [-1.0, 1.0]],
        dtype=torch.float64,
    )


def test_fit_affine_shift() -> None:
    source = _points()
    offset = torch.tensor([2.5, -1.25], dtype=torch.float64)
    target = source + offset

    fit = fit_affine_2d(source, target, geometry="shift")

    torch.testing.assert_close(fit.matrix, torch.eye(2, dtype=torch.float64))
    torch.testing.assert_close(fit.offset, offset)
    torch.testing.assert_close(fit.apply(source), target)
    assert fit.geometry == "shift"


def test_fit_affine_rscale_rotation_scale_translation() -> None:
    source = _points()
    angle = math.radians(17.0)
    scale = 1.3
    matrix = scale * torch.tensor(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=torch.float64,
    )
    offset = torch.tensor([-0.2, 0.7], dtype=torch.float64)
    target = torch.matmul(source, matrix.T) + offset

    fit = fit_affine_2d(source, target, geometry="rscale")

    torch.testing.assert_close(fit.matrix, matrix, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(fit.offset, offset, atol=1e-12, rtol=1e-12)
    assert float(fit.rms) < 1e-12


def test_fit_affine_general_and_sigma_clip() -> None:
    source = _points()
    matrix = torch.tensor([[1.1, 0.2], [-0.3, 0.9]], dtype=torch.float64)
    offset = torch.tensor([0.4, -0.8], dtype=torch.float64)
    target = torch.matmul(source, matrix.T) + offset
    target_with_outlier = target.clone()
    target_with_outlier[-1] += torch.tensor([10.0, -10.0], dtype=torch.float64)

    fit = fit_affine_2d(
        source,
        target_with_outlier,
        geometry="general",
        sigma_clip=1.0,
        max_iter=5,
    )

    torch.testing.assert_close(fit.matrix, matrix, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(fit.offset, offset, atol=1e-12, rtol=1e-12)
    assert not bool(fit.inlier_mask[-1])
    assert int(fit.inlier_mask.sum().item()) >= 3


def test_tangent_plane_frame_roundtrip_and_autograd() -> None:
    frame = TangentPlaneFrame(ra0=180.0, dec0=2.0)
    ra = torch.tensor([179.9, 180.0, 180.2], dtype=torch.float64, requires_grad=True)
    dec = torch.tensor([1.9, 2.0, 2.2], dtype=torch.float64)

    xi, eta = frame.world_to_plane(ra, dec)
    ra_back, dec_back = frame.plane_to_world(xi, eta)
    xi.sum().backward()

    torch.testing.assert_close(ra_back, ra.detach(), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(dec_back, dec, atol=1e-12, rtol=1e-12)
    assert ra.grad is not None
    assert bool(torch.isfinite(ra.grad).all())


def test_fit_sky_affine_shift_in_tangent_plane() -> None:
    frame = TangentPlaneFrame(ra0=42.0, dec0=-12.0)
    source_plane = torch.tensor(
        [[-0.02, -0.01], [0.03, -0.01], [0.00, 0.04], [0.04, 0.03]],
        dtype=torch.float64,
    )
    shift = torch.tensor([0.004, -0.006], dtype=torch.float64)
    source_ra, source_dec = frame.plane_to_world(source_plane[:, 0], source_plane[:, 1])
    target_ra, target_dec = frame.plane_to_world(
        source_plane[:, 0] + shift[0],
        source_plane[:, 1] + shift[1],
    )

    fit = fit_sky_affine(
        source_ra,
        source_dec,
        target_ra,
        target_dec,
        frame=frame,
        geometry="shift",
    )
    corrected_ra, corrected_dec = fit.apply(source_ra, source_dec)

    torch.testing.assert_close(fit.affine.offset, shift, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(corrected_ra, target_ra, atol=1e-10, rtol=1e-12)
    torch.testing.assert_close(corrected_dec, target_dec, atol=1e-10, rtol=1e-12)
    assert float(fit.rms) < 1e-12
