"""Tests for constrained learned instrument-operator families."""

import torch

from torchsky.map.operators import (
    AffineRadiometry,
    ConstrainedKernelPSF,
    GatedResidualOperator,
    InstrumentStateEncoder,
    LearnedInstrumentOperator,
    SubPixelShift,
    gaussian_kernel2d,
)


def test_gaussian_kernel2d_unit_sum():
    k = gaussian_kernel2d(1.5, 15)
    assert k.shape == (15, 15)
    assert torch.isclose(k.sum(), torch.tensor(1.0), atol=1e-6)


def test_constrained_kernel_psf_flux_conservation():
    psf = ConstrainedKernelPSF(n_basis=3, kernel_size=9, init_fwhm_pix=2.0)
    img = torch.zeros(2, 1, 33, 33)
    img[:, :, 16, 16] = 1.0
    out = psf(img)
    assert out.shape == img.shape
    # Unit-sum kernels conserve flux for sources away from the boundary.
    assert torch.allclose(out.sum(dim=(-2, -1)), img.sum(dim=(-2, -1)), atol=1e-4)
    # Output must be a blurred point source (positive, peaked at centre).
    assert out.min() >= -1e-7
    assert out[0, 0, 16, 16] == out[0, 0].max()


def test_constrained_kernel_psf_conditioning_changes_kernel():
    psf = ConstrainedKernelPSF(n_basis=3, kernel_size=9, init_fwhm_pix=2.0, cond_dim=4)
    with torch.no_grad():
        psf.cond_head.weight.normal_(std=1.0)
    cond = torch.randn(2, 4)
    k = psf.effective_kernels(cond, batch_size=2)
    assert k.shape == (2, 9, 9)
    assert not torch.allclose(k[0], k[1])


def test_gated_residual_identity_at_init():
    op = GatedResidualOperator(hidden_channels=8)
    img = torch.randn(2, 1, 16, 16)
    out = op(img)
    assert torch.allclose(out, img)
    assert float(op.gate_penalty().detach()) == 0.0


def test_affine_radiometry_init_matches_gain_init():
    rad = AffineRadiometry()
    img = torch.randn(3, 1, 8, 8)
    gain = torch.tensor([1.0, 2.0, 0.5])
    out = rad(img, gain_init=gain)
    assert torch.allclose(out, img * gain.view(-1, 1, 1, 1), atol=1e-6)


def test_subpixel_shift_moves_point_source():
    shifter = SubPixelShift()
    img = torch.zeros(1, 1, 17, 17)
    img[0, 0, 8, 8] = 1.0
    out = shifter(img, torch.tensor([[2.0, 0.0]]))
    # Peak should move +2 pixels in x.
    peak = out[0, 0].argmax()
    y, x = divmod(int(peak), 17)
    assert (y, x) == (8, 10)


def test_instrument_state_encoder_masked_pooling():
    enc = InstrumentStateEncoder(in_channels=2, latent_dim=8)
    stamps = torch.randn(2, 3, 2, 16, 16)
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    z = enc(stamps, mask)
    assert z.shape == (2, 8)
    # Masked stamps must not affect the latent.
    stamps2 = stamps.clone()
    stamps2[0, 2] = 999.0
    z2 = enc(stamps2, mask)
    assert torch.allclose(z, z2, atol=1e-5)


def test_learned_instrument_operator_curriculum_and_grads():
    op = LearnedInstrumentOperator(
        n_basis=2, kernel_size=7, init_fwhm_pix=2.0, cond_dim=6
    )
    assert op.stage == "frozen"
    assert not op.psf.raw_kernels.requires_grad
    op.set_stage("core")
    assert op.psf.raw_kernels.requires_grad
    assert not op.residual.gate.requires_grad
    op.set_stage("full")
    assert op.residual.gate.requires_grad

    img = torch.randn(2, 1, 24, 24)
    gain = torch.ones(2)
    cond = torch.randn(2, 6)
    out = op(img, gain_init=gain, cond=cond)
    assert out.shape == img.shape
    (out.mean() + op.regularization(cond)).backward()
    assert op.psf.raw_kernels.grad is not None
