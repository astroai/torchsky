"""Tests for constrained learned instrument-operator families."""

import torch
from torch import nn

from torchsky.map.operators import (
    AdaptiveAxisFeatures,
    AdaptiveSpatioSpectralTemporalEncoder,
    AffineRadiometry,
    ConditionedBackgroundField,
    ConditionedKernelPSF,
    ConstrainedKernelPSF,
    GatedResidualOperator,
    InstrumentStateEncoder,
    LearnedInstrumentOperator,
    SpatioSpectralTemporalEncoder,
    SubPixelShift,
    gaussian_kernel2d,
)


def _conditioning_encoder() -> SpatioSpectralTemporalEncoder:
    return SpatioSpectralTemporalEncoder(
        spatial_center=(16.0, 16.0),
        spatial_scale=(12.0, 20.0),
        wavelength_center=7000.0,
        spectral_scale=1000.0,
        time_center_mjd=60000.0,
        temporal_scale_days=30.0,
        spatial_frequencies=2,
        spectral_frequencies=2,
        temporal_frequencies=1,
    )


def _adaptive_conditioning_encoder() -> AdaptiveSpatioSpectralTemporalEncoder:
    return AdaptiveSpatioSpectralTemporalEncoder(
        spatial_center=(16.0, 16.0),
        spatial_scale=(12.0, 20.0),
        spatial_length_bounds=(2.0, 64.0),
        wavelength_center=7000.0,
        spectral_scale=1000.0,
        spectral_length_bounds=(100.0, 4000.0),
        time_center_mjd=60000.0,
        temporal_scale_days=30.0,
        temporal_length_bounds_days=(0.25, 120.0),
        spatial_bands=3,
        spectral_bands=2,
        temporal_bands=2,
    )


def test_adaptive_axis_features_are_unit_invariant_and_bounded() -> None:
    axis = AdaptiveAxisFeatures(
        center=10.0,
        reference_scale=4.0,
        length_bounds=(1.0, 16.0),
        num_bands=4,
    )
    rescaled = AdaptiveAxisFeatures(
        center=100.0,
        reference_scale=40.0,
        length_bounds=(10.0, 160.0),
        num_bands=4,
    )
    values = torch.linspace(-2.0, 22.0, 31)
    torch.testing.assert_close(axis(values), rescaled(10.0 * values))

    with torch.no_grad():
        axis.scale_shift_raw.fill_(100.0)
    assert bool(torch.all(axis.length_scales <= 16.0))
    assert bool(torch.all(axis.length_scales >= 1.0))
    with torch.no_grad():
        axis.scale_shift_raw.fill_(-100.0)
    assert bool(torch.all(axis.length_scales <= 16.0))
    assert bool(torch.all(axis.length_scales >= 1.0))


def test_adaptive_axis_state_dict_preserves_physical_metadata() -> None:
    source = AdaptiveAxisFeatures(
        center=10.0,
        reference_scale=4.0,
        length_bounds=(1.0, 16.0),
        num_bands=3,
    )
    restored = AdaptiveAxisFeatures(
        center=0.0,
        reference_scale=1.0,
        length_bounds=(0.1, 1.0),
        num_bands=3,
    )
    with torch.no_grad():
        source.gate_logits.copy_(torch.tensor([-1.0, 0.5, 2.0]))
        source.scale_shift_raw.fill_(0.4)
    restored.load_state_dict(source.state_dict())
    values = torch.linspace(-2.0, 22.0, 31)

    torch.testing.assert_close(restored(values), source(values))
    assert restored.center == 10.0
    assert restored.reference_scale == 4.0
    assert restored.length_bounds == (1.0, 16.0)


def test_adaptive_axis_learns_continuous_correlation_length() -> None:
    torch.manual_seed(12)
    axis = AdaptiveAxisFeatures(
        center=0.0,
        reference_scale=8.0,
        length_bounds=(4.0, 16.0),
        num_bands=1,
    )
    head = nn.Linear(axis.output_dim, 1, bias=False)
    values = torch.linspace(-8.0, 8.0, 257)
    target = torch.sin((2.0 * torch.pi / 10.0) * values)
    optimizer = torch.optim.Adam((*axis.parameters(), *head.parameters()), lr=0.03)

    for _ in range(250):
        optimizer.zero_grad()
        prediction = head(axis(values)).squeeze(-1)
        loss = torch.mean(torch.square(prediction - target))
        loss.backward()
        optimizer.step()

    assert float(loss.detach()) < 2.0e-3
    torch.testing.assert_close(
        axis.length_scales.detach(),
        torch.tensor([10.0]),
        rtol=2e-2,
        atol=2e-2,
    )


def test_adaptive_conditioning_regularization_reaches_all_axes() -> None:
    encoder = _adaptive_conditioning_encoder()
    y, x = torch.meshgrid(torch.arange(9.0), torch.arange(11.0), indexing="ij")
    encoded = encoder(x, y, 7500.0, 60010.0)
    penalty = encoder.regularization(gate_weight=0.2, scale_shift_weight=0.1)
    (encoded.square().mean() + penalty).backward()

    assert encoded.shape == (9, 11, encoder.output_dim)
    for axis in (
        encoder.x_axis,
        encoder.y_axis,
        encoder.spectral_axis,
        encoder.temporal_axis,
    ):
        assert axis.gate_logits.grad is not None
        assert axis.scale_shift_raw.grad is not None


def test_multiscale_conditioning_axes_have_independent_scales():
    encoder = _conditioning_encoder()
    x = torch.tensor([16.0, 28.0])
    y = torch.tensor([16.0, 16.0])
    encoded = encoder(x, y, torch.tensor([7000.0, 8000.0]), 60030.0)
    assert encoded.shape == (2, encoder.output_dim)
    assert torch.isfinite(encoded).all()
    # Raw normalized x is the first feature: one spatial scale is exactly one.
    torch.testing.assert_close(encoded[:, 0], torch.tensor([0.0, 1.0]))


def test_conditioned_kernel_psf_is_flux_conserving_and_differentiable():
    psf = ConditionedKernelPSF(
        _adaptive_conditioning_encoder(),
        n_basis=3,
        kernel_size=9,
        init_fwhm_pix=2.0,
    )
    with torch.no_grad():
        psf.mixer[-1].weight.normal_(std=0.2)
    y, x = torch.meshgrid(torch.arange(33.0), torch.arange(33.0), indexing="ij")
    image = torch.zeros((33, 33), requires_grad=True)
    with torch.no_grad():
        image[16, 16] = 1.0
    output = psf(
        image,
        spatial_x=x,
        spatial_y=y,
        wavelength=7500.0,
        time_mjd=60010.0,
    )
    torch.testing.assert_close(output.sum(), torch.tensor(1.0), atol=1e-5, rtol=1e-5)
    assert bool((output >= 0).all())
    (
        output.square().sum()
        + 0.01
        * psf.conditioning_regularization(
            gate_weight=1.0,
            scale_shift_weight=1.0,
        )
    ).backward()
    assert image.grad is not None
    assert psf.kernel_basis.raw_kernels.grad is not None
    assert isinstance(psf.encoder, AdaptiveSpatioSpectralTemporalEncoder)
    assert psf.encoder.x_axis.gate_logits.grad is not None


def test_conditioned_background_zero_initialization_and_gradients():
    background = ConditionedBackgroundField(_conditioning_encoder(), hidden_dim=16)
    y, x = torch.meshgrid(torch.arange(7.0), torch.arange(9.0), indexing="ij")
    initial = background(x, y, 7000.0, 60000.0)
    torch.testing.assert_close(initial, torch.zeros_like(initial))
    with torch.no_grad():
        background.network[-1].weight.normal_(std=0.1)
    changed = background(x, y, 8000.0, 60030.0)
    assert not torch.allclose(changed, initial)
    changed.square().mean().backward()
    assert background.network[0].weight.grad is not None


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
