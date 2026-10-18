"""Finite source quadrature, chromatic native transport and exact adjoints."""

from dataclasses import replace
import math

import numpy as np
import pytest
import torch
from scipy.special import ndtr

from torchsky.map.operators.continuous_response import (
    AchromaticPixelResponse,
    NativeQuadratureResponseOp,
)
from torchsky.map.operators.epsf import GaussianPixelResponse


def _grid(lo=-3, hi=4, *, dtype=torch.float64, device="cpu"):
    axis = torch.arange(lo, hi, dtype=dtype, device=device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(-1, 2)


def _op(*, dtype=torch.float64, device="cpu", source_xy=None, response=None):
    sources = (
        torch.tensor([[0.21, 0.37], [-0.52, 1.13]], dtype=dtype, device=device)
        if source_xy is None
        else source_xy
    )
    return NativeQuadratureResponseOp(
        response=AchromaticPixelResponse(
            GaussianPixelResponse(torch.tensor(0.8, dtype=dtype, device=device))
        )
        if response is None
        else response,
        source_xy=sources,
        detector_xy=_grid(dtype=dtype, device=device),
        wavelength_angstrom=torch.tensor([5000.0, 7000.0], dtype=dtype, device=device),
        source_measure=torch.ones(sources.shape[0], dtype=dtype, device=device),
        wavelength_measure=torch.tensor([2.0, 3.0], dtype=dtype, device=device),
        source_chunk_size=1,
        detector_chunk_size=11,
        wavelength_chunk_size=1,
    )


def _all_pairs(op):
    return torch.cartesian_prod(
        torch.arange(op.detector_xy.shape[0], device=op.source_xy.device),
        torch.arange(op.source_xy.shape[0], device=op.source_xy.device),
    )


@pytest.mark.parametrize("sparse", [False, True])
def test_linearity_adjoint_prefix_masks_and_chunk_equivalence(sparse):
    op = _op()
    op.source_spectral_weight = torch.tensor(
        [[0.7, 0.3], [0.2, 0.9]], dtype=torch.float64
    )
    op.mask = torch.arange(op.detector_xy.shape[0]) % 3 != 0
    if sparse:
        op = replace(
            op,
            interaction_pairs=_all_pairs(op),
            interaction_support="all pairs retained",
            pair_chunk_size=13,
        )
    generator = torch.Generator().manual_seed(31)
    x = torch.randn(2, 3, 2, 2, generator=generator, dtype=torch.float64)
    z = torch.randn(x.shape, generator=generator, dtype=torch.float64)
    y = torch.randn(
        2, 3, op.detector_xy.shape[0], generator=generator, dtype=torch.float64
    )
    torch.testing.assert_close(op.forward(2 * x - z), 2 * op.forward(x) - op.forward(z))
    torch.testing.assert_close(
        (op.forward(x) * y).sum(), (x * op.adjoint(y)).sum(), atol=2e-14, rtol=2e-14
    )
    assert bool((op.forward(x)[..., ~op.mask] == 0).all())
    bad_masked = y.clone()
    bad_masked[..., ~op.mask] = torch.nan
    torch.testing.assert_close(op.adjoint(bad_masked), op.adjoint(y))
    wide = replace(
        op,
        source_chunk_size=99,
        detector_chunk_size=99,
        wavelength_chunk_size=99,
        pair_chunk_size=999,
    )
    torch.testing.assert_close(op.forward(x), wide.forward(x), atol=2e-14, rtol=2e-14)
    torch.testing.assert_close(op.adjoint(y), wide.adjoint(y), atol=2e-14, rtol=2e-14)
    dense = replace(op, interaction_pairs=None, interaction_support=None)
    torch.testing.assert_close(op.forward(x), dense.forward(x))
    torch.testing.assert_close(op.adjoint(y), dense.adjoint(y))


def test_extended_gaussian_source_matches_analytic_convolution():
    # Gauss-Hermite integrates the source measure before detector integration.
    # Independent oracle: convolution adds source and PSF variances.
    nodes, weights = np.polynomial.hermite.hermgauss(18)
    sigma_source = np.array([0.45, 0.65])
    centroid = np.array([0.17, -0.24])
    xx, yy = np.meshgrid(nodes, nodes, indexing="xy")
    xy = (
        np.stack((xx, yy), axis=-1).reshape(-1, 2) * np.sqrt(2) * sigma_source
        + centroid
    )
    measure = np.outer(weights, weights).reshape(-1) / np.pi
    op = _op(source_xy=torch.from_numpy(xy))
    op.source_measure = torch.from_numpy(measure)
    op.wavelength_measure = torch.ones_like(op.wavelength_measure)
    op.source_chunk_size = 64
    psf_sigma = np.array([0.7, 0.9])
    op.response = AchromaticPixelResponse(
        GaussianPixelResponse(torch.from_numpy(psf_sigma))
    )
    flux = torch.zeros((len(xy), 2), dtype=torch.float64)
    flux[:, 0] = 7.0
    offset = op.detector_xy.numpy() - centroid
    convolved_sigma = np.sqrt(psf_sigma**2 + sigma_source**2)
    expected = 7 * (
        ndtr((offset + 0.5) / convolved_sigma) - ndtr((offset - 0.5) / convolved_sigma)
    ).prod(axis=-1)
    torch.testing.assert_close(
        op.forward(flux), torch.from_numpy(expected), atol=3e-11, rtol=2e-9
    )


class _ChromaticGaussian:
    def __call__(
        self, detector_xy, source_xy, wavelength_angstrom, *, detector_footprint
    ):
        assert detector_footprint is None
        offset = detector_xy - source_xy
        shift = (wavelength_angstrom - 5000) * 0.0003
        offset = offset - torch.stack((shift, torch.zeros_like(shift)), dim=-1)
        sigma = 0.9 + (wavelength_angstrom - 5000) * 0.0001
        lo, hi = (
            (offset - 0.5) / (math.sqrt(2) * sigma[..., None]),
            (offset + 0.5) / (math.sqrt(2) * sigma[..., None]),
        )
        fraction = (0.5 * (torch.erf(hi) - torch.erf(lo))).prod(dim=-1)
        throughput = 1 - (wavelength_angstrom - 5000) * 0.00025
        return fraction * throughput


def test_chromatic_response_changes_flux_and_centroid_before_pixel_output():
    op = _op(
        source_xy=torch.tensor([[0.2, 0.3]], dtype=torch.float64),
        response=_ChromaticGaussian(),
    )
    op.detector_xy = _grid(-8, 9)
    op.wavelength_measure = torch.ones(2, dtype=torch.float64)
    colors = torch.tensor([[[9.0, 1.0]], [[1.0, 9.0]]], dtype=torch.float64)
    prediction = op.forward(colors)
    offsets = op.detector_xy.numpy() - np.array([0.2, 0.3])
    columns = []
    for wavelength in (5000, 7000):
        shifted = offsets - np.array([0.0003 * (wavelength - 5000), 0.0])
        sigma = 0.9 + 0.0001 * (wavelength - 5000)
        throughput = 1 - 0.00025 * (wavelength - 5000)
        columns.append(
            throughput
            * (ndtr((shifted + 0.5) / sigma) - ndtr((shifted - 0.5) / sigma)).prod(
                axis=-1
            )
        )
    expected = colors[:, 0].numpy() @ np.stack(columns)
    torch.testing.assert_close(
        prediction, torch.from_numpy(expected), atol=4e-15, rtol=2e-12
    )
    total = prediction.sum(dim=-1)
    torch.testing.assert_close(
        total, torch.tensor([9.5, 5.5], dtype=torch.float64), atol=1e-11, rtol=1e-11
    )
    centroid_x = (prediction * op.detector_xy[:, 0]).sum(dim=-1) / total
    torch.testing.assert_close(
        centroid_x,
        torch.tensor([0.2 + 0.3 / 9.5, 0.2 + 2.7 / 5.5], dtype=torch.float64),
        atol=5e-8,
        rtol=0,
    )


def test_source_position_conditioning_and_source_side_band_weight():
    def response(detector_xy, source_xy, wavelength_angstrom, *, detector_footprint):
        offset = detector_xy - source_xy
        sigma = 0.9 + 0.1 * source_xy[..., 0] + (wavelength_angstrom - 5000) * 0.0001
        scale = math.sqrt(2) * sigma[..., None]
        return (
            0.5
            * (torch.erf((offset + 0.5) / scale) - torch.erf((offset - 0.5) / scale))
        ).prod(dim=-1)

    op = _op(response=response)
    op.source_spectral_weight = torch.tensor([0.3, 0.8], dtype=torch.float64)
    intensity = torch.tensor([[1.0, 2.0], [3.0, 0.5]], dtype=torch.float64)
    expected = np.zeros(op.detector_xy.shape[0])
    for source in range(2):
        offset = op.detector_xy.numpy() - op.source_xy[source].numpy()
        for wave in range(2):
            sigma = (
                0.9
                + 0.1 * op.source_xy[source, 0].item()
                + (op.wavelength_angstrom[wave].item() - 5000) * 0.0001
            )
            integrated = (
                ndtr((offset + 0.5) / sigma) - ndtr((offset - 0.5) / sigma)
            ).prod(axis=-1)
            expected += (
                integrated
                * intensity[source, wave].item()
                * op.wavelength_measure[wave].item()
                * op.source_spectral_weight[wave].item()
            )
    torch.testing.assert_close(
        op.forward(intensity), torch.from_numpy(expected), atol=2e-15, rtol=2e-12
    )


def test_radiance_uses_source_solid_angle_and_one_detector_integral():
    def response(detector_xy, source_xy, wavelength_angstrom, *, detector_footprint):
        shape = torch.broadcast_shapes(
            detector_xy.shape[:-1], source_xy.shape[:-1], wavelength_angstrom.shape
        )
        return (detector_footprint[:, 0, None, None] / 10).expand(shape)

    op = _op(response=response)
    op.detector_xy = torch.zeros((2, 2), dtype=torch.float64)
    op.detector_footprint = torch.tensor([[1.0], [2.0]], dtype=torch.float64)
    op.source_measure = torch.tensor([0.4, 0.6], dtype=torch.float64)  # sr
    spectrum = torch.tensor([[2.0, 3.0], [2.0, 3.0]], dtype=torch.float64)
    expected = torch.tensor([1.3, 2.6], dtype=torch.float64)
    torch.testing.assert_close(op.forward(spectrum), expected)
    # Point-flux nodes do not inherit these solid-angle weights.
    torch.testing.assert_close(
        replace(op, source_measure=torch.ones(2, dtype=torch.float64)).forward(
            spectrum
        ),
        expected * 2,
    )


def test_boundary_and_mask_loss_keep_tails_without_aperture_renormalization():
    op = _op(source_xy=torch.tensor([[-0.3, 0.2]], dtype=torch.float64))
    op.detector_xy = _grid(0, 3)
    op.wavelength_measure = torch.ones(2, dtype=torch.float64)
    intensity = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    offset = op.detector_xy.numpy() - op.source_xy.numpy()[0]
    expected = (ndtr((offset + 0.5) / 0.8) - ndtr((offset - 0.5) / 0.8)).prod(axis=-1)
    torch.testing.assert_close(op.forward(intensity), torch.from_numpy(expected))
    assert 0 < op.forward(intensity).sum() < 0.7
    op.mask = torch.arange(9) != 0
    assert op.forward(intensity).sum() < expected.sum()


class _CompactUniformResponse:
    def __init__(self):
        self.evaluated = 0
        self.max_block = 0

    def __call__(
        self, detector_xy, source_xy, wavelength_angstrom, *, detector_footprint
    ):
        offset = detector_xy - source_xy
        # Uniform square PSF with exact finite support [-0.6,0.6]^2.
        overlap = (
            (offset + 0.5).clamp(max=0.6) - (offset - 0.5).clamp(min=-0.6)
        ).clamp(min=0)
        result = (
            (overlap / 1.2)
            .prod(dim=-1)
            .expand(
                *torch.broadcast_shapes(offset.shape[:-1], wavelength_angstrom.shape)
            )
        )
        self.evaluated += result.numel()
        self.max_block = max(self.max_block, result.numel())
        return result


def test_exact_compact_support_graph_evaluates_only_declared_edges():
    response = _CompactUniformResponse()
    op = _op(response=response)
    dense_values = op.forward(torch.ones(2, 2, dtype=torch.float64))
    offset = op.detector_xy[:, None] - op.source_xy[None]
    pairs = torch.nonzero((offset.abs() < 1.1).all(dim=-1))
    response.evaluated = response.max_block = 0
    sparse = replace(
        op,
        interaction_pairs=pairs,
        interaction_support="exact overlap of unit pixels and square PSF support [-0.6,0.6]^2",
        pair_chunk_size=3,
    )
    torch.testing.assert_close(
        sparse.forward(torch.ones(2, 2, dtype=torch.float64)), dense_values
    )
    assert response.evaluated == pairs.shape[0] * 2
    assert response.max_block <= 3
    assert pairs.shape[0] < op.detector_xy.shape[0] * 2
    y = torch.arange(op.detector_xy.shape[0], dtype=torch.float64)
    torch.testing.assert_close(sparse.adjoint(y), op.adjoint(y))


@pytest.mark.parametrize("sparse", [False, True])
def test_first_and_second_gradients_in_positions_response_and_inputs(sparse):
    sources = torch.tensor([[0.21, 0.37]], dtype=torch.float64, requires_grad=True)
    log_sigma = torch.tensor(-0.2, dtype=torch.float64, requires_grad=True)
    intensity = torch.tensor([[0.7, 1.2]], dtype=torch.float64, requires_grad=True)
    detector_values = torch.tensor([0.3, -0.7], dtype=torch.float64, requires_grad=True)

    def apply(xy, sigma, x, y):
        op = _op(
            source_xy=xy,
            response=AchromaticPixelResponse(GaussianPixelResponse(sigma.exp())),
        )
        op.detector_xy = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
        if sparse:
            op = replace(
                op,
                interaction_pairs=_all_pairs(op),
                interaction_support="all pairs retained",
                pair_chunk_size=1,
            )
        return torch.cat((op.forward(x), op.adjoint(y).flatten()))

    args = (sources, log_sigma, intensity, detector_values)
    assert torch.autograd.gradcheck(apply, args, fast_mode=True)
    assert torch.autograd.gradgradcheck(apply, args, fast_mode=True)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_dtype_device_and_empty_contracts(dtype, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if device == "mps" and (
        not torch.backends.mps.is_available() or dtype == torch.float64
    ):
        pytest.skip("MPS unavailable or float64 unsupported")
    op = _op(dtype=dtype, device=device)
    x = torch.ones((2, 2), dtype=dtype, device=device)
    out = op.forward(x)
    assert out.dtype == dtype and out.device.type == device
    torch.testing.assert_close((out * out).sum(), (x * op.adjoint(out)).sum())
    empty_source = replace(
        op, source_xy=op.source_xy[:0], source_measure=op.source_measure[:0]
    )
    torch.testing.assert_close(empty_source.forward(x[:0]), torch.zeros_like(out))
    assert empty_source.adjoint(out).shape == (0, 2)
    empty_detector = replace(op, detector_xy=op.detector_xy[:0])
    assert empty_detector.forward(x).shape == (0,)
    torch.testing.assert_close(empty_detector.adjoint(out[:0]), torch.zeros_like(x))


def test_invalid_shapes_weights_and_interaction_graph_fail_closed():
    op = _op()
    with pytest.raises(ValueError, match="trailing shape"):
        op.forward(torch.ones(2, dtype=torch.float64))
    with pytest.raises(ValueError, match="nonnegative"):
        replace(op, source_measure=-op.source_measure)
    pairs = _all_pairs(op)
    with pytest.raises(ValueError, match="declaration"):
        replace(op, interaction_pairs=pairs)
    with pytest.raises(ValueError, match="duplicate"):
        replace(
            op,
            interaction_pairs=torch.cat((pairs, pairs[:1])),
            interaction_support="all",
        )
    with pytest.raises(ValueError, match="out of bounds"):
        replace(op, interaction_pairs=pairs + 100, interaction_support="all")
    with pytest.raises(ValueError, match="unit-square"):
        replace(
            op,
            detector_footprint=torch.ones(
                op.detector_xy.shape[0], 1, dtype=torch.float64
            ),
        ).forward(torch.ones(2, 2, dtype=torch.float64))


@pytest.mark.parametrize("empty", ["graph", "sources", "detectors"])
@pytest.mark.parametrize("prefix", [(), (2, 3)])
def test_empty_support_preserves_exact_zero_input_gradients(empty, prefix):
    op = _op()
    if empty == "graph":
        op = replace(
            op,
            interaction_pairs=torch.empty((0, 2), dtype=torch.int64),
            interaction_support="exact empty support",
        )
    elif empty == "sources":
        op = replace(
            op, source_xy=op.source_xy[:0], source_measure=op.source_measure[:0]
        )
    else:
        op = replace(op, detector_xy=op.detector_xy[:0])
    intensity = torch.ones(
        (*prefix, op.source_xy.shape[0], 2), dtype=torch.float64, requires_grad=True
    )
    dual = torch.ones(
        (*prefix, op.detector_xy.shape[0]), dtype=torch.float64, requires_grad=True
    )
    forward, adjoint = op.forward(intensity), op.adjoint(dual)
    assert forward.shape == dual.shape and adjoint.shape == intensity.shape
    torch.testing.assert_close(forward, torch.zeros_like(dual))
    torch.testing.assert_close(adjoint, torch.zeros_like(intensity))
    torch.testing.assert_close(
        torch.autograd.grad(forward.sum(), intensity)[0], torch.zeros_like(intensity)
    )
    torch.testing.assert_close(
        torch.autograd.grad(adjoint.sum(), dual)[0], torch.zeros_like(dual)
    )
    if empty == "graph":
        assert torch.autograd.gradcheck(op.forward, (intensity,), fast_mode=True)
        assert torch.autograd.gradgradcheck(op.forward, (intensity,), fast_mode=True)
        assert torch.autograd.gradcheck(op.adjoint, (dual,), fast_mode=True)
        assert torch.autograd.gradgradcheck(op.adjoint, (dual,), fast_mode=True)
