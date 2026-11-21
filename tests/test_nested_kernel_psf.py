"""Nested-level PSF: transport contract, accuracy against direct convolution, cost."""

from __future__ import annotations

import time
import weakref
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

import torchsky.map.operators.nested_kernel as nested_module
from torchsky.map.field.rendering import _apply_learned_psf
from torchsky.map.operators import LearnedPSFOp
from torchsky.map.operators.nested_kernel import NestedKernelPSF
from torchsky.map.operators.psf import apply_kernel2d

F64 = torch.float64


def _grid(height, width, x0=0.0, y0=0.0, dtype=F64):
    y, x = torch.meshgrid(
        torch.arange(height, dtype=dtype) + y0,
        torch.arange(width, dtype=dtype) + x0,
        indexing="ij",
    )
    return x, y


def _scatter(image, kernels):
    """Independent reference: each source deposits its stamp at source + offset."""
    height, width = image.shape
    ky, kx = kernels.shape[-2:]
    kernels = kernels.expand(height, width, ky, kx)
    out = torch.zeros_like(image)
    for dy in range(ky):
        for dx in range(kx):
            oy, ox = dy - ky // 2, dx - kx // 2
            sy0, sy1 = max(0, -oy), min(height, height - oy)
            sx0, sx1 = max(0, -ox), min(width, width - ox)
            if (
                sy0 >= sy1 or sx0 >= sx1
            ):  # stamp larger than the image: offset is off the detector
                continue
            out[sy0 + oy : sy1 + oy, sx0 + ox : sx1 + ox] += (
                image[sy0:sy1, sx0:sx1] * kernels[sy0:sy1, sx0:sx1, dy, dx]
            )
    return out


def _taper(r, extent, fraction=0.85):
    """The C1 cut-off of the kernel at ``extent``, written independently of the module."""
    t = ((r - fraction * extent) / ((1.0 - fraction) * extent)).clamp(0.0, 1.0)
    return 1.0 - t * t * (3.0 - 2.0 * t)


def _wing_field(offsets, condition):
    """Gaussian core (FWHM 2 pixels) plus a power-law wing; density per pixel^2."""
    r2 = (offsets**2).sum(-1)
    sigma = 2.0 / 2.3548200450309493
    core = 0.6 * torch.exp(-r2 / (2.0 * sigma**2)) / (2.0 * torch.pi * sigma**2)
    wing = 0.2 * (1.0 + r2 / 9.0) ** (-1.5) / (2.0 * torch.pi * 9.0)
    return (core + wing)[None]


def _ideal_stamp(field, extent, dtype=F64):
    """Dense unit-sum stamp of the continuous field, cut at ``extent`` (the target)."""
    half = int(extent)
    axis = torch.arange(-half, half + 1, dtype=dtype)
    iy, ix = torch.meshgrid(axis, axis, indexing="ij")
    offsets = torch.stack([ix.reshape(-1), iy.reshape(-1)], dim=-1)
    stamp = field(offsets, None).reshape(ix.shape) * _taper(torch.hypot(ix, iy), extent)
    return stamp / stamp.sum()


def _direct(image, stamp, strip=4):
    """``F.conv2d`` with the dense stamp, in float32 and strips of kernel rows.

    One ``conv2d`` call with a 400 x 400 kernel needs tens of gigabytes of
    workspace; summing over strips of kernel rows is the same convolution.
    """
    height, width = image.shape
    ky, kx = stamp.shape
    flipped = stamp.float().flip(-2, -1)
    padded = F.pad(image.float()[None, None], (kx // 2, kx // 2, ky // 2, ky // 2))
    out = torch.zeros(1, 1, height, width)
    for r in range(0, ky, strip):
        part = flipped[r : r + strip][None, None]
        out = out + F.conv2d(padded[:, :, r : r + part.shape[2] + height - 1, :], part)
    return out[0, 0].to(image.dtype)


# --------------------------------------------------------------- the contract


def test_stamp_constructor_is_apply_kernel2d_with_orientation_edge_loss_and_flux():
    stamp = torch.tensor(
        [
            [1.0, 2.0, 3.0, 0.5, 0.0],
            [4.0, 8.0, 16.0, 2.0, 1.0],
            [5.0, 6.0, 7.0, 1.0, 3.0],
        ],
        dtype=F64,
    )
    stamp = stamp / stamp.sum()
    psf = NestedKernelPSF.from_stamp(stamp)
    assert psf.num_levels == 1 and psf.kernel_size == 5
    x, y = _grid(9, 11)
    image = torch.zeros(9, 11, dtype=F64)
    image[4, 5] = 1.0
    out = psf(image, spatial_x=x, spatial_y=y, wavelength=0.0)
    torch.testing.assert_close(out, apply_kernel2d(image, stamp), atol=1e-14, rtol=0)
    torch.testing.assert_close(
        out[3:6, 3:8], stamp, atol=1e-14, rtol=0
    )  # impulse response
    assert abs(float(out.sum()) - 1.0) < 1e-14
    torch.testing.assert_close(
        psf.effective_kernels(x[4:5, 5], y[4:5, 5], 0.0)[0, 1:4, 0:5],
        stamp,
        atol=1e-14,
        rtol=0,
    )
    corner = torch.zeros(9, 11, dtype=F64)
    corner[0, 0] = 3.0
    lost = psf(corner, spatial_x=x, spatial_y=y, wavelength=0.0)
    torch.testing.assert_close(lost, _scatter(corner, stamp), atol=1e-14, rtol=0)
    torch.testing.assert_close(lost.sum(), 3.0 * stamp[1:, 2:].sum())
    assert float(lost.sum()) < 3.0
    random = torch.randn(9, 11, dtype=F64, generator=torch.Generator().manual_seed(1))
    torch.testing.assert_close(
        psf(random, spatial_x=x, spatial_y=y, wavelength=0.0),
        apply_kernel2d(random, stamp),
        atol=1e-13,
        rtol=0,
    )


def test_one_level_field_equals_apply_kernel2d_to_round_off():
    psf = NestedKernelPSF(_wing_field, core_half_width=16, extent=12.0)
    assert psf.num_levels == 1
    stamp = _ideal_stamp(_wing_field, 12.0)
    x, y = _grid(40, 50)
    image = torch.randn(40, 50, dtype=F64, generator=torch.Generator().manual_seed(2))
    out = psf(image, spatial_x=x, spatial_y=y, wavelength=0.0)
    torch.testing.assert_close(out, apply_kernel2d(image, stamp), atol=1e-13, rtol=0)
    assert psf.kernel_size == 2 * 11 + 1  # the taper is exactly zero at r = extent


def _asymmetric_field(offsets, condition):
    """Peaked at (dx, dy) = (+3, -2) with an asymmetric wing."""
    dx, dy = offsets[:, 0], offsets[:, 1]
    peak = torch.exp(-((dx - 3.0) ** 2 + (dy + 2.0) ** 2) / (2.0 * 1.5**2))
    wing = (
        0.02
        * (1.0 + (dx**2 + 0.5 * dy**2) / 16.0) ** (-1.5)
        * (1.0 + 0.5 * torch.tanh(dx / 4.0))
    )
    return (peak + wing)[None]


def test_multilevel_transport_matches_dense_scatter_of_effective_stamps():
    # the peak (r = 3.6) lies inside the exactly resolved core, n_h / 2 = 4 pixels
    psf = NestedKernelPSF(_asymmetric_field, core_half_width=8, extent=24.0)
    assert psf.num_levels == 3
    height, width = 14, 17
    x, y = _grid(
        height, width, x0=3.0, y0=-2.0
    )  # the lattice follows the detector coordinates
    stamps = psf.effective_kernels(x, y, 0.0)
    half = psf.kernel_size // 2
    assert stamps.shape == (height, width, psf.kernel_size, psf.kernel_size)
    # Orientation: the stamp is the impulse response, peaked at (dx, dy) = (+3, -2).
    flat = stamps[5, 6].argmax()
    assert (
        int(flat // psf.kernel_size) - half,
        int(flat % psf.kernel_size) - half,
    ) == (-2, 3)
    # Unit flux when the stamp is on the detector (stamps are computed off the edge).
    torch.testing.assert_close(
        stamps.sum((-2, -1)), torch.ones(height, width, dtype=F64), atol=1e-12, rtol=0
    )
    assert (
        float(stamps.min()) > -1e-15
    )  # linear interpolation: non-negative up to FFT round-off
    # Dense transport matrices by unit impulses, against the scatter of the stamps.
    eye = torch.eye(height * width, dtype=F64).reshape(-1, height, width)
    forward = torch.stack(
        [psf(e, spatial_x=x, spatial_y=y, wavelength=0.0).reshape(-1) for e in eye],
        dim=1,
    )
    reference = torch.stack([_scatter(e, stamps).reshape(-1) for e in eye], dim=1)
    torch.testing.assert_close(forward, reference, atol=1e-13, rtol=0)
    adjoint = torch.stack(
        [
            psf.adjoint(e, spatial_x=x, spatial_y=y, wavelength=0.0).reshape(-1)
            for e in eye
        ],
        dim=1,
    )
    torch.testing.assert_close(adjoint, forward.T, atol=1e-13, rtol=0)
    # Boundary loss is not renormalised: a corner source keeps only what lands on the detector.
    corner_sum = float(forward[:, 0].sum())
    assert 0.0 < corner_sum < 1.0
    expected = float(stamps[0, 0, half:, half:][:height, :width].sum())
    assert abs(corner_sum - expected) < 1e-13
    # Interior source on a detector that holds its stamp: all the flux is rendered.
    big = torch.zeros(2 * half + 9, 2 * half + 9, dtype=F64)
    big[half + 4, half + 4] = 1.0
    bx, by = _grid(*big.shape)
    assert (
        abs(float(psf(big, spatial_x=bx, spatial_y=by, wavelength=0.0).sum()) - 1.0)
        < 1e-12
    )


def test_stamp_is_close_to_the_dense_target_near_the_core():
    psf = NestedKernelPSF(_wing_field, core_half_width=16, extent=40.0)
    stamp = psf.effective_kernels(torch.tensor([200.0]), torch.tensor([-33.0]), 0.0)[0]
    ideal = _ideal_stamp(_wing_field, 40.0)
    half, ideal_half = psf.kernel_size // 2, 40
    cropped = stamp[
        half - ideal_half : half + ideal_half + 1,
        half - ideal_half : half + ideal_half + 1,
    ]
    assert float((cropped - ideal).abs().max() / ideal.max()) < 1e-4
    # Level 0 is the exact kernel: the core differs from the target only by the
    # normalisation of the node quadrature.
    core = slice(half - 4, half + 5)
    torch.testing.assert_close(
        stamp[core, core], ideal[36:45, 36:45], rtol=2e-4, atol=0
    )


# ---------------------------------------------------------------- the adjoint


def _two_width_field(offsets, condition):
    sigma = 1.5 + 1.5 * (condition[:, 0:1] - 16.0) / 32.0  # [N, 1]
    r2 = (offsets**2).sum(-1)[None]
    return torch.exp(-r2 / (2.0 * sigma**2)) + 0.02 * (1.0 + r2 / 16.0) ** (-1.5)


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
@pytest.mark.parametrize("blended", [False, True])
def test_adjoint_is_the_exact_transpose(interpolation, blended):
    kwargs = {"nodes": (2, 3), "bounds": (10.0, 54.0, 5.0, 40.0)} if blended else {}
    psf = NestedKernelPSF(
        _two_width_field if blended else _wing_field,
        core_half_width=4,
        extent=26.0,
        interpolation=interpolation,
        **kwargs,
    )
    gen = torch.Generator().manual_seed(3)
    x, y = _grid(37, 53, x0=7.0, y0=-3.0)  # odd sizes, lattice phase not zero
    a = torch.randn(37, 53, dtype=F64, generator=gen)
    b = torch.randn(37, 53, dtype=F64, generator=gen)
    lhs = (psf(a, spatial_x=x, spatial_y=y, wavelength=0.5) * b).sum()
    rhs = (a * psf.adjoint(b, spatial_x=x, spatial_y=y, wavelength=0.5)).sum()
    assert abs(float(lhs - rhs)) < 1e-10 * max(1.0, abs(float(lhs)))


# ------------------------------------------------- accuracy against direct conv


def _accuracy(core_half_width, extent, size, interpolation="linear", seed=0):
    psf = NestedKernelPSF(
        _wing_field,
        core_half_width=core_half_width,
        extent=extent,
        interpolation=interpolation,
    )
    x, y = _grid(size, size)
    stamp = _ideal_stamp(_wing_field, extent)
    gen = torch.Generator().manual_seed(seed)
    point = torch.zeros(size, size, dtype=F64)
    point[size // 2 + 1, size // 2 + 2] = 1.0
    smooth = torch.exp(
        -((x - size / 2) ** 2 + (y - size / 2) ** 2) / (2.0 * (size / 12.0) ** 2)
    )
    noise = torch.randn(size, size, dtype=F64, generator=gen)
    errors = {}
    for name, image in (("point", point), ("smooth", smooth), ("white noise", noise)):
        out = psf(image, spatial_x=x, spatial_y=y, wavelength=0.0)
        reference = _direct(image, stamp)
        errors[name] = float((out - reference).abs().max() / reference.abs().max())
    return errors


def test_accuracy_against_direct_convolution_with_wings_to_200_pixels():
    errors = _accuracy(16, 200.0, 192)
    print(
        "\nnested vs F.conv2d, 192 x 192, extent 200, n_h 16 (max error / max |direct|):",
        errors,
    )
    # The target is 1e-3 of the peak for a point source and a smooth source.
    assert errors["point"] < 1e-4
    assert errors["smooth"] < 1e-4
    # White noise is the worst case: its power sits at the pixel scale, which the
    # coarse levels (pitch up to 16 pixels, read back by interpolation) cannot
    # carry; only the smooth kernel wing multiplying it is represented there, so
    # the error is the interpolation error of every coarse shell at once.
    assert errors["white noise"] < 5e-4
    assert errors["white noise"] > errors["smooth"]


def test_error_falls_with_core_half_width_and_does_not_hinge_on_the_window_fraction(
    monkeypatch,
):
    coarse = _accuracy(8, 60.0, 112)
    fine = _accuracy(16, 60.0, 112)
    print("\nerror vs n_h (extent 60):", {8: coarse, 16: fine})
    for name in coarse:
        assert fine[name] < coarse[name]
    # _INNER_FRACTION is a threshold of the scheme: it places the cross-over of
    # adjacent windows.  The error is flat across a wide range, so the choice is
    # not tuned; this test would flag a change that made it matter.
    results = {}
    for fraction in (0.35, 0.5, 0.65):
        monkeypatch.setattr(nested_module, "_INNER_FRACTION", fraction)
        results[fraction] = _accuracy(16, 60.0, 112)
    print("error vs window fraction (n_h 16, extent 60):", results)
    for errors in results.values():
        assert max(errors.values()) < 5e-4


def test_cubic_interpolation_is_no_worse_on_smooth_sources():
    linear = _accuracy(8, 60.0, 112, "linear")
    cubic = _accuracy(8, 60.0, 112, "cubic")
    print("\nlinear:", linear, "cubic:", cubic)
    assert cubic["smooth"] < linear["smooth"]


# ----------------------------------------------------------- spatial variation


def _width_field(sigma):
    def field(offsets, condition):
        r2 = (offsets**2).sum(-1)
        return (torch.exp(-r2 / (2.0 * sigma**2)) + 0.02 * (1.0 + r2 / 16.0) ** (-1.5))[
            None
        ]

    return field


def test_region_nodes_render_node_kernels_and_blend_with_tents():
    narrow, wide = _width_field(1.5), _width_field(3.0)

    def varying(offsets, condition):
        # node 0 (x = 16) narrow, node 1 (x = 48) wide: pick by the condition's x
        return torch.where(
            condition[:, 0:1] < 32.0, narrow(offsets, None), wide(offsets, None)
        )

    kwargs = {"core_half_width": 4, "extent": 20.0}
    blended = NestedKernelPSF(
        varying, nodes=(1, 2), bounds=(16.0, 48.0, 0.0, 0.0), **kwargs
    )
    stat_narrow = NestedKernelPSF(narrow, **kwargs)
    stat_wide = NestedKernelPSF(wide, **kwargs)
    ys = torch.full((7,), 21.0, dtype=F64)
    xs = torch.tensor([16.0, 20.0, 24.0, 32.0, 40.0, 48.0, 51.0], dtype=F64)
    got = blended.effective_kernels(xs, ys, 0.0)
    a = stat_narrow.effective_kernels(xs, ys, 0.0)
    b = stat_wide.effective_kernels(xs, ys, 0.0)
    assert not torch.allclose(a, b, atol=1e-3)
    torch.testing.assert_close(got[0], a[0], atol=1e-13, rtol=0)  # at node 0
    torch.testing.assert_close(got[5], b[5], atol=1e-13, rtol=0)  # at node 1
    torch.testing.assert_close(
        got[6], b[6], atol=1e-13, rtol=0
    )  # constant beyond the last node
    for i, w in ((1, 0.125), (2, 0.25), (3, 0.5), (4, 0.75)):
        torch.testing.assert_close(
            got[i], (1 - w) * a[i] + w * b[i], atol=1e-13, rtol=0
        )
    # Partition of unity: unit flux for a source anywhere, including between nodes.
    torch.testing.assert_close(
        got.sum((-2, -1)), torch.ones(7, dtype=F64), atol=1e-12, rtol=0
    )
    # The rendered image is the scatter of the blended stamps (source-position weights).
    x, y = _grid(10, 36, x0=14.0, y0=18.0)
    image = torch.randn(10, 36, dtype=F64, generator=torch.Generator().manual_seed(4))
    out = blended(image, spatial_x=x, spatial_y=y, wavelength=0.0)
    expected = _scatter(image, blended.effective_kernels(x, y, 0.0))
    torch.testing.assert_close(out, expected, atol=1e-12, rtol=0)


def test_two_dimensional_nodes_interpolate_bilinearly_in_the_source_position():
    corners = {(0, 0): None, (40, 0): None, (0, 30): None, (40, 30): None}

    def make_field(node_x, node_y):
        def field(offsets, condition):
            sigma = (
                1.2 + 0.03 * node_x + 0.05 * node_y
            )  # a different width at every node
            r2 = (offsets**2).sum(-1)[None]
            return torch.exp(-r2 / (2.0 * sigma**2)) + 0.01 / (1.0 + r2 / 9.0)

        return field

    def varying(offsets, condition):
        return torch.cat(
            [
                make_field(float(c[0]), float(c[1]))(offsets, None)
                for c in condition[:, :2]
            ],
            dim=0,
        )

    kwargs = {"core_half_width": 4, "extent": 16.0}
    psf = NestedKernelPSF(
        varying, nodes=(2, 2), bounds=(0.0, 40.0, 0.0, 30.0), **kwargs
    )
    stationary = {c: NestedKernelPSF(make_field(*c), **kwargs) for c in corners}

    def at(model, x, y):
        return model.effective_kernels(torch.tensor([x]), torch.tensor([y]), 0.0)[0]

    # The stamp depends on the source position modulo the coarsest pitch, so the
    # blend is compared with stationary operators evaluated at the same position.
    for px, py, weights in (
        (20.0, 15.0, {(0, 0): 0.25, (40, 0): 0.25, (0, 30): 0.25, (40, 30): 0.25}),
        (10.0, 0.0, {(0, 0): 0.75, (40, 0): 0.25}),
        (40.0, 22.5, {(40, 0): 0.25, (40, 30): 0.75}),
        (
            -7.0,
            3.0,
            {(0, 0): 0.9, (0, 30): 0.1},
        ),  # constant beyond the outer nodes in x
    ):
        expected = sum(w * at(stationary[c], px, py) for c, w in weights.items())
        torch.testing.assert_close(at(psf, px, py), expected, atol=1e-13, rtol=0)


# -------------------------------------------------------------------- gradients


def test_gradients_with_respect_to_image_and_kernel_field_parameters():
    def run(image, log_sigma, log_wing, slope):
        def field(offsets, condition):
            sigma = torch.exp(log_sigma + slope * condition[:, 0:1] / 10.0)
            r2 = (offsets**2).sum(-1)[None]
            return torch.exp(-r2 / (2.0 * sigma**2)) + 0.05 * (
                1.0 + r2 / torch.exp(log_wing) ** 2
            ) ** (-1.5)

        psf = NestedKernelPSF(
            field,
            core_half_width=3,
            extent=9.0,
            nodes=(2, 2),
            bounds=(0.0, 11.0, 0.0, 9.0),
        )
        x, y = _grid(10, 12)
        return psf(image, spatial_x=x, spatial_y=y, wavelength=0.0)

    gen = torch.Generator().manual_seed(5)
    inputs = (
        torch.randn(10, 12, dtype=F64, generator=gen, requires_grad=True),
        torch.tensor(0.2, dtype=F64, requires_grad=True),
        torch.tensor(1.1, dtype=F64, requires_grad=True),
        torch.tensor(0.3, dtype=F64, requires_grad=True),
    )
    assert torch.autograd.gradcheck(run, inputs, eps=1e-6, atol=1e-7, rtol=1e-5)


class _MoffatField(nn.Module):
    def __init__(self, dtype=F64):
        super().__init__()
        self.log_width = nn.Parameter(torch.tensor(0.5, dtype=dtype))
        self.log_beta = nn.Parameter(torch.tensor(0.4, dtype=dtype))

    def forward(self, offsets, condition):
        r2 = (offsets**2).sum(-1)[None]
        return (1.0 + r2 / torch.exp(self.log_width) ** 2) ** (
            -torch.exp(self.log_beta) - 1.0
        )


def test_module_field_parameters_receive_gradients_and_adjoint_is_differentiable():
    field = _MoffatField()
    psf = NestedKernelPSF(field, core_half_width=4, extent=20.0)
    x, y = _grid(24, 24)
    image = torch.rand(24, 24, dtype=F64, requires_grad=True)
    dual = torch.rand(24, 24, dtype=F64)
    loss = (psf(image, spatial_x=x, spatial_y=y, wavelength=0.0) * dual).sum()
    grads = torch.autograd.grad(loss, (image, *field.parameters()))
    assert all(g is not None and bool(torch.isfinite(g).all()) for g in grads)
    assert float(grads[1].abs()) > 0.0 and float(grads[2].abs()) > 0.0
    # d<A x, y>/dx = A^T y, and the same value through the adjoint.
    torch.testing.assert_close(
        grads[0], psf.adjoint(dual, spatial_x=x, spatial_y=y, wavelength=0.0)
    )
    adj = (psf.adjoint(dual, spatial_x=x, spatial_y=y, wavelength=0.0) * image).sum()
    parameter_grads = torch.autograd.grad(adj, tuple(field.parameters()))
    for g, ref in zip(parameter_grads, grads[1:], strict=True):
        torch.testing.assert_close(g, ref, atol=1e-12, rtol=1e-10)


def test_float32_matches_float64_and_keeps_dtype():
    x64, y64 = _grid(48, 48)
    image = torch.rand(48, 48, dtype=F64, generator=torch.Generator().manual_seed(6))
    ref = NestedKernelPSF(_wing_field, core_half_width=8, extent=40.0)(
        image, spatial_x=x64, spatial_y=y64, wavelength=0.0
    )
    psf = NestedKernelPSF(_wing_field, core_half_width=8, extent=40.0)
    x32, y32 = x64.float(), y64.float()
    out = psf(image.float(), spatial_x=x32, spatial_y=y32, wavelength=0.0)
    assert out.dtype == torch.float32
    assert float((out.double() - ref).abs().max() / ref.abs().max()) < 1e-5
    back = psf.adjoint(image.float(), spatial_x=x32, spatial_y=y32, wavelength=0.0)
    assert back.dtype == torch.float32
    lhs = float((out.double() * image).sum())
    rhs = float((image.float().double() * back.double()).sum())
    assert abs(lhs - rhs) < 1e-4 * abs(lhs)
    # A float32 coordinate grid far from the origin is still a unit-pitch grid.
    far32 = _grid(48, 48, x0=5000.0, y0=-7000.0, dtype=torch.float32)
    far64 = _grid(48, 48, x0=5000.0, y0=-7000.0)
    torch.testing.assert_close(
        psf(
            image,
            spatial_x=far32[0].double(),
            spatial_y=far32[1].double(),
            wavelength=0.0,
        ),
        psf(image, spatial_x=far64[0], spatial_y=far64[1], wavelength=0.0),
        atol=0,
        rtol=0,
    )
    far_out = psf(image.float(), spatial_x=far32[0], spatial_y=far32[1], wavelength=0.0)
    assert (
        float(
            (
                far_out.double()
                - psf(image, spatial_x=far64[0], spatial_y=far64[1], wavelength=0.0)
            )
            .abs()
            .max()
        )
        < 1e-5
    )
    # A float32 kernel module with a float32 image; a float64 module with a float64 image.
    f32 = NestedKernelPSF(_MoffatField(torch.float32), core_half_width=4, extent=12.0)
    assert (
        f32(
            image.float()[:20, :20],
            spatial_x=x32[:20, :20],
            spatial_y=y32[:20, :20],
            wavelength=0.0,
        ).dtype
        == torch.float32
    )


# --------------------------------------------------- integration with rendering


def test_learned_psf_op_and_tiled_rendering_match_the_untiled_result():
    psf = NestedKernelPSF(
        _two_width_field,
        core_half_width=4,
        extent=18.0,
        nodes=(2, 2),
        bounds=(10.0, 55.0, 3.0, 40.0),
    )
    operator = LearnedPSFOp(model=psf)
    assert operator.learned and operator.model is psf
    observation = SimpleNamespace(detector_x0_pix=5.0, detector_y0_pix=-2.0)
    image = torch.randn(41, 58, dtype=F64, generator=torch.Generator().manual_seed(7))
    kwargs = {"wavelength": 0.5, "forward_context": None}
    whole = _apply_learned_psf(
        image, observation, operator, spatial_tile_hw=None, **kwargs
    )
    tiled = _apply_learned_psf(
        image, observation, operator, spatial_tile_hw=(13, 17), **kwargs
    )
    # Lattices are anchored to detector coordinates, so tiling with a halo of
    # kernel_size // 2 reproduces the untiled render to round-off.
    torch.testing.assert_close(tiled, whole, atol=1e-12, rtol=0)
    assert psf.kernel_size % 2 == 1 and psf.kernel_size > 2 * 18


# ------------------------------------------------------------------- failures


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"core_half_width": 1}, "core_half_width"),
        ({"core_half_width": 4.5}, "core_half_width"),
        ({"extent": 0.0}, "extent"),
        ({"extent": float("nan")}, "extent"),
        ({"extent": float("inf")}, "extent"),
        ({"taper_fraction": 0.0}, "taper_fraction"),
        ({"taper_fraction": 1.0}, "taper_fraction"),
        ({"nodes": (0, 1)}, "nodes"),
        ({"nodes": (2, 2)}, "bounds"),
        ({"nodes": (1, 2), "bounds": (5.0, 5.0, 0.0, 0.0)}, "bounds"),
        ({"nodes": (2, 1), "bounds": (0.0, 1.0, 3.0, 2.0)}, "bounds"),
        ({"bounds": (0.0, float("nan"), 0.0, 1.0)}, "bounds"),
        ({"interpolation": "nearest"}, "interpolation"),
    ],
)
def test_bad_geometry_is_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        NestedKernelPSF(_wing_field, **{"core_half_width": 4, "extent": 20.0, **kwargs})


def test_non_callable_field_is_rejected():
    with pytest.raises(TypeError, match="callable"):
        NestedKernelPSF(torch.ones(3))


@pytest.mark.parametrize(
    ("stamp", "match"),
    [
        (torch.ones(4, 3, dtype=F64), "odd"),
        (torch.ones(3, 2, dtype=F64), "odd"),
        (torch.ones(3, dtype=F64), "2-D"),
        (torch.zeros(3, 3, dtype=F64), "positive sum"),
        (-torch.ones(3, 3, dtype=F64), "non-negative"),
        (torch.tensor([[1.0, float("nan"), 1.0]], dtype=F64), "finite"),
    ],
)
def test_bad_stamps_are_rejected(stamp, match):
    with pytest.raises(ValueError, match=match):
        NestedKernelPSF.from_stamp(stamp)


def test_bad_kernel_fields_and_coordinates_are_rejected_at_call():
    x, y = _grid(12, 12)
    image = torch.ones(12, 12, dtype=F64)
    call = {"spatial_x": x, "spatial_y": y, "wavelength": 0.0}
    kwargs = {"core_half_width": 4, "extent": 10.0}

    def negative(offsets, condition):
        return (torch.exp(-(offsets**2).sum(-1)) - 0.5)[None]

    with pytest.raises(ValueError, match="non-negative"):
        NestedKernelPSF(negative, **kwargs)(image, **call)
    with pytest.raises(ValueError, match="non-finite"):
        NestedKernelPSF(
            lambda o, c: torch.full((1, o.shape[0]), float("nan"), dtype=o.dtype),
            **kwargs,
        )(image, **call)
    with pytest.raises(ValueError, match="zero total weight"):
        NestedKernelPSF(
            lambda o, c: torch.zeros(1, o.shape[0], dtype=o.dtype), **kwargs
        )(image, **call)
    with pytest.raises(ValueError, match="must return"):
        NestedKernelPSF(
            lambda o, c: torch.ones(3, o.shape[0], dtype=o.dtype), **kwargs
        )(image, **call)
    psf = NestedKernelPSF(_wing_field, **kwargs)
    with pytest.raises(ValueError, match="unit-pitch"):
        psf(image, spatial_x=x * 1.5, spatial_y=y, wavelength=0.0)
    with pytest.raises(ValueError, match="unit-pitch"):
        psf(image, spatial_x=x.T, spatial_y=y.T, wavelength=0.0)
    with pytest.raises(ValueError, match="match image shape"):
        psf(image, spatial_x=x[:, :-1], spatial_y=y, wavelength=0.0)
    with pytest.raises(ValueError, match="2-D"):
        psf(image[None], spatial_x=x, spatial_y=y, wavelength=0.0)
    with pytest.raises(ValueError, match="floating"):
        psf(image.long(), spatial_x=x, spatial_y=y, wavelength=0.0)
    with pytest.raises(ValueError, match="same shape"):
        psf.effective_kernels(torch.zeros(2), torch.zeros(3), 0.0)


# ----------------------------------------------------------------------- cost


class _LiveBytes(TorchDispatchMode):
    """Peak bytes of distinct live tensor storages created while active."""

    def __init__(self):
        super().__init__()
        self.live: dict[int, list[int]] = {}
        self.current = 0
        self.peak = 0

    def _release(self, key):
        entry = self.live[key]
        entry[1] -= 1
        if entry[1] == 0:
            self.current -= entry[0]
            del self.live[key]

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for tensor in tree_leaves(out):
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.untyped_storage().nbytes() == 0
            ):
                continue
            storage = tensor.untyped_storage()
            key = storage.data_ptr()
            if key not in self.live:
                self.live[key] = [storage.nbytes(), 0]
                self.current += storage.nbytes()
                self.peak = max(self.peak, self.current)
            self.live[key][1] += 1
            weakref.finalize(tensor, self._release, key)
        return out


def _measure(fn, repeats=2):
    fn()  # warm-up (FFT plans, allocator)
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    with torch.no_grad(), _LiveBytes() as tracker:
        fn()
    return best, tracker.peak / 2**20


def test_cost_does_not_grow_with_the_kernel_extent(capsys):
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        _cost_table(capsys)
    finally:
        torch.set_num_threads(threads)


def _cost_table(capsys):
    size = 512
    dtype = torch.float32
    x, y = _grid(size, size, dtype=dtype)
    image = torch.rand(
        size, size, dtype=dtype, generator=torch.Generator().manual_seed(8)
    )
    rows = {}
    for extent in (32, 128, 512):
        psf = NestedKernelPSF(_wing_field, core_half_width=16, extent=float(extent))
        nested = _measure(
            lambda psf=psf: psf(image, spatial_x=x, spatial_y=y, wavelength=0.0)
        )
        stamp = _ideal_stamp(_wing_field, extent, dtype=dtype)
        work = size * size * (2 * extent + 1) ** 2
        if (
            work > 4.0e10
        ):  # > ~60 s here, and the 4-row-strip workspace is small only by design
            dense = None
        else:
            dense = _measure(lambda stamp=stamp: _direct(image, stamp), repeats=1)
        rows[extent] = (psf.num_levels, nested, dense)
    with capsys.disabled():
        print(
            "\nextent  levels  nested s  nested MiB  dense s  dense MiB (512 x 512, float32, 1 thread)"
        )
        for extent, (levels, nested, dense) in rows.items():
            tail = (
                "   skipped (> 4e10 multiply-adds)"
                if dense is None
                else f"{dense[0]:9.2f} {dense[1]:10.1f}"
            )
            print(f"{extent:6d} {levels:7d} {nested[0]:9.2f} {nested[1]:11.1f} {tail}")
    assert rows[512][1][0] < 4.0 * rows[32][1][0]
