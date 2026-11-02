"""Declared detector calibration, correlated covariance and stale-state guards."""

from dataclasses import FrozenInstanceError, replace

import pytest
import torch

from torchsky.map.evidence import EvidenceSnapshot
from torchsky.map.operators.continuous_response import NativeQuadratureResponseOp
from torchsky.map.operators.pixel_calibration import PixelCalibrationOp


def _calibration(scale=None, *, dependencies=None):
    if scale is None:
        scale = torch.tensor([0.7, 1.2, 2.1], dtype=torch.float64)
    return PixelCalibrationOp(
        scale=scale,
        input_unit="electron",
        output_unit="sky_reference_flux",
        calibration_id="flat-v1",
        provenance_id="independent-flat-exposure",
        flat_convention="sky_flat",
        reference_convention="spatially uniform spectral radiance with declared reference SED",
        dependencies=dependencies,
    )


def test_signed_means_transpose_full_covariance_and_prefixes():
    op = _calibration()
    values = torch.tensor([[1.0, -2.0, 3.0], [-0.5, 0.8, 1.1]], dtype=torch.float64)
    residuals = values.flip(-1)
    matrix = torch.tensor(
        [[2.0, 0.4, -0.2], [0.4, 1.3, 0.3], [-0.2, 0.3, 2.4]], dtype=torch.float64
    )
    diagonal = torch.diag(op.scale)
    torch.testing.assert_close(op.forward(values), values @ diagonal)
    torch.testing.assert_close(
        (op.forward(values) * residuals).sum(), (values * op.adjoint(residuals)).sum()
    )
    dense = diagonal @ matrix @ diagonal
    torch.testing.assert_close(op.transform_covariance(matrix), dense)
    torch.testing.assert_close(
        op.transform_covariance(torch.stack((matrix, matrix * 2))),
        torch.stack((dense, dense * 2)),
    )
    action = op.apply_covariance(values, covariance_action=lambda x: x @ matrix.T)
    torch.testing.assert_close(action, values @ dense.T)
    assert dense[0, 1] != 0
    assert not torch.allclose(action, (values @ matrix.T) * op.scale)
    assert not torch.allclose(op.adjoint(values), values / op.scale)


def test_varying_psf_sky_flat_does_not_preserve_every_point_source_flux():
    # Independent declared finite native responses: each point's fractions sum
    # to one, but the spatially varying columns illuminate detector rows unequally.
    fractions = torch.tensor(
        [[0.8, 0.1, 0.2], [0.15, 0.7, 0.1], [0.05, 0.2, 0.7]], dtype=torch.float64
    )
    torch.testing.assert_close(fractions.sum(0), torch.ones(3, dtype=torch.float64))
    assert not torch.allclose(fractions.sum(1), torch.ones(3, dtype=torch.float64))

    def response(detector_xy, source_xy, wavelength_angstrom, *, detector_footprint):
        return fractions[detector_xy[..., 0].long(), source_xy[..., 0].long()].expand(
            *torch.broadcast_shapes(
                detector_xy.shape[:-1], source_xy.shape[:-1], wavelength_angstrom.shape
            )
        )

    xy = torch.stack(
        (torch.arange(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64)),
        dim=-1,
    )
    transport = NativeQuadratureResponseOp(
        response=response,
        source_xy=xy,
        detector_xy=xy,
        wavelength_angstrom=torch.tensor([5000.0], dtype=torch.float64),
        source_measure=torch.ones(3, dtype=torch.float64),
        wavelength_measure=torch.ones(1, dtype=torch.float64),
        source_chunk_size=1,
        detector_chunk_size=2,
    )
    sky_flat = _calibration(1 / fractions.sum(1))
    sky = transport.forward(torch.ones(3, 1, dtype=torch.float64))
    torch.testing.assert_close(
        sky_flat.forward(sky), torch.ones(3, dtype=torch.float64)
    )
    point = transport.forward(torch.tensor([[10.0], [0.0], [0.0]], dtype=torch.float64))
    transformed_point = sky_flat.forward(point)
    expected_point = 10 * fractions[:, 0] / fractions.sum(1)
    torch.testing.assert_close(transformed_point, expected_point)
    assert not torch.isclose(transformed_point.sum(), point.sum())
    input_covariance = (
        fractions
        @ torch.diag(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
        @ fractions.T
    )
    expected_covariance = input_covariance / (
        fractions.sum(1)[:, None] * fractions.sum(1)[None, :]
    )
    torch.testing.assert_close(
        sky_flat.transform_covariance(input_covariance), expected_covariance
    )
    values = torch.tensor([0.4, -0.1, 0.8], dtype=torch.float64)
    torch.testing.assert_close(
        (transformed_point * values).sum(),
        (
            torch.tensor([[10.0], [0.0], [0.0]], dtype=torch.float64)
            * transport.adjoint(sky_flat.adjoint(values))
        ).sum(),
    )


def test_first_and_second_gradients_in_calibration_mean_and_covariance():
    log_scale = torch.tensor([-0.1, 0.2, 0.3], dtype=torch.float64, requires_grad=True)
    value = torch.tensor([0.2, -0.3, 0.7], dtype=torch.float64, requires_grad=True)
    factor = torch.tensor(
        [[1.1, 0.3, 0.0], [0.1, 0.7, 0.2], [0.4, 0.2, 0.9]],
        dtype=torch.float64,
        requires_grad=True,
    )

    def apply(scale, x, b):
        op = _calibration(
            scale.exp(), dependencies=EvidenceSnapshot.capture({"covariance_factor": b})
        )
        covariance = b @ b.T
        return torch.cat(
            (
                op.forward(x),
                op.adjoint(x),
                op.apply_covariance(x, covariance_action=lambda z: covariance @ z),
                op.transform_covariance(covariance).flatten(),
            )
        )

    args = (log_scale, value, factor)
    assert torch.autograd.gradcheck(apply, args, fast_mode=True)
    assert torch.autograd.gradgradcheck(apply, args, fast_mode=True)


def test_tensor_mutation_and_hidden_revision_invalidate_calibration():
    op = _calibration()
    op.scale[0] = 1.0
    with pytest.raises(RuntimeError, match="stale evidence dependency"):
        op.forward(torch.ones(3, dtype=torch.float64))
    revision = [0]
    scale = torch.tensor([0.7, 1.2, 2.1], dtype=torch.float64)
    op = _calibration(
        scale, dependencies=EvidenceSnapshot.capture(revision=lambda: revision[0])
    )
    scale.numpy()[0] = 0.9  # NumPy alias bypasses Torch's mutation counter.
    revision[0] += 1
    with pytest.raises(RuntimeError, match="stale evidence revision"):
        op.adjoint(torch.ones(3, dtype=torch.float64))
    with pytest.raises(FrozenInstanceError):
        op.calibration_id = "replacement"


def test_covariance_dependency_mutations_before_and_during_action_fail_closed():
    covariance = torch.eye(3, dtype=torch.float64)
    op = _calibration(
        dependencies=EvidenceSnapshot.capture({"input_covariance": covariance})
    )
    values = torch.ones(3, dtype=torch.float64)

    def action(value):
        covariance[0, 0] += 1
        return covariance @ value

    with pytest.raises(RuntimeError, match="stale evidence dependency"):
        op.apply_covariance(values, covariance_action=action)
    with pytest.raises(RuntimeError, match="stale evidence dependency"):
        op.transform_covariance(covariance)


def test_inference_scale_requires_explicit_revision():
    with torch.inference_mode():
        scale = torch.tensor([0.7, 1.2, 2.1], dtype=torch.float64)
    with pytest.raises(ValueError, match="explicit revision"):
        _calibration(scale)
    revision = [0]
    op = _calibration(
        scale, dependencies=EvidenceSnapshot.capture(revision=lambda: revision[0])
    )
    torch.testing.assert_close(op.forward(torch.ones(3, dtype=torch.float64)), scale)
    with torch.inference_mode():
        scale[0] = 0.8
    revision[0] += 1
    with pytest.raises(RuntimeError, match="stale evidence revision"):
        op.validate_evidence()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_device_dtype_and_empty_contract(dtype, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if device == "mps" and (
        not torch.backends.mps.is_available() or dtype == torch.float64
    ):
        pytest.skip("MPS unavailable or float64 unsupported")
    scale = torch.tensor([0.7, 1.2, 2.1], dtype=dtype, device=device)
    op = _calibration(scale)
    value = torch.tensor([1.0, -2.0, 3.0], dtype=dtype, device=device)
    output = op.forward(value)
    assert output.dtype == dtype and output.device.type == device
    torch.testing.assert_close(
        op.apply_covariance(value, covariance_action=lambda x: x),
        value * scale.square(),
    )
    empty = replace(op, scale=scale[:0])
    assert empty.forward(value[:0]).shape == (0,)
    assert empty.transform_covariance(
        torch.empty((0, 0), dtype=dtype, device=device)
    ).shape == (0, 0)


def test_invalid_scale_metadata_vectors_and_covariance_actions():
    op = _calibration()
    for scale in (
        torch.tensor([0.0, 1.0, 2.0]),
        torch.tensor([-0.1, 1.0, 2.0]),
        torch.tensor([float("nan"), 1.0, 2.0]),
    ):
        with pytest.raises(ValueError, match="strictly positive"):
            replace(op, scale=scale)
    with pytest.raises(TypeError, match="float32 or float64"):
        replace(op, scale=torch.ones(3, dtype=torch.int64))
    for field in (
        "input_unit",
        "output_unit",
        "calibration_id",
        "provenance_id",
        "flat_convention",
        "reference_convention",
    ):
        with pytest.raises(ValueError, match="nonempty"):
            replace(op, **{field: " "})
    with pytest.raises(ValueError, match="trailing shape"):
        op.forward(torch.ones(2, dtype=torch.float64))
    with pytest.raises(ValueError, match="dtype and device"):
        op.forward(torch.ones(3, dtype=torch.float32))
    with pytest.raises(ValueError, match="finite"):
        op.adjoint(torch.tensor([0.0, float("inf"), 0.0], dtype=torch.float64))
    with pytest.raises(ValueError, match="trailing shape"):
        op.transform_covariance(torch.eye(2, dtype=torch.float64))
    with pytest.raises(ValueError, match="preserve"):
        op.apply_covariance(
            torch.ones(3, dtype=torch.float64), covariance_action=lambda x: x[None]
        )
