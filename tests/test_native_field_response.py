"""Independent bounded native-field transport, GLS and lifetime references."""

from dataclasses import replace
import math

import numpy as np
import pytest
from scipy.linalg import solve
from scipy.special import ndtr
import torch
from torch import nn

from torchsky.map.evidence import EvidenceSnapshot
from torchsky.map.field.core import SineImplicitField
from torchsky.map.field.native_response import (
    NativeFieldResponse,
    bind_native_field_likelihood,
    render_native_field_observation,
)
from torchsky.map.field.types import (
    FieldDomain,
    FieldObservation,
    ForwardContext,
    PSFSpec,
)
from torchsky.map.likelihood import GaussianLikelihoodSystem
from torchsky.map.operators.background import (
    AdditiveComponent,
    AdditiveFrame,
    AdditiveProcessingState,
    AdditiveStage,
)
from torchsky.map.operators.continuous_response import NativeQuadratureResponseOp
from torchsky.map.operators.pixel_calibration import PixelCalibrationOp
from torchsky.map.query import fit_linear_query


class _Table:
    def __init__(self, matrix):
        self.matrix = matrix

    def __call__(
        self, detector_xy, source_xy, wavelength_angstrom, *, detector_footprint
    ):
        return self.matrix[
            detector_xy[..., 0].long(),
            source_xy[..., 0].long(),
            (wavelength_angstrom / 5000 - 1).long(),
        ]


class _Field(nn.Module):
    def __init__(self, offset=0.3):
        super().__init__()
        self.offset = nn.Parameter(torch.tensor(offset, dtype=torch.float64))
        self.calls = []

    def forward(self, ra, dec, wavelength, *, forward_context=None):
        self.calls.append((ra.detach().clone(), float(wavelength)))
        epoch = 0 if forward_context is None else forward_context.time_mjd
        return self.offset + (ra - 100) * (wavelength / 5000) + dec * 0.1 + epoch

    def render_bandpass(self, *args, **kwargs):
        raise AssertionError("monochromatic values must precede native response")


def _response(*, mask=False, dtype=torch.float64, device="cpu"):
    matrix = torch.tensor(
        [
            [[0.8, 0.1], [0.1, 0.1]],
            [[0.1, 0.8], [0.1, 0.1]],
            [[0.1, 0.1], [0.8, 0.1]],
            [[0.1, 0.1], [0.1, 0.8]],
        ],
        dtype=dtype,
        device=device,
    )
    kernel = _Table(matrix)
    native = NativeQuadratureResponseOp(
        response=kernel,
        source_xy=torch.tensor([[0, 0], [1, 0]], dtype=dtype, device=device),
        detector_xy=torch.tensor(
            [[0, 0], [1, 0], [2, 0], [3, 0]], dtype=dtype, device=device
        ),
        wavelength_angstrom=torch.tensor([5000, 10000], dtype=dtype, device=device),
        source_measure=torch.tensor([0.2, 0.5], dtype=dtype, device=device),
        wavelength_measure=torch.tensor([10, 20], dtype=dtype, device=device),
        source_spectral_weight=torch.tensor(
            [[0.8, 0.6], [0.4, 0.9]], dtype=dtype, device=device
        ),
        mask=torch.tensor([True, False, True, True], device=device) if mask else None,
    )
    calibration = PixelCalibrationOp(
        torch.tensor([0.8, 1.1, 2.0, 0.4], dtype=dtype, device=device),
        input_unit="photon/s/m2",
        output_unit="ADU",
        calibration_id="cal-v1",
        provenance_id="flat-v1",
        flat_convention="sky_flat",
        reference_convention="declared uniform spectral radiance v1",
    )
    response = NativeFieldResponse(
        quadrature=native,
        calibration=calibration,
        source_ra_deg=torch.tensor([100, 101], dtype=dtype, device=device),
        source_dec_deg=torch.tensor([2, 3], dtype=dtype, device=device),
        source_value_unit="photon/s/m2/Angstrom/sr",
        quadrature_output_unit="photon/s/m2",
        source_measure_kind="solid_angle_sr",
        spectral_measure_kind="density_per_angstrom",
        band_name="band",
        geometry_id="source-index-chart-v1",
        response_id="native-v1",
        dependencies=EvidenceSnapshot.capture({"response_matrix": matrix}),
    )
    observation = FieldObservation(
        image=torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=dtype, device=device),
        header={"NAXIS1": 2, "NAXIS2": 2},
        weight=torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=dtype, device=device),
        band_name="band",
        value_unit="ADU",
        value_semantics="surface_brightness",
        detector_law="gaussian",
        processing_state=AdditiveProcessingState.CALIBRATED_IMAGE,
    )
    return response, observation, matrix


def _dense(response, matrix):
    q = response.quadrature
    a = matrix.numpy() * q.source_measure.numpy()[None, :, None]
    a *= q.wavelength_measure.numpy()[None, None, :]
    a *= q.source_spectral_weight.numpy()[None, :, :]
    a *= response.calibration.scale.numpy()[:, None, None]
    if q.mask is not None:
        a *= q.mask.numpy()[:, None, None]
    return a.reshape(4, 4)


@pytest.mark.parametrize("chunk", [1, 3])
def test_source_nodes_context_spectrum_and_output_background_order(chunk):
    response, observation, matrix = _response()
    observation.background_offset = 0.5
    observation.sky_background = torch.full_like(observation.image, 0.7)
    observation.additive_components = (
        AdditiveComponent(
            torch.full_like(observation.image, -0.2),
            AdditiveStage.CALIBRATED_OUTPUT,
            AdditiveFrame.DETECTOR,
            "ADU",
            "surface_brightness",
            AdditiveProcessingState.CALIBRATED_IMAGE,
        ),
    )
    # Metadata must not apply a second bandpass/time/gain conversion.
    observation.exposure_time_s = 1000
    observation.collecting_area_m2 = 30
    observation.gain_electrons_per_adu = 7
    model = _Field()
    context = ForwardContext(time_mjd=1.0)
    result = render_native_field_observation(
        model, observation, response, forward_context=context, source_chunk_size=chunk
    )
    state = np.array([[1.5, 1.5], [2.6, 3.6]])
    expected = _dense(response, matrix) @ state.ravel() + 1.0
    np.testing.assert_allclose(result.detach().numpy().ravel(), expected, atol=1e-12)
    assert all(len(ra) <= chunk for ra, _ in model.calls)
    assert {wave for _, wave in model.calls} == {5000, 10000}
    assert all(float(ra.min()) >= 100 for ra, _ in model.calls)
    # A prediction must not depend on observed values, weights or data validity.
    observation.image.fill_(float("nan"))
    observation.weight.zero_()
    torch.testing.assert_close(
        render_native_field_observation(
            model, observation, response, forward_context=context
        ),
        result,
    )


@pytest.mark.parametrize("mask", [False, True])
def test_native_gaussian_evidence_matches_dense_gls_and_normalization(mask):
    response, observation, matrix = _response(mask=mask)
    observation.background_offset = 0.2
    observation.mask = torch.tensor([[True, True], [True, False]]) if mask else None
    factor = bind_native_field_likelihood(
        observation, response, measurement_ids=("native",), provenance_id="product-v1"
    )
    system = GaussianLikelihoodSystem((factor,))
    keep = np.array([True, not mask, True, not mask])
    a = _dense(response, matrix)[keep]
    weights = observation.weight.numpy().ravel()[keep]
    y = observation.image.numpy().ravel()[keep] - 0.2
    state = torch.tensor([[0.4, 0.2], [-0.1, 0.3]], dtype=torch.float64)
    h, b = a.T @ (weights[:, None] * a), a.T @ (weights * y)
    np.testing.assert_allclose(system.rhs().numpy().ravel(), b, atol=1e-12)
    np.testing.assert_allclose(
        system.information(state).numpy().ravel(), h @ state.numpy().ravel(), atol=1e-12
    )
    residual = y - a @ state.numpy().ravel()
    expected = 0.5 * (
        residual @ (weights * residual)
        + len(y) * np.log(2 * np.pi)
        - np.log(weights).sum()
    )
    np.testing.assert_allclose(
        system.negative_log_likelihood(state).numpy(), expected, atol=1e-12
    )
    if not mask:
        templates = torch.from_numpy(a.T.copy())
        query = fit_linear_query(system, (templates,))
        np.testing.assert_allclose(
            query.coefficients.numpy(), solve(h, b, assume_a="pos"), atol=1e-12
        )
        np.testing.assert_allclose(
            query.sampling_covariance.numpy(),
            solve(h, np.eye(4), assume_a="pos"),
            atol=1e-12,
        )


def test_exact_transpose_state_and_field_parameter_gradients():
    response, observation, matrix = _response()
    state = torch.tensor(
        [[0.2, 0.4], [-0.2, 0.7]], dtype=torch.float64, requires_grad=True
    )
    dual = torch.tensor([0.3, 0.1, -0.6, 0.9], dtype=torch.float64, requires_grad=True)
    expected = _dense(response, matrix)
    np.testing.assert_allclose(
        response.adjoint(dual).detach().numpy().ravel(),
        expected.T @ dual.detach().numpy(),
        atol=1e-12,
    )
    torch.testing.assert_close(
        (response.forward(state) * dual).sum(), (state * response.adjoint(dual)).sum()
    )
    assert torch.autograd.gradcheck(response.forward, (state,))
    assert torch.autograd.gradgradcheck(response.forward, (state,))
    assert torch.autograd.gradcheck(response.adjoint, (dual,))
    factor = bind_native_field_likelihood(
        observation, response, measurement_ids=("native",), provenance_id="v1"
    )
    system = GaussianLikelihoodSystem((factor,))
    direct = system.quadratic(state)
    normal = system.normal_equations().quadratic(state)
    torch.testing.assert_close(
        torch.autograd.grad(direct, state)[0], torch.autograd.grad(normal, state)[0]
    )
    field = _Field()
    rendered = render_native_field_observation(field, observation, response)
    grad = torch.autograd.grad(rendered.sum(), field.offset)[0]
    torch.testing.assert_close(grad, torch.tensor(expected.sum(), dtype=torch.float64))


@pytest.mark.parametrize(
    "changed",
    [
        "scale",
        "response",
        "source",
        "mask",
        "binding",
        "data",
        "weight",
        "header",
        "background",
    ],
)
def test_cached_evidence_rejects_response_and_measurement_changes(changed):
    response, observation, matrix = _response(mask=True)
    observation.detector_background = torch.zeros_like(observation.image)
    factor = bind_native_field_likelihood(
        observation, response, measurement_ids=("native",), provenance_id="v1"
    )
    equations = GaussianLikelihoodSystem((factor,)).normal_equations()
    if changed == "scale":
        response.calibration.scale.add_(0.1)
    elif changed == "response":
        matrix.add_(0.1)
    elif changed == "source":
        response.source_ra_deg.add_(0.01)
    elif changed == "mask":
        response.quadrature.mask[0] = False
    elif changed == "binding":
        response.quadrature.response = _Table(matrix.clone())
    elif changed == "data":
        observation.image.add_(0.1)
    elif changed == "weight":
        observation.weight.mul_(2)
    elif changed == "header":
        observation.header["OFFSET"] = 1
    else:
        observation.detector_background.add_(0.1)
    with pytest.raises(RuntimeError, match="stale evidence"):
        _ = equations.rhs


def test_unsupported_double_response_and_mutable_additive_units_fail_closed():
    response, observation, _ = _response()
    for bad in (
        replace(observation, psf_spec=PSFSpec.gaussian(0.5)),
        replace(observation, value_unit="electrons"),
        replace(observation, band_name="other"),
        replace(observation, detector_law="poisson_gaussian"),
        replace(observation, processing_state=AdditiveProcessingState.RAW_READS),
    ):
        with pytest.raises(ValueError):
            render_native_field_observation(_Field(), bad, response)
    observation.additive_components = (
        AdditiveComponent(
            torch.ones_like(observation.image),
            AdditiveStage.CALIBRATED_OUTPUT,
            AdditiveFrame.DETECTOR,
            "electron",
            "surface_brightness",
            AdditiveProcessingState.CALIBRATED_IMAGE,
        ),
    )
    with pytest.raises(ValueError, match="units"):
        render_native_field_observation(_Field(), observation, response)


def test_integrated_bins_are_direct_states_not_implicit_field_integrals():
    response, _, _ = _response()
    q = replace(
        response.quadrature, wavelength_measure=torch.ones(2, dtype=torch.float64)
    )
    direct = replace(response, quadrature=q, spectral_measure_kind="integrated_bins")
    with pytest.raises(ValueError, match="direct sampled states"):
        direct.sample_field(_Field())
    assert direct.forward(torch.ones(2, 2, dtype=torch.float64)).shape == (4,)
    with pytest.raises(ValueError, match="source measures one"):
        replace(response, source_measure_kind="point_flux")


def test_forward_and_adjoint_reject_inconsistent_prefixes():
    response, _, _ = _response()
    with pytest.raises(ValueError, match="shape"):
        response.forward(torch.ones(2, 2, 2, dtype=torch.float64))
    with pytest.raises(ValueError, match="shape"):
        response.adjoint(torch.ones(2, 4, dtype=torch.float64))


def test_final_rendering_rejects_combined_mean_overflow():
    response, observation, _ = _response(dtype=torch.float32)
    observation.background_offset = torch.finfo(torch.float32).max

    class Bright(nn.Module):
        def forward(self, ra, dec, wavelength):
            return torch.ones_like(ra) * 1e35

    with pytest.raises(ValueError, match="combined native prediction"):
        render_native_field_observation(Bright(), observation, response)


def test_ignored_conversion_metadata_still_invalidates_cached_provenance():
    response, observation, _ = _response()
    observation.exposure_time_s = 30
    factor = bind_native_field_likelihood(
        observation, response, measurement_ids=("native",), provenance_id="v1"
    )
    evidence = GaussianLikelihoodSystem((factor,)).normal_equations()
    observation.exposure_time_s = 60
    with pytest.raises(RuntimeError, match="stale evidence"):
        _ = evidence.data_quadratic


def test_different_instrument_addition_retains_full_spectral_information_and_lineage():
    response, observation, matrix = _response()
    blurred = matrix.mean(0, keepdim=True).expand_as(matrix).clone()
    poorer = replace(
        response,
        quadrature=replace(response.quadrature, response=_Table(blurred)),
        band_name="older-band",
        response_id="older-response",
        dependencies=EvidenceSnapshot.capture({"older-response": blurred}),
    )
    older = replace(
        observation, band_name="older-band", weight=observation.weight * 0.01
    )
    good = bind_native_field_likelihood(
        observation, response, measurement_ids=("good",), provenance_id="v1"
    )
    bad = bind_native_field_likelihood(
        older, poorer, measurement_ids=("older",), provenance_id="v1"
    )
    joint = GaussianLikelihoodSystem((good, bad))
    a, old_a = _dense(response, matrix), _dense(poorer, blurred)
    h_good = a.T @ (observation.weight.numpy().ravel()[:, None] * a)
    h_old = old_a.T @ (older.weight.numpy().ravel()[:, None] * old_a)
    h = np.column_stack(
        [
            joint.information(torch.from_numpy(column.reshape(2, 2))).numpy().ravel()
            for column in np.eye(4)
        ]
    )
    np.testing.assert_allclose(h, h_good + h_old, atol=1e-12)
    assert np.linalg.eigvalsh(h - h_good).min() > -1e-12
    assert np.any(np.abs(h - np.diag(np.diag(h))) > 0.01)
    with pytest.raises(ValueError, match="duplicate"):
        GaussianLikelihoodSystem((good, replace(bad, measurement_ids=("good",))))


class _ChromaticGaussian:
    def __call__(
        self, detector_xy, source_xy, wavelength_angstrom, *, detector_footprint
    ):
        sigma = (0.3 * wavelength_angstrom / 5000)[..., None]
        offset = detector_xy - source_xy
        low = (offset - 0.5) / (math.sqrt(2) * sigma)
        high = (offset + 0.5) / (math.sqrt(2) * sigma)
        return (0.5 * (torch.erf(high) - torch.erf(low))).prod(-1)


def test_extended_chromatic_field_matches_independent_continuous_pixel_integral():
    # Hermite nodes integrate the Gaussian sky; no detector-center rasterization.
    nodes, weights = np.polynomial.hermite.hermgauss(32)
    x, y = np.meshgrid(nodes * math.sqrt(2) * 0.4, nodes * math.sqrt(2) * 0.4)
    xy = np.stack([x.ravel(), y.ravel()], -1)
    # Unweighted source solid angles: undo the Hermite density weight. This
    # planar reference declares a constant chart Jacobian, not a spherical WCS.
    area_sr = 1e-10
    source_weights = (
        np.outer(weights, weights).ravel()
        * 2
        * 0.4**2
        * np.exp((xy**2).sum(-1) / (2 * 0.4**2))
        * area_sr
    )
    detector = np.array([[-1, 0], [0, 0], [1, 0]], dtype=float)
    q = NativeQuadratureResponseOp(
        response=_ChromaticGaussian(),
        source_xy=torch.from_numpy(xy),
        detector_xy=torch.from_numpy(detector),
        wavelength_angstrom=torch.tensor([5000, 10000], dtype=torch.float64),
        source_measure=torch.from_numpy(source_weights),
        wavelength_measure=torch.tensor([20, 30], dtype=torch.float64),
    )
    calibration = PixelCalibrationOp(
        torch.ones(3, dtype=torch.float64) * 50,
        input_unit="photon/s/m2",
        output_unit="electron",
        calibration_id="v1",
        provenance_id="v1",
        flat_convention="detector_counts",
        reference_convention="physical aperture area*time, no flat",
    )
    response = NativeFieldResponse(
        quadrature=q,
        calibration=calibration,
        source_ra_deg=torch.from_numpy(xy[:, 0]),
        source_dec_deg=torch.from_numpy(xy[:, 1]),
        source_value_unit="photon/s/m2/Angstrom/sr",
        quadrature_output_unit="photon/s/m2",
        source_measure_kind="solid_angle_sr",
        spectral_measure_kind="density_per_angstrom",
        band_name="band",
        geometry_id="local-node-chart",
        response_id="chromatic-reference",
        dependencies=EvidenceSnapshot.capture(),
    )

    class GaussianSky(nn.Module):
        def forward(self, ra, dec, wavelength):
            return (
                torch.exp(-(ra**2 + dec**2) / (2 * 0.4**2))
                / (2 * math.pi * 0.4**2 * area_sr)
                * (wavelength / 5000)
            )

    observed = response.forward(
        response.sample_field(GaussianSky(), source_chunk_size=13)
    )
    expected = np.zeros(3)
    for wavelength, measure in [(5000, 20), (10000, 30)]:
        sigma = np.sqrt(0.4**2 + (0.3 * wavelength / 5000) ** 2)
        fraction = (
            ndtr((detector + 0.5) / sigma) - ndtr((detector - 0.5) / sigma)
        ).prod(-1)
        expected += 50 * measure * (wavelength / 5000) * fraction
    np.testing.assert_allclose(observed.numpy(), expected, rtol=1e-7, atol=1e-6)


def test_existing_scalar_field_module_interoperability():
    response, observation, _ = _response()
    model = SineImplicitField(
        FieldDomain(100, 2, 1, 7500, 2500), hidden_dim=4, hidden_layers=1
    ).double()
    rendered = render_native_field_observation(
        model, observation, response, source_chunk_size=1
    )
    assert rendered.shape == observation.image.shape
    assert torch.isfinite(rendered).all()
    torch.autograd.grad(rendered.sum(), tuple(model.parameters()))


@pytest.mark.parametrize(
    "device,dtype",
    [("cpu", torch.float32), ("cuda", torch.float64), ("mps", torch.float32)],
)
def test_dtype_device_direct_state_contract(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    response, observation, _ = _response(dtype=dtype, device=device)
    state = torch.ones(2, 2, dtype=dtype, device=device)
    factor = bind_native_field_likelihood(
        observation, response, measurement_ids=("native",), provenance_id="v1"
    )
    system = GaussianLikelihoodSystem((factor,))
    torch.testing.assert_close(
        system.quadratic(state), system.normal_equations().quadratic(state)
    )
