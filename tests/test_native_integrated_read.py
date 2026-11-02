"""Independent bounded physical-stage, density and lifecycle client checks."""

from dataclasses import replace
import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import logsumexp
from scipy.stats import norm, poisson
import torch
from torch import nn

from torchsky.map.evidence import EvidenceSnapshot
from torchsky.map import field as field_api
from torchsky.map.field.native_integrated_read import (
    bind_native_integrated_read_likelihood,
)
from torchsky.map.field.native_response import NativeFieldResponse
from torchsky.map.field.types import FieldObservation, PSFSpec
from torchsky.map.operators.background import (
    AdditiveComponent,
    AdditiveFrame,
    AdditiveProcessingState,
    AdditiveStage,
)
from torchsky.map.operators.continuous_response import NativeQuadratureResponseOp
from torchsky.map.operators.detector_response import LinearDetectorResponse
from torchsky.map.operators.pixel_calibration import PixelCalibrationOp


class _Kernel:
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


def _component(values, stage):
    source = stage == AdditiveStage.PHOTON_ILLUMINATION
    unit, semantics, stochastic = {
        AdditiveStage.PHOTON_ILLUMINATION: (
            "photon/s/Angstrom",
            "density_per_angstrom",
            "photon_shot",
        ),
        AdditiveStage.DARK_CHARGE: ("electron/s", "electron_rate", "charge_shot"),
        AdditiveStage.READ_OFFSET: (
            "electron",
            "electron_offset",
            "deterministic_mean",
        ),
        AdditiveStage.CALIBRATED_OUTPUT: (
            "ADU",
            "single_integrated_read",
            "deterministic_mean",
        ),
    }[stage]
    return AdditiveComponent(
        values,
        stage,
        AdditiveFrame.SAMPLED_SKY if source else AdditiveFrame.DETECTOR,
        unit,
        semantics,
        AdditiveProcessingState.RAW_READS,
        stochastic,
    )


def _fixture(
    *, dtype=torch.float64, device="cpu", revision=None, mask=False, components=True
):
    def tensor(value):
        return torch.tensor(value, dtype=dtype, device=device)

    matrix = tensor([[[0.4, 0.1], [0.1, 0.3]], [[0.2, 0.4], [0.5, 0.2]]])
    q = NativeQuadratureResponseOp(
        response=_Kernel(matrix),
        source_xy=tensor([[0, 0], [1, 0]]),
        detector_xy=tensor([[0, 0], [1, 0]]),
        wavelength_angstrom=tensor([5000, 10000]),
        source_measure=tensor([1, 1]),
        wavelength_measure=tensor([1, 2]),
        source_chunk_size=1,
        detector_chunk_size=1,
        wavelength_chunk_size=1,
    )
    optical = PixelCalibrationOp(
        tensor([0.8, 0.7]),
        input_unit="photon/s",
        output_unit="photon/s",
        calibration_id="optical-probability-v1",
        provenance_id="routing-evidence",
        flat_convention="photon_routing_qe_excluded",
        reference_convention="disjoint incident-photon bins, fixed physical routing, QE excluded",
        dependencies=None
        if revision is None
        else EvidenceSnapshot.capture(revision=revision),
    )
    response = NativeFieldResponse(
        quadrature=q,
        calibration=optical,
        source_ra_deg=tensor([100, 101]),
        source_dec_deg=tensor([2, 3]),
        source_value_unit="photon/s/Angstrom",
        quadrature_output_unit="photon/s",
        source_measure_kind="point_flux",
        spectral_measure_kind="density_per_angstrom",
        band_name="native-band",
        geometry_id="source-chart-v1",
        response_id="optical-v1",
        dependencies=EvidenceSnapshot.capture({"kernel": matrix}, revision=revision),
    )
    output = PixelCalibrationOp(
        tensor([1.3, 0.7]),
        input_unit="electron",
        output_unit="ADU",
        calibration_id="electronic-output-v1",
        provenance_id="readout-calibration",
        flat_convention="detector_counts",
        reference_convention="single integrated read, fixed output units/electron",
        dependencies=None
        if revision is None
        else EvidenceSnapshot.capture(revision=revision),
    )
    image = tensor([[-0.4, 1.2]])
    valid = torch.tensor([[True, not mask]], device=device)
    read_var = tensor([0.7, 0.4])
    qe = tensor([0.7, 0.5])
    if mask:
        image[0, 1] = torch.nan
        read_var[1] = torch.nan
        qe[1] = torch.nan
    observation = FieldObservation(
        image=image,
        mask=valid,
        header={"NAXIS1": 2, "NAXIS2": 1},
        value_unit="ADU",
        value_semantics="single_integrated_read",
        band_name="native-band",
        detector_law="poisson_gaussian",
        processing_state=AdditiveProcessingState.RAW_READS,
    )
    parts = (
        ()
        if not components
        else (
            _component(
                tensor([[0.1, 0.2], [0.0, 0.3]]), AdditiveStage.PHOTON_ILLUMINATION
            ),
            _component(tensor([[0.2, 0.4]]), AdditiveStage.DARK_CHARGE),
            _component(tensor([[-0.1, 0.2]]), AdditiveStage.READ_OFFSET),
            _component(tensor([[-1.0, 0.1]]), AdditiveStage.CALIBRATED_OUTPUT),
        )
    )
    kwargs = dict(
        exposure_time_s=tensor(1.5),
        electrons_per_photon=qe,
        read_variance_electrons2=read_var,
        measurement_ids=("native-integrated-read",),
        calibration_id="physical-v1",
        provenance_id="native-lineage",
        additive_components=parts,
        revision=revision,
        tail_sigma=3.0,
        max_terms=128,
    )
    return observation, response, output, kwargs, matrix


def _bind(fixture):
    observation, response, output, kwargs, _ = fixture
    return bind_native_integrated_read_likelihood(
        observation, response, output, **kwargs
    )


def test_public_physical_binding_exports_are_actual_client():
    assert (
        field_api.bind_native_integrated_read_likelihood
        is bind_native_integrated_read_likelihood
    )
    fixture = _fixture()
    bound = field_api.bind_native_integrated_read_likelihood(*fixture[:3], **fixture[3])
    assert isinstance(bound, field_api.NativeIntegratedReadLikelihood)
    assert bool(torch.isfinite(bound.nll(torch.ones(2, 2, dtype=torch.float64))))


def _independent_parameters(fixture, state):
    observation, response, output, kwargs, matrix = fixture
    source = state.detach().cpu().numpy().copy()
    dark = np.zeros(2)
    read_offset = np.zeros(2)
    output_offset = np.zeros(2)
    for component in kwargs["additive_components"]:
        values = component.values.detach().cpu().numpy()
        if component.stage == AdditiveStage.PHOTON_ILLUMINATION:
            source += values
        elif component.stage == AdditiveStage.DARK_CHARGE:
            dark += values.ravel()
        elif component.stage == AdditiveStage.READ_OFFSET:
            read_offset += values.ravel()
        else:
            output_offset += values.ravel()
    photon_rate = (
        np.einsum(
            "pnl,nl,l->p", matrix.detach().cpu().numpy(), source, np.array([1, 2])
        )
        * response.calibration.scale.detach().cpu().numpy()
    )
    rate = kwargs["exposure_time_s"].item() * (
        kwargs["electrons_per_photon"].detach().cpu().numpy() * photon_rate + dark
    )
    gain = output.scale.detach().cpu().numpy()
    read_var = kwargs["read_variance_electrons2"].detach().cpu().numpy() * gain**2
    background = gain * read_offset + output_offset
    return rate, gain, read_var, background


def _scipy_nll(y, rate, gain, read_var, offset, tail_sigma=3.0):
    radius = tail_sigma * np.sqrt(rate + 1) + tail_sigma**2
    counts = np.arange(max(0, np.floor(rate - radius)), np.ceil(rate + radius) + 1)
    weights = poisson.logpmf(counts, rate)
    weights -= logsumexp(weights)
    return -logsumexp(
        weights + norm.logpdf(y, loc=gain * counts + offset, scale=np.sqrt(read_var))
    )


@pytest.mark.parametrize("mask", [False, True])
def test_real_source_node_client_matches_independent_stage_and_density_reference(mask):
    fixture = _fixture(mask=mask)
    bound = _bind(fixture)
    state = torch.tensor([[0.2, 0.3], [0.4, 0.1]], dtype=torch.float64)
    rate, gain, variance, offset = _independent_parameters(fixture, state)
    observation = fixture[0]
    support = observation.mask.numpy().ravel()
    expected = np.zeros(2)
    for pixel in np.flatnonzero(support):
        expected[pixel] = _scipy_nll(
            observation.image.numpy().ravel()[pixel],
            rate[pixel],
            gain[pixel],
            variance[pixel],
            offset[pixel],
        )
    torch.testing.assert_close(
        bound.nll(state, reduction="none"),
        torch.from_numpy(expected.reshape(1, 2)),
        atol=3e-14,
        rtol=2e-13,
    )
    torch.testing.assert_close(
        bound.nll(state), torch.tensor(expected.sum()), atol=3e-14, rtol=2e-13
    )
    torch.testing.assert_close(
        bound.nll(state, reduction="mean"),
        torch.tensor(expected[support].mean()),
        atol=3e-14,
        rtol=2e-13,
    )
    mean = np.where(support, gain * rate + offset, 0)
    torch.testing.assert_close(
        bound.expected_output(state), torch.from_numpy(mean.reshape(1, 2))
    )
    assert bound.measurement_ids == ("native-integrated-read",)
    # Each matched retained density integrates to one, including its readout
    # Jacobian; this is not an integral over a Gaussian moment approximation.
    for pixel in np.flatnonzero(support):

        def density(y):
            return np.exp(
                -_scipy_nll(y, rate[pixel], gain[pixel], variance[pixel], offset[pixel])
            )

        integral = quad(density, -np.inf, np.inf, epsabs=1e-10)[0]
        assert integral == pytest.approx(1, abs=2e-10)


def test_output_gain_density_jacobian_and_signed_offsets_preserve_shot_law():
    fixture = _fixture()
    bound = _bind(fixture)
    state = torch.tensor([[0.2, 0.3], [0.4, 0.1]], dtype=torch.float64)
    rate, gain, variance, offset = _independent_parameters(fixture, state)
    y = fixture[0].image.numpy().ravel()
    expected_electron = np.array(
        [
            _scipy_nll(
                (y[p] - offset[p]) / gain[p], rate[p], 1, variance[p] / gain[p] ** 2, 0
            )
            + np.log(gain[p])
            for p in range(2)
        ]
    )
    torch.testing.assert_close(
        bound.nll(state, reduction="none"),
        torch.from_numpy(expected_electron.reshape(1, 2)),
        atol=2e-14,
        rtol=2e-13,
    )
    assert not np.allclose(expected_electron, expected_electron - np.log(gain))
    assert bound.expected_output(state)[0, 0] < 0  # Signed detector output is valid.
    no_offsets = dict(
        fixture[3], additive_components=fixture[3]["additive_components"][:2]
    )
    without = bind_native_integrated_read_likelihood(*fixture[:3], **no_offsets)
    # Removing signed offsets changes output means, not source/dark arrival mean.
    parameters = without._law_parameters(state)
    torch.testing.assert_close(parameters[0], torch.from_numpy(rate))
    assert not torch.allclose(
        bound.expected_output(state), without.expected_output(state)
    )


def test_photon_illumination_passes_through_optics_but_dark_does_not():
    fixture = _fixture()
    state = torch.zeros(2, 2, dtype=torch.float64)
    bound = _bind(fixture)
    rate, _, _, _ = bound._law_parameters(state)
    kwargs = fixture[3]
    independent, _, _, _ = _independent_parameters(fixture, state)
    torch.testing.assert_close(rate, torch.from_numpy(independent))
    dark = kwargs["additive_components"][1].values.ravel()
    lamp = kwargs["additive_components"][0].values
    correct_photo = (
        fixture[1].forward(lamp)
        * kwargs["electrons_per_photon"]
        * kwargs["exposure_time_s"]
    )
    torch.testing.assert_close(rate - correct_photo, dark * kwargs["exposure_time_s"])
    wrong_dark = (
        fixture[1].forward(dark[:, None].expand(2, 2))
        * kwargs["electrons_per_photon"]
        * kwargs["exposure_time_s"]
    )
    assert not torch.allclose(rate, correct_photo + wrong_dark)
    deterministic = replace(
        kwargs["additive_components"][1], stochastic_contribution="deterministic_mean"
    )
    with pytest.raises(ValueError, match="stochastic"):
        bind_native_integrated_read_likelihood(
            *fixture[:3], **(kwargs | {"additive_components": (deterministic,)})
        )


class _PositiveField(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_amplitude = nn.Parameter(torch.tensor(-1.0, dtype=torch.float64))

    def forward(self, ra, dec, wavelength):
        return self.log_amplitude.exp() * (1 + 0.1 * (ra - 100)) * wavelength / 5000


def test_actual_monochromatic_field_client_needs_no_global_field_and_keeps_ad():
    fixture = _fixture()
    bound = _bind(fixture)
    field = _PositiveField()
    sampled = fixture[1].sample_field(field, source_chunk_size=1)
    direct = bound.nll(sampled)
    actual = bound.nll_field(field, source_chunk_size=1)
    torch.testing.assert_close(direct, actual)
    gradient = torch.autograd.grad(actual, field.log_amplitude, create_graph=True)[0]
    curvature = torch.autograd.grad(gradient, field.log_amplitude)[0]
    assert bool(torch.isfinite(gradient)) and bool(torch.isfinite(curvature))
    with pytest.raises(ValueError, match="nonnegative"):
        bound.nll(-sampled)


@pytest.mark.parametrize("mixture_chunk_size", [None, 1, 3])
def test_first_second_parameter_gradients_inside_declared_pg_domain(mixture_chunk_size):
    fixture = _fixture(components=False)
    state = torch.tensor(
        [[0.2, 0.3], [0.4, 0.1]], dtype=torch.float64, requires_grad=True
    )
    duration = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
    qe = fixture[3]["electrons_per_photon"].clone().requires_grad_()
    read = fixture[3]["read_variance_electrons2"].clone().requires_grad_()
    gain = fixture[2].scale.clone().requires_grad_()
    dark = torch.tensor([[0.2, 0.4]], dtype=torch.float64, requires_grad=True)
    offset = torch.tensor([[-0.1, 0.2]], dtype=torch.float64, requires_grad=True)

    def evaluate(x, t, q, r, g, d, b):
        output = replace(fixture[2], scale=g)
        kwargs = fixture[3] | dict(
            exposure_time_s=t,
            electrons_per_photon=q,
            read_variance_electrons2=r,
            additive_components=(
                _component(d, AdditiveStage.DARK_CHARGE),
                _component(b, AdditiveStage.READ_OFFSET),
            ),
        )
        return bind_native_integrated_read_likelihood(
            fixture[0], fixture[1], output, **kwargs
        ).nll(x, mixture_chunk_size=mixture_chunk_size)

    args = (state, duration, qe, read, gain, dark, offset)
    assert torch.autograd.gradcheck(evaluate, args, fast_mode=True)
    assert torch.autograd.gradgradcheck(evaluate, args, fast_mode=True)


@pytest.mark.parametrize(
    "changed",
    [
        "data",
        "mask",
        "source",
        "response",
        "response_binding",
        "photon_scale",
        "output_scale",
        "duration",
        "qe",
        "read_var",
        "component",
        "header",
        "psf_sigma",
        "noise_binding",
    ],
)
def test_original_evidence_and_bindings_reject_stale_physics(changed):
    fixture = _fixture()
    bound = _bind(fixture)
    observation, response, output, kwargs, matrix = fixture
    if changed == "data":
        observation.image.add_(0.1)
    elif changed == "mask":
        observation.mask[0, 1] = False
    elif changed == "source":
        response.source_ra_deg.add_(0.1)
    elif changed == "response":
        matrix.add_(0.01)
    elif changed == "response_binding":
        response.quadrature.response = _Kernel(matrix.clone())
    elif changed == "photon_scale":
        response.calibration.scale.add_(0.1)
    elif changed == "output_scale":
        output.scale.add_(0.1)
    elif changed == "duration":
        kwargs["exposure_time_s"].add_(0.1)
    elif changed == "qe":
        kwargs["electrons_per_photon"].add_(0.1)
    elif changed == "read_var":
        kwargs["read_variance_electrons2"].add_(0.1)
    elif changed == "component":
        kwargs["additive_components"][0].values.add_(0.1)
    elif changed == "header":
        observation.header["ORIGIN"] = "changed"
    elif changed == "psf_sigma":
        observation.psf_sigma_pix = 0.2
    else:
        observation.err = torch.ones_like(observation.image)
    with pytest.raises(RuntimeError, match="stale evidence"):
        bound.nll(torch.ones(2, 2, dtype=torch.float64))


def test_alias_and_inference_require_explicit_revision():
    revision = [0]
    fixture = _fixture(revision=lambda: revision[0])
    bound = _bind(fixture)
    fixture[3]["read_variance_electrons2"].numpy()[0] += 0.1
    revision[0] += 1
    with pytest.raises(RuntimeError, match="stale evidence revision"):
        bound.validate_evidence()
    with torch.inference_mode():
        fixture = _fixture(revision=lambda: revision[0])
        with pytest.raises(ValueError, match="explicit caller revision"):
            bind_native_integrated_read_likelihood(
                *fixture[:3], **(fixture[3] | {"revision": None})
            )
        bound = _bind(fixture)
        value = bound.nll(torch.ones(2, 2, dtype=torch.float64))
        assert bool(torch.isfinite(value))
    revision[0] += 1
    with pytest.raises(RuntimeError, match="stale evidence"):
        bound.validate_evidence()


@pytest.mark.parametrize("client", ["photon_response", "field"])
def test_mutation_during_actual_callback_rejects_stale_evidence(client):
    fixture = _fixture()
    observation, response, output, kwargs, matrix = fixture
    if client == "photon_response":

        class MutatingResponse(_Kernel):
            def __call__(self, *args, **call_kwargs):
                kwargs["electrons_per_photon"].add_(0.01)
                return super().__call__(*args, **call_kwargs)

        response = replace(
            response,
            quadrature=replace(response.quadrature, response=MutatingResponse(matrix)),
        )
    bound = bind_native_integrated_read_likelihood(
        observation, response, output, **kwargs
    )
    if client == "field":

        class MutatingField(nn.Module):
            def forward(self, ra, dec, wavelength):
                observation.image.add_(0.01)
                return torch.ones_like(ra)

        with pytest.raises(RuntimeError, match="stale evidence"):
            bound.nll_field(MutatingField())
    else:
        with pytest.raises(RuntimeError, match="stale evidence"):
            bound.nll(torch.ones(2, 2, dtype=torch.float64))


def test_masks_units_processing_and_unsupported_noise_physics_fail_closed():
    fixture = _fixture()
    observation, response, output, kwargs, _ = fixture
    for change in (
        {"processing_state": AdditiveProcessingState.CALIBRATED_IMAGE},
        {"detector_law": "gaussian"},
        {"value_semantics": "surface_brightness"},
        {"weight": torch.ones_like(observation.image)},
        {"var_rnoise": torch.ones_like(observation.image)},
        {"background_offset": 1.0},
        {"psf_spec": PSFSpec.gaussian(0.3)},
        {"exposure_time_s": 2.0},
        {"gain_electrons_per_adu": 2.0},
        {"mask": torch.zeros_like(observation.mask)},
    ):
        with pytest.raises(ValueError):
            bind_native_integrated_read_likelihood(
                replace(observation, **change), response, output, **kwargs
            )
    identity = LinearDetectorResponse(lambda x: x, lambda x: x)
    for name in ("charge_response", "read_noise_response"):
        with pytest.raises(ValueError, match="mixing"):
            bind_native_integrated_read_likelihood(
                *fixture[:3], **(kwargs | {name: identity})
            )
    with pytest.raises(ValueError, match="units"):
        bind_native_integrated_read_likelihood(
            observation, response, replace(output, input_unit="photon/s"), **kwargs
        )
    wrong_optical = replace(
        response, calibration=replace(response.calibration, flat_convention="sky_flat")
    )
    with pytest.raises(ValueError, match="QE excluded"):
        bind_native_integrated_read_likelihood(
            observation, wrong_optical, output, **kwargs
        )
    # Physical stages remain opt-in; they do not widen legacy observation paths.
    with pytest.raises(ValueError, match="unsupported physical additive"):
        replace(observation, additive_components=(kwargs["additive_components"][0],))
    for change in (
        {"coordinate_frame": AdditiveFrame.DETECTOR},
        {"value_unit": "electron"},
        {"value_semantics": "unknown"},
        {"stochastic_contribution": "deterministic_mean"},
    ):
        component = replace(kwargs["additive_components"][0], **change)
        with pytest.raises(ValueError, match="mismatch"):
            bind_native_integrated_read_likelihood(
                *fixture[:3], **(kwargs | {"additive_components": (component,)})
            )


def test_mutated_observation_image_must_still_be_native_2d_at_bind():
    fixture = _fixture()
    observation = fixture[0]
    observation.image = observation.image.reshape(-1)
    observation.mask = None
    with pytest.raises(ValueError, match="2D"):
        bind_native_integrated_read_likelihood(*fixture[:3], **fixture[3])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_dtype_device_and_zero_source_law(dtype, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if device == "mps" and (
        not torch.backends.mps.is_available() or dtype == torch.float64
    ):
        pytest.skip("MPS unavailable or float64 unsupported")
    fixture = _fixture(dtype=dtype, device=device, components=False)
    bound = _bind(fixture)
    state = torch.zeros(2, 2, dtype=dtype, device=device, requires_grad=True)
    actual = bound.nll(state)
    gain = fixture[2].scale
    variance = fixture[3]["read_variance_electrons2"] * gain.square()
    observed = fixture[0].image.reshape(-1)
    expected = (
        0.5
        * (observed.square() / variance + variance.log() + math.log(2 * math.pi)).sum()
    )
    torch.testing.assert_close(actual, expected)
    gradient = torch.autograd.grad(actual, state, create_graph=True)[0]
    curvature = torch.autograd.grad(gradient.sum(), state)[0]
    assert bool(torch.isfinite(gradient).all()) and bool(
        torch.isfinite(curvature).all()
    )
    assert actual.dtype == dtype and actual.device.type == device


@pytest.mark.parametrize("gain,variance", [(1e20, 1e-20), (1e-20, 1e38)])
def test_finite_read_variance_survives_extreme_output_unit_scale(gain, variance):
    fixture = _fixture(dtype=torch.float32, components=False)
    output = replace(fixture[2], scale=torch.full((2,), gain, dtype=torch.float32))
    read_var = torch.tensor(variance, dtype=torch.float32)
    bound = bind_native_integrated_read_likelihood(
        fixture[0],
        fixture[1],
        output,
        **(fixture[3] | {"read_variance_electrons2": read_var}),
    )
    state = torch.zeros(2, 2, dtype=torch.float32)
    expected_variance = read_var.double() * output.scale.double().square()
    actual = bound._law_parameters(state)[2]
    torch.testing.assert_close(actual.double(), expected_variance, rtol=2e-7, atol=0)
    with torch.no_grad():
        per_pixel = bound.nll(state, reduction="none")
    y = fixture[0].image.double().reshape(-1)
    expected = 0.5 * (
        y.square() / expected_variance + expected_variance.log() + math.log(2 * math.pi)
    )
    torch.testing.assert_close(
        per_pixel.double().reshape(-1), expected, rtol=3e-7, atol=1e-6
    )
    torch.testing.assert_close(
        bound.expected_output(state), torch.zeros_like(fixture[0].image)
    )


def test_parameter_shapes_and_signs_fail_closed():
    fixture = _fixture()
    kwargs = fixture[3]
    for change, message in (
        (
            {"exposure_time_s": torch.tensor(0.0, dtype=torch.float64)},
            "positive scalar",
        ),
        (
            {"electrons_per_photon": torch.tensor(1.1, dtype=torch.float64)},
            "probabilities",
        ),
        (
            {"read_variance_electrons2": torch.tensor(0.0, dtype=torch.float64)},
            "strictly positive",
        ),
        (
            {"read_variance_electrons2": torch.ones(1, 2, dtype=torch.float64)},
            "explicit",
        ),
        ({"read_variance_electrons2": torch.ones(2, dtype=torch.float32)}, "dtype"),
    ):
        with pytest.raises(ValueError, match=message):
            bind_native_integrated_read_likelihood(*fixture[:3], **(kwargs | change))
    dark = replace(
        kwargs["additive_components"][1],
        values=-kwargs["additive_components"][1].values,
    )
    with pytest.raises(ValueError, match="nonnegative"):
        bind_native_integrated_read_likelihood(
            *fixture[:3], **(kwargs | {"additive_components": (dark,)})
        )
    wrong_shape = replace(
        kwargs["additive_components"][1], values=torch.zeros(2, 2, dtype=torch.float64)
    )
    with pytest.raises(ValueError, match="shape"):
        bind_native_integrated_read_likelihood(
            *fixture[:3], **(kwargs | {"additive_components": (wrong_shape,)})
        )


def test_real_finite_pixel_gaussian_limit_rejects_sum_and_mean_overflow():
    fixture = _fixture(dtype=torch.float32, components=False)
    observation, response, output, kwargs, matrix = fixture
    # Four independent native pixels retain a substochastic optical reference.
    # Source/dark mean is zero, so this is the real PG law's Gaussian limit;
    # value-only tensors do not request unsupported enormous rate derivatives.
    four_rows = matrix.repeat(2, 1, 1) * 0.5
    detector_xy = torch.stack(
        (torch.arange(4, dtype=torch.float32), torch.zeros(4)), dim=-1
    )
    response = replace(
        response,
        quadrature=replace(
            response.quadrature, response=_Kernel(four_rows), detector_xy=detector_xy
        ),
        calibration=replace(
            response.calibration, scale=response.calibration.scale.repeat(2)
        ),
        dependencies=EvidenceSnapshot.capture({"kernel": four_rows}),
    )
    observation = replace(
        observation,
        image=torch.full((1, 4), 1.4142135e19, dtype=torch.float32),
        mask=torch.ones(1, 4, dtype=torch.bool),
        header={"NAXIS1": 4, "NAXIS2": 1},
    )
    output = replace(output, scale=torch.ones(4, dtype=torch.float32))
    kwargs = kwargs | dict(
        exposure_time_s=torch.tensor(1.0),
        electrons_per_photon=torch.tensor(1.0),
        read_variance_electrons2=torch.tensor(1.0),
    )
    bound = bind_native_integrated_read_likelihood(
        observation, response, output, **kwargs
    )
    state = torch.zeros(2, 2, dtype=torch.float32)
    per_pixel = bound.nll(state, reduction="none")
    assert bool(torch.isfinite(per_pixel).all())
    # Independent Float64 expression verifies the finite individual log density.
    expected = 0.5 * (observation.image.double().square() + math.log(2 * math.pi))
    torch.testing.assert_close(per_pixel.double(), expected, rtol=1e-7, atol=0)
    for reduction in ("sum", "mean"):
        with pytest.raises(ValueError, match="destination dtype range"):
            bound.nll(state, reduction=reduction)


@pytest.mark.parametrize("reduction", ["sum", "mean", "none"])
@pytest.mark.parametrize("mask", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 2, 7])
def test_mixture_chunks_preserve_density_masks_and_gradient(
    reduction, mask, chunk_size
):
    bound = _bind(_fixture(mask=mask))
    state = torch.tensor(
        [[0.2, 0.3], [0.4, 0.1]], dtype=torch.float64, requires_grad=True
    )
    full = bound.nll(state, reduction=reduction)
    chunked = bound.nll(state, reduction=reduction, mixture_chunk_size=chunk_size)
    torch.testing.assert_close(chunked, full, rtol=1e-13, atol=1e-14)
    expected = torch.autograd.grad(full.sum(), state, create_graph=True)[0]
    actual = torch.autograd.grad(chunked.sum(), state, create_graph=True)[0]
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-14)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), state)[0],
        torch.autograd.grad(expected.sum(), state)[0],
        rtol=1e-11,
        atol=1e-13,
    )
    if reduction == "none" and mask:
        assert chunked[0, 1] == 0


def test_chunked_field_client_preserves_parameter_curvature():
    bound = _bind(_fixture())
    model = _PositiveField()
    expected = bound.nll_field(model)
    actual = bound.nll_field(model, source_chunk_size=1, mixture_chunk_size=1)
    torch.testing.assert_close(actual, expected)
    g_full = torch.autograd.grad(expected, model.log_amplitude, create_graph=True)[0]
    g_chunk = torch.autograd.grad(actual, model.log_amplitude, create_graph=True)[0]
    torch.testing.assert_close(g_chunk, g_full)
    torch.testing.assert_close(
        torch.autograd.grad(g_chunk, model.log_amplitude)[0],
        torch.autograd.grad(g_full, model.log_amplitude)[0],
    )


@pytest.mark.parametrize("chunk_size", [True, False, 0, -1, 1.5, "2"])
def test_invalid_mixture_chunk_size_rejects(chunk_size):
    bound = _bind(_fixture())
    with pytest.raises(ValueError, match="mixture_chunk_size"):
        bound.nll(torch.ones(2, 2, dtype=torch.float64), mixture_chunk_size=chunk_size)


@pytest.mark.parametrize("mutate", ["image", "read_var", "revision"])
def test_checkpoint_recomputation_rejects_changed_evidence(mutate):
    token = [0]
    fixture = _fixture(revision=lambda: token[0])
    bound = _bind(fixture)
    state = torch.ones(2, 2, dtype=torch.float64, requires_grad=True)
    loss = bound.nll(state, mixture_chunk_size=1)
    if mutate == "image":
        fixture[0].image.add_(0.1)
    elif mutate == "read_var":
        fixture[3]["read_variance_electrons2"].add_(0.1)
    else:
        token[0] += 1
    with pytest.raises(RuntimeError, match="stale"):
        torch.autograd.grad(loss, state)


def test_checkpoint_post_call_guard_rejects_mutation_during_recomputation(monkeypatch):
    from torchsky.map.field import native_integrated_read as module

    token = [0]
    bound = _bind(_fixture(revision=lambda: token[0]))
    state = torch.ones(2, 2, dtype=torch.float64, requires_grad=True)
    original = module.poisson_gaussian_nll
    calls = [0]

    def mutate_during_recompute(*args, **kwargs):
        calls[0] += 1
        try:
            return original(*args, **kwargs)
        finally:
            if calls[0] > 2:
                token[0] += 1

    monkeypatch.setattr(module, "poisson_gaussian_nll", mutate_during_recompute)
    loss = bound.nll(state, mixture_chunk_size=1)
    assert calls[0] == 2
    with pytest.raises(RuntimeError, match="stale"):
        torch.autograd.grad(loss, state)
    assert token[0] > 0


def test_chunked_forward_ad_and_combined_mode_contract():
    bound = _bind(_fixture())
    state = torch.ones(2, 2, dtype=torch.float64)
    tangent = torch.tensor([[0.1, 0.2], [-0.2, 0.3]], dtype=torch.float64)
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(state, tangent)
        expected = torch.autograd.forward_ad.unpack_dual(bound.nll(dual))
        actual = torch.autograd.forward_ad.unpack_dual(
            bound.nll(dual, mixture_chunk_size=1)
        )
        torch.testing.assert_close(actual.primal, expected.primal)
        torch.testing.assert_close(actual.tangent, expected.tangent)
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(state.requires_grad_(), tangent)
        expected = torch.autograd.forward_ad.unpack_dual(bound.nll(dual))
        actual = torch.autograd.forward_ad.unpack_dual(
            bound.nll(dual, mixture_chunk_size=1)
        )
        torch.testing.assert_close(actual.primal, expected.primal)
        torch.testing.assert_close(actual.tangent, expected.tangent)
        torch.testing.assert_close(
            torch.autograd.grad(actual.tangent, state)[0],
            torch.autograd.grad(expected.tangent, state)[0],
        )
    expected = torch.func.jvp(lambda x: bound.nll(x), (state,), (tangent,))
    actual = torch.func.jvp(
        lambda x: bound.nll(x, mixture_chunk_size=1), (state,), (tangent,)
    )
    torch.testing.assert_close(actual, expected)


def _many_pixel_fixture():
    observation, response, output, kwargs, matrix = _fixture(components=False)
    rows = 64
    kernel = matrix.repeat(rows // 2, 1, 1) * (2 / rows)
    xy = torch.stack(
        (torch.arange(rows, dtype=torch.float64), torch.zeros(rows)), dim=-1
    )
    response = replace(
        response,
        quadrature=replace(
            response.quadrature, response=_Kernel(kernel), detector_xy=xy
        ),
        calibration=replace(
            response.calibration, scale=response.calibration.scale.repeat(rows // 2)
        ),
        dependencies=EvidenceSnapshot.capture({"kernel": kernel}),
    )
    observation = replace(
        observation,
        image=torch.linspace(-1, 5, rows, dtype=torch.float64).reshape(1, -1),
        mask=torch.ones(1, rows, dtype=torch.bool),
        header={"NAXIS1": rows, "NAXIS2": 1},
    )
    output = replace(output, scale=output.scale.repeat(rows // 2))
    kwargs = kwargs | dict(
        electrons_per_photon=torch.linspace(0, 1, rows, dtype=torch.float64),
        read_variance_electrons2=kwargs["read_variance_electrons2"].repeat(rows // 2),
        additive_components=(
            _component(
                torch.linspace(0, 25, rows, dtype=torch.float64).reshape(1, -1),
                AdditiveStage.DARK_CHARGE,
            ),
        ),
    )
    return observation, response, output, kwargs, kernel


def test_checkpoint_bounds_mixture_calls_and_saved_tensors(monkeypatch):
    from torchsky.map.field import native_integrated_read as module

    bound = _bind(_many_pixel_fixture())
    state = torch.ones(2, 2, dtype=torch.float64, requires_grad=True)
    call_sizes = []
    original = module.poisson_gaussian_nll

    def record_call(y, *args, **kwargs):
        call_sizes.append(y.numel())
        return original(y, *args, **kwargs)

    monkeypatch.setattr(module, "poisson_gaussian_nll", record_call)

    def capture(chunk_size):
        shapes = []

        def pack(tensor):
            shapes.append(tuple(tensor.shape))
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            loss = bound.nll(state, mixture_chunk_size=chunk_size)
        return loss, shapes

    full, full_shapes = capture(None)
    full_sizes = call_sizes.copy()
    call_sizes.clear()
    chunked, chunk_shapes = capture(7)
    assert full_sizes == [64]
    assert call_sizes == [7] * 9 + [1]
    # The streaming Poisson--Gaussian Function saves O(pixels) moments, never
    # (pixel, count) tensors, with or without outer chunking.
    assert not any(len(shape) == 2 and shape[1] > 12 for shape in full_shapes)
    # Reverse-mode checkpoints save only row inputs, never mixture count axes.
    assert not any(len(shape) == 2 and shape[1] > 12 for shape in chunk_shapes)
    torch.testing.assert_close(chunked, full, rtol=1e-13, atol=1e-13)
    call_sizes.clear()
    actual = torch.autograd.grad(chunked, state)[0]
    assert len(call_sizes) == 10 and max(call_sizes) == 7
    expected = torch.autograd.grad(full, state)[0]
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_chunked_zero_rate_and_device_contract(dtype, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if device == "mps" and (
        not torch.backends.mps.is_available() or dtype == torch.float64
    ):
        pytest.skip("MPS unavailable or float64 unsupported")
    bound = _bind(_fixture(dtype=dtype, device=device, components=False))
    state = torch.zeros(2, 2, dtype=dtype, device=device, requires_grad=True)
    full = bound.nll(state)
    chunked = bound.nll(state, mixture_chunk_size=1)
    torch.testing.assert_close(chunked, full)
    full_grad = torch.autograd.grad(full, state, create_graph=True)[0]
    chunk_grad = torch.autograd.grad(chunked, state, create_graph=True)[0]
    torch.testing.assert_close(chunk_grad, full_grad)
    torch.testing.assert_close(
        torch.autograd.grad(chunk_grad.sum(), state)[0],
        torch.autograd.grad(full_grad.sum(), state)[0],
    )


@pytest.mark.parametrize("chunk_size", [1, 2, 7])
def test_chunking_preserves_derivative_range_rejection_and_value_only_law(chunk_size):
    fixture = _fixture(components=False)
    fixture[0].image.fill_(1000)
    bound = _bind(fixture)
    state = torch.zeros(2, 2, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        torch.testing.assert_close(
            bound.nll(state, mixture_chunk_size=chunk_size), bound.nll(state)
        )
    for size in (None, chunk_size):
        with pytest.raises(ValueError, match="zero-rate first/second derivatives"):
            bound.nll(state, mixture_chunk_size=size)


@pytest.mark.parametrize("device", ["cuda", "mps"])
@pytest.mark.parametrize("value", [0.0, 0.4])
def test_checkpointed_device_values_gradients_and_curvature_match_cpu(device, value):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")

    def evaluate(target):
        bound = _bind(_fixture(dtype=torch.float32, device=target))
        state = torch.full((2, 2), value, device=target, requires_grad=True)
        nll = bound.nll(state, mixture_chunk_size=1)
        gradient = torch.autograd.grad(nll, state, create_graph=True)[0]
        curvature = torch.autograd.grad(gradient.sum(), state)[0]
        return tuple(t.detach().cpu() for t in (nll, gradient, curvature))

    torch.testing.assert_close(evaluate(device), evaluate("cpu"), rtol=3e-5, atol=3e-6)
