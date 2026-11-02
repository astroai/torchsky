"""Small independent multi-product GLS and shared-nuisance references."""

from dataclasses import replace
import math

import numpy as np
import pytest
from scipy.linalg import solve
import torch

from torchsky.map.evidence import EvidenceSnapshot
from torchsky.map.field.joint_native_likelihood import bind_joint_native_likelihood
from torchsky.map.field.native_response import NativeFieldResponse
from torchsky.map.field.types import FieldObservation
from torchsky.map.likelihood import GaussianLikelihoodSystem
from torchsky.map.operators.background import AdditiveProcessingState
from torchsky.map.operators.continuous_response import NativeQuadratureResponseOp
from torchsky.map.operators.pixel_calibration import PixelCalibrationOp
from torchsky.map.query import fit_linear_query


class _Table:
    def __init__(self, matrix, chart_offset):
        self.matrix = matrix
        self.chart_offset = chart_offset

    def __call__(
        self, detector_xy, source_xy, wavelength_angstrom, *, detector_footprint
    ):
        return self.matrix[
            (detector_xy[..., 0] - 2 * self.chart_offset).long(),
            (source_xy[..., 0] - self.chart_offset).long(),
            (wavelength_angstrom / 5000 - 1).long(),
        ]


def _fixture(*, dtype=torch.float64, device="cpu", masked=False, revision=None):
    raw = (
        np.array(
            [
                [[0.8, 0.1], [0.1, 0.2]],
                [[0.2, 0.8], [0.3, 0.1]],
                [[0.1, 0.2], [0.8, 0.4]],
            ]
        ),
        np.array(
            [
                [[0.4, 0.2], [0.1, 0.8]],
                [[0.2, 0.4], [0.7, 0.1]],
                [[0.6, 0.1], [0.1, 0.3]],
                [[0.1, 0.7], [0.4, 0.1]],
            ]
        ),
    )
    responses, observations, matrices, bases, dense = [], [], [], [], []
    for index, matrix_np in enumerate(raw):
        matrix = torch.tensor(matrix_np, dtype=dtype, device=device)
        matrices.append(matrix)
        pixels = matrix.shape[0]
        chart_offset = index * 10
        q = NativeQuadratureResponseOp(
            response=_Table(matrix, chart_offset),
            source_xy=torch.tensor(
                [[chart_offset, 0], [chart_offset + 1, 0]], dtype=dtype, device=device
            ),
            detector_xy=torch.stack(
                (
                    torch.arange(pixels, dtype=dtype, device=device) + 2 * chart_offset,
                    torch.zeros(pixels, dtype=dtype, device=device),
                ),
                dim=-1,
            ),
            wavelength_angstrom=torch.tensor([5000, 10000], dtype=dtype, device=device),
            source_measure=torch.tensor([0.2, 0.5], dtype=dtype, device=device),
            wavelength_measure=torch.tensor([10, 20], dtype=dtype, device=device),
            source_spectral_weight=torch.tensor(
                [[0.8, 0.6], [0.4, 0.9]], dtype=dtype, device=device
            )
            * (1 - 0.2 * index),
            source_chunk_size=1,
            detector_chunk_size=2,
            wavelength_chunk_size=1,
            mask=(torch.arange(pixels, device=device) != 1)
            if masked and index == 1
            else None,
        )
        scale = torch.arange(1, pixels + 1, dtype=dtype, device=device) * (
            0.4 + 0.2 * index
        )
        unit = "electron" if index == 0 else "ADU"
        calibration = PixelCalibrationOp(
            scale,
            input_unit="photon/s/m2",
            output_unit=unit,
            calibration_id=f"instrument-{index}-flat",
            provenance_id=f"flat-evidence-{index}",
            flat_convention="sky_flat",
            reference_convention="declared uniform spectral radiance",
            dependencies=None
            if revision is None
            else EvidenceSnapshot.capture(revision=revision),
        )
        response = NativeFieldResponse(
            quadrature=q,
            calibration=calibration,
            source_ra_deg=torch.tensor([100, 101], dtype=dtype, device=device),
            source_dec_deg=torch.tensor([2, 3], dtype=dtype, device=device),
            source_value_unit="photon/s/m2/Angstrom/sr",
            quadrature_output_unit="photon/s/m2",
            source_measure_kind="solid_angle_sr",
            spectral_measure_kind="density_per_angstrom",
            band_name=f"band-{index}",
            geometry_id=f"chart-{index}",
            response_id=f"response-{index}",
            dependencies=EvidenceSnapshot.capture(
                {"response_matrix": matrix}, revision=revision
            ),
        )
        responses.append(response)
        # Build the independent dense reference directly from declared factors.
        a = matrix_np * np.array([0.2, 0.5])[None, :, None]
        a *= np.array([10, 20])[None, None, :]
        a *= np.array([[0.8, 0.6], [0.4, 0.9]])[None] * (1 - 0.2 * index)
        a *= np.arange(1, pixels + 1)[:, None, None] * (0.4 + 0.2 * index)
        dense.append(a.reshape(pixels, 4))
        basis = torch.stack(
            (
                torch.ones(pixels, dtype=dtype, device=device) * (1 + index),
                torch.arange(pixels, dtype=dtype, device=device) * (0.2 + 0.1 * index)
                - 0.3,
            ),
            dim=-1,
        )
        bases.append(basis)
        values = (
            a.reshape(pixels, 4) @ np.array([0.4, 0.9, 0.3, 0.8])
            + np.arange(pixels) * 0.1
            - 0.1
        )
        image = (
            torch.tensor(values, dtype=dtype, device=device).reshape(1, pixels) + 0.7
        )
        weight = torch.arange(1, pixels + 1, dtype=dtype, device=device).reshape(
            1, pixels
        ) * (0.8 + index)
        mask = torch.ones_like(image, dtype=torch.bool)
        if masked:
            mask[0, -1] = False
            image[0, -1] = torch.nan
            weight[0, -1] = torch.nan
            basis[-1] = torch.nan
        observations.append(
            FieldObservation(
                image=image,
                weight=weight,
                mask=mask,
                header={"NAXIS1": pixels, "NAXIS2": 1},
                band_name=response.band_name,
                value_unit=unit,
                value_semantics="surface_brightness",
                detector_law="gaussian",
                processing_state=AdditiveProcessingState.CALIBRATED_IMAGE,
                background_offset=0.7,
            )
        )
    return (
        tuple(observations),
        tuple(responses),
        tuple(bases),
        tuple(matrices),
        tuple(dense),
    )


def _bind(
    observations,
    responses,
    bases,
    *,
    treatment="calibrated",
    mean=None,
    factor=None,
    revision=None,
    **kwargs,
):
    if treatment == "calibrated" and factor is None:
        factor = torch.tensor(
            [[0.5, 0.0], [0.1, 0.3]],
            dtype=observations[0].image.dtype,
            device=observations[0].image.device,
        )
    return bind_joint_native_likelihood(
        observations,
        responses,
        measurement_ids=tuple(
            (f"native-{index}",) for index in range(len(observations))
        ),
        nuisance_bases=bases,
        nuisance_basis_units=tuple(
            observation.value_unit for observation in observations
        ),
        nuisance_ids=("shared-offset", "shared-drift"),
        nuisance_treatment=treatment,
        calibration_id="joint-calibration-v1",
        provenance_id="joint-native-v1",
        calibrated_mean=mean,
        covariance_factor=factor,
        revision=revision,
        **kwargs,
    )


@pytest.mark.parametrize("masked", [False, True])
def test_calibrated_multi_instrument_gls_matches_independent_scipy(masked):
    observations, responses, bases, _, dense = _fixture(masked=masked)
    mean = torch.tensor([0.15, -0.2], dtype=torch.float64)
    factor = torch.tensor([[0.5, 0.0], [0.1, 0.3]], dtype=torch.float64)
    joint = _bind(observations, responses, bases, mean=mean, factor=factor)
    system = GaussianLikelihoodSystem((joint,))
    valid = np.concatenate(
        [
            observation.mask.numpy().ravel()
            & (
                np.ones(observation.image.numel(), dtype=bool)
                if response.quadrature.mask is None
                else response.quadrature.mask.numpy()
            )
            for observation, response in zip(observations, responses, strict=True)
        ]
    )
    a = np.concatenate(dense)[valid]
    b = np.concatenate([basis.numpy() for basis in bases])[valid]
    y = (
        np.concatenate(
            [(observation.image.numpy() - 0.7).ravel() for observation in observations]
        )[valid]
        - b @ mean.numpy()
    )
    weights = np.concatenate(
        [observation.weight.numpy().ravel() for observation in observations]
    )[valid]
    c = np.diag(1 / weights) + b @ factor.numpy() @ factor.numpy().T @ b.T
    precision_a = solve(c, a, assume_a="pos")
    information = a.T @ precision_a
    rhs = a.T @ solve(c, y, assume_a="pos")
    x = np.array([0.2, 0.6, -0.1, 0.3])
    residual = y - a @ x
    expected_q = residual @ solve(c, residual, assume_a="pos")
    state = torch.from_numpy(x.reshape(2, 2))
    torch.testing.assert_close(
        system.quadratic(state), torch.tensor(expected_q), atol=3e-12, rtol=2e-12
    )
    equations = system.normal_equations()
    torch.testing.assert_close(
        equations.rhs, torch.from_numpy(rhs.reshape(2, 2)), atol=3e-12, rtol=2e-12
    )
    torch.testing.assert_close(
        equations.information(state),
        torch.from_numpy((information @ x).reshape(2, 2)),
        atol=3e-12,
        rtol=2e-12,
    )
    normalization = 0.5 * (len(y) * math.log(2 * math.pi) + np.linalg.slogdet(c)[1])
    torch.testing.assert_close(
        equations.log_normalization, torch.tensor(normalization), atol=2e-12, rtol=2e-12
    )
    torch.testing.assert_close(
        system.negative_log_likelihood(state),
        torch.tensor(expected_q * 0.5 + normalization),
        atol=4e-12,
        rtol=2e-12,
    )
    full_a = np.concatenate(dense)
    templates = (torch.from_numpy(full_a.T.copy()),)
    query = fit_linear_query(system, templates)
    torch.testing.assert_close(
        query.coefficients,
        torch.from_numpy(solve(information, rhs, assume_a="pos")),
        atol=3e-11,
        rtol=2e-11,
    )
    torch.testing.assert_close(
        query.sampling_covariance,
        torch.from_numpy(solve(information, np.eye(4), assume_a="pos")),
        atol=3e-10,
        rtol=2e-10,
    )
    assert joint.measurement_ids == ("native-0", "native-1")
    assert joint.eliminated_nuisance_ids == ("shared-offset", "shared-drift")
    predicted = joint.operator.apply(state)
    direction = torch.arange(predicted.numel(), dtype=torch.float64) / 10
    torch.testing.assert_close(
        (predicted * direction).sum(), (state * joint.operator.adjoint(direction)).sum()
    )


def test_profiled_joint_span_matches_dense_gls_and_has_no_normalized_density():
    observations, responses, bases, _, dense = _fixture()
    joint = _bind(observations, responses, bases, treatment="profiled")
    a = np.concatenate(dense)
    b = np.concatenate([basis.numpy() for basis in bases])
    y = np.concatenate(
        [(observation.image.numpy() - 0.7).ravel() for observation in observations]
    )
    weights = np.concatenate(
        [observation.weight.numpy().ravel() for observation in observations]
    )
    state = torch.tensor([[0.2, 0.6], [-0.1, 0.3]], dtype=torch.float64)
    residual = y - a @ state.numpy().ravel()
    beta = solve(
        b.T @ (weights[:, None] * b), b.T @ (weights * residual), assume_a="pos"
    )
    expected = np.sum(weights * (residual - b @ beta) ** 2)
    system = GaussianLikelihoodSystem((joint,))
    torch.testing.assert_close(
        system.quadratic(state), torch.tensor(expected), atol=5e-12, rtol=2e-12
    )
    assert joint.log_normalization is None
    with pytest.raises(ValueError, match="normalized"):
        system.negative_log_likelihood(state)
    with pytest.raises(ValueError, match="normalized"):
        system.normal_equations().negative_log_likelihood(state)


def test_shared_calibration_floor_and_profiled_source_gauge():
    observations, responses, _, _, dense = _fixture()
    source_direction = torch.tensor([[1.0, 0.3], [0.2, 0.4]], dtype=torch.float64)
    templates = tuple(response.forward(source_direction) for response in responses)
    bases = tuple(
        torch.stack((template, torch.zeros_like(template)), dim=-1)
        for template in templates
    )
    sigma = 0.8
    covariance = torch.tensor([[sigma], [0.0]], dtype=torch.float64)
    joint = _bind(observations, responses, bases, factor=covariance)
    packed = torch.cat(templates)
    weights = torch.cat(
        tuple(observation.weight.ravel() for observation in observations)
    )
    expected_variance = sigma**2 + 1 / (weights * packed.square()).sum()
    query = fit_linear_query(GaussianLikelihoodSystem((joint,)), (packed[None],))
    torch.testing.assert_close(
        query.sampling_covariance[0, 0], expected_variance, atol=2e-12, rtol=2e-12
    )
    # Splitting calibration into independent per-instrument random effects would
    # average down the shared sigma floor; this joint construction retains it.
    assert query.sampling_covariance[0, 0] > sigma**2
    profiled = _bind(observations, responses, bases, treatment="profiled")
    info_direction = GaussianLikelihoodSystem((profiled,)).information(source_direction)
    torch.testing.assert_close(
        info_direction, torch.zeros_like(source_direction), atol=2e-11, rtol=0
    )
    precision = profiled.precision(packed)
    torch.testing.assert_close(precision, torch.zeros_like(packed), atol=5e-13, rtol=0)


@pytest.mark.parametrize(
    "change", ["sky", "wavelength", "unit", "source_kind", "spectral_kind", "dtype"]
)
def test_inconsistent_shared_source_states_fail_closed(change):
    observations, responses, bases, _, _ = _fixture()
    other = responses[1]
    if change == "sky":
        other = replace(other, source_ra_deg=other.source_ra_deg + 0.01)
    elif change == "wavelength":
        other = replace(
            other,
            quadrature=replace(
                other.quadrature,
                wavelength_angstrom=other.quadrature.wavelength_angstrom + 1,
            ),
        )
    elif change == "unit":
        other = replace(other, source_value_unit="different source unit")
    elif change == "source_kind":
        other = replace(
            other,
            quadrature=replace(
                other.quadrature, source_measure=torch.ones(2, dtype=torch.float64)
            ),
            source_measure_kind="point_flux",
        )
    elif change == "spectral_kind":
        other = replace(
            other,
            quadrature=replace(
                other.quadrature, wavelength_measure=torch.ones(2, dtype=torch.float64)
            ),
            spectral_measure_kind="integrated_bins",
        )
    else:
        observations32, responses32, bases32, _, _ = _fixture(dtype=torch.float32)
        observations = (observations[0], observations32[1])
        other = responses32[1]
        bases = (bases[0], bases32[1])
    with pytest.raises(ValueError, match="same sampled|share ordered"):
        _bind(observations, (responses[0], other), bases)


@pytest.mark.parametrize(
    "change",
    [
        "scale",
        "matrix",
        "basis",
        "mean",
        "factor",
        "data",
        "weight",
        "header",
        "binding",
    ],
)
def test_parent_and_nuisance_changes_invalidate_cached_normal_and_query(change):
    observations, responses, bases, matrices, _ = _fixture()
    mean = torch.tensor([0.1, -0.2], dtype=torch.float64)
    factor = torch.eye(2, dtype=torch.float64) * 0.3
    joint = _bind(observations, responses, bases, mean=mean, factor=factor)
    system = GaussianLikelihoodSystem((joint,))
    normal = system.normal_equations()
    template = torch.cat(
        tuple(
            response.forward(torch.ones(2, 2, dtype=torch.float64))
            for response in responses
        )
    )[None]
    query = fit_linear_query(system, (template,))
    if change == "scale":
        responses[1].calibration.scale.add_(0.1)
    elif change == "matrix":
        matrices[1].add_(0.1)
    elif change == "basis":
        bases[1].add_(0.1)
    elif change == "mean":
        mean.add_(0.1)
    elif change == "factor":
        factor.add_(0.1)
    elif change == "data":
        observations[1].image.add_(0.1)
    elif change == "weight":
        observations[1].weight.add_(0.1)
    elif change == "header":
        observations[1].header["OFFSET"] = 1
    else:
        responses[1].quadrature.source_spectral_weight = responses[
            1
        ].quadrature.source_spectral_weight.clone()
    with pytest.raises(RuntimeError, match="stale evidence"):
        _ = normal.rhs
    with pytest.raises(RuntimeError, match="stale evidence"):
        _ = query.sampling_covariance
    with pytest.raises(RuntimeError, match="stale evidence"):
        joint.precision(torch.ones_like(joint.data))


def test_alias_revision_and_inference_constraints():
    revision = [0]
    observations, responses, bases, _, _ = _fixture(revision=lambda: revision[0])
    joint = _bind(observations, responses, bases, revision=lambda: revision[0])
    normal = GaussianLikelihoodSystem((joint,)).normal_equations()
    bases[0].numpy()[0, 0] += 0.1
    revision[0] += 1
    with pytest.raises(RuntimeError, match="stale evidence revision"):
        _ = normal.data_quadratic
    with torch.inference_mode():
        observations, responses, bases, _, _ = _fixture(revision=lambda: revision[0])
        with pytest.raises(ValueError, match="explicit revision"):
            _bind(observations, responses, bases)
        joint = _bind(observations, responses, bases, revision=lambda: revision[0])
        normal = GaussianLikelihoodSystem((joint,)).normal_equations()
        torch.testing.assert_close(
            normal.quadratic(torch.ones(2, 2, dtype=torch.float64)),
            GaussianLikelihoodSystem((joint,)).quadratic(
                torch.ones(2, 2, dtype=torch.float64)
            ),
        )
    revision[0] += 1
    with pytest.raises(RuntimeError, match="stale evidence"):
        _ = normal.rhs


def test_gradients_in_state_calibrated_mean_factor_and_basis():
    observations, responses, bases, _, _ = _fixture()
    state = torch.tensor(
        [[0.2, 0.5], [0.3, 0.8]], dtype=torch.float64, requires_grad=True
    )
    mean = torch.tensor([0.1, -0.2], dtype=torch.float64, requires_grad=True)
    factor = torch.tensor(
        [[0.5, 0.0], [0.1, 0.3]], dtype=torch.float64, requires_grad=True
    )
    first_basis = bases[0].clone().requires_grad_()

    def apply(x, mu, covariance, basis):
        joint = _bind(
            observations, responses, (basis, bases[1]), mean=mu, factor=covariance
        )
        return GaussianLikelihoodSystem((joint,)).negative_log_likelihood(x)

    args = (state, mean, factor, first_basis)
    assert torch.autograd.gradcheck(apply, args, fast_mode=True)
    assert torch.autograd.gradgradcheck(apply, args, fast_mode=True)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_dtype_device(dtype, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if device == "mps" and (
        not torch.backends.mps.is_available() or dtype == torch.float64
    ):
        pytest.skip("MPS unavailable or float64 unsupported")
    observations, responses, bases, _, _ = _fixture(dtype=dtype, device=device)
    joint = _bind(observations, responses, bases)
    state = torch.ones(2, 2, dtype=dtype, device=device)
    quadratic = GaussianLikelihoodSystem((joint,)).quadratic(state)
    assert quadratic.dtype == dtype and quadratic.device.type == device
    torch.testing.assert_close(
        quadratic,
        GaussianLikelihoodSystem((joint,)).normal_equations().quadratic(state),
        atol=2e-3 if dtype == torch.float32 else 1e-10,
        rtol=2e-5 if dtype == torch.float32 else 1e-12,
    )


def test_bad_basis_units_shapes_modes_and_duplicate_lineage():
    observations, responses, bases, _, _ = _fixture()
    common = dict(
        measurement_ids=(("native-0",), ("native-1",)),
        nuisance_bases=bases,
        nuisance_basis_units=("electron", "ADU"),
        nuisance_ids=("offset", "drift"),
        nuisance_treatment="profiled",
        calibration_id="joint",
        provenance_id="joint",
    )
    for changes, message in (
        ({"nuisance_basis_units": ("ADU", "electron")}, "basis units"),
        ({"nuisance_bases": (bases[0].reshape(1, 3, 2), bases[1])}, "explicit shape"),
        ({"measurement_ids": (("same",), ("same",))}, "duplicate"),
        ({"nuisance_ids": ("same", "same")}, "duplicate"),
        (
            {"calibrated_mean": torch.zeros(2, dtype=torch.float64)},
            "profiled nuisances",
        ),
        ({"nuisance_treatment": "calibrated"}, "covariance_factor"),
    ):
        with pytest.raises(ValueError, match=message):
            bind_joint_native_likelihood(observations, responses, **(common | changes))
