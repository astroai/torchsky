"""Native observation operators: detector moments, field response, pixel likelihood.

Three small, in-memory sections (no image files) that exercise the native
forward-model stack:

1. ``detector_moments``: a Gaussian pixel response, charge redistribution, dark
   and read noise give the exact mean and covariance action of detector output.
2. ``field_response``: a continuous (RA, Dec, wavelength) sky is rendered through
   an explicit quadrature/calibration chain and a linear template amplitude is
   queried with its sampling uncertainty.
3. ``exposure_calibration``: a known star in a 9x9 stamp is profiled and then
   recovered through a Gaussian likelihood with a calibrated nuisance
   background; the direct template query must agree with the solve.

The instrument numbers are illustrative, not a calibrated model. Runs on CPU in
about 2 s.

    pixi run -e test -- python examples/native_operators.py
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from torchsky.map.evidence import EvidenceSnapshot
from torchsky.map.field.native_response import (
    NativeFieldResponse,
    bind_native_field_likelihood,
    render_native_field_observation,
)
from torchsky.map.likelihood import GaussianLikelihoodFactor, GaussianLikelihoodSystem
from torchsky.map.operators.continuous_response import (
    AchromaticPixelResponse,
    NativeQuadratureResponseOp,
)
from torchsky.map.operators.detector_response import (
    LinearDetectorResponse,
    integrated_detector_moments,
    integrated_photoelectron_mean,
)
from torchsky.map.operators.epsf import (
    GaussianPixelResponse,
    native_pixel_stamp,
    profile_native_star,
)
from torchsky.map.operators.nuisance import CalibratedNuisancePrecision
from torchsky.map.operators.pixel_calibration import PixelCalibrationOp
from torchsky.map.query import fit_linear_query
from torchsky.map.solvers import BoundObservationOperator
from torchsky.scene import (
    DiscreteTargetImage,
    FieldObservation,
    ObservationOperatorBundle,
    SceneRepresentation,
    TargetRenderSpec,
)


def detector_moments() -> None:
    dtype = torch.float64
    # Bin rates are already integrated photons/s, so both measures are one.
    response = NativeQuadratureResponseOp(
        response=AchromaticPixelResponse(
            GaussianPixelResponse(torch.tensor(0.6, dtype=dtype))
        ),
        source_xy=torch.tensor([[-0.3, 0.15], [0.7, -0.1]], dtype=dtype),
        detector_xy=torch.tensor([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]], dtype=dtype),
        wavelength_angstrom=torch.tensor([5000.0, 9000.0], dtype=dtype),
        source_measure=torch.ones(2, dtype=dtype),
        wavelength_measure=torch.ones(2, dtype=dtype),
        source_spectral_weight=torch.tensor([0.8, 0.5], dtype=dtype),
    )
    incident = torch.tensor([[1.4, 2.0], [0.2, 0.6]], dtype=dtype)
    foreground = torch.full_like(incident, 0.05)
    photo = integrated_photoelectron_mean(
        incident,
        foreground_photon_rate=foreground,
        optical_mean_response=response.forward,
        exposure_time_s=100.0,
        electrons_per_photon=0.9,
    )
    # Fixed charge redistribution; each column conserves collected electrons.
    charge = torch.tensor(
        [[0.9, 0.1, 0.0], [0.1, 0.8, 0.1], [0.0, 0.1, 0.9]], dtype=dtype
    )
    moments = integrated_detector_moments(
        photo,
        dark_electron_mean=2.0,
        charge_response=LinearDetectorResponse(
            forward=lambda value: charge @ value,
            adjoint=lambda value: charge.T @ value,
        ),
        read_offset_electrons=10.0,
        read_variance_electrons2=4.0,
        electrons_per_output_unit=2.0,
        subtracted_mean_output_units=5.0,
    )
    aperture = torch.ones_like(photo)
    aperture_variance = aperture @ moments.covariance_action(aperture)
    print("Detector mean (ADU):", moments.mean.tolist())
    print("Summed-aperture variance (ADU^2):", float(aperture_variance))
    print(
        "Finite detector footprint retains PSF losses; subtraction retains shot noise."
    )


class Sky(nn.Module):
    def forward(self, ra, dec, wavelength):
        radius2 = ((ra - 150) * 3600) ** 2 + ((dec - 2) * 3600) ** 2
        return 1e7 * torch.exp(-radius2 / 2) * (wavelength / 5000)


def field_response() -> None:
    dtype = torch.float64
    y, x = torch.meshgrid(
        torch.arange(-1, 2, dtype=dtype),
        torch.arange(-1, 2, dtype=dtype),
        indexing="ij",
    )
    positions = torch.stack((x.flatten(), y.flatten()), -1)
    width = torch.tensor(0.7, dtype=dtype)
    quadrature = NativeQuadratureResponseOp(
        response=AchromaticPixelResponse(GaussianPixelResponse(width)),
        source_xy=positions,
        detector_xy=positions,
        wavelength_angstrom=torch.tensor([5000, 9000], dtype=dtype),
        source_measure=torch.full((9,), (math.pi / 180 / 3600) ** 2, dtype=dtype),
        wavelength_measure=torch.tensor([1000, 1000], dtype=dtype),
        source_spectral_weight=torch.tensor([0.5, 0.8], dtype=dtype),
    )
    calibration = PixelCalibrationOp(
        torch.full((9,), 10 * 100 / 2, dtype=dtype),
        input_unit="photon/s/m2",
        output_unit="ADU",
        calibration_id="illustrative-calibration",
        provenance_id="example",
        flat_convention="detector_counts",
        reference_convention="area 10 m2, duration 100 s, gain 2 electron/ADU; QE in source weights",
    )
    response = NativeFieldResponse(
        quadrature=quadrature,
        calibration=calibration,
        source_ra_deg=150 + positions[:, 0] / 3600,
        source_dec_deg=2 + positions[:, 1] / 3600,
        source_value_unit="photon/s/m2/Angstrom/sr",
        quadrature_output_unit="photon/s/m2",
        source_measure_kind="solid_angle_sr",
        spectral_measure_kind="density_per_angstrom",
        band_name="example-band",
        geometry_id="declared small planar chart",
        response_id="example-response",
        dependencies=EvidenceSnapshot.capture({"width": width}),
    )
    observation = FieldObservation(
        image=torch.zeros(3, 3, dtype=dtype),
        header={"NAXIS1": 3, "NAXIS2": 3},
        weight=torch.ones(3, 3, dtype=dtype),
        background_offset=0.2,
        band_name="example-band",
        value_unit="ADU",
        value_semantics="surface_brightness",
        processing_state="calibrated_image",
        detector_law="gaussian",
    )
    sky = Sky()
    prediction = render_native_field_observation(sky, observation, response)
    observation.image = prediction.detach().clone()
    factor = bind_native_field_likelihood(
        observation,
        response,
        measurement_ids=("example-exposure",),
        provenance_id="example",
    )
    template = response.forward(response.sample_field(sky))[None, :]
    query = fit_linear_query(GaussianLikelihoodSystem((factor,)), (template,))
    print("Native prediction (ADU):", prediction.tolist())
    print("Conditional field-template amplitude:", float(query.coefficients[0]))
    print(
        "Conditional amplitude standard deviation:",
        float(query.sampling_covariance[0, 0].sqrt()),
    )
    print(
        "Explicit density/response/calibration factors; no second PSF or spectral integration."
    )


def exposure_calibration() -> None:
    generator = torch.Generator().manual_seed(42)
    image = torch.zeros(9, 9, dtype=torch.float64)
    variance = torch.full_like(image, 0.04)
    stamp = native_pixel_stamp(
        image, variance, x_slice=slice(0, 9), y_slice=slice(0, 9)
    )
    center = image.new_tensor([4.17, 4.43])
    response = GaussianPixelResponse(image.new_tensor([0.65, 0.8]))
    template = response(stamp.pixel_xy - center)
    actual_background = 3 + 0.1 * torch.randn(
        (), dtype=image.dtype, generator=generator
    )
    stamp.data.copy_(
        17 * template
        + actual_background
        + variance.sqrt()
        * torch.randn(image.shape, dtype=image.dtype, generator=generator)
    )
    profiled = profile_native_star(stamp, response, center)

    def base_precision(value: Tensor) -> Tensor:
        return value / variance

    calibrated = CalibratedNuisancePrecision(
        torch.ones(*image.shape, 1, dtype=image.dtype),
        apply_precision=base_precision,
        covariance_factor=image.new_tensor([[0.1]]),
        mean=image.new_tensor([3.0]),
    )

    def forward(scene: SceneRepresentation, target: TargetRenderSpec) -> Tensor:
        del target
        assert isinstance(scene, DiscreteTargetImage)
        return template * scene.data.reshape(())

    def adjoint(value: Tensor) -> Tensor:
        return (template * value).sum().reshape(1, 1)

    operator = BoundObservationOperator(
        ObservationOperatorBundle(),
        TargetRenderSpec(header={}, shape=(1, 1)),
        apply_operator=forward,
        apply_adjoint=adjoint,
        apply_noise=calibrated,
    )
    normalization = 0.5 * (
        image.numel() * math.log(2 * math.pi)
        + variance.log().sum()
        + calibrated.logdet_covariance_correction
    )
    factor = GaussianLikelihoodFactor(
        operator,
        calibrated.center(stamp.data),
        ("reference-native-pixels",),
        "reference-calibration-v1",
        "in-memory-reference-v1",
        "calibrated",
        ("reference-background",),
        normalization,
    )
    likelihood = GaussianLikelihoodSystem((factor,))
    solution = likelihood.solve(iterations=2).solution
    uncertainty = likelihood.functional_uncertainty(torch.ones_like(solution))
    local_query = fit_linear_query(likelihood, (template.unsqueeze(0),))
    print(f"Native profiled flux: {profiled.flux.item():.6f}")
    print(f"Retained footprint response: {profiled.retained_response.item():.9f}")
    print(f"Calibrated-background flux: {solution.item():.6f}")
    print(f"Direct template-query flux: {local_query.coefficients.item():.6f}")
    torch.testing.assert_close(local_query.coefficients.reshape_as(solution), solution)
    torch.testing.assert_close(
        local_query.sampling_covariance.reshape(()), uncertainty.sampling_variance
    )
    print(
        f"Fixed-model sampling standard deviation: {uncertainty.sampling_variance.sqrt().item():.6f}"
    )
    print(
        f"Normalized Gaussian negative log likelihood: {likelihood.negative_log_likelihood(solution).item():.6f}"
    )


def main() -> None:
    for section in (detector_moments, field_response, exposure_calibration):
        print(f"--- {section.__name__}")
        section()


if __name__ == "__main__":
    main()
