"""Source-node field rendering and conditional native query, without image I/O.

Run ``pixi run -e default -- python examples/native_field_response.py``.
The small planar quadrature is illustrative, not a calibrated instrument model.
"""

import math

import torch
from torch import nn

from torchsky.map.evidence import EvidenceSnapshot
from torchsky.map.field.native_response import (
    NativeFieldResponse,
    bind_native_field_likelihood,
    render_native_field_observation,
)
from torchsky.map.field.types import FieldObservation
from torchsky.map.likelihood import GaussianLikelihoodSystem
from torchsky.map.operators import (
    AchromaticPixelResponse,
    AdditiveProcessingState,
    GaussianPixelResponse,
    NativeQuadratureResponseOp,
    PixelCalibrationOp,
)
from torchsky.map.query import fit_linear_query


class Sky(nn.Module):
    def forward(self, ra, dec, wavelength):
        radius2 = ((ra - 150) * 3600) ** 2 + ((dec - 2) * 3600) ** 2
        return 1e7 * torch.exp(-radius2 / 2) * (wavelength / 5000)


def main() -> None:
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
        processing_state=AdditiveProcessingState.CALIBRATED_IMAGE,
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


if __name__ == "__main__":
    main()
