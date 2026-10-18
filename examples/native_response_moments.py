"""Small native response/moment example; no image files or field solve.

Run with ``pixi run -e default -- python examples/native_response_moments.py``.
"""

import torch

from torchsky.map.operators import (
    AchromaticPixelResponse,
    GaussianPixelResponse,
    LinearDetectorResponse,
    NativeQuadratureResponseOp,
    integrated_detector_moments,
    integrated_photoelectron_mean,
)


def main() -> None:
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


if __name__ == "__main__":
    main()
