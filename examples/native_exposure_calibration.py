"""Run a bounded native-pixel Gaussian reference without a global field.

Run with ``pixi run -e default -- python examples/native_exposure_calibration.py``.
This is an in-memory API example, not independent science validation or a PSF
training campaign. Its source position, response and Gaussian noise are known.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from torchsky.map.likelihood import GaussianLikelihoodFactor, GaussianLikelihoodSystem
from torchsky.map.operators import (
    CalibratedNuisancePrecision,
    GaussianPixelResponse,
    ObservationOperatorBundle,
    native_pixel_stamp,
    profile_native_star,
)
from torchsky.map.scenes import DiscreteTargetImage, SceneRepresentation
from torchsky.map.query import fit_linear_query
from torchsky.map.solvers import BoundObservationOperator
from torchsky.map.target import TargetRenderSpec


def main() -> None:
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


if __name__ == "__main__":
    main()
