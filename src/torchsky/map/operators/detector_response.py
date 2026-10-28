"""Exact moments for one linear, integrated photon-counting exposure.

Independent incident photon arrivals are optically routed/thinned into pixel
photoelectrons, independently collected dark electrons are added, and fixed
linear charge mixing acts on their sum. Gaussian read noise and deterministic
read offsets follow charge mixing. These are moment references, not a
normalized Poisson-Gaussian likelihood or a calibrated-residual noise model.
Saturation, nonlinear charge transport, brighter-fatter effects, and read-ramp
history require other models and are deliberately outside this interface.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor


def _tensor(value: Tensor, *, name: str) -> None:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating tensor")
    if value.numel() == 0:
        raise ValueError(f"{name} must be nonempty")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")


def _nonnegative(value: Tensor, *, name: str) -> None:
    _tensor(value, name=name)
    if bool((value < 0).any()):
        raise ValueError(f"{name} must be non-negative physical rates or means")


def _matching(value: Tensor, reference: Tensor, *, name: str) -> None:
    _tensor(value, name=name)
    if value.dtype != reference.dtype or value.device != reference.device:
        raise ValueError(f"{name} must match the reference dtype and device")
    if value.shape != reference.shape:
        raise ValueError(f"{name} must match the reference shape")


def _parameter(value: Tensor | float, reference: Tensor, *, name: str) -> Tensor:
    if isinstance(value, Tensor):
        _tensor(value, name=name)
        if value.dtype != reference.dtype or value.device != reference.device:
            raise ValueError(f"{name} must match the reference dtype and device")
        if value.ndim != 0 and value.shape != reference.shape:
            raise ValueError(f"{name} must be scalar or match the reference shape")
        return value.expand_as(reference)
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError(f"{name} must be a floating tensor or real scalar")
    result = reference.new_full(reference.shape, value)
    _tensor(result, name=name)
    return result


@dataclass(frozen=True)
class LinearDetectorResponse:
    """A fixed, shape-preserving real linear map and its exact Euclidean adjoint.

    Callbacks must preserve tensor shape, dtype, and device. Their linearity
    and adjoint relationship are caller contracts; they cannot be established
    from one evaluation. Keep callbacks and all captured tensors fixed for the
    lifetime of any moments object that uses them; rebuild after calibration
    changes. This interface does not promise a particular callback complexity.
    """

    forward: Callable[[Tensor], Tensor]
    adjoint: Callable[[Tensor], Tensor]

    def __post_init__(self) -> None:
        if not callable(self.forward) or not callable(self.adjoint):
            raise TypeError("forward and adjoint must be callable")

    def _apply(self, value: Tensor, *, transpose: bool = False) -> Tensor:
        result = self.adjoint(value) if transpose else self.forward(value)
        _matching(result, value, name="linear response result")
        return result


def integrated_photoelectron_mean(
    incident_photon_rate: Tensor,
    *,
    exposure_time_s: Tensor | float,
    foreground_photon_rate: Tensor | None = None,
    optical_mean_response: Callable[[Tensor], Tensor] | None = None,
    electrons_per_photon: Tensor | float = 1.0,
) -> Tensor:
    """Convert incident and foreground photon/s into independent electron means.

    Input rates are photons/s in each bin of the same explicitly chosen
    incident spatial/spectral discretization; flux/radiance densities must
    first be integrated with their spatial/spectral measures by the caller.
    ``optical_mean_response`` is a fixed positive linear map from their sum to
    pixel photon/s and may change shape; the caller includes needed optical
    throughput and pixel integration there. It describes independent
    routing/thinning of Poisson arrivals, not deterministic mixing
    of already collected charge. Its coefficients are non-amplifying
    routing/thinning probabilities over disjoint integrated incident photon
    bins: each incident bin has total destination detection probability at most
    one. Runtime checks on evaluated callback outputs cannot establish global
    positivity, linearity or this substochasticity contract.
    ``electrons_per_photon`` is a pixel detection
    probability in [0, 1], including any remaining QE/throughput exactly once.
    Duration is in seconds. For photons capable of generating multiple
    electrons, these Poisson moment assumptions do not apply.
    """
    _nonnegative(incident_photon_rate, name="incident_photon_rate")
    photons = incident_photon_rate
    if foreground_photon_rate is not None:
        _matching(foreground_photon_rate, photons, name="foreground_photon_rate")
        _nonnegative(foreground_photon_rate, name="foreground_photon_rate")
        photons = photons + foreground_photon_rate
    _nonnegative(photons, name="total incident photon rate")
    if optical_mean_response is not None:
        if not callable(optical_mean_response):
            raise TypeError("optical_mean_response must be callable")
        pixels = optical_mean_response(photons)
        _nonnegative(pixels, name="optical_mean_response result")
        if pixels.dtype != photons.dtype or pixels.device != photons.device:
            raise ValueError("optical_mean_response must preserve dtype and device")
    else:
        pixels = photons
    duration = _parameter(exposure_time_s, pixels, name="exposure_time_s")
    efficiency = _parameter(electrons_per_photon, pixels, name="electrons_per_photon")
    _nonnegative(duration, name="exposure_time_s")
    _nonnegative(efficiency, name="electrons_per_photon")
    if bool((efficiency > 1).any()):
        raise ValueError("electrons_per_photon must be a probability in [0, 1]")
    result = pixels * efficiency * duration
    _nonnegative(result, name="photoelectron_mean")
    return result


@dataclass(frozen=True)
class IntegratedDetectorMoments:
    """Mean and exact matrix-free covariance, built by the factory below.

    Tensor values and callback captures are borrowed, not copied; do not
    mutate them during this object's lifetime. Rebuild for a new calibration
    or photon mean. ``mean`` is in output units and ``covariance_action(v)``
    is the covariance matrix in output-units squared applied to ``v``.
    Correlation means this object intentionally exposes no variance plane.
    """

    mean: Tensor
    photoelectron_mean: Tensor
    dark_electron_mean: Tensor
    _read_variance: Tensor
    _gain: Tensor
    _charge_response: LinearDetectorResponse | None
    _read_noise_response: LinearDetectorResponse | None

    def __post_init__(self) -> None:
        _nonnegative(self.photoelectron_mean, name="photoelectron_mean")
        for value, name in (
            (self.mean, "detector mean"),
            (self.dark_electron_mean, "dark_electron_mean"),
            (self._read_variance, "read_variance_electrons2"),
            (self._gain, "electrons_per_output_unit"),
        ):
            _matching(value, self.photoelectron_mean, name=name)
        _nonnegative(self.dark_electron_mean, name="dark_electron_mean")
        _nonnegative(self._read_variance, name="read_variance_electrons2")
        if bool((self._gain <= 0).any()):
            raise ValueError("electrons_per_output_unit must be positive")
        for response in (self._charge_response, self._read_noise_response):
            if response is not None and not isinstance(
                response, LinearDetectorResponse
            ):
                raise TypeError("responses must be LinearDetectorResponse or None")

    def covariance_action(self, vector: Tensor) -> Tensor:
        """Apply G [D diag(photo + dark) D^T + C_read] G^T exactly.

        G is diagonal ``1 / electrons_per_output_unit``. Read noise has
        ``C_read = R diag(read_variance_electrons2) R^T``; identity R gives
        independent read noise. R operates after charge mixing and need not
        commute with D. Both terms retain off-diagonal covariance.
        """
        _matching(vector, self.mean, name="covariance vector")
        scaled = vector / self._gain
        charge = self._charge_response
        backprojected = (
            scaled if charge is None else charge._apply(scaled, transpose=True)
        )
        shot = (self.photoelectron_mean + self.dark_electron_mean) * backprojected
        if charge is not None:
            shot = charge._apply(shot)
        read_response = self._read_noise_response
        read_backprojected = (
            scaled
            if read_response is None
            else read_response._apply(scaled, transpose=True)
        )
        read = self._read_variance * read_backprojected
        if read_response is not None:
            read = read_response._apply(read)
        result = (shot + read) / self._gain
        _matching(result, vector, name="covariance result")
        return result


def integrated_detector_moments(
    photoelectron_mean: Tensor,
    *,
    dark_electron_mean: Tensor | float = 0.0,
    charge_response: LinearDetectorResponse | None = None,
    read_offset_electrons: Tensor | float = 0.0,
    read_variance_electrons2: Tensor | float = 0.0,
    read_noise_response: LinearDetectorResponse | None = None,
    electrons_per_output_unit: Tensor | float = 1.0,
    subtracted_mean_output_units: Tensor | float = 0.0,
) -> IntegratedDetectorMoments:
    """Build exact moments of independent Poisson collection plus Gaussian read.

    ``photoelectron_mean`` and ``dark_electron_mean`` are expected integrated
    electrons *before* charge mixing; dark includes its own duration conversion
    and does not undergo the optical response. Both are finite nonnegative
    physical means, never signed calibrated residuals. Read offsets are signed
    deterministic electron means *after* charge mixing. Read variance is in
    electrons squared; optional R factors correlated Gaussian noise.

    All tensor parameters must be scalar or exactly match the photoelectron
    shape, dtype and device. Gain is positive electrons per output unit.
    Subtraction is a signed deterministic output-unit mean: subtracting a
    background or dark expectation retains its original shot covariance.
    A deterministic flat calibration is expressible through gain and changes
    covariance on both sides. Uncertainty in fitted subtraction/flat terms is
    not represented by these fixed-calibration moments.
    """
    _nonnegative(photoelectron_mean, name="photoelectron_mean")
    for response, name in (
        (charge_response, "charge_response"),
        (read_noise_response, "read_noise_response"),
    ):
        if response is not None and not isinstance(response, LinearDetectorResponse):
            raise TypeError(f"{name} must be a LinearDetectorResponse or None")
    dark = _parameter(dark_electron_mean, photoelectron_mean, name="dark_electron_mean")
    read_variance = _parameter(
        read_variance_electrons2, photoelectron_mean, name="read_variance_electrons2"
    )
    read_offset = _parameter(
        read_offset_electrons, photoelectron_mean, name="read_offset_electrons"
    )
    gain = _parameter(
        electrons_per_output_unit, photoelectron_mean, name="electrons_per_output_unit"
    )
    subtraction = _parameter(
        subtracted_mean_output_units,
        photoelectron_mean,
        name="subtracted_mean_output_units",
    )
    _nonnegative(dark, name="dark_electron_mean")
    _nonnegative(read_variance, name="read_variance_electrons2")
    if bool((gain <= 0).any()):
        raise ValueError("electrons_per_output_unit must be positive")
    collected = photoelectron_mean + dark
    _nonnegative(collected, name="total electron mean")
    mixed = collected if charge_response is None else charge_response._apply(collected)
    mean = (mixed + read_offset) / gain - subtraction
    _matching(mean, photoelectron_mean, name="detector mean")
    return IntegratedDetectorMoments(
        mean,
        photoelectron_mean,
        dark,
        read_variance,
        gain,
        charge_response,
        read_noise_response,
    )


__all__ = [
    "IntegratedDetectorMoments",
    "LinearDetectorResponse",
    "integrated_detector_moments",
    "integrated_photoelectron_mean",
]
