"""Explicit diagonal calibration of an already integrated detector product."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch import Tensor

from ..evidence import EvidenceSnapshot


@dataclass(frozen=True, kw_only=True)
class PixelCalibrationOp:
    """A fixed positive detector-product transform ``D = diag(scale)``.

    ``scale[P]`` multiplies native pixel values after detector integration.
    ``input_unit`` and ``output_unit`` declare its dimensional ratio; labels
    do not trigger conversions or infer pixel area, exposure time or QE.
    ``flat_convention`` identifies the declared response product (e.g.
    ``sky_flat``, ``point_source`` or ``detector_counts``).
    ``reference_convention`` explicitly identifies the illuminating spatial
    and spectral reference used to calibrate that product. Neither convention
    asserts simultaneous uniform-sky and point-source-flux invariants. In a
    spatially varying response, row and column sums can differ: a sky-flat
    correction can change the pixel sum of a point-source response.

    Compose ``calibration.forward(response.forward(source_intensity))`` with
    a source-quadrature operator. Source solid-angle and wavelength measures
    stay inside that operator; no pixel integral or normalization is repeated
    here. Signed mean vectors are allowed; a physical count/rate model must
    enforce its own support. This primitive has no additive offset.

    The scale and metadata bindings are fixed. Ordinary in-place scale writes
    invalidate the captured ``EvidenceSnapshot``. Supply ``dependencies`` with
    an explicit revision getter for hidden state, inference tensors or writes
    through NumPy/``.data`` aliases. Calibration/provenance IDs are lineage,
    not freshness checks. Include covariance callback state in the same
    snapshot when using ``apply_covariance``; callbacks are not inspected.
    Rebuild the transform after changing dependencies or conventions, and
    when a new autograd graph is required. No tensors are copied or detached.

    Mean actions cost O(P) per vector. The matrix-free covariance action uses
    one caller covariance action plus two diagonal products; the explicit
    dense covariance transform costs O(P^2) per matrix. The covariance is
    conditional on this fixed calibration: uncertainty in the scale is not
    estimated or marginalized by this transform.
    """

    scale: Tensor = field(kw_only=False)
    input_unit: str
    output_unit: str
    calibration_id: str
    provenance_id: str
    flat_convention: str
    reference_convention: str
    dependencies: EvidenceSnapshot | None = None
    _snapshot: EvidenceSnapshot = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.scale, Tensor):
            raise TypeError("scale must be a tensor")
        if self.scale.dtype not in (torch.float32, torch.float64):
            raise TypeError("scale must have real float32 or float64 dtype")
        if self.scale.ndim != 1:
            raise ValueError("scale must have shape [P]")
        if not bool(torch.isfinite(self.scale).all()) or bool((self.scale <= 0).any()):
            raise ValueError("scale must be finite and strictly positive")
        for name in (
            "input_unit",
            "output_unit",
            "calibration_id",
            "provenance_id",
            "flat_convention",
            "reference_convention",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty declaration")
        if self.dependencies is not None and not isinstance(
            self.dependencies, EvidenceSnapshot
        ):
            raise TypeError("dependencies must be an EvidenceSnapshot")
        tensors = {"pixel_calibration_scale": self.scale}
        snapshot = (
            EvidenceSnapshot.capture(tensors)
            if self.dependencies is None
            else self.dependencies.with_tensors(tensors)
        )
        object.__setattr__(self, "_snapshot", snapshot)

    def validate_evidence(self) -> None:
        """Reject changed declared calibration/covariance dependencies."""
        self._snapshot.validate()

    def _compatible(self, value: Tensor, name: str) -> None:
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a tensor")
        if value.dtype != self.scale.dtype or value.device != self.scale.device:
            raise ValueError(f"{name} must match scale dtype and device")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be finite")

    def _vector(self, value: Tensor, name: str) -> None:
        self._compatible(value, name)
        if value.ndim < 1 or value.shape[-1] != self.scale.numel():
            raise ValueError(f"{name} must have trailing shape [P]")

    def forward(self, value: Tensor) -> Tensor:
        """Apply ``D`` to signed mean values ``value[...,P]``."""
        self.validate_evidence()
        self._vector(value, "mean input")
        result = value * self.scale
        self._vector(result, "mean output")
        return result

    def adjoint(self, value: Tensor) -> Tensor:
        """Apply the Euclidean transpose ``D.T = D`` to ``value[...,P]``.

        This is not inverse calibration; positivity does not change the
        distinction between a transpose and multiplication by reciprocal scale.
        """
        return self.forward(value)

    def apply_covariance(
        self,
        value: Tensor,
        *,
        covariance_action: Callable[[Tensor], Tensor],
    ) -> Tensor:
        """Apply the full output covariance ``D C D.T`` to ``value[...,P]``.

        ``covariance_action`` must implement the input product's fixed linear
        covariance on the same vector shape/dtype/device, including any
        correlations. Its symmetry/PSD are the caller's contract. Scaling only
        its output or only diagonal variances would change this action.
        """
        self.validate_evidence()
        self._vector(value, "covariance vector")
        if not callable(covariance_action):
            raise TypeError("covariance_action must be callable")
        weighted = covariance_action(value * self.scale)
        self.validate_evidence()
        self._vector(weighted, "input covariance output")
        if weighted.shape != value.shape:
            raise ValueError("covariance_action must preserve the vector shape")
        result = self.scale * weighted
        self._vector(result, "output covariance action")
        return result

    def transform_covariance(self, covariance: Tensor) -> Tensor:
        """Transform a full matrix ``covariance[...,P,P]`` as ``D C D.T``.

        This optional dense operation preserves off-diagonal terms. The input
        is caller-declared covariance; it does not estimate/check its PSD or
        substitute a diagonal approximation.
        """
        self.validate_evidence()
        self._compatible(covariance, "covariance")
        count = self.scale.numel()
        if covariance.ndim < 2 or covariance.shape[-2:] != (count, count):
            raise ValueError("covariance must have trailing shape [P,P]")
        result = self.scale[:, None] * covariance * self.scale[None, :]
        self._compatible(result, "output covariance")
        return result


__all__ = ["PixelCalibrationOp"]
