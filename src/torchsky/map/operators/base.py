"""Internal observation-operator interfaces for mapmaking and field rendering."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field


@dataclass(frozen=True)
class OperatorConditioning:
    """Declared conditioning channels for explicit or learned operators."""

    detector_coordinates: bool = False
    wavelength: bool = False
    time: bool = False
    exposure_metadata_keys: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class ObservationOperator(ABC):
    """Base class for internal operator families."""

    name: str
    learned: bool = False
    enabled: bool = True
    frozen: bool = False
    conditioning: OperatorConditioning = field(default_factory=OperatorConditioning)


@dataclass
class GeometryOp(ObservationOperator):
    pass


@dataclass
class SpectralResponseOp(ObservationOperator):
    pass


@dataclass
class PSFOp(ObservationOperator):
    pixel_response_included: bool | None = None
    pixel_response_provenance: str | None = None

    def __post_init__(self) -> None:
        if self.pixel_response_included is not None and not isinstance(
            self.pixel_response_included, bool
        ):
            raise TypeError("pixel_response_included must be bool or None")
        if self.pixel_response_provenance is not None and (
            not isinstance(self.pixel_response_provenance, str)
            or not self.pixel_response_provenance.strip()
        ):
            raise TypeError("pixel_response_provenance must be a nonempty string or None")


@dataclass
class RadiometryOp(ObservationOperator):
    pass


@dataclass
class BackgroundOp(ObservationOperator):
    pass


@dataclass
class MaskWeightOp(ObservationOperator):
    pass


@dataclass
class NoiseOp(ObservationOperator):
    pass
