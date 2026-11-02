from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import torch
from torch import Tensor

from .temporal import convert_time_scale

UnitDimension = Literal[
    "dimensionless",
    "pixel",
    "angle",
    "length",
    "frequency",
    "energy",
    "velocity",
    "time",
]
TimeFormat = Literal["jd", "mjd", "jyear", "byear", "unix"]
TimeScale = Literal["utc", "tai", "tt", "gps", "ut1", "tcb", "tdb"]


_UNIT_DEFINITIONS: dict[str, tuple[str, UnitDimension, float]] = {
    "": ("", "dimensionless", 1.0),
    "1": ("", "dimensionless", 1.0),
    "pix": ("pix", "pixel", 1.0),
    "pixel": ("pix", "pixel", 1.0),
    "pixels": ("pix", "pixel", 1.0),
    "rad": ("rad", "angle", 1.0),
    "radian": ("rad", "angle", 1.0),
    "radians": ("rad", "angle", 1.0),
    "deg": ("deg", "angle", torch.pi / 180.0),
    "degree": ("deg", "angle", torch.pi / 180.0),
    "degrees": ("deg", "angle", torch.pi / 180.0),
    "arcmin": ("arcmin", "angle", torch.pi / 10800.0),
    "arcsec": ("arcsec", "angle", torch.pi / 648000.0),
    "mas": ("mas", "angle", torch.pi / 648000000.0),
    "m": ("m", "length", 1.0),
    "cm": ("cm", "length", 1.0e-2),
    "mm": ("mm", "length", 1.0e-3),
    "um": ("um", "length", 1.0e-6),
    "µm": ("um", "length", 1.0e-6),
    "nm": ("nm", "length", 1.0e-9),
    "angstrom": ("Angstrom", "length", 1.0e-10),
    "å": ("Angstrom", "length", 1.0e-10),
    "km": ("km", "length", 1.0e3),
    "au": ("au", "length", 1.495978707e11),
    "pc": ("pc", "length", 3.085677581491367e16),
    "kpc": ("kpc", "length", 3.085677581491367e19),
    "hz": ("Hz", "frequency", 1.0),
    "khz": ("kHz", "frequency", 1.0e3),
    "mhz": ("MHz", "frequency", 1.0e6),
    "ghz": ("GHz", "frequency", 1.0e9),
    "j": ("J", "energy", 1.0),
    "ev": ("eV", "energy", 1.602176634e-19),
    "kev": ("keV", "energy", 1.602176634e-16),
    "m/s": ("m/s", "velocity", 1.0),
    "km/s": ("km/s", "velocity", 1.0e3),
    "s": ("s", "time", 1.0),
    "ms": ("ms", "time", 1.0e-3),
    "us": ("us", "time", 1.0e-6),
    "ns": ("ns", "time", 1.0e-9),
    "min": ("min", "time", 60.0),
    "h": ("h", "time", 3600.0),
    "d": ("d", "time", 86400.0),
    "yr": ("yr", "time", 31557600.0),
}


@dataclass(frozen=True)
class UnitSpec:
    symbol: str = ""
    dimension: UnitDimension | None = None

    def __post_init__(self) -> None:
        key = str(self.symbol).strip().lower()
        try:
            symbol, dimension, _ = _UNIT_DEFINITIONS[key]
        except KeyError as exc:
            raise ValueError(f"unsupported unit {self.symbol!r}") from exc
        if self.dimension is not None and self.dimension != dimension:
            raise ValueError(
                f"unit {self.symbol!r} has dimension {dimension!r}, not {self.dimension!r}"
            )
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "dimension", dimension)

    @property
    def scale(self) -> float:
        return float(_UNIT_DEFINITIONS[self.symbol.lower()][2])

    def conversion_factor_to(self, other: UnitSpec | str) -> float:
        destination = other if isinstance(other, UnitSpec) else UnitSpec(other)
        if self.dimension != destination.dimension:
            raise ValueError(
                f"cannot convert {self.dimension!r} unit {self.symbol!r} to "
                f"{destination.dimension!r} unit {destination.symbol!r}"
            )
        return self.scale / destination.scale

    def convert(self, value: Tensor, other: UnitSpec | str) -> Tensor:
        return value * self.conversion_factor_to(other)

    def to_dict(self) -> dict[str, str]:
        return {"symbol": self.symbol, "dimension": str(self.dimension)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> UnitSpec:
        return cls(str(data["symbol"]), str(data["dimension"]))  # type: ignore[arg-type]


@dataclass(frozen=True)
class DomainSpec:
    bounds: tuple[tuple[float | None, float | None], ...]
    closed: tuple[tuple[bool, bool], ...] = ()

    def __post_init__(self) -> None:
        bounds = tuple((lower, upper) for lower, upper in self.bounds)
        if not bounds:
            raise ValueError("domain must contain at least one axis")
        for lower, upper in bounds:
            if lower is not None and upper is not None and lower > upper:
                raise ValueError("domain lower bounds must not exceed upper bounds")
        closed = self.closed or tuple((True, True) for _ in bounds)
        if len(closed) != len(bounds):
            raise ValueError("domain closed flags must match domain dimensionality")
        object.__setattr__(self, "bounds", bounds)
        object.__setattr__(self, "closed", tuple(closed))

    @property
    def n_dim(self) -> int:
        return len(self.bounds)

    def contains(self, value: Tensor) -> Tensor:
        if value.ndim == 0 or value.shape[-1] != self.n_dim:
            raise ValueError(f"domain expects trailing dimension {self.n_dim}")
        mask = torch.ones(value.shape[:-1], dtype=torch.bool, device=value.device)
        for axis, ((lower, upper), (lower_closed, upper_closed)) in enumerate(
            zip(self.bounds, self.closed, strict=True)
        ):
            coordinate = value[..., axis]
            if lower is not None:
                mask &= coordinate >= lower if lower_closed else coordinate > lower
            if upper is not None:
                mask &= coordinate <= upper if upper_closed else coordinate < upper
        return mask

    def clamp(self, value: Tensor) -> Tensor:
        if value.ndim == 0 or value.shape[-1] != self.n_dim:
            raise ValueError(f"domain expects trailing dimension {self.n_dim}")
        coordinates = []
        for axis, (lower, upper) in enumerate(self.bounds):
            coordinate = value[..., axis]
            if lower is not None:
                coordinate = torch.clamp_min(coordinate, lower)
            if upper is not None:
                coordinate = torch.clamp_max(coordinate, upper)
            coordinates.append(coordinate)
        return torch.stack(coordinates, dim=-1)

    def centre(self, *, dtype: torch.dtype, device: torch.device) -> Tensor:
        values = []
        for lower, upper in self.bounds:
            if lower is None or upper is None:
                raise ValueError("domain centre requires finite lower and upper bounds")
            values.append((lower + upper) * 0.5)
        return torch.tensor(values, dtype=dtype, device=device)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bounds": [[lower, upper] for lower, upper in self.bounds],
            "closed": [[lower, upper] for lower, upper in self.closed],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DomainSpec:
        return cls(
            tuple((bound[0], bound[1]) for bound in data["bounds"]),
            tuple((bool(flags[0]), bool(flags[1])) for flags in data["closed"]),
        )


@dataclass(frozen=True)
class AxisSpec:
    name: str
    unit: UnitSpec | str = UnitSpec()
    physical_type: str | None = None
    domain: DomainSpec | None = None
    wrap: float | None = None

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("axis name must not be empty")
        unit = self.unit if isinstance(self.unit, UnitSpec) else UnitSpec(self.unit)
        if self.domain is not None and self.domain.n_dim != 1:
            raise ValueError("an axis domain must be one-dimensional")
        if self.wrap is not None and self.wrap <= 0.0:
            raise ValueError("axis wrap must be positive")
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "unit", unit)

    @property
    def unit_spec(self) -> UnitSpec:
        """The unit as a :class:`UnitSpec` (strings are normalised on construction)."""
        unit = self.unit
        assert isinstance(unit, UnitSpec)
        return unit

    def compatible_with(self, other: AxisSpec, *, require_unit: bool = True) -> bool:
        if self.physical_type is not None and other.physical_type is not None:
            if self.physical_type != other.physical_type:
                return False
        if self.unit_spec.dimension != other.unit_spec.dimension:
            return False
        return not require_unit or self.unit == other.unit

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "unit": self.unit_spec.to_dict(),
            "physical_type": self.physical_type,
            "domain": None if self.domain is None else self.domain.to_dict(),
            "wrap": self.wrap,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AxisSpec:
        domain = data.get("domain")
        return cls(
            name=str(data["name"]),
            unit=UnitSpec.from_dict(data["unit"]),
            physical_type=data.get("physical_type"),
            domain=None if domain is None else DomainSpec.from_dict(domain),
            wrap=data.get("wrap"),
        )


@dataclass(frozen=True)
class TimeSpec:
    format: TimeFormat = "jyear"
    scale: TimeScale = "tcb"
    reference_position: str | None = None

    def __post_init__(self) -> None:
        time_format = str(self.format).strip().lower()
        scale = str(self.scale).strip().lower()
        if time_format not in {"jd", "mjd", "jyear", "byear", "unix"}:
            raise ValueError(f"unsupported time format {self.format!r}")
        if scale not in {"utc", "tai", "tt", "gps", "ut1", "tcb", "tdb"}:
            raise ValueError(f"unsupported time scale {self.scale!r}")
        object.__setattr__(self, "format", time_format)
        object.__setattr__(self, "scale", scale)
        if self.reference_position is not None:
            object.__setattr__(self, "reference_position", self.reference_position.upper())

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "scale": self.scale,
            "reference_position": self.reference_position,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TimeSpec:
        return cls(
            format=str(data["format"]),  # type: ignore[arg-type]
            scale=str(data["scale"]),  # type: ignore[arg-type]
            reference_position=data.get("reference_position"),
        )


def _time_to_mjd(value: Tensor, time_format: TimeFormat) -> Tensor:
    if time_format == "mjd":
        return value
    if time_format == "jd":
        return value - 2400000.5
    if time_format == "jyear":
        return 51544.5 + (value - 2000.0) * 365.25
    if time_format == "byear":
        return 15019.81352 + (value - 1900.0) * 365.242198781
    if time_format == "unix":
        return 40587.0 + value / 86400.0
    raise ValueError(f"unsupported time format {time_format!r}")


def _mjd_to_time(value: Tensor, time_format: TimeFormat) -> Tensor:
    if time_format == "mjd":
        return value
    if time_format == "jd":
        return value + 2400000.5
    if time_format == "jyear":
        return 2000.0 + (value - 51544.5) / 365.25
    if time_format == "byear":
        return 1900.0 + (value - 15019.81352) / 365.242198781
    if time_format == "unix":
        return (value - 40587.0) * 86400.0
    raise ValueError(f"unsupported time format {time_format!r}")


@dataclass(frozen=True)
class Epoch:
    value: Tensor
    spec: TimeSpec = TimeSpec()

    def __init__(self, value: Any, spec: TimeSpec | None = None) -> None:
        tensor = value if isinstance(value, Tensor) else torch.as_tensor(value, dtype=torch.float64)
        if not tensor.is_floating_point():
            tensor = tensor.to(dtype=torch.float64)
        object.__setattr__(self, "value", tensor)
        object.__setattr__(self, "spec", TimeSpec() if spec is None else spec)

    def to(self, spec: TimeSpec) -> Epoch:
        if self.spec == spec:
            return self
        mjd = _time_to_mjd(self.value, self.spec.format)
        if self.spec.scale != spec.scale:
            supported = {"utc", "tai", "tt", "gps"}
            if self.spec.scale not in supported or spec.scale not in supported:
                raise ValueError(
                    f"unsupported time-scale conversion {self.spec.scale!r}->{spec.scale!r}"
                )
            mjd = convert_time_scale(mjd, self.spec.scale, spec.scale)
        return Epoch(_mjd_to_time(mjd, spec.format), spec)

    def to_state(self) -> tuple[dict[str, Any], dict[str, Tensor]]:
        return {"spec": self.spec.to_dict(), "value": "value"}, {"value": self.value}

    @classmethod
    def from_state(
        cls, metadata: Mapping[str, Any], tensors: Mapping[str, Tensor]
    ) -> Epoch:
        return cls(tensors[str(metadata["value"])], TimeSpec.from_dict(metadata["spec"]))


@dataclass(frozen=True)
class FrameSpec:
    name: str
    axes: tuple[AxisSpec, ...]
    representation: str = "cartesian"
    equinox: Epoch | None = None
    obstime: Epoch | None = None
    observer: tuple[float, ...] | None = None
    reference_position: str | None = None

    def __init__(
        self,
        name: str,
        axes: Sequence[AxisSpec],
        representation: str = "cartesian",
        equinox: Epoch | None = None,
        obstime: Epoch | None = None,
        observer: Sequence[float] | None = None,
        reference_position: str | None = None,
    ) -> None:
        axes_tuple = tuple(axes)
        if not str(name).strip():
            raise ValueError("frame name must not be empty")
        if not axes_tuple:
            raise ValueError("frame must contain at least one axis")
        if len({axis.name for axis in axes_tuple}) != len(axes_tuple):
            raise ValueError("frame axis names must be unique")
        object.__setattr__(self, "name", str(name))
        object.__setattr__(self, "axes", axes_tuple)
        object.__setattr__(self, "representation", str(representation))
        object.__setattr__(self, "equinox", equinox)
        object.__setattr__(self, "obstime", obstime)
        object.__setattr__(
            self, "observer", None if observer is None else tuple(float(v) for v in observer)
        )
        object.__setattr__(
            self,
            "reference_position",
            None if reference_position is None else str(reference_position).upper(),
        )

    @property
    def n_dim(self) -> int:
        return len(self.axes)


def axes_domain(axes: Sequence[AxisSpec]) -> DomainSpec | None:
    if not axes or any(axis.domain is None for axis in axes):
        return None
    return DomainSpec(
        tuple(axis.domain.bounds[0] for axis in axes if axis.domain is not None),
        tuple(axis.domain.closed[0] for axis in axes if axis.domain is not None),
    )
