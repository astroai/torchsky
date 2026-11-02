"""Axis metadata, spectral/temporal axes, and celestial frame handling for :class:`torchsky.wcs.WCS`."""

from __future__ import annotations

import torch
from torch import Tensor
from typing import (
    TYPE_CHECKING,
    Any,
    Tuple,
)


from .spectral import (
    canonical_spectral_type,
    frequency_to_spectral as _frequency_to_spectral,
    rest_frequency_from_header,
    spectral_to_frequency as _spectral_to_frequency,
)
from .temporal import TemporalWCS
from .frames import (
    canonical_celestial_frame,
    convert_celestial as _convert_celestial,
    precess_fk5 as _precess_fk5,
)

if TYPE_CHECKING:
    from .mapping import WCSMapping


class _AxesFramesMixin:
    if TYPE_CHECKING:
        # State and methods supplied by the host class.
        def __getattr__(self, name: str) -> Any: ...

    @property
    def pixel_n_dim(self) -> int:
        """Number of pixel axes."""
        return self.naxis

    @property
    def world_n_dim(self) -> int:
        """Number of world axes."""
        return self.naxis

    def as_mapping(self) -> WCSMapping:
        """Return a typed tensor mapping backed by this optimized WCS."""
        from .mapping import WCSMapping

        return WCSMapping(self)

    @property
    def world_axis_names(self) -> tuple[str, ...]:
        """World axis names from ``CTYPE`` with blanks filled by FITS axis labels."""
        names: list[str] = []
        for i in range(1, self.naxis + 1):
            ctype = str(self.wcs_params.get(f"CTYPE{i}", "")).strip()
            names.append(ctype or f"WORLD{i}")
        return tuple(names)

    @property
    def world_axis_units(self) -> tuple[str, ...]:
        """World axis units from ``CUNIT`` keywords."""
        return tuple(
            str(self.wcs_params.get(f"CUNIT{i}", "")) for i in range(1, self.naxis + 1)
        )

    @property
    def world_axis_physical_types(self) -> tuple[str | None, ...]:
        """Best-effort FITS CTYPE to Astropy/GWCS-style physical type mapping."""
        result: list[str | None] = []
        for i in range(1, self.naxis + 1):
            ctype = str(self.wcs_params.get(f"CTYPE{i}", "")).strip().upper()
            if ctype.startswith("RA"):
                result.append("pos.eq.ra")
            elif ctype.startswith("DEC"):
                result.append("pos.eq.dec")
            elif ctype.startswith("GLON"):
                result.append("pos.galactic.lon")
            elif ctype.startswith("GLAT"):
                result.append("pos.galactic.lat")
            elif ctype.startswith("ELON"):
                result.append("pos.ecliptic.lon")
            elif ctype.startswith("ELAT"):
                result.append("pos.ecliptic.lat")
            elif ctype.startswith(("HLON", "SLON")):
                result.append("pos.bodyrc.lon")
            elif ctype.startswith(("HLAT", "SLAT")):
                result.append("pos.bodyrc.lat")
            elif ctype.startswith(
                ("FREQ", "ENER", "WAVN", "WAVE", "AWAV", "VELO", "VRAD", "VOPT", "ZOPT")
            ):
                result.append("em." + ctype.split("-")[0].lower())
            elif ctype.startswith(("TIME", "MJD", "JD")):
                result.append("time")
            else:
                result.append(None)
        return tuple(result)

    @property
    def spectral_axis_indices(self) -> tuple[int, ...]:
        """0-based indices of spectral-like world axes."""
        axes: list[int] = []
        for i in range(1, self.naxis + 1):
            ctype = str(self.wcs_params.get(f"CTYPE{i}", "")).strip()
            try:
                canonical_spectral_type(ctype)
            except ValueError:
                continue
            axes.append(i - 1)
        return tuple(axes)

    def _resolve_spectral_axis(self, axis: int | None = None) -> int:
        axes = self.spectral_axis_indices
        if axis is not None:
            idx = int(axis)
            if idx < 0:
                idx += self.naxis
            if idx < 0 or idx >= self.naxis:
                raise ValueError(f"spectral axis {axis!r} outside WCS axes")
            return idx
        if len(axes) != 1:
            raise ValueError(
                "axis must be specified when WCS has zero or multiple spectral axes"
            )
        return axes[0]

    def _rest_frequency_for_spectral(self) -> float | None:
        return rest_frequency_from_header(self.wcs_params)

    def spectral_to_frequency(
        self, value: Any, *, axis: int | None = None, unit: str | None = None
    ) -> Tensor:
        """Convert a WCS spectral world value to frequency in Hz."""
        idx = self._resolve_spectral_axis(axis)
        ctype = str(self.wcs_params.get(f"CTYPE{idx + 1}", ""))
        cunit = (
            unit
            if unit is not None
            else str(self.wcs_params.get(f"CUNIT{idx + 1}", ""))
        )
        return _spectral_to_frequency(
            value,
            ctype,
            unit=cunit or None,
            rest_frequency=self._rest_frequency_for_spectral(),
        )

    def frequency_to_spectral(
        self, frequency: Any, *, axis: int | None = None, unit: str | None = None
    ) -> Tensor:
        """Convert frequency in Hz to this WCS spectral world coordinate."""
        idx = self._resolve_spectral_axis(axis)
        ctype = str(self.wcs_params.get(f"CTYPE{idx + 1}", ""))
        cunit = (
            unit
            if unit is not None
            else str(self.wcs_params.get(f"CUNIT{idx + 1}", ""))
        )
        return _frequency_to_spectral(
            frequency,
            ctype,
            unit=cunit or None,
            rest_frequency=self._rest_frequency_for_spectral(),
        )

    @property
    def temporal_axis_indices(self) -> tuple[int, ...]:
        """0-based indices of temporal-like world axes."""
        axes: list[int] = []
        for i in range(1, self.naxis + 1):
            ctype = str(self.wcs_params.get(f"CTYPE{i}", "")).strip().upper()
            if ctype.startswith(("TIME", "MJD", "JD")):
                axes.append(i - 1)
        return tuple(axes)

    def _resolve_temporal_axis(self, axis: int | None = None) -> int:
        axes = self.temporal_axis_indices
        if axis is not None:
            idx = int(axis)
            if idx < 0:
                idx += self.naxis
            if idx < 0 or idx >= self.naxis:
                raise ValueError(f"temporal axis {axis!r} outside WCS axes")
            return idx
        if len(axes) != 1:
            raise ValueError(
                "axis must be specified when WCS has zero or multiple temporal axes"
            )
        return axes[0]

    def temporal_wcs(self, *, axis: int | None = None) -> TemporalWCS:
        """Return a ``TemporalWCS`` view using this WCS temporal metadata."""
        idx = self._resolve_temporal_axis(axis)
        header = dict(self.wcs_params)
        if "TIMEUNIT" not in header and f"CUNIT{idx + 1}" in header:
            header["TIMEUNIT"] = header[f"CUNIT{idx + 1}"]
        if "MJDREF" not in header and "MJDREFI" not in header:
            header["MJDREF"] = float(self.crval[idx].detach().cpu().item())
        return TemporalWCS(header)

    @property
    def celestial_frame(self) -> str:
        """Celestial reference frame from FITS metadata."""
        ctype1 = (
            str(self.wcs_params.get(f"CTYPE{self._celestial_lon_axis + 1}", ""))
            .strip()
            .upper()
        )
        ctype2 = (
            str(self.wcs_params.get(f"CTYPE{self._celestial_lat_axis + 1}", ""))
            .strip()
            .upper()
        )
        if ctype1.startswith("GLON") or ctype2.startswith("GLAT"):
            return "GALACTIC"
        if ctype1.startswith("ELON") or ctype2.startswith("ELAT"):
            return "ECLIPTIC"
        frame = self.wcs_params.get("RADESYS", self.wcs_params.get("RADECSYS", "ICRS"))
        return canonical_celestial_frame(str(frame))

    def _world_from_celestial_pair(self, lon: Tensor, lat: Tensor) -> Tensor:
        values = [lon if axis == self._celestial_lon_axis else lat for axis in range(2)]
        return torch.stack(values, dim=-1)

    def _pixel_affine_axes(self) -> tuple[int, int]:
        if len(self._celestial_axes) == 2:
            return self._celestial_axes
        return (0, 1)

    def _celestial_pair_from_world(self, world: Tensor) -> tuple[Tensor, Tensor]:
        return (
            world[..., self._celestial_lon_axis],
            world[..., self._celestial_lat_axis],
        )

    @property
    def celestial_equinox(self) -> float | None:
        """Julian equinox from FITS ``EQUINOX`` / ``EPOCH`` metadata."""
        raw = self.wcs_params.get("EQUINOX", self.wcs_params.get("EPOCH", None))
        return None if raw is None else float(raw)

    def precess_celestial(
        self, ra: Any, dec: Any, to_equinox: float, *, from_equinox: float | None = None
    ) -> Tuple[Tensor, Tensor]:
        """Precess celestial coordinates using WCS frame/equinox metadata."""
        frame = self.celestial_frame
        if frame == "FK4":
            source = (
                float(from_equinox)
                if from_equinox is not None
                else float(self.celestial_equinox or 1950.0)
            )
            return _convert_celestial(
                ra,
                dec,
                from_frame="FK4",
                to_frame="FK4",
                from_equinox=source,
                to_equinox=float(to_equinox),
            )
        if frame in {"GALACTIC", "GAL", "ECLIPTIC", "ECL"}:
            # Mean Galactic and ecliptic of J2000 are ICRS-fixed rotations.
            return ra, dec
        if frame in {"GCRS", "CIRS", "TETE", "ALTAZ", "ITRS", "HORIZON"}:
            raise NotImplementedError(
                f"{frame} needs obstime and EarthOrientation via wcs.transform; "
                "it is not a geometric equinox precession"
            )
        if frame not in {"ICRS", "FK5"}:
            raise NotImplementedError(f"precession is not implemented for {frame}")
        source = (
            float(from_equinox)
            if from_equinox is not None
            else float(self.celestial_equinox or 2000.0)
        )
        return _precess_fk5(ra, dec, from_equinox=source, to_equinox=float(to_equinox))

    def convert_celestial(
        self,
        lon: Any,
        lat: Any,
        to_frame: str,
        *,
        to_equinox: float | None = None,
    ) -> Tuple[Tensor, Tensor]:
        """Convert WCS celestial coordinates to a supported target frame."""
        return _convert_celestial(
            lon,
            lat,
            from_frame=self.celestial_frame,
            to_frame=to_frame,
            from_equinox=self.celestial_equinox,
            to_equinox=to_equinox,
        )
