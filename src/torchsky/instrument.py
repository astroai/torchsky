"""Small, versioned instrument metadata used by torchsky operator models.

Values here are reproducibility inputs, not a replacement for calibration
reference files. NIRCam filter values use the STScI version-7 commissioning
throughputs (2025-01), and PSF widths use the STScI on-orbit PSF table.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

__all__ = [
    "NIRCAM_METADATA_SOURCES",
    "NIRCAM_UPDEC_BANDS",
    "NIRCamImagingBand",
    "JWSTExposureTiming",
    "jwst_exposure_timing",
    "nircam_imaging_band",
]


NIRCAM_METADATA_SOURCES = {
    "filters": "https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-instrumentation/nircam-filters",
    "detectors": "https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-instrumentation/nircam-detector-overview",
    "psf": "https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-performance/nircam-point-spread-functions",
}
NIRCAM_UPDEC_BANDS = ("F090W", "F150W", "F200W", "F277W", "F356W", "F444W")


@dataclass(frozen=True)
class NIRCamImagingBand:
    """Official detector-averaged metadata for one NIRCam imaging band."""

    name: str
    channel: str
    detector_pixel_scale_arcsec: float
    pivot_wavelength_angstrom: float
    bandwidth_angstrom: float
    half_power_min_angstrom: float
    half_power_max_angstrom: float
    simulated_psf_fwhm_arcsec: float
    empirical_psf_fwhm_arcsec: float

    @property
    def detector_nyquist_period_arcsec(self) -> float:
        """Shortest detector-sampled spatial period, before dither recovery."""
        return 2.0 * self.detector_pixel_scale_arcsec

    @property
    def simulated_psf_fwhm_pixels(self) -> float:
        """STPSF FWHM in native detector pixels."""
        return self.simulated_psf_fwhm_arcsec / self.detector_pixel_scale_arcsec

    @property
    def empirical_psf_fwhm_pixels(self) -> float:
        """On-orbit empirical FWHM in native detector pixels."""
        return self.empirical_psf_fwhm_arcsec / self.detector_pixel_scale_arcsec


@dataclass(frozen=True)
class JWSTExposureTiming:
    """Canonical MJD UTC timing for one JWST exposure."""

    midpoint_mjd_utc: float
    start_mjd_utc: float | None
    end_mjd_utc: float | None
    midpoint_source: str

    @property
    def duration_seconds(self) -> float | None:
        if self.start_mjd_utc is None or self.end_mjd_utc is None:
            return None
        return (self.end_mjd_utc - self.start_mjd_utc) * 86_400.0

    def as_dict(self) -> dict[str, float | str | None]:
        return {
            "midpoint_mjd_utc": self.midpoint_mjd_utc,
            "start_mjd_utc": self.start_mjd_utc,
            "end_mjd_utc": self.end_mjd_utc,
            "duration_seconds": self.duration_seconds,
            "midpoint_source": self.midpoint_source,
        }


_NIRCAM_IMAGING_BANDS = {
    "F090W": NIRCamImagingBand(
        "F090W", "short", 0.031, 9020.0, 1940.0, 7950.0, 10050.0, 0.030, 0.033
    ),
    "F150W": NIRCamImagingBand(
        "F150W", "short", 0.031, 15010.0, 3180.0, 13310.0, 16680.0, 0.049, 0.050
    ),
    "F200W": NIRCamImagingBand(
        "F200W", "short", 0.031, 19880.0, 4610.0, 17550.0, 22270.0, 0.064, 0.066
    ),
    "F277W": NIRCamImagingBand(
        "F277W", "long", 0.063, 27760.0, 6720.0, 24220.0, 31310.0, 0.088, 0.092
    ),
    "F356W": NIRCamImagingBand(
        "F356W", "long", 0.063, 35650.0, 7860.0, 31360.0, 39810.0, 0.114, 0.116
    ),
    "F444W": NIRCamImagingBand(
        "F444W", "long", 0.063, 44020.0, 10240.0, 38810.0, 49820.0, 0.140, 0.145
    ),
}


def nircam_imaging_band(name: str) -> NIRCamImagingBand:
    """Return frozen NIRCam metadata, failing closed for unsupported filters."""
    key = str(name).strip().upper()
    try:
        return _NIRCAM_IMAGING_BANDS[key]
    except KeyError as exc:
        raise KeyError(f"unsupported NIRCam imaging band {name!r}") from exc


def jwst_exposure_timing(
    primary_header: Mapping[str, Any],
    science_header: Mapping[str, Any] | None = None,
    *,
    alias_tolerance_seconds: float = 0.002,
) -> JWSTExposureTiming:
    """Resolve official JWST exposure timing aliases without FITS I/O.

    ``MJD-BEG/MID/END`` live in the SCI header and are defined to match primary
    ``EXPSTART/EXPMID/EXPEND``. Midpoint aliases are preferred; their absence
    falls back to the arithmetic midpoint of start/end. Conflicting aliases
    fail closed instead of silently choosing one.
    """
    tolerance_seconds = float(alias_tolerance_seconds)
    if not math.isfinite(tolerance_seconds) or tolerance_seconds < 0.0:
        raise ValueError("alias_tolerance_seconds must be finite and non-negative")
    tolerance_days = tolerance_seconds / 86_400.0
    primary = {str(key).upper(): value for key, value in primary_header.items()}
    science = (
        {}
        if science_header is None
        else {str(key).upper(): value for key, value in science_header.items()}
    )

    def resolve(
        name: str, candidates: tuple[tuple[str, Mapping[str, Any]], ...]
    ) -> float | None:
        found: list[tuple[str, float]] = []
        for keyword, header in candidates:
            if keyword not in header or header[keyword] in (None, ""):
                continue
            try:
                value = float(header[keyword])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"JWST timing keyword {keyword} must be numeric"
                ) from exc
            if not math.isfinite(value):
                raise ValueError(f"JWST timing keyword {keyword} must be finite")
            found.append((keyword, value))
        if not found:
            return None
        reference_keyword, reference = found[0]
        for keyword, value in found[1:]:
            if abs(value - reference) > tolerance_days:
                raise ValueError(
                    f"conflicting JWST {name} aliases: {reference_keyword} and {keyword}"
                )
        return reference

    start = resolve(
        "start",
        (("MJD-BEG", science), ("EXPSTART", primary)),
    )
    midpoint = resolve(
        "midpoint",
        (("MJD-MID", science), ("EXPMID", primary)),
    )
    end = resolve(
        "end",
        (("MJD-END", science), ("EXPEND", primary)),
    )
    if midpoint is None:
        if start is None or end is None:
            raise ValueError(
                "JWST timing requires MJD-MID/EXPMID or both start and end"
            )
        midpoint = 0.5 * (start + end)
        midpoint_source = "derived_start_end"
    elif "MJD-MID" in science:
        midpoint_source = "SCI:MJD-MID"
    else:
        midpoint_source = "PRIMARY:EXPMID"

    if start is not None and end is not None and end < start:
        raise ValueError("JWST exposure end precedes start")
    if start is not None and midpoint < start - tolerance_days:
        raise ValueError("JWST exposure midpoint precedes start")
    if end is not None and midpoint > end + tolerance_days:
        raise ValueError("JWST exposure midpoint follows end")
    return JWSTExposureTiming(
        midpoint_mjd_utc=midpoint,
        start_mjd_utc=start,
        end_mjd_utc=end,
        midpoint_source=midpoint_source,
    )
