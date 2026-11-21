"""Small, versioned instrument metadata used by torchsky operator models.

Values here are reproducibility inputs, not a replacement for calibration
reference files. NIRCam filter values use the STScI version-7 commissioning
throughputs (2025-01), and PSF widths use the STScI on-orbit PSF table.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import IntFlag
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor

    from .map.quality import DQPolicy

__all__ = [
    "NIRCAM_METADATA_SOURCES",
    "NIRCAM_UPDEC_BANDS",
    "ACSWFCDataQuality",
    "InstrumentDescription",
    "JWSTDataQuality",
    "JWSTExposureTiming",
    "NIRCamImagingBand",
    "WFC3IRDataQuality",
    "instrument_description",
    "jwst_exposure_timing",
    "nircam_imaging_band",
]


NIRCAM_METADATA_SOURCES = {
    "filters": "https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-instrumentation/nircam-filters",
    "detectors": "https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-instrumentation/nircam-detector-overview",
    "psf": "https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-performance/nircam-point-spread-functions",
}
NIRCAM_UPDEC_BANDS = ("F090W", "F150W", "F200W", "F277W", "F356W", "F444W")


class JWSTDataQuality(IntFlag):
    """JWST pixel data-quality (``DQ``) bits, as set by the calibration pipeline.

    Each member is ``1 << bit``; combine with ``|`` and pass the result as
    ``DQPolicy(reject_bits=JWSTDataQuality.DO_NOT_USE | JWSTDataQuality.SATURATED)``.
    Source: the JWST pipeline documentation, "Data Quality Flags" (``jwst`` and
    ``stdatamodels.jwst.datamodels.dqflags.pixel``), also summarised in the JWST User
    Documentation, "Understanding Data Quality Flags". The pipeline writes the plane
    as ``uint32``.

    Not included, because their meaning is not certain from the sources the author
    could consult offline: bit 7 (value 128, listed as reserved) and any bit added to
    the pipeline after this table was written. A bit that is absent here is simply not
    named; it is still handled correctly as a number.
    """

    DO_NOT_USE = 1 << 0
    SATURATED = 1 << 1
    JUMP_DET = 1 << 2
    DROPOUT = 1 << 3
    OUTLIER = 1 << 4
    PERSISTENCE = 1 << 5
    AD_FLOOR = 1 << 6
    UNRELIABLE_ERROR = 1 << 8
    NON_SCIENCE = 1 << 9
    DEAD = 1 << 10
    HOT = 1 << 11
    WARM = 1 << 12
    LOW_QE = 1 << 13
    RC = 1 << 14
    TELEGRAPH = 1 << 15
    NONLINEAR = 1 << 16
    BAD_REF_PIXEL = 1 << 17
    NO_FLAT_FIELD = 1 << 18
    NO_GAIN_VALUE = 1 << 19
    NO_LIN_CORR = 1 << 20
    NO_SAT_CHECK = 1 << 21
    UNRELIABLE_BIAS = 1 << 22
    UNRELIABLE_DARK = 1 << 23
    UNRELIABLE_SLOPE = 1 << 24
    UNRELIABLE_FLAT = 1 << 25
    OPEN = 1 << 26
    ADJ_OPEN = 1 << 27
    UNRELIABLE_RESET = 1 << 28
    MSA_FAILED_OPEN = 1 << 29
    OTHER_BAD_PIXEL = 1 << 30
    REFERENCE_PIXEL = 1 << 31


class WFC3IRDataQuality(IntFlag):
    """HST WFC3/IR ``DQ`` bits that are certain, for ``_flt`` and ``_ima`` files.

    Source: the WFC3 Data Handbook (STScI), chapter "WFC3 Data Calibration" (the
    data-quality flags table for the IR channel). Only bits whose meaning the author
    is sure of are named; combine with ``|`` for ``DQPolicy(reject_bits=...)``.

    Left out on purpose (not certain enough to name): 8, 32, 64, 128 (zero-read,
    unstable, warm and reference-pixel flags, whose order differs between the UVIS and
    IR tables), 1024, 2048 (zero-read signal), 4096 and above (cosmic-ray and
    related flags). Use the number directly for those after checking the handbook.
    """

    RS_DECODE_ERROR = 1 << 0
    DATA_LOST = 1 << 1
    BAD_DETECTOR_PIXEL = 1 << 2
    HOT_PIXEL = 1 << 4
    FULL_WELL_SATURATION = 1 << 8
    BAD_FLAT = 1 << 9


class ACSWFCDataQuality(IntFlag):
    """HST ACS/WFC ``DQ`` bits that are certain, for ``_flt`` and ``_flc`` files.

    Source: the ACS Data Handbook (STScI), chapter "ACS Data Calibration" (the
    data-quality flags table). Only bits whose meaning the author is sure of are
    named; combine with ``|`` for ``DQPolicy(reject_bits=...)``.

    Left out on purpose (not certain enough to name): 8, 32, 64, 128 (aperture mask,
    warm-pixel, bad column and bias flags), 1024 (charge trap), 4096, 8192 and 16384 (the
    cosmic-ray and related flags: which of them calacs and AstroDrizzle set differs
    by product). Use the number directly for those after checking the handbook.
    """

    RS_DECODE_ERROR = 1 << 0
    DATA_LOST = 1 << 1
    BAD_DETECTOR_PIXEL = 1 << 2
    HOT_PIXEL = 1 << 4
    FULL_WELL_SATURATION = 1 << 8
    BAD_FLAT = 1 << 9
    AD_SATURATION = 1 << 11


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


# ---------------------------------------------------------------------------------------------
# Instrument descriptions for exposure-level fitting
# ---------------------------------------------------------------------------------------------

_C_ANGSTROM_PER_S = 2.99792458e18
_ARCSEC2_TO_SR = (math.pi / 648000.0) ** 2
_NIRCAM_CHANNEL = {
    **{f"NRC{module}{index}": "NIRCAM_SW" for module in "AB" for index in "1234"},
    **{f"NRC{module}LONG": "NIRCAM_LW" for module in "AB"},
}
_UNIT_PER_SR = re.compile(r"^\s*MJY\s*(/|\s)\s*SR\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class InstrumentDescription:
    """What the exposure-level fit and the exposure loader assume about one camera.

    Build one with :func:`instrument_description`. The description holds facts only (no file is
    opened). Attributes:

    * ``name``: ``"NIRCAM"``, ``"WFC3_IR"`` or ``"ACS_WFC"``; ``telescope``; ``diameter_m``,
      the telescope aperture diameter [m] (used for the diffraction scale of the kernel
      start).
    * ``detectors``: detector names the camera produces, as the ``DETECTOR`` header card
      spells them (NIRCam); the two ACS chips are named ``ACS_WFC1`` and ``ACS_WFC2`` after their
      ``CCDCHIP`` card; ``channel_of(detector)`` gives the channel, and ``pixel_scale_arcsec[channel]`` the
      nominal pixel scale [arcsec] of that channel.
    * ``science_unit``: unit of the science plane in the calibrated file. The factor to
      MJy/sr is ``sky_factor(...)``: 1 for ``"MJy/sr"`` (JWST ``_cal``), and for the HST unit
      ``"ELECTRONS"`` or ``"ELECTRONS/S"`` ``PHOTFNU * 1e-6 / (Omega_nominal * t_unit)`` with
      ``PHOTFNU`` [Jy s / electron] from the header (or ``1e23 PHOTFLAM PHOTPLAM**2 / c``
      when absent), ``Omega_nominal = IDCSCALE**2`` [sr], and ``t_unit = EXPTIME`` for
      ``ELECTRONS`` and 1 for ``ELECTRONS/S``. The result is surface brightness per
      *nominal* pixel area (the pixel-area map is not applied).
    * ``dq_flags``, ``dq_bit_width`` and ``dq_ignore_bits``: the bit table of the data-quality
      plane (from this module), the width of the plane in bits (the bits above it are not
      read), and the bits that do **not** reject a pixel by default. ``dq_policy()`` returns
      the matching :class:`~torchsky.map.quality.DQPolicy`.
    * ``detector_terms``: per-exposure detector terms the fit may add (``"stripes"``: row and
      column offsets).
    * ``bandpass_prefix``: the filter-profile identifier is the prefix plus the filter name;
      ``filter_curves`` optionally maps a filter name to ``(wavelength_um, transmission)``
      tensors (see :meth:`with_filter_curves`); nothing is fetched from the network.
    * ``science_hdu``, ``error_hdu``, ``quality_hdu`` and ``chip_extvers``: extension names
      and the ``EXTVER`` of each chip (one observation per chip).
    * ``wcs_policy``: ``"header"`` uses the science header's own solution; ``"apriori"`` (HST)
      uses the alternate solution written from the guide-star catalogue, as the JWST
      ``_cal`` files do, and never an alternate solution fitted to other images.
    * ``min_valid_pixels``: a cut with fewer usable pixels is dropped.
    """

    name: str
    telescope: str
    diameter_m: float
    pixel_scale_arcsec: Mapping[str, float]
    detectors: tuple[str, ...]
    science_unit: str
    dq_flags: type[IntFlag]
    dq_bit_width: int
    dq_ignore_bits: int
    detector_terms: tuple[str, ...]
    bandpass_prefix: str
    chip_extvers: tuple[int, ...] = (1,)
    science_hdu: str = "SCI"
    error_hdu: str = "ERR"
    quality_hdu: str = "DQ"
    wcs_policy: str = "header"
    min_valid_pixels: int = 2000
    filter_curves: Mapping[str, tuple[Tensor, Tensor]] | None = None
    notes: tuple[str, ...] = ()

    def channel_of(self, detector: str) -> str:
        """Channel (kernel and render grid are per channel) of a detector name."""
        if self.name == "NIRCAM":
            return _NIRCAM_CHANNEL.get(str(detector).strip().upper(), str(detector))
        return self.name

    def bandpass_id(self, band: str) -> str:
        """Filter-profile identifier of ``band`` (prefix + name)."""
        return f"{self.bandpass_prefix}{band}"

    def dq_policy(self, ignore_bits: int | None = None) -> DQPolicy:
        """:class:`~torchsky.map.quality.DQPolicy` that rejects every bit of the plane except ``ignore_bits``.

        ``None`` uses ``dq_ignore_bits``. The policy reads only the low ``dq_bit_width``
        bits (16 for HST), as the plane is an unsigned bit field.
        """
        from .map.quality import DQPolicy

        ignore = self.dq_ignore_bits if ignore_bits is None else int(ignore_bits)
        full = (1 << self.dq_bit_width) - 1
        return DQPolicy(reject_bits=full & ~ignore)

    def sky_factor(
        self,
        primary: Mapping[str, Any],
        science: Mapping[str, Any],
        *,
        nominal_scale_arcsec: float | None = None,
    ) -> tuple[float, dict[str, Any]]:
        """Factor from the science plane to MJy/sr, and how it was obtained.

        Reads ``BUNIT``, ``EXPTIME``, ``PHOTFNU``, ``PHOTFLAM``, ``PHOTPLAM`` (science header
        first, then primary) and, for HST, ``IDCSCALE`` (or ``nominal_scale_arcsec``, or the
        nominal scale of the channel). Raises ``ValueError`` for a unit the description does
        not know (fail closed).
        """

        def get(key: str, default: Any = None) -> Any:
            for header in (science, primary):
                if key in header and header[key] not in (None, ""):
                    return header[key]
            return default

        unit = str(get("BUNIT", "")).strip().upper()
        if self.science_unit == "MJy/sr":
            if not _UNIT_PER_SR.match(unit):
                raise ValueError(
                    f"{self.name}: BUNIT {unit!r}, expected MJy/sr (calibrated file)"
                )
            return 1.0, {"bunit": unit, "source": "file is MJy/sr"}
        if unit == "ELECTRONS/S":
            t_unit = 1.0
        elif unit == "ELECTRONS":
            exptime = float(get("EXPTIME", 0.0) or 0.0)
            if exptime <= 0:
                raise ValueError("BUNIT ELECTRONS without a positive EXPTIME")
            t_unit = exptime
        else:
            raise ValueError(
                f"{self.name}: unsupported BUNIT {get('BUNIT', '')!r} (ELECTRONS or ELECTRONS/S)"
            )
        photfnu = get("PHOTFNU")
        source = "PHOTFNU"
        if photfnu is None:
            photflam, photplam = get("PHOTFLAM"), get("PHOTPLAM")
            if photflam is None or photplam is None:
                raise ValueError(
                    "neither PHOTFNU nor PHOTFLAM and PHOTPLAM in the header"
                )
            photfnu = 1e23 * float(photflam) * float(photplam) ** 2 / _C_ANGSTROM_PER_S
            source = "1e23 PHOTFLAM PHOTPLAM^2 / c"
        scale = nominal_scale_arcsec
        scale_source = "argument"
        if scale is None:
            scale = get("IDCSCALE")
            scale_source = "IDCSCALE"
        if scale is None:
            scale = self.pixel_scale_arcsec[self.channel_of("")]
            scale_source = "nominal table"
        omega = float(scale) ** 2 * _ARCSEC2_TO_SR
        factor = float(photfnu) * 1e-6 / (omega * t_unit)
        return factor, {
            "bunit": unit,
            "photfnu_jy_s_per_e": float(photfnu),
            "photfnu_source": source,
            "omega_nominal_sr": omega,
            "t_unit_s": t_unit,
            "nominal_scale_arcsec": float(scale),
            "nominal_scale_source": scale_source,
        }

    def with_filter_curves(
        self, curves: Mapping[str, tuple[Tensor, Tensor]]
    ) -> InstrumentDescription:
        """Copy with filter curves ``{band: (wavelength_um, transmission)}`` (1D tensors of equal length, at least 2 points)."""
        checked: dict[str, tuple[Tensor, Tensor]] = {}
        for band, pair in curves.items():
            wavelength, transmission = pair
            if (
                wavelength.ndim != 1
                or wavelength.shape != transmission.shape
                or wavelength.numel() < 2
            ):
                raise ValueError(
                    f"filter curve {band!r} needs two 1D tensors of equal length >= 2"
                )
            checked[str(band).strip().upper()] = (wavelength, transmission)
        return replace(self, filter_curves=checked)


# HST defaults follow drizzlepac's final bits for these detectors (WFC3/IR 528 = hot pixel + bad flat;
# ACS 80 = hot pixel + warm pixel, bit 64, which the ACS table of this module does not name) without the
# saturated bit: a saturated pixel is not linear and the fit has no non-linear detector term.
_WFC3_IR_IGNORE = int(WFC3IRDataQuality.HOT_PIXEL | WFC3IRDataQuality.BAD_FLAT)
_ACS_WFC_IGNORE = int(ACSWFCDataQuality.HOT_PIXEL) | 64

_INSTRUMENTS: dict[str, InstrumentDescription] = {
    "NIRCAM": InstrumentDescription(
        name="NIRCAM",
        telescope="JWST",
        diameter_m=6.5,
        pixel_scale_arcsec={"NIRCAM_SW": 0.031, "NIRCAM_LW": 0.063},
        detectors=tuple(_NIRCAM_CHANNEL),
        science_unit="MJy/sr",
        dq_flags=JWSTDataQuality,
        dq_bit_width=32,
        dq_ignore_bits=0,
        detector_terms=("stripes",),
        bandpass_prefix="JWST/NIRCam.",
        notes=(
            "saturation: DQ bit 2; the pipeline corrects non-linearity below saturation",
        ),
    ),
    "WFC3_IR": InstrumentDescription(
        name="WFC3_IR",
        telescope="HST",
        diameter_m=2.4,
        pixel_scale_arcsec={"WFC3_IR": 0.12825},
        detectors=("WFC3_IR",),
        science_unit="ELECTRONS/S",
        dq_flags=WFC3IRDataQuality,
        dq_bit_width=16,
        dq_ignore_bits=_WFC3_IR_IGNORE,
        detector_terms=(),
        bandpass_prefix="HST/WFC3_IR.",
        wcs_policy="apriori",
        notes=("SIP distortion only, no look-up tables",),
    ),
    "ACS_WFC": InstrumentDescription(
        name="ACS_WFC",
        telescope="HST",
        diameter_m=2.4,
        pixel_scale_arcsec={"ACS_WFC": 0.05},
        detectors=("ACS_WFC1", "ACS_WFC2"),
        science_unit="ELECTRONS",
        dq_flags=ACSWFCDataQuality,
        dq_bit_width=16,
        dq_ignore_bits=_ACS_WFC_IGNORE,
        detector_terms=(),
        bandpass_prefix="HST/ACS_WFC.",
        chip_extvers=(1, 2),
        wcs_policy="apriori",
        notes=(
            "two chips are two observations, each with its own detector terms",
            "look-up-table distortion corrections (D2IMARR, WCSDVARR) are not applied by torchsky's WCS",
        ),
    ),
}
_ALIASES = {
    "WFC3": "WFC3_IR",
    "WFC3/IR": "WFC3_IR",
    "ACS": "ACS_WFC",
    "ACS/WFC": "ACS_WFC",
}


def instrument_description(name: str) -> InstrumentDescription:
    """Return the description of ``"NIRCAM"``, ``"WFC3_IR"`` or ``"ACS_WFC"`` (aliases ``WFC3``, ``WFC3/IR``, ``ACS``, ``ACS/WFC``).

    Raises:
        KeyError: The name is not known.

    Examples:
        >>> d = instrument_description("wfc3/ir")
        >>> d.name, d.diameter_m, d.dq_bit_width
        ('WFC3_IR', 2.4, 16)
    """
    key = str(name).strip().upper()
    key = _ALIASES.get(key, key)
    try:
        return _INSTRUMENTS[key]
    except KeyError as exc:
        raise KeyError(
            f"unknown instrument {name!r}; known: {sorted(_INSTRUMENTS)}"
        ) from exc
