"""Small, versioned instrument metadata used by torchsky operator models.

Values here are reproducibility inputs, not a replacement for calibration
reference files. NIRCam filter values use the STScI version-7 commissioning
throughputs (2025-01), and PSF widths use the STScI on-orbit PSF table.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "NIRCAM_METADATA_SOURCES",
    "NIRCAM_UPDEC_BANDS",
    "NIRCamImagingBand",
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
