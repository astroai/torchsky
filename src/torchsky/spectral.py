from __future__ import annotations

"""
Spectral and data cube support for torchsky.

Implements 1D spectra with inverse variance/mask arrays and 3D data cubes
for IFU instruments and radio astronomy.

Note:
This module is an experimental helper layer built on top of torchsky FITS I/O.
It is not part of the core FITS read/write parity surface.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from .wcs import WCS


_SPECTRAL_TYPE_KEYS = ("WAVE", "FREQ", "VELO", "ENER")


def _normalize_spectral_type(ctype: str) -> str:
    upper = ctype.upper()
    for key in _SPECTRAL_TYPE_KEYS:
        if key in upper:
            return key
    return upper or "WAVE"


def _spectral_axis_length_from_header(
    header: Dict[str, Any],
    *,
    fits_axis: int,
    data_shape: Optional[Tuple[int, ...]],
) -> int:
    header_length = header.get(f"NAXIS{fits_axis}")
    if header_length is not None:
        return int(header_length)
    if data_shape is not None:
        tensor_axis = len(data_shape) - fits_axis
        return int(data_shape[tensor_axis])
    raise ValueError(f"Missing NAXIS{fits_axis} for spectral axis reconstruction")


def spectral_axis_from_header(
    header: Dict[str, Any],
    *,
    data_shape: Optional[Tuple[int, ...]] = None,
) -> Tuple[SpectralAxis, int]:
    """Build a linear SpectralAxis from FITS WCS header metadata."""
    naxis = int(header.get("NAXIS", len(data_shape) if data_shape is not None else 0))
    if naxis <= 0:
        raise ValueError("Header must define NAXIS or data_shape for spectral axis reconstruction")

    spectral_fits_axis = None
    spectral_type = "WAVE"
    spectral_unit = "Angstrom"
    for fits_axis in range(1, naxis + 1):
        ctype = str(header.get(f"CTYPE{fits_axis}", ""))
        if any(key in ctype.upper() for key in _SPECTRAL_TYPE_KEYS):
            spectral_fits_axis = fits_axis
            spectral_type = _normalize_spectral_type(ctype)
            spectral_unit = str(header.get(f"CUNIT{fits_axis}", "Angstrom"))
            break

    if spectral_fits_axis is None:
        spectral_fits_axis = naxis

    axis_length = _spectral_axis_length_from_header(
        header,
        fits_axis=spectral_fits_axis,
        data_shape=data_shape,
    )
    crval = float(header.get(f"CRVAL{spectral_fits_axis}", 0.0))
    cdelt = float(header.get(f"CDELT{spectral_fits_axis}", 1.0))
    crpix = float(header.get(f"CRPIX{spectral_fits_axis}", 1.0))
    pixels = torch.arange(axis_length, dtype=torch.float64)
    values = crval + cdelt * (pixels - (crpix - 1.0))
    tensor_axis_index = naxis - spectral_fits_axis
    return (
        SpectralAxis(values=values, unit=spectral_unit, type=spectral_type),
        tensor_axis_index,
    )


def spectral_bin_edges_from_header(
    header: Dict[str, Any],
    *,
    data_shape: Optional[Tuple[int, ...]] = None,
) -> Tuple[torch.Tensor, int]:
    """Build linear spectral bin edges from FITS WCS header metadata."""
    axis, tensor_axis_index = spectral_axis_from_header(header, data_shape=data_shape)
    values = axis.values.to(dtype=torch.float64)
    if values.numel() == 1:
        return torch.tensor([values[0], values[0]], dtype=torch.float64), tensor_axis_index
    diffs = values[1:] - values[:-1]
    if not bool(torch.allclose(diffs, torch.full_like(diffs, diffs[0]))):
        raise ValueError("Nonlinear spectral headers are not yet supported for bin-edge reconstruction")
    step = diffs[0]
    mid = 0.5 * (values[1:] + values[:-1])
    edges = torch.empty((values.numel() + 1,), dtype=torch.float64)
    edges[1:-1] = mid
    edges[0] = values[0] - 0.5 * step
    edges[-1] = values[-1] + 0.5 * step
    return edges, tensor_axis_index


@dataclass
class SpectralAxis:
    """Represents a spectral axis with wavelength/frequency information."""

    values: torch.Tensor  # Wavelength/frequency values
    unit: str  # Units (e.g., 'Angstrom', 'Hz', 'keV')
    type: str  # Type ('WAVE', 'FREQ', 'ENER', 'VELO')
    rest_frequency: Optional[float] = None
    redshift: Optional[float] = None

    def to_wavelength(self) -> torch.Tensor:
        """Convert to wavelength in Angstroms."""
        if self.type == "WAVE":
            if self.unit.lower() in ["angstrom", "a"]:
                return self.values
            elif self.unit.lower() in ["nm", "nanometer"]:
                return self.values * 10.0
            elif self.unit.lower() in ["um", "micrometer", "micron"]:
                return self.values * 10000.0
        elif self.type == "FREQ":
            # c = 2.998e18 Angstrom/s
            c_angstrom_per_s = 2.998e18
            return c_angstrom_per_s / self.values

        raise ValueError(
            f"Unsupported spectral conversion to wavelength for type={self.type!r}, unit={self.unit!r}. "
            "Supported: WAVE[A, Angstrom, nm, um] and FREQ[Hz/GHz/MHz]."
        )

    def to_frequency(self) -> torch.Tensor:
        """Convert to frequency in Hz."""
        if self.type == "FREQ":
            if self.unit.lower() == "hz":
                return self.values
            elif self.unit.lower() == "ghz":
                return self.values * 1e9
            elif self.unit.lower() == "mhz":
                return self.values * 1e6
        elif self.type == "WAVE":
            wavelength_m = self.to_wavelength() * 1e-10  # Convert to meters
            c_m_per_s = 2.998e8
            return c_m_per_s / wavelength_m

        raise ValueError(
            f"Unsupported spectral conversion to frequency for type={self.type!r}, unit={self.unit!r}. "
            "Supported: FREQ[Hz/GHz/MHz] and WAVE[A, Angstrom, nm, um]."
        )


@dataclass
class Spectrum1D:
    """1D spectrum with optional inverse variance and mask arrays."""

    flux: torch.Tensor  # Shape: (n_wavelength,)
    spectral_axis: SpectralAxis
    ivar: Optional[torch.Tensor] = None  # Inverse variance
    mask: Optional[torch.Tensor] = None  # Boolean mask (True = good)
    header: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate spectrum data."""
        if self.flux.ndim != 1:
            raise ValueError("Flux must be 1D")

        n_wave = len(self.flux)

        if len(self.spectral_axis.values) != n_wave:
            raise ValueError("Spectral axis length must match flux length")

        if self.ivar is not None and self.ivar.shape != self.flux.shape:
            raise ValueError("Inverse variance shape must match flux shape")

        if self.mask is not None and self.mask.shape != self.flux.shape:
            raise ValueError("Mask shape must match flux shape")

    @property
    def wavelength(self) -> torch.Tensor:
        """Get wavelength values in Angstroms."""
        return self.spectral_axis.to_wavelength()

    @property
    def frequency(self) -> torch.Tensor:
        """Get frequency values in Hz."""
        return self.spectral_axis.to_frequency()

    @property
    def error(self) -> Optional[torch.Tensor]:
        """Get error from inverse variance."""
        if self.ivar is not None:
            return torch.sqrt(1.0 / torch.clamp(self.ivar, min=1e-30))
        return None

    def apply_mask(self) -> "Spectrum1D":
        """Return spectrum with mask applied."""
        if self.mask is None:
            return self

        good = self.mask.bool()
        return Spectrum1D(
            flux=self.flux[good],
            spectral_axis=SpectralAxis(
                values=self.spectral_axis.values[good],
                unit=self.spectral_axis.unit,
                type=self.spectral_axis.type,
                rest_frequency=self.spectral_axis.rest_frequency,
                redshift=self.spectral_axis.redshift,
            ),
            ivar=self.ivar[good] if self.ivar is not None else None,
            mask=self.mask[good],
            header=self.header,
        )

    def resample(self, new_wavelength: torch.Tensor) -> "Spectrum1D":
        """Resample spectrum to new wavelength grid."""
        current_wave = self.wavelength

        # Simple linear interpolation
        flux_interp = torch.nn.functional.interpolate(
            self.flux.unsqueeze(0).unsqueeze(0),
            size=len(new_wavelength),
            mode="linear",
            align_corners=True,
        ).squeeze()

        # Alternative: manual interpolation if needed
        if len(flux_interp) != len(new_wavelength):
            # Fallback to manual interpolation
            flux_interp = torch.zeros_like(new_wavelength)
            for i, wave in enumerate(new_wavelength):
                # Find closest wavelengths
                diffs = torch.abs(current_wave - wave)
                idx = torch.argmin(diffs)
                flux_interp[i] = self.flux[idx]

        ivar_interp = None
        if self.ivar is not None:
            # Manual interpolation for ivar
            ivar_interp = torch.zeros_like(new_wavelength)
            for i, wave in enumerate(new_wavelength):
                diffs = torch.abs(current_wave - wave)
                idx = torch.argmin(diffs)
                ivar_interp[i] = self.ivar[idx]

        new_spectral_axis = SpectralAxis(
            values=new_wavelength,
            unit="Angstrom",
            type="WAVE",
            rest_frequency=self.spectral_axis.rest_frequency,
            redshift=self.spectral_axis.redshift,
        )

        return Spectrum1D(
            flux=flux_interp,
            spectral_axis=new_spectral_axis,
            ivar=ivar_interp,
            mask=None,  # Mask not preserved in resampling
            header=self.header,
        )


@dataclass
class DataCube:
    """3D data cube for IFU instruments and radio astronomy."""

    data: torch.Tensor  # Shape: (n_spectral, n_y, n_x) or (n_x, n_y, n_spectral)
    spectral_axis: SpectralAxis
    spatial_wcs: Optional[WCS] = None
    ivar: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None
    header: Optional[Dict[str, Any]] = None
    spectral_axis_index: int = 0  # Which axis is spectral (0, 1, or 2)

    def __post_init__(self):
        """Validate cube data."""
        if self.data.ndim != 3:
            raise ValueError("Data must be 3D")

        spectral_size = self.data.shape[self.spectral_axis_index]
        if len(self.spectral_axis.values) != spectral_size:
            raise ValueError("Spectral axis length must match data spectral dimension")

        if self.ivar is not None and self.ivar.shape != self.data.shape:
            raise ValueError("Inverse variance shape must match data shape")

        if self.mask is not None and self.mask.shape != self.data.shape:
            raise ValueError("Mask shape must match data shape")

    @property
    def shape(self) -> Tuple[int, ...]:
        """Get cube shape."""
        return tuple(self.data.shape)

    @property
    def n_spectral(self) -> int:
        """Number of spectral channels."""
        return self.data.shape[self.spectral_axis_index]

    @property
    def spatial_shape(self) -> Tuple[int, int]:
        """Get spatial dimensions (y, x)."""
        shape = list(self.data.shape)
        shape.pop(self.spectral_axis_index)
        return tuple(shape)

    def extract_spectrum(self, y: int, x: int) -> Spectrum1D:
        """Extract 1D spectrum at spatial position (y, x)."""
        if self.spectral_axis_index == 0:
            flux = self.data[:, y, x]
            ivar = self.ivar[:, y, x] if self.ivar is not None else None
            mask = self.mask[:, y, x] if self.mask is not None else None
        elif self.spectral_axis_index == 1:
            flux = self.data[x, :, y]  # Note: x,y swapped for axis=1
            ivar = self.ivar[x, :, y] if self.ivar is not None else None
            mask = self.mask[x, :, y] if self.mask is not None else None
        elif self.spectral_axis_index == 2:
            flux = self.data[y, x, :]
            ivar = self.ivar[y, x, :] if self.ivar is not None else None
            mask = self.mask[y, x, :] if self.mask is not None else None
        else:
            raise ValueError("Invalid spectral_axis_index")

        return Spectrum1D(
            flux=flux,
            spectral_axis=self.spectral_axis,
            ivar=ivar,
            mask=mask,
            header=self.header,
        )

    def collapse_spectral(
        self,
        method: str = "mean",
        wavelength_range: Optional[Tuple[float, float]] = None,
    ) -> torch.Tensor:
        """Collapse cube along spectral axis."""
        data = self.data

        # Apply wavelength range if specified
        if wavelength_range is not None:
            wave = self.spectral_axis.to_wavelength()
            mask = (wave >= wavelength_range[0]) & (wave <= wavelength_range[1])

            if self.spectral_axis_index == 0:
                data = data[mask, :, :]
            elif self.spectral_axis_index == 1:
                data = data[:, mask, :]
            elif self.spectral_axis_index == 2:
                data = data[:, :, mask]

        # Collapse along spectral axis
        if method == "mean":
            return torch.mean(data, dim=self.spectral_axis_index)
        elif method == "sum":
            return torch.sum(data, dim=self.spectral_axis_index)
        elif method == "median":
            return torch.median(data, dim=self.spectral_axis_index).values
        elif method == "max":
            return torch.max(data, dim=self.spectral_axis_index).values
        else:
            raise ValueError(f"Unknown collapse method: {method}")

    def extract_slice(
        self, wavelength: float, width: Optional[float] = None
    ) -> torch.Tensor:
        """Extract 2D slice at specific wavelength."""
        wave = self.spectral_axis.to_wavelength()

        if width is None:
            # Find closest wavelength
            idx = torch.argmin(torch.abs(wave - wavelength))
            if self.spectral_axis_index == 0:
                return self.data[idx, :, :]
            elif self.spectral_axis_index == 1:
                return self.data[:, idx, :]
            elif self.spectral_axis_index == 2:
                return self.data[:, :, idx]
        else:
            # Average over wavelength range
            mask = torch.abs(wave - wavelength) <= width / 2
            if self.spectral_axis_index == 0:
                return torch.mean(self.data[mask, :, :], dim=0)
            elif self.spectral_axis_index == 1:
                return torch.mean(self.data[:, mask, :], dim=1)
            elif self.spectral_axis_index == 2:
                return torch.mean(self.data[:, :, mask], dim=2)

    def to(self, device: torch.device) -> "DataCube":
        """Move cube to device."""
        return DataCube(
            data=self.data.to(device),
            spectral_axis=SpectralAxis(
                values=self.spectral_axis.values.to(device),
                unit=self.spectral_axis.unit,
                type=self.spectral_axis.type,
                rest_frequency=self.spectral_axis.rest_frequency,
                redshift=self.spectral_axis.redshift,
            ),
            spatial_wcs=self.spatial_wcs,
            ivar=self.ivar.to(device) if self.ivar is not None else None,
            mask=self.mask.to(device) if self.mask is not None else None,
            header=self.header,
            spectral_axis_index=self.spectral_axis_index,
        )


class SpectralReader:
    """Reader for spectral data from FITS files."""

    @staticmethod
    def read_spectrum_1d(
        filepath: str,
        hdu: int = 0,
        flux_col: str = "FLUX",
        wave_col: str = "WAVELENGTH",
        ivar_col: Optional[str] = "IVAR",
        mask_col: Optional[str] = "MASK",
    ) -> Spectrum1D:
        """Read 1D spectrum from FITS table."""
        import torchsky

        with torchsky.open(filepath) as hdul:
            table_hdu = hdul[hdu]

            # Read required columns
            flux = table_hdu.data[flux_col]
            wavelength = table_hdu.data[wave_col]

            # Read optional columns
            ivar = (
                table_hdu.data[ivar_col]
                if ivar_col and ivar_col in table_hdu.data.columns
                else None
            )
            mask = (
                table_hdu.data[mask_col]
                if mask_col and mask_col in table_hdu.data.columns
                else None
            )

            # Create spectral axis
            spectral_axis = SpectralAxis(
                values=wavelength,
                unit="Angstrom",  # Default, should be read from header
                type="WAVE",
            )

            return Spectrum1D(
                flux=flux,
                spectral_axis=spectral_axis,
                ivar=ivar,
                mask=mask,
                header=dict(table_hdu.header),
            )

    @staticmethod
    def read_data_cube(filepath: str, hdu: int = 0) -> DataCube:
        """Read 3D data cube from FITS image."""
        import torchsky

        with torchsky.open(filepath) as hdul:
            image_hdu = hdul[hdu]
            data = image_hdu.to_tensor()
            header = dict(image_hdu.header)

            spectral_axis, spectral_axis_index = spectral_axis_from_header(
                header,
                data_shape=tuple(int(v) for v in data.shape),
            )

            # Create spatial WCS if available
            spatial_wcs = None
            try:
                spatial_wcs = WCS.from_header(header)
            except Exception:
                pass  # WCS creation failed, continue without it

            return DataCube(
                data=data,
                spectral_axis=spectral_axis,
                spatial_wcs=spatial_wcs,
                header=header,
                spectral_axis_index=spectral_axis_index,
            )
