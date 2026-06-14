"""Spectral and data cube support for torchsky.

Implements 1D spectra with inverse variance/mask arrays and 3D data cubes
for IFU instruments and radio astronomy.

Note:
This module is an experimental helper layer built on top of torchsky FITS I/O.
It is not part of the core FITS read/write parity surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch

from .wcs import WCS


_SPECTRAL_TYPE_KEYS = ("WAVE", "FREQ", "VELO", "ENER")
_SPEED_OF_LIGHT_M_PER_S = 299_792_458.0
_SPEED_OF_LIGHT_ANGSTROM_PER_S = _SPEED_OF_LIGHT_M_PER_S * 1.0e10


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
    tables: Optional[Any] = None,
) -> Tuple["SpectralAxis", int]:
    """Build a linear or tabular SpectralAxis from FITS WCS header metadata."""
    naxis = int(header.get("NAXIS", len(data_shape) if data_shape is not None else 0))
    if naxis <= 0:
        raise ValueError(
            "Header must define NAXIS or data_shape for spectral axis reconstruction"
        )

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

    # Reconstruct coordinate values
    ctype = str(header.get(f"CTYPE{spectral_fits_axis}", ""))
    if ctype.upper().endswith("-TAB"):
        from .wcs.tabular import tabular_axes_from_fits

        if tables is None:
            raise ValueError("tabular spectral axis requires coordinate tables")
        axes = tabular_axes_from_fits(header, tables)
        if spectral_fits_axis not in axes:
            raise ValueError("tabular spectral axis mismatch")
        tab_axis = axes[spectral_fits_axis]
        values = tab_axis.world.to(dtype=torch.float64)
        spectral_unit = tab_axis.unit or spectral_unit
    else:
        header_upper = {str(k).upper(): v for k, v in header.items()}
        custom_key_prefix = spectral_type.upper()
        if f"{custom_key_prefix}0" in header_upper:
            custom_vals = []
            for i in range(axis_length):
                k = f"{custom_key_prefix}{i}"
                if k in header_upper:
                    custom_vals.append(float(header_upper[k]))
                else:
                    custom_vals.append(crval + cdelt * (i - (crpix - 1.0)))
            values = torch.tensor(custom_vals, dtype=torch.float64)
        else:
            pixels = torch.arange(axis_length, dtype=torch.float64)
            values = crval + cdelt * (pixels - (crpix - 1.0))

    tensor_axis_index = naxis - spectral_fits_axis
    return (
        SpectralAxis(values=values, unit=spectral_unit, type=spectral_type),
        tensor_axis_index,
    )


def _strip_axis3_and_naxis_keys(header: Dict[str, Any]) -> Dict[str, Any]:
    """Remove NAXIS* and axis-3 WCS keys so a 2D spatial header can become cube axis 1–2."""
    out: Dict[str, Any] = {}
    for key, value in header.items():
        ku = key.upper()
        if ku == "NAXIS" or ku.startswith("NAXIS"):
            continue
        if ku.startswith("CTYPE3"):
            continue
        if ku.startswith("CUNIT3"):
            continue
        if ku.startswith("CRVAL3"):
            continue
        if ku.startswith("CRPIX3"):
            continue
        if ku.startswith("CDELT3"):
            continue
        if ku.startswith("CROTA3"):
            continue
        if ku.startswith("PC03") or ku.startswith("PC3_") or ku.startswith("PC_3"):
            continue
        if ku.startswith("CD3_") or ku.startswith("CD_3"):
            continue
        out[key] = value
    return out


def write_cube_fits(
    path: str,
    cube: torch.Tensor,
    *,
    spatial_header: Dict[str, Any],
    spectral_axis: SpectralAxis,
    spectral_axis_index: int = 0,
    overwrite: bool = False,
    compress: Union[bool, str] = False,
    allow_nonuniform: bool = False,
    nonuniform_fallback: str = "custom_cards",
) -> Dict[str, Any]:
    """Write a 3D spectral cube to a FITS primary image HDU (with potential coordinate tables).

    The cube is written with **FITS axis 3 = spectral** (``CTYPE3`` / ``CRVAL3`` /
    ``CDELT3``), matching :meth:`SpectralReader.read_data_cube` when the spectral
    axis is the slowest-varying FITS axis (tensor index ``0``).

    Args:
        path: Output ``.fits`` path.
        cube: Three-dimensional tensor. Default layout is ``(L, H, W)`` with
            ``L == len(spectral_axis.values)`` when ``spectral_axis_index=0``.
        spatial_header: Header-like dict for the **spatial** plane (axes 1 and 2):
            e.g. ``CTYPE1/2``, ``CRPIX1/2``, ``CRVAL1/2``, ``CD*`` or ``CDELT*``.
            ``NAXIS`` / ``NAXIS1`` / ``NAXIS2`` are replaced using ``cube`` shape.
        spectral_axis: Spectral coordinates for axis 3; ``values`` length
            must match the spectral dimension of ``cube``.
        spectral_axis_index: Which dimension of ``cube`` is spectral (default ``0``).
        overwrite: Passed to :func:`torchsky.write`.
        compress: Passed to :func:`torchsky.write`.
        allow_nonuniform: If True, allows non-uniform spacing of spectral axis coordinates.
        nonuniform_fallback: Method to handle non-uniform coordinates: "custom_cards"
            (writes each wavelength into custom WAVE0, WAVE1... cards) or "table"
            (writes a standards-compliant -TAB coordinate table binary extension).

    Returns:
        The merged FITS header dictionary written with the array.

    Raises:
        ValueError: If ``cube`` is not 3D, shapes disagree, or the spectral grid
            is not linear and allow_nonuniform is False.
    """
    if cube.ndim != 3:
        raise ValueError(f"cube must be 3D, got shape {tuple(cube.shape)}")
    if spectral_axis_index not in (0, 1, 2):
        raise ValueError("spectral_axis_index must be 0, 1, or 2")
    L = int(cube.shape[spectral_axis_index])
    spatial_dims = [
        int(d) for i, d in enumerate(cube.shape) if i != spectral_axis_index
    ]
    height, width = spatial_dims[0], spatial_dims[1]

    vals = spectral_axis.values.detach().to(dtype=torch.float64).reshape(-1)
    if vals.numel() != L:
        raise ValueError(
            f"spectral_axis.values length {vals.numel()} != spectral cube dim {L}"
        )

    is_uniform = True
    if vals.numel() >= 2:
        steps = vals[1:] - vals[:-1]
        step0 = steps[0]
        if not torch.allclose(steps, step0, rtol=1e-5, atol=1e-8):
            is_uniform = False
            if not allow_nonuniform:
                raise ValueError(
                    "write_cube_fits requires uniformly spaced spectral_axis.values "
                    "(single CDELT3); use a linear grid or resample first."
                )
            cdelt3 = float(torch.mean(steps).item())
        else:
            cdelt3 = float(step0.item())
    else:
        cdelt3 = 1.0

    perm = (spectral_axis_index,) + tuple(
        i for i in range(3) if i != spectral_axis_index
    )
    # Tensor layout (L, H, W): FITS NAXIS3=L, NAXIS2=H, NAXIS1=W (fast axis).
    data_write = cube.permute(*perm).contiguous()

    header: Dict[str, Any] = dict(_strip_axis3_and_naxis_keys(spatial_header))
    header["NAXIS"] = 3
    header["NAXIS1"] = width
    header["NAXIS2"] = height
    header["NAXIS3"] = L
    header["CUNIT3"] = str(spectral_axis.unit)
    header["CRPIX3"] = 1.0
    header["CRVAL3"] = float(vals[0].item())
    header["CDELT3"] = float(cdelt3)

    if not is_uniform and allow_nonuniform:
        from .io import write

        if nonuniform_fallback == "custom_cards":
            header["CTYPE3"] = str(spectral_axis.type)
            prefix = str(spectral_axis.type).upper()
            for i in range(L):
                header[f"{prefix}{i}"] = float(vals[i].item())
            write(
                path,
                data_write,
                header=header,
                overwrite=overwrite,
                compress=compress,
            )
        elif nonuniform_fallback == "table":
            from .hdu import HDUList, Header, TableHDU, TensorHDU

            header["CTYPE3"] = f"{spectral_axis.type}-TAB"
            header["PS3_0"] = "WCS-TAB"
            header["PS3_1"] = str(spectral_axis.type).upper()

            primary_hdu = TensorHDU(data=data_write, header=Header(header))
            tensor_dict = {str(spectral_axis.type).upper(): vals}
            table_header = {"EXTNAME": "WCS-TAB"}
            table_hdu = TableHDU(tensor_dict, header=Header(table_header))
            hdul = HDUList([primary_hdu, table_hdu])
            write(path, hdul, overwrite=overwrite, compress=compress)
        else:
            raise ValueError(f"Unknown nonuniform_fallback: {nonuniform_fallback}")
    else:
        from .io import write

        header["CTYPE3"] = str(spectral_axis.type)
        write(
            path,
            data_write,
            header=header,
            overwrite=overwrite,
            compress=compress,
        )
    return dict(header)


def spectral_bin_edges_from_header(
    header: Dict[str, Any],
    *,
    data_shape: Optional[Tuple[int, ...]] = None,
    tables: Optional[Any] = None,
) -> Tuple[torch.Tensor, int]:
    """Build linear or non-linear spectral bin edges from FITS WCS header metadata."""
    axis, tensor_axis_index = spectral_axis_from_header(
        header, data_shape=data_shape, tables=tables
    )
    values = axis.values.to(dtype=torch.float64)
    if values.numel() == 1:
        return torch.tensor(
            [values[0], values[0]], dtype=torch.float64
        ), tensor_axis_index
    diffs = values[1:] - values[:-1]
    if not bool(
        torch.allclose(diffs, torch.full_like(diffs, diffs[0]), rtol=1e-5, atol=1e-8)
    ):
        mid = 0.5 * (values[1:] + values[:-1])
        edges = torch.empty((values.numel() + 1,), dtype=torch.float64)
        edges[1:-1] = mid
        edges[0] = values[0] - 0.5 * diffs[0]
        edges[-1] = values[-1] + 0.5 * diffs[-1]
        return edges, tensor_axis_index
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
        kind = self.type.upper()
        unit = self.unit.lower()
        if kind == "WAVE":
            if unit in ["angstrom", "a"]:
                return self.values
            elif unit in ["nm", "nanometer"]:
                return self.values * 10.0
            elif unit in ["um", "micrometer", "micron"]:
                return self.values * 10000.0
        elif kind == "FREQ":
            return _SPEED_OF_LIGHT_ANGSTROM_PER_S / self.to_frequency()

        raise ValueError(
            f"Unsupported spectral conversion to wavelength for type={self.type!r}, unit={self.unit!r}. "
            "Supported: WAVE[A, Angstrom, nm, um] and FREQ[Hz/GHz/MHz]."
        )

    def to_frequency(self) -> torch.Tensor:
        """Convert to frequency in Hz."""
        kind = self.type.upper()
        unit = self.unit.lower()
        if kind == "FREQ":
            if unit == "hz":
                return self.values
            elif unit == "ghz":
                return self.values * 1e9
            elif unit == "mhz":
                return self.values * 1e6
        elif kind == "WAVE":
            wavelength_m = self.to_wavelength() * 1e-10  # Convert to meters
            return _SPEED_OF_LIGHT_M_PER_S / wavelength_m

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
        current_wave = self.wavelength.to(device=self.flux.device, dtype=torch.float64)
        target_wave = new_wavelength.to(device=self.flux.device, dtype=torch.float64)
        if current_wave.ndim != 1 or target_wave.ndim != 1:
            raise ValueError("wavelength grids must be 1D")
        if current_wave.numel() < 2:
            raise ValueError("resampling requires at least two input wavelengths")
        if not bool(torch.all(current_wave[1:] > current_wave[:-1])):
            raise ValueError("input wavelength grid must be strictly increasing")

        idx_hi = torch.searchsorted(current_wave, target_wave, right=False).clamp(
            1, current_wave.numel() - 1
        )
        idx_lo = idx_hi - 1
        alpha64 = (
            (target_wave - current_wave[idx_lo])
            / (current_wave[idx_hi] - current_wave[idx_lo])
        ).clamp(0.0, 1.0)

        flux_dtype = self.flux.dtype if self.flux.is_floating_point() else torch.float32
        flux = self.flux.to(device=self.flux.device, dtype=flux_dtype)
        alpha = alpha64.to(dtype=flux_dtype)
        flux_lo = flux.index_select(0, idx_lo.to(flux.device))
        flux_hi = flux.index_select(0, idx_hi.to(flux.device))
        flux_interp = flux_lo + (flux_hi - flux_lo) * alpha

        ivar_interp = None
        if self.ivar is not None:
            ivar = self.ivar.to(device=self.flux.device)
            ivar_lo = ivar.index_select(0, idx_lo.to(ivar.device))
            ivar_hi = ivar.index_select(0, idx_hi.to(ivar.device))
            inf = torch.full_like(ivar_lo, float("inf"))
            var_lo = torch.where(ivar_lo > 0, 1.0 / ivar_lo, inf)
            var_hi = torch.where(ivar_hi > 0, 1.0 / ivar_hi, inf)
            weight = alpha64.to(dtype=ivar.dtype)
            var_interp = (1.0 - weight).square() * var_lo + weight.square() * var_hi
            ivar_interp = torch.where(
                torch.isfinite(var_interp) & (var_interp > 0),
                1.0 / var_interp,
                torch.zeros_like(var_interp),
            )

        new_spectral_axis = SpectralAxis(
            values=new_wavelength.to(device=self.flux.device),
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

    def extract_spectrum(self, y: int, x: int) -> "Spectrum1D":
        """Extract 1D spectrum at spatial position (y, x)."""
        if self.spectral_axis_index == 0:
            flux = self.data[:, y, x]
            ivar = self.ivar[:, y, x] if self.ivar is not None else None
            mask = self.mask[:, y, x] if self.mask is not None else None
        elif self.spectral_axis_index == 1:
            flux = self.data[y, :, x]
            ivar = self.ivar[y, :, x] if self.ivar is not None else None
            mask = self.mask[y, :, x] if self.mask is not None else None
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
    ) -> "Spectrum1D":
        """Read 1D spectrum from FITS table."""
        from .io import open as open_fits

        with open_fits(filepath) as hdul:
            table_hdu = hdul[hdu]

            # Read required columns
            flux = table_hdu.data[flux_col]
            wavelength = table_hdu.data[wave_col]

            # Read optional columns
            ivar = (
                table_hdu.data[ivar_col]
                if ivar_col and ivar_col in table_hdu.columns
                else None
            )
            mask = (
                table_hdu.data[mask_col]
                if mask_col and mask_col in table_hdu.columns
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
    def read_data_cube(filepath: str, hdu: int = 0) -> "DataCube":
        """Read 3D data cube from FITS image."""
        from .io import open as open_fits

        with open_fits(filepath) as hdul:
            image_hdu = hdul[hdu]
            data = image_hdu.to_tensor()
            header = dict(image_hdu.header)

            spectral_axis, spectral_axis_index = spectral_axis_from_header(
                header,
                data_shape=tuple(data.shape),
                tables=hdul,
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
