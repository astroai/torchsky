"""Spectral and data cube support for torchsky.

Implements 1D spectra with inverse variance/mask arrays and 3D data cubes
for IFU instruments and radio astronomy.

Note:
This module is an experimental helper layer built on top of torchsky FITS I/O.
It is not part of the core FITS read/write parity surface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch

from .log_lambda import (
    convolve_log_lambda as convolve_log_lambda,
    sigma_R_kms as sigma_R_kms,
)
from .redshift import (
    C_KM_S as C_KM_S,
    V_KIND_BETA as V_KIND_BETA,
    V_KIND_KMS as V_KIND_KMS,
    V_KIND_REDSHIFT as V_KIND_REDSHIFT,
    apply_redshift as apply_redshift,
    apply_redshift_f_lambda as apply_redshift_f_lambda,
    beta_from_redshift as beta_from_redshift,
    compose_rapidities as compose_rapidities,
    convert_velocity_kind as convert_velocity_kind,
    dimming_f_lambda as dimming_f_lambda,
    dimming_tolman_sb as dimming_tolman_sb,
    los_barycentric_beta as los_barycentric_beta,
    normalize_v_kind as normalize_v_kind,
    one_plus_z_from_beta as one_plus_z_from_beta,
    one_plus_z_from_codes as one_plus_z_from_codes,
    one_plus_z_from_kind as one_plus_z_from_kind,
    one_plus_z_from_redshift as one_plus_z_from_redshift,
    one_plus_z_from_velocity_kms as one_plus_z_from_velocity_kms,
    one_plus_z_from_zeta as one_plus_z_from_zeta,
    redshift_from_velocity_kms as redshift_from_velocity_kms,
    require_one_plus_z as require_one_plus_z,
    shift_ln_wavelength as shift_ln_wavelength,
    shift_wavelength as shift_wavelength,
    velocity_kms_from_redshift as velocity_kms_from_redshift,
    zeta_from_beta as zeta_from_beta,
    zeta_from_redshift as zeta_from_redshift,
    zeta_from_velocity_kms as zeta_from_velocity_kms,
)
from .wcs import WCS
from .wcs.spectral import (
    air_wavelength_to_vacuum,
    canonical_spectral_type,
    frequency_to_spectral,
    spectral_to_frequency as _wcs_spectral_to_frequency,
    spectral_unit_scale,
)


_SPECTRAL_TYPE_KEYS = ("WAVE", "AWAV", "FREQ", "ENER", "WAVN", "VELO")
_SPEED_OF_LIGHT_M_PER_S = 299_792_458.0
_PLANCK_JOULE_SECOND = 6.626_070_15e-34
_ELECTRON_VOLT_JOULE = 1.602_176_634e-19


@dataclass(frozen=True)
class SpectralRebinResult:
    """Flux-density samples and validity metadata on a spectral edge grid."""

    flux: torch.Tensor
    spectral_edges: torch.Tensor
    ivar: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None


def _validate_spectral_edges(
    edges: torch.Tensor, expected_bins: int, name: str
) -> None:
    if edges.ndim != 1 or edges.numel() != expected_bins + 1:
        raise ValueError(f"{name} must be 1D with {expected_bins + 1} elements")
    if not edges.is_floating_point() or not bool(torch.isfinite(edges).all()):
        raise ValueError(f"{name} must contain finite floating-point coordinates")
    if not bool(torch.all(edges[1:] > edges[:-1])):
        raise ValueError(f"{name} must be strictly increasing")


def _broadcast_spectral_inputs(
    flux: torch.Tensor,
    query_edges: torch.Tensor,
    ivar: Optional[torch.Tensor],
    mask: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    leading_shape = torch.broadcast_shapes(flux.shape[:-1], query_edges.shape[:-1])
    flux = flux.expand(leading_shape + (flux.shape[-1],))
    query_edges = query_edges.expand(leading_shape + (query_edges.shape[-1],))

    if ivar is not None:
        if ivar.shape[-1:] != flux.shape[-1:]:
            raise ValueError("ivar must have the same spectral length as flux")
        try:
            ivar = ivar.expand(leading_shape + (flux.shape[-1],))
        except RuntimeError as exc:
            raise ValueError("ivar is not broadcastable to flux") from exc
    if mask is not None:
        if mask.shape[-1:] != flux.shape[-1:]:
            raise ValueError("mask must have the same spectral length as flux")
        try:
            mask = mask.expand(leading_shape + (flux.shape[-1],))
        except RuntimeError as exc:
            raise ValueError("mask is not broadcastable to flux") from exc
    return flux, query_edges, ivar, mask


def _piecewise_constant_integral(
    values: torch.Tensor,
    input_edges: torch.Tensor,
    points: torch.Tensor,
) -> torch.Tensor:
    """Integral from the first input edge to each query point."""
    widths = (input_edges[1:] - input_edges[:-1]).to(dtype=values.dtype)
    prefix = torch.cat(
        [torch.zeros_like(values[..., :1]), torch.cumsum(values * widths, dim=-1)],
        dim=-1,
    )
    clipped = points.clamp(min=input_edges[0], max=input_edges[-1])
    indices = torch.searchsorted(input_edges, clipped, right=True).sub(1)
    indices = indices.clamp(0, values.shape[-1] - 1)
    prefix_at = torch.gather(prefix, -1, indices)
    value_at = torch.gather(values, -1, indices)
    left_edge = input_edges[indices]
    return prefix_at + value_at * (clipped - left_edge).to(dtype=values.dtype)


def _piecewise_constant_variance(
    variance: torch.Tensor,
    input_edges: torch.Tensor,
    query_edges: torch.Tensor,
) -> torch.Tensor:
    """Variance of integrated independent piecewise-constant samples."""
    a = query_edges[..., :-1].clamp(min=input_edges[0], max=input_edges[-1])
    b = query_edges[..., 1:].clamp(min=input_edges[0], max=input_edges[-1])
    n_bins = variance.shape[-1]
    start = torch.searchsorted(input_edges, a, right=True).sub(1).clamp(0, n_bins - 1)
    stop = torch.searchsorted(input_edges, b, right=False).sub(1).clamp(0, n_bins - 1)

    widths = (input_edges[1:] - input_edges[:-1]).to(dtype=variance.dtype)
    prefix = torch.cat(
        [
            torch.zeros_like(variance[..., :1]),
            torch.cumsum(variance * widths.square(), dim=-1),
        ],
        dim=-1,
    )
    start_var = torch.gather(variance, -1, start)
    stop_var = torch.gather(variance, -1, stop)
    left_width = (input_edges[start + 1] - a).to(dtype=variance.dtype)
    right_width = (b - input_edges[stop]).to(dtype=variance.dtype)
    middle = torch.gather(prefix, -1, stop) - torch.gather(prefix, -1, start + 1)
    split = start_var * left_width.square() + middle + stop_var * right_width.square()
    same = start_var * (b - a).to(dtype=variance.dtype).square()
    return torch.where(start == stop, same, split).clamp_min(0)


def _rebin_piecewise_constant(
    flux: torch.Tensor,
    input_edges: torch.Tensor,
    query_edges: torch.Tensor,
    normalization_widths: torch.Tensor,
    *,
    ivar: Optional[torch.Tensor],
    mask: Optional[torch.Tensor],
) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    flux, query_edges, ivar, mask = _broadcast_spectral_inputs(
        flux, query_edges, ivar, mask
    )
    valid = torch.isfinite(flux)
    if mask is not None:
        valid = valid & mask.to(dtype=torch.bool)
    if ivar is not None:
        valid = valid & torch.isfinite(ivar) & (ivar > 0)

    safe_flux = torch.where(valid, flux, torch.zeros_like(flux))
    integral = _piecewise_constant_integral(safe_flux, input_edges, query_edges)
    integrated_flux = integral[..., 1:] - integral[..., :-1]
    widths = normalization_widths.to(device=flux.device, dtype=flux.dtype)
    output_flux = integrated_flux / widths

    coord_dtype = input_edges.dtype
    invalid = (~valid).to(dtype=coord_dtype)
    invalid_integral = _piecewise_constant_integral(invalid, input_edges, query_edges)
    invalid_overlap = invalid_integral[..., 1:] - invalid_integral[..., :-1]
    inside = (query_edges[..., :-1] >= input_edges[0]) & (
        query_edges[..., 1:] <= input_edges[-1]
    )
    scale = torch.maximum(
        normalization_widths.to(device=flux.device, dtype=coord_dtype).abs(),
        torch.ones((), device=flux.device, dtype=coord_dtype),
    )
    tolerance = 8 * torch.finfo(coord_dtype).eps * scale
    output_mask = inside & (invalid_overlap.abs() <= tolerance)

    output_ivar = None
    if ivar is not None:
        safe_variance = torch.where(valid, ivar.reciprocal(), torch.zeros_like(ivar))
        integrated_variance = _piecewise_constant_variance(
            safe_variance, input_edges, query_edges
        )
        output_variance = integrated_variance / widths.square()
        output_ivar = torch.where(
            output_mask & torch.isfinite(output_variance) & (output_variance > 0),
            output_variance.reciprocal(),
            torch.zeros_like(output_variance),
        )
    return output_flux, output_ivar, output_mask


def flux_conserving_rebin(
    flux: torch.Tensor,
    input_edges: torch.Tensor,
    output_edges: torch.Tensor,
    *,
    ivar: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
) -> SpectralRebinResult:
    """Rebin piecewise-constant flux density while preserving integrated flux.

    Spectral samples occupy bins described by ``input_edges``. Leading flux
    dimensions are batches; all tensors remain on the input device. Output bins
    that extend beyond the input range or overlap invalid input samples are
    marked false without renormalising away missing flux.
    """
    if flux.ndim < 1 or not flux.is_floating_point():
        raise ValueError("flux must be a floating-point tensor with a spectral axis")
    input_edges = torch.as_tensor(input_edges, device=flux.device)
    output_edges = torch.as_tensor(output_edges, device=flux.device)
    _validate_spectral_edges(input_edges, flux.shape[-1], "input_edges")
    _validate_spectral_edges(output_edges, output_edges.numel() - 1, "output_edges")
    if input_edges.dtype != output_edges.dtype:
        output_edges = output_edges.to(dtype=input_edges.dtype)

    widths = output_edges[1:] - output_edges[:-1]
    rebinned, rebinned_ivar, rebinned_mask = _rebin_piecewise_constant(
        flux,
        input_edges,
        output_edges,
        widths,
        ivar=ivar,
        mask=mask,
    )
    return SpectralRebinResult(rebinned, output_edges, rebinned_ivar, rebinned_mask)


def relativistic_doppler_shift(
    flux: torch.Tensor,
    input_edges: torch.Tensor,
    radial_velocity: Union[float, torch.Tensor],
    *,
    output_edges: Optional[torch.Tensor] = None,
    ivar: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
) -> SpectralRebinResult:
    """Apply a signed relativistic radial-velocity shift to flux density.

    Positive velocity is receding. The Doppler factor is
    ``sqrt((1 + v/c) / (1 - v/c))``. When ``output_edges`` is omitted, each
    batch receives its shifted edge grid. Supplying a common output grid rebins
    the shifted spectra onto that grid without losing integrated flux.
    """
    if flux.ndim < 1 or not flux.is_floating_point():
        raise ValueError("flux must be a floating-point tensor with a spectral axis")
    input_edges = torch.as_tensor(input_edges, device=flux.device)
    _validate_spectral_edges(input_edges, flux.shape[-1], "input_edges")
    velocity = torch.as_tensor(
        radial_velocity, device=flux.device, dtype=input_edges.dtype
    )
    if not bool(torch.isfinite(velocity).all()) or bool(
        torch.any(velocity.abs() >= _SPEED_OF_LIGHT_M_PER_S)
    ):
        raise ValueError("radial_velocity must be finite with absolute value below c")
    factor = torch.sqrt(
        (1.0 + velocity / _SPEED_OF_LIGHT_M_PER_S)
        / (1.0 - velocity / _SPEED_OF_LIGHT_M_PER_S)
    )

    if output_edges is None:
        leading_shape = torch.broadcast_shapes(flux.shape[:-1], factor.shape)
        factor = factor.expand(leading_shape)
        shifted_flux = flux.expand(
            leading_shape + (flux.shape[-1],)
        ) / factor.unsqueeze(-1)
        shifted_edges = input_edges.expand(
            leading_shape + input_edges.shape
        ) * factor.unsqueeze(-1)
        shifted_mask = torch.isfinite(shifted_flux)
        if mask is not None:
            shifted_mask = shifted_mask & mask.expand_as(shifted_flux).to(
                dtype=torch.bool
            )
        shifted_ivar = None
        if ivar is not None:
            shifted_ivar = ivar.expand_as(shifted_flux) * factor.unsqueeze(-1).square()
            shifted_mask = (
                shifted_mask & torch.isfinite(shifted_ivar) & (shifted_ivar > 0)
            )
            shifted_ivar = torch.where(
                shifted_mask, shifted_ivar, torch.zeros_like(shifted_ivar)
            )
        shifted_flux = torch.where(
            shifted_mask, shifted_flux, torch.zeros_like(shifted_flux)
        )
        return SpectralRebinResult(
            shifted_flux, shifted_edges, shifted_ivar, shifted_mask
        )

    output_edges = torch.as_tensor(output_edges, device=flux.device)
    _validate_spectral_edges(output_edges, output_edges.numel() - 1, "output_edges")
    output_edges = output_edges.to(dtype=input_edges.dtype)
    query_edges = output_edges / factor.unsqueeze(-1)
    widths = output_edges[1:] - output_edges[:-1]
    shifted_flux, shifted_ivar, shifted_mask = _rebin_piecewise_constant(
        flux,
        input_edges,
        query_edges,
        widths,
        ivar=ivar,
        mask=mask,
    )
    return SpectralRebinResult(shifted_flux, output_edges, shifted_ivar, shifted_mask)


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
        overwrite: Passed to :func:`torchfits.write`.
        compress: Passed to :func:`torchfits.write`.
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
        from torchfits import write

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
            from torchfits.hdu import HDUList, Header, TableHDU, TensorHDU

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
        from torchfits import write

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
    bin_edges: Optional[torch.Tensor] = None

    @property
    def quantity(self) -> str:
        """Return a normalized physical quantity name."""
        kind = canonical_spectral_type(self.type)
        quantities = {
            "WAVE": "wavelength",
            "AWAV": "wavelength",
            "FREQ": "frequency",
            "ENER": "energy",
            "WAVN": "wavenumber",
            "VRAD": "velocity",
            "VOPT": "velocity",
            "VELO": "velocity",
            "ZOPT": "redshift",
            "BETA": "beta",
        }
        if kind not in quantities:
            raise ValueError(f"Unsupported spectral axis type {self.type!r}")
        return quantities[kind]

    @property
    def direction(self) -> int:
        """Return ``1`` for ascending and ``-1`` for descending coordinates."""
        values = self._validated_values()
        if values.numel() < 2:
            return 1
        differences = torch.diff(values)
        if bool(torch.any(differences == 0)):
            raise ValueError("spectral axis coordinates must not contain duplicates")
        if bool(torch.all(differences > 0)):
            return 1
        if bool(torch.all(differences < 0)):
            return -1
        raise ValueError("spectral axis coordinates must be strictly monotonic")

    def is_regular(self, *, rtol: float = 1.0e-6, atol: float = 0.0) -> bool:
        """Return whether adjacent coordinate steps are equal within tolerance."""
        values = self._validated_values()
        if values.numel() < 3:
            return True
        differences = torch.diff(values)
        _ = self.direction
        return bool(
            torch.allclose(
                differences,
                differences[0].expand_as(differences),
                rtol=rtol,
                atol=atol,
            )
        )

    def edges(self) -> torch.Tensor:
        """Return explicit or midpoint-derived channel edges in axis units."""
        values = self._validated_values()
        _ = self.direction
        if self.bin_edges is not None:
            edges = self.bin_edges
            if (
                edges.ndim != 1
                or edges.numel() != values.numel() + 1
                or edges.device != values.device
                or edges.dtype != values.dtype
                or not bool(torch.isfinite(edges).all())
            ):
                raise ValueError(
                    "bin_edges must be finite and have length coordinates + 1 "
                    "with matching dtype/device"
                )
            if bool(torch.any(torch.diff(edges) * self.direction <= 0)):
                raise ValueError("bin_edges must follow the coordinate direction")
            return edges
        if values.numel() < 2:
            raise ValueError("one-channel axes require explicit bin_edges")
        midpoints = 0.5 * (values[:-1] + values[1:])
        edges = torch.empty(
            values.numel() + 1, dtype=values.dtype, device=values.device
        )
        edges[1:-1] = midpoints
        edges[0] = values[0] - 0.5 * (values[1] - values[0])
        edges[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
        return edges

    def channel_widths(self) -> torch.Tensor:
        """Return positive channel widths in the declared axis unit."""
        return torch.abs(torch.diff(self.edges()))

    def quadrature_weights(self) -> torch.Tensor:
        """Return positive trapezoid weights for samples on this axis."""
        values = self._validated_values()
        _ = self.direction
        if values.numel() == 1:
            return self.channel_widths()
        spacing = torch.abs(torch.diff(values))
        weights = torch.empty_like(values)
        weights[0] = 0.5 * spacing[0]
        weights[-1] = 0.5 * spacing[-1]
        if values.numel() > 2:
            weights[1:-1] = 0.5 * (spacing[:-1] + spacing[1:])
        return weights

    def select_interval(
        self, lower: float, upper: float
    ) -> tuple["SpectralAxis", torch.Tensor]:
        """Select a closed physical interval and return the axis plus source mask."""
        if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
            raise ValueError("interval bounds must be finite with lower <= upper")
        values = self._validated_values()
        mask = (values >= lower) & (values <= upper)
        indices = torch.nonzero(mask, as_tuple=False).flatten()
        if indices.numel() == 0:
            raise ValueError("interval does not overlap the spectral axis")
        if indices.numel() > 1 and not bool(torch.all(torch.diff(indices) == 1)):
            raise RuntimeError("monotonic spectral interval must be contiguous")
        selected_edges = None
        if self.bin_edges is not None:
            start = int(indices[0])
            stop = int(indices[-1]) + 2
            selected_edges = self.bin_edges[start:stop]
        return (
            SpectralAxis(
                values=values[mask],
                unit=self.unit,
                type=self.type,
                rest_frequency=self.rest_frequency,
                redshift=self.redshift,
                bin_edges=selected_edges,
            ),
            mask,
        )

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "SpectralAxis":
        """Transfer coordinate tensors without changing physical metadata."""
        return SpectralAxis(
            values=self.values.to(device=device, dtype=dtype),
            unit=self.unit,
            type=self.type,
            rest_frequency=self.rest_frequency,
            redshift=self.redshift,
            bin_edges=(
                None
                if self.bin_edges is None
                else self.bin_edges.to(device=device, dtype=dtype)
            ),
        )

    def _validated_values(self) -> torch.Tensor:
        if (
            not self.values.is_floating_point()
            or self.values.ndim != 1
            or self.values.numel() == 0
            or not bool(torch.isfinite(self.values).all())
        ):
            raise ValueError(
                "spectral axis coordinates must be a non-empty finite "
                "floating-point vector"
            )
        return self.values

    def _positive_values(self) -> torch.Tensor:
        values = self._validated_values()
        if bool(torch.any(values <= 0)):
            raise ValueError("spectral conversion coordinates must be positive")
        return values

    def to_wavelength(self) -> torch.Tensor:
        """Convert to vacuum wavelength in Angstroms.

        Air wavelengths (``AWAV``) are refracted to vacuum first.
        Length axes use FITS unit scales; other types go through frequency.
        """
        values = self._validated_values()
        kind = canonical_spectral_type(self.type)
        if kind == "WAVE":
            angstrom = values * (
                spectral_unit_scale("WAVE", self.unit)
                / spectral_unit_scale("WAVE", "Angstrom")
            )
            return angstrom.to(dtype=values.dtype)
        if kind == "AWAV":
            vac = air_wavelength_to_vacuum(values, unit=self.unit)
            angstrom = vac * (
                spectral_unit_scale("WAVE", self.unit)
                / spectral_unit_scale("WAVE", "Angstrom")
            )
            return angstrom.to(dtype=values.dtype)
        freq = self.to_frequency()
        wave = frequency_to_spectral(freq, "WAVE", unit="Angstrom")
        return wave.to(dtype=values.dtype, device=values.device)

    def to_frequency(self) -> torch.Tensor:
        """Convert to frequency in Hz via FITS spectral conventions."""
        values = self._validated_values()
        kind = canonical_spectral_type(self.type)
        out = _wcs_spectral_to_frequency(
            values,
            kind,
            unit=self.unit,
            rest_frequency=self.rest_frequency,
        )
        return out.to(dtype=values.dtype, device=values.device)

    def to_energy(self, unit: str = "eV") -> torch.Tensor:
        """Convert to photon energy in joules or electron-volt units."""
        energy_joule = self.to_frequency() * _PLANCK_JOULE_SECOND
        scales = {
            "j": 1.0,
            "joule": 1.0,
            "ev": _ELECTRON_VOLT_JOULE,
            "kev": 1.0e3 * _ELECTRON_VOLT_JOULE,
            "mev": 1.0e6 * _ELECTRON_VOLT_JOULE,
            "gev": 1.0e9 * _ELECTRON_VOLT_JOULE,
        }
        scale = scales.get(unit.lower())
        if scale is None:
            raise ValueError(f"Unsupported energy unit {unit!r}")
        return energy_joule / scale

    def to_wavenumber(self, unit: str = "1/m") -> torch.Tensor:
        """Convert to reciprocal wavelength (cycles per metre or centimetre)."""
        inverse_metre = self.to_frequency() / _SPEED_OF_LIGHT_M_PER_S
        normalized = unit.lower()
        if normalized in {"1/m", "m^-1", "m-1"}:
            return inverse_metre
        if normalized in {"1/cm", "cm^-1", "cm-1"}:
            return inverse_metre / 100.0
        raise ValueError(f"Unsupported wavenumber unit {unit!r}")


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
        work_dtype = (
            torch.float32 if self.flux.device.type == "mps" else torch.float64
        )
        current_wave = self.wavelength.to(device=self.flux.device, dtype=work_dtype)
        target_wave = new_wavelength.to(device=self.flux.device, dtype=work_dtype)
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

        if width is not None:
            mask = torch.abs(wave - wavelength) <= width / 2
            if bool(mask.any()):
                if self.spectral_axis_index == 0:
                    return torch.mean(self.data[mask, :, :], dim=0)
                elif self.spectral_axis_index == 1:
                    return torch.mean(self.data[:, mask, :], dim=1)
                elif self.spectral_axis_index == 2:
                    return torch.mean(self.data[:, :, mask], dim=2)

        # Find closest wavelength (or fallback when width window contains no channels)
        idx = torch.argmin(torch.abs(wave - wavelength))
        if self.spectral_axis_index == 0:
            return self.data[idx, :, :]
        elif self.spectral_axis_index == 1:
            return self.data[:, idx, :]
        elif self.spectral_axis_index == 2:
            return self.data[:, :, idx]
        raise ValueError("Invalid spectral_axis_index")

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
                bin_edges=(
                    self.spectral_axis.bin_edges.to(device)
                    if self.spectral_axis.bin_edges is not None
                    else None
                ),
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
        from torchfits import open as open_fits

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
                unit=table_hdu.header.get("CUNIT1", "Angstrom"),
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
        from torchfits import open as open_fits

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
