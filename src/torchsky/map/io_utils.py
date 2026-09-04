"""Internal file-system I/O policy and helper utilities for mapmaking."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import torch
from torch import Tensor

import torchfits.cpp as _cpp

from ..hdu import HDUList, Header, TensorHDU
from ..io import read
from .operators.image_grid import validate_image_plane_2d

if TYPE_CHECKING:
    from .product import MapProductResult

_HEADER_PROBE_EXC: tuple[type[BaseException], ...] = (
    OSError,
    RuntimeError,
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
    ImportError,
    ModuleNotFoundError,
)

_FITS_STRUCTURAL_KEYS = {
    "SIMPLE",
    "BITPIX",
    "NAXIS",
    "EXTEND",
    "XTENSION",
    "PCOUNT",
    "GCOUNT",
}


def _product_plane(tensor: Tensor) -> Tensor:
    if tensor.ndim >= 3 and tensor.shape[0] == 1:
        return tensor.squeeze(0)
    return tensor


def _product_header(
    base: dict[str, Any],
    name: str,
    *,
    primary: bool,
) -> Header:
    header = dict(base)
    if not primary:
        header = {
            key: value
            for key, value in header.items()
            if key not in _FITS_STRUCTURAL_KEYS and not key.startswith("NAXIS")
        }
        header["XTENSION"] = "IMAGE"
    header["EXTNAME"] = name
    return Header(header)


def _context_cards(result: MapProductResult) -> dict[str, str]:
    provenance = result.provenance
    if provenance is None:
        return {}
    observations = provenance.get("context_observations")
    if not isinstance(observations, Iterable):
        return {}
    cards: dict[str, str] = {}
    for fallback_index, item in enumerate(observations):
        if not isinstance(item, dict):
            continue
        index = int(item.get("index", fallback_index))
        if index >= 10_000:
            raise ValueError("FITS CTX mapping supports at most 10,000 observations")
        path = Path(str(item.get("path", ""))).name
        hdu = item.get("hdu")
        cards[f"CTX{index:04d}"] = (
            f"{index // 32}:{index % 32}:{index}:{path}[{hdu}]"
        )
    return cards


def write_map_product(
    path: str | Path,
    result: MapProductResult,
    *,
    requested_planes: Iterable[str],
    overwrite: bool = False,
) -> tuple[str, ...]:
    """Write available map products as one torchfits-owned MEF."""
    requested = tuple(dict.fromkeys(str(name).upper() for name in requested_planes))
    if "SCI" not in requested:
        raise ValueError("map products must request SCI")
    planes: list[tuple[str, Tensor, dict[str, Any]]] = [
        ("SCI", _product_plane(result.data), {})
    ]
    if "WHT" in requested and result.weight is not None:
        planes.append(("WHT", _product_plane(result.weight), {}))
    if "IVAR" in requested and result.uncertainty is not None:
        planes.append(
            (
                "IVAR",
                _product_plane(result.uncertainty.inverse_variance),
                {"UNCKIND": result.uncertainty.kind},
            )
        )
    if "CTX" in requested and result.context is not None:
        planes.append(
            ("CTX", _product_plane(result.context), _context_cards(result))
        )
    if "MASK" in requested and result.mask is not None:
        planes.append(
            ("MASK", _product_plane(result.mask).to(dtype=torch.uint8), {})
        )
    if "NEXP" in requested and result.overlap_count is not None:
        planes.append(("NEXP", _product_plane(result.overlap_count), {}))
    if (
        "NCORR" in requested
        and result.uncertainty is not None
        and result.uncertainty.correlation_factor is not None
    ):
        correlation = torch.as_tensor(result.uncertainty.correlation_factor)
        if correlation.numel() == 1:
            correlation = correlation.expand_as(result.data)
        planes.append(
            (
                "NCORR",
                _product_plane(correlation),
                {
                    "NCORTYPE": "LOCAL_ENBW",
                    "NCORDEF": "W1/W2",
                },
            )
        )
    if "IVAR" in requested and result.uncertainty is not None:
        for component_name, component in sorted(result.uncertainty.components.items()):
            name = f"VAR_{component_name.upper()}"
            planes.append((name, _product_plane(component), {}))

    hdus = []
    for index, (name, data, extra_header) in enumerate(planes):
        header = _product_header(result.header, name, primary=index == 0)
        for key, value in extra_header.items():
            header[key] = value
        hdus.append(TensorHDU(data=data.detach(), header=header))
    HDUList(hdus).write(str(path), overwrite=overwrite)
    return tuple(name for name, _, _ in planes)


def read_image_plane_observation(
    path: str,
    hdu: int | str | None,
    device: str,
) -> tuple[Tensor, dict]:
    """Load observation image and header, validating shape to be 2D."""
    image, header = read(path, hdu=hdu, return_header=True, device=device)
    image = validate_image_plane_2d(image)
    return image, dict(header)


def _observation_header_for_area_scaling(
    obs: object,
    *,
    cached_header: dict | None = None,
) -> dict:
    """Read minimal header cards needed for area scaling computation."""
    if cached_header is not None:
        return dict(cached_header)
    try:
        hdu = getattr(obs, "hdu", None)
        path = getattr(obs, "path")
        if isinstance(hdu, int) or hdu is None:
            return dict(Header(_cpp.read_header_dict(path, 0 if hdu is None else hdu)))
    except _HEADER_PROBE_EXC:
        pass
    path = getattr(obs, "path")
    hdu = getattr(obs, "hdu", None)
    _, header = read(path, hdu=hdu, return_header=True)
    return dict(header)
