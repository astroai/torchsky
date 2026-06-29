"""Internal file-system I/O policy and helper utilities for mapmaking."""

from torch import Tensor

import torchfits.cpp as _cpp

from ..hdu import Header
from ..io import read
from .operators.image_grid import validate_image_plane_2d

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
