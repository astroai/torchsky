"""Image-grid validation and FITS-like header helpers for map operators."""

from __future__ import annotations

from typing import Mapping

from torch import Tensor


GRID_EQUIVALENT_IGNORE_KEYS = frozenset(
    {
        "COMMENT",
        "HISTORY",
        "SIMPLE",
        "BITPIX",
        "EXTEND",
        "XTENSION",
        "EXTNAME",
        "EXTVER",
        "PCOUNT",
        "GCOUNT",
        "CHECKSUM",
        "DATASUM",
    }
)


def validate_image_plane_2d(image: Tensor) -> Tensor:
    if image.ndim != 2:
        raise ValueError("Image-plane mapmaking currently supports 2D images only")
    return image


def target_shape_from_header(
    header: Mapping[str, object],
    *,
    fallback: tuple[int, int],
) -> tuple[int, int]:
    naxis1 = header.get("NAXIS1")
    naxis2 = header.get("NAXIS2")
    if naxis1 is None or naxis2 is None:
        return fallback
    width = int(naxis1)
    height = int(naxis2)
    if width <= 0 or height <= 0:
        raise ValueError("Target header must define positive NAXIS1/NAXIS2")
    return height, width


def normalized_header_for_shape(
    header: Mapping[str, object],
    shape: tuple[int, int],
    *,
    overrides: Mapping[str, object] | None = None,
) -> dict[str, object]:
    out = dict(header)
    out["NAXIS"] = 2
    out["NAXIS1"] = int(shape[1])
    out["NAXIS2"] = int(shape[0])
    if overrides is not None:
        out.update(dict(overrides))
    return out


def headers_equivalent_for_direct_copy(
    source: Mapping[str, object],
    target: Mapping[str, object],
) -> bool:
    def normalize(header: Mapping[str, object]) -> dict[object, object]:
        out: dict[object, object] = {}
        for key, value in header.items():
            key_text = str(key)
            if key_text.endswith("_COMMENT"):
                continue
            if key_text in GRID_EQUIVALENT_IGNORE_KEYS:
                continue
            out[key] = value
        return out

    return normalize(source) == normalize(target)
