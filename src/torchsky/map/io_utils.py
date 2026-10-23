"""Internal file-system I/O policy and helper utilities for mapmaking."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable

import torch
import torchfits
from torch import Tensor

from torchfits.hdu import HDUList, Header, TensorHDU
from torchfits import read
from .operators.image_grid import validate_image_plane_2d

if TYPE_CHECKING:
    from .product import ProductResult

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


def _context_cards(result: ProductResult) -> dict[str, str]:
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
    result: ProductResult,
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
    bunit = dict(result.header).get("BUNIT")
    bunit_text = bunit.strip() if isinstance(bunit, str) else ""
    if "IVAR" in requested and result.uncertainty is not None:
        ivar_header: dict[str, Any] = {"UNCKIND": result.uncertainty.kind}
        if bunit_text:
            ivar_header["BUNIT"] = f"({bunit_text})**-2"
        planes.append(
            (
                "IVAR",
                _product_plane(result.uncertainty.inverse_variance),
                ivar_header,
            )
        )
    if "DQ" in requested and result.quality is not None:
        policy = result.quality.policy
        dq_header: dict[str, Any] = {"DQOR": "BITWISE_OR"}
        if policy is not None:
            dq_header.update(
                {
                    "DQREJECT": -1
                    if policy.reject_bits is None
                    else policy.reject_bits,
                    "DQCRBIT": policy.cosmic_ray_bit,
                    "DQUNKN": policy.unknown_bit_policy,
                }
            )
        planes.append(
            (
                "DQ",
                _product_plane(result.quality.dq),
                dq_header,
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


def _parse_scamp_head_file(
    head_path: str | Path,
    hdu: int | str | None = None,
) -> dict[str, Any]:
    """Parse a SCAMP/SWarp ASCII `.head` sidecar file into a header dictionary."""
    text = Path(head_path).read_text(encoding="utf-8", errors="replace")
    blocks: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("COMMENT", "HISTORY")):
            continue
        if line == "END" or line.startswith("END "):
            blocks.append(current)
            current = {}
            continue
        if "=" not in raw_line:
            continue
        key_part, val_part = raw_line.split("=", 1)
        key = key_part.strip().upper()
        if not key:
            continue
        # Strip inline FITS comment outside quotes
        val_str = val_part.strip()
        if val_str.startswith("'"):
            end_quote = val_str.find("'", 1)
            if end_quote != -1:
                current[key] = val_str[1:end_quote].strip()
                continue
        if "/" in val_str:
            val_str = val_str.split("/", 1)[0].strip()
        if val_str in ("T", "True", "true"):
            current[key] = True
        elif val_str in ("F", "False", "false"):
            current[key] = False
        else:
            try:
                if "." in val_str or "E" in val_str.upper() or "D" in val_str.upper():
                    current[key] = float(val_str.replace("D", "E").replace("d", "e"))
                else:
                    current[key] = int(val_str)
            except ValueError:
                current[key] = val_str.strip("'\" ")
    if current:
        blocks.append(current)
    if not blocks:
        return {}
    if isinstance(hdu, int) and 0 <= hdu < len(blocks):
        return blocks[hdu]
    return blocks[0]


def _resolve_observation_sidecar_header(obs: object) -> dict[str, Any]:
    head_path = getattr(obs, "head_path", None)
    if head_path is not None:
        return _parse_scamp_head_file(head_path, getattr(obs, "hdu", None))
    if bool(getattr(obs, "use_sidecar_head", False)):
        obs_path = getattr(obs, "path", None)
        if obs_path is not None:
            candidate = Path(str(obs_path)).with_suffix(".head")
            if candidate.is_file():
                return _parse_scamp_head_file(candidate, getattr(obs, "hdu", None))
    return {}


def read_image_plane_observation(
    path: str,
    hdu: int | str | None,
    device: str,
    *,
    read_fn: Callable[..., Any] | None = None,
) -> tuple[Tensor, dict]:
    """Load observation image and header, validating shape to be 2D."""
    reader = read if read_fn is None else read_fn
    image, header = reader(path, hdu=hdu, return_header=True, device=device)
    image = validate_image_plane_2d(image)
    return image, dict(header)


def load_observation_image_and_header(
    obs: object,
    device: str | torch.device,
    *,
    read_fn: Callable[..., Any] | None = None,
) -> tuple[Tensor, dict]:
    """Load an observation's 2D image and header from in-memory tensors or FITS path."""
    dev_str = str(device)
    data = getattr(obs, "data", None)
    if data is not None:
        image = validate_image_plane_2d(torch.as_tensor(data)).to(device=dev_str)
        hdr: dict[str, Any] = {}
        obs_wcs = getattr(obs, "wcs", None)
        if obs_wcs is not None and hasattr(obs_wcs, "to_header"):
            hdr.update(obs_wcs.to_header())
        elif isinstance(obs_wcs, dict):
            hdr.update(obs_wcs)
        obs_header = getattr(obs, "header", None)
        if isinstance(obs_header, dict):
            hdr.update(obs_header)
        sidecar = _resolve_observation_sidecar_header(obs)
        if sidecar:
            hdr.update(sidecar)
        hdr.setdefault("NAXIS", 2)
        hdr["NAXIS1"] = int(image.shape[1])
        hdr["NAXIS2"] = int(image.shape[0])
        return image, hdr

    path = getattr(obs, "path", None)
    if path is None:
        raise ValueError(
            "Observation must provide either in-memory `data` or a FITS `path`"
        )
    image, hdr = read_image_plane_observation(
        str(path), getattr(obs, "hdu", None), dev_str, read_fn=read_fn
    )
    obs_wcs = getattr(obs, "wcs", None)
    if obs_wcs is not None and hasattr(obs_wcs, "to_header"):
        hdr.update(obs_wcs.to_header())
    elif isinstance(obs_wcs, dict):
        hdr.update(obs_wcs)
    obs_header = getattr(obs, "header", None)
    if isinstance(obs_header, dict):
        hdr.update(obs_header)
    sidecar = _resolve_observation_sidecar_header(obs)
    if sidecar:
        hdr.update(sidecar)
    return image, hdr


def _observation_header_for_area_scaling(
    obs: object,
    *,
    cached_header: dict | None = None,
) -> dict:
    """Read minimal header cards needed for area scaling computation."""
    if cached_header is not None:
        return dict(cached_header)
    data = getattr(obs, "data", None)
    if data is not None:
        image = validate_image_plane_2d(torch.as_tensor(data))
        hdr: dict[str, Any] = {}
        obs_wcs = getattr(obs, "wcs", None)
        if obs_wcs is not None and hasattr(obs_wcs, "to_header"):
            hdr.update(obs_wcs.to_header())
        elif isinstance(obs_wcs, dict):
            hdr.update(obs_wcs)
        obs_header = getattr(obs, "header", None)
        if isinstance(obs_header, dict):
            hdr.update(obs_header)
        sidecar = _resolve_observation_sidecar_header(obs)
        if sidecar:
            hdr.update(sidecar)
        hdr.setdefault("NAXIS", 2)
        hdr["NAXIS1"] = int(image.shape[1])
        hdr["NAXIS2"] = int(image.shape[0])
        return hdr
    hdr_out: dict[str, Any] | None = None
    try:
        hdu = getattr(obs, "hdu", None)
        path = getattr(obs, "path")
        if path is not None and (isinstance(hdu, int) or hdu is None):
            hdr_out = dict(
                Header(torchfits.read_header(str(path), 0 if hdu is None else hdu))
            )
    except _HEADER_PROBE_EXC:
        hdr_out = None
    if hdr_out is None:
        path = getattr(obs, "path")
        hdu = getattr(obs, "hdu", None)
        _, header = read(str(path), hdu=hdu, return_header=True)
        hdr_out = dict(header)
    obs_wcs = getattr(obs, "wcs", None)
    if obs_wcs is not None and hasattr(obs_wcs, "to_header"):
        hdr_out.update(obs_wcs.to_header())
    elif isinstance(obs_wcs, dict):
        hdr_out.update(obs_wcs)
    obs_header = getattr(obs, "header", None)
    if isinstance(obs_header, dict):
        hdr_out.update(obs_header)
    sidecar = _resolve_observation_sidecar_header(obs)
    if sidecar:
        hdr_out.update(sidecar)
    return hdr_out
