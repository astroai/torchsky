"""Low-level deterministic FITS image reads."""

from __future__ import annotations

from typing import Callable, Tuple, Union

import torch
from torch import Tensor

from ..hdu import Header


def validate_read_image_args(
    path: str, hdu: int, mmap: bool, handle_cache: bool, device: str
) -> None:
    """Validate arguments for low-level read_image."""
    if not isinstance(path, str) or not path:
        raise ValueError("path must be a non-empty string")
    if not isinstance(hdu, int) or hdu < 0:
        raise ValueError("hdu must be a non-negative integer")
    if not isinstance(mmap, bool):
        raise ValueError("read_image requires explicit mmap=True/False")
    if not isinstance(handle_cache, bool):
        raise ValueError("handle_cache must be bool")
    if device not in ["cpu", "cuda", "mps"] and not device.startswith("cuda:"):
        raise ValueError("device must be 'cpu', 'cuda', 'mps' or 'cuda:N'")


def dispatch_read_image_cpp(
    cpp, path: str, hdu: int, mmap: bool, handle_cache: bool, raw_scale: bool
) -> Tensor:
    """Dispatch the correct C++ function for low-level image reading."""
    if raw_scale:
        if not mmap and hasattr(cpp, "read_full_unmapped_raw"):
            return cpp.read_full_unmapped_raw(path, hdu)
        elif hasattr(cpp, "read_full_raw"):
            return cpp.read_full_raw(path, hdu, mmap)
        else:
            return cpp.read_full(path, hdu, mmap)
    else:
        if handle_cache and hasattr(cpp, "read_full_cached"):
            return cpp.read_full_cached(path, hdu, mmap)
        elif not mmap and hasattr(cpp, "read_full_unmapped"):
            return cpp.read_full_unmapped(path, hdu)
        elif hasattr(cpp, "read_full_nocache"):
            return cpp.read_full_nocache(path, hdu, mmap)
        else:
            return cpp.read_full(path, hdu, mmap)


def read_image(
    path: str,
    hdu: int = 0,
    device: str = "cpu",
    mmap: bool = True,
    handle_cache: bool = True,
    fp16: bool = False,
    bf16: bool = False,
    raw_scale: bool = False,
    return_header: bool = False,
    fallback_get_header: Callable[[str, int], Header] | None = None,
) -> Union[Tensor, Tuple[Tensor, Header]]:
    """Read image data through a direct low-level path."""
    import torchsky.cpp as cpp

    validate_read_image_args(path, hdu, mmap, handle_cache, device)

    data = dispatch_read_image_cpp(cpp, path, hdu, mmap, handle_cache, raw_scale)

    if fp16:
        data = data.to(torch.float16)
    elif bf16:
        data = data.to(torch.bfloat16)

    if device != "cpu" and data.device.type == "cpu":
        data = data.to(device)

    if return_header:
        try:
            return data, Header(cpp.read_header_dict(path, hdu))
        except Exception:
            if fallback_get_header is None:
                raise
            return data, fallback_get_header(path, hdu)
    return data
