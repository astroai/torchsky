from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from astropy.io import fits as _fits


def _resolve_hdu(hdul: _fits.HDUList, hdu: int | str | None) -> int | str:
    if hdu is None:
        return 0
    if isinstance(hdu, str) and hdu.isdigit():
        return int(hdu)
    return hdu


def open(path: str | Path, *args: Any, **kwargs: Any):
    return _fits.open(path, *args, **kwargs)


def get_header(path: str | Path, hdu: int | str = 0, *args: Any, **kwargs: Any) -> dict:
    with _fits.open(path, memmap=False) as hdul:
        ext = _resolve_hdu(hdul, hdu)
        return dict(hdul[ext].header)


def read(
    path: str | Path,
    hdu: int | str | None = 0,
    return_header: bool = False,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    **kwargs: Any,
):
    with _fits.open(path, memmap=False) as hdul:
        ext = _resolve_hdu(hdul, hdu)
        data = np.asarray(hdul[ext].data)
        # FITS arrays are often big-endian; torch.as_tensor requires native byte order.
        if data.dtype.byteorder not in ("=", "|"):
            data = data.astype(data.dtype.newbyteorder("="), copy=False)
        else:
            data = np.asarray(data)
        header = dict(hdul[ext].header)

    tensor = torch.as_tensor(data)
    if dtype is not None:
        tensor = tensor.to(dtype)
    elif tensor.dtype == torch.float64:
        tensor = tensor.float()

    if device is not None:
        tensor = tensor.to(device)

    if return_header:
        return tensor, header
    return tensor


def write(
    path: str | Path,
    data: Any,
    header: dict | None = None,
    overwrite: bool = True,
    **kwargs: Any,
):
    arr = data.detach().cpu().numpy() if isinstance(data, torch.Tensor) else np.asarray(data)
    hdu = _fits.PrimaryHDU(arr)

    if header is not None:
        for key, value in dict(header).items():
            try:
                hdu.header[key] = value
            except Exception:
                pass

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    hdu.writeto(path, overwrite=overwrite)


def read_image(*args: Any, **kwargs: Any):
    return read(*args, **kwargs)


def _not_supported(*args: Any, **kwargs: Any):
    raise NotImplementedError("Local torchfits shim only supports image FITS I/O for CRF benchmark use.")


read_table = _not_supported
read_hdus = _not_supported
read_subset = _not_supported
read_batch = _not_supported
read_large_table = _not_supported
open_subset_reader = _not_supported
stream_table = _not_supported
delete_hdu = _not_supported
insert_hdu = _not_supported
replace_hdu = _not_supported


def get_batch_info(*args: Any, **kwargs: Any):
    return {}


def get_cache_performance(*args: Any, **kwargs: Any):
    return {}


def clear_file_cache(*args: Any, **kwargs: Any):
    return None


def verify_checksums(*args: Any, **kwargs: Any):
    return True


def write_checksums(*args: Any, **kwargs: Any):
    return None


def _invalidate_path_caches(*args: Any, **kwargs: Any):
    return None


def _get_image_meta(path: str | Path, hdu: int = 0):
    with _fits.open(path, memmap=False) as hdul:
        data = hdul[hdu].data
        return {"shape": data.shape, "dtype": str(data.dtype)}


def _resolve_image_mmap(*args: Any, **kwargs: Any):
    return False


def _should_use_cold_nommap(*args: Any, **kwargs: Any):
    return True


create_dataloader = _not_supported
create_fits_dataloader = _not_supported
create_streaming_dataloader = _not_supported
create_table_dataloader = _not_supported
