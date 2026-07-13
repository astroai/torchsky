"""Lean public API for torchsky.

The package root intentionally stays light: importing :mod:`torchsky` must not
load tensor runtimes, NumPy, compiled extensions, map models, or optional
astronomy ecosystem packages. Domain APIs live under their own namespaces and
are imported on first access.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__version__ = "0.4.0.dev0"

_NAMESPACES: dict[str, str] = {
    "io": "torchsky.io",
    "table": "torchsky.table",
    "wcs": "torchsky.wcs",
    "sphere": "torchsky.sphere",
    "spectral": "torchsky.spectral",
    "spectrophotometry": "torchsky.spectrophotometry",
    "map": "torchsky.map",
    "scene": "torchsky.scene",
    "field": "torchsky.field",
    "transforms": "torchsky.transforms",
    "datasets": "torchsky.datasets",
    "cache": "torchsky.cache",
    "interop": "torchfits.interop",
    "encoders": "torchsky.encoders",
    "catalogs": "torchsky.catalogs",
    "quality": "torchsky.quality",
}

_ROOT_FUNCTIONS: dict[str, tuple[str, str]] = {
    "read": ("torchsky.io", "read"),
    "write": ("torchsky.io.fits", "write"),
    "open": ("torchsky.io.fits", "open"),
    "get_header": ("torchsky.io.fits", "get_header"),
    "get_wcs": ("torchsky.io.fits", "get_wcs"),
}

_ROOT_OBJECTS: dict[str, tuple[str, str]] = {
    "WCS": ("torchsky.wcs", "WCS"),
    "Header": ("torchsky.hdu", "Header"),
    "Card": ("torchsky.hdu", "Card"),
    "HDUList": ("torchsky.hdu", "HDUList"),
    "TensorHDU": ("torchsky.hdu", "TensorHDU"),
    "TableHDU": ("torchsky.hdu", "TableHDU"),
}

_INTERNAL_ATTRS: dict[str, tuple[str, str | None]] = {
    "ChecksumVerifier": ("torchsky.core", "ChecksumVerifier"),
    "TableHDURef": ("torchsky.hdu", "TableHDURef"),
    "read_image": ("torchsky.io.fits", "read_image"),
    "read_table": ("torchsky.io.fits", "read_table"),
    "read_hdus": ("torchsky.io.fits", "read_hdus"),
    "read_subset": ("torchsky.io.fits", "read_subset"),
    "open_subset_reader": ("torchsky.io.fits", "open_subset_reader"),
    "read_batch": ("torchsky.io.fits", "read_batch"),
    "get_batch_info": ("torchsky.io.fits", "get_batch_info"),
    "stream_table": ("torchsky.io.fits", "stream_table"),
    "insert_hdu": ("torchsky.io.fits", "insert_hdu"),
    "replace_hdu": ("torchsky.io.fits", "replace_hdu"),
    "delete_hdu": ("torchsky.io.fits", "delete_hdu"),
    "write_checksums": ("torchsky.io.fits", "write_checksums"),
    "verify_checksums": ("torchsky.io.fits", "verify_checksums"),
    "get_cache_performance": ("torchsky.io.fits", "get_cache_performance"),
    "clear_file_cache": ("torchsky.io.fits", "clear_file_cache"),
    "FitsImageDataset": ("torchsky.datasets", "FitsImageDataset"),
    "FitsImageIterableDataset": (
        "torchsky.datasets",
        "FitsImageIterableDataset",
    ),
    # Legacy torchsky aliases; torchfits no longer provides these names.
    "FITSDataset": ("torchsky.datasets", "FITSDataset"),
    "IterableFITSDataset": ("torchsky.datasets", "IterableFITSDataset"),
    "TableChunkDataset": ("torchsky.datasets", "TableChunkDataset"),
    "create_dataloader": ("torchsky.dataloader", "create_dataloader"),
    "create_fits_dataloader": ("torchsky.dataloader", "create_fits_dataloader"),
    "create_streaming_dataloader": (
        "torchsky.dataloader",
        "create_streaming_dataloader",
    ),
    "create_table_dataloader": ("torchsky.dataloader", "create_table_dataloader"),
    "AsinhStretch": ("torchsky.transforms", "AsinhStretch"),
    "CenterCrop": ("torchsky.transforms", "CenterCrop"),
    "Compose": ("torchsky.transforms", "Compose"),
    "GaussianNoise": ("torchsky.transforms", "GaussianNoise"),
    "LogStretch": ("torchsky.transforms", "LogStretch"),
    "MinMaxScale": ("torchsky.transforms", "MinMaxScale"),
    "Normalize": ("torchsky.transforms", "Normalize"),
    "PerturbByError": ("torchsky.transforms", "PerturbByError"),
    "PoissonNoise": ("torchsky.transforms", "PoissonNoise"),
    "PowerStretch": ("torchsky.transforms", "PowerStretch"),
    "RandomCrop": ("torchsky.transforms", "RandomCrop"),
    "RandomFlip": ("torchsky.transforms", "RandomFlip"),
    "RandomRotation": ("torchsky.transforms", "RandomRotation"),
    "RedshiftShift": ("torchsky.transforms", "RedshiftShift"),
    "RobustScale": ("torchsky.transforms", "RobustScale"),
    "ToDevice": ("torchsky.transforms", "ToDevice"),
    "ZScale": ("torchsky.transforms", "ZScale"),
    "create_inference_transform": (
        "torchsky.transforms",
        "create_inference_transform",
    ),
    "create_training_transform": ("torchsky.transforms", "create_training_transform"),
    "create_validation_transform": (
        "torchsky.transforms",
        "create_validation_transform",
    ),
    "fast_parse_header": ("torchsky.header_parser", "fast_parse_header"),
    "configure_for_environment": ("torchfits.cache", "configure_for_environment"),
    "get_cache_stats": ("torchfits.cache", "get_cache_stats"),
    "clear_cache": ("torchfits.cache", "clear_cache"),
    "to_arrow": ("torchfits.interop", "to_arrow"),
    "to_pandas": ("torchfits.interop", "to_pandas"),
    "_invalidate_path_caches": ("torchsky.io.fits", "_invalidate_path_caches"),
    "_resolve_image_mmap": ("torchsky.io.fits", "_resolve_image_mmap"),
}

__all__ = tuple(
    [
        "read",
        "write",
        "open",
        "get_header",
        "get_wcs",
        "WCS",
        "Header",
        "Card",
        "HDUList",
        "TensorHDU",
        "TableHDU",
        *_NAMESPACES,
    ]
)


_RUNTIME_INITIALIZED = False


def _ensure_runtime_init() -> None:
    """Initialize optional runtime caches when an I/O/WCS entry point is used."""

    global _RUNTIME_INITIALIZED
    if _RUNTIME_INITIALIZED:
        return

    cache = import_module("torchfits.cache")
    cache.configure_for_environment()

    _RUNTIME_INITIALIZED = True


def _runtime_function(name: str) -> Any:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        _ensure_runtime_init()
        module_name, attr_name = _ROOT_FUNCTIONS[name]
        return getattr(import_module(module_name), attr_name)(*args, **kwargs)

    wrapper.__name__ = name
    wrapper.__qualname__ = name
    wrapper.__module__ = __name__
    return wrapper


def __getattr__(name: str) -> Any:
    if name in _NAMESPACES:
        module = import_module(_NAMESPACES[name])
        globals()[name] = module
        return module

    if name in _ROOT_FUNCTIONS:
        function = _runtime_function(name)
        globals()[name] = function
        return function

    if name in _ROOT_OBJECTS:
        module_name, attr_name = _ROOT_OBJECTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value

    if name in _INTERNAL_ATTRS:
        module_name, attr_name = _INTERNAL_ATTRS[name]
        module = import_module(module_name)
        value = module if attr_name is None else getattr(module, attr_name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | set(_INTERNAL_ATTRS))


if TYPE_CHECKING:
    from . import (
        cache as cache,
        catalogs as catalogs,
        datasets as datasets,
        encoders as encoders,
        field as field,
        io as io,
        map as map,
        scene as scene,
        sphere as sphere,
        spectral as spectral,
        spectrophotometry as spectrophotometry,
        table as table,
        transforms as transforms,
        wcs as wcs,
    )
    from .hdu import Card as Card
    from .hdu import HDUList as HDUList
    from .hdu import Header as Header
    from .hdu import TableHDU as TableHDU
    from .hdu import TensorHDU as TensorHDU
    from .io import get_header as get_header
    from .io import get_wcs as get_wcs
    from .io import open as open
    from .io import read as read
    from .io import write as write
    from .wcs import WCS as WCS
