"""Native extension module.

The compiled shared object is ``_cpp_ext``; this module loads it after importing
torch so the extension resolves libtorch symbols from the already-loaded torch
runtime (PyPI torch wheels do not add their lib directory to the loader path).
All native names are re-exported unchanged.
"""

from __future__ import annotations

import torch  # noqa: F401 - preload libtorch for the extension loader

from . import _cpp_ext as _ext

for _name in dir(_ext):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_ext, _name)
del _name
