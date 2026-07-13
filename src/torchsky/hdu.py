"""HDU types delegated to the published torchfits public boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torchfits.hdu as _fits_hdu

HDUList = _fits_hdu.HDUList
Header = _fits_hdu.Header
TableHDU = _fits_hdu.TableHDU
TableHDURef = _fits_hdu.TableHDURef
TensorHDU = _fits_hdu.TensorHDU

# These names were added after the published 0.7 wheel.  They are only type
# surface aliases in torchsky, so object is the safe compatibility fallback.
DataView = getattr(_fits_hdu, "DataView", object)
TableDataAccessor = getattr(_fits_hdu, "TableDataAccessor", object)


@dataclass(frozen=True)
class Card:
    """One FITS header card."""

    key: str
    value: Any = None
    comment: str = ""

    @property
    def keyword(self) -> str:
        return self.key

    def __iter__(self):
        yield self.key
        yield self.value
        yield self.comment

    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int) -> Any:
        return (self.key, self.value, self.comment)[index]


__all__ = [
    "Card",
    "DataView",
    "HDUList",
    "Header",
    "TableDataAccessor",
    "TableHDU",
    "TableHDURef",
    "TensorHDU",
]
