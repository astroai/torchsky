"""HDU types — delegated to torchfits (Card kept locally until torchfits 0.5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torchfits.hdu import (
    DataView as DataView,
    HDUList as HDUList,
    Header as Header,
    TableDataAccessor as TableDataAccessor,
    TableHDU as TableHDU,
    TableHDURef as TableHDURef,
    TensorFrame as TensorFrame,
    TensorHDU as TensorHDU,
)


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
    "TensorFrame",
    "TensorHDU",
]
