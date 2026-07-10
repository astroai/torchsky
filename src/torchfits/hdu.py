from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class Header(dict):
    pass


class HDUList(list):
    pass


@dataclass
class DataView:
    data: Any = None
    header: Any = None


@dataclass
class TensorHDU:
    data: Any = None
    header: Any = None
    name: str | None = None


@dataclass
class TableHDU:
    data: Any = None
    header: Any = None
    name: str | None = None


@dataclass
class TableHDURef:
    path: str | None = None
    hdu: int | str | None = None


@dataclass
class TableDataAccessor:
    data: Any = None


@dataclass
class TensorFrame:
    data: Any = None
    header: Any = None
