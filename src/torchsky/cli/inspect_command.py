"""Read-only FITS inspection commands."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from .. import open as open_fits
from .. import read
from ..hdu import Header, TableHDU, TableHDURef, TensorHDU


def _json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return _json_value(value.item())
        return [_json_value(item) for item in value.detach().cpu().tolist()]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Mapping):
        return {str(key): _json_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_value(value.item())
        except Exception:
            pass
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _header_json(header: Header) -> dict[str, Any]:
    return {str(key): _json_value(value) for key, value in header.items()}


def _header_card_count(header: Header) -> int:
    return len(header.cards) if hasattr(header, "cards") else len(header)


def _hdu_shape(hdu: TensorHDU | TableHDU | TableHDURef) -> list[int] | None:
    if isinstance(hdu, TensorHDU):
        naxis = int(hdu.header.get("NAXIS", 0) or 0)
        if naxis <= 0:
            return []
        dims = [int(hdu.header.get(f"NAXIS{i}", 0) or 0) for i in range(1, naxis + 1)]
        return list(reversed(dims))
    return None


def _hdu_dtype(hdu: TensorHDU | TableHDU | TableHDURef) -> str | None:
    if isinstance(hdu, TensorHDU):
        return hdu._get_dtype_str()
    return None


def _hdu_kind(index: int, hdu: TensorHDU | TableHDU | TableHDURef) -> str:
    if isinstance(hdu, (TableHDU, TableHDURef)):
        return "table"
    if index == 0:
        return "primary_image"
    return "image"


def _hdu_info(index: int, hdu: TensorHDU | TableHDU | TableHDURef) -> dict[str, Any]:
    header = hdu.header
    info: dict[str, Any] = {
        "index": index,
        "name": str(header.get("EXTNAME", "PRIMARY" if index == 0 else "")),
        "kind": _hdu_kind(index, hdu),
        "cards": _header_card_count(header),
        "shape": _hdu_shape(hdu),
        "dtype": _hdu_dtype(hdu),
    }
    if isinstance(hdu, (TableHDU, TableHDURef)):
        info.update(
            {
                "rows": int(hdu.num_rows),
                "columns": list(hdu.columns),
            }
        )
    return info


def _info_payload(path: str, *, include_headers: bool) -> dict[str, Any]:
    with open_fits(path) as hdul:
        hdus = [_hdu_info(i, hdu) for i, hdu in enumerate(hdul._hdus)]
        if include_headers:
            for item, hdu in zip(hdus, hdul._hdus):
                item["header"] = _header_json(hdu.header)
        return {"path": path, "hdu_count": len(hdul), "hdus": hdus}


def _print_info_text(payload: dict[str, Any], *, include_headers: bool) -> None:
    print(f"File: {payload['path']}")
    print("No.  Name         Type           Shape/Rows        Format/Columns")
    for hdu in payload["hdus"]:
        shape = hdu["shape"]
        if hdu["kind"] == "table":
            extent = f"{hdu['rows']} rows"
            fmt = f"{len(hdu['columns'])} columns"
        else:
            extent = "x".join(str(dim) for dim in shape) if shape is not None else ""
            fmt = hdu["dtype"] or ""
        print(
            f"{hdu['index']:<4} {hdu['name']:<12} {hdu['kind']:<14} "
            f"{extent:<16} {fmt}"
        )
        if include_headers:
            for key, value in hdu.get("header", {}).items():
                print(f"    {key} = {value}")


def _split_columns(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    cols = [item.strip() for item in raw.split(",") if item.strip()]
    return cols or None


def _parse_hdu(raw: int | str) -> int | str:
    if isinstance(raw, int):
        return raw
    text = str(raw).strip()
    if text.isdecimal():
        return int(text)
    return text


def _table_num_rows(data: Mapping[str, Any]) -> int:
    if not data:
        return 0
    first = next(iter(data.values()))
    try:
        return int(len(first))
    except TypeError:
        return 1


def _row_value(column: Any, row_index: int) -> Any:
    if isinstance(column, torch.Tensor):
        item = column[row_index]
        return _json_value(item)
    return _json_value(column[row_index])


def _rows_from_columns(data: Mapping[str, Any], *, max_rows: int) -> list[dict[str, Any]]:
    nrows = min(max(0, int(max_rows)), _table_num_rows(data))
    names = [str(name) for name in data.keys()]
    return [
        {name: _row_value(data[name], row_index) for name in names}
        for row_index in range(nrows)
    ]


def _cat_payload(
    path: str,
    *,
    hdu: int | str,
    columns: Sequence[str] | None,
    rows: int,
    start_row: int,
) -> dict[str, Any]:
    data = read(
        path,
        hdu=hdu,
        mode="table",
        columns=list(columns) if columns is not None else None,
        start_row=start_row,
        num_rows=rows,
        mmap=False,
    )
    if not isinstance(data, Mapping):
        raise RuntimeError(f"{path}: selected HDU did not return a table")
    names = [str(name) for name in columns] if columns is not None else [str(name) for name in data.keys()]
    return {
        "path": path,
        "hdu": hdu,
        "start_row": start_row,
        "columns": names,
        "rows": _rows_from_columns({name: data[name] for name in names}, max_rows=rows),
    }


def _print_cat_text(payload: dict[str, Any]) -> None:
    columns = list(payload["columns"])
    print("\t".join(columns))
    for row in payload["rows"]:
        print("\t".join(str(row.get(name, "")) for name in columns))


def _info_handler(args: argparse.Namespace) -> int:
    payload = _info_payload(str(args.path), include_headers=bool(args.headers))
    if args.json:
        json.dump(payload, sys.stdout, indent=2, sort_keys=True)
        print()
    else:
        _print_info_text(payload, include_headers=bool(args.headers))
    return 0


def _cat_handler(args: argparse.Namespace) -> int:
    payload = _cat_payload(
        str(args.path),
        hdu=_parse_hdu(args.hdu),
        columns=_split_columns(args.columns),
        rows=int(args.rows),
        start_row=int(args.start_row),
    )
    if args.json:
        json.dump(payload, sys.stdout, indent=2, sort_keys=True)
        print()
    else:
        _print_cat_text(payload)
    return 0


def register_inspect_parsers(subparsers: argparse.Action) -> None:
    info = subparsers.add_parser(
        "info",
        help="Summarize FITS HDUs without materializing data.",
    )
    info.add_argument("path", help="FITS file to inspect.")
    info.add_argument("--json", action="store_true", help="Write JSON to stdout.")
    info.add_argument(
        "--headers",
        action="store_true",
        help="Include per-HDU FITS headers in the output.",
    )
    info.set_defaults(handler=_info_handler)

    cat = subparsers.add_parser(
        "cat",
        help="Print a bounded sample of a FITS table HDU.",
    )
    cat.add_argument("path", help="FITS file containing a table HDU.")
    cat.add_argument("--hdu", default=1, help="Table HDU index or EXTNAME (default: 1).")
    cat.add_argument(
        "--columns",
        default=None,
        help="Comma-separated column names to read.",
    )
    cat.add_argument(
        "--rows",
        type=int,
        default=10,
        help="Maximum number of rows to print (default: 10).",
    )
    cat.add_argument(
        "--start-row",
        type=int,
        default=1,
        help="1-based first table row to read (default: 1).",
    )
    cat.add_argument("--json", action="store_true", help="Write JSON to stdout.")
    cat.set_defaults(handler=_cat_handler)
