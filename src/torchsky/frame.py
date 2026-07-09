"""TensorFrame ↔ FITS conversion helpers.

`torchfits.frame` was removed in torchfits 0.6.0.  These functions now live in
torchsky and bridge :mod:`torch_frame` with torchfits I/O.
"""

from __future__ import annotations

from typing import Any

import torch
import torchfits


def to_tensor_frame(
    data: dict[str, torch.Tensor],
    schema: dict[str, Any] | None = None,
) -> Any:
    """Convert a dict of column tensors into a :class:`torch_frame.TensorFrame`.

    Parameters
    ----------
    data:
        Mapping from column name to 1-D tensor.
    schema:
        Optional mapping from column name to ``torch_frame.stype``.  When
        omitted, float columns are classified as ``numerical`` and integer
        columns as ``categorical``.
    """
    from torch_frame import TensorFrame, stype as _stype

    if schema is None:
        schema = {}
        for col_name, col_tensor in data.items():
            if col_tensor.dtype in (torch.float32, torch.float64):
                schema[col_name] = _stype.numerical
            elif col_tensor.dtype in (
                torch.int32,
                torch.int64,
                torch.int16,
                torch.int8,
                torch.uint8,
            ):
                schema[col_name] = _stype.categorical
            else:
                schema[col_name] = _stype.numerical

    feat_dict: dict[Any, torch.Tensor] = {}
    col_names_dict: dict[Any, list[str]] = {}

    for col_name, col_tensor in data.items():
        st = schema.get(col_name, _stype.numerical)
        # Ensure 2-D: (N, 1)
        if col_tensor.ndim == 1:
            col_tensor = col_tensor.unsqueeze(-1)
        if st not in feat_dict:
            feat_dict[st] = col_tensor
            col_names_dict[st] = [col_name]
        else:
            feat_dict[st] = torch.cat([feat_dict[st], col_tensor], dim=-1)
            col_names_dict[st].append(col_name)

    return TensorFrame(feat_dict=feat_dict, col_names_dict=col_names_dict)


def write_tensor_frame(
    filename: str,
    tf: Any,
    overwrite: bool = True,
) -> None:
    """Write a :class:`~torch_frame.TensorFrame` to a FITS file.

    Multi-nested tensor columns are flattened to lists of 1-D tensors.
    """
    data: dict[str, Any] = {}

    for st, cols in tf.col_names_dict.items():
        for i, col_name in enumerate(cols):
            tensor_col = tf.feat_dict[st]

            # MultiNestedTensor-like objects — flatten first
            if (
                hasattr(tensor_col, "values")
                and hasattr(tensor_col, "offset")
                and not isinstance(tensor_col, torch.Tensor)
            ):
                # Extract column i from MultiNestedTensor (2-D indexed)
                col_slice = tensor_col[:, i]
                val = _flatten_multinested(col_slice)
            elif tensor_col.ndim == 2:
                # Standard (N, C) — extract single column as 1-D
                val = tensor_col[:, i]
            elif tensor_col.ndim == 3:
                # Embedded / 3-D tensor — extract (N, D)
                val = tensor_col[:, i, :]
            else:
                val = tensor_col[:, i]

            data[col_name] = val

    torchfits.write(filename, data, overwrite=overwrite)


def read_tensor_frame(filename: str) -> Any:
    """Read a FITS file back into a :class:`~torch_frame.TensorFrame`."""
    raw = torchfits.read(filename)
    if not isinstance(raw, dict):
        raise TypeError(f"Expected dict from torchfits.read, got {type(raw)}")

    data: dict[str, torch.Tensor] = {}
    for key, val in raw.items():
        if key.startswith("_"):
            continue
        if isinstance(val, torch.Tensor):
            data[key] = val
        elif isinstance(val, list):
            data[key] = torch.as_tensor(val)

    return to_tensor_frame(data)


def _flatten_multinested(mnt: Any) -> list[torch.Tensor]:
    """Convert a MultiNestedTensor to a list of 1-D tensors."""
    values = mnt.values
    offset = mnt.offset
    num_rows = getattr(mnt, "num_rows", len(offset) - 1)
    result = []
    for r in range(num_rows):
        start = int(offset[r])
        end = int(offset[r + 1])
        result.append(values[start:end])
    return result
