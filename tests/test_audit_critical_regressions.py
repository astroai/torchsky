"""Regression tests for historically critical correctness issues (see AUDIT_REPORT.md).

These guard against:
  * ``MOC.flatten`` dense mask living on a different device than UNIQ pixels.
  * FITS complex-column plumbing treating ``_COMPLEX_DTYPE_MAP`` as a mere code set
    (must remain a mapping for dtype lookup).
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch

from torchsky.sphere.moc import MOC, nest2uniq


def test_moc_flatten_dense_mask_matches_uniq_device_cpu() -> None:
    u = nest2uniq(4, torch.tensor([0, 1], dtype=torch.int64))
    moc = MOC(u)
    mask = moc.flatten(4)
    assert mask.dtype == torch.bool
    assert mask.device == moc.uniq.device
    assert int(mask.sum().item()) >= 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_moc_flatten_dense_mask_matches_uniq_device_cuda() -> None:
    dev = torch.device("cuda")
    u = nest2uniq(4, torch.tensor([0, 1], dtype=torch.int64, device=dev))
    moc = MOC(u)
    mask = moc.flatten(4)
    assert mask.device.type == "cuda"
    assert mask.device == moc.uniq.device


def test_complex_tform_map_remains_subscriptable_after_update_rows() -> None:
    pytest.importorskip("torchsky.cpp")
    import torchsky
    import torchsky.table as tbl

    path = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    path.close()
    try:
        torchsky.write(
            path.name,
            {
                "ID": np.array([1, 2], dtype=np.int32),
                "Z": np.array([1 + 2j, 3 + 4j], dtype=np.complex64),
            },
            overwrite=True,
        )
        torchsky.table.update_rows(
            path.name,
            {"ID": np.array([10, 20], dtype=np.int32)},
            row_slice=slice(0, 2),
            hdu=1,
        )
        assert isinstance(tbl._COMPLEX_DTYPE_MAP, dict)
        assert np.dtype(tbl._COMPLEX_DTYPE_MAP["C"]) == np.dtype(np.complex64)
        assert np.dtype(tbl._COMPLEX_DTYPE_MAP["M"]) == np.dtype(np.complex128)
        assert "C" in tbl._COMPLEX_TFORM_CODES
    finally:
        os.unlink(path.name)
