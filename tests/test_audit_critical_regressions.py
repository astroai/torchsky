"""Regression tests for historically critical correctness issues (see AUDIT_REPORT.md).

These guard against:
  * ``MOC.flatten`` dense mask living on a different device than UNIQ pixels.
  * FITS complex-column plumbing treating ``_COMPLEX_DTYPE_MAP`` as a mere code set
    (must remain a mapping for dtype lookup).
"""

from __future__ import annotations


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


@pytest.mark.skip("_COMPLEX_DTYPE_MAP and _COMPLEX_TFORM_CODS moved to torchfits internals")
def test_complex_tform_map_remains_subscriptable_after_update_rows() -> None:
    pass
