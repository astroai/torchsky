import pytest

from torchsky.core import FITSCore


def test_parse_cutout_spec_valid():
    file_path, hdu_index, slices = FITSCore.parse_cutout_spec(
        "myimage.fits[1][10:20,30:40]"
    )
    assert file_path == "myimage.fits"
    assert hdu_index == 1
    assert slices == (slice(10, 20, None), slice(30, 40, None))


def test_parse_cutout_spec_single_slice():
    file_path, hdu_index, slices = FITSCore.parse_cutout_spec(
        "myimage.fits[0][100:200]"
    )
    assert file_path == "myimage.fits"
    assert hdu_index == 0
    assert slices == (slice(100, 200, None),)


def test_parse_cutout_spec_no_start():
    file_path, hdu_index, slices = FITSCore.parse_cutout_spec(
        "myimage.fits[2][:100,50:]"
    )
    assert file_path == "myimage.fits"
    assert hdu_index == 2
    assert slices == (slice(None, 100, None), slice(50, None, None))


def test_parse_cutout_spec_single_index():
    file_path, hdu_index, slices = FITSCore.parse_cutout_spec(
        "myimage.fits[3][5,10:20]"
    )
    assert file_path == "myimage.fits"
    assert hdu_index == 3
    assert slices == (slice(5, 6, None), slice(10, 20, None))


def test_parse_cutout_spec_invalid():
    with pytest.raises(ValueError):
        FITSCore.parse_cutout_spec("myimage.fits")
    with pytest.raises(ValueError):
        FITSCore.parse_cutout_spec("myimage.fits[1]")
    with pytest.raises(ValueError):
        FITSCore.parse_cutout_spec("myimage.fits[]")
    with pytest.raises(ValueError):
        FITSCore.parse_cutout_spec("myimage.fits[1][,10]")

import torch
from torchsky.core import FITSDataTypeHandler

def test_apply_scaling_no_scaling():
    data = torch.tensor([1, 2, 3])
    result = FITSDataTypeHandler.apply_scaling(data)
    # Should return the same tensor object
    assert result is data

def test_apply_scaling_bscale_only():
    data = torch.tensor([1, 2, 3], dtype=torch.int32)
    result = FITSDataTypeHandler.apply_scaling(data, bscale=2.0)

    # Original shouldn't be modified
    assert torch.equal(data, torch.tensor([1, 2, 3], dtype=torch.int32))

    # Result should be scaled and float32
    assert result.dtype == torch.float32
    assert torch.allclose(result, torch.tensor([2.0, 4.0, 6.0]))

def test_apply_scaling_bzero_only():
    data = torch.tensor([1, 2, 3], dtype=torch.int32)
    result = FITSDataTypeHandler.apply_scaling(data, bzero=5.0)

    # Original shouldn't be modified
    assert torch.equal(data, torch.tensor([1, 2, 3], dtype=torch.int32))

    # Result should be offset and float32
    assert result.dtype == torch.float32
    assert torch.allclose(result, torch.tensor([6.0, 7.0, 8.0]))

def test_apply_scaling_both():
    data = torch.tensor([1, 2, 3], dtype=torch.int32)
    result = FITSDataTypeHandler.apply_scaling(data, bscale=2.0, bzero=5.0)

    # Original shouldn't be modified
    assert torch.equal(data, torch.tensor([1, 2, 3], dtype=torch.int32))

    # Result should be scaled + offset and float32
    assert result.dtype == torch.float32
    assert torch.allclose(result, torch.tensor([7.0, 9.0, 11.0]))
