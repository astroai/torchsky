import numpy as np
import pytest
import torch

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


def test_process_data_base_case():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    header = {}
    tensor = FITSCore.process_data(data, header)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert torch.allclose(tensor, torch.tensor([1.0, 2.0, 3.0]))

def test_process_data_byteorder():
    # Big-endian float32
    data = np.array([1.0, 2.0, 3.0], dtype=">f4")
    header = {}
    tensor = FITSCore.process_data(data, header)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert torch.allclose(tensor, torch.tensor([1.0, 2.0, 3.0]))

def test_process_data_scaling_float64():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    header = {"BSCALE": 2.0, "BZERO": 1.0}
    tensor = FITSCore.process_data(data, header)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float64
    assert torch.allclose(tensor, torch.tensor([3.0, 5.0, 7.0], dtype=torch.float64))

def test_process_data_scaling_float32():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    header = {"BSCALE": 2.0, "BZERO": 1.0}
    tensor = FITSCore.process_data(data, header)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert torch.allclose(tensor, torch.tensor([3.0, 5.0, 7.0]))

def test_process_data_scaling_int():
    data = np.array([1, 2, 3], dtype=np.int32)
    header = {"BSCALE": 2.0, "BZERO": 1.0}
    tensor = FITSCore.process_data(data, header)
    assert isinstance(tensor, torch.Tensor)
    # Should convert to float32
    assert tensor.dtype == torch.float32
    assert torch.allclose(tensor, torch.tensor([3.0, 5.0, 7.0]))

def test_process_data_partial_scaling():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    header_bscale = {"BSCALE": 2.0}
    tensor1 = FITSCore.process_data(data, header_bscale)
    assert torch.allclose(tensor1, torch.tensor([2.0, 4.0, 6.0]))

    header_bzero = {"BZERO": 1.0}
    tensor2 = FITSCore.process_data(data, header_bzero)
    assert torch.allclose(tensor2, torch.tensor([2.0, 3.0, 4.0]))
