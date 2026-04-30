import pytest
import torch

from torchsky.core import FITSCore, CompressionType


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


def test_get_data_info_naxis_0():
    header = {"NAXIS": 0}
    info = FITSCore.get_data_info(header)
    assert info == {"shape": (), "dtype": None, "compressed": False}


def test_get_data_info_standard_2d():
    header = {
        "NAXIS": 2,
        "NAXIS1": 100,
        "NAXIS2": 200,
        "BITPIX": 32,
    }
    info = FITSCore.get_data_info(header)
    assert info["shape"] == (200, 100)  # FORTRAN order
    assert info["dtype"] == torch.int32
    assert info["bitpix"] == 32
    assert info["compressed"] is False
    assert info["compression_type"] == CompressionType.NONE
    assert info["bzero"] == 0.0
    assert info["bscale"] == 1.0


def test_get_data_info_unsupported_bitpix():
    header = {
        "NAXIS": 1,
        "NAXIS1": 10,
        "BITPIX": 15,  # Unsupported
    }
    info = FITSCore.get_data_info(header)
    assert info["dtype"] == torch.float32  # Fallback


def test_get_data_info_compressed():
    header = {
        "NAXIS": 2,
        "NAXIS1": 50,
        "NAXIS2": 50,
        "BITPIX": -32,
        "ZCMPTYPE": "RICE_1",
        "ZTILE1": 10,
        "ZTILE2": 10,
    }
    info = FITSCore.get_data_info(header)
    assert info["compressed"] is True
    assert info["compression_type"] == CompressionType.RICE
    assert info["compression_params"]["tile_dims"] == (10, 10)


def test_get_data_info_bzero_bscale():
    header = {
        "NAXIS": 1,
        "NAXIS1": 10,
        "BITPIX": 16,
        "BZERO": 32768.0,
        "BSCALE": 2.5,
    }
    info = FITSCore.get_data_info(header)
    assert info["bzero"] == 32768.0
    assert info["bscale"] == 2.5
