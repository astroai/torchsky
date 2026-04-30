import numpy as np
import pytest
import torch

from torchsky.core import (
    ChecksumVerifier,
    CompressionHandler,
    CompressionType,
    FITSCore,
    FITSDataTypeHandler,
)


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


def test_to_torch_dtype_supported():
    assert FITSDataTypeHandler.to_torch_dtype(8) == torch.uint8
    assert FITSDataTypeHandler.to_torch_dtype(16) == torch.int16
    assert FITSDataTypeHandler.to_torch_dtype(32) == torch.int32
    assert FITSDataTypeHandler.to_torch_dtype(64) == torch.int64
    assert FITSDataTypeHandler.to_torch_dtype(-32) == torch.float32
    assert FITSDataTypeHandler.to_torch_dtype(-64) == torch.float64


def test_to_torch_dtype_unsupported():
    with pytest.raises(ValueError, match="Unsupported BITPIX: 10"):
        FITSDataTypeHandler.to_torch_dtype(10)
    with pytest.raises(ValueError, match="Unsupported BITPIX: -16"):
        FITSDataTypeHandler.to_torch_dtype(-16)


def test_to_numpy_dtype_supported():
    assert FITSDataTypeHandler.to_numpy_dtype(8) == np.uint8
    assert FITSDataTypeHandler.to_numpy_dtype(16) == np.int16
    assert FITSDataTypeHandler.to_numpy_dtype(32) == np.int32
    assert FITSDataTypeHandler.to_numpy_dtype(64) == np.int64
    assert FITSDataTypeHandler.to_numpy_dtype(-32) == np.float32
    assert FITSDataTypeHandler.to_numpy_dtype(-64) == np.float64


def test_to_numpy_dtype_unsupported():
    with pytest.raises(ValueError, match="Unsupported BITPIX: 42"):
        FITSDataTypeHandler.to_numpy_dtype(42)


def test_to_numpy_dtype_initializes_cache():
    original_cache = FITSDataTypeHandler._BITPIX_TO_NUMPY
    try:
        FITSDataTypeHandler._BITPIX_TO_NUMPY = None
        assert FITSDataTypeHandler.to_numpy_dtype(8) == np.uint8
        assert FITSDataTypeHandler._BITPIX_TO_NUMPY is not None
    finally:
        FITSDataTypeHandler._BITPIX_TO_NUMPY = original_cache


def test_apply_scaling_no_scaling_returns_original_tensor():
    data = torch.tensor([1, 2, 3], dtype=torch.int32)
    assert FITSDataTypeHandler.apply_scaling(data) is data


def test_apply_scaling_applies_bscale_without_mutating_input():
    data = torch.tensor([1, 2, 3], dtype=torch.int32)
    result = FITSDataTypeHandler.apply_scaling(data, bscale=2.0)

    assert torch.equal(data, torch.tensor([1, 2, 3], dtype=torch.int32))
    assert result.dtype == torch.float32
    assert torch.allclose(result, torch.tensor([2.0, 4.0, 6.0]))


def test_apply_scaling_applies_bzero_without_mutating_input():
    data = torch.tensor([1, 2, 3], dtype=torch.int32)
    result = FITSDataTypeHandler.apply_scaling(data, bzero=5.0)

    assert torch.equal(data, torch.tensor([1, 2, 3], dtype=torch.int32))
    assert result.dtype == torch.float32
    assert torch.allclose(result, torch.tensor([6.0, 7.0, 8.0]))


def test_apply_scaling_applies_bscale_and_bzero():
    data = torch.tensor([1, 2, 3], dtype=torch.int32)
    result = FITSDataTypeHandler.apply_scaling(data, bscale=2.0, bzero=5.0)

    assert torch.equal(data, torch.tensor([1, 2, 3], dtype=torch.int32))
    assert result.dtype == torch.float32
    assert torch.allclose(result, torch.tensor([7.0, 9.0, 11.0]))


def test_detect_compression_none_or_empty():
    assert CompressionHandler.detect_compression({}) == (CompressionType.NONE, {})
    assert CompressionHandler.detect_compression({"ZCMPTYPE": "   "}) == (
        CompressionType.NONE,
        {},
    )


def test_detect_compression_valid_type_and_tiles():
    header = {"ZCMPTYPE": "RICE_1", "ZTILE1": 100, "ZTILE2": 200}
    comp_type, params = CompressionHandler.detect_compression(header)

    assert comp_type == CompressionType.RICE
    assert params == {"tile_dims": (100, 200)}


def test_detect_compression_unknown_type_keeps_tile_metadata():
    header = {"ZCMPTYPE": "UNKNOWN_COMPRESSION", "ZTILE1": 50, "ZTILE2": 50}
    comp_type, params = CompressionHandler.detect_compression(header)

    assert comp_type == CompressionType.NONE
    assert params == {"tile_dims": (50, 50)}


def test_detect_compression_strips_type_whitespace():
    header = {"ZCMPTYPE": "  GZIP_1  ", "ZTILE1": 256, "ZTILE2": 256}
    comp_type, params = CompressionHandler.detect_compression(header)

    assert comp_type == CompressionType.GZIP
    assert params == {"tile_dims": (256, 256)}


def test_is_compressed():
    assert CompressionHandler.is_compressed({"SIMPLE": True}) is False
    assert CompressionHandler.is_compressed({"ZCMPTYPE": ""}) is False
    assert CompressionHandler.is_compressed({"ZCMPTYPE": "   "}) is False
    assert CompressionHandler.is_compressed({"ZCMPTYPE": "GZIP_1"}) is True


def test_checksum_verifier_in_memory_methods_fail_closed():
    with pytest.raises(NotImplementedError, match="verify_hdu_checksums"):
        ChecksumVerifier.verify_datasum(np.arange(3), "0")

    with pytest.raises(NotImplementedError, match="verify_hdu_checksums"):
        ChecksumVerifier.verify_checksum({"CHECKSUM": "unused"})


def test_get_data_info_naxis_zero():
    assert FITSCore.get_data_info({"NAXIS": 0}) == {
        "shape": (),
        "dtype": None,
        "compressed": False,
    }


def test_get_data_info_standard_2d_header():
    header = {"NAXIS": 2, "NAXIS1": 100, "NAXIS2": 200, "BITPIX": 32}
    info = FITSCore.get_data_info(header)

    assert info["shape"] == (200, 100)
    assert info["dtype"] == torch.int32
    assert info["bitpix"] == 32
    assert info["compressed"] is False
    assert info["compression_type"] == CompressionType.NONE
    assert info["bzero"] == 0.0
    assert info["bscale"] == 1.0


def test_get_data_info_falls_back_for_unsupported_bitpix():
    info = FITSCore.get_data_info({"NAXIS": 1, "NAXIS1": 10, "BITPIX": 15})
    assert info["dtype"] == torch.float32


def test_get_data_info_compressed_header():
    header = {
        "NAXIS": 2,
        "NAXIS1": 50,
        "NAXIS2": 60,
        "BITPIX": -32,
        "ZCMPTYPE": "RICE_1",
        "ZTILE1": 10,
        "ZTILE2": 20,
    }
    info = FITSCore.get_data_info(header)

    assert info["shape"] == (60, 50)
    assert info["compressed"] is True
    assert info["compression_type"] == CompressionType.RICE
    assert info["compression_params"]["tile_dims"] == (10, 20)


def test_get_data_info_preserves_scaling_keywords():
    header = {"NAXIS": 1, "NAXIS1": 10, "BITPIX": 16, "BZERO": 32768.0, "BSCALE": 2.5}
    info = FITSCore.get_data_info(header)

    assert info["bzero"] == 32768.0
    assert info["bscale"] == 2.5


def test_process_data_base_case_zero_copy_dtype():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    tensor = FITSCore.process_data(data, {})

    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert torch.allclose(tensor, torch.tensor([1.0, 2.0, 3.0]))


def test_process_data_normalizes_big_endian_input():
    data = np.array([1.0, 2.0, 3.0], dtype=">f4")
    tensor = FITSCore.process_data(data, {})

    assert tensor.dtype == torch.float32
    assert torch.allclose(tensor, torch.tensor([1.0, 2.0, 3.0]))


def test_process_data_applies_fits_scaling_without_mutating_input():
    data = np.array([1, 2, 3], dtype=np.int16)
    tensor = FITSCore.process_data(data, {"BSCALE": 2.0, "BZERO": 5.0})

    np.testing.assert_array_equal(data, np.array([1, 2, 3], dtype=np.int16))
    assert tensor.dtype == torch.float32
    assert torch.allclose(tensor, torch.tensor([7.0, 9.0, 11.0]))
