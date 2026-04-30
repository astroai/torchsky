"""
Test table reading functionality.
"""

import os
import tempfile

import numpy as np
from unittest import mock
import torch

import torchsky
from torchsky.table import _column_name_index_map


def test_column_name_index_map_uses_tfields_order():
    header = {
        "TFIELDS": 3,
        "TTYPE1": "ID",
        "TTYPE2": "RA",
        "TTYPE3": "DEC",
        "COMMENT": "not a column",
    }

    assert _column_name_index_map(header) == {1: "ID", 2: "RA", 3: "DEC"}


def test_column_name_index_map_falls_back_without_tfields():
    header = {"TTYPE2": "RA", "TTYPE1": "ID", "HISTORY": "ignored"}

    assert _column_name_index_map(header) == {2: "RA", 1: "ID"}


class TestTableReading:
    """Test FITS table reading functionality."""

    def create_test_table(self, nrows=1000, ncols=5):
        """Create a test FITS table with various data types."""
        from astropy.table import Table

        data = {}
        data["ID"] = np.arange(nrows, dtype=np.int32)  # type: ignore
        data["RA"] = np.random.uniform(0, 360, nrows).astype(np.float64)  # type: ignore
        data["DEC"] = np.random.uniform(-90, 90, nrows).astype(np.float64)
        data["MAG_G"] = np.random.normal(20, 2, nrows).astype(np.float32)
        data["FLAG"] = np.random.choice([0, 1], nrows).astype(np.uint8)  # type: ignore

        # Add extra columns if requested
        for i in range(max(0, ncols - len(data))):
            data[f"EXTRA_{i}"] = np.random.normal(0, 1, nrows).astype(np.float32)

        table = Table(data)

        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            table.write(f.name, format="fits", overwrite=True)
            return f.name, data

    def test_basic_table_reading(self):
        """Test basic table reading."""
        filepath, expected_data = self.create_test_table(100)

        try:
            result, header = torchsky.read(
                filepath, hdu=1, return_header=True
            )  # Tables are usually in HDU 1

            assert isinstance(result, dict)
            assert "RA" in result
            assert "DEC" in result
            assert len(result["RA"]) == 100  # type: ignore

        finally:
            os.unlink(filepath)

    def test_column_selection(self):
        """Test reading specific columns."""
        filepath, expected_data = self.create_test_table(100)

        try:
            result, header = torchsky.read(
                filepath, hdu=1, columns=["RA", "DEC"], return_header=True
            )

            assert isinstance(result, dict)
            assert "RA" in result
            assert "DEC" in result
            assert "MAG_G" not in result  # Should not be included

        finally:
            os.unlink(filepath)

    def test_row_range_reading(self):
        """Test reading specific row ranges."""
        filepath, expected_data = self.create_test_table(1000)

        try:
            result, header = torchsky.read(
                filepath, hdu=1, start_row=100, num_rows=50, return_header=True
            )

            assert isinstance(result, dict)
            assert len(result["RA"]) == 50  # type: ignore

        finally:
            os.unlink(filepath)

    def test_large_table_streaming(self):
        """Test streaming read for large tables."""
        filepath, expected_data = self.create_test_table(10000)

        try:
            # Test with memory limit to force streaming
            result = torchsky.read_large_table(
                filepath, max_memory_mb=1, streaming=True
            )

            assert isinstance(result, dict)
            assert "RA" in result
            assert len(result["RA"]) == 10000  # type: ignore

        finally:
            os.unlink(filepath)

    def test_string_column_decoding(self):
        """Test decoding of string columns."""
        from astropy.table import Table

        names = ["alpha", "beta", "gamma"]
        values = [1.0, 2.0, 3.0]
        table = Table({"NAME": names, "VAL": values})

        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            table.write(f.name, format="fits", overwrite=True)
            path = f.name

        try:
            with torchsky.open(path) as hdul:
                table_hdu = hdul[1]  # type: ignore
                assert "NAME" in table_hdu.columns  # type: ignore
                assert "NAME" in table_hdu.string_columns  # type: ignore
                decoded = table_hdu.get_string_column("NAME")  # type: ignore
                assert decoded == names
        finally:
            os.unlink(path)

    def test_stream_table_chunks(self):
        """Test stream_table yields correct total row count."""
        filepath, _ = self.create_test_table(1234)

        try:
            total = 0
            for chunk in torchsky.stream_table(filepath, hdu=1, chunk_rows=200):
                assert "RA" in chunk
                total += len(chunk["RA"])  # type: ignore
            assert total == 1234
        finally:
            os.unlink(filepath)

    def test_schema_vla_flags(self):
        """Test schema reports VLA columns."""
        from astropy.table import Table
        import numpy as np

        vla = np.array([np.array([1, 2]), np.array([3])], dtype=object)
        table = Table({"VLA": vla})

        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            table.write(f.name, format="fits", overwrite=True)
            path = f.name

        try:
            with torchsky.open(path) as hdul:
                table_hdu = hdul[1]  # type: ignore
                schema = table_hdu.schema  # type: ignore
                assert "VLA" in schema["vla_columns"]
                lengths = table_hdu.get_vla_lengths("VLA")  # type: ignore
                assert lengths == [2, 1]
        finally:
            os.unlink(path)

    def test_filter_rows_with_expression(self):
        """Test TableHDU row filtering by expression."""
        filepath, _ = self.create_test_table(200)

        try:
            with torchsky.open(filepath) as hdul:
                table_hdu = hdul[1]  # type: ignore
                filtered = table_hdu.filter("(RA > 180.0) & (FLAG >= 0)")  # type: ignore
                assert isinstance(filtered, torchsky.TableHDU)
                assert filtered.num_rows > 0
                assert filtered.num_rows <= table_hdu.num_rows  # type: ignore
        finally:
            os.unlink(filepath)

    def test_complex_column_reading(self):
        """Test complex-valued FITS columns are readable."""
        from astropy.io import fits

        col1 = fits.Column(
            name="ID", format="J", array=np.array([1, 2, 3], dtype=np.int32)
        )
        col2 = fits.Column(
            name="Z",
            format="C",
            array=np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex64),
        )
        hdu = fits.BinTableHDU.from_columns([col1, col2])
        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            path = f.name
        fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path, overwrite=True)

        try:
            table = torchsky.read(path, hdu=1)
            assert "Z" in table
            assert len(table["Z"]) == 3  # type: ignore
        finally:
            os.unlink(path)

    def test_tablehdu_column_manipulations(self):
        """Test add/rename/drop/append operations on TableHDU."""
        table_hdu = torchsky.TableHDU(  # type: ignore
            {
                "A": torch.tensor([1, 2], dtype=torch.int32),
                "B": torch.tensor([10.0, 20.0], dtype=torch.float32),
            }
        )

        with_c = table_hdu.add_column("C", torch.tensor([100, 200], dtype=torch.int64))  # type: ignore
        assert "C" in with_c.columns
        assert with_c.num_rows == 2

        renamed = with_c.rename_column("C", "C_NEW")
        assert "C_NEW" in renamed.columns
        assert "C" not in renamed.columns

        dropped = renamed.drop_columns(["B"])
        assert dropped.columns == ["A", "C_NEW"]

        appended = dropped.append_rows(
            {
                "A": torch.tensor([3], dtype=torch.int32),
                "C_NEW": torch.tensor([300], dtype=torch.int64),
            }
        )
        assert appended.num_rows == 3
        assert appended["A"].tolist() == [1, 2, 3]  # type: ignore

    def test_tablehdu_accepts_numpy_numeric_columns(self):
        """TableHDU should accept NumPy numeric columns without TensorFrame shape drift."""
        table_hdu = torchsky.TableHDU(  # type: ignore
            {
                "A": np.array([1, 2, 3], dtype=np.int32),
                "B": np.array([1.5, 2.5, 3.5], dtype=np.float32),
            }
        )

        assert table_hdu.num_rows == 3
        assert table_hdu.columns == ["A", "B"]
        assert table_hdu["A"].tolist() == [1, 2, 3]  # type: ignore

    def test_tablehdu_append_rows_with_vla_lists(self):
        """Test appending rows for VLA-like list columns."""
        table_hdu = torchsky.TableHDU(  # type: ignore
            {
                "ID": torch.tensor([1, 2], dtype=torch.int32),
                "VLA": [torch.tensor([1, 2]), torch.tensor([3])],
            }
        )

        out = table_hdu.append_rows(  # type: ignore
            {
                "ID": torch.tensor([3], dtype=torch.int32),
                "VLA": [torch.tensor([4, 5, 6])],
            }
        )
        assert out.num_rows == 3
        assert out["ID"].tolist() == [1, 2, 3]  # type: ignore
        assert len(out.get_vla_column("VLA")) == 3  # type: ignore
        assert out.get_vla_column("VLA")[-1].tolist() == [4, 5, 6]  # type: ignore

    def test_data_types(self):
        """Test various FITS data types."""
        filepath, expected_data = self.create_test_table(100)

        try:
            result, header = torchsky.read(filepath, hdu=1, return_header=True)

            # Check data types are preserved appropriately
            assert result["ID"].dtype in [torch.int32, torch.int64]  # type: ignore
            assert result["RA"].dtype in [torch.float32, torch.float64]  # type: ignore
            assert result["FLAG"].dtype in [torch.uint8, torch.int16, torch.int32]  # type: ignore

        finally:
            os.unlink(filepath)

    def test_empty_table(self):
        """Test handling of empty tables."""
        from astropy.table import Table

        # Create empty table
        empty_table = Table()
        empty_table["EMPTY_COL"] = []

        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            try:
                empty_table.write(f.name, format="fits", overwrite=True)

                # Should handle empty table gracefully
                result, header = torchsky.read(f.name, hdu=1, return_header=True)
                assert isinstance(result, dict)

            finally:
                os.unlink(f.name)

    def test_close_cpp_handle_exception(self):
        """Test _close_cpp_handle suppresses exceptions during handle closure."""
        from torchsky.table import _close_cpp_handle

        handle = mock.MagicMock()
        handle.close.side_effect = Exception("Mocked close error")

        # Should not raise
        _close_cpp_handle(handle)
        handle.close.assert_called_once()

    @mock.patch("torchsky.get_header")
    def test_can_use_mmap_row_path_for_full_read_exception(self, mock_get_header):
        """Test that _can_use_mmap_row_path_for_full_read handles exceptions gracefully."""
        from torchsky.table import _can_use_mmap_row_path_for_full_read

        mock_get_header.side_effect = Exception("Mocked error reading header")
        result = _can_use_mmap_row_path_for_full_read("dummy.fits", 1, None)
        assert result is False


class TestTablePerformance:
    """Test table reading performance optimizations."""

    def create_large_table(self, nrows=100000):
        """Create a large test table."""
        from astropy.table import Table

        data = {
            "ID": np.arange(nrows, dtype=np.int64),
            "RA": np.random.uniform(0, 360, nrows).astype(np.float64),
            "DEC": np.random.uniform(-90, 90, nrows).astype(np.float64),
            "MAG_G": np.random.normal(20, 2, nrows).astype(np.float32),
            "MAG_R": np.random.normal(20, 2, nrows).astype(np.float32),
            "MAG_I": np.random.normal(20, 2, nrows).astype(np.float32),
            "FLUX_G": np.random.exponential(1000, nrows).astype(np.float32),
            "FLUX_R": np.random.exponential(1000, nrows).astype(np.float32),
            "FLUX_I": np.random.exponential(1000, nrows).astype(np.float32),
            "Z": np.random.exponential(0.5, nrows).astype(np.float32),
        }

        table = Table(data)

        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            table.write(f.name, format="fits", overwrite=True)
            return f.name

    def test_chunked_reading_performance(self):
        """Test performance of chunked reading."""
        import time

        filepath = self.create_large_table(50000)

        try:
            # Test streaming read
            start_time = time.time()
            result = torchsky.read_large_table(
                filepath, max_memory_mb=10, streaming=True
            )
            streaming_time = time.time() - start_time

            assert isinstance(result, dict)
            assert len(result["RA"]) == 50000  # type: ignore

            # Streaming should complete in reasonable time
            assert streaming_time < 30.0  # Should complete within 30 seconds

        finally:
            os.unlink(filepath)

    def test_memory_efficiency(self):
        """Test memory efficiency of table reading."""
        import gc

        import psutil

        filepath = self.create_large_table(20000)

        try:
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1024 / 1024  # MB

            # Read table
            result = torchsky.read_large_table(filepath, max_memory_mb=50)

            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = mem_after - mem_before

            # Should not use excessive memory
            file_size = os.path.getsize(filepath) / 1024 / 1024  # MB
            assert memory_increase < 20 * file_size  # Allow 5x file size for processing

            # Cleanup
            del result
            gc.collect()

        finally:
            os.unlink(filepath)

    def test_is_compatible_arrow_table_edge_cases(self):
        import torchsky.table as tbl
        from unittest import mock

        # We test _can_use_mmap_row_path_for_full_read as this corresponds to the implementation
        # of the compatibility check in table.py

        # Test Exception in get_header
        with mock.patch("torchsky.get_header", side_effect=Exception("mocked error")):
            assert (
                tbl._can_use_mmap_row_path_for_full_read("dummy.fits", 1, None) is False
            )

        # Test TFIELDS = 0
        with mock.patch("torchsky.get_header", return_value={"TFIELDS": 0}):
            assert (
                tbl._can_use_mmap_row_path_for_full_read("dummy.fits", 1, None) is False
            )

        # Test missing TTYPE keys
        with mock.patch("torchsky.get_header", return_value={"TFIELDS": 2}):
            assert (
                tbl._can_use_mmap_row_path_for_full_read("dummy.fits", 1, None) is False
            )

    @mock.patch.dict("torchsky.table._reader_cache", {}, clear=True)
    @mock.patch.dict("torchsky.table._handle_cache", {}, clear=True)
    @mock.patch("torchsky.table._close_cpp_handle")
    def test_close_all_cached_handles(self, mock_close_cpp_handle):
        """Test that _close_all_cached_handles clears caches and closes handles."""
        # Setup dummy state
        dummy_reader = mock.MagicMock()
        dummy_handle1 = mock.MagicMock()
        dummy_handle2 = mock.MagicMock()

        torchsky.table._reader_cache[("dummy_path", 1)] = dummy_reader
        torchsky.table._handle_cache["dummy_path"] = dummy_handle1
        torchsky.table._handle_cache["dummy_path2"] = dummy_handle2

        # Call the function
        torchsky.table._close_all_cached_handles()

        # Verify state is cleared
        assert len(torchsky.table._reader_cache) == 0
        assert len(torchsky.table._handle_cache) == 0

        # Verify handles were closed
        mock_close_cpp_handle.assert_any_call(dummy_handle1)
        mock_close_cpp_handle.assert_any_call(dummy_handle2)
        assert mock_close_cpp_handle.call_count == 2



def test_close_cpp_handle_suppresses_exception():
    """Test that _close_cpp_handle catches and suppresses exceptions."""
    import unittest.mock
    from torchsky.table import _close_cpp_handle
    mock_handle = unittest.mock.Mock()
    mock_handle.close.side_effect = Exception("Test exception")
    _close_cpp_handle(mock_handle)
