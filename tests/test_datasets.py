import pytest
import torch
from unittest.mock import patch

from torchsky.datasets import TableChunkDataset

class TestTableChunkDataset:
    """Tests for TableChunkDataset using mocked file operations to avoid C++ dependencies."""

    @patch('torchsky.table.scan_torch')
    def test_basic_iteration(self, mock_scan_torch):
        """Test basic iteration over chunks."""
        # Setup mock to return chunks
        mock_scan_torch.return_value = [
            {"ID": torch.arange(30), "RA": torch.rand(30)},
            {"ID": torch.arange(30, 60), "RA": torch.rand(30)},
            {"ID": torch.arange(60, 90), "RA": torch.rand(30)},
            {"ID": torch.arange(90, 100), "RA": torch.rand(10)},
        ]

        dataset = TableChunkDataset(
            file_paths=["dummy1.fits"],
            hdu=1,
            chunk_rows=30
        )

        chunks = list(dataset)

        assert len(chunks) == 4
        assert len(chunks[0]["ID"]) == 30
        assert len(chunks[-1]["ID"]) == 10

        # Verify scan_torch was called correctly
        mock_scan_torch.assert_called_once_with(
            "dummy1.fits",
            hdu=1,
            columns=None,
            batch_size=30,
            mmap=False,
            device="cpu",
            non_blocking=True,
            pin_memory=False
        )

    @patch('torchsky.table.scan_torch')
    @patch('torchsky.get_header')
    def test_include_header(self, mock_get_header, mock_scan_torch):
        """Test iteration when include_header is True."""
        mock_scan_torch.return_value = [{"ID": torch.arange(50)}, {"ID": torch.arange(50, 100)}]
        mock_get_header.return_value = {"NAXIS1": 100}

        dataset = TableChunkDataset(
            file_paths=["dummy1.fits"],
            hdu=1,
            chunk_rows=50,
            include_header=True
        )

        chunks = list(dataset)

        assert len(chunks) == 2
        for chunk, header in chunks:
            assert isinstance(chunk, dict)
            assert isinstance(header, dict)
            assert "ID" in chunk
            assert header["NAXIS1"] == 100

    @patch('torchsky.table.scan_torch')
    def test_transform(self, mock_scan_torch):
        """Test that transform function is applied to chunks."""
        mock_scan_torch.return_value = [{"ID": torch.tensor([1, 2, 3])}]

        def double_id(chunk):
            chunk["ID"] = chunk["ID"] * 2
            return chunk

        dataset = TableChunkDataset(
            file_paths=["dummy1.fits"],
            hdu=1,
            chunk_rows=100,
            transform=double_id
        )

        chunks = list(dataset)
        assert len(chunks) == 1

        # Check that transform was applied
        assert chunks[0]["ID"][1].item() == 4  # 2 * 2

    @patch('torchsky.table.scan_torch')
    def test_max_chunks(self, mock_scan_torch):
        """Test limiting the number of chunks yielded via max_chunks."""
        mock_scan_torch.return_value = [
            {"ID": torch.arange(20)},
            {"ID": torch.arange(20, 40)},
            {"ID": torch.arange(40, 60)},
        ]

        dataset = TableChunkDataset(
            file_paths=["dummy1.fits"],
            hdu=1,
            chunk_rows=20,
            max_chunks=2
        )

        chunks = list(dataset)

        # Would normally be 3 chunks, but capped at 2
        assert len(chunks) == 2
        assert len(chunks[0]["ID"]) == 20
        assert len(chunks[1]["ID"]) == 20

    @patch('torchsky.table.scan_torch')
    def test_columns(self, mock_scan_torch):
        """Test specifying specific columns."""
        mock_scan_torch.return_value = [{"ID": torch.arange(100), "RA": torch.rand(100)}]

        dataset = TableChunkDataset(
            file_paths=["dummy1.fits"],
            hdu=1,
            chunk_rows=100,
            columns=["ID", "RA"]
        )

        chunks = list(dataset)
        assert len(chunks) == 1

        mock_scan_torch.assert_called_once_with(
            "dummy1.fits",
            hdu=1,
            columns=["ID", "RA"],
            batch_size=100,
            mmap=False,
            device="cpu",
            non_blocking=True,
            pin_memory=False
        )

    @patch('torchsky.table.scan_torch')
    def test_multiple_files(self, mock_scan_torch):
        """Test iterating over multiple files."""
        # mock returns 2 chunks for each file it's called with
        mock_scan_torch.side_effect = [
            [{"ID": torch.arange(60)}, {"ID": torch.arange(60, 100)}], # file 1
            [{"ID": torch.arange(60)}, {"ID": torch.arange(60, 100)}]  # file 2
        ]

        dataset = TableChunkDataset(
            file_paths=["dummy1.fits", "dummy2.fits"],
            hdu=1,
            chunk_rows=60
        )

        chunks = list(dataset)

        # 100 rows per file -> chunks of 60, 40 from first file, then 60, 40 from second
        assert len(chunks) == 4
        assert len(chunks[0]["ID"]) == 60
        assert len(chunks[1]["ID"]) == 40
        assert len(chunks[2]["ID"]) == 60
        assert len(chunks[3]["ID"]) == 40

        assert mock_scan_torch.call_count == 2
        mock_scan_torch.assert_any_call(
            "dummy1.fits", hdu=1, columns=None, batch_size=60, mmap=False, device="cpu", non_blocking=True, pin_memory=False
        )
        mock_scan_torch.assert_any_call(
            "dummy2.fits", hdu=1, columns=None, batch_size=60, mmap=False, device="cpu", non_blocking=True, pin_memory=False
        )

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
