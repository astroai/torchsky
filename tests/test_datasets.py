from unittest.mock import MagicMock, patch

import pytest
import torch

from torchsky.datasets import FITSDataset, IterableFITSDataset, TableChunkDataset

# Tests will be implemented here


class TestFITSDataset:
    def test_initialization(self):
        file_paths = ["file1.fits", "file2.fits"]
        dataset = FITSDataset(file_paths, hdu=1, include_header=True)
        assert dataset.file_paths == file_paths
        assert dataset.hdu == 1
        assert dataset.include_header is True
        assert len(dataset) == 2

        manifest = dataset.manifest
        assert len(manifest) == 2
        assert manifest[0] == {"file_path": "file1.fits", "hdu": 1, "sample_id": 0}
        assert manifest[1] == {"file_path": "file2.fits", "hdu": 1, "sample_id": 1}

    @patch("torchsky.io.read")
    def test_getitem_without_header(self, mock_read):
        mock_data = torch.tensor([1, 2, 3])
        mock_read.return_value = mock_data

        dataset = FITSDataset(["file1.fits"], hdu="auto", include_header=False)
        item = dataset[0]

        assert torch.equal(item, mock_data)
        mock_read.assert_called_once_with(
            "file1.fits",
            hdu="auto",
            device="cpu",
            mmap="auto",
            cache_capacity=0,
            handle_cache_capacity=64,
            scale_on_device=True,
            raw_scale=False,
            return_header=False,
        )

    @patch("torchsky.io.read")
    def test_getitem_with_header(self, mock_read):
        mock_data = torch.tensor([1, 2, 3])
        mock_header = {"NAXIS": 1}
        mock_read.return_value = (mock_data, mock_header)

        dataset = FITSDataset(["file1.fits"], include_header=True)
        item, header = dataset[0]

        assert torch.equal(item, mock_data)
        assert header == mock_header
        mock_read.assert_called_once_with(
            "file1.fits",
            hdu="auto",
            device="cpu",
            mmap="auto",
            cache_capacity=0,
            handle_cache_capacity=64,
            scale_on_device=True,
            raw_scale=False,
            return_header=True,
        )

    @patch("torchsky.io.read")
    def test_getitem_with_transform(self, mock_read):
        mock_data = torch.tensor([1, 2, 3])
        mock_read.return_value = mock_data

        def mock_transform(x):
            return x * 2

        dataset = FITSDataset(["file1.fits"], transform=mock_transform)
        item = dataset[0]

        assert torch.equal(item, torch.tensor([2, 4, 6]))

    def test_out_of_bounds(self):
        dataset = FITSDataset(["file1.fits"])
        with pytest.raises(IndexError, match="out of range"):
            _ = dataset[1]
        with pytest.raises(IndexError, match="out of range"):
            _ = dataset[-1]


class TestIterableFITSDataset:
    def test_initialization(self):
        dataset = IterableFITSDataset("http://index.url", hdu=2, shard_size=500)
        assert dataset.index_url == "http://index.url"
        assert dataset.hdu == 2
        assert dataset.shard_size == 500

    @patch("torch.utils.data.get_worker_info")
    @patch("torchsky.io.read")
    def test_iteration_single_worker(self, mock_read, mock_worker_info, tmp_path):
        mock_worker_info.return_value = None  # Single process
        mock_data = torch.tensor([1, 2, 3])
        mock_read.return_value = mock_data
        index = tmp_path / "index.txt"
        index.write_text(
            "\n".join(["file0.fits", "file1.fits", "file2.fits", "file3.fits"]),
            encoding="utf-8",
        )

        dataset = IterableFITSDataset(str(index), shard_size=2)

        # Test just the first few items to avoid infinite loops if shards are unbounded
        items = []
        for i, item in enumerate(dataset):
            items.append(item)
            if i >= 3:  # Only need to check a few
                break

        assert len(items) == 4
        assert all(torch.equal(item, mock_data) for item in items)

    @patch("torch.utils.data.get_worker_info")
    def test_worker_sharding(self, mock_worker_info, tmp_path):
        # Mock a worker info object
        mock_worker = MagicMock()
        mock_worker.id = 1
        mock_worker.num_workers = 4
        mock_worker_info.return_value = mock_worker
        index = tmp_path / "index.json"
        index.write_text(
            '{"shards": [["f0"], ["f1"], ["f2"], ["f3"], ["f4"], ["f5"], ["f6"], ["f7"]]}',
            encoding="utf-8",
        )

        dataset = IterableFITSDataset(str(index))
        shards = dataset._get_assigned_shards(1, 4)

        assert shards == [2, 3]

        # Last worker
        shards_last = dataset._get_assigned_shards(3, 4)
        assert shards_last == [6, 7]

    @patch("torch.utils.data.get_worker_info")
    @patch("torchsky.io.read")
    def test_iteration_with_transform(self, mock_read, mock_worker_info, tmp_path):
        mock_worker_info.return_value = None
        mock_data = torch.tensor([1, 2, 3])
        mock_read.return_value = mock_data
        index = tmp_path / "index.json"
        index.write_text('{"files": ["file0.fits", "file1.fits"]}', encoding="utf-8")

        def mock_transform(x):
            return x + 1

        dataset = IterableFITSDataset(str(index), transform=mock_transform)

        items = []
        for i, item in enumerate(dataset):
            items.append(item)
            if i >= 1:
                break

        assert len(items) == 2
        assert all(torch.equal(item, torch.tensor([2, 3, 4])) for item in items)

    @patch("torch.utils.data.get_worker_info")
    @patch("torchsky.io.read")
    def test_error_handling(self, mock_read, mock_worker_info, tmp_path):
        mock_worker_info.return_value = None
        index = tmp_path / "index.txt"
        index.write_text("bad.fits\ngood.fits\n", encoding="utf-8")

        # Make read fail on the first call, succeed on the second
        mock_data = torch.tensor([1, 2, 3])
        mock_read.side_effect = [IOError("Simulated read error"), mock_data]

        dataset = IterableFITSDataset(str(index), shard_size=2)

        # Should gracefully handle the error and continue to the second item
        items = []
        for i, item in enumerate(dataset):
            items.append(item)
            if i >= 0:  # Get just 1 item since the first failed
                break

        assert len(items) == 1
        assert torch.equal(items[0], mock_data)


class TestTableChunkDataset:
    def test_initialization(self):
        file_paths = ["file1.fits", "file2.fits"]
        columns = ["col1", "col2"]
        dataset = TableChunkDataset(
            file_paths,
            hdu=2,
            columns=columns,
            chunk_rows=5000,
            max_chunks=10,
            include_header=True,
        )
        assert dataset.file_paths == file_paths
        assert dataset.hdu == 2
        assert dataset.columns == columns
        assert dataset.chunk_rows == 5000
        assert dataset.max_chunks == 10
        assert dataset.include_header is True

    @patch("torchsky.table.scan_torch")
    def test_iteration_no_header(self, mock_scan):
        mock_chunk = {"col1": torch.tensor([1, 2]), "col2": torch.tensor([3, 4])}
        mock_scan.return_value = [
            mock_chunk,
            mock_chunk,
        ]  # Simulate yielding 2 chunks per file

        dataset = TableChunkDataset(["file1.fits"], include_header=False)
        items = list(dataset)

        assert len(items) == 2
        assert items[0] == mock_chunk
        mock_scan.assert_called_once_with(
            "file1.fits",
            hdu=1,
            columns=None,
            batch_size=10000,
            mmap=False,
            device="cpu",
            non_blocking=True,
            pin_memory=False,
        )

    @patch("torchsky.io.get_header")
    @patch("torchsky.table.scan_torch")
    def test_iteration_with_header(self, mock_scan, mock_get_header):
        mock_chunk = {"col1": torch.tensor([1, 2])}
        mock_scan.return_value = [mock_chunk]
        mock_header = {"KEY": "VALUE"}
        mock_get_header.return_value = mock_header

        dataset = TableChunkDataset(["file1.fits"], include_header=True)
        items = list(dataset)

        assert len(items) == 1
        assert items[0] == (mock_chunk, mock_header)
        mock_get_header.assert_called_once_with("file1.fits", 1)

    @patch("torchsky.table.scan_torch")
    def test_iteration_with_transform(self, mock_scan):
        mock_chunk = torch.tensor([1, 2, 3])
        mock_scan.return_value = [mock_chunk]

        def mock_transform(x):
            return x * 10

        dataset = TableChunkDataset(["file1.fits"], transform=mock_transform)
        items = list(dataset)

        assert len(items) == 1
        assert torch.equal(items[0], torch.tensor([10, 20, 30]))

    @patch("torchsky.table.scan_torch")
    def test_max_chunks(self, mock_scan):
        mock_chunk = torch.tensor([1, 2, 3])

        # Infinite generator for mock_scan
        def infinite_chunks(*args, **kwargs):
            while True:
                yield mock_chunk

        mock_scan.side_effect = infinite_chunks

        dataset = TableChunkDataset(["file1.fits", "file2.fits"], max_chunks=3)
        items = list(dataset)

        assert len(items) == 3
