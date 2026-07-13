from unittest.mock import MagicMock, patch

import torch

from torchsky.datasets import (
    FitsImageDataset,
    FitsImageIterableDataset,
    TableChunkDataset,
)


class TestFitsImageDataset:
    def test_initialization(self):
        dataset = FitsImageDataset(["file1.fits", "file2.fits"], hdu=1, labels=[3, 4])
        assert dataset.files == ["file1.fits", "file2.fits"]
        assert dataset.hdu == 1
        assert len(dataset) == 2

    @patch("torchfits.read")
    def test_getitem_and_transform(self, mock_read):
        mock_read.return_value = torch.tensor([[1, 2]])
        dataset = FitsImageDataset(
            ["file1.fits"], labels=[7], transform=lambda image: image + 1
        )

        image, label = dataset[0]

        assert torch.equal(image, torch.tensor([[[2, 3]]]))
        assert torch.equal(label, torch.tensor(7))
        mock_read.assert_called_once_with(
            "file1.fits", hdu=0, mode="image", device="cpu", mmap=True
        )

    def test_out_of_bounds(self):
        dataset = FitsImageDataset(["file1.fits"])
        try:
            dataset[1]
        except IndexError:
            pass
        else:
            raise AssertionError("out-of-range dataset index did not fail")


class TestFitsImageIterableDataset:
    def test_initialization(self):
        dataset = FitsImageIterableDataset(["file1.fits"], hdu=2)
        assert dataset.files == ["file1.fits"]
        assert dataset.hdu == 2

    @patch("torchfits.read")
    def test_iteration_and_transform(self, mock_read):
        mock_read.return_value = torch.tensor([1, 2, 3])
        dataset = FitsImageIterableDataset(
            ["file0.fits", "file1.fits"], transform=lambda image: image + 1
        )

        items = list(dataset)

        assert len(items) == 2
        assert all(torch.equal(item, torch.tensor([2, 3, 4])) for item in items)
        assert mock_read.call_count == 2

    @patch("torch.utils.data.get_worker_info")
    def test_worker_sharding(self, mock_worker_info):
        worker = MagicMock(id=1, num_workers=2)
        mock_worker_info.return_value = worker
        dataset = FitsImageIterableDataset(["file0.fits", "file1.fits", "file2.fits"])

        with patch("torchfits.read", return_value=torch.ones(2)) as read:
            list(dataset)

        assert [call.args[0] for call in read.call_args_list] == ["file2.fits"]


class TestTableChunkDataset:
    @patch("torchsky.datasets._tf_scan_torch")
    def test_iteration_no_header(self, mock_scan):
        chunks = [{"col1": torch.tensor([1, 2])}, {"col1": torch.tensor([3])}]
        mock_scan.return_value = iter(chunks)
        dataset = TableChunkDataset(["file1.fits"], chunk_rows=2)

        assert list(dataset) == chunks
        mock_scan.assert_called_once_with(
            "file1.fits",
            hdu=1,
            columns=None,
            batch_size=2,
            mmap=False,
            device="cpu",
            non_blocking=True,
            pin_memory=False,
        )

    @patch("torchsky.datasets._tf_get_header", return_value={"KEY": "VALUE"})
    @patch("torchsky.datasets._tf_scan_torch")
    def test_iteration_with_header(self, mock_scan, mock_header):
        mock_scan.return_value = iter([{"col1": torch.tensor([1, 2])}])
        dataset = TableChunkDataset(["file1.fits"], include_header=True)

        items = list(dataset)

        assert len(items) == 1
        assert torch.equal(items[0]["col1"], torch.tensor([1, 2]))
        assert items[0]["_header"] == {"KEY": "VALUE"}
        mock_header.assert_called_once_with("file1.fits", 1)

    @patch("torchsky.datasets._tf_scan_torch")
    def test_transform_and_max_chunks(self, mock_scan):
        mock_scan.return_value = iter(
            [{"col1": torch.tensor([1])}, {"col1": torch.tensor([2])}]
        )
        dataset = TableChunkDataset(
            ["file1.fits"],
            max_chunks=1,
            transform=lambda chunk: {"x": chunk["col1"] * 10},
        )

        assert list(dataset) == [{"x": torch.tensor([10])}]
