"""I/O tests — validation and integration, not torchsky.cpp dispatch internals."""
import pytest
import torch

import torchsky
from torchsky import io


class TestIORead:
    """Input validation and integration tests for torchsky.io.read."""

    def test_read_empty_path(self):
        with pytest.raises(ValueError, match="Path must be a non-empty string"):
            io.read("")

    def test_read_invalid_path_type(self):
        with pytest.raises(ValueError, match="Path must be a string or list of strings"):
            io.read(123)

    def test_read_batch_path_invalid_element_type(self):
        with pytest.raises(RuntimeError, match="Failed to read FITS file"):
            io.read(["file1.fits", 123], hdu=0)

    def test_read_batch_path_invalid_hdu(self):
        with pytest.raises(ValueError, match="Batch read requires a single integer HDU"):
            io.read(["file1.fits", "file2.fits"], hdu=[0, 1])

    def test_read_invalid_hdu_type(self):
        with pytest.raises(ValueError, match="HDU index must be non-negative"):
            io.read("file.fits", hdu=-1)

    def test_read_invalid_device(self):
        with pytest.raises(
            ValueError, match="device must be 'cpu', 'cuda', 'mps' or 'cuda:N'"
        ):
            io.read("file.fits", device="invalid_device")

    def test_full_io_helpers_do_not_delegate_to_root(self, tmp_path, monkeypatch):
        path = tmp_path / "io_helpers.fits"
        io.write(str(path), torch.arange(4, dtype=torch.float32).reshape(2, 2))

        def fail_root_helper(*args, **kwargs):
            raise AssertionError("torchsky.io helper delegated to the root facade")

        for name in (
            "insert_hdu",
            "replace_hdu",
            "delete_hdu",
            "write_checksums",
            "verify_checksums",
            "read_batch",
            "get_batch_info",
            "get_cache_performance",
            "clear_file_cache",
        ):
            monkeypatch.setattr(torchsky, name, fail_root_helper)

        io.insert_hdu(
            str(path),
            torch.ones((2, 2), dtype=torch.float32),
            index=1,
            header={"EXTNAME": "SCI"},
        )
        io.replace_hdu(str(path), "SCI", torch.full((2, 2), 2.0))
        torch.testing.assert_close(
            io.read(str(path), hdu=1, mmap=False), torch.full((2, 2), 2.0)
        )
        io.delete_hdu(str(path), 1)

        io.write_checksums(str(path), hdu=0)
        assert io.verify_checksums(str(path), hdu=0)["ok"] is True

        assert io.read_batch([]) == []
        assert io.get_batch_info([]) == {"num_files": 0, "valid_files": 0}
        assert "hit_rate" in io.get_cache_performance()
        io.clear_file_cache(cpp=False)
