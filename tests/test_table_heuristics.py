import unittest.mock
from torchsky.table import (
    _can_use_mmap_row_path_for_full_read,
    _can_use_torch_table_path_for_full_read,
)


def test_mmap_row_path_heuristic_error_handling():
    """Verify that _can_use_mmap_row_path_for_full_read returns False on error."""
    with unittest.mock.patch(
        "torchsky.get_header", side_effect=Exception("Failed to get header")
    ):
        assert _can_use_mmap_row_path_for_full_read("dummy.fits", 1, None) is False


def test_torch_table_path_heuristic_error_handling():
    """Verify that _can_use_torch_table_path_for_full_read returns False on error."""
    with unittest.mock.patch(
        "torchsky.get_header", side_effect=Exception("Failed to get header")
    ):
        assert _can_use_torch_table_path_for_full_read("dummy.fits", 1, None) is False
