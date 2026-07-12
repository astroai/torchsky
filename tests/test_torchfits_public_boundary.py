from pathlib import Path


def test_torchsky_does_not_import_torchfits_private_modules():
    source_root = Path(__file__).parents[1] / "src" / "torchsky"
    offenders = []
    for path in source_root.rglob("*.py"):
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if "torchfits._" in line:
                offenders.append(f"{path}:{line_number}: {line.strip()}")
    assert not offenders, "\n".join(offenders)


def test_dataset_boundary_uses_torchfits_08_names():
    from torchfits.data import FitsImageDataset, FitsImageIterableDataset
    from torchsky.datasets import (
        FITSDataset,
        FitsImageDataset as TorchskyFitsImageDataset,
        FitsImageIterableDataset as TorchskyFitsImageIterableDataset,
        IterableFITSDataset,
    )

    assert TorchskyFitsImageDataset is FitsImageDataset
    assert TorchskyFitsImageIterableDataset is FitsImageIterableDataset
    assert FITSDataset is FitsImageDataset
    assert IterableFITSDataset is FitsImageIterableDataset


def test_table_boundary_uses_torchfits_08_public_helpers():
    from torchfits import table as fits_table

    for name in (
        "TABLE_BACKENDS",
        "clear_cache",
        "can_use_mmap_row_path_for_full_read",
        "can_use_torch_table_path_for_full_read",
        "column_name_index_map",
    ):
        assert hasattr(fits_table, name), name
