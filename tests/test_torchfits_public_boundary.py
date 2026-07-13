from pathlib import Path


def test_torchsky_does_not_import_torchfits_private_modules():
    source_root = Path(__file__).parents[1] / "src" / "torchsky"
    offenders = []
    for path in source_root.rglob("*.py"):
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if "torchfits._" in line:
                offenders.append(f"{path}:{line_number}: {line.strip()}")
    assert not offenders, "\n".join(offenders)


def test_dataset_boundary_uses_published_torchfits_names():
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


def test_table_boundary_keeps_compatibility_helpers_for_published_torchfits():
    import torchfits
    from torchfits import table as fits_table
    from torchsky import table as sky_table

    major, minor, *_ = (int(part) for part in torchfits.__version__.split("."))
    assert (major, minor) >= (0, 7)
    for name in ("TABLE_BACKENDS", "read", "scan", "scan_torch"):
        assert hasattr(fits_table, name), name
    for name in (
        "clear_cache",
        "_can_use_mmap_row_path_for_full_read",
        "_can_use_torch_table_path_for_full_read",
        "_column_name_index_map",
    ):
        assert hasattr(sky_table, name), name


def test_hdu_boundary_does_not_reexport_tensor_frame():
    import torchsky.hdu as hdu

    assert not hasattr(hdu, "TensorFrame")
