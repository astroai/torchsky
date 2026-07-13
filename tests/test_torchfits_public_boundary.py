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


def test_table_boundary_is_a_thin_public_torchfits_facade():
    import torchfits
    from torchfits import table as fits_table
    from torchsky import table as sky_table

    major, minor, *_ = (int(part) for part in torchfits.__version__.split("."))
    assert (major, minor) >= (0, 8)
    for name in ("read", "scan", "scan_torch", "to_polars", "to_polars_lazy"):
        assert getattr(sky_table, name) is getattr(fits_table, name)
    assert not any(name.startswith("_can_use_") for name in dir(sky_table))


def test_hdu_boundary_does_not_reexport_tensor_frame():
    import torchsky.hdu as hdu

    assert not hasattr(hdu, "TensorFrame")
