from pathlib import Path
import re


def test_torchsky_src_does_not_import_astropy_io_fits():
    repo = Path(__file__).parents[1]
    paths = list((repo / "src" / "torchsky").rglob("*.py"))
    paths.extend((repo / "tests").glob("test_wcs*.py"))
    paths.extend(
        [
            repo / "tests" / "test_catalog_compat.py",
            repo / "tests" / "test_canfar_wcs_sphere_runner.py",
            repo / "tests" / "test_release_smoke.py",
            repo / "tests" / "test_real_data_validation.py",
            repo / "tests" / "test_spectral.py",
            repo / "benchmarks" / "bench_catalog_compat.py",
            repo / "benchmarks" / "bench_wcs.py",
            repo / "benchmarks" / "bench_wcs_tpv_inverse.py",
            repo / "benchmarks" / "bench_wcs_legacy_only.py",
            repo / "benchmarks" / "bench_wcs_spectral_temporal.py",
        ]
    )
    offenders = []
    for path in paths:
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "astropy.io.fits" in line:
                offenders.append(f"{path}:{line_number}: {stripped}")
    assert not offenders, "\n".join(offenders)


def test_torchsky_does_not_import_python_fitsio():
    repo = Path(__file__).parents[1]
    roots = [
        repo / "src" / "torchsky",
        repo / "tests",
        repo / "benchmarks",
        repo / "scripts",
        repo / "examples",
    ]
    offenders = []
    for root in roots:
        for path in root.rglob("*.py"):
            for line_number, line in enumerate(path.read_text().splitlines(), start=1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if stripped.startswith("import fitsio") or stripped.startswith(
                    "from fitsio"
                ):
                    offenders.append(f"{path}:{line_number}: {stripped}")
    assert not offenders, "\n".join(offenders)


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

    release = tuple(int(part) for part in re.findall(r"\d+", torchfits.__version__)[:3])
    assert release >= (1, 0, 0)
    for name in ("read", "scan", "scan_torch", "read_polars", "scan_polars"):
        assert getattr(sky_table, name) is getattr(fits_table, name)
    assert not any(name.startswith("_can_use_") for name in dir(sky_table))


def test_hdu_boundary_does_not_reexport_tensor_frame():
    import torchsky.hdu as hdu

    assert not hasattr(hdu, "TensorFrame")
