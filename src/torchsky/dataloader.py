"""FITS dataloaders — thin helpers over PyTorch DataLoader + torchfits datasets."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import (
    DataLoader,
    DistributedSampler,
    IterableDataset,
    RandomSampler,
)

from torchfits.data import FitsImageDataset, FitsImageIterableDataset

# torchsky compatibility aliases; torchfits 0.8 exposes only the explicit
# ``Fits*Dataset`` names.
FITSDataset = FitsImageDataset
IterableFITSDataset = FitsImageIterableDataset


def create_dataloader(
    dataset: Any,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    prefetch_factor: int | None = None,
    drop_last: bool = False,
    persistent_workers: bool = False,
    **loader_kwargs: Any,
) -> DataLoader:
    """Create a DataLoader with sensible defaults for FITS workflows.

    If *dataset* is a list of file paths, a :class:`FitsImageDataset` is created
    internally.  When *dataset* is an :class:`IterableDataset`, ``shuffle`` is
    silently dropped (PyTorch does not support shuffling iterable-style
    datasets via a sampler).

    *pin_memory* is forced to ``False`` when no CUDA device is available.
    """
    # Accept raw file lists
    if isinstance(dataset, (list, tuple)):
        dataset = FitsImageDataset(list(dataset))

    # Iterable datasets cannot use a sampler.
    if isinstance(dataset, IterableDataset):
        kwargs: dict[str, Any] = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory and torch.cuda.is_available(),
            "drop_last": drop_last,
            "persistent_workers": persistent_workers,
        }
        if prefetch_factor is not None and num_workers > 0:
            kwargs["prefetch_factor"] = prefetch_factor
        kwargs.update(loader_kwargs)
        return DataLoader(dataset, **kwargs)

    # Map-style dataset
    sampler = RandomSampler(dataset) if shuffle else None
    kwargs = {
        "batch_size": batch_size,
        "shuffle": False,  # mutually exclusive with sampler
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": pin_memory and torch.cuda.is_available(),
        "drop_last": drop_last,
        "persistent_workers": persistent_workers,
    }
    if prefetch_factor is not None and num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    kwargs.update(loader_kwargs)
    return DataLoader(dataset, **kwargs)


def create_fits_dataloader(
    file_paths: list[str],
    hdu: int = 0,
    batch_size: int = 1,
    transform: Any = None,
    device: str = "cpu",
    mmap: str = "auto",
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    **loader_kwargs: Any,
) -> DataLoader:
    """Convenience dataloader for map-style FITS datasets.

    Creates a :class:`FitsImageDataset` with the given parameters and wraps it in a
    :class:`DataLoader`.  Extra keyword arguments are forwarded to
    :func:`create_dataloader`.
    """
    dataset = FitsImageDataset(
        paths=file_paths,
        hdu=hdu,
        transform=transform,
        device=device,
        mmap=mmap != "auto",
    )
    return create_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **loader_kwargs,
    )


def create_streaming_dataloader(
    file_paths: str | list[str] | None = None,
    hdu: int = 0,
    batch_size: int = 1,
    num_workers: int = 0,
    pin_memory: bool = False,
    *,
    index_url: str | None = None,
    **loader_kwargs: Any,
) -> DataLoader:
    """Streaming dataloader for iterable FITS datasets.

    Creates a :class:`FitsImageIterableDataset` over *file_paths* and wraps
    it in a :class:`DataLoader`.
    """
    if file_paths is None:
        file_paths = index_url
    if file_paths is None:
        raise TypeError("file_paths is required")
    paths = [file_paths] if isinstance(file_paths, str) else list(file_paths)
    dataset = FitsImageIterableDataset(paths=paths, hdu=hdu)
    return create_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **loader_kwargs,
    )


def create_table_dataloader(
    file_paths: list[str],
    hdu: int = 1,
    chunk_rows: int = 1000,
    columns: list[str] | None = None,
    max_chunks: int | None = None,
    mmap: bool = False,
    device: str = "cpu",
    non_blocking_transfer: bool = True,
    pin_memory_transfer: bool = False,
    transform: Any = None,
    include_header: bool = False,
    batch_size: int = 1,
    num_workers: int = 0,
    pin_memory: bool = False,
    **loader_kwargs: Any,
) -> DataLoader:
    """Dataloader for table data chunked by rows.

    Creates a :class:`TableChunkDataset` and wraps it in a :class:`DataLoader`.
    """
    from torchsky.datasets import TableChunkDataset

    dataset = TableChunkDataset(
        file_paths=file_paths,
        hdu=hdu,
        columns=columns,
        chunk_rows=chunk_rows,
        max_chunks=max_chunks,
        mmap=mmap,
        device=device,
        non_blocking_transfer=non_blocking_transfer,
        pin_memory_transfer=pin_memory_transfer,
        transform=transform,
        include_header=include_header,
    )
    return create_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **loader_kwargs,
    )


def create_distributed_dataloader(
    file_paths: list[str],
    batch_size: int = 1,
    num_replicas: int = 1,
    rank: int = 0,
    shuffle: bool = True,
    seed: int = 0,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    **loader_kwargs: Any,
) -> DataLoader:
    """Create a distributed DataLoader with a :class:`DistributedSampler`.

    Wraps a :class:`FitsImageDataset` over *file_paths*.
    """
    dataset = FitsImageDataset(list(file_paths))
    sampler = DistributedSampler(
        dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed
    )
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": pin_memory and torch.cuda.is_available(),
        "drop_last": drop_last,
    }
    kwargs.update(loader_kwargs)
    return DataLoader(dataset, **kwargs)


def create_ml_dataloader(
    file_paths: list[str],
    batch_size: int = 1,
    validation_split: float = 0.2,
    test_split: float = 0.1,
    random_seed: int = 42,
    transforms: dict[str, Any] | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    **loader_kwargs: Any,
) -> dict[str, DataLoader]:
    """Create train / validation / test DataLoaders from a file list.

    Returns a dict with keys ``"train"``, ``"val"`` (if *validation_split*
    > 0), and ``"test"`` (if *test_split* > 0).
    """
    import random

    rng = random.Random(random_seed)
    paths = list(file_paths)
    rng.shuffle(paths)

    n = len(paths)
    n_test = int(n * test_split)
    n_val = int(n * validation_split)
    n_train = n - n_val - n_test

    splits: dict[str, list[str]] = {}
    if n_train > 0:
        splits["train"] = paths[:n_train]
    idx = n_train
    if n_val > 0:
        splits["val"] = paths[idx : idx + n_val]
        idx += n_val
    if n_test > 0:
        splits["test"] = paths[idx : idx + n_test]

    dataloaders: dict[str, DataLoader] = {}
    for split_name, split_paths in splits.items():
        transform = transforms.get(split_name) if transforms else None
        dataloaders[split_name] = create_fits_dataloader(
            file_paths=split_paths,
            batch_size=batch_size,
            transform=transform,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
            **loader_kwargs,
        )
    return dataloaders


def get_optimal_dataloader_config(
    dataset_size: int,
    file_size_mb: float = 10.0,
    available_memory_gb: float = 8.0,
    num_gpus: int = 0,
) -> dict[str, Any]:
    """Heuristic DataLoader configuration for FITS workloads.

    Returns a dict with keys ``batch_size``, ``num_workers``, ``pin_memory``,
    ``prefetch_factor``, and ``persistent_workers``.
    """
    # Batch size: aim for ~64-512 MB per batch
    target_batch_mb = 128.0
    batch_size = max(1, int(target_batch_mb / max(file_size_mb, 0.1)))
    batch_size = min(batch_size, 256)

    # Scale down for very small datasets
    if dataset_size < 100:
        batch_size = min(batch_size, 4)
    elif dataset_size < 1000:
        batch_size = min(batch_size, 16)

    # Workers: approx CPU cores, capped by memory
    cpu_cores = max(1, _os_cpu_count() // 2)
    mem_mb = available_memory_gb * 1024
    max_workers_by_mem = max(1, int(mem_mb / (file_size_mb * batch_size * 2)))
    # Small datasets: stay conservative on workers
    if dataset_size < 1000:
        num_workers = min(cpu_cores, max_workers_by_mem, 2)
    else:
        num_workers = min(cpu_cores, max_workers_by_mem, 8 if num_gpus > 0 else 4)

    pin_memory = num_gpus > 0
    prefetch_factor = 2 if num_workers > 0 else None
    persistent_workers = num_workers > 0

    return {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "prefetch_factor": prefetch_factor,
        "persistent_workers": persistent_workers,
    }


def _os_cpu_count() -> int:
    """Return the number of available CPUs, defaulting to 4 on error."""
    try:
        import os as _os

        return _os.cpu_count() or 4
    except Exception:
        return 4
