"""
PyTorch Dataset classes for FITS data.

This module provides Dataset implementations for machine learning workflows:
- FITSDataset: Map-style dataset for random access
- IterableFITSDataset: Streaming dataset for large-scale data
"""

import json
from pathlib import Path
from typing import Any, Callable, Iterator, List, Optional, Union
from urllib.parse import unquote, urlparse
from urllib.request import urlopen

import torch
from torch.utils.data import Dataset, IterableDataset

# Import stream_table lazily in methods to avoid circular imports
# Import will be done locally to avoid circular imports


class FITSDataset(Dataset):
    """
    Map-style dataset for random access to a collection of FITS files.

    Upon initialization, builds a manifest of all possible samples for efficient
    randomization across a vast collection of files.
    """

    def __init__(
        self,
        file_paths: List[str],
        hdu: Union[int, str, None] = "auto",
        transform: Optional[Callable] = None,
        device: str = "cpu",
        include_header: bool = False,
        mmap: Union[bool, str] = "auto",
        cache_capacity: int = 0,
        handle_cache_capacity: int = 64,
        scale_on_device: bool = True,
        raw_scale: bool = False,
    ):
        """
        Initialize FITSDataset.

        Args:
            file_paths: List of FITS file paths
            hdu: HDU index/name or `"auto"` to detect first payload HDU
            transform: Optional transform to apply to each sample
            device: PyTorch device to load data onto
            mmap: Memory mapping mode for image reads (`True`, `False`, `'auto'`)
            cache_capacity: Python-side data cache entries for repeated reads
            handle_cache_capacity: Open-handle cache entries for repeated reads
            scale_on_device: Apply FITS scaling in the reader fast path
            raw_scale: Return raw stored values instead of physical scaled values
        """
        self.file_paths = file_paths
        self.hdu = hdu
        self.transform = transform
        self.device = device
        self.include_header = include_header
        self.mmap = mmap
        self.cache_capacity = cache_capacity
        self.handle_cache_capacity = handle_cache_capacity
        self.scale_on_device = scale_on_device
        self.raw_scale = raw_scale

        # Build manifest of all samples
        self._build_manifest()

    def _build_manifest(self):
        """Build a manifest of all possible samples."""
        self.manifest = []

        for file_path in self.file_paths:
            # For now, each file is one sample
            # In a full implementation, this could enumerate cutout positions, etc.
            self.manifest.append(
                {
                    "file_path": file_path,
                    "hdu": self.hdu,
                    "sample_id": len(self.manifest),
                }
            )

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Any:
        """Get a sample by index."""
        if idx < 0 or idx >= len(self.manifest):
            raise IndexError(f"Index {idx} out of range [0, {len(self.manifest)})")

        sample_info = self.manifest[idx]

        # Load data (import locally to avoid circular imports)
        from .io import read

        result = read(
            sample_info["file_path"],
            hdu=sample_info["hdu"],
            device=self.device,
            mmap=self.mmap,
            cache_capacity=self.cache_capacity,
            handle_cache_capacity=self.handle_cache_capacity,
            scale_on_device=self.scale_on_device,
            raw_scale=self.raw_scale,
            return_header=self.include_header,
        )

        if self.include_header:
            data, header = result
        else:
            data, header = result, None

        # Apply transform if provided
        if self.transform:
            data = self.transform(data)

        return (data, header) if self.include_header else data


class IterableFITSDataset(IterableDataset):
    """
    Streaming dataset for scenarios where the dataset is too large to list all files.

    Initialized with a URL to a sharded index file. Each worker processes assigned shards
    in a memory-efficient manner.
    """

    def __init__(
        self,
        index_url: str,
        hdu: int = 0,
        transform: Optional[Callable] = None,
        device: str = "cpu",
        shard_size: int = 1000,
    ):
        """
        Initialize IterableFITSDataset.

        Args:
            index_url: URL to sharded index file
            hdu: HDU index to read from each file
            transform: Optional transform to apply to each sample
            device: PyTorch device to load data onto
            shard_size: Number of files per shard
        """
        self.index_url = index_url
        self.hdu = hdu
        self.transform = transform
        self.device = device
        self.shard_size = int(shard_size)
        if self.shard_size <= 0:
            raise ValueError("shard_size must be positive")
        self._shards: list[list[str]] | None = None

    def __iter__(self) -> Iterator[Any]:
        """Iterate over the dataset."""
        # Get worker info for distributed processing (already highly optimized)
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            # Single process
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        # Load and process assigned shards
        for shard_id in self._get_assigned_shards(worker_id, num_workers):
            for sample in self._process_shard(shard_id):
                yield sample

    def _get_assigned_shards(self, worker_id: int, num_workers: int) -> List[int]:
        """Get shard IDs assigned to this worker."""
        shards = self._load_shards()
        total_shards = len(shards)
        start_shard = total_shards * int(worker_id) // int(num_workers)
        end_shard = total_shards * (int(worker_id) + 1) // int(num_workers)

        return list(range(start_shard, end_shard))

    def _process_shard(self, shard_id: int) -> Iterator[Any]:
        """Process a single shard and yield samples."""
        from .io import read
        from .logging import logger

        for file_path in self._load_shards()[shard_id]:
            try:
                data = read(file_path, hdu=self.hdu, device=self.device)

                # Apply transform if provided
                if self.transform:
                    data = self.transform(data)

                yield data
            except (IOError, RuntimeError, ValueError) as e:
                logger.error(f"Failed to process file {file_path}: {str(e)}")
                continue
            except Exception as e:
                logger.critical(f"Unexpected error processing {file_path}: {str(e)}")
                raise

    def _load_shards(self) -> list[list[str]]:
        """Load and cache the streaming manifest."""
        if self._shards is not None:
            return self._shards

        text = self._read_index_text()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            files = [
                line.strip()
                for line in text.splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
            shards = self._chunk_files(files)
        else:
            shards = self._shards_from_json(payload)

        self._shards = [shard for shard in shards if shard]
        return self._shards

    def _read_index_text(self) -> str:
        parsed = urlparse(self.index_url)
        if parsed.scheme in {"http", "https"}:
            with urlopen(self.index_url, timeout=30) as response:
                return response.read().decode("utf-8")
        if parsed.scheme == "file":
            return Path(unquote(parsed.path)).read_text(encoding="utf-8")
        return Path(self.index_url).read_text(encoding="utf-8")

    def _chunk_files(self, files: list[str]) -> list[list[str]]:
        return [
            files[start : start + self.shard_size]
            for start in range(0, len(files), self.shard_size)
        ]

    def _shards_from_json(self, payload: Any) -> list[list[str]]:
        if isinstance(payload, dict):
            if "shards" in payload:
                return [self._coerce_shard(shard) for shard in payload["shards"]]
            if "files" in payload:
                return self._chunk_files(self._coerce_file_list(payload["files"]))
        if isinstance(payload, list):
            if all(isinstance(item, (str, dict)) for item in payload):
                return self._chunk_files(self._coerce_file_list(payload))
            return [self._coerce_shard(shard) for shard in payload]
        raise ValueError("streaming index must be a JSON list or object")

    def _coerce_shard(self, shard: Any) -> list[str]:
        if isinstance(shard, dict):
            for key in ("files", "paths"):
                if key in shard:
                    return self._coerce_file_list(shard[key])
            raise ValueError("shard objects must contain 'files' or 'paths'")
        if isinstance(shard, list):
            return self._coerce_file_list(shard)
        if isinstance(shard, str):
            return [shard]
        raise ValueError("shards must be strings, lists, or objects")

    @staticmethod
    def _coerce_file_list(entries: Any) -> list[str]:
        if not isinstance(entries, list):
            raise ValueError("file entries must be a list")
        files: list[str] = []
        for entry in entries:
            if isinstance(entry, str):
                files.append(entry)
            elif isinstance(entry, dict):
                for key in ("path", "file", "url", "uri"):
                    if key in entry:
                        files.append(str(entry[key]))
                        break
                else:
                    raise ValueError("file objects must contain path/file/url/uri")
            else:
                raise ValueError("file entries must be strings or objects")
        return files


class TableChunkDataset(IterableDataset):
    """
    Iterable dataset that yields table chunks from one or more FITS files.
    """

    def __init__(
        self,
        file_paths: List[str],
        hdu: int = 1,
        columns: Optional[List[str]] = None,
        chunk_rows: int = 10000,
        max_chunks: Optional[int] = None,
        mmap: bool = False,
        device: str = "cpu",
        non_blocking_transfer: bool = True,
        pin_memory_transfer: bool = False,
        transform: Optional[Callable] = None,
        include_header: bool = False,
    ):
        self.file_paths = file_paths
        self.hdu = hdu
        self.columns = columns
        self.chunk_rows = chunk_rows
        self.max_chunks = max_chunks
        self.mmap = mmap
        self.device = device
        self.non_blocking_transfer = non_blocking_transfer
        self.pin_memory_transfer = pin_memory_transfer
        self.transform = transform
        self.include_header = include_header

    def __iter__(self) -> Iterator[Any]:
        from .io import get_header
        from .table import scan_torch

        emitted = 0
        for path in self.file_paths:
            header = get_header(path, self.hdu) if self.include_header else None
            for chunk in scan_torch(
                path,
                hdu=self.hdu,
                columns=self.columns,
                batch_size=self.chunk_rows,
                mmap=self.mmap,
                device=self.device,
                non_blocking=self.non_blocking_transfer,
                pin_memory=self.pin_memory_transfer,
            ):
                if self.transform:
                    chunk = self.transform(chunk)

                yield (chunk, header) if self.include_header else chunk
                emitted += 1
                if self.max_chunks is not None and emitted >= self.max_chunks:
                    break
            if self.max_chunks is not None and emitted >= self.max_chunks:
                break


def create_dataloader(*args: Any, **kwargs: Any) -> Any:
    from .dataloader import create_dataloader as factory

    return factory(*args, **kwargs)


def create_fits_dataloader(*args: Any, **kwargs: Any) -> Any:
    from .dataloader import create_fits_dataloader as factory

    return factory(*args, **kwargs)


def create_streaming_dataloader(*args: Any, **kwargs: Any) -> Any:
    from .dataloader import create_streaming_dataloader as factory

    return factory(*args, **kwargs)


def create_table_dataloader(*args: Any, **kwargs: Any) -> Any:
    from .dataloader import create_table_dataloader as factory

    return factory(*args, **kwargs)


__all__ = [
    "FITSDataset",
    "IterableFITSDataset",
    "TableChunkDataset",
    "create_dataloader",
    "create_fits_dataloader",
    "create_streaming_dataloader",
    "create_table_dataloader",
]
