# ML, Datasets, Transforms, And Runtime Tuning

This page covers PyTorch datasets, dataloader helpers, transforms, cache tuning,
and buffer utilities used in survey-scale ML pipelines.

## What This Surface Is For

Use these helpers when FITS images or table chunks are training data. The goal is
to keep ingestion tensor-native, chunkable, and compatible with PyTorch
`DataLoader` patterns.

## Core Concepts

- `FITSDataset` is map-style and works well for random image access.
- `IterableFITSDataset` streams large file collections and supports multi-worker
  loading.
- `TableChunkDataset` streams FITS table chunks.
- Transforms are tensor-first and can run on GPU when their operations support
  the target device.
- Cache and buffer helpers expose explicit knobs for long-running processes.

## Recipes

### Build An Image Dataset

```python
from torchsky.datasets import FITSDataset
from torchsky.transforms import Compose, RandomCrop, ZScale

dataset = FITSDataset(
    file_paths,
    hdu="auto",
    transform=Compose([RandomCrop(256), ZScale()]),
    mmap="auto",
)
```

### Create A DataLoader

```python
from torchsky.datasets import create_fits_dataloader

loader = create_fits_dataloader(
    file_paths,
    hdu="auto",
    batch_size=32,
    num_workers=4,
)
```

### Tune Runtime Caches

```python
from torchsky import cache

stats = cache.stats()
cache.clear()
```

## API Surface

Datasets and loaders:

- `FITSDataset`
- `IterableFITSDataset`
- `TableChunkDataset`
- `torchsky.datasets.create_dataloader(...)`
- `torchsky.datasets.create_fits_dataloader(...)`
- `torchsky.datasets.create_streaming_dataloader(...)`
- `torchsky.datasets.create_table_dataloader(...)`

Transforms:

- `ZScale`, `AsinhStretch`, `LogStretch`, `PowerStretch`
- `Normalize`, `MinMaxScale`, `RobustScale`
- `RandomCrop`, `CenterCrop`, `RandomFlip`, `RandomRotation`
- `GaussianNoise`, `PoissonNoise`, `PerturbByError`, `RedshiftShift`
- `ToDevice`, `Compose`
- `create_training_transform(...)`, `create_validation_transform(...)`,
  `create_inference_transform(...)`

Runtime:

- `torchsky.cache.stats(...)`
- `torchsky.cache.clear(...)`
- `torchsky.io.get_cache_performance(...)`
- `torchsky.io.clear_file_cache(...)` with knobs for data, handles, metadata, HDU types,
  stats, and C++ cache state
- `torchsky.buffer.configure_buffers(...)`,
  `torchsky.buffer.get_buffer_stats(...)`, `torchsky.buffer.clear_buffers(...)`

## Examples

- [Image dataset example](../examples/example_image_dataset.py)
- [ML pipeline example](../examples/example_ml_pipeline.py)

## Validation And Performance

Use `benchmarks/bench_ml_loader.py`, `benchmarks/bench_cache.py`, and
`benchmarks/bench_buffer.py` for throughput and runtime tuning checks.
