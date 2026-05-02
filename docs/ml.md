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
from torchsky import FITSDataset
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
import torchsky

loader = torchsky.create_fits_dataloader(
    file_paths,
    hdu="auto",
    batch_size=32,
    num_workers=4,
)
```

### Tune Runtime Caches

```python
import torchsky

torchsky.configure_for_environment()
stats = torchsky.get_cache_stats()
torchsky.clear_file_cache()
```

## API Surface

Datasets and loaders:

- `FITSDataset`
- `IterableFITSDataset`
- `TableChunkDataset`
- `create_dataloader(...)`
- `create_fits_dataloader(...)`
- `create_streaming_dataloader(...)`
- `create_table_dataloader(...)`

Transforms:

- `ZScale`, `AsinhStretch`, `LogStretch`, `PowerStretch`
- `Normalize`, `MinMaxScale`, `RobustScale`
- `RandomCrop`, `CenterCrop`, `RandomFlip`, `RandomRotation`
- `GaussianNoise`, `PoissonNoise`, `PerturbByError`, `RedshiftShift`
- `ToDevice`, `Compose`
- `create_training_transform(...)`, `create_validation_transform(...)`,
  `create_inference_transform(...)`

Runtime:

- `configure_for_environment(...)`
- `get_cache_stats(...)`, `get_cache_performance(...)`, `clear_cache(...)`
- `clear_file_cache(...)` with knobs for data, handles, metadata, HDU types,
  stats, and C++ cache state
- `configure_buffers(...)`, `get_buffer_stats(...)`, `clear_buffers(...)`

## Examples

- [Image dataset example](../examples/example_image_dataset.py)
- [ML pipeline example](../examples/example_ml_pipeline.py)

## Validation And Performance

Use `benchmarks/bench_ml_loader.py`, `benchmarks/bench_cache.py`, and
`benchmarks/bench_buffer.py` for throughput and runtime tuning checks.
