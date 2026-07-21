# ML, Transforms, And Runtime Tuning

Torchsky owns tensor transforms, observation models, training utilities, and
compute buffers. File-backed FITS datasets and FITS table chunk readers belong
to `torchfits` or to application code composed around it.

## Data Boundary

Build a PyTorch `Dataset` or `IterableDataset` that loads with `torchfits`, then
apply torchsky tensor transforms:

```python
import torchfits
from torch.utils.data import Dataset
from torchsky.transforms import Compose, RandomCrop, ZScale


class ImageDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths
        self.transform = Compose([RandomCrop(256), ZScale()])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image, header = torchfits.read(
            self.paths[index], hdu="auto", return_header=True
        )
        return self.transform(image), header
```

For massive datasets, prefer manifest-backed iterable datasets with explicit
sharding, bounded prefetch, deterministic worker partitioning, and measurable
peak memory. Do not hide file I/O in geometry or model objects.

## Torchsky-Owned Surface

Transforms:

- `ZScale`, `AsinhStretch`, `LogStretch`, `PowerStretch`
- `Normalize`, `MinMaxScale`, `RobustScale`
- `RandomCrop`, `CenterCrop`, `RandomFlip`, `RandomRotation`
- `GaussianNoise`, `PoissonNoise`, `PerturbByError`
- `ToDevice`, `Compose`
- `create_training_transform(...)`, `create_validation_transform(...)`,
  `create_inference_transform(...)`

Runtime buffers:

- `torchsky.buffer.configure_buffers(...)`
- `torchsky.buffer.get_buffer_stats(...)`
- `torchsky.buffer.clear_buffers(...)`

FITS handle/data/metadata cache policy belongs to `torchfits`.

Wavelength-aware flux-conserving rebinning and relativistic Doppler shifts live
in `torchsky.spectral`; they are not index-based image augmentation transforms.

## Validation And Performance

Measure end-to-end training input throughput separately from model throughput.
Report worker count, batch size, transfer volume, peak host/device memory, and
the point where input becomes the bottleneck. FITS decode/cache microbenchmarks
belong in `torchfits`; torchsky benchmarks should focus on transforms,
operators, and model execution.
