"""FITS dataloaders — delegated to torchfits."""

from torchfits import (  # noqa: F401
    create_dataloader as create_dataloader,
    create_fits_dataloader as create_fits_dataloader,
    create_streaming_dataloader as create_streaming_dataloader,
    create_table_dataloader as create_table_dataloader,
)
from torchfits.dataloader import (  # noqa: F401
    create_distributed_dataloader as create_distributed_dataloader,
    create_ml_dataloader as create_ml_dataloader,
    get_optimal_dataloader_config as get_optimal_dataloader_config,
)
from torchfits.datasets import (  # noqa: F401
    FITSDataset as FITSDataset,
    IterableFITSDataset as IterableFITSDataset,
    TableChunkDataset as TableChunkDataset,
)
