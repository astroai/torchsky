---
name: science-analysis
description: Scientific Python / PyTorch data analysis for torchsky — torch tensors, torchfits FITS boundary, pyarrow/polars tables
metadata:
  domains: astronomy, physics, data-science
---

# Science analysis patterns

## Compute

- Use `torch` tensors for numeric work, device-aware (CPU/CUDA/MPS).
- `float64` (torch.float64) for precise astronomy; avoid float32 unless measured.
- Preserve differentiability in public paths.
- Vectorize over loops; use `torch.index_add` / `scatter` / `gather` for sparse ops.
- Assert shape/dtype invariants at function boundaries.

## Data

- FITS I/O: delegate to `torchfits` (read, table.scan, get_header) — torchsky
  contains no FITS code.
- Tables: prefer `pyarrow` / `polars`; pandas in compat layers only.
- Large datasets: chunk, stream, use structured indices — no O(N*M) loops.

## Library preference order

1. `torch` — primary compute
2. `torchfits` — FITS I/O boundary
3. `pyarrow` / `polars` — table interchange
4. `numpy` — interop only where torch doesn't apply
5. `scipy` — test/bench extras only
