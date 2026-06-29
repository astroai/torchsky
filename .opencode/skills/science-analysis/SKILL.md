---
name: science-analysis
description: Scientific Python / PyTorch data analysis — numpy, torch tensors, FITS tables, astronomy data workflows
license: MIT
compatibility: opencode
metadata:
  domains: astronomy, physics, data-science
---

## Patterns

- Use `torch` tensors for numeric work (device-aware: CPU/CUDA/MPS)
- Table data: prefer `pyarrow` / `polars`; pandas in compat layers only
- FITS I/O: delegate to `torchfits` (read_table, get_header, read_image)
- Large datasets: chunk, stream, use structured indices — no O(N*M) loops

## Key libraries (in order of preference)

1. `torch` — primary compute
2. `torchfits` — FITS I/O boundary
3. `pyarrow` / `polars` — table interchange
4. `numpy` — interop only where torch doesn't apply
5. `scipy` — test/bench extras only

## Numerical hygiene

- `float64` (torch.float64) for precise astronomy; avoid float32 unless measured
- Preserve differentiability in public paths
- Vectorize over loops; use torch.index_add / scatter / gather for sparse ops
- Assert shape/dtype invariants at function boundaries
