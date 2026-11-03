# Scaling and tuning

How to run `torchsky` at scale, which knobs bound memory, and how to measure a
change without fooling yourself. This page is about *execution* performance.
Scientific parity, real-data validation, and scorecards are owned by
[`torchsky-harness`](https://github.com/astroai/torchsky-harness) and summarized
in the [kernel benchmarks](benchmarks.md); headline end-to-end results are in
[Performance highlights](highlights.md).

## Cost model

Optimize in this order:

1. **Algorithm and indexing.** Candidate pruning, spherical indexing, sparse
   representations, tiling, and chunking decide scale long before a kernel does.
2. **Data layout.** Contiguity, dtype, and avoiding host↔device round trips
   inside a loop usually dominate micro-optimizations.
3. **Kernel selection.** Choose the resampling kernel or HEALPix ordering the
   problem needs; the fastest kernel that changes the answer is not an
   optimization.
4. **Native fast paths.** `torchsky._cpp` accelerates proven hotspots. It is
   optional, and the pure-Torch paths remain the correctness baseline.

Every performance-sensitive change should be able to answer: which bottleneck is
removed, why it scales better at large `N`, and what the measurement was at both
a small and a large size.

## Memory-bounded execution

Survey-scale products are bounded by configuration, not by hope. The main knobs:

| Surface | Knob | What it bounds |
|---|---|---|
| `torchsky.map.coadd` | `stack=StackConfig(..., tile_size=1024)` | Output tile working set during stacking |
| `torchsky.map.coadd` | `spatial_tile_hw=(512, 512)` | Spatial tile for projection and accumulation |
| `torchsky.map.image_plane_coadd.coadd_image_plane` | `spatial_tile_hw=None` (automatic), `tile_memory_budget_bytes`, `auto_tile` | Tiles are chosen from a memory budget (30% of free device memory on CUDA, of available host/cgroup memory otherwise; `TORCHSKY_TILE_MEMORY_BUDGET_BYTES` overrides). Targets that fit are not tiled; a tiled CUDA run keeps the full-size accumulators on the host and returns the results on the compute device (on the host with a `RuntimeWarning` if they would not fit the free device memory) |
| `torchsky.map.image_plane_coadd.coadd_image_plane` | `propagate_variance=False` | Skips the exact diagonal-variance sampling pass (`inverse_variance`/`correlation_factor` are then `None`) |
| `torchsky.map.aperture_uncertainty` | `spatial_tile_hw=(512, 512)` | Tile size for the finite-aperture adjoint |
| `ResampleConfig` | `response_chunk_size=65536` | Source pixels processed per response chunk |
| `ResampleConfig` | `max_response_entries` | Hard cap on response entries; fails closed instead of growing |
| `torchsky.map` scratch paths | `scratch_dir=...` | Location of capacity-checked scratch storage for exact median, sigma-clipped stacks, and overlap background fits |
| `torchsky.sphere.query_*` | `chunk_size=1_000_000`, `fact` | Query working set and subsampling |
| `torchsky.catalogs` + `torchfits.table.scan` | `batch_size` | Streamed table batches instead of a full in-memory catalog |
| HEALPix | `SparseHealpixMap`, `MultiOrderHealpixMap` | Sparse coverage instead of dense `12 * nside**2` arrays |

Things that are deliberately not bounded by a silent fallback: a custom
background estimator without compact tile rendering is rejected in the bounded
path; unsupported uncertainty propagation returns an absent uncertainty instead
of a proxy; `max_response_entries` raises rather than allocating past the cap.
Those are contracts, not bugs.

## Device and dtype

- Inputs decide the device. Passing CUDA or MPS tensors keeps the computation
  there; nothing promotes to GPU behind your back. `coadd(..., device=...)` is an
  explicit override.
- `float64` coordinates preserve precision through WCS and geometry transforms;
  `float32` is faster but can lose sub-milliarcsecond accuracy in repeated
  round trips. Measure the accuracy you need before trading it away.
- CPU is the baseline and the reference for every correctness claim. CUDA and MPS
  are supported where the surrounding module claims them; a missing kernel is an
  error, not a silent CPU fallback.
- Broad device contract: [runtime contract](../reference/runtime-contract.md).

## Native extension and fallback

`torchsky._cpp` is an optional compiled extension for geometry and HEALPix
hotspots. Without it the same Python/Torch kernels run with identical semantics
and lower throughput.

- Set `TORCHSKY_DISABLE_CPP=1` to force the fallback when diagnosing a native
  crash or an ABI mismatch.
- Never publish a native performance claim from a run that used the fallback.
- `tests/test_perf_gate.py` skips itself when the extension is unavailable, so a
  pass there means the native path was exercised.
- Native-versus-Torch parity is a correctness gate, not a benchmark: the
  fallback must produce the same result, and that is what the package tests
  assert.

## Measuring

Use warmup plus repeated timings and report the median, not a single run:

<!-- docs: run -->
```python
import statistics
import time

import torch
import torchsky.sphere as sphere

nside = 512
n = 20_000
lon = torch.linspace(0.0, 359.0, steps=n, dtype=torch.float64)
lat = torch.linspace(-60.0, 60.0, steps=n, dtype=torch.float64)


def run() -> None:
    sphere.ang2pix(nside, lon, lat, nest=True, lonlat=True)


run()  # warmup (compiled dispatch, caches)
samples = []
for _ in range(5):
    start = time.perf_counter()
    run()
    samples.append(time.perf_counter() - start)

print(f"median {statistics.median(samples) * 1e3:.2f} ms for {n} points")
assert statistics.median(samples) > 0.0
```

The bounded package gates use the same shape and live in
`tests/test_perf_gate.py`. Run them with:

```bash
pixi run -e test -- pytest tests/test_perf_gate.py -q
```

Each gate is a loose upper bound, not a target, and can be overridden while
bisecting a regression:

- `TORCHSKY_PERF_GATE_ANG2PIX_S`
- `TORCHSKY_PERF_GATE_INTERP_WEIGHTS_S`
- `TORCHSKY_PERF_GATE_WCS_P2W_S`
- `TORCHSKY_PERF_GATE_WCS_W2P_S`
- `TORCHSKY_PERF_GATE_WCS_N`

Report, alongside every number:

- the exact operation and parameters (nside, `n`, shape, kernel, dtype, device);
- the median and the spread (min/max or IQR), with warmup policy;
- the machine and runtime: CPU/GPU model, PyTorch, torchsky, torchfits, git SHA,
  date, and thread settings;
- whether the compiled extension was active.

For memory, use `torch.cuda.max_memory_allocated()` on CUDA and resident-set
sampling (`psutil`) on CPU; state the tool and the sampling interval. Do not mix
GPU and CPU timings in one table.

### What not to measure

- Do not benchmark FITS read/write here; throughput and I/O parity belong to
  `torchfits`.
- Do not quote a single small run as evidence for survey scale.
- Do not compare against a comparator in a different environment and call the
  ratio a speedup. Comparator environments and campaigns live in the harness.
- Do not gate correctness on wall time; CI runners are noisy. The gates catch
  catastrophes, not percentage regressions.

## Where the numbers live

- **Reference tables.** A committed measurement artifact renders the
  `torchsky`-owned hot paths (WCS, HEALPix, queries, harmonic transforms,
  crossmatch, coadd) at small/medium/large sizes, with machine metadata. See the
  [reference benchmarks](benchmarks.md). Regenerate it deliberately; it is not a
  CI gate.
- **Scientific evidence.** Parity, survey-scale, and real-data results are
  reported by `torchsky-harness`, linked with its subject and harness revisions.
  This repository links those results; it does not re-run them.
- **Bounded smoke gates.** `tests/test_perf_gate.py` stays cheap enough for CI
  and covers only torchsky-owned hot paths.

## Related

- [Coadds and resampling](../maps/index.md) — value semantics, kernels, tiling, uncertainty.
- [`torchsky.map` reference](../reference/map.md) — configuration fields and defaults.
- [Sphere pixels and geometry](../sphere/index.md) — HEALPix, queries, sparse maps, MOC.
- [Runtime contract](../reference/runtime-contract.md) — device, dtype, and failure
  semantics.
- [Contributing](../project/contributing.md) — performance evidence expectations for PRs.
