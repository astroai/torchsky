# Reference benchmarks

These tables are a bounded, single-machine snapshot of operations `torchsky`
owns. They answer "what does this kernel cost on this class of input", not
"is torchsky scientifically correct" and not "is it faster than another
package". Upstream parity, survey-scale validation, real-data checks, and
comparable scorecards belong to
[`torchsky-harness`](https://github.com/astroai/torchsky-harness).
Torchsky retains production implementations and the bounded kernel snapshots below.

<!-- BENCH_TABLES -->

## How these were measured

- One process, one thread (`OMP_NUM_THREADS=1`, `torch.set_num_threads(1)`),
  CPU unless the environment table names a GPU.
- One warm-up call, then the median of three timed calls (`anafast`,
  `crossmatch_sky`, and `coadd` use two timed calls to keep the run bounded).
- Inputs are synthetic, float64 for coordinate geometry and float32 for image
  coadds. Sizes are chosen to expose scaling, not to represent a survey.
- Throughput counts the operation's natural unit: transformed points, analysed
  map pixels, left catalog rows, or input image pixels.

Regenerate the artifact and this page on the reference machine:

```bash
pixi run docs-bench
pixi run docs-bench-tables
```

The JSON snapshot is
[`docs/data/reference_benchmarks.json`](../data/reference_benchmarks.json).
Treat numbers from different machines as different measurements; do not compare
them without recording both environments.

## What the numbers do and do not show

- They show the cost of the current Python/Torch path and of the optional
  compiled kernels when installed. A snapshot taken with
  `TORCHSKY_DISABLE_CPP=1` must say so; the generator records the device but
  not that flag, so regenerate without it.
- They do not gate CI: shared runners are too noisy, and a regression gate
  belongs in the bounded performance smoke tests (`tests/test_perf_gate.py`).
- They are not a substitute for the harness scorecards. See
  [parity and correctness evidence](parity.md) for what is checked against
  trusted references.

See also: [scaling guide](scaling.md) and the
[runtime contract](../reference/runtime-contract.md).
