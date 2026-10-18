# Scientific evaluation

Scientific benchmarks, upstream parity, large-data validation, reference-tool
comparisons, scorecards, and CANFAR experiment runners live in the sibling
[`torchsky-harness`](https://github.com/astroai/torchsky-harness) repository.
Its [methodology](https://github.com/astroai/torchsky-harness/blob/main/docs/benchmarks.md)
and [scorecards](https://github.com/astroai/torchsky-harness/tree/main/benchmarks/scorecards)
preserve the existing contracts and historical evidence.

Torchsky retains production implementations, small unit/API/regression tests,
adjoint and gradient checks, and bounded device/performance smoke tests.
Torchfits owns FITS/table I/O correctness and performance.

Install the chosen torchsky build and run evaluation from the harness checkout:

```sh
pixi run torchsky-harness list --domain map
pixi run torchsky-harness subject
pixi run torchsky-harness run bench_map_compat -- --size 32 --n-obs 2 --runs 1 \
  --cases synthetic_shifted_wcs --torchsky-kernels bilinear
```

The harness records the imported subject and distinguishes process completion
from scientific validity. Core pre-push remains `pixi run preflight-push`;
full local tests and scientific campaigns remain opt-in.
