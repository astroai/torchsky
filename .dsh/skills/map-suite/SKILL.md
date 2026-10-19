---
name: map-suite
description: Run bounded core map API and operator contract tests; use torchsky-harness for scientific comparisons and campaigns
metadata:
  project: torchsky
---

# Core map contract tests

Map package tests protect runtime behavior: product validation, numerical
contracts, gradients, adjoints, and device support. Scientific truth metrics,
comparator suites, large-data validation, campaign scorecards, and CANFAR
evaluation live in
[`torchsky-harness`](https://github.com/astroai/torchsky-harness). FITS I/O
tests and throughput gates belong to `torchfits`.

`pixi run test` excludes `tests/test_map_*.py`; those files are exercised in the
GitHub CI map matrix. For focused local changes, run only the relevant core
tests, for example:

```bash
pixi run -e test -- pytest tests/test_map_image_plane.py tests/test_map_product.py -q
```

Do not use this skill to run the scientific benchmark suite. See the
[harness map evaluations](https://github.com/astroai/torchsky-harness/tree/main/benchmarks)
and its [`docs/benchmarks.md`](https://github.com/astroai/torchsky-harness/blob/main/docs/benchmarks.md)
for the subject installation, gate criteria, and evidence policy.
