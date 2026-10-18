# Package architecture history

This page records durable ownership decisions. Scientific run histories,
benchmark tables, scorecards, and experiment handoffs belong to
[`torchsky-harness`](https://github.com/astroai/torchsky-harness).

- **FITS and table ownership:** file parsing, HDUs, checksums, table mutation,
  Arrow conversion, and I/O throughput live in
  [`torchfits`](https://github.com/astroai/torchfits). Torchsky consumes tensors,
  header mappings, and table batches through explicit boundaries.
- **Native compute ownership:** torchsky owns its coordinate, WCS, spherical,
  catalog, map, scene, and continuous-field algorithms. Compatibility adapters
  sit above torchsky-native implementations.
- **Validation ownership:** core retains bounded unit/API/regression,
  adjoint, gradient, and device-contract tests. Exhaustive upstream parity,
  real-data validation, and performance acceptance moved to the harness.
- **Experiment records:** full pre-split implementation and campaign snapshots
  are preserved in the harness extraction archive; the core roadmap no longer
  treats historical run state as a live task queue.

The current package map is in [status.md](status.md), and stable engineering
priorities are in [next.md](next.md). See [`AGENTS.md`](../../AGENTS.md) for
the active workflow and test policy.
