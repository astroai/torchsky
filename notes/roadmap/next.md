# Core engineering priorities

This is a durable implementation checklist for `torchsky`, not a scientific
experiment queue. Prioritize package work in this order, in line with
[`AGENTS.md`](../../AGENTS.md):

1. Close correctness and boundary defects in shared geometry, WCS, map
   operators, and data interchange.
2. Keep algorithms viable at survey scale through indexing, tiling, chunking,
   streaming, and sparse execution before considering native kernels.
3. Preserve tensor-native execution, differentiability, and claimed CPU/CUDA/MPS
   behavior with bounded tests.
4. Keep public APIs narrow and compatibility adapters explicit.
5. Support performance claims with measured evidence from the sibling
   [`torchsky-harness`](https://github.com/astroai/torchsky-harness), including
   realistic workload sizes where relevant.

These priorities do not imply that every subsystem has an active project or a
completed parity claim. Current public behavior is described by the domain
guides under `docs/`; scientific evidence and campaign status live in the
harness. The earlier experiment workqueue is preserved in the harness archive.
