# WCS small-N investigation log

**Status:** parked (2026-06-12). No code changes merged; baseline remains committed
`main` before this work.

## Problem

On the two-argument bench path `pixel_to_world(x, y)` (CPU, `float64`, no grad),
zenithal projections (`TAN`, `SIN`, `ARC`) at **N ≤ ~1K** are ~**3× slower**
than Astropy `all_pix2world` on the same pixels. Parity is excellent (sub-arcsec);
the gap is wall time, not semantics.

At **N ≥ 10K** torchsky already wins on several WCS cases (see scorecard). The
annoying deficit is specifically **small-batch zenithal forward** on the legacy
two-arg API shape the smart WCS bench measures.

## Pre-work baseline (do not regress against mid-session numbers)

Measured on committed `main` **before** any small-N WCS edits (`1674ac8` at time
of investigation). Tool: `benchmarks/bench_wcs_small_n_paths.py`, `runs=31`,
`origin=0`, `pixi run -e bench-all dev` rebuild first.

| Case | N | Astropy `p2w` | torchsky `p2w_xy` | ts/ast |
|------|---|---------------|-------------------|--------|
| TAN | 256 | 0.0257 ms | 0.0730 ms | 2.84× |
| SIN | 256 | 0.0217 ms | 0.0669 ms | 3.08× |
| ARC | 256 | 0.0196 ms | 0.0592 ms | 3.02× |
| CEA | 256 | 0.0099 ms | 0.0152 ms | 1.54× |
| CAR | 256 | 0.0086 ms | 0.0101 ms | 1.17× |
| TAN | 1000 | 0.0739 ms | 0.0965 ms | 1.31× |
| TAN | 10000 | 0.7934 ms | 0.5439 ms | **0.69× (win)** |
| CAR | 10000 | 0.1272 ms | 0.0608 ms | **0.48× (win)** |

Reproduce:

```bash
pixi run -e bench-all dev
pixi run -e bench-all -- python benchmarks/bench_wcs_small_n_paths.py \
  --cases TAN,SIN,ARC,CEA,CAR,AIT,HPX,MER --n-points 256 --runs 31
```

Committed multi-domain scorecard snapshot: `benchmarks/scorecards/20260612_main_domains.{md,json}`.

### Committed dispatch order (reference)

For `pixel_to_world(x, y)` on pole-native zenithal (typical bench headers):

1. Center-equator C++ helper (N-gated; not used for pole TAN)
2. Inline cylindrical (CAR/SFL/CEA/MER only)
3. **`cpp_direct` zenithal** — vectorized C++ projection + pole rotation (small-N path today)
4. Fused C++ full pipeline (after `cpp_direct`; better at large N for some codes)
5. Python core / optional `compile()` stacked path

Micro-profile (N=256 TAN, pre-work build): raw fused C++ ~**0.035 ms** vs
`cpp_direct` chain ~**0.057 ms** vs full `_pixel_to_world_2d_fast` ~**0.058 ms**
vs Astropy ~**0.021 ms**. So fused kernel is faster than committed small-N path,
but still ~1.7× Astropy; Python dispatch overhead is only ~0.002 ms.

## What we tried (2026-06-12 session)

All experiments were **reverted**; nothing below is on `main`.

| # | Approach | Result | Verdict |
|---|----------|--------|---------|
| 1 | **Fused-first** for all fused-eligible projections | CEA/CAR regressed (~3× vs committed); zenithal OK only when extension loaded | Reject as blanket policy |
| 2 | **Instance scratch buffers** + reuse in fused path | Saved alloc; **bug**: returning views into scratch let `world_to_pixel` clobber `ra`/`dec` | Need `.clone()` on return or separate in/out pools |
| 3 | **Semantics-based center-cpp skip** when fused/inline applies | Theoretically sound; bundled with (1), hard to isolate | Revisit with zenithal-only scope |
| 4 | **Plain TAN vectorized fast path** before fused | ~0.077 ms at N=256 — slower than `cpp_direct` | Reject |
| 5 | **Fused-first for zenithal pole only** + scratch | TAN N=256: **0.050 ms** vs committed **0.073 ms** (~−23%); still **2.3×** Astropy | Best candidate; not merged |
| 6 | **C++ `wcs_pixel_to_world_zenithal_pole_cpu`** (CD+zenithal+rotation, one nanobind) | One-shot ~0.14 ms at N=256 when tested with stale build; not clearly faster than fused | Needs clean A/B |
| 7 | **C++ fused: serial loop for n ≤ 8192** (skip `parallel_for` setup) | Fused raw ~0.035–0.039 ms at N=256 on clean build | Small win; pair with (5) |
| 8 | **Benchmark-backed crossover** `_ZENITHAL_FUSED_P2W_MAX` (1536–8192) | Fused wins below ~1.5K–2K; `cpp_direct` wins above ~8K | Correct pattern if reviving (5)+(7) |
| 9 | **`torch.compile` on `_pixel_to_world_2d_fast`** | Slower (~0.078 ms) | Reject for two-arg path |

### Failure modes to avoid

- Routing to fused when the **editable extension is stale** (binding missing) while
  **`cpp_direct` for zenithal is disabled** → falls through to slow Python core (~0.15 ms).
- Always run `pixi run -e bench-all dev` before benchmarking C++ changes.

## Assessment

- **Not a correctness crisis:** scorecard deficits at N ≤ 10K are dispatch/overhead,
  not WCS semantics.
- **Survey-scale paths are fine:** N ≥ 10K and stacked/`compile()` paths are
  competitive or faster.
- **Beating Astropy on zenithal `p2w_xy` at N≈256** likely needs a **deeper kernel
  change** (single vectorized binding or WCSLIB-class tight loop), not more Python
  dispatch shuffling.

## Possible future directions

Ordered by estimated leverage vs risk.

1. **Revive (5)+(7)+(8) only for zenithal pole** — fused below ~1.5K, `cpp_direct`
   above; scratch inputs, fresh outputs, no blanket fused-first. Target: beat
   committed baseline; accept ~2× Astropy at N=256 unless kernel work follows.
2. **Monomorphic executor bound at `WCS.__init__`** — one function pointer per
   projection family; eliminates per-call branch chain (~0.002 ms, minor alone).
3. **Ship `wcs_pixel_to_world_zenithal_pole_cpu` properly** — one nanobind hop
   for CD+projection+rotation; benchmark vs fused scalar on clean builds.
4. **Improve fused C++ scaling at large N** — scalar fused loses to vectorized
   `cpp_direct` above ~2K–8K; vectorize the fused inner loop or tune grain.
5. **Optional second bench row** in `bench_wcs_small_n_paths.py` for “fused-only
   zenithal” to separate kernel vs dispatch in CI notes (not a gate).
6. **Long shot:** WCSLIB-backed or NumPy-buffer fast path for CPU `float64`
   two-arg calls (conflicts with torch-native mission unless behind explicit compat
   flag).

## Acceptance criteria (if resumed)

- **Must:** `ts_p2w_xy` ≤ pre-work baseline at N ∈ {256, 1000, 10000} for
  TAN/SIN/ARC (no large-N regression).
- **Stretch:** `ts_p2w_xy` < Astropy `ast_p2w` at N=256 for zenithal (requires
  kernel-level work, not dispatch-only).
- **Always:** `pixi run ci-local` green; parity unchanged.

## Related docs

- [`docs/benchmarks.md`](../benchmarks.md) — `bench_wcs_small_n_paths.py`, scorecard policy
- [`benchmarks/scorecards/README.md`](../../benchmarks/scorecards/README.md) — archived scorecards
- [`docs/archive/benchmark-history.md`](../archive/benchmark-history.md) — older benchmark narrative
