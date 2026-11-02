# Parity and correctness evidence

!!! warning "Historical baseline (torch 2.13 / torchsky 0.4 pre-release)"
    The tables below summarize the 0.4.0 release-candidate baseline (torch 2.13, measured 2026-10). Current-lane results (torch 2.14, torchsky 0.5) come from the
    `torchsky-harness` CANFAR evaluation campaigns; see the
    [CANFAR guide](https://github.com/astroai/torchsky-harness/blob/main/docs/canfar.md).

Scientific correctness is evaluated against trusted references in
[`torchsky-harness`](https://github.com/astroai/torchsky-harness), not in this
repository. This page records what that evidence covers, where it lives, and
what it does **not** prove. The tables summarize the 0.4.0 release-candidate
baseline linked below; they are reported by the harness, not re-measured here.

## What is compared, and against what

| Domain | Reference | Gate | Reported result |
|---|---|---|---|
| WCS celestial transforms | Astropy (WCSLIB) | 38 cases across 30 projections plus reference-latitude-offset cases | Angular parity max `6.1e-7 asec`; sky round trip `1.28e-6 asec`; pixel round trip `1.27e-6 px` |
| WCS throughput | Astropy | Same cases, same machine | Forward `0.46–2.65x`, inverse `0.88–4.32x` |
| HEALPix indexing and queries | healpy | 12 operations at nside 512 | Zero index/angle mismatches; max `1.14e-13 deg`; throughput `1.02–2.19x` |
| Spherical and spin SHT | healpy | Transform and spectrum parity | Gates pass; throughput ratios recorded, not gated |
| Catalog crossmatch | Astropy `SkyCoord` | Separation parity | Max difference `1.13e-10 asec` |
| CUDA WCS | Astropy | 17 cases including offset cases | All pass; max angular parity `8.2e-3 asec` |

Source: the 0.4.0 release-candidate baseline in
[`torchsky-harness/docs/releases/0.1.0/baseline.md`](https://github.com/astroai/torchsky-harness/blob/main/docs/releases/0.1.0/baseline.md).
That document carries the subject artifact identity, the environment, and each
gate's status.

## What the gates mean

- **Parity** means the same operation on the same inputs agrees with the
  trusted implementation to the stated tolerance — not that the two packages
  share an API or internal representation.
- **Round trips** check `pixel → world → pixel` and the inverse where the
  transform is invertible. They catch sign, origin, and projection-boundary
  errors that a single forward comparison can miss.
- **Throughput ratios** are context for the parity run. They are machine- and
  size-specific and are not a performance claim for other workloads.

## What is not covered here

- The bounded tests in `tests/` (adjoints, gradients, device contracts, API
  regressions) are necessary but are not a substitute for the reference
  comparisons above.
- Large-data, real-survey, and CANFAR campaign results live in the harness
  release bundle. They are not copied here because their inputs, versions, and
  environment must travel with them.
- The reference benchmark tables on this site measure torchsky's own runtime,
  not agreement with another package. See
  [reference benchmarks](benchmarks.md).

## Reproduce a gate

From a harness checkout with a subject build installed:

```bash
pixi run torchsky-harness list --domain wcs
pixi run torchsky-harness subject
pixi run torchsky-harness run bench_wcs_parity -- --cases all
```

The harness records the imported subject and distinguishes process success from
valid scientific results. Follow its methodology page rather than invoking
comparator packages directly.
