# SMACS grid-mapping validation screen — 2026-09-29

This is a three-batch validation screen, not a final depth claim. Both H100
headless 1024² reconstructions completed with exit code 0 from exact packaged
commit `5d7978ead61edbb853b611ce6c022870f5157ae6`. The 512² treatment completed with
exit code 0 in session `xjb93a1r` from exact packaged commit
`95a11d8454986d58f212a56d5a312d650dcced17`.

| Metric | Legacy 1024² | Footprint 1024² | Footprint 512² |
| --- | ---: | ---: | ---: |
| Mean grid cell, F150W/F277W | 0.142″ | 0.032″ | 0.064″ |
| Pooled completeness | 0.4405 | 0.3929 | 0.4643 |
| Pooled purity | 0.9737 | 0.9706 | 0.9512 |
| m50 (AB) | 28.50 [28.25, 28.67] | 28.25 [28.17, 28.50] | 28.50 [28.25, 28.67] |
| F150W effective FWHM | 4.790 px | 5.085 px | 5.042 px |
| F277W effective FWHM | 4.875 px | 4.466 px | 4.529 px |
| F150W negative-lobe fraction | 0.613 | 0.798 | 0.781 |
| F277W negative-lobe fraction | 0.257 | 0.355 | 0.341 |

The target-footprint mapping improves several reconstruction-domain fidelity
metrics, especially in F277W, but the injected-source evidence does not meet the
combined goal of greater depth with no resolution loss. The 1024² candidate is
rejected and must not be expanded to six bands or the final 20-batch test.

The 512² treatment increases pooled completeness from 0.4405 to 0.4643 and ties
the legacy `m50`, but it does not improve depth. Its F150W effective FWHM is
5.3% worse than legacy while F277W is 7.1% better, so it fails the requirement
of non-worse resolution in both bands. Negative-lobe fractions are also higher
in both bands. The candidate is rejected and will not expand to six bands or the
final 20-batch test. Because the denser 1024² treatment also failed, a four-cell-
per-output-pixel mapping is not justified by this screen; the next grid work is
the independently solved weighted inverse problem in the roadmap.

Scoring session `mlv5wipx` completed with exit code 0 on production CANFAR. It
used 10,000 batch-bootstrap draws, common support of 237,204 pixels and scorer
SHA-256 `7885bef5283acbdbc624fa9532e95d5a69ac7f9dc80cce8c60e0a30e18e81c4b`.
All four configuration audits permit only the declared spatial-resolution and
mapping/bounds changes. Results remain screening-only (`depth_claim=false`).
