## Top Contenders for Best Reconstruction Accuracy

### 🥇 **Neural Experts (MoE-FINER)** — *Activation-based, NeurIPS 2024*
Achieves 90.84 PSNR vs. 84.56 for InstantNGP with an order of magnitude fewer parameters (366k vs 7.7M) — a MoE wrapper over FINER activations. Best raw fidelity numbers reported in cross-domain benchmarks.

### 🥈 **NSTR (Neural Spectral Transport Representations)** — *Novel activation/encoding, Nov 2025*
Claims significantly better accuracy–parameter trade-offs than SIREN, Fourier-feature MLPs, and Instant-NGP, requiring fewer global frequencies and converging faster by treating signals as spatially-evolving spectral fields rather than coordinate mappings. Still pre-publication but very strong.

### 🥉 **HC-INR (Hyper-Coordinate INR)** — *Hypernetwork-driven, 2025*
Achieves up to 2–4× higher reconstruction fidelity than strong INR baselines while using 30–60% fewer parameters, by learning signal-adaptive coordinate transformations instead of enriching encoding.

---

### Also notable, domain-specific:

- **M-plicits** (Multiscale residual SDF) — achieves state-of-the-art accuracy in surface reconstruction across standard benchmarks with a compact representation and real-time sphere tracing.
- **Spatially-Adaptive Hash** (WACV 2025) — beats Neuralangelo by allowing the network to dynamically choose encoding resolution as a function of space. Best hash-based method for surface reconstruction.
- **INCODE / Fr** — INCODE is the most robust approach for high-quality CT/inverse reconstruction when more data is available, while Fr is the most balanced in speed vs. accuracy.

---

**Bottom line:** If maximising reconstruction fidelity unconditionally, **Neural Experts FINER** is the current empirical leader. If you care about efficiency *and* accuracy at the frontier, **NSTR** is the most interesting recent idea. Hash-based methods remain dominant for speed but have been definitively surpassed on raw fidelity by well-tuned activation-based MoE approaches.

