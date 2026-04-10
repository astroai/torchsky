## 2024-04-10 - [Vectorizing Range Finding]
**Learning:** Finding contiguous ranges in sorted NumPy arrays using Python loops is significantly slower than using vectorized NumPy operations. Utilizing `np.diff` with `np.nonzero` allows calculating range breaks and lengths efficiently, yielding over a 5x speedup for 10k-element arrays while remaining clean and concise. Edge cases involving empty arrays must be explicitly handled.
**Action:** Always prefer NumPy vectorization (like `np.diff` and `np.nonzero`) over raw Python iteration when identifying continuous blocks or calculating intervals within numeric arrays.

## 2025-04-10 - [PyTorch Fourier Features Vectorization]
**Learning:** Iterating over tensors and appending PyTorch operations iteratively to lists (`for scale in scales: feats.append(...)`) is slow. By broadcasting dimensions using `unsqueeze`, we can perform operations simultaneously. Interleaving `sin` and `cos` features can be elegantly done by combining `torch.stack` and `.view()`.
**Action:** Always look for opportunities to replace `for` loops with vectorized PyTorch operations using dimension broadcasting, stacking, and reshaping.
