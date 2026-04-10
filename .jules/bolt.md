
## 2024-05-24 - Vectorize bandpass loop **Learning:** Replaced loops inside PyTorch feature extractor with broadcasting over scales and then summing over spatial dimensions. Achieved 15x speedup by eliminating Python loops and multiple small tensor allocations. **Action:** Prefer vectorized broadcasting operations in PyTorch. Avoid using Python loop iterations to append calculated features to lists which is later stacked.

## 2024-05-24 - Faster Squaring **Learning:** We confirmed that `centered_sq = centered * centered` is faster than `torch.square(centered)`. **Action:** Always use element-wise multiplication `x * x` instead of `torch.square(x)` or `x ** 2` for better performance.
