## 2024-05-18 - PyTorch Tensor Power Operation Bottleneck
**Learning:** In PyTorch, computing the square of elements using the power operator `tensor**2` incurs significant overhead compared to element-wise multiplication `tensor * tensor`. This was identified in the Newton-Raphson hot loop for WCS inversion (`src/torchsky/wcs/utils.py`).
**Action:** When working on numerical algorithms in PyTorch, especially in tight loops, always prefer `x * x` over `x**2` for computing squares to improve performance.
