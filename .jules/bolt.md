## 2024-04-04 - PyTorch Tensor Exponentiation Overhead
**Learning:** PyTorch tensor exponentiation (`** 2`) is measurably slower than explicit element-wise multiplication (`*`) due to the overhead of the underlying `pow` dispatch.
**Action:** When squaring PyTorch tensors on CPU, use explicit multiplication (`x * x`) instead of `x ** 2` or `torch.square(x)` in hot paths like coordinate transformations.
