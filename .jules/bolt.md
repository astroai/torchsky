## 2024-04-06 - [PyTorch Tensor Power Operations]
**Learning:** In PyTorch, using the power operator `**2` on tensors is noticeably slower than explicitly performing element-wise multiplication `x * x` due to the overhead of the generalized power function kernel dispatch.
**Action:** When calculating squared values in hot loops or iterative algorithms like Newton-Raphson, replace `tensor**2` with `tensor * tensor` or `torch.square(tensor)` for micro-optimizations that compound over iterations.
