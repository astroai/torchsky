## 2024-04-08 - PyTorch Power Operator Performance
**Learning:** PyTorch **2 power operator is slower than element-wise multiplication (*).
**Action:** Replace x**2 with x * x in hot paths, especially on tensors.
