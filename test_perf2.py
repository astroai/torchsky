import torch
import time

def bench_vectorized_sum():
    # Simulate field difference
    diff = torch.randn(10000, 100)
    scales = torch.randn(100)

    t0 = time.time()
    for _ in range(100):
        # Current implementation in field/core.py:990
        radial2_1 = torch.sum(torch.square(diff / scales), dim=-1)
    t1 = time.time()

    t2 = time.time()
    for _ in range(100):
        # Alternative
        scaled_diff = diff / scales
        radial2_2 = torch.sum(scaled_diff * scaled_diff, dim=-1)
    t3 = time.time()

    print(f"torch.square : {t1-t0:.4f}s")
    print(f"x * x        : {t3-t2:.4f}s")

bench_vectorized_sum()
