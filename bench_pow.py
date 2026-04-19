import torch
import time

def bench():
    x = torch.randn(10000, 1000)

    t0 = time.time()
    for _ in range(100):
        y = x**2
    t1 = time.time()

    t2 = time.time()
    for _ in range(100):
        y = x * x
    t3 = time.time()

    print(f"**2 time: {t1-t0:.4f}s")
    print(f"* time  : {t3-t2:.4f}s")

bench()
