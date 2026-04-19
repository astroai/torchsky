import torch
import torch.nn.functional as F
import time

def bench_mse():
    print("--- MSE comparison ---")
    pred = torch.randn(1000000)
    target = torch.randn(1000000)

    t0 = time.time()
    for _ in range(100):
        loss1 = torch.mean(torch.square(pred - target))
    t1 = time.time()

    t2 = time.time()
    for _ in range(100):
        diff = pred - target
        loss2 = torch.mean(diff * diff)
    t3 = time.time()

    t4 = time.time()
    for _ in range(100):
        loss3 = F.mse_loss(pred, target)
    t5 = time.time()

    print(f"torch.square : {t1-t0:.4f}s")
    print(f"x * x        : {t3-t2:.4f}s")
    print(f"F.mse_loss   : {t5-t4:.4f}s")

bench_mse()
