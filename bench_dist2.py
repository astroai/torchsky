import torch
import time

def bench():
    # Simulate WCS dist2 calculation
    r_ra = torch.randn(1000000)
    r_dec = torch.randn(1000000)

    # Warmup
    for _ in range(10):
        dist2 = r_ra**2 + r_dec**2
        dist2 = r_ra * r_ra + r_dec * r_dec

    # Pow method
    t0 = time.time()
    for _ in range(100):
        dist2 = r_ra**2 + r_dec**2
    t1 = time.time()

    # Mult method
    t2 = time.time()
    for _ in range(100):
        dist2 = r_ra * r_ra + r_dec * r_dec
    t3 = time.time()

    print(f"dist2 = r_ra**2 + r_dec**2      : {t1-t0:.4f}s")
    print(f"dist2 = r_ra * r_ra + r_dec * r_dec : {t3-t2:.4f}s")

bench()
