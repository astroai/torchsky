import torch
import time

def bench_all():
    print("--- wcs/utils.py dist2 ---")
    r_ra = torch.randn(1000000)
    r_dec = torch.randn(1000000)
    t0 = time.time()
    for _ in range(100):
        dist2 = r_ra**2 + r_dec**2
    t1 = time.time()
    t2 = time.time()
    for _ in range(100):
        dist2 = r_ra * r_ra + r_dec * r_dec
    t3 = time.time()
    print(f"**2 : {t1-t0:.4f}s")
    print(f"x*x : {t3-t2:.4f}s")

    print("\n--- image_plane.py chord ---")
    dra = torch.randn(1000000)
    cos_dec = torch.randn(1000000)
    dec_deg = torch.randn(1000000)
    dec0 = torch.randn(1000000)
    t0 = time.time()
    for _ in range(100):
        chord1 = torch.sqrt((dra * cos_dec) ** 2 + (dec_deg - dec0) ** 2)
    t1 = time.time()
    t2 = time.time()
    for _ in range(100):
        dra_cos = dra * cos_dec
        ddec = dec_deg - dec0
        chord2 = torch.sqrt(dra_cos * dra_cos + ddec * ddec)
    t3 = time.time()
    print(f"pow  : {t1-t0:.4f}s")
    print(f"mult : {t3-t2:.4f}s")

    print("\n--- field/core.py radial2 ---")
    diff = torch.randn(10000, 100)
    scales = torch.randn(100)
    t0 = time.time()
    for _ in range(100):
        radial2_1 = torch.sum(torch.square(diff / scales), dim=-1)
    t1 = time.time()
    t2 = time.time()
    for _ in range(100):
        scaled_diff = diff / scales
        radial2_2 = torch.sum(scaled_diff * scaled_diff, dim=-1)
    t3 = time.time()
    print(f"torch.square : {t1-t0:.4f}s")
    print(f"x * x        : {t3-t2:.4f}s")

bench_all()
