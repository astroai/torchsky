import torch
import time

def bench():
    N = 1000000
    dra = torch.randn(N)
    cos_dec = torch.randn(N)
    dec_deg = torch.randn(N)
    dec0 = torch.randn(N)

    # Warmup
    for _ in range(10):
        chord1 = torch.sqrt((dra * cos_dec) ** 2 + (dec_deg - dec0) ** 2)

        dra_cos = dra * cos_dec
        ddec = dec_deg - dec0
        chord2 = torch.sqrt(dra_cos * dra_cos + ddec * ddec)

    # Pow method
    t0 = time.time()
    for _ in range(100):
        chord1 = torch.sqrt((dra * cos_dec) ** 2 + (dec_deg - dec0) ** 2)
    t1 = time.time()

    # Mult method
    t2 = time.time()
    for _ in range(100):
        dra_cos = dra * cos_dec
        ddec = dec_deg - dec0
        chord2 = torch.sqrt(dra_cos * dra_cos + ddec * ddec)
    t3 = time.time()

    print(f"pow  : {t1-t0:.4f}s")
    print(f"mult : {t3-t2:.4f}s")

bench()
